"""CUDA-graph lockstep decode for pinned-EMIT training rollouts.

The lockstep rollout is a sequential loop of ``prompt + budget`` decode
steps. At the production shape (64 prompts x 16 samples, 768 generated
slots) a step is memory-bound on the device, not launch-bound: a 1024-row
tick streams each KDA layer's fp32 state (201 MB) in and out and each
attention layer's live KV range. Removing the host from the step alone
(the arena over the previous eager loop) measured only ~8% faster; the
tick's cost is set by the two Triton kernels it runs --
``kda_decode_kernel`` (one read and one write of the state) and
``decode_attention`` (only each row's live keys).

``PinnedDecodeArena`` removes the host from the step. One decode tick --
record the pending token, embed it (plus the carried belief under
``hidden_carry``, or a read of the row's memory under the memory carry), step
every layer, write the new belief into the memory, sample the next token --
is one compiled function, captured as one CUDA graph per live-row bucket.
Everything a tick reads or writes lives in arena buffers, and an optimizer
step, which updates parameters in place, is visible to the next replay. The
large buffers -- decode caches and carry record -- and the graphs over them
are held only for the duration of a rollout (``workspace``), so they never
add to the update's peak memory; the compiled tick outlives them, and each
rollout re-captures its buckets on fresh buffers.

- Rows. Survivors are compacted to the FRONT of the arena at the periodic
  host check, and the tick runs on the smallest row bucket that holds them.
  Surplus rows in a bucket are ended rows: they step on token 0 like every
  finished row of the eager loop and record nothing.
- Keys. Attention layers attend each row's live range -- from its
  left-pad start through the write head -- through
  ``ranged_decode_attention``, which on CUDA reads only those cache slots.
  Shapes never depend on the position, so no graph is keyed on a width.
- Sampling. Tokens are drawn by ``counter_gumbel_tokens`` keyed on each
  trajectory's seed and its own (unpadded) stream slot, which makes the draw
  independent of row order, left-pad width, bucket padding and compaction -- the property that lets a static-shape
  graph reproduce ``rollout_continuations(..., token_seeds=...)``. Training
  arenas race over the whole policy distribution; an evaluation arena may
  restrict the race to the ``top_p`` nucleus, a compile-time constant of its
  tick.
- Budget. The emitted-token budget is a device scalar read by the tick, so
  one arena serves panels with different budgets up to its KV width.
- Write noise. A ``write_noise`` arena explores a memory carry's writes: a
  per-row signed scale, set per rollout, multiplies noise keyed on a per-row
  noise seed and the unpadded slot (``memory_carry.write_noise``), so one
  capture serves noiseless, positive and antithetic rollouts.
- Foreign memory reads. A ``memory_read_ablation`` of ``foreign_state``
  (or ``foreign_sample``) evaluation arena makes every row read another
  prompt's (or another sample's) memory state at the same tick (``memory_eval_ablation``): a side buffer holds each
  original row's latest state, indexed by original row, so the donor map
  survives compaction, and finished rows' entries stay frozen.

Contract: given the same prompts, seeds and parameters, a rollout here is the
pinned-EMIT ``rollout_continuations`` rollout with ``token_seeds``; the only
differences are floating-point reduction order in the compiled step (the same
class of difference as the existing compiled decode step). Stochastic latent
thinking (``reasoning_mode latent``) keeps the eager loop.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence

import torch
from torch import Tensor, nn

from postraining.latent_rollout import (
    PAD_SLOT,
    RECURRENT_CACHE_ARITY,
    TOKEN_SLOT,
    LatentRolloutBatch,
    compact_stream_to_device,
    counter_gumbel_tokens,
    validated_write_noise,
)
from postraining.latent_thought import EMIT, LatentThoughtModel
from postraining.memory_eval_ablation import (
    FOREIGN_READ_ABLATIONS,
    foreign_donor_rows,
    foreign_read_state,
    record_foreign_state,
)
from pretraining.nanogpt_mini import nanogpt_mini_kda_model, nanogpt_mini_model

# Live-row buckets. Above ~256 rows a step is GPU-bound and padding costs
# proportional compute, so the grid is fine there; below it the step is
# latency-bound and coarser buckets cost nothing but captures.
DEFAULT_ROW_BUCKETS = (
    1024, 896, 768, 640, 512, 448, 384, 320, 256, 192, 128, 96, 64, 48, 32,
    16, 8,
)
# Linear layers whose forward is exactly ``F.linear(x, weight.type_as(x),
# bias.type_as(x))``. Under autocast F.linear rounds weight and bias to the
# autocast dtype whatever dtype they arrive in, so feeding it a copy already
# rounded to that dtype is bitwise the same computation -- without re-reading
# and re-casting every fp32 master weight on every one of ~1000 decode ticks.
_TYPE_AS_LINEARS = (
    nanogpt_mini_kda_model.Linear,
    nanogpt_mini_kda_model.BiasFreeLinear,
    nanogpt_mini_model.Linear,
)


def autocast_linear_parameter_names(wrapper: nn.Module) -> list[str]:
    """fp32 weights/biases of ``wrapper``'s ``_TYPE_AS_LINEARS`` modules."""
    names = []
    for module_name, module in wrapper.named_modules():
        if type(module) not in _TYPE_AS_LINEARS:
            continue
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter.dtype == torch.float32:
                names.append(f"{module_name}.{parameter_name}")
    return names


class _Tick(nn.Module):
    """The decode tick as a module over the wrapper, so ``functional_call``
    can substitute the arena's rounded linear weights for the masters."""

    def __init__(self, wrapper: LatentThoughtModel, **options):
        super().__init__()
        self.wrapper = wrapper
        self.options = options

    def forward(self, *args) -> None:
        _decode_tick(self.wrapper, *args, **self.options)


def _decode_tick(
    wrapper: LatentThoughtModel,
    caches: Sequence[tuple[Tensor, ...]],
    live_rows: Tensor,
    starts: Tensor,
    emitted: Tensor,
    ended: Tensor,
    pending: Tensor,
    belief: Tensor,
    seeds: Tensor,
    kind: Tensor,
    token_ids: Tensor,
    actions: Tensor,
    action_mask: Tensor,
    hiddens: Tensor,
    memory_state: Tensor,
    position: Tensor,
    budget: Tensor,
    stop_ids: Tensor,
    policy_logits: Tensor | None = None,
    write_noise_seeds: Tensor | None = None,
    write_noise_scale: Tensor | None = None,
    foreign_states: Tensor | None = None,
    foreign_donors: Tensor | None = None,
    *,
    temperature: float,
    top_p: float,
    hidden_carry: bool,
    memory_carry: bool,
    replay_storage: bool,
    autocast_dtype: torch.dtype | None,
) -> None:
    """Record slot ``position``'s pending token, step, and sample the next.

    Stream writes go to the ORIGINAL row (``live_rows``). A row that is not
    active writes its slots' reset values, which is exactly what they already
    hold: only an active row ever writes a generated slot, and a row never
    becomes active again once it stops. That is what lets the tick scatter
    unconditionally instead of reading the old value back.
    """
    slot = position.expand_as(live_rows)
    next_position = position + 1
    next_slot = next_position.expand_as(live_rows)
    active = ~ended & (emitted < budget)
    action_mask.index_put_((live_rows, slot), active.to(action_mask.dtype))
    actions.index_put_((live_rows, slot), active.to(actions.dtype) * EMIT)
    kind.index_put_(
        (live_rows, next_slot),
        torch.where(active, TOKEN_SLOT, PAD_SLOT).to(kind.dtype),
    )
    token = torch.where(active, pending, 0)
    token_ids.index_put_((live_rows, next_slot), token)
    if hidden_carry and replay_storage:
        hiddens.index_put_(
            (live_rows, next_slot),
            torch.where(active[:, None], belief, 0).to(hiddens.dtype),
        )
    emitted.add_(active.to(emitted.dtype))
    ended.logical_or_(active & torch.isin(pending, stop_ids))
    with torch.autocast(
        device_type=pending.device.type,
        dtype=autocast_dtype or torch.bfloat16,
        enabled=autocast_dtype is not None,
    ):
        if memory_carry:
            read_state = (
                memory_state
                if foreign_states is None
                else foreign_read_state(foreign_states, foreign_donors, live_rows)
            )
            step_input = wrapper.memory_input(token, belief, read_state)
        elif hidden_carry:
            step_input = wrapper.combined_input(token, belief)
        else:
            step_input = wrapper.embed_tokens(token[:, None])
        # The class method, not the attribute: the trainer patches
        # ``wrapper.step_core`` with its own compiled artifact for the eager
        # loop, which must not nest inside this one.
        new_belief, _, _, logits = LatentThoughtModel.step_core(
            wrapper, step_input, list(caches), next_position, key_starts=starts
        )
    if hidden_carry:
        if new_belief.dtype != belief.dtype:
            # The carry record must hold the live belief losslessly, as the
            # eager rollout requires; copy_ would silently cast.
            raise RuntimeError(
                f"decode belief dtype {new_belief.dtype} differs from the "
                f"carry record's {belief.dtype}"
            )
        belief.copy_(new_belief)
    if memory_carry:
        # The write, at the network output: the next tick's read sees it.
        # Its noise is keyed on the slot that computed the belief.
        noise = (
            None
            if write_noise_scale is None
            else wrapper.memory_write_noise(
                write_noise_seeds, next_position - starts, write_noise_scale
            )
        )
        memory_state.copy_(
            wrapper.memory_write(memory_state, new_belief, noise)
        )
        if foreign_states is not None:
            # After every row's read: the next tick's donors see this write.
            record_foreign_state(foreign_states, live_rows, memory_state, active)
    if policy_logits is not None:
        policy_logits.copy_(logits.float())
    pending.copy_(
        counter_gumbel_tokens(
            logits, seeds, next_position - starts, temperature, top_p
        )
    )
    position.add_(1)


class PinnedDecodeArena:
    """Static decode state plus one CUDA graph per live-row bucket.

    Owned by the trainer for the life of the run: the compiled tick and the
    small per-row buffers persist, while the caches and the graphs, which
    hold their addresses and the wrapper's parameters', are rebuilt per
    rollout. ``rows`` is the most trajectories one rollout may produce and
    ``kv_width`` the longest prompt plus generated stream it may hold.
    ``top_p`` below one makes an evaluation arena: the policy gradient needs
    samples from the untruncated policy, so training arenas keep it at one.
    ``write_noise`` makes a memory-carry arena whose rollouts take one write
    noise scale per trajectory; evaluation arenas never explore.
    ``memory_read_ablation`` ``foreign_state``/``foreign_sample`` makes a
    memory-carry evaluation arena whose rows read another prompt's/sample's
    memory (``memory_eval_ablation``).

    On CPU (tests) or with ``cuda_graphs=False`` the same tick runs as an
    ordinary call, optionally compiled, which is the reference the graph
    replays are held to.
    """

    def __init__(
        self,
        wrapper: LatentThoughtModel,
        *,
        rows: int,
        kv_width: int,
        temperature: float,
        stop_ids: Sequence[int],
        device: torch.device,
        hidden_carry: bool,
        top_p: float = 1.0,
        cache_dtype: torch.dtype = torch.bfloat16,
        autocast_dtype: torch.dtype | None = torch.bfloat16,
        row_buckets: Sequence[int] = DEFAULT_ROW_BUCKETS,
        sync_every: int = 16,
        compile: bool = True,
        cuda_graphs: bool = True,
        cache_linear_weights: bool = True,
        replay_storage: bool = True,
        record_policy_logits: bool = False,
        write_noise: bool = False,
        memory_read_ablation: str = "none",
    ):
        if rows < 1 or kv_width < 2:
            raise ValueError("arena rows and kv width must be positive")
        if temperature <= 0.0:
            raise ValueError("counter-Gumbel decoding needs temperature > 0")
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must lie in (0, 1], got {top_p}")
        if sync_every < 1:
            raise ValueError("rollout sync period must be positive")
        if hidden_carry and not wrapper.hidden_carry:
            raise ValueError(
                "hidden-carry decoding needs a hidden-carry wrapper; this "
                "one's combiner was trained for another policy"
            )
        if cuda_graphs and device.type != "cuda":
            raise ValueError("CUDA graphs need a CUDA arena")
        if write_noise and not (hidden_carry and wrapper.memory_carry):
            raise ValueError("write noise explores a memory carry's writes")
        if memory_read_ablation not in ("none", *FOREIGN_READ_ABLATIONS):
            raise ValueError(
                "a decode arena ablates the memory read only by foreign "
                f"reads {FOREIGN_READ_ABLATIONS}, not {memory_read_ablation!r}"
            )
        # The foreign-read ablation's name, or None.
        self.foreign_reads = (
            None if memory_read_ablation == "none" else memory_read_ablation
        )
        if self.foreign_reads and not (hidden_carry and wrapper.memory_carry):
            raise ValueError("foreign memory reads need a memory-carry arena")
        if self.foreign_reads and (write_noise or replay_storage):
            raise ValueError(
                "foreign memory reads are an evaluation ablation: no write "
                "noise, no replay record"
            )
        self.wrapper = wrapper
        self.rows = rows
        self.kv_width = kv_width
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.device = device
        self.hidden_carry = hidden_carry
        self.memory_carry = hidden_carry and wrapper.memory_carry
        # Evaluation never replays, so its carry arenas need no record:
        # the record is rows x kv_width x dim (4.5 GiB at 1024 x 2304 fp32),
        # plus the same again for the returned clone.
        self.replay_storage = replay_storage
        self.cache_dtype = cache_dtype
        self.autocast_dtype = autocast_dtype
        self.sync_every = sync_every
        self.cuda_graphs = cuda_graphs
        buckets = sorted({b for b in row_buckets if 0 < b < rows} | {rows})
        self.row_buckets = tuple(buckets)
        self.stop_ids = torch.tensor(
            sorted(set(stop_ids)) or [-1], dtype=torch.long, device=device
        )

        long = dict(dtype=torch.long, device=device)
        self.live_rows = torch.arange(rows, **long)
        self.starts = torch.zeros(rows, **long)
        self.emitted = torch.zeros(rows, **long)
        self.ended = torch.ones(rows, dtype=torch.bool, device=device)
        self.pending = torch.zeros(rows, **long)
        self.policy_logits = (
            torch.empty((rows, wrapper.backbone.tok_emb.num_embeddings),
                        dtype=torch.float32, device=device)
            if record_policy_logits else None
        )
        self.seeds = torch.zeros(rows, **long)
        self.write_noise_seeds = torch.zeros(rows, **long) if write_noise else None
        self.write_noise_scale = (
            torch.zeros(rows, dtype=torch.float32, device=device)
            if write_noise else None
        )
        # Foreign reads: each original row's donor (persistent, so captures
        # keep its address) and, inside the workspace, every original row's
        # latest memory state.
        self.foreign_donors = (
            torch.zeros(rows, **long) if self.foreign_reads else None
        )
        self.foreign_states: Tensor | None = None
        self.position = torch.zeros((), **long)
        self.budget = torch.zeros((), **long)
        self.kind = torch.full((rows, kv_width), PAD_SLOT, **long)
        self.token_ids = torch.zeros((rows, kv_width), **long)
        self.actions = torch.zeros((rows, kv_width), **long)
        self.action_mask = torch.zeros(
            (rows, kv_width), dtype=torch.float32, device=device
        )
        # The workspace: decode caches, carried belief and carry record, and
        # the graphs captured over them. Present only inside ``workspace``.
        # The belief dtype is fixed by the first rollout's prefill: the carry
        # record must store the live belief losslessly.
        self.belief_dtype: torch.dtype | None = None
        self.caches: list[tuple[Tensor, ...]] | None = None
        self.belief: Tensor | None = None
        self.hiddens: Tensor | None = None
        # The memory carry's per-row fp32 state (zero-width otherwise).
        self.memory_state: Tensor | None = None
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._warm = False

        # Rounded copies of the linear weights, refreshed from the masters
        # at the start of every rollout (one cast per rollout instead of one
        # per tick). Only meaningful under autocast, where the rounding is
        # what F.linear would do anyway; ``cache_linear_weights=False`` is
        # the uncached reference the bitwise tests compare against.
        self._linear_masters: list[Tensor] = []
        self._linear_copies: dict[str, Tensor] = {}
        if cache_linear_weights and autocast_dtype is not None:
            parameters = dict(wrapper.named_parameters())
            for name in autocast_linear_parameter_names(wrapper):
                master = parameters[name]
                self._linear_masters.append(master)
                self._linear_copies[f"wrapper.{name}"] = torch.empty_like(
                    master, dtype=autocast_dtype
                )
        tick_module = _Tick(
            wrapper,
            temperature=self.temperature,
            top_p=self.top_p,
            hidden_carry=hidden_carry,
            memory_carry=self.memory_carry,
            replay_storage=replay_storage,
            autocast_dtype=autocast_dtype,
        )
        linear_copies = self._linear_copies

        def tick(*args):
            torch.func.functional_call(
                tick_module, linear_copies, args, strict=False
            )

        self._tick = (
            torch.compile(tick, dynamic=True, fullgraph=True) if compile else tick
        )
        self.ticks = 0

    # ----------------------------------------------------------- buckets
    def row_bucket(self, count: int) -> int:
        for bucket in self.row_buckets:
            if bucket >= count:
                return bucket
        raise ValueError(f"{count} rows exceed the arena's {self.rows}")

    def _tick_args(self, rows: int) -> tuple:
        if (
            self.caches is None
            or self.belief is None
            or self.hiddens is None
            or self.memory_state is None
        ):
            raise RuntimeError("decode ticks run only inside the arena workspace")
        caches = [tuple(tensor[:rows] for tensor in layer) for layer in self.caches]
        return (
            caches,
            self.live_rows[:rows],
            self.starts[:rows],
            self.emitted[:rows],
            self.ended[:rows],
            self.pending[:rows],
            self.belief[:rows],
            self.seeds[:rows],
            self.kind,
            self.token_ids,
            self.actions,
            self.action_mask,
            self.hiddens,
            self.memory_state[:rows],
            self.position,
            self.budget,
            self.stop_ids,
            None if self.policy_logits is None else self.policy_logits[:rows],
            *(
                (None, None)
                if self.write_noise_scale is None
                else (self.write_noise_seeds[:rows], self.write_noise_scale[:rows])
            ),
            # Indexed by original row: never sliced to the live bucket.
            self.foreign_states,
            self.foreign_donors,
        )

    def _run(self, rows: int) -> None:
        self.ticks += 1
        if self.cuda_graphs:
            self._graphs[rows].replay()
        else:
            self._call_tick(rows)

    def _call_tick(self, rows: int) -> None:
        """One tick under a fixed ambient context.

        The compiled tick guards on global grad and autocast state; it sets
        its own autocast inside, so pinning the ambient state here keeps a
        caller's context from forcing a recompile -- which inside a stream
        capture would be an illegal operation, not merely a slow one.
        """
        with torch.no_grad(), torch.autocast(self.device.type, enabled=False):
            self._tick(*self._tick_args(rows))

    # ------------------------------------------------------------ set-up
    @contextlib.contextmanager
    def workspace(self, belief_dtype: torch.dtype) -> Iterator[None]:
        """Allocate the decode state and capture every bucket over it.

        The caches -- 1024 rows x 1024 slots is ~4.3 GB of attention KV
        plus ~1.2 GB of fp32 KDA state -- the carry record and the graphs
        that hold their addresses exist only inside this block, so the
        update between rollouts never pays for them and peak memory is the
        larger of the two phases, not their sum. The compiled tick
        persists across workspaces; re-entering one costs a capture per
        bucket (no warm-up, no compile) and a memset of the caches.
        """
        if self.belief_dtype is None:
            self.belief_dtype = belief_dtype
        elif belief_dtype != self.belief_dtype:
            raise RuntimeError(
                f"decode belief dtype {belief_dtype} changed from the arena's "
                f"{self.belief_dtype}"
            )
        if self.caches is not None:
            raise RuntimeError("the arena workspace is already open")
        try:
            caches = self.wrapper.make_generation_cache(
                self.rows, self.kv_width, self.device, dtype=self.cache_dtype
            )
            for layer in caches:
                for tensor in layer:
                    # The masked-SDPA reference (off CUDA) reads every slot
                    # with weight zero, so slots must hold finite values.
                    tensor.zero_()
            self.caches = caches
            dim = self.wrapper.backbone.tok_emb.embedding_dim
            self.belief = torch.zeros(
                (self.rows, dim), dtype=belief_dtype, device=self.device
            )
            self.hiddens = torch.zeros(
                (
                    self.rows,
                    self.kv_width,
                    dim if self.hidden_carry and self.replay_storage else 0,
                ),
                dtype=belief_dtype,
                device=self.device,
            )
            self.memory_state = (
                self.wrapper.memory_initial_state(self.rows, self.device)
                if self.memory_carry
                else torch.zeros((self.rows, 0), device=self.device)
            )
            if self.foreign_reads:
                self.foreign_states = self.wrapper.memory_initial_state(
                    self.rows, self.device
                )
            self._capture()
            yield
        finally:
            # Graphs first: they reference the buffers' memory.
            self._graphs = {}
            self.caches = self.belief = self.hiddens = None
            self.memory_state = None
            self.foreign_states = None

    def _capture(self) -> None:
        """Capture each bucket's tick as one graph, warming them the first time.

        The first capture warms every bucket's tick eagerly on a side stream
        (compilation, autotuning and library initialisation cannot run under
        capture); those ticks scribble finite garbage over state the rollout
        resets anyway. Every capture shares one memory pool. That is safe
        because a tick returns nothing -- every value that outlives a replay
        lives in an arena buffer -- so no graph's intermediates are ever
        read after another graph runs.
        """
        if not self.cuda_graphs:
            return
        if not self._warm:
            side = torch.cuda.Stream(self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for rows in self.row_buckets:
                    self.live_rows[:rows].copy_(
                        torch.arange(rows, device=self.device)
                    )
                    self.starts.zero_()
                    self.emitted.zero_()
                    self.ended.zero_()
                    self.position.fill_(self.kv_width - 2)
                    self._call_tick(rows)
            torch.cuda.current_stream(self.device).wait_stream(side)
            torch.cuda.synchronize(self.device)
            self._warm = True
        pool = torch.cuda.graph_pool_handle()
        for rows in self.row_buckets:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                self._call_tick(rows)
            self._graphs[rows] = graph

    # ----------------------------------------------------------- rollout
    def _active(self, rows: int) -> Tensor:
        return ~self.ended[:rows] & (self.emitted[:rows] < self.budget)

    def _compact(self, rows: int, keep_rows: int, live_prefix: int) -> None:
        """Move the survivors (plus ended fillers) to the arena front."""
        active = self._active(rows)
        order = torch.sort(
            (~active).to(torch.uint8), stable=True
        ).indices[:keep_rows]
        assert self.belief is not None and self.memory_state is not None
        for tensor in (
            self.live_rows,
            self.starts,
            self.emitted,
            self.ended,
            self.pending,
            self.seeds,
            self.belief,
            self.memory_state,
        ):
            tensor[:keep_rows].copy_(tensor[:rows].index_select(0, order))
        for tensor in (
            self.policy_logits, self.write_noise_seeds, self.write_noise_scale
        ):
            if tensor is not None:
                tensor[:keep_rows].copy_(tensor[:rows].index_select(0, order))
        for layer in self.caches:
            for tensor in layer:
                if len(layer) == RECURRENT_CACHE_ARITY:
                    tensor[:keep_rows].copy_(tensor[:rows].index_select(0, order))
                else:
                    tensor[:keep_rows, :, :live_prefix].copy_(
                        tensor[:rows, :, :live_prefix].index_select(0, order)
                    )

    @torch.no_grad()
    def rollout(
        self,
        prompt_ids: Tensor,
        prompt_lengths: Tensor,
        *,
        prompt_repeats: int,
        max_new_tokens: int,
        max_stream_steps: int,
        token_seeds: Tensor,
        write_noise_seeds: Tensor | None = None,
        write_noise_scale: Tensor | None = None,
        offload_device: torch.device | None = None,
    ) -> LatentRolloutBatch:
        """Roll out ``prompt_repeats`` samples of each left-padded prompt.

        ``prompt_ids`` is (prompts, width), LEFT-padded to a shared width,
        with true lengths ``prompt_lengths``; ``token_seeds`` holds one int64
        per output trajectory (prompt-major). Each row emits at most
        ``max_new_tokens``. Returns the same stream layout as
        ``rollout_continuations`` over the same arguments.
        ``write_noise_seeds``/``write_noise_scale`` (one int64 and one fp32
        per trajectory) are required exactly by a ``write_noise`` arena.

        ``offload_device`` (the host) returns the batch there as
        ``compact_stream_to_device`` would, copied straight from the arena
        and trimmed to the used stream prefix. The device-side clone it
        skips is the whole replay record (rows x width x dim fp32, 4.5 GiB
        at 1024 x 2304) on top of the arena itself.
        """
        device = self.device
        if prompt_ids.dim() != 2 or prompt_ids.size(1) < 1:
            raise ValueError("prompt_ids must be (prompts, length>=1)")
        prefix_batch, width = prompt_ids.shape
        batch = prefix_batch * prompt_repeats
        max_stream = width + max_stream_steps
        if prompt_repeats < 1:
            raise ValueError("prompt_repeats must be positive")
        if batch > self.rows:
            raise ValueError(f"{batch} trajectories exceed the arena's {self.rows}")
        if max_stream > self.kv_width:
            raise ValueError(
                f"prompt+stream {max_stream} exceeds the arena width {self.kv_width}"
            )
        if max_new_tokens < 1:
            raise ValueError("the token budget must be positive")
        if max_stream_steps < max_new_tokens:
            raise ValueError("max_stream_steps must fit the token budget")
        if token_seeds.shape != (batch,) or token_seeds.dtype != torch.int64:
            raise ValueError("token_seeds must be one int64 seed per trajectory")
        noise_given = {write_noise_seeds is not None, write_noise_scale is not None}
        if noise_given != {self.write_noise_scale is not None}:
            raise ValueError(
                "a write-noise arena takes write noise seeds and scales per "
                "trajectory, and only it does"
            )
        if write_noise_scale is not None:
            write_noise_seeds, write_noise_scale = validated_write_noise(
                write_noise_seeds, write_noise_scale, batch, device
            )
        prompt_ids = prompt_ids.to(device)
        prompt_lengths = prompt_lengths.to(device=device, dtype=torch.long)
        if prompt_lengths.shape != (prefix_batch,):
            raise ValueError("prompt_lengths must be one true length per prompt")
        if bool(((prompt_lengths < 1) | (prompt_lengths > width)).any()):
            raise ValueError("prompt_lengths must lie in [1, prompt width]")
        pad_lengths = width - prompt_lengths
        prompt_valid = (
            torch.arange(width, device=device)[None, :] >= pad_lengths[:, None]
        )

        transient = self.wrapper.make_generation_cache(
            prefix_batch, width, device, dtype=self.cache_dtype
        )
        with torch.autocast(
            device_type=device.type,
            dtype=self.autocast_dtype or torch.bfloat16,
            enabled=self.autocast_dtype is not None,
        ):
            prefilled = self.wrapper.prefill(prompt_ids, transient, prompt_valid)
        if self._linear_copies:
            # In place: the graphs capture these addresses.
            torch._foreach_copy_(
                list(self._linear_copies.values()), self._linear_masters
            )
        with self.workspace(prefilled.belief.dtype):
            # Fan the unique-prompt prefix out into the arena rows.
            for source_layer, target_layer in zip(
                prefilled.caches, self.caches, strict=True
            ):
                recurrent = len(source_layer) == RECURRENT_CACHE_ARITY
                for source, target in zip(source_layer, target_layer, strict=True):
                    grouped = target[:batch].view(
                        prefix_batch, prompt_repeats, *target.shape[1:]
                    )
                    expanded = source[:, None].expand(
                        prefix_batch, prompt_repeats, *source.shape[1:]
                    )
                    if recurrent:
                        grouped.copy_(expanded)
                    else:
                        grouped[:, :, :, :width].copy_(expanded)
            del transient

            rows = self.row_bucket(batch)
            # Filler rows step on token 0 and record nothing, but their
            # recurrent state would otherwise evolve from whatever warm-up
            # left there. Reset it so every row a tick computes starts from
            # a defined state.
            for layer in self.caches:
                if len(layer) == RECURRENT_CACHE_ARITY:
                    for tensor in layer:
                        tensor[batch:rows].zero_()
            repeat = lambda value: value.repeat_interleave(prompt_repeats, dim=0)
            self.live_rows[:rows].copy_(torch.arange(rows, device=device))
            self.starts[:rows].zero_()
            self.starts[:batch].copy_(repeat(pad_lengths))
            self.emitted[:rows].zero_()
            self.ended[:rows].fill_(True)
            self.ended[:batch].fill_(False)
            self.seeds[:rows].zero_()
            self.seeds[:batch].copy_(token_seeds.to(device))
            if self.write_noise_scale is not None:
                self.write_noise_seeds[:rows].zero_()
                self.write_noise_seeds[:batch].copy_(write_noise_seeds)
                self.write_noise_scale[:rows].zero_()
                self.write_noise_scale[:batch].copy_(write_noise_scale)
            assert self.belief is not None and self.hiddens is not None
            assert self.memory_state is not None
            self.belief[batch:rows].zero_()
            self.belief[:batch].copy_(repeat(prefilled.belief))
            self.memory_state[batch:rows].zero_()
            self.pending[:rows].zero_()
            if self.policy_logits is not None:
                self.policy_logits[:batch].copy_(repeat(prefilled.logits).float())
            self.pending[:batch].copy_(
                counter_gumbel_tokens(
                    repeat(prefilled.logits),
                    self.seeds[:batch],
                    repeat(prompt_lengths) - 1,
                    self.temperature,
                    self.top_p,
                )
            )
            valid_rows = repeat(prompt_valid)
            self.kind[:batch, :max_stream].fill_(PAD_SLOT)
            self.kind[:batch, :width].copy_(
                torch.where(valid_rows, TOKEN_SLOT, PAD_SLOT)
            )
            self.token_ids[:batch, :max_stream].zero_()
            self.token_ids[:batch, :width].copy_(repeat(prompt_ids) * valid_rows)
            self.actions[:batch, :max_stream].zero_()
            self.action_mask[:batch, :max_stream].zero_()
            self.hiddens[:batch, :max_stream].zero_()
            if self.memory_carry:
                # Every real prompt belief, written once per unique prompt.
                beliefs = prefilled.prefix_beliefs
                self.memory_state[:batch].copy_(
                    repeat(
                        self.wrapper.memory_prompt_state(beliefs, prompt_valid)
                    )
                )
                if self.replay_storage:
                    # Recorded at the slot after the one that computed it,
                    # as the tick records the beliefs it computes.
                    self.hiddens[:batch, 1:width].copy_(
                        repeat(
                            torch.where(
                                prompt_valid[:, :-1, None], beliefs[:, :-1], 0
                            )
                        )
                    )
                del beliefs
            if self.foreign_reads:
                self.foreign_states.zero_()
                self.foreign_states[:batch].copy_(self.memory_state[:batch])
                self.foreign_donors[:rows].copy_(torch.arange(rows, device=device))
                self.foreign_donors[:batch].copy_(
                    foreign_donor_rows(
                        prefix_batch, prompt_repeats, self.foreign_reads, device
                    )
                )
            self.position.fill_(width - 1)
            self.budget.fill_(max_new_tokens)
            del prefilled

            first_position = width - 1
            position = first_position
            while position < max_stream - 1:
                if (position - first_position) % self.sync_every == 0:
                    active_count = int(self._active(rows).sum())
                    if active_count == 0:
                        break
                    keep_rows = self.row_bucket(active_count)
                    if keep_rows < rows:
                        self._compact(rows, keep_rows, position + 1)
                        rows = keep_rows
                self._run(rows)
                position += 1

            stream = (slice(0, batch), slice(0, max_stream))
            # Views when offloading: the compacting copy below is then the
            # only one, and it completes before this returns.
            def own(view: Tensor) -> Tensor:
                return view if offload_device is not None else view.clone()

            kind = own(self.kind[stream])
            actions = own(self.actions[stream])
            action_mask = own(self.action_mask[stream])
            thoughts = torch.zeros((batch, max_stream, 0), device=device)
            zeros = torch.zeros_like(action_mask)
            rolled = LatentRolloutBatch(
                kind=kind,
                token_ids=own(self.token_ids[stream]),
                thoughts=thoughts,
                hiddens=(
                    own(self.hiddens[stream])
                    if self.hidden_carry and self.replay_storage
                    else torch.zeros((batch, max_stream, 0), device=device)
                ),
                actions=actions,
                action_mask=action_mask,
                stop_mask=zeros.clone(),
                emit_mask=(actions == EMIT).float() * action_mask,
                old_stop_logprobs=zeros.clone(),
                old_token_logprobs=zeros.clone(),
                old_token_log_odds=zeros.clone(),
                old_thought_logprobs=torch.zeros_like(thoughts),
                old_thought_means=torch.zeros_like(thoughts),
                old_thought_log_sigmas=torch.zeros_like(thoughts),
                old_values=zeros.clone(),
                old_action_values=zeros.clone(),
                rewards=zeros.clone(),
                reward_scalar=torch.zeros(batch, dtype=torch.float32, device=device),
                prompt_length=width,
                carry_injected=self.hidden_carry,
                memory_carry=self.memory_carry,
                # The arguments, not the arena buffers: compaction
                # reorders those, while the stream rows keep prompt order.
                write_noise_seeds=(
                    None if write_noise_seeds is None else own(write_noise_seeds)
                ),
                write_noise_scale=(
                    None if write_noise_scale is None else own(write_noise_scale)
                ),
            )
            if offload_device is None:
                return rolled
            return compact_stream_to_device(rolled, offload_device)
