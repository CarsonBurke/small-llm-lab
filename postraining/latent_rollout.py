"""Deterministic hidden-carry rollouts and their parallel replay for VAPO.

A thought is the belief (post-final-norm hidden) that produced a generated
token. During decode the model feeds each generated token back as
``embed(x) + W(h) + b`` through the combiner; prompt tokens and the input
of the first generation step carry no hidden. The only actions are tokens.

Everything PPO needs later is stored as replayable *data* (token ids, carried
hiddens), not activations: ``replay_beliefs`` reassembles the exact stream
inputs and recomputes every belief in one parallel teacher-forced forward,
which is where new log-probs and values come from — and, with the trunk
trainable at RL time, where every policy gradient enters the model. The
stored hiddens are behavior-time constants, so there is no BPTT. Rewards are
terminal and task-defined by the caller (the DAPO trainer writes exact
verifier reward plus bounded numeric distance shaping through
``assign_terminal_rewards``); ``continuation_reward`` survives only for the
``sample_latent --fineweb`` inspection tool.
"""

from __future__ import annotations

import gc
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, fields

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from postraining.core import top_p_sample
from postraining.latent_thought import (
    DecodeRangeMask,
    LatentThoughtModel,
    StepOutput,
)

TOKEN_SLOT, PAD_SLOT = 0, -1


@dataclass
class LatentRolloutBatch:
    """Row-aligned stream storage; every tensor is (batch, stream) unless noted.

    The stream starts with ``prompt_length`` teacher-forced prompt tokens.
    Positions from ``prompt_length - 1`` onward carry token actions: the token
    decided after consuming that position's input is stored at the next slot.
    ``token_ids`` holds prompt and emitted tokens at TOKEN slots; ``hiddens``
    (batch, stream, dim, fp32) holds, at each generated token's slot, the
    belief that produced it — the +1-shifted carry the combined embedding
    injects. A slot carries a hidden exactly where the previous slot took an
    action (``generated_slot_mask``), which stays exact under replay packing
    where the scalar ``prompt_length`` is only a lower bound.
    """

    kind: Tensor
    token_ids: Tensor
    hiddens: Tensor
    action_mask: Tensor
    old_token_logprobs: Tensor
    old_values: Tensor
    rewards: Tensor
    reward_scalar: Tensor  # (batch,)
    prompt_length: int
    # Set by refresh_old_statistics: old_values/old_token_logprobs have been
    # recomputed through the update-step replay path. The actor update
    # refuses unrefreshed batches; a tensor-width proxy cannot express this
    # for pinned-EMIT rollouts, whose hidden tensors are always zero-width.
    statistics_refreshed: bool = False
    # True when the rollout injected carried hiddens into its decode inputs
    # (latent mode). Zero-width ``hiddens`` are replayable only when this is
    # False (pinned token-only modes): a latent rollout that discarded its
    # carry (``replay_storage=False``) cannot be replayed, because the replay
    # would silently rebuild plain token inputs the behavior policy never saw.
    carry_injected: bool = False
    # Plans are tied to a logical packed layout, not merely its shape. Device
    # copies preserve the token; structural transforms create a fresh one so
    # an unrelated same-shape batch cannot silently reuse stale indices.
    # Structural tensors (kind/action masks) are immutable while
    # a plan is live: the token is a lifecycle identity, not a content hash,
    # and deliberately cannot observe an in-place tensor mutation.
    replay_layout_token: object = field(
        default_factory=object, repr=False, compare=False
    )

    def to(
        self, device: torch.device, non_blocking: bool = False
    ) -> "LatentRolloutBatch":
        moved = {}
        for field in fields(self):
            value = getattr(self, field.name)
            moved[field.name] = (
                value.to(device, non_blocking=non_blocking)
                if isinstance(value, Tensor)
                else value
            )
        return LatentRolloutBatch(**moved)

    @property
    def stream_length(self) -> int:
        return self.kind.size(1)


@dataclass(frozen=True)
class ReplayShardSpec:
    """Host metadata and flat-index ranges for one deterministic replay shard."""

    host_rows: tuple[int, ...]
    stream_length: int
    row_offset: int
    row_count: int
    emit_offset: int
    emit_count: int


@dataclass(frozen=True)
class ReplayShardPlan:
    """Device-ready views of one shard's rows and compact action positions."""

    host_rows: tuple[int, ...]
    stream_length: int
    rows: Tensor
    emit_index: Tensor


@dataclass(frozen=True)
class ReplayPlan:
    """Immutable replay work discovered once while the packed batch is on CPU.

    All row and action-slot indices share one flat tensor, so moving a plan to
    CUDA is one pinned asynchronous transfer rather than one transfer per
    shard. The plan is then reusable by behavior refresh, the gradient update,
    and post-update drift. Its source batch's structural tensors must remain
    immutable for that lifetime; behavior statistics may be refreshed in place.
    """

    batch_rows: int
    batch_stream_length: int
    batch_layout_token: object
    max_trajectories: int
    attention_budget: int
    bucket_multiple: int
    slot_budget: int | None
    has_action_indices: bool
    specs: tuple[ReplayShardSpec, ...]
    indices: Tensor

    def validate_for(self, batch: LatentRolloutBatch) -> None:
        if batch.kind.size(0) != self.batch_rows:
            raise ValueError("replay plan row count does not match batch")
        if batch.stream_length != self.batch_stream_length:
            raise ValueError("replay plan stream length does not match batch")
        if batch.replay_layout_token is not self.batch_layout_token:
            raise ValueError("replay plan belongs to a different packed layout")
        if self.indices.device != batch.kind.device:
            raise ValueError("replay plan and batch must be on the same device")

    def validate_settings(
        self,
        max_trajectories: int,
        attention_budget: int,
        bucket_multiple: int,
        slot_budget: int | None,
        *,
        require_action_indices: bool,
    ) -> None:
        settings = (
            max_trajectories,
            attention_budget,
            bucket_multiple,
            slot_budget,
        )
        if settings != (
            self.max_trajectories,
            self.attention_budget,
            self.bucket_multiple,
            self.slot_budget,
        ):
            raise ValueError("replay plan settings do not match replay request")
        if require_action_indices and not self.has_action_indices:
            raise ValueError("replay plan does not contain action indices")

    def to(
        self, device: torch.device, non_blocking: bool = False
    ) -> "ReplayPlan":
        device = torch.device(device)
        indices = self.indices
        if indices.device == device:
            return self
        if (
            device.type == "cuda"
            and indices.device.type == "cpu"
            and not indices.is_pinned()
        ):
            indices = indices.pin_memory()
        indices = indices.to(device, non_blocking=non_blocking)
        return ReplayPlan(
            batch_rows=self.batch_rows,
            batch_stream_length=self.batch_stream_length,
            batch_layout_token=self.batch_layout_token,
            max_trajectories=self.max_trajectories,
            attention_budget=self.attention_budget,
            bucket_multiple=self.bucket_multiple,
            slot_budget=self.slot_budget,
            has_action_indices=self.has_action_indices,
            specs=self.specs,
            indices=indices,
        )

    def shards(self):
        for spec in self.specs:
            yield ReplayShardPlan(
                host_rows=spec.host_rows,
                stream_length=spec.stream_length,
                rows=self.indices.narrow(
                    0, spec.row_offset, spec.row_count
                ),
                emit_index=self.indices.narrow(
                    0, spec.emit_offset, spec.emit_count
                ),
            )


# Recurrent (KDA) layers carry ``(conv_q, conv_k, conv_v, state)`` decode
# caches. Unlike the dense KV pair / PoPE triple, none of those tensors has a
# stream-length axis at dim 2 — the conv windows are ``[rows, D, kernel]`` and
# the delta-rule state is ``[rows, H, Dv, Dk]`` — so every prefix-window slice
# below must skip them and every "cache width" question must be answered by a
# dense layer (or not at all, when the trunk is fully recurrent).
RECURRENT_CACHE_ARITY = 4


def _dense_cache_width(caches: "list[tuple[Tensor, ...]]") -> "int | None":
    """KV width of the first length-addressed layer; None if all recurrent."""
    for layer in caches:
        if len(layer) != RECURRENT_CACHE_ARITY:
            return layer[0].size(2)
    return None


def compact_stream_to_device(
    batch: LatentRolloutBatch, device: torch.device
) -> LatentRolloutBatch:
    """Copy only the used stream prefix into owning storage on ``device``.

    Rollout storage is provisioned for the full context, while ordinary
    trajectories use only a small prefix. Slicing before a device transfer
    avoids copying the unused tail and the destination copy already owns its
    compact storage, unlike a same-device narrow view.
    """
    used = int((batch.kind != PAD_SLOT).any(0).nonzero().max()) + 1
    compact = {}
    async_cuda_to_cpu = (
        batch.kind.device.type == "cuda" and device.type == "cpu"
    )
    for field in fields(batch):
        if field.name == "replay_layout_token":
            continue
        value = getattr(batch, field.name)
        if (
            isinstance(value, Tensor)
            and value.dim() >= 2
            and value.size(1) == batch.stream_length
        ):
            value = value[:, :used]
        if isinstance(value, Tensor):
            if async_cuda_to_cpu:
                target = torch.empty_like(
                    value, device=device, pin_memory=True
                )
                target.copy_(value, non_blocking=True)
                value = target
            elif value.device == device:
                value = value.clone()
            else:
                value = value.to(device)
        compact[field.name] = value
    if async_cuda_to_cpu:
        # All pinned copies are enqueued on the current stream. One barrier
        # makes the complete compact batch host-readable without serializing
        # once per dataclass field.
        torch.cuda.current_stream(batch.kind.device).synchronize()
    return LatentRolloutBatch(**compact)


def continuation_reward(generated: str, reference: str) -> float:
    """Tier-0 reward: mean of prefix-match fraction and character F1.

    The F1 half is bag-of-characters and therefore gameable (generic common
    characters harvest recall without content match); Tier 0 is a
    learnability probe, not the final objective — read its learnability gate
    accordingly and rely on the prefix half plus later tiers for content.
    """
    if not reference:
        return 0.0
    prefix = 0
    for generated_char, reference_char in zip(generated, reference):
        if generated_char != reference_char:
            break
        prefix += 1
    prefix_score = prefix / len(reference)
    generated_counts = Counter(generated)
    overlap = sum((generated_counts & Counter(reference)).values())
    if generated:
        precision = overlap / len(generated)
        recall = overlap / len(reference)
        f1 = 2 * precision * recall / (precision + recall) if overlap else 0.0
    else:
        f1 = 0.0
    return 0.5 * prefix_score + 0.5 * f1


@torch.no_grad()
def rollout_continuations(
    wrapper: LatentThoughtModel,
    prompt_ids: Tensor,
    max_new_tokens: int,
    max_stream_steps: int,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None = None,
    stop_ids: int | Sequence[int] | None = None,
    caches: list[tuple[Tensor, ...]] | None = None,
    prompt_lengths: Tensor | None = None,
    tensor_positions: bool = False,
    replay_storage: bool = True,
    record_likelihoods: bool = True,
    compact_finished: bool = True,
    finished_batch_size: int | None = None,
    cache_dtype: torch.dtype | None = None,
    prompt_repeats: int = 1,
    pin_emit: bool = False,
    tail_caches: list[tuple[Tensor, ...]] | None = None,
    tail_step_core=None,
    sync_every: int = 16,
    compact_dead_ratio: float = 0.25,
    decode_mask: DecodeRangeMask | None = None,
    tail_decode_mask: DecodeRangeMask | None = None,
) -> LatentRolloutBatch:
    """Roll the hidden-carry token stream forward from a (batch, P) prompt.

    In latent mode every decode input past the first generation step is the
    combined embedding of the previous token and the belief that produced it;
    the first generation step consumes the last prompt token plain.
    ``max_stream_steps`` is the generated-slot budget per row and must cover
    ``max_new_tokens`` (there are no thought slots, so the two are normally
    equal). With ``stop_ids`` set, a row finishes the moment it emits any of
    those tokens (the stop token itself is recorded), matching the
    stop-truncated decode the verifier scores.  BOS belongs in ``stop_ids``
    alongside EOS: pretraining shards never append EOS, so an emitted BOS
    ("next document starts here") is the model's only learned end-of-document
    signal.

    ``pin_emit`` disables the hidden carry entirely: the rollout is a plain
    token policy (cot/none reasoning modes) whose next input is the bare
    token embedding, and hidden storage keeps a zero-width final dimension
    exactly like ``replay_storage=False``.

    ``caches`` switches to the fixed-shape step path: the caller passes
    preallocated caches (``make_static_generation_cache``, at least
    ``prompt + max_stream_steps`` long), positions become 0-dim tensors and
    every step attends over the full cache under a ``key_mask`` — constant
    shapes, so a compiled ``wrapper.step`` replays one CUDA graph.  Reusing
    one cache set across calls of identical shape is what keeps the graph
    from re-recording.

    ``prompt_lengths`` (per-row true lengths) batches rollouts over prompts
    of unequal length: ``prompt_ids`` arrives LEFT-padded to a shared width
    and each row's padded prefix slots are masked out of attention.  The
    positional shift this introduces is exact — PoPE and RoPE scores depend
    only on position differences — and ``split_rollout_groups`` undoes the
    padding afterwards. It is mutually exclusive with preallocated static
    ``caches``; the ordinary narrow-cache step may be eager or dynamically
    compiled.

    Evaluation sets ``replay_storage=False`` because it never replays PPO
    actions. Its full-stream hidden tensor then has a zero-width final
    dimension. This avoids multiple GiB of dead storage at large batches
    while the live belief is still carried into the next step.

    ``prompt_repeats`` declares how many output trajectories each UNIQUE input
    prompt owns. Its deterministic prefix is evaluated once, then its final
    policy output, prompt storage, and populated KV prefix are expanded across
    members before stochastic actions consume RNG. This structural interface
    makes accidentally supplying unequal repeated prompts impossible.

    ``record_likelihoods=False`` skips rollout-time token likelihoods
    when the caller will immediately recompute them through parallel replay.
    ``compact_finished`` removes completed rows and their KV cache entries at
    the existing 16-position synchronization points once at least 25% of the
    current rows have finished. With ``finished_batch_size``, compaction waits
    until the live count fits that one fixed tail size and retains ended filler
    rows to fill it. Compiled evaluation uses this to expose exactly one
    bounded tail specialization instead of arbitrary survivor shapes. Original
    row indices remain attached to all trajectory records, so compaction changes
    compute and RNG consumption, not the sampled policy distribution or output
    attribution.

    ``tail_caches`` (with ``finished_batch_size``) switches the compacted tail
    onto caller-owned static caches (``make_static_generation_cache`` of
    exactly ``finished_batch_size`` rows, at least ``max_stream`` long):
    at the tail compaction the survivors' live cache prefix is copied into
    them and every remaining step attends the full static width under a
    fixed-shape per-row ``key_mask`` that combines causality with each row's
    left-pad validity — constant shapes, so ``tail_step_core`` (a
    CUDA-graph-compiled ``step_core``) replays one graph per step. Reusing one
    cache set across rollouts of identical shape keeps the graph from
    re-recording; stale values it carries stay masked, exactly like the zero
    fill at allocation. Without ``tail_step_core`` the tail runs through the
    ordinary ``wrapper.step`` (the eager 2-D-mask path), which is what CPU
    tests exercise.

    ``decode_mask``/``tail_decode_mask`` (``DecodeRangeMask``) replace the
    boolean row mask of the main loop and of the static tail with a flex
    decoding block table. The tail's live keys are one
    contiguous range per row — left padding is a prefix, the write head is a
    suffix bound — so no information is lost, and the step leaves the
    memory-efficient SDPA kernel that a boolean ``attn_mask`` forces it onto
    (32% of pool device time in the v25 profile). Inductor lowers flex
    decoding for fully static shapes only. The tail is static already
    (``finished_batch_size`` rows on fixed-width caches); the MAIN loop, which
    is 98.5% of the decode iterations at ``--rollout-groups 32``, becomes
    static by compacting to a ``decode_mask.row_bucket`` multiple instead of
    to the exact survivor count and giving the surplus rows an EMPTY key
    range, which reads no KV and returns exactly zero.

    The MAIN loop gets there a different way, because its survivor count is
    data-dependent and flex decoding only lowers for static shapes. With
    ``decode_mask`` set, compaction rounds the survivor count UP to a multiple
    of ``decode_mask.row_bucket`` instead of down to the exact count, and the
    surplus rows — along with every row that has already stopped — are given an
    EMPTY range. An empty range reads no KV and returns exactly zero, so the
    padding is free in TIME: at the production shape a 256-row bucket holding
    144 live rows measured 0.6% over an exact 144-row batch, against 59% for
    the SDPA step it replaces. It is not free in MEMORY — the rounded count is
    also the width the KV cache is reallocated at — which is why the grid is
    linear rather than powers of two; see ``DecodeRangeMask``. A bounded
    number of buckets then covers the whole rollout, so the step specializes a
    bounded number of times.

    Caller-owned ``caches`` are the arena of the fully static main loop: they
    fix the row count and the KV width for every rollout that shares them, so
    compaction is off and one recording serves the run. A chunk with fewer
    prompts than the arena holds is padded up to it with filler prompts whose
    rows are ended before the first step -- empty key range, nothing recorded
    -- and those rows are dropped from the returned batch. The arena's row
    count must therefore be a whole number of ``prompt_repeats`` groups.

    Both are CALLER-owned for the same reason the tail caches are: each
    carries buffers whose addresses a cudagraph replay captures.
    ``decode_mask.kv_width`` also pins the main cache width, which keeps that
    width from varying with the padded prompt width and respecializing the
    step; it must cover ``prompt + stream`` for every call sharing the
    artifact. Both paths attend the whole cache under the mask, so caches are
    zero-filled at allocation: an unwritten slot inside a live block would
    otherwise be uninitialized memory.
    """
    if prompt_ids.dim() != 2 or prompt_ids.size(1) < 1:
        raise ValueError("prompt_ids must be (batch, length>=1)")
    if max_stream_steps < max_new_tokens:
        raise ValueError("max_stream_steps must fit max_new_tokens")
    device = prompt_ids.device
    prefix_batch, prompt_length = prompt_ids.shape
    if prompt_repeats < 1:
        raise ValueError("prompt_repeats must be positive")
    batch = prefix_batch * prompt_repeats

    if finished_batch_size is not None and finished_batch_size < 1:
        raise ValueError("finished_batch_size must be positive")
    model_dim = wrapper.backbone.tok_emb.embedding_dim
    max_stream = prompt_length + max_stream_steps
    stop_tensor = None
    if stop_ids is not None:
        ids = (stop_ids,) if isinstance(stop_ids, int) else tuple(stop_ids)
        if ids:
            stop_tensor = torch.tensor(ids, dtype=torch.long, device=device)
    if prompt_lengths is not None:
        if caches is not None and decode_mask is None:
            # The boolean preallocated path shares ONE causal mask row across
            # the batch, which cannot express a per-row left pad. A block
            # table can: every row carries its own range start, so the flex
            # arena handles left padding exactly as the allocating path does.
            raise ValueError(
                "prompt_lengths (left-padded batching) needs either an "
                "allocated cache or a decode_mask"
            )
        if prompt_lengths.shape != (prefix_batch,):
            raise ValueError("prompt_lengths must be one true length per prompt")
        prompt_lengths = prompt_lengths.to(device=device, dtype=torch.long)
        if bool((prompt_lengths < 1).any()) or bool(
            (prompt_lengths > prompt_length).any()
        ):
            raise ValueError("prompt_lengths must lie in [1, prompt_ids width]")

    # A caller-owned arena fixes the row count for the whole rollout --
    # compaction is disabled under it, see ``should_compact`` -- and the point
    # of owning it is that ONE recording serves every chunk. A chunk carrying
    # fewer prompts than the arena was sized for therefore cannot simply run
    # narrower: that records a second graph at a second shape, and short
    # chunks are routine (the value warmup takes prompts_per_minibatch, the
    # final pool takes whatever prompts remain, --consume-all-prompts takes an
    # arbitrary count). The prompt batch is padded up to the arena instead.
    # Filler rows are ended from step zero, so they get an empty key range
    # every step -- no KV read, exactly zero out -- record nothing, and are
    # dropped from the returned trajectories.
    filler_rows = 0
    if caches is not None:
        arena_rows = caches[0][0].size(0)
        if arena_rows % prompt_repeats:
            raise ValueError(
                f"preallocated caches hold {arena_rows} rows, not a whole "
                f"number of {prompt_repeats}-sample groups"
            )
        if arena_rows < batch:
            raise ValueError(
                f"preallocated caches hold {arena_rows} rows, fewer than the "
                f"{batch} this rollout produces"
            )
        filler_rows = arena_rows - batch
        if filler_rows:
            filler_prompts = filler_rows // prompt_repeats
            prompt_ids = torch.cat(
                (
                    prompt_ids,
                    prompt_ids[-1:].expand(filler_prompts, prompt_length),
                )
            )
            if prompt_lengths is not None:
                prompt_lengths = torch.cat(
                    (prompt_lengths, prompt_lengths[-1:].expand(filler_prompts))
                )
            prefix_batch += filler_prompts
            batch = arena_rows

    pad_lengths = None
    valid_slots = None
    if prompt_lengths is not None:
        pad_lengths = (
            prompt_length - prompt_lengths
        ).repeat_interleave(prompt_repeats)
        # Slot k is a real (attendable) slot for row b iff k >= pad_lengths[b].
        valid_slots = (
            torch.arange(max_stream, device=device)[None, :] >= pad_lengths[:, None]
        )
    if decode_mask is not None:
        if not tensor_positions:
            raise ValueError("decode_mask requires tensor_positions")
        preallocated_width = (
            _dense_cache_width(caches) if caches is not None else None
        )
        if preallocated_width is not None and (
            decode_mask.kv_width != preallocated_width
        ):
            # Preallocated caches are the CUDA-graph path: the block table
            # spans the whole allocated width, so the two must agree exactly
            # rather than the mask merely covering the stream. When the caller
            # owns the caches it owns the width, so this is its error to fix.
            raise ValueError(
                f"decode_mask width {decode_mask.kv_width} does not match the "
                f"preallocated cache width {preallocated_width}"
            )
        if decode_mask.kv_width < max_stream:
            raise ValueError(
                f"decode_mask width {decode_mask.kv_width} does not cover "
                f"prompt+stream {max_stream}"
            )
        if decode_mask.kv_starts.numel() < batch:
            raise ValueError(
                f"decode_mask holds {decode_mask.kv_starts.numel()} rows, "
                f"fewer than the {batch} rolled out"
            )
        if decode_mask.kv_starts.device != device:
            raise ValueError(
                f"decode_mask is on {decode_mask.kv_starts.device}, not the "
                f"rollout device {device}"
            )

    def new_caches(rows: int, length: int) -> list[tuple[Tensor, ...]]:
        allocated = wrapper.make_generation_cache(
            rows, length, device, dtype=cache_dtype
        )
        if decode_mask is not None and length > prompt_length:
            # The block table spans the whole cache width, and its live blocks
            # are read whole; make_generation_cache hands back uninitialized
            # memory, which would put garbage inside the block holding the
            # write head. A cache exactly ``prompt_length`` wide is the
            # transient the fan-out prefills into, and the prefill writes every
            # one of its columns, so there is nothing there to hide -- and it
            # is allocated once per chunk, so the memset is not free.
            for layer in allocated:
                for tensor in layer:
                    tensor.zero_()
        return allocated

    preallocated_caches = caches is not None
    # Only ever set when the caller owns the decode arena AND the prompt fans
    # out: the prefill still runs once per UNIQUE prompt in a narrow transient
    # cache, and its result is expanded into the caller's arena below. Without
    # this the fan-out would prefill every sample separately.
    expansion_target = None
    if preallocated_caches and prompt_repeats > 1:
        preallocated_width = _dense_cache_width(caches)
        if preallocated_width is not None and preallocated_width < max_stream:
            raise ValueError(
                f"preallocated caches are {preallocated_width} wide, "
                f"shorter than prompt+stream {max_stream}"
            )
        expansion_target = caches
        caches = None
    if caches is None:
        cache_batch = prefix_batch if prompt_repeats > 1 else batch
        cache_length = (
            prompt_length
            if prompt_repeats > 1
            else max_stream if decode_mask is None else decode_mask.kv_width
        )
        caches = new_caches(cache_batch, cache_length)
        position_index = (
            torch.zeros((), dtype=torch.long, device=device)
            if tensor_positions
            else None
        )
        key_masks = None
    else:
        cache_length = _dense_cache_width(caches)
        if cache_length is None:
            # A fully recurrent trunk has no KV width to bound the stream.
            cache_length = max_stream
        elif cache_length < max_stream:
            raise ValueError(
                f"preallocated caches ({cache_length} wide) do not fit "
                f"stream {max_stream}"
            )
        position_index = torch.zeros((), dtype=torch.long, device=device)
        # Row p is the step-p key mask; indexing it is a view, so the hot
        # loop adds no mask-construction kernels. The flex path carries the
        # same information in its block table, so it needs none of this --
        # and a cache_length-squared boolean is not free at the widths the
        # graph path runs at.
        key_masks = (
            None
            if decode_mask is not None
            else torch.ones(
                (cache_length, cache_length), dtype=torch.bool, device=device
            ).tril_()
        )

    tail_mask: Tensor | None = None
    tail_length = 0
    if tail_caches is not None:
        if preallocated_caches:
            raise ValueError(
                "tail_caches is the mid-rollout static switch; preallocated "
                "caches are already static for the whole rollout"
            )
        if finished_batch_size is None or not compact_finished:
            raise ValueError(
                "tail_caches requires compact_finished and "
                "finished_batch_size: the static switch happens at the "
                "fixed-size tail compaction"
            )
        if position_index is None:
            raise ValueError("tail_caches requires tensor_positions")
        tail_width = _dense_cache_width(tail_caches)
        tail_length = tail_width if tail_width is not None else max_stream
        if (
            tail_caches[0][0].size(0) != finished_batch_size
            or tail_length < max_stream
        ):
            raise ValueError(
                f"tail caches ({tail_caches[0][0].size(0)} rows x "
                f"{tail_length} wide) do not fit "
                f"tail batch {finished_batch_size} x stream {max_stream}"
            )
        if tail_caches[0][0].dtype != caches[0][0].dtype:
            raise ValueError(
                f"tail cache dtype {tail_caches[0][0].dtype} must match the "
                f"rollout cache dtype {caches[0][0].dtype}"
            )
    if tail_decode_mask is not None:
        if tail_caches is None:
            raise ValueError("tail_decode_mask masks the static tail caches")
        if tail_decode_mask.kv_width != tail_length:
            raise ValueError(
                f"tail_decode_mask width {tail_decode_mask.kv_width} must "
                f"match the tail cache width {tail_length}"
            )
        if tail_decode_mask.kv_starts.device != device:
            # A cross-device copy into kv_starts is legal, so without this the
            # run dies inside flex_attention with a kernel-level message that
            # points away from the caller.
            raise ValueError(
                f"tail_decode_mask is on {tail_decode_mask.kv_starts.device}, "
                f"not the rollout device {device}"
            )
        if tail_decode_mask.kv_starts.numel() < finished_batch_size:
            raise ValueError(
                f"tail_decode_mask holds "
                f"{tail_decode_mask.kv_starts.numel()} rows, fewer than the "
                f"{finished_batch_size} in the tail"
            )

    # Row b's live keys are always the contiguous range
    # ``[decode_starts[b], position]``, which is what lets the block tables
    # carry the same information as the boolean masks they replace. It tracks
    # ``pad_lengths`` through compaction, and is the tail's starts after the
    # static switch.
    decode_starts: Tensor | None = None
    if decode_mask is not None or tail_decode_mask is not None:
        decode_starts = (
            pad_lengths.clone()
            if pad_lengths is not None
            else torch.zeros(batch, dtype=torch.long, device=device)
        )
    tail_starts: Tensor | None = None

    def step_position(
        position: int,
    ) -> tuple[int | Tensor, Tensor | None, BlockMask | None]:
        if tail_starts is not None:
            position_index.fill_(position)
            return (
                position_index,
                None,
                tail_decode_mask.build(tail_starts, position + 1),
            )
        if tail_mask is not None:
            # Fixed-shape tail: the full-width row mask grows by one column
            # per step in place, so the step's shapes never change and the
            # mask write is one tiny kernel. Generated slots are always
            # attendable; only each row's left-pad prefix stays False.
            position_index.fill_(position)
            tail_mask[:, position] = True
            return position_index, tail_mask, None
        if position_index is not None:
            position_index.fill_(position)
            step_position_value: int | Tensor = position_index
            if decode_mask is not None:
                # Finished rows and bucket fillers get an empty range, which
                # is what makes the padding to a static row count free. Rows
                # still inside their own left pad fall out for nothing: their
                # start already exceeds ``position``, so the range is empty
                # there too and the kernel returns zero rather than the NaN a
                # fully masked SDPA row would have produced.
                #
                # ``active``, not ``~ended``: ``ended`` covers only stop-token
                # termination, so a row that finished by exhausting
                # max_new_tokens would keep a full, GROWING key range for
                # every remaining step. Without a stop token it never gets set
                # at all. Nothing an inactive row produces is recorded --
                # ``record = active`` gates every stream write below -- so
                # zeroing its attention is as safe as it is for a stopped row.
                # One iteration stale (``active`` is recomputed at the top of
                # the next pass) and stale in the conservative direction.
                return (
                    step_position_value,
                    None,
                    decode_mask.build(decode_starts, position + 1, active),
                )
            if key_masks is not None:
                return step_position_value, key_masks[position], None
        else:
            step_position_value = position
        if valid_slots is None:
            return step_position_value, None, None
        mask = valid_slots[:, : position + 1]
        if position < prompt_length:
            # Rows whose query at ``position`` is still inside their own pad
            # region attend everything instead: a fully-masked SDPA row is
            # NaN, and that NaN would enter deeper layers' K/V at this slot
            # and later poison REAL queries (a masked score is -inf, and
            # -inf + NaN is NaN inside the softmax).  The finite garbage
            # output is discarded, and the slot itself stays masked for all
            # real queries via ``valid_slots``.
            mask = mask | (pad_lengths[:, None] > position)
        return step_position_value, mask, None

    kind = torch.full((batch, max_stream), PAD_SLOT, dtype=torch.long, device=device)
    token_ids = torch.zeros((batch, max_stream), dtype=torch.long, device=device)
    stored_hidden_dim = model_dim if replay_storage and not pin_emit else 0
    hiddens = torch.zeros(
        (batch, max_stream, stored_hidden_dim),
        dtype=torch.float32,
        device=device,
    )
    action_mask = torch.zeros((batch, max_stream), dtype=torch.float32, device=device)
    old_token_logprobs = torch.zeros_like(action_mask)
    # Stays zero through the rollout; refresh_old_statistics fills it from
    # the separate critic before anything consumes it.
    old_values = torch.zeros_like(action_mask)

    if valid_slots is None:
        kind[:, :prompt_length] = TOKEN_SLOT
        token_ids[:, :prompt_length] = prompt_ids.repeat_interleave(
            prompt_repeats, dim=0
        )
    else:
        prompt_valid = valid_slots[:, :prompt_length]
        kind[:, :prompt_length] = torch.where(prompt_valid, TOKEN_SLOT, PAD_SLOT)
        token_ids[:, :prompt_length] = (
            prompt_ids.repeat_interleave(prompt_repeats, dim=0)
            * prompt_valid
        )

    prefix_prompt_ids = prompt_ids
    prefix_valid_slots = (
        valid_slots[::prompt_repeats] if valid_slots is not None else None
    )
    prefix_key_valid = (
        prefix_valid_slots[:, :prompt_length]
        if prefix_valid_slots is not None
        else None
    )
    output = wrapper.prefill(prefix_prompt_ids, caches, prefix_key_valid)
    caches = output.caches

    if prompt_repeats > 1:
        expanded_caches = expansion_target
        if expanded_caches is None:
            expanded_caches = new_caches(
                batch,
                max_stream if decode_mask is None else decode_mask.kv_width,
            )
        for source_layer, target_layer in zip(
            caches, expanded_caches, strict=True
        ):
            recurrent_layer = len(source_layer) == RECURRENT_CACHE_ARITY
            for source, target in zip(source_layer, target_layer, strict=True):
                grouped_target = target.view(
                    prefix_batch,
                    prompt_repeats,
                    *target.shape[1:],
                )
                expanded = source[:, None].expand(
                    prefix_batch,
                    prompt_repeats,
                    *source.shape[1:],
                )
                if recurrent_layer:
                    # Conv windows and delta-rule states have no length axis;
                    # the prefill cache and the arena cache are shape-equal,
                    # so the fan-out copies each tensor whole.
                    grouped_target.copy_(expanded)
                else:
                    grouped_target[:, :, :, :prompt_length].copy_(expanded)
        caches = expanded_caches

        def expand_rows(value: Tensor) -> Tensor:
            return value.repeat_interleave(prompt_repeats, dim=0)

        output = output.__class__(
            belief=expand_rows(output.belief),
            input_latent=expand_rows(output.input_latent),
            logits=expand_rows(output.logits),
            caches=caches,
        )

    emitted = torch.zeros(batch, dtype=torch.long, device=device)
    ended = torch.zeros(batch, dtype=torch.bool, device=device)
    if filler_rows:
        # Ended before the first step is what makes the arena padding free:
        # ``active`` is False for these rows forever, so their key range is
        # empty (no KV read, exactly zero out) and ``record`` never writes a
        # stream slot for them. They still occupy the batch dimension, which
        # is the entire point.
        ended[batch - filler_rows :] = True
    live_rows = torch.arange(batch, device=device)
    position = prompt_length - 1
    # ``int(active.sum())`` is a device-to-host sync that serializes this
    # launch-bound loop (the CPU cannot run ahead of the GPU), so the finish
    # and compaction check runs only every SYNC_EVERY steps.  The extra
    # <=SYNC_EVERY-1 steps after all rows finish are no-ops: ``record`` is
    # all-False, and finished rows already keep stepping on token 0 by
    # design.  This is the ONLY host sync in the loop -- the stream writes
    # below deliberately avoid boolean row indexing, which would reintroduce
    # one per masked write per step.
    #
    # ``sync_every`` also sets how often compaction can fire, and that is now
    # the dominant term.  A chunk steps at its longest row's length, so with a
    # median 60 steps per chunk a 16-step period leaves only three or four
    # chances to narrow the batch, and the dead rows keep paying full width
    # until the next one.  The check costs a single scalar D2H against a step
    # that costs milliseconds, so trading more checks for earlier compaction
    # is close to free -- see ``--rollout-sync-every``.
    if sync_every < 1:
        raise ValueError("rollout sync period must be positive")
    if not 0.0 < compact_dead_ratio <= 1.0:
        raise ValueError("compaction dead-row ratio must be in (0, 1]")
    SYNC_EVERY = sync_every
    first_position = position
    while position < max_stream - 1:
        active = ~ended & (emitted < max_new_tokens)
        if (position - first_position) % SYNC_EVERY == 0:
            active_count = int(active.sum())
            if active_count == 0:
                break
            current_count = active.numel()
            compacted_count = active_count
            if decode_mask is not None:
                # Flex decoding lowers for static shapes only, so the survivor
                # count is rounded UP to a multiple of ROW_BUCKET. The surplus
                # rows cost no attention work (empty range), which is what
                # makes this affordable; compacting to the exact count instead
                # would respecialize the step on every distinct count. They do
                # still cost cache memory, which is why the grid is linear --
                # see DecodeRangeMask.DEFAULT_ROW_BUCKET.
                bucket = decode_mask.row_bucket
                compacted_count = min(
                    current_count, -(-active_count // bucket) * bucket
                )
            snap_to_tail = (
                finished_batch_size is not None
                and active_count <= finished_batch_size
            )
            if snap_to_tail:
                compacted_count = min(current_count, finished_batch_size)
            # Above the tail width, compact progressively under the same
            # >=25%-dead hysteresis as the tail-free path: a few long
            # survivors must not keep stepping hundreds of finished rows at
            # full width until the final tail snap. Intermediate widths never
            # equal the static tail size, so they run on the dynamic-shape
            # step core and cannot engage the tail graph early.
            should_compact = (
                compact_finished
                and not preallocated_caches
                and compacted_count < current_count
                and (
                    snap_to_tail
                    or current_count - active_count
                    >= max(1, int(current_count * compact_dead_ratio))
                )
            )
            if should_compact:
                keep = active.nonzero().squeeze(-1)
                if compacted_count > active_count:
                    fillers = (~active).nonzero().squeeze(-1)[
                        : compacted_count - active_count
                    ]
                    keep = torch.cat((keep, fillers))
                live_rows = live_rows.index_select(0, keep)
                if decode_starts is not None:
                    decode_starts = decode_starts.index_select(0, keep)
                emitted = emitted.index_select(0, keep)
                ended = ended.index_select(0, keep)
                if valid_slots is not None:
                    valid_slots = valid_slots.index_select(0, keep)
                    pad_lengths = pad_lengths.index_select(0, keep)
                live_prefix = position + 1
                if (
                    tail_caches is not None
                    and compacted_count == tail_caches[0][0].size(0)
                ):
                    # Static-tail switch: land the survivors' live prefix in
                    # the caller-owned graph-static caches. Slots at or after
                    # ``live_prefix`` keep stale-but-finite values from
                    # earlier rollouts; the fixed-width row mask hides them,
                    # exactly like the zero fill at allocation.
                    for layer, cache in enumerate(caches):
                        static_layer = tail_caches[layer]
                        recurrent_layer = (
                            len(cache) == RECURRENT_CACHE_ARITY
                        )
                        for tensor, target in zip(
                            cache, static_layer, strict=True
                        ):
                            if recurrent_layer:
                                # No length axis: the survivors' whole conv
                                # window / state moves.
                                target.copy_(tensor.index_select(0, keep))
                            else:
                                target[:, :, :live_prefix].copy_(
                                    tensor[:, :, :live_prefix].index_select(
                                        0, keep
                                    )
                                )
                        # Replace one layer at a time so the dynamic caches
                        # free as the static ones fill.
                        caches[layer] = static_layer
                    if tail_decode_mask is not None:
                        # Rows were compacted above, so this is the survivors'
                        # first live key: their left-pad width, or zero when
                        # the rollout was not left-padded at all.
                        tail_starts = decode_starts
                    else:
                        tail_mask = torch.zeros(
                            (compacted_count, tail_length),
                            dtype=torch.bool,
                            device=device,
                        )
                        if valid_slots is not None:
                            tail_mask[:, :live_prefix] = valid_slots[
                                :, :live_prefix
                            ]
                        else:
                            tail_mask[:, :live_prefix] = True
                elif snap_to_tail or finished_batch_size is None:
                    for layer, cache in enumerate(caches):
                        compacted = []
                        recurrent_layer = (
                            len(cache) == RECURRENT_CACHE_ARITY
                        )
                        for tensor in cache:
                            if recurrent_layer:
                                # index_select materializes owning storage of
                                # exactly the survivor rows; there is no
                                # write-head garbage to zero in a cache with
                                # no length axis.
                                compacted.append(
                                    tensor.index_select(0, keep)
                                )
                                continue
                            target = torch.empty(
                                (compacted_count, *tensor.shape[1:]),
                                dtype=tensor.dtype,
                                device=tensor.device,
                            )
                            target[:, :, :live_prefix].copy_(
                                tensor[:, :, :live_prefix].index_select(
                                    0, keep
                                )
                            )
                            if decode_mask is not None:
                                # torch.empty above; the block holding the
                                # write head is read whole.
                                target[:, :, live_prefix:].zero_()
                            compacted.append(target)
                        # Replace one layer at a time so old+new
                        # full-capacity caches do not coexist across all
                        # layers at peak memory.
                        caches[layer] = tuple(compacted)
                else:
                    # Progressive above-tail compaction gathers the
                    # survivors into the FRONT of the existing storage and
                    # keeps narrowed contiguous views. Reallocating here
                    # (the branch above) adds a full-capacity transient per
                    # tensor at peak KV pressure — that exact allocation
                    # OOMed the v23 smoke run. The view keeps the original
                    # storage alive until the tail snap frees it, which is
                    # the same footprint the pre-progressive code held at
                    # full width; the transient shrinks to survivors x
                    # live-prefix. The survivor gather is materialized
                    # BEFORE the in-place copy, so overlapping rows cannot
                    # alias.
                    for layer, cache in enumerate(caches):
                        compacted = []
                        recurrent_layer = (
                            len(cache) == RECURRENT_CACHE_ARITY
                        )
                        for tensor in cache:
                            if recurrent_layer:
                                survivors = tensor.index_select(0, keep)
                                tensor[:compacted_count].copy_(survivors)
                                compacted.append(tensor[:compacted_count])
                                continue
                            survivors = tensor[
                                :, :, :live_prefix
                            ].index_select(0, keep)
                            tensor[
                                :compacted_count, :, :live_prefix
                            ].copy_(survivors)
                            compacted.append(tensor[:compacted_count])
                        caches[layer] = tuple(compacted)
                output = output.__class__(
                    belief=output.belief.index_select(0, keep),
                    input_latent=output.input_latent.index_select(0, keep),
                    logits=output.logits.index_select(0, keep),
                    caches=caches,
                )
                active = active.index_select(0, keep)

        # RNG draw order (one token draw per step) is part of the execution
        # schema; every mode consumes it identically.
        token = top_p_sample(
            output.logits, temperature, top_p, generator=generator
        )
        token_logprob = None
        if record_likelihoods:
            token_logprob = (
                output.logits.float()
                .log_softmax(-1)
                .gather(-1, token[:, None])
                .squeeze(-1)
            )

        # Stream writes address rows through the dense ``live_rows`` index
        # tensor and select participants with ``torch.where``, never with a
        # boolean row mask. A boolean index has a data-dependent output
        # shape, so each one copies its count to the host and drains this
        # launch-bound loop, which would make the ``SYNC_EVERY``
        # guard above pointless. Values are unchanged: every (row, slot) is
        # written at most once, so keeping the slot's current value for
        # non-participating rows is exactly what the masked write left there.
        record = active
        row_slots = (live_rows, position)
        next_position = position + 1
        next_slots = (live_rows, next_position)
        ones = action_mask.new_ones(())
        action_mask[row_slots] = torch.where(
            record, ones, action_mask[row_slots]
        )
        if token_logprob is not None:
            old_token_logprobs[row_slots] = torch.where(
                record, token_logprob.float(), old_token_logprobs[row_slots]
            )

        next_kind = kind[next_slots]
        kind[next_slots] = torch.where(
            record, next_kind.new_full((), TOKEN_SLOT), next_kind
        )
        token_ids[next_slots] = torch.where(
            record, token, token_ids[next_slots]
        )
        if stored_hidden_dim:
            # The belief that produced this token, stored at the token's own
            # slot: the +1-shifted carry the combined embedding replays. fp32
            # storage of a bf16 belief is exact, and the combiner upcasts the
            # live belief the same way, so rollout and replay inject the
            # identical value.
            hiddens[next_slots] = torch.where(
                record[:, None], output.belief.float(), hiddens[next_slots]
            )
        emitted += record.long()
        if stop_tensor is not None:
            ended |= record & torch.isin(token, stop_tensor)

        # Finished rows keep stepping on token 0 (their next slot stays PAD,
        # so the zero-initialized token_ids row feeds the embedding; every
        # loss masks it); batched caches make per-row early exit impractical.
        next_token_ids = token_ids[live_rows, next_position]
        if pin_emit:
            next_input = wrapper.embed_tokens(next_token_ids[:, None])
        else:
            # Every decode input carries the belief that produced its token.
            # Finished rows carry a stale belief onto token 0; nothing they
            # produce is recorded, exactly like their token embedding.
            next_input = wrapper.combined_input(next_token_ids, output.belief)
        step_pos, key_mask, step_block_mask = step_position(next_position)
        # Under reduce-overhead the step outputs live in the CUDA graph's
        # static pool and are only valid until the NEXT replay: everything
        # read from ``output`` above happens before this call, and every
        # consumer copies out (float()/gather/where).  Keep it that way.
        in_tail = tail_mask is not None or tail_starts is not None
        if in_tail and tail_step_core is not None:
            belief, logits = tail_step_core(
                next_input, caches, step_pos, key_mask, step_block_mask
            )
            output = StepOutput(
                belief=belief,
                input_latent=next_input.squeeze(1),
                logits=logits,
                caches=caches,
            )
        else:
            output = wrapper.step(
                next_input, caches, step_pos, key_mask, step_block_mask
            )
        caches = output.caches
        position = next_position

    # The KV cache is the largest allocation in the process (512 rows x 2304
    # keys x 6 layers = 14.5 GiB at the production shape) and nothing below
    # this point needs it: the batch stores replayable data, not activations.
    # On the flex path the cache does NOT die with these names, though --
    # something in the compiled step leaves it in a reference CYCLE, so only
    # the cyclic collector can reclaim it. Measured, chunk N's cache still
    # resident when chunk N+1 expanded its own: 14.880 GiB against the boolean
    # path's 1.379 GiB, which OOMs a 32 GB card. ``del`` alone does not fix it
    # (measured: same OOM, byte for byte); the collection does. It runs once
    # per rolled-out chunk -- twice a pool -- to reclaim 14.5 GiB, so the walk
    # is not worth conditioning on anything finer than the path that needs it.
    # ...and none of that applies when the caller owns the arena: there is no
    # per-chunk cache to reclaim, the arena is deliberately still referenced
    # elsewhere, and a full cyclic walk twice a pool would buy nothing.
    del caches, output
    if decode_mask is not None and not preallocated_caches:
        gc.collect()

    rolled = LatentRolloutBatch(
        kind=kind,
        token_ids=token_ids,
        hiddens=hiddens,
        action_mask=action_mask,
        old_token_logprobs=old_token_logprobs,
        old_values=old_values,
        rewards=torch.zeros_like(action_mask),
        reward_scalar=torch.zeros(batch, dtype=torch.float32, device=device),
        prompt_length=prompt_length,
        carry_injected=not pin_emit,
    )
    return rolled if not filler_rows else _drop_filler_rows(rolled, filler_rows)


def _drop_filler_rows(
    batch: LatentRolloutBatch, filler_rows: int
) -> LatentRolloutBatch:
    """Drop the trailing rows that existed only to hold a static row count.

    Fillers are appended after the real prompts and every row tensor is
    prompt-major, so the real trajectories are exactly the leading prefix.
    That prefix is only the right answer because row ORDER never moves, and
    the single thing keeping it still is ``not preallocated_caches`` in
    ``should_compact``: a compaction gathers by ``active.nonzero()``, which
    would leave the fillers scattered rather than trailing. Relaxing that
    clause to let an arena compact silently drops real rows and scores filler
    ones -- no shape disagrees anywhere.

    The prefix is CLONED rather than viewed: a view of dimension zero keeps
    the padded storage alive for as long as the batch lives, and this batch
    outlives the rollout by design. The fresh ``replay_layout_token`` the
    constructor mints is correct -- this is a structural transform, and a plan
    built against the padded layout must not survive it.
    """
    kept = {}
    for field in fields(batch):
        if field.name == "replay_layout_token":
            continue
        value = getattr(batch, field.name)
        if isinstance(value, Tensor):
            value = value[: value.size(0) - filler_rows].clone()
        kept[field.name] = value
    return LatentRolloutBatch(**kept)


def split_rollout_groups(
    batch: LatentRolloutBatch, group_size: int, prompt_lengths: Tensor
) -> list[LatentRolloutBatch]:
    """Undo a left-padded multi-group rollout into per-group batches.

    Each consecutive ``group_size`` block of rows shares one prompt (hence
    one pad length); dropping that group's pad columns makes its batch
    column-identical to a sequential single-prompt rollout, so scoring,
    refresh, and updates run on it unchanged.
    """
    total = batch.kind.size(0)
    if total % group_size:
        raise ValueError("batch rows must divide evenly into groups")
    groups = []
    for start in range(0, total, group_size):
        rows = slice(start, start + group_size)
        group_prompt = int(prompt_lengths[start])
        if bool((prompt_lengths[rows] != group_prompt).any()):
            raise ValueError("rows within a group must share one prompt length")
        pad = batch.prompt_length - group_prompt
        group_kind = batch.kind[rows, pad:]
        used = int((group_kind != PAD_SLOT).any(0).nonzero().max()) + 1
        sliced = {}
        for field in fields(batch):
            if field.name == "replay_layout_token":
                continue
            value = getattr(batch, field.name)
            if field.name == "prompt_length":
                value = group_prompt
            elif (
                isinstance(value, Tensor)
                and value.dim() >= 2
                and value.size(1) == batch.stream_length
            ):
                # Slice to this group's ACTUAL used tail before cloning. A
                # full-capacity clone retained ~8 GiB across sixteen groups
                # in the 5K-context run even though typical streams used
                # only ~140 positions.
                value = value[rows, pad : pad + used].clone()
            elif isinstance(value, Tensor) and value.dim() == 1:
                value = value[rows].clone()
            sliced[field.name] = value
        groups.append(LatentRolloutBatch(**sliced))
    return groups


def pack_rollout_groups_for_replay(
    groups: Sequence[LatentRolloutBatch],
    pin_memory: bool = False,
) -> LatentRolloutBatch:
    """Right-pad independent prompt groups into one exact replay batch.

    ``pin_memory`` allocates the packed CPU tensors in page-locked memory
    (via the caching host allocator, so repeated same-size packs recycle
    their blocks) so the subsequent H2D upload can run as a fast DMA copy;
    it is ignored for on-device groups or CUDA-less hosts.

    Every source stream keeps its original position zero and action boundary;
    only unused tail columns are appended. This matters because full-sequence
    replay is causal but has no left-padding key mask. A conventional
    left-padded batch would therefore change attention normalization and break
    behavior-policy log-probability equality.

    The result is intended for replay/update, where action masks carry the
    per-row boundaries. Its scalar ``prompt_length`` is the minimum source
    prompt length solely so length-aware row selection remains valid; callers
    must decode/score the individual groups before combining them.
    """
    if not groups:
        raise ValueError("at least one rollout group is required")
    devices = {group.kind.device for group in groups}
    if len(devices) != 1:
        raise ValueError("rollout groups must share one device")
    max_stream = max(group.stream_length for group in groups)
    total_rows = sum(group.kind.size(0) for group in groups)
    pin = (
        pin_memory
        and next(iter(devices)).type == "cpu"
        and torch.cuda.is_available()
    )
    combined: dict[str, Tensor | int] = {}
    for field in fields(groups[0]):
        if field.name == "prompt_length":
            combined[field.name] = min(group.prompt_length for group in groups)
            continue
        if field.name == "statistics_refreshed":
            combined[field.name] = all(
                group.statistics_refreshed for group in groups
            )
            continue
        if field.name == "carry_injected":
            flags = {group.carry_injected for group in groups}
            if len(flags) != 1:
                raise ValueError(
                    "cannot pack carry-injected and token-only rollouts"
                )
            combined[field.name] = flags.pop()
            continue
        if field.name == "replay_layout_token":
            # Packing creates a new logical layout; callers that deliberately
            # repack the same optimizer minibatch may replace this fresh token
            # before binding it to an existing plan.
            continue
        values = [getattr(group, field.name) for group in groups]
        if not all(isinstance(value, Tensor) for value in values):
            raise TypeError(f"unexpected non-tensor rollout field {field.name}")
        first = values[0]
        if first.dim() >= 2:
            if not all(
                value.dim() == first.dim()
                and value.shape[2:] == first.shape[2:]
                and value.size(1) == group.stream_length
                for value, group in zip(values, groups, strict=True)
            ):
                raise ValueError(
                    f"rollout field {field.name} has incompatible stream shapes"
                )
            fill = PAD_SLOT if field.name == "kind" else 0
            output = torch.full(
                (total_rows, max_stream, *first.shape[2:]),
                fill,
                dtype=first.dtype,
                device=first.device,
                pin_memory=pin,
            )
            row_start = 0
            for value in values:
                row_end = row_start + value.size(0)
                output[row_start:row_end, : value.size(1)].copy_(value)
                row_start = row_end
            combined[field.name] = output
        elif first.dim() == 1:
            if not all(value.dim() == 1 for value in values):
                raise ValueError(
                    f"rollout field {field.name} has incompatible row shapes"
                )
            rows = torch.cat(values)
            combined[field.name] = rows.pin_memory() if pin else rows
        else:
            raise ValueError(f"unsupported scalar rollout field {field.name}")
    return LatentRolloutBatch(**combined)


def scatter_replay_statistics(
    packed: LatentRolloutBatch,
    groups: Sequence[LatentRolloutBatch],
) -> None:
    """Copy refreshed packed behavior/value statistics into compact groups.

    Compact groups remain the canonical frozen behavior pool and can later be
    repacked in the identical order for exact PPO replay.

    Each group's slice is narrowed ON DEVICE before it crosses the bus. The
    packed batch is padded to its longest group, so transferring it whole and
    slicing on the host moved roughly 2.4x the bytes any group keeps; the
    values written are unchanged. The destinations stay pageable on purpose:
    they become the pool's long-lived behavior statistics, and pinning several
    gigabytes for the pool's lifetime would cost more in unswappable host
    memory than the extra transfer bandwidth is worth.
    """
    if not groups:
        raise ValueError("at least one rollout group is required")
    if packed.kind.size(0) != sum(group.kind.size(0) for group in groups):
        raise ValueError("packed rows do not match rollout groups")
    target_device = groups[0].kind.device
    if any(group.kind.device != target_device for group in groups):
        raise ValueError("rollout groups must share one device")
    statistic_names = (
        "old_token_logprobs",
        "old_values",
    )
    row_start = 0
    for group in groups:
        row_end = row_start + group.kind.size(0)
        for name in statistic_names:
            source = getattr(packed, name)[
                row_start:row_end, : group.stream_length
            ]
            if source.device == target_device:
                setattr(group, name, source.clone())
                continue
            # The row/stream narrowing above leaves a strided view; make it
            # dense on the source device (cheap at device bandwidth) so the
            # bus carries one contiguous block per statistic.
            setattr(group, name, source.contiguous().to(target_device))
        row_start = row_end
        # The groups now carry the packed batch's statistics, so they share
        # its refresh state.
        group.statistics_refreshed = packed.statistics_refreshed


def trim_stream(batch: LatentRolloutBatch, multiple: int = 1) -> LatentRolloutBatch:
    """Drop all-PAD tail columns into owning storage for compact replay.

    ``multiple`` rounds the kept length up to a bucket boundary, PADDING
    BEYOND the original stream when the content reaches into the last
    partial bucket — capping at the original length instead would leak one
    arbitrary stream shape per capped group and silently defeat the
    bounded-shape guarantee the compiled replay relies on (measured: the
    per-shape compiles never stopped).  Replay is padding-invariant — PAD
    inputs are zeroed, attention is causal, every loss is masked — so
    bucketing only bounds the set of stream shapes the compiled replay
    functions ever see.
    """
    used = int((batch.kind != PAD_SLOT).any(0).nonzero().max()) + 1
    if multiple > 1:
        used = -(-used // multiple) * multiple
    trimmed = {}
    for field in fields(batch):
        if field.name == "replay_layout_token":
            continue
        value = getattr(batch, field.name)
        if isinstance(value, Tensor) and value.dim() >= 2 and value.size(1) == batch.stream_length:
            if used <= batch.stream_length:
                if used < batch.stream_length:
                    # A narrow view would retain the full worst-case backing
                    # allocation and defeats the memory purpose of trimming.
                    value = value[:, :used].clone()
            else:
                padding = value.new_full(
                    (value.size(0), used - batch.stream_length, *value.shape[2:]),
                    PAD_SLOT if field.name == "kind" else 0,
                )
                value = torch.cat((value, padding), dim=1)
        trimmed[field.name] = value
    return LatentRolloutBatch(**trimmed)


def emitted_token_rows(batch: LatentRolloutBatch) -> list[list[int]]:
    """Emitted token ids per row, in stream order (excludes the prompt)."""
    # Bulk device transfers avoid a GPU synchronization for every trajectory.
    # Decoding/verifying is CPU work anyway, and this helper is called once
    # per rollout by both training and evaluation.
    generated = (
        batch.kind[:, batch.prompt_length:] == TOKEN_SLOT
    ).to(device="cpu")
    tokens = batch.token_ids[:, batch.prompt_length:].to(device="cpu")
    return [
        tokens[row][generated[row]].tolist()
        for row in range(batch.kind.size(0))
    ]


def assign_terminal_rewards(batch: LatentRolloutBatch, scores: Tensor) -> None:
    """Write one terminal reward per row at its final action position."""
    if scores.shape != batch.reward_scalar.shape:
        raise ValueError("one score per trajectory is required")
    positions = (
        batch.action_mask.size(1) - 1 - batch.action_mask.flip(1).argmax(1)
    ).long()
    batch.rewards.zero_()
    batch.rewards[torch.arange(batch.rewards.size(0), device=scores.device), positions] = scores
    batch.reward_scalar.copy_(scores)


def generated_slot_mask(batch: LatentRolloutBatch) -> Tensor:
    """Slots holding a model-generated token: the hasThought flag.

    A slot holds a generated token exactly where the previous slot took an
    action, so the flag is ``action_mask`` shifted right by one. Deriving it
    from per-row data rather than the scalar ``prompt_length`` is what keeps
    it exact for packed replay batches, whose scalar prompt length is only
    the minimum across groups.
    """
    mask = batch.action_mask.bool()
    shifted = torch.zeros_like(mask)
    shifted[:, 1:] = mask[:, :-1]
    return shifted


def assemble_stream_latents(
    wrapper: LatentThoughtModel, batch: LatentRolloutBatch
) -> Tensor:
    """Rebuild the exact (batch, stream, dim) inputs the rollout consumed."""
    token_latent = wrapper.embed_tokens(batch.token_ids)
    pad_scale = (batch.kind != PAD_SLOT)[..., None].to(token_latent.dtype)
    if batch.hiddens.size(-1) == 0:
        if batch.carry_injected:
            raise ValueError(
                "latent rollout discarded its carried hiddens "
                "(replay_storage=False); the stream cannot be replayed"
            )
        # Pinned-EMIT rollouts store zero-width hiddens and never inject; the
        # combiner cannot consume a zero-width carry.
        return token_latent * pad_scale
    # The dense combiner evaluation with a flag-select is value/gradient-
    # equivalent to boolean-index assignment. Unlike the latter, it has
    # static output shapes and keeps the full replay trunk inside one
    # Inductor graph.
    inputs = wrapper.combiner(
        token_latent, batch.hiddens, generated_slot_mask(batch)
    )
    return inputs * pad_scale


def replay_beliefs(
    wrapper: LatentThoughtModel, batch: LatentRolloutBatch
) -> tuple[Tensor, Tensor]:
    """One parallel teacher-forced pass over the stored stream.

    Returns (stream_inputs, beliefs).  Stored hiddens and tokens are
    constants (behavior data), so there is no BPTT through the carry; with
    grad enabled, gradients flow into the trunk, embeddings, and combiner.
    """
    stream_inputs = assemble_stream_latents(wrapper, batch)
    beliefs = wrapper.backbone.temporal_belief_from_token_latent(stream_inputs)
    return stream_inputs, beliefs


def replay_head_inputs(
    wrapper: LatentThoughtModel, batch: LatentRolloutBatch
) -> tuple[Tensor, Tensor]:
    """Replay the stream and derive everything the PPO heads consume.

    Returns (beliefs, stream_inputs). Renderer features are deliberately
    formed only at their consuming positions; they are positionwise, so dense
    prompt/token/pad evaluation is pure waste. Both
    ``refresh_old_statistics`` and the trainer's update step go through this
    single code path; that is what makes the recomputed "old" statistics
    exact — behavior-age-0 PPO ratios are one by construction.
    """
    stream_inputs, beliefs = replay_beliefs(wrapper, batch)
    # Downstream compaction deliberately uses ``view`` to reject hidden dense
    # copies. Inductor output strides are not a public contract, so make the
    # replay boundary's row-major contract explicit.
    return beliefs.contiguous(), stream_inputs.contiguous()


def compact_emit_token_logprobs(
    wrapper, emit_inputs: Tensor, emit_beliefs: Tensor, emit_targets: Tensor
) -> Tensor:
    """log P(target token) at each compact EMIT slot.

    Refresh and the trainer's update step both come through this one helper
    so their forwards stay bit-identical (the behavior-age-0 zero-clip
    canary). Its vocabulary-wide temporaries scale with slots x vocab; the
    replay planner's slot budget bounds that, not this function.

    The whole tail — renderer features, the readout GEMM, the logit softcap,
    the fp32 log-softmax and the target gather — is deliberately ONE function
    so the trainer can hand it to a single ``torch.compile`` artifact. Run
    eagerly it is roughly eight separate passes over a (slots, vocab) fp32
    tensor: ``_raw_logits`` already returns fp32, then the softcap spends a
    pow, an add, an rsqrt and two multiplies, then log-softmax reads and
    writes it again. Only the gathered (slots,) result is ever consumed, so
    every one of those intermediates is bandwidth spent to produce something
    immediately discarded.
    """
    features = wrapper.renderer_features(emit_inputs, emit_beliefs)
    logits = wrapper.backbone.logits_from_features(features)
    return (
        logits.float()
        .log_softmax(-1)
        .gather(-1, emit_targets[..., None])
        .squeeze(-1)
    )


def slot_index(mask: Tensor) -> Tensor:
    """Flat positions of a (rows, stream) mask's true slots, in row order.

    ``values[mask]`` gathers exactly these slots in exactly this order, so
    ``compact_slots``/``scatter_slots`` reproduce boolean indexing bit for
    bit. The point of naming the index is that deriving it costs ONE
    device->host synchronization -- boolean indexing has a data-dependent
    output shape, so every separate ``values[mask]`` pays that stall again,
    and the replay tail runs eagerly with a dozen of them per shard.
    """
    return mask.reshape(-1).nonzero().squeeze(-1)


def _slot_rows(values: Tensor) -> Tensor:
    """(rows, stream, ...) seen as (slots, ...), or an error.

    ``view`` rather than ``reshape`` on purpose: a non-contiguous input
    would send ``reshape`` through a copy of the whole dense tensor, which
    is the cost this path exists to avoid, and silently. Every operand
    here is contiguous today, but two of them (``beliefs``,
    ``stream_inputs``) come out of a compiled artifact whose output
    strides are an Inductor default rather than a contract, so the failure
    is deliberately loud. The slot count is spelled out instead of ``-1``
    because a zero-width trailing dimension makes ``-1`` ambiguous, and
    pinned-EMIT batches carry zero-width hiddens.
    """
    return values.view(values.shape[0] * values.shape[1], *values.shape[2:])


def compact_slots(values: Tensor, index: Tensor) -> Tensor:
    """``values[mask]`` for a (rows, stream, ...) tensor and a slot index."""
    return _slot_rows(values).index_select(0, index)


def compact_next_slots(values: Tensor, action_index: Tensor) -> Tensor:
    """Values consumed one stream position after compact action slots.

    Row/column indexing is intentional: a malformed final-column action must
    fail instead of silently wrapping to the next row's first slot.
    """
    stream_length = values.shape[1]
    rows = torch.div(action_index, stream_length, rounding_mode="floor")
    columns = action_index.remainder(stream_length) + 1
    return values[rows, columns]


def scatter_slots(
    destination: Tensor, index: Tensor, source: Tensor
) -> Tensor:
    """``destination[mask] = source`` for a slot index. In place."""
    _slot_rows(destination).index_copy_(0, index, source)
    return destination


def trajectory_used_lengths(batch: LatentRolloutBatch) -> Tensor:
    """Last non-padding column plus one for every independent trajectory."""
    positions = torch.arange(
        1, batch.stream_length + 1, device=batch.kind.device
    )
    return ((batch.kind != PAD_SLOT) * positions).amax(1)


def select_trajectory_rows(
    batch: LatentRolloutBatch, rows: Tensor, stream_length: int
) -> LatentRolloutBatch:
    """Materialize selected rows at a compact, shared stream length.

    Every stream dimension AND the row dimension leave here WEAKLY marked
    dynamic — a hint that steers the first trace and still lets a later
    guard specialize, unlike ``mark_dynamic``, which would raise on a
    length-1 shard. ``dynamic=True`` alone gives the first trace DUCK-typed
    sizes, so any input dim that happens to MATCH another gets the same
    symbol: ``hiddens`` is (rows, stream, model_dim) with model_dim 512, and
    512 is a legal ``--replay-bucket`` multiple, so a first shard of
    bucketed length 512 unifies the stream symbol with the hidden width and
    the combiner's 512-wide Linear then specializes it (measured: the NEXT
    shard length costs a full 18.5 s recompile; with the mark, 9 ms). The
    dim-0 mark on the row dimension is defensive hygiene for the duck-on
    configuration only: under the default ``--no-duck-shape``, DUCK and
    DYNAMIC both allocate a fresh symbol, so unmarked row counts were
    already dynamic and never the source of mid-run compiles (the ~12 s
    stalls k3_latent_10h paid per distinct row count were TileLang JIT
    compiles of FLA's KDA kernels, which bake the batch size into their
    cache key — fixed at the ``chunk_kda`` call site by the varlen form,
    not here). This is the one path refresh and update share, so marking
    here is what keeps a mark from drifting between them and splitting
    their single compiled artifact.
    """
    if stream_length < batch.prompt_length or stream_length > batch.stream_length:
        raise ValueError("invalid replay stream length")
    selected = {}
    for field in fields(batch):
        if field.name == "replay_layout_token":
            continue
        value = getattr(batch, field.name)
        if (
            isinstance(value, Tensor)
            and value.dim() >= 2
            and value.size(1) == batch.stream_length
        ):
            value = value[rows, :stream_length]
            torch._dynamo.maybe_mark_dynamic(value, 0)
            torch._dynamo.maybe_mark_dynamic(value, 1)
        elif isinstance(value, Tensor) and value.dim() >= 1:
            value = value[rows]
            torch._dynamo.maybe_mark_dynamic(value, 0)
        selected[field.name] = value
    return LatentRolloutBatch(**selected)


def iter_length_aware_microbatches(
    batch: LatentRolloutBatch,
    max_trajectories: int,
    attention_budget: int,
    bucket_multiple: int = 1,
    slot_budget: int | None = None,
):
    """Yield stable length-sorted replay shards under B*L^2 and B*L budgets.

    Causal full-stream replay is governed by two independent memory terms:
    attention area (B*L^2, quadratic) and the vocabulary head over the
    shard's slots (B*L x 50257-wide logits plus their autograd-retained
    log-softmax — the LINEAR term, and the larger one for fat shards of
    ordinary-length trajectories). ``slot_budget`` bounds the linear term;
    without it, raising ``attention_budget`` alone lets short-L shards grow
    their slot count unboundedly and the emit-logits pass OOMs before
    attention does. Stable sorting groups similarly sized independent
    trajectories; rare 1K-4K outliers automatically receive smaller shards.
    ``rows`` maps each compact shard back into the parent batch for
    refresh-stat writes. The final element repeats those row indices as the
    host-side Python list they were built from, so callers can make
    per-shard branch decisions (has-action) against a once-per-batch
    CPU table instead of a blocking device sync inside every shard.

    The whole plan is decided on the host before the first shard is
    yielded, which is what lets every shard's row indices reach the device
    in ONE asynchronous transfer from pinned memory. Building them per
    shard with ``torch.tensor(..., device=cuda)`` copies from pageable
    memory, and that is a blocking copy: it stalls the host at the top of
    every shard, right where the launch queue is deepest.
    """
    for shard_rows, shard_length, row_tensor in plan_length_aware_shards(
        batch, max_trajectories, attention_budget, bucket_multiple, slot_budget
    ):
        yield (
            select_trajectory_rows(batch, row_tensor, shard_length),
            row_tensor,
            shard_length,
            shard_rows,
        )


def _plan_host_shards(
    lengths: list[int],
    stream_length: int,
    max_trajectories: int,
    attention_budget: int,
    bucket_multiple: int,
    slot_budget: int | None,
) -> list[tuple[list[int], int]]:
    if max_trajectories < 1:
        raise ValueError("replay max trajectories must be positive")
    if attention_budget < 1:
        raise ValueError("replay attention budget must be positive")
    if bucket_multiple < 1:
        raise ValueError("replay bucket multiple must be positive")
    if slot_budget is not None and slot_budget < 1:
        raise ValueError("replay slot budget must be positive")

    order = sorted(range(len(lengths)), key=lambda row: (-lengths[row], row))
    plan: list[tuple[list[int], int]] = []
    shard: list[int] = []
    shard_length = 0

    def bucketed(length: int) -> int:
        return min(
            stream_length,
            -(-length // bucket_multiple) * bucket_multiple,
        )

    for row in order:
        candidate_length = max(shard_length, bucketed(lengths[row]))
        candidate_rows = len(shard) + 1
        exceeds = (
            candidate_rows > max_trajectories
            or candidate_rows * candidate_length * candidate_length
            > attention_budget
            or (
                slot_budget is not None
                and candidate_rows * candidate_length > slot_budget
            )
        )
        if shard and exceeds:
            plan.append((shard, shard_length))
            shard = []
            shard_length = 0
        shard.append(row)
        shard_length = max(shard_length, bucketed(lengths[row]))
    if shard:
        plan.append((shard, shard_length))
    return plan


def build_replay_plan(
    batch: LatentRolloutBatch,
    max_trajectories: int,
    attention_budget: int,
    bucket_multiple: int = 1,
    slot_budget: int | None = None,
    *,
    include_action_indices: bool = True,
) -> ReplayPlan:
    """Plan shards and compact action positions before replay starts.

    The production caller invokes this on the CPU batch in its packing worker.
    The fallback accepts any device for standalone callers, but a CUDA batch
    necessarily pays the metadata transfer that the production path avoids.
    """
    replay_masks = (
        torch.stack(
            (
                batch.kind != PAD_SLOT,
                batch.action_mask.bool(),
            )
        )
        if include_action_indices
        else (batch.kind != PAD_SLOT).unsqueeze(0)
    )
    if replay_masks.device.type != "cpu":
        # One metadata transfer for fallback callers that have already moved
        # the batch. The trainer's packing-worker path never enters this arm.
        replay_masks = replay_masks.to("cpu")
    positions = torch.arange(1, batch.stream_length + 1)
    lengths = (replay_masks[0] * positions).amax(1).tolist()
    host_shards = _plan_host_shards(
        lengths,
        batch.stream_length,
        max_trajectories,
        attention_budget,
        bucket_multiple,
        slot_budget,
    )

    planned_actions = []
    for host_row_list, stream_length in host_shards:
        host_rows = tuple(host_row_list)
        local_rows = torch.tensor(host_rows, dtype=torch.long)
        planned_actions.append(
            (
                host_rows,
                stream_length,
                (
                    slot_index(
                        replay_masks[1, local_rows, :stream_length]
                    )
                    if include_action_indices
                    else torch.empty(0, dtype=torch.long)
                ),
                local_rows,
            )
        )

    def concatenate_indices(parts: list[Tensor]) -> Tensor:
        return (
            torch.cat(parts)
            if parts
            else torch.empty(0, dtype=torch.long)
        )

    row_values = concatenate_indices(
        [local_rows for *_, local_rows in planned_actions]
    )
    emit_values = concatenate_indices(
        [emit_index for _, _, emit_index, _ in planned_actions]
    )
    emit_base = row_values.numel()
    row_offset = emit_offset = 0
    specs = []
    for (
        host_rows,
        stream_length,
        emit_index,
        _,
    ) in planned_actions:
        specs.append(
            ReplayShardSpec(
                host_rows=host_rows,
                stream_length=stream_length,
                row_offset=row_offset,
                row_count=len(host_rows),
                emit_offset=emit_base + emit_offset,
                emit_count=len(emit_index),
            )
        )
        row_offset += len(host_rows)
        emit_offset += emit_index.numel()
    indices = torch.cat((row_values, emit_values))
    if batch.kind.device.type == "cpu" and batch.kind.is_pinned():
        indices = indices.pin_memory()
    # In production this allocation happens pinned in the CPU packing worker,
    # so ``ReplayPlan.to`` only enqueues one asynchronous index DMA.
    return ReplayPlan(
        batch_rows=batch.kind.size(0),
        batch_stream_length=batch.stream_length,
        batch_layout_token=batch.replay_layout_token,
        max_trajectories=max_trajectories,
        attention_budget=attention_budget,
        bucket_multiple=bucket_multiple,
        slot_budget=slot_budget,
        has_action_indices=include_action_indices,
        specs=tuple(specs),
        indices=indices,
    )


def iter_planned_replay_microbatches(
    batch: LatentRolloutBatch,
    replay_plan: ReplayPlan,
):
    """Yield microbatches and their precomputed compact action indices."""
    replay_plan.validate_for(batch)
    for shard in replay_plan.shards():
        yield (
            select_trajectory_rows(
                batch, shard.rows, shard.stream_length
            ),
            shard,
        )


def plan_length_aware_shards(
    batch: LatentRolloutBatch,
    max_trajectories: int,
    attention_budget: int,
    bucket_multiple: int = 1,
    slot_budget: int | None = None,
) -> list[tuple[list[int], int, Tensor]]:
    """The shard plan: (host rows, stream length, device row index) each.

    Separated from the yielding loop so the plan is complete before any
    shard runs, and so tests can assert on it without materializing
    microbatches.
    """
    lengths = trajectory_used_lengths(batch).to(device="cpu").tolist()
    plan = _plan_host_shards(
        lengths,
        batch.stream_length,
        max_trajectories,
        attention_budget,
        bucket_multiple,
        slot_budget,
    )

    device = batch.kind.device
    flat = torch.tensor(
        [row for shard_rows, _ in plan for row in shard_rows],
        dtype=torch.long,
    )
    if device.type == "cuda":
        # One pinned, asynchronous transfer for every shard's rows. The
        # staging buffer loses its last Python reference on this line; it
        # survives because ``pin_memory`` allocates through the CUDA
        # caching HOST allocator, which records a stream event on free and
        # will not recycle the block until the copy retires. Disabling that
        # allocator would turn this into a race that yields wrong row
        # indices without crashing.
        flat = flat.pin_memory().to(device, non_blocking=True)
    else:
        flat = flat.to(device)
    planned = []
    offset = 0
    for shard_rows, length in plan:
        planned.append(
            (shard_rows, length, flat.narrow(0, offset, len(shard_rows)))
        )
        offset += len(shard_rows)
    return planned


def refresh_old_statistics(
    wrapper: LatentThoughtModel,
    critic,
    batch: LatentRolloutBatch,
    max_trajectories: int = 32,
    attention_budget: int = 4 * 1024 * 1024,
    bucket_multiple: int = 1,
    slot_budget: int | None = None,
    replay_plan: ReplayPlan | None = None,
) -> None:
    """Overwrite the stored PPO statistics with parallel-replay recomputations.

    The stepwise rollout and the parallel replay reduce through the trunk in
    different orders; at bf16 scale that drifts log-probs enough to put a
    noise floor under PPO ratios, clip fractions, and GAE inputs.  Rewriting
    ``old_values``/``old_token_logprobs`` through the
    exact update-step code path removes the drift; positions outside the
    consuming masks are overwritten too, but nothing ever reads them.

    The forward here runs GRAD-ENABLED on purpose, even though the graph is
    discarded: under torch.compile the grad mode is a guard, and a no-grad
    trace would give this refresh a different compiled artifact (different
    kernel fusions, different bf16 reduction order) than the update step —
    reintroducing exactly the behavior-age-0 ratio drift it exists to remove.
    Eagerly the numerics are identical either way; the cost is one
    forward's transient activation memory.

    ``old_values`` come from the separate critic model — the rollout itself
    never computes values, so this is where GAE's baseline is filled in.
    Replay is stably length-sorted and split only across independent
    trajectories under a B*L^2 budget, bounding quadratic attention memory
    without changing any consumed statistic. The update path uses the same
    deterministic planner so compiled refresh/update forwards remain
    numerically identical for the first behavior minibatch.
    """
    if replay_plan is None:
        replay_plan = build_replay_plan(
            batch,
            max_trajectories,
            attention_budget,
            bucket_multiple,
            slot_budget,
        ).to(batch.kind.device)
    replay_plan.validate_settings(
        max_trajectories,
        attention_budget,
        bucket_multiple,
        slot_budget,
        require_action_indices=True,
    )
    for microbatch, shard in iter_planned_replay_microbatches(
        batch, replay_plan
    ):
        rows = shard.rows
        stream_length = shard.stream_length
        beliefs, stream_inputs = replay_head_inputs(
            wrapper, microbatch
        )
        values = critic.values(microbatch).float()
        emit_index = shard.emit_index
        # Grad-enabled on purpose, for the same reason the compiled replay
        # above is: the readout tail is itself a compiled artifact now, and
        # grad mode is a dynamo guard, so a no-grad refresh would trace a
        # SECOND artifact whose fusions — and therefore whose reduction order
        # — need not match the update step's. That is exactly the drift the
        # behavior-age-0 canary exists to catch. The graph is discarded
        # immediately; the cost is one forward's saved activations.
        compact_token_logprobs = compact_emit_token_logprobs(
            wrapper,
            compact_slots(stream_inputs, emit_index),
            compact_slots(beliefs, emit_index),
            compact_next_slots(microbatch.token_ids, emit_index),
        ).detach()
        token_logprobs = torch.zeros_like(microbatch.old_token_logprobs)
        scatter_slots(token_logprobs, emit_index, compact_token_logprobs)
        with torch.no_grad():
            # Advanced row indexing materializes a copy, so assignment must
            # target the parent explicitly (``view.copy_`` would update only
            # the temporary).
            batch.old_values[rows, :stream_length] = values
            batch.old_token_logprobs[rows, :stream_length] = token_logprobs
    batch.statistics_refreshed = True
