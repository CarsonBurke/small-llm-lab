"""Latent-thought policy modules layered over the backbone.

A thought is the deterministic hidden state that produced a generated token:
the post-final-norm belief, exactly what the CE readout consumes. When a
generated token is fed back as input, its producing belief rides along as a
gated residual on the token embedding —
``combined = embed(x) + has_thought * (g * W(h) + b)`` — followed by a
prenorm-residual relu^2 MLP stack before the trunk. Prompt (read) tokens carry
no hidden, and neither does the input of the step that produces the first
output token: a hidden exists only where the input token was itself generated
by the model.

The carried hidden is detached rollout data, so training needs no BPTT and no
thought-specific objective: the only actions are tokens and the update is
standard VAPO. The stream has no thought slots — every position is a token
position, so stream length equals prompt plus emitted tokens.

Deferred by design: test-time-read-compute (carrying hiddens for read
tokens).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask

import nanogpt_mini_model

RENDERER_FEATURES_SCHEMA = "combined_input+belief/v1"
ROLLOUT_POLICY_SCHEMA = "deterministic_hidden_carry/v1"
# Pinned-EMIT reasoning modes never inject a hidden: the rollout is a plain
# token policy. The tag embeds the mode because cot and none differ in their
# trained emission budgets, so their checkpoints are not one policy.
PINNED_EMIT_ROLLOUT_POLICY_SCHEMAS = {
    "cot": "pinned_emit_token_only_cot/v2",
    "none": "pinned_emit_token_only_answer_prefix/v2",
}


def rollout_policy_schema_for_mode(reasoning_mode: str) -> str:
    """The rollout-policy schema tag a reasoning mode trains and resumes."""
    if reasoning_mode == "latent":
        return ROLLOUT_POLICY_SCHEMA
    try:
        return PINNED_EMIT_ROLLOUT_POLICY_SCHEMAS[reasoning_mode]
    except KeyError:
        raise ValueError(f"unknown reasoning mode {reasoning_mode!r}") from None


THOUGHT_INPUT_SCHEMA = "gated_hidden_residual_prenorm_mlp/v1"


def combiner_init_kwargs_from_checkpoint(payload: dict) -> dict:
    """Recover combiner geometry before constructing a checkpoint wrapper.

    The combiner's tensor shapes are fixed by the saved CLI arguments, so a
    caller must rebuild the wrapper with the recorded geometry before strict
    loading rather than relying on shape checks.
    """
    saved_args = payload.get("args", {})
    return {
        "mlp_hidden": saved_args.get("combined_mlp_hidden"),
        "num_blocks": saved_args.get("combined_mlp_blocks", 1),
        "gain_init": saved_args.get("hidden_carry_gain_init", 0.0),
    }


def validate_renderer_checkpoint(
    payload: dict,
    checkpoint: str,
    *,
    expected_rollout_policy_schema: str = ROLLOUT_POLICY_SCHEMA,
) -> None:
    """Reject wrapper checkpoints trained with incompatible policy semantics."""
    actual = payload.get("renderer_features_schema")
    if actual != RENDERER_FEATURES_SCHEMA:
        raise ValueError(
            f"incompatible latent-policy checkpoint {checkpoint!r}: renderer "
            f"schema is {actual!r}, expected {RENDERER_FEATURES_SCHEMA!r}. "
            "Checkpoints predating deterministic hidden carry (including "
            "every stochastic-thought policy) cannot be resumed or evaluated "
            "as this policy."
        )
    rollout_policy = payload.get("rollout_policy_schema")
    if rollout_policy != expected_rollout_policy_schema:
        raise ValueError(
            f"incompatible latent-policy checkpoint {checkpoint!r}: rollout "
            f"schema is {rollout_policy!r}, expected "
            f"{expected_rollout_policy_schema!r}. The checkpoint was trained "
            "under a different reasoning mode or deployed policy and cannot "
            "be resumed or evaluated as this policy."
        )
    thought_input = payload.get("thought_input_schema")
    if thought_input != THOUGHT_INPUT_SCHEMA:
        raise ValueError(
            f"incompatible latent-policy checkpoint {checkpoint!r}: thought "
            f"input schema is {thought_input!r}, expected "
            f"{THOUGHT_INPUT_SCHEMA!r}. The checkpoint injected thoughts "
            "through a different input path and cannot be resumed or "
            "evaluated as this policy."
        )


class CombinedEmbedding(nn.Module):
    """Token+thought combined-embedding stack shared by actor and critic.

    ``forward`` maps a token latent and its (detached) carried hidden into the
    stream input the trunk consumes:

        combined = base + gain * W(hidden) + type_bias
        mixed    = combined + mlp(rms_norm(combined))   (per block)
        out      = mixed where has_thought else base

    The whole path — injection and MLP stack — applies only at hasThought
    positions: prompt (read) tokens enter the trunk as plain embeddings in
    rollout prefill, so replay must give them the identical plain path or the
    behavior-age-0 exactness contract breaks. The select is a ``torch.where``
    rather than ``base + flag * (mixed - base)`` because the latter is not
    bitwise ``mixed`` at flagged positions.

    ``W`` is a unit-orthogonal full-width map behind one learned scalar gain,
    initialized at zero so a fresh combiner is exactly the pretrained token
    path; the gain has a first-step gradient through ``W(h) . dL/dcombined``
    while matrix updates stay scaled by the explicit gain, the same treatment
    the codebase gives every fresh full-width map. ``type_bias`` is the
    learned hasThought type embedding: it is gated by the flag but not by the
    gain, so the "this token was generated" signal can learn independently of
    the content channel. Each MLP block reuses the pretraining ``MLP``
    (relu^2) behind a fresh ``RMSNorm`` with a zero-initialized ``proj``, so
    the whole stack is an exact identity at initialization.

    The injection branch is computed in fp32 and rounded to the base latent's
    dtype once, before the residual add — the same cast order at rollout and
    replay, which is what keeps behavior-age-0 PPO ratios exactly one.
    """

    def __init__(
        self,
        model_dim: int,
        mlp_hidden: int | None = None,
        num_blocks: int = 1,
        gain_init: float = 0.0,
    ):
        super().__init__()
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")
        self.carry = nn.Linear(model_dim, model_dim, bias=False)
        nn.init.orthogonal_(self.carry.weight)
        self.gain = nn.Parameter(torch.tensor(float(gain_init)))
        self.type_bias = nn.Parameter(torch.zeros(model_dim))
        self.norms = nn.ModuleList(
            nanogpt_mini_model.RMSNorm(model_dim) for _ in range(num_blocks)
        )
        self.mlps = nn.ModuleList(
            nanogpt_mini_model.MLP(model_dim, mlp_hidden)
            for _ in range(num_blocks)
        )
        with torch.no_grad():
            for mlp in self.mlps:
                # The pretraining init recipe: forward weights at the trunk
                # scale, projections zeroed so each residual block starts as
                # an exact identity.
                mlp.fc.weight.normal_(
                    std=0.33**0.5 / mlp.fc.weight.size(-1) ** 0.5
                )
                mlp.fc.bias.zero_()
                mlp.proj.weight.zero_()
                mlp.proj.bias.zero_()

    def inject(self, base: Tensor, hidden: Tensor) -> Tensor:
        """Add the gated hidden residual to every position densely."""
        with torch.autocast(device_type=base.device.type, enabled=False):
            injected = (
                self.gain.float() * F.linear(hidden.float(), self.carry.weight)
                + self.type_bias.float()
            )
        return base + injected.to(base.dtype)

    def mix(self, combined: Tensor) -> Tensor:
        """Run the prenorm-residual relu^2 MLP stack over every position."""
        for norm, mlp in zip(self.norms, self.mlps, strict=True):
            combined = combined + mlp(norm(combined))
        return combined

    def forward(
        self, base: Tensor, hidden: Tensor, has_thought: Tensor | None = None
    ) -> Tensor:
        """Combined stream inputs; ``None`` flag means every position carries.

        The dense compute-then-select keeps static shapes (no boolean
        indexing) while making flagged positions bitwise equal to the
        all-injected decode path and unflagged positions bitwise equal to the
        plain token path.
        """
        mixed = self.mix(self.inject(base, hidden))
        if has_thought is None:
            return mixed
        return torch.where(has_thought[..., None], mixed, base)


def kv_range_blocks(
    kv_starts: Tensor,
    kv_lengths: Tensor,
    *,
    block_size: int,
    blocks_per_row: int,
    max_block: int,
    page_table: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Block metadata for rows that each attend one contiguous ``[start, stop)``.

    Returns ``(partial_count, partial_indices, full_count, full_indices)`` in
    the layout ``BlockMask.from_kv_blocks`` wants. A range touches at most two
    partial blocks — its first, when the start is not block-aligned, and its
    last, when the stop is not — with everything between them fully live.
    Splitting them out matters: the flex kernel skips the mask evaluation
    entirely on full blocks, which is most of them.

    Both tables are ``blocks_per_row`` wide even though a range touches at
    most two partial blocks. That is not slack: the Triton decode kernel
    offsets FULL_KV_IDX by ``stride("KV_IDX")`` and bounds it by
    ``size("KV_IDX", -1)``, so a narrower partial table makes the kernel read
    the full table at the wrong row stride and modulus. Full blocks skip
    ``mask_mod``, so nothing downstream corrects it — the kernel silently
    attends the wrong keys for every row but the first (measured: 0.68 max
    error against a 0.20-magnitude signal). Entries past the block count are
    never read.

    Without ``page_table`` the rows address one common block space, which is
    what an ordinary batched cache wants. Every decode mask in this codebase is
    a contiguous range — left padding is a prefix, causality is a suffix bound
    — so this one shape serves them all.

    ``page_table`` (rows x ``blocks_per_row``, already row-selected) maps each
    row's logical block to a physical one through a GATHER, which is what makes
    a shared page pool possible: a row's blocks may sit anywhere in the pool
    rather than in one reserved run. The range stays contiguous in LOGICAL
    space and only the emitted indices are scattered, so every property above
    still holds.

    An EMPTY range (``start >= stop``) yields zero blocks of both kinds, and
    the kernel returns exactly zero for that row (measured, not assumed — a
    zero softmax denominator could as easily have given NaN, which 0-weighted
    masking would then spread). That is what lets a caller pad a decode batch
    up to a static bucket: a filler row reads no KV at all, so the padding is
    free rather than a full row of attention.
    """
    if page_table is not None and page_table.shape != (
        kv_starts.numel(),
        blocks_per_row,
    ):
        # Both gathers bound their index by ``blocks_per_row``, so a narrower
        # table indexes out of range. Raised here it is a caller-side
        # ValueError; left to the gather it is a device-side assert from
        # inside a compiled decode step.
        raise ValueError("page_table must have shape [rows, blocks_per_row]")
    empty = kv_starts >= kv_lengths
    first_block = torch.div(kv_starts, block_size, rounding_mode="floor")
    first_is_partial = (kv_starts % block_size != 0) & ~empty
    full_start = first_block + first_is_partial
    full_count = torch.where(
        empty,
        torch.zeros_like(kv_starts),
        (
            torch.div(kv_lengths, block_size, rounding_mode="floor") - full_start
        ).clamp_min(0),
    )
    offsets = torch.arange(
        blocks_per_row, dtype=torch.int32, device=kv_starts.device
    )
    if page_table is None:
        full_indices = (
            full_start.to(torch.int32)[:, None, None, None]
            + offsets[None, None, None]
        ).clamp_max(max_block)
    else:
        # Clamped only to keep the gather in bounds: entries past ``full_count``
        # are never read, so which valid page they name does not matter.
        logical_full = (full_start[:, None] + offsets[None].to(full_start.dtype)).clamp(
            0, blocks_per_row - 1
        )
        full_indices = (
            page_table.gather(1, logical_full)
            .to(torch.int32)[:, None, None, :]
            .clamp_max(max_block)
        )

    last_block = torch.div(
        (kv_lengths - 1).clamp_min(0), block_size, rounding_mode="floor"
    )
    last_is_partial = (kv_lengths % block_size != 0) & ~empty
    distinct_last = last_block != first_block
    partial_count = first_is_partial.to(torch.int32) + (
        last_is_partial & (distinct_last | ~first_is_partial)
    ).to(torch.int32)
    if page_table is not None:
        bounded = torch.stack(
            (
                first_block.clamp(0, blocks_per_row - 1),
                last_block.clamp(0, blocks_per_row - 1),
            ),
            dim=-1,
        )
        mapped = page_table.gather(1, bounded)
        first_block, last_block = mapped[:, 0], mapped[:, 1]
    first_block = first_block.clamp_max(max_block)
    last_block = last_block.clamp_max(max_block)
    # Pack the present partial blocks at the front. When the first block is
    # full, the final partial block occupies index zero.
    partial_zero = torch.where(first_is_partial, first_block, last_block)
    present = torch.stack((partial_zero, last_block), dim=-1).to(torch.int32)
    partial_indices = torch.zeros_like(full_indices)
    front = min(present.size(-1), blocks_per_row)
    partial_indices[..., :front] = present[:, None, None, :front]
    return (
        partial_count[:, None, None],
        partial_indices,
        full_count.to(torch.int32)[:, None, None],
        full_indices,
    )


class DecodeRangeMask:
    """Per-step decode ``BlockMask``: row ``b`` attends ``[start_b, stop)``.

    This is the ordinary batched-cache counterpart to
    ``PagedGenerationCache.block_mask``. It exists because handing SDPA a
    boolean ``attn_mask`` disqualifies every fused attention backend and drops
    the step onto the memory-efficient kernel, which the v25 kernel profile put
    at 32% of pool device time with 12672 of its 12684 calls coming from the
    compiled decode step. A range carries the same information as the boolean
    row mask it replaces and reaches flex decoding instead.

    The mask covers the WHOLE preallocated cache width, so the decode step's
    shapes never depend on the position — the block table alone decides how
    much work the kernel does, and empty blocks cost nothing. Cache slots past
    the write head are unreachable through the mask but must still hold finite
    values, exactly as on the boolean full-cache path: a masked score is -inf,
    whose zero weight times a NaN value is still NaN.

    ``mask_mod`` and the two tensors it closes over are allocated ONCE and
    refilled in place, because ``mode="reduce-overhead"`` replays a CUDA graph
    that captured their addresses. NOT for compile-count reasons: measured on
    torch 2.13, constructing a fresh ``DecodeRangeMask`` every step costs zero
    extra compiles, since Dynamo guards on the code object and tensor metadata
    rather than closure identity. Only the block tables are rebuilt each step;
    those are ordinary tensor inputs the cudagraph copies in.
    """

    DEFAULT_BLOCK_SIZE = 128

    #: Granularity the caller rounds its live ROW count up to, so the decode
    #: step sees finitely many static batch sizes. Deliberately a linear grid,
    #: not powers of two: the rounded count is also the width the KV cache is
    #: reallocated at, so a power-of-two grid holds a 512-row cache until
    #: fewer than 256 rows are live. Measured at the production shape, that
    #: alone put flex 6.2 GiB above the boolean control's peak (20.50 vs
    #: 14.27 GiB) and OOMed a 32 GB card. A multiple of 64 caps the surplus at
    #: 64 rows while costing about as many specializations as powers of two
    #: did, because the >=25%-dead compaction hysteresis, not the grid, is
    #: what decides how often the count actually moves.
    DEFAULT_ROW_BUCKET = 64

    def __init__(
        self,
        batch_size: int,
        kv_width: int,
        device: torch.device,
        *,
        block_size: int = DEFAULT_BLOCK_SIZE,
        row_bucket: int = DEFAULT_ROW_BUCKET,
    ) -> None:
        if block_size < 16 or block_size & (block_size - 1):
            # Same floor as make_paged_generation_cache: the Triton decode
            # kernel tiles the KV axis in powers of two and its minimum tile
            # is 16, so anything else fails to lower rather than running slow.
            raise ValueError(
                f"block_size {block_size} must be a power of two >= 16"
            )
        if kv_width % block_size:
            raise ValueError(
                f"kv_width {kv_width} must be a multiple of the flex KV block "
                f"size {block_size}; a ragged final block would attend cache "
                f"slots that do not exist"
            )
        if row_bucket < 1:
            raise ValueError(f"row_bucket {row_bucket} must be positive")
        self.kv_width = kv_width
        self.block_size = block_size
        self.row_bucket = row_bucket
        self.blocks = kv_width // block_size
        self.kv_starts = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.kv_stops = torch.ones(batch_size, dtype=torch.long, device=device)
        # The compiled step closes over these; a CUDA-graph replay needs their
        # addresses to stay put, exactly like the static caches.
        torch._dynamo.mark_static_address(self.kv_starts)
        torch._dynamo.mark_static_address(self.kv_stops)
        starts, stops = self.kv_starts, self.kv_stops

        def live_key(
            batch: Tensor, head: Tensor, query: Tensor, key: Tensor
        ) -> Tensor:
            del head, query
            return (key >= starts[batch]) & (key < stops[batch])

        self.mask_mod = live_key

    def build(
        self,
        kv_starts: Tensor,
        kv_stop: Tensor | int,
        live: Tensor | None = None,
    ) -> BlockMask:
        """Mask for one step; ``kv_stop`` is exclusive and shared by all rows.

        Fewer rows than the builder holds is normal — the rollout compacts
        finished rows away — and the surplus buffer entries are simply never
        addressed. One builder covers every survivor count for the buffers'
        sake, not the compiler's: see the class docstring on why rebuilding it
        costs no extra compiles.

        ``live`` is the point of the whole exercise. Rows it marks False get an
        empty range, which costs no KV read and returns exactly zero, so a
        caller can pad the batch up to a static row count instead of compacting
        to an exact one. Flex decoding only lowers for static shapes, and the
        rollout's survivor count is data-dependent; padding to a bucket is what
        reconciles the two, and it is only affordable because the padding is
        free. Measured at the production shape: a 256-row bucket holding 144
        live rows costs 0.6% more than an exact 144-row batch, while the SDPA
        step it replaces costs 59% more.
        """
        rows = kv_starts.numel()
        starts = self.kv_starts[:rows]
        stops = self.kv_stops[:rows]
        starts.copy_(kv_starts)
        stops.fill_(kv_stop)
        if live is not None:
            # start >= stop is the empty range; zero is below every start.
            stops.mul_(live)
        partial_count, partial_indices, full_count, full_indices = kv_range_blocks(
            starts,
            stops,
            block_size=self.block_size,
            blocks_per_row=self.blocks,
            max_block=self.blocks - 1,
        )
        return BlockMask.from_kv_blocks(
            partial_count,
            partial_indices,
            full_count,
            full_indices,
            # FlexDecoding uses a minimum BLOCK_M of 16 on CUDA and requires
            # the sparse Q tile to be divisible by it. The logical query is
            # still length one; a 16-row metadata tile simply covers it.
            BLOCK_SIZE=(16, self.block_size),
            mask_mod=self.mask_mod,
            seq_lengths=(1, self.kv_width),
            compute_q_blocks=False,
        )


@dataclass
class StepOutput:
    """Everything one stream step exposes to rollout and training code.

    ``belief`` is the post-final-norm hidden — the thought the next generated
    token carries. No value: the critic is a separate model that scores stored
    streams in parallel (``refresh_old_statistics``); the stepwise path never
    values.
    """

    belief: Tensor
    input_latent: Tensor
    logits: Tensor
    caches: list[tuple[Tensor, ...]]


@dataclass
class PromptPrefixBank:
    """Read-only pool-wide prompt states and dense prefix KV.

    The bank is built once by a single right-aligned batched prefill. Admission
    only selects pool-global rows and scatters their cached state into reusable
    paged slots; it never traverses the transformer trunk.
    """

    layers: list[tuple[Tensor, ...]]
    output: StepOutput
    kv_starts: Tensor
    prompt_width: int

    @property
    def num_prompts(self) -> int:
        return self.kv_starts.numel()


@dataclass
class PagedGenerationCache:
    """Fixed-capacity packed KV storage for continuous-refill decoding.

    Slot ``s``'s logical page ``l`` lives at physical page
    ``page_table[s, l]``, and ``page_home`` is its inverse — the logical
    address ``s * pages_per_lane + l`` that each physical page stands for.
    Folding the owning slot into the inverse rather than storing the
    within-slot index alone is what keeps ``mask_mod`` a total function: a
    page belonging to another slot resolves outside this row's window and is
    rejected on value, without a second gather to check ownership. Pages no
    slot owns must therefore be given a home of ``capacity * pages_per_lane``
    or beyond, which no row's window can contain. Refill overwrites the slot's
    right-aligned prompt window; per-row KV starts/lengths in the
    FlexAttention mask exclude left padding and make any stale suffix
    unreachable without clearing it.

    The indirection exists so the pool can be SHARED. With one contiguous run
    reserved per slot, every slot must reserve ``rounded_length`` whatever it
    actually generates, and at the production shape the median row uses about
    a fifth of it — so capacity, and with it the useful work per kernel
    launch, is set by the worst case rather than the average. A page table
    costs one gather per step and lets a slot's pages come from anywhere.
    ``make_paged_generation_cache`` still hands back the contiguous identity
    mapping, which keeps today's behavior exactly; only the addressing is
    decoupled from the reservation.

    Recurrent (KDA) layers opt out of all of the above: their entry in
    ``layers`` is a 4-tuple of lane-row arenas (three conv windows plus the
    fp32 delta-rule state) indexed directly by slot on dim 0, with rows
    ``[capacity, 2 * capacity)`` reserved as per-row scratch sinks for dead
    decode rows. Pages, addresses, and masks never apply to them; admission
    overwrites a lane's rows wholesale, which is what stands in for the
    ``kv_starts`` stale-suffix masking a recurrent state cannot express.
    """

    layers: list[tuple[Tensor, ...]]
    capacity: int
    max_length: int
    page_size: int
    rounded_length: int
    kv_starts: Tensor
    page_table: Tensor
    page_home: Tensor

    @property
    def pages_per_lane(self) -> int:
        return self.rounded_length // self.page_size

    @property
    def total_pages(self) -> int:
        return self.page_home.numel()

    @property
    def scratch_addresses(self) -> Tensor:
        """One write sink per row, in pages no slot owns."""
        base = self.capacity * self.pages_per_lane * self.page_size
        return torch.arange(
            base, base + self.capacity, device=self.page_home.device
        )

    def assign_pages(
        self, slot_ids: Tensor, logical_pages: Tensor, physical_pages: Tensor
    ) -> None:
        """Rehome ``slot_ids``' logical pages onto ``physical_pages``.

        The only supported way to move a page, because the mapping and its
        inverse must move together: the write side and the block table read
        ``page_table``, ``mask_mod`` reads ``page_home``, and an update that
        touched one alone would leave a row attending fewer keys than it wrote
        — finite logits, wrong context, no error anywhere. An allocator that
        goes through here cannot make that mistake.
        """
        self.page_table[slot_ids, logical_pages] = physical_pages
        self.page_home[physical_pages] = (
            slot_ids * self.pages_per_lane + logical_pages
        )

    def validate(self) -> None:
        """Assert the two directions of the mapping still agree.

        Not on the hot path — this is for tests and for an allocator's own
        debug assertions. It is the invariant every silent-corruption failure
        mode in this class reduces to.
        """
        owned = self.page_table.flatten()
        expected = torch.arange(owned.numel(), device=owned.device)
        if owned.unique().numel() != owned.numel():
            raise ValueError("a physical page is assigned to two slots")
        if not bool((self.page_home[owned] == expected).all()):
            raise ValueError("page_home is not the inverse of page_table")

    def token_addresses(
        self, slot_ids: Tensor, positions: Tensor
    ) -> Tensor:
        """Physical cache index of each row's logical ``position``.

        The one definition of logical-to-physical on the write side; the read
        side goes through the same table in ``block_mask``. Keeping them in one
        place is the point — a write that disagreed with the block table would
        put a token somewhere the mask never looks, and the row would silently
        attend a stale slot instead.
        """
        logical_page = torch.div(
            positions, self.page_size, rounding_mode="floor"
        )
        physical_page = self.page_table[slot_ids, logical_page]
        return physical_page * self.page_size + positions % self.page_size

    def block_mask(self, slot_ids: Tensor, kv_lengths: Tensor) -> BlockMask:
        """Build O(active_rows * pages_per_lane) physical-page metadata."""
        if slot_ids.ndim != 1 or kv_lengths.shape != slot_ids.shape:
            raise ValueError("slot_ids and kv_lengths must both have shape [rows]")
        if slot_ids.dtype != torch.long or kv_lengths.dtype != torch.long:
            raise TypeError("slot_ids and kv_lengths must be torch.long")
        if slot_ids.device != kv_lengths.device:
            raise ValueError("slot_ids and kv_lengths must share a device")
        if slot_ids.numel() == 0:
            raise ValueError("paged decode requires at least one active row")

        pages = self.pages_per_lane
        kv_starts = self.kv_starts.index_select(0, slot_ids)
        partial_count, partial_indices, full_count, full_indices = kv_range_blocks(
            kv_starts,
            kv_lengths,
            block_size=self.page_size,
            blocks_per_row=pages,
            max_block=self.total_pages - 1,
            page_table=self.page_table.index_select(0, slot_ids),
        )
        page_size = self.page_size
        page_home = self.page_home
        # Compare in pool-global logical addresses so one range test decides
        # both questions the mask must answer: is this key inside the row's
        # live window, and does the row own it at all. Under the identity
        # mapping ``page_home`` is ``arange`` and this collapses to the
        # contiguous ``slot * rounded_length + position`` arithmetic it
        # replaces. Ownership cannot be left to the block table alone --
        # eager ``flex_attention`` never reads the table, so a mask that
        # admitted foreign pages would make every eager parity test agree
        # with a decode the Triton kernel does not perform.
        row_base = slot_ids * self.rounded_length
        lower = row_base + kv_starts
        upper = row_base + kv_lengths

        def live_token(
            batch: Tensor, head: Tensor, query: Tensor, physical_kv: Tensor
        ) -> Tensor:
            del head, query
            physical_page = torch.div(
                physical_kv, page_size, rounding_mode="floor"
            )
            logical_kv = (
                page_home[physical_page] * page_size + physical_kv % page_size
            )
            return (logical_kv >= lower[batch]) & (logical_kv < upper[batch])

        return BlockMask.from_kv_blocks(
            partial_count,
            partial_indices,
            full_count,
            full_indices,
            # FlexDecoding uses a minimum BLOCK_M of 16 on CUDA and requires
            # the sparse Q tile to be divisible by it. The logical query is
            # still length one; a 16-row metadata tile simply covers it.
            BLOCK_SIZE=(16, self.page_size),
            mask_mod=live_token,
            seq_lengths=(1, self.total_pages * self.page_size),
            compute_q_blocks=False,
        )


class LatentThoughtModel(nn.Module):
    """Backbone wrapper adding the combined token+thought input path.

    ``step`` consumes one already-embedded stream input (the combined
    embedding) and mirrors the backbone's block loop. The renderer consumes
    the combined input and the raw temporal belief.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        mlp_hidden: int | None = None,
        num_blocks: int = 1,
        gain_init: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        model_dim = backbone.tok_emb.embedding_dim
        self.thought_input_schema = THOUGHT_INPUT_SCHEMA
        self.combiner = CombinedEmbedding(
            model_dim,
            mlp_hidden=mlp_hidden,
            num_blocks=num_blocks,
            gain_init=gain_init,
        )

    def embed_tokens(self, token_ids: Tensor) -> Tensor:
        return self.backbone.embed_tokens(token_ids)

    def make_generation_cache(
        self,
        batch_size: int,
        max_length: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ):
        return self.backbone.make_generation_cache(
            batch_size, max_length, device, dtype=dtype
        )

    @staticmethod
    def renderer_features(input_latent: Tensor, belief: Tensor) -> Tensor:
        """Features for vocab rendering.

        The policy probe keeps its pretrained 2*model_dim input shape: the
        combined stream input beside the raw temporal belief.
        """
        return torch.cat((input_latent, belief), dim=-1)

    def policy_logits(self, input_ids: Tensor) -> Tensor:
        """Teacher-forced token-only vocab logits (no hidden injection)."""
        input_latent = self.embed_tokens(input_ids)
        belief = self.backbone.temporal_belief_from_token_latent(input_latent)
        return self.backbone.logits_from_features(
            self.renderer_features(input_latent, belief)
        )

    # Tokens rendered per chunk in the teacher-forced CE below. The full
    # [batch, seq, vocab] logits tensor (plus the softcap temporaries) does
    # not fit on one GPU for large vocabularies — at GPT-2's 50257 vocab a
    # 64x1024 eval batch needs tens of GiB — while the trunk activations
    # feeding it are small. 8192 tokens keeps each logits chunk under ~2 GiB.
    BPB_EVAL_CHUNK_TOKENS = 8192

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Teacher-forced CE used by the policy-renderer BPB guard.

        Renders logits in token chunks and reduces with a sum so the result
        is the exact token mean one-shot ``cross_entropy`` would return.
        """
        input_latent = self.embed_tokens(input_ids)
        belief = self.backbone.temporal_belief_from_token_latent(input_latent)
        features = self.renderer_features(input_latent, belief).flatten(0, 1)
        targets = target_ids.flatten()
        loss_sum = torch.zeros((), device=features.device, dtype=torch.float32)
        for start in range(0, targets.numel(), self.BPB_EVAL_CHUNK_TOKENS):
            stop = start + self.BPB_EVAL_CHUNK_TOKENS
            logits = self.backbone.logits_from_features(features[start:stop])
            loss_sum += F.cross_entropy(
                logits.float(), targets[start:stop], reduction="sum"
            )
        return loss_sum / targets.numel()

    def make_static_generation_cache(
        self,
        batch_size: int,
        cache_length: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ):
        """Preallocated caches for the fixed-shape (``key_mask``) step path.

        Zero-filled once — masked slots must stay finite or the full-cache
        SDPA turns their garbage scores into NaN — and marked as static
        addresses so a CUDA-graph capture of ``step`` may mutate them in
        place across replays.  Reuse one cache set across rollouts of the
        same shape: a fresh allocation forces a graph re-record.
        """
        caches = self.make_generation_cache(
            batch_size, cache_length, device, dtype=dtype
        )
        # Cache tuples are architecture-defined (RoPE stores K/V pairs, PoPE
        # stores k_real/k_imag/value triples) — treat them generically.
        for cache in caches:
            for tensor in cache:
                tensor.zero_()
                torch._dynamo.mark_static_address(tensor)
        return caches

    def make_paged_generation_cache(
        self,
        capacity: int,
        max_length: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
        *,
        page_size: int = 64,
    ) -> PagedGenerationCache:
        """Allocate fixed physical lanes for compile-stable continuous refill."""
        if capacity <= 0 or max_length <= 0 or page_size <= 0:
            raise ValueError("capacity, max_length, and page_size must be positive")
        if device.type == "cuda" and (
            page_size < 16 or page_size & (page_size - 1)
        ):
            raise ValueError(
                "CUDA FlexDecoding requires a power-of-two page_size >= 16"
            )
        rounded_length = (
            (max_length + page_size - 1) // page_size
        ) * page_size
        pages_per_lane = rounded_length // page_size
        # A scratch region beyond the pool, owned by nobody and wide enough to
        # give every row its own sink token. Padding rows write their throwaway
        # K/V here instead of through some free slot's page table, which is
        # only a safe target while a slot permanently owns its lane -- exactly
        # the assumption the page table exists to remove. Distinct sinks
        # matter: ``index_copy_`` with a repeated index is undefined.
        scratch_page = capacity * pages_per_lane
        scratch_pages = (capacity + page_size - 1) // page_size
        physical_length = (scratch_page + scratch_pages) * page_size
        templates = self.make_generation_cache(1, 1, device, dtype=dtype)
        layers: list[tuple[Tensor, ...]] = []
        for template in templates:
            if len(template) == 3:
                # PoPE's two D-wide complex components are concatenated once
                # on write, rather than over the full cache on every step.
                k_real, _, value = template
                key_shape = (
                    1,
                    k_real.size(1),
                    physical_length,
                    2 * k_real.size(3),
                )
                value_shape = (
                    1,
                    value.size(1),
                    physical_length,
                    value.size(3),
                )
                layer = (
                    torch.zeros(key_shape, device=device, dtype=k_real.dtype),
                    torch.zeros(value_shape, device=device, dtype=value.dtype),
                )
            elif len(template) == 2:
                key, value = template
                key_shape = (1, key.size(1), physical_length, key.size(3))
                value_shape = (
                    1,
                    value.size(1),
                    physical_length,
                    value.size(3),
                )
                layer = (
                    torch.zeros(key_shape, device=device, dtype=key.dtype),
                    torch.zeros(value_shape, device=device, dtype=value.dtype),
                )
            elif len(template) == 4:
                # Recurrent (KDA) layer: three conv windows plus the fp32
                # delta-rule state, all indexed by lane on dim 0 — no pages,
                # no addresses, no length axis. Rows [capacity, 2 * capacity)
                # are per-row scratch sinks, the recurrent analogue of
                # ``scratch_addresses``: a padding row's slot_id may name a
                # LIVE lane (the scheduler pads with slot 0), and unlike a
                # masked KV read the recurrent step writes its row
                # unconditionally, so dead rows must be redirected to rows
                # nobody owns. Distinct sinks matter for the same reason they
                # do for KV: ``index_copy_`` with a repeated index is
                # undefined.
                rows = 2 * capacity
                layer = tuple(
                    torch.zeros(
                        (rows, *tensor.shape[1:]),
                        device=device,
                        dtype=tensor.dtype,
                    )
                    for tensor in template
                )
            else:
                raise ValueError(
                    f"paged decode does not support {len(template)} cache tensors"
                )
            layers.append(layer)
        kv_starts = torch.zeros(capacity, dtype=torch.long, device=device)
        # The identity mapping: slot s owns the contiguous run of pages
        # [s * pages_per_lane, (s + 1) * pages_per_lane). This reproduces the
        # reserved-lane layout exactly, so nothing observable changes here; the
        # table is what lets a pool allocator hand out pages in any order.
        pool = torch.arange(scratch_page, device=device)
        page_home = torch.empty(
            scratch_page + scratch_pages, dtype=torch.long, device=device
        )
        page_home[:scratch_page] = pool
        # The unowned-page sentinel: at or past every row's upper bound, so
        # ``live_token`` rejects a scratch page by value even if a block table
        # ever named one.
        page_home[scratch_page:] = scratch_page
        return PagedGenerationCache(
            layers=layers,
            capacity=capacity,
            max_length=max_length,
            page_size=page_size,
            rounded_length=rounded_length,
            kv_starts=kv_starts,
            page_table=pool.view(capacity, pages_per_lane),
            page_home=page_home,
        )

    def paged_step_core(
        self,
        input_latent: Tensor,
        caches: list[tuple[Tensor, ...]],
        positions: Tensor,
        block_mask: BlockMask,
        cache_addresses: Tensor,
        lane_rows: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compiled fixed-capacity surface for independently advancing rows."""
        backbone = self.backbone
        x = input_latent
        skips: list[Tensor] = []
        for i in range(backbone.num_encoder_layers):
            x, _ = backbone._block_paged_step(
                backbone.blocks[i],
                x,
                input_latent,
                caches[i],
                positions,
                block_mask,
                cache_addresses,
                lane_rows,
            )
            skips.append(x)
        for j in range(backbone.num_decoder_layers):
            i = backbone.num_encoder_layers + j
            if skips:
                x = x + backbone.skip_weights[j].to(x.dtype)[
                    None, None
                ] * skips.pop()
            x, _ = backbone._block_paged_step(
                backbone.blocks[i],
                x,
                input_latent,
                caches[i],
                positions,
                block_mask,
                cache_addresses,
                lane_rows,
            )
        belief = backbone.final_norm(x)
        logits = backbone.logits_from_features(
            self.renderer_features(input_latent, belief)
        ).squeeze(1)
        return belief.squeeze(1), logits

    def paged_step(
        self,
        input_latent: Tensor,
        cache: PagedGenerationCache,
        *,
        slot_ids: Tensor,
        positions: Tensor,
        live: Tensor | None = None,
    ) -> StepOutput:
        """Advance arbitrary physical lanes at independent logical positions.

        ``live`` marks which rows carry real work. A false row reads no KV --
        an empty range costs zero blocks, so padding up to a static batch
        bucket is free in time -- and writes its K/V to a scratch page no slot
        owns. Both halves matter: routing padding through a currently-free
        slot's page table is only safe while that slot permanently owns a
        lane, and a shared pool hands those pages to whoever needs them next.
        """
        if input_latent.shape[:2] != (slot_ids.numel(), 1):
            raise ValueError("input_latent must have shape [rows, 1, model_dim]")
        if positions.shape != slot_ids.shape:
            raise ValueError("positions and slot_ids must have the same shape")
        addresses = cache.token_addresses(slot_ids, positions)
        kv_lengths = positions + 1
        lane_rows = slot_ids
        if live is not None:
            if live.shape != slot_ids.shape:
                raise ValueError("live and slot_ids must have the same shape")
            kv_lengths = kv_lengths * live
            addresses = torch.where(
                live, addresses, cache.scratch_addresses[: live.numel()]
            )
            # The recurrent analogue of the scratch-address redirect above:
            # a dead row's slot_id may name a live lane, and a recurrent
            # layer's step writes its row unconditionally, so dead rows are
            # sent to the per-row scratch lanes in [capacity, 2 * capacity).
            lane_rows = torch.where(
                live,
                slot_ids,
                cache.capacity
                + torch.arange(live.numel(), device=live.device),
            )
        block_mask = cache.block_mask(slot_ids, kv_lengths)
        belief, logits = self.paged_step_core(
            input_latent,
            cache.layers,
            positions,
            block_mask,
            addresses,
            lane_rows,
        )
        return StepOutput(
            belief=belief,
            input_latent=input_latent.squeeze(1),
            logits=logits,
            caches=cache.layers,
        )

    def token_paged_step(
        self,
        token_ids: Tensor,
        cache: PagedGenerationCache,
        *,
        slot_ids: Tensor,
        positions: Tensor,
        live: Tensor | None = None,
    ) -> StepOutput:
        return self.paged_step(
            self.embed_tokens(token_ids[:, None]),
            cache,
            slot_ids=slot_ids,
            positions=positions,
            live=live,
        )

    @staticmethod
    def _scatter_paged_prefix(
        destination: Tensor,
        source: Tensor,
        slot_ids: Tensor,
        addresses: Tensor,
    ) -> None:
        """Repeat each source row across its destination slots and scatter."""
        groups, repeats = slot_ids.shape
        length = source.size(2)
        repeated = source.repeat_interleave(repeats, dim=0).to(
            destination.dtype
        )
        packed = repeated.permute(1, 0, 2, 3).reshape(
            1, source.size(1), groups * repeats * length, source.size(3)
        )
        destination.index_copy_(2, addresses.flatten(), packed)

    @torch.no_grad()
    def build_prompt_prefix_bank(
        self,
        unique_prompt_ids: Tensor,
        unique_prompt_lengths: Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> PromptPrefixBank:
        """Prefill every unique pool prompt together and retain its full state.

        Prompt IDs are right-aligned and all unique prompts traverse the trunk
        exactly once. The retained padded KV preserves absolute RoPE/PoPE
        positions; ``kv_starts`` records the lower attention bound per prompt.
        """
        groups, padded_length = unique_prompt_ids.shape
        if unique_prompt_lengths.shape != (groups,):
            raise ValueError("unique_prompt_lengths must have shape [groups]")
        if unique_prompt_lengths.dtype != torch.long:
            raise TypeError("unique_prompt_lengths must be torch.long")
        if unique_prompt_lengths.device != unique_prompt_ids.device:
            raise ValueError("prompt IDs and lengths must share a device")
        if bool(
            (
                (unique_prompt_lengths <= 0)
                | (unique_prompt_lengths > padded_length)
            ).any()
        ):
            raise ValueError("prompt lengths fall outside the padded prompt width")

        key_valid = torch.arange(
            padded_length, device=unique_prompt_ids.device
        )[None] >= (padded_length - unique_prompt_lengths)[:, None]
        dense_caches = self.make_generation_cache(
            groups,
            padded_length,
            unique_prompt_ids.device,
            dtype=dtype,
        )
        dense_output = self.prefill(
            unique_prompt_ids, dense_caches, key_valid
        )
        return PromptPrefixBank(
            layers=dense_caches,
            output=dense_output,
            kv_starts=padded_length - unique_prompt_lengths,
            prompt_width=padded_length,
        )

    @torch.no_grad()
    def admit_prompt_prefixes(
        self,
        bank: PromptPrefixBank,
        group_indices: Tensor,
        group_slot_ids: Tensor,
        cache: PagedGenerationCache,
    ) -> StepOutput:
        """Fan selected bank rows into destination slots without trunk work."""
        if group_indices.ndim != 1 or group_indices.dtype != torch.long:
            raise TypeError("group_indices must be a one-dimensional LongTensor")
        groups = group_indices.numel()
        if groups == 0:
            raise ValueError("admission requires at least one prompt group")
        if group_slot_ids.ndim != 2 or group_slot_ids.size(0) != groups:
            raise ValueError("group_slot_ids must have shape [groups, repeats]")
        if group_slot_ids.dtype != torch.long:
            raise TypeError("group_slot_ids must be torch.long")
        if (
            group_indices.device != bank.kv_starts.device
            or group_slot_ids.device != bank.kv_starts.device
            or cache.kv_starts.device != bank.kv_starts.device
        ):
            raise ValueError("bank, group indices, slot IDs, and cache must share a device")
        if bank.prompt_width > cache.max_length:
            raise ValueError("prompt bank width exceeds paged-cache max_length")
        flat_slots = group_slot_ids.flatten()
        # The production scheduler creates both tensors from validated host
        # request/slot lists. Rechecking their values on CUDA would add two
        # stream barriers to every refill. Keep the defensive content checks
        # on CPU-facing/library calls, while shape/device checks above remain
        # unconditional and sync-free.
        if group_indices.device.type != "cuda":
            if bool(
                (
                    (group_indices < 0)
                    | (group_indices >= bank.num_prompts)
                ).any()
            ):
                raise ValueError("group_indices fall outside the prompt bank")
            if bool(
                ((flat_slots < 0) | (flat_slots >= cache.capacity)).any()
            ):
                raise ValueError(
                    "group_slot_ids fall outside paged-cache capacity"
                )
            if flat_slots.unique().numel() != flat_slots.numel():
                raise ValueError(
                    "each destination slot must appear exactly once"
                )

        # One address computation for every KV layer: the prompt window is
        # the same logical [0, length) span in every one, so its physical
        # scatter is too. Recurrent layers never touch it — their whole
        # decode continuation is one row per lane. ``bank.prompt_width``
        # rather than a layer-0 axis: in a hybrid trunk layer 0 may be
        # recurrent, whose conv window has no KV-length axis.
        prefix_addresses = cache.token_addresses(
            group_slot_ids.flatten()[:, None],
            torch.arange(
                bank.prompt_width, device=group_slot_ids.device
            )[None],
        ).flatten()
        if group_indices.device.type != "cuda":
            # Distinct slots no longer imply distinct addresses: that held only
            # while a slot permanently owned one contiguous lane. What the
            # scatter below actually needs is duplicate-free indices --
            # ``index_copy_`` with a repeated index is undefined behaviour, and
            # an allocator that handed one page to two slots would corrupt the
            # cache nondeterministically while ``page_home`` still named a
            # single plausible owner, so no mask or parity check would notice.
            if prefix_addresses.unique().numel() != prefix_addresses.numel():
                raise ValueError(
                    "destination slots map to overlapping physical pages"
                )
        for bank_layer, paged_layer in zip(
            bank.layers, cache.layers, strict=True
        ):
            dense_layer = tuple(
                tensor.index_select(0, group_indices)
                for tensor in bank_layer
            )
            if len(dense_layer) == 4:
                # Recurrent layer: fan the selected groups' conv windows and
                # delta-rule state across their destination lanes. The copy
                # overwrites each admitted lane's rows completely, which is
                # also what makes lane recycling sound here — a recurrent
                # layer has no kv_starts window to hide a previous
                # occupant's stale state behind.
                for arena, source in zip(
                    paged_layer, dense_layer, strict=True
                ):
                    arena.index_copy_(
                        0,
                        flat_slots,
                        source.repeat_interleave(
                            group_slot_ids.size(1), dim=0
                        ).to(arena.dtype),
                    )
                continue
            if len(dense_layer) == 3:
                dense_key = torch.cat(
                    (dense_layer[0], dense_layer[1]), dim=-1
                )
                dense_value = dense_layer[2]
            else:
                dense_key, dense_value = dense_layer
            self._scatter_paged_prefix(
                paged_layer[0],
                dense_key,
                group_slot_ids,
                prefix_addresses,
            )
            self._scatter_paged_prefix(
                paged_layer[1],
                dense_value,
                group_slot_ids,
                prefix_addresses,
            )
        repeats = group_slot_ids.size(1)
        starts = bank.kv_starts.index_select(
            0, group_indices
        ).repeat_interleave(repeats)
        cache.kv_starts.index_copy_(0, flat_slots, starts)

        def select_state(value: Tensor) -> Tensor:
            return value.index_select(0, group_indices).repeat_interleave(
                repeats, dim=0
            )

        return StepOutput(
            belief=select_state(bank.output.belief),
            input_latent=select_state(bank.output.input_latent),
            logits=select_state(bank.output.logits),
            caches=cache.layers,
        )

    @torch.no_grad()
    def prefill_into_paged_slots(
        self,
        unique_prompt_ids: Tensor,
        unique_prompt_lengths: Tensor,
        group_slot_ids: Tensor,
        cache: PagedGenerationCache,
    ) -> StepOutput:
        """Compatibility composition of pool prefill and immediate admission."""
        bank = self.build_prompt_prefix_bank(
            unique_prompt_ids,
            unique_prompt_lengths,
            dtype=cache.layers[0][0].dtype,
        )
        return self.admit_prompt_prefixes(
            bank,
            torch.arange(
                unique_prompt_ids.size(0),
                dtype=torch.long,
                device=unique_prompt_ids.device,
            ),
            group_slot_ids,
            cache,
        )

    def step_core(
        self,
        input_latent: Tensor,
        caches: list[tuple[Tensor, ...]],
        position: int | Tensor,
        key_mask: Tensor | None = None,
        block_mask: BlockMask | None = None,
    ) -> tuple[Tensor, Tensor]:
        """The compiled surface: belief and logits.

        Caches are mutated strictly in place.

        ``key_mask`` selects the static full-cache attention path (see
        ``_attention_step``): shapes stay constant across positions, which is
        what makes this method capturable as a single CUDA graph.  No
        dataclass construction and no cache aliasing cross the compile
        boundary — the trainer patches THIS method with torch.compile.
        ``block_mask`` (see ``DecodeRangeMask``) expresses the same constant
        shapes as a flex-decoding block table instead, which is what keeps the
        step off the memory-efficient SDPA kernel. One mask serves every layer.
        """
        backbone = self.backbone
        x = input_latent
        skips: list[Tensor] = []
        for i in range(backbone.num_encoder_layers):
            # Every _attention_step variant mutates the cache tensors in
            # place and returns the same objects; the returned handle is
            # deliberately dropped so mutated inputs never alias outputs
            # inside a CUDA-graph capture.
            x, _ = backbone._block_step(
                backbone.blocks[i],
                x,
                input_latent,
                caches[i],
                position,
                key_mask,
                block_mask,
            )
            skips.append(x)
        for j in range(backbone.num_decoder_layers):
            i = backbone.num_encoder_layers + j
            if skips:
                x = x + backbone.skip_weights[j].to(x.dtype)[None, None] * skips.pop()
            x, _ = backbone._block_step(
                backbone.blocks[i],
                x,
                input_latent,
                caches[i],
                position,
                key_mask,
                block_mask,
            )
        belief = backbone.final_norm(x)
        features = self.renderer_features(input_latent, belief)
        logits = backbone.logits_from_features(features).squeeze(1)
        return belief.squeeze(1), logits

    def prefill_core(
        self,
        input_latent: Tensor,
        caches: list[tuple[Tensor, ...]],
        key_valid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Densely ingest a deterministic prefix and expose its final state."""
        belief = self.backbone.prefill_belief(
            input_latent, caches, key_valid
        )[:, -1:]
        final_input = input_latent[:, -1:]
        logits = self.backbone.logits_from_features(
            self.renderer_features(final_input, belief)
        ).squeeze(1)
        return belief.squeeze(1), logits

    def prefill(
        self,
        token_ids: Tensor,
        caches: list[tuple[Tensor, ...]],
        key_valid: Tensor | None = None,
    ) -> StepOutput:
        """Populate prefix caches without running policy heads per token."""
        input_latent = self.embed_tokens(token_ids)
        belief, logits = self.prefill_core(input_latent, caches, key_valid)
        return StepOutput(
            belief=belief,
            input_latent=input_latent[:, -1],
            logits=logits,
            caches=list(caches),
        )

    def step(
        self,
        input_latent: Tensor,
        caches: list[tuple[Tensor, ...]],
        position: int | Tensor,
        key_mask: Tensor | None = None,
        block_mask: BlockMask | None = None,
    ) -> StepOutput:
        """Advance one stream position from an embedded input."""
        belief, logits = self.step_core(
            input_latent, caches, position, key_mask, block_mask
        )
        return StepOutput(
            belief=belief,
            input_latent=input_latent.squeeze(1),
            logits=logits,
            caches=list(caches),
        )

    def token_step(
        self,
        token_ids: Tensor,
        caches: list[tuple[Tensor, ...]],
        position: int | Tensor,
        key_mask: Tensor | None = None,
        block_mask: BlockMask | None = None,
    ) -> StepOutput:
        return self.step(
            self.embed_tokens(token_ids[:, None]),
            caches,
            position,
            key_mask,
            block_mask,
        )

    def combined_input(self, token_ids: Tensor, hidden: Tensor) -> Tensor:
        """The decode-side combined embedding for one generated token per row.

        Every decode input past the first generation step feeds back a token
        the model generated, so the hasThought flag is implicitly all-ones
        here; pinned modes bypass this method entirely. The injection branch
        runs in fp32 and is rounded to the base dtype inside the combiner —
        the same op order as ``assemble_stream_latents`` — so rollout and
        replay agree exactly.
        """
        base = self.embed_tokens(token_ids[:, None])
        return self.combiner(base, hidden[:, None])

    def new_parameters(self):
        """Post-training parameters that do not exist in the pretrained checkpoint."""
        yield from self.combiner.parameters()

    def load_backbone_checkpoint(self, state: dict[str, Tensor]) -> None:
        """Strict backbone load: every checkpoint key must land in the backbone."""
        self.backbone.load_state_dict(state, strict=True)
