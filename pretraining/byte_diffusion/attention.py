"""Production attention dispatch for clean and branched byte-diffusion paths.

Clean documents use native variable-length Flash attention on CUDA.  Canvas
and introspection queries use FlexAttention over one physical K/V bank whose
clean prefix is shared by every branch.  Dense implementations are small-
tensor correctness oracles only and must be opted into explicitly.

RoPE is deliberately outside this module.  :class:`PackedCleanQKV` carries the
absolute positions used to rotate Q/K before attention so packing can never
silently replace semantic positions with physical offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor

from .kernels import (
    AttentionBackend,
    AttentionBackendCapabilities,
    AttentionPattern,
    DEFAULT_DENSE_REFERENCE_LIMIT,
    detect_attention_backend_capabilities,
    select_attention_backend,
)


CleanBackend = Literal["auto", "varlen_flash", "dense_reference"]
BranchBackend = Literal["auto", "flex", "dense_reference"]


def _require_same_device(name: str, tensors: tuple[Tensor, ...]) -> None:
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError(f"{name} tensors must share a device, got {devices}")


def _require_bool(name: str, tensor: Tensor, rank: int) -> None:
    if tensor.dtype != torch.bool or tensor.ndim != rank:
        raise ValueError(f"{name} must be a rank-{rank} boolean tensor")


@dataclass(frozen=True)
class PackedCleanQKV:
    """Packed document-local self-attention inputs.

    Q/K/V use Flash's ``[total_tokens, heads, head_dim]`` layout.  Sequence
    boundaries are CUDA-style int32 cumulative offsets of shape ``[N + 1]``.
    ``absolute_positions`` is int64 ``[total_tokens]`` and records the positions
    that callers must use for RoPE before invoking attention; the attention
    implementation itself does not rotate Q/K.

    A packed batch may contain multiple documents from one original row.  PAD
    positions must not be present at all.  Q may have more heads than K/V for
    GQA, but the head dimension and token count must agree.
    """

    query: Tensor
    key: Tensor
    value: Tensor
    cu_seqlens: Tensor
    absolute_positions: Tensor
    max_seqlen: int

    def __post_init__(self) -> None:
        if self.query.ndim != 3 or self.key.ndim != 3 or self.value.ndim != 3:
            raise ValueError("packed Q/K/V must have shape [tokens, heads, head_dim]")
        total = self.query.shape[0]
        if self.key.shape[0] != total or self.value.shape[0] != total:
            raise ValueError("packed Q/K/V token counts differ")
        if self.key.shape != self.value.shape:
            raise ValueError("packed K/V shapes must match")
        if self.query.shape[-1] != self.key.shape[-1]:
            raise ValueError("packed Q/K head dimensions differ")
        if self.query.shape[1] % self.key.shape[1]:
            raise ValueError("query heads must be divisible by key/value heads")
        if self.query.dtype != self.key.dtype or self.query.dtype != self.value.dtype:
            raise ValueError("packed Q/K/V dtypes must match")
        if not self.query.is_floating_point():
            raise TypeError("packed Q/K/V must be floating point")
        if self.cu_seqlens.ndim != 1 or self.cu_seqlens.dtype != torch.int32:
            raise ValueError("cu_seqlens must be a rank-1 int32 tensor")
        if self.cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must contain at least one sequence")
        if (
            self.absolute_positions.shape != (total,)
            or self.absolute_positions.dtype != torch.int64
        ):
            raise ValueError("absolute_positions must be int64 [total_tokens]")
        if self.max_seqlen <= 0:
            raise ValueError("max_seqlen must be positive")
        _require_same_device(
            "packed attention",
            (
                self.query,
                self.key,
                self.value,
                self.cu_seqlens,
                self.absolute_positions,
            ),
        )

    @property
    def total_tokens(self) -> int:
        return self.query.shape[0]

    def validate_boundaries(self) -> None:
        """Perform value-level boundary validation outside production hot paths."""

        offsets = self.cu_seqlens.detach().cpu().tolist()
        if offsets[0] != 0 or offsets[-1] != self.total_tokens:
            raise ValueError("cu_seqlens must start at zero and end at total_tokens")
        lengths = [stop - start for start, stop in zip(offsets, offsets[1:])]
        if any(length < 0 for length in lengths):
            raise ValueError("packed sequences must be nonoverlapping")
        if not any(length > 0 for length in lengths):
            raise ValueError("packed attention requires at least one nonempty sequence")
        first_empty = next(
            (index for index, length in enumerate(lengths) if length == 0), None
        )
        if first_empty is not None and any(length > 0 for length in lengths[first_empty:]):
            raise ValueError("empty packed sequence capacity must form a tail")
        if max(lengths) > self.max_seqlen:
            raise ValueError("max_seqlen is smaller than a packed sequence")


@dataclass(frozen=True)
class CanvasBranchLayout:
    """Canvas branch geometry for branch-query to shared-bank attention."""

    clean_valid: Tensor
    branch_valid: Tensor
    prefix_lengths: Tensor
    prefix_window: int | None = None
    clean_positions: Tensor | None = None
    branch_positions: Tensor | None = None
    clean_segment_ids: Tensor | None = None
    branch_segment_ids: Tensor | None = None

    def __post_init__(self) -> None:
        _require_bool("clean_valid", self.clean_valid, 2)
        _require_bool("branch_valid", self.branch_valid, 3)
        batch, branches, canvas = self.branch_valid.shape
        if self.clean_valid.shape[0] != batch:
            raise ValueError("clean and branch batch sizes differ")
        if self.prefix_lengths.shape != (batch, branches):
            raise ValueError("prefix_lengths must have shape [batch, branches]")
        if self.prefix_lengths.dtype != torch.int64:
            raise ValueError("prefix_lengths must be int64")
        tensors = (self.clean_valid, self.branch_valid, self.prefix_lengths)
        if (self.clean_segment_ids is None) != (self.branch_segment_ids is None):
            raise ValueError("clean and branch segment ids must be provided together")
        if self.clean_segment_ids is not None:
            assert self.branch_segment_ids is not None
            if (
                self.clean_segment_ids.shape != self.clean_valid.shape
                or self.clean_segment_ids.dtype != torch.int64
            ):
                raise ValueError(
                    "clean_segment_ids must be int64 [batch, clean_length]"
                )
            if (
                self.branch_segment_ids.shape != (batch, branches)
                or self.branch_segment_ids.dtype != torch.int64
            ):
                raise ValueError("branch_segment_ids must be int64 [batch, branches]")
            tensors += (self.clean_segment_ids, self.branch_segment_ids)
        if (self.clean_positions is None) != (self.branch_positions is None):
            raise ValueError("clean and branch positions must be provided together")
        if self.clean_positions is not None:
            assert self.branch_positions is not None
            if (
                self.clean_positions.shape != self.clean_valid.shape
                or self.clean_positions.dtype != torch.int64
            ):
                raise ValueError("clean_positions must be int64 [batch, clean_length]")
            if (
                self.branch_positions.shape != self.branch_valid.shape
                or self.branch_positions.dtype != torch.int64
            ):
                raise ValueError(
                    "branch_positions must be int64 [batch, branches, canvas_length]"
                )
            tensors += (self.clean_positions, self.branch_positions)
        if self.prefix_window is not None:
            if self.prefix_window <= 0:
                raise ValueError("prefix_window must be positive")
            if self.clean_positions is None:
                raise ValueError("windowed branches require clean and branch positions")
        _require_same_device("canvas layout", tensors)
        if canvas <= 0 or branches <= 0:
            raise ValueError("canvas branches and length must be positive")

    @property
    def batch_size(self) -> int:
        return self.clean_valid.shape[0]

    @property
    def clean_length(self) -> int:
        return self.clean_valid.shape[1]

    @property
    def branches(self) -> int:
        return self.branch_valid.shape[1]

    @property
    def branch_length(self) -> int:
        return self.branch_valid.shape[2]

    @property
    def query_length(self) -> int:
        return self.branches * self.branch_length

    @property
    def kv_length(self) -> int:
        return self.clean_length + self.query_length

    def validate_prefix_lengths(self) -> None:
        """Validate sampled prefix boundaries outside production hot paths."""

        prefix_lengths = self.prefix_lengths.detach().cpu()
        if bool(((prefix_lengths < 0) | (prefix_lengths > self.clean_length)).any()):
            raise ValueError("a branch prefix length is outside the clean bank")


@dataclass(frozen=True)
class IntrospectionBranchLayout:
    """I-DLM proposal-copy geometry over a shared clean/proposal K/V bank."""

    clean_valid: Tensor
    proposal_valid: Tensor
    block_size: int
    clean_segment_ids: Tensor | None = None
    proposal_segment_ids: Tensor | None = None

    def __post_init__(self) -> None:
        _require_bool("clean_valid", self.clean_valid, 2)
        _require_bool("proposal_valid", self.proposal_valid, 2)
        if self.clean_valid.shape != self.proposal_valid.shape:
            raise ValueError("clean and proposal validity shapes differ")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        tensors = (self.clean_valid, self.proposal_valid)
        if (self.clean_segment_ids is None) != (self.proposal_segment_ids is None):
            raise ValueError("clean and proposal segment ids must be provided together")
        if self.clean_segment_ids is not None:
            assert self.proposal_segment_ids is not None
            if (
                self.clean_segment_ids.shape != self.clean_valid.shape
                or self.clean_segment_ids.dtype != torch.int64
            ):
                raise ValueError(
                    "clean_segment_ids must be int64 [batch, sequence_length]"
                )
            if (
                self.proposal_segment_ids.shape != self.proposal_valid.shape
                or self.proposal_segment_ids.dtype != torch.int64
            ):
                raise ValueError(
                    "proposal_segment_ids must be int64 [batch, sequence_length]"
                )
            tensors += (self.clean_segment_ids, self.proposal_segment_ids)
        _require_same_device("introspection layout", tensors)

    @property
    def batch_size(self) -> int:
        return self.clean_valid.shape[0]

    @property
    def clean_length(self) -> int:
        return self.clean_valid.shape[1]

    @property
    def query_length(self) -> int:
        return self.proposal_valid.shape[1]

    @property
    def kv_length(self) -> int:
        return self.clean_length + self.query_length


BranchLayout = CanvasBranchLayout | IntrospectionBranchLayout


@dataclass(frozen=True)
class CanvasBlockMaskMetadata:
    """Exact sparse block topology independent of attention values.

    Partial blocks retain :func:`_canvas_mask_mod`; full blocks are proven to
    satisfy that rule for every token and let Flex skip token-mask work.  The
    proof includes document, PAD, prefix-window, branch-isolation, and padded
    physical-block boundaries.
    Keeping this topology as O(query_blocks * prefix_blocks) integer metadata
    avoids materializing a token mask and lets CPU collation prepare it before
    the fullgraph CUDA forward.
    """

    kv_num_blocks: Tensor
    kv_indices: Tensor
    q_num_blocks: Tensor
    q_indices: Tensor
    block_size: tuple[int, int]
    query_length: int
    kv_length: int
    full_kv_num_blocks: Tensor | None = None
    full_kv_indices: Tensor | None = None
    full_q_num_blocks: Tensor | None = None
    full_q_indices: Tensor | None = None

    def __post_init__(self) -> None:
        if self.kv_num_blocks.dtype != torch.int32 or self.q_num_blocks.dtype != torch.int32:
            raise ValueError("canvas block counts must be int32")
        if self.kv_indices.dtype != torch.int32 or self.q_indices.dtype != torch.int32:
            raise ValueError("canvas block indices must be int32")
        if self.kv_num_blocks.ndim != 3 or self.kv_indices.ndim != 4:
            raise ValueError("canvas block metadata must have [B,H,Q(,K)] shape")
        if self.kv_num_blocks.shape != self.kv_indices.shape[:-1]:
            raise ValueError("canvas block counts and indices do not align")
        if self.q_num_blocks.shape != self.q_indices.shape[:-1]:
            raise ValueError("transposed canvas block counts and indices do not align")
        if self.kv_num_blocks.shape[1] != 1:
            raise ValueError("canvas block metadata must be shared across heads")
        full_values = (
            self.full_kv_num_blocks,
            self.full_kv_indices,
            self.full_q_num_blocks,
            self.full_q_indices,
        )
        if any(value is None for value in full_values) and not all(
            value is None for value in full_values
        ):
            raise ValueError("full canvas block metadata must be provided together")
        if self.full_kv_num_blocks is not None:
            assert self.full_kv_indices is not None
            assert self.full_q_num_blocks is not None
            assert self.full_q_indices is not None
            if (
                self.full_kv_num_blocks.dtype != torch.int32
                or self.full_kv_indices.dtype != torch.int32
                or self.full_q_num_blocks.dtype != torch.int32
                or self.full_q_indices.dtype != torch.int32
            ):
                raise ValueError("full canvas block metadata must be int32")
            if (
                self.full_kv_num_blocks.shape != self.kv_num_blocks.shape
                or self.full_kv_indices.shape != self.kv_indices.shape
                or self.full_q_num_blocks.shape != self.q_num_blocks.shape
                or self.full_q_indices.shape != self.q_indices.shape
            ):
                raise ValueError("full and partial canvas block metadata must align")
        if any(size <= 0 for size in self.block_size):
            raise ValueError("canvas block sizes must be positive")
        if self.query_length <= 0 or self.kv_length <= 0:
            raise ValueError("canvas sequence lengths must be positive")
        expected_query_blocks = math.ceil(self.query_length / self.block_size[0])
        expected_kv_blocks = math.ceil(self.kv_length / self.block_size[1])
        if self.kv_indices.shape[-2:] != (
            expected_query_blocks,
            expected_kv_blocks,
        ):
            raise ValueError(
                "canvas block metadata capacity must cover every physical "
                "query and K/V block"
            )
        if self.q_indices.shape[-2:] != (
            expected_kv_blocks,
            expected_query_blocks,
        ):
            raise ValueError("transposed canvas block capacity is incomplete")
        if self.kv_num_blocks.device.type == "cpu":
            if bool(
                ((self.kv_num_blocks < 0) | (self.kv_num_blocks > expected_kv_blocks)).any()
            ):
                raise ValueError("canvas block counts exceed K/V block capacity")
            slots = torch.arange(expected_kv_blocks)[None, None, None, :]
            populated = slots < self.kv_num_blocks[..., None]
            populated_indices = self.kv_indices[populated]
            if bool(
                (
                    (populated_indices < 0)
                    | (populated_indices >= expected_kv_blocks)
                ).any()
            ):
                raise ValueError("canvas block index lies outside K/V capacity")
            if self.full_kv_num_blocks is not None:
                assert self.full_kv_indices is not None
                if bool(
                    (
                        (self.full_kv_num_blocks < 0)
                        | (self.full_kv_num_blocks > expected_kv_blocks)
                    ).any()
                ):
                    raise ValueError("full canvas block counts exceed K/V capacity")
                full_populated = slots < self.full_kv_num_blocks[..., None]
                full_indices = self.full_kv_indices[full_populated]
                if bool(
                    ((full_indices < 0) | (full_indices >= expected_kv_blocks)).any()
                ):
                    raise ValueError("full canvas block index lies outside K/V capacity")

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "CanvasBlockMaskMetadata":
        return CanvasBlockMaskMetadata(
            self.kv_num_blocks.to(device, non_blocking=non_blocking),
            self.kv_indices.to(device, non_blocking=non_blocking),
            self.q_num_blocks.to(device, non_blocking=non_blocking),
            self.q_indices.to(device, non_blocking=non_blocking),
            self.block_size,
            self.query_length,
            self.kv_length,
            None
            if self.full_kv_num_blocks is None
            else self.full_kv_num_blocks.to(device, non_blocking=non_blocking),
            None
            if self.full_kv_indices is None
            else self.full_kv_indices.to(device, non_blocking=non_blocking),
            None
            if self.full_q_num_blocks is None
            else self.full_q_num_blocks.to(device, non_blocking=non_blocking),
            None
            if self.full_q_indices is None
            else self.full_q_indices.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "CanvasBlockMaskMetadata":
        return CanvasBlockMaskMetadata(
            self.kv_num_blocks.pin_memory(),
            self.kv_indices.pin_memory(),
            self.q_num_blocks.pin_memory(),
            self.q_indices.pin_memory(),
            self.block_size,
            self.query_length,
            self.kv_length,
            None
            if self.full_kv_num_blocks is None
            else self.full_kv_num_blocks.pin_memory(),
            None if self.full_kv_indices is None else self.full_kv_indices.pin_memory(),
            None
            if self.full_q_num_blocks is None
            else self.full_q_num_blocks.pin_memory(),
            None if self.full_q_indices is None else self.full_q_indices.pin_memory(),
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        self.kv_num_blocks.record_stream(stream)
        self.kv_indices.record_stream(stream)
        self.q_num_blocks.record_stream(stream)
        self.q_indices.record_stream(stream)
        for value in (
            self.full_kv_num_blocks,
            self.full_kv_indices,
            self.full_q_num_blocks,
            self.full_q_indices,
        ):
            if value is not None:
                value.record_stream(stream)


def canvas_block_mask_metadata(
    layout: CanvasBranchLayout,
    *,
    block_size: int | tuple[int, int] = 128,
) -> CanvasBlockMaskMetadata:
    """Build scalable exact partial/full block lists for canvas Flex attention.

    This setup routine intentionally runs on CPU.  It inspects one scalar per
    token along each axis, never a ``query_length * kv_length`` token grid.  A
    query tile receives the union of candidate clean-prefix and own-branch
    blocks.  Axis-wise reductions then prove which candidates are wholly
    allowed; all boundary or mixed candidates retain the exact token mask.
    """

    if layout.clean_valid.device.type != "cpu":
        raise ValueError("canvas block metadata must be constructed from CPU layout")
    q_block, kv_block = (
        (block_size, block_size) if isinstance(block_size, int) else block_size
    )
    if q_block <= 0 or kv_block <= 0:
        raise ValueError("canvas block sizes must be positive")
    query_blocks = math.ceil(layout.query_length / q_block)
    # ``BlockMask.from_kv_blocks`` uses the final ``kv_indices`` dimension as
    # the number of physical K/V block columns when it transposes the sparse
    # topology for backward.  It is therefore a dense *capacity* dimension,
    # not merely the maximum number of populated entries in one query row.
    # Allocating only ``max(kv_num_blocks)`` makes otherwise-valid high block
    # indices out of bounds on real 8K rows (the small oracle layouts did not
    # expose this).
    kv_blocks = math.ceil(layout.kv_length / kv_block)
    # Construct one conservative candidate bitset per branch entirely with
    # broadcast tensor arithmetic.  The previous nested Python/set loop made
    # ~64K set updates per optimizer update at B4/128 and scaled especially
    # poorly toward exhaustive origins.
    block_starts = torch.arange(kv_blocks, dtype=torch.long) * kv_block
    block_stops = block_starts + kv_block
    prefix_stop = layout.prefix_lengths
    prefix_start = (
        torch.zeros_like(prefix_stop)
        if layout.prefix_window is None
        else (prefix_stop - layout.prefix_window).clamp_min(0)
    )
    if layout.branch_positions is not None:
        # Document offsets reset at every packed-document boundary. Convert
        # each origin's document-relative position back to its physical start
        # and exclude all earlier documents before creating candidate tiles.
        document_start = prefix_stop - layout.branch_positions[..., 0]
        prefix_start = torch.maximum(prefix_start, document_start)
    prefix_candidates = (
        (block_stops[None, None, :] > prefix_start[..., None])
        & (block_starts[None, None, :] < prefix_stop[..., None])
    )
    branch_numbers = torch.arange(layout.branches, dtype=torch.long)
    own_start = layout.clean_length + branch_numbers * layout.branch_length
    own_stop = own_start + layout.branch_length
    own_candidates = (
        (block_stops[None, :] > own_start[:, None])
        & (block_starts[None, :] < own_stop[:, None])
    )
    per_branch = prefix_candidates | own_candidates[None]
    per_branch &= layout.branch_valid.any(-1)[..., None]

    if q_block >= layout.branch_length and q_block % layout.branch_length == 0:
        branches_per_query_block = q_block // layout.branch_length
        padded_branches = query_blocks * branches_per_query_block
        if padded_branches != layout.branches:
            per_branch = torch.nn.functional.pad(
                per_branch,
                (0, 0, 0, padded_branches - layout.branches),
            )
        active = per_branch.view(
            layout.batch_size,
            query_blocks,
            branches_per_query_block,
            kv_blocks,
        ).any(2)
    else:
        # Generic non-aligned oracle/configuration path.  Boolean semiring
        # matmul expresses the same query-block/branch overlap union without a
        # Python loop; production B4/B8/B16 uses the cheaper grouped path.
        query_starts = torch.arange(query_blocks, dtype=torch.long) * q_block
        query_stops = query_starts + q_block
        branch_starts = branch_numbers * layout.branch_length
        branch_stops = branch_starts + layout.branch_length
        membership = (
            (query_stops[:, None] > branch_starts[None])
            & (query_starts[:, None] < branch_stops[None])
        )
        active = torch.matmul(
            membership.to(torch.float32), per_branch.to(torch.float32)
        ).gt(0)
    # A full block must satisfy the token rule over the complete physical
    # q_block x kv_block tile.  Reduce query and key properties independently,
    # then combine their extrema.  This is exact for arbitrary validity masks,
    # packed document IDs, resetting absolute positions, and block alignment,
    # while keeping setup O(B * (Q + KV) + B * QB * KB).
    padded_query_length = query_blocks * q_block
    query_padding = padded_query_length - layout.query_length
    flat_query_valid = layout.branch_valid.flatten(1, 2)
    blocked_query_valid = torch.nn.functional.pad(
        flat_query_valid, (0, query_padding), value=False
    ).view(layout.batch_size, query_blocks, q_block)
    query_all_valid = blocked_query_valid.all(-1)

    query_branch = branch_numbers.repeat_interleave(layout.branch_length)
    query_branch = torch.nn.functional.pad(
        query_branch, (0, query_padding), value=-1
    ).view(query_blocks, q_block)
    query_branch_min = query_branch.amin(-1)
    query_branch_max = query_branch.amax(-1)
    query_single_branch = query_branch_min == query_branch_max

    def block_query_values(values: Tensor) -> Tensor:
        expanded = values.repeat_interleave(layout.branch_length, dim=-1)
        return torch.nn.functional.pad(
            expanded, (0, query_padding), value=0
        ).view(layout.batch_size, query_blocks, q_block)

    prefix_by_query = block_query_values(layout.prefix_lengths)
    query_prefix_min = prefix_by_query.amin(-1)

    padded_kv_length = kv_blocks * kv_block
    key_physical = torch.arange(padded_kv_length, dtype=torch.long)
    key_in_bounds = key_physical < layout.kv_length
    clean_key = key_physical < layout.clean_length
    branch_key = key_in_bounds & ~clean_key
    blocked_clean_key = clean_key.view(kv_blocks, kv_block)
    blocked_branch_key = branch_key.view(kv_blocks, kv_block)
    clean_key_present = blocked_clean_key.any(-1)
    branch_key_present = blocked_branch_key.any(-1)
    key_block_in_bounds = key_in_bounds.view(kv_blocks, kv_block).all(-1)

    clean_offset = key_physical.clamp_max(layout.clean_length - 1)
    clean_valid_by_key = layout.clean_valid[:, clean_offset] | ~clean_key[None]
    clean_all_valid = clean_valid_by_key.view(
        layout.batch_size, kv_blocks, kv_block
    ).all(-1)

    branch_physical = (key_physical - layout.clean_length).clamp_min(0)
    branch_offset = branch_physical.clamp_max(layout.query_length - 1)
    branch_valid_by_key = (
        flat_query_valid[:, branch_offset] | ~branch_key[None]
    )
    branch_all_valid = branch_valid_by_key.view(
        layout.batch_size, kv_blocks, kv_block
    ).all(-1)
    key_branch = torch.div(
        branch_physical, layout.branch_length, rounding_mode="floor"
    )
    key_branch_min = key_branch.masked_fill(~branch_key, layout.branches).view(
        kv_blocks, kv_block
    ).amin(-1)
    key_branch_max = key_branch.masked_fill(~branch_key, -1).view(
        kv_blocks, kv_block
    ).amax(-1)
    own_branch_block = (
        query_single_branch[:, None]
        & (query_branch_min[:, None] == key_branch_min[None])
        & (query_branch_max[:, None] == key_branch_max[None])
    )
    branch_allowed = branch_all_valid[:, None, :] & (
        ~branch_key_present[None, None, :] | own_branch_block[None]
    )

    if layout.clean_positions is None:
        clean_key_max = key_physical.masked_fill(~clean_key, -1).view(
            kv_blocks, kv_block
        ).amax(-1)
        clean_before_prefix = (
            clean_key_max[None, None, :] < query_prefix_min[:, :, None]
        )
    else:
        assert layout.branch_positions is not None
        clean_position_by_key = layout.clean_positions[:, clean_offset]
        clean_position_max = clean_position_by_key.masked_fill(
            ~clean_key[None], torch.iinfo(torch.int64).min
        ).view(layout.batch_size, kv_blocks, kv_block).amax(-1)
        branch_origin_by_query = block_query_values(
            layout.branch_positions[..., 0]
        )
        query_origin_min = branch_origin_by_query.amin(-1)
        query_origin_max = branch_origin_by_query.amax(-1)
        clean_before_prefix = (
            clean_position_max[:, None, :] < query_origin_min[:, :, None]
        )
        if layout.prefix_window is not None:
            clean_position_min = clean_position_by_key.masked_fill(
                ~clean_key[None], torch.iinfo(torch.int64).max
            ).view(layout.batch_size, kv_blocks, kv_block).amin(-1)
            clean_before_prefix &= clean_position_min[:, None, :] >= (
                query_origin_max[:, :, None] - layout.prefix_window
            )

    clean_allowed = clean_all_valid[:, None, :] & (
        ~clean_key_present[None, None, :] | clean_before_prefix
    )
    if layout.clean_segment_ids is not None:
        assert layout.branch_segment_ids is not None
        clean_segment_by_key = layout.clean_segment_ids[:, clean_offset]
        clean_segment_min = clean_segment_by_key.masked_fill(
            ~clean_key[None], torch.iinfo(torch.int64).max
        ).view(layout.batch_size, kv_blocks, kv_block).amin(-1)
        clean_segment_max = clean_segment_by_key.masked_fill(
            ~clean_key[None], torch.iinfo(torch.int64).min
        ).view(layout.batch_size, kv_blocks, kv_block).amax(-1)
        query_segment = block_query_values(layout.branch_segment_ids)
        query_segment_min = query_segment.amin(-1)
        query_segment_max = query_segment.amax(-1)
        same_segment = (
            (clean_segment_min[:, None, :] == clean_segment_max[:, None, :])
            & (query_segment_min == query_segment_max)[:, :, None]
            & (clean_segment_min[:, None, :] == query_segment_min[:, :, None])
        )
        clean_allowed &= ~clean_key_present[None, None, :] | same_segment

    full = (
        active
        & query_all_valid[:, :, None]
        & key_block_in_bounds[None, None, :]
        & clean_allowed
        & branch_allowed
    )
    partial = active & ~full

    def ordered(topology: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        counts = topology.sum(-1, dtype=torch.int32)[:, None]
        indices = torch.argsort(
            topology.to(torch.int8), dim=-1, descending=True, stable=True
        ).to(torch.int32)[:, None]
        transposed = topology.transpose(-2, -1)
        q_counts = transposed.sum(-1, dtype=torch.int32)[:, None]
        q_indices = torch.argsort(
            transposed.to(torch.int8), dim=-1, descending=True, stable=True
        ).to(torch.int32)[:, None]
        return counts, indices, q_counts, q_indices

    counts, indices, q_counts, q_indices = ordered(partial)
    full_counts, full_indices, full_q_counts, full_q_indices = ordered(full)
    return CanvasBlockMaskMetadata(
        counts,
        indices,
        q_counts,
        q_indices,
        (q_block, kv_block),
        layout.query_length,
        layout.kv_length,
        full_counts,
        full_indices,
        full_q_counts,
        full_q_indices,
    )


def canvas_branch_allowed(layout: CanvasBranchLayout) -> Tensor:
    """Dense ``[batch, branch_query, clean_plus_branch_key]`` canvas oracle."""

    batch = layout.batch_size
    clean_length = layout.clean_length
    canvas = layout.branch_length
    query_positions = torch.arange(layout.query_length, device=layout.clean_valid.device)
    branch_index = torch.div(query_positions, canvas, rounding_mode="floor")
    branch_offset = query_positions % canvas
    clean_index = torch.arange(clean_length, device=layout.clean_valid.device)
    prefix_stop = layout.prefix_lengths[:, branch_index]
    if layout.clean_positions is None:
        clean_before_prefix = clean_index[None, None, :] < prefix_stop[:, :, None]
    else:
        assert layout.branch_positions is not None
        clean_before_prefix = layout.clean_positions[:, None, :] < (
            layout.branch_positions[:, branch_index, 0, None]
        )
    clean_allowed = (
        layout.branch_valid[:, branch_index, branch_offset, None]
        & layout.clean_valid[:, None, :]
        & clean_before_prefix
    )
    if layout.clean_segment_ids is not None:
        assert layout.branch_segment_ids is not None
        clean_allowed &= (
            layout.clean_segment_ids[:, None, :]
            == layout.branch_segment_ids[:, branch_index, None]
        )
    if layout.prefix_window is not None:
        assert layout.clean_positions is not None
        assert layout.branch_positions is not None
        prefix_absolute = layout.branch_positions[:, branch_index, 0]
        key_absolute = layout.clean_positions[:, None, :]
        clean_allowed &= key_absolute >= (
            prefix_absolute[:, :, None] - layout.prefix_window
        )

    branch_key_index = torch.arange(layout.query_length, device=layout.clean_valid.device)
    key_branch = torch.div(branch_key_index, canvas, rounding_mode="floor")
    key_offset = branch_key_index % canvas
    own_branch = key_branch[None, :] == branch_index[:, None]
    branch_allowed = (
        layout.branch_valid[:, branch_index, branch_offset, None]
        & layout.branch_valid[:, key_branch, key_offset][:, None, :]
        & own_branch[None, :, :]
    )
    return torch.cat((clean_allowed, branch_allowed), dim=-1).reshape(
        batch, layout.query_length, layout.kv_length
    )


def introspection_branch_allowed(layout: IntrospectionBranchLayout) -> Tensor:
    """Dense proposal-query I-DLM oracle over clean plus proposal keys."""

    length = layout.clean_length
    positions = torch.arange(length, device=layout.clean_valid.device)
    query_block = torch.div(positions, layout.block_size, rounding_mode="floor")
    key_block = torch.div(positions, layout.block_size, rounding_mode="floor")
    query_valid = layout.proposal_valid[:, :, None]
    clean_allowed = (
        query_valid
        & layout.clean_valid[:, None, :]
        & (key_block[None, None, :] < query_block[None, :, None])
    )
    proposal_allowed = (
        query_valid
        & layout.proposal_valid[:, None, :]
        & (key_block[None, None, :] == query_block[None, :, None])
        & (positions[None, None, :] <= positions[None, :, None])
    )
    if layout.clean_segment_ids is not None:
        assert layout.proposal_segment_ids is not None
        proposal_segments = layout.proposal_segment_ids[:, :, None]
        clean_allowed &= proposal_segments == layout.clean_segment_ids[:, None, :]
        proposal_allowed &= proposal_segments == layout.proposal_segment_ids[:, None, :]
    return torch.cat((clean_allowed, proposal_allowed), dim=-1)


def _canvas_mask_mod(layout: CanvasBranchLayout):
    clean_length = layout.clean_length
    canvas = layout.branch_length
    branches = layout.branches

    def mask_mod(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
        del head
        branch = torch.div(query, canvas, rounding_mode="floor").clamp(0, branches - 1)
        query_offset = query % canvas
        clean_key = key < clean_length
        clean_offset = key.clamp(0, clean_length - 1)
        branch_physical = (key - clean_length).clamp_min(0)
        key_branch = torch.div(
            branch_physical, canvas, rounding_mode="floor"
        ).clamp(0, branches - 1)
        key_offset = branch_physical % canvas
        query_valid = layout.branch_valid[batch, branch, query_offset]
        if layout.clean_positions is None:
            before_prefix = key < layout.prefix_lengths[batch, branch]
        else:
            assert layout.branch_positions is not None
            before_prefix = layout.clean_positions[batch, clean_offset] < (
                layout.branch_positions[batch, branch, 0]
            )
        prefix = (
            clean_key
            & before_prefix
            & layout.clean_valid[batch, clean_offset]
        )
        if layout.clean_segment_ids is not None:
            assert layout.branch_segment_ids is not None
            prefix = prefix & (
                layout.clean_segment_ids[batch, clean_offset]
                == layout.branch_segment_ids[batch, branch]
            )
        if layout.prefix_window is not None:
            assert layout.clean_positions is not None
            assert layout.branch_positions is not None
            prefix_absolute = layout.branch_positions[batch, branch, 0]
            key_absolute = layout.clean_positions[batch, clean_offset]
            prefix = prefix & (
                key_absolute >= prefix_absolute - layout.prefix_window
            )
        own_branch = (
            ~clean_key
            & (key_branch == branch)
            & layout.branch_valid[batch, key_branch, key_offset]
        )
        return query_valid & (prefix | own_branch)

    return mask_mod


def _introspection_mask_mod(layout: IntrospectionBranchLayout):
    length = layout.clean_length
    block_size = layout.block_size

    def mask_mod(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
        del head
        clean_key = key < length
        key_offset = torch.where(clean_key, key, key - length).clamp(0, length - 1)
        query_offset = query.clamp(0, length - 1)
        query_block = torch.div(query_offset, block_size, rounding_mode="floor")
        key_block = torch.div(key_offset, block_size, rounding_mode="floor")
        query_valid = layout.proposal_valid[batch, query_offset]
        from_clean = (
            clean_key
            & layout.clean_valid[batch, key_offset]
            & (key_block < query_block)
        )
        from_proposal = (
            ~clean_key
            & layout.proposal_valid[batch, key_offset]
            & (key_block == query_block)
            & (key_offset <= query_offset)
        )
        if layout.clean_segment_ids is not None:
            assert layout.proposal_segment_ids is not None
            query_segment = layout.proposal_segment_ids[batch, query_offset]
            from_clean = from_clean & (
                layout.clean_segment_ids[batch, key_offset] == query_segment
            )
            from_proposal = from_proposal & (
                layout.proposal_segment_ids[batch, key_offset] == query_segment
            )
        return query_valid & (from_clean | from_proposal)

    return mask_mod


@torch.compiler.disable
def build_canvas_block_mask(
    layout: CanvasBranchLayout,
    *,
    block_size: int | tuple[int, int] = 128,
    compile_mask: bool | None = None,
    metadata: CanvasBlockMaskMetadata | None = None,
):
    """Build the exact sparse Flex ``BlockMask`` for canvas branch queries."""

    try:
        from torch.nn.attention.flex_attention import BlockMask, create_block_mask
    except Exception as error:
        raise RuntimeError("FlexAttention BlockMask construction is unavailable") from error
    if metadata is not None:
        if metadata.query_length != layout.query_length or metadata.kv_length != layout.kv_length:
            raise ValueError("canvas block metadata does not match layout geometry")
        if metadata.kv_num_blocks.device != layout.clean_valid.device:
            raise ValueError("canvas block metadata and layout must share a device")
        return BlockMask(
            seq_lengths=(layout.query_length, layout.kv_length),
            kv_num_blocks=metadata.kv_num_blocks,
            kv_indices=metadata.kv_indices,
            full_kv_num_blocks=metadata.full_kv_num_blocks,
            full_kv_indices=metadata.full_kv_indices,
            q_num_blocks=metadata.q_num_blocks,
            q_indices=metadata.q_indices,
            full_q_num_blocks=metadata.full_q_num_blocks,
            full_q_indices=metadata.full_q_indices,
            BLOCK_SIZE=metadata.block_size,
            mask_mod=_canvas_mask_mod(layout),
        )
    if compile_mask is None:
        # PyTorch 2.13 Inductor can fail symbolic indexing for tensor-backed
        # layouts on sm120. Mask construction is once per geometry; compile the
        # reused Flex attention call, not this setup step.
        compile_mask = False
    return create_block_mask(
        _canvas_mask_mod(layout),
        B=layout.batch_size,
        H=None,
        Q_LEN=layout.query_length,
        KV_LEN=layout.kv_length,
        device=layout.clean_valid.device,
        BLOCK_SIZE=block_size,
        _compile=compile_mask,
        separate_full_blocks=True,
    )


@torch.compiler.disable
def build_introspection_block_mask(
    layout: IntrospectionBranchLayout,
    *,
    block_size: int | tuple[int, int] = 128,
    compile_mask: bool | None = None,
):
    """Build the exact sparse Flex ``BlockMask`` for I-DLM proposal queries."""

    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except Exception as error:
        raise RuntimeError("FlexAttention BlockMask construction is unavailable") from error
    if compile_mask is None:
        compile_mask = False
    return create_block_mask(
        _introspection_mask_mod(layout),
        B=layout.batch_size,
        H=None,
        Q_LEN=layout.query_length,
        KV_LEN=layout.kv_length,
        device=layout.clean_valid.device,
        BLOCK_SIZE=block_size,
        _compile=compile_mask,
        separate_full_blocks=True,
    )


def _expand_gqa(key_or_value: Tensor, query_heads: int, head_axis: int) -> Tensor:
    kv_heads = key_or_value.shape[head_axis]
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by key/value heads")
    if query_heads == kv_heads:
        return key_or_value
    return key_or_value.repeat_interleave(query_heads // kv_heads, dim=head_axis)


def _dense_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    allowed: Tensor,
    *,
    scale: float | None,
) -> Tensor:
    """Dense branch oracle for ``[B,H,Q,D]`` tensors and ``[B,Q,K]`` mask."""

    key = _expand_gqa(key, query.shape[1], 1)
    value = _expand_gqa(value, query.shape[1], 1)
    effective_scale = query.shape[-1] ** -0.5 if scale is None else scale
    scores = torch.einsum("bhqd,bhkd->bhqk", query.float(), key.float())
    scores = scores * effective_scale
    allowed_heads = allowed[:, None]
    safe_rows = allowed_heads.any(-1, keepdim=True)
    masked_scores = scores.masked_fill(~allowed_heads, -torch.inf)
    masked_scores = torch.where(safe_rows, masked_scores, torch.zeros_like(masked_scores))
    weights = masked_scores.softmax(-1) * safe_rows
    output = torch.einsum("bhqk,bhkd->bhqd", weights, value.float())
    return output.to(query.dtype)


def dense_packed_clean_attention(
    packed: PackedCleanQKV,
    *,
    window: int | None = None,
    scale: float | None = None,
) -> Tensor:
    """Dense document-by-document oracle for packed causal attention."""

    packed.validate_boundaries()
    if window is not None and window <= 0:
        raise ValueError("window must be positive")
    offsets = packed.cu_seqlens.detach().cpu().tolist()
    outputs = []
    for start, stop in zip(offsets, offsets[1:]):
        if start == stop:
            continue
        query = packed.query[start:stop].transpose(0, 1)
        key = packed.key[start:stop].transpose(0, 1)
        value = packed.value[start:stop].transpose(0, 1)
        key = _expand_gqa(key, query.shape[0], 0)
        value = _expand_gqa(value, query.shape[0], 0)
        length = stop - start
        positions = torch.arange(length, device=query.device)
        allowed = positions[None, :] <= positions[:, None]
        if window is not None:
            allowed &= positions[None, :] > positions[:, None] - window
        effective_scale = query.shape[-1] ** -0.5 if scale is None else scale
        scores = torch.einsum("hqd,hkd->hqk", query.float(), key.float())
        weights = (scores * effective_scale).masked_fill(~allowed, -torch.inf).softmax(-1)
        output = torch.einsum("hqk,hkd->hqd", weights, value.float())
        outputs.append(output.to(query.dtype).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def _resolve_backend(
    requested: str,
    *,
    pattern: AttentionPattern,
    device: torch.device,
    sequence_length: int,
    capabilities: AttentionBackendCapabilities | None,
    allow_dense_reference: bool,
    dense_reference_limit: int,
    production: AttentionBackend,
) -> AttentionBackend:
    if requested == "auto":
        return select_attention_backend(
            pattern,
            device=device,
            sequence_length=sequence_length,
            capabilities=capabilities,
            allow_dense_reference=allow_dense_reference,
            dense_reference_limit=dense_reference_limit,
        )
    try:
        selected = AttentionBackend(requested)
    except ValueError as error:
        raise ValueError(f"unknown attention backend: {requested!r}") from error
    if selected is AttentionBackend.DENSE_REFERENCE:
        if not allow_dense_reference or sequence_length > dense_reference_limit:
            raise RuntimeError("dense reference backend is not permitted for this call")
        return selected
    if selected is not production:
        raise RuntimeError(
            f"backend={selected.value!r} cannot execute pattern={pattern.value!r}"
        )
    # Capability probing is a launch-time fail-closed check.  Repeating its
    # Python/lru-cache machinery inside every compiled attention layer creates
    # graph breaks and cannot change after the graph was admitted.
    if torch.compiler.is_compiling():
        if device.type != "cuda":
            raise RuntimeError("compiled production attention requires CUDA")
        return selected
    capabilities = capabilities or detect_attention_backend_capabilities()
    capability = {
        AttentionBackend.VARLEN_FLASH: capabilities.varlen_flash,
        AttentionBackend.FLEX: capabilities.flex_attention,
    }[production]
    if device.type != "cuda" or not capabilities.cuda_runtime or not capability:
        raise RuntimeError(
            f"requested production backend {production.value!r} is unavailable on {device}"
        )
    return selected


def packed_clean_attention(
    packed: PackedCleanQKV,
    *,
    window: int | None = None,
    scale: float | None = None,
    backend: CleanBackend = "auto",
    capabilities: AttentionBackendCapabilities | None = None,
    allow_dense_reference: bool = False,
    dense_reference_limit: int = DEFAULT_DENSE_REFERENCE_LIMIT,
) -> Tensor:
    """Run packed causal clean attention with native varlen Flash on CUDA.

    ``window`` counts the query itself.  The Flash call therefore receives
    ``window_size=(window - 1, 0)`` to match the dense rule
    ``query - window < key <= query`` exactly.
    """

    if window is not None and window <= 0:
        raise ValueError("window must be positive")
    pattern = AttentionPattern.CAUSAL if window is None else AttentionPattern.CAUSAL_WINDOW
    selected = _resolve_backend(
        backend,
        pattern=pattern,
        device=packed.query.device,
        sequence_length=packed.max_seqlen,
        capabilities=capabilities,
        allow_dense_reference=allow_dense_reference,
        dense_reference_limit=dense_reference_limit,
        production=AttentionBackend.VARLEN_FLASH,
    )
    if selected is AttentionBackend.DENSE_REFERENCE:
        return dense_packed_clean_attention(packed, window=window, scale=scale)
    try:
        from torch.nn.attention.varlen import varlen_attn
    except Exception as error:
        raise RuntimeError("native varlen Flash attention is unavailable") from error
    flash_window = (-1, 0) if window is None else (window - 1, 0)
    return varlen_attn(
        packed.query,
        packed.key,
        packed.value,
        packed.cu_seqlens,
        packed.cu_seqlens,
        packed.max_seqlen,
        packed.max_seqlen,
        scale=scale,
        window_size=flash_window,
        enable_gqa=packed.query.shape[1] != packed.key.shape[1],
    )


def branch_attention(
    query: Tensor,
    clean_key: Tensor,
    clean_value: Tensor,
    branch_key: Tensor,
    branch_value: Tensor,
    layout: BranchLayout,
    *,
    scale: float | None = None,
    backend: BranchBackend = "auto",
    block_mask=None,
    block_size: int | tuple[int, int] = 128,
    capabilities: AttentionBackendCapabilities | None = None,
    allow_dense_reference: bool = False,
    dense_reference_limit: int = DEFAULT_DENSE_REFERENCE_LIMIT,
) -> Tensor:
    """Attend branch queries over one shared clean-plus-branch K/V bank.

    Tensors use Flex's ``[batch, heads, sequence, head_dim]`` layout.  The clean
    K/V tensors are concatenated exactly once, regardless of branch count.
    Callers may construct and reuse ``block_mask`` across layers with the same
    geometry.
    """

    tensors = (query, clean_key, clean_value, branch_key, branch_value)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("branch attention Q/K/V must have rank four")
    _require_same_device("branch attention", tensors)
    if not all(tensor.is_floating_point() for tensor in tensors):
        raise TypeError("branch attention Q/K/V must be floating point")
    if len({tensor.dtype for tensor in tensors}) != 1:
        raise ValueError("branch attention Q/K/V dtypes must match")
    if clean_key.shape != clean_value.shape or branch_key.shape != branch_value.shape:
        raise ValueError("branch attention K/V shapes differ")
    if query.shape[0] != layout.batch_size or query.shape[2] != layout.query_length:
        raise ValueError("branch query shape does not match its layout")
    if clean_key.shape[0] != layout.batch_size or clean_key.shape[2] != layout.clean_length:
        raise ValueError("clean K/V shape does not match its layout")
    if branch_key.shape[0] != layout.batch_size or branch_key.shape[2] != layout.query_length:
        raise ValueError("branch K/V shape does not match its layout")
    if query.shape[-1] != clean_key.shape[-1] or query.shape[-1] != branch_key.shape[-1]:
        raise ValueError("branch attention head dimensions differ")
    if clean_key.shape[1] != branch_key.shape[1]:
        raise ValueError("clean and branch K/V head counts differ")
    if query.shape[1] % clean_key.shape[1]:
        raise ValueError("query heads must be divisible by key/value heads")
    if layout.clean_valid.device != query.device:
        raise ValueError("branch layout and Q/K/V must share a device")

    pattern = (
        AttentionPattern.BRANCH
        if isinstance(layout, CanvasBranchLayout)
        else AttentionPattern.INTROSPECTION
    )
    selected = _resolve_backend(
        backend,
        pattern=pattern,
        device=query.device,
        sequence_length=layout.kv_length,
        capabilities=capabilities,
        allow_dense_reference=allow_dense_reference,
        dense_reference_limit=dense_reference_limit,
        production=AttentionBackend.FLEX,
    )
    bank_key = torch.cat((clean_key, branch_key), dim=2)
    bank_value = torch.cat((clean_value, branch_value), dim=2)
    if selected is AttentionBackend.DENSE_REFERENCE:
        allowed = (
            canvas_branch_allowed(layout)
            if isinstance(layout, CanvasBranchLayout)
            else introspection_branch_allowed(layout)
        )
        return _dense_attention(query, bank_key, bank_value, allowed, scale=scale)

    try:
        from torch.nn.attention.flex_attention import flex_attention
    except Exception as error:
        raise RuntimeError("FlexAttention is unavailable") from error
    if block_mask is None:
        block_mask = (
            build_canvas_block_mask(layout, block_size=block_size)
            if isinstance(layout, CanvasBranchLayout)
            else build_introspection_block_mask(layout, block_size=block_size)
        )
    return flex_attention(
        query,
        bank_key,
        bank_value,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=query.shape[1] != bank_key.shape[1],
        kernel_options={"BACKEND": "TRITON", "ROWS_GUARANTEED_SAFE": False},
    )


__all__ = [
    "BranchLayout",
    "CanvasBlockMaskMetadata",
    "CanvasBranchLayout",
    "IntrospectionBranchLayout",
    "PackedCleanQKV",
    "branch_attention",
    "build_canvas_block_mask",
    "canvas_block_mask_metadata",
    "build_introspection_block_mask",
    "canvas_branch_allowed",
    "dense_packed_clean_attention",
    "introspection_branch_allowed",
    "packed_clean_attention",
]
