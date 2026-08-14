"""Core byte-diffusion layers with packed projections and absolute RoPE."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class CleanAttentionBank:
    """Rotated clean K/V tensors reused by branch-only inference passes."""

    key: Tensor
    value: Tensor

    def __post_init__(self) -> None:
        if self.key.ndim != 4 or self.value.shape != self.key.shape:
            raise ValueError("cached clean attention K/V must be aligned [B,H,L,D]")
        if self.key.device != self.value.device or self.key.dtype != self.value.dtype:
            raise ValueError("cached clean attention K/V must share dtype and device")


def packed_sequence_offsets(lengths: Tensor) -> Tensor:
    """Build int32 exclusive offsets without Inductor's cumsum matcher.

    Batches are at most a few dozen rows, so the O(batch²) broadcast reduction
    is negligible next to attention and avoids a PyTorch 2.13 dynamic-shape
    compiler failure in ``pointless_cumsum_replacement``.
    """

    if lengths.ndim != 1:
        raise ValueError("packed sequence lengths must be rank one")
    row = torch.arange(lengths.shape[0] + 1, device=lengths.device)[:, None]
    column = torch.arange(lengths.shape[0], device=lengths.device)[None, :]
    included = column < row
    return (
        lengths.to(torch.int32)[None, :]
        .mul(included)
        .sum(dim=1, dtype=torch.int32)
    )


def pack_valid(values: Tensor, valid: Tensor) -> Tensor:
    """Pack leading dimensions selected by ``valid`` without index-put ops."""

    if values.shape[: valid.ndim] != valid.shape or valid.dtype != torch.bool:
        raise ValueError("valid mask must match the leading value dimensions")
    trailing = values.shape[valid.ndim :]
    expanded = valid.reshape(*valid.shape, *([1] * len(trailing))).expand_as(values)
    return torch.masked_select(values, expanded).reshape(-1, *trailing)


def prefix_row_indices(valid: Tensor) -> Tensor:
    """Return flat row indices for a prefix-valid padded bank.

    The returned data-dependent index vector is intentionally reusable across
    every hidden width at the same sequence resolution.  Calling
    :func:`pack_valid` separately for embeddings, positions, conditions, and
    logits expands the boolean mask across each trailing dimension and makes
    every call perform its own dynamic ``masked_select``.  A single row index
    vector followed by fixed-width ``index_select``/``index_copy`` operations
    preserves the exact packed order with substantially less host dispatch.
    """

    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError("prefix validity must be a rank-2 boolean tensor")
    lengths = valid.sum(1)
    expected = torch.arange(valid.shape[1], device=valid.device)[None] < lengths[:, None]
    if not torch.compiler.is_compiling() and not torch.equal(valid, expected):
        raise ValueError("valid rows must be contiguous prefixes")
    return torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()


def valid_row_indices(valid: Tensor) -> Tensor:
    """Return flat indices for an arbitrary padded validity mask.

    Document-aligned pages may contain at most three invalid alignment slots
    between two valid document segments.  The packed kernels must skip those
    slots without imposing the older contiguous-prefix row contract.
    """

    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError("validity must be a rank-2 boolean tensor")
    return torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()


def pack_rows(values: Tensor, indices: Tensor) -> Tensor:
    """Pack the first two dimensions using reusable flat row ``indices``."""

    if values.ndim < 2 or indices.ndim != 1 or indices.dtype != torch.long:
        raise ValueError("values must be rank >=2 and indices rank-1 int64")
    trailing = values.shape[2:]
    return values.reshape(-1, *trailing).index_select(0, indices)


def unpack_rows(packed: Tensor, indices: Tensor, template: Tensor) -> Tensor:
    """Scatter a packed row bank into a zero-filled padded template shape."""

    if template.ndim < 2 or indices.ndim != 1 or indices.dtype != torch.long:
        raise ValueError("template must be rank >=2 and indices rank-1 int64")
    trailing = template.shape[2:]
    if packed.shape[1:] != trailing or packed.shape[0] != indices.shape[0]:
        raise ValueError("packed rows, indices, and template shape do not align")
    flat = packed.new_zeros(
        (template.shape[0] * template.shape[1], *trailing)
    )
    return flat.index_copy(0, indices, packed).view(template.shape)


def unpack_valid(
    packed: Tensor,
    valid: Tensor,
    template: Tensor,
) -> Tensor:
    """Inverse of :func:`pack_valid` using compile-safe out-of-place scatter."""

    if template.shape[: valid.ndim] != valid.shape or valid.dtype != torch.bool:
        raise ValueError("valid mask must match the leading template dimensions")
    trailing = template.shape[valid.ndim :]
    if packed.shape[1:] != trailing:
        raise ValueError("packed values and template trailing dimensions differ")
    expanded = valid.reshape(*valid.shape, *([1] * len(trailing))).expand_as(
        template
    )
    # ``template`` supplies only the padded shape. Under autocast the packed
    # transform may legitimately be BF16 even when the pre-transform template
    # came from an FP32 embedding lookup. The inverse of ``pack_valid`` must
    # preserve the packed values' dtype rather than inherit the template dtype.
    return torch.masked_scatter(
        packed.new_zeros(template.shape), expanded, packed.reshape(-1)
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized.to(x.dtype) * self.weight).to(x.dtype)


def apply_rotary(q: Tensor, k: Tensor, positions: Tensor, theta: float) -> tuple[Tensor, Tensor]:
    """Apply RoPE to ``[batch, heads, sequence, head_dim]`` tensors."""

    head_dim = q.shape[-1]
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    frequencies = torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32)
    frequencies = theta ** (-frequencies / head_dim)
    angles = positions.to(torch.float32)[..., None] * frequencies
    if positions.ndim == 1:
        angles = angles[None, None, :, :]
    elif positions.ndim == 2:
        angles = angles[:, None, :, :]
    else:
        raise ValueError("positions must be shared rank-1 or per-row rank-2")
    cosine, sine = angles.cos().to(q.dtype), angles.sin().to(q.dtype)

    def rotate(x: Tensor) -> Tensor:
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cosine - odd * sine, odd * cosine + even * sine), dim=-1).flatten(-2)

    return rotate(q), rotate(k)


def _packed_full_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_query: Tensor,
    cu_key: Tensor,
    *,
    max_query: int,
    max_key: int,
    allow_dense_reference: bool,
) -> Tensor:
    """Asymmetric varlen attention for isolated diffusion branches."""

    if allow_dense_reference:
        outputs: list[Tensor] = []
        for sequence in range(cu_query.numel() - 1):
            q_start, q_stop = (int(value) for value in cu_query[sequence : sequence + 2])
            k_start, k_stop = (int(value) for value in cu_key[sequence : sequence + 2])
            attended = F.scaled_dot_product_attention(
                query[q_start:q_stop].transpose(0, 1)[None],
                key[k_start:k_stop].transpose(0, 1)[None],
                value[k_start:k_stop].transpose(0, 1)[None],
            )
            outputs.append(attended[0].transpose(0, 1))
        return torch.cat(outputs) if outputs else query.new_empty(query.shape)
    if not query.is_cuda:
        raise RuntimeError("varlen branch attention requires CUDA")
    from torch.nn.attention.varlen import varlen_attn

    return varlen_attn(
        query,
        key,
        value,
        cu_query,
        cu_key,
        max_query,
        max_key,
    )


class PackedSelfAttention(nn.Module):
    """Bias-free packed QKV attention suitable for Inductor fusion."""

    def __init__(self, dim: int, heads: int, rope_theta: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("attention width must divide head count")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.rope_theta = rope_theta
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: Tensor,
        *,
        positions: Tensor | None = None,
        allowed: Tensor | None = None,
        causal: bool = False,
    ) -> Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        if positions is None:
            positions = torch.arange(length, device=x.device)
        q, k = apply_rotary(q, k, positions, self.rope_theta)
        attention_mask = allowed
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                raise TypeError("allowed attention mask must be boolean")
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[None, None]
            elif attention_mask.ndim == 3:
                attention_mask = attention_mask[:, None]
            elif attention_mask.ndim != 4:
                raise ValueError("attention mask must have rank 2, 3, or 4")
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=causal and attention_mask is None,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, length, self.dim))

    def forward_packed(
        self,
        x: Tensor,
        *,
        cu_seqlens: Tensor,
        positions: Tensor,
        max_seqlen: int,
        window: int | None = None,
        allow_dense_reference: bool = False,
    ) -> Tensor:
        """True variable-length clean attention over PAD-free packed states."""

        from .attention import PackedCleanQKV, packed_clean_attention

        if x.ndim != 2 or positions.shape != x.shape[:1]:
            raise ValueError("packed states and absolute positions must align")
        qkv = self.qkv(x).view(x.shape[0], 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(1)
        rotated_q, rotated_k = apply_rotary(
            q.transpose(0, 1)[None],
            k.transpose(0, 1)[None],
            positions,
            self.rope_theta,
        )
        q = rotated_q[0].transpose(0, 1)
        k = rotated_k[0].transpose(0, 1)
        packed = PackedCleanQKV(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            absolute_positions=positions,
            max_seqlen=max_seqlen,
        )
        attended = packed_clean_attention(
            packed,
            window=window,
            backend="dense_reference" if allow_dense_reference else "varlen_flash",
            allow_dense_reference=allow_dense_reference,
        )
        return self.output(attended.reshape(x.shape[0], self.dim))

    def forward_clean_and_branches(
        self,
        x: Tensor,
        *,
        clean_valid: Tensor,
        positions: Tensor,
        layout,
        clean_window: int | None,
        allow_dense_reference: bool,
        assume_full_clean: bool = False,
        block_mask=None,
    ) -> Tensor:
        """One packed QKV bank for varlen clean and isolated branch queries."""

        from .attention import PackedCleanQKV, branch_attention, packed_clean_attention

        batch, total_length, _ = x.shape
        clean_length = clean_valid.shape[1]
        if positions.shape != (batch, total_length) or layout.kv_length != total_length:
            raise ValueError("branched positions/layout do not match the physical bank")
        lengths = clean_valid.sum(1)
        expected = torch.arange(clean_length, device=x.device)[None] < lengths[:, None]
        if not torch.compiler.is_compiling() and not torch.equal(clean_valid, expected):
            raise ValueError("clean branch bank must contain one contiguous document per row")
        qkv = self.qkv(x).view(batch, total_length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        q, k = apply_rotary(q, k, positions, self.rope_theta)

        if assume_full_clean:
            cu = torch.arange(
                batch + 1, dtype=torch.int32, device=x.device
            ) * clean_length
            clean_q = q[:, :, :clean_length].transpose(1, 2).reshape(
                batch * clean_length, self.heads, self.head_dim
            )
            clean_k = k[:, :, :clean_length].transpose(1, 2).reshape_as(clean_q)
            clean_v = v[:, :, :clean_length].transpose(1, 2).reshape_as(clean_q)
            clean_positions = positions[:, :clean_length].reshape(-1)
        else:
            cu = packed_sequence_offsets(lengths)
            clean_q = pack_valid(
                q[:, :, :clean_length].transpose(1, 2), clean_valid
            )
            clean_k = pack_valid(
                k[:, :, :clean_length].transpose(1, 2), clean_valid
            )
            clean_v = pack_valid(
                v[:, :, :clean_length].transpose(1, 2), clean_valid
            )
            clean_positions = pack_valid(
                positions[:, :clean_length], clean_valid
            )
        clean_packed = PackedCleanQKV(
            clean_q,
            clean_k,
            clean_v,
            cu,
            clean_positions,
            clean_length,
        )
        clean_attended = packed_clean_attention(
            clean_packed,
            window=clean_window,
            backend="dense_reference" if allow_dense_reference else "varlen_flash",
            allow_dense_reference=allow_dense_reference,
        )
        if assume_full_clean:
            clean_output = clean_attended.view(
                batch, clean_length, self.heads, self.head_dim
            )
        else:
            clean_output = unpack_valid(
                clean_attended,
                clean_valid,
                clean_attended.new_empty(
                    (batch, clean_length, self.heads, self.head_dim)
                ),
            )
        branch_output = branch_attention(
            q[:, :, clean_length:],
            k[:, :, :clean_length],
            v[:, :, :clean_length],
            k[:, :, clean_length:],
            v[:, :, clean_length:],
            layout,
            backend="dense_reference" if allow_dense_reference else "flex",
            block_mask=block_mask,
            allow_dense_reference=allow_dense_reference,
        ).transpose(1, 2)
        # Project the two banks before concatenation.  Reshaping a concatenated
        # symbolic ``clean + branches * block`` extent makes Inductor's dynamic
        # backward scheduler prove divisibility across a sum; on sm120 it
        # raises ``CantSplit`` even though both operands independently contain
        # complete heads.  The linear map is position-wise, so this rewrite is
        # equation-identical and gives each reshape a simple product extent.
        clean_projected = self.output(clean_output.flatten(-2))
        branch_projected = self.output(branch_output.flatten(-2))
        return torch.cat((clean_projected, branch_projected), dim=1)

    def forward_document_branches(
        self,
        x: Tensor,
        *,
        clean_valid: Tensor,
        clean_indices: Tensor,
        clean_cu_seqlens: Tensor,
        clean_document_ids: Tensor,
        positions: Tensor,
        branch_valid: Tensor,
        branch_starts: Tensor,
        clean_window: int | None,
        allow_dense_reference: bool,
    ) -> Tensor:
        """Document-local clean attention plus asymmetric branch attention."""

        batch, total_length, _ = x.shape
        clean_length = clean_valid.shape[1]
        branches, branch_length = branch_valid.shape[1:]
        if total_length != clean_length + branches * branch_length:
            raise ValueError("clean and branch banks do not match states")
        if clean_document_ids.shape != clean_valid.shape:
            raise ValueError("clean document ids do not align")
        qkv = self.qkv(x).view(
            batch, total_length, 3, self.heads, self.head_dim
        )
        q, k, v = qkv.unbind(2)

        clean_positions = positions[:, :clean_length]
        packed_clean_positions = pack_rows(clean_positions, clean_indices)
        clean_q = pack_rows(q[:, :clean_length], clean_indices)
        clean_k = pack_rows(k[:, :clean_length], clean_indices)
        clean_v = pack_rows(v[:, :clean_length], clean_indices)
        rotated_q, rotated_k = apply_rotary(
            clean_q.transpose(0, 1)[None],
            clean_k.transpose(0, 1)[None],
            packed_clean_positions,
            self.rope_theta,
        )
        from .attention import PackedCleanQKV, packed_clean_attention

        clean_packed = PackedCleanQKV(
            rotated_q[0].transpose(0, 1),
            rotated_k[0].transpose(0, 1),
            clean_v,
            clean_cu_seqlens,
            packed_clean_positions,
            clean_length,
        )
        clean_attended = packed_clean_attention(
            clean_packed,
            window=clean_window,
            backend="dense_reference" if allow_dense_reference else "varlen_flash",
            allow_dense_reference=allow_dense_reference,
        )
        clean_output = unpack_rows(
            clean_attended,
            clean_indices,
            clean_attended.new_empty(
                (batch, clean_length, self.heads, self.head_dim)
            ),
        )

        branch_q = q[:, clean_length:].view(
            batch * branches, branch_length, self.heads, self.head_dim
        )
        branch_k = k[:, clean_length:].view_as(branch_q)
        branch_v = v[:, clean_length:].view_as(branch_q)
        branch_positions = positions[:, clean_length:].view(
            batch * branches, branch_length
        )
        branch_valid_flat = branch_valid.reshape(batch * branches, branch_length)
        active_branches = branch_valid_flat.any(-1)
        active_indices = torch.nonzero(active_branches, as_tuple=False).flatten()
        active_valid = branch_valid_flat.index_select(0, active_indices)
        query_indices = valid_row_indices(active_valid)
        packed_query = pack_rows(
            branch_q.index_select(0, active_indices), query_indices
        )
        packed_query_positions = pack_rows(
            branch_positions.index_select(0, active_indices), query_indices
        )

        prefix_width = clean_length if clean_window is None else clean_window
        prefix_offsets = torch.arange(prefix_width, device=x.device)
        flat_starts = branch_starts.reshape(-1).index_select(0, active_indices)
        prefix_columns = flat_starts[:, None] - prefix_width + prefix_offsets
        prefix_exists = prefix_columns.ge(0)
        safe_columns = prefix_columns.clamp_min(0)
        active_rows = torch.div(active_indices, branches, rounding_mode="floor")
        prefix_documents = clean_document_ids[active_rows[:, None], safe_columns]
        start_documents = clean_document_ids[active_rows, flat_starts]
        prefix_valid = (
            prefix_exists
            & clean_valid[active_rows[:, None], safe_columns]
            & prefix_documents.eq(start_documents[:, None])
        )
        prefix_k = k[active_rows[:, None], safe_columns]
        prefix_v = v[active_rows[:, None], safe_columns]
        prefix_positions = clean_positions[active_rows[:, None], safe_columns]
        active_branch_k = branch_k.index_select(0, active_indices)
        active_branch_v = branch_v.index_select(0, active_indices)
        active_branch_positions = branch_positions.index_select(0, active_indices)
        kv_valid = torch.cat((prefix_valid, active_valid), dim=1)
        kv_indices = valid_row_indices(kv_valid)
        packed_key = pack_rows(
            torch.cat((prefix_k, active_branch_k), dim=1), kv_indices
        )
        packed_value = pack_rows(
            torch.cat((prefix_v, active_branch_v), dim=1), kv_indices
        )
        packed_key_positions = pack_rows(
            torch.cat((prefix_positions, active_branch_positions), dim=1),
            kv_indices,
        )
        rotated_query, _ = apply_rotary(
            packed_query.transpose(0, 1)[None],
            packed_query.transpose(0, 1)[None],
            packed_query_positions,
            self.rope_theta,
        )
        rotated_key, _ = apply_rotary(
            packed_key.transpose(0, 1)[None],
            packed_key.transpose(0, 1)[None],
            packed_key_positions,
            self.rope_theta,
        )
        query_lengths = active_valid.sum(-1).to(torch.int32)
        key_lengths = kv_valid.sum(-1).to(torch.int32)
        cu_query = torch.cat(
            (
                query_lengths.new_zeros(1),
                query_lengths.cumsum(0, dtype=torch.int32),
            )
        )
        cu_key = torch.cat(
            (
                key_lengths.new_zeros(1),
                key_lengths.cumsum(0, dtype=torch.int32),
            )
        )
        branch_attended = _packed_full_attention(
            rotated_query[0].transpose(0, 1),
            rotated_key[0].transpose(0, 1),
            packed_value,
            cu_query,
            cu_key,
            max_query=branch_length,
            max_key=prefix_width + branch_length,
            allow_dense_reference=allow_dense_reference,
        )
        active_branch_output = unpack_rows(
            branch_attended,
            query_indices,
            branch_attended.new_empty(
                (active_indices.numel(), branch_length, self.heads, self.head_dim)
            ),
        )
        branch_output = branch_attended.new_zeros(
            (batch * branches, branch_length, self.heads, self.head_dim)
        )
        branch_output.index_copy_(0, active_indices, active_branch_output)
        attended = torch.cat(
            (clean_output, branch_output.view(batch, -1, self.heads, self.head_dim)),
            dim=1,
        )
        return self.output(attended.reshape(batch, total_length, self.dim))

    def forward_shared_document_branches(
        self,
        x: Tensor,
        *,
        clean_length: int,
        clean_indices: Tensor,
        clean_cu_seqlens: Tensor,
        positions: Tensor,
        layout,
        clean_window: int | None,
        allow_dense_reference: bool,
        assume_physical_clean: bool = False,
        block_mask=None,
    ) -> Tensor:
        """Document-packed clean Flash plus shared-prefix branch Flex attention.

        Q/K/V share weights but are projected as separate clean and branch
        banks, avoiding reductions over a symbolic ``clean + branch`` extent.
        Clean document sequences are packed only for varlen Flash. Branch
        attention consumes the original padded clean K/V bank once and the
        original branch K/V bank once; no clean prefix is gathered per origin.
        """

        from .attention import PackedCleanQKV, branch_attention, packed_clean_attention

        batch, total_length, _ = x.shape
        if not 0 < clean_length < total_length:
            raise ValueError("clean length must split the physical state bank")
        if positions.shape != (batch, total_length):
            raise ValueError("shared branch positions must align with states")
        if layout.clean_length != clean_length or layout.kv_length != total_length:
            raise ValueError("shared branch layout does not match the physical bank")
        if clean_indices.ndim != 1 or clean_indices.dtype != torch.long:
            raise ValueError("clean indices must be rank-1 int64")
        if clean_cu_seqlens.ndim != 1 or clean_cu_seqlens.dtype != torch.int32:
            raise ValueError("clean offsets must be rank-1 int32")

        # QKV is also a shared positionwise projection. One static combined
        # GEMM is materially more efficient than separate clean/branch calls.
        qkv = self.qkv(x).view(
            batch, total_length, 3, self.heads, self.head_dim
        )
        clean_qkv = qkv[:, :clean_length]
        branch_length = total_length - clean_length
        branch_qkv = qkv[:, clean_length:]
        clean_q, clean_k, clean_v = (
            value.transpose(1, 2) for value in clean_qkv.unbind(2)
        )
        branch_q, branch_k, branch_v = (
            value.transpose(1, 2) for value in branch_qkv.unbind(2)
        )
        clean_q, clean_k = apply_rotary(
            clean_q, clean_k, positions[:, :clean_length], self.rope_theta
        )
        branch_q, branch_k = apply_rotary(
            branch_q, branch_k, positions[:, clean_length:], self.rope_theta
        )

        # Production compile-stable metadata covers the complete physical
        # clean bank in row-major order. In that case packing/unpacking is an
        # identity, but generic index_select/index_copy left a material
        # index_add backward hotspot. Compact metadata still takes the exact
        # gather/scatter path used by CPU/reference tests.
        full_physical_clean = assume_physical_clean
        if full_physical_clean and clean_indices.numel() != batch * clean_length:
            raise ValueError("physical clean metadata must cover the complete bank")
        clean_q_rows = clean_q.transpose(1, 2).reshape(
            batch * clean_length, self.heads, self.head_dim
        )
        clean_k_rows = clean_k.transpose(1, 2).reshape_as(clean_q_rows)
        clean_v_rows = clean_v.transpose(1, 2).reshape_as(clean_q_rows)
        packed_clean_q = (
            clean_q_rows
            if full_physical_clean
            else clean_q_rows.index_select(0, clean_indices)
        )
        packed_clean_k = (
            clean_k_rows
            if full_physical_clean
            else clean_k_rows.index_select(0, clean_indices)
        )
        packed_clean_v = (
            clean_v_rows
            if full_physical_clean
            else clean_v_rows.index_select(0, clean_indices)
        )
        clean_position_rows = positions[:, :clean_length].reshape(-1)
        clean_positions = (
            clean_position_rows
            if full_physical_clean
            else clean_position_rows.index_select(0, clean_indices)
        )
        clean_packed = PackedCleanQKV(
            packed_clean_q,
            packed_clean_k,
            packed_clean_v,
            clean_cu_seqlens,
            clean_positions,
            clean_length,
        )
        clean_attended = packed_clean_attention(
            clean_packed,
            window=clean_window,
            backend="dense_reference" if allow_dense_reference else "varlen_flash",
            allow_dense_reference=allow_dense_reference,
        )
        clean_output = (
            clean_attended.view(batch, clean_length, self.heads, self.head_dim)
            if full_physical_clean
            else unpack_rows(
                clean_attended,
                clean_indices,
                clean_attended.new_empty(
                    (batch, clean_length, self.heads, self.head_dim)
                ),
            )
        )
        branch_output = branch_attention(
            branch_q,
            clean_k,
            clean_v,
            branch_k,
            branch_v,
            layout,
            backend="dense_reference" if allow_dense_reference else "flex",
            block_mask=block_mask,
            allow_dense_reference=allow_dense_reference,
        ).transpose(1, 2)
        # Projection is positionwise and shared. Compile-stable production
        # geometry makes the combined extent static, so one larger GEMM is
        # exactly equivalent and avoids two launches per attention block.
        attended = torch.cat((clean_output, branch_output), dim=1)
        return self.output(attended.flatten(-2))

    def prepare_clean_bank(
        self,
        clean_states: Tensor,
        *,
        clean_indices: Tensor,
        clean_cu_seqlens: Tensor,
        positions: Tensor,
        clean_window: int | None,
        allow_dense_reference: bool,
        assume_physical_clean: bool = False,
    ) -> tuple[Tensor, CleanAttentionBank]:
        """Run the clean attention once and retain its rotated K/V bank."""

        from .attention import PackedCleanQKV, packed_clean_attention

        if clean_states.ndim != 3 or positions.shape != clean_states.shape[:2]:
            raise ValueError("clean states and positions must align")
        batch, clean_length, _ = clean_states.shape
        qkv = self.qkv(clean_states).view(
            batch, clean_length, 3, self.heads, self.head_dim
        )
        clean_q, clean_k, clean_v = (
            value.transpose(1, 2) for value in qkv.unbind(2)
        )
        clean_q, clean_k = apply_rotary(
            clean_q, clean_k, positions, self.rope_theta
        )
        full_physical_clean = assume_physical_clean
        if full_physical_clean and clean_indices.numel() != batch * clean_length:
            raise ValueError("physical clean metadata must cover the complete bank")
        q_rows = clean_q.transpose(1, 2).reshape(
            batch * clean_length, self.heads, self.head_dim
        )
        k_rows = clean_k.transpose(1, 2).reshape_as(q_rows)
        v_rows = clean_v.transpose(1, 2).reshape_as(q_rows)
        position_rows = positions.reshape(-1)
        packed = PackedCleanQKV(
            q_rows if full_physical_clean else q_rows.index_select(0, clean_indices),
            k_rows if full_physical_clean else k_rows.index_select(0, clean_indices),
            v_rows if full_physical_clean else v_rows.index_select(0, clean_indices),
            clean_cu_seqlens,
            (
                position_rows
                if full_physical_clean
                else position_rows.index_select(0, clean_indices)
            ),
            clean_length,
        )
        attended = packed_clean_attention(
            packed,
            window=clean_window,
            backend="dense_reference" if allow_dense_reference else "varlen_flash",
            allow_dense_reference=allow_dense_reference,
        )
        clean_output = (
            attended.view(batch, clean_length, self.heads, self.head_dim)
            if full_physical_clean
            else unpack_rows(
                attended,
                clean_indices,
                attended.new_zeros(
                    (batch, clean_length, self.heads, self.head_dim)
                ),
            )
        )
        return self.output(clean_output.flatten(-2)), CleanAttentionBank(
            clean_k, clean_v
        )

    def forward_branch_from_clean_bank(
        self,
        branch_states: Tensor,
        clean_bank: CleanAttentionBank,
        *,
        positions: Tensor,
        layout,
        allow_dense_reference: bool,
        block_mask=None,
    ) -> Tensor:
        """Attend revisable branch queries to cached clean K/V and branch K/V."""

        from .attention import branch_attention

        if branch_states.ndim != 3 or positions.shape != branch_states.shape[:2]:
            raise ValueError("branch states and positions must align")
        batch, branch_length, _ = branch_states.shape
        if clean_bank.key.shape[0] != batch:
            raise ValueError("cached clean bank batch does not match branches")
        if layout.clean_length != clean_bank.key.shape[2]:
            raise ValueError("cached clean bank length does not match branch layout")
        qkv = self.qkv(branch_states).view(
            batch, branch_length, 3, self.heads, self.head_dim
        )
        branch_q, branch_k, branch_v = (
            value.transpose(1, 2) for value in qkv.unbind(2)
        )
        if branch_k.dtype != clean_bank.key.dtype:
            raise ValueError(
                "branch projection dtype differs from the prepared clean bank; "
                "prepare and consume the cache under the same autocast policy"
            )
        branch_q, branch_k = apply_rotary(
            branch_q, branch_k, positions, self.rope_theta
        )
        attended = branch_attention(
            branch_q,
            clean_bank.key,
            clean_bank.value,
            branch_k,
            branch_v,
            layout,
            backend="dense_reference" if allow_dense_reference else "flex",
            block_mask=block_mask,
            allow_dense_reference=allow_dense_reference,
        ).transpose(1, 2)
        return self.output(attended.flatten(-2))

    def forward_packed_document_branches(
        self,
        x: Tensor,
        *,
        clean_length: int,
        clean_indices: Tensor,
        clean_cu_seqlens: Tensor,
        positions: Tensor,
        branch_query_indices: Tensor,
        branch_kv_indices: Tensor,
        branch_query_cu_seqlens: Tensor,
        branch_kv_cu_seqlens: Tensor,
        max_branch_query_length: int,
        max_branch_kv_length: int,
        clean_window: int | None,
        allow_dense_reference: bool,
    ) -> Tensor:
        """Document-local clean and branch attention from prepacked metadata.

        Branch indices address the flattened physical ``[batch, total_length]``
        bank.  They are constructed before the CUDA forward: query indices list
        each valid noisy byte in branch order, while K/V indices list that
        branch's exact same-document clean prefix followed by its valid noisy
        bytes.  Their cumulative offsets delimit one sequence per active branch.

        Keeping these selections explicit is important for the training hot
        path.  Computing them from CUDA validity masks with ``nonzero`` makes
        their output shapes data-dependent, which both breaks ``torch.compile``
        and synchronizes the host.  Prepacked indices keep every operation here
        fixed-shape with respect to its tensor inputs while preserving exact
        native-varlen attention and backward semantics.
        """

        batch, total_length, _ = x.shape
        if not 0 < clean_length < total_length:
            raise ValueError("clean length must split the physical state bank")
        if positions.shape != (batch, total_length):
            raise ValueError("packed branch positions must align with states")
        index_tensors = (clean_indices, branch_query_indices, branch_kv_indices)
        if any(index.ndim != 1 or index.dtype != torch.long for index in index_tensors):
            raise ValueError("packed row indices must be rank-1 int64 tensors")
        cu_tensors = (
            clean_cu_seqlens,
            branch_query_cu_seqlens,
            branch_kv_cu_seqlens,
        )
        if any(cu.ndim != 1 or cu.dtype != torch.int32 for cu in cu_tensors):
            raise ValueError("packed sequence offsets must be rank-1 int32 tensors")
        if branch_query_cu_seqlens.shape != branch_kv_cu_seqlens.shape:
            raise ValueError("branch Q and K/V offsets must describe the same sequences")
        if max_branch_query_length <= 0 or max_branch_kv_length <= 0:
            raise ValueError("packed branch maximum lengths must be positive")

        qkv = self.qkv(x).view(
            batch, total_length, 3, self.heads, self.head_dim
        )
        q, k, v = qkv.unbind(2)

        clean_positions = positions[:, :clean_length]
        packed_clean_positions = pack_rows(clean_positions, clean_indices)
        clean_q = pack_rows(q[:, :clean_length], clean_indices)
        clean_k = pack_rows(k[:, :clean_length], clean_indices)
        clean_v = pack_rows(v[:, :clean_length], clean_indices)
        rotated_q, rotated_k = apply_rotary(
            clean_q.transpose(0, 1)[None],
            clean_k.transpose(0, 1)[None],
            packed_clean_positions,
            self.rope_theta,
        )
        from .attention import PackedCleanQKV, packed_clean_attention

        clean_packed = PackedCleanQKV(
            rotated_q[0].transpose(0, 1),
            rotated_k[0].transpose(0, 1),
            clean_v,
            clean_cu_seqlens,
            packed_clean_positions,
            clean_length,
        )
        clean_attended = packed_clean_attention(
            clean_packed,
            window=clean_window,
            backend="dense_reference" if allow_dense_reference else "varlen_flash",
            allow_dense_reference=allow_dense_reference,
        )
        clean_output = unpack_rows(
            clean_attended,
            clean_indices,
            clean_attended.new_empty(
                (batch, clean_length, self.heads, self.head_dim)
            ),
        )

        flat_q = q.reshape(-1, self.heads, self.head_dim)
        flat_k = k.reshape(-1, self.heads, self.head_dim)
        flat_v = v.reshape(-1, self.heads, self.head_dim)
        flat_positions = positions.reshape(-1)
        packed_query = flat_q.index_select(0, branch_query_indices)
        packed_key = flat_k.index_select(0, branch_kv_indices)
        packed_value = flat_v.index_select(0, branch_kv_indices)
        packed_query_positions = flat_positions.index_select(
            0, branch_query_indices
        )
        packed_key_positions = flat_positions.index_select(0, branch_kv_indices)
        if branch_query_indices.numel() == 0:
            # A page containing only sub-patch documents has no eligible BLT
            # origin. This is a shape-level branch (and therefore compile-safe),
            # not a CUDA value inspection. Native varlen kernels need at least
            # one sequence, so return the correctly empty branch bank directly.
            branch_attended = packed_query
        else:
            rotated_query, _ = apply_rotary(
                packed_query.transpose(0, 1)[None],
                packed_query.transpose(0, 1)[None],
                packed_query_positions,
                self.rope_theta,
            )
            rotated_key, _ = apply_rotary(
                packed_key.transpose(0, 1)[None],
                packed_key.transpose(0, 1)[None],
                packed_key_positions,
                self.rope_theta,
            )
            branch_attended = _packed_full_attention(
                rotated_query[0].transpose(0, 1),
                rotated_key[0].transpose(0, 1),
                packed_value,
                branch_query_cu_seqlens,
                branch_kv_cu_seqlens,
                max_query=max_branch_query_length,
                max_key=max_branch_kv_length,
                allow_dense_reference=allow_dense_reference,
            )

        # Query indices address the combined physical bank. Convert them to the
        # flattened branch-only bank without selecting on data-dependent masks.
        branch_width = total_length - clean_length
        query_rows = torch.div(
            branch_query_indices, total_length, rounding_mode="floor"
        )
        query_columns = branch_query_indices.remainder(total_length)
        branch_output_indices = (
            query_rows * branch_width + query_columns - clean_length
        )
        branch_output = branch_attended.new_zeros(
            (batch * branch_width, self.heads, self.head_dim)
        ).index_copy(0, branch_output_indices, branch_attended)
        attended = torch.cat(
            (
                clean_output,
                branch_output.view(batch, branch_width, self.heads, self.head_dim),
            ),
            dim=1,
        )
        return self.output(attended.reshape(batch, total_length, self.dim))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.up_gate = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        up, gate = self.up_gate(x).chunk(2, dim=-1)
        return self.down(up * F.silu(gate))


class SquaredReLU(nn.Module):
    """Parameter-matched two-matrix ReLU-squared feed-forward block."""

    def __init__(self, dim: int, swiglu_hidden_dim: int) -> None:
        super().__init__()
        numerator = 3 * swiglu_hidden_dim
        if numerator % 2:
            raise ValueError("parameter-matched ReLU-squared width must be integral")
        hidden_dim = numerator // 2
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.relu(self.up(x)).square())


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        ffn_dim: int,
        rope_theta: float,
        ffn_kind: str = "swiglu",
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.attention = PackedSelfAttention(dim, heads, rope_theta)
        self.ffn_norm = RMSNorm(dim)
        if ffn_kind == "swiglu":
            self.ffn = SwiGLU(dim, ffn_dim)
        elif ffn_kind == "relu_squared":
            self.ffn = SquaredReLU(dim, ffn_dim)
        else:
            raise ValueError(f"unsupported feed-forward kind {ffn_kind!r}")

    def forward(
        self,
        x: Tensor,
        *,
        positions: Tensor | None = None,
        allowed: Tensor | None = None,
        causal: bool = False,
    ) -> Tensor:
        x = x + self.attention(
            self.attention_norm(x), positions=positions, allowed=allowed, causal=causal
        )
        return x + self.ffn(self.ffn_norm(x))

    def forward_packed(
        self,
        x: Tensor,
        *,
        cu_seqlens: Tensor,
        positions: Tensor,
        max_seqlen: int,
        window: int | None = None,
        allow_dense_reference: bool = False,
    ) -> Tensor:
        x = x + self.attention.forward_packed(
            self.attention_norm(x),
            cu_seqlens=cu_seqlens,
            positions=positions,
            max_seqlen=max_seqlen,
            window=window,
            allow_dense_reference=allow_dense_reference,
        )
        return x + self.ffn(self.ffn_norm(x))

    def forward_clean_and_branches(
        self,
        x: Tensor,
        *,
        clean_valid: Tensor,
        positions: Tensor,
        layout,
        clean_window: int | None,
        allow_dense_reference: bool,
        assume_full_clean: bool = False,
        block_mask=None,
    ) -> Tensor:
        x = x + self.attention.forward_clean_and_branches(
            self.attention_norm(x),
            clean_valid=clean_valid,
            positions=positions,
            layout=layout,
            clean_window=clean_window,
            allow_dense_reference=allow_dense_reference,
            assume_full_clean=assume_full_clean,
            block_mask=block_mask,
        )
        return x + self.ffn(self.ffn_norm(x))

    def forward_document_branches(self, x: Tensor, **kwargs) -> Tensor:
        x = x + self.attention.forward_document_branches(
            self.attention_norm(x), **kwargs
        )
        return x + self.ffn(self.ffn_norm(x))

    def forward_shared_document_branches(self, x: Tensor, **kwargs) -> Tensor:
        clean_length = kwargs["clean_length"]
        clean = x[:, :clean_length]
        branch = x[:, clean_length:]
        normalized = torch.cat(
            (self.attention_norm(clean), self.attention_norm(branch)), dim=1
        )
        x = x + self.attention.forward_shared_document_branches(
            normalized, **kwargs
        )
        clean = x[:, :clean_length]
        branch = x[:, clean_length:]
        clean = clean + self.ffn(self.ffn_norm(clean))
        branch = branch + self.ffn(self.ffn_norm(branch))
        return torch.cat((clean, branch), dim=1)

    def forward_shared_document_branches_adaln(
        self,
        x: Tensor,
        branch_modulation: Tensor,
        **kwargs,
    ) -> Tensor:
        """Shared clean bank plus DiT-style AdaLN-Zero noisy branches.

        The clean prefix is deliberately independent of diffusion time and
        follows the ordinary transformer path. Only the revisable branch bank
        receives the six per-block shift/scale/gate values used by Duo's DiT
        reference implementation.
        """

        clean_length = kwargs["clean_length"]
        clean = x[:, :clean_length]
        branch = x[:, clean_length:]
        if branch_modulation.ndim != 3 or branch.shape[1] % branch_modulation.shape[1]:
            raise ValueError("branch AdaLN values must align with complete branches")
        branch_count = branch_modulation.shape[1]
        branch_width = branch.shape[1] // branch_count
        branch = branch.view(branch.shape[0], branch_count, branch_width, branch.shape[-1])
        parameters = tuple(
            value[:, :, None, :] for value in branch_modulation.chunk(6, dim=-1)
        )
        (
            attention_shift,
            attention_scale,
            attention_gate,
            ffn_shift,
            ffn_scale,
            ffn_gate,
        ) = parameters

        clean_normalized = self.attention_norm(clean)
        branch_normalized = self.attention_norm(branch)
        branch_normalized = branch_normalized * (1 + attention_scale) + attention_shift
        branch_normalized = branch_normalized.flatten(1, 2)
        attended = self.attention.forward_shared_document_branches(
            torch.cat((clean_normalized, branch_normalized), dim=1), **kwargs
        )
        clean_attended = attended[:, :clean_length]
        branch_attended = attended[:, clean_length:]
        clean = clean + clean_attended
        branch = branch + attention_gate * branch_attended.view_as(branch)

        clean_normalized = self.ffn_norm(clean)
        branch_normalized = self.ffn_norm(branch)
        branch_normalized = branch_normalized * (1 + ffn_scale) + ffn_shift
        combined_ffn = self.ffn(
            torch.cat((clean_normalized, branch_normalized.flatten(1, 2)), dim=1)
        )
        clean = clean + combined_ffn[:, :clean_length]
        branch = branch + ffn_gate * combined_ffn[:, clean_length:].view_as(branch)
        return torch.cat((clean, branch.flatten(1, 2)), dim=1)

    def prepare_clean_bank(
        self, clean: Tensor, **kwargs
    ) -> tuple[Tensor, CleanAttentionBank]:
        """Advance a time-independent clean stream and cache layer K/V."""

        attended, bank = self.attention.prepare_clean_bank(
            self.attention_norm(clean), **kwargs
        )
        clean = clean + attended
        clean = clean + self.ffn(self.ffn_norm(clean))
        return clean, bank

    def forward_branch_from_clean_bank_adaln(
        self,
        branch: Tensor,
        branch_modulation: Tensor,
        clean_bank: CleanAttentionBank,
        **kwargs,
    ) -> Tensor:
        """Run only the time-conditioned branch stream against cached K/V."""

        if branch_modulation.ndim != 3 or branch.shape[1] % branch_modulation.shape[1]:
            raise ValueError("branch AdaLN values must align with complete branches")
        branch_count = branch_modulation.shape[1]
        branch_width = branch.shape[1] // branch_count
        shaped = branch.view(
            branch.shape[0], branch_count, branch_width, branch.shape[-1]
        )
        (
            attention_shift,
            attention_scale,
            attention_gate,
            ffn_shift,
            ffn_scale,
            ffn_gate,
        ) = tuple(
            value[:, :, None, :] for value in branch_modulation.chunk(6, dim=-1)
        )
        normalized = self.attention_norm(shaped)
        normalized = normalized * (1 + attention_scale) + attention_shift
        attended = self.attention.forward_branch_from_clean_bank(
            normalized.flatten(1, 2), clean_bank, **kwargs
        ).view_as(shaped)
        shaped = shaped + attention_gate * attended
        normalized = self.ffn_norm(shaped)
        normalized = normalized * (1 + ffn_scale) + ffn_shift
        shaped = shaped + ffn_gate * self.ffn(normalized)
        return shaped.flatten(1, 2)

    def forward_packed_document_branches(self, x: Tensor, **kwargs) -> Tensor:
        x = x + self.attention.forward_packed_document_branches(
            self.attention_norm(x), **kwargs
        )
        return x + self.ffn(self.ffn_norm(x))


class ConditionedTransformerBlock(nn.Module):
    def __init__(
        self,
        local_dim: int,
        global_dim: int,
        heads: int,
        ffn_dim: int,
        rope_theta: float,
        conditioning: str,
        split_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.conditioning = conditioning
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.heads = heads
        self.head_dim = local_dim // heads
        self.split_residual_scale = float(split_residual_scale)
        if conditioning == "split_cross_attention":
            if global_dim % local_dim:
                raise ValueError(
                    "split cross-attention requires global width to be an "
                    "integer multiple of decoder width"
                )
            self.condition_splits = global_dim // local_dim
            # BLT's D_C transform preserves the global width and splits each
            # latent into k=h_G/h_D decoder-width keys. Query-dependent
            # attention over those keys is materially different from adding
            # one projected vector (BLT Eq. 11 and Fast BLT Fig. 3).
            self.condition = nn.Linear(global_dim, global_dim, bias=False)
            self.query_norm = RMSNorm(local_dim)
            self.condition_norm = RMSNorm(local_dim)
            self.cross_query = nn.Linear(local_dim, local_dim, bias=False)
            self.cross_key = nn.Linear(local_dim, local_dim, bias=False)
            self.cross_value = nn.Linear(local_dim, local_dim, bias=False)
            self.cross_output = nn.Linear(local_dim, local_dim, bias=False)
            self.register_parameter("condition_gate", None)
        elif conditioning == "gated_projection":
            self.condition_splits = 1
            self.condition = nn.Linear(global_dim, local_dim, bias=False)
            # The aligned BLT cross-attention has one latent key per byte, so
            # this control intentionally collapses reference split routing to
            # a projected residual. Start it at byte-embedding scale.
            self.condition_gate = nn.Parameter(
                torch.tensor(global_dim**-0.5, dtype=torch.float32)
            )
            self.query_norm = None
            self.condition_norm = None
        elif conditioning == "rmsnorm_projection":
            self.condition_splits = 1
            self.condition = nn.Linear(global_dim, local_dim, bias=False)
            # Retained only as the exact learning-control arm for the original
            # implementation that produced a roughly 35x condition residual.
            self.register_parameter("condition_gate", None)
            self.query_norm = None
            self.condition_norm = RMSNorm(local_dim)
        else:
            raise ValueError(f"unsupported conditioning {conditioning!r}")
        self.block = TransformerBlock(local_dim, heads, ffn_dim, rope_theta)

    def _add_condition(self, x: Tensor, condition: Tensor) -> Tensor:
        if self.conditioning == "split_cross_attention":
            if self.query_norm is None or self.condition_norm is None:
                raise AssertionError("split cross-attention norms are missing")
            keys = self.condition(condition).view(
                *condition.shape[:-1], self.condition_splits, self.local_dim
            )
            keys = self.condition_norm(keys)
            query = self.cross_query(self.query_norm(x)).view(
                *x.shape[:-1], self.heads, 1, self.head_dim
            )
            key = self.cross_key(keys).view(
                *keys.shape[:-2], self.condition_splits, self.heads, self.head_dim
            ).transpose(-3, -2)
            value = self.cross_value(keys).view(
                *keys.shape[:-2], self.condition_splits, self.heads, self.head_dim
            ).transpose(-3, -2)
            scores = (query * key).sum(-1) / math.sqrt(self.head_dim)
            weights = scores.softmax(-1)
            attended = (weights[..., None] * value).sum(-2).flatten(-2)
            return x + self.split_residual_scale * self.cross_output(attended)
        projected = self.condition(condition)
        if self.condition_gate is not None:
            return x + self.condition_gate.to(x.dtype) * projected
        if self.condition_norm is None:
            raise AssertionError("conditioner omitted both scale paths")
        return x + self.condition_norm(projected)

    def _add_projected_condition(self, x: Tensor, projected: Tensor) -> Tensor:
        """Apply a preprojected condition for repeated patch assignments."""

        if self.conditioning == "split_cross_attention":
            raise ValueError("split cross-attention cannot reuse projected residuals")
        if projected.shape != x.shape:
            raise ValueError("projected condition and decoder states must align")
        if self.condition_gate is not None:
            return x + self.condition_gate.to(x.dtype) * projected
        if self.condition_norm is None:
            raise AssertionError("conditioner omitted both scale paths")
        return x + self.condition_norm(projected)

    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        *,
        positions: Tensor | None = None,
        allowed: Tensor | None = None,
        causal: bool = False,
    ) -> Tensor:
        x = self._add_condition(x, condition)
        return self.block(x, positions=positions, allowed=allowed, causal=causal)

    def forward_packed(
        self,
        x: Tensor,
        condition: Tensor,
        *,
        cu_seqlens: Tensor,
        positions: Tensor,
        max_seqlen: int,
        window: int | None = None,
        allow_dense_reference: bool = False,
    ) -> Tensor:
        x = self._add_condition(x, condition)
        return self.block.forward_packed(
            x,
            cu_seqlens=cu_seqlens,
            positions=positions,
            max_seqlen=max_seqlen,
            window=window,
            allow_dense_reference=allow_dense_reference,
        )

    def forward_clean_and_branches(
        self,
        x: Tensor,
        condition: Tensor,
        *,
        clean_valid: Tensor,
        positions: Tensor,
        layout,
        clean_window: int | None,
        allow_dense_reference: bool,
        assume_full_clean: bool = False,
        block_mask=None,
    ) -> Tensor:
        x = self._add_condition(x, condition)
        return self.block.forward_clean_and_branches(
            x,
            clean_valid=clean_valid,
            positions=positions,
            layout=layout,
            clean_window=clean_window,
            allow_dense_reference=allow_dense_reference,
            assume_full_clean=assume_full_clean,
            block_mask=block_mask,
        )

    def forward_document_branches(
        self, x: Tensor, condition: Tensor, **kwargs
    ) -> Tensor:
        return self.block.forward_document_branches(
            self._add_condition(x, condition), **kwargs
        )

    def forward_shared_document_branches(
        self, x: Tensor, condition: Tensor, **kwargs
    ) -> Tensor:
        return self.block.forward_shared_document_branches(
            self._add_condition(x, condition), **kwargs
        )

    def forward_shared_document_branches_adaln(
        self,
        x: Tensor,
        condition: Tensor,
        branch_modulation: Tensor,
        **kwargs,
    ) -> Tensor:
        """Apply BLT patch conditioning, then branch-only AdaLN-Zero."""

        return self.block.forward_shared_document_branches_adaln(
            self._add_condition(x, condition),
            branch_modulation,
            **kwargs,
        )

    def prepare_clean_bank(
        self, clean: Tensor, condition: Tensor, **kwargs
    ) -> tuple[Tensor, CleanAttentionBank]:
        """Cache a decoder clean stream after its BLT condition projection."""

        return self.block.prepare_clean_bank(
            self._add_condition(clean, condition), **kwargs
        )

    def forward_branch_from_clean_bank_adaln(
        self,
        branch: Tensor,
        condition: Tensor,
        branch_modulation: Tensor,
        clean_bank: CleanAttentionBank,
        **kwargs,
    ) -> Tensor:
        return self.block.forward_branch_from_clean_bank_adaln(
            self._add_condition(branch, condition),
            branch_modulation,
            clean_bank,
            **kwargs,
        )

    def forward_shared_document_branches_projected(
        self, x: Tensor, projected_condition: Tensor, **kwargs
    ) -> Tensor:
        """Shared-bank decoder path with condition projection already fused."""

        return self.block.forward_shared_document_branches(
            self._add_projected_condition(x, projected_condition), **kwargs
        )

    def forward_packed_document_branches(
        self, x: Tensor, condition: Tensor, **kwargs
    ) -> Tensor:
        return self.block.forward_packed_document_branches(
            self._add_condition(x, condition), **kwargs
        )


class PatchPool(nn.Module):
    """BLT-style max initialized, within-patch Perceiver pooling."""

    def __init__(self, local_dim: int, global_dim: int, heads: int, stride: int) -> None:
        super().__init__()
        if global_dim % heads:
            raise ValueError("global width must divide pooling heads")
        self.stride = stride
        self.heads = heads
        self.head_dim = global_dim // heads
        self.local_norm = RMSNorm(local_dim)
        self.global_norm = RMSNorm(global_dim)
        self.max_projection = nn.Linear(local_dim, global_dim, bias=False)
        self.query = nn.Linear(global_dim, global_dim, bias=False)
        self.key_value = nn.Linear(local_dim, 2 * global_dim, bias=False)
        self.output = nn.Linear(global_dim, global_dim, bias=False)

    def _pool_padded(self, local: Tensor, patch_valid: Tensor) -> Tensor:
        """Pool ``[..., patches, max_patch, dim]`` padded patch banks."""

        if local.ndim < 3 or patch_valid.shape != local.shape[:-1]:
            raise ValueError("padded patch states and validity must align")
        if patch_valid.dtype != torch.bool:
            raise TypeError("padded patch validity must be boolean")
        has_valid = patch_valid.any(-1)
        pooled = local.masked_fill(~patch_valid[..., None], -torch.inf).amax(-2)
        pooled = torch.where(has_valid[..., None], pooled, torch.zeros_like(pooled))
        pooled = self.max_projection(pooled)
        query = self.query(self.global_norm(pooled)).view(
            *pooled.shape[:-1], self.heads, self.head_dim
        )
        key, value = self.key_value(local).chunk(2, dim=-1)
        key = key.view(*local.shape[:-1], self.heads, self.head_dim)
        value = value.view(*local.shape[:-1], self.heads, self.head_dim)
        scores = torch.einsum("...hd,...shd->...hs", query, key) / math.sqrt(
            self.head_dim
        )
        scores = scores.masked_fill(~patch_valid[..., None, :], -torch.inf)
        scores = torch.where(
            has_valid[..., None, None], scores, torch.zeros_like(scores)
        )
        weights = scores.softmax(-1)
        weights = torch.where(
            has_valid[..., None, None], weights, torch.zeros_like(weights)
        )
        attended = torch.einsum("...hs,...shd->...hd", weights, value).flatten(-2)
        return pooled + self.output(attended)

    def forward(self, local: Tensor, valid: Tensor) -> Tensor:
        if local.shape[:2] != valid.shape or valid.dtype != torch.bool:
            raise ValueError("local states and valid mask must align")
        batch, length, local_dim = local.shape
        if length % self.stride:
            raise ValueError("pooling length must be patch aligned")
        patches = length // self.stride
        normalized = self.local_norm(local).view(
            batch, patches, self.stride, local_dim
        )
        patch_valid = valid.view(batch, patches, self.stride)
        return self._pool_padded(normalized, patch_valid)

    def forward_packed(
        self,
        local: Tensor,
        patch_byte_cu_seqlens: Tensor,
        *,
        max_patch_size: int,
    ) -> Tensor:
        """Pool a document-packed variable-length patch partition.

        ``patch_byte_cu_seqlens`` addresses the packed local-state bank and
        therefore carries no artificial row padding.  A bounded gather lowers
        ragged patches to one static ``[patches, max_patch_size, dim]`` bank,
        which keeps the Perceiver pooling math identical to fixed stride.
        """

        if local.ndim != 2:
            raise ValueError("packed local states must be rank two")
        if (
            patch_byte_cu_seqlens.ndim != 1
            or patch_byte_cu_seqlens.dtype != torch.int32
            or patch_byte_cu_seqlens.numel() < 2
        ):
            raise ValueError("patch byte offsets must be nonempty rank-1 int32")
        if max_patch_size <= 0:
            raise ValueError("max_patch_size must be positive")
        if not torch.compiler.is_compiling():
            if int(patch_byte_cu_seqlens[0]) != 0 or int(
                patch_byte_cu_seqlens[-1]
            ) != local.shape[0]:
                raise ValueError("patch byte offsets must partition local states")
        starts = patch_byte_cu_seqlens[:-1].to(torch.long)
        lengths = torch.diff(patch_byte_cu_seqlens).to(torch.long)
        if not torch.compiler.is_compiling() and bool(
            ((lengths <= 0) | (lengths > max_patch_size)).any()
        ):
            raise ValueError("packed patch length lies outside the configured bound")
        offsets = torch.arange(max_patch_size, device=local.device)
        valid = offsets[None] < lengths[:, None]
        indices = starts[:, None] + offsets[None]
        indices = indices.clamp_max(local.shape[0] - 1)
        padded = self.local_norm(local).index_select(0, indices.reshape(-1)).view(
            starts.numel(), max_patch_size, local.shape[-1]
        )
        return self._pool_padded(padded, valid)
