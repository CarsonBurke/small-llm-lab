"""Core byte-diffusion layers with packed projections and absolute RoPE."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

def pack_valid(values: Tensor, valid: Tensor) -> Tensor:
    """Pack leading dimensions selected by ``valid`` without index-put ops."""

    if values.shape[: valid.ndim] != valid.shape or valid.dtype != torch.bool:
        raise ValueError("valid mask must match the leading value dimensions")
    trailing = values.shape[valid.ndim :]
    expanded = valid.reshape(*valid.shape, *([1] * len(trailing))).expand_as(values)
    return torch.masked_select(values, expanded).reshape(-1, *trailing)


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
    return torch.masked_scatter(
        torch.zeros_like(template), expanded, packed.reshape(-1)
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


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

        cu = torch.cat(
            (torch.zeros(1, dtype=torch.int32, device=x.device), lengths.cumsum(0).to(torch.int32))
        )
        if assume_full_clean:
            clean_q = q[:, :, :clean_length].transpose(1, 2).reshape(
                batch * clean_length, self.heads, self.head_dim
            )
            clean_k = k[:, :, :clean_length].transpose(1, 2).reshape_as(clean_q)
            clean_v = v[:, :, :clean_length].transpose(1, 2).reshape_as(clean_q)
            clean_positions = positions[:, :clean_length].reshape(-1)
        else:
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
        attended = torch.cat((clean_output, branch_output), dim=1)
        return self.output(attended.reshape(batch, total_length, self.dim))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.up_gate = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        up, gate = self.up_gate(x).chunk(2, dim=-1)
        return self.down(up * F.silu(gate))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, rope_theta: float) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.attention = PackedSelfAttention(dim, heads, rope_theta)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)

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


class ConditionedTransformerBlock(nn.Module):
    def __init__(
        self,
        local_dim: int,
        global_dim: int,
        heads: int,
        ffn_dim: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.condition = nn.Linear(global_dim, local_dim, bias=False)
        self.condition_norm = RMSNorm(local_dim)
        self.block = TransformerBlock(local_dim, heads, ffn_dim, rope_theta)

    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        *,
        positions: Tensor | None = None,
        allowed: Tensor | None = None,
        causal: bool = False,
    ) -> Tensor:
        x = x + self.condition_norm(self.condition(condition))
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
        x = x + self.condition_norm(self.condition(condition))
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
        x = x + self.condition_norm(self.condition(condition))
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
        self.key = nn.Linear(local_dim, global_dim, bias=False)
        self.value = nn.Linear(local_dim, global_dim, bias=False)
        self.output = nn.Linear(global_dim, global_dim, bias=False)

    def forward(self, local: Tensor, valid: Tensor) -> Tensor:
        if local.shape[:2] != valid.shape or valid.dtype != torch.bool:
            raise ValueError("local states and valid mask must align")
        batch, length, local_dim = local.shape
        if length % self.stride:
            raise ValueError("pooling length must be patch aligned")
        patches = length // self.stride
        local = self.local_norm(local).view(batch, patches, self.stride, local_dim)
        patch_valid = valid.view(batch, patches, self.stride)
        has_valid = patch_valid.any(-1)
        pooled = local.masked_fill(~patch_valid[..., None], -torch.inf).amax(2)
        pooled = torch.where(has_valid[..., None], pooled, torch.zeros_like(pooled))
        pooled = self.max_projection(pooled)
        query = self.query(self.global_norm(pooled)).view(
            batch, patches, self.heads, self.head_dim
        )
        key = self.key(local).view(batch, patches, self.stride, self.heads, self.head_dim)
        value = self.value(local).view(batch, patches, self.stride, self.heads, self.head_dim)
        scores = torch.einsum("bphd,bpshd->bphs", query, key) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~patch_valid[:, :, None, :], -torch.inf)
        scores = torch.where(has_valid[:, :, None, None], scores, torch.zeros_like(scores))
        weights = scores.softmax(-1)
        weights = torch.where(has_valid[:, :, None, None], weights, torch.zeros_like(weights))
        attended = torch.einsum("bphs,bpshd->bphd", weights, value).flatten(-2)
        return pooled + self.output(attended)
