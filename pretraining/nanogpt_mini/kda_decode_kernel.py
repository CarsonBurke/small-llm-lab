"""Fused one-token KDA delta-rule decode step (CUDA, Triton).

The decode recurrence is memory-bound on its fp32 state (``[B, H, Dv, Dk]``,
201 MB per layer at 1024 rows of the 3x128 production mixer). Written as
PyTorch ops it makes several passes over that state per step -- decay,
the ``S k`` contraction, the rank-one update, the copy back into the cache,
the ``S q`` contraction -- and under autocast the two einsums ran as bf16
GEMMs over a bf16 copy of the state. Every quantity the step needs is local
to one value row of the state, so one program per (row, head, value block)
loads its block once, does all of it in fp32 registers and stores once: one
read and one write of the state, the traffic floor for this recurrence.

Semantics are exactly ``kda_recurrent_step``'s, on already-normalized
``q``/``k`` (see ``nanogpt_mini_kda_model``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

# Bare `wrap_triton`, not `torch.library.wrap_triton`: AOTAutograd's cache
# finds a triton_op's kernels by matching this spelling in its source, and a
# kernel it misses is absent from the cache key, so an edited kernel would be
# served its stale compiled graph.
from torch.library import wrap_triton


@triton.jit
def _kda_decode_step_kernel(
    state_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    gate_ptr,
    beta_ptr,
    out_ptr,
    heads,
    stride_sb,
    stride_sh,
    stride_sv,
    stride_sk,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_vb,
    stride_vh,
    stride_gb,
    stride_gh,
    stride_bb,
    stride_bh,
    stride_ob,
    stride_oh,
    DK: tl.constexpr,
    BV: tl.constexpr,
):
    row_head = tl.program_id(0)
    block = tl.program_id(1)
    row = (row_head // heads).to(tl.int64)  # row x state stride can pass 2**31
    head = row_head % heads
    offs_k = tl.arange(0, DK)
    offs_v = block * BV + tl.arange(0, BV)

    state_ptrs = (
        state_ptr
        + row * stride_sb
        + head * stride_sh
        + offs_v[:, None] * stride_sv
        + offs_k[None, :] * stride_sk
    )
    state = tl.load(state_ptrs)
    q = tl.load(q_ptr + row * stride_qb + head * stride_qh + offs_k)
    k = tl.load(k_ptr + row * stride_kb + head * stride_kh + offs_k)
    gate = tl.load(gate_ptr + row * stride_gb + head * stride_gh + offs_k)
    v = tl.load(v_ptr + row * stride_vb + head * stride_vh + offs_v)
    beta = tl.load(beta_ptr + row * stride_bb + head * stride_bh)

    decayed = state * tl.exp(gate)[None, :]
    delta = v - tl.sum(decayed * k[None, :], axis=1)
    updated = decayed + beta * delta[:, None] * k[None, :]
    tl.store(state_ptrs, updated)
    out = tl.sum(updated * q[None, :], axis=1)
    tl.store(out_ptr + row * stride_ob + head * stride_oh + offs_v, out)


# 32 value rows x 128 keys of fp32 = 16 KiB per program: enough in flight to
# saturate bandwidth at every production row count, and 4 blocks per head
# keep the grid at >= 96 programs even for an 8-row tail bucket.
_BLOCK_V = 32
_NUM_WARPS = 4


@torch.library.triton_op("nanogpt_kda::decode_step", mutates_args={"state"})
def kda_decode_step(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gate: Tensor,
    beta: Tensor,
    state: Tensor,
) -> Tensor:
    """``state`` [B, H, Dv, Dk] fp32 updated in place; returns o [B, H, Dv].

    q/k: [B, H, Dk] fp32, already l2-normalized (q also scaled); v: [B, H,
    Dv] fp32; gate: [B, H, Dk] fp32 log-space decay; beta: [B, H] fp32.
    Head vectors must be unit-stride in their last dimension.
    """
    batch, heads, value_dim, key_dim = state.shape
    out = torch.empty((batch, heads, value_dim), device=q.device, dtype=torch.float32)
    wrap_triton(_kda_decode_step_kernel)[
        (batch * heads, value_dim // _BLOCK_V)
    ](
        state,
        q,
        k,
        v,
        gate,
        beta,
        out,
        heads,
        *state.stride(),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        gate.stride(0),
        gate.stride(1),
        beta.stride(0),
        beta.stride(1),
        out.stride(0),
        out.stride(1),
        DK=key_dim,
        BV=_BLOCK_V,
        num_warps=_NUM_WARPS,
    )
    return out


def fused_decode_supported(state: Tensor) -> bool:
    """Shapes the kernel's static tiling covers (the production mixer's)."""
    _, _, value_dim, key_dim = state.shape
    return (
        state.is_cuda
        and state.dtype == torch.float32
        and value_dim % _BLOCK_V == 0
        and key_dim & (key_dim - 1) == 0
        and key_dim >= 16
    )
