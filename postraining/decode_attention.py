"""One-token decode attention over each row's live key range (CUDA, Triton).

Lockstep decode left-pads every prompt of a pool to the pool's longest one,
so row ``b`` attends cache slots ``[starts[b], position]`` and nothing
before them. A boolean ``attn_mask`` states that correctly but still makes
SDPA read every masked slot, and it forces the memory-efficient backend.
Measured on the production mixture, the median prompt is 17 tokens while a
64-prompt pool pads to ~190, so most of a decode step's prompt KV reads
were padding, and the tail past the write head cost more reads until the
width bucket ended.

``ranged_decode_attention`` reads exactly the live range. It is
flash-decoding: the range is split across programs when rows x heads alone
cannot fill the GPU (small row buckets), each program keeps an online
softmax in fp32 registers, and a second kernel merges the partials by their
log-sum-exp. Scores, softmax and the value sum run in fp32 on the cache's
values, the same arithmetic contract as SDPA's fused kernels. Registered as a
``torch.library.triton_op`` so it traces into a compiled decode step, and
captured CUDA graphs replay it with ``starts``/``position`` read on device.
The split count is a runtime value, never a compile-time constant, so a
step compiled over a symbolic row count stays one graph across every row
bucket.

Contract: every row's range is non-empty (``starts[b] <= position``); the
lockstep decode guarantees it because a row's own token is written at
``position`` before it attends.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor

# Bare `wrap_triton`, not `torch.library.wrap_triton`: AOTAutograd's cache
# finds a triton_op's kernels by matching this spelling in its source, and a
# kernel it misses is absent from the cache key, so an edited kernel would be
# served its stale compiled graph.
from torch.library import wrap_triton


@triton.jit
def _ranged_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    starts_ptr,
    position_ptr,
    acc_ptr,
    stats_ptr,
    heads,
    splits,
    scale,
    stride_qb,
    stride_qh,
    stride_kb,
    stride_kh,
    stride_kl,
    stride_vb,
    stride_vh,
    stride_vl,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_head = tl.program_id(0)
    split = tl.program_id(1)
    # int64: row x row-stride passes 2**31 at production row counts once the
    # cache is a few thousand slots wide.
    row = (row_head // heads).to(tl.int64)
    head = row_head % heads
    start = tl.load(starts_ptr + row)
    stop = tl.load(position_ptr) + 1
    # Splits own whole blocks of the range; trailing splits may own none.
    per_split = tl.cdiv(tl.cdiv(stop - start, splits), BLOCK_N) * BLOCK_N
    lo = start + split * per_split
    hi = tl.minimum(lo + per_split, stop)

    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + row * stride_qb + head * stride_qh + offs_d).to(tl.float32)
    # Inductor may hand a Python float over as fp64; keep the math fp32.
    q = (q * scale).to(tl.float32)
    k_base = k_ptr + row * stride_kb + head * stride_kh
    v_base = v_ptr + row * stride_vb + head * stride_vh
    m = tl.full((), float("-inf"), tl.float32)
    l = tl.zeros((), tl.float32)
    acc = tl.zeros((D,), tl.float32)
    for block in range(lo, hi, BLOCK_N):
        offs_n = block + tl.arange(0, BLOCK_N)
        valid = offs_n < hi
        k = tl.load(
            k_base + offs_n[:, None] * stride_kl + offs_d[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.where(valid, tl.sum(k * q[None, :], axis=1), float("-inf"))
        # Every block holds at least one valid key, so m_new is finite and
        # the first block's rescale is exp(-inf) = 0, never NaN.
        m_new = tl.maximum(m, tl.max(scores, axis=0))
        alpha = tl.exp(m - m_new)
        p = tl.exp(scores - m_new)
        v = tl.load(
            v_base + offs_n[:, None] * stride_vl + offs_d[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l = l * alpha + tl.sum(p, axis=0)
        m = m_new

    partial = row_head.to(tl.int64) * splits + split
    tl.store(acc_ptr + partial * D + offs_d, acc)
    tl.store(stats_ptr + partial * 2, m)
    tl.store(stats_ptr + partial * 2 + 1, l)


@triton.jit
def _merge_splits_kernel(
    acc_ptr,
    stats_ptr,
    out_ptr,
    heads,
    splits,
    stride_ob,
    stride_oh,
    MAX_SPLITS: tl.constexpr,
    D: tl.constexpr,
):
    row_head = tl.program_id(0)
    row = (row_head // heads).to(tl.int64)
    head = row_head % heads
    offs_s = tl.arange(0, MAX_SPLITS)
    present = offs_s < splits
    partial = row_head.to(tl.int64) * splits + offs_s
    m = tl.load(stats_ptr + partial * 2, mask=present, other=float("-inf"))
    l = tl.load(stats_ptr + partial * 2 + 1, mask=present, other=0.0)
    # Empty splits carry m = -inf, l = 0 and weigh exactly zero; the range
    # is non-empty, so the maximum is finite.
    weight = tl.exp(m - tl.max(m, axis=0))
    offs_d = tl.arange(0, D)
    acc = tl.load(
        acc_ptr + partial[:, None] * D + offs_d[None, :],
        mask=present[:, None],
        other=0.0,
    )
    out = tl.sum(weight[:, None] * acc, axis=0) / tl.sum(weight * l, axis=0)
    tl.store(
        out_ptr + row * stride_ob + head * stride_oh + offs_d,
        out.to(out_ptr.dtype.element_ty),
    )


_BLOCK_N = 64
_NUM_WARPS = 4
# Programs per SM the split count aims for: enough resident work to cover
# memory latency when rows x heads alone would leave most SMs idle.
_PROGRAMS_PER_SM = 4
# The merge kernel's static tile over splits.
_MAX_SPLITS = 32


@functools.cache
def _multiprocessors(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _split_count(programs, cache_width, device: torch.device):
    """Splits per (row, head); symbolic in, symbolic out.

    ``torch.sym_min``/``sym_max`` keep a symbolic row count symbolic where
    the builtins would guard on it and recompile the step per row bucket.
    Sized by the cache width because the live range is only known on device.
    """
    target = _PROGRAMS_PER_SM * _multiprocessors(device.index or 0)
    wanted = torch.sym_min(-(-target // programs), -(-cache_width // (2 * _BLOCK_N)))
    return torch.sym_max(1, torch.sym_min(wanted, _MAX_SPLITS))


@torch.library.triton_op("nanogpt_attn::ranged_decode", mutates_args={})
def _ranged_decode(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    starts: Tensor,
    position: Tensor,
    scale: float,
) -> Tensor:
    batch, heads, head_dim = q.shape
    splits = _split_count(batch * heads, k_cache.size(2), q.device)
    # Always partials plus a merge, even for one split: a branch on the
    # split count would guard on it. At one split the merge rereads
    # [B*H, D] fp32, against a whole live range of bf16 K and V.
    partials = batch * heads * splits
    acc = torch.empty((partials, head_dim), device=q.device, dtype=torch.float32)
    stats = torch.empty((partials, 2), device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)
    wrap_triton(_ranged_decode_kernel)[(batch * heads, splits)](
        q,
        k_cache,
        v_cache,
        starts,
        position,
        acc,
        stats,
        heads,
        splits,
        scale,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        D=head_dim,
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
    )
    wrap_triton(_merge_splits_kernel)[(batch * heads,)](
        acc,
        stats,
        out,
        heads,
        splits,
        out.stride(0),
        out.stride(1),
        MAX_SPLITS=_MAX_SPLITS,
        D=head_dim,
        num_warps=_NUM_WARPS,
    )
    return out


def ranged_decode_attention_reference(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    starts: Tensor,
    position: Tensor,
    scale: float,
) -> Tensor:
    """The same attention as a boolean-masked SDPA over the whole cache."""
    keys = torch.arange(k_cache.size(2), device=q.device)
    mask = (keys[None, :] >= starts[:, None]) & (keys[None, :] <= position)
    return F.scaled_dot_product_attention(
        q[:, :, None], k_cache, v_cache, attn_mask=mask[:, None, None, :],
        scale=scale,
    ).squeeze(2)


def ranged_decode_attention(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    starts: Tensor,
    position: Tensor,
    scale: float,
) -> Tensor:
    """Row ``b`` attends cache slots ``[starts[b], position]``.

    ``q`` [B, H, D]; caches [B, H, L, D] (any leading-row slice of a larger
    cache); ``starts`` [B] int64; ``position`` a 0-dim int64 tensor. Returns
    [B, H, D] in ``q``'s dtype. The Triton kernel serves CUDA inputs whose
    head vectors and ``starts`` are unit-stride with a power-of-two head dim;
    everything else takes the masked-SDPA reference.
    """
    if starts.dim() != 1 or position.dim() != 0:
        raise ValueError("starts must be (batch,) and position 0-dim")
    head_dim = q.size(-1)
    if (
        q.is_cuda
        and head_dim >= 16
        and head_dim & (head_dim - 1) == 0
        and starts.stride(0) == 1
        and q.stride(-1) == k_cache.stride(-1) == v_cache.stride(-1) == 1
    ):
        return _ranged_decode(q, k_cache, v_cache, starts, position, scale)
    return ranged_decode_attention_reference(
        q, k_cache, v_cache, starts, position, scale
    )
