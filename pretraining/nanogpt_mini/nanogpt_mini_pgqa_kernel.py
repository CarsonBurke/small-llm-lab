"""pretraining/nanogpt_mini/nanogpt_mini_pgqa_kernel.py

Fused Triton kernels for pointer-GQA attention (``nanogpt_mini_pgqa_model``).

Forward (one program per query): gathers each pointer's K/V row straight
from the shared ``[B*T, 2D]`` table into registers, runs an online softmax
over the ``P`` pointers for all query heads at once (``[HP, D] x [D, P]``
tensor-core dot, ``HP`` = heads padded to 16), and saves only the output
and the per-head logsumexp -- flash-attention layout, so no ``[B, T, P, D]``
candidate tensor ever exists and no activation checkpointing is needed.

Backward, query side (one program per query): recomputes the softmax from
``lse``, produces ``dq`` and ``d log_prior`` directly, and writes every
pointer's ``dK``/``dV`` contribution (``[P, D]`` each) to a ``[B, T, P, 2D]``
scratch -- plain stores, no atomics.

Backward, key side (one program per key): the inverse of the pointer map
is built once per call by a radix sort of the ``B*T*P`` flattened keys
(``key_segments``: ~2 ms for 8M keys); each key's program then reads
exactly its own scratch rows, unmasked, and sums them in fp32. No atomics
(the fp32 ``index_add_`` of the PyTorch path was 33% of the step under
256-way contention) and no candidate scan (an enumeration of the
``2**k * T/W`` candidate queries per pointer was measured at 16x the
useful loads). The batched ``[4, 128] x [128, 256]`` matmuls of the
PyTorch path (~1% tensor-core utilization, 25%) are gone with it.

Numerics: scores and softmax in fp32 regardless of the input dtype; with
fp32 inputs the dots run in IEEE precision, which is what the equivalence
test against the PyTorch reference uses.
"""

from __future__ import annotations

import torch
from torch import Tensor
import triton
import triton.language as tl

# Finite "minus infinity" for masked scores: keeps every exp() finite even
# for all-masked chunks (those rows contribute exactly 0 mass).
NEG_FILL = tl.constexpr(-1e30)


@triton.jit
def _pgqa_fwd(
    q_ptr, kv_ptr, pos_ptr, lp_ptr, out_ptr, lse_ptr,
    T, R, scale,
    H: tl.constexpr, HP: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
    BLOCK_P: tl.constexpr, IEEE: tl.constexpr,
):
    bt = tl.program_id(0)
    b = bt // T
    h = tl.arange(0, HP)
    d = tl.arange(0, D)
    hmask = h < H
    q = tl.load(q_ptr + bt * H * D + h[:, None] * D + d[None, :], mask=hmask[:, None], other=0.0)
    m_i = tl.full([HP], NEG_FILL, tl.float32)
    l_i = tl.zeros([HP], tl.float32)
    acc = tl.zeros([HP, D], tl.float32)
    for start in range(0, P, BLOCK_P):
        p = start + tl.arange(0, BLOCK_P)
        pos = tl.load(pos_ptr + bt * P + p)
        lp = tl.load(lp_ptr + bt * P + p)
        valid = pos >= 0
        row = b * R + tl.where(valid, pos, 0)
        kv_off = row[:, None] * (2 * D) + d[None, :]
        k = tl.load(kv_ptr + kv_off, mask=valid[:, None], other=0.0)
        if IEEE:
            s = tl.dot(q, tl.trans(k), input_precision="ieee")
        else:
            s = tl.dot(q, tl.trans(k))
        s = s * scale + lp[None, :]
        s = tl.where(valid[None, :], s, NEG_FILL)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        e = tl.exp(s - m_new[:, None])
        e = tl.where(valid[None, :], e, 0.0)
        l_i = l_i * alpha + tl.sum(e, 1)
        acc = acc * alpha[:, None]
        v = tl.load(kv_ptr + kv_off + D, mask=valid[:, None], other=0.0)
        if IEEE:
            acc += tl.dot(e.to(v.dtype), v, input_precision="ieee")
        else:
            acc += tl.dot(e.to(v.dtype), v)
        m_i = m_new
    has_mass = l_i > 0
    l_safe = tl.where(has_mass, l_i, 1.0)
    out = acc / l_safe[:, None]
    tl.store(out_ptr + bt * H * D + h[:, None] * D + d[None, :], out.to(out_ptr.dtype.element_ty),
             mask=hmask[:, None])
    lse = tl.where(has_mass, m_i + tl.log(l_safe), 0.0)
    tl.store(lse_ptr + bt * H + h, lse, mask=hmask)


@triton.jit
def _pgqa_bwd_q(
    q_ptr, kv_ptr, pos_ptr, lp_ptr, out_ptr, do_ptr, lse_ptr,
    dq_ptr, dlp_ptr, scratch_ptr,
    T, R, scale,
    H: tl.constexpr, HP: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
    BLOCK_P: tl.constexpr, IEEE: tl.constexpr,
):
    bt = tl.program_id(0)
    b = bt // T
    h = tl.arange(0, HP)
    d = tl.arange(0, D)
    hmask = h < H
    hd_off = bt * H * D + h[:, None] * D + d[None, :]
    q = tl.load(q_ptr + hd_off, mask=hmask[:, None], other=0.0)
    o = tl.load(out_ptr + hd_off, mask=hmask[:, None], other=0.0)
    do = tl.load(do_ptr + hd_off, mask=hmask[:, None], other=0.0)
    lse = tl.load(lse_ptr + bt * H + h, mask=hmask, other=0.0)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
    dq = tl.zeros([HP, D], tl.float32)
    for start in range(0, P, BLOCK_P):
        p = start + tl.arange(0, BLOCK_P)
        pos = tl.load(pos_ptr + bt * P + p)
        lp = tl.load(lp_ptr + bt * P + p)
        valid = pos >= 0
        row = b * R + tl.where(valid, pos, 0)
        kv_off = row[:, None] * (2 * D) + d[None, :]
        k = tl.load(kv_ptr + kv_off, mask=valid[:, None], other=0.0)
        v = tl.load(kv_ptr + kv_off + D, mask=valid[:, None], other=0.0)
        if IEEE:
            s = tl.dot(q, tl.trans(k), input_precision="ieee")
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        else:
            s = tl.dot(q, tl.trans(k))
            dp = tl.dot(do, tl.trans(v))
        s = s * scale + lp[None, :]
        w = tl.exp(s - lse[:, None])
        w = tl.where(valid[None, :], w, 0.0)
        ds = w * (dp - delta[:, None])  # padded heads: w finite, dp = delta = 0 -> ds = 0
        if IEEE:
            dq += tl.dot(ds.to(k.dtype), k, input_precision="ieee")
            dk_c = tl.dot(tl.trans(ds).to(q.dtype), q, input_precision="ieee")
            dv_c = tl.dot(tl.trans(w).to(do.dtype), do, input_precision="ieee")
        else:
            dq += tl.dot(ds.to(k.dtype), k)
            dk_c = tl.dot(tl.trans(ds).to(q.dtype), q)
            dv_c = tl.dot(tl.trans(w).to(do.dtype), do)
        sc_off = (bt * P + p).to(tl.int64)[:, None] * (2 * D) + d[None, :]
        tl.store(scratch_ptr + sc_off, (dk_c * scale).to(scratch_ptr.dtype.element_ty), mask=valid[:, None])
        tl.store(scratch_ptr + sc_off + D, dv_c.to(scratch_ptr.dtype.element_ty), mask=valid[:, None])
        tl.store(dlp_ptr + bt * P + p, tl.sum(ds, 0))
    tl.store(dq_ptr + hd_off, (dq * scale).to(dq_ptr.dtype.element_ty), mask=hmask[:, None])


@triton.jit
def _pgqa_bwd_kv(
    order_ptr, starts_ptr, scratch_ptr, dkv_ptr,
    D: tl.constexpr, BLOCK_R: tl.constexpr,
):
    bj = tl.program_id(0)
    seg_start = tl.load(starts_ptr + bj)
    seg_end = tl.load(starts_ptr + bj + 1)
    d = tl.arange(0, D)
    acc_k = tl.zeros([D], tl.float32)
    acc_v = tl.zeros([D], tl.float32)
    for i in range(seg_start, seg_end, BLOCK_R):
        idx = i + tl.arange(0, BLOCK_R)
        inseg = idx < seg_end
        rows = tl.load(order_ptr + idx, mask=inseg, other=0)
        sc_off = rows[:, None] * (2 * D) + d[None, :]
        rk = tl.load(scratch_ptr + sc_off, mask=inseg[:, None], other=0.0)
        rv = tl.load(scratch_ptr + sc_off + D, mask=inseg[:, None], other=0.0)
        acc_k += tl.sum(rk.to(tl.float32), 0)
        acc_v += tl.sum(rv.to(tl.float32), 0)
    tl.store(dkv_ptr + bj * (2 * D) + d, acc_k.to(dkv_ptr.dtype.element_ty))
    tl.store(dkv_ptr + bj * (2 * D) + D + d, acc_v.to(dkv_ptr.dtype.element_ty))


def key_segments(pos: Tensor, rows: int) -> tuple[Tensor, Tensor]:
    """Inverse of the pointer map for the key-side backward.

    ``pos`` int32 ``[B, T, P]`` (``-1`` invalid) indexing a ``rows``-row
    table per batch element. Returns ``(order, starts)``: ``order`` int64
    ``[B*T*P]`` lists flattened ``(b, t, p)`` entries sorted by the row they
    read (invalid entries last), ``starts`` int64 ``[B*rows + 1]`` gives row
    ``b*rows + j`` the slice ``order[starts[k]:starts[k + 1]]``. A radix sort
    of ``B*T*P`` keys; every valid entry is then read exactly once,
    unmasked, by its row's program (no atomics, no candidate scan).
    """
    B = pos.size(0)
    keys = torch.where(pos >= 0, pos + (torch.arange(B, device=pos.device, dtype=pos.dtype) * rows).view(B, 1, 1),
                       B * rows).reshape(-1)
    sorted_keys, order = torch.sort(keys)
    starts = torch.searchsorted(sorted_keys, torch.arange(B * rows + 1, device=pos.device, dtype=pos.dtype))
    return order, starts


def _check(q: Tensor, kv: Tensor, pos: Tensor, log_prior: Tensor) -> tuple[int, int, int, int, int, int]:
    B, T, H, D = q.shape
    P = pos.size(-1)
    R = kv.size(1)
    if kv.shape != (B, R, 2 * D):
        raise ValueError(f"kv must be [B, R, 2D] with D = {D}, got {tuple(kv.shape)}")
    if pos.shape != (B, T, P) or pos.dtype != torch.int32:
        raise ValueError("pos must be int32 [B, T, P] with -1 for invalid pointers")
    if log_prior.shape != (B, T, P) or log_prior.dtype != torch.float32:
        raise ValueError("log_prior must be fp32 [B, T, P]")
    if D & (D - 1) or D < 16 or P & (P - 1) or P < 16:
        raise ValueError(f"D and P must be powers of two >= 16, got D={D} P={P}")
    return B, T, H, D, P, R


class FusedPointerAttention(torch.autograd.Function):
    """``gqa_attend`` semantics on the fused kernels.

    ``q`` ``[B, T, H, D]``; ``kv`` ``[B, R, 2D]`` (K then V; ``R`` rows per
    batch element, ``R == T`` for plain token tables); ``pos`` int32
    ``[B, T, P]`` with ``-1`` for invalid; ``log_prior`` fp32 ``[B, T, P]``.
    """

    BLOCK_P = 64
    BLOCK_R = 32

    @staticmethod
    def forward(ctx, q: Tensor, kv: Tensor, pos: Tensor, log_prior: Tensor, scale: float) -> Tensor:
        B, T, H, D, P, R = _check(q, kv, pos, log_prior)
        q, kv, pos, log_prior = q.contiguous(), kv.contiguous(), pos.contiguous(), log_prior.contiguous()
        out = torch.empty_like(q)
        lse = torch.empty(B, T, H, dtype=torch.float32, device=q.device)
        HP = max(16, triton.next_power_of_2(H))
        _pgqa_fwd[(B * T,)](
            q, kv, pos, log_prior, out, lse, T, R, scale,
            H=H, HP=HP, D=D, P=P, BLOCK_P=min(FusedPointerAttention.BLOCK_P, P),
            IEEE=q.dtype == torch.float32,
        )
        ctx.save_for_backward(q, kv, pos, log_prior, out, lse)
        ctx.scale, ctx.HP = scale, HP
        return out

    @staticmethod
    def backward(ctx, do: Tensor):
        q, kv, pos, log_prior, out, lse = ctx.saved_tensors
        B, T, H, D = q.shape
        P = pos.size(-1)
        R = kv.size(1)
        do = do.contiguous()
        dq = torch.empty_like(q)
        dlp = torch.empty_like(log_prior)
        scratch = torch.empty(B, T, P, 2 * D, dtype=kv.dtype, device=q.device)
        _pgqa_bwd_q[(B * T,)](
            q, kv, pos, log_prior, out, do, lse, dq, dlp, scratch, T, R, ctx.scale,
            H=H, HP=ctx.HP, D=D, P=P, BLOCK_P=min(FusedPointerAttention.BLOCK_P, P),
            IEEE=q.dtype == torch.float32,
        )
        order, starts = key_segments(pos, R)
        dkv = torch.empty_like(kv)
        _pgqa_bwd_kv[(B * R,)](order, starts, scratch, dkv, D=D, BLOCK_R=FusedPointerAttention.BLOCK_R)
        return dq, dkv, None, dlp, None


def fused_attend(q: Tensor, kv: Tensor, pos: Tensor, log_prior: Tensor, scale: float) -> Tensor:
    return FusedPointerAttention.apply(q, kv, pos, log_prior, scale)
