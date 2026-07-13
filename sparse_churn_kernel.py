"""Dedicated Triton kernel for the churn tail: gate + draws + dedup + stats.

One program per query row.  Replaces the Inductor-compiled torch tail
(special.entr + rand + where + broadcast [KB, KB] dedup ~= 3.35 ms/layer at
production shape) with a single pass at ~2.5 ms/layer, warps=1 (the [KB, KB]
dedup reduction is shuffle-only inside one warp; warps=2 is ~20% slower from
cross-warp layout conversions).  Fusing this INTO the attention forward was
measured and rejected: the K x K dedup alongside live attention registers
costs +5.5 ms/layer (register pressure + 2D-reduction conversions).

Semantics are exactly ``sparse_churn_train_gpt._churn`` (the torch reference
the tests check against):

    q_rewire = clamp(max(H/log(m+1), p_null), CHURN_EPS, 1)
    eligible = (p <= 1/(m+1)) & (slot != 0)
    rewire   ~ Bernoulli(q_rewire) on eligible; draws ~ U{0..t}
    dedup: kept slots always beat rewired candidates; earliest candidate wins

Randomness is Philox (``tl.rand``) keyed by a device seed tensor plus the
layer index: the trainer bumps the seed once per train forward, eval passes a
fixed seed - deterministic eval, zero RNG state consumed, and no per-step
host scalar leaking into the compiled graph as a guard.

Stats come back as a fresh [5] tensor of sums (rewires, support slots, valid
slots, H_norm, rows) so the op stays functionally pure.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

_STATS_BANKS = 64


@triton.jit
def _churn_tail_kernel(
    IDX, VALID, P, PN, SEED, NIDX, NVALID, STATS,
    eps, T, LI,
    KB: tl.constexpr,
    DO_STATS: tl.constexpr,
    BANKS: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid % T
    offs_k = tl.arange(0, KB)
    row = pid * KB

    idx_raw = tl.load(IDX + row + offs_k)
    val = tl.load(VALID + row + offs_k) != 0
    p = tl.load(P + row + offs_k)
    pn = tl.load(PN + pid)

    m_incl = tl.sum(val.to(tl.float32), axis=0) + 1.0
    on_sup = val & (p > 0)
    plog = tl.where(on_sup, p * tl.log(tl.maximum(p, 1e-30)), 0.0)
    h_row = -tl.sum(plog, axis=0) - tl.where(pn > 0, pn * tl.log(tl.maximum(pn, 1e-30)), 0.0)
    h_norm = h_row / tl.maximum(tl.log(m_incl), 0.6931471805599453)
    q_rw = tl.minimum(tl.maximum(tl.maximum(h_norm, pn), eps), 1.0)

    share = 1.0 / m_incl
    eligible = (p <= share) & (offs_k != 0)  # slot 0 (self) never rewires

    seed = (tl.load(SEED) + LI * 1000003).to(tl.int32)
    lane = row + offs_k
    u_gate = tl.rand(seed, lane)
    u_draw = tl.rand(seed ^ 0x5851F42D, lane)
    rewire = eligible & (u_gate < q_rw)
    # causality relies on tl.rand < 1.0 (Triton folds int32 Philox to
    # [0, 2^31-1] * fp32(2^-31) -> max 1 - 2^-24) AND on u*(t+1) never
    # rounding up to t+1 (exact product <= n - n/2^24, below the rounding
    # boundary for every n <= 2^24).  A Triton change letting tl.rand reach
    # 1.0 would silently wire FUTURE tokens and read OOB at t = T-1.
    draws = (u_draw * (t + 1).to(tl.float32)).to(tl.int32)

    new_idx = tl.where(rewire, draws, idx_raw)  # dead slots keep their stored idx
    new_val = val | rewire

    # dedup in registers: kept slots always beat rewired candidates at the same
    # index; among candidates the earliest slot wins; never kills all copies
    eq = new_idx[:, None] == new_idx[None, :]
    earlier = offs_k[None, :] < offs_k[:, None]
    wins = new_val[None, :] & (~rewire[None, :] | earlier)
    dup = tl.sum((eq & wins).to(tl.int32), axis=1) > 0
    new_val = new_val & ~(rewire & dup)

    tl.store(NIDX + row + offs_k, new_idx)
    tl.store(NVALID + row + offs_k, new_val.to(tl.int8))

    if DO_STATS:
        # bank by program id: 5 same-address atomics from every one of the
        # B*H*T programs serialize at L2; spreading over BANKS copies (host
        # sums them) keeps the stats path off the kernel's critical path
        bank = (pid % BANKS) * 5
        tl.atomic_add(STATS + bank + 0, tl.sum(rewire.to(tl.float32), axis=0))
        tl.atomic_add(STATS + bank + 1, tl.sum(on_sup.to(tl.float32), axis=0))
        tl.atomic_add(STATS + bank + 2, m_incl - 1.0)
        tl.atomic_add(STATS + bank + 3, h_norm)
        tl.atomic_add(STATS + bank + 4, 1.0)


@torch.library.custom_op("parameter_golf::churn_rewire", mutates_args=())
def churn_rewire(
    idx: Tensor, valid: Tensor, p: Tensor, pn: Tensor, seed: Tensor,
    layer: int, churn_eps: float, do_stats: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    bsz, num_heads, seqlen, kb = idx.shape
    idx = idx.contiguous()
    valid = valid.contiguous()
    p = p.contiguous()
    pn = pn.contiguous()
    new_idx = torch.empty_like(idx)
    new_valid = torch.empty(idx.shape, device=idx.device, dtype=torch.int8)
    stats = torch.zeros(_STATS_BANKS * 5, device=idx.device, dtype=torch.float32)
    grid = (bsz * num_heads * seqlen,)
    _churn_tail_kernel[grid](
        idx, valid, p, pn, seed, new_idx, new_valid, stats,
        churn_eps, seqlen, layer,
        KB=kb, DO_STATS=do_stats, BANKS=_STATS_BANKS,
        num_warps=1,
    )
    return new_idx, new_valid, stats.view(_STATS_BANKS, 5).sum(0)


@churn_rewire.register_fake
def _(idx, valid, p, pn, seed, layer, churn_eps, do_stats):
    return (
        idx.new_empty(idx.shape),
        idx.new_empty(idx.shape, dtype=torch.int8),
        p.new_empty(5, dtype=torch.float32),
    )
