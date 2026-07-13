"""Fused streaming candidate-selection kernel for sparse entmax attention.

Replaces the eager selection pipeline (materialize T x T coarse scores ->
masked_fill x2 -> topk -> softmax -> entropy: ~25 ms/layer compiled, ~5 GB of
T^2 traffic per layer) with one Triton program per query row.  Each program:

* sweeps the 16 sketch dimensions of the transposed sketched keys (whole
  sketched K per (batch, kv-head) is ~2 MB and stays L2-resident; each sweep
  is one contiguous row load), accumulating the full coarse score row in
  registers - no T x T tensor ever exists,
* rounds scores to bf16 so ranking compares exactly the values the reference
  pipeline ranks (its coarse matmul emits bf16),
* computes the exploration entropy of the causal row INCLUDING self (the
  reference's softmax entropy) from the same registers,
* packs each (score, index) into one order-preserving 26-bit key (16 bf16
  score bits above 10 index bits) and PARTITIONS the causal, self-excluded
  candidates with a bitwise binary search for the two rank thresholds
  (n_strong-th and n_rank-th largest key) plus a cumsum compaction.  No sort:
  a full descending order of 1024 keys costs ~4x the rest of the kernel, and
  downstream only consumes two rank SETS, not an order.

The kernel runs ONE warp per program: every reduction (26 double counts in
the threshold search, the entropy sums, the compaction scans) then compiles
to register shuffles with no shared-memory barrier.  Measured against the
alternatives at B=32/T=1024: warp-1 rows 5.5 ms, warp-4 rows 10.5 ms, a
tensor-core QT=16 row-tile variant 9.0 ms (Triton 2D reductions pay a
shared-memory layout conversion per op), tl.sort rows 8.7 ms.

Output layout (per query row, n_rank slots):
  [0, n_strong)       the strongest n_strong candidates, ascending position
  [n_strong, n_rank)  ranks n_strong+1..n_rank ("weakest"), ascending position
  -inf / index-0 fillers pad each partition when fewer candidates exist; they
  occupy the same slot positions the reference's descending sort gives them.

Semantics vs the reference selection: identical candidate SETS on both sides
of the partition boundary, identical values, validity counts and positions,
and entropy.  Downstream exploration replaces each of the last r = n_rank -
n_strong slots i.i.d., so a permutation within the weak partition leaves the
training distribution unchanged (slots are exchangeable); entmax and dedup
are set-operations.  The only defined differences from the reference are the
within-partition order (reference: score-descending; here: position-
ascending) and tie order among exactly-equal bf16 scores (torch.topk's is
unspecified; here ties break toward the lower position index,
deterministically).  The tie order has one second-order effect: when the
n_strong-th and next scores are exactly-equal bf16, it decides which tied
candidate lands in the strong partition and is thereby protected from
exploration replacement - still inside torch.topk's unspecified-tie
envelope, just more than cosmetic.  Exploration, validity, dedup, and the
RNG stream stay in PyTorch downstream, untouched.

Selection runs under no_grad by design: no backward exists.
T is capped at 1024 by the 10 index bits; the long-context selector
(two-stage recall prefilter) is a separate, planned kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

IDX_BITS = 10
MAX_T = 1 << IDX_BITS


@triton.jit
def _select_kernel(
    QS, KST, VAL, IDXO, ENT,
    scale,
    sq_b, sq_h, sq_t, sq_c,
    sk_b, sk_h, sk_c, sk_t,
    H, T, NS,
    GROUP: tl.constexpr,
    C: tl.constexpr,
    TB: tl.constexpr,      # padded row width (power of two >= T)
    NR: tl.constexpr,      # ranked slots to emit (n_rank)
    IDXB: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid % T
    h = (pid // T) % H
    b = pid // (T * H)
    kv_h = h // GROUP

    offs_j = tl.arange(0, TB)
    in_row = (offs_j <= t) & (offs_j < T)  # causal keys 0..t (incl. self)

    # Coarse scores for the whole row: C contiguous column sweeps.
    scores = tl.zeros([TB], dtype=tl.float32)
    k_col = KST + b * sk_b + kv_h * sk_h
    q_ptr = QS + b * sq_b + h * sq_h + t * sq_t
    for c in tl.static_range(C):
        qc = tl.load(q_ptr + c * sq_c).to(tl.float32)
        kc = tl.load(k_col + c * sk_c + offs_j * sk_t, mask=in_row, other=0.0)
        scores += qc * kc.to(tl.float32)
    # Rank exactly what the reference ranks.  Its bf16 pipeline rounds twice:
    # the coarse matmul emits bf16, then the scale multiply rounds again.
    scores = scores.to(tl.bfloat16).to(tl.float32) * scale
    scores = scores.to(tl.bfloat16).to(tl.float32)

    # Entropy of softmax over the causal row incl. self (reference formula
    # H = logZ - sum(z * e^z)/Z, computed fp32 with the usual max shift).
    neg = float("-inf")
    z = tl.where(in_row, scores, neg)
    m = tl.max(z, axis=0)
    e = tl.where(in_row, tl.exp(z - m), 0.0)
    l = tl.sum(e, axis=0)
    w = tl.sum(tl.where(in_row, scores * e, 0.0), axis=0)
    entropy = m + tl.log(l) - w / l
    tl.store(ENT + (b * H + h) * T + t, entropy)

    # Pack (score, index) into int32 sort keys.  Order-preserving float map:
    # positive bf16 bits get the sign bit set; negative bits are inverted.
    rankable = in_row & (offs_j != t)  # self is force-included downstream
    idx_mask: tl.constexpr = (1 << IDXB) - 1
    bits = scores.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
    bits = tl.where((bits & 0x8000) != 0, 0xFFFF - bits, bits + 0x8000)
    # Lower position wins ties: encode (idx_mask - j) below the score bits.
    key = (bits << IDXB) | ((idx_mask - offs_j).to(tl.uint32) & idx_mask)
    key = tl.where(rankable, key, 0).to(tl.int32)  # non-candidates sink to 0

    # Rank thresholds partition the row into [strong | weak | rest]: bitwise
    # binary search for the largest v with count(key >= v) >= k, both
    # thresholds per loop trip.  Keys are unique except the 0-valued sunk
    # lanes, so for k >= 1 and at least k nonzero keys this is exactly the
    # k-th largest key; with fewer it resolves to 0 (clamped to 1 below:
    # the selection mask then takes every nonzero key).
    KEY_BITS: tl.constexpr = 16 + IDXB
    thr_s = tl.zeros((), dtype=tl.int32)
    thr_w = tl.zeros((), dtype=tl.int32)
    for i in tl.static_range(KEY_BITS):
        mid_s = thr_s + (1 << (KEY_BITS - 1 - i))
        mid_w = thr_w + (1 << (KEY_BITS - 1 - i))
        cnt_s = tl.sum((key >= mid_s).to(tl.int32), axis=0)
        cnt_w = tl.sum((key >= mid_w).to(tl.int32), axis=0)
        thr_s = tl.where(cnt_s >= NS, mid_s, thr_s)
        thr_w = tl.where(cnt_w >= NR, mid_w, thr_w)
    strong = (key >= tl.maximum(thr_s, 1)) & (NS > 0)
    weak = (key >= tl.maximum(thr_w, 1)) & ~strong

    # Compact each partition into its output span (ascending position order).
    sc = tl.cumsum(strong.to(tl.int32), axis=0)
    wc = tl.cumsum(weak.to(tl.int32), axis=0)
    n_s = tl.sum(strong.to(tl.int32), axis=0)
    n_w = tl.sum(weak.to(tl.int32), axis=0)
    dest = tl.where(strong, sc - 1, NS + wc - 1)
    sel = strong | weak

    out_row = ((b * H + h) * T + t) * NR
    tl.store(VAL + out_row + dest, scores, mask=sel)
    tl.store(IDXO + out_row + dest, offs_j.to(tl.int32), mask=sel)
    # -inf / index-0 fillers pad each partition (disjoint from `dest` slots).
    fill = ((offs_j >= n_s) & (offs_j < NS)) | (
        (offs_j >= NS + n_w) & (offs_j < NR)
    )
    tl.store(VAL + out_row + offs_j, neg, mask=fill)
    tl.store(IDXO + out_row + offs_j, tl.zeros([TB], tl.int32), mask=fill)


def select_topk_entropy(
    qs: Tensor, ks: Tensor, scale: float, n_rank: int, n_strong: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Fused causal coarse top-(n_rank) selection + row entropy.

    qs: [B, H, T, C] sketched queries; ks: [B, KV, T, C] sketched keys
    (unexpanded GQA; head h reads kv head h // (H/KV)).  Returns
    (top_val fp32 [B,H,T,n_rank], -inf on padding slots;
     top_idx int32 [B,H,T,n_rank], 0 on padding slots;
     entropy fp32 [B,H,T] of the causal softmax row including self).
    Slots [0, n_strong) hold the strongest n_strong candidates and
    [n_strong, n_rank) the next n_rank - n_strong, each partition in
    ascending position order (see module docstring for why this preserves
    the reference semantics).  Self and future positions never rank.
    """
    bsz, num_heads, seqlen, cdim = qs.shape
    assert qs.dtype == torch.bfloat16 and ks.dtype == torch.bfloat16, (
        "fused selection ranks at bf16 precision (int32 sort keys); "
        "use the torch pipeline for other dtypes"
    )
    assert ks.shape[0] == bsz and ks.shape[2] == seqlen and ks.shape[3] == cdim
    assert num_heads % ks.shape[1] == 0
    assert 0 < n_rank < seqlen
    assert 0 <= n_strong <= n_rank
    assert seqlen <= MAX_T, f"packed index budget is {MAX_T}"
    assert cdim & (cdim - 1) == 0
    tb = max(triton.next_power_of_2(seqlen), triton.next_power_of_2(n_rank))
    qs = qs.contiguous()
    ks_t = ks.transpose(-2, -1).contiguous()  # [B, KV, C, T]: contiguous cols
    val = torch.empty(bsz, num_heads, seqlen, n_rank, device=qs.device, dtype=torch.float32)
    idx = torch.empty(bsz, num_heads, seqlen, n_rank, device=qs.device, dtype=torch.int32)
    ent = torch.empty(bsz, num_heads, seqlen, device=qs.device, dtype=torch.float32)
    grid = (bsz * num_heads * seqlen,)
    _select_kernel[grid](
        qs, ks_t, val, idx, ent,
        scale,
        *qs.stride(), *ks_t.stride(),
        num_heads, seqlen, n_strong,
        GROUP=num_heads // ks.shape[1], C=cdim, TB=tb, NR=n_rank, IDXB=IDX_BITS,
        num_warps=1,  # single warp: all reductions stay shuffle-only
    )
    return val, idx, ent
