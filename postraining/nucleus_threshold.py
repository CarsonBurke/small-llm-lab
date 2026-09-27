"""Exact top-p nucleus thresholds without a vocabulary sort (CUDA, Triton).

A token belongs to the top-p nucleus of ``softmax(scores)`` when the
probability mass strictly above its score is at most ``top_p`` (the rule
``latent_rollout.nucleus_mask`` states by sorting). Write ``A(t)`` for the
mass strictly above ``t``: it is non-increasing in ``t``, so the nucleus is
``{v : scores[v] >= tau}`` for the smallest row score ``tau`` with
``A(tau) <= top_p``. Tokens tied at ``tau`` are all kept, exactly as the
sorted rule keeps them.

Sorting 1024 rows of the 50,304-token vocabulary costs ~8 ms per decode
tick, 60x the Gumbel draw it gates. This module finds ``tau`` by a bracket
search instead. Every comparison runs on "keys", the order-preserving int32
image of fp32, so signed zeros and flushed denormals cannot bend the counts:

1. Statistics: the maximum, the minimum finite score and the softmax
   normaliser. The bracket ``(lo, hi]`` starts one key below the minimum
   (mass 1 > top_p above it) and at the maximum (none above it).
2. Each pass evaluates ``K`` cuts over the bracket and keeps the tightest
   ``(lo, hi]`` with ``A(lo) > top_p >= A(hi)``. It carries ``#(s > lo)``,
   ``#(s > hi)`` and the mass above ``hi``, so a pass needs only the scores
   inside the bracket, only their mass and count strictly above each cut,
   and skips every block holding none. The row is done once no score lies
   strictly between ``lo`` and ``hi`` -- the bracket is empty or its keys are
   adjacent -- and then ``s > lo`` is exactly the nucleus: every score above
   ``lo`` is at or above ``hi``, so its mass-above is at most ``A(hi)``.
   Later passes skip the row.

The first pass spaces its cuts evenly in value between the minimum and the
maximum. That leaves ~1/33 of the score range, typically a few hundred
tokens. Later passes space them evenly in key space, which shrinks a bracket
of ``d`` keys to at most ``ceil(d / (K + 1))``. A key range is below 2**32,
so ``_PASSES`` passes bring every row to adjacent keys whatever the logits.
The threshold returned is the value of key ``lo + 1``. The caller's float
comparison against it agrees with ``key > lo`` except that it ties -0 with
+0, and that tie is the rule's own. The cuts live in
a buffer written by the step that chose them, so the partial and combining
kernels compare against identical values.

The decode tail runs this on 8-16 rows, where one program per row would be a
serial chain. Every statistics and cut pass instead splits the vocabulary
across enough programs to fill the GPU. Each program accumulates its slice
elementwise and reduces once. A one-program-per-row kernel then combines
the partials and moves the bracket. The launch count is fixed and there is
no host synchronisation, so the whole search captures into a CUDA graph.

Masses are summed in fp32 in split order rather than the sort's cumsum order,
so a row whose boundary mass sits within fp32 rounding of ``top_p`` can
resolve the boundary token either way. That is a measure-zero difference
between two exact implementations of the same rule, not an approximation of
it. Within one pass every cut's mass is the same carried mass plus a sum
over the same tree, and fp32 addition is monotone, so the cuts inside the
budget are still a suffix. The kernels read the caller's own scores --
``logits`` itself at unit temperature (a bf16 -> fp32 widening is exact),
otherwise the tensor the caller already divided -- so the threshold compares
exactly against them. Recomputing the division here would not: inductor
lowers division by a constant to multiplication by its reciprocal, which can
land an ulp away.
"""

from __future__ import annotations

import functools

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
def _ordered_key(x):
    """fp32 -> int32 whose signed order is the float order (-0 < +0)."""
    bits = x.to(tl.int32, bitcast=True)
    return bits ^ ((bits >> 31) & 0x7FFFFFFF)


@triton.jit
def _key_value(key):
    """Inverse of ``_ordered_key``; the map is an involution."""
    return (key ^ ((key >> 31) & 0x7FFFFFFF)).to(tl.float32, bitcast=True)


@triton.jit
def _load_scores(base, idx, stop):
    return tl.load(base + idx, mask=idx < stop, other=float("-inf")).to(tl.float32)


@triton.jit
def _stats_partial_kernel(
    scores_ptr,
    partial_ptr,
    vocab,
    stride_row,
    splits,
    per_split,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    base = scores_ptr + row * stride_row
    start = split * per_split
    stop = tl.minimum(start + per_split, vocab)
    offs = tl.arange(0, BLOCK)
    peak = tl.full((BLOCK,), float("-inf"), tl.float32)
    total = tl.zeros((BLOCK,), tl.float32)
    floor = tl.full((BLOCK,), float("inf"), tl.float32)
    finite = tl.zeros((BLOCK,), tl.float32)
    for block in range(start, stop, BLOCK):
        scores = _load_scores(base, block + offs, stop)
        live = scores > float("-inf")
        new_peak = tl.maximum(peak, scores)
        # While a lane has seen only -inf, exp(peak - new_peak) is NaN.
        total = tl.where(
            live,
            total * tl.exp(peak - new_peak) + tl.exp(scores - new_peak),
            total,
        )
        peak = new_peak
        floor = tl.minimum(floor, tl.where(live, scores, float("inf")))
        finite += live.to(tl.float32)
    row_peak = tl.max(peak, axis=0)
    weight = tl.where(total > 0, tl.exp(peak - row_peak), 0.0)
    out = partial_ptr + (row * splits + split) * 4
    tl.store(out, row_peak)
    tl.store(out + 1, tl.sum(total * weight, axis=0))
    tl.store(out + 2, tl.min(floor, axis=0))
    tl.store(out + 3, tl.sum(finite, axis=0))


@triton.jit
def _stats_combine_kernel(
    partial_ptr,
    stats_ptr,
    mass_hi_ptr,
    bracket_ptr,
    counts_ptr,
    cuts_ptr,
    threshold_ptr,
    splits,
    K: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, MAX_SPLITS)
    present = offs < splits
    part = partial_ptr + (row * splits + offs) * 4
    peaks = tl.load(part, mask=present, other=float("-inf"))
    totals = tl.load(part + 1, mask=present, other=0.0)
    floors = tl.load(part + 2, mask=present, other=float("inf"))
    finite = tl.load(part + 3, mask=present, other=0.0)
    peak = tl.max(peaks, axis=0)
    floor = tl.min(floors, axis=0)
    total = tl.sum(tl.where(totals > 0, totals * tl.exp(peaks - peak), 0.0), axis=0)
    lo = _ordered_key(floor).to(tl.int64) - 1
    hi = _ordered_key(peak).to(tl.int64)
    tl.store(stats_ptr + row * 2, peak)
    tl.store(stats_ptr + row * 2 + 1, total)
    tl.store(mass_hi_ptr + row, 0.0)
    tl.store(bracket_ptr + row * 2, lo)
    tl.store(bracket_ptr + row * 2 + 1, hi)
    tl.store(counts_ptr + row * 2, tl.sum(finite, axis=0).to(tl.int32))
    tl.store(counts_ptr + row * 2 + 1, 0)
    # Exact already when every finite score is one value (hi = lo + 1).
    tl.store(threshold_ptr + row, _key_value((lo + 1).to(tl.int32)))
    # First cuts: evenly spaced in value, clamped into [lo + 1, hi - 1] (a
    # row with hi = lo + 1 is done and never reads them). Monotone rounding
    # keeps them non-decreasing.
    fraction = (tl.arange(0, K) + 1).to(tl.float32) / (K + 1)
    keys = _ordered_key(floor + (peak - floor) * fraction).to(tl.int64)
    keys = tl.maximum(tl.minimum(keys, hi - 1), lo + 1)
    tl.store(cuts_ptr + row * K + tl.arange(0, K), keys)


@triton.jit
def _cut_partial_kernel(
    scores_ptr,
    stats_ptr,
    bracket_ptr,
    counts_ptr,
    cuts_ptr,
    mass_ptr,
    count_ptr,
    vocab,
    stride_row,
    splits,
    per_split,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    lo = tl.load(bracket_ptr + row * 2)
    hi = tl.load(bracket_ptr + row * 2 + 1)
    above_lo = tl.load(counts_ptr + row * 2)
    above_hi = tl.load(counts_ptr + row * 2 + 1)
    if (hi - lo > 1) & (above_lo > above_hi):
        peak = tl.load(stats_ptr + row * 2)
        key_lo = lo.to(tl.int32)
        key_hi = hi.to(tl.int32)
        cut = tl.arange(0, K)
        cuts = tl.load(cuts_ptr + row * K + cut).to(tl.int32)
        base = scores_ptr + row * stride_row
        start = split * per_split
        stop = tl.minimum(start + per_split, vocab)
        offs = tl.arange(0, BLOCK)
        mass_over = tl.zeros((BLOCK, K), tl.float32)
        count_over = tl.zeros((BLOCK, K), tl.int32)
        for block in range(start, stop, BLOCK):
            scores = _load_scores(base, block + offs, stop)
            keys = _ordered_key(scores)
            inner = (keys > key_lo) & (keys <= key_hi)
            # Outside the bracket a score's contribution to every cut is
            # already carried; after the first passes most blocks hold
            # nothing inside it.
            if tl.max(inner.to(tl.int32), axis=0) > 0:
                over = (keys[:, None] > cuts[None, :]) & inner[:, None]
                weight = tl.exp(scores - peak)
                mass_over += tl.where(over, weight[:, None], 0.0)
                count_over += over.to(tl.int32)
        out = (row * splits + split) * K
        tl.store(mass_ptr + out + cut, tl.sum(mass_over, axis=0))
        tl.store(count_ptr + out + cut, tl.sum(count_over, axis=0))


@triton.jit
def _cut_combine_kernel(
    stats_ptr,
    mass_hi_ptr,
    bracket_ptr,
    counts_ptr,
    cuts_ptr,
    mass_ptr,
    count_ptr,
    threshold_ptr,
    splits,
    top_p,
    K: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lo = tl.load(bracket_ptr + row * 2)
    hi = tl.load(bracket_ptr + row * 2 + 1)
    above_lo = tl.load(counts_ptr + row * 2)
    above_hi = tl.load(counts_ptr + row * 2 + 1)
    if (hi - lo > 1) & (above_lo > above_hi):
        # Inductor may hand a Python float over as fp64; keep the math fp32.
        budget = top_p.to(tl.float32) * tl.load(stats_ptr + row * 2 + 1)
        mass_hi = tl.load(mass_hi_ptr + row)
        cut = tl.arange(0, K)
        cuts = tl.load(cuts_ptr + row * K + cut)
        offs = tl.arange(0, MAX_SPLITS)
        present = (offs < splits)[:, None]
        part = (row * splits + offs[:, None]) * K + cut[None, :]
        mass_over = tl.sum(tl.load(mass_ptr + part, mask=present, other=0.0), axis=0)
        count_over = tl.sum(tl.load(count_ptr + part, mask=present, other=0), axis=0)
        # A(cut) = (mass above hi) + (bracket mass above cut) <= top_p holds
        # on a suffix of the cuts. The first cut of the suffix becomes hi and
        # the last cut before it becomes lo. Every per-cut quantity is
        # monotone in the cut, so max/min pick exactly those cuts' values.
        inside = mass_hi + mass_over <= budget
        new_hi = tl.min(tl.where(inside, cuts, hi), axis=0)
        new_lo = tl.max(tl.where(inside, lo, cuts), axis=0)
        tl.store(
            mass_hi_ptr + row,
            mass_hi + tl.max(tl.where(inside, mass_over, 0.0), axis=0),
        )
        tl.store(
            counts_ptr + row * 2,
            above_hi
            + tl.min(tl.where(inside, above_lo - above_hi, count_over), axis=0),
        )
        tl.store(
            counts_ptr + row * 2 + 1,
            above_hi + tl.max(tl.where(inside, count_over, 0), axis=0),
        )
        tl.store(bracket_ptr + row * 2, new_lo)
        tl.store(bracket_ptr + row * 2 + 1, new_hi)
        tl.store(threshold_ptr + row, _key_value((new_lo + 1).to(tl.int32)))
        # Next cuts: evenly spaced in key space over [new_lo, new_hi).
        # Non-decreasing, and one lies strictly inside whenever
        # new_hi - new_lo >= 2.
        steps = (cut + 1).to(tl.int64)
        tl.store(
            cuts_ptr + row * K + cut,
            new_lo + ((new_hi - new_lo) * steps) // (K + 1),
        )


# Cuts per pass; the two [BLOCK, K] accumulators stay in registers.
_K = 32
_BLOCK = 128
_NUM_WARPS = 4
# Largest split count; the combine kernels tile it statically.
_MAX_SPLITS = 64
# Programs per SM a pass aims for when rows alone cannot fill the GPU.
_PROGRAMS_PER_SM = 2


def _passes_to_close(cuts: int) -> int:
    """Key-space passes that shrink any bracket (< 2**32 keys) to adjacent
    keys, plus the value-space first pass (which never widens it)."""
    width, passes = 2**32, 0
    while width > 1:
        width = -(-width // (cuts + 1))
        passes += 1
    return passes + 1


_PASSES = _passes_to_close(_K)


@functools.cache
def _multiprocessors(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _split_count(rows, vocab: int, device: torch.device):
    """Splits per row; symbolic in, symbolic out.

    ``torch.sym_min``/``sym_max`` keep a symbolic row count symbolic where
    the builtins would guard on it and recompile per row bucket.
    """
    target = _PROGRAMS_PER_SM * _multiprocessors(device.index or 0)
    wanted = torch.sym_min(
        -(-target // torch.sym_max(rows, 1)), -(-vocab // (4 * _BLOCK))
    )
    return torch.sym_max(1, torch.sym_min(wanted, _MAX_SPLITS))


@torch.library.triton_op("nanogpt_sampling::nucleus_threshold", mutates_args={})
def _nucleus_threshold(scores: Tensor, top_p: float) -> Tensor:
    rows, vocab = scores.shape
    device = scores.device
    splits = _split_count(rows, vocab, device)
    # Whole blocks per split; trailing splits may own none.
    per_split = -(-vocab // (splits * _BLOCK)) * _BLOCK
    float32 = dict(device=device, dtype=torch.float32)
    stats_partial = torch.empty((rows * splits * 4,), **float32)
    stats = torch.empty((rows, 2), **float32)
    mass_hi = torch.empty((rows,), **float32)
    bracket = torch.empty((rows, 2), device=device, dtype=torch.int64)
    counts = torch.empty((rows, 2), device=device, dtype=torch.int32)
    cuts = torch.empty((rows, _K), device=device, dtype=torch.int64)
    mass = torch.empty((rows * splits * _K,), **float32)
    count = torch.empty((rows * splits * _K,), device=device, dtype=torch.int32)
    threshold = torch.empty((rows,), **float32)
    wrap_triton(_stats_partial_kernel)[(rows, splits)](
        scores,
        stats_partial,
        vocab,
        scores.stride(0),
        splits,
        per_split,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
    )
    wrap_triton(_stats_combine_kernel)[(rows,)](
        stats_partial,
        stats,
        mass_hi,
        bracket,
        counts,
        cuts,
        threshold,
        splits,
        K=_K,
        MAX_SPLITS=_MAX_SPLITS,
        num_warps=1,
    )
    for _ in range(_PASSES):
        wrap_triton(_cut_partial_kernel)[(rows, splits)](
            scores,
            stats,
            bracket,
            counts,
            cuts,
            mass,
            count,
            vocab,
            scores.stride(0),
            splits,
            per_split,
            K=_K,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
        wrap_triton(_cut_combine_kernel)[(rows,)](
            stats,
            mass_hi,
            bracket,
            counts,
            cuts,
            mass,
            count,
            threshold,
            splits,
            top_p,
            K=_K,
            MAX_SPLITS=_MAX_SPLITS,
            num_warps=4,
        )
    return threshold


def nucleus_threshold(scores: Tensor, top_p: float) -> Tensor:
    """Per-row fp32 threshold of the top-p nucleus of ``softmax(scores)``.

    ``scores.float() >= threshold[:, None]`` is the nucleus. Pass the exact
    scores the caller compares: temperature-scaled fp32 scores, or bf16/fp32
    logits at unit temperature. Requires a CUDA tensor with a unit-stride
    vocabulary dimension and ``0 < top_p < 1``; every row needs at least one
    finite score.
    """
    if scores.dim() != 2 or scores.stride(-1) != 1:
        raise ValueError("nucleus_threshold needs [rows, vocab] with unit vocab stride")
    if not 0.0 < top_p < 1.0:
        raise ValueError(f"nucleus_threshold needs 0 < top_p < 1, got {top_p}")
    return _nucleus_threshold(scores, float(top_p))
