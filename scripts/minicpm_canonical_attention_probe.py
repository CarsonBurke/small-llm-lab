"""Numerical forward probe: fixed-tile BF16 causal GQA, not trained backward yet.

This is an independent experiment, not a production attention replacement. QK
and PV use BF16 tensor-core operands and FP32 accumulators; online softmax state
stays FP32 and probabilities are rounded to BF16 before PV. There is no fallback,
split-KV, autotuning, dropout, attention bias, or autograd implementation.

Every query/head row traverses ascending 64-key blocks with the same reduction
geometry, regardless of query width and cache capacity. A wholly future-masked
block preserves that row's state exactly. Queries need not be consecutive or
sorted. Arbitrary input strides are used directly, without replicated KV or a
materialized attention matrix. Output is contiguous, with the query shape/dtype.

The fixed reduction sequence is a design constraint, not a measured bitwise
parity claim: Main must qualify compiled row-lane invariance and performance on
the target GPU. Warm specializations before CUDA graph capture. This module does
not launch work on import.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
import triton
import triton.language as tl


KERNEL_METADATA = {
    "query_positions_per_program": 16,
    "query_heads_per_pack": 8,
    "packed_rows": 128,
    "key_block": 64,
    "head_dimension": 128,
    "num_warps": 4,
    "num_stages": 1,
    "enable_fp_fusion": False,
    "qk_operands": "bf16",
    "pv_operands": "bf16 (including rounded probabilities)",
    "accumulators_and_softmax": "fp32",
    "key_reduction": "ascending unsplit 64-key blocks",
    "grid": "(ceil(Q/16), B, Hkv*ceil((Hq/Hkv)/8))",
    "workspace": "output only; no global attention-score workspace",
    "backward": False,
    "qualified": False,
}


@triton.jit(do_not_specialize=["Q_LEN", "K_CAP"])
def _canonical_attention_kernel(
    QUERY, KEY, VALUE, POSITIONS, LENGTHS, OUTPUT,
    Q_LEN, K_CAP,
    Q_HEADS: tl.constexpr, HEAD_GROUP: tl.constexpr,
    QB: tl.constexpr, QT: tl.constexpr, QH: tl.constexpr, QD: tl.constexpr,
    KB: tl.constexpr, KT: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
    VB: tl.constexpr, VT: tl.constexpr, VH: tl.constexpr, VD: tl.constexpr,
    PB: tl.constexpr, PT: tl.constexpr, LB: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
):
    batch = tl.program_id(1)
    head_pack = tl.program_id(2)
    packs_per_kv: tl.constexpr = tl.cdiv(HEAD_GROUP, 8)
    kv_head = head_pack // packs_per_kv
    local_pack = head_pack % packs_per_kv
    rows = tl.arange(0, 128)
    query_index = tl.program_id(0) * 16 + rows // 8
    group_head = local_pack * 8 + rows % 8
    query_head = kv_head * HEAD_GROUP + group_head
    row_valid = (query_index < Q_LEN) & (group_head < HEAD_GROUP)
    position = tl.load(
        POSITIONS + batch * PB + query_index * PT,
        mask=row_valid, other=-1,
    ).to(tl.int64)
    length = tl.load(LENGTHS + batch * LB).to(tl.int64)
    # Bounds are also enforced against physical allocation. Negative positions
    # denote empty/padded query rows; empty prefixes produce zero, not NaN.
    visible_end = tl.minimum(tl.minimum(length, K_CAP), position + 1)
    visible_end = tl.where(row_valid, tl.maximum(visible_end, 0), 0)
    tile_end = tl.max(visible_end, axis=0)

    dims = tl.arange(0, 128)
    query = tl.load(
        QUERY + batch * QB + query_index[:, None] * QT
        + query_head[:, None] * QH + dims[None, :] * QD,
        mask=row_valid[:, None], other=0,
    )
    running_max = tl.full((128,), float("-inf"), tl.float32)
    denominator = tl.full((128,), 0.0, tl.float32)
    accumulator = tl.full((128, 128), 0.0, tl.float32)
    key_lanes = tl.arange(0, 64)

    # Dynamic bounds do not specialize the arithmetic to Q or cache allocation.
    # No software pipelining: each update consumes the previous FP32 state.
    for block in range(tl.cdiv(tile_end, 64)):
        keys = block * 64 + key_lanes
        # Mask to the tile's visible prefix, not just allocation/length. This
        # also prevents reading uninitialized padding in the final key block.
        key_valid = keys < tile_end
        key = tl.load(
            KEY + batch * KB + keys[None, :] * KT
            + kv_head * KH + dims[:, None] * KD,
            mask=key_valid[None, :], other=0,
        )
        value = tl.load(
            VALUE + batch * VB + keys[:, None] * VT
            + kv_head * VH + dims[None, :] * VD,
            mask=key_valid[:, None], other=0,
        )
        logits = tl.dot(query, key, out_dtype=tl.float32) * SCALE_LOG2
        allowed = keys[None, :] < visible_end[:, None]
        logits = tl.where(allowed, logits, float("-inf"))
        active = block * 64 < visible_end
        block_max = tl.max(logits, axis=1)
        next_max = tl.maximum(running_max, block_max)
        # Avoid -inf - -inf for padded rows and future-only blocks. Explicit
        # state selection below makes wholly masked blocks *true* no-ops,
        # including signed zero/NaN state, rather than relying on x*1 + 0.
        safe_max = tl.where(active, next_max, 0.0)
        alpha = tl.exp2(running_max - safe_max)
        probabilities = tl.exp2(logits - safe_max[:, None])
        block_sum = tl.sum(probabilities, axis=1)
        next_denominator = denominator * alpha + block_sum
        scaled_accumulator = accumulator * alpha[:, None]
        next_accumulator = tl.dot(
            probabilities.to(tl.bfloat16), value, scaled_accumulator,
            out_dtype=tl.float32,
        )
        running_max = tl.where(active, next_max, running_max)
        denominator = tl.where(active, next_denominator, denominator)
        accumulator = tl.where(active[:, None], next_accumulator, accumulator)

    divisor = tl.where(denominator > 0.0, denominator, 1.0)
    result = accumulator / divisor[:, None]
    # Q_LEN affects only address calculation and masks, never a dot/reduction
    # shape. Each valid query/head has exactly one writer, including GQA tails.
    output_rows = (batch * Q_LEN + query_index) * Q_HEADS + query_head
    tl.store(
        OUTPUT + output_rows[:, None] * 128 + dims[None, :], result,
        mask=row_valid[:, None],
    )


def canonical_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    query_positions: Tensor,
    key_lengths: Tensor,
    scale: float,
) -> Tensor:
    """Attend BF16 ``[B,Q,Hq,128]`` to ``[B,K,Hkv,128]`` causal prefixes.

    ``query_positions[B,Q]`` gives absolute, zero-based positions in the KV
    sequence; key index ``j`` is visible iff ``j <= position`` and
    ``j < key_lengths[b]``. Negative query positions and zero key lengths give
    zero output. Lengths must be in ``[0,K]``; device values are not host-read or
    synchronized. Allocation bounds are independently enforced by the kernel.
    Positions/lengths are CUDA int32 or int64 on the operand device. Hq must be
    a positive multiple of Hkv; groups larger than eight use additional fixed
    eight-head packs, and smaller groups mask unused head rows.

    Operands must contain finite values within their live regions. Positive and
    zero strides (including transposed/expanded views) are supported directly.
    Empty B/Q returns an empty result; K=0 returns zeros through the same kernel.
    Input tensors requiring gradients need an explicit no_grad/inference_mode
    context: this primitive deliberately has no backward implementation.
    """
    operands = (query, key, value)
    if any(t.ndim != 4 for t in operands):
        raise ValueError("expected query/key/value with four dimensions")
    if any(t.layout != torch.strided or not t.is_cuda
           or t.dtype != torch.bfloat16 for t in operands):
        raise ValueError("query/key/value must be strided CUDA bf16 tensors")
    if any(t.device != query.device for t in operands):
        raise ValueError("query/key/value devices must match")
    batch, width, q_heads, dimension = query.shape
    if dimension != 128 or key.shape[-1] != 128 or value.shape[-1] != 128:
        raise ValueError("canonical attention requires head dimension 128")
    if key.shape != value.shape or key.shape[0] != batch:
        raise ValueError("key/value shapes must match and share the query batch")
    kv_heads = key.shape[2]
    if kv_heads < 1 or q_heads < 1 or q_heads % kv_heads:
        raise ValueError("query heads must be a positive multiple of KV heads")
    if query_positions.shape != (batch, width) or key_lengths.shape != (batch,):
        raise ValueError("expected query_positions[B,Q] and key_lengths[B]")
    for tensor in (query_positions, key_lengths):
        if (tensor.device != query.device or tensor.layout != torch.strided
                or tensor.dtype not in (torch.int32, torch.int64)):
            raise ValueError("positions/lengths must be strided int32/int64 on the query device")
    if torch.is_grad_enabled() and any(t.requires_grad for t in operands):
        raise RuntimeError("canonical attention is forward-only; use no_grad or inference_mode")
    if not math.isfinite(scale):
        raise ValueError("attention scale must be finite")

    output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    if batch == 0 or width == 0:
        return output
    head_group = q_heads // kv_heads
    with torch.cuda.device(query.device):
        _canonical_attention_kernel[
            (triton.cdiv(width, 16), batch, kv_heads * triton.cdiv(head_group, 8))
        ](
            query, key, value, query_positions, key_lengths, output,
            width, key.shape[1], q_heads, head_group,
            *query.stride(), *key.stride(), *value.stride(),
            *query_positions.stride(), key_lengths.stride(0),
            float(scale) * math.log2(math.e),
            num_warps=4, num_stages=1, enable_fp_fusion=False,
        )
    return output
