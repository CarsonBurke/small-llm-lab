"""Single-query Mini attention with read-only, detached ring-buffer history.

Only the query and the current key/value are differentiated. The CUDA kernels
read old keys/values in place: neither a concatenated history nor historical
key/value gradients are allocated. Callers must finish backward before changing
any cache or position tensor passed to ``cached_attention``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _attention_forward(
    Q,
    K,
    V,
    KEY_HISTORY,
    VALUE_HISTORY,
    POSITIONS,
    OUTPUT,
    OUTPUT_FP32,
    LOGSUMEXP,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    CAPACITY: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    bh = tl.program_id(0)
    lane = bh // HEADS
    position = tl.load(POSITIONS + lane)
    history_size = tl.minimum(position, CAPACITY)
    overwritten = position % CAPACITY
    d = tl.arange(0, WIDTH)
    current_offset = bh.to(tl.int64) * WIDTH + d
    q = tl.load(Q + current_offset).to(tl.float32)
    current_key = tl.load(K + current_offset).to(tl.float32)
    current_value = tl.load(V + current_offset).to(tl.float32)

    # Seed online softmax with the always-valid current token. Empty histories,
    # capacity=1 and an all-masked last tile therefore need no special path.
    maximum = tl.sum(q * current_key, 0) * SCALE
    denominator = tl.full((), 1.0, tl.float32)
    numerator = current_value
    cache_base = bh.to(tl.int64) * CAPACITY * WIDTH
    for start in range(0, history_size, BLOCK_T):
        slots = start + tl.arange(0, BLOCK_T)
        valid = (slots < history_size) & (
            (position < CAPACITY) | (slots != overwritten)
        )
        offsets = cache_base + slots[:, None] * WIDTH + d[None, :]
        keys = tl.load(KEY_HISTORY + offsets, mask=valid[:, None], other=0.0).to(
            tl.float32
        )
        scores = tl.sum(keys * q[None, :], 1) * SCALE
        scores = tl.where(valid, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, 0))
        old_weight = tl.exp(maximum - next_maximum)
        weights = tl.exp(scores - next_maximum)
        values = tl.load(VALUE_HISTORY + offsets, mask=valid[:, None], other=0.0).to(
            tl.float32
        )
        numerator = numerator * old_weight + tl.sum(weights[:, None] * values, 0)
        denominator = denominator * old_weight + tl.sum(weights, 0)
        maximum = next_maximum

    output = numerator / denominator
    tl.store(OUTPUT + current_offset, output)
    # Keep the unrounded softmax output for delta = dot(dout, output). Saving
    # only BF16 here would introduce a spurious softmax gradient from rounding.
    tl.store(OUTPUT_FP32 + current_offset, output)
    tl.store(LOGSUMEXP + bh, maximum + tl.log(denominator))


@triton.jit
def _attention_backward(
    Q,
    K,
    V,
    KEY_HISTORY,
    VALUE_HISTORY,
    POSITIONS,
    OUTPUT_FP32,
    LOGSUMEXP,
    GRAD_OUTPUT,
    GRAD_Q,
    GRAD_K,
    GRAD_V,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    CAPACITY: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    bh = tl.program_id(0)
    lane = bh // HEADS
    position = tl.load(POSITIONS + lane)
    history_size = tl.minimum(position, CAPACITY)
    overwritten = position % CAPACITY
    d = tl.arange(0, WIDTH)
    current_offset = bh.to(tl.int64) * WIDTH + d
    q = tl.load(Q + current_offset).to(tl.float32)
    current_key = tl.load(K + current_offset).to(tl.float32)
    current_value = tl.load(V + current_offset).to(tl.float32)
    output = tl.load(OUTPUT_FP32 + current_offset)
    grad_output = tl.load(GRAD_OUTPUT + current_offset).to(tl.float32)
    logsumexp = tl.load(LOGSUMEXP + bh)
    delta = tl.sum(grad_output * output, 0)
    current_score = tl.sum(q * current_key, 0) * SCALE
    current_probability = tl.exp(current_score - logsumexp)
    current_ds = current_probability * (tl.sum(grad_output * current_value, 0) - delta)
    grad_q = current_ds * current_key
    tl.store(GRAD_K + current_offset, SCALE * current_ds * q)
    tl.store(GRAD_V + current_offset, current_probability * grad_output)

    cache_base = bh.to(tl.int64) * CAPACITY * WIDTH
    for start in range(0, history_size, BLOCK_T):
        slots = start + tl.arange(0, BLOCK_T)
        valid = (slots < history_size) & (
            (position < CAPACITY) | (slots != overwritten)
        )
        offsets = cache_base + slots[:, None] * WIDTH + d[None, :]
        keys = tl.load(KEY_HISTORY + offsets, mask=valid[:, None], other=0.0).to(
            tl.float32
        )
        scores = tl.sum(keys * q[None, :], 1) * SCALE
        probabilities = tl.where(valid, tl.exp(scores - logsumexp), 0.0)
        values = tl.load(VALUE_HISTORY + offsets, mask=valid[:, None], other=0.0).to(
            tl.float32
        )
        dp = tl.sum(values * grad_output[None, :], 1)
        ds = probabilities * (dp - delta)
        grad_q += tl.sum(ds[:, None] * keys, 0)
    tl.store(GRAD_Q + current_offset, SCALE * grad_q)


@triton.jit
def _grouped_attention_forward(
    Q,
    K,
    V,
    KEY_HISTORY,
    VALUE_HISTORY,
    POSITIONS,
    OUTPUT,
    OUTPUT_FP32,
    LOGSUMEXP,
    CURRENT_PROBABILITY,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    CAPACITY: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # One CTA owns an entire query group and loads each shared history tile once.
    # Padding the group to >=16 rows enables tensor-core QK and probability-V
    # products without expanding historical K/V to the number of query heads.
    bkv = tl.program_id(0)
    lane = bkv // KV_HEADS
    kv_head = bkv % KV_HEADS
    GROUP_SIZE: tl.constexpr = QUERY_HEADS // KV_HEADS
    rows = tl.arange(0, BLOCK_Q)
    d = tl.arange(0, WIDTH)
    query_heads = lane.to(tl.int64) * QUERY_HEADS + kv_head * GROUP_SIZE + rows
    query_offsets = query_heads[:, None] * WIDTH + d[None, :]
    current_offsets = bkv.to(tl.int64) * WIDTH + d
    q = tl.load(Q + query_offsets, mask=rows[:, None] < GROUP_SIZE, other=0.0)
    current_key = tl.load(K + current_offsets).to(tl.float32)
    current_value = tl.load(V + current_offsets).to(tl.float32)
    position = tl.load(POSITIONS + lane)
    history_size = tl.minimum(position, CAPACITY)
    overwritten = position % CAPACITY
    maximum = tl.sum(q.to(tl.float32) * current_key[None, :], 1) * SCALE
    denominator = tl.full((BLOCK_Q,), 1.0, tl.float32)
    current_weight = tl.full((BLOCK_Q,), 1.0, tl.float32)
    numerator = tl.broadcast_to(current_value[None, :], (BLOCK_Q, WIDTH))
    cache_base = bkv.to(tl.int64) * CAPACITY * WIDTH
    for start in range(0, history_size, BLOCK_T):
        slots = start + tl.arange(0, BLOCK_T)
        valid = (slots < history_size) & (
            (position < CAPACITY) | (slots != overwritten)
        )
        offsets = cache_base + slots[:, None] * WIDTH + d[None, :]
        keys = tl.load(KEY_HISTORY + offsets, mask=valid[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(keys)) * SCALE
        scores = tl.where(valid[None, :], scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        old_weight = tl.exp(maximum - next_maximum)
        current_weight = current_weight * old_weight
        weights = tl.exp(scores - next_maximum[:, None])
        values = tl.load(VALUE_HISTORY + offsets, mask=valid[:, None], other=0.0)
        # Preserve the FP32 softmax weights to two BF16 terms, rather than
        # introducing a full BF16 rounding error into the weighted value sum.
        weights_hi = weights.to(tl.bfloat16)
        weights_lo = (weights - weights_hi.to(tl.float32)).to(tl.bfloat16)
        numerator = tl.dot(weights_hi, values, numerator * old_weight[:, None])
        numerator = tl.dot(weights_lo, values, numerator)
        denominator = denominator * old_weight + tl.sum(weights, 1)
        maximum = next_maximum

    output = numerator / denominator[:, None]
    tl.store(OUTPUT + query_offsets, output, mask=rows[:, None] < GROUP_SIZE)
    tl.store(OUTPUT_FP32 + query_offsets, output, mask=rows[:, None] < GROUP_SIZE)
    tl.store(
        LOGSUMEXP + query_heads,
        maximum + tl.log(denominator),
        mask=rows < GROUP_SIZE,
    )
    tl.store(
        CURRENT_PROBABILITY + query_heads,
        current_weight / denominator,
        mask=rows < GROUP_SIZE,
    )


@triton.jit
def _grouped_attention_backward(
    Q,
    K,
    V,
    KEY_HISTORY,
    VALUE_HISTORY,
    POSITIONS,
    OUTPUT_FP32,
    LOGSUMEXP,
    CURRENT_PROBABILITY,
    GRAD_OUTPUT,
    GRAD_Q,
    GRAD_K,
    GRAD_V,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    CAPACITY: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    bkv = tl.program_id(0)
    lane = bkv // KV_HEADS
    kv_head = bkv % KV_HEADS
    GROUP_SIZE: tl.constexpr = QUERY_HEADS // KV_HEADS
    rows = tl.arange(0, BLOCK_Q)
    d = tl.arange(0, WIDTH)
    query_heads = lane.to(tl.int64) * QUERY_HEADS + kv_head * GROUP_SIZE + rows
    query_offsets = query_heads[:, None] * WIDTH + d[None, :]
    current_offsets = bkv.to(tl.int64) * WIDTH + d
    q = tl.load(Q + query_offsets, mask=rows[:, None] < GROUP_SIZE, other=0.0)
    current_key = tl.load(K + current_offsets).to(tl.float32)
    current_value = tl.load(V + current_offsets).to(tl.float32)
    output = tl.load(
        OUTPUT_FP32 + query_offsets, mask=rows[:, None] < GROUP_SIZE, other=0.0
    )
    grad_output = tl.load(
        GRAD_OUTPUT + query_offsets, mask=rows[:, None] < GROUP_SIZE, other=0.0
    )
    logsumexp = tl.load(LOGSUMEXP + query_heads, mask=rows < GROUP_SIZE, other=0.0)
    delta = tl.sum(grad_output.to(tl.float32) * output, 1)
    # Reuse the forward coefficient instead of recomputing a differently
    # rounded dot/logsumexp difference. A lone current token has exactly p=1.
    current_probability = tl.load(
        CURRENT_PROBABILITY + query_heads, mask=rows < GROUP_SIZE, other=0.0
    )
    current_dp = tl.sum(grad_output.to(tl.float32) * current_value[None, :], 1)
    current_ds = current_probability * (current_dp - delta)
    grad_q = current_ds[:, None] * current_key[None, :]
    # This CTA owns all queries using this current K/V: reduction needs neither
    # atomics nor a separate gradient buffer, and history stays nondifferentiable.
    tl.store(
        GRAD_K + current_offsets,
        SCALE * tl.sum(current_ds[:, None] * q.to(tl.float32), 0),
    )
    tl.store(
        GRAD_V + current_offsets,
        tl.sum(current_probability[:, None] * grad_output.to(tl.float32), 0),
    )
    position = tl.load(POSITIONS + lane)
    history_size = tl.minimum(position, CAPACITY)
    overwritten = position % CAPACITY
    cache_base = bkv.to(tl.int64) * CAPACITY * WIDTH
    for start in range(0, history_size, BLOCK_T):
        slots = start + tl.arange(0, BLOCK_T)
        valid = (slots < history_size) & (
            (position < CAPACITY) | (slots != overwritten)
        )
        offsets = cache_base + slots[:, None] * WIDTH + d[None, :]
        keys = tl.load(KEY_HISTORY + offsets, mask=valid[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(keys)) * SCALE
        probabilities = tl.where(
            valid[None, :] & (rows[:, None] < GROUP_SIZE),
            tl.exp(scores - logsumexp[:, None]),
            0.0,
        )
        values = tl.load(VALUE_HISTORY + offsets, mask=valid[:, None], other=0.0)
        dp = tl.dot(grad_output, tl.trans(values))
        ds = probabilities * (dp - delta[:, None])
        ds_hi = ds.to(tl.bfloat16)
        ds_lo = (ds - ds_hi.to(tl.float32)).to(tl.bfloat16)
        grad_q = tl.dot(ds_hi, keys, grad_q)
        grad_q = tl.dot(ds_lo, keys, grad_q)

    tl.store(GRAD_Q + query_offsets, SCALE * grad_q, mask=rows[:, None] < GROUP_SIZE)


def _check_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    keys: Tensor,
    values: Tensor,
    positions: Tensor,
) -> tuple[int, int, int, int, int]:
    if q.ndim != 3:
        raise ValueError("q must have shape [batch, query_heads, head_dim]")
    batch, heads, width = q.shape
    if batch < 1 or heads < 1 or width < 16 or width & (width - 1):
        raise ValueError("head_dim must be a power of two >=16; batch/heads >0")
    if k.ndim != 3 or v.shape != k.shape:
        raise ValueError(
            "current key/value must have matching [batch, kv_heads, head_dim] shapes"
        )
    kv_heads = k.shape[1]
    if kv_heads < 1 or heads % kv_heads or k.shape != (batch, kv_heads, width):
        raise ValueError(
            "current K/V must match query batch/head_dim; kv_heads must divide query_heads"
        )
    if keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError(
            "history must have matching [batch, kv_heads, capacity, head_dim] shapes"
        )
    capacity = keys.shape[2]
    if keys.shape != (batch, kv_heads, capacity, width) or capacity < 1:
        raise ValueError(
            "history dimensions must match current K/V, with positive capacity"
        )
    if positions.shape != (batch,) or positions.dtype != torch.int64:
        raise ValueError("positions must be int64 [batch]")
    if q.device.type != "cuda":
        raise ValueError("stream attention requires CUDA; there is no CPU fallback")
    for tensor in (q, k, v, keys, values):
        if tensor.dtype != torch.bfloat16 or tensor.device != q.device:
            raise ValueError(
                "query, current key/value and history must be BF16 on the same CUDA device"
            )
    if positions.device != q.device:
        raise ValueError("positions must be on the query device")
    # Copying a noncontiguous history would silently add O(capacity) work.
    if not keys.is_contiguous() or not values.is_contiguous():
        raise ValueError("history caches must be contiguous")
    return batch, heads, kv_heads, width, capacity


class _CachedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, keys, values, positions, scale):
        batch, heads, kv_heads, width, capacity = _check_inputs(
            q, k, v, keys, values, positions
        )
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        positions = positions.contiguous()
        output = torch.empty_like(q)
        output_fp32 = torch.empty(q.shape, device=q.device, dtype=torch.float32)
        logsumexp = torch.empty((batch, heads), device=q.device, dtype=torch.float32)
        current_probability = None
        if heads == kv_heads:
            _attention_forward[(batch * heads,)](
                q,
                k,
                v,
                keys,
                values,
                positions,
                output,
                output_fp32,
                logsumexp,
                HEADS=heads,
                WIDTH=width,
                CAPACITY=capacity,
                SCALE=scale,
                BLOCK_T=64,
                num_warps=4,
            )
        else:
            current_probability = torch.empty_like(logsumexp)
            _grouped_attention_forward[(batch * kv_heads,)](
                q,
                k,
                v,
                keys,
                values,
                positions,
                output,
                output_fp32,
                logsumexp,
                current_probability,
                QUERY_HEADS=heads,
                KV_HEADS=kv_heads,
                WIDTH=width,
                CAPACITY=capacity,
                SCALE=scale,
                BLOCK_Q=max(16, triton.next_power_of_2(heads // kv_heads)),
                BLOCK_T=64,
                num_warps=4,
            )
        ctx.save_for_backward(
            q,
            k,
            v,
            keys,
            values,
            positions,
            output_fp32,
            logsumexp,
            current_probability,
        )
        ctx.scale = scale
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            q,
            k,
            v,
            keys,
            values,
            positions,
            output_fp32,
            logsumexp,
            current_probability,
        ) = ctx.saved_tensors
        batch, heads, width = q.shape
        kv_heads = k.shape[1]
        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        if heads == kv_heads:
            _attention_backward[(batch * heads,)](
                q,
                k,
                v,
                keys,
                values,
                positions,
                output_fp32,
                logsumexp,
                grad_output.contiguous(),
                grad_q,
                grad_k,
                grad_v,
                HEADS=heads,
                WIDTH=width,
                CAPACITY=keys.shape[2],
                SCALE=ctx.scale,
                BLOCK_T=64,
                num_warps=4,
            )
        else:
            _grouped_attention_backward[(batch * kv_heads,)](
                q,
                k,
                v,
                keys,
                values,
                positions,
                output_fp32,
                logsumexp,
                current_probability,
                grad_output.contiguous(),
                grad_q,
                grad_k,
                grad_v,
                QUERY_HEADS=heads,
                KV_HEADS=kv_heads,
                WIDTH=width,
                CAPACITY=keys.shape[2],
                SCALE=ctx.scale,
                BLOCK_Q=max(16, triton.next_power_of_2(heads // kv_heads)),
                BLOCK_T=64,
                num_warps=4,
            )
        return grad_q, grad_k, grad_v, None, None, None, None


def cached_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    keys: Tensor,
    values: Tensor,
    positions: Tensor,
    scale: float = 0.12,
) -> Tensor:
    """Attend to the current token and its last ``capacity-1`` predecessors.

    ``positions[B]`` counts already-committed tokens in each lane's current
    document. The caller must pass zero at BOS, irrespective of the old cache.
    Historical token ``t`` occupies slot ``t % capacity``. Partial histories use
    slots ``[0, position)``; full histories exclude the slot being overwritten.
    The new key/value participate directly and remain differentiable, without
    being written to history until the caller explicitly commits after backward.

    Inputs/current output are BF16 CUDA tensors. First-order gradients are
    returned only for ``q``, ``k`` and ``v``; history and positions are detached
    even when their input tensors happen to require gradients.
    Queries have shape ``[B,Hq,d]``; current K/V are ``[B,Hkv,d]`` and histories
    are ``[B,Hkv,C,d]``. ``Hkv`` must divide ``Hq``. Each contiguous group of
    ``Hq/Hkv`` query heads shares a K/V head; its current K/V gradients are summed.
    """
    return _CachedAttention.apply(
        q, k, v, keys.detach(), values.detach(), positions.detach(), scale
    )


__all__ = ["cached_attention"]
