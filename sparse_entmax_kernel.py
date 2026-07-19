"""Fused Triton kernel for sparse candidate attention with entmax-1.5 routing.

Replaces the dense-cost reference path in ``sparse_entmax_attn_train_gpt.py``
(materialize T x T scores -> gather -> entmax -> scatter -> dense AV) with one
kernel launch per direction.  Each Triton program handles a single query: it
gathers that query's candidate K rows straight into SRAM, forms the fine
logits in fp32 registers, solves the entmax-1.5 threshold by bisection (the
simplex objective sum(max(x - tau, 0)^2) - 1 is monotone in tau, and tau* lies
in [x_max - 1, x_max] after the standard shift), then streams the candidate V
rows for the weighted sum.  No T x T tensor ever exists, so the module needs
no activation checkpointing and the backward recomputes nothing.

Backward uses the closed-form entmax-1.5 Jacobian-vector product on the
support (with s = sqrt(p)):

    dL/dlogit_j = s_j * (g_j - <g, s> / <s, 1>),   g_j = dL/dp_j

which is the same "support locally constant" derivative the reference autograd
path takes through its re-derived threshold.  The null column joins the
projection with g_null = 0 (its mass is dropped from the output, not from the
simplex).  dK/dV are accumulated with fp32 atomics into the unexpanded KV-head
layout, which also folds the GQA group reduction.

Layouts: q [B, H, T, D], k/v [B, KV, T, D] (unexpanded; head h reads KV head
h // (H // KV)), idx int32 [B, H, T, KB] with KB a power of two, valid bool of
the same shape.  Invalid slots are excluded exactly (masked to -inf, p = 0, no
gradient).  Routing weights p (and the null weight) are saved fp32 for
backward; rows sum to 1 including the null, and the output uses only the
candidate part, so null mass attenuates exactly as in the reference.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

BISECT_ITERS = 8


@triton.jit
def _entmax_tau(x, active, NULL_X, BITER: tl.constexpr):
    """Newton solve for the entmax-1.5 threshold over active lanes + the null.

    ``x`` is shifted so max(x, null) == 0; f(tau) = sum(max(x - tau, 0)^2) - 1
    is convex, strictly decreasing on [-1, 0] with f(-1) >= 0 >= f(0).  Newton
    from tau = -1 stays below tau* and converges monotonically (quadratically
    within each parabolic piece - NOT one-step-exact; do not reduce BITER
    below 8: the worst case, all-equal logits at KB=128, still has tau error
    ~1e-2 after 4 iters and only reaches the fp32 noise floor ~7e-9 at 7-8.
    At 8 iters measured error <= 6e-8 with support sets identical to exact
    bisection).  Was: 30 bisection rounds = 30 sequential cross-warp
    reductions; Newton needs 2 reductions/iter but ~4x fewer iters.
    """
    tau = -1.0
    for _ in tl.static_range(BITER):
        t = tl.where(active, tl.maximum(x - tau, 0.0), 0.0)
        tn = tl.maximum(NULL_X - tau, 0.0)
        f = tl.sum(t * t, axis=0) + tn * tn - 1.0
        d = 2.0 * (tl.sum(t, axis=0) + tn)
        tau += f / tl.maximum(d, 1e-30)
    return tau


@triton.jit
def _sparse_entmax_fwd_kernel(
    Q, K, V, IDX, VALID, NULLB, Y, P, PN,
    scale,
    sq_b, sq_h, sq_t, sq_d,
    sk_b, sk_h, sk_t, sk_d,
    sv_b, sv_h, sv_t, sv_d,
    sy_b, sy_h, sy_t, sy_d,
    si_b, si_h, si_t, si_k,
    sm_b, sm_h, sm_t, sm_k,
    H, T,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    KB: tl.constexpr,
    BITER: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid % T
    h = (pid // T) % H
    b = pid // (T * H)
    kv_h = h // GROUP

    offs_k = tl.arange(0, KB)
    offs_d = tl.arange(0, D)

    row = ((b * H + h) * T + t) * KB
    idx = tl.load(IDX + b * si_b + h * si_h + t * si_t + offs_k * si_k)
    val = tl.load(VALID + b * sm_b + h * sm_h + t * sm_t + offs_k * sm_k) != 0
    idx = tl.where(val, idx, 0)  # masked lanes never dereference, keep in-range

    q = tl.load(Q + b * sq_b + h * sq_h + t * sq_t + offs_d * sq_d).to(tl.float32)
    k_ptr = K + b * sk_b + kv_h * sk_h
    k_rows = tl.load(
        k_ptr + idx[:, None] * sk_t + offs_d[None, :] * sk_d,
        mask=val[:, None],
        other=0.0,
    ).to(tl.float32)
    logits = tl.sum(q[None, :] * k_rows, axis=1) * scale
    null_logit = tl.load(NULLB + h).to(tl.float32)

    mx = tl.maximum(tl.max(tl.where(val, logits, float("-inf")), axis=0), null_logit)
    x = tl.where(val, (logits - mx) * 0.5, -1.0)  # anything <= tau_min works
    xn = (null_logit - mx) * 0.5
    tau = _entmax_tau(x, val, xn, BITER)

    p = tl.maximum(x - tau, 0.0)
    p = p * p
    pn = tl.maximum(xn - tau, 0.0)
    pn = pn * pn
    z = tl.sum(tl.where(val, p, 0.0), axis=0) + pn
    p = tl.where(val, p / z, 0.0)
    pn = pn / z

    # entmax support is exact: p == 0 slots contribute nothing to y, so only
    # gather V rows on the support (predicated-off lanes issue no traffic).
    # Support shrinks as heads sharpen, so this gets cheaper over training.
    on_sup = val & (p > 0)
    v_ptr = V + b * sv_b + kv_h * sv_h
    v_rows = tl.load(
        v_ptr + idx[:, None] * sv_t + offs_d[None, :] * sv_d,
        mask=on_sup[:, None],
        other=0.0,
    ).to(tl.float32)
    y = tl.sum(p[:, None] * v_rows, axis=0)

    tl.store(Y + b * sy_b + h * sy_h + t * sy_t + offs_d * sy_d, y.to(Y.dtype.element_ty))
    tl.store(P + row + offs_k, p)
    tl.store(PN + (b * H + h) * T + t, pn)


@triton.jit
def _sparse_entmax_bwd_kernel(
    DY, Q, K, V, IDX, VALID, P, PN,
    DQ, DK, DV, DN,
    scale,
    sq_b, sq_h, sq_t, sq_d,
    sk_b, sk_h, sk_t, sk_d,
    sv_b, sv_h, sv_t, sv_d,
    sy_b, sy_h, sy_t, sy_d,
    si_b, si_h, si_t, si_k,
    sm_b, sm_h, sm_t, sm_k,
    H, T,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    KB: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid % T
    h = (pid // T) % H
    b = pid // (T * H)
    kv_h = h // GROUP

    offs_k = tl.arange(0, KB)
    offs_d = tl.arange(0, D)

    row = ((b * H + h) * T + t) * KB
    idx = tl.load(IDX + b * si_b + h * si_h + t * si_t + offs_k * si_k)
    val = tl.load(VALID + b * sm_b + h * sm_h + t * sm_t + offs_k * sm_k) != 0
    idx = tl.where(val, idx, 0)
    p = tl.load(P + row + offs_k)
    pn = tl.load(PN + (b * H + h) * T + t)

    dy = tl.load(DY + b * sy_b + h * sy_h + t * sy_t + offs_d * sy_d).to(tl.float32)

    # EVERYTHING downstream is exactly zero off the entmax support: g enters
    # only via g * s (s = sqrt(p) = 0), dlog = s * (...) = 0, and the dk/dv
    # atomics were already support-masked.  So gather V and K rows on the
    # support only - backward traffic scales with support fraction, not KB.
    on_support = val & (p > 0)
    v_ptr = V + b * sv_b + kv_h * sv_h
    v_rows = tl.load(
        v_ptr + idx[:, None] * sv_t + offs_d[None, :] * sv_d,
        mask=on_support[:, None],
        other=0.0,
    ).to(tl.float32)
    g = tl.sum(dy[None, :] * v_rows, axis=1)
    dv_ptr = DV + b * sv_b + kv_h * sv_h
    tl.atomic_add(
        dv_ptr + idx[:, None] * sv_t + offs_d[None, :] * sv_d,
        p[:, None] * dy[None, :],
        mask=on_support[:, None],
    )

    # entmax-1.5 JVP on the support; the null joins the projection with g = 0.
    s = tl.sqrt(p)
    sn = tl.sqrt(pn)
    s_sum = tl.sum(tl.where(on_support, s, 0.0), axis=0) + sn
    gs = tl.sum(tl.where(on_support, g * s, 0.0), axis=0)
    dlog = tl.where(on_support, s * (g - gs / s_sum), 0.0)
    dnull = sn * (0.0 - gs / s_sum)
    tl.store(DN + (b * H + h) * T + t, dnull)

    # dQ = scale * sum_i dlog_i k_i ; dK_i += scale * dlog_i * q
    q = tl.load(Q + b * sq_b + h * sq_h + t * sq_t + offs_d * sq_d).to(tl.float32)
    k_ptr = K + b * sk_b + kv_h * sk_h
    k_rows = tl.load(
        k_ptr + idx[:, None] * sk_t + offs_d[None, :] * sk_d,
        mask=on_support[:, None],
        other=0.0,
    ).to(tl.float32)
    dq = tl.sum(dlog[:, None] * k_rows, axis=0) * scale
    tl.store(DQ + b * sq_b + h * sq_h + t * sq_t + offs_d * sq_d, dq)
    dk_ptr = DK + b * sk_b + kv_h * sk_h
    tl.atomic_add(
        dk_ptr + idx[:, None] * sk_t + offs_d[None, :] * sk_d,
        (dlog * scale)[:, None] * q[None, :],
        mask=on_support[:, None],
    )


def _check_layout(q: Tensor, k: Tensor, v: Tensor, idx: Tensor, valid: Tensor) -> None:
    bsz, num_heads, seqlen, head_dim = q.shape
    kb = idx.size(-1)
    # k/v may hold MORE rows than the query length (cross-layer arms gather
    # from a [B, Hkv, n_layers*T, D] stack); wiring guarantees idx stays in range
    assert k.ndim == 4 and k.shape == v.shape and k.shape[0] == bsz and k.shape[3] == head_dim
    assert num_heads % k.shape[1] == 0
    assert idx.shape == (bsz, num_heads, seqlen, kb) and valid.shape == idx.shape
    assert idx.dtype == torch.int32 and valid.dtype == torch.bool
    assert kb & (kb - 1) == 0, "candidate slots must be padded to a power of two"
    assert head_dim & (head_dim - 1) == 0


@torch.library.custom_op("parameter_golf::sparse_entmax_attn", mutates_args=())
def _sparse_entmax_attn_fwd(
    q: Tensor, k: Tensor, v: Tensor, idx: Tensor, valid: Tensor,
    null_bias: Tensor, scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    _check_layout(q, k, v, idx, valid)
    bsz, num_heads, seqlen, head_dim = q.shape
    kb = idx.size(-1)
    # Q/K/V stay contiguous because the backward indexes their contiguous
    # grad buffers with the same strides. IDX/VALID may be batch-broadcast
    # views: persistent graphs are shared across examples, and retaining a
    # batch copy per layer wastes gigabytes at production shape.
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    y = torch.empty_like(q)
    p = torch.empty(bsz, num_heads, seqlen, kb, device=q.device, dtype=torch.float32)
    pn = torch.empty(bsz, num_heads, seqlen, device=q.device, dtype=torch.float32)
    grid = (bsz * num_heads * seqlen,)
    _sparse_entmax_fwd_kernel[grid](
        q, k, v, idx, valid, null_bias.float(), y, p, pn,
        scale,
        *q.stride(), *k.stride(), *v.stride(), *y.stride(),
        *idx.stride(), *valid.stride(),
        num_heads, seqlen,
        GROUP=num_heads // k.shape[1], D=head_dim, KB=kb, BITER=BISECT_ITERS,
        num_warps=2,
    )
    return y, p, pn


@_sparse_entmax_attn_fwd.register_fake
def _(q, k, v, idx, valid, null_bias, scale):
    bsz, num_heads, seqlen, _ = q.shape
    return (
        # Explicitly contiguous: the real op contiguous()es q, so the fake
        # must promise contiguous strides for any input layout.
        q.new_empty(q.shape),
        q.new_empty(bsz, num_heads, seqlen, idx.size(-1), dtype=torch.float32),
        q.new_empty(bsz, num_heads, seqlen, dtype=torch.float32),
    )


@torch.library.custom_op("parameter_golf::sparse_entmax_attn_bwd", mutates_args=())
def _sparse_entmax_attn_bwd(
    dy: Tensor, q: Tensor, k: Tensor, v: Tensor, idx: Tensor, valid: Tensor,
    p: Tensor, pn: Tensor, scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    bsz, num_heads, seqlen, head_dim = q.shape
    dy = dy.contiguous()
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    # Grad buffers are allocated contiguous and the kernel indexes them with
    # q/k/v's strides, so those must be contiguous too (ensured above).
    dq = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    dk = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
    dv = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
    dn = torch.empty(bsz, num_heads, seqlen, device=q.device, dtype=torch.float32)
    grid = (bsz * num_heads * seqlen,)
    _sparse_entmax_bwd_kernel[grid](
        dy, q, k, v, idx, valid, p, pn,
        dq, dk, dv, dn,
        scale,
        *q.stride(), *k.stride(), *v.stride(), *dy.stride(),
        *idx.stride(), *valid.stride(),
        num_heads, seqlen,
        GROUP=num_heads // k.shape[1], D=head_dim, KB=idx.size(-1),
        num_warps=8,  # measured best for bwd (fwd stays at 2): atomic-heavy,
    )                 # more warps hide the gather+atomic latency
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dn.sum(dim=(0, 2))


@_sparse_entmax_attn_bwd.register_fake
def _(dy, q, k, v, idx, valid, p, pn, scale):
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        q.new_empty(q.shape[1], dtype=torch.float32),
    )


def _setup_context(ctx, inputs, output):
    q, k, v, idx, valid, null_bias, scale = inputs
    _, p, pn = output
    ctx.save_for_backward(q, k, v, idx, valid, p, pn)
    ctx.scale = scale


def _backward(ctx, dy, dp, dpn):
    q, k, v, idx, valid, p, pn = ctx.saved_tensors
    dq, dk, dv, dnull = _sparse_entmax_attn_bwd(dy, q, k, v, idx, valid, p, pn, ctx.scale)
    return dq, dk, dv, None, None, dnull, None


_sparse_entmax_attn_fwd.register_autograd(_backward, setup_context=_setup_context)


def sparse_entmax_attention(
    q: Tensor, k: Tensor, v: Tensor, idx: Tensor, valid: Tensor,
    null_bias: Tensor, scale: float,
) -> Tensor:
    """Candidate attention with entmax-1.5 routing and a per-head null logit.

    ``k``/``v`` stay in the unexpanded [B, KV, T, D] layout; ``idx``/``valid``
    are [B, H, T, KB] with KB a power of two (pad with valid=False).  Returns
    [B, H, T, D] in q's dtype.  Gradients flow to q, k, v, and null_bias.
    """
    # Contiguous here (not just inside the op) so the tensors autograd saves
    # for backward are already in kernel layout - no second copy at bwd time.
    y, _, _ = _sparse_entmax_attn_fwd(
        q.contiguous(), k.contiguous(), v.contiguous(),
        idx, valid, null_bias, scale,
    )
    return y


def sparse_entmax_attention_stats(
    q: Tensor, k: Tensor, v: Tensor, idx: Tensor, valid: Tensor,
    null_bias: Tensor, scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """As ``sparse_entmax_attention`` but also returns the routing weights.

    Returns (y, p, pn): p fp32 [B, H, T, KB] entmax mass per candidate slot
    (0 on invalid slots), pn fp32 [B, H, T] null mass; rows of (p, pn) sum to
    1.  The kernel already saves both for backward, so this costs nothing.
    Consumers use them as read-only routing statistics (e.g. churn wiring);
    gradients flow through y exactly as in the plain wrapper.
    """
    y, p, pn = _sparse_entmax_attn_fwd(
        q.contiguous(), k.contiguous(), v.contiguous(),
        idx, valid, null_bias, scale,
    )
    return y, p, pn
