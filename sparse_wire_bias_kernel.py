"""Wire-bias + wire-gate tail for the LEARNED-GRAPH arm.

Fork of ``sparse_entmax_kernel`` (same gather + entmax-1.5 solve + closed-form
JVP backward) with two per-(query-head, slot) fp32 inputs that carry gradient
into the learned graph tables:

* ``wire_bias`` is added to the fine logits before the entmax solve::

      score_e = (q . k_e) * scale + wire_bias[h, e]

  Its gradient is exactly the per-row score gradient ``dlog`` the backward
  already computes for dq/dk, summed over rows.  This is a *marginal,
  redistribution* signal: it pulls cold-but-useful wires into the entmax
  support and calibrates their prior weight, but it vanishes once a wire's
  mass is locally optimal.

* ``wire_gate`` multiplies each wire's output contribution::

      y = sum_e p_e * gate_e * v_e

  Its gradient ``dgate_e = sum_rows p_e * (dy . v_e)`` is the wire's *gross
  contribution* alignment - it does NOT vanish at the mass optimum, so a
  calibrated wire keeps earning selection score instead of drifting out of
  the top-K.  The training script passes ``gate = 1 + theta - sg(theta)``
  (straight-through identity), so the forward is numerically the ungated
  attention while both signals accumulate into one logit table.

Both gradients land in tiny [H, K] buffers; a naive atomic would hotspot
~1e7 updates onto 1k addresses, so the kernel banks the accumulators
``pid % BANKS`` ways (same trick as the churn stats) and the host folds the
banks.  Neither tensor is saved for backward: given the saved routing weights
p/pn, ``dlog`` and ``dgate`` at gate == 1 do not depend on their values.
Everything else - layouts, GQA mapping, invalid-slot masking, null logit
semantics, support-restricted backward traffic - is identical to
``sparse_entmax_kernel`` and covered by its numerical validation; the tests
pin bias=0/gate=1 parity against it plus finite differences for both inputs.

Also hosts the list-input cross-layer wrapper (fork of the one in
``sparse_xlayer_kernel``): per-layer K/V lists are cat'd transiently in fwd
and rebuilt in bwd so no consumer retains a concatenated bank.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from sparse_entmax_kernel import BISECT_ITERS, _check_layout, _entmax_tau
from sparse_xlayer_kernel import _cat_kv

_DBIAS_BANKS = 64


@triton.jit
def _sparse_entmax_bias_fwd_kernel(
    Q, K, V, IDX, VALID, WB, WG, NULLB, Y, P, PN,
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
    wb = tl.load(WB + h * KB + offs_k)
    logits = tl.sum(q[None, :] * k_rows, axis=1) * scale + wb
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

    on_sup = val & (p > 0)
    v_ptr = V + b * sv_b + kv_h * sv_h
    v_rows = tl.load(
        v_ptr + idx[:, None] * sv_t + offs_d[None, :] * sv_d,
        mask=on_sup[:, None],
        other=0.0,
    ).to(tl.float32)
    wg = tl.load(WG + h * KB + offs_k)
    y = tl.sum((p * wg)[:, None] * v_rows, axis=0)

    tl.store(Y + b * sy_b + h * sy_h + t * sy_t + offs_d * sy_d, y.to(Y.dtype.element_ty))
    tl.store(P + row + offs_k, p)
    tl.store(PN + (b * H + h) * T + t, pn)


@triton.jit
def _sparse_entmax_bias_bwd_kernel(
    DY, Q, K, V, IDX, VALID, WG, P, PN,
    DQ, DK, DV, DB, DG, DN,
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
    BANKS: tl.constexpr,
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

    on_support = val & (p > 0)
    v_ptr = V + b * sv_b + kv_h * sv_h
    v_rows = tl.load(
        v_ptr + idx[:, None] * sv_t + offs_d[None, :] * sv_d,
        mask=on_support[:, None],
        other=0.0,
    ).to(tl.float32)
    wg = tl.load(WG + h * KB + offs_k)
    c = tl.sum(dy[None, :] * v_rows, axis=1)  # per-wire contribution dy . v_e
    g = wg * c                                # dL/dp_e through the gated sum
    dv_ptr = DV + b * sv_b + kv_h * sv_h
    tl.atomic_add(
        dv_ptr + idx[:, None] * sv_t + offs_d[None, :] * sv_d,
        (p * wg)[:, None] * dy[None, :],
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

    # The bias is additive in the score, so dbias_e is dlog summed over rows;
    # the gate scales the contribution, so dgate_e is p_e * (dy . v_e).
    bank = pid % BANKS
    tl.atomic_add(DB + (bank * H + h) * KB + offs_k, dlog, mask=on_support)
    tl.atomic_add(DG + (bank * H + h) * KB + offs_k, p * c, mask=on_support)

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


@torch.library.custom_op("parameter_golf::sparse_entmax_attn_bias", mutates_args=())
def _sparse_entmax_bias_attn_fwd(
    q: Tensor, k: Tensor, v: Tensor, idx: Tensor, valid: Tensor,
    wire_bias: Tensor, wire_gate: Tensor, null_bias: Tensor, scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    _check_layout(q, k, v, idx, valid)
    bsz, num_heads, seqlen, head_dim = q.shape
    kb = idx.size(-1)
    assert wire_bias.shape == (num_heads, kb) and wire_bias.dtype == torch.float32
    assert wire_gate.shape == (num_heads, kb) and wire_gate.dtype == torch.float32
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    wire_bias = wire_bias.contiguous()
    wire_gate = wire_gate.contiguous()
    y = torch.empty_like(q)
    p = torch.empty(bsz, num_heads, seqlen, kb, device=q.device, dtype=torch.float32)
    pn = torch.empty(bsz, num_heads, seqlen, device=q.device, dtype=torch.float32)
    grid = (bsz * num_heads * seqlen,)
    _sparse_entmax_bias_fwd_kernel[grid](
        q, k, v, idx, valid, wire_bias, wire_gate, null_bias.float(), y, p, pn,
        scale,
        *q.stride(), *k.stride(), *v.stride(), *y.stride(),
        *idx.stride(), *valid.stride(),
        num_heads, seqlen,
        GROUP=num_heads // k.shape[1], D=head_dim, KB=kb, BITER=BISECT_ITERS,
        num_warps=2,
    )
    return y, p, pn


@_sparse_entmax_bias_attn_fwd.register_fake
def _(q, k, v, idx, valid, wire_bias, wire_gate, null_bias, scale):
    bsz, num_heads, seqlen, _ = q.shape
    return (
        q.new_empty(q.shape),
        q.new_empty(bsz, num_heads, seqlen, idx.size(-1), dtype=torch.float32),
        q.new_empty(bsz, num_heads, seqlen, dtype=torch.float32),
    )


@torch.library.custom_op("parameter_golf::sparse_entmax_attn_bias_bwd", mutates_args=())
def _sparse_entmax_bias_attn_bwd(
    dy: Tensor, q: Tensor, k: Tensor, v: Tensor, idx: Tensor, valid: Tensor,
    wire_gate: Tensor, p: Tensor, pn: Tensor, scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    bsz, num_heads, seqlen, head_dim = q.shape
    kb = idx.size(-1)
    dy = dy.contiguous()
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    dq = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    dk = torch.zeros(k.shape, device=k.device, dtype=torch.float32)
    dv = torch.zeros(v.shape, device=v.device, dtype=torch.float32)
    db = torch.zeros(_DBIAS_BANKS, num_heads, kb, device=q.device, dtype=torch.float32)
    dg = torch.zeros(_DBIAS_BANKS, num_heads, kb, device=q.device, dtype=torch.float32)
    dn = torch.empty(bsz, num_heads, seqlen, device=q.device, dtype=torch.float32)
    grid = (bsz * num_heads * seqlen,)
    _sparse_entmax_bias_bwd_kernel[grid](
        dy, q, k, v, idx, valid, wire_gate.contiguous(), p, pn,
        dq, dk, dv, db, dg, dn,
        scale,
        *q.stride(), *k.stride(), *v.stride(), *dy.stride(),
        *idx.stride(), *valid.stride(),
        num_heads, seqlen,
        GROUP=num_heads // k.shape[1], D=head_dim, KB=kb, BANKS=_DBIAS_BANKS,
        num_warps=8,
    )
    return (
        dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
        db.sum(0), dg.sum(0), dn.sum(dim=(0, 2)),
    )


@_sparse_entmax_bias_attn_bwd.register_fake
def _(dy, q, k, v, idx, valid, wire_gate, p, pn, scale):
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        q.new_empty(q.shape[1], idx.size(-1), dtype=torch.float32),
        q.new_empty(q.shape[1], idx.size(-1), dtype=torch.float32),
        q.new_empty(q.shape[1], dtype=torch.float32),
    )


def _setup_context(ctx, inputs, output):
    q, k, v, idx, valid, wire_bias, wire_gate, null_bias, scale = inputs
    _, p, pn = output
    ctx.save_for_backward(q, k, v, idx, valid, wire_gate, p, pn)
    ctx.scale = scale


def _backward(ctx, dy, dp, dpn):
    q, k, v, idx, valid, wire_gate, p, pn = ctx.saved_tensors
    dq, dk, dv, dbias, dgate, dnull = _sparse_entmax_bias_attn_bwd(
        dy, q, k, v, idx, valid, wire_gate, p, pn, ctx.scale
    )
    return dq, dk, dv, None, None, dbias, dgate, dnull, None


_sparse_entmax_bias_attn_fwd.register_autograd(_backward, setup_context=_setup_context)


# --- list-input cross-layer wrapper ------------------------------------------
# Same rationale as sparse_xlayer_kernel: the ctx saves references to the 9
# shared per-layer K/V tensors instead of a per-consumer cat copy; the cat is
# transient in fwd and rebuilt in bwd.

@torch.library.custom_op("parameter_golf::xlayer_entmax_attn_bias", mutates_args=())
def _xlayer_entmax_bias_attn_fwd(
    q: Tensor, ks: list[Tensor], vs: list[Tensor], idx: Tensor, valid: Tensor,
    wire_bias: Tensor, wire_gate: Tensor, null_bias: Tensor, scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    k_all, v_all = _cat_kv(ks, vs)
    return _sparse_entmax_bias_attn_fwd(
        q, k_all, v_all, idx, valid, wire_bias, wire_gate, null_bias, scale
    )


@_xlayer_entmax_bias_attn_fwd.register_fake
def _(q, ks, vs, idx, valid, wire_bias, wire_gate, null_bias, scale):
    bsz, num_heads, seqlen, _ = q.shape
    return (
        q.new_empty(q.shape),
        q.new_empty(bsz, num_heads, seqlen, idx.size(-1), dtype=torch.float32),
        q.new_empty(bsz, num_heads, seqlen, dtype=torch.float32),
    )


def _xlayer_setup_context(ctx, inputs, output):
    q, ks, vs, idx, valid, wire_bias, wire_gate, null_bias, scale = inputs
    _, p, pn = output
    ctx.n_seg = len(ks)
    ctx.scale = scale
    ctx.save_for_backward(q, *ks, *vs, idx, valid, wire_gate, p, pn)


def _xlayer_backward(ctx, dy, dp, dpn):
    n = ctx.n_seg
    q, *rest = ctx.saved_tensors
    ks, vs = list(rest[:n]), list(rest[n:2 * n])
    idx, valid, wire_gate, p, pn = rest[2 * n:]
    k_all, v_all = _cat_kv(ks, vs)
    seg_t = ks[0].size(2)
    dq, dk_all, dv_all, dbias, dgate, dnull = _sparse_entmax_bias_attn_bwd(
        dy, q, k_all, v_all, idx, valid, wire_gate, p, pn, ctx.scale
    )
    dks = list(dk_all.split(seg_t, dim=2))
    dvs = list(dv_all.split(seg_t, dim=2))
    return dq, dks, dvs, None, None, dbias, dgate, dnull, None


_xlayer_entmax_bias_attn_fwd.register_autograd(
    _xlayer_backward, setup_context=_xlayer_setup_context
)


def xlayer_entmax_bias_attention_stats(
    q: Tensor, ks: list[Tensor], vs: list[Tensor], idx: Tensor, valid: Tensor,
    wire_bias: Tensor, wire_gate: Tensor, null_bias: Tensor, scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """``xlayer_entmax_attention_stats`` with learned [H, K] wire bias + gate.

    ``ks``/``vs`` are the per-layer [B, Hkv, T, D] tensors (equal T); segment
    s of the virtual stack holds layer s.  Gradients flow to q, every ks/vs
    element, ``wire_bias``, ``wire_gate``, and ``null_bias``; p/pn are routing
    stats exactly as in the unbiased op."""
    return _xlayer_entmax_bias_attn_fwd(
        q.contiguous(), [k.contiguous() for k in ks], [v.contiguous() for v in vs],
        idx, valid, wire_bias.float().contiguous(), wire_gate.float().contiguous(),
        null_bias, scale,
    )
