"""Triton kernels of the fused CWM core (``pretraining/cwm/fused.py``).

Every kernel computes in fp32 from activation-dtype inputs and rounds once
on store; where the reference module path (``model.py``) rounds an
intermediate to the activation dtype (an ``rms`` or ``rope`` result), the
kernel rounds at the same point, so fp32 execution matches it to fp32
precision and bf16 execution to bf16 rounding.

Layout conventions (see ``fused.Layout``):

* ``P`` rows are one source's projections of its input state: writer
  columns (bank key/value/strength, store key/value), reader columns (bank
  query, source mixes, store query) and the two low-rank inputs.
* ``Z`` rows are the low-rank outputs: decay logits then the bank gate.
* Banks are value-major ``[S, B, H, D(v), D(k)]`` in fp32 or bf16; they are
  read into fp32, and a commit rounds its fp32 result once on store. The
  bank-state gradient ``G`` has the banks' dtype and is handled the same
  way by the commit backward.
* Store entries are ``[B, T, S, Hs, Ds]`` indexed by absolute token, so a
  token's window is the slice ``[pos - W, pos)``. Their gradients live in a
  ring ``[B, W, S, Hs, Ds]`` (slot ``tau mod W``): an entry receives
  gradient only from the W tokens after it, which the backward sweep (in
  reverse) has processed when it reaches the entry's own token; that
  token's commit backward consumes the slot and clears it for the entry
  entering the window at the same step.
* ``FB`` is the feedback mode: 0 none (lattice, or ablated), 1 full,
  2 placebo (feedback reads of row b use row ``b - shift``).

Backward kernels write per-row parameter-gradient partials (reduced once
per chunk) instead of accumulating, so no atomics are used anywhere and
results are deterministic.
"""

from __future__ import annotations

import triton
import triton.language as tl

EPS = tl.constexpr(1.1920928955078125e-07)  # torch.finfo(torch.float32).eps
L2_EPS = tl.constexpr(1e-6)
DECAY_LOWER_BOUND = tl.constexpr(-5.0)


@triton.jit
def _round(x, like_ptr):
    """Round fp32 ``x`` to ``like_ptr``'s element type and back."""
    return x.to(like_ptr.dtype.element_ty).to(tl.float32)


@triton.jit
def _rope(x1, x2, cos, sin):
    return x1 * cos + x2 * sin, x2 * cos - x1 * sin


@triton.jit
def _rope_t(d1, d2, cos, sin):
    """Transpose (inverse) of ``_rope``."""
    return d1 * cos - d2 * sin, d1 * sin + d2 * cos


# ---------------------------------------------------------------------------
# Residual stream: h = h + y + b, n = rms(h); and its backward.
# ---------------------------------------------------------------------------


@triton.jit
def _split_sum(ptr, stride_s, mask, like_ptr, SPLIT: tl.constexpr):
    """A GEMM product as its unsplit launch would have stored it: one
    activation-dtype load, or the sum of ``SPLIT`` fp32 partials rounded to
    ``like_ptr``'s type."""
    if SPLIT == 1:
        return tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(ptr, mask=mask, other=0.0)
    for i in tl.static_range(1, SPLIT):
        y += tl.load(ptr + i * stride_s, mask=mask, other=0.0)
    return _round(y, like_ptr)


@triton.jit
def residual_rms_kernel(
    h_ptr, y_ptr, b_ptr, out_h_ptr, out_n_ptr,
    stride_h, stride_ys, stride_y, stride_oh, stride_on,
    N: tl.constexpr, BLOCK: tl.constexpr, ADD: tl.constexpr, SPLIT: tl.constexpr,
):
    """``y`` is a product, or (``SPLIT > 1``) its fp32 split-K partials."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(h_ptr + row * stride_h + cols, mask=mask, other=0.0).to(tl.float32)
    if ADD:
        x += _split_sum(y_ptr + row * stride_y + cols, stride_ys, mask, out_h_ptr, SPLIT)
        x += tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = _round(x, out_h_ptr)
        tl.store(out_h_ptr + row * stride_oh + cols, x.to(out_h_ptr.dtype.element_ty), mask=mask)
    n = x * tl.rsqrt(tl.sum(x * x, 0) / N + EPS)
    tl.store(out_n_ptr + row * stride_on + cols, n.to(out_n_ptr.dtype.element_ty), mask=mask)


@triton.jit
def rms_backward_kernel(
    x_ptr, dn_ptr, dres_ptr, dx_ptr,
    stride_x, stride_dns, stride_dn, stride_dres, stride_dx,
    N: tl.constexpr, BLOCK: tl.constexpr, SPLIT: tl.constexpr,
):
    """``dx = dres + d rms(x) / dx . dn``; ``dn`` as ``y`` of
    ``residual_rms_kernel``."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    dn = _split_sum(dn_ptr + row * stride_dn + cols, stride_dns, mask, dx_ptr, SPLIT)
    dres = tl.load(dres_ptr + row * stride_dres + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 0) / N + EPS)
    y = x * r
    dx = dres + r * (dn - y * (tl.sum(dn * y, 0) / N))
    tl.store(dx_ptr + row * stride_dx + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


@triton.jit
def relu_square_kernel(f_ptr, a_ptr, n_elements, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    f = tl.load(f_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    f = tl.maximum(f, 0.0)
    tl.store(a_ptr + offs, (f * f).to(a_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Per-token linear maps: out = epilogue(a @ w^T), w addressed by strides so
# both a weight [N, K] and its transpose serve. fp32 inputs use exact fp32
# products (the CPU tests); 16-bit inputs accumulate in fp32.
# ---------------------------------------------------------------------------

LINEAR_NONE = tl.constexpr(0)
LINEAR_RELU_SQUARE = tl.constexpr(1)  # relu(x)^2
LINEAR_RELU_SQUARE_BACKWARD = tl.constexpr(2)  # 2 relu(aux) x


@triton.jit
def linear_kernel(
    a_ptr, w_ptr, bias_ptr, aux_ptr, out_ptr,
    M, N, K, KC, stride_am, stride_wn, stride_wk, stride_xm, stride_om, stride_os,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    HAS_BIAS: tl.constexpr, EPILOGUE: tl.constexpr,
):
    """Split-K: program ``(.., .., ps)`` reduces ``K`` columns ``[ps KC,
    (ps + 1) KC)`` (``KC`` a multiple of ``BK``) into fp32 partial ``ps`` of
    ``out`` (stride ``stride_os``); a consumer sums the partials. One split
    (``KC >= K``) is the plain product."""
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    ps = tl.program_id(2)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    n_ok = rn < N
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(ps * KC, ps * KC + KC, BK):
        k = k0 + rk
        k_ok = k < K
        x = tl.load(a_ptr + rm[:, None].to(tl.int64) * stride_am + k[None, :], mask=m_ok[:, None] & k_ok[None, :], other=0.0)
        y = tl.load(w_ptr + rn[None, :] * stride_wn + k[:, None] * stride_wk, mask=n_ok[None, :] & k_ok[:, None], other=0.0)
        if a_ptr.dtype.element_ty == tl.float32:
            acc = tl.dot(x, y, acc, input_precision="ieee")
        else:
            acc = tl.dot(x, y, acc)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + rn, mask=n_ok, other=0.0).to(tl.float32)[None, :]
    if EPILOGUE != LINEAR_NONE:
        # The module rounds the product (MLP pre-activation, or the gradient
        # of the activation) before the nonlinearity.
        acc = _round(acc, out_ptr)
    if EPILOGUE == LINEAR_RELU_SQUARE:
        acc = tl.maximum(acc, 0.0)
        acc = acc * acc
    elif EPILOGUE == LINEAR_RELU_SQUARE_BACKWARD:
        f = tl.load(aux_ptr + rm[:, None].to(tl.int64) * stride_xm + rn[None, :], mask=m_ok[:, None] & n_ok[None, :], other=0.0)
        acc = 2.0 * tl.maximum(f.to(tl.float32), 0.0) * acc
    out = out_ptr + ps * stride_os + rm[:, None].to(tl.int64) * stride_om + rn[None, :]
    tl.store(out, acc.to(out_ptr.dtype.element_ty), mask=m_ok[:, None] & n_ok[None, :])


# ---------------------------------------------------------------------------
# Addressed depth read over the token's earlier rounds.
# ``Y`` holds, per entry j (u, h_1, ...), [q_{j+1} | k_r, v_r for r > j].
# ---------------------------------------------------------------------------


@triton.jit
def _entry_offset(j: tl.constexpr, R: tl.constexpr, HD: tl.constexpr):
    # sum_{j' < j} HD * (1 + 2 (R - j'))
    return HD * (j + 2 * R * j - j * (j - 1))


@triton.jit
def depth_forward_kernel(
    y_ptr, o_ptr, stride_y, stride_o, scale,
    R: tl.constexpr, ROUND: tl.constexpr, HD: tl.constexpr, DD: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    cols = tl.arange(0, DD)
    row = y_ptr + b * stride_y + h * DD
    q = tl.load(row + _entry_offset(ROUND - 1, R, HD) + cols).to(tl.float32)
    q = _round(q * tl.rsqrt(tl.sum(q * q, 0) / DD + EPS), y_ptr)
    m = 0.0  # the sink: score 0, value 0
    l = 1.0
    acc = tl.zeros([DD], tl.float32)
    for j in tl.static_range(ROUND):
        base = _entry_offset(j, R, HD) + HD + 2 * (ROUND - j - 1) * HD
        k = tl.load(row + base + cols).to(tl.float32)
        k = _round(k * tl.rsqrt(tl.sum(k * k, 0) / DD + EPS), y_ptr)
        v = tl.load(row + base + HD + cols).to(tl.float32)
        s = tl.sum(q * k, 0) * scale
        m_new = tl.maximum(m, s)
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new)
        l = l * alpha + p
        acc = acc * alpha + p * v
        m = m_new
    tl.store(o_ptr + b * stride_o + h * DD + cols, (acc / l).to(o_ptr.dtype.element_ty))


@triton.jit
def depth_backward_kernel(
    y_ptr, do_ptr, dy_ptr, stride_y, stride_do, stride_dy, scale,
    R: tl.constexpr, ROUND: tl.constexpr, HD: tl.constexpr, DD: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    cols = tl.arange(0, DD)
    row = y_ptr + b * stride_y + h * DD
    drow = dy_ptr + b * stride_dy + h * DD
    q_off = _entry_offset(ROUND - 1, R, HD)
    q_raw = tl.load(row + q_off + cols).to(tl.float32)
    q_r = tl.rsqrt(tl.sum(q_raw * q_raw, 0) / DD + EPS)
    q = _round(q_raw * q_r, y_ptr)
    m = 0.0
    l = 1.0
    acc = tl.zeros([DD], tl.float32)
    for j in tl.static_range(ROUND):
        base = _entry_offset(j, R, HD) + HD + 2 * (ROUND - j - 1) * HD
        k = tl.load(row + base + cols).to(tl.float32)
        k = _round(k * tl.rsqrt(tl.sum(k * k, 0) / DD + EPS), y_ptr)
        v = tl.load(row + base + HD + cols).to(tl.float32)
        s = tl.sum(q * k, 0) * scale
        m_new = tl.maximum(m, s)
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new)
        l = l * alpha + p
        acc = acc * alpha + p * v
        m = m_new
    out = acc / l
    do = tl.load(do_ptr + b * stride_do + h * DD + cols).to(tl.float32)
    delta = tl.sum(do * out, 0)
    dq = tl.zeros([DD], tl.float32)
    for j in tl.static_range(ROUND):
        base = _entry_offset(j, R, HD) + HD + 2 * (ROUND - j - 1) * HD
        k_raw = tl.load(row + base + cols).to(tl.float32)
        k_r = tl.rsqrt(tl.sum(k_raw * k_raw, 0) / DD + EPS)
        k = _round(k_raw * k_r, y_ptr)
        v = tl.load(row + base + HD + cols).to(tl.float32)
        p = tl.exp(tl.sum(q * k, 0) * scale - m) / l
        ds = p * (tl.sum(do * v, 0) - delta)
        dq += ds * scale * k
        dk = ds * scale * q
        y = k_raw * k_r
        dk_raw = k_r * (dk - y * (tl.sum(dk * y, 0) / DD))
        tl.store(drow + base + cols, dk_raw.to(dy_ptr.dtype.element_ty))
        tl.store(drow + base + HD + cols, (p * do).to(dy_ptr.dtype.element_ty))
    y = q_raw * q_r
    dq_raw = q_r * (dq - y * (tl.sum(dq * y, 0) / DD))
    tl.store(drow + q_off + cols, dq_raw.to(dy_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Bank reads: legal sources [0, ROUND), feedback sources [ROUND, S).
# ---------------------------------------------------------------------------


@triton.jit
def _mix_weights(p_row, mix_off, h, n: tl.constexpr, SP: tl.constexpr):
    """Softmax over this head's ``n`` source logits (1 when ``n == 1``)."""
    idx = tl.arange(0, SP)
    if n == 1:
        w = tl.where(idx == 0, 1.0, 0.0)
    else:
        logits = tl.load(p_row + mix_off + h * n + idx, mask=idx < n, other=float("-inf")).to(tl.float32)
        e = tl.exp(logits - tl.max(logits, 0))
        w = e / tl.sum(e, 0)
    return w


@triton.jit
def _pick(vec, i: tl.constexpr, SP: tl.constexpr):
    return tl.sum(tl.where(tl.arange(0, SP) == i, vec, 0.0), 0)


@triton.jit
def _bank_read_forward(
    b, h, m_ptr, p_ptr, z_ptr, wn_ptr, mixed_ptr, ro_ptr,
    B, shift, stride_p, stride_z, stride_ro,
    q_off, mix_off, scale,
    S: tl.constexpr, H: tl.constexpr, D: tl.constexpr, ROUND: tl.constexpr,
    FB: tl.constexpr, SP: tl.constexpr,
):
    """Row b, head h of one round's bank reads. ``mixed_ptr`` ``[2, B, H,
    D]`` keeps each path's source mixture (before its norm) for the
    backward."""
    b64 = b.to(tl.int64)
    vi = tl.arange(0, D)
    ki = tl.arange(0, D)
    tile = vi[:, None] * D + ki[None, :]
    p_row = p_ptr + b64 * stride_p
    q = tl.load(p_row + q_off + h * D + ki).to(tl.float32)
    q = q * tl.rsqrt(tl.sum(q * q, 0) + L2_EPS) * scale
    gate = tl.load(z_ptr + b64 * stride_z + H * D + h * D + vi).to(tl.float32)
    sg = tl.sigmoid(gate)
    w_legal = _mix_weights(p_row, mix_off, h, ROUND, SP)
    mixed = tl.zeros([D], tl.float32)
    for s in tl.static_range(ROUND):
        M = tl.load(m_ptr + ((s * B + b64) * H + h) * D * D + tile).to(tl.float32)
        mixed += _pick(w_legal, s, SP) * tl.sum(M * q[None, :], 1)
    tl.store(mixed_ptr + (b64 * H + h) * D + vi, mixed)
    out = mixed * tl.rsqrt(tl.sum(mixed * mixed, 0) / D + EPS) * tl.load(wn_ptr + vi) * sg
    ro_row = ro_ptr + b64 * stride_ro
    tl.store(ro_row + h * D + vi, out.to(ro_ptr.dtype.element_ty))
    if FB == 0:
        tl.store(ro_row + H * D + h * D + vi, tl.zeros([D], tl.float32).to(ro_ptr.dtype.element_ty))
    else:
        src = b64
        if FB == 2:
            src = ((b + B - shift) % B).to(tl.int64)
        w_fb = _mix_weights(p_row, mix_off + H * ROUND, h, S - ROUND, SP)
        mixed = tl.zeros([D], tl.float32)
        for s in tl.static_range(ROUND, S):
            M = tl.load(m_ptr + ((s * B + src) * H + h) * D * D + tile).to(tl.float32)
            mixed += _pick(w_fb, s - ROUND, SP) * tl.sum(M * q[None, :], 1)
        tl.store(mixed_ptr + ((B + b64) * H + h) * D + vi, mixed)
        out = mixed * tl.rsqrt(tl.sum(mixed * mixed, 0) / D + EPS) * tl.load(wn_ptr + D + vi) * sg
        tl.store(ro_row + H * D + h * D + vi, out.to(ro_ptr.dtype.element_ty))


@triton.jit
def _bank_path_backward(
    m_ptr, mixed_ptr, pend_ptr, dmix_ptr, B, b64, src, h, q, d_out, wn, sg,
    w, lo: tl.constexpr, hi: tl.constexpr, n_mix_logits: tl.constexpr,
    H: tl.constexpr, D: tl.constexpr, SP: tl.constexpr,
):
    """Backward of one read path (sources [lo, hi)) from its saved mixture.
    With ``y_s = M_s^T d_mixed``: ``dq = sum_s w_s y_s`` and the mix weight
    gradient is ``<d_mixed, M_s q> = <y_s, q>``. Returns this path's (dq,
    d gate, d norm weight); writes the read gradients ``w_s d_mixed`` to
    ``pend`` and the mix-logit gradients to ``dmix_ptr``."""
    vi = tl.arange(0, D)
    tile = vi[:, None] * D + vi[None, :]
    mixed = tl.load(mixed_ptr + vi)
    r = tl.rsqrt(tl.sum(mixed * mixed, 0) / D + EPS)
    n = mixed * r
    d_n = d_out * wn * sg
    d_gate = d_out * n * wn * sg * (1.0 - sg)
    d_wn = d_out * n * sg
    d_mixed = r * (d_n - n * (tl.sum(d_n * n, 0) / D))
    dq = tl.zeros([D], tl.float32)
    dw = tl.zeros([SP], tl.float32)
    idx = tl.arange(0, SP)
    for s in tl.static_range(lo, hi):
        w_s = _pick(w, s - lo, SP)
        tl.store(pend_ptr + ((s * B + b64) * H + h) * D + vi, w_s * d_mixed)
        M = tl.load(m_ptr + ((s * B + src) * H + h) * D * D + tile).to(tl.float32)
        y = tl.sum(M * d_mixed[:, None], 0)
        dq += w_s * y
        dw += tl.where(idx == s - lo, tl.sum(y * q, 0), 0.0)
    if n_mix_logits > 1:
        d_logits = w * (dw - tl.sum(w * dw, 0))
        tl.store(dmix_ptr + h * n_mix_logits + idx, d_logits.to(dmix_ptr.dtype.element_ty), mask=idx < n_mix_logits)
    return dq, d_gate, d_wn


@triton.jit
def _bank_read_backward(
    b, h, m_ptr, p_ptr, z_ptr, wn_ptr, mixed_ptr, dro_ptr,
    dp_ptr, dz_ptr, dwn_ptr, pend_ptr, pend_q_ptr,
    B, shift, stride_p, stride_z, stride_dro, stride_dp, stride_dz,
    q_off, mix_off, scale,
    S: tl.constexpr, H: tl.constexpr, D: tl.constexpr, ROUND: tl.constexpr,
    FB: tl.constexpr, SP: tl.constexpr,
):
    """Gradients of row b, head h of one round's bank reads. ``pend`` receives each source's
    read gradient (indexed by the reader row) and ``pend_q`` the query: the
    bank-state gradient ``d_read q^T`` is applied by the previous token's
    commit backward. ``dwn`` gets this row's norm-weight partials."""
    b64 = b.to(tl.int64)
    vi = tl.arange(0, D)
    p_row = p_ptr + b64 * stride_p
    q_raw = tl.load(p_row + q_off + h * D + vi).to(tl.float32)
    q_r = tl.rsqrt(tl.sum(q_raw * q_raw, 0) + L2_EPS)
    q = q_raw * q_r * scale
    tl.store(pend_q_ptr + (b64 * H + h) * D + vi, q)
    gate = tl.load(z_ptr + b64 * stride_z + H * D + h * D + vi).to(tl.float32)
    sg = tl.sigmoid(gate)
    dro_row = dro_ptr + b64 * stride_dro
    dp_row = dp_ptr + b64 * stride_dp
    w = _mix_weights(p_row, mix_off, h, ROUND, SP)
    d_out = tl.load(dro_row + h * D + vi).to(tl.float32)
    dq, d_gate, d_wn = _bank_path_backward(
        m_ptr, mixed_ptr + (b64 * H + h) * D, pend_ptr, dp_row + mix_off, B, b64, b64, h, q, d_out,
        tl.load(wn_ptr + vi), sg, w, 0, ROUND, ROUND, H, D, SP,
    )
    tl.store(dwn_ptr + ((b64 * H + h) * 2) * D + vi, d_wn)
    if FB == 0:
        tl.store(dwn_ptr + ((b64 * H + h) * 2 + 1) * D + vi, tl.zeros([D], tl.float32))
        for s in tl.static_range(ROUND, S):
            tl.store(pend_ptr + ((s * B + b64) * H + h) * D + vi, tl.zeros([D], tl.float32))
    else:
        src = b64
        if FB == 2:
            src = ((b + B - shift) % B).to(tl.int64)
        w = _mix_weights(p_row, mix_off + H * ROUND, h, S - ROUND, SP)
        d_out = tl.load(dro_row + H * D + h * D + vi).to(tl.float32)
        dq_fb, d_gate_fb, d_wn = _bank_path_backward(
            m_ptr, mixed_ptr + ((B + b64) * H + h) * D, pend_ptr, dp_row + mix_off + H * ROUND, B, b64, src, h, q,
            d_out, tl.load(wn_ptr + D + vi), sg, w, ROUND, S, S - ROUND, H, D, SP,
        )
        dq += dq_fb
        d_gate += d_gate_fb
        tl.store(dwn_ptr + ((b64 * H + h) * 2 + 1) * D + vi, d_wn)
    tl.store(dz_ptr + b64 * stride_dz + H * D + h * D + vi, d_gate.to(dz_ptr.dtype.element_ty))
    dy = dq * scale
    y = q_raw * q_r
    dq_raw = q_r * (dy - y * tl.sum(dy * y, 0))
    tl.store(dp_row + q_off + h * D + vi, dq_raw.to(dp_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Commit: every source's delta-rule write, once per token.
# ---------------------------------------------------------------------------


@triton.jit
def _writes(p_ptr, z_ptr, dt_ptr, alog_ptr, s, b64, h, stride_ps, stride_p, stride_zs, stride_z,
            k_off, v_off, beta_off, beta_max, H: tl.constexpr, D: tl.constexpr):
    i = tl.arange(0, D)
    p_row = p_ptr + s * stride_ps + b64 * stride_p
    k_raw = tl.load(p_row + k_off + h * D + i).to(tl.float32)
    k_r = tl.rsqrt(tl.sum(k_raw * k_raw, 0) + L2_EPS)
    v = tl.load(p_row + v_off + h * D + i).to(tl.float32)
    beta_sig = tl.sigmoid(tl.load(p_row + beta_off + h).to(tl.float32))
    logit = tl.load(z_ptr + s * stride_zs + b64 * stride_z + h * D + i).to(tl.float32)
    logit += tl.load(dt_ptr + s * H * D + h * D + i)
    a_exp = tl.exp(tl.load(alog_ptr + s * H + h))
    arg = a_exp * logit
    arg_sig = tl.sigmoid(arg)
    g = DECAY_LOWER_BOUND * arg_sig
    return k_raw, k_r, v, beta_sig, beta_max * beta_sig, a_exp, arg, arg_sig, g


@triton.jit
def _commit_forward(
    s, b64, h, m_ptr, out_ptr, p_ptr, z_ptr, dt_ptr, alog_ptr,
    B, stride_ps, stride_p, stride_zs, stride_z, k_off, v_off, beta_off, beta_max,
    H: tl.constexpr, D: tl.constexpr,
):
    k_raw, k_r, v, _bs, beta, _ae, _arg, _as, g = _writes(
        p_ptr, z_ptr, dt_ptr, alog_ptr, s, b64, h, stride_ps, stride_p, stride_zs, stride_z,
        k_off, v_off, beta_off, beta_max, H, D,
    )
    key = k_raw * k_r
    i = tl.arange(0, D)
    tile = i[:, None] * D + i[None, :]
    base = ((s * B + b64) * H + h) * D * D
    A = tl.load(m_ptr + base + tile).to(tl.float32) * tl.exp(g)[None, :]
    pred = tl.sum(A * key[None, :], 1)
    new = A + (beta * (v - pred))[:, None] * key[None, :]
    tl.store(out_ptr + base + tile, new.to(out_ptr.dtype.element_ty))


@triton.jit
def _commit_backward(
    s, b, h, m_ptr, g_ptr, pend_ptr, pend_q_ptr, p_ptr, z_ptr, dt_ptr, alog_ptr,
    dp_ptr, dz_ptr, dalog_ptr, ddt_ptr,
    B, shift, stride_ps, stride_p, stride_zs, stride_z, stride_dps, stride_dp, stride_dzs, stride_dz,
    k_off, v_off, beta_off, beta_max,
    R: tl.constexpr, H: tl.constexpr, D: tl.constexpr, FB: tl.constexpr,
):
    """``g_ptr`` holds the dense part of dL/dM_{t+1} on entry and dL/dM_t's
    dense part on exit. The next token's reads, which also saw M_{t+1}, are
    folded in here from ``pend`` (rank-one terms ``d_read q^T``)."""
    b64 = b.to(tl.int64)
    i = tl.arange(0, D)
    tile = i[:, None] * D + i[None, :]
    base = ((s * B + b64) * H + h) * D * D
    G = tl.load(g_ptr + base + tile).to(tl.float32)
    for rr in tl.static_range(R):
        # Round rr + 1 read source s legally from its own row, or (feedback)
        # from row b's memory on behalf of reader row b + shift (placebo).
        legal = s < rr + 1
        reader = b64
        if FB == 2:
            reader = tl.where(legal, b64, ((b + shift) % B).to(tl.int64))
        # FB == 0 left the feedback entries of ``pend`` unread (zero).
        d_read = tl.load(pend_ptr + (((rr * (R + 1) + s) * B + reader) * H + h) * D + i)
        q = tl.load(pend_q_ptr + ((rr * B + reader) * H + h) * D + i)
        G += d_read[:, None] * q[None, :]
    k_raw, k_r, v, beta_sig, beta, a_exp, arg, arg_sig, g = _writes(
        p_ptr, z_ptr, dt_ptr, alog_ptr, s, b64, h, stride_ps, stride_p, stride_zs, stride_z,
        k_off, v_off, beta_off, beta_max, H, D,
    )
    key = k_raw * k_r
    decay = tl.exp(g)
    A = tl.load(m_ptr + base + tile).to(tl.float32) * decay[None, :]
    err = v - tl.sum(A * key[None, :], 1)
    Gk = tl.sum(G * key[None, :], 1)
    dA = G - (beta * Gk)[:, None] * key[None, :]
    tl.store(g_ptr + base + tile, (dA * decay[None, :]).to(g_ptr.dtype.element_ty))
    dv = beta * Gk
    dbeta = tl.sum(Gk * err, 0)
    dkey = beta * tl.sum(G * err[:, None], 0) - beta * tl.sum(A * Gk[:, None], 0)
    dg = tl.sum(dA * A, 0)
    dk_raw = k_r * (dkey - key * tl.sum(dkey * key, 0))
    dbeta_raw = dbeta * beta_max * beta_sig * (1.0 - beta_sig)
    darg = dg * DECAY_LOWER_BOUND * arg_sig * (1.0 - arg_sig)
    dp_row = dp_ptr + s * stride_dps + b64 * stride_dp
    tl.store(dp_row + k_off + h * D + i, dk_raw.to(dp_ptr.dtype.element_ty))
    tl.store(dp_row + v_off + h * D + i, dv.to(dp_ptr.dtype.element_ty))
    tl.store(dp_row + beta_off + h, dbeta_raw.to(dp_ptr.dtype.element_ty))
    ddt = darg * a_exp
    tl.store(dz_ptr + s * stride_dzs + b64 * stride_dz + h * D + i, ddt.to(dz_ptr.dtype.element_ty))
    # fp32 partials of the per-source decay parameters (dt_bias: ddt).
    tl.store(dalog_ptr + (s * B + b64) * H + h, tl.sum(darg * arg, 0))
    tl.store(ddt_ptr + ((s * B + b64) * H + h) * D + i, ddt)


# ---------------------------------------------------------------------------
# Exact windowed store.
# ---------------------------------------------------------------------------


@triton.jit
def _store_query(p_row, sq_off, h, freq_ptr, posf, like_ptr, DS: tl.constexpr):
    """q = rope(round(rms(q_raw))), as halves; also returns what the
    backward needs."""
    half = tl.arange(0, DS // 2)
    x1 = tl.load(p_row + sq_off + h * DS + half).to(tl.float32)
    x2 = tl.load(p_row + sq_off + h * DS + DS // 2 + half).to(tl.float32)
    r = tl.rsqrt((tl.sum(x1 * x1, 0) + tl.sum(x2 * x2, 0)) / DS + EPS)
    theta = posf * tl.load(freq_ptr + half)
    cos = tl.cos(theta)
    sin = tl.sin(theta)
    q1, q2 = _rope(_round(x1 * r, like_ptr), _round(x2 * r, like_ptr), cos, sin)
    return _round(q1, like_ptr), _round(q2, like_ptr), x1, x2, r, cos, sin


@triton.jit
def _store_path_forward(
    ek_ptr, ev_ptr, row, h, q1, q2, start, pos, T, scale,
    lo: tl.constexpr, hi: tl.constexpr, S: tl.constexpr, HS: tl.constexpr, DS: tl.constexpr,
    W: tl.constexpr, WP: tl.constexpr,
):
    half = tl.arange(0, DS // 2)
    n = tl.arange(0, WP)
    tau = start + n
    valid = (n < W) & (tau >= 0) & (tau < pos)
    m = 0.0
    l = 1.0
    acc1 = tl.zeros([DS // 2], tl.float32)
    acc2 = tl.zeros([DS // 2], tl.float32)
    for s in tl.static_range(lo, hi):
        entry = (((row * T + tau) * S + s) * HS + h) * DS
        k1 = tl.load(ek_ptr + entry[:, None] + half[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        k2 = tl.load(ek_ptr + entry[:, None] + DS // 2 + half[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        score = (tl.sum(k1 * q1[None, :], 1) + tl.sum(k2 * q2[None, :], 1)) * scale
        score = tl.where(valid, score, float("-inf"))
        m_new = tl.maximum(m, tl.max(score, 0))
        alpha = tl.exp(m - m_new)
        p = tl.exp(score - m_new)
        l = l * alpha + tl.sum(p, 0)
        v1 = tl.load(ev_ptr + entry[:, None] + half[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        v2 = tl.load(ev_ptr + entry[:, None] + DS // 2 + half[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        acc1 = acc1 * alpha + tl.sum(p[:, None] * v1, 0)
        acc2 = acc2 * alpha + tl.sum(p[:, None] * v2, 0)
        m = m_new
    return acc1 / l, acc2 / l, m, l


@triton.jit
def _store_read_forward(
    b, h, ek_ptr, ev_ptr, p_ptr, freq_ptr, pos_ptr, ro_ptr, stats_ptr,
    B, T, shift, step, stride_p, stride_ro, sq_off, ro_off, scale,
    S: tl.constexpr, HS: tl.constexpr, DS: tl.constexpr, W: tl.constexpr, WP: tl.constexpr,
    ROUND: tl.constexpr, FB: tl.constexpr,
):
    """Row b, store head h of one round's store reads. ``stats_ptr`` ``[B,
    HS, 2, 2]`` keeps each path's softmax statistics (max, normalizer) for
    the backward."""
    b64 = b.to(tl.int64)
    pos = tl.load(pos_ptr) + step
    half = tl.arange(0, DS // 2)
    q1, q2, _x1, _x2, _r, _c, _s = _store_query(p_ptr + b64 * stride_p, sq_off, h, freq_ptr, pos.to(tl.float32), ro_ptr, DS)
    start = pos - W
    ro_row = ro_ptr + b64 * stride_ro + ro_off
    o1, o2, m, l = _store_path_forward(ek_ptr, ev_ptr, b64, h, q1, q2, start, pos, T, scale, 0, ROUND, S, HS, DS, W, WP)
    stats = stats_ptr + (b64 * HS + h) * 4
    tl.store(stats, m)
    tl.store(stats + 1, l)
    tl.store(ro_row + h * DS + half, o1.to(ro_ptr.dtype.element_ty))
    tl.store(ro_row + h * DS + DS // 2 + half, o2.to(ro_ptr.dtype.element_ty))
    if FB == 0:
        o1 = tl.zeros([DS // 2], tl.float32)
        o2 = tl.zeros([DS // 2], tl.float32)
    else:
        src = b64
        if FB == 2:
            src = ((b + B - shift) % B).to(tl.int64)
        o1, o2, m, l = _store_path_forward(ek_ptr, ev_ptr, src, h, q1, q2, start, pos, T, scale, ROUND, S, S, HS, DS, W, WP)
        tl.store(stats + 2, m)
        tl.store(stats + 3, l)
    tl.store(ro_row + HS * DS + h * DS + half, o1.to(ro_ptr.dtype.element_ty))
    tl.store(ro_row + HS * DS + h * DS + DS // 2 + half, o2.to(ro_ptr.dtype.element_ty))


@triton.jit
def _store_path_backward(
    ek_ptr, ev_ptr, sp_ptr, sd_ptr, row, b64, h, q1, q2, d1, d2, m, l, start, pos, T, scale,
    lo: tl.constexpr, hi: tl.constexpr, S: tl.constexpr, HS: tl.constexpr, DS: tl.constexpr,
    W: tl.constexpr, WP: tl.constexpr,
):
    """dq of one path and its ``delta = sum p dp``, in one pass over the
    entries given the forward's softmax statistics ``(m, l)``: with score
    gradients ``ds = p (dp - delta)``, ``dq = scale (sum p dp k - delta sum
    p k)``. Stores the weights ``p`` and ``dp = <v, d>`` of its entries
    (``[B, HS, S, WP]``) for the deferred key/value gradients. The sums
    over sources accumulate per entry, so the entries (spread over warps)
    are reduced once per path rather than once per source."""
    half = tl.arange(0, DS // 2)
    n = tl.arange(0, WP)
    tau = start + n
    valid = (n < W) & (tau >= 0) & (tau < pos)
    a1 = tl.zeros([WP, DS // 2], tl.float32)
    a2 = tl.zeros([WP, DS // 2], tl.float32)
    c1 = tl.zeros([WP, DS // 2], tl.float32)
    c2 = tl.zeros([WP, DS // 2], tl.float32)
    pdp_sum = tl.zeros([WP], tl.float32)
    for s in tl.static_range(lo, hi):
        entry = (((row * T + tau) * S + s) * HS + h) * DS
        k1 = tl.load(ek_ptr + entry[:, None] + half[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        k2 = tl.load(ek_ptr + entry[:, None] + DS // 2 + half[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        v1 = tl.load(ev_ptr + entry[:, None] + half[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        v2 = tl.load(ev_ptr + entry[:, None] + DS // 2 + half[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        score = (tl.sum(k1 * q1[None, :], 1) + tl.sum(k2 * q2[None, :], 1)) * scale
        p = tl.where(valid, tl.exp(score - m) / l, 0.0)
        dp = tl.sum(v1 * d1[None, :], 1) + tl.sum(v2 * d2[None, :], 1)
        pdp = p * dp
        pdp_sum += pdp
        a1 += pdp[:, None] * k1
        a2 += pdp[:, None] * k2
        c1 += p[:, None] * k1
        c2 += p[:, None] * k2
        slot = ((b64 * HS + h) * S + s) * WP + n
        tl.store(sp_ptr + slot, p)
        tl.store(sd_ptr + slot, dp)
    delta = tl.sum(pdp_sum, 0)
    dq1 = (tl.sum(a1, 0) - delta * tl.sum(c1, 0)) * scale
    dq2 = (tl.sum(a2, 0) - delta * tl.sum(c2, 0)) * scale
    return dq1, dq2, delta


@triton.jit
def _store_read_backward(
    b, h, ek_ptr, ev_ptr, p_ptr, freq_ptr, pos_ptr, dro_ptr, dp_ptr, stats_ptr,
    sp_ptr, sd_ptr, sq_ptr, sdo_ptr, sdelta_ptr,
    B, T, shift, step, stride_p, stride_dro, stride_dp, sq_off, ro_off, scale,
    S: tl.constexpr, HS: tl.constexpr, DS: tl.constexpr, W: tl.constexpr, WP: tl.constexpr,
    ROUND: tl.constexpr, FB: tl.constexpr,
):
    """dq of row b, store head h of one round's store reads. The rotated query, both paths' output
    gradients and deltas go to ``sq``/``sdo``/``sdelta`` and the per-entry
    weights and value-gradient dots to ``sp``/``sd``: ``store_grad_kernel``
    turns them into key/value gradients once per token."""
    b64 = b.to(tl.int64)
    pos = tl.load(pos_ptr) + step
    half = tl.arange(0, DS // 2)
    p_row = p_ptr + b64 * stride_p
    q1, q2, x1, x2, r, cos, sin = _store_query(p_row, sq_off, h, freq_ptr, pos.to(tl.float32), dro_ptr, DS)
    qrow = sq_ptr + (b64 * HS + h) * DS
    tl.store(qrow + half, q1)
    tl.store(qrow + DS // 2 + half, q2)
    start = pos - W
    dro_row = dro_ptr + b64 * stride_dro + ro_off
    d1 = tl.load(dro_row + h * DS + half).to(tl.float32)
    d2 = tl.load(dro_row + h * DS + DS // 2 + half).to(tl.float32)
    dorow = sdo_ptr + ((b64 * HS + h) * 2) * DS
    tl.store(dorow + half, d1)
    tl.store(dorow + DS // 2 + half, d2)
    stats = stats_ptr + (b64 * HS + h) * 4
    drow = sdelta_ptr + (b64 * HS + h) * 2
    dq1, dq2, delta = _store_path_backward(
        ek_ptr, ev_ptr, sp_ptr, sd_ptr, b64, b64, h, q1, q2, d1, d2, tl.load(stats), tl.load(stats + 1),
        start, pos, T, scale, 0, ROUND, S, HS, DS, W, WP,
    )
    tl.store(drow, delta)
    if FB == 0:
        n = tl.arange(0, WP)
        for s in tl.static_range(ROUND, S):
            slot = ((b64 * HS + h) * S + s) * WP + n
            tl.store(sp_ptr + slot, tl.zeros([WP], tl.float32))
            tl.store(sd_ptr + slot, tl.zeros([WP], tl.float32))
        tl.store(dorow + DS + half, tl.zeros([DS // 2], tl.float32))
        tl.store(dorow + DS + DS // 2 + half, tl.zeros([DS // 2], tl.float32))
        tl.store(drow + 1, 0.0)
    else:
        src = b64
        if FB == 2:
            src = ((b + B - shift) % B).to(tl.int64)
        d1 = tl.load(dro_row + HS * DS + h * DS + half).to(tl.float32)
        d2 = tl.load(dro_row + HS * DS + h * DS + DS // 2 + half).to(tl.float32)
        tl.store(dorow + DS + half, d1)
        tl.store(dorow + DS + DS // 2 + half, d2)
        f1, f2, delta = _store_path_backward(
            ek_ptr, ev_ptr, sp_ptr, sd_ptr, src, b64, h, q1, q2, d1, d2, tl.load(stats + 2), tl.load(stats + 3),
            start, pos, T, scale, ROUND, S, S, HS, DS, W, WP,
        )
        tl.store(drow + 1, delta)
        dq1 += f1
        dq2 += f2
    # Through the rotation, then the rms.
    dx1, dx2 = _rope_t(dq1, dq2, cos, sin)
    y1 = x1 * r
    y2 = x2 * r
    dot = (tl.sum(dx1 * y1, 0) + tl.sum(dx2 * y2, 0)) / DS
    dp_row = dp_ptr + b64 * stride_dp + sq_off + h * DS
    tl.store(dp_row + half, (r * (dx1 - y1 * dot)).to(dp_ptr.dtype.element_ty))
    tl.store(dp_row + DS // 2 + half, (r * (dx2 - y2 * dot)).to(dp_ptr.dtype.element_ty))


@triton.jit
def store_grad_kernel(
    dek_ptr, dev_ptr, sp_ptr, sd_ptr, sq_ptr, sdo_ptr, sdelta_ptr, pos_ptr,
    B, shift, step, scale,
    R: tl.constexpr, S: tl.constexpr, HS: tl.constexpr, DS: tl.constexpr,
    W: tl.constexpr, WP: tl.constexpr, FB: tl.constexpr,
):
    """Key/value gradients of every entry token ``pos`` read, from all its
    rounds (``sp``/``sd``/``sq``/``sdo``/``sdelta`` stacked over rounds),
    added to the gradient ring."""
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)
    b64 = b.to(tl.int64)
    pos = tl.load(pos_ptr) + step
    half = tl.arange(0, DS // 2)
    n = tl.arange(0, WP)
    tau = pos - W + n
    valid = (n < W) & (tau >= 0)
    dk1 = tl.zeros([WP, DS // 2], tl.float32)
    dk2 = tl.zeros([WP, DS // 2], tl.float32)
    dv1 = tl.zeros([WP, DS // 2], tl.float32)
    dv2 = tl.zeros([WP, DS // 2], tl.float32)
    for rr in tl.static_range(R):
        legal = s < rr + 1
        reader = b64
        if FB == 2:
            reader = tl.where(legal, b64, ((b + shift) % B).to(tl.int64))
        # Under FB == 0 the feedback entries' weights were stored as zero.
        slot = (((rr * B + reader) * HS + h) * S + s) * WP + n
        path = tl.where(legal, 0, 1)
        p = tl.load(sp_ptr + slot)
        delta = tl.load(sdelta_ptr + ((rr * B + reader) * HS + h) * 2 + path)
        ds = p * (tl.load(sd_ptr + slot) - delta) * scale
        qrow = sq_ptr + ((rr * B + reader) * HS + h) * DS
        q1 = tl.load(qrow + half)
        q2 = tl.load(qrow + DS // 2 + half)
        dorow = sdo_ptr + (((rr * B + reader) * HS + h) * 2 + path) * DS
        g1 = tl.load(dorow + half)
        g2 = tl.load(dorow + DS // 2 + half)
        dk1 += ds[:, None] * q1[None, :]
        dk2 += ds[:, None] * q2[None, :]
        dv1 += p[:, None] * g1[None, :]
        dv2 += p[:, None] * g2[None, :]
    entry = (((b64 * W + (tau + W) % W) * S + s) * HS + h) * DS
    ptr1 = entry[:, None] + half[None, :]
    ptr2 = ptr1 + DS // 2
    mask = valid[:, None]
    tl.store(dek_ptr + ptr1, tl.load(dek_ptr + ptr1, mask=mask, other=0.0) + dk1, mask=mask)
    tl.store(dek_ptr + ptr2, tl.load(dek_ptr + ptr2, mask=mask, other=0.0) + dk2, mask=mask)
    tl.store(dev_ptr + ptr1, tl.load(dev_ptr + ptr1, mask=mask, other=0.0) + dv1, mask=mask)
    tl.store(dev_ptr + ptr2, tl.load(dev_ptr + ptr2, mask=mask, other=0.0) + dv2, mask=mask)


@triton.jit
def _store_append_forward(
    s, b64, h, p_ptr, tag_ptr, freq_ptr, pos_ptr, ek_ptr, ev_ptr,
    T, step, stride_ps, stride_p, sk_off, sv_off,
    S: tl.constexpr, HS: tl.constexpr, DS: tl.constexpr,
):
    """Token ``pos``'s entries: key rope(round(round(rms(k)) + tag)), value."""
    pos = tl.load(pos_ptr) + step
    half = tl.arange(0, DS // 2)
    p_row = p_ptr + s * stride_ps + b64 * stride_p
    x1 = tl.load(p_row + sk_off + h * DS + half).to(tl.float32)
    x2 = tl.load(p_row + sk_off + h * DS + DS // 2 + half).to(tl.float32)
    r = tl.rsqrt((tl.sum(x1 * x1, 0) + tl.sum(x2 * x2, 0)) / DS + EPS)
    tag = tag_ptr + (s * HS + h) * DS
    a1 = _round(_round(x1 * r, ek_ptr) + _round(tl.load(tag + half), ek_ptr), ek_ptr)
    a2 = _round(_round(x2 * r, ek_ptr) + _round(tl.load(tag + DS // 2 + half), ek_ptr), ek_ptr)
    theta = pos.to(tl.float32) * tl.load(freq_ptr + half)
    k1, k2 = _rope(a1, a2, tl.cos(theta), tl.sin(theta))
    entry = (((b64 * T + pos) * S + s) * HS + h) * DS
    tl.store(ek_ptr + entry + half, k1.to(ek_ptr.dtype.element_ty))
    tl.store(ek_ptr + entry + DS // 2 + half, k2.to(ek_ptr.dtype.element_ty))
    v = tl.load(p_row + sv_off + h * DS + tl.arange(0, DS))
    tl.store(ev_ptr + entry + tl.arange(0, DS), v.to(ev_ptr.dtype.element_ty))


@triton.jit
def _store_append_backward(
    s, b64, h, p_ptr, freq_ptr, pos_ptr, dek_ptr, dev_ptr, dp_ptr, dtag_ptr,
    B, step, stride_ps, stride_p, stride_dps, stride_dp, sk_off, sv_off,
    S: tl.constexpr, HS: tl.constexpr, DS: tl.constexpr, W: tl.constexpr,
):
    """Backward of token ``pos``'s entries from their (complete) gradients,
    which it takes out of the ring, clearing the slot."""
    pos = tl.load(pos_ptr) + step
    half = tl.arange(0, DS // 2)
    full = tl.arange(0, DS)
    entry = (((b64 * W + pos % W) * S + s) * HS + h) * DS
    theta = pos.to(tl.float32) * tl.load(freq_ptr + half)
    d1, d2 = _rope_t(tl.load(dek_ptr + entry + half), tl.load(dek_ptr + entry + DS // 2 + half), tl.cos(theta), tl.sin(theta))
    dv = tl.load(dev_ptr + entry + full)
    tl.store(dek_ptr + entry + full, tl.zeros([DS], tl.float32))
    tl.store(dev_ptr + entry + full, tl.zeros([DS], tl.float32))
    trow = dtag_ptr + ((s * B + b64) * HS + h) * DS
    tl.store(trow + half, d1)
    tl.store(trow + DS // 2 + half, d2)
    p_row = p_ptr + s * stride_ps + b64 * stride_p
    x1 = tl.load(p_row + sk_off + h * DS + half).to(tl.float32)
    x2 = tl.load(p_row + sk_off + h * DS + DS // 2 + half).to(tl.float32)
    r = tl.rsqrt((tl.sum(x1 * x1, 0) + tl.sum(x2 * x2, 0)) / DS + EPS)
    y1 = x1 * r
    y2 = x2 * r
    dot = (tl.sum(d1 * y1, 0) + tl.sum(d2 * y2, 0)) / DS
    dp_row = dp_ptr + s * stride_dps + b64 * stride_dp
    tl.store(dp_row + sk_off + h * DS + half, (r * (d1 - y1 * dot)).to(dp_ptr.dtype.element_ty))
    tl.store(dp_row + sk_off + h * DS + DS // 2 + half, (r * (d2 - y2 * dot)).to(dp_ptr.dtype.element_ty))
    tl.store(dp_row + sv_off + h * DS + full, dv.to(dp_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Launches. A round's bank and store reads are independent, as are a token's
# commit and store append, so each pair is one launch: the bank heads take
# the first ``H`` programs of the head axis and the store heads the rest
# (``STORE`` off: bank heads only, the store arguments unused).
# ---------------------------------------------------------------------------


@triton.jit
def read_forward_kernel(
    m_ptr, p_ptr, z_ptr, wn_ptr, mixed_ptr, ro_ptr, ek_ptr, ev_ptr, freq_ptr, pos_ptr, stats_ptr,
    B, T, shift, step, stride_p, stride_z, stride_ro,
    q_off, mix_off, sq_off, ro_off, scale, store_scale,
    S: tl.constexpr, H: tl.constexpr, D: tl.constexpr, HS: tl.constexpr, DS: tl.constexpr,
    W: tl.constexpr, WP: tl.constexpr, ROUND: tl.constexpr, FB: tl.constexpr, SP: tl.constexpr,
    STORE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    if h < H:
        _bank_read_forward(
            b, h, m_ptr, p_ptr, z_ptr, wn_ptr, mixed_ptr, ro_ptr,
            B, shift, stride_p, stride_z, stride_ro, q_off, mix_off, scale,
            S, H, D, ROUND, FB, SP,
        )
    elif STORE:
        _store_read_forward(
            b, h - H, ek_ptr, ev_ptr, p_ptr, freq_ptr, pos_ptr, ro_ptr, stats_ptr,
            B, T, shift, step, stride_p, stride_ro, sq_off, ro_off, store_scale,
            S, HS, DS, W, WP, ROUND, FB,
        )


@triton.jit
def read_backward_kernel(
    m_ptr, p_ptr, z_ptr, wn_ptr, mixed_ptr, dro_ptr, dp_ptr, dz_ptr, dwn_ptr, pend_ptr, pend_q_ptr,
    ek_ptr, ev_ptr, freq_ptr, pos_ptr, stats_ptr, sp_ptr, sd_ptr, sq_ptr, sdo_ptr, sdelta_ptr,
    B, T, shift, step, stride_p, stride_z, stride_dro, stride_dp, stride_dz,
    q_off, mix_off, sq_off, ro_off, scale, store_scale,
    S: tl.constexpr, H: tl.constexpr, D: tl.constexpr, HS: tl.constexpr, DS: tl.constexpr,
    W: tl.constexpr, WP: tl.constexpr, ROUND: tl.constexpr, FB: tl.constexpr, SP: tl.constexpr,
    STORE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    if h < H:
        _bank_read_backward(
            b, h, m_ptr, p_ptr, z_ptr, wn_ptr, mixed_ptr, dro_ptr, dp_ptr, dz_ptr, dwn_ptr, pend_ptr, pend_q_ptr,
            B, shift, stride_p, stride_z, stride_dro, stride_dp, stride_dz, q_off, mix_off, scale,
            S, H, D, ROUND, FB, SP,
        )
    elif STORE:
        _store_read_backward(
            b, h - H, ek_ptr, ev_ptr, p_ptr, freq_ptr, pos_ptr, dro_ptr, dp_ptr, stats_ptr,
            sp_ptr, sd_ptr, sq_ptr, sdo_ptr, sdelta_ptr,
            B, T, shift, step, stride_p, stride_dro, stride_dp, sq_off, ro_off, store_scale,
            S, HS, DS, W, WP, ROUND, FB,
        )


@triton.jit
def commit_forward_kernel(
    m_ptr, out_ptr, p_ptr, z_ptr, dt_ptr, alog_ptr, tag_ptr, freq_ptr, pos_ptr, ek_ptr, ev_ptr,
    B, T, step, stride_ps, stride_p, stride_zs, stride_z, k_off, v_off, beta_off, beta_max, sk_off, sv_off,
    S: tl.constexpr, H: tl.constexpr, D: tl.constexpr, HS: tl.constexpr, DS: tl.constexpr,
    STORE: tl.constexpr,
):
    """Every source's bank write and (``STORE``) store entry of one token."""
    s = tl.program_id(0)
    b64 = tl.program_id(1).to(tl.int64)
    h = tl.program_id(2)
    if h < H:
        _commit_forward(
            s, b64, h, m_ptr, out_ptr, p_ptr, z_ptr, dt_ptr, alog_ptr,
            B, stride_ps, stride_p, stride_zs, stride_z, k_off, v_off, beta_off, beta_max, H, D,
        )
    elif STORE:
        _store_append_forward(
            s, b64, h - H, p_ptr, tag_ptr, freq_ptr, pos_ptr, ek_ptr, ev_ptr,
            T, step, stride_ps, stride_p, sk_off, sv_off, S, HS, DS,
        )


@triton.jit
def commit_backward_kernel(
    m_ptr, g_ptr, pend_ptr, pend_q_ptr, p_ptr, z_ptr, dt_ptr, alog_ptr, dp_ptr, dz_ptr, dalog_ptr, ddt_ptr,
    freq_ptr, pos_ptr, dek_ptr, dev_ptr, dtag_ptr,
    B, shift, step, stride_ps, stride_p, stride_zs, stride_z, stride_dps, stride_dp, stride_dzs, stride_dz,
    k_off, v_off, beta_off, beta_max, sk_off, sv_off,
    R: tl.constexpr, H: tl.constexpr, D: tl.constexpr, HS: tl.constexpr, DS: tl.constexpr,
    W: tl.constexpr, FB: tl.constexpr, STORE: tl.constexpr,
):
    """Backward of one token's commits and (``STORE``) store entries; the
    entries' gradients are complete, every later token having added its
    key/value gradients."""
    s = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    if h < H:
        _commit_backward(
            s, b, h, m_ptr, g_ptr, pend_ptr, pend_q_ptr, p_ptr, z_ptr, dt_ptr, alog_ptr,
            dp_ptr, dz_ptr, dalog_ptr, ddt_ptr,
            B, shift, stride_ps, stride_p, stride_zs, stride_z, stride_dps, stride_dp, stride_dzs, stride_dz,
            k_off, v_off, beta_off, beta_max, R, H, D, FB,
        )
    elif STORE:
        _store_append_backward(
            s, b.to(tl.int64), h - H, p_ptr, freq_ptr, pos_ptr, dek_ptr, dev_ptr, dp_ptr, dtag_ptr,
            B, step, stride_ps, stride_p, stride_dps, stride_dp, sk_off, sv_off, R + 1, HS, DS, W,
        )
