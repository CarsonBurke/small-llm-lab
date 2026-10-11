"""Bank-state products of the chunk-parallel core (``parallel.py``) as
custom ops with Triton kernels.

Each product touches one fp32 ``[D, D]`` bank tile per (source, row, head)
with a chunk's worth of vectors, so it is bound by streaming the state;
cuBLAS runs these batched shapes as GEMV-like kernels at a fraction of the
bandwidth. Here one program owns one tile: it loads the tile once, does
the chunk's products with ``tl.dot`` in full fp32 (``ieee``) and, in the
backward, writes the tile's gradient once. A query shared by several
sources is read by each of their programs in place (no broadcast copy);
its gradient is summed over the sources after the kernel.

* ``bank_read(q, M)``: ``q [N, C, D] @ M[s, N]^T`` -> ``[s, N, C, D]``
  (state reads of one query from ``s`` sources; ``M`` value-major ``[s,
  N, D(v), D(k)]``).
* ``bank_end(M, e, u, k)``: ``M * e[:, None, :] + u^T @ k`` -> ``[N, D,
  D]`` (the chunk-end state: decayed start plus the chunk's writes).
* ``wy_write(M, k, v, beta, g)``: one source's chunk of delta-rule writes
  in WY form (``parallel.ChunkBank``): its ``u [N, C, D]``, from the
  cumulative decays, the ``[C, C]`` key products, the unit lower
  triangular system (inverted by repeated squaring of its nilpotent part)
  and the residuals of the start state.
* ``wy_read(q, M, k, g, u)``: reads of the state before each chunk token,
  ``q`` shared by the ``s`` stacked sources: the start state at the
  token's decay plus the chunk's earlier writes.
* ``bank_accumulate(dM, ...)``: the start-state gradient of a chunk's
  reads and writes, added in place. Each read or write touches a tile's
  gradient only through a rank-``C`` product, so instead of every
  backward op writing a dense ``[D, D]`` term per tile (and autograd
  summing them), the ops skip it (``need_dm=False``) and this pass adds
  all of them from the read-output and value gradients.

The WY ops recompute their decays and ``[C, C]`` products in the
backward, whose gradients are derived by hand (the PyTorch references
below hold the same derivation, checked against autograd).

Off CUDA, or for chunk or head sizes ``tl.dot`` cannot take (below 16, or
not powers of two), the ops run the same products in PyTorch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def bank_read_kernel(q_ptr, m_ptr, out_ptr, QN, C: tl.constexpr, D: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, C)
    d = tl.arange(0, D)
    rows = c[:, None] * D + d[None, :]
    q = tl.load(q_ptr + (n % QN) * C * D + rows)
    m = tl.load(m_ptr + n * D * D + d[:, None] * D + d[None, :])
    tl.store(out_ptr + n * C * D + rows, tl.dot(q, tl.trans(m), input_precision="ieee"))


@triton.jit
def bank_read_backward_kernel(q_ptr, m_ptr, do_ptr, dq_ptr, dm_ptr, QN, C: tl.constexpr, D: tl.constexpr,
                              HAS_DM: tl.constexpr):
    """``dq`` per tile (the caller sums the sources)."""
    n = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, C)
    d = tl.arange(0, D)
    rows = c[:, None] * D + d[None, :]
    tile = n * D * D + d[:, None] * D + d[None, :]
    q = tl.load(q_ptr + (n % QN) * C * D + rows)
    m = tl.load(m_ptr + tile)
    do = tl.load(do_ptr + n * C * D + rows)
    tl.store(dq_ptr + n * C * D + rows, tl.dot(do, m, input_precision="ieee"))
    if HAS_DM:
        tl.store(dm_ptr + tile, tl.dot(tl.trans(do), q, input_precision="ieee"))


@triton.jit
def bank_end_kernel(m_ptr, e_ptr, u_ptr, k_ptr, out_ptr, C: tl.constexpr, D: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, C)
    d = tl.arange(0, D)
    vec = n * C * D + c[:, None] * D + d[None, :]
    tile = n * D * D + d[:, None] * D + d[None, :]
    m = tl.load(m_ptr + tile)
    e = tl.load(e_ptr + n * D + d)
    u = tl.load(u_ptr + vec)
    k = tl.load(k_ptr + vec)
    out = m * e[None, :] + tl.dot(tl.trans(u), k, input_precision="ieee")
    tl.store(out_ptr + tile, out)


@triton.jit
def bank_end_backward_kernel(
    m_ptr, e_ptr, u_ptr, k_ptr, dout_ptr, dm_ptr, de_ptr, du_ptr, dk_ptr, C: tl.constexpr, D: tl.constexpr
):
    n = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, C)
    d = tl.arange(0, D)
    vec = n * C * D + c[:, None] * D + d[None, :]
    tile = n * D * D + d[:, None] * D + d[None, :]
    m = tl.load(m_ptr + tile)
    e = tl.load(e_ptr + n * D + d)
    u = tl.load(u_ptr + vec)
    k = tl.load(k_ptr + vec)
    dout = tl.load(dout_ptr + tile)
    tl.store(dm_ptr + tile, dout * e[None, :])
    tl.store(de_ptr + n * D + d, tl.sum(dout * m, axis=0))
    tl.store(du_ptr + vec, tl.dot(k, tl.trans(dout), input_precision="ieee"))
    tl.store(dk_ptr + vec, tl.dot(u, dout, input_precision="ieee"))


def _kernel_shapes(C: int, D: int) -> bool:
    return all(x >= 16 and x & (x - 1) == 0 for x in (C, D))


def _use_kernels(vectors: Tensor, tile: Tensor) -> bool:
    """Kernels for CUDA tensors of kernel shapes; ``vectors [*, C, D]`` and
    ``tile [*, D, D]`` must then share their leading dims (no broadcast)."""
    C, D = vectors.shape[-2:]
    if not (vectors.is_cuda and _kernel_shapes(C, D)):
        return False
    if vectors.shape[:-2] != tile.shape[:-2]:
        raise ValueError(f"bank products take matching leading dims, got {tuple(vectors.shape)} and {tuple(tile.shape)}")
    return True


def _flat(x: Tensor, tail: int) -> Tensor:
    return x.reshape(-1, *x.shape[-tail:]).contiguous()


def _no_state_grad(m: Tensor) -> Tensor:
    """The placeholder a backward op returns for a state gradient it was not
    asked for: the caller takes the state's gradient from its own deferred
    sum (``bank_accumulate``), so no dense per-op ``[*, D, D]`` term is
    written and summed."""
    return m.new_empty(0)


# ---------------------------------------------------------------------------
# bank_read
# ---------------------------------------------------------------------------


def _shared_query(q: Tensor, tile: Tensor) -> bool:
    """Kernels for CUDA tensors of kernel shapes; ``q [*, C, D]`` must be
    shared by the sources of ``tile [s, *, D, D]``."""
    if q.shape[:-2] != tile.shape[1:-2]:
        raise ValueError(f"a query [*, C, D] shared by the sources [s, *] of the state, got {tuple(q.shape)} and {tuple(tile.shape)}")
    return q.is_cuda and _kernel_shapes(*q.shape[-2:])


@torch.library.custom_op("cwm::bank_read", mutates_args=())
def bank_read(q: Tensor, m: Tensor) -> Tensor:
    """``q [*, C, D] @ m[s, *, D, D]^T`` -> ``[s, *, C, D]``: one query
    read from every source's state, in fp32."""
    if not _shared_query(q, m):
        return (q @ m.transpose(-1, -2)).contiguous()
    C, D = q.shape[-2:]
    qf, mf = _flat(q, 2), _flat(m, 2)
    out = torch.empty(*m.shape[:-2], C, D, device=q.device, dtype=q.dtype)
    bank_read_kernel[(mf.size(0),)](qf, mf, out, qf.size(0), C=C, D=D, num_warps=4)
    return out


@bank_read.register_fake
def _(q: Tensor, m: Tensor) -> Tensor:
    return q.new_empty(*m.shape[:-2], *q.shape[-2:])


@torch.library.custom_op("cwm::bank_read_backward", mutates_args=())
def bank_read_backward(q: Tensor, m: Tensor, d_out: Tensor, need_dm: bool) -> tuple[Tensor, Tensor]:
    """``(dq, dm)``; ``dm`` is empty unless ``need_dm`` (``_state_grad``)."""
    if not _shared_query(q, m):
        dm = (d_out.transpose(-1, -2) @ q).contiguous() if need_dm else _no_state_grad(m)
        return (d_out @ m).sum(0).contiguous(), dm
    C, D = q.shape[-2:]
    qf, mf, df = _flat(q, 2), _flat(m, 2), _flat(d_out, 2)
    dq = torch.empty_like(df)
    dm = torch.empty_like(mf) if need_dm else _no_state_grad(m)
    bank_read_backward_kernel[(mf.size(0),)](qf, mf, df, dq, dm if need_dm else dq, qf.size(0), C=C, D=D,
                                             HAS_DM=need_dm, num_warps=4)
    return dq.view(d_out.shape).sum(0), dm.view(m.shape) if need_dm else dm


@bank_read_backward.register_fake
def _(q: Tensor, m: Tensor, d_out: Tensor, need_dm: bool) -> tuple[Tensor, Tensor]:
    return (torch.empty_like(q, memory_format=torch.contiguous_format),
            torch.empty_like(m, memory_format=torch.contiguous_format) if need_dm else _no_state_grad(m))


def _bank_read_setup(ctx, inputs, output) -> None:
    ctx.save_for_backward(*inputs)


def _bank_read_grad(ctx, d_out: Tensor):
    q, m = ctx.saved_tensors
    need_dm = ctx.needs_input_grad[1]
    dq, dm = bank_read_backward(q, m, d_out, need_dm)
    return dq, dm if need_dm else None


bank_read.register_autograd(_bank_read_grad, setup_context=_bank_read_setup)


# ---------------------------------------------------------------------------
# bank_end
# ---------------------------------------------------------------------------


@torch.library.custom_op("cwm::bank_end", mutates_args=())
def bank_end(m: Tensor, e: Tensor, u: Tensor, k: Tensor) -> Tensor:
    """``m [*, D, D] * e[*, None, :] + u[*, C, D]^T @ k[*, C, D]`` in fp32."""
    if not _use_kernels(u, m):
        return (m * e[..., None, :] + u.transpose(-1, -2) @ k).contiguous()
    C, D = u.shape[-2:]
    mf, ef, uf, kf = _flat(m, 2), _flat(e, 1), _flat(u, 2), _flat(k, 2)
    out = torch.empty_like(mf)
    bank_end_kernel[(mf.size(0),)](mf, ef, uf, kf, out, C=C, D=D, num_warps=4)
    return out.view(m.shape)


@bank_end.register_fake
def _(m: Tensor, e: Tensor, u: Tensor, k: Tensor) -> Tensor:
    return torch.empty_like(m, memory_format=torch.contiguous_format)


@torch.library.custom_op("cwm::bank_end_backward", mutates_args=())
def bank_end_backward(m: Tensor, e: Tensor, u: Tensor, k: Tensor, d_out: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not _use_kernels(u, m):
        grads = d_out * e[..., None, :], (d_out * m).sum(-2), k @ d_out.transpose(-1, -2), u @ d_out
        return tuple(g.contiguous() for g in grads)
    C, D = u.shape[-2:]
    mf, ef, uf, kf, df = _flat(m, 2), _flat(e, 1), _flat(u, 2), _flat(k, 2), _flat(d_out, 2)
    dm, de, du, dk = (torch.empty_like(t) for t in (mf, ef, uf, kf))
    bank_end_backward_kernel[(mf.size(0),)](mf, ef, uf, kf, df, dm, de, du, dk, C=C, D=D, num_warps=4)
    return dm.view(m.shape), de.view(e.shape), du.view(u.shape), dk.view(k.shape)


@bank_end_backward.register_fake
def _(m: Tensor, e: Tensor, u: Tensor, k: Tensor, d_out: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return tuple(torch.empty_like(t, memory_format=torch.contiguous_format) for t in (m, e, u, k))


def _bank_end_setup(ctx, inputs, output) -> None:
    ctx.save_for_backward(*inputs)


def _bank_end_grad(ctx, d_out: Tensor):
    return bank_end_backward(*ctx.saved_tensors, d_out)


bank_end.register_autograd(_bank_end_grad, setup_context=_bank_end_setup)


# ---------------------------------------------------------------------------
# WY chunk banks: shared in-kernel pieces
# ---------------------------------------------------------------------------


@triton.jit
def _decays(g, rows, MID: tl.constexpr):
    """``G = cumsum(g)`` over the chunk and its reference row ``G[MID]``."""
    G = tl.cumsum(g, axis=0)
    ref = tl.sum(tl.where(rows[:, None] == MID, G, 0.0), axis=0)
    return G, ref


@triton.jit
def _unit_lower_inverse(A, rows, LOG2C: tl.constexpr):
    """``(I + A)^-1`` of a strictly lower ``A [C, C]``: with ``N = -A``
    nilpotent, ``sum_m N^m = (I + N)(I + N^2)(I + N^4)...``."""
    eye = tl.where(rows[:, None] == rows[None, :], 1.0, 0.0)
    N = -A
    X = eye + N
    P = N
    for _ in tl.static_range(LOG2C - 1):
        P = tl.dot(P, P, input_precision="ieee")
        X = X + tl.dot(X, P, input_precision="ieee")
    return X


# ---------------------------------------------------------------------------
# wy_write kernels: U of one source's chunk
# ---------------------------------------------------------------------------


@triton.jit
def wy_write_kernel(m_ptr, k_ptr, v_ptr, b_ptr, g_ptr, u_ptr,
                    C: tl.constexpr, D: tl.constexpr, MID: tl.constexpr, LOG2C: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, C)
    d = tl.arange(0, D)
    vec = n * C * D + c[:, None] * D + d[None, :]
    k = tl.load(k_ptr + vec)
    v = tl.load(v_ptr + vec)
    g = tl.load(g_ptr + vec)
    beta = tl.load(b_ptr + n * C + c)
    m = tl.load(m_ptr + n * D * D + d[:, None] * D + d[None, :])
    G, ref = _decays(g, c, MID)
    kin = k * tl.exp(ref[None, :] - G)
    kout = k * tl.exp(G - ref[None, :])
    lower = c[:, None] > c[None, :]
    A = tl.where(lower, tl.dot(kout, tl.trans(kin), input_precision="ieee"), 0.0) * beta[:, None]
    wm = tl.dot(k * tl.exp(G), tl.trans(m), input_precision="ieee")
    rhs = beta[:, None] * (v - wm)
    U = tl.dot(_unit_lower_inverse(A, c, LOG2C), rhs, input_precision="ieee")
    tl.store(u_ptr + vec, U)


@triton.jit
def wy_write_backward_kernel(m_ptr, k_ptr, v_ptr, b_ptr, g_ptr, du_ptr,
                             dm_ptr, dk_ptr, dv_ptr, db_ptr, dg_ptr,
                             C: tl.constexpr, D: tl.constexpr, MID: tl.constexpr, LOG2C: tl.constexpr,
                             HAS_DM: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, C)
    d = tl.arange(0, D)
    vec = n * C * D + c[:, None] * D + d[None, :]
    tile = n * D * D + d[:, None] * D + d[None, :]
    k = tl.load(k_ptr + vec)
    v = tl.load(v_ptr + vec)
    g = tl.load(g_ptr + vec)
    beta = tl.load(b_ptr + n * C + c)
    m = tl.load(m_ptr + tile)
    dU = tl.load(du_ptr + vec)
    G, ref = _decays(g, c, MID)
    e_in = tl.exp(ref[None, :] - G)
    e_out = tl.exp(G - ref[None, :])
    e_w = tl.exp(G)
    kin, kout, w = k * e_in, k * e_out, k * e_w
    lower = c[:, None] > c[None, :]
    S = tl.where(lower, tl.dot(kout, tl.trans(kin), input_precision="ieee"), 0.0)
    A = S * beta[:, None]
    resid = v - tl.dot(w, tl.trans(m), input_precision="ieee")
    rhs = beta[:, None] * resid
    Linv = _unit_lower_inverse(A, c, LOG2C)
    U = tl.dot(Linv, rhs, input_precision="ieee")
    # U = (I + A)^-1 rhs.
    drhs = tl.dot(tl.trans(Linv), dU, input_precision="ieee")
    dA = -tl.where(lower, tl.dot(drhs, tl.trans(U), input_precision="ieee"), 0.0)
    dbeta = tl.sum(drhs * resid, axis=1) + tl.sum(dA * S, axis=1)
    dwm = -beta[:, None] * drhs
    dw = tl.dot(dwm, m, input_precision="ieee")
    if HAS_DM:
        tl.store(dm_ptr + tile, tl.dot(tl.trans(dwm), w, input_precision="ieee"))
    dS = dA * beta[:, None]
    dkout = tl.dot(dS, kin, input_precision="ieee")
    dkin = tl.dot(tl.trans(dS), kout, input_precision="ieee")
    dk = dkout * e_out + dkin * e_in + dw * e_w
    dG = dkout * kout - dkin * kin + dw * w
    dref = tl.sum(dkin * kin - dkout * kout, axis=0)
    dG += tl.where(c[:, None] == MID, dref[None, :], 0.0)
    tl.store(dk_ptr + vec, dk)
    tl.store(dv_ptr + vec, beta[:, None] * drhs)
    tl.store(db_ptr + n * C + c, dbeta)
    tl.store(dg_ptr + vec, tl.cumsum(dG, axis=0, reverse=True))


# ---------------------------------------------------------------------------
# wy_read kernels: the state before each token, for a query shared by the
# sources
# ---------------------------------------------------------------------------


@triton.jit
def wy_read_kernel(q_ptr, m_ptr, k_ptr, g_ptr, u_ptr, out_ptr, QN,
                   C: tl.constexpr, D: tl.constexpr, MID: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, C)
    d = tl.arange(0, D)
    rows = c[:, None] * D + d[None, :]
    vec = n * C * D + rows
    q = tl.load(q_ptr + (n % QN) * C * D + rows)
    k = tl.load(k_ptr + vec)
    g = tl.load(g_ptr + vec)
    U = tl.load(u_ptr + vec)
    m = tl.load(m_ptr + n * D * D + d[:, None] * D + d[None, :])
    G, ref = _decays(g, c, MID)
    before = G - g
    kin = k * tl.exp(ref[None, :] - G)
    qb = q * tl.exp(before)
    qr = q * tl.exp(before - ref[None, :])
    P = tl.where(c[:, None] > c[None, :], tl.dot(qr, tl.trans(kin), input_precision="ieee"), 0.0)
    out = tl.dot(qb, tl.trans(m), input_precision="ieee") + tl.dot(P, U, input_precision="ieee")
    tl.store(out_ptr + vec, out)


@triton.jit
def wy_read_backward_kernel(q_ptr, m_ptr, k_ptr, g_ptr, u_ptr, do_ptr,
                            dq_ptr, dm_ptr, dk_ptr, dg_ptr, du_ptr, QN,
                            C: tl.constexpr, D: tl.constexpr, MID: tl.constexpr, HAS_DM: tl.constexpr):
    """``dq`` per tile (the caller sums the sources)."""
    n = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, C)
    d = tl.arange(0, D)
    rows = c[:, None] * D + d[None, :]
    vec = n * C * D + rows
    tile = n * D * D + d[:, None] * D + d[None, :]
    q = tl.load(q_ptr + (n % QN) * C * D + rows)
    k = tl.load(k_ptr + vec)
    g = tl.load(g_ptr + vec)
    U = tl.load(u_ptr + vec)
    m = tl.load(m_ptr + tile)
    dR = tl.load(do_ptr + vec)
    G, ref = _decays(g, c, MID)
    before = G - g
    e_in = tl.exp(ref[None, :] - G)
    kin = k * e_in
    e_b = tl.exp(before)
    e_r = tl.exp(before - ref[None, :])
    qb, qr = q * e_b, q * e_r
    lower = c[:, None] > c[None, :]
    P = tl.where(lower, tl.dot(qr, tl.trans(kin), input_precision="ieee"), 0.0)
    tl.store(du_ptr + vec, tl.dot(tl.trans(P), dR, input_precision="ieee"))
    dP = tl.where(lower, tl.dot(dR, tl.trans(U), input_precision="ieee"), 0.0)
    dqb = tl.dot(dR, m, input_precision="ieee")
    if HAS_DM:
        tl.store(dm_ptr + tile, tl.dot(tl.trans(dR), qb, input_precision="ieee"))
    dqr = tl.dot(dP, kin, input_precision="ieee")
    dkin = tl.dot(tl.trans(dP), qr, input_precision="ieee")
    tl.store(dq_ptr + vec, dqb * e_b + dqr * e_r)
    dbefore = dqb * qb + dqr * qr
    dref = tl.sum(dkin * kin - dqr * qr, axis=0)
    dG = dbefore - dkin * kin + tl.where(c[:, None] == MID, dref[None, :], 0.0)
    tl.store(dk_ptr + vec, dkin * e_in)
    tl.store(dg_ptr + vec, tl.cumsum(dG, axis=0, reverse=True) - dbefore)


# ---------------------------------------------------------------------------
# WY PyTorch references: the ops off CUDA and at other shapes (forward and
# the kernels' hand-derived backward), and the tests' ground truth
# ---------------------------------------------------------------------------


def _strictly_lower(x: Tensor) -> Tensor:
    """Entries below the diagonal of ``x [*, C, C]``, selected (the factored
    decay ratios above it may overflow)."""
    C = x.size(-1)
    mask = torch.ones(C, C, dtype=torch.bool, device=x.device).tril(-1)
    return torch.where(mask, x, 0.0)


def _at_ref(x: Tensor, C: int) -> Tensor:
    """``x [*, 1, D]`` placed on the reference row of a ``[*, C, D]`` gradient."""
    out = torch.zeros(*x.shape[:-2], C, x.size(-1), dtype=x.dtype, device=x.device)
    out[..., (C - 1) // 2, :] = x[..., 0, :]
    return out


def _ref_decays(g: Tensor) -> tuple[Tensor, Tensor]:
    G = g.cumsum(-2)
    return G, G[..., (g.size(-2) - 1) // 2, None, :]


def wy_write_reference(m: Tensor, k: Tensor, v: Tensor, beta: Tensor, g: Tensor) -> Tensor:
    G, ref = _ref_decays(g)
    kin = k * (ref - G).exp()
    A = _strictly_lower((k * (G - ref).exp()) @ kin.transpose(-1, -2)) * beta[..., None]
    rhs = beta[..., None] * (v - (k * G.exp()) @ m.transpose(-1, -2))
    return torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True)


def wy_read_reference(q: Tensor, m: Tensor, k: Tensor, g: Tensor, u: Tensor) -> Tensor:
    G, ref = _ref_decays(g)
    before = G - g
    kin = k * (ref - G).exp()
    P = _strictly_lower((q * (before - ref).exp()) @ kin.transpose(-1, -2))
    return (q * before.exp()) @ m.transpose(-1, -2) + P @ u


def wy_write_backward_reference(m, k, v, beta, g, dU):
    C = k.size(-2)
    G, ref = _ref_decays(g)
    e_in, e_out, e_w = (ref - G).exp(), (G - ref).exp(), G.exp()
    kin, kout, w = k * e_in, k * e_out, k * e_w
    S = _strictly_lower(kout @ kin.transpose(-1, -2))
    A = S * beta[..., None]
    resid = v - w @ m.transpose(-1, -2)
    U = torch.linalg.solve_triangular(A, beta[..., None] * resid, upper=False, unitriangular=True)
    drhs = torch.linalg.solve_triangular(A.transpose(-1, -2), dU, upper=True, unitriangular=True)
    dA = -_strictly_lower(drhs @ U.transpose(-1, -2))
    dbeta = (drhs * resid).sum(-1) + (dA * S).sum(-1)
    dwm = -beta[..., None] * drhs
    dw = dwm @ m
    dm = dwm.transpose(-1, -2) @ w
    dS = dA * beta[..., None]
    dkout, dkin = dS @ kin, dS.transpose(-1, -2) @ kout
    dk = dkout * e_out + dkin * e_in + dw * e_w
    dref = (dkin * kin - dkout * kout).sum(-2, keepdim=True)
    dG = dkout * kout - dkin * kin + dw * w + _at_ref(dref, C)
    dg = dG.flip(-2).cumsum(-2).flip(-2)
    return dm, dk, beta[..., None] * drhs, dbeta, dg


def wy_read_backward_reference(q, m, k, g, u, dR):
    C = k.size(-2)
    G, ref = _ref_decays(g)
    before = G - g
    e_in, e_b, e_r = (ref - G).exp(), before.exp(), (before - ref).exp()
    kin, qb, qr = k * e_in, q * e_b, q * e_r
    P = _strictly_lower(qr @ kin.transpose(-1, -2))
    du = P.transpose(-1, -2) @ dR
    dP = _strictly_lower(dR @ u.transpose(-1, -2))
    dqb = dR @ m
    dm = dR.transpose(-1, -2) @ qb
    dqr, dkin = dP @ kin, dP.transpose(-1, -2) @ qr
    dq = (dqb * e_b + dqr * e_r).sum(0)
    dbefore = dqb * qb + dqr * qr
    dref = (dkin * kin - dqr * qr).sum(-2, keepdim=True)
    dG = dbefore - dkin * kin + _at_ref(dref, C)
    dg = dG.flip(-2).cumsum(-2).flip(-2) - dbefore
    return dq, dm, dkin * e_in, dg, du


# ---------------------------------------------------------------------------
# wy_write, wy_read
# ---------------------------------------------------------------------------


def _consts(C: int) -> dict:
    return dict(MID=(C - 1) // 2)


def _check_write(m: Tensor, k: Tensor, v: Tensor, beta: Tensor, g: Tensor) -> bool:
    lead = k.shape[:-2]
    if not (m.shape[:-2] == v.shape[:-2] == g.shape[:-2] == beta.shape[:-1] == lead):
        raise ValueError("wy_write takes matching leading dims")
    return k.is_cuda and _kernel_shapes(*k.shape[-2:])


@torch.library.custom_op("cwm::wy_write", mutates_args=())
def wy_write(m: Tensor, k: Tensor, v: Tensor, beta: Tensor, g: Tensor) -> Tensor:
    """``U`` of one chunk's delta-rule writes against the chunk-start state
    ``m [*, Dv, Dk]``: ``k``, ``g [*, C, Dk]`` (unit keys, log decays),
    ``v [*, C, Dv]``, ``beta [*, C]``; fp32."""
    if not _check_write(m, k, v, beta, g):
        return wy_write_reference(m, k, v, beta, g).contiguous()
    C, D = k.shape[-2:]
    mf, kf, vf, bf, gf = _flat(m, 2), _flat(k, 2), _flat(v, 2), _flat(beta, 1), _flat(g, 2)
    u = torch.empty_like(kf)
    wy_write_kernel[(kf.size(0),)](mf, kf, vf, bf, gf, u, C=C, D=D, LOG2C=C.bit_length() - 1, num_warps=4, **_consts(C))
    return u.view(k.shape)


@wy_write.register_fake
def _(m, k, v, beta, g):
    return torch.empty_like(k, memory_format=torch.contiguous_format)


@torch.library.custom_op("cwm::wy_write_backward", mutates_args=())
def wy_write_backward(m: Tensor, k: Tensor, v: Tensor, beta: Tensor, g: Tensor, du: Tensor,
                      need_dm: bool) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """``(dm, dk, dv, dbeta, dg)``; ``dm`` is empty unless ``need_dm``."""
    if not _check_write(m, k, v, beta, g):
        dm, *grads = wy_write_backward_reference(m, k, v, beta, g, du)
        return tuple(x.contiguous() for x in (dm if need_dm else _no_state_grad(m), *grads))
    C, D = k.shape[-2:]
    ins = [_flat(m, 2), _flat(k, 2), _flat(v, 2), _flat(beta, 1), _flat(g, 2)]
    outs = [torch.empty_like(t) for t in ins[1:]]
    dm = torch.empty_like(ins[0]) if need_dm else _no_state_grad(m)
    wy_write_backward_kernel[(ins[1].size(0),)](*ins, _flat(du, 2), dm if need_dm else outs[0], *outs, C=C, D=D,
                                                LOG2C=C.bit_length() - 1, HAS_DM=need_dm, num_warps=4, **_consts(C))
    return (dm.view(m.shape) if need_dm else dm, *(o.view(t.shape) for o, t in zip(outs, (k, v, beta, g))))


@wy_write_backward.register_fake
def _(m, k, v, beta, g, du, need_dm):
    dm = torch.empty_like(m, memory_format=torch.contiguous_format) if need_dm else _no_state_grad(m)
    return (dm, *(torch.empty_like(t, memory_format=torch.contiguous_format) for t in (k, v, beta, g)))


def _wy_write_setup(ctx, inputs, output) -> None:
    ctx.save_for_backward(*inputs)


def _wy_write_grad(ctx, du: Tensor):
    need_dm = ctx.needs_input_grad[0]
    dm, *grads = wy_write_backward(*ctx.saved_tensors, du, need_dm)
    return dm if need_dm else None, *grads


wy_write.register_autograd(_wy_write_grad, setup_context=_wy_write_setup)


def _check_read(q: Tensor, m: Tensor, k: Tensor, g: Tensor, u: Tensor) -> bool:
    if not (m.shape[:-2] == k.shape[:-2] == g.shape[:-2] == u.shape[:-2] and q.shape == k.shape[1:]):
        raise ValueError("wy_read takes q [*, C, D] shared over the sources of m, k, g, u [s, *, ...]")
    return k.is_cuda and _kernel_shapes(*k.shape[-2:])


@torch.library.custom_op("cwm::wy_read", mutates_args=())
def wy_read(q: Tensor, m: Tensor, k: Tensor, g: Tensor, u: Tensor) -> Tensor:
    """Reads ``M_{j-1} q_j`` of the state before each chunk token, per source:
    ``q [*, C, Dk]`` shared by the ``s`` sources of ``m [s, *, Dv, Dk]``,
    ``k``, ``g [s, *, C, Dk]`` and ``u [s, *, C, Dv]`` (``wy_write``) ->
    ``[s, *, C, Dv]``; fp32."""
    if not _check_read(q, m, k, g, u):
        return wy_read_reference(q, m, k, g, u).contiguous()
    C, D = k.shape[-2:]
    qf, mf, kf, gf, uf = _flat(q, 2), _flat(m, 2), _flat(k, 2), _flat(g, 2), _flat(u, 2)
    out = torch.empty_like(kf)
    wy_read_kernel[(kf.size(0),)](qf, mf, kf, gf, uf, out, qf.size(0), C=C, D=D, num_warps=4, **_consts(C))
    return out.view(k.shape)


@wy_read.register_fake
def _(q, m, k, g, u):
    return torch.empty_like(k, memory_format=torch.contiguous_format)


@torch.library.custom_op("cwm::wy_read_backward", mutates_args=())
def wy_read_backward(q: Tensor, m: Tensor, k: Tensor, g: Tensor, u: Tensor, d_out: Tensor,
                     need_dm: bool) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """``(dq, dm, dk, dg, du)``; ``dm`` is empty unless ``need_dm``."""
    if not _check_read(q, m, k, g, u):
        dq, dm, *grads = wy_read_backward_reference(q, m, k, g, u, d_out)
        return tuple(x.contiguous() for x in (dq, dm if need_dm else _no_state_grad(m), *grads))
    C, D = k.shape[-2:]
    ins = [_flat(q, 2), _flat(m, 2), _flat(k, 2), _flat(g, 2), _flat(u, 2)]
    dq = torch.empty_like(ins[2])
    dm = torch.empty_like(ins[1]) if need_dm else _no_state_grad(m)
    outs = [torch.empty_like(t) for t in ins[2:]]
    wy_read_backward_kernel[(ins[2].size(0),)](*ins, _flat(d_out, 2), dq, dm if need_dm else dq, *outs, ins[0].size(0),
                                               C=C, D=D, HAS_DM=need_dm, num_warps=4, **_consts(C))
    return (dq.view(k.shape).sum(0), dm.view(m.shape) if need_dm else dm,
            *(o.view(t.shape) for o, t in zip(outs, (k, g, u))))


@wy_read_backward.register_fake
def _(q, m, k, g, u, d_out, need_dm):
    dm = torch.empty_like(m, memory_format=torch.contiguous_format) if need_dm else _no_state_grad(m)
    q, *rest = (torch.empty_like(t, memory_format=torch.contiguous_format) for t in (q, k, g, u))
    return (q, dm, *rest)


def _wy_read_setup(ctx, inputs, output) -> None:
    ctx.save_for_backward(*inputs)


def _wy_read_grad(ctx, d_out: Tensor):
    need_dm = ctx.needs_input_grad[1]
    dq, dm, *grads = wy_read_backward(*ctx.saved_tensors, d_out, need_dm)
    return dq, dm if need_dm else None, *grads


wy_read.register_autograd(_wy_read_grad, setup_context=_wy_read_setup)


# ---------------------------------------------------------------------------
# bank_accumulate: a chunk's state gradient from its reads and writes,
# deferred
# ---------------------------------------------------------------------------


@triton.jit
def bank_accumulate_kernel(dm_ptr, q_ptr, qf_ptr, dr_ptr, k_ptr, g_ptr, dv_ptr, N, S,
                           R: tl.constexpr, C: tl.constexpr, D: tl.constexpr, EXACT: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    s = pid // N
    n = pid % N
    c = tl.arange(0, C)
    d = tl.arange(0, D)
    rows = c[:, None] * D + d[None, :]
    vec = pid * C * D + rows
    tile = pid * D * D + d[:, None] * D + d[None, :]
    g = tl.load(g_ptr + vec)
    G = tl.cumsum(g, axis=0)
    # The write: its residuals read the start state at the token's decay.
    acc = -tl.dot(tl.trans(tl.load(dv_ptr + vec)), tl.load(k_ptr + vec) * tl.exp(G), input_precision="ieee")
    decay = tl.exp(G - g)
    for i in tl.static_range(R):
        legal = s < i + 1
        q = tl.load(tl.where(legal, q_ptr, qf_ptr) + (i * N + n) * C * D + rows)
        if EXACT:
            q = q * tl.where(legal, decay, 1.0)
        dr = tl.load(dr_ptr + (i * S * N + pid) * C * D + rows)
        acc += tl.dot(tl.trans(dr), q, input_precision="ieee")
    tl.store(dm_ptr + tile, tl.load(dm_ptr + tile) + acc)


def bank_accumulate_reference(q: Tensor, q_fb: Tensor, d_reads: Tensor, k: Tensor, g: Tensor, dv: Tensor,
                              exact: bool) -> Tensor:
    """``bank_accumulate``'s increment of ``dm``."""
    G = g.cumsum(-2)
    out = -(dv.transpose(-1, -2) @ (k * G.exp()))
    decay = (G - g).exp()
    for i in range(q.size(0)):
        r = i + 1
        legal = q[i] * decay[:r] if exact else q[i]
        out[:r] += d_reads[i, :r].transpose(-1, -2) @ legal
        out[r:] += d_reads[i, r:].transpose(-1, -2) @ q_fb[i]
    return out


@torch.library.custom_op("cwm::bank_accumulate", mutates_args=("dm",))
def bank_accumulate(dm: Tensor, q: Tensor, q_fb: Tensor, d_reads: Tensor, k: Tensor, g: Tensor, dv: Tensor,
                    exact: bool) -> None:
    """Adds to ``dm [S, *, Dv, Dk]`` the start-state gradient of a chunk's
    reads and writes, each a rank-``C`` term per tile, in one pass over the
    state. Round ``i`` (``i + 1`` legal sources) read every source ``s``
    with output gradient ``d_reads[i, s]`` (``[R, S, *, C, Dv]``): legal
    sources with ``q[i]``, at the token's decay ``exp(G - g)`` when
    ``exact`` (``wy_read``), the others with ``q_fb[i]`` (``[R, *, C,
    Dk]``). Source ``s``'s writes (``k``, ``g [S, *, C, Dk]``) took the
    value gradient ``dv [S, *, C, Dv]`` (``wy_write``'s residuals)."""
    R, S = d_reads.shape[:2]
    C, D = k.shape[-2:]
    if not (k.is_cuda and _kernel_shapes(C, D)):
        dm.add_(bank_accumulate_reference(q, q_fb, d_reads, k, g, dv, exact))
        return
    if not dm.is_contiguous():
        raise ValueError("bank_accumulate adds into a contiguous dm")
    flat = [_flat(t, 2) for t in (q, q_fb, d_reads, k, g, dv)]
    N = flat[3].size(0) // S
    bank_accumulate_kernel[(S * N,)](dm, *flat, N, S, R=R, C=C, D=D, EXACT=exact, num_warps=4)


# ---------------------------------------------------------------------------
# Store reads: a round's queries over the kept window of every source and
# the chunk's own entries. Legal sources (their window slots and the
# chunk's entries) share one softmax with a sink; feedback sources (window
# only, of a shifted row under the placebo) another. Keys and values ``[B,
# N, S, H, Dh]`` with any strides (unit last); a part's flattened (slot,
# source) pairs in blocks of ``BL``.
# ---------------------------------------------------------------------------

STORE_BLOCK = 64


@triton.jit
def _store_block(q, k_ptr, sb, sn, ss, sh, row, h, start, c, base, s0, s_count, n_count,
                 IS_WINDOW: tl.constexpr, W: tl.constexpr, BL: tl.constexpr, DH: tl.constexpr, scale):
    """Masked scores ``[C, BL]`` of the chunk's queries against one block
    of a part's keys (sources ``s0 ..`` of row ``row``), the mask, the
    block's slots, sources and validity, and its keys."""
    l = base + tl.arange(0, BL)
    d = tl.arange(0, DH)
    n = l // s_count
    src = s0 + l % s_count
    valid = l < n_count * s_count
    k = tl.load(k_ptr + row * sb + n.to(tl.int64)[:, None] * sn + src[:, None] * ss + h * sh + d[None, :],
                mask=valid[:, None], other=0.0).to(tl.float32)
    scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
    if IS_WINDOW:
        # Window slot n holds token start - W + n.
        seen = valid[None, :] & (n[None, :] >= W - start) & (n[None, :] >= c[:, None])
    else:
        seen = valid[None, :] & (n[None, :] < c[:, None]) & (n[None, :] >= c[:, None] - W)
    return tl.where(seen, scores, float("-inf")), seen, n, src, valid, k


@triton.jit
def _load_block(ptr, sb, sn, ss, sh, row, h, n, src, valid, DH: tl.constexpr):
    d = tl.arange(0, DH)
    return tl.load(ptr + row * sb + n.to(tl.int64)[:, None] * sn + src[:, None] * ss + h * sh + d[None, :],
                   mask=valid[:, None], other=0.0).to(tl.float32)


@triton.jit
def _store_part(q, m, total, acc, k_ptr, k_sb, k_sn, k_ss, k_sh, v_ptr, v_sb, v_sn, v_ss, v_sh,
                row, h, start, c, s0, s_count, n_count,
                IS_WINDOW: tl.constexpr, W: tl.constexpr, BL: tl.constexpr, DH: tl.constexpr, scale):
    """One part's entries into the online softmax ``(m, total, acc)``."""
    for base in range(0, n_count * s_count, BL):
        s, seen, n, src, valid, _ = _store_block(q, k_ptr, k_sb, k_sn, k_ss, k_sh, row, h, start, c, base,
                                                 s0, s_count, n_count, IS_WINDOW, W, BL, DH, scale)
        m_new = tl.maximum(m, tl.max(s, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.where(seen, tl.exp(s - m_new[:, None]), 0.0)
        v = _load_block(v_ptr, v_sb, v_sn, v_ss, v_sh, row, h, n, src, valid, DH)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        total = total * alpha + tl.sum(p, axis=1)
        m = m_new
    return m, total, acc


@triton.jit
def store_attend_kernel(q_ptr, out_ptr, lse_ptr, fb_out_ptr, fb_lse_ptr, start_ptr,
                        wk_ptr, wk_sb, wk_sn, wk_ss, wk_sh, wv_ptr, wv_sb, wv_sn, wv_ss, wv_sh,
                        ck_ptr, ck_sb, ck_sn, ck_ss, ck_sh, cv_ptr, cv_sb, cv_sn, cv_ss, cv_sh, c_s,
                        B, H, S, legal, shift, scale, FEEDBACK: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
                        DH: tl.constexpr, BL: tl.constexpr):
    pid = tl.program_id(0)
    b = (pid // H).to(tl.int64)
    h = pid % H
    c = tl.arange(0, C)
    d = tl.arange(0, DH)
    start = tl.load(start_ptr)
    rows = ((b * C + c) * H + h) * DH
    q = tl.load(q_ptr + rows[:, None] + d[None, :])
    # Each read starts from the sink (logit 0).
    m = tl.zeros([C], dtype=tl.float32)
    total = tl.full([C], 1.0, dtype=tl.float32)
    acc = tl.zeros([C, DH], dtype=tl.float32)
    m, total, acc = _store_part(q, m, total, acc, wk_ptr, wk_sb, wk_sn, wk_ss, wk_sh, wv_ptr, wv_sb, wv_sn, wv_ss, wv_sh,
                                b, h, start, c, 0, legal, W, True, W, BL, DH, scale)
    m, total, acc = _store_part(q, m, total, acc, ck_ptr, ck_sb, ck_sn, ck_ss, ck_sh, cv_ptr, cv_sb, cv_sn, cv_ss, cv_sh,
                                b, h, start, c, 0, c_s, C, False, W, BL, DH, scale)
    tl.store(out_ptr + rows[:, None] + d[None, :], acc / total[:, None])
    tl.store(lse_ptr + pid * C + c, m + tl.log(total))
    if FEEDBACK:
        row = (b + B - shift) % B
        m = tl.zeros([C], dtype=tl.float32)
        total = tl.full([C], 1.0, dtype=tl.float32)
        acc = tl.zeros([C, DH], dtype=tl.float32)
        m, total, acc = _store_part(q, m, total, acc, wk_ptr, wk_sb, wk_sn, wk_ss, wk_sh, wv_ptr, wv_sb, wv_sn, wv_ss, wv_sh,
                                    row, h, start, c, legal, S - legal, W, True, W, BL, DH, scale)
        tl.store(fb_out_ptr + rows[:, None] + d[None, :], acc / total[:, None])
        tl.store(fb_lse_ptr + pid * C + c, m + tl.log(total))


@triton.jit
def _store_part_backward(q, do, delta, lse, dq, k_ptr, k_sb, k_sn, k_ss, k_sh, v_ptr, v_sb, v_sn, v_ss, v_sh,
                         dk_ptr, dv_ptr, row, h, start, c, s0, s_count, n_count, s_total, H,
                         IS_WINDOW: tl.constexpr, W: tl.constexpr, BL: tl.constexpr, DH: tl.constexpr, scale):
    """One part: writes its key and value gradients (into contiguous ``[B,
    n_count, s_total, H, DH]``) and adds its share of the query gradient."""
    d = tl.arange(0, DH)
    for base in range(0, n_count * s_count, BL):
        s, seen, n, src, valid, k = _store_block(q, k_ptr, k_sb, k_sn, k_ss, k_sh, row, h, start, c, base,
                                                 s0, s_count, n_count, IS_WINDOW, W, BL, DH, scale)
        v = _load_block(v_ptr, v_sb, v_sn, v_ss, v_sh, row, h, n, src, valid, DH)
        p = tl.where(seen, tl.exp(s - lse[:, None]), 0.0)
        ds = p * (tl.dot(do, tl.trans(v), input_precision="ieee") - delta[:, None]) * scale
        out = (((row * n_count + n.to(tl.int64)) * s_total + src) * H + h) * DH
        tl.store(dk_ptr + out[:, None] + d[None, :], tl.dot(tl.trans(ds), q, input_precision="ieee"), mask=valid[:, None])
        tl.store(dv_ptr + out[:, None] + d[None, :], tl.dot(tl.trans(p), do, input_precision="ieee"), mask=valid[:, None])
        dq += tl.dot(ds, k, input_precision="ieee")
    return dq


@triton.jit
def store_attend_backward_kernel(q_ptr, out_ptr, lse_ptr, do_ptr, fb_out_ptr, fb_lse_ptr, fb_do_ptr, start_ptr, dq_ptr,
                                 wk_ptr, wk_sb, wk_sn, wk_ss, wk_sh, wv_ptr, wv_sb, wv_sn, wv_ss, wv_sh, dwk_ptr, dwv_ptr,
                                 ck_ptr, ck_sb, ck_sn, ck_ss, ck_sh, cv_ptr, cv_sb, cv_sn, cv_ss, cv_sh, c_s,
                                 dck_ptr, dcv_ptr, B, H, S, legal, shift, scale, FEEDBACK: tl.constexpr,
                                 W: tl.constexpr, C: tl.constexpr, DH: tl.constexpr, BL: tl.constexpr):
    pid = tl.program_id(0)
    b = (pid // H).to(tl.int64)
    h = pid % H
    c = tl.arange(0, C)
    d = tl.arange(0, DH)
    start = tl.load(start_ptr)
    rows = ((b * C + c) * H + h) * DH
    q = tl.load(q_ptr + rows[:, None] + d[None, :])
    do = tl.load(do_ptr + rows[:, None] + d[None, :])
    delta = tl.sum(do * tl.load(out_ptr + rows[:, None] + d[None, :]), axis=1)
    lse = tl.load(lse_ptr + pid * C + c)
    dq = tl.zeros([C, DH], dtype=tl.float32)
    dq = _store_part_backward(q, do, delta, lse, dq, wk_ptr, wk_sb, wk_sn, wk_ss, wk_sh, wv_ptr, wv_sb, wv_sn, wv_ss, wv_sh,
                              dwk_ptr, dwv_ptr, b, h, start, c, 0, legal, W, S, H, True, W, BL, DH, scale)
    dq = _store_part_backward(q, do, delta, lse, dq, ck_ptr, ck_sb, ck_sn, ck_ss, ck_sh, cv_ptr, cv_sb, cv_sn, cv_ss, cv_sh,
                              dck_ptr, dcv_ptr, b, h, start, c, 0, c_s, C, c_s, H, False, W, BL, DH, scale)
    if FEEDBACK:
        # Row b's feedback read of row b - shift's window: each window slot
        # of a feedback source is read by exactly one program.
        row = (b + B - shift) % B
        do = tl.load(fb_do_ptr + rows[:, None] + d[None, :])
        delta = tl.sum(do * tl.load(fb_out_ptr + rows[:, None] + d[None, :]), axis=1)
        lse = tl.load(fb_lse_ptr + pid * C + c)
        dq = _store_part_backward(q, do, delta, lse, dq, wk_ptr, wk_sb, wk_sn, wk_ss, wk_sh, wv_ptr, wv_sb, wv_sn, wv_ss, wv_sh,
                                  dwk_ptr, dwv_ptr, row, h, start, c, legal, S - legal, W, S, H, True, W, BL, DH, scale)
    tl.store(dq_ptr + rows[:, None] + d[None, :], dq)


def store_biases(start: Tensor, C: int, W: int) -> tuple[Tensor, Tensor]:
    """Masks of chunk token ``start + c`` over the kept window (slot ``n``
    holds token ``start - W + n``) and over the chunk's own tokens (``m``):
    an entry is seen if it exists, precedes the token and lies within
    ``W`` of it."""
    c = torch.arange(C, device=start.device)[:, None]
    n = torch.arange(W, device=start.device)
    m = torch.arange(C, device=start.device)
    window = (n >= W - start) & (n >= c)
    chunk = (m < c) & (m >= c - W)
    ninf = float("-inf")
    return torch.where(window, 0.0, ninf), torch.where(chunk, 0.0, ninf)


def _store_parts(q, wk, wv, ck, cv, start, scale):
    """Per part: keys, values (``q``'s dtype) and masked scores ``[B, H, C,
    N, s]``."""
    window_bias, chunk_bias = store_biases(start, q.size(1), wk.size(1))
    parts = [(wk, wv, window_bias)] + ([(ck, cv, chunk_bias)] if ck is not None else [])
    out = []
    for k, v, bias in parts:
        k, v = k.to(q.dtype), v.to(q.dtype)
        out.append((k, v, torch.einsum("bchd,bnshd->bhcns", q, k) * scale + bias[:, :, None]))
    return out


def _store_softmax_reference(q, wk, wv, ck, cv, start, scale) -> tuple[Tensor, Tensor]:
    """One read: ``q`` over the window ``wk, wv`` and, if given, the chunk
    entries ``ck, cv``, one softmax with a sink -> ``(out, lse [B * H,
    C])``."""
    B, C, H, _ = q.shape
    parts = _store_parts(q, wk, wv, ck, cv, start, scale)
    scores = [s.flatten(-2) for _, _, s in parts]
    # The sink's logit 0 joins the logsumexp itself: a token that sees no
    # entry has all scores -inf, where logsumexp's own gradient is NaN.
    lse = torch.cat(scores + [scores[0].new_zeros(scores[0].shape[:-1] + (1,))], dim=-1).logsumexp(-1)
    out = sum(torch.einsum("bhcns,bnshd->bchd", (s - lse[..., None, None]).exp(), v) for _, v, s in parts)
    return out, lse.reshape(B * H, C)


def _store_softmax_backward_reference(q, wk, wv, ck, cv, start, scale, out, lse, d_out) -> list[Tensor]:
    B, C, H, _ = q.shape
    lse = lse.view(B, H, C)
    delta = (d_out * out).sum(-1).permute(0, 2, 1)  # [B, H, C]
    dq = torch.zeros_like(q)
    grads = []
    for (k, v, s), (kk, vv) in zip(_store_parts(q, wk, wv, ck, cv, start, scale), [(wk, wv), (ck, cv)]):
        p = (s - lse[..., None, None]).exp()
        ds = p * (torch.einsum("bchd,bnshd->bhcns", d_out, v) - delta[..., None, None]) * scale
        dq = dq + torch.einsum("bhcns,bnshd->bchd", ds, k)
        grads += [torch.einsum("bhcns,bchd->bnshd", ds, q).to(kk.dtype), torch.einsum("bhcns,bchd->bnshd", p, d_out).to(vv.dtype)]
    return [dq] + grads


def store_attend_reference(q, wk, wv, ck, cv, start, scale, legal, feedback, shift) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``store_attend`` in PyTorch."""
    out, lse = _store_softmax_reference(q, wk[:, :, :legal], wv[:, :, :legal], ck, cv, start, scale)
    if not feedback:
        return out, q.new_empty(0), lse, q.new_empty(0)
    fb = [t.roll(shift, 0)[:, :, legal:] for t in (wk, wv)]
    fb_out, fb_lse = _store_softmax_reference(q, *fb, None, None, start, scale)
    return out, fb_out, lse, fb_lse


def store_attend_backward_reference(q, wk, wv, ck, cv, start, scale, legal, feedback, shift,
                                    out, fb_out, lse, fb_lse, d_out, d_fb) -> list[Tensor]:
    dq, dwk_legal, dwv_legal, dck, dcv = _store_softmax_backward_reference(
        q, wk[:, :, :legal], wv[:, :, :legal], ck, cv, start, scale, out, lse, d_out)
    dwk, dwv = torch.zeros_like(wk), torch.zeros_like(wv)
    dwk[:, :, :legal], dwv[:, :, :legal] = dwk_legal, dwv_legal
    if feedback:
        fb = [t.roll(shift, 0)[:, :, legal:] for t in (wk, wv)]
        dq_fb, dwk_fb, dwv_fb = _store_softmax_backward_reference(q, *fb, None, None, start, scale, fb_out, fb_lse, d_fb)
        dq = dq + dq_fb
        dwk[:, :, legal:], dwv[:, :, legal:] = dwk_fb.roll(-shift, 0), dwv_fb.roll(-shift, 0)
    return [dq, dwk, dwv, dck, dcv]


def _store_kernels(q: Tensor) -> bool:
    C, DH = q.shape[1], q.shape[3]
    return q.is_cuda and _kernel_shapes(C, DH)


def _unit_last(x: Tensor) -> Tensor:
    return x if x.stride(-1) == 1 else x.contiguous()


def _store_args(k: Tensor, v: Tensor) -> tuple:
    k, v = _unit_last(k), _unit_last(v)
    return (k, *k.stride()[:4], v, *v.stride()[:4])


def _check_store(q: Tensor, wk: Tensor, ck: Tensor, legal: int, feedback: bool) -> None:
    if not 0 < legal <= wk.size(2) or (feedback and legal == wk.size(2)):
        raise ValueError(f"store reads of {legal} legal sources of {wk.size(2)} (feedback {feedback})")
    if wk.shape[0] != q.shape[0] or ck.shape[:2] != q.shape[:2]:
        raise ValueError("store reads take the window and chunk entries of the queries' rows and chunk")


@torch.library.custom_op("cwm::store_attend", mutates_args=())
def store_attend(q: Tensor, wk: Tensor, wv: Tensor, ck: Tensor, cv: Tensor, start: Tensor, scale: float,
                 legal: int, feedback: bool, shift: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """A round's store reads: queries ``q [B, C, H, Dh]`` fp32 (chunk tokens
    ``start + c``) over the kept window ``wk, wv [B, W, S, H, Dh]`` and the
    chunk's own entries ``ck, cv [B, C, s, H, Dh]``. The legal read covers
    window sources ``:legal`` and the chunk entries under one softmax with
    a sink (logit 0); with ``feedback``, the feedback read covers window
    sources ``legal:`` of row ``b - shift`` (the placebo's shift, else 0)
    under another. -> ``(out, fb_out [B, C, H, Dh], lse, fb_lse [B * H,
    C])`` fp32; the feedback pair is empty without feedback."""
    _check_store(q, wk, ck, legal, feedback)
    if not _store_kernels(q):
        return tuple(t.contiguous() for t in store_attend_reference(q, wk, wv, ck, cv, start, scale, legal, feedback, shift))
    B, C, H, DH = q.shape
    q = q.contiguous()
    out = torch.empty_like(q)
    lse = torch.empty(B * H, C, device=q.device, dtype=torch.float32)
    fb_out, fb_lse = (torch.empty_like(out), torch.empty_like(lse)) if feedback else (q.new_empty(0), q.new_empty(0))
    store_attend_kernel[(B * H,)](q, out, lse, fb_out if feedback else out, fb_lse if feedback else lse, start,
                                  *_store_args(wk, wv), *_store_args(ck, cv), ck.size(2), B, H, wk.size(2), legal,
                                  shift, scale, FEEDBACK=feedback, W=wk.size(1), C=C, DH=DH, BL=STORE_BLOCK,
                                  num_warps=4)
    return out, fb_out, lse, fb_lse


@store_attend.register_fake
def _(q, wk, wv, ck, cv, start, scale, legal, feedback, shift):
    B, C, H, _ = q.shape
    out, lse = torch.empty_like(q, memory_format=torch.contiguous_format), q.new_empty(B * H, C)
    if not feedback:
        return out, q.new_empty(0), lse, q.new_empty(0)
    return out, torch.empty_like(out), lse, torch.empty_like(lse)


@torch.library.custom_op("cwm::store_attend_backward", mutates_args=())
def store_attend_backward(q: Tensor, wk: Tensor, wv: Tensor, ck: Tensor, cv: Tensor, start: Tensor, scale: float,
                          legal: int, feedback: bool, shift: int, out: Tensor, fb_out: Tensor, lse: Tensor,
                          fb_lse: Tensor, d_out: Tensor, d_fb: Tensor) -> list[Tensor]:
    """``[dq, dwk, dwv, dck, dcv]``, contiguous; the window's gradient is
    zero at sources no read covers."""
    if not _store_kernels(q):
        grads = store_attend_backward_reference(q, wk, wv, ck, cv, start, scale, legal, feedback, shift,
                                                out, fb_out, lse, fb_lse, d_out, d_fb)
        return [g.contiguous() for g in grads]
    B, C, H, DH = q.shape
    q, d_out = q.contiguous(), d_out.contiguous()
    dq = torch.empty_like(q)
    # Every window slot is written when the feedback read covers the rest.
    window_grad = torch.empty if feedback else torch.zeros
    dwk, dwv = (window_grad(t.shape, device=q.device, dtype=t.dtype) for t in (wk, wv))
    dck, dcv = (torch.empty(t.shape, device=q.device, dtype=t.dtype) for t in (ck, cv))
    fb = (fb_out, fb_lse, d_fb.contiguous()) if feedback else (out, lse, d_out)
    store_attend_backward_kernel[(B * H,)](q, out, lse, d_out, *fb, start, dq, *_store_args(wk, wv), dwk, dwv,
                                           *_store_args(ck, cv), ck.size(2), dck, dcv, B, H, wk.size(2), legal, shift,
                                           scale, FEEDBACK=feedback, W=wk.size(1), C=C, DH=DH, BL=STORE_BLOCK,
                                           num_warps=4)
    return [dq, dwk, dwv, dck, dcv]


@store_attend_backward.register_fake
def _(q, wk, wv, ck, cv, start, scale, legal, feedback, shift, out, fb_out, lse, fb_lse, d_out, d_fb):
    return [torch.empty_like(t, memory_format=torch.contiguous_format) for t in (q, wk, wv, ck, cv)]


def _store_attend_setup(ctx, inputs, output) -> None:
    q, wk, wv, ck, cv, start, scale, legal, feedback, shift = inputs
    ctx.args = (scale, legal, feedback, shift)
    ctx.save_for_backward(q, wk, wv, ck, cv, start, *output)


def _store_attend_grad(ctx, d_out: Tensor, d_fb: Tensor, d_lse: Tensor, d_fb_lse: Tensor):
    q, wk, wv, ck, cv, start, out, fb_out, lse, fb_lse = ctx.saved_tensors
    scale, legal, feedback, shift = ctx.args
    # A read whose output no loss reaches gets no gradient from autograd.
    d_out = torch.zeros_like(out) if d_out is None else d_out
    d_fb = torch.zeros_like(fb_out) if d_fb is None else d_fb
    grads = store_attend_backward(q, wk, wv, ck, cv, start, scale, legal, feedback, shift,
                                  out, fb_out, lse, fb_lse, d_out, d_fb)
    return (*grads, None, None, None, None, None)


store_attend.register_autograd(_store_attend_grad, setup_context=_store_attend_setup)
