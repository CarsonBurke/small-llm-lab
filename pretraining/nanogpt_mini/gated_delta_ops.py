"""Opaque custom operators around the installed FLA GDN-2 Triton kernels.

FLA wraps its kernels in autograd functions declared with
``torch.compiler.disable``. Compiling the production model around them splits
every layer into several graphs. Each split materializes boundary tensors,
copies the seven packed-projection gradient slices before concatenating them,
and accumulates the fp32 residual-stream gradient eagerly between graphs.

These operators launch exactly the same kernels on exactly the same kernel
inputs, but register them through ``torch.library`` so the surrounding dense
arithmetic compiles as one graph. Training shapes only: no cache, no initial or
final state, no variable-length packing. Cached evaluation keeps FLA's modules.
Backward gates return the kernels' fp32 gradients; the caller's compiled graph
fuses the cast into the consuming sigmoid derivative instead of a copy pass.

Two installed-kernel options are exposed on the chunk operator. With ``A_log``
and ``dt_bias`` given, the decay gate ``-exp(A_log) * softplus(g + dt_bias)``
is computed inside FLA's chunk cumsum kernel from the raw projection instead of
in the surrounding graph, and its backward returns the two parameter gradients.
FLA's sub-chunk ``safe_gate`` kernels are deliberately not exposed: they
exponentiate log-decay differences against the sub-chunk midpoint in both
directions and require per-token log-decay within [-5, 0), which GDN2's
unbounded -exp(A_log) * softplus(...) activation violates on trained models.
"""
from __future__ import annotations

import torch
from torch import Tensor
import fla.utils

CHUNK_SIZE = 64
# FLA memoizes its packed-sequence index helpers by argument *identity*. Under CUDA-graph
# capture a hit would splice a stale tensor into the graph instead of recording the
# computation, so the identity cache is off for this process (the batch path never uses it).
fla.utils.FLA_DISABLE_TENSOR_CACHE = True
LIBRARY = "gdn2"
# Inductor must hand the real kernels exactly the strides the fake kernels traced: the Triton
# launchers index raw pointers with the strides they were compiled for. This is torch's current
# default for custom operators; declaring it keeps the contract if that default ever changes.
TAGS = (torch.Tag.needs_exact_strides,)
Tensor2 = tuple[Tensor, Tensor]
Tensor3 = tuple[Tensor, Tensor, Tensor]
Tensor8 = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
Tensor14 = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,
                 Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


def _require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def _require_layout(name: str, tensor: Tensor, dims: int):
    _require(tensor.is_cuda, f"{name} must be a CUDA tensor")
    _require(tensor.ndim == dims, f"{name} must have {dims} dimensions")
    _require(tensor.is_contiguous(), f"{name} must be contiguous; kernels read it row-major")


def _require_packing(q: Tensor, cu_seqlens: Tensor | None, chunk_indices: Tensor | None) -> bool:
    """Whether the call is a packed (variable-length) sequence; validates the index pair.

    Packed calls carry one row of ``T`` tokens split into segments at
    ``cu_seqlens`` (int32 offsets ending at ``T``), and ``chunk_indices`` maps
    every 64-token chunk to (segment, chunk within the segment). Segments must
    be chunk-aligned so the chunk count stays ``T // 64``: shapes are then
    static and the call captures into CUDA graphs like the batch path.
    """
    _require((cu_seqlens is None) == (chunk_indices is None), "cu_seqlens and chunk_indices pack sequences together")
    if cu_seqlens is None:
        return False
    _require(q.shape[0] == 1, "packed sequences occupy one batch row")
    _require(cu_seqlens.is_cuda and cu_seqlens.ndim == 1 and cu_seqlens.dtype == torch.int32
             and cu_seqlens.is_contiguous() and cu_seqlens.numel() >= 2, "cu_seqlens must be contiguous int32 offsets")
    _require(chunk_indices.is_cuda and chunk_indices.dtype == torch.int32 and chunk_indices.is_contiguous()
             and chunk_indices.shape == (q.shape[1] // CHUNK_SIZE, 2),
             "chunk_indices must map exactly T // 64 chunks to (segment, chunk) pairs in int32")
    return True


def _require_decay_parameters(g: Tensor, A_log: Tensor | None, dt_bias: Tensor | None) -> bool:
    """Whether the kernels compute the decay gate; validates the parameter pair."""
    _require((A_log is None) == (dt_bias is None), "A_log and dt_bias select the in-kernel decay gate together")
    if A_log is None:
        return False
    heads, key_dim = g.shape[2], g.shape[3]
    _require(A_log.shape == (heads,) and dt_bias.shape == (heads * key_dim,),
             "A_log is one log-rate per head and dt_bias one bias per key channel")
    _require(A_log.dtype == dt_bias.dtype == torch.float32 and A_log.is_contiguous() and dt_bias.is_contiguous(),
             "decay parameters must be contiguous fp32")
    return True


def _chunk_fwd_real(q: Tensor, k: Tensor, v: Tensor, g: Tensor, b: Tensor, w: Tensor,
                    A_log: Tensor | None, dt_bias: Tensor | None, cu_seqlens: Tensor | None,
                    chunk_indices: Tensor | None, scale: float, state_v_first: bool) -> Tensor14:
    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.gdn2.chunk_fwd import chunk_gdn2_fwd

    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("b", b), ("w", w)):
        _require_layout(name, tensor, 4)
    _require(q.shape == k.shape == g.shape == b.shape and v.shape == w.shape
             and v.shape[:3] == q.shape[:3], "q, k, g, b and v, w shapes must agree")
    _require(q.shape[1] % CHUNK_SIZE == 0, "training sequences must be a multiple of the kernel chunk")
    gate_in_kernel = _require_decay_parameters(g, A_log, dt_bias)
    _require_packing(q, cu_seqlens, chunk_indices)
    normalized_q, q_rstd = l2norm_fwd(q)
    normalized_k, k_rstd = l2norm_fwd(k)
    (o, final_state, g_cumsum, Aqk, Akk, w_wy, u_wy, qg, kg, v_new, h, _) = chunk_gdn2_fwd(
        q=normalized_q, k=normalized_k, v=v, g=g, b=b, w_gate=w, scale=scale,
        initial_state=None, output_final_state=False, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        chunk_size=CHUNK_SIZE, safe_gate=False, lower_bound=None, use_gate_in_kernel=gate_in_kernel,
        A_log=A_log, dt_bias=dt_bias, disable_recompute=True, return_intermediate_states=False,
        state_v_first=state_v_first)
    _require(final_state is None and o.dtype == q.dtype and g_cumsum.dtype == torch.float32
             and g_cumsum.shape == g.shape, "unexpected kernel output contract")
    return (o, normalized_q, q_rstd, normalized_k, k_rstd, g_cumsum, Aqk, Akk,
            w_wy, u_wy, qg, kg, v_new, h)


def _chunk_fwd_fake(q: Tensor, k: Tensor, v: Tensor, g: Tensor, b: Tensor, w: Tensor,
                    A_log: Tensor | None, dt_bias: Tensor | None, cu_seqlens: Tensor | None,
                    chunk_indices: Tensor | None, scale: float, state_v_first: bool) -> Tensor14:
    batch, time, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    # Packed calls keep one row of chunk-aligned segments, so the state count is time // 64 either way.
    chunks = time // CHUNK_SIZE
    rows = (batch, time, heads)
    state_shape = (batch, chunks, heads) + ((value_dim, key_dim) if state_v_first else (key_dim, value_dim))
    return (torch.empty_like(v), torch.empty_like(q), q.new_empty(rows, dtype=torch.float32),
            torch.empty_like(k), k.new_empty(rows, dtype=torch.float32),
            g.new_empty(g.shape, dtype=torch.float32),
            k.new_empty((*rows, CHUNK_SIZE)), k.new_empty((*rows, CHUNK_SIZE)),
            torch.empty_like(k), torch.empty_like(v), torch.empty_like(q), torch.empty_like(k),
            torch.empty_like(v), k.new_empty(state_shape))


chunk_fwd = torch.library.custom_op(f"{LIBRARY}::chunk_fwd", _chunk_fwd_real, mutates_args=(), tags=TAGS)
chunk_fwd.register_fake(_chunk_fwd_fake)


def _chunk_bwd_real(do: Tensor, normalized_q: Tensor, q_rstd: Tensor, normalized_k: Tensor,
                    k_rstd: Tensor, v: Tensor, g_cumsum: Tensor, b: Tensor, w: Tensor,
                    Aqk: Tensor, Akk: Tensor, w_wy: Tensor, u_wy: Tensor, qg: Tensor, kg: Tensor,
                    v_new: Tensor, h: Tensor, g: Tensor | None, A_log: Tensor | None, dt_bias: Tensor | None,
                    cu_seqlens: Tensor | None, chunk_indices: Tensor | None,
                    scale: float, state_v_first: bool) -> Tensor8:
    """Kernel backward; ``g`` is the raw decay projection when the kernels computed the gate.

    The last two outputs are the A_log and dt_bias gradients, or empty tensors
    when the gate was computed outside the kernels.
    """
    from fla.modules.l2norm import l2norm_bwd
    from fla.ops.gdn2.chunk_bwd import chunk_gdn2_bwd

    _require_layout("do", do, 4)
    gate_in_kernel = _require_decay_parameters(g_cumsum, A_log, dt_bias)
    _require(gate_in_kernel == (g is not None), "the raw decay projection accompanies the decay parameters")
    _require_packing(normalized_q, cu_seqlens, chunk_indices)
    dq, dk, dv, db, dw, dg, dh0, dA_log, dt_bias_grad = chunk_gdn2_bwd(
        q=normalized_q, k=normalized_k, v=v, b=b, w_gate=w, Aqk=Aqk, Akk=Akk, scale=scale,
        initial_state=None, do=do, dht=None, g=g_cumsum, g_org=g, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, chunk_size=CHUNK_SIZE, safe_gate=False, lower_bound=None,
        use_gate_in_kernel=gate_in_kernel, A_log=A_log, dt_bias=dt_bias, state_v_first=state_v_first,
        w_wy=w_wy, u_wy=u_wy, qg=qg, kg=kg, v_new=v_new, h=h, disable_recompute=True)
    _require(dh0 is None and (dA_log is None) == (dt_bias_grad is None) == (not gate_in_kernel),
             "unexpected kernel gradient contract")
    dq = l2norm_bwd(normalized_q, q_rstd, dq)
    dk = l2norm_bwd(normalized_k, k_rstd, dk)
    _require(dq.dtype == normalized_q.dtype and dk.dtype == normalized_k.dtype and dv.dtype == v.dtype
             and db.dtype == dw.dtype == torch.float32, "unexpected kernel gradient dtypes")
    if gate_in_kernel:
        _require(dg.dtype == g.dtype and dg.shape == g.shape and dA_log.shape == A_log.shape
                 and dt_bias_grad.shape == dt_bias.shape and dA_log.dtype == dt_bias_grad.dtype == torch.float32,
                 "unexpected decay gate gradient contract")
        return dq, dk, dv, dg, db, dw, dA_log, dt_bias_grad
    _require(dg.dtype == torch.float32, "unexpected kernel gradient dtypes")
    return dq, dk, dv, dg, db, dw, g_cumsum.new_empty(0), g_cumsum.new_empty(0)


def _chunk_bwd_fake(do: Tensor, normalized_q: Tensor, q_rstd: Tensor, normalized_k: Tensor,
                    k_rstd: Tensor, v: Tensor, g_cumsum: Tensor, b: Tensor, w: Tensor,
                    Aqk: Tensor, Akk: Tensor, w_wy: Tensor, u_wy: Tensor, qg: Tensor, kg: Tensor,
                    v_new: Tensor, h: Tensor, g: Tensor | None, A_log: Tensor | None, dt_bias: Tensor | None,
                    cu_seqlens: Tensor | None, chunk_indices: Tensor | None,
                    scale: float, state_v_first: bool) -> Tensor8:
    if A_log is None:
        dg, dA_log, dt_bias_grad = torch.empty_like(g_cumsum), g_cumsum.new_empty(0), g_cumsum.new_empty(0)
    else:
        dg, dA_log, dt_bias_grad = torch.empty_like(g), torch.empty_like(A_log), torch.empty_like(dt_bias)
    return (torch.empty_like(normalized_q), torch.empty_like(normalized_k), torch.empty_like(v),
            dg, b.new_empty(b.shape, dtype=torch.float32), w.new_empty(w.shape, dtype=torch.float32),
            dA_log, dt_bias_grad)


chunk_bwd = torch.library.custom_op(f"{LIBRARY}::chunk_bwd", _chunk_bwd_real, mutates_args=(), tags=TAGS)
chunk_bwd.register_fake(_chunk_bwd_fake)


def _chunk_setup(ctx, inputs, output):
    q, k, v, g, b, w, A_log, dt_bias, cu_seqlens, chunk_indices, scale, state_v_first = inputs
    ctx.set_materialize_grads(False)
    ctx.scale, ctx.state_v_first = scale, state_v_first
    ctx.gate_dtypes = (b.dtype, w.dtype)
    ctx.save_for_backward(v, b, w, g if A_log is not None else None, A_log, dt_bias, cu_seqlens, chunk_indices,
                          *output[1:])


def _chunk_backward(ctx, do, *unused):
    (v, b, w, g, A_log, dt_bias, cu_seqlens, chunk_indices, normalized_q, q_rstd, normalized_k, k_rstd, g_cumsum,
     Aqk, Akk, w_wy, u_wy, qg, kg, v_new, h) = ctx.saved_tensors
    # Incoming gradients are contiguous in the compiled training graph; the
    # copy below only materializes expanded or strided gradients elsewhere.
    dq, dk, dv, dg, db, dw, dA_log, dt_bias_grad = chunk_bwd(
        do.contiguous(), normalized_q, q_rstd, normalized_k, k_rstd, v, g_cumsum, b, w,
        Aqk, Akk, w_wy, u_wy, qg, kg, v_new, h, g, A_log, dt_bias, cu_seqlens, chunk_indices,
        ctx.scale, ctx.state_v_first)
    b_dtype, w_dtype = ctx.gate_dtypes
    if A_log is None:
        dA_log, dt_bias_grad = None, None
    return dq, dk, dv, dg, db.to(b_dtype), dw.to(w_dtype), dA_log, dt_bias_grad, None, None, None, None


chunk_fwd.register_autograd(_chunk_backward, setup_context=_chunk_setup)


def chunk_gdn2_training(q: Tensor, k: Tensor, v: Tensor, g: Tensor, b: Tensor, w: Tensor, *,
                        A_log: Tensor | None = None, dt_bias: Tensor | None = None,
                        cu_seqlens: Tensor | None = None, chunk_indices: Tensor | None = None,
                        scale: float | None = None, state_v_first: bool) -> Tensor:
    """Training-shaped ``chunk_gdn2`` with in-kernel q/k L2 normalization.

    ``g`` is the channel-wise log-decay, or the raw decay projection when
    ``A_log`` and ``dt_bias`` are given and the kernels compute the gate.
    ``cu_seqlens`` and ``chunk_indices`` run one batch row as packed,
    chunk-aligned segments with independent states (see ``_require_packing``).
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return chunk_fwd(q, k, v, g, b, w, A_log, dt_bias, cu_seqlens, chunk_indices, float(scale), bool(state_v_first))[0]


def _conv_fwd_real(x: Tensor, weight: Tensor) -> Tensor:
    from fla.modules.conv.triton.ops import causal_conv1d_fwd

    _require(x.is_cuda and x.ndim == 3 and x.stride(-1) == 1, "x must be [batch, time, channels] with unit channel stride")
    _require(weight.ndim == 2 and weight.shape[0] == x.shape[-1], "weight must be [channels, width]")
    y, final_state = causal_conv1d_fwd(x=x, weight=weight, bias=None, residual=None, initial_state=None,
                                       output_final_state=False, activation="silu", BT=CHUNK_SIZE)
    _require(final_state is None and y.is_contiguous(), "unexpected convolution output contract")
    return y


def _conv_fwd_fake(x: Tensor, weight: Tensor) -> Tensor:
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


conv_fwd = torch.library.custom_op(f"{LIBRARY}::causal_conv1d_silu_fwd", _conv_fwd_real, mutates_args=(), tags=TAGS)
conv_fwd.register_fake(_conv_fwd_fake)


def _conv_bwd_real(x: Tensor, dy: Tensor, weight: Tensor) -> Tensor2:
    from fla.modules.conv.triton.ops import causal_conv1d_bwd

    _require(dy.shape == x.shape and dy.stride(-1) == 1, "dy must match x with unit channel stride")
    dx, dw, db, dr, dh0 = causal_conv1d_bwd(x=x, dy=dy, dht=None, weight=weight, bias=None, residual=None,
                                            initial_state=None, activation="silu", BT=CHUNK_SIZE)
    _require(db is None and dr is None and dh0 is None and dw.dtype == weight.dtype,
             "unexpected convolution gradient contract")
    return dx, dw


def _conv_bwd_fake(x: Tensor, dy: Tensor, weight: Tensor) -> Tensor2:
    return torch.empty(x.shape, dtype=x.dtype, device=x.device), torch.empty_like(weight)


conv_bwd = torch.library.custom_op(f"{LIBRARY}::causal_conv1d_silu_bwd", _conv_bwd_real, mutates_args=(), tags=TAGS)
conv_bwd.register_fake(_conv_bwd_fake)


def _conv_setup(ctx, inputs, output):
    x, weight = inputs
    ctx.save_for_backward(x, weight)


def _conv_backward(ctx, dy):
    x, weight = ctx.saved_tensors
    return conv_bwd(x, dy if dy.stride(-1) == 1 else dy.contiguous(), weight)


conv_fwd.register_autograd(_conv_backward, setup_context=_conv_setup)


def causal_conv1d_silu(x: Tensor, weight: Tensor) -> Tensor:
    """Depthwise causal short convolution with SiLU; ``x`` may be a channel slice."""
    return conv_fwd(x, weight)


def _norm_fwd_real(x: Tensor, gate: Tensor, weight: Tensor, eps: float) -> Tensor2:
    from fla.modules.fused_norm_gate import layer_norm_gated_fwd

    _require_layout("x", x, 4)
    _require_layout("gate", gate, 4)
    _require(gate.shape == x.shape and weight.shape == (x.shape[-1],), "gate and weight must match x")
    rows = x.reshape(-1, x.shape[-1])
    y, mean, rstd, saved = layer_norm_gated_fwd(x=rows, g=gate.reshape(rows.shape), weight=weight, bias=None,
                                                activation="swish", eps=eps, residual=None, out_dtype=None,
                                                residual_dtype=None, is_rms_norm=True)
    _require(mean is None and saved is rows and y.dtype == x.dtype, "unexpected gated norm contract")
    return y.view(x.shape), rstd


def _norm_fwd_fake(x: Tensor, gate: Tensor, weight: Tensor, eps: float) -> Tensor2:
    return torch.empty_like(x), x.new_empty((x.numel() // x.shape[-1],), dtype=torch.float32)


norm_fwd = torch.library.custom_op(f"{LIBRARY}::rms_norm_swish_gate_fwd", _norm_fwd_real, mutates_args=(), tags=TAGS)
norm_fwd.register_fake(_norm_fwd_fake)


def _norm_bwd_real(dy: Tensor, x: Tensor, gate: Tensor, weight: Tensor, rstd: Tensor, eps: float) -> Tensor3:
    from fla.modules.fused_norm_gate import layer_norm_gated_bwd

    _require_layout("dy", dy, 4)
    rows = x.reshape(-1, x.shape[-1])
    dx, dg, dw, db, dresidual = layer_norm_gated_bwd(dy=dy.reshape(rows.shape), x=rows, g=gate.reshape(rows.shape),
                                                     weight=weight, bias=None, activation="swish", eps=eps,
                                                     mean=None, rstd=rstd, dresidual=None, has_residual=False,
                                                     is_rms_norm=True, x_dtype=x.dtype)
    _require(db is None and dresidual is None and dw.dtype == weight.dtype and dx.dtype == dg.dtype == x.dtype,
             "unexpected gated norm gradient contract")
    return dx.view(x.shape), dg.view(x.shape), dw


def _norm_bwd_fake(dy: Tensor, x: Tensor, gate: Tensor, weight: Tensor, rstd: Tensor, eps: float) -> Tensor3:
    return torch.empty_like(x), torch.empty_like(x), torch.empty_like(weight)


norm_bwd = torch.library.custom_op(f"{LIBRARY}::rms_norm_swish_gate_bwd", _norm_bwd_real, mutates_args=(), tags=TAGS)
norm_bwd.register_fake(_norm_bwd_fake)


def _norm_setup(ctx, inputs, output):
    x, gate, weight, eps = inputs
    ctx.set_materialize_grads(False)
    ctx.eps = eps
    ctx.save_for_backward(x, gate, weight, output[1])


def _norm_backward(ctx, dy, unused_rstd):
    x, gate, weight, rstd = ctx.saved_tensors
    dx, dg, dw = norm_bwd(dy.contiguous(), x, gate, weight, rstd, ctx.eps)
    return dx, dg, dw, None


norm_fwd.register_autograd(_norm_backward, setup_context=_norm_setup)


def rms_norm_swish_gate(x: Tensor, gate: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Per-head RMS normalization of ``x`` multiplied by SiLU(gate)."""
    return norm_fwd(x, gate, weight, float(eps))[0]
