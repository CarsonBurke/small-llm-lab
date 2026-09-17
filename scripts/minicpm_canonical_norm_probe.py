"""Opt-in fixed-row RMSNorm reduction, preserving HF casts and parameter names."""
from types import MethodType
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_kernel(X, W, Y, RSTD, WIDTH: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    x = tl.load(X + row*WIDTH + columns, columns < WIDTH, other=0).to(tl.float32)
    variance = tl.sum(x*x, axis=0) / WIDTH
    inverse = tl.rsqrt(variance + EPS)
    # HF rounds the normalized activation before multiplication by its weight.
    normalized = (x*inverse).to(X.dtype.element_ty).to(tl.float32)
    weight = tl.load(W+columns, columns < WIDTH, other=0).to(tl.float32)
    tl.store(Y+row*WIDTH+columns, normalized*weight, columns < WIDTH)
    tl.store(RSTD+row, inverse)


class _RMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, epsilon):
        x = inputs.contiguous()
        output = torch.empty_like(x)
        inverse = torch.empty(x.numel()//x.shape[-1], device=x.device, dtype=torch.float32)
        _rms_kernel[(inverse.numel(),)](x, weight, output, inverse, x.shape[-1], epsilon,
            triton.next_power_of_2(x.shape[-1]), num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(x, weight, inverse)
        return output

    @staticmethod
    def backward(ctx, gradient):
        inputs, weight, inverse = ctx.saved_tensors
        with torch.autocast('cuda', enabled=False):
            x = inputs.float().reshape(-1, inputs.shape[-1])
            inverse = inverse[:, None]
            normalized = x*inverse
            grad = gradient.float().reshape_as(x)
            scaled = grad*weight.float()
            dx = (scaled - normalized*(scaled*normalized).mean(-1, keepdim=True))*inverse
            dw = (grad*normalized).sum(0) if ctx.needs_input_grad[1] else None
        return dx.to(inputs.dtype).reshape_as(inputs), dw, None


def canonical_norm(inputs, weight, epsilon):
    if not inputs.is_cuda or inputs.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError('canonical RMSNorm requires CUDA bf16/fp32 residuals')
    if weight.dtype != torch.bfloat16:
        raise ValueError('canonical RMSNorm keeps bf16 frozen model weights')
    return _RMS.apply(inputs, weight, epsilon)


def _forward(module, inputs):
    return canonical_norm(inputs, module.weight, module.variance_epsilon)


def install_canonical_norms_(causal_lm):
    modules = [causal_lm.model.norm]
    for layer in causal_lm.model.layers:
        modules.extend((layer.input_layernorm, layer.post_attention_layernorm))
    for module in modules:
        module.forward = MethodType(_forward, module)
    return len(modules)
