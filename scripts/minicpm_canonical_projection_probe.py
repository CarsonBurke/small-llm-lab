"""Opt-in canonical bf16 projections; importing this module changes no model.

Install on ``policy.causal_lm``, never on the whole latent policy: the Gaussian
transition/head keeps its separate fp32 arithmetic. Training evaluates a single
quantized merged weight, not base(x) + adapter(x). Its bf16 cast uses PyTorch's
straight-through cast derivative, so fp32 LoRA masters remain trainable.

Forward uses fixed M64/N128/K32 tensorcore tiles and fp32 accumulators. Backward
is the usual dense linear derivative with bf16 PyTorch matmuls; it is not claimed
to be width-invariant. Exact forward parity still needs GPU qualification.

QKV and gate/up fusion MUST start each original projection on an N128 boundary.
The installation/synchronization helpers reject incompatible fused layouts;
using an unaligned packed projection would change its column-lane arithmetic.
Existing production merge/synchronization helpers use DIFFERENT rounding and
must not refresh a canonical replica. Use synchronize_canonical_lora_ instead.
"""
from __future__ import annotations

from types import MethodType
from typing import Any

import torch
from torch import Tensor, nn
import triton
import triton.language as tl

from postraining.fast_inference import _FusedFirstProjection, _FusedProjectionSlice
from postraining.vapo.model.lora import LoRALinear


@triton.jit
def _canonical_linear_kernel(
    inputs, weight, output,
    ROWS: tl.constexpr, WIDTH: tl.constexpr,
    INPUT: tl.constexpr, OUTPUT: tl.constexpr,
    STRIDE_BATCH: tl.constexpr, STRIDE_TOKEN: tl.constexpr,
    STRIDE_INPUT: tl.constexpr,
):
    rows = tl.program_id(0) * 64 + tl.arange(0, 64)
    columns = tl.program_id(1) * 128 + tl.arange(0, 128)
    reduction = tl.arange(0, 32)
    offsets = (rows // WIDTH) * STRIDE_BATCH + (rows % WIDTH) * STRIDE_TOKEN
    accumulator = tl.full((64, 128), 0, tl.float32)
    for block in range(tl.cdiv(INPUT, 32)):
        indices = block * 32 + reduction
        left = tl.load(
            inputs + offsets[:, None] + indices[None, :] * STRIDE_INPUT,
            (rows[:, None] < ROWS) & (indices[None, :] < INPUT), other=0,
        )
        right = tl.load(
            weight + columns[None, :] * INPUT + indices[:, None],
            (columns[None, :] < OUTPUT) & (indices[:, None] < INPUT), other=0,
        )
        accumulator = tl.dot(left, right, accumulator, out_dtype=tl.float32)
    tl.store(
        output + rows[:, None] * OUTPUT + columns[None, :], accumulator,
        (rows[:, None] < ROWS) & (columns[None, :] < OUTPUT),
    )


class _CanonicalLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(inputs, weight)
        rows = inputs[:, None, :] if inputs.ndim == 2 else inputs
        batch, width, hidden = rows.shape
        output = torch.empty(
            (*inputs.shape[:-1], weight.shape[0]),
            dtype=torch.bfloat16, device=inputs.device,
        )
        if batch and width and weight.shape[0]:
            _canonical_linear_kernel[
                (triton.cdiv(batch * width, 64), triton.cdiv(weight.shape[0], 128))
            ](
                rows, weight, output, batch * width, width, hidden,
                weight.shape[0], *rows.stride(), num_warps=4, num_stages=3,
                enable_fp_fusion=False,
            )
        return output

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor | None, Tensor | None]:
        inputs, weight = ctx.saved_tensors
        grad_inputs = grad_weight = None
        # Explicit bf16 operands/output, independent of the enclosing autocast
        # state. cuBLAS selects its normal bf16 tensorcore implementation. This
        # does not change global TF32/reduced-precision-reduction settings.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            rows = inputs.numel() // inputs.shape[-1]
            grad_rows = gradient.to(torch.bfloat16).reshape(rows, weight.shape[0])
            if ctx.needs_input_grad[0]:
                grad_inputs = (grad_rows @ weight).reshape(inputs.shape)
            if ctx.needs_input_grad[1]:
                input_rows = inputs.reshape(rows, inputs.shape[-1])
                grad_weight = grad_rows.transpose(0, 1) @ input_rows
        return grad_inputs, grad_weight


def canonical_linear(inputs: Tensor, weight: Tensor) -> Tensor:
    """CUDA bf16 [rows, K] or [batch, tokens, K] times contiguous [N, K].

    Input strides are honored without a forward copy. Output is contiguous bf16
    with the original leading dimensions. Cast fp32 residual storage to bf16
    before calling; that differentiable cast returns fp32 residual gradients.
    """
    if inputs.ndim not in (2, 3) or weight.ndim != 2:
        raise ValueError("canonical linear requires 2D/3D inputs and a 2D weight")
    if inputs.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("canonical linear requires bf16 operands")
    if not inputs.is_cuda or inputs.device != weight.device:
        raise ValueError("canonical linear requires operands on the same CUDA device")
    if inputs.shape[-1] != weight.shape[1] or not inputs.shape[-1]:
        raise ValueError("canonical linear requires matching nonempty input features")
    if not weight.is_contiguous():
        raise ValueError("canonical linear requires a contiguous weight")
    return _CanonicalLinearFunction.apply(inputs, weight)


def canonical_merged_weight(module: LoRALinear) -> Tensor:
    """One merge definition for live autograd and no-grad replica refresh.

    Do not detach, wrap the result in Parameter, or copy it into a training base:
    float -> bf16 has a straight-through cast backward, connecting the dense
    weight gradient to both fp32 masters through the actual B @ A construction.
    Only weight construction is fp32; all token projection GEMMs remain bf16.
    """
    if module.base.weight.dtype != torch.bfloat16:
        raise ValueError("canonical merge requires a bf16 base weight")
    if module.lora_a.dtype != torch.float32 or module.lora_b.dtype != torch.float32:
        raise ValueError("canonical merge requires fp32 LoRA master parameters")
    with torch.autocast(device_type=module.base.weight.device.type, enabled=False):
        return (
            module.base.weight.float()
            + module.scaling * (module.lora_b.float() @ module.lora_a.float())
        ).to(torch.bfloat16)


def _canonical_lora_forward(module: LoRALinear, inputs: Tensor) -> Tensor:
    return canonical_linear(inputs.to(torch.bfloat16), canonical_merged_weight(module))


def _canonical_merged_forward(module: nn.Linear, inputs: Tensor) -> Tensor:
    return canonical_linear(inputs.to(torch.bfloat16), module.weight)


def install_canonical_lora_(causal_lm: nn.Module) -> tuple[str, ...]:
    """Override only LoRALinear.forward, preserving all parameter names/identities.

    Pass the causal LM, not the containing latent policy. No modules, parameters,
    optimizer references, state-dict keys, or original class symbols are replaced.
    Install before constructing/compiling the probe's forwards.
    """
    projections = [
        (name, module) for name, module in causal_lm.named_modules()
        if isinstance(module, LoRALinear)
    ]
    for name, module in projections:
        if module.base.bias is not None:
            raise ValueError(f"canonical LoRA requires bias-free projections: {name}")
        if module.base.weight.dtype != torch.bfloat16:
            raise ValueError(f"canonical LoRA requires bf16 base weights: {name}")
        if module.lora_a.dtype != torch.float32 or module.lora_b.dtype != torch.float32:
            raise ValueError(f"canonical LoRA requires fp32 masters: {name}")
    for _, module in projections:
        module.forward = MethodType(_canonical_lora_forward, module)
    return tuple(name for name, _ in projections)


def _check_fused_layout(owner: _FusedFirstProjection) -> None:
    if not isinstance(owner.fused, nn.Linear):
        raise TypeError("canonical fusion requires an ordinary merged nn.Linear owner")
    if sum(owner.output_sizes) != owner.fused.out_features:
        raise ValueError("fused output sizes do not match the merged projection")
    offset = 0
    for size in owner.output_sizes:
        if size <= 0 or offset % 128:
            raise ValueError("each fused projection must begin on the same N128 boundary")
        offset += size


def install_canonical_merged_(
    causal_lm: nn.Module, *, include_lm_head: bool = False,
) -> tuple[str, ...]:
    """Override merged nn.Linear forwards, including QKV/gate-up owner.fused.

    Reject live LoRA models rather than silently canonicalizing their base branch.
    The vocabulary head is unchanged by default, matching LoRA-only installation
    on the actor. If requested, install its actor counterpart separately as well.
    Gaussian/transition modules are outside the supplied causal_lm and untouched.
    """
    modules = list(causal_lm.named_modules())
    if any(isinstance(module, LoRALinear) for _, module in modules):
        raise ValueError("inference canonical installation requires merged projections")
    for _, module in modules:
        if isinstance(module, _FusedFirstProjection):
            _check_fused_layout(module)
    projections = [
        (name, module) for name, module in modules
        if isinstance(module, nn.Linear)
        and (include_lm_head or name.rsplit(".", 1)[-1] != "lm_head")
    ]
    for name, module in projections:
        if module.bias is not None:
            raise ValueError(f"canonical merged projection requires no bias: {name}")
        if module.weight.dtype != torch.bfloat16 or not module.weight.is_contiguous():
            raise ValueError(f"canonical merged projection requires contiguous bf16: {name}")
    for _, module in projections:
        module.forward = MethodType(_canonical_merged_forward, module)
    return tuple(name for name, _ in projections)


def _destination_weight(module: nn.Module) -> Tensor:
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            raise ValueError("canonical synchronization requires bias-free projections")
        return module.weight
    if isinstance(module, _FusedFirstProjection):
        owner, index = module, 0
    elif isinstance(module, _FusedProjectionSlice):
        owner, index = module._owner, module.index
    else:
        raise TypeError("replica projection is neither a merged Linear nor a fused slice")
    _check_fused_layout(owner)
    return owner.fused.weight.narrow(
        0, sum(owner.output_sizes[:index]), owner.output_sizes[index],
    )


@torch.no_grad()
def synchronize_canonical_lora_(destination: nn.Module, source: nn.Module) -> int:
    """Refresh ordinary OR fused causal-LM projections from a live LoRA actor.

    Pass ``replica.causal_lm, actor.causal_lm``. Frozen non-LoRA weights are assumed
    already cloned from the actor. This does not synchronize latent heads, caches,
    or policy state; retain the caller's separate head/state synchronization.
    Run after building a replica (the production builder's initial merge differs)
    and after every actor update, before any cached/full comparison or capture.
    Destination Parameter/storage identities remain stable for existing consumers.
    """
    copies: list[tuple[Tensor, LoRALinear]] = []
    for name, module in source.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        if module.base.bias is not None:
            raise ValueError(f"canonical synchronization requires no bias: {name}")
        weight = _destination_weight(destination.get_submodule(name))
        if weight.shape != module.base.weight.shape:
            raise ValueError(f"replica projection dimensions differ: {name}")
        if weight.dtype != torch.bfloat16 or weight.device != module.base.weight.device:
            raise ValueError(f"replica projection must share actor bf16 dtype/device: {name}")
        if not weight.is_contiguous():
            raise ValueError(f"replica projection weight slice is not contiguous: {name}")
        copies.append((weight, module))
    for weight, module in copies:
        weight.copy_(canonical_merged_weight(module))
    return len(copies)
