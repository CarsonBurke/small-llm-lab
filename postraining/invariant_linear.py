"""Width-invariant bf16 projections for the opt-in Uno inference target.

Tile geometry and reduction order are part of the arithmetic contract. A token
always occupies the same row of a 64-lane tile, regardless of suffix width. The
backbone remains bf16; dot products accumulate in fp32 without split-K or bf16
partial-sum truncation. Prefill deliberately retains the shared cuBLAS path.
"""

from __future__ import annotations
from typing import Any, cast
from collections.abc import Callable
from functools import wraps

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import triton
import triton.language as tl


INVARIANT_ARITHMETIC = "bf16-lane64-n128-k32-casts/v1"
LEGACY_ARITHMETIC = "bf16-cublas-inductor-default/v1"
OPTIMIZED_ARITHMETIC = "bf16-cublas-fa4-fullgraph-casts/v1"


def compile_invariant(function: Callable[..., Any]) -> Callable[..., Any]:
    """Compile the numerical target; never silently fall back to different arithmetic."""
    compiled = torch.compile(
        function, fullgraph=True, options={"emulate_precision_casts": True}
    )

    @wraps(function)
    def run(*args, **kwargs):
        with torch.compiler.config.patch(fail_on_recompile_limit_hit=True):
            return compiled(*args, **kwargs)

    return run


@triton.jit
def _lane_linear_kernel(
    inputs,
    weight,
    output,
    BATCH: tl.constexpr,
    WIDTH: tl.constexpr,
    INPUT: tl.constexpr,
    OUTPUT: tl.constexpr,
    STRIDE_BATCH: tl.constexpr,
    STRIDE_TOKEN: tl.constexpr,
    STRIDE_INPUT: tl.constexpr,
):
    tile = tl.program_id(0)
    lanes = (tile // WIDTH) * 64 + tl.arange(0, 64)
    token = tile % WIDTH
    columns = tl.program_id(1) * 128 + tl.arange(0, 128)
    reduction = tl.arange(0, 32)
    accumulator = tl.full((64, 128), 0, tl.float32)
    for start in range(tl.cdiv(INPUT, 32)):
        indices = start * 32 + reduction
        left = tl.load(
            inputs
            + lanes[:, None] * STRIDE_BATCH
            + token * STRIDE_TOKEN
            + indices[None, :] * STRIDE_INPUT,
            (lanes[:, None] < BATCH) & (indices[None, :] < INPUT),
            other=0,
        )
        right = tl.load(
            weight + columns[None, :] * INPUT + indices[:, None],
            (columns[None, :] < OUTPUT) & (indices[:, None] < INPUT),
            other=0,
        )
        accumulator = tl.dot(left, right, accumulator)
    tl.store(
        output + (lanes[:, None] * WIDTH + token) * OUTPUT + columns[None, :],
        accumulator,
        (lanes[:, None] < BATCH) & (columns[None, :] < OUTPUT),
    )


class InvariantLinear(nn.Linear):
    """A merged projection with explicit prefill/decode dispatch, sharing weights."""

    decoding: bool = False

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> InvariantLinear:
        if linear.bias is not None:
            raise ValueError("invariant rollout requires bias-free projections")
        if linear.weight.dtype != torch.bfloat16 or not linear.weight.is_contiguous():
            raise ValueError("invariant rollout requires contiguous bf16 weights")
        result = cls(
            linear.in_features,
            linear.out_features,
            bias=False,
            device="meta",
            dtype=linear.weight.dtype,
        )
        result.weight = linear.weight
        result.train(linear.training)
        return result

    def forward(self, input: Tensor) -> Tensor:
        inputs = input
        if not self.decoding:
            return F.linear(inputs, self.weight)
        if (
            inputs.ndim not in (2, 3)
            or inputs.dtype != torch.bfloat16
            or not inputs.is_cuda
        ):
            raise ValueError(
                "invariant decode requires CUDA bf16 [lanes, width?, hidden]"
            )
        # The actual AR engine projects a 2D last-hidden tensor through lm_head.
        # Treat it as width one, exactly like the block verifier's 3D head input.
        rows = inputs[:, None, :] if inputs.ndim == 2 else inputs
        batch, width, hidden = rows.shape
        if hidden != self.in_features or inputs.device != self.weight.device:
            raise ValueError("invariant projection input dimensions or device differ")
        output = torch.empty(
            (batch, width, self.out_features), dtype=inputs.dtype, device=inputs.device
        )
        cast(Any, _lane_linear_kernel)[
            (triton.cdiv(batch, 64) * width, triton.cdiv(self.out_features, 128))
        ](
            rows,
            self.weight,
            output,
            batch,
            width,
            hidden,
            self.out_features,
            *rows.stride(),
            num_warps=4,
            num_stages=3,
        )
        return output[:, 0] if inputs.ndim == 2 else output


def install_invariant_linears(causal_lm: nn.Module) -> tuple[InvariantLinear, ...]:
    """Retain merged projection identities and actor synchronization semantics."""
    replacements = [
        (name, module)
        for name, module in causal_lm.named_modules()
        if isinstance(module, nn.Linear)
    ]
    installed = []
    for name, module in replacements:
        parent_name, _, child_name = name.rpartition(".")
        parent = causal_lm.get_submodule(parent_name) if parent_name else causal_lm
        replacement = InvariantLinear.from_linear(module)
        setattr(parent, child_name, replacement)
        installed.append(replacement)
    return tuple(installed)
