from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("flash_attn.cute")

from fresh_lejepa_train_v1_probe_shared_rms_pope_zero_fa4 import (
    ZeroPhaseFA4PolarCausalSelfAttention,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set RUN_CUDA_TESTS=1 in the FA4 environment",
)


def _attention() -> ZeroPhaseFA4PolarCausalSelfAttention:
    old_mode = ZeroPhaseFA4PolarCausalSelfAttention.position_mode
    ZeroPhaseFA4PolarCausalSelfAttention.position_mode = "pope"
    try:
        return ZeroPhaseFA4PolarCausalSelfAttention(
            512, 8, 4, 10_000.0, 1.5
        ).cuda().bfloat16()
    finally:
        ZeroPhaseFA4PolarCausalSelfAttention.position_mode = old_mode


def test_fa4_qk128_v64_gqa_forward_and_backward_match_sdpa() -> None:
    torch.manual_seed(37)
    attention = _attention()
    q_shape = (1, 8, 128, 64)
    kv_shape = (1, 4, 128, 64)
    inputs = [
        torch.randn(q_shape, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(q_shape, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16, requires_grad=True),
    ]
    q_real, q_imag, k_real, k_imag, value = inputs
    actual = attention._complex_attention(
        q_real, q_imag, k_real, k_imag, value, is_causal=True
    )
    query = torch.cat((q_real, q_imag), dim=-1)
    key = torch.cat((k_real, k_imag), dim=-1)
    padded_value = torch.cat((value, torch.zeros_like(value)), dim=-1)
    expected = F.scaled_dot_product_attention(
        query,
        key,
        padded_value,
        is_causal=True,
        enable_gqa=True,
        scale=64**-0.5,
    )[..., :64]
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)

    output_grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, inputs, output_grad, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, inputs, output_grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=8e-2, atol=8e-2)


def test_fa4_runs_inside_compiled_attention_with_backward() -> None:
    torch.manual_seed(41)
    attention = _attention()
    compiled = torch.compile(attention, dynamic=False, fullgraph=False)
    x = torch.randn(1, 128, 512, device="cuda", dtype=torch.bfloat16)
    compiled(x).float().square().mean().backward()
    assert attention.delta_c.grad is not None
    assert torch.isfinite(attention.delta_c.grad).all()
