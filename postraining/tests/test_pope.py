from __future__ import annotations

import math
import os

import pytest
import torch
import torch.nn.functional as F

from fresh_lejepa_train_v1_probe_shared_rms_pope import PolarCausalSelfAttention
from fresh_lejepa_train_v1_probe_shared_rms_pope_zero import (
    ZeroPhasePolarCausalSelfAttention,
)
from fresh_lejepa_train_v1_probe_shared_rms_pope_zero_flash import (
    ZeroPhaseFlashPolarCausalSelfAttention,
)


def _attention() -> PolarCausalSelfAttention:
    old_mode = PolarCausalSelfAttention.position_mode
    PolarCausalSelfAttention.position_mode = "pope"
    try:
        return PolarCausalSelfAttention(32, 4, 2, 10_000.0, 1.5)
    finally:
        PolarCausalSelfAttention.position_mode = old_mode


def test_gqa_query_phase_shift_matches_reference_key_shift() -> None:
    torch.manual_seed(7)
    attention = _attention()
    batch, heads, kv_heads, length, dim = 2, 4, 2, 5, 8
    q = torch.randn(batch, heads, length, dim)
    k = torch.randn(batch, kv_heads, length, dim)
    positions = torch.arange(length)
    q_real, q_imag, k_real, k_imag = attention._polar_components(q, k, positions)
    actual = q_real @ k_real.repeat_interleave(2, 1).transpose(-2, -1)
    actual += q_imag @ k_imag.repeat_interleave(2, 1).transpose(-2, -1)

    theta = torch.outer(positions.float(), attention.polar_inv_freq)[None, None]
    delta = attention.delta_c.clamp(-2 * math.pi, 0)
    q_mag = F.softplus(q)
    k_mag = F.softplus(k).repeat_interleave(2, 1)
    reference_q_real = q_mag * theta.cos()
    reference_q_imag = q_mag * theta.sin()
    shifted = theta + delta
    reference_k_real = k_mag * shifted.cos()
    reference_k_imag = k_mag * shifted.sin()
    gain = attention.q_gain[None, :, None, None]
    expected = (reference_q_real * gain) @ reference_k_real.transpose(-2, -1)
    expected += (reference_q_imag * gain) @ reference_k_imag.transpose(-2, -1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_full_sequence_matches_incremental_cache() -> None:
    torch.manual_seed(11)
    attention = _attention().eval()
    x = torch.randn(2, 7, 32)
    full = attention(x)
    shape = (2, attention.num_kv_heads, x.size(1), attention.head_dim)
    cache = tuple(torch.empty(shape) for _ in range(3))
    cached = []
    for position in range(x.size(1)):
        output, cache = attention.forward_step(
            x[:, position : position + 1], cache, position
        )
        cached.append(output)
    torch.testing.assert_close(
        torch.cat(cached, dim=1), full, rtol=2e-5, atol=2e-5
    )


def test_theta_bias_receives_gradient() -> None:
    attention = _attention()
    attention(torch.randn(2, 6, 32)).square().mean().backward()
    assert attention.delta_c.grad is not None
    assert torch.isfinite(attention.delta_c.grad).all()
    assert attention.delta_c.grad.abs().sum() > 0


def test_zero_phase_variant_initializes_exactly_at_zero() -> None:
    old_mode = ZeroPhasePolarCausalSelfAttention.position_mode
    ZeroPhasePolarCausalSelfAttention.position_mode = "pope"
    try:
        attention = ZeroPhasePolarCausalSelfAttention(32, 4, 2, 10_000.0, 1.5)
    finally:
        ZeroPhasePolarCausalSelfAttention.position_mode = old_mode
    torch.testing.assert_close(attention.delta_c, torch.zeros_like(attention.delta_c))


@pytest.mark.skipif(
    os.environ.get("RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set RUN_CUDA_TESTS=1 on CUDA host",
)
def test_reference_flash_gqa_forward_and_backward_match_sdpa_oracle() -> None:
    torch.manual_seed(23)
    old_mode = ZeroPhaseFlashPolarCausalSelfAttention.position_mode
    ZeroPhaseFlashPolarCausalSelfAttention.position_mode = "pope"
    try:
        attention = ZeroPhaseFlashPolarCausalSelfAttention(
            256, 4, 2, 10_000.0, 1.5
        ).cuda().bfloat16()
    finally:
        ZeroPhaseFlashPolarCausalSelfAttention.position_mode = old_mode

    shape_q = (1, 4, 128, 64)
    shape_kv = (1, 2, 128, 64)
    inputs = [
        torch.randn(shape_q, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(shape_q, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16, requires_grad=True),
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


@pytest.mark.skipif(
    os.environ.get("RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set RUN_CUDA_TESTS=1 on CUDA host",
)
def test_reference_flash_gqa_runs_inside_compiled_model_with_backward() -> None:
    torch.manual_seed(29)
    old_mode = ZeroPhaseFlashPolarCausalSelfAttention.position_mode
    ZeroPhaseFlashPolarCausalSelfAttention.position_mode = "pope"
    try:
        attention = ZeroPhaseFlashPolarCausalSelfAttention(
            256, 4, 2, 10_000.0, 1.5
        ).cuda().bfloat16()
    finally:
        ZeroPhaseFlashPolarCausalSelfAttention.position_mode = old_mode
    # The reference Triton autograd function is an intentional opaque kernel;
    # JEPA compiles modules with fullgraph=False for the same reason as SIGReg.
    compiled = torch.compile(attention, dynamic=False, fullgraph=False)
    x = torch.randn(1, 128, 256, device="cuda", dtype=torch.bfloat16)
    compiled(x).float().square().mean().backward()
    assert attention.delta_c.grad is not None
    assert torch.isfinite(attention.delta_c.grad).all()


@pytest.mark.skipif(
    os.environ.get("RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set RUN_CUDA_TESTS=1 on CUDA host",
)
def test_cuda_compile_full_and_cached() -> None:
    torch.manual_seed(19)
    attention = _attention().cuda().bfloat16().eval()
    compiled_full = torch.compile(attention, dynamic=False, fullgraph=True)
    x = torch.randn(2, 64, 32, device="cuda", dtype=torch.bfloat16)
    full = compiled_full(x)
    shape = (2, attention.num_kv_heads, x.size(1), attention.head_dim)
    cache = tuple(
        torch.empty(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    )
    for position in range(x.size(1) - 1):
        _, cache = attention.forward_step(
            x[:, position : position + 1], cache, position
        )

    # Rollout graphs are shape-specialized by cache-prefix length.  Compile a
    # representative cached step rather than asking Dynamo to specialize on a
    # data-dependent tensor length inside one graph.
    def final_step(token, state):
        return attention.forward_step(token, state, x.size(1) - 1)

    compiled_step = torch.compile(final_step, dynamic=False, fullgraph=True)
    cached_last, _ = compiled_step(x[:, -1:], cache)
    torch.testing.assert_close(
        cached_last, full[:, -1:], rtol=4e-2, atol=4e-2
    )
