"""CUDA/BF16 contracts for LayerScale on the latent-carry residual branch."""

import pytest
import torch
import torch.nn.functional as F

from pretraining.nanogpt_mini.nanogpt_mini_full_bandwidth_model import FullBandwidthGPT

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def model():
    torch.manual_seed(29)
    return (
        FullBandwidthGPT(
            fusion="layerscale", layerscale_init=0.1, detach_carry=True, noise=0.0
        )
        .cuda()
        .eval()
    )


def test_layerscale_is_diagonal_residual_not_sigmoid_or_normalization(model):
    inputs = torch.randint(0, 1024, (2, 6), device="cuda")
    previous = torch.randn(2, 6, 512, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        model.carry_scale.copy_(torch.linspace(-0.2, 0.3, 512, device="cuda"))
    forward = torch.compile(model.fused_input, fullgraph=True, dynamic=False)
    plain_forward = torch.compile(model.fused_input, fullgraph=True, dynamic=False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        plain = plain_forward(inputs)
        residual = F.linear(previous[:, :-1], model.fuse_value.weight.bfloat16())
        residual = residual + F.linear(plain[:, 1:], model.fuse_gate.weight.bfloat16())
        expected = plain[:, 1:] + (residual.float() * model.carry_scale).bfloat16()
        actual = forward(inputs, previous)
    torch.testing.assert_close(actual[:, :1], plain[:, :1], rtol=0, atol=0)
    torch.testing.assert_close(actual[:, 1:], expected, rtol=0.01, atol=0.015)


def test_zero_layerscale_preserves_plain_input_and_can_learn_to_open(model):
    inputs = torch.randint(0, 1024, (2, 6), device="cuda")
    previous = torch.randn(
        2, 6, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    with torch.no_grad():
        model.carry_scale.zero_()
    forward = torch.compile(model.fused_input, fullgraph=True, dynamic=False)
    plain_forward = torch.compile(model.fused_input, fullgraph=True, dynamic=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = forward(inputs, previous)
        plain = plain_forward(inputs)
        loss = (actual[:, 1:].float() * torch.randn_like(actual[:, 1:])).sum()
    carry_grad, scale_grad = torch.autograd.grad(
        loss, (previous, model.carry_scale), allow_unused=True
    )
    torch.testing.assert_close(actual, plain, rtol=0, atol=0)
    assert carry_grad is None or torch.count_nonzero(carry_grad) == 0
    assert torch.isfinite(scale_grad).all() and scale_grad.norm() > 0


def test_recurrent_document_resets_use_plain_inputs(model):
    tokens = torch.randint(0, 1024, (3,), device="cuda")
    previous = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
    resets = torch.tensor([True, False, True], device="cuda")
    forward = torch.compile(model.recurrent_input, fullgraph=True, dynamic=False)
    plain_forward = torch.compile(model.fused_input, fullgraph=True, dynamic=False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        plain = plain_forward(tokens[:, None])
        actual = forward(tokens, previous, resets)
        changed = forward(tokens, previous + 10, resets)
    torch.testing.assert_close(actual[resets], plain[resets], rtol=0, atol=0)
    torch.testing.assert_close(changed[resets], actual[resets], rtol=0, atol=0)
    assert not torch.equal(changed[~resets], actual[~resets])
