"""Normalized binary likelihoods; GPU execution must use mlq."""

import math

import pytest
import torch

from pretraining.nanogpt_mini.bit_density import binary_nll

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def density():
    return torch.compile(binary_nll, dynamic=True, fullgraph=True)


@pytest.mark.parametrize("components", [1, 8])
def test_probability_normalizes_over_all_codes_without_hiding_reserved_mass(
    density, components
):
    bits = 3
    codes = (
        (torch.arange(8, device="cuda")[:, None] >> torch.arange(3, device="cuda")) & 1
    ).float()
    width = bits if components == 1 else components * (bits + 1)
    parameters = torch.linspace(-2, 3, width, device="cuda").expand(8, -1)
    probability = (-density(parameters, codes, components)).exp()
    torch.testing.assert_close(probability.sum(), torch.ones((), device="cuda"))
    # A 5-character alphabet cannot silently renormalize away the other 3 codes.
    assert 0 < float(probability[:5].sum()) < 1
    torch.testing.assert_close(
        probability[:5].sum(), 1 - probability[5:].sum(), rtol=1e-5, atol=1e-6
    )


def test_correlated_modes_pay_mixture_choice_not_independent_bit_cost(density):
    bits = 128
    codes = torch.tensor([[0.0] * bits, [1.0] * bits], device="cuda")
    component = torch.zeros((2, bits + 1), device="cuda")
    component[0, 1:] = -15
    component[1, 1:] = 15
    parameters = component.flatten().expand(2, -1)
    joint_bits = density(parameters, codes, 2) / math.log(2)
    independent_bits = density(torch.zeros_like(codes), codes, 1) / math.log(2)
    torch.testing.assert_close(
        joint_bits, torch.ones_like(joint_bits), atol=1e-3, rtol=0
    )
    torch.testing.assert_close(independent_bits, torch.full_like(independent_bits, 128))
    # Unequal mode weights must be charged, rather than choosing the best mode free.
    component[:, 0] = torch.tensor([math.log(0.25), math.log(0.75)], device="cuda")
    charged = density(component.flatten().expand(2, -1), codes, 2) / math.log(2)
    expected = torch.tensor([2.0, -math.log2(0.75)], device="cuda")
    torch.testing.assert_close(charged, expected, atol=1e-3, rtol=0)


def test_joint_density_trains_both_probabilities_and_current_binary_targets(density):
    parameters = torch.tensor(
        [[0.2, -1.0, 2.0, -0.4, 3.0, -2.0]], device="cuda", requires_grad=True
    )
    targets = torch.tensor([[1.0, 0.0]], device="cuda", requires_grad=True)
    nll = density(parameters, targets, 2).sum()
    param_grad, target_grad = torch.autograd.grad(nll, (parameters, targets))
    assert bool(torch.isfinite(param_grad).all())
    assert bool(torch.isfinite(target_grad).all())
    assert bool((param_grad.reshape(2, 3)[:, 0] != 0).all())
    assert bool((param_grad.reshape(2, 3)[:, 1:] != 0).all())
    assert bool((target_grad != 0).all())
