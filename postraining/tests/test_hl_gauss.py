from __future__ import annotations

import torch

from postraining.hl_gauss import HLGaussSupport


def _support(num_bins: int = 101) -> HLGaussSupport:
    return HLGaussSupport(num_bins, 0.0, 1.0, sigma_ratio=2.0)


def test_projection_is_a_distribution_and_decodes_back_to_the_target():
    support = _support()
    targets = torch.tensor([0.0, 0.1234, 0.5, 0.87, 1.0])
    probs = support.project(targets)
    torch.testing.assert_close(probs.sum(-1), torch.ones(5))
    assert float(probs.min()) >= 0.0
    # Interior targets round-trip through the expected-scalar decode.  Exact
    # edge targets lose half their Gaussian to truncation and decode inward
    # by sigma * sqrt(2/pi) ~ 1.6 bins — inherent to HL-Gauss at sigma_ratio
    # 2.0, so a reward of exactly 0 reads as ~0.016.
    decoded = (probs * support.centers).sum(-1)
    torch.testing.assert_close(decoded[1:4], targets[1:4], atol=1e-4, rtol=0)
    inward_bias = support.sigma * (2.0 / torch.pi) ** 0.5
    assert abs(float(decoded[0]) - inward_bias) < 0.1 * support.bin_width
    assert abs(float(1.0 - decoded[-1]) - inward_bias) < 0.1 * support.bin_width


def test_out_of_range_targets_clamp_to_the_support():
    support = _support()
    probs = support.project(torch.tensor([-3.0, 7.0]))
    torch.testing.assert_close(probs, support.project(torch.tensor([0.0, 1.0])))


def test_cross_entropy_is_minimized_at_the_target():
    support = _support(num_bins=21)
    target = torch.tensor([0.6])
    # Logits peaked at the target's projection beat logits peaked elsewhere.
    at_target = support.project_to_logprobs(target) * 20.0
    elsewhere = support.project_to_logprobs(torch.tensor([0.2])) * 20.0
    assert float(support.cross_entropy(at_target, target)) < float(
        support.cross_entropy(elsewhere, target)
    )


def test_expected_scalar_decode_of_the_projected_prior():
    support = _support()
    logits = support.project_to_logprobs(torch.tensor(0.0))
    # The zero prior sits on the support edge, so it decodes the truncation
    # bias inward (~1.6 bins), not exactly zero.
    assert 0.0 < float(support.to_expected_scalar(logits)) < 2.0 * support.bin_width
