import pytest
import torch

from scripts.analyze_minicpm_controller_geometry import paired_variance_sums


def test_pair_variance_matches_centered_samples_not_iid_trajectories():
    left = torch.tensor([[1, 2], [3, -1], [-2, 5], [7, 0]], dtype=torch.float64)
    right = left.flip(0) + torch.tensor([4, -3], dtype=torch.float64)
    pairs = left.size(0)
    actual = paired_variance_sums(
        left.sum(0), right.sum(0),
        (left.square().sum() + right.square().sum()).item(),
        (left + right).square().sum().item(), pairs,
    )
    paired = pairs * (left + right).var(dim=0, correction=1).sum().item()
    independent = pairs * (
        left.var(dim=0, correction=1) + right.var(dim=0, correction=1)
    ).sum().item()
    covariance = (
        (left - left.mean(0)) * (right - right.mean(0))
    ).sum().item() / (pairs - 1)
    assert paired != pytest.approx(independent)
    assert actual['paired_sum_variance_trace'] == pytest.approx(paired)
    assert actual['independent_marginal_sum_variance_trace'] == pytest.approx(independent)
    assert actual['within_pair_cross_covariance_trace'] == pytest.approx(covariance)
