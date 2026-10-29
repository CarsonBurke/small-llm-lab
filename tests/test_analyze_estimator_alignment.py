import math

import pytest
import torch

from scripts.analyze_estimator_alignment import analyze, terms

POOLS = list(range(8))
GROUPS = {"m": "muon_matrices", "b": "vectors"}


def _synthetic(seed=0, baseline_noise=0.2, critic_noise=0.05, scale_truth=1.0):
    """Two unbiased estimators and one biased at 60 degrees with the truth's norm."""
    generator = torch.Generator().manual_seed(seed)
    truth = {"m": scale_truth * torch.randn(48, 64, generator=generator),
             "b": scale_truth * torch.randn(200, generator=generator)}
    other = {k: torch.randn(v.shape, generator=generator) for k, v in truth.items()}
    flat_t = torch.cat([v.flatten() for v in truth.values()])
    flat_o = torch.cat([v.flatten() for v in other.values()])
    other = {k: v - (flat_o @ flat_t) / (flat_t @ flat_t) * truth[k] for k, v in other.items()}
    angle = math.radians(60)
    scale = flat_t.norm() / torch.cat([v.flatten() for v in other.values()]).norm()
    biased = {k: math.cos(angle) * truth[k] + math.sin(angle) * scale * other[k] for k in truth}
    gradients = {}
    for pool in POOLS:
        for name, noise in (("a/sampled_baseline", baseline_noise), ("a/control_variate", baseline_noise / 2)):
            gradients[(name, pool)] = {
                k: v + noise * torch.randn(v.shape, generator=generator) for k, v in truth.items()}
        gradients[("a/all_action", pool)] = {
            k: v + critic_noise * torch.randn(v.shape, generator=generator) for k, v in biased.items()}
    return gradients, truth, angle


def test_terms_parses_a_difference_and_rejects_a_bare_name():
    assert terms("v7/all_action-v7/all_action_nogain") == ["v7/all_action", "v7/all_action_nogain"]
    with pytest.raises(ValueError):
        terms("all_action")


def test_alignment_recovers_the_bias_angle_and_the_unbiased_references():
    gradients, truth, angle = _synthetic()
    series = ["a/sampled_baseline", "a/control_variate", "a/all_action"]
    stats = analyze(gradients, series, POOLS, GROUPS, {"m"})["groups"]["all"]
    true_sq = float(sum(v.square().sum() for v in truth.values()))
    assert stats["_"]["grad_J_sq"] == pytest.approx(true_sq, rel=0.02)
    assert stats["a/control_variate~a/sampled_baseline"]["disagreement"] == pytest.approx(0, abs=0.01)
    assert stats["a/sampled_baseline"]["cos_bias"] == pytest.approx(1.0, abs=0.01)
    assert stats["a/all_action"]["cos_bias"] == pytest.approx(math.cos(angle), abs=0.01)
    assert stats["a/all_action"]["cos_bias@a/control_variate"] == pytest.approx(math.cos(angle), abs=0.01)
    # Noise lowers one update's alignment, and averaging recovers it.
    unbiased = stats["a/sampled_baseline"]
    assert unbiased["cos_step"] < unbiased["cos_avg4"] < unbiased["cos_avg20"] <= unbiased["cos_bias"] + 0.01


def test_noise_power_sets_the_one_update_cosine():
    gradients, truth, _ = _synthetic(baseline_noise=2.0)
    result = analyze(gradients, ["a/sampled_baseline"], POOLS, GROUPS, set())
    true_sq = float(sum(v.square().sum() for v in truth.values()))
    numel = sum(v.numel() for v in truth.values())
    expected = math.sqrt(true_sq / (true_sq + numel * 2.0 ** 2))
    assert result["groups"]["all"]["a/sampled_baseline"]["cos_step"] == pytest.approx(expected, rel=0.05)


def test_a_difference_series_and_a_paired_comparison():
    gradients, _, _ = _synthetic()
    series = ["a/sampled_baseline", "a/all_action", "a/all_action-a/sampled_baseline"]
    result = analyze(gradients, series, POOLS, GROUPS, {"m"},
                     compare=[("a/all_action", "a/sampled_baseline")])
    stats = result["groups"]["all"]
    # biased - truth sits at 120 degrees to truth for a 60 degree, equal-norm bias.
    assert stats["a/all_action-a/sampled_baseline"]["cos_bias"] == pytest.approx(-0.5, abs=0.02)
    paired = stats["a/all_action - a/sampled_baseline"]
    assert paired["cos_bias"] == pytest.approx(math.cos(math.radians(60)) - 1, abs=0.02)
    assert math.isfinite(paired["cos_bias_se"])
    # A nonlinear step of a difference is not the difference of steps.
    assert "muon_gain_step" not in result["groups"]["muon_matrices"].get("a/all_action-a/sampled_baseline", {})


def test_optimizer_gains_of_a_precise_estimator():
    gradients, truth, _ = _synthetic(baseline_noise=0.01)
    groups = analyze(gradients, ["a/sampled_baseline"], POOLS, GROUPS, {"m"})["groups"]
    entry = groups["muon_matrices"]["a/sampled_baseline"]
    nuclear = float(torch.linalg.matrix_norm(truth["m"], ord="nuc"))
    # Polar Express in bf16 stops short of the exact polar factor.
    assert entry["muon_gain_avg"] == pytest.approx(nuclear, rel=0.1)
    assert entry["muon_gain_step"] == pytest.approx(nuclear, rel=0.1)
    vectors = groups["vectors"]["a/sampled_baseline"]
    assert vectors["sign_gain"] == pytest.approx(float(truth["b"].abs().sum()), rel=0.02)
    # At high SNR g / sqrt(v) is sign(g).
    assert vectors["adam_gain"] == pytest.approx(vectors["sign_gain"], rel=0.02)


def test_pure_noise_reports_nan_rather_than_failing():
    gradients, _, _ = _synthetic(scale_truth=0.0)
    stats = analyze(gradients, ["a/sampled_baseline", "a/all_action"], POOLS, GROUPS, {"m"})["groups"]["all"]
    assert "cos_bias" in stats["a/all_action"]
