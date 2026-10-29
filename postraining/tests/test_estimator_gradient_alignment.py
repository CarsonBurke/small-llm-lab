"""The DG head-gate alignment tooling on CPU: the alignment script's argv
contract, per-pool shuffle seeds and entropy-bonus gradient, and the
analyzer's entropy projections. No model runs on a GPU."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from postraining.latent_rollout import (
    build_replay_plan, compact_slots, replay_head_inputs, slot_index,
)
from postraining.tests.test_mac_update import _actor_parameters, _cot_batch, _cot_wrapper

REPO = Path(__file__).resolve().parents[2]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


alignment = _script("estimator_gradient_alignment")
analyzer = _script("analyze_estimator_alignment")

ARGV = ["--checkpoint", "c.pt", "--critic", "k.pt", "--rollouts", "r", "--pools", "0", "1",
        "--label", "x", "--output", "o"]


def test_the_shuffle_seed_comes_with_and_only_with_the_shuffled_gate(capsys):
    args = alignment.parse_args([*ARGV, "--estimators", "sampled_baseline",
                                 "delightful_head_gate", "delightful_shuffled_head_gate",
                                 "--gate-shuffle-seed", "20261021", "--entropy-gradient"])
    assert args.gate_shuffle_seed == 20261021 and args.entropy_gradient
    for extra in (["--estimators", "delightful_shuffled_head_gate"],
                  ["--estimators", "delightful_head_gate", "--gate-shuffle-seed", "1"],
                  ["--estimators", "sampled_baseline", "sampled_baseline"]):
        with pytest.raises(SystemExit):
            alignment.parse_args([*ARGV, *extra])
    with pytest.raises(SystemExit):
        alignment.parse_args([*ARGV[:-6], "70000", "--label", "x", "--output", "o",
                              "--estimators", "delightful_shuffled_head_gate",
                              "--gate-shuffle-seed", "1"])
    assert "65536" in capsys.readouterr().err


def test_pool_shuffle_seeds_are_distinct_per_seed_and_pool():
    seeds = {alignment.pool_shuffle_seed(seed, pool) for seed in range(4) for pool in range(64)}
    assert len(seeds) == 4 * 64
    with pytest.raises(ValueError):
        alignment.pool_shuffle_seed(1, 1 << 16)


def test_the_entropy_bonus_is_minus_the_mean_emit_entropys_gradient():
    """Against autograd through the renderer's full softmax at every emit
    state, over two replay shards and readout chunks of 3."""
    wrapper = _cot_wrapper()
    batch = _cot_batch(wrapper)
    parameters = _actor_parameters(wrapper)
    beliefs, inputs = replay_head_inputs(wrapper, batch)
    emit_index = slot_index(batch.emit_mask.bool())
    logits = wrapper.backbone.logits_from_features(wrapper.renderer_features(
        compact_slots(inputs, emit_index), compact_slots(beliefs, emit_index))).double()
    log_probabilities = logits.log_softmax(-1)
    entropy = -(log_probabilities.exp() * log_probabilities).sum(-1)
    expected = torch.autograd.grad(-entropy.mean(), parameters, allow_unused=True)
    wrapper.zero_grad(set_to_none=True)
    plan = build_replay_plan(batch, 2, 4 * 1024 * 1024)
    assert len(list(plan.shards())) == 2
    alignment.entropy_bonus_gradient(wrapper, batch, plan, 3)
    assert any(want is not None and bool(want.abs().sum()) for want in expected)
    for parameter, want in zip(parameters, expected, strict=True):
        got = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad
        want = torch.zeros_like(parameter) if want is None else want
        torch.testing.assert_close(got.double(), want.double(), rtol=1e-4,
                                   atol=1e-6 * max(1.0, float(want.abs().max())))


def test_the_analyzer_projects_every_series_on_the_entropy_bonus():
    """Noise-free pools: every cross-pool product is exact, so the entropy
    gain is E g . E g_H over all parameters, and Muon's is the polar step of
    E g scored on E g_H over the matrices."""
    generator = torch.Generator().manual_seed(4)
    means = {name: {"m": torch.randn(4, 3, generator=generator),
                    "v": torch.randn(5, generator=generator)}
             for name in ("x/sampled_baseline", "x/delightful_head_gate", "x/entropy_bonus")}
    pools = [0, 1, 2]
    gradients = {(name, pool): mean for name, mean in means.items() for pool in pools}
    series = list(means)
    muon = {"m"}
    result = analyzer.analyze(
        gradients, series, pools, {n: analyzer.group_of(n, muon) for n in ("m", "v")}, muon,
        compare=[("x/delightful_head_gate", "x/sampled_baseline")], entropy="x/entropy_bonus")
    assert result["entropy"] == "x/entropy_bonus"
    entropy = means["x/entropy_bonus"]
    for name in ("x/sampled_baseline", "x/delightful_head_gate"):
        mean = means[name]
        inner = float((mean["m"] * entropy["m"]).sum() + (mean["v"] * entropy["v"]).sum())
        stats = result["groups"]["all"][name]
        assert stats["entropy_gain"] == pytest.approx(inner, rel=1e-6)
        norms = (sum(float(t.square().sum()) for t in mean.values())
                 * sum(float(t.square().sum()) for t in entropy.values())) ** 0.5
        assert stats["entropy_cos"] == pytest.approx(inner / norms, rel=1e-6)
        step = float((analyzer.polar(mean["m"]) * entropy["m"]).sum())
        muon_stats = result["groups"]["muon_matrices"][name]
        assert muon_stats["muon_entropy_step"] == pytest.approx(step, rel=1e-5)
        assert muon_stats["muon_entropy_avg"] == pytest.approx(step, rel=1e-5)
    difference = result["groups"]["muon_matrices"]["x/delightful_head_gate - x/sampled_baseline"]
    assert difference["muon_entropy_step"] == pytest.approx(
        result["groups"]["muon_matrices"]["x/delightful_head_gate"]["muon_entropy_step"]
        - result["groups"]["muon_matrices"]["x/sampled_baseline"]["muon_entropy_step"])
    # The entropy series is not projected on itself.
    assert "entropy_gain" not in result["groups"]["all"]["x/entropy_bonus"]
    with pytest.raises(ValueError, match="entropy series"):
        analyzer.analyze(gradients, series[:2], pools,
                         {n: analyzer.group_of(n, muon) for n in ("m", "v")}, muon,
                         entropy="x/entropy_bonus")
