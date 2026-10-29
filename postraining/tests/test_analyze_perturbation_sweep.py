"""The sweep analysis recovers a known directional derivative.

A toy policy with common random numbers: trajectory (cursor, prompt, sample)
has a fixed uniform threshold t, and its return is 1[t < p(theta)] with
p = 1/2 + c . (theta - theta0) / sqrt(n) on the reader-out target. Then
J = p exactly, dJ/d eps along d is c . d / sqrt(n), and an antithetic pair
flips exactly the trajectories whose t lies between p(-) and p(+).
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from postraining.perturbation_sweep import (
    PERTURBATION_SWEEP_SCHEMA,
    SweepConfig,
    SweepWriter,
    run_sweep,
)
from postraining.tests.test_perturbation_sweep import fake_group
from postraining.vapo.mixture import file_sha256

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "analyze_perturbation_sweep", REPO / "scripts" / "analyze_perturbation_sweep.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

NAME = "combiner.reader.out.weight"


def build_sweep(tmp_path: Path, token_noise: float, token_bias: float = 0.0, pools: int = 8,
                warmup: int = 2):
    generator = torch.Generator().manual_seed(0)
    theta0 = torch.randn(4, 8, generator=generator) * 0.01
    n = theta0.numel()
    slope = torch.randn(4, 8, generator=generator)  # dJ/dtheta * sqrt(n)
    snapshot = tmp_path / "snapshot.pt"
    torch.save({"model": {NAME: theta0.clone()}}, snapshot)
    parameters = {NAME: torch.nn.Parameter(theta0.clone())}
    sampler = SimpleNamespace(cursor=0)
    prompts, samples = 32, 16

    def collect_at(cursor):
        sampler.cursor = cursor + prompts
        p = 0.5 + float((slope * (parameters[NAME].detach() - theta0)).sum()) / math.sqrt(n)
        groups = []
        for prompt in range(prompts):
            thresholds = torch.rand(samples, generator=torch.Generator().manual_seed(cursor * 997 + prompt))
            wins = (thresholds < p).float()
            groups.append(fake_group([cursor + prompt], [[int(w)] for w in wins.tolist()], wins.tolist()))
        return groups

    noise = torch.Generator().manual_seed(1)

    def token_gradient(groups):
        g = slope / math.sqrt(n) * (1 + token_bias) + token_noise * torch.randn(4, 8, generator=noise)
        return {NAME: g.double()}, {"trajectories": prompts * samples}

    root = tmp_path / "run" / "perturbation_sweep"
    writer = SweepWriter(root, {
        "schema": PERTURBATION_SWEEP_SCHEMA, "policy_state": str(snapshot),
        "policy_state_sha256": file_sha256(snapshot), "seed": 5, "samples_per_prompt": samples,
        "config": {"targets": ["reader_out"]},
    })
    run_sweep(SweepConfig(("reader_out",), (0.5, 2.0), 3, True, True, 5, gradient_warmup_pools=warmup),
              {"reader_out": parameters},
              pools, cursor=lambda: sampler.cursor, collect_at=collect_at, token_gradient=token_gradient,
              writer=writer, out_of_time=lambda estimate: False, log=lambda values: None)
    return root, slope / math.sqrt(n), theta0


def test_analysis_recovers_the_true_derivative_and_token_agreement(tmp_path, capsys):
    root, true_gradient, theta0 = build_sweep(tmp_path, token_noise=0.002)
    report = analysis.analyze(root, verify_snapshot=True)
    assert report["null"] == {"pools": 6, "changed": 0, "abs_reward_shift": 0.0}
    target = report["targets"]["reader_out"]
    rms2 = float(theta0.double().square().mean())
    metric_sq = rms2 * float(true_gradient.double().square().sum())
    n = theta0.numel()
    for scale in (0.5, 2.0):
        gradient = target[f"gradient/{scale:g}"]
        measured, se = gradient["measured_mean"]
        # The gradient direction is near the true steepest ascent: D close to sqrt(n |P^1/2 g|^2).
        assert measured == pytest.approx(math.sqrt(n * metric_sq), rel=0.25)
        residual, residual_se = gradient["residual_true_minus_token"]
        assert abs(residual) < 4 * residual_se + 1e-3
        assert gradient["flip_rate"] > 0
        radial = target[f"radial/{scale:g}"]
        true_radial = float((true_gradient.double() * theta0.double()).sum())
        assert radial["measured_mean"][0] == pytest.approx(true_radial, abs=4 * radial["measured_mean"][1] + 1e-4)
    # Linear toy: the two scales agree.
    difference, difference_se = target["linearity/gradient"]["difference"]
    assert abs(difference) < 4 * difference_se + 1e-3
    excess, excess_se = target["random/2"]["true_metric_grad_sq"]
    assert abs(excess - metric_sq) < 4 * excess_se + 0.5 * metric_sq


def test_analysis_detects_a_token_path_that_overstates_the_derivative(tmp_path, capsys):
    root, _, _ = build_sweep(tmp_path, token_noise=0.0005, token_bias=1.0, pools=8)
    report = analysis.analyze(root, verify_snapshot=True)
    gradient = report["targets"]["reader_out"]["gradient/2"]
    measured, _ = gradient["measured_mean"]
    residual, residual_se = gradient["residual_true_minus_token"]
    # Token path predicts twice the true slope: measured - token = -measured.
    assert residual == pytest.approx(-measured, rel=0.3)
    assert residual / residual_se < -4


def test_analysis_refuses_a_direction_that_does_not_rebuild(tmp_path):
    root, _, _ = build_sweep(tmp_path, token_noise=0.002, pools=3)
    lines = (root / "collects.jsonl").read_text().splitlines()
    tampered = []
    for line in lines:
        record = json.loads(line)
        if record["kind"] == "random":
            record["direction_sha256"] = "0" * 64
        tampered.append(json.dumps(record))
    (root / "collects.jsonl").write_text("\n".join(tampered) + "\n")
    with pytest.raises(ValueError, match="digest"):
        analysis.analyze(root, verify_snapshot=True)
