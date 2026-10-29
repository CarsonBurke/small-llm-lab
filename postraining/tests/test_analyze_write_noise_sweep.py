"""The write-noise sweep analysis separates shared signal from flip noise.

A toy policy with common random numbers: prompt p draws one noise value
eps_p (shared by its samples, flipped by the scale's sign) and sample i a
fixed uniform threshold t_i; its return is 1[t_i < 1/2 + a s eps_p]. Then
D_i = (G(+s) - G(-s)) / 2 has E[D | p, eps] = a s eps exactly for small
a s |eps|, so the shared-signal variance is (a s)^2 E[eps^2] = (a s)^2,
while E[D^2] = P(flip) / 4 = a s E|eps| / 2 is far larger: every flip is
all-or-nothing.

A second toy responds to the noise's magnitude only, 1[t_i < 1/2 + a s
|eps_p|]: D is zero and E[C | p, eps] = a s |eps|, so ``even_lower`` is
(a s)^2 and ``cost`` is a s E|eps|.
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
    WRITE_NOISE_SWEEP_SCHEMA,
    SweepWriter,
    WriteNoiseSweepConfig,
    run_write_noise_sweep,
)
from postraining.tests.test_perturbation_sweep import fake_group

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "analyze_write_noise_sweep", REPO / "scripts" / "analyze_write_noise_sweep.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

SLOPE, SAMPLES, PROMPTS = 0.05, 8, 64


def odd_response(scale, eps):
    return SLOPE * scale * eps


def build_sweep(tmp_path: Path, scales=(0.5, 1.0), pools=8, response=odd_response) -> Path:
    sampler = SimpleNamespace(cursor=0)

    def collect_at(cursor, plan):
        assert plan.shared_per_prompt
        sampler.cursor = cursor + PROMPTS
        groups = []
        for prompt in range(PROMPTS):
            generator = torch.Generator().manual_seed(1000 * cursor + prompt)
            eps = float(torch.randn((), generator=generator))
            thresholds = torch.rand(SAMPLES, generator=generator)
            wins = (thresholds < 0.5 + response(plan.scale, eps)).float()
            groups.append(fake_group(
                [cursor + prompt], [[int(w) + 1] for w in wins.tolist()], wins.tolist()))
        return groups

    root = tmp_path / "write_noise_sweep"
    writer = SweepWriter(root, {
        "schema": WRITE_NOISE_SWEEP_SCHEMA, "samples_per_prompt": SAMPLES,
        "scales": list(scales),
        "noise": {"shared_per_prompt": True, "memory_heads": 4, "memory_head_dim": 64},
    })
    completed = run_write_noise_sweep(
        WriteNoiseSweepConfig(tuple(scales)), pools, cursor=lambda: sampler.cursor,
        collect_at=collect_at, writer=writer, out_of_time=lambda estimate: False,
        log=lambda values: None,
    )
    assert completed == pools
    return root


def run_analysis(root: Path, tmp_path: Path, monkeypatch) -> dict:
    output = tmp_path / "analysis.json"
    monkeypatch.setattr("sys.argv", ["analyze", str(root), "--bootstrap", "300",
                                     "--json", str(output)])
    analysis.main()
    return json.loads(output.read_text())


def test_lower_bound_recovers_the_shared_signal_and_upper_bounds_it(tmp_path, monkeypatch):
    result = run_analysis(build_sweep(tmp_path), tmp_path, monkeypatch)
    assert result["pools"] == 8 and result["trajectories_per_pool"] == PROMPTS * SAMPLES
    assert result["null"] == {"changed": 0, "abs_reward_shift": 0.0}
    assert result["noise_coordinates"] == 2 * 4 * 64 + 2 * 4
    for scale in (0.5, 1.0):
        stats = result["scales"][f"{scale:g}"]
        signal = (SLOPE * scale) ** 2
        assert stats["lower"] == pytest.approx(signal, abs=4 * stats["lower_se"])
        assert abs(stats["control"]) < 4 * stats["control_se"]
        expected_upper = SLOPE * scale * math.sqrt(2 / math.pi) / 2
        assert stats["upper"] == pytest.approx(expected_upper, abs=4 * stats["upper_se"])
        assert stats["upper"] > 5 * stats["lower"]
        # Each sign flips the samples with t between 1/2 and 1/2 + a s eps:
        # probability a s E|eps|.
        assert stats["flip_rate"] == pytest.approx(
            SLOPE * scale * math.sqrt(2 / math.pi), abs=4 * stats["flip_rate_se"])
        # Symmetric in eps: no second-order cost.
        assert abs(stats["cost"]) < 4 * stats["cost_se"] + 1e-12
        assert abs(stats["even_lower"]) < 4 * stats["even_lower_se"] + 1e-12
        assert stats["rho2_lower"] == pytest.approx(stats["lower"] / stats["var_within"])
    # Signal scales as s^2, flip noise as s.
    low, high = result["scales"]["0.5"], result["scales"]["1"]
    assert high["lower_over_scale_sq"] == pytest.approx(low["lower_over_scale_sq"], rel=0.5)
    assert high["upper_over_scale"] == pytest.approx(low["upper_over_scale"], rel=0.2)


def test_a_magnitude_response_is_even_and_invisible_to_both_odd_bounds(tmp_path, monkeypatch):
    root = build_sweep(tmp_path, response=lambda scale, eps: SLOPE * abs(scale * eps))
    result = run_analysis(root, tmp_path, monkeypatch)
    for scale in (0.5, 1.0):
        stats = result["scales"][f"{scale:g}"]
        assert stats["upper"] == 0.0 and stats["lower"] == 0.0
        assert stats["cost"] == pytest.approx(
            SLOPE * scale * math.sqrt(2 / math.pi), abs=4 * stats["cost_se"])
        assert stats["even_lower"] == pytest.approx(
            (SLOPE * scale) ** 2, abs=4 * stats["even_lower_se"])


def test_incomplete_pools_are_skipped(tmp_path, monkeypatch):
    root = build_sweep(tmp_path, pools=2)
    lines = (root / "collects.jsonl").read_text().splitlines()
    (root / "collects.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    assert run_analysis(root, tmp_path, monkeypatch)["pools"] == 1
