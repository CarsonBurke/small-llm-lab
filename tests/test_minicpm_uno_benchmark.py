"""CPU-only adoption-gate regressions; no model workloads."""

import json
from pathlib import Path
import sys

import pytest

from scripts import benchmark_minicpm_uno as benchmark


@pytest.mark.parametrize("uno_throughput, passed", [(187.4, False), (187.5, True)])
def test_uno_must_beat_current_production_ar(
    tmp_path, monkeypatch, uno_throughput, passed
):
    checkpoint = tmp_path / "adapter.pt"
    checkpoint.touch()
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_minicpm_uno.py",
            "--uno-checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ],
    )

    def measured_worker(command, *, check):
        args = benchmark.build_parser().parse_args(command[2:])
        throughput = {
            "legacy-ar": 100.0,
            "ar": 110.0,
            "optimized-ar": 150.0,
            "uno": uno_throughput,
        }[args.engine]
        report = {
            "model_id": "same-actor",
            "revision": "same-revision",
            "actor_sha256": "same-weights",
            "uno_sha256": "same-adapter",
            "data_sha256": "same-corpus",
            "prompt_token_ids": [[1, 2]],
            "rollout_arithmetic": "invariant"
            if args.engine in {"ar", "uno"}
            else args.engine,
            "median_useful_tokens_per_second": throughput,
            "median_rollout_seconds": 1000 / throughput,
        }
        Path(args.output).write_text(json.dumps(report))

    monkeypatch.setattr(benchmark.subprocess, "run", measured_worker)
    with pytest.raises(SystemExit) as exit_info:
        benchmark.main()
    assert exit_info.value.code == (0 if passed else 2)
    report = json.loads(output.read_text())
    assert report["performance_gate_passed"] is passed
    assert report["optimized_ar_useful_rollout_speedup"] == uno_throughput / 150.0
