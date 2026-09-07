from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from torch.utils.tensorboard import SummaryWriter

import pytest

from scripts import benchmark_minicpm_vapo_sweep as sweep


def test_one_at_a_time_candidates_change_exactly_one_axis() -> None:
    baseline = sweep.Settings()
    candidates = sweep.build_candidates(
        baseline,
        logit_chunks=(64, 128, 256),
        nextlat_chunks=(8, 16, 32),
        replay_budgets=(8192, 12288),
        checkpoint_intervals=(0, 2, 4),
    )

    assert candidates[0] == sweep.Candidate("baseline", None, baseline)
    assert len(candidates) == 8
    for candidate in candidates[1:]:
        changed = [
            name
            for name in sweep.CONTROLLED_OPTIONS
            if getattr(candidate.settings, name) != getattr(baseline, name)
        ]
        assert changed == [candidate.varied_axis]


def test_candidate_command_owns_output_and_sweep_controls(tmp_path: Path) -> None:
    candidate = sweep.Candidate("baseline", None, sweep.Settings())
    command = sweep.candidate_command(
        ["python", "-m", "postraining.train_minicpm_vapo", "--steps", "404"],
        candidate,
        tmp_path,
    )

    assert command[command.index("--output") + 1] == str(tmp_path)
    for field, option in sweep.CONTROLLED_OPTIONS.items():
        assert command[command.index(option) + 1] == str(
            getattr(candidate.settings, field)
        )
    with pytest.raises(ValueError, match="sweep owns"):
        sweep.candidate_command(["python", "train.py", "--logit-chunk-tokens", "8"], candidate, tmp_path)


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"job": {"id": 17}}, "17"),
        ({"result": {"job": {"id": "18"}}}, "18"),
        ({"job_id": 19}, "19"),
    ],
)
def test_parse_job_id_supports_mlq_json_shapes(payload: dict, expected: str) -> None:
    assert sweep.parse_job_id(json.dumps(payload)) == expected


def test_summary_ranks_success_by_normalized_throughput_then_vram() -> None:
    results = [
        {
            "candidate_id": "slow",
            "status": "success",
            "tokens_per_second": 90.0,
            "wall_time_seconds": 9.0,
            "peak_vram_bytes": 2,
        },
        {
            "candidate_id": "fast-large",
            "status": "success",
            "tokens_per_second": 100.0,
            "wall_time_seconds": 8.0,
            "peak_vram_bytes": 3,
        },
        {
            "candidate_id": "fast-small",
            "status": "success",
            "tokens_per_second": 100.0,
            "wall_time_seconds": 8.0,
            "peak_vram_bytes": 1,
        },
        {
            "candidate_id": "oom",
            "status": "oom",
            "tokens_per_second": None,
            "wall_time_seconds": 1.0,
            "peak_vram_bytes": None,
        },
    ]

    summary = sweep.summarize(results)

    assert [row["candidate_id"] for row in summary["ranked_candidates"]] == [
        "fast-small",
        "fast-large",
        "slow",
    ]
    assert summary["success_count"] == 3
    assert [row["candidate_id"] for row in summary["failures"]] == ["oom"]


def test_dry_run_writes_fixed_checkpoint_manifest_without_mlq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixed checkpoint")

    def unexpected_run(*args, **kwargs):
        raise AssertionError("dry run must not invoke mlq")

    monkeypatch.setattr(sweep.subprocess, "run", unexpected_run)
    parser = sweep.build_parser()
    args = parser.parse_args(
        [
            "run",
            "--name",
            "dry",
            "--output-root",
            str(tmp_path / "results"),
            "--dry-run",
            "--",
            "python",
            "-m",
            "postraining.train_minicpm_vapo",
            "--resume",
            str(checkpoint),
            "--steps",
            "404",
        ]
    )

    assert sweep.run_sweep(args) == 0
    manifest = json.loads(
        (tmp_path / "results" / "dry" / "manifest.json").read_text()
    )
    assert manifest["resume_checkpoint_sha256"] == sweep._sha256(checkpoint)
    assert manifest["timeout_seconds"] == 110
    assert len(manifest["candidates"]) == 9
    for candidate in manifest["candidates"]:
        assert "--output" in candidate["command"]


def test_sweep_rejects_multi_epoch_commands_before_creating_output(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixed checkpoint")
    output_root = tmp_path / "results"
    args = sweep.build_parser().parse_args(
        [
            "run",
            "--name",
            "multi-epoch",
            "--output-root",
            str(output_root),
            "--dry-run",
            "--",
            "python",
            "train.py",
            "--resume",
            str(checkpoint),
            "--ppo-epochs=2",
        ]
    )

    with pytest.raises(ValueError, match="exactly one PPO epoch"):
        sweep.run_sweep(args)
    assert not output_root.exists()


def test_worker_reads_exact_training_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_output = tmp_path / "run"
    writer = SummaryWriter(run_output / "tensorboard")
    writer.add_scalar("rollout_performance/generated_tokens", 800, 4)
    writer.add_scalar("rollout_performance/rollout_seconds", 4.0, 4)
    writer.add_scalar("rollout_performance/generated_tokens", 200, 5)
    writer.add_scalar("rollout_performance/rollout_seconds", 2.0, 5)
    writer.add_scalar("optimization/behavior_refresh_seconds", 0.5, 5)
    writer.add_scalar("optimization/update_seconds", 3.0, 6)
    writer.add_scalar("system_rollout/peak_vram_bytes", 14, 5)
    writer.add_scalar("system_update/peak_vram_bytes", 11, 6)
    writer.add_scalar("optimization/behavior_refresh_seconds", 1.0, 4)
    writer.add_scalar("optimization/update_seconds", 2.0, 5)
    writer.add_scalar("system_rollout/peak_vram_bytes", 12, 4)
    writer.add_scalar("system_update/peak_vram_bytes", 10, 5)
    writer.close()
    monkeypatch.setattr(
        sweep.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )

    result = sweep._worker_result(
        candidate=sweep.Candidate("baseline", None, sweep.Settings()),
        command=["python", "train.py"],
        run_output=run_output,
        timeout_seconds=100,
        log_path=tmp_path / "workload.log",
    )
    assert result["generated_tokens"] == 1000
    assert result["peak_vram_bytes"] == 14
    assert result["rollout_seconds"] == 6.0
    assert result["behavior_refresh_seconds"] == 1.5
    assert result["update_seconds"] == 5.0
    assert result["measured_seconds"] == 12.5
    assert result["tokens_per_second"] == pytest.approx(80.0)


def test_worker_keeps_complete_metrics_after_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_output = tmp_path / "run"
    writer = SummaryWriter(run_output / "tensorboard")
    writer.add_scalar("rollout_performance/generated_tokens", 1_000, 5)
    writer.add_scalar("rollout_performance/rollout_seconds", 6.0, 5)
    writer.add_scalar("system_rollout/peak_vram_bytes", 12, 5)
    writer.add_scalar("optimization/behavior_refresh_seconds", 1.5, 5)
    writer.add_scalar("optimization/update_seconds", 5.0, 6)
    writer.add_scalar("system_update/peak_vram_bytes", 14, 6)
    writer.add_scalar("rollout_performance/generated_tokens", 200, 6)
    writer.add_scalar("rollout_performance/rollout_seconds", 2.0, 6)
    writer.add_scalar("system_rollout/peak_vram_bytes", 16, 6)
    writer.add_scalar("optimization/behavior_refresh_seconds", 0.5, 6)
    writer.add_scalar("optimization/update_seconds", 1.0, 7)
    writer.close()

    def time_out(*args, **kwargs):
        raise sweep.subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(sweep.subprocess, "run", time_out)

    result = sweep._worker_result(
        candidate=sweep.Candidate("baseline", None, sweep.Settings()),
        command=["python", "train.py"],
        run_output=run_output,
        timeout_seconds=100,
        log_path=tmp_path / "workload.log",
    )

    assert result["status"] == "success"
    assert result["completed_cycle_count"] == 1
    assert result["generated_tokens"] == 1_000
    assert result["peak_vram_bytes"] == 14
    assert result["timed_out_after_measurement"] is True
    assert result["tokens_per_second"] == pytest.approx(80.0)


def test_worker_refuses_direct_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(sweep.WORKER_TOKEN_ENV, raising=False)
    args = sweep.build_parser().parse_args(
        [
            "_worker",
            "--token",
            "secret",
            "--candidate-json",
            json.dumps(
                {
                    "candidate_id": "baseline",
                    "varied_axis": None,
                    "settings": sweep.asdict(sweep.Settings()),
                }
            ),
            "--result",
            str(tmp_path / "result.json"),
            "--run-output",
            str(tmp_path / "run"),
            "--timeout-seconds",
            "100",
            "--",
            "python",
            "train.py",
        ]
    )

    with pytest.raises(RuntimeError, match="through mlq"):
        sweep.run_worker(args)
