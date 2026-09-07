#!/usr/bin/env python3
"""Queue bounded one-at-a-time MiniCPM VAPO replay sweeps through mlq."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
WORKER_TOKEN_ENV = "PARAMETER_GOLF_VAPO_SWEEP_TOKEN"
CONTROLLED_OPTIONS = {
    "logit_chunk_tokens": "--logit-chunk-tokens",
    "nextlat_kl_chunk_tokens": "--nextlat-kl-chunk-tokens",
    "replay_token_budget": "--replay-token-budget",
    "replay_checkpoint_interval": "--replay-checkpoint-interval",
}


@dataclass(frozen=True)
class Settings:
    logit_chunk_tokens: int = 128
    nextlat_kl_chunk_tokens: int = 16
    replay_token_budget: int = 8192
    replay_checkpoint_interval: int = 4


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    varied_axis: str | None
    settings: Settings


def _csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(item) for item in value.split(",") if item))
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated nonnegative integers")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_candidates(
    baseline: Settings,
    *,
    logit_chunks: Sequence[int],
    nextlat_chunks: Sequence[int],
    replay_budgets: Sequence[int],
    checkpoint_intervals: Sequence[int],
) -> list[Candidate]:
    candidates = [Candidate("baseline", None, baseline)]
    axes = {
        "logit_chunk_tokens": logit_chunks,
        "nextlat_kl_chunk_tokens": nextlat_chunks,
        "replay_token_budget": replay_budgets,
        "replay_checkpoint_interval": checkpoint_intervals,
    }
    for axis, values in axes.items():
        current = getattr(baseline, axis)
        for value in values:
            if value == current:
                continue
            settings = replace(baseline, **{axis: value})
            candidates.append(Candidate(f"{axis}_{value}", axis, settings))
    return candidates


def candidate_command(
    base_command: Sequence[str],
    candidate: Candidate,
    output: Path,
    *,
    fixed_resume: Path | None = None,
) -> list[str]:
    command = list(base_command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("benchmark command is required after --")
    for option in (*CONTROLLED_OPTIONS.values(), "--output"):
        if any(token == option or token.startswith(option + "=") for token in command):
            raise ValueError(f"the sweep owns {option}")
    if fixed_resume is not None:
        command[command.index("--resume") + 1] = str(fixed_resume)
    command.extend(("--output", str(output)))
    for field, option in CONTROLLED_OPTIONS.items():
        command.extend((option, str(getattr(candidate.settings, field))))
    return command


def _resume_checkpoint(command: Sequence[str]) -> Path:
    try:
        index = command.index("--resume")
        path = Path(command[index + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise ValueError("benchmark command must use one fixed --resume checkpoint") from error
    if not path.is_file():
        raise ValueError(f"resume checkpoint does not exist: {path}")
    return path


def parse_job_id(stdout: str) -> str:
    payload = json.loads(stdout)
    values = (
        payload.get("job", {}).get("id"),
        payload.get("result", {}).get("job", {}).get("id"),
        payload.get("result", {}).get("id"),
        payload.get("job_id"),
        payload.get("id"),
    )
    for value in values:
        if isinstance(value, (str, int)) and str(value):
            return str(value)
    raise ValueError("mlq submit JSON did not contain a job id")


def _ppo_epochs(command: Sequence[str]) -> int:
    value = 1
    for index, token in enumerate(command):
        if token == "--ppo-epochs":
            try:
                value = int(command[index + 1])
            except (IndexError, ValueError) as error:
                raise ValueError("--ppo-epochs requires an integer") from error
        elif token.startswith("--ppo-epochs="):
            try:
                value = int(token.partition("=")[2])
            except ValueError as error:
                raise ValueError("--ppo-epochs requires an integer") from error
    return value


@dataclass(frozen=True)
class ScalarPoint:
    step: int
    wall_time: float
    value: float


def _scalar_series(run_output: Path) -> dict[str, tuple[ScalarPoint, ...]]:
    event_dir = run_output / "tensorboard"
    if not event_dir.is_dir():
        return {}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {
        tag: tuple(
            ScalarPoint(
                step=int(event.step),
                wall_time=float(event.wall_time),
                value=float(event.value),
            )
            for event in accumulator.Scalars(tag)
        )
        for tag in accumulator.Tags().get("scalars", [])
        if accumulator.Scalars(tag)
    }


def _metric_by_step(
    metrics: dict[str, tuple[ScalarPoint, ...]], *names: str
) -> dict[int, float]:
    for name in names:
        if points := metrics.get(name):
            return {
                point.step: point.value
                for point in sorted(points, key=lambda point: point.wall_time)
            }
    return {}


def _worker_result(
    *,
    candidate: Candidate,
    command: Sequence[str],
    run_output: Path,
    timeout_seconds: int,
    log_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        returncode = completed.returncode
        output = completed.stdout + completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        returncode = 124
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout or ""
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr or ""
        output = stdout + stderr
    wall_seconds = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output)
    lowered = output.lower()
    oom = any(
        marker in lowered
        for marker in ("out of memory", "cuda oom", "cuda_error_out_of_memory")
    )
    metrics = _scalar_series(run_output)
    generated_by_step = _metric_by_step(
        metrics,
        "rollout_performance/generated_tokens",
        "rollout_quality/generated_tokens",
    )
    rollout_by_step = _metric_by_step(
        metrics, "rollout_performance/rollout_seconds"
    )
    refresh_by_step = _metric_by_step(
        metrics, "optimization/behavior_refresh_seconds"
    )
    update_by_step = _metric_by_step(
        metrics, "optimization/update_seconds"
    )
    rollout_peak_by_step = _metric_by_step(
        metrics, "system_rollout/peak_vram_bytes"
    )
    update_peak_by_step = _metric_by_step(
        metrics, "system_update/peak_vram_bytes"
    )
    completed_steps = sorted(
        step
        for step in (
            generated_by_step.keys()
            & rollout_by_step.keys()
            & refresh_by_step.keys()
            & rollout_peak_by_step.keys()
        )
        if step + 1 in update_by_step and step + 1 in update_peak_by_step
    )
    if completed_steps:
        generated_tokens = sum(
            generated_by_step[step] for step in completed_steps
        )
        rollout_seconds = sum(
            rollout_by_step[step] for step in completed_steps
        )
        behavior_refresh_seconds = sum(
            refresh_by_step[step] for step in completed_steps
        )
        update_seconds = sum(
            update_by_step[step + 1] for step in completed_steps
        )
        core_phase_seconds = (
            rollout_seconds + behavior_refresh_seconds + update_seconds
        )
    else:
        generated_tokens = None
        rollout_seconds = None
        behavior_refresh_seconds = None
        update_seconds = None
        core_phase_seconds = None
    post_update_kl_by_step = _metric_by_step(
        metrics, "optimization/post_update_kl_seconds"
    )
    peak_values = [
        value
        for step in completed_steps
        for value in (
            rollout_peak_by_step[step],
            update_peak_by_step[step + 1],
        )
    ]
    peak_vram = max(peak_values, default=None)
    if oom:
        status = "oom"
    elif returncode and not timed_out:
        status = "error"
    elif (
        generated_tokens is None
        or peak_vram is None
        or core_phase_seconds is None
    ):
        status = "timeout" if timed_out else "missing_metrics"
    else:
        status = "success"
    return {
        "schema": "minicpm_vapo_replay_sweep/v1",
        "candidate_id": candidate.candidate_id,
        "varied_axis": candidate.varied_axis,
        "settings": asdict(candidate.settings),
        "status": status,
        "returncode": returncode,
        "timed_out_after_measurement": timed_out and status == "success",
        "completed_cycle_count": len(completed_steps),
        "wall_time_seconds": wall_seconds,
        "measured_seconds": core_phase_seconds,
        "generated_tokens": generated_tokens,
        "tokens_per_second": (
            generated_tokens / core_phase_seconds
            if status == "success"
            and generated_tokens is not None
            and core_phase_seconds is not None
            else None
        ),
        "peak_vram_bytes": peak_vram,
        "rollout_seconds": rollout_seconds,
        "behavior_refresh_seconds": behavior_refresh_seconds,
        "update_seconds": update_seconds,
        "post_update_kl_seconds": (
            sum(
                post_update_kl_by_step[step + 1]
                for step in completed_steps
                if step + 1 in post_update_kl_by_step
            )
            or None
        ),
        "command": list(command),
        "log": str(log_path),
    }


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in results if row["status"] == "success"]
    ranked = sorted(
        successful,
        key=lambda row: (-row["tokens_per_second"], row["peak_vram_bytes"]),
    )
    return {
        "schema": "minicpm_vapo_replay_sweep_summary/v1",
        "candidate_count": len(results),
        "success_count": len(successful),
        "ranked_candidates": [
            {"rank": rank, **row} for rank, row in enumerate(ranked, 1)
        ],
        "failures": [row for row in results if row["status"] != "success"],
    }


def run_sweep(args: argparse.Namespace) -> int:
    if _ppo_epochs(args.command) != 1:
        raise ValueError("benchmark commands must use exactly one PPO epoch")
    if not 10 <= args.timeout_seconds < 120:
        raise ValueError("timeout must be between 10 and 119 seconds")
    baseline = Settings(
        args.baseline_logit_chunk,
        args.baseline_nextlat_chunk,
        args.baseline_replay_budget,
        args.baseline_checkpoint_interval,
    )
    candidates = build_candidates(
        baseline,
        logit_chunks=args.logit_chunks,
        nextlat_chunks=args.nextlat_chunks,
        replay_budgets=args.replay_budgets,
        checkpoint_intervals=args.checkpoint_intervals,
    )
    root = (args.output_root / args.name).resolve()
    checkpoint = _resume_checkpoint(args.command)
    checkpoint_hash = _sha256(checkpoint)
    root.mkdir(parents=True, exist_ok=False)
    token = secrets.token_hex(24)
    plans = []
    for candidate in candidates:
        candidate_root = root / "candidates" / candidate.candidate_id
        run_output = candidate_root / "run"
        command = candidate_command(
            args.command, candidate, run_output, fixed_resume=checkpoint
        )
        result_path = candidate_root / "metrics.json"
        worker = [
            sys.executable,
            str(SCRIPT_PATH),
            "_worker",
            "--token",
            token,
            "--candidate-json",
            json.dumps({**asdict(candidate), "settings": asdict(candidate.settings)}),
            "--result",
            str(result_path),
            "--run-output",
            str(run_output),
            "--timeout-seconds",
            str(args.timeout_seconds - 5),
            "--",
            *command,
        ]
        submit = [
            "mlq",
            "submit",
            "--name",
            f"{args.name}_{candidate.candidate_id}"[:120],
            "--priority",
            str(args.priority),
            "--cwd",
            str(REPO_ROOT),
            "--max-parallel-runs",
            "1",
            "--time-limit",
            f"{args.timeout_seconds}s",
            "--env",
            f"{WORKER_TOKEN_ENV}={token}",
            "--json",
            "--",
            *worker,
        ]
        plans.append((candidate, command, result_path, submit))
    manifest = {
        "schema": "minicpm_vapo_replay_sweep_manifest/v1",
        "name": args.name,
        "timeout_seconds": args.timeout_seconds,
        "priority": args.priority,
        "resume_checkpoint": str(checkpoint),
        "resume_checkpoint_sha256": checkpoint_hash,
        "base_command": list(args.command),
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "varied_axis": candidate.varied_axis,
                "settings": asdict(candidate.settings),
                "command": command,
            }
            for candidate, command, _, _ in plans
        ],
    }
    _write_json(root / "manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    jobs: list[tuple[Path, str]] = []
    results: list[dict[str, Any]] = []
    for candidate, _, result_path, submit in plans:
        submitted = subprocess.run(submit, text=True, capture_output=True, check=True)
        job_id = parse_job_id(submitted.stdout)
        jobs.append((result_path, job_id))
        print(f"submitted {candidate.candidate_id} as mlq job {job_id}", flush=True)
    for result_path, job_id in jobs:
        subprocess.run(("mlq", "wait", job_id), check=False)
        if not result_path.is_file():
            raise RuntimeError(f"mlq job {job_id} produced no result at {result_path}")
        results.append(json.loads(result_path.read_text()))
    root.mkdir(parents=True, exist_ok=True)
    (root / "metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results)
    )
    summary = summarize(results)
    _write_json(root / "result.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["success_count"] else 1


def run_worker(args: argparse.Namespace) -> int:
    if os.environ.get(WORKER_TOKEN_ENV) != args.token:
        raise RuntimeError("_worker must be dispatched by this script through mlq")
    payload = json.loads(args.candidate_json)
    candidate = Candidate(
        payload["candidate_id"],
        payload["varied_axis"],
        Settings(**payload["settings"]),
    )
    result = _worker_result(
        candidate=candidate,
        command=args.command[1:] if args.command[:1] == ["--"] else args.command,
        run_output=args.run_output,
        timeout_seconds=args.timeout_seconds,
        log_path=args.result.with_name("workload.log"),
    )
    _write_json(args.result, result)
    print(json.dumps(result, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--name", required=True)
    run.add_argument("--priority", type=int, default=0)
    run.add_argument("--output-root", type=Path, default=Path("ablation_results"))
    run.add_argument("--timeout-seconds", type=int, default=110)
    run.add_argument("--baseline-logit-chunk", type=int, default=128)
    run.add_argument("--baseline-nextlat-chunk", type=int, default=16)
    run.add_argument("--baseline-replay-budget", type=int, default=8192)
    run.add_argument("--baseline-checkpoint-interval", type=int, default=4)
    run.add_argument("--logit-chunks", type=_csv_ints, default=(64, 128, 256))
    run.add_argument("--nextlat-chunks", type=_csv_ints, default=(8, 16, 32))
    run.add_argument("--replay-budgets", type=_csv_ints, default=(8192, 10240, 12288))
    run.add_argument("--checkpoint-intervals", type=_csv_ints, default=(0, 2, 4))
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("command", nargs=argparse.REMAINDER)

    worker = subparsers.add_parser("_worker")
    worker.add_argument("--token", required=True)
    worker.add_argument("--candidate-json", required=True)
    worker.add_argument("--result", type=Path, required=True)
    worker.add_argument("--run-output", type=Path, required=True)
    worker.add_argument("--timeout-seconds", type=int, required=True)
    worker.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_sweep(args) if args.action == "run" else run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
