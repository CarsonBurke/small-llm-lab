#!/usr/bin/env python3
"""Measure the upstream AR trainer during its steady training window.

This wrapper leaves ``train_gpt.py`` untouched. It samples GPU telemetry only
between the initial validation and the final requested training update, so
compile warmup and validation do not dilute the reference utilization.
GPU use must be queued through ``mlq``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--warmup-updates", type=int, default=20)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--run-id", default="ar_reference_benchmark")
    return parser.parse_args()


def sample_telemetry(
    active: threading.Event,
    stop: threading.Event,
    powers: list[float],
    utilizations: list[float],
    memory_mib: list[float],
) -> None:
    while not stop.wait(0.1):
        if not active.is_set():
            continue
        observed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if observed.returncode:
            continue
        try:
            power, utilization, memory = (
                observed.stdout.strip().splitlines()[0].split(",")
            )
            powers.append(float(power.strip()))
            utilizations.append(float(utilization.strip().removesuffix(" %")))
            memory_mib.append(float(memory.strip()))
        except (IndexError, ValueError):
            continue


def main() -> None:
    args = parse_args()
    if args.updates <= 1 or args.warmup_updates <= 0:
        raise ValueError("updates must exceed one and warmup must be positive")

    environment = os.environ.copy()
    environment.update(
        {
            "ITERATIONS": str(args.updates),
            "WARMUP_STEPS": str(args.warmup_updates),
            "VAL_LOSS_EVERY": str(args.updates + 1),
            "TRAIN_LOG_EVERY": "1",
            "WARMDOWN_ITERS": "0",
            "MAX_WALLCLOCK_SECONDS": "0",
            "RUN_ID": args.run_id,
            "PYTHONUNBUFFERED": "1",
        }
    )

    active = threading.Event()
    stop = threading.Event()
    powers: list[float] = []
    utilizations: list[float] = []
    memory_mib: list[float] = []
    sampler = threading.Thread(
        target=sample_telemetry,
        args=(active, stop, powers, utilizations, memory_mib),
        daemon=True,
    )
    sampler.start()

    process = subprocess.Popen(
        [sys.executable, "-u", "train_gpt.py"],
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    final_step_avg_ms: float | None = None
    model_parameters: int | None = None
    active_started: float | None = None
    active_elapsed: float | None = None
    initial_validation = re.compile(rf"^step:0/{args.updates} val_loss:")
    final_training = re.compile(
        rf"^step:{args.updates}/{args.updates} train_loss:.*step_avg:([0-9.]+)ms"
    )
    model_line = re.compile(r"^model_params:(\d+)")
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            if match := model_line.match(line):
                model_parameters = int(match.group(1))
            if initial_validation.match(line):
                active_started = time.perf_counter()
                active.set()
            if match := final_training.match(line):
                final_step_avg_ms = float(match.group(1))
                active.clear()
                if active_started is not None:
                    active_elapsed = time.perf_counter() - active_started
        returncode = process.wait()
    finally:
        active.clear()
        stop.set()
        sampler.join(timeout=1.0)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)

    failures: list[str] = []
    if returncode:
        failures.append(f"train_gpt.py exited with code {returncode}")
    if final_step_avg_ms is None or active_elapsed is None:
        failures.append("training-window markers were not observed")
    if not powers or not utilizations:
        failures.append("GPU telemetry produced no active-window samples")

    train_batch_tokens = int(environment.get("TRAIN_BATCH_TOKENS", "524288"))
    result = {
        "schema": "ar_reference_benchmark/v1",
        "script": "train_gpt.py",
        "updates": args.updates,
        "warmup_updates": args.warmup_updates,
        "model_parameters": model_parameters,
        "train_batch_tokens": train_batch_tokens,
        "final_step_avg_ms": final_step_avg_ms,
        "active_elapsed_seconds": active_elapsed,
        "tokens_per_second": (
            train_batch_tokens / (final_step_avg_ms / 1_000.0)
            if final_step_avg_ms is not None
            else None
        ),
        "average_power_w": sum(powers) / len(powers) if powers else None,
        "average_gpu_utilization": (
            sum(utilizations) / len(utilizations) if utilizations else None
        ),
        "peak_power_w": max(powers) if powers else None,
        "peak_gpu_utilization": max(utilizations) if utilizations else None,
        "average_memory_used_mib": (
            sum(memory_mib) / len(memory_mib) if memory_mib else None
        ),
        "peak_memory_used_mib": max(memory_mib) if memory_mib else None,
        "telemetry_samples": len(powers),
        "failures": failures,
    }
    encoded = json.dumps(result, sort_keys=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(encoded + "\n")
    print("benchmark_ar_reference " + encoded, flush=True)
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
