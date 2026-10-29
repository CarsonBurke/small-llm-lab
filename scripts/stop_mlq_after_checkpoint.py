#!/usr/bin/env python3
"""Stop a specific MLQ attempt after its atomic boundary checkpoint exists.

This is a CPU-only queue-control watcher, not a model workload. The trainer
must publish the supplied checkpoint by atomic rename after saving its full
state. A later retry is never stopped by this watcher.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def queue_job(job_id: int) -> dict:
    return json.loads(subprocess.check_output(["mlq", "show", str(job_id), "--json"], text=True))


def write_state(path: Path, **state) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    args = parser.parse_args()
    initial = queue_job(args.job)
    if initial["state"] != "running" or initial["attempts"][-1]["id"] != args.attempt:
        raise RuntimeError("expected training attempt is not running")
    if args.checkpoint.exists():
        raise RuntimeError("checkpoint already exists; refusing an ambiguous boundary")
    write_state(args.state, status="armed", job=args.job, attempt=args.attempt,
                checkpoint=str(args.checkpoint))
    next_job_check = time.monotonic() + 5
    while not args.checkpoint.is_file():
        if time.monotonic() >= next_job_check:
            job = queue_job(args.job)
            if job["state"] != "running" or job["attempts"][-1]["id"] != args.attempt:
                write_state(args.state, status="ended_before_checkpoint", job=args.job,
                            attempt=args.attempt, observed_state=job["state"])
                return
            next_job_check = time.monotonic() + 5
        time.sleep(0.1)
    # Rename is atomic: the final name is never a partially serialized file.
    current = queue_job(args.job)
    if current["attempts"][-1]["id"] != args.attempt:
        write_state(args.state, status="superseded_attempt", job=args.job, attempt=args.attempt)
        return
    if current["state"] == "running" and current["attempts"][-1]["id"] == args.attempt:
        subprocess.run(["mlq", "cancel", str(args.job), "--idempotency-key",
                        f"stop-at-checkpoint-{args.job}-{args.attempt}", "--json"], check=True)
    while True:
        current = queue_job(args.job)
        if current["state"] not in ("running", "cancelling", "cancel_requested"):
            break
        time.sleep(0.5)
    # Read checkpoint metadata with mapped CPU storage; no model is executed.
    import torch
    payload = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    if payload["step"] != 0 or payload["value_warmup_step"] != args.warmup_steps:
        raise RuntimeError("checkpoint is not the requested critic-only boundary")
    write_state(args.state, status="stopped_after_checkpoint", job=args.job,
                attempt=args.attempt, checkpoint=str(args.checkpoint),
                checkpoint_bytes=args.checkpoint.stat().st_size,
                actor_step=payload["step"], critic_warmup_step=payload["value_warmup_step"],
                sampler_cursor=payload["sampler_cursor"],
                observed_state=current["state"])


if __name__ == "__main__":
    main()
