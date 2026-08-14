#!/usr/bin/env python3
"""Derive the standard ablation summary from a completed native Byte-Duo run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    native = load_object(args.native_result)
    if native.get("schema") != "byte_duo_result/v1":
        raise ValueError("native result is not Byte-Duo")
    run_name = native.get("run_name")
    steps = native.get("steps")
    metrics_name = native.get("metrics")
    if not isinstance(run_name, str) or not isinstance(steps, int):
        raise ValueError("native Byte-Duo identity is incomplete")
    if not isinstance(metrics_name, str):
        raise ValueError("native Byte-Duo result omitted its metrics path")
    metrics_path = REPO_ROOT / metrics_name
    validations: list[dict[str, object]] = []
    last_session_elapsed = 0.0
    for line_number, line in enumerate(metrics_path.read_text().splitlines(), 1):
        record = json.loads(line)
        if not isinstance(record, dict):
            raise TypeError(f"metrics line {line_number} is not an object")
        if record.get("kind") == "train":
            elapsed = float(record.get("session_elapsed_seconds", math.nan))
            if not math.isfinite(elapsed) or elapsed < last_session_elapsed:
                raise ValueError("training elapsed time is invalid or nonmonotonic")
            last_session_elapsed = elapsed
        if record.get("kind") != "validation":
            continue
        step = record.get("step")
        nats = float(
            record.get("conditional_canvas_duo_nelbo_nats_per_atom", math.nan)
        )
        if not isinstance(step, int) or not math.isfinite(nats):
            raise ValueError(f"validation metrics line {line_number} is invalid")
        validations.append(
            {
                "step": step,
                "type": "val",
                "val_loss": nats,
                "val_diffusion_nelbo_bits_per_atom": nats / math.log(2.0),
            }
        )
    if not validations or validations[-1]["step"] != steps:
        raise ValueError("native metrics omit the final validation")
    result = {
        "name": run_name,
        "script": "scripts/train_byte_duo.py",
        "steps": steps,
        "elapsed_seconds": last_session_elapsed,
        "final_val_bpb": None,
        "final_val_loss": validations[-1]["val_loss"],
        "final_diffusion_nelbo_bits_per_atom": validations[-1][
            "val_diffusion_nelbo_bits_per_atom"
        ],
        "promotion_metric": "gsm8k_exact_match_generation_accuracy",
        "val_entries": validations,
        "returncode": 0,
        "training_returncode": 0,
        "metric_integrity_errors": [],
        "derived_from_native_result": str(args.native_result),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
