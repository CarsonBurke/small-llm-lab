#!/usr/bin/env python3
"""Evaluate Byte-Duo's conditional-canvas NELBO on independent ledgers.

This is the within-recipe diffusion metric, not autoregressive BPB. GPU model
execution must be submitted through ``mlq``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.duo_model import DuoModel
from pretraining.byte_diffusion.duo import DuoSchedule
from pretraining.byte_diffusion.readiness import (
    DUO_CANONICAL_VALIDATION_BRANCHES,
    DUO_CANONICAL_VALIDATION_CANVAS_LENGTH,
    DUO_DATA_BRANCH_SPAN_LENGTH,
    DUO_DATA_REQUIRED_BRANCH_BYTES,
    duo_geometry_contract,
)
from pretraining.byte_diffusion.training import load_data_directory, model_config_from_dict
from pretraining.byte_diffusion.training_duo import validate_duo
from scripts.train_byte_duo import _prepared_validation_ledger, source_provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-rows", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, required=True)
    parser.add_argument(
        "--ledger-seeds",
        default="11337,21337,31337,41337,51337",
        help="comma-separated independent deterministic validation ledgers",
    )
    parser.add_argument(
        "--serving-distribution",
        action="store_true",
        help=(
            "Evaluate the topology's serving-origin distribution: exact byte "
            "origins for full-resolution Duo, or deterministic clean-prefix "
            "phases for legacy patched-global Duo."
        ),
    )
    parser.add_argument(
        "--native-training-geometry",
        action="store_true",
        help=(
            "diagnostic only: evaluate the checkpoint's train geometry instead "
            "of the canonical comparable 512x8 headline geometry"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Byte-Duo NELBO evaluation must run through mlq on CUDA")
    if args.validation_rows <= 0 or args.validation_batch_size <= 0:
        raise ValueError("validation dimensions must be positive")
    seeds = tuple(int(value) for value in args.ledger_seeds.split(","))
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("NELBO reliability requires at least three unique ledger seeds")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "byte_duo_checkpoint/v2":
        raise ValueError("NELBO evaluation requires a Byte-Duo checkpoint")
    provenance = source_provenance()
    if payload.get("source_sha256") != provenance["sha256"]:
        raise ValueError("NELBO evaluator source differs from the checkpoint")
    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError("Byte-Duo checkpoint has no training contract")
    train_canvas = int(training["canvas_length"])
    train_branches = int(training["branches"])
    expected_geometry = duo_geometry_contract(train_canvas, train_branches)
    for key, value in expected_geometry.items():
        if training.get(key) != value:
            raise ValueError(f"Byte-Duo checkpoint {key} contract differs")
    if args.native_training_geometry:
        eval_canvas, eval_branches = train_canvas, train_branches
        evaluation_geometry = "native_training_geometry_diagnostic"
    else:
        eval_canvas = DUO_CANONICAL_VALIDATION_CANVAS_LENGTH
        eval_branches = DUO_CANONICAL_VALIDATION_BRANCHES
        evaluation_geometry = "canonical_512x8_headline"
    dataset_manifest = json.loads((args.data_path / "manifest.json").read_text())
    data_sha256 = dataset_manifest.get("payload_sha256")
    if data_sha256 != payload.get("dataset_payload_sha256"):
        raise ValueError("NELBO evaluation dataset differs from the checkpoint")
    config = model_config_from_dict(payload["model_config"])
    serving_distribution = bool(args.serving_distribution)
    if serving_distribution:
        if config.duo_mutable_topology == "full_resolution_decoder":
            evaluation_origin_policy = "exact_byte_origins"
            evaluation_origin_stride = config.duo_origin_stride
            fixed_clean_phase_policy = "none"
        else:
            evaluation_origin_policy = "patch_origin_with_clean_prefix_phase"
            evaluation_origin_stride = config.patch_stride
            fixed_clean_phase_policy = "deterministic_0_to_patch_stride_minus_1"
    else:
        evaluation_origin_policy = "canonical_patch_aligned"
        evaluation_origin_stride = config.patch_stride
        fixed_clean_phase_policy = "none"
    model = DuoModel(config, schedule_eps=float(training["schedule_eps"]))
    model.load_state_dict(payload["model"], strict=True)
    model = model.to("cuda").eval()
    execution_model = torch.compile(model, dynamic=True, fullgraph=False)
    schedule = DuoSchedule(float(training["schedule_eps"]))
    _, _, validation = load_data_directory(
        args.data_path,
        chunk_size=int(training.get("chunk_size", 8192)),
        recipe="blt_d",
        required_branch_bytes=DUO_DATA_REQUIRED_BRANCH_BYTES,
        branch_span_length=DUO_DATA_BRANCH_SPAN_LENGTH,
        validation_chunk_limit=args.validation_rows,
        expected_payload_sha256=data_sha256,
    )
    evaluated_rows = min(args.validation_rows, len(validation))
    records: list[dict[str, object]] = []
    for seed in seeds:
        ledger = _prepared_validation_ledger(
            model,
            validation,
            rows=evaluated_rows,
            batch_size=args.validation_batch_size,
            patch_stride=config.patch_stride,
            canvas_length=eval_canvas,
            branches=eval_branches,
            seed=seed,
            schedule=schedule,
            device=torch.device("cuda"),
            row_indices=np.arange(evaluated_rows, dtype=np.int64),
            expose_random_phase=serving_distribution,
        )
        metric = validate_duo(
            execution_model,  # type: ignore[arg-type]
            ledger,
            canvas_length=eval_canvas,
            branches=eval_branches,
            seed=seed,
            total_rows=evaluated_rows,
            compute_ar_diagnostic=False,
        )
        record = {"ledger_seed": seed, **asdict(metric)}
        records.append(record)
        print("byte_duo_nelbo " + json.dumps(record, sort_keys=True), flush=True)

    values = tuple(
        float(record["conditional_canvas_nelbo_nats_per_atom"])
        for record in records
    )
    mean = statistics.mean(values)
    sample_sd = statistics.stdev(values)
    standard_error = sample_sd / math.sqrt(len(values))
    # Five ledgers are the production contract. For any other count, use the
    # normal coefficient and label it honestly rather than inventing a t table.
    coefficient = 2.776 if len(values) == 5 else 1.96
    report = {
        "schema": "byte_duo_nelbo_multiledger/v1",
        "metric_semantics": (
            "serving_distribution_conditional_canvas_duo_nelbo_not_ar_bpb"
            if serving_distribution
            else "native_geometry_conditional_canvas_duo_nelbo_diagnostic_not_ar_bpb"
            if args.native_training_geometry
            else "conditional_canvas_duo_nelbo_not_ar_bpb"
        ),
        "serving_distribution": serving_distribution,
        "evaluation_origin_policy": evaluation_origin_policy,
        "evaluation_origin_stride": evaluation_origin_stride,
        "fixed_clean_phase_policy": fixed_clean_phase_policy,
        "evaluation_geometry": evaluation_geometry,
        "evaluation_canvas_length": eval_canvas,
        "evaluation_branches": eval_branches,
        "training_geometry": expected_geometry["training_geometry"],
        "canonical_validation_geometry": expected_geometry[
            "canonical_validation_geometry"
        ],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "completed_steps": int(payload["steps"]),
        "source_sha256": provenance["sha256"],
        "dataset_payload_sha256": data_sha256,
        "validation_rows": evaluated_rows,
        "validation_batch_size": args.validation_batch_size,
        "ledger_seeds": seeds,
        "records": records,
        "mean_nats_per_atom": mean,
        "sample_sd_nats_per_atom": sample_sd,
        "standard_error_nats_per_atom": standard_error,
        "confidence_interval_coefficient": coefficient,
        "approximate_95_percent_interval_nats_per_atom": (
            mean - coefficient * standard_error,
            mean + coefficient * standard_error,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print("byte_duo_nelbo_summary " + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
