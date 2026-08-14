#!/usr/bin/env python3
"""Diagnose Byte-Duo denoising quality as a function of diffusion time.

The aggregate NELBO can hide a weak prior-to-data transition because easy,
low-noise states dominate denoising accuracy.  This evaluator reports exact
NELBO together with clean-token CE, entropy, confidence, and accuracy in fixed
time buckets.  It is a diagnostic, not autoregressive BPB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.duo import DuoSchedule, duo_nelbo_token_loss
from pretraining.byte_diffusion.duo_model import DuoModel
from pretraining.byte_diffusion.readiness import (
    DUO_CANONICAL_VALIDATION_BRANCHES,
    DUO_CANONICAL_VALIDATION_CANVAS_LENGTH,
    DUO_DATA_BRANCH_SPAN_LENGTH,
    DUO_DATA_REQUIRED_BRANCH_BYTES,
    duo_geometry_contract,
)
from pretraining.byte_diffusion.training import load_data_directory, model_config_from_dict
from scripts.train_byte_duo import _prepared_validation_ledger, source_provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-rows", type=int, default=2_048)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--ledger-seed", type=int, default=11_337)
    parser.add_argument("--time-buckets", type=int, default=8)
    parser.add_argument(
        "--native-training-geometry",
        action="store_true",
        help=(
            "diagnostic only: use the checkpoint's train geometry instead of "
            "the canonical comparable 512x8 geometry"
        ),
    )
    return parser.parse_args()


def _bucket_index(times: torch.Tensor, buckets: int) -> torch.Tensor:
    if buckets <= 0:
        raise ValueError("time bucket count must be positive")
    if times.ndim != 1 or not times.is_floating_point():
        raise ValueError("diffusion times must be a floating-point vector")
    return torch.floor(times * buckets).to(torch.long).clamp_(0, buckets - 1)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Byte-Duo time diagnostics must run through mlq on CUDA")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable result {args.output}")
    if min(
        args.validation_rows,
        args.validation_batch_size,
        args.time_buckets,
    ) <= 0:
        raise ValueError("diagnostic dimensions must be positive")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "byte_duo_checkpoint/v2":
        raise ValueError("time diagnostics require a Byte-Duo checkpoint")
    provenance = source_provenance()
    if payload.get("source_sha256") != provenance["sha256"]:
        raise ValueError("diagnostic source differs from checkpoint provenance")
    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError("Byte-Duo checkpoint omitted its training contract")
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
    manifest = json.loads((args.data_path / "manifest.json").read_text())
    data_sha256 = manifest.get("payload_sha256")
    if data_sha256 != payload.get("dataset_payload_sha256"):
        raise ValueError("diagnostic dataset differs from checkpoint provenance")

    config = model_config_from_dict(payload["model_config"])
    schedule = DuoSchedule(float(training["schedule_eps"]))
    model = DuoModel(config, schedule_eps=schedule.eps)
    model.load_state_dict(payload["model"], strict=True)
    model = model.to("cuda").eval()
    execution_model = torch.compile(model, dynamic=True, fullgraph=False)
    _, _, validation = load_data_directory(
        args.data_path,
        chunk_size=int(training["chunk_size"]),
        recipe="blt_d",
        required_branch_bytes=DUO_DATA_REQUIRED_BRANCH_BYTES,
        branch_span_length=DUO_DATA_BRANCH_SPAN_LENGTH,
        validation_chunk_limit=args.validation_rows,
        expected_payload_sha256=str(data_sha256),
    )
    evaluated_rows = min(args.validation_rows, len(validation))
    ledger = _prepared_validation_ledger(
        model,
        validation,
        rows=evaluated_rows,
        batch_size=args.validation_batch_size,
        patch_stride=config.patch_stride,
        canvas_length=eval_canvas,
        branches=eval_branches,
        seed=args.ledger_seed,
        schedule=schedule,
        device=torch.device("cuda"),
        row_indices=np.arange(evaluated_rows, dtype=np.int64),
    )

    # target count, changed count, NELBO, CE, correct count, target-prob sum,
    # entropy sum, special-probability sum
    totals = torch.zeros(
        (args.time_buckets, 8), dtype=torch.float64, device="cuda"
    )
    for item in ledger:
        batch = item.batch
        prepared = item.prepared
        rows = batch.clean_ids.shape[0]
        branches = prepared.branches
        canvas = prepared.selection.valid.shape[-1]
        noisy = prepared.corruption.ids.view(rows, branches, canvas)
        branch_times = prepared.corruption.t.view(rows, branches)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = execution_model(
                batch.clean_ids,
                batch.clean_valid,
                batch.document_ids,
                batch.positions,
                noisy,
                prepared.selection.valid,
                prepared.selection.starts,
                branch_times,
                attention_metadata=batch.attention_metadata,
                local_block_mask_metadata=prepared.local_block_mask_metadata,
                global_block_mask_metadata=prepared.global_block_mask_metadata,
                return_clean_logits=False,
            )
        logits = output.branch_logits.flatten(0, 1)[
            ..., : config.duo_diffusion_atoms
        ].float()
        corruption = prepared.corruption
        active = corruption.active
        safe_targets = torch.where(active, corruption.targets, 0)
        safe_noisy = torch.where(active, corruption.ids, 0)
        log_probabilities = F.log_softmax(logits, dim=-1)
        probabilities = log_probabilities.exp()
        token_ce = -log_probabilities.gather(
            -1, safe_targets[..., None]
        ).squeeze(-1)
        token_nelbo = duo_nelbo_token_loss(
            logits,
            safe_noisy,
            safe_targets,
            corruption.alpha,
            -(1.0 - schedule.eps),
            clean_atoms=config.duo_diffusion_atoms,
        )
        target_probability = probabilities.gather(
            -1, safe_targets[..., None]
        ).squeeze(-1)
        entropy = -(probabilities * log_probabilities).sum(-1)
        special_probability = probabilities[..., 256:].sum(-1)
        correct = logits.argmax(-1).eq(safe_targets)
        # Aggregate every active token in one device-side segmented reduction.
        # A Python loop over time bins repeatedly materializes canvas-sized
        # masks and launches eight independent reduction groups per batch.
        token_bucket = _bucket_index(
            corruption.t, args.time_buckets
        )[:, None].expand_as(active)
        flat_active = active.reshape(-1)
        flat_bucket = token_bucket.reshape(-1)[flat_active]
        contributions = torch.stack(
            (
                torch.ones_like(token_nelbo),
                corruption.changed.to(token_nelbo.dtype),
                token_nelbo,
                token_ce,
                correct.to(token_nelbo.dtype),
                target_probability,
                entropy,
                special_probability,
            ),
            dim=-1,
        ).reshape(-1, 8)[flat_active]
        totals.index_add_(0, flat_bucket, contributions.double())

    host = totals.cpu()
    records: list[dict[str, object]] = []
    for index, values in enumerate(host):
        targets = int(values[0])
        if targets == 0:
            raise ValueError(f"time bucket {index} received no targets")
        records.append(
            {
                "bucket": index,
                "time_start_inclusive": index / args.time_buckets,
                "time_stop_exclusive": (index + 1) / args.time_buckets,
                "targets": targets,
                "changed_targets": int(values[1]),
                "changed_rate": float(values[1] / targets),
                "conditional_canvas_nelbo_nats_per_atom": float(values[2] / targets),
                "clean_token_ce_nats_per_atom": float(values[3] / targets),
                "clean_token_ce_bits_per_atom": float(
                    values[3] / targets / math.log(2.0)
                ),
                "denoising_accuracy": float(values[4] / targets),
                "mean_target_probability": float(values[5] / targets),
                "mean_prediction_entropy_nats": float(values[6] / targets),
                "mean_special_probability": float(values[7] / targets),
            }
        )
    report = {
        "schema": "byte_duo_time_bucket_diagnostic/v1",
        "metric_semantics": (
            "native_geometry_conditional_canvas_diagnostics_not_ar_bpb"
            if args.native_training_geometry
            else "canonical_512x8_conditional_canvas_diagnostics_not_ar_bpb"
        ),
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
        "ledger_seed": args.ledger_seed,
        "time_buckets": args.time_buckets,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print("byte_duo_time_buckets " + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
