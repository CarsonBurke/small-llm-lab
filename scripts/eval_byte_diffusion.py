#!/usr/bin/env python3
"""Evaluate the diffusion ELBO proxy plus a separately named AR anchor.

Queue GPU evaluation through mlq:

    mlq submit --name bd_eval --cwd "$PWD" --max-parallel-runs 1 -- \
      python3 scripts/eval_byte_diffusion.py --checkpoint CHECKPOINT \
      --data-path DATA_DIRECTORY
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.data import AtomicIdManifest, DeterministicChunkCursor
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.training import (
    CHECKPOINT_SCHEMA,
    ByteDiffusionTrainer,
    DistributedContext,
    format_validation_metric,
    load_data_directory,
    model_config_from_dict,
    run_config_from_dict,
)
from scripts.train_byte_diffusion import training_source_provenance


def validate_evaluation_provenance(
    payload: dict,
    trainer: ByteDiffusionTrainer,
    dataset_provenance: dict,
    *,
    full_validation: bool,
) -> None:
    """Fail closed before labeling a checkpoint metric proxy or challenge."""

    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("evaluation checkpoint schema is unsupported")
    if payload.get("dataset_provenance") != dataset_provenance:
        raise ValueError("evaluation dataset payload/source provenance differs")
    if payload.get("source_provenance") != trainer.source_provenance:
        raise ValueError(
            "evaluation code differs from the checkpoint's pinned training source"
        )
    if payload.get("dataset_sha256") != trainer.train_cursor.dataset_sha256:
        raise ValueError("evaluation training split differs from checkpoint")
    if not full_validation and payload.get(
        "validation_sha256"
    ) != trainer.validation_sha256:
        raise ValueError("proxy validation split differs from checkpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--cpu-reference", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--full-validation",
        action="store_true",
        help="Score the complete challenge validation split instead of the saved proxy limit.",
    )
    parser.add_argument(
        "--validation-seed-offset",
        type=int,
        default=0,
        help="Offset the checkpoint seed for an independent exact-K ELBO draw.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validation_seed_offset < 0:
        raise ValueError("validation seed offset must be non-negative")
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable result {args.output}")
    cpu = args.cpu_reference
    if not cpu and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA evaluation requires mlq; use --cpu-reference only for tests"
        )
    device = (
        torch.device("cpu")
        if cpu
        else torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    )
    distributed = DistributedContext.from_environment(device.type)
    try:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("evaluation checkpoint schema is unsupported")
        model_config = model_config_from_dict(payload["model_config"])
        saved_run = run_config_from_dict(payload["run_contract"])
        run = (
            replace(
                saved_run,
                attention_policy="dense_reference",
                allow_cpu_reference=True,
            )
            if cpu
            else saved_run
        )
        chunk_size = int(
            os.environ.get(
                "BYTE_DIFFUSION_CHUNK_SIZE", "8192"
            )
        )
        manifest, train_chunks, validation_chunks = load_data_directory(
            args.data_path,
            chunk_size=chunk_size,
            recipe=run.recipe,
            required_branch_bytes=run.corruption.corrupted_positions_per_row,
            branch_span_length=run.corruption.canvas_length,
            validation_chunk_limit=(None if args.full_validation else run.validation_chunks),
            require_challenge_validation=args.full_validation,
        )
        if args.full_validation and run.recipe != "causal_only":
            # A diffusion-primary full evaluation must actually cover every
            # loaded validation row. Retaining the training-time 256-row
            # diagnostic limit and labeling it challenge scope is false.
            run = replace(
                run,
                preset=None,
                validation_chunks=len(validation_chunks),
                diffusion_validation_chunks=len(validation_chunks),
            )
        if args.validation_seed_offset:
            run = replace(
                run,
                preset=None,
                seed=run.seed + args.validation_seed_offset,
            )
        dataset_manifest = json.loads(
            (args.data_path / "manifest.json").read_text()
        )
        dataset_provenance = {
            "payload_sha256": dataset_manifest.get("payload_sha256"),
            "source_manifests": dataset_manifest.get("source_manifests", []),
        }
        checkpoint_manifest = AtomicIdManifest.from_dict(payload["atomic_manifest"])
        if checkpoint_manifest.sha256 != manifest.sha256:
            raise ValueError("evaluation data atomic manifest differs from checkpoint")
        model = ByteDiffusionModel(model_config)
        model.load_state_dict(payload["model"], strict=True)
        source_provenance = training_source_provenance()
        trainer = ByteDiffusionTrainer(
            model,
            DeterministicChunkCursor(train_chunks, seed=run.seed),
            validation_chunks,
            run,
            device=device,
            distributed=distributed,
            atomic_manifest=manifest,
            dataset_provenance=dataset_provenance,
            source_provenance=source_provenance,
        )
        validate_evaluation_provenance(
            payload,
            trainer,
            dataset_provenance,
            full_validation=args.full_validation,
        )
        metrics = trainer.validate()
        if distributed.is_primary:
            checkpoint_sha256 = hashlib.sha256(
                args.checkpoint.read_bytes()
            ).hexdigest()
            print(
                "byte_diffusion_evaluation_provenance "
                + json.dumps(
                    {
                        "checkpoint_sha256": checkpoint_sha256,
                        "checkpoint_schema": payload["schema"],
                        "training_source_sha256": source_provenance["sha256"],
                        "completed_steps": int(payload["completed_steps"]),
                        "dataset_payload_sha256": dataset_provenance[
                            "payload_sha256"
                        ],
                        "source_manifests": dataset_provenance[
                            "source_manifests"
                        ],
                        "train_dataset_sha256": trainer.train_cursor.dataset_sha256,
                        "validation_dataset_sha256": trainer.validation_sha256,
                        "validation_scope": (
                            "challenge" if args.full_validation else "proxy"
                        ),
                        "validation_seed": run.seed,
                        "validation_seed_offset": args.validation_seed_offset,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            print(
                format_validation_metric(
                    int(payload["completed_steps"]),
                    saved_run.iterations,
                    metrics,
                    float(payload.get("training_time_ms", 0.0)),
                    scope=("challenge" if args.full_validation else "proxy"),
                ),
                flush=True,
            )
            if args.output is not None:
                result = {
                    "schema": "byte_diffusion_evaluation/v2",
                    "scope": "challenge" if args.full_validation else "proxy",
                    "primary_metric": (
                        "diffusion_elbo_proxy_bpb"
                        if metrics.diffusion_elbo_proxy_bpb is not None
                        else "ar_bpb"
                    ),
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_schema": payload["schema"],
                    "training_source_sha256": source_provenance["sha256"],
                    "completed_steps": int(payload["completed_steps"]),
                    "dataset_payload_sha256": dataset_provenance[
                        "payload_sha256"
                    ],
                    "validation_dataset_sha256": trainer.validation_sha256,
                    "validation_seed": run.seed,
                    "validation_seed_offset": args.validation_seed_offset,
                    "metrics": asdict(metrics),
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(result, indent=1, sort_keys=True) + "\n"
                )
    finally:
        distributed.close()


if __name__ == "__main__":
    main()
