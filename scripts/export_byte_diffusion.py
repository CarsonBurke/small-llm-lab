#!/usr/bin/env python3
"""Export a deterministic packed-int4 byte-diffusion artifact.

This is a model workload. Queue it through mlq, for example:

    mlq submit --name bd_export --cwd "$PWD" --max-parallel-runs 1 -- \
      python3 scripts/export_byte_diffusion.py --checkpoint CHECKPOINT --output model.bdi4
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
from pretraining.byte_diffusion.export import (
    build_artifact,
    load_artifact,
    parse_artifact,
    write_artifact,
)
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.training import (
    CHECKPOINT_SCHEMA,
    ByteDiffusionTrainer,
    DistributedContext,
    load_data_directory,
    model_config_from_dict,
    run_config_from_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--cpu-reference", action="store_true")
    return parser.parse_args()


def counted_code_paths() -> tuple[Path, ...]:
    package = sorted((REPO_ROOT / "pretraining" / "byte_diffusion").glob("*.py"))
    entrypoints = [
        REPO_ROOT / "scripts" / "train_byte_diffusion.py",
        REPO_ROOT / "scripts" / "eval_byte_diffusion.py",
        REPO_ROOT / "scripts" / "export_byte_diffusion.py",
    ]
    return tuple(path for path in (*package, *entrypoints) if path.is_file())


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("export requires a canonical byte-diffusion checkpoint")
    run_contract = payload.get("run_contract")
    if not isinstance(run_contract, dict):
        raise ValueError("checkpoint omitted its training run contract")
    completed_steps = int(payload.get("completed_steps", -1))
    if completed_steps != int(run_contract.get("iterations", -2)):
        raise ValueError("final export requires a completed training schedule")
    config = model_config_from_dict(payload["model_config"])
    if "atomic_manifest" not in payload:
        raise ValueError("checkpoint omitted its atomic vocabulary manifest")
    checkpoint_manifest = AtomicIdManifest.from_dict(payload["atomic_manifest"])
    model = ByteDiffusionModel(config)
    model.load_state_dict(payload["model"], strict=True)
    paths = counted_code_paths()
    code_bytes = sum(path.stat().st_size for path in paths)
    candidate = build_artifact(
        model,
        config,
        group_size=args.group_size,
        code_bytes=code_bytes,
        atomic_manifest=checkpoint_manifest,
    )
    quantized_model = ByteDiffusionModel(config)
    load_artifact(quantized_model, candidate)
    cpu = args.cpu_reference
    if not cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA export evaluation requires mlq; use --cpu-reference only for tests")
    device = torch.device("cpu" if cpu else "cuda")
    run = run_config_from_dict(run_contract)
    if cpu:
        run = replace(
            run,
            attention_policy="dense_reference",
            allow_cpu_reference=True,
            compile_model=False,
        )
    chunk_size = int(os.environ.get("BYTE_DIFFUSION_CHUNK_SIZE", "8192"))
    data_manifest, train_chunks, validation_chunks = load_data_directory(
        args.data_path,
        chunk_size=chunk_size,
        recipe=run.recipe,
        required_branch_bytes=run.corruption.corrupted_positions_per_row,
        branch_span_length=run.corruption.canvas_length,
        validation_chunk_limit=run.validation_chunks,
    )
    if data_manifest.sha256 != checkpoint_manifest.sha256:
        raise ValueError("export data atomic manifest differs from the checkpoint")
    train_sha256 = getattr(train_chunks, "dataset_sha256", None)
    validation_sha256 = getattr(validation_chunks, "dataset_sha256", None)
    if train_sha256 != payload.get("dataset_sha256"):
        raise ValueError("export training dataset differs from the checkpoint")
    if validation_sha256 != payload.get("validation_sha256"):
        raise ValueError("export validation dataset differs from the checkpoint")
    trainer = ByteDiffusionTrainer(
        quantized_model,
        DeterministicChunkCursor(train_chunks, seed=run.seed),
        validation_chunks,
        run,
        device=device,
        distributed=DistributedContext(),
        atomic_manifest=data_manifest,
    )
    evaluation = asdict(trainer.validate())
    provenance = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_step": completed_steps,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "training_dataset_sha256": train_sha256,
        "validation_dataset_sha256": validation_sha256,
    }
    artifact = build_artifact(
        model,
        config,
        group_size=args.group_size,
        code_bytes=code_bytes,
        atomic_manifest=checkpoint_manifest,
        post_quantization_metrics=evaluation,
        provenance=provenance,
        require_evaluation=True,
    )
    digest = write_artifact(args.output, artifact)
    metadata, _ = parse_artifact(artifact)
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "artifact_bytes": len(artifact),
                "code_bytes": code_bytes,
                "complete_bytes": len(artifact) + code_bytes,
                "parameter_count": metadata["parameter_count"],
                "post_quantization_metrics": evaluation,
                "sha256": digest,
                "counted_code_files": [str(path.relative_to(REPO_ROOT)) for path in paths],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
