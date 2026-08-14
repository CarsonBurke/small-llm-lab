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
from typing import Mapping

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
from pretraining.byte_diffusion.patching import CausalEntropyPatcher
from pretraining.byte_diffusion.training import (
    CHECKPOINT_SCHEMA,
    ByteDiffusionTrainer,
    DistributedContext,
    load_data_directory,
    model_config_from_dict,
    run_config_from_dict,
)
from pretraining.byte_diffusion.variable_patching import load_dataset_patching_spec


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


def checkpoint_patcher_artifact(
    payload: Mapping[str, object], data_path: Path
) -> bytes | None:
    """Resolve the patcher only through checkpoint-pinned dataset provenance."""

    run_contract = payload.get("run_contract")
    if not isinstance(run_contract, Mapping):
        raise ValueError("checkpoint omitted its training run contract")
    policy = run_contract.get("patching_policy")
    if policy == "fixed_stride_v1":
        return None
    if policy != "causal_entropy_v1":
        raise ValueError(f"unknown checkpoint patching policy {policy!r}")
    manifest_path = data_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"entropy dataset manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, Mapping):
        raise ValueError("entropy dataset manifest must be an object")
    claimed_payload = manifest.get("payload_sha256")
    unsigned = dict(manifest)
    unsigned.pop("payload_sha256", None)
    observed_payload = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if claimed_payload != observed_payload:
        raise ValueError("entropy dataset manifest payload_sha256 mismatch")
    provenance = payload.get("dataset_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("entropy checkpoint omitted dataset provenance")
    if provenance.get("payload_sha256") != observed_payload:
        raise ValueError(
            "entropy export dataset differs from the checkpoint-pinned dataset"
        )
    spec = load_dataset_patching_spec(data_path, manifest)
    if spec.name != "causal_entropy_v1" or not spec.variable:
        raise ValueError("entropy checkpoint dataset uses the wrong patching policy")
    corruption = run_contract.get("corruption")
    if not isinstance(corruption, Mapping):
        raise ValueError("entropy checkpoint omitted its corruption contract")
    if int(corruption.get("canvas_length", -1)) != spec.max_patch_size:
        raise ValueError(
            "entropy checkpoint canvas differs from authenticated max_patch_size"
        )
    patching = manifest.get("patching")
    if not isinstance(patching, Mapping):
        raise ValueError("entropy dataset omitted its patching contract")
    artifact_info = patching.get("patcher_artifact")
    if not isinstance(artifact_info, Mapping):
        raise ValueError("entropy dataset omitted its patcher artifact")
    relative = Path(str(artifact_info.get("path", "")))
    artifact = (data_path / relative).read_bytes()
    patcher = CausalEntropyPatcher.from_bytes(artifact)
    if patcher.sha256 != spec.artifact_sha256:
        raise ValueError("entropy patcher differs from authenticated dataset artifact")
    return artifact


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
    entropy_patcher = checkpoint_patcher_artifact(payload, args.data_path)
    paths = counted_code_paths()
    code_bytes = sum(path.stat().st_size for path in paths)
    candidate = build_artifact(
        model,
        config,
        group_size=args.group_size,
        code_bytes=code_bytes,
        atomic_manifest=checkpoint_manifest,
        entropy_patcher=entropy_patcher,
    )
    candidate_metadata, _ = parse_artifact(candidate)
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
        expected_patching_policy=run.patching_policy,
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
    evaluation["encoding_plan"] = {
        "policy": candidate_metadata["encoding_policy"],
        "group_size": candidate_metadata["group_size"],
        "fallbacks": candidate_metadata["encoding_fallbacks"],
    }
    dataset_provenance = payload.get("dataset_provenance")
    if not isinstance(dataset_provenance, Mapping):
        raise ValueError("checkpoint omitted dataset provenance")
    provenance = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_step": completed_steps,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "training_dataset_sha256": train_sha256,
        "validation_dataset_sha256": validation_sha256,
        "dataset_payload_sha256": dataset_provenance.get("payload_sha256"),
        "patching_policy": run.patching_policy,
        "entropy_patcher_sha256": (
            hashlib.sha256(entropy_patcher).hexdigest()
            if entropy_patcher is not None
            else None
        ),
    }
    artifact = build_artifact(
        model,
        config,
        group_size=args.group_size,
        code_bytes=code_bytes,
        atomic_manifest=checkpoint_manifest,
        post_quantization_metrics=evaluation,
        provenance=provenance,
        entropy_patcher=entropy_patcher,
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
                "headroom_bytes": 16_000_000 - len(artifact) - code_bytes,
                "embedded_entropy_patcher_bytes": len(entropy_patcher or b""),
                "embedded_entropy_patcher_sha256": (
                    hashlib.sha256(entropy_patcher).hexdigest()
                    if entropy_patcher is not None
                    else None
                ),
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
