#!/usr/bin/env python3
"""Scratch-train the byte-diffusion family.

GPU usage must be queued, for example:

    mlq submit --name bd_fast_blt_b4_2k --cwd "$PWD" --max-parallel-runs 1 -- \
      python3 scripts/ablation.py --steps 2000 --name bd_fast_blt_b4_2k \
      --script scripts/train_byte_diffusion.py

For multi-GPU final validation, submit ``torchrun`` itself through ``mlq``.
This script never launches a child GPU workload.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.config import ByteDiffusionConfig, model_config_from_env
from pretraining.byte_diffusion.data import DeterministicChunkCursor
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.training import (
    CANONICAL_PRESET,
    ByteDiffusionTrainer,
    DeterministicSubsetChunkDataset,
    DistributedContext,
    TrainingRunConfig,
    format_train_metric,
    format_validation_metric,
    load_data_directory,
)


def training_source_provenance() -> dict[str, object]:
    """Fingerprint every runtime source file that defines a training cell."""

    paths = sorted(
        [
            *(REPO_ROOT / "pretraining" / "byte_diffusion").glob("*.py"),
            REPO_ROOT / "scripts" / "ablation.py",
            REPO_ROOT / "scripts" / "train_byte_diffusion.py",
        ],
        key=lambda path: path.relative_to(REPO_ROOT).as_posix(),
    )
    digest = hashlib.sha256()
    files: dict[str, str] = {}
    for path in paths:
        relative = path.relative_to(REPO_ROOT).as_posix()
        file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        files[relative] = file_sha256
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256))
    return {
        "schema": "byte_diffusion_source_provenance/v1",
        "sha256": digest.hexdigest(),
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(os.environ.get("BYTE_DIFFUSION_DATA_PATH", "data/byte_diffusion")),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Update-boundary checkpoint path; defaults under ablation_results/RUN_ID.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=Path(os.environ["BYTE_DIFFUSION_RESUME"])
        if "BYTE_DIFFUSION_RESUME" in os.environ
        else None,
    )
    parser.add_argument("--cpu-reference", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cpu_reference = args.cpu_reference or os.environ.get(
        "BYTE_DIFFUSION_ALLOW_CPU_REFERENCE", "0"
    ) == "1"
    if torch.cuda.is_available() and not cpu_reference:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device("cuda", local_rank)
    elif cpu_reference:
        device = torch.device("cpu")
    else:
        raise RuntimeError(
            "CUDA is required; CPU execution needs --cpu-reference and is not "
            "experimental evidence"
        )
    distributed = DistributedContext.from_environment(device.type)
    try:
        run = TrainingRunConfig.from_env()
        source_provenance = training_source_provenance()
        expected_source_sha256 = os.environ.get(
            "BYTE_DIFFUSION_EXPECTED_SOURCE_SHA256"
        )
        if run.iterations >= 2_000 and expected_source_sha256 is None:
            raise ValueError(
                "2k and longer runs require "
                "BYTE_DIFFUSION_EXPECTED_SOURCE_SHA256"
            )
        if (
            expected_source_sha256 is not None
            and source_provenance["sha256"] != expected_source_sha256
        ):
            raise ValueError(
                "training source differs from the pinned run contract: "
                f"expected {expected_source_sha256}, "
                f"observed {source_provenance['sha256']}"
            )
        if device.type == "cpu":
            run = replace(
                run,
                attention_policy="dense_reference",
                allow_cpu_reference=True,
            )
        random.seed(run.seed)
        torch.manual_seed(run.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(run.seed)
            torch.set_float32_matmul_precision("high")

        model_config = model_config_from_env(
            tiny=(
                args.tiny
                or os.environ.get("BYTE_DIFFUSION_TINY", "0") == "1"
            )
        )
        if run.preset == CANONICAL_PRESET and model_config != ByteDiffusionConfig():
            raise ValueError(
                f"preset {CANONICAL_PRESET!r} requires the production model config"
            )
        chunk_size = int(
            os.environ.get(
                "BYTE_DIFFUSION_CHUNK_SIZE", "8192"
            )
        )
        expected_data_sha256 = os.environ.get(
            "BYTE_DIFFUSION_EXPECTED_DATA_SHA256"
        )
        if run.iterations >= 2_000 and not expected_data_sha256:
            raise ValueError(
                "2k and longer runs require BYTE_DIFFUSION_EXPECTED_DATA_SHA256"
            )
        manifest, train_chunks, validation_chunks = load_data_directory(
            args.data_path,
            chunk_size=chunk_size,
            recipe=run.recipe,
            required_branch_bytes=run.corruption.corrupted_positions_per_row,
            branch_span_length=run.corruption.canvas_length,
            validation_chunk_limit=run.validation_chunks,
            require_challenge_validation=run.preset == CANONICAL_PRESET,
            expected_payload_sha256=expected_data_sha256,
        )
        dataset_manifest = json.loads(
            (args.data_path / "manifest.json").read_text()
        )
        dataset_payload_sha256 = dataset_manifest.get("payload_sha256")
        dataset_provenance = {
            "payload_sha256": dataset_payload_sha256,
            "source_manifests": dataset_manifest.get("source_manifests", []),
        }
        if run.preset == CANONICAL_PRESET and not expected_data_sha256:
            raise ValueError(
                f"preset {CANONICAL_PRESET!r} requires "
                "BYTE_DIFFUSION_EXPECTED_DATA_SHA256"
            )
        if (
            expected_data_sha256 is not None
            and dataset_payload_sha256 != expected_data_sha256
        ):
            raise ValueError(
                "dataset manifest hash differs from the pinned run contract: "
                f"expected {expected_data_sha256}, observed {dataset_payload_sha256}"
            )
        if (
            manifest.output_size != model_config.vocab.output_size
            or manifest.mask_id != model_config.vocab.mask_id
            or manifest.pad_id != model_config.vocab.pad_id
            or manifest.eot_id != model_config.vocab.eot_id
        ):
            raise ValueError("data atomic manifest does not match the model vocabulary")
        cursor = DeterministicChunkCursor(train_chunks, seed=run.seed, shuffle=True)
        effective_global_batch = run.global_batch_size or (
            run.microbatch_per_rank
            * run.gradient_accumulation
            * distributed.world_size
        )
        planned_rows = run.iterations * effective_global_batch
        if run.preset == CANONICAL_PRESET and len(train_chunks) < planned_rows:
            raise ValueError(
                f"preset {CANONICAL_PRESET!r} needs at least {planned_rows:,} "
                f"unique rows before wraparound; dataset has {len(train_chunks):,}"
            )
        trainer = ByteDiffusionTrainer(
            ByteDiffusionModel(model_config),
            cursor,
            validation_chunks,
            run,
            device=device,
            distributed=distributed,
            atomic_manifest=manifest,
            dataset_provenance=dataset_provenance,
        )
        validation_scope = (
            "proxy"
            if isinstance(validation_chunks, DeterministicSubsetChunkDataset)
            else "challenge"
        )
        checkpoint = args.checkpoint or (
            REPO_ROOT / "ablation_results" / run.run_id / "checkpoint.pt"
        )
        if args.resume is not None:
            trainer.load_checkpoint(args.resume)

        if (
            os.environ.get("BYTE_DIFFUSION_GRAD_DIAGNOSTIC", "0") == "1"
            and run.recipe != "causal_only"
        ):
            diagnostic_chunks = [
                validation_chunks[index]
                for index in range(
                    min(run.microbatch_per_rank, len(validation_chunks))
                )
            ]
            interference = trainer.measure_gradient_interference(
                diagnostic_chunks
            )
            if distributed.is_primary:
                print(
                    "byte_diffusion_gradient_interference "
                    + json.dumps(asdict(interference), sort_keys=True),
                    flush=True,
                )

        if distributed.is_primary:
            source_record = (
                REPO_ROOT
                / "ablation_results"
                / run.run_id
                / "source_provenance.json"
            )
            source_record.parent.mkdir(parents=True, exist_ok=True)
            source_record.write_text(
                json.dumps(source_provenance, indent=2, sort_keys=True) + "\n"
            )
            print(
                "byte_diffusion_source_provenance "
                + json.dumps(source_provenance, sort_keys=True),
                flush=True,
            )
            print(
                "byte_diffusion_contract "
                + json.dumps(
                    {
                        "run": run.contract_dict(),
                        "model": model_config.to_dict(),
                        "data_manifest_sha256": manifest.sha256,
                        "dataset_payload_sha256": dataset_payload_sha256,
                        "train_dataset_sha256": getattr(
                            train_chunks, "dataset_sha256", None
                        ),
                        "train_exposure": getattr(
                            train_chunks, "exposure_summary", None
                        ),
                        "planned_training_rows": planned_rows,
                        "train_chunks": len(train_chunks),
                        "validation_chunks": len(validation_chunks),
                        "world_size": distributed.world_size,
                        "initialization_kind": run.initialization_kind,
                        "resumed": args.resume is not None,
                        "causal_only_control": run.recipe == "causal_only",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if trainer.completed_steps == 0:
            validation = trainer.validate()
            if distributed.is_primary:
                print(
                    format_validation_metric(
                        0,
                        run.iterations,
                        validation,
                        trainer.training_time_ms,
                        scope=validation_scope,
                    ),
                    flush=True,
                )
        while trainer.completed_steps < run.iterations:
            next_step = trainer.completed_steps + 1
            log_update = (
                next_step % run.train_log_every == 0
                or next_step % run.val_loss_every == 0
                or next_step == run.iterations
            )
            metrics = trainer.run_update(materialize_metrics=log_update)
            step = trainer.completed_steps
            if distributed.is_primary and (
                step % run.train_log_every == 0 or step == run.iterations
            ):
                if metrics is None:
                    raise AssertionError("logging update omitted its metrics")
                print(format_train_metric(metrics, run.iterations), flush=True)
            if (
                step % run.val_loss_every == 0
                or step == run.iterations
            ):
                validation = trainer.validate()
                if distributed.is_primary:
                    print(
                        format_validation_metric(
                            step,
                            run.iterations,
                            validation,
                            trainer.training_time_ms,
                            scope=validation_scope,
                        ),
                        flush=True,
                    )
                trainer.save_checkpoint(checkpoint)
    finally:
        distributed.close()


if __name__ == "__main__":
    main()
