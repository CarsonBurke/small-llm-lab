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
import ast
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import torch
from checkpointing import RecoveryCheckpointPolicy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.config import (
    FAST_BLT_ENTROPY_B4_COMPLETE_PRESETS,
    ByteDiffusionConfig,
    fast_blt_complete_model_config,
    model_config_from_env,
)
from pretraining.byte_diffusion.data import DeterministicChunkCursor
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.readiness_fast_blt import (
    FastBltGeometry,
    require_fast_blt_allocator_environment,
    validate_fast_blt_readiness,
)
from pretraining.byte_diffusion.training import (
    PRODUCTION_PRESETS,
    ByteDiffusionTrainer,
    DeterministicSubsetChunkDataset,
    DistributedContext,
    TrainingRunConfig,
    format_train_metric,
    format_validation_metric,
    load_data_directory,
)


def _local_imports(path: Path) -> tuple[Path, ...]:
    """Resolve repository-local imports without importing executable modules."""

    tree = ast.parse(path.read_text(), filename=str(path))
    candidates: set[Path] = set()

    def add_package_initializers(target: Path) -> None:
        parent = target.parent
        while parent != REPO_ROOT and parent.is_relative_to(REPO_ROOT):
            initializer = parent / "__init__.py"
            if initializer.is_file():
                candidates.add(initializer.resolve())
            parent = parent.parent

    def add_module(parts: list[str]) -> None:
        if not parts:
            return
        module_path = REPO_ROOT.joinpath(*parts)
        for candidate in (module_path.with_suffix(".py"), module_path / "__init__.py"):
            if candidate.is_file():
                candidates.add(candidate.resolve())
                add_package_initializers(candidate)

    relative = path.relative_to(REPO_ROOT).with_suffix("")
    package = list(relative.parts[:-1])
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add_module(alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package) - (node.level - 1)
                if keep < 0:
                    raise ValueError(f"invalid relative import in {path}: {node.module}")
                base = package[:keep]
            else:
                base = []
            module = [] if node.module is None else node.module.split(".")
            add_module(base + module)
            for alias in node.names:
                if alias.name != "*":
                    add_module(base + module + alias.name.split("."))
    return tuple(sorted(candidates))


def _training_source_paths() -> tuple[Path, ...]:
    """Return the deterministic transitive source closure of a training run."""

    pending = [
        (REPO_ROOT / "scripts" / "ablation.py").resolve(),
        (REPO_ROOT / "scripts" / "train_byte_diffusion.py").resolve(),
    ]
    observed: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in observed:
            continue
        if not path.is_relative_to(REPO_ROOT):
            raise ValueError(f"training source escaped the repository: {path}")
        observed.add(path)
        pending.extend(
            imported for imported in _local_imports(path) if imported not in observed
        )
    return tuple(
        sorted(observed, key=lambda item: item.relative_to(REPO_ROOT).as_posix())
    )


def training_source_provenance() -> dict[str, object]:
    """Fingerprint the transitive source closure that defines a training cell."""

    paths = _training_source_paths()
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
        "schema": "byte_diffusion_source_provenance/v2",
        "sha256": digest.hexdigest(),
        "files": files,
    }


def expected_model_config_for_run(run: TrainingRunConfig) -> ByteDiffusionConfig:
    """Resolve the exact model identity required by a production run."""

    if run.preset in FAST_BLT_ENTROPY_B4_COMPLETE_PRESETS:
        assert run.preset is not None
        return fast_blt_complete_model_config(run.preset)
    return ByteDiffusionConfig()


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
    parser.add_argument(
        "--fast-blt-readiness-report",
        type=Path,
        default=(
            Path(os.environ["BYTE_DIFFUSION_FAST_BLT_READINESS_REPORT"])
            if "BYTE_DIFFUSION_FAST_BLT_READINESS_REPORT" in os.environ
            else None
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if os.environ.get("BYTE_DIFFUSION_PRESET") in FAST_BLT_ENTROPY_B4_COMPLETE_PRESETS:
        # Validate before the first CUDA query/allocation.  Setting an allocator
        # variable here would be too late to establish a process-start contract.
        require_fast_blt_allocator_environment()
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
        expected_model_config = expected_model_config_for_run(run)
        if (
            run.preset in PRODUCTION_PRESETS
            and model_config != expected_model_config
        ):
            raise ValueError(
                f"preset {run.preset!r} requires its exact production model config"
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
            require_challenge_validation=run.preset in PRODUCTION_PRESETS,
            expected_payload_sha256=expected_data_sha256,
            expected_patching_policy=run.patching_policy,
        )
        dataset_manifest = json.loads(
            (args.data_path / "manifest.json").read_text()
        )
        dataset_payload_sha256 = dataset_manifest.get("payload_sha256")
        dataset_provenance = {
            "payload_sha256": dataset_payload_sha256,
            "source_manifests": dataset_manifest.get("source_manifests", []),
        }
        if run.preset in PRODUCTION_PRESETS and not expected_data_sha256:
            raise ValueError(
                f"preset {run.preset!r} requires "
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
        readiness_evidence: dict[str, object] | None = None
        if run.preset in FAST_BLT_ENTROPY_B4_COMPLETE_PRESETS:
            if args.fast_blt_readiness_report is None:
                raise ValueError(
                    "the complete Fast-BLT preset requires "
                    "--fast-blt-readiness-report"
                )
            source_manifests = dataset_manifest.get("source_manifests")
            patching = getattr(train_chunks, "patching", None)
            if (
                not isinstance(source_manifests, list)
                or len(source_manifests) != 1
                or not isinstance(source_manifests[0], dict)
                or patching is None
            ):
                raise ValueError(
                    "complete Fast-BLT data omitted source or patcher identity"
                )
            from scripts.benchmark_byte_diffusion_fast_blt import (
                build_benchmark_run_config,
            )

            selected_microbatch = 32
            selected_run = build_benchmark_run_config(
                run, selected_microbatch
            )
            readiness_evidence = validate_fast_blt_readiness(
                args.fast_blt_readiness_report,
                source_sha256=str(source_provenance["sha256"]),
                dataset_payload_sha256=str(dataset_payload_sha256),
                source_manifest_sha256=str(source_manifests[0]["sha256"]),
                atomic_manifest_sha256=manifest.sha256,
                train_dataset_sha256=str(
                    getattr(train_chunks, "dataset_sha256", "")
                ),
                validation_dataset_sha256=str(
                    getattr(validation_chunks, "dataset_sha256", "")
                ),
                patcher_sha256=str(patching.artifact_sha256),
                max_patch_size=int(patching.max_patch_size),
                model_config=model_config,
                production_run_config=run,
                selected_run_config=selected_run,
                geometry=FastBltGeometry(),
                candidate_microbatches=(selected_microbatch,),
                selected_microbatch=selected_microbatch,
            )
            dataset_provenance = {
                **dataset_provenance,
                "fast_blt_readiness": readiness_evidence,
            }
        cursor = DeterministicChunkCursor(train_chunks, seed=run.seed, shuffle=True)
        effective_global_batch = run.global_batch_size or (
            run.microbatch_per_rank
            * run.gradient_accumulation
            * distributed.world_size
        )
        planned_rows = run.iterations * effective_global_batch
        if run.preset in PRODUCTION_PRESETS and len(train_chunks) < planned_rows:
            raise ValueError(
                f"preset {run.preset!r} needs at least {planned_rows:,} "
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
            source_provenance=source_provenance,
        )
        validation_scope = (
            "proxy"
            if isinstance(validation_chunks, DeterministicSubsetChunkDataset)
            else "challenge"
        )
        checkpoint = args.checkpoint or (
            REPO_ROOT / "ablation_results" / run.run_id / "checkpoint.pt"
        )
        checkpoint_policy = RecoveryCheckpointPolicy(
            float(
                os.environ.get(
                    "BYTE_DIFFUSION_CHECKPOINT_INTERVAL_SECONDS", "480"
                )
            )
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
                        "checkpoint_interval_seconds": (
                            checkpoint_policy.interval_seconds
                        ),
                        "causal_only_control": run.recipe == "causal_only",
                        "fast_blt_readiness": readiness_evidence,
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
            # Recovery is independent of validation and is checked at every
            # completed exact-resume update boundary.
            checkpoint_due = (
                checkpoint_policy.due()
                if distributed.is_primary
                else False
            )
            if distributed.world_size > 1:
                due_flag = torch.tensor(
                    int(checkpoint_due), device=device, dtype=torch.int32
                )
                torch.distributed.broadcast(due_flag, src=0)
                checkpoint_due = bool(due_flag.item())
            if checkpoint_due or (
                step == run.iterations
                and checkpoint_policy.terminal_due(step)
            ):
                trainer.save_checkpoint(checkpoint)
                checkpoint_policy.committed(step)
    finally:
        distributed.close()


if __name__ == "__main__":
    main()
