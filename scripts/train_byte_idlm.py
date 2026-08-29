#!/usr/bin/env python3
"""Train the standalone reference-aligned byte I-DLM cell.

GPU invocations are model workloads and must be submitted through ``mlq``::

    mlq submit --name byte_idlm_exact_2k --cwd "$PWD" --max-parallel-runs 1 -- \
      python3 scripts/ablation.py --steps 2000 --name byte_idlm_exact_2k \
      --script scripts/train_byte_idlm.py
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
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

from pretraining.byte_diffusion.data import DeterministicChunkCursor
from pretraining.byte_diffusion.idlm_model import IDLMModel, IDLMModelConfig
from pretraining.byte_diffusion.readiness import (
    diagnostic_cadence_contract,
    materialize_training_diagnostics,
    validate_architecture_readiness,
)
from pretraining.byte_diffusion.training import load_data_directory
from pretraining.byte_diffusion.training_idlm import (
    IDLMTrainer,
    IDLMTrainingConfig,
    format_train_metric,
    format_validation_metric,
)


def _positive_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _stride_curriculum_env(iterations: int) -> tuple[tuple[int, int], ...]:
    """Parse explicit ``start:stride`` entries from the run contract."""

    raw = os.environ.get("BYTE_IDLM_STRIDE_CURRICULUM", "0:4")
    try:
        schedule = tuple(
            (int(start), int(stride))
            for item in raw.split(",")
            for start, stride in (item.split(":"),)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "BYTE_IDLM_STRIDE_CURRICULUM must be comma-separated start:stride pairs"
        ) from error
    # IDLMTrainingConfig performs the authoritative topology validation.
    if any(start >= iterations for start, _ in schedule):
        raise ValueError("stride curriculum contains a start beyond this run")
    return schedule


def _local_imports(path: Path) -> tuple[Path, ...]:
    tree = ast.parse(path.read_text(), filename=str(path))
    candidates: set[Path] = set()

    def add_module(parts: list[str]) -> None:
        if not parts:
            return
        module = REPO_ROOT.joinpath(*parts)
        for candidate in (module.with_suffix(".py"), module / "__init__.py"):
            if candidate.is_file():
                candidates.add(candidate.resolve())

    package = list(path.relative_to(REPO_ROOT).with_suffix("").parts[:-1])
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add_module(alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package) - (node.level - 1)
                if keep < 0:
                    raise ValueError(f"invalid relative import in {path}")
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
    pending = [
        (REPO_ROOT / "scripts" / "train_byte_idlm.py").resolve(),
    ]
    observed: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in observed:
            continue
        if not path.is_relative_to(REPO_ROOT):
            raise ValueError(f"I-DLM training source escaped repository: {path}")
        observed.add(path)
        pending.extend(
            imported for imported in _local_imports(path) if imported not in observed
        )
    return tuple(
        sorted(observed, key=lambda path: path.relative_to(REPO_ROOT).as_posix())
    )


def source_provenance() -> dict[str, object]:
    """Hash the deterministic transitive source closure of this training cell."""

    digest = hashlib.sha256()
    files: dict[str, str] = {}
    for path in _training_source_paths():
        relative = path.relative_to(REPO_ROOT).as_posix()
        payload = path.read_bytes()
        file_hash = hashlib.sha256(payload).hexdigest()
        files[relative] = file_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_hash))
    return {
        "schema": "byte_idlm_source_provenance/v1",
        "sha256": digest.hexdigest(),
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(
            os.environ.get("BYTE_DIFFUSION_DATA_PATH", "data/byte_diffusion_aligned_v5")
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--cpu-reference", action="store_true")
    parser.add_argument("--print-provenance", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provenance = source_provenance()
    if args.print_provenance:
        print(json.dumps(provenance, indent=2, sort_keys=True))
        return

    cpu_reference = args.cpu_reference or os.environ.get(
        "BYTE_IDLM_ALLOW_CPU_REFERENCE", "0"
    ) == "1"
    if torch.cuda.is_available() and not cpu_reference:
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    elif cpu_reference:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA is required unless --cpu-reference is explicit")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError(
            "the standalone reference cell is single-device; add measured DDP only "
            "after the exact objective wins its 2k ablation"
        )

    iterations = _positive_env("ITERATIONS", 2_000)
    stride_curriculum = _stride_curriculum_env(iterations)
    trained_stride = stride_curriculum[-1][1]
    expected_source = os.environ.get("BYTE_IDLM_EXPECTED_SOURCE_SHA256")
    if iterations >= 2_000 and expected_source is None:
        raise ValueError("2k I-DLM runs require BYTE_IDLM_EXPECTED_SOURCE_SHA256")
    if expected_source is not None and provenance["sha256"] != expected_source:
        raise ValueError(
            "I-DLM source differs from its pinned run contract: "
            f"expected {expected_source}, observed {provenance['sha256']}"
        )

    run_id = os.environ.get("RUN_ID", "byte_idlm")
    seed = int(os.environ.get("SEED", "1337"))
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")

    variant = os.environ.get("BYTE_IDLM_VARIANT", "exact_full_prefix")
    if variant not in {"exact_full_prefix", "blt_window512"}:
        raise ValueError(f"unknown BYTE_IDLM_VARIANT {variant!r}")
    if args.tiny or os.environ.get("BYTE_IDLM_TINY", "0") == "1":
        model_config = IDLMModelConfig.tiny(
            clean_prefix_window=512 if variant == "blt_window512" else None,
            block_size=trained_stride,
        )
    elif variant == "exact_full_prefix":
        model_config = IDLMModelConfig(block_size=trained_stride)
    elif variant == "blt_window512":
        model_config = IDLMModelConfig.blt_window512(block_size=trained_stride)
    microbatch = _positive_env("BYTE_IDLM_MICROBATCH", 1)
    config = IDLMTrainingConfig(
        iterations=iterations,
        batch_size=microbatch,
        global_batch_size=_positive_env("BYTE_IDLM_GLOBAL_BATCH", 249),
        validation_batch_size=_positive_env("VAL_BATCH_SIZE", microbatch),
        validation_rows=_positive_env("VAL_TOKENS", 256),
        learning_rate=float(os.environ.get("LEARNING_RATE", "3e-4")),
        weight_decay=float(os.environ.get("WEIGHT_DECAY", "0.1")),
        warmdown_iters=int(os.environ.get("WARMDOWN_ITERS", "1200")),
        seed=seed,
        compile_model=os.environ.get("BYTE_IDLM_COMPILE", "1") == "1",
        stride_curriculum=stride_curriculum,
    )
    if iterations >= 2_000 and config.global_batch_size != 249:
        raise ValueError("matched 2k I-DLM runs require global batch 249")
    chunk_size = _positive_env("BYTE_DIFFUSION_CHUNK_SIZE", 8_192)
    if iterations >= 2_000 and (
        device.type != "cuda"
        or args.tiny
        or os.environ.get("BYTE_IDLM_TINY", "0") == "1"
        or variant != "exact_full_prefix"
        or model_config != IDLMModelConfig()
        or not config.compile_model
        or config.validation_rows != 256
        or config.stride_curriculum != ((0, 4),)
        or chunk_size != 8_192
    ):
        raise ValueError("2k I-DLM runs require the benchmarked production contract")
    expected_data = os.environ.get("BYTE_DIFFUSION_EXPECTED_DATA_SHA256")
    if iterations >= 2_000 and expected_data is None:
        raise ValueError("2k I-DLM runs require BYTE_DIFFUSION_EXPECTED_DATA_SHA256")
    manifest, train_dataset, validation_dataset = load_data_directory(
        args.data_path,
        chunk_size=chunk_size,
        recipe="causal_only",
        validation_chunk_limit=min(config.validation_rows, 2_048),
        expected_payload_sha256=expected_data,
    )
    if (
        manifest.output_size != model_config.output_size
        or manifest.mask_id != model_config.mask_id
        or manifest.pad_id != model_config.pad_id
        or manifest.eot_id != model_config.eot_id
    ):
        raise ValueError("I-DLM model and atomic data vocabularies differ")
    val_every = _positive_env("VAL_LOSS_EVERY", 20)
    log_every = _positive_env("TRAIN_LOG_EVERY", 10)
    readiness_path_value = os.environ.get("BYTE_IDLM_READINESS_REPORT")
    if iterations >= 2_000 and readiness_path_value is None:
        raise ValueError("2k I-DLM runs require BYTE_IDLM_READINESS_REPORT")
    readiness_evidence = (
        validate_architecture_readiness(
            Path(readiness_path_value),
            architecture="idlm",
            source_sha256=str(provenance["sha256"]),
            dataset_payload_sha256=str(expected_data),
            global_batch_size=config.global_batch_size,
            microbatch_size=config.batch_size,
            validation_batch_size=config.validation_batch_size,
            model_config=asdict(model_config),
            workload={
                "block_size": model_config.block_size,
                "diagnostic_cadence": diagnostic_cadence_contract(
                    log_every=log_every,
                    validation_every=val_every,
                ),
                "objective": "proposal_plus_clean_exact_optimizer_step_balance",
            },
            runtime={
                "compiled": config.compile_model,
                "device_type": device.type,
                "validation_rows": config.validation_rows,
                "world_size": 1,
            },
        )
        if readiness_path_value is not None
        else None
    )

    cursor = DeterministicChunkCursor(train_dataset, seed=seed, shuffle=True)
    trainer = IDLMTrainer(
        IDLMModel(model_config),
        train_dataset,
        validation_dataset,
        cursor,
        config,
        device=device,
        atomic_manifest=manifest,
    )
    try:
        checkpoint = args.checkpoint or (
            REPO_ROOT / "ablation_results" / run_id / "checkpoint.pt"
        )
        if args.resume is not None:
            trainer.load_checkpoint(args.resume)
        checkpoint_policy = RecoveryCheckpointPolicy(
            float(
                os.environ.get(
                    "BYTE_IDLM_CHECKPOINT_INTERVAL_SECONDS", "480"
                )
            )
        )

        result_directory = REPO_ROOT / "ablation_results" / run_id
        result_directory.mkdir(parents=True, exist_ok=True)
        contract = {
            "schema": "byte_idlm_run/v1",
            "run_id": run_id,
            "variant": variant,
            "model": asdict(model_config),
            "parameter_count": trainer.model.parameter_count(),
            "training": asdict(config),
            "training_diagnostic_cadence": diagnostic_cadence_contract(
                log_every=log_every,
                validation_every=val_every,
            ),
            "checkpoint_interval_seconds": (
                checkpoint_policy.interval_seconds
            ),
            "serving": {
                "trained_stride": model_config.block_size,
                "require_exact_trained_stride": True,
                "stride_curriculum": [list(item) for item in stride_curriculum],
                "promotion_metric": "gsm8k_exact_match_generation_accuracy",
                "ar_anchor_metric": "val_ar_anchor_bpb",
            },
            "atomic_manifest_sha256": manifest.sha256,
            "dataset_payload_sha256": expected_data,
            "source": provenance,
            "readiness_evidence": readiness_evidence,
        }
        (result_directory / "contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n"
        )
        print(
            "byte_idlm_contract " + json.dumps(contract, sort_keys=True), flush=True
        )

        if trainer.completed_steps == 0:
            metric = trainer.validate()
            print(
                format_validation_metric(
                    0, iterations, metric, trainer.training_time_ms
                ),
                flush=True,
            )
        while trainer.completed_steps < iterations:
            next_step = trainer.completed_steps + 1
            materialize = materialize_training_diagnostics(
                next_step,
                iterations,
                log_every=log_every,
                validation_every=val_every,
            )
            metric = trainer.run_update(materialize_metrics=materialize)
            if metric is not None and (
                metric.step % log_every == 0 or metric.step == iterations
            ):
                print(format_train_metric(metric, iterations), flush=True)
            if next_step % val_every == 0 or next_step == iterations:
                validation = trainer.validate()
                print(
                    format_validation_metric(
                        next_step, iterations, validation, trainer.training_time_ms
                    ),
                    flush=True,
                )
            if checkpoint_policy.due() or (
                next_step == iterations
                and checkpoint_policy.terminal_due(next_step)
            ):
                trainer.save_checkpoint(
                    checkpoint,
                    extra={
                        "run_id": run_id,
                        "source_sha256": provenance["sha256"],
                        "dataset_payload_sha256": expected_data,
                    },
                )
                checkpoint_policy.committed(next_step)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
