#!/usr/bin/env python3
"""Queued real-data CUDA preflight for reproducible compiled checkpoint resume."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.config import ByteDiffusionConfig, CorruptionConfig
from pretraining.byte_diffusion.data import DeterministicChunkCursor
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.training import (
    ByteDiffusionTrainer,
    TrainingRunConfig,
    load_data_directory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify"))
    parser.add_argument("--data-path", type=Path, default=Path("data/byte_diffusion"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ablation_results/byte_diffusion_resume_preflight"),
    )
    return parser.parse_args()


def _run_config() -> TrainingRunConfig:
    return TrainingRunConfig(
        iterations=2,
        val_loss_every=2,
        train_log_every=1,
        validation_chunks=16,
        validation_microbatch_per_rank=16,
        diffusion_validation_chunks=16,
        warmdown_iters=0,
        run_id="byte_diffusion_resume_preflight",
        recipe="canvas",
        corruption=CorruptionConfig.canvas512(),
        microbatch_per_rank=24,
        microbatch_token_budget=208_896,
        gradient_accumulation=11,
        global_batch_size=256,
        compile_model=True,
        compile_dynamic_shapes=True,
    )


def _trainer(data_path: Path) -> ByteDiffusionTrainer:
    run = _run_config()
    dataset_manifest = json.loads((data_path / "manifest.json").read_text())
    provenance = {
        "payload_sha256": dataset_manifest.get("payload_sha256"),
        "source_manifests": dataset_manifest.get("source_manifests", []),
    }
    manifest, train_chunks, validation_chunks = load_data_directory(
        data_path,
        chunk_size=8_192,
        recipe=run.recipe,
        required_branch_bytes=run.corruption.corrupted_positions_per_row,
        branch_span_length=run.corruption.canvas_length,
        validation_chunk_limit=run.validation_chunks,
        require_challenge_validation=False,
    )
    random.seed(run.seed)
    torch.manual_seed(run.seed)
    torch.cuda.manual_seed_all(run.seed)
    torch.set_float32_matmul_precision("high")
    return ByteDiffusionTrainer(
        ByteDiffusionModel(ByteDiffusionConfig()),
        DeterministicChunkCursor(train_chunks, seed=run.seed, shuffle=True),
        validation_chunks,
        run,
        device=torch.device("cuda", 0),
        atomic_manifest=manifest,
        dataset_provenance=provenance,
    )


def _stable_metrics(metrics) -> dict[str, Any]:
    result = asdict(metrics)
    result.pop("elapsed_ms")
    # Normalize tuples the same way the persisted JSON does so the fresh-process
    # comparison is about values, not Python container types.
    return json.loads(json.dumps(result))


def _assert_equal(left: Any, right: Any, path: str = "root") -> None:
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
            raise AssertionError(f"tensor mismatch at {path}")
        return
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f"mapping mismatch at {path}")
        for key in left:
            _assert_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise AssertionError(f"sequence mismatch at {path}")
        for index, (left_item, right_item) in enumerate(
            zip(left, right, strict=True)
        ):
            _assert_equal(left_item, right_item, f"{path}[{index}]")
        return
    if left != right:
        raise AssertionError(f"value mismatch at {path}: {left!r} != {right!r}")


def _assert_numerically_close(
    left: Any,
    right: Any,
    *,
    absolute_tolerance: float,
    path: str,
) -> dict[str, int | float]:
    """Require exact structure/non-floats and bounded CUDA floating drift."""

    stats: dict[str, int | float] = {
        "tensors": 0,
        "elements": 0,
        "different_elements": 0,
        "max_abs_difference": 0.0,
        "mean_abs_difference": 0.0,
    }
    absolute_sum = 0.0

    def visit(expected: Any, actual: Any, current_path: str) -> None:
        nonlocal absolute_sum
        if isinstance(expected, torch.Tensor):
            if not isinstance(actual, torch.Tensor):
                raise AssertionError(f"tensor type mismatch at {current_path}")
            if expected.dtype != actual.dtype or expected.shape != actual.shape:
                raise AssertionError(f"tensor contract mismatch at {current_path}")
            if not expected.is_floating_point():
                if not torch.equal(expected, actual):
                    raise AssertionError(f"non-floating tensor mismatch at {current_path}")
                return
            difference = (expected.float() - actual.float()).abs()
            if not torch.isfinite(difference).all():
                raise AssertionError(f"non-finite difference at {current_path}")
            maximum = float(difference.max()) if difference.numel() else 0.0
            if maximum > absolute_tolerance:
                raise AssertionError(
                    f"floating tensor drift at {current_path}: "
                    f"{maximum} > {absolute_tolerance}"
                )
            stats["tensors"] += 1
            stats["elements"] += difference.numel()
            stats["different_elements"] += int((difference != 0).sum())
            stats["max_abs_difference"] = max(
                float(stats["max_abs_difference"]), maximum
            )
            absolute_sum += float(difference.double().sum())
            return
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or expected.keys() != actual.keys():
                raise AssertionError(f"mapping mismatch at {current_path}")
            for key in expected:
                visit(expected[key], actual[key], f"{current_path}.{key}")
            return
        if isinstance(expected, (list, tuple)):
            if not isinstance(actual, type(expected)) or len(expected) != len(actual):
                raise AssertionError(f"sequence mismatch at {current_path}")
            for index, (expected_item, actual_item) in enumerate(
                zip(expected, actual, strict=True)
            ):
                visit(expected_item, actual_item, f"{current_path}[{index}]")
            return
        if expected != actual:
            raise AssertionError(
                f"value mismatch at {current_path}: {expected!r} != {actual!r}"
            )

    visit(left, right, path)
    elements = int(stats["elements"])
    stats["mean_abs_difference"] = absolute_sum / max(elements, 1)
    return stats


def _checkpoint_state(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        key: value
        for key, value in payload.items()
        if key not in {"training_time_ms"}
    }


def _git_evidence() -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
    )
    return {"git_revision": revision, "git_dirty": dirty}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("resume preflight requires queued CUDA execution")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step1_path = args.output_dir / "step1.pt"
    expected_path = args.output_dir / "expected_step2.pt"
    actual_path = args.output_dir / "resumed_step2.pt"
    metrics_path = args.output_dir / "expected_metrics.json"
    result_path = args.output_dir / "result.json"
    trainer = _trainer(args.data_path)

    if args.mode == "prepare":
        validation = trainer.validate()
        first = trainer.run_update()
        if first.microsteps != 11 or first.max_microbatch != 24:
            raise AssertionError("production update did not use the 24+tail plan")
        trainer.save_checkpoint(step1_path)
        second = trainer.run_update()
        trainer.save_checkpoint(expected_path)
        evidence = {
            **_git_evidence(),
            "validation_bpb": validation.bpb,
            "validation_sha256": trainer.validation_sha256,
            "train_dataset_sha256": trainer.train_cursor.dataset_sha256,
            "first_update": _stable_metrics(first),
            "second_update": _stable_metrics(second),
        }
        metrics_path.write_text(json.dumps(evidence, sort_keys=True) + "\n")
        print("byte_diffusion_resume_prepare " + json.dumps(evidence, sort_keys=True))
        return

    expected_metrics = json.loads(metrics_path.read_text())
    trainer.load_checkpoint(step1_path)
    resumed = trainer.run_update()
    trainer.save_checkpoint(actual_path)
    if _stable_metrics(resumed) != expected_metrics["second_update"]:
        raise AssertionError("first post-resume metrics differ from uninterrupted run")
    expected_state = _checkpoint_state(expected_path)
    actual_state = _checkpoint_state(actual_path)
    model_drift = _assert_numerically_close(
        expected_state.pop("model"),
        actual_state.pop("model"),
        absolute_tolerance=1e-4,
        path="model",
    )
    optimizer_drift = _assert_numerically_close(
        expected_state.pop("optimizer"),
        actual_state.pop("optimizer"),
        absolute_tolerance=1e-5,
        path="optimizer",
    )
    _assert_equal(expected_state, actual_state)
    bit_exact = (
        model_drift["different_elements"] == 0
        and optimizer_drift["different_elements"] == 0
    )
    result = {
        **_git_evidence(),
        "schema": "byte_diffusion_resume_preflight/v2",
        "numerically_equivalent": True,
        "bit_exact": bit_exact,
        "model_drift": model_drift,
        "optimizer_drift": optimizer_drift,
        "completed_steps": trainer.completed_steps,
        "train_dataset_sha256": trainer.train_cursor.dataset_sha256,
        "validation_sha256": trainer.validation_sha256,
        "resumed_update": _stable_metrics(resumed),
    }
    result_path.write_text(json.dumps(result, sort_keys=True) + "\n")
    print("byte_diffusion_resume_verify " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
