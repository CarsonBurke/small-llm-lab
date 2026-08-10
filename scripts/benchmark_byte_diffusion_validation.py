#!/usr/bin/env python3
"""Benchmark the production proxy-validation path on real packed data.

This is a GPU workload and must be submitted through ``mlq``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import torch
from torch._dynamo.utils import counters as dynamo_counters

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.config import ByteDiffusionConfig
from pretraining.byte_diffusion.data import DeterministicChunkCursor
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.training import (
    CHECKPOINT_SCHEMA,
    ByteDiffusionTrainer,
    DeterministicSubsetChunkDataset,
    TrainingRunConfig,
    load_data_directory,
    model_config_from_dict,
    run_config_from_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/byte_diffusion"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--expected-recipe",
        choices=("canvas", "blt_d", "causal_only"),
        required=True,
        help="Fail closed unless the resolved run uses this recipe.",
    )
    parser.add_argument("--validation-chunks", type=int, default=2_048)
    parser.add_argument("--validation-microbatch", type=int, default=64)
    parser.add_argument("--ar-validation-microbatch", type=int, default=128)
    parser.add_argument("--diffusion-validation-chunks", type=int, default=256)
    parser.add_argument("--measured-validations", type=int, default=2)
    parser.add_argument("--max-validation-seconds", type=float, default=15.0)
    parser.add_argument("--max-warmup-seconds", type=float, default=180.0)
    parser.add_argument("--expected-ar-targets", type=int)
    parser.add_argument("--expected-diffusion-targets", type=int)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-data-sha256")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def evaluation_source_sha256() -> str:
    paths = sorted(
        [
            *(REPO_ROOT / "pretraining" / "byte_diffusion").glob("*.py"),
            Path(__file__).resolve(),
        ],
        key=lambda path: path.relative_to(REPO_ROOT).as_posix(),
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(REPO_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def sample_telemetry(
    stop: threading.Event,
    power_samples: list[float],
    utilization_samples: list[float],
) -> None:
    while not stop.wait(0.1):
        observed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw,utilization.gpu",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if observed.returncode != 0:
            continue
        try:
            power, utilization = observed.stdout.splitlines()[0].split(",")
            power_samples.append(float(power.strip()))
            utilization_samples.append(float(utilization.strip()))
        except (IndexError, ValueError):
            continue


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark must be submitted through mlq")
    if min(
        args.validation_chunks,
        args.validation_microbatch,
        args.ar_validation_microbatch,
        args.diffusion_validation_chunks,
        args.measured_validations,
    ) <= 0:
        raise ValueError("validation benchmark dimensions must be positive")
    if args.diffusion_validation_chunks > args.validation_chunks:
        raise ValueError("diffusion subset cannot exceed the BPB validation set")

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.set_float32_matmul_precision("high")
    payload = None
    checkpoint_schema = None
    checkpoint_step = None
    checkpoint_sha256 = None
    source_sha256 = evaluation_source_sha256()
    if args.checkpoint is not None and args.expected_source_sha256 is None:
        raise ValueError("checkpoint evaluation requires its expected source SHA-256")
    if (
        args.expected_source_sha256 is not None
        and source_sha256 != args.expected_source_sha256
    ):
        raise ValueError(
            "evaluation source hash mismatch: "
            f"expected {args.expected_source_sha256}, observed {source_sha256}"
        )
    if args.checkpoint is not None:
        checkpoint_sha256 = sha256_file(args.checkpoint)
        if args.expected_checkpoint_sha256 is None:
            raise ValueError("checkpoint evaluation requires its expected SHA-256")
        if checkpoint_sha256 != args.expected_checkpoint_sha256:
            raise ValueError(
                "checkpoint hash mismatch: "
                f"expected {args.expected_checkpoint_sha256}, "
                f"observed {checkpoint_sha256}"
            )
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(
                f"checkpoint schema must be {CHECKPOINT_SCHEMA!r}, "
                f"observed {payload.get('schema')!r}"
            )
        model_config = model_config_from_dict(payload["model_config"])
        run = replace(
            run_config_from_dict(payload["run_contract"]),
            preset=None,
            validation_chunks=args.validation_chunks,
            validation_microbatch_per_rank=args.validation_microbatch,
            ar_validation_microbatch_per_rank=args.ar_validation_microbatch,
            diffusion_validation_chunks=args.diffusion_validation_chunks,
            compile_model=True,
            compile_dynamic_shapes=True,
        )
        checkpoint_schema = payload["schema"]
        checkpoint_step = payload.get("completed_steps")
    else:
        model_config = ByteDiffusionConfig()
        run = TrainingRunConfig.from_env(
            preset=None,
            validation_chunks=args.validation_chunks,
            validation_microbatch_per_rank=args.validation_microbatch,
            ar_validation_microbatch_per_rank=args.ar_validation_microbatch,
            diffusion_validation_chunks=args.diffusion_validation_chunks,
            compile_model=True,
            compile_dynamic_shapes=True,
        )
    if run.recipe != args.expected_recipe:
        raise ValueError(
            f"expected recipe {args.expected_recipe!r}, observed {run.recipe!r}"
        )
    manifest, train_chunks, validation_chunks = load_data_directory(
        args.data_path,
        chunk_size=8_192,
        recipe=run.recipe,
        required_branch_bytes=run.corruption.corrupted_positions_per_row,
        branch_span_length=run.corruption.canvas_length,
        validation_chunk_limit=run.validation_chunks,
        require_challenge_validation=True,
        expected_payload_sha256=args.expected_data_sha256,
    )
    dataset_payload_sha256 = json.loads(
        (args.data_path / "manifest.json").read_text()
    ).get("payload_sha256")
    checkpoint_data_sha256 = (
        payload.get("dataset_provenance", {}).get("payload_sha256")
        if payload is not None
        else None
    )
    if args.expected_data_sha256 is not None:
        if dataset_payload_sha256 != args.expected_data_sha256:
            raise ValueError(
                "dataset payload hash mismatch: "
                f"expected {args.expected_data_sha256}, "
                f"observed {dataset_payload_sha256}"
            )
    elif payload is not None:
        raise ValueError("checkpoint evaluation requires its expected data SHA-256")
    if payload is not None and checkpoint_data_sha256 != dataset_payload_sha256:
        raise ValueError(
            "checkpoint dataset provenance differs from the evaluation dataset: "
            f"checkpoint {checkpoint_data_sha256}, "
            f"evaluation {dataset_payload_sha256}"
        )
    model = ByteDiffusionModel(model_config)
    if payload is not None:
        model.load_state_dict(payload["model"], strict=True)
    trainer = ByteDiffusionTrainer(
        model,
        DeterministicChunkCursor(train_chunks, seed=run.seed),
        validation_chunks,
        run,
        device=device,
        atomic_manifest=manifest,
    )

    warmup = trainer.validate()
    dynamo_counters.clear()
    torch.cuda.reset_peak_memory_stats()
    powers: list[float] = []
    utilizations: list[float] = []
    stop = threading.Event()
    sampler = threading.Thread(
        target=sample_telemetry,
        args=(stop, powers, utilizations),
        daemon=True,
    )
    sampler.start()
    started = time.perf_counter()
    measurements = [
        trainer.validate() for _ in range(args.measured_validations)
    ]
    torch.cuda.synchronize()
    wall_seconds = (time.perf_counter() - started) / args.measured_validations
    stop.set()
    sampler.join(timeout=1.0)

    failures: list[str] = []
    warmup_seconds = warmup.elapsed_ms / 1_000
    if warmup_seconds > args.max_warmup_seconds:
        failures.append(
            f"{warmup_seconds:.3f}s cold validation exceeds "
            f"{args.max_warmup_seconds:.3f}s gate"
        )
    if wall_seconds > args.max_validation_seconds:
        failures.append(
            f"{wall_seconds:.3f}s exceeds {args.max_validation_seconds:.3f}s gate"
        )
    final = measurements[-1]
    expected_diffusion_chunks = (
        0
        if run.recipe == "causal_only"
        else min(args.diffusion_validation_chunks, len(validation_chunks))
    )
    if (
        args.expected_ar_targets is not None
        and final.ar_targets != args.expected_ar_targets
    ):
        failures.append(
            f"AR target count {final.ar_targets} differs from expected "
            f"{args.expected_ar_targets}"
        )
    if (
        args.expected_diffusion_targets is not None
        and final.diffusion_targets != args.expected_diffusion_targets
    ):
        failures.append(
            f"diffusion target count {final.diffusion_targets} differs from "
            f"expected {args.expected_diffusion_targets}"
        )
    for index, measurement in enumerate(measurements):
        if measurement.ar_targets != warmup.ar_targets:
            failures.append(
                f"measurement {index} changed AR target count from "
                f"{warmup.ar_targets} to {measurement.ar_targets}"
            )
        if measurement.diffusion_targets != warmup.diffusion_targets:
            failures.append(
                f"measurement {index} changed diffusion target count from "
                f"{warmup.diffusion_targets} to {measurement.diffusion_targets}"
            )
        if measurement.bpb != warmup.bpb:
            failures.append(
                f"measurement {index} changed deterministic BPB from "
                f"{warmup.bpb:.12f} to {measurement.bpb:.12f}"
            )
        if measurement.diffusion_loss != warmup.diffusion_loss:
            failures.append(
                f"measurement {index} changed deterministic diffusion loss from "
                f"{warmup.diffusion_loss:.12f} to {measurement.diffusion_loss:.12f}"
            )
        if measurement.diffusion_chunks != expected_diffusion_chunks:
            failures.append(
                f"measurement {index} evaluated {measurement.diffusion_chunks} "
                f"diffusion chunks, expected {expected_diffusion_chunks}"
            )
    recompiles = int(sum(dynamo_counters["recompiles"].values()))
    if recompiles:
        failures.append(f"{recompiles} recompiles appeared after warmup")
    graph_breaks = int(sum(dynamo_counters["graph_break"].values()))
    if graph_breaks:
        failures.append(f"{graph_breaks} graph breaks appeared after warmup")
    result = {
        "schema": "byte_diffusion_validation_readiness/v2",
        "checkpoint_schema": checkpoint_schema,
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_payload_sha256": dataset_payload_sha256,
        "checkpoint_dataset_payload_sha256": checkpoint_data_sha256,
        "evaluation_source_sha256": source_sha256,
        "requested_validation_chunks": args.validation_chunks,
        "validation_chunks": len(validation_chunks),
        "validation_scope": (
            "proxy"
            if isinstance(validation_chunks, DeterministicSubsetChunkDataset)
            else "challenge"
        ),
        "validation_microbatch": args.validation_microbatch,
        "validation_dataset_sha256": trainer.validation_sha256,
        "requested_diffusion_validation_chunks": args.diffusion_validation_chunks,
        "expected_diffusion_chunks": expected_diffusion_chunks,
        "warmup_seconds": warmup_seconds,
        "validation_seconds": wall_seconds,
        "reported_validation_seconds": [
            item.elapsed_ms / 1_000 for item in measurements
        ],
        "bpb": measurements[-1].bpb,
        "ar_targets": measurements[-1].ar_targets,
        "literal_bytes": measurements[-1].literal_bytes,
        "special_targets": measurements[-1].special_targets,
        "diffusion_loss": measurements[-1].diffusion_loss,
        "diffusion_targets": measurements[-1].diffusion_targets,
        "diffusion_chunks": measurements[-1].diffusion_chunks,
        "average_power_w": sum(powers) / len(powers) if powers else None,
        "average_gpu_utilization": (
            sum(utilizations) / len(utilizations) if utilizations else None
        ),
        "peak_power_w": max(powers) if powers else None,
        "peak_gpu_utilization": max(utilizations) if utilizations else None,
        "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "compile_recompiles_after_warmup": recompiles,
        "compile_graph_breaks_after_warmup": graph_breaks,
        "failures": failures,
    }
    encoded = json.dumps(result, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n")
    print("benchmark_byte_diffusion_validation " + encoded, flush=True)
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
