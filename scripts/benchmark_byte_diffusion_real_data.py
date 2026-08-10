#!/usr/bin/env python3
"""Fail-closed production benchmark on diverse real packed corpus batches.

This is a GPU workload. Queue it through ``mlq``; do not run it directly.
Unlike the synthetic benchmark, this exercises variable valid-token counts,
native varlen Flash attention, FlexAttention branches, gradient accumulation,
and the production optimizer together.
"""

from __future__ import annotations

import argparse
import json
import math
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

from pretraining.byte_diffusion.config import model_config_from_env
from pretraining.byte_diffusion.data import DeterministicChunkCursor
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.training import (
    ByteDiffusionTrainer,
    DistributedContext,
    TrainingRunConfig,
    load_data_directory,
)
from scripts.train_byte_diffusion import training_source_provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/byte_diffusion"))
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--measured-updates", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--global-batch", type=int, default=256)
    parser.add_argument(
        "--expected-recipe",
        choices=("canvas", "blt_d", "causal_only"),
        required=True,
        help="Fail closed unless the environment resolves to this recipe.",
    )
    parser.add_argument("--static-shapes", action="store_true")
    parser.add_argument(
        "--microbatch-token-budget",
        type=int,
        help=(
            "Maximum clean-plus-canvas positions in one microbatch. Defaults "
            "to the fixed full-width workload for --microbatch."
        ),
    )
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="Enable the experimental global-stack activation checkpointing path.",
    )
    parser.add_argument("--max-update-ms", type=float, default=2_000.0)
    parser.add_argument("--min-average-power-w", type=float, default=500.0)
    parser.add_argument("--min-average-gpu-utilization", type=float, default=90.0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--profile-table",
        type=Path,
        help="Write one post-warmup torch.profiler operator table.",
    )
    return parser.parse_args()


def sample_telemetry(
    stop: threading.Event, power_samples: list[float], utilization_samples: list[float]
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
        if observed.returncode == 0:
            try:
                power, utilization = observed.stdout.strip().splitlines()[0].split(",")
                power_samples.append(float(power.strip()))
                utilization_samples.append(float(utilization.strip().removesuffix(" %")))
            except (IndexError, ValueError):
                pass


def counter_total(name: str) -> int:
    return int(sum(dynamo_counters[name].values()))


def build_benchmark_run_config(
    args: argparse.Namespace, *, total_updates: int
) -> TrainingRunConfig:
    """Resolve the benchmark through the production environment contract."""

    run = TrainingRunConfig.from_env(
        iterations=total_updates,
        val_loss_every=total_updates,
        train_log_every=total_updates,
        validation_chunks=16,
        validation_microbatch_per_rank=16,
        diffusion_validation_chunks=16,
        warmdown_iters=0,
        run_id="byte_diffusion_real_data_readiness",
        microbatch_per_rank=args.microbatch,
        microbatch_token_budget=args.microbatch_token_budget,
        gradient_accumulation=math.ceil(args.global_batch / args.microbatch),
        global_batch_size=args.global_batch,
        compile_model=True,
        compile_dynamic_shapes=not args.static_shapes,
        activation_checkpointing=args.activation_checkpointing,
    )
    if run.recipe != args.expected_recipe:
        raise ValueError(
            f"expected recipe {args.expected_recipe!r}, observed {run.recipe!r}"
        )
    return run


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark must be submitted through mlq")
    if min(args.warmup_updates, args.measured_updates, args.microbatch) <= 0:
        raise ValueError("warmup, measured updates, and microbatch must be positive")
    token_budget = args.microbatch_token_budget
    if token_budget is None:
        token_budget = args.microbatch * (8_192 + 512)
    if token_budget <= 0:
        raise ValueError("microbatch token budget must be positive")

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.set_float32_matmul_precision("high")
    source_provenance = training_source_provenance()
    expected_source_sha256 = os.environ.get(
        "BYTE_DIFFUSION_EXPECTED_SOURCE_SHA256"
    )
    if expected_source_sha256 is None:
        raise ValueError(
            "BYTE_DIFFUSION_EXPECTED_SOURCE_SHA256 is required for a readiness run"
        )
    if source_provenance["sha256"] != expected_source_sha256:
        raise ValueError(
            "training source differs from the pinned readiness contract: "
            f"expected {expected_source_sha256}, "
            f"observed {source_provenance['sha256']}"
        )
    total_updates = (
        args.warmup_updates
        + args.measured_updates
        + int(args.profile_table is not None)
    )
    args.microbatch_token_budget = token_budget
    run = build_benchmark_run_config(args, total_updates=total_updates)
    expected_data_sha256 = os.environ.get("BYTE_DIFFUSION_EXPECTED_DATA_SHA256")
    if expected_data_sha256 is None:
        raise ValueError(
            "BYTE_DIFFUSION_EXPECTED_DATA_SHA256 is required for a readiness run"
        )
    manifest, train_chunks, validation_chunks = load_data_directory(
        args.data_path,
        chunk_size=8192,
        recipe=run.recipe,
        required_branch_bytes=run.corruption.corrupted_positions_per_row,
        branch_span_length=run.corruption.canvas_length,
        validation_chunk_limit=run.validation_chunks,
        require_challenge_validation=True,
        expected_payload_sha256=expected_data_sha256,
    )
    dataset_manifest = json.loads((args.data_path / "manifest.json").read_text())
    dataset_payload_sha256 = dataset_manifest.get("payload_sha256")
    if dataset_payload_sha256 != expected_data_sha256:
        raise ValueError(
            "dataset manifest hash differs from the pinned readiness contract: "
            f"expected {expected_data_sha256}, observed {dataset_payload_sha256}"
        )
    trainer = ByteDiffusionTrainer(
        ByteDiffusionModel(model_config_from_env()),
        DeterministicChunkCursor(train_chunks, seed=run.seed, shuffle=True),
        validation_chunks,
        run,
        device=device,
        distributed=DistributedContext(),
        atomic_manifest=manifest,
    )

    for update in range(args.warmup_updates):
        trainer.run_update(
            materialize_metrics=update + 1 == args.warmup_updates
        )
    torch.cuda.synchronize()
    if args.profile_table is not None:
        with torch.profiler.profile(
            activities=(
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ),
            profile_memory=True,
        ) as profile:
            trainer.run_update(materialize_metrics=True)
        torch.cuda.synchronize()
        args.profile_table.parent.mkdir(parents=True, exist_ok=True)
        args.profile_table.write_text(
            profile.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=60
            )
            + "\n"
        )
    training_time_before_measurement = trainer.training_time_ms
    warmup_unique_graphs = int(dynamo_counters["stats"]["unique_graphs"])
    warmup_graph_breaks = counter_total("graph_break")
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
    update_metrics = None
    for update in range(args.measured_updates):
        update_metrics = trainer.run_update(
            materialize_metrics=update + 1 == args.measured_updates
        )
    if update_metrics is None:
        raise AssertionError("final measured update omitted metrics")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    stop.set()
    sampler.join(timeout=1.0)

    update_ms = 1_000.0 * elapsed / args.measured_updates
    unique_graphs = int(dynamo_counters["stats"]["unique_graphs"])
    recompiles = counter_total("recompiles")
    graph_breaks = counter_total("graph_break")
    average_power = sum(powers) / len(powers) if powers else None
    average_utilization = (
        sum(utilizations) / len(utilizations) if utilizations else None
    )
    failures: list[str] = []
    if unique_graphs:
        failures.append(f"{unique_graphs} new compiled graphs appeared after warmup")
    if recompiles:
        failures.append(f"{recompiles} recompiles appeared after warmup")
    if graph_breaks:
        failures.append(f"{graph_breaks} graph breaks appeared after warmup")
    if update_ms > args.max_update_ms:
        failures.append(
            f"{update_ms:.1f} ms/update exceeds {args.max_update_ms:.1f} ms readiness limit"
        )
    if average_power is None:
        failures.append("GPU power sampling produced no observations")
    elif average_power < args.min_average_power_w:
        failures.append(
            f"{average_power:.1f} W average is below {args.min_average_power_w:.1f} W"
        )
    if average_utilization is None:
        failures.append("GPU utilization sampling produced no observations")
    elif average_utilization < args.min_average_gpu_utilization:
        failures.append(
            f"{average_utilization:.1f}% average GPU utilization is below "
            f"{args.min_average_gpu_utilization:.1f}%"
        )

    result = {
        "schema": "byte_diffusion_real_data_readiness/v2",
        "training_source_sha256": source_provenance["sha256"],
        "warmup_updates": args.warmup_updates,
        "measured_updates": args.measured_updates,
        "microbatch": args.microbatch,
        "microbatch_token_budget": run.microbatch_token_budget,
        "global_batch": args.global_batch,
        "dynamic_shapes": run.compile_dynamic_shapes,
        "activation_checkpointing": run.activation_checkpointing,
        "recipe": run.recipe,
        "model_config": trainer.model_config.to_dict(),
        "dataset_payload_sha256": dataset_payload_sha256,
        "update_ms": update_ms,
        "trainer_device_ms_per_update": (
            update_metrics.elapsed_ms - training_time_before_measurement
        )
        / args.measured_updates,
        "ar_targets_per_update": update_metrics.ar_targets,
        "diffusion_targets_per_update": update_metrics.diffusion_targets,
        "microsteps_per_update": update_metrics.microsteps,
        "max_microbatch_per_update": update_metrics.max_microbatch,
        "max_physical_positions_per_update": update_metrics.max_physical_positions,
        "compile_unique_graphs_after_warmup": unique_graphs,
        "compile_recompiles_after_warmup": recompiles,
        "compile_graph_breaks_after_warmup": graph_breaks,
        "compile_unique_graphs_during_warmup": warmup_unique_graphs,
        "compile_graph_breaks_during_warmup": warmup_graph_breaks,
        "average_power_w": average_power,
        "average_gpu_utilization": average_utilization,
        "peak_gpu_utilization": max(utilizations) if utilizations else None,
        "peak_power_w": max(powers) if powers else None,
        "power_samples": len(powers),
        "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "failures": failures,
    }
    encoded = json.dumps(result, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n")
    print("benchmark_byte_diffusion_real_data " + encoded, flush=True)
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
