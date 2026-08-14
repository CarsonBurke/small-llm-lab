#!/usr/bin/env python3
"""Authenticated eight-H100 readiness benchmark for production Byte-Duo.

Launch this only through ``mlq`` with ``torchrun --nproc-per-node=8``.  The
benchmark exercises the exact uneven 249-row optimizer update and the exact
256-row distributed validation ledger used by the production trainer.
"""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
from typing import Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch._dynamo.utils import counters as dynamo_counters
from torch.nn.parallel import DistributedDataParallel as DDP


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.config import ByteDiffusionConfig
from pretraining.byte_diffusion.data import DeterministicChunkCursor
from pretraining.byte_diffusion.duo import (
    DuoSchedule,
    compiled_duo_nelbo_token_loss,
    sample_branch_antithetic_times,
)
from pretraining.byte_diffusion.duo_model import DuoModel
from pretraining.byte_diffusion.pipeline import DeviceBatchPrefetcher
from pretraining.byte_diffusion.readiness import (
    DUO_CANONICAL_VALIDATION_BRANCHES,
    DUO_CANONICAL_VALIDATION_CANVAS_LENGTH,
    DUO_DATA_BRANCH_SPAN_LENGTH,
    DUO_DATA_REQUIRED_BRANCH_BYTES,
    DUO_DISTRIBUTED_READINESS_SCHEMA,
    HEADROOM_FRACTION,
    HEADROOM_MINIMUM_GIB,
    SUSTAINED_GPU_POLICY,
    _distributed_aggregate,
    diagnostic_cadence_contract,
    duo_geometry_contract,
)
from pretraining.byte_diffusion.telemetry import nvidia_smi_selector
from pretraining.byte_diffusion.training import DistributedContext, load_data_directory
from pretraining.byte_diffusion.training_duo import (
    DUO_TRAINING_OBJECTIVES,
    PURE_DUO_OBJECTIVE,
    PreparedDuoUpdate,
    accumulate_duo_validation_stats,
    duo_clean_ar_weight,
    duo_loss,
    duo_objective_contract,
    prepare_duo_update,
)
from scripts.train_byte_duo import (
    _attach_duo_interface,
    _local_imports,
    _prepared_validation_ledger,
    _recipe_batch,
    distributed_loss_scale,
    distributed_rank_positions,
    source_provenance,
)


WORLD_SIZE = 8
GLOBAL_BATCH = 249
MICROBATCH = 32
VALIDATION_ROWS = 256
VALIDATION_BATCH = 32
ROW_LENGTH = 8_192
SCHEDULE_EPS = 1e-3
GIB = 1 << 30


def distributed_readiness_workload_contract(
    *,
    canvas_length: int = 512,
    branches: int = 8,
    objective: str = PURE_DUO_OBJECTIVE,
    log_every: int = 10,
    validation_every: int = 20,
) -> dict[str, object]:
    """Return the exact workload mapping consumed by the trainer validator."""

    return {
        **duo_geometry_contract(canvas_length, branches),
        "diagnostic_cadence": diagnostic_cadence_contract(
            log_every=log_every, validation_every=validation_every
        ),
        **duo_objective_contract(objective),
        "schedule_eps": SCHEDULE_EPS,
        "time_sampling": "global_branch_antithetic_striped_uniform_0_1",
    }


def distributed_readiness_geometry() -> dict[str, object]:
    return {
        "world_size": WORLD_SIZE,
        "global_batch": GLOBAL_BATCH,
        "microbatch": MICROBATCH,
        "base_local_rows": [32, 31, 31, 31, 31, 31, 31, 31],
        "remainder_rotation": "update_index_modulo_world_size",
        "backward_calls_per_rank_update": 1,
        "validation_global_rows": VALIDATION_ROWS,
        "validation_local_rows": VALIDATION_ROWS // WORLD_SIZE,
        "validation_batch_size": VALIDATION_BATCH,
    }


def benchmark_source_provenance() -> dict[str, object]:
    """Hash the complete repository-local implementation closure."""

    pending = {Path(__file__).resolve()}
    observed: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in observed:
            continue
        if not path.is_relative_to(REPO_ROOT):
            raise ValueError(f"distributed readiness source escaped repository: {path}")
        observed.add(path)
        pending.update(item for item in _local_imports(path) if item not in observed)
    digest = hashlib.sha256()
    files: dict[str, str] = {}
    for path in sorted(observed, key=lambda item: item.relative_to(REPO_ROOT).as_posix()):
        relative = path.relative_to(REPO_ROOT).as_posix()
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        files[relative] = file_hash
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_hash))
    return {
        "schema": "byte_duo_distributed_readiness_harness_source/v1",
        "sha256": digest.hexdigest(),
        "files": files,
    }


def _counter_total(name: str) -> int:
    return int(sum(dynamo_counters[name].values()))


def _telemetry_loop(
    stop: threading.Event, selector: str, powers: array, utilizations: array
) -> None:
    while not stop.wait(0.1):
        result = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=power.draw,utilization.gpu",
                "--format=csv,noheader,nounits",
                f"--id={selector}",
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            continue
        try:
            power, separator, utilization = result.stdout.partition("\n")[0].partition(",")
            if separator:
                powers.append(float(power.strip()))
                utilizations.append(float(utilization.strip().removesuffix(" %")))
        except ValueError:
            continue


def _summary(samples: array) -> dict[str, float | int | None]:
    if not samples:
        return {"count": 0, "mean": None, "p10": None, "peak": None}
    values = np.frombuffer(samples, dtype=np.float64)
    return {
        "count": len(samples),
        "mean": math.fsum(samples) / len(samples),
        "p10": float(np.quantile(values, 0.1, method="linear")),
        "peak": max(samples),
    }


def _required_headroom(total_memory: int) -> int:
    return max(
        math.ceil(total_memory * HEADROOM_FRACTION),
        math.ceil(HEADROOM_MINIMUM_GIB * GIB),
    )


def _telemetry_eligible(
    power: Mapping[str, object], utilization: Mapping[str, object]
) -> bool:
    return bool(
        int(power.get("count") or 0) >= 3
        and int(utilization.get("count") or 0) >= 3
        and float(power.get("mean") or -math.inf)
        >= SUSTAINED_GPU_POLICY["minimum_mean_power_w"]
        and float(power.get("p10") or -math.inf)
        >= SUSTAINED_GPU_POLICY["minimum_p10_power_w"]
        and float(utilization.get("mean") or -math.inf)
        >= SUSTAINED_GPU_POLICY["minimum_mean_utilization_percent"]
        and float(utilization.get("p10") or -math.inf)
        >= SUSTAINED_GPU_POLICY["minimum_p10_utilization_percent"]
    )


def _phase_eligible(
    *, total_memory: int, peak_reserved: int, power: Mapping[str, object],
    utilization: Mapping[str, object], unique_graphs: int, recompiles: int,
    graph_breaks: int, require_sustained_power: bool = True,
) -> bool:
    return bool(
        total_memory - peak_reserved >= _required_headroom(total_memory)
        and unique_graphs == 0
        and recompiles == 0
        and graph_breaks == 0
        and (
            _telemetry_eligible(power, utilization)
            if require_sustained_power
            else bool(
                int(power.get("count") or 0) >= 3
                and int(utilization.get("count") or 0) >= 3
                and float(utilization.get("mean") or -math.inf)
                >= SUSTAINED_GPU_POLICY["minimum_mean_utilization_percent"]
                and float(utilization.get("p10") or -math.inf)
                >= SUSTAINED_GPU_POLICY["minimum_p10_utilization_percent"]
            )
        )
    )


def _gpu_identity(device: torch.device, *, rank: int, local_rank: int) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "rank": rank,
        "local_rank": local_rank,
        "device_index": torch.cuda.current_device(),
        "name": torch.cuda.get_device_name(device),
        "total_memory_bytes": properties.total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nvidia_smi_selector": nvidia_smi_selector(device),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", type=Path, default=REPO_ROOT / "data/byte_diffusion_aligned_v5"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-updates", type=int, default=8)
    parser.add_argument("--measured-updates", type=int, default=8)
    parser.add_argument("--validation-warmups", type=int, default=2)
    parser.add_argument("--validation-repetitions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--validation-every", type=int, default=20)
    parser.add_argument("--canvas-length", type=int, default=512)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument(
        "--objective",
        choices=DUO_TRAINING_OBJECTIVES,
        default=os.getenv("BYTE_DUO_OBJECTIVE", PURE_DUO_OBJECTIVE),
    )
    parser.add_argument(
        "--expected-source-sha256",
        default=os.environ.get("BYTE_DUO_EXPECTED_SOURCE_SHA256"),
    )
    parser.add_argument(
        "--expected-data-sha256",
        default=os.environ.get("BYTE_DIFFUSION_EXPECTED_DATA_SHA256"),
    )
    return parser.parse_args()


def _assert_launch(args: argparse.Namespace) -> DistributedContext:
    duo_geometry_contract(args.canvas_length, args.branches)
    if args.warmup_updates < 8 or args.measured_updates < 8:
        raise ValueError("distributed readiness needs at least eight warmup and measured updates")
    if args.validation_warmups < 1 or args.validation_repetitions < 8:
        raise ValueError("distributed readiness needs validation warmup and eight measurements")
    if not args.expected_source_sha256 or not args.expected_data_sha256:
        raise ValueError("distributed readiness requires pinned source and data SHA-256 values")
    if not all(name in os.environ for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        raise ValueError("distributed readiness must be launched by torchrun")
    context = DistributedContext.from_environment("cuda")
    if (
        context.world_size != WORLD_SIZE
        or not dist.is_initialized()
        or dist.get_backend() != "nccl"
        or context.local_rank != context.rank
        or torch.cuda.current_device() != context.local_rank
        or "H100" not in torch.cuda.get_device_name(context.local_rank)
    ):
        context.close()
        raise ValueError("distributed readiness requires one-node, eight-rank NCCL torchrun")
    return context


def _run(args: argparse.Namespace, context: DistributedContext) -> None:
    device = torch.device("cuda", context.local_rank)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    clean_ar_weight = duo_clean_ar_weight(args.objective)

    recipe_source = source_provenance()
    if recipe_source["sha256"] != args.expected_source_sha256:
        raise ValueError("Byte-Duo source differs from the pinned readiness contract")
    manifest, train_chunks, validation_chunks = load_data_directory(
        args.data_path,
        chunk_size=ROW_LENGTH,
        recipe="blt_d",
        required_branch_bytes=DUO_DATA_REQUIRED_BRANCH_BYTES,
        branch_span_length=DUO_DATA_BRANCH_SPAN_LENGTH,
        validation_chunk_limit=VALIDATION_ROWS,
        expected_payload_sha256=args.expected_data_sha256,
    )
    config = ByteDiffusionConfig()
    if (
        manifest.output_size != config.vocab.output_size
        or manifest.pad_id != config.vocab.pad_id
        or manifest.eot_id != config.vocab.eot_id
    ):
        raise ValueError("dataset vocabulary and Byte-Duo model differ")
    if len(validation_chunks) < VALIDATION_ROWS:
        raise ValueError("distributed readiness requires 256 validation rows")

    model = DuoModel(config, schedule_eps=SCHEDULE_EPS).to(device)
    model.validate_production_parameterization()
    execution_model = torch.compile(model, dynamic=False, fullgraph=False)
    forward_model = DDP(
        execution_model,
        device_ids=[device.index],
        output_device=device.index,
        forward_sync_buffers=False,
    )
    _attach_duo_interface(forward_model, model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True
    )
    schedule = DuoSchedule(SCHEDULE_EPS)
    cursor = DeterministicChunkCursor(train_chunks, seed=args.seed, shuffle=True)
    time_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    corruption_generator = torch.Generator(device="cpu").manual_seed(
        args.seed + 1 + 1_000_003 * context.rank
    )

    validation_positions = distributed_rank_positions(
        VALIDATION_ROWS, rank=context.rank, world_size=WORLD_SIZE
    )
    if validation_positions.size != VALIDATION_BATCH:
        raise AssertionError("validation did not shard to 32 rows per rank")
    validation_ledger = _prepared_validation_ledger(
        model,
        validation_chunks,
        rows=VALIDATION_ROWS,
        batch_size=VALIDATION_BATCH,
        patch_stride=config.patch_stride,
        canvas_length=DUO_CANONICAL_VALIDATION_CANVAS_LENGTH,
        branches=DUO_CANONICAL_VALIDATION_BRANCHES,
        seed=args.seed + 10_000,
        schedule=schedule,
        device=device,
        row_indices=validation_positions,
    )

    total_updates = args.warmup_updates + args.measured_updates
    observed_rows = np.empty(total_updates, dtype=np.int64)

    def prepare_update(update_index: int) -> PreparedDuoUpdate:
        global_indices = cursor.next_indices(GLOBAL_BATCH)
        rank_positions = distributed_rank_positions(
            GLOBAL_BATCH,
            rank=context.rank,
            world_size=WORLD_SIZE,
            rotation=update_index,
        )
        observed_rows[update_index] = rank_positions.size
        native = train_chunks.training_batch(global_indices[rank_positions])
        batch = _recipe_batch(native, patch_stride=config.patch_stride)
        global_times = sample_branch_antithetic_times(
            GLOBAL_BATCH, args.branches, device="cpu", generator=time_generator
        )
        local_times = global_times.index_select(0, torch.from_numpy(rank_positions))
        return prepare_duo_update(
            model,
            batch,
            microbatch_size=MICROBATCH,
            canvas_length=args.canvas_length,
            branches=args.branches,
            schedule=schedule,
            times=local_times,
            generator=corruption_generator,
            include_clean_ar=clean_ar_weight != 0.0,
        )

    def train_update(
        update: PreparedDuoUpdate,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(update.microbatches) != 1:
            raise AssertionError("m32 distributed readiness requires one backward call")
        local_denominators = torch.stack(
            (update.nelbo_denominator, update.ar_denominator)
        )
        global_denominators = local_denominators.clone()
        dist.all_reduce(global_denominators, op=dist.ReduceOp.SUM)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = duo_loss(
                forward_model,  # type: ignore[arg-type]
                update.microbatches[0].batch,
                canvas_length=args.canvas_length,
                branches=args.branches,
                prepared=update.microbatches[0].prepared,
                schedule=schedule,
                nelbo_denominator=global_denominators[0],
                ar_denominator=global_denominators[1],
                clean_ar_weight=clean_ar_weight,
                objective_fn=compiled_duo_nelbo_token_loss,
            )
            scaled_loss = loss.total * distributed_loss_scale(WORLD_SIZE)
        scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return (
            local_denominators[0].detach(),
            global_denominators[0].detach(),
            local_denominators[1].detach(),
            global_denominators[1].detach(),
        )

    with DeviceBatchPrefetcher(device=device) as prefetcher:
        updates = iter(prefetcher.batches(range(total_updates), prepare_update))
        for _ in range(args.warmup_updates):
            train_update(next(updates))
        torch.cuda.synchronize(device)
        warmup_unique_graphs = int(dynamo_counters["stats"]["unique_graphs"])
        warmup_recompiles = _counter_total("recompiles")
        warmup_graph_breaks = _counter_total("graph_break")
        dynamo_counters.clear()
        torch.cuda.reset_peak_memory_stats(device)
        powers, utilizations = array("d"), array("d")
        stop = threading.Event()
        sampler = threading.Thread(
            target=_telemetry_loop,
            args=(stop, nvidia_smi_selector(device), powers, utilizations),
            daemon=True,
        )
        context.barrier()
        sampler.start()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        measured_local_counts = torch.empty(
            args.measured_updates, dtype=torch.long, device=device
        )
        measured_global_denominators = torch.empty_like(measured_local_counts)
        measured_local_ar_counts = torch.empty_like(measured_local_counts)
        measured_global_ar_denominators = torch.empty_like(measured_local_counts)
        try:
            for measurement in range(args.measured_updates):
                (
                    local_count,
                    global_denominator,
                    local_ar_count,
                    global_ar_denominator,
                ) = train_update(next(updates))
                measured_local_counts[measurement].copy_(local_count)
                measured_global_denominators[measurement].copy_(global_denominator)
                measured_local_ar_counts[measurement].copy_(local_ar_count)
                measured_global_ar_denominators[measurement].copy_(
                    global_ar_denominator
                )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
        finally:
            stop.set()
            sampler.join(timeout=2.0)
    training_power = _summary(powers)
    training_utilization = _summary(utilizations)
    training_unique_graphs = int(dynamo_counters["stats"]["unique_graphs"])
    training_recompiles = _counter_total("recompiles")
    training_graph_breaks = _counter_total("graph_break")
    training_peak_allocated = torch.cuda.max_memory_allocated(device)
    training_peak_reserved = torch.cuda.max_memory_reserved(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    training_eligible = _phase_eligible(
        total_memory=total_memory,
        peak_reserved=training_peak_reserved,
        power=training_power,
        utilization=training_utilization,
        unique_graphs=training_unique_graphs,
        recompiles=training_recompiles,
        graph_breaks=training_graph_breaks,
    ) and warmup_graph_breaks == 0

    def validation_once() -> torch.Tensor:
        local = accumulate_duo_validation_stats(
            execution_model,
            validation_ledger,
            canvas_length=DUO_CANONICAL_VALIDATION_CANVAS_LENGTH,
            branches=DUO_CANONICAL_VALIDATION_BRANCHES,
            seed=args.seed + 10_000,
            schedule=schedule,
            objective_fn=compiled_duo_nelbo_token_loss,
            total_rows=VALIDATION_ROWS,
            expected_row_ids=torch.from_numpy(validation_positions.copy()),
            compute_ar_diagnostic=clean_ar_weight != 0.0,
        )
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        return local

    dynamo_counters.clear()
    for _ in range(args.validation_warmups):
        validation_once()
    torch.cuda.synchronize(device)
    validation_warmup_unique_graphs = int(
        dynamo_counters["stats"]["unique_graphs"]
    )
    validation_warmup_recompiles = _counter_total("recompiles")
    validation_warmup_graph_breaks = _counter_total("graph_break")
    dynamo_counters.clear()
    torch.cuda.reset_peak_memory_stats(device)
    validation_powers, validation_utilizations = array("d"), array("d")
    validation_stop = threading.Event()
    validation_sampler = threading.Thread(
        target=_telemetry_loop,
        args=(
            validation_stop,
            nvidia_smi_selector(device),
            validation_powers,
            validation_utilizations,
        ),
        daemon=True,
    )
    context.barrier()
    validation_sampler.start()
    torch.cuda.synchronize(device)
    validation_started = time.perf_counter()
    validation_totals = torch.zeros(6, dtype=torch.float64, device=device)
    try:
        for _ in range(args.validation_repetitions):
            validation_totals += validation_once()
        torch.cuda.synchronize(device)
        validation_elapsed = time.perf_counter() - validation_started
    finally:
        validation_stop.set()
        validation_sampler.join(timeout=2.0)
    validation_power = _summary(validation_powers)
    validation_utilization = _summary(validation_utilizations)
    validation_unique_graphs = int(dynamo_counters["stats"]["unique_graphs"])
    validation_recompiles = _counter_total("recompiles")
    validation_graph_breaks = _counter_total("graph_break")
    validation_peak_allocated = torch.cuda.max_memory_allocated(device)
    validation_peak_reserved = torch.cuda.max_memory_reserved(device)
    validation_eligible = _phase_eligible(
        total_memory=total_memory,
        peak_reserved=validation_peak_reserved,
        power=validation_power,
        utilization=validation_utilization,
        unique_graphs=validation_unique_graphs,
        recompiles=validation_recompiles,
        graph_breaks=validation_graph_breaks,
        require_sustained_power=False,
    ) and validation_warmup_graph_breaks == 0

    required_headroom = _required_headroom(total_memory)
    rank_record: dict[str, object] = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "gpu": _gpu_identity(device, rank=context.rank, local_rank=context.local_rank),
        "training": {
            "global_batch": GLOBAL_BATCH,
            "microbatch": MICROBATCH,
            "world_size": WORLD_SIZE,
            "warmup_updates": args.warmup_updates,
            "measured_updates": args.measured_updates,
            "rotations": list(range(total_updates)),
            "local_rows": observed_rows.tolist(),
            "backward_calls": [1] * total_updates,
            "global_denominator_allreduce": True,
            "clean_ar_global_denominator_allreduce": True,
            "ddp_world_size_loss_scale": float(WORLD_SIZE),
            "measured_local_target_counts": measured_local_counts.cpu().tolist(),
            "measured_global_denominators": (
                measured_global_denominators.cpu().tolist()
            ),
            "measured_local_clean_ar_target_counts": (
                measured_local_ar_counts.cpu().tolist()
            ),
            "measured_global_clean_ar_denominators": (
                measured_global_ar_denominators.cpu().tolist()
            ),
            "elapsed_seconds": elapsed,
            "update_ms": 1_000.0 * elapsed / args.measured_updates,
            "power_w": training_power,
            "gpu_utilization_percent": training_utilization,
            "cuda_peak_allocated_bytes": training_peak_allocated,
            "cuda_peak_reserved_bytes": training_peak_reserved,
            "cuda_reserved_headroom_bytes": total_memory - training_peak_reserved,
            "required_headroom_bytes": required_headroom,
            "warmup_unique_graphs": warmup_unique_graphs,
            "warmup_recompiles": warmup_recompiles,
            "warmup_graph_breaks": warmup_graph_breaks,
            "measured_unique_graphs": training_unique_graphs,
            "measured_recompiles": training_recompiles,
            "measured_graph_breaks": training_graph_breaks,
            "eligible": training_eligible,
        },
        "validation": {
            "global_rows": VALIDATION_ROWS,
            "local_rows": VALIDATION_BATCH,
            "row_ids": validation_positions.tolist(),
            "batch_size": VALIDATION_BATCH,
            "world_size": WORLD_SIZE,
            "warmup_repetitions": args.validation_warmups,
            "measured_repetitions": args.validation_repetitions,
            "global_statistics_allreduce": True,
            "elapsed_seconds": validation_elapsed,
            "global_statistics_sum": validation_totals.cpu().tolist(),
            "power_w": validation_power,
            "gpu_utilization_percent": validation_utilization,
            "cuda_peak_allocated_bytes": validation_peak_allocated,
            "cuda_peak_reserved_bytes": validation_peak_reserved,
            "cuda_reserved_headroom_bytes": total_memory - validation_peak_reserved,
            "required_headroom_bytes": required_headroom,
            "warmup_unique_graphs": validation_warmup_unique_graphs,
            "warmup_recompiles": validation_warmup_recompiles,
            "warmup_graph_breaks": validation_warmup_graph_breaks,
            "measured_unique_graphs": validation_unique_graphs,
            "measured_recompiles": validation_recompiles,
            "measured_graph_breaks": validation_graph_breaks,
            "eligible": validation_eligible,
        },
        "eligible": training_eligible and validation_eligible,
    }
    gathered: list[dict[str, object] | None] | None = (
        [None] * WORLD_SIZE if context.is_primary else None
    )
    dist.gather_object(rank_record, gathered, dst=0)
    if context.is_primary:
        assert gathered is not None and all(record is not None for record in gathered)
        records = [record for record in gathered if record is not None]
        records.sort(key=lambda record: int(record["rank"]))
        aggregate = _distributed_aggregate(records)
        report = {
            "schema": DUO_DISTRIBUTED_READINESS_SCHEMA,
            "architecture": "duo",
            "evidence_kind": (
                "exact_distributed_training_and_validation_systems_benchmark"
            ),
            "recipe_source": recipe_source,
            "benchmark_source": benchmark_source_provenance(),
            "dataset_payload_sha256": args.expected_data_sha256,
            "model_config": config.to_dict(),
            "parameter_count": model.parameter_count,
            "workload": distributed_readiness_workload_contract(
                canvas_length=args.canvas_length,
                branches=args.branches,
                objective=args.objective,
                log_every=args.log_every, validation_every=args.validation_every
            ),
            "geometry": distributed_readiness_geometry(),
            "runtime": {
                "compiled": True,
                "distributed_backend": "nccl",
                "launcher": "torchrun",
                "model_wrapper": "DDP(torch.compile(DuoModel))",
                "world_size": WORLD_SIZE,
            },
            "sustained_gpu_policy": SUSTAINED_GPU_POLICY,
            "headroom_policy": {
                "fraction": HEADROOM_FRACTION,
                "minimum_gib": HEADROOM_MINIMUM_GIB,
            },
            "ranks": records,
            "aggregate": aggregate,
            "eligible": aggregate["all_ranks_eligible"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".working")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output)
        print("byte_duo_distributed_readiness " + json.dumps(report, sort_keys=True))
    context.barrier()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("distributed Byte-Duo readiness requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if not 0 <= local_rank < WORLD_SIZE:
        raise ValueError("distributed readiness requires LOCAL_RANK in [0, 8)")
    torch.cuda.set_device(local_rank)
    context = _assert_launch(args)
    try:
        _run(args, context)
    finally:
        context.close()


if __name__ == "__main__":
    main()


__all__ = (
    "benchmark_source_provenance",
    "distributed_readiness_geometry",
    "distributed_readiness_workload_contract",
)
