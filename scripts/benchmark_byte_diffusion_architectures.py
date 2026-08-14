#!/usr/bin/env python3
"""Queue-only, full-row readiness benchmark for the standalone diffusion cells.

This is systems evidence, not a reduced training-quality run.  Every measured
update consumes the production 249-row global batch of real 8,192-byte pages,
runs compiled forward and backward passes, and applies the fused optimizer.
Candidate failures are isolated so an I-DLM microbatch sweep can continue after
CUDA OOM and select the fastest candidate with explicit memory headroom.
"""

from __future__ import annotations

import argparse
from array import array
from dataclasses import asdict
import gc
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
from torch._dynamo.utils import counters as dynamo_counters


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.config import ByteDiffusionConfig, model_config_from_env
from pretraining.byte_diffusion.data import DeterministicChunkCursor
from pretraining.byte_diffusion.diffusion_gemma import sample_static_half_batch_mask
from pretraining.byte_diffusion.diffusion_gemma_model import DiffusionGemmaModel
from pretraining.byte_diffusion.duo import (
    DuoSchedule,
    compiled_duo_nelbo_token_loss,
    sample_branch_antithetic_times,
)
from pretraining.byte_diffusion.duo_model import DuoModel
from pretraining.byte_diffusion.idlm_model import IDLMModel, IDLMModelConfig
from pretraining.byte_diffusion.readiness import (
    DUO_CANONICAL_VALIDATION_BRANCHES,
    DUO_CANONICAL_VALIDATION_CANVAS_LENGTH,
    DUO_DATA_BRANCH_SPAN_LENGTH,
    DUO_DATA_REQUIRED_BRANCH_BYTES,
    HEADROOM_FRACTION,
    HEADROOM_MINIMUM_GIB,
    SUSTAINED_GPU_POLICY,
    diagnostic_cadence_contract,
    duo_geometry_contract,
)
from pretraining.byte_diffusion.pipeline import DeviceBatchPrefetcher
from pretraining.byte_diffusion.telemetry import nvidia_smi_selector
from pretraining.byte_diffusion.training import load_data_directory
from pretraining.byte_diffusion.training_diffusion_gemma import (
    PreparedDiffusionGemmaValidationBatch,
    diffusion_gemma_loss,
    prepare_diffusion_gemma_inputs,
    validate_diffusion_gemma,
)
from pretraining.byte_diffusion.training_idlm import IDLMTrainer, IDLMTrainingConfig
from pretraining.byte_diffusion.training_duo import (
    DUO_TRAINING_OBJECTIVES,
    PURE_DUO_OBJECTIVE,
    PreparedDuoUpdate,
    duo_clean_ar_weight,
    duo_loss,
    duo_objective_contract,
    prepare_duo_update,
    prepare_duo_validation_batch,
    validate_duo,
)
from scripts.train_byte_diffusion_gemma import (
    _prepared_validation_ledger as _diffusion_gemma_validation_ledger,
    _validation_batches as _diffusion_gemma_validation_batches,
    _recipe_batch as _diffusion_gemma_batch,
    source_provenance as diffusion_gemma_source_provenance,
)
from scripts.train_byte_duo import (
    _local_imports,
    _validation_batches as _duo_validation_batches,
    _recipe_batch as _duo_batch,
    source_provenance as duo_source_provenance,
)
from scripts.train_byte_idlm import source_provenance as idlm_source_provenance


GIB = 1 << 30
PRODUCTION_ROW_LENGTH = 8_192
MATCHED_GLOBAL_BATCH = 249


def readiness_workload_contract(
    architecture: str,
    *,
    duo_objective: str = PURE_DUO_OBJECTIVE,
    duo_canvas_length: int = 512,
    duo_branches: int = 8,
) -> tuple[dict[str, object], dict[str, object]]:
    """Exact model and forward geometry exercised by this harness."""

    if architecture == "idlm":
        return asdict(IDLMModelConfig()), {
            "block_size": 4,
            "diagnostic_cadence": diagnostic_cadence_contract(),
            "objective": "proposal_plus_clean_exact_optimizer_step_balance",
        }
    model = ByteDiffusionConfig().to_dict()
    if architecture == "diffusion_gemma":
        return model, {
            "branches": 1,
            "canvas_length": 512,
            "clean_ar_weight": 1.0,
            "diagnostic_cadence": diagnostic_cadence_contract(),
            "integrated_exact_k_training": False,
            "runtime_candidate_validation": False,
            "self_condition": True,
        }
    if architecture == "duo":
        return model, {
            **duo_geometry_contract(duo_canvas_length, duo_branches),
            "diagnostic_cadence": diagnostic_cadence_contract(),
            **duo_objective_contract(duo_objective),
            "schedule_eps": 0.001,
            "time_sampling": "global_branch_antithetic_striped_uniform_0_1",
        }
    raise ValueError(f"unknown architecture {architecture!r}")


def parse_candidate_microbatches(value: str) -> tuple[int, ...]:
    """Parse a unique ascending candidate set without a mutable row list."""

    try:
        candidates = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("microbatches must be comma-separated ints") from error
    if not candidates or any(candidate <= 0 for candidate in candidates):
        raise argparse.ArgumentTypeError("candidate microbatches must be positive")
    if len(set(candidates)) != len(candidates):
        raise argparse.ArgumentTypeError("candidate microbatches must be unique")
    return tuple(sorted(candidates))


def tail_geometry(global_batch: int, microbatch: int) -> tuple[int, int]:
    if global_batch <= 0 or microbatch <= 0:
        raise ValueError("global batch and microbatch must be positive")
    microsteps = math.ceil(global_batch / microbatch)
    return microsteps, global_batch - (microsteps - 1) * microbatch


def required_headroom_bytes(
    total_bytes: int, *, fraction: float, minimum_gib: float
) -> int:
    if total_bytes <= 0 or not 0 <= fraction < 1 or minimum_gib < 0:
        raise ValueError("invalid memory headroom policy")
    return max(math.ceil(total_bytes * fraction), math.ceil(minimum_gib * GIB))


def choose_fastest_fitting(results: Mapping[str, Mapping[str, object]]) -> int | None:
    """Select the minimum measured update time among fully eligible candidates."""

    eligible = (
        (int(microbatch), float(result["update_ms"]))
        for microbatch, result in results.items()
        if result.get("status") == "ok" and bool(result.get("eligible"))
    )
    return min(eligible, key=lambda item: (item[1], -item[0]), default=(None, 0.0))[0]


def counter_total(name: str) -> int:
    return int(sum(dynamo_counters[name].values()))


def accumulate_target_counts(
    primary: torch.Tensor,
    anchor: torch.Tensor,
    counts: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Accumulate benchmark counters without materializing CUDA scalars."""

    primary.add_(counts[0])
    anchor.add_(counts[1])


def _telemetry_loop(
    stop: threading.Event,
    gpu_index: str,
    powers: array,
    utilizations: array,
) -> None:
    while not stop.wait(0.1):
        observed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=power.draw,utilization.gpu",
                "--format=csv,noheader,nounits",
                f"--id={gpu_index}",
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if observed.returncode != 0:
            continue
        try:
            line = observed.stdout.partition("\n")[0]
            power, separator, utilization = line.partition(",")
            if not separator:
                continue
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


def _meets_sustained_gpu_policy(
    power: Mapping[str, object],
    utilization: Mapping[str, object],
    *,
    minimum_mean_power_w: float,
    minimum_p10_power_w: float,
    minimum_mean_utilization: float,
    minimum_p10_utilization: float,
) -> bool:
    return (
        int(power.get("count") or 0) >= 3
        and int(utilization.get("count") or 0) >= 3
        and power.get("mean") is not None
        and float(power["mean"]) >= minimum_mean_power_w
        and power.get("p10") is not None
        and float(power["p10"]) >= minimum_p10_power_w
        and utilization.get("mean") is not None
        and float(utilization["mean"]) >= minimum_mean_utilization
        and utilization.get("p10") is not None
        and float(utilization["p10"]) >= minimum_p10_utilization
    )


def _meets_utilization_policy(
    power: Mapping[str, object],
    utilization: Mapping[str, object],
    *,
    minimum_mean_utilization: float,
    minimum_p10_utilization: float,
) -> bool:
    return (
        int(power.get("count") or 0) >= 3
        and int(utilization.get("count") or 0) >= 3
        and utilization.get("mean") is not None
        and float(utilization["mean"]) >= minimum_mean_utilization
        and utilization.get("p10") is not None
        and float(utilization["p10"]) >= minimum_p10_utilization
    )


def benchmark_harness_provenance() -> dict[str, object]:
    """Bind readiness evidence to the timing/telemetry harness closure."""

    pending = {Path(__file__).resolve()}
    observed: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in observed:
            continue
        observed.add(path)
        pending.update(item for item in _local_imports(path) if item not in observed)
    digest = hashlib.sha256()
    relative_paths = tuple(sorted(path.relative_to(REPO_ROOT) for path in observed))
    for relative in relative_paths:
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update((REPO_ROOT / relative).read_bytes())
        digest.update(b"\0")
    return {
        "schema": "byte_diffusion_readiness_harness_source/v1",
        "sha256": digest.hexdigest(),
        "files": list(map(str, relative_paths)),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


class DiffusionGemmaUpdateHarness:
    """Production-equivalent DiffusionGemma update without logging or validation."""

    def __init__(
        self,
        train_dataset,
        validation_dataset,
        *,
        manifest,
        device: torch.device,
        microbatch: int,
        validation_batch_size: int,
        global_batch: int,
        seed: int,
        batch_prefetcher: DeviceBatchPrefetcher,
    ) -> None:
        self.config = ByteDiffusionConfig()
        if (
            manifest.output_size != self.config.vocab.output_size
            or manifest.pad_id != self.config.vocab.pad_id
            or manifest.eot_id != self.config.vocab.eot_id
        ):
            raise ValueError("dataset vocabulary and DiffusionGemma model differ")
        self.model = DiffusionGemmaModel(self.config).to(device)
        self.model.validate_production_parameterization()
        self.execution_model = torch.compile(
            self.model, dynamic=False, fullgraph=False
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True,
        )
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.cursor = DeterministicChunkCursor(train_dataset, seed=seed, shuffle=True)
        self.generator = torch.Generator(device=device).manual_seed(seed + 1)
        self.device = device
        self.microbatch = microbatch
        self.validation_batch_size = validation_batch_size
        self.global_batch = global_batch
        self.batch_prefetcher = batch_prefetcher
        self.microbatch_keys = tuple(
            (start, min(start + microbatch, global_batch))
            for start in range(0, global_batch, microbatch)
        )
        self.validation_ledger = _diffusion_gemma_validation_ledger(
            self.model,
            validation_dataset,
            rows=min(256, len(validation_dataset)),
            batch_size=validation_batch_size,
            patch_stride=self.config.patch_stride,
            canvas_length=512,
            branches=1,
            seed=10_000 + 1_337,
            device=device,
        )

    @property
    def parameter_count(self) -> int:
        return self.model.parameter_count

    def run_update(self) -> tuple[torch.Tensor, torch.Tensor]:
        update_indices = self.cursor.next_indices(self.global_batch)
        native_update = self.train_dataset.training_batch(update_indices).pin_memory()
        update_batch = _diffusion_gemma_batch(
            native_update, patch_stride=self.config.patch_stride
        ).to(self.device, non_blocking=True)
        prepared_update = prepare_diffusion_gemma_inputs(
            self.execution_model,
            update_batch,
            canvas_length=512,
            branches=1,
            integrated_exact_k=False,
            generator=self.generator,
            validate_candidates=False,
        )
        diffusion_denominator = prepared_update.corruption.active.sum()
        ar_denominator = (
            update_batch.ar_targets.ne(-100).sum()
            + update_batch.bos_targets.numel()
        )
        del update_batch, native_update

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        microbatch_keys = self.microbatch_keys

        def prepare_microbatch(key: tuple[int, int]):
            start, stop = key
            return _diffusion_gemma_batch(
                self.train_dataset.training_batch(update_indices[start:stop]),
                patch_stride=self.config.patch_stride,
            )

        device_batches = self.batch_prefetcher.batches(
            microbatch_keys, prepare_microbatch
        )
        for (start, stop), batch in zip(
            microbatch_keys, device_batches, strict=True
        ):
            self_conditioned_rows = sample_static_half_batch_mask(
                stop - start,
                device=self.device,
                generator=self.generator,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = diffusion_gemma_loss(
                    self.execution_model,
                    batch,
                    canvas_length=512,
                    branches=1,
                    clean_ar_weight=1.0,
                    self_condition=True,
                    integrated_exact_k=False,
                    generator=self.generator,
                    prepared=prepared_update.slice_rows(start, stop),
                    self_conditioned_rows=self_conditioned_rows,
                    diffusion_denominator=diffusion_denominator,
                    ar_denominator=ar_denominator,
                )
            loss.total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return diffusion_denominator.detach(), ar_denominator.detach()

    @torch.no_grad()
    def run_validation(self, rows: int = 256):
        return validate_diffusion_gemma(
            self.execution_model,
            self.validation_ledger,
            canvas_length=512,
            branches=1,
            seed=10_000 + 1_337,
            self_condition=True,
            integrated_exact_k=True,
        )


class DuoUpdateHarness:
    """Production-equivalent update for one named Byte-Duo objective."""

    def __init__(
        self,
        train_dataset,
        validation_dataset,
        *,
        manifest,
        device: torch.device,
        microbatch: int,
        validation_batch_size: int,
        global_batch: int,
        canvas_length: int,
        branches: int,
        objective: str = PURE_DUO_OBJECTIVE,
        seed: int,
        batch_prefetcher: DeviceBatchPrefetcher,
    ) -> None:
        self.config = model_config_from_env()
        if (
            manifest.output_size != self.config.vocab.output_size
            or manifest.pad_id != self.config.vocab.pad_id
            or manifest.eot_id != self.config.vocab.eot_id
        ):
            raise ValueError("dataset vocabulary and Byte-Duo model differ")
        self.model = DuoModel(self.config).to(device)
        if self.config == ByteDiffusionConfig():
            self.model.validate_production_parameterization()
        self.execution_model = torch.compile(
            self.model, dynamic=False, fullgraph=False
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True,
        )
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.cursor = DeterministicChunkCursor(train_dataset, seed=seed, shuffle=True)
        self.generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        self.schedule = DuoSchedule()
        self.device = device
        self.microbatch = microbatch
        self.validation_batch_size = validation_batch_size
        self.global_batch = global_batch
        self.canvas_length = canvas_length
        self.branches = branches
        self.objective = objective
        self.clean_ar_weight = duo_clean_ar_weight(objective)
        self.batch_prefetcher = batch_prefetcher
        validation_rows = min(256, len(validation_dataset))
        cpu_batches = _duo_validation_batches(
            validation_dataset,
            rows=validation_rows,
            batch_size=validation_batch_size,
            patch_stride=self.config.patch_stride,
            device=torch.device("cpu"),
        )
        self.validation_ledger = tuple(
            prepare_duo_validation_batch(
                self.model,
                batch,
                total_rows=validation_rows,
                canvas_length=DUO_CANONICAL_VALIDATION_CANVAS_LENGTH,
                branches=DUO_CANONICAL_VALIDATION_BRANCHES,
                seed=10_000 + 1_337,
                schedule=self.schedule,
            )
            .pin_memory()
            .to(device, non_blocking=True)
            for batch in cpu_batches
        )
        torch.cuda.synchronize(device)

    @property
    def parameter_count(self) -> int:
        return self.model.parameter_count

    def prepared_updates(self, count: int):
        """Overlap complete update collation and transfer with GPU training."""

        if count <= 0:
            raise ValueError("prefetched update count must be positive")

        def prepare_cpu(_: int) -> PreparedDuoUpdate:
            update_indices = self.cursor.next_indices(self.global_batch)
            native_update = self.train_dataset.training_batch(update_indices)
            update_batch = _duo_batch(
                native_update, patch_stride=self.config.patch_stride
            )
            update_times = sample_branch_antithetic_times(
                self.global_batch,
                self.branches,
                device="cpu",
                generator=self.generator,
            )
            return prepare_duo_update(
                self.model,
                update_batch,
                microbatch_size=self.microbatch,
                canvas_length=self.canvas_length,
                branches=self.branches,
                schedule=self.schedule,
                times=update_times,
                generator=self.generator,
                include_clean_ar=self.clean_ar_weight != 0.0,
            )

        return self.batch_prefetcher.batches(range(count), prepare_cpu)

    def run_update(
        self, update: PreparedDuoUpdate
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nelbo_denominator = update.nelbo_denominator
        ar_denominator = update.ar_denominator

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for item in update.microbatches:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = duo_loss(
                    self.execution_model,
                    item.batch,
                    canvas_length=self.canvas_length,
                    branches=self.branches,
                    prepared=item.prepared,
                    schedule=self.schedule,
                    nelbo_denominator=nelbo_denominator,
                    ar_denominator=ar_denominator,
                    clean_ar_weight=self.clean_ar_weight,
                    objective_fn=compiled_duo_nelbo_token_loss,
                )
            loss.total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return nelbo_denominator.detach(), (
            ar_denominator.detach()
            if self.clean_ar_weight != 0.0
            else nelbo_denominator.new_zeros(())
        )

    @torch.no_grad()
    def run_validation(self, rows: int = 256):
        evaluated_rows = min(rows, len(self.validation_dataset))
        return validate_duo(
            self.execution_model,
            self.validation_ledger,
            canvas_length=DUO_CANONICAL_VALIDATION_CANVAS_LENGTH,
            branches=DUO_CANONICAL_VALIDATION_BRANCHES,
            seed=10_000 + 1_337,
            schedule=self.schedule,
            objective_fn=compiled_duo_nelbo_token_loss,
            total_rows=evaluated_rows,
            compute_ar_diagnostic=self.clean_ar_weight != 0.0,
        )


def _build_idlm_harness(
    train_dataset,
    validation_dataset,
    *,
    manifest,
    device: torch.device,
    microbatch: int,
    validation_batch_size: int,
    global_batch: int,
    total_updates: int,
    seed: int,
    batch_prefetcher: DeviceBatchPrefetcher,
) -> IDLMTrainer:
    return IDLMTrainer(
        IDLMModel(IDLMModelConfig()),
        train_dataset,
        validation_dataset,
        DeterministicChunkCursor(train_dataset, seed=seed, shuffle=True),
        IDLMTrainingConfig(
            iterations=total_updates,
            batch_size=microbatch,
            global_batch_size=global_batch,
            validation_batch_size=validation_batch_size,
            validation_rows=min(256, len(validation_dataset)),
            warmdown_iters=0,
            seed=seed,
            compile_model=True,
        ),
        device=device,
        atomic_manifest=manifest,
        batch_prefetcher=batch_prefetcher,
    )


def _reset_candidate_state() -> None:
    gc.collect()
    torch._dynamo.reset()
    dynamo_counters.clear()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _benchmark_candidate(
    architecture: str,
    *,
    manifest,
    train_dataset,
    validation_dataset,
    device: torch.device,
    microbatch: int,
    validation_batch_size: int,
    global_batch: int,
    duo_canvas_length: int,
    duo_branches: int,
    duo_objective: str,
    warmup_updates: int,
    measured_updates: int,
    seed: int,
    minimum_headroom: int,
    minimum_mean_power_w: float,
    minimum_p10_power_w: float,
    minimum_mean_utilization: float,
    minimum_p10_utilization: float,
    batch_prefetcher: DeviceBatchPrefetcher,
) -> dict[str, object]:
    _reset_candidate_state()
    _seed_everything(seed)
    total_updates = warmup_updates + measured_updates
    duo_update_iterator = None
    if architecture == "idlm":
        harness = _build_idlm_harness(
            train_dataset,
            validation_dataset,
            manifest=manifest,
            device=device,
            microbatch=microbatch,
            validation_batch_size=validation_batch_size,
            global_batch=global_batch,
            total_updates=total_updates,
            seed=seed,
            batch_prefetcher=batch_prefetcher,
        )
        run_update = lambda: harness.run_update(materialize_metrics=False)
        parameter_count = harness.model.parameter_count()
    elif architecture == "diffusion_gemma":
        harness = DiffusionGemmaUpdateHarness(
            train_dataset,
            validation_dataset,
            manifest=manifest,
            device=device,
            microbatch=microbatch,
            validation_batch_size=validation_batch_size,
            global_batch=global_batch,
            seed=seed,
            batch_prefetcher=batch_prefetcher,
        )
        run_update = harness.run_update
        parameter_count = harness.parameter_count
    else:
        harness = DuoUpdateHarness(
            train_dataset,
            validation_dataset,
            manifest=manifest,
            device=device,
            microbatch=microbatch,
            validation_batch_size=validation_batch_size,
            global_batch=global_batch,
            canvas_length=duo_canvas_length,
            branches=duo_branches,
            objective=duo_objective,
            seed=seed,
            batch_prefetcher=batch_prefetcher,
        )
        duo_update_iterator = harness.prepared_updates(total_updates)
        run_update = lambda: harness.run_update(next(duo_update_iterator))
        parameter_count = harness.parameter_count

    for _ in range(warmup_updates):
        run_update()
    torch.cuda.synchronize(device)
    warmup_unique_graphs = int(dynamo_counters["stats"]["unique_graphs"])
    warmup_graph_breaks = counter_total("graph_break")
    dynamo_counters.clear()
    torch.cuda.reset_peak_memory_stats(device)

    powers = array("d")
    utilizations = array("d")
    stop_telemetry = threading.Event()
    sampler = threading.Thread(
        target=_telemetry_loop,
        args=(stop_telemetry, nvidia_smi_selector(device), powers, utilizations),
        daemon=True,
    )
    # Both timing and telemetry begin only after the synchronized warmup and
    # end only after all measured optimizer work has completed on the GPU.
    sampler.start()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    primary_targets_device = torch.zeros((), dtype=torch.long, device=device)
    anchor_targets_device = torch.zeros_like(primary_targets_device)
    try:
        for _ in range(measured_updates):
            metric = run_update()
            counts = (
                (
                    harness.last_update_counts[0],
                    harness.last_update_counts[1],
                )
                if architecture == "idlm"
                else metric
            )
            accumulate_target_counts(
                primary_targets_device,
                anchor_targets_device,
                counts,
            )
        torch.cuda.synchronize(device)
        elapsed_seconds = time.perf_counter() - started
    finally:
        stop_telemetry.set()
        sampler.join(timeout=2.0)
        if duo_update_iterator is not None:
            close = getattr(duo_update_iterator, "close", None)
            if close is not None:
                close()
    primary_targets = int(primary_targets_device)
    anchor_targets = int(anchor_targets_device)

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserved_headroom = total_memory - peak_reserved
    unique_graphs = int(dynamo_counters["stats"]["unique_graphs"])
    recompiles = counter_total("recompiles")
    graph_breaks = counter_total("graph_break")
    microsteps, tail = tail_geometry(global_batch, microbatch)
    update_ms = 1_000.0 * elapsed_seconds / measured_updates
    power_summary = _summary(powers)
    utilization_summary = _summary(utilizations)
    # Validation is a first-class systems gate for every architecture.  The
    # old readiness sweep only measured Byte-Duo validation, which allowed an
    # otherwise fast training microbatch to promote while I-DLM or
    # DiffusionGemma validation serialized on tiny batches and scalar reads.
    run_validation = (
        harness.validate if architecture == "idlm" else harness.run_validation
    )
    dynamo_counters.clear()
    run_validation()
    torch.cuda.synchronize(device)
    validation_warmup_unique_graphs = int(
        dynamo_counters["stats"]["unique_graphs"]
    )
    validation_warmup_graph_breaks = counter_total("graph_break")
    dynamo_counters.clear()
    torch.cuda.reset_peak_memory_stats(device)
    validation_powers = array("d")
    validation_utilizations = array("d")
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
    validation_sampler.start()
    torch.cuda.synchronize(device)
    validation_started = time.perf_counter()
    try:
        validation = run_validation()
        torch.cuda.synchronize(device)
        validation_seconds = time.perf_counter() - validation_started
    finally:
        validation_stop.set()
        validation_sampler.join(timeout=2.0)
    validation_power = _summary(validation_powers)
    validation_utilization = _summary(validation_utilizations)
    validation_unique_graphs = int(dynamo_counters["stats"]["unique_graphs"])
    validation_recompiles = counter_total("recompiles")
    validation_graph_breaks = counter_total("graph_break")
    validation_peak_allocated = torch.cuda.max_memory_allocated(device)
    validation_peak_reserved = torch.cuda.max_memory_reserved(device)
    validation_eligible = (
        total_memory - validation_peak_reserved >= minimum_headroom
        and validation_unique_graphs == 0
        and validation_recompiles == 0
        and validation_graph_breaks == 0
        and _meets_utilization_policy(
            validation_power,
            validation_utilization,
            minimum_mean_utilization=minimum_mean_utilization,
            minimum_p10_utilization=minimum_p10_utilization,
        )
    )
    validation_rows = min(256, len(harness.validation_dataset))
    validation_metrics = (
        {
            "clean_bpb": validation.clean_bpb,
            "proposal_loss": validation.proposal_loss,
            "targets": validation.proposal_targets,
        }
        if architecture == "idlm"
        else {
            "denoising_ce_nats_per_atom": validation.denoising_ce_nats_per_atom,
            "ar_anchor_bpb": validation.ar_anchor_bpb,
            "targets": validation.diffusion_targets,
        }
        if architecture == "diffusion_gemma"
        else {
            "conditional_canvas_duo_nelbo_nats_per_atom": (
                validation.conditional_canvas_nelbo_nats_per_atom
            ),
            "clean_ar_nats_per_atom": validation.clean_ar_nats_per_atom,
            "ar_anchor_bpb": validation.ar_anchor_bpb,
            "targets": validation.targets,
        }
    )
    validation_evidence: dict[str, object] = {
        "rows": validation_rows,
        "batch_size": validation_batch_size,
        "elapsed_seconds": validation_seconds,
        "rows_per_second": validation_rows / validation_seconds,
        **validation_metrics,
        "power_w": validation_power,
        "gpu_utilization_percent": validation_utilization,
        "cuda_peak_allocated_bytes": validation_peak_allocated,
        "cuda_peak_reserved_bytes": validation_peak_reserved,
        "cuda_reserved_headroom_bytes": total_memory - validation_peak_reserved,
        "warmup_unique_graphs": validation_warmup_unique_graphs,
        "warmup_graph_breaks": validation_warmup_graph_breaks,
        "measured_unique_graphs": validation_unique_graphs,
        "measured_recompiles": validation_recompiles,
        "measured_graph_breaks": validation_graph_breaks,
        "eligible": validation_eligible,
    }
    eligible = (
        reserved_headroom >= minimum_headroom
        and unique_graphs == 0
        and recompiles == 0
        and graph_breaks == 0
        and _meets_utilization_policy(
            power_summary,
            utilization_summary,
            minimum_mean_utilization=minimum_mean_utilization,
            minimum_p10_utilization=minimum_p10_utilization,
        )
        and validation_eligible
        and math.isfinite(update_ms)
    )
    return {
        "status": "ok",
        "eligible": eligible,
        "microbatch": microbatch,
        "global_batch": global_batch,
        "microsteps_per_update": microsteps,
        "tail_microbatch": tail,
        "tail_exercised": tail != microbatch,
        "row_length_bytes": PRODUCTION_ROW_LENGTH,
        "warmup_updates": warmup_updates,
        "measured_updates": measured_updates,
        "elapsed_seconds": elapsed_seconds,
        "update_ms": update_ms,
        "updates_per_second": measured_updates / elapsed_seconds,
        "primary_target_kind": (
            "proposal_bytes"
            if architecture == "idlm"
            else "duo_nelbo_atoms"
            if architecture == "duo"
            else "diffusion_atoms"
        ),
        "primary_targets": primary_targets,
        "primary_targets_per_second": primary_targets / elapsed_seconds,
        "anchor_target_kind": (
            "none"
            if architecture == "duo" and duo_clean_ar_weight(duo_objective) == 0.0
            else "clean_bytes"
        ),
        "anchor_targets": anchor_targets,
        "anchor_targets_per_second": anchor_targets / elapsed_seconds,
        "all_useful_targets": primary_targets + anchor_targets,
        "all_useful_targets_per_second": (
            primary_targets + anchor_targets
        ) / elapsed_seconds,
        "parameter_count": parameter_count,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_total_bytes": total_memory,
        "cuda_reserved_headroom_bytes": reserved_headroom,
        "required_headroom_bytes": minimum_headroom,
        "power_w": power_summary,
        "gpu_utilization_percent": utilization_summary,
        "warmup_unique_graphs": warmup_unique_graphs,
        "warmup_graph_breaks": warmup_graph_breaks,
        "measured_unique_graphs": unique_graphs,
        "measured_recompiles": recompiles,
        "measured_graph_breaks": graph_breaks,
        "validation": validation_evidence,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture", choices=("idlm", "diffusion_gemma", "duo"), required=True
    )
    parser.add_argument(
        "--candidate-microbatches",
        type=parse_candidate_microbatches,
        required=True,
    )
    parser.add_argument(
        "--data-path", type=Path, default=Path("data/byte_diffusion_aligned_v5")
    )
    parser.add_argument("--row-length", type=int, default=PRODUCTION_ROW_LENGTH)
    parser.add_argument("--global-batch", type=int, default=MATCHED_GLOBAL_BATCH)
    parser.add_argument(
        "--duo-canvas-length",
        type=int,
        default=512,
        help="training canvas width for the named Byte-Duo geometry",
    )
    parser.add_argument(
        "--duo-branches",
        type=int,
        default=8,
        help="simultaneous independently corrupted canvases per Byte-Duo row",
    )
    parser.add_argument(
        "--duo-objective",
        choices=DUO_TRAINING_OBJECTIVES,
        default=os.getenv("BYTE_DUO_OBJECTIVE", PURE_DUO_OBJECTIVE),
        help="named Byte-Duo optimizer objective; ignored by other architectures",
    )
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--measured-updates", type=int, default=4)
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        help="independent held-row batch; defaults to the largest training candidate",
    )
    parser.add_argument("--seed", type=int, default=1_337)
    parser.add_argument(
        "--min-headroom-fraction", type=float, default=HEADROOM_FRACTION
    )
    parser.add_argument(
        "--min-headroom-gib", type=float, default=HEADROOM_MINIMUM_GIB
    )
    parser.add_argument(
        "--min-mean-power-w",
        type=float,
        default=SUSTAINED_GPU_POLICY["minimum_mean_power_w"],
    )
    parser.add_argument(
        "--min-p10-power-w",
        type=float,
        default=SUSTAINED_GPU_POLICY["minimum_p10_power_w"],
    )
    parser.add_argument(
        "--min-mean-utilization",
        type=float,
        default=SUSTAINED_GPU_POLICY["minimum_mean_utilization_percent"],
    )
    parser.add_argument(
        "--min-p10-utilization",
        type=float,
        default=SUSTAINED_GPU_POLICY["minimum_p10_utilization_percent"],
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _validate_source(architecture: str) -> dict[str, object]:
    if architecture == "idlm":
        observed = idlm_source_provenance()
        variable = "BYTE_IDLM_EXPECTED_SOURCE_SHA256"
    elif architecture == "diffusion_gemma":
        observed = diffusion_gemma_source_provenance()
        variable = "BYTE_DIFFUSION_GEMMA_EXPECTED_SOURCE_SHA256"
    else:
        observed = duo_source_provenance()
        variable = "BYTE_DUO_EXPECTED_SOURCE_SHA256"
    expected = os.environ.get(variable)
    if expected is None:
        raise ValueError(f"{variable} is required for a readiness benchmark")
    if observed["sha256"] != expected:
        raise ValueError(
            f"source differs from pinned readiness contract: expected {expected}, "
            f"observed {observed['sha256']}"
        )
    return observed


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark must be submitted through mlq")
    if args.row_length != PRODUCTION_ROW_LENGTH:
        raise ValueError("readiness evidence requires exact 8,192-byte rows")
    if args.global_batch != MATCHED_GLOBAL_BATCH:
        raise ValueError("matched readiness evidence requires global batch 249")
    if args.architecture == "duo":
        duo_geometry_contract(args.duo_canvas_length, args.duo_branches)
    elif args.duo_canvas_length != 512 or args.duo_branches != 8:
        raise ValueError("Duo geometry selectors are Duo-only")
    if args.architecture != "duo" and args.duo_objective != PURE_DUO_OBJECTIVE:
        raise ValueError("the Duo objective selector is Duo-only")
    if min(args.warmup_updates, args.measured_updates) <= 0:
        raise ValueError("warmup and measured update counts must be positive")
    if args.warmup_updates < 2 or args.measured_updates < 4:
        raise ValueError("readiness requires at least 2 warmup and 4 measured updates")
    observed_policy = {
        "minimum_mean_power_w": args.min_mean_power_w,
        "minimum_p10_power_w": args.min_p10_power_w,
        "minimum_mean_utilization_percent": args.min_mean_utilization,
        "minimum_p10_utilization_percent": args.min_p10_utilization,
    }
    if observed_policy != SUSTAINED_GPU_POLICY or (
        args.min_headroom_fraction != HEADROOM_FRACTION
        or args.min_headroom_gib != HEADROOM_MINIMUM_GIB
    ):
        raise ValueError("readiness thresholds must equal the production policy")
    validation_batch_size = (
        max(args.candidate_microbatches)
        if args.validation_batch_size is None
        else args.validation_batch_size
    )
    if validation_batch_size <= 0:
        raise ValueError("validation batch size must be positive")
    source = _validate_source(args.architecture)
    expected_data = os.environ.get("BYTE_DIFFUSION_EXPECTED_DATA_SHA256")
    if expected_data is None:
        raise ValueError("BYTE_DIFFUSION_EXPECTED_DATA_SHA256 is required")

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.set_float32_matmul_precision("high")
    recipe = "causal_only" if args.architecture == "idlm" else "blt_d"
    manifest, train_dataset, validation_dataset = load_data_directory(
        args.data_path,
        chunk_size=args.row_length,
        recipe=recipe,
        required_branch_bytes=(
            DUO_DATA_REQUIRED_BRANCH_BYTES if args.architecture == "duo"
            else 512 if args.architecture == "diffusion_gemma"
            else 0
        ),
        branch_span_length=(
            DUO_DATA_BRANCH_SPAN_LENGTH if args.architecture == "duo"
            else 512 if args.architecture == "diffusion_gemma"
            else 0
        ),
        validation_chunk_limit=256,
        expected_payload_sha256=expected_data,
    )
    if len(validation_dataset) != 256:
        raise ValueError("architecture readiness requires exactly 256 validation rows")
    dataset_manifest = json.loads((args.data_path / "manifest.json").read_text())
    if dataset_manifest.get("payload_sha256") != expected_data:
        raise ValueError("dataset payload differs from pinned readiness contract")

    total_memory = torch.cuda.get_device_properties(device).total_memory
    minimum_headroom = required_headroom_bytes(
        total_memory,
        fraction=args.min_headroom_fraction,
        minimum_gib=args.min_headroom_gib,
    )
    results: dict[str, dict[str, object]] = {}
    for microbatch in args.candidate_microbatches:
        try:
            with DeviceBatchPrefetcher(device=device) as batch_prefetcher:
                result = _benchmark_candidate(
                    args.architecture,
                    manifest=manifest,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    device=device,
                    microbatch=microbatch,
                    validation_batch_size=validation_batch_size,
                    global_batch=args.global_batch,
                    duo_canvas_length=args.duo_canvas_length,
                    duo_branches=args.duo_branches,
                    duo_objective=args.duo_objective,
                    warmup_updates=args.warmup_updates,
                    measured_updates=args.measured_updates,
                    seed=args.seed,
                    minimum_headroom=minimum_headroom,
                    minimum_mean_power_w=args.min_mean_power_w,
                    minimum_p10_power_w=args.min_p10_power_w,
                    minimum_mean_utilization=args.min_mean_utilization,
                    minimum_p10_utilization=args.min_p10_utilization,
                    batch_prefetcher=batch_prefetcher,
                )
        except BaseException as error:
            if not _is_cuda_oom(error):
                raise
            microsteps, tail = tail_geometry(args.global_batch, microbatch)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            result = {
                "status": "oom",
                "eligible": False,
                "error": str(error),
                "microbatch": microbatch,
                "global_batch": args.global_batch,
                "microsteps_per_update": microsteps,
                "tail_microbatch": tail,
                "tail_exercised": tail != microbatch,
                "row_length_bytes": args.row_length,
                "cuda_peak_allocated_bytes": peak_allocated,
                "cuda_peak_reserved_bytes": peak_reserved,
                "cuda_total_bytes": total_memory,
                "cuda_reserved_headroom_bytes": total_memory - peak_reserved,
                "required_headroom_bytes": minimum_headroom,
            }
        results[str(microbatch)] = result
        print("readiness_candidate " + json.dumps(result, sort_keys=True), flush=True)
        del result
        _reset_candidate_state()

    selected = choose_fastest_fitting(results)
    model_contract, workload_contract = readiness_workload_contract(
        args.architecture,
        duo_objective=args.duo_objective,
        duo_canvas_length=args.duo_canvas_length,
        duo_branches=args.duo_branches,
    )
    if args.architecture == "duo":
        model_contract = model_config_from_env().to_dict()
    report = {
        "schema": "byte_diffusion_architecture_readiness/v1",
        "architecture": args.architecture,
        "evidence_kind": "full_size_compiled_training_and_validation_systems_benchmark",
        "training_quality_evidence": False,
        "source": source,
        "model_config": model_contract,
        "workload": workload_contract,
        "benchmark_harness_source": benchmark_harness_provenance(),
        "dataset_payload_sha256": expected_data,
        "row_length_bytes": args.row_length,
        "global_batch": args.global_batch,
        "duo_branches": args.duo_branches,
        "duo_canvas_length": args.duo_canvas_length,
        "candidate_microbatches": args.candidate_microbatches,
        "headroom_policy": {
            "fraction": args.min_headroom_fraction,
            "minimum_gib": args.min_headroom_gib,
            "required_bytes": minimum_headroom,
        },
        "sustained_gpu_policy": {
            "minimum_mean_power_w": args.min_mean_power_w,
            "minimum_p10_power_w": args.min_p10_power_w,
            "minimum_mean_utilization_percent": args.min_mean_utilization,
            "minimum_p10_utilization_percent": args.min_p10_utilization,
        },
        "gpu": {
            "name": torch.cuda.get_device_name(device),
            "index": device.index,
            "total_memory_bytes": total_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nvidia_smi_selector": nvidia_smi_selector(device),
        },
        "runtime": {
            "compiled": True,
            "device_type": "cuda",
            "validation_rows": 256,
            "world_size": 1,
        },
        "results": results,
        "selected_microbatch": selected,
        "selected_validation_batch_size": validation_batch_size,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("readiness_result " + json.dumps(report, sort_keys=True), flush=True)
    if selected is None:
        raise RuntimeError("no candidate met OOM, headroom, telemetry, and graph criteria")


if __name__ == "__main__":
    main()
