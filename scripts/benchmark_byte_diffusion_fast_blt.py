#!/usr/bin/env python3
"""Authenticated single-GPU readiness benchmark for exhaustive Fast-BLT-D4.

This executes full 249-row training updates, all 2,048 declared AR validation
rows, and the trainer's exact evenly spaced 256-row diffusion proxy. Submit it
through ``mlq``; it intentionally has no reduced or synthetic GPU mode.
"""

from __future__ import annotations

import argparse
from array import array
from dataclasses import asdict, dataclass
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
import traceback
from typing import Mapping, Sequence

import numpy as np
import torch
from torch._dynamo.utils import counters as dynamo_counters
from torch._dynamo.exc import (
    FailOnRecompileLimitHit,
    InvalidBackend,
    RecompileError,
    TorchDynamoException,
)
from torch._inductor.exc import (
    CppCompileError,
    CppWrapperCodegenError,
    InvalidCxxCompiler,
    OperatorIssue,
    SubgraphLoweringException,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.config import (
    FAST_BLT_ENTROPY_B4_COMPLETE_PRESETS,
    ByteDiffusionConfig,
    fast_blt_complete_model_config,
    model_config_from_env,
)
from pretraining.byte_diffusion.attention import (
    RaggedBltLayout,
    build_ragged_blt_block_mask,
)
from pretraining.byte_diffusion.data import DeterministicChunkCursor
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.readiness import SUSTAINED_GPU_POLICY
from pretraining.byte_diffusion.readiness_fast_blt import (
    FAST_BLT_AR_VALIDATION_BATCH,
    FAST_BLT_BLOCK_LENGTH,
    FAST_BLT_COMPILE_ERROR_TYPES,
    FAST_BLT_DIFFUSION_VALIDATION_INDICES_SHA256,
    FAST_BLT_DIFFUSION_VALIDATION_ROWS,
    FAST_BLT_GRAPH_QUIESCENCE_SCHEMA,
    FAST_BLT_GLOBAL_BATCH,
    FAST_BLT_MAX_GRAPH_WARMUP_UPDATES,
    FAST_BLT_ORIGIN_POLICY,
    FAST_BLT_READINESS_SCHEMA,
    FAST_BLT_ROW_LENGTH,
    FAST_BLT_RUNTIME_SCHEMA,
    FAST_BLT_TIMING_SCHEMA,
    FAST_BLT_TELEMETRY_LIFECYCLE_SCHEMA,
    FAST_BLT_TELEMETRY_MAX_COVERAGE_GAP_SECONDS,
    FAST_BLT_TELEMETRY_QUERY_INTERVAL_SECONDS,
    FAST_BLT_TELEMETRY_QUERY_TIMEOUT_SECONDS,
    FAST_BLT_TELEMETRY_THREAD_JOIN_TIMEOUT_SECONDS,
    FAST_BLT_TELEMETRY_TERMINAL_SCHEMA,
    FAST_BLT_VALIDATION_BATCH,
    FAST_BLT_VALIDATION_ROWS,
    FAST_BLT_WORKLOAD_SCHEMA,
    TIMING_COHERENCE_ABS_MS,
    TIMING_COHERENCE_REL_TOL,
    FastBltGeometry,
    candidate_passes,
    canonical_json_sha256,
    common_run_contract,
    fast_blt_allocator_contract,
    require_fast_blt_allocator_environment,
    required_headroom_bytes,
    run_contract_without_identity,
)
from pretraining.byte_diffusion.telemetry import nvidia_smi_selector
from pretraining.byte_diffusion.training import (
    ByteDiffusionTrainer,
    DistributedContext,
    StepMetrics,
    TrainingRunConfig,
    UpdateExecutionLedger,
    ValidationExecutionLedger,
    ValidationMetrics,
    group_rows_by_ragged_physical_workload,
    load_data_directory,
    prepare_exhaustive_blt_corruption,
)
from scripts.train_byte_diffusion import _local_imports, training_source_provenance


MIN_WARMUP_UPDATES = 2
MIN_MEASURED_UPDATES = 4
MIN_VALIDATION_WARMUP = 1
MIN_VALIDATION_MEASURED = 2


class GraphNonquiescenceError(RuntimeError):
    """A bounded warmup exhausted without reaching a representative plateau."""

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        evidence: Mapping[str, object],
        workload: Mapping[str, object],
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.evidence = dict(evidence)
        self.workload = dict(workload)


class ValidationCacheNonquiescenceError(RuntimeError):
    """Measured validation rebuilt deterministic topology after warmup."""

    def __init__(
        self,
        *,
        evidence: Mapping[str, object],
        workload: Mapping[str, object],
    ) -> None:
        super().__init__("validation staging caches were not quiescent")
        self.evidence = dict(evidence)
        self.workload = dict(workload)


def _require_unchanged_harness(
    before: Mapping[str, object], after: Mapping[str, object]
) -> None:
    if before != after:
        raise RuntimeError("Fast-BLT benchmark harness changed during execution")


def _materialized_update_count(run: TrainingRunConfig) -> int:
    """Count production updates which execute synchronized metric materialization."""

    iterations = run.iterations
    log_every = run.train_log_every
    validation_every = run.val_loss_every
    common = math.lcm(log_every, validation_every)
    scheduled = (
        iterations // log_every
        + iterations // validation_every
        - iterations // common
    )
    if iterations % log_every and iterations % validation_every:
        scheduled += 1
    return scheduled


def parse_candidate_microbatches(value: str) -> tuple[int, ...]:
    try:
        candidates = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "candidate microbatches must be comma-separated integers"
        ) from error
    if not candidates or any(candidate <= 0 for candidate in candidates):
        raise argparse.ArgumentTypeError("candidate microbatches must be positive")
    if len(set(candidates)) != len(candidates):
        raise argparse.ArgumentTypeError("candidate microbatches must be unique")
    return tuple(sorted(candidates))


def benchmark_harness_provenance() -> dict[str, object]:
    """Fingerprint the benchmark closure and production trainer sources."""

    pending = {
        Path(__file__).resolve(),
        (REPO_ROOT / "pretraining/byte_diffusion/readiness_fast_blt.py").resolve(),
    }
    observed: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in observed:
            continue
        if not path.is_relative_to(REPO_ROOT):
            raise ValueError(f"readiness source escaped repository: {path}")
        observed.add(path)
        pending.update(item for item in _local_imports(path) if item not in observed)
    training = training_source_provenance()
    training_files = training.get("files")
    if not isinstance(training_files, Mapping):
        raise ValueError("training provenance omitted its source map")
    observed.update((REPO_ROOT / name).resolve() for name in training_files)
    files: dict[str, str] = {}
    digest = hashlib.sha256()
    for path in sorted(observed, key=lambda item: item.relative_to(REPO_ROOT).as_posix()):
        relative = path.relative_to(REPO_ROOT).as_posix()
        file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        files[relative] = file_sha
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha))
    return {
        "schema": "byte_diffusion_fast_blt_harness_source/v2",
        "sha256": digest.hexdigest(),
        "files": files,
    }


def build_benchmark_run_config(
    production: TrainingRunConfig, microbatch: int
) -> TrainingRunConfig:
    """Return the exact named production run; readiness is not a sweep."""

    if production.preset not in FAST_BLT_ENTROPY_B4_COMPLETE_PRESETS:
        raise ValueError("readiness requires the complete Fast-BLT production preset")
    if microbatch != 32 or production.microbatch_per_rank != 32:
        raise ValueError("complete-preset readiness is pinned to microbatch 32")
    return production


def _counter_total(name: str) -> int:
    return int(sum(dynamo_counters[name].values()))


def _graph_snapshot(prefix: str) -> dict[str, int]:
    return {
        f"{prefix}_{name}": value for name, value in _graph_counts().items()
    }


def _graph_counts() -> dict[str, int]:
    """Return unprefixed compiler counters for per-update plateau checks."""

    return {
        "unique_graphs": int(dynamo_counters["stats"]["unique_graphs"]),
        "recompiles": _counter_total("recompiles"),
        "graph_breaks": _counter_total("graph_break"),
    }


def _graph_delta(
    before: Mapping[str, int], after: Mapping[str, int]
) -> dict[str, int]:
    """Return a fail-closed nonnegative compiler-counter delta."""

    if before.keys() != after.keys():
        raise ValueError("compiler counter snapshots do not align")
    delta = {name: int(after[name]) - int(before[name]) for name in before}
    if any(value < 0 for value in delta.values()):
        raise ValueError("compiler counters moved backwards during one update")
    return delta


def _attach_update_timing(
    record: Mapping[str, object],
    *,
    host_duration_ms: float,
    cuda_event_ms: float,
    synchronized_host_call: bool,
) -> dict[str, object]:
    """Bind unambiguous host-call and device-stream durations to one update."""

    if (
        not math.isfinite(host_duration_ms)
        or host_duration_ms <= 0
        or not math.isfinite(cuda_event_ms)
        or cuda_event_ms <= 0
    ):
        raise ValueError("update timings must be finite and positive")
    result = dict(record)
    field = (
        "host_synchronized_call_ms"
        if synchronized_host_call
        else "host_enqueue_ms"
    )
    result[field] = float(host_duration_ms)
    result["cuda_event_ms"] = float(cuda_event_ms)
    return result


def _attach_execution_ledger(
    topology: Mapping[str, object], ledger: UpdateExecutionLedger | None
) -> dict[str, object]:
    """Bind cheap actual batching facts and reject planner/executor drift."""

    if ledger is None:
        raise AssertionError("ordinary update omitted its execution ledger")
    record = dict(topology)
    actual = {
        "actual_step": ledger.step,
        "actual_rows": ledger.rows,
        "actual_microsteps": ledger.microsteps,
        "actual_group_row_cu_seqlens": ledger.group_row_cu_seqlens,
        "actual_group_physical_positions": ledger.group_physical_positions,
        "actual_max_microbatch": ledger.max_microbatch,
        "actual_max_physical_positions": ledger.max_physical_positions,
    }
    expected = {
        "actual_rows": record["rows"],
        "actual_microsteps": record["planned_microsteps"],
        "actual_group_row_cu_seqlens": tuple(
            record["planned_group_row_cu_seqlens"]  # type: ignore[arg-type]
        ),
        "actual_group_physical_positions": tuple(
            record["planned_group_physical_positions"]  # type: ignore[arg-type]
        ),
        "actual_max_microbatch": record["planned_max_microbatch"],
        "actual_max_physical_positions": record[
            "planned_max_physical_positions"
        ],
    }
    if any(actual[name] != value for name, value in expected.items()):
        raise AssertionError("trainer execution differed from planned ragged topology")
    record.update(actual)
    return record


def _reset_compiler_counters() -> None:
    dynamo_counters.clear()


def _reset_candidate_state(device: torch.device) -> None:
    gc.collect()
    torch.compiler.reset()
    _reset_compiler_counters()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _summary(samples: array) -> dict[str, object]:
    raw = tuple(float(value) for value in samples)
    if not raw:
        return {
            "count": 0,
            "mean": None,
            "p10": None,
            "minimum": None,
            "maximum": None,
            "samples": raw,
            "samples_sha256": canonical_json_sha256(raw),
        }
    values = np.frombuffer(samples, dtype=np.float64)
    return {
        "count": len(samples),
        "mean": math.fsum(samples) / len(samples),
        "p10": float(np.quantile(values, 0.1, method="linear")),
        "minimum": min(samples),
        "maximum": max(samples),
        "samples": raw,
        "samples_sha256": canonical_json_sha256(raw),
    }


@dataclass(frozen=True)
class _TelemetryObservation:
    seq: int
    query_start_ns: int
    query_end_ns: int
    power_w: float
    utilization_percent: float
    free_mib: float
    used_mib: float

    def record(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class _TelemetrySamplerError:
    seq: int
    timestamp_ns: int
    kind: str
    returncode: int | None

    def record(self) -> dict[str, int | str | None]:
        return asdict(self)


class TelemetrySamplerLifecycleError(RuntimeError):
    """The bounded nvidia-smi worker failed or outlived its context."""

    def __init__(self, evidence: Mapping[str, object]) -> None:
        super().__init__("telemetry sampler did not terminate cleanly")
        self.evidence = dict(evidence)


class _Telemetry:
    def __init__(self, device: torch.device) -> None:
        self.selector = nvidia_smi_selector(device)
        self._observations: list[_TelemetryObservation] = []
        self._errors: list[_TelemetrySamplerError] = []
        self._samples_lock = threading.Lock()
        self._measurement_begin_ns: int | None = None
        self._measurement_end_ns: int | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self._stop_requested = False
        self._thread_terminated = False

    def _loop(self) -> None:
        while not self.stop.wait(FAST_BLT_TELEMETRY_QUERY_INTERVAL_SECONDS):
            query_start_ns = time.monotonic_ns()
            try:
                observed = subprocess.run(
                    (
                        "nvidia-smi",
                        "--query-gpu=power.draw,utilization.gpu,memory.free,memory.used",
                        "--format=csv,noheader,nounits",
                        f"--id={self.selector}",
                    ),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=FAST_BLT_TELEMETRY_QUERY_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                self._record_error("timeout", returncode=None)
                continue
            except BaseException:
                self._record_error("exception", returncode=None)
                return
            query_end_ns = time.monotonic_ns()
            if observed.returncode:
                self._record_error("nonzero_returncode", observed.returncode)
                continue
            parts = observed.stdout.partition("\n")[0].split(",")
            if len(parts) != 4:
                self._record_error("malformed_output", returncode=None)
                continue
            try:
                power, utilization, free_mib, used_mib = map(
                    lambda part: float(part.strip().removesuffix(" %")), parts
                )
            except ValueError:
                self._record_error("malformed_value", returncode=None)
                continue
            self._record_observation(
                query_start_ns=query_start_ns,
                query_end_ns=query_end_ns,
                power_w=power,
                utilization_percent=utilization,
                free_mib=free_mib,
                used_mib=used_mib,
            )

    def _record_error(self, kind: str, returncode: int | None) -> None:
        with self._samples_lock:
            self._errors.append(
                _TelemetrySamplerError(
                    seq=len(self._errors),
                    timestamp_ns=time.monotonic_ns(),
                    kind=kind,
                    returncode=returncode,
                )
            )

    def _record_observation(
        self,
        *,
        query_start_ns: int,
        query_end_ns: int,
        power_w: float,
        utilization_percent: float,
        free_mib: float,
        used_mib: float,
    ) -> None:
        if query_start_ns < 0 or query_end_ns < query_start_ns:
            raise ValueError("telemetry query timestamps are invalid")
        with self._samples_lock:
            self._observations.append(
                _TelemetryObservation(
                    seq=len(self._observations),
                    query_start_ns=query_start_ns,
                    query_end_ns=query_end_ns,
                    power_w=power_w,
                    utilization_percent=utilization_percent,
                    free_mib=free_mib,
                    used_mib=used_mib,
                )
            )

    def mark_measurement_start(self) -> None:
        """Mark the exact lower timestamp bound for scored observations."""

        with self._samples_lock:
            if self._measurement_begin_ns is not None:
                raise RuntimeError("telemetry measurement start is already marked")
            self._measurement_begin_ns = time.monotonic_ns()

    def mark_measurement_end(self) -> None:
        """Mark the exact upper timestamp bound for scored observations."""

        with self._samples_lock:
            if self._measurement_begin_ns is None:
                raise RuntimeError("telemetry measurement start is not marked")
            if self._measurement_end_ns is not None:
                raise RuntimeError("telemetry measurement end is already marked")
            self._measurement_end_ns = time.monotonic_ns()

    def __enter__(self) -> "_Telemetry":
        self.thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop.set()
        with self._samples_lock:
            self._stop_requested = True
        self.thread.join(timeout=FAST_BLT_TELEMETRY_THREAD_JOIN_TIMEOUT_SECONDS)
        with self._samples_lock:
            self._thread_terminated = not self.thread.is_alive()
            clean = self._thread_terminated and not self._errors
        if not clean:
            raise TelemetrySamplerLifecycleError(self._terminal_evidence())

    def _terminal_evidence(self) -> dict[str, object]:
        with self._samples_lock:
            errors = tuple(value.record() for value in self._errors)
            return {
                "schema": FAST_BLT_TELEMETRY_TERMINAL_SCHEMA,
                "query_timeout_seconds": FAST_BLT_TELEMETRY_QUERY_TIMEOUT_SECONDS,
                "thread_join_timeout_seconds": (
                    FAST_BLT_TELEMETRY_THREAD_JOIN_TIMEOUT_SECONDS
                ),
                "stop_requested": self._stop_requested,
                "thread_terminated": self._thread_terminated,
                "clean_completion": self._thread_terminated and not errors,
                "error_count": len(errors),
                "errors": errors,
                "errors_sha256": canonical_json_sha256(errors),
            }

    def report(self, *, required: int) -> dict[str, object]:
        mib = 1 << 20
        with self._samples_lock:
            observations = tuple(self._observations)
            errors = tuple(self._errors)
            begin_ns = self._measurement_begin_ns
            end_ns = self._measurement_end_ns
            stop_requested = self._stop_requested
            thread_terminated = self._thread_terminated
        if (begin_ns is None) != (end_ns is None):
            raise RuntimeError("telemetry measurement bounds are incomplete")
        scored = (
            observations
            if begin_ns is None
            else tuple(
                observation
                for observation in observations
                if observation.query_start_ns >= begin_ns
                and observation.query_end_ns <= end_ns
            )
        )
        measured_power = array("d", (value.power_w for value in scored))
        measured_utilization = array(
            "d", (value.utilization_percent for value in scored)
        )
        # Physical headroom is a whole-lifecycle safety domain: retain warmup
        # and boundary-straddling observations even though utilization scoring
        # uses only queries wholly inside the measured interval.
        free_bytes = tuple(int(value.free_mib * mib) for value in observations)
        used_bytes = tuple(int(value.used_mib * mib) for value in observations)
        observation_records = tuple(value.record() for value in observations)
        scored_sequences = tuple(value.seq for value in scored)
        error_records = tuple(value.record() for value in errors)
        first_gap = last_gap = maximum_gap = None
        if begin_ns is not None and end_ns is not None and scored:
            # nvidia-smi yields one observation when the subprocess returns;
            # time spent inside the query is not continuous telemetry coverage.
            first_gap = (scored[0].query_end_ns - begin_ns) / 1_000_000_000.0
            last_gap = (end_ns - scored[-1].query_end_ns) / 1_000_000_000.0
            maximum_gap = max(
                (
                    right.query_end_ns - left.query_end_ns
                    for left, right in zip(scored[:-1], scored[1:], strict=True)
                ),
                default=0,
            ) / 1_000_000_000.0
        coverage_complete = bool(
            begin_ns is not None
            and end_ns is not None
            and scored
            and first_gap is not None
            and last_gap is not None
            and maximum_gap is not None
            and max(first_gap, last_gap, maximum_gap)
            <= FAST_BLT_TELEMETRY_MAX_COVERAGE_GAP_SECONDS
        )
        return {
            "power_w": _summary(measured_power),
            "gpu_utilization_percent": _summary(measured_utilization),
            "telemetry_lifecycle": {
                "schema": FAST_BLT_TELEMETRY_LIFECYCLE_SCHEMA,
                "measurement_begin_ns": begin_ns,
                "measurement_end_ns": end_ns,
                "measurement_duration_seconds": (
                    None
                    if begin_ns is None or end_ns is None
                    else (end_ns - begin_ns) / 1_000_000_000.0
                ),
                "observation_count": len(observation_records),
                "observations": observation_records,
                "observations_sha256": canonical_json_sha256(observation_records),
                "scored_observation_sequences": scored_sequences,
                "scored_observation_sequences_sha256": canonical_json_sha256(
                    scored_sequences
                ),
                "query_interval_seconds": (
                    FAST_BLT_TELEMETRY_QUERY_INTERVAL_SECONDS
                ),
                "query_timeout_seconds": FAST_BLT_TELEMETRY_QUERY_TIMEOUT_SECONDS,
                "thread_join_timeout_seconds": (
                    FAST_BLT_TELEMETRY_THREAD_JOIN_TIMEOUT_SECONDS
                ),
                "maximum_coverage_gap_seconds": (
                    FAST_BLT_TELEMETRY_MAX_COVERAGE_GAP_SECONDS
                ),
                "first_scored_start_gap_seconds": first_gap,
                "last_scored_end_gap_seconds": last_gap,
                "maximum_scored_query_gap_seconds": maximum_gap,
                "coverage_complete": coverage_complete,
                "stop_requested": stop_requested,
                "thread_terminated": thread_terminated,
                "clean_completion": thread_terminated and not errors,
                "error_count": len(error_records),
                "errors": error_records,
                "errors_sha256": canonical_json_sha256(error_records),
            },
            "physical_environment": {
                "sample_count": len(free_bytes),
                "minimum_free_bytes": min(free_bytes) if free_bytes else 0,
                "maximum_used_bytes": max(used_bytes) if used_bytes else 0,
                "required_free_bytes": required,
                "free_bytes_samples": free_bytes,
                "free_bytes_samples_sha256": canonical_json_sha256(free_bytes),
                "used_bytes_samples": used_bytes,
                "used_bytes_samples_sha256": canonical_json_sha256(used_bytes),
            },
        }


def _memory_snapshot(device: torch.device, *, required: int) -> dict[str, int]:
    total = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    physical_free, physical_total = torch.cuda.mem_get_info(device)
    return {
        "cuda_peak_allocated_bytes": allocated,
        "cuda_peak_reserved_bytes": reserved,
        "cuda_total_bytes": total,
        "intrinsic_reserved_headroom_bytes": total - reserved,
        "required_headroom_bytes": required,
        "instantaneous_physical_free_bytes": physical_free,
        "instantaneous_physical_total_bytes": physical_total,
    }


def _origin_counts(chunks, indices: Sequence[int]) -> np.ndarray:
    counter = getattr(chunks, "training_blt_origin_counts", None)
    resolved_indices = np.asarray(indices, dtype=np.int64)
    # Validation is an authenticated deterministic view over the mapped split.
    # Resolve that view vectorially rather than materializing 256 Python rows.
    if counter is None:
        source = getattr(chunks, "source", None)
        subset_indices = getattr(chunks, "indices", None)
        counter = getattr(source, "training_blt_origin_counts", None)
        if counter is not None and subset_indices is not None:
            resolved_indices = np.take(
                np.asarray(subset_indices, dtype=np.int64), resolved_indices
            )
    if counter is None:
        raise ValueError("readiness dataset omitted native exhaustive-origin counts")
    counts = np.asarray(
        counter(
            resolved_indices,
            block_length=FAST_BLT_BLOCK_LENGTH,
            eot_id=256,
        ),
        dtype=np.int64,
    )
    if counts.shape != (len(indices),) or bool((counts <= 0).any()):
        raise ValueError("readiness encountered invalid per-row entropy-origin counts")
    return counts


def _diffusion_validation_indices() -> np.ndarray:
    """Mirror trainer.validate's exact evenly spaced 256-of-2,048 selection."""

    positions = np.arange(FAST_BLT_DIFFUSION_VALIDATION_ROWS, dtype=np.int64)
    indices = (
        (2 * positions + 1)
        * FAST_BLT_VALIDATION_ROWS
        // (2 * FAST_BLT_DIFFUSION_VALIDATION_ROWS)
    )
    if (
        indices.shape != (FAST_BLT_DIFFUSION_VALIDATION_ROWS,)
        or np.unique(indices).size != FAST_BLT_DIFFUSION_VALIDATION_ROWS
        or int(indices[-1]) >= FAST_BLT_VALIDATION_ROWS
    ):
        raise AssertionError("diffusion validation selector is not exact and unique")
    indices.setflags(write=False)
    if hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest() != (
        FAST_BLT_DIFFUSION_VALIDATION_INDICES_SHA256
    ):
        raise AssertionError("diffusion validation selector identity drifted")
    return indices


def _topology_evidence(
    *, indices: np.ndarray, chunks, run: TrainingRunConfig, validation: bool,
) -> dict[str, object]:
    counts = _origin_counts(chunks, indices)
    workloads = FAST_BLT_ROW_LENGTH + counts * FAST_BLT_BLOCK_LENGTH
    grouping = group_rows_by_ragged_physical_workload(
        indices, workloads,
        max_batch_size=(
            run.validation_microbatch_per_rank
            if validation
            else run.microbatch_per_rank
        ),
        physical_token_budget=run.microbatch_token_budget,
        sort_by_workload=(False if validation else run.length_sorted_microbatches),
    )
    boundaries = grouping.row_cu_seqlens
    group_totals = grouping.group_physical_positions
    group_sizes = np.diff(boundaries)
    ordered_counts = (
        counts[np.argsort(workloads, kind="stable")]
        if (False if validation else run.length_sorted_microbatches)
        else counts
    )
    origin_prefix = np.concatenate(
        (np.zeros(1, dtype=np.int64), ordered_counts.cumsum(dtype=np.int64))
    )
    group_origin_counts = origin_prefix[boundaries[1:]] - origin_prefix[
        boundaries[:-1]
    ]
    serialized_boundaries = tuple(int(value) for value in boundaries)
    serialized_group_totals = tuple(int(value) for value in group_totals)
    serialized_ordered_origins = tuple(int(value) for value in ordered_counts)
    serialized_group_origins = tuple(int(value) for value in group_origin_counts)
    group_row_hashes = tuple(
        hashlib.sha256(
            np.asarray(grouping.ordered_rows[start:stop], dtype="<i8").tobytes()
        ).hexdigest()
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
    )
    return {
        "rows": int(len(indices)),
        "origin_count": int(counts.sum(dtype=np.int64)),
        "max_row_origin_count": int(counts.max()),
        "total_physical_positions": int(workloads.sum(dtype=np.int64)),
        "planned_microsteps": int(group_sizes.size),
        "planned_min_microbatch": int(group_sizes.min()),
        "planned_max_microbatch": int(group_sizes.max()),
        "planned_max_physical_positions": int(group_totals.max()),
        "planned_group_row_cu_seqlens": serialized_boundaries,
        "planned_group_row_cu_seqlens_sha256": canonical_json_sha256(
            serialized_boundaries
        ),
        "planned_group_physical_positions": serialized_group_totals,
        "planned_group_physical_positions_sha256": canonical_json_sha256(
            serialized_group_totals
        ),
        "ordered_row_origin_counts": serialized_ordered_origins,
        "ordered_row_origin_counts_sha256": canonical_json_sha256(
            serialized_ordered_origins
        ),
        "planned_group_origin_counts": serialized_group_origins,
        "planned_group_origin_counts_sha256": canonical_json_sha256(
            serialized_group_origins
        ),
        "planned_group_row_indices_sha256": group_row_hashes,
        "planned_group_row_indices_sha256_sha256": canonical_json_sha256(
            group_row_hashes
        ),
        "physical_token_budget": int(run.microbatch_token_budget),
        "row_indices_sha256": hashlib.sha256(
            np.asarray(indices, dtype="<i8").tobytes()
        ).hexdigest(),
        "row_origin_counts_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
    }


def _workload(
    records: list[dict[str, object]], *, rows: int, kind: str
) -> dict[str, object]:
    return {
        "schema": FAST_BLT_WORKLOAD_SCHEMA,
        "origin_policy": FAST_BLT_ORIGIN_POLICY,
        "block_length": FAST_BLT_BLOCK_LENGTH,
        "rows_per_update": rows,
        "batching_policy": "token_budgeted_ragged",
        "repetitions": len(records),
        "updates": records,
        "updates_sha256": canonical_json_sha256(records),
    }


def _attach_step_metrics(topology: dict[str, object], metrics: StepMetrics) -> dict[str, object]:
    record = dict(topology)
    if (
        metrics.microsteps != record["planned_microsteps"]
        or metrics.max_microbatch != record["planned_max_microbatch"]
        or metrics.max_physical_positions
        != record["planned_max_physical_positions"]
        or metrics.branch_blocks != record["origin_count"]
    ):
        raise AssertionError("trainer execution differed from precomputed ragged grouping")
    record.update(
        microsteps=metrics.microsteps,
        max_microbatch=metrics.max_microbatch,
        max_physical_positions=metrics.max_physical_positions,
        branch_blocks=metrics.branch_blocks,
        branch_atoms=metrics.branch_atoms,
        block_mask_build_ms=metrics.block_mask_build_ms,
    )
    record["active_bytes"] = metrics.diffusion_targets
    return record


def _attach_validation_metrics(
    topology: dict[str, object],
    metrics: ValidationMetrics,
    ledger: ValidationExecutionLedger | None,
) -> dict[str, object]:
    if ledger is None:
        raise AssertionError("validation omitted its execution ledger")
    record = dict(topology)
    record.update(
        microsteps=record["planned_microsteps"],
        max_microbatch=record["planned_max_microbatch"],
        max_physical_positions=record["planned_max_physical_positions"],
        branch_blocks=record["origin_count"],
        branch_atoms=int(metrics.diffusion_elbo_atoms),
        block_mask_build_ms=None,
        schedule_cache_misses=ledger.schedule_cache_misses,
        corruption_plan_cache_misses=ledger.corruption_plan_cache_misses,
        ragged_mask_metadata_cache_misses=(
            ledger.ragged_mask_metadata_cache_misses
        ),
        first_batch_cache_misses=ledger.first_batch_cache_misses,
        first_batch_waits=ledger.first_batch_waits,
        cache_misses=ledger.cache_misses,
    )
    record["active_bytes"] = metrics.diffusion_targets
    return record


def _benchmark_block_mask_build(
    *,
    train_chunks,
    run: TrainingRunConfig,
    model_config: ByteDiffusionConfig,
    device: torch.device,
    required: int,
    warmup: int = 1,
    measured: int = 4,
) -> dict[str, object]:
    """Time compiler-disabled BlockMask construction on the heaviest update group."""

    ledger = DeterministicChunkCursor(train_chunks, seed=run.seed, shuffle=True)
    update_indices = ledger.next_indices(FAST_BLT_GLOBAL_BATCH)
    counts = _origin_counts(train_chunks, update_indices)
    workloads = FAST_BLT_ROW_LENGTH + counts * FAST_BLT_BLOCK_LENGTH
    grouping = group_rows_by_ragged_physical_workload(
        update_indices,
        workloads,
        max_batch_size=run.microbatch_per_rank,
        physical_token_budget=run.microbatch_token_budget,
        sort_by_workload=run.length_sorted_microbatches,
    )
    group_index = int(np.argmax(grouping.group_physical_positions))
    start = int(grouping.row_cu_seqlens[group_index])
    stop = int(grouping.row_cu_seqlens[group_index + 1])
    selected = grouping.ordered_rows[start:stop]
    cpu_batch = train_chunks.training_batch(selected)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(run.seed + 65_537)
    cpu_plan = prepare_exhaustive_blt_corruption(
        cpu_batch, run.corruption, model_config.vocab, generator
    )
    batch = cpu_batch.to(device)
    plan = cpu_plan.to(device)
    if batch.document_ids is None:
        raise AssertionError("mask-build benchmark omitted document ids")
    layout = RaggedBltLayout(
        clean_valid=batch.valid,
        clean_positions=batch.positions,
        clean_segment_ids=batch.document_ids,
        block_valid=plan.block_valid,
        block_rows=plan.block_rows,
        block_starts=plan.block_starts,
        row_cu_offsets=plan.row_cu_seqlens,
    )

    for _ in range(warmup):
        block_mask = build_ragged_blt_block_mask(layout)
        torch.cuda.synchronize(device)
        del block_mask
    torch.cuda.reset_peak_memory_stats(device)
    timings = array("d")
    started_all = time.perf_counter()
    for _ in range(measured):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        block_mask = build_ragged_blt_block_mask(layout)
        torch.cuda.synchronize(device)
        timings.append(1_000.0 * (time.perf_counter() - started))
        del block_mask
    elapsed = time.perf_counter() - started_all
    clean_positions = batch.ids.numel()
    branch_blocks = plan.block_rows.numel()
    physical_positions = clean_positions + plan.noisy_blocks.numel()
    if physical_positions != int(grouping.group_physical_positions[group_index]):
        raise AssertionError("standalone mask topology differs from scheduled work")
    return {
        "kind": "ragged_block_mask_build",
        "status": "ok",
        "compiler_disabled": True,
        "warmup_repetitions": warmup,
        "measured_repetitions": measured,
        "elapsed_seconds": elapsed,
        "build_ms": _summary(timings),
        "topology": {
            "source_update_row_indices_sha256": hashlib.sha256(
                np.asarray(update_indices, dtype="<i8").tobytes()
            ).hexdigest(),
            "source_update_row_origin_counts_sha256": hashlib.sha256(
                counts.tobytes()
            ).hexdigest(),
            "group_index": group_index,
            "group_row_indices_sha256": hashlib.sha256(
                np.asarray(selected, dtype="<i8").tobytes()
            ).hexdigest(),
            "rows": int(batch.ids.shape[0]),
            "block_length": FAST_BLT_BLOCK_LENGTH,
            "branch_blocks": int(branch_blocks),
            "branch_atoms": int(plan.block_valid.sum()),
            "clean_positions": int(clean_positions),
            "physical_positions": int(physical_positions),
        },
        **_memory_snapshot(device, required=required),
    }


def _phase_report(
    *, kind: str, warmup_records: list[dict[str, object]],
    measured_records: list[dict[str, object]], elapsed: float,
    graph_warmup: Mapping[str, int], telemetry: _Telemetry,
    device: torch.device, required: int, metrics: object | None = None,
    graph_quiescence: Mapping[str, object] | None = None,
    diagnostic_workload: Mapping[str, object] | None = None,
    diagnostic_graph_quiescence: Mapping[str, object] | None = None,
    run: TrainingRunConfig | None = None,
) -> dict[str, object]:
    rows = (
        FAST_BLT_GLOBAL_BATCH
        if kind == "training"
        else FAST_BLT_DIFFUSION_VALIDATION_ROWS
    )
    report: dict[str, object] = {
        "kind": kind, "status": "ok", "eligible": True,
        "warmup_repetitions": len(warmup_records),
        "measured_repetitions": len(measured_records),
        "elapsed_seconds": elapsed,
        "warmup_workload": _workload(warmup_records, rows=rows, kind=kind),
        "measured_workload": _workload(measured_records, rows=rows, kind=kind),
        **graph_warmup, **_graph_snapshot("measured"),
        **_memory_snapshot(device, required=required), **telemetry.report(required=required),
    }
    if kind == "training":
        if (
            graph_quiescence is None
            or diagnostic_workload is None
            or diagnostic_graph_quiescence is None
            or run is None
        ):
            raise ValueError(
                "training timing requires ordinary and diagnostic graph evidence"
            )
        host_enqueue_timings = array(
            "d", (float(record["host_enqueue_ms"]) for record in measured_records)
        )
        cuda_timings = array(
            "d", (float(record["cuda_event_ms"]) for record in measured_records)
        )
        host_enqueue_summary = _summary(host_enqueue_timings)
        cuda_summary = _summary(cuda_timings)
        diagnostic_records = diagnostic_workload.get("updates")
        if not isinstance(diagnostic_records, list) or not diagnostic_records:
            raise ValueError("diagnostic graph workload omitted its plateau update")
        diagnostic_update = diagnostic_records[-1]
        if not isinstance(diagnostic_update, Mapping):
            raise ValueError("diagnostic plateau update is not an object")
        diagnostic_update = dict(diagnostic_update)
        execution_steps = tuple(
            int(record["actual_step"])
            for record in (
                *diagnostic_records,
                *warmup_records,
                *measured_records,
            )
        )
        if any(
            right != left + 1
            for left, right in zip(
                execution_steps[:-1], execution_steps[1:], strict=True
            )
        ):
            raise AssertionError("training execution ledgers are not contiguous")
        if execution_steps[0] != 1 or execution_steps[-1] != len(execution_steps):
            raise AssertionError(
                "fresh candidate execution ledgers must span steps 1..N"
            )
        ordinary_update_ms = 1_000.0 * elapsed / len(measured_records)
        diagnostic_update_ms = float(
            diagnostic_update["host_synchronized_call_ms"]
        )
        materialized_updates = _materialized_update_count(run)
        ordinary_updates = run.iterations - materialized_updates
        schedule_update_ms = (
            ordinary_updates * ordinary_update_ms
            + materialized_updates * diagnostic_update_ms
        ) / run.iterations
        clean_bytes_per_update = FAST_BLT_GLOBAL_BATCH * FAST_BLT_ROW_LENGTH
        report.update(
            graph_quiescence=dict(graph_quiescence),
            diagnostic_workload=dict(diagnostic_workload),
            diagnostic_graph_quiescence=dict(diagnostic_graph_quiescence),
            diagnostic_update=diagnostic_update,
            diagnostic_update_sha256=canonical_json_sha256(diagnostic_update),
            diagnostic_update_ms=diagnostic_update_ms,
            execution_step_range={
                "first_completed_step": execution_steps[0],
                "last_completed_step": execution_steps[-1],
                "diagnostic_warmup_updates": len(diagnostic_records),
                "ordinary_warmup_updates": len(warmup_records),
                "ordinary_measured_updates": len(measured_records),
                "total_updates": len(execution_steps),
                "contiguous": True,
            },
            host_enqueue_ms=host_enqueue_summary,
            cuda_event_update_ms=cuda_summary,
            cuda_event_elapsed_seconds=math.fsum(cuda_timings) / 1_000.0,
            phase_wall_update_ms=ordinary_update_ms,
            update_ms=ordinary_update_ms,
            timing_semantics={
                "schema": FAST_BLT_TIMING_SCHEMA,
                "ordinary_host_field": "host_enqueue_ms",
                "ordinary_host_definition": (
                    "host duration of asynchronous run_update(False) submission"
                ),
                "diagnostic_host_field": "host_synchronized_call_ms",
                "diagnostic_host_definition": (
                    "host duration of run_update(True), including its required "
                    "CUDA synchronization"
                ),
                "cuda_event_field": "cuda_event_ms",
                "canonical_ordinary_update_ms": "phase_wall_update_ms",
                "timed_updates_are_sequential": True,
                "coherence_relative_tolerance": TIMING_COHERENCE_REL_TOL,
                "coherence_absolute_ms": TIMING_COHERENCE_ABS_MS,
            },
            production_schedule={
                "iterations": run.iterations,
                "train_log_every": run.train_log_every,
                "validation_every": run.val_loss_every,
                "ordinary_updates": ordinary_updates,
                "materialized_updates": materialized_updates,
                "ordinary_update_ms": ordinary_update_ms,
                "materialized_update_ms": diagnostic_update_ms,
            },
            production_schedule_update_ms=schedule_update_ms,
            ordinary_clean_bytes_per_second=(
                1_000.0 * clean_bytes_per_update / ordinary_update_ms
            ),
            production_schedule_clean_bytes_per_second=(
                1_000.0 * clean_bytes_per_update / schedule_update_ms
            ),
        )
    else:
        measured_cache_misses = sum(
            int(record["cache_misses"]) for record in measured_records
        )
        measured_first_batch_waits = sum(
            int(record["first_batch_waits"]) for record in measured_records
        )
        cache_records = tuple(
            {
                "schedule_cache_misses": record["schedule_cache_misses"],
                "corruption_plan_cache_misses": record[
                    "corruption_plan_cache_misses"
                ],
                "ragged_mask_metadata_cache_misses": record[
                    "ragged_mask_metadata_cache_misses"
                ],
                "first_batch_cache_misses": record[
                    "first_batch_cache_misses"
                ],
                "first_batch_waits": record["first_batch_waits"],
            }
            for record in measured_records
        )
        cache_quiescence = {
            "measured_cache_misses": measured_cache_misses,
            "measured_repetitions": len(measured_records),
            "quiescent": not (
                measured_cache_misses or measured_first_batch_waits
            ),
            "measured_first_batch_waits": measured_first_batch_waits,
            "records_sha256": canonical_json_sha256(cache_records),
        }
        if measured_cache_misses or measured_first_batch_waits:
            raise ValidationCacheNonquiescenceError(
                evidence=cache_quiescence,
                workload=_workload(measured_records, rows=rows, kind=kind),
            )
        report.update(
            total_validation_rows=FAST_BLT_VALIDATION_ROWS,
            diffusion_validation_rows=FAST_BLT_DIFFUSION_VALIDATION_ROWS,
            ar_max_batch_size=FAST_BLT_AR_VALIDATION_BATCH,
            diffusion_max_batch_size=FAST_BLT_VALIDATION_BATCH,
            batching_policy="token_budgeted_ragged_diffusion",
            end_to_end_total_rows_per_second=(
                FAST_BLT_VALIDATION_ROWS * len(measured_records) / elapsed
            ),
            end_to_end_diffusion_rows_per_second=(
                FAST_BLT_DIFFUSION_VALIDATION_ROWS
                * len(measured_records)
                / elapsed
            ),
            metrics=asdict(metrics),
            cache_quiescence=cache_quiescence,
        )
    return report


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def _is_torch_compile_error(error: BaseException) -> bool:
    """Recognize compiler exceptions by type, never by mutable message text."""

    return _qualified_exception_type(error) in FAST_BLT_COMPILE_ERROR_TYPES and isinstance(
        error,
        (
            TorchDynamoException,
            FailOnRecompileLimitHit,
            RecompileError,
            InvalidBackend,
            CppCompileError,
            CppWrapperCodegenError,
            OperatorIssue,
            SubgraphLoweringException,
            InvalidCxxCompiler,
        ),
    )


def _qualified_exception_type(error: BaseException) -> str:
    kind = type(error)
    return f"{kind.__module__}.{kind.__qualname__}"


def _benchmark_candidate(
    *, microbatch: int, production_run: TrainingRunConfig, model_config, manifest,
    train_chunks, validation_chunks, dataset_provenance: Mapping[str, object],
    source_provenance: Mapping[str, object], device: torch.device,
    warmup_updates: int, measured_updates: int, validation_warmup: int,
    validation_measured: int, required: int,
) -> dict[str, object]:
    _reset_candidate_state(device)
    _seed_everything(1_337)
    phase = "construct"
    trainer: ByteDiffusionTrainer | None = None
    parameter_count: int | None = None
    run = build_benchmark_run_config(production_run, microbatch)
    execution_contract = run_contract_without_identity(run)
    try:
        model = ByteDiffusionModel(model_config)
        parameter_count = model.parameter_count()
        if parameter_count != model_config.production_parameter_target:
            raise ValueError("complete Fast-BLT model parameter count drifted")
        trainer = ByteDiffusionTrainer(
            model,
            DeterministicChunkCursor(train_chunks, seed=run.seed, shuffle=True),
            validation_chunks, run, device=device, distributed=DistributedContext(),
            atomic_manifest=manifest, dataset_provenance=dataset_provenance,
            source_provenance=source_provenance,
        )

        ledger = DeterministicChunkCursor(train_chunks, seed=run.seed, shuffle=True)

        def next_training_topology() -> dict[str, object]:
            return _topology_evidence(
                indices=ledger.next_indices(FAST_BLT_GLOBAL_BATCH),
                chunks=train_chunks,
                run=run,
                validation=False,
            )

        def quiesce_training_graphs(
            *, materialize_metrics: bool,
        ) -> tuple[
            list[dict[str, object]], dict[str, object], dict[str, int]
        ]:
            _reset_compiler_counters()
            pending: list[
                tuple[
                    dict[str, object],
                    StepMetrics | None,
                    UpdateExecutionLedger | None,
                    float,
                    torch.cuda.Event,
                    torch.cuda.Event,
                ]
            ] = []
            observations: list[dict[str, object]] = []
            quiescent = False
            for warmup_index in range(FAST_BLT_MAX_GRAPH_WARMUP_UPDATES):
                topology = next_training_topology()
                before = _graph_counts()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                host_started = time.perf_counter()
                start_event.record()
                metrics = trainer.run_update(
                    materialize_metrics=materialize_metrics
                )
                execution_ledger = trainer.last_execution_ledger
                end_event.record()
                host_duration_ms = 1_000.0 * (
                    time.perf_counter() - host_started
                )
                if (metrics is not None) is not materialize_metrics:
                    raise AssertionError("training metrics mode returned wrong result")
                delta = _graph_delta(before, _graph_counts())
                representative = (
                    int(topology["planned_min_microbatch"]) > 1
                    and int(topology["planned_max_microbatch"]) > 1
                )
                observations.append(
                    {
                        "update": warmup_index + 1,
                        "materialize_metrics": materialize_metrics,
                        "representative_non_singleton": representative,
                        "planned_min_microbatch": int(
                            topology["planned_min_microbatch"]
                        ),
                        "planned_max_microbatch": int(
                            topology["planned_max_microbatch"]
                        ),
                        "new_unique_graphs": delta["unique_graphs"],
                        "new_recompiles": delta["recompiles"],
                        "new_graph_breaks": delta["graph_breaks"],
                    }
                )
                pending.append(
                    (
                        topology,
                        metrics,
                        execution_ledger,
                        host_duration_ms,
                        start_event,
                        end_event,
                    )
                )
                quiescent = bool(
                    warmup_index + 1 >= warmup_updates
                    and representative
                    and all(value == 0 for value in delta.values())
                )
                if quiescent:
                    break
            torch.cuda.synchronize(device)
            records: list[dict[str, object]] = []
            for (
                topology,
                metrics,
                execution_ledger,
                host_duration_ms,
                start_event,
                end_event,
            ) in pending:
                actual_record = _attach_execution_ledger(
                    topology, execution_ledger
                )
                record = (
                    _attach_step_metrics(actual_record, metrics)
                    if metrics is not None
                    else actual_record
                )
                record["materialize_metrics"] = materialize_metrics
                records.append(
                    _attach_update_timing(
                        record,
                        host_duration_ms=host_duration_ms,
                        cuda_event_ms=start_event.elapsed_time(end_event),
                        synchronized_host_call=materialize_metrics,
                    )
                )
            counts = _graph_counts()
            evidence = {
                "schema": FAST_BLT_GRAPH_QUIESCENCE_SCHEMA,
                "achieved": quiescent,
                "materialize_metrics": materialize_metrics,
                "minimum_updates": warmup_updates,
                "maximum_updates": FAST_BLT_MAX_GRAPH_WARMUP_UPDATES,
                "observed_updates": len(observations),
                "representative_non_singleton": any(
                    bool(observation["representative_non_singleton"])
                    for observation in observations
                ),
                "total_unique_graphs": counts["unique_graphs"],
                "total_recompiles": counts["recompiles"],
                "total_graph_breaks": counts["graph_breaks"],
                "observations": observations,
                "observations_sha256": canonical_json_sha256(observations),
            }
            if not quiescent:
                mode = "diagnostic" if materialize_metrics else "ordinary"
                workload = _workload(
                    records,
                    rows=FAST_BLT_GLOBAL_BATCH,
                    kind=(
                        "training_diagnostic"
                        if materialize_metrics
                        else "training"
                    ),
                )
                raise GraphNonquiescenceError(
                    f"Fast-BLT {mode} training graphs did not quiesce on a "
                    "representative B>1 update within "
                    f"{FAST_BLT_MAX_GRAPH_WARMUP_UPDATES} warmups",
                    phase=(
                        "training_diagnostic_warmup"
                        if materialize_metrics
                        else "training_warmup"
                    ),
                    evidence=evidence,
                    workload=workload,
                )
            return records, evidence, _graph_snapshot("warmup")

        phase = "training_diagnostic_warmup"
        diagnostic_records, diagnostic_graph_quiescence, _ = (
            quiesce_training_graphs(materialize_metrics=True)
        )
        diagnostic_workload = _workload(
            diagnostic_records, rows=FAST_BLT_GLOBAL_BATCH,
            kind="training_diagnostic",
        )

        phase = "training_warmup"
        warmup_records, graph_quiescence, warmup_graphs = (
            quiesce_training_graphs(materialize_metrics=False)
        )

        phase = "training_measured"
        _reset_compiler_counters()
        torch.cuda.reset_peak_memory_stats(device)
        measured_pending: list[
            tuple[
                dict[str, object], float, torch.cuda.Event, torch.cuda.Event
            ]
        ] = []
        with _Telemetry(device) as telemetry:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            for _ in range(measured_updates):
                topology = next_training_topology()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                host_started = time.perf_counter()
                start_event.record()
                metrics = trainer.run_update(materialize_metrics=False)
                end_event.record()
                host_enqueue_ms = 1_000.0 * (
                    time.perf_counter() - host_started
                )
                if metrics is not None:
                    raise AssertionError("ordinary training unexpectedly returned metrics")
                record = _attach_execution_ledger(
                    topology, trainer.last_execution_ledger
                )
                record["materialize_metrics"] = False
                measured_pending.append(
                    (record, host_enqueue_ms, start_event, end_event)
                )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
        measured_records = [
            _attach_update_timing(
                record,
                host_duration_ms=host_enqueue_ms,
                cuda_event_ms=start_event.elapsed_time(end_event),
                synchronized_host_call=False,
            )
            for record, host_enqueue_ms, start_event, end_event in measured_pending
        ]
        training = _phase_report(
            kind="training", warmup_records=warmup_records,
            measured_records=measured_records, elapsed=elapsed,
            graph_warmup=warmup_graphs, telemetry=telemetry, device=device,
            required=required, graph_quiescence=graph_quiescence,
            diagnostic_workload=diagnostic_workload,
            diagnostic_graph_quiescence=diagnostic_graph_quiescence,
            run=run,
        )

        validation_indices = _diffusion_validation_indices()
        validation_topology = _topology_evidence(
            indices=validation_indices, chunks=validation_chunks, run=run, validation=True,
        )
        with _Telemetry(device) as validation_telemetry:
            phase = "validation_warmup"
            _reset_compiler_counters()
            validation_warmup_records: list[dict[str, object]] = []
            for _ in range(validation_warmup):
                validation_metrics = trainer.validate()
                validation_warmup_records.append(
                    _attach_validation_metrics(
                        validation_topology,
                        validation_metrics,
                        trainer.last_validation_execution_ledger,
                    )
                )
            torch.cuda.synchronize(device)
            validation_warmup_graphs = _graph_snapshot("warmup")

            phase = "validation_measured"
            _reset_compiler_counters()
            torch.cuda.reset_peak_memory_stats(device)
            validation_records: list[dict[str, object]] = []
            metrics = None
            torch.cuda.synchronize(device)
            validation_telemetry.mark_measurement_start()
            validation_started = time.perf_counter()
            for _ in range(validation_measured):
                metrics = trainer.validate()
                validation_records.append(
                    _attach_validation_metrics(
                        validation_topology,
                        metrics,
                        trainer.last_validation_execution_ledger,
                    )
                )
            torch.cuda.synchronize(device)
            validation_elapsed = time.perf_counter() - validation_started
            validation_telemetry.mark_measurement_end()
        if metrics is None:
            raise AssertionError("validation measured phase produced no metrics")
        validation = _phase_report(
            kind="validation", warmup_records=validation_warmup_records,
            measured_records=validation_records, elapsed=validation_elapsed,
            graph_warmup=validation_warmup_graphs, telemetry=validation_telemetry,
            device=device, required=required, metrics=metrics,
        )
        phase = "block_mask_build"
        block_mask_build = _benchmark_block_mask_build(
            train_chunks=train_chunks,
            run=run,
            model_config=model_config,
            device=device,
            required=required,
        )
        candidate: dict[str, object] = {
            "status": "ok", "eligible": True, "microbatch": microbatch,
            "run_contract": execution_contract,
            "run_contract_sha256": canonical_json_sha256(execution_contract),
            "parameter_count": parameter_count, "training": training,
            "validation": validation, "block_mask_build": block_mask_build,
        }
        candidate["eligible"] = candidate_passes(
            candidate, microbatch=microbatch, expected_run_contract=execution_contract,
            total_memory=torch.cuda.get_device_properties(device).total_memory,
            expected_parameter_count=model_config.production_parameter_target,
        )
        return candidate
    except GraphNonquiescenceError as error:
        return {
            "status": "graph_nonquiescent",
            "eligible": False,
            "graph_phase": error.phase,
            "error": str(error),
            "microbatch": microbatch,
            "run_contract": execution_contract,
            "run_contract_sha256": canonical_json_sha256(execution_contract),
            "parameter_count": parameter_count,
            "graph_quiescence": error.evidence,
            "warmup_workload": error.workload,
            **_memory_snapshot(device, required=required),
        }
    except ValidationCacheNonquiescenceError as error:
        return {
            "status": "validation_cache_nonquiescent",
            "eligible": False,
            "cache_phase": phase,
            "error": str(error),
            "microbatch": microbatch,
            "run_contract": execution_contract,
            "run_contract_sha256": canonical_json_sha256(execution_contract),
            "parameter_count": parameter_count,
            "cache_quiescence": error.evidence,
            "measured_workload": error.workload,
            **_memory_snapshot(device, required=required),
        }
    except TelemetrySamplerLifecycleError as error:
        return {
            "status": "telemetry_error",
            "eligible": False,
            "telemetry_phase": phase,
            "error": str(error),
            "microbatch": microbatch,
            "run_contract": execution_contract,
            "run_contract_sha256": canonical_json_sha256(execution_contract),
            "parameter_count": parameter_count,
            "telemetry_terminal": error.evidence,
            **_memory_snapshot(device, required=required),
        }
    except BaseException as error:
        if not _is_cuda_oom(error) and not _is_torch_compile_error(error):
            raise
        if _is_torch_compile_error(error) and not _is_cuda_oom(error):
            error_text = str(error)
            error_repr = repr(error)
            traceback_text = traceback.format_exc()
            return {
                "status": "compile_error",
                "eligible": False,
                "compile_phase": phase,
                "error_type": _qualified_exception_type(error),
                "error": error_text,
                "error_repr": error_repr,
                "error_sha256": hashlib.sha256(
                    error_text.encode("utf-8")
                ).hexdigest(),
                "error_repr_sha256": hashlib.sha256(
                    error_repr.encode("utf-8")
                ).hexdigest(),
                "traceback": traceback_text,
                "traceback_sha256": hashlib.sha256(
                    traceback_text.encode("utf-8")
                ).hexdigest(),
                "microbatch": microbatch,
                "run_contract": execution_contract,
                "run_contract_sha256": canonical_json_sha256(
                    execution_contract
                ),
                "parameter_count": parameter_count,
                **_memory_snapshot(device, required=required),
            }
        free, total = torch.cuda.mem_get_info(device)
        return {
            "status": "oom", "eligible": False, "oom_phase": phase,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "microbatch": microbatch,
            "run_contract": execution_contract,
            "run_contract_sha256": canonical_json_sha256(execution_contract),
            **_memory_snapshot(device, required=required),
            "oom_physical_free_bytes": free, "oom_physical_total_bytes": total,
        }
    finally:
        if trainer is not None:
            trainer.close_validation_prefetch()
        del trainer
        _reset_candidate_state(device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--candidate-microbatches",
        type=parse_candidate_microbatches,
        required=True,
    )
    parser.add_argument("--warmup-updates", type=int, default=MIN_WARMUP_UPDATES)
    parser.add_argument("--measured-updates", type=int, default=MIN_MEASURED_UPDATES)
    parser.add_argument("--validation-warmup", type=int, default=MIN_VALIDATION_WARMUP)
    parser.add_argument("--validation-measured", type=int, default=MIN_VALIDATION_MEASURED)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-benchmark-harness-sha256", required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-atomic-manifest-sha256", required=True)
    parser.add_argument("--expected-train-dataset-sha256", required=True)
    parser.add_argument("--expected-validation-dataset-sha256", required=True)
    parser.add_argument("--expected-patcher-sha256", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--expected-production-run-sha256", required=True)
    parser.add_argument("--expected-common-run-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    harness_snapshot = benchmark_harness_provenance()
    if harness_snapshot.get("sha256") != args.expected_benchmark_harness_sha256:
        raise ValueError("benchmark harness differs from its pinned pre-run identity")
    # Allocator configuration is parsed at process/CUDA initialization time;
    # readiness must authenticate it rather than mutate it after importing
    # torch and accidentally certify a different allocator behavior.
    require_fast_blt_allocator_environment()
    if not torch.cuda.is_available():
        raise RuntimeError("Fast-BLT readiness must run through mlq on CUDA")
    repetitions = (
        args.warmup_updates, args.measured_updates,
        args.validation_warmup, args.validation_measured,
    )
    if repetitions[0] < 2 or repetitions[1] < 4 or repetitions[2] < 1 or repetitions[3] < 2:
        raise ValueError("readiness requires 2/4 training and 1/2 validation repetitions")
    if args.warmup_updates > FAST_BLT_MAX_GRAPH_WARMUP_UPDATES:
        raise ValueError(
            "training warmups exceed the authenticated graph-quiescence window"
        )
    if args.candidate_microbatches != (32,):
        raise ValueError("complete-preset readiness requires --candidate-microbatches 32")

    geometry = FastBltGeometry()
    source = training_source_provenance()
    if source.get("sha256") != args.expected_source_sha256:
        raise ValueError("training source differs from pinned Fast-BLT readiness")
    production_run = TrainingRunConfig.from_env()
    if production_run.preset not in FAST_BLT_ENTROPY_B4_COMPLETE_PRESETS:
        raise ValueError(
            "BYTE_DIFFUSION_PRESET must select a supported complete Fast-BLT cell"
        )
    production_contract = run_contract_without_identity(production_run)
    if canonical_json_sha256(production_contract) != args.expected_production_run_sha256:
        raise ValueError("production run config differs from pinned readiness")

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    model_config = model_config_from_env()
    if model_config != fast_blt_complete_model_config(production_run.preset):
        raise ValueError("model differs from the named complete Fast-BLT cell")
    model_contract = model_config.to_dict()
    if canonical_json_sha256(model_contract) != args.expected_model_sha256:
        raise ValueError("model config differs from pinned Fast-BLT readiness")

    candidate_runs = {
        value: build_benchmark_run_config(production_run, value)
        for value in args.candidate_microbatches
    }
    common_contracts = {
        canonical_json_sha256(common_run_contract(run)) for run in candidate_runs.values()
    }
    if common_contracts != {args.expected_common_run_sha256}:
        raise ValueError("candidate run config differs from pinned readiness")
    common_contract = common_run_contract(next(iter(candidate_runs.values())))

    manifest, train_chunks, validation_chunks = load_data_directory(
        args.data_path, chunk_size=FAST_BLT_ROW_LENGTH, recipe="blt_d",
        required_branch_bytes=FAST_BLT_ROW_LENGTH,
        branch_span_length=FAST_BLT_BLOCK_LENGTH,
        validation_chunk_limit=FAST_BLT_VALIDATION_ROWS,
        require_challenge_validation=True,
        expected_payload_sha256=args.expected_data_sha256,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        expected_patching_policy="causal_entropy_v1",
    )
    if len(validation_chunks) != FAST_BLT_VALIDATION_ROWS:
        raise ValueError("Fast-BLT readiness requires exactly 2,048 validation rows")
    train_patching = getattr(train_chunks, "patching", None)
    validation_patching = getattr(validation_chunks, "patching", None)
    if (
        train_patching is None or train_patching != validation_patching
        or train_patching.name != "causal_entropy_v1"
        or train_patching.artifact_sha256 != args.expected_patcher_sha256
    ):
        raise ValueError("dataset patcher differs from pinned Fast-BLT readiness")
    dataset_manifest = json.loads((args.data_path / "manifest.json").read_text())
    train_sha = str(getattr(train_chunks, "dataset_sha256", ""))
    validation_sha = str(getattr(validation_chunks, "dataset_sha256", ""))
    if (
        dataset_manifest.get("payload_sha256") != args.expected_data_sha256
        or manifest.sha256 != args.expected_atomic_manifest_sha256
        or train_sha != args.expected_train_dataset_sha256
        or validation_sha != args.expected_validation_dataset_sha256
    ):
        raise ValueError("dataset identity differs from pinned Fast-BLT readiness")

    total_memory = torch.cuda.get_device_properties(device).total_memory
    required = required_headroom_bytes(total_memory)
    dataset_provenance = {
        "payload_sha256": args.expected_data_sha256,
        "source_manifests": dataset_manifest.get("source_manifests", []),
    }
    results: dict[str, dict[str, object]] = {}
    for candidate, candidate_run in candidate_runs.items():
        result = _benchmark_candidate(
            microbatch=candidate, production_run=production_run,
            model_config=model_config, manifest=manifest, train_chunks=train_chunks,
            validation_chunks=validation_chunks, dataset_provenance=dataset_provenance,
            source_provenance=source, device=device,
            warmup_updates=args.warmup_updates, measured_updates=args.measured_updates,
            validation_warmup=args.validation_warmup,
            validation_measured=args.validation_measured, required=required,
        )
        if result.get("run_contract") != run_contract_without_identity(candidate_run):
            raise AssertionError("candidate execution contract drifted during benchmark")
        results[str(candidate)] = result
        print("fast_blt_readiness_candidate " + json.dumps(result, sort_keys=True), flush=True)

    eligible = [
        (
            candidate,
            float(
                results[str(candidate)]["training"][
                    "production_schedule_update_ms"
                ]
            ),
        )
        for candidate in args.candidate_microbatches
        if results[str(candidate)].get("eligible") is True
    ]
    selected = min(eligible, key=lambda item: (item[1], -item[0]), default=(None, 0.0))[0]
    source_records = dataset_manifest.get("source_manifests")
    if not isinstance(source_records, list) or len(source_records) != 1:
        raise ValueError("readiness requires one authenticated source manifest")
    report = {
        "schema": FAST_BLT_READINESS_SCHEMA,
        "preset": production_run.preset,
        "source_sha256": source["sha256"],
        "benchmark_harness_source": harness_snapshot,
        "dataset": {
            "payload_sha256": args.expected_data_sha256,
            "source_manifest_sha256": source_records[0]["sha256"],
            "atomic_manifest_sha256": manifest.sha256,
            "train_dataset_sha256": train_sha,
            "validation_dataset_sha256": validation_sha,
            "validation_rows": FAST_BLT_VALIDATION_ROWS,
            "diffusion_validation_rows": FAST_BLT_DIFFUSION_VALIDATION_ROWS,
            "patching_policy": train_patching.name,
            "patcher_sha256": train_patching.artifact_sha256,
            "max_patch_size": train_patching.max_patch_size,
        },
        "model_config": model_contract,
        "model_config_sha256": canonical_json_sha256(model_contract),
        "production_run_contract": production_contract,
        "production_run_contract_sha256": canonical_json_sha256(production_contract),
        "common_run_contract": common_contract,
        "common_run_contract_sha256": canonical_json_sha256(common_contract),
        "geometry": geometry.contract(),
        "candidate_microbatches": list(args.candidate_microbatches),
        "results": results, "selected_microbatch": selected,
        "headroom_policy": {
            "fraction": 0.15, "minimum_gib": 4.0, "required_bytes": required,
            "intrinsic_and_physical_required": True,
        },
        "sustained_gpu_policy": SUSTAINED_GPU_POLICY,
        "gpu": {
            "name": torch.cuda.get_device_name(device), "total_memory_bytes": total_memory,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "nvidia_smi_selector": nvidia_smi_selector(device),
        },
        "runtime": {
            "schema": FAST_BLT_RUNTIME_SCHEMA, "compiled": True,
            "compile_dynamic_shapes": True, "device_type": "cuda", "world_size": 1,
            "validation_model_is_separate": True,
            "allocator": fast_blt_allocator_contract(),
        },
    }
    _require_unchanged_harness(
        harness_snapshot, benchmark_harness_provenance()
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("fast_blt_readiness_result " + json.dumps(report, sort_keys=True), flush=True)
    if selected is None:
        raise RuntimeError("no Fast-BLT candidate passed the complete readiness contract")


if __name__ == "__main__":
    main()
