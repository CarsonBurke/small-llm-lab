"""Authenticated readiness contract for exhaustive Fast-BLT-D4.

The benchmark report is deliberately self-describing and fail closed.  In
particular, 2,048 is not treated as a branch count: entropy patching produces
a variable number of eligible origins per row and the report binds the exact
work executed by every timed update.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

from .config import (
    FAST_BLT_ENTROPY_B4_COMPLETE_PARAMETER_TARGET,
    FAST_BLT_ENTROPY_B4_COMPLETE_PRESET,
    FAST_BLT_ENTROPY_B4_COMPLETE_PRESETS,
    fast_blt_complete_model_config,
)
from .readiness import HEADROOM_FRACTION, HEADROOM_MINIMUM_GIB, SUSTAINED_GPU_POLICY


FAST_BLT_READINESS_SCHEMA = "byte_diffusion_fast_blt_readiness/v8"
FAST_BLT_GEOMETRY_SCHEMA = "byte_diffusion_fast_blt_geometry/v2"
FAST_BLT_RUNTIME_SCHEMA = "byte_diffusion_fast_blt_runtime/v3"
FAST_BLT_WORKLOAD_SCHEMA = "byte_diffusion_fast_blt_ragged_workload/v5"
FAST_BLT_GRAPH_QUIESCENCE_SCHEMA = "byte_diffusion_fast_blt_graph_quiescence/v3"
FAST_BLT_TIMING_SCHEMA = "byte_diffusion_fast_blt_timing/v1"
FAST_BLT_TELEMETRY_LIFECYCLE_SCHEMA = (
    "byte_diffusion_fast_blt_telemetry_lifecycle/v2"
)
FAST_BLT_TELEMETRY_TERMINAL_SCHEMA = (
    "byte_diffusion_fast_blt_telemetry_terminal/v1"
)
FAST_BLT_TELEMETRY_QUERY_INTERVAL_SECONDS = 0.1
# Keep process liveness separate from measured-window coverage.  nvidia-smi can
# occasionally take longer than one driver sampling epoch while CUDA is busy;
# the stricter coverage bound below still rejects any such gap that overlaps
# the scored interval.
FAST_BLT_TELEMETRY_QUERY_TIMEOUT_SECONDS = 2.0
FAST_BLT_TELEMETRY_THREAD_JOIN_TIMEOUT_SECONDS = 3.0
FAST_BLT_TELEMETRY_MAX_COVERAGE_GAP_SECONDS = 0.5
FAST_BLT_MAX_GRAPH_WARMUP_UPDATES = 8
FAST_BLT_ALLOCATOR_SCHEMA = "byte_diffusion_fast_blt_allocator/v1"
FAST_BLT_ALLOCATOR_ENV = "PYTORCH_ALLOC_CONF"
FAST_BLT_ALLOCATOR_VALUE = "expandable_segments:True"
FAST_BLT_LEGACY_ALLOCATOR_ENV = "PYTORCH_CUDA_ALLOC_CONF"
FAST_BLT_ROW_LENGTH = 8_192
FAST_BLT_GLOBAL_BATCH = 249
FAST_BLT_VALIDATION_ROWS = 2_048
FAST_BLT_DIFFUSION_VALIDATION_ROWS = 256
FAST_BLT_VALIDATION_BATCH = 64
FAST_BLT_AR_VALIDATION_BATCH = 128
FAST_BLT_BLOCK_LENGTH = 4
FAST_BLT_ORIGIN_POLICY = "all_entropy_patch_starts"
FAST_BLT_FAILURE_PHASES = frozenset(
    {
        "construct",
        "training_diagnostic_warmup",
        "training_warmup",
        "training_measured",
        "validation_warmup",
        "validation_measured",
        "block_mask_build",
    }
)
# Exact public/private PyTorch exception identities which the benchmark is
# allowed to classify as compiler failures.  Keeping this an allowlist (rather
# than accepting arbitrary RuntimeError text containing "compile") prevents a
# model/data failure from being laundered into an authenticated failed run.
FAST_BLT_COMPILE_ERROR_TYPES = frozenset(
    {
        "torch._dynamo.exc.BackendCompilerFailed",
        "torch._dynamo.exc.FailOnRecompileLimitHit",
        "torch._dynamo.exc.InternalTorchDynamoError",
        "torch._dynamo.exc.InvalidBackend",
        "torch._dynamo.exc.RecompileError",
        "torch._dynamo.exc.TorchRuntimeError",
        "torch._dynamo.exc.UncapturedHigherOrderOpError",
        "torch._dynamo.exc.Unsupported",
        "torch._dynamo.exc.UserError",
        "torch._inductor.exc.CUDACompileError",
        "torch._inductor.exc.CppCompileError",
        "torch._inductor.exc.CppWrapperCodegenError",
        "torch._inductor.exc.GPUTooOldForTriton",
        "torch._inductor.exc.InductorError",
        "torch._inductor.exc.InvalidCxxCompiler",
        "torch._inductor.exc.LoweringException",
        "torch._inductor.exc.MissingOperatorWithDecomp",
        "torch._inductor.exc.MissingOperatorWithoutDecomp",
        "torch._inductor.exc.OperatorIssue",
        "torch._inductor.exc.SubgraphLoweringException",
        "torch._inductor.exc.TritonMissing",
    }
)
MIN_TELEMETRY_SAMPLES = 3
TIMING_COHERENCE_REL_TOL = 0.02
TIMING_COHERENCE_ABS_MS = 5.0
GIB = 1 << 30


def _diffusion_validation_indices_sha256() -> str:
    digest = hashlib.sha256()
    for position in range(FAST_BLT_DIFFUSION_VALIDATION_ROWS):
        index = (
            (2 * position + 1)
            * FAST_BLT_VALIDATION_ROWS
            // (2 * FAST_BLT_DIFFUSION_VALIDATION_ROWS)
        )
        digest.update(index.to_bytes(8, "little", signed=True))
    return digest.hexdigest()


FAST_BLT_DIFFUSION_VALIDATION_INDICES_SHA256 = (
    _diffusion_validation_indices_sha256()
)


def fast_blt_allocator_contract() -> dict[str, object]:
    """Return the process-start allocator contract for complete Fast-BLT.

    PyTorch 2.13 names ``PYTORCH_ALLOC_CONF`` as the primary allocator
    setting.  Its CUDA compatibility path nevertheless checks the deprecated
    ``PYTORCH_CUDA_ALLOC_CONF`` alias first, so even a valid canonical value
    can be shadowed by a stale alias.  Requiring the alias to be absent makes
    the effective setting unambiguous and reproducible.
    """

    return {
        "schema": FAST_BLT_ALLOCATOR_SCHEMA,
        "environment_variable": FAST_BLT_ALLOCATOR_ENV,
        "value": FAST_BLT_ALLOCATOR_VALUE,
        "legacy_alias": FAST_BLT_LEGACY_ALLOCATOR_ENV,
        "legacy_alias_must_be_unset": True,
    }


def require_fast_blt_allocator_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Fail closed unless this process started with the pinned allocator."""

    values = os.environ if environ is None else environ
    observed = values.get(FAST_BLT_ALLOCATOR_ENV)
    if observed != FAST_BLT_ALLOCATOR_VALUE:
        raise ValueError(
            "complete Fast-BLT requires exact process environment "
            f"{FAST_BLT_ALLOCATOR_ENV}={FAST_BLT_ALLOCATOR_VALUE!r}; "
            f"observed {observed!r}"
        )
    legacy = values.get(FAST_BLT_LEGACY_ALLOCATOR_ENV)
    if legacy is not None:
        raise ValueError(
            f"{FAST_BLT_LEGACY_ALLOCATOR_ENV} must be unset because PyTorch "
            f"CUDA may let it shadow {FAST_BLT_ALLOCATOR_ENV}; observed "
            f"{legacy!r}"
        )
    return fast_blt_allocator_contract()


@dataclass(frozen=True)
class FastBltGeometry:
    """The sole complete-cell geometry: variable exhaustive D4 origins."""

    block_length: int = FAST_BLT_BLOCK_LENGTH
    origin_policy: str = FAST_BLT_ORIGIN_POLICY

    def __post_init__(self) -> None:
        if self.block_length != FAST_BLT_BLOCK_LENGTH:
            raise ValueError("complete Fast-BLT readiness is pinned to D4")
        if self.origin_policy != FAST_BLT_ORIGIN_POLICY:
            raise ValueError("complete Fast-BLT readiness requires every entropy origin")

    def contract(self) -> dict[str, object]:
        return {
            "schema": FAST_BLT_GEOMETRY_SCHEMA,
            "block_length": FAST_BLT_BLOCK_LENGTH,
            "origin_policy": FAST_BLT_ORIGIN_POLICY,
            "origin_count": "variable_per_row",
            "exhaustive": True,
            "virtual_bos_is_origin": False,
            "first_physical_patch_condition": "virtual_bos",
            "overflow_policy": "pad_at_finite_training_sequence_end",
            "row_length_bytes": FAST_BLT_ROW_LENGTH,
            "global_batch": FAST_BLT_GLOBAL_BATCH,
            "validation_rows": FAST_BLT_VALIDATION_ROWS,
            "diffusion_validation_rows": FAST_BLT_DIFFUSION_VALIDATION_ROWS,
            "ar_validation_max_batch_size": FAST_BLT_AR_VALIDATION_BATCH,
            "diffusion_validation_max_batch_size": FAST_BLT_VALIDATION_BATCH,
            "ar_validation_batching": "fixed_max_batch",
            "diffusion_validation_batching": "token_budgeted_ragged",
        }


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        _json_native(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_native(value: object) -> object:
    if is_dataclass(value):
        value = asdict(value)
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def run_contract_without_identity(run: object) -> dict[str, object]:
    """Retain every execution field and remove only the human run label."""

    native = _json_native(run)
    if not isinstance(native, dict):
        raise TypeError("Fast-BLT run contract must be an object or dataclass")
    result = dict(native)
    result.pop("run_id", None)
    return result


def common_run_contract(run: object) -> dict[str, object]:
    """Normalize only the two candidate-dependent batching fields."""

    result = run_contract_without_identity(run)
    result.pop("microbatch_per_rank", None)
    result.pop("gradient_accumulation", None)
    return result


def required_headroom_bytes(total_bytes: int) -> int:
    if total_bytes <= 0:
        raise ValueError("GPU total memory must be positive")
    return max(
        math.ceil(total_bytes * HEADROOM_FRACTION),
        math.ceil(HEADROOM_MINIMUM_GIB * GIB),
    )


def _read_report_once(path: Path) -> tuple[dict[str, object], str]:
    payload = path.read_bytes()
    report = json.loads(payload)
    if not isinstance(report, dict):
        raise ValueError("Fast-BLT readiness report must contain one object")
    return report, hashlib.sha256(payload).hexdigest()


def _finite_positive(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def _summary_values(summary: object) -> tuple[float, ...] | None:
    """Authenticate raw samples and every derived summary statistic."""

    if not isinstance(summary, Mapping):
        return None
    samples = summary.get("samples")
    if not isinstance(samples, (list, tuple)):
        return None
    try:
        values = tuple(float(value) for value in samples)
        if not values or not all(math.isfinite(value) for value in values):
            return None
        ordered = sorted(values)
        rank = 0.1 * (len(ordered) - 1)
        lower = math.floor(rank)
        upper = math.ceil(rank)
        p10 = ordered[lower] + (rank - lower) * (
            ordered[upper] - ordered[lower]
        )
        expected = (
            math.fsum(values) / len(values),
            p10,
            min(values),
            max(values),
        )
        observed = tuple(
            float(summary[key])
            for key in ("mean", "p10", "minimum", "maximum")
        )
        if not (
            int(summary["count"]) == len(values)
            and summary.get("samples_sha256") == canonical_json_sha256(samples)
            and all(math.isfinite(value) for value in observed)
            and observed[2] <= observed[1] <= observed[0] <= observed[3]
            and all(
                math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9)
                for left, right in zip(observed, expected, strict=True)
            )
        ):
            return None
        return values
    except (KeyError, TypeError, ValueError):
        return None


def _summary_passes(summary: object, *, mean: float, p10: float) -> bool:
    values = _summary_values(summary)
    return bool(
        values is not None
        and isinstance(summary, Mapping)
        and len(values) >= MIN_TELEMETRY_SAMPLES
        and math.fsum(values) / len(values) >= mean
        and float(summary["p10"]) >= p10
    )


def _graph_phase_passes(phase: Mapping[str, object]) -> bool:
    try:
        return (
            int(phase["warmup_unique_graphs"]) >= 0
            and int(phase["warmup_recompiles"]) >= 0
            and phase.get("warmup_graph_breaks") == 0
            and phase.get("measured_unique_graphs") == 0
            and phase.get("measured_recompiles") == 0
            and phase.get("measured_graph_breaks") == 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def _timing_summary_passes(
    summary: object,
    records: Sequence[object],
    *,
    field: str,
) -> bool:
    """Authenticate a reported timing summary against its update records."""

    if not records:
        return False
    try:
        values = tuple(float(record[field]) for record in records)  # type: ignore[index]
        return bool(
            all(math.isfinite(value) and value > 0 for value in values)
            and _summary_values(summary) == values
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def _materialized_update_count(
    *, iterations: int, train_log_every: int, validation_every: int
) -> int:
    common = math.lcm(train_log_every, validation_every)
    count = (
        iterations // train_log_every
        + iterations // validation_every
        - iterations // common
    )
    if iterations % train_log_every and iterations % validation_every:
        count += 1
    return count


def _timing_semantics_contract() -> dict[str, object]:
    return {
        "schema": FAST_BLT_TIMING_SCHEMA,
        "ordinary_host_field": "host_enqueue_ms",
        "ordinary_host_definition": (
            "host duration of asynchronous run_update(False) submission"
        ),
        "diagnostic_host_field": "host_synchronized_call_ms",
        "diagnostic_host_definition": (
            "host duration of run_update(True), including its required CUDA "
            "synchronization"
        ),
        "cuda_event_field": "cuda_event_ms",
        "canonical_ordinary_update_ms": "phase_wall_update_ms",
        "timed_updates_are_sequential": True,
        "coherence_relative_tolerance": TIMING_COHERENCE_REL_TOL,
        "coherence_absolute_ms": TIMING_COHERENCE_ABS_MS,
    }


def _graph_quiescence_passes(
    evidence: object,
    *,
    warmup_records: Sequence[object],
    materialize_metrics: bool,
    achieved: bool = True,
) -> bool:
    """Authenticate the representative graph plateau preceding measurement."""

    if not isinstance(evidence, Mapping):
        return False
    observations = evidence.get("observations")
    if not isinstance(observations, list):
        return False
    try:
        minimum = int(evidence["minimum_updates"])
        maximum = int(evidence["maximum_updates"])
        observed = int(evidence["observed_updates"])
        if not (
            evidence.get("schema") == FAST_BLT_GRAPH_QUIESCENCE_SCHEMA
            and evidence.get("achieved") is achieved
            and evidence.get("materialize_metrics") is materialize_metrics
            and 2 <= minimum <= observed <= maximum
            and maximum == FAST_BLT_MAX_GRAPH_WARMUP_UPDATES
            and (achieved or observed == maximum)
            and observed == len(observations) == len(warmup_records)
            and evidence.get("observations_sha256")
            == canonical_json_sha256(observations)
        ):
            return False

        totals = {"unique_graphs": 0, "recompiles": 0, "graph_breaks": 0}
        first_plateau = None
        any_representative = False
        for index, (observation, record) in enumerate(
            zip(observations, warmup_records, strict=True), start=1
        ):
            if not isinstance(observation, Mapping) or not isinstance(record, Mapping):
                return False
            planned_min = int(observation["planned_min_microbatch"])
            planned_max = int(observation["planned_max_microbatch"])
            representative = planned_min > 1 and planned_max > 1
            any_representative = any_representative or representative
            deltas = {
                "unique_graphs": int(observation["new_unique_graphs"]),
                "recompiles": int(observation["new_recompiles"]),
                "graph_breaks": int(observation["new_graph_breaks"]),
            }
            if (
                observation.get("update") != index
                or observation.get("materialize_metrics")
                is not materialize_metrics
                or observation.get("representative_non_singleton")
                is not representative
                or record.get("planned_min_microbatch") != planned_min
                or record.get("planned_max_microbatch") != planned_max
                or any(value < 0 for value in deltas.values())
            ):
                return False
            for key, value in deltas.items():
                totals[key] += value
            if index >= minimum and representative and all(
                value == 0 for value in deltas.values()
            ):
                first_plateau = index if first_plateau is None else first_plateau

        return bool(
            evidence.get("representative_non_singleton") is any_representative
            and (
                first_plateau == observed if achieved else first_plateau is None
            )
            and totals["unique_graphs"] == int(evidence["total_unique_graphs"])
            and totals["recompiles"] == int(evidence["total_recompiles"])
            and totals["graph_breaks"] == int(evidence["total_graph_breaks"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def _memory_phase_passes(
    phase: Mapping[str, object], *, total_memory: int, required: int
) -> bool:
    environment = phase.get("physical_environment")
    if not isinstance(environment, Mapping):
        return False
    try:
        peak_reserved = int(phase["cuda_peak_reserved_bytes"])
        intrinsic = total_memory - peak_reserved
        free_samples = tuple(
            int(value) for value in environment["free_bytes_samples"]
        )
        used_samples = tuple(
            int(value) for value in environment["used_bytes_samples"]
        )
        return (
            0 <= peak_reserved <= total_memory
            and phase.get("cuda_total_bytes") == total_memory
            and phase.get("required_headroom_bytes") == required
            and phase.get("intrinsic_reserved_headroom_bytes") == intrinsic
            and intrinsic >= required
            and len(free_samples) == len(used_samples)
            and int(environment["sample_count"]) == len(free_samples)
            and len(free_samples) >= MIN_TELEMETRY_SAMPLES
            and all(0 <= value <= total_memory for value in free_samples)
            and all(0 <= value <= total_memory for value in used_samples)
            and environment.get("free_bytes_samples_sha256")
            == canonical_json_sha256(free_samples)
            and environment.get("used_bytes_samples_sha256")
            == canonical_json_sha256(used_samples)
            and int(environment["minimum_free_bytes"]) == min(free_samples)
            and int(environment["minimum_free_bytes"]) >= required
            and int(environment["maximum_used_bytes"]) == max(used_samples)
            and int(environment["maximum_used_bytes"]) <= total_memory - required
            and environment.get("required_free_bytes") == required
        )
    except (KeyError, TypeError, ValueError):
        return False


def _telemetry_domains_pass(phase: Mapping[str, object]) -> bool:
    """Authenticate full-lifecycle memory and bounded scored observations."""

    environment = phase.get("physical_environment")
    lifecycle = phase.get("telemetry_lifecycle")
    if not isinstance(environment, Mapping) or not isinstance(lifecycle, Mapping):
        return False
    if set(lifecycle) != {
        "schema",
        "measurement_begin_ns",
        "measurement_end_ns",
        "measurement_duration_seconds",
        "observation_count",
        "observations",
        "observations_sha256",
        "scored_observation_sequences",
        "scored_observation_sequences_sha256",
        "query_interval_seconds",
        "query_timeout_seconds",
        "thread_join_timeout_seconds",
        "maximum_coverage_gap_seconds",
        "first_scored_start_gap_seconds",
        "last_scored_end_gap_seconds",
        "maximum_scored_query_gap_seconds",
        "coverage_complete",
        "stop_requested",
        "thread_terminated",
        "clean_completion",
        "error_count",
        "errors",
        "errors_sha256",
    }:
        return False
    utilization = _summary_values(phase.get("gpu_utilization_percent"))
    power = _summary_values(phase.get("power_w"))
    if utilization is None or power is None:
        return False
    try:
        free = tuple(environment["free_bytes_samples"])
        used = tuple(environment["used_bytes_samples"])
        observations = tuple(lifecycle["observations"])
        scored_sequences = tuple(
            int(value) for value in lifecycle["scored_observation_sequences"]
        )
        begin_ns = lifecycle["measurement_begin_ns"]
        end_ns = lifecycle["measurement_end_ns"]
        marked = begin_ns is not None or end_ns is not None
        if marked:
            if (
                isinstance(begin_ns, bool)
                or isinstance(end_ns, bool)
                or not isinstance(begin_ns, int)
                or not isinstance(end_ns, int)
                or not 0 <= begin_ns <= end_ns
            ):
                return False
        elif begin_ns is not None or end_ns is not None:
            return False
        expected_fields = {
            "seq",
            "query_start_ns",
            "query_end_ns",
            "power_w",
            "utilization_percent",
            "free_mib",
            "used_mib",
        }
        parsed: list[tuple[int, int, int, float, float, float, float]] = []
        for expected_seq, observation in enumerate(observations):
            if not isinstance(observation, Mapping) or set(observation) != expected_fields:
                return False
            seq = observation["seq"]
            query_start_ns = observation["query_start_ns"]
            query_end_ns = observation["query_end_ns"]
            if (
                isinstance(seq, bool)
                or isinstance(query_start_ns, bool)
                or isinstance(query_end_ns, bool)
                or not isinstance(seq, int)
                or not isinstance(query_start_ns, int)
                or not isinstance(query_end_ns, int)
                or seq != expected_seq
                or query_start_ns < 0
                or query_end_ns < query_start_ns
            ):
                return False
            values = tuple(
                float(observation[name])
                for name in (
                    "power_w",
                    "utilization_percent",
                    "free_mib",
                    "used_mib",
                )
            )
            if not all(math.isfinite(value) and value >= 0.0 for value in values):
                return False
            if not 0.0 <= values[1] <= 100.0:
                return False
            parsed.append((seq, query_start_ns, query_end_ns, *values))
        if any(
            right[1] < left[1] or right[2] < left[2]
            for left, right in zip(parsed[:-1], parsed[1:], strict=True)
        ):
            return False
        expected_scored = tuple(
            value[0]
            for value in parsed
            if not marked or (value[1] >= begin_ns and value[2] <= end_ns)
        )
        scored = tuple(parsed[index] for index in scored_sequences)
        mib = 1 << 20
        duration = lifecycle["measurement_duration_seconds"]
        if marked:
            expected_duration = (end_ns - begin_ns) / 1_000_000_000.0
            tolerance = max(
                TIMING_COHERENCE_ABS_MS / 1_000.0,
                TIMING_COHERENCE_REL_TOL * float(phase["elapsed_seconds"]),
            )
            duration_ok = bool(
                _finite_positive(duration)
                and math.isclose(
                    float(duration), expected_duration, rel_tol=0.0, abs_tol=1e-12
                )
                and abs(expected_duration - float(phase["elapsed_seconds"]))
                <= tolerance
            )
        else:
            duration_ok = duration is None and phase.get("kind") == "training"
        errors = tuple(lifecycle["errors"])
        errors_ok = bool(
            lifecycle.get("error_count") == len(errors) == 0
            and lifecycle.get("errors_sha256") == canonical_json_sha256(errors)
        )
        if marked and scored:
            # An nvidia-smi result is a point observation at query completion;
            # subprocess runtime cannot count as sampled coverage.
            first_gap = (scored[0][2] - begin_ns) / 1_000_000_000.0
            last_gap = (end_ns - scored[-1][2]) / 1_000_000_000.0
            maximum_gap = max(
                (
                    right[2] - left[2]
                    for left, right in zip(scored[:-1], scored[1:], strict=True)
                ),
                default=0,
            ) / 1_000_000_000.0
            coverage_ok = bool(
                lifecycle.get("coverage_complete") is True
                and math.isclose(
                    float(lifecycle["first_scored_start_gap_seconds"]),
                    first_gap,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(lifecycle["last_scored_end_gap_seconds"]),
                    last_gap,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(lifecycle["maximum_scored_query_gap_seconds"]),
                    maximum_gap,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and max(first_gap, last_gap, maximum_gap)
                <= FAST_BLT_TELEMETRY_MAX_COVERAGE_GAP_SECONDS
            )
        else:
            coverage_ok = bool(
                not marked
                and phase.get("kind") == "training"
                and lifecycle.get("coverage_complete") is False
                and lifecycle.get("first_scored_start_gap_seconds") is None
                and lifecycle.get("last_scored_end_gap_seconds") is None
                and lifecycle.get("maximum_scored_query_gap_seconds") is None
            )
        return bool(
            lifecycle.get("schema") == FAST_BLT_TELEMETRY_LIFECYCLE_SCHEMA
            and int(lifecycle["observation_count"]) == len(observations)
            and lifecycle.get("observations_sha256")
            == canonical_json_sha256(observations)
            and lifecycle.get("scored_observation_sequences_sha256")
            == canonical_json_sha256(scored_sequences)
            and scored_sequences == expected_scored
            and len(utilization) == len(power) == len(scored_sequences)
            and tuple(power) == tuple(value[3] for value in scored)
            and tuple(utilization) == tuple(value[4] for value in scored)
            and len(free) == len(used) == len(observations)
            and len(observations) == int(environment["sample_count"])
            and tuple(free) == tuple(int(value[5] * mib) for value in parsed)
            and tuple(used) == tuple(int(value[6] * mib) for value in parsed)
            and duration_ok
            and lifecycle.get("query_interval_seconds")
            == FAST_BLT_TELEMETRY_QUERY_INTERVAL_SECONDS
            and lifecycle.get("query_timeout_seconds")
            == FAST_BLT_TELEMETRY_QUERY_TIMEOUT_SECONDS
            and lifecycle.get("thread_join_timeout_seconds")
            == FAST_BLT_TELEMETRY_THREAD_JOIN_TIMEOUT_SECONDS
            and lifecycle.get("maximum_coverage_gap_seconds")
            == FAST_BLT_TELEMETRY_MAX_COVERAGE_GAP_SECONDS
            and lifecycle.get("stop_requested") is True
            and lifecycle.get("thread_terminated") is True
            and lifecycle.get("clean_completion") is True
            and errors_ok
            and coverage_ok
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return False


def _validation_metrics_pass(
    metrics: object,
    *,
    phase_elapsed_seconds: float,
    measured_records: Sequence[object],
) -> bool:
    """Authenticate the complete validation metric surface and denominators."""

    if not isinstance(metrics, Mapping):
        return False
    expected_fields = {
        "ar_loss",
        "bpb",
        "atomic_bpb",
        "diffusion_loss",
        "diffusion_elbo_proxy_bpb",
        "diffusion_elbo_proxy_nats_per_block_atom",
        "diffusion_elbo_atoms",
        "ar_targets",
        "literal_bytes",
        "special_targets",
        "diffusion_targets",
        "diffusion_chunks",
        "elapsed_ms",
        "diffusion_role_nll",
        "diffusion_role_counts",
    }
    if set(metrics) != expected_fields:
        return False
    try:
        floats = {
            name: float(metrics[name])
            for name in (
                "ar_loss",
                "bpb",
                "atomic_bpb",
                "diffusion_loss",
                "diffusion_elbo_proxy_bpb",
                "diffusion_elbo_proxy_nats_per_block_atom",
                "diffusion_elbo_atoms",
                "elapsed_ms",
            )
        }
        integer_names = (
            "ar_targets",
            "literal_bytes",
            "special_targets",
            "diffusion_targets",
            "diffusion_chunks",
        )
        if not all(
            isinstance(metrics[name], int)
            and not isinstance(metrics[name], bool)
            for name in integer_names
        ):
            return False
        counts = {name: int(metrics[name]) for name in integer_names}
        role_nll_raw = metrics["diffusion_role_nll"]
        role_counts_raw = metrics["diffusion_role_counts"]
        if not isinstance(role_nll_raw, (list, tuple)) or not isinstance(
            role_counts_raw, (list, tuple)
        ):
            return False
        role_nll = tuple(float(value) for value in role_nll_raw)
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in role_counts_raw
        ):
            return False
        role_counts = tuple(int(value) for value in role_counts_raw)
        weighted_role_loss = math.fsum(
            loss * count
            for loss, count in zip(role_nll, role_counts, strict=True)
        ) / counts["diffusion_targets"]
        return bool(
            all(math.isfinite(value) and value >= 0 for value in floats.values())
            and floats["elapsed_ms"] > 0
            and floats["elapsed_ms"]
            <= 1_000.0 * phase_elapsed_seconds + TIMING_COHERENCE_ABS_MS
            and counts["ar_targets"] > 0
            and counts["literal_bytes"] > 0
            and counts["special_targets"] >= 0
            and counts["ar_targets"]
            == counts["literal_bytes"] + counts["special_targets"]
            and counts["diffusion_targets"] > 0
            and counts["diffusion_chunks"]
            == FAST_BLT_DIFFUSION_VALIDATION_ROWS
            and bool(measured_records)
            and all(isinstance(record, Mapping) for record in measured_records)
            and all(
                int(record["active_bytes"]) == counts["diffusion_targets"]  # type: ignore[index]
                and int(record["branch_atoms"])  # type: ignore[index]
                == int(floats["diffusion_elbo_atoms"])
                and int(record["rows"]) == counts["diffusion_chunks"]  # type: ignore[index]
                for record in measured_records
            )
            and floats["diffusion_elbo_atoms"]
            >= counts["diffusion_targets"]
            and len(role_nll) == len(role_counts) == 6
            and all(math.isfinite(value) and value >= 0 for value in role_nll)
            and all(value >= 0 for value in role_counts)
            and sum(role_counts) == counts["diffusion_targets"]
            and math.isclose(
                floats["bpb"],
                floats["ar_loss"] / math.log(2.0),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and math.isclose(
                floats["atomic_bpb"],
                floats["ar_loss"] / math.log(2.0),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and math.isclose(
                floats["diffusion_elbo_proxy_bpb"],
                floats["diffusion_elbo_proxy_nats_per_block_atom"]
                / math.log(2.0),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            and math.isclose(
                floats["diffusion_loss"],
                weighted_role_loss,
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def _update_evidence_passes(
    record: object,
    *,
    rows: int,
    batching_policy: str,
    kind: str,
    expected_token_budget: int,
) -> bool:
    if not isinstance(record, Mapping):
        return False
    try:
        origins = int(record["origin_count"])
        planned_microsteps = int(record["planned_microsteps"])
        planned_maximum = int(record["planned_max_physical_positions"])
        max_row_origins = int(record["max_row_origin_count"])
        planned_min_microbatch = int(record["planned_min_microbatch"])
        planned_max_microbatch = int(record["planned_max_microbatch"])
        boundaries = tuple(
            int(value) for value in record["planned_group_row_cu_seqlens"]
        )
        group_physical = tuple(
            int(value) for value in record["planned_group_physical_positions"]
        )
        ordered_row_origins = tuple(
            int(value) for value in record["ordered_row_origin_counts"]
        )
        group_origins = tuple(
            int(value) for value in record["planned_group_origin_counts"]
        )
        group_row_hashes = tuple(
            str(value)
            for value in record["planned_group_row_indices_sha256"]
        )
        group_sizes = tuple(
            stop - start
            for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
        )
        total_physical = rows * FAST_BLT_ROW_LENGTH + origins * FAST_BLT_BLOCK_LENGTH
        batching_ok = (
            int(record["physical_token_budget"]) == expected_token_budget
            and planned_maximum <= expected_token_budget
            if batching_policy == "token_budgeted_ragged"
            else record.get("physical_token_budget") is None
        )
        common = (
            record.get("rows") == rows
            and origins > 0
            and planned_microsteps > 0
            and len(boundaries) == planned_microsteps + 1
            and len(group_physical) == planned_microsteps
            and len(ordered_row_origins) == rows
            and len(group_origins) == planned_microsteps
            and len(group_row_hashes) == planned_microsteps
            and boundaries[0] == 0
            and boundaries[-1] == rows
            and all(size > 0 for size in group_sizes)
            and sum(group_sizes) == rows
            and 0 < planned_min_microbatch <= planned_max_microbatch
            and min(group_sizes) == planned_min_microbatch
            and max(group_sizes) == planned_max_microbatch
            and planned_max_microbatch <= rows
            and all(
                size <= planned_max_microbatch for size in group_sizes
            )
            and all(value > 0 for value in group_physical)
            and all(value > 0 for value in ordered_row_origins)
            and all(value > 0 for value in group_origins)
            and sum(ordered_row_origins) == origins
            and sum(group_origins) == origins
            and all(
                group_origin
                == sum(ordered_row_origins[start:stop])
                for group_origin, start, stop in zip(
                    group_origins,
                    boundaries[:-1],
                    boundaries[1:],
                    strict=True,
                )
            )
            and sum(group_physical) == total_physical
            and max(group_physical) == planned_maximum
            and all(
                value == size * FAST_BLT_ROW_LENGTH
                + group_origin * FAST_BLT_BLOCK_LENGTH
                and (value - size * FAST_BLT_ROW_LENGTH)
                % FAST_BLT_BLOCK_LENGTH
                == 0
                for value, size, group_origin in zip(
                    group_physical, group_sizes, group_origins, strict=True
                )
            )
            and record.get("planned_group_row_cu_seqlens_sha256")
            == canonical_json_sha256(boundaries)
            and record.get("planned_group_physical_positions_sha256")
            == canonical_json_sha256(group_physical)
            and record.get("ordered_row_origin_counts_sha256")
            == canonical_json_sha256(ordered_row_origins)
            and record.get("planned_group_origin_counts_sha256")
            == canonical_json_sha256(group_origins)
            and record.get("planned_group_row_indices_sha256_sha256")
            == canonical_json_sha256(group_row_hashes)
            and all(len(value) == 64 for value in group_row_hashes)
            and max_row_origins > 0
            and record.get("total_physical_positions") == total_physical
            and planned_maximum
            >= FAST_BLT_ROW_LENGTH + max_row_origins * FAST_BLT_BLOCK_LENGTH
            and batching_ok
            and isinstance(record.get("row_origin_counts_sha256"), str)
            and len(str(record["row_origin_counts_sha256"])) == 64
            and isinstance(record.get("row_indices_sha256"), str)
            and len(str(record["row_indices_sha256"])) == 64
        )
        if not common:
            return False
        if kind == "training":
            return bool(
                record.get("materialize_metrics") is False
                and _finite_positive(record.get("host_enqueue_ms"))
                and _finite_positive(record.get("cuda_event_ms"))
                and int(record["actual_step"]) > 0
                and int(record["actual_rows"]) == rows
                and int(record["actual_microsteps"]) == planned_microsteps
                and tuple(record["actual_group_row_cu_seqlens"]) == boundaries
                and tuple(record["actual_group_physical_positions"])
                == group_physical
                and int(record["actual_max_microbatch"])
                == planned_max_microbatch
                and int(record["actual_max_physical_positions"])
                == planned_maximum
                and all(
                    field not in record
                    for field in (
                        "active_bytes",
                        "branch_atoms",
                        "branch_blocks",
                        "microsteps",
                        "max_microbatch",
                        "max_physical_positions",
                        "block_mask_build_ms",
                    )
                )
            )
        if kind not in {"training_diagnostic", "validation"}:
            return False
        branch_blocks = int(record["branch_blocks"])
        branch_atoms = int(record["branch_atoms"])
        active = int(record["active_bytes"])
        microsteps = int(record["microsteps"])
        maximum = int(record["max_physical_positions"])
        is_diagnostic = kind == "training_diagnostic"
        validation_cache_fields_ok = True
        if kind == "validation":
            cache_values = tuple(
                record.get(name)
                for name in (
                    "schedule_cache_misses",
                    "corruption_plan_cache_misses",
                    "ragged_mask_metadata_cache_misses",
                    "first_batch_cache_misses",
                    "first_batch_waits",
                )
            )
            validation_cache_fields_ok = bool(
                all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    for value in cache_values
                )
                and isinstance(record.get("cache_misses"), int)
                and not isinstance(record.get("cache_misses"), bool)
                and int(record["cache_misses"]) == sum(cache_values[:-1])
            )
        return bool(
            branch_blocks == origins
            and 0 < active <= branch_atoms <= branch_blocks * FAST_BLT_BLOCK_LENGTH
            and microsteps == planned_microsteps
            and int(record["max_microbatch"]) == planned_max_microbatch
            and maximum == planned_maximum
            and validation_cache_fields_ok
            and (
                record.get("materialize_metrics") is True
                and _finite_positive(record.get("block_mask_build_ms"))
                and _finite_positive(record.get("host_synchronized_call_ms"))
                and _finite_positive(record.get("cuda_event_ms"))
                and float(record["host_synchronized_call_ms"])
                + max(
                    TIMING_COHERENCE_ABS_MS,
                    TIMING_COHERENCE_REL_TOL
                    * float(record["cuda_event_ms"]),
                )
                >= float(record["cuda_event_ms"])
                and int(record["actual_step"]) > 0
                and int(record["actual_rows"]) == rows
                and int(record["actual_microsteps"]) == planned_microsteps
                and tuple(record["actual_group_row_cu_seqlens"])
                == boundaries
                and tuple(record["actual_group_physical_positions"])
                == group_physical
                and int(record["actual_max_microbatch"])
                == planned_max_microbatch
                and int(record["actual_max_physical_positions"])
                == planned_maximum
                if is_diagnostic
                else record.get("materialize_metrics") is None
                and record.get("block_mask_build_ms") is None
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def _workload_passes(
    workload: object,
    *,
    rows: int,
    repetitions: int,
    kind: str,
    expected_token_budget: int,
) -> bool:
    if not isinstance(workload, Mapping):
        return False
    records = workload.get("updates")
    if not isinstance(records, list):
        return False
    try:
        batching_policy = "token_budgeted_ragged"
        return bool(
            workload.get("schema") == FAST_BLT_WORKLOAD_SCHEMA
            and workload.get("origin_policy") == FAST_BLT_ORIGIN_POLICY
            and workload.get("block_length") == FAST_BLT_BLOCK_LENGTH
            and workload.get("rows_per_update") == rows
            and workload.get("batching_policy") == batching_policy
            and workload.get("repetitions") == repetitions
            and len(records) == repetitions
            and workload.get("updates_sha256") == canonical_json_sha256(records)
            and all(
                _update_evidence_passes(
                    record, rows=rows, batching_policy=batching_policy,
                    kind=kind,
                    expected_token_budget=expected_token_budget,
                )
                for record in records
            )
            and (
                kind == "validation"
                or all(
                    int(right["actual_step"]) == int(left["actual_step"]) + 1
                    for left, right in zip(records[:-1], records[1:], strict=True)
                )
            )
            and (
                kind != "validation"
                or all(
                    record.get("row_indices_sha256")
                    == FAST_BLT_DIFFUSION_VALIDATION_INDICES_SHA256
                    for record in records
                )
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


_VALIDATION_CACHE_COUNTERS = (
    "schedule_cache_misses",
    "corruption_plan_cache_misses",
    "ragged_mask_metadata_cache_misses",
    "first_batch_cache_misses",
    "first_batch_waits",
)


def _validation_cache_quiescence_passes(
    evidence: object,
    *,
    measured_records: object,
    measured_repetitions: int,
    expected_quiescent: bool = True,
) -> bool:
    """Authenticate that measured validation reused all invariant staging."""

    if not isinstance(evidence, Mapping) or not isinstance(measured_records, list):
        return False
    if set(evidence) != {
        "measured_cache_misses",
        "measured_repetitions",
        "quiescent",
        "measured_first_batch_waits",
        "records_sha256",
    }:
        return False
    if len(measured_records) != measured_repetitions:
        return False
    authenticated_records: list[dict[str, int]] = []
    try:
        for record in measured_records:
            if not isinstance(record, Mapping):
                return False
            counters: dict[str, int] = {}
            for name in _VALIDATION_CACHE_COUNTERS:
                value = record.get(name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    return False
                counters[name] = value
            cache_misses = record.get("cache_misses")
            expected_misses = sum(
                counters[name]
                for name in _VALIDATION_CACHE_COUNTERS
                if name != "first_batch_waits"
            )
            if (
                isinstance(cache_misses, bool)
                or not isinstance(cache_misses, int)
                or cache_misses != expected_misses
            ):
                return False
            authenticated_records.append(counters)
        measured_cache_misses = sum(
            int(record["cache_misses"]) for record in measured_records
        )
        measured_first_batch_waits = sum(
            int(record["first_batch_waits"]) for record in measured_records
        )
        quiescent = measured_cache_misses == 0 and measured_first_batch_waits == 0
        return bool(
            evidence.get("measured_cache_misses") == measured_cache_misses
            and evidence.get("measured_first_batch_waits")
            == measured_first_batch_waits
            and evidence.get("measured_repetitions") == measured_repetitions
            and evidence.get("quiescent") is quiescent
            and quiescent is expected_quiescent
            and evidence.get("records_sha256")
            == canonical_json_sha256(tuple(authenticated_records))
        )
    except (KeyError, TypeError, ValueError):
        return False


def _phase_passes(
    phase: object,
    *,
    kind: str,
    total_memory: int,
    required: int,
    minimum_warmup: int,
    minimum_measured: int,
    expected_run_contract: Mapping[str, object],
) -> bool:
    if not isinstance(phase, Mapping):
        return False
    try:
        warmup = int(phase["warmup_repetitions"])
        measured = int(phase["measured_repetitions"])
        expected_token_budget = int(
            expected_run_contract["microbatch_token_budget"]
        )
        rows = (
            FAST_BLT_GLOBAL_BATCH
            if kind == "training"
            else FAST_BLT_DIFFUSION_VALIDATION_ROWS
        )
        base = (
            phase.get("kind") == kind
            and phase.get("status") == "ok"
            and phase.get("eligible") is True
            and warmup >= minimum_warmup
            and measured >= minimum_measured
            and _finite_positive(phase.get("elapsed_seconds"))
            and _graph_phase_passes(phase)
            and (
                kind == "training" or int(phase["warmup_unique_graphs"]) >= 1
            )
            and _memory_phase_passes(phase, total_memory=total_memory, required=required)
            and _telemetry_domains_pass(phase)
            and _summary_passes(
                phase.get("gpu_utilization_percent"),
                mean=SUSTAINED_GPU_POLICY["minimum_mean_utilization_percent"],
                p10=SUSTAINED_GPU_POLICY["minimum_p10_utilization_percent"],
            )
            and _summary_passes(
                phase.get("power_w"),
                mean=SUSTAINED_GPU_POLICY["minimum_mean_power_w"],
                p10=SUSTAINED_GPU_POLICY["minimum_p10_power_w"],
            )
            and _workload_passes(
                phase.get("warmup_workload"), rows=rows,
                repetitions=warmup, kind=kind,
                expected_token_budget=expected_token_budget,
            )
            and _workload_passes(
                phase.get("measured_workload"), rows=rows,
                repetitions=measured, kind=kind,
                expected_token_budget=expected_token_budget,
            )
        )
        if kind == "training":
            warmup_workload = phase.get("warmup_workload")
            measured_workload = phase.get("measured_workload")
            diagnostic_workload = phase.get("diagnostic_workload")
            if (
                not isinstance(warmup_workload, Mapping)
                or not isinstance(measured_workload, Mapping)
                or not isinstance(diagnostic_workload, Mapping)
            ):
                return False
            warmup_records = warmup_workload.get("updates")
            measured_records = measured_workload.get("updates")
            diagnostic_records = diagnostic_workload.get("updates")
            if not isinstance(warmup_records, list) or not isinstance(
                measured_records, list
            ) or not isinstance(diagnostic_records, list):
                return False
            execution_step_range = phase.get("execution_step_range")
            if not isinstance(execution_step_range, Mapping):
                return False
            execution_steps = tuple(
                int(record["actual_step"])
                for record in (
                    *diagnostic_records,
                    *warmup_records,
                    *measured_records,
                )
            )
            host_summary = phase.get("host_enqueue_ms")
            cuda_summary = phase.get("cuda_event_update_ms")
            graph_quiescence = phase.get("graph_quiescence")
            diagnostic_quiescence = phase.get("diagnostic_graph_quiescence")
            if not isinstance(graph_quiescence, Mapping) or not isinstance(
                diagnostic_quiescence, Mapping
            ):
                return False
            diagnostic_update = phase.get("diagnostic_update")
            schedule = phase.get("production_schedule")
            if not isinstance(diagnostic_update, Mapping) or not isinstance(
                schedule, Mapping
            ):
                return False
            elapsed_ms = 1_000.0 * float(phase["elapsed_seconds"])
            coherence_tolerance = max(
                TIMING_COHERENCE_ABS_MS,
                TIMING_COHERENCE_REL_TOL * elapsed_ms,
            )
            host_sum = math.fsum(
                float(record["host_enqueue_ms"])
                for record in measured_records
            )
            cuda_sum = math.fsum(
                float(record["cuda_event_ms"])
                for record in measured_records
            )
            iterations = int(expected_run_contract["iterations"])
            train_log_every = int(expected_run_contract["train_log_every"])
            validation_every = int(expected_run_contract["val_loss_every"])
            materialized_updates = _materialized_update_count(
                iterations=iterations,
                train_log_every=train_log_every,
                validation_every=validation_every,
            )
            ordinary_updates = iterations - materialized_updates
            ordinary_update_ms = float(phase["phase_wall_update_ms"])
            diagnostic_update_ms = float(
                diagnostic_update["host_synchronized_call_ms"]
            )
            schedule_update_ms = (
                ordinary_updates * ordinary_update_ms
                + materialized_updates * diagnostic_update_ms
            ) / iterations
            clean_bytes = FAST_BLT_GLOBAL_BATCH * FAST_BLT_ROW_LENGTH
            return bool(
                base
                and bool(execution_steps)
                and execution_steps[0] == 1
                and execution_steps[-1] == len(execution_steps)
                and all(
                    right == left + 1
                    for left, right in zip(
                        execution_steps[:-1], execution_steps[1:], strict=True
                    )
                )
                and execution_step_range.get("first_completed_step")
                == 1
                and execution_step_range.get("last_completed_step")
                == len(execution_steps)
                and execution_step_range.get("diagnostic_warmup_updates")
                == len(diagnostic_records)
                and execution_step_range.get("ordinary_warmup_updates")
                == len(warmup_records)
                and execution_step_range.get("ordinary_measured_updates")
                == len(measured_records)
                and execution_step_range.get("total_updates")
                == len(execution_steps)
                and execution_step_range.get("contiguous") is True
                and _graph_quiescence_passes(
                    graph_quiescence,
                    warmup_records=warmup_records,
                    materialize_metrics=False,
                )
                and graph_quiescence.get("total_unique_graphs")
                == phase.get("warmup_unique_graphs")
                and graph_quiescence.get("total_recompiles")
                == phase.get("warmup_recompiles")
                and graph_quiescence.get("total_graph_breaks")
                == phase.get("warmup_graph_breaks")
                and _workload_passes(
                    diagnostic_workload,
                    rows=rows,
                    repetitions=len(diagnostic_records),
                    kind="training_diagnostic",
                    expected_token_budget=expected_token_budget,
                )
                and _graph_quiescence_passes(
                    diagnostic_quiescence,
                    warmup_records=diagnostic_records,
                    materialize_metrics=True,
                )
                and int(graph_quiescence["total_unique_graphs"])
                + int(diagnostic_quiescence["total_unique_graphs"])
                >= 1
                and _timing_summary_passes(
                    host_summary, measured_records, field="host_enqueue_ms"
                )
                and _timing_summary_passes(
                    cuda_summary, measured_records, field="cuda_event_ms"
                )
                and _finite_positive(phase.get("phase_wall_update_ms"))
                and phase.get("timing_semantics")
                == _timing_semantics_contract()
                and math.isclose(
                    float(phase["phase_wall_update_ms"]),
                    1_000.0 * float(phase["elapsed_seconds"]) / measured,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
                and math.isclose(
                    float(phase["cuda_event_elapsed_seconds"]),
                    cuda_sum / 1_000.0,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
                and isinstance(host_summary, Mapping)
                and host_sum <= elapsed_ms + coherence_tolerance
                and cuda_sum <= elapsed_ms + coherence_tolerance
                and math.isclose(
                    float(phase["update_ms"]),
                    float(phase["phase_wall_update_ms"]),
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
                and diagnostic_update == diagnostic_records[-1]
                and phase.get("diagnostic_update_sha256")
                == canonical_json_sha256(diagnostic_update)
                and math.isclose(
                    float(phase["diagnostic_update_ms"]),
                    diagnostic_update_ms,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
                and schedule.get("iterations") == iterations
                and schedule.get("train_log_every") == train_log_every
                and schedule.get("validation_every") == validation_every
                and schedule.get("ordinary_updates") == ordinary_updates
                and schedule.get("materialized_updates")
                == materialized_updates
                and math.isclose(
                    float(schedule["ordinary_update_ms"]),
                    ordinary_update_ms,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
                and math.isclose(
                    float(schedule["materialized_update_ms"]),
                    diagnostic_update_ms,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
                and math.isclose(
                    float(phase["production_schedule_update_ms"]),
                    schedule_update_ms,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
                and math.isclose(
                    float(phase["ordinary_clean_bytes_per_second"]),
                    1_000.0 * clean_bytes / ordinary_update_ms,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
                and math.isclose(
                    float(
                        phase["production_schedule_clean_bytes_per_second"]
                    ),
                    1_000.0 * clean_bytes / schedule_update_ms,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
            )
        return bool(
            base
            and phase.get("total_validation_rows") == FAST_BLT_VALIDATION_ROWS
            and phase.get("diffusion_validation_rows")
            == FAST_BLT_DIFFUSION_VALIDATION_ROWS
            and phase.get("ar_max_batch_size") == FAST_BLT_AR_VALIDATION_BATCH
            and phase.get("diffusion_max_batch_size") == FAST_BLT_VALIDATION_BATCH
            and phase.get("batching_policy") == "token_budgeted_ragged_diffusion"
            and _validation_cache_quiescence_passes(
                phase.get("cache_quiescence"),
                measured_records=(
                    phase["measured_workload"]["updates"]  # type: ignore[index]
                ),
                measured_repetitions=measured,
            )
            and _finite_positive(
                phase.get("end_to_end_total_rows_per_second")
            )
            and _finite_positive(
                phase.get("end_to_end_diffusion_rows_per_second")
            )
            and _validation_metrics_pass(
                phase.get("metrics"),
                phase_elapsed_seconds=float(phase["elapsed_seconds"]),
                measured_records=(
                    phase["measured_workload"]["updates"]  # type: ignore[index]
                ),
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def _block_mask_build_passes(
    phase: object,
    *,
    training: object,
    microbatch: int,
    total_memory: int,
    required: int,
) -> bool:
    if not isinstance(phase, Mapping) or not isinstance(training, Mapping):
        return False
    timing = phase.get("build_ms")
    topology = phase.get("topology")
    if not isinstance(timing, Mapping) or not isinstance(topology, Mapping):
        return False
    diagnostic_workload = training.get("diagnostic_workload")
    if not isinstance(diagnostic_workload, Mapping):
        return False
    diagnostic_records = diagnostic_workload.get("updates")
    if not isinstance(diagnostic_records, list) or not diagnostic_records:
        return False
    source = diagnostic_records[0]
    if not isinstance(source, Mapping):
        return False
    try:
        peak_reserved = int(phase["cuda_peak_reserved_bytes"])
        group_index = int(topology["group_index"])
        boundaries = tuple(
            int(value) for value in source["planned_group_row_cu_seqlens"]
        )
        group_physical = tuple(
            int(value) for value in source["planned_group_physical_positions"]
        )
        group_origins = tuple(
            int(value) for value in source["planned_group_origin_counts"]
        )
        group_row_hashes = tuple(
            str(value)
            for value in source["planned_group_row_indices_sha256"]
        )
        heaviest = max(
            range(len(group_physical)),
            key=group_physical.__getitem__,
        )
        expected_rows = boundaries[group_index + 1] - boundaries[group_index]
        expected_physical = group_physical[group_index]
        expected_origins = group_origins[group_index]
        return bool(
            phase.get("kind") == "ragged_block_mask_build"
            and phase.get("status") == "ok"
            and phase.get("compiler_disabled") is True
            and int(phase["warmup_repetitions"]) >= 1
            and int(phase["measured_repetitions"]) >= 4
            and _finite_positive(phase.get("elapsed_seconds"))
            and (timing_values := _summary_values(timing)) is not None
            and len(timing_values) == int(phase["measured_repetitions"])
            and all(value > 0 for value in timing_values)
            and math.fsum(timing_values)
            <= 1_000.0 * float(phase["elapsed_seconds"])
            + TIMING_COHERENCE_ABS_MS
            and topology.get("block_length") == FAST_BLT_BLOCK_LENGTH
            and group_index == heaviest
            and int(topology["rows"]) == expected_rows
            and 0 < int(topology["rows"]) <= microbatch
            and int(topology["branch_blocks"]) == expected_origins
            and 0 < int(topology["branch_atoms"]) <= int(topology["branch_blocks"]) * 4
            and int(topology["clean_positions"])
            == expected_rows * FAST_BLT_ROW_LENGTH
            and int(topology["physical_positions"]) == expected_physical
            and int(topology["physical_positions"])
            == int(topology["clean_positions"])
            + int(topology["branch_blocks"]) * FAST_BLT_BLOCK_LENGTH
            and topology.get("source_update_row_indices_sha256")
            == source.get("row_indices_sha256")
            and topology.get("source_update_row_origin_counts_sha256")
            == source.get("row_origin_counts_sha256")
            and topology.get("group_row_indices_sha256")
            == group_row_hashes[group_index]
            and phase.get("cuda_total_bytes") == total_memory
            and phase.get("required_headroom_bytes") == required
            and phase.get("intrinsic_reserved_headroom_bytes") == total_memory - peak_reserved
            and total_memory - peak_reserved >= required
        )
    except (KeyError, TypeError, ValueError):
        return False


def candidate_passes(
    candidate: object,
    *,
    microbatch: int,
    expected_run_contract: Mapping[str, object],
    total_memory: int,
    expected_parameter_count: int = FAST_BLT_ENTROPY_B4_COMPLETE_PARAMETER_TARGET,
) -> bool:
    """Recompute every candidate gate without trusting emitted eligibility."""

    if not isinstance(candidate, Mapping):
        return False
    required = required_headroom_bytes(total_memory)
    try:
        return bool(
            candidate.get("status") == "ok"
            and candidate.get("eligible") is True
            and candidate.get("microbatch") == microbatch
            and candidate.get("run_contract") == _json_native(dict(expected_run_contract))
            and candidate.get("run_contract_sha256")
            == canonical_json_sha256(expected_run_contract)
            and candidate.get("parameter_count")
            == expected_parameter_count
            and _block_mask_build_passes(
                candidate.get("block_mask_build"),
                training=candidate.get("training"),
                microbatch=microbatch,
                total_memory=total_memory,
                required=required,
            )
            and _phase_passes(
                candidate.get("training"), kind="training", total_memory=total_memory,
                required=required, minimum_warmup=2, minimum_measured=4,
                expected_run_contract=expected_run_contract,
            )
            and _phase_passes(
                candidate.get("validation"), kind="validation", total_memory=total_memory,
                required=required, minimum_warmup=1, minimum_measured=2,
                expected_run_contract=expected_run_contract,
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def _oom_candidate_passes(
    candidate: object,
    *,
    microbatch: int,
    expected_run_contract: Mapping[str, object],
    total_memory: int,
) -> bool:
    """Authenticate a failed candidate without promoting it as eligible."""

    if not isinstance(candidate, Mapping):
        return False
    try:
        return bool(
            candidate.get("status") == "oom"
            and candidate.get("eligible") is False
            and candidate.get("microbatch") == microbatch
            and candidate.get("oom_phase") in FAST_BLT_FAILURE_PHASES
            and isinstance(candidate.get("error"), str)
            and bool(candidate["error"])
            and candidate.get("run_contract")
            == _json_native(dict(expected_run_contract))
            and candidate.get("run_contract_sha256")
            == canonical_json_sha256(expected_run_contract)
            and candidate.get("cuda_total_bytes") == total_memory
            and candidate.get("required_headroom_bytes")
            == required_headroom_bytes(total_memory)
            and candidate.get("oom_physical_total_bytes") == total_memory
            and 0 <= int(candidate["oom_physical_free_bytes"]) <= total_memory
        )
    except (KeyError, TypeError, ValueError):
        return False


def _compile_error_candidate_passes(
    candidate: object,
    *,
    microbatch: int,
    expected_run_contract: Mapping[str, object],
    total_memory: int,
    expected_parameter_count: int = FAST_BLT_ENTROPY_B4_COMPLETE_PARAMETER_TARGET,
) -> bool:
    """Authenticate a recognized PyTorch compiler failure as ineligible."""

    if not isinstance(candidate, Mapping):
        return False
    try:
        traceback_text = candidate["traceback"]
        error = candidate["error"]
        error_repr = candidate["error_repr"]
        error_type = candidate["error_type"]
        peak_allocated = int(candidate["cuda_peak_allocated_bytes"])
        peak_reserved = int(candidate["cuda_peak_reserved_bytes"])
        physical_free = int(candidate["instantaneous_physical_free_bytes"])
        physical_total = int(candidate["instantaneous_physical_total_bytes"])
        if (
            not isinstance(error, str)
            or not all(
                isinstance(value, str) and bool(value)
                for value in (traceback_text, error_repr, error_type)
            )
        ):
            return False
        simple_type = error_type.rsplit(".", 1)[-1]
        return bool(
            candidate.get("status") == "compile_error"
            and candidate.get("eligible") is False
            and candidate.get("microbatch") == microbatch
            and candidate.get("compile_phase") in FAST_BLT_FAILURE_PHASES
            and error_type in FAST_BLT_COMPILE_ERROR_TYPES
            and simple_type in traceback_text
            and (not error or error in traceback_text)
            and candidate.get("error_sha256")
            == hashlib.sha256(error.encode("utf-8")).hexdigest()
            and candidate.get("error_repr_sha256")
            == hashlib.sha256(error_repr.encode("utf-8")).hexdigest()
            and candidate.get("traceback_sha256")
            == hashlib.sha256(traceback_text.encode("utf-8")).hexdigest()
            and candidate.get("run_contract")
            == _json_native(dict(expected_run_contract))
            and candidate.get("run_contract_sha256")
            == canonical_json_sha256(expected_run_contract)
            and candidate.get("parameter_count")
            == expected_parameter_count
            and 0 <= peak_allocated <= peak_reserved <= total_memory
            and candidate.get("cuda_total_bytes") == total_memory
            and candidate.get("required_headroom_bytes")
            == required_headroom_bytes(total_memory)
            and candidate.get("intrinsic_reserved_headroom_bytes")
            == total_memory - peak_reserved
            and physical_total == total_memory
            and 0 <= physical_free <= total_memory
        )
    except (KeyError, TypeError, ValueError):
        return False


def _nonquiescent_candidate_passes(
    candidate: object,
    *,
    microbatch: int,
    expected_run_contract: Mapping[str, object],
    total_memory: int,
    expected_parameter_count: int = FAST_BLT_ENTROPY_B4_COMPLETE_PARAMETER_TARGET,
) -> bool:
    """Authenticate bounded graph-plateau failure as an ineligible result."""

    if not isinstance(candidate, Mapping):
        return False
    workload = candidate.get("warmup_workload")
    evidence = candidate.get("graph_quiescence")
    if not isinstance(workload, Mapping) or not isinstance(evidence, Mapping):
        return False
    records = workload.get("updates")
    if not isinstance(records, list):
        return False
    phase = candidate.get("graph_phase")
    diagnostic = phase == "training_diagnostic_warmup"
    if not diagnostic and phase != "training_warmup":
        return False
    required = required_headroom_bytes(total_memory)
    try:
        peak_reserved = int(candidate["cuda_peak_reserved_bytes"])
        expected_token_budget = int(
            expected_run_contract["microbatch_token_budget"]
        )
        return bool(
            candidate.get("status") == "graph_nonquiescent"
            and candidate.get("eligible") is False
            and candidate.get("microbatch") == microbatch
            and candidate.get("parameter_count")
            == expected_parameter_count
            and isinstance(candidate.get("error"), str)
            and bool(candidate["error"])
            and candidate.get("run_contract")
            == _json_native(dict(expected_run_contract))
            and candidate.get("run_contract_sha256")
            == canonical_json_sha256(expected_run_contract)
            and _workload_passes(
                workload,
                rows=FAST_BLT_GLOBAL_BATCH,
                repetitions=FAST_BLT_MAX_GRAPH_WARMUP_UPDATES,
                kind=("training_diagnostic" if diagnostic else "training"),
                expected_token_budget=expected_token_budget,
            )
            and _graph_quiescence_passes(
                evidence,
                warmup_records=records,
                materialize_metrics=diagnostic,
                achieved=False,
            )
            and candidate.get("cuda_total_bytes") == total_memory
            and candidate.get("required_headroom_bytes") == required
            and candidate.get("intrinsic_reserved_headroom_bytes")
            == total_memory - peak_reserved
            and total_memory - peak_reserved >= required
        )
    except (KeyError, TypeError, ValueError):
        return False


def _cache_nonquiescent_candidate_passes(
    candidate: object,
    *,
    microbatch: int,
    expected_run_contract: Mapping[str, object],
    total_memory: int,
    expected_parameter_count: int = FAST_BLT_ENTROPY_B4_COMPLETE_PARAMETER_TARGET,
) -> bool:
    """Authenticate bounded validation-cache failure without promoting it."""

    if not isinstance(candidate, Mapping):
        return False
    workload = candidate.get("measured_workload")
    evidence = candidate.get("cache_quiescence")
    if not isinstance(workload, Mapping):
        return False
    records = workload.get("updates")
    if not isinstance(records, list):
        return False
    required = required_headroom_bytes(total_memory)
    try:
        repetitions = int(workload["repetitions"])
        peak_reserved = int(candidate["cuda_peak_reserved_bytes"])
        return bool(
            candidate.get("status") == "validation_cache_nonquiescent"
            and candidate.get("eligible") is False
            and candidate.get("cache_phase") == "validation_measured"
            and candidate.get("microbatch") == microbatch
            and candidate.get("parameter_count") == expected_parameter_count
            and isinstance(candidate.get("error"), str)
            and bool(candidate["error"])
            and candidate.get("run_contract")
            == _json_native(dict(expected_run_contract))
            and candidate.get("run_contract_sha256")
            == canonical_json_sha256(expected_run_contract)
            and _workload_passes(
                workload,
                rows=FAST_BLT_DIFFUSION_VALIDATION_ROWS,
                repetitions=repetitions,
                kind="validation",
                expected_token_budget=int(
                    expected_run_contract["microbatch_token_budget"]
                ),
            )
            and _validation_cache_quiescence_passes(
                evidence,
                measured_records=records,
                measured_repetitions=repetitions,
                expected_quiescent=False,
            )
            and 0 <= int(candidate["cuda_peak_allocated_bytes"])
            <= peak_reserved
            <= total_memory
            and candidate.get("cuda_total_bytes") == total_memory
            and candidate.get("required_headroom_bytes") == required
            and candidate.get("intrinsic_reserved_headroom_bytes")
            == total_memory - peak_reserved
        )
    except (KeyError, TypeError, ValueError):
        return False


def _telemetry_error_candidate_passes(
    candidate: object,
    *,
    microbatch: int,
    expected_run_contract: Mapping[str, object],
    total_memory: int,
    expected_parameter_count: int = FAST_BLT_ENTROPY_B4_COMPLETE_PARAMETER_TARGET,
) -> bool:
    """Authenticate a bounded sampler failure without promoting it."""

    if not isinstance(candidate, Mapping):
        return False
    terminal = candidate.get("telemetry_terminal")
    if not isinstance(terminal, Mapping) or set(terminal) != {
        "schema",
        "query_timeout_seconds",
        "thread_join_timeout_seconds",
        "stop_requested",
        "thread_terminated",
        "clean_completion",
        "error_count",
        "errors",
        "errors_sha256",
    }:
        return False
    try:
        errors = tuple(terminal["errors"])
        for seq, error in enumerate(errors):
            if (
                not isinstance(error, Mapping)
                or set(error) != {"seq", "timestamp_ns", "kind", "returncode"}
                or error.get("seq") != seq
                or isinstance(error.get("timestamp_ns"), bool)
                or not isinstance(error.get("timestamp_ns"), int)
                or int(error["timestamp_ns"]) < 0
                or error.get("kind")
                not in {
                    "timeout",
                    "exception",
                    "nonzero_returncode",
                    "malformed_output",
                    "malformed_value",
                }
                or (
                    error.get("returncode") is not None
                    and (
                        isinstance(error.get("returncode"), bool)
                        or not isinstance(error.get("returncode"), int)
                    )
                )
            ):
                return False
        peak_reserved = int(candidate["cuda_peak_reserved_bytes"])
        dirty = not bool(terminal["thread_terminated"]) or bool(errors)
        return bool(
            candidate.get("status") == "telemetry_error"
            and candidate.get("eligible") is False
            and candidate.get("telemetry_phase") in FAST_BLT_FAILURE_PHASES
            and candidate.get("microbatch") == microbatch
            and candidate.get("parameter_count") == expected_parameter_count
            and isinstance(candidate.get("error"), str)
            and bool(candidate["error"])
            and candidate.get("run_contract")
            == _json_native(dict(expected_run_contract))
            and candidate.get("run_contract_sha256")
            == canonical_json_sha256(expected_run_contract)
            and terminal.get("schema") == FAST_BLT_TELEMETRY_TERMINAL_SCHEMA
            and terminal.get("query_timeout_seconds")
            == FAST_BLT_TELEMETRY_QUERY_TIMEOUT_SECONDS
            and terminal.get("thread_join_timeout_seconds")
            == FAST_BLT_TELEMETRY_THREAD_JOIN_TIMEOUT_SECONDS
            and terminal.get("stop_requested") is True
            and terminal.get("clean_completion") is False
            and terminal.get("error_count") == len(errors)
            and terminal.get("errors_sha256") == canonical_json_sha256(errors)
            and dirty
            and 0 <= int(candidate["cuda_peak_allocated_bytes"])
            <= peak_reserved
            <= total_memory
            and candidate.get("cuda_total_bytes") == total_memory
            and candidate.get("required_headroom_bytes")
            == required_headroom_bytes(total_memory)
            and candidate.get("intrinsic_reserved_headroom_bytes")
            == total_memory - peak_reserved
        )
    except (KeyError, TypeError, ValueError):
        return False


def validate_fast_blt_readiness(
    path: Path,
    *,
    source_sha256: str,
    dataset_payload_sha256: str,
    source_manifest_sha256: str,
    atomic_manifest_sha256: str,
    train_dataset_sha256: str,
    validation_dataset_sha256: str,
    patcher_sha256: str,
    max_patch_size: int,
    model_config: object,
    production_run_config: object,
    selected_run_config: object,
    geometry: FastBltGeometry,
    candidate_microbatches: Sequence[int],
    selected_microbatch: int,
    verify_runtime: bool = True,
) -> dict[str, object]:
    """Authenticate complete-cell evidence before production training."""

    report, report_sha256 = _read_report_once(path)
    preset = getattr(production_run_config, "preset", None)
    if preset not in FAST_BLT_ENTROPY_B4_COMPLETE_PRESETS:
        raise ValueError("readiness run is not a supported complete Fast-BLT preset")
    named_model = fast_blt_complete_model_config(preset)
    expected_parameter_count = named_model.production_parameter_target
    expected_model = _json_native(model_config)
    if expected_model != _json_native(named_model):
        raise ValueError("readiness model is not the named complete Fast-BLT cell")
    expected_production = run_contract_without_identity(production_run_config)
    expected_selected = run_contract_without_identity(selected_run_config)
    expected_common = common_run_contract(selected_run_config)
    candidates = tuple(int(value) for value in candidate_microbatches)
    if candidates != (32,) or selected_microbatch != 32:
        raise ValueError("complete-preset readiness is an exact microbatch-32 proof")
    gpu = report.get("gpu")
    results = report.get("results")
    if not isinstance(gpu, Mapping) or not isinstance(results, Mapping):
        raise ValueError("Fast-BLT readiness omitted GPU or candidate evidence")
    total_memory = int(gpu.get("total_memory_bytes", 0))
    required = required_headroom_bytes(total_memory)
    selected = report.get("selected_microbatch")

    from scripts.benchmark_byte_diffusion_fast_blt import (
        benchmark_harness_provenance,
        build_benchmark_run_config,
    )

    expected_candidate_runs = {
        value: run_contract_without_identity(
            build_benchmark_run_config(production_run_config, value)
        )
        for value in candidates
    }
    if expected_selected != expected_candidate_runs.get(selected_microbatch):
        raise ValueError("selected readiness run is not derived from the named preset")
    if expected_selected != expected_production:
        raise ValueError("readiness candidate must be the exact named production run")
    if {
        canonical_json_sha256(common_run_contract(contract))
        for contract in expected_candidate_runs.values()
    } != {canonical_json_sha256(expected_common)}:
        raise ValueError("candidate runs do not share one authenticated common contract")

    complete = (
        report.get("candidate_microbatches") == list(candidates)
        and set(results) == {str(value) for value in candidates}
    )
    recomputed: list[tuple[int, float]] = []
    if complete:
        for value in candidates:
            record = results.get(str(value))
            expected_candidate_run = expected_candidate_runs[value]
            if candidate_passes(
                record, microbatch=value,
                expected_run_contract=expected_candidate_run,
                total_memory=total_memory,
                expected_parameter_count=expected_parameter_count,
            ):
                assert isinstance(record, Mapping)
                training = record["training"]
                assert isinstance(training, Mapping)
                recomputed.append(
                    (
                        value,
                        float(training["production_schedule_update_ms"]),
                    )
                )
            elif not _oom_candidate_passes(
                record, microbatch=value,
                expected_run_contract=expected_candidate_run,
                total_memory=total_memory,
            ) and not _nonquiescent_candidate_passes(
                record,
                microbatch=value,
                expected_run_contract=expected_candidate_run,
                total_memory=total_memory,
                expected_parameter_count=expected_parameter_count,
            ) and not _cache_nonquiescent_candidate_passes(
                record,
                microbatch=value,
                expected_run_contract=expected_candidate_run,
                total_memory=total_memory,
                expected_parameter_count=expected_parameter_count,
            ) and not _telemetry_error_candidate_passes(
                record,
                microbatch=value,
                expected_run_contract=expected_candidate_run,
                total_memory=total_memory,
                expected_parameter_count=expected_parameter_count,
            ) and not _compile_error_candidate_passes(
                record,
                microbatch=value,
                expected_run_contract=expected_candidate_run,
                total_memory=total_memory,
                expected_parameter_count=expected_parameter_count,
            ):
                complete = False
                break
    recomputed_selected = min(
        recomputed, key=lambda item: (item[1], -item[0]), default=(None, 0.0)
    )[0]
    candidate = results.get(str(selected))
    dataset = report.get("dataset")
    expected_dataset = {
        "payload_sha256": dataset_payload_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "atomic_manifest_sha256": atomic_manifest_sha256,
        "train_dataset_sha256": train_dataset_sha256,
        "validation_dataset_sha256": validation_dataset_sha256,
        "validation_rows": FAST_BLT_VALIDATION_ROWS,
        "diffusion_validation_rows": FAST_BLT_DIFFUSION_VALIDATION_ROWS,
        "patching_policy": "causal_entropy_v1",
        "patcher_sha256": patcher_sha256,
        "max_patch_size": max_patch_size,
    }
    policy = {
        "fraction": HEADROOM_FRACTION,
        "minimum_gib": HEADROOM_MINIMUM_GIB,
        "required_bytes": required,
        "intrinsic_and_physical_required": True,
    }
    runtime = {
        "schema": FAST_BLT_RUNTIME_SCHEMA,
        "compiled": True,
        "compile_dynamic_shapes": True,
        "device_type": "cuda",
        "world_size": 1,
        "validation_model_is_separate": True,
        "allocator": fast_blt_allocator_contract(),
    }
    if (
        report.get("schema") != FAST_BLT_READINESS_SCHEMA
        or report.get("preset") != preset
        or report.get("source_sha256") != source_sha256
        or report.get("benchmark_harness_source") != _json_native(benchmark_harness_provenance())
        or dataset != expected_dataset
        or report.get("model_config") != expected_model
        or report.get("model_config_sha256") != canonical_json_sha256(expected_model)
        or report.get("production_run_contract") != expected_production
        or report.get("production_run_contract_sha256")
        != canonical_json_sha256(expected_production)
        or report.get("common_run_contract") != expected_common
        or report.get("common_run_contract_sha256") != canonical_json_sha256(expected_common)
        or report.get("geometry") != geometry.contract()
        or report.get("headroom_policy") != policy
        or report.get("sustained_gpu_policy") != SUSTAINED_GPU_POLICY
        or report.get("runtime") != runtime
        or not complete
        or selected != selected_microbatch
        or selected != recomputed_selected
        or not isinstance(candidate, Mapping)
        or candidate.get("run_contract") != expected_selected
        or not candidate_passes(
            candidate, microbatch=selected_microbatch,
            expected_run_contract=expected_selected, total_memory=total_memory,
            expected_parameter_count=expected_parameter_count,
        )
    ):
        raise ValueError("Fast-BLT readiness provenance, contract, or telemetry differs")

    if verify_runtime:
        import torch
        from .telemetry import nvidia_smi_selector

        require_fast_blt_allocator_environment()
        if not torch.cuda.is_available():
            raise ValueError("Fast-BLT readiness requires a CUDA runtime")
        device = torch.device("cuda", torch.cuda.current_device())
        if (
            gpu.get("name") != torch.cuda.get_device_name(device)
            or gpu.get("total_memory_bytes")
            != torch.cuda.get_device_properties(device).total_memory
            or gpu.get("torch") != torch.__version__
            or gpu.get("cuda") != torch.version.cuda
            or gpu.get("nvidia_smi_selector") != nvidia_smi_selector(device)
        ):
            raise ValueError("Fast-BLT readiness runtime differs from training")

    return {
        "path": str(path),
        "sha256": report_sha256,
        "benchmark_harness_sha256": report["benchmark_harness_source"]["sha256"],
        "selected_microbatch": selected,
        "validation_rows": FAST_BLT_VALIDATION_ROWS,
        "diffusion_validation_rows": FAST_BLT_DIFFUSION_VALIDATION_ROWS,
        "ar_validation_max_batch_size": FAST_BLT_AR_VALIDATION_BATCH,
        "diffusion_validation_max_batch_size": FAST_BLT_VALIDATION_BATCH,
        "geometry": geometry.contract(),
    }


__all__ = (
    "FAST_BLT_ALLOCATOR_ENV", "FAST_BLT_ALLOCATOR_SCHEMA",
    "FAST_BLT_ALLOCATOR_VALUE", "FAST_BLT_AR_VALIDATION_BATCH", "FAST_BLT_BLOCK_LENGTH",
    "FAST_BLT_DIFFUSION_VALIDATION_INDICES_SHA256",
    "FAST_BLT_DIFFUSION_VALIDATION_ROWS", "FAST_BLT_GLOBAL_BATCH",
    "FAST_BLT_GRAPH_QUIESCENCE_SCHEMA", "FAST_BLT_MAX_GRAPH_WARMUP_UPDATES",
    "FAST_BLT_ORIGIN_POLICY",
    "FAST_BLT_LEGACY_ALLOCATOR_ENV", "FAST_BLT_READINESS_SCHEMA",
    "FAST_BLT_ROW_LENGTH", "FAST_BLT_RUNTIME_SCHEMA",
    "FAST_BLT_TIMING_SCHEMA", "FAST_BLT_VALIDATION_BATCH",
    "FAST_BLT_VALIDATION_ROWS", "FAST_BLT_WORKLOAD_SCHEMA",
    "FastBltGeometry", "candidate_passes", "canonical_json_sha256",
    "common_run_contract", "required_headroom_bytes", "run_contract_without_identity",
    "fast_blt_allocator_contract", "require_fast_blt_allocator_environment",
    "validate_fast_blt_readiness",
)
