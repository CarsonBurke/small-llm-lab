"""Authenticated systems-readiness contracts for production 2k runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping


READINESS_SCHEMA = "byte_diffusion_architecture_readiness/v1"
DUO_DISTRIBUTED_READINESS_SCHEMA = "byte_duo_distributed_readiness/v1"
DUO_INFERENCE_READINESS_SCHEMA = "byte_duo_inference_readiness/v5"
DUO_CANONICAL_VALIDATION_CANVAS_LENGTH = 512
DUO_CANONICAL_VALIDATION_BRANCHES = 8
DUO_DATA_REQUIRED_BRANCH_BYTES = 4_096
DUO_DATA_BRANCH_SPAN_LENGTH = 512
DUO_PRODUCTION_TRAIN_GEOMETRIES = frozenset(((512, 8), (256, 15)))
SUSTAINED_GPU_POLICY = {
    # Power is an auxiliary guard against the historical idle/host-bound
    # failure, not a proxy for tensor-core occupancy. The 5090 evidence is
    # memory/attention-bound at 99% utilization and sustains 440-470 W; a
    # 500 W absolute floor incorrectly rejected both training and fully
    # batched validation. Keep a material floor while making throughput and
    # utilization the primary performance signals.
    "minimum_mean_power_w": 425.0,
    "minimum_p10_power_w": 400.0,
    "minimum_mean_utilization_percent": 90.0,
    "minimum_p10_utilization_percent": 80.0,
}
HEADROOM_FRACTION = 0.15
HEADROOM_MINIMUM_GIB = 4.0
MIN_TELEMETRY_SAMPLES = 3
GIB = 1 << 30


def duo_geometry_contract(canvas_length: int, branches: int) -> dict[str, object]:
    """Authenticate one named train geometry and the fixed comparison ledger.

    Training geometry is the ablated variable.  Validation and the underlying
    clean/data bank stay invariant so a 256-wide result remains directly
    comparable to the 512-wide reference rather than silently changing three
    experimental variables at once.
    """

    geometry = (int(canvas_length), int(branches))
    if geometry not in DUO_PRODUCTION_TRAIN_GEOMETRIES:
        supported = ", ".join(
            f"{canvas}x{branch_count}"
            for canvas, branch_count in sorted(DUO_PRODUCTION_TRAIN_GEOMETRIES)
        )
        raise ValueError(
            f"unsupported production Byte-Duo train geometry {geometry[0]}x"
            f"{geometry[1]}; expected one of {supported}"
        )
    return {
        "training_geometry": {
            "canvas_length": geometry[0],
            "branches": geometry[1],
        },
        "canonical_validation_geometry": {
            "canvas_length": DUO_CANONICAL_VALIDATION_CANVAS_LENGTH,
            "branches": DUO_CANONICAL_VALIDATION_BRANCHES,
        },
        "dataset_geometry": {
            "required_branch_bytes": DUO_DATA_REQUIRED_BRANCH_BYTES,
            "branch_span_length": DUO_DATA_BRANCH_SPAN_LENGTH,
        },
    }


def diagnostic_cadence_contract(
    *, log_every: int = 10, validation_every: int = 20
) -> dict[str, object]:
    """Describe which production updates materialize training diagnostics."""

    if log_every <= 0 or validation_every <= 0:
        raise ValueError("diagnostic cadence intervals must be positive")
    return {
        "schema": "byte_diffusion_diagnostic_cadence/v1",
        "readiness_update_class": "non_logging_non_validation_steady_state",
        "readiness_materializes_diagnostics": False,
        "production_materializes_on": [
            "log_every",
            "validation_every",
            "final_step",
        ],
        "log_every": log_every,
        "validation_every": validation_every,
    }


def materialize_training_diagnostics(
    step: int,
    total_steps: int,
    *,
    log_every: int,
    validation_every: int,
) -> bool:
    """Return whether an update belongs to a declared diagnostic cadence."""

    if step <= 0 or total_steps <= 0 or step > total_steps:
        raise ValueError("diagnostic step must lie in the positive training range")
    diagnostic_cadence_contract(
        log_every=log_every, validation_every=validation_every
    )
    return (
        step % log_every == 0
        or step % validation_every == 0
        or step == total_steps
    )


def _read_json_once(path: Path) -> tuple[dict[str, object], str]:
    payload = path.read_bytes()
    report = json.loads(payload)
    if not isinstance(report, dict):
        raise ValueError("readiness report must contain one JSON object")
    return report, hashlib.sha256(payload).hexdigest()


def _json_native(value: object) -> object:
    """Normalize tuples and other JSON containers before contract equality."""

    return json.loads(json.dumps(value, sort_keys=True))


def sha256_file(path: Path) -> str:
    """Hash a file for non-authentication callers.

    Readiness validators use :func:`_read_json_once` so parsing and evidence
    hashing always bind the same immutable byte payload.
    """

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_headroom(total_bytes: int) -> int:
    if total_bytes <= 0:
        raise ValueError("readiness report has invalid GPU memory")
    return max(
        math.ceil(total_bytes * HEADROOM_FRACTION),
        math.ceil(HEADROOM_MINIMUM_GIB * GIB),
    )


def _telemetry_passes(
    power: object, utilization: object
) -> bool:
    if not isinstance(power, Mapping) or not isinstance(utilization, Mapping):
        return False
    try:
        return (
            int(power["count"]) >= MIN_TELEMETRY_SAMPLES
            and int(utilization["count"]) >= MIN_TELEMETRY_SAMPLES
            and float(power["mean"])
            >= SUSTAINED_GPU_POLICY["minimum_mean_power_w"]
            and float(power["p10"])
            >= SUSTAINED_GPU_POLICY["minimum_p10_power_w"]
            and float(utilization["mean"])
            >= SUSTAINED_GPU_POLICY["minimum_mean_utilization_percent"]
            and float(utilization["p10"])
            >= SUSTAINED_GPU_POLICY["minimum_p10_utilization_percent"]
        )
    except (KeyError, TypeError, ValueError):
        return False


def _telemetry_present(power: object, utilization: object) -> bool:
    """Validate sampled latency telemetry without demanding saturation."""

    if not isinstance(power, Mapping) or not isinstance(utilization, Mapping):
        return False
    try:
        return (
            int(power["count"]) >= MIN_TELEMETRY_SAMPLES
            and int(utilization["count"]) >= MIN_TELEMETRY_SAMPLES
            and math.isfinite(float(power["mean"]))
            and math.isfinite(float(utilization["mean"]))
        )
    except (KeyError, TypeError, ValueError):
        return False


def _utilization_passes(power: object, utilization: object) -> bool:
    """Require a busy measured phase while treating watts as diagnostic."""

    if not _telemetry_present(power, utilization):
        return False
    assert isinstance(utilization, Mapping)
    try:
        return (
            float(utilization["mean"])
            >= SUSTAINED_GPU_POLICY["minimum_mean_utilization_percent"]
            and float(utilization["p10"])
            >= SUSTAINED_GPU_POLICY["minimum_p10_utilization_percent"]
        )
    except (KeyError, TypeError, ValueError):
        return False


def _graphs_pass(record: Mapping[str, object], *, prefix: str = "measured_") -> bool:
    return all(
        record.get(prefix + suffix) == 0
        for suffix in ("unique_graphs", "recompiles", "graph_breaks")
    )


def _distributed_phase_passes(
    phase: Mapping[str, object],
    *,
    total_memory: int,
    warmup_key: str,
    measured_key: str,
    minimum_warmup: int,
    minimum_measured: int,
    require_sustained_power: bool,
) -> bool:
    """Recompute one rank-local distributed systems gate."""

    required = _required_headroom(total_memory)
    try:
        peak_reserved = int(phase["cuda_peak_reserved_bytes"])
        observed_headroom = total_memory - peak_reserved
        return bool(
            int(phase[warmup_key]) >= minimum_warmup
            and int(phase[measured_key]) >= minimum_measured
            and math.isfinite(float(phase["elapsed_seconds"]))
            and float(phase["elapsed_seconds"]) > 0
            and peak_reserved >= 0
            and observed_headroom >= required
            and phase.get("required_headroom_bytes") == required
            and phase.get("cuda_reserved_headroom_bytes") == observed_headroom
            and int(phase["warmup_unique_graphs"]) >= 0
            and int(phase["warmup_recompiles"]) >= 0
            and phase.get("warmup_graph_breaks") == 0
            and _graphs_pass(phase)
            and (
                _telemetry_passes
                if require_sustained_power
                else _utilization_passes
            )(phase.get("power_w"), phase.get("gpu_utilization_percent"))
            and phase.get("eligible") is True
        )
    except (KeyError, TypeError, ValueError):
        return False


def _distributed_rank_passes(record: Mapping[str, object], *, rank: int) -> bool:
    """Authenticate fixed 8-way Byte-Duo geometry for one reported rank."""

    gpu = record.get("gpu")
    training = record.get("training")
    validation = record.get("validation")
    if not all(isinstance(item, Mapping) for item in (gpu, training, validation)):
        return False
    assert isinstance(gpu, Mapping)
    assert isinstance(training, Mapping)
    assert isinstance(validation, Mapping)
    try:
        total_memory = int(gpu["total_memory_bytes"])
        rotations = list(map(int, training["rotations"]))
        local_rows = list(map(int, training["local_rows"]))
        backward_calls = list(map(int, training["backward_calls"]))
        expected_rows = [
            32 if (rank - rotation) % 8 == 0 else 31
            for rotation in rotations
        ]
        geometry_ok = (
            record.get("rank") == rank
            and record.get("local_rank") == rank
            and gpu.get("rank") == rank
            and gpu.get("local_rank") == rank
            and gpu.get("device_index") == rank
            and "H100" in str(gpu.get("name", ""))
            and bool(gpu.get("nvidia_smi_selector"))
            and training.get("global_batch") == 249
            and training.get("microbatch") == 32
            and training.get("world_size") == 8
            and rotations
            == list(range(int(training["warmup_updates"]) + int(training["measured_updates"])))
            and local_rows == expected_rows
            and backward_calls == [1] * len(rotations)
            and training.get("global_denominator_allreduce") is True
            and training.get("clean_ar_global_denominator_allreduce") is True
            and training.get("ddp_world_size_loss_scale") == 8.0
            and len(training.get("measured_local_target_counts", ()))
            == int(training["measured_updates"])
            and len(training.get("measured_global_denominators", ()))
            == int(training["measured_updates"])
            and len(training.get("measured_local_clean_ar_target_counts", ()))
            == int(training["measured_updates"])
            and len(training.get("measured_global_clean_ar_denominators", ()))
            == int(training["measured_updates"])
            and validation.get("global_rows") == 256
            and validation.get("local_rows") == 32
            and len(validation.get("row_ids", ())) == 32
            and validation.get("batch_size") == 32
            and validation.get("world_size") == 8
            and validation.get("global_statistics_allreduce") is True
        )
    except (KeyError, TypeError, ValueError):
        return False
    training_ok = _distributed_phase_passes(
        training,
        total_memory=total_memory,
        warmup_key="warmup_updates",
        measured_key="measured_updates",
        minimum_warmup=8,
        minimum_measured=8,
        require_sustained_power=True,
    )
    validation_ok = _distributed_phase_passes(
        validation,
        total_memory=total_memory,
        warmup_key="warmup_repetitions",
        measured_key="measured_repetitions",
        minimum_warmup=1,
        minimum_measured=8,
        require_sustained_power=False,
    )
    return bool(geometry_ok and training_ok and validation_ok and record.get("eligible") is True)


def _distributed_aggregate(
    rank_records: list[Mapping[str, object]],
) -> dict[str, object]:
    """Recompute cluster throughput from the slowest participating rank."""

    def phase(record: Mapping[str, object], name: str) -> Mapping[str, object]:
        value = record[name]
        if not isinstance(value, Mapping):
            raise ValueError(f"distributed rank omitted {name} telemetry")
        return value

    training = [phase(record, "training") for record in rank_records]
    validation = [phase(record, "validation") for record in rank_records]
    training_elapsed = [float(item["elapsed_seconds"]) for item in training]
    validation_elapsed = [float(item["elapsed_seconds"]) for item in validation]
    measured_updates = int(training[0]["measured_updates"])
    validation_repetitions = int(validation[0]["measured_repetitions"])
    local_target_counts = [
        list(map(int, item["measured_local_target_counts"])) for item in training
    ]
    global_denominators = [
        list(map(int, item["measured_global_denominators"])) for item in training
    ]
    local_clean_ar_counts = [
        list(map(int, item["measured_local_clean_ar_target_counts"]))
        for item in training
    ]
    global_clean_ar_denominators = [
        list(map(int, item["measured_global_clean_ar_denominators"]))
        for item in training
    ]
    if any(len(counts) != measured_updates for counts in local_target_counts):
        raise ValueError("distributed rank target counts have the wrong extent")
    if any(len(counts) != measured_updates for counts in global_denominators):
        raise ValueError("distributed rank denominators have the wrong extent")
    if any(len(counts) != measured_updates for counts in local_clean_ar_counts):
        raise ValueError("distributed rank clean-AR counts have the wrong extent")
    if any(
        len(counts) != measured_updates for counts in global_clean_ar_denominators
    ):
        raise ValueError("distributed rank clean-AR denominators have the wrong extent")
    recomputed_global_targets = [
        sum(local_target_counts[rank][update] for rank in range(8))
        for update in range(measured_updates)
    ]
    if any(
        denominators != recomputed_global_targets
        for denominators in global_denominators
    ):
        raise ValueError("distributed global denominator does not equal local targets")
    recomputed_global_clean_ar_targets = [
        sum(local_clean_ar_counts[rank][update] for rank in range(8))
        for update in range(measured_updates)
    ]
    if any(
        denominators != recomputed_global_clean_ar_targets
        for denominators in global_clean_ar_denominators
    ):
        raise ValueError(
            "distributed global clean-AR denominator does not equal local targets"
        )
    slowest_training_rank = max(range(8), key=training_elapsed.__getitem__)
    slowest_validation_rank = max(range(8), key=validation_elapsed.__getitem__)
    slowest_training = training_elapsed[slowest_training_rank]
    slowest_validation = validation_elapsed[slowest_validation_rank]
    return {
        "slowest_training_rank": slowest_training_rank,
        "slowest_training_elapsed_seconds": slowest_training,
        "update_ms_slowest_rank": 1_000.0 * slowest_training / measured_updates,
        "global_rows_per_second": measured_updates * 249 / slowest_training,
        "measured_global_target_counts": recomputed_global_targets,
        "measured_global_targets": sum(recomputed_global_targets),
        "measured_global_clean_ar_target_counts": (
            recomputed_global_clean_ar_targets
        ),
        "measured_global_clean_ar_targets": sum(
            recomputed_global_clean_ar_targets
        ),
        "slowest_validation_rank": slowest_validation_rank,
        "slowest_validation_elapsed_seconds": slowest_validation,
        "validation_rows_per_second": (
            validation_repetitions * 256 / slowest_validation
        ),
        "all_ranks_eligible": all(bool(record.get("eligible")) for record in rank_records),
    }


def _candidate_tail_geometry_passes(
    candidate: Mapping[str, object],
    *,
    global_batch_size: int,
    microbatch_size: int,
) -> bool:
    """Bind a readiness record to its exact full-batch partition."""

    microsteps = math.ceil(global_batch_size / microbatch_size)
    tail = global_batch_size - (microsteps - 1) * microbatch_size
    return bool(
        candidate.get("microbatch") == microbatch_size
        and candidate.get("global_batch") == global_batch_size
        and candidate.get("microsteps_per_update") == microsteps
        and candidate.get("tail_microbatch") == tail
        and candidate.get("tail_exercised") is (tail != microbatch_size)
    )


def _candidate_passes(
    candidate: Mapping[str, object],
    *,
    global_batch_size: int,
    microbatch_size: int,
    validation_batch_size: int,
    total_memory: int,
) -> bool:
    required = _required_headroom(total_memory)
    validation = candidate.get("validation")
    if not isinstance(validation, Mapping):
        return False
    try:
        peak_reserved = int(candidate["cuda_peak_reserved_bytes"])
        observed_headroom = total_memory - peak_reserved
        validation_peak_reserved = int(validation["cuda_peak_reserved_bytes"])
        validation_headroom = total_memory - validation_peak_reserved
        training_ok = (
            candidate.get("status") == "ok"
            and _candidate_tail_geometry_passes(
                candidate,
                global_batch_size=global_batch_size,
                microbatch_size=microbatch_size,
            )
            and candidate.get("row_length_bytes") == 8_192
            and int(candidate["warmup_updates"]) >= 2
            and int(candidate["measured_updates"]) >= 4
            and math.isfinite(float(candidate["update_ms"]))
            and float(candidate["update_ms"]) > 0
            and candidate.get("cuda_total_bytes") == total_memory
            and candidate.get("required_headroom_bytes") == required
            and peak_reserved >= 0
            and observed_headroom >= required
            and candidate.get("cuda_reserved_headroom_bytes") == observed_headroom
            and _graphs_pass(candidate)
            and candidate.get("warmup_graph_breaks") == 0
            and _utilization_passes(
                candidate.get("power_w"),
                candidate.get("gpu_utilization_percent"),
            )
        )
        validation_ok = (
            validation.get("rows") == 256
            and validation.get("batch_size") == validation_batch_size
            and math.isfinite(float(validation["elapsed_seconds"]))
            and float(validation["elapsed_seconds"]) > 0
            and validation_peak_reserved >= 0
            and validation_headroom >= required
            and validation.get("cuda_reserved_headroom_bytes") == validation_headroom
            and _graphs_pass(validation)
            and validation.get("warmup_graph_breaks") == 0
            and _utilization_passes(
                validation.get("power_w"),
                validation.get("gpu_utilization_percent"),
            )
        )
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        training_ok
        and validation_ok
        and candidate.get("eligible") is True
        and validation.get("eligible") is True
    )


def validate_architecture_readiness(
    path: Path,
    *,
    architecture: str,
    source_sha256: str,
    dataset_payload_sha256: str,
    global_batch_size: int,
    microbatch_size: int,
    model_config: Mapping[str, object],
    workload: Mapping[str, object],
    validation_batch_size: int,
    runtime: Mapping[str, object],
    required_microbatches: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Recompute every gate from raw telemetry and bind the current harness."""

    report, report_sha256 = _read_json_once(path)
    from scripts.benchmark_byte_diffusion_architectures import (
        benchmark_harness_provenance,
    )

    selected_validation = report.get("selected_validation_batch_size")
    required_validation = validation_batch_size
    gpu = report.get("gpu")
    results = report.get("results")
    selected = report.get("selected_microbatch")
    candidate = (
        results.get(str(selected)) if isinstance(results, Mapping) else None
    )
    candidates = report.get("candidate_microbatches")
    candidate_records_complete = False
    recomputed_eligible: list[tuple[int, float]] = []
    if (
        isinstance(candidates, list)
        and candidates == sorted(set(candidates))
        and isinstance(results, Mapping)
        and isinstance(gpu, Mapping)
        and "total_memory_bytes" in gpu
    ):
        expected_keys = {str(microbatch) for microbatch in candidates}
        candidate_records_complete = set(results) == expected_keys and all(
            isinstance(results.get(str(microbatch)), Mapping)
            and _candidate_tail_geometry_passes(
                results[str(microbatch)],
                global_batch_size=global_batch_size,
                microbatch_size=int(microbatch),
            )
            for microbatch in candidates
        )
        for microbatch in candidates:
            record = results.get(str(microbatch))
            if isinstance(record, Mapping) and _candidate_passes(
                record,
                global_batch_size=global_batch_size,
                microbatch_size=int(microbatch),
                validation_batch_size=required_validation,
                total_memory=int(gpu["total_memory_bytes"]),
            ):
                recomputed_eligible.append(
                    (int(microbatch), float(record["update_ms"]))
                )
    recomputed_selected = min(
        recomputed_eligible,
        key=lambda item: (item[1], -item[0]),
        default=(None, 0.0),
    )[0]
    if (
        report.get("schema") != READINESS_SCHEMA
        or report.get("architecture") != architecture
        or not isinstance(report.get("source"), Mapping)
        or report["source"].get("sha256") != source_sha256
        or report.get("benchmark_harness_source")
        != _json_native(benchmark_harness_provenance())
        or report.get("dataset_payload_sha256") != dataset_payload_sha256
        or report.get("model_config") != _json_native(dict(model_config))
        or report.get("workload") != _json_native(dict(workload))
        or report.get("row_length_bytes") != 8_192
        or report.get("global_batch") != global_batch_size
        or not candidate_records_complete
        or (
            required_microbatches is not None
            and candidates != list(required_microbatches)
        )
        or report.get("sustained_gpu_policy") != SUSTAINED_GPU_POLICY
        or report.get("headroom_policy")
        != {
            "fraction": HEADROOM_FRACTION,
            "minimum_gib": HEADROOM_MINIMUM_GIB,
            "required_bytes": (
                _required_headroom(int(gpu["total_memory_bytes"]))
                if isinstance(gpu, Mapping) and "total_memory_bytes" in gpu
                else None
            ),
        }
        or report.get("runtime") != _json_native(dict(runtime))
        or selected != microbatch_size
        or selected != recomputed_selected
        or selected_validation != required_validation
        or not isinstance(gpu, Mapping)
        or not isinstance(candidate, Mapping)
        or not _candidate_passes(
            candidate,
            global_batch_size=global_batch_size,
            microbatch_size=microbatch_size,
            validation_batch_size=required_validation,
            total_memory=int(gpu["total_memory_bytes"]),
        )
    ):
        raise ValueError(
            f"{architecture} readiness telemetry, provenance, or geometry differs"
        )
    import torch
    from .telemetry import nvidia_smi_selector

    if (
        runtime.get("device_type") != "cuda"
        or not torch.cuda.is_available()
        or gpu.get("name") != torch.cuda.get_device_name(torch.cuda.current_device())
        or gpu.get("total_memory_bytes")
        != torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        or gpu.get("torch") != torch.__version__
        or gpu.get("cuda") != torch.version.cuda
        or gpu.get("nvidia_smi_selector")
        != nvidia_smi_selector(torch.device("cuda", torch.cuda.current_device()))
    ):
        raise ValueError(f"{architecture} readiness runtime differs from this process")
    return {
        "path": str(path),
        "sha256": report_sha256,
        "benchmark_harness_sha256": report["benchmark_harness_source"]["sha256"],
        "selected_microbatch": selected,
        "selected_validation_batch_size": selected_validation,
    }


def validate_duo_distributed_readiness(
    path: Path,
    *,
    source_sha256: str,
    dataset_payload_sha256: str,
    model_config: Mapping[str, object],
    parameter_count: int,
    workload: Mapping[str, object],
    rank: int,
) -> dict[str, object]:
    """Authenticate the dedicated eight-H100 Byte-Duo readiness report.

    This validator is intentionally callable by every training rank.  In
    addition to recomputing cluster-wide gates, it binds the calling rank to
    the exact GPU identity recorded for that rank by the readiness torchrun.
    """

    report, report_sha256 = _read_json_once(path)
    from scripts.benchmark_byte_duo_distributed import benchmark_source_provenance
    from scripts.train_byte_duo import source_provenance

    import torch
    import torch.distributed as dist
    from .telemetry import nvidia_smi_selector

    rank_records = report.get("ranks")
    if not isinstance(rank_records, list) or len(rank_records) != 8:
        raise ValueError("Byte-Duo distributed readiness omitted eight rank records")
    if not all(isinstance(record, Mapping) for record in rank_records):
        raise ValueError("Byte-Duo distributed readiness rank records are invalid")
    typed_records: list[Mapping[str, object]] = list(rank_records)
    if [record.get("rank") for record in typed_records] != list(range(8)):
        raise ValueError("Byte-Duo distributed readiness ranks are incomplete")
    recomputed_aggregate = _distributed_aggregate(typed_records)
    measurement_shapes: set[tuple[int, int, int, int]] = set()
    selectors: set[object] = set()
    validation_row_ids: list[int] = []
    for record in typed_records:
        gpu = record.get("gpu")
        training = record.get("training")
        validation = record.get("validation")
        if not all(isinstance(item, Mapping) for item in (gpu, training, validation)):
            raise ValueError("Byte-Duo distributed readiness phases are invalid")
        assert isinstance(gpu, Mapping)
        assert isinstance(training, Mapping)
        assert isinstance(validation, Mapping)
        selectors.add(gpu.get("nvidia_smi_selector"))
        validation_row_ids.extend(map(int, validation["row_ids"]))
        measurement_shapes.add(
            (
                int(training["warmup_updates"]),
                int(training["measured_updates"]),
                int(validation["warmup_repetitions"]),
                int(validation["measured_repetitions"]),
            )
        )
    current_source = source_provenance()
    fixed_geometry = {
        "world_size": 8,
        "global_batch": 249,
        "microbatch": 32,
        "base_local_rows": [32, 31, 31, 31, 31, 31, 31, 31],
        "remainder_rotation": "update_index_modulo_world_size",
        "backward_calls_per_rank_update": 1,
        "validation_global_rows": 256,
        "validation_local_rows": 32,
        "validation_batch_size": 32,
    }
    root_ok = (
        report.get("schema") == DUO_DISTRIBUTED_READINESS_SCHEMA
        and report.get("architecture") == "duo"
        and report.get("evidence_kind")
        == "exact_distributed_training_and_validation_systems_benchmark"
        and current_source.get("sha256") == source_sha256
        and report.get("recipe_source") == _json_native(current_source)
        and report.get("benchmark_source")
        == _json_native(benchmark_source_provenance())
        and report.get("dataset_payload_sha256") == dataset_payload_sha256
        and report.get("model_config") == _json_native(dict(model_config))
        and report.get("parameter_count") == parameter_count
        and report.get("workload") == _json_native(dict(workload))
        and report.get("geometry") == fixed_geometry
        and report.get("runtime")
        == {
            "compiled": True,
            "distributed_backend": "nccl",
            "launcher": "torchrun",
            "model_wrapper": "DDP(torch.compile(DuoModel))",
            "world_size": 8,
        }
        and report.get("sustained_gpu_policy") == SUSTAINED_GPU_POLICY
        and report.get("headroom_policy")
        == {"fraction": HEADROOM_FRACTION, "minimum_gib": HEADROOM_MINIMUM_GIB}
        and report.get("aggregate") == recomputed_aggregate
        and recomputed_aggregate["all_ranks_eligible"] is True
        and report.get("eligible") is True
        and len(measurement_shapes) == 1
        and len(selectors) == 8
        and sorted(validation_row_ids) == list(range(256))
        and all(
            _distributed_rank_passes(record, rank=rank)
            for rank, record in enumerate(typed_records)
        )
    )
    if not root_ok:
        raise ValueError(
            "Byte-Duo distributed readiness telemetry, provenance, or geometry differs"
        )

    backend = dist.get_backend() if dist.is_available() and dist.is_initialized() else None
    if (
        not torch.cuda.is_available()
        or backend != "nccl"
        or dist.get_world_size() != 8
        or not 0 <= rank < 8
        or dist.get_rank() != rank
    ):
        raise ValueError("Byte-Duo distributed readiness requires the live 8-rank NCCL run")
    calling_rank = rank
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    device_index = torch.cuda.current_device()
    record = typed_records[calling_rank]
    gpu = record["gpu"]
    assert isinstance(gpu, Mapping)
    properties = torch.cuda.get_device_properties(device_index)
    if (
        record.get("local_rank") != local_rank
        or gpu.get("local_rank") != local_rank
        or gpu.get("device_index") != device_index
        or gpu.get("name") != torch.cuda.get_device_name(device_index)
        or gpu.get("total_memory_bytes") != properties.total_memory
        or gpu.get("torch") != torch.__version__
        or gpu.get("cuda") != torch.version.cuda
        or gpu.get("nvidia_smi_selector")
        != nvidia_smi_selector(torch.device("cuda", device_index))
    ):
        raise ValueError("Byte-Duo distributed readiness GPU differs for this rank")
    return {
        "path": str(path),
        "sha256": report_sha256,
        "benchmark_sha256": report["benchmark_source"]["sha256"],
        "world_size": 8,
        "rank": calling_rank,
        "microbatch": 32,
        "validation_batch_size": 32,
        "update_ms_slowest_rank": recomputed_aggregate["update_ms_slowest_rank"],
    }


def validate_duo_inference_readiness(
    path: Path,
    *,
    source_sha256: str,
    canvas_length: int,
    model_config: Mapping[str, object],
    parameter_count: int,
    branches: int = 8,
    dataset_payload_sha256: str | None = None,
    dataset_patching: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Authenticate Byte-Duo inference telemetry and implementation closure."""

    report, report_sha256 = _read_json_once(path)
    from scripts.benchmark_byte_duo_inference import benchmark_source_provenance

    dynamo = report.get("torch_dynamo")
    phases = report.get("phases")
    gpu = report.get("gpu")
    if not isinstance(gpu, Mapping) or "total_memory_bytes" not in gpu:
        raise ValueError("Byte-Duo inference readiness omitted GPU geometry")
    total_memory = int(gpu["total_memory_bytes"])
    required = _required_headroom(total_memory)
    peak_reserved = int(report.get("cuda_peak_reserved_bytes", -1))
    observed_headroom = total_memory - peak_reserved
    geometry = duo_geometry_contract(canvas_length, branches)
    full_resolution = (
        model_config.get("duo_mutable_topology") == "full_resolution_decoder"
    )
    expected_canvases = [
        math.ceil(
            (
                DUO_CANONICAL_VALIDATION_CANVAS_LENGTH
                + (0 if full_resolution else phase)
            )
            / canvas_length
        )
        for phase in range(4)
    ]
    valid_phases = (
        isinstance(phases, list)
        and [record.get("phase") for record in phases]
        == [0, 1, 2, 3]
        and all(
            record.get("requested_atoms")
            == DUO_CANONICAL_VALIDATION_CANVAS_LENGTH
            for record in phases
        )
        and [record.get("canvases") for record in phases] == expected_canvases
    )
    eligible = (
        report.get("schema") == DUO_INFERENCE_READINESS_SCHEMA
        and isinstance(report.get("recipe_source"), Mapping)
        and report["recipe_source"].get("sha256") == source_sha256
        and report.get("benchmark_source")
        == _json_native(benchmark_source_provenance())
        and report.get("phase_coverage") == [0, 1, 2, 3]
        and report.get("canvas_length") == canvas_length
        and report.get("branches") == branches
        and report.get("requested_atoms_per_trajectory")
        == DUO_CANONICAL_VALIDATION_CANVAS_LENGTH
        and all(report.get(key) == value for key, value in geometry.items())
        and report.get("model_config") == _json_native(model_config)
        and report.get("parameter_count") == parameter_count
        and (
            dataset_payload_sha256 is None
            or report.get("dataset_payload_sha256") == dataset_payload_sha256
        )
        and (
            dataset_patching is None
            or report.get("dataset_patching")
            == _json_native(dict(dataset_patching))
        )
        and report.get("serving_origin_policy")
        == (
            "exact_prompt_length"
            if full_resolution
            else "floor_to_patch_and_carry_clean_phase"
        )
        and report.get("posterior_backend") == "triton"
        and report.get("semantic_generation") is False
        and report.get("sustained_gpu_policy") == SUSTAINED_GPU_POLICY
        and report.get("required_headroom_bytes") == required
        and peak_reserved >= 0
        and observed_headroom >= required
        and report.get("cuda_reserved_headroom_bytes") == observed_headroom
        and report.get("runtime") == {"compiled": True, "world_size": 1}
        and report.get("batch_size") == 8
        and report.get("base_prompt_length") == 2_048
        and report.get("diffusion_steps") == 8
        and report.get("repetitions") == 12
        and isinstance(dynamo, Mapping)
        and dynamo.get("warmup_graph_breaks") == 0
        and _graphs_pass(dynamo)
        and _telemetry_present(
            report.get("power_w"), report.get("gpu_utilization_percent")
        )
        and math.isfinite(float(report.get("measured_seconds", math.nan)))
        and float(report.get("measured_seconds", 0.0)) > 0
        and valid_phases
        and report.get("eligible") is True
    )
    if not eligible:
        raise ValueError("Byte-Duo inference readiness is incomplete or differs")
    import torch
    from .telemetry import nvidia_smi_selector

    if (
        not torch.cuda.is_available()
        or gpu.get("name") != torch.cuda.get_device_name(torch.cuda.current_device())
        or gpu.get("total_memory_bytes")
        != torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        or gpu.get("torch") != torch.__version__
        or gpu.get("cuda") != torch.version.cuda
        or gpu.get("nvidia_smi_selector")
        != nvidia_smi_selector(torch.device("cuda", torch.cuda.current_device()))
    ):
        raise ValueError("Byte-Duo inference readiness runtime differs")
    return {
        "path": str(path),
        "sha256": report_sha256,
        "benchmark_sha256": report["benchmark_source"]["sha256"],
        "batch_size": report.get("batch_size"),
        "diffusion_steps": report.get("diffusion_steps"),
    }


__all__ = (
    "DUO_CANONICAL_VALIDATION_BRANCHES",
    "DUO_CANONICAL_VALIDATION_CANVAS_LENGTH",
    "DUO_DATA_BRANCH_SPAN_LENGTH",
    "DUO_DATA_REQUIRED_BRANCH_BYTES",
    "DUO_DISTRIBUTED_READINESS_SCHEMA",
    "DUO_INFERENCE_READINESS_SCHEMA",
    "DUO_PRODUCTION_TRAIN_GEOMETRIES",
    "HEADROOM_FRACTION",
    "HEADROOM_MINIMUM_GIB",
    "MIN_TELEMETRY_SAMPLES",
    "READINESS_SCHEMA",
    "SUSTAINED_GPU_POLICY",
    "diagnostic_cadence_contract",
    "duo_geometry_contract",
    "materialize_training_diagnostics",
    "sha256_file",
    "validate_architecture_readiness",
    "validate_duo_distributed_readiness",
    "validate_duo_inference_readiness",
)
