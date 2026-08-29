from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import inspect
import json
import math
import textwrap
import threading
from types import SimpleNamespace

import pytest
import scripts.benchmark_byte_diffusion_fast_blt as benchmark_fast_blt

from pretraining.byte_diffusion.config import (
    FAST_BLT_ENTROPY_B4_COMPLETE_PAPER_RATIO_PRESET,
    FAST_BLT_ENTROPY_B4_COMPLETE_PARAMETER_TARGET,
    FAST_BLT_ENTROPY_B4_COMPLETE_PRESET,
    ByteDiffusionConfig,
    fast_blt_complete_model_config,
)
from pretraining.byte_diffusion.readiness import SUSTAINED_GPU_POLICY
from pretraining.byte_diffusion.readiness_fast_blt import (
    FAST_BLT_ALLOCATOR_ENV,
    FAST_BLT_ALLOCATOR_VALUE,
    FAST_BLT_GEOMETRY_SCHEMA,
    FAST_BLT_DIFFUSION_VALIDATION_INDICES_SHA256,
    FAST_BLT_GRAPH_QUIESCENCE_SCHEMA,
    FAST_BLT_MAX_GRAPH_WARMUP_UPDATES,
    FAST_BLT_READINESS_SCHEMA,
    FAST_BLT_RUNTIME_SCHEMA,
    FAST_BLT_TIMING_SCHEMA,
    FAST_BLT_TELEMETRY_LIFECYCLE_SCHEMA,
    FAST_BLT_TELEMETRY_TERMINAL_SCHEMA,
    FAST_BLT_VALIDATION_ROWS,
    FAST_BLT_LEGACY_ALLOCATOR_ENV,
    FAST_BLT_WORKLOAD_SCHEMA,
    FastBltGeometry,
    _compile_error_candidate_passes,
    _cache_nonquiescent_candidate_passes,
    _nonquiescent_candidate_passes,
    _oom_candidate_passes,
    _telemetry_domains_pass,
    _telemetry_error_candidate_passes,
    candidate_passes,
    canonical_json_sha256,
    common_run_contract,
    fast_blt_allocator_contract,
    require_fast_blt_allocator_environment,
    required_headroom_bytes,
    run_contract_without_identity,
    validate_fast_blt_readiness,
)
from pretraining.byte_diffusion.training import TrainingRunConfig
from scripts.benchmark_byte_diffusion_fast_blt import (
    _Telemetry,
    _attach_update_timing,
    _benchmark_candidate,
    _diffusion_validation_indices,
    _graph_delta,
    _is_torch_compile_error,
    _materialized_update_count,
    _require_unchanged_harness,
    benchmark_harness_provenance,
    build_benchmark_run_config,
)


TOTAL_MEMORY = 32 * (1 << 30)


def _summary(samples: list[float]) -> dict[str, object]:
    ordered = sorted(samples)
    rank = 0.1 * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    p10 = ordered[lower] + (rank - lower) * (
        ordered[upper] - ordered[lower]
    )
    return {
        "count": len(samples),
        "mean": sum(samples) / len(samples),
        "p10": p10,
        "minimum": min(samples),
        "maximum": max(samples),
        "samples": samples,
        "samples_sha256": canonical_json_sha256(samples),
    }


def _compile_error_candidate(
    contract: dict[str, object],
    parameter_count: int = FAST_BLT_ENTROPY_B4_COMPLETE_PARAMETER_TARGET,
) -> dict[str, object]:
    required = required_headroom_bytes(TOTAL_MEMORY)
    error = "CantSplit: symbolic output extent is not divisible"
    error_repr = f"InductorError({error!r})"
    traceback_text = (
        "Traceback (most recent call last):\n"
        "torch._inductor.exc.InductorError: " + error
    )
    return {
        "status": "compile_error",
        "eligible": False,
        "compile_phase": "training_diagnostic_warmup",
        "error_type": "torch._inductor.exc.InductorError",
        "error": error,
        "error_repr": error_repr,
        "error_sha256": hashlib.sha256(error.encode("utf-8")).hexdigest(),
        "error_repr_sha256": hashlib.sha256(
            error_repr.encode("utf-8")
        ).hexdigest(),
        "traceback": traceback_text,
        "traceback_sha256": hashlib.sha256(
            traceback_text.encode("utf-8")
        ).hexdigest(),
        "microbatch": 32,
        "run_contract": contract,
        "run_contract_sha256": canonical_json_sha256(contract),
        "parameter_count": parameter_count,
        "cuda_peak_allocated_bytes": 16 * (1 << 30),
        "cuda_peak_reserved_bytes": 20 * (1 << 30),
        "cuda_total_bytes": TOTAL_MEMORY,
        "intrinsic_reserved_headroom_bytes": 12 * (1 << 30),
        "required_headroom_bytes": required,
        "instantaneous_physical_free_bytes": 10 * (1 << 30),
        "instantaneous_physical_total_bytes": TOTAL_MEMORY,
    }


def _validation_cache_error_candidate(
    contract: dict[str, object],
    parameter_count: int = FAST_BLT_ENTROPY_B4_COMPLETE_PARAMETER_TARGET,
) -> dict[str, object]:
    workload = _workload(rows=256, repetitions=1, validation=True)
    record = workload["updates"][0]
    record["first_batch_cache_misses"] = 1
    record["first_batch_waits"] = 1
    record["cache_misses"] = 1
    workload["updates_sha256"] = canonical_json_sha256(workload["updates"])
    cache_record = {
        name: record[name]
        for name in (
            "schedule_cache_misses",
            "corruption_plan_cache_misses",
            "ragged_mask_metadata_cache_misses",
            "first_batch_cache_misses",
            "first_batch_waits",
        )
    }
    return {
        "status": "validation_cache_nonquiescent",
        "eligible": False,
        "cache_phase": "validation_measured",
        "error": "validation staging caches were not quiescent",
        "microbatch": 32,
        "run_contract": contract,
        "run_contract_sha256": canonical_json_sha256(contract),
        "parameter_count": parameter_count,
        "cache_quiescence": {
            "measured_cache_misses": 1,
            "measured_repetitions": 1,
            "quiescent": False,
            "measured_first_batch_waits": 1,
            "records_sha256": canonical_json_sha256((cache_record,)),
        },
        "measured_workload": workload,
        "cuda_peak_allocated_bytes": 8 * (1 << 30),
        "cuda_peak_reserved_bytes": 20 * (1 << 30),
        "cuda_total_bytes": TOTAL_MEMORY,
        "intrinsic_reserved_headroom_bytes": 12 * (1 << 30),
        "required_headroom_bytes": required_headroom_bytes(TOTAL_MEMORY),
    }


def _telemetry_error_candidate(
    contract: dict[str, object],
    parameter_count: int = FAST_BLT_ENTROPY_B4_COMPLETE_PARAMETER_TARGET,
) -> dict[str, object]:
    errors = (
        {
            "seq": 0,
            "timestamp_ns": 1_234_567,
            "kind": "timeout",
            "returncode": None,
        },
    )
    return {
        "status": "telemetry_error",
        "eligible": False,
        "telemetry_phase": "validation_measured",
        "error": "telemetry sampler did not terminate cleanly",
        "microbatch": 32,
        "run_contract": contract,
        "run_contract_sha256": canonical_json_sha256(contract),
        "parameter_count": parameter_count,
        "telemetry_terminal": {
            "schema": FAST_BLT_TELEMETRY_TERMINAL_SCHEMA,
            "query_timeout_seconds": 2.0,
            "thread_join_timeout_seconds": 3.0,
            "stop_requested": True,
            "thread_terminated": True,
            "clean_completion": False,
            "error_count": 1,
            "errors": errors,
            "errors_sha256": canonical_json_sha256(errors),
        },
        "cuda_peak_allocated_bytes": 8 * (1 << 30),
        "cuda_peak_reserved_bytes": 20 * (1 << 30),
        "cuda_total_bytes": TOTAL_MEMORY,
        "intrinsic_reserved_headroom_bytes": 12 * (1 << 30),
        "required_headroom_bytes": required_headroom_bytes(TOTAL_MEMORY),
    }


def _record(
    *, rows: int, validation: bool, diagnostic: bool = False
) -> dict[str, object]:
    origins = rows * 2_000
    microsteps = 20
    base, remainder = divmod(rows, microsteps)
    group_sizes = [base + (index < remainder) for index in range(microsteps)]
    boundaries = [0]
    for size in group_sizes:
        boundaries.append(boundaries[-1] + size)
    per_row_physical = 8_192 + 2_000 * 4
    group_physical = [size * per_row_physical for size in group_sizes]
    ordered_row_origins = [2_000] * rows
    group_origins = [size * 2_000 for size in group_sizes]
    group_row_hashes = [f"{index + 1:064x}" for index in range(microsteps)]
    max_microbatch = max(group_sizes)
    max_physical = max(group_physical)
    record: dict[str, object] = {
        "rows": rows,
        "origin_count": origins,
        "planned_microsteps": microsteps,
        "planned_min_microbatch": min(group_sizes),
        "planned_max_microbatch": max_microbatch,
        "max_row_origin_count": 2_100,
        "total_physical_positions": rows * 8_192 + origins * 4,
        "planned_max_physical_positions": max_physical,
        "planned_group_row_cu_seqlens": boundaries,
        "planned_group_row_cu_seqlens_sha256": canonical_json_sha256(
            boundaries
        ),
        "planned_group_physical_positions": group_physical,
        "planned_group_physical_positions_sha256": canonical_json_sha256(
            group_physical
        ),
        "ordered_row_origin_counts": ordered_row_origins,
        "ordered_row_origin_counts_sha256": canonical_json_sha256(
            ordered_row_origins
        ),
        "planned_group_origin_counts": group_origins,
        "planned_group_origin_counts_sha256": canonical_json_sha256(
            group_origins
        ),
        "planned_group_row_indices_sha256": group_row_hashes,
        "planned_group_row_indices_sha256_sha256": canonical_json_sha256(
            group_row_hashes
        ),
        "physical_token_budget": 212_992,
        "row_indices_sha256": (
            FAST_BLT_DIFFUSION_VALIDATION_INDICES_SHA256
            if validation
            else "b" * 64
        ),
        "row_origin_counts_sha256": "a" * 64,
    }
    if validation or diagnostic:
        record.update(
            active_bytes=origins * 2,
            branch_blocks=origins,
            branch_atoms=origins * 4 - rows,
            microsteps=microsteps,
            max_microbatch=max_microbatch,
            max_physical_positions=max_physical,
            block_mask_build_ms=None if validation else 12.5,
        )
        if validation:
            record.update(
                schedule_cache_misses=0,
                corruption_plan_cache_misses=0,
                ragged_mask_metadata_cache_misses=0,
                first_batch_cache_misses=0,
                first_batch_waits=0,
                cache_misses=0,
            )
    if not validation:
        record.update(
            materialize_metrics=diagnostic,
            **{
                (
                    "host_synchronized_call_ms"
                    if diagnostic
                    else "host_enqueue_ms"
                ): 250.0
            },
            cuda_event_ms=245.0,
            actual_step=1,
            actual_rows=rows,
            actual_microsteps=microsteps,
            actual_group_row_cu_seqlens=boundaries,
            actual_group_physical_positions=group_physical,
            actual_max_microbatch=max_microbatch,
            actual_max_physical_positions=max_physical,
        )
    return record


def _workload(
    *, rows: int, repetitions: int, validation: bool,
    diagnostic: bool = False, start_step: int = 1,
) -> dict[str, object]:
    records = []
    for index in range(repetitions):
        record = _record(
            rows=rows, validation=validation, diagnostic=diagnostic
        )
        if not validation:
            record["actual_step"] = start_step + index
        records.append(record)
    return {
        "schema": FAST_BLT_WORKLOAD_SCHEMA,
        "origin_policy": "all_entropy_patch_starts",
        "block_length": 4,
        "rows_per_update": rows,
        "batching_policy": "token_budgeted_ragged",
        "repetitions": repetitions,
        "updates": records,
        "updates_sha256": canonical_json_sha256(records),
    }


def _graph_quiescence(
    records: list[dict[str, object]], *, diagnostic: bool
) -> dict[str, object]:
    observations = []
    for index, record in enumerate(records, start=1):
        observations.append(
            {
                "update": index,
                "materialize_metrics": diagnostic,
                "representative_non_singleton": True,
                "planned_min_microbatch": record["planned_min_microbatch"],
                "planned_max_microbatch": record["planned_max_microbatch"],
                "new_unique_graphs": 3 if index == 1 else 0,
                "new_recompiles": 2 if index == 1 else 0,
                "new_graph_breaks": 0,
            }
        )
    return {
        "schema": FAST_BLT_GRAPH_QUIESCENCE_SCHEMA,
        "achieved": True,
        "materialize_metrics": diagnostic,
        "minimum_updates": 2,
        "maximum_updates": FAST_BLT_MAX_GRAPH_WARMUP_UPDATES,
        "observed_updates": len(observations),
        "representative_non_singleton": True,
        "total_unique_graphs": 3,
        "total_recompiles": 2,
        "total_graph_breaks": 0,
        "observations": observations,
        "observations_sha256": canonical_json_sha256(observations),
    }


def _phase(*, validation: bool) -> dict[str, object]:
    required = required_headroom_bytes(TOTAL_MEMORY)
    warmup, measured = ((1, 2) if validation else (2, 4))
    warmup_workload = _workload(
        rows=256 if validation else 249,
        repetitions=warmup,
        validation=validation,
        start_step=3 if not validation else 1,
    )
    measured_workload = _workload(
        rows=256 if validation else 249,
        repetitions=measured,
        validation=validation,
        start_step=5 if not validation else 1,
    )
    free_samples = [8 * (1 << 30)] * 4
    used_samples = [22 * (1 << 30)] * 4
    power_samples = [480.0, 500.0, 510.0, 510.0]
    utilization_samples = [98.0, 99.0, 99.0, 100.0]
    begin_ns = 1_000_000_000 if validation else None
    end_ns = 2_000_000_000 if validation else None
    observations = tuple(
        {
            "seq": index,
            "query_start_ns": 1_100_000_000 + index * 250_000_000,
            "query_end_ns": 1_150_000_000 + index * 250_000_000,
            "power_w": power_samples[index],
            "utilization_percent": utilization_samples[index],
            "free_mib": 8 * 1_024.0,
            "used_mib": 22 * 1_024.0,
        }
        for index in range(4)
    )
    scored_sequences = list(range(4))
    result = {
        "kind": "validation" if validation else "training",
        "status": "ok",
        "eligible": True,
        "warmup_repetitions": warmup,
        "measured_repetitions": measured,
        "elapsed_seconds": 1.0,
        "warmup_unique_graphs": 3,
        "warmup_recompiles": 2,
        "warmup_graph_breaks": 0,
        "measured_unique_graphs": 0,
        "measured_recompiles": 0,
        "measured_graph_breaks": 0,
        "cuda_peak_reserved_bytes": 20 * (1 << 30),
        "cuda_total_bytes": TOTAL_MEMORY,
        "intrinsic_reserved_headroom_bytes": 12 * (1 << 30),
        "required_headroom_bytes": required,
        "physical_environment": {
            "sample_count": 4,
            "minimum_free_bytes": 8 * (1 << 30),
            "maximum_used_bytes": 22 * (1 << 30),
            "required_free_bytes": required,
            "free_bytes_samples": free_samples,
            "free_bytes_samples_sha256": canonical_json_sha256(free_samples),
            "used_bytes_samples": used_samples,
            "used_bytes_samples_sha256": canonical_json_sha256(used_samples),
        },
        "gpu_utilization_percent": _summary(utilization_samples),
        "power_w": _summary(power_samples),
        "telemetry_lifecycle": {
            "schema": FAST_BLT_TELEMETRY_LIFECYCLE_SCHEMA,
            "measurement_begin_ns": begin_ns,
            "measurement_end_ns": end_ns,
            "measurement_duration_seconds": 1.0 if validation else None,
            "observation_count": len(observations),
            "observations": observations,
            "observations_sha256": canonical_json_sha256(observations),
            "scored_observation_sequences": scored_sequences,
            "scored_observation_sequences_sha256": canonical_json_sha256(
                scored_sequences
            ),
            "query_interval_seconds": 0.1,
            "query_timeout_seconds": 2.0,
            "thread_join_timeout_seconds": 3.0,
            "maximum_coverage_gap_seconds": 0.5,
            "first_scored_start_gap_seconds": 0.15 if validation else None,
            "last_scored_end_gap_seconds": 0.1 if validation else None,
            "maximum_scored_query_gap_seconds": 0.25 if validation else None,
            "coverage_complete": validation,
            "stop_requested": True,
            "thread_terminated": True,
            "clean_completion": True,
            "error_count": 0,
            "errors": (),
            "errors_sha256": canonical_json_sha256(()),
        },
        "warmup_workload": warmup_workload,
        "measured_workload": measured_workload,
    }
    if validation:
        role_nll = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        diffusion_targets = measured_workload["updates"][0]["active_bytes"]
        role_counts = [170_667] * 4 + [170_666] * 2
        assert sum(role_counts) == diffusion_targets
        diffusion_loss = sum(
            loss * count
            for loss, count in zip(role_nll, role_counts, strict=True)
        ) / diffusion_targets
        ar_loss = 1.2 * math.log(2.0)
        proxy_nats = 0.8
        result.update(
            total_validation_rows=2_048,
            diffusion_validation_rows=256,
            ar_max_batch_size=128,
            diffusion_max_batch_size=64,
            batching_policy="token_budgeted_ragged_diffusion",
            end_to_end_total_rows_per_second=2_048.0 * measured,
            end_to_end_diffusion_rows_per_second=256.0 * measured,
            cache_quiescence={
                "measured_cache_misses": 0,
                "measured_repetitions": measured,
                "quiescent": True,
                "measured_first_batch_waits": 0,
                "records_sha256": canonical_json_sha256(
                    tuple(
                        {
                            "schedule_cache_misses": record[
                                "schedule_cache_misses"
                            ],
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
                        for record in measured_workload["updates"]
                    )
                ),
            },
            metrics={
                "ar_loss": ar_loss,
                "bpb": 1.2,
                "atomic_bpb": 1.2,
                "diffusion_loss": diffusion_loss,
                "diffusion_elbo_proxy_bpb": proxy_nats / math.log(2.0),
                "diffusion_elbo_proxy_nats_per_block_atom": proxy_nats,
                "diffusion_elbo_atoms": float(
                    measured_workload["updates"][0]["branch_atoms"]
                ),
                "ar_targets": 1_024,
                "literal_bytes": 1_000,
                "special_targets": 24,
                "diffusion_targets": diffusion_targets,
                "diffusion_chunks": 256,
                "elapsed_ms": 500.0,
                "diffusion_role_nll": role_nll,
                "diffusion_role_counts": role_counts,
            },
        )
    else:
        diagnostic_workload = _workload(
            rows=249, repetitions=2, validation=False, diagnostic=True,
            start_step=1,
        )
        diagnostic_update = diagnostic_workload["updates"][-1]
        clean_bytes = 249 * 8_192
        result.update(
            update_ms=250.0,
            host_enqueue_ms=_summary([250.0] * 4),
            cuda_event_update_ms=_summary([245.0] * 4),
            cuda_event_elapsed_seconds=0.98,
            phase_wall_update_ms=250.0,
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
                "coherence_relative_tolerance": 0.02,
                "coherence_absolute_ms": 5.0,
            },
            graph_quiescence=_graph_quiescence(
                warmup_workload["updates"], diagnostic=False
            ),
            diagnostic_workload=diagnostic_workload,
            diagnostic_graph_quiescence=_graph_quiescence(
                diagnostic_workload["updates"], diagnostic=True
            ),
            diagnostic_update=diagnostic_update,
            diagnostic_update_sha256=canonical_json_sha256(diagnostic_update),
            diagnostic_update_ms=250.0,
            execution_step_range={
                "first_completed_step": 1,
                "last_completed_step": 8,
                "diagnostic_warmup_updates": 2,
                "ordinary_warmup_updates": 2,
                "ordinary_measured_updates": 4,
                "total_updates": 8,
                "contiguous": True,
            },
            production_schedule={
                "iterations": 2_000,
                "train_log_every": 10,
                "validation_every": 20,
                "ordinary_updates": 1_800,
                "materialized_updates": 200,
                "ordinary_update_ms": 250.0,
                "materialized_update_ms": 250.0,
            },
            production_schedule_update_ms=250.0,
            ordinary_clean_bytes_per_second=4.0 * clean_bytes,
            production_schedule_clean_bytes_per_second=4.0 * clean_bytes,
        )
    return result


def _production_run(
    monkeypatch: pytest.MonkeyPatch,
    preset: str = FAST_BLT_ENTROPY_B4_COMPLETE_PRESET,
) -> TrainingRunConfig:
    monkeypatch.setenv("BYTE_DIFFUSION_PRESET", preset)
    return TrainingRunConfig.from_env()


def _candidate(run: TrainingRunConfig) -> dict[str, object]:
    contract = run_contract_without_identity(run)
    training = _phase(validation=False)
    source = training["diagnostic_workload"]["updates"][0]
    group_index = source["planned_group_physical_positions"].index(
        max(source["planned_group_physical_positions"])
    )
    boundaries = source["planned_group_row_cu_seqlens"]
    rows = boundaries[group_index + 1] - boundaries[group_index]
    branch_blocks = source["planned_group_origin_counts"][group_index]
    clean_positions = rows * 8_192
    physical_positions = source["planned_group_physical_positions"][group_index]
    return {
        "status": "ok",
        "eligible": True,
        "microbatch": run.microbatch_per_rank,
        "run_contract": contract,
        "run_contract_sha256": canonical_json_sha256(contract),
        "parameter_count": fast_blt_complete_model_config(
            run.preset
        ).production_parameter_target,
        "block_mask_build": {
            "kind": "ragged_block_mask_build",
            "status": "ok",
            "compiler_disabled": True,
            "warmup_repetitions": 1,
            "measured_repetitions": 4,
            "elapsed_seconds": 0.04,
            "build_ms": {
                **_summary([8.0, 9.0, 11.0, 12.0]),
            },
            "topology": {
                "source_update_row_indices_sha256": source[
                    "row_indices_sha256"
                ],
                "source_update_row_origin_counts_sha256": source[
                    "row_origin_counts_sha256"
                ],
                "group_index": group_index,
                "group_row_indices_sha256": source[
                    "planned_group_row_indices_sha256"
                ][group_index],
                "rows": rows,
                "block_length": 4,
                "branch_blocks": branch_blocks,
                "branch_atoms": branch_blocks * 4 - 17,
                "clean_positions": clean_positions,
                "physical_positions": physical_positions,
            },
            "cuda_peak_reserved_bytes": 20 * (1 << 30),
            "cuda_total_bytes": TOTAL_MEMORY,
            "intrinsic_reserved_headroom_bytes": 12 * (1 << 30),
            "required_headroom_bytes": required_headroom_bytes(TOTAL_MEMORY),
        },
        "training": training,
        "validation": _phase(validation=True),
    }


def test_complete_geometry_has_no_rectangular_branch_claim() -> None:
    contract = FastBltGeometry().contract()
    assert contract["schema"] == FAST_BLT_GEOMETRY_SCHEMA
    assert contract["origin_count"] == "variable_per_row"
    assert contract["exhaustive"] is True
    assert contract["virtual_bos_is_origin"] is False
    assert contract["first_physical_patch_condition"] == "virtual_bos"
    assert contract["overflow_policy"] == "pad_at_finite_training_sequence_end"
    assert contract["ar_validation_batching"] == "fixed_max_batch"
    assert contract["diffusion_validation_batching"] == "token_budgeted_ragged"
    assert contract["validation_rows"] == 2_048
    assert contract["diffusion_validation_rows"] == 256
    assert "branches_per_row" not in contract
    assert "branch_byte_budget" not in contract
    with pytest.raises(ValueError, match="pinned to D4"):
        FastBltGeometry(block_length=8)


def test_diffusion_validation_indices_match_trainer_even_spacing() -> None:
    indices = _diffusion_validation_indices()
    assert indices.shape == (256,)
    assert indices.tolist() == list(range(4, 2_048, 8))
    assert not indices.flags.writeable


@pytest.mark.parametrize(
    "preset",
    [
        FAST_BLT_ENTROPY_B4_COMPLETE_PRESET,
        FAST_BLT_ENTROPY_B4_COMPLETE_PAPER_RATIO_PRESET,
    ],
)
def test_benchmark_run_is_exact_complete_named_preset(
    monkeypatch: pytest.MonkeyPatch,
    preset: str,
) -> None:
    production = _production_run(monkeypatch, preset)
    candidate = build_benchmark_run_config(production, 32)
    assert candidate is production
    assert candidate.preset == preset
    assert candidate.blt_origin_policy == "all_entropy_patch_starts"
    assert candidate.corruption.kind == "blt_bernoulli"
    assert candidate.corruption.canvas_length == 4
    assert candidate.objective_reduction == "paper_sum"
    assert candidate.activation_checkpointing is True
    assert candidate.global_batch_size == 249
    assert candidate.validation_chunks == 2_048
    assert candidate.diffusion_validation_chunks == 256
    assert candidate.gradient_accumulation == 8
    with pytest.raises(ValueError, match="microbatch 32"):
        build_benchmark_run_config(production, 16)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("parameter_count",), FAST_BLT_ENTROPY_B4_COMPLETE_PARAMETER_TARGET - 1),
        (("training", "measured_unique_graphs"), 1),
        (("training", "measured_workload", "origin_policy"), "sampled"),
        (("training", "diagnostic_workload", "updates", 0, "active_bytes"), 10**12),
        (("training", "diagnostic_workload", "updates", 0, "microsteps"), 0),
        (
            (
                "training", "measured_workload", "updates", 0,
                "planned_max_physical_positions",
            ),
            300_000,
        ),
        (("validation", "measured_workload", "batching_policy"), "fixed_64_rows"),
        (("validation", "cache_quiescence", "quiescent"), False),
        (("validation", "cache_quiescence", "measured_cache_misses"), 1),
        (("validation", "cache_quiescence", "records_sha256"), "0" * 64),
        (
            (
                "validation", "measured_workload", "updates", 0,
                "first_batch_waits",
            ),
            1,
        ),
        (("training", "diagnostic_workload", "updates", 0, "branch_blocks"), 1),
        (("training", "diagnostic_workload", "updates", 0, "block_mask_build_ms"), 0.0),
        (("training", "measured_workload", "updates", 0, "physical_token_budget"), 278_528),
        (("training", "measured_workload", "updates", 0, "materialize_metrics"), True),
        (("training", "cuda_event_update_ms", "mean"), 1.0),
        (("training", "gpu_utilization_percent", "minimum"), float("nan")),
        (("training", "power_w", "samples_sha256"), "0" * 64),
        (
            (
                "validation", "telemetry_lifecycle",
                "scored_observation_sequences", 0,
            ),
            -1,
        ),
        (
            (
                "validation", "telemetry_lifecycle",
                "scored_observation_sequences", 0,
            ),
            99,
        ),
        (("block_mask_build", "build_ms", "maximum"), float("nan")),
        (("training", "phase_wall_update_ms"), 1.0),
        (("training", "graph_quiescence", "achieved"), False),
        (("training", "diagnostic_graph_quiescence", "materialize_metrics"), False),
        (("validation", "measured_repetitions"), 1),
        (("validation", "metrics", "bpb"), float("nan")),
        (("validation", "metrics", "ar_loss"), -1.0),
        (("validation", "metrics", "diffusion_loss"), float("nan")),
        (("validation", "metrics", "elapsed_ms"), float("nan")),
        (("validation", "metrics", "ar_targets"), 1_025),
        (("validation", "metrics", "literal_bytes"), 1_000.0),
        (("validation", "metrics", "diffusion_role_nll", 0), float("nan")),
        (("validation", "metrics", "diffusion_role_counts", 0), 999),
        (("block_mask_build", "compiler_disabled"), False),
    ),
)
def test_candidate_evidence_fails_closed(monkeypatch: pytest.MonkeyPatch, path, value) -> None:
    run = build_benchmark_run_config(_production_run(monkeypatch), 32)
    candidate = _candidate(run)
    cursor = candidate
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    assert not candidate_passes(
        candidate, microbatch=32,
        expected_run_contract=run_contract_without_identity(run),
        total_memory=TOTAL_MEMORY,
    )


def test_update_timing_and_graph_deltas_fail_closed() -> None:
    record = _attach_update_timing(
        {"rows": 249}, host_duration_ms=250.0, cuda_event_ms=245.0,
        synchronized_host_call=False,
    )
    assert record == {
        "rows": 249,
        "host_enqueue_ms": 250.0,
        "cuda_event_ms": 245.0,
    }
    assert _graph_delta(
        {"unique_graphs": 2, "recompiles": 1, "graph_breaks": 0},
        {"unique_graphs": 3, "recompiles": 1, "graph_breaks": 0},
    ) == {"unique_graphs": 1, "recompiles": 0, "graph_breaks": 0}
    with pytest.raises(ValueError, match="finite and positive"):
        _attach_update_timing(
            {"rows": 249}, host_duration_ms=0.0, cuda_event_ms=245.0,
            synchronized_host_call=False,
        )
    with pytest.raises(ValueError, match="moved backwards"):
        _graph_delta(
            {"unique_graphs": 2, "recompiles": 1, "graph_breaks": 0},
            {"unique_graphs": 1, "recompiles": 1, "graph_breaks": 0},
        )


def test_harness_identity_and_materialization_schedule_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = benchmark_harness_provenance()
    _require_unchanged_harness(snapshot, deepcopy(snapshot))
    changed = deepcopy(snapshot)
    changed["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="changed during execution"):
        _require_unchanged_harness(snapshot, changed)

    run = _production_run(monkeypatch)
    assert _materialized_update_count(run) == 200


def test_topology_arithmetic_fails_after_rehashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = build_benchmark_run_config(_production_run(monkeypatch), 32)
    candidate = _candidate(run)
    workload = candidate["training"]["measured_workload"]
    record = workload["updates"][0]
    record.update(
        planned_microsteps=1,
        planned_min_microbatch=2,
        planned_max_microbatch=2,
        planned_group_row_cu_seqlens=[0, 249],
        planned_group_physical_positions=[record["total_physical_positions"]],
    )
    record["planned_group_row_cu_seqlens_sha256"] = canonical_json_sha256(
        record["planned_group_row_cu_seqlens"]
    )
    record["planned_group_physical_positions_sha256"] = canonical_json_sha256(
        record["planned_group_physical_positions"]
    )
    workload["updates_sha256"] = canonical_json_sha256(workload["updates"])
    assert not candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=run_contract_without_identity(run),
        total_memory=TOTAL_MEMORY,
    )


@pytest.mark.parametrize("delta", (1, 4))
def test_group_physical_work_is_derived_from_group_origins(
    monkeypatch: pytest.MonkeyPatch, delta: int
) -> None:
    run = build_benchmark_run_config(_production_run(monkeypatch), 32)
    candidate = _candidate(run)
    workload = candidate["training"]["measured_workload"]
    record = workload["updates"][0]
    for field in (
        "planned_group_physical_positions",
        "actual_group_physical_positions",
    ):
        values = record[field]
        values[10] += delta
        values[11] -= delta
    record["planned_group_physical_positions_sha256"] = canonical_json_sha256(
        record["planned_group_physical_positions"]
    )
    workload["updates_sha256"] = canonical_json_sha256(workload["updates"])
    assert not candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=run_contract_without_identity(run),
        total_memory=TOTAL_MEMORY,
    )


def test_diagnostic_host_call_must_cover_its_cuda_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = build_benchmark_run_config(_production_run(monkeypatch), 32)
    candidate = _candidate(run)
    workload = candidate["training"]["diagnostic_workload"]
    workload["updates"][0]["host_synchronized_call_ms"] = 1.0
    workload["updates_sha256"] = canonical_json_sha256(workload["updates"])
    assert not candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=run_contract_without_identity(run),
        total_memory=TOTAL_MEMORY,
    )


def test_block_mask_topology_is_linked_to_heaviest_scheduled_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = build_benchmark_run_config(_production_run(monkeypatch), 32)
    candidate = _candidate(run)
    topology = candidate["block_mask_build"]["topology"]
    topology["clean_positions"] = 1
    topology["physical_positions"] = 1 + topology["branch_blocks"] * 4
    assert not candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=run_contract_without_identity(run),
        total_memory=TOTAL_MEMORY,
    )


@pytest.mark.parametrize(
    ("field", "samples"),
    (
        ("gpu_utilization_percent", [101.0] * 4),
        ("power_w", [1_000.0] * 3),
    ),
)
def test_telemetry_domains_and_channel_alignment_fail_closed(
    monkeypatch: pytest.MonkeyPatch, field: str, samples: list[float]
) -> None:
    run = build_benchmark_run_config(_production_run(monkeypatch), 32)
    candidate = _candidate(run)
    candidate["training"][field] = _summary(samples)
    assert not candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=run_contract_without_identity(run),
        total_memory=TOTAL_MEMORY,
    )


def test_telemetry_marker_excludes_warmup_only_from_scored_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark_fast_blt, "nvidia_smi_selector", lambda _: "0")
    telemetry = _Telemetry(benchmark_fast_blt.torch.device("cpu"))
    for power, utilization, free, used in (
        (100.0, 0.0, 30_000.0, 2_000.0),
        (120.0, 10.0, 29_000.0, 3_000.0),
    ):
        query_start = benchmark_fast_blt.time.monotonic_ns()
        telemetry._record_observation(
            query_start_ns=query_start,
            query_end_ns=benchmark_fast_blt.time.monotonic_ns(),
            power_w=power,
            utilization_percent=utilization,
            free_mib=free,
            used_mib=used,
        )
    telemetry.mark_measurement_start()
    for power, utilization, free, used in (
        (480.0, 3.0, 28_000.0, 4_000.0),
        (500.0, 100.0, 27_000.0, 5_000.0),
    ):
        query_start = benchmark_fast_blt.time.monotonic_ns()
        telemetry._record_observation(
            query_start_ns=query_start,
            query_end_ns=benchmark_fast_blt.time.monotonic_ns(),
            power_w=power,
            utilization_percent=utilization,
            free_mib=free,
            used_mib=used,
        )
    telemetry.mark_measurement_end()
    assert telemetry._measurement_end_ns is not None
    telemetry._record_observation(
        query_start_ns=telemetry._measurement_end_ns - 1,
        query_end_ns=telemetry._measurement_end_ns + 1,
        power_w=1.0,
        utilization_percent=0.0,
        free_mib=26_000.0,
        used_mib=6_000.0,
    )

    report = telemetry.report(required=4 << 30)

    assert report["power_w"]["samples"] == (480.0, 500.0)
    assert report["power_w"]["count"] == 2
    assert report["power_w"]["samples_sha256"] == canonical_json_sha256(
        (480.0, 500.0)
    )
    # A real low-utilization sample wholly inside measurement remains scored.
    assert report["gpu_utilization_percent"]["samples"] == (3.0, 100.0)
    assert report["gpu_utilization_percent"]["count"] == 2
    assert report["gpu_utilization_percent"][
        "samples_sha256"
    ] == canonical_json_sha256((3.0, 100.0))
    physical = report["physical_environment"]
    assert physical["sample_count"] == 5
    assert physical["free_bytes_samples"] == tuple(
        value * (1 << 20)
        for value in (30_000, 29_000, 28_000, 27_000, 26_000)
    )
    assert physical["used_bytes_samples_sha256"] == canonical_json_sha256(
        tuple(
            value * (1 << 20) for value in (2_000, 3_000, 4_000, 5_000, 6_000)
        )
    )
    with pytest.raises(RuntimeError, match="already marked"):
        telemetry.mark_measurement_start()


def test_validation_benchmark_uses_one_marked_telemetry_lifetime() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(_benchmark_candidate)))
    validation_scopes = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With) or len(node.items) != 1:
            continue
        item = node.items[0]
        if (
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id == "_Telemetry"
            and isinstance(item.optional_vars, ast.Name)
            and item.optional_vars.id == "validation_telemetry"
        ):
            validation_scopes.append(node)
    assert len(validation_scopes) == 1
    constants = {
        node.value
        for node in ast.walk(validation_scopes[0])
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {"validation_warmup", "validation_measured"} <= constants
    marker_calls = [
        node
        for node in ast.walk(validation_scopes[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "mark_measurement_start"
    ]
    assert len(marker_calls) == 1


@pytest.mark.parametrize("boundary", ("begin", "end"))
def test_telemetry_excludes_queries_blocked_across_measurement_boundary(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    query_started = threading.Event()
    release_query = threading.Event()
    observation_recorded = threading.Event()

    def blocked_query(*args, **kwargs):
        del args, kwargs
        query_started.set()
        assert release_query.wait(timeout=2.0)
        return SimpleNamespace(returncode=0, stdout="400, 99, 28000, 4000\n")

    monkeypatch.setattr(benchmark_fast_blt, "nvidia_smi_selector", lambda _: "0")
    monkeypatch.setattr(benchmark_fast_blt.subprocess, "run", blocked_query)
    telemetry = _Telemetry(benchmark_fast_blt.torch.device("cpu"))
    original_record = telemetry._record_observation

    def recording_observation(**kwargs):
        original_record(**kwargs)
        observation_recorded.set()

    telemetry._record_observation = recording_observation
    if boundary == "end":
        telemetry.mark_measurement_start()
    telemetry.__enter__()
    try:
        assert query_started.wait(timeout=2.0)
        if boundary == "begin":
            telemetry.mark_measurement_start()
        else:
            telemetry.mark_measurement_end()
        release_query.set()
        assert observation_recorded.wait(timeout=2.0)
        if boundary == "begin":
            telemetry.mark_measurement_end()
    finally:
        telemetry.__exit__(None, None, None)

    report = telemetry.report(required=4 << 30)
    assert report["power_w"]["samples"] == ()
    assert report["gpu_utilization_percent"]["samples"] == ()
    assert report["physical_environment"]["sample_count"] == 1
    observation = report["telemetry_lifecycle"]["observations"][0]
    begin_ns = report["telemetry_lifecycle"]["measurement_begin_ns"]
    end_ns = report["telemetry_lifecycle"]["measurement_end_ns"]
    assert observation["query_start_ns"] < end_ns
    assert observation["query_end_ns"] > begin_ns


def test_telemetry_exit_rejects_query_blocked_beyond_join_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_started = threading.Event()
    release_query = threading.Event()

    def stuck_query(*args, **kwargs):
        del args
        assert kwargs["timeout"] == 2.0
        query_started.set()
        release_query.wait(timeout=5.0)
        return SimpleNamespace(returncode=0, stdout="400, 99, 28000, 4000\n")

    monkeypatch.setattr(benchmark_fast_blt, "nvidia_smi_selector", lambda _: "0")
    monkeypatch.setattr(benchmark_fast_blt.subprocess, "run", stuck_query)
    telemetry = _Telemetry(benchmark_fast_blt.torch.device("cpu"))
    telemetry.__enter__()
    assert query_started.wait(timeout=2.0)
    with pytest.raises(RuntimeError, match="did not terminate cleanly"):
        telemetry.__exit__(None, None, None)
    release_query.set()
    telemetry.thread.join(timeout=2.0)
    assert not telemetry.thread.is_alive()
    lifecycle = telemetry.report(required=4 << 30)["telemetry_lifecycle"]
    assert lifecycle["stop_requested"] is True
    assert lifecycle["thread_terminated"] is False
    assert lifecycle["clean_completion"] is False


def test_telemetry_sampler_dying_after_three_queries_is_authenticated_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sampler_failed = threading.Event()

    def dying_query(*args, **kwargs):
        nonlocal calls
        del args
        assert kwargs["timeout"] == 2.0
        calls += 1
        if calls == 4:
            sampler_failed.set()
            raise RuntimeError("sampler died")
        return SimpleNamespace(returncode=0, stdout="400, 99, 28000, 4000\n")

    monkeypatch.setattr(benchmark_fast_blt, "nvidia_smi_selector", lambda _: "0")
    monkeypatch.setattr(benchmark_fast_blt.subprocess, "run", dying_query)
    telemetry = _Telemetry(benchmark_fast_blt.torch.device("cpu"))
    telemetry.mark_measurement_start()
    telemetry.__enter__()
    assert sampler_failed.wait(timeout=2.0)
    telemetry.thread.join(timeout=2.0)
    telemetry.mark_measurement_end()
    with pytest.raises(RuntimeError, match="did not terminate cleanly"):
        telemetry.__exit__(None, None, None)
    lifecycle = telemetry.report(required=4 << 30)["telemetry_lifecycle"]
    assert lifecycle["observation_count"] == 3
    assert lifecycle["error_count"] == 1
    assert lifecycle["errors"][0]["kind"] == "exception"
    assert lifecycle["clean_completion"] is False


def test_three_early_samples_cannot_certify_four_second_measurement() -> None:
    phase = _phase(validation=True)
    lifecycle = phase["telemetry_lifecycle"]
    observations = list(lifecycle["observations"][:3])
    sequences = [0, 1, 2]
    lifecycle.update(
        measurement_end_ns=5_000_000_000,
        measurement_duration_seconds=4.0,
        observation_count=3,
        observations=observations,
        observations_sha256=canonical_json_sha256(observations),
        scored_observation_sequences=sequences,
        scored_observation_sequences_sha256=canonical_json_sha256(sequences),
        first_scored_start_gap_seconds=0.15,
        last_scored_end_gap_seconds=3.35,
        maximum_scored_query_gap_seconds=0.25,
        coverage_complete=True,
    )
    phase["elapsed_seconds"] = 4.0
    phase["power_w"] = _summary([480.0, 500.0, 510.0])
    phase["gpu_utilization_percent"] = _summary([98.0, 99.0, 99.0])
    environment = phase["physical_environment"]
    environment["sample_count"] = 3
    environment["free_bytes_samples"] = environment["free_bytes_samples"][:3]
    environment["used_bytes_samples"] = environment["used_bytes_samples"][:3]
    environment["free_bytes_samples_sha256"] = canonical_json_sha256(
        environment["free_bytes_samples"]
    )
    environment["used_bytes_samples_sha256"] = canonical_json_sha256(
        environment["used_bytes_samples"]
    )

    assert not _telemetry_domains_pass(phase)


def test_slow_successful_query_does_not_count_as_continuous_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark_fast_blt, "nvidia_smi_selector", lambda _: "0")
    telemetry = _Telemetry(benchmark_fast_blt.torch.device("cpu"))
    telemetry._measurement_begin_ns = 1_000_000_000
    telemetry._measurement_end_ns = 2_000_000_000
    telemetry._record_observation(
        query_start_ns=1_100_000_000,
        query_end_ns=1_700_000_000,
        power_w=480.0,
        utilization_percent=99.0,
        free_mib=8_192.0,
        used_mib=22_528.0,
    )
    telemetry._record_observation(
        query_start_ns=1_710_000_000,
        query_end_ns=1_900_000_000,
        power_w=490.0,
        utilization_percent=99.0,
        free_mib=8_192.0,
        used_mib=22_528.0,
    )

    lifecycle = telemetry.report(required=4 << 30)["telemetry_lifecycle"]
    assert lifecycle["first_scored_start_gap_seconds"] == pytest.approx(0.7)
    assert lifecycle["maximum_scored_query_gap_seconds"] == pytest.approx(0.2)
    assert lifecycle["coverage_complete"] is False


def test_validator_rejects_slow_query_hidden_inside_query_interval() -> None:
    phase = _phase(validation=True)
    lifecycle = phase["telemetry_lifecycle"]
    observations = [dict(value) for value in lifecycle["observations"]]
    bounds = (
        (1_100_000_000, 1_700_000_000),
        (1_710_000_000, 1_750_000_000),
        (1_760_000_000, 1_800_000_000),
        (1_810_000_000, 1_900_000_000),
    )
    for observation, (start_ns, end_ns) in zip(
        observations, bounds, strict=True
    ):
        observation["query_start_ns"] = start_ns
        observation["query_end_ns"] = end_ns
    lifecycle.update(
        observations=observations,
        observations_sha256=canonical_json_sha256(observations),
        first_scored_start_gap_seconds=0.7,
        last_scored_end_gap_seconds=0.1,
        maximum_scored_query_gap_seconds=0.1,
        coverage_complete=True,
    )

    assert not _telemetry_domains_pass(phase)


def test_telemetry_error_candidate_is_authenticated_but_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = build_benchmark_run_config(_production_run(monkeypatch), 32)
    contract = run_contract_without_identity(run)
    candidate = _telemetry_error_candidate(contract)
    assert _telemetry_error_candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=contract,
        total_memory=TOTAL_MEMORY,
    )
    assert not candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=contract,
        total_memory=TOTAL_MEMORY,
    )
    candidate["telemetry_terminal"]["errors_sha256"] = "0" * 64
    assert not _telemetry_error_candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=contract,
        total_memory=TOTAL_MEMORY,
    )


def test_execution_ledgers_cannot_replay_steps_between_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = build_benchmark_run_config(_production_run(monkeypatch), 32)
    candidate = _candidate(run)
    workload = candidate["training"]["warmup_workload"]
    for index, record in enumerate(workload["updates"], start=1):
        record["actual_step"] = index
    workload["updates_sha256"] = canonical_json_sha256(workload["updates"])
    assert not candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=run_contract_without_identity(run),
        total_memory=TOTAL_MEMORY,
    )


def test_execution_ledgers_are_anchored_to_fresh_candidate_step_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = build_benchmark_run_config(_production_run(monkeypatch), 32)
    candidate = _candidate(run)
    training = candidate["training"]
    for workload_name in (
        "diagnostic_workload",
        "warmup_workload",
        "measured_workload",
    ):
        workload = training[workload_name]
        for record in workload["updates"]:
            record["actual_step"] += 100
        workload["updates_sha256"] = canonical_json_sha256(workload["updates"])
    training["diagnostic_update_sha256"] = canonical_json_sha256(
        training["diagnostic_update"]
    )
    training["execution_step_range"]["first_completed_step"] = 101
    training["execution_step_range"]["last_completed_step"] = 108
    assert not candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=run_contract_without_identity(run),
        total_memory=TOTAL_MEMORY,
    )


def test_timing_coherence_rejects_impossible_sequential_totals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = build_benchmark_run_config(_production_run(monkeypatch), 32)
    candidate = _candidate(run)
    training = candidate["training"]
    workload = training["measured_workload"]
    for record in workload["updates"]:
        record["host_enqueue_ms"] = 1_000.0
        record["cuda_event_ms"] = 1_000.0
    workload["updates_sha256"] = canonical_json_sha256(workload["updates"])
    training["host_enqueue_ms"] = _summary([1_000.0] * 4)
    training["cuda_event_update_ms"] = _summary([1_000.0] * 4)
    training["cuda_event_elapsed_seconds"] = 4.0
    assert not candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=run_contract_without_identity(run),
        total_memory=TOTAL_MEMORY,
    )


def test_graph_nonquiescence_is_authenticated_but_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = build_benchmark_run_config(_production_run(monkeypatch), 32)
    contract = run_contract_without_identity(run)
    workload = _workload(
        rows=249,
        repetitions=FAST_BLT_MAX_GRAPH_WARMUP_UPDATES,
        validation=False,
    )
    records = workload["updates"]
    observations = [
        {
            "update": index,
            "materialize_metrics": False,
            "representative_non_singleton": True,
            "planned_min_microbatch": record["planned_min_microbatch"],
            "planned_max_microbatch": record["planned_max_microbatch"],
            "new_unique_graphs": 1,
            "new_recompiles": 0,
            "new_graph_breaks": 0,
        }
        for index, record in enumerate(records, start=1)
    ]
    evidence = {
        "schema": FAST_BLT_GRAPH_QUIESCENCE_SCHEMA,
        "achieved": False,
        "materialize_metrics": False,
        "minimum_updates": 2,
        "maximum_updates": FAST_BLT_MAX_GRAPH_WARMUP_UPDATES,
        "observed_updates": FAST_BLT_MAX_GRAPH_WARMUP_UPDATES,
        "representative_non_singleton": True,
        "total_unique_graphs": FAST_BLT_MAX_GRAPH_WARMUP_UPDATES,
        "total_recompiles": 0,
        "total_graph_breaks": 0,
        "observations": observations,
        "observations_sha256": canonical_json_sha256(observations),
    }
    candidate = {
        "status": "graph_nonquiescent",
        "eligible": False,
        "graph_phase": "training_warmup",
        "error": "bounded warmup exhausted",
        "microbatch": 32,
        "run_contract": contract,
        "run_contract_sha256": canonical_json_sha256(contract),
        "parameter_count": FAST_BLT_ENTROPY_B4_COMPLETE_PARAMETER_TARGET,
        "graph_quiescence": evidence,
        "warmup_workload": workload,
        "cuda_peak_reserved_bytes": 20 * (1 << 30),
        "cuda_total_bytes": TOTAL_MEMORY,
        "intrinsic_reserved_headroom_bytes": 12 * (1 << 30),
        "required_headroom_bytes": required_headroom_bytes(TOTAL_MEMORY),
    }
    assert _nonquiescent_candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=contract,
        total_memory=TOTAL_MEMORY,
    )
    assert not candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=contract,
        total_memory=TOTAL_MEMORY,
    )
    evidence["observations"][0]["new_unique_graphs"] = 0
    assert not _nonquiescent_candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=contract,
        total_memory=TOTAL_MEMORY,
    )


def test_validation_cache_nonquiescence_is_authenticated_but_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = build_benchmark_run_config(_production_run(monkeypatch), 32)
    contract = run_contract_without_identity(run)
    workload = _workload(rows=256, repetitions=1, validation=True)
    record = workload["updates"][0]
    record["first_batch_cache_misses"] = 1
    record["first_batch_waits"] = 1
    record["cache_misses"] = 1
    workload["updates_sha256"] = canonical_json_sha256(workload["updates"])
    cache_record = {
        name: record[name]
        for name in (
            "schedule_cache_misses",
            "corruption_plan_cache_misses",
            "ragged_mask_metadata_cache_misses",
            "first_batch_cache_misses",
            "first_batch_waits",
        )
    }
    evidence = {
        "measured_cache_misses": 1,
        "measured_repetitions": 1,
        "quiescent": False,
        "measured_first_batch_waits": 1,
        "records_sha256": canonical_json_sha256((cache_record,)),
    }
    candidate = {
        "status": "validation_cache_nonquiescent",
        "eligible": False,
        "cache_phase": "validation_measured",
        "error": "validation staging caches were not quiescent",
        "microbatch": 32,
        "run_contract": contract,
        "run_contract_sha256": canonical_json_sha256(contract),
        "parameter_count": FAST_BLT_ENTROPY_B4_COMPLETE_PARAMETER_TARGET,
        "cache_quiescence": evidence,
        "measured_workload": workload,
        "cuda_peak_allocated_bytes": 8 * (1 << 30),
        "cuda_peak_reserved_bytes": 20 * (1 << 30),
        "cuda_total_bytes": TOTAL_MEMORY,
        "intrinsic_reserved_headroom_bytes": 12 * (1 << 30),
        "required_headroom_bytes": required_headroom_bytes(TOTAL_MEMORY),
    }

    assert _cache_nonquiescent_candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=contract,
        total_memory=TOTAL_MEMORY,
    )
    assert not candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=contract,
        total_memory=TOTAL_MEMORY,
    )
    evidence["records_sha256"] = "0" * 64
    assert not _cache_nonquiescent_candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=contract,
        total_memory=TOTAL_MEMORY,
    )


def test_oom_candidate_authentication_accepts_only_exact_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = build_benchmark_run_config(_production_run(monkeypatch), 32)
    contract = run_contract_without_identity(run)
    required = required_headroom_bytes(TOTAL_MEMORY)
    candidate = {
        "status": "oom",
        "eligible": False,
        "oom_phase": "training_measured",
        "error": "CUDA out of memory",
        "microbatch": 32,
        "run_contract": contract,
        "run_contract_sha256": canonical_json_sha256(contract),
        "cuda_total_bytes": TOTAL_MEMORY,
        "required_headroom_bytes": required,
        "oom_physical_total_bytes": TOTAL_MEMORY,
        "oom_physical_free_bytes": 2 * (1 << 30),
    }
    assert _oom_candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=contract,
        total_memory=TOTAL_MEMORY,
    )
    assert not candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=contract,
        total_memory=TOTAL_MEMORY,
    )

    mutations = (
        ("status", "ok"),
        ("eligible", True),
        ("oom_phase", "unknown"),
        ("error", ""),
        ("microbatch", 16),
        ("run_contract_sha256", "0" * 64),
        ("cuda_total_bytes", TOTAL_MEMORY - 1),
        ("required_headroom_bytes", required - 1),
        ("oom_physical_total_bytes", TOTAL_MEMORY - 1),
        ("oom_physical_free_bytes", -1),
        ("oom_physical_free_bytes", TOTAL_MEMORY + 1),
    )
    for key, value in mutations:
        broken = deepcopy(candidate)
        broken[key] = value
        assert not _oom_candidate_passes(
            broken,
            microbatch=32,
            expected_run_contract=contract,
            total_memory=TOTAL_MEMORY,
        )


def test_compile_error_candidate_is_authenticated_and_never_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = build_benchmark_run_config(_production_run(monkeypatch), 32)
    contract = run_contract_without_identity(run)
    required = required_headroom_bytes(TOTAL_MEMORY)
    candidate = _compile_error_candidate(contract)
    assert _compile_error_candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=contract,
        total_memory=TOTAL_MEMORY,
    )
    assert not candidate_passes(
        candidate,
        microbatch=32,
        expected_run_contract=contract,
        total_memory=TOTAL_MEMORY,
    )

    empty_message = deepcopy(candidate)
    empty_message["error_type"] = (
        "torch._dynamo.exc.FailOnRecompileLimitHit"
    )
    empty_message["error"] = ""
    empty_message["error_sha256"] = hashlib.sha256(b"").hexdigest()
    empty_message["error_repr"] = "FailOnRecompileLimitHit()"
    empty_message["error_repr_sha256"] = hashlib.sha256(
        empty_message["error_repr"].encode("utf-8")
    ).hexdigest()
    empty_message["traceback"] = (
        "Traceback (most recent call last):\n"
        "torch._dynamo.exc.FailOnRecompileLimitHit"
    )
    empty_message["traceback_sha256"] = hashlib.sha256(
        empty_message["traceback"].encode("utf-8")
    ).hexdigest()
    assert _compile_error_candidate_passes(
        empty_message,
        microbatch=32,
        expected_run_contract=contract,
        total_memory=TOTAL_MEMORY,
    )

    mutations = (
        ("status", "ok"),
        ("eligible", True),
        ("compile_phase", "unknown"),
        ("error_type", "builtins.RuntimeError"),
        ("error", ""),
        ("error_sha256", "0" * 64),
        ("error_repr", ""),
        ("error_repr_sha256", "0" * 64),
        ("traceback", "not a compiler traceback"),
        ("traceback_sha256", "0" * 64),
        ("microbatch", 16),
        ("run_contract_sha256", "0" * 64),
        ("parameter_count", FAST_BLT_ENTROPY_B4_COMPLETE_PARAMETER_TARGET - 1),
        ("cuda_peak_allocated_bytes", 21 * (1 << 30)),
        ("cuda_peak_reserved_bytes", TOTAL_MEMORY + 1),
        ("cuda_total_bytes", TOTAL_MEMORY - 1),
        ("intrinsic_reserved_headroom_bytes", 0),
        ("required_headroom_bytes", required - 1),
        ("instantaneous_physical_free_bytes", -1),
        ("instantaneous_physical_total_bytes", TOTAL_MEMORY - 1),
    )
    for key, value in mutations:
        broken = deepcopy(candidate)
        broken[key] = value
        assert not _compile_error_candidate_passes(
            broken,
            microbatch=32,
            expected_run_contract=contract,
            total_memory=TOTAL_MEMORY,
        )


def test_compile_error_classifier_uses_pytorch_exception_identity() -> None:
    from torch._dynamo.exc import (
        FailOnRecompileLimitHit,
        InternalTorchDynamoError,
        InvalidBackend,
        RecompileError,
    )
    from torch._inductor.exc import InductorError, InvalidCxxCompiler

    assert _is_torch_compile_error(
        InductorError(RuntimeError("CantSplit"), None)
    )
    assert _is_torch_compile_error(InternalTorchDynamoError("trace failed"))
    assert _is_torch_compile_error(InvalidCxxCompiler("missing-cxx"))
    assert _is_torch_compile_error(FailOnRecompileLimitHit())
    assert _is_torch_compile_error(RecompileError("recompile failed"))
    assert _is_torch_compile_error(InvalidBackend("missing-backend"))
    assert not _is_torch_compile_error(RuntimeError("Inductor CantSplit"))


def test_main_rejects_one_validation_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = {"sha256": "a" * 64}
    args = SimpleNamespace(
        expected_benchmark_harness_sha256=harness["sha256"],
        warmup_updates=2,
        measured_updates=4,
        validation_warmup=1,
        validation_measured=1,
    )
    monkeypatch.setattr(benchmark_fast_blt, "_parse_args", lambda: args)
    monkeypatch.setattr(
        benchmark_fast_blt, "benchmark_harness_provenance", lambda: harness
    )
    monkeypatch.setattr(
        benchmark_fast_blt, "require_fast_blt_allocator_environment", lambda: None
    )
    monkeypatch.setattr(benchmark_fast_blt.torch.cuda, "is_available", lambda: True)

    with pytest.raises(ValueError, match="1/2 validation repetitions"):
        benchmark_fast_blt.main()


@pytest.mark.parametrize(
    "preset",
    [
        FAST_BLT_ENTROPY_B4_COMPLETE_PRESET,
        FAST_BLT_ENTROPY_B4_COMPLETE_PAPER_RATIO_PRESET,
    ],
)
@pytest.mark.parametrize(
    "failure_kind",
    ("compile_error", "validation_cache_nonquiescent", "telemetry_error"),
)
def test_main_writes_authenticated_failure_report_before_no_eligible_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    preset: str,
    failure_kind: str,
) -> None:
    production = _production_run(monkeypatch, preset)
    candidate_run = build_benchmark_run_config(production, 32)
    candidate_contract = run_contract_without_identity(candidate_run)
    model = fast_blt_complete_model_config(preset)
    harness = {"path": "scripts/benchmark_byte_diffusion_fast_blt.py", "sha256": "a" * 64}
    source = {"sha256": "b" * 64}
    payload_sha256 = "c" * 64
    source_manifest_sha256 = "d" * 64
    atomic_manifest_sha256 = "e" * 64
    train_sha256 = "f" * 64
    validation_sha256 = "1" * 64
    patcher_sha256 = "2" * 64
    data_path = tmp_path / "data"
    data_path.mkdir()
    (data_path / "manifest.json").write_text(
        json.dumps(
            {
                "payload_sha256": payload_sha256,
                "source_manifests": [{"sha256": source_manifest_sha256}],
            }
        )
    )
    patching = SimpleNamespace(
        name="causal_entropy_v1",
        artifact_sha256=patcher_sha256,
        max_patch_size=8,
    )

    class FakeChunks:
        def __init__(self, *, rows: int, dataset_sha256: str) -> None:
            self.rows = rows
            self.dataset_sha256 = dataset_sha256
            self.patching = patching

        def __len__(self) -> int:
            return self.rows

    train_chunks = FakeChunks(rows=4_096, dataset_sha256=train_sha256)
    validation_chunks = FakeChunks(
        rows=FAST_BLT_VALIDATION_ROWS,
        dataset_sha256=validation_sha256,
    )
    manifest = SimpleNamespace(sha256=atomic_manifest_sha256)
    output_json = tmp_path / "readiness.json"
    args = SimpleNamespace(
        data_path=data_path,
        candidate_microbatches=(32,),
        warmup_updates=2,
        measured_updates=4,
        validation_warmup=1,
        validation_measured=2,
        expected_source_sha256=source["sha256"],
        expected_benchmark_harness_sha256=harness["sha256"],
        expected_data_sha256=payload_sha256,
        expected_source_manifest_sha256=source_manifest_sha256,
        expected_atomic_manifest_sha256=atomic_manifest_sha256,
        expected_train_dataset_sha256=train_sha256,
        expected_validation_dataset_sha256=validation_sha256,
        expected_patcher_sha256=patcher_sha256,
        expected_model_sha256=canonical_json_sha256(model.to_dict()),
        expected_production_run_sha256=canonical_json_sha256(
            run_contract_without_identity(production)
        ),
        expected_common_run_sha256=canonical_json_sha256(
            common_run_contract(candidate_run)
        ),
        output_json=output_json,
    )
    monkeypatch.setattr(benchmark_fast_blt, "_parse_args", lambda: args)
    monkeypatch.setattr(
        benchmark_fast_blt, "benchmark_harness_provenance", lambda: harness
    )
    monkeypatch.setattr(
        benchmark_fast_blt, "require_fast_blt_allocator_environment", lambda: None
    )
    monkeypatch.setattr(benchmark_fast_blt.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark_fast_blt.torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(
        benchmark_fast_blt.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=TOTAL_MEMORY),
    )
    monkeypatch.setattr(
        benchmark_fast_blt.torch.cuda, "get_device_name", lambda _: "Test GPU"
    )
    monkeypatch.setattr(
        benchmark_fast_blt.torch, "set_float32_matmul_precision", lambda _: None
    )
    monkeypatch.setattr(
        benchmark_fast_blt.TrainingRunConfig,
        "from_env",
        classmethod(lambda cls: production),
    )
    monkeypatch.setattr(
        benchmark_fast_blt, "training_source_provenance", lambda: source
    )
    monkeypatch.setattr(
        benchmark_fast_blt, "model_config_from_env", lambda: model
    )
    monkeypatch.setattr(
        benchmark_fast_blt,
        "load_data_directory",
        lambda *args, **kwargs: (manifest, train_chunks, validation_chunks),
    )
    monkeypatch.setattr(
        benchmark_fast_blt,
        "_benchmark_candidate",
        lambda **kwargs: (
            _compile_error_candidate(
                candidate_contract, model.production_parameter_target
            )
            if failure_kind == "compile_error"
            else _validation_cache_error_candidate(
                candidate_contract, model.production_parameter_target
            )
            if failure_kind == "validation_cache_nonquiescent"
            else _telemetry_error_candidate(
                candidate_contract, model.production_parameter_target
            )
        ),
    )
    monkeypatch.setattr(
        benchmark_fast_blt, "nvidia_smi_selector", lambda _: "0000:01:00.0"
    )

    with pytest.raises(RuntimeError, match="no Fast-BLT candidate passed"):
        benchmark_fast_blt.main()

    report = json.loads(output_json.read_text())
    assert report["preset"] == preset
    assert report["model_config_sha256"] == canonical_json_sha256(model.to_dict())
    assert report["selected_microbatch"] is None
    assert report["results"]["32"]["status"] == failure_kind
    assert report["results"]["32"]["eligible"] is False


@pytest.mark.parametrize(
    "preset",
    [
        FAST_BLT_ENTROPY_B4_COMPLETE_PRESET,
        FAST_BLT_ENTROPY_B4_COMPLETE_PAPER_RATIO_PRESET,
    ],
)
def test_validate_authenticates_every_complete_identity_and_selected_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    preset: str,
) -> None:
    production = _production_run(monkeypatch, preset)
    selected_run = build_benchmark_run_config(production, 32)
    selected = _candidate(selected_run)
    required = required_headroom_bytes(TOTAL_MEMORY)
    model = fast_blt_complete_model_config(preset)
    identities = {
        "source_sha256": "1" * 64,
        "dataset_payload_sha256": "2" * 64,
        "source_manifest_sha256": "3" * 64,
        "atomic_manifest_sha256": "4" * 64,
        "train_dataset_sha256": "5" * 64,
        "validation_dataset_sha256": "6" * 64,
        "patcher_sha256": "7" * 64,
    }
    report = {
        "schema": FAST_BLT_READINESS_SCHEMA,
        "preset": preset,
        "source_sha256": identities["source_sha256"],
        "benchmark_harness_source": benchmark_harness_provenance(),
        "dataset": {
            "payload_sha256": identities["dataset_payload_sha256"],
            "source_manifest_sha256": identities["source_manifest_sha256"],
            "atomic_manifest_sha256": identities["atomic_manifest_sha256"],
            "train_dataset_sha256": identities["train_dataset_sha256"],
            "validation_dataset_sha256": identities["validation_dataset_sha256"],
            "validation_rows": 2_048,
            "diffusion_validation_rows": 256,
            "patching_policy": "causal_entropy_v1",
            "patcher_sha256": identities["patcher_sha256"],
            "max_patch_size": 8,
        },
        "model_config": model.to_dict(),
        "model_config_sha256": canonical_json_sha256(model.to_dict()),
        "production_run_contract": run_contract_without_identity(production),
        "production_run_contract_sha256": canonical_json_sha256(
            run_contract_without_identity(production)
        ),
        "common_run_contract": common_run_contract(selected_run),
        "common_run_contract_sha256": canonical_json_sha256(common_run_contract(selected_run)),
        "geometry": FastBltGeometry().contract(),
        "candidate_microbatches": [32],
        "results": {"32": selected},
        "selected_microbatch": 32,
        "headroom_policy": {
            "fraction": 0.15, "minimum_gib": 4.0, "required_bytes": required,
            "intrinsic_and_physical_required": True,
        },
        "sustained_gpu_policy": SUSTAINED_GPU_POLICY,
        "gpu": {
            "name": "test", "total_memory_bytes": TOTAL_MEMORY,
            "torch": "test", "cuda": "test", "nvidia_smi_selector": "0",
        },
        "runtime": {
            "schema": FAST_BLT_RUNTIME_SCHEMA, "compiled": True,
            "compile_dynamic_shapes": True, "device_type": "cuda", "world_size": 1,
            "validation_model_is_separate": True,
            "allocator": fast_blt_allocator_contract(),
        },
    }
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps(report))
    result = validate_fast_blt_readiness(
        path, **identities, max_patch_size=8, model_config=model,
        production_run_config=production, selected_run_config=selected_run,
        geometry=FastBltGeometry(), candidate_microbatches=(32,),
        selected_microbatch=32, verify_runtime=False,
    )
    assert result["selected_microbatch"] == 32

    parameter_drift = deepcopy(report)
    parameter_drift["results"]["32"]["parameter_count"] = (
        model.production_parameter_target + 1
    )
    path.write_text(json.dumps(parameter_drift))
    with pytest.raises(ValueError, match="provenance, contract, or telemetry"):
        validate_fast_blt_readiness(
            path, **identities, max_patch_size=8, model_config=model,
            production_run_config=production, selected_run_config=selected_run,
            geometry=FastBltGeometry(), candidate_microbatches=(32,),
            selected_microbatch=32, verify_runtime=False,
        )

    allocator_drift = deepcopy(report)
    allocator_drift["runtime"]["allocator"]["value"] = "expandable_segments:False"
    path.write_text(json.dumps(allocator_drift))
    with pytest.raises(ValueError, match="provenance, contract, or telemetry"):
        validate_fast_blt_readiness(
            path, **identities, max_patch_size=8, model_config=model,
            production_run_config=production, selected_run_config=selected_run,
            geometry=FastBltGeometry(), candidate_microbatches=(32,),
            selected_microbatch=32, verify_runtime=False,
        )

    broken = deepcopy(report)
    broken["dataset"]["train_dataset_sha256"] = "0" * 64
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="provenance, contract, or telemetry"):
        validate_fast_blt_readiness(
            path, **identities, max_patch_size=8, model_config=model,
            production_run_config=production, selected_run_config=selected_run,
            geometry=FastBltGeometry(), candidate_microbatches=(32,),
            selected_microbatch=32, verify_runtime=False,
        )


def test_allocator_environment_is_exact_and_canonical() -> None:
    expected = fast_blt_allocator_contract()
    assert require_fast_blt_allocator_environment(
        {FAST_BLT_ALLOCATOR_ENV: FAST_BLT_ALLOCATOR_VALUE}
    ) == expected

    with pytest.raises(ValueError, match=FAST_BLT_ALLOCATOR_ENV):
        require_fast_blt_allocator_environment({})
    with pytest.raises(ValueError, match="exact process environment"):
        require_fast_blt_allocator_environment(
            {FAST_BLT_ALLOCATOR_ENV: "expandable_segments:False"}
        )
    with pytest.raises(ValueError, match="must be unset"):
        require_fast_blt_allocator_environment(
            {
                FAST_BLT_ALLOCATOR_ENV: FAST_BLT_ALLOCATOR_VALUE,
                FAST_BLT_LEGACY_ALLOCATOR_ENV: FAST_BLT_ALLOCATOR_VALUE,
            }
        )
    with pytest.raises(ValueError, match=FAST_BLT_ALLOCATOR_ENV):
        require_fast_blt_allocator_environment(
            {FAST_BLT_LEGACY_ALLOCATOR_ENV: FAST_BLT_ALLOCATOR_VALUE}
        )
