from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pretraining.byte_diffusion.config import ByteDiffusionConfig
from pretraining.byte_diffusion.readiness import (
    DUO_DISTRIBUTED_READINESS_SCHEMA,
    HEADROOM_FRACTION,
    HEADROOM_MINIMUM_GIB,
    SUSTAINED_GPU_POLICY,
    _distributed_aggregate,
    validate_duo_distributed_readiness,
)
from pretraining.byte_diffusion.training_duo import (
    JOINT_DUO_CLEAN_AR_OBJECTIVE,
    PURE_DUO_OBJECTIVE,
)
from scripts.benchmark_byte_duo_distributed import (
    benchmark_source_provenance,
    distributed_readiness_geometry,
    distributed_readiness_workload_contract,
)
from scripts.train_byte_duo import distributed_rank_positions, source_provenance


GIB = 1 << 30


def test_exact_249_row_rotation_has_one_32_row_rank_and_one_backward() -> None:
    observed = []
    for rotation in range(8):
        counts = tuple(
            distributed_rank_positions(
                249, rank=rank, world_size=8, rotation=rotation
            ).size
            for rank in range(8)
        )
        assert sum(counts) == 249
        assert counts.count(32) == 1
        assert counts.count(31) == 7
        assert all(-(-count // 32) == 1 for count in counts)
        observed.append(counts.index(32))
    assert observed == list(range(8))
    assert distributed_readiness_geometry()["validation_local_rows"] == 32


def test_distributed_readiness_binds_named_objective_and_fixed_weight() -> None:
    pure = distributed_readiness_workload_contract(objective=PURE_DUO_OBJECTIVE)
    joint = distributed_readiness_workload_contract(
        objective=JOINT_DUO_CLEAN_AR_OBJECTIVE
    )
    assert pure["clean_ar_weight"] == 0.0
    assert joint["clean_ar_weight"] == 1.0
    assert pure["objective"] != joint["objective"]
    assert pure["clean_ar_reduction"] == joint["clean_ar_reduction"]


def test_distributed_readiness_parameterizes_only_training_geometry() -> None:
    reference = distributed_readiness_workload_contract(
        canvas_length=512, branches=8
    )
    short = distributed_readiness_workload_contract(
        canvas_length=256, branches=15
    )
    assert short["training_geometry"] == {
        "canvas_length": 256,
        "branches": 15,
    }
    assert (
        short["canonical_validation_geometry"]
        == reference["canonical_validation_geometry"]
        == {"canvas_length": 512, "branches": 8}
    )
    assert short["dataset_geometry"] == reference["dataset_geometry"]


def _telemetry() -> dict[str, dict[str, float | int]]:
    return {
        "power_w": {"count": 8, "mean": 550.0, "p10": 510.0, "peak": 600.0},
        "gpu_utilization_percent": {
            "count": 8,
            "mean": 98.0,
            "p10": 95.0,
            "peak": 100.0,
        },
    }


def _rank_record(rank: int, *, training_elapsed: float | None = None) -> dict[str, object]:
    total_memory = 80 * GIB
    peak_reserved = 60 * GIB
    warmup, measured = 8, 8
    rotations = list(range(warmup + measured))
    rows = [32 if (rank - rotation) % 8 == 0 else 31 for rotation in rotations]
    local_target_counts = [100 + rank] * measured
    global_denominators = [828] * measured
    local_clean_ar_counts = [200 + rank] * measured
    global_clean_ar_denominators = [1_628] * measured
    phase_common = {
        **_telemetry(),
        "cuda_peak_allocated_bytes": peak_reserved - GIB,
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_reserved_headroom_bytes": total_memory - peak_reserved,
        "required_headroom_bytes": 12 * GIB,
        "warmup_unique_graphs": 2,
        "warmup_recompiles": 0,
        "warmup_graph_breaks": 0,
        "measured_unique_graphs": 0,
        "measured_recompiles": 0,
        "measured_graph_breaks": 0,
        "eligible": True,
    }
    elapsed = training_elapsed if training_elapsed is not None else 8.0 + rank / 10
    return {
        "rank": rank,
        "local_rank": rank,
        "gpu": {
            "rank": rank,
            "local_rank": rank,
            "device_index": rank,
            "name": "NVIDIA H100 80GB HBM3",
            "total_memory_bytes": total_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nvidia_smi_selector": str(rank),
        },
        "training": {
            "global_batch": 249,
            "microbatch": 32,
            "world_size": 8,
            "warmup_updates": warmup,
            "measured_updates": measured,
            "rotations": rotations,
            "local_rows": rows,
            "backward_calls": [1] * len(rotations),
            "global_denominator_allreduce": True,
            "clean_ar_global_denominator_allreduce": True,
            "ddp_world_size_loss_scale": 8.0,
            "measured_local_target_counts": local_target_counts,
            "measured_global_denominators": global_denominators,
            "measured_local_clean_ar_target_counts": local_clean_ar_counts,
            "measured_global_clean_ar_denominators": (
                global_clean_ar_denominators
            ),
            "elapsed_seconds": elapsed,
            "update_ms": 1_000.0 * elapsed / measured,
            **phase_common,
        },
        "validation": {
            "global_rows": 256,
            "local_rows": 32,
            "row_ids": distributed_rank_positions(
                256, rank=rank, world_size=8
            ).tolist(),
            "batch_size": 32,
            "world_size": 8,
            "warmup_repetitions": 2,
            "measured_repetitions": 8,
            "global_statistics_allreduce": True,
            "elapsed_seconds": 4.0 + rank / 10,
            "global_statistics_sum": [1.0] * 6,
            **phase_common,
        },
        "eligible": True,
    }


def _valid_report() -> dict[str, object]:
    ranks = [_rank_record(rank) for rank in range(8)]
    return {
        "schema": DUO_DISTRIBUTED_READINESS_SCHEMA,
        "architecture": "duo",
        "evidence_kind": (
            "exact_distributed_training_and_validation_systems_benchmark"
        ),
        "recipe_source": source_provenance(),
        "benchmark_source": benchmark_source_provenance(),
        "dataset_payload_sha256": "d" * 64,
        "model_config": ByteDiffusionConfig().to_dict(),
        "parameter_count": 1,
        "workload": distributed_readiness_workload_contract(),
        "geometry": distributed_readiness_geometry(),
        "runtime": {
            "compiled": True,
            "distributed_backend": "nccl",
            "launcher": "torchrun",
            "model_wrapper": "DDP(torch.compile(DuoModel))",
            "world_size": 8,
        },
        "sustained_gpu_policy": SUSTAINED_GPU_POLICY,
        "headroom_policy": {
            "fraction": HEADROOM_FRACTION,
            "minimum_gib": HEADROOM_MINIMUM_GIB,
        },
        "ranks": ranks,
        "aggregate": _distributed_aggregate(ranks),
        "eligible": True,
    }


def test_slowest_rank_aggregation_controls_cluster_throughput() -> None:
    ranks = [_rank_record(rank, training_elapsed=8.0 + rank) for rank in range(8)]
    aggregate = _distributed_aggregate(ranks)
    assert aggregate["slowest_training_rank"] == 7
    assert aggregate["slowest_training_elapsed_seconds"] == 15.0
    assert aggregate["update_ms_slowest_rank"] == 1_875.0
    assert aggregate["global_rows_per_second"] == pytest.approx(8 * 249 / 15.0)
    assert aggregate["measured_global_target_counts"] == [828] * 8
    assert aggregate["measured_global_targets"] == 8 * 828
    assert aggregate["measured_global_clean_ar_target_counts"] == [1_628] * 8
    assert aggregate["measured_global_clean_ar_targets"] == 8 * 1_628
    ranks[3]["eligible"] = False
    assert _distributed_aggregate(ranks)["all_ranks_eligible"] is False


def _mock_live_rank(monkeypatch: pytest.MonkeyPatch, *, rank: int = 3) -> None:
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: rank)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda _: "NVIDIA H100 80GB HBM3"
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=80 * GIB),
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "nccl")
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.write_text(json.dumps(report, sort_keys=True))


def test_validator_authenticates_calling_rank_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _valid_report()
    path = tmp_path / "distributed.json"
    _write_report(path, report)
    _mock_live_rank(monkeypatch)
    evidence = validate_duo_distributed_readiness(
        path,
        source_sha256=str(report["recipe_source"]["sha256"]),  # type: ignore[index]
        dataset_payload_sha256="d" * 64,
        model_config=ByteDiffusionConfig().to_dict(),
        parameter_count=1,
        workload=distributed_readiness_workload_contract(),
        rank=3,
    )
    assert evidence["rank"] == 3
    assert evidence["world_size"] == 8


@pytest.mark.parametrize(
    "mutate",
    (
        lambda report: report["ranks"][2]["training"]["local_rows"].__setitem__(2, 31),
        lambda report: report["ranks"][5]["training"]["backward_calls"].__setitem__(1, 2),
        lambda report: report["ranks"][6]["training"]["power_w"].__setitem__("mean", 424.0),
        lambda report: report["aggregate"].__setitem__("slowest_training_rank", 0),
    ),
)
def test_validator_rejects_tampered_rank_or_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate
) -> None:
    report = copy.deepcopy(_valid_report())
    mutate(report)
    path = tmp_path / "tampered.json"
    _write_report(path, report)
    _mock_live_rank(monkeypatch)
    with pytest.raises(ValueError, match="telemetry, provenance, or geometry"):
        validate_duo_distributed_readiness(
            path,
            source_sha256=str(report["recipe_source"]["sha256"]),  # type: ignore[index]
            dataset_payload_sha256="d" * 64,
            model_config=ByteDiffusionConfig().to_dict(),
            parameter_count=1,
            workload=distributed_readiness_workload_contract(),
            rank=3,
        )


def test_validator_rejects_non_nccl_live_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _valid_report()
    path = tmp_path / "distributed.json"
    _write_report(path, report)
    _mock_live_rank(monkeypatch)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")
    with pytest.raises(ValueError, match="live 8-rank NCCL"):
        validate_duo_distributed_readiness(
            path,
            source_sha256=str(report["recipe_source"]["sha256"]),  # type: ignore[index]
            dataset_payload_sha256="d" * 64,
            model_config=ByteDiffusionConfig().to_dict(),
            parameter_count=1,
            workload=distributed_readiness_workload_contract(),
            rank=3,
        )


def test_validator_rejects_a_duplicated_allreduced_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _valid_report()
    report["ranks"][0]["training"]["measured_global_denominators"][0] *= 8  # type: ignore[index]
    path = tmp_path / "duplicated-denominator.json"
    _write_report(path, report)
    _mock_live_rank(monkeypatch)
    with pytest.raises(ValueError, match="global denominator does not equal local targets"):
        validate_duo_distributed_readiness(
            path,
            source_sha256=str(report["recipe_source"]["sha256"]),  # type: ignore[index]
            dataset_payload_sha256="d" * 64,
            model_config=ByteDiffusionConfig().to_dict(),
            parameter_count=1,
            workload=distributed_readiness_workload_contract(),
            rank=3,
        )


def test_validator_rejects_a_mismatched_clean_ar_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _valid_report()
    report["ranks"][0]["training"]["measured_global_clean_ar_denominators"][
        0
    ] += 1  # type: ignore[index]
    path = tmp_path / "bad-clean-ar-denominator.json"
    _write_report(path, report)
    _mock_live_rank(monkeypatch)
    with pytest.raises(
        ValueError,
        match="global clean-AR denominator does not equal local targets",
    ):
        validate_duo_distributed_readiness(
            path,
            source_sha256=str(report["recipe_source"]["sha256"]),  # type: ignore[index]
            dataset_payload_sha256="d" * 64,
            model_config=ByteDiffusionConfig().to_dict(),
            parameter_count=1,
            workload=distributed_readiness_workload_contract(),
            rank=3,
        )
