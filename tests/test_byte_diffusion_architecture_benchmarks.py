from __future__ import annotations

import argparse
from contextlib import nullcontext
from array import array

import pytest
import torch

from scripts.benchmark_byte_diffusion_architectures import (
    GIB,
    MATCHED_GLOBAL_BATCH,
    PRODUCTION_ROW_LENGTH,
    _summary,
    _meets_sustained_gpu_policy,
    choose_fastest_fitting,
    accumulate_target_counts,
    parse_candidate_microbatches,
    readiness_workload_contract,
    required_headroom_bytes,
    tail_geometry,
)
from pretraining.byte_diffusion.readiness import (
    diagnostic_cadence_contract,
    duo_geometry_contract,
    materialize_training_diagnostics,
)
from pretraining.byte_diffusion.training_duo import (
    JOINT_DUO_CLEAN_AR_OBJECTIVE,
    PURE_DUO_OBJECTIVE,
)
from pretraining.byte_diffusion import pipeline
from pretraining.byte_diffusion.pipeline import (
    DeviceBatchPrefetcher,
    prefetched_device_batches,
)
from pretraining.byte_diffusion.telemetry import nvidia_smi_selector


class _CpuTransferable:
    def __init__(self, value: int) -> None:
        self.value = value

    def pin_memory(self):
        raise AssertionError("CPU pipeline must not pin")

    def to(self, device, *, non_blocking=False):
        raise AssertionError("CPU pipeline must not transfer")

    def record_stream(self, stream):
        raise AssertionError("CPU pipeline must not record CUDA streams")


def test_candidate_parser_returns_unique_ascending_tuple() -> None:
    assert parse_candidate_microbatches("8, 2,4") == (2, 4, 8)
    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        parse_candidate_microbatches("2,2")
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        parse_candidate_microbatches("0,2")
    with pytest.raises(argparse.ArgumentTypeError, match="comma-separated"):
        parse_candidate_microbatches("two")


def test_cpu_batch_pipeline_preserves_lazy_input_order() -> None:
    prepared: list[int] = []

    def prepare(value: int) -> _CpuTransferable:
        prepared.append(value)
        return _CpuTransferable(value * 2)

    observed = tuple(
        item.value
        for item in prefetched_device_batches(
            (3, 1, 4), prepare, device=torch.device("cpu")
        )
    )
    assert observed == (6, 2, 8)
    assert prepared == [3, 1, 4]


def test_cuda_batch_prefetcher_reuses_one_worker_and_stream(monkeypatch) -> None:
    events: list[tuple[object, ...]] = []
    resource_counts = {"executor": 0, "stream": 0, "shutdown": 0}

    class ImmediateFuture:
        def __init__(self, value) -> None:
            self.value = value

        def result(self):
            return self.value

        def done(self) -> bool:
            return True

        def cancel(self) -> bool:
            return False

    class ImmediateExecutor:
        def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
            resource_counts["executor"] += 1
            assert max_workers == 1
            assert thread_name_prefix == "byte-device-prefetch"

        def submit(self, function, key):
            events.append(("submit", key))
            return ImmediateFuture(function(key))

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            resource_counts["shutdown"] += 1
            assert wait and cancel_futures

    class FakeTransferStream:
        pass

    class FakeExecutionStream:
        def wait_event(self, event) -> None:
            events.append(("wait", event.key))

    class FakeEvent:
        def __init__(self) -> None:
            self.key = -1

        def record(self, stream) -> None:
            assert stream is transfer_stream
            self.key = events[-1][1]
            events.append(("event", self.key))

    class FakeBatch:
        def __init__(self, value: int) -> None:
            self.value = value

        def pin_memory(self):
            events.append(("pin", self.value))
            return self

        def to(self, device, *, non_blocking=False):
            assert device == torch.device("cuda")
            assert non_blocking
            events.append(("copy", self.value))
            return self

        def record_stream(self, stream) -> None:
            assert stream is execution_stream
            events.append(("record", self.value))

    transfer_stream = FakeTransferStream()
    execution_stream = FakeExecutionStream()

    def make_stream(*, device):
        assert device == torch.device("cuda")
        resource_counts["stream"] += 1
        return transfer_stream

    monkeypatch.setattr(pipeline, "ThreadPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(torch.cuda, "Stream", make_stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: execution_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)

    with DeviceBatchPrefetcher(device=torch.device("cuda")) as prefetcher:
        first_update = prefetcher.batches((1, 2, 3), FakeBatch)
        assert next(first_update).value == 1
        assert ("pin", 2) in events
        assert ("copy", 2) in events
        assert ("pin", 3) in events
        assert ("copy", 3) not in events
        events.append(("consumer_compute", 1))
        assert [batch.value for batch in first_update] == [2, 3]
        assert [batch.value for batch in prefetcher.batches((4, 5), FakeBatch)] == [
            4,
            5,
        ]
        assert resource_counts == {"executor": 1, "stream": 1, "shutdown": 0}

    assert resource_counts == {"executor": 1, "stream": 1, "shutdown": 1}
    assert events.index(("copy", 2)) < events.index(("consumer_compute", 1))
    assert [event for event in events if event[0] == "record"] == [
        ("record", 1),
        ("record", 2),
        ("record", 3),
        ("record", 4),
        ("record", 5),
    ]
    with pytest.raises(RuntimeError, match="closed"):
        prefetcher.batches((6,), FakeBatch)


def test_closing_prefetcher_closes_suspended_cpu_iterator() -> None:
    prefetcher = DeviceBatchPrefetcher(device=torch.device("cpu"))
    batches = prefetcher.batches((1, 2), _CpuTransferable)
    assert next(batches).value == 1
    prefetcher.close()
    assert tuple(batches) == ()
    with pytest.raises(RuntimeError, match="closed"):
        prefetcher.batches((3,), _CpuTransferable)


def test_exact_update_global_tail_geometry_is_reported() -> None:
    assert PRODUCTION_ROW_LENGTH == 8_192
    assert MATCHED_GLOBAL_BATCH == 249
    assert tail_geometry(249, 24) == (11, 9)
    assert tail_geometry(249, 8) == (32, 1)
    assert tail_geometry(249, 1) == (249, 1)


def test_headroom_policy_uses_stricter_absolute_or_fractional_floor() -> None:
    assert required_headroom_bytes(32 * GIB, fraction=0.15, minimum_gib=4) == (
        5_153_960_756
    )
    assert required_headroom_bytes(16 * GIB, fraction=0.15, minimum_gib=4) == 4 * GIB
    with pytest.raises(ValueError, match="headroom"):
        required_headroom_bytes(32 * GIB, fraction=1.0, minimum_gib=4)


def test_selector_ignores_oom_headroom_and_graph_failures() -> None:
    results = {
        "1": {"status": "ok", "eligible": True, "update_ms": 100.0},
        "2": {"status": "oom", "eligible": False},
        "4": {"status": "ok", "eligible": False, "update_ms": 40.0},
        "8": {"status": "ok", "eligible": True, "update_ms": 80.0},
    }
    assert choose_fastest_fitting(results) == 8
    assert choose_fastest_fitting({"2": results["2"]}) is None


def test_selector_prefers_larger_microbatch_on_exact_time_tie() -> None:
    results = {
        "4": {"status": "ok", "eligible": True, "update_ms": 50.0},
        "8": {"status": "ok", "eligible": True, "update_ms": 50.0},
    }
    assert choose_fastest_fitting(results) == 8


def test_telemetry_summary_does_not_require_python_sample_lists() -> None:
    assert _summary(array("d")) == {
        "count": 0,
        "mean": None,
        "p10": None,
        "peak": None,
    }


def test_sustained_gpu_policy_rejects_low_mean_or_low_tail() -> None:
    passing_power = {"count": 3, "mean": 510.0, "p10": 470.0}
    passing_util = {"count": 3, "mean": 97.0, "p10": 91.0}
    policy = {
        "minimum_mean_power_w": 500.0,
        "minimum_p10_power_w": 450.0,
        "minimum_mean_utilization": 90.0,
        "minimum_p10_utilization": 80.0,
    }
    assert _meets_sustained_gpu_policy(passing_power, passing_util, **policy)
    assert not _meets_sustained_gpu_policy(
        {"mean": 499.0, "p10": 470.0}, passing_util, **policy
    )
    assert not _meets_sustained_gpu_policy(
        passing_power, {"mean": 97.0, "p10": 79.0}, **policy
    )
    assert not _meets_sustained_gpu_policy(
        {**passing_power, "count": 2}, passing_util, **policy
    )
    assert _summary(array("d", (100.0, 300.0))) == {
        "count": 2,
        "mean": 200.0,
        "p10": 120.0,
        "peak": 300.0,
    }


def test_readiness_workloads_authenticate_nonlogging_diagnostic_cadence() -> None:
    expected = diagnostic_cadence_contract()
    for architecture in ("idlm", "diffusion_gemma", "duo"):
        _, workload = readiness_workload_contract(architecture)
        assert workload["diagnostic_cadence"] == expected
        if architecture == "diffusion_gemma":
            assert workload["runtime_candidate_validation"] is False
    assert expected["readiness_update_class"] == (
        "non_logging_non_validation_steady_state"
    )
    assert expected["readiness_materializes_diagnostics"] is False

    _, pure = readiness_workload_contract(
        "duo", duo_objective=PURE_DUO_OBJECTIVE
    )
    _, joint = readiness_workload_contract(
        "duo", duo_objective=JOINT_DUO_CLEAN_AR_OBJECTIVE
    )
    assert pure["clean_ar_weight"] == 0.0
    assert joint["clean_ar_weight"] == 1.0
    assert pure["objective"] != joint["objective"]
    assert pure["duo_nelbo_reduction"] == joint["duo_nelbo_reduction"]


def test_duo_readiness_geometry_changes_training_only() -> None:
    _, reference = readiness_workload_contract(
        "duo", duo_canvas_length=512, duo_branches=8
    )
    _, short = readiness_workload_contract(
        "duo", duo_canvas_length=256, duo_branches=15
    )
    assert reference["training_geometry"] == {
        "canvas_length": 512,
        "branches": 8,
    }
    assert short["training_geometry"] == {
        "canvas_length": 256,
        "branches": 15,
    }
    for fixed in ("canonical_validation_geometry", "dataset_geometry"):
        assert short[fixed] == reference[fixed]
    assert short["dataset_geometry"] == {
        "required_branch_bytes": 4_096,
        "branch_span_length": 512,
    }
    with pytest.raises(ValueError, match="unsupported production"):
        duo_geometry_contract(256, 16)


def test_production_diagnostics_materialize_only_on_declared_cadence() -> None:
    cadence = {"log_every": 10, "validation_every": 20}
    assert not materialize_training_diagnostics(1, 25, **cadence)
    assert materialize_training_diagnostics(10, 25, **cadence)
    assert materialize_training_diagnostics(20, 25, **cadence)
    assert materialize_training_diagnostics(25, 25, **cadence)
    with pytest.raises(ValueError, match="intervals"):
        diagnostic_cadence_contract(log_every=0)


def test_benchmark_target_accumulation_never_materializes_tensor_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_scalar_materialization(_tensor: torch.Tensor):
        raise AssertionError("benchmark materialized a per-update tensor scalar")

    monkeypatch.setattr(torch.Tensor, "__int__", reject_scalar_materialization)
    monkeypatch.setattr(torch.Tensor, "__float__", reject_scalar_materialization)
    primary = torch.zeros((), dtype=torch.long)
    anchor = torch.zeros_like(primary)
    accumulate_target_counts(
        primary,
        anchor,
        (torch.tensor(7), torch.tensor(11)),
    )
    assert torch.equal(primary, torch.tensor(7))
    assert torch.equal(anchor, torch.tensor(11))


def test_nvidia_smi_selector_maps_logical_to_visible_physical_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,GPU-abc")
    assert nvidia_smi_selector(torch.device("cuda", 0)) == "4"
    assert nvidia_smi_selector(torch.device("cuda", 1)) == "GPU-abc"


def test_nvidia_smi_selector_resolves_implicit_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,GPU-abc")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert nvidia_smi_selector(torch.device("cuda")) == "GPU-abc"


def test_nvidia_smi_selector_resolves_implicit_device_without_visibility_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert nvidia_smi_selector(torch.device("cuda")) == "2"


def test_nvidia_smi_selector_does_not_resolve_explicit_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def fail() -> int:
        raise AssertionError("explicit CUDA device must not query the current device")

    monkeypatch.setattr(torch.cuda, "current_device", fail)
    assert nvidia_smi_selector(torch.device("cuda", 0)) == "0"


def test_nvidia_smi_selector_rejects_disabled_or_missing_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    with pytest.raises(ValueError, match="disables"):
        nvidia_smi_selector(torch.device("cuda", 0))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(ValueError, match="absent"):
        nvidia_smi_selector(torch.device("cuda", 1))
    with pytest.raises(ValueError, match="CUDA device"):
        nvidia_smi_selector(torch.device("cpu"))
