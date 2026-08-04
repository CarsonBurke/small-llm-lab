from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
import time

import pytest
import torch

from postraining.runtime.profiling import (
    DEVICE_SAMPLE_FIELDS,
    INERT_PHASE,
    PROFILE_SCHEMA,
    DeviceSampler,
    DisabledProfiler,
    RunProfiler,
    SyncDetector,
    compilation_record,
    format_compile_split,
    device_timed,
    elapsed_seconds,
    format_profile_summary,
)
from postraining.vapo.config import build_arg_parser, validate_args


def _profile_args(**overrides) -> argparse.Namespace:
    defaults = {
        "profile_pools": 0,
        "profile_skip_pools": 0,
        "profile_stack": False,
        "profile_trace": False,
        "profile_sync_pools": 0,
        "profile_top_kernels": 5,
        "profile_device_interval_ms": 0,
        "profile_power_floor": 150.0,
        "profile_compile_records": 64,
        "profile_reconcile_tolerance": 0.02,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_disabled_profiler_hands_out_one_shared_inert_phase() -> None:
    """The off path must allocate nothing.

    A profiling feature that taxes real runs is worse than none, so entering
    a phase without --profile has to be one attribute lookup and two empty
    calls against a singleton, not a fresh context manager per call.
    """
    profiler = DisabledProfiler()
    assert profiler.phase("decode") is INERT_PHASE
    assert profiler.phase("refresh") is INERT_PHASE
    with profiler.phase("decode"):
        pass
    assert not profiler.enabled
    # A registered artifact must come back unwrapped, so no counting frame
    # sits in front of a compiled callable in a real run.
    def artifact() -> int:
        return 3

    assert profiler.register_artifact("replay", artifact) is artifact
    assert profiler.pool_finished(0, 1.0) == {}
    assert profiler.close() == {}


def test_phase_tree_self_times_and_remainder_close_over_the_pool(
    tmp_path,
) -> None:
    """Root self times plus the remainder must equal the pool wall time."""
    profiler = RunProfiler(tmp_path, torch.device("cpu"), _profile_args())
    profiler.pool_started(0)
    pool_started = time.perf_counter()
    with profiler.phase("collect"):
        with profiler.phase("decode"):
            time.sleep(0.02)
        with profiler.phase("decode"):
            time.sleep(0.02)
        time.sleep(0.01)
    with profiler.phase("refresh_pipeline"):
        time.sleep(0.01)
    time.sleep(0.01)
    pool = profiler.pool_finished(0, time.perf_counter() - pool_started)
    phases = {tuple(phase["path"]): phase for phase in pool["phases"]}
    assert phases[("collect", "decode")]["count"] == 2
    # The durations themselves, not just their algebra: an implementation
    # that recorded garbage timestamps would still satisfy the identity
    # below, because the remainder is defined as the difference.
    assert phases[("collect", "decode")]["wall_seconds"] == pytest.approx(
        0.04, abs=0.02
    )
    assert phases[("collect",)]["self_seconds"] == pytest.approx(
        0.01, abs=0.02
    )
    assert phases[("collect",)]["self_seconds"] == pytest.approx(
        phases[("collect",)]["wall_seconds"]
        - phases[("collect", "decode")]["wall_seconds"],
        abs=1e-9,
    )
    roots = sum(
        phase["self_seconds"] for phase in pool["phases"] if not phase["depth"]
    )
    nested = sum(
        phase["self_seconds"] for phase in pool["phases"] if phase["depth"]
    )
    assert (
        roots + nested + pool["unaccounted_seconds"]
        == pytest.approx(pool["wall_seconds"], abs=1e-6)
    )
    assert pool["unaccounted_seconds"] == pytest.approx(0.01, abs=0.02)
    # Depth-first, so a phase never prints under a parent it does not have.
    assert [tuple(phase["path"]) for phase in pool["phases"]] == [
        ("collect",),
        ("collect", "decode"),
        ("refresh_pipeline",),
    ]
    # Every profiled pool is also written where a later run can read it.
    written = json.loads((tmp_path / "profile" / "pool_0.json").read_text())
    assert written["wall_seconds"] == pool["wall_seconds"]
    profiler.close()


def test_worker_time_is_attributed_to_the_phase_it_overlaps(tmp_path) -> None:
    """The scoring worker runs concurrently with the next chunk's decode.
    Its cost is invisible as a phase — it would double count — so it is
    reported as overlap with the main-thread phase it competes with."""
    profiler = RunProfiler(tmp_path, torch.device("cpu"), _profile_args())
    profiler.pool_started(0)
    from concurrent.futures import ThreadPoolExecutor

    def worker() -> None:
        with profiler.worker_span("score"):
            time.sleep(0.04)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with profiler.phase("collect"):
            with profiler.phase("decode"):
                future = pool.submit(worker)
                time.sleep(0.06)
            future.result()
    pool_summary = profiler.pool_finished(0, 0.2)
    phases = {tuple(phase["path"]): phase for phase in pool_summary["phases"]}
    assert phases[("collect", "decode")]["worker_seconds"] == pytest.approx(
        0.04, abs=0.02
    )
    # The overlap must not inflate the tree: a worker span is not a phase.
    assert pool_summary["unaccounted_seconds"] == pytest.approx(
        0.2 - phases[("collect",)]["wall_seconds"], abs=1e-9
    )
    assert pool_summary["worker_seconds_total"] == pytest.approx(
        0.04, abs=0.02
    )
    profiler.close()


def test_pool_finished_rejects_a_pool_shorter_than_its_own_phases(
    tmp_path,
) -> None:
    """A remainder can only be nonnegative; a negative one is a bug here,
    not a finding about the run, and must not be reported as one."""
    profiler = RunProfiler(tmp_path, torch.device("cpu"), _profile_args())
    profiler.pool_started(0)
    with profiler.phase("collect"):
        time.sleep(0.02)
    with pytest.raises(RuntimeError, match="phase tree is broken"):
        profiler.pool_finished(0, 0.001)
    profiler.close()


def test_an_unbalanced_phase_stack_fails_loudly(tmp_path) -> None:
    profiler = RunProfiler(tmp_path, torch.device("cpu"), _profile_args())
    profiler.pool_started(0)
    phase = profiler.phase("collect")
    phase.__enter__()
    with pytest.raises(RuntimeError, match="still open at pool end"):
        profiler.pool_finished(0, 1.0)
    profiler.close()


def test_summary_flags_a_compiled_artifact_that_is_never_called(
    tmp_path,
) -> None:
    """A compiled artifact nothing calls still costs its compilation."""
    profiler = RunProfiler(tmp_path, torch.device("cpu"), _profile_args())
    called = profiler.register_artifact("replay", lambda value: value + 1)
    profiler.register_artifact("rollout_tail_step", lambda: None)
    assert called(1) == 2
    summary = profiler.close()
    assert summary["artifact_calls"] == {"replay": 1, "rollout_tail_step": 0}
    assert summary["dead_artifacts"] == ["rollout_tail_step"]
    assert "NEVER CALLED" in format_profile_summary(summary)
    assert json.loads(
        (tmp_path / "profile" / "profile_summary.json").read_text()
    )["dead_artifacts"] == ["rollout_tail_step"]


def test_sync_report_is_withheld_when_positive_controls_do_not_trip() -> None:
    """An empty sync result and a broken detector look identical, and an
    earlier investigation reported a false sync for exactly that reason."""
    detector = SyncDetector(torch.device("cpu"))
    detector.run_controls()
    assert not detector.controls_passed
    report = detector.report()
    assert not report["trusted"]
    assert report["sites"] == []
    # A caller that ignores the controls still gets nothing recorded.
    detector.enable()
    assert not detector._active


def test_the_trace_window_skips_the_pools_it_is_told_to(tmp_path) -> None:
    """Pool 0 carries cold compilation and allocator growth, so the default
    is to trace the pool after it, and only as many as asked for."""
    profiler = RunProfiler(
        tmp_path,
        torch.device("cpu"),
        _profile_args(profile_pools=1, profile_skip_pools=1),
    )
    profiler.pool_started(0)
    assert profiler._torch_profile is None
    assert profiler.pool_finished(0, 1.0)["kernels"] == {}
    profiler.pool_started(1)
    assert profiler._torch_profile is not None
    kernels = profiler.pool_finished(1, 1.0)["kernels"]
    # Kernel accounting without the multi-gigabyte Chrome trace, which is
    # what --profile-trace is for.
    assert "device_kernel_launches" in kernels
    assert kernels["trace"] is None
    assert not (tmp_path / "profile" / "trace_pool_1.json").exists()
    profiler.pool_started(2)
    assert profiler._torch_profile is None
    profiler.pool_finished(2, 1.0)
    profiler.close()


def test_sync_detection_runs_after_the_traced_pools(tmp_path) -> None:
    """Sync warnings and a kineto trace distort each other, so neither may
    land on the same pool."""
    profiler = RunProfiler(
        tmp_path,
        torch.device("cpu"),
        _profile_args(
            profile_pools=1, profile_skip_pools=1, profile_sync_pools=1
        ),
    )
    enabled = []
    profiler.sync_detector.enable = lambda: enabled.append(
        profiler.pool_index
    )
    for pool_index in range(4):
        profiler.pool_started(pool_index)
        profiler.pool_finished(pool_index, 1.0)
    assert enabled == [2]
    profiler.close()


def test_close_is_idempotent_because_atexit_also_calls_it(tmp_path) -> None:
    profiler = RunProfiler(tmp_path, torch.device("cpu"), _profile_args())
    assert profiler.close()["schema"]
    assert profiler.close() == {}


def test_elapsed_seconds_over_no_events_is_zero() -> None:
    assert elapsed_seconds([]) == 0.0


def _summary(**overrides) -> dict:
    summary = {
        "schema": "test",
        "pools": [],
        "compilations_outside_pools": [],
        "syncs": {
            "trusted": False,
            "reason": "controls",
            "controls": {},
            "sites": [],
            "total": 0,
        },
        "artifact_calls": {},
        "compile_seconds_by_function": {},
        "runtime_seconds_by_compile_id": {},
        "dead_artifacts": [],
        "device_sampler_error": None,
        "device_samples": 0,
    }
    summary.update(overrides)
    return summary


def _compilation(**overrides) -> dict:
    entry = {
        "compile_id": "0/0",
        # The default is a lazy backward compile: no code object, so file and
        # line stay None. ``kind`` and ``is_runtime`` must be present because
        # the billing path keys off them -- a record without them is not a
        # record compilation_record would ever produce.
        "kind": "backward",
        "function": "backward of 0/0",
        "file": None,
        "line": None,
        "cache_size": "None",
        "is_forward": False,
        "is_runtime": False,
        "recompile_reason": None,
        "seconds": 0.25,
        "dynamo_seconds": 0.0,
        "aot_seconds": 0.25,
        "inductor_seconds": 0.1,
    }
    entry.update(overrides)
    return entry


def _real_pool_keys() -> set:
    """The keys pool_finished actually emits, straight from the profiler."""
    with tempfile.TemporaryDirectory() as directory:
        profiler = RunProfiler(
            pathlib.Path(directory), torch.device("cpu"), _profile_args()
        )
        profiler.pool_started(0)
        keys = set(profiler.pool_finished(0, 1.0))
        # close() is also registered with atexit, and it writes into this
        # directory. Run it while the directory still exists; the second call
        # at interpreter shutdown is then a no-op.
        profiler.close()
        return keys


def test_summary_renders_a_backward_compilation() -> None:
    """A backward graph has no code object, so co_name, co_filename and
    co_firstlineno all come back None. Formatting must survive that: it is
    the common case, not an edge case, on any run that compiles."""
    pool = {
        "schema": PROFILE_SCHEMA,
        "pool_index": 0,
        "step": 4,
        "wall_seconds": 12.0,
        "phases": [],
        "unaccounted_seconds": 12.0,
        "unaccounted_fraction": 1.0,
        "reconciled": False,
        "compilations_before_pool": [_compilation()],
        "compilations": [],
        "compile_seconds": 0.25,
        "runtime_seconds": 0.0,
        "artifact_calls": {},
        "counters": {},
        "kernels": {},
        "allocator": {"num_alloc_retries": 2},
        "worker_seconds_total": 0.0,
    }
    # Hand-built, so it can drift from what pool_finished actually emits and
    # the renderer would then fail only in production. Assert the two agree.
    assert pool.keys() == _real_pool_keys()
    rendered = format_profile_summary(
        _summary(
            pools=[pool],
            compile_seconds_by_function={"backward of 0/0": 0.25},
        )
    )
    assert "backward of 0/0" in rendered
    assert "NOT CLOSED" in rendered
    assert "alloc_retries=2" in rendered


def test_compilations_outside_every_pool_are_still_reported(tmp_path) -> None:
    """Startup compiles before the first pool and evaluation-only modes
    never reach a pool at all; neither may vanish from the report."""
    profiler = RunProfiler(tmp_path, torch.device("cpu"), _profile_args())
    profiler._before_pool = [_compilation(function="step_core", seconds=4.0)]
    summary = profiler.close()
    assert [entry["function"] for entry in summary["compilations_outside_pools"]] == [
        "step_core"
    ]
    assert summary["compile_seconds_by_function"] == {"step_core": 4.0}
    assert "outside any pool" in format_profile_summary(summary)


def test_startup_compiles_are_not_billed_to_the_first_pool(tmp_path) -> None:
    profiler = RunProfiler(tmp_path, torch.device("cpu"), _profile_args())
    profiler.pool_started(0)
    profiler._before_pool = [_compilation(function="step_core", seconds=4.0)]
    pool = profiler.pool_finished(0, 1.0)
    assert pool["compilations"] == []
    assert [entry["function"] for entry in pool["compilations_before_pool"]] == [
        "step_core"
    ]
    assert profiler.close()["compilations_outside_pools"] == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_timed_measures_a_region_without_a_barrier() -> None:
    events: list = []
    with device_timed(events):
        torch.ones(1024, 1024, device="cuda").mul_(2.0)
    assert len(events) == 1
    assert elapsed_seconds(events) > 0.0


def _parse(*extra: str) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = build_arg_parser()
    args = parser.parse_args(
        ["--checkpoint", "checkpoint.pt", "--output", "out", *extra]
    )
    return parser, args


def test_profile_refuses_a_long_run_unless_forced() -> None:
    """A profiled run's timings are perturbed, so it must not be able to
    masquerade as the training run whose numbers get quoted."""
    parser, args = _parse("--profile", "--steps", "2000")
    with pytest.raises(SystemExit):
        validate_args(parser, args)
    parser, args = _parse("--profile", "--steps", "2000", "--profile-force")
    validate_args(parser, args)
    parser, args = _parse("--profile", "--steps", "8")
    validate_args(parser, args)


def test_profiling_is_off_by_default() -> None:
    _, args = _parse()
    assert not args.profile


def test_max_train_hours_guards() -> None:
    _, args = _parse()
    assert args.max_train_hours is None
    parser, args = _parse("--max-train-hours", "10")
    validate_args(parser, args)
    assert args.max_train_hours == 10.0
    parser, args = _parse("--max-train-hours", "0")
    with pytest.raises(SystemExit):
        validate_args(parser, args)
    # A wall-clock truncation cannot promise the one-pass prompt contract.
    parser, args = _parse(
        "--max-train-hours", "10", "--consume-all-prompts"
    )
    with pytest.raises(SystemExit):
        validate_args(parser, args)


class _Metric:
    """A CompilationMetrics stand-in carrying only the fields the record
    builder reads. Constructed rather than compiled for real because the
    combination that matters -- a RUNTIME row -- needs Triton autotuning or a
    cudagraph re-record on a GPU, and the classification is pure logic."""

    def __init__(self, **fields):
        defaults = {
            "compile_id": "0/0",
            "co_name": None,
            "co_filename": None,
            "co_firstlineno": None,
            "cache_size": None,
            "is_forward": True,
            "is_runtime": False,
            "recompile_reason": None,
            "duration_us": 250_000,
            "dynamo_cumulative_compile_time_us": None,
            "aot_autograd_cumulative_compile_time_us": None,
            "inductor_cumulative_compile_time_us": None,
            "backward_cumulative_compile_time_us": None,
            "runtime_triton_autotune_time_us": None,
            "runtime_cudagraphify_time_us": None,
        }
        self.__dict__.update({**defaults, **fields})


def test_a_runtime_row_is_not_mistaken_for_a_backward_compile() -> None:
    """Three writers append to the compilation metrics stream and only the
    forward one fills co_name, so `co_name is None` does NOT mean backward.
    A runtime row -- Triton autotuning, a cudagraph re-record -- also has no
    code object and, unlike a backward, reports is_forward True."""
    runtime = compilation_record(
        _Metric(is_runtime=True, runtime_triton_autotune_time_us=200_000)
    )
    assert runtime["kind"] == "runtime"
    assert runtime["function"] == "runtime of 0/0"
    assert runtime["runtime_autotune_seconds"] == pytest.approx(0.2)

    backward = compilation_record(_Metric(is_forward=False))
    assert backward["kind"] == "backward"
    assert backward["function"] == "backward of 0/0"

    forward = compilation_record(
        _Metric(co_name="step_core", dynamo_cumulative_compile_time_us=100_000)
    )
    assert forward["kind"] == "forward"
    assert forward["function"] == "step_core"


def test_an_unreported_sub_timing_stays_none_instead_of_zero() -> None:
    """The runtime writer fills one column and leaves the rest unset. Turning
    those into 0.0 would print a measurement where there was silence."""
    record = compilation_record(_Metric(is_runtime=True))
    assert record["dynamo_seconds"] is None
    assert record["inductor_seconds"] is None
    assert "no sub-timings reported" in format_compile_split(record)
    assert "dynamo" not in format_compile_split(record)


def test_runtime_time_is_kept_out_of_the_pools_compile_total(tmp_path) -> None:
    """Autotuning is billed against the compile id that provoked it, so
    counting it as compile time inflates the headline number and makes the
    convergence test ("no pool compiles after the first") unpassable: a
    reduce-overhead artifact can re-record its cudagraph at any point."""
    profiler = RunProfiler(tmp_path, torch.device("cpu"), _profile_args())
    profiler.pool_started(0)
    profiler._before_pool = [
        compilation_record(_Metric(co_name="step_core", duration_us=2_000_000)),
        compilation_record(_Metric(is_runtime=True, duration_us=3_000_000)),
    ]
    pool = profiler.pool_finished(0, 10.0)
    assert pool["compile_seconds"] == pytest.approx(2.0)
    assert pool["runtime_seconds"] == pytest.approx(3.0)
    summary = profiler.close()
    assert summary["compile_seconds_by_function"] == {"step_core": 2.0}
    assert summary["runtime_seconds_by_compile_id"] == {"0/0": 3.0}


def test_artifact_calls_are_per_pool_not_cumulative(tmp_path) -> None:
    """A cumulative count makes every pool look busier than the last and
    hides an artifact that stopped being called."""
    profiler = RunProfiler(tmp_path, torch.device("cpu"), _profile_args())
    counted = profiler.register_artifact("step", lambda: None)
    for expected in (2, 3):
        profiler.pool_started(0)
        for _ in range(expected):
            counted()
        assert profiler.pool_finished(0, 1.0)["artifact_calls"] == {
            "step": expected
        }
    assert profiler.close()["artifact_calls"] == {"step": 5}


def _sampler(rows: list[tuple[float, tuple[float, ...]]]) -> DeviceSampler:
    """A sampler holding a hand-built series, with no nvidia-smi behind it."""
    sampler = DeviceSampler(250, torch.device("cpu"))
    sampler.samples = rows
    return sampler


def _series(
    count: int, interval: float, power, clocks=2800.0, start: float = 100.0
) -> list[tuple[float, tuple[float, ...]]]:
    return [
        (
            start + index * interval,
            (
                90.0,
                40.0,
                power(index) if callable(power) else power,
                clocks,
                1024.0,
            ),
        )
        for index in range(count)
    ]


def test_polling_faster_than_the_driver_refreshes_counts_one_reading() -> None:
    """Power latches every 500 ms against a 250 ms poll, so half the rows
    are copies. Counting the copies weights one measurement as two."""
    sampler = _sampler(_series(40, 0.25, lambda index: 300.0 + index // 2))
    assert sampler.period_ms("power_draw_watts") == pytest.approx(500, rel=0.1)
    readings = sampler._readings("power_draw_watts", [(100.0, 110.0)])
    # 41 samples span the window; every second one is a copy.
    assert 18 <= len(readings) <= 21
    assert [value for _, value in readings] == sorted(
        {value for _, value in readings}
    )
    assert sum(covered for covered, _ in readings) <= 10.0


def test_a_phase_shorter_than_the_refresh_withholds_that_field() -> None:
    """The failure this exists to stop: a phase containing one sample
    reported a mean and a minimum that were the same single sample."""
    sampler = _sampler(_series(40, 0.25, lambda index: 300.0 + index // 2))
    # A full second, so four samples fall inside it: enough to print a
    # mean and a minimum under a rule that only counts samples.
    window = [(100.6, 101.6)]
    assert len([1 for stamp, _ in sampler.samples if 100.6 <= stamp <= 101.6]) == 4
    stats = sampler.window(window, power_floor=150.0)
    assert "power_draw_watts_mean" not in stats
    assert "power_draw_watts" in stats.get("withheld", [])


def test_a_value_latched_before_the_phase_is_not_attributed_to_it() -> None:
    """A reading stamped t describes [t - period, t]. Admitting it for a
    phase that started at t - 10 ms credits the phase with a collapse that
    happened before it began."""
    rows = _series(80, 0.25, lambda index: 90.0 if index <= 40 else 350.0)
    sampler = _sampler(rows)
    # Starts just before the last 90 W sample, so a rule that only asks
    # whether the stamp is inside the window reports a 90 W minimum for a
    # phase during which the GPU never drew 90 W.
    started = rows[40][0] - 0.01
    naive = [
        values[DEVICE_SAMPLE_FIELDS.index("power_draw_watts")]
        for stamp, values in rows
        if started <= stamp <= started + 5.0
    ]
    assert min(naive) == pytest.approx(90.0)
    stats = sampler.window([(started, started + 5.0)], power_floor=150.0)
    assert stats["power_draw_watts_min"] == pytest.approx(350.0)


def test_a_pinned_clock_still_reports() -> None:
    """A field that never moves cannot be misattributed, so withholding it
    would cost the reader the one number that says the clock held."""
    sampler = _sampler(_series(40, 0.25, 320.0, clocks=2800.0))
    stats = sampler.window([(100.0, 110.0)], power_floor=150.0)
    assert stats["clocks_sm_mhz_min"] == pytest.approx(2800.0)
    assert stats["clocks_sm_mhz_readings"] > 3


def test_seconds_below_the_floor_use_the_measured_period() -> None:
    """Charging a low reading the poll interval halves a collapse that
    lasted a full refresh."""
    sampler = _sampler(_series(40, 0.25, lambda index: 90.0 + index // 2))
    stats = sampler.window([(100.0, 110.0)], power_floor=100.0)
    # Nine readings at 91-99 W, one per 500 ms refresh. Not ten: the two
    # samples at the very start of the window were latched before it and
    # the formation rule drops them, which is the point of that rule.
    assert stats["seconds_below_power_floor"] == pytest.approx(4.5)
    assert stats["power_draw_watts_period_ms"] == pytest.approx(500.0)


def test_seconds_below_the_floor_cannot_exceed_the_phase() -> None:
    """Charging every reading a whole period counted 7 s of collapse
    inside a 4 s phase, because a changed value is admitted early."""
    quiet = _series(200, 0.25, lambda index: 300.0 + index // 2)
    busy = _series(16, 0.25, lambda index: 50.0 + index, start=quiet[-1][0] + 0.25)
    sampler = _sampler(quiet + busy)
    window = (busy[0][0] - 0.6, busy[-1][0])
    stats = sampler.window([window], power_floor=100.0)
    assert stats["seconds_below_power_floor"] <= window[1] - window[0]


def test_a_period_measured_over_pool_zero_does_not_freeze_the_run() -> None:
    """Pool 0 idles through cold compilation, which reads as a quiet
    field. Freezing that answer sets the run's resolution from its worst
    window and lets every later pool report stale readings."""
    idle = _series(80, 0.25, 320.0)
    busy = _series(240, 0.25, lambda index: 300.0 + index // 2, start=idle[-1][0] + 0.25)
    sampler = _sampler(idle)
    assert sampler.period_ms("power_draw_watts") == pytest.approx(250.0)
    sampler.samples = idle + busy
    assert sampler.period_ms("power_draw_watts") == pytest.approx(500.0)


def test_a_delivered_gap_wider_than_the_request_is_believed() -> None:
    """-lms is a request. A poller delivering a line a second apart must
    not have its readings credited with a quarter second of coverage."""
    sampler = _sampler(_series(40, 1.0, lambda index: 300.0 + index))
    assert sampler.spacing_ms() == pytest.approx(1000.0)
    assert sampler.period_ms("power_draw_watts") >= 1000.0


def test_the_period_estimate_is_bounded_above_the_poll_rate() -> None:
    """A field that moves in bins rather than continuously has long gaps
    between changes, and an unbounded median would withhold every phase
    shorter than one of those quiet stretches."""
    sampler = _sampler(_series(400, 0.25, lambda index: 300.0 + index // 30))
    assert sampler.period_ms("power_draw_watts") <= 4 * 250.0


def test_readings_are_thinned_across_every_occurrence_of_one_phase() -> None:
    """A phase that runs four times is one distribution, and the thinning
    clock has to survive the gaps between its occurrences."""
    sampler = _sampler(_series(120, 0.25, lambda index: 300.0 + index // 2))
    windows = [(100.0 + 5.0 * turn, 102.0 + 5.0 * turn) for turn in range(4)]
    readings = sampler._readings("power_draw_watts", windows)
    # Four occurrences, each 2 s long and each losing its first 500 ms to
    # the formation rule, at a 500 ms refresh.
    assert 12 <= len(readings) <= 16
    assert sum(covered for covered, _ in readings) <= sum(
        ended - started for started, ended in windows
    )


def _phase(name: str, spans: list[tuple[float, float]]) -> dict:
    wall = sum(ended - started for started, ended in spans)
    return {
        "path": [name],
        "name": name,
        "depth": 0,
        "count": len(spans),
        "wall_seconds": wall,
        "self_seconds": wall,
        "device_seconds": None,
        "worker_seconds": 0.0,
        "windows": [list(span) for span in spans],
        "device": {},
    }


def _rendered_pool(phases: list[dict]) -> str:
    pool = {
        "schema": PROFILE_SCHEMA,
        "pool_index": 0,
        "step": 4,
        "wall_seconds": 40.0,
        "phases": phases,
        "unaccounted_seconds": 0.0,
        "unaccounted_fraction": 0.0,
        "reconciled": True,
        "compilations_before_pool": [],
        "compilations": [],
        "compile_seconds": 0.0,
        "runtime_seconds": 0.0,
        "artifact_calls": {},
        "counters": {},
        "kernels": {},
        "allocator": {},
        "worker_seconds_total": 0.0,
    }
    assert pool.keys() == _real_pool_keys()
    return format_profile_summary(_summary(pools=[pool]))


def test_one_stalled_occurrence_is_named_not_averaged_away() -> None:
    """The stall this exists for: three refreshes near two seconds and one
    at sixteen sum to a row that reads as a slow pool, not as an event."""
    rendered = _rendered_pool(
        [
            _phase(
                "refresh",
                [(0.0, 2.1), (3.0, 5.2), (6.0, 8.3), (9.0, 25.2)],
            )
        ]
    )
    assert "uneven: refresh longest occurrence 16.200 s" in rendered
    assert "started 9.000" in rendered


def test_an_evenly_paced_phase_is_not_flagged() -> None:
    """A phase whose occurrences all cost the same is the normal case and
    a line about it every pool would train the reader to skip the section."""
    rendered = _rendered_pool(
        [_phase("decode", [(0.0, 4.0), (5.0, 9.1), (10.0, 13.9)])]
    )
    assert "uneven:" not in rendered


def test_a_phase_that_ran_twice_and_stalled_once_is_flagged() -> None:
    """Two occurrences, 0.7 s and 16 s, is the clearest stall a pool can
    hold. Comparing the longest against the mean of all of them buries it:
    no span can be twice a mean it is half of."""
    rendered = _rendered_pool([_phase("refresh", [(0.0, 0.7), (1.0, 17.0)])])
    assert "uneven: refresh longest occurrence 16.000 s" in rendered


def test_a_withheld_field_renders_as_absent_not_as_zero() -> None:
    """A phase can keep its clocks and lose its power. The row must show
    the clock and a dash, not a clock and a zero watt reading."""
    phase = _phase("refresh", [(0.0, 4.0)])
    phase["device"] = {
        "clocks_sm_mhz_min": 2820.0,
        "clocks_sm_mhz_readings": 8,
        "withheld": ["power_draw_watts"],
    }
    row = next(
        line
        for line in _rendered_pool([phase]).splitlines()
        if line.strip().startswith("refresh")
    )
    assert "2820" in row
    assert row.split()[-5:-1].count("-") == 3


def test_graph_decode_scheduler_gating() -> None:
    # continuous_refill captures its paged step directly: no extra flag.
    parser, args = _parse(
        "--no-delightful-policy-gradient", "--prompts-per-rollout", "64",
        "--prompts-per-minibatch", "16",
        "--rollout-scheduler", "continuous_refill", "--rollout-graph-decode"
    )
    validate_args(parser, args)
    assert args.rollout_graph_decode
    # The lockstep scheduler's capture rides the flex-decode static arena;
    # asking for graphs without it must fail at config time, naming the
    # conflict, rather than degrade at runtime.
    parser, args = _parse("--rollout-graph-decode")
    with pytest.raises(SystemExit):
        validate_args(parser, args)
