"""Optional, low-overhead runtime profiling for latent VAPO training."""

from __future__ import annotations

import argparse
import atexit
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import threading
import time
import warnings

import torch


PROFILE_SCHEMA = "latent_vapo_profile_v1"

# nvidia-smi query order. Power leads because a power collapse is the
# user-visible symptom; the SM clock separates an idle GPU from a throttled
# one at the same reported utilization.
DEVICE_SAMPLE_FIELDS = (
    "utilization_gpu_percent",
    "utilization_memory_percent",
    "power_draw_watts",
    "clocks_sm_mhz",
    "memory_used_mib",
)

# Host-side launch calls, as kineto names the CUPTI runtime and driver
# callbacks. Triton launches every compiled kernel through the DRIVER entry
# point cuLaunchKernelEx, so matching only the cudaLaunchKernel family would
# miss most of an Inductor-compiled decode loop's dispatches — which is the
# one number the launch-bound decode loop is diagnosed by.
LAUNCH_EVENT_PREFIXES = (
    "cudaLaunchKernel",
    "cudaLaunchCooperativeKernel",
    "cuLaunchKernel",
    "cuLaunchCooperativeKernel",
)


def compilation_record(metric) -> dict:
    """Turn one Dynamo CompilationMetrics row into a reportable record.

    Three different things append to that stream and only one of them is a
    frame compilation:

    * a forward compile, from ``_dynamo/convert_frame.py``, which is the
      only writer that fills ``co_name``, and hardcodes ``is_forward=True``;
    * a lazy backward compile, from ``_aot_autograd/runtime_wrappers.py``,
      which carries no code object and sets ``is_forward=False``;
    * a RUNTIME row, from ``_dynamo/utils.py``, billing Triton autotuning or
      a cudagraph re-record against the compile id that caused it. It also
      has no code object and sets ``is_forward = not is_backward``.

    So ``co_name is None`` does not mean "backward" — the pair
    ``(is_runtime, is_forward)`` is what separates them, and conflating the
    last two inflates compile time by whatever autotuning cost. Fields the
    writer left unset stay ``None`` here rather than becoming ``0.0``: an
    absent column and a measured zero are different claims.
    """

    def seconds(micros: int | None) -> float | None:
        return None if micros is None else micros / 1e6

    if metric.is_runtime:
        kind = "runtime"
        name = f"runtime of {metric.compile_id}"
    elif metric.is_forward is False:
        kind = "backward"
        name = f"backward of {metric.compile_id}"
    else:
        kind = "forward"
        name = metric.co_name or f"compile {metric.compile_id}"
    return {
        "compile_id": str(metric.compile_id),
        "kind": kind,
        "function": name,
        "file": metric.co_filename,
        "line": metric.co_firstlineno,
        "cache_size": str(metric.cache_size),
        "is_forward": metric.is_forward,
        "is_runtime": bool(metric.is_runtime),
        "recompile_reason": metric.recompile_reason,
        "seconds": (metric.duration_us or 0) / 1e6,
        "dynamo_seconds": seconds(metric.dynamo_cumulative_compile_time_us),
        "aot_seconds": seconds(metric.aot_autograd_cumulative_compile_time_us),
        "inductor_seconds": seconds(
            metric.inductor_cumulative_compile_time_us
        ),
        "backward_seconds": seconds(metric.backward_cumulative_compile_time_us),
        "runtime_autotune_seconds": seconds(
            metric.runtime_triton_autotune_time_us
        ),
        "runtime_cudagraph_seconds": seconds(metric.runtime_cudagraphify_time_us),
    }


class InertPhase:
    """The disabled profiler's phase object.

    One shared instance with empty ``__enter__``/``__exit__`` is what makes
    ``--profile`` free when it is off: a phase costs one attribute lookup
    and two empty calls, and nothing is timed, recorded, or synchronized.
    """

    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exception) -> bool:
        return False


INERT_PHASE = InertPhase()


class DisabledProfiler:
    """Every profiler entry point, doing nothing.

    Call sites stay unconditional rather than wrapped in ``if profiling:``
    so the profiled and unprofiled control flow cannot drift apart.
    """

    enabled = False

    def phase(self, name: str) -> InertPhase:
        return INERT_PHASE

    def worker_span(self, name: str) -> InertPhase:
        return INERT_PHASE

    def register_artifact(self, name: str, function):
        return function

    def note_counter(self, name: str, value: float) -> None:
        pass

    def pool_started(self, step: int) -> None:
        pass

    def pool_finished(self, step: int, wall_seconds: float) -> dict:
        return {}

    def close(self) -> dict:
        return {}


class DeviceSampler:
    """Background ``nvidia-smi`` poller stamped on the profiler's clock.

    Samples carry a ``perf_counter`` stamp taken at READ time rather than
    nvidia-smi's own wall clock, so they share one monotonic timeline with
    the phase records and can be sliced per phase without clock conversion.

    Polling is not measuring. nvidia-smi returns whatever NVML last
    latched, and each field latches on its own schedule: on this device
    power refreshes about every 500 ms against the default 250 ms poll, so
    half the readings are copies of the one before. A phase shorter than
    one refresh can therefore contain nothing but a value formed before it
    started. ``window`` measures each field's refresh period from the data
    and reports a field only when enough readings were formed entirely
    inside the phase; otherwise it withholds that field and says so. A
    withheld number is better than a stale one attributed to the wrong
    phase.
    """

    # Below this many attributable readings a field's distribution is not a
    # distribution. Three is not principled, it is the smallest count for
    # which a minimum and a mean say different things.
    MINIMUM_READINGS = 3

    # A field changing on fewer than this share of polls is quiet, not
    # slow, and its median gap measures silence instead of a refresh rate.
    # The live fields here change on 20-50% of polls, and the fallback is
    # the poll interval, so the boundary is nowhere near either case.
    QUIET_FRACTION = 0.05

    # The most a field's refresh may be believed to lag the poller. This
    # device refreshes at 2x the default poll and the bound is 4x, so it
    # binds only where the median has stopped measuring a refresh rate.
    MAXIMUM_OVERSAMPLE = 4

    def __init__(self, interval_ms: int, device: torch.device):
        self.interval_ms = interval_ms
        self.device = device
        self.samples: list[tuple[float, tuple[float, ...]]] = []
        self.error: str | None = None
        self._process = None
        self._thread = None
        self._periods: list[float] | None = None
        self._periods_at = -1

    def _selector(self) -> list[str]:
        """Pin nvidia-smi to the training GPU, by UUID.

        Without this every visible GPU emits its own line each tick and idle
        devices are averaged into the phase summary. The UUID is used rather
        than an index because nvidia-smi numbers devices physically while
        CUDA_VISIBLE_DEVICES renumbers them.
        """
        if self.device.type != "cuda":
            return []
        uuid = getattr(
            torch.cuda.get_device_properties(self.device), "uuid", None
        )
        return ["-i", f"GPU-{uuid}"] if uuid is not None else []

    def start(self) -> None:
        if self.interval_ms <= 0:
            self.error = "sampling disabled"
            return
        try:
            self._process = subprocess.Popen(
                [
                    "nvidia-smi",
                    *self._selector(),
                    "--query-gpu=utilization.gpu,utilization.memory,"
                    "power.draw,clocks.sm,memory.used",
                    "--format=csv,noheader,nounits",
                    f"-lms={self.interval_ms}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError as failure:
            self.error = f"nvidia-smi unavailable: {failure}"
            return
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        for line in self._process.stdout:
            columns = line.strip().split(", ")
            if len(columns) != len(DEVICE_SAMPLE_FIELDS):
                continue
            try:
                values = tuple(float(column) for column in columns)
            except ValueError:
                continue
            self.samples.append((time.perf_counter(), values))
        # nvidia-smi rejecting the selector looks exactly like a quiet GPU
        # otherwise: stderr goes nowhere and the report shows no samples.
        if not self.samples and self._process.poll():
            self.error = (
                f"nvidia-smi exited {self._process.returncode} without "
                "producing a sample"
            )

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def period_ms(self, field: str) -> float:
        """How often this field actually changes underneath the poller.

        Measured, not assumed, because the fields do not share a schedule.
        The estimate is the median gap between CHANGES. On this device that
        reads 500 ms for power, clocks and both utilizations against a
        250 ms poll, and the gap histogram is a single sharp mode there, so
        the median is measuring the driver's refresh rather than how often
        the quantity happens to move.

        A field that changes rarely falls back to the delivered sample
        spacing instead of taking a median of long quiet stretches. That is
        not a concession: aliasing is the risk of reporting a value formed
        outside the phase as the phase's, and a field that barely moves
        holds the same value inside and out.

        The estimate is clamped into ``[spacing, MAXIMUM_OVERSAMPLE *
        spacing]``. Below the spacing it would claim a resolution the
        poller never delivered; far above it, the median has stopped
        measuring a refresh and started measuring a quiet stretch, and
        there is no reading of the data that makes an unbounded answer
        safer than a bounded wrong one. ``spacing`` is the delivered gap,
        not the requested interval: ``-lms`` is a request, and a starved
        reader thread that delivers a line a second apart would otherwise
        have every reading credited with a quarter-second of coverage it
        does not have.

        Recomputed whenever the series has grown. The first caller is the
        end of pool 0, whose samples are the least representative in the
        run -- cold compilation holds the GPU at a flat idle, which reads
        as a quiet field -- and freezing that answer would set the run's
        resolution from its worst window.
        """
        if self._periods is None or self._periods_at != len(self.samples):
            self._periods_at = len(self.samples)
            self._periods = [
                self._measure_period(position)
                for position in range(len(DEVICE_SAMPLE_FIELDS))
            ]
        return self._periods[DEVICE_SAMPLE_FIELDS.index(field)]

    def spacing_ms(self) -> float:
        """The gap the poller actually delivered, in milliseconds."""
        gaps = sorted(
            later - earlier
            for (earlier, _), (later, _) in zip(self.samples, self.samples[1:])
        )
        if not gaps:
            return float(self.interval_ms)
        return max(float(self.interval_ms), gaps[len(gaps) // 2] * 1e3)

    def _measure_period(self, index: int) -> float:
        spacing = self.spacing_ms()
        changes = [
            stamp
            for position, (stamp, values) in enumerate(self.samples)
            if position > 0
            and values[index] != self.samples[position - 1][1][index]
        ]
        gaps = sorted(
            later - earlier for earlier, later in zip(changes, changes[1:])
        )
        if len(gaps) < 4 or len(changes) < self.QUIET_FRACTION * len(
            self.samples
        ):
            return spacing
        return min(
            self.MAXIMUM_OVERSAMPLE * spacing,
            max(spacing, gaps[len(gaps) // 2] * 1e3),
        )

    def _readings(
        self, field: str, windows: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """One field's readings that belong to this phase and no other.

        The driver refreshes the field every ``period``, so a reading
        stamped ``t`` was latched somewhere in ``[t - period, t]``. It is
        admitted only when that whole interval falls inside one occurrence,
        which is what keeps a value formed before the phase from being
        reported as the phase's.

        Admitted samples are then thinned to one per period, since polling
        faster than the refresh re-reads the same latch. A changed value is
        always kept: a change proves a new latch regardless of the clock.

        Returns ``(seconds_covered, value)`` pairs rather than bare values.
        A reading covers one refresh period, EXCEPT where a change let it
        through early, and only the caller that turns readings into seconds
        can tell the difference. Charging every reading a whole period once
        reported more seconds below the power floor than the phase lasted.
        """
        index = DEVICE_SAMPLE_FIELDS.index(field)
        period = self.period_ms(field) / 1e3
        readings: list[tuple[float, float]] = []
        previous: float | None = None
        admitted = float("-inf")
        for stamp, values in self.samples:
            value = values[index]
            changed = value != previous
            previous = value
            formed = any(
                started <= stamp - period and stamp <= ended
                for started, ended in windows
            )
            if formed and (changed or stamp - admitted >= period):
                readings.append((min(period, stamp - admitted), value))
                admitted = stamp
        return readings

    def window(
        self, windows: list[tuple[float, float]], power_floor: float
    ) -> dict:
        """Power and clock statistics pooled over one phase's occurrences.

        Deliberately not a mean of per-occurrence means. A power collapse
        lasting a few hundred milliseconds inside a ten-second decode phase
        moves the mean by a couple of watts and disappears; the minimum
        and the seconds spent under the floor are what make it a line
        item. There is no percentile column: at a 500 ms refresh even a
        ten-second phase yields about twenty readings, and a twentieth of
        twenty is the minimum under another name. Readings are pooled across occurrences so
        four decode chunks are one distribution, not four averages.

        Fields are reported independently. A short phase usually keeps its
        utilization numbers and loses its power numbers, because those two
        refresh at different rates, and reporting the pair as though they
        were equally well measured is what made a one-sample phase look
        like a measurement.
        """
        # Nothing sampled at all is not the same claim as sampled and
        # withheld, and a run on a device with no sampler should not report
        # every field as suppressed.
        if not self.samples:
            return {}
        drawn = self._readings("power_draw_watts", windows)
        clocked = self._readings("clocks_sm_mhz", windows)
        used = self._readings("utilization_gpu_percent", windows)
        power = sorted(watts for _, watts in drawn)
        clocks = sorted(mhz for _, mhz in clocked)
        utilization = [percent for _, percent in used]
        stats: dict = {}
        withheld = []
        for field, readings in (
            ("power_draw_watts", power),
            ("clocks_sm_mhz", clocks),
            ("utilization_gpu_percent", utilization),
        ):
            if len(readings) < self.MINIMUM_READINGS:
                withheld.append(field)
            else:
                stats[f"{field}_period_ms"] = self.period_ms(field)
                stats[f"{field}_readings"] = len(readings)
        if "power_draw_watts_readings" in stats:
            stats.update(
                {
                    "power_draw_watts_mean": sum(power) / len(power),
                    "power_draw_watts_min": power[0],
                    "power_draw_watts_max": power[-1],
                    # Seconds, not a count: comparable across phases of
                    # different lengths and answerable against the phase's
                    # own wall, which it can no longer exceed because each
                    # reading is charged only the span it covers.
                    "seconds_below_power_floor": sum(
                        covered
                        for covered, watts in drawn
                        if watts < power_floor
                    ),
                }
            )
        if "clocks_sm_mhz_readings" in stats:
            stats["clocks_sm_mhz_mean"] = sum(clocks) / len(clocks)
            stats["clocks_sm_mhz_min"] = clocks[0]
        if "utilization_gpu_percent_readings" in stats:
            stats["utilization_gpu_percent_mean"] = sum(utilization) / len(
                utilization
            )
        if withheld:
            stats["withheld"] = withheld
        return stats


class SyncDetector:
    """Blocking host syncs, believed only after positive controls pass.

    ``set_sync_debug_mode("warn")`` reports each blocking synchronization as
    a Python warning located at the calling line. The detector first trips
    three syncs it knows must fire and refuses to report anything unless it
    caught all three: a broken detector and a sync-free window look
    identical otherwise, which is how an earlier investigation came to
    report a host sync from a process that was running on CPU.

    One known blind spot, reported alongside the counts rather than left for
    someone to trip over: torch installs its warning handler per thread, so
    a sync raised on an autograd backward worker never reaches this hook.
    Absence of a site here is not proof that the thread is sync-free.
    """

    # c10 emits "called a synchronizing CUDA operation" through PyErr_WarnEx
    # with stacklevel 1, so the warning lands on the caller's own line.
    SYNC_MESSAGE = "called a synchronizing CUDA operation"

    def __init__(self, device: torch.device):
        self.device = device
        self.sites: dict[str, int] = {}
        self.controls_passed = False
        self.control_detail: dict[str, bool] = {}
        self._active = False
        self._previous_showwarning = None
        self._previous_filters = None

    def _install(self) -> None:
        self._previous_showwarning = warnings.showwarning
        self._previous_filters = warnings.filters[:]
        fallback = self._previous_showwarning

        def record(message, category, filename, lineno, file=None, line=None):
            if self.SYNC_MESSAGE in str(message):
                key = f"{filename}:{lineno}"
                self.sites[key] = self.sites.get(key, 0) + 1
            else:
                fallback(message, category, filename, lineno, file, line)

        warnings.showwarning = record
        # The sync warning repeats from one location every call, and the
        # default once-per-location filter would collapse a hot loop's
        # thousands of syncs into a single report. Restored on the way out:
        # the filter list is process-global state this must not leak.
        warnings.simplefilter("always")

    def _restore(self) -> None:
        if self._previous_showwarning is not None:
            warnings.showwarning = self._previous_showwarning
            self._previous_showwarning = None
        if self._previous_filters is not None:
            warnings.filters[:] = self._previous_filters
            warnings._filters_mutated()
            self._previous_filters = None

    def run_controls(self) -> None:
        if self.device.type != "cuda":
            self.control_detail = {"cuda_device": False}
            return
        probe = torch.ones(4, device=self.device)
        self._install()
        torch.cuda.set_sync_debug_mode("warn")
        try:
            for name, call in (
                ("item", lambda: probe.sum().item()),
                ("cpu", lambda: probe.cpu()),
                ("bool_any", lambda: bool(probe.any())),
            ):
                before = sum(self.sites.values())
                call()
                self.control_detail[name] = sum(self.sites.values()) > before
        finally:
            torch.cuda.set_sync_debug_mode("default")
            self._restore()
        self.sites.clear()
        self.controls_passed = all(self.control_detail.values())

    def enable(self) -> None:
        if not self.controls_passed or self._active:
            return
        self._install()
        torch.cuda.set_sync_debug_mode("warn")
        self._active = True

    def disable(self) -> None:
        if not self._active:
            return
        torch.cuda.set_sync_debug_mode("default")
        self._restore()
        self._active = False

    def report(self) -> dict:
        if not self.controls_passed:
            return {
                "trusted": False,
                "reason": "positive controls did not all trip, so an empty "
                "result would be indistinguishable from a broken detector",
                "controls": self.control_detail,
                "sites": [],
                "total": 0,
            }
        ranked = sorted(self.sites.items(), key=lambda entry: -entry[1])
        return {
            "trusted": True,
            "controls": self.control_detail,
            "blind_spot": "torch's warning handler is per thread, so syncs "
            "on autograd backward workers are not counted here",
            "sites": [
                {"location": location, "count": count}
                for location, count in ranked
            ],
            "total": sum(self.sites.values()),
        }


class RunProfiler:
    """Per-phase accounting for one profiling run.

    Three properties this exists to provide, none of which the hand-rolled
    ``pool_*_seconds`` counters it supplements had:

    - Phases nest, and each one reports SELF time (its wall minus its
      children's). Root self times plus one explicit ``unaccounted`` row
      equal the pool wall time by construction, so work that no phase covers
      shows up as a remainder instead of hiding inside a plausible number.
    - Phase timing takes no barrier. Host time comes from ``perf_counter``
      and device time from a CUDA event pair resolved ONCE per pool, so
      measuring a phase does not drain the pipeline into it.
    - A blocking host sync is a line item attributed to a source location
      rather than silently folded into whatever phase it lands in.
    """

    enabled = True

    def __init__(
        self,
        output: Path,
        device: torch.device,
        args: argparse.Namespace,
    ):
        # Created at the first write, not here: the profiler is built before
        # the trainer's fresh-output check, which a directory made now would
        # trip on every fresh profiled run.
        self._directory = output / "profile"
        self.device = device
        self.args = args
        self._path: list[str] = []
        self._records: list[dict] = []
        self._events: list[tuple[tuple[str, ...], object, object]] = []
        self._worker_spans: list[tuple[str, float, float]] = []
        self._call_counts: dict[str, int] = {}
        self._calls_before_pool: dict[str, int] = {}
        self._counters: dict[str, float] = {}
        self._compile_seconds: dict[str, float] = {}
        self._runtime_seconds: dict[str, float] = {}
        self._seen_compilations: set[tuple] = set()
        self._before_pool: list[dict] = []
        self._memory_before: dict = {}
        self._kernels: dict | None = None
        self._torch_profile = None
        self._closed = False
        self.pools: list[dict] = []
        self.pool_index = 0

        # The compilation metrics deque is bounded (64 by default) and evicts
        # silently, so without raising it a profiled run would drop the
        # earliest compilations, which are the ones worth seeing.
        torch._dynamo.utils.set_compilation_metrics_limit(
            max(args.profile_compile_records, 64)
        )
        # Anything already compiled before this object existed belongs to
        # whoever compiled it, not to this run's accounting.
        self._new_compilations()
        self.sampler = DeviceSampler(args.profile_device_interval_ms, device)
        self.sampler.start()
        self.sync_detector = SyncDetector(device)
        self.sync_detector.run_controls()
        # The sampler owns a child process and sync debug mode is global
        # state; a run that ends by raising must still release both, and a
        # partial profile is worth printing.
        atexit.register(self.close)

    @contextmanager
    def phase(self, name: str):
        self._path.append(name)
        path = tuple(self._path)
        start_event = None
        if self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        started = time.perf_counter()
        try:
            yield
        finally:
            ended = time.perf_counter()
            if start_event is not None:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                self._events.append((path, start_event, end_event))
            self._records.append(
                {"path": path, "started": started, "ended": ended}
            )
            self._path.pop()

    @contextmanager
    def worker_span(self, name: str):
        """Time a region running on a background worker thread.

        Kept out of the phase tree: a worker span overlaps whatever
        main-thread phase is open, so summing it in would break the
        reconciliation. It is reported per phase as overlap instead.
        ``list.append`` is the only shared mutation, which the GIL makes
        atomic, so this needs no lock.
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            self._worker_spans.append(
                (name, started, time.perf_counter())
            )

    def register_artifact(self, name: str, function):
        """Count calls into a compiled artifact so its compile time can be
        priced against its use. An artifact that is compiled and never
        called is pure cost, and this run has had one."""
        if function is None:
            return None
        self._call_counts[name] = 0

        def counted(*call_args, **call_kwargs):
            self._call_counts[name] += 1
            return function(*call_args, **call_kwargs)

        return counted

    def note_counter(self, name: str, value: float) -> None:
        """Attach a per-pool scalar the phase tree cannot derive on its own.

        The launch-count and phase records answer "how much work"; a counter
        like the pool's decode step total supplies the denominator, so
        ratios such as launches per decode step come out of the profile
        alone instead of a manual join against the metrics stream. Values
        accumulate within a pool and reset with it.
        """
        self._counters[name] = self._counters.get(name, 0.0) + value

    def pool_started(self, step: int) -> None:
        self._records.clear()
        self._events.clear()
        self._worker_spans.clear()
        self._counters.clear()
        self._kernels = None
        # Allocator counters, read as a per-pool delta. An allocation retry
        # flushes the cache and synchronizes every stream, which presents
        # exactly as a power collapse and costs nothing to rule in or out.
        self._memory_before = (
            torch.cuda.memory_stats() if self.device.type == "cuda" else {}
        )
        # Drained here as well as at pool end so cold compilation — step-0
        # eval, the value-warmup loop, everything before the first pool — is
        # reported in its own bucket instead of being billed to pool 0, whose
        # own wall time it can exceed.
        self._before_pool = self._new_compilations()
        # Snapshot rather than zero: close() reports run totals from the same
        # counters, and a pool's figure is the delta against this.
        self._calls_before_pool = dict(self._call_counts)
        if self._path:
            raise RuntimeError(
                f"profile phase stack left open across pools: {self._path}"
            )
        # Each instrument gets its own pool. Sync detection raises a Python
        # warning per blocking call, which would both distort a kineto trace
        # and be distorted by one, so it runs on the pools AFTER the traced
        # window rather than sharing them.
        traced_from = self.args.profile_skip_pools
        sync_from = traced_from + self.args.profile_pools
        if sync_from <= self.pool_index < sync_from + self.args.profile_sync_pools:
            self.sync_detector.enable()
        if traced_from <= self.pool_index < sync_from:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            # A fresh profile per pool rather than one scheduled window:
            # kernel launch counts are only meaningful attributed to a single
            # pool, and torch's schedule() spends a whole pool on warmup
            # between active windows.
            self._torch_profile = torch.profiler.profile(
                activities=activities,
                record_shapes=False,
                profile_memory=False,
                with_stack=self.args.profile_stack,
            )
            self._torch_profile.start()

    def _device_seconds(self) -> dict[tuple[str, ...], float]:
        if not self._events:
            return {}
        # The one barrier this profiler takes, once per pool and only under
        # the flag: elapsed_time is undefined until both events complete.
        torch.cuda.synchronize()
        totals: dict[tuple[str, ...], float] = {}
        for path, start_event, end_event in self._events:
            totals[path] = totals.get(path, 0.0) + (
                start_event.elapsed_time(end_event) / 1e3
            )
        return totals

    def _summarize_kernels(self) -> dict:
        """Top kernels by device time, and the launch count two ways.

        Device-side kernel invocations and host-side ``cudaLaunchKernel``
        calls answer different questions and may disagree; a launch-bound
        loop is diagnosed by the host-side number.
        """
        profile, self._torch_profile = self._torch_profile, None
        if profile is None:
            return {}
        profile.stop()
        # Measured on this trainer: one pool is 1.03M device launches, and
        # its Chrome trace is 2.3 GB — past what Perfetto will open, and the
        # launch counts and kernel ranking come from key_averages anyway.
        # So the export is opt-in and the counts are not.
        trace = None
        exported = None
        if self.args.profile_trace:
            trace = self.directory() / f"trace_pool_{self.pool_index}.json"
            try:
                profile.export_chrome_trace(str(trace))
                exported = trace.stat().st_size
            except (OSError, MemoryError, RuntimeError) as failure:
                return {"trace": str(trace), "error": f"export: {failure}"}
        try:
            averages = profile.key_averages()
        except (AssertionError, MemoryError, RuntimeError) as failure:
            return {
                "trace": str(trace) if trace else None,
                "trace_bytes": exported,
                "error": f"key_averages: {failure}",
            }
        kernels = []
        device_launches = 0
        host_launches = 0
        for entry in averages:
            if getattr(entry, "device_type", None) == (
                torch.autograd.DeviceType.CUDA
            ):
                device_launches += entry.count
                kernels.append(
                    {
                        "name": entry.key,
                        "count": entry.count,
                        "device_seconds": (
                            getattr(entry, "self_device_time_total", 0) or 0
                        )
                        / 1e6,
                    }
                )
            elif entry.key.startswith(LAUNCH_EVENT_PREFIXES):
                host_launches += entry.count
        kernels.sort(key=lambda kernel: -kernel["device_seconds"])
        return {
            "trace": str(trace) if trace else None,
            "trace_bytes": exported,
            "device_kernel_launches": device_launches,
            "host_cuda_launch_calls": host_launches,
            "total_device_seconds": sum(
                kernel["device_seconds"] for kernel in kernels
            ),
            "top_kernels": kernels[: self.args.profile_top_kernels],
        }

    def _new_compilations(self) -> list[dict]:
        fresh = []
        for metric in torch._dynamo.utils.get_compilation_metrics():
            key = (
                str(metric.compile_id),
                metric.co_name,
                metric.is_forward,
                metric.is_runtime,
                metric.start_time_us,
            )
            if key in self._seen_compilations:
                continue
            self._seen_compilations.add(key)
            fresh.append(compilation_record(metric))
        return fresh

    def _bill_compilations(self, entries: list[dict]) -> None:
        """Add records to the run totals, keeping the two kinds apart.

        Compile time is charged to the frame, which is what a fix targets.
        Runtime autotuning and cudagraph re-records have no frame, so they
        are charged to the compile id that provoked them.
        """
        for entry in entries:
            if entry["kind"] == "runtime":
                self._runtime_seconds[entry["compile_id"]] = (
                    self._runtime_seconds.get(entry["compile_id"], 0.0)
                    + entry["seconds"]
                )
            else:
                self._compile_seconds[entry["function"]] = (
                    self._compile_seconds.get(entry["function"], 0.0)
                    + entry["seconds"]
                )

    def _allocator_delta(self) -> dict:
        """Allocator events this pool caused.

        ``num_alloc_retries`` is the one to watch: a retry empties the
        caching allocator and synchronizes every stream, which is a
        device-wide stall that looks like nothing else in the phase table.
        """
        if self.device.type != "cuda":
            return {}
        after = torch.cuda.memory_stats()
        return {
            counter: after.get(counter, 0)
            - self._memory_before.get(counter, 0)
            for counter in (
                "num_alloc_retries",
                "num_ooms",
                "num_device_alloc",
                "num_device_free",
                "num_sync_all_streams",
            )
        }

    def pool_finished(self, step: int, wall_seconds: float) -> dict:
        if self._path:
            raise RuntimeError(
                f"profile phase stack still open at pool end: {self._path}"
            )
        self.sync_detector.disable()
        kernels = self._summarize_kernels()
        device_seconds = self._device_seconds()
        walls: dict[tuple[str, ...], float] = {}
        counts: dict[tuple[str, ...], int] = {}
        windows: dict[tuple[str, ...], list[tuple[float, float]]] = {}
        for record in self._records:
            path = record["path"]
            walls[path] = walls.get(path, 0.0) + (
                record["ended"] - record["started"]
            )
            counts[path] = counts.get(path, 0) + 1
            windows.setdefault(path, []).append(
                (record["started"], record["ended"])
            )
        # Worker spans overlap main-thread phases by construction, so they
        # are reported alongside a phase rather than summed into the tree.
        # The quantity that matters is worker CPU running INSIDE decode: the
        # decode loop is launch-bound, and a worker holding the GIL there
        # stops the host from launching and drains the GPU.
        worker_seconds: dict[tuple[str, ...], float] = {}
        for path, spans in windows.items():
            worker_seconds[path] = sum(
                max(0.0, min(ended, worker_ended) - max(started, worker_started))
                for _, worker_started, worker_ended in self._worker_spans
                for started, ended in spans
            )
        # Depth-first, siblings by descending wall time. The report indents by
        # depth, so any order that separates a phase from its parent prints a
        # tree that lies about who owns what.
        def tree_order(path: tuple[str, ...]) -> tuple:
            return tuple(
                (-walls[path[: depth + 1]], path[depth])
                for depth in range(len(path))
            )

        phases = []
        for path in sorted(walls, key=tree_order):
            children = sum(
                wall
                for other, wall in walls.items()
                if len(other) == len(path) + 1 and other[: len(path)] == path
            )
            phases.append(
                {
                    "path": list(path),
                    "name": path[-1],
                    "depth": len(path) - 1,
                    "count": counts[path],
                    "wall_seconds": walls[path],
                    "self_seconds": walls[path] - children,
                    "device_seconds": device_seconds.get(path),
                    "worker_seconds": worker_seconds.get(path, 0.0),
                    # On the same perf_counter clock as device_samples.json,
                    # so the raw power series can be sliced by phase offline.
                    "windows": windows[path],
                    "device": self.sampler.window(
                        windows[path], self.args.profile_power_floor
                    ),
                }
            )
        roots = sum(wall for path, wall in walls.items() if len(path) == 1)
        unaccounted = wall_seconds - roots
        if unaccounted < -1e-6:
            raise RuntimeError(
                "profile phase tree is broken: root phases total "
                f"{roots:.3f} s inside a {wall_seconds:.3f} s pool"
            )
        before_pool, self._before_pool = self._before_pool, []
        compilations = self._new_compilations()
        self._bill_compilations(before_pool + compilations)
        pool = {
            "schema": PROFILE_SCHEMA,
            "pool_index": self.pool_index,
            "step": step,
            "wall_seconds": wall_seconds,
            "phases": phases,
            "unaccounted_seconds": unaccounted,
            "unaccounted_fraction": (
                unaccounted / wall_seconds if wall_seconds else 0.0
            ),
            "reconciled": unaccounted
            <= self.args.profile_reconcile_tolerance * max(wall_seconds, 1e-9),
            # Split so "did this pool compile anything?" has an answer that
            # cold startup cannot contaminate. Steady state means both lists
            # hold no record of kind "forward" or "backward" from some pool
            # onwards. Runtime records are NOT part of that test: a
            # reduce-overhead artifact re-records its cudagraph whenever the
            # pool it captured against is invalidated, which can happen at
            # any point in a perfectly converged run.
            "compilations_before_pool": before_pool,
            "compilations": compilations,
            "compile_seconds": sum(
                entry["seconds"]
                for entry in before_pool + compilations
                if entry["kind"] != "runtime"
            ),
            "runtime_seconds": sum(
                entry["seconds"]
                for entry in before_pool + compilations
                if entry["kind"] == "runtime"
            ),
            "artifact_calls": {
                name: count - self._calls_before_pool.get(name, 0)
                for name, count in self._call_counts.items()
            },
            "counters": dict(self._counters),
            "kernels": kernels,
            "allocator": self._allocator_delta(),
            "worker_seconds_total": sum(
                ended - started for _, started, ended in self._worker_spans
            ),
        }
        self.pools.append(pool)
        (self.directory() / f"pool_{self.pool_index}.json").write_text(
            json.dumps(pool, indent=2)
        )
        if not pool["reconciled"]:
            print(
                f"WARNING profile pool {self.pool_index}: "
                f"{unaccounted:.3f} s of {wall_seconds:.3f} s "
                f"({100.0 * pool['unaccounted_fraction']:.1f}%) is covered by "
                "no phase; the accounting is not closed",
                flush=True,
            )
        self.pool_index += 1
        return pool

    def directory(self) -> Path:
        self._directory.mkdir(parents=True, exist_ok=True)
        return self._directory

    def close(self) -> dict:
        if self._closed:
            return {}
        self._closed = True
        if self._torch_profile is not None:
            self._torch_profile.stop()
            self._torch_profile = None
        self.sampler.stop()
        self.sync_detector.disable()
        # A run whose modes exit before the pool loop (--bench-only,
        # --bpb-only, --rollout-only) still compiles, and so does anything
        # after the last pool. Without this drain those compilations are
        # simply never reported.
        outside_pools = self._before_pool + self._new_compilations()
        self._bill_compilations(outside_pools)
        summary = {
            "schema": PROFILE_SCHEMA,
            "pools": self.pools,
            "compilations_outside_pools": outside_pools,
            "syncs": self.sync_detector.report(),
            "artifact_calls": dict(self._call_counts),
            "compile_seconds_by_function": dict(self._compile_seconds),
            "runtime_seconds_by_compile_id": dict(self._runtime_seconds),
            "dead_artifacts": [
                name for name, calls in self._call_counts.items() if not calls
            ],
            "device_sampler_error": self.sampler.error,
            "device_samples": len(self.sampler.samples),
            # The poll interval is a request; these are what the driver
            # actually delivered, and they are the reason a short phase
            # reports no power.
            "device_sample_period_ms": {
                field: self.sampler.period_ms(field)
                for field in DEVICE_SAMPLE_FIELDS
            },
        }
        (self.directory() / "profile_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        # The raw series, not just its per-phase digest. A periodic collapse
        # is defined by its period, and no summary statistic carries that;
        # at a quarter-second interval a whole run is a few thousand rows.
        (self.directory() / "device_samples.json").write_text(
            json.dumps(
                {
                    "fields": ["perf_counter", *DEVICE_SAMPLE_FIELDS],
                    "samples": [
                        [stamp, *values] for stamp, values in self.sampler.samples
                    ],
                }
            )
        )
        print(format_profile_summary(summary), flush=True)
        return summary


@contextmanager
def device_timed(sink: list):
    """Time a device region with a CUDA event pair instead of a barrier.

    A host wall clock around an asynchronous region measures launch cost,
    and making it measure device time needs a ``synchronize`` that drains
    the pipeline into the very region being timed. The pair is recorded on
    the stream and read later, after a barrier the caller already takes.
    """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    try:
        yield
    finally:
        end_event.record()
        sink.append((start_event, end_event))


def elapsed_seconds(events: list) -> float:
    """Total device seconds over recorded pairs.

    Waits on each end event rather than requiring the caller to have taken a
    barrier: elapsed_time raises on an event that has not completed, and
    where a barrier has already happened this returns immediately.
    """
    total = 0.0
    for start_event, end_event in events:
        end_event.synchronize()
        total += start_event.elapsed_time(end_event)
    return total / 1e3


def format_compile_split(entry: dict) -> str:
    """Render only the sub-timings the writer actually filled in.

    A runtime record has no dynamo or inductor column and a forward has no
    autotune column; printing the absent ones as ``0.00`` reads as a
    measurement rather than as silence.
    """
    parts = [
        (label, entry[field])
        for label, field in (
            ("dynamo", "dynamo_seconds"),
            ("aot", "aot_seconds"),
            ("inductor", "inductor_seconds"),
            ("backward", "backward_seconds"),
            ("autotune", "runtime_autotune_seconds"),
            ("cudagraph", "runtime_cudagraph_seconds"),
        )
        if entry.get(field) is not None
    ]
    return (
        " / ".join(f"{label} {value:.2f}" for label, value in parts)
        if parts
        else "no sub-timings reported"
    )


def format_profile_summary(summary: dict) -> str:
    """End-of-run table. Plain text so it survives the job log."""
    lines = ["", "=" * 96, "PROFILE SUMMARY", "=" * 96]
    for pool in summary["pools"]:
        records = pool["compilations_before_pool"] + pool["compilations"]
        compiles = [entry for entry in records if entry["kind"] != "runtime"]
        lines.append(
            f"\npool {pool['pool_index']} (actor step {pool['step']}): "
            f"{pool['wall_seconds']:.3f} s wall, "
            f"{pool['compile_seconds']:.3f} s compiling in "
            f"{len(compiles)} compilations, "
            f"{pool['runtime_seconds']:.3f} s in "
            f"{len(records) - len(compiles)} runtime records"
        )
        lines.append(
            f"  {'phase':<30}{'n':>5}{'wall s':>9}{'self s':>9}{'% pool':>8}"
            f"{'gpu s':>9}{'wrkr s':>8}{'W mean':>8}{'W min':>7}"
            f"{'s<flr':>7}{'MHz':>6}"
        )
        for phase in pool["phases"]:
            label = "  " * phase["depth"] + phase["name"]
            share = 100.0 * phase["self_seconds"] / max(pool["wall_seconds"], 1e-9)
            device = phase["device_seconds"]
            sampled = phase["device"]

            def column(name: str, digits: int = 0, width: int = 7) -> str:
                value = sampled.get(name)
                text = "-" if value is None else f"{value:.{digits}f}"
                return f"{text:>{width}}"

            lines.append(
                f"  {label:<30}{phase['count']:>5}"
                f"{phase['wall_seconds']:>9.3f}{phase['self_seconds']:>9.3f}"
                f"{share:>8.1f}"
                f"{(f'{device:.3f}' if device is not None else '-'):>9}"
                f"{phase['worker_seconds']:>8.2f}"
                + column("power_draw_watts_mean", width=8)
                + column("power_draw_watts_min")
                + column("seconds_below_power_floor", digits=1)
                + column("clocks_sm_mhz_min", width=6)
            )
        flag = "" if pool["reconciled"] else "  <-- NOT CLOSED"
        lines.append(
            f"  {'unaccounted':<30}{'':>5}{'':>9}"
            f"{pool['unaccounted_seconds']:>9.3f}"
            f"{100.0 * pool['unaccounted_fraction']:>8.1f}{flag}"
        )
        # A phase total hides a stall. Four refreshes averaging 4.5 s and
        # three refreshes at 0.7 s plus one at 16 s are the same row above,
        # and only the second is worth chasing. Printed with the start
        # stamp because device_samples.json shares this clock, so an
        # outlier can be looked up against power and clocks directly.
        for phase in pool["phases"]:
            spans = [ended - started for started, ended in phase["windows"]]
            if len(spans) < 2:
                continue
            longest = max(spans)
            # Twice the mean of the OTHER occurrences. Comparing against the
            # mean of all of them buries the outlier in its own average: at
            # two occurrences no span can reach twice a mean it is half of,
            # so the one case this is for -- 0.7 s and then 16 s -- would
            # never print.
            if longest < 1.0 or longest * (len(spans) - 1) < 2.0 * (
                sum(spans) - longest
            ):
                continue
            started = max(phase["windows"], key=lambda pair: pair[1] - pair[0])[0]
            lines.append(
                f"    uneven: {'.'.join(phase['path'])} longest occurrence "
                f"{longest:.3f} s of {phase['wall_seconds']:.3f} s over "
                f"{len(spans)}, started {started:.3f}"
            )
        allocator = pool["allocator"]
        if any(allocator.values()):
            lines.append(
                "    allocator this pool: "
                + ", ".join(
                    f"{counter.removeprefix('num_')}={value}"
                    for counter, value in allocator.items()
                    if value
                )
            )
        for label, entries in (
            ("before pool", pool["compilations_before_pool"]),
            ("in pool", pool["compilations"]),
        ):
            lines.extend(
                f"    {entry['kind']} ({label}) {entry['function']}:"
                f"{entry['line']} {entry['seconds']:.2f} s "
                f"({format_compile_split(entry)}) cache_size="
                f"{entry['cache_size']} reason="
                f"{entry['recompile_reason'] or 'first compile'}"
                for entry in entries
            )
        kernels = pool["kernels"]
        if kernels and "error" not in kernels:
            lines.append(
                f"    kernel launches: {kernels['device_kernel_launches']} "
                f"device-side, {kernels['host_cuda_launch_calls']} host "
                f"launch calls; {kernels['total_device_seconds']:.3f} s "
                "total device time"
                + (
                    f"; trace {kernels['trace_bytes'] / 2**20:.0f} MiB"
                    if kernels.get("trace_bytes")
                    else ""
                )
            )
            for kernel in kernels["top_kernels"]:
                lines.append(
                    f"      {kernel['device_seconds']:8.3f} s "
                    f"x{kernel['count']:<9} {kernel['name'][:56]}"
                )
        elif kernels:
            lines.append(f"    kernel summary failed: {kernels['error']}")
        counters = pool.get("counters") or {}
        if counters:
            lines.append(
                "    counters this pool: "
                + ", ".join(
                    f"{name}={value:g}"
                    for name, value in sorted(counters.items())
                )
            )
            decode_steps = counters.get("decode_steps")
            if decode_steps and kernels and "error" not in kernels:
                # An upper bound: the pool's launch total includes refresh
                # and update work, not just the decode loop.
                lines.append(
                    "    host launch calls per decode step (upper bound): "
                    f"{kernels['host_cuda_launch_calls'] / decode_steps:.0f}"
                )
    outside = summary["compilations_outside_pools"]
    if outside:
        lines.append(
            f"\n{len(outside)} compilations outside any pool "
            "(startup, evaluation, or after the last pool):"
        )
        for entry in outside:
            lines.append(
                f"  {entry['kind']} {entry['function']}:{entry['line']} "
                f"{entry['seconds']:.2f} s reason="
                f"{entry['recompile_reason'] or 'first compile'}"
            )
    calls = summary["artifact_calls"]
    if calls:
        lines.append("\ncompiled artifact calls over the run:")
        for name, count in sorted(calls.items()):
            note = "  <-- NEVER CALLED; its compile time is waste" if not count else ""
            lines.append(f"  {name:<40}{count:>12}{note}")
    compiles = summary["compile_seconds_by_function"]
    if compiles:
        lines.append("\ncompile seconds by frame:")
        for name, seconds in sorted(compiles.items(), key=lambda item: -item[1]):
            lines.append(f"  {name:<40}{seconds:>12.2f}")
    runtimes = summary["runtime_seconds_by_compile_id"]
    if runtimes:
        lines.append(
            "\nruntime seconds (Triton autotuning and cudagraph re-records,"
            " charged to the compile id that caused them):"
        )
        for name, seconds in sorted(runtimes.items(), key=lambda item: -item[1]):
            lines.append(f"  {name:<40}{seconds:>12.2f}")
    syncs = summary["syncs"]
    lines.append("\nblocking host syncs:")
    if not syncs["trusted"]:
        lines.append(f"  NOT REPORTED: {syncs['reason']}")
        lines.append(f"  positive controls: {syncs['controls']}")
    else:
        lines.append(f"  positive controls all tripped: {syncs['controls']}")
        lines.append(
            f"  {syncs['total']} blocking syncs over the sampled pools"
        )
        for site in syncs["sites"][:20]:
            lines.append(f"    {site['count']:>9}  {site['location']}")
        lines.append(f"  blind spot: {syncs['blind_spot']}")
    if summary["device_sampler_error"]:
        lines.append(f"\ndevice sampler: {summary['device_sampler_error']}")
    else:
        lines.append(f"\ndevice samples: {summary['device_samples']}")
        # With no samples every period is the fallback, and printing those
        # under "measured" would be the report inventing its own evidence.
        periods = summary.get("device_sample_period_ms") or {}
        if periods and summary["device_samples"]:
            lines.append(
                "  measured refresh: "
                + ", ".join(
                    f"{field} {period:.0f} ms"
                    for field, period in periods.items()
                )
            )
            lines.append(
                "  a phase shorter than a field's refresh reports '-' for it "
                "rather than a value formed before the phase began"
            )
    lines.append("=" * 96)
    return "\n".join(lines)

