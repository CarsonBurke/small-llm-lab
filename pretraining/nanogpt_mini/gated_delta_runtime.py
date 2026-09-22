"""Compiled dense regions around verified opaque Gated Delta Triton operators.

The released operator explicitly disables Dynamo tracing. That declared kernel
boundary is intentional; compilation errors and unrelated graph breaks remain
fatal. Models built with custom_ops register the same kernels as compilable
operators and must compile as one whole graph. The complete forward/backward
is captured by the CUDA graph executor either way.
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F


class CompiledGatedDeltaLoss:
    def __init__(self, model, segment_size=64):
        import torch._dynamo.config as config
        from torch._dynamo.utils import counters
        config.suppress_errors = False
        config.fail_on_recompile_limit_hit = True
        self.model = model
        self.segment_size = segment_size
        self.fullgraph = bool(model.config.get("custom_ops", False))
        self._breaks_before = dict(counters["graph_break"])
        self._compiled = torch.compile(self._loss, fullgraph=self.fullgraph, dynamic=False)

    def _loss(self, inputs, targets, diagnostics=False):
        hidden, _ = self.model.forward_hidden(inputs, segment_size=self.segment_size)
        loss = F.cross_entropy(self.model.logits(hidden).flatten(0, 1),
                               targets.flatten(), reduction="sum")
        if diagnostics:
            # No training-state diagnostics are exposed by the official module.
            return (loss,)
        return loss

    def __call__(self, inputs, targets, *, diagnostics=False):
        if inputs.device.type != "cuda" or targets.device != inputs.device:
            raise ValueError("Gated Delta requires CUDA inputs and targets")
        if inputs.ndim != 2 or inputs.shape != targets.shape:
            raise ValueError("inputs and targets must share [batch,time] shape")
        return self._compiled(inputs, targets, diagnostics)

    def audit_graph_breaks(self):
        """Allow only verified opaque GDN/FLA kernel dispatch boundaries."""
        from torch._dynamo.utils import counters
        changes = {str(reason): count - self._breaks_before.get(reason, 0)
                   for reason, count in counters["graph_break"].items()
                   if count > self._breaks_before.get(reason, 0)}
        # FLA's dispatch decorator disables tracing for these Triton kernels.
        # They are captured by CUDA graphs, not substituted with eager math.
        memory_rule = self.model.config.get("memory_rule", "gdn2")
        if memory_rule not in {"gdn2", "scalar_delta"}:
            raise RuntimeError(f"Unknown compiled memory rule: {memory_rule}")
        operator = "chunk_gated_delta_rule" if memory_rule == "scalar_delta" else "chunk_gdn2"
        whole_graph = bool(self.model.config.get("custom_ops", False))
        allowed = set() if whole_graph else {operator, "causal_conv1d_fwd", "layer_norm_gated_fwd"}
        unexpected = []
        for reason in changes:
            functions = set(re.findall(r"<function ([A-Za-z_][A-Za-z_0-9]*) at ", reason))
            if "torch.compiler.disable" not in reason or not functions or not functions <= allowed:
                unexpected.append(reason)
        if unexpected:
            raise RuntimeError(f"Unexpected Gated Delta graph breaks: {unexpected}")
        return changes


class CompiledSharedPoolLoss(CompiledGatedDeltaLoss):
    """Both Jacobi passes scored; diagnostics expose the pool's routing statistics.

    The training objective averages the two passes' summed cross-entropies and
    adds the routers' Switch balance and z-losses scaled to the token count, so
    the coefficients mean the same as they do against a per-token mean loss.
    Diagnostics return the second (headline) pass first, then the first pass,
    the summed balance loss, the summed z-loss and the peak bank load.
    """

    def _loss(self, inputs, targets, diagnostics=False):
        first, second, stats = self.model.forward_passes(inputs)
        targets = targets.flatten()
        first_loss, second_loss = (F.cross_entropy(self.model.logits(hidden).flatten(0, 1), targets, reduction="sum")
                                   for hidden in (first, second))
        if diagnostics:
            return (second_loss, first_loss, stats["balance"], stats["z"], stats["load_max"])
        config = self.model.config
        regularizer = config["pool_balance_coefficient"] * stats["balance"] + config["pool_z_coefficient"] * stats["z"]
        return 0.5 * (first_loss + second_loss) + targets.numel() * regularizer


def compiled_gated_delta_loss(model, segment_size=64):
    """The compiled loss matching the model's configuration."""
    wrapper = CompiledSharedPoolLoss if model.config.get("shared_pool") else CompiledGatedDeltaLoss
    return wrapper(model, segment_size)


def build_gated_delta_optimizers(model):
    """Matched-mini optimizer adapted explicitly for convolution and decay gates.

    Shared-pool routers take a small AdamW rate without weight decay: their
    decisions only receive gradient through the selected banks' read weights
    and the balance loss, and orthogonalized Muon steps at the matrix rate
    would reshuffle routes faster than the banks can be learned. The pool's
    per-bank adapters are stacks of square matrices and take Muon like every
    other matrix (its orthogonalization is batched over the leading axis).
    """
    from scripts.train_recurrent_slots import Muon
    embedding, head = model.embed.weight, model.proj.weight
    special = {id(embedding), id(head)}
    scalars, matrices, convolutions, no_decay, routers = [], [], [], [], []
    for name, parameter in model.named_parameters():
        if id(parameter) in special:
            continue
        if getattr(parameter, "_no_weight_decay", False):
            no_decay.append(parameter)
        elif ".router." in name:
            routers.append(parameter)
        elif "_conv1d." in name:
            convolutions.append(parameter)
        elif parameter.ndim == 2 or (parameter.ndim == 3 and name.startswith("adapters.")):
            matrices.append(parameter)
        elif parameter.ndim < 2:
            scalars.append(parameter)
        else:
            raise ValueError(f"Unclassified Gated Delta parameter: {name}")
    groups = [{"params": [embedding], "lr": 0.7},
              {"params": [head], "lr": 0.004}]
    for parameters, options in ((scalars, {"lr": 0.015}),
                                (convolutions, {"lr": 0.002}),
                                (no_decay, {"lr": 0.002, "weight_decay": 0.0}),
                                (routers, {"lr": 0.001, "weight_decay": 0.0})):
        if parameters:
            groups.append({"params": parameters, **options})
    if not matrices or not convolutions or not no_decay:
        raise ValueError("Expected dense, short-convolution and no-decay gate groups")
    if bool(routers) != bool(model.config.get("shared_pool")):
        raise ValueError("Routers belong to shared-pool models only")
    result = [torch.optim.AdamW(groups, betas=(0.8, 0.95), eps=1e-10,
                               weight_decay=0.001, fused=True), Muon(matrices)]
    grouped = [p for optimizer in result for group in optimizer.param_groups for p in group["params"]]
    if len(grouped) != len(set(grouped)) or set(grouped) != set(model.parameters()):
        raise ValueError("Gated Delta optimizers must cover every parameter exactly once")
    return result


def pinned_autotune_digest(directory):
    """Hash every JSON file in a pinned config directory.

    FLA reads only the kernel config files, but the pin script's manifest.json is hashed too on
    purpose: the manifest carries the provenance that justifies the kernel selections, and profile
    directories are immutable artifacts, so a rewritten manifest must invalidate every throughput
    report and trainer gate bound to the old digest.
    """
    import hashlib
    from pathlib import Path

    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"FLA_CONFIG_DIR is not a directory: {directory}")
    files = sorted(path for path in directory.iterdir() if path.suffix == ".json")
    if not files:
        raise ValueError(f"FLA_CONFIG_DIR holds no kernel config files: {directory}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def runtime_autotuning_policy():
    """Read import-resolved autotuning policy without changing process settings.

    A pinned config directory is bound by content hash: identical policy
    dictionaries mean identical kernel selections for identical shapes.
    """
    import os
    import triton
    from fla.utils._config import FLA_CACHE_RESULTS
    from fla.utils._compat import autotune_cache_kwargs
    from fla.ops.utils.cache import FLA_CACHE_MODE
    from pathlib import Path
    config_dir = os.environ.get("FLA_CONFIG_DIR")
    root = Path(__file__).resolve().parents[2]
    return dict(fla_cache_results=bool(FLA_CACHE_RESULTS),
                fla_cache_mode=FLA_CACHE_MODE.value,
                fla_config_dir=None if config_dir is None else os.path.relpath(Path(config_dir).resolve(), root),
                fla_config_dir_sha256=None if config_dir is None else pinned_autotune_digest(config_dir),
                triton_cache_autotuning=bool(triton.knobs.autotuning.cache),
                autotune_cache_kwargs=dict(autotune_cache_kwargs),
                effective_triton_persistent_results=bool(
                    (autotune_cache_kwargs.get("cache_results", False) or triton.knobs.autotuning.cache)
                    and not triton.knobs.runtime.interpret))


def _serialize_triton_config(config):
    return dict(kwargs=dict(config.kwargs), num_warps=config.num_warps, num_stages=config.num_stages,
                num_ctas=config.num_ctas, maxnreg=getattr(config, "maxnreg", None))


def fla_autotuner_selections():
    """Snapshot every imported FLA autotuner's in-process kernel selections.

    Entries are keyed exactly as FLA's strict config files key them, so a
    snapshot can be written as a pinned profile or compared against one.
    Plain Triton autotuners (no config-file support) are reported too: if one
    is populated it autotuned in-process under every policy.
    """
    import sys
    from fla.ops.utils.cache import AutotuneKey, CachedAutotuner
    from triton.runtime.autotuner import Autotuner

    found = {}
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("fla.") or module is None:
            continue
        for attribute, value in vars(module).items():
            seen = set()
            while value is not None and id(value) not in seen:
                seen.add(id(value))
                if isinstance(value, Autotuner):
                    pinnable = isinstance(value, CachedAutotuner)
                    entries = {}
                    for key, config in value.cache.items():
                        normalized = AutotuneKey.normalize_autotune_key(list(key))
                        entries[AutotuneKey.key_hash(normalized)] = dict(
                            autotune_key=normalized, config=_serialize_triton_config(config))
                    found[f"{module_name}.{attribute}"] = dict(
                        kernel_name=value.kernel_name if pinnable else value.base_fn.__name__,
                        config_file_support=pinnable, keys=list(value.keys),
                        cache_results=bool(value.cache_results), entries=entries)
                    break
                value = getattr(value, "fn", None)
    return found


def unpinnable_selections(selections):
    """Populated plain Triton autotuners: they retune in-process under every policy."""
    return {name: tuner["entries"] for name, tuner in selections.items()
            if tuner["entries"] and not tuner["config_file_support"]}


def write_pinned_autotune_profile(directory, selections):
    """Write strict-mode FLA config files from populated in-process selections.

    Only FLA's config-file autotuners are written; see unpinnable_selections
    for the kernels a strict profile cannot cover.
    """
    import json
    from pathlib import Path
    import triton

    by_kernel = {}
    for tuner_name, tuner in selections.items():
        if not tuner["config_file_support"]:
            continue
        kernel = by_kernel.setdefault(tuner["kernel_name"], {})
        for key_hash, entry in tuner["entries"].items():
            if key_hash in kernel and kernel[key_hash] != entry:
                raise ValueError(f"Conflicting selections for {tuner['kernel_name']} from {tuner_name}")
            kernel[key_hash] = entry
    by_kernel = {name: entries for name, entries in by_kernel.items() if entries}
    if not by_kernel:
        raise ValueError("No populated FLA autotuner selections to pin")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for kernel_name, entries in sorted(by_kernel.items()):
        content = dict(kernel_name=kernel_name, triton_version=triton.__version__,
                       autotune_entries=dict(sorted(entries.items())))
        (directory / f"{kernel_name}.json").write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
    return sorted(by_kernel)


def pinned_autotune_mismatches(directory, selections):
    """Populated config-file autotuner selections that a strict profile did not dictate."""
    import json
    from pathlib import Path
    from fla.ops.utils.cache import AutotuneKey

    directory = Path(directory)
    mismatches = []
    for tuner_name, tuner in selections.items():
        if not tuner["config_file_support"]:
            continue
        path = directory / f"{tuner['kernel_name']}.json"
        pinned = json.loads(path.read_text()).get("autotune_entries", {}) if path.exists() else {}
        for key_hash, entry in tuner["entries"].items():
            expected = pinned.get(key_hash)
            if (expected is None or expected.get("config") != entry["config"]
                    or AutotuneKey.normalize_autotune_key(expected.get("autotune_key")) != entry["autotune_key"]):
                mismatches.append(dict(tuner=tuner_name, kernel=tuner["kernel_name"], key=entry["autotune_key"],
                                       pinned=expected, observed=entry["config"]))
    return mismatches


def gpu_process_activity(device_index=0):
    """Per-process device use from ``nvidia-smi pmon``: framebuffer MiB and SM share.

    ``pmon`` is the only observer that attributes streaming-multiprocessor time to processes
    outside this interpreter; ``-s um`` reports utilization and memory in one sample. A dash
    means the process showed no activity in the sampling window. Returns None when the
    observer itself fails so callers can separate an unobservable device from a busy one.
    """
    import subprocess

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "pmon", "-c", "1", "-s", "um", "--id", str(device_index)],
            text=True, stderr=subprocess.STDOUT, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    processes, unobservable = [], 0
    for line in output.splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 11 or not fields[1].isdigit():
            unobservable += 1
            continue
        pid, kind, sm, framebuffer = int(fields[1]), fields[2], fields[3], fields[9]
        processes.append(dict(pid=pid, kind=kind,
                              sm_percent=int(sm) if sm.isdigit() else 0,
                              memory_mib=int(framebuffer) if framebuffer.isdigit() else None,
                              command=" ".join(fields[11:])))
        if not framebuffer.isdigit():
            unobservable += 1
    return dict(processes=processes, unobservable=unobservable)


def gpu_utilization_percent(device_index=0):
    """Whole-device SM utilization; before this process computes, all of it is foreign."""
    import subprocess

    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--id={device_index}", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=30)
        return int(output.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def foreign_gpu_use(activity):
    """Summarize other processes' device use from one ``gpu_process_activity`` sample.

    Compute rows (type ``C``) are training or inference jobs; any SM time from one of them is
    contention. Graphics-capable rows (``G`` and ``C+G``: the compositor, terminals, browsers)
    redraw the desktop in short bursts that every measurement on this machine shares, so they
    are tracked separately against a looser limit that only a real graphics workload reaches.
    """
    import os

    own = os.getpid()
    foreign = [p for p in activity["processes"] if p["pid"] != own]
    compute = [p for p in foreign if "G" not in p["kind"]]
    graphics = [p for p in foreign if "G" in p["kind"]]
    busiest_compute = max(compute, key=lambda p: p["sm_percent"], default=None)
    busiest_graphics = max(graphics, key=lambda p: p["sm_percent"], default=None)
    return dict(memory_mib=sum(p["memory_mib"] or 0 for p in foreign),
                compute_sm_percent=busiest_compute["sm_percent"] if busiest_compute else 0,
                busiest_compute_pid=busiest_compute["pid"] if busiest_compute and busiest_compute["sm_percent"] else None,
                graphics_sm_percent=busiest_graphics["sm_percent"] if busiest_graphics else 0,
                busiest_graphics_pid=busiest_graphics["pid"] if busiest_graphics and busiest_graphics["sm_percent"] else None,
                unobservable=activity["unobservable"])


# pmon counts pure-graphics desktop rows (compositor, browsers, terminals: about 3.3 GiB here), and the
# largest production executor reserves 21.6 GiB of the 31.8 GiB device. Compositor redraws showed up as
# a 10% SM sample for the compositor once in 21 samples of a profile run, and the browser's GPU process
# held 11% for two consecutive samples of a throughput run. A browser compute workload (WebGPU, a C+G
# process) instead holds SM time for minutes, so the sampler also limits how many consecutive samples may
# show graphics SM above the compute limit (four passes every desktop burst seen so far with a 2x margin)
# and, beyond that allowance, what share of all samples may do so: a workload that pulses four seconds on
# and one second off would satisfy the run-length rule alone while contaminating 80% of the samples.
EXCLUSIVITY_LIMITS = dict(memory_limit_mib=8192, compute_sm_limit_percent=5, graphics_sm_limit_percent=25,
                          graphics_burst_limit_samples=4, graphics_duty_limit_percent=25,
                          utilization_limit_percent=10)


def wait_for_exclusive_gpu(limits=EXCLUSIVITY_LIMITS, consecutive=3, interval_seconds=10.0,
                           deadline_seconds=4 * 3600, device_index=0):
    """Block until other processes leave the device idle for several consecutive polls.

    Production GDN2 executors reserve most of the device and their timings assume an idle one.
    Work launched outside the queue can occupy it at any moment, so every measurement checks
    this precondition itself before allocating and records what it saw. The memory limit
    keeps the executor from running out of device memory; the per-process SM share and the
    whole-device utilization (entirely foreign before this process computes) catch
    contention from small-footprint processes. Raises on the deadline or when the observer
    keeps failing.
    """
    import time

    started = time.monotonic()
    polls = streak = errors = consecutive_errors = 0
    last = None
    while True:
        activity, utilization = gpu_process_activity(device_index), gpu_utilization_percent(device_index)
        polls += 1
        if activity is None or utilization is None:
            errors += 1
            consecutive_errors += 1
            streak = 0
            if consecutive_errors >= 5:
                raise RuntimeError("nvidia-smi cannot observe the device; refusing to run blind")
        else:
            consecutive_errors = 0
            last = dict(foreign_gpu_use(activity), utilization_percent=utilization)
            idle = (last["unobservable"] == 0 and last["memory_mib"] <= limits["memory_limit_mib"]
                    and last["compute_sm_percent"] <= limits["compute_sm_limit_percent"]
                    and last["graphics_sm_percent"] <= limits["graphics_sm_limit_percent"]
                    and utilization <= limits["utilization_limit_percent"])
            streak = streak + 1 if idle else 0
            if streak >= consecutive:
                return dict(last, waited_seconds=time.monotonic() - started, polls=polls,
                            observer_errors=errors, limits=dict(limits))
        if time.monotonic() - started + interval_seconds > deadline_seconds:
            raise RuntimeError(f"GPU still in use by other processes after {deadline_seconds} s: {last}")
        time.sleep(interval_seconds)


class ForeignGpuSampler:
    """Sample other processes' device use on a daemon thread while a measurement runs.

    A process that arrives and leaves between two point checks would otherwise contaminate
    timings unseen. ``summary`` reports the peaks, how many samples and distinct bursts showed
    graphics SM above the compute limit, and the longest such burst; ``require_exclusive`` fails
    closed when a peak crossed a limit, graphics SM stayed up for more consecutive samples than
    the burst limit, more samples than the burst limit and more than the duty share showed it,
    a process could not be attributed, or the device was never observed. A sample the observer
    could not take is blind, so it extends the current burst instead of ending it.
    The thread publishes each sample as one tuple assignment, so a summary taken while it runs
    is always internally consistent.
    """

    def __init__(self, limits=EXCLUSIVITY_LIMITS, interval_seconds=1.0, device_index=0):
        import threading

        self.limits, self.interval_seconds, self.device_index = dict(limits), interval_seconds, device_index
        # samples, observer_errors, unobservable, peak_memory_mib, peak_compute_sm, busiest_compute_pid,
        # peak_graphics_sm, busiest_graphics_pid, current_graphics_burst, longest_graphics_burst,
        # graphics_samples_above_compute_limit, graphics_bursts
        self._state = (0, 0, 0, 0, 0, None, 0, None, 0, 0, 0, 0)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="foreign-gpu-sampler", daemon=True)

    def _run(self):
        while not self._stop.is_set():
            activity = gpu_process_activity(self.device_index)
            (samples, errors, unobservable, peak_memory, peak_compute, compute_pid,
             peak_graphics, graphics_pid, burst, longest_burst, above_limit, bursts) = self._state
            if activity is None:
                errors += 1
                burst += 1
                longest_burst = max(longest_burst, burst)
            else:
                use = foreign_gpu_use(activity)
                samples += 1
                unobservable = max(unobservable, use["unobservable"])
                peak_memory = max(peak_memory, use["memory_mib"])
                if use["compute_sm_percent"] > peak_compute:
                    peak_compute, compute_pid = use["compute_sm_percent"], use["busiest_compute_pid"]
                if use["graphics_sm_percent"] > peak_graphics:
                    peak_graphics, graphics_pid = use["graphics_sm_percent"], use["busiest_graphics_pid"]
                if use["graphics_sm_percent"] > self.limits["compute_sm_limit_percent"]:
                    above_limit += 1
                    bursts += burst == 0
                    burst += 1
                    longest_burst = max(longest_burst, burst)
                else:
                    burst = 0
            self._state = (samples, errors, unobservable, peak_memory, peak_compute, compute_pid,
                           peak_graphics, graphics_pid, burst, longest_burst, above_limit, bursts)
            self._stop.wait(self.interval_seconds)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def summary(self):
        (samples, errors, unobservable, peak_memory, peak_compute, compute_pid,
         peak_graphics, graphics_pid, _, longest_burst, above_limit, bursts) = self._state
        allowance = max(self.limits["graphics_burst_limit_samples"],
                        samples * self.limits["graphics_duty_limit_percent"] / 100)
        exclusive = (samples > 0 and unobservable == 0 and peak_memory <= self.limits["memory_limit_mib"]
                     and peak_compute <= self.limits["compute_sm_limit_percent"]
                     and peak_graphics <= self.limits["graphics_sm_limit_percent"]
                     and longest_burst <= self.limits["graphics_burst_limit_samples"]
                     and above_limit <= allowance)
        return dict(samples=samples, observer_errors=errors, unobservable=unobservable,
                    peak_foreign_memory_mib=peak_memory,
                    peak_foreign_compute_sm_percent=peak_compute, busiest_foreign_compute_pid=compute_pid,
                    peak_foreign_graphics_sm_percent=peak_graphics, busiest_foreign_graphics_pid=graphics_pid,
                    longest_foreign_graphics_burst_samples=longest_burst,
                    foreign_graphics_samples_above_compute_limit=above_limit, foreign_graphics_bursts=bursts,
                    limits=dict(self.limits), exclusive=exclusive)

    def require_exclusive(self):
        summary = self.summary()
        if summary["samples"] == 0:
            raise RuntimeError(f"Device use by other processes was never observed: {summary}")
        if summary["unobservable"]:
            raise RuntimeError(f"Device use by {summary['unobservable']} process(es) could not be attributed: {summary}")
        if not summary["exclusive"]:
            raise RuntimeError(f"Another process used the device during measurement: {summary}")
        return summary


def runtime_dependency_versions():
    from importlib.metadata import PackageNotFoundError, version
    result = {}
    for distribution in ("fla-core", "flash-linear-attention", "triton"):
        try:
            result[distribution] = version(distribution)
        except PackageNotFoundError:
            result[distribution] = None
    return result


def _fla_dependency_provenance(required):
    """Hash installed FLA sources without importing its model/frontend modules.

    RECORD mismatches are disclosed rather than silently called upstream code.
    Benchmark and training must agree on the exact installed sources.
    """
    import base64
    import hashlib
    from importlib.metadata import distribution

    versions, sources, mismatches = {}, {}, []
    for name in ("fla-core", "flash-linear-attention"):
        installed = distribution(name)
        versions[name] = installed.version
        files = installed.files
        if files is None:
            raise ValueError(f"Missing installed file manifest for {name}")
        for entry in files:
            relative = str(entry)
            if not relative.startswith("fla/") or not relative.endswith(".py"):
                continue
            digest = hashlib.sha256(installed.locate_file(entry).read_bytes()).digest()
            key = f"{name}:{relative}"
            sources[key] = digest.hex()
            recorded = entry.hash
            actual = base64.urlsafe_b64encode(digest).decode().rstrip("=")
            if recorded is None or recorded.mode != "sha256" or recorded.value != actual:
                mismatches.append(key)
    for suffix in required:
        if not any(key.endswith(":" + suffix) for key in sources):
            raise ValueError(f"Missing installed FLA source: {suffix}")
    return {"distribution_versions": versions, "source_sha256": dict(sorted(sources.items())),
            "wheel_record_mismatches": sorted(mismatches)}


def scalar_dependency_provenance():
    return _fla_dependency_provenance(("fla/layers/gated_deltanet.py", "fla/ops/gated_delta_rule/chunk.py"))


def gdn2_dependency_provenance():
    """Bind optimized GDN2 and its installed FLA helpers to executed source."""
    return _fla_dependency_provenance(("fla/ops/gdn2/chunk.py",))


def verify_throughput_report(args):
    """Require current complete timing evidence; explicitly mark slow diagnostics."""
    import hashlib
    import json
    import math
    import statistics
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    path = Path(args.throughput_report)
    if not path.is_absolute():
        path = root / path
    raw = path.read_bytes()
    report = json.loads(raw)
    candidate, baseline = report["candidate"], report["baseline"]
    if args.architecture not in {"gdn1", "gdn2"}:
        raise ValueError("Throughput report must target gdn1 or gdn2")
    scalar = args.architecture == "gdn1"
    seq_len = getattr(args, "seq_len", 1024)
    if seq_len not in (1024, 4096) or (scalar and seq_len != 1024):
        raise ValueError("Unsupported Gated Delta sequence length")
    if args.microbatch <= 0 or (524288 // seq_len) % args.microbatch:
        raise ValueError("Microbatch must divide packed training rows")
    diagnostic = args.allow_slow_diagnostic
    if diagnostic and scalar:
        raise ValueError("Slow diagnostics are only supported for GDN2")
    expected_config = dict(vocab_size=1024, num_layers=6, model_dim=512,
                           head_dim=args.memory_head_dim, mixer_dim=args.memory_width, expand_v=args.value_expansion, use_short_conv=True,
                           conv_size=4, allow_neg_eigval=False, mixer_norm_eps=1e-5,
                           kernel_chunk_size=64, fused_projections=args.fused_projections)
    if scalar:
        expected_config.update(memory_rule="scalar_delta", initialization="gdn2_matched_distributions")
    else:
        expected_config.update(gdn_backend=args.gdn_backend, state_v_first=args.state_v_first,
                               disable_recompute=args.disable_recompute, custom_ops=args.custom_ops,
                               gate_in_kernel=args.gate_in_kernel)
        if args.shared_pool:
            from pretraining.nanogpt_mini.gated_delta_pool import pool_policy
            expected_config.update(pool_policy(args))
    expected_flags = {"status": "completed",
                      "batch_tokens": 524288, "seq_len": seq_len,
                      "optimizer_included": True, "compiled": True,
                      "cuda_graph": True, "checkpointing": False}
    for key, value in expected_flags.items():
        if report.get(key) != value:
            raise ValueError(f"Gated Delta throughput report has invalid {key}")
    expected_model = "scalar_delta" if scalar else "gated_delta"
    if candidate.get("model") != expected_model:
        raise ValueError("Gated Delta throughput candidate model differs")
    if report.get("repeats", 0) < 5 or baseline.get("model") != "nanogpt_mini":
        raise ValueError("Gated Delta gate needs at least five updates against plain mini")
    if (candidate.get("model_config") != expected_config or args.segment_size != 64
            or candidate.get("chunk_size") != 64 or candidate.get("microbatch") != args.microbatch
            or baseline.get("microbatch") != 65536 // seq_len):
        raise ValueError("Gated Delta throughput configuration does not match training")
    if seq_len == 4096:
        if report.get("train_seq_len") != seq_len or report.get("val_seq_len") != seq_len:
            raise ValueError("4K report must bind training and validation contexts")
        for arm in (candidate, baseline):
            if (arm.get("seq_len") != seq_len or arm.get("validation_seq_len") != seq_len
                    or arm.get("validation_microbatch") != arm.get("microbatch")
                    or arm.get("validation_residency_passed") is not True
                    or arm.get("microsteps_per_update") != 524288 // (seq_len * arm["microbatch"])):
                raise ValueError("4K report must bind full graph and accumulation shapes")
    if scalar and report.get("scalar_dependency_provenance") != scalar_dependency_provenance():
        raise ValueError("Scalar delta installed-source provenance differs")
    if not scalar and report.get("gdn2_dependency_provenance") != gdn2_dependency_provenance():
        raise ValueError("GDN2 installed-source provenance differs")
    if not scalar and (candidate.get("validation_microbatch") != args.microbatch
                       or candidate.get("validation_residency_passed") is not True):
        raise ValueError("GDN2 needs co-resident training and validation graph qualification")
    policy = runtime_autotuning_policy()
    if report.get("autotuning_policy") != policy:
        raise ValueError("Gated Delta throughput autotuning policy differs")
    if report.get("dependency_versions") != runtime_dependency_versions():
        raise ValueError("Gated Delta throughput dependency versions differ")
    if report.get("torch") != str(torch.__version__):
        raise ValueError("Gated Delta throughput PyTorch version differs")
    if report.get("gpu") != torch.cuda.get_device_name():
        raise ValueError("Gated Delta throughput GPU differs")
    baseline_rate, candidate_rate = baseline["tokens_per_second"], candidate["tokens_per_second"]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
               for value in (baseline_rate, candidate_rate)):
        raise ValueError("Gated Delta throughput must be finite and positive")
    if not diagnostic and candidate_rate < 1.05 * baseline_rate:
        raise ValueError("Gated Delta throughput improvement is below five percent")
    for arm in (baseline, candidate):
        samples = arm.get("update_seconds", [])
        if (len(samples) < 5 or arm.get("measured_optimizer_updates", 0) != len(samples)
                or arm.get("warmup_optimizer_updates", 0) < 5
                or not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                           for value in samples)):
            raise ValueError("Gated Delta gate requires complete positive timing samples")
        if not math.isclose(arm["tokens_per_second"], 524288 / statistics.median(samples), rel_tol=1e-6):
            raise ValueError("Gated Delta throughput rate disagrees with measured update times")
    separated = max(candidate["update_seconds"]) < min(baseline["update_seconds"])
    speed_passed = candidate_rate >= 1.05 * baseline_rate and separated
    if report.get("gate_passed") is not speed_passed:
        raise ValueError("Gated Delta gate_passed disagrees with measured update times")
    if not diagnostic and not separated:
        raise ValueError("Gated Delta repeated timings overlap; repeat benchmark before training")
    files = [root / name for name in (
        "pretraining/nanogpt_mini/gated_delta_model.py",
        "pretraining/nanogpt_mini/gated_delta_ops.py",
        "pretraining/nanogpt_mini/gated_delta_pool.py",
        "pretraining/nanogpt_mini/gated_delta_bank_linear.py",
        "pretraining/nanogpt_mini/gated_delta_runtime.py",
        "pretraining/nanogpt_mini/chunk_memory_runtime.py",
        "pretraining/nanogpt_mini/nanogpt_mini_model.py",
        "pretraining/nanogpt_mini/recurrent_slots_runtime.py",
        "scripts/train_recurrent_slots.py", "scripts/benchmark_gated_delta.py",
        "scripts/benchmark_chunk_memory.py")]
    if scalar:
        files.append(root / "pretraining/nanogpt_mini/scalar_delta_model.py")
    vendor = root / "pretraining/gated_delta/vendor"
    if not vendor.is_dir():
        raise ValueError("Missing pinned GDN-2 kernel directory")
    files.extend(p for p in vendor.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for source in files:
        key = str(source.relative_to(root))
        if report.get("source_sha256", {}).get(key) != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError(f"Gated Delta throughput source mismatch: {key}")
    return dict(gpu=report["gpu"], path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                autotuning_policy=policy,
                candidate_tokens_per_second=candidate_rate, baseline_tokens_per_second=baseline_rate,
                speed_passed=speed_passed, speedup=candidate_rate / baseline_rate,
                diagnostic_only=not speed_passed)
