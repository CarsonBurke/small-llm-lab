"""Isolate optimized GDN-2 full-update costs against plain mini; use mlq.

Both arms process the same GPU-resident 524288-token update at T1024.
Plain mini stays at B64; GDN-2 may use B32/B64/B128. Training and validation
CUDA graphs remain co-resident, matching the throughput benchmark.
Five uninstrumented full updates establish wall time. A separate full update
has synchronized stage timings, then one warmed microbatch replay and one
optimizer update are profiled independently. These optimizer updates measure
execution cost only; they are not quality evidence and save no checkpoint.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from scripts.benchmark_chunk_memory import PlainMini, optimizers
from scripts.train_recurrent_slots import atomic_json
from pretraining.nanogpt_mini.chunk_memory_runtime import CompiledFullLoss
from pretraining.nanogpt_mini.gated_delta_runtime import (
    CompiledGatedDeltaLoss, build_gated_delta_optimizers, runtime_dependency_versions,
    gdn2_dependency_provenance,
)
from pretraining.nanogpt_mini.recurrent_slots_runtime import CUDAGraphMicrobatch, CUDAGraphValidation


def cuda_kernel_report(trace_path):
    """Aggregate actual CUDA kernel events, excluding annotation spans/copies."""
    raw = trace_path.read_bytes()
    events = json.loads(raw)["traceEvents"]
    totals = defaultdict(lambda: {"calls": 0, "total_cuda_us": 0.0})
    for event in events:
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        duration = event["dur"]
        if not isinstance(duration, (int, float)) or duration < 0:
            raise ValueError(f"Invalid CUDA kernel duration in {trace_path}")
        item = totals[event["name"]]
        item["calls"] += 1
        item["total_cuda_us"] += duration
    kernels = [dict(name=name, **item) for name, item in totals.items()]
    kernels.sort(key=lambda item: item["total_cuda_us"], reverse=True)
    total = sum(item["total_cuda_us"] for item in kernels)
    for item in kernels:
        item["fraction_cuda_time"] = item["total_cuda_us"] / total if total else 0.0
    return dict(total_cuda_us=total, kernel_events=sum(item["calls"] for item in kernels),
                kernels=kernels, aggregation="chrome_trace_cat_kernel_complete_events_only",
                trace_file=trace_path.name, trace_sha256=hashlib.sha256(raw).hexdigest())


def reaggregate_existing(output):
    """Repair v1 totals from raw traces without executing CUDA/model workloads."""
    original_path = output / "profile.v1.json"
    original_source = output / "profile_gated_delta.v1.py"
    original_bytes = original_path.read_bytes()
    report = json.loads(original_bytes)
    if hashlib.sha256(original_source.read_bytes()).hexdigest() != report["source_sha256"]["scripts/profile_gated_delta.py"]:
        raise ValueError("Preserved v1 profiler source does not match execution provenance")
    for label in ("plain_mini", "gdn2"):
        if label not in report:
            continue
        for name in ("microbatch_replay", "optimizer"):
            report[label][f"{name}_profile"] = cuda_kernel_report(output / f"{label}_{name}.json")
    report["aggregation_correction"] = dict(
        version=2, gpu_execution_repeated=False,
        reason="v1 counted CUDA user annotations and non-kernel events, double-counting annotated spans",
        included_categories=["kernel"],
        excluded_categories=["gpu_user_annotation", "gpu_memcpy", "gpu_memset"],
        original_report=original_path.name,
        original_report_sha256=hashlib.sha256(original_bytes).hexdigest(),
        original_execution_source=original_source.name,
        aggregation_source="scripts/profile_gated_delta.py",
        aggregation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(output / "profile.json", report)
    return report


def profile_arm(label, inputs, targets, output, repeats, *, microbatch, model_options):
    torch.manual_seed(1337)
    if label == "gdn2":
        from pretraining.nanogpt_mini.gated_delta_model import GatedDeltaGPT
        model = GatedDeltaGPT(**model_options).cuda().train()
        loss_fn = CompiledGatedDeltaLoss(model, segment_size=64)
        opts = build_gated_delta_optimizers(model)
    else:
        model = PlainMini().cuda().train()
        loss_fn = CompiledFullLoss(model, segment_size=64)
        opts = optimizers(model)
    before_capture = dict(allocated_mib=torch.cuda.memory_allocated() / 2**20,
                          reserved_mib=torch.cuda.memory_reserved() / 2**20)
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    validation = CUDAGraphValidation(loss_fn, batch_size=microbatch, seq_len=1024)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        validation.replay(inputs[:microbatch], targets[:microbatch])
    model.train()
    graph = CUDAGraphMicrobatch(loss_fn, batch_size=microbatch, seq_len=1024)
    if label == "gdn2":
        loss_fn.audit_graph_breaks()
    after_capture = dict(allocated_mib=torch.cuda.memory_allocated() / 2**20,
                         reserved_mib=torch.cuda.memory_reserved() / 2**20,
                         peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                         peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)

    def replay_update():
        total = torch.zeros((), device="cuda")
        for row in range(0, 512, microbatch):
            total += graph.replay(inputs[row:row + microbatch], targets[row:row + microbatch])
        return total

    def finite_check(loss):
        gradient_max = torch.stack(torch._foreach_norm(
            [p.grad for p in model.parameters()], float("inf"))).amax()
        if not bool(torch.isfinite(gradient_max) & torch.isfinite(loss)):
            raise FloatingPointError(f"{label}: nonfinite loss or gradient")

    def optimizer_step():
        for optimizer in opts:
            optimizer.step()

    def full_update():
        graph.zero_grad()
        loss = replay_update()
        finite_check(loss)
        optimizer_step()
        return loss

    for _ in range(5):
        full_update()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    full_times, losses = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        loss = full_update()
        torch.cuda.synchronize()
        full_times.append((time.perf_counter() - start) * 1000)
        losses.append(float(loss) / 524288)
    steady_memory = dict(allocated_mib=torch.cuda.memory_allocated() / 2**20,
                         reserved_mib=torch.cuda.memory_reserved() / 2**20,
                         peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                         peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)

    def timed_stage(function):
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record()
        result = function()
        end.record()
        torch.cuda.synchronize()
        return result, dict(wall_ms=(time.perf_counter() - start) * 1000,
                            gpu_stream_ms=begin.elapsed_time(end))

    _, zero_time = timed_stage(graph.zero_grad)
    loss, replay_time = timed_stage(replay_update)
    _, finite_time = timed_stage(lambda: finite_check(loss))
    _, optimizer_time = timed_stage(optimizer_step)
    stages = dict(zero_grad=zero_time, microbatch_replays=replay_time,
                  finite_check=finite_time, optimizer_step=optimizer_time)
    stages["total_instrumented_wall_ms"] = sum(value["wall_ms"] for value in stages.values())

    def trace(name, function):
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profiler:
            with torch.profiler.record_function(name):
                function()
            torch.cuda.synchronize()
        (output / f"{label}_{name}.txt").write_text(
            profiler.key_averages().table(sort_by="self_device_time_total", row_limit=80))
        trace_path = output / f"{label}_{name}.json"
        profiler.export_chrome_trace(str(trace_path))
        return cuda_kernel_report(trace_path)

    graph.zero_grad()
    replay_profile = trace("microbatch_replay", lambda: graph.replay(inputs[:microbatch], targets[:microbatch]))
    # Rebuild the full update's gradients outside the optimizer profiler.
    graph.zero_grad()
    loss = replay_update()
    finite_check(loss)
    optimizer_profile = trace("optimizer", optimizer_step)
    result = dict(model_config=model.config, microbatch=microbatch, microbatches_per_update=512 // microbatch,
                  parameters=sum(p.numel() for p in model.parameters()),
                  full_update_ms=full_times, median_full_update_ms=statistics.median(full_times),
                  tokens_per_second=524288000 / statistics.median(full_times),
                  losses=losses, stages=stages, microbatch_replay_profile=replay_profile,
                  optimizer_profile=optimizer_profile, memory_before_capture=before_capture,
                  memory_after_capture=after_capture, memory_steady_state=steady_memory,
                  capture_seconds=graph.capture_seconds, preparation_seconds=graph.preparation_seconds,
                  warmup_optimizer_updates=5, timed_optimizer_updates=repeats,
                  diagnostic_optimizer_updates=2,
                  profiler_microbatch_tokens=microbatch * 1024, optimizer_batch_tokens=524288)
    if label == "gdn2":
        result["graph_breaks"] = loss_fn.audit_graph_breaks()
        result["short_convolution_backends"] = {
            name: module.backend for name, module in model.named_modules()
            if name.endswith(("q_conv1d", "k_conv1d", "v_conv1d"))}
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        val_loss = validation.replay(inputs[:microbatch], targets[:microbatch])[0]
        if not bool(torch.isfinite(val_loss)):
            raise FloatingPointError(f"{label}: nonfinite co-resident validation loss")
    result.update(validation_microbatch=microbatch, validation_residency_passed=True,
                  validation_capture_seconds=validation.capture_seconds,
                  validation_preparation_seconds=validation.preparation_seconds)
    torch.cuda.synchronize()
    validation.graph.reset()
    graph.graph.reset()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "ablation_results/gated_delta_profile")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--microbatch", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--gdn-backend", choices=("vendor", "fla"), default="vendor")
    parser.add_argument("--state-v-first", action="store_true")
    parser.add_argument("--disable-recompute", action="store_true")
    parser.add_argument("--fused-projections", action="store_true")
    parser.add_argument("--architecture", choices=("both", "gdn2", "plain_mini"), default="both")
    parser.add_argument("--reaggregate-existing", action="store_true",
                        help="repair preserved v1 report using existing traces, without CUDA execution")
    args = parser.parse_args(argv)
    if args.reaggregate_existing:
        reaggregate_existing(args.output)
        return
    if args.repeats < 5:
        parser.error("at least five complete timed updates are required")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Profiling requires CUDA bf16")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "profile.json").exists() or (args.output / "source").exists():
        raise FileExistsError("Choose a fresh profile output directory")
    model_options = dict(gdn_backend=args.gdn_backend, state_v_first=args.state_v_first,
                         disable_recompute=args.disable_recompute, fused_projections=args.fused_projections)
    sources = [ROOT / p for p in (
        "scripts/profile_gated_delta.py", "scripts/benchmark_chunk_memory.py",
        "scripts/train_recurrent_slots.py", "pretraining/nanogpt_mini/gated_delta_model.py",
        "pretraining/nanogpt_mini/gated_delta_runtime.py", "pretraining/nanogpt_mini/chunk_memory_runtime.py",
        "pretraining/nanogpt_mini/nanogpt_mini_model.py", "pretraining/nanogpt_mini/recurrent_slots_runtime.py")]
    sources.extend(p for p in (ROOT / "pretraining/gated_delta/vendor").rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts)
    report = dict(status="running", purpose="execution_cost_diagnosis_not_quality_evidence",
                  batch_tokens=524288, seq_len=1024, microbatch=args.microbatch, baseline_microbatch=64,
                  model_options=model_options, warmup_optimizer_updates=5,
                  validation_training_graphs_co_resident=True,
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  dependency_versions=runtime_dependency_versions(),
                  gdn2_dependency_provenance=gdn2_dependency_provenance(),
                  host_data_transfer_included=False, optimizer_included=True,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sources})
    for source in sources:
        destination = args.output / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    from importlib.metadata import distribution
    for key, digest in report["gdn2_dependency_provenance"]["source_sha256"].items():
        package, relative = key.split(":", 1)
        content = distribution(package).locate_file(relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError(f"Installed source changed while snapshotting: {key}")
        destination = args.output / "installed_source" / package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    path = args.output / "profile.json"
    atomic_json(path, report)
    try:
        torch.manual_seed(1337)
        inputs = torch.randint(1024, (512, 1024), device="cuda", dtype=torch.int32)
        targets = inputs.roll(-1, 1).long()
        arms = ("plain_mini", "gdn2") if args.architecture == "both" else (args.architecture,)
        for label in arms:
            print(f"profiling {label} full optimizer updates and CUDA kernels", flush=True)
            report[label] = profile_arm(label, inputs, targets, args.output, args.repeats,
                                        microbatch=64 if label == "plain_mini" else args.microbatch,
                                        model_options=model_options)
            atomic_json(path, report)
            print(json.dumps({"architecture": label, "tokens_per_second": report[label]["tokens_per_second"],
                              "stages": report[label]["stages"]}), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = "completed"
        atomic_json(path, report)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(path, report)
        raise


if __name__ == "__main__":
    main()
