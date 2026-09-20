"""Profile the actual full-update latent-carry CUDA graph; submit through mlq.

No source changes to the running trainer/model. Five complete optimizer warmups
precede one uninstrumented update and one or two profiled full updates. Timing
is diagnostic only, with no quality claim or saved model checkpoint. Inputs are
GPU resident and each update contains exactly 524288 tokens at context1024.

Kernel aggregation uses only Chrome trace cat=kernel complete events, avoiding
CUDA user-annotation double counting. The trace is streamed from the output
folder to bound parsing memory; --keep-trace retains it after aggregation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from scripts.train_latent_carry import make_model_and_loss, make_optimizers, source_hashes
from scripts.train_recurrent_slots import atomic_json
from pretraining.nanogpt_mini.recurrent_slots_runtime import CUDAGraphMicrobatch


def trace_events(path):
    """Stream Torch's Chrome trace array without retaining the full JSON tree."""
    decoder = json.JSONDecoder()
    with path.open() as handle:
        buffer = ""
        while '"traceEvents"' not in buffer:
            chunk = handle.read(65536)
            if not chunk:
                raise ValueError("Missing traceEvents array")
            buffer += chunk
        buffer = buffer[buffer.index('"traceEvents"') + len('"traceEvents"'):]
        while "[" not in buffer:
            chunk = handle.read(65536)
            if not chunk:
                raise ValueError("Missing traceEvents opening bracket")
            buffer += chunk
        buffer = buffer[buffer.index("[") + 1:]
        while True:
            buffer = buffer.lstrip(" \t\r\n,")
            if buffer.startswith("]"):
                return
            try:
                event, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = handle.read(65536)
                if not chunk:
                    raise ValueError("Incomplete traceEvents array") from None
                buffer += chunk
                continue
            yield event
            buffer = buffer[end:]


def kernel_category(name):
    """Name-based hints, not attribution to specific model operations."""
    lower = name.lower()
    if "gemm" in lower or "cublas" in lower:
        return "gemm"
    if any(word in lower for word in ("reduce", "reduction", "triton_red", "norm", "_sum")):
        return "reduction_or_normalization"
    if any(word in lower for word in ("triton_poi", "elementwise", "vectorized")):
        return "pointwise_or_vectorized"
    return "other"


def aggregate_trace(path):
    kernels = defaultdict(lambda: {"calls": 0, "self_cuda_us": 0.0})
    categories = defaultdict(lambda: {"calls": 0, "self_cuda_us": 0.0})
    excluded = defaultdict(int)
    for event in trace_events(path):
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            excluded[str(event.get("cat"))] += 1
            continue
        value = kernels[event["name"]]
        value["calls"] += 1
        value["self_cuda_us"] += event["dur"]
    rows = [dict(name=name, **value) for name, value in kernels.items()]
    rows.sort(key=lambda row: row["self_cuda_us"], reverse=True)
    total = sum(row["self_cuda_us"] for row in rows)
    for row in rows:
        row["cuda_fraction"] = row["self_cuda_us"] / total if total else 0.0
        category = categories[kernel_category(row["name"])]
        category["calls"] += row["calls"]
        category["self_cuda_us"] += row["self_cuda_us"]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return dict(kernel_events=sum(row["calls"] for row in rows), total_kernel_cuda_us=total,
                kernels=rows, name_based_categories=dict(categories), excluded_event_categories=dict(excluded),
                trace_bytes=path.stat().st_size, trace_sha256=digest.hexdigest(),
                aggregation="complete Chrome events with cat=kernel only")


def memory_stats():
    return dict(allocated_mib=torch.cuda.memory_allocated() / 2**20,
                reserved_mib=torch.cuda.memory_reserved() / 2**20,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microbatch", type=int, choices=(64, 512), default=512)
    parser.add_argument("--profile-updates", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep-trace", action="store_true")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Latent-carry profiling requires CUDA bf16")
    output = args.output or ROOT / f"ablation_results/latent_carry_profile_b{args.microbatch}"
    output.mkdir(parents=True, exist_ok=True)
    sources = source_hashes()
    sources["scripts/profile_latent_carry.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = dict(status="running", purpose="execution_cost_diagnosis_not_quality_evidence",
                  batch_tokens=524288, seq_len=1024, microbatch=args.microbatch,
                  segment_size=16, optimizer_included=True, cuda_graph=True,
                  host_data_transfer_included=False, warmup_optimizer_updates=5,
                  profile_optimizer_updates=args.profile_updates,
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  source_sha256=sources,
                  caveats=["CUPTI instrumentation changes runtime; use uninstrumented time for context.",
                           "CUDA graph kernels may lack CPU operator attribution.",
                           "Kernel categories are name-based hints, not proof of memory bandwidth limitation.",
                           "Checkpoint recomputation executes forward kernels again during backward; this trace does not uniquely label those calls.",
                           "Summed kernel durations are distinct from GPU utilization and may differ from wall time."])
    path = output / "profile.json"
    atomic_json(path, report)
    try:
        torch.manual_seed(1337)
        inputs = torch.randint(1024, (512, 1024), device="cuda", dtype=torch.int32)
        targets = inputs.roll(-1, 1).long()
        torch.manual_seed(1337)
        model, loss_fn = make_model_and_loss("carry")
        model.train()
        optimizers = make_optimizers(model)
        report["model_config"] = model.config
        report["parameters"] = sum(p.numel() for p in model.parameters())
        report["memory_before_capture"] = memory_stats()
        torch.cuda.reset_peak_memory_stats()
        print("preparing existing compiled full-BPTT CUDA graph", flush=True)
        graph = CUDAGraphMicrobatch(loss_fn, batch_size=args.microbatch, seq_len=1024)
        report["memory_after_capture"] = memory_stats()
        report["capture_seconds"] = graph.capture_seconds
        report["preparation_seconds"] = graph.preparation_seconds
        atomic_json(path, report)

        def update(annotate=False):
            from contextlib import nullcontext
            region = torch.profiler.record_function if annotate else lambda _name: nullcontext()
            with region("zero_grad"):
                graph.zero_grad()
            with region("forward_backward_cuda_graph"):
                total = torch.zeros((), device="cuda")
                for row in range(0, 512, args.microbatch):
                    total += graph.replay(inputs[row:row + args.microbatch], targets[row:row + args.microbatch])
            with region("finite_gradient_and_loss_check"):
                gradients = [parameter.grad for parameter in model.parameters()]
                if any(gradient is None for gradient in gradients):
                    raise RuntimeError("Disconnected parameter")
                maximum = torch.stack(torch._foreach_norm(gradients, float("inf"))).amax()
                if not bool(torch.isfinite(maximum) & torch.isfinite(total)):
                    raise FloatingPointError("Nonfinite profile loss or gradient")
            with region("optimizers"):
                for optimizer in optimizers:
                    optimizer.step()
                graph.zero_grad()
            return total

        print("warming five complete optimizer updates", flush=True)
        for _ in range(5):
            update()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        loss = update()
        torch.cuda.synchronize()
        report["uninstrumented_update_seconds"] = time.perf_counter() - start
        report["uninstrumented_tokens_per_second"] = 524288 / report["uninstrumented_update_seconds"]
        report["uninstrumented_train_loss"] = float(loss) / 524288
        report["memory_steady_state"] = memory_stats()
        atomic_json(path, report)
        print("profiling full update CPU dispatch and CUDA kernels", flush=True)
        start = time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=False, profile_memory=False, with_stack=False) as profiler:
            for _ in range(args.profile_updates):
                update(annotate=True)
            torch.cuda.synchronize()
        report["instrumented_seconds_including_profiler_processing"] = time.perf_counter() - start
        averages = profiler.key_averages()
        (output / "operations.txt").write_text(averages.table(sort_by="self_device_time_total", row_limit=100))
        operations = [dict(name=event.key, calls=event.count,
                           self_cpu_us=event.self_cpu_time_total, total_cpu_us=event.cpu_time_total,
                           self_cuda_us=event.self_device_time_total, total_cuda_us=event.device_time_total,
                           user_annotation=bool(event.is_user_annotation))
                      for event in averages if event.device_type == torch.autograd.DeviceType.CPU]
        operations.sort(key=lambda event: event["self_cuda_us"], reverse=True)
        report["cpu_operation_aggregates"] = operations
        trace_path = output / "trace.json"
        profiler.export_chrome_trace(str(trace_path))
        # CUPTI events can be numerous; release profiler storage before streaming
        # the exported events, and never place this trace in RAM-backed /tmp.
        del averages, profiler
        gc.collect()
        report["cuda"] = aggregate_trace(trace_path)
        report["raw_trace_retained"] = args.keep_trace
        if not args.keep_trace:
            trace_path.unlink()
        lines = ["CUDA kernel self time (complete cat=kernel events only)",
                 "Milliseconds  Calls  Percent  Kernel"]
        for row in report["cuda"]["kernels"]:
            lines.append(f"{row['self_cuda_us'] / 1000:12.3f} {row['calls']:6d} {100 * row['cuda_fraction']:7.2f}  {row['name']}")
        (output / "kernels.txt").write_text("\n".join(lines) + "\n")
        report["status"] = "completed"
        atomic_json(path, report)
        print(json.dumps({"report": str(path), "uninstrumented_tokens_per_second": report["uninstrumented_tokens_per_second"],
                          "kernel_events": report["cuda"]["kernel_events"],
                          "total_kernel_cuda_ms": report["cuda"]["total_kernel_cuda_us"] / 1000,
                          "categories": report["cuda"]["name_based_categories"]}), flush=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(path, report)
        raise


if __name__ == "__main__":
    main()
