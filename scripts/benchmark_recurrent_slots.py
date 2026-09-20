"""Profile and verify exact recurrent execution; submit through mlq."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from pretraining.nanogpt_mini.recurrent_slots import RecurrentSlots
from pretraining.nanogpt_mini.recurrent_slots_runtime import (
    CUDAGraphMicrobatch, CUDAGraphValidation, RecurrentLoss,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; never run this benchmark on CPU")
    if args.repeats < 3:
        parser.error("at least three repeated measurements are required")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1337)
    model = RecurrentSlots().cuda().train()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("proj.weight"):
                parameter.normal_(std=0.003)
    model.compile_segments()
    loss_fn = RecurrentLoss(model)
    inputs = torch.randint(1024, (args.microbatch, 1024), device="cuda", dtype=torch.int32)
    targets = inputs.roll(-1, 1).long()
    alternate = (inputs + 71) % 1024
    alternate_targets = alternate.roll(-1, 1).long()

    def ordinary(x=inputs, y=targets):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(x, y)
        loss.backward()
        return loss.detach()

    def clear():
        model.zero_grad(set_to_none=False)

    print("warming exact compiled forward/backward", flush=True)
    ordinary()
    clear()
    ordinary()
    clear()
    torch.cuda.synchronize()

    def measure(function):
        samples = []
        for _ in range(args.repeats):
            clear()
            torch.cuda.synchronize()
            start = time.perf_counter()
            function()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        return {"samples_ms": samples, "median_ms": statistics.median(samples)}

    torch.cuda.reset_peak_memory_stats()
    baseline = measure(ordinary)
    baseline["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    print(json.dumps({"baseline": baseline}), flush=True)
    clear()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        ordinary()
        torch.cuda.synchronize()
    events = profiler.events()
    cuda_events = [event for event in events if event.device_type == torch.autograd.DeviceType.CUDA]
    profile = {
        "cuda_kernel_events": len(cuda_events),
        "summed_cuda_event_ms": sum(event.device_time_total for event in cuda_events) / 1000,
        "self_cpu_event_ms": sum(event.self_cpu_time_total for event in events) / 1000,
    }
    (args.output / "baseline_profile.txt").write_text(
        profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=40))
    (args.output / "baseline_benchmark.json").write_text(
        json.dumps({"baseline": baseline, "profile": profile}, indent=2) + "\n")
    print(json.dumps({"profile": profile}), flush=True)

    # Two different microbatches must accumulate, not overwrite captured grads.
    clear()
    reference_loss = ordinary().clone() + ordinary(alternate, alternate_targets).clone()
    reference_gradients = {name: parameter.grad.detach().float().clone()
                           for name, parameter in model.named_parameters()}
    print("capturing complete microbatch forward/backward", flush=True)
    graph = CUDAGraphMicrobatch(loss_fn, batch_size=args.microbatch, seq_len=1024)
    graph.zero_grad()
    actual_loss = graph.replay(inputs, targets).clone()
    actual_loss += graph.replay(alternate, alternate_targets)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual_loss, reference_loss, rtol=1e-5, atol=1e-3)
    errors = {}
    for name, parameter in model.named_parameters():
        actual = parameter.grad.float()
        expected = reference_gradients[name]
        error = float((actual.double() - expected.double()).norm()
                      / expected.double().norm().clamp_min(1e-12))
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all() or not math.isfinite(error) or error > 0.01:
            raise AssertionError(f"CUDA graph changed accumulated gradient {name}: {error}")
        errors[name] = error
    del reference_gradients
    # Optimizers update parameters in place. Captured casts must read the new
    # weights rather than replaying a stale autocast-cache value.
    with torch.no_grad():
        model.proj.weight.add_(torch.randn_like(model.proj.weight) * 0.001)
        model.writer_candidate.weight.mul_(0.95)
    clear()
    reference_loss = ordinary().clone() + ordinary(alternate, alternate_targets).clone()
    reference_gradients = {name: parameter.grad.detach().float().clone()
                           for name, parameter in model.named_parameters()}
    graph.zero_grad()
    actual_loss = graph.replay(inputs, targets).clone()
    actual_loss += graph.replay(alternate, alternate_targets)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual_loss, reference_loss, rtol=1e-5, atol=1e-3)
    for name, parameter in model.named_parameters():
        actual, expected = parameter.grad.float(), reference_gradients[name]
        error = float((actual.double() - expected.double()).norm()
                      / expected.double().norm().clamp_min(1e-12))
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all() or not math.isfinite(error) or error > 0.01:
            raise AssertionError(f"CUDA graph stale weight/gradient after update {name}: {error}")
        errors[name] = max(errors[name], error)
    del reference_gradients
    torch.cuda.reset_peak_memory_stats()
    captured = measure(lambda: graph.replay(inputs, targets))
    captured["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    model.eval()
    evaluation = CUDAGraphValidation(loss_fn, batch_size=args.microbatch, seq_len=1024)
    for x, y in ((inputs, targets), (alternate, alternate_targets)):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = loss_fn(x, y, diagnostics=True)
        actual = evaluation.replay(x, y)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-4)
    with torch.no_grad():
        model.proj.weight.mul_(0.95)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = loss_fn(inputs, targets, diagnostics=True)
    for left, right in zip(evaluation.replay(inputs, targets), expected):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-4)
    eval_timing = measure(lambda: evaluation.replay(inputs, targets))
    model.train()
    graph.replay(inputs, targets)  # validation must preserve training storage
    report = {
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "microbatch": args.microbatch, "seq_len": 1024, "segment_size": 16,
        "optimizer_updates": 0, "baseline": baseline, "cuda_graph": captured,
        "cuda_graph_eval": eval_timing,
        "profile": profile, "capture_seconds": graph.capture_seconds,
        "preparation_seconds": graph.preparation_seconds,
        "speedup": baseline["median_ms"] / captured["median_ms"],
        "projected_training_update_ms": captured["median_ms"] * 512 / args.microbatch,
        "max_accumulated_gradient_relative_error": max(errors.values()),
        "accumulated_gradient_errors": errors,
    }
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
