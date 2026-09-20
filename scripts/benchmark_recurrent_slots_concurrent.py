"""Benchmark concurrent B64 graphs without changing execution microbatch math."""
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
from pretraining.nanogpt_mini.recurrent_slots_runtime import ConcurrentMicrobatches, RecurrentLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.repeats < 3:
        raise ValueError("CUDA and at least three timing repetitions are required")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1337)
    model = RecurrentSlots().cuda().train()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("proj.weight"):
                parameter.normal_(std=0.003)
    loss_fn = RecurrentLoss(model)
    inputs = torch.randint(1024, (512, 1024), dtype=torch.int32, device="cuda")
    targets = inputs.roll(-1, 1).long()

    def ordinary():
        model.zero_grad(set_to_none=False)
        total = torch.zeros((), device="cuda")
        for start in range(0, 512, 64):
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                loss = loss_fn(inputs[start:start + 64], targets[start:start + 64])
                loss.backward()
            total += loss.detach()
        return total

    def timings(function):
        values = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            function()
            torch.cuda.synchronize()
            values.append((time.perf_counter() - start) * 1000)
        return {"samples_ms": values, "median_ms": statistics.median(values)}

    print("warming reference 8 x B64", flush=True)
    ordinary()
    baseline = timings(ordinary)
    reference_loss = ordinary().clone()
    reference = {name: parameter.grad.detach().float().clone()
                 for name, parameter in model.named_parameters()}
    print(json.dumps({"baseline": baseline, "capturing_concurrency": args.concurrency}), flush=True)
    torch.cuda.reset_peak_memory_stats()
    free_before, _ = torch.cuda.mem_get_info()
    prepare_start = time.perf_counter()
    concurrent = ConcurrentMicrobatches(model, microbatch=64, rows=512,
                                        segment_size=16, concurrency=args.concurrency)
    torch.cuda.synchronize()
    preparation_seconds = time.perf_counter() - prepare_start
    free_after, _ = torch.cuda.mem_get_info()
    footprint = {"peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                 "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                 "device_free_delta_mib": (free_before - free_after) / 2**20}
    errors = {}

    def verify(expected_loss, expected):
        actual_loss = concurrent.replay(inputs, targets)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-3)
        for name, parameter in model.named_parameters():
            a, b = parameter.grad.double().flatten(), expected[name].double().flatten()
            error = float((a - b).norm() / b.norm().clamp_min(1e-30))
            tolerance = 0.05 if name == "embed.weight" else 0.005
            if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not math.isfinite(error) or error > tolerance:
                raise AssertionError(f"Concurrent gradient changed: {name} relative error {error}")
            errors[name] = max(errors.get(name, 0), error)

    verify(reference_loss, reference)
    del reference
    # Shared replicas must observe optimizer-style in-place parameter changes.
    with torch.no_grad():
        model.proj.weight.add_(torch.randn_like(model.proj.weight) * 0.001)
        model.writer_candidate.weight.mul_(0.95)
    reference_loss = ordinary().clone()
    reference = {name: parameter.grad.detach().float().clone()
                 for name, parameter in model.named_parameters()}
    verify(reference_loss, reference)
    configurations = {}
    for active in (1, 2, 4, 8):
        if active > args.concurrency:
            continue
        concurrent.concurrency = active
        configurations[str(active)] = timings(lambda: concurrent.replay(inputs, targets))
        verify(reference_loss, reference)
        print(json.dumps({"active_streams": active, "timing": configurations[str(active)]}), flush=True)
    del reference
    best = min(configurations, key=lambda key: configurations[key]["median_ms"])
    candidate = configurations[best]
    report = {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
              "optimizer_updates": 0, "microbatch": 64, "rows": 512, "seq_len": 1024,
              "concurrency": args.concurrency, "baseline": baseline, "concurrent": candidate,
              "best_concurrency": int(best), "configurations": configurations,
              "speedup": baseline["median_ms"] / candidate["median_ms"],
              "preparation_seconds": preparation_seconds, "memory": footprint,
              "gradient_relative_errors": errors,
              "max_gradient_relative_error": max(errors.values())}
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
