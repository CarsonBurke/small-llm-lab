"""Queued microbenchmark: per-layer versus per-step split-KV partition plans.

Replays 24 decode-shaped FA4 split-KV launches (production geometry: 64 lanes,
11264-token cache) inside CUDA graphs with the plan derived inside every
launch versus one shared plan computed once per step, as the rollout engine
now does. Checks bitwise equality across retirement and refill length
patterns and reports graph replay time and kernel counts per step.
"""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "ablation_results/minicpm_split_kv_plan_20260918"
OUT.mkdir(parents=True, exist_ok=True)
import torch

from postraining.invariant_attention import invariant_fa4
from postraining.split_kv_plan import plan_split_kv, split_kv_metadata, split_kv_plan_shapes

LAYERS = 24
BATCH = 64
CAPACITY = 11264
SCALE = 128**-0.5


def main():
    device = torch.device("cuda")
    torch.manual_seed(5)
    keys = torch.randn(BATCH, CAPACITY, 2, 128, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    queries = torch.randn(LAYERS, BATCH, 1, 16, 128, device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        lengths = torch.full((BATCH,), 1, device=device, dtype=torch.int32)
        metadata = split_kv_metadata(BATCH, CAPACITY, device)
        offsets_size, live_size = split_kv_plan_shapes(BATCH)
        offsets = torch.zeros(offsets_size, device=device, dtype=torch.int32)
        live = torch.zeros(live_size, device=device, dtype=torch.int32)
    compiled_plan = torch.compile(plan_split_kv, fullgraph=True)

    def per_layer_step():
        return [
            invariant_fa4(queries[layer], keys, values, lengths, 1, CAPACITY, SCALE, True, None, None)
            for layer in range(LAYERS)
        ]

    def shared_step():
        step_offsets, step_live = compiled_plan(lengths, *metadata)
        offsets.copy_(step_offsets)
        live.copy_(step_live)
        return [
            invariant_fa4(queries[layer], keys, values, lengths, 1, CAPACITY, SCALE, True, offsets, live)
            for layer in range(LAYERS)
        ]

    patterns = {
        "full": [CAPACITY] * BATCH,
        "mixed": [(CAPACITY * (row + 1)) // BATCH for row in range(BATCH)],
        "retired_half": [CAPACITY if row % 2 else 1 for row in range(BATCH)],
        "short": [17 + row for row in range(BATCH)],
    }
    results = {}
    with torch.inference_mode():
        lengths.copy_(torch.tensor(patterns["mixed"], device=device, dtype=torch.int32))
        for _ in range(3):
            per_layer_step()
            shared_step()
        torch.cuda.synchronize()
        graphs = {}
        outputs = {}
        for name, step in (("per_layer", per_layer_step), ("shared", shared_step)):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs[name] = step()
            graphs[name] = graph
        for pattern, pattern_lengths in patterns.items():
            lengths.copy_(torch.tensor(pattern_lengths, device=device, dtype=torch.int32))
            for graph in graphs.values():
                graph.replay()
            torch.cuda.synchronize()
            equal = all(
                torch.equal(a, b) for a, b in zip(outputs["per_layer"], outputs["shared"], strict=True)
            )
            finite = all(torch.isfinite(output).all().item() for output in outputs["shared"])
            timings = {}
            for name, graph in graphs.items():
                for _ in range(5):
                    graph.replay()
                torch.cuda.synchronize()
                started = torch.cuda.Event(enable_timing=True)
                stopped = torch.cuda.Event(enable_timing=True)
                started.record()
                for _ in range(50):
                    graph.replay()
                stopped.record()
                torch.cuda.synchronize()
                timings[name] = started.elapsed_time(stopped) / 50 / 1000
            results[pattern] = {"bitwise_equal": equal, "finite": finite, "step_seconds": timings}
            print(json.dumps({"pattern": pattern, **results[pattern]}), flush=True)
        lengths.copy_(torch.tensor(patterns["mixed"], device=device, dtype=torch.int32))
        kernel_counts = {}
        from torch.profiler import ProfilerActivity, profile

        for name, step in (("per_layer", per_layer_step), ("shared", shared_step)):
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as trace:
                step()
                torch.cuda.synchronize()
            kernel_counts[name] = sum(
                1 for event in trace.events() if event.device_type == torch.autograd.DeviceType.CUDA
            )
    summary = {
        "schema": "minicpm_split_kv_plan/v1",
        "geometry": {"layers": LAYERS, "batch": BATCH, "capacity": CAPACITY},
        "patterns": results,
        "kernels_per_step": kernel_counts,
        "reproduction_source": Path(__file__).read_text(),
    }
    (OUT / "result.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"check": "complete", "kernels_per_step": kernel_counts}), flush=True)


if __name__ == "__main__":
    main()
