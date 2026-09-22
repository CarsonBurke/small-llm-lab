"""Kernel-cost probe for the next shared-pool designs.

Measures, at the production microbatch of 16 sequences x 1024 tokens, the
packed pool chunk-kernel call (forward plus backward) for candidate
geometries: bank width (4 heads of 128 as now, or 2 heads of 128 for a
half-width latent pool), bank count (12 routed banks or 6 single-writer
banks) and events per (token, layer) (1 as now; 2 or 3 with read-only
events for top-2 reads). It also measures ``torch._grouped_mm`` forward plus
backward at the per-bank adapter shape (one 256 x 256 matrix per bank,
applied to every packed row by segment) and checks that it captures into a
CUDA graph on this device.

    mlq submit --name gdn2_pool_variant_probe --cwd "$PWD" --max-parallel-runs 1 \
      --max-attempts 1 --time-limit 30m -- .venv/bin/python scripts/probe_gdn2_pool_variants.py \
      --output ablation_results/gdn2_pool_variant_probe
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pretraining.nanogpt_mini import gated_delta_ops  # noqa: E402
from pretraining.nanogpt_mini.gated_delta_pool import pool_layout  # noqa: E402

MICROBATCH, SEQ_LEN, LAYERS, HEAD_DIM, CHUNK = 16, 1024, 6, 128, 64
MICROBATCHES_PER_UPDATE = 512 // MICROBATCH
GEOMETRIES = {
    # name: (banks, heads, events per (token, layer))
    "current_b12_h4_e1": (12, 4, 1),
    "routed_b12_h2_e2": (12, 2, 2),      # write event doubles as top-1 read, one read-only event
    "routed_b12_h4_e2": (12, 4, 2),
    "writer_b6_h2_e2": (6, 2, 2),        # write-only event plus one read-only event (top-1 read)
    "writer_b6_h2_e3": (6, 2, 3),        # write-only event plus two read-only events (top-2 reads)
    "writer_b6_h4_e3": (6, 4, 3),
}


def timed(call, repeats: int) -> float:
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def kernel_call(banks: int, heads: int, events_per_token: int, repeats: int) -> dict:
    events = SEQ_LEN * LAYERS * events_per_token
    route = torch.randint(banks, (MICROBATCH, events), device="cuda")
    layout = pool_layout(route, banks, CHUNK)
    rows = layout.rows
    make = lambda dtype: torch.randn(1, rows, heads, HEAD_DIM, device="cuda", dtype=dtype, requires_grad=True)
    q, k, v = make(torch.bfloat16), make(torch.bfloat16), make(torch.bfloat16)
    g = (-torch.rand(1, rows, heads, HEAD_DIM, device="cuda") * .1).requires_grad_()
    b, w = make(torch.bfloat16), make(torch.bfloat16)

    def call():
        gated_delta_ops.chunk_gdn2_training(q, k, v, g, b, w, cu_seqlens=layout.cu_seqlens,
                                            chunk_indices=layout.chunk_indices, state_v_first=False).sum().backward()
    per_call = timed(call, repeats)
    return dict(banks=banks, heads=heads, events_per_token=events_per_token, rows=rows, chunks_per_row=layout.chunks,
                per_call_seconds=per_call, per_update_seconds=per_call * MICROBATCHES_PER_UPDATE)


def grouped_mm_probe(banks: int, events_per_token: int, width: int, repeats: int) -> dict:
    events = SEQ_LEN * LAYERS * events_per_token
    route = torch.randint(banks, (MICROBATCH, events), device="cuda")
    layout = pool_layout(route, banks, CHUNK)
    groups = MICROBATCH * (banks + 1)
    offs = layout.cu_seqlens[1:].contiguous()                       # group ends, chunk-aligned
    bank_of_group = torch.arange(groups, device="cuda") % (banks + 1)
    weights = torch.randn(banks + 1, width, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x = torch.randn(layout.rows, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    # A materialized upstream gradient: the grouped kernel rejects the expanded (zero-stride)
    # gradient that ``.sum().backward()`` would hand it.
    upstream = torch.randn(layout.rows, width, device="cuda", dtype=torch.bfloat16)
    result = dict(rows=layout.rows, groups=groups, width=width)

    def call():
        per_group = weights.index_select(0, bank_of_group)
        torch._grouped_mm(x, per_group, offs=offs).backward(upstream)
    try:
        result["per_call_seconds"] = timed(call, repeats)
        result["per_update_seconds"] = result["per_call_seconds"] * MICROBATCHES_PER_UPDATE
    except Exception as error:  # noqa: BLE001
        result["error"] = repr(error)
        return result
    # Dense reference: one 256 x 256 matmul over every row, for scale.
    dense = torch.randn(width, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    result["dense_single_matmul_per_call_seconds"] = timed(lambda: (x @ dense).backward(upstream), repeats)
    # Graph capture of forward + backward.
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graph.replay()
        torch.cuda.synchronize()
        result["cuda_graph_capture"] = True
    except Exception as error:  # noqa: BLE001
        result["cuda_graph_capture"] = repr(error)
    # Correctness against a per-group loop.
    with torch.no_grad():
        per_group = weights.index_select(0, bank_of_group)
        out = torch._grouped_mm(x, per_group, offs=offs)
        starts = torch.cat((offs.new_zeros(1), offs[:-1])).tolist()
        errors = []
        for group, (start, end) in enumerate(zip(starts, offs.tolist())):
            if end > start:
                ref = x[start:end] @ per_group[group]
                errors.append(((out[start:end].float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)).item())
        result["max_relative_error_vs_loop"] = max(errors)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--grouped-mm-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, microbatch=MICROBATCH,
                  kernels={}, grouped_mm={})
    for name, (banks, heads, events_per_token) in ({} if args.grouped_mm_only else GEOMETRIES).items():
        report["kernels"][name] = kernel_call(banks, heads, events_per_token, args.repeats)
        print(name, json.dumps(report["kernels"][name]), flush=True)
        torch.cuda.empty_cache()
    for name, (banks, events_per_token) in dict(routed_b12_e2=(12, 2), writer_b6_e3=(6, 3)).items():
        report["grouped_mm"][name] = grouped_mm_probe(banks, events_per_token, 256, args.repeats)
        print("grouped_mm", name, json.dumps(report["grouped_mm"][name]), flush=True)
        torch.cuda.empty_cache()
    (output / "probe.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
