"""Microbatch and kernel-time scan for the shared-pool GDN2 model.

Answers, before the pool's training run is queued, three questions that
decide its execution: how much the private model loses per update at
microbatch 32 and 16 against the production 64; what the pool model costs at
the (training, validation) microbatch pairs whose CUDA graphs fit next to
each other on the 32 GiB device; and how the pool update splits between the
private chunk kernels, the pool chunk kernel and everything else. Timings
use random tokens: the pool's static-shape packed dispatch makes its cost
independent of the routing decisions.

    mlq submit --name gdn2_pool_microbatch_scan --cwd "$PWD" --max-parallel-runs 1 \
        --max-attempts 1 --time-limit 60m --env FLA_CACHE_RESULTS=0 -- \
        .venv/bin/python scripts/scan_gdn2_pool_microbatch.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from pretraining.nanogpt_mini import gated_delta_ops  # noqa: E402
from pretraining.nanogpt_mini.gated_delta_model import GatedDeltaGPT  # noqa: E402
from pretraining.nanogpt_mini.gated_delta_pool import SharedPoolGatedDeltaGPT, pool_layout  # noqa: E402
from pretraining.nanogpt_mini.gated_delta_runtime import build_gated_delta_optimizers, compiled_gated_delta_loss  # noqa: E402
from pretraining.nanogpt_mini.recurrent_slots_runtime import CUDAGraphMicrobatch, CUDAGraphValidation  # noqa: E402
from scripts.train_recurrent_slots import atomic_json  # noqa: E402

PRODUCTION = dict(gdn_backend="fla", disable_recompute=True, custom_ops=True, fused_projections=True)
UPDATE_ROWS, SEQ_LEN, HEADS, HEAD_DIM, LAYERS = 512, 1024, 4, 128, 6
CONFIGS = (
    dict(model="private", microbatch=64, validation_microbatch=64),
    dict(model="private", microbatch=32, validation_microbatch=32),
    dict(model="private", microbatch=16, validation_microbatch=16),
    dict(model="shared_pool", microbatch=16, validation_microbatch=16),
    dict(model="shared_pool", microbatch=32, validation_microbatch=16),
    dict(model="shared_pool", microbatch=32, validation_microbatch=32),
)


def build(kind: str):
    torch.manual_seed(1337)
    if kind == "shared_pool":
        model = SharedPoolGatedDeltaGPT(**PRODUCTION)
        with torch.no_grad():  # a nonzero readout so pass two's pool path is exercised in full
            for block in model.blocks:
                block.pool_o_proj.weight.normal_(std=.003)
        return model.cuda()
    return GatedDeltaGPT(**PRODUCTION).cuda()


def release():
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()


def measure_update(config: dict, tokens, targets, repeats: int, profile: bool) -> dict:
    microbatch, validation_microbatch = config["microbatch"], config["validation_microbatch"]
    result = dict(config)
    model = build(config["model"]).train()
    loss = compiled_gated_delta_loss(model, 64)
    optimizers = build_gated_delta_optimizers(model)
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    validation = CUDAGraphValidation(loss, batch_size=validation_microbatch, seq_len=SEQ_LEN)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        validation.replay(tokens[:validation_microbatch], targets[:validation_microbatch])
    torch.cuda.synchronize()
    result["validation_graph_reserved_mib"] = torch.cuda.memory_reserved() / 2 ** 20
    model.train()
    graph = CUDAGraphMicrobatch(loss, batch_size=microbatch, seq_len=SEQ_LEN)
    loss.audit_graph_breaks()

    def update():
        graph.zero_grad()
        total = torch.zeros((), device="cuda")
        for row in range(0, UPDATE_ROWS, microbatch):
            total += graph.replay(tokens[row:row + microbatch], targets[row:row + microbatch])
        for optimizer in optimizers:
            optimizer.step()
        return total

    for _ in range(3):
        update()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        update()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    median = statistics.median(samples)
    result.update(median_update_seconds=median, update_seconds=samples, tokens_per_second=UPDATE_ROWS * SEQ_LEN / median,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20,
                  capture_seconds=graph.capture_seconds, preparation_seconds=graph.preparation_seconds)
    if profile:
        from torch.profiler import ProfilerActivity, profile as torch_profile
        with torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
            update()
            torch.cuda.synchronize()
        by_name = defaultdict(float)
        for event in prof.events():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                by_name[event.name] += event.device_time_total if hasattr(event, "device_time_total") else event.cuda_time_total
        total = sum(by_name.values())
        top = sorted(by_name.items(), key=lambda item: -item[1])[:25]
        result["kernel_time_seconds"] = total / 1e6
        result["top_kernels"] = [dict(name=name[:120], seconds=micro / 1e6, fraction=micro / total) for name, micro in top]
    del graph, validation, loss, optimizers, model
    return result


def time_kernel(shape: tuple[int, int], repeats: int = 10, packed: dict | None = None) -> float:
    """Forward plus backward seconds of one training chunk kernel call at ``shape`` = (batch, time).

    ``packed`` supplies ``cu_seqlens`` and ``chunk_indices`` for the pool's
    packed (variable-length) call over one row.
    """
    batch, length = shape
    make = lambda width, dtype: torch.randn(batch, length, HEADS, width, device="cuda", dtype=dtype, requires_grad=True)
    q, k, v = make(HEAD_DIM, torch.bfloat16), make(HEAD_DIM, torch.bfloat16), make(HEAD_DIM, torch.bfloat16)
    g = (-torch.rand(batch, length, HEADS, HEAD_DIM, device="cuda") * .1).requires_grad_()
    b, w = (torch.rand(batch, length, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16).requires_grad_() for _ in range(2))
    def call():
        gated_delta_ops.chunk_gdn2_training(q, k, v, g, b, w, state_v_first=PRODUCTION.get("state_v_first", False),
                                            **(packed or {})).sum().backward()
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / repeats


def kernel_decomposition() -> dict:
    """Chunk-kernel seconds per 524,288-token update for the private and pool shapes."""
    out = {}
    for batch in (64, 32, 16):
        per_call = time_kernel((batch, SEQ_LEN))
        out[f"private_b{batch}"] = dict(per_call_seconds=per_call,
                                        per_update_one_pass_seconds=per_call * (UPDATE_ROWS // batch) * LAYERS)
    for batch in (32, 16):
        # The pool packs batch x 6 x 1024 events bank by bank into batch x 108 chunks; a random
        # route stands in for the router (the layout's cost is independent of the routing).
        route = torch.randint(12, (batch, LAYERS * SEQ_LEN), device="cuda")
        layout = pool_layout(route, 12, 64)
        per_call = time_kernel((1, layout.rows), packed=dict(cu_seqlens=layout.cu_seqlens,
                                                             chunk_indices=layout.chunk_indices))
        out[f"pool_b{batch}"] = dict(per_call_seconds=per_call, per_update_seconds=per_call * (UPDATE_ROWS // batch),
                                     rows=layout.rows, chunks_per_row=layout.chunks)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "ablation_results/gdn2_pool_microbatch_scan")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(7)
    tokens = torch.randint(1024, (UPDATE_ROWS, SEQ_LEN), device="cuda", dtype=torch.int32)
    targets = torch.randint(1024, tokens.shape, device="cuda")
    report = dict(status="running", torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                  kernels=kernel_decomposition(), updates=[])
    print(json.dumps(dict(kernels=report["kernels"])), flush=True)
    atomic_json(args.output / "scan.json", report)
    for config in CONFIGS:
        release()
        try:
            result = measure_update(config, tokens, targets, args.repeats, profile=config["model"] == "shared_pool")
        except torch.OutOfMemoryError as error:
            result = dict(config, error=str(error).splitlines()[0][:300])
        release()
        report["updates"].append(result)
        summary = {k: v for k, v in result.items() if k not in ("top_kernels", "update_seconds")}
        print(json.dumps(summary), flush=True)
        for kernel in result.get("top_kernels", [])[:12]:
            print(f"  {kernel['fraction'] * 100:5.1f}%  {kernel['seconds'] * 1000:7.1f} ms  {kernel['name'][:90]}", flush=True)
        atomic_json(args.output / "scan.json", report)
    report["status"] = "completed"
    atomic_json(args.output / "scan.json", report)


if __name__ == "__main__":
    main()
