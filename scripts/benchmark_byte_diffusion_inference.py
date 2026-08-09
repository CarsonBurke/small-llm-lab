#!/usr/bin/env python3
"""Benchmark full-recompute versus hierarchical-K/V-cached canvas NFE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.config import ByteDiffusionConfig
from pretraining.byte_diffusion.inference import (
    denoise_canvas_cached,
    prefill_prefix,
    prepare_cached_canvas,
)
from pretraining.byte_diffusion.kernels import flash_sdpa_only
from pretraining.byte_diffusion.model import ByteDiffusionModel
from scripts.benchmark_byte_diffusion import sample_power


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefix-length", type=int, default=4096)
    parser.add_argument("--canvas-length", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--mode", choices=("cached", "recompute"), default="cached")
    parser.add_argument("--eager", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("inference benchmark requires queued CUDA execution")
    if (
        min(args.batch_size, args.prefix_length, args.canvas_length, args.iterations) <= 0
        or args.prefix_length % 4
        or args.canvas_length % 4
    ):
        raise ValueError("batch/iterations and patch-aligned lengths must be positive")
    device = torch.device("cuda", 0)
    torch.set_float32_matmul_precision("high")
    model = ByteDiffusionModel(ByteDiffusionConfig()).eval().to(device)
    total_length = args.prefix_length + args.canvas_length
    clean = torch.randint(
        model.config.vocab.output_size,
        (args.batch_size, total_length),
        dtype=torch.long,
        device=device,
    )
    noisy = torch.full(
        (args.batch_size, 1, args.canvas_length),
        model.config.vocab.mask_id,
        dtype=torch.long,
        device=device,
    )
    valid = torch.ones_like(clean, dtype=torch.bool)
    branch_valid = torch.ones_like(noisy, dtype=torch.bool)
    starts = torch.full(
        (args.batch_size, 1),
        args.prefix_length,
        dtype=torch.long,
        device=device,
    )
    positions = torch.arange(total_length, device=device)[None].expand_as(clean)
    with flash_sdpa_only(), torch.autocast("cuda", dtype=torch.bfloat16):
        cache = prefill_prefix(model, clean)
        plan = prepare_cached_canvas(model, cache, starts, args.canvas_length)

    def forward(noisy_ids: torch.Tensor) -> torch.Tensor:
        if args.mode == "cached":
            return denoise_canvas_cached(
                model, cache, noisy_ids, starts, plan=plan
            )
        return model.forward_canvas_branches(
            clean,
            valid,
            noisy_ids,
            branch_valid,
            starts,
            positions=positions,
            assume_full_clean=True,
        ).branch_logits[:, 0]

    compiled = forward if args.eager else torch.compile(
        forward, fullgraph=False, dynamic=False, mode="default"
    )
    with flash_sdpa_only(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(args.warmup):
            compiled(noisy)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    power_samples: list[float] = []
    stop_power = threading.Event()
    power_thread = threading.Thread(
        target=sample_power, args=(stop_power, power_samples), daemon=True
    )
    power_thread.start()
    started = time.perf_counter()
    with flash_sdpa_only(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(args.iterations):
            compiled(noisy)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    stop_power.set()
    power_thread.join(timeout=1)
    print(
        "benchmark_byte_diffusion_inference "
        + json.dumps(
            {
                "mode": args.mode,
                "compiled": not args.eager,
                "batch_size": args.batch_size,
                "prefix_length": args.prefix_length,
                "canvas_length": args.canvas_length,
                "iterations": args.iterations,
                "nfe_ms": 1000 * elapsed / args.iterations,
                "canvas_bytes_per_second": (
                    args.batch_size
                    * args.canvas_length
                    * args.iterations
                    / elapsed
                ),
                "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "average_power_w": (
                    sum(power_samples) / len(power_samples) if power_samples else None
                ),
                "peak_power_w": max(power_samples) if power_samples else None,
                "power_samples": len(power_samples),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
