#!/usr/bin/env python3
"""Benchmark a fixed byte-diffusion forward/backward/AdamW update.

Benchmarks are GPU workloads and must be queued:

    mlq submit --name bd_benchmark --cwd "$PWD" --max-parallel-runs 1 -- \
      python3 scripts/benchmark_byte_diffusion.py --length 128 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import torch
from torch._dynamo.utils import counters as dynamo_counters

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.config import ByteDiffusionConfig, ModelMode
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.training import (
    JointForward,
    TrainingRunConfig,
    attention_context,
    create_optimizer,
    model_config_from_dict,
)


def sample_power(stop: threading.Event, samples: list[float]) -> None:
    """Collect non-blocking board-power evidence while the benchmark runs."""

    while not stop.wait(0.1):
        observed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if observed.returncode == 0:
            try:
                samples.append(float(observed.stdout.strip().splitlines()[0]))
            except (ValueError, IndexError):
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--canvas-length", type=int, default=128)
    parser.add_argument("--branches", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=("ar", "canvas", "joint", "blt", "blt_joint"),
        default="joint",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument(
        "--static-shapes",
        action="store_true",
        help="Specialize compiled graphs to fixed shapes (not valid for real packed batches).",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs"),
        default="default",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA and must run through mlq")
    if min(args.batch_size, args.length, args.iterations) <= 0 or args.length % 4:
        raise ValueError("batch, iterations, and patch-aligned length must be positive")
    device = torch.device("cuda", 0)
    payload = (
        torch.load(args.checkpoint, map_location=device, weights_only=False)
        if args.checkpoint
        else None
    )
    config = (
        model_config_from_dict(payload["model_config"])
        if payload is not None
        else ByteDiffusionConfig()
    )
    model = ByteDiffusionModel(config).to(device)
    if payload is not None:
        model.load_state_dict(payload["model"], strict=True)
    run = TrainingRunConfig(iterations=args.iterations, warmdown_iters=0)
    optimizer = create_optimizer(model.parameters(), run, device)
    ids = torch.randint(
        config.vocab.output_size,
        (args.batch_size, args.length),
        device=device,
    )
    valid = torch.ones_like(ids, dtype=torch.bool)
    positions = torch.arange(args.length, device=device)[None].expand_as(ids)
    if args.canvas_length <= 0 or args.canvas_length % 4:
        raise ValueError("canvas length must be a positive patch multiple")
    if args.branches <= 0 or args.branches * args.canvas_length > args.length:
        raise ValueError("nonoverlapping branches must fit in the clean row")
    starts = torch.arange(args.branches, device=device)[None].expand(
        args.batch_size, -1
    ) * args.canvas_length
    offsets = torch.arange(args.canvas_length, device=device)
    indices = starts[:, :, None] + offsets[None, None, :]
    noisy = torch.gather(
        ids[:, None, :].expand(-1, args.branches, -1), 2, indices
    )
    noisy[:, :, ::2] = config.vocab.mask_id
    noisy_valid = torch.ones_like(noisy, dtype=torch.bool)
    forward_model: torch.nn.Module = JointForward(model)
    dynamo_counters.clear()
    if not args.eager:
        forward_model = torch.compile(
            forward_model,
            fullgraph=False,
            dynamic=not args.static_shapes,
            mode=args.compile_mode,
        )

    def iteration() -> None:
        optimizer.zero_grad(set_to_none=True)
        with attention_context(
            "flash_sdpa", device, allow_cpu_reference=False
        ), torch.autocast("cuda", dtype=torch.bfloat16):
            if args.mode == "ar":
                clean_logits, _, _ = forward_model(
                    ids, valid, positions, None, -1, None, None, 0, True
                )
                loss = clean_logits.float().square().mean()
            else:
                diffusion_mode = (
                    int(ModelMode.BLT_D)
                    if args.mode in {"blt", "blt_joint"}
                    else int(ModelMode.CANVAS)
                )
                clean_logits, branch_logits, _ = forward_model(
                    ids,
                    valid,
                    positions,
                    noisy,
                    diffusion_mode,
                    noisy_valid,
                    starts,
                    args.canvas_length,
                    True,
                )
                assert branch_logits is not None
                loss = branch_logits.float().square().mean()
                if args.mode in {"joint", "blt_joint"}:
                    loss = loss + clean_logits.float().square().mean()
        loss.backward()
        optimizer.step()
        model.enforce_padding_invariant()

    for _ in range(args.warmup):
        iteration()
    graph_breaks = sum(dynamo_counters["graph_break"].values())
    unique_graphs = int(dynamo_counters["stats"]["unique_graphs"])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    power_samples: list[float] = []
    stop_power = threading.Event()
    power_thread = threading.Thread(
        target=sample_power, args=(stop_power, power_samples), daemon=True
    )
    power_thread.start()
    started = time.perf_counter()
    for _ in range(args.iterations):
        iteration()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    stop_power.set()
    power_thread.join(timeout=1.0)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    result = {
        "batch_size": args.batch_size,
        "length": args.length,
        "canvas_length": args.canvas_length,
        "branches": args.branches,
        "compiled": not args.eager,
        "compile_mode": args.compile_mode if not args.eager else None,
        "dynamic_shapes": not args.static_shapes if not args.eager else None,
        "mode": args.mode,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "step_ms": 1_000 * elapsed / args.iterations,
        "positions_per_second": (
            args.batch_size * args.length * args.iterations / elapsed
        ),
        "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "average_power_w": (
            sum(power_samples) / len(power_samples) if power_samples else None
        ),
        "peak_power_w": max(power_samples) if power_samples else None,
        "power_samples": len(power_samples),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "git_revision": revision or None,
        "working_tree_dirty": dirty,
        "compile_graph_breaks": graph_breaks,
        "compile_unique_graphs": unique_graphs,
    }
    encoded = json.dumps(result, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n")
    print("benchmark_byte_diffusion " + encoded, flush=True)


if __name__ == "__main__":
    main()
