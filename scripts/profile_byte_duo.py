#!/usr/bin/env python3
"""Queue-only kernel profile for one production-geometry Byte-Duo update."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pretraining.byte_diffusion.pipeline import DeviceBatchPrefetcher
from pretraining.byte_diffusion.training import load_data_directory
from scripts.benchmark_byte_diffusion_architectures import DuoUpdateHarness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", type=Path, default=Path("data/byte_diffusion_aligned_v5")
    )
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--global-batch", type=int, default=249)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1_337)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--record-shapes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Byte-Duo profiling must be submitted through mlq")
    if min(
        args.microbatch,
        args.branches,
        args.global_batch,
        args.validation_batch_size,
        args.warmup_updates,
    ) <= 0:
        raise ValueError("profile geometry must be positive")

    device = torch.device("cuda", 0)
    torch.set_float32_matmul_precision("high")
    manifest, train_dataset, validation_dataset = load_data_directory(
        args.data_path,
        chunk_size=8_192,
        recipe="blt_d",
        required_branch_bytes=512 * args.branches,
        branch_span_length=512,
        validation_chunk_limit=256,
    )
    with DeviceBatchPrefetcher(device=device) as prefetcher:
        harness = DuoUpdateHarness(
            train_dataset,
            validation_dataset,
            manifest=manifest,
            device=device,
            microbatch=args.microbatch,
            validation_batch_size=args.validation_batch_size,
            global_batch=args.global_batch,
            branches=args.branches,
            seed=args.seed,
            batch_prefetcher=prefetcher,
        )
        updates = harness.prepared_updates(args.warmup_updates + 1)
        for _ in range(args.warmup_updates):
            harness.run_update(next(updates))
        torch.cuda.synchronize(device)
        with torch.profiler.profile(
            activities=(
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ),
            record_shapes=args.record_shapes,
            profile_memory=False,
            with_stack=False,
        ) as profile:
            with torch.profiler.record_function("byte_duo_optimizer_update"):
                harness.run_update(next(updates))
            torch.cuda.synchronize(device)

    table = profile.key_averages(
        group_by_input_shape=args.record_shapes
    ).table(
        sort_by="self_cuda_time_total",
        row_limit=100,
    )
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(table + "\n")
    print(table, flush=True)
    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(args.trace))


if __name__ == "__main__":
    main()
