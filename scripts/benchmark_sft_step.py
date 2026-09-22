"""Throughput of the production SFT step on real packed corpus rows.

Drives ``postraining.sft_trace_train``'s own ``SupervisedCE``,
``accumulate_step_gradients`` and optimizers -- the benchmark cannot drift
from the trainer because it has no step implementation of its own. Rows come
from the head of a real corpus, tokenized and packed exactly as the trainer
does, so the supervised fraction and document mix are the run's.

Each ``--config`` is ``micro_rows:compile[:fence]`` (e.g. ``8:1``) and runs
on a freshly loaded backbone after ``torch._dynamo`` reset.
``fence=forward`` reproduces the historical whole-``forward``
``torch.compiler.disable`` on the KDA mixer, which left its projections,
convolutions and gated norm eager; the default ``kernel`` fences only the FLA
recurrence, as the model now does. Warmup steps are discarded: the first
steps carry compile and autotuning cost and are not throughput.

GPU workload -- submit through mlq:

    mlq submit --name bench_sft_step --cwd "$PWD" --max-parallel-runs 1 -- \\
        .venv/bin/python -u -m scripts.benchmark_sft_step \\
        --config 2:0 --config 2:1:forward --config 8:1
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

import train_gpt as baseline  # noqa: F401  (import order: patches must load first)
from postraining.core import load_posttraining_tokenizer
from postraining.model_io import load_model
from postraining.prepare_sft_traces import INSTRUCTION_SUFFIX_ANSWER
from postraining.sft_trace_train import (
    IGNORE_INDEX,
    SupervisedCE,
    accumulate_step_gradients,
    build_optimizers,
    pack_rows,
    register_special_tokens,
    tokenize_documents,
)
from pretraining.fresh_lejepa.fresh_lejepa_train import FreshHyperparameters
from pretraining.nanogpt_mini import nanogpt_mini_kda_model as kda_model


def parse_config(text: str) -> tuple[int, bool, str]:
    micro, compiled, *fence = text.split(":")
    fence = fence[0] if fence else "kernel"
    if fence not in {"kernel", "forward"}:
        raise argparse.ArgumentTypeError(f"unknown fence {fence!r}")
    return int(micro), compiled == "1", fence


def load_rows(args, backbone) -> tuple[list, object]:
    tokenizer = load_posttraining_tokenizer(
        backbone.architecture,
        FreshHyperparameters.tokenizer_path,
        think_tokens=True,
        answer_tokens=True,
        tokenizer_provenance=backbone.model_config.get("tokenizer_provenance"),
    )
    table = pq.read_table(args.traces).slice(0, args.documents)
    documents = table.to_pylist()
    started = time.perf_counter()
    tokenized = tokenize_documents(
        tokenizer, documents, args.seq_len,
        instruction_suffix=INSTRUCTION_SUFFIX_ANSWER,
    )
    tokenize_seconds = time.perf_counter() - started
    rows = pack_rows(
        tokenized, args.seq_len, tokenizer.eos_id(), random.Random(1234)
    )
    print(
        f"{len(documents)} documents -> {len(tokenized)} fit -> "
        f"{len(rows)} rows; tokenize {tokenize_seconds:.1f}s "
        f"({len(documents) / tokenize_seconds:,.0f} docs/s)",
        flush=True,
    )
    return rows, tokenizer


def run_config(args, rows, micro: int, compiled: bool, fence: str):
    mixer = kda_model.KimiDeltaAttention
    kernel_fenced_forward = mixer.forward
    if fence == "forward":
        mixer.forward = torch.compiler.disable(kernel_fenced_forward)
    try:
        return _run_config(args, rows, micro, compiled, fence)
    finally:
        mixer.forward = kernel_fenced_forward


def _run_config(args, rows, micro: int, compiled: bool, fence: str):
    # The previous config's backbone and optimizer state die with its frame;
    # collect them before this config's peak-memory reading.
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    backbone = load_model(args.checkpoint, device)
    for parameter in backbone.parameters():
        parameter.requires_grad_(True)
    tokenizer = load_posttraining_tokenizer(
        backbone.architecture,
        FreshHyperparameters.tokenizer_path,
        think_tokens=True,
        answer_tokens=True,
        tokenizer_provenance=backbone.model_config.get("tokenizer_provenance"),
    )
    register_special_tokens(backbone, tokenizer)
    optimizers = build_optimizers(backbone, 0.1)
    loss_fn = SupervisedCE(backbone, compiled=compiled)
    backbone.train()

    needed = (args.warmup + args.steps) * args.rows_per_step
    if len(rows) < needed:
        raise ValueError(f"need {needed} rows, have {len(rows)}")

    def step(index: int):
        step_rows = rows[
            index * args.rows_per_step:(index + 1) * args.rows_per_step
        ]
        loss = accumulate_step_gradients(loss_fn, step_rows, micro, device)
        for optimizer in optimizers:
            optimizer.step()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        return loss

    warm_started = time.perf_counter()
    for index in range(args.warmup):
        step(index)
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warm_started
    torch.cuda.reset_peak_memory_stats()

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    wall = time.perf_counter()
    start.record()
    for index in range(args.warmup, args.warmup + args.steps):
        loss = step(index)
    end.record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - wall
    seconds = start.elapsed_time(end) / 1000
    timed = rows[
        args.warmup * args.rows_per_step:
        (args.warmup + args.steps) * args.rows_per_step
    ]
    supervised = sum(
        int(np.count_nonzero(targets != IGNORE_INDEX)) for _, targets in timed
    )
    result = {
        "micro_rows": micro,
        "grad_accum": args.rows_per_step // micro,
        "compiled": compiled,
        "kda_fence": fence,
        "ms_per_step": 1000 * seconds / args.steps,
        "tokens_per_second": len(timed) * args.seq_len / seconds,
        "supervised_tokens_per_second": supervised / seconds,
        "host_wall_over_device": wall / seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "warmup_seconds": warmup_seconds,
        "final_loss": float(loss),
    }
    print(json.dumps(result), flush=True)

    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for index in range(args.warmup, args.warmup + 3):
                step(index)
            torch.cuda.synchronize()
        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=args.profile_rows
            ),
            flush=True,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="logs/nanogpt_gpt2_kda8_kkkdkkkd_triton_mbs32_optimized_2k_"
        "final_model.pt",
    )
    parser.add_argument(
        "--traces",
        default="postraining/data/sft_mix_ud2605_omi2_drills_v2.parquet",
    )
    parser.add_argument("--documents", type=int, default=6000)
    parser.add_argument("--seq-len", type=int, default=5120)
    parser.add_argument("--rows-per-step", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument(
        "--config", action="append", required=True, type=parse_config
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-rows", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for micro, *_ in args.config:
        if args.rows_per_step % micro:
            parser.error("micro_rows must divide --rows-per-step")

    probe = load_model(args.checkpoint, torch.device("cuda"))
    rows, _ = load_rows(args, probe)
    del probe
    torch.cuda.empty_cache()
    results = [run_config(args, rows, *config) for config in args.config]
    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
