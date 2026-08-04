#!/usr/bin/env python3
"""Benchmark batched Qwen3 answer embedding across sequence lengths."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModel

from probe_qwen3_answer_embeddings import disable_optional_vision_imports, last_token_pool


def resolved_batch_size(
    sequence_length: int, target_batch_tokens: int, max_batch_size: int
) -> int:
    if sequence_length < 1 or target_batch_tokens < 1 or max_batch_size < 1:
        raise ValueError(
            "length, target batch tokens, and maximum batch size must be positive"
        )
    # A single long answer necessarily exceeds a smaller batching target.
    return max(1, min(max_batch_size, target_batch_tokens // sequence_length))


def benchmark_case(
    model: torch.nn.Module,
    *,
    sequence_length: int,
    batch_size: int,
    warmup_repetitions: int,
    repetitions: int,
    device: torch.device,
) -> dict[str, float | int]:
    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (batch_size, sequence_length),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)

    def embed() -> None:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_token_pool(outputs.last_hidden_state, attention_mask)

    with torch.inference_mode():
        for _ in range(warmup_repetitions):
            embed()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        durations_ms = []
        for _ in range(repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            embed()
            end.record()
            end.synchronize()
            durations_ms.append(start.elapsed_time(end))
        peak_memory = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

    median_ms = statistics.median(durations_ms)
    return {
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "batch_tokens": sequence_length * batch_size,
        "median_ms": median_ms,
        "mean_ms": statistics.mean(durations_ms),
        "min_ms": min(durations_ms),
        "max_ms": max(durations_ms),
        "sequences_per_second": 1_000.0 * batch_size / median_ms,
        "tokens_per_second": 1_000.0 * sequence_length * batch_size / median_ms,
        "incremental_peak_torch_allocated_gib": (peak_memory - baseline_memory)
        / 2**30,
        "peak_torch_allocated_gib": peak_memory / 2**30,
        "incremental_peak_torch_reserved_gib": (peak_reserved - baseline_reserved)
        / 2**30,
        "peak_torch_reserved_gib": peak_reserved / 2**30,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[128, 512, 1024, 2048, 4096, 8192, 16384],
    )
    parser.add_argument("--target-batch-tokens", type=int, default=8192)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument(
        "--attention-backend", choices=("sdpa", "eager"), default="sdpa"
    )
    parser.add_argument("--warmup-repetitions", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.warmup_repetitions < 1 or args.repetitions < 1:
        raise ValueError("warmup and measured repetition counts must be positive")
    if any(length < 1 for length in args.lengths):
        raise ValueError("sequence lengths must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Qwen embedding benchmarking requires CUDA")

    disable_optional_vision_imports()
    load_started = time.perf_counter()
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation=args.attention_backend,
        local_files_only=True,
    ).eval()
    load_seconds = time.perf_counter() - load_started
    maximum_length = int(model.config.max_position_embeddings)
    unsupported = [length for length in args.lengths if length > maximum_length]
    if unsupported:
        raise ValueError(
            f"requested lengths exceed model maximum {maximum_length}: {unsupported}"
        )

    measurements = []
    for sequence_length in args.lengths:
        batch_size = resolved_batch_size(
            sequence_length, args.target_batch_tokens, args.max_batch_size
        )
        try:
            measurement = benchmark_case(
                model,
                sequence_length=sequence_length,
                batch_size=batch_size,
                warmup_repetitions=args.warmup_repetitions,
                repetitions=args.repetitions,
                device=device,
            )
        except torch.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            measurement = {
                "sequence_length": sequence_length,
                "batch_size": batch_size,
                "batch_tokens": sequence_length * batch_size,
                "error": f"{type(error).__name__}: {error}",
            }
        measurements.append(measurement)
        print(json.dumps(measurement), flush=True)

    result = {
        "schema": "qwen3_answer_embedding_benchmark/v1",
        "model": args.model,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "dtype": str(next(model.parameters()).dtype),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "maximum_model_tokens": maximum_length,
        "attention_backend": args.attention_backend,
        "load_seconds": load_seconds,
        "benchmark_scope": "model forward plus last-token pooling; target pre-encoding excluded",
        "warmup_repetitions": args.warmup_repetitions,
        "repetitions": args.repetitions,
        "target_batch_tokens": args.target_batch_tokens,
        "max_batch_size": args.max_batch_size,
        "measurements": measurements,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
