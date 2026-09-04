#!/usr/bin/env python3
"""Benchmark exact Qwen3.5-0.8B MTP rollouts with SGLang on one GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import time
from pathlib import Path
from typing import Any

PROMPTS = (
    "Prove that the square root of 2 is irrational, with complete reasoning.",
    "Find every integer n for which n squared plus n plus 1 is divisible by 7.",
    "A fair coin is tossed ten times. Compute the probability of exactly four heads.",
    "Solve x squared minus 7x plus 10 equals zero and justify every step.",
    "Prove by induction that the sum of the first n odd integers is n squared.",
    "How many distinct arrangements are there of the letters in MATHEMATICS?",
    "Evaluate the integral of x times exp of negative x from zero to infinity.",
    "A triangle has side lengths 13, 14, and 15. Find its area exactly.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--batch-sizes", default="1,64")
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--warmup-tokens", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17291)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--mem-fraction-static", type=float, default=0.65)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.batch_sizes = tuple(int(value) for value in args.batch_sizes.split(","))
    if not args.batch_sizes or min(args.batch_sizes) < 1:
        parser.error("--batch-sizes must contain positive integers")
    if args.output_tokens < 1 or args.warmup_tokens < 1 or args.repetitions < 1:
        parser.error("token counts and repetitions must be positive")
    return args


def build_prompt_token_ids(tokenizer: Any, batch_size: int) -> list[list[int]]:
    prompt_ids: list[list[int]] = []
    for index in range(batch_size):
        messages = [{"role": "user", "content": PROMPTS[index % len(PROMPTS)]}]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        prompt_ids.append([int(token_id) for token_id in ids])
    return prompt_ids


def make_sampling_params(
    args: argparse.Namespace, batch_size: int, output_tokens: int
) -> list[dict[str, Any]]:
    return [
        {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "ignore_eos": True,
            "min_new_tokens": output_tokens,
            "max_new_tokens": output_tokens,
        }
        for _ in range(batch_size)
    ]


def run_batch(
    engine: Any,
    args: argparse.Namespace,
    prompt_token_ids: list[list[int]],
    output_tokens: int,
) -> tuple[float, int, str]:
    start = time.perf_counter()
    outputs = engine.generate(
        input_ids=prompt_token_ids,
        sampling_params=make_sampling_params(
            args, len(prompt_token_ids), output_tokens
        ),
        return_logprob=True,
        logprob_start_len=-1,
    )
    elapsed = time.perf_counter() - start
    if isinstance(outputs, dict):
        outputs = [outputs]

    generated_ids: list[int] = []
    for output in outputs:
        token_ids = [int(token_id) for token_id in output["output_ids"]]
        if len(token_ids) != output_tokens:
            raise RuntimeError(
                f"expected {output_tokens} output tokens, received {len(token_ids)}"
            )
        output_logprobs = output["meta_info"].get("output_token_logprobs")
        if output_logprobs is None or len(output_logprobs) != output_tokens:
            raise RuntimeError("SGLang did not return one chosen-token logprob per token")
        generated_ids.extend(token_ids)

    digest = hashlib.sha256(
        b"".join(token_id.to_bytes(4, "little") for token_id in generated_ids)
    ).hexdigest()
    return elapsed, len(generated_ids), digest


def main() -> None:
    args = parse_args()

    from sglang import Engine  # pyright: ignore[reportMissingImports]
    from transformers import AutoTokenizer

    engine = Engine(
        model_path=args.model,
        dtype="bfloat16",
        context_length=args.max_model_len,
        mem_fraction_static=args.mem_fraction_static,
        tp_size=1,
        random_seed=args.seed,
        disable_radix_cache=True,
        page_size=64,
        attention_backend="flashinfer",
        cuda_graph_max_bs_decode=max(args.batch_sizes),
        speculative_algorithm="NEXTN",
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
        speculative_use_rejection_sampling=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    results: list[dict[str, Any]] = []
    try:
        for batch_size in args.batch_sizes:
            prompt_token_ids = build_prompt_token_ids(tokenizer, batch_size)
            run_batch(engine, args, prompt_token_ids, args.warmup_tokens)

            repetitions: list[dict[str, Any]] = []
            for repetition in range(args.repetitions):
                elapsed, generated_tokens, digest = run_batch(
                    engine, args, prompt_token_ids, args.output_tokens
                )
                repetitions.append(
                    {
                        "repetition": repetition,
                        "elapsed_seconds": elapsed,
                        "generated_tokens": generated_tokens,
                        "aggregate_tokens_per_second": generated_tokens / elapsed,
                        "tokens_per_second_per_stream": generated_tokens
                        / elapsed
                        / batch_size,
                        "generated_token_ids_sha256": digest,
                    }
                )

            aggregate_rates = [
                repetition["aggregate_tokens_per_second"]
                for repetition in repetitions
            ]
            results.append(
                {
                    "batch_size": batch_size,
                    "prompt_tokens_min": min(map(len, prompt_token_ids)),
                    "prompt_tokens_max": max(map(len, prompt_token_ids)),
                    "median_aggregate_tokens_per_second": statistics.median(
                        aggregate_rates
                    ),
                    "median_tokens_per_second_per_stream": statistics.median(
                        aggregate_rates
                    )
                    / batch_size,
                    "repetitions": repetitions,
                }
            )
    finally:
        engine.shutdown()

    payload = {
        "model": args.model,
        "engine": "mtp",
        "runtime": "sglang",
        "speculative_num_steps": 3,
        "speculative_num_draft_tokens": 4,
        "exact_rejection_sampling": True,
        "dtype": "bfloat16",
        "logprobs": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "warmup_tokens_per_stream": args.warmup_tokens,
        "measured_tokens_per_stream": args.output_tokens,
        "measurement_repetitions": args.repetitions,
        "max_model_len": args.max_model_len,
        "sglang_version": importlib.metadata.version("sglang"),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
