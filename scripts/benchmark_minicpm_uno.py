#!/usr/bin/env python3
"""Measure matched AR/Uno continuous rollouts; run this workload through mlq."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uno-checkpoint", type=Path)
    parser.add_argument("--actor-checkpoint", type=Path)
    parser.add_argument(
        "--data", type=Path, default=Path("postraining/data/dapo-math-17k.parquet")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--engine",
        choices=("all", "ar", "optimized-ar", "legacy-ar", "uno"),
        default="all",
    )
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--samples-per-prompt", type=int, default=16)
    parser.add_argument("--physical-batch-size", type=int, default=64)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=10000)
    parser.add_argument(
        "--cache-length",
        type=int,
        help="Fixed logical KV capacity; defaults to prompt-tokens + max-new-tokens.",
    )
    parser.add_argument(
        "--export-responses",
        action="store_true",
        help="Include generated text and token IDs in each response's outcome record.",
    )
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--minimum-speedup", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def _publish(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _grade_responses(
    tokenizer,
    rows: list[dict],
    responses,
    *,
    samples_per_prompt: int,
    stop_ids,
    export_responses: bool,
) -> dict[str, Any]:
    """Score the existing rollout with the trainer's non-symbolic answer grader."""
    from postraining.core import answer_style, verify_answer

    if len(responses) != len(rows) * samples_per_prompt:
        raise RuntimeError("benchmark rollout lost prompt/sample correspondence")
    outcomes = []
    for index, response in enumerate(responses):
        prompt_index, sample_index = divmod(index, samples_per_prompt)
        row = rows[prompt_index]
        token_ids = response.tolist()
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        correct, prediction = verify_answer(
            text, row["reward_model"]["ground_truth"], answer_style(row)
        )
        completed = bool(token_ids) and token_ids[-1] in stop_ids
        outcome = {
            "prompt_index": prompt_index,
            "sample_index": sample_index,
            "response_tokens": len(token_ids),
            "prediction": prediction,
            "correct": bool(correct),
            "completed": completed,
            "truncated": not completed,
        }
        if export_responses:
            outcome.update(text=text, token_ids=token_ids)
        outcomes.append(outcome)
    count = len(outcomes)
    correct = sum(item["correct"] for item in outcomes)
    completed = sum(item["completed"] for item in outcomes)
    completed_correct = sum(
        item["correct"] and item["completed"] for item in outcomes
    )
    return {
        "response_count": count,
        "correct_count": correct,
        "correct_fraction": correct / count,
        "completed_count": completed,
        "completion_fraction": completed / count,
        "truncated_count": count - completed,
        "truncation_fraction": (count - completed) / count,
        "completed_correct_count": completed_correct,
        "completed_correct_fraction": completed_correct / completed
        if completed
        else None,
        "truncated_correct_count": correct - completed_correct,
        "responses": outcomes,
    }


def _worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from postraining.core import answer_style, load_unique_math_rows
    from postraining.fast_inference import CapturedTrainingRolloutEngine
    from postraining.minicpm_vapo import (
        MINICPM5_MODEL_ID,
        MINICPM5_REVISION,
        LoRAConfig,
        TrajectoryRecord,
        MiniCPMVAPOPolicy,
        load_adapter_state_dict,
    )
    from postraining.train_minicpm_vapo import (
        encode_math_prompt,
        file_sha256,
        _stop_ids,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    state = None
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    actor = None
    if args.actor_checkpoint:
        with torch.serialization.safe_globals([TrajectoryRecord]):
            state = torch.load(
                args.actor_checkpoint, map_location="cpu", weights_only=True
            )
        if state["policy"].get("schema") != "minicpm5_vapo_adapter/v6":
            raise ValueError("actor checkpoint must use the MiniCPM VAPO v6 schema")
        actor = state["policy"]["actor"]
    model_id = actor["model_id"] if actor else MINICPM5_MODEL_ID
    revision = actor["revision"] if actor else MINICPM5_REVISION
    config = (
        LoRAConfig(**actor["lora_config"])
        if actor
        else LoRAConfig(initialization="standard")
    )
    policy, tokenizer = MiniCPMVAPOPolicy.from_pretrained(
        model_id=model_id,
        revision=revision,
        device=torch.device("cuda"),
        lora_config=config,
        gradient_checkpointing=False,
        nextlat_projection_factor=float(actor["nextlat_projection_factor"])
        if actor
        else 1.6,
    )
    if actor:
        load_adapter_state_dict(policy.causal_lm, actor["adapter"])
        del actor, state
    policy.eval()
    rows = load_unique_math_rows(str(args.data))
    random.shuffle(rows)
    if len(rows) < args.prompts:
        raise ValueError("benchmark corpus has fewer unique prompts than requested")
    rows = rows[: args.prompts]
    prompts = [
        encode_math_prompt(
            tokenizer, row, prompt_tokens=args.prompt_tokens, enable_thinking=True
        )
        for row in rows
    ]
    options: dict[str, Any] = dict(
        stop_ids=_stop_ids(policy, tokenizer),
        prompts_per_rollout=args.prompts,
        samples_per_prompt=args.samples_per_prompt,
        physical_batch_size=args.physical_batch_size,
        cache_length=args.cache_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        compile_decode=True,
        invariant_decode=args.engine == "ar",
        optimized_decode=args.engine == "optimized-ar",
    )
    if args.engine == "uno":
        from postraining.uno_speculative import UnoTrainingRolloutEngine

        engine = UnoTrainingRolloutEngine(
            policy,
            uno_checkpoint=str(args.uno_checkpoint),
            uno_block_size=args.block_size,
            **options,
        )
    else:
        engine = CapturedTrainingRolloutEngine(policy, **options)

    # Warm the exact allocated cache and captured schedule; these outputs are not evidence.
    engine.generate_prompt_pool(prompts, max_new_tokens=min(args.max_new_tokens, 64))
    torch.cuda.synchronize()
    allocated_cache_length = int(engine.cache.layers[0].keys.shape[-2])
    repetitions = []
    for repetition in range(args.repetitions):
        # Same seed within each method; cross-method ID equality is not a correctness gate.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = engine.generate_prompt_pool(
            prompts, max_new_tokens=args.max_new_tokens
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        lengths = [int(response.numel()) for response in result.responses]
        math_outcomes = _grade_responses(
            tokenizer,
            rows,
            result.responses,
            samples_per_prompt=args.samples_per_prompt,
            stop_ids=engine.stop_ids,
            export_responses=args.export_responses,
        )
        repetitions.append(
            {
                "repetition": repetition,
                "rollout_seconds": elapsed,
                "useful_tokens": result.useful_tokens,
                "useful_tokens_per_second": result.useful_tokens / elapsed,
                "prefill_seconds": result.prefill_seconds,
                "decode_seconds": result.decode_seconds,
                "capacity_row_steps": result.capacity_row_steps,
                "productive_utilization": result.productive_utilization,
                "admission_events": result.admission_events,
                "minimum_active_rows_with_backlog": result.minimum_active_rows_with_backlog,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "lengths": lengths,
                "truncation_fraction": math_outcomes["truncation_fraction"],
                "math_outcomes": math_outcomes,
                **getattr(engine, "last_uno_metrics", {}),
            }
        )
    arithmetic = engine.arithmetic
    engine.release_cache()
    del engine
    gc.collect()
    report = {
        "schema": "minicpm_uno_benchmark/v3",
        "engine": args.engine,
        "rollout_arithmetic": arithmetic,
        "allocated_cache_length": allocated_cache_length,
        "model_id": model_id,
        "revision": revision,
        "actor_sha256": file_sha256(args.actor_checkpoint)
        if args.actor_checkpoint
        else "base",
        "uno_sha256": file_sha256(args.uno_checkpoint) if args.uno_checkpoint else None,
        "data_sha256": file_sha256(args.data),
        "prompt_token_ids": [prompt.tolist() for prompt in prompts],
        "prompt_answers": [
            {
                "prompt_index": index,
                "ground_truth": row["reward_model"]["ground_truth"],
                "answer_style": answer_style(row),
            }
            for index, row in enumerate(rows)
        ],
        "grading": {
            "api": "postraining.core.verify_answer",
            "policy": "Same non-symbolic grader and row-specific answer style as MiniCPM training; Minerva searches the final 300 characters, exact style searches the token-capped response.",
            "timing": "Decoding and grading occur after the generation timer stops.",
            "denominators": "correct_fraction, completion_fraction and truncation_fraction use all responses per repetition; completed_correct_fraction uses completed responses only (null if none).",
            "completion": "Completed means the last token is a stop ID; truncated means no terminal stop ID. Answer correctness is scored independently of completion.",
            "quality_note": "These sampled math outcomes are diagnostic only, not a learning-quality, equivalence, or losslessness proof.",
        },
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "repetitions": repetitions,
        "median_useful_tokens_per_second": statistics.median(
            item["useful_tokens_per_second"] for item in repetitions
        ),
        "median_rollout_seconds": statistics.median(
            item["rollout_seconds"] for item in repetitions
        ),
    }
    return report


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if (
        min(
            args.prompts,
            args.samples_per_prompt,
            args.physical_batch_size,
            args.prompt_tokens,
            args.max_new_tokens,
            args.repetitions,
        )
        < 1
    ):
        parser.error("benchmark dimensions must be positive")
    minimum_cache_length = args.prompt_tokens + args.max_new_tokens
    if args.cache_length is None:
        args.cache_length = minimum_cache_length
    elif args.cache_length < minimum_cache_length:
        parser.error("--cache-length must be >= prompt-tokens + max-new-tokens")
    if args.physical_batch_size > args.prompts * args.samples_per_prompt:
        parser.error("physical lanes cannot exceed logical trajectories")
    if args.engine in {"all", "uno"} and (
        args.uno_checkpoint is None or not args.uno_checkpoint.is_file()
    ):
        parser.error("--uno-checkpoint must be a trained adapter file for Uno")
    if not math.isfinite(args.minimum_speedup) or args.minimum_speedup <= 1:
        parser.error("--minimum-speedup must be finite and exceed one")
    if not 2 <= args.block_size <= 16:
        parser.error("--block-size must be between 2 and 16")
    if (
        not math.isfinite(args.temperature)
        or args.temperature <= 0
        or not 0 < args.top_p <= 1
        or args.top_k < 1
    ):
        parser.error("sampling parameters must define a finite nonempty distribution")
    if args.engine != "all":
        report = _worker(args)
        _publish(args.output, report)
        print(json.dumps(report, sort_keys=True), flush=True)
        return

    # Separate processes reclaim CUDA graph/compile pools between matched methods.
    reports = {}
    base_arguments = sys.argv[1:]
    for engine_name in ("legacy-ar", "optimized-ar", "ar", "uno"):
        child_output = args.output.with_name(f"{args.output.stem}.{engine_name}.json")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *base_arguments,
            "--engine",
            engine_name,
            "--output",
            str(child_output),
        ]
        subprocess.run(command, check=True)
        reports[engine_name] = json.loads(child_output.read_text())
    for key in (
        "model_id",
        "revision",
        "actor_sha256",
        "uno_sha256",
        "data_sha256",
        "prompt_token_ids",
    ):
        if any(
            reports[name][key] != reports["uno"][key]
            for name in ("ar", "legacy-ar", "optimized-ar")
        ):
            raise RuntimeError(f"benchmark arms differ in {key}")
    if reports["ar"]["rollout_arithmetic"] != reports["uno"]["rollout_arithmetic"]:
        raise RuntimeError("AR and Uno must share the invariant numerical target")
    speedup = (
        reports["uno"]["median_useful_tokens_per_second"]
        / reports["ar"]["median_useful_tokens_per_second"]
    )
    legacy_speedup = (
        reports["uno"]["median_useful_tokens_per_second"]
        / reports["legacy-ar"]["median_useful_tokens_per_second"]
    )
    optimized_speedup = (
        reports["uno"]["median_useful_tokens_per_second"]
        / reports["optimized-ar"]["median_useful_tokens_per_second"]
    )
    wall_speedup = (
        reports["ar"]["median_rollout_seconds"]
        / reports["uno"]["median_rollout_seconds"]
    )
    report = {
        "schema": "minicpm_uno_comparison/v3",
        "ar": reports["ar"],
        "legacy_ar": reports["legacy-ar"],
        "optimized_ar": reports["optimized-ar"],
        "uno": reports["uno"],
        "useful_rollout_speedup": speedup,
        "legacy_ar_useful_rollout_speedup": legacy_speedup,
        "optimized_ar_useful_rollout_speedup": optimized_speedup,
        "logical_pool_wall_speedup": wall_speedup,
        "required_speedup": args.minimum_speedup,
        "performance_gate_passed": min(speedup, legacy_speedup, optimized_speedup)
        >= args.minimum_speedup,
        "correctness_note": "AR and Uno share the invariant bf16 target; optimized AR is the production default and legacy AR is a historical control. All three AR speedup gates must pass. Throughput is not a losslessness or learning-quality proof.",
    }
    _publish(args.output, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    raise SystemExit(0 if report["performance_gate_passed"] else 2)


if __name__ == "__main__":
    main()
