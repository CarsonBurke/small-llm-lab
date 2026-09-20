"""On-demand AIME evaluation using MiniCPM's compiled, carry-aware rollout engine.

Launch model evaluation through mlq. --dry-run inspects local metadata only.
Defaults inherit the supplied training checkpoint; --stock uses its untouched
base model with exactly the same resolved evaluation protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stock", action="store_true", help="evaluate the unmodified base, using the checkpoint's protocol")
    parser.add_argument("--dry-run", action="store_true", help="validate checkpoint, dataset and protocol without loading a model or writing outputs")
    parser.add_argument("--suite", choices=(
        "aime_2026", "aime_2026_i", "aime_2026_ii",
        "aime_2025", "aime_2025_i", "aime_2025_ii", "aime_2024",
    ), default="aime_2026")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None)
    for option in (
        "prompt-tokens", "max-new-tokens", "context-tokens", "samples-per-prompt", "prompts-per-rollout",
        "rollout-physical-batch-size", "top-k", "answer-reserve-tokens", "seed",
    ):
        parser.add_argument(f"--{option}", type=int, default=None, help="override the saved training setting")
    for option in ("temperature", "top-p"):
        parser.add_argument(f"--{option}", type=float, default=None, help="override the saved training setting")
    parser.add_argument("--prompt-suffix", default=None, help="override the saved suffix; an empty string removes it")
    return parser


def score_response_batch(tokenizer, rows, responses, *, begin, samples, stop_ids, style):
    """Preserve prompt/sample identity, including an incomplete final prompt batch."""
    from postraining.eval_hf_math import _attempt_record

    if len(responses) != len(rows) * samples:
        raise ValueError("rollout response count differs from the evaluation batch")
    attempts = []
    for index, response in enumerate(responses):
        prompt_index, sample_index = divmod(index, samples)
        continuation = response.tolist()
        if not continuation:
            raise ValueError("rollout returned an empty response")
        attempts.append(_attempt_record(
            row=rows[prompt_index], problem_index=begin + prompt_index,
            sample_index=sample_index, continuation=continuation,
            terminated=continuation[-1] in stop_ids, tokenizer=tokenizer, style=style,
        ))
    return attempts


def main() -> None:
    args = build_parser().parse_args()
    import torch
    from postraining.core import load_unique_math_rows
    from postraining.eval_hf_math import AVAILABLE_SUITES, _atomic_json, _atomic_jsonl, summarize_attempts
    from postraining.minicpm_eval import load_evaluation_policy, resolve_evaluation_config
    from postraining.train_minicpm_vapo import (
        _stop_ids, encode_math_prompt, file_sha256, resolve_thinking_end_token,
        validate_output_directory,
    )

    validate_output_directory(args.output)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"evaluation output must be new or empty: {args.output}")
    before = args.checkpoint.stat()
    checkpoint_sha256 = file_sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    after = args.checkpoint.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError("checkpoint changed during evaluation setup; retry with a stable checkpoint")
    overrides = {key: value for key, value in vars(args).items() if key not in {
        "checkpoint", "output", "stock", "dry_run", "suite",
    }}
    config = resolve_evaluation_config(checkpoint, overrides)
    suite = next(suite for suite in AVAILABLE_SUITES if suite.name == args.suite)
    dataset = ROOT / suite.path
    rows = load_unique_math_rows(str(dataset))
    expected_problems = 15 if args.suite.endswith(("_i", "_ii")) else 30
    if len(rows) != expected_problems:
        raise ValueError(f"{args.suite} requires {expected_problems} unique problems, got {len(rows)}")
    manifest = {
        "schema": "minicpm_vapo_evaluation/v1",
        "status": "configured",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": int(checkpoint["step"]),
        "arm": "stock" if args.stock else "trained",
        "token_carry": bool(checkpoint["policy"]["actor"].get("token_carry", False)) and not args.stock,
        "slot_memory": None if args.stock else checkpoint["policy"]["actor"].get("slot_memory"),
        "protocol": config,
        "overrides": {key: value for key, value in overrides.items() if value is not None},
        "suite": suite.name,
        "dataset": str(dataset),
        "dataset_sha256": file_sha256(dataset),
        "problems": len(rows),
        "attempts": len(rows) * config["samples_per_prompt"],
        "output": str(args.output.resolve()),
        "primary_metric": "contract_content_accuracy",
        "primary_metric_definition": "mean per-attempt answer correctness, including capped responses; not pass@k",
        "published_reference_percent": 40.42 if suite.name in ("aime_2025", "aime_2026") else None,
        "published_reference_metric": "Avg@16" if suite.name in ("aime_2025", "aime_2026") else None,
        "publisher_protocol_matched": False,
    }
    if suite.name.startswith(("aime_2025", "aime_2026")):
        year = suite.name.split("_")[1]
        manifest["dataset_provenance"] = json.loads((ROOT / f"postraining/data/aime-{year}.manifest.json").read_text())
    if args.dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("MiniCPM evaluation requires CUDA; no CPU model fallback")

    from postraining.fast_inference import CapturedTrainingRolloutEngine
    from torch.utils.tensorboard import SummaryWriter

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    args.output.mkdir(parents=True, exist_ok=True)
    manifest["status"] = "running"
    _atomic_json(args.output / "result.json", manifest)
    started = time.perf_counter()
    attempts = []
    try:
        policy, tokenizer = load_evaluation_policy(checkpoint, stock=args.stock, device=device)
        del checkpoint
        policy.eval()
        stop_ids = _stop_ids(policy, tokenizer)
        engine = CapturedTrainingRolloutEngine(
            policy, stop_ids=stop_ids,
            prompts_per_rollout=config["prompts_per_rollout"],
            samples_per_prompt=config["samples_per_prompt"],
            physical_batch_size=config["rollout_physical_batch_size"] or None,
            cache_length=config["context_tokens"] or (config["prompt_tokens"] + config["max_new_tokens"]),
            temperature=config["temperature"], top_k=config["top_k"], top_p=config["top_p"],
            compile_decode=True, record_carry_history=False,
            answer_reserve_tokens=config["answer_reserve_tokens"],
            thinking_end_token_id=(resolve_thinking_end_token(tokenizer, stop_ids=stop_ids)
                                   if config["answer_reserve_tokens"] else None),
        )
        manifest["rollout_arithmetic"] = engine.arithmetic
        manifest["stop_ids"] = list(stop_ids)
        manifest["physical_batch_size"] = engine.batch_size
        manifest["load_seconds"] = time.perf_counter() - started
        _atomic_json(args.output / "result.json", manifest)
        # Model/adapter construction consumes different RNG draws in the two arms.
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.reset_peak_memory_stats(device)
        evaluation_started = time.perf_counter()
        with SummaryWriter(str(args.output / "tensorboard")) as writer:
            writer.add_text("evaluation/protocol", json.dumps(manifest, indent=2), 0)
            for begin in range(0, len(rows), config["prompts_per_rollout"]):
                batch = rows[begin:begin + config["prompts_per_rollout"]]
                prompts = [encode_math_prompt(
                    tokenizer, row, prompt_tokens=config["prompt_tokens"],
                    enable_thinking=config["thinking"], prompt_suffix=config["prompt_suffix"],
                    truncate_prompt=config["context_tokens"] is None,
                ) for row in batch]
                generated = engine.generate_prompt_pool(
                    prompts, max_new_tokens=config["max_new_tokens"],
                    context_tokens=config["context_tokens"],
                )
                attempts.extend(score_response_batch(
                    tokenizer, batch, generated.responses, begin=begin,
                    samples=config["samples_per_prompt"], stop_ids=stop_ids,
                    style=suite.answer_style_override,
                ))
                for attempt, limit in zip(
                    attempts[-len(generated.responses):], generated.response_limits, strict=True,
                ):
                    attempt["response_limit"] = limit
                completed_problems = begin + len(batch)
                metrics = summarize_attempts(
                    attempts, problem_count=completed_problems,
                    samples_per_problem=config["samples_per_prompt"],
                    elapsed_seconds=time.perf_counter() - evaluation_started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                )
                _atomic_jsonl(args.output / "attempts.jsonl", attempts)
                event = {"event": "evaluation", "step": completed_problems, "suite": suite.name, "metrics": metrics}
                with (args.output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(event) + "\n")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"{suite.name}/{key}", value, completed_problems)
                writer.flush()
                print(json.dumps(event), flush=True)
            manifest["metrics"] = metrics
        manifest["status"] = "completed"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["wall_seconds"] = time.perf_counter() - started
        manifest["completed_attempts"] = len(attempts)
        _atomic_json(args.output / "result.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
