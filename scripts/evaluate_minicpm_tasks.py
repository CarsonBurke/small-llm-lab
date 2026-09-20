"""Evaluate a fixed verifiable-task development set with the compiled MiniCPM actor.

Model execution must be submitted through mlq. --dry-run only reads metadata.
The development split may have been seen in base-model training; these are local
adaptation holdouts, not a claim of pretraining-clean benchmark performance.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from postraining.repetition import repetition_metrics


def select_rows(rows: list[dict], limit: int, seed: int) -> list[dict]:
    if limit < 1:
        raise ValueError("per-domain problem limit must be positive")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    counts: Counter = Counter()
    selected = []
    for row in shuffled:
        domain = row["extra_info"]["domain"]
        if not isinstance(domain, str) or not domain:
            raise ValueError("each task requires a nonempty domain label")
        if counts[domain] < limit:
            selected.append(row)
            counts[domain] += 1
    if not selected:
        raise ValueError("development corpus is empty")
    return selected


def summarize(attempts: list[dict], samples: int, domains: list[str]) -> dict:
    def summary(items):
        scored = [item for item in items if isinstance(item.get("correct"), bool)]
        groups = defaultdict(list)
        for item in scored:
            groups[
                (item["domain"], item.get("question_sha256", item["question_id"]))
            ].append(item)
        complete = [group for group in groups.values() if len(group) == samples]
        successes = sum(item["correct"] for item in scored)
        metrics = {
            "attempts": len(scored),
            "pending_attempts": len(items) - len(scored),
            "successes": successes,
            "accuracy": successes / len(scored) if scored else None,
            "complete_questions": len(complete),
            "mixed_questions": sum(
                0 < sum(a["correct"] for a in g) < samples for g in complete
            ),
            "all_fail_questions": sum(
                not any(a["correct"] for a in g) for g in complete
            ),
            "all_pass_questions": sum(all(a["correct"] for a in g) for g in complete),
            "capped_attempts": sum(item["truncated"] for item in scored),
            "mean_tokens": sum(item["tokens"] for item in scored) / len(scored)
            if scored
            else None,
            "outcomes": dict(Counter(item["result"] for item in scored)),
        }
        for name in repetition_metrics(""):
            values = [item[name] for item in items if name in item]
            metrics[f"{name}_mean"] = sum(values) / len(values) if values else None
            metrics[f"{name}_max"] = max(values) if values else None
        return metrics

    return {
        "overall": summary(attempts),
        "domains": {
            domain: summary([a for a in attempts if a["domain"] == domain])
            for domain in domains
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="canonical verifiable-task development parquet",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-problems-per-domain", type=int, default=32)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument(
        "--prompt-suffix",
        default=None,
        help="inherit the checkpoint suffix unless explicitly overridden",
    )
    parser.add_argument("--max-new-tokens", type=int, default=10000)
    parser.add_argument(
        "--context-tokens", type=int, default=None,
        help="total prompt plus response cap; inherit checkpoint when omitted",
    )
    parser.add_argument("--answer-reserve-tokens", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--require-domain-success", action="store_true")
    args = parser.parse_args()

    import torch
    from postraining.eval_hf_math import _atomic_json, _atomic_jsonl
    from postraining.minicpm_eval import (
        load_evaluation_policy,
        resolve_evaluation_config,
    )
    from postraining.train_minicpm_vapo import (
        _stop_ids,
        encode_math_prompt,
        file_sha256,
        load_training_math_corpus,
        resolve_thinking_end_token,
        score_training_response,
        validate_output_directory,
    )
    from postraining.verifiable_tasks import verifiable_reward_identity

    validate_output_directory(args.output)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output must be new or empty: {args.output}")
    dataset = args.data
    rows, audit, corpus_identity = load_training_math_corpus(dataset, seed=args.seed)
    rows = select_rows(rows, args.max_problems_per_domain, args.seed)
    question_hashes = [
        hashlib.sha256(
            json.dumps(row["prompt"], sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        for row in rows
    ]
    before = args.checkpoint.stat()
    checkpoint_hash = file_sha256(args.checkpoint)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    after = args.checkpoint.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("checkpoint changed during metadata inspection")
    config = resolve_evaluation_config(
        checkpoint,
        {
            "prompt_tokens": args.prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "context_tokens": args.context_tokens,
            "samples_per_prompt": args.samples_per_prompt,
            "prompts_per_rollout": 4,
            "rollout_physical_batch_size": min(32, 4 * args.samples_per_prompt),
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "thinking": True,
            "answer_reserve_tokens": args.answer_reserve_tokens,
            "prompt_suffix": args.prompt_suffix,
            "seed": args.seed,
        },
    )
    report = {
        "schema": "minicpm_task_evaluation/v1",
        "status": "configured",
        "arm": "untouched_base" if args.stock else "trained",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": checkpoint["step"],
        "protocol": config,
        "dataset": str(dataset),
        "dataset_sha256": file_sha256(dataset),
        "corpus_identity": corpus_identity,
        "corpus_audit": audit,
        "reward_identity": verifiable_reward_identity(),
        "problems": len(rows),
        "domain_counts": dict(Counter(r["extra_info"]["domain"] for r in rows)),
        "question_ids": [
            row["extra_info"].get("index", question_hash)
            for row, question_hash in zip(rows, question_hashes)
        ],
        "expected_attempts": len(rows) * args.samples_per_prompt,
        "scope": "Fixed local development split; prior base-model exposure is possible.",
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("compiled BF16 CUDA evaluation required; no model fallback")
    from postraining.fast_inference import CapturedTrainingRolloutEngine
    from torch.utils.tensorboard import SummaryWriter

    args.output.mkdir(parents=True)
    attempts = []
    started = time.perf_counter()
    report["status"] = "loading"
    _atomic_json(args.output / "result.json", report)
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        policy, tokenizer = load_evaluation_policy(
            checkpoint, stock=args.stock, device=torch.device("cuda")
        )
        del checkpoint
        policy.eval()
        stop_ids = _stop_ids(policy, tokenizer)
        prompts = [
            encode_math_prompt(
                tokenizer,
                row,
                prompt_tokens=args.prompt_tokens,
                enable_thinking=True,
                prompt_suffix=config["prompt_suffix"],
            )
            for row in rows
        ]
        engine = CapturedTrainingRolloutEngine(
            policy,
            stop_ids=stop_ids,
            prompts_per_rollout=4,
            samples_per_prompt=args.samples_per_prompt,
            physical_batch_size=config["rollout_physical_batch_size"],
            cache_length=config["context_tokens"] or (args.prompt_tokens + args.max_new_tokens),
            temperature=1.0,
            top_k=-1,
            top_p=1.0,
            compile_decode=True,
            record_carry_history=False,
            answer_reserve_tokens=args.answer_reserve_tokens,
            thinking_end_token_id=resolve_thinking_end_token(
                tokenizer, stop_ids=stop_ids
            ),
        )
        report.update(status="running", rollout_arithmetic=engine.arithmetic)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        with (
            SummaryWriter(str(args.output / "tensorboard")) as writer,
            ThreadPoolExecutor(max_workers=8) as pool,
        ):
            writer.add_text("evaluation/configuration", json.dumps(report), 0)
            for begin in range(0, len(rows), 4):
                batch = rows[begin : begin + 4]
                generated = engine.generate_prompt_pool(
                    prompts[begin : begin + len(batch)],
                    max_new_tokens=args.max_new_tokens,
                    context_tokens=config["context_tokens"],
                )
                if len(generated.responses) != len(batch) * args.samples_per_prompt:
                    raise RuntimeError(
                        "rollout response count differs from prompt groups"
                    )
                pending = []
                for index, response in enumerate(generated.responses):
                    group, sample = divmod(index, args.samples_per_prompt)
                    row = batch[group]
                    ids = response.tolist()
                    if not ids or any(token in stop_ids for token in ids[:-1]):
                        raise RuntimeError("invalid response stop boundary")
                    terminated = ids[-1] in stop_ids
                    response_limit = generated.response_limits[index]
                    if len(ids) > response_limit or (not terminated and len(ids) != response_limit):
                        raise RuntimeError("response did not respect its own output budget")
                    content = ids[:-1] if terminated else ids
                    text = tokenizer.decode(
                        content,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    # Preserve raw generated evidence before invoking any verifier.
                    attempt = {
                        "question_id": row["extra_info"].get(
                            "index", question_hashes[begin + group]
                        ),
                        "domain": row["extra_info"]["domain"],
                        "question_sha256": question_hashes[begin + group],
                        "sample_index": sample,
                        "text": text,
                        "raw_text": text,
                        "tokens": len(ids),
                        "prompt_tokens": int(prompts[begin + group].numel()),
                        "response_limit": response_limit,
                        **repetition_metrics(text),
                        "truncated": not terminated and len(ids) == response_limit,
                        "correct": None,
                        "result": "pending",
                    }
                    attempts.append(attempt)
                    pending.append((attempt, row, prompts[begin + group], response))
                _atomic_jsonl(args.output / "attempts.jsonl", attempts)
                futures = [
                    (
                        attempt,
                        pool.submit(
                            score_training_response, tokenizer, row, prompt, response
                        ),
                    )
                    for attempt, row, prompt, response in pending
                ]
                for attempt, future in futures:
                    attempt["text"], attempt["correct"], attempt["result"] = (
                        future.result()
                    )
                report["metrics"] = summarize(
                    attempts, args.samples_per_prompt, sorted(report["domain_counts"])
                )
                _atomic_jsonl(args.output / "attempts.jsonl", attempts)
                _atomic_json(args.output / "result.json", report)
                event = {
                    "event": "evaluation",
                    "step": begin + len(batch),
                    "metrics": report["metrics"],
                    "elapsed_seconds": time.perf_counter() - started,
                }
                with (args.output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(event) + "\n")
                for domain, metrics in {
                    "overall": report["metrics"]["overall"],
                    **report["metrics"]["domains"],
                }.items():
                    for name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            writer.add_scalar(
                                f"evaluation/{domain}/{name}", value, begin + len(batch)
                            )
                writer.flush()
                print(json.dumps(event), flush=True)
        if len(attempts) != report["expected_attempts"] or any(
            a["correct"] is None for a in attempts
        ):
            raise RuntimeError("incomplete evaluation coverage")
        missing_signal = [
            domain
            for domain, metrics in report["metrics"]["domains"].items()
            if not metrics["successes"]
        ]
        report["domains_without_positive_reward"] = missing_signal
        if args.require_domain_success and missing_signal:
            raise RuntimeError(
                f"no positive development reward in domains: {missing_signal}"
            )
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report.update(
            wall_seconds=time.perf_counter() - started, generated_attempts=len(attempts)
        )
        _atomic_jsonl(args.output / "attempts.jsonl", attempts)
        _atomic_json(args.output / "result.json", report)


if __name__ == "__main__":
    main()
