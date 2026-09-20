"""Evaluate untouched MiniCPM5-1B on pinned KodCode executable rewards.

Submit model execution through mlq. --dry-run reads checkpoint metadata only;
--prepare-only performs CPU dataset selection and sandboxed reference checks.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import re
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL = "openbmb/MiniCPM5-1B"
REVISION = "87179e5c1f455ef22e6223592d2d61351b525bfc"
DATASET = "KodCode/KodCode-Light-RL-10K"
SCHEMA = "minicpm_kodcode_evaluation/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="stable trusted local checkpoint; only base identity/config is used, never trained weights",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new or empty directory under postraining/runs",
    )
    parser.add_argument(
        "--dataset-revision",
        required=True,
        help="immutable 40-character Hugging Face commit",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--dry-run",
        action="store_true",
        help="validate arguments/checkpoint metadata only; no outputs",
    )
    modes.add_argument(
        "--prepare-only",
        action="store_true",
        help="write selected rows and reference-check manifest; no model",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        help="reuse digest-verified prepared_rows.jsonl and manifest.json",
    )
    for name, default in (
        ("problems", 256),
        ("samples-per-prompt", 8),
        ("prompts-per-rollout", 4),
        ("rollout-physical-batch-size", 32),
        ("prompt-tokens", 2048),
        ("max-new-tokens", 10000),
        ("answer-reserve-tokens", 1000),
        ("seed", 42),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--context-tokens", type=int, default=None)
    return parser


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload, *, lines: bool = False) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        if lines:
            for item in payload:
                stream.write(
                    json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n"
                )
        else:
            stream.write(
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
                + "\n"
            )
    temporary.replace(path)


def load_metadata(args):
    """No dataset, tokenizer, model, trainer or compiled-engine imports here."""
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.dataset_revision):
        raise ValueError("--dataset-revision must be an immutable 40-character commit")
    args.dataset_revision = args.dataset_revision.lower()
    if args.problems < 1:
        raise ValueError("--problems must be positive")
    if args.answer_reserve_tokens < 1:
        raise ValueError("this thinking evaluation requires a positive answer reserve")
    if args.rollout_physical_batch_size < 1:
        raise ValueError("--rollout-physical-batch-size must be positive")
    if args.prepare_only and args.prepared_dir:
        raise ValueError("--prepare-only cannot reuse --prepared-dir")
    output = args.output.resolve()
    runs_root = (ROOT / "postraining/runs").resolve()
    if output == runs_root or not output.is_relative_to(runs_root):
        raise ValueError(f"--output must be a run directory beneath {runs_root}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"evaluation output must be new or empty: {output}")
    import torch
    from postraining.minicpm_eval import resolve_evaluation_config

    before = args.checkpoint.stat()
    checkpoint_digest = file_sha256(args.checkpoint)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    after = args.checkpoint.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(
            "checkpoint changed during setup; supply a preserved stable checkpoint"
        )
    overrides = {
        name: getattr(args, name)
        for name in (
            "samples_per_prompt",
            "prompts_per_rollout",
            "rollout_physical_batch_size",
            "prompt_tokens",
            "max_new_tokens",
            "context_tokens",
            "answer_reserve_tokens",
            "seed",
        )
    }
    overrides.update(
        thinking=True,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        prompt_suffix=(
            f"You have a thinking budget of {args.max_new_tokens - args.answer_reserve_tokens} "
            "tokens. Put your final solution in a Python code block."
        ),
    )
    config = resolve_evaluation_config(checkpoint, overrides)
    if config["context_tokens"] is not None:
        config["prompt_suffix"] = "Put your final solution in a Python code block."
        overrides["prompt_suffix"] = config["prompt_suffix"]
    if (config["model"], config["revision"]) != (MODEL, REVISION):
        raise ValueError(f"this evaluation requires {MODEL} at {REVISION}")
    versions = {}
    for package in ("torch", "transformers", "datasets", "tensorboard"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {
        "schema": SCHEMA,
        "status": "configured",
        "arm": "untouched_base",
        "model": MODEL,
        "model_revision": REVISION,
        "stock": True,
        "trained_adapter_loaded": False,
        "token_carry": False,
        "latent_thinking": False,
        "base_loading_note": "stock=True creates zero-effect standard LoRA; saved adapter, NextLat and carry weights are not loaded",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_role": "base identity and validated metadata only",
        "dataset": DATASET,
        "dataset_revision": args.dataset_revision,
        "license": "CC-BY-NC-4.0",
        "requested_problems": args.problems,
        "selection_seed": args.seed,
        "protocol": config,
        "overrides": overrides,
        "compiled_decode": True,
        "dtype": "bfloat16",
        "prompt_overflow_policy": "exclude without truncation or replacement",
        "primary_metric": "observed per-attempt executable-test accuracy",
        "group_metric": "observed any-success fraction of observed groups, including incomplete groups; not Avg@k or an estimator",
        "completion_scope": "selected sandbox-compatible problems that fit the prompt limit, not the universal dataset",
        "output": str(output),
        "versions": versions,
    }
    return checkpoint, config, manifest


def prepare(args, manifest):
    if args.prepared_dir:
        source = args.prepared_dir.resolve()
        saved = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        for field in (
            "schema",
            "dataset",
            "dataset_revision",
            "requested_problems",
            "selection_seed",
        ):
            if saved.get(field) != manifest[field]:
                raise ValueError(f"prepared manifest differs in {field}")
        if saved.get("preparation_status") != "completed":
            raise ValueError("prepared dataset did not finish reference preflight")
        path = source / "prepared_rows.jsonl"
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != saved.get("prepared_rows_sha256"):
            raise ValueError("prepared dataset digest mismatch")
        rows = [
            json.loads(line)
            for line in raw.decode("utf-8").splitlines()
            if line.strip()
        ]
        selection = saved["selection"]
        manifest["prepared_source"] = str(source)
        manifest["prepared_source_manifest_sha256"] = file_sha256(
            source / "manifest.json"
        )
    else:
        from postraining.kodcode_eval import prepare_rows

        rows, selection = prepare_rows(
            args.dataset_revision, problems=args.problems, seed=args.seed
        )
    manifest.update(selection=selection, selected_problems=len(rows))
    atomic_json(args.output / "manifest.json", manifest)
    if len(rows) != args.problems or not selection.get("sample_complete"):
        atomic_json(args.output / "prepared_rows.jsonl", rows, lines=True)
        manifest.update(
            preparation_status="incomplete",
            prepared_rows_sha256=file_sha256(args.output / "prepared_rows.jsonl"),
        )
        raise ValueError(
            f"reference-compatible selection contains {len(rows)} of {args.problems} requested problems"
        )
    ids = [row["question_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("prepared selection contains duplicate question IDs")
    for row in rows:
        for field in (
            "subset",
            "difficulty",
            "prompt",
            "verification_info",
            "question_sha256",
        ):
            if field not in row:
                raise ValueError(f"prepared row missing {field}")
        if not isinstance(row["prompt"], list) or not row["prompt"]:
            raise ValueError("prepared prompt must be a nonempty message list")
        if not any(message.get("role") == "user" for message in row["prompt"]):
            raise ValueError("prepared prompt has no user message")
    atomic_json(args.output / "prepared_rows.jsonl", rows, lines=True)
    manifest.update(
        selection=selection,
        preparation_status="completed",
        selected_problems=len(rows),
        prepared_rows_sha256=file_sha256(args.output / "prepared_rows.jsonl"),
    )
    atomic_json(args.output / "manifest.json", manifest)
    return rows


def encode_prompt(tokenizer, row, config):
    messages = [dict(message) for message in row["prompt"]]
    user_index = max(
        index for index, message in enumerate(messages) if message["role"] == "user"
    )
    messages[user_index]["content"] += "\n\n" + config["prompt_suffix"]
    options = dict(add_generation_prompt=True, enable_thinking=True)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **options)
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, **options)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    ids = list(ids)
    if not ids or any(type(token) is not int for token in ids):
        raise ValueError("chat tokenizer did not produce a nonempty flat token-ID list")
    return ids, rendered


def decode_response(
    tokenizer, row, response, *, sample_index, stop_ids, max_new_tokens
):
    continuation = response.tolist()
    if not continuation or len(continuation) > max_new_tokens:
        raise ValueError("rollout returned an empty or over-budget response")
    if any(token in stop_ids for token in continuation[:-1]):
        raise ValueError("rollout returned tokens after a terminal stop")
    terminated = continuation[-1] in stop_ids
    raw = tokenizer.decode(
        continuation, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    content_ids = continuation[:-1] if terminated else continuation
    text = tokenizer.decode(
        content_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    # A chat generation prefix can open thinking outside the generated response.
    # Include that context for the extractor without altering the raw evidence.
    prompt = row["rendered_prompt"]
    thinking_open = prompt.rfind("<think>") > prompt.rfind("</think>")
    prefix = (
        "<think>" if thinking_open and not text.lstrip().startswith("<think>") else ""
    )
    return {
        "question_id": row["question_id"],
        "subset": row["subset"],
        "difficulty": row["difficulty"],
        "sample_index": sample_index,
        "correct": None,
        "result": "pending_verification",
        "raw_text": raw,
        "text": prefix + text,
        "scoring_prefix": prefix,
        "token_ids": continuation,
        "tokens": len(continuation),
        "terminated": terminated,
        "stop_token_id": continuation[-1] if terminated else None,
        "extracted_code": None,
        "truncated": not terminated,
        "prompt_tokens": row["prompt_tokens"],
        "response_limit": max_new_tokens,
    }


def metrics_for(attempts, rows, config):
    from postraining.kodcode_eval import summarize

    scored = [item for item in attempts if isinstance(item["correct"], bool)]
    metrics = summarize(scored, samples_per_problem=config["samples_per_prompt"])
    counts = Counter(item["question_id"] for item in scored)
    missing = [row["question_id"] for row in rows if not counts[row["question_id"]]]
    incomplete = [
        row["question_id"]
        for row in rows
        if 0 < counts[row["question_id"]] < config["samples_per_prompt"]
    ]
    metrics["coverage"] = {
        "effective_problems": len(rows),
        "expected_attempts": len(rows) * config["samples_per_prompt"],
        "generated_attempts": len(attempts),
        "scored_attempts": len(scored),
        "pending_verification_attempts": len(attempts) - len(scored),
        "missing_problem_ids": missing,
        "incomplete_problem_ids": incomplete,
        "complete_problems": sum(
            count == config["samples_per_prompt"] for count in counts.values()
        ),
    }
    return metrics


def write_html(path, manifest, rows, attempts):
    escape = lambda value: html.escape(str(value), quote=True)
    grouped = {}
    for attempt in attempts:
        grouped.setdefault(attempt["question_id"], []).append(attempt)
    temporary = path.with_suffix(".html.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(
            '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            "<title>KodCode executable reward evaluation</title><style>"
            "body{max-width:1100px;margin:2rem auto;padding:0 1rem;font:16px system-ui;background:#fafafa;color:#171717}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eee;padding:1rem;border-radius:6px}"
            "section{border-top:2px solid #bbb;margin-top:2rem}details{margin:1rem 0}summary{cursor:pointer}"
            ".pass{color:#17622b}.fail{color:#912121}</style><h1>KodCode executable reward evaluation</h1>"
        )
        stream.write(
            f"<p>Status: <strong>{escape(manifest['status'])}</strong>. Untouched base; no trained adapter or token carry.</p>"
        )
        stream.write(
            "<p>Any-success is an observed-group fraction, including incomplete groups, not Avg@k. Complete outcome groups, missing groups, and incomplete groups are reported separately.</p>"
        )
        stream.write(
            "<details><summary>Manifest and metrics</summary><pre>"
            + escape(json.dumps(manifest, indent=2))
            + "</pre></details>"
        )
        for row in rows:
            stream.write(
                f"<section><h2>{escape(row['question_id'])}</h2><p>Subset: {escape(row['subset'])}; difficulty: {escape(row['difficulty'])}</p>"
            )
            prompt = row.get(
                "rendered_prompt",
                json.dumps(row["prompt"], ensure_ascii=False, indent=2),
            )
            stream.write(
                "<details><summary>Full prompt</summary><pre>"
                + escape(prompt)
                + "</pre></details>"
            )
            if not grouped.get(row["question_id"]):
                stream.write("<p>No scored completion yet.</p>")
            for attempt in grouped.get(row["question_id"], []):
                css = "pass" if attempt["correct"] else "fail"
                stream.write(
                    f'<details><summary class="{css}">Sample {attempt["sample_index"] + 1}: {escape(attempt["result"])}; '
                    f"{attempt['tokens']} tokens; terminated={attempt['terminated']}</summary>"
                )
                stream.write(
                    "<h3>Raw decoded response (including special tokens)</h3><pre>"
                    + escape(attempt["raw_text"])
                    + "</pre>"
                )
                stream.write(
                    "<h3>Extracted code</h3><pre>"
                    + escape(attempt["extracted_code"] or "No eligible final code")
                    + "</pre>"
                )
                stream.write(
                    "<p>Execution reason: "
                    + escape(attempt.get("execution_error", attempt["result"]))
                    + "</p></details>"
                )
            stream.write("</section>")
        stream.write("</html>")
    temporary.replace(path)


def write_scalars(writer, metrics, step, prefix="kodcode"):
    for key, value in metrics.items():
        tag = f"{prefix}/{key}"
        if isinstance(value, dict):
            write_scalars(writer, value, step, tag)
        elif isinstance(value, (int, float)):
            writer.add_scalar(tag, value, step)


def main() -> None:
    args = build_parser().parse_args()
    checkpoint, config, manifest = load_metadata(args)
    if args.dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    attempts, rows = [], []
    manifest["status"] = "preparing"
    atomic_json(args.output / "manifest.json", manifest)
    atomic_json(args.output / "result.json", manifest)

    def interrupt(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    previous_handlers = {
        sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        rows = prepare(args, manifest)
        if args.prepare_only:
            manifest["status"] = "prepared"
        else:
            import torch
            from torch.utils.tensorboard import SummaryWriter
            from postraining.fast_inference import CapturedTrainingRolloutEngine
            from postraining.kodcode_eval import extract_code, score_completion
            from postraining.minicpm_eval import load_evaluation_policy
            from postraining.train_minicpm_vapo import (
                _stop_ids,
                resolve_thinking_end_token,
            )

            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "compiled BF16 CUDA evaluation is required; no CPU/eager/FP32 fallback"
                )
            device = torch.device("cuda")
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])
            manifest["status"] = "loading"
            atomic_json(args.output / "result.json", manifest)
            policy, tokenizer = load_evaluation_policy(
                checkpoint, stock=True, device=device
            )
            del checkpoint
            if getattr(policy, "token_carry", False) or getattr(
                policy, "latent_thinking", False
            ):
                raise RuntimeError(
                    "stock policy unexpectedly enabled carry or latent thinking"
                )
            eligible, prompts, excluded = [], [], []
            for row in rows:
                ids, rendered = encode_prompt(tokenizer, row, config)
                if len(ids) > config["prompt_tokens"]:
                    excluded.append(
                        {
                            "question_id": row["question_id"],
                            "prompt_tokens": len(ids),
                            "reason": "prompt_overflow",
                        }
                    )
                    continue
                eligible.append(
                    {**row, "rendered_prompt": rendered, "prompt_tokens": len(ids)}
                )
                prompts.append(torch.tensor(ids, dtype=torch.long))
            rows = eligible
            manifest.update(
                prompt_overflow_count=len(excluded),
                prompt_exclusions=excluded,
                effective_problems=len(rows),
                effective_problem_ids=[row["question_id"] for row in rows],
                expected_attempts=len(rows) * config["samples_per_prompt"],
            )
            atomic_json(
                args.output / "prompts.jsonl",
                [
                    {
                        key: row[key]
                        for key in ("question_id", "prompt_tokens", "rendered_prompt")
                    }
                    for row in rows
                ],
                lines=True,
            )
            atomic_json(args.output / "manifest.json", manifest)
            if not rows:
                raise ValueError("every selected question exceeds the prompt limit")
            stop_ids = _stop_ids(policy, tokenizer)
            engine = CapturedTrainingRolloutEngine(
                policy,
                stop_ids=stop_ids,
                prompts_per_rollout=config["prompts_per_rollout"],
                samples_per_prompt=config["samples_per_prompt"],
                physical_batch_size=config["rollout_physical_batch_size"],
                cache_length=config["context_tokens"] or (config["prompt_tokens"] + config["max_new_tokens"]),
                temperature=1.0,
                top_k=-1,
                top_p=1.0,
                compile_decode=True,
                record_carry_history=False,
                answer_reserve_tokens=config["answer_reserve_tokens"],
                thinking_end_token_id=resolve_thinking_end_token(
                    tokenizer, stop_ids=stop_ids
                ),
            )
            manifest.update(
                status="running",
                rollout_arithmetic=engine.arithmetic,
                physical_batch_size=engine.batch_size,
                stop_ids=list(stop_ids),
                device=torch.cuda.get_device_name(device),
                load_seconds=time.perf_counter() - started,
            )
            atomic_json(args.output / "manifest.json", manifest)
            atomic_json(args.output / "result.json", manifest)
            torch.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])
            torch.cuda.reset_peak_memory_stats(device)
            with SummaryWriter(str(args.output / "tensorboard")) as writer:
                writer.add_text("kodcode/protocol", json.dumps(manifest, indent=2), 0)
                for begin in range(0, len(rows), config["prompts_per_rollout"]):
                    batch = rows[begin : begin + config["prompts_per_rollout"]]
                    batch_started = time.perf_counter()
                    generated = engine.generate_prompt_pool(
                        prompts[begin : begin + len(batch)],
                        max_new_tokens=config["max_new_tokens"],
                        context_tokens=config["context_tokens"],
                        prefill_batch_prompts=config["prompts_per_rollout"],
                    )
                    expected = len(batch) * config["samples_per_prompt"]
                    batch_attempts = []
                    try:
                        if len(generated.responses) != expected:
                            raise ValueError(
                                f"rollout returned {len(generated.responses)} responses; expected {expected}"
                            )
                        for index, response in enumerate(generated.responses):
                            prompt_index, sample_index = divmod(
                                index, config["samples_per_prompt"]
                            )
                            attempt = decode_response(
                                tokenizer,
                                batch[prompt_index],
                                response,
                                sample_index=sample_index,
                                stop_ids=stop_ids,
                                max_new_tokens=generated.response_limits[index],
                            )
                            batch_attempts.append(attempt)
                            attempts.append(attempt)
                    except BaseException:
                        atomic_json(
                            args.output / "unexpected_responses.jsonl",
                            [
                                {
                                    "index": index,
                                    "token_ids": response.tolist(),
                                    "raw_text": tokenizer.decode(
                                        response.tolist(),
                                        skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False,
                                    ),
                                }
                                for index, response in enumerate(generated.responses)
                            ],
                            lines=True,
                        )
                        raise
                    # Persist all generated evidence before any sandbox invocation.
                    atomic_json(args.output / "attempts.jsonl", attempts, lines=True)
                    for index, attempt in enumerate(batch_attempts):
                        row = batch[index // config["samples_per_prompt"]]
                        attempt["extracted_code"] = extract_code(attempt["text"])
                        try:
                            attempt["result"] = score_completion(
                                attempt["text"], row["verification_info"]
                            )
                            attempt["correct"] = attempt["result"] == "pass"
                        except BaseException as error:
                            attempt["execution_error"] = repr(error)
                            raise
                    manifest["metrics"] = metrics_for(attempts, rows, config)
                    manifest["completed_attempts"] = sum(
                        isinstance(item["correct"], bool) for item in attempts
                    )
                    event = {
                        "event": "evaluation",
                        "step": begin + len(batch),
                        "metrics": manifest["metrics"],
                        "batch_seconds": time.perf_counter() - batch_started,
                        "elapsed_seconds": time.perf_counter() - started,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    }
                    atomic_json(args.output / "attempts.jsonl", attempts, lines=True)
                    atomic_json(args.output / "result.json", manifest)
                    with (args.output / "metrics.jsonl").open(
                        "a", encoding="utf-8"
                    ) as stream:
                        stream.write(json.dumps(event, allow_nan=False) + "\n")
                    write_scalars(writer, event["metrics"], event["step"])
                    writer.add_scalar(
                        "system/batch_seconds", event["batch_seconds"], event["step"]
                    )
                    writer.add_scalar(
                        "system/peak_allocated_bytes",
                        event["peak_allocated_bytes"],
                        event["step"],
                    )
                    writer.flush()
                    write_html(args.output / "responses.html", manifest, rows, attempts)
                    print(json.dumps(event), flush=True)
            if len(attempts) != manifest["expected_attempts"] or any(
                item["correct"] is None for item in attempts
            ):
                raise RuntimeError("evaluation ended with incomplete attempt coverage")
            manifest["status"] = "completed"
    except BaseException as error:
        manifest["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        manifest["error"] = repr(error)
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        manifest["wall_seconds"] = time.perf_counter() - started
        manifest["generated_attempts"] = len(attempts)
        manifest["completed_attempts"] = sum(
            isinstance(item["correct"], bool) for item in attempts
        )
        if rows and not args.prepare_only:
            manifest["metrics"] = metrics_for(attempts, rows, config)
        atomic_json(args.output / "attempts.jsonl", attempts, lines=True)
        atomic_json(args.output / "result.json", manifest)
        atomic_json(args.output / "manifest.json", manifest)
        write_html(args.output / "responses.html", manifest, rows, attempts)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
