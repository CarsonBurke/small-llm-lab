"""Frozen three-arm gate for whether answer privilege improves the OPSD teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch

from pretraining.fresh_lejepa.fresh_lejepa_train import FreshHyperparameters
from postraining.core import answer_style, load_posttraining_tokenizer
from postraining.latent_eval import evaluate_latent_math, verify_terminated_answer
from postraining.latent_thought import LatentThoughtModel
from postraining.math_prompt import (
    ANSWER_FENCE_PROMPT_SCHEMA,
    answer_fence_prompt,
    require_answer_fence_prompt_schema,
)
from postraining.model_io import load_model
from postraining.opsd.data import TEACHER_PROMPT_SCHEMA, build_teacher_prompt
from postraining.prepare_sft_traces import INSTRUCTION_SUFFIX_ANSWER
from postraining.opsd.schemas import OPSD_PROMPT_SCHEMA
from postraining.opsd.prepare_dapo import (
    DAPO_OPSD_DATA_SCHEMA,
    DAPO_OPSD_SPLIT_SCHEMA,
    file_sha256,
)


TEACHER_UPLIFT_SCHEMA = "opsd_frozen_teacher_uplift/v3"
ARMS = ("question_only", "correct_answer", "permuted_answer")


def paired_prompt_stats(
    counts_a: list[int],
    counts_b: list[int],
    samples: int,
    seed: int,
    resamples: int = 20_000,
) -> dict[str, float | int]:
    if len(counts_a) != len(counts_b) or not counts_a:
        raise ValueError("paired stats require identical nonempty panels")
    deltas = torch.tensor(
        [(a - b) / samples for a, b in zip(counts_a, counts_b, strict=True)],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(seed)
    prompts = deltas.numel()
    indices = torch.randint(0, prompts, (resamples, prompts), generator=generator)
    bootstrap = deltas[indices].mean(1)
    signs = (
        torch.randint(
            0, 2, (resamples, prompts), generator=generator,
            dtype=torch.float64,
        )
        * 2
        - 1
    )
    mean_delta = float(deltas.mean())
    permuted = (signs * deltas).mean(1)
    extreme = int((permuted.abs() >= abs(mean_delta) - 1e-12).sum())
    return {
        "prompts": prompts,
        "samples_per_prompt": samples,
        "mean_delta": mean_delta,
        "bootstrap_ci_low": float(torch.quantile(bootstrap, 0.025)),
        "bootstrap_ci_high": float(torch.quantile(bootstrap, 0.975)),
        "permutation_p": (extreme + 1) / (resamples + 1),
        "prompts_improved": int((deltas > 0).sum()),
        "prompts_regressed": int((deltas < 0).sum()),
        "prompts_tied": int((deltas == 0).sum()),
    }


def repeated_ngram_fraction(tokens: list[int], n: int = 4) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[index:index + n]) for index in range(len(tokens) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def terminal_loop(tokens: list[int]) -> bool:
    tail = tokens[-48:]
    for period in range(1, 9):
        if len(tail) >= period * 3 and tail[-period:] * 3 == tail[-period * 3:]:
            return True
    return False


def arm_rows(rows: list[dict], arm: str) -> list[dict]:
    built = []
    for row in rows:
        student = answer_fence_prompt(str(row["problem"]))
        if arm == "question_only":
            prompt = student
        elif arm == "correct_answer":
            prompt = build_teacher_prompt(
                str(row["problem"]),
                str(row["solution"]),
                "final_answer",
                INSTRUCTION_SUFFIX_ANSWER,
            )
        elif arm == "permuted_answer":
            prompt = build_teacher_prompt(
                str(row["problem"]),
                str(row["permuted_solution"]),
                "final_answer",
                INSTRUCTION_SUFFIX_ANSWER,
            )
        else:
            raise ValueError(f"unknown uplift arm {arm!r}")
        built.append(
            {
                "prompt": [{"content": prompt}],
                "reward_model": {
                    "ground_truth": str(row["ground_truth"]),
                    "style": str(row["reward_style"]),
                },
                "extra_info": {"index": str(row["example_id"])},
            }
        )
    return built


def summarize_attempts(
    attempts: list[dict],
    source_rows: list[dict],
    tokenizer,
    stop_ids: tuple[int, ...],
    answer_fence_ids: tuple[int, int],
) -> dict[str, float | int]:
    donor_by_index = {
        index: (
            str(row["permuted_solution"]),
            answer_style(
                {"reward_model": {"style": str(row["reward_style"])}}
            ),
        )
        for index, row in enumerate(source_rows)
    }
    donor_correct = 0
    repetitions = []
    loops = 0
    hashes_by_prompt: dict[int, list[str]] = defaultdict(list)
    for attempt in attempts:
        tokens = [int(token) for token in attempt["emitted_token_ids"]]
        stop_cut = next(
            (index for index, token in enumerate(tokens) if token in stop_ids),
            len(tokens),
        )
        content_tokens = tokens[:stop_cut]
        prompt_index = int(attempt["problem_index"])
        donor_answer, reward_style = donor_by_index[prompt_index]
        correct, _ = verify_terminated_answer(
            tokens,
            donor_answer,
            tokenizer,
            stop_ids,
            reward_style,
            answer_fence_ids=answer_fence_ids,
        )
        donor_correct += int(correct and attempt["structural_format_ok"])
        repetitions.append(repeated_ngram_fraction(content_tokens))
        loops += int(terminal_loop(content_tokens))
        hashes_by_prompt[prompt_index].append(
            hashlib.sha256(bytes(str(tokens), "utf-8")).hexdigest()
        )
        attempt["permuted_answer_correct"] = bool(
            correct and attempt["structural_format_ok"]
        )
        attempt["repeated_4gram_fraction"] = repetitions[-1]
        attempt["terminal_loop"] = terminal_loop(content_tokens)
    total = max(len(attempts), 1)
    all_identical = sum(
        len(set(hashes)) == 1 for hashes in hashes_by_prompt.values()
    ) / max(len(hashes_by_prompt), 1)
    return {
        "permuted_answer_contract_accuracy": donor_correct / total,
        "repeated_4gram_fraction_mean": sum(repetitions) / total,
        "terminal_loop_fraction": loops / total,
        "all_samples_identical_prompt_fraction": all_identical,
        "unique_transcript_fraction": len(
            {item for hashes in hashes_by_prompt.values() for item in hashes}
        ) / total,
    }


def gate_decision(
    arms: dict[str, dict], comparisons: dict[str, dict]
) -> tuple[str, list[str]]:
    reasons = []
    for name in ("correct_vs_question", "correct_vs_permuted"):
        comparison = comparisons[name]
        if comparison["mean_delta"] < 0.01:
            reasons.append(f"{name} delta below +0.010")
        if comparison["bootstrap_ci_low"] <= 0:
            reasons.append(f"{name} bootstrap lower bound not positive")
        if comparison["permutation_p"] >= 0.05:
            reasons.append(f"{name} permutation p is not below 0.05")
    correct = arms["correct_answer"]
    question = arms["question_only"]
    if correct["structural_format_fraction"] < 0.90:
        reasons.append("correct-answer structural rate below 0.90")
    if correct["ended_fraction"] < 0.95:
        reasons.append("correct-answer termination below 0.95")
    if (
        correct["structural_format_fraction"]
        < question["structural_format_fraction"] - 0.02
    ):
        reasons.append("correct-answer structural rate regressed by over 0.02")
    if correct["ended_fraction"] < question["ended_fraction"] - 0.02:
        reasons.append("correct-answer termination regressed by over 0.02")
    if correct["terminal_loop_fraction"] > 0.01:
        reasons.append("correct-answer terminal loops exceed 0.01")
    if correct["repeated_4gram_fraction_mean"] > 0.25:
        reasons.append("correct-answer repeated 4-grams exceed 0.25")
    if (
        correct["repeated_4gram_fraction_mean"]
        > question["repeated_4gram_fraction_mean"] + 0.02
    ):
        reasons.append("correct-answer repetition regressed by over 0.02")
    if (
        correct["all_samples_identical_prompt_fraction"]
        > question["all_samples_identical_prompt_fraction"] + 0.10
    ):
        reasons.append("correct-answer identical groups regressed by over 0.10")
    if correct["all_samples_identical_prompt_fraction"] > 0.10:
        reasons.append("correct-answer identical groups exceed 0.10")
    if correct["unique_transcript_fraction"] < 0.50:
        reasons.append("correct-answer unique transcript fraction below 0.50")
    if not reasons:
        return "pass", []
    return "fail", reasons


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--gate-data",
        default="postraining/data/opsd_dapo17k_contractlast_gate.parquet",
    )
    parser.add_argument(
        "--data-manifest",
        default="postraining/data/opsd_dapo17k_contractlast.manifest.json",
    )
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--think-min-tokens", type=int, default=33)
    parser.add_argument("--batch-trajectories", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rollout-compile", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    if min(
        args.rows,
        args.samples,
        args.max_completion_length,
        args.think_min_tokens,
    ) < 1:
        parser.error(
            "rows, samples, completion length, and think minimum must be positive"
        )
    output = Path("postraining/runs") / args.name / "teacher_uplift"
    if output.exists():
        parser.error(f"refusing to overwrite existing uplift output {output}")

    checkpoint_path = Path(args.checkpoint)
    gate_path = Path(args.gate_data)
    manifest_path = Path(args.data_manifest)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != DAPO_OPSD_DATA_SCHEMA:
        parser.error("data manifest uses a different DAPO schema")
    if manifest.get("split_schema") != DAPO_OPSD_SPLIT_SCHEMA:
        parser.error("data manifest uses a different DAPO split schema")
    if file_sha256(gate_path) != manifest.get("gate_sha256"):
        parser.error("gate parquet does not match its manifest hash")
    if args.rows != manifest.get("gate_rows"):
        parser.error(
            "rows must equal the full manifest gate size so the split-local "
            "permutation control preserves the answer multiset"
        )
    if manifest.get("teacher_prompt_schema") != TEACHER_PROMPT_SCHEMA:
        parser.error("data manifest uses a different teacher prompt schema")
    if manifest.get("opsd_prompt_schema") != OPSD_PROMPT_SCHEMA:
        parser.error("data manifest uses a different OPSD prompt schema")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sft = payload.get("sft") or {}
    require_answer_fence_prompt_schema(
        sft, answer_fence=True, source="OPSD uplift checkpoint"
    )
    if sft.get("traces_sha256") != manifest.get("sft_corpus_sha256"):
        parser.error("checkpoint SFT corpus does not match DAPO gate manifest")
    context_tokens = int(payload.get("train_seq_len", 1024))
    if args.max_completion_length >= context_tokens:
        parser.error("completion length must leave room for the prompt")
    prompt_tokens = context_tokens - args.max_completion_length

    rows = pq.read_table(gate_path).to_pylist()[: args.rows]
    if len(rows) != args.rows:
        parser.error(f"gate contains only {len(rows)} rows")
    tokenizer = load_posttraining_tokenizer(
        payload["architecture"],
        FreshHyperparameters.tokenizer_path,
        think_tokens=True,
        answer_tokens=True,
    )
    device = torch.device("cuda")
    model = load_model(checkpoint_path, device, payload=payload)
    del payload
    wrapper = LatentThoughtModel(model, num_blocks=0).to(device)
    wrapper.eval()
    compiled_step = None
    if args.rollout_compile:
        compiled_step = torch.compile(
            wrapper.step_core,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
            dynamic=True,
        )
    output.mkdir(parents=True)
    think_ids = (tokenizer.think_open_id, tokenizer.think_close_id)
    answer_ids = (tokenizer.answer_open_id, tokenizer.answer_close_id)
    stop_ids = tuple(
        dict.fromkeys(
            token
            for token in (tokenizer.eos_id(), tokenizer.bos_id())
            if token >= 0
        )
    )
    arm_results = {}
    attempts_by_arm = {}
    for arm in ARMS:
        attempts: list[dict] = []
        metrics = evaluate_latent_math(
            wrapper,
            tokenizer,
            arm_rows(rows, arm),
            args.samples,
            args.max_completion_length,
            args.max_completion_length,
            args.samples,
            args.seed,
            device,
            prompt_tokens,
            batch_trajectories=args.batch_trajectories,
            compiled_step_core=compiled_step,
            captured_attempts=attempts,
            capture_problem_count=len(rows),
            capture_samples_per_problem=args.samples,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            pin_emit=True,
            think_fence_ids=think_ids,
            answer_fence_ids=answer_ids,
            min_think_tokens=args.think_min_tokens,
        )
        metrics.update(
            summarize_attempts(attempts, rows, tokenizer, stop_ids, answer_ids)
        )
        arm_results[arm] = metrics
        attempts_by_arm[arm] = attempts

    execution_modes = {
        (bool(metrics["compiled"]), bool(metrics["compile_fallback"]))
        for metrics in arm_results.values()
    }
    if len(execution_modes) != 1 or any(
        fallback for _, fallback in execution_modes
    ):
        raise RuntimeError(
            "uplift arms did not share one stable execution mode; rerun all "
            "arms with --no-rollout-compile"
        )
    attempt_paths = {}
    for arm, attempts in attempts_by_arm.items():
        attempt_path = output / f"{arm}_attempts.json"
        atomic_json(
            {"schema": TEACHER_UPLIFT_SCHEMA, "arm": arm, "attempts": attempts},
            attempt_path,
        )
        attempt_paths[arm] = attempt_path

    comparisons = {
        "correct_vs_question": paired_prompt_stats(
            arm_results["correct_answer"]["contract_prompt_correct_counts"],
            arm_results["question_only"]["contract_prompt_correct_counts"],
            args.samples,
            args.seed + 1,
        ),
        "correct_vs_permuted": paired_prompt_stats(
            arm_results["correct_answer"]["contract_prompt_correct_counts"],
            arm_results["permuted_answer"]["contract_prompt_correct_counts"],
            args.samples,
            args.seed + 2,
        ),
    }
    decision, reasons = gate_decision(arm_results, comparisons)
    result = {
        "schema": TEACHER_UPLIFT_SCHEMA,
        "decision": decision,
        "reasons": reasons,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "gate_data": str(gate_path),
        "gate_sha256": manifest["gate_sha256"],
        "data_manifest_sha256": file_sha256(manifest_path),
        "attempts_sha256": {
            arm: file_sha256(path) for arm, path in attempt_paths.items()
        },
        "teacher_prompt_schema": TEACHER_PROMPT_SCHEMA,
        "opsd_prompt_schema": OPSD_PROMPT_SCHEMA,
        "arms": arm_results,
        "comparisons": comparisons,
        "args": vars(args),
    }
    atomic_json(result, output / "results.json")
    print(json.dumps({"decision": decision, "reasons": reasons}, indent=2))
    raise SystemExit(0 if decision == "pass" else 2)


if __name__ == "__main__":
    main()
