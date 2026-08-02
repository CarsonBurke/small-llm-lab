"""Paired causal comparison of baseline, correct, and permuted OPSD arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from postraining.opsd.teacher_uplift import atomic_json, paired_prompt_stats


OPSD_POLICY_COMPARISON_SCHEMA = "opsd_dapo_policy_comparison/v1"


def classify(
    arms: dict[str, dict], comparisons: dict[str, dict]
) -> tuple[str, list[str]]:
    reasons = []
    for name in ("correct_vs_baseline", "correct_vs_permuted"):
        comparison = comparisons[name]
        if comparison["mean_delta"] < 0.01:
            reasons.append(f"{name} delta below +0.010")
        if comparison["bootstrap_ci_low"] <= 0:
            reasons.append(f"{name} bootstrap lower bound not positive")
        if comparison["permutation_p"] >= 0.05:
            reasons.append(f"{name} permutation p is not below 0.05")
    correct = arms["correct"]
    baseline = arms["baseline"]
    if (
        correct["structural_format_fraction"]
        < baseline["structural_format_fraction"] - 0.02
    ):
        reasons.append("correct OPSD structural rate regressed by over 0.02")
    if correct["ended_fraction"] < baseline["ended_fraction"] - 0.02:
        reasons.append("correct OPSD termination regressed by over 0.02")
    if correct["terminal_loop_fraction"] > 0.01:
        reasons.append("correct OPSD terminal loops exceed 0.01")
    if not reasons:
        return "beneficial", []
    versus_baseline = comparisons["correct_vs_baseline"]
    if (
        versus_baseline["mean_delta"] <= -0.01
        and versus_baseline["bootstrap_ci_high"] < 0
        and versus_baseline["permutation_p"] < 0.05
    ):
        return "harmful", reasons
    return "neutral_or_inconclusive", reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--correct", required=True)
    parser.add_argument("--permuted", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path("postraining/runs") / args.name
    if output.exists():
        parser.error(f"refusing to overwrite comparison output {output}")

    baseline_result = json.loads(Path(args.baseline).read_text())
    correct_result = json.loads(Path(args.correct).read_text())
    permuted_result = json.loads(Path(args.permuted).read_text())
    arms = {
        "baseline": baseline_result["arms"]["question_only"],
        "correct": correct_result["metrics"],
        "permuted": permuted_result["metrics"],
    }
    samples = int(baseline_result["args"]["samples"])
    for label, result in (
        ("correct", correct_result),
        ("permuted", permuted_result),
    ):
        if int(result["args"]["samples"]) != samples:
            parser.error(f"{label} samples do not match the baseline")
        if result["gate_sha256"] != baseline_result["gate_sha256"]:
            parser.error(f"{label} gate bytes do not match the baseline")
        for field in (
            "rows",
            "max_completion_length",
            "temperature",
            "top_p",
            "top_k",
            "think_min_tokens",
            "seed",
        ):
            if result["args"][field] != baseline_result["args"][field]:
                parser.error(f"{label} evaluation differs on {field}")

    def compare(left: str, right: str, offset: int) -> dict[str, float | int]:
        return paired_prompt_stats(
            arms[left]["contract_prompt_correct_counts"],
            arms[right]["contract_prompt_correct_counts"],
            samples,
            args.seed + offset,
        )

    comparisons = {
        "correct_vs_baseline": compare("correct", "baseline", 1),
        "correct_vs_permuted": compare("correct", "permuted", 2),
        "permuted_vs_baseline": compare("permuted", "baseline", 3),
    }
    decision, reasons = classify(arms, comparisons)
    result = {
        "schema": OPSD_POLICY_COMPARISON_SCHEMA,
        "decision": decision,
        "reasons": reasons,
        "arms": arms,
        "comparisons": comparisons,
        "inputs": {
            "baseline": args.baseline,
            "correct": args.correct,
            "permuted": args.permuted,
        },
        "args": vars(args),
    }
    output.mkdir(parents=True)
    atomic_json(result, output / "results.json")
    print(json.dumps({"decision": decision, "reasons": reasons}, indent=2))


if __name__ == "__main__":
    main()
