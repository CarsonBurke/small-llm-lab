"""Summarize complete gate transcripts by module, refusing partial evidence."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def summarize(path: Path, expected_prompts: int, samples: int) -> dict:
    groups = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("schema") != "frozen_policy_gate_transcript/v1":
            raise ValueError("unsupported transcript schema")
        key = json.dumps(row["prompt"], sort_keys=True)
        groups[key].append(row)
    if len(groups) != expected_prompts or any(
        len(rows) != samples or {row["sample"] for row in rows} != set(range(samples))
        for rows in groups.values()
    ):
        raise ValueError("incomplete or repeated prompt/sample coverage")
    modules = defaultdict(list)
    for rows in groups.values():
        info = rows[0].get("extra_info") or {}
        module = str(info.get("module", rows[0].get("source", "math")))
        modules[module].append(rows)
    result = {}
    for module, prompt_groups in sorted(modules.items()):
        rows = [row for group in prompt_groups for row in group]
        entry = {
            "prompts": len(prompt_groups), "samples": len(rows),
            "contract_accuracy": sum(row["reward"] == 1.0 for row in rows) / len(rows),
            "format_fraction": sum(row["structural_format_ok"] for row in rows) / len(rows),
            "terminated_fraction": sum(row["terminated"] for row in rows) / len(rows),
            "mixed_prompt_fraction": sum(
                0 < sum(row["reward"] == 1.0 for row in group) < samples
                for group in prompt_groups
            ) / len(prompt_groups),
            "all_wrong_prompt_fraction": sum(
                not any(row["reward"] == 1.0 for row in group)
                for group in prompt_groups
            ) / len(prompt_groups),
            "all_correct_prompt_fraction": sum(
                all(row["reward"] == 1.0 for row in group)
                for group in prompt_groups
            ) / len(prompt_groups),
            "mean_binary_within_group_std": sum(
                math.sqrt((p := sum(row["reward"] == 1.0 for row in group) / samples) * (1 - p))
                for group in prompt_groups
            ) / len(prompt_groups),
        }
        entry["any_correct_prompt_fraction"] = 1 - entry["all_wrong_prompt_fraction"]
        # Prompt-level rates keep the 16 correlated samples from looking
        # like 16 independent questions in uncertainty estimates.
        rates = [sum(row["reward"] == 1.0 for row in group) / samples for group in prompt_groups]
        entry["contract_accuracy_prompt_standard_error"] = (
            math.sqrt(sum((rate - entry["contract_accuracy"]) ** 2 for rate in rates)
                      / (len(rates) * (len(rates) - 1))) if len(rates) > 1 else None
        )
        if "choice_" in module:
            entry["uniform_guess_accuracy"] = sum(
                row["extra_info"]["chance"] for row in rows
            ) / len(rows)
            # The scorer preserves the relaxed parsed field on a terminated
            # format failure. Unterminated responses remain incorrect.
            entry["terminated_parsed_letter_accuracy"] = sum(
                row["terminated"] and row["parsed_answer"] == row["reward_model"]["ground_truth"]
                for row in rows
            ) / len(rows)
        result[module] = entry
    return {"complete": True, "modules": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompts", type=int, default=512)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    results = {}
    for run in args.runs:
        try:
            results[str(run)] = summarize(run / "gate_transcripts.jsonl", args.prompts, args.samples)
        except (OSError, ValueError, KeyError) as error:
            results[str(run)] = {"complete": False, "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump({"schema": "posttraining_readiness/v1", "runs": results}, stream, indent=2)
        stream.write("\n")
    print(json.dumps(results, indent=2))
    if not all(result["complete"] for result in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
