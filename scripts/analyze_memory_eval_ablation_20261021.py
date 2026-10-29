"""Paired per-prompt comparison of the memory read ablation benches.

Reads each bench run's canonical metrics.jsonl (``contract_prompt_correct_counts``
over the same 512 prompts, 16 samples each) and prints, per drill and step:
the arm's bench, its paired difference from the same-code ``none`` replay,
from the drill's own logged bench, and from the token-only reference 14165
at the same step. SE = per-prompt SD / sqrt(prompts), in accuracy units.
Writes the table as JSON to --out.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "postraining" / "runs"
DRILLS = {
    "fresh": "mac_sampled_baseline_memory_fresh_lr_drill_20261021_r2",
    "adjoint": "mac_sampled_baseline_memory_adjoint1_drill_20261021",
}
TOKEN_ONLY = "mac_sampled_baseline_allstates_drill_20261019"  # job 14165
ARMS = ("none", "zero_read", "foreign_state", "foreign_sample")


def bench_record(run: Path, step: int | None = None) -> dict | None:
    path = run / "metrics.jsonl"
    if not path.is_file():
        return None
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if '"bench"' in line
    ]
    records = [r for r in records if r.get("type") == "bench"]
    if step is not None:
        records = [r for r in records if r["step"] == step]
    return records[-1] if records else None


def paired(left: dict, right: dict) -> dict:
    a = left["contract_prompt_correct_counts"]
    b = right["contract_prompt_correct_counts"]
    samples = left["policy_samples"] // len(a)
    if len(a) != len(b) or samples != right["policy_samples"] // len(b):
        raise ValueError("benches are not over the same prompt x sample grid")
    diffs = [(x - y) / samples for x, y in zip(a, b)]
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    return {
        "diff": mean,
        "se": math.sqrt(var / len(diffs)),
        "prompts_changed": sum(1 for d in diffs if d),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    table = []
    for drill, drill_run in DRILLS.items():
        for step in (40, 100, 200, 300):
            logged = bench_record(RUNS / drill_run, step)
            token_only = bench_record(RUNS / TOKEN_ONLY, step)
            replays = {
                arm: bench_record(
                    RUNS / f"memory_eval_ablation_{drill}_s{step:03d}_{arm}_20261021"
                )
                for arm in ARMS
            }
            if not any(replays.values()):
                continue
            for arm, record in replays.items():
                if record is None:
                    continue
                if record.get("memory_eval_ablation", "none") != arm:
                    raise ValueError(f"{drill} {step} {arm}: record says otherwise")
                row = {
                    "drill": drill,
                    "step": step,
                    "arm": arm,
                    "bench": record["contract_accuracy"],
                    "logged": None if logged is None else logged["contract_accuracy"],
                    "token_only": (
                        None if token_only is None else token_only["contract_accuracy"]
                    ),
                    "mean_emitted": record.get("emitted_tokens_mean"),
                    "terminated": record.get("ended_fraction"),
                }
                if replays["none"] is not None and arm != "none":
                    row["vs_none"] = paired(record, replays["none"])
                if logged is not None:
                    row["vs_logged"] = paired(record, logged)
                if token_only is not None:
                    row["vs_token_only"] = paired(record, token_only)
                table.append(row)

    def fmt(entry):
        if entry is None:
            return "-"
        return f"{entry['diff']:+.4f} ± {entry['se']:.4f} ({entry['prompts_changed']})"

    print("| drill | step | arm | bench | vs none | vs logged | vs 14165 | mean tokens | ended |")
    print("|---|---|---|---|---|---|---|---|---|")
    for row in table:
        print(
            f"| {row['drill']} | {row['step']} | {row['arm']} | {row['bench']:.4f} | "
            f"{fmt(row.get('vs_none'))} | {fmt(row.get('vs_logged'))} | "
            f"{fmt(row.get('vs_token_only'))} | {row['mean_emitted']} | {row['terminated']} |"
        )
    if args.out is not None:
        args.out.write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
