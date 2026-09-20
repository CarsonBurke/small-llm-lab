"""Matched comparison of streaming future-bag-carry ablation arms.

Reads completed ``ablation_results/<run>/`` artifacts (CPU only, no model
execution), verifies that the arms share sources, data, tokenizer, and every
configuration field other than the objective knobs, and writes
``comparison.json`` into the candidate's directory. The proxy BPB is the fixed
partial-document panel, not full challenge validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "ablation_results"
OBJECTIVE_KNOBS = {"objective", "run_id", "backbone_future_weight", "discount", "horizon"}
DIAGNOSTICS = ("bag_loss", "hidden_bag_loss", "bag_entropy", "gate_mean", "carry_cosine")


def load_run(run_id: str) -> dict:
    directory = RESULTS / run_id
    result = json.loads((directory / "result.json").read_text())
    if result.get("status") != "completed":
        raise SystemExit(f"{run_id}: status {result.get('status')!r} is not completed")
    config = json.loads((directory / "run_config.json").read_text())
    entries = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines() if line]
    train = [entry for entry in entries if entry.get("type") == "train"]
    if not train:
        raise SystemExit(f"{run_id}: no training entries in metrics.jsonl")
    # Median update time over the second half of training excludes warmup
    # and compile effects; it is a timing summary, not a benchmark.
    tail = train[len(train) // 2:]
    update_ms = statistics.median(entry["step_avg_ms"] for entry in tail)
    final = train[-1]
    return {
        "run": run_id,
        "objective": config["config"]["objective"],
        "config": config["config"],
        "sources": config["sources"],
        "provenance": {key: config[key] for key in ("tokenizer_sha256", "train_fingerprint", "val_fingerprint")},
        "parameters": config["parameters"],
        "steps": result["steps"],
        "tokens_seen": result["tokens_seen"],
        "proxy_bpb": result["validation"]["val_bpb"],
        "reset_latent_bpb": result["validation"]["val_bpb_reset_latent"],
        "memory_gain_bpb": result["validation"]["val_memory_gain_bpb"],
        "training_seconds": result["training_seconds"],
        "median_update_ms_second_half": update_ms,
        "final_diagnostics": {key: final[key] for key in DIAGNOSTICS if key in final},
    }


def check_matched(candidate: dict, other: dict) -> dict:
    differences = {
        key: [candidate["config"].get(key), other["config"].get(key)]
        for key in set(candidate["config"]) | set(other["config"])
        if candidate["config"].get(key) != other["config"].get(key)
    }
    unexpected = sorted(set(differences) - OBJECTIVE_KNOBS)
    if unexpected:
        raise SystemExit(f"{other['run']}: configuration differs from {candidate['run']} in {unexpected}")
    if candidate["sources"] != other["sources"]:
        raise SystemExit(f"{other['run']}: source hashes differ from {candidate['run']}")
    if candidate["provenance"] != other["provenance"]:
        raise SystemExit(f"{other['run']}: data/tokenizer provenance differs from {candidate['run']}")
    if candidate["steps"] != other["steps"] or candidate["tokens_seen"] != other["tokens_seen"]:
        raise SystemExit(f"{other['run']}: step or token budget differs from {candidate['run']}")
    return differences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="future_bag run id")
    parser.add_argument("--control", required=True, help="matched ce run id")
    parser.add_argument("--reference", action="append", default=[],
                        help="additional matched arms (tbptt, other discounts or weights)")
    parser.add_argument("--threshold", type=float, default=0.005,
                        help="proxy BPB improvement required for promotion")
    args = parser.parse_args()

    candidate = load_run(args.candidate)
    control = load_run(args.control)
    references = [load_run(run) for run in args.reference]
    if candidate["objective"] != "future_bag" or control["objective"] != "ce":
        raise SystemExit("candidate must be a future_bag run and control a ce run")
    differences = {control["run"]: check_matched(candidate, control)}
    for arm in references:
        differences[arm["run"]] = check_matched(candidate, arm)

    improvement = control["proxy_bpb"] - candidate["proxy_bpb"]
    summary = {
        "scope": "one-seed 2,000-update partial-document proxy, not full challenge BPB",
        "candidate": candidate["run"],
        "control": control["run"],
        "improvement_bpb": improvement,
        "memory_gain_delta_bpb": candidate["memory_gain_bpb"] - control["memory_gain_bpb"],
        "candidate_over_control_update_time": candidate["median_update_ms_second_half"] / control["median_update_ms_second_half"],
        "promotion_threshold_bpb": args.threshold,
        "meets_proxy_promotion_gate": improvement > args.threshold,
        "config_differences": differences,
        "arms": {arm["run"]: {key: value for key, value in arm.items() if key not in ("config", "sources", "provenance")}
                 for arm in [candidate, control, *references]},
    }
    output = RESULTS / candidate["run"] / "comparison.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
