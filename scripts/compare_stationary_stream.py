"""Matched comparison of stationary-buffer streaming arms against CE recursion.

Reads completed ``ablation_results/<run>/`` artifacts (CPU only, no model
execution) and writes ``comparison.json`` into the candidate's directory.
Same-line references must match the candidate in everything but the read
knobs and the shared backbone and data sources; a reference that predates
a read knob is compared at that knob's default, and hash differences in
the stationary package itself are reported, not refused. The control is a
``future_credit_stream`` CE-recursion run from the sibling package: it
cannot share source hashes or the read knobs, so it is matched on data and
tokenizer provenance, budget, and every training field the two
configurations share, and the remaining differences are reported.
The proxy BPB is the fixed partial-document panel, not full challenge
validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "ablation_results"
READ_KNOBS = {"run_id", "buffer_slots", "read_heads", "read_key_dim", "read_entry", "read_recency_slope"}
# Knobs added after earlier arms completed: an arm that lacks one ran at its default.
KNOB_DEFAULTS = {"read_recency_slope": 1.0}
DIAGNOSTICS = ("null_mass", "read_age", "read_rms")
CONTROL_SOURCES = (
    "pretraining/future_credit_stream/data.py",
    "pretraining/nanogpt_mini/nanogpt_mini_model.py",
    "pretraining/nextlat.py",
    "train_gpt.py",
)


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
    final = train[-1]
    return {
        "run": run_id,
        "architecture": config["architecture"],
        "objective": config["config"].get("objective", "ce"),
        "config": config["config"],
        "sources": config["sources"],
        "provenance": {key: config[key] for key in ("tokenizer_sha256", "train_fingerprint", "val_fingerprint")},
        "parameters": config["parameters"],
        "read_parameters": config.get("read_parameters", 0),
        "steps": result["steps"],
        "tokens_seen": result["tokens_seen"],
        "proxy_bpb": result["validation"]["val_bpb"],
        "reset_latent_bpb": result["validation"]["val_bpb_reset_latent"],
        "memory_gain_bpb": result["validation"]["val_memory_gain_bpb"],
        "training_seconds": result["training_seconds"],
        "median_update_ms_second_half": statistics.median(entry["step_avg_ms"] for entry in tail),
        "final_diagnostics": {key: final[key] for key in DIAGNOSTICS if key in final},
    }


def _differences(candidate: dict, other: dict, defaults: dict | None = None) -> dict:
    defaults = defaults or {}
    ours, theirs = ({**defaults, **arm["config"]} for arm in (candidate, other))
    return {key: [ours.get(key), theirs.get(key)] for key in set(ours) | set(theirs) if ours.get(key) != theirs.get(key)}


def check_budget_and_provenance(candidate: dict, other: dict) -> None:
    if candidate["provenance"] != other["provenance"]:
        raise SystemExit(f"{other['run']}: data/tokenizer provenance differs from {candidate['run']}")
    if candidate["steps"] != other["steps"] or candidate["tokens_seen"] != other["tokens_seen"]:
        raise SystemExit(f"{other['run']}: step or token budget differs from {candidate['run']}")


def check_reference(candidate: dict, other: dict) -> dict:
    """A same-line arm: everything but the read knobs must match.

    Returns the knob differences plus, under ``sources``, the stationary
    package files whose hashes differ (a later arm may carry a knob the
    earlier one predates); the shared backbone and data sources must match.
    """
    check_budget_and_provenance(candidate, other)
    differences = _differences(candidate, other, KNOB_DEFAULTS)
    unexpected = sorted(set(differences) - READ_KNOBS)
    if unexpected:
        raise SystemExit(f"{other['run']}: configuration differs from {candidate['run']} in {unexpected}")
    changed = sorted(key for key in CONTROL_SOURCES if candidate["sources"][key] != other["sources"][key])
    if changed:
        raise SystemExit(f"{other['run']}: shared backbone or data sources differ from {candidate['run']}: {changed}")
    drift = {key: [candidate["sources"].get(key), other["sources"].get(key)]
             for key in set(candidate["sources"]) | set(other["sources"])
             if candidate["sources"].get(key) != other["sources"].get(key)}
    if drift:
        differences["sources"] = drift
    return differences


def check_control(candidate: dict, control: dict) -> dict:
    """The sibling-line CE recursion run: shared training fields must match."""
    check_budget_and_provenance(candidate, control)
    if control["objective"] != "ce":
        raise SystemExit(f"{control['run']}: the control must be a CE recursion run")
    shared = set(candidate["config"]) & set(control["config"]) - {"run_id"}
    mismatched = sorted(key for key in shared if candidate["config"][key] != control["config"][key])
    if mismatched:
        raise SystemExit(f"{control['run']}: shared training fields differ from {candidate['run']}: {mismatched}")
    # Only the files that define the shared data contract and backbone must
    # match; the sibling trainer and package files legitimately diverge.
    changed = sorted(key for key in CONTROL_SOURCES if candidate["sources"][key] != control["sources"][key])
    if changed:
        raise SystemExit(f"{control['run']}: shared backbone or data sources differ from {candidate['run']}: {changed}")
    return _differences(candidate, control)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="stationary-buffer run id")
    parser.add_argument("--control", required=True, help="future_credit_stream CE recursion run id")
    parser.add_argument("--reference", action="append", default=[],
                        help="additional stationary-buffer arms (other slot counts or read shapes)")
    parser.add_argument("--threshold", type=float, default=0.005,
                        help="proxy BPB improvement required for promotion")
    args = parser.parse_args()

    candidate = load_run(args.candidate)
    control = load_run(args.control)
    references = [load_run(run) for run in args.reference]
    if candidate["architecture"] != "streaming_ffn_stationary_buffer_v1":
        raise SystemExit("candidate must be a stationary-buffer run")
    differences = {control["run"]: check_control(candidate, control)}
    for arm in references:
        differences[arm["run"]] = check_reference(candidate, arm)

    improvement = control["proxy_bpb"] - candidate["proxy_bpb"]
    summary = {
        "scope": "one-seed 2,000-update partial-document proxy, not full challenge BPB",
        "candidate": candidate["run"],
        "control": control["run"],
        "improvement_bpb": improvement,
        "memory_gain_delta_bpb": candidate["memory_gain_bpb"] - control["memory_gain_bpb"],
        "candidate_over_control_update_time": candidate["median_update_ms_second_half"] / control["median_update_ms_second_half"],
        "candidate_over_control_parameters": candidate["parameters"] / control["parameters"],
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
