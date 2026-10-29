"""Answer-level DG against its RLOO control, paired by prompt and seed.

Reads each run's canonical ``metrics.jsonl`` bench records (512 held-out
prompts x 16 samples) and, if present, a greedy (T = 0) bench-only run per
(run, step) (``scripts/submit_greedy_bench_20261021.sh``). For each pair of
runs (DG, RLOO) sharing a seed, at one bench step, per prompt:

- ``kt``: log((k + 1/2) / (n + 1)), the Krichevsky-Trofimov estimate of
  log p from k contract-correct samples of n; its sum over prompts is the
  cross-entropy oracle's objective sum_n log p_n that DG's Prop. 2 says the
  gate reallocates toward;
- ``majority``: maj@n, a vote tie scored as its random break;
- ``pass``: pass@n, k > 0;
- ``accuracy``: k / n, the T = 1 mean accuracy, PG's own objective (the
  declared cost);
- ``entropy``: the plug-in entropy of the prompt's answer clusters;
- ``greedy``: the greedy bench's correctness.

T = 1 metrics are read at the last bench step every run logged; greedy
accuracy at the last snapshot step every run saved (greedy benches read
snapshots, which are sparser than benches). Greedy is a pre-registered
primary metric, so a missing greedy result for any run at that step is an
error unless ``--allow-missing-greedy`` marks the verdict provisional.

Differences DG - RLOO are averaged over seeds per prompt; the mean and its
standard error are over prompts. That SE holds the policies fixed: it
measures bench sampling and prompt variation, not training-seed variation,
which two seeds cannot estimate. The pre-registered rule (NOTES.md,
"Answer-level DG: pre-registration") therefore also requires every seed's
difference to share the sign. Harm is checked before win: a run that
clears the win bar on the hard panel while significantly harming another
primary metric is harm. Panels: ``full`` (every bench prompt) and
``hard`` (``scripts/build_hard_bench_panel.py``'s manifest).

    python3 scripts/analyze_answer_gate_drill.py \\
        --pair <dg run s1337> <rloo run s1337> --pair <dg run s2026> <rloo run s2026> \\
        --hard-panel postraining/data/drill_rl_study_heldout_hard_s000_20261021_v1.manifest.json \\
        --output <out>.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

GREEDY_TEMPLATE = "postraining/runs/greedy_bench_{run}_s{step:03d}_20261021"
METRICS = ("kt", "majority", "pass", "accuracy", "entropy", "greedy")
# Pre-registered thresholds (NOTES.md). Differences are DG - RLOO.
SIGNIFICANCE_SE = 2.0
HARD_KT_MIN_EFFECT = 0.10  # nats per prompt


def bench_records(run: Path) -> dict[int, dict]:
    records = {}
    for line in (run / "metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record.get("type") == "bench":
            # A resumed session re-benches nothing it already logged, but a
            # restart from an older checkpoint can: the last record wins.
            records[int(record["step"])] = record
    return records


def per_prompt(record: dict) -> dict[str, list[float]]:
    counts = record["contract_prompt_correct_counts"]
    samples = record["samples"] // record["prompt_groups"]
    values = {
        "kt": [math.log((k + 0.5) / (samples + 1)) for k in counts],
        "pass": [float(k > 0) for k in counts],
        "accuracy": [k / samples for k in counts],
    }
    if "contract_prompt_majority_scores" in record:
        values["majority"] = list(record["contract_prompt_majority_scores"])
        values["entropy"] = list(record["prompt_answer_cluster_entropy"])
    return values


def snapshot_steps(run: Path) -> set[int]:
    return {int(path.stem.removeprefix("step_"))
            for path in (run / "snapshots").glob("step_*.pt")}


def greedy_values(run: Path, step: int, template: str) -> list[float] | None:
    path = Path(template.format(run=run.name, step=step))
    if not (path / "metrics.jsonl").exists():
        return None
    records = bench_records(path)
    if len(records) != 1:
        raise SystemExit(f"{path}: expected one greedy bench record")
    record = next(iter(records.values()))
    if record["samples"] != record["prompt_groups"]:
        raise SystemExit(f"{path}: a greedy bench draws one sample per prompt")
    return [float(k) for k in record["contract_prompt_correct_counts"]]


def summarize(differences: list[list[float]]) -> dict[str, float]:
    """Mean and SE over prompts of the seed-averaged difference, and each
    seed's own mean."""
    prompts = len(differences[0])
    averaged = [sum(seed[i] for seed in differences) / len(differences) for i in range(prompts)]
    mean = sum(averaged) / prompts
    variance = sum((value - mean) ** 2 for value in averaged) / max(prompts - 1, 1)
    return {"mean": mean, "se": math.sqrt(variance / prompts), "prompts": prompts,
            "seed_means": [sum(seed) / prompts for seed in differences]}


def verdict(primary: dict[str, dict]) -> str:
    """The pre-registered decision on the hard-panel KT score, the full-panel
    maj@n and the full-panel greedy accuracy (absent only under
    ``--allow-missing-greedy``). Harm takes precedence over win."""
    def significant(stat, sign):
        return (sign * stat["mean"] > SIGNIFICANCE_SE * stat["se"]
                and all(sign * value > 0 for value in stat["seed_means"]))

    if any(significant(stat, -1) for stat in primary.values() if stat):
        return "harm"
    kt = primary["hard_kt"]
    if significant(kt, +1) and kt["mean"] >= HARD_KT_MIN_EFFECT:
        return "win"
    return "null"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pair", nargs=2, type=Path, action="append", required=True,
                        metavar=("DG_RUN", "RLOO_RUN"), help="one per seed")
    parser.add_argument("--hard-panel", type=Path, required=True)
    parser.add_argument("--step", type=int, default=None,
                        help="bench step (default: the last every run logged)")
    parser.add_argument("--greedy-step", type=int, default=None,
                        help="snapshot step for greedy accuracy (default: the last "
                        "every run saved)")
    parser.add_argument("--greedy-template", default=GREEDY_TEMPLATE)
    parser.add_argument("--allow-missing-greedy", action="store_true",
                        help="score without greedy accuracy; the verdict is then "
                        "marked provisional")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    runs = [run for pair in args.pair for run in pair]
    records = {run: bench_records(run) for run in runs}
    common = set.intersection(*(set(steps) for steps in records.values()))
    if not common:
        raise SystemExit("the runs share no bench step")
    step = max(common) if args.step is None else args.step
    if step not in common:
        raise SystemExit(f"step {step} is not benched by every run; common: {sorted(common)}")
    snapshots = set.intersection(*(snapshot_steps(run) for run in runs))
    greedy_step = (max(snapshots, default=None) if args.greedy_step is None
                   else args.greedy_step)
    if greedy_step is not None and greedy_step not in snapshots:
        raise SystemExit(f"step {greedy_step} is not a snapshot of every run; "
                         f"common: {sorted(snapshots)}")
    manifest = json.loads(args.hard_panel.read_text())
    hard = sorted(row["problem_index"] for row in manifest["rows"])
    values = {}
    for run in runs:
        values[run] = per_prompt(records[run][step])
        if len(values[run]["kt"]) != manifest["bench_problems"]:
            raise SystemExit(f"{run}: the bench panel is not the hard panel's")
    if greedy_step is None and not args.allow_missing_greedy:
        raise SystemExit("the runs share no snapshot step to score greedy accuracy at")
    greedy = {run: greedy_values(run, greedy_step, args.greedy_template)
              for run in runs} if greedy_step is not None else {}
    missing = [str(run) for run in runs if greedy.get(run) is None]
    if missing and not args.allow_missing_greedy:
        raise SystemExit(
            f"greedy accuracy is pre-registered but missing at snapshot step "
            f"{greedy_step} for: {', '.join(missing)} "
            "(scripts/submit_greedy_bench_20261021.sh RUN STEP); pass "
            "--allow-missing-greedy for a provisional verdict")
    if not missing:
        for run in runs:
            if len(greedy[run]) != manifest["bench_problems"]:
                raise SystemExit(f"{run}: the greedy bench is not the full bench panel")
            values[run]["greedy"] = greedy[run]
    result = {"step": step, "greedy_step": None if missing else greedy_step,
              "hard_prompts": len(hard), "pairs": [[str(a), str(b)] for a, b in args.pair],
              "arms": {}, "differences": {}}
    panels = {"full": list(range(manifest["bench_problems"])), "hard": hard}
    for arm, index in (("dg", 0), ("rloo", 1)):
        arm_runs = [pair[index] for pair in args.pair]
        result["arms"][arm] = {
            f"{panel}_{metric}": [sum(values[run][metric][i] for i in prompts) / len(prompts)
                                  for run in arm_runs]
            for panel, prompts in panels.items() for metric in METRICS
            if all(metric in values[run] for run in arm_runs)
        }
    for panel, prompts in panels.items():
        for metric in METRICS:
            if not all(metric in values[run] for run in runs):
                continue
            result["differences"][f"{panel}_{metric}"] = summarize([
                [values[dg][metric][i] - values[rloo][metric][i] for i in prompts]
                for dg, rloo in args.pair])
    differences = result["differences"]
    if "full_majority" not in differences:
        raise SystemExit("maj@n is a pre-registered primary metric but a run's bench "
                         "records lack contract_prompt_majority_scores")
    result["primary"] = {
        "hard_kt": differences["hard_kt"],
        "full_majority": differences["full_majority"],
        "full_greedy": differences.get("full_greedy"),
    }
    result["verdict"] = verdict(result["primary"])
    if missing:
        result["verdict"] = f"provisional_{result['verdict']}_without_greedy"
    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n")
    for name, stat in differences.items():
        print(f"{name:>16}: {stat['mean']:+.4f} +/- {stat['se']:.4f}"
              f"  seeds {', '.join(f'{v:+.4f}' for v in stat['seed_means'])}")
    print(f"step {step} (greedy step {result['greedy_step']}): verdict {result['verdict']}")


if __name__ == "__main__":
    main()
