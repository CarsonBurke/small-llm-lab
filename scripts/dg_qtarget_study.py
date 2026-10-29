#!/usr/bin/env python3
"""Read canonical DG study metrics without importing or executing a model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DIAGNOSTICS = {
    "gate_mean": "delightful/gate_mean",
    "positive_gate_mean": "delightful/positive_gate_mean",
    "negative_gate_mean": "delightful/negative_gate_mean",
    "entropy": "entropy/policy_mean",
    "surprisal": "delightful/surprisal_mean",
    "advantage_std": "advantage/std",
    "q_mc_ev": "value/mc_explained_variance",
}


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def read_metrics(path: Path):
    """Keep complete records; expose corruption without failing an active writer."""
    if not path.exists():
        return [], ["metrics.jsonl is not available yet"]
    rows, warnings = [], []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("record is not an object")
            except (ValueError, json.JSONDecodeError) as error:
                warnings.append(f"line {line_number} skipped (possibly an incomplete write): {error}")
                continue
            rows.append(row)
    return rows, warnings


def weighted(groups, key, weight):
    pairs = [(number(g.get(key)), number(g.get(weight))) for g in groups]
    if not pairs or any(v is None or w is None or w < 0 for v, w in pairs):
        return None
    denominator = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / denominator if denominator else None


def pooled_mc_ev(groups):
    target_mean = weighted(groups, "mc_target_mean", "action_count")
    residual_mean = weighted(groups, "mc_residual_mean", "action_count")
    if target_mean is None or residual_mean is None:
        return None
    moments = []
    for g in groups:
        values = [number(g.get(k)) for k in ("mc_target_variance", "mc_target_mean", "mc_residual_variance", "mc_residual_mean", "action_count")]
        if any(v is None for v in values):
            return None
        tv, tm, rv, rm, n = values
        moments.append((n, tv + tm * tm, rv + rm * rm))
    count = sum(n for n, _, _ in moments)
    if not count:
        return None
    tv = sum(n * t for n, t, _ in moments) / count - target_mean**2
    rv = sum(n * r for n, _, r in moments) / count - residual_mean**2
    return 1 - rv / tv if tv > 1e-12 else 0.0


def train_diagnostics(row):
    dashboard = row.get("dashboard", {})
    groups = row.get("group_metrics", [])
    result = {name: number(dashboard.get(key)) for name, key in DIAGNOSTICS.items()}
    fallbacks = {
        "gate_mean": weighted(groups, "delightful_gate_mean", "emit_action_count"),
        "positive_gate_mean": weighted(groups, "delightful_positive_gate_mean", "delightful_positive_count"),
        "negative_gate_mean": weighted(groups, "delightful_negative_gate_mean", "delightful_negative_count"),
        "surprisal": weighted(groups, "delightful_surprisal_mean", "emit_action_count"),
        "q_mc_ev": pooled_mc_ev(groups),
    }
    if groups and all(number(g.get("policy_entropy_count")) is not None and number(g.get("policy_entropy_sum")) is not None for g in groups):
        count = sum(g["policy_entropy_count"] for g in groups)
        fallbacks["entropy"] = sum(g["policy_entropy_sum"] for g in groups) / count if count else None
    mean = weighted(groups, "advantage_mean", "action_count")
    if mean is not None and all(number(g.get("advantage_std")) is not None for g in groups):
        count = sum(g["action_count"] for g in groups)
        if count:
            second = sum(g["action_count"] * (g["advantage_std"]**2 + g["advantage_mean"]**2) for g in groups) / count
            fallbacks["advantage_std"] = math.sqrt(max(0.0, second - mean**2))
    for key, value in fallbacks.items():
        if result[key] is None:
            result[key] = value
    return {k: v for k, v in result.items() if v is not None}


def arm_summary(arm, study_path):
    path = Path(arm["output"])
    if not path.is_absolute():
        path = study_path.parent / path
    rows, warnings = read_metrics(path / "metrics.jsonl")
    metrics_sources = [str(path / "metrics.jsonl")]
    for kind in ("bench", "aime", "bpb"):
        final_path = path / "final_eval" / kind / "metrics.jsonl"
        if final_path.exists():
            final_rows, final_warnings = read_metrics(final_path)
            # Evaluation artifacts cannot advance optimizer-update counters.
            rows.extend(row for row in final_rows if row.get("type") == kind)
            warnings.extend(f"{final_path}: {warning}" for warning in final_warnings)
            metrics_sources.append(str(final_path))
    series = {k: {} for k in ("bench", "aime", "bench_contract", "aime_contract", "bpb", "reward", "rollout_sources", "train")}
    warmup_steps, joint_steps = set(), set()
    warmup_diagnostics, evaluation_context = {}, {k: {} for k in ("bench", "aime", "bpb")}
    for row in rows:
        step = row.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            continue
        kind = row.get("type")
        if kind == "value_warmup":
            warmup_steps.add(step)
            warmup_diagnostics[step] = {k: v for k in ("mc_explained_variance", "value_loss", "value_mean") if (v := number(row.get(k))) is not None}
        elif kind == "train":
            joint_steps.add(step)
            series["train"][step] = train_diagnostics(row)
        elif kind == "rollout":
            value = number(row.get("reward_mean"))
            if value is not None:
                series["reward"][step] = value
            sources = row.get("source_metrics", {})
            series["rollout_sources"][step] = {
                name: {k: v for k in ("reward_mean", "exact_accuracy", "think_format_fraction", "think_gate_zeroed_correct_fraction", "ended_fraction", "actions_per_trajectory", "think_tokens_mean", "length_reward_covariance") if (v := number(source.get(k))) is not None}
                for name, source in sources.items()
            }
        elif kind in ("bench", "aime", "bpb"):
            value = number(row.get("val_bpb" if kind == "bpb" else "accuracy"))
            if value is not None:
                # Exact resume may replace a row at the same step; last wins.
                series[kind][step] = value
                evaluation_context[kind][step] = {
                    k: row[k] for k in ("evaluation_metric_schema", "sampling_schema", "temperature", "top_p", "samples", "prompt_groups", "dataset_modal_answer_accuracy", "subset_modal_answer_accuracy", "contract_accuracy", "structural_format_fraction", "ended_fraction", "emitted_tokens_mean", "emitted_tokens_p95", "module_accuracy", "prompt_correct_counts", "contract_prompt_correct_counts")
                    if k in row
                }
            if kind in ("bench", "aime"):
                contract = number(row.get("contract_accuracy"))
                if contract is not None:
                    series[f"{kind}_contract"][step] = contract
    warmup, joint = max(warmup_steps, default=0), max(joint_steps, default=0)
    latest_source_step = max(series["rollout_sources"], default=None)
    return {
        "output": str(path), "q_critic_lambda": arm.get("q_critic_lambda"),
        "observed_warmup_updates": warmup, "observed_joint_updates": joint,
        "observed_total_updates": warmup + joint,
        "observed_warmup_rows": len(warmup_steps), "observed_joint_rows": len(joint_steps),
        "warnings": warnings, "series": series,
        "metrics_sources": metrics_sources,
        "warmup_diagnostics": warmup_diagnostics, "evaluation_context": evaluation_context,
        "latest_rollout_sources": None if latest_source_step is None else {
            "joint_step": latest_source_step, "sources": series["rollout_sources"][latest_source_step],
        },
    }


def build_report(study_path: Path):
    study = json.loads(study_path.read_text())
    arms = {name: arm_summary(arm, study_path) for name, arm in study["arms"].items()}
    matched = {}
    for metric in ("bench", "aime", "bench_contract", "aime_contract", "bpb", "reward", "rollout_sources", "train"):
        sets = [set(arm["series"][metric]) for arm in arms.values()]
        steps = sorted(set.intersection(*sets)) if sets else []
        matched[metric] = [
            {"joint_step": step, "total_updates": 0 if step == 0 else study["critic_warmup_updates"] + step,
             "arms": {name: arm["series"][metric][step] for name, arm in arms.items()}}
            for step in steps
        ]
    final_step = study["joint_updates"]
    training_complete = {
        name: arm["observed_warmup_updates"] == study["critic_warmup_updates"] and arm["observed_joint_updates"] == final_step
        for name, arm in arms.items()
    }
    final_metrics = {
        metric: any(row["joint_step"] == final_step for row in matched[metric])
        for metric in ("bench", "aime", "bench_contract", "aime_contract", "bpb")
    }
    return {
        "schema": "dg_qtarget_study_analysis/v1", "study": str(study_path),
        "planned_budget": {"warmup": study["critic_warmup_updates"], "joint": study["joint_updates"], "total": study["total_updates"]},
        "arms": arms, "matched_steps": matched,
        "final_budget_comparison": {
            "joint_step": final_step, "total_updates": study["total_updates"],
            "training_budget_reached": training_complete, "matched_evaluations_available": final_metrics,
            "ready": bool(arms) and all(training_complete.values()) and all(final_metrics.values()),
        },
        "interpretation": "Gate separation, entropy and Q explained variance are diagnostics, not evidence of a gain or outcome-linked credit. Positive/negative gates condition on the critic's advantage sign, not verifier success. Compare held-out bench/AIME contract scores, raw accuracy and BPB at identical joint steps; inspect modal-answer, format and reward confounds. Missing rows are not zero scores. Observed update counts are metrics-based, not checkpoint or queue status.",
    }


def render(report):
    budget = report["planned_budget"]
    lines = [f"Budget: {budget['warmup']} critic warmup + {budget['joint']} joint = {budget['total']} updates"]
    for name, arm in report["arms"].items():
        lines.append(f"{name}: observed {arm['observed_warmup_updates']} warmup + {arm['observed_joint_updates']} joint; lambda={arm['q_critic_lambda']}")
        lines.extend(f"  {warning}" for warning in arm["warnings"])
        if arm["latest_rollout_sources"] is not None:
            lines.append(f"  latest rollout source diagnostics: {json.dumps(arm['latest_rollout_sources'], sort_keys=True)}")
    for metric, rows in report["matched_steps"].items():
        if not rows:
            lines.append(f"{metric}: no matched observations yet")
            continue
        row = rows[-1]
        values = "; ".join(f"{name}={json.dumps(value, sort_keys=True)}" for name, value in row["arms"].items())
        lines.append(f"{metric}: matched joint step {row['joint_step']} ({row['total_updates']} total), {values}")
        if metric in ("bench", "aime"):
            for name, arm in report["arms"].items():
                context = arm["evaluation_context"][metric][row["joint_step"]]
                baseline = context.get("subset_modal_answer_accuracy", context.get("dataset_modal_answer_accuracy"))
                if baseline is not None:
                    lines.append(f"  {name} recorded modal-answer baseline: {baseline}")
    lines.append(report["interpretation"])
    final = report["final_budget_comparison"]
    lines.append(f"Final {final['total_updates']}-update comparison at joint step {final['joint_step']}: {'ready' if final['ready'] else 'incomplete'}; earlier matched evaluations are interim.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True, help="Matched study.json manifest")
    parser.add_argument("--output", type=Path, help="Optional JSON report destination")
    args = parser.parse_args()
    report = build_report(args.study.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(render(report))


if __name__ == "__main__":
    main()
