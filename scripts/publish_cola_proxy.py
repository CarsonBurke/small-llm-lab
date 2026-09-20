#!/usr/bin/env python3
"""Publish completed Cola CNF estimates to canonical metrics and TensorBoard.

This reads existing numerical evaluation evidence; it executes no model. It does
not turn reconstruction CE or flow MSE into a language-model rate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretraining.nanogpt_mini.native_bits_data import sha256_file
from scripts.ablation import MetricsWriter

DEFINITION = (
    "Numerical CNF negative-ELBO estimate in bits per literal source byte: "
    "(reconstruction NLL - latent-prior log density + posterior log density) / "
    "(source bytes * ln(2)). One posterior sample and Gaussian trace probe per "
    "context; finest evaluated Heun resolution. Not exact marginal/challenge BPB. "
    "The solver-resolution delta is a convergence diagnostic, not an error bound."
)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".working")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def publish(run_dir: Path, evaluation_path: Path | None = None) -> dict:
    run_dir = run_dir.resolve()
    evaluation_path = evaluation_path or run_dir / "evaluation.json"
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text())
    report = json.loads(evaluation_path.read_text())
    if result["name"] != run_dir.name or run_dir.parent != ROOT / "ablation_results":
        raise ValueError("publish to a canonical named ablation run")
    if (
        result["stage"] != "joint"
        or result["completed_steps"] != result["steps"]
        or result["returncode"] != 0
    ):
        raise ValueError("proxy publication requires a completed joint control")
    if (
        report.get("status") != "complete"
        or report["checkpoint_sha256"] != result["checkpoint_sha256"]
    ):
        raise ValueError(
            "evaluation is incomplete or belongs to a different checkpoint"
        )
    digest = sha256_file(evaluation_path)
    summary_path = run_dir / "proxy_bpb.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous["evaluation_sha256"] == digest:
            return previous
    resolutions = sorted(report["density"], key=lambda row: row["steps"])
    if not resolutions or len({row["steps"] for row in resolutions}) != len(
        resolutions
    ):
        raise ValueError("density evaluation must contain distinct solver resolutions")
    expected_bytes = report["data"]["validation"]["bytes"]
    for row in resolutions:
        totals = row["totals"]
        if totals["bytes"] != expected_bytes or expected_bytes <= 0:
            raise ValueError("proxy must cover the complete literal validation stream")
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in totals.values()
        ):
            raise ValueError("non-finite or nonnumeric density totals")
        derived = (
            totals["reconstruction_nll_nats"]
            - totals["prior_logprob_nats"]
            + totals["posterior_logprob_nats"]
        )
        if not math.isclose(
            derived, totals["negative_elbo_estimate_nats"], rel_tol=1e-6, abs_tol=1e-3
        ):
            raise ValueError("density components do not reconstruct the negative ELBO")
        if not math.isclose(
            totals["negative_elbo_estimate_bpb"],
            totals["negative_elbo_estimate_nats"] / (expected_bytes * math.log(2)),
            rel_tol=1e-9,
        ):
            raise ValueError("proxy denominator is not literal source bytes")
    finest = resolutions[-1]
    totals = finest["totals"]
    divisor = expected_bytes * math.log(2)
    entry = {
        "type": "diag",
        "step": result["completed_steps"],
        "val_proxy_bpb": totals["negative_elbo_estimate_bpb"],
        "val_proxy_bpb_heun_steps": finest["steps"],
        "val_proxy_bpb_source_bytes": expected_bytes,
        "val_proxy_bpb_positions": totals["positions"],
        "val_proxy_reconstruction_bpb": totals["reconstruction_nll_nats"] / divisor,
        "val_proxy_prior_nll_bpb": -totals["prior_logprob_nats"] / divisor,
        "val_proxy_posterior_logq_bpb": totals["posterior_logprob_nats"] / divisor,
    }
    if len(resolutions) > 1:
        entry["val_proxy_bpb_solver_delta"] = abs(
            totals["negative_elbo_estimate_bpb"]
            - resolutions[-2]["totals"]["negative_elbo_estimate_bpb"]
        )
    summary = {
        "kind": "cnf_negative_elbo_estimate_bpb",
        "definition": DEFINITION,
        "run": result["name"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "evaluation_path": str(evaluation_path.resolve()),
        "evaluation_sha256": digest,
        "validation_sha256": report["data"]["validation"]["sha256"],
        "tensorboard_tag": "val/proxy_bpb",
        "metric": entry,
    }
    writer = MetricsWriter(run_dir / "metrics.jsonl", result["name"])
    try:
        writer.write_entry(entry)
        writer.writer.add_text("val/proxy_bpb_definition", DEFINITION, entry["step"])
    finally:
        writer.close()
    result["final_proxy_val_bpb"] = entry["val_proxy_bpb"]
    result["proxy_metric"] = summary["kind"]
    result["proxy_bpb_report"] = str(summary_path)
    # Keep final_val_bpb null and the training-objective promotion classification:
    # this numerical proxy must not enter the exact-BPB leaderboard silently.
    atomic_json(result_path, result)
    atomic_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.run_dir, args.evaluation), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
