#!/usr/bin/env python3
"""How much signal sibling-token Q gaps carry, and how much of it a critic's
action head recovers, from a critic study's ``branch_predictions.jsonl``.

CPU only: it reads saved branch outcomes and predictions, it runs no model.

Each record is one think position of a held-out trajectory where the policy
left mass off its token (``postraining/branch_rollouts.py``): the taken token
a and a policy-drawn alternative a', each with n continuations' binary
rewards. The observed gap g = mean(taken) - mean(alternative) is the true
gap D = Q(s, a) - Q(s, a') plus binomial noise that is independent of
anything a critic computes, so for any predictor x (the head's
u . (x(a) - x(a')), or the policy's log-prior gap):

* ``var_true``: Var(D) = Var(g) - E[noise variance], the noise variance per
  record from the unbiased p(1 - p) n / (n - 1) of each arm;
* ``calibration``: Cov(x, g) / Var(x), the slope of g on x, 1 for a
  calibrated predictor, unbiased under noise in g alone;
* ``signal_share``: Cov(x, g)^2 / (Var(x) Var(D)), the R^2 of x for the true
  gap: the share of the true gap's variance x explains, which is also the
  share of x's variance that is signal under a linear model;
* ``corr_observed``: the plain correlation with g, attenuated by the noise.

``--joint`` also regresses g on the critic and the log-prior together: the
critic's coefficient is what it adds beyond the policy's own preference.
Uncertainty is a bootstrap over prompts (records of one prompt share a
policy state).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def moments(x, g, noise):
    xc, gc = x - x.mean(), g - g.mean()
    var_x, var_g, cov = float(xc @ xc) / len(x), float(gc @ gc) / len(g), float(xc @ gc) / len(x)
    var_true = var_g - float(noise.mean())
    out = {"var_true": var_true, "var_observed": var_g}
    if var_x > 0:
        out["calibration"] = cov / var_x
        out["corr_observed"] = cov / np.sqrt(var_x * var_g)
        out["signal_share"] = cov * cov / (var_x * var_true) if var_true > 0 else float("nan")
    return out


def joint(x, prior, g):
    design = np.column_stack([np.ones_like(x), x, prior])
    coefficients, *_ = np.linalg.lstsq(design, g, rcond=None)
    return {"critic_coefficient": float(coefficients[1]), "prior_coefficient": float(coefficients[2])}


def statistics(records, bootstrap, seed):
    taken = np.array([r["taken_rewards"] for r in records], dtype=np.float64)
    alternative = np.array([r["alternative_rewards"] for r in records], dtype=np.float64)
    n_taken, n_alternative = taken.shape[1], alternative.shape[1]
    p, q = taken.mean(1), alternative.mean(1)
    g = p - q
    noise = p * (1 - p) / (n_taken - 1) + q * (1 - q) / (n_alternative - 1)
    critic = np.array([r["predicted_gap"] for r in records])
    prior = np.array([r["logprior_gap"] for r in records])
    _, groups = np.unique([r["prompt_sha256"] for r in records], return_inverse=True)

    def compute(index):
        out = {f"critic_{k}": v for k, v in moments(critic[index], g[index], noise[index]).items()}
        out.update({f"prior_{k}": v for k, v in moments(prior[index], g[index], noise[index]).items()
                    if k not in ("var_true", "var_observed")})
        if critic[index].std() > 0:
            out.update(joint(critic[index], prior[index], g[index]))
        out["resolved_share"] = float((np.abs(g[index]) > 2 * np.sqrt(noise[index] + 1e-12)).mean())
        return out

    full = compute(np.arange(len(g)))
    members = [np.flatnonzero(groups == k) for k in range(groups.max() + 1)]
    rng = np.random.default_rng(seed)
    draws = defaultdict(list)
    for _ in range(bootstrap):
        index = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        for key, value in compute(index).items():
            draws[key].append(value)
    result = {"records": len(g), "prompts": len(members), "continuations": [n_taken, n_alternative]}
    for key, value in full.items():
        values = np.array(draws[key])
        values = values[np.isfinite(values)]
        result[key] = value
        result[f"{key}_se"] = float(values.std(ddof=1)) if values.size > 1 else float("nan")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--steps", type=int, nargs="*", default=None)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    by_step = defaultdict(list)
    for line in args.predictions.read_text().splitlines():
        record = json.loads(line)
        by_step[record["step"]].append(record)
    steps = sorted(by_step) if args.steps is None else args.steps
    results = {step: statistics(by_step[step], args.bootstrap, args.seed) for step in steps}
    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
    shown = ("var_true", "resolved_share", "critic_calibration", "critic_signal_share",
             "critic_corr_observed", "prior_calibration", "prior_signal_share",
             "critic_coefficient", "prior_coefficient")
    for step, result in results.items():
        print(f"step {step}  records {result['records']}  prompts {result['prompts']}")
        print("   " + "  ".join(f"{k} {result[k]:+.4f}±{result[k + '_se']:.4f}"
                               for k in shown if k in result))


if __name__ == "__main__":
    main()
