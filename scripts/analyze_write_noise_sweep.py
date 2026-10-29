#!/usr/bin/env python3
"""How much of the return a memory write-noise draw explains, from a
``--write-noise-sweep`` artifact (``postraining/perturbation_sweep.py``).

CPU only: it reads ``write_noise_sweep/collects.jsonl``, it runs no model.

Per pool and scale s, every trajectory i has a noiseless return G0 and the
coupled returns G(+s), G(-s) under one noise pattern shared by its prompt's
samples. With D = (G(+s) - G(-s)) / 2 (the odd part of the return's response
to the noise) and C = (G(+s) + G(-s)) / 2 - G0 (the even part):

D is the first-order response a DDPG-style critic's action gradient
follows; both bounds are on it, not on all noise-explained variance:

* ``upper``: E[D^2]. It contains the signal and the continuation noise of
  every trajectory a flipped token sent elsewhere, so it bounds the
  variance of the expected odd response from above.
* ``lower``: the mean of D_i D_j over distinct samples of one prompt. They
  share the noise but not the continuation noise, so it is an unbiased
  estimate of Var_eps(E[D | prompt, eps]): what a critic that sees only the
  prompt and the noise could learn. A critic that also sees the state can
  learn more, up to ``upper``.
* ``control``: the same product across neighboring prompts, whose noise
  patterns are independent; zero in expectation. Samples are not exactly
  independent given the noise -- decode batch shape, which follows when
  other rows stop, moves low logit bits -- so ``control`` is also the check
  that this numerical coupling is small next to ``lower``.
* ``cost``: E[C], exploration's second-order effect on the return
  (negative at a local maximum of the return in write space).
* ``even_lower``: the within-prompt product C_i C_j, an unbiased estimate of
  E[E[C | prompt, eps]^2]: the response to the noise's magnitude, which D
  never sees. Less ``cost^2`` it is that response's variance.
* ``rho2_*``: each bound over the within-prompt return variance -- the
  variance left after a per-prompt baseline, which is what a write critic's
  regression of G on the noise competes with.
* ``pools_to_half_fidelity``: the red-team sample-size rule n = p / rho^2
  for a state-independent slope with p = 2 H K + 2 H noise coordinates, in
  pools of this size, at ``rho2_lower``.

Lower scaling as s^2 and an upper bound dominated by flips (scaling as s)
separate signal from flip noise; both are reported per scale. Uncertainty
is a bootstrap over prompts.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    pools = defaultdict(dict)
    for line in (path / "collects.jsonl").read_text().splitlines():
        record = json.loads(line)
        pool = pools[record["pool"]]
        if record["kind"] == "base":
            pool["base"] = record
        elif record["kind"] == "null":
            pool["null"] = record
        else:
            pool[(record["scale"], record["sign"])] = record
    return manifest, pools


def per_prompt_terms(base, plus, minus, samples):
    """(prompts, samples) arrays of one pool at one scale."""
    shape = (-1, samples)
    g0 = np.asarray(base["rewards"], dtype=np.float64).reshape(shape)
    gp = np.asarray(plus["rewards"], dtype=np.float64).reshape(shape)
    gm = np.asarray(minus["rewards"], dtype=np.float64).reshape(shape)
    changed = (
        np.asarray(plus["changed"], dtype=np.float64)
        + np.asarray(minus["changed"], dtype=np.float64)
    ).reshape(shape) / 2
    d = (gp - gm) / 2
    c = (gp + gm) / 2 - g0
    def within_prompt_cross(x):
        return (x.sum(1) ** 2 - (x**2).sum(1)) / (samples * (samples - 1))

    means = d.mean(1)
    # Each prompt against the next one of its pool: independent noise.
    control = means * np.roll(means, -1)
    return {
        "upper": (d**2).mean(1),
        "lower": within_prompt_cross(d),
        "control": control,
        "cost": c.mean(1),
        "even_lower": within_prompt_cross(c),
        "flip_rate": changed.mean(1),
        "var_within": g0.var(1, ddof=1),
        "reward_plus": gp.mean(1),
        "reward_minus": gm.mean(1),
        "reward_base": g0.mean(1),
    }


def summarize(terms, bootstrap, rng):
    """Means of per-prompt terms, the ratios, and prompt-bootstrap SEs."""
    def compute(index):
        mean = {key: float(value[index].mean()) for key, value in terms.items()}
        var = mean["var_within"]
        mean["rho2_upper"] = mean["upper"] / var if var > 0 else float("nan")
        mean["rho2_lower"] = mean["lower"] / var if var > 0 else float("nan")
        return mean

    prompts = len(terms["upper"])
    full = compute(np.arange(prompts))
    draws = defaultdict(list)
    for _ in range(bootstrap):
        for key, value in compute(rng.integers(0, prompts, prompts)).items():
            draws[key].append(value)
    out = {}
    for key, value in full.items():
        out[key] = value
        out[f"{key}_se"] = float(np.std(draws[key], ddof=1)) if bootstrap > 1 else float("nan")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep", type=Path, help="the run's write_noise_sweep directory")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    manifest, pools = load(args.sweep)
    samples = manifest["samples_per_prompt"]
    noise = manifest["noise"]
    coordinates = 2 * noise["memory_heads"] * noise["memory_head_dim"] + 2 * noise["memory_heads"]
    complete = sorted(
        index for index, pool in pools.items()
        if "base" in pool and "null" in pool
        and all((scale, sign) in pool for scale in manifest["scales"] for sign in (1, -1))
    )
    if not complete:
        raise SystemExit("no complete pool in the sweep")
    null_changed = sum(sum(pools[index]["null"]["changed"]) for index in complete)
    null_shift = max(
        float(np.abs(np.subtract(pools[index]["null"]["rewards"], pools[index]["base"]["rewards"])).max())
        for index in complete
    )
    trajectories = len(pools[complete[0]]["base"]["rewards"])
    rng = np.random.default_rng(args.seed)
    result = {
        "pools": len(complete),
        "trajectories_per_pool": trajectories,
        "samples_per_prompt": samples,
        "noise_coordinates": coordinates,
        "null": {"changed": null_changed, "abs_reward_shift": null_shift},
        "scales": {},
    }
    for scale in manifest["scales"]:
        per_pool = [
            per_prompt_terms(
                pools[index]["base"], pools[index][(scale, 1)], pools[index][(scale, -1)], samples
            )
            for index in complete
        ]
        terms = {key: np.concatenate([pool[key] for pool in per_pool]) for key in per_pool[0]}
        stats = summarize(terms, args.bootstrap, rng)
        rho2 = stats["rho2_lower"]
        stats["pools_to_half_fidelity"] = (
            coordinates / (rho2 * trajectories) if rho2 > 0 else float("inf")
        )
        stats["lower_over_scale_sq"] = stats["lower"] / scale**2
        stats["upper_over_scale"] = stats["upper"] / scale
        result["scales"][f"{scale:g}"] = stats
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"pools {result['pools']}  trajectories/pool {trajectories}  null {result['null']}")
    shown = ("flip_rate", "cost", "even_lower", "upper", "lower", "control",
             "rho2_upper", "rho2_lower")
    for scale, stats in result["scales"].items():
        print(f"scale {scale}  pools_to_half_fidelity {stats['pools_to_half_fidelity']:.3g}")
        print("   " + "  ".join(f"{k} {stats[k]:+.5f}±{stats[k + '_se']:.5f}" for k in shown))


if __name__ == "__main__":
    main()
