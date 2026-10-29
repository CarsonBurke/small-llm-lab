#!/usr/bin/env python3
"""Analyze an antithetic memory perturbation sweep (CPU only, no model).

Reads ``<run>/perturbation_sweep/`` (``postraining.perturbation_sweep``),
rebuilds every direction from the bound policy snapshot and the saved
token-path gradients, refuses any whose digest differs from the one the run
recorded, and reports, per target:

- the null control: trajectories changed and reward shift on the
  unperturbed re-collect (must be zero for exact coupling);
- per kind and scale: the directional derivative D = E[G(+) - G(-)] /
  (2 delta), in return per one-RMS move, with standard errors clustered by
  prompt (a pool's 16 samples of one prompt are one cluster), the flip
  rate, the share of the paired difference's second moment that is
  systematic, E[dG]^2 / E[dG^2], and the share of the return's variance a
  +/- delta move explains, (delta D)^2 / Var(G);
- random directions: E[D^2] minus its noise floor, an unbiased estimate of
  |P^1/2 grad J|^2 (P = rms^2 per tensor), against the token path's
  |P^1/2 g|^2 from cross-pool gradient products;
- the gradient direction: D on pool k along the mean of the earlier pools'
  token-path gradients, against the token path's own prediction g . d from
  pool k (independent of d) and from every pool j >= k;
- linearity: D at the smaller scale against D at the larger, paired by
  direction, and the one-sided curvature (G(+) + G(-) - 2 G(0)) / delta^2.

Uncertainties are 1 standard error; ``z`` is estimate / SE.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from postraining.perturbation_sweep import (  # noqa: E402
    PERTURBATION_SWEEP_SCHEMA,
    PERTURBATION_TARGETS,
    direction_digest,
    gradient_direction,
    radial_direction,
    random_direction,
)
from postraining.vapo.mixture import file_sha256  # noqa: E402


def clustered_mean(values: np.ndarray, clusters: int) -> tuple[float, float]:
    """Mean and SE of equal-size clusters laid out cluster-major."""
    means = values.reshape(clusters, -1).mean(1)
    if clusters < 2:
        return float(means.mean()), float("nan")
    return float(means.mean()), float(means.std(ddof=1) / math.sqrt(clusters))


def mean_se(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return float(values.mean()) if values.size else float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(values.size))


def fmt(estimate: float, se: float, digits: int = 4) -> str:
    z = estimate / se if se and math.isfinite(se) and se > 0 else float("nan")
    return f"{estimate:+.{digits}g} +/- {se:.2g} (z {z:+.2f})"


def load(root: Path, verify_snapshot: bool):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != PERTURBATION_SWEEP_SCHEMA:
        raise ValueError(f"{root} is not a {PERTURBATION_SWEEP_SCHEMA} artifact")
    records = [json.loads(line) for line in (root / "collects.jsonl").read_text().splitlines()]
    snapshot_path = Path(manifest["policy_state"])
    if verify_snapshot and file_sha256(snapshot_path) != manifest["policy_state_sha256"]:
        raise ValueError(f"{snapshot_path} is not the snapshot the sweep perturbed")
    model = torch.load(snapshot_path, map_location="cpu", weights_only=False)["model"]
    targets = {
        target: {name: model[name] for name in PERTURBATION_TARGETS[target]}
        for target in manifest["config"]["targets"]
    }
    gradients = {
        int(path.stem.split("_")[1]): torch.load(path, map_location="cpu", weights_only=True)
        for path in sorted((root / "gradients").glob("pool_*.pt"))
    }
    return manifest, records, targets, gradients


def rebuild(kind, target, parameters, pool, index, sources, gradients, seed):
    if kind == "random":
        return random_direction(parameters, seed, target, pool, index)
    if kind == "radial":
        return radial_direction(parameters)
    if kind == "gradient":
        mean = {
            name: torch.stack([gradients[j][name].double() for j in sources]).mean(0)
            for name in parameters
        }
        return gradient_direction(parameters, mean)
    raise ValueError(kind)


def dot(gradient, direction) -> float:
    return float(sum((gradient[name].double() * value.double()).sum() for name, value in direction.items()))


def metric_dot(a, b, parameters) -> float:
    """<a, b>_P with P = rms^2 per tensor (rms on CPU in fp64)."""
    total = 0.0
    for name, value in parameters.items():
        rms2 = float(value.double().square().mean())
        total += rms2 * float((a[name].double() * b[name].double()).sum())
    return total


def analyze(root: Path, verify_snapshot: bool) -> dict:
    manifest, records, targets, gradients = load(root, verify_snapshot)
    seed = int(manifest["seed"])
    samples = int(manifest["samples_per_prompt"])
    base = {r["pool"]: r for r in records if r["kind"] == "base"}
    null = {r["pool"]: r for r in records if r["kind"] == "null"}
    pools = sorted(base)
    # A pool counts only once every collect it planned was written.
    perturbed = defaultdict(dict)
    for r in records:
        if r["kind"] in ("gradient", "radial", "random"):
            perturbed[(r["pool"], r["target"], r["kind"], r["index"], r["scale"])][r["sign"]] = r
    complete = sorted({key[0] for key, pair in perturbed.items() if set(pair) == {1, -1}})
    report = {"pools": len(pools), "trajectories_per_pool": len(base[pools[0]]["rewards"]),
              "samples_per_prompt": samples}

    # --- null control
    warmup = sum(bool(base[p].get("warmup")) for p in pools)
    report["warmup_pools"] = warmup
    null_changed = sum(sum(null[p]["changed"]) for p in null)
    null_shift = sum(
        sum(abs(a - b) for a, b in zip(null[p]["rewards"], base[p]["rewards"], strict=True)) for p in null)
    report["null"] = {"pools": len(null), "changed": null_changed, "abs_reward_shift": null_shift}
    print(f"pools {len(pools)} ({warmup} warmup, {len(complete)} complete test pools); null control: {null_changed} trajectories "
          f"changed, total |reward shift| {null_shift:g}")

    base_rewards = np.concatenate([np.asarray(base[p]["rewards"]) for p in pools])
    clusters_per_pool = len(base[pools[0]]["rewards"]) // samples
    var_total = float(base_rewards.var())
    var_within = float(base_rewards.reshape(-1, samples).var(1, ddof=1).mean())
    report["return"] = {"mean": float(base_rewards.mean()), "var_total": var_total, "var_within_prompt": var_within}
    print(f"base return mean {base_rewards.mean():.4f}, Var(G) {var_total:.4f}, "
          f"within-prompt Var {var_within:.4f}")

    # --- rebuild and verify directions, compute token predictions
    instances = []
    for (pool, target, kind, index, scale), pair in sorted(perturbed.items(), key=lambda kv: str(kv[0])):
        if set(pair) != {1, -1}:
            continue
        plus, minus = pair[1], pair[-1]
        parameters = targets[target]
        direction = rebuild(kind, target, parameters, pool, index, plus["gradient_source_pools"],
                            gradients, seed)
        if direction_digest(direction) != plus["direction_sha256"]:
            raise ValueError(f"direction {(pool, target, kind, index)} does not rebuild to its recorded digest")
        g0 = np.asarray(base[pool]["rewards"])
        gp, gm = np.asarray(plus["rewards"]), np.asarray(minus["rewards"])
        # Every gradient estimate independent of d: all pools but its sources.
        excluded = set(plus["gradient_source_pools"])
        predictors = [j for j in sorted(gradients) if j not in excluded]
        predictions = [dot(gradients[j], direction) for j in predictors]
        instances.append({
            "pool": pool, "target": target, "kind": kind, "index": index, "scale": scale,
            "g0": g0, "gp": gp, "gm": gm,
            "changed_plus": np.asarray(plus["changed"]), "changed_minus": np.asarray(minus["changed"]),
            "same_pool_prediction": plus["same_pool_token_prediction"],
            "predictions": predictions,
        })

    for target in manifest["config"]["targets"]:
        parameters = targets[target]
        n = sum(value.numel() for value in parameters.values())
        print(f"\n=== target {target}: {n} elements")
        out = report.setdefault("targets", {}).setdefault(target, {"elements": n})
        # Token-path metric gradient norm: cross-pool products are unbiased for |P^1/2 g|^2.
        pool_ids = sorted(gradients)
        if len(pool_ids) >= 2:
            cross = [metric_dot(gradients[a], gradients[b], parameters)
                     for i, a in enumerate(pool_ids) for b in pool_ids[i + 1:]]
            own = [metric_dot(gradients[a], gradients[a], parameters) for a in pool_ids]
            signal, signal_se = mean_se(cross)
            out["token_metric_grad_sq"] = [signal, signal_se]
            out["token_cross_pool_signal_fraction"] = signal / float(np.mean(own))
            print(f"token path |P^1/2 g|^2 (cross-pool): {fmt(signal, signal_se)}; "
                  f"per-pool |P^1/2 g_k|^2 mean {np.mean(own):.4g}; signal fraction "
                  f"{signal / float(np.mean(own)):+.3f}  (SE from non-independent pairs: indicative)")
            print(f"  => the perfect gradient direction would give D = sqrt(n |P^1/2 g|^2) = "
                  f"{math.sqrt(max(signal, 0.0) * n):.4g}")
        rows = [i for i in instances if i["target"] == target]
        for kind in ("gradient", "radial", "random"):
            for scale in sorted({i["scale"] for i in rows if i["kind"] == kind}):
                group = [i for i in rows if i["kind"] == kind and i["scale"] == scale]
                if not group:
                    continue
                dg = np.concatenate([(i["gp"] - i["gm"]) for i in group])
                clusters = len(group) * clusters_per_pool
                # Even in d, so it pools over random directions too.
                curvature, curvature_se = clustered_mean(
                    np.concatenate([(i["gp"] + i["gm"] - 2 * i["g0"]) for i in group]) / scale**2, clusters)
                flips = float(np.mean([np.r_[i["changed_plus"], i["changed_minus"]].mean() for i in group]))
                second = float(np.mean(dg**2))
                per_direction = [clustered_mean((i["gp"] - i["gm"]) / (2 * scale), clusters_per_pool)
                                 for i in group]
                key = f"{kind}/{scale:g}"
                entry = {"directions": len(group), "flip_rate": flips, "curvature": [curvature, curvature_se],
                         "dG_second_moment": second, "per_direction_D": per_direction}
                print(f"\n  {kind} scale {scale:g}: {len(group)} direction-pools, flip rate {flips:.3f}, "
                      f"E[dG^2] {second:.4g}")
                print(f"    curvature (G+ + G- - 2G0)/delta^2: {fmt(curvature, curvature_se)}")
                if kind == "random":
                    # Random signs: G+ - G- pools to zero across directions, so
                    # everything here is per direction. E[Dhat^2] = D^2 + SE^2,
                    # and the excess estimates |P^1/2 grad J|^2.
                    excess = [d**2 - se**2 for d, se in per_direction]
                    e, e_se = mean_se(excess)
                    floor = float(np.mean([se**2 for _, se in per_direction]))
                    explained = scale**2 * e / var_total
                    systematic = 4 * scale**2 * e / second if second > 0 else float("nan")
                    entry.update(true_metric_grad_sq=[e, e_se], noise_floor=floor,
                                 return_variance_share=explained, systematic_share=systematic)
                    print(f"    E[D^2] - floor: {fmt(e, e_se)} (floor {floor:.3g}); "
                          f"upper 2-SE bound on |P^1/2 grad J| {math.sqrt(max(e + 2 * e_se, 0.0)):.4g}")
                    print(f"    delta^2 E[D^2]/Var(G) {explained:.3g}; systematic share 4 delta^2 E[D^2]/E[dG^2] "
                          f"{systematic:.3g}")
                    tok = [float(np.mean(i["predictions"])) for i in group if i["predictions"]]
                    if tok:
                        measured = [d for d, _ in per_direction]
                        corr = float(np.corrcoef(measured, tok)[0, 1]) if len(tok) > 2 else float("nan")
                        print(f"    token prediction RMS {math.sqrt(np.mean(np.square(tok))):.4g}; "
                              f"corr(measured D, token prediction) over directions {corr:+.3f}")
                else:
                    # One fixed direction per kind: D pools over pools.
                    d_all, d_se = clustered_mean(dg / (2 * scale), clusters)
                    systematic = float(np.mean(dg)) ** 2 / second if second > 0 else float("nan")
                    explained = (scale * d_all) ** 2 / var_total
                    m, m_se = mean_se([d for d, _ in per_direction])
                    entry.update(D=[d_all, d_se], measured_mean=[m, m_se], systematic_share=systematic,
                                 return_variance_share=explained)
                    print(f"    D (prompt-clustered over all pools) {fmt(d_all, d_se)}; across-pool {fmt(m, m_se)}")
                    print(f"    systematic share E[dG]^2/E[dG^2] {systematic:.3g}; "
                          f"(delta D)^2/Var(G) {explained:.3g}")
                    same = [i["same_pool_prediction"] for i in group]
                    if all(t is not None for t in same):
                        measured = [d for d, _ in per_direction]
                        residual, residual_se = mean_se(np.subtract(measured, same))
                        t_mean, t_se = mean_se(same)
                        entry.update(same_pool_prediction_mean=[t_mean, t_se],
                                     residual_true_minus_token=[residual, residual_se],
                                     ratio_true_over_token=d_all / t_mean if t_mean else float("nan"))
                        print("    per pool: " + ", ".join(
                            f"p{i['pool']}: D {d:+.4f}+/-{se:.4f} tok {t:+.4f}"
                            for i, (d, se), t in zip(group, per_direction, same)))
                        print(f"    token prediction g_k . d (same pool) {fmt(t_mean, t_se)}")
                        print(f"    measured - token: {fmt(residual, residual_se)}; ratio {d_all / t_mean:+.3f}"
                              if t_mean else "")
                out[key] = entry
        # Linearity: same direction instances at two scales.
        scales = sorted({i["scale"] for i in rows})
        if len(scales) >= 2:
            small, large = scales[0], scales[-1]
            by_key = defaultdict(dict)
            for i in rows:
                by_key[(i["pool"], i["kind"], i["index"])][i["scale"]] = i
            # Random directions are left out: their G+ - G- has a random
            # sign, and the two scales' flip noise is correlated.
            for kind in ("gradient", "radial"):
                diffs, ds, dl = [], [], []
                for key, pair in by_key.items():
                    if key[1] != kind or small not in pair or large not in pair:
                        continue
                    a = (pair[small]["gp"] - pair[small]["gm"]) / (2 * small)
                    b = (pair[large]["gp"] - pair[large]["gm"]) / (2 * large)
                    diffs.append(b - a)
                    ds.append(a)
                    dl.append(b)
                if diffs:
                    clusters = len(diffs) * clusters_per_pool
                    d, se = clustered_mean(np.concatenate(diffs), clusters)
                    a, a_se = clustered_mean(np.concatenate(ds), clusters)
                    b, b_se = clustered_mean(np.concatenate(dl), clusters)
                    print(f"  linearity {kind}: D({small:g}) {fmt(a, a_se)}, D({large:g}) {fmt(b, b_se)}, "
                          f"difference {fmt(d, se)}")
                    out[f"linearity/{kind}"] = {"small": [a, a_se], "large": [b, b_se], "difference": [d, se]}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="the sweep's run directory")
    parser.add_argument("--json", type=Path, default=None, help="also write the report here")
    parser.add_argument("--skip-snapshot-hash", action="store_true",
                        help="skip re-hashing the policy snapshot (the direction digests still bind it)")
    args = parser.parse_args()
    report = analyze(args.run / "perturbation_sweep", not args.skip_snapshot_hash)
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
