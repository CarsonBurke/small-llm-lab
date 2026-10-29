#!/usr/bin/env python3
"""Alignment of MAC's estimators with the true policy gradient, from the
per-pool gradients of ``scripts/estimator_gradient_alignment.py``.

CPU only: it reads saved gradients, it runs no model.

Every pool is an independent draw of one update's data from one frozen
policy. For estimators E, F and pools k != l, E[g_E^k . g_F^l] =
E[g_E] . E[g_F]. sampled_baseline and control_variate are unbiased, so with
F unbiased that is E[g_E] . grad J, and with both unbiased it is |grad J|^2.

Euclidean statistics, per parameter group and over the whole actor:

* ``cos_step``: E[g_E] . grad J / (|grad J| sqrt(E|g_E|^2)), one update's
  alignment, bias and noise together (a ratio of expectations);
* ``cos_bias``: E[g_E] . grad J / (|grad J| |E g_E|), the alignment of the
  estimator's mean, bias alone (1 for an unbiased estimator), also against
  each reference alone (``cos_bias@ref``);
* ``snr``: |E g_E|^2 / (E|g_E|^2 - |E g_E|^2);
* ``cos_avg{m}``: the alignment of a mean of m independent updates, a
  momentum proxy: noise falls as 1/m, bias does not;
* ``disagreement`` between two references: |E g_a - E g_b|^2 / |grad J|^2,
  zero if both are unbiased for the same grad J, the data-level test of that
  assumption.

The drills do not step by g. First-order objective gains in each optimizer's
own geometry, E[step(g_E)] . grad J, estimated as step(g_E^k) . g_U^l:

* ``muon_gain_step``: Polar Express O(g) over the Muon matrices, one update;
* ``muon_gain_avg``: O of the mean of the other pools, scored on pool l,
  the direction momentum averages toward;
* ``sign_gain``: sign(g) over each AdamW group, Adam's first step;
* ``adam_gain``: g / (sqrt(v) + eps) over each AdamW group, v the series'
  mean squared gradient over the pools other than l, Adam's steady state
  without momentum.

``--entropy SERIES`` names a saved ``entropy_bonus`` series (the gradient of
minus the mean policy entropy, ``estimator_gradient_alignment.py
--entropy-gradient``) and adds each series' first-order entropy change, the
same cross-pool products with it in place of grad J:

* ``entropy_gain``: E[g_E] . E[g_H], the entropy change of a unit descent
  step along the mean (positive raises the entropy), and ``entropy_cos``,
  its cosine;
* ``muon_entropy_step`` and ``muon_entropy_avg``: the same for Muon's step
  of one update and of the other pools' mean, over the Muon matrices.

Every estimator's Muon and sign steps have the same norm, so their gains
compare directly. The step maps are nonlinear, so a difference series has
Euclidean statistics only. A series is ``label/estimator`` or a difference
``label/a-label/b``; all-action's belief-gain term is
``v7/all_action-v7/all_action_nogain``. ``--compare A:B`` adds the paired
difference A - B of every statistic. Uncertainty is a leave-one-pool-out
jackknife (``muon_gain_avg``'s is approximate: a dropped pool stays in the
other pools' means).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from postraining.muon import _polar_express  # noqa: E402

UNBIASED = ("sampled_baseline", "control_variate")
NAN = float("nan")
CHUNK = 1 << 21


def terms(spec: str) -> list[str]:
    """The saved series a series spec sums: ``label/a-label/b`` is a - b."""
    out = spec.split("-")
    if any(term.count("/") != 1 for term in out):
        raise ValueError(f"series {spec!r} is not label/estimator[-label/estimator]")
    return out


def polar(matrix: torch.Tensor) -> torch.Tensor:
    """Muon's update direction for one matrix: Polar Express, no momentum."""
    return _polar_express(matrix.float().clone(), torch.zeros(matrix.shape),
                          torch.tensor(0.0)).float()


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else NAN


class Moments:
    """Every inner product the statistics read, accumulated parameter by parameter.

    ``series`` are all requested series; ``base`` the saved single-term
    series, the only ones a nonlinear step map applies to.
    """

    def __init__(self, series, pools, groups, muon, adam_eps):
        self.series, self.P = series, len(pools)
        self.base = [spec for spec in series if len(terms(spec)) == 1]
        self.groups = sorted(set(groups.values()) | {"all"})
        S, B, P = len(series), len(self.base), self.P
        self.gram = {g: np.zeros((S * P, S * P)) for g in self.groups}
        adamw = sorted({groups[n] for n in groups if n not in muon})
        self.sign = {g: np.zeros((B * P, B * P)) for g in adamw}
        self.adam = {g: np.zeros((B * P, B * P)) for g in adamw}
        self.muon = np.zeros((B * P, B * P))
        self.muon_avg = np.zeros((B, P, B))
        self.adam_eps = adam_eps

    def add(self, name, group, rows, is_muon, shape):
        """``rows(start, end)`` is the (series x pool, slice) matrix of one parameter."""
        P, B = self.P, len(self.base)
        base_rows = [self.series.index(spec) for spec in self.base]
        size = math.prod(shape)
        for start in range(0, size, CHUNK):
            X = rows(start, start + CHUNK)
            Xd = X.double()
            gram = (Xd @ Xd.T).numpy()
            self.gram[group] += gram
            self.gram["all"] += gram
            if is_muon:
                continue
            Xb = X.reshape(len(self.series), P, -1)[base_rows]          # B, P, n
            flat = Xb.reshape(B * P, -1)
            self.sign[group] += (torch.sign(flat).double() @ flat.double().T).numpy()
            squares = Xb.square()
            total = squares.sum(1)
            for l in range(P):
                v = (total - squares[:, l]) / (P - 1)                    # B, n
                step = Xb / (v.sqrt() + self.adam_eps).unsqueeze(1)      # B, P, n
                scored = Xb[:, l]                                        # B, n
                self.adam[group][:, [b * P + l for b in range(B)]] += (
                    step.reshape(B * P, -1).double() @ scored.double().T).numpy()
        if is_muon:
            Xb = rows(None, None).reshape(len(self.series), P, *shape)[base_rows]
            flat = Xb.reshape(B * P, -1).double()
            steps = torch.stack([polar(x) for x in Xb.reshape(B * P, *shape)])
            self.muon += (steps.reshape(B * P, -1).double() @ flat.T).numpy()
            for s in range(B):
                for l in range(P):
                    others = [k for k in range(P) if k != l]
                    direction = polar(Xb[s, others].mean(0)).flatten().double()
                    self.muon_avg[s, l] += (flat[[b * P + l for b in range(B)]] @ direction).numpy()

    def statistics(self, keep, compare, entropy=None):
        """Flat ``group|series|statistic`` dict over the pools in ``keep``;
        with ``entropy``, every series' projection on that series too."""
        P = self.P
        cross = [(k, l) for k in keep for l in keep if k != l]
        refs = [spec for spec in self.base if spec.split("/")[1] in UNBIASED]
        S_index = {spec: i for i, spec in enumerate(self.series)}
        B_index = {spec: i for i, spec in enumerate(self.base)}
        out = {}

        def pair(matrix, a, b, index):
            return float(np.mean([matrix[index[a] * P + k, index[b] * P + l] for k, l in cross]))

        for group in self.groups:
            g = self.gram[group]
            truth = float(np.mean([pair(g, u, v, S_index) for u in refs for v in refs]))
            out[f"{group}||grad_J_sq"] = truth
            for a in refs:
                for b in refs:
                    if a < b:
                        gap = pair(g, a, a, S_index) + pair(g, b, b, S_index) - 2 * pair(g, a, b, S_index)
                        out[f"{group}|{a}~{b}|disagreement"] = ratio(gap, truth)
            for spec in self.series:
                signal = float(np.mean([pair(g, spec, u, S_index) for u in refs]))
                power = float(np.mean([g[S_index[spec] * P + k, S_index[spec] * P + k] for k in keep]))
                mean_sq = pair(g, spec, spec, S_index)
                noise = power - mean_sq
                key = f"{group}|{spec}|"
                out[key + "cos_step"] = ratio(signal, math.sqrt(truth * power)) if truth > 0 else NAN
                out[key + "cos_bias"] = (ratio(signal, math.sqrt(truth * mean_sq))
                                         if truth > 0 and mean_sq > 0 else NAN)
                for u in refs:
                    ref_sq = pair(g, u, u, S_index)
                    out[key + f"cos_bias@{u}"] = (
                        ratio(pair(g, spec, u, S_index), math.sqrt(ref_sq * mean_sq))
                        if ref_sq > 0 and mean_sq > 0 else NAN)
                out[key + "snr"] = ratio(mean_sq, noise)
                for m in (4, 20):
                    out[key + f"cos_avg{m}"] = (
                        ratio(signal, math.sqrt(truth * (mean_sq + noise / m)))
                        if truth > 0 and mean_sq + noise / m > 0 else NAN)
                out[key + "norm"] = math.sqrt(power)
                if entropy is not None and spec != entropy:
                    inner = pair(g, spec, entropy, S_index)
                    out[key + "entropy_gain"] = inner
                    scale = mean_sq * pair(g, entropy, entropy, S_index)
                    out[key + "entropy_cos"] = inner / math.sqrt(scale) if scale > 0 else NAN

        def gain(matrix, spec):
            return float(np.mean([pair(matrix, spec, u, B_index) for u in refs]))

        for spec in self.base:
            out[f"muon_matrices|{spec}|muon_gain_step"] = gain(self.muon, spec)
            out[f"muon_matrices|{spec}|muon_gain_avg"] = float(np.mean(
                [self.muon_avg[B_index[spec], l, B_index[u]] for u in refs for l in keep]))
            for group in self.sign:
                out[f"{group}|{spec}|sign_gain"] = gain(self.sign[group], spec)
                out[f"{group}|{spec}|adam_gain"] = gain(self.adam[group], spec)
            if entropy is not None and spec != entropy:
                out[f"muon_matrices|{spec}|muon_entropy_step"] = pair(
                    self.muon, spec, entropy, B_index)
                out[f"muon_matrices|{spec}|muon_entropy_avg"] = float(np.mean(
                    [self.muon_avg[B_index[spec], l, B_index[entropy]] for l in keep]))
        for a, b in compare:
            for key in list(out):
                group, spec, name = key.split("|")
                other = f"{group}|{b}|{name}"
                if spec == a and other in out:
                    out[f"{group}|{a} - {b}|{name}"] = out[key] - out[other]
        return out


def analyze(gradients, series, pools, groups, muon, compare=(), adam_eps=1e-8, entropy=None):
    """``gradients[(saved series, pool)][parameter]``; ``groups[parameter]``;
    ``entropy`` a saved single-term series among ``series``."""
    refs = [spec for spec in series if len(terms(spec)) == 1 and spec.split("/")[1] in UNBIASED]
    if not refs:
        raise ValueError("an unbiased series (sampled_baseline, control_variate) is the reference")
    if entropy is not None and (entropy not in series or len(terms(entropy)) != 1):
        raise ValueError(f"the entropy series {entropy!r} must be a saved series analyzed")
    for a, b in compare:
        if a not in series or b not in series:
            raise ValueError(f"comparison {a}:{b} names a series not analyzed")
    moments = Moments(series, pools, groups, muon, adam_eps)
    first = (terms(series[0])[0], pools[0])
    for name, tensor in gradients[first].items():
        def rows(start, end, name=name):
            out = []
            for spec in series:
                for pool in pools:
                    head, *rest = terms(spec)
                    value = gradients[(head, pool)][name].float().flatten()[start:end]
                    for term in rest:
                        value = value - gradients[(term, pool)][name].float().flatten()[start:end]
                    out.append(value)
            return torch.stack(out)

        moments.add(name, groups[name], rows, name in muon, tuple(tensor.shape))

    P = len(pools)
    full = moments.statistics(list(range(P)), compare, entropy)
    jack = [moments.statistics([p for p in range(P) if p != drop], compare, entropy)
            for drop in range(P)]
    result = {}
    for key, value in full.items():
        values = np.array([j[key] for j in jack])
        se = (float(np.sqrt((P - 1) / P * np.sum((values - values.mean()) ** 2)))
              if np.all(np.isfinite(values)) else NAN)
        group, spec, name = key.split("|")
        entry = result.setdefault(group, {}).setdefault(spec or "_", {})
        entry[name], entry[f"{name}_se"] = value, se
    return {"series": series, "pools": pools, "references": [s for s in refs],
            "entropy": entropy, "compare": [list(pair) for pair in compare], "groups": result}


def group_of(name: str, muon: set[str]) -> str:
    if name in muon:
        return "muon_matrices"
    if name == "backbone.embed.weight":
        return "embedding"
    if name.startswith("backbone.proj."):
        return "readout"
    return "vectors"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--series", nargs="+", required=True)
    parser.add_argument("--compare", nargs="*", default=[], help="A:B pairs, reported as A - B")
    parser.add_argument("--entropy", default=None,
                        help="a saved entropy_bonus series; added to --series if absent")
    parser.add_argument("--pools", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(16)
    if args.entropy is not None and args.entropy not in args.series:
        args.series.append(args.entropy)

    saved = sorted({term for spec in args.series for term in terms(spec)})
    gradients = {(term, pool): torch.load(args.root / term.split("/")[0]
                                          / f"{term.split('/')[1]}__pool{pool:02d}.pt",
                                          map_location="cpu", mmap=True, weights_only=True)
                 for term in saved for pool in args.pools}
    configs = [json.loads((args.root / label / "config.json").read_text())
               for label in sorted({term.split("/")[0] for term in saved})]
    if len({json.dumps([c["actor_parameters"], c["muon_parameters"]]) for c in configs}) != 1:
        raise ValueError("the labels hold different actors or Muon partitions")
    names = list(gradients[(saved[0], args.pools[0])])
    for key, gradient in gradients.items():
        if list(gradient) != names:
            raise ValueError(f"{key} holds other parameters")
    muon = set(configs[0]["muon_parameters"])
    compare = [tuple(pair.split(":")) for pair in args.compare]
    result = analyze(gradients, args.series, args.pools, {n: group_of(n, muon) for n in names},
                     muon, compare, entropy=args.entropy)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    def show(entry, name):
        value, se = entry.get(name, NAN), entry.get(f"{name}_se", NAN)
        return f"{value:+.4g}±{se:.2g}"

    for group, specs in result["groups"].items():
        print(f"\n[{group}]  |grad J|^2 {show(specs['_'], 'grad_J_sq')}")
        for spec, entry in specs.items():
            if spec == "_":
                continue
            names = [n for n in entry if not n.endswith("_se")]
            print(f"  {spec}")
            print("    " + "  ".join(f"{n} {show(entry, n)}" for n in names))


if __name__ == "__main__":
    main()
