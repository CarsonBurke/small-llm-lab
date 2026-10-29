#!/usr/bin/env python3
"""Mean gradients of MAC's estimators and how their directions relate, from
the per-pool gradients of ``scripts/estimator_gradient_alignment.py``.

CPU only: it reads saved gradients, it runs no model.

``scripts/analyze_estimator_alignment.py`` scores every series against
grad J, estimated from the unbiased series. This reads any pair of series
against each other instead, for example a biased estimator's departure
from the sampled baseline (``label/a-label/b``) against another estimator's.

From the Gram matrix of every (series, pool) pair, per parameter group and
over the whole actor, for pools k != l (independent data, also across labels
that share rollouts):

* ``mean_sq``: |E g|^2, estimated by mean g^k . g^l, unbiased;
* ``total_sq``: E|g|^2, mean g^k . g^k;
* ``snr``: mean_sq / (total_sq - mean_sq);
* ``--cross A:B``: ``cos``, E g_A . E g_B / (|E g_A| |E g_B|), and
  ``inner``, E g_A . E g_B, unnormalized so a difference series is readable.

Uncertainty is a leave-one-pool-out jackknife.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_estimator_alignment import CHUNK, NAN, group_of, ratio, terms  # noqa: E402


def statistics(gram, series, pools, cross, keep):
    """``gram[i, j]`` over (series, pool) rows, restricted to pools ``keep``."""
    P = len(pools)

    def mean_pair(a, b, same):
        values = [gram[a * P + k, b * P + l] for k in keep for l in keep if (k == l) == same]
        return float(np.mean(values))

    out = {}
    for a, spec in enumerate(series):
        mean_sq, total_sq = mean_pair(a, a, False), mean_pair(a, a, True)
        out[f"{spec}|mean_sq"] = mean_sq
        out[f"{spec}|total_sq"] = total_sq
        out[f"{spec}|snr"] = ratio(mean_sq, total_sq - mean_sq)
    for left, right in cross:
        a, b = series.index(left), series.index(right)
        inner = mean_pair(a, b, False)
        norms = out[f"{left}|mean_sq"] * out[f"{right}|mean_sq"]
        out[f"{left}:{right}|inner"] = inner
        out[f"{left}:{right}|cos"] = inner / math.sqrt(norms) if norms > 0 else NAN
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--series", nargs="+", required=True)
    parser.add_argument("--cross", nargs="*", default=[], help="A:B pairs of series")
    parser.add_argument("--pools", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(16)
    cross = [tuple(pair.split(":")) for pair in args.cross]
    for pair in cross:
        if len(pair) != 2 or any(spec not in args.series for spec in pair):
            raise ValueError(f"cross pair {pair} names a series not analyzed")

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

    rows = len(args.series) * len(args.pools)
    grams = {}
    for name in names:
        size = gradients[(saved[0], args.pools[0])][name].numel()
        gram = np.zeros((rows, rows))
        for start in range(0, size, CHUNK):
            stacked = []
            for spec in args.series:
                for pool in args.pools:
                    head, *rest = terms(spec)
                    value = gradients[(head, pool)][name].flatten()[start:start + CHUNK].double()
                    for term in rest:
                        value = value - gradients[(term, pool)][name].flatten()[start:start + CHUNK].double()
                    stacked.append(value)
            stacked = torch.stack(stacked)
            gram += (stacked @ stacked.T).numpy()
        for group in (group_of(name, muon), "all"):
            grams[group] = grams.get(group, np.zeros((rows, rows))) + gram

    P = len(args.pools)
    result = {}
    for group, gram in grams.items():
        full = statistics(gram, args.series, args.pools, cross, list(range(P)))
        jack = [statistics(gram, args.series, args.pools, cross, [p for p in range(P) if p != drop])
                for drop in range(P)]
        for key, value in full.items():
            values = np.array([j[key] for j in jack])
            se = (float(np.sqrt((P - 1) / P * np.sum((values - values.mean()) ** 2)))
                  if np.all(np.isfinite(values)) else NAN)
            spec, stat = key.split("|")
            entry = result.setdefault(group, {}).setdefault(spec, {})
            entry[stat], entry[f"{stat}_se"] = value, se
    args.output.write_text(json.dumps({"series": args.series, "pools": args.pools,
                                       "cross": [list(pair) for pair in cross],
                                       "groups": result}, indent=2) + "\n")

    for group, specs in result.items():
        print(f"\n[{group}]")
        for spec, entry in specs.items():
            stats = [n for n in entry if not n.endswith("_se")]
            print(f"  {spec}\n    " + "  ".join(
                f"{n} {entry[n]:+.4g}±{entry[n + '_se']:.2g}" for n in stats))


if __name__ == "__main__":
    main()
