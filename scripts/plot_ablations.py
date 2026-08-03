"""
Plot ablation results from ablation_results/ or tensorboard logs.

Usage:
    python3 -m scripts.plot_ablations                # Plot all runs
    python3 -m scripts.plot_ablations baseline_*     # Plot matching runs
    python3 -m scripts.plot_ablations --metric bpb   # Plot BPB (default)
    python3 -m scripts.plot_ablations --metric loss  # Plot val loss
"""

from __future__ import annotations

import argparse
import json
import sys
from fnmatch import fnmatch
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[1] / "ablation_results"


def load_results(patterns: list[str] | None = None) -> list[dict]:
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if patterns and not any(fnmatch(f.stem, p) for p in patterns):
            continue
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def plot_training_curves(results: list[dict], metric: str = "bpb"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: validation metric over steps
    ax = axes[0]
    for r in results:
        val_entries = r.get("val_entries", [])
        if not val_entries:
            continue
        steps = [e["step"] for e in val_entries]
        if metric == "bpb":
            values = [e["val_bpb"] for e in val_entries]
            ylabel = "Val BPB"
        else:
            values = [e["val_loss"] for e in val_entries]
            ylabel = "Val Loss"
        label = f"{r['name']} ({values[-1]:.4f})" if values else r['name']
        ax.plot(steps, values, marker="o", markersize=3, label=label)

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Validation {metric.upper()} vs Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: bar chart of final BPB
    ax = axes[1]
    names = []
    bpbs = []
    for r in sorted(results, key=lambda x: x.get("final_val_bpb") or 99):
        if r.get("final_val_bpb"):
            names.append(r["name"])
            bpbs.append(r["final_val_bpb"])

    if names:
        colors = plt.cm.viridis([i / max(len(names) - 1, 1) for i in range(len(names))])
        bars = ax.barh(range(len(names)), bpbs, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Final Val BPB")
        ax.set_title("Final BPB Comparison")
        ax.grid(True, alpha=0.3, axis="x")
        # Add value labels
        for bar, val in zip(bars, bpbs):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    out_path = RESULTS_DIR / "comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="*", help="Glob patterns to filter run names")
    parser.add_argument("--metric", default="bpb", choices=["bpb", "loss"])
    args = parser.parse_args()

    results = load_results(args.patterns or None)
    if not results:
        print("No results found in ablation_results/")
        sys.exit(1)

    print(f"Loaded {len(results)} run(s)")
    plot_training_curves(results, args.metric)


if __name__ == "__main__":
    main()
