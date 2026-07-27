"""Plot native stop-thinking-policy AIME accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="postraining/runs/fresh_lejepa_vapo/metrics.jsonl")
    parser.add_argument("--output", default="postraining/runs/fresh_lejepa_vapo/aime_accuracy_vs_step.png")
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    entries = [
        item
        for item in (
            json.loads(line)
            for line in Path(args.metrics).read_text().splitlines()
            if line.strip()
        )
        if item.get("type") == "aime"
        or ("accuracy" in item and set(item) >= {"step", "samples"})
    ]
    entries.sort(key=lambda item: item["step"])
    plt.figure(figsize=(7, 4.5))
    native = [item for item in entries if "policy_accuracy" in item]
    if native:
        plt.plot(
            [item["step"] for item in native],
            [100 * item["policy_accuracy"] for item in native],
            marker="o",
            label="Native policy",
        )
    all_rollouts = [item for item in entries if "policy_accuracy" not in item]
    if all_rollouts:
        plt.plot(
            [item["step"] for item in all_rollouts],
            [100 * item["accuracy"] for item in all_rollouts],
            marker="o",
            label="All rollouts",
        )
    plt.xlabel("Gradient update step")
    plt.ylabel("AIME 2024 accuracy (%)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
