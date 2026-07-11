"""Plot AIME 2024 accuracy against VAPO update step."""

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
        item for item in (
            json.loads(line) for line in Path(args.metrics).read_text().splitlines() if line.strip()
        )
        if item.get("type") == "aime" or ("accuracy" in item and set(item) >= {"step", "samples"})
    ]
    entries.sort(key=lambda item: item["step"])
    plt.figure(figsize=(7, 4.5))
    plt.plot([x["step"] for x in entries], [100 * x["accuracy"] for x in entries], marker="o", label="Fresh LeJEPA + VAPO")
    plt.xlabel("Gradient update step")
    plt.ylabel("AIME 2024 avg@32 accuracy (%)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
