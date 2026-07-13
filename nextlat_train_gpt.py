"""Baseline pretraining with only the NextLat auxiliary objective.

Isolates change 2 of ``nextlat_pope_train_gpt.py``: the NextLat next-latent
auxiliary at its method defaults, on the otherwise unmodified baseline model
(stock RoPE attention).  See that module for the port provenance and details.
"""

from __future__ import annotations

import train_gpt as baseline
from nextlat_pope_train_gpt import NextLatGPT


def main() -> None:
    original_gpt = baseline.GPT
    baseline.GPT = NextLatGPT
    try:
        baseline.main()
    finally:
        baseline.GPT = original_gpt


if __name__ == "__main__":
    main()
