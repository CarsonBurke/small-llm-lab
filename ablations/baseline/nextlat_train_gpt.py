"""Baseline pretraining with only the NextLat auxiliary objective.

Isolates change 2 of ``ablations/baseline/nextlat_pope_train_gpt.py``: the NextLat next-latent
auxiliary at its method defaults, on the otherwise unmodified baseline model
(stock RoPE attention).  See that module for the port provenance and details.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import train_gpt as baseline
from ablations.baseline.nextlat_pope_train_gpt import NextLatGPT


def main() -> None:
    original_gpt = baseline.GPT
    baseline.GPT = NextLatGPT
    try:
        baseline.main()
    finally:
        baseline.GPT = original_gpt


if __name__ == "__main__":
    main()
