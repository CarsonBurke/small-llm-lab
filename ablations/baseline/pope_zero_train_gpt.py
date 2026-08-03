"""Baseline pretraining with only the PoPE attention change.

Isolates change 1 of ``ablations/baseline/nextlat_pope_train_gpt.py``: PoPE Q/K geometry with the
zero phase-offset initialization, on the otherwise unmodified baseline model
and loss.  See that module for the port provenance and details.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import train_gpt as baseline
from ablations.baseline.nextlat_pope_train_gpt import PolarCausalSelfAttention


def main() -> None:
    original_attention = baseline.CausalSelfAttention
    original_patterns = baseline.CONTROL_TENSOR_NAME_PATTERNS
    original_int8_patterns = baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS
    baseline.CausalSelfAttention = PolarCausalSelfAttention
    baseline.CONTROL_TENSOR_NAME_PATTERNS = original_patterns + ("delta_c",)
    baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = original_int8_patterns + ("delta_c",)
    try:
        baseline.main()
    finally:
        baseline.CausalSelfAttention = original_attention
        baseline.CONTROL_TENSOR_NAME_PATTERNS = original_patterns
        baseline.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = original_int8_patterns


if __name__ == "__main__":
    main()
