from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmark_qwen3_answer_embeddings.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_qwen3_answer_embeddings", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPT.parent))
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("length", "expected"),
    ((128, 16), (512, 16), (1024, 8), (4096, 2), (8192, 1), (16384, 1)),
)
def test_resolved_batch_size_uses_target_with_singleton_fallback(
    length: int, expected: int
):
    assert MODULE.resolved_batch_size(length, 8192, 16) == expected


def test_resolved_batch_size_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        MODULE.resolved_batch_size(0, 8192, 16)
