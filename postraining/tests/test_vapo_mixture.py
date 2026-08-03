from __future__ import annotations

import json
import shutil
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from postraining.vapo.code_reward import (
    PYTHON_REWARD_SCHEMA,
    normalize_python_answer,
    python_candidate_allowed,
    python_tests_pass,
)
from postraining.vapo.mixture import (
    VAPO_MIXTURE_SCHEMA,
    MixedPromptSampler,
    file_sha256,
    load_mixture_manifest,
    mixture_identity,
)
from postraining.math_prompt import (
    ANSWER_FENCE_INSTRUCTION,
    canonicalize_answer_fence_rows,
)


def _row(identity: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": f"Compute {identity}."}],
        "reward_model": {"ground_truth": identity, "style": "minerva"},
        "extra_info": {
            "index": identity,
            "module": "test",
            "prompt_contract": "bare",
        },
    }


def test_mixture_manifest_binds_bytes_and_sampler_is_exact_and_resumable(
    tmp_path,
) -> None:
    entries = []
    for name, quota, count in (("a", 3, 5), ("b", 1, 3)):
        path = tmp_path / f"{name}.parquet"
        pq.write_table(
            pa.Table.from_pylist([_row(f"{name}{index}") for index in range(count)]),
            path,
        )
        entries.append(
            {
                "name": name,
                "path": str(path),
                "quota": quota,
                "verifier": "math",
                "rows": count,
                "sha256": file_sha256(path),
            }
        )
    manifest_path = tmp_path / "mixture.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": VAPO_MIXTURE_SCHEMA,
                "groups_per_cycle": 4,
                "sources": entries,
            }
        )
    )
    rows, sources, manifest = load_mixture_manifest(manifest_path)
    identity = mixture_identity(manifest_path, manifest)
    assert len(rows) == 8

    sampler = MixedPromptSampler(sources, seed=17, dataset_identity=identity)
    first = sampler.next_rows(12)
    assert Counter(row["_rl_source"] for row in first) == {"a": 9, "b": 3}
    resumed = MixedPromptSampler(
        sources, seed=17, dataset_identity=identity, cursor=12
    )
    assert resumed.next_rows(16) == sampler.next_rows(16)

    # Byte identity is fail-closed even when the logical rows remain plausible.
    entries[0]["sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(
            {
                "schema": VAPO_MIXTURE_SCHEMA,
                "groups_per_cycle": 4,
                "sources": entries,
            }
        )
    )
    with pytest.raises(ValueError, match="bytes differ"):
        load_mixture_manifest(manifest_path)


def test_explicit_bare_prompt_gets_one_canonical_contract() -> None:
    row = _row("7")
    canonical = canonicalize_answer_fence_rows([row])[0]
    content = canonical["prompt"][0]["content"]
    assert content == f"Compute 7.\n\n{ANSWER_FENCE_INSTRUCTION}"
    assert content.count(ANSWER_FENCE_INSTRUCTION) == 1


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap is unavailable")
def test_python_reward_is_binary_and_executes_all_tests() -> None:
    verification = {
        "schema": PYTHON_REWARD_SCHEMA,
        "test_setup": [],
        "tests": ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"],
    }
    assert python_tests_pass("def add(a, b):\n    return a + b", verification)
    assert not python_tests_pass("def add(a, b):\n    return a - b", verification)
    assert not python_tests_pass("raise SystemExit(0)", verification)
    assert not python_tests_pass(
        "import os\nos._exit(0)", verification
    )
    sentinel_theft = """\
import os, sys
frame = sys._getframe()
while frame:
    for value in frame.f_code.co_consts:
        if isinstance(value, bytes) and len(value) == 32:
            os.write(1, value)
            os._exit(0)
    frame = frame.f_back
"""
    assert not python_candidate_allowed(sentinel_theft)
    assert not python_tests_pass(sentinel_theft, verification)
    traceback_theft = """\
try:
    1 / 0
except Exception as error:
    frame = error.__traceback__.tb_frame.f_back
"""
    assert not python_candidate_allowed(traceback_theft)
    alias_theft = "import sys\ns = sys\nf = s._getframe().f_back"
    assert not python_candidate_allowed(alias_theft)
    getattr_alias = "g = getattr\ng(object(), '__class__')"
    assert not python_candidate_allowed(getattr_alias)
    assert python_candidate_allowed(
        "import math\nfrom sys import maxsize\ndef f(x): return math.sqrt(x) + maxsize"
    )
    assert normalize_python_answer("```python\ndef f():\n    pass\n```") == (
        "def f():\n    pass"
    )
