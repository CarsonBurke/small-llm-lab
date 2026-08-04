from __future__ import annotations

import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from postraining.vapo.code_reward import (
    PYTHON_REWARD_SCHEMA,
    batch_python_test_results,
    normalize_python_answer,
    python_candidate_allowed,
    python_test_result,
    python_tests_pass,
)
from postraining.vapo.mixture import (
    VAPO_MIXTURE_SCHEMA,
    MixtureSource,
    MixedPromptSampler,
    file_sha256,
    load_mixture_manifest,
    mixture_identity,
    rollout_window_source_quotas,
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


def test_rollout_windows_preserve_exact_source_composition() -> None:
    def source(name: str, quota: int) -> MixtureSource:
        return MixtureSource(name, Path(f"{name}.parquet"), quota, "math", ())

    balanced = [source("a", 2), source("b", 2)]
    assert rollout_window_source_quotas(balanced, 2) == [1, 1]
    assert rollout_window_source_quotas(balanced, 4) == [2, 2]

    broad = [
        source("dapo", 28),
        source("deepmind", 20),
        source("gsm8k", 8),
        source("mbpp", 8),
    ]
    assert rollout_window_source_quotas(broad, 16) == [7, 5, 2, 2]
    with pytest.raises(ValueError, match="identical source quotas"):
        rollout_window_source_quotas(broad, 24)
    assert (
        rollout_window_source_quotas(
            broad, 24, allow_balanced_rotation=True, start_cursor=32
        )
        is None
    )
    with pytest.raises(ValueError, match="sampler cursor 3"):
        rollout_window_source_quotas(
            broad, 24, allow_balanced_rotation=True, start_cursor=3
        )
    broad_sampler = MixedPromptSampler(
        broad, seed=3, dataset_identity="test"
    )
    assert broad_sampler.source_counts(0, 24) == {
        "dapo": 10, "deepmind": 8, "gsm8k": 3, "mbpp": 3,
    }
    assert broad_sampler.source_counts(24, 24) == {
        "dapo": 11, "deepmind": 7, "gsm8k": 3, "mbpp": 3,
    }

    uneven = [source("a", 3), source("b", 1)]
    with pytest.raises(ValueError, match="identical source quotas"):
        rollout_window_source_quotas(uneven, 2)
    with pytest.raises(ValueError, match="identical source quotas"):
        rollout_window_source_quotas(balanced, 3)

    shifted = [source("a", 2), source("b", 2), source("c", 8)]
    shifted_quotas = rollout_window_source_quotas(shifted, 6)
    assert shifted_quotas == [1, 1, 4]
    shifted_sampler = MixedPromptSampler(
        shifted, seed=3, dataset_identity="test", cursor=4
    )
    with pytest.raises(ValueError, match="sampler cursor 4"):
        shifted_sampler.validate_next_source_quotas(shifted_quotas)


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
    magic_equality = """\
class AlwaysEqual:
    def __eq__(self, other):
        return True
def add(a, b):
    return AlwaysEqual()
"""
    assert not python_candidate_allowed(magic_equality)
    assert not python_tests_pass(magic_equality, verification)
    generator_frame = """\
g = None
def gen():
    yield g.gi_frame.f_back
g = gen()
frame = next(g)
"""
    assert not python_candidate_allowed(generator_frame)
    magic_assignment = """\
class AlwaysContains:
    pass
AlwaysContains.__contains__ = lambda self, value: True
"""
    assert not python_candidate_allowed(magic_assignment)
    assert python_candidate_allowed(
        "import math\nfrom sys import maxsize\ndef f(x): return math.sqrt(x) + maxsize"
    )
    assert python_test_result(
        "def add(a, b): return a + b", verification
    ) == "pass"
    assert python_test_result(
        "raise SystemExit(0)", verification
    ) == "policy_rejected"


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap is unavailable")
def test_python_fixture_runs_after_candidate_definitions() -> None:
    verification = {
        "schema": PYTHON_REWARD_SCHEMA,
        "test_setup": ["root = Node(3)"],
        "tests": ["assert value(root) == 3"],
    }
    code = (
        "class Node:\n    def __init__(self, x): self.data = x\n"
        "def value(n): return n.data"
    )
    assert python_tests_pass(code, verification)


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap is unavailable")
def test_nested_python_scoring_respects_global_sandbox_capacity() -> None:
    verification = {
        "schema": PYTHON_REWARD_SCHEMA,
        "test_setup": [],
        "tests": ["assert identity(3) == 3"],
    }
    answers = ["def identity(x): return x"] * 8
    with ThreadPoolExecutor(max_workers=4) as pool:
        nested = list(
            pool.map(
                lambda _: batch_python_test_results(answers, verification),
                range(4),
            )
        )
    assert nested == [["pass"] * 8] * 4
    assert normalize_python_answer("```python\ndef f():\n    pass\n```") == (
        "def f():\n    pass"
    )
