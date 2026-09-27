from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from postraining.core import (
    load_unique_math_rows,
    math_corpus_identity,
    math_corpus_policy_sha256,
)
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
)
from postraining.vapo.mixture import _balanced_schedule
from postraining.math_prompt import (
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


def _write_manifest(path: Path, entries: list[dict], **overrides) -> None:
    manifest = {
        "schema": VAPO_MIXTURE_SCHEMA,
        "math_corpus_policy_sha256": math_corpus_policy_sha256(),
        "prompts_per_cycle": sum(entry["rows"] for entry in entries),
        "sources": entries,
        **overrides,
    }
    path.write_text(json.dumps(manifest))


def test_mixture_manifest_binds_bytes_and_sampler_is_exact_pass_and_resumable(
    tmp_path,
) -> None:
    entries = []
    for name, count in (("a", 5), ("b", 3)):
        path = tmp_path / f"{name}.parquet"
        pq.write_table(
            pa.Table.from_pylist([_row(f"{name}{index}") for index in range(count)]),
            path,
        )
        entries.append(
            {
                "name": name,
                "path": str(path),
                "verifier": "math",
                "rows": count,
                "sha256": file_sha256(path),
                "math_corpus_identity": math_corpus_identity(load_unique_math_rows(path)),
            }
        )
    manifest_path = tmp_path / "mixture.json"
    _write_manifest(manifest_path, entries)
    rows, sources, manifest = load_mixture_manifest(manifest_path)
    identity = mixture_identity(manifest_path, manifest)
    assert len(rows) == 8

    sampler = MixedPromptSampler(sources, seed=17, dataset_identity=identity)
    # Every cycle is one exact pass: each row exactly once, whatever the
    # source sizes, and cycle 0 keeps each source's file order.
    cycles = [sampler.next_rows(8) for _ in range(3)]
    for cycle in cycles:
        assert sorted(row["_qualified_identity"] for row in cycle) == sorted(
            row["_qualified_identity"] for row in rows
        )
    assert [
        row["_qualified_identity"] for row in cycles[0] if row["_rl_source"] == "a"
    ] == [f"a:a{index}" for index in range(5)]
    assert cycles[1] != cycles[2]
    resumed = MixedPromptSampler(
        sources, seed=17, dataset_identity=identity, cursor=11
    )
    replay = MixedPromptSampler(sources, seed=17, dataset_identity=identity)
    replay.next_rows(11)
    assert resumed.next_rows(9) == replay.next_rows(9)
    assert resumed.epoch == 2

    # A cycle must be the corpus: a stale total or a v1 quota is rejected.
    _write_manifest(manifest_path, entries, prompts_per_cycle=4)
    with pytest.raises(ValueError, match="prompts_per_cycle"):
        load_mixture_manifest(manifest_path)
    _write_manifest(manifest_path, [{**entries[0], "quota": 3}, entries[1]])
    with pytest.raises(ValueError, match="has a quota"):
        load_mixture_manifest(manifest_path)
    _write_manifest(manifest_path, entries, schema="vapo_verifiable_mixture/v1")
    with pytest.raises(ValueError, match="unsupported VAPO mixture schema"):
        load_mixture_manifest(manifest_path)

    # Byte identity is fail-closed even when the logical rows remain plausible.
    entries[0]["sha256"] = "0" * 64
    _write_manifest(manifest_path, entries)
    with pytest.raises(ValueError, match="bytes differ"):
        load_mixture_manifest(manifest_path)

    entries[0]["sha256"] = file_sha256(entries[0]["path"])
    for missing in (True, False):
        if missing:
            entries[0].pop("math_corpus_identity")
        else:
            entries[0]["math_corpus_identity"] = "stale-effective-corpus"
        manifest["sources"] = entries
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="effective corpus"):
            load_mixture_manifest(manifest_path)
    manifest.pop("math_corpus_policy_sha256")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="corpus policy"):
        load_mixture_manifest(manifest_path)


def test_balanced_schedule_keeps_every_window_proportional() -> None:
    # The production corpus sizes: 71,744 / 2,306 / 17,390.
    counts = {"deepmind_easy": 71_744, "ultradata_math": 2_306, "dapo": 17_390}
    schedule = _balanced_schedule(counts)
    total = len(schedule)
    assert Counter(schedule) == counts
    for name, count in counts.items():
        used = 0
        worst = 0.0
        for position, entry in enumerate(schedule, start=1):
            used += entry == name
            worst = max(worst, abs(used - position * count / total))
        assert worst < 1.0, (name, worst)
    sampler = MixedPromptSampler(
        [
            MixtureSource(name, Path(f"{name}.parquet"), "math", ((),) * count)
            for name, count in counts.items()
        ],
        seed=3,
        dataset_identity="test",
    )
    for start in (0, 12_345, total - 100, total + 7):
        window = sampler.source_counts(start, 256)
        for name, count in counts.items():
            assert abs(window.get(name, 0) - 256 * count / total) < 2.0


def test_empty_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        _balanced_schedule({"a": 3, "b": 0})


def test_explicit_bare_prompt_stays_bare() -> None:
    row = _row("7")
    canonical = canonicalize_answer_fence_rows([row])[0]
    content = canonical["prompt"][0]["content"]
    assert content == "Compute 7."
    assert canonical["extra_info"]["prompt_contract"] == "bare"


def test_knowledge_source_is_opt_in_and_letter_rows_stay_bare() -> None:
    from postraining.prepare_vapo_mixture import SOURCE_SPECS

    specs = {name: (verifier, default) for name, _, verifier, default in SOURCE_SPECS}
    assert specs["ultradata_knowledge"] == ("math", False)
    assert specs["science_mc"] == ("math", False)
    # The builder's single-choice row shape: the option block survives
    # canonicalization, and ``rule`` grades the one letter exactly.
    problem = "Which gas is a noble gas?\n\nA. Argon\nB. Nitrogen\nC. Oxygen"
    row = {
        "prompt": [{"role": "user", "content": problem}],
        "reward_model": {"ground_truth": "A", "style": "rule"},
        "extra_info": {"index": "k1", "module": "choice_3",
                       "prompt_contract": "bare"},
    }
    canonical = canonicalize_answer_fence_rows([row])[0]
    assert canonical["prompt"][0]["content"] == problem
    from postraining.core import verify_answer

    assert verify_answer("Answer: A", "A", "exact", window=None)[0]
    assert not verify_answer("Answer: Argon", "A", "exact", window=None)[0]


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap is unavailable")
def test_python_reward_is_binary_and_executes_all_tests() -> None:
    verification = {
        "schema": PYTHON_REWARD_SCHEMA,
        "entry_points": ["add"],
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
        "entry_points": ["Node", "value"],
        "test_setup": ["root = Node(3)"],
        "tests": ["assert value(root) == 3"],
    }
    code = (
        "class Node:\n    def __init__(self, x): self.data = x\n"
        "def value(n): return n.data"
    )
    assert python_tests_pass(code, verification)


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap is unavailable")
def test_candidate_cannot_rebind_what_the_tests_use() -> None:
    tolerance = {
        "schema": PYTHON_REWARD_SCHEMA,
        "entry_points": ["root"],
        "test_setup": ["import math"],
        "tests": [
            "assert abs(root(2.0) - 1.4142135) < 1e-6",
            "assert math.isclose(root(9.0), 3.0)",
            "assert sorted([root(16.0)]) == [4.0]",
        ],
    }
    assert python_test_result(
        "import math\ndef root(x): return math.sqrt(x)", tolerance
    ) == "pass"
    near_miss = "def root(x): return x / 2"
    assert python_test_result(near_miss, tolerance) == "tests_failed"
    # Each hack below passed the v5 harness, where candidate and tests shared
    # a namespace. Builtin rebinding at module scope, or deferred inside the
    # entry point, now stays in the candidate's namespace.
    tolerance_only = {**tolerance, "tests": tolerance["tests"][:1]}
    sorted_only = {**tolerance, "tests": tolerance["tests"][2:]}
    # The call runs before ``abs`` is looked up, as when a test binds a result.
    called_first = {
        **tolerance,
        "tests": ["value = root(2.0)", "assert abs(value - 1.4142135) < 1e-6"],
    }
    for hack, verification in (
        ("abs = lambda x: 0\n" + near_miss, tolerance_only),
        (
            "def root(x):\n    global abs\n    abs = lambda y: 0\n    return x / 2",
            called_first,
        ),
        ("sorted = lambda x: [4.0]\n" + near_miss, sorted_only),
    ):
        assert python_candidate_allowed(hack)
        assert python_test_result(hack, verification) == "tests_failed"
    # Module patching is rejected statically, and a candidate's module
    # is a private copy even when the store is reached through an alias.
    for patch in (
        "import math\nmath.isclose = lambda a, b: True\n" + near_miss,
        "from collections import Counter\nCounter.most_common = None\n" + near_miss,
        "import math as m\nm.pi = 3\n" + near_miss,
        "import math\nmath.__dict__['isclose'] = 1\n" + near_miss,
    ):
        assert not python_candidate_allowed(patch)
    reads_sqrt = {**tolerance, "tests": ["assert root(16.0) == math.sqrt(16.0)"]}
    aliased = "import math\nm = math\nm.sqrt = lambda x: x / 2\n" + near_miss
    assert python_candidate_allowed(aliased)
    assert python_test_result(aliased, reads_sqrt) == "tests_failed"
    deferred = (
        "import math\nm = math\n"
        "def root(x):\n    m.sqrt = lambda y: y / 2\n    return x / 2"
    )
    assert python_test_result(deferred, reads_sqrt) == "tests_failed"
    # Only the declared entry points reach the tests.
    helper_only = {**tolerance, "entry_points": ["root"],
                   "tests": ["assert helper() == 1"]}
    assert python_test_result(
        "def helper(): return 1\ndef root(x): return x", helper_only
    ) == "tests_failed"
    assert python_test_result("def other(x): return x", tolerance) == "tests_failed"


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap is unavailable")
def test_aliased_library_class_patch_is_detected() -> None:
    verification = {
        "schema": PYTHON_REWARD_SCHEMA,
        "entry_points": ["top"],
        "test_setup": ["from collections import Counter"],
        "tests": ["assert top('abca') == Counter('abca').most_common(1)"],
    }
    assert python_test_result(
        "from collections import Counter\n"
        "def top(s): return Counter(s).most_common(1)",
        verification,
    ) == "pass"
    patched = (
        "from collections import Counter\nC = Counter\n"
        "C.most_common = lambda self, n=None: [('x', 9)]\n"
        "def top(s): return [('x', 9)]"
    )
    assert python_candidate_allowed(patched)
    assert python_test_result(patched, verification) == "tests_failed"


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap is unavailable")
def test_printing_candidates_and_hash_order_are_graded_deterministically() -> None:
    verification = {
        "schema": PYTHON_REWARD_SCHEMA,
        "entry_points": ["f"],
        "test_setup": [],
        "tests": ["assert f() == 1"],
    }
    assert python_test_result("def f():\n    print(1)\n    return 1", verification) == (
        "pass"
    )
    seeded = subprocess.run(
        ["/usr/bin/python3", "-c", "print(hash('abc'))"],
        env={"PYTHONHASHSEED": "0"}, capture_output=True, text=True, check=True,
    ).stdout.strip()
    hashed = {**verification, "tests": [f"assert f() == {seeded}"]}
    for _ in range(3):
        assert python_test_result("def f(): return hash('abc')", hashed) == "pass"


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap is unavailable")
def test_declared_entry_point_may_shadow_a_builtin() -> None:
    verification = {
        "schema": PYTHON_REWARD_SCHEMA,
        "entry_points": ["pow"],
        "test_setup": [],
        "tests": ["assert pow(2, 3) == 8", "assert abs(pow(2, 0) - 1) == 0"],
    }
    loop = (
        "def pow(a, b):\n    result = 1\n    for _ in range(b):\n"
        "        result *= a\n    return result"
    )
    assert python_test_result(loop, verification) == "pass"
    assert python_test_result("def pow(a, b): return 0", verification) == (
        "tests_failed"
    )


def test_python_verifier_requires_entry_points() -> None:
    base = {"schema": PYTHON_REWARD_SCHEMA, "test_setup": [],
            "tests": ["assert f() == 1"]}
    for entry_points in (None, [], ["__builtins__"], ["not an identifier"], "f"):
        info = dict(base)
        if entry_points is not None:
            info["entry_points"] = entry_points
        with pytest.raises(ValueError):
            python_test_result("def f(): return 1", info)


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap is unavailable")
def test_nested_python_scoring_respects_global_sandbox_capacity() -> None:
    verification = {
        "schema": PYTHON_REWARD_SCHEMA,
        "entry_points": ["identity"],
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
