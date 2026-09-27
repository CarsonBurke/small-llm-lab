"""Admission contracts for UltraData-Code L3 exercises (SFT pool and RL rows).

Each rejection here is a way an exercise could teach or reward the wrong
thing: a reference the RL policy would reject, a test suite that certifies
nothing, a prompt that never names the function it grades, or an RL prompt
that leaks every graded assertion.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

import postraining.prepare_ultradata_code as code
from postraining.core import ANSWER_CLOSE, ANSWER_OPEN, THINK_CLOSE, THINK_OPEN
from postraining.math_prompt import (
    answer_fence_prompt,
    canonicalize_answer_fence_rows,
)
from postraining.prepare_sft_corpus import (
    ADAPTERS,
    CODE_CONTAINMENT_TARGETS,
    build_containment_index,
    build_document,
    parse_code_exercise,
)
from postraining.prepare_vapo_mixture import SOURCE_SPECS
from postraining.vapo.code_reward import (
    PYTHON_REWARD_SCHEMA,
    normalize_python_answer,
)

TASK = (
    "Write a Python function `add_pairs(a, b)` that returns a list whose i-th "
    "element is a[i] + b[i]. Both lists have the same length."
)
ANALYSIS = "Walk both lists in step with zip and add each pair. O(n) time."
SOLUTION = (
    "def add_pairs(a, b):\n"
    "    out = []\n"
    "    for x, y in zip(a, b):\n"
    "        out.append(x + y)\n"
    "    return out\n"
)
TESTS = (
    "assert add_pairs([], []) == []\n"
    "assert add_pairs([1], [2]) == [3]\n"
    "result = add_pairs([1, 2], [3, 4])\n"
    "assert result == [4, 6]\n"
    "assert add_pairs([-1, 0], [1, 0]) == [0, 0]\n"
)


@pytest.fixture(autouse=True)
def screening_state(monkeypatch):
    from postraining.core import GPT2BPETokenizer

    monkeypatch.setattr(
        code, "_TOKENIZER", GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    )
    monkeypatch.setattr(code, "_MATH_EXACT", set())
    monkeypatch.setattr(code, "_MATH_NGRAMS", set())
    monkeypatch.setattr(code, "_REF_EXACT", set())
    monkeypatch.setattr(code, "_REF_GRAM_TO_IDS", {})
    monkeypatch.setattr(code, "_REF_SIZES", [])


def row(**overrides) -> dict:
    base = {
        "uuid": "u-1",
        "task": TASK,
        "analysis": ANALYSIS,
        "solution": SOLUTION,
        "test": TESTS,
    }
    base.update(overrides)
    return base


def test_spread_covers_the_whole_release() -> None:
    indices = code.spread_shard_indices(19)
    assert indices[0] == 1 and indices[-1] == code.PUBLISHED_SHARDS
    assert indices == sorted(set(indices)) and len(indices) == 19
    assert code.spread_shard_indices(1) == [1]
    with pytest.raises(ValueError):
        code.spread_shard_indices(code.PUBLISHED_SHARDS + 1)


def test_admitted_exercise_keeps_every_statement_and_shows_two() -> None:
    payload, reason = code.screen_exercise(row())
    assert reason == ""
    assert payload["tests"] == [line for line in TESTS.splitlines()]
    assert payload["entry_points"] == ["add_pairs"]
    assert payload["top_level_asserts"] == 4
    assert payload["shown_examples"] == 2
    assert payload["rl_prompt"] == (
        f"{TASK}\n\nExamples:\n"
        "assert add_pairs([], []) == []\nassert add_pairs([1], [2]) == [3]"
    )
    assert payload["rl_prompt_tokens"] + 1 <= code.RL_PROMPT_TOKENS
    assert payload["minhash"].shape == (code.MINHASH_PERMUTATIONS,)


def test_examples_skip_assertions_that_depend_on_earlier_statements() -> None:
    tests = (
        "result = add_pairs([1, 2], [3, 4])\n"
        "assert result == [4, 6]\n"
        "assert add_pairs([5], [5]) == [10]\n"
        "assert len(add_pairs([1], [1])) == 1\n"
    )
    payload, _ = code.screen_exercise(row(test=tests))
    assert payload["rl_prompt"].endswith(
        "Examples:\nassert add_pairs([5], [5]) == [10]\n"
        "assert len(add_pairs([1], [1])) == 1"
    )


def test_rl_prompt_falls_back_to_one_example_then_none() -> None:
    def prompt_tokens(task: str, count: int) -> int:
        examples = [
            "assert add_pairs([], []) == []",
            "assert add_pairs([1], [2]) == [3]",
        ][:count]
        return len(code._TOKENIZER.encode(code.render_rl_prompt(task, examples)))

    # Pad until two examples overflow the budget while one still fits.
    padding = 0
    while prompt_tokens(TASK + " Keep the order." * padding, 2) + 1 <= (
        code.RL_PROMPT_TOKENS
    ):
        padding += 1
    task = TASK + " Keep the order." * padding
    assert prompt_tokens(task, 1) + 1 <= code.RL_PROMPT_TOKENS
    payload, reason = code.screen_exercise(row(task=task))
    assert reason == "" and payload["shown_examples"] == 1
    assert payload["rl_prompt"].endswith("Example:\nassert add_pairs([], []) == []")

    payload, reason = code.screen_exercise(row(task=TASK + " Keep it." * 200))
    # Still admitted for SFT; only the RL prompt is unavailable.
    assert reason == "" and payload["rl_prompt"] is None
    assert payload["shown_examples"] == 0


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        # The RL reward's attribute allowlist rejects ``.startswith``.
        (
            {"solution": "def add_pairs(a, b):\n    return a.startswith(b)\n"},
            "solution_policy_rejected",
        ),
        ({"analysis": "The snippet loops over both lists."}, "meta_reference"),
        ({"analysis": "The original code uses zip."}, "meta_reference"),
        (
            {"task": "Write a function that adds two lists elementwise."},
            "entry_point_not_in_task",
        ),
        (
            {"test": "def add_pairs(a, b):\n    return []\n" + TESTS},
            "test_rebinds_solution_name",
        ),
        ({"test": "import random\n" + TESTS}, "test_import_unsupported"),
        (
            {"test": "assert add_pairs([], []) == []\nassert add_pairs([1], [1])\n"},
            "too_few_top_level_asserts",
        ),
        (
            {"test": "def test_it():\n    assert add_pairs([], []) == []\n"},
            "too_few_top_level_asserts",
        ),
        ({"test": "assert x ==\n"}, "test_syntax_error"),
        ({"test": "assert len([1]) == 1\n" * 3}, "tests_call_no_solution_function"),
        ({"analysis": f"use {THINK_CLOSE} here"}, "fence_literal"),
        (
            {"solution": SOLUTION + "# ```\n"},
            "solution_contains_fence",
        ),
        (
            {"task": TASK + " Answer: give the list."},
            "uncanonicalizable_task",
        ),
        ({"test": ""}, "empty_test"),
    ],
)
def test_rejections(overrides: dict, reason: str) -> None:
    payload, got = code.screen_exercise(row(**overrides))
    assert payload is None
    assert got == reason


def test_function_locals_do_not_collide_with_test_bindings() -> None:
    # ``out`` is local to the solution function; a test variable of the same
    # name is not a rebinding of anything the solution exports.
    tests = "out = add_pairs([1], [1])\n" + TESTS
    payload, reason = code.screen_exercise(row(test=tests))
    assert reason == "" and payload is not None


def test_reimporting_a_solution_import_is_not_a_rebinding() -> None:
    solution = "import math\n" + SOLUTION
    tests = "import math\n" + TESTS
    payload, reason = code.screen_exercise(row(solution=solution, test=tests))
    assert reason == "" and payload is not None


def test_reference_containment_drops_evaluation_restatements(monkeypatch) -> None:
    reference = (
        "Write a function that returns the elementwise sum of two lists of "
        "equal length as a new list of integers."
    )
    gram_to_ids, sizes = build_containment_index([reference])
    monkeypatch.setattr(code, "_REF_GRAM_TO_IDS", gram_to_ids)
    monkeypatch.setattr(code, "_REF_SIZES", sizes)
    payload, reason = code.screen_exercise(
        row(task=f"{reference} Name it `add_pairs(a, b)`.")
    )
    assert payload is None and reason == "contains_reference_problem"


def test_code_targets_cover_mbpp_and_humaneval() -> None:
    names = {path.name for path, _ in CODE_CONTAINMENT_TARGETS}
    assert names == {"mbpp-problems.parquet", "humaneval-problems.parquet"}
    assert ADAPTERS["ultradata_code_l3"].containment_targets == (
        CODE_CONTAINMENT_TARGETS
    )


def test_stub_replaces_function_bodies_and_keeps_everything_else() -> None:
    stub = code.stub_module(
        "import math\nLIMIT = 3\n" + SOLUTION + "class Box:\n    size = 1\n"
    )
    assert stub == (
        "import math\nLIMIT = 3\n\ndef add_pairs(a, b):\n    return None\n\n"
        "class Box:\n    size = 1"
    )


def test_decorated_test_helpers_keep_their_decorators() -> None:
    tests = (
        "import functools\n"
        "@functools.lru_cache(maxsize=None)\n"
        "def expected(n):\n"
        "    return [n + n]\n"
        + TESTS
        + "assert add_pairs([7], [7]) == expected(7)\n"
    )
    payload, reason = code.screen_exercise(row(test=tests))
    assert reason == ""
    assert payload["tests"][1] == (
        "@functools.lru_cache(maxsize=None)\ndef expected(n):\n    return [n + n]"
    )


def test_tests_may_not_read_names_only_the_solution_defines() -> None:
    solution = "import math\nSCALE = 1\n" + SOLUTION
    tests = TESTS + "assert math.floor(add_pairs([1.5], [0])[0]) == SCALE\n"
    payload, reason = code.screen_exercise(row(solution=solution, test=tests))
    assert payload is None and reason == "test_reads_solution_name"
    payload, reason = code.screen_exercise(
        row(test=TESTS + "assert add_pairs([1], [1]) == missing\n")
    )
    assert payload is None and reason == "test_unbound_name"
    # The test's own import, helper, loop variable and handler all resolve.
    own = (
        "import math\n"
        "def total(xs):\n    return sum(x for x in xs)\n"
        "for n in range(2):\n    assert add_pairs([n], [n]) == [2 * n]\n"
        "try:\n    add_pairs([1], [])\nexcept Exception as error:\n    pass\n"
        + TESTS
        + "assert total(add_pairs([1], [1])) == math.floor(2.5)\n"
    )
    payload, reason = code.screen_exercise(row(test=own))
    assert reason == "" and payload is not None


def test_repeated_asserts_count_once() -> None:
    tests = "assert add_pairs([], []) == []\n" * 3 + "assert add_pairs([1], [2]) == [3]\n"
    payload, reason = code.screen_exercise(row(test=tests))
    assert payload is None and reason == "too_few_top_level_asserts"


def test_rl_prompt_requires_a_hidden_call_no_example_reveals() -> None:
    # Every graded call appears in a shown example: returning the shown
    # outputs would pass, so the row stays SFT-only.
    tests = (
        "assert add_pairs([1], [2]) == [3]\n"
        "assert add_pairs([1], [2]) != [4]\n"
        "assert len(add_pairs([1], [2])) == 1\n"
    )
    payload, reason = code.screen_exercise(row(test=tests))
    assert reason == "" and payload["rl_prompt"] is None
    # With one example shown, the other call is hidden and discriminates.
    tests = (
        "assert add_pairs([], []) == []\n"
        "assert add_pairs([1], [2]) == [3]\n"
        "assert add_pairs([1], [2]) != [4]\n"
    )
    payload, reason = code.screen_exercise(row(test=tests))
    assert reason == "" and payload["shown_examples"] == 1


def test_near_duplicates_keep_the_earliest_row_of_each_cluster() -> None:
    base = code.minhash(TASK)
    near = code.minhash(TASK + " Return a new list.")
    far = code.minhash(
        "Implement Dijkstra's shortest path over an adjacency list and return "
        "the distance to every vertex from the source vertex."
    )
    keep, joined = code.near_duplicate_representatives(np.stack([base, far, near]))
    assert keep.tolist() == [True, True, False]
    assert joined == 1


def test_near_duplicate_clusters_are_transitive() -> None:
    n = code.MINHASH_PERMUTATIONS
    cut = int(0.45 * n)
    signature = np.arange(n, dtype=np.uint32)
    b = signature.copy()
    b[:cut] += 1000  # agreement with a: 55%
    c = b.copy()
    c[cut : 2 * cut] += 1000  # agreement with b: 55%, with a: 10%
    assert np.mean(signature == c) < code.NEAR_DUPLICATE_JACCARD
    keep, _ = code.near_duplicate_representatives(np.stack([signature, b, c]))
    assert keep.tolist() == [True, False, False]


def admitted() -> dict:
    payload, reason = code.screen_exercise(row())
    assert reason == ""
    return payload


def test_rl_row_matches_the_mbpp_schema_and_canonicalizes() -> None:
    item = admitted()
    rl = code.rl_row(item)
    assert set(rl) == {
        "data_source", "prompt", "ability", "reward_model", "extra_info",
        "verification_info",
    }
    assert rl["verification_info"] == {
        "schema": PYTHON_REWARD_SCHEMA,
        "entry_points": item["entry_points"],
        "test_setup": [],
        "tests": item["tests"],
    }
    assert rl["extra_info"]["prompt_contract"] == "bare"
    # Canonicalization is the identity on these prompts.
    content = rl["prompt"][0]["content"]
    assert answer_fence_prompt(content) == content
    assert canonicalize_answer_fence_rows([rl])[0]["prompt"] == rl["prompt"]
    # Grading covers strictly more than the prompt reveals.
    shown = content.split("Examples:\n", 1)[1].splitlines()
    assert set(shown) < set(rl["verification_info"]["tests"])


def test_pools_share_no_problem_identity_and_cap_repeats() -> None:
    def item(name: str, prompt: bool = True) -> dict:
        return {"entry_points": [name], "rl_prompt": "p" if prompt else None}

    assigner = code.PoolAssigner(rl_rows=2, sft_rows=4)
    order = [
        item("factorial"),  # opens RL for factorial
        item("Factorial"),  # same identity, RL cap reached: dropped, not SFT
        item("fib", prompt=False),  # no RL prompt: fib is SFT for good
        item("fib"),  # follows its identity into SFT
        item("gcd"),  # RL has room
        item("lcm"),  # RL full: SFT
        item("fib"),
        item("fib"),  # fourth fib: SFT cap
    ]
    got = [assigner.offer(row) for row in order]
    assert got == ["rl", "dropped", "sft", "sft", "rl", "sft", "sft", "dropped"]
    assert assigner.full
    assert assigner.dropped == {"identity_repeat_rl": 1, "identity_repeat_sft": 1}
    rl = {code.problem_identity(row) for row in assigner.rl}
    assert not rl & {code.problem_identity(row) for row in assigner.sft}


def test_sft_pool_stops_at_its_target_even_before_rl_fills() -> None:
    assigner = code.PoolAssigner(rl_rows=5, sft_rows=2)
    rows = [{"entry_points": [f"f{i}"], "rl_prompt": None} for i in range(4)]
    assert [assigner.offer(row) for row in rows] == ["sft", "sft", "dropped", "dropped"]
    assert assigner.dropped["sft_full"] == 2 and len(assigner.sft) == 2
    # New prompt-less identities are skipped unsandboxed; RL-eligible are not.
    assert assigner.saturated({"entry_points": ["g"], "rl_prompt": None})
    assert not assigner.saturated({"entry_points": ["g"], "rl_prompt": "p"})


def test_cross_pool_twins_are_found_below_the_dedupe_threshold() -> None:
    rl = np.stack([code.minhash(TASK)])
    n = code.MINHASH_PERMUTATIONS
    twin = rl[0].copy()
    twin[: int(0.6 * n)] += 7  # 40% agreement: below dedupe, above cross-pool
    far = code.minhash("Implement Dijkstra's shortest path over an adjacency list.")
    assert code.near_twins(np.stack([twin, far]), rl).tolist() == [True, False]


def test_decorator_recovery_ignores_form_feeds_in_earlier_lines() -> None:
    tests = (
        "PAD = 'a\x0cb'\n"
        "@staticmethod\n"
        "def expected(n):\n"
        "    return [n + n]\n"
        + TESTS
        + "assert add_pairs([7], [7]) == expected(7)\n"
    )
    payload, reason = code.screen_exercise(row(test=tests))
    assert reason == ""
    assert payload["tests"][1].startswith("@staticmethod\ndef expected(n):")


def test_saturated_rows_are_exactly_those_offer_would_drop() -> None:
    assigner = code.PoolAssigner(rl_rows=1, sft_rows=10)
    rl = {"entry_points": ["gcd"], "rl_prompt": "p"}
    sft = {"entry_points": ["fib"], "rl_prompt": None}
    assert not assigner.saturated(rl)
    assigner.offer(rl)
    assert assigner.saturated(rl)
    for _ in range(code.SFT_PER_IDENTITY):
        assert not assigner.saturated(sft)
        assert assigner.offer(sft) == "sft"
    assert assigner.saturated(sft) and assigner.offer(sft) == "dropped"


def test_packed_body_round_trips() -> None:
    payload = admitted()
    original = dict(payload)
    payload["_body"] = code.pack_body(payload)
    assert not set(code.BODY_FIELDS) & set(payload)
    restored = code.unpack_body(payload)
    assert {key: restored[key] for key in code.BODY_FIELDS} == {
        key: original[key] for key in code.BODY_FIELDS
    }


def test_trainer_refuses_a_partly_verified_corpus(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from postraining.sft_trace_train import load_documents

    path = tmp_path / "mixed.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"problem": "p1", "document": "p1 d", "verified": True},
                {"problem": "p2", "document": "p2 d", "verified": False},
            ]
        ),
        path,
    )
    # Filtering would silently train on the verified code slice alone.
    with pytest.raises(ValueError, match="mixes 1 verified and 1 unverified"):
        load_documents(path)
    assert len(load_documents(path, allow_unverified=True)) == 2


def test_sft_adapter_renders_the_canonical_answer_fence_document() -> None:
    adapter = ADAPTERS["ultradata_code_l3"]
    record, reason = parse_code_exercise(
        adapter,
        {"problem": TASK, "analysis": ANALYSIS, "solution": SOLUTION,
         "provenance": "L3/py"},
    )
    assert reason == ""
    assert not record.gradeable and record.multiline_answer and record.verified
    document = build_document(record.problem, record.solution, record.answer)
    assert document == (
        f"{TASK}{THINK_OPEN}\n{ANALYSIS}\n{THINK_CLOSE}\n{ANSWER_OPEN}"
        f"```python\n{SOLUTION.strip()}\n```{ANSWER_CLOSE}"
    )
    # The <answer> span is exactly what the Python reward normalises.
    assert normalize_python_answer(record.answer) == SOLUTION.strip()
    assert adapter.known_sources == frozenset({"L3/py"})


def test_sft_adapter_rejects_nested_fences_and_missing_provenance() -> None:
    adapter = ADAPTERS["ultradata_code_l3"]
    base = {"problem": TASK, "analysis": ANALYSIS, "solution": SOLUTION,
            "provenance": "L3/py"}
    assert parse_code_exercise(adapter, {**base, "solution": "x = '```'"})[1] == (
        "program_contains_fence"
    )
    assert parse_code_exercise(adapter, {**base, "provenance": ""})[1] == (
        "missing_provenance"
    )


def test_rl_source_is_selectable_and_off_by_default() -> None:
    specs = {name: (path, verifier, on) for name, path, verifier, on in SOURCE_SPECS}
    path, verifier, on = specs["ultradata_code_l3"]
    assert verifier == "python_mbpp" and on is False
    assert path.name == "ultradata-code-l3-v2-rl.parquet"


needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None, reason="bwrap is unavailable"
)


@needs_bwrap
def test_verification_accepts_an_agreeing_reference() -> None:
    verdict = code.verify_exercise(admitted())
    assert verdict["verdict"] == "verified"
    assert len(verdict["walls"]) == 2


@needs_bwrap
def test_verification_rejects_a_disagreeing_test() -> None:
    item = admitted()
    item["tests"] = [*item["tests"], "assert add_pairs([1], [1]) == [3]"]
    assert code.verify_exercise(item)["verdict"] == "reference_tests_failed"


@needs_bwrap
def test_verification_rejects_a_suite_a_stub_also_passes() -> None:
    item = admitted()
    item["tests"] = [
        "add_pairs([1], [2])",
        "assert True",
        "assert 1 == 1",
        "assert [] == []",
    ]
    assert code.verify_exercise(item)["verdict"] == "stub_passes"
