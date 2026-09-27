from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from postraining.core import load_unique_math_rows, math_corpus_identity
from postraining.math_prompt import (
    ULTRADATA_FINAL_ANSWER_INSTRUCTION,
    ULTRADATA_REASONING_INSTRUCTION,
)
from postraining.math_rl_pool import (
    canonical_problem,
    load_ultradata_math,
    prompt_token_audit,
    screen_math_pool,
    target_self_verifies,
)

REPO = Path(__file__).resolve().parents[2]

PAIRS = (
    "Compute the number of ordered pairs of integers (x, y) with "
    "1 <= x < y <= 200 such that i^x + i^y is a real number."
)
FORM_CLAUSE = (
    " The answer is in the form \\frac{m}{n}, where gcd(m, n) = 1. Please "
    "provide the value of m + n."
)


def ultradata_row(problem: str, truth: str, identity: str, domain="Math") -> dict:
    content = (
        f"{problem}\n\n{ULTRADATA_REASONING_INSTRUCTION}\n\n"
        f"{ULTRADATA_FINAL_ANSWER_INSTRUCTION}"
    )
    return {
        "prompt": [{"role": "user", "content": content}],
        "reward_model": {"ground_truth": truth, "style": "verifiable_task/v1"},
        "extra_info": {
            "domain": domain,
            "index": identity,
            "original_query_sha256": identity,
        },
    }


class Guard:
    def __init__(self, blocked: dict[str, str]):
        self.blocked = blocked

    def reason(self, problem: str) -> str | None:
        return self.blocked.get(problem)


class Budget:
    budget = 12

    def tokens(self, problem: str) -> int:
        return len(problem.split())


def screen(rows, **kwargs):
    return screen_math_pool(
        rows,
        identity=lambda row: row["extra_info"]["original_query_sha256"],
        guard=kwargs.pop("guard", Guard({})),
        budget=Budget(),
        **kwargs,
    )


def test_canonical_problem_strips_source_framing_and_quarantines_answer_demands():
    assert canonical_problem(ultradata_row("Find x.", "1", "a")) == "Find x."
    with pytest.raises(ValueError, match="Answer:"):
        canonical_problem(ultradata_row("Find x. Answer: 3", "3", "a"))


def test_canonical_problem_joins_messages_like_the_trainer():
    row = ultradata_row("Find x.", "1", "a")
    row["prompt"] = [{"role": "system", "content": "Context."}, *row["prompt"]]
    assert canonical_problem(row) == "Context.\nFind x."


def test_control_bytes_drop_the_row_instead_of_being_repaired():
    rows = [
        ultradata_row("Write it as \x0crac{m}{n}.", "3", "corrupt"),
        ultradata_row("Line one.\r\nLine two\twith tab.", "4", "clean"),
    ]
    result = screen(rows)
    assert [candidate.identity for candidate in result.kept] == ["clean"]
    assert result.dropped == {"control_character": ["corrupt"]}


def test_target_self_verification_uses_the_trainer_call():
    assert target_self_verifies(ultradata_row("Find x.", "\\frac{1}{2}", "a"))
    assert not target_self_verifies(ultradata_row("Find x.", "  ", "a"))


def test_equation_targets_are_quarantined_but_named_values_are_kept():
    rows = [
        ultradata_row("Find the line.", "x + 2y - 5 = 0", "line"),
        ultradata_row("Find all equal.", "a = b = c", "chain"),
        ultradata_row("Find the sum.", "\\sum_{k=1}^{n} k", "subscript"),
        ultradata_row("Find n.", "n = 25", "named"),
        ultradata_row("Find f.", "f(x) = x^2 + 1", "function"),
    ]
    result = screen(rows)
    # An ``=`` inside a subscript is not an equation.
    assert [candidate.identity for candidate in result.kept] == [
        "function",
        "named",
        "subscript",
    ]
    assert result.counts["equation_target"] == 2
    assert [entry["id"] for entry in result.quarantined] == ["line", "chain"]


def test_screens_run_in_order_and_count_every_drop():
    long_problem = " ".join(["word"] * 20)
    rows = [
        ultradata_row("Find x. Answer: 3", "3", "q1"),
        ultradata_row("Find y.", "", "q2"),
        ultradata_row("Leaked eval problem.", "4", "q3"),
        ultradata_row(long_problem, "5", "q4"),
        ultradata_row("Clean problem here.", "6", "q5"),
    ]
    result = screen(rows, guard=Guard({"Leaked eval problem.": "contaminated"}))
    assert [candidate.identity for candidate in result.kept] == ["q5"]
    assert result.counts["uncanonicalizable"] == 1
    assert result.counts["unverifiable_target"] == 1
    assert result.counts["contaminated"] == 1
    assert result.counts["prompt_over_budget"] == 1
    assert result.counts["kept"] == 1
    assert [entry["id"] for entry in result.quarantined] == ["q1", "q2"]
    assert result.dropped == {"contaminated": ["q3"], "prompt_over_budget": ["q4"]}
    # Every problem that reached the budget screen is measured, dropped or not.
    assert sorted(result.prompt_tokens) == [3, 20]


def test_pool_yields_problems_an_owner_restates():
    rows = [
        ultradata_row(PAIRS.replace("200", "100"), "7", "sibling"),
        ultradata_row(PAIRS + FORM_CLAUSE, "9", "restated"),
        ultradata_row(PAIRS.upper(), "8", "verbatim"),
    ]
    result = screen_math_pool(
        rows,
        identity=lambda row: row["extra_info"]["original_query_sha256"],
        guard=Guard({}),
        budget=type("Wide", (), {"budget": 256, "tokens": lambda self, p: 1})(),
        owners={"owner": [("o1", PAIRS)]},
    )
    assert [candidate.identity for candidate in result.kept] == ["sibling"]
    assert result.counts["owned_by:owner"] == 2
    assert result.counts["owned_by:owner:whitespace"] == 1
    assert result.counts["owned_by:owner:shingle"] == 1
    assert {entry["id"]: entry["owner"] for entry in result.dropped["owned_by:owner"]} == {
        "restated": "o1",
        "verbatim": "o1",
    }


WIDE = type("Wide", (), {"budget": 256, "tokens": lambda self, p: 1})()


def screen_wide(rows):
    return screen_math_pool(
        rows,
        identity=lambda row: row["extra_info"]["original_query_sha256"],
        guard=Guard({}),
        budget=WIDE,
    )


def test_same_text_restatements_collapse_or_quarantine_on_conflict():
    result = screen_wide(
        [
            ultradata_row(PAIRS, "\\frac{1}{2}", "b"),
            ultradata_row(PAIRS.replace(" ", "  "), "0.5", "a"),
            ultradata_row("How many primes lie below 50, counting each once?", "15", "c"),
        ]
    )
    assert [candidate.identity for candidate in result.kept] == ["a", "c"]
    assert result.counts["near_duplicate:same_text"] == 1
    assert result.near_duplicates["collapsed"] == [
        {"kept": "a", "dropped": "b", "matcher": "skeleton"}
    ]

    result = screen_wide(
        [
            ultradata_row(PAIRS, "380", "b"),
            ultradata_row(PAIRS.replace(" ", "  "), "382", "a"),
        ]
    )
    assert result.kept == []
    assert result.counts["conflicting_same_text_quarantine"] == 2
    assert result.near_duplicates["conflicting_same_text_groups"] == [
        {"ids": ["a", "b"], "targets": ["382", "380"]}
    ]


def test_answer_form_rewrite_is_kept_as_a_distinct_question():
    result = screen_wide(
        [
            ultradata_row(PAIRS, "\\frac{1}{4}", "a"),
            ultradata_row(PAIRS + FORM_CLAUSE, "5", "b"),
        ]
    )
    assert [candidate.identity for candidate in result.kept] == ["a", "b"]
    assert result.near_duplicates["shingle_pairs_kept_with_distinct_targets"] == 1


def test_shingle_collapse_is_direct_not_transitive():
    first = (
        "a jar holds red green and blue marbles and mia removes marbles one "
        "at a time without replacement"
    )
    second = (
        "every order of draws is equally likely and she stops as soon as any "
        "colour is exhausted from the jar"
    )
    rows = [
        ultradata_row(
            first + " what is the largest number of draws she could possibly "
            "make before her very first stop occurs today", "7", "a"
        ),
        ultradata_row(first + " " + second, "7", "b"),
        ultradata_row(
            second + " what is the probability that the final remaining colour "
            "happens to be the blue marbles in the end", "7", "c"
        ),
    ]
    from postraining.problem_overlap import ProblemOverlapIndex

    problems = [canonical_problem(row) for row in rows]
    index = ProblemOverlapIndex(problems)
    # b restates part of a and part of c; a and c share nothing.
    assert {m.reference for m in index.matches(problems[1], exclude=1)} == {0, 2}
    assert [m.reference for m in index.matches(problems[0], exclude=0)] == [1]
    result = screen_wide(rows)
    # a keeps and absorbs b; c matched only b, so it is not fused into a.
    assert [candidate.identity for candidate in result.kept] == ["a", "c"]
    assert result.counts["near_duplicate:shingle"] == 1


def _write_extraction(directory: Path, rows: list[dict], schema: str) -> None:
    directory.mkdir()
    manifest = {
        "schema": schema,
        "reward_identity": "verifiable_task/v1:test",
        "sources": [
            {"dataset": "openbmb/UltraData-RL-2609", "revision": "rev"}
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    pq.write_table(pa.Table.from_pylist(rows[:2]), directory / "train.parquet")
    pq.write_table(pa.Table.from_pylist(rows[2:]), directory / "validation.parquet")


def test_load_ultradata_math_filters_domain_and_identity(tmp_path):
    missing = ultradata_row("Find z.", "2", "")
    rows = [
        ultradata_row("Find x.", "1", "q1"),
        ultradata_row("Write code.", "1", "q2", domain="Code"),
        ultradata_row("Find x.", "1", "q1"),
        missing,
    ]
    _write_extraction(tmp_path / "extraction", rows, "verifiable_corpus/v1")
    loaded, counts, provenance = load_ultradata_math([tmp_path / "extraction"])
    assert [row["extra_info"]["original_query_sha256"] for row in loaded] == ["q1"]
    assert counts == {
        "extracted": 4,
        "non_math": 1,
        "duplicate_identity": 1,
        "missing_identity": 1,
    }
    assert provenance["source_revision"] == "rev"
    assert len(provenance["input_sha256"]) == 3

    changed = [*rows[:2], ultradata_row("Find x again.", "1", "q1"), missing]
    _write_extraction(tmp_path / "changed", changed, "verifiable_corpus/v1")
    with pytest.raises(ValueError, match="differs"):
        load_ultradata_math([tmp_path / "changed"])

    _write_extraction(tmp_path / "other", rows, "unknown/v9")
    with pytest.raises(ValueError, match="schema"):
        load_ultradata_math([tmp_path / "other"])


def test_prompt_token_audit():
    audit = prompt_token_audit([10, 300, 20, 30], 256)
    assert audit["over_budget"] == 1
    assert audit["over_budget_fraction"] == 0.25
    assert audit["max"] == 300
    assert prompt_token_audit([], 256) == {"measured": 0}


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dapo_rebuild_writes_first_physical_variants_that_reload_identically(tmp_path):
    builder = _load_script("build_dapo_rl_prompts")

    def dapo_row(problem: str, truth: str, index: str) -> dict:
        return {
            "data_source": "math_dapo",
            "prompt": [{"role": "user", "content": problem}],
            "reward_model": {"ground_truth": truth, "style": "rule-lighteval/MATH_v2"},
            "extra_info": {"index": index, "prompt_contract": "bare"},
        }

    physical = [
        dapo_row("Find a.", "1", "i1"),
        dapo_row("Find b.", "2", "i2"),
        dapo_row("Find a.", "1", "i1"),
        dapo_row("Find c.", "3", "i3"),
        dapo_row("Find b.", "2", "i2"),
    ]
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(physical), source)
    effective = [row for row in load_unique_math_rows(source) if row["prompt"][0]["content"] != "Find b."]
    raw = builder.first_physical_rows(
        source, {builder.prompt_key(row["prompt"]) for row in effective}
    )
    rows = [raw[builder.prompt_key(row["prompt"])] for row in effective]
    assert [row["extra_info"]["index"] for row in rows] == ["i1", "i3"]
    output = tmp_path / "pool.parquet"
    pq.write_table(pa.Table.from_pylist(rows), output)
    assert math_corpus_identity(load_unique_math_rows(output)) == math_corpus_identity(
        effective
    )
