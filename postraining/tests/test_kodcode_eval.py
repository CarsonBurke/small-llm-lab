"""Guard executable reward against scratchpad and code-block cherry-picking."""

import shutil

import pytest

from postraining.kodcode_eval import (
    _adapt_row,
    extract_code,
    score_completion,
    summarize,
)
from postraining.vapo.code_reward import PYTHON_REWARD_SCHEMA


def test_unfinished_thinking_is_not_a_submitted_solution():
    assert (
        extract_code("<think>Try this:\n```python\ndef identity(x): return x\n```")
        is None
    )


def test_final_answer_wins_over_correct_scratchpad():
    response = (
        "<think>```python\ndef identity(x): return x\n```</think>\n"
        "```python\ndef identity(x): return 0\n```"
    )
    assert extract_code(response) == "def identity(x): return 0"


def test_last_submitted_block_wins_without_trying_each_candidate():
    response = (
        "```python\ndef identity(x): return x\n```\n"
        "Revised solution:\n```python\ndef identity(x): return 0\n```"
    )
    assert extract_code(response) == "def identity(x): return 0"


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap is unavailable")
def test_scoring_executes_hidden_edge_cases_and_rejects_process_exit():
    verification = {
        "schema": PYTHON_REWARD_SCHEMA,
        "entry_points": ["identity"],
        "test_setup": [],
        "tests": ["assert identity(0) == 0", "assert identity(-3) == -3"],
    }
    assert score_completion("def identity(x): return x", verification) == "pass"
    assert (
        score_completion("def identity(x): return abs(x)", verification)
        == "tests_failed"
    )
    assert score_completion("raise SystemExit(0)", verification) != "pass"


def test_incomplete_groups_do_not_become_all_fail_or_all_pass():
    attempts = [
        {
            "question_id": question,
            "subset": "Prefill",
            "difficulty": "easy",
            "sample_index": index,
            "correct": result == "pass",
            "result": result,
            "text": "response",
            "tokens": 12,
            "terminated": terminated,
        }
        for question, index, result, terminated in [
            ("mixed", 0, "pass", True),
            ("mixed", 1, "format_ineligible", False),
            ("unfinished", 0, "tests_failed", True),
        ]
    ]
    metrics = summarize(attempts, samples_per_problem=2)
    assert metrics["observed_per_attempt_accuracy"] == pytest.approx(1 / 3)
    assert metrics["mixed_outcome_fraction"] == 0.5
    assert metrics["incomplete_fraction"] == 0.5
    assert metrics["all_fail_problem_count"] == 0
    assert metrics["all_pass_problem_count"] == 0
    assert metrics["truncation_rate"] == pytest.approx(1 / 3)
    assert metrics["missing_attempt_count_for_observed_problems"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        summarize(attempts + [attempts[0]], samples_per_problem=2)


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap is unavailable")
def test_adapted_test_alias_runs_every_case_and_never_drops_fixtures():
    source = {
        "question_id": "identity",
        "question": "Return the given integer unchanged.",
        "solution": "def identity(x): return x",
        "test": (
            "from solution import identity as subject\n"
            "def test_zero():\n    assert subject(0) == 0\n"
            "def test_negative():\n    assert subject(-3) == -3\n"
        ),
        "subset": "Prefill",
        "style": "instruct",
        "test_info": [
            {
                "function_name": "identity",
                "function_declaration": "def identity(x):",
            }
        ],
    }
    row, reference = _adapt_row(source)
    assert score_completion(reference, row["verification_info"]) == "pass"
    assert (
        score_completion("def identity(x): return abs(x)", row["verification_info"])
        == "tests_failed"
    )
    source["test"] += "def test_fixture(capsys):\n    assert subject(1) == 1\n"
    with pytest.raises(ValueError):
        _adapt_row(source)
