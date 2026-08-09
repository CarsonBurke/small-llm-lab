"""Deterministic contracts of the few-shot GSM8K harness.

None of this touches a model, so it runs directly rather than through mlq.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pretraining.eval_fewshot_gsm8k import (
    DAPO_PREAMBLE,
    PROMPT_FORMATS,
    build_prompt,
    extract_gold,
    extract_prediction,
    normalize_number,
    select_exemplars,
    truncate_at_stop,
)

HARNESS = PROMPT_FORMATS["harness"]
BARE = PROMPT_FORMATS["bare"]
DAPO = PROMPT_FORMATS["dapo"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1,000", "1000"),
        ("72", "72"),
        ("-5.0", "-5"),
        ("3.5", "3.5"),
        ("0018", "18"),
        ("x", None),
    ],
)
def test_normalize_number_canonicalizes_or_refuses(text, expected):
    assert normalize_number(text) == expected


def test_normalize_number_keeps_integers_exact_above_float_precision():
    # Routing through float first would collapse these onto one value and
    # score a wrong answer as correct.
    assert normalize_number("12345678901234567") != normalize_number(
        "12345678901234568"
    )


def test_normalize_number_refuses_a_repetition_collapse_digit_run():
    assert normalize_number("9" * 309) == "9" * 309
    assert normalize_number("9" * 309 + ".5") is None


def test_prediction_takes_the_first_answer_and_gold_the_last():
    drifted = " The answer is 18.\n#### 18\nAnswer: 18\n#### 4"
    assert extract_prediction(drifted, HARNESS) == "18"
    assert extract_gold(drifted) == "4"


def test_digit_grouping_must_be_well_formed():
    # `#### 1,8` is two numbers the model ran together. Reading it as 1,8 ->
    # 18 would score as correct against a gold of 18.
    assert extract_prediction("#### 1,8", HARNESS) == "1"
    assert extract_prediction("#### 1,000", HARNESS) == "1000"


def test_extraction_refuses_a_window_with_no_delimiter():
    assert extract_prediction(" 2, 2, 2, 2, 2", HARNESS) is None


@pytest.mark.parametrize("stop", HARNESS.stops)
def test_truncate_cuts_at_every_stop_string(stop):
    assert truncate_at_stop(f" 18{stop} and then noise", HARNESS.stops) == " 18"


def test_truncate_cuts_at_the_earliest_stop_not_the_first_listed():
    # `\nAnswer:` appears before `\nQuestion:`; scanning in listed order and
    # taking the first hit would keep the drifted continuation.
    text = " 18\nAnswer: 20\nQuestion: next"
    assert truncate_at_stop(text, HARNESS.stops) == " 18"


def test_truncate_is_identity_without_a_stop():
    assert truncate_at_stop(" 18", HARNESS.stops) == " 18"


def _train_frame(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question": [f"q{i}" for i in range(rows)],
            "answer": [f"work <<{i}+1={i + 1}>>{i + 1}\n#### {i + 1}" for i in range(rows)],
        }
    )


def test_exemplar_draws_are_nested_across_shot_counts():
    train = _train_frame()
    one = select_exemplars(train, 1, 5, seed=0)
    five = select_exemplars(train, 5, 5, seed=0)
    assert five[:1] == one


def test_exemplar_draws_differ_across_seeds():
    train = _train_frame()
    draws = {tuple(select_exemplars(train, 5, 5, seed=s)) for s in range(3)}
    assert len(draws) == 3


def test_prompt_strips_calculator_spans_and_ends_open():
    train = _train_frame()
    prompt = build_prompt([train.iloc[3]], "how many?", HARNESS, strip_calculator=True)
    assert "<<" not in prompt
    assert prompt.endswith("Question: how many?\nAnswer:")
    assert prompt.count("Question:") == 2


def test_prompt_keeps_calculator_spans_when_asked():
    train = _train_frame()
    prompt = build_prompt([train.iloc[3]], "how many?", HARNESS, strip_calculator=False)
    assert "<<3+1=4>>" in prompt


def test_exemplar_solutions_survive_gold_extraction():
    train = _train_frame()
    prompt = build_prompt([train.iloc[3]], "how many?", HARNESS, strip_calculator=True)
    # The exemplar must still demonstrate the answer format it is teaching.
    assert extract_prediction(prompt, HARNESS) == "4"


def test_bare_format_matches_the_openmath_rendering():
    train = _train_frame()
    prompt = build_prompt([train.iloc[3]], "how many?", BARE, strip_calculator=True)
    # `{problem}\n{solution}\nAnswer: {answer}`, then the bare query.
    assert prompt.startswith("q3\nwork 4\nAnswer: 4")
    assert prompt.endswith("how many?\n")
    assert "####" not in prompt
    assert "Question:" not in prompt


def test_dapo_format_wraps_every_block():
    train = _train_frame()
    prompt = build_prompt([train.iloc[3]], "how many?", DAPO, strip_calculator=True)
    assert prompt.count(DAPO_PREAMBLE) == 2
    assert prompt.endswith('after "Answer:".\n')


def test_prediction_delimiter_follows_the_format():
    text = " some work\nAnswer: 42"
    assert extract_prediction(text, BARE) == "42"
    # The harness format looks for `####`, which is absent here.
    assert extract_prediction(text, HARNESS) is None


def test_gold_extraction_is_format_independent():
    # Gold always comes from GSM8K's own `#### N`, whatever the prompt format.
    assert extract_gold("reasoning\n#### 72") == "72"
