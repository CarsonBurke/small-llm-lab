"""Fail-closed contracts for the UltraData chat adapter and the gate panel.

Every case here is a way a malformed or ungradeable row could reach training
data or the sampling gate. They are unit tests rather than corpus assertions
because the corpus is multi-gigabyte and its bad rows are rare: the empty
``\\boxed{}`` case below occurs at roughly one row in a thousand, which a
spot check would miss and a full pass would take an hour to find.
"""

from __future__ import annotations

import pytest

from postraining.prepare_sft_corpus import (
    ADAPTERS,
    excluded,
    known,
    parse_ultradata_chat,
)
from postraining.sft_trace_train import split_holdout

ADAPTER = ADAPTERS["ultradata_sft_2605"]


def record(content: str, *, reasoning: str = "work", domain: str = "Math",
           source: str = "OpenMathInstruct-2", think: str = "think") -> dict:
    return {
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": content,
             "reasoning_content": reasoning},
        ],
        "source": source,
        "domain": domain,
        "think_type": think,
    }


@pytest.mark.parametrize("body", ["", " ", "\n"])
def test_empty_boxed_answer_is_rejected(body: str) -> None:
    """``\\boxed{}`` must not become a zero-width <answer></answer> span.

    ``last_boxed_answer`` returns the stripped body, so an empty box yields
    "" rather than None; a None-only check would admit it and teach the
    model an answer span the anchored gate rejects.
    """

    parsed, reason = parse_ultradata_chat(
        ADAPTER, record(f"Therefore \\boxed{{{body}}}")
    )
    assert parsed is None
    assert reason == "no_boxed_answer"


def test_boxed_answer_is_taken_from_the_last_balanced_span() -> None:
    parsed, reason = parse_ultradata_chat(
        ADAPTER, record("First \\boxed{1}, corrected to \\boxed{\\frac{3}{4}}")
    )
    assert reason == ""
    assert parsed is not None
    assert parsed.answer == "\\frac{3}{4}"
    assert parsed.gradeable is True
    assert parsed.solution == "work"


def test_no_think_split_is_rejected() -> None:
    """The contract needs a reasoning span; no_think rows carry none."""

    parsed, reason = parse_ultradata_chat(
        ADAPTER, record("\\boxed{4}", think="no_think")
    )
    assert parsed is None
    assert reason == "not_think_split"


def test_missing_provenance_is_rejected() -> None:
    parsed, reason = parse_ultradata_chat(ADAPTER, record("\\boxed{4}", source=""))
    assert parsed is None
    assert reason == "missing_provenance"


def test_code_rows_are_multiline_and_ungradeable() -> None:
    """A program is graded by executing tests, not by answer comparison."""

    content = "Here it is:\n```python\ndef f():\n    return 1\n```"
    parsed, reason = parse_ultradata_chat(
        ADAPTER, record(content, domain="Code", source="OpenCodeReasoning")
    )
    assert reason == ""
    assert parsed is not None
    assert parsed.gradeable is False
    assert parsed.multiline_answer is True
    assert parsed.answer == "```python\ndef f():\n    return 1\n```"
    assert parsed.origin == "Code/OpenCodeReasoning"


def test_multi_turn_records_are_rejected() -> None:
    row = record("\\boxed{4}")
    row["messages"].insert(0, {"role": "system", "content": "be helpful"})
    parsed, reason = parse_ultradata_chat(ADAPTER, row)
    assert parsed is None
    assert reason == "unexpected_turn_structure"


def test_exclusions_match_case_insensitively_and_through_the_domain() -> None:
    """KodCode is an evaluation set; a case variant must not slip past."""

    assert excluded("Code/KodCode-V1-SFT", ADAPTER.excluded_provenance)
    assert excluded("Code/kodcode-v1-sft", ADAPTER.excluded_provenance)
    assert not excluded("Code/OpenCodeReasoning", ADAPTER.excluded_provenance)


def test_unknown_sources_are_not_allowlisted() -> None:
    allowed = frozenset({"Math/OpenMathInstruct-2"})
    assert known("Math/OpenMathInstruct-2", allowed)
    assert known("math/openmathinstruct-2", allowed)
    assert not known("Math/KodCode-Derived", allowed)


def test_ungradeable_rows_stay_out_of_the_sampling_gate_panel() -> None:
    """Code rows train and count toward CE but cannot certify the run.

    The gate scores every panel row with the MATH verifier, so a program in
    ``final_answer`` would be wrong by construction and would deflate both
    gate accuracy and the mixed-group rate the RL stage is sized from.
    """

    documents = [
        {"problem": f"problem {index}", "document": "d",
         "final_answer": "1", "verified": False,
         "gradeable": index % 2 == 0}
        for index in range(40)
    ]
    train_docs, holdout_docs, panel = split_holdout(documents, 10)
    assert len(holdout_docs) == 10
    assert len(train_docs) == 30
    assert panel, "some held-out problems are gradeable"
    assert all(document["gradeable"] for document in panel)
    assert len(panel) < len(holdout_docs)


def test_corpora_without_the_column_are_wholly_gradeable() -> None:
    documents = [
        {"problem": f"problem {index}", "document": "d",
         "final_answer": "1", "verified": True}
        for index in range(20)
    ]
    _, holdout_docs, panel = split_holdout(documents, 5)
    assert len(panel) == len(holdout_docs) == 5


def _containment(reference: list[str]):
    from postraining.prepare_sft_corpus import build_containment_index

    gram_to_ids, sizes = build_containment_index(reference)
    exact = {" ".join(text.split()).lower() for text in reference}
    return exact, gram_to_ids, sizes


REFERENCE = (
    "Given an array of integers nums and an integer target, return the "
    "indices of the two numbers such that they add up to target. You may "
    "assume that each input would have exactly one solution, and you may "
    "not use the same element twice."
)


def test_containment_catches_a_verbatim_reference_problem() -> None:
    from postraining.prepare_sft_corpus import (
        CONTAINMENT_MIN_OVERLAP,
        contains_reference_problem,
    )

    exact, gram_to_ids, sizes = _containment([REFERENCE])
    assert contains_reference_problem(
        REFERENCE, set(), gram_to_ids, sizes, CONTAINMENT_MIN_OVERLAP
    )
    assert contains_reference_problem(
        REFERENCE, exact, {}, [], CONTAINMENT_MIN_OVERLAP
    )


def test_containment_catches_a_reference_problem_embedded_in_a_longer_one():
    """The failure mode a similarity ratio has and containment does not.

    Scoring the share of the *candidate's* grams that the index contains lets
    a reference problem dilute away inside a longer problem: the ratio falls
    to |K|/(|K|+|U|) and drops under any fixed threshold once the surrounding
    text is big enough. Measured by concatenation on the real corpora, that
    rule admitted 56.9% of embedded reference problems, because reference
    problems are about 3x smaller than the candidates. Containment measures
    against the reference problem's own gram count, so padding cannot help.
    """

    from postraining.prepare_sft_corpus import (
        CONTAINMENT_MIN_OVERLAP,
        contains_reference_problem,
    )

    exact, gram_to_ids, sizes = _containment([REFERENCE])
    padding = " ".join(
        f"Constraint {index}: the value must remain positive and bounded."
        for index in range(120)
    )
    embedded = f"{padding}\n\n{REFERENCE}\n\n{padding}"
    assert embedded not in exact
    assert contains_reference_problem(
        embedded, set(), gram_to_ids, sizes, CONTAINMENT_MIN_OVERLAP
    )


def test_containment_ignores_shared_boilerplate() -> None:
    """Programming statements share phrasing; that is not contamination.

    Folding the code evaluation set into the shared any-8-gram-matches index
    rejected 466 of 800 distinct UltraData code problems while catching zero
    real duplicates, on grams like "the first line of the input contains
    two" and digit runs out of example I/O blocks.
    """

    from postraining.prepare_sft_corpus import (
        CONTAINMENT_MIN_OVERLAP,
        contains_reference_problem,
    )

    # A realistic reference length matters: real evaluation problems run to a
    # median of 87 word 8-grams, so shared boilerplate is a small minority of
    # them. A 19-gram toy reference would let a shared opening clause alone
    # reach 37% containment, which says nothing about the rule.
    reference = (
        "The first line of the input contains a single integer n the number "
        "of elements in the array that follows on the next line of input. "
        "Kevin has prepared a deck of cards for a tournament and wants to "
        "know how many distinct hands can be dealt without repeating any "
        "suit more than twice, given that the deck is shuffled uniformly at "
        "random before every round and that jokers are removed beforehand. "
        "Print a single integer, the number of such hands modulo 998244353, "
        "followed by a newline character at the end of the output stream."
    )
    distinct = (
        "The first line of the input contains a single integer n the number "
        "of cards in the bundle. Berland has n cities connected by m "
        "bidirectional roads and the president wants to rebuild exactly one "
        "of them so that every city becomes reachable from the capital, "
        "while the total rebuilding cost stays within the given budget."
    )
    _, gram_to_ids, sizes = _containment([reference])
    assert not contains_reference_problem(
        distinct, set(), gram_to_ids, sizes, CONTAINMENT_MIN_OVERLAP
    )


def test_gate_refuses_a_panel_whose_prompts_would_be_truncated() -> None:
    """A truncated gate prompt silently changes what accuracy means.

    ``encode_prompt`` keeps BOS plus the *last* ``max_tokens - 1`` tokens, so
    an overlong problem loses its opening and the policy is graded on a
    question it never saw in full. The gate certifies the run, so it must
    refuse rather than measure that.
    """

    import torch

    from postraining import sft_trace_train

    tokenizer = type(
        "Tokenizer",
        (),
        {
            "bos_id": lambda self: 7,
            "eos_id": lambda self: 7,
            "encode": lambda self, text: list(range(len(text))),
        },
    )()
    args = type(
        "Args",
        (),
        {
            "gate_prompts": 1,
            "gate_samples": 2,
            "gate_max_new_tokens": 8,
            "seed": 1,
            "gate_prompt_tokens": 16,
            "gate_think_min_tokens": 1,
        },
    )()
    panel = [{"problem": "x" * 400, "final_answer": "1", "source": "s"}]
    with pytest.raises(ValueError, match="gate prompts exceed"):
        sft_trace_train.run_sampling_gate(
            object(), tokenizer, panel, args, torch.device("cpu")
        )
