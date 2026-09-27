"""Single-choice parsing and the UltraData Knowledge SFT adapter.

Each case is a way a single-choice row could reach training with the wrong
problem text or the wrong label: a surviving format header, options that do
not match the declared labels, or a conclusion the answer does not commit to.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from postraining.choice_prompt import (
    cites_answer_format,
    concluded_label,
    render_choice_problem,
    split_options,
    strip_choice_framing,
)
from postraining.core import verify_answer
from postraining.math_prompt import strip_math_prompt_framing
from postraining.prepare_sft_corpus import (
    ADAPTERS,
    QUESTION_TEMPLATE_DOCUMENTS,
    STEM_QUESTION_TARGETS,
    build_question_index,
    contains_benchmark_question,
    known,
    parse_ultradata_chat,
    shard_paths,
)

KNOWLEDGE = ADAPTERS["ultradata_sft_2605_knowledge"]

HEADER = (
    "Answer the question below.\n"
    "The last line of your response should be of the following format: "
    "'The correct answer is $LETTER' (without quotes). Where LETTER must be "
    "one of A, B, C, or D. Reason carefully step by step.\n"
)
QUESTION = (
    "For an exponential distribution with decay parameter $m$, what is the "
    "mean $\\mu$?\n"
    "A. $m$\nB. $\\frac{1}{m}$\nC. $m^2$\nD. $e^{-m}$\n"
)


def knowledge_record(user: str, presented: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": presented,
             "reasoning_content": "The mean is 1/m, so option B."},
        ],
        "source": "UltraData-sft-2605",
        "domain": "Knowledge",
        "think_type": "think",
    }


@pytest.mark.parametrize(
    ("block", "labels"),
    [
        ("A. x\nB. y\nC. z", ("A", "B", "C")),
        ("a - x\na - y", None),  # repeated label
        ("(A) x\n(B) y\n\n(C) z", ("A", "B", "C")),
        ("1: x\n2: y\n3: z\n4: w", ("1", "2", "3", "4")),
        ("a- x\nb- y", ("a", "b")),
        ("A) x\nC) y", None),  # skipped label
        ("B. x\nC. y", None),  # does not start at the first label
    ],
)
def test_split_options_requires_a_contiguous_trailing_block(block, labels):
    parsed = split_options(f"Which is right?\n{block}")
    assert (parsed.labels if parsed else None) == labels
    if parsed:
        assert parsed.question == "Which is right?"


def test_numbered_premises_are_question_text_not_options() -> None:
    problem = (
        "Consider the statements:\n1. Tin is toxic.\n2. Tin is inert.\n"
        "Which combination is correct?\nA - 1 only\nB - 2 only\nC - both"
    )
    parsed = split_options(problem)
    assert parsed is not None
    assert parsed.labels == ("A", "B", "C")
    assert parsed.question.endswith("Which combination is correct?")


def test_a_multiline_option_is_not_parsed() -> None:
    assert split_options("Pick one.\nA. x\nB. first line\ncontinued") is None


def test_header_is_removed_and_labels_declared() -> None:
    framing = strip_choice_framing(HEADER + QUESTION)
    assert framing is not None
    assert framing.problem == QUESTION.strip()
    assert framing.labels == ("A", "B", "C", "D")
    assert framing.conclusion == "The correct answer is"
    # The bare problem passes the canonicalizer unchanged.
    assert strip_math_prompt_framing(framing.problem) == (framing.problem, 0)


def test_untemplated_prompts_are_not_choice_framing() -> None:
    assert strip_choice_framing(QUESTION) is None


def test_options_must_match_the_declared_labels() -> None:
    wrong = HEADER.replace("A, B, C, or D", "A, B, C, D, or E")
    with pytest.raises(ValueError, match="declared labels"):
        strip_choice_framing(wrong + QUESTION)


def test_a_template_outside_the_header_fails_closed() -> None:
    with pytest.raises(ValueError):
        strip_choice_framing(QUESTION + HEADER)


def test_a_surviving_letter_template_fails_canonicalization() -> None:
    with pytest.raises(ValueError, match=r"\$LETTER"):
        strip_math_prompt_framing(
            "What is x?\nConclude with: 'The answer is $LETTER'."
        )


@pytest.mark.parametrize(
    ("presented", "label"),
    [
        ("Work.\n\nThe correct answer is B", "B"),
        ("Work.\n\nThe correct answer is B.\n\nThe correct answer is B", "B"),
        ("The correct answer is A\n\nThe correct answer is B", None),
        ("The correct answer is B\n\nSome trailing prose.", None),
        ("Work.\n\nThe correct answer is E", None),  # undeclared label
        ("Work.\n\nThe correct answer is **B**", None),
    ],
)
def test_the_final_line_must_commit_to_one_declared_label(presented, label):
    framing = strip_choice_framing(HEADER + QUESTION)
    assert concluded_label(presented, framing) == label


def test_knowledge_adapter_emits_bare_problem_and_label() -> None:
    parsed, reason = parse_ultradata_chat(
        KNOWLEDGE,
        knowledge_record(HEADER + QUESTION, "Work.\nThe correct answer is B"),
    )
    assert reason == ""
    assert parsed.problem == QUESTION.strip()
    assert parsed.answer == "B"
    assert parsed.solution == "The mean is 1/m, so option B."
    assert parsed.gradeable is True
    assert known(parsed.origin, KNOWLEDGE.known_sources)


@pytest.mark.parametrize(
    ("user", "presented", "reason"),
    [
        (QUESTION, "The correct answer is B", "not_templated_choice"),
        ("Explain entropy.", "Entropy is ...", "not_templated_choice"),
        (HEADER + QUESTION, "It is probably B.", "no_choice_conclusion"),
        (HEADER + "No options here?", "The correct answer is B",
         "malformed_choice_framing"),
    ],
)
def test_knowledge_adapter_drops_rows_without_a_single_label(
    user, presented, reason
):
    parsed, got = parse_ultradata_chat(
        KNOWLEDGE, knowledge_record(user, presented)
    )
    assert parsed is None
    assert got == reason


def test_math_adapter_does_not_admit_knowledge_rows() -> None:
    parsed, reason = parse_ultradata_chat(
        ADAPTERS["ultradata_sft_2605"],
        knowledge_record(HEADER + QUESTION, "The correct answer is B"),
    )
    # Parsed, but outside the Math/Code adapter's provenance allowlist.
    assert parsed is not None
    assert not known(
        parsed.origin, ADAPTERS["ultradata_sft_2605"].known_sources
    )


def test_knowledge_adapter_requires_stem_benchmark_screen() -> None:
    assert KNOWLEDGE.question_targets == STEM_QUESTION_TARGETS
    # Not the 0.30 containment rule, which exam boilerplate trips.
    assert KNOWLEDGE.containment_targets == ()
    declared = " ".join(str(path) for path, _ in STEM_QUESTION_TARGETS)
    for benchmark in ("cais__mmlu", "MMLU-Pro", "gpqa"):
        assert benchmark in declared


# Exam phrasing shared by more than QUESTION_TEMPLATE_DOCUMENTS references.
BOILERPLATE = "Which of the following statements best describes the"
REFERENCES = [
    *(f"{BOILERPLATE} {topic}?" for topic in (
        "role of mitochondria", "Treaty of Westphalia", "Coase theorem",
        "function of the loop of Henle",
    )),
    "A 45-year-old man presents with crushing substernal chest pain "
    "radiating to the left arm and diaphoresis for two hours. Which enzyme "
    "rises first after myocardial injury?",
    # Fewer than QUESTION_MIN_GRAMS distinctive grams.
    "Which of the following best describes the Doppler effect?",
    "Which of the following is not a true statement?",
    "Which of the following is true?",
]
assert sum(BOILERPLATE in text for text in REFERENCES) > QUESTION_TEMPLATE_DOCUMENTS


@pytest.fixture(scope="module")
def questions():
    return build_question_index(REFERENCES)


@pytest.mark.parametrize(
    "candidate",
    [
        # Embedded, reworded around, and re-rendered with options.
        "Background: a patient case. A 45-year-old man presents with "
        "crushing substernal chest pain radiating to the left arm and "
        "diaphoresis for two hours. Which enzyme rises first after "
        "myocardial injury?\n\nA. Troponin\nB. CK-MB\nC. LDH",
        # A short question matches exactly, as the problem or its stem;
        # punctuation and case do not matter.
        "which of the following best describes the doppler effect",
        "Which of the following best describes the Doppler effect?\n\n"
        "A. x\nB. y",
        "Which of the following is true?\n\nA. 1 + 1 = 2\nB. 1 + 1 = 3",
    ],
)
def test_benchmark_questions_are_caught(questions, candidate) -> None:
    assert contains_benchmark_question(candidate, questions)


@pytest.mark.parametrize(
    "candidate",
    [
        # Shares only exam phrasing with four references.
        f"{BOILERPLATE} the effect of quantitative easing on bond yields?"
        "\n\nA. up\nB. down",
        "In chemistry, which of the following is true about the equilibrium "
        "constant?\nA. x\nB. y",
        # A short reference inside a different question.
        "Which of the following is NOT a true statement about fluorescence?"
        "\n1: It is emission of radiation.\n2: It is elastic scattering.",
        "Which of the following best describes the Doppler shift of light "
        "from a receding galaxy?",
    ],
)
def test_exam_phrasing_alone_is_not_a_match(questions, candidate) -> None:
    assert not contains_benchmark_question(candidate, questions)


def test_domains_select_shards_and_fail_closed(tmp_path: Path) -> None:
    for domain in ("Code", "Knowledge", "Math"):
        (tmp_path / domain).mkdir()
        (tmp_path / domain / f"{domain}_part-1.jsonl").write_text("")
    adapter = ADAPTERS["ultradata_sft_2605"]
    paths = list(shard_paths(replace(adapter, local_shards=tmp_path), 0, tmp_path))
    assert [path.parent.name for path in paths] == ["Code", "Math"]
    missing = replace(adapter, local_shards=tmp_path, domains=("Code", "IF"))
    with pytest.raises(FileNotFoundError):
        shard_paths(missing, 0, tmp_path)


def test_render_uses_upper_case_letters() -> None:
    rendered = render_choice_problem("Which?", ("x", "y", "z"))
    assert rendered == "Which?\n\nA. x\nB. y\nC. z"
    assert split_options(rendered).options == ("x", "y", "z")
    with pytest.raises(ValueError):
        render_choice_problem("Which?", ("x", "two\nlines"))


@pytest.mark.parametrize(
    ("span", "correct"),
    [("B", True), (" B ", True), ("b", False), ("(B)", False), ("B.", False),
     ("B, C", False), ("The answer is B", False), ("C", False)],
)
def test_exact_style_grades_exactly_one_letter(span, correct) -> None:
    """RL letter targets use the ``exact`` style: the answer span, stripped,
    must be the single upper-case letter and nothing else."""

    assert verify_answer("Answer: " + span, "B", "exact", window=None)[0] is correct


def test_knowledge_documents_are_capped_below_a_longer_window(monkeypatch):
    import postraining.prepare_sft_corpus as corpus
    from postraining.core import GPT2BPETokenizer

    monkeypatch.setattr(
        corpus, "_TOKENIZER", GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    )
    for name, value in (("_EXACT", set()), ("_NGRAMS", set()),
                        ("_REF_EXACT", set()), ("_REF_GRAM_TO_IDS", {}),
                        ("_REF_SIZES", []), ("_QUESTIONS", None)):
        monkeypatch.setattr(corpus, name, value)
    assert KNOWLEDGE.max_doc_tokens == 2048

    def row(words: int) -> dict:
        record = knowledge_record(HEADER + QUESTION, "Work.\nThe correct answer is B")
        record["messages"][1]["reasoning_content"] = (
            " ".join(f"step{i}" for i in range(words)) + " So option B."
        )
        return record

    kept, reason = corpus.build_row(KNOWLEDGE, row(600), 5120, 16)
    assert reason == "" and kept["doc_tokens"] + 1 <= 2048
    dropped, reason = corpus.build_row(KNOWLEDGE, row(1800), 5120, 16)
    assert dropped is None and reason.startswith("over_")
    # The cap is the source's own; the build's window still binds the rest.
    uncapped = replace(KNOWLEDGE, max_doc_tokens=None)
    assert corpus.build_row(uncapped, row(1800), 5120, 16)[1] == ""


def knowledge_reasoning(reasoning: str) -> tuple:
    record = knowledge_record(HEADER + QUESTION, "Work.\nThe correct answer is B")
    record["messages"][1]["reasoning_content"] = reasoning
    return parse_ultradata_chat(KNOWLEDGE, record)


def test_a_derivation_recap_is_cut_from_the_knowledge_row() -> None:
    kept, reason = knowledge_reasoning(
        "The mean is 1/m, so option B.\n\n**Step-by-step derivation:**\n"
        "1. Recap.\n2. The answer is B."
    )
    assert reason == "" and kept.solution == "The mean is 1/m, so option B."
    assert kept.answer == "B"
    # Only a header line ends the reasoning; the phrase in prose does not.
    kept, reason = knowledge_reasoning(
        "A step-by-step derivation of the mean gives 1/m, so option B."
    )
    assert reason == "" and kept.solution.endswith("so option B.")


def test_a_recap_that_leaves_nothing_drops_the_knowledge_row() -> None:
    assert knowledge_reasoning("### Step-by-step derivation\n1. Recap.") == (
        None, "empty_answer_or_solution"
    )


def test_a_planned_response_above_the_cut_drops_the_knowledge_row() -> None:
    assert knowledge_reasoning(
        "The mean is 1/m.\n**Drafting the explanation:** say B.\n"
        "**Step-by-step derivation:**\n1. Recap."
    ) == (None, "reasoning_drafts_response")


def test_reasoning_that_discusses_the_removed_header_is_dropped() -> None:
    record = knowledge_record(HEADER + QUESTION, "Work.\nThe correct answer is B")
    record["messages"][1]["reasoning_content"] = (
        "The mean is 1/m. Final line must be 'The correct answer is $LETTER'."
    )
    assert parse_ultradata_chat(KNOWLEDGE, record) == (
        None, "reasoning_cites_answer_format"
    )


@pytest.mark.parametrize(
    ("reasoning", "cites"),
    [
        ("Format: 'The answer is $LETTER'.", True),
        ("So the last line should read the answer.", True),
        ("Check the required format once more.", True),
        ("The user wants the mean of an exponential distribution.", False),
        ("The CPU decodes ADD instructions in one cycle.", False),
        ("DNA carries the instructions for life.", False),
        ("Its genome holds the instructions for building proteins.", False),
        ("Per the instructions, reply with the letter.", True),
        ("The instructions say to end with the letter.", True),
        # Science that shares the planning vocabulary (the third red team).
        ("Glycine has the one letter code G.", False),
        ("The rate constant is denoted by the letter k.", False),
        ("Consider the format of a hash table.", False),
        ("Fever is the final response of the immune system.", False),
        ("I need to select the letter and end with it.", True),
        ("The question asks for a single letter.", True),
        # "The format" only as the reply's format (the fourth red team).
        ('The eigenvalues follow the format "eigen-".', False),
        ("Match the format.", True),
        ("Is the format correct?", True),
        ("Final formatting:", True),
    ],
)
def test_format_talk_is_recognised(reasoning, cites) -> None:
    from postraining.choice_prompt import cites_answer_format

    assert cites_answer_format(reasoning) is cites


def test_an_empty_screening_target_fails_closed(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from postraining.prepare_sft_corpus import read_reference_texts

    path = tmp_path / "refs.parquet"
    pq.write_table(pa.table({"question": ["", None]}), path)
    with pytest.raises(ValueError, match="no problem text"):
        read_reference_texts(path, "question")


@pytest.mark.parametrize(
    ("options", "truth", "target"),
    [
        (("x", "y", "z"), "B", 1),  # label
        (("red", "green", "blue"), "green", 1),  # unique text
        (("red", "green", "green"), "green", None),  # two options' text
        # "3" is the label of the option "2" and the text of the fourth.
        (("0", "1", "2", "3"), "3", None),
        (("0", "1", "2", "3"), "1", None),  # label of "0", text of "1"
        (("10", "20", "30"), "2", 1),  # label only
    ],
)
def test_rl_choice_target_must_name_exactly_one_option(options, truth, target):
    from scripts.build_ultradata_knowledge_rl_prompts import choice_target

    digits = options[0] in {"0", "10"}
    labels = [str(i + 1) if digits else "ABC"[i] for i in range(len(options))]
    block = "\n".join(f"{label}: {text}" for label, text in zip(labels, options))
    parsed = split_options(f"Pick one.\n{block}")
    assert choice_target(parsed, truth) == target


def test_rl_permutation_is_deterministic_and_places_target_uniformly() -> None:
    from collections import Counter

    from postraining.choice_rl_pool import balanced_order

    positions = Counter()
    for seed in range(4000):
        order = balanced_order(4, 2, seed.to_bytes(8, "big"))
        assert sorted(order) == [0, 1, 2, 3]
        assert order == balanced_order(4, 2, seed.to_bytes(8, "big"))
        positions[order.index(2)] += 1
    assert all(850 < positions[p] < 1150 for p in range(4))


# --- Answer balancing (drop-only) -------------------------------------------


def balance_rows(positions: list[int], first: int = 0) -> list[dict]:
    rows = []
    for number, position in enumerate(positions, first):
        problem = f"Question {number}?\n" + "\n".join(
            f"{'ABCD'[i]}. value {number}-{i}" for i in range(4)
        )
        rows.append({
            "problem": problem, "final_answer": "ABCD"[position],
            "_choice": {"count": 4, "position": position},
        })
    return rows


def test_balance_caps_the_modal_position_only() -> None:
    from collections import Counter

    from postraining.prepare_sft_corpus import balance_choice_answers

    positions = [0] * 10 + [1] * 30 + [2] * 20 + [3] * 6
    rows = balance_rows(positions)
    exact, exact_report = balance_choice_answers(rows, 0.0)
    loose, loose_report = balance_choice_answers(rows, 0.25)
    # At 0 the modal position holds at most ceil(n / 4); D's 6 rows are a
    # floor on nothing.
    assert sorted(Counter(r["final_answer"] for r in exact).values()) == [6, 9, 9, 9]
    assert exact_report["4"]["modal_share_over_chance"] == 9 * 4 / 33
    # At 0.25 no position exceeds 1.25x chance: 13 of 42.
    assert sorted(Counter(r["final_answer"] for r in loose).values()) == [6, 10, 13, 13]
    assert loose_report["4"]["modal_share_over_chance"] <= 1.25
    assert loose_report["4"]["least_share_over_chance"] == 6 * 4 / 42
    assert loose_report["4"]["positions_before"] == [10, 30, 20, 6]


def test_balance_only_drops_and_ignores_row_order() -> None:
    from postraining.prepare_sft_corpus import balance_choice_answers

    rows = balance_rows([1] * 30 + [0] * 5 + [2] * 5 + [3] * 5)
    kept, _ = balance_choice_answers(rows, 0.25)
    # Every kept row is one of the inputs, unchanged: nothing is relabelled.
    assert all(any(row is original for original in rows) for row in kept)
    again, _ = balance_choice_answers(rows[::-1], 0.25)
    assert sorted(r["problem"] for r in again) == sorted(r["problem"] for r in kept)


def test_an_option_count_too_small_to_fill_every_position_is_kept() -> None:
    from postraining.prepare_sft_corpus import _capped_counts, balance_choice_answers

    assert _capped_counts([5, 7, 0, 3], 0.0) == [3, 3, 0, 3]
    assert _capped_counts([5, 7, 0, 3], 1.0) == [5, 7, 0, 3]
    assert _capped_counts([0, 1, 0, 0, 0], 0.0) == [0, 1, 0, 0, 0]
    kept, report = balance_choice_answers(balance_rows([0, 1, 1, 3]), 0.0)
    assert len(kept) == 3 and report["4"]["least_share_over_chance"] == 0


def knowledge_corpus() -> Path:
    from scripts.build_science_mc_rl_prompts import DEFAULT_OWNERS

    (corpus,) = [path for path in DEFAULT_OWNERS if path.name.startswith("sft_")]
    return corpus


def test_every_built_knowledge_row_presents_canonical_labels() -> None:
    import pyarrow.parquet as pq

    from postraining.choice_prompt import LABEL_STYLES, label_style

    corpus = knowledge_corpus()
    if not corpus.exists():
        pytest.skip(f"{corpus} is not built")
    rows = pq.read_table(corpus).to_pylist()
    for row in rows:
        parsed = split_options(row["problem"])
        assert parsed is not None, row["problem"][-200:]
        style = label_style(parsed.labels[0])
        assert parsed.labels == LABEL_STYLES[style][: len(parsed.labels)]
        assert row["final_answer"] in parsed.labels


def test_rl_rendering_must_parse_back_to_its_options() -> None:
    from postraining.choice_rl_pool import choice_problem

    options = ("amino", "boric", "hydrochloric", "bacterial")
    assert choice_problem(
        "E. coli need what kind of acids to survive?", options, 0, b"\0" * 8, 3
    ) == "ambiguous_rendering"
    problem, letter, order = choice_problem(
        "What do bacteria need to survive?", options, 0, b"\0" * 8, 3
    )
    parsed = split_options(problem)
    assert parsed.options[parsed.labels.index(letter)] == "amino"
    assert [options[i] for i in order] == list(parsed.options)
    assert choice_problem("Pick.", ("x", "Both A and B"), 0, b"\0" * 8, 2) == (
        "option_cross_reference"
    )


@pytest.mark.parametrize(
    "reasoning, reason",
    [
        ("Option B. " + "The. " * 30, "degenerate_trace"),
        ("Option B is the correct answer.cw", "trace_trailing_debris"),
        ("Option B fits. " + "Enantioselectivity varies with the ligand. " * 10
         + "The ee is typically very low (often", "trace_truncated"),
        # The red team's v3 rows: a list mention near the end passed the old
        # 300-character label check.
        ("The candidates include A, B, and D. Low $T_c$ (", "trace_truncated"),
        ("So option B, which is the correct answer. It suggests temporal "
         "resolution of", "trace_truncated"),
        ("Stepwise.\n" + "-" * 40 + "\nSo option B.", None),
        ("Conclusion: Option B.", None),
        # Outcome-only verification: the prose is not parsed for a
        # conclusion, so a trace that never names its answer, hedges or
        # revises is kept as written; the <answer> letter is what was graded.
        ("The candidates include A, B, and D. Each is plausible here.", None),
        ("Final Answer seems to be B.", None),
        ("It is probably D... wait, no, B fits better.", None),
    ],
)
def test_defective_traces_are_named(reasoning, reason) -> None:
    from postraining.choice_prompt import ChoiceProblem, choice_trace_defect

    choice = ChoiceProblem(
        "Which is it?", tuple("ABCD"), ("alpha", "beta", "gamma", "delta")
    )
    assert choice_trace_defect(reasoning, choice) == reason


@pytest.mark.parametrize(
    "reasoning",
    [
        "Option B.\nConclude with 'The answer is B'.",
        "Option B.\n*   End with \"So, the final answer is B\".",
        "Option B.\n7.  **Format:** answer: B",
        "Option B.\n9.  **Format the output:**\nFinal answer: B",
        "The final answer should be formatted as requested.",
        "Reply with the letter and the specific concluding phrase.",
        "Choose LETTER b.",
    ],
)
def test_template_planning_is_format_talk(reasoning) -> None:
    assert cites_answer_format(reasoning)


def test_choice_identity_ignores_labels_and_order() -> None:
    from postraining.prepare_sft_corpus import choice_identity

    upper = "Which gas?\nA. Argon\nB. Neon"
    digits = "Which  gas?\n1) Neon\n2) argon"
    assert choice_identity(upper) == choice_identity(digits)
    assert choice_identity(upper) != choice_identity("Which gas?\nA. Argon\nB. Xenon")
