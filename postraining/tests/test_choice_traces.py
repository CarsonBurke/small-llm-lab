"""The science MC teacher-trace source: split, verification and rendering.

Each case is a way a distilled single-choice trace could reach training
wrongly: a question on both sides of the SFT/RL split, a split that moves
with row order, a completion whose letter is not the gold one, a question
the teacher answers right only by chance, a trace that talks about
instructions the student never sees, or a document that is not the
canonical episode. Verification is outcome-only: the prose is never parsed
for a conclusion.
"""

from __future__ import annotations

import json
import random
from itertools import product
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from postraining.choice_prompt import (
    choice_trace_defect,
    render_choice_problem,
    split_options,
)
from postraining.choice_rl_pool import partition_questions, split_draw
from postraining.core import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    GPT2BPETokenizer,
)
from postraining.generate_choice_traces import (
    DEFAULT_QUESTIONS,
    SYSTEM_PROMPT,
    drop_torn_tail,
    finished,
    judge,
    read_records,
    request_seed,
    split_answer_line,
)
from postraining.math_prompt import strip_math_prompt_framing
from postraining.prepare_sft_corpus import (
    ADAPTERS,
    QUESTION_TEMPLATE_DOCUMENTS,
    build_document,
    build_question_index,
    matching_questions,
    parse_choice_trace,
    trace_pool_provenance,
    trained_length,
)
from postraining.prepare_vapo_mixture import SOURCE_SPECS

TRACES = ADAPTERS["science_mc_traces"]
V2 = Path("postraining/data/science-mc-rl-v2.parquet")
RL = dict((name, path) for name, path, *_ in SOURCE_SPECS)["science_mc"]
SFT_QUESTIONS = DEFAULT_QUESTIONS
SPLIT_SCHEMA = "question_component_hash_split/v5"

OPTIONS = ("oxygen", "nitrogen", "carbon dioxide", "helium")
PROBLEM = render_choice_problem(
    "Which gas do plants take in from the air to make food?", OPTIONS
)
QUESTION = {
    "key": "k0",
    "source": "arc_easy",
    "module": "arc_easy_choice_4",
    "problem": PROBLEM,
    "choice": split_options(PROBLEM),
    "gold": "C",
    "style": "exact",
}
REASONING = (
    "Plants make food by photosynthesis, which combines water with a gas "
    "taken from the air. That gas is carbon dioxide; oxygen is released, not "
    "taken in. So the correct answer is option C."
)


@pytest.fixture(scope="module")
def tokenizer() -> GPT2BPETokenizer:
    return GPT2BPETokenizer(think_tokens=True, answer_tokens=True)


def completion(content: str, **overrides) -> dict:
    return {
        "content": content, "reasoning_content": "", "finish_reason": "stop",
        "completion_tokens": 50, "seed": 1, **overrides,
    }


# --- split -----------------------------------------------------------------


def stem(problem: str) -> str:
    return split_options(problem).question


def partition(problems, answers=None, salt="salt", keys=None):
    keys = keys or [f"key{i:04d}" for i in range(len(problems))]
    answers = answers or ["A"] * len(problems)
    sides, report = partition_questions(
        keys, problems, [stem(p) for p in problems], answers, 0.5, salt
    )
    return dict(zip(keys, sides)), report


def test_rows_sharing_or_restating_a_question_are_one_component() -> None:
    restated = (
        "The cell membrane controls which substances enter and leave the "
        "cell by selective permeability through its lipid bilayer structure"
    )
    problems = [
        render_choice_problem("Which is true?", ("a1", "b1", "c1")),
        render_choice_problem("Which is true?", ("a2", "b2", "c2")),
        render_choice_problem(restated + "?", ("x", "y", "z")),
        render_choice_problem("In short: " + restated + ", and which part does "
                              "this?", ("p", "q", "r")),
    ]
    for salt in ("s0", "s1", "s2", "s3"):
        sides, report = partition(problems, salt=salt)
        assert sides["key0000"] == sides["key0001"]
        assert sides["key0002"] == sides["key0003"]
    assert report["component_sizes"] == {"2": 2}


def test_reworded_questions_with_the_same_options_and_answer_are_one_component() -> None:
    # The n-gram rule cannot see a reordered question; the options can.
    options = ("anaerobic respiration", "aerobic respiration", "fermentation",
               "glycolysis")
    problems = [
        render_choice_problem("Cellular respiration that proceeds in the "
                              "presence of oxygen is known as what?", options),
        render_choice_problem("What is cellular respiration that proceeds in "
                              "the presence of oxygen known as?", options[::-1]),
        # Same options, different correct option: a different question.
        render_choice_problem("Which process needs no oxygen and makes "
                              "lactic acid in muscles?", options),
    ]
    answers = ["B", "C", "C"]
    for salt in ("s0", "s1", "s2", "s3"):
        sides, report = partition(problems, answers, salt)
        assert sides["key0000"] == sides["key0001"]
    assert report["joins_by_option_set_and_answer"] == 1
    assert report["component_sizes"] == {"1": 1, "2": 1}


def test_close_stems_whose_answers_differ_by_an_article_plural_or_word_join() -> None:
    problems = [
        render_choice_problem("What is the sac-like organ at the end of the "
                              "esophagus?", ("stomach", "liver", "colon", "lung")),
        render_choice_problem("What is a sac-like organ at the end of the "
                              "esophagus?", ("heart", "the stomach", "spleen", "skin")),
        render_choice_problem("Which organs filter blood to make urine?",
                              ("the kidneys", "lungs", "heart", "skin")),
        render_choice_problem("Which organ filters blood to make urine?",
                              ("liver", "the kidney", "brain", "bone")),
        render_choice_problem("What are winds that blow over short distances "
                              "called?", ("local winds", "trade winds", "jet streams",
                                          "westerlies")),
        render_choice_problem("What are the winds that blow over short "
                              "distances called?", ("global", "polar", "local",
                                                    "prevailing")),
        # A close stem with an unrelated answer stays apart.
        render_choice_problem("Which organ filters blood to make bile?",
                              ("liver", "the kidney", "brain", "bone")),
        render_choice_problem("Where do most ecosystems get their energy from?",
                              ("sun", "soil", "wind", "rock")),
        render_choice_problem("Where do most ecosystems get energy from?",
                              ("heat", "sunlight", "water", "air")),
    ]
    answers = ["A", "B", "A", "B", "A", "C", "A", "A", "B"]
    for salt in ("s0", "s1", "s2", "s3"):
        sides, report = partition(problems, answers, salt)
        assert sides["key0000"] == sides["key0001"]
        assert sides["key0002"] == sides["key0003"]
        assert sides["key0004"] == sides["key0005"]
        assert sides["key0007"] == sides["key0008"]
    assert report["joins_by_reworded_stem_and_answer"] == 4
    assert report["component_sizes"] == {"1": 1, "2": 4}


def test_reworded_stems_with_the_same_answer_are_one_component() -> None:
    # Different distractors, so the option-set join cannot see these; the
    # stems differ by a word ("the", "does"/"do") that the n-gram rule
    # counts. Same answer and close stems join; a close stem with another
    # answer, or the same answer under an unrelated stem, does not.
    problems = [
        render_choice_problem("What is the change in velocity over time called?",
                              ("acceleration", "speed", "mass", "force")),
        render_choice_problem("What is the change in the velocity over time "
                              "called?", ("momentum", "acceleration", "work", "power")),
        render_choice_problem("How does bacteria reproduce?",
                              ("binary fission", "meiosis", "budding", "spores")),
        render_choice_problem("How do bacteria reproduce?",
                              ("mitosis", "binary fission", "seeds", "pollen")),
        render_choice_problem("What is the change in position over time called?",
                              ("velocity", "acceleration", "mass", "force")),
        render_choice_problem("Which quantity do seismographs record during "
                              "an earthquake?", ("acceleration", "heat", "mass", "light")),
    ]
    answers = ["A", "B", "A", "B", "A", "A"]
    for salt in ("s0", "s1", "s2", "s3"):
        sides, report = partition(problems, answers, salt)
        assert sides["key0000"] == sides["key0001"]
        assert sides["key0002"] == sides["key0003"]
    assert report["joins_by_reworded_stem_and_answer"] == 2
    assert report["component_sizes"] == {"1": 2, "2": 2}


def test_a_paraphrase_with_the_same_answer_joins_at_looser_thresholds() -> None:
    # Content-word Jaccard 0.375 and 0.333, difflib ratio about 0.66 and
    # 0.64: below the thresholds for related answers, above those for equal
    # ones.
    problems = [
        render_choice_problem("Which gas makes up most of the air we breathe?",
                              ("nitrogen", "oxygen", "argon", "helium")),
        render_choice_problem("Which gas do plants release that we breathe?",
                              ("carbon", "the nitrogen", "neon", "xenon")),
        render_choice_problem("Which layer of the atmosphere absorbs harmful "
                              "ultraviolet radiation?", ("ozone layer", "crust",
                                                        "mantle", "core")),
        render_choice_problem("What protects living things from harmful "
                              "ultraviolet radiation?", ("water", "ozone", "soil",
                                                         "rock")),
    ]
    answers = ["A", "B", "A", "B"]
    for salt in ("s0", "s1", "s2", "s3"):
        sides, report = partition(problems, answers, salt)
        assert sides["key0000"] == sides["key0001"]
    assert report["joins_by_reworded_stem_and_answer"] == 1
    assert report["component_sizes"] == {"1": 2, "2": 1}


def test_the_stem_ratio_does_not_depend_on_row_order() -> None:
    # difflib's ratio is 0.733 with one stem first and 0.756 with the other.
    problems = [
        render_choice_problem("What protects reptiles from injury and loss of "
                              "water?", ("scales", "hairs", "tails", "claws")),
        render_choice_problem("What protects reptiles from drying out?",
                              ("feathers", "scales", "shells", "fins")),
    ]
    for order in (problems, problems[::-1]):
        answers = ["A", "B"] if order is problems else ["B", "A"]
        _, report = partition(order, answers)
        assert report["joins_by_reworded_stem_and_answer"] == 1


def test_a_match_only_one_partition_index_sees_is_joined_across() -> None:
    # R shares its first block with C, D1 and D2 and its second with E0-E2.
    # Indexed whole, both blocks are in four stems and so not distinctive,
    # and with every gram distinctive C holds too little of R; but once R's
    # side is indexed alone, only R holds the first block, which C restates.
    first = ("granite basalt obsidian pumice quartzite marble slate gneiss "
             "schist shale chalk flint").split()
    second = ("volcanic ash settles slowly across wide valleys forming thick "
              "fertile soil layers today").split()

    def own(prefix: str) -> list[str]:
        return [f"{prefix}{i}" for i in range(12)]

    stems = {
        "R": first + second,
        "C": first + own("cword"),
        "D1": first + own("done"),
        "D2": first + own("dtwo"),
        **{f"E{i}": first[-7:] + second + own(f"e{i}x") for i in range(3)},
    }
    problems = [
        render_choice_problem(" ".join(words) + "?",
                              (f"{name} one", f"{name} two", f"{name} three"))
        for name, words in stems.items()
    ]
    sides, report = partition(problems, salt="s0", keys=list(stems))
    assert report["cross_partition_rounds"] == 2
    assert sides["R"] == sides["C"]
    assert report["component_sizes"] == {"7": 1}


def test_partition_is_a_function_of_content_not_order() -> None:
    problems = [
        render_choice_problem(f"Question number {i} about topic {i % 7}?",
                              (f"a{i}", f"b{i}", f"c{i}", f"d{i}"))
        for i in range(400)
    ]
    problems += [render_choice_problem("Which is true?", (f"x{i}", "y", "z"))
                 for i in range(5)]
    keys = [f"key{i:04d}" for i in range(len(problems))]
    answers = ["ABCD"[i % 4] for i in range(400)] + ["A"] * 5
    sides, report = partition_questions(
        keys, problems, [stem(p) for p in problems], answers, 0.5, "salt"
    )
    by_key = dict(zip(keys, sides))
    order = list(range(len(problems)))
    random.Random(0).shuffle(order)
    shuffled, _ = partition_questions(
        [keys[i] for i in order], [problems[i] for i in order],
        [stem(problems[i]) for i in order], [answers[i] for i in order],
        0.5, "salt",
    )
    assert {keys[i]: side for i, side in zip(order, shuffled)} == by_key
    # The generic stem is one component, on one side.
    assert len({by_key[key] for key in keys[400:]}) == 1
    assert report["component_sizes"]["5"] == 1
    assert 150 < sum(sides) < 250
    # Another salt is another split.
    other, _ = partition_questions(
        keys, problems, [stem(p) for p in problems], answers, 0.5, "other"
    )
    assert other != sides


def test_split_draw_is_uniform_and_fraction_is_checked() -> None:
    draws = [split_draw(f"key{i}", "salt") for i in range(4000)]
    assert all(0.0 <= draw < 1.0 for draw in draws)
    assert 0.47 < sum(draw < 0.5 for draw in draws) / len(draws) < 0.53
    for fraction in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            partition_questions(
                ["k"], [PROBLEM], [stem(PROBLEM)], ["C"], fraction, "s"
            )


def pool_rows(path: Path) -> list[dict]:
    if not path.exists():
        pytest.skip(f"{path} is not built")
    return pq.read_table(path).to_pylist()


def test_built_partitions_are_disjoint_and_cover_v2() -> None:
    rl, sft, v2 = pool_rows(RL), pool_rows(SFT_QUESTIONS), pool_rows(V2)
    key = lambda row: row["extra_info"]["original_query_sha256"]  # noqa: E731
    rl_keys, sft_keys = {key(r) for r in rl}, {key(r) for r in sft}
    assert not rl_keys & sft_keys
    assert rl_keys | sft_keys == {key(r) for r in v2}
    by_key = {key(r): r for r in v2}
    assert all(by_key[key(r)] == r for r in rl + sft)
    # No SFT question restates an RL question under the owner rule, and no
    # RL question restates an SFT one, whether the other partition is indexed
    # whole or each of its questions alone.
    for (candidates, references), template_documents in product(
        ((sft, rl), (rl, sft)), (QUESTION_TEMPLATE_DOCUMENTS, None)
    ):
        index = build_question_index(
            [stem(r["prompt"][-1]["content"]) for r in references],
            template_documents,
        )
        leaked = [
            r["prompt"][-1]["content"][:80] for r in candidates
            if matching_questions(r["prompt"][-1]["content"], index)
        ]
        assert not leaked, leaked[:5]
    # Nor do they share an option set with the same correct option.
    def answer_identity(row):
        parsed = split_options(row["prompt"][-1]["content"])
        options = [" ".join(o.split()).casefold() for o in parsed.options]
        gold = parsed.labels.index(row["reward_model"]["ground_truth"])
        return tuple(sorted(options)), options[gold]

    assert not {answer_identity(r) for r in sft} & {answer_identity(r) for r in rl}
    # Nor a close rewording with the same correct option text.
    from postraining.choice_rl_pool import _Components

    both = sft + rl
    components = _Components(len(both))
    joins = components.link_reworded(
        [r["prompt"][-1]["content"] for r in both],
        [stem(r["prompt"][-1]["content"]) for r in both],
        [r["reward_model"]["ground_truth"] for r in both],
    )
    rl_roots = {components.find(len(sft) + j) for j in range(len(rl))}
    crossing = [
        row["prompt"][-1]["content"][:80]
        for i, row in enumerate(sft) if components.find(i) in rl_roots
    ]
    assert joins and not crossing, crossing[:5]
    # Every source keeps a healthy RL share.
    for source in ("arc_challenge", "arc_easy", "openbookqa", "sciq"):
        both = [r for r in v2 if r["data_source"] == f"science_mc_{source}"]
        mine = [r for r in rl if r["data_source"] == f"science_mc_{source}"]
        assert 0.4 < len(mine) / len(both) < 0.6


def test_both_manifests_record_the_split() -> None:
    for path in (RL, SFT_QUESTIONS):
        pool_rows(path)
        manifest = json.loads(path.with_suffix(".manifest.json").read_text())
        split = manifest["split"]
        assert split["schema"] == SPLIT_SCHEMA
        assert split["sft_fraction"] == 0.5 and split["salt"]
        assert split["rl_output"] == str(RL)
        assert split["sft_output"] == str(SFT_QUESTIONS)
    manifest = json.loads(RL.with_suffix(".manifest.json").read_text())
    assert manifest["split"]["sft_output_sha256"] == json.loads(
        SFT_QUESTIONS.with_suffix(".manifest.json").read_text()
    )["output_sha256"]


# --- teacher completions ---------------------------------------------------


def test_answer_line_parsing() -> None:
    assert split_answer_line(REASONING + "\nAnswer: C") == (REASONING, "C")
    assert split_answer_line(REASONING + "\n\nAnswer: C  \n") == (REASONING, "C")
    assert split_answer_line(REASONING) == "no_answer_line"
    assert split_answer_line(REASONING + "\nAnswer: C.") == "no_answer_line"
    assert split_answer_line(REASONING + "\nAnswer: C\nHope this helps.") == (
        "no_answer_line"
    )
    assert split_answer_line("Answer: C") == "no_reasoning"
    # An answer line run onto the concluding sentence keeps the sentence.
    assert split_answer_line("Carbon dioxide.\nSo the correct option is C. Answer: C") == (
        "Carbon dioxide.\nSo the correct option is C.", "C"
    )
    assert split_answer_line("So the correct option is Answer: C") == "no_answer_line"


def test_a_correct_clean_completion_is_kept(tokenizer) -> None:
    verdict = judge(completion(REASONING + "\nAnswer: C"), QUESTION, tokenizer, 1024)
    assert verdict["correct"] and "reason" not in verdict
    assert verdict["reasoning"] == REASONING and verdict["letter"] == "C"


@pytest.mark.parametrize(
    ("content", "overrides", "reason", "correct"),
    [
        (REASONING.replace("option C", "option B") + "\nAnswer: B", {},
         "wrong_answer", False),
        (REASONING + "\nAnswer: E", {}, "letter_not_an_option", False),
        (REASONING + "\nAnswer: C", {"finish_reason": "length"}, "truncated", False),
        (REASONING + "\nAnswer: C", {"reasoning_content": "hmm"},
         "reasoning_channel", True),
        ("The last line must hold only the letter. " + REASONING + "\nAnswer: C",
         {}, "reasoning_cites_answer_format", True),
        ("As a worked solution for a small student model: " + REASONING
         + "\nAnswer: C", {}, "reasoning_cites_teacher_instructions", True),
        # Outcome-only: the graded letter decides, and the prose is kept as
        # the teacher wrote it, second thoughts and all.
        (REASONING + " Then again, the answer is option D.\nAnswer: C", {}, None, True),
        (REASONING + " Options A, B and D are wrong.\nAnswer: C", {}, None, True),
        (f"Plants <answer>C</answer> {REASONING}\nAnswer: C", {},
         "fence_literal", True),
        # A second answer line, wherever it starts, would train into <think>.
        ("Answer: B\n" + REASONING + "\nAnswer: C", {},
         "reasoning_holds_answer_line", True),
        (REASONING + "\n#### Answer: C\nAnswer: C", {},
         "reasoning_holds_answer_line", True),
        (REASONING + "\n- Answer: C\nAnswer: C", {},
         "reasoning_holds_answer_line", True),
        (REASONING + "\n> Answer: C\nAnswer: C", {},
         "reasoning_holds_answer_line", True),
        (REASONING + "\nThe answer: C\nAnswer: C", {},
         "reasoning_holds_answer_line", True),
        ("So, Answer: C is right. " + REASONING + "\nAnswer: C", {},
         "reasoning_holds_answer_line", True),
        (REASONING + " **Answer:** C\nAnswer: C", {},
         "reasoning_holds_answer_line", True),
        ("Photosynthesis takes in carbon dioxide, so it is C... wait, no, "
         "it is still C.\nAnswer: C", {}, None, True),
        ("Carbon dioxide is taken in; option A is wrong. So D is correct... "
         "no, C is correct.\nAnswer: C", {}, None, True),
    ],
)
def test_rejected_completions(tokenizer, content, overrides, reason, correct) -> None:
    verdict = judge(completion(content, **overrides), QUESTION, tokenizer, 1024)
    assert verdict.get("reason") == reason
    assert verdict["correct"] is correct


def defect(reasoning: str) -> str | None:
    return choice_trace_defect(reasoning, QUESTION["choice"])


@pytest.mark.parametrize(
    "reasoning",
    [
        # The earlier red teams' "false accepts" of the conclusion screen,
        # which is gone: with outcome-only verification a trace that hedges,
        # revises, ends on a dismissal or never names its answer is kept as
        # written, and its <answer> letter is the graded one (NOTES
        # 2026-09-24). Guesses are controlled by self-consistency instead.
        "So the correct option is C. On reflection, option B is correct, because "
        "the question does not ask about photosynthesis.",
        "So the answer is likely C.",
        "So is the correct answer C?",
        "I think the answer is C, but I am not sure.",
        "The chloroplast.",
        "Wait, let me reconsider. Carbon dioxide is taken in, so option C.",
        "The user wants me to select the correct option from the given list "
        "(A through D).",
    ],
)
def test_the_prose_is_not_parsed_for_a_conclusion(reasoning) -> None:
    assert defect(reasoning) is None


@pytest.mark.parametrize(
    "reasoning",
    [
        "Option C is wrong because it names the mitochondria. So the correct option is C.",
        "Option C (carbon dioxide) is incorrect. So the correct option is C.",
        "Choice C was not right. So the correct option is C.",
    ],
)
def test_trace_rejecting_its_verified_answer_is_dropped(reasoning) -> None:
    assert choice_trace_defect(reasoning, QUESTION["choice"], "C") == (
        "reasoning_rejects_answer"
    )
    assert choice_trace_defect(reasoning, QUESTION["choice"], "A") is None


@pytest.mark.parametrize(
    "reasoning",
    [
        "The answer is: C. So option C.",
        "Answer - C\nSo option C.",
        "Answer = C. So option C.",
        "Answer\uff1aC. So option C.",
        "Final answer\nC\nSo option C.",
        "**Answer** C. So option C.",
        "Correct option: C. So option C.",
        # Closing lines under another heading (the fourth red team).
        "Plants take in carbon dioxide.\nFinal selection: C.\nSo option C.",
        "Plants take in carbon dioxide.\n**Final Selection:** C.\nSo option C.",
        "Plants take in carbon dioxide.\nOption: c.\nSo option C.",
        "Plants take in carbon dioxide.\nDecision: C\nSo option C.",
        "Plants take in carbon dioxide.\nNumber: 3.\nSo option C.",
        "Plants take in carbon dioxide.\n6.  **Final Selection:** C.\nSo option C.",
    ],
)
def test_every_answer_line_shape_is_caught(reasoning) -> None:
    assert defect(reasoning) == "reasoning_holds_answer_line"


@pytest.mark.parametrize(
    "reasoning",
    [
        "Is the format correct? Yes. So the correct option is C.",
        "Final string: So the correct option is C.",
        "The question asks for a single letter. So the correct option is C.",
        "Letter: C. So the correct option is C.",
        "I must make sure I output the letter. So the correct option is C.",
        "That will fit the constraints. So the correct option is C.",
        "Step-by-step derivation for the final output: so the correct option is C.",
        "I need to select the letter. So the correct option is C.",
        "Final formatting: So the correct option is C.",
        "Check the output constraints. So the correct option is C.",
        "The user wants the answer as a letter. So the correct option is C.",
        "Does it fit the format? So the correct option is C.",
        "So the correct option is C.\nDouble check the letter options. The "
        "correct one is C.",
        # The sixth red team's.
        "The question asks for a single symbol from 1 to 10. So the correct "
        "option is C.",
        "So the correct option is C. It needs a single symbol response.",
        "So the correct option is C.\nLetter choice: C.",
        "Plants take in carbon dioxide, option C. So I should output C.",
    ],
)
def test_output_planning_is_format_talk(reasoning) -> None:
    assert defect(reasoning) == "reasoning_cites_answer_format"


TRACE = "Plants take in carbon dioxide. So the correct option is C."


@pytest.mark.parametrize(
    "reasoning",
    [
        # A response drafted after the reasoning, under a header of its own
        # (532 of 3,182 knowledge v3 traces), or a plan for the response
        # (the sixth and seventh red teams): dropped, not cut.
        f"{TRACE}\n\n**Step-by-step derivation:**\n1. Recap.\nOption C.",
        f"{TRACE}\n\nStep-by-step reasoning:\n1. Recap.",
        f"{TRACE}\n\n  3. **Step-by-step explanation**\n1. Recap.",
        f"{TRACE}\n\n5.  **Drafting the Explanation:**\n    *   Select Option C.",
        "Plants take in carbon dioxide.\n\n5.  **Formulate the explanation:**\n"
        "    *   Select option C.",
        "Option A is wrong.\n\n**Explanation Construction:**\n1. Select option C.",
        "Plants take in carbon dioxide.\n**6. Formulate the step-by-step reasoning:**"
        "\n- Select option C.",
        f"{TRACE}\n\nStructure the reasoning:\n1. Gas exchange.",
    ],
)
def test_a_drafted_response_drops_the_trace(reasoning) -> None:
    assert defect(reasoning) == "reasoning_drafts_response"


def test_structure_that_is_not_a_drafted_response_stays() -> None:
    # A header inside a line, and step lists or checks, are the work itself.
    assert defect("The step-by-step derivation: carbon dioxide. So option C.") is None
    assert defect("**Steps:**\n1. Plants take in carbon dioxide.\n" + TRACE) is None
    assert defect("Step 1: Plants take in carbon dioxide.\n" + TRACE) is None
    assert defect(f"{TRACE}\n\n**Final Check:**\nOxygen is released, not taken in.") is None
    # A recap that answers again holds a second answer line.
    assert defect(f"{TRACE}\n\nSteps:\n1. Recall photosynthesis.\nNumber: 3.") == (
        "reasoning_holds_answer_line"
    )
    assert defect(f"{TRACE}\n\n6.  **Final Selection:** C.") == (
        "reasoning_holds_answer_line"
    )


@pytest.mark.parametrize(
    "reasoning",
    [
        "Therefore, option C is the intended answer.",
        "Most answer keys say carbon dioxide, so the correct option is C.",
        "Similar questions online accept C, so the correct option is C.",
        "The question writer wants C, so the correct option is C.",
        "The question contains a typo, likely intending carbon dioxide. So "
        "the correct option is C.",
        "Is there a trick? No. So the correct option is C.",
        "Maybe this is a trick question. So the correct option is C.",
        "For this type of question the gas is carbon dioxide. So the correct "
        "option is C.",
        "The conventional answer is carbon dioxide. So the correct option is C.",
        "The question is flawed, but carbon dioxide fits. So the correct "
        "option is C.",
        "The question author intended carbon dioxide. So the correct option is C.",
        # Exam talk (the fourth red team: 55 of 751 knowledge v4 dry-build
        # documents before these were caught).
        "In multiple-choice questions like this, the gas is carbon dioxide. "
        "So the correct option is C.",
        "The target answer is carbon dioxide. So the correct option is C.",
        "The 'gold standard' correct answer is carbon dioxide. So the correct "
        "option is C.",
        "On a board exam this is carbon dioxide. So the correct option is C.",
        "Questions of this nature test photosynthesis. So the correct option is C.",
        "This is a play on words. So the correct option is C.",
        "This fact is often tested. So the correct option is C.",
        "The examiner wants carbon dioxide. So the correct option is C.",
        "USMLE-style questions test this. So the correct option is C.",
        # The fifth red team's.
        "In multiple-choice contexts, the gas is carbon dioxide. So the correct "
        "option is C.",
        "Option A is a common distractor. So the correct option is C.",
        "The wording strongly hints that this is the correct answer. So the "
        "correct option is C.",
        "In the context of general biology questions, the gas is carbon dioxide. "
        "So the correct option is C.",
        "Such questions usually require the gas taken in. So the correct option is C.",
        "C is the longest option. So the correct option is C.",
        "Carbon dioxide is the traditional answer. So the correct option is C.",
        # The sixth red team's.
        "Usually, in such MCQs, you pick the closest value. So the correct "
        "option is C.",
        "In MCQs regarding gas exchange, carbon dioxide is taken in. So the "
        "correct option is C.",
        "Specificity usually wins in multiple choice. So the correct option is C.",
        "Usually, in physics/optics questions, hardware takes precedence. So "
        "the correct option is C.",
        "This is a board-exam style clue. So the correct option is C.",
        "This is a high-yield fact. So the correct option is C.",
        "Uptake is the buzzword. So the correct option is C.",
        "So 'b' is likely intended to be false. So the correct option is C.",
        "Option A is likely a distractor. So the correct option is C.",
        "Option A is a distractor testing respiration. So the correct option is C.",
        '**Consider the "Distractor" Information:** the light level is '
        "irrelevant. So the correct option is C.",
        "In PubMedQA the label for this question is typically yes. So the "
        "correct option is C.",
        "The benchmark label is C. So the correct option is C.",
        "So the correct option is C.\n*   Ref: UpToDate on photosynthesis.",
        # The seventh red team's.
        "In the context of standard organic synthesis questions, the gas is "
        "carbon dioxide. So the correct option is C.",
        'Usually, "kinetics" questions look for the order. So the correct option is C.',
        "These questions test gas exchange. So the correct option is C.",
        "Sometimes questions conflate uptake and release. So the correct option is C.",
        "In the context of condensed matter physics questions regarding leaves, "
        "the gas is carbon dioxide. So the correct option is C.",
        "Options A and B use absolute qualifiers, which make them false. So the "
        "correct option is C.",
        "Uptake is a high-yield fact. So the correct option is C.",
        "Leaves fix carbon (Vander Heiden et al., Science 2009). So the correct "
        "option is C.",
    ],
)
def test_answer_key_reasoning_is_dropped(reasoning) -> None:
    assert defect(reasoning) == "reasoning_cites_answer_key"


@pytest.mark.parametrize(
    "reasoning",
    [
        # Checks on the problem the student also sees, and science.
        "Let me re-read the options. So the correct option is C.",
        "The prompt gives the gas plants take in. So the correct option is C.",
        "A leaf measured with flawed equipment still takes in carbon dioxide. "
        "So the correct option is C.",
        "A textbook definition of photosynthesis names carbon dioxide. So the "
        "correct option is C.",
        'Stomata follow the format "open by day". So the correct option is C.',
        "Oxygen is released, so option A is a distractor. So the correct "
        "option is C.",
        "Leaves are often in shade. So the correct option is C.",
        "The reference electrode is silver chloride. So the correct option is C.",
        "A single symbol denotes the element. So the correct option is C.",
        "Let's re-read the prompt constraints: in the dark. So the correct option is C.",
        "Toluene is the high-yield solvent here. So the correct option is C.",
        "This is one of the open questions in botany. So the correct option is C.",
        "In absolute terms the uptake is small. So the correct option is C.",
    ],
)
def test_checking_the_problem_is_not_answer_key_talk(reasoning) -> None:
    assert defect(reasoning) is None


@pytest.mark.parametrize(
    "reasoning",
    [
        "So the correct option is C. In the femtosecond regime (",
        "So the correct option is C, and the rate is $k",
        "So the correct option is C because of",
        "So the correct option is C;",
        "So the correct option is C, because chloroplasts contain",
        "So the correct option is C (the chloroplast)",
        "So the correct option is C, i.e.",
        "So the correct option is C...",
        "So the correct option is C, as $E = h\\nu.",
        "So the correct option is C, and the.",
        "So the correct option is C.\n\n$$E = h\\nu$$",
        "So the correct option is C, with $x$ and $y.",
        "So the correct option is C: $$E = h\\nu.",
    ],
)
def test_a_trace_cut_off_mid_sentence_is_dropped(reasoning) -> None:
    assert defect(reasoning) == "trace_truncated"


@pytest.mark.parametrize(
    "reasoning",
    [
        # A label ending the trace is not the article.
        "Plants give off oxygen, so the answer is A.",
        'So the correct option is C, the gas plants are "made of."',
        "So the correct option is C, which is what leaves take in.",
        "So the correct option is C, the gas the leaf is composed of.",
        "So the correct option is C (carbon dioxide).",
        # Inline math by pandoc's rule: "$4s^1$" is closed, "$5" is money.
        "So the correct option is C, the $4s^1$ electron.",
        "So the correct option is C, which costs $5.",
        "So the correct option is C: $$E = h\\nu$$.",
    ],
)
def test_a_closed_final_sentence_is_not_truncated(reasoning) -> None:
    assert defect(reasoning) is None


def test_a_letter_the_grader_rejects_is_not_correct(tokenizer, monkeypatch) -> None:
    import postraining.generate_choice_traces as generation

    monkeypatch.setattr(generation, "verify_answer", lambda *_, **__: (False, None))
    verdict = judge(completion(REASONING + "\nAnswer: C"), QUESTION, tokenizer, 1024)
    assert verdict == {"correct": False, "reason": "grader_disagrees"}


def test_documents_over_the_cap_are_dropped(tokenizer) -> None:
    long = " ".join(f"Fact {i} about leaves." for i in range(400)) + " " + REASONING
    verdict = judge(completion(long + "\nAnswer: C"), QUESTION, tokenizer, 1024)
    assert verdict["reason"] == "over_doc_tokens"
    assert "reason" not in judge(
        completion(long + "\nAnswer: C"), QUESTION, tokenizer, 4096
    )


def test_the_cap_counts_what_the_trainer_frames(tokenizer) -> None:
    # sft_trace_train needs BOS + prompt + completion + stop <= seq_len, with
    # prompt and completion tokenized apart: one more than encode(document).
    document = build_document(PROBLEM, REASONING, "C")
    framed = trained_length(PROBLEM, document, tokenizer)
    assert framed >= len(tokenizer.encode(document)) + 2
    content = REASONING + "\nAnswer: C"
    assert judge(completion(content), QUESTION, tokenizer, framed - 1)["reason"] == (
        "over_doc_tokens"
    )
    assert "reason" not in judge(completion(content), QUESTION, tokenizer, framed)


def test_repetition_is_caught() -> None:
    loop = "The gas that plants take in is carbon dioxide. " * 3
    assert defect(loop + "So option C.") == "repetitive_trace"
    phrase = " ".join(["plants take in carbon dioxide from air"] * 12)
    assert defect(phrase + " So option C.") == "repetitive_trace"
    assert defect(REASONING) is None
    # A verdict repeated once per option is how an option walk reads.
    walk = " ".join(
        f"Option {label} names a gas plants release. This option is incorrect."
        for label in "ABD"
    )
    assert defect(walk + " " + REASONING) is None
    # Repeated math is not a loop: matrix rows, Punnett cells.
    rows = "\n".join(["$| 1 & 0 & 0 & 1 & 0 |$ row."] * 4)
    assert defect(rows + "\n" + REASONING) is None


def test_the_system_prompt_is_value_blind() -> None:
    # No letter, option text or gold information may reach the teacher
    # through the shared system turn.
    assert "carbon" not in SYSTEM_PROMPT
    assert request_seed("a", 0, 0) != request_seed("a", 1, 0)
    assert request_seed("a", 0, 0) == request_seed("a", 0, 0) < 2**32


def test_resume_repairs_a_torn_record_before_appending(tmp_path) -> None:
    output = tmp_path / "gen.jsonl"
    whole = json.dumps({"key": "a", "sample": 0}) + "\n"
    output.write_text(whole + '{"key": "b", "sa')
    # select refuses a torn file rather than crash or guess.
    with pytest.raises(SystemExit, match="torn"):
        read_records(output)
    assert drop_torn_tail(output) == len('{"key": "b", "sa')
    assert output.read_text() == whole
    assert drop_torn_tail(output) == 0
    # What generate does next: append whole records after the repair.
    with output.open("a") as stream:
        stream.write(json.dumps({"key": "b", "sample": 0}) + "\n")
    assert finished(output) == {("a", 0), ("b", 0)}
    output.write_text(whole + "not json\n")
    with pytest.raises(SystemExit, match=":2"):
        finished(output)


def write_generations(tmp_path, contents: list[str], samples: int | None = None):
    """A one-question partition and its generations, one record per content."""

    import hashlib

    import pyarrow as pa

    from postraining.generate_choice_traces import GENERATION_SCHEMA

    questions = tmp_path / "questions.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "prompt": [{"role": "user", "content": PROBLEM}],
        "data_source": "science_mc_arc_easy",
        "reward_model": {"ground_truth": "C", "style": "rule"},
        "extra_info": {"original_query_sha256": "k0", "module": "arc_easy_choice_4"},
    }]), questions)
    generations = tmp_path / "gen.jsonl"
    generations.with_suffix(".meta.json").write_text(json.dumps({
        "schema": GENERATION_SCHEMA, "questions": str(questions),
        "questions_sha256": hashlib.sha256(questions.read_bytes()).hexdigest(),
        "sampling": {"samples_per_question": samples or len(contents), "seed": 0},
    }))
    generations.write_text("".join(json.dumps({
        "schema": GENERATION_SCHEMA, "key": "k0", "sample": sample,
        "seed": request_seed("k0", sample, 0), "content": content,
        "reasoning_content": "", "finish_reason": "stop", "prompt_tokens": 60,
        "completion_tokens": 50,
    }) + "\n" for sample, content in enumerate(contents)))
    return generations


def select_args(tmp_path, generations, share: float = 0.5, name: str = "pool"):
    import argparse

    return argparse.Namespace(
        generations=generations, output=tmp_path / f"{name}.parquet",
        max_doc_tokens=1024, min_correct_share=share,
    )


def test_select_hashes_what_it_parsed_and_waits_out_a_writer(tmp_path) -> None:
    import fcntl
    import hashlib

    from postraining.generate_choice_traces import select

    generations = write_generations(tmp_path, [REASONING + "\nAnswer: C"])
    args = select_args(tmp_path, generations)
    # A generate still appending holds the file exclusively.
    with generations.open("ab") as writer:
        fcntl.flock(writer, fcntl.LOCK_EX)
        with pytest.raises(SystemExit, match="being written by another run"):
            select(args)
    select(args)
    manifest = json.loads(args.output.with_suffix(".manifest.json").read_text())
    assert manifest["generations_sha256"] == hashlib.sha256(
        generations.read_bytes()
    ).hexdigest()
    assert manifest["traces"] == 1


def test_select_admits_a_question_only_if_enough_samples_are_correct(tmp_path) -> None:
    from postraining.generate_choice_traces import required_correct, select

    assert [required_correct(share, 4) for share in (0.25, 0.5, 0.75, 1.0)] == [1, 2, 3, 4]
    assert [required_correct(share, 2) for share in (0.25, 0.5, 0.75)] == [1, 1, 2]
    right = REASONING + "\nAnswer: C"
    wrong = REASONING.replace("option C", "option B") + "\nAnswer: B"
    # One right of four: a guess at four options, below k/2.
    generations = write_generations(tmp_path, [right, wrong, wrong, wrong])
    with pytest.raises(SystemExit, match="no trace survived"):
        select(select_args(tmp_path, generations))
    args = select_args(tmp_path, generations, share=0.25, name="loose")
    select(args)
    manifest = json.loads(args.output.with_suffix(".manifest.json").read_text())
    assert manifest["traces"] == 1
    consistency = manifest["self_consistency"]
    assert consistency["required_correct"] == 1
    assert consistency["correct_per_question"] == {"1/4": 1}
    assert {share: row["questions"] for share, row in
            consistency["admitted_by_share"].items()} == {"0.25": 1, "0.5": 0, "0.75": 0}
    assert consistency["admitted_by_share"]["0.25"]["yield"] == 1.0
    assert consistency["admitted_by_share"]["0.5"]["per_source"] == {
        "arc_easy": {"questions": 0, "questions_with_a_trace": 0, "yield": 0.0}
    }
    assert consistency["admitted_by_share"]["0.25"]["per_option_count"]["4"][
        "yield"] == 1.0
    assert manifest["per_option_count"] == {"4": {"complete": 1, "kept": 1, "yield": 1.0}}
    assert manifest["kept_share_over_question_share"] == {"C": 1.0}
    # Two right of four meets k/2; the kept trace is a correct one. A right
    # sample the screens drop still counts toward consistency.
    screened = "The last line must hold only the letter. " + right
    two = tmp_path / "two"
    two.mkdir()
    args = select_args(two, write_generations(two, [wrong, screened, right, wrong]))
    select(args)
    (row,) = pq.read_table(args.output).to_pylist()
    assert row["letter"] == "C" and row["correct_candidates"] == 2
    assert row["passing_candidates"] == 1
    manifest = json.loads(args.output.with_suffix(".manifest.json").read_text())
    assert manifest["rejections"]["reasoning_cites_answer_format"] == 1
    assert manifest["per_source"]["arc_easy"]["yield"] == 1.0
    # Below the threshold, the tallies count what consistency removed.
    three = tmp_path / "three"
    three.mkdir()
    with pytest.raises(SystemExit, match="no trace survived"):
        select(select_args(three, write_generations(three, [wrong, right, right, wrong]),
                           share=0.75))


def test_select_does_not_admit_a_question_it_has_not_finished(tmp_path) -> None:
    from postraining.generate_choice_traces import select

    generations = write_generations(
        tmp_path, [REASONING + "\nAnswer: C"], samples=2
    )
    with pytest.raises(SystemExit, match="no trace survived"):
        select(select_args(tmp_path, generations, share=0.25))


# --- SFT adapter and episode rendering -------------------------------------


def trace_row(**overrides) -> dict:
    return {
        "problem": PROBLEM, "reasoning": REASONING, "letter": "C",
        "source": "arc_easy", "verified": True, **overrides,
    }


def test_adapter_renders_the_canonical_episode(monkeypatch) -> None:
    import postraining.prepare_sft_corpus as corpus

    monkeypatch.setattr(
        corpus, "_TOKENIZER", GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    )
    for name, value in (("_EXACT", set()), ("_NGRAMS", set()),
                        ("_REF_EXACT", set()), ("_REF_GRAM_TO_IDS", {}),
                        ("_REF_SIZES", []), ("_QUESTIONS", None)):
        monkeypatch.setattr(corpus, name, value)
    kept, reason = corpus.build_row(TRACES, trace_row(), 1024, 16)
    assert reason == ""
    # The prompt is the bare rendered problem the RL policy sees: the
    # canonicalizer leaves it unchanged and appends nothing.
    assert kept["problem"] == PROBLEM == strip_math_prompt_framing(PROBLEM)[0]
    assert kept["document"] == (
        f"{PROBLEM}{THINK_OPEN}\n{REASONING}\n{THINK_CLOSE}\n"
        f"{ANSWER_OPEN}C{ANSWER_CLOSE}"
    )
    assert kept["final_answer"] == "C"
    assert kept["verified"] is True and kept["gradeable"] is False
    assert kept["source"] == "science_mc_traces:arc_easy"
    assert kept.pop("_identity") == corpus.choice_identity(PROBLEM)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"verified": False}, "unverified_trace"),
        ({"letter": "E"}, "answer_not_a_label"),
        ({"problem": "Which gas do plants take in?"}, "not_single_choice"),
        ({"reasoning": "Carbon dioxide. Final line: 'Answer: C'. Option C."},
         "reasoning_holds_answer_line"),
        ({"reasoning": "Carbon dioxide, so option C; the last line is the letter."},
         "reasoning_cites_answer_format"),
        ({"reasoning": f"{TRACE}\n\n**Step-by-step derivation:**\n1. Recap."},
         "reasoning_drafts_response"),
        ({"source": ""}, "missing_provenance"),
    ],
)
def test_adapter_rejections(overrides, reason) -> None:
    assert parse_choice_trace(TRACES, trace_row(**overrides)) == (None, reason)


def test_adapter_is_declared_for_the_1024_window() -> None:
    assert TRACES.kind == "choice_trace"
    assert TRACES.max_doc_tokens == 1024
    assert TRACES.single_choice and not TRACES.balance_choice_answers
    assert TRACES.known_sources == {"arc_challenge", "arc_easy", "openbookqa", "sciq"}
    assert TRACES.question_targets


def test_mixture_serves_the_rl_partition() -> None:
    from postraining.prepare_vapo_mixture import SOURCE_SPECS

    specs = {name: (path, default) for name, path, _, default in SOURCE_SPECS}
    assert specs["science_mc"] == (Path("postraining/data/science-mc-rl-v5.parquet"),
                                   False)
    assert TRACES.rl_counterpart == specs["science_mc"][0]
    assert SFT_QUESTIONS == Path("postraining/data/science-mc-sft-questions-v3.parquet")


def write_bound_pool(tmp_path, letter="C", partition="sft", rl_names_sft=True,
                     traces=1, schema=TRACES.revision, complete=1):
    """A minimal questions partition, RL counterpart and trace pool."""

    import hashlib

    import pyarrow as pa

    tmp_path.mkdir()
    questions = tmp_path / "questions.parquet"
    rl = tmp_path / "rl.parquet"
    pool = tmp_path / "pool.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "prompt": [{"role": "user", "content": PROBLEM}],
        "data_source": "science_mc_arc_easy",
        "reward_model": {"ground_truth": "C", "style": "rule"},
        "extra_info": {"original_query_sha256": "k0"},
    }]), questions)
    digest = hashlib.sha256(questions.read_bytes()).hexdigest()
    split = {"schema": SPLIT_SCHEMA, "rl_output": str(rl),
             "sft_output": str(questions)}
    questions.with_suffix(".manifest.json").write_text(json.dumps(
        {"partition": partition, "split": split, "output_sha256": digest}
    ))
    rl.write_bytes(b"rl pool bytes")
    rl.with_suffix(".manifest.json").write_text(json.dumps({
        "partition": "rl",
        "split": {**split, "sft_output_sha256": digest if rl_names_sft else "0"},
        "output_sha256": hashlib.sha256(rl.read_bytes()).hexdigest(),
    }))
    pq.write_table(pa.Table.from_pylist([{
        "key": "k0", "source": "arc_easy", "problem": PROBLEM,
        "reasoning": REASONING, "letter": letter, "verified": True,
    }] * traces), pool)
    pool_digest = hashlib.sha256(pool.read_bytes()).hexdigest()
    generation = {
        "questions": str(questions), "questions_sha256": digest, "teacher": {},
        "draft_model": None, "server": {}, "sampling": {}, "system_prompt": "",
    }
    pool.with_suffix(".manifest.json").write_text(json.dumps({
        "schema": schema, "generation": generation,
        "generations_sha256": "g", "max_doc_tokens": 1024, "selection": "",
        "verifier": "", "self_consistency": {"questions_complete": complete},
        "per_source": {}, "per_module": {}, "rejections": {},
        "output_sha256": pool_digest,
    }))
    return pool


def test_a_trace_pool_must_come_from_an_sft_partition_and_match_gold(tmp_path) -> None:
    def check(name, **overrides):
        pool = write_bound_pool(tmp_path / name, **overrides)
        return trace_pool_provenance(
            pool, pool.with_name("rl.parquet"), TRACES.revision
        )

    bound = check("ok")
    assert bound["questions_rl_counterpart"].endswith("rl.parquet")
    assert bound["questions_rl_counterpart_sha256"]
    with pytest.raises(ValueError, match="gold letter"):
        check("wrong", letter="B")
    with pytest.raises(ValueError, match="two traces"):
        check("twice", traces=2)
    with pytest.raises(ValueError, match="SFT question partition"):
        check("rl", partition="rl")
    with pytest.raises(ValueError, match="SFT side"):
        check("loose", rl_names_sft=False)
    # A pool selected under the retired conclusion screen is another contract.
    with pytest.raises(ValueError, match="choice_trace_pool/v1"):
        check("v1", schema="choice_trace_pool/v1")
    # Nor may a pool selected before generation finished train.
    with pytest.raises(ValueError, match="once generation has finished"):
        check("partial", complete=0)


def test_a_trace_pool_must_be_split_from_the_rl_pool_rl_trains_on(tmp_path) -> None:
    # A pool distilled from an older split's questions (the red team's v1
    # questions beside rl-v3) must not pass against the current RL pool.
    pool = write_bound_pool(tmp_path / "old")
    with pytest.raises(ValueError, match="not the RL pool"):
        trace_pool_provenance(pool, tmp_path / "current-rl.parquet", TRACES.revision)
    # Nor may the named counterpart's bytes differ from its manifest.
    pool.with_name("rl.parquet").write_bytes(b"rebuilt rl pool")
    with pytest.raises(ValueError, match="does not match its manifest"):
        trace_pool_provenance(pool, pool.with_name("rl.parquet"), TRACES.revision)
