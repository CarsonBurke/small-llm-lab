"""Tests for the relaxed-bar prepare_sft_traces rebuild.

Unit tests cover the repair helpers and each adapter's verification
gates on synthetic rows; the real-parquet tests run every adapter over a
slice of the actual fetched data so a source schema drift fails here,
not silently at corpus-build time.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import postraining.prepare_sft_traces as prep

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def make_args(**overrides) -> argparse.Namespace:
    defaults = {
        "k3_jsonl": None,
        "k3_think_channel": "visible",
        "seed": 0,
        "think_tags": True,
        "answer_tags": True,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_corpus_outputs_are_immutable(tmp_path: Path) -> None:
    output = tmp_path / "canonical.parquet"
    manifest = prep.require_fresh_outputs(output)
    assert manifest == tmp_path / "canonical.manifest.json"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prep.require_fresh_outputs(output)
    output.unlink()
    manifest.write_text("{}")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prep.require_fresh_outputs(output)


def test_canonical_problem_matches_composed_prompt_bytes() -> None:
    raw = "  Work out 2 + 2.\n\n"
    problem = prep.canonical_problem(raw)
    document = prep.compose_document(
        problem, "2 + 2 = 4.", "4", think_tags=True, answer_tags=True
    )
    assert problem == "Work out 2 + 2."
    assert document.startswith(problem + prep.INSTRUCTION_SUFFIX_ANSWER)


def test_last_boxed_balanced_nesting() -> None:
    assert prep.last_boxed(r"so \boxed{\frac{3}{\sqrt{4}}}") == r"\frac{3}{\sqrt{4}}"
    assert prep.last_boxed(r"\boxed{1} then \boxed{2}") == "2"
    assert prep.last_boxed(r"\boxed{\frac{3}{4}") is None
    assert prep.last_boxed("no box") is None


def test_strip_latex_wrappers_peels_nested_display_forms() -> None:
    assert prep.strip_latex_wrappers(r"**\(-6\)**") == "-6"
    assert prep.strip_latex_wrappers(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert prep.strip_latex_wrappers("$42$.") == "42"
    assert prep.strip_latex_wrappers("  7  ") == "7"


def test_unwrap_bytes_literal() -> None:
    assert prep.unwrap_bytes_literal("b'-6\\n'") == "-6"
    assert prep.unwrap_bytes_literal('b"Let i(p) = 3*p."') == "Let i(p) = 3*p."
    assert prep.unwrap_bytes_literal("plain text") == "plain text"
    assert prep.unwrap_bytes_literal("b'unclosed") is None


def test_strip_trailing_answer_lines_only_strips_the_tail() -> None:
    body = "Answer depends on x.\nCompute 2+2 = 4.\n\n**Answer:** \\(4\\)"
    assert prep.strip_trailing_answer_lines(body) == (
        "Answer depends on x.\nCompute 2+2 = 4."
    )
    assert prep.strip_trailing_answer_lines("just a derivation") == (
        "just a derivation"
    )
    # Marker header with the statement on the NEXT line: the whole
    # block goes, including stacked blocks.
    block = (
        "f''(o) = 1020o^3 - 2\n\n**Answer:**  \n"
        "The second derivative is \\(1020o^3 - 2\\)."
    )
    assert prep.strip_trailing_answer_lines(block) == "f''(o) = 1020o^3 - 2"
    stacked = "derive 4.\n\n**Final Answer:**\nblah\n\nAnswer: 4"
    assert prep.strip_trailing_answer_lines(stacked) == "derive 4."
    # Mid-derivation prose mentioning "answer:" is NOT a marker line.
    prose = "we subtract to get the answer: 36 - 10 = 26.\nCheck it."
    assert prep.strip_trailing_answer_lines(prose) == prose
    # Cascade cap: a source with per-part markers loses at most
    # MAX_ANSWER_BLOCK_LINES lines, never the whole derivation.
    parts = "\n".join(
        f"Part {i} work\nAnswer: {i}" for i in range(1, 13)
    )
    survived = prep.strip_trailing_answer_lines(parts)
    assert survived.startswith("Part 1 work")
    removed = len(parts.splitlines()) - len(survived.splitlines())
    assert removed <= prep.MAX_ANSWER_BLOCK_LINES + 4


def test_symbolic_agree_bridges_latex_to_python_syntax() -> None:
    assert prep.symbolic_agree(r"-96a^2", "-96*a**2")
    assert prep.symbolic_agree(r"\frac{1}{v^{\frac{51}{2}}}", "v**(-51/2)")
    assert prep.symbolic_agree(r"\dfrac{15}{143}", "15/143")
    assert not prep.symbolic_agree(r"\frac{9}{20}", "70/143")
    assert not prep.symbolic_agree(r"w^{-\frac{3018}{5}}", "w**(-3058)")
    assert prep.symbolic_agree(r"w^{-3058}", "w**(-3058)")
    assert not prep.symbolic_agree("prose about the answer", "4")


def test_extract_answer_statement_reads_markdown_bold_form() -> None:
    assert prep.extract_answer_statement("... **Answer:** \\(-6\\)") == "\\(-6\\)"
    assert prep.extract_answer_statement("thus Answer: 42") == "42"
    assert prep.extract_answer_statement("no statement") is None


def test_adapt_a1_deepmind_unwraps_verifies_and_strips(monkeypatch) -> None:
    rows = [
        {  # kept: bytes-wrapped, trailing markdown answer agrees
            "question": "b'Let i(p) = 3*p + 6. Give i(-4).'",
            "answer": "b'-6\\n'",
            "deepseek_solution": "i(-4) = 3*(-4) + 6 = -6.\n\n**Answer:** \\(-6\\)",
        },
        {  # dropped: teacher final disagrees with canonical truth
            "question": "b'What is 2 + 2?'",
            "answer": "b'4\\n'",
            "deepseek_solution": "2 + 2 = 5.\n\n**Answer:** \\(5\\)",
        },
        {  # dropped: malformed bytes literal
            "question": "b'unclosed",
            "answer": "b'1\\n'",
            "deepseek_solution": "**Answer:** \\(1\\)",
        },
        {  # kept: boxed rendering agrees under Minerva normalization
            "question": "b'Half of one?'",
            "answer": "b'1/2\\n'",
            "deepseek_solution": "It is \\boxed{\\frac{1}{2}}.",
        },
    ]
    monkeypatch.setattr(prep, "rows_of", lambda name: rows)
    stats: Counter[str] = Counter()
    kept = list(
        prep.screen_candidates(
            "a1_deepmind",
            prep.adapt_a1_deepmind(stats, make_args()),
            stats,
            set(),
            set(),
        )
    )
    assert [row["final"] for row in kept] == ["-6", "1/2"]
    assert kept[0]["problem"] == "Let i(p) = 3*p + 6. Give i(-4)."
    assert all("framing" not in row for row in kept)
    # The trailing answer statement never enters the think body
    # (stripped centrally in screen_candidates).
    assert "Answer" not in kept[0]["reasoning"]
    assert stats["a1_deepmind/disagrees_dropped"] == 1
    assert stats["a1_deepmind/unwrap_failed_dropped"] == 1


def test_adapt_had653_requires_gold_join_and_correct_claim(monkeypatch) -> None:
    def rows_of(name: str) -> list[dict]:
        if name == "gsm8k_main_train.parquet":
            return [
                {"question": "Albert buys pizzas. Total?", "answer": "x <<2>>\n#### 48"},
                {"question": "Beth reads books. Total?", "answer": "y\n#### 9"},
            ]
        return [
            {  # kept: joins gold, claim matches
                "question": "Albert buys pizzas. Total?",
                "cot": "Problem:\nAlbert...\n\nReasoning:\n2*16+2*8 = 48.\n\nAnswer:\n48",
                "final_answer": "48",
            },
            {  # dropped: claim contradicts joined gold
                "question": "Beth reads books. Total?",
                "cot": "Problem:\nBeth...\n\nReasoning:\n3*3 = 9? No, 10.\n\nAnswer:\n10",
                "final_answer": "10",
            },
            {  # dropped: no gold join (teacher-derived truth only)
                "question": "Unjoinable problem?",
                "cot": "Problem:\nU\n\nReasoning:\nr\n\nAnswer:\n1",
                "final_answer": "1",
            },
            {  # dropped: cot missing the Reasoning:/Answer: blocks
                "question": "Albert buys pizzas. Total?",
                "cot": "freeform 48",
                "final_answer": "48",
            },
        ]

    monkeypatch.setattr(prep, "rows_of", rows_of)
    stats: Counter[str] = Counter()
    kept = list(prep.adapt_had653(stats, make_args()))
    assert len(kept) == 1
    assert kept[0]["final"] == "48"
    assert kept[0]["reasoning"] == "2*16+2*8 = 48."
    assert stats["had653_gold/wrong_dropped"] == 1
    assert stats["had653_gold/no_gold_join_dropped"] == 1
    assert stats["had653_gold/malformed_cot_dropped"] == 1


def test_adapt_openmath_excludes_augmented_and_verifies_vs_gold(
    monkeypatch,
) -> None:
    def rows_of(name: str) -> list[dict]:
        if name == "gsm8k_main_train.parquet":
            return [
                {"question": "P1?", "answer": "steps\n#### 14"},
                {"question": "P2?", "answer": "steps\n#### 14"},
                {"question": "P3?", "answer": "steps\n#### 1"},
            ]
        return [
            {  # kept: genuine source, boxed grades correct vs OUR gold
                "problem": "P1?",
                "generated_solution": "So the total is \\boxed{14}.",
                "expected_answer": "14",
                "problem_source": "gsm8k",
            },
            {  # dropped: teacher error vs gold
                "problem": "P2?",
                "generated_solution": "So the total is \\boxed{13}.",
                "expected_answer": "13",
                "problem_source": "gsm8k",
            },
            {  # dropped: no boxed conclusion
                "problem": "P3?",
                "generated_solution": "no box",
                "expected_answer": "1",
                "problem_source": "gsm8k",
            },
            {  # dropped: label is the teacher's own value by construction
                "problem": "Novel rewrite?",
                "generated_solution": "So it is \\boxed{5}.",
                "expected_answer": "5",
                "problem_source": "augmented_gsm8k",
            },
            {  # dropped: genuine source but not in the local gold bank
                "problem": "Unjoinable?",
                "generated_solution": "\\boxed{2}",
                "expected_answer": "2",
                "problem_source": "gsm8k",
            },
        ]

    monkeypatch.setattr(prep, "rows_of", rows_of)
    stats: Counter[str] = Counter()
    kept = list(prep.adapt_openmath(stats, make_args()))
    assert [row["problem"] for row in kept] == ["P1?"]
    assert kept[0]["final"] == "14"
    assert stats["openmath_gsm8k/wrong_dropped"] == 1
    assert stats["openmath_gsm8k/no_boxed_dropped"] == 1
    assert stats["openmath_gsm8k/augmented_excluded"] == 1
    assert stats["openmath_gsm8k/no_gold_join_dropped"] == 1


def test_adapt_socratic_strips_calculator_marks(monkeypatch) -> None:
    rows = [
        {
            "question": "Q?",
            "answer": "How many? ** 2*16=<<2*16=32>>32 slices.\n#### 32",
        },
        {  # dropped: derivation never states the final value
            "question": "Q2?",
            "answer": "Some steps without the value.\n#### 77",
        },
    ]
    monkeypatch.setattr(prep, "rows_of", lambda name: rows)
    stats: Counter[str] = Counter()
    kept = list(prep.adapt_socratic(stats, make_args()))
    assert len(kept) == 1
    assert "<<" not in kept[0]["reasoning"]
    assert kept[0]["final"] == "32"
    assert stats["gsm8k_socratic/inconclusive_dropped"] == 1


def test_adapt_k3_reverifies_and_selects_channel(tmp_path) -> None:
    from postraining.generate_k3_traces import GENERATION_SCHEMA

    def record(**overrides) -> dict:
        base = {
            "schema": GENERATION_SCHEMA,
            "key": "gsm8k/0",
            "problem": "P?",
            "ground_truth": "4",
            "style": "minerva",
            "reasoning_channel": "native channel",
            "visible_prose": "clean prose",
            "final_answer": "4",
            "correct": True,
        }
        base.update(overrides)
        return base

    path = tmp_path / "k3.jsonl"
    lines = [
        json.dumps(record()),
        json.dumps(record(key="gsm8k/1", correct=False)),
        json.dumps(record(key="gsm8k/0")),  # duplicate key
        json.dumps(record(key="gsm8k/2", final_answer="5")),  # reverify fails
        json.dumps(record(key="gsm8k/3", visible_prose=None)),
        json.dumps({"schema": "other/v1", "correct": True, "key": "x"}),
        '{"torn',
    ]
    path.write_text("\n".join(lines) + "\n")

    stats: Counter[str] = Counter()
    kept = list(prep.adapt_k3(stats, make_args(k3_jsonl=str(path))))
    assert [row["reasoning"] for row in kept] == ["clean prose"]
    assert stats["k3_traces/not_correct_skipped"] == 1
    assert stats["k3_traces/duplicate_key_dropped"] == 1
    assert stats["k3_traces/reverify_failed_dropped"] == 1
    assert stats["k3_traces/empty_visible_prose_dropped"] == 1
    assert stats["k3_traces/foreign_schema_skipped"] == 1
    assert stats["k3_traces/torn_line_skipped"] == 1

    stats.clear()
    kept = list(
        prep.adapt_k3(
            stats, make_args(k3_jsonl=str(path), k3_think_channel="reasoning")
        )
    )
    assert kept[0]["reasoning"] == "native channel"


def test_adapt_k3_missing_file_yields_nothing(capsys) -> None:
    stats: Counter[str] = Counter()
    assert list(prep.adapt_k3(stats, make_args(k3_jsonl="/nonexistent"))) == []
    assert "WARNING" in capsys.readouterr().out


def test_graded_correct_refuses_empty_truth() -> None:
    # Minerva grades empty-vs-empty correct (field regex captures the
    # trailing space); an empty final would compose the zero-width
    # <answer></answer> the anchored gate rejects.
    assert not prep.graded_correct("", "")
    assert not prep.graded_correct("14", "  ")
    assert prep.graded_correct("14", "14")


def test_all_math_families_use_one_prompt_contract() -> None:
    from postraining.generate_k3_traces import bare_problem
    from postraining.train_latent_vapo import rewrite_prompts_for_answer_fence

    # End to end on real DeepMind rows: source wrappers disappear and SFT/RL
    # both use bare problem + the one shared suffix byte for byte.
    from postraining.core import load_unique_math_rows

    rows = load_unique_math_rows(
        DATA_DIR / "deepmind-interpolate-rl.parquet"
    )[:3]
    bare = [bare_problem(row) for row in rows]
    for problem, row in zip(bare, rewrite_prompts_for_answer_fence(rows)):
        content = row["prompt"][0]["content"]
        assert content == problem + prep.INSTRUCTION_SUFFIX_ANSWER


def test_screen_candidates_fence_guard_and_answer_line_strip() -> None:
    stats: Counter[str] = Counter()
    candidates = [
        {  # fence string in teacher prose: would tokenize to a real
            # special id and break the single-pair gate invariant
            "problem": "P?",
            "reasoning": "I will use <think> to reason.",
            "final": "4",
        },
        {"problem": "P?", "reasoning": "ok", "final": "4</answer>"},
        {  # trailing Answer: line stripped, remainder kept
            "problem": "P?",
            "reasoning": "2+2 = 4.\nAnswer: 4",
            "final": "4",
        },
        {  # nothing but an Answer: line -> empty after strip
            "problem": "P?",
            "reasoning": "Answer: 4",
            "final": "4",
        },
    ]
    kept = list(
        prep.screen_candidates("src", iter(candidates), stats, set(), set())
    )
    assert len(kept) == 1
    assert kept[0]["reasoning"] == "2+2 = 4."
    assert stats["src/fence_string_dropped"] == 2
    assert stats["src/empty_field_dropped"] == 1


def test_composed_documents_pass_the_anchored_gate() -> None:
    from postraining.core import GPT2BPETokenizer
    from postraining.train_latent_vapo import structural_format_ok

    tokenizer = GPT2BPETokenizer(think_tokens=True, answer_tokens=True)
    think_ids = (
        tokenizer.encode("<think>")[0], tokenizer.encode("</think>")[0],
    )
    answer_ids = (
        tokenizer.encode("<answer>")[0], tokenizer.encode("</answer>")[0],
    )
    cases = [
        ("Bare problem?", "Reason a bit.\nCompute 2+2 = 4.", "4"),
        (
            "Work out 1 - 1.",
            "Subtracting gives 0.",
            "0",
        ),
    ]
    for problem, reasoning, final in cases:
        document = prep.compose_document(
            problem, reasoning, final, think_tags=True, answer_tags=True
        )
        prompt = problem + prep.INSTRUCTION_SUFFIX_ANSWER
        assert document.startswith(prompt)
        completion = tokenizer.encode(document[len(prompt):])
        stop_terminated = completion + [tokenizer.encode("!")[0]]
        assert structural_format_ok(stop_terminated, think_ids, answer_ids)


def test_contamination_exact_and_ngram() -> None:
    exact = {"work out 64339656 - 0."}
    ngrams = prep.word_ngrams(
        "janet sells sixteen eggs at the farmers market every single day"
    )
    assert prep.contaminated("Work out  64339656 - 0.", exact, ngrams)
    assert prep.contaminated(
        "Suppose Janet sells sixteen eggs at the farmers market every "
        "single day in spring.",
        exact,
        ngrams,
    )
    assert not prep.contaminated("A fresh unrelated problem.", exact, ngrams)


REAL_DATA = pytest.mark.skipif(
    not (DATA_DIR / "relaxed_bar" / "manifest.json").exists(),
    reason="relaxed_bar parquets not fetched",
)


@REAL_DATA
def test_adapters_yield_on_real_source_slices(monkeypatch) -> None:
    """Every HF adapter must keep a healthy majority of a real slice —
    a source schema drift or repair regression collapses this to ~0."""
    real_rows_of = prep.rows_of

    def sliced(name: str) -> list[dict]:
        rows = pq.read_table(
            DATA_DIR / "relaxed_bar" / name
        ).to_pylist()
        # The gold bank must stay complete for the gold joins.
        if name == "gsm8k_main_train.parquet":
            return rows
        # The genuine rows the adapter keeps sit behind the excluded
        # augmented majority; slice from the kept population.
        if name == "openmathinstruct2_gsm8k_band.parquet":
            rows = [r for r in rows if r["problem_source"] == "gsm8k"]
        return rows[:300]

    monkeypatch.setattr(prep, "rows_of", sliced)
    floors = {
        prep.adapt_openmath: 250,
        prep.adapt_a1_deepmind: 150,
        prep.adapt_had653: 100,
        prep.adapt_socratic: 250,
    }
    for adapt, floor in floors.items():
        stats: Counter[str] = Counter()
        kept = list(adapt(stats, make_args()))
        assert len(kept) >= floor, (adapt.__name__, dict(stats))
        for row in kept:
            assert row["problem"] and row["reasoning"] and row["final"]
    # sxiong's first 300 rows may be any level; check the level filter
    # against the full file instead.
    stats = Counter()
    monkeypatch.setattr(prep, "rows_of", real_rows_of)
    kept = list(prep.adapt_sxiong(stats, make_args()))
    assert len(kept) >= 5_000
    # Two known degenerate rows ("no roots" with empty \boxed{} and empty
    # answer column) reach the adapter; the main loop's empty-field guard
    # drops them before composition. More than a handful means drift.
    assert sum(1 for row in kept if not row["final"]) <= 5


@REAL_DATA
def test_decontamination_index_builds_and_catches_bench_row() -> None:
    exact, ngrams = prep.build_decontamination_index()
    assert len(exact) > 1_400  # gsm8k test + bench + aime
    bench = pq.read_table(
        DATA_DIR / "deepmind-interpolate-easy.parquet"
    ).to_pylist()
    from postraining.generate_k3_traces import bare_problem

    problem = bare_problem(bench[0])
    assert prep.contaminated(problem, exact, ngrams)
    test_question = pq.read_table(
        DATA_DIR / "relaxed_bar" / "gsm8k_test_questions.parquet"
    ).to_pylist()[0]["question"]
    assert prep.contaminated(test_question, exact, ngrams)
