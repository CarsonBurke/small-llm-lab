"""Evaluation-only benchmarks can screen training data but never enter it.

GPQA, MMLU, MMLU-Pro and the SciQ/ARC/OpenBookQA validation and test splits
live under ``EVALUATION_ONLY_DATA`` and are read only as decontamination
targets. These tests fail if any declared training input could come from
there, if a GPQA question could pass the benchmark-question screen in the
shape a training row would carry it, or if a built pool or corpus holds one.
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from pathlib import Path

import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
import pytest

from postraining.choice_prompt import render_choice_problem, split_options
from postraining.prepare_sft_corpus import (
    ADAPTERS,
    EVALUATION_ONLY_DATA,
    QUESTION_TEMPLATE_DOCUMENTS,
    STEM_QUESTION_TARGETS,
    build_question_index,
    contains_benchmark_question,
    read_reference_texts,
    refuse_evaluation_only,
    shard_paths,
)
from postraining.generate_choice_traces import DEFAULT_QUESTIONS
from postraining.prepare_vapo_mixture import SOURCE_SPECS
from scripts.build_science_mc_rl_prompts import DEFAULT_OWNERS

GPQA = EVALUATION_ONLY_DATA / "Idavidrein__gpqa" / "gpqa_extended.csv"
# The science MC teacher-trace chain: the RL partition, the question
# partition the teacher saw, the verified trace pool and the SFT corpus.
# Taken from the code that consumes them where there is one, so a version
# bump cannot leave these tests checking a stale file.
SCIENCE_TRACE_CHAIN = (
    dict((name, path) for name, path, *_ in SOURCE_SPECS)["science_mc"],
    DEFAULT_QUESTIONS,
    ADAPTERS["science_mc_traces"].local_parquet,
    Path("postraining/data/sft_science_mc_traces_v1.parquet"),
)
# Every current single-choice training artifact: the UltraData Knowledge RL
# pool and SFT corpus (the science builder's owners) and the science chain.
BUILT = (*DEFAULT_OWNERS, *SCIENCE_TRACE_CHAIN)


def under_evaluation_only(path: Path) -> bool:
    return path.resolve().is_relative_to(EVALUATION_ONLY_DATA.resolve())


def test_no_declared_training_input_is_evaluation_only() -> None:
    from scripts.build_science_mc_rl_prompts import SOURCES
    from scripts.build_ultradata_knowledge_rl_prompts import WINDOWS

    inputs = [
        root
        for adapter in ADAPTERS.values()
        for root in (adapter.local_shards, adapter.local_parquet)
        if root is not None
    ]
    inputs += [path for _, path, _, _ in SOURCE_SPECS if path is not None]
    inputs += [source.path for source in SOURCES] + [WINDOWS]
    assert inputs
    offending = [path for path in inputs if under_evaluation_only(path)]
    assert not offending


def test_every_gpqa_file_is_a_screen_target() -> None:
    targets = {(path.name, column) for path, column in STEM_QUESTION_TARGETS}
    assert ("gpqa_extended.csv", "Question") in targets
    assert ("gpqa_extended.csv", "Pre-Revision Question") in targets
    # Main and diamond are subsets of extended; screening extended covers them.
    assert all(
        under_evaluation_only(path) for path, _ in STEM_QUESTION_TARGETS
    )


def test_evaluation_only_paths_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="evaluation-only"):
        refuse_evaluation_only(GPQA)
    with pytest.raises(ValueError, match="evaluation-only"):
        refuse_evaluation_only(EVALUATION_ONLY_DATA / "anything" / "..")
    adapter = dataclasses.replace(
        next(iter(ADAPTERS.values())),
        local_shards=None,
        local_parquet=GPQA.parent,
    )
    with pytest.raises(ValueError, match="evaluation-only"):
        list(shard_paths(adapter, 1, tmp_path))
    with pytest.raises(ValueError, match="evaluation-only"):
        list(shard_paths(next(iter(ADAPTERS.values())), 1, GPQA.parent))
    assert refuse_evaluation_only(tmp_path) == tmp_path


def test_pool_builders_refuse_missing_screen_targets(tmp_path: Path) -> None:
    from postraining.choice_rl_pool import load_reference_texts

    missing = ((tmp_path / "gpqa_extended.csv", "Question"),)
    with pytest.raises(SystemExit, match="unscreened"):
        load_reference_texts(missing, allow_missing=False)


def gpqa_rows() -> list[dict]:
    if not GPQA.exists():
        pytest.skip(f"{GPQA} is not downloaded (gated); builders refuse without it")
    return pa_csv.read_csv(
        GPQA, parse_options=pa_csv.ParseOptions(newlines_in_values=True)
    ).to_pylist()


@lru_cache(maxsize=1)
def production_index():
    texts: list[str] = []
    for path, column in STEM_QUESTION_TARGETS:
        texts.extend(read_reference_texts(path, column))
    return build_question_index(texts)


def test_every_gpqa_question_is_screened_as_a_training_row_would_carry_it() -> None:
    rows = gpqa_rows()
    if not all(path.exists() for path, _ in STEM_QUESTION_TARGETS):
        pytest.skip("screen targets are not all downloaded")
    index = production_index()
    escaped = []
    for row in rows:
        for prefix in ("", "Pre-Revision "):
            question = row[f"{prefix}Question"]
            if not question or not question.strip():
                continue
            options = (
                row[f"{prefix}Correct Answer"],
                *(row[f"{prefix}Incorrect Answer {n}"] for n in (1, 2, 3)),
            )
            shapes = [question, "  " + question.replace("\n", " \n") + "\n"]
            if all(option and "\n" not in option for option in options):
                shapes.append(render_choice_problem(
                    question.strip(), tuple(str(o).strip() for o in options)
                ))
            escaped += [
                shape[:80] for shape in shapes
                if not contains_benchmark_question(shape, index)
            ]
    assert not escaped, escaped[:5]


def built_problems(path: Path) -> list[str]:
    if not path.exists():
        pytest.skip(f"{path} is not built")
    table = pq.read_table(path)
    if "prompt" in table.column_names:
        problems = [turns[-1]["content"] for turns in table.column("prompt").to_pylist()]
    else:
        problems = table.column("problem").to_pylist()
    assert problems
    return problems


@pytest.mark.parametrize("path", BUILT, ids=lambda path: path.name)
def test_built_training_artifacts_hold_no_gpqa_question(path: Path) -> None:
    problems = built_problems(path)
    texts = []
    for column in ("Question", "Pre-Revision Question"):
        texts += [
            row[column] for row in gpqa_rows()
            if row[column] and row[column].strip()
        ]
    index = build_question_index(texts)
    leaked = [p[:80] for p in problems if contains_benchmark_question(p, index)]
    assert not leaked, leaked[:5]


@pytest.mark.parametrize("path", BUILT, ids=lambda path: path.name)
def test_built_artifacts_hold_no_screened_benchmark_question(path: Path) -> None:
    """Every artifact, the teacher's questions included, is clean against
    all of ``STEM_QUESTION_TARGETS``: MMLU, MMLU-Pro, GPQA and the
    SciQ/ARC/OpenBookQA validation and test splits."""

    problems = built_problems(path)
    if not all(target.exists() for target, _ in STEM_QUESTION_TARGETS):
        pytest.skip("screen targets are not all downloaded")
    index = production_index()
    leaked = [p[:80] for p in problems if contains_benchmark_question(p, index)]
    assert not leaked, leaked[:5]


def test_science_trace_corpus_shares_no_question_with_the_rl_pool() -> None:
    """The distilled corpus, indexed as an owner screen would index it,
    catches no question of the RL partition, and vice versa."""

    rl_pool, corpus = SCIENCE_TRACE_CHAIN[0], SCIENCE_TRACE_CHAIN[3]
    rl_problems, sft_problems = built_problems(rl_pool), built_problems(corpus)

    def stem(problem: str) -> str:
        parsed = split_options(problem)
        return parsed.question if parsed is not None else problem

    for candidates, references in (
        (rl_problems, sft_problems), (sft_problems, rl_problems)
    ):
        for template_documents in (QUESTION_TEMPLATE_DOCUMENTS, None):
            index = build_question_index(
                [stem(problem) for problem in references], template_documents
            )
            leaked = [
                p[:80] for p in candidates if contains_benchmark_question(p, index)
            ]
            assert not leaked, leaked[:5]


@pytest.mark.parametrize("owner", DEFAULT_OWNERS, ids=lambda path: path.name)
@pytest.mark.parametrize(
    "partition", SCIENCE_TRACE_CHAIN[:2], ids=lambda path: path.name
)
def test_science_partitions_hold_no_question_an_owner_serves(
    owner: Path, partition: Path
) -> None:
    """The owner screen, rerun against the owners the builder names now,
    catches nothing: a rebuilt owner cannot silently serve a science question
    twice (the knowledge corpus was rebuilt as v5 after the split was)."""

    from scripts.build_science_mc_rl_prompts import owner_stems

    if not owner.exists():
        pytest.skip(f"{owner} is not built")
    problems = built_problems(partition)
    index = build_question_index(owner_stems(owner))
    served = [p[:80] for p in problems if contains_benchmark_question(p, index)]
    assert not served, served[:5]
