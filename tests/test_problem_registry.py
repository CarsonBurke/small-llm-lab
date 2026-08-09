from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from postraining.decontaminate import (
    DEFAULT_NGRAM_SIZE,
    INDEX_SCHEMA,
    ContaminationIndex,
    ProblemGuard,
    index_words,
    load_guard,
    ngram_hashes,
)
from postraining.problem_registry import (
    SPLIT_PRIORITY,
    ProblemRegistry,
    SourceDeclaration,
    load_sources,
    problem_key,
    prompt_content,
    read_problems,
    strip_framing,
)
from scripts.build_problem_registry import protected_problems

WORD_PROBLEM = (
    "Janet has 16 eggs and she eats three of them for breakfast every single "
    "morning before she bakes muffins with four more of the eggs she has left."
)
SECOND_PROBLEM = (
    "A train leaves the station at nine in the morning travelling north at "
    "sixty kilometres per hour for three and a half hours without stopping."
)


def write_parquet(path, column, values):
    pq.write_table(pa.table({column: pa.array(values, type=pa.string())}), path)
    return path


# -- identity ------------------------------------------------------------


def test_framing_variants_reduce_to_one_key():
    bare = "What is 2 + 2?"
    dapo = (
        "Solve the following math problem step by step. The last line of your "
        "response should be of the form Answer: $Answer (without quotes) where "
        "$Answer is the answer to the problem.\n\n"
        f"{bare}\n\n"
        'Remember to put your answer on its own line after "Answer:".'
    )
    fenced = (
        f"{bare}\n\nStart your response with <think> and reason until "
        "</think>, then end it with only the final answer inside "
        "<answer></answer>."
    )
    assert problem_key(dapo) == problem_key(bare)
    assert problem_key(fenced) == problem_key(bare)


def test_identity_ignores_case_and_spacing_but_not_operators():
    assert problem_key("Compute 5 + 3") == problem_key("compute   5 + 3\n")
    assert problem_key("Compute 5 + 3") != problem_key("Compute 5 - 3")


def test_framing_list_matches_the_canonical_prompt_module():
    # `postraining.math_prompt` is the source of truth for prompt wording; the
    # registry duplicates the list only to stay importable without torch.
    from postraining.math_prompt import SOURCE_INSTRUCTIONS

    from postraining.problem_registry import FRAMING_INSTRUCTIONS

    assert set(FRAMING_INSTRUCTIONS) == set(SOURCE_INSTRUCTIONS)


def test_strip_framing_tolerates_a_problem_containing_the_word_answer():
    # `math_prompt.strip_math_prompt_framing` raises here by design. The
    # registry must not: refusing to hash a problem would drop it from
    # decontamination entirely, which is the opposite of failing closed.
    text = "The answer: is written on the board. What is 2 + 2?"
    assert strip_framing(text) == text


# -- priority ------------------------------------------------------------


def test_higher_priority_split_wins_regardless_of_declaration_order(tmp_path):
    shared = "What is the value of x when 3x equals twelve exactly?"
    sft = SourceDeclaration(
        "pool", write_parquet(tmp_path / "sft.parquet", "problem", [shared]), "sft"
    )
    evaluation = SourceDeclaration(
        "panel", write_parquet(tmp_path / "eval.parquet", "problem", [shared]), "eval"
    )
    for order in ([sft, evaluation], [evaluation, sft]):
        registry = ProblemRegistry.build(order)
        assert registry.split_of(shared) == "eval"
        assert registry.provenance["conflicts"] == {"pool": 1}


def test_pretrain_is_excluded_from_every_other_split(tmp_path):
    sources = [
        SourceDeclaration(
            split,
            write_parquet(tmp_path / f"{split}.parquet", "problem", [f"problem {split}"]),
            split,
        )
        for split in ("eval", "rl", "sft")
    ]
    registry = ProblemRegistry.build(sources)
    assert len(registry.excluded_from("pretrain")) == 3
    assert len(registry.excluded_from("sft")) == 2
    assert len(registry.excluded_from("rl")) == 1
    assert registry.excluded_from("eval") == set()


def test_repeated_rows_are_counted_as_one_problem(tmp_path):
    source = SourceDeclaration(
        "repeats",
        write_parquet(tmp_path / "r.parquet", "problem", ["same problem"] * 100),
        "rl",
    )
    registry = ProblemRegistry.build([source])
    record = registry.provenance["sources"]["repeats"]
    assert record["rows"] == 100
    assert record["distinct_problems"] == 1
    assert record["claimed"] == 1
    assert record["yielded_to_higher_priority"] == 0


def test_unknown_split_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown split"):
        SourceDeclaration("x", tmp_path / "x.parquet", "validation")


def test_split_priority_lists_pretrain_last():
    # `excluded_from("pretrain")` returning everything depends on this.
    assert SPLIT_PRIORITY[-1] == "pretrain"


# -- reading -------------------------------------------------------------


def test_chat_shaped_prompts_are_read(tmp_path):
    path = tmp_path / "chat.parquet"
    pq.write_table(
        pa.table(
            {
                "prompt": pa.array(
                    [[{"role": "user", "content": "What is 2 + 2?"}]],
                    type=pa.list_(
                        pa.struct([("role", pa.string()), ("content", pa.string())])
                    ),
                )
            }
        ),
        path,
    )
    assert list(read_problems(path, ("problem", "prompt"))) == ["What is 2 + 2?"]


def test_missing_column_names_what_the_file_actually_has(tmp_path):
    path = write_parquet(tmp_path / "q.parquet", "question", ["a problem"])
    with pytest.raises(ValueError, match="question"):
        list(read_problems(path, ("problem", "prompt")))


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DECLARED_SOURCES = REPOSITORY_ROOT / "postraining" / "problem_sources.json"


def test_declared_sources_all_exist_and_expose_their_column():
    sources = load_sources(DECLARED_SOURCES, REPOSITORY_ROOT)
    assert sources, "the source declaration must not be empty"
    for source in sources:
        assert source.path.exists(), source.path
        available = set(pq.ParquetFile(source.path).schema_arrow.names)
        assert available & set(source.columns), (source.name, sorted(available))


def test_every_declared_split_is_known():
    sources = load_sources(DECLARED_SOURCES, REPOSITORY_ROOT)
    assert {source.split for source in sources} <= set(SPLIT_PRIORITY)


# -- storage -------------------------------------------------------------


def test_registry_round_trips_and_detects_tampering(tmp_path):
    source = SourceDeclaration(
        "panel", write_parquet(tmp_path / "e.parquet", "problem", [WORD_PROBLEM]), "eval"
    )
    registry = ProblemRegistry.build([source])
    registry.write(tmp_path / "v1")
    restored = ProblemRegistry.read(tmp_path / "v1")
    assert restored.split_of(WORD_PROBLEM) == "eval"
    assert len(restored) == len(registry)

    table = pq.read_table(tmp_path / "v1" / "registry.parquet")
    pq.write_table(
        pa.table(
            {
                "key": table["key"],
                "split": pa.array(["pretrain"] * table.num_rows, type=pa.string()),
            }
        ),
        tmp_path / "v1" / "registry.parquet",
    )
    with pytest.raises(ValueError, match="not the one its manifest describes"):
        ProblemRegistry.read(tmp_path / "v1")


def test_index_round_trips_and_detects_tampering(tmp_path):
    index = ContaminationIndex.build([WORD_PROBLEM], ngram_size=13)
    index.write(tmp_path)
    restored = ContaminationIndex.read(tmp_path)
    assert len(restored) == len(index)
    assert restored.hit_count(f"prefix {WORD_PROBLEM} suffix") > 0

    np.save(tmp_path / "ngrams.npy", np.zeros(4, dtype=np.uint64))
    with pytest.raises(ValueError, match="not the one its manifest describes"):
        ContaminationIndex.read(tmp_path)


def test_index_write_is_byte_deterministic(tmp_path):
    first = ContaminationIndex.build([WORD_PROBLEM, SECOND_PROBLEM])
    second = ContaminationIndex.build([SECOND_PROBLEM, WORD_PROBLEM])
    assert first.write(tmp_path / "a") == second.write(tmp_path / "b")


# -- n-gram detection ----------------------------------------------------


def test_ngram_hashing_is_position_independent_and_order_sensitive():
    words = index_words(WORD_PROBLEM)
    direct = ngram_hashes(words, 13)
    shifted = ngram_hashes(["preamble", *words], 13)
    assert set(direct.tolist()) < set(shifted.tolist())
    assert set(ngram_hashes(list(reversed(words)), 13).tolist()).isdisjoint(
        direct.tolist()
    )


def test_ngram_hashing_survives_punctuation_and_case_differences():
    quoted = WORD_PROBLEM.upper().replace(" ", "  ").replace(".", " ...")
    index = ContaminationIndex.build([WORD_PROBLEM])
    assert index.hit_count(quoted) > 0


def test_problems_shorter_than_the_window_are_reported_not_hidden():
    short = "What is the difference between -221017 and -1429.06?"
    index = ContaminationIndex.build([short, WORD_PROBLEM], ngram_size=13)
    assert index.short_rows == 1
    assert index.covered_rows == 1
    assert index.hit_count(f"see also: {short}") == 0


def test_ngram_size_below_two_is_refused():
    with pytest.raises(ValueError, match="at least 2"):
        ContaminationIndex.build([WORD_PROBLEM], ngram_size=1)


def test_empty_index_admits_everything():
    index = ContaminationIndex.build([])
    assert len(index) == 0
    assert index.hit_count(WORD_PROBLEM) == 0


# -- guard ---------------------------------------------------------------


def test_guard_catches_whole_document_and_embedded_copies():
    guard = ProblemGuard.from_problems([WORD_PROBLEM])
    assert guard.reason(WORD_PROBLEM) == "registry_exact"
    assert guard.reason(f"Blog post.\n\n{WORD_PROBLEM}\n\nThanks!") == "registry_ngram"
    assert guard.reason("An unrelated paragraph about weather and gardening.") is None


def test_guard_checks_quality_keys_as_well_as_the_document(tmp_path):
    # A QA source builds one document from several problems, so the exact test
    # must run against each problem, not only the concatenation.
    guard = ProblemGuard.from_problems(["What is 7 times 8?"])
    document = "Some other question\nAnswer: 4\nWhat is 7 times 8?\nAnswer: 56"
    assert guard.reason(document) is None
    assert (
        guard.reason(document, ("Some other question", "What is 7 times 8?"))
        == "registry_exact"
    )


def test_guard_min_hits_raises_the_bar():
    guard = ProblemGuard.from_problems([WORD_PROBLEM], min_ngram_hits=1000)
    assert guard.reason(f"Blog post.\n\n{WORD_PROBLEM}\n\nThanks!") is None


def test_load_guard_reads_a_written_registry(tmp_path):
    source = SourceDeclaration(
        "panel", write_parquet(tmp_path / "e.parquet", "problem", [WORD_PROBLEM]), "eval"
    )
    ProblemRegistry.build([source]).write(tmp_path / "v1")
    ContaminationIndex.build([WORD_PROBLEM], splits=("eval",)).write(tmp_path / "v1")
    guard, registry = load_guard(tmp_path / "v1")
    assert guard.reason(WORD_PROBLEM) == "registry_exact"
    provenance = guard.provenance(registry)
    assert provenance["split"] == "pretrain"
    assert provenance["excluded_problem_keys"] == 1
    assert provenance["registry_sha256"]
    assert provenance["index_sha256"]
    assert provenance["index_splits"] == ["eval"]
    assert provenance["protected_splits"] == ["eval"]


def test_registry_schema_mismatch_is_refused(tmp_path):
    source = SourceDeclaration(
        "panel", write_parquet(tmp_path / "e.parquet", "problem", [WORD_PROBLEM]), "eval"
    )
    ProblemRegistry.build([source]).write(tmp_path / "v1")
    manifest_path = tmp_path / "v1" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema"] = "math_problem_registry/v0"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="is not"):
        ProblemRegistry.read(tmp_path / "v1")


# -- the two defects a 2026-08-06 red-team review found -------------------


def test_indexing_ignores_instruction_framing():
    """The index must protect the problem, not the template around it.

    Measured before this was fixed: indexing raw DAPO prompts put 19
    boilerplate 13-grams into the index, and the corpus builder then refused
    20.2% of `deepmind_math` and 16.9% of `openmath_instruct` -- documents
    whose only crime was being rendered with the same instruction wrapper.
    """
    preamble = (
        "Solve the following math problem step by step. The last line of your "
        "response should be of the form Answer: $Answer (without quotes) where "
        "$Answer is the answer to the problem.\n\n"
    )
    guard = ProblemGuard.from_problems([preamble + WORD_PROBLEM])
    unrelated = (
        "A train leaves the station at noon carrying two hundred passengers "
        "bound for the coastal city several hours away."
    )
    assert guard.reason(preamble + unrelated) is None
    assert guard.reason(preamble + WORD_PROBLEM) == "registry_exact"
    assert guard.reason(f"Worksheet.\n{WORD_PROBLEM}\nSolution below.") == "registry_ngram"


def test_repetitive_ngrams_do_not_enter_the_index():
    """A window one word dominates is notation, not a problem statement.

    This bounds the damage rather than eliminating it: an alternating pattern
    like `a_1 a_2 ...` still has enough word-level variety that half its
    windows survive, because `index_words` drops the subscript operators that
    would distinguish it. The measured false-positive rate that leaves is the
    number to judge, and `scripts/audit_problem_overlap.py` reports it.
    """
    banner = " ".join(["=" * 4, *("page" for _ in range(20))])
    index = ContaminationIndex.build([banner, WORD_PROBLEM])
    assert index.provenance["low_diversity_ngrams"] > 0
    assert index.hit_count("intro " + banner) == 0
    assert index.hit_count(f"quoted: {WORD_PROBLEM}") > 0


def test_guard_refuses_an_index_that_does_not_cover_the_protected_splits(tmp_path):
    """An `eval`-only index may not be used as if it protected rl and sft."""
    evaluation = SourceDeclaration(
        "panel", write_parquet(tmp_path / "e.parquet", "problem", [WORD_PROBLEM]), "eval"
    )
    training = SourceDeclaration(
        "pool", write_parquet(tmp_path / "s.parquet", "problem", [SECOND_PROBLEM]), "sft"
    )
    registry = ProblemRegistry.build([evaluation, training])
    narrow = ContaminationIndex.build([WORD_PROBLEM], splits=("eval",))
    with pytest.raises(ValueError, match="sft"):
        ProblemGuard(registry, narrow, split="pretrain")
    wide = ContaminationIndex.build(
        [WORD_PROBLEM, SECOND_PROBLEM], splits=("eval", "sft")
    )
    guard = ProblemGuard(registry, wide, split="pretrain")
    assert guard.reason(f"blog\n{SECOND_PROBLEM}\nend") == "registry_ngram"


def test_a_split_with_no_problems_needs_no_index_coverage(tmp_path):
    source = SourceDeclaration(
        "panel", write_parquet(tmp_path / "e.parquet", "problem", [WORD_PROBLEM]), "eval"
    )
    registry = ProblemRegistry.build([source])
    index = ContaminationIndex.build([WORD_PROBLEM], splits=("eval",))
    assert ProblemGuard(registry, index, split="pretrain").protected_splits == {"eval"}


def test_registry_records_the_bytes_each_source_came_from(tmp_path):
    path = write_parquet(tmp_path / "e.parquet", "problem", [WORD_PROBLEM])
    registry = ProblemRegistry.build([SourceDeclaration("panel", path, "eval")])
    import hashlib

    assert registry.provenance["sources"]["panel"]["sha256"] == (
        hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_written_index_reports_its_own_digest(tmp_path):
    index = ContaminationIndex.build([WORD_PROBLEM], splits=("eval",))
    digest = index.write(tmp_path / "v1")
    assert index.provenance["ngrams_sha256"] == digest
    assert ContaminationIndex.read(tmp_path / "v1").provenance == index.provenance


def test_unsorted_hashes_are_refused():
    import numpy as np

    with pytest.raises(ValueError, match="sorted"):
        ContaminationIndex(
            np.array([5, 2], dtype=np.uint64), {"schema": INDEX_SCHEMA, "ngram_size": 13}
        )


def test_duplicate_source_names_are_refused(tmp_path):
    path = write_parquet(tmp_path / "e.parquet", "problem", [WORD_PROBLEM])
    declaration = tmp_path / "sources.json"
    declaration.write_text(
        json.dumps(
            {
                "sources": [
                    {"name": "panel", "path": "e.parquet", "split": "eval"},
                    {"name": "panel", "path": "e.parquet", "split": "sft"},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate source names"):
        load_sources(declaration, tmp_path)


def test_a_chat_prompt_keys_on_its_user_turn():
    assert prompt_content(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": WORD_PROBLEM},
        ]
    ) == WORD_PROBLEM


# -- positive control ----------------------------------------------------


def control_sources(tmp_path):
    return [
        SourceDeclaration(
            "panel",
            write_parquet(tmp_path / "eval.parquet", "problem", [WORD_PROBLEM]),
            "eval",
        ),
        SourceDeclaration(
            "pool",
            write_parquet(tmp_path / "sft.parquet", "problem", [SECOND_PROBLEM]),
            "sft",
        ),
    ]


def test_detection_control_passes_on_an_index_that_covers_what_it_claims(tmp_path):
    from scripts.build_problem_registry import detection_control

    sources = control_sources(tmp_path)
    splits = {"eval", "sft"}
    index = ContaminationIndex.build(
        protected_problems(sources, splits), splits=splits
    )
    control = detection_control(index, sources, splits, per_source=10)
    assert control["indexable"] == 2
    assert control["detected"] == 2
    assert control["by_split"]["eval"]["detected"] == 1
    assert control["by_split"]["sft"]["detected"] == 1


def test_detection_control_fires_on_an_index_that_lies_about_its_splits(tmp_path):
    """The declaration is what every consumer trusts, so measure it.

    `ProblemGuard` refuses an index whose declared splits do not cover the
    registry's populated ones, but nothing checks that a declared split was
    actually indexed. This turns that claim into evidence at build time, while
    the source text is still in hand.
    """
    from scripts.build_problem_registry import detection_control

    sources = control_sources(tmp_path)
    # Indexed from eval alone, but the manifest claims sft too.
    index = ContaminationIndex.build(
        protected_problems(sources, {"eval"}), splits={"eval", "sft"}
    )
    assert index.splits == {"eval", "sft"}
    with pytest.raises(SystemExit, match="failed its own positive control"):
        detection_control(index, sources, {"eval", "sft"}, per_source=10)


def test_detection_control_does_not_blame_the_index_for_short_problems(tmp_path):
    """A problem below one n-gram window is protected by exact key only."""
    from scripts.build_problem_registry import detection_control

    sources = [
        SourceDeclaration(
            "panel",
            write_parquet(tmp_path / "eval.parquet", "problem", ["What is 2 + 2?"]),
            "eval",
        )
    ]
    index = ContaminationIndex.build(
        protected_problems(sources, {"eval"}), splits={"eval"}
    )
    control = detection_control(index, sources, {"eval"}, per_source=10)
    assert control["indexable"] == 0
    assert control["uncoverable"] == 1


def test_detection_control_does_not_blame_the_index_for_repetitive_problems(tmp_path):
    """Long is not the same as coverable, and length is the wrong predicate.

    This row is 18 words and yields six raw 13-grams, so a length test calls
    it indexable -- but twelve of those words are `d`, so `informative`
    rejects every window and the builder indexes nothing. Verbatim from
    `relaxed_bar_a1_math_deepmind`, where 4 of 2,879 sampled rows are of this
    shape; a control that tested length would abort the real build and blame
    the index for behaving correctly.
    """
    from scripts.build_problem_registry import detection_control

    repetitive = "Simplify (d*d*((d*(d*d**3)/d)/d*d)/d)/(d/d**6) assuming d is positive."
    assert len(index_words(strip_framing(repetitive))) > DEFAULT_NGRAM_SIZE
    sources = [
        SourceDeclaration(
            "pool",
            write_parquet(tmp_path / "sft.parquet", "problem", [repetitive]),
            "sft",
        )
    ]
    index = ContaminationIndex.build(
        protected_problems(sources, {"sft"}), splits={"sft"}
    )
    assert len(index) == 0
    control = detection_control(index, sources, {"sft"}, per_source=10)
    assert control["indexable"] == 0
    assert control["uncoverable"] == 1


def test_the_control_counts_coverage_the_way_the_builder_decides_it(tmp_path):
    """The two predicates must not be allowed to drift apart again.

    `short_rows` in the provenance is `protected_rows - covered_rows`, and
    `covered_rows` counts rows that produced an informative n-gram. Any
    control that re-derives its own notion of coverage from the field's name
    will disagree with the builder on exactly the repetitive rows above.
    """
    from scripts.build_problem_registry import detection_control

    problems = [
        WORD_PROBLEM,
        "What is 2 + 2?",
        "Simplify (d*d*((d*(d*d**3)/d)/d*d)/d)/(d/d**6) assuming d is positive.",
    ]
    sources = [
        SourceDeclaration(
            "pool",
            write_parquet(tmp_path / "sft.parquet", "problem", problems),
            "sft",
        )
    ]
    index = ContaminationIndex.build(
        protected_problems(sources, {"sft"}), splits={"sft"}
    )
    control = detection_control(index, sources, {"sft"}, per_source=10)
    assert control["indexable"] == index.covered_rows == 1
    assert control["uncoverable"] == index.short_rows == 2
