"""Focused contracts for the immutable answer-only OPSD math mixture."""

from __future__ import annotations

from argparse import Namespace
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from postraining.opsd import prepare_math_mixture as mixture
from postraining.opsd.manifest import validate_final_answer_manifest


LEGACY_PREFIX = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: $Answer (without quotes) where "
    "$Answer is the answer to the problem.\n\n"
)
LEGACY_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


class _WhitespaceTokenizer:
    def __init__(self, **_kwargs):
        pass

    def bos_id(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        return list(range(1, len(text.split()) + 1))


def _wrapped(problem: str) -> str:
    return LEGACY_PREFIX + problem + LEGACY_SUFFIX


def _deepmind(
    index: int,
    *,
    answer: str | None = None,
    problem: str | None = None,
) -> dict:
    return {
        "prompt": [
            {
                "role": "user",
                "content": _wrapped(problem or f"Problem deepmind {index}"),
            }
        ],
        "reward_model": {
            "ground_truth": answer or f"word{index}",
            "style": "rule",
        },
        "extra_info": {"index": f"module/{index}", "module": "module"},
    }


def _gsm8k(index: int) -> dict:
    return {
        "data_source": "gsm8k_train",
        "prompt": [{"content": f"Problem gsm {index}{LEGACY_SUFFIX}"}],
        "ability": "MATH",
        "reward_model": {"ground_truth": str(1000 + index)},
        "extra_info": {"index": f"gsm8k_train_{index}"},
    }


def _dapo(index: int) -> dict:
    return {
        "data_source": "math_dapo",
        "prompt": [{"role": "user", "content": _wrapped(f"Problem dapo {index}")}],
        "ability": "MATH",
        "reward_model": {
            "ground_truth": str(2000 + index),
            "style": "rule-lighteval/MATH_v2",
        },
        "extra_info": {"index": f"dapo-{index}"},
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_canonicalization_assigns_explicit_sources_and_reward_contracts():
    deepmind = mixture.canonical_source_record(
        _deepmind(1, answer="True"), mixture.SOURCE_SPECS[0]
    )
    gsm8k = mixture.canonical_source_record(_gsm8k(2), mixture.SOURCE_SPECS[1])
    dapo = mixture.canonical_source_record(_dapo(3), mixture.SOURCE_SPECS[2])

    assert deepmind["source"] == "deepmind_math"
    assert deepmind["reward_style"] == "rule"
    assert deepmind["solution"] == "True"
    assert gsm8k["source"] == "gsm8k"
    assert gsm8k["reward_style"] == "rule-lighteval/MATH_v2"
    assert dapo["source"] == "dapo_math_17k"
    assert dapo["problem"] == "Problem dapo 3"
    assert all(
        record["reference_kind"] == "final_answer"
        for record in (deepmind, gsm8k, dapo)
    )


def test_physical_deduplication_quarantines_conflicting_rows(tmp_path: Path):
    duplicate_path = tmp_path / "duplicate.parquet"
    row = _deepmind(0)
    _write_rows(duplicate_path, [row, row])
    unique, physical = mixture.deduplicate_source(
        duplicate_path, mixture.DEEPMIND_SOURCE_ID
    )
    assert physical == 2
    assert [record["reward_model"] for record in unique] == [row["reward_model"]]

    conflict_path = tmp_path / "conflict.parquet"
    survivor = _deepmind(1)
    _write_rows(conflict_path, [row, _deepmind(0, answer="different"), survivor])
    unique, physical = mixture.deduplicate_source(
        conflict_path, mixture.DEEPMIND_SOURCE_ID
    )
    assert unique == [survivor]
    assert physical == 3

    _write_rows(conflict_path, [row, _deepmind(0, problem="Different prompt")])
    with pytest.raises(ValueError, match="ambiguous"):
        mixture.deduplicate_source(conflict_path, mixture.DEEPMIND_SOURCE_ID)


def test_exact_chat_deduplication_preserves_case_and_quarantines_contract_conflicts():
    first = mixture.canonical_source_record(
        _deepmind(0, answer="5", problem="Find X."), mixture.SOURCE_SPECS[0]
    )
    duplicate = {**first, "example_id": "deepmind_math:duplicate"}
    case_distinct = mixture.canonical_source_record(
        _deepmind(1, answer="5", problem="Find x."), mixture.SOURCE_SPECS[0]
    )
    unique, duplicates, conflicts = mixture.deduplicate_canonical_records(
        [first, duplicate, case_distinct]
    )
    assert unique == [first, case_distinct]
    assert duplicates == {"deepmind_math": 1}
    assert conflicts == {}
    conflicting = {**duplicate, "reward_style": "rule-lighteval/MATH_v2"}
    quarantined, _, conflicts = mixture.deduplicate_canonical_records(
        [first, conflicting, case_distinct]
    )
    assert quarantined == [case_distinct]
    assert conflicts == Counter({"deepmind_math": 2})


def test_derangement_is_position_matched_and_handles_exact_text_answers():
    records = [
        {
            "example_id": f"deepmind_math:{index}",
            "source": "deepmind_math",
            "solution": answer,
            "reward_style": "rule",
            "answer_slot_tokens": 9,
        }
        for index, answer in enumerate(("True", "False", "red", "blue"))
    ]
    records += [
        {
            "example_id": f"gsm8k:{index}",
            "source": "gsm8k",
            "solution": str(index + 1),
            "reward_style": "rule-lighteval/MATH_v2",
            "answer_slot_tokens": 9,
        }
        for index in range(4)
    ]
    donors = mixture.deranged_answer_donors(records, seed=17)

    assert Counter(row["solution"] for row in donors.values()) == Counter(
        row["solution"] for row in records
    )
    for record in records:
        donor = donors[record["example_id"]]
        assert donor["answer_slot_tokens"] == record["answer_slot_tokens"]
        assert donor["source"] == record["source"]
        assert mixture.answer_key(donor) != mixture.answer_key(record)

    assert mixture.answer_key(
        {"solution": "5", "reward_style": "rule"}
    ) == mixture.answer_key(
        {"solution": "5.0", "reward_style": "rule-lighteval/MATH_v2"}
    )


def test_builder_makes_exact_clean_gate_and_immutable_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(mixture, "GPT2BPETokenizer", _WhitespaceTokenizer)
    deepmind_path = tmp_path / "deepmind.parquet"
    gsm8k_path = tmp_path / "gsm8k.parquet"
    dapo_path = tmp_path / "dapo.parquet"
    sft_path = tmp_path / "sft.parquet"
    _write_rows(
        deepmind_path,
        [
            *[_deepmind(index) for index in range(9)],
            _deepmind(99, answer="1006", problem="Problem gsm 6"),
        ],
    )
    _write_rows(gsm8k_path, [_gsm8k(index) for index in range(7)])
    _write_rows(dapo_path, [_dapo(index) for index in range(3)])
    contaminated = {
        "Problem deepmind 0",
        "Problem gsm 0",
        "Problem dapo 0",
    }
    _write_rows(sft_path, [{"problem": problem} for problem in contaminated])
    prefix = tmp_path / "mixture"
    args = Namespace(
        dapo_source=str(dapo_path),
        deepmind_source=str(deepmind_path),
        gsm8k_source=str(gsm8k_path),
        sft_corpus=str(sft_path),
        output_prefix=str(prefix),
        gate_rows=8,
        max_prompt_length=1000,
        max_completion_length=64,
        context_tokens=1000,
        seed=23,
    )

    manifest = mixture.build(args)
    validate_final_answer_manifest(manifest)

    wrong_prompt_schema = {
        **manifest,
        "answer_fence_prompt_schema": "legacy/wrong",
    }
    with pytest.raises(ValueError, match="answer-fence prompt schema"):
        validate_final_answer_manifest(wrong_prompt_schema)
    old_policy = {**manifest, "math_corpus_policy_sha256": "old"}
    with pytest.raises(ValueError, match="reviewed math corpus policy"):
        validate_final_answer_manifest(old_policy)
    missing_identity = {
        **manifest,
        "sources": {
            source: {key: value for key, value in metadata.items()
                     if key != "math_corpus_identity"}
            for source, metadata in manifest["sources"].items()
        },
    }
    with pytest.raises(ValueError, match="effective source corpus identities"):
        validate_final_answer_manifest(missing_identity)
    train_path, gate_path, manifest_path = mixture.output_paths(prefix)
    train = pq.read_table(train_path).to_pylist()
    gate = pq.read_table(gate_path).to_pylist()
    train_ids = {row["example_id"] for row in train}
    gate_ids = {row["example_id"] for row in gate}

    assert manifest["source_quotas"] == {
        "deepmind_math": 24,
        "gsm8k": 18,
        "dapo_math_17k": 6,
    }
    assert manifest["groups_per_cycle"] == 48
    assert manifest["gate_source_rows"] == {
        "dapo_math_17k": 1,
        "deepmind_math": 4,
        "gsm8k": 3,
    }
    assert Counter(row["source"] for row in gate) == manifest["gate_source_rows"]
    assert len(gate) == 8
    assert train_ids.isdisjoint(gate_ids)
    assert {
        mixture.normalized_text(row["problem"]) for row in train
    }.isdisjoint(mixture.normalized_text(row["problem"]) for row in gate)
    assert contaminated.isdisjoint(row["problem"] for row in gate)
    assert manifest["sources"]["gsm8k"]["exact_prompt_duplicates"] == 0
    assert all(row["permuted_donor_id"] in train_ids for row in train)
    assert all(row["permuted_donor_id"] in gate_ids for row in gate)
    assert all(
        row["teacher_prompt_tokens"] == row["permuted_teacher_prompt_tokens"]
        for row in train + gate
    )
    assert manifest["train_sha256"] == mixture.file_sha256(train_path)
    assert manifest["gate_sha256"] == mixture.file_sha256(gate_path)
    assert manifest_path.exists()
    assert not list(tmp_path.glob("*.tmp"))

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        mixture.build(args)


def test_atomic_publication_cleans_partial_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    train_path, gate_path, manifest_path = mixture.output_paths(tmp_path / "atomic")
    real_publish = mixture._publish_no_replace
    calls = 0

    def fail_second_publish(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        real_publish(source, destination)

    monkeypatch.setattr(mixture, "_publish_no_replace", fail_second_publish)
    with pytest.raises(OSError, match="injected publication failure"):
        mixture._write_outputs_atomically(
            [{"value": 1}],
            [{"value": 2}],
            train_path,
            gate_path,
            manifest_path,
            {"schema": "test"},
        )
    assert not train_path.exists()
    assert not gate_path.exists()
    assert not manifest_path.exists()
    assert not list(tmp_path.glob("*.tmp"))
