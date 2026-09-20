from __future__ import annotations

import copy
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from postraining import core


def _fingerprint(prompt: list[dict]) -> str:
    serialized = json.dumps(
        prompt, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _row(
    index: str, question: str = "Compute the divisor count.", target: str = "12"
) -> dict:
    return {
        "prompt": [{"role": "user", "content": question}],
        "reward_model": {"style": "rule-lighteval/MATH_v2", "ground_truth": target},
        "extra_info": {"index": index},
        "data_source": "fixture",
    }


def _parquet(tmp_path, rows: list[dict], name: str = "math.parquet"):
    path = tmp_path / name
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


@pytest.fixture
def registry(tmp_path, monkeypatch):
    path = tmp_path / "reviews.json"
    monkeypatch.setattr(core, "_MATH_TARGET_REVIEWS_PATH", path)

    def install(entries=()):
        path.write_text(
            json.dumps({"schema": "math_target_reviews/v1", "entries": list(entries)})
        )
        return path

    install()
    return install


def _review(row: dict, action: str = "correct", corrected: str = "6") -> dict:
    entry = {
        "prompt": copy.deepcopy(row["prompt"]),
        "prompt_sha256": _fingerprint(row["prompt"]),
        "action": action,
        "expected_targets": [row["reward_model"]["ground_truth"]],
        "reason": "Reviewed arithmetic fixture.",
        "evidence": {"calculation": "A fixture's independently specified answer."},
    }
    if action == "correct":
        entry["corrected_target"] = corrected
    return entry


def test_exact_prompt_dedup_preserves_order_and_does_not_drop_reused_source_ids(
    tmp_path, registry
):
    first = _row("shared")
    second = _row("shared", "A different question.", "3")
    duplicate = _row("another-id")
    third = _row("last", "One last question.", "8")
    path = _parquet(
        tmp_path, [first, second, duplicate, first, third, second, duplicate]
    )
    audit = {}
    rows = core.load_unique_math_rows(path, audit=audit)

    assert [row["prompt"] for row in rows] == [
        first["prompt"],
        second["prompt"],
        third["prompt"],
    ]
    assert len({_fingerprint(row["prompt"]) for row in rows}) == len(rows)
    assert audit["physical_rows"] == 7
    assert audit["input_rows"] == 4
    assert audit["unique_prompt_rows"] == audit["output_rows"] == 3
    assert audit["physical_repeat_rows"] == 3
    assert audit["duplicate_prompt_rows"] == 1
    assert audit["duplicate_prompt_groups"][0]["physical_rows"] == 4
    assert audit["source_id_collisions"] == [
        {
            "source": {"index": "shared", "data_source": "fixture"},
            "prompt_sha256": [
                _fingerprint(first["prompt"]),
                _fingerprint(second["prompt"]),
            ],
        }
    ]


@pytest.mark.parametrize("difference", ["target", "style", "tests"])
def test_conflicting_effective_contract_quarantines_every_copy(
    tmp_path, registry, difference
):
    first = _row("first")
    conflict = _row("first")
    if difference == "target":
        conflict["reward_model"]["ground_truth"] = "6"
    elif difference == "style":
        conflict["reward_model"]["style"] = "rule"
    else:
        first["verification_info"] = {
            "schema": "python/v1",
            "tests": ["assert f(1) == 1"],
        }
        conflict["verification_info"] = {
            "schema": "python/v1",
            "tests": ["assert f(1) == 2"],
        }
    safe = _row("safe", "Unambiguous question.", "4")
    rows = [first, safe, conflict, first, conflict]
    if difference == "tests":
        safe["verification_info"] = None
    audit = {}
    effective = core.load_unique_math_rows(_parquet(tmp_path, rows), audit=audit)

    assert [row["prompt"] for row in effective] == [safe["prompt"]]
    assert len(audit["conflicting_prompt_groups"]) == 1
    assert audit["conflicting_prompt_groups"][0]["physical_rows"] == 4
    assert len(audit["conflicting_prompt_groups"][0]["effective_contracts"]) == 2
    assert audit["quarantined_prompt_groups"][0]["reasons"] == [
        "conflicting_effective_contracts"
    ]


@pytest.mark.parametrize(("old", "corrected"), [("12", "6"), ("403", "404")])
def test_reviewed_targets_grade_correctly_across_ids_and_file_copies(
    tmp_path, registry, old, corrected
):
    first = _row("original", target=old)
    other_id = _row("copied", target=old)
    already_correct = _row("correct", target=corrected)
    registry([_review(first, corrected=corrected)])
    path = _parquet(tmp_path, [first, other_id, first, already_correct])
    before = path.read_bytes()
    audit = {}
    rows = core.load_unique_math_rows(path, audit=audit)

    assert len(rows) == 1
    reward = rows[0]["reward_model"]
    assert core.verify_answer(
        f"Answer: {corrected}", reward["ground_truth"], core.answer_style(rows[0])
    )[0]
    assert not core.verify_answer(
        f"Answer: {old}", reward["ground_truth"], core.answer_style(rows[0])
    )[0]
    assert path.read_bytes() == before
    assert audit["conflicting_prompt_groups"] == []
    assert sum(receipt["physical_rows"] for receipt in audit["corrections"]) == 4
    assert [receipt["status"] for receipt in audit["corrections"]] == [
        "corrected",
        "corrected",
        "already_correct",
    ]
    provenance = rows[0]["extra_info"]["math_target_review"]
    assert provenance["original_target"] == old
    assert provenance["corrected_target"] == corrected
    assert provenance["prompt_sha256"] == _fingerprint(first["prompt"])

    copied = _parquet(tmp_path, [other_id], "copy.parquet")
    copied_rows = core.load_unique_math_rows(copied)
    assert core.math_corpus_identity(copied_rows) == core.math_corpus_identity(rows)
    assert (
        copied_rows[0]["extra_info"]["math_target_review"]["corrected_target"]
        == corrected
    )


def test_reviewed_quarantine_excludes_all_source_ids(tmp_path, registry):
    wrong = _row("wrong", "Find the smallest admissible d.", "10")
    duplicate = copy.deepcopy(wrong)
    duplicate["extra_info"]["index"] = "copy"
    safe = _row("safe", "Compute 2+2.", "4")
    registry([_review(wrong, action="quarantine")])
    audit = {}
    rows = core.load_unique_math_rows(
        _parquet(tmp_path, [wrong, safe, duplicate, wrong]), audit=audit
    )

    assert [row["prompt"] for row in rows] == [safe["prompt"]]
    quarantine = audit["quarantined_prompt_groups"][0]
    assert quarantine["physical_rows"] == 3
    assert quarantine["reasons"] == ["reviewed_quarantine"]
    assert quarantine["reviews"][0]["original_target"] == "10"


def test_unexpected_review_source_target_quarantines_instead_of_overwriting(
    tmp_path, registry
):
    old = _row("old")
    unexpected = _row("new-source", target="99")
    safe = _row("safe", "Unreviewed question.", "4")
    registry([_review(old)])
    audit = {}
    rows = core.load_unique_math_rows(
        _parquet(tmp_path, [old, unexpected, safe, unexpected]), audit=audit
    )

    assert [row["prompt"] for row in rows] == [safe["prompt"]]
    quarantine = audit["quarantined_prompt_groups"][0]
    assert "unexpected_review_target" in quarantine["reasons"]
    assert {
        contract["reward_model"]["ground_truth"]
        for contract in quarantine["effective_contracts"]
    } == {"6", "99"}
    assert any(
        receipt["status"] == "unexpected_target" for receipt in quarantine["reviews"]
    )
    assert not audit["corrections"][0]["retained"]


def test_prompt_identity_preserves_context_roles_order_whitespace_and_case(
    tmp_path, registry
):
    first = _row("0", "Evaluate x.")
    rows = [first]
    for index, prompt in enumerate(
        [
            [{"role": "user", "content": "Evaluate X."}],
            [{"role": "user", "content": "Evaluate x. "}],
            [{"role": "system", "content": "Evaluate x."}],
            first["prompt"] + [{"role": "user", "content": "Use x=3."}],
            [{"role": "user", "content": "Use x=3."}] + first["prompt"],
        ],
        1,
    ):
        row = _row(str(index))
        row["prompt"] = prompt
        rows.append(row)
    result = core.load_unique_math_rows(_parquet(tmp_path, rows + rows))
    assert [row["prompt"] for row in result] == [row["prompt"] for row in rows]


def test_missing_source_ids_still_deduplicate_by_full_prompt(tmp_path, registry):
    first = _row("unused")
    first["extra_info"] = {"index": None}
    second = _row("unused", "A distinct question.")
    second["extra_info"] = {"index": None}
    rows = core.load_unique_math_rows(_parquet(tmp_path, [first, second, first]))
    assert [row["prompt"] for row in rows] == [first["prompt"], second["prompt"]]


def test_corpus_identity_binds_order_labels_styles_tests_and_policy_not_source_ids(
    registry, monkeypatch
):
    rows = [_row("a"), _row("b", "Another question.", "4")]
    baseline = core.math_corpus_identity(rows)
    metadata_only = copy.deepcopy(rows)
    metadata_only[0]["extra_info"] = {"index": "renamed", "other_provenance": "copied"}
    metadata_only[0]["data_source"] = "different-parquet"
    metadata_only[0]["prompt"][0] = {
        "content": "Compute the divisor count.",
        "role": "user",
    }
    assert core.math_corpus_identity(metadata_only) == baseline
    assert core.math_corpus_identity(rows[::-1]) != baseline
    assert core.math_corpus_identity(rows + [rows[0]]) != baseline
    for field, value in [("ground_truth", "6"), ("style", "rule")]:
        changed = copy.deepcopy(rows)
        changed[0]["reward_model"][field] = value
        assert core.math_corpus_identity(changed) != baseline
    changed = copy.deepcopy(rows)
    changed[0]["prompt"].append({"role": "assistant", "content": "New context."})
    assert core.math_corpus_identity(changed) != baseline
    changed = copy.deepcopy(rows)
    changed[0]["verification_info"] = {
        "schema": "python/v1",
        "tests": ["assert f() == 1"],
    }
    assert core.math_corpus_identity(changed) != baseline
    tested_identity = core.math_corpus_identity(changed)
    changed[0]["verification_info"]["tests"] = ["assert f() == 2"]
    assert core.math_corpus_identity(changed) != tested_identity
    policy = core.math_corpus_policy_sha256()
    registry([_review(rows[0])])
    assert core.math_corpus_policy_sha256() != policy
    assert core.math_corpus_identity(rows) != baseline
    reviewed_identity = core.math_corpus_identity(rows)
    monkeypatch.setattr(core, "MATH_CORPUS_SCHEMA", "a-new-policy-version")
    assert core.math_corpus_identity(rows) != reviewed_identity


def test_review_fingerprint_must_match_full_prompt(tmp_path, registry):
    row = _row("a")
    review = _review(row)
    review["prompt"][0]["content"] += " Changed after review."
    registry([review])
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        core.load_unique_math_rows(_parquet(tmp_path, [row]))


def test_empty_effective_corpus_fails_with_quarantine_receipts(tmp_path, registry):
    first = _row("first")
    conflict = _row("second", target="99")
    audit = {}
    with pytest.raises(ValueError, match="no effective math corpus rows"):
        core.load_unique_math_rows(
            _parquet(tmp_path, [first, conflict, first]), audit=audit
        )
    assert audit["physical_rows"] == 3
    assert audit["output_rows"] == 0
    assert audit["quarantined_prompt_groups"][0]["physical_rows"] == 3
