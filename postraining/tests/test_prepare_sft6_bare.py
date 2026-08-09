"""Prompt-only migration invariants for the immutable SFT6 corpus."""

from __future__ import annotations

import pytest

from postraining.core import ANSWER_CLOSE, ANSWER_OPEN, THINK_CLOSE, THINK_OPEN
from postraining.math_prompt import LEGACY_ANSWER_FENCE_SUFFIX
from postraining.prepare_sft6_bare import migrate_row, migrate_rows


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode())


def _legacy_row(problem: str = "What is 2 + 2?") -> dict:
    completion = (
        f"{THINK_OPEN}\n2 + 2 is 4.\n{THINK_CLOSE}\n"
        f"{ANSWER_OPEN}4{ANSWER_CLOSE}"
    )
    return {
        "source": "unit",
        "problem": problem,
        "document": problem + LEGACY_ANSWER_FENCE_SUFFIX + completion,
        "final_answer": "4",
        "verified": True,
        "doc_tokens": 999,
    }


def test_migration_removes_only_legacy_prompt_prose() -> None:
    source = _legacy_row()
    migrated = migrate_row(source, _Tokenizer())
    old_completion = source["document"][
        len(source["problem"] + LEGACY_ANSWER_FENCE_SUFFIX):
    ]

    assert migrated["document"] == source["problem"] + old_completion
    assert migrated["document"].startswith(source["problem"] + THINK_OPEN)
    assert LEGACY_ANSWER_FENCE_SUFFIX not in migrated["document"]
    assert migrated["final_answer"] == source["final_answer"]
    assert migrated["source"] == source["source"]
    assert source["document"].endswith(old_completion)
    assert migrated["doc_tokens"] == len(migrated["document"].encode()) + 1


def test_migration_rejects_mixed_or_unverified_parent_rows() -> None:
    already_bare = _legacy_row()
    already_bare["document"] = already_bare["document"].replace(
        LEGACY_ANSWER_FENCE_SUFFIX, "", 1
    )
    with pytest.raises(ValueError, match="legacy prompt contract"):
        migrate_row(already_bare, _Tokenizer())

    unverified = {**_legacy_row(), "verified": False}
    with pytest.raises(ValueError, match="verified"):
        migrate_row(unverified, _Tokenizer())


def test_migration_rejects_new_document_collisions() -> None:
    row = _legacy_row()
    with pytest.raises(ValueError, match="duplicate documents"):
        migrate_rows([row, dict(row)], _Tokenizer())
