from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import build_math_mix_dataset as builder
from fresh_lejepa_train import (
    OnePassTokenStream,
    cumulative_training_step,
    required_stream_tokens,
    validate_deterministic_dataset_manifest,
    validate_one_pass_capacity,
)


def test_exact_one_pass_budget_matches_loader_geometry():
    assert required_stream_tokens(2_000, 524_288, 8) == 1_048_592_000


def test_stream_budget_tracks_configured_accumulation(monkeypatch):
    monkeypatch.setenv("GRAD_ACCUM_STEPS", "16")
    assert required_stream_tokens(2_000, 524_288) == 1_048_608_000


def test_resume_steps_remain_cumulative_across_repeated_resumes():
    first_checkpoint = cumulative_training_step(0, 1_500)
    second_checkpoint = cumulative_training_step(first_checkpoint, 100)
    final_checkpoint = cumulative_training_step(second_checkpoint, 400)
    assert (first_checkpoint, second_checkpoint, final_checkpoint) == (
        1_500,
        1_600,
        2_000,
    )


def test_integer_source_budgets_sum_exactly_without_rng():
    fractions = {"web": 0.45, "math": 0.25, "qa": 0.30}
    budgets = builder.allocate_token_budgets(1_048_592_000, fractions)
    assert budgets == {"web": 471_866_400, "math": 262_148_000, "qa": 314_577_600}
    assert sum(budgets.values()) == 1_048_592_000
    assert "import random" not in Path(builder.__file__).read_text()

    with pytest.raises(ValueError, match="sum exactly"):
        builder.allocate_token_budgets(
            100, {"web": 0.5, "math": 0.499_999_5}
        )
    with pytest.raises(ValueError, match="too small"):
        builder.allocate_token_budgets(1, {"web": 0.5, "math": 0.5})


def test_deterministic_fraction_has_no_random_placement():
    assert [builder.deterministic_fraction(i, 0.5) for i in range(8)] == [
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    assert sum(builder.deterministic_fraction(i, 0.21) for i in range(100)) == 21


def test_exact_document_stream_is_deterministic_and_only_truncates_final_doc():
    def build() -> list[tuple[str, list[int], int]]:
        sources = {
            "long": iter([np.arange(4), np.arange(10, 14)]),
            "short": iter([np.arange(20, 23), np.arange(30, 35)]),
        }
        return [
            (name, document.tolist(), truncated)
            for name, document, truncated in builder.exact_document_stream(
                sources, {"long": 4, "short": 4}, 8
            )
        ]

    expected = [
        ("long", [0, 1, 2, 3], 0),
        ("short", [20, 21, 22], 0),
        ("short", [30], 4),
    ]
    assert build() == expected
    assert build() == expected
    assert sum(len(document) for _, document, _ in expected) == 8


def test_exact_document_stream_fails_instead_of_reusing_exhausted_source():
    sources = {
        "first": iter([np.arange(4)]),
        "second": iter([np.arange(10, 14), np.arange(20, 24)]),
    }
    with pytest.raises(RuntimeError, match="source 'first' exhausted"):
        list(
            builder.exact_document_stream(
                sources, {"first": 5, "second": 5}, 10
            )
        )


def test_one_pass_stream_refuses_to_wrap(tmp_path: Path):
    first = np.arange(7, dtype=np.uint16)
    second = np.arange(7, 12, dtype=np.uint16)
    builder.write_shard(tmp_path / "fineweb_train_000000.bin", first)
    builder.write_shard(tmp_path / "fineweb_train_000001.bin", second)
    pattern = str(tmp_path / "fineweb_train_*.bin")

    stream = OnePassTokenStream(pattern)
    np.testing.assert_array_equal(stream.take(12).numpy(), np.arange(12))
    with pytest.raises(RuntimeError, match="refusing to wrap"):
        stream.take(1)

    assert validate_one_pass_capacity(
        pattern, 1, 4, loader_spans_per_step=8
    ) == (12, 12)
    assert validate_one_pass_capacity(
        pattern, 1, 4, exact=True, loader_spans_per_step=8
    ) == (12, 12)
    with pytest.raises(ValueError, match="refusing to wrap"):
        validate_one_pass_capacity(
            pattern, 2, 4, loader_spans_per_step=8
        )
    with pytest.raises(ValueError, match="exact one-pass"):
        validate_one_pass_capacity(
            pattern, 1, 3, exact=True, loader_spans_per_step=8
        )


def test_strict_manifest_requires_exact_size_one_pass_and_no_rng(tmp_path: Path):
    path = tmp_path / "mix_manifest.json"
    manifest = {
        "total_tokens": 12,
        "one_pass": True,
        "rng": "none",
        "ordering": "deterministic_least_completed_token_budget_no_rng",
        "training_steps": 1,
        "train_batch_tokens": 4,
        "loader_spans_per_step": 8,
        "required_stream_tokens": 12,
    }
    path.write_text(json.dumps(manifest))
    validation_args = {
        "required_tokens": 12,
        "training_steps": 1,
        "train_batch_tokens": 4,
        "loader_spans_per_step": 8,
    }
    assert (
        validate_deterministic_dataset_manifest(tmp_path, **validation_args)
        == manifest
    )

    invalid_fields = (
        ("total_tokens", 11),
        ("one_pass", False),
        ("rng", "seeded"),
        ("ordering", "random"),
    )
    for key, invalid in invalid_fields:
        broken = dict(manifest)
        broken[key] = invalid
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError):
            validate_deterministic_dataset_manifest(tmp_path, **validation_args)

    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="loader geometry"):
        validate_deterministic_dataset_manifest(
            tmp_path, **(validation_args | {"loader_spans_per_step": 16})
        )


def test_deepmind_pairs_are_module_round_robin(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(builder, "DEEPMIND_EASY_MODULES", ["first", "second"])
    (tmp_path / "first.txt").write_text("q1\na1\nq3\na3\n")
    (tmp_path / "second.txt").write_text("q2\na2\n")
    assert list(builder.deepmind_qa_pairs(tmp_path)) == [
        ("q1", "a1"),
        ("q2", "a2"),
        ("q3", "a3"),
    ]
