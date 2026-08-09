from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from postraining.decontaminate import ProblemGuard
from scripts import build_k3_pretrain_dataset as base
from scripts import build_k3_pretrain_dataset_checkpointed as checkpointed


def test_token_cache_round_trip(tmp_path):
    path = tmp_path / "tokens.bin"
    documents = [
        np.asarray([base.GPT2_EOT_ID, 1, 2, 3], dtype=np.int32),
        np.asarray([base.GPT2_EOT_ID, 50_256], dtype=np.int32),
    ]
    with checkpointed.TokenCacheWriter(path) as writer:
        for document in documents:
            writer.append(document)
        assert writer.tokens == 6
        assert writer.documents == 2

    restored = list(checkpointed.token_cache_documents(path))
    assert len(restored) == len(documents)
    for actual, expected in zip(restored, documents, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_token_cache_rejects_truncated_payload(tmp_path):
    path = tmp_path / "tokens.bin"
    path.write_bytes(checkpointed.TOKEN_LENGTH.pack(10) + b"\0\0")

    with pytest.raises(ValueError, match="truncated token-cache payload"):
        list(checkpointed.token_cache_documents(path))


def test_dedup_journal_restores_exact_and_formatting_sets(tmp_path):
    path = tmp_path / "dedup.bin"
    web = base.RawDocument(
        source="web",
        domain="web",
        segments=("A sufficiently long web document for deduplication.",),
        quality_keys=(),
    )
    math = base.RawDocument(
        source="math",
        domain="math",
        segments=("A sufficiently long math document for deduplication.",),
        quality_keys=(),
    )
    with path.open("wb") as journal:
        dedup = checkpointed.JournaledDeduplicator(ProblemGuard.from_problems([]), journal)
        assert dedup.accept(web)
        assert dedup.accept(math)

    restored = checkpointed.JournaledDeduplicator(ProblemGuard.from_problems([]))
    restored.load_journal(path)
    assert len(restored.exact) == 2
    assert len(restored.formatting) == 1
    assert not restored.accept(web)
    assert not restored.accept(math)


def test_validation_checkpoint_round_trip(tmp_path):
    original = base.ValidationCollector(("web", "math"), 4)
    original.add("web", np.asarray([1, 2, 3, 4, 5], dtype=np.int32))
    original.add("math", np.asarray([6, 7, 8], dtype=np.int32))
    checkpointed.save_validation(tmp_path, original)

    restored = checkpointed.load_validation(
        tmp_path,
        ("web", "math"),
        4,
        dict(original.counts),
    )
    np.testing.assert_array_equal(
        restored.tokens["web"][0], np.asarray([1, 2, 3, 4, 5])
    )
    np.testing.assert_array_equal(
        restored.tokens["math"][0], np.asarray([6, 7, 8])
    )
    assert restored.counts == original.counts


def test_raw_source_iterator_uses_the_target_tokenizer_boundary(
    tmp_path, monkeypatch
):
    text = "A sufficiently long raw document whose custom boundary is observable."
    source = {"name": "example", "kind": "parquet_text", "domain": "web"}
    monkeypatch.setattr(
        base,
        "parquet_documents",
        lambda *args, **kwargs: iter(
            [base.RawDocument("example", "web", (text,), (text,))]
        ),
    )
    monkeypatch.setattr(base, "quality_reason", lambda *args, **kwargs: None)

    class Encoder:
        def encode(self, texts, out_type=int):
            return [[11, 12] for _ in texts]

    documents = checkpointed.source_iterator(
        source,
        [tmp_path / "unused.parquet"],
        Encoder(),
        checkpointed.JournaledDeduplicator(ProblemGuard.from_problems([])),
        base.ValidationCollector(("web",), 8),
        0,
        Counter(),
        {},
        32_768,
        8_192,
        0.2,
        eot_id=7,
    )
    assert next(documents).tolist() == [7, 11, 12]


def test_completed_checkpoint_validation_detects_changed_budget(tmp_path):
    source = {"name": "example", "kind": "parquet_text"}
    signature = {"cache_format_version": 1}
    fingerprints = [{"path": "part.parquet", "size_bytes": 1, "sha256": "x"}]
    tokens = np.asarray([base.GPT2_EOT_ID, 1, 2], dtype=np.int32)
    with checkpointed.TokenCacheWriter(tmp_path / "tokens.bin") as writer:
        writer.append(tokens)
    (tmp_path / "dedup.bin").write_bytes(b"")
    (tmp_path / "state.json").write_text("{}")
    np.save(tmp_path / "validation_web.npy", np.asarray([1], dtype=np.int32))
    artifact_names = {
        "tokens.bin",
        "dedup.bin",
        "state.json",
        "validation_web.npy",
    }
    artifacts = {
        name: (
            checkpointed.token_cache_stats(tmp_path / name)
            if name == "tokens.bin"
            else checkpointed.artifact_fingerprint(tmp_path / name)
        )
        for name in artifact_names
    }
    metadata = {
        "source": "example",
        "source_index": 0,
        "source_spec_sha256": checkpointed.canonical_sha256(source),
        "budget": 3,
        "tokens": 3,
        "domains": ["web"],
        "signature": signature,
        "input_fingerprints": fingerprints,
        "artifacts": artifacts,
    }
    (tmp_path / "checkpoint.json").write_text(json.dumps(metadata))

    checkpointed.validate_completed_checkpoint(
        tmp_path, source, 0, 3, signature, fingerprints
    )
    with pytest.raises(ValueError, match="incompatible"):
        checkpointed.validate_completed_checkpoint(
            tmp_path, source, 0, 4, signature, fingerprints
        )


def test_completed_checkpoint_detects_artifact_corruption(tmp_path):
    source = {"name": "example", "kind": "parquet_text"}
    signature = {"cache_format_version": 1}
    fingerprints = []
    with checkpointed.TokenCacheWriter(tmp_path / "tokens.bin") as writer:
        writer.append(np.asarray([1, 2], dtype=np.int32))
    (tmp_path / "dedup.bin").write_bytes(b"")
    (tmp_path / "state.json").write_text("{}")
    np.save(tmp_path / "validation_web.npy", np.asarray([1], dtype=np.int32))
    artifact_names = {
        "tokens.bin",
        "dedup.bin",
        "state.json",
        "validation_web.npy",
    }
    metadata = {
        "source": "example",
        "source_index": 0,
        "source_spec_sha256": checkpointed.canonical_sha256(source),
        "budget": 2,
        "tokens": 2,
        "domains": ["web"],
        "signature": signature,
        "input_fingerprints": fingerprints,
        "artifacts": {
            name: (
                checkpointed.token_cache_stats(tmp_path / name)
                if name == "tokens.bin"
                else checkpointed.artifact_fingerprint(tmp_path / name)
            )
            for name in artifact_names
        },
    }
    (tmp_path / "checkpoint.json").write_text(json.dumps(metadata))
    with (tmp_path / "tokens.bin").open("ab") as handle:
        handle.write(b"\0")

    with pytest.raises(ValueError, match="truncated token-cache header"):
        checkpointed.validate_completed_checkpoint(
            tmp_path, source, 0, 2, signature, fingerprints
        )


def test_validation_budgets_preserve_source_mix():
    sources = [
        {"name": "raw", "domain": "web", "weight": 0.05},
        {"name": "edu", "domain": "web", "weight": 0.45},
        {"name": "code", "domain": "code", "weight": 0.20},
    ]

    budgets = checkpointed.validation_source_budgets(sources, 999)

    assert budgets == {"raw": 100, "edu": 900, "code": 1000}


def test_validation_budgets_ignore_a_deliberately_disabled_source():
    """A profile that zeroes a source must not be rejected for excluding it.

    `k3_weights_quality.json` carries `math_drills: 0.0` so the four ablation
    arms name the same source set. Demanding a positive validation budget for
    a source that contributes no training tokens would refuse that profile,
    and dividing a domain's budget among members that include it would hand
    tokens to a source with no cache to draw them from.
    """
    sources = [
        {"name": "raw", "domain": "web", "weight": 0.05},
        {"name": "edu", "domain": "web", "weight": 0.45},
        {"name": "drills", "domain": "math", "weight": 0.0},
        {"name": "openmath", "domain": "math", "weight": 0.20},
    ]

    budgets = checkpointed.validation_source_budgets(sources, 999)

    assert budgets == {"raw": 100, "edu": 900, "openmath": 1000}
    assert "drills" not in budgets
