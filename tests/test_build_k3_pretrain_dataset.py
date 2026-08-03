from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.build_k3_pretrain_dataset import (
    DocumentDeduplicator,
    LoaderAlignedShardWriter,
    RawDocument,
    logical_interleave,
    paragraph_chunks,
    split_token_document,
    stable_digest,
)


def read_payload(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype="<i4", count=256)
    assert header[0] == 20240520
    assert header[1] == 1
    return np.fromfile(path, dtype="<u2", offset=256 * 4, count=int(header[2]))


def test_source_manifest_weights_match_domains() -> None:
    manifest = json.loads(Path("pretraining/k3_sources.json").read_text())
    assert sum(source["weight"] for source in manifest["sources"]) == 1.0
    for domain, expected in manifest["domains"].items():
        actual = sum(
            source["weight"]
            for source in manifest["sources"]
            if source["domain"] == domain
        )
        assert abs(actual - expected) < 1e-12
    assert len(manifest["sources"]) >= 10
    for source in manifest["sources"]:
        if "full_files" in source:
            assert set(source["files"]).issubset(source["full_files"])
            assert len(source["full_files"]) > len(source["files"])


def test_weight_profiles_cover_every_source_and_match_domains() -> None:
    manifest = json.loads(Path("pretraining/k3_sources.json").read_text())
    source_domains = {
        source["name"]: source["domain"] for source in manifest["sources"]
    }
    for path in Path("pretraining").glob("k3_weights_*.json"):
        profile = json.loads(path.read_text())
        assert set(profile["sources"]) == set(source_domains)
        assert abs(sum(profile["sources"].values()) - 1) < 1e-12
        for domain, expected in profile["domains"].items():
            actual = sum(
                weight
                for source, weight in profile["sources"].items()
                if source_domains[source] == domain
            )
            assert abs(actual - expected) < 1e-12


def test_loader_aligned_shards_preserve_one_logical_stream(tmp_path: Path) -> None:
    batch_tokens = 8
    total_steps = 5
    logical = np.arange(total_steps * batch_tokens + 1, dtype=np.uint16)
    writer = LoaderAlignedShardWriter(
        tmp_path,
        batch_tokens=batch_tokens,
        total_steps=total_steps,
        steps_per_shard=2,
    )
    writer.append(logical[:7], "a")
    writer.append(logical[7:31], "b")
    writer.append(logical[31:], "c")
    writer.finish()

    shards = [read_payload(path) for path in sorted(tmp_path.glob("*.bin"))]
    assert [len(shard) for shard in shards] == [17, 17, 9]
    assert all(
        sum(shard["source_tokens"].values()) == shard["tokens"]
        for shard in writer.shards
    )
    assert sum((len(shard) - 1) // batch_tokens for shard in shards) == total_steps
    assert all(left[-1] == right[0] for left, right in zip(shards, shards[1:]))
    reconstructed = np.concatenate(
        [shards[0], *(shard[1:] for shard in shards[1:])]
    )
    np.testing.assert_array_equal(reconstructed, logical)


def test_deduplicator_rejects_exact_formatting_and_heldout() -> None:
    heldout_text = "Solve 2 + 2."
    heldout_key = stable_digest(
        heldout_text.casefold(), person=b"pgolf-holdout"
    )
    dedup = DocumentDeduplicator({heldout_key})
    heldout = RawDocument("math", "math", (heldout_text,), (heldout_text,))
    assert not dedup.accept(heldout)

    first = RawDocument("web", "web", ("Hello, WORLD!",), ("Hello, WORLD!",))
    formatting_duplicate = RawDocument(
        "knowledge", "knowledge", ("hello world",), ("hello world",)
    )
    assert dedup.accept(first)
    assert not dedup.accept(formatting_duplicate)
    assert dedup.counts["heldout_overlap"] == 1
    assert dedup.counts["formatting_duplicate"] == 1


def test_code_and_math_operators_are_not_formatting_deduplicated() -> None:
    dedup = DocumentDeduplicator()
    plus = RawDocument("code", "code", ("return x + y" * 20,), ("return x + y",))
    minus = RawDocument("code", "code", ("return x - y" * 20,), ("return x - y",))
    assert dedup.accept(plus)
    assert dedup.accept(minus)


def test_token_split_preserves_content_and_inserts_boundaries() -> None:
    tokens = np.arange(11, dtype=np.int32)
    tokens[0] = 50256
    chunks = list(split_token_document(tokens, max_document_tokens=5))
    assert [chunk.size for chunk in chunks] == [5, 5, 3]
    assert all(chunk[0] == 50256 for chunk in chunks)
    reconstructed = np.concatenate(
        [chunks[0], *(chunk[1:] for chunk in chunks[1:])]
    )
    np.testing.assert_array_equal(reconstructed, tokens)


def test_paragraph_chunks_preserve_content_and_limit() -> None:
    text = "a" * 8 + "\n\n" + "b" * 8 + "\n\n" + "c" * 20
    chunks = list(paragraph_chunks(text, max_chars=10))
    assert chunks == ["a" * 8, "b" * 8, "c" * 10, "c" * 10]
    assert all(len(chunk) <= 10 for chunk in chunks)


def test_logical_interleave_hits_exact_source_budgets() -> None:
    sources = {
        "a": iter([np.arange(7), np.arange(7, 14)]),
        "b": iter([np.arange(100, 106)]),
    }
    budgets = {"a": 10, "b": 4}
    emitted = list(logical_interleave(sources, budgets))
    totals = {
        name: sum(tokens.size for source, tokens in emitted if source == name)
        for name in sources
    }
    assert totals == budgets
    assert sum(tokens.size for _, tokens in emitted) == sum(budgets.values())
