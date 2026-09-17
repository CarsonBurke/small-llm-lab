"""CPU-only streaming contracts using tiny synthetic llmc token shards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pretraining.future_credit_stream import data
from pretraining.future_credit_stream.data import DocumentIndex, StreamingDocuments


def write_shard(path: Path, tokens: list[int]) -> None:
    header = np.zeros(256, dtype="<i4")
    header[:3] = [20240520, 1, len(tokens)]
    with path.open("wb") as output:
        output.write(header.tobytes())
        output.write(np.asarray(tokens, dtype="<u2").tobytes())


def make_index(tmp_path: Path, shards: list[list[int]], **kwargs) -> DocumentIndex:
    for number, tokens in enumerate(shards):
        write_shard(tmp_path / f"tokens_{number:03d}.bin", tokens)
    return DocumentIndex(str(tmp_path / "tokens_*.bin"), cache_dir=tmp_path / "cache", **kwargs)


def assert_pages_equal(left: dict, right: dict) -> None:
    for key in ("inputs", "targets", "resets"):
        np.testing.assert_array_equal(left[key], right[key])


def test_partial_documents_cross_shards_and_end_on_real_bos(tmp_path: Path, monkeypatch) -> None:
    # Force multiple scan chunks as well as a document crossing a shard boundary.
    monkeypatch.setattr(data, "_SCAN_TOKENS", 2)
    index = make_index(tmp_path, [[9, 1, 2], [], [3, 4, 1], [1, 5, 1, 6, 7]])
    details = index.describe()
    assert index.document_count == 3
    assert details["ignored_prefix_tokens"] == 1
    assert details["discarded_trailing_tokens"] == 3
    assert details["discarded_unclosed_documents"] == 1
    assert details["prediction_tokens_per_epoch"] == 7
    stream = StreamingDocuments(index, batch_size=2, seed=0, shuffle=False)
    pages = [stream.next_chunk(2), stream.next_chunk(1), stream.next_chunk(4)]
    actual = {key: np.concatenate([page[key] for page in pages])
              for key in ("inputs", "targets", "resets")}
    np.testing.assert_array_equal(actual["inputs"], [
        [1, 1], [2, 1], [3, 5], [4, 1], [1, 2], [1, 3], [5, 4],
    ])
    np.testing.assert_array_equal(actual["targets"], [
        [2, 1], [3, 5], [4, 1], [1, 2], [1, 3], [5, 4], [1, 1],
    ])
    np.testing.assert_array_equal(actual["resets"], [
        [True, True], [False, True], [False, False], [False, True],
        [True, False], [True, False], [False, False],
    ])
    assert actual["inputs"].dtype == actual["targets"].dtype == np.int64
    assert actual["resets"].dtype == np.bool_


@pytest.mark.parametrize("shuffle", [False, True])
def test_one_prediction_documents_refill_every_tick_when_batch_exceeds_corpus(
    tmp_path: Path, shuffle: bool,
) -> None:
    index = make_index(tmp_path, [[1], [1]])
    page = StreamingDocuments(index, batch_size=9, seed=3, shuffle=shuffle).next_chunk(4)
    np.testing.assert_array_equal(page["inputs"], np.ones((4, 9), dtype=np.int64))
    np.testing.assert_array_equal(page["targets"], np.ones((4, 9), dtype=np.int64))
    np.testing.assert_array_equal(page["resets"], np.ones((4, 9), dtype=np.bool_))


def test_shuffle_assigns_every_document_once_before_repeating(tmp_path: Path) -> None:
    tokens = [token for number in range(7) for token in (1, number + 2)] + [1, 40]
    index = make_index(tmp_path, [tokens])
    page = StreamingDocuments(index, batch_size=5, seed=42).next_chunk(10)
    assigned = page["targets"][page["resets"]]
    for start in range(0, 21, 7):
        np.testing.assert_array_equal(np.sort(assigned[start:start + 7]), np.arange(2, 9))
    different_seed = StreamingDocuments(index, batch_size=5, seed=43).next_chunk(10)
    assert not np.array_equal(page["targets"], different_seed["targets"])


@pytest.mark.parametrize("shuffle", [False, True])
def test_checkpoint_continuation_is_exact_and_independent_of_page_size(
    tmp_path: Path, shuffle: bool,
) -> None:
    index = make_index(tmp_path, [[1, 2, 3, 1, 1, 4], [5, 6, 1, 7, 1, 8]])
    uninterrupted = StreamingDocuments(index, batch_size=7, seed=91, shuffle=shuffle)
    uninterrupted.next_chunk(5)
    checkpoint = uninterrupted.state_dict()
    saved_documents = checkpoint["document_ids"].copy()
    saved_positions = checkpoint["positions"].copy()
    expected = uninterrupted.next_chunk(19)
    np.testing.assert_array_equal(checkpoint["document_ids"], saved_documents)
    np.testing.assert_array_equal(checkpoint["positions"], saved_positions)
    restored = StreamingDocuments(index, batch_size=7, seed=91, shuffle=shuffle)
    restored.load_state_dict(checkpoint)
    # The consumer owns the checkpoint; mutating it cannot mutate the restored stream.
    checkpoint["document_ids"][:] = -1
    checkpoint["positions"][:] = -1
    pages = [restored.next_chunk(steps) for steps in (1, 4, 2, 12)]
    actual = {key: np.concatenate([page[key] for page in pages])
              for key in ("inputs", "targets", "resets")}
    assert_pages_equal(actual, expected)


def test_initial_checkpoint_restores_first_page_resets(tmp_path: Path) -> None:
    index = make_index(tmp_path, [[1, 2, 1, 3, 4, 1, 5]])
    original = StreamingDocuments(index, batch_size=3, seed=18)
    restored = StreamingDocuments(index, batch_size=3, seed=18)
    restored.next_chunk(3)
    restored.load_state_dict(original.state_dict())
    actual = restored.next_chunk(5)
    assert_pages_equal(actual, original.next_chunk(5))
    np.testing.assert_array_equal(actual["resets"][0], [True, True, True])


@pytest.mark.parametrize("change", [{"batch_size": 3}, {"seed": 4}, {"shuffle": False}])
def test_checkpoint_rejects_different_stream_configuration(tmp_path: Path, change: dict) -> None:
    index = make_index(tmp_path, [[1, 2, 1, 3, 1]])
    options = {"batch_size": 2, "seed": 3, "shuffle": True}
    checkpoint = StreamingDocuments(index, **options).state_dict()
    with pytest.raises(ValueError, match="mismatch"):
        StreamingDocuments(index, **(options | change)).load_state_dict(checkpoint)


def test_changed_source_invalidates_cache_and_checkpoint(tmp_path: Path) -> None:
    index = make_index(tmp_path, [[1, 2, 1, 3]])
    stream = StreamingDocuments(index, batch_size=1, seed=0, shuffle=False)
    checkpoint = stream.state_dict()
    write_shard(tmp_path / "tokens_000.bin", [1, 5, 1, 3])
    with pytest.raises(ValueError, match="source shards changed"):
        stream.load_state_dict(checkpoint)
    replacement = DocumentIndex(str(tmp_path / "tokens_*.bin"), cache_dir=tmp_path / "cache")
    assert replacement.fingerprint != index.fingerprint
    new_stream = StreamingDocuments(replacement, batch_size=1, seed=0, shuffle=False)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        new_stream.load_state_dict(checkpoint)
    np.testing.assert_array_equal(new_stream.next_chunk(2)["targets"], [[5], [1]])


@pytest.mark.parametrize("damage", ["header", "positions", "truncated"])
def test_corrupt_boundary_cache_rebuilds_without_changing_stream(tmp_path: Path, damage: str) -> None:
    index = make_index(tmp_path, [[1, 2, 1, 3, 4, 1, 5]])
    expected = StreamingDocuments(index, batch_size=3, seed=4).next_chunk(9)
    cached = DocumentIndex(str(tmp_path / "tokens_*.bin"), cache_dir=tmp_path / "cache")
    assert cached.describe()["cache_hit"] is True
    with index.cache_path.open("r+b") as output:
        if damage == "truncated":
            output.truncate(50)
        else:
            output.seek(40 if damage == "header" else -1, 0 if damage == "header" else 2)
            previous = output.read(1)
            output.seek(-1, 1)
            output.write(bytes([previous[0] ^ 1]))
    rebuilt = DocumentIndex(str(tmp_path / "tokens_*.bin"), cache_dir=tmp_path / "cache")
    assert rebuilt.describe()["cache_hit"] is False
    assert_pages_equal(StreamingDocuments(rebuilt, batch_size=3, seed=4).next_chunk(9), expected)


def test_stale_cache_cannot_hide_out_of_vocabulary_tokens(tmp_path: Path) -> None:
    make_index(tmp_path, [[1, 2, 1, 3]], vocab_size=8)
    write_shard(tmp_path / "tokens_000.bin", [1, 2, 1, 8])
    with pytest.raises(ValueError, match="outside vocab_size=8"):
        DocumentIndex(str(tmp_path / "tokens_*.bin"), vocab_size=8, cache_dir=tmp_path / "cache")


@pytest.mark.parametrize("tokens", [[], [2, 3], [2, 1, 3]])
def test_no_closed_document_is_not_fabricated(tmp_path: Path, tokens: list[int]) -> None:
    index = make_index(tmp_path, [tokens])
    assert index.document_count == 0
    with pytest.raises(ValueError, match="No complete BOS-delimited documents"):
        StreamingDocuments(index, batch_size=2, seed=0)


@pytest.mark.parametrize("damage", ["magic", "version", "count", "truncated_header", "extra_token"])
def test_malformed_llmc_shards_are_rejected(tmp_path: Path, damage: str) -> None:
    path = tmp_path / "bad.bin"
    write_shard(path, [1, 2, 1])
    with path.open("r+b") as output:
        if damage == "truncated_header":
            output.truncate(24)
        elif damage == "extra_token":
            output.seek(0, 2)
            output.write(b"\0\0")
        else:
            output.seek({"magic": 0, "version": 4, "count": 8}[damage])
            output.write(np.asarray([99], dtype="<i4").tobytes())
    with pytest.raises(ValueError, match="llmc"):
        DocumentIndex(str(path), cache_dir=tmp_path / "cache")


def test_bos_and_vocabulary_configuration_cannot_reuse_incompatible_index(tmp_path: Path) -> None:
    index = make_index(tmp_path, [[1, 2, 1, 2, 3, 2]], vocab_size=8)
    alternate = DocumentIndex(
        str(tmp_path / "tokens_*.bin"), bos_id=2, vocab_size=8, cache_dir=tmp_path / "cache",
    )
    assert alternate.fingerprint != index.fingerprint
    page = StreamingDocuments(alternate, batch_size=1, seed=0, shuffle=False).next_chunk(5)
    np.testing.assert_array_equal(page["inputs"], [[2], [1], [2], [3], [2]])
    np.testing.assert_array_equal(page["targets"], [[1], [2], [3], [2], [1]])
    np.testing.assert_array_equal(page["resets"], [[True], [False], [True], [False], [True]])
    with pytest.raises(ValueError, match="outside vocab_size=3"):
        DocumentIndex(str(tmp_path / "tokens_*.bin"), vocab_size=3, cache_dir=tmp_path / "cache")


def test_rejected_checkpoint_leaves_stream_unchanged(tmp_path: Path) -> None:
    index = make_index(tmp_path, [[1, 2, 3, 1, 4, 1, 5]])
    stream = StreamingDocuments(index, batch_size=2, seed=11)
    stream.next_chunk(1)
    checkpoint = stream.state_dict()
    reference = StreamingDocuments(index, batch_size=2, seed=11)
    reference.load_state_dict(checkpoint)
    checkpoint["positions"][:] = index.total_tokens + 1
    with pytest.raises(ValueError, match="document positions"):
        stream.load_state_dict(checkpoint)
    assert_pages_equal(stream.next_chunk(8), reference.next_chunk(8))


