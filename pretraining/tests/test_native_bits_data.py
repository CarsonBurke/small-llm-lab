"""CPU-only contracts for the external Unicode/opaque-ID boundary."""

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from pretraining.nanogpt_mini import native_bits_data as data


def test_streaming_unicode_preserves_scalars_bytes_and_validation_tail(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(data, "CHUNK_BYTES", 7)
    train = "a\x00\n\r\n\ufeffé e\u0301 🙂𐀀aa"
    validation = "🙂\n\x00e\u0301éz"
    (tmp_path / "train.txt").write_bytes(train.encode("utf-8"))
    (tmp_path / "val.txt").write_bytes(validation.encode("utf-8"))
    output = data.prepare_data(
        tmp_path / "prepared",
        train_text=tmp_path / "train.txt",
        val_text=tmp_path / "val.txt",
        alphabet_seed=17,
    )
    prepared = data.PreparedData(output)
    assert "".join(prepared.alphabet[index] for index in prepared.train) == train
    assert (
        "".join(prepared.alphabet[index] for index in prepared.validation) == validation
    )
    assert set(prepared.alphabet) == set(train + validation)
    assert prepared.metadata["train"]["bytes"] == len(train.encode("utf-8"))
    assert prepared.metadata["validation"]["characters"] == len(validation)
    batches = list(data.microbatches(prepared.validation, seq_len=3, mbs=2))
    np.testing.assert_array_equal(
        np.concatenate([batch.ravel() for batch in batches]), prepared.validation
    )
    assert [batch.shape for batch in batches] == [(2, 3), (1, 1)]
    assert int(prepared.byte_lengths[prepared.validation].sum()) == len(
        validation.encode("utf-8")
    )


def test_cyclic_source_keeps_tail_before_wrapping():
    actual = data.cyclic_ids(np.asarray([8, 4, 2], dtype=np.uint32), 2, 8)
    np.testing.assert_array_equal(actual, [2, 8, 4, 2, 8, 4, 2, 8])
    batches = list(data.microbatches(actual, seq_len=3, mbs=1))
    np.testing.assert_array_equal(
        np.concatenate([batch.ravel() for batch in batches]), actual
    )


def test_alias_and_invalid_utf8_never_publish_prepared_data(tmp_path: Path):
    source = tmp_path / "train.txt"
    source.write_bytes(b"train")
    alias = tmp_path / "alias.txt"
    os.link(source, alias)
    with pytest.raises(ValueError):
        data.prepare_data(tmp_path / "alias-output", train_text=source, val_text=alias)
    malformed = tmp_path / "bad.txt"
    malformed.write_bytes(b"valid prefix\xf0\x9f")
    with pytest.raises(UnicodeDecodeError):
        data.prepare_data(
            tmp_path / "bad-output", train_text=source, val_text=malformed
        )
    assert not (tmp_path / "bad-output").exists()
    assert not (tmp_path / "alias-output").exists()


def test_unknown_character_does_not_normalize_or_fallback():
    with pytest.raises(ValueError):
        data.encode_text("é", ["e", "\u0301"])
    np.testing.assert_array_equal(data.encode_text("e\u0301", ["e", "\u0301"]), [0, 1])


def _make_cache(root: Path, documents: list[list[int]], validation: list[int]):
    root.mkdir()
    with (root / "tokens.bin").open("wb") as handle:
        for document in documents:
            handle.write(struct.pack("<I", len(document)))
            handle.write(np.asarray(document, dtype="<u2").tobytes())
    np.save(root / "validation_web.npy", np.asarray(validation, dtype=np.int32))
    checkpoint = {
        "signature": {
            "cache_format_version": 2,
            "tokenizer": {"kind": "utf8_bytes", "eot_id": 256},
        },
        "artifacts": {
            name: {
                "sha256": data.sha256_file(root / name),
                "size_bytes": (root / name).stat().st_size,
            }
            for name in ("tokens.bin", "validation_web.npy")
        },
    }
    (root / "checkpoint.json").write_text(json.dumps(checkpoint))


def test_byte_cache_decodes_documents_not_storage_headers(tmp_path: Path):
    cache = tmp_path / "cache"
    _make_cache(
        cache,
        [[256, *"é\x00".encode()], [256, *"🙂\n".encode()]],
        [256, *"𐀀z".encode()],
    )
    output = data.prepare_data(tmp_path / "prepared", byte_cache=cache)
    prepared = data.PreparedData(output)
    assert (output / "train.txt").read_bytes() == "é\x00🙂\n".encode()
    assert (output / "val.txt").read_bytes() == "𐀀z".encode()
    assert "".join(prepared.alphabet[index] for index in prepared.train) == "é\x00🙂\n"


def test_byte_cache_cannot_join_partial_utf8_across_documents(tmp_path: Path):
    cache = tmp_path / "cache"
    _make_cache(cache, [[256, 0xC3], [256, 0xA9]], [256, 97])
    with pytest.raises(UnicodeDecodeError):
        data.prepare_data(tmp_path / "prepared", byte_cache=cache)
    assert not (tmp_path / "prepared").exists()


def test_signed_validation_ids_cannot_wrap_into_literal_bytes(tmp_path: Path):
    cache = tmp_path / "cache"
    _make_cache(cache, [[256, 97]], [256, -1])
    with pytest.raises(ValueError):
        data.prepare_data(tmp_path / "prepared", byte_cache=cache)
    assert not (tmp_path / "prepared").exists()


def test_prepared_array_tampering_is_rejected(tmp_path: Path):
    train, val = tmp_path / "train.txt", tmp_path / "val.txt"
    train.write_bytes(b"abc")
    val.write_bytes(b"cab")
    output = data.prepare_data(tmp_path / "prepared", train_text=train, val_text=val)
    ids = np.load(output / "val.npy", mmap_mode="r+")
    ids[0] = (int(ids[0]) + 1) % 3
    ids.flush()
    del ids
    with pytest.raises(ValueError):
        data.PreparedData(output)
