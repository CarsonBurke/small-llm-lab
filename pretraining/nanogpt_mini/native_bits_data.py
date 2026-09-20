"""Strict UTF-8 boundary, shuffled opaque IDs, and bounded prepared-data reads."""

from __future__ import annotations

import codecs
import hashlib
import json
import os
import random
import struct
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np

CHUNK_BYTES = 1024 * 1024
SCHEMA = "nanogpt_mini_native_bits_data_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def text_chunks(path: Path, digest=None) -> Iterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    with path.open("rb") as handle:
        while block := handle.read(CHUNK_BYTES):
            if digest is not None:
                digest.update(block)
            yield decoder.decode(block)
    final = decoder.decode(b"", final=True)
    if final:
        yield final


def scan_text(path: Path) -> tuple[set[str], dict]:
    alphabet: set[str] = set()
    digest = hashlib.sha256()
    count = 0
    byte_count = 0
    for text in text_chunks(path, digest):
        alphabet.update(text)
        count += len(text)
        byte_count += len(text.encode("utf-8"))
    if not count:
        raise ValueError(f"source text must not be empty: {path}")
    return alphabet, {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "bytes": byte_count,
        "characters": count,
    }


def validate_alphabet(alphabet: object) -> list[str]:
    if not isinstance(alphabet, list) or not alphabet:
        raise ValueError("alphabet must be a nonempty list of Unicode scalars")
    if any(
        not isinstance(char, str) or len(char) != 1 or 0xD800 <= ord(char) <= 0xDFFF
        for char in alphabet
    ):
        raise ValueError("alphabet entries must each be one Unicode scalar")
    if len(set(alphabet)) != len(alphabet):
        raise ValueError("alphabet contains duplicate characters")
    return alphabet


def encode_text(text: str, alphabet: list[str]) -> np.ndarray:
    lookup = {char: index for index, char in enumerate(alphabet)}
    try:
        return np.fromiter(
            (lookup[char] for char in text), dtype=np.uint32, count=len(text)
        )
    except KeyError as error:
        raise ValueError(
            f"character {error.args[0]!r} is outside this checkpoint's observed alphabet; no fallback or normalization"
        ) from None


def _write_ids(
    path: Path, source: Path, lookup: dict[str, int], expected: dict
) -> None:
    ids = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.uint32, shape=(expected["characters"],)
    )
    digest = hashlib.sha256()
    cursor = 0
    for text in text_chunks(source, digest):
        end = cursor + len(text)
        if end > ids.size:
            raise ValueError(f"source text changed during preparation: {source}")
        ids[cursor:end] = np.fromiter(
            (lookup[char] for char in text), dtype=np.uint32, count=len(text)
        )
        cursor = end
    ids.flush()
    del ids
    if cursor != expected["characters"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"source text changed during preparation: {source}")


def _write_literal_tokens(chunks: Iterator[np.ndarray], output: Path) -> dict:
    """Strip only EOT=256; validate UTF-8 independently on every boundary."""
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    separators = 0
    with output.open("wb") as handle:
        for tokens in chunks:
            if tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer):
                raise ValueError("byte cache must contain flat integer token vectors")
            if tokens.size and (int(tokens.min()) < 0 or int(tokens.max()) > 256):
                raise ValueError(
                    "byte cache contains an ID other than literal bytes or EOT=256"
                )
            start = 0
            for stop in np.flatnonzero(tokens == 256):
                block = tokens[start:stop].astype(np.uint8).tobytes()
                decoder.decode(block, final=True)
                handle.write(block)
                decoder = codecs.getincrementaldecoder("utf-8")("strict")
                separators += 1
                start = int(stop) + 1
            block = tokens[start:].astype(np.uint8).tobytes()
            decoder.decode(block)
            handle.write(block)
        decoder.decode(b"", final=True)
    return {"removed_eot_markers": separators, "raw_sha256": sha256_file(output)}


def _cache_train_chunks(path: Path) -> Iterator[np.ndarray]:
    # cache_format_version=2: <I document length, then that many <u2 IDs.
    with path.open("rb") as handle:
        while header := handle.read(4):
            if len(header) != 4:
                raise ValueError("truncated byte-cache document header")
            count = struct.unpack("<I", header)[0]
            if count == 0:
                raise ValueError("empty byte-cache document record")
            first = True
            while count:
                take = min(count, CHUNK_BYTES // 2)
                payload = handle.read(take * 2)
                if len(payload) != take * 2:
                    raise ValueError("truncated byte-cache document payload")
                tokens = np.frombuffer(payload, dtype="<u2")
                if first and tokens[0] != 256:
                    raise ValueError("byte-cache document must begin with EOT=256")
                first = False
                yield tokens
                count -= take


def extract_byte_cache(cache: Path, output: Path) -> dict:
    checkpoint_path = cache / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    signature = checkpoint["signature"]
    tokenizer = signature["tokenizer"]
    if (
        signature["cache_format_version"] != 2
        or tokenizer["kind"] != "utf8_bytes"
        or tokenizer["eot_id"] != 256
    ):
        raise ValueError(
            "expected version-2 UTF8-byte cache with literal IDs 0..255 and EOT=256"
        )
    provenance = {
        "path": str(cache.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "format": "length_prefixed_u32_then_u16_documents",
        "artifacts": {},
        "validation_note": "cache validation_web; not comparable to original sp1024 validation",
    }
    for filename in ("tokens.bin", "validation_web.npy"):
        path = cache / filename
        actual = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        expected = checkpoint["artifacts"][filename]
        if any(actual[key] != expected[key] for key in actual):
            raise ValueError(
                f"byte-cache artifact disagrees with checkpoint fingerprint: {filename}"
            )
        provenance["artifacts"][filename] = actual
    provenance["train"] = _write_literal_tokens(
        _cache_train_chunks(cache / "tokens.bin"), output / "train.txt"
    )
    validation = np.load(
        cache / "validation_web.npy", mmap_mode="r", allow_pickle=False
    )
    provenance["validation"] = _write_literal_tokens(
        (
            validation[start : start + CHUNK_BYTES // 2]
            for start in range(0, validation.size, CHUNK_BYTES // 2)
        ),
        output / "val.txt",
    )
    return provenance


def prepare_data(
    output: str | Path,
    *,
    train_text: str | Path | None = None,
    val_text: str | Path | None = None,
    byte_cache: str | Path | None = None,
    alphabet_seed: int = 1337,
) -> Path:
    """Prepare all observed characters; no normalization, filtering or text tokenization."""
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"prepared output already exists: {output}")
    if byte_cache is not None:
        if train_text is not None or val_text is not None:
            raise ValueError("choose --byte-cache OR both --train-text and --val-text")
    elif train_text is None or val_text is None:
        raise ValueError("provide both --train-text and --val-text")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        cache_meta = None
        if byte_cache is not None:
            cache_meta = extract_byte_cache(
                Path(byte_cache).expanduser().resolve(strict=True), staging
            )
            train, validation = staging / "train.txt", staging / "val.txt"
        else:
            train = Path(train_text).expanduser().resolve(strict=True)
            validation = Path(val_text).expanduser().resolve(strict=True)
        if (
            not train.is_file()
            or not validation.is_file()
            or os.path.samefile(train, validation)
        ):
            raise ValueError(
                "train and validation must be distinct regular files, not aliases"
            )
        train_chars, train_meta = scan_text(train)
        val_chars, val_meta = scan_text(validation)
        if train_meta["sha256"] == val_meta["sha256"]:
            raise ValueError("train and validation contain identical text")
        alphabet = sorted(train_chars | val_chars)
        random.Random(alphabet_seed).shuffle(alphabet)
        (staging / "alphabet.json").write_text(
            json.dumps(alphabet, ensure_ascii=True) + "\n"
        )
        lookup = {char: index for index, char in enumerate(alphabet)}
        _write_ids(staging / "train.npy", train, lookup, train_meta)
        _write_ids(staging / "val.npy", validation, lookup, val_meta)
        if cache_meta is not None:
            train_meta["path"], val_meta["path"] = (
                str(output / "train.txt"),
                str(output / "val.txt"),
            )
        metadata = {
            "schema": SCHEMA,
            "alphabet_seed": alphabet_seed,
            "alphabet_size": len(alphabet),
            "alphabet_scope": "shuffled sorted union of observed train and heldout characters; not all Unicode",
            "normalization": "none",
            "train": train_meta,
            "validation": val_meta,
            "byte_cache": cache_meta,
            "artifacts": {
                name: sha256_file(staging / name)
                for name in ("alphabet.json", "train.npy", "val.npy")
            },
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(output)
    return output


class PreparedData:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve(strict=True)
        self.metadata = json.loads((self.root / "metadata.json").read_text())
        if self.metadata.get("schema") != SCHEMA:
            raise ValueError("unsupported native-bit dataset schema")
        self.alphabet = validate_alphabet(
            json.loads((self.root / "alphabet.json").read_text())
        )
        self.byte_lengths = np.asarray(
            [len(char.encode("utf-8")) for char in self.alphabet], dtype=np.uint8
        )
        for filename in ("alphabet.json", "train.npy", "val.npy"):
            if (
                sha256_file(self.root / filename)
                != self.metadata["artifacts"][filename]
            ):
                raise ValueError(f"prepared artifact fingerprint mismatch: {filename}")
        self.train = self._load("train.npy", "train")
        self.validation = self._load("val.npy", "validation")
        if os.path.samefile(self.root / "train.npy", self.root / "val.npy"):
            raise ValueError("prepared training and validation arrays alias")

    def _load(self, filename: str, split: str) -> np.ndarray:
        values = np.load(self.root / filename, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.uint32 or values.ndim != 1 or not values.size:
            raise ValueError("prepared IDs must be nonempty flat uint32 arrays")
        if values.size != self.metadata[split]["characters"]:
            raise ValueError(f"prepared {split} character count mismatch")
        for start in range(0, values.size, CHUNK_BYTES // 4):
            if int(values[start : start + CHUNK_BYTES // 4].max()) >= len(
                self.alphabet
            ):
                raise ValueError(f"prepared {split} contains an unknown opaque ID")
        return values


def microbatches(ids: np.ndarray, seq_len: int, mbs: int) -> Iterator[np.ndarray]:
    if ids.ndim != 1 or seq_len <= 0 or mbs <= 0:
        raise ValueError("expected flat IDs and positive sequence/microbatch sizes")
    full = ids.size // seq_len * seq_len
    for offset in range(0, full, seq_len * mbs):
        yield ids[offset : min(offset + seq_len * mbs, full)].reshape(-1, seq_len)
    if full < ids.size:
        yield ids[full:].reshape(1, -1)


def cyclic_ids(values: np.ndarray, offset: int, count: int) -> np.ndarray:
    if values.ndim != 1 or not values.size or offset < 0 or count <= 0:
        raise ValueError("invalid cyclic ID request")
    offset %= values.size
    if offset + count <= values.size:
        return np.asarray(values[offset : offset + count])
    result = np.empty(count, dtype=values.dtype)
    position = 0
    while position < count:
        take = min(count - position, values.size - offset)
        result[position : position + take] = values[offset : offset + take]
        position += take
        offset = 0
    return result
