#!/usr/bin/env python3
"""Prepare the released Cola BPE view of immutable literal FineWeb sources.

Queue this CPU-tokenization entrypoint through mlq; training never builds caches.
Existing outputs are verified and reused only when all provenance and hashes match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from pretraining.cola.data import (
    BOUNDARY_POLICY,
    BPE_CACHE_SCHEMA,
    BPE_VOCAB_SIZE,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    TOKENIZER_SHA256,
    ColaCodec,
    load_bpe_cache,
    source_manifest,
)
from pretraining.nanogpt_mini.native_bits_data import sha256_file

# The pinned tokenizer's regex cannot cross a newline followed by non-whitespace:
# newline-consuming whitespace/punctuation alternatives finish before that next
# character. Its added tokens contain no newlines and do not strip whitespace.
# Thus these are exact full-stream BPE boundaries, not arbitrary tokenization cuts.
_SAFE_BOUNDARY = re.compile(r"[\r\n](?=\S)")
_CHUNK_CHARACTERS = 1 << 20


def _text_chunks(path: Path) -> Iterator[str]:
    pending = ""
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        while chunk := handle.read(_CHUNK_CHARACTERS):
            pending += chunk
            boundary = 0
            for match in _SAFE_BOUNDARY.finditer(pending):
                boundary = match.end()
            if boundary:
                yield pending[:boundary]
                pending = pending[boundary:]
        if pending:
            yield pending


def _write_split(source: Path, output: Path, codec: ColaCodec, expected: dict) -> dict:
    digest = hashlib.sha256()
    byte_count = 0
    positions = 0
    with output.open("xb") as handle:
        for text in _text_chunks(source):
            raw = text.encode("utf-8")
            ids = codec.encode_text(text)
            if codec.decode_ids(ids) != raw:
                raise ValueError(f"BPE roundtrip changed literal bytes in {source}")
            values = np.asarray(ids, dtype="<i4")
            represented_bytes = int(codec.byte_lengths[values].sum())
            if represented_bytes != len(raw):
                raise ValueError(
                    f"BPE token lengths changed literal byte count in {source}"
                )
            values.tofile(handle)
            digest.update(raw)
            byte_count += represented_bytes
            positions += values.size
    if digest.hexdigest() != expected["sha256"] or byte_count != expected["bytes"]:
        raise ValueError(f"source changed during BPE preparation: {source}")
    return {
        "positions": positions,
        "source_bytes": byte_count,
        "bytes_per_position": byte_count / positions,
        "roundtrip_sha256": digest.hexdigest(),
    }


def prepare_bpe(root: Path, output: Path, tokenizer_path: Path | None = None) -> dict:
    root = root.resolve()
    output = output.resolve()
    sources = source_manifest(root)
    source_manifest_sha256 = sha256_file(root / "metadata.json")
    if output.exists():
        metadata, _ = load_bpe_cache(output, sources)
        if metadata.get("source_manifest_sha256") != source_manifest_sha256:
            raise ValueError(
                "existing BPE cache was prepared from a different source manifest"
            )
        return metadata
    if tokenizer_path is None:
        from huggingface_hub import hf_hub_download

        tokenizer_path = Path(
            hf_hub_download(
                repo_id=TOKENIZER_REPO,
                filename="tokenizer.json",
                revision=TOKENIZER_REVISION,
            )
        )
    if sha256_file(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("tokenizer SHA256 differs from the pinned Cola release")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.", dir=output.parent
    ) as temporary:
        cache = Path(temporary) / "cache"
        cache.mkdir()
        shutil.copyfile(tokenizer_path, cache / "tokenizer.json")
        codec = ColaCodec(cache / "tokenizer.json")
        np.save(cache / "byte_lengths.npy", codec.byte_lengths, allow_pickle=False)
        metadata = {
            "schema": BPE_CACHE_SCHEMA,
            "sources": sources,
            "source_manifest_sha256": source_manifest_sha256,
            "normalization": "none",
            "boundary_policy": BOUNDARY_POLICY,
            "dtype": "<i4",
            "vocab_size": BPE_VOCAB_SIZE,
            "tokenizer": {
                "repo": TOKENIZER_REPO,
                "revision": TOKENIZER_REVISION,
                "filename": "tokenizer.json",
                "sha256": TOKENIZER_SHA256,
            },
        }
        for split, filename in (("train", "train.txt"), ("validation", "val.txt")):
            metadata[split] = _write_split(
                root / filename, cache / f"{split}.bin", codec, sources[split]
            )
        # This small heldout stream is also encoded as one sequence, independently
        # confirming the streaming boundaries preserve full-file tokenization.
        validation_text = (
            (root / "val.txt").read_bytes().decode("utf-8", errors="strict")
        )
        validation_ids = np.asarray(codec.encode_text(validation_text), dtype="<i4")
        if not np.array_equal(
            validation_ids, np.fromfile(cache / "validation.bin", dtype="<i4")
        ):
            raise ValueError(
                "streamed validation differs from full-stream released-tokenizer encoding"
            )
        if sha256_file(root / "metadata.json") != source_manifest_sha256:
            raise ValueError("source manifest changed during BPE preparation")
        metadata["artifacts"] = {
            filename: {
                "bytes": (cache / filename).stat().st_size,
                "sha256": sha256_file(cache / filename),
            }
            for filename in (
                "tokenizer.json",
                "byte_lengths.npy",
                "train.bin",
                "validation.bin",
            )
        }
        (cache / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        if output.exists():
            raise FileExistsError(
                f"BPE cache appeared during preparation; refusing to replace {output}"
            )
        cache.rename(output)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=REPOSITORY / "data/datasets/native_bits_fineweb"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new BPE cache directory; matching existing caches are verified",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="optional local released tokenizer.json (SHA256 checked)",
    )
    args = parser.parse_args()
    metadata = prepare_bpe(args.root, args.output, args.tokenizer_path)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
