"""Literal-byte and released-Cola BPE views of the same immutable text streams."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pretraining.nanogpt_mini.native_bits_data import sha256_file

TOKENIZER_REPO = "ByteDance-Seed/Cola-DLM"
TOKENIZER_REVISION = "c1eafdd9cfd8064aeb917d569ef70a075b353eed"
TOKENIZER_SHA256 = "3ca996cca8afea58b34e95c353e859333592642a5e51d695d7a6dbbaf692dfe9"
BPE_VOCAB_SIZE = 100278
BPE_CACHE_SCHEMA = "cola_bpe_literal_v1"
BOUNDARY_POLICY = (
    "existing concatenated literal stream; no BOS/EOT insertion or document isolation"
)


def source_manifest(root: Path) -> dict:
    """Verify literal sources, without consulting the native-bits alphabet."""
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("normalization") != "none":
        raise ValueError("Cola controls require an unnormalized literal stream")
    sources = {}
    for split, filename in (("train", "train.txt"), ("validation", "val.txt")):
        path = root / filename
        digest = sha256_file(path)
        size = path.stat().st_size
        if digest != metadata[split]["sha256"] or size != metadata[split]["bytes"]:
            raise ValueError(f"{split} byte stream differs from its manifest")
        if size == 0:
            raise ValueError(f"empty {split} byte stream")
        sources[split] = {"bytes": size, "sha256": digest}
    return sources


def _byte_decoder() -> dict[str, int]:
    # GPT-2/ByteLevel's reversible byte-to-Unicode alphabet, not UTF-8 decoding.
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    result = {chr(value): value for value in visible}
    for index, value in enumerate(
        value for value in range(256) if value not in visible
    ):
        result[chr(256 + index)] = value
    return result


class ColaCodec:
    """Decode IDs to bytes, including IDs ending inside a UTF-8 character."""

    def __init__(self, tokenizer_path: str | Path | None = None):
        self.tokenizer = None
        if tokenizer_path is None:
            self.token_bytes = tuple(bytes([value]) for value in range(256))
        else:
            from tokenizers import Tokenizer

            tokenizer_path = Path(tokenizer_path)
            if sha256_file(tokenizer_path) != TOKENIZER_SHA256:
                raise ValueError(
                    "Cola tokenizer SHA256 does not match the released tokenizer"
                )
            document = json.loads(tokenizer_path.read_text())
            if document["normalizer"] is not None or document["model"]["type"] != "BPE":
                raise ValueError(
                    "expected the released unnormalized byte-level BPE tokenizer"
                )
            decoder = _byte_decoder()
            added = {
                item["id"]: item["content"].encode("utf-8")
                for item in document["added_tokens"]
            }
            vocab = document["model"]["vocab"]
            if len(vocab) != BPE_VOCAB_SIZE or set(vocab.values()) != set(
                range(BPE_VOCAB_SIZE)
            ):
                raise ValueError("unexpected Cola vocabulary IDs")
            pieces = [b""] * BPE_VOCAB_SIZE
            for piece, index in vocab.items():
                pieces[index] = (
                    added[index]
                    if index in added
                    else bytes(decoder[char] for char in piece)
                )
            if any(not piece for piece in pieces):
                raise ValueError("empty token byte representation")
            self.token_bytes = tuple(pieces)
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self.tokenizer.no_padding()
            self.tokenizer.no_truncation()
        self.byte_lengths = np.asarray(
            [len(piece) for piece in self.token_bytes], dtype=np.int64
        )

    def encode_text(self, text: str) -> list[int]:
        if self.tokenizer is None:
            return list(text.encode("utf-8"))
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode_ids(self, ids: list[int]) -> bytes:
        if any(index < 0 or index >= len(self.token_bytes) for index in ids):
            raise ValueError("token ID outside the codec vocabulary")
        return b"".join(self.token_bytes[index] for index in ids)


def load_bpe_cache(cache: Path, sources: dict) -> tuple[dict, ColaCodec]:
    """Reject incomplete, corrupt, or differently sourced existing caches."""
    metadata = json.loads((cache / "metadata.json").read_text())
    if (
        metadata.get("schema") != BPE_CACHE_SCHEMA
        or metadata.get("sources") != sources
        or metadata.get("normalization") != "none"
        or metadata.get("boundary_policy") != BOUNDARY_POLICY
        or metadata.get("vocab_size") != BPE_VOCAB_SIZE
        or metadata.get("dtype") != "<i4"
        or metadata.get("tokenizer", {}).get("sha256") != TOKENIZER_SHA256
        or metadata.get("tokenizer", {}).get("revision") != TOKENIZER_REVISION
        or metadata.get("tokenizer", {}).get("repo") != TOKENIZER_REPO
    ):
        raise ValueError(
            "BPE cache provenance does not match the requested literal corpus/tokenizer"
        )
    expected_artifacts = {
        "tokenizer.json",
        "byte_lengths.npy",
        "train.bin",
        "validation.bin",
    }
    if set(metadata.get("artifacts", {})) != expected_artifacts:
        raise ValueError("BPE cache artifact manifest is incomplete")
    for filename, expected in metadata["artifacts"].items():
        path = cache / filename
        if (
            path.stat().st_size != expected["bytes"]
            or sha256_file(path) != expected["sha256"]
        ):
            raise ValueError(
                f"BPE cache artifact differs from its manifest: {filename}"
            )
    codec = ColaCodec(cache / "tokenizer.json")
    lengths = np.load(cache / "byte_lengths.npy", allow_pickle=False)
    if lengths.dtype != np.dtype(np.int64) or not np.array_equal(
        lengths, codec.byte_lengths
    ):
        raise ValueError("BPE cache token-byte lengths do not match the tokenizer")
    for split in sources:
        path = cache / f"{split}.bin"
        positions = metadata[split]["positions"]
        if positions <= 0 or path.stat().st_size != positions * 4:
            raise ValueError(f"invalid {split} BPE cache shape")
        ids = np.memmap(path, dtype="<i4", mode="r")
        byte_count = 0
        for start in range(0, positions, 1 << 20):
            chunk = ids[start : start + (1 << 20)]
            if chunk.min() < 0 or chunk.max() >= BPE_VOCAB_SIZE:
                raise ValueError(f"invalid {split} BPE token ID")
            byte_count += int(lengths[chunk].sum())
        if (
            byte_count != sources[split]["bytes"]
            or metadata[split]["source_bytes"] != byte_count
        ):
            raise ValueError(
                f"{split} BPE token lengths do not reconstruct the source length"
            )
    return metadata, codec


@torch.compile(fullgraph=True, dynamic=False)
def _cyclic_batch(source, offsets, cursor):
    return source[(offsets + cursor) % source.numel()].long()


class ColaCorpus:
    train: torch.Tensor
    validation: torch.Tensor

    def __init__(
        self,
        root: str | Path,
        device: torch.device,
        tokenization: str = "byte",
        bpe_cache_path: str | Path | None = None,
    ):
        if tokenization not in ("byte", "bpe"):
            raise ValueError("tokenization must be 'byte' or 'bpe'")
        self.root = Path(root).resolve()
        sources = source_manifest(self.root)
        self.tokenization = tokenization
        self.metadata: dict = {
            "root": str(self.root),
            "source_manifest_sha256": sha256_file(self.root / "metadata.json"),
            "boundary_policy": BOUNDARY_POLICY,
            "normalization": "none",
            "evaluation_scope": "FineWeb byte-cache validation_web proxy, not challenge heldout",
            "tokenization": tokenization,
            "resident_dtype": "uint8" if tokenization == "byte" else "int32",
        }
        cache = None
        if tokenization == "bpe":
            if bpe_cache_path is None:
                raise ValueError(
                    "BPE requires an explicitly prepared --bpe-cache-path directory"
                )
            cache = Path(bpe_cache_path).resolve()
            cache_metadata, self.codec = load_bpe_cache(cache, sources)
            if (
                cache_metadata.get("source_manifest_sha256")
                != self.metadata["source_manifest_sha256"]
            ):
                raise ValueError("BPE cache source manifest has changed")
            self.metadata["bpe_cache"] = {
                "root": str(cache),
                "manifest": cache_metadata,
            }
        else:
            self.codec = ColaCodec()
        self.byte_lengths = torch.from_numpy(self.codec.byte_lengths.copy()).to(device)
        for split, filename in (("train", "train.txt"), ("validation", "val.txt")):
            path = self.root / filename if cache is None else cache / f"{split}.bin"
            mapped = np.memmap(
                path, dtype=np.uint8 if cache is None else "<i4", mode="r"
            )
            # An owned array avoids writable-tensor aliases to a read-only mmap.
            tensor = torch.from_numpy(np.array(mapped, copy=True)).to(device)
            setattr(self, split, tensor)
            self.metadata[split] = {
                "path": str(self.root / filename),
                **sources[split],
                "positions": tensor.numel(),
            }
        self._offsets: dict[int, torch.Tensor] = {}
        self._cursor = torch.empty((), device=device, dtype=torch.long)

    def training_batch(self, cursor: int, count: int, seq_len: int) -> torch.Tensor:
        """Gather complete contexts; cursor and count are token/byte positions."""
        if cursor < 0 or seq_len < 1 or count < 1 or count % seq_len:
            raise ValueError("invalid consumed-position cursor or training shape")
        if count not in self._offsets:
            self._offsets[count] = torch.arange(count, device=self.train.device)
        self._cursor.fill_(cursor % self.train.numel())
        return _cyclic_batch(self.train, self._offsets[count], self._cursor).view(
            -1, seq_len
        )

    def validation_prefix(self, count: int, seq_len: int) -> torch.Tensor:
        """Return complete contexts; arbitrary final tails remain in .validation."""
        if seq_len < 1 or not 0 < count <= self.validation.numel() or count % seq_len:
            raise ValueError(
                "validation must be an in-range prefix of complete contexts"
            )
        return self.validation[:count].long().view(-1, seq_len)

    def source_bytes(self, ids: torch.Tensor) -> torch.Tensor:
        return self.byte_lengths[ids.long()].sum()

    def encode_text(self, text: str) -> list[int]:
        return self.codec.encode_text(text)

    def decode_ids(self, ids: list[int]) -> bytes:
        return self.codec.decode_ids(ids)
