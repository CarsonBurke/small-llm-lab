"""Bounded, document-isolated pages from BOS-delimited llmc token shards."""

from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


_HEADER_BYTES = 256 * 4
_CACHE_HEADER_BYTES = 4096
_SCAN_TOKENS = 1 << 20
_CACHE_VERSION = 1
_STATE_VERSION = 1


def _source_info(paths: list[Path]) -> list[dict[str, Any]]:
    sources = []
    for path in paths:
        with path.open("rb") as source:
            stat = os.fstat(source.fileno())
            header_bytes = source.read(_HEADER_BYTES)
        if len(header_bytes) != _HEADER_BYTES:
            raise ValueError(f"Truncated llmc header: {path}")
        header = np.frombuffer(header_bytes, dtype="<i4")
        if header[0] != 20240520 or header[1] != 1:
            raise ValueError(f"Unsupported llmc magic/version: {path}")
        count = int(header[2])
        if count < 0 or stat.st_size != _HEADER_BYTES + count * 2:
            raise ValueError(f"llmc token count/filesize mismatch: {path}")
        sources.append({
            "path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns, "device": stat.st_dev, "inode": stat.st_ino,
            "header_sha256": hashlib.sha256(header_bytes).hexdigest(), "tokens": count,
        })
    return sources


class DocumentIndex:
    """Memory-map shards and a checksummed, atomically published BOS index.

    Shards form one logical token sequence. A document ends immediately before
    the next *real* BOS, which is nevertheless its final prediction target.
    The final BOS-led suffix is unclosed and is not sampled.
    """

    def __init__(
        self, pattern: str, bos_id: int = 1, vocab_size: int = 1024,
        cache_dir: Path | str | None = None,
    ) -> None:
        if not 1 <= vocab_size <= 65536 or not 0 <= bos_id < vocab_size:
            raise ValueError("Require 0 <= bos_id < vocab_size <= 65536")
        self.bos_id = int(bos_id)
        self.vocab_size = int(vocab_size)
        self.paths = sorted({Path(path).resolve() for path in glob.glob(pattern)})
        if not self.paths:
            raise FileNotFoundError(f"No llmc shards match {pattern!r}")
        self._sources = _source_info(self.paths)
        signature = {"version": _CACHE_VERSION, "bos_id": self.bos_id,
                     "vocab_size": self.vocab_size, "sources": self._sources}
        self.fingerprint = hashlib.sha256(
            json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        counts = np.asarray([source["tokens"] for source in self._sources], dtype=np.int64)
        self._shard_ends = np.cumsum(counts)
        self._shard_starts = self._shard_ends - counts
        self.total_tokens = int(self._shard_ends[-1])
        self._shards = [
            np.memmap(path, dtype="<u2", mode="r", offset=_HEADER_BYTES, shape=(int(count),))
            if count else np.empty(0, dtype="<u2")
            for path, count in zip(self.paths, counts, strict=True)
        ]
        directory = Path(cache_dir) if cache_dir is not None else self.paths[0].parent / ".nextlat_stream_index"
        directory.mkdir(parents=True, exist_ok=True)
        self.cache_path = directory / f"bos-{self.fingerprint}.idx"
        self.cache_hit = self._load_cache()
        if not self.cache_hit:
            self._build_cache()
            if not self._load_cache():
                raise RuntimeError(f"Could not validate newly built BOS index: {self.cache_path}")
        if _source_info(self.paths) != self._sources:
            raise RuntimeError("Source shards changed while opening the BOS index")
        self.document_count = max(0, len(self._boundaries) - 1)

    def _load_cache(self) -> bool:
        try:
            with self.cache_path.open("rb") as source:
                header = source.read(_CACHE_HEADER_BYTES)
                if (len(header) != _CACHE_HEADER_BYTES
                        or hashlib.sha256(header[32:]).digest() != header[:32]):
                    return False
                metadata = json.loads(header[32:].rstrip(b"\0"))
                if metadata["version"] != _CACHE_VERSION or metadata["fingerprint"] != self.fingerprint:
                    return False
                count = metadata["boundary_count"]
                if type(count) is not int or not 0 <= count <= self.total_tokens:
                    return False
                if os.fstat(source.fileno()).st_size != _CACHE_HEADER_BYTES + count * 8:
                    return False
                digest = hashlib.sha256()
                previous = -1
                for offset in range(0, count, _SCAN_TOKENS):
                    raw = source.read(min(_SCAN_TOKENS, count - offset) * 8)
                    positions = np.frombuffer(raw, dtype="<i8")
                    digest.update(raw)
                    if (positions[0] <= previous or positions[-1] >= self.total_tokens
                            or np.any(positions[1:] <= positions[:-1])):
                        return False
                    previous = int(positions[-1])
                if digest.hexdigest() != metadata["boundary_sha256"]:
                    return False
                minimum, maximum = metadata["token_min"], metadata["token_max"]
                if self.total_tokens:
                    if not (type(minimum) is int and type(maximum) is int
                            and 0 <= minimum <= maximum < self.vocab_size):
                        return False
                elif minimum is not None or maximum is not None:
                    return False
                self._boundaries = (
                    np.memmap(source, dtype="<i8", mode="r", offset=_CACHE_HEADER_BYTES, shape=(count,))
                    if count else np.empty(0, dtype=np.int64)
                )
                self._metadata = metadata
            return True
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            return False

    def _build_cache(self) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b", prefix=f".{self.cache_path.name}.", dir=self.cache_path.parent, delete=False,
            ) as output:
                temporary = Path(output.name)
                output.write(b"\0" * _CACHE_HEADER_BYTES)
                digest = hashlib.sha256()
                boundary_count = 0
                token_min: int | None = None
                token_max: int | None = None
                for shard, start in zip(self._shards, self._shard_starts, strict=True):
                    for offset in range(0, len(shard), _SCAN_TOKENS):
                        tokens = shard[offset:offset + _SCAN_TOKENS]
                        minimum, maximum = int(tokens.min()), int(tokens.max())
                        if minimum < 0 or maximum >= self.vocab_size:
                            raise ValueError(
                                f"Token range [{minimum}, {maximum}] outside vocab_size={self.vocab_size} "
                                f"at global offset {int(start) + offset}"
                            )
                        token_min = minimum if token_min is None else min(token_min, minimum)
                        token_max = maximum if token_max is None else max(token_max, maximum)
                        positions = np.flatnonzero(tokens == self.bos_id).astype("<i8", copy=False)
                        positions += int(start) + offset
                        raw = positions.tobytes()
                        output.write(raw)
                        digest.update(raw)
                        boundary_count += len(positions)
                if _source_info(self.paths) != self._sources:
                    raise RuntimeError("Source shards changed while building the BOS index")
                metadata = {"version": _CACHE_VERSION, "fingerprint": self.fingerprint,
                            "boundary_count": boundary_count, "boundary_sha256": digest.hexdigest(),
                            "token_min": token_min, "token_max": token_max}
                header = json.dumps(metadata, sort_keys=True).encode()
                if len(header) > _CACHE_HEADER_BYTES - 32:
                    raise RuntimeError("BOS cache header exceeds reserved space")
                output.seek(0)
                header = header.ljust(_CACHE_HEADER_BYTES - 32, b"\0")
                output.write(hashlib.sha256(header).digest() + header)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.cache_path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def describe(self) -> dict[str, Any]:
        has_bos = bool(len(self._boundaries))
        prefix = int(self._boundaries[0]) if has_bos else self.total_tokens
        trailing = self.total_tokens - int(self._boundaries[-1]) if has_bos else 0
        return {
            "fingerprint": self.fingerprint, "document_count": self.document_count,
            "shard_count": len(self.paths), "total_tokens": self.total_tokens,
            "prediction_tokens_per_epoch": self.total_tokens - prefix - trailing,
            "ignored_prefix_tokens": prefix, "discarded_trailing_tokens": trailing,
            "discarded_unclosed_documents": int(has_bos),
            "bos_id": self.bos_id, "vocab_size": self.vocab_size,
            "token_min": self._metadata["token_min"], "token_max": self._metadata["token_max"],
            "cache_path": str(self.cache_path), "cache_hit": self.cache_hit,
            "sources": [dict(source) for source in self._sources],
        }

    def _read_positions(self, positions: np.ndarray) -> np.ndarray:
        """Gather a bounded page, grouping random accesses by mapped shard."""
        flat = positions.reshape(-1)
        shard_ids = np.searchsorted(self._shard_ends, flat, side="right")
        result = np.empty(flat.shape, dtype=np.int64)
        for shard_id in np.unique(shard_ids):
            selected = np.flatnonzero(shard_ids == shard_id)
            result[selected] = self._shards[shard_id][flat[selected] - self._shard_starts[shard_id]]
        return result.reshape(positions.shape)


class StreamingDocuments:
    """Refill document lanes in deterministic epoch order, independent of page size.

    A checkpoint stores O(batch_size) lane state plus epoch/cursor, never the
    O(document_count) permutation. PCG64 seeded by (seed, epoch) regenerates the
    current permutation on restore; ordinary pages reuse that permutation.
    """

    def __init__(
        self, index: DocumentIndex, batch_size: int, seed: int, shuffle: bool = True,
    ) -> None:
        if batch_size <= 0 or seed < 0:
            raise ValueError("batch_size must be positive and seed nonnegative")
        if index.document_count == 0:
            raise ValueError("No complete BOS-delimited documents in source shards")
        self.index = index
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self._epoch = 0
        self._cursor = 0
        self._order = self._make_order(0)
        self._document_ids = np.full(self.batch_size, -1, dtype=np.int64)
        self._positions = np.zeros(self.batch_size, dtype=np.int64)
        self._ends = np.zeros(self.batch_size, dtype=np.int64)

    def _make_order(self, epoch: int) -> np.ndarray | None:
        if not self.shuffle:
            return None
        order = np.arange(self.index.document_count, dtype=np.int64)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed, epoch])))
        rng.shuffle(order)
        return order

    def _take_documents(self, count: int) -> np.ndarray:
        if not self.shuffle or self.index.document_count == 1:
            result = (np.arange(count, dtype=np.int64) + self._cursor) % self.index.document_count
            consumed = self._cursor + count
            self._epoch += (consumed - 1) // self.index.document_count
            self._cursor = (consumed - 1) % self.index.document_count + 1
            return result
        result = np.empty(count, dtype=np.int64)
        written = 0
        while written < count:
            if self._cursor == self.index.document_count:
                self._epoch += 1
                self._cursor = 0
                # Drop the old epoch before allocating the next permutation.
                self._order = None
                self._order = self._make_order(self._epoch)
            take = min(count - written, self.index.document_count - self._cursor)
            assert self._order is not None
            result[written:written + take] = self._order[self._cursor:self._cursor + take]
            self._cursor += take
            written += take
        return result

    def next_chunk(self, steps: int) -> dict[str, np.ndarray]:
        if not isinstance(steps, (int, np.integer)) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        positions = np.empty((steps, self.batch_size), dtype=np.int64)
        resets = np.empty((steps, self.batch_size), dtype=np.bool_)
        # One vectorized operation per time tick, not Python loops over tokens/lanes.
        for tick in range(steps):
            refill = self._positions >= self._ends
            resets[tick] = refill
            if np.any(refill):
                documents = self._take_documents(int(np.count_nonzero(refill)))
                self._document_ids[refill] = documents
                self._positions[refill] = self.index._boundaries[documents]
                self._ends[refill] = self.index._boundaries[documents + 1]
            positions[tick] = self._positions
            self._positions += 1
        return {"inputs": self.index._read_positions(positions),
                "targets": self.index._read_positions(positions + 1), "resets": resets}

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": _STATE_VERSION, "fingerprint": self.index.fingerprint,
            "batch_size": self.batch_size, "seed": self.seed, "shuffle": self.shuffle,
            "epoch": self._epoch, "cursor": self._cursor,
            "document_ids": self._document_ids.copy(), "positions": self._positions.copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {"version": _STATE_VERSION, "fingerprint": self.index.fingerprint,
                    "batch_size": self.batch_size, "seed": self.seed, "shuffle": self.shuffle}
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"Stream checkpoint {key} mismatch")
        # Also reject mutations of source files after this index was opened.
        if _source_info(self.index.paths) != self.index._sources:
            raise ValueError("Stream checkpoint source shards changed")
        epoch, cursor = state.get("epoch"), state.get("cursor")
        if (type(epoch) is not int or epoch < 0 or type(cursor) is not int
                or not 0 <= cursor <= self.index.document_count):
            raise ValueError("Invalid stream checkpoint epoch/cursor")
        arrays = []
        for name in ("document_ids", "positions"):
            array = np.asarray(state.get(name))
            if array.shape != (self.batch_size,) or array.dtype.kind not in "iu":
                raise ValueError(f"Invalid stream checkpoint {name}")
            arrays.append(array.astype(np.int64, copy=True))
        documents, positions = arrays
        initial = np.all(documents == -1)
        if initial:
            if epoch != 0 or cursor != 0 or np.any(positions != 0):
                raise ValueError("Invalid initial stream checkpoint")
            ends = np.zeros(self.batch_size, dtype=np.int64)
        else:
            if np.any(documents < 0) or np.any(documents >= self.index.document_count):
                raise ValueError("Invalid checkpoint document IDs")
            starts = self.index._boundaries[documents]
            ends = np.asarray(self.index._boundaries[documents + 1]).copy()
            if cursor == 0 or np.any(positions <= starts) or np.any(positions > ends):
                raise ValueError("Invalid checkpoint document positions")
        order = self._make_order(epoch)
        self._epoch, self._cursor, self._order = epoch, cursor, order
        self._document_ids, self._positions, self._ends = documents, positions, ends
