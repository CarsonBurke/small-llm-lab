"""Compact on-disk format for the byte n-gram counts.

Split-tree inference needs these counts at encode time, so they travel with the
tokenizer rather than being discarded after training. JSON costs several times
the size and is slow to parse at these entry counts, so the format is a flat
binary record stream, sorted by n-gram for a deterministic byte image and
therefore a stable sha256.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from pathlib import Path

MAGIC = b"PGNG"
FORMAT_VERSION = 1
_HEADER = struct.Struct("<4sIQ")
_RECORD = struct.Struct("<BI")
MAX_NGRAM_BYTES = 255
MAX_COUNT = 0xFFFFFFFF


def write_counts(path: Path, counts: Mapping[bytes, int]) -> str:
    """Write counts sorted by n-gram; returns the file's sha256."""
    items = sorted(counts.items())
    digest = hashlib.sha256()
    with Path(path).open("wb") as handle:
        header = _HEADER.pack(MAGIC, FORMAT_VERSION, len(items))
        handle.write(header)
        digest.update(header)
        for gram, count in items:
            if not 0 < len(gram) <= MAX_NGRAM_BYTES:
                raise ValueError(
                    f"n-gram of {len(gram)} bytes is outside 1..{MAX_NGRAM_BYTES}"
                )
            if not 0 < count <= MAX_COUNT:
                raise ValueError(f"count {count} is outside 1..{MAX_COUNT}")
            record = _RECORD.pack(len(gram), count) + gram
            handle.write(record)
            digest.update(record)
    return digest.hexdigest()


def read_counts(path: Path) -> dict[bytes, int]:
    payload = Path(path).read_bytes()
    magic, version, entries = _HEADER.unpack_from(payload, 0)
    if magic != MAGIC:
        raise ValueError(f"{path} is not an n-gram count file")
    if version != FORMAT_VERSION:
        raise ValueError(f"{path} has unsupported format version {version}")
    counts: dict[bytes, int] = {}
    offset = _HEADER.size
    for _ in range(entries):
        length, count = _RECORD.unpack_from(payload, offset)
        offset += _RECORD.size
        counts[payload[offset : offset + length]] = count
        offset += length
    if offset != len(payload):
        raise ValueError(
            f"{path} has {len(payload) - offset} trailing bytes after "
            f"{entries} entries"
        )
    return counts


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
