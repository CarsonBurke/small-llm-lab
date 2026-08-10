#!/usr/bin/env python3
"""Build deterministic, document-aligned byte-diffusion dataset shards."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence
import zipfile

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pretraining.bolmo_data import SourceTokenByteifier, validate_source_tokenizer
from pretraining.byte_diffusion.data import (
    AtomicDocument,
    AtomicIdManifest,
    AtomicSpecial,
    ChunkDocumentSpan,
    PackedChunk,
)
from scripts.build_bolmo_dataset import (
    canonical_sha256,
    discover_source_manifest,
    expand_patterns,
    fingerprint_file,
    infer_train_overlap,
    read_challenge_shard,
    sha256_file,
    validate_manifest_tokenizer,
)
from tokenization.tokenizer import SplitTreeNumericTokenizer


DATASET_SCHEMA = "byte_diffusion_dataset/v5"
ARTIFACT_SCHEMA = "byte_diffusion_mapped_chunks/v5"
ARTIFACT_ALIGNMENT = 4_096


@dataclass
class DocumentReadStats:
    physical_tokens: int = 0
    unique_tokens: int = 0
    complete_documents: int = 0
    empty_documents: int = 0
    incomplete_tail_tokens: int | None = None
    truncated_by_max_documents: bool = False


class ChallengeDocumentReader:
    """Read one deduplicated token stream and split it at source EOT ids."""

    def __init__(
        self,
        paths: Sequence[Path],
        *,
        eot_id: int,
        overlap_tokens: int = 0,
        require_terminal_eot: bool = False,
    ) -> None:
        if not paths:
            raise ValueError("document reader requires at least one source shard")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative")
        self.paths = tuple(Path(path) for path in paths)
        self.eot_id = int(eot_id)
        self.overlap_tokens = int(overlap_tokens)
        self.require_terminal_eot = bool(require_terminal_eot)
        self.stats = DocumentReadStats()
        self._validate_overlap_chain()

    def _validate_overlap_chain(self) -> None:
        previous_tail: np.ndarray | None = None
        for shard_index, path in enumerate(self.paths):
            tokens = read_challenge_shard(path)
            if self.overlap_tokens and len(tokens) < self.overlap_tokens:
                raise ValueError(f"shard is too short for overlap check: {path}")
            if shard_index and self.overlap_tokens:
                observed = np.asarray(
                    tokens[: self.overlap_tokens], dtype=np.int64
                )
                if previous_tail is None or not np.array_equal(
                    observed, previous_tail
                ):
                    raise ValueError(
                        f"declared {self.overlap_tokens}-token overlap does not "
                        f"match at {path}"
                    )
            if self.overlap_tokens:
                previous_tail = np.asarray(
                    tokens[-self.overlap_tokens :], dtype=np.int64
                ).copy()

    def iter_documents(
        self, *, max_documents: int | None = None
    ) -> Iterator[tuple[int, ...]]:
        if max_documents is not None and max_documents <= 0:
            raise ValueError("max_documents must be positive")
        self.stats = DocumentReadStats()
        pending: list[int] = []
        for shard_index, path in enumerate(self.paths):
            tokens = read_challenge_shard(path)
            self.stats.physical_tokens += len(tokens)
            start = self.overlap_tokens if shard_index else 0
            for value in tokens[start:]:
                source_id = int(value)
                self.stats.unique_tokens += 1
                pending.append(source_id)
                if source_id != self.eot_id:
                    continue
                self.stats.complete_documents += 1
                self.stats.empty_documents += int(len(pending) == 1)
                yield tuple(pending)
                pending.clear()
                if (
                    max_documents is not None
                    and self.stats.complete_documents == max_documents
                ):
                    self.stats.truncated_by_max_documents = True
                    self.stats.incomplete_tail_tokens = None
                    return
        self.stats.incomplete_tail_tokens = len(pending)
        if pending and self.require_terminal_eot:
            raise ValueError(
                "source stream ends inside a document with "
                f"{len(pending)} unterminated source tokens"
            )

    def iter_document_batches(
        self,
        *,
        max_documents: int | None = None,
        target_tokens: int = 2_000_000,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield contiguous complete documents without iterating source tokens."""

        if max_documents is not None and max_documents <= 0:
            raise ValueError("max_documents must be positive")
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        self.stats = DocumentReadStats()
        pending = np.empty(0, dtype="<u2")
        emitted = 0
        for shard_index, path in enumerate(self.paths):
            tokens = read_challenge_shard(path)
            self.stats.physical_tokens += len(tokens)
            start = self.overlap_tokens if shard_index else 0
            source = tokens[start:]
            cursor = 0
            if pending.size:
                terminal = np.flatnonzero(source == self.eot_id)
                if not terminal.size:
                    pending = np.concatenate((pending, np.asarray(source)))
                    self.stats.unique_tokens += len(source)
                    continue
                stop = int(terminal[0]) + 1
                document = np.concatenate((pending, np.asarray(source[:stop])))
                pending = np.empty(0, dtype="<u2")
                self.stats.unique_tokens += stop
                self.stats.complete_documents += 1
                self.stats.empty_documents += int(document.size == 1)
                yield document, np.asarray([document.size], dtype=np.int64)
                emitted += 1
                cursor = stop
                if max_documents is not None and emitted == max_documents:
                    self.stats.truncated_by_max_documents = True
                    self.stats.incomplete_tail_tokens = None
                    return
            terminal = np.flatnonzero(source[cursor:] == self.eot_id) + cursor
            terminal_cursor = 0
            while terminal_cursor < terminal.size:
                remaining_documents = (
                    terminal.size - terminal_cursor
                    if max_documents is None
                    else min(terminal.size - terminal_cursor, max_documents - emitted)
                )
                if remaining_documents <= 0:
                    self.stats.truncated_by_max_documents = True
                    self.stats.incomplete_tail_tokens = None
                    return
                target_stop = cursor + target_tokens
                stop_cursor = int(
                    np.searchsorted(terminal, target_stop, side="right")
                )
                stop_cursor = max(stop_cursor, terminal_cursor + 1)
                stop_cursor = min(stop_cursor, terminal_cursor + remaining_documents)
                selected_terminal = terminal[terminal_cursor:stop_cursor]
                stop = int(selected_terminal[-1]) + 1
                block = np.asarray(source[cursor:stop])
                starts = np.r_[cursor - cursor, selected_terminal[:-1] + 1 - cursor]
                lengths = selected_terminal + 1 - (starts + cursor)
                lengths = lengths.astype(np.int64, copy=False)
                self.stats.unique_tokens += len(block)
                self.stats.complete_documents += len(lengths)
                self.stats.empty_documents += int((lengths == 1).sum())
                yield block, lengths
                emitted += len(lengths)
                cursor = stop
                terminal_cursor = stop_cursor
                if max_documents is not None and emitted == max_documents:
                    self.stats.truncated_by_max_documents = True
                    self.stats.incomplete_tail_tokens = None
                    return
            tail = np.asarray(source[cursor:])
            self.stats.unique_tokens += len(tail)
            pending = tail.copy()
        self.stats.incomplete_tail_tokens = int(pending.size)
        if pending.size and self.require_terminal_eot:
            raise ValueError(
                "source stream ends inside a document with "
                f"{pending.size} unterminated source tokens"
            )


class StreamingDocumentPacker:
    """Bounded streaming equivalent of ``data.pack_documents``."""

    def __init__(
        self,
        manifest: AtomicIdManifest,
        *,
        chunk_size: int,
        close_rows_at_document: bool = False,
        document_aligned_pages: bool = False,
    ) -> None:
        if chunk_size <= 0 or chunk_size % 4:
            raise ValueError("chunk_size must be a positive multiple of four")
        self.manifest = manifest
        self.chunk_size = int(chunk_size)
        self.close_rows_at_document = bool(close_rows_at_document)
        self.document_aligned_pages = bool(document_aligned_pages)
        if self.close_rows_at_document and self.document_aligned_pages:
            raise ValueError("packing layouts are mutually exclusive")
        self.chunk_index = 0
        self.stream_start = 0
        self.document_keys: dict[int, str] = {}
        self.alignment_padding = 0
        self._input: list[int] = []
        self._target: list[int] = []
        self._valid: list[bool] = []
        self._score: list[bool] = []
        self._documents: list[int] = []
        self._document_offsets: list[int] = []
        self._patch_offsets: list[int] = []

    def add_document(
        self, document_index: int, document: AtomicDocument
    ) -> tuple[PackedChunk, ...]:
        if document_index in self.document_keys:
            raise ValueError(f"duplicate document index {document_index}")
        document.validate(self.manifest)
        self.document_keys[document_index] = document.key
        ready: list[PackedChunk] = []
        if (
            not self.close_rows_at_document
            and not self.document_aligned_pages
            and self._input
        ):
            if self._input[-1] != self.manifest.eot_id or self._score[-1]:
                raise AssertionError("dense stream did not end at an unscored EOT")
            # Match ordinary next-token stream training: terminal EOT predicts
            # the first atom of the following document. This recovers every
            # document-start target without a synthetic per-document forward.
            self._target[-1] = document.atomic_ids[0]
            self._score[-1] = True
            if len(self._input) == self.chunk_size:
                ready.append(self._flush(full=True))
        if self.document_aligned_pages and self._input:
            # Start every document on its own patch.  At most three physical
            # PAD slots are introduced; they are neither valid nor scored and
            # are skipped by the packed kernels.
            while len(self._input) % 4:
                self._append(
                    input_id=self.manifest.pad_id,
                    target_id=self.manifest.pad_id,
                    valid=False,
                    score=False,
                    document_index=-1,
                    document_offset=-1,
                    patch_offset=-1,
                )
                self.alignment_padding += 1
                if len(self._input) == self.chunk_size:
                    ready.append(self._flush(full=True))
        for offset, atomic_id in enumerate(document.atomic_ids):
            has_target = offset + 1 < len(document.atomic_ids)
            self._append(
                input_id=atomic_id,
                target_id=(
                    document.atomic_ids[offset + 1]
                    if has_target
                    else self.manifest.pad_id
                ),
                valid=True,
                score=has_target,
                document_index=document_index,
                document_offset=offset,
                # Fixed-stride patches follow the packed stream, exactly like
                # the challenge AR loader. EOT remains an atomic boundary;
                # it does not waste the rest of an 8,192-position row.
                patch_offset=(
                    offset % 4
                    if self.document_aligned_pages
                    else (self.stream_start + len(self._input))
                    % 4
                ),
            )
            if len(self._input) == self.chunk_size and (
                self.close_rows_at_document
                or self.document_aligned_pages
                or has_target
            ):
                ready.append(self._flush(full=True))
        # The legacy diagnostic layout closed every short document into a new
        # 8,192-position row. Production rows instead follow the challenge's
        # dense EOT-delimited stream. The opt-in legacy arm remains useful for
        # exact historical test fixtures, but is never the builder default.
        if self.close_rows_at_document and self._input:
            ready.append(self._flush(full=False))
        return tuple(ready)

    def finish(self) -> tuple[PackedChunk, ...]:
        if not self._input:
            return ()
        return (self._flush(full=False),)

    def _append(
        self,
        *,
        input_id: int,
        target_id: int,
        valid: bool,
        score: bool,
        document_index: int,
        document_offset: int,
        patch_offset: int,
    ) -> None:
        self._input.append(input_id)
        self._target.append(target_id)
        self._valid.append(valid)
        self._score.append(score)
        self._documents.append(document_index)
        self._document_offsets.append(document_offset)
        self._patch_offsets.append(patch_offset)

    def _flush(self, *, full: bool) -> PackedChunk:
        real_width = len(self._input)
        if full and real_width != self.chunk_size:
            raise AssertionError("full chunk flushed at the wrong width")
        if not full and not 0 < real_width < self.chunk_size:
            raise AssertionError("partial chunk flushed at the wrong width")
        storage_padding = self.chunk_size - real_width
        input_ids = self._input + [self.manifest.pad_id] * storage_padding
        target_ids = self._target + [self.manifest.pad_id] * storage_padding
        valid = self._valid + [False] * storage_padding
        score = self._score + [False] * storage_padding
        documents = self._documents + [-1] * storage_padding
        document_offsets = self._document_offsets + [-1] * storage_padding
        patch_offsets = self._patch_offsets + [-1] * storage_padding
        spans = self._document_spans(documents, document_offsets, real_width)
        halo_valid = bool(full and self._score[-1])
        halo_id = self._target[-1] if halo_valid else self.manifest.pad_id
        chunk = PackedChunk(
            chunk_index=self.chunk_index,
            stream_start=self.stream_start,
            stream_stop=self.stream_start + real_width,
            input_ids=tuple(input_ids),
            target_ids=tuple(target_ids),
            valid_mask=tuple(valid),
            score_mask=tuple(score),
            document_indices=tuple(documents),
            document_offsets=tuple(document_offsets),
            patch_offsets=tuple(patch_offsets),
            label_halo_id=halo_id,
            label_halo_valid=halo_valid,
            document_spans=spans,
        )
        self.chunk_index += 1
        self.stream_start += real_width
        self._input.clear()
        self._target.clear()
        self._valid.clear()
        self._score.clear()
        self._documents.clear()
        self._document_offsets.clear()
        self._patch_offsets.clear()
        return chunk

    def _document_spans(
        self,
        documents: Sequence[int],
        document_offsets: Sequence[int],
        width: int,
    ) -> tuple[ChunkDocumentSpan, ...]:
        spans: list[ChunkDocumentSpan] = []
        cursor = 0
        while cursor < width:
            document_index = documents[cursor]
            if document_index < 0:
                cursor += 1
                continue
            start = cursor
            document_start = document_offsets[cursor]
            cursor += 1
            while cursor < width and documents[cursor] == document_index:
                cursor += 1
            spans.append(
                ChunkDocumentSpan(
                    document_index=document_index,
                    document_key=self.document_keys[document_index],
                    chunk_start=start,
                    chunk_stop=cursor,
                    document_start=document_start,
                    document_stop=document_offsets[cursor - 1] + 1,
                )
            )
        return tuple(spans)


def validate_atomic_utf8(atomic_ids: Sequence[int], *, document_key: str) -> None:
    """Reject byte-native records containing an invalid UTF-8 literal segment."""

    pending = bytearray()
    for atomic_id in atomic_ids:
        if 0 <= atomic_id < 256:
            pending.append(atomic_id)
            continue
        if pending:
            try:
                bytes(pending).decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"byte-native document {document_key!r} contains invalid UTF-8"
                ) from error
            pending.clear()
    if pending:
        try:
            bytes(pending).decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"byte-native document {document_key!r} contains invalid UTF-8"
            ) from error


def validate_atomic_utf8_batch(atomic_ids: np.ndarray, *, batch_key: str) -> None:
    """Vectorized strict UTF-8 validation with atomic specials as boundaries."""

    values = np.asarray(atomic_ids)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("byte-native batch must be a one-dimensional integer array")
    literal = values < 256
    byte = values.astype(np.uint16, copy=False)
    continuation = literal & (byte >= 0x80) & (byte <= 0xBF)
    expected = np.zeros(values.size, dtype=np.bool_)
    lead1 = literal & (byte >= 0xC2) & (byte <= 0xDF)
    lead2 = literal & (byte >= 0xE0) & (byte <= 0xEF)
    lead3 = literal & (byte >= 0xF0) & (byte <= 0xF4)
    expected[1:] |= lead1[:-1] | lead2[:-1] | lead3[:-1]
    expected[2:] |= lead2[:-2] | lead3[:-2]
    expected[3:] |= lead3[:-3]
    invalid_lead = literal & (byte >= 0xC0) & ~(lead1 | lead2 | lead3)
    truncated = bool(lead1[-1:].any() or lead2[-2:].any() or lead3[-3:].any())
    valid = (
        not bool(invalid_lead.any())
        and not truncated
        and np.array_equal(continuation, expected)
    )
    if valid and values.size > 1:
        next_byte = byte[1:]
        valid = not bool(
            (
                (lead2[:-1] & (byte[:-1] == 0xE0) & (next_byte < 0xA0))
                | (lead2[:-1] & (byte[:-1] == 0xED) & (next_byte > 0x9F))
                | (lead3[:-1] & (byte[:-1] == 0xF0) & (next_byte < 0x90))
                | (lead3[:-1] & (byte[:-1] == 0xF4) & (next_byte > 0x8F))
            ).any()
        )
    if not valid:
        raise ValueError(f"byte-native batch {batch_key!r} contains invalid UTF-8")


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write an NPZ whose byte representation is stable across invocations."""

    temporary = path.with_suffix(path.suffix + ".working")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _npy_bytes(np.asarray(arrays[name])))
    os.replace(temporary, path)


def write_deterministic_mapped_artifact(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    alignment: int = ARTIFACT_ALIGNMENT,
) -> dict[str, dict[str, Any]]:
    """Write one deterministic, row-addressable raw-array container.

    The hash-bound manifest is the container index: every array has an explicit
    dtype, shape, byte offset, and byte length. Arrays begin on independent
    filesystem pages, so selecting a few rows through ``numpy.memmap`` does not
    inflate or decompress the rest of the artifact.
    """

    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("artifact alignment must be a positive power of two")
    normalized = {
        name: np.ascontiguousarray(array)
        for name, array in sorted(arrays.items())
    }
    if not normalized:
        raise ValueError("cannot serialize an empty mapped artifact")
    descriptors: dict[str, dict[str, Any]] = {}
    offset = 0
    for name, array in normalized.items():
        if array.dtype.hasobject:
            raise ValueError(f"mapped artifact field {name!r} cannot contain objects")
        offset = (offset + alignment - 1) & -alignment
        descriptors[name] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "byte_offset": offset,
            "byte_length": int(array.nbytes),
        }
        offset += int(array.nbytes)

    temporary = path.with_suffix(path.suffix + ".working")
    zero_page = bytes(alignment)
    with temporary.open("wb") as handle:
        cursor = 0
        for name, array in normalized.items():
            descriptor = descriptors[name]
            target = int(descriptor["byte_offset"])
            padding = target - cursor
            while padding:
                block = min(padding, alignment)
                handle.write(zero_page[:block])
                padding -= block
            handle.write(memoryview(array).cast("B"))
            cursor = target + int(descriptor["byte_length"])
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return descriptors


def mapped_row_sha256(arrays: Mapping[str, np.ndarray]) -> list[str]:
    """Hash each logical row across every artifact field."""

    normalized = {
        name: np.ascontiguousarray(array)
        for name, array in sorted(arrays.items())
    }
    row_counts = {array.shape[0] for array in normalized.values() if array.ndim}
    if len(row_counts) != 1 or any(array.ndim == 0 for array in normalized.values()):
        raise ValueError("mapped artifact fields must share one leading row axis")
    rows = row_counts.pop()
    result: list[str] = []
    for row in range(rows):
        digest = hashlib.sha256(b"byte_diffusion_mapped_row/v1\0")
        for name, array in normalized.items():
            digest.update(name.encode("ascii"))
            digest.update(b"\0")
            digest.update(memoryview(array[row : row + 1]).cast("B"))
        result.append(digest.hexdigest())
    return result


def _chunk_arrays(chunks: Sequence[PackedChunk]) -> dict[str, np.ndarray]:
    if not chunks:
        raise ValueError("cannot serialize an empty chunk shard")
    return {
        "chunk_index": np.asarray([chunk.chunk_index for chunk in chunks], dtype="<i8"),
        "stream_start": np.asarray([chunk.stream_start for chunk in chunks], dtype="<i8"),
        "stream_stop": np.asarray([chunk.stream_stop for chunk in chunks], dtype="<i8"),
        "input_ids": np.asarray([chunk.input_ids for chunk in chunks], dtype="<u2"),
        "target_ids": np.asarray([chunk.target_ids for chunk in chunks], dtype="<u2"),
        "valid_mask": np.asarray([chunk.valid_mask for chunk in chunks], dtype=np.bool_),
        "score_mask": np.asarray([chunk.score_mask for chunk in chunks], dtype=np.bool_),
        "document_indices": np.asarray(
            [chunk.document_indices for chunk in chunks], dtype="<i8"
        ),
        "document_offsets": np.asarray(
            [chunk.document_offsets for chunk in chunks], dtype="<i4"
        ),
        "patch_offsets": np.asarray(
            [chunk.patch_offsets for chunk in chunks], dtype=np.int8
        ),
        "label_halo_id": np.asarray(
            [chunk.label_halo_id for chunk in chunks], dtype="<u2"
        ),
        "label_halo_valid": np.asarray(
            [chunk.label_halo_valid for chunk in chunks], dtype=np.bool_
        ),
    }


def _write_artifact(
    output_dir: Path,
    split: str,
    artifact_index: int,
    chunks: Sequence[PackedChunk],
) -> dict[str, Any]:
    return _write_array_artifact(
        output_dir,
        split,
        artifact_index,
        _chunk_arrays(chunks),
    )


def _write_array_artifact(
    output_dir: Path,
    split: str,
    artifact_index: int,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    filename = f"{split}-{artifact_index:05d}.bdm"
    path = output_dir / filename
    arrays = {name: np.ascontiguousarray(array) for name, array in arrays.items()}
    row_count = int(arrays["input_ids"].shape[0])
    if row_count <= 0 or any(array.shape[0] != row_count for array in arrays.values()):
        raise ValueError("artifact arrays must share one nonempty row axis")
    descriptors = write_deterministic_mapped_artifact(path, arrays)
    valid_ids = arrays["input_ids"][arrays["valid_mask"]]
    literal_atomic_tokens = int((valid_ids < 256).sum())
    special_atomic_tokens = int((valid_ids >= 256).sum())
    physical_storage_positions = int(arrays["input_ids"].size)
    valid_atomic_tokens = int(arrays["valid_mask"].sum())
    valid_counts = arrays["valid_mask"].sum(axis=1, dtype=np.int64)
    physical_extents = np.where(
        arrays["valid_mask"],
        np.arange(arrays["valid_mask"].shape[1], dtype=np.int64)[None] + 1,
        0,
    ).max(axis=1)
    document_starts = (
        arrays["valid_mask"] & (arrays["document_offsets"] == 0)
    ).sum(axis=1, dtype=np.int64)
    return {
        "schema": ARTIFACT_SCHEMA,
        "path": filename,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "format": "aligned_raw_arrays/v1",
        "alignment": ARTIFACT_ALIGNMENT,
        "arrays": descriptors,
        "chunks": row_count,
        "chunk_size": int(arrays["input_ids"].shape[1]),
        "first_chunk_index": int(arrays["chunk_index"][0]),
        "last_chunk_index": int(arrays["chunk_index"][-1]),
        "valid_atomic_tokens": valid_atomic_tokens,
        "literal_atomic_tokens": literal_atomic_tokens,
        "special_atomic_tokens": special_atomic_tokens,
        "eot_atomic_tokens": int((valid_ids == 256).sum()),
        "physical_storage_positions": physical_storage_positions,
        "storage_padding_tokens": physical_storage_positions - valid_atomic_tokens,
        "canvas512_eligible_positions": int(
            np.minimum(arrays["valid_mask"].sum(axis=1), 512).sum()
        ),
        "scored_ar_targets": int(arrays["score_mask"].sum()),
        "halos": int(arrays["label_halo_valid"].sum()),
        # Compact, hash-bound scheduling index. The loader can construct its
        # row map and crop widths without inflating every 8192-wide payload at
        # process startup; full tensors are still verified when a shard is
        # first consumed.
        "row_valid_counts": valid_counts.tolist(),
        "row_physical_extents": physical_extents.tolist(),
        "row_document_starts": document_starts.tolist(),
        "row_sha256": mapped_row_sha256(arrays),
    }


class VectorizedDocumentPagePacker:
    """NumPy implementation of document-aligned byte-native page packing."""

    _POSITION_FIELDS = (
        "input_ids",
        "target_ids",
        "valid_mask",
        "score_mask",
        "document_indices",
        "document_offsets",
        "patch_offsets",
    )

    def __init__(self, manifest: AtomicIdManifest, *, chunk_size: int) -> None:
        if chunk_size <= 0 or chunk_size % 4:
            raise ValueError("chunk_size must be a positive multiple of four")
        self.manifest = manifest
        self.chunk_size = int(chunk_size)
        self.chunk_index = 0
        self.stream_start = 0
        self.alignment_padding = 0
        self._remainder = self._empty_positions(0)

    def _empty_positions(self, width: int) -> dict[str, np.ndarray]:
        return {
            "input_ids": np.full(width, self.manifest.pad_id, dtype="<u2"),
            "target_ids": np.full(width, self.manifest.pad_id, dtype="<u2"),
            "valid_mask": np.zeros(width, dtype=np.bool_),
            "score_mask": np.zeros(width, dtype=np.bool_),
            "document_indices": np.full(width, -1, dtype="<i8"),
            "document_offsets": np.full(width, -1, dtype="<i4"),
            "patch_offsets": np.full(width, -1, dtype=np.int8),
        }

    def add_batch(
        self,
        source_ids: np.ndarray,
        document_lengths: np.ndarray,
        *,
        first_document_index: int,
    ) -> dict[str, np.ndarray] | None:
        source = np.asarray(source_ids, dtype="<u2")
        lengths = np.asarray(document_lengths, dtype=np.int64)
        if (
            source.ndim != 1
            or lengths.ndim != 1
            or not lengths.size
            or bool((lengths <= 0).any())
            or int(lengths.sum()) != source.size
        ):
            raise ValueError("document batch has inconsistent lengths")
        document_stops = np.cumsum(lengths)
        if bool((source[document_stops - 1] != self.manifest.eot_id).any()):
            raise ValueError("document batch contains an unterminated document")
        if bool((source >= self.manifest.mask_id).any()):
            raise ValueError("document batch contains a non-clean atomic id")
        leading_padding = -(
            self.stream_start + self._remainder["input_ids"].size
        ) % 4
        padded_lengths = (lengths + 3) // 4 * 4
        starts = np.empty(lengths.size, dtype=np.int64)
        starts[0] = leading_padding
        if lengths.size > 1:
            starts[1:] = leading_padding + np.cumsum(padded_lengths[:-1])
        physical_width = int(starts[-1] + lengths[-1])
        positions = self._empty_positions(physical_width)
        source_starts = document_stops - lengths
        repeated_source_starts = np.repeat(source_starts, lengths)
        offsets = np.arange(source.size, dtype=np.int64) - repeated_source_starts
        packed_positions = offsets + np.repeat(starts, lengths)
        document_ids = np.repeat(
            np.arange(
                first_document_index,
                first_document_index + lengths.size,
                dtype=np.int64,
            ),
            lengths,
        )
        positions["input_ids"][packed_positions] = source
        positions["valid_mask"][packed_positions] = True
        positions["document_indices"][packed_positions] = document_ids
        positions["document_offsets"][packed_positions] = offsets.astype(
            np.int32, copy=False
        )
        positions["patch_offsets"][packed_positions] = (offsets % 4).astype(
            np.int8, copy=False
        )
        targets = np.empty_like(source)
        targets[:-1] = source[1:]
        targets[-1] = self.manifest.pad_id
        terminal = source == self.manifest.eot_id
        targets[terminal] = self.manifest.pad_id
        positions["target_ids"][packed_positions] = targets
        positions["score_mask"][packed_positions] = ~terminal
        inserted_padding = physical_width - source.size
        self.alignment_padding += inserted_padding

        combined = {
            name: np.concatenate((self._remainder[name], positions[name]))
            for name in self._POSITION_FIELDS
        }
        full_width = combined["input_ids"].size // self.chunk_size * self.chunk_size
        if not full_width:
            self._remainder = combined
            return None
        row_count = full_width // self.chunk_size
        result = {
            name: np.ascontiguousarray(values[:full_width]).reshape(
                row_count, self.chunk_size
            )
            for name, values in combined.items()
        }
        result.update(self._row_metadata(result, real_widths=None))
        self._remainder = {
            name: values[full_width:].copy() for name, values in combined.items()
        }
        return result

    def finish(self) -> dict[str, np.ndarray] | None:
        real_width = self._remainder["input_ids"].size
        if not real_width:
            return None
        result = self._empty_positions(self.chunk_size)
        for name in self._POSITION_FIELDS:
            result[name][:real_width] = self._remainder[name]
            result[name] = result[name][None, :]
        result.update(
            self._row_metadata(result, real_widths=np.asarray([real_width]))
        )
        self._remainder = self._empty_positions(0)
        return result

    def _row_metadata(
        self,
        arrays: Mapping[str, np.ndarray],
        *,
        real_widths: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        rows = arrays["input_ids"].shape[0]
        widths = (
            np.full(rows, self.chunk_size, dtype=np.int64)
            if real_widths is None
            else real_widths.astype(np.int64, copy=False)
        )
        starts = self.stream_start + np.arange(rows, dtype=np.int64) * self.chunk_size
        final_columns = widths - 1
        row_indices = np.arange(rows)
        full = widths == self.chunk_size
        halo_valid = full & arrays["score_mask"][row_indices, final_columns]
        halo_id = np.where(
            halo_valid,
            arrays["target_ids"][row_indices, final_columns],
            self.manifest.pad_id,
        ).astype("<u2", copy=False)
        metadata = {
            "chunk_index": np.arange(
                self.chunk_index, self.chunk_index + rows, dtype="<i8"
            ),
            "stream_start": starts.astype("<i8", copy=False),
            "stream_stop": (starts + widths).astype("<i8", copy=False),
            "label_halo_id": halo_id,
            "label_halo_valid": halo_valid.astype(np.bool_, copy=False),
        }
        self.chunk_index += rows
        self.stream_start += int(widths.sum())
        return metadata


class ArrayArtifactAccumulator:
    """Bounded row-block accumulator for deterministic mapped artifacts."""

    def __init__(
        self,
        output_dir: Path,
        split: str,
        *,
        rows_per_artifact: int,
    ) -> None:
        self.output_dir = output_dir
        self.split = split
        self.rows_per_artifact = int(rows_per_artifact)
        self.artifacts: list[dict[str, Any]] = []
        self._blocks: list[dict[str, np.ndarray]] = []
        self._rows = 0

    def accept(self, arrays: Mapping[str, np.ndarray] | None) -> None:
        if arrays is None:
            return
        cursor = 0
        total = int(arrays["input_ids"].shape[0])
        while cursor < total:
            take = min(self.rows_per_artifact - self._rows, total - cursor)
            self._blocks.append(
                {name: values[cursor : cursor + take] for name, values in arrays.items()}
            )
            self._rows += take
            cursor += take
            if self._rows == self.rows_per_artifact:
                self._flush()

    def finish(self) -> tuple[dict[str, Any], ...]:
        if self._rows:
            self._flush()
        return tuple(self.artifacts)

    def _flush(self) -> None:
        names = tuple(self._blocks[0])
        arrays = {
            name: (
                np.ascontiguousarray(self._blocks[0][name])
                if len(self._blocks) == 1
                else np.concatenate([block[name] for block in self._blocks])
            )
            for name in names
        }
        self.artifacts.append(
            _write_array_artifact(
                self.output_dir,
                self.split,
                len(self.artifacts),
                arrays,
            )
        )
        self._blocks.clear()
        self._rows = 0


def _build_byte_native_document_aligned_split(
    *,
    name: str,
    reader: ChallengeDocumentReader,
    output_dir: Path,
    atomic_manifest: AtomicIdManifest,
    chunk_size: int,
    chunks_per_shard: int,
    max_documents: int | None,
    overlap_tokens: int,
) -> dict[str, Any]:
    packer = VectorizedDocumentPagePacker(atomic_manifest, chunk_size=chunk_size)
    accumulator = ArrayArtifactAccumulator(
        output_dir, name, rows_per_artifact=chunks_per_shard
    )
    accepted_documents = 0
    source_tokens_in_documents = 0
    stream_bos_boundaries = 0
    first_source_document = True
    for source, lengths in reader.iter_document_batches():
        if first_source_document:
            first_source_document = False
            if lengths[0] == 1 and source[0] == atomic_manifest.eot_id:
                stream_bos_boundaries = 1
                source = source[1:]
                lengths = lengths[1:]
                if not lengths.size:
                    continue
        if max_documents is not None:
            remaining = max_documents - accepted_documents
            if remaining <= 0:
                break
            if lengths.size > remaining:
                kept_tokens = int(lengths[:remaining].sum())
                removed_lengths = lengths[remaining:]
                reader.stats.complete_documents -= len(removed_lengths)
                reader.stats.empty_documents -= int((removed_lengths == 1).sum())
                reader.stats.unique_tokens -= int(removed_lengths.sum())
                reader.stats.truncated_by_max_documents = True
                reader.stats.incomplete_tail_tokens = None
                source = source[:kept_tokens]
                lengths = lengths[:remaining]
        if not lengths.size:
            continue
        if bool((source >= atomic_manifest.mask_id).any()):
            raise ValueError(f"byte-native batch {name!r} has non-clean atomic ids")
        validate_atomic_utf8_batch(source, batch_key=f"{name}:{accepted_documents}")
        accumulator.accept(
            packer.add_batch(
                source,
                lengths,
                first_document_index=accepted_documents,
            )
        )
        accepted_documents += len(lengths)
        source_tokens_in_documents += int(source.size)
        if max_documents is not None and accepted_documents == max_documents:
            reader.stats.truncated_by_max_documents = True
            reader.stats.incomplete_tail_tokens = None
            break
    accumulator.accept(packer.finish())
    artifacts = list(accumulator.finish())
    if not artifacts:
        raise ValueError(f"split {name!r} produced no complete documents")
    stats = reader.stats
    atomic_tokens = source_tokens_in_documents
    scored_targets = sum(int(item["scored_ar_targets"]) for item in artifacts)
    return {
        "input_shards": [str(path) for path in reader.paths],
        "overlap_tokens_per_transition": overlap_tokens,
        "max_documents": max_documents,
        "truncated_by_max_documents": stats.truncated_by_max_documents,
        "physical_source_tokens_read": stats.physical_tokens,
        "unique_source_tokens_read": stats.unique_tokens,
        "source_tokens_in_complete_documents": source_tokens_in_documents,
        "complete_documents": accepted_documents,
        "stream_bos_boundaries": stream_bos_boundaries,
        "empty_documents": max(0, stats.empty_documents - stream_bos_boundaries),
        "incomplete_tail_source_tokens": stats.incomplete_tail_tokens,
        "valid_atomic_tokens": atomic_tokens,
        "literal_atomic_tokens": sum(
            int(item["literal_atomic_tokens"]) for item in artifacts
        ),
        "special_atomic_tokens": sum(
            int(item["special_atomic_tokens"]) for item in artifacts
        ),
        "eot_atomic_tokens": sum(int(item["eot_atomic_tokens"]) for item in artifacts),
        "bos_ar_targets": accepted_documents,
        "document_padding_tokens": packer.alignment_padding,
        "packed_stream_tokens": atomic_tokens + packer.alignment_padding,
        "physical_storage_positions": sum(
            int(item["physical_storage_positions"]) for item in artifacts
        ),
        "storage_padding_tokens": sum(
            int(item["storage_padding_tokens"]) for item in artifacts
        ),
        "canvas512_eligible_positions": sum(
            int(item["canvas512_eligible_positions"]) for item in artifacts
        ),
        "chunks": sum(int(item["chunks"]) for item in artifacts),
        "scored_ar_targets": scored_targets,
        "total_ar_targets": accepted_documents + scored_targets,
        "artifacts": artifacts,
    }


def build_split(
    *,
    name: str,
    paths: Sequence[Path],
    output_dir: Path,
    tokenizer: SplitTreeNumericTokenizer | None,
    atomic_manifest: AtomicIdManifest,
    overlap_tokens: int,
    chunk_size: int,
    chunks_per_shard: int,
    max_documents: int | None,
    require_one_chunk_per_document: bool,
    require_terminal_eot: bool,
    close_rows_at_document: bool,
    document_aligned_pages: bool,
    vectorized_byte_native: bool = True,
) -> dict[str, Any]:
    reader = ChallengeDocumentReader(
        paths,
        eot_id=(
            tokenizer.eot_id
            if tokenizer is not None
            else atomic_manifest.eot_id
        ),
        overlap_tokens=overlap_tokens,
        require_terminal_eot=require_terminal_eot,
    )
    if (
        tokenizer is None
        and vectorized_byte_native
        and document_aligned_pages
        and not close_rows_at_document
        and not require_one_chunk_per_document
    ):
        return _build_byte_native_document_aligned_split(
            name=name,
            reader=reader,
            output_dir=output_dir,
            atomic_manifest=atomic_manifest,
            chunk_size=chunk_size,
            chunks_per_shard=chunks_per_shard,
            max_documents=max_documents,
            overlap_tokens=overlap_tokens,
        )
    byteifier = SourceTokenByteifier(tokenizer) if tokenizer is not None else None
    packer = StreamingDocumentPacker(
        atomic_manifest,
        chunk_size=chunk_size,
        close_rows_at_document=close_rows_at_document,
        document_aligned_pages=document_aligned_pages,
    )
    buffered: list[PackedChunk] = []
    artifacts: list[dict[str, Any]] = []
    source_tokens_in_documents = 0
    atomic_tokens = 0

    def accept(chunks: Sequence[PackedChunk]) -> None:
        nonlocal buffered
        for chunk in chunks:
            buffered.append(chunk)
            if len(buffered) == chunks_per_shard:
                artifacts.append(
                    _write_artifact(
                        output_dir, name, len(artifacts), buffered
                    )
                )
                buffered = []

    accepted_documents = 0
    stream_bos_boundaries = 0
    first_source_document = True
    for source_ids in reader.iter_documents():
        if byteifier is None:
            # A loader-aligned utf8_bytes stream already uses the model's
            # atomic ids. The first leading EOT is a stream BOS, not a
            # zero-content training document.
            if first_source_document and source_ids == (atomic_manifest.eot_id,):
                stream_bos_boundaries = 1
                first_source_document = False
                continue
            atomic_ids = tuple(source_ids)
        else:
            byteifier.reset()
            atomic_ids = byteifier.byteify(source_ids).atomic_ids
        first_source_document = False
        document = AtomicDocument(
            key=f"{name}:{accepted_documents:012d}",
            atomic_ids=atomic_ids,
        )
        document.validate(atomic_manifest)
        if byteifier is None:
            validate_atomic_utf8(document.atomic_ids, document_key=document.key)
        source_tokens_in_documents += len(source_ids)
        atomic_tokens += len(document.atomic_ids)
        document_chunks = packer.add_document(accepted_documents, document)
        if require_one_chunk_per_document and len(document_chunks) != 1:
            raise ValueError(
                f"split {name!r} document {accepted_documents} produced "
                f"{len(document_chunks)} rows; the fixed one-pass contract "
                "requires exactly one row per document"
            )
        accept(document_chunks)
        accepted_documents += 1
        if max_documents is not None and accepted_documents == max_documents:
            reader.stats.truncated_by_max_documents = True
            reader.stats.incomplete_tail_tokens = None
            break
    accept(packer.finish())
    if buffered:
        artifacts.append(_write_artifact(output_dir, name, len(artifacts), buffered))
    if not artifacts:
        raise ValueError(f"split {name!r} produced no complete documents")
    stats = reader.stats
    return {
        "input_shards": [str(path) for path in paths],
        "overlap_tokens_per_transition": overlap_tokens,
        "max_documents": max_documents,
        "truncated_by_max_documents": stats.truncated_by_max_documents,
        "physical_source_tokens_read": stats.physical_tokens,
        "unique_source_tokens_read": stats.unique_tokens,
        "source_tokens_in_complete_documents": source_tokens_in_documents,
        "complete_documents": accepted_documents,
        "stream_bos_boundaries": stream_bos_boundaries,
        "empty_documents": max(
            0,
            stats.empty_documents - stream_bos_boundaries,
        ),
        "incomplete_tail_source_tokens": stats.incomplete_tail_tokens,
        "valid_atomic_tokens": atomic_tokens,
        "literal_atomic_tokens": sum(
            int(artifact["literal_atomic_tokens"]) for artifact in artifacts
        ),
        "special_atomic_tokens": sum(
            int(artifact["special_atomic_tokens"]) for artifact in artifacts
        ),
        "eot_atomic_tokens": sum(
            int(artifact["eot_atomic_tokens"]) for artifact in artifacts
        ),
        "bos_ar_targets": (
            accepted_documents
            if close_rows_at_document or document_aligned_pages
            else 1
        ),
        "document_padding_tokens": packer.alignment_padding,
        "packed_stream_tokens": atomic_tokens + packer.alignment_padding,
        "physical_storage_positions": sum(
            int(artifact["physical_storage_positions"])
            for artifact in artifacts
        ),
        "storage_padding_tokens": sum(
            int(artifact["storage_padding_tokens"])
            for artifact in artifacts
        ),
        "canvas512_eligible_positions": sum(
            int(artifact["canvas512_eligible_positions"])
            for artifact in artifacts
        ),
        "chunks": sum(int(artifact["chunks"]) for artifact in artifacts),
        "scored_ar_targets": sum(
            int(artifact["scored_ar_targets"]) for artifact in artifacts
        ),
        "total_ar_targets": (
            accepted_documents
            if close_rows_at_document or document_aligned_pages
            else 1
        )
        + sum(int(artifact["scored_ar_targets"]) for artifact in artifacts),
        "artifacts": artifacts,
    }


def _atomic_manifest(tokenizer: SplitTreeNumericTokenizer) -> AtomicIdManifest:
    return AtomicIdManifest(
        specials=tuple(
            AtomicSpecial(atomic_id=256 + source_id, name=name)
            for source_id, name in enumerate(tokenizer.spec.specials)
        ),
        eot_id=256 + tokenizer.eot_id,
    )


def _validate_byte_source_manifest(
    discovered: tuple[Path, dict[str, Any]],
    atomic_manifest: AtomicIdManifest,
) -> dict[str, Any]:
    path, manifest = discovered
    provenance = manifest.get("tokenizer_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{path} does not bind its source encoding")
    expected = {
        "kind": "utf8_bytes",
        "vocab_size": atomic_manifest.output_size,
        "eot_id": atomic_manifest.eot_id,
        "spec_sha256": atomic_manifest.sha256,
        "ngrams_sha256": None,
    }
    mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"source manifest {path} is not byte-native or uses another atomic "
            f"vocabulary: {mismatches}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "payload_sha256": canonical_sha256(manifest),
    }


def build_dataset(
    *,
    tokenizer_dir: Path | None = None,
    output_dir: Path,
    train_paths: Sequence[Path],
    validation_paths: Sequence[Path],
    chunk_size: int,
    chunks_per_shard: int,
    train_overlap_tokens: int | None = None,
    validation_overlap_tokens: int = 0,
    max_train_documents: int | None = None,
    max_validation_documents: int | None = None,
    require_one_train_chunk_per_document: bool = False,
    require_terminal_eot: bool = False,
    close_rows_at_document: bool = False,
    document_aligned_pages: bool = False,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build train/validation artifacts and atomically publish their manifest."""

    if chunks_per_shard <= 0:
        raise ValueError("chunks_per_shard must be positive")
    if close_rows_at_document and document_aligned_pages:
        raise ValueError("packing layouts are mutually exclusive")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_paths = tuple(Path(path) for path in train_paths)
    validation_paths = tuple(Path(path) for path in validation_paths)
    if not train_paths or not validation_paths:
        raise ValueError("both train and validation source shards are required")

    all_paths = tuple(dict.fromkeys((*train_paths, *validation_paths)))
    source_discoveries: list[tuple[Path, dict[str, Any]]] = []
    for parent in sorted({path.resolve().parent for path in all_paths}):
        source_manifest = parent / "mix_manifest.json"
        if not source_manifest.is_file():
            raise ValueError(
                f"source shard parent {parent} has no tokenizer-bound mix_manifest.json"
            )
        source_discoveries.append(
            (source_manifest, json.loads(source_manifest.read_text()))
        )
    train_discovery = discover_source_manifest(train_paths)
    if train_discovery is None:
        raise ValueError("training shards must share one source manifest")
    overlap = (
        infer_train_overlap(train_discovery, train_paths)
        if train_overlap_tokens is None
        else train_overlap_tokens
    )

    repository = Path(__file__).resolve().parents[1]
    base_implementation_paths = (
        Path(__file__).resolve(),
        repository / "pretraining" / "byte_diffusion" / "data.py",
        repository / "pretraining" / "byte_diffusion" / "tokenizer.py",
        *(item[0] for item in source_discoveries),
    )
    if tokenizer_dir is None:
        tokenizer = None
        atomic_manifest = AtomicIdManifest.reference()
        implementation_paths = base_implementation_paths
        source_manifests = [
            _validate_byte_source_manifest(item, atomic_manifest)
            for item in source_discoveries
        ]
        source_encoding = {
            "kind": "utf8_bytes",
            "logical_vocab_size": atomic_manifest.output_size,
            "eot_id": atomic_manifest.eot_id,
            "atomic_manifest_sha256": atomic_manifest.sha256,
        }
    else:
        tokenizer_dir = Path(tokenizer_dir)
        tokenizer = SplitTreeNumericTokenizer.from_directory(tokenizer_dir)
        validate_source_tokenizer(tokenizer)
        atomic_manifest = _atomic_manifest(tokenizer)
        spec_path = tokenizer_dir / "tokenizer.json"
        ngrams_path = tokenizer_dir / tokenizer.spec.ngrams.filename
        implementation_paths = (
            *base_implementation_paths,
            repository / "pretraining" / "bolmo_data.py",
            repository / "scripts" / "build_bolmo_dataset.py",
            spec_path,
            ngrams_path,
        )
        source_manifests = [
            validate_manifest_tokenizer(item, tokenizer_dir, tokenizer)
            for item in source_discoveries
        ]
        source_encoding = {
            "kind": "toast_tst_bridge",
            "directory": str(tokenizer_dir),
            "logical_vocab_size": tokenizer.vocab_size,
            "eot_id": tokenizer.eot_id,
            "spec_file_sha256": fingerprint_file(spec_path)["sha256"],
            "spec_canonical_sha256": tokenizer.spec.sha256(),
            "ngrams_sha256": fingerprint_file(ngrams_path)["sha256"],
            "tst_group_size": tokenizer.scheme.group_size,
            "tst_compound": tokenizer.scheme.compound,
        }
    consumed_paths = tuple(dict.fromkeys((*implementation_paths, *all_paths)))
    consumed_before = {
        str(path): fingerprint_file(path) for path in consumed_paths
    }
    splits = {
        "train": build_split(
            name="train",
            paths=train_paths,
            output_dir=output_dir,
            tokenizer=tokenizer,
            atomic_manifest=atomic_manifest,
            overlap_tokens=overlap,
            chunk_size=chunk_size,
            chunks_per_shard=chunks_per_shard,
            max_documents=max_train_documents,
            require_one_chunk_per_document=require_one_train_chunk_per_document,
            require_terminal_eot=require_terminal_eot,
            close_rows_at_document=close_rows_at_document,
            document_aligned_pages=document_aligned_pages,
        ),
        "validation": build_split(
            name="validation",
            paths=validation_paths,
            output_dir=output_dir,
            tokenizer=tokenizer,
            atomic_manifest=atomic_manifest,
            overlap_tokens=validation_overlap_tokens,
            chunk_size=chunk_size,
            chunks_per_shard=chunks_per_shard,
            max_documents=max_validation_documents,
            require_one_chunk_per_document=False,
            require_terminal_eot=require_terminal_eot,
            close_rows_at_document=close_rows_at_document,
            document_aligned_pages=document_aligned_pages,
        ),
    }
    consumed_after = {
        str(path): fingerprint_file(path) for path in consumed_paths
    }
    if consumed_after != consumed_before:
        changed = [
            path
            for path, before in consumed_before.items()
            if consumed_after.get(path) != before
        ]
        raise RuntimeError(
            "a consumed input changed during the dataset build: " + repr(changed)
        )

    manifest: dict[str, Any] = {
        "schema": DATASET_SCHEMA,
        "artifact_schema": ARTIFACT_SCHEMA,
        "artifact_format": {
            "kind": "aligned_raw_arrays/v1",
            "container_files_per_artifact": 1,
            "alignment_bytes": ARTIFACT_ALIGNMENT,
            "access": "read_only_memory_map",
            "compression": "none",
            "index": "hash-bound per-artifact arrays descriptors",
        },
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": consumed_before[str(Path(__file__).resolve())]["sha256"],
        "command": list(command) if command is not None else None,
        "atomic_vocabulary": atomic_manifest.to_dict(),
        "source_encoding": source_encoding,
        "packing": {
            "chunk_size": chunk_size,
            "patch_stride": 4,
            "layout": (
                "one_document_per_row"
                if close_rows_at_document
                else (
                    "document_aligned_pages"
                    if document_aligned_pages
                    else "dense_eot_delimited_stream"
                )
            ),
            "document_padding": (
                "independent zero-to-three PAD positions after EOT"
                if close_rows_at_document or document_aligned_pages
                else "none"
            ),
            "patch_phase": (
                "resets to zero after each terminal EOT"
                if close_rows_at_document or document_aligned_pages
                else "continuous across EOT-delimited documents"
            ),
            "chunk_alignment": "artificial starts are fixed-stride boundaries",
            "label_halo": "one clean next-atomic target when a chunk splits a document",
            "pad_is_valid": False,
            "pad_is_scored": False,
            "row_document_segments": (
                1 if close_rows_at_document else "variable"
            ),
            "one_train_chunk_per_document_required": (
                require_one_train_chunk_per_document
            ),
        },
        "tensor_fields": {
            "input_ids": "uint16 [N,chunk_size]",
            "target_ids": "uint16 [N,chunk_size]",
            "valid_mask": "bool [N,chunk_size]",
            "score_mask": "bool [N,chunk_size] aligned with input logits",
            "document_indices": "int64 [N,chunk_size], -1 on PAD",
            "document_offsets": "int32 [N,chunk_size], -1 on PAD",
            "patch_offsets": "int8 [N,chunk_size], -1 on PAD",
            "label_halo_id": "uint16 [N]",
            "label_halo_valid": "bool [N]",
            "chunk_index": "int64 [N]",
            "stream_start": "int64 [N]",
            "stream_stop": "int64 [N]",
        },
        "artifact_index_fields": {
            "row_valid_counts": "int list [N]",
            "row_physical_extents": "int list [N], last valid column plus one",
            "row_document_starts": "int list [N], document_offset-zero atoms",
            "row_sha256": (
                "hex SHA-256 list [N], each digest binds all fields in one row"
            ),
        },
        "source_manifests": source_manifests,
        "input_fingerprints": [
            consumed_before[str(path)] for path in all_paths
        ],
        "implementation_fingerprints": [
            consumed_before[str(path)] for path in implementation_paths
        ],
        "splits": splits,
    }
    manifest["payload_sha256"] = canonical_sha256(manifest)
    working = output_dir / "manifest.json.working"
    working.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(working, output_dir / "manifest.json")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="legacy ToaST bridge; omit for a byte-native utf8_bytes source",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", action="append", required=True, metavar="GLOB")
    parser.add_argument(
        "--validation", action="append", required=True, metavar="GLOB"
    )
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--chunks-per-shard", type=int, default=256)
    parser.add_argument("--train-overlap-tokens", type=int, default=None)
    parser.add_argument("--validation-overlap-tokens", type=int, default=0)
    parser.add_argument("--max-train-documents", type=int, default=None)
    parser.add_argument(
        "--require-one-train-chunk-per-document",
        action="store_true",
        help=(
            "fail unless each selected training document produces exactly one "
            "physical row; used by fixed one-pass optimizer-step contracts"
        ),
    )
    parser.add_argument("--max-validation-documents", type=int, default=None)
    parser.add_argument("--require-terminal-eot", action="store_true")
    parser.add_argument(
        "--close-rows-at-document",
        action="store_true",
        help="legacy one-document-per-row diagnostic layout",
    )
    parser.add_argument(
        "--packing-layout",
        choices=("document_aligned_pages", "dense_eot_delimited_stream"),
        default="document_aligned_pages",
        help=(
            "production default isolates documents while packing many into a page; "
            "dense stream is a causal-only control"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_dataset(
        tokenizer_dir=args.tokenizer,
        output_dir=args.output,
        train_paths=expand_patterns(args.train),
        validation_paths=expand_patterns(args.validation),
        chunk_size=args.chunk_size,
        chunks_per_shard=args.chunks_per_shard,
        train_overlap_tokens=args.train_overlap_tokens,
        validation_overlap_tokens=args.validation_overlap_tokens,
        max_train_documents=args.max_train_documents,
        max_validation_documents=args.max_validation_documents,
        require_one_train_chunk_per_document=(
            args.require_one_train_chunk_per_document
        ),
        require_terminal_eot=args.require_terminal_eot,
        close_rows_at_document=args.close_rows_at_document,
        document_aligned_pages=(
            not args.close_rows_at_document
            and args.packing_layout == "document_aligned_pages"
        ),
        command=sys.argv if argv is None else [str(Path(__file__)), *argv],
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_sha256": manifest["payload_sha256"],
                "splits": {
                    name: {
                        "documents": split["complete_documents"],
                        "chunks": split["chunks"],
                        "atomic_tokens": split["valid_atomic_tokens"],
                    }
                    for name, split in manifest["splits"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
