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


DATASET_SCHEMA = "byte_diffusion_dataset/v3"
ARTIFACT_SCHEMA = "byte_diffusion_chunks/v2"


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


class StreamingDocumentPacker:
    """Bounded streaming equivalent of ``data.pack_documents``."""

    def __init__(self, manifest: AtomicIdManifest, *, chunk_size: int) -> None:
        if chunk_size <= 0 or chunk_size % 4:
            raise ValueError("chunk_size must be a positive multiple of four")
        self.manifest = manifest
        self.chunk_size = int(chunk_size)
        self.chunk_index = 0
        self.stream_start = 0
        self.document_keys: dict[int, str] = {}
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
                patch_offset=offset % 4,
            )
            if len(self._input) == self.chunk_size:
                ready.append(self._flush(full=True))
        for _ in range((-len(document.atomic_ids)) % 4):
            self._append(
                input_id=self.manifest.pad_id,
                target_id=self.manifest.pad_id,
                valid=False,
                score=False,
                document_index=-1,
                document_offset=-1,
                patch_offset=-1,
            )
            if len(self._input) == self.chunk_size:
                ready.append(self._flush(full=True))
        # Production branch attention has one varlen document segment per
        # physical row.  Close a partial row here instead of co-packing the
        # next document and later materializing/repacking the entire corpus.
        if self._input:
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
    filename = f"{split}-{artifact_index:05d}.npz"
    path = output_dir / filename
    arrays = _chunk_arrays(chunks)
    write_deterministic_npz(path, arrays)
    valid_ids = arrays["input_ids"][arrays["valid_mask"]]
    literal_atomic_tokens = int((valid_ids < 256).sum())
    special_atomic_tokens = int((valid_ids >= 256).sum())
    physical_storage_positions = int(arrays["input_ids"].size)
    valid_atomic_tokens = int(arrays["valid_mask"].sum())
    return {
        "schema": ARTIFACT_SCHEMA,
        "path": filename,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "chunks": len(chunks),
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
    byteifier = SourceTokenByteifier(tokenizer) if tokenizer is not None else None
    packer = StreamingDocumentPacker(atomic_manifest, chunk_size=chunk_size)
    buffered: list[PackedChunk] = []
    artifacts: list[dict[str, Any]] = []
    source_tokens_in_documents = 0
    atomic_tokens = 0
    document_padding = 0

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
        document_padding += (-len(document.atomic_ids)) % 4
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
        "bos_ar_targets": accepted_documents,
        "document_padding_tokens": document_padding,
        "packed_stream_tokens": atomic_tokens + document_padding,
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
        "total_ar_targets": accepted_documents
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
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build train/validation artifacts and atomically publish their manifest."""

    if chunks_per_shard <= 0:
        raise ValueError("chunks_per_shard must be positive")
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
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": consumed_before[str(Path(__file__).resolve())]["sha256"],
        "command": list(command) if command is not None else None,
        "atomic_vocabulary": atomic_manifest.to_dict(),
        "source_encoding": source_encoding,
        "packing": {
            "chunk_size": chunk_size,
            "patch_stride": 4,
            "document_padding": "independent zero-to-three PAD positions after EOT",
            "patch_phase": "resets to zero after each terminal EOT",
            "chunk_alignment": "artificial starts are fixed-stride boundaries",
            "label_halo": "one clean next-atomic target when a chunk splits a document",
            "pad_is_valid": False,
            "pad_is_scored": False,
            "row_document_segments": 1,
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
