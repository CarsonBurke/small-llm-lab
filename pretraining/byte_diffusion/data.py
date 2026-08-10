"""Dataset contracts and deterministic document packing for byte diffusion.

The model vocabulary is deliberately smaller than a tokenizer vocabulary:
literal octets occupy ids 0..255, five manifest-ordered controls occupy
256..260, and MASK/PAD are input-only ids 261 and 262. This module keeps the
control-token ordering in the model/data manifest rather than deriving it from
an unrelated source tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ATOMIC_MANIFEST_SCHEMA = "byte_diffusion_atomic_ids/v2"
CURSOR_SCHEMA = "byte_diffusion_cursor/v2"
BYTE_COUNT = 256
CLEAN_SPECIAL_COUNT = 5
MASK_ID = 261
PAD_ID = 262
PATCH_STRIDE = 4


@dataclass(frozen=True, order=True)
class AtomicSpecial:
    """One explicitly ordered clean control token."""

    atomic_id: int
    name: str

    def __post_init__(self) -> None:
        if type(self.atomic_id) is not int:
            raise TypeError("atomic special id must be an integer")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("special name must be non-empty")


@dataclass(frozen=True)
class AtomicIdManifest:
    """Validated, serializable mapping between controls and atomic ids.

    ``specials`` is a sequence rather than a mapping so its order is part of
    the serialized contract.  Validation additionally requires the records to
    be sorted by ``atomic_id``; JSON object insertion order is never consulted.
    """

    specials: tuple[AtomicSpecial, ...]
    eot_id: int
    schema: str = ATOMIC_MANIFEST_SCHEMA
    byte_count: int = BYTE_COUNT
    mask_id: int = MASK_ID
    pad_id: int = PAD_ID

    def __post_init__(self) -> None:
        scalar_ids = {
            "byte_count": self.byte_count,
            "eot_id": self.eot_id,
            "mask_id": self.mask_id,
            "pad_id": self.pad_id,
        }
        if any(type(value) is not int for value in scalar_ids.values()):
            raise TypeError(f"atomic manifest ids must be integers: {scalar_ids}")
        if self.schema != ATOMIC_MANIFEST_SCHEMA:
            raise ValueError(
                f"unsupported atomic manifest schema {self.schema!r}; "
                f"expected {ATOMIC_MANIFEST_SCHEMA!r}"
            )
        if self.byte_count != BYTE_COUNT:
            raise ValueError("version 1 requires exactly 256 literal byte ids")
        expected_ids = tuple(range(BYTE_COUNT, BYTE_COUNT + CLEAN_SPECIAL_COUNT))
        actual_ids = tuple(special.atomic_id for special in self.specials)
        if actual_ids != expected_ids:
            raise ValueError(
                "specials must be an explicit ordered list with atomic ids "
                f"{expected_ids}, got {actual_ids}"
            )
        names = tuple(special.name for special in self.specials)
        if len(set(names)) != len(names):
            raise ValueError("special names must be unique")
        if self.eot_id not in actual_ids:
            raise ValueError("eot_id must identify one of the clean specials")
        if self.mask_id != BYTE_COUNT + CLEAN_SPECIAL_COUNT:
            raise ValueError("MASK must immediately follow the clean output ids")
        if self.pad_id != self.mask_id + 1:
            raise ValueError("PAD must immediately follow MASK")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AtomicIdManifest":
        raw_specials = value.get("specials")
        if not isinstance(raw_specials, list):
            raise ValueError("atomic manifest specials must be an ordered JSON list")
        specials: list[AtomicSpecial] = []
        for index, record in enumerate(raw_specials):
            if not isinstance(record, Mapping):
                raise ValueError(f"special record {index} must be an object")
            try:
                specials.append(
                    AtomicSpecial(
                        atomic_id=record["atomic_id"],
                        name=record["name"],
                    )
                )
            except KeyError as error:
                raise ValueError(
                    f"special record {index} is missing {error.args[0]!r}"
                ) from error
        try:
            return cls(
                specials=tuple(specials),
                eot_id=value["eot_id"],
                schema=value.get("schema", ATOMIC_MANIFEST_SCHEMA),
                byte_count=value.get("byte_count", BYTE_COUNT),
                mask_id=value.get("mask_id", MASK_ID),
                pad_id=value.get("pad_id", PAD_ID),
            )
        except KeyError as error:
            raise ValueError(f"atomic manifest is missing {error.args[0]!r}") from error

    @classmethod
    def reference(cls) -> "AtomicIdManifest":
        """Return the current five-special challenge vocabulary."""

        names = (
            "<|endoftext|>",
            "<think>",
            "</think>",
            "<answer>",
            "</answer>",
        )
        return cls(
            specials=tuple(
                AtomicSpecial(atomic_id=BYTE_COUNT + i, name=name)
                for i, name in enumerate(names)
            ),
            eot_id=BYTE_COUNT,
        )

    @property
    def output_size(self) -> int:
        return self.mask_id

    @property
    def input_size(self) -> int:
        return self.pad_id + 1

    @property
    def special_by_name(self) -> dict[str, AtomicSpecial]:
        return {special.name: special for special in self.specials}

    @property
    def special_by_id(self) -> dict[int, AtomicSpecial]:
        return {special.atomic_id: special for special in self.specials}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "byte_count": self.byte_count,
            "specials": [
                {
                    "atomic_id": special.atomic_id,
                    "name": special.name,
                }
                for special in self.specials
            ],
            "eot_id": self.eot_id,
            "mask_id": self.mask_id,
            "pad_id": self.pad_id,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate_clean_id(self, atomic_id: int) -> None:
        if type(atomic_id) is not int:
            raise TypeError(f"clean atomic id must be an integer, got {atomic_id!r}")
        if not 0 <= atomic_id < self.output_size:
            raise ValueError(
                f"clean atomic id must be in [0, {self.output_size}), got {atomic_id}"
            )


@dataclass(frozen=True)
class SpecialAtom:
    """A named special used by :class:`AtomicCodec` encode/decode calls."""

    name: str


AtomicPart = bytes | SpecialAtom


class AtomicCodec:
    """Lossless conversion between literal byte segments/specials and ids.

    Text strings are intentionally not accepted: callers must make UTF-8
    encoding explicit by passing ``text.encode("utf-8")``.  Decode coalesces
    adjacent literal ids into byte segments and preserves specials as typed
    :class:`SpecialAtom` values.
    """

    def __init__(self, manifest: AtomicIdManifest):
        self.manifest = manifest

    def encode(self, parts: Iterable[AtomicPart]) -> tuple[int, ...]:
        encoded: list[int] = []
        by_name = self.manifest.special_by_name
        for part in parts:
            if isinstance(part, bytes):
                encoded.extend(part)
            elif isinstance(part, SpecialAtom):
                try:
                    encoded.append(by_name[part.name].atomic_id)
                except KeyError as error:
                    raise ValueError(f"unknown atomic special {part.name!r}") from error
            else:
                raise TypeError(
                    "atomic parts must be bytes or SpecialAtom, "
                    f"got {type(part).__name__}"
                )
        return tuple(encoded)

    def decode(self, atomic_ids: Iterable[int]) -> tuple[AtomicPart, ...]:
        decoded: list[AtomicPart] = []
        literals = bytearray()
        by_id = self.manifest.special_by_id

        def flush_literals() -> None:
            if literals:
                decoded.append(bytes(literals))
                literals.clear()

        for raw_id in atomic_ids:
            if type(raw_id) is not int:
                raise TypeError(f"atomic id must be an integer, got {raw_id!r}")
            atomic_id = raw_id
            if 0 <= atomic_id < BYTE_COUNT:
                literals.append(atomic_id)
                continue
            special = by_id.get(atomic_id)
            if special is None:
                raise ValueError(
                    f"input-only or invalid atomic id {atomic_id} cannot be decoded"
                )
            flush_literals()
            decoded.append(SpecialAtom(special.name))
        flush_literals()
        return tuple(decoded)

    def decode_bytes(self, atomic_ids: Iterable[int]) -> bytes:
        parts = self.decode(atomic_ids)
        if any(isinstance(part, SpecialAtom) for part in parts):
            raise ValueError("cannot decode clean specials as literal bytes")
        return b"".join(part for part in parts if isinstance(part, bytes))


@dataclass(frozen=True)
class AtomicDocument:
    """One clean document whose final atomic symbol is EOT."""

    key: str
    atomic_ids: tuple[int, ...]

    def validate(self, manifest: AtomicIdManifest) -> None:
        if not self.key:
            raise ValueError("document key must be non-empty")
        if not self.atomic_ids:
            raise ValueError(f"document {self.key!r} is empty")
        for atomic_id in self.atomic_ids:
            manifest.validate_clean_id(atomic_id)
        eot_positions = tuple(
            index
            for index, atomic_id in enumerate(self.atomic_ids)
            if atomic_id == manifest.eot_id
        )
        if eot_positions != (len(self.atomic_ids) - 1,):
            raise ValueError(
                f"document {self.key!r} must contain exactly one EOT at its end; "
                f"found positions {eot_positions}"
            )


@dataclass(frozen=True)
class ChunkDocumentSpan:
    document_index: int
    document_key: str
    chunk_start: int
    chunk_stop: int
    document_start: int
    document_stop: int


@dataclass(frozen=True)
class PackedChunk:
    """One patch-aligned training chunk and its shifted-label metadata."""

    chunk_index: int
    stream_start: int
    stream_stop: int
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    valid_mask: tuple[bool, ...]
    score_mask: tuple[bool, ...]
    document_indices: tuple[int, ...]
    document_offsets: tuple[int, ...]
    patch_offsets: tuple[int, ...]
    label_halo_id: int
    label_halo_valid: bool
    document_spans: tuple[ChunkDocumentSpan, ...]

    def __post_init__(self) -> None:
        width = len(self.input_ids)
        aligned_fields = (
            self.target_ids,
            self.valid_mask,
            self.score_mask,
            self.document_indices,
            self.document_offsets,
            self.patch_offsets,
        )
        if width == 0 or width % PATCH_STRIDE:
            raise ValueError("packed chunk width must be a positive multiple of four")
        if any(len(field) != width for field in aligned_fields):
            raise ValueError("all packed chunk fields must have identical widths")
        if self.stream_start % PATCH_STRIDE:
            raise ValueError("packed chunk stream_start must be patch aligned")
        for index, (valid, score) in enumerate(
            zip(self.valid_mask, self.score_mask, strict=True)
        ):
            if score and not valid:
                raise ValueError(f"score position {index} is not valid input")
            if not valid:
                if self.input_ids[index] != PAD_ID or self.target_ids[index] != PAD_ID:
                    raise ValueError("invalid storage positions must contain PAD")
                if self.document_indices[index] != -1:
                    raise ValueError("PAD positions cannot belong to a document")
                if self.document_offsets[index] != -1 or self.patch_offsets[index] != -1:
                    raise ValueError("PAD positions cannot carry document offsets")
                continue
            if not 0 <= self.input_ids[index] < MASK_ID:
                raise ValueError("clean packed input positions cannot contain MASK or PAD")
            if self.document_indices[index] < 0 or self.document_offsets[index] < 0:
                raise ValueError("valid positions require document metadata")
            if not 0 <= self.patch_offsets[index] < PATCH_STRIDE:
                raise ValueError("valid positions require a patch offset in [0, 4)")
            if score and not 0 <= self.target_ids[index] < MASK_ID:
                raise ValueError("scored packed targets must be clean atomic ids")
            if not score and self.target_ids[index] != PAD_ID:
                raise ValueError("unscored packed positions must have PAD targets")
        if self.label_halo_valid:
            if not 0 <= self.label_halo_id < MASK_ID:
                raise ValueError("valid halo must be a clean atomic id")
            if not self.score_mask[-1] or self.target_ids[-1] != self.label_halo_id:
                raise ValueError("valid halo must supply the final shifted target")
        elif self.label_halo_id != PAD_ID:
            raise ValueError("invalid halo must contain PAD")


def pack_documents(
    documents: Sequence[AtomicDocument],
    manifest: AtomicIdManifest,
    *,
    chunk_size: int,
) -> tuple[PackedChunk, ...]:
    """Pad documents independently, concatenate them, and make aligned chunks.

    A document must contain one terminal EOT.  Its EOT closes the final short
    patch; zero to three PAD slots are then inserted before the next document,
    whose patch offset restarts at zero.  Artificial chunk boundaries are
    multiples of four.  When a chunk ends inside a document,
    ``label_halo_id`` carries its one clean shifted target without adding that
    target to the input width.
    """

    if chunk_size <= 0 or chunk_size % PATCH_STRIDE:
        raise ValueError("chunk_size must be a positive multiple of four")
    keys = tuple(document.key for document in documents)
    if len(set(keys)) != len(keys):
        raise ValueError("document keys must be unique")
    for document in documents:
        document.validate(manifest)

    stream_ids: list[int] = []
    stream_targets: list[int] = []
    stream_valid: list[bool] = []
    stream_score: list[bool] = []
    stream_document: list[int] = []
    stream_document_offset: list[int] = []
    stream_patch_offset: list[int] = []

    for document_index, document in enumerate(documents):
        for offset, atomic_id in enumerate(document.atomic_ids):
            has_target = offset + 1 < len(document.atomic_ids)
            stream_ids.append(atomic_id)
            stream_targets.append(
                document.atomic_ids[offset + 1] if has_target else manifest.pad_id
            )
            stream_valid.append(True)
            stream_score.append(has_target)
            stream_document.append(document_index)
            stream_document_offset.append(offset)
            stream_patch_offset.append(offset % PATCH_STRIDE)
        document_padding = (-len(document.atomic_ids)) % PATCH_STRIDE
        for _ in range(document_padding):
            stream_ids.append(manifest.pad_id)
            stream_targets.append(manifest.pad_id)
            stream_valid.append(False)
            stream_score.append(False)
            stream_document.append(-1)
            stream_document_offset.append(-1)
            stream_patch_offset.append(-1)

    if not stream_ids:
        return ()

    chunks: list[PackedChunk] = []
    real_stream_length = len(stream_ids)
    for chunk_index, start in enumerate(range(0, real_stream_length, chunk_size)):
        stop = min(start + chunk_size, real_stream_length)
        width = stop - start
        storage_padding = chunk_size - width

        input_ids = stream_ids[start:stop] + [manifest.pad_id] * storage_padding
        target_ids = stream_targets[start:stop] + [manifest.pad_id] * storage_padding
        valid_mask = stream_valid[start:stop] + [False] * storage_padding
        score_mask = stream_score[start:stop] + [False] * storage_padding
        document_indices = stream_document[start:stop] + [-1] * storage_padding
        document_offsets = stream_document_offset[start:stop] + [-1] * storage_padding
        patch_offsets = stream_patch_offset[start:stop] + [-1] * storage_padding

        last_real = stop - 1
        halo_valid = bool(
            stop == start + chunk_size and stream_score[last_real]
        )
        halo_id = stream_targets[last_real] if halo_valid else manifest.pad_id

        spans: list[ChunkDocumentSpan] = []
        cursor = 0
        while cursor < width:
            document_index = document_indices[cursor]
            if document_index < 0:
                cursor += 1
                continue
            span_start = cursor
            document_start = document_offsets[cursor]
            cursor += 1
            while cursor < width and document_indices[cursor] == document_index:
                cursor += 1
            spans.append(
                ChunkDocumentSpan(
                    document_index=document_index,
                    document_key=documents[document_index].key,
                    chunk_start=span_start,
                    chunk_stop=cursor,
                    document_start=document_start,
                    document_stop=document_offsets[cursor - 1] + 1,
                )
            )

        chunks.append(
            PackedChunk(
                chunk_index=chunk_index,
                stream_start=start,
                stream_stop=stop,
                input_ids=tuple(input_ids),
                target_ids=tuple(target_ids),
                valid_mask=tuple(valid_mask),
                score_mask=tuple(score_mask),
                document_indices=tuple(document_indices),
                document_offsets=tuple(document_offsets),
                patch_offsets=tuple(patch_offsets),
                label_halo_id=halo_id,
                label_halo_valid=halo_valid,
                document_spans=tuple(spans),
            )
        )
    return tuple(chunks)


def packed_chunks_sha256(chunks: Sequence[PackedChunk]) -> str:
    """Stable identity used to reject cursor restore against different data."""

    digest = hashlib.sha256()
    for chunk in chunks:
        record = {
            "chunk_index": chunk.chunk_index,
            "stream_start": chunk.stream_start,
            "stream_stop": chunk.stream_stop,
            "input_ids": chunk.input_ids,
            "target_ids": chunk.target_ids,
            "valid_mask": chunk.valid_mask,
            "score_mask": chunk.score_mask,
            "document_indices": chunk.document_indices,
            "document_offsets": chunk.document_offsets,
            "patch_offsets": chunk.patch_offsets,
            "label_halo_id": chunk.label_halo_id,
            "label_halo_valid": chunk.label_halo_valid,
            "document_spans": [
                {
                    "document_index": span.document_index,
                    "document_key": span.document_key,
                    "chunk_start": span.chunk_start,
                    "chunk_stop": span.chunk_stop,
                    "document_start": span.document_start,
                    "document_stop": span.document_stop,
                }
                for span in chunk.document_spans
            ],
        }
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


class DeterministicChunkCursor:
    """Epoch-shuffled chunk cursor with exact, validated resume state."""

    def __init__(
        self,
        chunks: Sequence[PackedChunk],
        *,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        if len(chunks) == 0:
            raise ValueError("cursor requires at least one packed chunk")
        self._chunks = chunks
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        identity = getattr(chunks, "dataset_sha256", None)
        self.dataset_sha256 = str(
            identity if identity is not None else packed_chunks_sha256(self._chunks)
        )
        self.epoch = 0
        self.position = 0
        self._order = self._order_for_epoch(self.epoch)

    def _order_for_epoch(self, epoch: int) -> np.ndarray:
        order = np.arange(len(self._chunks), dtype=np.int64)
        if self.shuffle:
            seed_material = f"{self.seed}:{epoch}".encode("ascii")
            epoch_seed = int.from_bytes(
                hashlib.sha256(seed_material).digest()[:16], "big"
            )
            custom_order = getattr(self._chunks, "shuffled_indices", None)
            if custom_order is None:
                np.random.default_rng(epoch_seed).shuffle(order)
            else:
                order = np.asarray(custom_order(epoch_seed), dtype=np.int64)
        if order.shape != (len(self._chunks),):
            raise ValueError("custom epoch order has the wrong shape")
        return order

    def next_index(self) -> int:
        if self.position == len(self._order):
            self.epoch += 1
            self.position = 0
            self._order = self._order_for_epoch(self.epoch)
        index = int(self._order[self.position])
        self.position += 1
        return index

    def next_indices(self, count: int) -> np.ndarray:
        """Advance through ``count`` rows without boxing every index."""

        if count <= 0:
            raise ValueError("cursor batch size must be positive")
        result = np.empty(count, dtype=np.int64)
        filled = 0
        while filled < count:
            if self.position == len(self._order):
                self.epoch += 1
                self.position = 0
                self._order = self._order_for_epoch(self.epoch)
            take = min(count - filled, len(self._order) - self.position)
            result[filled : filled + take] = self._order[
                self.position : self.position + take
            ]
            self.position += take
            filled += take
        return result

    def next_chunk(self) -> PackedChunk:
        return self._chunks[self.next_index()]

    def chunk_at(self, index: int) -> PackedChunk:
        """Materialize one known cursor index without advancing its state."""

        return self._chunks[index]

    @property
    def chunks(self) -> Sequence[PackedChunk]:
        """Expose the immutable bound dataset for native batch materialization."""

        return self._chunks

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": CURSOR_SCHEMA,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "dataset_sha256": self.dataset_sha256,
            "chunk_count": len(self._chunks),
            "epoch": self.epoch,
            "position": self.position,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema": CURSOR_SCHEMA,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "dataset_sha256": self.dataset_sha256,
            "chunk_count": len(self._chunks),
        }
        for key, expected_value in expected.items():
            if state.get(key) != expected_value:
                raise ValueError(
                    f"cursor state {key} mismatch: expected {expected_value!r}, "
                    f"got {state.get(key)!r}"
                )
        epoch = int(state.get("epoch", -1))
        position = int(state.get("position", -1))
        if epoch < 0:
            raise ValueError("cursor epoch must be non-negative")
        if not 0 <= position <= len(self._chunks):
            raise ValueError("cursor position is outside the epoch")
        self.epoch = epoch
        self.position = position
        self._order = self._order_for_epoch(epoch)
