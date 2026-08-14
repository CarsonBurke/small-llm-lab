"""Causal, reproducible entropy patching for byte-diffusion models.

The boundary decision for byte ``i`` is made from an estimate of
``p(x_i | x_<i)``.  Consequently neither context hashing nor patch routing may
read byte ``i``.  The compact estimator below is deliberately simple enough to
fit from scratch, serialize with the submission, and reproduce exactly during
inference.  It is a count-smoothed hashed byte n-gram model, not a claim that a
small count model matches the separately trained entropy Transformer used by
BLT.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import struct
from typing import Literal, Mapping

import numpy as np
import torch
from torch import Tensor


_MODEL_MAGIC = b"BDENTR1\0"
_PATCHER_MAGIC = b"BDPTCH1\0"
_HEADER_LENGTH_BYTES = 8
_DIGEST_BYTES = 32
_MAX_HEADER_BYTES = 1 << 20
_HASH_BASE = 1_000_000_007
_DEPTH_MIX = 97_531

EntropyUnit = Literal["nats", "bits"]
BoundaryMode = Literal["threshold", "cumulative"]


def _canonical_json(record: Mapping[str, object]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _numpy_1d(values: np.ndarray | Tensor, *, name: str) -> np.ndarray:
    if isinstance(values, Tensor):
        if values.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU when converted to NumPy")
        values = values.detach().numpy()
    result = np.asarray(values)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _validate_document_ids(document_ids: np.ndarray, length: int) -> np.ndarray:
    documents = _numpy_1d(document_ids, name="document_ids").astype(
        np.int64, copy=False
    )
    if documents.size != length:
        raise ValueError("document_ids must align with the byte sequence")
    if length == 0:
        return documents
    if bool((documents < 0).any()):
        raise ValueError("packed document ids must be nonnegative")
    starts = np.r_[True, documents[1:] != documents[:-1]]
    run_ids = documents[starts]
    if np.unique(run_ids).size != run_ids.size:
        raise ValueError("each document must occupy one contiguous packed run")
    return documents


def document_ids_from_eot(
    ids: np.ndarray | Tensor, *, eot_id: int = 256
) -> np.ndarray:
    """Return contiguous document ids, resetting context after every EOT byte."""

    byte_ids = _numpy_1d(ids, name="ids")
    if byte_ids.size == 0:
        return np.empty(0, dtype=np.int64)
    starts = np.zeros(byte_ids.size, dtype=np.int64)
    starts[1:] = np.asarray(byte_ids[:-1] == eot_id, dtype=np.int64)
    return np.cumsum(starts, dtype=np.int64)


@dataclass(frozen=True)
class HashedNgramEntropyConfig:
    """Artifact-budgeted causal entropy estimator configuration.

    The default count payload is ``1024 * 261 * 4 = 1,069,056`` bytes.  Hash
    collisions trade estimator quality for a small, deterministic artifact.
    """

    schema_version: int = 1
    vocab_size: int = 261
    context_order: int = 4
    table_size: int = 1_024
    additive_smoothing: float = 0.5
    entropy_unit: EntropyUnit = "nats"
    hash_seed: int = 17

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported entropy-estimator schema version")
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must exceed one")
        if self.context_order <= 0 or self.table_size <= 0:
            raise ValueError("context_order and table_size must be positive")
        if not math.isfinite(self.additive_smoothing) or self.additive_smoothing <= 0:
            raise ValueError("additive_smoothing must be finite and positive")
        if self.entropy_unit not in {"nats", "bits"}:
            raise ValueError(f"unsupported entropy unit {self.entropy_unit!r}")
        if self.hash_seed < 0:
            raise ValueError("hash_seed must be nonnegative")


def context_hashes_numpy(
    ids: np.ndarray | Tensor,
    document_ids: np.ndarray | Tensor,
    config: HashedNgramEntropyConfig,
) -> np.ndarray:
    """Hash only the visible prefix preceding each target byte.

    The small Python loop is over the fixed n-gram order, never over bytes.
    Every operation within an order is vectorized across the packed sequence.
    """

    byte_ids = _numpy_1d(ids, name="ids").astype(np.int64, copy=False)
    documents = _validate_document_ids(
        _numpy_1d(document_ids, name="document_ids"), byte_ids.size
    )
    if bool(((byte_ids < 0) | (byte_ids >= config.vocab_size)).any()):
        raise ValueError("ids fall outside the entropy estimator vocabulary")
    hashes = np.full(
        byte_ids.size, config.hash_seed % config.table_size, dtype=np.int64
    )
    depths = np.zeros(byte_ids.size, dtype=np.int64)
    # Consume the potential context from oldest to newest. Inactive leading
    # slots remain explicit zero sentinels, while depth mixing distinguishes
    # short prefixes from full contexts with the same byte polynomial.
    for lag in range(config.context_order, 0, -1):
        shifted = np.zeros(byte_ids.size, dtype=np.int64)
        active = np.zeros(byte_ids.size, dtype=np.bool_)
        if lag < byte_ids.size:
            same_document = documents[lag:] == documents[:-lag]
            active[lag:] = same_document
            shifted[lag:] = np.where(same_document, byte_ids[:-lag] + 1, 0)
        hashes = (hashes * _HASH_BASE + shifted) % config.table_size
        depths += active
    hashes = (hashes + depths * _DEPTH_MIX) % config.table_size
    return hashes


def context_hashes_torch(
    ids: Tensor, document_ids: Tensor, config: HashedNgramEntropyConfig
) -> Tensor:
    """Torch equivalent of :func:`context_hashes_numpy` for exact inference."""

    if ids.ndim != 1 or document_ids.ndim != 1 or ids.shape != document_ids.shape:
        raise ValueError("ids and document_ids must be aligned one-dimensional tensors")
    byte_ids = ids.to(torch.int64)
    documents = document_ids.to(device=ids.device, dtype=torch.int64)
    if not torch.compiler.is_compiling():
        if bool(((byte_ids < 0) | (byte_ids >= config.vocab_size)).any()):
            raise ValueError("ids fall outside the entropy estimator vocabulary")
        if bool((documents < 0).any()):
            raise ValueError("packed document ids must be nonnegative")
    hashes = torch.full_like(byte_ids, config.hash_seed % config.table_size)
    depths = torch.zeros_like(byte_ids)
    for lag in range(config.context_order, 0, -1):
        shifted = torch.zeros_like(byte_ids)
        active = torch.zeros_like(byte_ids, dtype=torch.bool)
        if lag < byte_ids.numel():
            same_document = documents[lag:] == documents[:-lag]
            active[lag:] = same_document
            shifted[lag:] = torch.where(
                same_document, byte_ids[:-lag] + 1, torch.zeros_like(byte_ids[:-lag])
            )
        hashes = torch.remainder(hashes * _HASH_BASE + shifted, config.table_size)
        depths += active.to(depths.dtype)
    return torch.remainder(hashes + depths * _DEPTH_MIX, config.table_size)


class HashedNgramEntropyModel:
    """Count-smoothed hashed n-gram next-byte entropy estimator."""

    def __init__(
        self,
        config: HashedNgramEntropyConfig = HashedNgramEntropyConfig(),
        counts: np.ndarray | None = None,
    ) -> None:
        self.config = config
        expected_shape = (config.table_size, config.vocab_size)
        if counts is None:
            self.counts = np.zeros(expected_shape, dtype="<u4")
        else:
            observed = np.asarray(counts)
            if observed.shape != expected_shape:
                raise ValueError(
                    f"counts shape must be {expected_shape}, observed {observed.shape}"
                )
            if observed.dtype.kind not in {"u", "i"} or bool((observed < 0).any()):
                raise ValueError("counts must be nonnegative integers")
            if observed.size and int(observed.max()) > np.iinfo(np.uint32).max:
                raise ValueError("counts exceed the uint32 serialization contract")
            self.counts = np.ascontiguousarray(observed, dtype="<u4")
        self._entropy_cache: np.ndarray | None = None
        self._torch_entropy_cache: dict[str, Tensor] = {}

    @classmethod
    def fit(
        cls,
        ids: np.ndarray | Tensor,
        document_ids: np.ndarray | Tensor,
        config: HashedNgramEntropyConfig = HashedNgramEntropyConfig(),
    ) -> "HashedNgramEntropyModel":
        model = cls(config)
        model.update(ids, document_ids)
        return model

    @property
    def parameter_bytes(self) -> int:
        return int(self.counts.nbytes)

    def update(
        self, ids: np.ndarray | Tensor, document_ids: np.ndarray | Tensor
    ) -> None:
        """Accumulate one packed corpus block without per-byte Python work."""

        byte_ids = _numpy_1d(ids, name="ids").astype(np.int64, copy=False)
        documents = _validate_document_ids(
            _numpy_1d(document_ids, name="document_ids"), byte_ids.size
        )
        hashes = context_hashes_numpy(byte_ids, documents, self.config)
        flat = hashes * self.config.vocab_size + byte_ids
        increments = np.bincount(
            flat,
            minlength=self.config.table_size * self.config.vocab_size,
        ).reshape(self.counts.shape)
        updated = self.counts.astype(np.uint64) + increments.astype(np.uint64)
        if updated.size and int(updated.max()) > np.iinfo(np.uint32).max:
            raise OverflowError("entropy-estimator count exceeds uint32 capacity")
        self.counts[...] = updated.astype("<u4")
        self._entropy_cache = None
        self._torch_entropy_cache.clear()

    def entropy_by_hash(self) -> np.ndarray:
        if self._entropy_cache is None:
            values = self.counts.astype(np.float64)
            values += self.config.additive_smoothing
            probabilities = values / values.sum(axis=1, keepdims=True)
            entropies = -(probabilities * np.log(probabilities)).sum(axis=1)
            if self.config.entropy_unit == "bits":
                entropies /= math.log(2.0)
            self._entropy_cache = np.asarray(entropies, dtype=np.float64)
        return self._entropy_cache

    def predict_entropies_numpy(
        self, ids: np.ndarray | Tensor, document_ids: np.ndarray | Tensor
    ) -> np.ndarray:
        hashes = context_hashes_numpy(ids, document_ids, self.config)
        return self.entropy_by_hash()[hashes]

    def predict_entropies_torch(self, ids: Tensor, document_ids: Tensor) -> Tensor:
        hashes = context_hashes_torch(ids, document_ids, self.config)
        device_key = str(ids.device)
        table = self._torch_entropy_cache.get(device_key)
        if table is None:
            table = torch.from_numpy(self.entropy_by_hash()).to(
                device=ids.device, dtype=torch.float64
            )
            self._torch_entropy_cache[device_key] = table
        return table.index_select(0, hashes)

    def to_bytes(self) -> bytes:
        payload = self.counts.astype("<u4", copy=False).tobytes(order="C")
        header = _canonical_json(
            {
                "schema": "byte_entropy_hash_model/v1",
                "config": asdict(self.config),
                "counts_dtype": "<u4",
                "counts_shape": list(self.counts.shape),
                "counts_sha256": _sha256(payload),
            }
        )
        body = struct.pack("<Q", len(header)) + header + payload
        return _MODEL_MAGIC + hashlib.sha256(body).digest() + body

    @property
    def sha256(self) -> str:
        return _sha256(self.to_bytes())

    @classmethod
    def from_bytes(cls, artifact: bytes) -> "HashedNgramEntropyModel":
        if not artifact.startswith(_MODEL_MAGIC):
            raise ValueError("invalid entropy-estimator artifact magic")
        cursor = len(_MODEL_MAGIC)
        if len(artifact) < cursor + _DIGEST_BYTES + _HEADER_LENGTH_BYTES:
            raise ValueError("truncated entropy-estimator artifact")
        expected_digest = artifact[cursor : cursor + _DIGEST_BYTES]
        cursor += _DIGEST_BYTES
        if hashlib.sha256(artifact[cursor:]).digest() != expected_digest:
            raise ValueError("entropy-estimator artifact hash mismatch")
        header_length = struct.unpack_from("<Q", artifact, cursor)[0]
        cursor += _HEADER_LENGTH_BYTES
        if header_length > _MAX_HEADER_BYTES or len(artifact) < cursor + header_length:
            raise ValueError("invalid entropy-estimator header length")
        try:
            header = json.loads(artifact[cursor : cursor + header_length])
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid entropy-estimator metadata") from error
        cursor += header_length
        if header.get("schema") != "byte_entropy_hash_model/v1":
            raise ValueError("unsupported entropy-estimator artifact schema")
        config = HashedNgramEntropyConfig(**header["config"])
        expected_shape = [config.table_size, config.vocab_size]
        if header.get("counts_dtype") != "<u4" or header.get("counts_shape") != expected_shape:
            raise ValueError("entropy-estimator count metadata mismatch")
        payload = artifact[cursor:]
        expected_bytes = config.table_size * config.vocab_size * 4
        if len(payload) != expected_bytes:
            raise ValueError("entropy-estimator count payload length mismatch")
        if _sha256(payload) != header.get("counts_sha256"):
            raise ValueError("entropy-estimator count payload hash mismatch")
        counts = np.frombuffer(payload, dtype="<u4").reshape(expected_shape).copy()
        return cls(config, counts)


@dataclass(frozen=True)
class EntropyPatchConfig:
    """Causal boundary policy.

    ``threshold`` implements BLT's global-entropy rule ``H(x_i) > theta``.
    ``cumulative`` starts a patch whenever document-cumulative predicted
    entropy strictly crosses another multiple of ``theta``.  The latter keeps
    the boundary computation vectorizable and carries threshold overshoot
    forward rather than discarding information at a boundary.
    """

    schema_version: int = 1
    mode: BoundaryMode = "threshold"
    threshold: float = 3.0
    max_patch_size: int = 8

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported entropy-patch schema version")
        if self.mode not in {"threshold", "cumulative"}:
            raise ValueError(f"unsupported entropy patch mode {self.mode!r}")
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("entropy threshold must be finite and positive")
        if self.max_patch_size <= 0:
            raise ValueError("max_patch_size must be positive")


def entropy_patch_start_mask(
    entropies: np.ndarray | Tensor,
    document_ids: np.ndarray | Tensor,
    config: EntropyPatchConfig,
) -> np.ndarray:
    """Return causal patch-start decisions with a hard maximum patch size."""

    entropy = _numpy_1d(entropies, name="entropies").astype(np.float64, copy=False)
    documents = _validate_document_ids(
        _numpy_1d(document_ids, name="document_ids"), entropy.size
    )
    if bool((~np.isfinite(entropy) | (entropy < 0)).any()):
        raise ValueError("entropies must be finite and nonnegative")
    if entropy.size == 0:
        return np.empty(0, dtype=np.bool_)
    document_starts = np.r_[True, documents[1:] != documents[:-1]]
    if config.mode == "threshold":
        natural = document_starts | (entropy > config.threshold)
    else:
        global_cumulative = np.cumsum(entropy, dtype=np.float64)
        cumulative_before_document = np.where(
            document_starts, global_cumulative - entropy, 0.0
        )
        offsets = np.maximum.accumulate(cumulative_before_document)
        document_cumulative = global_cumulative - offsets
        # A value exactly equal to a threshold has not strictly crossed it.
        strictly_below = np.nextafter(document_cumulative, -np.inf)
        buckets = np.maximum(
            np.floor(strictly_below / config.threshold).astype(np.int64), 0
        )
        previous = np.r_[buckets[0], buckets[:-1]]
        natural = document_starts | (buckets != previous)

    indices = np.arange(entropy.size, dtype=np.int64)
    last_natural = np.maximum.accumulate(np.where(natural, indices, -entropy.size))
    forced = np.remainder(indices - last_natural, config.max_patch_size) == 0
    starts = natural | forced
    starts[document_starts] = True
    return starts


@dataclass(frozen=True)
class PatchLayout:
    """Ragged packed-byte to patch topology, with document-local ordinals."""

    byte_document_ids: np.ndarray
    byte_cu_seqlens: np.ndarray
    byte_to_patch: np.ndarray
    byte_patch_ordinals: np.ndarray
    patch_byte_cu_seqlens: np.ndarray
    patch_document_ids: np.ndarray
    patch_ordinals: np.ndarray
    patch_cu_seqlens: np.ndarray
    patch_starts: np.ndarray
    patch_stops: np.ndarray
    condition_patch_indices: np.ndarray

    def __post_init__(self) -> None:
        byte_count = self.byte_document_ids.size
        patch_count = self.patch_starts.size
        byte_fields = (
            self.byte_document_ids,
            self.byte_to_patch,
            self.byte_patch_ordinals,
            self.condition_patch_indices,
        )
        patch_fields = (
            self.patch_document_ids,
            self.patch_ordinals,
            self.patch_starts,
            self.patch_stops,
        )
        if any(value.ndim != 1 or value.size != byte_count for value in byte_fields):
            raise ValueError("byte-level patch layout fields must align")
        if any(value.ndim != 1 or value.size != patch_count for value in patch_fields):
            raise ValueError("patch-level layout fields must align")
        if self.patch_byte_cu_seqlens.shape != (patch_count + 1,):
            raise ValueError("patch byte offsets must contain one terminal offset")
        if byte_count and (
            self.patch_starts[0] != 0
            or self.patch_stops[-1] != byte_count
            or bool((self.patch_stops <= self.patch_starts).any())
        ):
            raise ValueError("patches must partition the packed byte sequence")

    @property
    def patch_lengths(self) -> np.ndarray:
        return self.patch_stops - self.patch_starts


def patch_layout_from_starts(
    document_ids: np.ndarray | Tensor, starts: np.ndarray | Tensor
) -> PatchLayout:
    documents = _numpy_1d(document_ids, name="document_ids").astype(
        np.int64, copy=False
    )
    documents = _validate_document_ids(documents, documents.size)
    start_mask = _numpy_1d(starts, name="starts").astype(np.bool_, copy=False)
    if start_mask.size != documents.size:
        raise ValueError("patch starts must align with document ids")
    if documents.size == 0:
        empty_i64 = np.empty(0, dtype=np.int64)
        zero_i32 = np.zeros(1, dtype=np.int32)
        return PatchLayout(
            empty_i64,
            zero_i32,
            empty_i64.copy(),
            empty_i64.copy(),
            zero_i32.copy(),
            empty_i64.copy(),
            empty_i64.copy(),
            zero_i32.copy(),
            empty_i64.copy(),
            empty_i64.copy(),
            empty_i64.copy(),
        )
    document_starts = np.r_[True, documents[1:] != documents[:-1]]
    if not bool(start_mask[document_starts].all()):
        raise ValueError("every document must begin a patch")
    patch_starts = np.flatnonzero(start_mask).astype(np.int64)
    patch_stops = np.r_[patch_starts[1:], documents.size].astype(np.int64)
    if bool((documents[patch_starts] != documents[patch_stops - 1]).any()):
        raise ValueError("a patch cannot cross a document boundary")
    byte_to_patch = np.cumsum(start_mask, dtype=np.int64) - 1
    patch_documents = documents[patch_starts]
    patch_document_starts = np.r_[
        True, patch_documents[1:] != patch_documents[:-1]
    ]
    patch_run_starts = np.flatnonzero(patch_document_starts).astype(np.int64)
    patch_run_index = np.cumsum(patch_document_starts, dtype=np.int64) - 1
    patch_ordinals = np.arange(patch_starts.size, dtype=np.int64) - patch_run_starts[
        patch_run_index
    ]
    byte_patch_ordinals = patch_ordinals[byte_to_patch]
    byte_run_starts = np.flatnonzero(document_starts).astype(np.int64)
    byte_cu = np.r_[byte_run_starts, documents.size].astype(np.int32)
    patch_cu = np.r_[patch_run_starts, patch_starts.size].astype(np.int32)
    patch_byte_cu = np.r_[patch_starts, documents.size].astype(np.int32)
    prior_patch = byte_to_patch - 1
    prior_patch[byte_patch_ordinals == 0] = -1
    patch_ends = np.zeros(documents.size, dtype=np.bool_)
    patch_ends[patch_stops - 1] = True
    condition = np.where(patch_ends, byte_to_patch, prior_patch).astype(np.int64)
    return PatchLayout(
        byte_document_ids=np.ascontiguousarray(documents),
        byte_cu_seqlens=byte_cu,
        byte_to_patch=byte_to_patch,
        byte_patch_ordinals=byte_patch_ordinals,
        patch_byte_cu_seqlens=patch_byte_cu,
        patch_document_ids=patch_documents,
        patch_ordinals=patch_ordinals,
        patch_cu_seqlens=patch_cu,
        patch_starts=patch_starts,
        patch_stops=patch_stops,
        condition_patch_indices=condition,
    )


def fixed_stride_patch_layout(
    document_ids: np.ndarray | Tensor, *, stride: int = 4
) -> PatchLayout:
    """Reference layout for a fixed document-local stride, including short tails."""

    if stride <= 0:
        raise ValueError("stride must be positive")
    documents = _numpy_1d(document_ids, name="document_ids").astype(
        np.int64, copy=False
    )
    documents = _validate_document_ids(documents, documents.size)
    if documents.size == 0:
        return patch_layout_from_starts(documents, np.empty(0, dtype=np.bool_))
    document_starts = np.r_[True, documents[1:] != documents[:-1]]
    run_starts = np.maximum.accumulate(
        np.where(document_starts, np.arange(documents.size), 0)
    )
    offsets = np.arange(documents.size, dtype=np.int64) - run_starts
    starts = document_starts | (np.remainder(offsets, stride) == 0)
    return patch_layout_from_starts(documents, starts)


@dataclass(frozen=True)
class EntropyPatchPlan:
    entropies: np.ndarray
    starts: np.ndarray
    layout: PatchLayout


class CausalEntropyPatcher:
    """Serializable estimator plus boundary policy used by training and inference."""

    def __init__(
        self,
        model: HashedNgramEntropyModel,
        config: EntropyPatchConfig = EntropyPatchConfig(),
    ) -> None:
        self.model = model
        self.config = config

    def patch(
        self, ids: np.ndarray | Tensor, document_ids: np.ndarray | Tensor
    ) -> EntropyPatchPlan:
        entropies = self.model.predict_entropies_numpy(ids, document_ids)
        starts = entropy_patch_start_mask(entropies, document_ids, self.config)
        layout = patch_layout_from_starts(document_ids, starts)
        return EntropyPatchPlan(entropies, starts, layout)

    def to_bytes(self) -> bytes:
        model_artifact = self.model.to_bytes()
        header = _canonical_json(
            {
                "schema": "causal_entropy_patcher/v1",
                "patch_config": asdict(self.config),
                "model_bytes": len(model_artifact),
                "model_sha256": _sha256(model_artifact),
            }
        )
        body = struct.pack("<Q", len(header)) + header + model_artifact
        return _PATCHER_MAGIC + hashlib.sha256(body).digest() + body

    @property
    def sha256(self) -> str:
        return _sha256(self.to_bytes())

    @classmethod
    def from_bytes(cls, artifact: bytes) -> "CausalEntropyPatcher":
        if not artifact.startswith(_PATCHER_MAGIC):
            raise ValueError("invalid entropy-patcher artifact magic")
        cursor = len(_PATCHER_MAGIC)
        if len(artifact) < cursor + _DIGEST_BYTES + _HEADER_LENGTH_BYTES:
            raise ValueError("truncated entropy-patcher artifact")
        expected_digest = artifact[cursor : cursor + _DIGEST_BYTES]
        cursor += _DIGEST_BYTES
        if hashlib.sha256(artifact[cursor:]).digest() != expected_digest:
            raise ValueError("entropy-patcher artifact hash mismatch")
        header_length = struct.unpack_from("<Q", artifact, cursor)[0]
        cursor += _HEADER_LENGTH_BYTES
        if header_length > _MAX_HEADER_BYTES or len(artifact) < cursor + header_length:
            raise ValueError("invalid entropy-patcher header length")
        try:
            header = json.loads(artifact[cursor : cursor + header_length])
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid entropy-patcher metadata") from error
        cursor += header_length
        if header.get("schema") != "causal_entropy_patcher/v1":
            raise ValueError("unsupported entropy-patcher artifact schema")
        model_artifact = artifact[cursor:]
        if len(model_artifact) != header.get("model_bytes"):
            raise ValueError("entropy-patcher model length mismatch")
        if _sha256(model_artifact) != header.get("model_sha256"):
            raise ValueError("entropy-patcher model hash mismatch")
        patch_config = EntropyPatchConfig(**header["patch_config"])
        return cls(HashedNgramEntropyModel.from_bytes(model_artifact), patch_config)
