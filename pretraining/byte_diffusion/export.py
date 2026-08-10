"""Deterministic, budget-aware byte-diffusion artifact encoding."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import torch
from torch import Tensor, nn

from .config import ByteDiffusionConfig
from .data import AtomicIdManifest


MAGIC = b"BDI4\x03\x00\x00\x00"
# Groupwise int4 costs 0.53125 bytes/parameter at group size 64. This cap
# leaves roughly 370 KiB inside the 16 MB artifact for metadata and submission
# code while allowing a 12-layer 512-wide global trunk.
PARAMETER_CAP = 29_410_000
ARTIFACT_CAP_BYTES = 16_000_000

TensorEncoding = Literal["group_int4", "group_int8", "float16"]
ArtifactEncodingPolicy = Literal["uniform_int4", "mixed_sensitive"]


@dataclass(frozen=True)
class QuantizedTensor:
    """One encoded tensor and the information required for exact decoding."""

    shape: tuple[int, ...]
    count: int
    group_size: int
    packed: bytes
    scales: bytes
    encoding: TensorEncoding = "group_int4"


@dataclass(frozen=True)
class ArtifactSizeReport:
    parameter_count: int
    artifact_bytes: int
    code_bytes: int
    complete_bytes: int
    headroom_bytes: int
    encoding_counts: Mapping[str, int]
    encoding_parameter_counts: Mapping[str, int]
    encoding_fallbacks: Mapping[str, str]


@dataclass(frozen=True)
class _ArtifactPlan:
    encodings: Mapping[str, TensorEncoding]
    fallbacks: Mapping[str, str]
    metadata: Mapping[str, object]
    artifact_bytes: int


def parameter_table(model: nn.Module) -> dict[str, int]:
    return {name: parameter.numel() for name, parameter in model.named_parameters()}


def enforce_parameter_cap(model: nn.Module, cap: int = PARAMETER_CAP) -> int:
    count = sum(parameter_table(model).values())
    if count > cap:
        raise ValueError(f"model has {count:,} parameters, exceeding {cap:,}")
    return count


def _pack_nibbles(values: Tensor) -> bytes:
    values = values.to(torch.uint8).flatten().cpu()
    if bool((values > 15).any()):
        raise ValueError("nibble value outside [0, 15]")
    if values.numel() % 2:
        values = torch.cat((values, torch.zeros(1, dtype=torch.uint8)))
    packed = values[0::2] | (values[1::2] << 4)
    return packed.numpy().tobytes()


def _unpack_nibbles(payload: bytes, count: int) -> Tensor:
    packed = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    values = torch.empty(packed.numel() * 2, dtype=torch.uint8)
    values[0::2] = packed & 0x0F
    values[1::2] = packed >> 4
    return values[:count]


def _group_layout(count: int, group_size: int) -> tuple[int, int]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    groups = math.ceil(count / group_size)
    return groups, groups * group_size


def quantize_int4(tensor: Tensor, group_size: int = 64) -> QuantizedTensor:
    flat = tensor.detach().float().flatten().cpu()
    count = flat.numel()
    groups, padded_count = _group_layout(count, group_size)
    padded = torch.zeros(padded_count, dtype=torch.float32)
    padded[:count] = flat
    grouped = padded.view(groups, group_size)
    scales = (grouped.abs().amax(1) / 7.0).clamp_min(torch.finfo(torch.float16).tiny)
    quantized = torch.round(grouped / scales[:, None]).clamp(-7, 7).to(torch.int8)
    return QuantizedTensor(
        shape=tuple(tensor.shape),
        count=count,
        group_size=group_size,
        packed=_pack_nibbles((quantized.flatten() + 8).to(torch.uint8)),
        scales=scales.to(torch.float16).numpy().tobytes(),
        encoding="group_int4",
    )


def dequantize_int4(
    quantized: QuantizedTensor, dtype: torch.dtype = torch.float32
) -> Tensor:
    if quantized.encoding != "group_int4":
        raise ValueError(f"expected group_int4, observed {quantized.encoding!r}")
    groups, padded_count = _group_layout(quantized.count, quantized.group_size)
    if len(quantized.packed) != (padded_count + 1) // 2:
        raise ValueError("group_int4 payload length does not match its tensor metadata")
    if len(quantized.scales) != groups * 2:
        raise ValueError("group_int4 scale length does not match its tensor metadata")
    nibbles = _unpack_nibbles(quantized.packed, padded_count)
    values = (nibbles.to(torch.int16) - 8).float()
    scales = torch.frombuffer(bytearray(quantized.scales), dtype=torch.float16).float()
    expanded = scales.repeat_interleave(quantized.group_size)[: quantized.count]
    return (values[: quantized.count] * expanded).reshape(quantized.shape).to(dtype)


def quantize_int8(tensor: Tensor, group_size: int = 64) -> QuantizedTensor:
    """Symmetric groupwise int8 for the collision-sensitive n-gram table."""

    flat = tensor.detach().float().flatten().cpu()
    count = flat.numel()
    groups, padded_count = _group_layout(count, group_size)
    padded = torch.zeros(padded_count, dtype=torch.float32)
    padded[:count] = flat
    grouped = padded.view(groups, group_size)
    scales = (grouped.abs().amax(1) / 127.0).clamp_min(
        torch.finfo(torch.float16).tiny
    )
    quantized = torch.round(grouped / scales[:, None]).clamp(-127, 127).to(torch.int8)
    return QuantizedTensor(
        shape=tuple(tensor.shape),
        count=count,
        group_size=group_size,
        packed=quantized.flatten().numpy().tobytes(),
        scales=scales.to(torch.float16).numpy().tobytes(),
        encoding="group_int8",
    )


def dequantize_int8(
    quantized: QuantizedTensor, dtype: torch.dtype = torch.float32
) -> Tensor:
    if quantized.encoding != "group_int8":
        raise ValueError(f"expected group_int8, observed {quantized.encoding!r}")
    groups, padded_count = _group_layout(quantized.count, quantized.group_size)
    if len(quantized.packed) != padded_count:
        raise ValueError("group_int8 payload length does not match its tensor metadata")
    if len(quantized.scales) != groups * 2:
        raise ValueError("group_int8 scale length does not match its tensor metadata")
    values = torch.frombuffer(bytearray(quantized.packed), dtype=torch.int8)
    scales = torch.frombuffer(bytearray(quantized.scales), dtype=torch.float16).float()
    expanded = scales.repeat_interleave(quantized.group_size)[: quantized.count]
    return (values[: quantized.count].float() * expanded).reshape(quantized.shape).to(dtype)


def encode_float16(tensor: Tensor) -> QuantizedTensor:
    """Store the exact IEEE float16 representation without quantization scales."""

    values = tensor.detach().to(device="cpu", dtype=torch.float16).contiguous()
    return QuantizedTensor(
        shape=tuple(tensor.shape),
        count=tensor.numel(),
        group_size=0,
        packed=values.numpy().tobytes(),
        scales=b"",
        encoding="float16",
    )


def decode_float16(
    encoded: QuantizedTensor, dtype: torch.dtype = torch.float32
) -> Tensor:
    if encoded.encoding != "float16" or encoded.group_size != 0 or encoded.scales:
        raise ValueError("invalid float16 tensor encoding")
    values = torch.frombuffer(bytearray(encoded.packed), dtype=torch.float16)
    if values.numel() != encoded.count:
        raise ValueError("float16 payload length does not match its tensor metadata")
    return values.reshape(encoded.shape).to(dtype)


def decode_tensor(
    encoded: QuantizedTensor, dtype: torch.dtype = torch.float32
) -> Tensor:
    if encoded.encoding == "group_int4":
        return dequantize_int4(encoded, dtype)
    if encoded.encoding == "group_int8":
        return dequantize_int8(encoded, dtype)
    if encoded.encoding == "float16":
        return decode_float16(encoded, dtype)
    raise ValueError(f"unknown tensor encoding {encoded.encoding!r}")


def _encoded_lengths(
    count: int, encoding: TensorEncoding, group_size: int
) -> tuple[int, int]:
    if encoding == "float16":
        return count * 2, 0
    groups, padded_count = _group_layout(count, group_size)
    scale_bytes = groups * 2
    if encoding == "group_int4":
        return (padded_count + 1) // 2, scale_bytes
    if encoding == "group_int8":
        return padded_count, scale_bytes
    raise ValueError(f"unknown tensor encoding {encoding!r}")


def _preferred_encoding(
    name: str, parameter: Tensor, policy: ArtifactEncodingPolicy
) -> TensorEncoding:
    if policy == "uniform_int4":
        return "group_int4"
    if policy != "mixed_sensitive":
        raise ValueError(f"unknown artifact encoding policy {policy!r}")
    if _is_ngram_table(name):
        return "group_int8"
    if name in {"embedding.weight", "output.weight", "mode_embedding.weight"}:
        return "float16"
    # Norm gains, scalar residual/condition gates, and future one-dimensional
    # control vectors are both tiny and unusually sensitive to four-bit error.
    if parameter.ndim < 2:
        return "float16"
    return "group_int4"


def _is_ngram_table(name: str) -> bool:
    return name == "ngrams.table.weight" or (
        name.startswith("ngrams.tables.") and name.endswith(".weight")
    )


def _encoding_summaries(
    parameters: Mapping[str, Tensor], encodings: Mapping[str, TensorEncoding]
) -> tuple[dict[str, int], dict[str, int]]:
    tensor_counts = {encoding: 0 for encoding in ("group_int4", "group_int8", "float16")}
    parameter_counts = dict(tensor_counts)
    for name, parameter in parameters.items():
        encoding = encodings[name]
        tensor_counts[encoding] += 1
        parameter_counts[encoding] += parameter.numel()
    return tensor_counts, parameter_counts


def _metadata_for_plan(
    parameters: Mapping[str, Tensor],
    config: ByteDiffusionConfig,
    *,
    parameter_count: int,
    group_size: int,
    code_bytes: int,
    encoding_policy: ArtifactEncodingPolicy,
    encodings: Mapping[str, TensorEncoding],
    encoding_fallbacks: Mapping[str, str],
    atomic_manifest: AtomicIdManifest | None,
    post_quantization_metrics: Mapping[str, object] | None,
    provenance: Mapping[str, object] | None,
) -> dict[str, object]:
    encoding_counts, encoding_parameter_counts = _encoding_summaries(
        parameters, encodings
    )
    metadata: dict[str, object] = {
        "schema": 3,
        "config": config.to_dict(),
        "parameter_count": parameter_count,
        "group_size": group_size,
        "code_bytes": code_bytes,
        "encoding_policy": encoding_policy,
        "encoding_counts": encoding_counts,
        "encoding_parameter_counts": encoding_parameter_counts,
        "encoding_fallbacks": dict(encoding_fallbacks),
        "atomic_manifest": (
            atomic_manifest.to_dict() if atomic_manifest is not None else None
        ),
        "atomic_manifest_sha256": (
            atomic_manifest.sha256 if atomic_manifest is not None else None
        ),
        "post_quantization_metrics": (
            dict(post_quantization_metrics)
            if post_quantization_metrics is not None
            else None
        ),
        "provenance": dict(provenance) if provenance is not None else None,
        "tensors": {},
    }
    payload_offset = 0
    tensor_metadata: dict[str, object] = {}
    for name, parameter in parameters.items():
        encoding = encodings[name]
        packed_bytes, scale_bytes = _encoded_lengths(
            parameter.numel(), encoding, group_size
        )
        tensor_metadata[name] = {
            "shape": tuple(parameter.shape),
            "count": parameter.numel(),
            "encoding": encoding,
            "group_size": 0 if encoding == "float16" else group_size,
            "packed_offset": payload_offset,
            "packed_bytes": packed_bytes,
            "scale_offset": payload_offset + packed_bytes,
            "scale_bytes": scale_bytes,
        }
        payload_offset += packed_bytes + scale_bytes
    metadata["tensors"] = tensor_metadata
    return metadata


def _metadata_bytes(metadata: Mapping[str, object]) -> bytes:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()


def _planned_artifact_bytes(metadata: Mapping[str, object]) -> int:
    tensors = metadata["tensors"]
    if not isinstance(tensors, dict):
        raise TypeError("artifact tensor metadata must be a dictionary")
    payload_bytes = sum(
        int(info["packed_bytes"]) + int(info["scale_bytes"])
        for info in tensors.values()
    )
    return len(MAGIC) + 8 + len(_metadata_bytes(metadata)) + payload_bytes


def _make_plan(
    model: nn.Module,
    config: ByteDiffusionConfig,
    *,
    group_size: int,
    code_bytes: int,
    encoding_policy: ArtifactEncodingPolicy,
    atomic_manifest: AtomicIdManifest | None,
    post_quantization_metrics: Mapping[str, object] | None,
    provenance: Mapping[str, object] | None,
) -> _ArtifactPlan:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if code_bytes < 0:
        raise ValueError("code_bytes must be non-negative")
    if encoding_policy not in {"uniform_int4", "mixed_sensitive"}:
        raise ValueError(f"unknown artifact encoding policy {encoding_policy!r}")
    parameter_count = enforce_parameter_cap(model)
    parameters = dict(sorted(model.named_parameters()))
    encodings: dict[str, TensorEncoding] = {
        name: _preferred_encoding(name, parameter, encoding_policy)
        for name, parameter in parameters.items()
    }
    fallbacks: dict[str, str] = {}

    def materialize_metadata() -> dict[str, object]:
        return _metadata_for_plan(
            parameters,
            config,
            parameter_count=parameter_count,
            group_size=group_size,
            code_bytes=code_bytes,
            encoding_policy=encoding_policy,
            encodings=encodings,
            encoding_fallbacks=fallbacks,
            atomic_manifest=atomic_manifest,
            post_quantization_metrics=post_quantization_metrics,
            provenance=provenance,
        )

    metadata = materialize_metadata()
    artifact_bytes = _planned_artifact_bytes(metadata)
    if artifact_bytes + code_bytes > ARTIFACT_CAP_BYTES:
        # FP16 sensitive tensors are mandatory for this arm. The n-gram table is
        # the only opportunistic precision allocation and may fall back to int4
        # when a near-capacity topology would otherwise be unexportable.
        int8_ngrams = [
            name
            for name, encoding in encodings.items()
            if _is_ngram_table(name) and encoding == "group_int8"
        ]
        # Downgrade the tables with the largest byte savings first, stopping as
        # soon as the complete artifact fits. This retains the maximum number
        # of int8 per-order tables when their sizes differ.
        def int8_savings(name: str) -> int:
            count = parameters[name].numel()
            int8_bytes = sum(_encoded_lengths(count, "group_int8", group_size))
            int4_bytes = sum(_encoded_lengths(count, "group_int4", group_size))
            return int8_bytes - int4_bytes

        int8_ngrams.sort(
            key=lambda name: (-int8_savings(name), name)
        )
        for name in int8_ngrams:
            encodings[name] = "group_int4"
            fallbacks[name] = "group_int8_to_group_int4_artifact_budget"
            metadata = materialize_metadata()
            artifact_bytes = _planned_artifact_bytes(metadata)
            if artifact_bytes + code_bytes <= ARTIFACT_CAP_BYTES:
                break
    return _ArtifactPlan(dict(encodings), dict(fallbacks), metadata, artifact_bytes)


def artifact_size_report(
    model: nn.Module,
    config: ByteDiffusionConfig,
    *,
    group_size: int = 64,
    code_bytes: int = 0,
    encoding_policy: ArtifactEncodingPolicy = "uniform_int4",
    atomic_manifest: AtomicIdManifest | None = None,
    post_quantization_metrics: Mapping[str, object] | None = None,
    provenance: Mapping[str, object] | None = None,
) -> ArtifactSizeReport:
    """Return exact serialized size without reading or quantizing tensor values."""

    plan = _make_plan(
        model,
        config,
        group_size=group_size,
        code_bytes=code_bytes,
        encoding_policy=encoding_policy,
        atomic_manifest=atomic_manifest,
        post_quantization_metrics=post_quantization_metrics,
        provenance=provenance,
    )
    complete_bytes = plan.artifact_bytes + code_bytes
    return ArtifactSizeReport(
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        artifact_bytes=plan.artifact_bytes,
        code_bytes=code_bytes,
        complete_bytes=complete_bytes,
        headroom_bytes=ARTIFACT_CAP_BYTES - complete_bytes,
        encoding_counts=plan.metadata["encoding_counts"],  # type: ignore[arg-type]
        encoding_parameter_counts=plan.metadata[  # type: ignore[arg-type]
            "encoding_parameter_counts"
        ],
        encoding_fallbacks=plan.fallbacks,
    )


def _validate_final_evaluation(
    post_quantization_metrics: Mapping[str, object] | None,
    provenance: Mapping[str, object] | None,
    *,
    encoding_policy: ArtifactEncodingPolicy,
    group_size: int,
    encoding_fallbacks: Mapping[str, str],
) -> None:
    if post_quantization_metrics is None:
        raise ValueError("final artifacts require post-quantization evaluation")
    if provenance is None:
        raise ValueError("final artifacts require checkpoint/data provenance")
    bpb = post_quantization_metrics.get("bpb")
    literal_bytes = post_quantization_metrics.get("literal_bytes")
    if (
        not isinstance(bpb, (float, int))
        or not math.isfinite(float(bpb))
        or not isinstance(literal_bytes, int)
        or literal_bytes <= 0
    ):
        raise ValueError(
            "final evaluation needs finite BPB and a positive literal-byte count"
        )
    expected_plan = {
        "policy": encoding_policy,
        "group_size": group_size,
        "fallbacks": dict(encoding_fallbacks),
    }
    if post_quantization_metrics.get("encoding_plan") != expected_plan:
        raise ValueError(
            "final evaluation encoding_plan does not match the serialized artifact"
        )


def _encode_parameter(
    parameter: Tensor, encoding: TensorEncoding, group_size: int
) -> QuantizedTensor:
    if encoding == "group_int4":
        return quantize_int4(parameter, group_size)
    if encoding == "group_int8":
        return quantize_int8(parameter, group_size)
    if encoding == "float16":
        return encode_float16(parameter)
    raise ValueError(f"unknown tensor encoding {encoding!r}")


def build_artifact(
    model: nn.Module,
    config: ByteDiffusionConfig,
    *,
    group_size: int = 64,
    code_bytes: int = 0,
    encoding_policy: ArtifactEncodingPolicy = "uniform_int4",
    atomic_manifest: AtomicIdManifest | None = None,
    post_quantization_metrics: Mapping[str, object] | None = None,
    provenance: Mapping[str, object] | None = None,
    require_evaluation: bool = False,
) -> bytes:
    """Build a deterministic, metadata-inclusive weight artifact.

    ``uniform_int4`` remains the default until a post-quantization BPB ablation
    promotes the explicit ``mixed_sensitive`` arm.
    """

    plan = _make_plan(
        model,
        config,
        group_size=group_size,
        code_bytes=code_bytes,
        encoding_policy=encoding_policy,
        atomic_manifest=atomic_manifest,
        post_quantization_metrics=post_quantization_metrics,
        provenance=provenance,
    )
    if require_evaluation:
        _validate_final_evaluation(
            post_quantization_metrics,
            provenance,
            encoding_policy=encoding_policy,
            group_size=group_size,
            encoding_fallbacks=plan.fallbacks,
        )
    complete_size = plan.artifact_bytes + code_bytes
    if complete_size > ARTIFACT_CAP_BYTES:
        raise ValueError(
            f"complete artifact uses {complete_size:,} bytes, exceeding "
            f"{ARTIFACT_CAP_BYTES:,}"
        )
    parameters = dict(sorted(model.named_parameters()))
    encoded = {
        name: _encode_parameter(parameter, plan.encodings[name], group_size)
        for name, parameter in parameters.items()
    }
    payload = bytearray()
    for name in parameters:
        tensor = encoded[name]
        info = plan.metadata["tensors"][name]  # type: ignore[index]
        if len(payload) != int(info["packed_offset"]):
            raise AssertionError("artifact payload offset drifted from its size plan")
        if len(tensor.packed) != int(info["packed_bytes"]):
            raise AssertionError("encoded payload length drifted from its size plan")
        if len(tensor.scales) != int(info["scale_bytes"]):
            raise AssertionError("encoded scale length drifted from its size plan")
        payload.extend(tensor.packed)
        payload.extend(tensor.scales)
    metadata_bytes = _metadata_bytes(plan.metadata)
    artifact = MAGIC + struct.pack("<Q", len(metadata_bytes)) + metadata_bytes + bytes(payload)
    if len(artifact) != plan.artifact_bytes:
        raise AssertionError("serialized artifact length drifted from its exact size plan")
    return artifact


def _parse_tensor(
    name: str, info: Mapping[str, object], payload: bytes
) -> QuantizedTensor:
    try:
        encoding = str(info["encoding"])
        if encoding not in {"group_int4", "group_int8", "float16"}:
            raise ValueError(f"unknown tensor encoding {encoding!r}")
        shape = tuple(int(value) for value in info["shape"])  # type: ignore[union-attr]
        count = int(info["count"])
        group_size = int(info["group_size"])
        packed_start = int(info["packed_offset"])
        packed_bytes = int(info["packed_bytes"])
        scale_start = int(info["scale_offset"])
        scale_bytes = int(info["scale_bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid tensor metadata for {name!r}") from error
    if count < 0 or any(dimension < 0 for dimension in shape):
        raise ValueError(f"tensor {name!r} has a negative shape or element count")
    if count != math.prod(shape):
        raise ValueError(f"tensor {name!r} count does not match its shape")
    if encoding == "float16":
        if group_size != 0:
            raise ValueError(f"float16 tensor {name!r} must have group_size zero")
        expected_packed, expected_scales = _encoded_lengths(count, encoding, 1)
    else:
        if group_size <= 0:
            raise ValueError(
                f"quantized tensor {name!r} must have a positive group_size"
            )
        expected_packed, expected_scales = _encoded_lengths(
            count, encoding, group_size
        )
    if packed_bytes != expected_packed or scale_bytes != expected_scales:
        raise ValueError(f"tensor {name!r} payload lengths do not match its encoding")
    if scale_start != packed_start + packed_bytes:
        raise ValueError(f"tensor {name!r} scale payload is not contiguous")
    packed_end = packed_start + packed_bytes
    scale_end = scale_start + scale_bytes
    if packed_start < 0 or scale_end > len(payload):
        raise ValueError(f"tensor {name!r} payload lies outside the artifact")
    return QuantizedTensor(
        shape=shape,
        count=count,
        group_size=group_size,
        packed=payload[packed_start:packed_end],
        scales=payload[scale_start:scale_end],
        encoding=encoding,  # type: ignore[arg-type]
    )


def parse_artifact(
    artifact: bytes,
) -> tuple[dict[str, object], dict[str, QuantizedTensor]]:
    if artifact[: len(MAGIC)] != MAGIC:
        raise ValueError("invalid byte-diffusion artifact magic")
    header_start = len(MAGIC)
    if len(artifact) < header_start + 8:
        raise ValueError("truncated byte-diffusion artifact header")
    metadata_size = struct.unpack("<Q", artifact[header_start : header_start + 8])[0]
    metadata_start = header_start + 8
    metadata_end = metadata_start + metadata_size
    if metadata_end > len(artifact):
        raise ValueError("truncated byte-diffusion artifact metadata")
    try:
        metadata = json.loads(artifact[metadata_start:metadata_end])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid byte-diffusion artifact metadata") from error
    if not isinstance(metadata, dict):
        raise ValueError("artifact metadata must be a JSON object")
    if metadata.get("schema") != 3:
        raise ValueError("unsupported byte-diffusion artifact schema")
    tensor_metadata = metadata.get("tensors")
    if not isinstance(tensor_metadata, dict):
        raise ValueError("artifact omitted tensor metadata")
    payload = artifact[metadata_end:]
    tensors = {
        name: _parse_tensor(name, info, payload)
        for name, info in tensor_metadata.items()
    }
    ordered_ranges = sorted(
        (
            int(info["packed_offset"]),
            int(info["scale_offset"]) + int(info["scale_bytes"]),
            name,
        )
        for name, info in tensor_metadata.items()
    )
    cursor = 0
    for start, stop, name in ordered_ranges:
        if start != cursor:
            raise ValueError(f"tensor {name!r} leaves a gap or overlap in the payload")
        cursor = stop
    if cursor != len(payload):
        raise ValueError("artifact contains unclaimed tensor payload bytes")
    return metadata, tensors


def load_artifact(
    model: nn.Module, artifact: bytes, *, strict: bool = True
) -> dict[str, object]:
    metadata, tensors = parse_artifact(artifact)
    expected = dict(model.named_parameters())
    if strict and set(tensors) != set(expected):
        missing = sorted(set(expected) - set(tensors))
        extra = sorted(set(tensors) - set(expected))
        raise ValueError(f"artifact tensor mismatch: missing={missing}, extra={extra}")
    with torch.no_grad():
        for name, encoded in tensors.items():
            if name not in expected:
                continue
            restored = decode_tensor(encoded, expected[name].dtype)
            if restored.shape != expected[name].shape:
                raise ValueError(f"shape mismatch for {name}")
            expected[name].copy_(restored.to(expected[name].device))
    return metadata


def write_artifact(path: str | Path, artifact: bytes) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(artifact)
    return hashlib.sha256(artifact).hexdigest()
