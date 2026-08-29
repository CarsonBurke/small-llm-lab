"""Deterministic, budget-aware byte-diffusion artifact encoding."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping

import torch
from torch import Tensor, nn

from .config import ByteDiffusionConfig
from .data import AtomicIdManifest
from .patching import CausalEntropyPatcher


MAGIC = b"BDI4\x03\x00\x00\x00"
# Groupwise int4 costs 0.53125 bytes/parameter at group size 64. This cap
# leaves roughly 370 KiB inside the 16 MB artifact for metadata and submission
# code while allowing a 12-layer 512-wide global trunk.
PARAMETER_CAP = 29_410_000
ARTIFACT_CAP_BYTES = 16_000_000
MAX_ENTROPY_PATCHER_RAW_BYTES = 8 * 1024 * 1024
ENTROPY_PATCHER_ENCODING = "zlib"

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
    entropy_patcher_payload: bytes | None


@dataclass(frozen=True)
class _EncodedEntropyPatcher:
    raw: bytes
    compressed: bytes
    metadata: Mapping[str, object]


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
    entropy_patcher_metadata: Mapping[str, object] | None,
    duo_serving_metadata: Mapping[str, object] | None,
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
    if entropy_patcher_metadata is not None:
        metadata["entropy_patcher"] = {
            **entropy_patcher_metadata,
            "payload_offset": payload_offset,
        }
    if duo_serving_metadata is not None:
        metadata["duo_serving"] = dict(duo_serving_metadata)
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
    patcher = metadata.get("entropy_patcher")
    if patcher is not None:
        if not isinstance(patcher, Mapping):
            raise TypeError("artifact entropy patcher metadata must be an object")
        payload_bytes += int(patcher["payload_bytes"])
    return len(MAGIC) + 8 + len(_metadata_bytes(metadata)) + payload_bytes


def _encoded_entropy_patcher(
    artifact: bytes | None,
) -> _EncodedEntropyPatcher | None:
    if artifact is None:
        return None
    if not artifact:
        raise ValueError("embedded entropy patcher artifact cannot be empty")
    if len(artifact) > MAX_ENTROPY_PATCHER_RAW_BYTES:
        raise ValueError(
            "embedded entropy patcher exceeds the bounded decompression limit"
        )
    try:
        patcher = CausalEntropyPatcher.from_bytes(artifact)
    except (TypeError, ValueError) as error:
        raise ValueError("embedded entropy patcher artifact is invalid") from error
    compressed = zlib.compress(artifact, level=9)
    metadata = {
        "schema": "causal_entropy_patcher/v2",
        "encoding": ENTROPY_PATCHER_ENCODING,
        "raw_bytes": len(artifact),
        "compressed_bytes": len(compressed),
        "payload_bytes": len(compressed),
        "raw_sha256": hashlib.sha256(artifact).hexdigest(),
        # Retain the generic provenance spelling used by final-export checks.
        "sha256": hashlib.sha256(artifact).hexdigest(),
        "boundary_config": asdict(patcher.config),
        "entropy_model_config": asdict(patcher.model.config),
        "max_patch_size": patcher.config.max_patch_size,
    }
    return _EncodedEntropyPatcher(artifact, compressed, metadata)


def _duo_serving_metadata(
    model: nn.Module,
    config: ByteDiffusionConfig,
    patcher: _EncodedEntropyPatcher | None,
) -> dict[str, object] | None:
    schedule_eps = getattr(model, "schedule_eps", None)
    if schedule_eps is None:
        return None
    if (
        not isinstance(schedule_eps, (float, int))
        or not math.isfinite(float(schedule_eps))
        or not 0.0 < float(schedule_eps) < 0.5
    ):
        raise ValueError("Duo artifact schedule_eps must lie in (0, 0.5)")
    policy = config.duo_clean_patching
    if policy == "causal_entropy_v1" and patcher is None:
        raise ValueError("entropy-patched Duo artifacts require an embedded patcher")
    if policy == "fixed_stride_v1" and patcher is not None:
        raise ValueError("fixed-stride Duo artifacts cannot embed an entropy patcher")
    return {
        "schema": "byte_duo_serving/v1",
        "schedule_eps": float(schedule_eps),
        "patching_policy": policy,
        "entropy_patcher_sha256": (
            patcher.metadata["raw_sha256"] if patcher is not None else None
        ),
        "entropy_patcher_max_patch_size": (
            patcher.metadata["max_patch_size"] if patcher is not None else None
        ),
    }


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
    entropy_patcher: bytes | None,
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
    encoded_entropy_patcher = _encoded_entropy_patcher(entropy_patcher)
    entropy_patcher_metadata = (
        encoded_entropy_patcher.metadata
        if encoded_entropy_patcher is not None
        else None
    )
    duo_serving_metadata = _duo_serving_metadata(
        model, config, encoded_entropy_patcher
    )

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
            entropy_patcher_metadata=entropy_patcher_metadata,
            duo_serving_metadata=duo_serving_metadata,
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
    return _ArtifactPlan(
        dict(encodings),
        dict(fallbacks),
        metadata,
        artifact_bytes,
        (
            encoded_entropy_patcher.compressed
            if encoded_entropy_patcher is not None
            else None
        ),
    )


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
    entropy_patcher: bytes | None = None,
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
        entropy_patcher=entropy_patcher,
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
    entropy_patcher_metadata: Mapping[str, object] | None,
) -> None:
    if post_quantization_metrics is None:
        raise ValueError("final artifacts require post-quantization evaluation")
    if provenance is None:
        raise ValueError("final artifacts require checkpoint/data provenance")
    patching_policy = provenance.get("patching_policy")
    if patching_policy == "causal_entropy_v1":
        if entropy_patcher_metadata is None:
            raise ValueError(
                "final entropy artifacts require an embedded authenticated patcher"
            )
        if provenance.get("entropy_patcher_sha256") != (
            entropy_patcher_metadata.get("sha256")
        ):
            raise ValueError(
                "final entropy artifact provenance disagrees with its patcher"
            )
    elif patching_policy == "fixed_stride_v1" and (
        entropy_patcher_metadata is not None
    ):
        raise ValueError("fixed-stride artifacts cannot embed an entropy patcher")
    if provenance.get("architecture") == "byte_duo_uniform_state_diffusion":
        proxy = post_quantization_metrics.get(
            "conditional_canvas_nelbo_nats_per_atom"
        )
        targets = post_quantization_metrics.get("targets")
        if (
            not isinstance(proxy, (float, int))
            or not math.isfinite(float(proxy))
            or not isinstance(targets, int)
            or targets <= 0
        ):
            raise ValueError(
                "final Byte-Duo evaluation needs a finite conditional-canvas "
                "NELBO proxy and positive target count"
            )
        delta = post_quantization_metrics.get("quantization_delta_bits_per_atom")
        delta_limit = post_quantization_metrics.get(
            "max_quantization_delta_bits_per_atom"
        )
        cached_smoke = post_quantization_metrics.get("cached_inference_smoke")
        if (
            not isinstance(delta, (float, int))
            or not math.isfinite(float(delta))
            or not isinstance(delta_limit, (float, int))
            or not math.isfinite(float(delta_limit))
            or float(delta_limit) < 0
            or float(delta) > float(delta_limit)
        ):
            raise ValueError(
                "final Byte-Duo evaluation must pass its finite quantization-"
                "degradation threshold"
            )
        if (
            not isinstance(cached_smoke, Mapping)
            or cached_smoke.get("finite") is not True
            or not isinstance(
                cached_smoke.get("full_vs_cached_max_abs_error"), (float, int)
            )
            or not math.isfinite(
                float(cached_smoke["full_vs_cached_max_abs_error"])
            )
            or float(cached_smoke["full_vs_cached_max_abs_error"]) > 1e-5
        ):
            raise ValueError(
                "final Byte-Duo evaluation must pass cached inference parity"
            )
    else:
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
    entropy_patcher: bytes | None = None,
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
        entropy_patcher=entropy_patcher,
    )
    if require_evaluation:
        _validate_final_evaluation(
            post_quantization_metrics,
            provenance,
            encoding_policy=encoding_policy,
            group_size=group_size,
            encoding_fallbacks=plan.fallbacks,
            entropy_patcher_metadata=(
                plan.metadata.get("entropy_patcher")  # type: ignore[arg-type]
            ),
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
    if plan.entropy_patcher_payload is not None:
        patcher_info = plan.metadata.get("entropy_patcher")
        if not isinstance(patcher_info, Mapping):
            raise AssertionError("entropy patcher disappeared from the size plan")
        if len(payload) != int(patcher_info["payload_offset"]):
            raise AssertionError("entropy patcher offset drifted from its size plan")
        if len(plan.entropy_patcher_payload) != int(patcher_info["payload_bytes"]):
            raise AssertionError("entropy patcher length drifted from its size plan")
        payload.extend(plan.entropy_patcher_payload)
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


def _artifact_metadata_and_payload(
    artifact: bytes,
) -> tuple[dict[str, object], bytes]:
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
    return metadata, artifact[metadata_end:]


def _validate_embedded_entropy_patcher(
    metadata: Mapping[str, object], payload: bytes, cursor: int
) -> int:
    info = metadata.get("entropy_patcher")
    if info is None:
        return cursor
    if not isinstance(info, Mapping):
        raise ValueError("artifact entropy patcher metadata must be an object")
    try:
        schema = str(info["schema"])
        encoding = str(info["encoding"])
        offset = int(info["payload_offset"])
        payload_bytes = int(info["payload_bytes"])
        compressed_bytes = int(info["compressed_bytes"])
        raw_bytes = int(info["raw_bytes"])
        expected_sha256 = str(info["raw_sha256"])
        compatibility_sha256 = str(info["sha256"])
        max_patch_size = int(info["max_patch_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid embedded entropy patcher metadata") from error
    if schema != "causal_entropy_patcher/v2":
        raise ValueError("unsupported embedded entropy patcher schema")
    if encoding != ENTROPY_PATCHER_ENCODING:
        raise ValueError("unsupported embedded entropy patcher encoding")
    if payload_bytes != compressed_bytes or compressed_bytes <= 0:
        raise ValueError("embedded entropy patcher compressed size mismatch")
    if raw_bytes <= 0 or raw_bytes > MAX_ENTROPY_PATCHER_RAW_BYTES:
        raise ValueError("embedded entropy patcher raw size exceeds its bound")
    if compatibility_sha256 != expected_sha256:
        raise ValueError("embedded entropy patcher sha256 metadata disagrees")
    if offset != cursor:
        raise ValueError("embedded entropy patcher leaves a gap or overlap")
    stop = offset + payload_bytes
    if stop > len(payload):
        raise ValueError("embedded entropy patcher lies outside the artifact")
    artifact = _strict_zlib_decompress(
        payload[offset:stop], expected_size=raw_bytes
    )
    if hashlib.sha256(artifact).hexdigest() != expected_sha256:
        raise ValueError("embedded entropy patcher sha256 mismatch")
    try:
        patcher = CausalEntropyPatcher.from_bytes(artifact)
    except (TypeError, ValueError) as error:
        raise ValueError("embedded entropy patcher artifact is invalid") from error
    if max_patch_size != patcher.config.max_patch_size:
        raise ValueError("embedded entropy patcher maximum size metadata mismatch")
    if info.get("boundary_config") != asdict(patcher.config):
        raise ValueError("embedded entropy patcher boundary metadata mismatch")
    if info.get("entropy_model_config") != asdict(patcher.model.config):
        raise ValueError("embedded entropy model metadata mismatch")
    return stop


def _strict_zlib_decompress(payload: bytes, *, expected_size: int) -> bytes:
    """Decompress one zlib stream without permitting bombs or hidden trailers."""

    if not 0 < expected_size <= MAX_ENTROPY_PATCHER_RAW_BYTES:
        raise ValueError("embedded entropy patcher raw size exceeds its bound")
    decoder = zlib.decompressobj()
    try:
        decoded = decoder.decompress(payload, expected_size + 1)
        if len(decoded) > expected_size or decoder.unconsumed_tail:
            raise ValueError("embedded entropy patcher exceeds its declared raw size")
        remaining = expected_size + 1 - len(decoded)
        decoded += decoder.flush(remaining)
    except zlib.error as error:
        raise ValueError("embedded entropy patcher zlib payload is invalid") from error
    if len(decoded) != expected_size:
        raise ValueError("embedded entropy patcher raw size mismatch")
    if not decoder.eof:
        raise ValueError("embedded entropy patcher zlib stream is truncated")
    if decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("embedded entropy patcher contains trailing compressed data")
    return decoded


def duo_serving_contract(metadata: Mapping[str, object]) -> dict[str, object]:
    """Validate and return the artifact-only Byte-Duo runtime contract."""

    info = metadata.get("duo_serving")
    if not isinstance(info, Mapping):
        raise ValueError("artifact omitted its Byte-Duo serving contract")
    try:
        schema = str(info["schema"])
        schedule_eps = float(info["schedule_eps"])
        patching_policy = str(info["patching_policy"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid Byte-Duo serving contract") from error
    if schema != "byte_duo_serving/v1":
        raise ValueError("unsupported Byte-Duo serving contract schema")
    if not math.isfinite(schedule_eps) or not 0.0 < schedule_eps < 0.5:
        raise ValueError("Byte-Duo serving schedule_eps must lie in (0, 0.5)")
    if patching_policy not in {"fixed_stride_v1", "causal_entropy_v1"}:
        raise ValueError("unknown Byte-Duo serving patching policy")
    config = metadata.get("config")
    if not isinstance(config, Mapping) or config.get("duo_clean_patching") != (
        patching_policy
    ):
        raise ValueError("Byte-Duo serving policy disagrees with model config")
    patcher = metadata.get("entropy_patcher")
    if patching_policy == "fixed_stride_v1":
        if patcher is not None:
            raise ValueError("fixed-stride Byte-Duo artifact embeds an entropy patcher")
        if info.get("entropy_patcher_sha256") is not None:
            raise ValueError("fixed-stride Byte-Duo artifact claims patcher provenance")
        if info.get("entropy_patcher_max_patch_size") is not None:
            raise ValueError("fixed-stride Byte-Duo artifact claims a patch size")
    else:
        if not isinstance(patcher, Mapping):
            raise ValueError("entropy Byte-Duo artifact omitted its patcher")
        if info.get("entropy_patcher_sha256") != patcher.get("raw_sha256"):
            raise ValueError("Byte-Duo serving patcher sha256 mismatch")
        if info.get("entropy_patcher_max_patch_size") != patcher.get(
            "max_patch_size"
        ):
            raise ValueError("Byte-Duo serving patch size mismatch")
    return dict(info)


def parse_artifact(
    artifact: bytes,
) -> tuple[dict[str, object], dict[str, QuantizedTensor]]:
    metadata, payload = _artifact_metadata_and_payload(artifact)
    tensor_metadata = metadata.get("tensors")
    if not isinstance(tensor_metadata, dict):
        raise ValueError("artifact omitted tensor metadata")
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
    cursor = _validate_embedded_entropy_patcher(metadata, payload, cursor)
    if cursor != len(payload):
        raise ValueError("artifact contains unclaimed payload bytes")
    if "duo_serving" in metadata:
        duo_serving_contract(metadata)
    return metadata, tensors


def load_embedded_entropy_patcher(
    artifact: bytes,
) -> CausalEntropyPatcher | None:
    """Return the hash-validated self-contained patcher, when present."""

    metadata, _ = parse_artifact(artifact)
    info = metadata.get("entropy_patcher")
    if info is None:
        return None
    if not isinstance(info, Mapping):
        raise AssertionError("validated entropy patcher metadata disappeared")
    _, payload = _artifact_metadata_and_payload(artifact)
    offset = int(info["payload_offset"])
    stop = offset + int(info["payload_bytes"])
    raw = _strict_zlib_decompress(
        payload[offset:stop], expected_size=int(info["raw_bytes"])
    )
    return CausalEntropyPatcher.from_bytes(raw)


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
