"""Deterministic signed-int4 packing and complete model artifact accounting."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn

from .config import ByteDiffusionConfig
from .data import AtomicIdManifest


MAGIC = b"BDI4\x01\x00\x00\x00"
PARAMETER_CAP = 24_800_000
ARTIFACT_CAP_BYTES = 16_000_000


@dataclass(frozen=True)
class QuantizedTensor:
    shape: tuple[int, ...]
    count: int
    group_size: int
    packed: bytes
    scales: bytes


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


def quantize_int4(tensor: Tensor, group_size: int = 64) -> QuantizedTensor:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    flat = tensor.detach().float().flatten().cpu()
    count = flat.numel()
    groups = math.ceil(count / group_size)
    padded = torch.zeros(groups * group_size, dtype=torch.float32)
    padded[:count] = flat
    grouped = padded.view(groups, group_size)
    scales = (grouped.abs().amax(1) / 7.0).clamp_min(torch.finfo(torch.float16).tiny)
    quantized = torch.round(grouped / scales[:, None]).clamp(-7, 7).to(torch.int8)
    nibbles = (quantized.flatten() + 8).to(torch.uint8)
    return QuantizedTensor(
        shape=tuple(tensor.shape),
        count=count,
        group_size=group_size,
        packed=_pack_nibbles(nibbles),
        scales=scales.to(torch.float16).numpy().tobytes(),
    )


def dequantize_int4(quantized: QuantizedTensor, dtype: torch.dtype = torch.float32) -> Tensor:
    nibbles = _unpack_nibbles(quantized.packed, quantized.count)
    values = (nibbles.to(torch.int16) - 8).float()
    scales = torch.frombuffer(bytearray(quantized.scales), dtype=torch.float16).float()
    expanded = scales.repeat_interleave(quantized.group_size)[: quantized.count]
    return (values * expanded).reshape(quantized.shape).to(dtype)


def build_artifact(
    model: nn.Module,
    config: ByteDiffusionConfig,
    *,
    group_size: int = 64,
    code_bytes: int = 0,
    atomic_manifest: AtomicIdManifest | None = None,
    post_quantization_metrics: Mapping[str, object] | None = None,
    provenance: Mapping[str, object] | None = None,
    require_evaluation: bool = False,
) -> bytes:
    """Build a deterministic, metadata-inclusive weight artifact."""

    parameter_count = enforce_parameter_cap(model)
    if require_evaluation:
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
    quantized = {
        name: quantize_int4(parameter, group_size)
        for name, parameter in sorted(model.named_parameters())
    }
    metadata: dict[str, object] = {
        "schema": 1,
        "config": config.to_dict(),
        "parameter_count": parameter_count,
        "group_size": group_size,
        "code_bytes": code_bytes,
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
    payload = bytearray()
    tensor_metadata: dict[str, object] = {}
    for name, tensor in quantized.items():
        packed_offset = len(payload)
        payload.extend(tensor.packed)
        scale_offset = len(payload)
        payload.extend(tensor.scales)
        tensor_metadata[name] = {
            "shape": tensor.shape,
            "count": tensor.count,
            "packed_offset": packed_offset,
            "packed_bytes": len(tensor.packed),
            "scale_offset": scale_offset,
            "scale_bytes": len(tensor.scales),
        }
    metadata["tensors"] = tensor_metadata
    metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    artifact = MAGIC + struct.pack("<Q", len(metadata_bytes)) + metadata_bytes + bytes(payload)
    complete_size = len(artifact) + code_bytes
    if complete_size > ARTIFACT_CAP_BYTES:
        raise ValueError(
            f"complete artifact uses {complete_size:,} bytes, exceeding {ARTIFACT_CAP_BYTES:,}"
        )
    return artifact


def parse_artifact(artifact: bytes) -> tuple[dict[str, object], dict[str, QuantizedTensor]]:
    if artifact[: len(MAGIC)] != MAGIC:
        raise ValueError("invalid byte-diffusion artifact magic")
    header_start = len(MAGIC)
    metadata_size = struct.unpack("<Q", artifact[header_start : header_start + 8])[0]
    metadata_start = header_start + 8
    metadata_end = metadata_start + metadata_size
    metadata = json.loads(artifact[metadata_start:metadata_end])
    payload = artifact[metadata_end:]
    group_size = int(metadata["group_size"])
    tensors: dict[str, QuantizedTensor] = {}
    for name, info in metadata["tensors"].items():
        packed_start = int(info["packed_offset"])
        packed_end = packed_start + int(info["packed_bytes"])
        scale_start = int(info["scale_offset"])
        scale_end = scale_start + int(info["scale_bytes"])
        tensors[name] = QuantizedTensor(
            shape=tuple(info["shape"]),
            count=int(info["count"]),
            group_size=group_size,
            packed=payload[packed_start:packed_end],
            scales=payload[scale_start:scale_end],
        )
    return metadata, tensors


def load_artifact(model: nn.Module, artifact: bytes, *, strict: bool = True) -> dict[str, object]:
    metadata, tensors = parse_artifact(artifact)
    expected = dict(model.named_parameters())
    if strict and set(tensors) != set(expected):
        missing = sorted(set(expected) - set(tensors))
        extra = sorted(set(tensors) - set(expected))
        raise ValueError(f"artifact tensor mismatch: missing={missing}, extra={extra}")
    with torch.no_grad():
        for name, quantized in tensors.items():
            if name not in expected:
                continue
            restored = dequantize_int4(quantized, expected[name].dtype)
            if restored.shape != expected[name].shape:
                raise ValueError(f"shape mismatch for {name}")
            expected[name].copy_(restored.to(expected[name].device))
    return metadata


def write_artifact(path: str | Path, artifact: bytes) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(artifact)
    return hashlib.sha256(artifact).hexdigest()
