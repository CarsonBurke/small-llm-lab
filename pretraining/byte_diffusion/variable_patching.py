"""Manifest authentication and packed layouts for causal variable patches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .patching import CausalEntropyPatcher


PATCHING_POLICY_SCHEMA = "byte_diffusion_patching_policy/v1"
FIXED_DATASET_SCHEMA = "byte_diffusion_dataset/v5"
ENTROPY_DATASET_SCHEMA = "byte_diffusion_dataset/v6"


@dataclass(frozen=True)
class DatasetPatchingSpec:
    """Authenticated byte-to-patch policy from a dataset manifest."""

    name: str
    patch_stride: int | None
    max_patch_size: int
    artifact_sha256: str | None = None

    @property
    def variable(self) -> bool:
        return self.patch_stride is None


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_dataset_patching_spec(
    directory: Path, manifest: Mapping[str, Any]
) -> DatasetPatchingSpec:
    """Validate fixed-v5 or entropy-v6 patching, including the patcher bytes."""

    schema = manifest.get("schema")
    packing = manifest.get("packing")
    if not isinstance(packing, Mapping):
        raise ValueError("dataset manifest omitted its packing contract")
    policy = manifest.get("patching")
    if schema == FIXED_DATASET_SCHEMA:
        stride = _positive_int(packing.get("patch_stride"), name="patch_stride")
        if policy is not None:
            if not isinstance(policy, Mapping):
                raise ValueError("fixed patching policy must be an object")
            if (
                policy.get("schema") != PATCHING_POLICY_SCHEMA
                or policy.get("name") != "fixed_stride_v1"
            ):
                raise ValueError("v5 requires the fixed_stride_v1 patching policy")
            if _positive_int(
                policy.get("max_patch_size"), name="fixed max_patch_size"
            ) != stride:
                raise ValueError("fixed patching policy disagrees with packing stride")
        return DatasetPatchingSpec("fixed_stride_v1", stride, stride)
    if schema != ENTROPY_DATASET_SCHEMA:
        raise ValueError(
            "training requires a mapped byte-native v5 or entropy-patched v6 dataset"
        )
    if packing.get("patch_stride") is not None:
        raise ValueError("entropy-patched v6 must declare packing.patch_stride=null")
    if not isinstance(policy, Mapping):
        raise ValueError("entropy-patched v6 omitted its patching policy")
    if (
        policy.get("schema") != PATCHING_POLICY_SCHEMA
        or policy.get("name") != "causal_entropy_v1"
    ):
        raise ValueError("v6 requires the causal_entropy_v1 patching policy")
    max_patch_size = _positive_int(
        policy.get("max_patch_size"), name="entropy max_patch_size"
    )
    boundary = policy.get("boundary_config")
    if not isinstance(boundary, Mapping) or boundary.get("max_patch_size") != max_patch_size:
        raise ValueError("entropy boundary config disagrees with max_patch_size")
    artifact = policy.get("patcher_artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("entropy patching policy omitted its patcher artifact")
    relative = Path(str(artifact.get("path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("entropy patcher artifact path escapes the dataset")
    path = directory / relative
    payload = path.read_bytes()
    expected_bytes = _positive_int(artifact.get("bytes"), name="patcher bytes")
    if len(payload) != expected_bytes:
        raise ValueError("entropy patcher artifact size mismatch")
    digest = hashlib.sha256(payload).hexdigest()
    if artifact.get("sha256") != digest:
        raise ValueError("entropy patcher artifact sha256 mismatch")
    try:
        patcher = CausalEntropyPatcher.from_bytes(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("entropy patcher artifact is invalid") from error
    if patcher.config.max_patch_size != max_patch_size:
        raise ValueError("serialized patcher disagrees with manifest max_patch_size")
    if policy.get("boundary_config") != asdict(patcher.config):
        raise ValueError("serialized patcher boundary config disagrees with manifest")
    if policy.get("entropy_model_config") != asdict(patcher.model.config):
        raise ValueError("serialized entropy model config disagrees with manifest")
    return DatasetPatchingSpec(
        "causal_entropy_v1", None, max_patch_size, artifact_sha256=digest
    )


@dataclass(frozen=True)
class VariablePatchLayout:
    """PAD-free local/global topology derived from per-byte patch offsets."""

    byte_indices: Tensor
    byte_cu_seqlens: Tensor
    patch_byte_cu_seqlens: Tensor
    patch_cu_seqlens: Tensor
    condition_patch_indices: Tensor
    global_patch_sources: Tensor
    global_patch_positions: Tensor
    physical_to_global_patch_indices: Tensor
    bos_condition_indices: Tensor
    physical_patch_row_indices: Tensor
    physical_patch_start_columns: Tensor
    physical_patch_lengths: Tensor
    physical_patch_prior_condition_indices: Tensor


def _segment_cu(rows: Tensor, documents: Tensor) -> Tensor:
    boundary = torch.ones(documents.numel(), dtype=torch.bool)
    boundary[1:] = (rows[1:] != rows[:-1]) | (documents[1:] != documents[:-1])
    starts = torch.nonzero(boundary, as_tuple=False).flatten()
    lengths = torch.diff(torch.cat((starts, starts.new_tensor([documents.numel()]))))
    return torch.cat(
        (
            torch.zeros(1, dtype=torch.int32),
            lengths.to(torch.int32).cumsum(0, dtype=torch.int32),
        )
    )


def build_variable_patch_layout(
    valid: Tensor,
    document_ids: Tensor,
    document_offsets: Tensor,
    patch_offsets: Tensor,
    *,
    max_patch_size: int,
) -> VariablePatchLayout:
    """Build causal variable-patch pooling and decoder-condition mappings."""

    tensors = (valid, document_ids, document_offsets, patch_offsets)
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("variable patch layout must be collated on CPU")
    if any(tensor.shape != valid.shape for tensor in tensors[1:]):
        raise ValueError("variable patch metadata must align")
    if valid.dtype != torch.bool:
        raise TypeError("valid must be boolean")
    if any(tensor.dtype != torch.long for tensor in tensors[1:]):
        raise TypeError("document and patch offsets must be int64")
    if max_patch_size <= 0:
        raise ValueError("max_patch_size must be positive")

    byte_indices = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
    if byte_indices.numel() == 0:
        raise ValueError("variable patch layout requires at least one valid byte")
    width = valid.shape[1]
    rows = torch.div(byte_indices, width, rounding_mode="floor")
    columns = byte_indices.remainder(width)
    documents = document_ids.reshape(-1).index_select(0, byte_indices)
    offsets = document_offsets.reshape(-1).index_select(0, byte_indices)
    within_patch = patch_offsets.reshape(-1).index_select(0, byte_indices)
    if bool((documents < 0).any()) or bool((offsets < 0).any()):
        raise ValueError("valid bytes require document metadata")
    if bool(((within_patch < 0) | (within_patch >= max_patch_size)).any()):
        raise ValueError("patch offset lies outside the manifest-bound maximum")

    byte_cu = _segment_cu(rows, documents)
    patch_start_mask = within_patch.eq(0)
    patch_starts = torch.nonzero(patch_start_mask, as_tuple=False).flatten()
    if patch_starts.numel() == 0 or int(patch_starts[0]) != 0:
        raise ValueError("every packed row must begin on a variable-patch boundary")
    patch_byte_cu = torch.cat(
        (
            patch_starts.to(torch.int32),
            patch_starts.new_tensor([byte_indices.numel()]).to(torch.int32),
        )
    )
    patch_lengths = torch.diff(patch_byte_cu).to(torch.long)
    if bool(((patch_lengths <= 0) | (patch_lengths > max_patch_size)).any()):
        raise ValueError("variable patch length lies outside the manifest bound")
    expected_offsets = torch.arange(byte_indices.numel()) - torch.repeat_interleave(
        patch_starts, patch_lengths
    )
    if not torch.equal(within_patch, expected_offsets):
        raise ValueError("patch offsets do not form contiguous zero-based runs")
    patch_rows = rows.index_select(0, patch_starts)
    patch_documents = documents.index_select(0, patch_starts)
    expanded_patch_documents = torch.repeat_interleave(patch_documents, patch_lengths)
    if not torch.equal(documents, expanded_patch_documents):
        raise ValueError("one variable patch mixes multiple documents")
    patch_columns = columns.index_select(0, patch_starts)
    patch_document_offsets = offsets.index_select(0, patch_starts)

    physical_patch_cu = _segment_cu(patch_rows, patch_documents)
    segment_starts = physical_patch_cu[:-1].to(torch.long)
    segment_lengths = torch.diff(physical_patch_cu).to(torch.long)
    begins_document = patch_document_offsets.index_select(0, segment_starts).eq(0)
    segment_ids = torch.repeat_interleave(
        torch.arange(segment_lengths.numel()), segment_lengths
    )
    bos_prefix = begins_document.to(torch.long).cumsum(0)
    physical_to_global = (
        torch.arange(patch_starts.numel(), dtype=torch.long)
        + bos_prefix.index_select(0, segment_ids)
    )
    total_global = int(patch_starts.numel() + begins_document.sum())
    global_sources = torch.full((total_global,), -1, dtype=torch.long)
    global_sources[physical_to_global] = torch.arange(
        patch_starts.numel(), dtype=torch.long
    )
    # Byte offsets are stable across continuation pages; +1 reserves zero for BOS.
    global_positions = torch.zeros(total_global, dtype=torch.long)
    global_positions[physical_to_global] = patch_document_offsets + 1
    global_patch_cu = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32),
            (segment_lengths + begins_document.to(torch.long))
            .to(torch.int32)
            .cumsum(0, dtype=torch.int32),
        )
    )
    bos_conditions = (
        segment_starts
        + torch.cat((torch.zeros(1, dtype=torch.long), bos_prefix[:-1]))
    )[begins_document]

    within_segment = torch.arange(patch_starts.numel()) - torch.repeat_interleave(
        segment_starts, segment_lengths
    )
    prior = torch.where(
        within_segment.gt(0),
        physical_to_global - 1,
        torch.where(
            begins_document.index_select(0, segment_ids), physical_to_global - 1, -1
        ),
    )
    current_per_byte = torch.repeat_interleave(physical_to_global, patch_lengths)
    prior_per_byte = torch.repeat_interleave(prior, patch_lengths)
    patch_final = within_patch.eq(
        torch.repeat_interleave(patch_lengths - 1, patch_lengths)
    )
    condition = torch.where(patch_final, current_per_byte, prior_per_byte)

    return VariablePatchLayout(
        byte_indices=byte_indices,
        byte_cu_seqlens=byte_cu,
        patch_byte_cu_seqlens=patch_byte_cu,
        patch_cu_seqlens=global_patch_cu,
        condition_patch_indices=condition,
        global_patch_sources=global_sources,
        global_patch_positions=global_positions,
        physical_to_global_patch_indices=physical_to_global,
        bos_condition_indices=bos_conditions,
        physical_patch_row_indices=patch_rows,
        physical_patch_start_columns=patch_columns,
        physical_patch_lengths=patch_lengths,
        physical_patch_prior_condition_indices=prior,
    )
