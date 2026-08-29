from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

import numpy as np
import pytest
import torch

from pretraining.byte_diffusion.config import ByteDiffusionConfig
from pretraining.byte_diffusion.config import CorruptionConfig
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.patching import (
    CausalEntropyPatcher,
    EntropyPatchConfig,
    HashedNgramEntropyConfig,
    HashedNgramEntropyModel,
)
from pretraining.byte_diffusion.variable_patching import (
    DuoCleanPatchMetadata,
    DatasetPatchingSpec,
    ENTROPY_DATASET_SCHEMA,
    PATCHING_POLICY_SCHEMA,
    build_variable_patch_layout,
    build_duo_clean_patch_metadata,
    load_dataset_patching_spec,
)
from pretraining.byte_diffusion.data import AtomicIdManifest
from pretraining.byte_diffusion.export import (
    ARTIFACT_CAP_BYTES,
    artifact_size_report,
)
from pretraining.byte_diffusion.training import (
    MappedPackedChunkDataset,
    TrainingBatch,
    ValidationChunkIdentity,
    load_data_directory,
    prepare_blt_corruption,
    prepare_blt_sampling,
    sample_validation_blt_starts,
)
from scripts.build_bolmo_dataset import (
    CHALLENGE_HEADER_INTS,
    CHALLENGE_MAGIC,
    CHALLENGE_VERSION,
)
from scripts.build_byte_diffusion_dataset import (
    ARTIFACT_ALIGNMENT,
    ARTIFACT_SCHEMA,
    build_dataset,
    mapped_row_sha256,
    write_deterministic_mapped_artifact,
)


def _variable_inputs():
    valid = torch.tensor([[True, True, True, True, True, True, False, False]])
    documents = torch.tensor([[0, 0, 0, 1, 1, 1, -1, -1]])
    document_offsets = torch.tensor([[0, 1, 2, 0, 1, 2, -1, -1]])
    patch_offsets = torch.tensor([[0, 1, 0, 0, 1, 2, -1, -1]])
    return valid, documents, document_offsets, patch_offsets


def _variable_training_batch() -> TrainingBatch:
    valid, documents, document_offsets, patch_offsets = _variable_inputs()
    layout = build_variable_patch_layout(
        valid,
        documents,
        document_offsets,
        patch_offsets,
        max_patch_size=3,
    )
    ids = torch.tensor([[65, 66, 256, 70, 71, 256, 262, 262]])
    return TrainingBatch(
        ids=ids,
        valid=valid,
        ar_targets=torch.full_like(ids, -100),
        bos_targets=torch.empty(0, dtype=torch.long),
        bos_row_indices=torch.empty(0, dtype=torch.long),
        positions=document_offsets.clamp_min(0),
        full_valid=False,
        document_ids=documents,
        byte_indices=layout.byte_indices,
        byte_cu_seqlens=layout.byte_cu_seqlens,
        patch_indices=None,
        patch_cu_seqlens=layout.patch_cu_seqlens,
        condition_patch_indices=layout.condition_patch_indices,
        global_patch_sources=layout.global_patch_sources,
        global_patch_positions=layout.global_patch_positions,
        physical_to_global_patch_indices=layout.physical_to_global_patch_indices,
        bos_condition_indices=layout.bos_condition_indices,
        prior_condition_indices=None,
        patch_offsets=patch_offsets,
        patch_byte_cu_seqlens=layout.patch_byte_cu_seqlens,
        max_patch_size=3,
        physical_patch_row_indices=layout.physical_patch_row_indices,
        physical_patch_start_columns=layout.physical_patch_start_columns,
        physical_patch_lengths=layout.physical_patch_lengths,
        physical_patch_prior_condition_indices=(
            layout.physical_patch_prior_condition_indices
        ),
    )


def test_variable_patch_layout_maps_pooling_bos_and_causal_conditions() -> None:
    valid, documents, document_offsets, patch_offsets = _variable_inputs()
    layout = build_variable_patch_layout(
        valid,
        documents,
        document_offsets,
        patch_offsets,
        max_patch_size=3,
    )

    torch.testing.assert_close(
        layout.byte_cu_seqlens, torch.tensor([0, 3, 6], dtype=torch.int32)
    )
    torch.testing.assert_close(
        layout.patch_byte_cu_seqlens,
        torch.tensor([0, 2, 3, 6], dtype=torch.int32),
    )
    torch.testing.assert_close(
        layout.patch_cu_seqlens, torch.tensor([0, 3, 5], dtype=torch.int32)
    )
    torch.testing.assert_close(
        layout.global_patch_sources, torch.tensor([-1, 0, 1, -1, 2])
    )
    torch.testing.assert_close(
        layout.global_patch_positions, torch.tensor([0, 1, 3, 0, 1])
    )
    torch.testing.assert_close(
        layout.physical_to_global_patch_indices, torch.tensor([1, 2, 4])
    )
    torch.testing.assert_close(layout.bos_condition_indices, torch.tensor([0, 3]))
    torch.testing.assert_close(
        layout.condition_patch_indices, torch.tensor([0, 1, 2, 3, 3, 4])
    )
    torch.testing.assert_close(
        layout.physical_patch_row_indices, torch.tensor([0, 0, 0])
    )
    torch.testing.assert_close(
        layout.physical_patch_start_columns, torch.tensor([0, 2, 3])
    )
    torch.testing.assert_close(
        layout.physical_patch_lengths, torch.tensor([2, 1, 3])
    )
    torch.testing.assert_close(
        layout.physical_patch_prior_condition_indices, torch.tensor([0, 1, 3])
    )


def test_duo_entropy_layout_has_no_virtual_bos_and_document_local_ordinals() -> None:
    valid, documents, _, patch_offsets = _variable_inputs()
    layout = build_duo_clean_patch_metadata(
        valid,
        documents,
        patch_offsets,
        max_patch_size=3,
    )

    assert isinstance(layout, DuoCleanPatchMetadata)
    torch.testing.assert_close(
        layout.patch_byte_cu_seqlens,
        torch.tensor([0, 2, 3, 6], dtype=torch.int32),
    )
    # Exactly three physical patches: no synthetic BOS entries.
    torch.testing.assert_close(
        layout.patch_cu_seqlens, torch.tensor([0, 2, 3], dtype=torch.int32)
    )
    assert layout.max_patch_seqlen == 2
    torch.testing.assert_close(layout.patch_ordinals, torch.tensor([0, 1, 0]))
    # A clean byte receives its current latent only when that patch closes.
    torch.testing.assert_close(
        layout.byte_condition_indices, torch.tensor([-1, 0, 1, -1, -1, 2])
    )
    # A mutable origin never receives the patch containing the origin.
    torch.testing.assert_close(
        layout.origin_condition_indices,
        torch.tensor([[-1, -1, 0, -1, -1, -1, -1, -1]]),
    )


def test_duo_entropy_keeps_byte_axis_static_while_pool_axis_is_ragged() -> None:
    documents = torch.tensor(
        [[0] * 8, [1] * 8], dtype=torch.long
    )
    valid_a = torch.tensor(
        [[True] * 6 + [False] * 2, [True] * 4 + [False] * 4]
    )
    valid_b = torch.tensor(
        [[True] * 8, [True] * 4 + [False] * 4]
    )

    def build(valid: torch.Tensor) -> DuoCleanPatchMetadata:
        offsets = torch.where(valid, torch.zeros_like(documents), -1)
        return build_duo_clean_patch_metadata(
            valid,
            documents,
            offsets,
            max_patch_size=1,
            max_segments_per_row=4,
        )

    a, b = build(valid_a), build(valid_b)
    torch.testing.assert_close(a.byte_indices, torch.arange(16))
    torch.testing.assert_close(a.byte_indices, b.byte_indices)
    torch.testing.assert_close(
        a.byte_cu_seqlens,
        torch.tensor([0, 8, 16, 16, 16, 16, 16, 16, 16], dtype=torch.int32),
    )
    torch.testing.assert_close(a.byte_cu_seqlens, b.byte_cu_seqlens)
    assert a.pool_byte_indices.numel() == 10
    assert b.pool_byte_indices.numel() == 12
    assert a.patch_ordinals.numel() == 10
    assert b.patch_ordinals.numel() == 12
    assert a.max_patch_seqlen == 6
    assert b.max_patch_seqlen == 8


def test_entropy_origins_are_exhaustive_and_preserve_overflow_validity() -> None:
    batch = _variable_training_batch()
    config = CorruptionConfig(
        kind="blt_exact_k", canvas_length=4, branches_per_row=4
    )
    sampling = prepare_blt_sampling(
        batch,
        config,
        torch.Generator().manual_seed(41),
        clean_window=512,
    )

    torch.testing.assert_close(
        sampling.block_starts, torch.tensor([[0, 2, 3, 0]])
    )
    torch.testing.assert_close(
        sampling.selected, torch.tensor([[True, True, True, False]])
    )
    torch.testing.assert_close(
        sampling.condition_indices, torch.tensor([[0, 1, 3, -1]])
    )
    torch.testing.assert_close(sampling.sampling_weight, torch.ones(1))
    assert sampling.branch_valid.sum(2).tolist() == [[3, 1, 3, 0]]

    plan = prepare_blt_corruption(
        batch,
        config,
        ByteDiffusionConfig.tiny().vocab,
        torch.Generator().manual_seed(43),
        sampling=sampling,
    )
    torch.testing.assert_close(plan.block_starts, sampling.block_starts)
    torch.testing.assert_close(plan.condition_indices, sampling.condition_indices)
    torch.testing.assert_close(plan.branch_valid, sampling.branch_valid)


def test_entropy_origin_sampling_is_without_replacement_and_ht_weighted() -> None:
    ids = torch.tensor([[*range(65, 72), 256]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    documents = torch.zeros_like(ids)
    positions = torch.arange(8)[None]
    patch_offsets = torch.zeros_like(ids)
    layout = build_variable_patch_layout(
        valid,
        documents,
        positions,
        patch_offsets,
        max_patch_size=1,
    )
    batch = TrainingBatch(
        ids=ids,
        valid=valid,
        ar_targets=torch.full_like(ids, -100),
        bos_targets=torch.empty(0, dtype=torch.long),
        positions=positions,
        full_valid=True,
        document_ids=documents,
        physical_patch_row_indices=layout.physical_patch_row_indices,
        physical_patch_start_columns=layout.physical_patch_start_columns,
        physical_patch_prior_condition_indices=(
            layout.physical_patch_prior_condition_indices
        ),
    )
    config = CorruptionConfig(
        kind="blt_exact_k", canvas_length=4, branches_per_row=3
    )
    sampling = prepare_blt_sampling(
        batch,
        config,
        torch.Generator().manual_seed(47),
        clean_window=512,
    )

    starts = sampling.block_starts[0]
    assert starts.tolist() == sorted(starts.tolist())
    assert starts.unique().numel() == 3
    assert bool(sampling.selected.all())
    torch.testing.assert_close(sampling.sampling_weight, torch.tensor([8 / 3]))


def test_entropy_validation_origins_are_stateless_and_repeatable() -> None:
    batch = _variable_training_batch()
    identity = ValidationChunkIdentity(chunk_index=17, stream_start=123)
    expected = sample_validation_blt_starts(
        batch,
        (identity,),
        block_length=4,
        count=2,
        patch_stride=4,
        seed=53,
    )
    observed = sample_validation_blt_starts(
        batch,
        (identity,),
        block_length=4,
        count=2,
        patch_stride=4,
        seed=53,
    )

    for expected_tensor, observed_tensor in zip(expected, observed, strict=True):
        torch.testing.assert_close(expected_tensor, observed_tensor)


def test_clean_model_uses_ragged_patch_pool_end_to_end() -> None:
    torch.manual_seed(19)
    valid, documents, document_offsets, patch_offsets = _variable_inputs()
    layout = build_variable_patch_layout(
        valid,
        documents,
        document_offsets,
        patch_offsets,
        max_patch_size=3,
    )
    ids = torch.tensor([[65, 66, 256, 70, 71, 256, 262, 262]])
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).eval()

    output = model.forward_ar_varlen(
        ids,
        valid,
        positions=document_offsets.clamp_min(0),
        allow_dense_reference=True,
        document_ids=documents,
        byte_indices=layout.byte_indices,
        byte_cu_seqlens=layout.byte_cu_seqlens,
        patch_cu_seqlens=layout.patch_cu_seqlens,
        condition_patch_indices=layout.condition_patch_indices,
        global_patch_sources=layout.global_patch_sources,
        global_patch_positions=layout.global_patch_positions,
        physical_to_global_patch_indices=layout.physical_to_global_patch_indices,
        bos_condition_indices=layout.bos_condition_indices,
        patch_byte_cu_seqlens=layout.patch_byte_cu_seqlens,
        max_patch_size=3,
    )

    assert output.logits.shape == (1, 8, model.config.vocab.output_size)
    assert torch.isfinite(output.logits[valid]).all()
    assert output.bos_patch_states is not None
    assert output.bos_patch_states.shape == (2, model.config.global_dim)


def _entropy_manifest(tmp_path):
    model = HashedNgramEntropyModel(
        HashedNgramEntropyConfig(table_size=16, vocab_size=261)
    )
    patcher = CausalEntropyPatcher(
        model, EntropyPatchConfig(threshold=100.0, max_patch_size=5)
    )
    payload = patcher.to_bytes()
    artifact = tmp_path / "entropy-patcher.bdpatch"
    artifact.write_bytes(payload)
    return {
        "schema": ENTROPY_DATASET_SCHEMA,
        "packing": {"patch_stride": None},
        "patching": {
            "schema": PATCHING_POLICY_SCHEMA,
            "name": "causal_entropy_v1",
            "patcher_artifact": {
                "path": artifact.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            },
            "entropy_model_config": asdict(model.config),
            "boundary_config": asdict(patcher.config),
            "max_patch_size": 5,
        },
    }


def test_entropy_manifest_authenticates_policy_and_patcher(tmp_path) -> None:
    manifest = _entropy_manifest(tmp_path)
    spec = load_dataset_patching_spec(tmp_path, manifest)
    assert spec.variable
    assert spec.max_patch_size == 5

    manifest["patching"]["patcher_artifact"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_dataset_patching_spec(tmp_path, manifest)


def test_entropy_patcher_keeps_default_model_inside_artifact_cap() -> None:
    config = ByteDiffusionConfig()
    report = artifact_size_report(
        ByteDiffusionModel(config),
        config,
        encoding_policy="mixed_sensitive",
    )
    patcher = CausalEntropyPatcher(HashedNgramEntropyModel())

    assert report.parameter_count == 23_010_306
    assert report.artifact_bytes + len(patcher.to_bytes()) < ARTIFACT_CAP_BYTES


def test_variable_layout_rejects_patch_crossing_document_boundary() -> None:
    valid, documents, document_offsets, patch_offsets = _variable_inputs()
    patch_offsets[0, 3] = 1
    with pytest.raises(
        ValueError, match="outside the manifest bound|zero-based runs|mixes multiple"
    ):
        build_variable_patch_layout(
            valid,
            documents,
            document_offsets,
            patch_offsets,
            max_patch_size=3,
        )


def test_native_mapped_v6_collation_preserves_variable_patch_metadata(
    tmp_path,
) -> None:
    valid, documents, document_offsets, patch_offsets = _variable_inputs()
    ids = np.asarray([[65, 66, 256, 70, 71, 256, 262, 262]], dtype="<u2")
    valid_numpy = valid.numpy()
    arrays = {
        "chunk_index": np.asarray([0], dtype="<i8"),
        "stream_start": np.asarray([0], dtype="<i8"),
        "stream_stop": np.asarray([6], dtype="<i8"),
        "input_ids": ids,
        "target_ids": np.asarray(
            [[66, 256, 262, 71, 256, 262, 262, 262]], dtype="<u2"
        ),
        "valid_mask": valid_numpy,
        "score_mask": np.asarray(
            [[True, True, False, True, True, False, False, False]]
        ),
        "document_indices": documents.numpy().astype("<i8"),
        "document_offsets": document_offsets.numpy().astype("<i4"),
        "patch_offsets": patch_offsets.numpy().astype(np.int8),
        "label_halo_id": np.asarray([262], dtype="<u2"),
        "label_halo_valid": np.asarray([False]),
    }
    path = tmp_path / "train-00000.bdm"
    descriptors = write_deterministic_mapped_artifact(path, arrays)
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "format": "aligned_raw_arrays/v1",
        "alignment": ARTIFACT_ALIGNMENT,
        "arrays": descriptors,
        "row_sha256": mapped_row_sha256(arrays),
        "chunks": 1,
        "chunk_size": 8,
        "first_chunk_index": 0,
        "last_chunk_index": 0,
        "valid_atomic_tokens": 6,
        "literal_atomic_tokens": 4,
        "special_atomic_tokens": 2,
        "eot_atomic_tokens": 2,
        "physical_storage_positions": 8,
        "storage_padding_tokens": 2,
        "canvas512_eligible_positions": 6,
        "scored_ar_targets": 4,
        "row_valid_counts": [6],
        "row_physical_extents": [6],
        "row_document_starts": [2],
    }
    dataset = MappedPackedChunkDataset(
        tmp_path,
        (artifact,),
        artifact_schema=ARTIFACT_SCHEMA,
        chunk_size=8,
        required_branch_bytes=0,
        branch_span_length=0,
        split="train",
        dense_stream=False,
        document_aligned_pages=True,
        patching=DatasetPatchingSpec("causal_entropy_v1", None, 3),
    )

    batch = dataset.training_batch((0,))

    torch.testing.assert_close(batch.patch_offsets, patch_offsets)
    torch.testing.assert_close(
        batch.patch_byte_cu_seqlens,
        torch.tensor([0, 2, 3, 6], dtype=torch.int32),
    )
    torch.testing.assert_close(
        batch.physical_patch_start_columns, torch.tensor([0, 2, 3])
    )
    assert batch.patch_indices is None
    assert batch.prior_condition_indices is None
    assert batch.max_patch_size == 3


def test_built_v6_loads_rows_that_end_early_at_a_patch_boundary(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    atomic = AtomicIdManifest.reference()
    tokens = [atomic.eot_id, *b"abcdefghijk", atomic.eot_id]

    def write_shard(path) -> None:
        header = np.zeros(CHALLENGE_HEADER_INTS, dtype="<i4")
        header[0] = CHALLENGE_MAGIC
        header[1] = CHALLENGE_VERSION
        header[2] = len(tokens)
        with path.open("wb") as handle:
            handle.write(header.tobytes())
            handle.write(np.asarray(tokens, dtype="<u2").tobytes())

    train_path = source / "train.bin"
    validation_path = source / "validation.bin"
    write_shard(train_path)
    write_shard(validation_path)
    (source / "mix_manifest.json").write_text(
        json.dumps(
            {
                "tokenizer_provenance": {
                    "kind": "utf8_bytes",
                    "name": "utf8_bytes",
                    "vocab_size": atomic.output_size,
                    "eot_id": atomic.eot_id,
                    "spec_sha256": atomic.sha256,
                    "ngrams_sha256": None,
                },
                "loader_aligned": True,
                "physical_shard_tokens": len(tokens),
                "unique_stream_tokens": len(tokens),
            }
        )
    )
    patcher_manifest = _entropy_manifest(tmp_path)
    patcher_path = tmp_path / patcher_manifest["patching"]["patcher_artifact"]["path"]
    build_dataset(
        output_dir=output,
        train_paths=(train_path,),
        validation_paths=(validation_path,),
        chunk_size=8,
        chunks_per_shard=2,
        document_aligned_pages=True,
        patching_policy_name="causal_entropy_v1",
        entropy_patcher_path=patcher_path,
    )

    _, train, _ = load_data_directory(
        output,
        chunk_size=8,
        recipe="causal_only",
        expected_patching_policy="causal_entropy_v1",
    )
    batch = train.training_batch((0,))
    assert batch.valid.sum().item() == 5
    assert batch.patch_byte_cu_seqlens is not None
    assert batch.max_patch_size == 5

    with pytest.raises(ValueError, match="patching policy differs"):
        load_data_directory(
            output,
            chunk_size=8,
            recipe="causal_only",
            expected_patching_policy="fixed_stride_v1",
        )

    _, blt_train, _ = load_data_directory(
        output,
        chunk_size=8,
        recipe="blt_d",
        required_branch_bytes=4,
        branch_span_length=4,
        expected_patching_policy="causal_entropy_v1",
    )
    assert blt_train.branch_span_length == 4
    assert blt_train.patching.max_patch_size == 5
