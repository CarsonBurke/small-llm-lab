"""CPU contracts for scratch training, metrics, and exact resume."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import pretraining.byte_diffusion.model as model_module
import pretraining.byte_diffusion.training as training_module
from pretraining.byte_diffusion.config import (
    FAST_BLT_ENTROPY_B4_COMPLETE_PRESET,
    FAST_BLT_ENTROPY_B4_COMPLETE_PAPER_RATIO_PRESET,
    ByteDiffusionConfig,
    CorruptionConfig,
)
from pretraining.byte_diffusion.data import (
    AtomicDocument,
    AtomicIdManifest,
    DeterministicChunkCursor,
    pack_documents,
)
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.variable_patching import DatasetPatchingSpec
from pretraining.byte_diffusion.training import (
    CAUSAL_CONTROL_PRESET,
    CANONICAL_PRESET,
    CHECKPOINT_SCHEMA,
    DENSE_FAST_BLT_PRESET,
    ENTROPY_FAST_BLT_PRESET,
    BltCorruptionPlan,
    ByteDiffusionTrainer,
    CanvasCorruptionPlan,
    DistributedContext,
    DeterministicSubsetChunkDataset,
    MappedPackedChunkDataset,
    TrainingBatch,
    TrainingRunConfig,
    ValidationChunkIdentity,
    attention_context,
    chunks_to_batch,
    create_model_optimizer,
    create_optimizer,
    format_train_metric,
    format_validation_metric,
    gradient_interference_metrics,
    learning_rate_multiplier,
    load_data_directory,
    prepare_blt_corruption,
    prepare_canvas_corruption,
    sample_nonoverlapping_patch_starts,
    sample_validation_starts,
    take_distributed_chunks,
    take_uneven_distributed_indices,
    validation_exact_k_mask,
)
from scripts.ablation import compare_results, parse_log_line
from scripts.benchmark_byte_diffusion_real_data import build_benchmark_run_config
from scripts.build_byte_diffusion_dataset import (
    ARTIFACT_SCHEMA,
    ARTIFACT_ALIGNMENT,
    mapped_row_sha256,
    write_deterministic_mapped_artifact,
)
from scripts.train_byte_diffusion import (
    expected_model_config_for_run,
    model_config_from_env,
    training_source_provenance,
)


def _chunks(*, offset: int = 0):
    manifest = AtomicIdManifest.reference()
    documents = [
        AtomicDocument(
            key=f"doc-{offset + index}",
            atomic_ids=(
                65 + index,
                66,
                67,
                68,
                69,
                70,
                71,
                manifest.eot_id,
            ),
        )
        for index in range(4)
    ]
    return pack_documents(documents, manifest, chunk_size=8)


def _hand_enumerated_entropy_batch() -> TrainingBatch:
    ids = torch.tensor(
        [
            [65, 66, 256, 70, 71, 72, 73, 256],
            [80, 81, 82, 83, 84, 85, 86, 256],
        ],
        dtype=torch.long,
    )
    valid = torch.ones_like(ids, dtype=torch.bool)
    documents = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2]],
        dtype=torch.long,
    )
    positions = torch.tensor(
        [[0, 1, 2, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5, 6, 7]],
        dtype=torch.long,
    )
    patch_offsets = torch.tensor(
        [[0, 1, 2, 0, 1, 0, 1, 2], [0, 1, 0, 1, 2, 3, 0, 1]],
        dtype=torch.long,
    )
    layout = training_module.build_variable_patch_layout(
        valid,
        documents,
        positions,
        patch_offsets,
        max_patch_size=4,
    )
    return TrainingBatch(
        ids=ids,
        valid=valid,
        ar_targets=torch.full_like(ids, -100),
        bos_targets=torch.tensor([65, 70, 80]),
        positions=positions,
        full_valid=False,
        document_ids=documents,
        bos_row_indices=torch.tensor([0, 0, 1]),
        isolate_documents=True,
        byte_indices=layout.byte_indices,
        byte_cu_seqlens=layout.byte_cu_seqlens,
        patch_cu_seqlens=layout.patch_cu_seqlens,
        condition_patch_indices=layout.condition_patch_indices,
        global_patch_sources=layout.global_patch_sources,
        global_patch_positions=layout.global_patch_positions,
        physical_to_global_patch_indices=(
            layout.physical_to_global_patch_indices
        ),
        bos_condition_indices=layout.bos_condition_indices,
        patch_offsets=patch_offsets,
        patch_byte_cu_seqlens=layout.patch_byte_cu_seqlens,
        max_patch_size=4,
        physical_patch_row_indices=layout.physical_patch_row_indices,
        physical_patch_start_columns=layout.physical_patch_start_columns,
        physical_patch_prior_condition_indices=(
            layout.physical_patch_prior_condition_indices
        ),
    )


def _document_model_metadata(layout):
    return {
        "byte_indices": layout[0],
        "byte_cu_seqlens": layout[1],
        "patch_indices": layout[2],
        "patch_cu_seqlens": layout[3],
        "condition_patch_indices": layout[4],
        "global_patch_sources": layout[5],
        "global_patch_positions": layout[6],
        "physical_to_global_patch_indices": layout[7],
        "bos_condition_indices": layout[8],
    }


def _run_config(*, recipe="canvas", iterations=2, accumulation=1):
    return TrainingRunConfig(
        iterations=iterations,
        val_loss_every=1,
        train_log_every=1,
        warmdown_iters=0,
        run_id="cpu-contract",
        seed=17,
        recipe=recipe,
        corruption=CorruptionConfig(
            kind="absorbing_rb", canvas_length=8, branches_per_row=1
        ),
        microbatch_per_rank=1,
        gradient_accumulation=accumulation,
        attention_policy="dense_reference",
        allow_cpu_reference=True,
    )


def _trainer(model, config, chunks=None):
    chunks = tuple(chunks or _chunks())
    return ByteDiffusionTrainer(
        model,
        DeterministicChunkCursor(chunks, seed=config.seed),
        chunks,
        config,
        device=torch.device("cpu"),
    )


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _valid_document_aligned_artifact() -> dict[str, np.ndarray]:
    manifest = AtomicIdManifest.reference()
    return {
        "chunk_index": np.asarray([0], dtype="<i8"),
        "stream_start": np.asarray([0], dtype="<i8"),
        "stream_stop": np.asarray([8], dtype="<i8"),
        "input_ids": np.asarray(
            [
                [
                    65,
                    manifest.eot_id,
                    manifest.pad_id,
                    manifest.pad_id,
                    70,
                    71,
                    72,
                    manifest.eot_id,
                ]
            ],
            dtype="<u2",
        ),
        "target_ids": np.asarray(
            [
                [
                    manifest.eot_id,
                    manifest.pad_id,
                    manifest.pad_id,
                    manifest.pad_id,
                    71,
                    72,
                    manifest.eot_id,
                    manifest.pad_id,
                ]
            ],
            dtype="<u2",
        ),
        "valid_mask": np.asarray(
            [[True, True, False, False, True, True, True, True]],
            dtype=np.bool_,
        ),
        "score_mask": np.asarray(
            [[True, False, False, False, True, True, True, False]],
            dtype=np.bool_,
        ),
        "document_indices": np.asarray(
            [[0, 0, -1, -1, 1, 1, 1, 1]], dtype="<i8"
        ),
        "document_offsets": np.asarray(
            [[0, 1, -1, -1, 0, 1, 2, 3]], dtype="<i4"
        ),
        "patch_offsets": np.asarray(
            [[0, 1, -1, -1, 0, 1, 2, 3]], dtype=np.int8
        ),
        "label_halo_id": np.asarray([manifest.pad_id], dtype="<u2"),
        "label_halo_valid": np.asarray([False], dtype=np.bool_),
    }


def _load_document_aligned_artifact(
    tmp_path: Path,
    arrays: dict[str, np.ndarray],
    *,
    defer_payload_validation: bool = False,
    trust_pinned_row_index: bool = False,
    mutate_artifact: Callable[[dict], None] | None = None,
    branch_span_length: int = 4,
) -> MappedPackedChunkDataset:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "train-00000.bdm"
    descriptors = write_deterministic_mapped_artifact(path, arrays)
    valid = arrays["valid_mask"]
    valid_ids = arrays["input_ids"][valid]
    physical_positions = int(arrays["input_ids"].size)
    valid_atoms = int(valid.sum())
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "format": "aligned_raw_arrays/v1",
        "alignment": ARTIFACT_ALIGNMENT,
        "arrays": descriptors,
        "row_sha256": mapped_row_sha256(arrays),
        "chunks": int(valid.shape[0]),
        "chunk_size": int(valid.shape[1]),
        "first_chunk_index": int(arrays["chunk_index"][0]),
        "last_chunk_index": int(arrays["chunk_index"][-1]),
        "valid_atomic_tokens": valid_atoms,
        "literal_atomic_tokens": int((valid_ids < 256).sum()),
        "special_atomic_tokens": int((valid_ids >= 256).sum()),
        "eot_atomic_tokens": int((valid_ids == 256).sum()),
        "physical_storage_positions": physical_positions,
        "storage_padding_tokens": physical_positions - valid_atoms,
        "canvas512_eligible_positions": int(
            np.minimum(valid.sum(axis=1), 512).sum()
        ),
        "scored_ar_targets": int(arrays["score_mask"].sum()),
        "row_valid_counts": valid.sum(axis=1, dtype=np.int64).tolist(),
        "row_physical_extents": np.where(
            valid,
            np.arange(valid.shape[1], dtype=np.int64)[None] + 1,
            0,
        ).max(axis=1).tolist(),
        "row_document_starts": (
            valid & (arrays["document_offsets"] == 0)
        ).sum(axis=1, dtype=np.int64).tolist(),
    }
    if mutate_artifact is not None:
        mutate_artifact(artifact)
    return MappedPackedChunkDataset(
        tmp_path,
        (artifact,),
        artifact_schema=ARTIFACT_SCHEMA,
        chunk_size=int(valid.shape[1]),
        required_branch_bytes=4,
        branch_span_length=branch_span_length,
        split="train",
        dense_stream=False,
        document_aligned_pages=True,
        eot_id=AtomicIdManifest.reference().eot_id,
        defer_payload_validation=defer_payload_validation,
        trust_pinned_row_index=trust_pinned_row_index,
    )


def test_mapped_training_metadata_plans_batches_and_exact_ar_units(
    tmp_path: Path,
) -> None:
    arrays = _valid_document_aligned_artifact()
    dataset = _load_document_aligned_artifact(tmp_path, arrays)

    groups = dataset.training_batch_groups(
        [0], max_batch_size=4, physical_token_budget=64
    )
    batch = dataset.training_batch(groups[0])

    assert groups == [(0,)]
    expected = int(batch.ar_targets.ne(-100).sum() + batch.bos_targets.numel())
    assert dataset.training_ar_units(np.asarray([0], dtype=np.int64)) == expected == 6
    assert dataset.training_ar_units(np.empty(0, dtype=np.int64)) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_native_mapped_batch_completes_one_logged_update(tmp_path: Path) -> None:
    arrays = _valid_document_aligned_artifact()
    dataset = _load_document_aligned_artifact(tmp_path, arrays)
    run = replace(
        _run_config(recipe="blt_d", iterations=1),
        corruption=CorruptionConfig(
            kind="blt_exact_k", canvas_length=4, branches_per_row=2
        ),
        microbatch_per_rank=1,
        gradient_accumulation=1,
        attention_policy="flash_sdpa",
        allow_cpu_reference=False,
        compile_model=False,
    )
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    trainer = ByteDiffusionTrainer(
        model,
        DeterministicChunkCursor(dataset, seed=run.seed, shuffle=False),
        dataset,
        run,
        device=torch.device("cuda"),
    )

    metrics = trainer.run_update(materialize_metrics=True)

    assert metrics is not None
    assert metrics.step == 1
    assert metrics.microsteps == 1
    assert metrics.max_microbatch == 1
    assert metrics.max_physical_positions == 16


def test_standard_ablation_environment_is_the_run_contract(monkeypatch) -> None:
    monkeypatch.setenv("ITERATIONS", "2000")
    monkeypatch.setenv("VAL_LOSS_EVERY", "20")
    monkeypatch.setenv("TRAIN_LOG_EVERY", "10")
    monkeypatch.setenv("WARMDOWN_ITERS", "1200")
    monkeypatch.setenv("RUN_ID", "bd_canvas128")
    monkeypatch.setenv("BYTE_DIFFUSION_RECIPE", "canvas")
    observed = TrainingRunConfig.from_env()
    assert observed.iterations == 2_000
    assert observed.val_loss_every == 20
    assert observed.train_log_every == 10
    assert observed.warmdown_iters == 1_200
    assert observed.run_id == "bd_canvas128"
    assert observed.recipe == "canvas"


def test_model_ablation_environment_selects_one_explicit_cell(monkeypatch) -> None:
    monkeypatch.setenv("BYTE_DIFFUSION_NGRAM_HASH", "legacy257")
    monkeypatch.setenv("BYTE_DIFFUSION_NGRAM_FACTOR_INIT", "scale_matched")
    monkeypatch.setenv("BYTE_DIFFUSION_NGRAM_AGGREGATION", "mean")
    monkeypatch.setenv(
        "BYTE_DIFFUSION_DECODER_CONDITIONING", "rmsnorm_projection"
    )
    monkeypatch.setenv("BYTE_DIFFUSION_NGRAM_ENABLED", "1")
    monkeypatch.setenv("BYTE_DIFFUSION_NGRAM_TABLE_SIZE", "4096")
    monkeypatch.setenv("BYTE_DIFFUSION_NGRAM_RANK", "8")
    monkeypatch.setenv("BYTE_DIFFUSION_DECODER_LAYERS", "3")
    monkeypatch.setenv("BYTE_DIFFUSION_DECODER_FFN_DIM", "48")
    monkeypatch.setenv("BYTE_DIFFUSION_OUTPUT_TIED", "1")
    monkeypatch.setenv("BYTE_DIFFUSION_EXPLICIT_TIMESTEP", "1")
    monkeypatch.setenv("BYTE_DIFFUSION_SELF_CONDITIONING", "1")
    monkeypatch.setenv("BYTE_DIFFUSION_DECODER_SPLIT_RESIDUAL_SCALE", "0.25")
    monkeypatch.setenv("BYTE_DUO_TIME_FEATURES", "256")
    monkeypatch.setenv("BYTE_DUO_TIME_CONDITION_DIM", "128")
    monkeypatch.setenv("BYTE_DUO_NOISY_NGRAMS", "0")

    observed = model_config_from_env(tiny=True)

    assert observed.ngram_hash == "legacy257"
    assert observed.ngram_factor_init == "scale_matched"
    assert observed.ngram_aggregation == "mean"
    assert observed.decoder_conditioning == "rmsnorm_projection"
    assert observed.ngram_enabled
    assert observed.ngram_table_size == 4_096
    assert observed.ngram_rank == 8
    assert observed.decoder_layers == 3
    assert observed.decoder_ffn_dim == 48
    assert observed.output_tied
    assert observed.explicit_timestep
    assert observed.self_conditioning
    assert observed.decoder_split_residual_scale == 0.25
    assert observed.duo_time_features == 256
    assert observed.duo_time_condition_dim == 128
    assert not observed.duo_noisy_ngrams


def test_real_data_benchmark_uses_and_checks_production_recipe(monkeypatch) -> None:
    monkeypatch.setenv("BYTE_DIFFUSION_RECIPE", "causal_only")
    args = SimpleNamespace(
        microbatch=32,
        microbatch_token_budget=278_528,
        global_batch=256,
        static_shapes=False,
        activation_checkpointing=False,
        expected_recipe="causal_only",
    )
    observed = build_benchmark_run_config(args, total_updates=11)
    assert observed.recipe == "causal_only"
    assert observed.microbatch_per_rank == 32
    assert observed.gradient_accumulation == 8

    args.expected_recipe = "canvas"
    with pytest.raises(ValueError, match="expected recipe"):
        build_benchmark_run_config(args, total_updates=11)


def test_training_source_provenance_is_complete_and_stable() -> None:
    first = training_source_provenance()
    second = training_source_provenance()
    assert first == second
    assert first["schema"] == "byte_diffusion_source_provenance/v2"
    assert len(first["sha256"]) == 64
    files = first["files"]
    assert "scripts/ablation.py" in files
    assert "scripts/train_byte_diffusion.py" in files
    assert "pretraining/byte_diffusion/training.py" in files
    assert "pretraining/byte_diffusion/model.py" in files
    assert "pretraining/byte_diffusion/corruption.py" in files
    assert "pretraining/__init__.py" in files
    assert "pretraining/byte_diffusion/__init__.py" in files
    assert "pretraining/byte_diffusion/inference.py" not in files


def test_loader_enforces_pinned_dataset_and_source_manifest_hashes(
    tmp_path: Path,
) -> None:
    unsigned = {
        "schema": "byte_diffusion_dataset/v5",
        "artifact_schema": ARTIFACT_SCHEMA,
        "atomic_vocabulary": AtomicIdManifest.reference().to_dict(),
        "packing": {"chunk_size": 8, "layout": "document_aligned_pages"},
        "source_manifests": [{"path": "source.json", "sha256": "1" * 64}],
        "splits": {"train": {}, "validation": {}},
    }
    payload_sha256 = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps({**unsigned, "payload_sha256": payload_sha256})
    )

    with pytest.raises(ValueError, match="dataset manifest differs"):
        load_data_directory(
            tmp_path,
            chunk_size=8,
            recipe="causal_only",
            expected_payload_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="source manifest differs"):
        load_data_directory(
            tmp_path,
            chunk_size=8,
            recipe="causal_only",
            expected_payload_sha256=payload_sha256,
            expected_source_manifest_sha256="0" * 64,
        )


def test_ablation_comparison_labels_proxy_bpb(tmp_path, capsys) -> None:
    run_dir = tmp_path / "causal-control"
    run_dir.mkdir()
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "name": "causal-control",
                "steps": 2_000,
                "final_val_bpb": None,
                "final_proxy_val_bpb": 1.2345,
                "final_val_loss": 0.9,
                "elapsed_seconds": 10.0,
            }
        )
    )
    compare_results(tmp_path)
    output = capsys.readouterr().out
    assert "proxy" in output
    assert "1.2345" in output
    assert "FAIL" not in output


def test_canonical_fast_blt_b4_preset_is_fully_pinned(monkeypatch) -> None:
    monkeypatch.setenv("BYTE_DIFFUSION_PRESET", CANONICAL_PRESET)
    monkeypatch.setenv("BYTE_DIFFUSION_MICROBATCH", "32")
    observed = TrainingRunConfig.from_env()
    assert observed.preset == CANONICAL_PRESET
    assert observed.iterations == 2_000
    assert observed.recipe == "blt_d"
    assert observed.corruption == CorruptionConfig(
        kind="blt_bernoulli", canvas_length=4, branches_per_row=128
    )
    assert observed.objective_reduction == "paper_sum"
    assert observed.global_batch_size == 249
    assert observed.microbatch_per_rank == 32
    assert observed.microbatch_token_budget == 278_528
    assert observed.gradient_accumulation == 8

    monkeypatch.setenv("BYTE_DIFFUSION_CANVAS_LENGTH", "8")
    with pytest.raises(ValueError, match="contract mismatch"):
        TrainingRunConfig.from_env()


@pytest.mark.parametrize(
    ("preset", "recipe", "patching_policy", "corruption", "reduction"),
    [
        (
            CAUSAL_CONTROL_PRESET,
            "causal_only",
            "fixed_stride_v1",
            CorruptionConfig.canvas512(),
            "equal_mean",
        ),
        (
            DENSE_FAST_BLT_PRESET,
            "blt_d",
            "fixed_stride_v1",
            CorruptionConfig(
                kind="blt_exact_k", canvas_length=4, branches_per_row=2_048
            ),
            "row_normalized_sum",
        ),
        (
            ENTROPY_FAST_BLT_PRESET,
            "blt_d",
            "causal_entropy_v1",
            CorruptionConfig.blt_entropy_reference(),
            "row_normalized_sum",
        ),
    ],
)
def test_reference_presets_resolve_complete_contracts(
    monkeypatch, preset, recipe, patching_policy, corruption, reduction
) -> None:
    monkeypatch.setenv("BYTE_DIFFUSION_PRESET", preset)

    observed = TrainingRunConfig.from_env()

    assert observed.recipe == recipe
    assert observed.patching_policy == patching_policy
    assert observed.corruption == corruption
    assert observed.objective_reduction == reduction
    assert observed.global_batch_size == 249
    assert observed.microbatch_per_rank == 32
    assert observed.microbatch_token_budget == 278_528
    assert observed.gradient_accumulation == 8


def test_reference_presets_reject_microbatch_drift(monkeypatch) -> None:
    monkeypatch.setenv("BYTE_DIFFUSION_PRESET", DENSE_FAST_BLT_PRESET)
    monkeypatch.setenv("BYTE_DIFFUSION_MICROBATCH", "16")
    with pytest.raises(ValueError, match="microbatch_per_rank"):
        TrainingRunConfig.from_env()


@pytest.mark.parametrize(
    "preset",
    [
        FAST_BLT_ENTROPY_B4_COMPLETE_PRESET,
        FAST_BLT_ENTROPY_B4_COMPLETE_PAPER_RATIO_PRESET,
    ],
)
def test_fast_blt_complete_presets_pin_all_origin_entropy_d4_contract(
    monkeypatch: pytest.MonkeyPatch,
    preset: str,
) -> None:
    monkeypatch.setenv("BYTE_DIFFUSION_PRESET", preset)

    observed = TrainingRunConfig.from_env()

    assert observed.preset == preset
    assert observed.recipe == "blt_d"
    assert observed.patching_policy == "causal_entropy_v1"
    assert observed.blt_origin_policy == "all_entropy_patch_starts"
    assert observed.corruption == CorruptionConfig(
        kind="blt_bernoulli",
        canvas_length=4,
        branches_per_row=2_048,
    )
    assert observed.corruption.corrupted_positions_per_row == 8_192
    assert observed.objective_reduction == "paper_sum"
    assert observed.global_batch_size == 249
    assert observed.microbatch_per_rank == 32
    assert observed.microbatch_token_budget == 212_992
    assert observed.gradient_accumulation == 8
    assert observed.activation_checkpointing is True
    expected_model = expected_model_config_for_run(observed)
    if preset == FAST_BLT_ENTROPY_B4_COMPLETE_PRESET:
        assert expected_model == ByteDiffusionConfig.fast_blt_entropy_b4_complete()
    else:
        assert expected_model == (
            ByteDiffusionConfig.fast_blt_entropy_b4_complete_paper_ratio()
        )


def test_paper_ratio_complete_changes_only_model_allocation_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "BYTE_DIFFUSION_PRESET", FAST_BLT_ENTROPY_B4_COMPLETE_PRESET
    )
    retained = TrainingRunConfig.from_env()
    monkeypatch.setenv(
        "BYTE_DIFFUSION_PRESET",
        FAST_BLT_ENTROPY_B4_COMPLETE_PAPER_RATIO_PRESET,
    )
    paper_ratio = TrainingRunConfig.from_env()

    retained_contract = retained.contract_dict()
    paper_contract = paper_ratio.contract_dict()
    retained_contract.pop("preset")
    paper_contract.pop("preset")
    assert paper_contract == retained_contract


@pytest.mark.parametrize(
    ("environment", "value"),
    [
        ("BYTE_DIFFUSION_PATCHING_POLICY", "fixed_stride_v1"),
        ("BYTE_DIFFUSION_BLT_ORIGIN_POLICY", "sampled"),
        ("BYTE_DIFFUSION_CORRUPTION", "blt_exact_k"),
        ("BYTE_DIFFUSION_CANVAS_LENGTH", "8"),
        ("BYTE_DIFFUSION_BRANCHES", "1024"),
        ("BYTE_DIFFUSION_OBJECTIVE_REDUCTION", "row_normalized_sum"),
        ("BYTE_DIFFUSION_MICROBATCH_TOKEN_BUDGET", "278528"),
        ("BYTE_DIFFUSION_ACTIVATION_CHECKPOINTING", "0"),
    ],
)
@pytest.mark.parametrize(
    "preset",
    [
        FAST_BLT_ENTROPY_B4_COMPLETE_PRESET,
        FAST_BLT_ENTROPY_B4_COMPLETE_PAPER_RATIO_PRESET,
    ],
)
def test_fast_blt_complete_presets_reject_training_contract_drift(
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
    value: str,
    preset: str,
) -> None:
    monkeypatch.setenv("BYTE_DIFFUSION_PRESET", preset)
    monkeypatch.setenv(environment, value)

    with pytest.raises(
        ValueError,
        match="contract mismatch|requires BLT-D|only by the named complete ragged",
    ):
        TrainingRunConfig.from_env()


def test_all_origin_policy_rejects_non_entropy_training() -> None:
    with pytest.raises(ValueError, match="requires BLT-D with causal entropy"):
        TrainingRunConfig(blt_origin_policy="all_entropy_patch_starts")


def test_entropy_preset_rejects_fixed_stride_dataset_policy(monkeypatch) -> None:
    monkeypatch.setenv("BYTE_DIFFUSION_PRESET", ENTROPY_FAST_BLT_PRESET)
    monkeypatch.setenv("BYTE_DIFFUSION_PATCHING_POLICY", "fixed_stride_v1")
    with pytest.raises(ValueError, match="patching_policy"):
        TrainingRunConfig.from_env()


def test_entropy_preset_matches_patcher_max_without_changing_branch_budget(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BYTE_DIFFUSION_PRESET", ENTROPY_FAST_BLT_PRESET)
    entropy = TrainingRunConfig.from_env().corruption
    dense = CorruptionConfig(
        kind="blt_exact_k", canvas_length=4, branches_per_row=2_048
    )

    assert entropy == CorruptionConfig.blt_entropy_reference()
    assert entropy.canvas_length == 8
    assert entropy.branches_per_row == 1_024
    assert entropy.corrupted_positions_per_row == 8_192
    assert entropy.corrupted_positions_per_row == dense.corrupted_positions_per_row

    monkeypatch.setenv("BYTE_DIFFUSION_CANVAS_LENGTH", "4")
    monkeypatch.setenv("BYTE_DIFFUSION_BRANCHES", "2048")
    with pytest.raises(ValueError, match="contract mismatch"):
        TrainingRunConfig.from_env()


@pytest.mark.parametrize(
    ("block_length", "branches"), [(4, 2_048), (8, 1_024), (16, 512)]
)
def test_entropy_recipe_expresses_block_horizons_independent_of_patcher(
    block_length: int, branches: int
) -> None:
    config = CorruptionConfig.blt_entropy_reference(block_length=block_length)

    assert config.canvas_length == block_length
    assert config.branches_per_row == branches
    assert config.corrupted_positions_per_row == 8_192


def test_entropy_trainer_requires_authenticated_patcher_provenance() -> None:
    class EntropyChunks(list):
        patching = training_module.DatasetPatchingSpec(
            "causal_entropy_v1", None, 8, artifact_sha256="a" * 64
        )

    chunks = EntropyChunks(_chunks())
    independent_b4 = replace(
        _run_config(recipe="blt_d"),
        patching_policy="causal_entropy_v1",
        corruption=CorruptionConfig(
            kind="blt_exact_k", canvas_length=4, branches_per_row=2_048
        ),
    )
    trainer = ByteDiffusionTrainer(
        ByteDiffusionModel(ByteDiffusionConfig.tiny()),
        DeterministicChunkCursor(chunks, seed=independent_b4.seed),
        chunks,
        independent_b4,
        device=torch.device("cpu"),
    )
    assert trainer.run_config.corruption.canvas_length == 4
    assert chunks.patching.max_patch_size == 8

    chunks.patching = training_module.DatasetPatchingSpec(
        "causal_entropy_v1", None, 8
    )
    matching = replace(
        independent_b4, corruption=CorruptionConfig.blt_entropy_reference()
    )
    with pytest.raises(ValueError, match="authenticated patcher provenance"):
        ByteDiffusionTrainer(
            ByteDiffusionModel(ByteDiffusionConfig.tiny()),
            DeterministicChunkCursor(chunks, seed=matching.seed),
            chunks,
            matching,
            device=torch.device("cpu"),
        )


def test_canonical_preset_rejects_global_batch_drift(monkeypatch) -> None:
    monkeypatch.setenv("BYTE_DIFFUSION_PRESET", CANONICAL_PRESET)
    monkeypatch.setenv("BYTE_DIFFUSION_MICROBATCH", "32")
    monkeypatch.setenv("BYTE_DIFFUSION_GLOBAL_BATCH", "128")
    with pytest.raises(ValueError, match="global_batch_size"):
        TrainingRunConfig.from_env()


def test_canonical_preset_rejects_effective_batch_drift(monkeypatch) -> None:
    monkeypatch.setenv("BYTE_DIFFUSION_PRESET", CANONICAL_PRESET)
    monkeypatch.setenv("BYTE_DIFFUSION_MICROBATCH", "32")
    monkeypatch.setenv("BYTE_DIFFUSION_GRAD_ACCUM", "15")
    with pytest.raises(ValueError, match="GRAD_ACCUM"):
        TrainingRunConfig.from_env()


def test_training_lineage_is_explicitly_scratch() -> None:
    assert TrainingRunConfig().initialization_kind == "scratch"
    with pytest.raises(ValueError, match="scratch lineages only"):
        TrainingRunConfig(initialization_kind="warm_start")  # type: ignore[arg-type]


def test_non_complete_training_rejects_activation_checkpointing() -> None:
    with pytest.raises(ValueError, match="only by the named complete ragged Fast-BLT"):
        TrainingRunConfig(activation_checkpointing=True)


def test_activation_checkpointing_rejects_partial_fast_blt_lookalikes() -> None:
    corruption = CorruptionConfig(
        kind="blt_bernoulli", canvas_length=4, branches_per_row=2_048
    )
    with pytest.raises(ValueError, match="only by the named complete ragged Fast-BLT"):
        TrainingRunConfig(
            preset=None,
            recipe="blt_d",
            patching_policy="causal_entropy_v1",
            blt_origin_policy="all_entropy_patch_starts",
            corruption=corruption,
            activation_checkpointing=True,
        )


def test_canvas_global_stack_honors_activation_checkpointing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).train()
    model.activation_checkpointing = True
    calls: list[tuple[bool, bool]] = []

    def record_checkpoint(function, states, *, use_reentrant, preserve_rng_state):
        calls.append((use_reentrant, preserve_rng_state))
        return function(states)

    monkeypatch.setattr(model_module, "activation_checkpoint", record_checkpoint)
    clean = torch.tensor([[65, 66, 67, 68, 69, 70, 71, 256]])
    valid = torch.ones_like(clean, dtype=torch.bool)
    noisy = torch.tensor([[[261, 261, 71, 256]]])
    branch_valid = torch.ones_like(noisy, dtype=torch.bool)

    model.forward_canvas_branches(
        clean,
        valid,
        noisy,
        branch_valid,
        torch.tensor([[4]]),
    )

    assert calls == [(False, False)]


def test_blt_environment_defaults_to_128_independent_four_byte_blocks(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BYTE_DIFFUSION_RECIPE", "blt_d")
    monkeypatch.delenv("BYTE_DIFFUSION_CORRUPTION", raising=False)
    monkeypatch.delenv("BYTE_DIFFUSION_CANVAS_LENGTH", raising=False)
    monkeypatch.delenv("BYTE_DIFFUSION_BRANCHES", raising=False)
    observed = TrainingRunConfig.from_env()
    assert observed.corruption.kind == "blt_bernoulli"
    assert observed.corruption.canvas_length == 4
    assert observed.corruption.branches_per_row == 128
    assert observed.corruption.corrupted_positions_per_row == 512
    assert observed.global_batch_size == 249


def test_environment_derives_accumulation_for_global_batch(monkeypatch) -> None:
    monkeypatch.setenv("BYTE_DIFFUSION_MICROBATCH", "8")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("BYTE_DIFFUSION_GLOBAL_BATCH", "256")
    monkeypatch.delenv("BYTE_DIFFUSION_GRAD_ACCUM", raising=False)
    assert TrainingRunConfig.from_env().gradient_accumulation == 4

    monkeypatch.setenv("BYTE_DIFFUSION_GRAD_ACCUM", "3")
    with pytest.raises(ValueError, match="GRAD_ACCUM"):
        TrainingRunConfig.from_env()

    monkeypatch.delenv("BYTE_DIFFUSION_GRAD_ACCUM")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("BYTE_DIFFUSION_MICROBATCH", "24")
    monkeypatch.setenv("BYTE_DIFFUSION_GLOBAL_BATCH", "256")
    assert TrainingRunConfig.from_env().gradient_accumulation == 11

    monkeypatch.setenv("BYTE_DIFFUSION_GLOBAL_BATCH", "250")
    assert TrainingRunConfig.from_env().gradient_accumulation == 11

    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("BYTE_DIFFUSION_MICROBATCH", "8")
    monkeypatch.setenv("BYTE_DIFFUSION_GLOBAL_BATCH", "249")
    assert TrainingRunConfig.from_env().gradient_accumulation == 4

    monkeypatch.setenv("BYTE_DIFFUSION_MICROBATCH", "31")
    with pytest.raises(ValueError, match="different backward-call counts"):
        TrainingRunConfig.from_env()


def test_learning_rate_is_flat_then_warms_down_on_planned_updates() -> None:
    assert learning_rate_multiplier(0, 10, 4) == 1.0
    assert learning_rate_multiplier(5, 10, 4) == 1.0
    assert learning_rate_multiplier(6, 10, 4) == 1.0
    assert learning_rate_multiplier(8, 10, 4) == 0.5
    assert learning_rate_multiplier(9, 10, 4) == 0.25
    with pytest.raises(ValueError, match="planned update"):
        learning_rate_multiplier(10, 10, 4)


def test_gradient_interference_reports_norms_and_cosine_without_grad_mutation() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    ar_loss = (parameter.square()).sum()
    diffusion_loss = (-parameter).sum()
    metrics = gradient_interference_metrics(
        ar_loss, diffusion_loss, (parameter,)
    )
    assert metrics.ar_norm == pytest.approx(math.sqrt(20))
    assert metrics.diffusion_norm == pytest.approx(math.sqrt(2))
    assert metrics.cosine == pytest.approx(-3 / math.sqrt(10))
    assert parameter.grad is None


def test_cpu_backend_requires_two_explicit_reference_opt_ins() -> None:
    with pytest.raises(RuntimeError, match="correctness reference"):
        attention_context(
            "dense_reference", torch.device("cpu"), allow_cpu_reference=False
        )
    with pytest.raises(RuntimeError, match="correctness reference"):
        attention_context(
            "flash_sdpa", torch.device("cpu"), allow_cpu_reference=True
        )
    with attention_context(
        "dense_reference", torch.device("cpu"), allow_cpu_reference=True
    ):
        pass


def test_cuda_optimizer_is_declared_fused_and_cpu_is_reference(monkeypatch) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    config = _run_config()
    cpu = create_optimizer([parameter], config, torch.device("cpu"))
    assert cpu.defaults["fused"] is False

    captured = {}

    class FakeAdamW:
        def __init__(self, parameters, **kwargs):
            del parameters
            captured.update(kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", FakeAdamW)
    create_optimizer([parameter], config, torch.device("cuda"))
    assert captured["fused"] is True


def test_nanogpt_muon_optimizer_partitions_every_parameter_once() -> None:
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    config = replace(_run_config(), optimizer_kind="nanogpt_muon")
    optimizer = create_model_optimizer(model, config, torch.device("cpu"))

    groups = {group["tag"]: group for group in optimizer.param_groups}
    assert groups.keys() == {"embedding", "output", "scalar", "matrix"}
    selected = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len({id(parameter) for parameter in selected}) == len(selected)
    assert sum(parameter.numel() for parameter in selected) == model.parameter_count()
    assert groups["embedding"]["base_lr"] == config.embedding_learning_rate
    assert groups["output"]["base_lr"] == config.output_learning_rate
    assert groups["scalar"]["base_lr"] == config.scalar_learning_rate
    assert groups["matrix"]["base_lr"] == config.matrix_learning_rate


def test_nanogpt_muon_schedule_preserves_group_ratios_and_warms_momentum() -> None:
    config = replace(
        _run_config(iterations=1_000),
        optimizer_kind="nanogpt_muon",
        warmdown_iters=100,
        muon_momentum_warmup_steps=500,
    )
    trainer = _trainer(ByteDiffusionModel(ByteDiffusionConfig.tiny()), config)

    assert trainer._set_learning_rate() == config.embedding_learning_rate
    matrix = next(
        group for group in trainer.optimizer.param_groups if group["tag"] == "matrix"
    )
    output = next(
        group for group in trainer.optimizer.param_groups if group["tag"] == "output"
    )
    assert matrix["momentum"] == config.muon_momentum_warmup_start
    assert matrix["lr"] / output["lr"] == pytest.approx(
        config.matrix_learning_rate / config.output_learning_rate
    )

    trainer.completed_steps = 500
    trainer._set_learning_rate()
    assert matrix["momentum"] == config.muon_momentum

    trainer.completed_steps = 950
    trainer._set_learning_rate()
    assert matrix["lr"] == pytest.approx(config.matrix_learning_rate * 0.5)


def test_chunk_batch_uses_stored_halo_targets_and_document_positions() -> None:
    batch = chunks_to_batch(_chunks()[:1])
    assert batch.ids.shape == (1, 8)
    assert batch.ar_targets[0, -1] == -100
    torch.testing.assert_close(batch.positions[0], torch.arange(8))
    assert batch.ar_targets.ne(-100).sum() == 7
    assert batch.bos_targets.tolist() == [ord("A")]
    # The virtual BOS objective closes the codelength gap without shifting the
    # physical four-byte patch phase of the clean document.
    assert int(batch.ar_targets.ne(-100).sum() + batch.bos_targets.ne(-100).sum()) == 8


def test_document_packed_causal_logits_equal_separate_documents() -> None:
    torch.manual_seed(181)
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).eval()
    pad = model.config.vocab.pad_id
    packed_ids = torch.tensor([[65, 66, 256, pad, 70, 71, 72, 256]])
    packed_valid = packed_ids.ne(pad)
    packed_documents = torch.tensor([[0, 0, 0, -1, 1, 1, 1, 1]])
    packed_positions = torch.tensor([[0, 1, 2, 0, 0, 1, 2, 3]])
    layout = training_module._packed_document_layout(
        packed_valid, packed_documents, packed_positions
    )
    packed_logits = model.forward_ar_varlen(
        packed_ids,
        packed_valid,
        positions=packed_positions,
        allow_dense_reference=True,
        **_document_model_metadata(layout),
    ).logits

    separate_ids = torch.tensor(
        [[65, 66, 256, pad], [70, 71, 72, 256]]
    )
    separate_valid = separate_ids.ne(pad)
    separate_positions = torch.tensor([[0, 1, 2, 0], [0, 1, 2, 3]])
    separate_documents = torch.tensor([[0, 0, 0, -1], [1, 1, 1, 1]])
    separate_layout = training_module._packed_document_layout(
        separate_valid, separate_documents, separate_positions
    )
    separate_logits = model.forward_ar_varlen(
        separate_ids,
        separate_valid,
        positions=separate_positions,
        allow_dense_reference=True,
        **_document_model_metadata(separate_layout),
    ).logits

    torch.testing.assert_close(
        packed_logits[0, :3], separate_logits[0, :3], rtol=0, atol=0
    )
    torch.testing.assert_close(
        packed_logits[0, 4:], separate_logits[1], rtol=0, atol=0
    )

    changed_ids = packed_ids.clone()
    changed_ids[0, :3] = torch.tensor([90, 91, 256])
    changed_logits = model.forward_ar_varlen(
        changed_ids,
        packed_valid,
        positions=packed_positions,
        allow_dense_reference=True,
        **_document_model_metadata(layout),
    ).logits
    torch.testing.assert_close(
        packed_logits[0, 4:], changed_logits[0, 4:], rtol=0, atol=0
    )


def test_document_layout_uses_explicit_virtual_bos_priors() -> None:
    valid = torch.ones(1, 16, dtype=torch.bool)
    documents = torch.tensor([[0] * 8 + [1] * 8])
    positions = torch.tensor([[*range(8), *range(8)]])

    layout = training_module._packed_document_layout(
        valid, documents, positions
    )

    assert layout[5].tolist() == [-1, 0, 1, -1, 2, 3]
    assert layout[8].tolist() == [0, 3]
    # The second document's first physical patch has packed-global prior 3,
    # not physical-patch ordinal 2 minus one. Consumers must use this map.
    assert layout[9].tolist() == [[0, 1, 3, 4]]


def test_continuation_page_origin_zero_has_no_synthetic_prior() -> None:
    valid = torch.ones(1, 8, dtype=torch.bool)
    documents = torch.zeros_like(valid, dtype=torch.long)
    positions = torch.arange(8, 16).view(1, -1)
    layout = training_module._packed_document_layout(
        valid, documents, positions
    )
    batch = TrainingBatch(
        ids=torch.arange(8).view(1, -1),
        valid=valid,
        ar_targets=torch.full((1, 8), -100),
        bos_targets=torch.empty(0, dtype=torch.long),
        positions=positions,
        full_valid=True,
        document_ids=documents,
        bos_row_indices=torch.empty(0, dtype=torch.long),
        isolate_documents=True,
        prior_condition_indices=layout[9],
    )

    assert layout[8].numel() == 0
    assert layout[9].tolist() == [[-1, 0]]
    plan = prepare_blt_corruption(
        batch,
        CorruptionConfig(
            kind="blt_bernoulli", canvas_length=4, branches_per_row=2
        ),
        ByteDiffusionConfig.tiny().vocab,
        torch.Generator().manual_seed(3),
    )
    assert plan.block_starts.tolist() == [[4, 0]]
    assert plan.branch_valid.sum((0, 2)).tolist() == [4, 0]
    assert plan.condition_indices.tolist() == [[0, -1]]


@pytest.mark.parametrize("page_width", [8, 16])
def test_continuation_page_rejects_blocks_past_storage(
    page_width: int,
) -> None:
    valid = torch.ones(1, page_width, dtype=torch.bool)
    documents = torch.zeros_like(valid, dtype=torch.long)
    positions = torch.arange(8, 8 + page_width).view(1, -1)
    layout = training_module._packed_document_layout(
        valid, documents, positions
    )

    starts, _, selected, conditions = training_module.sample_blt_patch_starts(
        valid,
        count=page_width // 4,
        patch_stride=4,
        generator=torch.Generator().manual_seed(11),
        document_ids=documents,
        clean_ids=torch.arange(page_width).view(1, -1),
        prior_condition_indices=layout[9],
        block_length=8,
    )

    chosen = starts[selected]
    assert bool((chosen + 8 <= page_width).all())
    assert bool(chosen.ne(0).all())
    assert bool(conditions[selected].ge(0).all())


@pytest.mark.parametrize("block_length", [4, 8, 16])
@pytest.mark.parametrize("ends_with_eot", [False, True])
def test_variable_blt_tail_requires_full_same_document_block_or_eot(
    block_length: int,
    ends_with_eot: bool,
) -> None:
    """Page-continuation PAD is not a semantic short-block terminator."""

    width = 24
    start = 4
    valid_length = start + block_length - 2
    valid = torch.arange(width)[None].lt(valid_length)
    documents = torch.where(valid, torch.zeros_like(valid, dtype=torch.long), -1)
    ids = torch.full((1, width), 262, dtype=torch.long)
    ids[valid] = torch.arange(valid_length, dtype=torch.long)
    if ends_with_eot:
        ids[0, valid_length - 1] = 256
    batch = TrainingBatch(
        ids=ids,
        valid=valid,
        ar_targets=torch.full_like(ids, -100),
        bos_targets=torch.empty(0, dtype=torch.long),
        positions=torch.arange(width)[None],
        full_valid=False,
        document_ids=documents,
        physical_patch_row_indices=torch.tensor([0]),
        physical_patch_start_columns=torch.tensor([start]),
        physical_patch_prior_condition_indices=torch.tensor([0]),
    )

    starts, priors, eligible = training_module._variable_blt_candidate_matrices(
        batch, block_length=block_length
    )

    assert starts.tolist() == [[start]]
    assert priors.tolist() == [[0]]
    assert eligible.tolist() == [[ends_with_eot]]


@pytest.mark.parametrize("block_length", [4, 8, 16])
def test_variable_blt_full_same_document_block_remains_eligible(
    block_length: int,
) -> None:
    width = 24
    start = 4
    valid = torch.ones((1, width), dtype=torch.bool)
    ids = torch.arange(width, dtype=torch.long)[None]
    batch = TrainingBatch(
        ids=ids,
        valid=valid,
        ar_targets=torch.full_like(ids, -100),
        bos_targets=torch.empty(0, dtype=torch.long),
        positions=torch.arange(width)[None],
        full_valid=True,
        document_ids=torch.zeros_like(ids),
        physical_patch_row_indices=torch.tensor([0]),
        physical_patch_start_columns=torch.tensor([start]),
        physical_patch_prior_condition_indices=torch.tensor([0]),
    )

    _, _, eligible = training_module._variable_blt_candidate_matrices(
        batch, block_length=block_length
    )

    assert eligible.tolist() == [[True]]


def test_packed_bos_supervises_every_document_start_in_one_row() -> None:
    manifest = AtomicIdManifest.reference()
    chunks = pack_documents(
        (
            AtomicDocument("a", (65, 66, 67, manifest.eot_id)),
            AtomicDocument("b", (70, 71, 72, manifest.eot_id)),
        ),
        manifest,
        chunk_size=8,
    )
    batch = chunks_to_batch(chunks)

    assert batch.bos_targets.tolist() == [65, 70]
    assert batch.bos_row_indices is not None
    assert batch.bos_row_indices.tolist() == [0, 0]
    assert int(batch.ar_targets.ne(-100).sum()) == 6
    assert int(batch.ar_targets.ne(-100).sum() + batch.bos_targets.numel()) == 8


def test_v5_loader_rejects_nonzero_offset_at_midpage_document_start(
    tmp_path: Path,
) -> None:
    arrays = _valid_document_aligned_artifact()
    arrays["document_offsets"][0, 4:] = np.asarray([7, 8, 9, 10])
    arrays["patch_offsets"][0, 4:] = np.asarray([3, 0, 1, 2])

    with pytest.raises(ValueError):
        _load_document_aligned_artifact(tmp_path, arrays)


def test_v5_loader_rejects_unscored_terminal_atom_without_eot_or_halo(
    tmp_path: Path,
) -> None:
    arrays = _valid_document_aligned_artifact()
    arrays["input_ids"][0, 7] = 73
    arrays["target_ids"][0, 6] = 73

    with pytest.raises(ValueError):
        _load_document_aligned_artifact(tmp_path, arrays)


@pytest.mark.parametrize(
    "mutation",
    (
        "invalid_input",
        "invalid_target",
        "invalid_metadata",
        "out_of_range_input",
        "out_of_range_target",
    ),
)
def test_v5_loader_rejects_invalid_slot_payloads_and_out_of_range_ids(
    tmp_path: Path,
    mutation: str,
) -> None:
    arrays = _valid_document_aligned_artifact()
    if mutation == "invalid_input":
        arrays["input_ids"][0, 2] = 42
    elif mutation == "invalid_target":
        arrays["target_ids"][0, 2] = 42
    elif mutation == "invalid_metadata":
        arrays["document_indices"][0, 2] = 0
        arrays["document_offsets"][0, 2] = 2
        arrays["patch_offsets"][0, 2] = 2
    elif mutation == "out_of_range_input":
        arrays["input_ids"][0, 4] = 500
    elif mutation == "out_of_range_target":
        arrays["target_ids"][0, 4] = 500
    else:
        raise AssertionError(f"unknown mutation {mutation!r}")

    with pytest.raises(ValueError):
        _load_document_aligned_artifact(tmp_path, arrays)


def test_v5_deferred_loader_validates_semantics_on_first_use(tmp_path: Path) -> None:
    arrays = _valid_document_aligned_artifact()
    arrays["patch_offsets"][0, 4:] = np.asarray([1, 2, 3, 0])
    dataset = _load_document_aligned_artifact(
        tmp_path, arrays, defer_payload_validation=True
    )

    assert not dataset._semantically_validated_artifacts
    with pytest.raises(ValueError, match="patch phase"):
        dataset.training_batch([0])


def test_v5_pinned_training_hashes_each_dense_artifact_once_without_semantic_scan(
    tmp_path: Path, monkeypatch
) -> None:
    arrays = _valid_document_aligned_artifact()
    arrays["patch_offsets"][0, 4:] = np.asarray([1, 2, 3, 0])
    dataset = _load_document_aligned_artifact(
        tmp_path,
        arrays,
        defer_payload_validation=True,
        trust_pinned_row_index=True,
    )

    def unexpected_scan(*_args, **_kwargs):
        raise AssertionError("trusted pinned payload performed a semantic scan")

    monkeypatch.setattr(dataset, "_validate_deferred_artifact", unexpected_scan)
    monkeypatch.setattr(
        dataset,
        "_verify_selected_rows",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("dense training used scalar row hashes")
        ),
    )
    batch = dataset.training_batch([0])
    dataset.training_batch([0])
    assert batch.ids.shape == (1, 8)
    assert dataset._verified_artifacts == {0}
    assert dataset._verified_rows[0].tolist() == [False]
    assert not dataset._semantically_validated_artifacts


def test_v5_pinned_loader_still_validates_identity_arrays(tmp_path: Path) -> None:
    arrays = _valid_document_aligned_artifact()

    def break_stream_shape(artifact: dict) -> None:
        artifact["arrays"]["stream_start"]["shape"] = [2]
        artifact["arrays"]["stream_start"]["byte_length"] = 16

    with pytest.raises(ValueError, match="stream_start"):
        _load_document_aligned_artifact(
            tmp_path,
            arrays,
            defer_payload_validation=True,
            trust_pinned_row_index=True,
            mutate_artifact=break_stream_shape,
        )


def test_v5_deferred_loader_hashes_artifact_before_first_use(tmp_path: Path) -> None:
    dataset = _load_document_aligned_artifact(
        tmp_path,
        _valid_document_aligned_artifact(),
        defer_payload_validation=True,
    )
    path = tmp_path / "train-00000.bdm"
    corrupted = bytearray(path.read_bytes())
    corrupted[len(corrupted) // 2] ^= 1
    path.write_bytes(corrupted)

    with pytest.raises(ValueError, match="sha256 mismatch"):
        dataset.training_batch([0])


def test_v5_pinned_training_rejects_corrupt_artifact(
    tmp_path: Path,
) -> None:
    dataset = _load_document_aligned_artifact(
        tmp_path,
        _valid_document_aligned_artifact(),
        defer_payload_validation=True,
        trust_pinned_row_index=True,
    )
    descriptor = dataset.artifacts[0]["arrays"]["input_ids"]
    path = tmp_path / "train-00000.bdm"
    with path.open("r+b") as handle:
        handle.seek(int(descriptor["byte_offset"]))
        original = handle.read(1)
        handle.seek(int(descriptor["byte_offset"]))
        handle.write(bytes([original[0] ^ 1]))

    with pytest.raises(ValueError, match="artifact sha256 mismatch"):
        dataset.training_batch([0])


def test_v5_pinned_validation_verifies_sparse_rows_without_full_hash(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = _load_document_aligned_artifact(
        tmp_path,
        _valid_document_aligned_artifact(),
        defer_payload_validation=True,
        trust_pinned_row_index=True,
    )
    descriptor = dataset.artifacts[0]["arrays"]["input_ids"]
    path = tmp_path / "train-00000.bdm"
    with path.open("r+b") as handle:
        handle.seek(int(descriptor["byte_offset"]))
        original = handle.read(1)
        handle.seek(int(descriptor["byte_offset"]))
        handle.write(bytes([original[0] ^ 1]))
    monkeypatch.setattr(
        dataset,
        "_sha256_file",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("trusted pinned payload hashed the whole artifact")
        ),
    )

    with pytest.raises(ValueError, match="row sha256 mismatch"):
        dataset.validation_batch([0])


def test_v5_loader_rejects_replacement_after_cache_eviction(tmp_path: Path) -> None:
    dataset = _load_document_aligned_artifact(
        tmp_path,
        _valid_document_aligned_artifact(),
        defer_payload_validation=True,
    )
    dataset.training_batch([0])
    path = tmp_path / "train-00000.bdm"
    corrupted = bytearray(path.read_bytes())
    corrupted[len(corrupted) // 2] ^= 1
    path.write_bytes(corrupted)
    dataset._arrays.cache_clear()
    dataset._batch_arrays.cache_clear()

    with pytest.raises(ValueError, match="changed after verification"):
        dataset.training_batch([0])


def test_v5_loader_keeps_compact_row_index(tmp_path: Path) -> None:
    dataset = _load_document_aligned_artifact(
        tmp_path,
        _valid_document_aligned_artifact(),
        defer_payload_validation=True,
    )

    assert not hasattr(dataset, "_selected")
    assert not hasattr(dataset, "_groups")
    assert isinstance(dataset._valid_counts, np.ndarray)
    assert dataset._valid_counts.dtype == np.int32
    assert dataset._physical_extents.dtype == np.int32
    assert dataset._row_document_starts[0].dtype == np.int32
    assert dataset._row_digests[0].dtype == np.uint8
    assert dataset._row_digests[0].shape == (1, 32)
    assert "row_valid_counts" not in dataset.artifacts[0]
    assert "row_physical_extents" not in dataset.artifacts[0]
    assert "row_document_starts" not in dataset.artifacts[0]
    assert "row_sha256" not in dataset.artifacts[0]
    assert dataset._resolve_index(0) == (0, 0)


def test_mapped_loader_selects_32_rows_without_materializing_shard(
    tmp_path: Path,
) -> None:
    source = _valid_document_aligned_artifact()
    arrays = {
        name: np.repeat(array, 64, axis=0)
        for name, array in source.items()
    }
    arrays["chunk_index"] = np.arange(64, dtype="<i8")
    arrays["stream_start"] = np.arange(64, dtype="<i8") * 8
    arrays["stream_stop"] = arrays["stream_start"] + 8
    dataset = _load_document_aligned_artifact(
        tmp_path,
        arrays,
        defer_payload_validation=True,
        trust_pinned_row_index=True,
    )

    batch = dataset.training_batch(list(range(32)))
    mapped = dataset._batch_arrays(0)
    assert all(isinstance(array, np.memmap) for array in mapped.values())
    assert mapped["input_ids"].shape == (64, 8)
    assert batch.ids.shape == (32, 8)
    assert batch.ids.numel() == mapped["input_ids"].size // 2


def test_loader_retains_short_origin_halo_for_b8_b16(tmp_path: Path) -> None:
    arrays = _valid_document_aligned_artifact()
    pad_values = {
        "input_ids": AtomicIdManifest.reference().pad_id,
        "target_ids": AtomicIdManifest.reference().pad_id,
        "valid_mask": False,
        "score_mask": False,
        "document_indices": -1,
        "document_offsets": -1,
        "patch_offsets": -1,
    }
    for name, fill in pad_values.items():
        arrays[name] = np.pad(
            arrays[name], ((0, 0), (0, 8)), constant_values=fill
        )

    b4 = _load_document_aligned_artifact(
        tmp_path / "b4", arrays, branch_span_length=4
    )
    b8 = _load_document_aligned_artifact(
        tmp_path / "b8", arrays, branch_span_length=8
    )
    b16 = _load_document_aligned_artifact(
        tmp_path / "b16", arrays, branch_span_length=16
    )

    assert b4.training_batch([0]).ids.shape[1] == 8
    assert b8.training_batch([0]).ids.shape[1] == 12
    assert b16.training_batch([0]).ids.shape[1] == 16


def test_distributed_cursor_sharding_is_disjoint_and_cursor_identical() -> None:
    chunks = _chunks()
    left = DeterministicChunkCursor(chunks, seed=3, shuffle=False)
    right = DeterministicChunkCursor(chunks, seed=3, shuffle=False)
    rank0 = take_distributed_chunks(
        left, 2, DistributedContext(rank=0, local_rank=0, world_size=2)
    )
    rank1 = take_distributed_chunks(
        right, 2, DistributedContext(rank=1, local_rank=1, world_size=2)
    )
    assert [chunk.chunk_index for chunk in rank0] == [0, 2]
    assert [chunk.chunk_index for chunk in rank1] == [1, 3]
    assert left.state_dict() == right.state_dict()


def test_exact_249_page_schedule_is_balanced_rotating_and_cursor_identical() -> None:
    chunks = tuple(_chunks()[0] for _ in range(300))
    cursors = [
        DeterministicChunkCursor(chunks, seed=3, shuffle=False) for _ in range(8)
    ]
    first = [
        take_uneven_distributed_indices(
            cursors[rank],
            249,
            DistributedContext(rank=rank, local_rank=rank, world_size=8),
            rotation=0,
        )
        for rank in range(8)
    ]
    assert [len(indices) for indices in first] == [32, 31, 31, 31, 31, 31, 31, 31]
    assert sorted(index for indices in first for index in indices) == list(range(249))
    assert all(cursor.state_dict() == cursors[0].state_dict() for cursor in cursors)

    second = [
        take_uneven_distributed_indices(
            cursors[rank],
            249,
            DistributedContext(rank=rank, local_rank=rank, world_size=8),
            rotation=1,
        )
        for rank in range(8)
    ]
    assert [len(indices) for indices in second] == [31, 32, 31, 31, 31, 31, 31, 31]
    assert all(-(-len(indices) // 8) == 4 for indices in first + second)
    assert all(cursor.state_dict() == cursors[0].state_dict() for cursor in cursors)


def test_validation_subset_evenly_covers_and_binds_the_full_split() -> None:
    source = tuple(_chunks(offset=10 * group)[0] for group in range(10))
    subset = DeterministicSubsetChunkDataset(source, 3)
    np.testing.assert_array_equal(subset.indices, np.array([1, 5, 8]))
    assert all(
        subset[index] is source[position]
        for index, position in enumerate(subset.indices)
    )
    repeated = DeterministicSubsetChunkDataset(source, 3)
    assert repeated.dataset_sha256 == subset.dataset_sha256
    changed = DeterministicSubsetChunkDataset(source, 4)
    assert changed.dataset_sha256 != subset.dataset_sha256


def test_validation_subset_preserves_authenticated_patching_contract() -> None:
    class SourceWithPatching:
        def __init__(self) -> None:
            self.rows = tuple(_chunks(offset=10 * group)[0] for group in range(10))
            self.patching = object()
            self.chunk_size = 8

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            return self.rows[index]

        def training_blt_origin_counts(self, indices, *, block_length, eot_id):
            assert block_length == 4
            assert eot_id == 256
            return np.asarray(indices, dtype=np.int64) + 1

    source = SourceWithPatching()
    subset = DeterministicSubsetChunkDataset(source, 3)
    assert subset.patching is source.patching
    assert subset.chunk_size == 8
    np.testing.assert_array_equal(
        subset.training_blt_origin_counts((0, 2), block_length=4, eot_id=256),
        subset.indices[[0, 2]] + 1,
    )


def test_mixed_scratch_update_has_both_losses_and_changes_parameters() -> None:
    torch.manual_seed(7)
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    trainer = _trainer(model, _run_config(recipe="canvas"))
    before = model.output.weight.detach().clone()
    metrics = trainer.run_update()
    assert math.isfinite(metrics.total)
    assert metrics.ar > 0
    assert metrics.diffusion > 0
    assert metrics.ar_targets == 8
    assert 1 <= metrics.diffusion_targets <= 8
    assert not torch.equal(before, model.output.weight)


def test_global_batch_uses_a_smaller_tail_microbatch_without_extra_rows() -> None:
    torch.manual_seed(9)
    config = replace(
        _run_config(recipe="canvas", iterations=1),
        microbatch_per_rank=3,
        gradient_accumulation=2,
        global_batch_size=4,
    )
    trainer = _trainer(ByteDiffusionModel(ByteDiffusionConfig.tiny()), config)

    metrics = trainer.run_update()

    assert metrics.microsteps == 2
    assert metrics.max_microbatch == 3
    assert metrics.ar_targets == 32
    assert trainer.train_cursor.epoch == 0
    assert trainer.train_cursor.position == 4


def test_tail_microbatch_matches_fixed_partition_gradient_for_causal_loss() -> None:
    torch.manual_seed(101)
    fixed_model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    tail_model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    tail_model.load_state_dict(fixed_model.state_dict())
    base = _run_config(recipe="causal_only", iterations=1)
    fixed = replace(
        base,
        microbatch_per_rank=4,
        gradient_accumulation=1,
        global_batch_size=4,
    )
    tailed = replace(
        base,
        microbatch_per_rank=3,
        gradient_accumulation=2,
        global_batch_size=4,
    )

    fixed_metrics = _trainer(fixed_model, fixed).run_update()
    tail_metrics = _trainer(tail_model, tailed).run_update()

    assert tail_metrics.ar == pytest.approx(fixed_metrics.ar, abs=1e-6)
    for fixed_parameter, tail_parameter in zip(
        fixed_model.parameters(), tail_model.parameters(), strict=True
    ):
        torch.testing.assert_close(fixed_parameter, tail_parameter, atol=3e-6, rtol=1e-6)


def test_packed_causal_loss_and_gradients_match_padded_ragged_reference() -> None:
    pad = ByteDiffusionConfig.tiny().vocab.pad_id
    batch = TrainingBatch(
        ids=torch.tensor(
            [
                [65, 66, 256, pad, pad, pad, pad, pad],
                [70, 71, 72, 73, 74, 75, 76, 77],
            ]
        ),
        valid=torch.tensor(
            [
                [True, True, True, False, False, False, False, False],
                [True, True, True, True, True, True, True, True],
            ]
        ),
        ar_targets=torch.tensor(
            [
                [66, 256, -100, -100, -100, -100, -100, -100],
                [71, 72, 73, 74, 75, 76, 77, 259],
            ]
        ),
        # The second row is a mid-document continuation: it has a stored
        # final-position halo target but deliberately no synthetic BOS target.
        bos_targets=torch.tensor([65]),
        positions=torch.tensor(
            [
                [0, 1, 2, 0, 0, 0, 0, 0],
                [100, 101, 102, 103, 104, 105, 106, 107],
            ]
        ),
        full_valid=False,
        bos_row_indices=torch.tensor([0]),
    )
    assert int(batch.ar_targets[1, -1]) == 259
    assert batch.bos_targets.tolist() == [65]
    torch.manual_seed(109)
    packed_model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    padded_model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    padded_model.load_state_dict(packed_model.state_dict())
    config = _run_config(recipe="causal_only", iterations=1)
    trainer = _trainer(packed_model, config)

    packed_loss = trainer._compute_loss(batch).total
    padded_output = padded_model.forward_ar_varlen(
        batch.ids, batch.valid, positions=batch.positions,
        allow_dense_reference=True,
    )
    padded_bos_state = padded_model.virtual_bos_global_states(
        batch.bos_targets.shape[0],
        device=batch.ids.device,
        allow_dense_reference=True,
    )
    padded_bos = padded_model.forward_bos_logits(
        padded_bos_state, allow_dense_reference=True
    )
    ar_total = F.cross_entropy(
        padded_output.logits.flatten(0, 1),
        batch.ar_targets.flatten(),
        ignore_index=-100,
        reduction="sum",
    )
    bos_total = F.cross_entropy(
        padded_bos,
        batch.bos_targets,
        ignore_index=-100,
        reduction="sum",
    )
    target_count = batch.ar_targets.ne(-100).sum() + batch.bos_targets.numel()
    padded_loss = config.lambda_ar * (ar_total + bos_total) / target_count

    torch.testing.assert_close(packed_loss, padded_loss, rtol=1e-6, atol=1e-7)
    packed_loss.backward()
    padded_loss.backward()
    for packed_parameter, padded_parameter in zip(
        packed_model.parameters(), padded_model.parameters(), strict=True
    ):
        if packed_parameter.grad is None or padded_parameter.grad is None:
            assert packed_parameter.grad is None and padded_parameter.grad is None
        else:
            torch.testing.assert_close(
                packed_parameter.grad,
                padded_parameter.grad,
                rtol=2e-5,
                atol=2e-7,
            )


def test_production_24_plus_16_canvas_partition_matches_fixed_partition(
    monkeypatch,
) -> None:
    manifest = AtomicIdManifest.reference()
    chunks = tuple(
        pack_documents(
            (
                AtomicDocument(
                    f"row-{row}",
                    (row % 256, 17, 33, 49, 65, 81, 97, manifest.eot_id),
                ),
            ),
            manifest,
            chunk_size=8,
        )[0]
        for row in range(256)
    )

    def deterministic_canvas(batch, config, vocab, generator):
        del generator
        clean = batch.ids[:, None, : config.canvas_length]
        branch_valid = batch.valid[:, None, : config.canvas_length]
        limits = batch.ids[:, :1].remainder(config.canvas_length).add(1)
        active = branch_valid & (
            torch.arange(config.canvas_length)[None, None, :] < limits[:, :, None]
        )
        return CanvasCorruptionPlan(
            clean_branches=clean,
            noisy_branches=torch.where(active, vocab.mask_id, clean),
            branch_valid=branch_valid,
            branch_starts=torch.zeros((batch.ids.shape[0], 1), dtype=torch.long),
            active=active,
        )

    monkeypatch.setattr(
        training_module, "prepare_canvas_corruption", deterministic_canvas
    )
    base = replace(
        _run_config(recipe="canvas", iterations=1),
        max_grad_norm=None,
    )
    tail_config = replace(
        base,
        microbatch_per_rank=24,
        gradient_accumulation=11,
        global_batch_size=256,
    )
    fixed_config = replace(
        base,
        microbatch_per_rank=16,
        gradient_accumulation=16,
        global_batch_size=256,
    )
    torch.manual_seed(103)
    initial = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    tail_model = copy.deepcopy(initial)
    fixed_model = copy.deepcopy(initial)
    batch_sizes = []
    original_collate = training_module.chunks_to_batch

    def recording_collate(selected):
        batch_sizes.append(len(selected))
        return original_collate(selected)

    monkeypatch.setattr(training_module, "chunks_to_batch", recording_collate)
    tail = _trainer(tail_model, tail_config, chunks)
    tail_metrics = tail.run_update()
    monkeypatch.setattr(training_module, "chunks_to_batch", original_collate)
    fixed_metrics = _trainer(fixed_model, fixed_config, chunks).run_update()

    assert batch_sizes == [24] * 10 + [16]
    assert tail.train_cursor.epoch == 0
    assert tail.train_cursor.position == 256
    assert tail_metrics.ar_targets == fixed_metrics.ar_targets == 2_048
    assert tail_metrics.diffusion_targets == fixed_metrics.diffusion_targets
    assert tail_metrics.total == pytest.approx(fixed_metrics.total, abs=2e-6)
    assert tail_metrics.ar == pytest.approx(fixed_metrics.ar, abs=2e-6)
    assert tail_metrics.diffusion == pytest.approx(
        fixed_metrics.diffusion, abs=2e-6
    )
    for tail_parameter, fixed_parameter in zip(
        tail_model.parameters(), fixed_model.parameters(), strict=True
    ):
        torch.testing.assert_close(
            tail_parameter, fixed_parameter, atol=2e-6, rtol=2e-6
        )


def test_canvas_plan_slices_nonoverlapping_patch_aligned_branches() -> None:
    batch = chunks_to_batch(_chunks()[:1])
    config = CorruptionConfig(
        kind="absorbing_rb", canvas_length=4, branches_per_row=2
    )
    plan = prepare_canvas_corruption(
        batch,
        config,
        ByteDiffusionConfig.tiny().vocab,
        torch.Generator().manual_seed(31),
    )
    assert plan.clean_branches.shape == (1, 2, 4)
    assert plan.noisy_branches.shape == (1, 2, 4)
    assert plan.branch_valid.shape == (1, 2, 4)
    assert bool((plan.branch_starts % 4 == 0).all())
    assert abs(int(plan.branch_starts[0, 1] - plan.branch_starts[0, 0])) >= 4
    assert not bool((plan.active & ~plan.branch_valid).any())
    assert bool(
        (
            plan.noisy_branches[plan.active]
            == ByteDiffusionConfig.tiny().vocab.mask_id
        ).all()
    )


def test_canvas_update_uses_one_shared_prefix_forward_for_multiple_branches() -> None:
    class CountingModel(ByteDiffusionModel):
        def __init__(self):
            super().__init__(ByteDiffusionConfig.tiny())
            self.ordinary_forwards = 0
            self.canvas_forwards = 0

        def forward(self, *args, **kwargs):
            self.ordinary_forwards += 1
            return super().forward(*args, **kwargs)

        def forward_canvas_branches(self, *args, **kwargs):
            self.canvas_forwards += 1
            return super().forward_canvas_branches(*args, **kwargs)

    config = _run_config()
    config = TrainingRunConfig(
        **{
            **config.contract_dict(),
            "corruption": CorruptionConfig(
                kind="absorbing_rb", canvas_length=4, branches_per_row=2
            ),
        }
    )
    model = CountingModel()
    trainer = _trainer(model, config)
    trainer.run_update()
    assert model.canvas_forwards == 1
    assert model.ordinary_forwards == 0


def test_blt_plan_samples_reference_patch_origins_with_ht_weight() -> None:
    batch = chunks_to_batch(_chunks()[:1])
    config = CorruptionConfig(
        kind="absorbing_rb", canvas_length=4, branches_per_row=2
    )
    plan = prepare_blt_corruption(
        batch,
        config,
        ByteDiffusionConfig.tiny().vocab,
        torch.Generator().manual_seed(37),
    )
    assert plan.block_starts.shape == (1, 2)
    assert bool((plan.block_starts % 4 == 0).all())
    assert plan.block_starts.tolist() == [[0, 4]]
    torch.testing.assert_close(plan.sampling_weight, torch.tensor([1.0]))
    assert plan.branch_valid.sum((0, 2)).tolist() == [4, 4]
    assert bool(plan.condition_indices.ge(0).all())
    assert not bool((plan.active & ~plan.branch_valid).any())
    torch.testing.assert_close(
        plan.noisy_blocks[~plan.active], plan.clean_blocks[~plan.active]
    )


def test_exhaustive_entropy_plan_is_exact_ragged_m_minus_one_population() -> None:
    batch = _hand_enumerated_entropy_batch()
    config = CorruptionConfig(
        kind="blt_bernoulli", canvas_length=4, branches_per_row=2_048
    )
    row_t = torch.tensor([0.25, 0.75])
    generator = torch.Generator().manual_seed(101)
    plan = training_module.prepare_exhaustive_blt_corruption(
        batch,
        config,
        ByteDiffusionConfig.tiny().vocab,
        generator,
        row_t=row_t,
    )

    assert plan.clean_blocks.shape == (6, 4)
    assert plan.noisy_blocks.shape == plan.clean_blocks.shape
    assert plan.block_valid.shape == plan.clean_blocks.shape
    assert plan.active.shape == plan.clean_blocks.shape
    assert plan.block_rows.tolist() == [0, 0, 0, 1, 1, 1]
    assert plan.block_starts.tolist() == [0, 3, 5, 0, 2, 6]
    assert batch.document_ids[plan.block_rows, plan.block_starts].tolist() == [
        0,
        1,
        1,
        2,
        2,
        2,
    ]
    # Each document has one virtual BOS latent plus 1, 2, and 3 physical
    # patches respectively, so all M-1 physical starts total six blocks.
    assert plan.clean_blocks.shape[0] == (2 - 1) + (3 - 1) + (4 - 1)
    assert plan.block_starts.unique().numel() < plan.block_starts.numel()
    assert len(set(zip(plan.block_rows.tolist(), plan.block_starts.tolist()))) == 6
    assert plan.condition_indices.tolist() == (
        batch.physical_patch_prior_condition_indices.tolist()
    )
    assert bool(plan.condition_indices.ge(0).all())
    assert plan.row_cu_seqlens.tolist() == [0, 3, 6]
    assert plan.row_block_counts.tolist() == [3, 3]
    assert plan.block_valid.sum(1).tolist() == [3, 4, 3, 4, 4, 2]
    assert plan.clean_blocks[0].tolist() == [65, 66, 256, 262]
    assert plan.clean_blocks[2].tolist() == [72, 73, 256, 262]
    assert plan.clean_blocks[5].tolist() == [86, 256, 262, 262]
    torch.testing.assert_close(plan.row_t, row_t)
    torch.testing.assert_close(
        plan.block_t, torch.tensor([0.25, 0.25, 0.25, 0.75, 0.75, 0.75])
    )
    assert plan.row_physical_positions.tolist() == [20, 20]
    assert not hasattr(plan, "sampling_weight")


def test_exhaustive_entropy_topology_consumes_only_bernoulli_rng() -> None:
    batch = _hand_enumerated_entropy_batch()
    config = CorruptionConfig(
        kind="blt_bernoulli", canvas_length=4, branches_per_row=2_048
    )
    row_t = torch.tensor([0.25, 0.75])
    left_generator = torch.Generator().manual_seed(211)
    right_generator = torch.Generator().manual_seed(307)
    left = training_module.prepare_exhaustive_blt_corruption(
        batch,
        config,
        ByteDiffusionConfig.tiny().vocab,
        left_generator,
        row_t=row_t,
    )
    right = training_module.prepare_exhaustive_blt_corruption(
        batch,
        config,
        ByteDiffusionConfig.tiny().vocab,
        right_generator,
        row_t=row_t,
    )

    for name in (
        "clean_blocks",
        "block_valid",
        "block_rows",
        "block_starts",
        "condition_indices",
        "row_cu_seqlens",
        "row_physical_positions",
    ):
        torch.testing.assert_close(getattr(left, name), getattr(right, name))

    expected_generator = torch.Generator().manual_seed(211)
    torch.rand(left.clean_blocks.shape, generator=expected_generator)
    torch.testing.assert_close(left_generator.get_state(), expected_generator.get_state())


def test_stateless_exhaustive_noise_is_invariant_to_row_grouping() -> None:
    keys = torch.tensor([101, 202], dtype=torch.long)
    rows = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    starts = torch.tensor([3, 9, 1, 4, 12], dtype=torch.long)
    full_t, full_uniforms = training_module._stateless_exhaustive_blt_uniforms(
        keys, rows, starts, block_length=4, seed=1337
    )
    first_t, first_uniforms = training_module._stateless_exhaustive_blt_uniforms(
        keys[:1], torch.zeros(2, dtype=torch.long), starts[:2],
        block_length=4, seed=1337,
    )
    second_t, second_uniforms = training_module._stateless_exhaustive_blt_uniforms(
        keys[1:], torch.zeros(3, dtype=torch.long), starts[2:],
        block_length=4, seed=1337,
    )
    torch.testing.assert_close(full_t, torch.cat((first_t, second_t)), rtol=0, atol=0)
    torch.testing.assert_close(
        full_uniforms, torch.cat((first_uniforms, second_uniforms)), rtol=0, atol=0
    )
    assert bool(((full_t > 0) & (full_t < 1)).all())
    assert bool(((full_uniforms > 0) & (full_uniforms < 1)).all())


def test_stateless_exhaustive_plan_does_not_consume_generator_state() -> None:
    batch = _hand_enumerated_entropy_batch()
    generator = torch.Generator().manual_seed(701)
    before = generator.get_state().clone()
    identities = ((11, 101), (12, 202))
    plan = training_module.prepare_exhaustive_blt_corruption(
        batch,
        CorruptionConfig(
            kind="blt_bernoulli", canvas_length=4, branches_per_row=2_048
        ),
        ByteDiffusionConfig.tiny().vocab,
        generator,
        stateless_row_keys=training_module.validation_row_identity_keys(identities),
        stateless_seed=1337 + 104_729,
    )
    torch.testing.assert_close(generator.get_state(), before)
    assert plan.row_t.shape == (2,)
    assert plan.active.shape == plan.clean_blocks.shape
    torch.testing.assert_close(
        plan.active_flat_indices,
        torch.nonzero(plan.active.reshape(-1), as_tuple=False).flatten(),
    )
    torch.testing.assert_close(
        plan.active_targets,
        plan.clean_blocks.reshape(-1).index_select(0, plan.active_flat_indices),
    )
    with pytest.raises(ValueError, match="indices do not match"):
        replace(
            plan,
            active_flat_indices=plan.active_flat_indices.flip(0),
        )
    with pytest.raises(ValueError, match="unique keys"):
        training_module.validation_row_identity_keys(((11, 101), (11, 101)))


def test_exhaustive_entropy_paper_sum_reduces_blocks_to_clean_rows() -> None:
    batch = _hand_enumerated_entropy_batch()
    plan = training_module.prepare_exhaustive_blt_corruption(
        batch,
        CorruptionConfig(
            kind="blt_bernoulli", canvas_length=4, branches_per_row=2_048
        ),
        ByteDiffusionConfig.tiny().vocab,
        torch.Generator().manual_seed(401),
        row_t=torch.tensor([0.25, 0.75]),
    )
    all_active_indices = torch.nonzero(
        plan.block_valid.reshape(-1), as_tuple=False
    ).flatten()
    all_active = replace(
        plan,
        active=plan.block_valid,
        active_flat_indices=all_active_indices,
        active_targets=plan.clean_blocks.reshape(-1).index_select(
            0, all_active_indices
        ),
    )
    rows = training_module.exhaustive_blt_paper_sum_rows(
        torch.ones_like(plan.clean_blocks, dtype=torch.float32), all_active
    )
    torch.testing.assert_close(rows, torch.tensor([40.0, 40.0 / 3.0]))

    none_active = replace(
        plan,
        active=torch.zeros_like(plan.active),
        active_flat_indices=torch.empty(0, dtype=torch.long),
        active_targets=torch.empty(0, dtype=torch.long),
    )
    zero = training_module.exhaustive_blt_paper_sum_rows(
        torch.full_like(plan.clean_blocks, float("inf"), dtype=torch.float32),
        none_active,
    )
    torch.testing.assert_close(zero, torch.zeros(2))
    assert bool(torch.isfinite(zero).all())

    active_nll = torch.arange(
        1,
        plan.active_flat_indices.numel() + 1,
        dtype=torch.float32,
    )
    dense_nll = torch.zeros_like(plan.clean_blocks, dtype=torch.float32)
    dense_nll.reshape(-1).index_copy_(
        0, plan.active_flat_indices, active_nll
    )
    torch.testing.assert_close(
        training_module.exhaustive_blt_active_paper_sum_rows(active_nll, plan),
        training_module.exhaustive_blt_paper_sum_rows(dense_nll, plan),
    )


def test_joint_forward_and_trainer_loss_consume_exhaustive_ragged_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EntropyChunks(tuple):
        patching = DatasetPatchingSpec(
            "causal_entropy_v1", None, 4, "1" * 64
        )

    batch = replace(
        _hand_enumerated_entropy_batch(),
        ar_targets=torch.tensor(
            [
                [66, 256, -100, 71, 72, 73, 256, -100],
                [81, 82, 83, 84, 85, 86, 256, -100],
            ]
        ),
    )
    plan = training_module.prepare_exhaustive_blt_corruption(
        batch,
        CorruptionConfig(
            kind="blt_bernoulli", canvas_length=4, branches_per_row=2_048
        ),
        ByteDiffusionConfig.tiny().vocab,
        torch.Generator().manual_seed(503),
        row_t=torch.tensor([0.25, 0.75]),
    )
    model_config = replace(
        ByteDiffusionConfig.tiny(),
        decoder_conditioning="split_cross_attention",
        decoder_prefix_window=None,
    )
    run = TrainingRunConfig(
        iterations=1,
        val_loss_every=1,
        train_log_every=1,
        validation_chunks=2,
        validation_microbatch_per_rank=1,
        ar_validation_microbatch_per_rank=1,
        diffusion_validation_chunks=2,
        warmdown_iters=0,
        run_id="ragged-loss-oracle",
        recipe="blt_d",
        patching_policy="causal_entropy_v1",
        blt_origin_policy="all_entropy_patch_starts",
        corruption=CorruptionConfig(
            kind="blt_bernoulli", canvas_length=4, branches_per_row=2_048
        ),
        microbatch_per_rank=1,
        microbatch_token_budget=128,
        gradient_accumulation=1,
        global_batch_size=1,
        objective_reduction="paper_sum",
        attention_policy="dense_reference",
        allow_cpu_reference=True,
        compile_model=False,
    )
    chunks = EntropyChunks(_chunks()[:2])
    trainer = ByteDiffusionTrainer(
        ByteDiffusionModel(model_config),
        DeterministicChunkCursor(chunks, seed=run.seed),
        chunks,
        run,
        device=torch.device("cpu"),
    )
    dense_oracle = copy.deepcopy(trainer.joint.model)
    original_linear_ce = F.linear_cross_entropy
    projected_rows: list[tuple[int, int]] = []

    def record_linear_ce(
        input: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        projected_rows.append(
            (input.shape[0], int(target.ne(kwargs.get("ignore_index", -1)).sum()))
        )
        return original_linear_ce(input, weight, target, **kwargs)

    monkeypatch.setattr(training_module.F, "linear_cross_entropy", record_linear_ce)
    dense_projection_rows: list[int] = []
    assert trainer.joint.model.output is not None
    hook = trainer.joint.model.output.register_forward_pre_hook(
        lambda _module, inputs: dense_projection_rows.append(inputs[0].shape[0])
    )
    try:
        losses = trainer._compute_loss(batch, blt_sampling=plan)
    finally:
        hook.remove()
    assert projected_rows == [
        (batch.ar_targets.numel(), int(batch.ar_targets.ne(-100).sum())),
        (int(plan.active.sum()), int(plan.active.sum())),
    ]
    assert dense_projection_rows == [batch.bos_targets.numel()]
    dense = dense_oracle.forward_blt_d_ragged(
        batch.ids,
        batch.valid,
        plan.noisy_blocks,
        plan.block_rows,
        plan.block_starts,
        plan.condition_indices,
        plan.row_cu_seqlens,
        plan.block_valid,
        positions=batch.positions,
        document_ids=batch.document_ids,
        byte_indices=batch.byte_indices,
        byte_cu_seqlens=batch.byte_cu_seqlens,
        patch_cu_seqlens=batch.patch_cu_seqlens,
        condition_patch_indices=batch.condition_patch_indices,
        global_patch_sources=batch.global_patch_sources,
        global_patch_positions=batch.global_patch_positions,
        physical_to_global_patch_indices=(
            batch.physical_to_global_patch_indices
        ),
        bos_condition_indices=batch.bos_condition_indices,
        patch_byte_cu_seqlens=batch.patch_byte_cu_seqlens,
        max_patch_size=batch.max_patch_size,
        return_clean_patch_states=False,
        allow_dense_reference=True,
    )
    dense_ar = training_module.cross_entropy_per_row(
        dense.clean_logits, batch.ar_targets
    )
    assert dense.bos_patch_states is not None
    dense_bos_logits = dense_oracle.forward_bos_logits(
        dense.bos_patch_states, allow_dense_reference=True
    )
    dense_bos_nll = F.cross_entropy(
        dense_bos_logits, batch.bos_targets, reduction="none"
    )
    dense_ar_rows = dense_ar.total.clone()
    dense_ar_rows.scatter_add_(
        0, batch.bos_row_indices, dense_bos_nll
    )
    dense_ar_count = batch.ar_targets.ne(-100).sum() + batch.bos_targets.numel()
    dense_ar_mean = dense_ar_rows.sum() / dense_ar_count
    dense_targets = training_module.same_position_targets(
        plan.clean_blocks,
        plan.active,
        output_size=model_config.vocab.output_size,
    )
    dense_diffusion_nll = training_module.masked_cross_entropy_per_target(
        dense.block_logits, dense_targets, plan.active
    )
    dense_diffusion_rows = training_module.exhaustive_blt_paper_sum_rows(
        dense_diffusion_nll, plan
    )
    dense_total = (
        dense_diffusion_rows + run.lambda_ar * dense_ar_rows
    ).mean()

    torch.testing.assert_close(losses.total, dense_total, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(losses.ar, dense_ar_mean, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(
        losses.diffusion, dense_diffusion_rows.mean(), rtol=2e-6, atol=2e-7
    )
    torch.testing.assert_close(losses.noise_nll, dense_diffusion_nll.sum(1))
    torch.testing.assert_close(
        losses.noise_correct,
        (
            dense.block_logits.argmax(-1).eq(dense_targets)
            & plan.active
        ).sum(1),
    )
    assert losses.total.ndim == 0
    assert losses.diffusion.ndim == 0
    assert losses.diffusion_targets == plan.active.sum()
    assert losses.noise_masked.shape == (plan.clean_blocks.shape[0],)
    assert bool(torch.isfinite(losses.total))
    losses.total.backward()
    dense_total.backward()
    dense_parameters = dict(dense_oracle.named_parameters())
    for name, parameter in trainer.joint.model.named_parameters():
        dense_parameter = dense_parameters[name]
        if parameter.grad is None or dense_parameter.grad is None:
            assert parameter.grad is None and dense_parameter.grad is None, name
            continue
        torch.testing.assert_close(
            parameter.grad,
            dense_parameter.grad,
            rtol=3e-5,
            atol=3e-7,
        )
    for block in trainer.joint.model.decoder:
        assert block.cross_key.weight.grad is not None
        assert bool(block.cross_key.weight.grad.abs().sum() > 0)


def test_active_linear_cross_entropy_compiles_fullgraph_on_cpu() -> None:
    torch.manual_seed(509)
    states = torch.randn(3, 4, 16, requires_grad=True)
    targets = torch.randint(0, 19, (3, 4))
    active = torch.tensor(
        [
            [True, False, True, False],
            [False, True, True, True],
            [True, False, False, True],
        ]
    )
    active_indices = torch.nonzero(active.reshape(-1), as_tuple=False).flatten()
    active_targets = targets.reshape(-1).index_select(0, active_indices)
    weight = torch.randn(19, 16, requires_grad=True)
    bias = torch.randn(19, requires_grad=True)

    def cell(
        hidden: torch.Tensor,
        indices: torch.Tensor,
        labels: torch.Tensor,
        projection: torch.Tensor,
        projection_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nll, correct = training_module.active_linear_cross_entropy(
            hidden,
            indices,
            labels,
            projection,
            projection_bias,
            collect_correct=True,
        )
        assert correct is not None
        return nll, correct

    compiled = torch.compile(cell, backend="aot_eager", fullgraph=True)
    nll, correct = compiled(
        states, active_indices, active_targets, weight, bias
    )
    assert nll.shape == active_targets.shape
    assert correct.shape == active_targets.shape
    nll.sum().backward()
    assert states.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None


def test_active_linear_cross_entropy_uses_precomputed_index_select(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = torch.randn(3, 4, 8)
    active = torch.tensor(
        [
            [True, False, False, True],
            [False, True, False, False],
            [True, False, True, False],
        ]
    )
    targets = torch.randint(0, 11, active.shape)
    indices = torch.nonzero(active.reshape(-1), as_tuple=False).flatten()
    active_targets = targets.reshape(-1).index_select(0, indices)
    weight = torch.randn(11, 8)
    observed: list[tuple[tuple[int, ...], torch.Tensor]] = []
    original_index_select = torch.index_select

    def record_index_select(input, dim, index, *, out=None):
        if input.shape == (12, 8):
            observed.append((tuple(input.shape), index.clone()))
        return original_index_select(input, dim, index, out=out)

    monkeypatch.setattr(training_module.torch, "index_select", record_index_select)
    nll, _ = training_module.active_linear_cross_entropy(
        states,
        indices,
        active_targets,
        weight,
        None,
    )

    assert len(observed) == 1
    torch.testing.assert_close(observed[0][1], indices)
    assert nll.shape == active_targets.shape


def test_active_linear_cross_entropy_bounds_diagnostic_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_count = training_module.DIAGNOSTIC_LINEAR_CHUNK_SIZE + 7
    states = torch.randn(active_count, 4)
    indices = torch.arange(active_count)
    targets = torch.randint(0, 9, (active_count,))
    weight = torch.randn(9, 4)
    projected_rows: list[int] = []
    original_linear = F.linear

    def record_linear(input, projection, bias=None):
        projected_rows.append(input.shape[0])
        return original_linear(input, projection, bias)

    monkeypatch.setattr(
        training_module.F,
        "linear_cross_entropy",
        lambda input, _weight, _target, **_kwargs: input.new_zeros(
            input.shape[0]
        ),
    )
    monkeypatch.setattr(training_module.F, "linear", record_linear)
    _, correct = training_module.active_linear_cross_entropy(
        states,
        indices,
        targets,
        weight,
        None,
        collect_correct=True,
    )

    assert correct is not None
    assert projected_rows == [
        training_module.DIAGNOSTIC_LINEAR_CHUNK_SIZE,
        7,
    ]
    assert max(projected_rows) <= training_module.DIAGNOSTIC_LINEAR_CHUNK_SIZE


def test_active_linear_cross_entropy_accepts_zero_active_atoms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_projection(*_args, **_kwargs):
        raise AssertionError("zero-active objective must not launch a projection")

    monkeypatch.setattr(training_module.F, "linear_cross_entropy", reject_projection)
    monkeypatch.setattr(training_module.F, "linear", reject_projection)
    nll, correct = training_module.active_linear_cross_entropy(
        torch.randn(2, 3, 8),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.randn(11, 8),
        None,
        collect_correct=True,
    )

    torch.testing.assert_close(nll, torch.zeros_like(nll))
    assert correct is not None
    assert not bool(correct.any())


@pytest.mark.parametrize("output_tied", [False, True])
def test_linear_ce_projection_preserves_registered_gradient_owner(
    output_tied: bool,
) -> None:
    class DdpLikeWrapper(torch.nn.Module):
        def __init__(self, module: torch.nn.Module) -> None:
            super().__init__()
            self.module = module

    model = ByteDiffusionModel(
        ByteDiffusionConfig.tiny(output_tied=output_tied)
    )
    wrapped = DdpLikeWrapper(training_module.JointForward(model))
    weight, bias = training_module._byte_output_projection(wrapped)
    states = torch.randn(2, 3, model.config.local_dim, requires_grad=True)
    targets = torch.randint(0, model.config.vocab.output_size, (2, 3))
    active = torch.tensor([[True, False, True], [False, True, False]])
    active_indices = torch.nonzero(active.reshape(-1), as_tuple=False).flatten()
    active_targets = targets.reshape(-1).index_select(0, active_indices)

    nll, _ = training_module.active_linear_cross_entropy(
        states,
        active_indices,
        active_targets,
        weight,
        bias,
    )
    nll.sum().backward()

    assert bias is None
    if output_tied:
        assert weight.data_ptr() == model.embedding.weight.data_ptr()
        assert model.embedding.weight.grad is not None
        assert bool(model.embedding.weight.grad.abs().sum() > 0)
    else:
        assert model.output is not None
        assert weight is model.output.weight
        assert model.output.weight.grad is not None
        assert bool(model.output.weight.grad.abs().sum() > 0)


def test_ragged_workload_grouping_uses_actual_unbounded_origin_counts() -> None:
    workloads = np.array(
        [
            8_192 + 4 * 2_047,
            8_192 + 4 * 2_309,
            8_192 + 4 * 2_738,
        ],
        dtype=np.int64,
    )
    groups = training_module.group_rows_by_ragged_physical_workload(
        (10, 11, 12),
        workloads,
        max_batch_size=3,
        physical_token_budget=int(2 * workloads.max()),
    )
    np.testing.assert_array_equal(groups.ordered_rows, np.array([10, 11, 12]))
    np.testing.assert_array_equal(groups.row_cu_seqlens, np.array([0, 2, 3]))
    np.testing.assert_array_equal(
        groups.group_physical_positions,
        np.array([workloads[0] + workloads[1], workloads[2]]),
    )
    assert groups.group_count == 2
    assert not groups.ordered_rows.flags.writeable
    assert not groups.row_cu_seqlens.flags.writeable
    assert not groups.group_physical_positions.flags.writeable
    assert workloads[2] > 8_192 + 4 * 2_048
    assert training_module.require_equal_ragged_microsteps((2, 2, 2)) == 2
    with pytest.raises(ValueError, match="different backward-call counts"):
        training_module.require_equal_ragged_microsteps((2, 3))


def test_ragged_grouping_uses_cumulative_work_and_can_equalize_ddp_calls() -> None:
    workloads = np.array([19_200, *([16_000] * 30)], dtype=np.int64)
    groups = training_module.group_rows_by_ragged_physical_workload(
        tuple(range(31)),
        workloads,
        max_batch_size=32,
        physical_token_budget=278_528,
    )
    assert groups.group_count == 2
    assert groups.row_cu_seqlens.tolist() == [0, 15, 31]
    assert bool((groups.group_physical_positions <= 278_528).all())

    original = tuple(
        tuple(int(value) for value in groups.ordered_rows[start:stop])
        for start, stop in zip(
            groups.row_cu_seqlens[:-1],
            groups.row_cu_seqlens[1:],
            strict=True,
        )
    )
    equalized = training_module.split_ragged_groups_to_count(original, 3)
    assert len(equalized) == 3
    assert tuple(value for group in equalized for value in group) == tuple(range(31))
    assert all(group for group in equalized)


def test_ragged_grouping_rebalances_a_singleton_tail_without_reordering() -> None:
    rows = np.arange(14, dtype=np.int64) + 100
    workloads = np.full(14, 15_000, dtype=np.int64)
    groups = training_module.group_rows_by_ragged_physical_workload(
        rows,
        workloads,
        max_batch_size=13,
        physical_token_budget=195_000,
    )

    np.testing.assert_array_equal(groups.ordered_rows, rows)
    np.testing.assert_array_equal(groups.row_cu_seqlens, np.array([0, 7, 14]))
    np.testing.assert_array_equal(
        groups.group_physical_positions,
        np.array([105_000, 105_000]),
    )
    assert int(np.diff(groups.row_cu_seqlens).min()) > 1
    assert groups.ordered_rows.tolist() == list(range(100, 114))


def test_ragged_grouping_keeps_an_already_balanced_final_pair() -> None:
    rows = np.arange(12, dtype=np.int64)
    workloads = np.array(
        [12_000, 13_000, 14_000, 15_000, 16_000, 17_000] * 2,
        dtype=np.int64,
    )
    groups = training_module.group_rows_by_ragged_physical_workload(
        rows,
        workloads,
        max_batch_size=6,
        physical_token_budget=100_000,
    )

    np.testing.assert_array_equal(groups.ordered_rows, rows)
    np.testing.assert_array_equal(groups.row_cu_seqlens, np.array([0, 6, 12]))
    np.testing.assert_array_equal(
        groups.group_physical_positions,
        np.array([87_000, 87_000]),
    )
    assert bool((np.diff(groups.row_cu_seqlens) <= 6).all())
    assert bool((groups.group_physical_positions <= 100_000).all())


def test_exhaustive_fast_blt_pads_a_non_eot_training_sequence_tail() -> None:
    ids = torch.tensor([[65, 66, 67, 68, 69, 70, 71, 72]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    documents = torch.zeros_like(ids)
    batch = TrainingBatch(
        ids=ids,
        valid=valid,
        ar_targets=torch.full_like(ids, -100),
        bos_targets=torch.empty(0, dtype=torch.long),
        positions=torch.arange(8)[None],
        full_valid=True,
        document_ids=documents,
        physical_patch_row_indices=torch.tensor([0, 0]),
        physical_patch_start_columns=torch.tensor([0, 6]),
        physical_patch_prior_condition_indices=torch.tensor([-1, 0]),
    )
    plan = training_module.prepare_exhaustive_blt_corruption(
        batch,
        CorruptionConfig(
            kind="blt_bernoulli", canvas_length=4, branches_per_row=2_048
        ),
        ByteDiffusionConfig.tiny().vocab,
        torch.Generator().manual_seed(607),
        row_t=torch.tensor([0.5]),
    )
    assert plan.block_starts.tolist() == [6]
    assert plan.block_valid.tolist() == [[True, True, False, False]]
    assert plan.clean_blocks.tolist() == [[71, 72, 262, 262]]


def test_blt_blocks_never_cross_a_packed_document_boundary() -> None:
    ids = torch.tensor(
        [[65, 66, 67, 68, 69, 70, 71, 256, 72, 73, 74, 75, 76, 77, 78, 256]]
    )
    valid = torch.ones_like(ids, dtype=torch.bool)
    documents = torch.tensor([[0] * 8 + [1] * 8])
    positions = torch.tensor([[*range(8), *range(8)]])
    layout = training_module._packed_document_layout(valid, documents, positions)
    batch = TrainingBatch(
        ids=ids,
        valid=valid,
        ar_targets=torch.full_like(ids, -100),
        bos_targets=torch.empty(0, dtype=torch.long),
        positions=positions,
        full_valid=True,
        document_ids=documents,
        bos_row_indices=torch.empty(0, dtype=torch.long),
        isolate_documents=True,
        byte_indices=layout[0],
        byte_cu_seqlens=layout[1],
        patch_indices=layout[2],
        patch_cu_seqlens=layout[3],
        condition_patch_indices=layout[4],
        global_patch_sources=layout[5],
        global_patch_positions=layout[6],
        physical_to_global_patch_indices=layout[7],
        bos_condition_indices=layout[8],
        prior_condition_indices=layout[9],
    )
    plan = prepare_blt_corruption(
        batch,
        CorruptionConfig(
            kind="blt_bernoulli", canvas_length=8, branches_per_row=2
        ),
        ByteDiffusionConfig.tiny().vocab,
        torch.Generator().manual_seed(9),
    )

    # All four aligned origins are eligible. Starts at columns 4 and 12 keep
    # their same-document four-byte tails and PAD-mask the overflowing suffix.
    torch.testing.assert_close(plan.sampling_weight, torch.tensor([2.0]))
    assert all(
        count in {4, 8}
        for count in plan.branch_valid.sum((0, 2)).tolist()
    )
    assert bool(plan.condition_indices.ge(0).all())
    for branch, start in enumerate(plan.block_starts[0].tolist()):
        document_stop = 8 if start < 8 else 16
        stop = min(start + 8, document_stop)
        valid_length = stop - start
        assert plan.clean_blocks[0, branch, :valid_length].tolist() == ids[
            0, start:stop
        ].tolist()
        assert bool(
            (plan.clean_blocks[0, branch, valid_length:] == 262).all()
        )


def test_exhaustive_document_blt_origins_are_ordered_and_rng_free() -> None:
    ids = torch.tensor(
        [[65, 66, 67, 256, 72, 73, 74, 256, 80, 81, 82, 256, 88, 89, 90, 256]]
    )
    valid = torch.ones_like(ids, dtype=torch.bool)
    documents = torch.tensor([[0] * 4 + [1] * 4 + [2] * 4 + [3] * 4])
    positions = torch.tensor([[*range(4), *range(4), *range(4), *range(4)]])
    layout = training_module._packed_document_layout(valid, documents, positions)
    generator = torch.Generator().manual_seed(919)
    before = generator.get_state().clone()

    starts, weight, selected, conditions = (
        training_module.sample_blt_patch_starts(
            valid,
            count=4,
            block_length=4,
            patch_stride=4,
            generator=generator,
            document_ids=documents,
            clean_ids=ids,
            prior_condition_indices=layout[9],
        )
    )

    torch.testing.assert_close(generator.get_state(), before)
    assert starts.tolist() == [[0, 4, 8, 12]]
    assert bool(selected.all())
    assert bool(conditions.ge(0).all())
    torch.testing.assert_close(weight, torch.ones(1))


def test_document_packed_blt_branch_cannot_read_another_document() -> None:
    torch.manual_seed(727)
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).eval()
    ids = torch.tensor(
        [[65, 66, 67, 68, 69, 70, 71, 256, 80, 81, 82, 83, 84, 85, 86, 256]]
    )
    valid = torch.ones_like(ids, dtype=torch.bool)
    documents = torch.tensor([[0] * 8 + [1] * 8])
    positions = torch.tensor([[*range(8), *range(8)]])
    layout = training_module._packed_document_layout(valid, documents, positions)
    batch = TrainingBatch(
        ids=ids,
        valid=valid,
        ar_targets=torch.full_like(ids, -100),
        bos_targets=torch.tensor([65, 80]),
        positions=positions,
        full_valid=False,
        document_ids=documents,
        bos_row_indices=torch.tensor([0, 0]),
        isolate_documents=True,
        byte_indices=layout[0],
        byte_cu_seqlens=layout[1],
        patch_indices=layout[2],
        patch_cu_seqlens=layout[3],
        condition_patch_indices=layout[4],
        global_patch_sources=layout[5],
        global_patch_positions=layout[6],
        physical_to_global_patch_indices=layout[7],
        bos_condition_indices=layout[8],
        prior_condition_indices=layout[9],
    )
    plan = prepare_blt_corruption(
        batch,
        CorruptionConfig(
            kind="blt_bernoulli", canvas_length=8, branches_per_row=2
        ),
        model.config.vocab,
        torch.Generator().manual_seed(9),
    )

    def forward(clean_ids: torch.Tensor):
        return model.forward_blt_d_branches(
            clean_ids,
            valid,
            plan.noisy_blocks,
            plan.branch_valid,
            plan.block_starts,
            positions=positions,
            document_ids=documents,
            branch_condition_indices=plan.condition_indices,
            **_document_model_metadata(layout),
        )

    baseline = forward(ids)
    causal = model.forward_ar_varlen(
        ids,
        valid,
        positions=positions,
        allow_dense_reference=True,
        **_document_model_metadata(layout),
    )
    # Projecting unique patch states before gathering them changes only the
    # floating-point reduction order relative to the standalone AR path.
    torch.testing.assert_close(
        baseline.clean_logits, causal.logits, rtol=1e-5, atol=1e-7
    )
    changed = ids.clone()
    changed[:, :7] = torch.arange(90, 97)
    perturbed = forward(changed)

    torch.testing.assert_close(
        baseline.clean_logits[:, 8:], perturbed.clean_logits[:, 8:], rtol=0, atol=0
    )
    second_branch = plan.block_starts[0].ge(8).nonzero(as_tuple=False).item()
    torch.testing.assert_close(
        baseline.branch_logits[:, second_branch],
        perturbed.branch_logits[:, second_branch],
        rtol=0,
        atol=0,
    )


def test_document_packed_blt_validation_scores_short_tail_origins() -> None:
    manifest = AtomicIdManifest.reference()
    chunks = pack_documents(
        (
            AtomicDocument("a", (65, 66, 67, 68, 69, 70, 71, manifest.eot_id)),
            AtomicDocument("b", (80, 81, 82, 83, 84, 85, 86, manifest.eot_id)),
        ),
        manifest,
        chunk_size=16,
    )
    config = _run_config(recipe="blt_d", iterations=1)
    config = replace(
        config,
        corruption=CorruptionConfig(
            kind="blt_bernoulli", canvas_length=8, branches_per_row=4
        ),
        diffusion_validation_chunks=1,
    )
    trainer = _trainer(
        ByteDiffusionModel(ByteDiffusionConfig.tiny()), config, chunks=chunks
    )

    metrics = trainer.validate()
    cached_plans = tuple(trainer._validation_blt_plan_cache.values())
    repeated = trainer.validate()

    assert metrics.ar_targets == 16
    assert metrics.diffusion_chunks == 1
    # Exact-K validation supervises a deterministic subset while the ELBO
    # denominator retains the complete origin-corrected atom coverage.
    assert 1 <= metrics.diffusion_targets <= 24
    assert metrics.diffusion_elbo_atoms == 24
    assert metrics.diffusion_elbo_proxy_bpb is not None
    assert math.isfinite(metrics.diffusion_elbo_proxy_bpb)
    assert math.isfinite(metrics.bpb)
    assert math.isfinite(metrics.diffusion_loss)
    assert cached_plans
    assert tuple(trainer._validation_blt_plan_cache.values()) == cached_plans
    assert repeated.diffusion_loss == metrics.diffusion_loss
    assert repeated.diffusion_elbo_proxy_bpb == metrics.diffusion_elbo_proxy_bpb


def test_canvas_start_never_crosses_pad_when_a_full_span_exists() -> None:
    valid = torch.zeros(1, 8192, dtype=torch.bool)
    valid[:, :513] = True
    for seed in range(32):
        starts = sample_nonoverlapping_patch_starts(
            valid,
            span_length=512,
            count=1,
            patch_stride=4,
            generator=torch.Generator().manual_seed(seed),
        )
        assert starts.tolist() == [[0]]


def test_validation_starts_are_invariant_to_batch_and_rank_order() -> None:
    chunks = tuple(_chunks())
    batch = chunks_to_batch(chunks)
    together = sample_validation_starts(
        batch.valid,
        chunks,
        span_length=4,
        count=1,
        patch_stride=4,
        seed=17,
    )
    reordered_chunks = (chunks[3], chunks[1])
    reordered = chunks_to_batch(reordered_chunks)
    subset = sample_validation_starts(
        reordered.valid,
        reordered_chunks,
        span_length=4,
        count=1,
        patch_stride=4,
        seed=17,
    )
    torch.testing.assert_close(subset, together[[3, 1]])


def test_validation_exact_k_is_batch_invariant_and_never_selects_padding() -> None:
    eligible = torch.tensor(
        [
            [[True, True, True, True], [True, True, False, False]],
            [[True, False, True, False], [True, True, True, False]],
            [[True, True, True, False], [False, False, False, False]],
        ]
    )
    chunks = tuple(
        ValidationChunkIdentity(chunk_index=10 + index, stream_start=31 * index)
        for index in range(3)
    )
    active, t = validation_exact_k_mask(eligible, chunks, seed=19)
    assert not bool((active & ~eligible).any())
    counts = eligible.flatten(1).sum(1)
    selected = active.flatten(1).sum(1)
    assert bool(((selected >= 1) & (selected <= counts)).all())
    torch.testing.assert_close(t, selected.float() / counts.float())

    reordered = torch.tensor([2, 0])
    subset_active, subset_t = validation_exact_k_mask(
        eligible[reordered],
        tuple(chunks[index] for index in reordered.tolist()),
        seed=19,
    )
    torch.testing.assert_close(subset_active, active[reordered])
    torch.testing.assert_close(subset_t, t[reordered])


def test_exact_k_and_origin_ht_estimators_match_tiny_analytic_oracles() -> None:
    per_atom_nll = np.array([0.2, 1.1, 2.3], dtype=np.float64)
    size = len(per_atom_nll)
    integrated = 0.0
    proposal = 0.0
    for k in range(1, size + 1):
        subsets = tuple(itertools.combinations(range(size), k))
        mean_selected_sum = np.mean(
            [per_atom_nll[np.array(subset)].sum() for subset in subsets]
        )
        integrated += mean_selected_sum / k
        proposal += (size / k) * mean_selected_sum / size
    assert proposal == pytest.approx(integrated, abs=1e-15)

    # Unequal tail coverage is corrected by the same M/m origin weight for
    # both numerator and atom denominator before their final ratio.
    origin_nll = np.array([3.0, 0.8, 0.1], dtype=np.float64)
    origin_atoms = np.array([4.0, 2.0, 1.0], dtype=np.float64)
    selections = tuple(itertools.combinations(range(3), 2))
    ht_nll = np.mean(
        [1.5 * origin_nll[np.array(selection)].sum() for selection in selections]
    )
    ht_atoms = np.mean(
        [1.5 * origin_atoms[np.array(selection)].sum() for selection in selections]
    )
    assert ht_nll == pytest.approx(origin_nll.sum())
    assert ht_atoms == pytest.approx(origin_atoms.sum())


def test_short_rows_retain_partial_diffusion_span_and_ar_targets() -> None:
    manifest = AtomicIdManifest.reference()
    chunks = pack_documents(
        (AtomicDocument("short", (65, 66, manifest.eot_id)),),
        manifest,
        chunk_size=8,
    )
    batch = chunks_to_batch(chunks)
    plan = prepare_canvas_corruption(
        batch,
        CorruptionConfig(kind="absorbing_rb", canvas_length=8, branches_per_row=1),
        ByteDiffusionConfig.tiny().vocab,
        torch.Generator().manual_seed(5),
    )
    scored = batch.ar_targets.ne(-100).sum() + batch.bos_targets.ne(-100).sum()
    assert int(scored) == 3
    assert plan.branch_starts.tolist() == [[0]]
    assert plan.branch_valid.sum() == 3
    assert not bool((plan.active & ~plan.branch_valid).any())


def test_pad_only_extra_canvas_branch_contributes_zero_targets() -> None:
    manifest = AtomicIdManifest.reference()
    chunks = pack_documents(
        (AtomicDocument("short-multi", (65, 66, manifest.eot_id)),),
        manifest,
        chunk_size=16,
    )
    plan = prepare_canvas_corruption(
        chunks_to_batch(chunks),
        CorruptionConfig(kind="absorbing_rb", canvas_length=8, branches_per_row=2),
        ByteDiffusionConfig.tiny().vocab,
        torch.Generator().manual_seed(5),
    )
    assert plan.branch_valid.sum((0, 2)).tolist() == [3, 0]
    assert plan.active.sum((0, 2))[1] == 0


def test_subpatch_blt_row_keeps_ar_and_has_zero_diffusion_targets() -> None:
    manifest = AtomicIdManifest.reference()
    chunks = pack_documents(
        (AtomicDocument("tiny-blt", (65, 66, manifest.eot_id)),),
        manifest,
        chunk_size=8,
    )
    base = _run_config(recipe="canvas", iterations=1)
    config = replace(
        base,
        recipe="blt_d",
        corruption=CorruptionConfig(
            kind="blt_bernoulli", canvas_length=4, branches_per_row=1
        ),
        objective_reduction="paper_sum",
    )
    trainer = _trainer(
        ByteDiffusionModel(ByteDiffusionConfig.tiny()), config, chunks
    )
    metrics = trainer.run_update()
    assert metrics.ar_targets == 3
    # Virtual BOS makes the first physical content patch a valid BLT origin.
    assert metrics.diffusion_targets == 3
    assert math.isfinite(metrics.total)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"recipe": "canavs"}, "unknown training recipe"),
        ({"objective_reduction": "sumish"}, "unknown objective reduction"),
        ({"compile_mode": "fastest"}, "unknown compile mode"),
        (
            {
                "corruption": CorruptionConfig(
                    kind="uniform_replacement", canvas_length=8, branches_per_row=1
                )
            },
            "self-conditioning",
        ),
    ],
)
def test_run_config_fails_closed_on_unknown_or_incomplete_modes(updates, match) -> None:
    with pytest.raises(ValueError, match=match):
        replace(_run_config(), **updates)


def test_blt_update_passes_block_geometry_to_the_model() -> None:
    class CountingModel(ByteDiffusionModel):
        def __init__(self):
            super().__init__(ByteDiffusionConfig.tiny())
            self.observed_starts = None
            self.observed_length = None
            self.ordinary_forwards = 0

        def forward(self, *args, **kwargs):
            self.ordinary_forwards += 1
            return super().forward(*args, **kwargs)

        def forward_blt_d_branches(
            self,
            *args,
            positions=None,
            **kwargs,
        ):
            noisy_blocks = args[2]
            block_starts = args[4]
            self.observed_starts = block_starts.detach().clone()
            self.observed_length = noisy_blocks.shape[-1]
            return super().forward_blt_d_branches(
                *args,
                positions=positions,
                **kwargs,
            )

    base = _run_config(recipe="canvas")
    config = TrainingRunConfig(
        **{
            **base.contract_dict(),
            "corruption": CorruptionConfig(
                kind="absorbing_rb", canvas_length=4, branches_per_row=2
            ),
            "recipe": "blt_d",
        }
    )
    model = CountingModel()
    trainer = _trainer(model, config)
    trainer.run_update()
    assert model.observed_starts is not None
    assert model.observed_starts.shape == (1, 2)
    assert model.observed_length == 4
    assert model.ordinary_forwards == 0


def test_blt_loss_sums_all_blocks_per_clean_row_before_batch_mean(
    monkeypatch,
) -> None:
    batch = chunks_to_batch(_chunks()[:1])
    clean_blocks = torch.zeros(1, 2, 4, dtype=torch.long)
    active = torch.ones_like(clean_blocks, dtype=torch.bool)
    plan = BltCorruptionPlan(
        clean_blocks=clean_blocks,
        noisy_blocks=clean_blocks.clone(),
        branch_valid=active,
        active=active,
        block_starts=torch.tensor([[0, 4]]),
        condition_indices=torch.tensor([[0, 1]]),
        block_length=4,
        t=torch.tensor([0.5]),
        sampling_weight=torch.tensor([2.0]),
    )
    monkeypatch.setattr(
        training_module,
        "prepare_blt_corruption",
        lambda *args, **kwargs: plan,
    )

    desired_nll = torch.tensor([[[1.0] * 4, [3.0] * 4]])
    probability = torch.exp(-desired_nll)
    other_probability = (1.0 - probability) / 260
    diffusion_logits = torch.cat(
        (
            probability.log().unsqueeze(-1),
            other_probability.log().unsqueeze(-1).expand(-1, -1, -1, 260),
        ),
        dim=-1,
    )

    class FixedForward(torch.nn.Module):
        def forward(self, *args, **kwargs):
            del args, kwargs
            ar_logits = torch.zeros(1, batch.ids.shape[1], 261)
            bos_logits = torch.zeros(1, 261)
            return ar_logits, diffusion_logits, bos_logits

    base = _run_config(recipe="canvas")
    config = TrainingRunConfig(
        **{
            **base.contract_dict(),
            "corruption": CorruptionConfig(
                kind="blt_bernoulli", canvas_length=4, branches_per_row=2
            ),
            "recipe": "blt_d",
            "objective_reduction": "paper_sum",
        }
    )
    trainer = _trainer(ByteDiffusionModel(ByteDiffusionConfig.tiny()), config)
    trainer.forward_model = FixedForward()
    loss = trainer._compute_loss(batch)

    # The sampled masked sum is 32, and the HT weight of two estimates the
    # complete fixed-patch bank at 64. Clean paper_sum adds all eight atomic
    # targets (virtual BOS + seven shifted positions) rather than mean AR CE.
    torch.testing.assert_close(loss.diffusion, torch.tensor(64.0))
    expected = 64.0 + 8.0 * math.log(261)
    torch.testing.assert_close(loss.total, torch.tensor(expected), rtol=1e-6, atol=1e-6)


def test_blt_row_normalized_sum_preserves_signal_ratio_without_raw_scale(
    monkeypatch,
) -> None:
    batch = chunks_to_batch(_chunks()[:1])
    clean_blocks = torch.zeros(1, 2, 4, dtype=torch.long)
    active = torch.ones_like(clean_blocks, dtype=torch.bool)
    plan = BltCorruptionPlan(
        clean_blocks=clean_blocks,
        noisy_blocks=clean_blocks.clone(),
        branch_valid=active,
        active=active,
        block_starts=torch.tensor([[0, 4]]),
        condition_indices=torch.tensor([[0, 1]]),
        block_length=4,
        t=torch.tensor([0.5]),
        sampling_weight=torch.tensor([2.0]),
    )
    monkeypatch.setattr(
        training_module,
        "prepare_blt_corruption",
        lambda *args, **kwargs: plan,
    )
    desired_nll = torch.tensor([[[1.0] * 4, [3.0] * 4]])
    probability = torch.exp(-desired_nll)
    other_probability = (1.0 - probability) / 260
    diffusion_logits = torch.cat(
        (
            probability.log().unsqueeze(-1),
            other_probability.log().unsqueeze(-1).expand(-1, -1, -1, 260),
        ),
        dim=-1,
    )

    class FixedForward(torch.nn.Module):
        def forward(self, *args, **kwargs):
            del args, kwargs
            return (
                torch.zeros(1, batch.ids.shape[1], 261),
                diffusion_logits,
                torch.zeros(1, 261),
            )

    base = _run_config(recipe="canvas")
    config = TrainingRunConfig(
        **{
            **base.contract_dict(),
            "corruption": CorruptionConfig(
                kind="blt_bernoulli", canvas_length=4, branches_per_row=2
            ),
            "recipe": "blt_d",
            "objective_reduction": "row_normalized_sum",
        }
    )
    trainer = _trainer(ByteDiffusionModel(ByteDiffusionConfig.tiny()), config)
    trainer.forward_model = FixedForward()
    loss = trainer._compute_loss(batch)

    expected_diffusion = 64.0 / 8.0
    torch.testing.assert_close(loss.diffusion, torch.tensor(expected_diffusion))
    torch.testing.assert_close(
        loss.total,
        torch.tensor(expected_diffusion + math.log(261)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_causal_only_is_a_named_control_and_does_not_consume_corruption_rng() -> None:
    class CountingModel(ByteDiffusionModel):
        def __init__(self):
            super().__init__(ByteDiffusionConfig.tiny())
            self.ordinary_forwards = 0
            self.varlen_forwards = 0

        def forward(self, *args, **kwargs):
            self.ordinary_forwards += 1
            return super().forward(*args, **kwargs)

        def forward_ar_varlen(self, *args, **kwargs):
            self.varlen_forwards += 1
            return super().forward_ar_varlen(*args, **kwargs)

    torch.manual_seed(11)
    model = CountingModel()
    trainer = _trainer(
        model,
        _run_config(recipe="causal_only"),
    )
    generator_before = trainer.corruption_generator.get_state().clone()
    metrics = trainer.run_update()
    assert metrics.ar > 0
    assert metrics.diffusion == 0
    assert metrics.diffusion_targets == 0
    assert model.varlen_forwards == 1
    assert model.ordinary_forwards == 0
    torch.testing.assert_close(
        trainer.corruption_generator.get_state(), generator_before, rtol=0, atol=0
    )


def test_update_boundary_checkpoint_resume_is_bit_exact(tmp_path: Path) -> None:
    torch.manual_seed(19)
    initial = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    uninterrupted = _trainer(copy.deepcopy(initial), _run_config())
    interrupted = _trainer(copy.deepcopy(initial), _run_config())

    uninterrupted.run_update()
    uninterrupted.run_update()
    interrupted.run_update()
    checkpoint = tmp_path / "resume.pt"
    interrupted.save_checkpoint(checkpoint)
    assert torch.load(checkpoint, map_location="cpu", weights_only=False)[
        "schema"
    ] == CHECKPOINT_SCHEMA

    # Constructor initialization and unrelated RNG consumption must not alter
    # continuation after the checkpoint restores every RNG stream.
    torch.manual_seed(999)
    resumed = _trainer(ByteDiffusionModel(ByteDiffusionConfig.tiny()), _run_config())
    resumed.load_checkpoint(checkpoint)
    resumed.run_update()

    _assert_nested_equal(
        uninterrupted.joint.model.state_dict(), resumed.joint.model.state_dict()
    )
    _assert_nested_equal(
        uninterrupted.optimizer.state_dict(), resumed.optimizer.state_dict()
    )
    assert uninterrupted.train_cursor.state_dict() == resumed.train_cursor.state_dict()
    torch.testing.assert_close(
        uninterrupted.corruption_generator.get_state(),
        resumed.corruption_generator.get_state(),
        rtol=0,
        atol=0,
    )


def test_checkpoint_rejects_obsolete_topology_schema(tmp_path: Path) -> None:
    trainer = _trainer(ByteDiffusionModel(ByteDiffusionConfig.tiny()), _run_config())
    current = tmp_path / "current.pt"
    obsolete = tmp_path / "obsolete.pt"
    trainer.save_checkpoint(current)
    payload = torch.load(current, map_location="cpu", weights_only=False)
    payload["schema"] = "byte_diffusion_training/v6"
    torch.save(payload, obsolete)

    with pytest.raises(ValueError, match="schema"):
        trainer.load_checkpoint(obsolete)


def test_checkpoint_rejects_different_data_identity(tmp_path: Path) -> None:
    torch.manual_seed(23)
    source = _trainer(ByteDiffusionModel(ByteDiffusionConfig.tiny()), _run_config())
    path = tmp_path / "checkpoint.pt"
    source.save_checkpoint(path)
    different = _trainer(
        ByteDiffusionModel(ByteDiffusionConfig.tiny()),
        _run_config(),
        chunks=_chunks(offset=100),
    )
    with pytest.raises(ValueError, match="dataset_sha256"):
        different.load_checkpoint(path)


def test_metrics_are_parseable_by_the_ablation_harness() -> None:
    torch.manual_seed(29)
    trainer = _trainer(ByteDiffusionModel(ByteDiffusionConfig.tiny()), _run_config())
    train = trainer.run_update()
    train_entry = parse_log_line(format_train_metric(train, 2))
    assert train_entry is not None
    assert train_entry["type"] == "train"
    assert train_entry["ar_loss"] == pytest.approx(train.ar, abs=1e-6)

    validation = trainer.validate()
    val_entry = parse_log_line(
        format_validation_metric(1, 2, validation, trainer.training_time_ms)
    )
    assert val_entry is not None
    assert val_entry["type"] == "val"
    expected_primary_bpb = (
        validation.diffusion_elbo_proxy_bpb
        if validation.diffusion_elbo_proxy_bpb is not None
        else validation.bpb
    )
    assert val_entry["val_bpb"] == pytest.approx(expected_primary_bpb, abs=1e-6)
    assert val_entry["val_proxy_bpb"] == pytest.approx(
        expected_primary_bpb, abs=1e-6
    )
    assert val_entry["val_diffusion_loss"] == pytest.approx(
        validation.diffusion_loss, abs=1e-6
    )

    challenge_entry = parse_log_line(
        format_validation_metric(
            1,
            2,
            validation,
            trainer.training_time_ms,
            scope="challenge",
        )
    )
    assert challenge_entry is not None
    assert challenge_entry["val_challenge_bpb"] == pytest.approx(
        expected_primary_bpb, abs=1e-6
    )


def test_validation_bpb_counts_bos_shift_eot_and_partial_microbatch_once() -> None:
    chunks = tuple(_chunks())
    config = _run_config(recipe="causal_only")
    config = replace(config, microbatch_per_rank=3)
    trainer = _trainer(ByteDiffusionModel(ByteDiffusionConfig.tiny()), config, chunks)

    class UniformForward(torch.nn.Module):
        def forward(self, clean_ids, *args, **kwargs):
            del args, kwargs
            batch, length = clean_ids.shape
            return (
                torch.zeros(batch * length, 261),
                None,
                torch.zeros(batch, 261),
            )

    trainer.validation_model = UniformForward()
    metrics = trainer.validate()

    assert metrics.ar_targets == 32
    assert metrics.literal_bytes == 28
    assert metrics.special_targets == 4
    assert metrics.ar_loss == pytest.approx(math.log(261), rel=1e-6)
    assert metrics.bpb == pytest.approx(
        math.log(261) / math.log(2), rel=1e-6
    )


def test_validation_reuses_joint_forward_for_diffusion_subset_ar_scores() -> None:
    chunks = tuple(_chunks(offset=0)) + tuple(_chunks(offset=10))
    config = replace(
        _run_config(),
        validation_microbatch_per_rank=3,
        ar_validation_microbatch_per_rank=3,
        diffusion_validation_chunks=2,
    )
    trainer = _trainer(ByteDiffusionModel(ByteDiffusionConfig.tiny()), config, chunks)

    class CountingUniformForward(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ar_batches: list[int] = []
            self.diffusion_batches: list[int] = []

        def forward(self, clean_ids, _valid, _positions, noisy_ids, *args):
            del args
            assert not torch.is_grad_enabled()
            batch, length = clean_ids.shape
            if noisy_ids is None:
                self.ar_batches.append(batch)
                diffusion = None
                causal = torch.zeros(batch * length, 261)
            else:
                self.diffusion_batches.append(batch)
                diffusion = torch.zeros((*noisy_ids.shape, 261))
                causal = torch.zeros(batch, length, 261)
            return (
                causal,
                diffusion,
                torch.zeros(batch, 261),
            )

    forward = CountingUniformForward()
    trainer.validation_model = forward
    metrics = trainer.validate()

    assert forward.ar_batches == [3, 3]
    assert forward.diffusion_batches == [2]
    assert metrics.ar_targets == 64
    assert metrics.diffusion_chunks == 2
    assert metrics.diffusion_targets == 16


def test_all_origin_validation_schedule_is_immutable_and_cached() -> None:
    class EntropyChunks(tuple):
        patching = DatasetPatchingSpec(
            "causal_entropy_v1", None, 4, "1" * 64
        )
        chunk_size = 8

        def __new__(cls, chunks):
            instance = super().__new__(cls, chunks)
            instance.origin_count_calls = 0
            return instance

        def training_blt_origin_counts(
            self, indices, *, block_length, eot_id
        ):
            self.origin_count_calls += 1
            assert block_length == 4
            assert eot_id == 256
            return np.asarray(indices, dtype=np.int64) + 1

    chunks = EntropyChunks(_chunks())
    config = replace(
        _run_config(recipe="blt_d"),
        patching_policy="causal_entropy_v1",
        blt_origin_policy="all_entropy_patch_starts",
        corruption=CorruptionConfig(
            kind="blt_bernoulli", canvas_length=4, branches_per_row=2_048
        ),
        objective_reduction="paper_sum",
        diffusion_validation_chunks=2,
        validation_microbatch_per_rank=2,
        microbatch_token_budget=128,
    )
    trainer = ByteDiffusionTrainer(
        ByteDiffusionModel(ByteDiffusionConfig.tiny()),
        DeterministicChunkCursor(chunks, seed=config.seed),
        chunks,
        config,
        device=torch.device("cpu"),
    )

    first = trainer._validation_schedule()
    second = trainer._validation_schedule()

    assert first is second
    assert chunks.origin_count_calls == 1
    assert not first.all_indices.flags.writeable
    assert not first.diffusion_indices.flags.writeable
    assert not first.ar_only_indices.flags.writeable
    assert tuple(first.all_indices) == tuple(range(len(chunks)))
    assert set(first.ar_only_indices).isdisjoint(first.diffusion_indices)
    assert sorted((*first.ar_only_indices, *first.diffusion_indices)) == list(
        range(len(chunks))
    )
    assert first.diffusion_groups is not None
    assert tuple(value for group in first.diffusion_groups for value in group) == tuple(
        first.diffusion_indices
    )


def test_update_can_skip_unlogged_metric_materialization() -> None:
    trainer = _trainer(
        ByteDiffusionModel(ByteDiffusionConfig.tiny()),
        _run_config(iterations=2),
    )

    assert trainer.run_update(materialize_metrics=False) is None
    assert trainer.completed_steps == 1
    ledger = trainer.last_execution_ledger
    assert ledger is not None
    assert ledger.step == 1
    assert ledger.rows == ledger.group_row_cu_seqlens[-1]
    assert ledger.group_row_cu_seqlens[0] == 0
    assert ledger.microsteps == len(ledger.group_physical_positions)
    assert ledger.microsteps == len(ledger.group_row_cu_seqlens) - 1
    assert ledger.max_microbatch == max(
        stop - start
        for start, stop in zip(
            ledger.group_row_cu_seqlens[:-1],
            ledger.group_row_cu_seqlens[1:],
            strict=True,
        )
    )
    assert ledger.max_physical_positions == max(
        ledger.group_physical_positions
    )
    metrics = trainer.run_update(materialize_metrics=True)

    assert metrics is not None
    assert metrics.step == 2
    assert trainer.last_execution_ledger is not None
    assert trainer.last_execution_ledger.step == 2
    assert trainer.training_time_ms > 0
