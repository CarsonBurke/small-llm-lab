"""CPU contracts for scratch training, metrics, and exact resume."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import pretraining.byte_diffusion.training as training_module
from pretraining.byte_diffusion.config import ByteDiffusionConfig, CorruptionConfig
from pretraining.byte_diffusion.data import (
    AtomicDocument,
    AtomicIdManifest,
    DeterministicChunkCursor,
    pack_documents,
)
from pretraining.byte_diffusion.model import ByteDiffusionModel
from pretraining.byte_diffusion.training import (
    CANONICAL_PRESET,
    CHECKPOINT_SCHEMA,
    BltCorruptionPlan,
    ByteDiffusionTrainer,
    CanvasCorruptionPlan,
    DistributedContext,
    DeterministicSubsetChunkDataset,
    MappedPackedChunkDataset,
    TrainingBatch,
    TrainingRunConfig,
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

    observed = model_config_from_env(tiny=True)

    assert observed.ngram_hash == "legacy257"
    assert observed.ngram_factor_init == "scale_matched"
    assert observed.ngram_aggregation == "mean"
    assert observed.decoder_conditioning == "rmsnorm_projection"
    assert observed.ngram_enabled


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
    assert first["schema"] == "byte_diffusion_source_provenance/v1"
    assert len(first["sha256"]) == 64
    files = first["files"]
    assert "scripts/ablation.py" in files
    assert "scripts/train_byte_diffusion.py" in files
    assert "pretraining/byte_diffusion/training.py" in files


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


def test_compiled_branch_training_rejects_activation_checkpointing() -> None:
    with pytest.raises(ValueError, match="escapes compiled FlexAttention"):
        TrainingRunConfig(activation_checkpointing=True)


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
    # Exhausting all four patch origins includes each full document start and
    # both short four-byte document tails.
    assert metrics.diffusion_targets == 24
    assert math.isfinite(metrics.bpb)
    assert math.isfinite(metrics.diffusion_loss)
    assert cached_plans
    assert tuple(trainer._validation_blt_plan_cache.values()) == cached_plans
    assert repeated.diffusion_loss == metrics.diffusion_loss


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
    assert val_entry["val_bpb"] == pytest.approx(validation.bpb, abs=1e-6)
    assert val_entry["val_proxy_bpb"] == pytest.approx(
        validation.bpb, abs=1e-6
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
        validation.bpb, abs=1e-6
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


def test_update_can_skip_unlogged_metric_materialization() -> None:
    trainer = _trainer(
        ByteDiffusionModel(ByteDiffusionConfig.tiny()),
        _run_config(iterations=2),
    )

    assert trainer.run_update(materialize_metrics=False) is None
    assert trainer.completed_steps == 1
    metrics = trainer.run_update(materialize_metrics=True)

    assert metrics is not None
    assert metrics.step == 2
    assert trainer.training_time_ms > 0
