from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pretraining.byte_diffusion.config import ByteDiffusionConfig, model_config_from_env
from pretraining.byte_diffusion.duo import DuoSchedule
from pretraining.byte_diffusion.duo_model import (
    DUO_PARAMETER_COUNT,
    CleanAtomEmbedding,
    DuoModel,
)
from pretraining.byte_diffusion.diffusion_gemma_model import (
    compile_stable_document_metadata,
)
from pretraining.byte_diffusion.inference_duo import (
    generate_duo_continuation,
    sample_duo_canvas,
)
from pretraining.byte_diffusion.training_duo import (
    JOINT_DUO_CLEAN_AR_OBJECTIVE,
    PURE_DUO_OBJECTIVE,
    DuoBatch,
    accumulate_duo_validation_stats,
    duo_validation_from_stats,
    duo_loss,
    duo_mutable_topology_contract,
    duo_objective_contract,
    prepare_duo_inputs,
    prepare_duo_update,
    prepare_duo_validation_batch,
    prepare_duo_validation_inputs,
    validate_duo,
)
from pretraining.byte_diffusion.training import TrainingBatch
from scripts.train_byte_duo import (
    REPO_ROOT,
    _authenticated_entropy_patcher_bytes,
    _local_imports,
    _recipe_batch,
    capture_rng_state,
    parse_args,
    production_batch_geometry,
    restore_rng_state,
    source_provenance,
    validate_readiness_evidence,
)
from pretraining.byte_diffusion.variable_patching import DatasetPatchingSpec
import pretraining.byte_diffusion.training_duo as training_duo_module
import pretraining.byte_diffusion.inference_duo as inference_duo_module


def test_local_import_closure_counts_parent_package_initializers() -> None:
    imports = set(_local_imports(REPO_ROOT / "scripts/train_byte_duo.py"))

    assert (REPO_ROOT / "pretraining/__init__.py").resolve() in imports
    assert (
        REPO_ROOT / "pretraining/byte_diffusion/__init__.py"
    ).resolve() in imports


def test_training_preflight_loads_authenticated_entropy_patcher(tmp_path) -> None:
    payload = b"authenticated entropy patcher"
    (tmp_path / "patcher.bin").write_bytes(payload)
    patching = DatasetPatchingSpec(
        name="causal_entropy_v1",
        patch_stride=None,
        max_patch_size=8,
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
    )
    manifest = {"patching": {"patcher_artifact": {"path": "patcher.bin"}}}

    assert (
        _authenticated_entropy_patcher_bytes(tmp_path, manifest, patching)
        == payload
    )


def test_duo_cli_inherits_ablation_schedule_and_batch_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["train_byte_duo.py"])
    monkeypatch.setenv("ITERATIONS", "200")
    monkeypatch.setenv("WARMDOWN_ITERS", "0")
    monkeypatch.setenv("VAL_LOSS_EVERY", "50")
    monkeypatch.setenv("TRAIN_LOG_EVERY", "7")
    monkeypatch.setenv("BYTE_DUO_MICROBATCH", "11")
    monkeypatch.setenv("BYTE_DUO_VALIDATION_BATCH", "13")
    args = parse_args()
    assert (
        args.steps,
        args.warmdown_steps,
        args.val_every,
        args.log_every,
        args.batch_size,
        args.validation_batch_size,
    ) == (200, 0, 50, 7, 11, 13)
    assert args.objective == PURE_DUO_OBJECTIVE


def test_duo_cli_exposes_only_named_fixed_joint_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["train_byte_duo.py"])
    monkeypatch.setenv("BYTE_DUO_OBJECTIVE", JOINT_DUO_CLEAN_AR_OBJECTIVE)

    args = parse_args()

    assert args.objective == JOINT_DUO_CLEAN_AR_OBJECTIVE
    assert duo_objective_contract(args.objective)["clean_ar_weight"] == 1.0
    with pytest.raises(ValueError, match="unknown Byte-Duo training objective"):
        duo_objective_contract("joint_nelbo_clean_ar_weight_0_5")


def test_production_batch_geometry_eliminates_h100_accumulation() -> None:
    assert production_batch_geometry(1) == (16, 64)
    assert production_batch_geometry(8) == (32, 32)
    with pytest.raises(ValueError, match="world size 1 or 8"):
        production_batch_geometry(2)


def test_single_gpu_production_microbatch_is_selected_by_readiness() -> None:
    from scripts.train_byte_duo import production_microbatches

    assert production_microbatches(1) == (12, 15, 16, 24, 32)
    assert production_microbatches(8) == (32,)
    with pytest.raises(ValueError):
        production_microbatches(2)


def test_world8_cli_defaults_to_one_local_training_and_validation_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["train_byte_duo.py"])
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.delenv("BYTE_DUO_MICROBATCH", raising=False)
    monkeypatch.delenv("BYTE_DUO_VALIDATION_BATCH", raising=False)

    args = parse_args()

    assert args.batch_size == 32
    assert args.validation_batch_size == 32


def tiny_config(**overrides: object) -> ByteDiffusionConfig:
    return ByteDiffusionConfig.tiny(
        local_dim=16,
        global_dim=32,
        local_heads=1,
        global_heads=1,
        encoder_layers=1,
        global_layers=1,
        decoder_layers=1,
        encoder_ffn_dim=32,
        global_ffn_dim=48,
        decoder_ffn_dim=32,
        local_window=16,
        decoder_prefix_window=16,
        **overrides,
    )


def one_document_batch(batch_size: int = 2) -> DuoBatch:
    ids = torch.randint(
        0, 256, (batch_size, 16), generator=torch.Generator().manual_seed(101)
    )
    valid = torch.ones_like(ids, dtype=torch.bool)
    documents = torch.arange(batch_size)[:, None].expand_as(ids)
    positions = torch.arange(16)[None].expand_as(ids)
    targets = torch.full_like(ids, -100)
    targets[:, :-1] = ids[:, 1:]
    return DuoBatch(
        ids,
        valid,
        documents,
        positions,
        targets,
        ids[:, 0].clone(),
        bos_row_indices=torch.arange(batch_size),
    )


def test_duo_recipe_uses_physical_patch_offsets_for_short_final_patches() -> None:
    ids = torch.tensor([[1, 2, 3, 262, 10, 11, 12, 13]])
    valid = torch.tensor([[True, True, True, False, True, True, True, True]])
    native = TrainingBatch(
        ids=ids,
        valid=valid,
        ar_targets=torch.full_like(ids, -100),
        bos_targets=torch.tensor([1, 10]),
        positions=torch.tensor([[0, 1, 2, 0, 0, 1, 2, 3]]),
        full_valid=False,
        document_ids=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]),
        isolate_documents=True,
        byte_indices=torch.tensor([0, 1, 2, 4, 5, 6, 7]),
        byte_cu_seqlens=torch.tensor([0, 3, 7], dtype=torch.int32),
        patch_indices=torch.tensor([0, 1]),
        patch_cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int32),
    )

    batch = _recipe_batch(native, patch_stride=4)

    assert batch.attention_metadata is not None
    assert batch.attention_metadata.byte_indices.tolist() == list(range(8))
    assert batch.attention_metadata.patch_indices.tolist() == [0, 1]
    assert batch.attention_metadata.byte_cu_seqlens[:3].tolist() == [0, 4, 8]
    assert batch.attention_metadata.patch_cu_seqlens[:3].tolist() == [0, 1, 2]
    assert batch.attention_metadata.byte_cu_seqlens.shape == (65,)
    assert batch.attention_metadata.patch_cu_seqlens.shape == (65,)
    assert bool(batch.attention_metadata.byte_cu_seqlens[3:].eq(8).all())
    assert bool(batch.attention_metadata.patch_cu_seqlens[3:].eq(2).all())


def test_duo_recipe_builds_ragged_entropy_metadata_without_fixed_patch_axis() -> None:
    ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 262]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    native = TrainingBatch(
        ids=ids,
        valid=valid,
        ar_targets=torch.full_like(ids, -100),
        bos_targets=torch.tensor([10]),
        positions=torch.arange(8)[None],
        full_valid=True,
        document_ids=torch.zeros_like(ids),
        isolate_documents=True,
        patch_offsets=torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2]]),
        max_patch_size=4,
    )

    batch = _recipe_batch(native, patch_stride=4)

    assert batch.attention_metadata is None
    assert batch.clean_patch_metadata is not None
    assert batch.clean_patch_metadata.byte_indices.tolist() == list(range(8))
    assert batch.clean_patch_metadata.pool_byte_indices.tolist() == list(range(8))
    assert batch.clean_patch_metadata.patch_byte_cu_seqlens.tolist() == [0, 3, 5, 8]
    assert batch.clean_patch_metadata.patch_ordinals.tolist() == [0, 1, 2]
    assert batch.patch_offsets is native.patch_offsets
    assert batch.max_patch_size == 4


def test_entropy_clean_policy_requires_full_resolution_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BYTE_DUO_CLEAN_PATCHING", "causal_entropy_v1")
    monkeypatch.setenv("BYTE_DUO_MUTABLE_TOPOLOGY", "full_resolution_decoder")
    monkeypatch.setenv("BYTE_DUO_RANDOM_PHASE_TRAINING", "1")

    config = model_config_from_env()

    assert config.duo_clean_patching == "causal_entropy_v1"
    assert duo_mutable_topology_contract(config)["compile_dynamic_shapes"] is True
    with pytest.raises(ValueError, match="require full_resolution_decoder"):
        replace(config, duo_mutable_topology="patched_global")


def test_compile_stable_padding_preserves_real_token_and_branch_outputs() -> None:
    model = DuoModel(tiny_config()).eval()
    ids = torch.tensor([[1, 2, 3, 262, 10, 11, 12, 13]])
    valid = torch.tensor([[True, True, True, False, True, True, True, True]])
    documents = torch.tensor([[0, 0, 0, -1, 1, 1, 1, 1]])
    positions = torch.tensor([[0, 1, 2, 0, 0, 1, 2, 3]])
    noisy = torch.tensor([[[10, 11, 12, 13, 262, 262, 262, 262]]])
    branch_valid = torch.tensor(
        [[[True, True, True, True, False, False, False, False]]]
    )
    branch_starts = torch.tensor([[4]])
    times = torch.tensor([[0.5]])
    packed = model.prepare_attention_metadata(valid, documents)
    stable = compile_stable_document_metadata(
        valid,
        documents,
        patch_stride=4,
        max_segments_per_row=4,
    )
    assert stable.physical_layout is True
    assert packed.physical_layout is False

    expected = model(
        ids,
        valid,
        documents,
        positions,
        noisy,
        branch_valid,
        branch_starts,
        times,
        attention_metadata=packed,
        return_clean_logits=True,
    )
    observed = model(
        ids,
        valid,
        documents,
        positions,
        noisy,
        branch_valid,
        branch_starts,
        times,
        attention_metadata=stable,
        return_clean_logits=True,
    )
    assert expected.clean_logits is not None and observed.clean_logits is not None
    torch.testing.assert_close(
        observed.clean_logits[valid], expected.clean_logits[valid], atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        observed.branch_logits, expected.branch_logits, atol=2e-5, rtol=2e-5
    )


def test_compile_stable_metadata_enforces_per_row_capacity_and_document_order() -> None:
    valid = torch.ones((2, 8), dtype=torch.bool)
    documents = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3], [4, 4, 4, 4, 4, 4, 4, 4]]
    )
    with pytest.raises(ValueError, match="one physical row exceeds"):
        compile_stable_document_metadata(
            valid,
            documents,
            patch_stride=2,
            max_segments_per_row=3,
        )

    decreasing = documents.clone()
    decreasing[0, 4:] = 0
    with pytest.raises(ValueError, match="nondecreasing"):
        compile_stable_document_metadata(
            valid,
            decreasing,
            patch_stride=2,
            max_segments_per_row=8,
        )


def test_production_duo_parameter_and_artifact_budget_has_no_mask_embedding() -> None:
    model = DuoModel(ByteDiffusionConfig())
    model.validate_production_parameterization()
    assert model.parameter_count == 24_094_791
    assert model.output is not None and model.output.bias is not None
    assert torch.count_nonzero(model.output.bias) == 0
    assert model.estimated_quantized_artifact_bytes == 13_434_865
    assert model.estimated_quantized_artifact_bytes < 16_000_000
    assert isinstance(model.embedding, CleanAtomEmbedding)
    assert model.embedding.weight.shape[0] == 262  # 261 clean atoms + private PAD.
    assert model.output.weight.shape[0] == 261
    names = {name for name, _ in model.named_parameters()}
    assert "time_adaln.weight" in names
    assert "time_adaln.bias" in names
    assert not any("encoder_adaln" in name for name in names)

    topology = duo_mutable_topology_contract(model.config)
    assert topology["mutable_origin_stride"] == 4
    assert topology["serving_origin_policy"] == (
        "floor_to_patch_and_carry_clean_phase"
    )


def test_decoder_reallocation_is_parameter_matched_and_under_budget() -> None:
    config = ByteDiffusionConfig(
        global_layers=8,
        decoder_layers=4,
        decoder_ffn_dim=688,
    )
    model = DuoModel(config)

    assert model.parameter_count == 24_078_409
    assert 24_094_791 - model.parameter_count == 16_382
    assert model.estimated_quantized_artifact_bytes < 16_000_000


def test_full_resolution_reallocation_spends_only_live_time_conditioning_budget() -> None:
    config = ByteDiffusionConfig(
        global_layers=8,
        decoder_layers=4,
        decoder_ffn_dim=960,
        duo_mutable_topology="full_resolution_decoder",
        duo_random_phase_training=True,
        duo_noisy_ngrams=False,
    )
    model = DuoModel(config)

    assert model.parameter_count == 24_052_297
    assert DUO_PARAMETER_COUNT - model.parameter_count == 42_494
    assert model.estimated_quantized_artifact_bytes < 16_000_000
    assert len(model._adaln_widths) == config.decoder_layers + 1


def test_full_resolution_environment_selects_authenticated_live_budget_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BYTE_DUO_MUTABLE_TOPOLOGY", "full_resolution_decoder")
    monkeypatch.setenv("BYTE_DUO_RANDOM_PHASE_TRAINING", "1")
    config = model_config_from_env()
    model = DuoModel(config)

    assert config.global_layers == 8
    assert config.decoder_layers == 4
    assert config.decoder_ffn_dim == 960
    assert config.duo_noisy_ngrams is False
    assert model.parameter_count == 24_052_297


def test_model_rejects_mask_and_time_is_mandatory_and_trainable() -> None:
    model = DuoModel(tiny_config())
    batch = one_document_batch()
    noisy = batch.clean_ids[:, None, :8].clone()
    starts = torch.zeros((2, 1), dtype=torch.long)
    branch_valid = torch.ones((2, 1, 8), dtype=torch.bool)
    with pytest.raises(TypeError):
        model(
            batch.clean_ids,
            batch.clean_valid,
            batch.document_ids,
            batch.positions,
            noisy,
            branch_valid,
            starts,
        )
    noisy[0, 0, 0] = model.config.vocab.mask_id
    with pytest.raises(ValueError, match="forbids the absorbing MASK"):
        model(
            batch.clean_ids,
            batch.clean_valid,
            batch.document_ids,
            batch.positions,
            noisy,
            branch_valid,
            starts,
            torch.full((2, 1), 0.5),
        )
    noisy[0, 0, 0] = batch.clean_ids[0, 0]
    output = model(
        batch.clean_ids,
        batch.clean_valid,
        batch.document_ids,
        batch.positions,
        noisy,
        branch_valid,
        starts,
        torch.tensor([[0.2], [0.8]]),
        return_clean_logits=True,
    )
    torch.nn.functional.cross_entropy(
        output.branch_logits.flatten(0, -2),
        batch.clean_ids[:, :8].reshape(-1),
    ).backward()
    # DiT zero-initializes both the final head and AdaLN projections. The first
    # optimizer update opens the head; time/backbone gradients begin after it.
    assert model.output.weight.grad is not None
    assert bool((model.output.weight.grad != 0).any())
    assert model.time_adaln.weight.grad is not None
    assert not bool((model.time_adaln.weight.grad != 0).any())


def test_compiled_embedding_fails_closed_on_forbidden_mask() -> None:
    embedding = CleanAtomEmbedding(261, 8, external_pad_id=262)
    compiled = torch.compile(embedding, dynamic=True, fullgraph=False)
    with pytest.raises(RuntimeError, match="forbids the absorbing MASK"):
        compiled(torch.tensor([261], dtype=torch.long))


def test_canvas_is_bidirectional_but_prefix_is_causal_and_document_isolated() -> None:
    model = DuoModel(tiny_config()).eval()
    ids = torch.randint(0, 256, (1, 16), generator=torch.Generator().manual_seed(103))
    valid = torch.ones_like(ids, dtype=torch.bool)
    documents = torch.tensor([[0] * 8 + [1] * 8])
    positions = torch.tensor([list(range(8)) + list(range(8))])
    starts = torch.tensor([[0, 8]])
    branch_valid = torch.ones((1, 2, 8), dtype=torch.bool)
    noisy = torch.randint(0, 256, (1, 2, 8), generator=torch.Generator().manual_seed(107))
    times = torch.tensor([[0.25, 0.75]])
    first = model(
        ids,
        valid,
        documents,
        positions,
        noisy,
        branch_valid,
        starts,
        times,
        return_clean_logits=True,
    )

    changed_ids = ids.clone()
    changed_ids[:, 8:] = (changed_ids[:, 8:] + 71) % 256
    changed_noisy = noisy.clone()
    changed_noisy[:, 1] = (changed_noisy[:, 1] + 89) % 256
    second = model(
        changed_ids,
        valid,
        documents,
        positions,
        changed_noisy,
        branch_valid,
        starts,
        times,
        return_clean_logits=True,
    )
    torch.testing.assert_close(first.clean_logits[:, :8], second.clean_logits[:, :8])
    torch.testing.assert_close(first.branch_logits[:, 0], second.branch_logits[:, 0])

    # Clean values under a canvas are future information: its branch sees the
    # noisy states at those positions, not the clean bank's future values.
    same_noisy = noisy.clone()
    changed_future = ids.clone()
    changed_future[:, 4:8] = (changed_future[:, 4:8] + 41) % 256
    future = model(
        changed_future,
        valid,
        documents,
        positions,
        same_noisy,
        branch_valid,
        starts,
        times,
        return_clean_logits=True,
    )
    torch.testing.assert_close(first.branch_logits[:, 0], future.branch_logits[:, 0])


def test_time_conditioning_changes_only_noisy_branch_and_clean_bank_is_cacheable() -> None:
    model = DuoModel(tiny_config()).eval()
    with torch.no_grad():
        final_width = 2 * model.config.local_dim
        model.time_adaln.weight[-final_width:].normal_(std=0.03)
        model.time_embedding[-1].bias.normal_(std=0.01)
        model.output.weight.normal_(std=0.02)
    batch = one_document_batch()
    noisy = batch.clean_ids[:, None, :8]
    valid = torch.ones((2, 1, 8), dtype=torch.bool)
    starts = torch.zeros((2, 1), dtype=torch.long)
    early = model(
        batch.clean_ids,
        batch.clean_valid,
        batch.document_ids,
        batch.positions,
        noisy,
        valid,
        starts,
        torch.full((2, 1), 0.1),
        return_clean_logits=True,
    )
    late = model(
        batch.clean_ids,
        batch.clean_valid,
        batch.document_ids,
        batch.positions,
        noisy,
        valid,
        starts,
        torch.full((2, 1), 0.9),
        return_clean_logits=True,
    )
    torch.testing.assert_close(early.clean_logits, late.clean_logits)
    assert not torch.equal(early.branch_logits, late.branch_logits)


def test_reference_sized_time_conditioner_is_an_explicit_model_cell() -> None:
    config = tiny_config(duo_time_features=256, duo_time_condition_dim=128)
    model = DuoModel(config)

    assert model.time_embedding[0].in_features == 256
    assert model.time_embedding[0].out_features == 128
    assert model.time_adaln.in_features == 128


def test_noisy_ngram_ablation_retains_clean_ngrams_only() -> None:
    model = DuoModel(tiny_config(ngram_enabled=True, duo_noisy_ngrams=False)).eval()
    batch = one_document_batch()
    with torch.no_grad():
        clean = model.prepare_clean_bank(
            batch.clean_ids,
            batch.clean_valid,
            batch.document_ids,
            batch.positions,
        )

    assert model.ngrams is not None
    assert model._branch_ngram_features(
        clean,
        batch.clean_ids[:, None, :8],
        torch.ones((2, 1, 8), dtype=torch.bool),
        torch.zeros((2, 1), dtype=torch.long),
    ) is None

    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim:
                parameter.normal_(std=0.02)
        noisy = batch.clean_ids[:, None, :8].clone()
        branch_valid = torch.ones((2, 1, 8), dtype=torch.bool)
        starts = torch.zeros((2, 1), dtype=torch.long)
        times = torch.tensor([[0.2], [0.8]])
        expected = model(
            batch.clean_ids,
            batch.clean_valid,
            batch.document_ids,
            batch.positions,
            noisy,
            branch_valid,
            starts,
            times,
            return_clean_logits=True,
        )
        clean = model.prepare_clean_bank(
            batch.clean_ids,
            batch.clean_valid,
            batch.document_ids,
            batch.positions,
        )
        cache = model.prepare_canvas_cache(clean, branch_valid, starts)
        observed = model.forward_prepared(
            cache, noisy, times, return_clean_logits=True
        )

    torch.testing.assert_close(observed.branch_logits, expected.branch_logits)
    torch.testing.assert_close(observed.clean_logits, expected.clean_logits)


def test_prepared_clean_bank_has_exact_full_forward_parity() -> None:
    model = DuoModel(tiny_config()).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim:
                parameter.normal_(std=0.02)
        batch = one_document_batch()
        noisy = batch.clean_ids[:, None, :8].clone()
        branch_valid = torch.ones((2, 1, 8), dtype=torch.bool)
        starts = torch.zeros((2, 1), dtype=torch.long)
        times = torch.tensor([[0.2], [0.8]])
        expected = model(
            batch.clean_ids,
            batch.clean_valid,
            batch.document_ids,
            batch.positions,
            noisy,
            branch_valid,
            starts,
            times,
            return_clean_logits=True,
        )
        clean_bank = model.prepare_clean_bank(
            batch.clean_ids,
            batch.clean_valid,
            batch.document_ids,
            batch.positions,
        )
        canvas_cache = model.prepare_canvas_cache(
            clean_bank, branch_valid, starts
        )
        observed = model.forward_prepared(
            canvas_cache, noisy, times, return_clean_logits=True
        )

    torch.testing.assert_close(observed.branch_logits, expected.branch_logits)
    torch.testing.assert_close(observed.clean_logits, expected.clean_logits)
    torch.testing.assert_close(
        observed.clean_patch_states, expected.clean_patch_states
    )
    torch.testing.assert_close(
        observed.branch_patch_states, expected.branch_patch_states
    )


def test_canvas_sampler_projects_clean_kv_once_and_reuses_it_for_every_nfe() -> None:
    model = DuoModel(tiny_config()).eval()
    batch = one_document_batch()
    observed_lengths: dict[str, list[int]] = {
        "encoder": [],
        "global": [],
        "decoder": [],
    }
    handles = []
    for name, block in (
        ("encoder", model.encoder[0]),
        ("global", model.global_blocks[0]),
        ("decoder", model.decoder[0].block),
    ):
        handles.append(
            block.attention.qkv.register_forward_pre_hook(
                lambda _module, inputs, name=name: observed_lengths[name].append(
                    inputs[0].shape[1]
                )
            )
        )
    try:
        result = sample_duo_canvas(
            model,
            batch.clean_ids,
            batch.clean_valid,
            batch.document_ids,
            batch.positions,
            torch.zeros((2, 1), dtype=torch.long),
            torch.ones((2, 1, 8), dtype=torch.bool),
            steps=2,
            generator=torch.Generator().manual_seed(149),
            return_trajectory=False,
        )
    finally:
        for handle in handles:
            handle.remove()

    # One clean preparation plus three branch-only denoiser evaluations.
    assert observed_lengths == {
        "encoder": [16, 8, 8, 8],
        "global": [4, 2, 2, 2],
        "decoder": [16, 8, 8, 8],
    }
    assert result.clean_bank_forwards == 1
    assert result.denoising_forwards == 3
    assert result.model_forwards == 4


def test_multicanvas_continuation_rebuilds_clean_cache_only_after_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DuoModel(tiny_config()).eval()
    prompt = torch.tensor([[10, 11, 12]])
    original_prepare = model.prepare_clean_bank
    prepared = 0

    def counted_prepare(*args, **kwargs):
        nonlocal prepared
        prepared += 1
        return original_prepare(*args, **kwargs)

    def fixed_sample(logits, noisy_ids, _alpha_s, _alpha_t, _uniforms, active, **_):
        del logits
        return torch.where(active, torch.full_like(noisy_ids, 7), noisy_ids)

    monkeypatch.setattr(model, "prepare_clean_bank", counted_prepare)
    monkeypatch.setattr(inference_duo_module, "duo_posterior_sample", fixed_sample)
    result = generate_duo_continuation(
        model,
        prompt,
        torch.ones_like(prompt, dtype=torch.bool),
        canvas_length=8,
        max_new_atoms=10,
        steps=1,
        generator=torch.Generator().manual_seed(151),
    )

    assert result.canvases == 2
    assert prepared == 2
    assert result.cache_backed
    assert result.clean_prefill_forwards == 1
    assert result.clean_commit_forwards == 1
    assert result.denoising_forwards == 4
    assert result.model_forwards == 6


def test_pure_duo_loss_never_unembeds_or_cross_entropies_clean_bank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DuoModel(tiny_config()).eval()
    batch = one_document_batch()
    observed_shapes: list[tuple[int, ...]] = []
    handle = model.output.register_forward_hook(
        lambda _module, inputs, _output: observed_shapes.append(tuple(inputs[0].shape))
    )
    def reject_cross_entropy(*_args, **_kwargs):
        raise AssertionError("pure Duo must not call cross_entropy")

    monkeypatch.setattr(training_duo_module.F, "cross_entropy", reject_cross_entropy)
    try:
        loss = duo_loss(
            model,
            batch,
            canvas_length=8,
            branches=1,
            generator=torch.Generator().manual_seed(108),
            clean_ar_weight=0.0,
        )
    finally:
        handle.remove()
    assert observed_shapes == [(2, 8, 16)]
    assert int(loss.ar_targets) == 0
    assert float(loss.ar_nll_sum) == 0.0


def test_duo_clean_states_are_exact_inputs_to_the_output_projection() -> None:
    model = DuoModel(tiny_config()).eval()
    batch = one_document_batch()
    prepared = prepare_duo_inputs(
        model,
        batch,
        canvas_length=8,
        branches=1,
        generator=torch.Generator().manual_seed(108),
    )
    output = model(
        batch.clean_ids,
        batch.clean_valid,
        batch.document_ids,
        batch.positions,
        prepared.corruption.ids.view(batch.clean_ids.shape[0], 1, 8),
        prepared.selection.valid,
        prepared.selection.starts,
        prepared.corruption.t.view(batch.clean_ids.shape[0], 1),
        return_clean_logits=True,
        return_clean_states=True,
    )
    assert output.clean_decoder_states is not None
    expected = torch.nn.functional.linear(
        output.clean_decoder_states,
        model.output.weight,
        model.output.bias,
    )
    torch.testing.assert_close(output.clean_logits, expected)


def test_joint_ar_projection_unwraps_ddp_around_compiled_duo_model() -> None:
    model = DuoModel(tiny_config()).eval()
    wrapped = SimpleNamespace(
        module=SimpleNamespace(_orig_mod=model),
        config=model.config,
        forward_bos_logits=model.forward_bos_logits,
    )
    batch = one_document_batch()
    states = torch.randn(
        *batch.clean_ids.shape,
        model.config.local_dim,
        requires_grad=True,
    )

    total, count = training_duo_module._ar_sum_and_count(wrapped, states, batch)
    total.backward()

    expected_count = batch.ar_targets.ne(-100).sum() + batch.bos_targets.numel()
    assert int(count) == int(expected_count)
    assert states.grad is not None
    assert model.output.weight.grad is not None


def test_joint_duo_update_splits_compact_bos_and_preserves_global_normalization() -> None:
    model = DuoModel(tiny_config()).eval()
    base = one_document_batch(5)
    batch = replace(
        base,
        bos_targets=torch.tensor([3, 5, 7, 11, 13, 17]),
        bos_row_indices=torch.tensor([0, 0, 1, 3, 3, 4]),
        attention_metadata=compile_stable_document_metadata(
            base.clean_valid,
            base.document_ids,
            patch_stride=model.config.patch_stride,
            max_segments_per_row=4,
        ),
    )
    schedule = DuoSchedule()
    times = torch.linspace(schedule.eps, 0.9, 5)
    split = prepare_duo_update(
        model,
        batch,
        microbatch_size=2,
        canvas_length=8,
        branches=1,
        schedule=schedule,
        times=times,
        generator=torch.Generator().manual_seed(109),
        include_clean_ar=True,
    )
    full = prepare_duo_update(
        model,
        batch,
        microbatch_size=5,
        canvas_length=8,
        branches=1,
        schedule=schedule,
        times=times,
        generator=torch.Generator().manual_seed(109),
        include_clean_ar=True,
    )

    assert [item.batch.bos_targets.tolist() for item in split.microbatches] == [
        [3, 5, 7],
        [11, 13],
        [17],
    ]
    assert [item.batch.bos_row_indices.tolist() for item in split.microbatches] == [
        [0, 0, 1],
        [1, 1],
        [0],
    ]

    def update_loss(prepared_update) -> torch.Tensor:
        return sum(
            (
                duo_loss(
                    model,
                    item.batch,
                    canvas_length=8,
                    branches=1,
                    prepared=item.prepared,
                    schedule=schedule,
                    nelbo_denominator=prepared_update.nelbo_denominator,
                    ar_denominator=prepared_update.ar_denominator,
                    clean_ar_weight=1.0,
                ).total
                for item in prepared_update.microbatches
            ),
            torch.zeros(()),
        )

    torch.testing.assert_close(update_loss(split), update_loss(full))


def test_update_global_normalization_is_exact_with_249_style_short_tail() -> None:
    model = DuoModel(tiny_config()).eval()
    batch = one_document_batch(5)
    schedule = DuoSchedule()
    times = torch.linspace(schedule.eps, 0.9, 5)
    prepared = prepare_duo_inputs(
        model,
        batch,
        canvas_length=8,
        branches=1,
        times=times,
        schedule=schedule,
        generator=torch.Generator().manual_seed(109),
    )
    denominator = prepared.corruption.active.sum()
    ar_denominator = batch.ar_targets.ne(-100).sum() + batch.bos_targets.numel()
    full = duo_loss(
        model,
        batch,
        canvas_length=8,
        branches=1,
        prepared=prepared,
        schedule=schedule,
        nelbo_denominator=denominator,
        ar_denominator=ar_denominator,
    )
    split = full.total.new_zeros(())
    for start, stop in ((0, 2), (2, 4), (4, 5)):
        part = DuoBatch(
            batch.clean_ids[start:stop],
            batch.clean_valid[start:stop],
            batch.document_ids[start:stop],
            batch.positions[start:stop],
            batch.ar_targets[start:stop],
            batch.bos_targets[start:stop],
        )
        split += duo_loss(
            model,
            part,
            canvas_length=8,
            branches=1,
            prepared=prepared.slice_rows(start, stop),
            schedule=schedule,
            nelbo_denominator=denominator,
            ar_denominator=ar_denominator,
        ).total
    torch.testing.assert_close(split, full.total, atol=2e-5, rtol=2e-5)


def test_random_phase_training_exposes_fixed_clean_prefix_without_supervising_it() -> None:
    model = DuoModel(tiny_config(duo_random_phase_training=True)).eval()
    batch = one_document_batch(16)
    prepared = prepare_duo_inputs(
        model,
        batch,
        canvas_length=8,
        branches=2,
        generator=torch.Generator().manual_seed(110),
    )
    corruption = prepared.corruption

    assert corruption.fixed_clean is not None
    assert bool(corruption.fixed_clean.any())
    assert not bool((corruption.fixed_clean & corruption.active).any())
    torch.testing.assert_close(
        corruption.ids.masked_select(corruption.fixed_clean),
        corruption.targets.masked_select(corruption.fixed_clean),
    )
    assert bool(
        corruption.ids.masked_select(
            ~(corruption.active | corruption.fixed_clean)
        ).eq(model.config.vocab.pad_id).all()
    )
    corruption.validate(
        clean_atoms=model.config.vocab.output_size,
        pad_id=model.config.vocab.pad_id,
    )


def test_random_phase_cell_keeps_canonical_validation_all_active() -> None:
    model = DuoModel(tiny_config(duo_random_phase_training=True)).eval()
    batch = one_document_batch(8)
    row_ids = torch.arange(8)
    canonical = prepare_duo_validation_inputs(
        model,
        batch,
        row_ids=row_ids,
        total_rows=8,
        canvas_length=8,
        branches=2,
        seed=111,
    )
    phase_robustness = prepare_duo_validation_inputs(
        model,
        batch,
        row_ids=row_ids,
        total_rows=8,
        canvas_length=8,
        branches=2,
        seed=111,
        expose_random_phase=True,
    )

    assert canonical.corruption.fixed_clean is not None
    assert not bool(canonical.corruption.fixed_clean.any())
    assert canonical.corruption.active.sum() == canonical.selection.valid.sum()
    assert phase_robustness.corruption.fixed_clean is not None
    assert bool(phase_robustness.corruption.fixed_clean.any())


def test_prepared_update_slices_compile_stable_rows_once_with_exact_tail() -> None:
    model = DuoModel(tiny_config()).eval()
    batch = one_document_batch(5)
    batch = replace(
        batch,
        attention_metadata=compile_stable_document_metadata(
            batch.clean_valid,
            batch.document_ids,
            patch_stride=model.config.patch_stride,
            max_segments_per_row=4,
        ),
    )
    update = prepare_duo_update(
        model,
        batch,
        microbatch_size=2,
        canvas_length=8,
        branches=2,
        generator=torch.Generator().manual_seed(110),
    )

    assert [item.batch.clean_ids.shape[0] for item in update.microbatches] == [2, 2, 1]
    assert int(update.nelbo_denominator) == int(
        sum(item.prepared.corruption.active.sum() for item in update.microbatches)
    )
    assert int(update.ar_denominator) == int(
        batch.ar_targets.ne(-100).sum() + batch.bos_targets.numel()
    )
    for item in update.microbatches:
        metadata = item.batch.attention_metadata
        assert metadata is not None
        rows = item.batch.clean_ids.shape[0]
        assert metadata.byte_indices.tolist() == list(range(rows * 16))
        assert metadata.patch_indices.tolist() == list(range(rows * 4))
        assert metadata.byte_cu_seqlens.shape == (rows * 4 + 1,)
        assert metadata.patch_cu_seqlens.shape == (rows * 4 + 1,)
        assert int(metadata.byte_cu_seqlens[-1]) == rows * 16
        assert int(metadata.patch_cu_seqlens[-1]) == rows * 4
        assert item.batch.bos_targets.numel() == 0


def test_prepared_update_reuses_full_batch_metadata_without_rebuilding() -> None:
    model = DuoModel(tiny_config()).eval()
    batch = one_document_batch(5)
    batch = replace(
        batch,
        attention_metadata=compile_stable_document_metadata(
            batch.clean_valid,
            batch.document_ids,
            patch_stride=model.config.patch_stride,
            max_segments_per_row=4,
        ),
    )

    update = prepare_duo_update(
        model,
        batch,
        microbatch_size=8,
        canvas_length=8,
        branches=2,
        generator=torch.Generator().manual_seed(111),
    )

    assert len(update.microbatches) == 1
    assert update.microbatches[0].batch is batch


def test_prepared_duo_inputs_pin_every_transfer_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DuoModel(tiny_config()).eval()
    prepared = prepare_duo_inputs(
        model,
        one_document_batch(),
        canvas_length=8,
        branches=1,
        generator=torch.Generator().manual_seed(111),
    )
    pinned: list[int] = []
    pinned_contiguous: list[bool] = []

    def record_pin(tensor: torch.Tensor) -> torch.Tensor:
        pinned.append(tensor.data_ptr())
        pinned_contiguous.append(tensor.is_contiguous())
        return tensor

    monkeypatch.setattr(torch.Tensor, "pin_memory", record_pin)
    observed = prepared.pin_memory()
    assert observed.branches == prepared.branches
    assert observed.local_block_mask_metadata is not None
    assert observed.global_block_mask_metadata is not None
    # Five selection fields, seven corruption fields, and two eight-tensor
    # partial/full sparse block-mask metadata objects.
    assert len(pinned) == 28
    assert all(pinned_contiguous)


def test_duo_batch_and_validation_pinning_materialize_expanded_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DuoModel(tiny_config()).eval()
    batch = replace(one_document_batch(), row_ids=torch.arange(2))
    prepared = prepare_duo_validation_batch(
        model,
        batch,
        total_rows=2,
        canvas_length=8,
        branches=1,
        seed=111,
    )
    observed_layouts: list[bool] = []

    def require_contiguous(tensor: torch.Tensor) -> torch.Tensor:
        observed_layouts.append(tensor.is_contiguous())
        return tensor

    monkeypatch.setattr(torch.Tensor, "pin_memory", require_contiguous)
    batch.pin_memory()
    prepared.pin_memory()
    assert observed_layouts
    assert all(observed_layouts)


def test_ancestral_sampling_starts_uniform_never_masks_and_revises_tokens() -> None:
    model = DuoModel(tiny_config()).eval()
    batch = one_document_batch()
    result = sample_duo_canvas(
        model,
        batch.clean_ids,
        batch.clean_valid,
        batch.document_ids,
        batch.positions,
        torch.zeros((2, 1), dtype=torch.long),
        torch.ones((2, 1, 8), dtype=torch.bool),
        steps=2,
        generator=torch.Generator().manual_seed(113),
    )
    assert result.trajectory.shape == (4, 2, 8)
    assert result.denoising_forwards == 3
    assert result.revision_count > 0
    assert not bool(result.trajectory.eq(model.config.vocab.mask_id).any())
    assert bool((result.trajectory < model.config.vocab.output_size).all())


def test_ancestral_sampler_uses_independent_reference_terminal_epsilon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DuoModel(tiny_config()).eval()
    batch = one_document_batch()
    endpoints: list[tuple[float, float]] = []

    def record_sample(logits, noisy_ids, alpha_s, alpha_t, uniforms, active, **_):
        del logits, uniforms
        endpoints.append((float(alpha_s), float(alpha_t)))
        return torch.where(active, noisy_ids, noisy_ids)

    monkeypatch.setattr(inference_duo_module, "duo_posterior_sample", record_sample)
    sample_duo_canvas(
        model,
        batch.clean_ids,
        batch.clean_valid,
        batch.document_ids,
        batch.positions,
        torch.zeros((2, 1), dtype=torch.long),
        torch.ones((2, 1, 8), dtype=torch.bool),
        steps=1,
        terminal_eps=1e-5,
        return_trajectory=False,
    )

    terminal_alpha = 1.0 - (1.0 - model.schedule_eps) * 1e-5
    assert endpoints[0] == pytest.approx((terminal_alpha, model.schedule_eps))
    assert endpoints[1] == pytest.approx((1.0, terminal_alpha))


def test_ancestral_sampler_generates_into_synthetic_suffix_not_noised_input() -> None:
    model = DuoModel(tiny_config()).eval()
    ids = torch.full((1, 16), model.config.vocab.pad_id, dtype=torch.long)
    ids[:, :8] = torch.arange(8)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    valid[:, :8] = True
    documents = torch.zeros_like(ids)
    positions = torch.arange(16)[None]
    result = sample_duo_canvas(
        model,
        ids,
        valid,
        documents,
        positions,
        torch.tensor([[8]]),
        torch.ones((1, 1, 8), dtype=torch.bool),
        steps=1,
        generator=torch.Generator().manual_seed(117),
    )
    assert result.ids.shape == (1, 8)
    assert not bool(result.trajectory.eq(model.config.vocab.mask_id).any())


class _ConstantDuo:
    def __init__(self, token_id: int) -> None:
        self.config = replace(tiny_config(), duo_diffusion_atoms=261)
        self.schedule_eps = 1e-3
        self.token_id = token_id
        self.calls = 0

    def prepare_attention_metadata(self, clean_valid, document_ids):
        del clean_valid, document_ids
        return None

    def __call__(
        self,
        clean_ids,
        clean_valid,
        document_ids,
        positions,
        noisy_ids,
        branch_valid,
        branch_starts,
        t,
        **kwargs,
    ):
        del clean_valid, document_ids, positions, branch_valid, branch_starts, t, kwargs
        self.calls += 1
        logits = torch.full(
            (*noisy_ids.shape, self.config.vocab.output_size),
            -30.0,
            device=clean_ids.device,
        )
        logits[..., self.token_id] = 30.0
        return SimpleNamespace(branch_logits=logits)


def test_duo_continuation_never_noises_prompt_and_uses_multiple_canvases() -> None:
    prompt = torch.tensor([[10, 11, 12]])
    prompt_valid = torch.ones_like(prompt, dtype=torch.bool)
    model = _ConstantDuo(7)
    result = generate_duo_continuation(
        model,
        prompt,
        prompt_valid,
        canvas_length=8,
        max_new_atoms=10,
        steps=1,
        generator=torch.Generator().manual_seed(119),
        return_trajectories=True,
    )
    assert result.ids[0, :3].tolist() == [10, 11, 12]
    assert result.lengths.tolist() == [13]
    assert result.generated_valid.sum().item() == 10
    assert result.canvases == 2
    assert result.denoising_forwards == 4
    assert result.clean_prefill_forwards == 0
    assert result.clean_commit_forwards == 0
    assert result.model_forwards == model.calls == 4
    assert result.trajectories is not None and len(result.trajectories) == 2
    assert all(trajectory.shape[0] == 3 for trajectory in result.trajectories)


@pytest.mark.parametrize(
    ("atom", "message"),
    [
        (-1, "clean output vocabulary"),
        (261, "MASK"),
        (262, "PAD"),
        (263, "clean output vocabulary"),
    ],
)
def test_duo_continuation_rejects_non_clean_prompt_atoms(
    atom: int, message: str
) -> None:
    model = _ConstantDuo(7)
    with pytest.raises(ValueError, match=message):
        generate_duo_continuation(
            model,
            torch.tensor([[10, atom]]),
            torch.ones((1, 2), dtype=torch.bool),
            canvas_length=8,
            max_new_atoms=1,
            steps=1,
        )


def test_duo_commit_width_discards_lookahead_and_advances_by_block() -> None:
    prompt = torch.tensor([[10, 11, 12]])
    model = _ConstantDuo(7)
    result = generate_duo_continuation(
        model,
        prompt,
        torch.ones_like(prompt, dtype=torch.bool),
        canvas_length=8,
        max_new_atoms=6,
        steps=1,
        commit_width=2,
        generator=torch.Generator().manual_seed(121),
    )
    assert result.ids[0, :9].tolist() == [10, 11, 12, 7, 7, 7, 7, 7, 7]
    assert result.lengths.tolist() == [9]
    assert result.canvases == 3
    assert result.denoising_forwards == model.calls == 6


def test_duo_visible_width_one_has_no_noisy_lookahead() -> None:
    prompt = torch.tensor([[10, 11, 12]])
    model = _ConstantDuo(7)
    result = generate_duo_continuation(
        model,
        prompt,
        torch.ones_like(prompt, dtype=torch.bool),
        canvas_length=8,
        max_new_atoms=3,
        steps=1,
        visible_width=1,
        commit_width=1,
        generator=torch.Generator().manual_seed(122),
        return_trajectories=True,
    )
    assert result.canvases == 3
    assert result.trajectory_active is not None
    assert [int(active.sum()) for active in result.trajectory_active] == [1, 1, 1]


def test_duo_variable_length_training_exposes_short_branches() -> None:
    model = DuoModel(
        replace(tiny_config(), duo_variable_length_probability=1.0)
    ).eval()
    batch = one_document_batch(16)
    prepared = prepare_duo_inputs(
        model,
        batch,
        canvas_length=8,
        branches=4,
        generator=torch.Generator().manual_seed(1234),
    )
    widths = prepared.corruption.active.sum(1)
    assert torch.equal(widths, prepared.selection.valid.flatten(0, 1).sum(1))
    assert bool(widths.ge(1).all())
    assert bool(widths.le(8).all())
    assert bool(widths.lt(8).any())


def test_duo_variable_length_and_phase_share_the_exact_visible_topology() -> None:
    model = DuoModel(
        replace(
            tiny_config(),
            duo_random_phase_training=True,
            duo_variable_length_probability=1.0,
        )
    ).eval()
    prepared = prepare_duo_inputs(
        model,
        one_document_batch(16),
        canvas_length=8,
        branches=4,
        generator=torch.Generator().manual_seed(1235),
    )
    visible = prepared.corruption.active | prepared.corruption.fixed_clean
    assert torch.equal(visible, prepared.selection.valid.flatten(0, 1))
    assert bool(visible.sum(1).lt(8).any())
    assert bool(prepared.corruption.fixed_clean.any())


def test_duo_continuation_commits_eot_and_stops() -> None:
    prompt = torch.tensor([[10, 11, 12]])
    model = _ConstantDuo(tiny_config().vocab.eot_id)
    result = generate_duo_continuation(
        model,
        prompt,
        torch.ones_like(prompt, dtype=torch.bool),
        canvas_length=8,
        max_new_atoms=10,
        steps=1,
        generator=torch.Generator().manual_seed(123),
    )
    assert result.ids[0, :4].tolist() == [10, 11, 12, model.config.vocab.eot_id]
    assert result.lengths.tolist() == [4]
    assert result.finished.tolist() == [True]
    assert result.canvases == 1


def test_duo_continuation_retires_text_stop_inside_first_canvas() -> None:
    prompt = torch.tensor([[10, 11, 12]])
    model = _ConstantDuo(ord("\n"))
    result = generate_duo_continuation(
        model,
        prompt,
        torch.ones_like(prompt, dtype=torch.bool),
        canvas_length=8,
        max_new_atoms=16,
        max_new_bytes=16,
        steps=1,
        stop_sequences=(b"\n\n",),
        generator=torch.Generator().manual_seed(123),
    )
    assert result.ids[0, :3].tolist() == [10, 11, 12]
    assert result.lengths.tolist() == [3]
    assert result.generated_valid.sum().item() == 0
    assert result.text_stopped.tolist() == [True]
    assert result.byte_capped.tolist() == [False]
    assert result.denoising_actions.tolist() == [2]
    assert result.model_forwards == model.calls == 2


def test_duo_continuation_excludes_stop_completed_across_prompt_boundary() -> None:
    prompt = torch.tensor([[ord("x"), ord("\n")]])
    model = _ConstantDuo(ord("\n"))
    result = generate_duo_continuation(
        model,
        prompt,
        torch.ones_like(prompt, dtype=torch.bool),
        canvas_length=8,
        max_new_atoms=4,
        max_new_bytes=4,
        steps=1,
        stop_sequences=(b"\n\n",),
        generator=torch.Generator().manual_seed(124),
    )
    assert result.ids[0, :2].tolist() == prompt[0].tolist()
    assert result.lengths.tolist() == [2]
    assert result.generated_valid.sum().item() == 0
    assert result.text_stopped.tolist() == [True]
    assert result.byte_capped.tolist() == [False]


def test_duo_stop_matching_ignores_typed_controls_in_prompt_tail() -> None:
    prompt = torch.tensor([[ord("A"), 257]])
    model = _ConstantDuo(ord("B"))
    result = generate_duo_continuation(
        model,
        prompt,
        torch.ones_like(prompt, dtype=torch.bool),
        canvas_length=8,
        max_new_atoms=4,
        steps=1,
        stop_sequences=(b"AB",),
        generator=torch.Generator().manual_seed(125),
    )
    assert result.ids[0, :2].tolist() == prompt[0].tolist()
    assert result.lengths.tolist() == [2]
    assert result.generated_valid.sum().item() == 0
    assert result.text_stopped.tolist() == [True]


def test_duo_continuation_rolls_back_stop_split_across_ltr_canvases() -> None:
    prompt = torch.tensor([[ord("x")]])
    model = _ConstantDuo(ord("\n"))
    result = generate_duo_continuation(
        model,
        prompt,
        torch.ones_like(prompt, dtype=torch.bool),
        canvas_length=8,
        max_new_atoms=4,
        steps=1,
        visible_width=1,
        commit_width=1,
        stop_sequences=(b"\n\n",),
        generator=torch.Generator().manual_seed(126),
    )
    assert result.ids[0, 0].item() == ord("x")
    assert result.lengths.tolist() == [1]
    assert result.generated_valid.sum().item() == 0
    assert result.text_stopped.tolist() == [True]
    assert result.canvases == 2


def test_duo_byte_cap_wins_before_a_later_stop_completion() -> None:
    prompt = torch.tensor([[ord("x")]])
    model = _ConstantDuo(ord("\n"))
    result = generate_duo_continuation(
        model,
        prompt,
        torch.ones_like(prompt, dtype=torch.bool),
        canvas_length=8,
        max_new_atoms=4,
        max_new_bytes=1,
        steps=1,
        stop_sequences=(b"\n\n",),
        generator=torch.Generator().manual_seed(127),
    )
    assert result.ids[0, :2].tolist() == [ord("x"), ord("\n")]
    assert result.lengths.tolist() == [2]
    assert result.generated_valid.sum().item() == 1
    assert result.text_stopped.tolist() == [False]
    assert result.byte_capped.tolist() == [True]


def test_duo_continuation_tracks_literal_byte_and_atom_caps_separately() -> None:
    prompt = torch.tensor([[10, 11, 12]])
    valid = torch.ones_like(prompt, dtype=torch.bool)
    byte_model = _ConstantDuo(ord("A"))
    byte_result = generate_duo_continuation(
        byte_model,
        prompt,
        valid,
        canvas_length=8,
        max_new_atoms=10,
        max_new_bytes=3,
        steps=1,
        generator=torch.Generator().manual_seed(123),
    )
    assert byte_result.lengths.tolist() == [6]
    assert byte_result.byte_capped.tolist() == [True]

    control_model = _ConstantDuo(257)
    atom_result = generate_duo_continuation(
        control_model,
        prompt,
        valid,
        canvas_length=8,
        max_new_atoms=4,
        max_new_bytes=4,
        steps=1,
        generator=torch.Generator().manual_seed(123),
    )
    assert atom_result.generated_valid.sum().item() == 4
    assert atom_result.byte_capped.tolist() == [False]
    assert atom_result.finished.tolist() == [False]


def test_validation_ledger_is_invariant_to_batching_and_row_order() -> None:
    model = DuoModel(tiny_config()).eval()
    batch = one_document_batch(4)

    def select(indices: torch.Tensor) -> DuoBatch:
        return DuoBatch(
            clean_ids=batch.clean_ids.index_select(0, indices),
            clean_valid=batch.clean_valid.index_select(0, indices),
            document_ids=batch.document_ids.index_select(0, indices),
            positions=batch.positions.index_select(0, indices),
            ar_targets=batch.ar_targets.index_select(0, indices),
            bos_targets=batch.bos_targets.index_select(0, indices),
            row_ids=indices,
        )

    full = prepare_duo_validation_inputs(
        model,
        select(torch.arange(4)),
        row_ids=torch.arange(4),
        total_rows=4,
        canvas_length=8,
        branches=1,
        seed=137,
    )
    permutation = torch.tensor([2, 0, 3, 1])
    shuffled = prepare_duo_validation_inputs(
        model,
        select(permutation),
        row_ids=permutation,
        total_rows=4,
        canvas_length=8,
        branches=1,
        seed=137,
    )
    inverse = permutation.argsort()
    torch.testing.assert_close(
        full.selection.starts,
        shuffled.selection.starts.index_select(0, inverse),
    )
    for name in ("ids", "targets", "active", "replaced", "changed", "t", "alpha"):
        expected = getattr(full.corruption, name)
        observed = getattr(shuffled.corruption, name).index_select(0, inverse)
        torch.testing.assert_close(expected, observed)


def test_validation_time_samples_do_not_shift_with_schedule_epsilon() -> None:
    batch = replace(one_document_batch(4), row_ids=torch.arange(4))
    low_eps = DuoModel(tiny_config(), schedule_eps=1e-3).eval()
    high_eps = DuoModel(tiny_config(), schedule_eps=0.1).eval()

    low = prepare_duo_validation_inputs(
        low_eps,
        batch,
        row_ids=batch.row_ids,
        total_rows=4,
        canvas_length=8,
        branches=2,
        seed=138,
        schedule=DuoSchedule(1e-3),
    )
    high = prepare_duo_validation_inputs(
        high_eps,
        batch,
        row_ids=batch.row_ids,
        total_rows=4,
        canvas_length=8,
        branches=2,
        seed=138,
        schedule=DuoSchedule(0.1),
    )

    torch.testing.assert_close(low.corruption.t, high.corruption.t)
    assert not torch.equal(low.corruption.alpha, high.corruption.alpha)


def test_disjoint_validation_shards_sum_to_the_full_authenticated_ledger() -> None:
    model = DuoModel(tiny_config()).eval()
    batch = one_document_batch(4)

    def select(indices: torch.Tensor) -> DuoBatch:
        return DuoBatch(
            clean_ids=batch.clean_ids.index_select(0, indices),
            clean_valid=batch.clean_valid.index_select(0, indices),
            document_ids=batch.document_ids.index_select(0, indices),
            positions=batch.positions.index_select(0, indices),
            ar_targets=batch.ar_targets.index_select(0, indices),
            bos_targets=batch.bos_targets.index_select(0, indices),
            row_ids=indices,
        )

    full_batch = select(torch.arange(4))
    full = accumulate_duo_validation_stats(
        model,
        (full_batch,),
        canvas_length=8,
        branches=1,
        seed=139,
        total_rows=4,
        compute_ar_diagnostic=True,
    )
    shard_ids = (torch.tensor([0, 2]), torch.tensor([1, 3]))
    shards = [
        accumulate_duo_validation_stats(
            model,
            (select(indices),),
            canvas_length=8,
            branches=1,
            seed=139,
            total_rows=4,
            expected_row_ids=indices,
            compute_ar_diagnostic=True,
        )
        for indices in shard_ids
    ]
    torch.testing.assert_close(sum(shards), full, atol=2e-5, rtol=2e-5)

    reduced = duo_validation_from_stats(sum(shards))
    reference = validate_duo(
        model,
        (full_batch,),
        canvas_length=8,
        branches=1,
        seed=139,
        total_rows=4,
        compute_ar_diagnostic=True,
    )
    assert reduced.targets == reference.targets
    assert reduced.changed_targets == reference.changed_targets
    assert reduced.ar_targets == reference.ar_targets
    assert reduced.ar_anchor_bpb == pytest.approx(reference.ar_anchor_bpb)
    assert reduced.denoising_accuracy == pytest.approx(reference.denoising_accuracy)
    assert reduced.conditional_canvas_nelbo_nats_per_atom == pytest.approx(
        reference.conditional_canvas_nelbo_nats_per_atom
    )


def test_cached_validation_ledger_matches_checked_on_demand_path() -> None:
    model = DuoModel(tiny_config()).eval()
    batch = one_document_batch(3)
    indexed = DuoBatch(
        batch.clean_ids,
        batch.clean_valid,
        batch.document_ids,
        batch.positions,
        batch.ar_targets,
        batch.bos_targets,
        row_ids=torch.arange(3),
    )
    expected = accumulate_duo_validation_stats(
        model,
        (indexed,),
        canvas_length=8,
        branches=1,
        seed=141,
        total_rows=3,
    )
    cached = prepare_duo_validation_batch(
        model,
        indexed,
        total_rows=3,
        canvas_length=8,
        branches=1,
        seed=141,
    )
    observed = accumulate_duo_validation_stats(
        model,
        (cached,),
        canvas_length=8,
        branches=1,
        seed=141,
        total_rows=3,
    )
    torch.testing.assert_close(observed, expected)


def test_validation_shard_authentication_rejects_missing_or_foreign_rows() -> None:
    model = DuoModel(tiny_config()).eval()
    batch = one_document_batch(2)
    indexed = DuoBatch(
        batch.clean_ids,
        batch.clean_valid,
        batch.document_ids,
        batch.positions,
        batch.ar_targets,
        batch.bos_targets,
        row_ids=torch.tensor([0, 1]),
    )
    with pytest.raises(ValueError, match="unassigned row"):
        accumulate_duo_validation_stats(
            model,
            (indexed,),
            canvas_length=8,
            branches=1,
            seed=149,
            total_rows=3,
            expected_row_ids=torch.tensor([0, 2]),
        )
    with pytest.raises(ValueError, match="cover the assigned ledger shard"):
        accumulate_duo_validation_stats(
            model,
            (indexed,),
            canvas_length=8,
            branches=1,
            seed=149,
            total_rows=3,
            expected_row_ids=torch.tensor([0, 1, 2]),
        )
    duplicated = DuoBatch(
        indexed.clean_ids,
        indexed.clean_valid,
        indexed.document_ids,
        indexed.positions,
        indexed.ar_targets,
        indexed.bos_targets,
        row_ids=torch.tensor([0, 0]),
    )
    with pytest.raises(ValueError, match="row ids must be unique"):
        accumulate_duo_validation_stats(
            model,
            (duplicated,),
            canvas_length=8,
            branches=1,
            seed=149,
            total_rows=2,
            expected_row_ids=torch.tensor([0]),
        )


def test_empty_validation_shard_returns_zero_sufficient_statistics() -> None:
    model = DuoModel(tiny_config()).eval()
    stats = accumulate_duo_validation_stats(
        model,
        (),
        canvas_length=8,
        branches=1,
        seed=151,
        total_rows=2,
        expected_row_ids=torch.empty(0, dtype=torch.long),
    )
    torch.testing.assert_close(stats, torch.zeros(6, dtype=torch.float64))


def test_source_provenance_binds_complete_duo_closure() -> None:
    provenance = source_provenance()
    files = provenance["files"]
    for expected in (
        "scripts/train_byte_duo.py",
        "pretraining/byte_diffusion/duo.py",
        "pretraining/byte_diffusion/duo_kernels.py",
        "pretraining/byte_diffusion/duo_model.py",
        "pretraining/byte_diffusion/training_duo.py",
    ):
        assert expected in files
    assert len(provenance["sha256"]) == 64


@pytest.mark.parametrize(
    ("canvas_length", "branches", "expected_canvases"),
    ((512, 8, (1, 2, 2, 2)), (256, 15, (2, 3, 3, 3))),
)
def test_2k_readiness_gate_authenticates_selected_batches_and_all_phases(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    canvas_length: int,
    branches: int,
    expected_canvases: tuple[int, ...],
) -> None:
    from scripts.benchmark_byte_diffusion_architectures import (
        benchmark_harness_provenance,
    )
    from scripts.benchmark_byte_duo_inference import benchmark_source_provenance
    from pretraining.byte_diffusion.config import ByteDiffusionConfig
    from pretraining.byte_diffusion.readiness import SUSTAINED_GPU_POLICY
    from pretraining.byte_diffusion.readiness import (
        diagnostic_cadence_contract,
        duo_geometry_contract,
    )
    from pretraining.byte_diffusion.telemetry import nvidia_smi_selector

    source = "a" * 64
    data = "b" * 64
    total_memory = 32 * (1 << 30)
    required_headroom = 5_153_960_756
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "test-gpu")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=total_memory),
    )
    telemetry = {
        "power_w": {"count": 3, "mean": 510.0, "p10": 470.0, "peak": 520.0},
        "gpu_utilization_percent": {
            "count": 3,
            "mean": 97.0,
            "p10": 91.0,
            "peak": 100.0,
        },
    }
    training_path = tmp_path / "training.json"
    inference_path = tmp_path / "inference.json"
    training = {
        "schema": "byte_diffusion_architecture_readiness/v1",
        "architecture": "duo",
        "source": {"sha256": source},
        "benchmark_harness_source": benchmark_harness_provenance(),
        "dataset_payload_sha256": data,
        "model_config": ByteDiffusionConfig().to_dict(),
        "workload": {
            **duo_geometry_contract(canvas_length, branches),
            "diagnostic_cadence": diagnostic_cadence_contract(
                log_every=10, validation_every=20
            ),
            **duo_objective_contract(PURE_DUO_OBJECTIVE),
                "schedule_eps": 0.001,
                "time_sampling": "global_branch_antithetic_striped_uniform_0_1",
                **duo_mutable_topology_contract(ByteDiffusionConfig()),
        },
        "row_length_bytes": 8_192,
        "global_batch": 249,
        "candidate_microbatches": [12, 15, 16, 24, 32],
        "selected_microbatch": 16,
        "selected_validation_batch_size": 16,
        "results": {
            "16": {
                "status": "ok",
                "eligible": True,
                "microbatch": 16,
                "global_batch": 249,
                "microsteps_per_update": 16,
                "tail_microbatch": 9,
                "tail_exercised": True,
                "row_length_bytes": 8_192,
                "warmup_updates": 2,
                "measured_updates": 4,
                "update_ms": 1.0,
                "cuda_total_bytes": total_memory,
                "required_headroom_bytes": required_headroom,
                "cuda_reserved_headroom_bytes": total_memory,
                "cuda_peak_reserved_bytes": 0,
                "warmup_graph_breaks": 0,
                "measured_unique_graphs": 0,
                "measured_recompiles": 0,
                "measured_graph_breaks": 0,
                **telemetry,
                "validation": {
                    "eligible": True,
                    "rows": 256,
                    "batch_size": 16,
                    "elapsed_seconds": 1.0,
                    "cuda_reserved_headroom_bytes": total_memory,
                    "cuda_peak_reserved_bytes": 0,
                    "warmup_graph_breaks": 0,
                    "measured_unique_graphs": 0,
                    "measured_recompiles": 0,
                    "measured_graph_breaks": 0,
                    **telemetry,
                },
            }
        },
        "sustained_gpu_policy": SUSTAINED_GPU_POLICY,
        "headroom_policy": {
            "fraction": 0.15,
            "minimum_gib": 4.0,
            "required_bytes": required_headroom,
        },
        "gpu": {
            "name": "test-gpu",
            "index": 0,
            "total_memory_bytes": total_memory,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nvidia_smi_selector": nvidia_smi_selector(torch.device("cuda", 0)),
        },
        "runtime": {
            "compiled": True,
            "device_type": "cuda",
            "validation_rows": 256,
            "world_size": 1,
        },
    }
    for microbatch in (12, 15, 24, 32):
        candidate = json.loads(json.dumps(training["results"]["16"]))
        candidate["microbatch"] = microbatch
        candidate["microsteps_per_update"] = math.ceil(249 / microbatch)
        candidate["tail_microbatch"] = 249 % microbatch or microbatch
        candidate["tail_exercised"] = 249 % microbatch != 0
        candidate["update_ms"] = 2.0 + microbatch
        candidate["validation"]["batch_size"] = 16
        training["results"][str(microbatch)] = candidate
    inference = {
        "schema": "byte_duo_inference_readiness/v5",
        "recipe_source": {"sha256": source},
        "benchmark_source": benchmark_source_provenance(),
        "model_config": ByteDiffusionConfig().to_dict(),
        "parameter_count": DUO_PARAMETER_COUNT,
        "serving_origin_policy": "floor_to_patch_and_carry_clean_phase",
        "phase_coverage": [0, 1, 2, 3],
        "canvas_length": canvas_length,
        "branches": branches,
        "requested_atoms_per_trajectory": 512,
        **duo_geometry_contract(canvas_length, branches),
        "posterior_backend": "triton",
        "semantic_generation": False,
        "torch_dynamo": {
            "warmup_graph_breaks": 0,
            "measured_unique_graphs": 0,
            "measured_graph_breaks": 0,
            "measured_recompiles": 0,
        },
        "phases": [
            {
                "phase": phase,
                "requested_atoms": 512,
                "canvases": expected_canvases[phase],
            }
            for phase in range(4)
        ],
        "batch_size": 8,
        "base_prompt_length": 2_048,
        "diffusion_steps": 8,
        "repetitions": 12,
        "measured_seconds": 1.0,
        "cuda_reserved_headroom_bytes": total_memory,
        "cuda_peak_reserved_bytes": 0,
        "required_headroom_bytes": required_headroom,
        "sustained_gpu_policy": SUSTAINED_GPU_POLICY,
        "gpu": training["gpu"],
        "runtime": {"compiled": True, "world_size": 1},
        "eligible": True,
        **telemetry,
    }
    training_path.write_text(json.dumps(training))
    inference_path.write_text(json.dumps(inference))
    args = SimpleNamespace(
        steps=2_000,
        readiness_report=training_path,
        inference_readiness_report=inference_path,
        global_batch_size=249,
        batch_size=16,
        validation_batch_size=16,
        canvas_length=canvas_length,
        branches=branches,
        schedule_eps=0.001,
        compile=True,
        cpu=False,
        validation_rows=256,
        log_every=10,
        val_every=20,
        objective=PURE_DUO_OBJECTIVE,
    )
    evidence = validate_readiness_evidence(
        args,
        source_sha256=source,
        dataset_payload_sha256=data,
        parameter_count=DUO_PARAMETER_COUNT,
    )
    assert evidence is not None
    assert evidence["training_validation"]["selected_microbatch"] == 16

    # MB15 is a first-class authenticated geometry: 16 full slices plus a
    # nine-row tail. It may be selected when it is the fastest fitting arm.
    training["results"]["15"]["update_ms"] = 0.5
    training["selected_microbatch"] = 15
    training_path.write_text(json.dumps(training))
    args.batch_size = 15
    evidence = validate_readiness_evidence(
        args,
        source_sha256=source,
        dataset_payload_sha256=data,
        parameter_count=DUO_PARAMETER_COUNT,
    )
    assert evidence is not None
    assert evidence["training_validation"]["selected_microbatch"] == 15

    training["results"]["15"]["update_ms"] = 17.0
    training["selected_microbatch"] = 16
    args.batch_size = 16
    missing_fifteen = training["results"].pop("15")
    training_path.write_text(json.dumps(training))
    with pytest.raises(ValueError, match="telemetry, provenance, or geometry"):
        validate_readiness_evidence(
            args,
            source_sha256=source,
            dataset_payload_sha256=data,
            parameter_count=DUO_PARAMETER_COUNT,
        )
    training["results"]["15"] = missing_fifteen
    training["results"]["15"]["tail_microbatch"] = 8
    training_path.write_text(json.dumps(training))
    with pytest.raises(ValueError, match="telemetry, provenance, or geometry"):
        validate_readiness_evidence(
            args,
            source_sha256=source,
            dataset_payload_sha256=data,
            parameter_count=DUO_PARAMETER_COUNT,
        )
    training["results"]["15"]["tail_microbatch"] = 9
    training_path.write_text(json.dumps(training))

    args.objective = JOINT_DUO_CLEAN_AR_OBJECTIVE
    with pytest.raises(ValueError, match="telemetry, provenance, or geometry"):
        validate_readiness_evidence(
            args,
            source_sha256=source,
            dataset_payload_sha256=data,
            parameter_count=DUO_PARAMETER_COUNT,
        )
    args.objective = PURE_DUO_OBJECTIVE

    args.batch_size = 8
    with pytest.raises(ValueError, match="telemetry, provenance, or geometry"):
        validate_readiness_evidence(
            args,
            source_sha256=source,
            dataset_payload_sha256=data,
            parameter_count=DUO_PARAMETER_COUNT,
        )

    args.batch_size = 16
    args.log_every = 5
    with pytest.raises(ValueError, match="telemetry, provenance, or geometry"):
        validate_readiness_evidence(
            args,
            source_sha256=source,
            dataset_payload_sha256=data,
            parameter_count=DUO_PARAMETER_COUNT,
        )

    args.log_every = 10
    training["results"]["16"]["cuda_peak_reserved_bytes"] = 0
    training_path.write_text(json.dumps(training))
    inference["cuda_peak_reserved_bytes"] = total_memory - required_headroom + 1
    inference_path.write_text(json.dumps(inference))
    with pytest.raises(ValueError, match="inference readiness"):
        validate_readiness_evidence(
            args,
            source_sha256=source,
            dataset_payload_sha256=data,
            parameter_count=DUO_PARAMETER_COUNT,
        )

    inference["cuda_peak_reserved_bytes"] = 0
    inference_path.write_text(json.dumps(inference))
    faster = json.loads(json.dumps(training["results"]["16"]))
    faster["microbatch"] = 8
    faster["update_ms"] = 0.5
    training["candidate_microbatches"] = [8, 12, 15, 16, 24, 32]
    training["results"]["8"] = faster
    training_path.write_text(json.dumps(training))
    with pytest.raises(ValueError, match="telemetry, provenance, or geometry"):
        validate_readiness_evidence(
            args,
            source_sha256=source,
            dataset_payload_sha256=data,
            parameter_count=DUO_PARAMETER_COUNT,
        )

    training["candidate_microbatches"] = [12, 15, 16, 24, 32]
    training["results"].pop("8")
    training["results"]["16"]["cuda_peak_reserved_bytes"] = (
        total_memory - required_headroom + 1
    )
    training_path.write_text(json.dumps(training))
    with pytest.raises(ValueError, match="telemetry, provenance, or geometry"):
        validate_readiness_evidence(
            args,
            source_sha256=source,
            dataset_payload_sha256=data,
            parameter_count=DUO_PARAMETER_COUNT,
        )


def test_rng_resume_restores_every_stochastic_stream_exactly() -> None:
    device = torch.device("cpu")
    random.seed(127)
    np.random.seed(127)
    torch.manual_seed(127)
    generator = torch.Generator().manual_seed(131)
    state = capture_rng_state(generator, device=device)
    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
        torch.rand(4, generator=generator),
    )
    restore_rng_state(state, generator, device=device)
    observed = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
        torch.rand(4, generator=generator),
    )
    assert observed[0] == expected[0]
    assert observed[1] == expected[1]
    torch.testing.assert_close(observed[2], expected[2])
    torch.testing.assert_close(observed[3], expected[3])
