from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch

from pretraining.byte_diffusion.attention import (
    CanvasBranchLayout,
    branch_attention,
    build_canvas_block_mask,
)
from pretraining.byte_diffusion.config import ByteDiffusionConfig, model_config_from_env
from pretraining.byte_diffusion.duo_model import DuoModel
from pretraining.byte_diffusion.inference_duo import (
    duo_entropy_clean_metadata,
    generate_duo_continuation,
)
from pretraining.byte_diffusion.patching import (
    CausalEntropyPatcher,
    EntropyPatchConfig,
    HashedNgramEntropyConfig,
    HashedNgramEntropyModel,
)
from pretraining.byte_diffusion.training_duo import (
    DuoBatch,
    duo_loss,
    duo_mutable_topology_contract,
    prepare_duo_inputs,
    prepare_duo_validation_inputs,
)
from pretraining.byte_diffusion.training import model_config_from_dict
from pretraining.byte_diffusion.variable_patching import (
    build_duo_clean_patch_metadata,
)


def full_resolution_config(**overrides: object) -> ByteDiffusionConfig:
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
        duo_mutable_topology="full_resolution_decoder",
        duo_random_phase_training=True,
        duo_noisy_ngrams=False,
        **overrides,
    )


def one_document_batch(rows: int = 2) -> DuoBatch:
    generator = torch.Generator().manual_seed(3101)
    ids = torch.randint(0, 256, (rows, 16), generator=generator)
    valid = torch.ones_like(ids, dtype=torch.bool)
    documents = torch.arange(rows)[:, None].expand_as(ids)
    positions = torch.arange(16)[None].expand_as(ids)
    ar_targets = torch.full_like(ids, -100)
    ar_targets[:, :-1] = ids[:, 1:]
    return DuoBatch(
        ids,
        valid,
        documents,
        positions,
        ar_targets,
        ids[:, 0].clone(),
        bos_row_indices=torch.arange(rows),
    )


def test_full_resolution_config_and_checkpoint_contract_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "BYTE_DUO_MUTABLE_TOPOLOGY", "full_resolution_decoder"
    )
    monkeypatch.setenv("BYTE_DUO_RANDOM_PHASE_TRAINING", "1")
    config = model_config_from_env(tiny=True)

    assert config.duo_mutable_topology == "full_resolution_decoder"
    assert config.duo_origin_stride == 1
    assert duo_mutable_topology_contract(config) == {
        "mutable_topology": "full_resolution_decoder",
        "mutable_origin_stride": 1,
        "serving_origin_policy": "exact_prompt_length",
        "mutable_noisy_ngrams": False,
        "clean_patching": "fixed_stride_v1",
        "clean_byte_axis": "compile_stable_physical",
        "clean_patch_axis": "compile_stable_fixed_stride",
        "compile_dynamic_shapes": False,
        "decoder_conditioning_control": "gated_projection",
    }
    assert config.to_dict()["duo_mutable_topology"] == "full_resolution_decoder"
    with pytest.raises(ValueError, match="requires random-phase training"):
        replace(config, duo_random_phase_training=False)
    with pytest.raises(ValueError, match="forbids noisy n-grams"):
        replace(config, duo_noisy_ngrams=True)
    assert config.duo_mutable_ngrams_enabled is False
    legacy = config.to_dict()
    legacy.pop("duo_mutable_topology")
    legacy_config = model_config_from_dict(legacy)
    assert legacy_config.duo_mutable_topology == "patched_global"
    legacy_model = DuoModel(legacy_config)
    legacy_model.load_state_dict(DuoModel(legacy_config).state_dict(), strict=True)
    with pytest.raises(ValueError, match="unsupported duo_mutable_topology"):
        replace(config, duo_mutable_topology="overlap_stride")  # type: ignore[arg-type]


def test_full_resolution_time_conditioner_has_no_dead_encoder_or_global_rows() -> None:
    model = DuoModel(full_resolution_config())

    assert model._adaln_widths == (
        6 * model.config.local_dim,
        2 * model.config.local_dim,
    )
    time = model._time_condition(
        torch.zeros((2, 3, 5), dtype=torch.long),
        torch.full((2, 3), 0.5),
    )
    encoder, global_blocks, decoder, final = model._adaln_modulations(time)
    assert encoder == ()
    assert global_blocks == ()
    assert len(decoder) == model.config.decoder_layers
    assert final.shape == (2, 3, 2 * model.config.local_dim)


def test_entropy_clean_hierarchy_changes_only_patch_partition() -> None:
    config = full_resolution_config(duo_clean_patching="causal_entropy_v1")
    model = DuoModel(config).eval()
    ids = torch.tensor([[65, 66, 256, 70, 71, 256, 262, 262]])
    valid = torch.tensor([[True, True, True, True, True, True, False, False]])
    documents = torch.tensor([[0, 0, 0, 1, 1, 1, -1, -1]])
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 0]])
    patch_offsets = torch.tensor([[0, 1, 0, 0, 1, 2, -1, -1]])
    metadata = build_duo_clean_patch_metadata(
        valid, documents, patch_offsets, max_patch_size=3
    )
    starts = torch.tensor([[1, 4]])
    branch_valid = torch.ones((1, 2, 2), dtype=torch.bool)
    noisy = torch.tensor([[[66, 256], [71, 256]]])

    with torch.no_grad():
        output = model(
            ids,
            valid,
            documents,
            positions,
            noisy,
            branch_valid,
            starts,
            torch.full((1, 2), 0.5),
            clean_patch_metadata=metadata,
        )

    assert output.branch_logits.shape == (1, 2, 2, config.vocab.output_size)
    assert output.clean_patch_states.shape == (3, config.global_dim)
    assert torch.isfinite(output.branch_logits).all()
    # The entropy policy changes no trainable tensor or parameter count.
    fixed = DuoModel(replace(config, duo_clean_patching="fixed_stride_v1"))
    assert model.parameter_count == fixed.parameter_count
    assert set(model.state_dict()) == set(fixed.state_dict())


def test_entropy_serving_probe_keeps_open_tail_out_of_patch_condition() -> None:
    patcher = CausalEntropyPatcher(
        HashedNgramEntropyModel(
            HashedNgramEntropyConfig(table_size=8, vocab_size=261)
        ),
        EntropyPatchConfig(threshold=100.0, max_patch_size=3),
    )
    ids = torch.tensor([[65, 66, 67, 68, 262, 262]])
    valid = torch.tensor([[True, True, True, True, False, False]])
    documents = torch.zeros_like(ids)

    metadata = duo_entropy_clean_metadata(patcher, ids, valid, documents)

    # Byte three is the first byte of an open patch. It and the mutable origin
    # at byte four both see only closed patch zero, never open patch one.
    assert metadata.patch_ordinals.tolist() == [0, 1]
    assert metadata.byte_condition_indices.tolist() == [-1, -1, 0, 0]
    assert metadata.origin_condition_indices[0, 4].item() == 0

    model = DuoModel(
        full_resolution_config(duo_clean_patching="causal_entropy_v1")
    ).eval()
    positions = torch.arange(ids.shape[1])[None]
    with torch.no_grad():
        clean = model.prepare_clean_bank(
            ids,
            valid,
            documents,
            positions,
            clean_patch_metadata=metadata,
        )
    # The open byte at column three remains in every decoder K/V bank. Only
    # its incomplete pooled patch latent is excluded from conditioning.
    assert all(bank.key.shape[2] == ids.shape[1] for bank in clean.decoder)


def test_entropy_model_fails_closed_without_matching_clean_metadata() -> None:
    model = DuoModel(
        full_resolution_config(duo_clean_patching="causal_entropy_v1")
    ).eval()
    ids = torch.tensor([[65, 66, 67, 262]])
    valid = torch.tensor([[True, True, True, False]])
    documents = torch.zeros_like(ids)
    positions = torch.arange(ids.shape[1])[None]

    with torch.no_grad(), pytest.raises(ValueError, match="config and supplied"):
        model.prepare_clean_bank(ids, valid, documents, positions)


def test_readiness_workloads_bind_exact_origin_full_resolution_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.benchmark_byte_diffusion_architectures import (
        readiness_workload_contract,
    )
    from scripts.benchmark_byte_duo_distributed import (
        distributed_readiness_workload_contract,
    )

    monkeypatch.setenv(
        "BYTE_DUO_MUTABLE_TOPOLOGY", "full_resolution_decoder"
    )
    monkeypatch.setenv("BYTE_DUO_RANDOM_PHASE_TRAINING", "1")
    monkeypatch.delenv("BYTE_DUO_NOISY_NGRAMS", raising=False)
    model_contract, local = readiness_workload_contract("duo")
    distributed = distributed_readiness_workload_contract()

    assert model_contract["duo_noisy_ngrams"] is False
    for workload in (local, distributed):
        assert workload["mutable_topology"] == "full_resolution_decoder"
        assert workload["mutable_origin_stride"] == 1
        assert workload["serving_origin_policy"] == "exact_prompt_length"
        assert workload["mutable_noisy_ngrams"] is False
        assert workload["decoder_conditioning_control"] == "gated_projection"

    from scripts.benchmark_byte_duo_inference import _serving_canvas_origin

    assert _serving_canvas_origin(2051, 4, full_resolution=True) == (2051, 0)
    assert _serving_canvas_origin(2051, 4, full_resolution=False) == (2048, 3)


def test_random_phase_training_samples_arbitrary_byte_origins_without_fixed_prefix() -> None:
    model = DuoModel(full_resolution_config()).eval()
    prepared = prepare_duo_inputs(
        model,
        one_document_batch(32),
        canvas_length=5,
        branches=3,
        generator=torch.Generator().manual_seed(3102),
    )

    assert bool(prepared.selection.starts.remainder(4).ne(0).any())
    assert prepared.corruption.fixed_clean is None or not bool(
        prepared.corruption.fixed_clean.any()
    )
    assert prepared.global_block_mask_metadata is None
    assert prepared.local_block_mask_metadata is not None


def test_full_resolution_loss_accepts_authenticated_short_canvas_prefixes() -> None:
    model = DuoModel(
        full_resolution_config(duo_variable_length_probability=1.0)
    ).eval()
    batch = one_document_batch(8)
    prepared = prepare_duo_inputs(
        model,
        batch,
        canvas_length=5,
        branches=2,
        generator=torch.Generator().manual_seed(3107),
    )

    assert bool(prepared.selection.valid.sum(-1).lt(5).any())
    loss = duo_loss(
        model,
        batch,
        canvas_length=5,
        branches=2,
        prepared=prepared,
    )
    assert bool(torch.isfinite(loss.total))


def test_canonical_validation_is_comparable_and_serving_ledger_uses_exact_origins() -> None:
    model = DuoModel(full_resolution_config()).eval()
    batch = one_document_batch(16)
    rows = torch.arange(16)
    canonical = prepare_duo_validation_inputs(
        model,
        batch,
        row_ids=rows,
        total_rows=16,
        canvas_length=5,
        branches=3,
        seed=3106,
    )
    phase = prepare_duo_validation_inputs(
        model,
        batch,
        row_ids=rows,
        total_rows=16,
        canvas_length=5,
        branches=3,
        seed=3106,
        expose_random_phase=True,
    )

    assert not bool(canonical.selection.starts.remainder(4).any())
    assert bool(phase.selection.starts.remainder(4).ne(0).any())
    assert not bool(phase.corruption.fixed_clean.any())


def test_full_resolution_mutable_states_never_enter_encoder_pool_or_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DuoModel(full_resolution_config()).eval()
    batch = one_document_batch()
    prepared = prepare_duo_inputs(
        model,
        batch,
        canvas_length=5,
        branches=2,
        generator=torch.Generator().manual_seed(3103),
    )
    observed: dict[str, list[int]] = {"encoder": [], "pool": [], "global": []}

    encoder_forward = model.encoder[0].forward_packed
    pool_forward = model.pool.forward
    global_forward = model.global_blocks[0].forward_packed

    def record_encoder(states: torch.Tensor, **kwargs):
        observed["encoder"].append(states.shape[0])
        return encoder_forward(states, **kwargs)

    def record_pool(states: torch.Tensor, valid: torch.Tensor):
        observed["pool"].append(states.shape[1])
        return pool_forward(states, valid)

    def record_global(states: torch.Tensor, **kwargs):
        observed["global"].append(states.shape[0])
        return global_forward(states, **kwargs)

    monkeypatch.setattr(model.encoder[0], "forward_packed", record_encoder)
    monkeypatch.setattr(model.pool, "forward", record_pool)
    monkeypatch.setattr(model.global_blocks[0], "forward_packed", record_global)
    monkeypatch.setattr(
        model,
        "_ngram_features",
        lambda *args, **kwargs: pytest.fail("mutable n-grams must be bypassed"),
    )
    monkeypatch.setattr(
        model,
        "_branch_ngram_features",
        lambda *args, **kwargs: pytest.fail("mutable n-grams must be bypassed"),
    )
    noisy = prepared.corruption.ids.view(2, 2, 5)
    model(
        batch.clean_ids,
        batch.clean_valid,
        batch.document_ids,
        batch.positions,
        noisy,
        prepared.selection.valid,
        prepared.selection.starts,
        prepared.corruption.t.view(2, 2),
    )

    # Encoder/global see exactly the packed clean bank, and pooling sees only
    # the clean physical row. Mutable bytes first appear in the decoder.
    assert observed == {"encoder": [32], "pool": [16], "global": [8]}


def test_preceding_patch_selection_resets_at_document_bos() -> None:
    model = DuoModel(full_resolution_config()).eval()
    clean_patch_states = torch.arange(4 * 32, dtype=torch.float32).view(1, 4, 32)
    valid = torch.ones((1, 16), dtype=torch.bool)
    documents = torch.tensor([[0] * 8 + [1] * 8])
    positions = torch.tensor([list(range(8)) + list(range(8))])
    starts = torch.tensor([[1, 5, 8, 13]])

    selected = model._preceding_patch_condition(
        clean_patch_states, valid, documents, positions, starts
    )

    torch.testing.assert_close(selected[0, 0], torch.zeros(32))
    torch.testing.assert_close(selected[0, 1], clean_patch_states[0, 0])
    torch.testing.assert_close(selected[0, 2], torch.zeros(32))
    torch.testing.assert_close(selected[0, 3], clean_patch_states[0, 2])


def test_full_resolution_forward_and_prepared_cache_match_and_isolate_branches() -> None:
    model = DuoModel(full_resolution_config()).eval()
    assert model.output is not None
    torch.nn.init.normal_(model.output.weight, std=0.02)
    torch.nn.init.normal_(model.output.bias, std=0.02)
    torch.nn.init.normal_(model.time_adaln.weight, std=0.02)
    torch.nn.init.normal_(model.time_adaln.bias, std=0.02)
    batch = one_document_batch()
    starts = torch.tensor([[3, 9], [2, 10]])
    canvas = 5
    offsets = torch.arange(canvas)
    valid = starts[:, :, None] + offsets < 16
    source = torch.gather(
        batch.clean_ids[:, None, :].expand(-1, 2, -1),
        2,
        (starts[:, :, None] + offsets).clamp_max(15),
    )
    noisy = torch.where(valid, source.roll(1, -1), model.config.vocab.pad_id)
    times = torch.tensor([[0.2, 0.8], [0.4, 0.6]])

    direct = model(
        batch.clean_ids,
        batch.clean_valid,
        batch.document_ids,
        batch.positions,
        noisy,
        valid,
        starts,
        times,
    )
    with torch.no_grad():
        clean = model.prepare_clean_bank(
            batch.clean_ids,
            batch.clean_valid,
            batch.document_ids,
            batch.positions,
        )
        cache = model.prepare_canvas_cache(clean, valid, starts)
        prepared = model.forward_prepared(cache, noisy, times)
        changed = noisy.clone()
        changed[:, 1] = torch.where(valid[:, 1], 7, model.config.vocab.pad_id)
        isolated = model.forward_prepared(cache, changed, times)

    assert cache.local_layout is None and cache.global_layout is None
    assert cache.local_block_mask is None and cache.global_block_mask is None
    assert clean.encoder == () and clean.global_blocks == ()
    assert clean.decoder_condition is None
    torch.testing.assert_close(
        direct.branch_logits, prepared.branch_logits, atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        prepared.branch_logits[:, 0], isolated.branch_logits[:, 0], atol=0, rtol=0
    )
    assert not torch.allclose(
        prepared.branch_logits[:, 1], isolated.branch_logits[:, 1]
    )


def test_branch_constant_condition_is_projected_once_and_cached_across_nfes() -> None:
    model = DuoModel(full_resolution_config()).eval()
    batch = one_document_batch()
    starts = torch.tensor([[3, 9], [2, 10]])
    canvas = 5
    offsets = torch.arange(canvas)
    valid = starts[:, :, None] + offsets < batch.clean_ids.shape[1]
    noisy = torch.where(
        valid,
        torch.zeros_like(valid, dtype=torch.long),
        model.config.vocab.pad_id,
    )
    times = torch.full((2, 2), 0.5)
    observed: list[tuple[int, ...]] = []
    handle = model.decoder[0].condition.register_forward_pre_hook(
        lambda _module, args: observed.append(tuple(args[0].shape))
    )

    with torch.no_grad():
        clean = model.prepare_clean_bank(
            batch.clean_ids,
            batch.clean_valid,
            batch.document_ids,
            batch.positions,
        )
        cache = model.prepare_canvas_cache(clean, valid, starts)
        calls_after_cache = tuple(observed)
        model.forward_prepared(cache, noisy, times)
        model.forward_prepared(cache, noisy, times)
    handle.remove()

    assert calls_after_cache == ((2, 4, 32), (2, 2, 32))
    assert tuple(observed) == calls_after_cache


def test_export_smoke_uses_a_non_aligned_exact_origin() -> None:
    from scripts.export_byte_duo import cached_inference_smoke

    report = cached_inference_smoke(DuoModel(full_resolution_config()).eval())

    assert report["branch_start"] == report["clean_atoms"] == 9
    assert report["origin_phase"] == 1
    assert report["origin_stride"] == 1
    assert report["origin_policy"] == "exact_non_aligned_prompt"


def test_full_resolution_branch_cannot_read_future_or_another_document() -> None:
    model = DuoModel(full_resolution_config()).eval()
    assert model.output is not None
    torch.nn.init.normal_(model.output.weight, std=0.02)
    torch.nn.init.normal_(model.time_adaln.weight, std=0.02)
    torch.nn.init.normal_(model.time_adaln.bias, std=0.02)
    ids = torch.arange(16)[None]
    valid = torch.ones_like(ids, dtype=torch.bool)
    documents = torch.tensor([[0] * 8 + [1] * 8])
    positions = torch.tensor([list(range(8)) + list(range(8))])
    starts = torch.tensor([[5, 9]])
    branch_valid = torch.ones((1, 2, 3), dtype=torch.bool)
    noisy = torch.tensor([[[31, 32, 33], [41, 42, 43]]])
    times = torch.tensor([[0.3, 0.7]])

    baseline = model(
        ids,
        valid,
        documents,
        positions,
        noisy,
        branch_valid,
        starts,
        times,
    ).branch_logits
    future_changed = ids.clone()
    future_changed[:, 5:8] += 100
    changed_future = model(
        future_changed,
        valid,
        documents,
        positions,
        noisy,
        branch_valid,
        starts,
        times,
    ).branch_logits
    previous_document_changed = ids.clone()
    previous_document_changed[:, :8] += 100
    changed_document = model(
        previous_document_changed,
        valid,
        documents,
        positions,
        noisy,
        branch_valid,
        starts,
        times,
    ).branch_logits

    # Origin 5 cannot read bytes 5..7. Origin 9 belongs to document 1 and
    # cannot read any byte from document 0.
    torch.testing.assert_close(baseline[:, 0], changed_future[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(
        baseline[:, 1], changed_document[:, 1], atol=0, rtol=0
    )


def test_full_resolution_generation_starts_exactly_and_prepares_cache_under_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DuoModel(full_resolution_config()).eval()
    autocast_active = False
    preparation_states: list[tuple[str, bool]] = []
    original_clean = model.prepare_clean_bank
    original_canvas = model.prepare_canvas_cache

    @contextmanager
    def tracked_autocast(**_kwargs):
        nonlocal autocast_active
        previous = autocast_active
        autocast_active = True
        try:
            yield
        finally:
            autocast_active = previous

    def prepare_clean(*args, **kwargs):
        preparation_states.append(("clean", autocast_active))
        return original_clean(*args, **kwargs)

    def prepare_canvas(*args, **kwargs):
        preparation_states.append(("canvas", autocast_active))
        return original_canvas(*args, **kwargs)

    monkeypatch.setattr(torch, "autocast", tracked_autocast)
    monkeypatch.setattr(model, "prepare_clean_bank", prepare_clean)
    monkeypatch.setattr(model, "prepare_canvas_cache", prepare_canvas)
    prompt = torch.tensor([[65, 66, 67]])
    result = generate_duo_continuation(
        model,
        prompt,
        torch.ones_like(prompt, dtype=torch.bool),
        canvas_length=5,
        max_new_atoms=2,
        steps=1,
        generator=torch.Generator().manual_seed(3104),
        return_trajectories=True,
    )

    assert result.trajectory_starts is not None
    assert result.trajectory_starts[0].tolist() == [3]
    assert result.ids[0, :3].tolist() == [65, 66, 67]
    assert preparation_states[:2] == [("clean", True), ("canvas", True)]


def test_branch_flex_matches_dense_oracle_for_arbitrary_phase_layout() -> None:
    generator = torch.Generator().manual_seed(3105)
    clean_valid = torch.ones((1, 8), dtype=torch.bool)
    branch_valid = torch.ones((1, 2, 3), dtype=torch.bool)
    starts = torch.tensor([[1, 5]])
    positions = torch.arange(8)[None]
    branch_positions = starts[:, :, None] + torch.arange(3)
    documents = torch.zeros((1, 8), dtype=torch.long)
    layout = CanvasBranchLayout(
        clean_valid,
        branch_valid,
        starts,
        None,
        positions,
        branch_positions,
        documents,
        torch.zeros((1, 2), dtype=torch.long),
    )
    query = torch.randn((1, 1, 6, 8), generator=generator)
    clean_key = torch.randn((1, 1, 8, 8), generator=generator)
    clean_value = torch.randn((1, 1, 8, 8), generator=generator)
    branch_key = torch.randn((1, 1, 6, 8), generator=generator)
    branch_value = torch.randn((1, 1, 6, 8), generator=generator)
    dense = branch_attention(
        query,
        clean_key,
        clean_value,
        branch_key,
        branch_value,
        layout,
        backend="dense_reference",
        allow_dense_reference=True,
    )
    mask = build_canvas_block_mask(layout, block_size=2)
    # The production dispatcher correctly refuses CPU as a deployment
    # backend; invoke PyTorch's same Flex kernel directly for this tiny CPU
    # topology oracle.
    from torch.nn.attention.flex_attention import flex_attention

    flex = flex_attention(
        query,
        torch.cat((clean_key, branch_key), dim=2),
        torch.cat((clean_value, branch_value), dim=2),
        block_mask=mask,
    )

    torch.testing.assert_close(flex, dense, atol=2e-5, rtol=2e-5)


def test_bfloat16_time_features_stay_finite_and_shape_stable() -> None:
    model = DuoModel(full_resolution_config()).to(torch.bfloat16)
    noisy = torch.zeros((2, 3, 5), dtype=torch.long)
    condition = model._time_condition(
        noisy, torch.tensor([[1e-6, 0.2, 1.0], [0.1, 0.5, 0.9]])
    )

    assert condition.dtype == torch.bfloat16
    assert condition.shape == (2, 3, model.config.duo_time_condition_dim)
    assert bool(torch.isfinite(condition).all())
