from __future__ import annotations

import pytest
import torch

from pretraining.byte_diffusion import ByteDiffusionConfig, ByteDiffusionModel, ModelMode
from pretraining.byte_diffusion.layers import (
    RMSNorm,
    pack_rows,
    packed_sequence_offsets,
    prefix_row_indices,
    unpack_rows,
    unpack_valid,
)


def test_rms_norm_preserves_low_precision_activation_dtype() -> None:
    norm = RMSNorm(8)
    values = torch.randn(2, 4, 8, dtype=torch.bfloat16)

    assert norm(values).dtype == torch.bfloat16


def test_unpack_valid_preserves_packed_dtype_under_mixed_precision() -> None:
    packed = torch.randn(3, 2, dtype=torch.bfloat16)
    valid = torch.tensor([[True, False], [True, True]])
    template = torch.empty(2, 2, 2, dtype=torch.float32)

    unpacked = unpack_valid(packed, valid, template)

    assert unpacked.dtype == torch.bfloat16
    torch.testing.assert_close(unpacked[valid], packed)
    torch.testing.assert_close(
        unpacked[~valid], torch.zeros(1, 2, dtype=torch.bfloat16)
    )


def test_packed_sequence_offsets_are_exact_without_cumsum() -> None:
    observed = packed_sequence_offsets(torch.tensor([3, 5, 2]))
    torch.testing.assert_close(
        observed, torch.tensor([0, 3, 8, 10], dtype=torch.int32)
    )


def test_reusable_prefix_indices_pack_and_unpack_all_trailing_widths() -> None:
    valid = torch.tensor(
        [[True, True, False, False], [True, True, True, True]]
    )
    indices = prefix_row_indices(valid)
    torch.testing.assert_close(indices, torch.tensor([0, 1, 4, 5, 6, 7]))

    scalar = torch.arange(8).view(2, 4)
    hidden = torch.arange(24).view(2, 4, 3)
    torch.testing.assert_close(pack_rows(scalar, indices), scalar[valid])
    torch.testing.assert_close(pack_rows(hidden, indices), hidden[valid])

    restored = unpack_rows(
        pack_rows(hidden, indices), indices, torch.empty_like(hidden)
    )
    torch.testing.assert_close(restored[valid], hidden[valid])
    torch.testing.assert_close(restored[~valid], torch.zeros(2, 3, dtype=hidden.dtype))


def test_packed_causal_logits_match_padded_valid_rows_exactly() -> None:
    torch.manual_seed(13)
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).eval()
    ids = torch.tensor(
        [
            [65, 66, 67, 256, 262, 262, 262, 262],
            [70, 71, 72, 73, 74, 75, 76, 256],
        ]
    )
    valid = ids.ne(model.config.vocab.pad_id)
    padded = model.forward_ar_varlen(
        ids, valid, allow_dense_reference=True
    ).logits
    packed = model.forward_ar_varlen(
        ids,
        valid,
        allow_dense_reference=True,
        return_padded_logits=False,
    ).logits

    assert packed.shape == (int(valid.sum()), model.config.vocab.output_size)
    torch.testing.assert_close(packed, padded[valid], rtol=0, atol=0)


def test_production_parameter_count_and_atomic_shapes() -> None:
    model = ByteDiffusionModel()
    assert model.parameter_count() == 23_011_074
    assert model.embedding.weight.shape == (263, 256)
    assert model.output is not None
    assert model.output.weight.shape == (261, 256)
    assert torch.count_nonzero(model.embedding.weight[262]) == 0


def test_tiny_all_modes_forward_backward_and_fresh_decoder_embedding() -> None:
    torch.manual_seed(7)
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    ids = torch.tensor([[256, 65, 195, 169, 256, 262, 262, 262]])
    valid = ids.ne(262)
    for mode in ModelMode:
        model.zero_grad(set_to_none=True)
        output = model(ids, valid, mode=mode)
        assert output.logits.shape == (1, 8, 261)
        assert output.patch_states.shape == (1, 2, 64)
        output.logits[valid].float().square().mean().backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )


def test_blt_d_encoder_does_not_read_noisy_ids() -> None:
    torch.manual_seed(11)
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    clean = torch.tensor([[65, 66, 67, 68, 69, 256, 262, 262]])
    noisy_a = clean.clone()
    noisy_b = clean.clone()
    noisy_a[:, 4:6] = 261
    noisy_b[:, 0:2] = 261
    valid = clean.ne(262)
    conditions = torch.zeros(1, 1, model.config.global_dim)
    output_a = model.forward_blt_d(
        clean, noisy_a, valid, block_conditions=conditions
    )
    output_b = model.forward_blt_d(
        clean, noisy_b, valid, block_conditions=conditions
    )
    torch.testing.assert_close(output_a.patch_states, output_b.patch_states, rtol=0, atol=0)


def test_blt_d_logits_cannot_read_current_clean_block() -> None:
    torch.manual_seed(13)
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    clean_a = torch.tensor([[65, 66, 67, 68, 69, 70, 71, 256]])
    clean_b = clean_a.clone()
    clean_b[:, 4:] = torch.tensor([90, 91, 92, 256])
    noisy = clean_a.clone()
    noisy[:, 4:] = 261
    valid = torch.ones_like(clean_a, dtype=torch.bool)
    starts = torch.tensor([[4]])
    conditions = torch.zeros(1, 1, model.config.global_dim)
    output_a = model.forward_blt_d(
        clean_a,
        noisy,
        valid,
        block_conditions=conditions,
        block_starts=starts,
        block_length=4,
    )
    output_b = model.forward_blt_d(
        clean_b,
        noisy,
        valid,
        block_conditions=conditions,
        block_starts=starts,
        block_length=4,
    )
    torch.testing.assert_close(output_a.logits[:, 4:], output_b.logits[:, 4:], rtol=0, atol=0)


def test_ar_conditioning_indices_close_patch_at_last_byte() -> None:
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    patches = torch.stack((torch.ones(64), 2 * torch.ones(64)))[None]
    condition = model._aligned_condition(patches, 8, ModelMode.AR)
    expected = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2], dtype=condition.dtype)
    torch.testing.assert_close(condition[0, :, 0], expected)


def test_padding_row_remains_zero_after_update_and_invariant_enforcement() -> None:
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    ids = torch.tensor([[65, 66, 67, 256]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    model(ids, valid).logits.square().mean().backward()
    optimizer.step()
    model.enforce_padding_invariant()
    assert torch.count_nonzero(model.embedding.weight[262]) == 0


def test_optional_tied_timestep_and_self_conditioning_arms_are_live() -> None:
    config = ByteDiffusionConfig.tiny(
        output_tied=True,
        explicit_timestep=True,
        self_conditioning=True,
        self_conditioning_hidden=8,
    )
    model = ByteDiffusionModel(config)
    ids = torch.tensor([[65, 66, 67, 256]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    probabilities = torch.full((1, 4, 261), 1 / 261)
    output = model(
        ids,
        valid,
        mode=ModelMode.CANVAS,
        timestep=torch.tensor([0.5]),
        self_condition_probs=probabilities,
    )
    output.logits.square().mean().backward()
    assert model.output is None
    assert model.timestep_vector is not None and model.timestep_vector.grad is not None
    assert all(parameter.grad is not None for parameter in model.self_conditioner.parameters())


def test_varlen_clean_path_matches_dense_oracle_on_valid_bytes() -> None:
    torch.manual_seed(19)
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    ids = torch.tensor(
        [
            [65, 66, 67, 256, 262, 262, 262, 262],
            [70, 71, 72, 73, 74, 75, 76, 256],
        ]
    )
    valid = ids.ne(262)
    positions = torch.arange(8)[None].expand(2, -1)
    dense = model(ids, valid, mode=ModelMode.AR, positions=positions)
    packed = model.forward_ar_varlen(
        ids, valid, positions=positions, allow_dense_reference=True
    )
    torch.testing.assert_close(
        packed.logits[valid], dense.logits[valid], rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        packed.patch_states[valid.view(2, 2, 4).any(-1)],
        dense.patch_states[valid.view(2, 2, 4).any(-1)],
        rtol=1e-5,
        atol=1e-5,
    )


def test_full_clean_fast_path_matches_generic_packed_path() -> None:
    torch.manual_seed(191)
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny()).eval()
    ids = torch.tensor(
        [
            [65, 66, 67, 68, 69, 70, 71, 256],
            [72, 73, 74, 75, 76, 77, 78, 256],
        ]
    )
    valid = torch.ones_like(ids, dtype=torch.bool)
    positions = torch.arange(8)[None].expand_as(ids)
    generic = model.forward_ar_varlen(
        ids, valid, positions=positions, allow_dense_reference=True
    )
    fixed = model.forward_ar_varlen(
        ids,
        valid,
        positions=positions,
        allow_dense_reference=True,
        assume_full_clean=True,
    )
    torch.testing.assert_close(fixed.logits, generic.logits, rtol=0, atol=0)
    torch.testing.assert_close(
        fixed.patch_states, generic.patch_states, rtol=0, atol=0
    )


def test_ar_logits_are_invariant_to_target_and_future_bytes_with_ngrams() -> None:
    torch.manual_seed(29)
    config = ByteDiffusionConfig.tiny(
        ngram_enabled=True,
        ngram_table_size=128,
        ngram_rank=4,
        ngram_orders=(3, 4),
    )
    model = ByteDiffusionModel(config)
    original = torch.tensor([[65, 66, 67, 68, 69, 70, 71, 256]])
    valid = torch.ones_like(original, dtype=torch.bool)
    baseline = model.forward_ar_varlen(
        original, valid, allow_dense_reference=True
    ).logits

    for target_index in (3, 4):
        changed = original.clone()
        changed[:, target_index:] = torch.tensor(
            [[90 + offset for offset in range(8 - target_index)]]
        )
        changed_logits = model.forward_ar_varlen(
            changed, valid, allow_dense_reference=True
        ).logits
        torch.testing.assert_close(
            baseline[:, target_index - 1],
            changed_logits[:, target_index - 1],
            rtol=0,
            atol=0,
        )


def test_ngram_polynomial_hash_distinguishes_permutations_mod_power_of_two() -> None:
    config = ByteDiffusionConfig.tiny(
        ngram_enabled=True,
        ngram_table_size=128,
        ngram_rank=4,
        ngram_orders=(3,),
    )
    ngrams = ByteDiffusionModel(config).ngrams
    assert ngrams is not None
    ids = torch.tensor([[1, 2, 3], [3, 2, 1]])

    hashes, active = ngrams._hash_ids(ids, 3)

    assert active[:, -1].tolist() == [True, True]
    assert hashes[0, -1] != hashes[1, -1]


def test_blt_prime_hash_matches_appendix_c_orientation_and_resets_context() -> None:
    config = ByteDiffusionConfig.tiny(
        ngram_enabled=True,
        ngram_table_size=32_768,
        ngram_rank=4,
        ngram_orders=(3,),
        ngram_hash="blt_prime",
    )
    ngrams = ByteDiffusionModel(config).ngrams
    assert ngrams is not None
    ids = torch.tensor([[1, 2, 3, config.vocab.eot_id, 4, 5, 6]])

    hashes, active = ngrams._hash_ids(ids, 3)
    prime = 1_000_000_007
    expected = (1 * prime**2 + 2 * prime + 3) % config.ngram_table_size

    assert int(hashes[0, 2]) == expected
    assert active.tolist() == [[False, False, True, True, False, False, True]]


@pytest.mark.parametrize("table_sharing", ["shared", "per_order"])
@pytest.mark.parametrize("hash_kind", ["blt_prime", "legacy257"])
def test_single_pass_ngram_embedding_matches_independent_order_reference(
    table_sharing: str,
    hash_kind: str,
) -> None:
    config = ByteDiffusionConfig.tiny(
        ngram_enabled=True,
        ngram_table_size=128,
        ngram_rank=4,
        ngram_orders=(3, 4, 5),
        ngram_table_sharing=table_sharing,
        ngram_hash=hash_kind,
    )
    ngrams = ByteDiffusionModel(config).ngrams
    assert ngrams is not None
    ids = torch.tensor(
        [[1, 2, 3, 4, config.vocab.eot_id, 5, 6, 7, 8]]
    )
    reference_features = []
    for index, order in enumerate(ngrams.orders):
        hashed, active = ngrams._hash_ids(ids, order)
        table = ngrams.table if ngrams.table is not None else ngrams.tables[index]
        reference_features.append(table(hashed) * active[:, :, None])
    expected = torch.einsum(
        "blor,odr->bld",
        torch.stack(reference_features, dim=2),
        ngrams.projection_weights,
    )

    torch.testing.assert_close(ngrams(ids), expected, rtol=0, atol=0)


def test_scale_matched_ngram_factor_initialization_changes_projection_scale() -> None:
    common = {
        "ngram_enabled": True,
        "ngram_table_size": 128,
        "ngram_rank": 4,
        "ngram_orders": (3,),
    }
    torch.manual_seed(43)
    weak = ByteDiffusionModel(
        ByteDiffusionConfig.tiny(**common, ngram_factor_init="weak")
    )
    torch.manual_seed(43)
    matched = ByteDiffusionModel(
        ByteDiffusionConfig.tiny(**common, ngram_factor_init="scale_matched")
    )
    assert weak.ngrams is not None and matched.ngrams is not None

    weak_std = weak.ngrams.projection_weights[0].std()
    matched_std = matched.ngrams.projection_weights[0].std()

    assert float(weak_std.detach()) == pytest.approx(0.02, abs=0.003)
    assert float(matched_std.detach()) == pytest.approx(0.5, abs=0.07)
    torch.testing.assert_close(
        matched.ngrams.projection_weights[0],
        weak.ngrams.projection_weights[0] * 25,
        rtol=1e-6,
        atol=1e-6,
    )


def test_per_order_ngram_tables_are_independent_and_preserve_output_shape() -> None:
    config = ByteDiffusionConfig.tiny(
        ngram_enabled=True,
        ngram_table_size=128,
        ngram_rank=4,
        ngram_orders=(3, 4, 5),
        ngram_table_sharing="per_order",
    )
    ngrams = ByteDiffusionModel(config).ngrams
    assert (
        ngrams is not None
        and ngrams.table is None
        and ngrams.tables is not None
    )
    assert len(ngrams.tables) == 3
    assert len({id(table.weight) for table in ngrams.tables}) == 3
    features = ngrams(torch.arange(16).view(1, 16))
    assert features.shape == (1, 16, config.local_dim)


def test_decoder_conditioning_starts_at_embedding_scale() -> None:
    torch.manual_seed(37)
    model = ByteDiffusionModel(
        ByteDiffusionConfig.tiny(decoder_conditioning="gated_projection")
    )
    block = model.decoder[0]
    states = torch.randn(8, model.config.local_dim) * 0.02
    condition = torch.randn(8, model.config.global_dim)

    conditioned = block._add_condition(states, condition)
    delta_rms = (conditioned - states).square().mean().sqrt()
    state_rms = states.square().mean().sqrt()

    assert block.condition_gate is not None
    assert block.condition_gate.requires_grad
    assert 0.5 < float((delta_rms / state_rms).detach()) < 2.0

    zero_condition = torch.zeros_like(condition, requires_grad=True)
    torch.testing.assert_close(
        block._add_condition(states, zero_condition), states, rtol=0, atol=0
    )
    conditioned.square().mean().backward()
    assert block.condition_gate.grad is not None
    assert block.condition.weight.grad is not None


def test_split_conditioning_routes_over_two_latent_keys() -> None:
    torch.manual_seed(41)
    model = ByteDiffusionModel(
        ByteDiffusionConfig.tiny(decoder_conditioning="split_cross_attention")
    )
    block = model.decoder[0]
    states = torch.randn(2, 3, model.config.local_dim)
    condition = torch.randn(2, 3, model.config.global_dim)

    conditioned = block._add_condition(states, condition)

    assert block.condition_splits == 2
    assert block.condition_gate is None
    assert conditioned.shape == states.shape
    assert torch.isfinite(conditioned).all()
    assert not torch.equal(conditioned, states)


def test_virtual_bos_matches_document_packed_global_state() -> None:
    torch.manual_seed(31)
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    ids = torch.tensor([[65, 66, 67, 68]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    output = model.forward_ar_varlen(
        ids,
        valid,
        allow_dense_reference=True,
        byte_indices=torch.arange(4),
        byte_cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
        patch_indices=torch.tensor([0]),
        patch_cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
        condition_patch_indices=torch.tensor([0, 0, 0, 1]),
        global_patch_sources=torch.tensor([-1, 0]),
        global_patch_positions=torch.tensor([0, 1]),
        physical_to_global_patch_indices=torch.tensor([1]),
        bos_condition_indices=torch.tensor([0]),
    )
    standalone = model.virtual_bos_global_states(
        1, device=ids.device, allow_dense_reference=True
    )
    assert output.bos_patch_states is not None
    torch.testing.assert_close(
        output.bos_patch_states, standalone, rtol=1e-6, atol=1e-7
    )
    logits = model.forward_bos_logits(
        output.bos_patch_states, allow_dense_reference=True
    )
    assert logits.shape == (1, model.config.vocab.output_size)


def test_canvas_branches_share_prefix_without_cross_branch_or_target_leakage() -> None:
    torch.manual_seed(17)
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    clean = torch.tensor([[65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 256]])
    valid = torch.ones_like(clean, dtype=torch.bool)
    noisy = torch.tensor([[[261, 261, 71, 72], [261, 74, 261, 256]]])
    branch_valid = torch.ones_like(noisy, dtype=torch.bool)
    starts = torch.tensor([[4, 8]])
    baseline = model.forward_canvas_branches(clean, valid, noisy, branch_valid, starts)

    other_branch = noisy.clone()
    other_branch[:, 1] = torch.tensor([90, 91, 92, 256])
    changed_branch = model.forward_canvas_branches(
        clean, valid, other_branch, branch_valid, starts
    )
    torch.testing.assert_close(
        baseline.branch_logits[:, 0], changed_branch.branch_logits[:, 0], rtol=0, atol=0
    )
    torch.testing.assert_close(
        baseline.clean_logits, changed_branch.clean_logits, rtol=0, atol=0
    )

    hidden_target = clean.clone()
    hidden_target[:, 4:8] = torch.tensor([100, 101, 102, 103])
    changed_target = model.forward_canvas_branches(
        hidden_target, valid, noisy, branch_valid, starts
    )
    torch.testing.assert_close(
        baseline.branch_logits[:, 0], changed_target.branch_logits[:, 0], rtol=0, atol=0
    )


def test_joint_canvas_clean_logits_match_standalone_ar_with_ngrams() -> None:
    torch.manual_seed(41)
    config = ByteDiffusionConfig.tiny(
        ngram_enabled=True,
        ngram_table_size=128,
        ngram_rank=4,
        ngram_orders=(3, 4),
    )
    model = ByteDiffusionModel(config).eval()
    clean = torch.tensor([[65, 66, 67, 68, 69, 70, 71, 256]])
    valid = torch.ones_like(clean, dtype=torch.bool)
    noisy = torch.tensor([[[261, 261, 71, 256]]])
    branch_valid = torch.ones_like(noisy, dtype=torch.bool)
    starts = torch.tensor([[4]])

    standalone = model.forward_ar_varlen(
        clean, valid, allow_dense_reference=True
    ).logits
    joint = model.forward_canvas_branches(
        clean, valid, noisy, branch_valid, starts
    ).clean_logits

    torch.testing.assert_close(joint, standalone, rtol=0, atol=0)


def test_blt_sampled_blocks_share_decoder_bank_and_are_isolated() -> None:
    torch.manual_seed(23)
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    clean = torch.tensor([[65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 256]])
    valid = torch.ones_like(clean, dtype=torch.bool)
    starts = torch.tensor([[4, 8]])
    noisy = torch.tensor([[[261, 261, 71, 72], [261, 74, 75, 256]]])
    branch_valid = torch.ones_like(noisy, dtype=torch.bool)
    conditions = starts // model.config.patch_stride - 1
    baseline = model.forward_blt_d_branches(
        clean,
        valid,
        noisy,
        branch_valid,
        starts,
        branch_condition_indices=conditions,
    )
    changed = noisy.clone()
    changed[:, 1] = torch.tensor([90, 91, 92, 256])
    other = model.forward_blt_d_branches(
        clean,
        valid,
        changed,
        branch_valid,
        starts,
        branch_condition_indices=conditions,
    )
    torch.testing.assert_close(
        baseline.branch_logits[:, 0], other.branch_logits[:, 0], rtol=0, atol=0
    )
    torch.testing.assert_close(baseline.clean_logits, other.clean_logits, rtol=0, atol=0)
