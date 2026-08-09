from __future__ import annotations

import torch

from pretraining.byte_diffusion import ByteDiffusionConfig, ByteDiffusionModel, ModelMode


def test_production_parameter_count_and_atomic_shapes() -> None:
    model = ByteDiffusionModel()
    assert model.parameter_count() == 23_011_584
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
    output_a = model.forward_blt_d(clean, noisy_a, valid)
    output_b = model.forward_blt_d(clean, noisy_b, valid)
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
    output_a = model.forward_blt_d(clean_a, noisy, valid, block_starts=starts, block_length=4)
    output_b = model.forward_blt_d(clean_b, noisy, valid, block_starts=starts, block_length=4)
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


def test_virtual_bos_matches_explicit_eot_causal_position() -> None:
    torch.manual_seed(31)
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    eot = torch.tensor([[model.config.vocab.eot_id, model.config.vocab.pad_id,
                         model.config.vocab.pad_id, model.config.vocab.pad_id]])
    valid = torch.tensor([[True, False, False, False]])
    explicit = model.forward_ar_varlen(
        eot, valid, allow_dense_reference=True
    ).logits[:, 0]
    synthetic = model.forward_bos_logits(
        1, device=eot.device, allow_dense_reference=True
    )
    torch.testing.assert_close(synthetic, explicit, rtol=0, atol=0)


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


def test_blt_sampled_blocks_share_decoder_bank_and_are_isolated() -> None:
    torch.manual_seed(23)
    model = ByteDiffusionModel(ByteDiffusionConfig.tiny())
    clean = torch.tensor([[65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 256]])
    valid = torch.ones_like(clean, dtype=torch.bool)
    starts = torch.tensor([[4, 8]])
    noisy = torch.tensor([[[261, 261, 71, 72], [261, 74, 75, 256]]])
    branch_valid = torch.ones_like(noisy, dtype=torch.bool)
    baseline = model.forward_blt_d_branches(clean, valid, noisy, branch_valid, starts)
    changed = noisy.clone()
    changed[:, 1] = torch.tensor([90, 91, 92, 256])
    other = model.forward_blt_d_branches(clean, valid, changed, branch_valid, starts)
    torch.testing.assert_close(
        baseline.branch_logits[:, 0], other.branch_logits[:, 0], rtol=0, atol=0
    )
    torch.testing.assert_close(baseline.clean_logits, other.clean_logits, rtol=0, atol=0)
