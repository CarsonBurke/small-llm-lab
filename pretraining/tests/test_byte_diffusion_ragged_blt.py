from __future__ import annotations

from copy import deepcopy
from types import MethodType

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import pretraining.byte_diffusion.model as model_module
import pretraining.byte_diffusion.attention as attention_module
from pretraining.byte_diffusion import (
    ByteDiffusionConfig,
    ByteDiffusionModel,
    ModelMode,
)
from pretraining.byte_diffusion.attention import (
    RaggedBltLayout,
    build_ragged_blt_block_mask,
    ragged_blt_allowed,
    ragged_blt_attention,
)
from pretraining.byte_diffusion.variable_patching import build_variable_patch_layout
from pretraining.byte_diffusion.training import JointForward


def _mask_layout() -> RaggedBltLayout:
    return RaggedBltLayout(
        clean_valid=torch.ones(2, 6, dtype=torch.bool),
        clean_positions=torch.tensor(
            [[0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 4, 5]], dtype=torch.long
        ),
        clean_segment_ids=torch.tensor(
            [[10, 10, 10, 10, 11, 11], [20, 20, 20, 20, 20, 20]],
            dtype=torch.long,
        ),
        block_valid=torch.tensor(
            [
                [True, True, False, False],
                [True, False, False, False],
                [True, True, True, True],
            ]
        ),
        block_rows=torch.tensor([0, 0, 1]),
        block_starts=torch.tensor([2, 5, 1]),
        row_cu_offsets=torch.tensor([0, 2, 3], dtype=torch.int32),
    )


def test_ragged_blt_mask_matches_hand_document_and_branch_routing() -> None:
    layout = _mask_layout()
    layout.validate_values()
    allowed = ragged_blt_allowed(layout)[0]

    # Block 0 starts at row-0/document-10 position 2: clean columns 0 and 1,
    # then its two in-document block keys and nothing else.
    expected_first = torch.zeros(layout.kv_length, dtype=torch.bool)
    expected_first[[0, 1]] = True
    expected_first[
        layout.clean_bank_length : layout.clean_bank_length + 2
    ] = True
    torch.testing.assert_close(allowed[0], expected_first)
    torch.testing.assert_close(allowed[1], expected_first)
    assert not allowed[2].any() and not allowed[3].any()

    # Block 1 is in the second document on the same physical row.  Position
    # comparisons alone are insufficient: only column 4 is its clean prefix.
    expected_second = torch.zeros_like(expected_first)
    expected_second[4] = True
    second_bank = layout.clean_bank_length + 4
    expected_second[second_bank] = True
    torch.testing.assert_close(allowed[4], expected_second)
    assert not allowed[5].any() and not allowed[6].any() and not allowed[7].any()

    # Block 2 sees row 1 only and remains isolated from both earlier blocks.
    expected_third = torch.zeros_like(expected_first)
    expected_third[6] = True
    third_bank = layout.clean_bank_length + 8
    expected_third[third_bank : third_bank + 4] = True
    torch.testing.assert_close(allowed[8], expected_third)
    torch.testing.assert_close(allowed[11], expected_third)


def test_ragged_blt_flex_token_rule_matches_dense_oracle() -> None:
    from torch.nn.attention.flex_attention import create_mask

    layout = _mask_layout()
    block_mask = build_ragged_blt_block_mask(
        layout, block_size=16, compile_mask=False
    )
    actual = create_mask(
        block_mask.mask_mod,
        B=1,
        H=1,
        Q_LEN=layout.query_length,
        KV_LEN=layout.kv_length,
        device="cpu",
    )[:, 0]
    torch.testing.assert_close(actual, ragged_blt_allowed(layout))


def test_ragged_blt_attention_has_no_future_or_cross_branch_leak_and_clean_grads() -> None:
    torch.manual_seed(7)
    layout = _mask_layout()
    heads, width = 2, 4
    query = torch.randn(1, heads, layout.query_length, width)
    clean_key = torch.randn(
        1, heads, layout.clean_bank_length, width, requires_grad=True
    )
    clean_value = torch.randn(
        1, heads, layout.clean_bank_length, width, requires_grad=True
    )
    branch_key = torch.randn(1, heads, layout.query_length, width)
    branch_value = torch.randn(1, heads, layout.query_length, width)

    def run(clean_v: torch.Tensor, branch_v: torch.Tensor) -> torch.Tensor:
        return ragged_blt_attention(
            query,
            clean_key,
            clean_v,
            branch_key,
            branch_v,
            layout,
            backend="dense_reference",
            allow_dense_reference=True,
        )

    baseline = run(clean_value, branch_value)
    changed_branch = branch_value.detach().clone()
    changed_branch[:, :, 4:] += 100
    torch.testing.assert_close(baseline[:, :, :4], run(clean_value, changed_branch)[:, :, :4])

    changed_future = clean_value.detach().clone()
    # Row-0 clean columns 2 onward are the target/future or another document.
    changed_future[:, :, 2:6] -= 100
    torch.testing.assert_close(baseline[:, :, :4], run(changed_future, branch_value)[:, :, :4])

    baseline[:, :, :4].sum().backward()
    assert clean_key.grad is not None and clean_value.grad is not None
    assert torch.count_nonzero(clean_key.grad[:, :, :2]) > 0
    assert torch.count_nonzero(clean_key.grad[:, :, 2:]) == 0
    assert torch.count_nonzero(clean_value.grad[:, :, :2]) > 0
    assert torch.count_nonzero(clean_value.grad[:, :, 2:]) == 0


def _variable_case_rows(row_count: int):
    valid = torch.tensor(
        [[True, True, True, True, True, True, False, False]]
    ).repeat(row_count, 1)
    documents = torch.tensor([[0, 0, 0, 1, 1, 1, -1, -1]]).repeat(
        row_count, 1
    )
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 0]]).repeat(
        row_count, 1
    )
    patch_offsets = torch.tensor([[0, 1, 0, 0, 1, 2, -1, -1]]).repeat(
        row_count, 1
    )
    metadata = build_variable_patch_layout(
        valid,
        documents,
        positions.where(valid, torch.full_like(positions, -1)),
        patch_offsets,
        max_patch_size=3,
    )
    ids = torch.tensor([[65, 66, 256, 70, 71, 256, 262, 262]]).repeat(
        row_count, 1
    )
    starts = torch.tensor([2, 3]).repeat(row_count)
    rows = torch.arange(row_count).repeat_interleave(2)
    block_valid = torch.tensor(
        [[True, False, False, False], [True, True, True, False]]
    ).repeat(row_count, 1)
    noisy = torch.tensor(
        [[261, 262, 262, 262], [261, 261, 256, 262]]
    ).repeat(row_count, 1)
    selected_origins = metadata.physical_patch_start_columns.ne(0)
    prior = metadata.physical_patch_prior_condition_indices[selected_origins]
    common = dict(
        positions=positions,
        document_ids=documents,
        byte_indices=metadata.byte_indices,
        byte_cu_seqlens=metadata.byte_cu_seqlens,
        patch_cu_seqlens=metadata.patch_cu_seqlens,
        condition_patch_indices=metadata.condition_patch_indices,
        global_patch_sources=metadata.global_patch_sources,
        global_patch_positions=metadata.global_patch_positions,
        physical_to_global_patch_indices=metadata.physical_to_global_patch_indices,
        bos_condition_indices=metadata.bos_condition_indices,
        patch_byte_cu_seqlens=metadata.patch_byte_cu_seqlens,
        max_patch_size=3,
        return_clean_patch_states=False,
    )
    return ids, valid, starts, rows, block_valid, noisy, prior, common


def _variable_case():
    return _variable_case_rows(1)


def _make_ragged_prior_kv_byte_expanded(model: ByteDiffusionModel) -> None:
    """Replace once-per-origin branch K/V with the byte-expanded oracle."""

    for decoder in model.decoder:
        original_condition = decoder.add_split_condition_from_cache

        def expanded_ragged_condition(
            _module,
            values: torch.Tensor,
            cache,
            block_indices: torch.Tensor,
            *,
            block_length: int,
            forward=original_condition,
        ) -> torch.Tensor:
            return forward(
                values.reshape(-1, values.shape[-1]),
                cache,
                block_indices[:, None].expand(-1, block_length).reshape(-1),
            ).view_as(values)

        decoder.add_block_split_condition_from_cache = MethodType(
            expanded_ragged_condition, decoder
        )


@pytest.mark.parametrize("seed", [37, 83])
def test_ragged_blt_once_per_origin_condition_matches_byte_expanded_oracle(
    seed: int,
) -> None:
    """Randomized dense forward and every parameter gradient match exactly."""

    torch.manual_seed(seed)
    config = ByteDiffusionConfig.tiny(
        decoder_layers=2,
        decoder_conditioning="split_cross_attention",
        decoder_prefix_window=None,
        decoder_branch_attention="shared_flex",
    )
    fused = ByteDiffusionModel(config).train()
    with torch.no_grad():
        for parameter in fused.parameters():
            parameter.uniform_(-0.15, 0.15)
    unfused = deepcopy(fused).train()
    ids, valid, starts, rows, block_valid, noisy, prior, common = _variable_case()

    # Randomize all modeled bytes while retaining document-end EOT and padded
    # slots, which are structural inputs to variable patching.
    ids = torch.randint(0, 256, ids.shape)
    ids = ids.masked_fill(~valid, config.vocab.pad_id)
    ids[0, 2] = config.vocab.eot_id
    ids[0, 5] = config.vocab.eot_id
    noisy = torch.randint(0, config.vocab.input_size, noisy.shape)
    noisy = noisy.masked_fill(~block_valid, config.vocab.pad_id)
    prior = prior.clone()
    prior[0] = -1
    arguments = (
        ids,
        valid,
        noisy,
        rows,
        starts,
        prior,
        torch.tensor([0, 2], dtype=torch.int32),
        block_valid,
    )
    _make_ragged_prior_kv_byte_expanded(unfused)

    fused_output = fused.forward_blt_d_ragged(
        *arguments, allow_dense_reference=True, **common
    )
    unfused_output = unfused.forward_blt_d_ragged(
        *arguments, allow_dense_reference=True, **common
    )
    torch.testing.assert_close(
        fused_output.clean_logits, unfused_output.clean_logits, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        fused_output.block_logits, unfused_output.block_logits, rtol=2e-5, atol=2e-6
    )

    fused_loss = fused_output.clean_logits[valid].square().mean()
    fused_loss = fused_loss + fused_output.block_logits[block_valid].square().mean()
    unfused_loss = unfused_output.clean_logits[valid].square().mean()
    unfused_loss = unfused_loss + unfused_output.block_logits[block_valid].square().mean()
    fused_loss.backward()
    unfused_loss.backward()

    unfused_parameters = dict(unfused.named_parameters())
    assert dict(fused.named_parameters()).keys() == unfused_parameters.keys()
    for name, parameter in fused.named_parameters():
        reference_gradient = unfused_parameters[name].grad
        assert parameter.grad is not None, name
        assert reference_gradient is not None, name
        torch.testing.assert_close(
            parameter.grad, reference_gradient, rtol=5e-4, atol=5e-6
        )


def test_ragged_blt_matches_rectangular_dense_forward_and_backward() -> None:
    torch.manual_seed(17)
    config = ByteDiffusionConfig.tiny(
        decoder_conditioning="split_cross_attention",
        decoder_prefix_window=None,
        decoder_branch_attention="shared_flex",
    )
    rectangular_model = ByteDiffusionModel(config).eval()
    ragged_model = deepcopy(rectangular_model).eval()
    ids, valid, starts, rows, block_valid, noisy, prior, common = _variable_case()

    rectangular = rectangular_model.forward_blt_d_branches(
        ids,
        valid,
        noisy[None],
        block_valid[None],
        starts[None],
        branch_condition_indices=prior[None],
        **common,
    )
    ragged = ragged_model.forward_blt_d_ragged(
        ids,
        valid,
        noisy,
        rows,
        starts,
        prior,
        torch.tensor([0, 2], dtype=torch.int32),
        block_valid,
        allow_dense_reference=True,
        **common,
    )
    torch.testing.assert_close(
        ragged.clean_logits, rectangular.clean_logits, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        ragged.block_logits[block_valid],
        rectangular.branch_logits[0][block_valid],
        rtol=2e-5,
        atol=2e-6,
    )

    rectangular_loss = rectangular.clean_logits[valid].square().mean()
    rectangular_loss = rectangular_loss + rectangular.branch_logits[0][block_valid].square().mean()
    ragged_loss = ragged.clean_logits[valid].square().mean()
    ragged_loss = ragged_loss + ragged.block_logits[block_valid].square().mean()
    rectangular_loss.backward()
    ragged_loss.backward()

    rectangular_parameters = dict(rectangular_model.named_parameters())
    for name, parameter in ragged_model.named_parameters():
        reference_grad = rectangular_parameters[name].grad
        if parameter.grad is None or reference_grad is None:
            assert parameter.grad is reference_grad
            continue
        torch.testing.assert_close(parameter.grad, reference_grad, rtol=3e-4, atol=3e-6)

    for layer, decoder in enumerate(ragged_model.decoder):
        for name in (
            "condition.weight",
            "cross_query.weight",
            "cross_key.weight",
            "cross_value.weight",
            "cross_output.weight",
        ):
            parameter = dict(decoder.named_parameters())[name]
            assert parameter.grad is not None, (layer, name)
            assert torch.count_nonzero(parameter.grad) > 0, (layer, name)


def test_ragged_blt_normalized_states_reproduce_dense_logits_without_projection() -> None:
    torch.manual_seed(23)
    config = ByteDiffusionConfig.tiny(
        decoder_conditioning="split_cross_attention",
        decoder_prefix_window=None,
        decoder_branch_attention="shared_flex",
    )
    model = ByteDiffusionModel(config).eval()
    ids, valid, starts, rows, block_valid, noisy, prior, common = _variable_case()
    arguments = (
        ids,
        valid,
        noisy,
        rows,
        starts,
        prior,
        torch.tensor([0, 2], dtype=torch.int32),
        block_valid,
    )

    dense = model.forward_blt_d_ragged(
        *arguments, allow_dense_reference=True, **common
    )
    states = model.forward_blt_d_ragged(
        *arguments,
        return_logits=False,
        return_decoder_states=True,
        allow_dense_reference=True,
        **common,
    )

    assert states.clean_logits.shape == (*ids.shape, 0)
    assert states.block_logits.shape == (*noisy.shape, 0)
    assert states.clean_decoder_states is not None
    assert states.block_decoder_states is not None
    assert model.output is not None
    torch.testing.assert_close(
        model.output(states.clean_decoder_states),
        dense.clean_logits,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        model.output(states.block_decoder_states),
        dense.block_logits,
        rtol=0,
        atol=0,
    )


def test_joint_ragged_forward_dynamic_fullgraph_matches_dense_all_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture the real mixed clean/origin graph over two symbolic products."""

    torch.manual_seed(29)
    # Dense CPU attention interprets cumulative offsets with Python loops and
    # cannot be a dynamic full graph; production uses compiled Flash/Flex.
    # Substitute shape- and gradient-preserving tensor cells only for those
    # backend kernels so the complete real JointForward/model topology around
    # them—including the formerly mixed symbolic pointwise banks—is captured.
    def packed_attention_cell(packed, **_kwargs):
        return packed.query

    def ragged_attention_cell(query, *_args, **_kwargs):
        return query

    monkeypatch.setattr(
        attention_module, "packed_clean_attention", packed_attention_cell
    )
    monkeypatch.setattr(
        model_module, "packed_clean_attention", packed_attention_cell
    )
    monkeypatch.setattr(
        model_module, "ragged_blt_attention", ragged_attention_cell
    )
    config = ByteDiffusionConfig.tiny(
        decoder_conditioning="split_cross_attention",
        decoder_prefix_window=None,
    )
    compiled_model = ByteDiffusionModel(config).train()
    compiled_model.activation_checkpointing = True
    reference_model = deepcopy(compiled_model).train()
    compiled_joint = torch.compile(
        JointForward(compiled_model),
        backend="aot_eager",
        dynamic=True,
        fullgraph=True,
    )
    reference_joint = JointForward(reference_model)

    def invoke(module, row_count: int):
        ids, valid, starts, rows, block_valid, noisy, prior, common = (
            _variable_case_rows(row_count)
        )
        row_cu = torch.arange(
            0, 2 * row_count + 1, 2, dtype=torch.int32
        )
        return module(
            ids,
            valid,
            common["positions"],
            noisy,
            int(ModelMode.BLT_D),
            noisy_valid=block_valid,
            starts=starts,
            block_length=4,
            bos_count=0,
            document_ids=common["document_ids"],
            byte_indices=common["byte_indices"],
            byte_cu_seqlens=common["byte_cu_seqlens"],
            patch_cu_seqlens=common["patch_cu_seqlens"],
            condition_patch_indices=common["condition_patch_indices"],
            global_patch_sources=common["global_patch_sources"],
            global_patch_positions=common["global_patch_positions"],
            physical_to_global_patch_indices=(
                common["physical_to_global_patch_indices"]
            ),
            bos_condition_indices=None,
            branch_condition_indices=prior,
            patch_byte_cu_seqlens=common["patch_byte_cu_seqlens"],
            max_patch_size=common["max_patch_size"],
            ragged_block_rows=rows,
            ragged_row_cu_offsets=row_cu,
            return_ragged_decoder_states=True,
        )

    for row_count in (1, 2):
        reference_output = invoke(reference_joint, row_count)
        compiled_output = invoke(compiled_joint, row_count)
        for actual, expected in zip(
            compiled_output, reference_output, strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        reference_loss = sum(
            value.square().mean() for value in reference_output[:2]
        )
        compiled_loss = sum(
            value.square().mean() for value in compiled_output[:2]
        )
        reference_loss.backward()
        compiled_loss.backward()
        reference_parameters = dict(reference_model.named_parameters())
        for name, parameter in compiled_model.named_parameters():
            reference_parameter = reference_parameters[name]
            if parameter.grad is None or reference_parameter.grad is None:
                assert parameter.grad is None and reference_parameter.grad is None, name
                continue
            torch.testing.assert_close(
                parameter.grad,
                reference_parameter.grad,
                rtol=5e-4,
                atol=5e-6,
            )
        compiled_model.zero_grad(set_to_none=True)
        reference_model.zero_grad(set_to_none=True)


def test_ragged_blt_per_layer_ffn_checkpoint_matches_forward_and_backward() -> None:
    """Separate clean/block FFN HOPs recompute while attention stays saved."""

    torch.manual_seed(29)
    config = ByteDiffusionConfig.tiny(
        decoder_layers=3,
        decoder_conditioning="split_cross_attention",
        decoder_prefix_window=None,
        decoder_branch_attention="shared_flex",
    )
    reference = ByteDiffusionModel(config).train()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.uniform_(-0.12, 0.12)
    checkpointed = deepcopy(reference).train()
    checkpointed.activation_checkpointing = True
    ids, valid, starts, rows, block_valid, noisy, prior, common = _variable_case()
    prior = prior.clone()
    prior[0] = -1
    arguments = (
        ids,
        valid,
        noisy,
        rows,
        starts,
        prior,
        torch.tensor([0, 2], dtype=torch.int32),
        block_valid,
    )

    reference_output = reference.forward_blt_d_ragged(
        *arguments, allow_dense_reference=True, **common
    )
    condition_calls = [0] * config.decoder_layers
    ffn_calls = [0] * config.decoder_layers
    hooks = []
    for layer_index, decoder in enumerate(checkpointed.decoder):
        def count_condition(_module, _inputs, _output, *, index=layer_index) -> None:
            condition_calls[index] += 1

        def count_ffn(_module, _inputs, *, index=layer_index) -> None:
            ffn_calls[index] += 1

        hooks.append(decoder.condition.register_forward_hook(count_condition))
        hooks.append(decoder.block.ffn.register_forward_pre_hook(count_ffn))
    try:
        checkpointed_output = checkpointed.forward_blt_d_ragged(
            *arguments, allow_dense_reference=True, **common
        )
        assert condition_calls == [1] * config.decoder_layers
        assert ffn_calls == [2] * config.decoder_layers
        torch.testing.assert_close(
            checkpointed_output.clean_logits,
            reference_output.clean_logits,
            rtol=2e-5,
            atol=2e-6,
        )
        torch.testing.assert_close(
            checkpointed_output.block_logits,
            reference_output.block_logits,
            rtol=2e-5,
            atol=2e-6,
        )

        reference_loss = reference_output.clean_logits[valid].square().mean()
        reference_loss = (
            reference_loss
            + reference_output.block_logits[block_valid].square().mean()
        )
        checkpointed_loss = checkpointed_output.clean_logits[valid].square().mean()
        checkpointed_loss = (
            checkpointed_loss
            + checkpointed_output.block_logits[block_valid].square().mean()
        )
        reference_loss.backward()
        checkpointed_loss.backward()
        assert condition_calls == [1] * config.decoder_layers
        assert ffn_calls == [4] * config.decoder_layers
    finally:
        for hook in hooks:
            hook.remove()

    reference_parameters = dict(reference.named_parameters())
    for name, parameter in checkpointed.named_parameters():
        reference_parameter = reference_parameters[name]
        assert parameter.grad is not None, name
        assert reference_parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(
            parameter.grad,
            reference_parameter.grad,
            rtol=5e-4,
            atol=5e-6,
        )

    for layer_index, decoder in enumerate(checkpointed.decoder):
        for name, parameter in decoder.named_parameters():
            assert parameter.grad is not None, (layer_index, name)
            assert torch.count_nonzero(parameter.grad) > 0, (layer_index, name)


def test_split_decoder_pointwise_checkpoint_compiles_fullgraph_on_cpu() -> None:
    torch.manual_seed(31)
    config = ByteDiffusionConfig.tiny(
        decoder_conditioning="split_cross_attention"
    )
    layer = ByteDiffusionModel(config).decoder[0].train()
    clean_states = torch.randn(2, 7, config.local_dim, requires_grad=True)
    block_states = torch.randn(5, 4, config.local_dim, requires_grad=True)

    clean_attended = torch.randn_like(clean_states)
    block_attended = torch.randn_like(block_states)

    def checkpoint_cell(
        clean: torch.Tensor,
        blocks: torch.Tensor,
        clean_attention: torch.Tensor,
        block_attention: torch.Tensor,
    ):
        # Exercise the independent symbolic clean/branch pointwise calls that
        # surround the separately-covered attention kernels.
        clean_qkv = layer.block.attention.qkv(
            layer.block.attention_norm(clean)
        )
        block_qkv = layer.block.attention.qkv(
            layer.block.attention_norm(blocks)
        )
        clean = clean + layer.block.attention.output(clean_attention)
        blocks = blocks + layer.block.attention.output(block_attention)
        clean_ffn = checkpoint(
            layer.block.ffn,
            layer.block.ffn_norm(clean),
            use_reentrant=False,
            preserve_rng_state=False,
        )
        block_ffn = checkpoint(
            layer.block.ffn,
            layer.block.ffn_norm(blocks),
            use_reentrant=False,
            preserve_rng_state=False,
        )
        clean_result = clean + clean_ffn
        block_result = blocks + block_ffn
        # JointForward uses the same decoder modules again for its clean AR
        # bank after the ragged bank. Reusing both RMSNorm and FFN outside the
        # HOP catches FX values leaking from a checkpoint subgraph.
        causal_tail = clean_result + layer.block.ffn(
            layer.block.ffn_norm(clean_result)
        )
        return causal_tail, block_result, clean_qkv, block_qkv

    compiled = torch.compile(
        checkpoint_cell, backend="aot_eager", fullgraph=True
    )
    clean_ffn, block_ffn, clean_qkv, block_qkv = compiled(
        clean_states, block_states, clean_attended, block_attended
    )
    (
        clean_ffn.square().mean()
        + block_ffn.square().mean()
        + clean_qkv.square().mean()
        + block_qkv.square().mean()
    ).backward()

    assert clean_states.grad is not None
    assert block_states.grad is not None
    for name, parameter in layer.named_parameters():
        if name.startswith("block.ffn") or name.startswith("block.ffn_norm"):
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_ragged_blt_checkpoint_is_bypassed_without_gradients(monkeypatch) -> None:
    config = ByteDiffusionConfig.tiny(
        decoder_conditioning="split_cross_attention",
        decoder_prefix_window=None,
        decoder_branch_attention="shared_flex",
    )
    model = ByteDiffusionModel(config).train()
    model.activation_checkpointing = True
    ids, valid, starts, rows, block_valid, noisy, prior, common = _variable_case()

    def forbidden_checkpoint(*_args, **_kwargs):
        raise AssertionError("no-grad execution must not invoke checkpoint")

    monkeypatch.setattr(model_module, "activation_checkpoint", forbidden_checkpoint)
    with torch.no_grad():
        output = model.forward_blt_d_ragged(
            ids,
            valid,
            noisy,
            rows,
            starts,
            prior,
            torch.tensor([0, 2], dtype=torch.int32),
            block_valid,
            allow_dense_reference=True,
            **common,
        )
    assert output.block_logits.shape[:2] == noisy.shape


def test_ragged_blt_preserves_original_rope_positions_and_fails_dense_closed() -> None:
    layout = _mask_layout()
    torch.testing.assert_close(
        layout.branch_positions,
        torch.tensor([[2, 3, 4, 5], [1, 2, 3, 4], [1, 2, 3, 4]]),
    )
    heads, width = 1, 4
    query = torch.zeros(1, heads, layout.query_length, width)
    clean = torch.zeros(1, heads, layout.clean_bank_length, width)
    branch = torch.zeros(1, heads, layout.query_length, width)
    with pytest.raises(RuntimeError, match="not permitted|unavailable"):
        ragged_blt_attention(
            query,
            clean,
            clean,
            branch,
            branch,
            layout,
            backend="dense_reference",
            allow_dense_reference=False,
        )
