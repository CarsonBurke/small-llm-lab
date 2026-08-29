from __future__ import annotations

import copy

import pytest
import torch
from torch import Tensor

from pretraining.byte_diffusion.layers import (
    ConditionedTransformerBlock,
    PatchPool,
)


def _patch_offsets(lengths: list[int]) -> Tensor:
    values = torch.tensor(lengths, dtype=torch.int32)
    return torch.cat(
        (torch.zeros(1, dtype=torch.int32), values.cumsum(0, dtype=torch.int32))
    )


def _split_block(*, dtype: torch.dtype = torch.float64) -> ConditionedTransformerBlock:
    return ConditionedTransformerBlock(
        local_dim=8,
        global_dim=16,
        heads=2,
        ffn_dim=16,
        rope_theta=10_000.0,
        conditioning="split_cross_attention",
        split_residual_scale=0.7,
    ).to(dtype=dtype)


def _expanded_conditions(unique: Tensor, indices: Tensor) -> Tensor:
    zero = unique.new_zeros((1, unique.shape[-1]))
    return torch.cat((zero, unique), dim=0).index_select(
        0, (indices.reshape(-1) + 1)
    ).view(*indices.shape, unique.shape[-1])


def test_segmented_patch_pool_forward_and_backward_match_padded_oracle() -> None:
    """Two packed documents jointly cover every supported patch length 1..8."""

    torch.manual_seed(101)
    segmented = PatchPool(local_dim=8, global_dim=16, heads=2, stride=4).double()
    oracle = copy.deepcopy(segmented)
    # Each half is one document's ordered patch partition.  Concatenating the
    # lengths does not create a padded or cross-document patch.
    offsets = _patch_offsets([1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1])
    local = torch.randn(int(offsets[-1]), 8, dtype=torch.float64, requires_grad=True)
    oracle_local = local.detach().clone().requires_grad_()

    actual = segmented.forward_packed(local, offsets, max_patch_size=8)
    expected = oracle.forward_packed_padded_reference(
        oracle_local, offsets, max_patch_size=8
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    output_gradient = torch.randn_like(actual)
    actual_inputs = (local, *segmented.parameters())
    expected_inputs = (oracle_local, *oracle.parameters())
    actual_gradients = torch.autograd.grad(actual, actual_inputs, output_gradient)
    expected_gradients = torch.autograd.grad(
        expected, expected_inputs, output_gradient
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=1e-11, atol=1e-11
        )


def test_segmented_patch_pool_never_calls_padded_pool(monkeypatch) -> None:
    pool = PatchPool(local_dim=8, global_dim=16, heads=2, stride=4)
    offsets = _patch_offsets([1, 8, 3, 6])
    local = torch.randn(int(offsets[-1]), 8)
    key_value_shapes: list[torch.Size] = []

    def reject_padded_bank(*_args, **_kwargs):
        raise AssertionError("packed production pooling formed a padded bank")

    monkeypatch.setattr(pool, "_pool_padded", reject_padded_bank)
    handle = pool.key_value.register_forward_pre_hook(
        lambda _module, inputs: key_value_shapes.append(inputs[0].shape)
    )
    result = pool.forward_packed(local, offsets, max_patch_size=8)
    handle.remove()

    assert result.shape == (4, 16)
    assert torch.isfinite(result).all()
    assert key_value_shapes == [torch.Size([18, 8])]


def test_segmented_patch_pool_is_bfloat16_safe() -> None:
    torch.manual_seed(103)
    segmented = PatchPool(local_dim=8, global_dim=16, heads=2, stride=4).to(
        dtype=torch.bfloat16
    )
    oracle = copy.deepcopy(segmented)
    offsets = _patch_offsets(list(range(1, 9)))
    local = torch.randn(int(offsets[-1]), 8, dtype=torch.bfloat16)

    actual = segmented.forward_packed(local, offsets, max_patch_size=8)
    expected = oracle.forward_packed_padded_reference(
        local, offsets, max_patch_size=8
    )

    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=8e-3)


def test_segmented_patch_pool_compiles_with_varying_packed_counts() -> None:
    torch.manual_seed(107)
    pool = PatchPool(local_dim=8, global_dim=16, heads=2, stride=4).eval()

    def forward(local: Tensor, offsets: Tensor) -> Tensor:
        return pool.forward_packed(local, offsets, max_patch_size=8)

    compiled = torch.compile(forward, backend="eager", fullgraph=True, dynamic=True)
    for lengths in ([1, 2, 8, 3], [8, 7, 6, 5, 4, 3, 2, 1]):
        offsets = _patch_offsets(list(lengths))
        local = torch.randn(int(offsets[-1]), 8, requires_grad=True)
        actual = compiled(local, offsets)
        expected = forward(local, offsets)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual_gradient = torch.autograd.grad(actual.square().sum(), local)[0]
        expected_gradient = torch.autograd.grad(expected.square().sum(), local)[0]
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=0, atol=0
        )


def test_split_condition_cache_matches_expanded_forward_and_backward() -> None:
    torch.manual_seed(109)
    expanded_block = _split_block()
    cached_block = copy.deepcopy(expanded_block)
    indices = torch.tensor(
        [[-1, 0, 1, 1, 2], [2, 0, -1, 2, 1]], dtype=torch.long
    )
    states = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    unique = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    cached_states = states.detach().clone().requires_grad_()
    cached_unique = unique.detach().clone().requires_grad_()

    expanded = expanded_block._add_condition(
        states, _expanded_conditions(unique, indices)
    )
    cache = cached_block.prepare_split_condition_cache(cached_unique)
    cached = cached_block.add_split_condition_from_cache(
        cached_states, cache, indices
    )
    torch.testing.assert_close(cached, expanded, rtol=0, atol=0)

    output_gradient = torch.randn_like(expanded)
    expanded_inputs = (states, unique, *expanded_block.parameters())
    cached_inputs = (cached_states, cached_unique, *cached_block.parameters())
    expanded_gradients = torch.autograd.grad(
        expanded,
        expanded_inputs,
        output_gradient,
        allow_unused=True,
    )
    cached_gradients = torch.autograd.grad(
        cached,
        cached_inputs,
        output_gradient,
        allow_unused=True,
    )
    for expanded_gradient, cached_gradient in zip(
        expanded_gradients, cached_gradients, strict=True
    ):
        assert (expanded_gradient is None) == (cached_gradient is None)
        if expanded_gradient is not None and cached_gradient is not None:
            # Projecting repeated latents once changes GEMM accumulation order
            # but not the gradient.  The difference is within FP32 rounding.
            torch.testing.assert_close(
                cached_gradient, expanded_gradient, rtol=1e-6, atol=5e-7
            )


def test_split_condition_cache_projects_unique_latents_only_once() -> None:
    torch.manual_seed(113)
    block = _split_block(dtype=torch.float32)
    unique = torch.randn(4, 16)
    indices = torch.tensor([[-1, 0, 0, 3], [2, 2, 1, 3]], dtype=torch.long)
    states = torch.randn(2, 4, 8)
    projection_shapes: list[torch.Size] = []

    handle = block.condition.register_forward_pre_hook(
        lambda _module, inputs: projection_shapes.append(inputs[0].shape)
    )
    cache = block.prepare_split_condition_cache(unique)
    first = block.add_split_condition_from_cache(states, cache, indices)
    second = block.add_split_condition_from_cache(states, cache, indices)
    handle.remove()

    assert projection_shapes == [torch.Size([4, 16])]
    assert cache.key.shape == (4, 2, 2, 4)
    assert cache.value.shape == cache.key.shape
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_split_condition_cache_bos_is_exact_zero_and_bfloat16_safe() -> None:
    torch.manual_seed(127)
    block = _split_block(dtype=torch.bfloat16)
    unique = torch.randn(3, 16, dtype=torch.bfloat16)
    states = torch.randn(6, 8, dtype=torch.bfloat16)
    all_bos = torch.full((6,), -1, dtype=torch.long)

    cache = block.prepare_split_condition_cache(unique)
    conditioned = block.add_split_condition_from_cache(states, cache, all_bos)

    assert conditioned.dtype == torch.bfloat16
    assert torch.isfinite(conditioned).all()
    torch.testing.assert_close(conditioned, states, rtol=0, atol=0)


def test_split_condition_cache_empty_bank_supports_only_virtual_bos() -> None:
    torch.manual_seed(128)
    block = _split_block(dtype=torch.float32)
    unique = torch.empty(0, 16, requires_grad=True)
    states = torch.randn(6, 8, requires_grad=True)
    all_bos = torch.full((6,), -1, dtype=torch.long)

    cache = block.prepare_split_condition_cache(unique)
    conditioned = block.add_split_condition_from_cache(states, cache, all_bos)
    torch.testing.assert_close(conditioned, states, rtol=0, atol=0)
    conditioned.square().sum().backward()
    assert states.grad is not None
    assert torch.isfinite(states.grad).all()

    with pytest.raises(ValueError, match="empty latent bank"):
        block.add_split_condition_from_cache(
            states.detach(), cache, torch.zeros(6, dtype=torch.long)
        )


def test_compiled_split_condition_cache_empty_bank_forward_backward() -> None:
    torch.manual_seed(129)
    block = _split_block(dtype=torch.float32)
    unique = torch.empty(0, 16)
    all_bos = torch.full((6,), -1, dtype=torch.long)

    def apply(states: torch.Tensor) -> torch.Tensor:
        cache = block.prepare_split_condition_cache(unique)
        return block.add_split_condition_from_cache(states, cache, all_bos)

    compiled = torch.compile(apply, dynamic=True, fullgraph=True)
    states = torch.randn(6, 8, requires_grad=True)
    actual = compiled(states)
    torch.testing.assert_close(actual, states, rtol=0, atol=0)
    actual.square().sum().backward()
    assert states.grad is not None
    torch.testing.assert_close(states.grad, 2 * states.detach())


def test_split_condition_cache_matches_expanded_path_under_autocast() -> None:
    torch.manual_seed(129)
    block = _split_block(dtype=torch.float32)
    unique = torch.randn(3, 16)
    states = torch.randn(6, 8)
    indices = torch.tensor([-1, 0, 1, 1, 2, 0], dtype=torch.long)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = block._add_condition(
            states, _expanded_conditions(unique, indices)
        )
        cache = block.prepare_split_condition_cache(unique)
        actual = block.add_split_condition_from_cache(states, cache, indices)

    assert cache.key.dtype == torch.bfloat16
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_split_condition_cache_compiles_with_varying_latent_and_byte_counts() -> None:
    torch.manual_seed(131)
    block = _split_block(dtype=torch.float32).eval()

    def forward(states: Tensor, unique: Tensor, indices: Tensor) -> Tensor:
        cache = block.prepare_split_condition_cache(unique)
        return block.add_split_condition_from_cache(states, cache, indices)

    compiled = torch.compile(forward, backend="eager", fullgraph=True, dynamic=True)
    for latent_count, byte_count in ((3, 7), (5, 11)):
        states = torch.randn(byte_count, 8, requires_grad=True)
        unique = torch.randn(latent_count, 16, requires_grad=True)
        indices = torch.arange(byte_count, dtype=torch.long) % latent_count
        indices[0] = -1

        actual = compiled(states, unique, indices)
        expected = forward(states, unique, indices)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        actual_gradients = torch.autograd.grad(
            actual.square().sum(), (states, unique)
        )
        expected_gradients = torch.autograd.grad(
            expected.square().sum(), (states, unique)
        )
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients, strict=True
        ):
            torch.testing.assert_close(
                actual_gradient, expected_gradient, rtol=0, atol=0
            )


@pytest.mark.parametrize("invalid_index", [-2, 3])
def test_split_condition_cache_rejects_invalid_indices(invalid_index: int) -> None:
    block = _split_block(dtype=torch.float32)
    states = torch.randn(2, 8)
    unique = torch.randn(3, 16)
    cache = block.prepare_split_condition_cache(unique)

    with pytest.raises(ValueError, match="outside the cached latent bank"):
        block.add_split_condition_from_cache(
            states, cache, torch.tensor([0, invalid_index], dtype=torch.long)
        )
