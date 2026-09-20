"""Compiled CUDA/BF16 checks for document-parallel, token-sequential training."""

import pytest
import torch
import torch.nn.functional as F

from pretraining.nanogpt_mini.full_bandwidth_stream import StreamState, stream_forward
from pretraining.nanogpt_mini.full_bandwidth_stream_attention import cached_attention
from pretraining.nanogpt_mini.nanogpt_mini_full_bandwidth_model import FullBandwidthGPT

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference_attention(q, k, v, keys, values, positions):
    capacity = keys.shape[2]
    slots = torch.arange(capacity, device=q.device)
    valid = (slots[None, :] < positions[:, None]) & (
        slots[None, :] != positions[:, None] % capacity
    )
    mask = torch.cat((valid, torch.ones_like(valid[:, :1])), dim=1)[:, None, None]
    return F.scaled_dot_product_attention(
        q.unsqueeze(2),
        torch.cat((keys.detach(), k.unsqueeze(2)), dim=2),
        torch.cat((values.detach(), v.unsqueeze(2)), dim=2),
        attn_mask=mask,
        scale=0.12,
        enable_gqa=True,
    ).squeeze(2)


def test_cached_attention_matches_sdpa_values_and_current_token_gradients():
    torch.manual_seed(73)
    q, k, v = [
        torch.randn(4, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    keys, values = [
        torch.randn(
            4, 4, 7, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(2)
    ]
    positions = torch.tensor([0, 2, 7, 19], device="cuda")
    weights = torch.randn_like(q)
    actual_fn = torch.compile(cached_attention, fullgraph=True, dynamic=False)
    reference_fn = torch.compile(reference_attention, fullgraph=True, dynamic=False)
    actual = actual_fn(q, k, v, keys, values, positions)
    actual_grads = torch.autograd.grad(
        (actual * weights).sum(), (q, k, v, keys, values), allow_unused=True
    )
    expected = reference_fn(q, k, v, keys, values, positions)
    expected_grads = torch.autograd.grad((expected * weights).sum(), (q, k, v))
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    for observed, reference in zip(actual_grads[:3], expected_grads):
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, reference, rtol=0.04, atol=0.03)
    for historical_gradient in actual_grads[3:]:
        assert (
            historical_gradient is None or torch.count_nonzero(historical_gradient) == 0
        )
    # At a document start only the current value exists; q and k cannot matter.
    torch.testing.assert_close(actual[0], v[0], rtol=0, atol=0)
    assert torch.count_nonzero(actual_grads[0][0]) == 0
    assert torch.count_nonzero(actual_grads[1][0]) == 0


@pytest.fixture(params=[None, 1], ids=["mha", "shared_kv"])
def model(request):
    torch.manual_seed(17)
    result = (
        FullBandwidthGPT(detach_carry=True, noise=0, num_kv_heads=request.param)
        .cuda()
        .eval()
    )
    with torch.no_grad():
        for block in result.blocks:
            block.attn.proj.weight.normal_(std=0.02)
            block.mlp.proj.weight.normal_(std=0.02)
    return result


def compiled_step(model):
    return torch.compile(
        lambda tokens, previous, keys, values, positions, resets: stream_forward(
            model, tokens, previous, keys, values, positions, resets
        ),
        fullgraph=True,
        dynamic=False,
    )


def test_sequential_stream_matches_existing_cached_recurrence(model):
    tokens = torch.randint(0, 1024, (2, 4), device="cuda")
    targets = torch.randint(0, 1024, tokens.shape, device="cuda")
    state = StreamState(model, 2, 4, torch.device("cuda"))
    step = compiled_step(model)
    reference = torch.compile(model.loss_sequential, fullgraph=True, dynamic=False)
    actual = torch.zeros((), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for position in range(4):
            resets = torch.full((2,), position == 0, device="cuda")
            logits, hidden, keys, values = step(
                tokens[:, position],
                state.previous,
                state.keys,
                state.values,
                state.positions,
                resets,
            )
            actual += F.cross_entropy(logits, targets[:, position], reduction="sum")
            state.commit(hidden, keys, values, resets)
        expected = reference(tokens, targets)
    torch.testing.assert_close(
        actual / targets.numel(), expected / targets.numel(), rtol=0, atol=0.04
    )
    torch.testing.assert_close(state.positions, torch.full_like(state.positions, 4))


def test_document_resets_and_other_lanes_cannot_leak_context(model):
    state = StreamState(model, 3, 7, torch.device("cuda"))
    with torch.no_grad():
        state.positions.copy_(torch.tensor([19, 2, 8], device="cuda"))
        state.previous.normal_()
        for key, value in zip(state.keys, state.values):
            key.normal_()
            value.normal_()
    tokens = torch.randint(0, 1024, (3,), device="cuda")
    resets = torch.tensor([True, False, False], device="cuda")
    step = compiled_step(model)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        before = step(
            tokens, state.previous, state.keys, state.values, state.positions, resets
        )
        saved = tuple(value.clone() for value in before)
        # Corrupt lane 0's dead document and lane 1's live document, leaving lane 2 intact.
        state.previous[:2].add_(20)
        for key, value in zip(state.keys, state.values):
            key[:2].mul_(3)
            value[:2].add_(20)
        after = step(
            tokens, state.previous, state.keys, state.values, state.positions, resets
        )
        torch.testing.assert_close(after[0][0], saved[0][0], rtol=0, atol=0)
        torch.testing.assert_close(after[0][2], saved[0][2], rtol=0, atol=0)
        assert not torch.equal(after[0][1], saved[0][1])
        state.commit(after[1], after[2], after[3], resets)
    torch.testing.assert_close(state.positions, torch.tensor([1, 3, 9], device="cuda"))


def test_local_backward_then_cache_commit_can_repeat_without_temporal_graph(model):
    model.train()
    state = StreamState(model, 3, 4, torch.device("cuda"))
    step = compiled_step(model)
    for position in range(6):
        tokens = torch.randint(0, 1024, (3,), device="cuda")
        targets = torch.randint(0, 1024, (3,), device="cuda")
        resets = torch.full((3,), position == 0, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, hidden, keys, values = step(
                tokens,
                state.previous,
                state.keys,
                state.values,
                state.positions,
                resets,
            )
            loss = F.cross_entropy(logits, targets)
        loss.backward()
        state.commit(hidden, keys, values, resets)
        torch.testing.assert_close(state.previous, hidden, rtol=0, atol=0)
        assert not state.previous.requires_grad
        assert all(not key.requires_grad for key in state.keys)
        if position > 0:
            assert model.fuse_value.weight.grad.norm() > 0
        assert torch.isfinite(model.blocks[0].attn.v.weight.grad).all()
        model.zero_grad(set_to_none=True)


def test_shared_kv_gradients_sum_all_query_heads_without_history_gradients():
    torch.manual_seed(71)
    q = torch.randn(4, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k, v = [
        torch.randn(4, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(2)
    ]
    keys, values = [
        torch.randn(
            4, 1, 1024, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(2)
    ]
    positions = torch.tensor([0, 2, 1024, 2049], device="cuda")
    weights = torch.randn_like(q)
    actual_fn = torch.compile(cached_attention, fullgraph=True, dynamic=False)
    reference_fn = torch.compile(reference_attention, fullgraph=True, dynamic=False)
    actual = actual_fn(q, k, v, keys, values, positions)
    actual_grads = torch.autograd.grad(
        (actual * weights).sum(), (q, k, v, keys, values), allow_unused=True
    )
    expected = reference_fn(q, k, v, keys, values, positions)
    expected_grads = torch.autograd.grad((expected * weights).sum(), (q, k, v))
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    for observed, reference in zip(actual_grads[:3], expected_grads):
        torch.testing.assert_close(observed, reference, rtol=0.04, atol=0.03)
        # Long histories have small current-token derivatives: an absolute
        # elementwise tolerance alone would incorrectly accept zero gradients.
        error = (observed.float() - reference.float()).flatten(1).norm(dim=1)
        magnitude = reference.float().flatten(1).norm(dim=1)
        assert torch.all(error <= 0.04 * magnitude + 1e-4)
    assert actual_grads[3] is None and actual_grads[4] is None
    torch.testing.assert_close(actual[0], v[0].expand(4, -1), rtol=0, atol=0)
    torch.testing.assert_close(
        actual_grads[2][0],
        weights[0].float().sum(dim=0, keepdim=True).bfloat16(),
        rtol=0,
        atol=0,
    )
    assert torch.count_nonzero(actual_grads[0][0]) == 0
    assert torch.count_nonzero(actual_grads[1][0]) == 0


def test_query_groups_cannot_read_another_kv_heads_history():
    torch.manual_seed(97)
    q = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16)
    k, v = [
        torch.randn(2, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    keys, values = [
        torch.randn(2, 2, 7, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    positions = torch.tensor([7, 9], device="cuda")
    forward = torch.compile(cached_attention, fullgraph=True, dynamic=False)
    actual = forward(q, k, v, keys, values, positions).clone()
    changed_values = values.clone()
    changed_values[0, 1].add_(10)
    changed = forward(q, k, v, keys, changed_values, positions)
    torch.testing.assert_close(actual[0, :2], changed[0, :2], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], changed[1], rtol=0, atol=0)
    assert not torch.equal(actual[0, 2:], changed[0, 2:])


def test_plain_cached_prefix_matches_parallel_attention(model):
    tokens = torch.randint(0, 1024, (2, 8), device="cuda")
    targets = torch.randint(0, 1024, tokens.shape, device="cuda")
    parallel = torch.compile(model, fullgraph=True, dynamic=False)
    cached = torch.compile(model.loss_sequential, fullgraph=True, dynamic=False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        parallel_loss = parallel(tokens, targets, passes=1)
        cached_loss = cached(tokens, targets, prefix_length=tokens.shape[1])
    torch.testing.assert_close(
        parallel_loss / targets.numel(),
        cached_loss / targets.numel(),
        rtol=0,
        atol=0.02,
    )
