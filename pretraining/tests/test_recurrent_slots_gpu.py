"""CUDA/bf16 recurrence contracts; execute only through mlq.

Eager execution is an independent numerical oracle, never a trainer fallback.
Set RECURRENT_SLOTS_REQUIRE_CUDA=1 in queued validation to fail on missing CUDA.
The compilation test uses production widths and a complete 16-token segment.
"""
from __future__ import annotations

import copy
import os

import pytest
import torch
import torch.nn.functional as F

from pretraining.nanogpt_mini.recurrent_slots import RecurrentSlots


@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        if os.environ.get("RECURRENT_SLOTS_REQUIRE_CUDA") == "1":
            pytest.fail("queued recurrence contracts require CUDA")
        pytest.skip("CUDA contracts must be submitted through mlq")
    if not torch.cuda.is_bf16_supported():
        pytest.fail("recurrent slots require CUDA bf16")


def model(*, production=False):
    torch.manual_seed(981)
    kwargs = {} if production else dict(vocab_size=32, num_layers=3,
                                        model_dim=32, head_dim=16,
                                        slots=5, writer_dim=16)
    net = RecurrentSlots(**kwargs).cuda()
    # Zero-initialized read and vocabulary outputs hide broken memory paths.
    with torch.no_grad():
        for name, param in net.named_parameters():
            if name.endswith("proj.weight"):
                param.normal_(std=0.03)
    return net


def inputs(net, length=7):
    return torch.randint(net.config["vocab_size"], (2, length), device="cuda")


def memory(net):
    return torch.randn(2, net.config["slots"], net.config["model_dim"],
                       dtype=torch.bfloat16, device="cuda") * 0.1


def objective(net, hidden, state, targets):
    # Final-state term also checks the state returned across checkpoint borders.
    return F.cross_entropy(net.logits(hidden).flatten(0, 1), targets.flatten()) + state.float().square().mean()


def gradients(net):
    result = {}
    for name, param in net.named_parameters():
        assert param.grad is not None, f"disconnected parameter: {name}"
        assert torch.isfinite(param.grad).all(), name
        result[name] = param.grad.detach().float().clone()
    return result


def test_causal_prefix_and_incremental_sequence_agree():
    net = model().eval()
    tokens = inputs(net)
    initial = memory(net)
    with torch.no_grad():
        hidden, final = net.forward_hidden(tokens, initial, segment_size=3)
        alternate = tokens.clone()
        alternate[:, 4:] = (alternate[:, 4:] + 1) % net.config["vocab_size"]
        changed, _ = net.forward_hidden(alternate, initial, segment_size=2)
        prefix, prefix_memory = net.forward_hidden(tokens[:, :4], initial, segment_size=3)
        torch.testing.assert_close(hidden[:, :4], changed[:, :4], rtol=0, atol=0)
        torch.testing.assert_close(hidden[:, :4], prefix, rtol=0, atol=0)
        states, state = [], initial
        for token in tokens.unbind(1):
            current, state = net.step(token, state)
            states.append(current)
        torch.testing.assert_close(hidden, torch.stack(states, 1), rtol=0, atol=0)
        torch.testing.assert_close(final, state, rtol=0, atol=0)
        suffix, resumed = net.forward_hidden(tokens[:, 4:], prefix_memory, segment_size=2)
        torch.testing.assert_close(hidden[:, 4:], suffix, rtol=0, atol=0)
        torch.testing.assert_close(final, resumed, rtol=0, atol=0)


def test_all_reads_share_old_snapshot_and_writes_are_out_of_place():
    net = model().eval()
    other = copy.deepcopy(net)
    with torch.no_grad():
        other.writer_candidate.weight.mul_(-2)
        other.writer_gate.bias.fill_(2)
    tokens, initial = inputs(net, 2), memory(net)
    old = initial.clone()
    seen = []
    hooks = [block.register_forward_pre_hook(
        lambda _module, args: seen.append((args[1].data_ptr(), args[2].data_ptr())))
        for block in net.blocks]
    try:
        with torch.no_grad():
            first, updated = net.step(tokens[:, 0], initial)
    finally:
        for hook in hooks:
            hook.remove()
    assert len(seen) == len(net.blocks) and len(set(seen)) == 1
    torch.testing.assert_close(initial, old, rtol=0, atol=0)
    assert updated.data_ptr() != initial.data_ptr()
    with torch.no_grad():
        other_first, other_updated = other.step(tokens[:, 0], initial)
        next_hidden, _ = net.step(tokens[:, 1], updated)
        other_next, _ = other.step(tokens[:, 1], other_updated)
    torch.testing.assert_close(first, other_first, rtol=0, atol=0)
    assert not torch.equal(updated, other_updated)
    assert not torch.equal(next_hidden, other_next), "writer changes must influence future reads"


def test_writer_gets_future_loss_without_affecting_current_prediction():
    net = model()
    tokens = inputs(net, 2)
    first, next_memory = net.step(tokens[:, 0], memory(net))
    writer = list(net.writer_candidate.parameters())
    current_grads = torch.autograd.grad(net.logits(first).square().mean(), writer,
                                        allow_unused=True, retain_graph=True)
    assert all(grad is None for grad in current_grads)
    next_memory.retain_grad()
    second, _ = net.step(tokens[:, 1], next_memory)
    F.cross_entropy(net.logits(second), (tokens[:, 1] + 1) % net.config["vocab_size"]).backward()
    assert next_memory.grad is not None and next_memory.grad.float().norm() > 0
    for name, param in net.named_parameters():
        if name.startswith("writer_"):
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            assert param.grad.float().norm() > 0, f"no future credit: {name}"


def test_checkpoint_segments_preserve_full_recurrence_gradients():
    net = model().train()
    oracle = copy.deepcopy(net)
    tokens, initial = inputs(net), memory(net)
    targets = (tokens + 1) % net.config["vocab_size"]
    tracked = initial.clone().requires_grad_()
    hidden, state = net.forward_hidden(tokens, tracked, segment_size=3)
    objective(net, hidden, state, targets).backward()
    actual = gradients(net)
    reference_memory = initial.clone().requires_grad_()
    reference, final = oracle._segment(tokens, reference_memory)
    objective(oracle, reference, final, targets).backward()
    expected = gradients(oracle)
    torch.testing.assert_close(hidden, reference, rtol=0, atol=0)
    torch.testing.assert_close(state, final, rtol=0, atol=0)
    torch.testing.assert_close(tracked.grad, reference_memory.grad, rtol=0.001, atol=1e-6)
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0.001, atol=1e-6, msg=name)


def test_compiled_production_segment_forward_and_backward_match_eager():
    net = model(production=True).train()
    oracle = copy.deepcopy(net)
    tokens, initial = inputs(net, 16), memory(net)
    targets = (tokens + 1) % net.config["vocab_size"]
    net.compile_segments()
    tracked = initial.clone().requires_grad_()
    hidden, state = net.forward_hidden(tokens, tracked, segment_size=16)
    loss = objective(net, hidden, state, targets)
    loss.backward()
    actual = gradients(net)
    reference_memory = initial.clone().requires_grad_()
    reference, final = oracle._segment(tokens, reference_memory)
    reference_loss = objective(oracle, reference, final, targets)
    reference_loss.backward()
    expected = gradients(oracle)
    # Fused bf16 kernels round differently; compare norms as well as outputs so
    # small gradients cannot silently pass a blanket absolute tolerance.
    torch.testing.assert_close(hidden.float(), reference.float(), rtol=0.05, atol=0.06)
    torch.testing.assert_close(state.float(), final.float(), rtol=0.05, atol=0.01)
    torch.testing.assert_close(loss, reference_loss, rtol=0.005, atol=0.005)
    actual["initial_memory"] = tracked.grad.float()
    expected["initial_memory"] = reference_memory.grad.float()
    for name in actual:
        lhs, rhs = actual[name].flatten(), expected[name].flatten()
        relative_error = (lhs - rhs).norm() / rhs.norm().clamp_min(1e-12)
        assert relative_error < 0.15, f"{name}: relative gradient error {relative_error.item():.5f}"
        if rhs.norm() > 1e-10:
            cosine = F.cosine_similarity(lhs, rhs, dim=0, eps=1e-20)
            assert cosine > 0.99, f"{name}: gradient cosine {cosine.item():.5f}"


def test_zero_initial_memory_has_finite_forward_and_backward():
    net = model().train()
    tokens = inputs(net, 4)
    hidden, state = net.forward_hidden(tokens, segment_size=2)
    assert hidden.dtype == torch.bfloat16 and state.dtype == torch.bfloat16
    assert torch.isfinite(hidden).all() and torch.isfinite(state).all()
    loss = objective(net, hidden, state, (tokens + 1) % net.config["vocab_size"])
    assert torch.isfinite(loss)
    loss.backward()
    gradients(net)


def test_compiled_full_horizon_zero_memory_stability():
    """Actual 1024-token horizon, with only batch reduced for contract cost."""
    net = model(production=True).train()
    with torch.no_grad():
        for name, param in net.named_parameters():
            if name.endswith("proj.weight"):
                param.mul_(0.1)
    net.compile_segments()
    tokens = inputs(net, 1024)
    hidden, state = net.forward_hidden(tokens, segment_size=16)
    assert hidden.dtype == torch.bfloat16 and state.dtype == torch.bfloat16
    assert torch.isfinite(hidden).all(), "nonfinite full-horizon hidden state"
    assert torch.isfinite(state).all(), "nonfinite full-horizon memory"
    loss = objective(net, hidden, state, (tokens + 1) % net.config["vocab_size"])
    assert torch.isfinite(loss)
    loss.backward()
    checked = gradients(net)
    assert checked["writer_candidate.weight"].norm() > 0
    assert checked["writer_gate.weight"].norm() > 0
