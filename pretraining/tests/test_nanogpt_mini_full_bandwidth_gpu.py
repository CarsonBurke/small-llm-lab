"""Paper invariants on the actual 6x512 mini trunk, compiled CUDA/BF16.

Run through mlq. These are correctness tests, not reduced training experiments
or evidence of language-model quality.
"""

import pytest
import torch

from pretraining.nanogpt_mini.nanogpt_mini_full_bandwidth_model import FullBandwidthGPT

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def model():
    torch.manual_seed(2718)
    result = FullBandwidthGPT(noise=0.0).cuda()
    result.initialize_parameters()
    # Exercise attention paths as well as the identity residual initialization.
    with torch.no_grad():
        for block in result.blocks:
            block.attn.proj.weight.normal_(std=0.02)
            block.mlp.proj.weight.normal_(std=0.02)
    return result.eval()


def batch(length=8):
    inputs = torch.randint(0, 1024, (2, length), device="cuda")
    targets = torch.randint(0, 1024, inputs.shape, device="cuda")
    return inputs, targets


def test_prefix_boundary_and_no_future_token_leakage(model):
    inputs, _ = batch()
    prefixes = torch.tensor([1, 5], device="cuda")
    run_pass = torch.compile(model.pass_forward, fullgraph=True, dynamic=False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        plain = run_pass(inputs)
        fused = run_pass(inputs, plain, prefixes)
        for row, prefix in enumerate((1, 5)):
            torch.testing.assert_close(
                fused[row, :prefix], plain[row, :prefix], rtol=0, atol=0
            )
        changed = inputs.clone()
        changed[:, 5:] = (changed[:, 5:] + 123) % 1024
        changed_plain = run_pass(changed)
        changed_fused = run_pass(changed, changed_plain, prefixes)
        torch.testing.assert_close(changed_fused[:, :5], fused[:, :5], rtol=0, atol=0)
        assert not torch.equal(fused[:, 5:], plain[:, 5:])


@pytest.mark.parametrize("detach_carry", [False, True])
def test_equation12_loss_and_cross_pass_gradient_policy(model, detach_carry):
    model.detach_carry = detach_carry
    inputs, targets = batch()

    def reference(tokens, labels):
        first = model.pass_forward(tokens)
        second = model.pass_forward(tokens, first.detach() if detach_carry else first)
        third = model.pass_forward(tokens, second.detach() if detach_carry else second)
        return (
            model.head_loss(first, labels)
            + (model.head_loss(second, labels) + model.head_loss(third, labels)) / 2
        )

    actual = torch.compile(model, fullgraph=True, dynamic=False)
    expected = torch.compile(reference, fullgraph=True, dynamic=False)
    parameters = (model.embed.weight, model.fuse_value.weight, model.fuse_gate.weight)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = actual(inputs, targets, passes=3)
    gradients = torch.autograd.grad(loss, parameters)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reference_loss = expected(inputs, targets)
    reference_gradients = torch.autograd.grad(reference_loss, parameters)
    torch.testing.assert_close(loss, reference_loss, rtol=0.002, atol=0.01)
    for gradient, reference_gradient in zip(gradients, reference_gradients):
        assert torch.isfinite(gradient).all()
        assert gradient.norm() > 0
        torch.testing.assert_close(gradient, reference_gradient, rtol=0.03, atol=0.01)


def test_cached_recurrence_matches_converged_jacobi_with_plain_prefix(model):
    inputs, targets = batch(length=4)
    prefixes = torch.full((2,), 2, device="cuda")
    run_pass = torch.compile(model.pass_forward, fullgraph=True, dynamic=False)
    sequential = torch.compile(model.loss_sequential, fullgraph=True, dynamic=False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = run_pass(inputs)
        # Plain positions 0/1, then two causal feedback transitions.
        for _ in range(2):
            state = run_pass(inputs, state, prefixes)
        expected = model.head_loss(state, targets)
        actual = sequential(inputs, targets, prefix_length=2)
    # Different SDPA query shapes accumulate BF16 rounding differences.
    torch.testing.assert_close(
        actual / targets.numel(), expected / targets.numel(), rtol=0, atol=0.04
    )


def test_glu_cannot_bypass_state_and_readout_is_tied(model):
    inputs, _ = batch()
    zero_state = torch.zeros(2, 8, 512, device="cuda", dtype=torch.bfloat16)
    fuse = torch.compile(model.fused_input, fullgraph=True, dynamic=False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        fused = fuse(inputs, zero_state)
        torch.testing.assert_close(
            fused[:, 1:], torch.zeros_like(fused[:, 1:]), rtol=0, atol=0
        )
        hidden = torch.randn_like(zero_state)
        before = model.logits(hidden)
        model.embed.weight[17].add_(0.125)
        after = model.logits(hidden)
        assert not torch.equal(before[..., 17], after[..., 17])
        torch.testing.assert_close(before[..., :17], after[..., :17], rtol=0, atol=0)


@pytest.mark.parametrize("detach_carry", [False, True])
def test_training_prefix_mixin_and_jitter_are_compilable(model, detach_carry):
    model.detach_carry = detach_carry
    inputs, targets = batch()
    model.noise = 0.02
    model.train()
    forward = torch.compile(model, fullgraph=True, dynamic=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = forward(inputs, targets, passes=3, z_loss=1e-5)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.fuse_gate.weight.grad is not None
    assert torch.isfinite(model.fuse_gate.weight.grad).all()
    assert model.fuse_gate.weight.grad.norm() > 0


def test_detached_carry_preserves_forward_and_cuts_only_producer_gradient(model):
    inputs, targets = batch()
    carried = torch.randn(
        2, 8, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    def feedback_loss(tokens, previous, labels):
        return model.head_loss(model.pass_forward(tokens, previous), labels)

    forward = torch.compile(feedback_loss, fullgraph=True, dynamic=False)
    readers = (
        model.embed.weight,
        model.fuse_value.weight,
        model.fuse_gate.weight,
        model.blocks[0].attn.k.weight,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        attached_loss = forward(inputs, carried, targets)
    attached_gradients = torch.autograd.grad(attached_loss, (carried, *readers))
    assert attached_gradients[0].norm() > 0

    model.detach_carry = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        detached_loss = forward(inputs, carried, targets)
    detached_gradients = torch.autograd.grad(
        detached_loss, (carried, *readers), allow_unused=True
    )
    torch.testing.assert_close(attached_loss, detached_loss, rtol=0, atol=0)
    assert (
        detached_gradients[0] is None or torch.count_nonzero(detached_gradients[0]) == 0
    )
    for attached, detached in zip(attached_gradients[1:], detached_gradients[1:]):
        assert detached is not None and torch.isfinite(detached).all()
        assert detached.norm() > 0
        # The externally supplied producer is the only cut edge in this graph.
        torch.testing.assert_close(attached, detached, rtol=0.03, atol=0.01)
