"""CUDA/BF16 writer, carry-readout, and detached recurrence contracts; execute through mlq."""

import pytest
import torch
import torch.nn.functional as F

from pretraining.future_credit_stream.model import GATE_BIAS_INIT, StreamingFFNModel


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _model(*, use_writer=True, activate_residuals=True):
    torch.manual_seed(31)
    model = StreamingFFNModel(
        vocab_size=32, model_dim=128, num_layers=3, mlp_hidden=256, use_writer=use_writer,
    ).cuda()
    if activate_residuals:
        # Nonzero trained-state readout and residuals expose gradients.
        with torch.no_grad():
            model.proj.weight.normal_(std=0.02)
            model.proj.bias.normal_(std=0.02)
            for block in model.blocks:
                block[1].proj.weight.normal_(std=0.02)
    return model


def _assert_signal(gradient):
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_writer_preserves_common_initialization_and_starts_as_ce_recursion():
    plain = _model(use_writer=False, activate_residuals=False)
    model = _model(activate_residuals=False)
    plain_parameters = dict(plain.named_parameters())
    parameters = dict(model.named_parameters())
    assert plain.writer is None
    assert set(plain_parameters) == {name for name in parameters if not name.startswith("writer.")}
    for name, parameter in plain_parameters.items():
        torch.testing.assert_close(parameter, parameters[name], rtol=0, atol=0)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())

    observed = torch.tensor([2, 3, 4], device="cuda")
    carry = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        plain_logits, plain_hidden = plain(observed, carry)
        logits, hidden = model(observed, carry)
        _, _, plain_next = plain.step(observed, carry)
        _, _, next_carry = model.step(observed, carry)
    torch.testing.assert_close(logits, plain_logits, rtol=0, atol=0)
    torch.testing.assert_close(hidden, plain_hidden, rtol=0, atol=0)
    torch.testing.assert_close(plain_next, plain_hidden, rtol=0, atol=0)
    # The untrained writer is CE recursion with a small retention leak; both
    # terms enter at unit RMS so the gate alone sets the mix.
    gate = torch.sigmoid(torch.tensor(GATE_BIAS_INIT))
    unit = lambda vector: F.rms_norm(vector.float(), (128,))
    expected = (gate * unit(hidden) + (1 - gate) * unit(carry)).bfloat16()
    torch.testing.assert_close(next_carry, expected, rtol=2e-2, atol=2e-2)
    assert next_carry.dtype == torch.bfloat16
    # A reset lane contributes nothing through the retention path.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        from_reset, _ = model.writer(hidden, model.initial_state(3, "cuda"))
    torch.testing.assert_close(from_reset, (gate * unit(hidden)).bfloat16(), rtol=2e-2, atol=2e-2)


def test_forward_detaches_the_carry_unless_a_temporal_gradient_is_requested():
    model = _model()
    incoming = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    observed = torch.tensor([4, 5], device="cuda")
    targets = torch.tensor([6, 7], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(observed, incoming)
        reset_logits, _ = model(observed, model.initial_state(2, "cuda"))
        F.cross_entropy(logits, targets).backward()
    assert incoming.grad is None
    assert not torch.equal(logits, reset_logits)
    assert all(parameter.grad is None for parameter in model.writer.parameters())
    _assert_signal(model.embed.weight.grad[observed])
    _assert_signal(model.proj.weight.grad)
    for block in model.blocks:
        _assert_signal(block[1].fc.weight.grad)

    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(observed, incoming, temporal_gradient=True)
        F.cross_entropy(logits, targets).backward()
    _assert_signal(incoming.grad)


def test_carry_readout_is_the_head_applied_to_the_carry_as_a_hidden():
    model = _model()
    carry = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = model.carry_readout(carry)
        scaled = model.carry_readout(carry * 8)
        shrunk = model.carry_readout(carry / 16)
    # The model's own forward also runs its norms with autocast disabled.
    with torch.autocast("cuda", enabled=False):
        expected = model.readout(model.final_norm(carry))
    assert actual.dtype == torch.float32 and actual.shape == (4, 32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Only the direction is judged: rescaling the carry earns nothing.
    torch.testing.assert_close(scaled, actual, rtol=2e-2, atol=1e-2)
    torch.testing.assert_close(shrunk, actual, rtol=2e-2, atol=1e-2)


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_carry_readout_scales_head_gradients_but_always_reaches_the_carry(weight):
    model = _model()
    carry = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    targets = torch.tensor([1, 2, 3], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        live = F.cross_entropy(model.carry_readout(carry, 1.0), targets)
    expected = torch.autograd.grad(live, (carry, model.proj.weight, model.proj.bias, model.final_norm.gains))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = F.cross_entropy(model.carry_readout(carry, weight), targets)
    loss.backward()
    _assert_signal(carry.grad)
    torch.testing.assert_close(carry.grad, expected[0], rtol=0, atol=0)
    for parameter, live_gradient in zip((model.proj.weight, model.proj.bias, model.final_norm.gains), expected[1:]):
        if weight == 0.0:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, weight * live_gradient, rtol=1e-6, atol=1e-7)
    # Nothing else is on the readout path.
    assert model.embed.weight.grad is None
    for block in model.blocks:
        assert block[1].fc.weight.grad is None


def test_frozen_head_trains_the_writer_but_not_the_backbone():
    model = _model()
    observed = torch.tensor([2, 3], device="cuda")
    targets = torch.tensor([5, 6], device="cuda")
    state = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, hidden = model(observed, state)
        # Callers hand the writer the detached reset state, as the objective does.
        next_carry, gate = model.writer(hidden.detach(), state.detach())
        loss = F.cross_entropy(model.carry_readout(next_carry), targets)
    assert gate.shape == next_carry.shape == (2, 128)
    assert gate.dtype == next_carry.dtype == torch.bfloat16
    loss.backward()
    _assert_signal(model.writer.write.weight.grad)
    _assert_signal(model.writer.gate_hidden.weight.grad)
    _assert_signal(model.writer.gate_hidden.bias.grad)
    _assert_signal(model.writer.gate_carry.grad)
    assert state.grad is None
    assert model.proj.weight.grad is None and model.embed.weight.grad is None
    assert model.final_norm.gains.grad is None
    for block in model.blocks:
        assert block[1].fc.weight.grad is None


def test_writer_retention_is_set_by_the_gate_alone():
    """Rescaling the write or the old carry cannot change what is retained."""
    model = _model()
    with torch.no_grad():
        model.writer.gate_hidden.weight.normal_(std=0.05)
        model.writer.gate_carry.normal_(std=0.05)
    hidden = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    previous = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        carry, gate = model.writer(hidden, previous)
        with torch.no_grad():
            model.writer.write.weight.mul_(16.0)
        louder_write, louder_gate = model.writer(hidden, previous)
    torch.testing.assert_close(louder_gate, gate, rtol=0, atol=0)
    torch.testing.assert_close(louder_write, carry, rtol=2e-2, atol=2e-2)
    # The gate reads the raw old carry, so rescaling it does move the gate,
    # but the retained direction itself is norm-free.
    unit = lambda vector: F.rms_norm(vector.float(), (128,))
    write = F.linear(hidden.float(), model.writer.write.weight, model.writer.write.bias)
    expected = gate.float() * unit(write) + (1 - gate.float()) * unit(previous)
    torch.testing.assert_close(louder_write.float(), expected, rtol=2e-2, atol=2e-2)


def test_lanes_have_independent_outputs_carries_and_writer_gradients():
    model = _model()
    observed = torch.tensor([2, 3, 4], device="cuda")
    state = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    changed_observed = observed.clone()
    changed_observed[0] = 8
    changed_state = state.clone()
    changed_state[0] = torch.randn_like(changed_state[0])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, hidden, next_carry = model.step(observed, state)
        changed_logits, changed_hidden, changed_next = model.step(changed_observed, changed_state)
        hidden = hidden.detach().requires_grad_(True)
        written, _ = model.writer(hidden, state)
        readout = model.carry_readout(written)
        gradient = torch.autograd.grad(readout[:1].square().sum(), hidden)[0]
    for original, changed in ((logits, changed_logits), (hidden, changed_hidden), (next_carry, changed_next)):
        assert not torch.equal(original[0], changed[0])
        torch.testing.assert_close(original[1:], changed[1:], rtol=0, atol=0)
    _assert_signal(gradient[:1])
    assert torch.count_nonzero(gradient[1:]) == 0
