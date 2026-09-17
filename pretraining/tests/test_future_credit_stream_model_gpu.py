"""CUDA/BF16 scalar-value and detached recurrence contracts; execute through mlq."""

import pytest
import torch
import torch.nn.functional as F

from pretraining.future_credit_stream.model import StreamingFFNModel


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _model(*, use_td_critic=True, activate_residuals=True):
    torch.manual_seed(31)
    model = StreamingFFNModel(
        vocab_size=32, model_dim=128, num_layers=3, mlp_hidden=256,
        use_td_critic=use_td_critic, critic_hidden=64,
    ).cuda()
    if activate_residuals:
        # Nonzero trained-state readout, residuals, and value head expose gradients.
        with torch.no_grad():
            model.proj.weight.normal_(std=0.02)
            model.proj.bias.normal_(std=0.02)
            for block in model.blocks:
                block[1].proj.weight.normal_(std=0.02)
            if model.critic is not None:
                model.critic.proj.weight.normal_(std=0.03)
    return model


def _assert_signal(gradient):
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_optional_critic_preserves_common_initialization_and_starts_at_zero():
    plain = _model(use_td_critic=False, activate_residuals=False)
    model = _model(activate_residuals=False)
    plain_parameters = dict(plain.named_parameters())
    parameters = dict(model.named_parameters())
    assert plain.critic is None
    assert set(plain_parameters) == {
        name for name in parameters if not name.startswith("critic.")
    }
    for name, parameter in plain_parameters.items():
        torch.testing.assert_close(parameter, parameters[name], rtol=0, atol=0)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert model.embed.weight.data_ptr() != model.proj.weight.data_ptr()

    observed = torch.tensor([2, 3, 4], device="cuda")
    carry = plain.initial_state(3, "cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        plain_logits, plain_hidden = plain(observed, carry)
        logits, hidden = model(observed, carry)
        value = model.value(hidden)
        unrelated_value = model.value(torch.randn_like(hidden))
    assert logits.dtype == value.dtype == torch.float32
    assert hidden.dtype == torch.bfloat16 and value.shape == (3,)
    torch.testing.assert_close(logits, plain_logits, rtol=0, atol=0)
    torch.testing.assert_close(hidden, plain_hidden, rtol=0, atol=0)
    assert torch.count_nonzero(value) == torch.count_nonzero(unrelated_value) == 0


def test_forward_uses_actual_carry_without_temporal_gradient():
    model = _model()
    incoming = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    first_tokens = torch.tensor([2, 3], device="cuda")
    observed = torch.tensor([4, 5], device="cuda")
    targets = torch.tensor([6, 7], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, previous_hidden = model(first_tokens, incoming)
        previous_hidden.retain_grad()
        logits, _ = model(observed, previous_hidden)
        reset_logits, _ = model(observed, model.initial_state(2, "cuda"))
        loss = F.cross_entropy(logits, targets)
    loss.backward()
    assert incoming.grad is previous_hidden.grad is None
    assert not torch.equal(logits, reset_logits)
    assert all(parameter.grad is None for parameter in model.critic.parameters())
    _assert_signal(model.embed.weight.grad[observed])
    assert torch.count_nonzero(model.embed.weight.grad[first_tokens]) == 0
    _assert_signal(model.proj.weight.grad)
    for block in model.blocks:
        _assert_signal(block[1].fc.weight.grad)
        _assert_signal(block[1].proj.weight.grad)


def test_producer_value_updates_current_hidden_and_core_but_not_critic_or_readout():
    model = _model()
    observed = torch.tensor([2, 3], device="cuda")
    state = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, hidden = model(observed, state)
        producer_value = model.value(hidden, detach_weights=True)
        hidden.retain_grad()
        loss = 0.7 * producer_value.mean()
    loss.backward()
    _assert_signal(hidden.grad)
    _assert_signal(model.embed.weight.grad[observed])
    for block in model.blocks:
        _assert_signal(block[1].fc.weight.grad)
        _assert_signal(block[1].proj.weight.grad)
    assert state.grad is None
    assert all(parameter.grad is None for parameter in model.critic.parameters())
    assert all(parameter.grad is None for parameter in model.proj.parameters())


def test_ordinary_value_keeps_input_gradient_and_can_fit_detached_features():
    model = _model()
    hidden = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        value = model.value(hidden)
        frozen_value = model.value(hidden, detach_weights=True)
    ordinary_gradient = torch.autograd.grad(value.sum(), hidden)[0]
    frozen_gradient = torch.autograd.grad(frozen_value.sum(), hidden)[0]
    _assert_signal(ordinary_gradient)
    torch.testing.assert_close(ordinary_gradient, frozen_gradient, rtol=0, atol=0)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model.value(hidden.detach())
        regression = 0.5 * (prediction - 3).square().mean()
    regression.backward()
    assert hidden.grad is None
    for name, parameter in model.named_parameters():
        if name.startswith("critic."):
            _assert_signal(parameter.grad)
        else:
            assert parameter.grad is None, name


def test_scalar_accumulation_preserves_td_residual_below_bf16_value_spacing():
    model = _model()
    with torch.no_grad():
        model.critic.fc.weight.zero_()
        model.critic.fc.bias.fill_(1)
        model.critic.proj.weight.zero_()
        model.critic.proj.weight[0, 0] = 0.125
        model.critic.proj.bias.fill_(1024)
    hidden = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        value = model.value(hidden)
        loss = 0.5 * (value - 1024).square().mean()
    torch.testing.assert_close(value, torch.full((2,), 1024.125, device="cuda"), rtol=0, atol=0)
    loss.backward()
    torch.testing.assert_close(model.critic.proj.bias.grad,
                               torch.tensor([0.125], device="cuda"), rtol=0, atol=0)


def test_lanes_have_independent_outputs_values_and_actor_gradients():
    model = _model()
    observed = torch.tensor([2, 3, 4], device="cuda")
    state = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    changed_observed = observed.clone()
    changed_observed[0] = 8
    changed_state = state.clone()
    changed_state[0] = torch.randn_like(changed_state[0])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, hidden = model(observed, state)
        changed_logits, changed_hidden = model(changed_observed, changed_state)
        value = model.value(hidden, detach_weights=True)
        changed_value = model.value(changed_hidden, detach_weights=True)
        gradient = torch.autograd.grad(value[:1].sum(), hidden)[0]
    for original, changed in ((logits, changed_logits), (hidden, changed_hidden), (value, changed_value)):
        assert not torch.equal(original[0], changed[0])
        torch.testing.assert_close(original[1:], changed[1:], rtol=0, atol=0)
    _assert_signal(gradient[:1])
    assert torch.count_nonzero(gradient[1:]) == 0
