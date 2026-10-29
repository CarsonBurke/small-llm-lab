"""Tensor-only checks of the binary estimator and prefix oracle."""

from types import SimpleNamespace

import pytest
import torch

from postraining.binary_q import (
    binary_q_delightful_policy_loss,
    exact_prefix_success_constraints,
)
from postraining.core import structural_format_ok


def batch(tokens, reward=0, prompt=(99, 10, 11, 12, 13), pad=2):
    ids = torch.tensor([[*prompt, *tokens, *([99] * pad)]])
    emit = torch.zeros_like(ids, dtype=torch.float32)
    emit[:, len(prompt) - 1:len(prompt) + len(tokens) - 1] = 1
    return SimpleNamespace(token_ids=ids, emit_mask=emit, reward_scalar=torch.tensor([reward]))


def constraints(value, **kwargs):
    return exact_prefix_success_constraints(
        value, think_fence_ids=(10, 11), answer_fence_ids=(12, 13),
        eos_id=14, min_think_tokens=2, **kwargs,
    )


@pytest.mark.parametrize("eta", [0.5, 1.0, 2.0])
def test_whole_coefficient_is_conditional_expectation(eta):
    probabilities = torch.tensor([0.001, 0.1, 0.8])
    q = torch.tensor([0.0, 0.2, 1.0], requires_grad=True)
    logp = probabilities.log().requires_grad_()
    loss, stats = binary_q_delightful_policy_loss(logp, q, torch.ones(3), eta)
    sampled_mean = q.detach() * torch.sigmoid(-logp.detach() / eta)
    torch.testing.assert_close(loss.detach(), -(sampled_mean * logp.detach()).mean())
    loss.backward()
    torch.testing.assert_close(logp.grad, -sampled_mean / 3)
    assert q.grad is None
    torch.testing.assert_close(stats["coefficient_sum"], sampled_mean.sum())


def test_no_second_q_gate_and_no_gradient_through_gate():
    logits = torch.tensor([1., -1.], requires_grad=True)
    q = torch.tensor([.2, .7])
    logp = logits.log_softmax(0)
    loss, _ = binary_q_delightful_policy_loss(logp, q, torch.ones(2), denominator=torch.tensor(1.))
    loss.backward()
    pi = logits.detach().softmax(0)
    weights = q / (1 + pi)
    torch.testing.assert_close(logits.grad, pi * weights.sum() - weights)


@pytest.mark.parametrize("bad", [1e-6, .1, float("nan"), float("inf"), -float("inf")])
def test_invalid_scores_are_diagnosed_without_device_assertion(bad):
    _, stats = binary_q_delightful_policy_loss(torch.tensor([bad]), torch.tensor([.02]), torch.ones(1))
    assert not stats["valid_logprobs"]
    assert stats["valid_q"]


def test_inactive_nan_is_excluded_from_loss_and_gradient():
    logp = torch.tensor([-.5, float("nan")], requires_grad=True)
    loss, stats = binary_q_delightful_policy_loss(logp, torch.tensor([.1, float("nan")]), torch.tensor([1., 0.]))
    loss.backward()
    assert torch.isfinite(loss)
    assert logp.grad[1] == 0
    assert all(torch.isfinite(v) for v in stats.values())


@pytest.mark.parametrize("tokens", [
    [10, 1, 2, 11, 12, 3, 13, 14],
    [10, 1, 2, 11, 4, 5, 12, 3, 13, 14],
])
def test_valid_eos_is_exact_terminal_probability(tokens):
    assert structural_format_ok(tokens, (10, 11), (12, 13), 2)
    for reward in (0, 1):
        b = batch(tokens, reward)
        known, exact = constraints(b)
        active = b.emit_mask.bool()
        assert known[active].tolist() == [False] * (len(tokens) - 1) + [True]
        assert exact[active][-1] == reward


@pytest.mark.parametrize("prefix", [
    [1], [10, 1, 11], [10, 1, 2, 10], [10, 1, 2, 12],
    [10, 1, 2, 11, 13], [10, 1, 2, 11, 12, 13],
    [10, 1, 2, 11, 12, 3, 13, 4], [10, 1, 2, 14],
])
def test_irreparable_prefix_remains_exact_zero(prefix):
    tail = [1, 2, 11, 12, 3, 13, 14]
    b = batch(prefix + tail)
    known, exact = constraints(b)
    start = len([99, 10, 11, 12, 13]) - 1 + len(prefix) - 1
    assert known[0, start:start + len(tail) + 1].all()
    assert not exact.any()
    # The scorer receives a stop-truncated response: appending after EOS is
    # not a permitted continuation, even though its format-only helper does
    # not itself scan interior EOS tokens.
    visible = prefix + tail
    if 14 in visible:
        visible = visible[:visible.index(14) + 1]
    assert not structural_format_ok(visible, (10, 11), (12, 13), 2)


def test_live_prefix_never_reads_future_outcome_or_tokens():
    prefix = [10, 1, 2, 11, 12]
    good = batch(prefix + [3, 13, 14], 1)
    bad = batch(prefix + [13, 4, 14], 0)
    kg, qg = constraints(good)
    kb, qb = constraints(bad)
    through = len([99, 10, 11, 12, 13]) - 1 + len(prefix)
    torch.testing.assert_close(kg[:, :through], kb[:, :through])
    torch.testing.assert_close(qg[:, :through], qb[:, :through])


def test_generation_budget_is_exact_zero_without_eos():
    b = batch([10, 1, 2])
    known, exact = constraints(b, max_new_tokens=3)
    assert known[b.emit_mask.bool()].tolist() == [False, False, True]
    assert not exact.any()


def test_every_actual_stop_id_is_a_valid_terminal():
    b = batch([10, 1, 2, 11, 12, 3, 13, 15], 1)
    known, exact = constraints(b, stop_ids=(14, 15))
    assert known[b.emit_mask.bool()].tolist() == [False] * 7 + [True]
    assert exact[b.emit_mask.bool()][-1] == 1


def test_oracle_rejects_reward_contract_mismatch():
    with pytest.raises(RuntimeError, match="binary"):
        constraints(batch([10, 1, 2, 11, 12, 3, 13, 14], .5))
    with pytest.raises(RuntimeError, match="contradicts"):
        constraints(batch([10, 1, 11, 12, 3, 13, 14], 1))
