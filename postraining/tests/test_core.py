from __future__ import annotations

import torch

from postraining.core import (
    clipped_policy_loss,
    generalized_advantage_estimate,
    length_adaptive_lambda,
    normalize_final_answer,
    positive_example_lm_loss,
    verify_answer,
)


def test_dapo_answer_normalization_and_extraction():
    assert normalize_final_answer(r"$\boxed{1,234\text{ minutes}}$") == "1234"
    assert verify_answer("work\nAnswer: $540$", "540") == (True, "540")
    assert verify_answer("540", "540")[0] is False


def test_length_adaptive_lambda_is_clamped():
    actual = length_adaptive_lambda(torch.tensor([1, 20, 100]))
    torch.testing.assert_close(actual, torch.tensor([0.0, 0.0, 0.8]))


def test_lambda_one_gae_equals_monte_carlo_for_terminal_reward():
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.tensor([[0.2, 0.3, 0.4]])
    mask = torch.ones_like(rewards)
    advantages, returns = generalized_advantage_estimate(rewards, values, mask, torch.ones(1))
    torch.testing.assert_close(returns, torch.ones_like(returns))
    torch.testing.assert_close(advantages, torch.tensor([[0.8, 0.7, 0.6]]))


def test_negative_padded_monte_carlo_returns():
    rewards = torch.tensor([[0.0, -1.0, 0.0]])
    values = torch.tensor([[0.2, 0.3, 9.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    _, returns = generalized_advantage_estimate(rewards, values, mask, torch.ones(1))
    torch.testing.assert_close(returns, torch.tensor([[-1.0, -1.0, 9.0]]))


def test_positive_lm_loss_weights_correct_trajectories_equally():
    logprobs = torch.tensor([[-2.0, -2.0, 0.0], [-1.0, -3.0, -5.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    loss = positive_example_lm_loss(logprobs, mask, torch.tensor([True, True]))
    torch.testing.assert_close(loss, torch.tensor(2.5))


def test_official_verifier_edge_normalization():
    assert normalize_final_answer(r"42\text{ minutes}") == "42"
    assert normalize_final_answer("7 childrentickets") == "7"


def test_asymmetric_clipping_uses_token_mean():
    old = torch.zeros(1, 2)
    new = torch.log(torch.tensor([[1.5, 0.5]]))
    advantages = torch.tensor([[1.0, -1.0]])
    loss, fraction = clipped_policy_loss(new, old, advantages, torch.ones_like(old))
    # Positive advantages clip at 1.28; negative advantages clip at 0.80.
    torch.testing.assert_close(loss, torch.tensor(-0.24))
    torch.testing.assert_close(fraction, torch.tensor(1.0))
