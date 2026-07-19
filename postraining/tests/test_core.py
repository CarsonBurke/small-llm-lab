from __future__ import annotations

import pytest
import torch

from postraining.core import (
    POSTTRAIN_CONTEXT_TOKENS,
    POSTTRAIN_PROMPT_TOKENS,
    POSTTRAIN_RESPONSE_TOKENS,
    POSTTRAIN_STREAM_TOKENS,
    clipped_policy_loss,
    encode_prompt,
    generalized_advantage_estimate,
    length_adaptive_lambda,
    normalize_final_answer,
    per_dim_clipped_policy_loss,
    positive_example_lm_loss,
    validate_posttraining_context_budget,
    verify_answer,
)


class _Tokenizer:
    def __init__(self, bos: int):
        self.bos = bos

    def bos_id(self) -> int:
        return self.bos

    def encode(self, text: str) -> list[int]:
        return list(range(10, 10 + len(text)))


def test_encode_prompt_frames_the_prompt_as_a_document_start():
    # Pretraining shards every document as [BOS] tokens, so prompts must be
    # BOS-first; truncation keeps the BOS plus the LAST content tokens.
    assert encode_prompt(_Tokenizer(bos=1), "abcde") == [1, 10, 11, 12, 13, 14]
    assert encode_prompt(_Tokenizer(bos=1), "abcde", 4) == [1, 12, 13, 14]
    assert encode_prompt(_Tokenizer(bos=-1), "abcde", 4) == [11, 12, 13, 14]
    assert encode_prompt(_Tokenizer(bos=-1), "abcde") == [10, 11, 12, 13, 14]


def test_posttraining_context_contract():
    assert POSTTRAIN_PROMPT_TOKENS == 1024
    assert POSTTRAIN_RESPONSE_TOKENS == 1024
    assert POSTTRAIN_STREAM_TOKENS == 4096
    assert POSTTRAIN_PROMPT_TOKENS + POSTTRAIN_STREAM_TOKENS == POSTTRAIN_CONTEXT_TOKENS
    validate_posttraining_context_budget(
        POSTTRAIN_PROMPT_TOKENS, POSTTRAIN_STREAM_TOKENS
    )


def test_posttraining_context_rejects_overflow_and_nonpositive_budgets():
    for prompt, stream in ((1024, 4097), (0, 4096), (1024, 0)):
        with pytest.raises(ValueError):
            validate_posttraining_context_budget(prompt, stream)


def test_dapo_answer_normalization_and_extraction():
    assert normalize_final_answer(r"$\boxed{1,234\text{ minutes}}$") == "1234"
    assert verify_answer("work\nAnswer: $540$", "540") == (True, "540")
    assert verify_answer("540", "540")[0] is False


def test_length_adaptive_lambda_floors_the_credit_horizon():
    # Horizon = max(alpha*l, min(l, 1/alpha)): whole-trajectory credit for
    # short responses (raw VAPO would clamp lambda to 0 there), the fixed
    # lambda = 1 - alpha baseline in the middle, VAPO's alpha*l for long.
    actual = length_adaptive_lambda(torch.tensor([1, 5, 20, 100, 2000]))
    torch.testing.assert_close(
        actual, torch.tensor([0.0, 0.8, 0.95, 0.95, 0.99])
    )


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


def test_per_dim_clip_reduces_to_the_scalar_clip_at_one_dimension():
    old = torch.zeros(1, 2, 1)
    new = torch.log(torch.tensor([[[1.5], [0.5]]]))
    advantages = torch.tensor([[1.0, -1.0]])
    mask = torch.ones(1, 2)
    loss, fraction = per_dim_clipped_policy_loss(new, old, advantages, mask)
    torch.testing.assert_close(loss, torch.tensor(-0.24))
    torch.testing.assert_close(fraction, torch.tensor(1.0))


def test_per_dim_clip_bounds_each_coordinate_independently():
    # One position, two dims: one inside the trust region, one clipped high.
    # A joint ratio would clip (or not) both together; the factored form
    # must clip exactly the offending coordinate.
    old = torch.zeros(1, 1, 2)
    new = torch.log(torch.tensor([[[1.1, 2.0]]]))
    advantages = torch.tensor([[1.0]])
    mask = torch.ones(1, 1)
    loss, fraction = per_dim_clipped_policy_loss(new, old, advantages, mask)
    torch.testing.assert_close(loss, torch.tensor(-(1.1 + 1.28) / 2))
    torch.testing.assert_close(fraction, torch.tensor(0.5))
