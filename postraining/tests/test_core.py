from __future__ import annotations

import json

import pytest
import torch

from postraining.core import (
    POSTTRAIN_CONTEXT_TOKENS,
    POSTTRAIN_PROMPT_TOKENS,
    POSTTRAIN_RESPONSE_TOKENS,
    POSTTRAIN_STREAM_TOKENS,
    JsonlLogger,
    answer_style,
    clipped_policy_loss,
    encode_prompt,
    generalized_advantage_estimate,
    length_adaptive_lambda,
    module_answer_baselines,
    normalize_final_answer,
    positive_example_lm_loss,
    validate_posttraining_context_budget,
    verify_answer,
)


def test_jsonl_logger_purges_only_stale_value_warmup(tmp_path):
    logger = JsonlLogger(tmp_path / "metrics.jsonl")
    logger.log(type="bench_guess_baseline")
    logger.log(type="value_warmup", step=20)
    logger.log(type="value_warmup", step=21)
    logger.log(type="train", step=21)

    assert logger.purge_value_warmup_after(20) == 1
    assert logger.purge_value_warmup_after(20) == 0
    records = [
        json.loads(line)
        for line in logger.path.read_text(encoding="utf-8").splitlines()
    ]
    assert [(record["type"], record.get("step")) for record in records] == [
        ("bench_guess_baseline", None),
        ("value_warmup", 20),
        ("train", 21),
    ]


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


def test_answer_style_follows_the_row_grading_rule():
    assert answer_style({"reward_model": {"style": "rule"}}) == "exact"
    assert (
        answer_style({"reward_model": {"style": "rule-lighteval/MATH_v2"}})
        == "minerva"
    )
    assert answer_style({"reward_model": {}}) == "minerva"
    assert answer_style({}) == "minerva"


def test_exact_style_grades_the_canonical_answer_string_only():
    # mathematics_dataset's official metric: exact match, no normalization.
    assert verify_answer("work\nAnswer: -1342, -1, 2795", "-1342, -1, 2795", "exact")[0]
    assert verify_answer("work\nAnswer: True", "True", "exact") == (True, "True")
    # Minerva's rewrites collapse a multi-candidate or hedged line onto a
    # single graded token; the exact style must not.
    assert verify_answer("Answer: a b", "b", "exact")[0] is False
    assert verify_answer("Answer: a b", "b", "minerva")[0] is True
    assert verify_answer("Answer: x = 5", "5", "exact")[0] is False
    assert verify_answer("Answer: x = 5", "5", "minerva")[0] is True
    assert verify_answer("Answer: $2$ and $3$", "2", "exact")[0] is False
    assert verify_answer("Answer: 7, 6, 2, 2, 2, 1, 5", "762, 22, 15", "exact")[0] is False
    assert verify_answer("Answer: 7, 6, 2, 2, 2, 1, 5", "762, 22, 15", "minerva")[0] is True
    # Booleans keep the canonical case.
    assert verify_answer("Answer: true", "True", "exact")[0] is False
    # Only the LAST Answer: line is graded — spamming candidates never
    # multiplies chances.
    assert verify_answer("Answer: 3\nAnswer: 4", "3", "exact")[0] is False
    assert verify_answer("Answer: 3\nAnswer: 4", "4", "exact")[0] is True
    # Unlike the Minerva tail window, the whole emission is searched.
    filler = "x" * 400
    assert verify_answer(f"Answer: True\n{filler}", "True", "exact")[0] is True
    assert verify_answer("no final line", "True", "exact") == (False, "[INVALID]")


def test_aime_style_requires_an_integer_in_range():
    assert verify_answer("work\nAnswer: 540", "540", "aime")[0] is True
    assert verify_answer("work\nAnswer: $540$", "540", "aime")[0] is True
    assert verify_answer("work\nAnswer: 540.", "540", "aime")[0] is True
    # Rejected: non-integers, out-of-range values, and normalization
    # coincidences that the Minerva style would accept.
    assert verify_answer("Answer: 5,40", "540", "aime")[0] is False
    assert verify_answer("Answer: 5,40", "540", "minerva")[0] is True
    assert verify_answer("Answer: 1000", "1000", "aime")[0] is False
    assert verify_answer("Answer: x = 540", "540", "aime")[0] is False
    assert verify_answer("no final line", "540", "aime") == (False, "[INVALID]")


def test_verify_answer_rejects_unknown_styles():
    with pytest.raises(ValueError):
        verify_answer("Answer: 1", "1", "sympy")


def test_module_answer_baselines_reports_modal_share():
    rows = [
        {"reward_model": {"ground_truth": t}, "extra_info": {"module": m}}
        for m, t in [
            ("comparison__pair", "True"),
            ("comparison__pair", "True"),
            ("comparison__pair", "False"),
            ("numbers__gcd", "3"),
        ]
    ] + [{"reward_model": {"ground_truth": "540"}, "extra_info": {}}]
    baselines = module_answer_baselines(rows)
    assert set(baselines) == {"comparison__pair", "numbers__gcd"}
    assert baselines["comparison__pair"] == {"rows": 3.0, "modal_share": 2 / 3}
    assert baselines["numbers__gcd"] == {"rows": 1.0, "modal_share": 1.0}
    assert module_answer_baselines([]) == {}


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


def test_positive_lm_loss_weights_correct_tokens_equally():
    logprobs = torch.tensor([[-2.0, -2.0, 0.0], [-1.0, -3.0, -5.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    loss = positive_example_lm_loss(logprobs, mask, torch.tensor([True, True]))
    torch.testing.assert_close(loss, torch.tensor(2.6))


def test_positive_lm_loss_shards_use_the_global_token_denominator():
    logprobs = torch.tensor([[-2.0, -2.0, 0.0], [-1.0, -3.0, -5.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    correct = torch.tensor([True, True])
    denominator = (mask * correct[:, None]).sum()
    sharded = sum(
        positive_example_lm_loss(
            logprobs[index : index + 1],
            mask[index : index + 1],
            correct[index : index + 1],
            denominator=denominator,
        )
        for index in range(2)
    )
    torch.testing.assert_close(sharded, torch.tensor(2.6))


def test_positive_lm_loss_is_differentiable_zero_without_positive_tokens():
    logprobs = torch.tensor([[-2.0, -3.0]], requires_grad=True)
    loss = positive_example_lm_loss(
        logprobs,
        torch.zeros_like(logprobs),
        torch.tensor([True]),
        denominator=torch.tensor(0.0),
    )
    assert loss == 0
    loss.backward()
    torch.testing.assert_close(logprobs.grad, torch.zeros_like(logprobs))


def test_official_verifier_edge_normalization():
    assert normalize_final_answer(r"42\text{ minutes}") == "42"
    assert normalize_final_answer("7 childrentickets") == "7"


def test_asymmetric_clipping_uses_token_mean():
    old = torch.zeros(1, 2)
    new = torch.log(torch.tensor([[1.5, 0.5]]))
    advantages = torch.tensor([[1.0, -1.0]])
    loss, fraction, approximate_kl = clipped_policy_loss(
        new, old, advantages, torch.ones_like(old)
    )
    # Positive advantages clip at 1.28; negative advantages clip at 0.80.
    torch.testing.assert_close(loss, torch.tensor(-0.24))
    torch.testing.assert_close(fraction, torch.tensor(1.0))
    log_ratio = new - old
    expected_kl = (torch.expm1(log_ratio) - log_ratio).mean()
    torch.testing.assert_close(approximate_kl, expected_kl)


def test_factorized_action_is_clipped_once_using_its_joint_ratio():
    # Both factors are individually inside the upper clip, but their product
    # is not. One composite action must therefore be clipped as a whole.
    old_factors = torch.zeros(1, 1, 2)
    new_factors = torch.log(torch.tensor([[[1.2, 1.2]]]))
    old = old_factors.sum(-1)
    new = new_factors.sum(-1)
    loss, fraction, approximate_kl = clipped_policy_loss(
        new, old, torch.ones_like(new), torch.ones_like(new)
    )
    torch.testing.assert_close(loss, torch.tensor(-1.28))
    torch.testing.assert_close(fraction, torch.tensor(1.0))
    log_ratio = new - old
    torch.testing.assert_close(
        approximate_kl, (torch.expm1(log_ratio) - log_ratio).mean()
    )


def test_negative_advantage_high_ratio_remains_unclipped():
    # This is PPO's deliberately harmful, corrective branch: when a sampled
    # action has negative advantage but the new policy made it more likely,
    # min(r*A, clip(r)*A) uses r*A without an upper clip. Do not silently
    # clamp this to make high-dimensional joint ratios numerically convenient.
    new = torch.tensor([[2.0]], requires_grad=True)
    loss, fraction, _ = clipped_policy_loss(
        new,
        torch.zeros_like(new),
        -torch.ones_like(new),
        torch.ones_like(new),
    )
    expected = new.exp().squeeze()
    torch.testing.assert_close(loss, expected)
    assert fraction == 1
    loss.backward()
    torch.testing.assert_close(new.grad, new.detach().exp())


def test_joint_action_score_sums_factor_gradients_without_dimensional_mean():
    factors = torch.zeros(1, 1, 2, requires_grad=True)
    joint = factors.sum(-1)
    loss = clipped_policy_loss(
        joint,
        torch.zeros_like(joint),
        torch.ones_like(joint),
        torch.ones_like(joint),
    )[0]
    loss.backward()
    torch.testing.assert_close(factors.grad, -torch.ones_like(factors))


def test_joint_action_ratio_allows_factor_drift_to_cancel():
    new_factors = torch.log(torch.tensor([[[1.2, 1.0 / 1.2]]]))
    new = new_factors.sum(-1)
    loss, fraction, _ = clipped_policy_loss(
        new,
        torch.zeros_like(new),
        torch.ones_like(new),
        torch.ones_like(new),
    )
    torch.testing.assert_close(loss, torch.tensor(-1.0))
    assert fraction == 0


def test_policy_loss_shards_use_the_global_action_denominator():
    old = torch.zeros(1, 2)
    new = torch.log(torch.tensor([[1.1, 0.9]]))
    advantages = torch.tensor([[2.0, -1.0]])
    mask = torch.ones_like(old)
    full = clipped_policy_loss(new, old, advantages, mask)[0]
    denominator = mask.sum()
    sharded = sum(
        clipped_policy_loss(
            new[:, index : index + 1],
            old[:, index : index + 1],
            advantages[:, index : index + 1],
            mask[:, index : index + 1],
            denominator=denominator,
        )[0]
        for index in range(2)
    )
    torch.testing.assert_close(sharded, full)


def test_behavior_kl_is_stable_and_nonnegative_near_zero_drift():
    old = torch.zeros(1, 2)
    new = torch.tensor([[1e-6, -1e-6]])
    _, _, approximate_kl = clipped_policy_loss(
        new, old, torch.ones_like(old), torch.ones_like(old)
    )
    assert approximate_kl >= 0
    torch.testing.assert_close(
        approximate_kl,
        torch.tensor(0.5e-12),
        rtol=0.15,
        atol=1e-16,
    )
