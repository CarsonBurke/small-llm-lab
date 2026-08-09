from __future__ import annotations

import torch

from pretraining.byte_diffusion.distillation import trajectory_distillation_loss
from pretraining.byte_diffusion.rl import (
    RewardWeights,
    byte_policy_loss,
    shaped_reward,
    within_prompt_advantages,
)


def test_distillation_detaches_teacher_and_final_ids() -> None:
    student = torch.randn(1, 3, 5, requires_grad=True)
    teacher = torch.randn(1, 3, 5, requires_grad=True)
    targets = torch.tensor([[1, 2, 3]])
    selected = torch.tensor([[True, False, True]])
    ar = student.sum() * 0.0 + 0.25
    terms = trajectory_distillation_loss(student, teacher, targets, selected, ar_loss=ar)
    terms.total.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_within_prompt_advantages_have_zero_group_mean() -> None:
    rewards = torch.tensor([1.0, 3.0, 2.0, 8.0, 5.0])
    prompts = torch.tensor([0, 0, 1, 1, 1])
    advantages = within_prompt_advantages(rewards, prompts)
    for prompt in prompts.unique():
        torch.testing.assert_close(advantages[prompts == prompt].mean(), torch.tensor(0.0))


def test_reward_and_policy_include_only_categorical_bytes() -> None:
    reward = shaped_reward(
        torch.tensor([10.0]),
        nfe=torch.tensor([2.0]),
        length=torch.tensor([4.0]),
        invalid_utf8=torch.tensor([1.0]),
        missing_eot=torch.tensor([0.0]),
        repetition=torch.tensor([0.5]),
        weights=RewardWeights(nfe=1, length=0.5, invalid_utf8=3, repetition=2),
    )
    torch.testing.assert_close(reward, torch.tensor([2.0]))
    log_prob = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    loss = byte_policy_loss(
        log_prob,
        torch.tensor([1.0, 100.0, -1.0]),
        torch.tensor([True, False, True]),
    )
    loss.backward()
    assert log_prob.grad.tolist() == [-0.5, 0.0, 0.5]

