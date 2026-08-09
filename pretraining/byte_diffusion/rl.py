"""Byte-action policy optimization primitives for sampler post-training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RewardWeights:
    nfe: float = 0.0
    length: float = 0.0
    invalid_utf8: float = 0.0
    missing_eot: float = 0.0
    repetition: float = 0.0


def shaped_reward(
    task_reward: Tensor,
    *,
    nfe: Tensor,
    length: Tensor,
    invalid_utf8: Tensor,
    missing_eot: Tensor,
    repetition: Tensor,
    weights: RewardWeights,
) -> Tensor:
    terms = (nfe, length, invalid_utf8, missing_eot, repetition)
    if any(term.shape != task_reward.shape for term in terms):
        raise ValueError("all reward features must align")
    return (
        task_reward
        - weights.nfe * nfe
        - weights.length * length
        - weights.invalid_utf8 * invalid_utf8
        - weights.missing_eot * missing_eot
        - weights.repetition * repetition
    )


def within_prompt_advantages(rewards: Tensor, prompt_ids: Tensor) -> Tensor:
    if rewards.ndim != 1 or prompt_ids.shape != rewards.shape:
        raise ValueError("one prompt id is required per scalar reward")
    advantages = torch.empty_like(rewards)
    for prompt_id in prompt_ids.unique():
        group = prompt_ids == prompt_id
        advantages[group] = rewards[group] - rewards[group].mean()
    return advantages


def byte_policy_loss(
    action_log_prob: Tensor,
    advantages: Tensor,
    categorical_action: Tensor,
    *,
    old_log_prob: Tensor | None = None,
    clip_ratio: float | None = None,
) -> Tensor:
    """Policy loss over recorded categorical byte actions only."""

    if action_log_prob.shape != advantages.shape or categorical_action.shape != advantages.shape:
        raise ValueError("policy tensors must align")
    if categorical_action.dtype != torch.bool or not bool(categorical_action.any()):
        raise ValueError("at least one categorical byte action is required")
    selected_advantage = advantages.detach()[categorical_action]
    if old_log_prob is None:
        objective = action_log_prob[categorical_action] * selected_advantage
    else:
        if old_log_prob.shape != action_log_prob.shape or clip_ratio is None or clip_ratio <= 0:
            raise ValueError("PPO requires aligned old log-probs and positive clip ratio")
        ratio = (action_log_prob[categorical_action] - old_log_prob[categorical_action]).exp()
        clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio)
        objective = torch.minimum(ratio * selected_advantage, clipped * selected_advantage)
    return -objective.sum() / categorical_action.sum()

