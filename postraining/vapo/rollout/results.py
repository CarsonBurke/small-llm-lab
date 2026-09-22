"""Rollout results and response budgets, shared by every rollout backend.

Nothing here knows how a model decodes. A backend hands back completed rows
in stable prompt-major order; the trainer turns them into replay records.
Keeping this currency out of any one engine is what lets a nano backbone and
a Hugging Face model feed the same VAPO update.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor

from postraining.thinking_budget import validate_thinking_budget


@dataclass(frozen=True)
class FastTrainingDecodeStats:
    prefill_seconds: float
    decode_seconds: float
    target_decode_calls: int
    target_decode_positions: int


@dataclass(frozen=True)
class ContinuousTrainingGeneration:
    """Completed rows in stable prompt-major/sample-major order.

    ``logprobs`` contains shape-compatible zeros because training refreshes
    exact behavior statistics from replay before PPO.
    ``carry_hiddens`` owns the CPU BF16 producer of every response token in
    token-carry mode when history recording is enabled, including the prompt
    producer and terminal action. Otherwise it is ``None``.
    """

    responses: tuple[Tensor, ...]
    logprobs: tuple[Tensor, ...]
    prefill_seconds: float
    decode_seconds: float
    decode_steps: int
    useful_tokens: int
    capacity_row_steps: int
    admission_events: int
    minimum_active_rows_with_backlog: int
    carry_hiddens: tuple[Tensor, ...] | None = None
    response_limits: tuple[int, ...] = ()
    slot_choices: tuple[Tensor, ...] | None = None

    @property
    def productive_utilization(self) -> float:
        if not self.capacity_row_steps:
            return 0.0
        return self.useful_tokens / self.capacity_row_steps

def response_token_limits(
    prompt_lengths: Sequence[int],
    *,
    max_new_tokens: int,
    context_tokens: int | None = None,
    answer_reserve_tokens: int = 0,
    thinking_end_token_id: int | None = None,
) -> tuple[int, ...]:
    """Resolve output budgets from full, untruncated chat prompt lengths."""
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if context_tokens is not None and (
        type(context_tokens) is not int or context_tokens < 1
    ):
        raise ValueError("context_tokens must be a positive integer")
    limits = []
    for length in prompt_lengths:
        if type(length) is not int or length < 1:
            raise ValueError("full chat prompt lengths must be positive integers")
        limit = (
            max_new_tokens if context_tokens is None
            else min(max_new_tokens, context_tokens - length)
        )
        if limit < 1:
            raise ValueError("full chat prompt leaves no response within context_tokens")
        validate_thinking_budget(
            answer_reserve_tokens, thinking_end_token_id, limit
        )
        limits.append(limit)
    return tuple(limits)
