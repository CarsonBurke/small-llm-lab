from __future__ import annotations

import torch
from torch import Tensor


def validate_thinking_budget(
    answer_reserve_tokens: int,
    thinking_end_token_id: int | None,
    max_new_tokens: int | None = None,
) -> None:
    if answer_reserve_tokens < 0:
        raise ValueError("answer_reserve_tokens must be nonnegative")
    if not answer_reserve_tokens:
        return
    if thinking_end_token_id is None or thinking_end_token_id < 0:
        raise ValueError("answer reserve requires a native thinking end token id")
    if max_new_tokens is not None and max_new_tokens <= answer_reserve_tokens + 1:
        raise ValueError("max_new_tokens must exceed answer_reserve_tokens + 1")


def force_thinking_end_(
    tokens: Tensor,
    output_position: int | Tensor,
    thinking_closed: Tensor,
    active: Tensor,
    boundary: int | Tensor,
    thinking_end_token_id: int,
) -> tuple[Tensor, Tensor]:
    """Finalize an emitted token and bounded lane state without host synchronization.

    Speculative callers must discard any verified suffix following a forced
    token: that suffix was conditioned on the original, unforced proposal.
    """
    forced = active & ~thinking_closed & (output_position == boundary)
    tokens = torch.where(forced, thinking_end_token_id, tokens)
    thinking_closed.logical_or_(active & (tokens == thinking_end_token_id))
    return tokens, forced
