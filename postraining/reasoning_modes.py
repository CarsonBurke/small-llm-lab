"""Shared rollout-budget semantics for latent, CoT, and answer-only policies."""

from __future__ import annotations

from collections.abc import Mapping


def mode_rollout_budget(
    reasoning_mode: str,
    max_tokens: int,
    *,
    answer_tokens: int,
) -> tuple[int, int]:
    """Return ``(emitted-token cap, stream-slot cap)`` for one evaluation.

    Every action is a token under the deterministic hidden carry — thinking
    rides inside the carried belief rather than occupying stream slots — so
    the stream cap always equals the emitted-token cap in every mode.
    """
    if reasoning_mode == "none":
        return answer_tokens, answer_tokens
    if reasoning_mode == "cot":
        return max_tokens, max_tokens
    if reasoning_mode != "latent":
        raise ValueError(f"unknown reasoning mode {reasoning_mode!r}")
    return max_tokens, max_tokens


def training_rollout_budget(
    reasoning_mode: str,
    continuation_tokens: int,
    *,
    answer_tokens: int,
) -> tuple[int, int]:
    """Resolve the training budget for one rollout pool."""
    return mode_rollout_budget(
        reasoning_mode,
        continuation_tokens,
        answer_tokens=answer_tokens,
    )


def checkpoint_training_rollout_budget(
    saved_args: Mapping[str, object],
) -> tuple[int, int]:
    """Resolve a saved checkpoint's effective training rollout budget."""
    if "resolved_train_max_stream_steps" in saved_args:
        return (
            int(saved_args["resolved_train_max_new_tokens"]),
            int(saved_args["resolved_train_max_stream_steps"]),
        )
    return training_rollout_budget(
        str(saved_args.get("reasoning_mode", "latent")),
        int(saved_args.get("continuation_tokens", 1024)),
        answer_tokens=int(saved_args.get("answer_tokens", 24)),
    )
