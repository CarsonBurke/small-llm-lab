"""Shared rollout-budget semantics for latent, CoT, and answer-only policies."""

from __future__ import annotations

from collections.abc import Mapping

from postraining.core import POSTTRAIN_STREAM_TOKENS


def mode_rollout_budget(
    reasoning_mode: str,
    max_tokens: int,
    *,
    answer_tokens: int,
    prompt_tokens: int,
    context_tokens: int,
) -> tuple[int, int]:
    """Return ``(emitted-token cap, stream-slot cap)`` for one evaluation."""
    if reasoning_mode == "none":
        return answer_tokens, answer_tokens
    if reasoning_mode == "cot":
        return max_tokens, max_tokens
    if reasoning_mode != "latent":
        raise ValueError(f"unknown reasoning mode {reasoning_mode!r}")
    return max_tokens, min(
        4 * max_tokens, context_tokens - prompt_tokens
    )


def training_rollout_budget(
    reasoning_mode: str,
    continuation_tokens: int,
    *,
    answer_tokens: int,
    prompt_tokens: int,
    context_tokens: int,
    max_stream_steps: int | None,
) -> tuple[int, int]:
    """Resolve the training budget, including the latent explicit override."""
    if reasoning_mode != "latent":
        if max_stream_steps is not None:
            raise ValueError(
                "max_stream_steps is only valid for latent reasoning"
            )
        return mode_rollout_budget(
            reasoning_mode,
            continuation_tokens,
            answer_tokens=answer_tokens,
            prompt_tokens=prompt_tokens,
            context_tokens=context_tokens,
        )

    if max_stream_steps is None:
        stream_steps = min(
            POSTTRAIN_STREAM_TOKENS, context_tokens - prompt_tokens
        )
    elif max_stream_steps == 0:
        _, stream_steps = mode_rollout_budget(
            reasoning_mode,
            continuation_tokens,
            answer_tokens=answer_tokens,
            prompt_tokens=prompt_tokens,
            context_tokens=context_tokens,
        )
    else:
        stream_steps = max_stream_steps
    return continuation_tokens, stream_steps


def checkpoint_training_rollout_budget(
    saved_args: Mapping[str, object],
    *,
    context_tokens: int,
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
        prompt_tokens=int(saved_args.get("prompt_tokens", 512)),
        context_tokens=context_tokens,
        max_stream_steps=(
            None
            if saved_args.get("max_stream_steps") is None
            else int(saved_args["max_stream_steps"])
        ),
    )
