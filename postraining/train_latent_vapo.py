"""Latent-thought VAPO: full-model PPO over THINK/EMIT gates, tokens, thoughts.

The WHOLE deployed policy path trains at RL time — no frozen trunk. A fresh
linear head owns the Gaussian thought mean instead of reusing pretraining's
next-token latent predictor, so reward can shape a scratch representation
without inheriting that discrete-token target.

- PPO trains everything through one differentiable teacher-forced replay.
  Gate and content log probabilities form one joint action probability at
  each stream position. EMIT clips its joint gate+token ratio. THINK clips
  each diagonal-Gaussian dimension separately while retaining the summed
  joint-score gradient; optional THINK clips its shared gate once as a
  separate factor.
- The critic is a SEPARATE from-scratch model (same architecture class,
  fresh weights, fully trainable, no SIGReg or latent prediction) trained
  purely by HL-Gauss cross-entropy on [0, 1] value targets.  It values
  every stream position — vocab AND latent — which is what gives thinking
  its training signal.
- An optional Bernoulli gate-entropy bonus can preserve THINK exploration;
  there is no continuous-policy entropy bonus, KL penalty, or beta-NLL.
  A zero-initialized belief-conditioned head learns diagonal per-dimension
  thought log-sigma, starting at -2 in every dimension.
- No pretraining anchor: SIGReg and the latent target-prediction objective are
  dropped at RL time. The fresh mean and recurrent policy train purely on
  their ability to think; the teacher-forced val-BPB guard is the drift
  detector.

Prompts are DAPO-Math-17K. An EOS-terminated, verifier-correct final Answer:
field receives reward 1; a wrong but strictly numeric final field receives
bounded distance shaping of at most 0.1. Malformed or unterminated responses
receive zero. The auxiliary positive-example LM loss remains exact-only via
its reward threshold, and AIME 2024 avg@k is the eval. Prompt groups roll out
and replay separately because their lengths differ, then accumulate into the
shared actor/critic optimizer minibatch. ``--rollout-only`` reports whether
rewards vary within prompt groups at the initial 50/50 gate before any update
is attempted.

    python3 -m postraining.train_latent_vapo \
        --checkpoint ablation_results/<run>/pretraining_checkpoint.pt \
        --output postraining/runs/<name> [--rollout-only]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import fields
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.tensorboard import SummaryWriter

import train_gpt as baseline
from fresh_lejepa_train import FreshHyperparameters
import postraining.latent_rollout
from postraining.latent_eval import (
    evaluate_latent_math,
    verify_terminated_answer,
)
from postraining.benchmark_report import (
    CAPTURE_PROBLEMS,
    CAPTURE_SAMPLES_PER_PROBLEM,
    write_benchmark_report,
)
from postraining.core import (
    POSTTRAIN_REWARD_SCHEMA,
    JsonlLogger,
    POSTTRAIN_PROMPT_TOKENS,
    POSTTRAIN_RESPONSE_TOKENS,
    POSTTRAIN_STREAM_TOKENS,
    answer_style,
    clipped_policy_loss,
    encode_prompt,
    extract_final_answer,
    deterministic_math_subset,
    generalized_advantage_estimate,
    length_adaptive_lambda,
    load_unique_math_rows,
    modal_answer_baseline,
    module_answer_baselines,
    nearby_numeric_reward,
    parse_numeric_answer,
    positive_example_lm_loss,
    validate_posttraining_context_budget,
    verify_answer,
)
from postraining.latent_rollout import (
    PAD_SLOT,
    THOUGHT_SLOT,
    LatentRolloutBatch,
    assign_terminal_rewards,
    pack_rollout_groups_for_replay,
    emitted_token_rows,
    half_forced_group_members,
    iter_length_aware_microbatches,
    refresh_old_statistics,
    replay_head_inputs,
    scatter_replay_statistics,
    select_thought_actions,
    rollout_continuations,
    split_rollout_groups,
    trim_stream,
)
from postraining.latent_thought import (
    EMIT,
    THINK,
    RENDERER_FEATURES_SCHEMA,
    ROLLOUT_POLICY_SCHEMA,
    THOUGHT_DISTRIBUTION_SCHEMA,
    THOUGHT_INPUT_SCHEMA,
    THOUGHT_MEAN_SCHEMA,
    LatentThoughtModel,
    migrate_legacy_wrapper_checkpoint,
    validate_renderer_checkpoint,
)
from postraining.model_io import fresh_trunk, load_model
from postraining.train_vapo import prompt_text
from postraining.value_model import SeparateCritic


EXECUTION_SCHEMA = (
    "shuffled_pool1024_disjoint_b256_per_dim_thought_clip_zero_affine_general_lr_sequential_data/v18"
)
PREVIOUS_EXECUTION_SCHEMA = (
    "shuffled_pool1024_disjoint_b256_zero_affine_general_lr_sequential_data/v17"
)
GAIN_SCALED_EXECUTION_SCHEMA = (
    "disjoint_b512_gain_scaled_gaussian_adapter_general_lr_sequential_data/v15"
)
GAIN_SCALED_THOUGHT_INPUT_SCHEMA = (
    "fresh_learned_scalar_identity_affine_s1e-4/v4"
)
PROMPT_ORDER_SCHEMA = "sequential_one_pass/v1"
DEFAULT_BPB_GUARD_TOKENS = 2 * 1024 * 1024
DEFAULT_PERIODIC_EVAL_EVERY = 150
# v2: the grading style follows each row's reward_model.style (Minerva for
# DAPO/AIME lineage data, official exact match for mathematics_dataset rows)
# instead of Minerva-normalizing everything.
REWARD_SCHEMA = POSTTRAIN_REWARD_SCHEMA


def math_dataset_identity(path: str | Path, exclude_modules: str) -> str:
    """Content identity that makes a sequential cursor safe to resume."""
    digest = hashlib.sha256()
    digest.update(PROMPT_ORDER_SCHEMA.encode())
    digest.update(b"\0")
    digest.update(exclude_modules.encode())
    digest.update(b"\0")
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def resume_execution_schema_compatible(payload: dict) -> bool:
    """Resume compatible policy state at a complete rollout-pool boundary."""
    execution_schema = payload.get("execution_schema")
    if execution_schema == EXECUTION_SCHEMA:
        return True
    if execution_schema != PREVIOUS_EXECUTION_SCHEMA:
        return False
    # v17 used a joint Gaussian-vector PPO ratio. Only its actor-untouched
    # critic-warmup checkpoint can safely enter v18's per-dimension objective.
    actor_optimizer = payload.get("optimizers", {}).get("actor", {})
    return (
        int(payload.get("step", -1)) == 0
        and not actor_optimizer.get("state", {})
    )


def migrate_zero_adapter_resume(
    payload: dict,
    wrapper: LatentThoughtModel,
) -> dict[str, object]:
    """Explicitly replace v15's gain-scaled adapter while preserving resume.

    The recurrent policy semantics change, so this migration is never silent:
    the caller must opt in. All actor state and Adam moments are retained
    except the three old adapter parameters. The new weight/bias start at exact
    zero with fresh optimizer state; the removed scalar has no successor.
    """
    actual_execution = payload.get("execution_schema")
    actual_input = payload.get("thought_input_schema")
    if actual_execution != GAIN_SCALED_EXECUTION_SCHEMA:
        raise ValueError(
            "zero-adapter migration requires execution schema "
            f"{GAIN_SCALED_EXECUTION_SCHEMA!r}; got {actual_execution!r}"
        )
    if actual_input != GAIN_SCALED_THOUGHT_INPUT_SCHEMA:
        raise ValueError(
            "zero-adapter migration requires thought-input schema "
            f"{GAIN_SCALED_THOUGHT_INPUT_SCHEMA!r}; got {actual_input!r}"
        )

    state = payload["model"]
    scalar_key = "adapter.interpolation_strength"
    if scalar_key not in state:
        raise ValueError("gain-scaled checkpoint is missing its adapter scalar")
    scalar = state[scalar_key]
    weight_key = "adapter.projection.weight"
    bias_key = "adapter.projection.bias"
    expected_shapes = {
        weight_key: tuple(wrapper.adapter.projection.weight.shape),
        bias_key: tuple(wrapper.adapter.projection.bias.shape),
    }
    for key, expected_shape in expected_shapes.items():
        if key not in state or tuple(state[key].shape) != expected_shape:
            actual_shape = tuple(state[key].shape) if key in state else None
            raise ValueError(
                f"gain-scaled checkpoint {key} shape must be "
                f"{expected_shape}; got {actual_shape}"
            )
    if scalar.numel() != 1 or not torch.isfinite(scalar).all():
        raise ValueError("gain-scaled checkpoint adapter scalar must be finite")

    actor_optimizer = payload["optimizers"]["actor"]
    groups = actor_optimizer["param_groups"]
    if len(groups) != 6 or len(groups[2]["params"]) != 3:
        raise ValueError(
            "gain-scaled actor optimizer does not have the expected six-group "
            "layout with scalar/weight/bias adapter parameters"
        )

    source_strength = float(scalar)
    scalar_id, weight_id, bias_id = groups[2]["params"]
    if len({scalar_id, weight_id, bias_id}) != 3:
        raise ValueError(
            "gain-scaled actor optimizer adapter parameter IDs must be distinct"
        )

    state.pop(scalar_key)
    state[weight_key] = torch.zeros_like(state[weight_key])
    state[bias_key] = torch.zeros_like(state[bias_key])
    old_state = actor_optimizer["state"]
    for parameter_id in (scalar_id, weight_id, bias_id):
        old_state.pop(parameter_id, None)
    # Optimizer state_dict loading maps saved parameter IDs to live parameters
    # positionally. Retain the old weight/bias IDs in their original order so
    # later groups and every unrelated moment remain aligned.
    groups[2]["params"] = [weight_id, bias_id]

    payload["execution_schema"] = EXECUTION_SCHEMA
    payload["thought_input_schema"] = THOUGHT_INPUT_SCHEMA
    return {
        "source_execution_schema": actual_execution,
        "source_thought_input_schema": actual_input,
        "source_adapter_strength": source_strength,
        "reset_parameters": [
            "adapter.projection.weight",
            "adapter.projection.bias",
        ],
        "reset_optimizer_state": True,
    }


def sample_prompt_batch(
    loader: "baseline.DistributedTokenLoader",
    prompt_tokens: int,
    continuation_tokens: int,
    prompts: int,
    samples_per_prompt: int,
    seq_len: int,
    grad_accum: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Carve (prompt, reference-continuation) pairs from the token stream.

    Training no longer uses this (RL data is DAPO-Math only); it remains for
    ``sample_latent --fineweb`` inspection.
    """
    needed = prompt_tokens + continuation_tokens
    if needed > seq_len:
        raise ValueError("prompt + continuation must fit in one training sequence")
    inputs, _ = loader.next_batch(prompts * seq_len * grad_accum, seq_len, grad_accum)
    rows = inputs[:prompts]
    prompt_ids = rows[:, :prompt_tokens]
    reference_ids = rows[:, prompt_tokens:needed]
    return (
        prompt_ids.repeat_interleave(samples_per_prompt, dim=0),
        reference_ids.repeat_interleave(samples_per_prompt, dim=0),
    )


class MathPromptSampler:
    """Strictly sequential, one-pass prompt stream with a resumable cursor."""

    def __init__(
        self,
        rows: list[dict],
        seed: int,
        dataset_identity: str | None = None,
    ):
        if not rows:
            raise ValueError("no math prompts loaded")
        self.rows = rows
        del seed  # retained in the API for checkpoint/CLI compatibility only
        self.cursor = 0
        self.dataset_identity = dataset_identity

    def next_rows(self, count: int) -> list[dict]:
        if count < 0:
            raise ValueError("prompt count must be nonnegative")
        end = self.cursor + count
        if end > len(self.rows):
            raise RuntimeError(
                f"prompt stream exhausted at {self.cursor}; requested {count} "
                f"from {len(self.rows)} one-pass rows"
            )
        picked = self.rows[self.cursor:end]
        self.cursor = end
        return picked


def score_math_rollout(
    batch: LatentRolloutBatch,
    truth: str,
    tokenizer,
    stop_ids: tuple[int, ...],
    style: str = "minerva",
    nearby_reward_max: float = 0.1,
) -> None:
    """Exact verifier reward plus bounded final-answer numeric proximity."""
    scores = []
    for emitted in emitted_token_rows(batch):
        stop_cut = next(
            (index for index, token in enumerate(emitted) if token in stop_ids),
            None,
        )
        if stop_cut is None:
            scores.append(0.0)
            continue
        solution = tokenizer.decode(emitted[: stop_cut + 1])
        raw_final_answer = extract_final_answer(solution)
        strict_numeric = (
            parse_numeric_answer(raw_final_answer)
            if raw_final_answer is not None
            else None
        )
        correct, _ = verify_answer(solution, truth, style)
        if correct:
            scores.append(1.0)
        elif strict_numeric is None:
            scores.append(0.0)
        else:
            # Only the raw final Answer: field enters proximity scoring.
            # Earlier candidates and multi-number final fields cannot add or
            # manufacture reward.
            scores.append(
                nearby_numeric_reward(
                    raw_final_answer, truth, nearby_reward_max
                )
            )
    assign_terminal_rewards(
        batch, torch.tensor(scores, dtype=torch.float32, device=batch.rewards.device)
    )


evaluate_aime_latent = evaluate_latent_math


def think_run_lengths(kind: torch.Tensor) -> torch.Tensor:
    """Lengths of every consecutive-THOUGHT run in a (batch, stream) kind map.

    This is total latent-compute telemetry: it includes any forced initial
    thought and merges it with an immediately following optional thought.
    Runs are per-row (prompts themselves are token slots), so flattening
    start/end indices row-major keeps them paired.
    """
    thinks = kind == THOUGHT_SLOT
    previous = torch.zeros_like(thinks)
    previous[:, 1:] = thinks[:, :-1]
    following = torch.zeros_like(thinks)
    following[:, :-1] = thinks[:, 1:]
    starts = (thinks & ~previous).flatten().nonzero().squeeze(-1)
    ends = (thinks & ~following).flatten().nonzero().squeeze(-1)
    return (ends - starts + 1).float()


def rollout_diagnostics(
    batch: LatentRolloutBatch,
    samples_per_prompt: int,
    stop_ids: tuple[int, ...] = (),
) -> dict[str, float | int]:
    gate_actions = batch.gate_mask.sum().clamp_min(1)
    stop_set = set(stop_ids)
    # Fraction of rows that terminated themselves (emitted BOS or EOS)
    # rather than exhausting the token/stream budget.
    ended = [
        float(any(token in stop_set for token in row))
        for row in emitted_token_rows(batch)
    ]
    generated = batch.action_mask.bool()
    think_fraction = float(
        ((batch.gate_actions == THINK).float() * batch.gate_mask).sum()
        / gate_actions
    )
    runs = think_run_lengths(batch.kind)
    grouped = batch.reward_scalar.reshape(-1, samples_per_prompt)
    exact = batch.reward_scalar == 1.0
    grouped_exact = exact.reshape(-1, samples_per_prompt).float()
    # Did thinking pay off THIS group?  Within-group Pearson correlation
    # between per-trajectory think counts and rewards (0.0 when either side
    # has no variance — undefined groups dilute the aggregate toward zero,
    # which is the honest prior for "no evidence either way").
    # Reward comparisons here isolate OPTIONAL Bernoulli-selected thoughts;
    # the paired forced/unforced prefix comparison is reported separately.
    think_counts = (
        (batch.gate_actions == THINK).float() * batch.gate_mask
    ).sum(1)
    forced_initial = (batch.action_mask - batch.gate_mask).sum(1) > 0
    thinkers = think_counts > 0

    def think_reward_correlation(rows: torch.Tensor) -> float:
        counts = think_counts[rows]
        rewards = batch.reward_scalar[rows]
        if counts.numel() == 0:
            return 0.0
        centered_counts = counts - counts.mean()
        centered_rewards = rewards - rewards.mean()
        scale = (
            centered_counts.square().mean().sqrt()
            * centered_rewards.square().mean().sqrt()
        )
        return (
            float((centered_counts * centered_rewards).mean() / scale)
            if float(scale) > 0
            else 0.0
        )
    return {
        "trajectories": batch.reward_scalar.numel(),
        "stream_length": batch.stream_length,
        "reward_mean": float(batch.reward_scalar.mean()),
        "reward_std": float(batch.reward_scalar.std(unbiased=False)),
        "within_group_reward_std": float(grouped.std(dim=1, unbiased=False).mean()),
        "exact_accuracy": float(exact.float().mean()),
        "exact_within_group_reward_std": float(
            grouped_exact.std(dim=1, unbiased=False).mean()
        ),
        "partial_reward_fraction": float(
            ((batch.reward_scalar > 0.0) & ~exact).float().mean()
        ),
        "think_fraction": think_fraction,
        "forced_initial_thinks_per_trajectory": float(
            (((batch.gate_actions == THINK).float() * batch.action_mask).sum(1)
             - think_counts).mean()
        ),
        "forced_initial_trajectory_fraction": float(
            forced_initial.float().mean()
        ),
        "reward_mean_forced_initial": (
            float(batch.reward_scalar[forced_initial].mean())
            if bool(forced_initial.any())
            else 0.0
        ),
        "reward_mean_unforced_initial": (
            float(batch.reward_scalar[~forced_initial].mean())
            if bool((~forced_initial).any())
            else 0.0
        ),
        "thoughts_per_trajectory": float(
            ((batch.gate_actions == THINK).float() * batch.action_mask).sum(1).mean()
        ),
        "think_run_mean": float(runs.mean()) if runs.numel() else 0.0,
        "think_run_std": float(runs.std(unbiased=False)) if runs.numel() else 0.0,
        "think_runs_per_trajectory": runs.numel() / batch.reward_scalar.numel(),
        "emits_per_trajectory": float(batch.emit_mask.sum(1).mean()),
        "actions_per_trajectory": float(batch.action_mask.sum(1).mean()),
        "old_value_mean": float(batch.old_values[generated].mean()) if generated.any() else 0.0,
        "optional_thinking_trajectory_fraction": float(thinkers.float().mean()),
        "reward_mean_optional_thinking": (
            float(batch.reward_scalar[thinkers].mean()) if bool(thinkers.any()) else 0.0
        ),
        "reward_mean_no_optional_thinking": (
            float(batch.reward_scalar[~thinkers].mean())
            if bool((~thinkers).any())
            else 0.0
        ),
        "think_reward_correlation": think_reward_correlation(
            torch.ones_like(forced_initial)
        ),
        "think_reward_correlation_forced_initial": think_reward_correlation(
            forced_initial
        ),
        "think_reward_correlation_unforced_initial": think_reward_correlation(
            ~forced_initial
        ),
        "ended_fraction": sum(ended) / max(len(ended), 1),
    }


def aggregate_diagnostics(
    groups: list[LatentRolloutBatch],
    samples_per_prompt: int,
    stop_ids: tuple[int, ...] = (),
) -> dict[str, float | int]:
    """Mean of per-group rollout diagnostics; trajectory counts are summed."""
    per_group = [
        rollout_diagnostics(group, samples_per_prompt, stop_ids) for group in groups
    ]
    aggregated: dict[str, float | int] = {}
    for key in per_group[0]:
        values = [metrics[key] for metrics in per_group]
        if key == "trajectories":
            aggregated[key] = int(sum(values))
        else:
            aggregated[key] = float(sum(values) / len(values))
    runs = [think_run_lengths(group.kind) for group in groups]
    nonempty_runs = [run for run in runs if run.numel()]
    if nonempty_runs:
        all_runs = torch.cat(nonempty_runs)
        aggregated["think_run_p95"] = float(
            torch.quantile(all_runs, 0.95, interpolation="higher")
        )
        aggregated["think_run_max"] = float(all_runs.max())
    else:
        aggregated["think_run_p95"] = 0.0
        aggregated["think_run_max"] = 0.0
    return aggregated


def _weighted_metric_mean(
    metrics: list[dict[str, float]], key: str, weight_key: str
) -> float:
    total_weight = sum(metric[weight_key] for metric in metrics)
    if total_weight <= 0:
        return 0.0
    return sum(
        metric[key] * metric[weight_key] for metric in metrics
    ) / total_weight


def aggregate_value_diagnostics(
    metrics: list[dict[str, float]],
) -> dict[str, float]:
    """Pool token moments across an online sequence of critic states."""
    if not metrics:
        raise ValueError("at least one metric group is required")
    target_mean = _weighted_metric_mean(
        metrics, "value_target_mean", "action_count"
    )
    residual_mean = _weighted_metric_mean(
        metrics, "value_residual_mean", "action_count"
    )
    action_count = sum(metric["action_count"] for metric in metrics)
    if action_count <= 0:
        raise ValueError("value diagnostics require at least one action")
    target_second_moment = sum(
        (
            metric["value_target_variance"]
            + metric["value_target_mean"] ** 2
        )
        * metric["action_count"]
        for metric in metrics
    ) / action_count
    residual_second_moment = sum(
        (
            metric["value_residual_variance"]
            + metric["value_residual_mean"] ** 2
        )
        * metric["action_count"]
        for metric in metrics
    ) / action_count
    target_variance = max(0.0, target_second_moment - target_mean**2)
    residual_variance = max(0.0, residual_second_moment - residual_mean**2)
    explained_variance = (
        1.0 - residual_variance / target_variance
        if target_variance > 1e-12
        else 0.0
    )
    return {
        "prediction_mean": _weighted_metric_mean(
            metrics, "value_mean", "action_count"
        ),
        "target_mean": target_mean,
        "residual_mean": residual_mean,
        "target_variance": target_variance,
        "residual_variance": residual_variance,
        "explained_variance": explained_variance,
        "excess_ce": _weighted_metric_mean(
            metrics, "value_excess_ce", "action_count"
        ),
    }


def aggregate_actor_tensorboard_metrics(
    metrics: list[dict[str, float]],
) -> dict[str, float]:
    """One compact, correctly weighted dashboard row per actor step."""
    if not metrics:
        raise ValueError("at least one actor metric group is required")

    value = aggregate_value_diagnostics(metrics)
    advantage_mean = _weighted_metric_mean(
        metrics, "advantage_mean", "action_count"
    )
    advantage_second_moment = _weighted_metric_mean(
        [
            {
                **metric,
                "advantage_second_moment": (
                    metric["advantage_std"] ** 2
                    + metric["advantage_mean"] ** 2
                ),
            }
            for metric in metrics
        ],
        "advantage_second_moment",
        "action_count",
    )
    advantage_std = math.sqrt(
        max(0.0, advantage_second_moment - advantage_mean**2)
    )
    log_sigma_mean = _weighted_metric_mean(
        metrics, "thought_log_sigma_mean", "thought_action_count"
    )
    log_sigma_second_moment = _weighted_metric_mean(
        [
            {
                **metric,
                "thought_log_sigma_second_moment": (
                    metric["thought_log_sigma_std"] ** 2
                    + metric["thought_log_sigma_mean"] ** 2
                ),
            }
            for metric in metrics
        ],
        "thought_log_sigma_second_moment",
        "thought_action_count",
    )
    last = metrics[-1]
    # Each group is already divided by the denominator of the complete actor
    # optimizer minibatch, so its loss is a contribution to be summed rather
    # than another independently normalized minibatch mean.
    policy_contribution = sum(metric["policy_loss"] for metric in metrics)
    positive_lm_contribution = sum(
        metric["positive_lm_loss"] * metric["positive_lm_weight"]
        for metric in metrics
    )
    gate_entropy_bonus = sum(
        metric["gate_entropy_bonus"] for metric in metrics
    )
    return {
        "loss/policy": policy_contribution,
        "loss/positive_lm_weighted": positive_lm_contribution,
        "bonus/gate_entropy_weighted": gate_entropy_bonus,
        "loss/actor_total": (
            policy_contribution
            + positive_lm_contribution
            - gate_entropy_bonus
        ),
        "value/token_weighted_excess_ce": value["excess_ce"],
        "value/prediction_mean": value["prediction_mean"],
        "value/target_mean": value["target_mean"],
        "value/residual_mean": value["residual_mean"],
        "value/online_epoch_explained_variance": value["explained_variance"],
        "advantage/mean": advantage_mean,
        "advantage/std": advantage_std,
        "advantage/optional_think_mean": _weighted_metric_mean(
            metrics, "think_advantage_mean", "think_action_count"
        ),
        "advantage/forced_initial_think_mean": _weighted_metric_mean(
            metrics,
            "forced_initial_think_advantage_mean",
            "forced_initial_think_action_count",
        ),
        "advantage/all_think_mean": _weighted_metric_mean(
            metrics, "thought_advantage_mean", "thought_action_count"
        ),
        "advantage/emit_mean": _weighted_metric_mean(
            metrics, "emit_advantage_mean", "emit_action_count"
        ),
        "behavior/optional_think_probability": 1.0 - _weighted_metric_mean(
            metrics, "emit_probability", "gate_action_count"
        ),
        "behavior/thought_adapter_weight_rms": last[
            "thought_adapter_weight_rms"
        ],
        "behavior/thought_adapter_bias_rms": last[
            "thought_adapter_bias_rms"
        ],
        "behavior/gate_entropy": _weighted_metric_mean(
            metrics, "gate_entropy", "gate_action_count"
        ),
        "sigma/log_std_mean": log_sigma_mean,
        "sigma/log_std_std": math.sqrt(
            max(0.0, log_sigma_second_moment - log_sigma_mean**2)
        ),
        "sigma/log_std_min": min(
            metric["thought_log_sigma_min"] for metric in metrics
        ),
        "sigma/log_std_max": max(
            metric["thought_log_sigma_max"] for metric in metrics
        ),
        "sigma/std_mean": _weighted_metric_mean(
            metrics, "thought_sigma_mean", "thought_action_count"
        ),
        "sigma/expected_noise_norm": _weighted_metric_mean(
            metrics, "thought_expected_noise_norm", "thought_action_count"
        ),
        "sigma/realized_noise_norm": _weighted_metric_mean(
            metrics, "thought_realized_noise_norm", "thought_action_count"
        ),
        "sigma/normalized_noise_rms": _weighted_metric_mean(
            metrics, "thought_normalized_noise_rms", "thought_action_count"
        ),
        "sigma/head_raw_bias_mean": last["thought_log_sigma_raw_bias_mean"],
        "sigma/head_weight_rms": last["thought_log_sigma_weight_rms"],
        "sigma/state_residual_gain": last["thought_log_sigma_residual_gain"],
        "sigma/mean_norm": _weighted_metric_mean(
            metrics, "thought_mean_norm", "thought_action_count"
        ),
        "sigma/noise_mean_norm_ratio": (
            _weighted_metric_mean(
                metrics, "thought_mean_norm", "thought_action_count"
            )
            / max(
                _weighted_metric_mean(
                    metrics, "thought_expected_noise_norm", "thought_action_count"
                ),
                1e-12,
            )
        ),
        "sigma/mean_head_weight_rms": last["thought_mean_weight_rms"],
        "sigma/mean_head_bias_rms": last["thought_mean_bias_rms"],
        "sigma/mean_output_gain": last["thought_mean_output_gain"],
        "kl/gate_behavior": _weighted_metric_mean(
            metrics, "gate_behavior_kl", "gate_action_count"
        ),
        "kl/renderer_behavior": _weighted_metric_mean(
            metrics, "renderer_behavior_kl", "emit_action_count"
        ),
        "kl/thought_behavior_joint": _weighted_metric_mean(
            metrics, "thought_behavior_kl_joint", "thought_action_count"
        ),
        "kl/policy_behavior_per_action": sum(
            metric["policy_behavior_kl_per_action"] for metric in metrics
        ),
        "clip/policy": sum(
            metric["policy_clip_fraction"] for metric in metrics
        ),
        "ratio/joint_abs_log_max": max(
            metric["joint_abs_log_ratio_max"] for metric in metrics
        ),
        "ratio/thought_dim_abs_log_max": max(
            metric["thought_dim_abs_log_ratio_max"] for metric in metrics
        ),
        "ratio/harmful_positive_log_max": max(
            metric["harmful_positive_log_ratio_max"] for metric in metrics
        ),
        # Both optimizers accumulate across groups; only the final group has
        # the complete pre-step gradient norm.
        "grad/trunk": last["trunk_grad_norm"],
        "grad/renderer": last["renderer_grad_norm"],
        "grad/adapter": last["adapter_grad_norm"],
        "grad/gate": last["gate_grad_norm"],
        "grad/sigma": last["sigma_grad_norm"],
        "grad/thought_mean": last["thought_mean_grad_norm"],
        "grad/critic": last["critic_grad_norm"],
    }


def rollout_tensorboard_metrics(metrics: dict[str, float | int]) -> dict[str, float]:
    """Select semantic rollout signals while leaving rich telemetry in JSON."""
    return {
        "reward/mean": float(metrics["reward_mean"]),
        "reward/within_group_std": float(metrics["within_group_reward_std"]),
        "reward/exact_accuracy": float(metrics["exact_accuracy"]),
        "reward/exact_within_group_std": float(
            metrics["exact_within_group_reward_std"]
        ),
        "reward/partial_fraction": float(metrics["partial_reward_fraction"]),
        "reward/forced_initial_mean": float(
            metrics["reward_mean_forced_initial"]
        ),
        "reward/unforced_initial_mean": float(
            metrics["reward_mean_unforced_initial"]
        ),
        "reward/forced_initial_delta": float(
            metrics["reward_mean_forced_initial"]
            - metrics["reward_mean_unforced_initial"]
        ),
        "reward/ended_fraction": float(metrics["ended_fraction"]),
        "behavior/optional_think_fraction": float(metrics["think_fraction"]),
        "behavior/thoughts_per_trajectory": float(
            metrics["thoughts_per_trajectory"]
        ),
        "behavior/emits_per_trajectory": float(
            metrics["emits_per_trajectory"]
        ),
        "behavior/think_run_p95": float(metrics["think_run_p95"]),
        "behavior/think_run_max": float(metrics["think_run_max"]),
    }


def optimizer_minibatch_orders(
    group_count: int,
    groups_per_minibatch: int,
    generator: torch.Generator | None = None,
    allow_partial_final: bool = False,
) -> list[list[int]]:
    """Shuffle and partition one frozen-policy pool without reuse.

    PPO keeps every stored old log-probability fixed to the behavior policy
    that sampled the pool.  As in CleanRL, one fresh permutation is drawn at
    the start of the (single) epoch; later minibatches therefore compare the
    updated policy to that same behavior snapshot rather than refreshing old
    statistics after each optimizer step.
    """
    if group_count < 1:
        raise ValueError("group_count must be positive")
    if groups_per_minibatch < 1:
        raise ValueError("groups_per_minibatch must be positive")
    if group_count % groups_per_minibatch and not allow_partial_final:
        raise ValueError("group_count must divide into complete minibatches")
    order = torch.randperm(group_count, generator=generator).tolist()
    minibatches = [
        order[start : start + groups_per_minibatch]
        for start in range(0, group_count, groups_per_minibatch)
    ]
    if any(not minibatch for minibatch in minibatches):
        raise RuntimeError("optimizer partition produced an empty minibatch")
    return minibatches


def plan_one_pass_training(
    *,
    dataset_rows: int,
    sampler_cursor: int,
    warmup_updates: int,
    remaining_actor_steps: int,
    prompts_per_minibatch: int,
) -> tuple[int, int]:
    """Validate exact one-pass consumption and return prompts and actor steps.

    Warmup consumes one complete prompt minibatch per critic update. Actor
    training then consumes every remaining row, with at most one terminal
    partial minibatch. This helper is deliberately pure so fresh curriculum
    starts and checkpoint resumes share testable no-reuse arithmetic.
    """
    if not 0 <= sampler_cursor <= dataset_rows:
        raise ValueError("sampler cursor must lie within the dataset")
    if warmup_updates < 0 or remaining_actor_steps < 0:
        raise ValueError("update counts must be nonnegative")
    if prompts_per_minibatch < 1:
        raise ValueError("prompts_per_minibatch must be positive")
    available = dataset_rows - sampler_cursor
    warmup_prompts = prompts_per_minibatch * warmup_updates
    if warmup_prompts > available:
        raise ValueError("critic warmup alone exceeds the unused target dataset")
    actor_prompts = available - warmup_prompts
    required_actor_steps = math.ceil(actor_prompts / prompts_per_minibatch)
    if remaining_actor_steps != required_actor_steps:
        raise ValueError(
            "--consume-all-prompts requires exactly "
            f"{required_actor_steps} remaining actor steps for "
            f"{actor_prompts} post-warmup prompts; got "
            f"{remaining_actor_steps}"
        )
    return available, required_actor_steps


def actor_minibatch_denominators(
    groups: list[LatentRolloutBatch],
    indices: list[int],
    positive_reward_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Global action/gate/positive-token denominators for one actor step."""
    if not indices:
        raise ValueError("at least one prompt group is required")
    action_count = torch.stack(
        [groups[index].action_mask.sum() for index in indices]
    ).sum()
    gate_action_count = torch.stack(
        [groups[index].gate_mask.sum() for index in indices]
    ).sum()
    positive_token_count = torch.stack(
        [
            (
                groups[index].emit_mask
                * (
                    groups[index].reward_scalar >= positive_reward_threshold
                )[:, None]
            ).sum()
            for index in indices
        ]
    ).sum()
    return action_count, gate_action_count, positive_token_count


def write_actor_tensorboard_metrics(
    tensorboard, dashboard: dict[str, float], behavior_age: int, step: int
) -> None:
    """Write one actor-step row, eliding exact fresh-behavior zeros."""
    if behavior_age == 0:
        refresh_drift = max(
            dashboard[tag]
            for tag in (
                "kl/gate_behavior",
                "kl/renderer_behavior",
                "kl/thought_behavior_joint",
                "kl/policy_behavior_per_action",
                "clip/policy",
                "ratio/joint_abs_log_max",
                "ratio/thought_dim_abs_log_max",
                "ratio/harmful_positive_log_max",
            )
        )
        tensorboard.add_scalar(
            "debug/behavior_refresh_max_drift", refresh_drift, step
        )
    for tag, value in dashboard.items():
        if behavior_age == 0 and tag in {
            "kl/gate_behavior",
            "kl/renderer_behavior",
            "kl/thought_behavior_joint",
            "kl/policy_behavior_per_action",
            "clip/policy",
            "ratio/joint_abs_log_max",
            "ratio/thought_dim_abs_log_max",
            "ratio/harmful_positive_log_max",
        }:
            continue
        tensorboard.add_scalar(tag, value, step)


def gradient_norm_tensor(parameters) -> torch.Tensor:
    parameters = list(parameters)
    grads = [
        parameter.grad.detach()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not grads:
        if not parameters:
            return torch.zeros((), dtype=torch.float32)
        return parameters[0].new_zeros((), dtype=torch.float32)
    # The old loop converted every parameter's sum to float separately,
    # forcing hundreds of device synchronizations. Foreach norms enqueue the
    # whole collection and synchronize once for the final scalar.
    norms = torch._foreach_norm(grads, 2.0)
    return torch.linalg.vector_norm(torch.stack(norms)).float()


def gradient_norm(parameters) -> float:
    """Public scalar form used by standalone diagnostics and tests."""
    return float(gradient_norm_tensor(parameters))


def scalar_tensors_to_floats(
    values: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Resolve same-device scalar telemetry with one device synchronization."""
    if not values:
        return {}
    keys = tuple(values)
    packed = torch.stack(
        [values[key].detach().reshape(()).float() for key in keys]
    ).cpu().tolist()
    return dict(zip(keys, packed, strict=True))


def build_optimizers(
    wrapper: LatentThoughtModel,
    critic: SeparateCritic,
    learning_rate: float,
    fused: bool = True,
) -> dict[str, torch.optim.Optimizer]:
    """The actor/critic optimizer layout.

    One actor AdamW retains six semantic groups for exact telemetry and
    checkpoint validation, but every trainable policy component uses the same
    general learning rate as the critic: trunk, Bernoulli gate, recurrent
    adapter, renderer, state-dependent log-sigma, and fresh mean. This removes
    the previous hand-tuned head-specific rates from the fresh-policy test.
    The probes live under
    ``blocks[-1]`` (so pretraining's optimizer saw them), which makes
    name-prefix filtering wrong — exclude them from the trunk by identity.
    """
    backbone = wrapper.backbone
    probe_parameter_ids = {
        id(parameter)
        for probe in (backbone.policy_probe, backbone.critic_probe)
        for parameter in probe.parameters()
    }
    trunk_parameters = [
        parameter
        for parameter in backbone.parameters()
        if id(parameter) not in probe_parameter_ids
    ]
    return {
        "actor": torch.optim.AdamW(
            [
                {"params": trunk_parameters, "lr": learning_rate},
                {"params": list(wrapper.gate.parameters()), "lr": learning_rate},
                {"params": list(wrapper.adapter.parameters()), "lr": learning_rate},
                {"params": list(backbone.policy_probe.parameters()), "lr": learning_rate},
                {
                    "params": list(wrapper.transition.log_sigma_head.parameters()),
                    "lr": learning_rate,
                },
                {
                    "params": list(wrapper.transition.mean_head.parameters()),
                    "lr": learning_rate,
                },
            ],
            weight_decay=0.0,
            fused=fused,
        ),
        "critic": torch.optim.AdamW(
            critic.parameters(), lr=learning_rate, weight_decay=0.0, fused=fused,
        ),
    }


def joint_action_logprobs(
    new_gate_logprobs: torch.Tensor,
    old_gate_logprobs: torch.Tensor,
    new_token_logprobs: torch.Tensor,
    old_token_logprobs: torch.Tensor,
    new_thought_logprobs: torch.Tensor,
    old_thought_logprobs: torch.Tensor,
    gate_mask: torch.Tensor,
    emit_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine conditional policy factors into one log-probability/action."""
    return (
        new_gate_logprobs * gate_mask
        + new_token_logprobs * emit_mask
        + new_thought_logprobs,
        old_gate_logprobs * gate_mask
        + old_token_logprobs * emit_mask
        + old_thought_logprobs,
    )


def per_dimension_thought_policy_loss(
    new_gate_logprobs: torch.Tensor,
    old_gate_logprobs: torch.Tensor,
    new_thought_logprobs: torch.Tensor,
    old_thought_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    gate_mask: torch.Tensor,
    action_denominator: torch.Tensor,
    epsilon_low: float = 0.20,
    epsilon_high: float = 0.28,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clip a diagonal-Gaussian THINK action one latent dimension at a time.

    Factorwise clipping must retain the score-function gradient of the joint
    diagonal Gaussian: ``A * sum_d grad(log p_d)``. Merely averaging the
    dimensional losses would weaken mean/sigma learning by ``latent_dim``.
    The straight-through rescaling below reports their mean surrogate value
    while preserving their summed gradient.

    The optional Bernoulli gate is clipped exactly once as its own factor. A
    detached baseline correction makes an unchanged optional THINK report one
    action's loss rather than two without altering either factor's gradient.
    A forced THINK has ``gate_mask == 0`` and trains only Gaussian factors.

    The returned clip fractions are action-normalized. A thought whose every
    dimension clips contributes one to the first; a clipped optional gate
    contributes one to the second.
    """
    if new_thought_logprobs.ndim != 2:
        raise ValueError("thought log-probabilities must have shape [N, D]")
    if new_thought_logprobs.shape != old_thought_logprobs.shape:
        raise ValueError("new and old thought log-probabilities must match")
    action_count, latent_dim = new_thought_logprobs.shape
    if latent_dim < 1:
        raise ValueError("thought log-probabilities need at least one dimension")
    expected_vector_shape = (action_count,)
    for name, value in (
        ("new gate log-probabilities", new_gate_logprobs),
        ("old gate log-probabilities", old_gate_logprobs),
        ("advantages", advantages),
        ("gate mask", gate_mask),
    ):
        if tuple(value.shape) != expected_vector_shape:
            raise ValueError(f"{name} must have shape {expected_vector_shape}")

    dimension_log_ratio = new_thought_logprobs - old_thought_logprobs
    log_lower = torch.log(
        dimension_log_ratio.new_tensor(1.0 - epsilon_low)
    )
    log_upper = torch.log(
        dimension_log_ratio.new_tensor(1.0 + epsilon_high)
    )
    dimension_advantages = advantages[:, None]
    effective_dimension_log_ratio = torch.where(
        dimension_advantages >= 0,
        torch.minimum(dimension_log_ratio, log_upper),
        torch.maximum(dimension_log_ratio, log_lower),
    )
    dimension_denominator = (
        action_denominator.to(dimension_log_ratio.device) * latent_dim
    ).clamp_min(1)
    dimension_mean_loss = -(
        effective_dimension_log_ratio.exp() * dimension_advantages
    ).sum() / dimension_denominator
    dimension_clip_fraction = (
        (dimension_log_ratio < log_lower)
        | (dimension_log_ratio > log_upper)
    ).sum() / dimension_denominator
    dimension_loss = (
        dimension_mean_loss.detach()
        + latent_dim * (dimension_mean_loss - dimension_mean_loss.detach())
    )
    gate_loss, gate_clip_fraction, _ = clipped_policy_loss(
        new_gate_logprobs,
        old_gate_logprobs,
        advantages,
        gate_mask,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        denominator=action_denominator,
        estimate_kl=False,
    )
    optional_gate_baseline = (
        advantages.detach() * gate_mask
    ).sum() / action_denominator.clamp_min(1)
    return (
        dimension_loss + gate_loss + optional_gate_baseline,
        dimension_clip_fraction,
        gate_clip_fraction,
    )


def update_minibatch(
    wrapper: LatentThoughtModel,
    critic: SeparateCritic,
    batch: LatentRolloutBatch,
    optimizers: dict[str, torch.optim.Optimizer],
    value_only: bool = False,
    positive_lm_weight: float = 0.0,
    positive_reward_threshold: float = 0.5,
    thought_pg_coef: float = 1.0,
    gate_pg_coef: float = 1.0,
    gate_entropy_coef: float = 0.0,
    actor_step: bool = True,
    critic_step: bool = True,
    policy_action_denominator: torch.Tensor | None = None,
    gate_action_denominator: torch.Tensor | None = None,
    positive_token_denominator: torch.Tensor | None = None,
    value_action_denominator: torch.Tensor | None = None,
    gae_lambda_alpha: float = 0.05,
    replay_max_trajectories: int = 32,
    replay_attention_budget: int = 4 * 1024 * 1024,
    replay_bucket: int = 1,
) -> dict[str, float]:
    """One minibatch update.

    Actor and critic gradients can be accumulated across prompt groups with
    ``actor_step=False`` and ``critic_step=False``. The trainer uses the same
    effective trajectory minibatch and one optimizer step for both; standalone
    callers keep the self-contained single-step defaults.

    The group is replayed in stable length-sorted shards because causal
    attention is quadratic in stream length. The planner bounds B*L^2 and
    trims every shard independently. The actor objectives keep denominators
    from the COMPLETE optimizer minibatch, which can span prompt groups, so
    sharding changes only floating-point reduction order.
    """
    backbone = wrapper.backbone
    if critic_step:
        optimizers["critic"].zero_grad(set_to_none=True)
    if actor_step and "actor" in optimizers:
        optimizers["actor"].zero_grad(set_to_none=True)
    if replay_max_trajectories < 1:
        raise ValueError("replay max trajectories must be positive")
    if replay_attention_budget < 1:
        raise ValueError("replay attention budget must be positive")
    if not value_only and batch.old_thought_logprobs.shape[-1] == 0:
        raise RuntimeError(
            "actor update requires refresh_old_statistics after rollout"
        )

    positive = batch.reward_scalar >= positive_reward_threshold
    thought_actions = (batch.gate_actions == THINK).float() * batch.action_mask
    optional_think_actions = (
        (batch.gate_actions == THINK).float() * batch.gate_mask
    )
    forced_initial_think_actions = thought_actions - optional_think_actions
    emit_actions = (batch.gate_actions == EMIT).float() * batch.action_mask
    denominators = {
        "action": batch.action_mask.sum().clamp_min(1),
        "gate": batch.gate_mask.sum().clamp_min(1),
        "emit": batch.emit_mask.sum().clamp_min(1),
        "thought": thought_actions.sum().clamp_min(1),
    }
    if policy_action_denominator is None:
        policy_action_denominator = batch.action_mask.sum()
    if gate_action_denominator is None:
        gate_action_denominator = batch.gate_mask.sum()
    if value_action_denominator is None:
        value_action_denominator = batch.action_mask.sum()
    if positive_token_denominator is None:
        positive_token_denominator = (
            batch.emit_mask * positive[:, None]
        ).sum()
    zero = batch.action_mask.new_zeros(())
    totals = {
        key: zero.clone()
        for key in (
            "value_loss", "value_sum", "value_target_sum", "target_entropy_sum",
            "policy_loss", "policy_clip", "emit_policy_clip",
            "thought_policy_clip", "thought_gate_policy_clip", "gate_kl_sum",
            "gate_entropy_sum", "gate_entropy_bonus",
            "emit_probability_sum",
            "positive_lm", "renderer_kl_sum", "thought_kl_sum",
            "advantage_sum", "advantage_square_sum",
            "optional_think_advantage_sum", "forced_initial_think_advantage_sum",
            "thought_advantage_sum", "emit_advantage_sum", "target_square_sum",
            "residual_sum", "residual_square_sum",
            "joint_abs_log_ratio_max", "thought_dim_abs_log_ratio_max",
            "harmful_positive_log_ratio_max",
            "thought_log_sigma_sum", "thought_log_sigma_square_sum",
            "thought_sigma_sum", "thought_expected_noise_norm_sum",
            "thought_realized_noise_norm_sum", "thought_normalized_noise_square_sum",
            "thought_mean_norm_sum",
        )
    }
    totals["thought_log_sigma_min"] = zero.new_full((), float("inf"))
    totals["thought_log_sigma_max"] = zero.new_full((), float("-inf"))

    for microbatch, _, _ in iter_length_aware_microbatches(
        batch,
        replay_max_trajectories,
        replay_attention_budget,
        replay_bucket,
    ):
        action_counts = microbatch.action_mask.sum(1)
        # GAE is row-separable. Computing it after slicing avoids retaining a
        # full-group tensor while preserving each trajectory exactly.
        lambdas = (
            torch.ones_like(action_counts)
            if value_only
            else length_adaptive_lambda(action_counts, gae_lambda_alpha)
        )
        advantages, _ = generalized_advantage_estimate(
            microbatch.rewards,
            microbatch.old_values,
            microbatch.action_mask,
            lambdas,
        )
        _, value_targets = generalized_advantage_estimate(
            microbatch.rewards,
            microbatch.old_values,
            microbatch.action_mask,
            torch.ones_like(action_counts),
        )
        advantages = advantages.detach()
        value_targets = value_targets.detach()
        local_action_count = microbatch.action_mask.sum()
        value_logits = critic.value_logits(microbatch)
        value_ce = critic.support.cross_entropy(value_logits, value_targets)
        local_value_numerator = (value_ce * microbatch.action_mask).sum()
        weighted_value_loss = (
            local_value_numerator / value_action_denominator.clamp_min(1)
        )
        if not torch.isfinite(weighted_value_loss):
            raise RuntimeError(
                f"non-finite value loss: {float(weighted_value_loss)}"
            )
        weighted_value_loss.backward()
        with torch.no_grad():
            values = critic.support.to_expected_scalar(value_logits)
            # During critic pretraining there is no actor objective, but the
            # diagnostics must still measure calibration against the CURRENT
            # critic.  The MC training target is independent of the baseline
            # at lambda=1; reporting the earlier zero-baseline GAE made every
            # successful action look positively advantageous by construction.
            diagnostic_advantages = (
                value_targets - values if value_only else advantages
            )
            target_probs = critic.support.project(value_targets)
            target_entropy = -(
                target_probs * target_probs.clamp_min(1e-20).log()
            ).sum(-1)
            micro_thought_actions = (
                (microbatch.gate_actions == THINK).float()
                * microbatch.action_mask
            )
            micro_optional_thinks = (
                (microbatch.gate_actions == THINK).float()
                * microbatch.gate_mask
            )
            micro_forced_initial_thinks = (
                micro_thought_actions - micro_optional_thinks
            )
            micro_emits = (
                (microbatch.gate_actions == EMIT).float()
                * microbatch.action_mask
            )
            totals["value_loss"] += local_value_numerator.detach()
            totals["value_sum"] += (values * microbatch.action_mask).sum()
            totals["value_target_sum"] += (
                value_targets * microbatch.action_mask
            ).sum()
            totals["target_square_sum"] += (
                value_targets.square() * microbatch.action_mask
            ).sum()
            residuals = value_targets - values
            totals["residual_sum"] += (
                residuals * microbatch.action_mask
            ).sum()
            totals["residual_square_sum"] += (
                residuals.square() * microbatch.action_mask
            ).sum()
            totals["target_entropy_sum"] += (
                target_entropy * microbatch.action_mask
            ).sum()
            totals["advantage_sum"] += (
                diagnostic_advantages * microbatch.action_mask
            ).sum()
            totals["advantage_square_sum"] += (
                diagnostic_advantages.square() * microbatch.action_mask
            ).sum()
            totals["optional_think_advantage_sum"] += (
                diagnostic_advantages * micro_optional_thinks
            ).sum()
            totals["forced_initial_think_advantage_sum"] += (
                diagnostic_advantages * micro_forced_initial_thinks
            ).sum()
            totals["thought_advantage_sum"] += (
                diagnostic_advantages * micro_thought_actions
            ).sum()
            totals["emit_advantage_sum"] += (
                diagnostic_advantages * micro_emits
            ).sum()
        if value_only:
            continue

        # A stream position is one MDP action. EMIT clips the joint gate+token
        # ratio. THINK clips one ratio per Gaussian dimension while retaining
        # their summed score gradient; its optional gate is clipped once.
        beliefs, predicted, stream_inputs, token_targets = replay_head_inputs(
            wrapper, microbatch
        )
        new_gate_logprobs = wrapper.gate.log_prob(
            microbatch.gate_actions.float(), beliefs
        )

        emit_mask = microbatch.emit_mask.bool()
        emit_features = wrapper.renderer_features(
            stream_inputs[emit_mask], beliefs[emit_mask]
        )
        emit_logits = backbone.logits_from_features(emit_features)
        compact_token_logprobs = (
            emit_logits.float()
            .log_softmax(-1)
            .gather(-1, token_targets[emit_mask][..., None])
            .squeeze(-1)
        )
        new_token_logprobs = torch.zeros_like(microbatch.old_token_logprobs)
        new_token_logprobs[emit_mask] = compact_token_logprobs

        micro_positive = microbatch.reward_scalar >= positive_reward_threshold
        weighted_positive_lm = positive_example_lm_loss(
            new_token_logprobs,
            microbatch.emit_mask,
            micro_positive,
            denominator=positive_token_denominator,
        )

        think_mask = (
            (microbatch.gate_actions == THINK)
            & microbatch.action_mask.bool()
        )
        new_thought_joint = torch.zeros_like(new_token_logprobs)
        old_thought_joint = torch.zeros_like(new_token_logprobs)
        policy_thought_logprobs = None
        compact_old_thought_logprobs = None
        if bool(think_mask.any()):
            thought_means, thought_targets, _ = select_thought_actions(
                microbatch, predicted
            )
            thought_log_sigma = wrapper.transition.predict_log_sigma(
                beliefs[think_mask]
            )
            if thought_pg_coef != 0.0:
                new_thought_logprobs = wrapper.transition.per_dim_log_prob(
                    thought_targets,
                    thought_means,
                    thought_log_sigma,
                )
            else:
                with torch.no_grad():
                    new_thought_logprobs = wrapper.transition.per_dim_log_prob(
                        thought_targets,
                        thought_means.detach(),
                        thought_log_sigma.detach(),
                    )
            new_thought_logprobs = new_thought_logprobs.float()
            compact_old_thought_logprobs = (
                microbatch.old_thought_logprobs[think_mask].float()
            )
            if thought_pg_coef == 0.0:
                policy_thought_logprobs = new_thought_logprobs.detach()
            else:
                policy_thought_logprobs = (
                    new_thought_logprobs.detach()
                    + thought_pg_coef
                    * (new_thought_logprobs - new_thought_logprobs.detach())
                )
            new_thought_joint = new_thought_joint.masked_scatter(
                think_mask, policy_thought_logprobs.sum(-1)
            )
            old_thought_joint = old_thought_joint.masked_scatter(
                think_mask, compact_old_thought_logprobs.sum(-1)
            )

        if gate_pg_coef == 0.0:
            policy_gate_logprobs = new_gate_logprobs.detach()
        else:
            policy_gate_logprobs = (
                new_gate_logprobs.detach()
                + gate_pg_coef
                * (new_gate_logprobs - new_gate_logprobs.detach())
            )
        new_joint_logprobs, old_joint_logprobs = joint_action_logprobs(
            policy_gate_logprobs,
            microbatch.old_gate_logprobs,
            new_token_logprobs,
            microbatch.old_token_logprobs,
            new_thought_joint,
            old_thought_joint,
            microbatch.gate_mask,
            microbatch.emit_mask,
        )
        with torch.no_grad():
            joint_log_ratio = new_joint_logprobs - old_joint_logprobs
            active_joint_log_ratio = joint_log_ratio * microbatch.action_mask
            totals["joint_abs_log_ratio_max"] = torch.maximum(
                totals["joint_abs_log_ratio_max"],
                active_joint_log_ratio.abs().max(),
            )
            # PPO intentionally leaves A<0, ratio>1+epsilon unclipped. This
            # tail statistic directly exposes the branch that overflowed in
            # the first frozen-pool run without changing the objective.
            harmful_positive_log_ratio = torch.where(
                (advantages < 0) & microbatch.action_mask.bool(),
                joint_log_ratio.clamp_min(0),
                torch.zeros_like(joint_log_ratio),
            )
            totals["harmful_positive_log_ratio_max"] = torch.maximum(
                totals["harmful_positive_log_ratio_max"],
                harmful_positive_log_ratio.max(),
            )
        weighted_emit_policy_loss, weighted_emit_policy_clip, _ = (
            clipped_policy_loss(
                policy_gate_logprobs * microbatch.gate_mask
                + new_token_logprobs,
                microbatch.old_gate_logprobs * microbatch.gate_mask
                + microbatch.old_token_logprobs,
                advantages,
                microbatch.emit_mask,
                denominator=policy_action_denominator,
                estimate_kl=False,
            )
        )
        weighted_policy_loss = weighted_emit_policy_loss
        weighted_policy_clip = weighted_emit_policy_clip
        if policy_thought_logprobs is not None:
            (
                weighted_thought_policy_loss,
                weighted_thought_policy_clip,
                weighted_thought_gate_clip,
            ) = (
                per_dimension_thought_policy_loss(
                    policy_gate_logprobs[think_mask],
                    microbatch.old_gate_logprobs[think_mask],
                    policy_thought_logprobs,
                    compact_old_thought_logprobs,
                    advantages[think_mask],
                    microbatch.gate_mask[think_mask],
                    policy_action_denominator,
                )
            )
            weighted_policy_loss = (
                weighted_policy_loss + weighted_thought_policy_loss
            )
            weighted_policy_clip = (
                weighted_policy_clip
                + weighted_thought_policy_clip
                + weighted_thought_gate_clip
            )
            totals["thought_policy_clip"] += weighted_thought_policy_clip
            totals["thought_gate_policy_clip"] += weighted_thought_gate_clip
        totals["emit_policy_clip"] += weighted_emit_policy_clip

        # Normalize over every optional gate decision in the complete
        # optimizer minibatch, matching the policy surrogate's global
        # normalization. The auxiliary objective deliberately updates only
        # the Bernoulli head. Sending a high-coefficient entropy gradient into
        # the shared belief trunk could erase useful state features instead of
        # making the gate use those features less deterministically.
        # gate_pg_coef=0 remains a true gate freeze, including this term.
        weighted_gate_entropy_bonus = (
            gate_entropy_coef
            * gate_pg_coef
            * (
                wrapper.gate.entropy(beliefs.detach())
                * microbatch.gate_mask
            ).sum()
            / gate_action_denominator.clamp_min(1)
        )

        actor_total = (
            weighted_policy_loss
            + positive_lm_weight * weighted_positive_lm
            - weighted_gate_entropy_bonus
        )
        if not torch.isfinite(actor_total):
            raise RuntimeError(
                "non-finite actor loss before optimizer step: "
                f"policy={float(weighted_policy_loss)} "
                f"positive_lm={float(weighted_positive_lm)} "
                f"gate_entropy_bonus={float(weighted_gate_entropy_bonus)} "
            )
        actor_total.backward()
        with torch.no_grad():
            gate_log_ratio = (
                new_gate_logprobs - microbatch.old_gate_logprobs
            )
            renderer_log_ratio = (
                new_token_logprobs - microbatch.old_token_logprobs
            )
            totals["policy_loss"] += weighted_policy_loss.detach()
            totals["policy_clip"] += weighted_policy_clip
            totals["gate_entropy_bonus"] += weighted_gate_entropy_bonus.detach()
            totals["gate_kl_sum"] += (
                (torch.expm1(gate_log_ratio) - gate_log_ratio)
                * microbatch.gate_mask
            ).sum()
            totals["gate_entropy_sum"] += (
                wrapper.gate.entropy(beliefs) * microbatch.gate_mask
            ).sum()
            totals["emit_probability_sum"] += (
                wrapper.gate.emit_logit(beliefs).sigmoid()
                * microbatch.gate_mask
            ).sum()
            totals["renderer_kl_sum"] += (
                (torch.expm1(renderer_log_ratio) - renderer_log_ratio)
                * microbatch.emit_mask
            ).sum()
            totals["positive_lm"] += weighted_positive_lm.detach()
            if bool(think_mask.any()):
                detached_log_sigma = thought_log_sigma.detach().float()
                residual = thought_targets.float() - thought_means.detach().float()
                normalized_residual = residual * (-detached_log_sigma).exp()
                totals["thought_mean_norm_sum"] += (
                    thought_means.detach().float().norm(dim=-1).sum()
                )
                totals["thought_log_sigma_sum"] += detached_log_sigma.sum()
                totals["thought_log_sigma_square_sum"] += (
                    detached_log_sigma.square().sum()
                )
                totals["thought_log_sigma_min"] = torch.minimum(
                    totals["thought_log_sigma_min"], detached_log_sigma.min()
                )
                totals["thought_log_sigma_max"] = torch.maximum(
                    totals["thought_log_sigma_max"], detached_log_sigma.max()
                )
                totals["thought_sigma_sum"] += detached_log_sigma.exp().sum()
                totals["thought_expected_noise_norm_sum"] += (
                    (2.0 * detached_log_sigma).exp().sum(-1).sqrt().sum()
                )
                totals["thought_realized_noise_norm_sum"] += (
                    residual.square().sum(-1).sqrt().sum()
                )
                totals["thought_normalized_noise_square_sum"] += (
                    normalized_residual.square().sum()
                )
                thought_log_ratio = (
                    new_thought_logprobs
                    - microbatch.old_thought_logprobs[think_mask]
                )
                totals["thought_dim_abs_log_ratio_max"] = torch.maximum(
                    totals["thought_dim_abs_log_ratio_max"],
                    thought_log_ratio.abs().max(),
                )
                totals["thought_kl_sum"] += (
                    torch.expm1(thought_log_ratio) - thought_log_ratio
                ).sum()

    action_denom = denominators["action"]
    advantage_mean = totals["advantage_sum"] / action_denom
    advantage_variance = (
        totals["advantage_square_sum"] / action_denom
        - advantage_mean.square()
    ).clamp_min(0)
    value_target_mean = totals["value_target_sum"] / action_denom
    target_variance = (
        totals["target_square_sum"] / action_denom
        - value_target_mean.square()
    ).clamp_min(0)
    residual_mean = totals["residual_sum"] / action_denom
    residual_variance = (
        totals["residual_square_sum"] / action_denom
        - residual_mean.square()
    ).clamp_min(0)
    explained_variance = torch.where(
        target_variance > 1e-12,
        1.0 - residual_variance / target_variance.clamp_min(1e-12),
        torch.zeros_like(target_variance),
    )
    metric_tensors = {
        "value_loss": totals["value_loss"] / action_denom,
        "value_mean": totals["value_sum"] / action_denom,
        "value_target_mean": value_target_mean,
        "value_target_variance": target_variance,
        "value_residual_mean": residual_mean,
        "value_residual_variance": residual_variance,
        "explained_variance": explained_variance,
        "value_excess_ce": (
            totals["value_loss"] / action_denom
            - totals["target_entropy_sum"] / action_denom
        ),
        "think_advantage_mean": (
            totals["optional_think_advantage_sum"]
            / optional_think_actions.sum().clamp_min(1)
        ),
        "forced_initial_think_advantage_mean": (
            totals["forced_initial_think_advantage_sum"]
            / forced_initial_think_actions.sum().clamp_min(1)
        ),
        "thought_advantage_mean": (
            totals["thought_advantage_sum"] / denominators["thought"]
        ),
        "emit_advantage_mean": (
            totals["emit_advantage_sum"] / emit_actions.sum().clamp_min(1)
        ),
        "think_action_count": optional_think_actions.sum(),
        "thought_action_count": thought_actions.sum(),
        "action_count": batch.action_mask.sum(),
        "gate_action_count": batch.gate_mask.sum(),
        "emit_action_count": emit_actions.sum(),
        "forced_initial_think_action_count": forced_initial_think_actions.sum(),
    }
    if value_only:
        metric_tensors["critic_grad_norm"] = gradient_norm_tensor(
            critic.parameters()
        )
        if critic_step:
            optimizers["critic"].step()
        return scalar_tensors_to_floats(metric_tensors)

    # Norms are cumulative across prompt groups; the final group reports the
    # complete pre-step norm. Probes are registered
    # under blocks[-1], so exclude them from the trunk norm by identity.
    probe_parameter_ids = {
        id(parameter)
        for probe in (backbone.policy_probe, backbone.critic_probe)
        for parameter in probe.parameters()
    }
    grad_norms = {
        "trunk_grad_norm": gradient_norm_tensor(
            parameter
            for parameter in backbone.parameters()
            if id(parameter) not in probe_parameter_ids
        ),
        "renderer_grad_norm": gradient_norm_tensor(
            backbone.policy_probe.parameters()
        ),
        "adapter_grad_norm": gradient_norm_tensor(wrapper.adapter.parameters()),
        "critic_grad_norm": gradient_norm_tensor(critic.parameters()),
        "gate_grad_norm": gradient_norm_tensor(wrapper.gate.parameters()),
        "sigma_grad_norm": gradient_norm_tensor(
            wrapper.transition.log_sigma_head.parameters()
        ),
        "thought_mean_grad_norm": gradient_norm_tensor(
            wrapper.transition.mean_head.parameters()
        ),
    }
    if critic_step:
        optimizers["critic"].step()
    if actor_step and "actor" in optimizers:
        optimizers["actor"].step()
    metric_tensors.update(
        policy_loss=totals["policy_loss"],
        policy_clip_fraction=(
            totals["policy_clip"]
            * policy_action_denominator.clamp_min(1)
            / (
                policy_action_denominator
                + optional_think_actions.sum()
            ).clamp_min(1)
        ),
        emit_policy_clip_fraction=(
            totals["emit_policy_clip"]
            * policy_action_denominator.clamp_min(1)
            / denominators["emit"]
        ),
        thought_policy_clip_fraction=(
            totals["thought_policy_clip"]
            * policy_action_denominator.clamp_min(1)
            / denominators["thought"]
        ),
        thought_gate_policy_clip_fraction=(
            totals["thought_gate_policy_clip"]
            * policy_action_denominator.clamp_min(1)
            / optional_think_actions.sum().clamp_min(1)
        ),
        gate_entropy=totals["gate_entropy_sum"] / denominators["gate"],
        gate_entropy_bonus=totals["gate_entropy_bonus"],
        emit_probability=totals["emit_probability_sum"] / denominators["gate"],
        positive_lm_loss=totals["positive_lm"],
        positive_fraction=positive.float().mean(),
        gate_behavior_kl=totals["gate_kl_sum"] / denominators["gate"],
        renderer_behavior_kl=totals["renderer_kl_sum"] / denominators["emit"],
        thought_behavior_kl_joint=(
            totals["thought_kl_sum"] / denominators["thought"]
        ),
        thought_behavior_kl_per_dim=(
            totals["thought_kl_sum"]
            / (denominators["thought"] * batch.old_thought_logprobs.size(-1))
        ),
        policy_behavior_kl_per_action=(
            totals["gate_kl_sum"]
            + totals["renderer_kl_sum"]
            + totals["thought_kl_sum"]
        ) / policy_action_denominator.clamp_min(1),
        joint_abs_log_ratio_max=totals["joint_abs_log_ratio_max"],
        thought_dim_abs_log_ratio_max=(
            totals["thought_dim_abs_log_ratio_max"]
        ),
        harmful_positive_log_ratio_max=(
            totals["harmful_positive_log_ratio_max"]
        ),
        advantage_mean=advantage_mean,
        advantage_std=advantage_variance.sqrt(),
        reward=batch.reward_scalar.mean(),
        thought_log_sigma_mean=(
            totals["thought_log_sigma_sum"]
            / (denominators["thought"] * batch.old_thought_logprobs.size(-1))
        ),
        thought_log_sigma_std=(
            totals["thought_log_sigma_square_sum"]
            / (denominators["thought"] * batch.old_thought_logprobs.size(-1))
            - (
                totals["thought_log_sigma_sum"]
                / (denominators["thought"] * batch.old_thought_logprobs.size(-1))
            ).square()
        ).clamp_min(0).sqrt(),
        thought_log_sigma_min=totals["thought_log_sigma_min"],
        thought_log_sigma_max=totals["thought_log_sigma_max"],
        thought_sigma_mean=(
            totals["thought_sigma_sum"]
            / (denominators["thought"] * batch.old_thought_logprobs.size(-1))
        ),
        thought_expected_noise_norm=(
            totals["thought_expected_noise_norm_sum"] / denominators["thought"]
        ),
        thought_realized_noise_norm=(
            totals["thought_realized_noise_norm_sum"] / denominators["thought"]
        ),
        thought_normalized_noise_rms=(
            totals["thought_normalized_noise_square_sum"]
            / (denominators["thought"] * batch.old_thought_logprobs.size(-1))
        ).sqrt(),
        thought_mean_norm=(
            totals["thought_mean_norm_sum"] / denominators["thought"]
        ),
        thought_mean_weight_rms=(
            wrapper.transition.mean_head.output_gain.detach().abs()
            * wrapper.transition.mean_head.weight.detach().square().mean().sqrt()
        ),
        thought_mean_bias_rms=(
            wrapper.transition.mean_head.bias.detach().square().mean().sqrt()
        ),
        thought_mean_output_gain=(
            wrapper.transition.mean_head.output_gain.detach()
        ),
        thought_adapter_weight_rms=(
            wrapper.adapter.projection.weight.detach().square().mean().sqrt()
        ),
        thought_adapter_bias_rms=(
            wrapper.adapter.projection.bias.detach().square().mean().sqrt()
        ),
        thought_log_sigma_raw_bias_mean=(
            wrapper.transition.log_sigma_head.bias.detach().mean()
        ),
        thought_log_sigma_weight_rms=(
            wrapper.transition.log_sigma_head.residual_gain.detach().abs()
            * wrapper.transition.log_sigma_head.weight.detach().square().mean().sqrt()
        ),
        thought_log_sigma_residual_gain=(
            wrapper.transition.log_sigma_head.residual_gain.detach()
        ),
        **grad_norms,
    )
    metrics = scalar_tensors_to_floats(metric_tensors)
    metrics.update(
        gate_pg_coef=float(gate_pg_coef),
        thought_pg_coef=float(thought_pg_coef),
        positive_lm_weight=float(positive_lm_weight),
        gate_entropy_coef=float(gate_entropy_coef),
    )
    return metrics


@torch.no_grad()
def measure_post_update_policy_drift(
    wrapper: LatentThoughtModel,
    batches: list[LatentRolloutBatch],
    *,
    replay_max_trajectories: int,
    replay_attention_budget: int,
    replay_bucket: int,
    replay_function: Callable = replay_head_inputs,
) -> dict[str, float]:
    """Evaluate the just-updated policy on its behavior trajectories.

    On behavior-age 0, optimized ratios are exactly one before the first actor
    step; clipping therefore cannot reveal how far that step moved the deployed
    policy. This read-only replay measures the actual post-step drift
    without reusing trajectories for a gradient. It runs only at the explicit
    diagnostic cadence because it costs one additional policy forward.
    ``replay_function`` deliberately stays eager in the trainer: invoking the
    grad-enabled training artifact under this function's no-grad context would
    force Dynamo/AOTAutograd to compile a second guarded graph, while enabling
    gradients here retains a full replay graph and can exceed peak memory.
    """
    if not batches:
        raise ValueError("post-update drift requires at least one rollout batch")
    device = next(wrapper.parameters()).device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    totals = {
        "gate_kl": zero.clone(),
        "renderer_kl": zero.clone(),
        "thought_kl": zero.clone(),
        "joint_abs_log_ratio_max": zero.clone(),
        "gate_count": zero.clone(),
        "emit_count": zero.clone(),
        "thought_count": zero.clone(),
        "action_count": zero.clone(),
    }
    for stored_batch in batches:
        # Actor behavior pools live in ordinary CPU memory so a 64-prompt
        # frozen-policy pool does not retain many GiB of dense fp32 thoughts
        # and old per-dimension Gaussian log-probabilities on the GPU. Stream
        # one prompt group at a time for this infrequent diagnostic, exactly
        # as the gradient-bearing update path does below.
        batch = (
            stored_batch
            if stored_batch.kind.device == device
            else stored_batch.to(device)
        )
        for microbatch, _, _ in iter_length_aware_microbatches(
            batch,
            replay_max_trajectories,
            replay_attention_budget,
            replay_bucket,
        ):
            beliefs, predicted, stream_inputs, token_targets = replay_function(
                wrapper, microbatch
            )
            gate_logprobs = wrapper.gate.log_prob(
                microbatch.gate_actions.float(), beliefs
            ).float()
            gate_log_ratio = gate_logprobs - microbatch.old_gate_logprobs.float()

            emit_mask = microbatch.emit_mask.bool()
            token_logprobs = torch.zeros_like(microbatch.old_token_logprobs).float()
            if bool(emit_mask.any()):
                emit_logits = wrapper.backbone.logits_from_features(
                    wrapper.renderer_features(
                        stream_inputs[emit_mask], beliefs[emit_mask]
                    )
                )
                compact_token_logprobs = (
                    emit_logits.float()
                    .log_softmax(-1)
                    .gather(-1, token_targets[emit_mask][..., None])
                    .squeeze(-1)
                )
                token_logprobs[emit_mask] = compact_token_logprobs
            token_log_ratio = (
                token_logprobs - microbatch.old_token_logprobs.float()
            )

            think_mask = (
                (microbatch.gate_actions == THINK)
                & microbatch.action_mask.bool()
            )
            thought_joint = torch.zeros_like(token_logprobs)
            old_thought_joint = torch.zeros_like(token_logprobs)
            thought_log_ratio = None
            if bool(think_mask.any()):
                thought_means, thought_targets, _ = select_thought_actions(
                    microbatch, predicted
                )
                thought_log_sigma = wrapper.transition.predict_log_sigma(
                    beliefs[think_mask]
                )
                thought_logprobs = wrapper.transition.per_dim_log_prob(
                    thought_targets, thought_means, thought_log_sigma
                ).float()
                old_thought_logprobs = microbatch.old_thought_logprobs[
                    think_mask
                ].float()
                thought_log_ratio = thought_logprobs - old_thought_logprobs
                thought_joint[think_mask] = thought_logprobs.sum(-1)
                old_thought_joint[think_mask] = old_thought_logprobs.sum(-1)

            new_joint, old_joint = joint_action_logprobs(
                gate_logprobs,
                microbatch.old_gate_logprobs.float(),
                token_logprobs,
                microbatch.old_token_logprobs.float(),
                thought_joint,
                old_thought_joint,
                microbatch.gate_mask,
                microbatch.emit_mask,
            )
            joint_log_ratio = new_joint - old_joint
            action_mask = microbatch.action_mask.float()
            totals["joint_abs_log_ratio_max"] = torch.maximum(
                totals["joint_abs_log_ratio_max"],
                (joint_log_ratio * action_mask).abs().max(),
            )
            totals["gate_kl"] += (
                (torch.expm1(gate_log_ratio) - gate_log_ratio)
                * microbatch.gate_mask
            ).sum()
            totals["renderer_kl"] += (
                (torch.expm1(token_log_ratio) - token_log_ratio)
                * microbatch.emit_mask
            ).sum()
            if thought_log_ratio is not None:
                totals["thought_kl"] += (
                    torch.expm1(thought_log_ratio) - thought_log_ratio
                ).sum()
            totals["gate_count"] += microbatch.gate_mask.sum()
            totals["emit_count"] += microbatch.emit_mask.sum()
            totals["thought_count"] += think_mask.sum()
            totals["action_count"] += action_mask.sum()
        del batch

    return scalar_tensors_to_floats(
        {
            "kl/post_update_gate_behavior": (
                totals["gate_kl"] / totals["gate_count"].clamp_min(1)
            ),
            "kl/post_update_renderer_behavior": (
                totals["renderer_kl"] / totals["emit_count"].clamp_min(1)
            ),
            "kl/post_update_thought_behavior_joint": (
                totals["thought_kl"] / totals["thought_count"].clamp_min(1)
            ),
            "kl/post_update_policy_behavior_per_action": (
                (
                    totals["gate_kl"]
                    + totals["renderer_kl"]
                    + totals["thought_kl"]
                )
                / totals["action_count"].clamp_min(1)
            ),
            "ratio/post_update_joint_abs_log_max": totals[
                "joint_abs_log_ratio_max"
            ],
        }
    )


def save_checkpoint(
    path: Path,
    wrapper: LatentThoughtModel,
    critic: SeparateCritic,
    optimizers: dict[str, torch.optim.Optimizer],
    step: int,
    args: argparse.Namespace,
    sampler: MathPromptSampler,
    warmup_step: int,
    actor_init_provenance: dict | None = None,
) -> None:
    payload = {
        "step": step,
        "value_warmup_step": warmup_step,
        "execution_schema": EXECUTION_SCHEMA,
        "prompt_order_schema": PROMPT_ORDER_SCHEMA,
        "math_data_identity": sampler.dataset_identity,
        "reward_schema": REWARD_SCHEMA,
        "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
        "rollout_policy_schema": ROLLOUT_POLICY_SCHEMA,
        "thought_input_schema": THOUGHT_INPUT_SCHEMA,
        "thought_distribution_schema": THOUGHT_DISTRIBUTION_SCHEMA,
        "thought_mean_schema": THOUGHT_MEAN_SCHEMA,
        "model": wrapper.state_dict(),
        "critic": critic.state_dict(),
        "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
        "args": vars(args),
        "sampler_cursor": sampler.cursor,
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(),
        "actor_init_provenance": actor_init_provenance,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def purge_benchmark_reports_after(output: Path, step: int) -> int:
    """Remove stale answer histories and restore latest to checkpoint time."""
    history_dir = output / "bench_answers"
    removed = 0
    if history_dir.exists():
        for path in history_dir.glob("step_*.json"):
            try:
                history_step = int(path.stem.removeprefix("step_"))
            except ValueError:
                continue
            if history_step > step:
                path.unlink()
                removed += 1
    remaining = sorted(history_dir.glob("step_*.json")) if history_dir.exists() else []
    if remaining:
        payload = json.loads(remaining[-1].read_text())
        write_benchmark_report(
            output,
            int(payload["step"]),
            payload["metrics"],
            payload["attempts"],
            reward_schema=str(payload.get("reward_schema", REWARD_SCHEMA)),
        )
    else:
        for path in (history_dir / "latest.json", output / "bench_answers.html"):
            if path.exists():
                path.unlink()
    return removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--math-data", default="postraining/data/dapo-math-17k.parquet")
    parser.add_argument(
        "--exclude-modules", default="",
        help="comma-separated extra_info.module names to drop from --math-data "
        "(module-tagged datasets only); names absent from the data are an error",
    )
    # Prompt budget: DAPO prompts longer than this keep their TAIL (the
    # question and answer-format instruction sit at the end).
    parser.add_argument(
        "--prompt-tokens", type=int, default=POSTTRAIN_PROMPT_TOKENS
    )
    parser.add_argument(
        "--continuation-tokens", type=int, default=POSTTRAIN_RESPONSE_TOKENS
    )
    # Compute-scaled VAPO topology: sample n=16 responses for 64 prompts under
    # one frozen behavior policy, then shuffle prompt groups once and take four
    # disjoint B256 optimizer minibatches. This activates PPO's behavior-policy
    # clipping without trajectory reuse. VAPO uses the same n=16 but a larger
    # 512-prompt / 8192-trajectory pool and B512 minibatches.
    parser.add_argument("--prompts-per-rollout", type=int, default=64)
    parser.add_argument("--prompts-per-minibatch", type=int, default=16)
    parser.add_argument("--samples-per-prompt", type=int, default=16)
    # One pass only: repeated PPO epochs reuse the same generated trajectories.
    parser.add_argument("--ppo-epochs", type=int, default=1)
    # Total generated-slot budget per trajectory (thinks + emits).  Thinking
    # is never forcibly interrupted; overthinking costs emitted tokens and
    # therefore reward. 0 means 4x the emit cap; the default is the explicit
    # 4096-slot side of the 1024-prompt + 4096-stream context contract.
    parser.add_argument(
        "--max-stream-steps", type=int, default=POSTTRAIN_STREAM_TOKENS
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    # One general rate for actor and critic. The fresh-policy experiment starts
    # every actor-side optimizer state empty from the critic-warm checkpoint;
    # using the critic's 3e-4 rate also removes the prior hand-tuned split
    # between trunk, renderer, gate, adapter, and continuous-policy heads.
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-bins", type=int, default=101)
    # HL-Gauss projection sigma as a fraction of bin width (cleanrl v215 /
    # Dreamer4 default).
    parser.add_argument("--value-sigma-ratio", type=float, default=2.0)
    # Head bias starts at the projected prior; binary verifier rewards start
    # near-zero for a small model, and a prior near the expected reward mean
    # removes the early decode transient a far-off prior causes.
    parser.add_argument("--value-prior", type=float, default=0.05)
    # Initialization only: the bounded output of the state-dependent
    # log-sigma head. Zero-init weights make noise state-independent at step 0;
    # -2 gives std 0.135 and expected 512-D noise norm 3.06. The fresh
    # adapter starts at exact zero, so neither exploration nor the small fresh
    # mean enters the recurrent trunk until replay gradients open its affine map.
    parser.add_argument("--thought-log-sigma-init", type=float, default=-2.0)
    # Gradient multiplier for the thought factor inside the joint action log
    # probability. The forward ratio stays exact; 0 detaches only that factor
    # as a control arm while token/gate gradients still train the trunk.
    parser.add_argument("--thought-pg-coef", type=float, default=1.0)
    # Head-only Bernoulli entropy bonus, averaged over optional gate
    # decisions. The 4e-3 default remains far below v13's 3e-2 intervention,
    # which overwhelmed the learned gate despite falling reward.
    parser.add_argument("--gate-entropy-coef", type=float, default=4e-3)
    # Freeze the gate for the first N steps: the gate learns "don't think"
    # from a clean binary signal far faster than the 512-dim thought content
    # can learn to be useful, so exploration dies before content training
    # bites (v3: think fraction 0.094 -> 0.04 by step 700 with think
    # advantage pinned negative).  Holding the gate at its init probability
    # gives the thought/adapter channels a head start; gate PPO switches on
    # at step N against whatever thought quality that warmup earned.
    parser.add_argument("--gate-freeze-steps", type=int, default=0)
    # Length-adaptive GAE lambda (core.length_adaptive_lambda): VAPO's
    # horizon alpha*l with a floor of min(l, 1/alpha).  The raw alpha=0.05
    # formula clamps lambda to 0 at this run's 12-23-action trajectories
    # (TD(0): terminal reward credits nothing more than one step back —
    # audited as the think-fraction ratchet); the floor restores
    # whole-trajectory credit for short answers while keeping VAPO's
    # variance control for long ones.
    parser.add_argument("--gae-lambda-alpha", type=float, default=0.05)
    # Initial THINK probability via the gate head's bias (weights stay zero,
    # so the gate is still belief-independent at init).  The uniform 50/50
    # start is reward-starved at cold start (measured Jul 18): interleaving
    # untrained noisy thoughts into half the stream collapses bench accuracy
    # from ~14% emit-only to ~1e-4, leaving every PPO advantage at critic
    # noise.  Starting emit-heavy keeps rollouts in the competent regime
    # (abundant within-group reward variance) while the gate stays free to
    # think more wherever it pays.
    parser.add_argument("--init-think-probability", type=float, default=0.1)
    parser.add_argument("--positive-lm-weight", type=float, default=0.1)
    parser.add_argument("--positive-reward-threshold", type=float, default=0.5)
    parser.add_argument(
        "--nearby-reward-max",
        type=float,
        default=0.1,
        help="maximum reward for a wrong, terminated numeric final answer",
    )
    # VAPO paper: 50 value-pretraining steps before policy updates.
    parser.add_argument("--value-warmup-steps", type=int, default=50)
    # The BPB guard is a do-no-harm regression check, not an optimization
    # target. A deterministic 2M-token prefix takes ~2.5s on the 5090 versus
    # ~75s for all 62M validation tokens, while still giving the guard ample
    # precision to detect renderer drift. Use 0 explicitly for a final,
    # challenge-comparable full-validation measurement.
    parser.add_argument(
        "--bpb-every", type=int, default=DEFAULT_PERIODIC_EVAL_EVERY
    )
    parser.add_argument(
        "--bpb-val-tokens", type=int, default=DEFAULT_BPB_GUARD_TOKENS,
        help="deterministic validation-prefix size for the BPB guard "
        f"(default: {DEFAULT_BPB_GUARD_TOKENS}; 0 = full set); guard values "
        "remain comparable only within one setting",
    )
    parser.add_argument(
        "--bpb-only", action="store_true",
        help="evaluate the BPB guard once and exit (for timing/config checks)",
    )
    parser.add_argument(
        "--bench-only", action="store_true",
        help="evaluate the full held-out benchmark and exit; requires a "
        "dedicated --output when used with --resume",
    )
    parser.add_argument("--bench-only-repeats", type=int, default=1)
    # The full 1024-answer / 4096-stream AIME eval is intentionally final-only
    # by default; periodic copies would dominate the posttraining workload.
    parser.add_argument("--aime-every", type=int, default=0)
    parser.add_argument("--aime-data", default="postraining/data/aime-2024.parquet")
    parser.add_argument("--aime-samples", type=int, default=32)
    parser.add_argument(
        "--aime-max-tokens", type=int, default=POSTTRAIN_RESPONSE_TOKENS
    )
    # Rollout positions are sequential, so batching all samples of a problem
    # into one rollout is nearly free parallelism; lower this only if VRAM
    # becomes the constraint.
    parser.add_argument("--aime-chunk", type=int, default=32)
    # The easier benchmark of record alongside AIME: held-out DeepMind
    # interpolate problems at exactly the mathmix QA training difficulty,
    # where a 27M model can show a real accuracy curve.
    parser.add_argument(
        "--bench-data", default="postraining/data/deepmind-interpolate-easy.parquet"
    )
    parser.add_argument(
        "--bench-every", type=int, default=DEFAULT_PERIODIC_EVAL_EVERY
    )
    parser.add_argument("--bench-samples", type=int, default=8)
    parser.add_argument(
        "--bench-max-rows",
        type=int,
        default=0,
        help="fixed hash-selected benchmark prompt count (0 evaluates all)",
    )
    parser.add_argument(
        "--bench-max-tokens", type=int, default=POSTTRAIN_RESPONSE_TOKENS
    )
    # Batch multiple problem groups into the same left-padded GPU rollout.
    # This is a trajectory rather than prompt count so avg@8 and avg@32 use
    # comparable memory; replay-free eval storage makes 128 rows practical.
    parser.add_argument("--eval-batch-trajectories", type=int, default=128)
    parser.add_argument(
        "--eval-tail-batch", type=int, default=16,
        help="single compiled survivor-batch size (0 disables compaction)",
    )
    # Compile the dynamic-prefix one-token model step used only by eval.
    # Static full-cache CUDA graphs are intentionally avoided: measured
    # attention over all 5K cache slots was 2.6x slower than eager narrowing.
    parser.add_argument(
        "--eval-compile", action=argparse.BooleanOptionalAction, default=True
    )
    # Compile the narrow-cache one-token training rollout exactly like eval.
    # Tensor-valued positions keep one graph valid across the whole stream;
    # stable batch shapes avoid recompilation at termination boundaries.
    parser.add_argument(
        "--rollout-compile", action=argparse.BooleanOptionalAction, default=True
    )
    # Checkpoint in true actor/critic optimizer-update units.
    parser.add_argument("--save-every", type=int, default=32)
    parser.add_argument("--warmup-save-every", type=int, default=10)
    # Compile the compute-bound parallel replay surfaces independently of the
    # narrow-cache rollout. The rejected old path coupled compilation to
    # full-5K static attention, which measured 2.6x slower.
    parser.add_argument(
        "--compile-replay", action=argparse.BooleanOptionalAction, default=True
    )
    # Stable length-sorted shards are bounded by B*L^2 attention area rather
    # than a fixed row count. This admits all 32 normal ~150-token rows and
    # automatically isolates rare 1K-4K outliers.
    parser.add_argument("--replay-bucket", type=int, default=64)
    parser.add_argument("--replay-max-trajectories", type=int, default=32)
    parser.add_argument(
        "--replay-attention-budget", type=int, default=4 * 1024 * 1024
    )
    parser.add_argument(
        "--post-update-kl-every",
        type=int,
        default=100,
        help="read-only same-batch policy-drift replay cadence; step 1 is "
        "always measured, 0 disables later measurements",
    )
    # Prompt groups rolled out together as one left-padded batch (measured:
    # the sequential per-group rollout is launch-bound at ~140 W, so stepping
    # groups*samples rows per launch is the utilization lever.
    parser.add_argument("--rollout-groups", type=int, default=16)
    parser.add_argument("--rollout-only", action="store_true")
    parser.add_argument("--gate-min-within-group-reward-std", type=float, default=0.01)
    parser.add_argument(
        "--actor-init", default=None,
        help="initialize only the actor from a latent-VAPO checkpoint, while "
        "resetting critic/optimizers and continuing its unused prompt stream",
    )
    parser.add_argument(
        "--actor-critic-init", default=None,
        help="initialize actor and pretrained critic weights from a warmup "
        "checkpoint, preserve optimizer state, and continue its unused prompt stream",
    )
    parser.add_argument(
        "--curriculum-init",
        default=None,
        help="initialize trained actor weights from another reward/dataset, "
        "while resetting critic, optimizers, step, and target prompt cursor",
    )
    parser.add_argument(
        "--consume-all-prompts",
        action="store_true",
        help="consume the target dataset exactly once, allowing one final "
        "undersized optimizer minibatch rather than dropping remainder rows",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--migrate-zero-adapter-resume",
        action="store_true",
        help="explicitly resume a v15 gain-scaled checkpoint while replacing "
        "only its adapter with the v16 zero-initialized affine and fresh "
        "adapter Adam state",
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.replay_max_trajectories < 1:
        parser.error("--replay-max-trajectories must be positive")
    if not math.isfinite(args.gate_entropy_coef) or args.gate_entropy_coef < 0.0:
        parser.error("--gate-entropy-coef must be finite and nonnegative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        parser.error("--learning-rate must be finite and positive")
    if (
        not math.isfinite(args.nearby_reward_max)
        or args.nearby_reward_max < 0.0
        or args.nearby_reward_max >= args.positive_reward_threshold
    ):
        parser.error(
            "--nearby-reward-max must be finite, nonnegative, and below "
            "--positive-reward-threshold"
        )
    if not -5.0 < args.thought_log_sigma_init < 2.0:
        parser.error("--thought-log-sigma-init must be strictly inside (-5, 2)")
    if args.replay_attention_budget < 1:
        parser.error("--replay-attention-budget must be positive")
    if args.post_update_kl_every < 0:
        parser.error("--post-update-kl-every must be nonnegative")
    if args.replay_bucket < 1:
        parser.error("--replay-bucket must be positive")
    if args.warmup_save_every < 1:
        parser.error("--warmup-save-every must be positive")
    if args.eval_batch_trajectories < 1:
        parser.error("--eval-batch-trajectories must be positive")
    if args.eval_tail_batch < 0:
        parser.error("--eval-tail-batch must be nonnegative")
    if args.bench_only_repeats < 1:
        parser.error("--bench-only-repeats must be positive")
    if args.bench_max_rows < 0:
        parser.error("--bench-max-rows must be nonnegative")
    if args.migrate_zero_adapter_resume and not args.resume:
        parser.error("--migrate-zero-adapter-resume requires --resume")
    if args.prompts_per_rollout < 1:
        parser.error("--prompts-per-rollout must be positive")
    if args.prompts_per_minibatch < 1:
        parser.error("--prompts-per-minibatch must be positive")
    if args.prompts_per_rollout % args.prompts_per_minibatch:
        parser.error(
            "--prompts-per-rollout must be divisible by --prompts-per-minibatch"
        )
    if args.ppo_epochs != 1:
        parser.error("--ppo-epochs must be 1; trajectory reuse is disabled")
    if args.bpb_val_tokens < 0:
        parser.error("--bpb-val-tokens must be nonnegative")
    exclusive_modes = sum(
        (args.bpb_only, args.bench_only, args.rollout_only)
    )
    if exclusive_modes > 1:
        parser.error(
            "--bpb-only, --bench-only, and --rollout-only are mutually exclusive"
        )
    if (
        args.bench_only
        and args.resume
        and Path(args.output).resolve() == Path(args.resume).resolve().parent
    ):
        parser.error(
            "--bench-only with --resume requires a dedicated --output so "
            "evaluation cannot purge or overwrite the training run"
        )
    initialization_modes = sum(
        option is not None
        for option in (
            args.actor_init,
            args.actor_critic_init,
            args.curriculum_init,
            args.resume,
        )
    )
    if initialization_modes > 1:
        parser.error(
            "--actor-init, --actor-critic-init, --curriculum-init, and "
            "--resume are mutually exclusive"
        )
    if args.samples_per_prompt < 2 or args.samples_per_prompt % 2:
        parser.error("--samples-per-prompt must be even for the 50/50 forced split")
    if args.prompts_per_minibatch * args.samples_per_prompt != 256:
        parser.error(
            "optimizer minibatches must contain exactly 256 trajectories"
        )
    for name, samples in (
        ("--aime-samples", args.aime_samples),
        ("--bench-samples", args.bench_samples),
    ):
        if samples < 2 or samples % 2:
            parser.error(f"{name} must be even for the 50/50 forced split")
    if (
        (args.bench_every > 0 or args.bench_only)
        and args.bench_samples < CAPTURE_SAMPLES_PER_PROBLEM
    ):
        parser.error(
            "--bench-samples must be at least "
            f"{CAPTURE_SAMPLES_PER_PROBLEM} for automatic answer capture"
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    max_stream_steps = args.max_stream_steps or 4 * args.continuation_tokens
    # The eval budget always scales with its own emit cap; an explicit
    # --max-stream-steps is a training-rollout knob.
    aime_stream_steps = 4 * args.aime_max_tokens
    bench_stream_steps = 4 * args.bench_max_tokens
    validate_posttraining_context_budget(args.prompt_tokens, max_stream_steps)
    validate_posttraining_context_budget(args.prompt_tokens, aime_stream_steps)
    validate_posttraining_context_budget(args.prompt_tokens, bench_stream_steps)

    backbone = load_model(args.checkpoint, device)
    if not backbone.architecture.endswith(
        "probes_pope_belief_attached_ce_onepass_2k"
    ):
        raise ValueError(
            "belief-renderer VAPO requires a checkpoint pretrained with "
            "[token_latent, raw_belief] CE; old predicted-latent probe "
            f"architecture {backbone.architecture!r} is incompatible"
        )
    backbone.eval()
    # Full-model RL: every parameter on a deployed policy path trains — trunk,
    # embeddings, fresh thought mean/sigma, renderer, gate, and adapter. The
    # retired pretrained prediction projector remains checkpointed but has no
    # graph edge; the backbone critic probe is frozen and unused.
    wrapper = LatentThoughtModel(backbone).to(device)
    # No module here behaves differently under train(): pin eval mode once so
    # the training flag (a dynamo guard) never flips between the step-0 evals
    # and the training loop and re-specializes the compiled step.
    wrapper.eval()
    if not 0.0 < args.init_think_probability < 1.0:
        raise SystemExit("--init-think-probability must be strictly inside (0, 1)")
    # The CLI owns the starting exploration noise: state-independent (zero
    # head weights) at the requested level.  Learned from there — no schedule.
    wrapper.transition.reset_noise(args.thought_log_sigma_init)
    with torch.no_grad():
        # P(EMIT) = sigmoid(bias) while the zero-init weights ignore the belief.
        wrapper.gate.head.bias.fill_(
            math.log((1.0 - args.init_think_probability) / args.init_think_probability)
        )
    actor_init_payload = None
    actor_init_provenance = None
    initialization_path = (
        args.actor_critic_init or args.actor_init or args.curriculum_init
    )
    if initialization_path:
        actor_init_payload = torch.load(
            initialization_path, map_location="cpu", weights_only=False
        )
        if actor_init_payload.get("prompt_order_schema") != PROMPT_ORDER_SCHEMA:
            raise ValueError(
                "initialization checkpoint predates deterministic sequential "
                "prompt traversal; its cursor cannot prove no prompt reuse"
            )
        if args.actor_critic_init:
            if int(actor_init_payload.get("step", -1)) != 0:
                raise ValueError(
                    "--actor-critic-init requires a pre-policy critic-warmup checkpoint"
                )
            if (
                int(actor_init_payload.get("value_warmup_step", -1))
                < args.value_warmup_steps
            ):
                raise ValueError(
                    "--actor-critic-init checkpoint has incomplete critic warmup"
                )
            if actor_init_payload.get("reward_schema") != REWARD_SCHEMA:
                raise ValueError(
                    "--actor-critic-init checkpoint uses an incompatible reward schema"
                )
        migrate_legacy_wrapper_checkpoint(
            actor_init_payload,
            wrapper,
            initialize_fresh_mean=bool(args.actor_critic_init),
            initialize_fresh_adapter=bool(args.actor_critic_init),
        )
        validate_renderer_checkpoint(
            actor_init_payload,
            initialization_path,
            # A critic-warmup checkpoint has not updated the actor, and its
            # transition head is reset below before the policy is ever used.
            allow_transition_reset=bool(args.actor_critic_init),
        )
        wrapper.load_state_dict(actor_init_payload["model"], strict=True)
        if args.actor_critic_init:
            # A critic-warmup checkpoint has never trained its actor, so this
            # run owns the initial exploration level. A trained --actor-init
            # source instead preserves its learned state-dependent noise.
            wrapper.transition.reset_noise(args.thought_log_sigma_init)
        actor_init_provenance = {
            "checkpoint": str(initialization_path),
            "critic_initialized": bool(args.actor_critic_init),
            "curriculum_transition": bool(args.curriculum_init),
            "source_step": actor_init_payload.get("step"),
            "source_reward_schema": actor_init_payload.get("reward_schema"),
            "source_execution_schema": actor_init_payload.get("execution_schema"),
            "source_value_warmup_step": actor_init_payload.get(
                "value_warmup_step"
            ),
            "source_sampler_cursor": int(actor_init_payload["sampler_cursor"]),
            "target_sampler_cursor": 0 if args.curriculum_init else int(
                actor_init_payload["sampler_cursor"]
            ),
            "fresh_mean_initialized": bool(args.actor_critic_init),
            "fresh_adapter_initialized": bool(args.actor_critic_init),
            "fresh_adapter_zero_initialized": bool(args.actor_critic_init),
        }
    for parameter in wrapper.parameters():
        parameter.requires_grad_(True)
    for parameter in backbone.critic_probe.parameters():
        parameter.requires_grad_(False)

    critic = SeparateCritic(
        fresh_trunk(backbone, device),
        num_bins=args.value_bins,
        sigma_ratio=args.value_sigma_ratio,
        prior_value=args.value_prior,
    ).to(device)
    critic.eval()  # no dropout in this architecture; keep norms deterministic
    if args.actor_critic_init:
        critic.load_state_dict(actor_init_payload["critic"], strict=True)

    optimizers = build_optimizers(
        wrapper, critic, learning_rate=args.learning_rate,
    )
    if args.actor_critic_init:
        # A step-0 critic-warm checkpoint has never stepped its actor. Its
        # optimizer state is therefore empty and carries no momentum to
        # preserve; loading its obsolete three-group layout would only couple
        # the gate and recurrent adapter again. Preserve the trained critic's
        # optimizer state and start the untouched actor optimizer in the
        # current six-group layout.
        source_actor_optimizer = actor_init_payload["optimizers"]["actor"]
        if source_actor_optimizer["state"]:
            raise ValueError(
                "--actor-critic-init requires a pristine actor optimizer"
            )
        optimizers["critic"].load_state_dict(
            actor_init_payload["optimizers"]["critic"]
        )
        for group in optimizers["critic"].param_groups:
            group["lr"] = args.learning_rate

    tokenizer = spm.SentencePieceProcessor(model_file=FreshHyperparameters.tokenizer_path)
    stop_ids = tuple(
        t for t in (tokenizer.eos_id(), tokenizer.bos_id()) if t >= 0
    )
    if not stop_ids:
        raise RuntimeError(
            "posttraining requires a valid BOS or EOS token for explicit "
            "trajectory termination"
        )
    aime_rows = (
        load_unique_math_rows(args.aime_data)
        if args.aime_every > 0 and not args.rollout_only
        else []
    )
    aime_modal_baseline = modal_answer_baseline(aime_rows)
    all_bench_rows = (
        load_unique_math_rows(args.bench_data)
        if (args.bench_every > 0 or args.bench_only) and not args.rollout_only
        else []
    )
    bench_rows = deterministic_math_subset(all_bench_rows, args.bench_max_rows)
    bench_dataset_baseline = modal_answer_baseline(all_bench_rows)
    bench_subset_baseline = modal_answer_baseline(bench_rows)
    math_rows = load_unique_math_rows(args.math_data)
    excluded_modules = {
        name for name in args.exclude_modules.split(",") if name
    }
    if excluded_modules:
        present_modules = {
            (row.get("extra_info") or {}).get("module") for row in math_rows
        }
        missing = sorted(excluded_modules - present_modules)
        if missing:
            raise ValueError(
                f"--exclude-modules names absent from {args.math_data}: {missing}"
            )
        kept_rows = [
            row
            for row in math_rows
            if (row.get("extra_info") or {}).get("module") not in excluded_modules
        ]
        print(
            f"excluded modules {sorted(excluded_modules)}: "
            f"{len(math_rows)} -> {len(kept_rows)} RL rows",
            flush=True,
        )
        math_rows = kept_rows
    math_modal_baseline = modal_answer_baseline(math_rows)
    modal_solution = f"Answer: {math_modal_baseline['answer']}"
    modal_shaped_rewards = []
    for row in math_rows:
        truth = str(row["reward_model"]["ground_truth"])
        correct, _ = verify_answer(
            modal_solution, truth, answer_style(row)
        )
        modal_shaped_rewards.append(
            1.0
            if correct
            else nearby_numeric_reward(
                str(math_modal_baseline["answer"]),
                truth,
                args.nearby_reward_max,
            )
        )
    math_modal_baseline["shaped_reward"] = (
        sum(modal_shaped_rewards) / len(modal_shaped_rewards)
        if modal_shaped_rewards
        else 0.0
    )
    data_identity = math_dataset_identity(args.math_data, args.exclude_modules)
    sampler = MathPromptSampler(
        math_rows, args.seed, dataset_identity=data_identity
    )
    if actor_init_payload is not None:
        if args.curriculum_init:
            print(
                f"actor curriculum-initialized from {initialization_path} at "
                f"source step {actor_init_payload.get('step')}; fresh target "
                "critic, optimizers, step, and sampler cursor",
                flush=True,
            )
        else:
            if actor_init_payload.get("math_data_identity") != data_identity:
                raise ValueError(
                    "initialization checkpoint's prompt cursor belongs to different "
                    "dataset bytes, exclusions, or ordering"
                )
            source_args = actor_init_payload.get("args", {})
            if hasattr(source_args, "__dict__"):
                source_args = vars(source_args)
            # The content hash above, not a cwd-relative filename, proves that
            # the cursor belongs to these exact dataset bytes. This also keeps
            # a checkpoint relocatable into a reproducible detached worktree.
            if int(source_args.get("seed", -1)) != args.seed:
                raise ValueError(
                    "initialization checkpoint must use the source seed so its "
                    "deterministic prompt order can continue without reuse"
                )
            source_exclusions = source_args.get("exclude_modules") or ""
            if source_exclusions != args.exclude_modules:
                raise ValueError(
                    "initialization checkpoint must use the same "
                    "--exclude-modules setting"
                )
            actor_cursor = int(actor_init_payload.get("sampler_cursor", -1))
            if not 0 <= actor_cursor < len(math_rows):
                raise ValueError(
                    "initialization source has no valid unused first-epoch prompt "
                    f"cursor: {actor_cursor} for {len(math_rows)} rows"
                )
            sampler.cursor = actor_cursor
            print(
                f"actor{' and critic' if args.actor_critic_init else ''} initialized "
                f"from {initialization_path} at source step "
                f"{actor_init_payload.get('step')} and sampler cursor {actor_cursor}",
                flush=True,
            )
    seq_len = FreshHyperparameters.train_seq_len
    luts = baseline.build_sentencepiece_luts(tokenizer, FreshHyperparameters.vocab_size, device)
    val_tokens = baseline.load_validation_tokens(FreshHyperparameters.val_files, seq_len)
    if args.bpb_val_tokens > 0:
        usable = (args.bpb_val_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(
                f"--bpb-val-tokens must cover at least one {seq_len}-token sequence"
            )
        val_tokens = val_tokens[: usable + 1]
        print(
            f"BPB guard subsampled to {usable} validation tokens", flush=True
        )

    start_step = 0
    warmup_step = args.value_warmup_steps if args.actor_critic_init else 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        adapter_migration = None
        if args.migrate_zero_adapter_resume:
            adapter_migration = migrate_zero_adapter_resume(payload, wrapper)
        migrate_legacy_wrapper_checkpoint(payload, wrapper)
        validate_renderer_checkpoint(payload, args.resume)
        if not resume_execution_schema_compatible(payload):
            raise ValueError(
                "resume checkpoint execution schema must be one of "
                f"{sorted((EXECUTION_SCHEMA, PREVIOUS_EXECUTION_SCHEMA))!r}; "
                f"got {payload.get('execution_schema')!r}. Use --actor-init or "
                "--actor-critic-init for an explicit initialization restart."
            )
        if payload.get("reward_schema") != REWARD_SCHEMA:
            raise ValueError(
                f"resume checkpoint reward schema must be {REWARD_SCHEMA!r}; "
                f"got {payload.get('reward_schema')!r}"
            )
        if payload.get("math_data_identity") != data_identity:
            raise ValueError(
                "resume checkpoint's prompt cursor belongs to different dataset "
                "bytes, exclusions, or ordering"
            )
        wrapper.load_state_dict(payload["model"], strict=True)
        critic.load_state_dict(payload["critic"], strict=True)
        optimizers["actor"].load_state_dict(payload["optimizers"]["actor"])
        optimizers["critic"].load_state_dict(
            payload["optimizers"]["critic"]
        )
        # The sigma head is learned: the model load above restored it, and
        # (unlike the old fixed-buffer scheme) the CLI must NOT reassert it on
        # resume — --thought-log-sigma-init is an initialization, not a
        # schedule.  Learning rates remain the documented cross-run knobs and
        # are reasserted below.
        for group in optimizers["actor"].param_groups:
            group["lr"] = args.learning_rate
        for group in optimizers["critic"].param_groups:
            group["lr"] = args.learning_rate
        start_step = int(payload["step"])
        warmup_step = int(
            payload.get(
                "value_warmup_step",
                args.value_warmup_steps if start_step > 0 else 0,
            )
        )
        torch.set_rng_state(payload["cpu_rng"])
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
        random.setstate(payload["python_rng"])
        sampler.cursor = int(payload["sampler_cursor"])
        actor_init_provenance = payload.get("actor_init_provenance")
        if adapter_migration is not None:
            actor_init_provenance = dict(actor_init_provenance or {})
            actor_init_provenance["zero_adapter_resume_migration"] = (
                adapter_migration
            )

    if args.actor_critic_init:
        torch.set_rng_state(actor_init_payload["cpu_rng"])
        torch.cuda.set_rng_state_all(actor_init_payload["cuda_rng"])
        random.setstate(actor_init_payload["python_rng"])

    if args.bpb_only or args.bench_only:
        planned_prompt_count = 0
    elif args.rollout_only:
        planned_prompt_count = args.prompts_per_rollout
    else:
        warmup_updates = (
            max(args.value_warmup_steps - warmup_step, 0) if start_step == 0 else 0
        )
        remaining_actor_steps = max(args.steps - start_step, 0)
        warmup_prompt_count = args.prompts_per_minibatch * warmup_updates
        if args.consume_all_prompts:
            planned_prompt_count, _ = plan_one_pass_training(
                dataset_rows=len(math_rows),
                sampler_cursor=sampler.cursor,
                warmup_updates=warmup_updates,
                remaining_actor_steps=remaining_actor_steps,
                prompts_per_minibatch=args.prompts_per_minibatch,
            )
        else:
            planned_prompt_count = args.prompts_per_minibatch * (
                warmup_updates + remaining_actor_steps
            )
    if sampler.cursor + planned_prompt_count > len(math_rows):
        raise ValueError(
            "posttraining run would cross the dataset's first epoch and "
            "reuse prompts: cursor "
            f"{sampler.cursor} + planned {planned_prompt_count} > "
            f"{len(math_rows)} rows"
        )

    # Rollout and evaluation use the identical dynamic narrow-prefix step.
    # Compile it once: separate wrappers paid the same large cold compilation
    # cost at step-0 eval and again at the first training collect. If eval ever
    # latches an eager fallback, the eval closure below also disables this
    # shared artifact for training before it can be called again.
    compiled_generation_step = None
    if args.rollout_compile or args.eval_compile:
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, 64
        )
        compiled_generation_step = torch.compile(
            wrapper.step_core,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
            dynamic=True,
        )
    rollout_step_core = (
        compiled_generation_step if args.rollout_compile else None
    )
    eval_step_core = compiled_generation_step if args.eval_compile else None

    # Preserve the eager function for infrequent no-grad diagnostics. Calling
    # the compiled training artifact under no-grad would create a distinct
    # AOTAutograd specialization solely because grad mode is a Dynamo guard.
    diagnostic_replay_head_inputs = replay_head_inputs

    # Replay compilation is independent of rollout. Dynamic B/L plus bounded
    # 64-token buckets lets one artifact cover the length-aware shard plan;
    # no CUDA graph owns outputs that remain live through eager losses/backward.
    trim_multiple = args.replay_bucket if args.compile_replay else 1
    if args.compile_replay:
        torch._dynamo.config.cache_size_limit = 64
        critic.value_logits = torch.compile(
            critic.value_logits,
            # Length buckets still span many GEMM shapes. Max-autotune
            # repeatedly stalls training to benchmark each new regime;
            # default Inductor dispatches them to stable cuBLAS kernels.
            mode="default",
            fullgraph=True,
            dynamic=True,
        )
        compiled_replay = torch.compile(
            replay_head_inputs,
            mode="default",
            fullgraph=True,
            dynamic=True,
        )
        # Both consumers bound the function by name at import time:
        # update_minibatch through this module's global,
        # refresh_old_statistics through latent_rollout's.  Rebinding both
        # to the SAME compiled object keeps refresh and update on one
        # compiled artifact — the property that makes behavior-age-0 PPO ratios
        # exactly one (see refresh_old_statistics on why it also runs
        # grad-enabled).
        globals()["replay_head_inputs"] = compiled_replay
        postraining.latent_rollout.replay_head_inputs = compiled_replay

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output / "metrics.jsonl")
    tensorboard_purge_step = None
    if args.resume:
        removed = logger.purge_after(start_step, warmup_step)
        removed_reports = purge_benchmark_reports_after(output, start_step)
        if removed:
            print(
                f"purged {removed} stale metrics after actor step {start_step} "
                f"and warmup step {warmup_step}",
                flush=True,
            )
        if removed_reports:
            print(
                f"purged {removed_reports} stale benchmark answer reports",
                flush=True,
            )
        tensorboard_purge_step = (
            warmup_step + 1 if start_step == 0 else start_step + 1
        )
    tensorboard = SummaryWriter(
        output / "tensorboard", purge_step=tensorboard_purge_step
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "phase": "latent_vapo_dapo",
                "execution_schema": EXECUTION_SCHEMA,
                "prompt_order_schema": PROMPT_ORDER_SCHEMA,
                "math_data_identity": data_identity,
                "reward_schema": REWARD_SCHEMA,
                "math_modal_answer_baseline": math_modal_baseline,
                "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
                "rollout_policy_schema": ROLLOUT_POLICY_SCHEMA,
                "thought_input_schema": THOUGHT_INPUT_SCHEMA,
                "thought_distribution_schema": THOUGHT_DISTRIBUTION_SCHEMA,
                "thought_mean_schema": THOUGHT_MEAN_SCHEMA,
                "args": vars(args),
                "base": {
                    "checkpoint": str(args.checkpoint),
                    "architecture": backbone.architecture,
                },
                "actor_init": actor_init_provenance,
                "critic": {
                    "init": (
                        "warm_checkpoint"
                        if args.actor_critic_init
                        else "scratch"
                    ),
                    "checkpoint": (
                        str(args.actor_critic_init)
                        if args.actor_critic_init
                        else None
                    ),
                    "source_execution_schema": (
                        actor_init_payload.get("execution_schema")
                        if args.actor_critic_init
                        else None
                    ),
                    "source_warmup_optimizer_semantics": (
                        "legacy_per_prompt_group"
                        if args.actor_critic_init
                        else None
                    ),
                    "architecture": backbone.architecture,
                    "value_bins": args.value_bins,
                    "value_sigma_ratio": args.value_sigma_ratio,
                    "value_prior": args.value_prior,
                    "parameters": sum(p.numel() for p in critic.parameters()),
                },
            },
            indent=2,
        )
        + "\n"
    )
    # Guess-calibration record: each module's modal-answer share is the
    # accuracy of answering its most common ground truth every time, so any
    # module accuracy is evidence of solving only above this line. Retain it
    # once per dataset in structured JSON for drill-down without creating
    # dozens of low-frequency TensorBoard series.
    for name, dataset_rows in (("bench", bench_rows), ("math", sampler.rows)):
        baselines = module_answer_baselines(dataset_rows)
        if baselines and not args.resume:
            # One nested kwarg: module names must never collide with the
            # logger's fixed "type" key.
            logger.log(type=f"{name}_guess_baseline", baselines=baselines)
    if not args.resume:
        logger.log(type="math_modal_answer_baseline", **math_modal_baseline)

    def finish_group(
        batch: LatentRolloutBatch,
        row: dict,
        refresh_statistics: bool,
    ) -> LatentRolloutBatch:
        batch = trim_stream(batch, multiple=trim_multiple)
        score_math_rollout(
            batch, row["reward_model"]["ground_truth"], tokenizer, stop_ids,
            answer_style(row),
            args.nearby_reward_max,
        )
        # Stepwise rollout and parallel replay disagree numerically at
        # bf16 scale; recompute the stored PPO statistics through the
        # update-step replay path so fresh-behavior ratios are exactly one.
        # This also fills old_values from the separate critic (the
        # rollout never values).
        if refresh_statistics:
            refresh_old_statistics(
                wrapper,
                critic,
                batch,
                max_trajectories=args.replay_max_trajectories,
                attention_budget=args.replay_attention_budget,
                bucket_multiple=args.replay_bucket,
            )
        return batch

    def _collect(
        refresh_statistics: bool,
        prompt_count: int,
        offload_to_cpu: bool,
    ) -> list[LatentRolloutBatch]:
        """One rollout: a scored prompt group per sampled DAPO problem."""
        rollout_rows = sampler.next_rows(prompt_count)
        groups = []
        cpu = torch.device("cpu")

        def retain_group(
            batch: LatentRolloutBatch,
            row: dict,
        ) -> LatentRolloutBatch:
            return finish_group(batch, row, refresh_statistics)

        if args.rollout_groups <= 1:
            for row in rollout_rows:
                prompt_ids = torch.tensor(
                    encode_prompt(tokenizer, prompt_text(row), args.prompt_tokens),
                    dtype=torch.long, device=device,
                )
                batch = rollout_continuations(
                    wrapper, prompt_ids[None].expand(args.samples_per_prompt, -1),
                    args.continuation_tokens, max_stream_steps,
                    args.temperature, args.top_p, stop_ids=stop_ids or None,
                    force_initial_think=half_forced_group_members(
                        1, args.samples_per_prompt, device
                    ),
                    record_likelihoods=False,
                    cache_dtype=torch.bfloat16,
                    tensor_positions=rollout_step_core is not None,
                    compact_finished=rollout_step_core is None,
                )
                if offload_to_cpu:
                    batch = batch.to(cpu)
                groups.append(retain_group(batch, row))
                del batch
            return groups
        # Left-padded batched rollout: all chunk groups step together, so
        # each launch carries chunk*samples rows instead of samples — the
        # sequential per-group loop is launch-bound, not compute-bound.
        samples = args.samples_per_prompt
        for chunk_start in range(0, len(rollout_rows), args.rollout_groups):
            chunk = rollout_rows[chunk_start : chunk_start + args.rollout_groups]
            encoded = [
                encode_prompt(tokenizer, prompt_text(row), args.prompt_tokens)
                for row in chunk
            ]
            width = max(len(ids) for ids in encoded)
            prompt_ids = torch.zeros(
                (len(chunk) * samples, width), dtype=torch.long, device=device
            )
            for index, ids in enumerate(encoded):
                rows = slice(index * samples, (index + 1) * samples)
                prompt_ids[rows, width - len(ids):] = torch.tensor(
                    ids, dtype=torch.long, device=device
                )
            prompt_lengths = torch.tensor(
                [len(ids) for ids in encoded], dtype=torch.long, device=device
            ).repeat_interleave(samples)
            batched = rollout_continuations(
                wrapper, prompt_ids, args.continuation_tokens, max_stream_steps,
                args.temperature, args.top_p, stop_ids=stop_ids or None,
                prompt_lengths=prompt_lengths,
                force_initial_think=half_forced_group_members(
                    len(chunk), samples, device
                ),
                record_likelihoods=False,
                cache_dtype=torch.bfloat16,
                tensor_positions=rollout_step_core is not None,
                compact_finished=rollout_step_core is None,
            )
            if offload_to_cpu:
                # One chunk-level D2H transfer, then all variable-length
                # splitting, trimming, decoding, and scoring stay on CPU.
                # This avoids one synchronization/transfer per prompt group.
                batched = batched.to(cpu)
                prompt_lengths = prompt_lengths.to(cpu)
            split_groups = split_rollout_groups(batched, samples, prompt_lengths)
            # Every split owns cloned storage; drop the much larger
            # groups*samples rollout before the first full-stream replay.
            del batched
            for group_index, (group, row) in enumerate(
                zip(split_groups, chunk, strict=True)
            ):
                groups.append(retain_group(group, row))
                # Once retained, remove the source from the temporary chunk
                # list immediately. This bounds both GPU storage in ordinary
                # collection and host storage in the offloaded actor path.
                if offload_to_cpu:
                    split_groups[group_index] = None
                del group
            del split_groups
        return groups

    def training_autocast():
        return torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        )

    def collect(
        refresh_statistics: bool = True,
        prompt_count: int | None = None,
        offload_to_cpu: bool = False,
    ) -> list[LatentRolloutBatch]:
        if offload_to_cpu and refresh_statistics:
            raise ValueError(
                "CPU-offloaded collection must defer behavior refresh until "
                "optimizer minibatches are assembled"
            )
        original_step_core = wrapper.step_core
        if rollout_step_core is not None:
            wrapper.step_core = rollout_step_core
        try:
            # rollout_continuations itself is no-grad, while the optional
            # refresh below must remain grad-enabled so refresh/update share
            # one compiled replay specialization and identical PPO numerics.
            with training_autocast():
                return _collect(
                    refresh_statistics,
                    args.prompts_per_rollout
                    if prompt_count is None
                    else prompt_count,
                    offload_to_cpu,
                )
        finally:
            wrapper.step_core = original_step_core

    def training_update(*update_args, **update_kwargs):
        with training_autocast():
            return update_minibatch(*update_args, **update_kwargs)

    def teacher_forced_bpb() -> float:
        """Teacher-forced BPB through the deployed belief renderer."""
        wrapper.eval()
        _, bpb = baseline.eval_val(
            FreshHyperparameters, wrapper, 0, 1, device, 8, val_tokens, *luts
        )
        wrapper.eval()
        return bpb

    def aime_eval(step: int) -> None:
        nonlocal rollout_step_core
        wrapper.eval()
        captured_attempts: list[dict[str, object]] = []
        eval_started = time.perf_counter()
        metrics = evaluate_aime_latent(
            wrapper, tokenizer, aime_rows, args.aime_samples, args.aime_max_tokens,
            aime_stream_steps, args.aime_chunk, args.seed, device,
            prompt_tokens=args.prompt_tokens,
            batch_trajectories=args.eval_batch_trajectories,
            compiled_step_core=eval_step_core,
            compiled_tail_batch=args.eval_tail_batch or None,
            answer_style_override="aime",
            captured_attempts=captured_attempts,
        )
        metrics["dataset_modal_answer"] = aime_modal_baseline["answer"]
        metrics["dataset_modal_answer_style"] = "aime"
        metrics["dataset_modal_answer_accuracy"] = aime_modal_baseline[
            "accuracy"
        ]
        if metrics["compile_fallback"] and eval_step_core is rollout_step_core:
            rollout_step_core = None
        eval_seconds = time.perf_counter() - eval_started
        tensorboard.add_scalar("perf/aime_eval_seconds", eval_seconds, step)
        logger.log(type="aime", step=step, seconds=eval_seconds, **metrics)
        write_benchmark_report(
            output / "aime_answers",
            step,
            metrics,
            captured_attempts,
            reward_schema=REWARD_SCHEMA,
        )
        tensorboard.add_scalar("aime/accuracy", metrics["accuracy"], step)
        tensorboard.add_scalar(
            "aime/forced_initial_accuracy",
            metrics["forced_initial_accuracy"],
            step,
        )
        tensorboard.add_scalar(
            "aime/unforced_initial_accuracy",
            metrics["unforced_initial_accuracy"],
            step,
        )
        tensorboard.add_scalar(
            "aime/optional_think_fraction", metrics["think_fraction"], step
        )
        print(f"step:{step} aime_avg@{args.aime_samples}:{metrics['accuracy']:.4f}", flush=True)

    def bench_eval(
        step: int,
        *,
        eval_seed: int | None = None,
        repeat: int | None = None,
    ) -> None:
        nonlocal rollout_step_core
        wrapper.eval()
        if eval_seed is None:
            eval_seed = args.seed
        captured_attempts: list[dict[str, object]] = []
        eval_started = time.perf_counter()
        metrics = evaluate_aime_latent(
            wrapper, tokenizer, bench_rows, args.bench_samples,
            args.bench_max_tokens, bench_stream_steps, args.bench_samples,
            eval_seed, device, prompt_tokens=args.prompt_tokens,
            batch_trajectories=args.eval_batch_trajectories,
            compiled_step_core=eval_step_core,
            compiled_tail_batch=args.eval_tail_batch or None,
            captured_attempts=captured_attempts,
        )
        metrics["dataset_modal_answer"] = bench_dataset_baseline["answer"]
        metrics["dataset_modal_answer_style"] = bench_dataset_baseline["style"]
        metrics["dataset_modal_answer_accuracy"] = bench_dataset_baseline[
            "accuracy"
        ]
        metrics["subset_modal_answer_accuracy"] = bench_subset_baseline[
            "accuracy"
        ]
        if metrics["compile_fallback"] and eval_step_core is rollout_step_core:
            rollout_step_core = None
        eval_seconds = time.perf_counter() - eval_started
        tensorboard.add_scalar("perf/bench_eval_seconds", eval_seconds, step)
        write_benchmark_report(
            output,
            step,
            metrics,
            captured_attempts,
            reward_schema=REWARD_SCHEMA,
        )
        logger.log(
            type="bench",
            step=step,
            seconds=eval_seconds,
            eval_seed=eval_seed,
            repeat=repeat,
            **metrics,
        )
        tensorboard.add_scalar("bench/accuracy", metrics["accuracy"], step)
        tensorboard.add_scalar(
            "bench/forced_initial_accuracy",
            metrics["forced_initial_accuracy"],
            step,
        )
        tensorboard.add_scalar(
            "bench/unforced_initial_accuracy",
            metrics["unforced_initial_accuracy"],
            step,
        )
        tensorboard.add_scalar(
            "bench/optional_think_fraction", metrics["think_fraction"], step
        )
        print(f"step:{step} bench_avg@{args.bench_samples}:{metrics['accuracy']:.4f}", flush=True)

    if args.bench_only:
        for repeat in range(args.bench_only_repeats):
            # Repeat zero warms the compiled decoder. Subsequent repeats use
            # matched but distinct workloads across evaluator modes, avoiding
            # a single stochastic tail being mistaken for expected speed.
            bench_eval(
                start_step,
                eval_seed=args.seed + repeat,
                repeat=repeat,
            )
        tensorboard.close()
        return

    if args.bpb_only:
        eval_started = time.perf_counter()
        bpb = teacher_forced_bpb()
        eval_seconds = time.perf_counter() - eval_started
        print(
            json.dumps(
                {
                    "type": "bpb",
                    "step": start_step,
                    "val_bpb": bpb,
                    "seconds": eval_seconds,
                    "val_tokens": val_tokens.numel() - 1,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        tensorboard.close()
        return

    if args.rollout_only:
        metrics = aggregate_diagnostics(collect(), args.samples_per_prompt, stop_ids)
        passed = metrics["within_group_reward_std"] >= args.gate_min_within_group_reward_std
        logger.log(type="rollout_gate", passed=passed, **metrics)
        print(json.dumps({"passed": bool(passed), **metrics}, sort_keys=True))
        tensorboard.close()
        raise SystemExit(0 if passed else 2)

    if start_step == 0 and (warmup_step == 0 or args.actor_critic_init):
        bpb = teacher_forced_bpb()
        logger.log(type="bpb", step=0, val_bpb=bpb)
        tensorboard.add_scalar("guard/val_bpb", bpb, 0)
        print(f"step:0 teacher-forced val_bpb:{bpb:.4f}", flush=True)
        if aime_rows:
            aime_eval(0)
        if bench_rows:
            bench_eval(0)
    if start_step == 0 and warmup_step < args.value_warmup_steps:
        for warmup in range(warmup_step + 1, args.value_warmup_steps + 1):
            warmup_started = time.perf_counter()
            collect_started = warmup_started
            groups = collect(
                refresh_statistics=False,
                prompt_count=args.prompts_per_minibatch,
            )
            torch.cuda.synchronize()
            collect_seconds = time.perf_counter() - collect_started
            update_started = time.perf_counter()
            value_action_denominator = torch.stack(
                [group.action_mask.sum() for group in groups]
            ).sum()
            optimizers["critic"].zero_grad(set_to_none=True)
            group_metrics = [
                training_update(
                    wrapper,
                    critic,
                    group,
                    optimizers,
                    value_only=True,
                    critic_step=False,
                    value_action_denominator=value_action_denominator,
                    replay_max_trajectories=args.replay_max_trajectories,
                    replay_attention_budget=args.replay_attention_budget,
                    replay_bucket=args.replay_bucket,
                )
                for group in groups
            ]
            optimizers["critic"].step()
            del groups
            update_seconds = time.perf_counter() - update_started
            metrics = {
                key: float(sum(m[key] for m in group_metrics) / len(group_metrics))
                for key in group_metrics[0]
            }
            value_metrics = aggregate_value_diagnostics(group_metrics)
            metrics.update(
                seconds=time.perf_counter() - warmup_started,
                collect_seconds=collect_seconds,
                update_seconds=update_seconds,
                action_count=sum(m["action_count"] for m in group_metrics),
                value_mean=value_metrics["prediction_mean"],
                value_target_mean=value_metrics["target_mean"],
                value_target_variance=value_metrics["target_variance"],
                value_residual_mean=value_metrics["residual_mean"],
                value_residual_variance=value_metrics["residual_variance"],
                explained_variance=value_metrics["explained_variance"],
                value_excess_ce=value_metrics["excess_ce"],
                value_loss=_weighted_metric_mean(
                    group_metrics, "value_loss", "action_count"
                ),
                critic_grad_norm=group_metrics[-1]["critic_grad_norm"],
                think_advantage_mean=_weighted_metric_mean(
                    group_metrics, "think_advantage_mean", "think_action_count"
                ),
                forced_initial_think_advantage_mean=_weighted_metric_mean(
                    group_metrics,
                    "forced_initial_think_advantage_mean",
                    "forced_initial_think_action_count",
                ),
                thought_advantage_mean=_weighted_metric_mean(
                    group_metrics,
                    "thought_advantage_mean",
                    "thought_action_count",
                ),
                emit_advantage_mean=_weighted_metric_mean(
                    group_metrics, "emit_advantage_mean", "emit_action_count"
                ),
            )
            logger.log(type="value_warmup", step=warmup, **metrics)
            warmup_dashboard = {
                "value_warmup/token_weighted_excess_ce": metrics[
                    "value_excess_ce"
                ],
                "value_warmup/prediction_mean": metrics["value_mean"],
                "value_warmup/target_mean": metrics["value_target_mean"],
                "value_warmup/residual_mean": metrics["value_residual_mean"],
                "value_warmup/online_explained_variance": metrics[
                    "explained_variance"
                ],
                "value_warmup/residual/optional_think_mean": metrics[
                    "think_advantage_mean"
                ],
                "value_warmup/residual/forced_initial_think_mean": metrics[
                    "forced_initial_think_advantage_mean"
                ],
                "value_warmup/residual/all_think_mean": metrics[
                    "thought_advantage_mean"
                ],
                "value_warmup/residual/emit_mean": metrics[
                    "emit_advantage_mean"
                ],
                "value_warmup/critic_grad_norm": metrics[
                    "critic_grad_norm"
                ],
            }
            for tag, value in warmup_dashboard.items():
                tensorboard.add_scalar(tag, value, warmup)
            # The optimizer state is sufficient for the next update and for
            # checkpoints. Keeping the accumulated gradient buffers alive
            # overlaps them with the next rollout's KV cache for no benefit.
            optimizers["critic"].zero_grad(set_to_none=True)
            warmup_step = warmup
            if (
                warmup % args.warmup_save_every == 0
                or warmup == args.value_warmup_steps
            ):
                # Preserve the clean actor-initial/critic-warm checkpoint.
                # The rolling checkpoint is overwritten by PPO updates, so
                # it cannot serve as a trustworthy actor restart point.
                checkpoint_name = (
                    "critic_warmup_checkpoint.pt"
                    if warmup == args.value_warmup_steps
                    else "latent_vapo_checkpoint.pt"
                )
                save_checkpoint(
                    output / checkpoint_name,
                    wrapper,
                    critic,
                    optimizers,
                    start_step,
                    args,
                    sampler,
                    warmup_step,
                    actor_init_provenance,
                )

    def crossed_interval(previous: int, current: int, interval: int) -> bool:
        return interval > 0 and current // interval > previous // interval

    step = start_step
    while step < args.steps:
        previous_step = step
        started = time.perf_counter()
        collect_started = started
        requested_pool_updates = min(
            args.prompts_per_rollout // args.prompts_per_minibatch,
            args.steps - step,
        )
        pool_prompt_count = (
            requested_pool_updates * args.prompts_per_minibatch
        )
        if args.consume_all_prompts:
            pool_prompt_count = min(
                pool_prompt_count, len(math_rows) - sampler.cursor
            )
        if pool_prompt_count < 1:
            raise RuntimeError("actor steps remain but no unused prompts are available")
        pool_updates = math.ceil(
            pool_prompt_count / args.prompts_per_minibatch
        )
        torch.cuda.reset_peak_memory_stats()
        groups = collect(
            refresh_statistics=False,
            prompt_count=pool_prompt_count,
            offload_to_cpu=True,
        )
        rollout_metrics = aggregate_diagnostics(
            groups, args.samples_per_prompt, stop_ids
        )
        minibatch_orders = optimizer_minibatch_orders(
            len(groups),
            args.prompts_per_minibatch,
            allow_partial_final=(
                args.consume_all_prompts and sampler.cursor == len(math_rows)
            ),
        )
        if len(minibatch_orders) != pool_updates:
            raise RuntimeError("rollout pool did not produce the planned updates")

        # Assemble each shuffled optimizer minibatch on CPU with RIGHT tail
        # padding, then refresh all old statistics under the still-frozen
        # behavior policy. Scatter only those statistics back into compact
        # groups: retaining four max-length padded B256 host batches would
        # waste up to ~20 GiB. Update repacks the identical order and width,
        # preserving the exact replay plan while keeping transfers batched.
        packed_batch_bytes_max = 0
        packed_real_slots = 0
        packed_capacity_slots = 0
        pool_cpu_pack_seconds = 0.0
        pool_h2d_seconds = 0.0
        pool_refresh_seconds = 0.0
        pool_d2h_seconds = 0.0
        for minibatch_order in minibatch_orders:
            selected_groups = [groups[index] for index in minibatch_order]
            pack_started = time.perf_counter()
            cpu_batch = pack_rollout_groups_for_replay(
                selected_groups
            )
            pool_cpu_pack_seconds += time.perf_counter() - pack_started
            transfer_started = time.perf_counter()
            device_batch = cpu_batch.to(device)
            torch.cuda.synchronize()
            pool_h2d_seconds += time.perf_counter() - transfer_started
            del cpu_batch
            refresh_started = time.perf_counter()
            with training_autocast():
                refresh_old_statistics(
                    wrapper,
                    critic,
                    device_batch,
                    max_trajectories=args.replay_max_trajectories,
                    attention_budget=args.replay_attention_budget,
                    bucket_multiple=args.replay_bucket,
                )
            torch.cuda.synchronize()
            pool_refresh_seconds += time.perf_counter() - refresh_started
            packed_batch_bytes_max = max(
                packed_batch_bytes_max,
                sum(
                    value.numel() * value.element_size()
                    for field in fields(device_batch)
                    if isinstance((value := getattr(device_batch, field.name)), torch.Tensor)
                ),
            )
            packed_real_slots += int((device_batch.kind != PAD_SLOT).sum())
            packed_capacity_slots += device_batch.kind.numel()
            transfer_started = time.perf_counter()
            scatter_replay_statistics(device_batch, selected_groups)
            pool_d2h_seconds += time.perf_counter() - transfer_started
            del device_batch
            del selected_groups
        old_value_sum = 0.0
        old_value_count = 0
        for group in groups:
            generated = group.action_mask.bool()
            old_value_sum += float(group.old_values[generated].sum())
            old_value_count += int(generated.sum())
        rollout_metrics["old_value_mean"] = (
            old_value_sum / old_value_count if old_value_count else 0.0
        )
        rollout_metrics["packed_batch_gib_max"] = packed_batch_bytes_max / 2**30
        rollout_metrics["packed_padding_utilization"] = (
            packed_real_slots / packed_capacity_slots
            if packed_capacity_slots
            else 0.0
        )
        rollout_metrics["pool_cpu_pack_seconds"] = pool_cpu_pack_seconds
        rollout_metrics["pool_h2d_seconds"] = pool_h2d_seconds
        rollout_metrics["pool_refresh_seconds"] = pool_refresh_seconds
        rollout_metrics["pool_d2h_seconds"] = pool_d2h_seconds
        torch.cuda.synchronize()
        collect_seconds = time.perf_counter() - collect_started
        rollout_peak_vram_bytes = torch.cuda.max_memory_allocated()
        rollout_step = step + pool_updates
        logger.log(
            type="rollout", step=rollout_step,
            collect_seconds=collect_seconds,
            pool_updates=pool_updates,
            pool_trajectories=pool_prompt_count * args.samples_per_prompt,
            peak_vram_bytes=rollout_peak_vram_bytes,
            **rollout_metrics,
        )
        tensorboard.add_scalar(
            "perf/collect_seconds", collect_seconds, rollout_step
        )
        tensorboard.add_scalar(
            "perf/rollout_peak_vram_gib",
            rollout_peak_vram_bytes / 2**30,
            rollout_step,
        )
        tensorboard.add_scalar(
            "perf/packed_batch_gib_max",
            rollout_metrics["packed_batch_gib_max"],
            rollout_step,
        )
        tensorboard.add_scalar(
            "perf/packed_padding_utilization",
            rollout_metrics["packed_padding_utilization"],
            rollout_step,
        )
        tensorboard.add_scalar(
            "perf/pool_cpu_pack_seconds", pool_cpu_pack_seconds, rollout_step
        )
        tensorboard.add_scalar(
            "perf/pool_h2d_seconds", pool_h2d_seconds, rollout_step
        )
        tensorboard.add_scalar(
            "perf/pool_refresh_seconds", pool_refresh_seconds, rollout_step
        )
        tensorboard.add_scalar(
            "perf/pool_d2h_seconds", pool_d2h_seconds, rollout_step
        )
        rollout_dashboard = rollout_tensorboard_metrics(rollout_metrics)
        # Horizontal anti-collapse calibration: on DAPO, a terminated
        # constant modal answer already earns nontrivial exact and shaped
        # reward. On-policy improvement is meaningful only relative to both.
        rollout_dashboard["reward/modal_constant_exact_baseline"] = float(
            math_modal_baseline["accuracy"]
        )
        rollout_dashboard["reward/modal_constant_shaped_baseline"] = float(
            math_modal_baseline["shaped_reward"]
        )
        for tag, value in rollout_dashboard.items():
            tensorboard.add_scalar(tag, value, rollout_step)

        for behavior_age, minibatch_order in enumerate(minibatch_orders):
            # Actor and critic each take one optimizer step over the same
            # effective trajectory minibatch. Prompt groups and replay shards
            # are memory partitions, not optimizer minibatches.
            update_started = time.perf_counter()
            next_step = step + 1
            optimizers["actor"].zero_grad(set_to_none=True)
            optimizers["critic"].zero_grad(set_to_none=True)
            sigma_parameters = list(
                wrapper.transition.log_sigma_head.parameters()
            )
            sigma_before = [
                parameter.detach().clone() for parameter in sigma_parameters
            ]
            mean_parameters = list(wrapper.transition.mean_head.parameters())
            mean_before = [
                parameter.detach().clone() for parameter in mean_parameters
            ]
            selected_groups = [groups[index] for index in minibatch_order]
            minibatch_pack_started = time.perf_counter()
            cpu_minibatch = pack_rollout_groups_for_replay(selected_groups)
            minibatch_pack_seconds = (
                time.perf_counter() - minibatch_pack_started
            )
            minibatch_h2d_started = time.perf_counter()
            device_minibatch = cpu_minibatch.to(device)
            torch.cuda.synchronize()
            minibatch_h2d_seconds = time.perf_counter() - minibatch_h2d_started
            del cpu_minibatch
            (
                policy_action_denominator,
                gate_action_denominator,
                positive_token_denominator,
            ) = actor_minibatch_denominators(
                [device_minibatch], [0], args.positive_reward_threshold
            )
            metrics = training_update(
                wrapper, critic, device_minibatch, optimizers,
                positive_lm_weight=args.positive_lm_weight,
                positive_reward_threshold=args.positive_reward_threshold,
                thought_pg_coef=args.thought_pg_coef,
                gate_entropy_coef=args.gate_entropy_coef,
                gate_pg_coef=(
                    0.0 if next_step <= args.gate_freeze_steps else 1.0
                ),
                actor_step=False,
                critic_step=False,
                policy_action_denominator=policy_action_denominator,
                gate_action_denominator=gate_action_denominator,
                positive_token_denominator=positive_token_denominator,
                value_action_denominator=policy_action_denominator,
                gae_lambda_alpha=args.gae_lambda_alpha,
                replay_max_trajectories=args.replay_max_trajectories,
                replay_attention_budget=args.replay_attention_budget,
                replay_bucket=args.replay_bucket,
            )
            # The first minibatch runs against refresh-computed behavior
            # statistics with the actor untouched. Later disjoint minibatches
            # intentionally have behavior age 1..N.
            if behavior_age == 0:
                guard = "policy_clip_fraction"
                if metrics[guard] > 1e-6:
                    print(
                        f"WARNING step {next_step}: behavior-age-0 {guard}="
                        f"{metrics[guard]:.3e} (expected exactly 0; "
                        "refresh/update code paths diverged)",
                        flush=True,
                    )
            minibatch_metrics = [metrics]
            actor_dashboard = aggregate_actor_tensorboard_metrics(
                minibatch_metrics
            )
            nonfinite_gradients = {
                name: actor_dashboard[name]
                for name in (
                    "grad/trunk", "grad/gate", "grad/adapter", "grad/renderer",
                    "grad/sigma", "grad/thought_mean", "grad/critic",
                )
                if not math.isfinite(actor_dashboard[name])
            }
            if nonfinite_gradients:
                raise RuntimeError(
                    "non-finite gradients before optimizer step: "
                    f"{nonfinite_gradients}"
                )
            optimizers["actor"].step()
            optimizers["critic"].step()
            # Gradient buffers have already been reduced to scalar telemetry.
            # Release them before the optional second replay so this read-only
            # diagnostic cannot stack an inference forward on top of the
            # training step's peak allocation.
            optimizers["actor"].zero_grad(set_to_none=True)
            optimizers["critic"].zero_grad(set_to_none=True)
            with torch.no_grad():
                sigma_deltas = [
                    parameter.detach() - before
                    for parameter, before in zip(
                        sigma_parameters, sigma_before, strict=True
                    )
                ]
                mean_deltas = [
                    parameter.detach() - before
                    for parameter, before in zip(
                        mean_parameters, mean_before, strict=True
                    )
                ]
                head_updates = scalar_tensors_to_floats(
                    {
                        "sigma/head_update_rms": (
                            torch.stack(
                                [delta.square().sum() for delta in sigma_deltas]
                            ).sum()
                            / sum(delta.numel() for delta in sigma_deltas)
                        ).sqrt(),
                        "sigma/head_update_abs_max": torch.stack(
                            [delta.abs().max() for delta in sigma_deltas]
                        ).max(),
                        "sigma/mean_head_update_rms": (
                            torch.stack(
                                [delta.square().sum() for delta in mean_deltas]
                            ).sum()
                            / sum(delta.numel() for delta in mean_deltas)
                        ).sqrt(),
                        "sigma/mean_head_update_abs_max": torch.stack(
                            [delta.abs().max() for delta in mean_deltas]
                        ).max(),
                        "sigma/mean_head_parameter_abs_max": torch.stack(
                            [
                                wrapper.transition.mean_head.output_gain.detach().abs()
                                * wrapper.transition.mean_head.weight.detach().abs().max(),
                                wrapper.transition.mean_head.bias.detach().abs().max(),
                            ]
                        ).max(),
                        # Overwrite the pre-update minibatch snapshot with the
                        # adapter magnitude the next rollout will deploy.
                        "behavior/thought_adapter_weight_rms": (
                            wrapper.adapter.projection.weight.detach()
                            .square().mean().sqrt()
                        ),
                        "behavior/thought_adapter_bias_rms": (
                            wrapper.adapter.projection.bias.detach()
                            .square().mean().sqrt()
                        ),
                        "sigma/mean_output_gain": (
                            wrapper.transition.mean_head.output_gain.detach()
                        ),
                        "sigma/state_residual_gain": (
                            wrapper.transition.log_sigma_head.residual_gain.detach()
                        ),
                    }
                )
            if not all(math.isfinite(value) for value in head_updates.values()):
                raise RuntimeError(
                    f"non-finite policy head after optimizer step {next_step}: "
                    f"{head_updates}"
                )
            actor_dashboard.update(head_updates)
            actor_dashboard["perf/minibatch_h2d_seconds"] = (
                minibatch_h2d_seconds
            )
            actor_dashboard["perf/minibatch_cpu_pack_seconds"] = (
                minibatch_pack_seconds
            )
            if next_step == 1 or (
                args.post_update_kl_every > 0
                and next_step % args.post_update_kl_every == 0
            ):
                drift_started = time.perf_counter()
                with training_autocast():
                    post_update_drift = measure_post_update_policy_drift(
                        wrapper,
                        [device_minibatch],
                        replay_max_trajectories=args.replay_max_trajectories,
                        replay_attention_budget=args.replay_attention_budget,
                        replay_bucket=args.replay_bucket,
                        replay_function=diagnostic_replay_head_inputs,
                    )
                if not all(
                    math.isfinite(value) for value in post_update_drift.values()
                ):
                    raise RuntimeError(
                        f"non-finite post-update policy drift at step {next_step}: "
                        f"{post_update_drift}"
                    )
                actor_dashboard.update(post_update_drift)
                actor_dashboard["perf/post_update_kl_seconds"] = (
                    time.perf_counter() - drift_started
                )
            # One PPO epoch uses every behavior group exactly once. Release
            # the compact host sources and packed device batch after its
            # update and optional drift.
            del device_minibatch
            del selected_groups
            for index in minibatch_order:
                groups[index] = None
            step = next_step
            update_seconds = time.perf_counter() - update_started
            logger.log(
                type="train",
                step=step,
                behavior_age=behavior_age,
                trajectories=(
                    len(minibatch_order) * args.samples_per_prompt
                ),
                seconds=update_seconds,
                pool_seconds=time.perf_counter() - started,
                peak_vram_bytes=torch.cuda.max_memory_allocated(),
                dashboard=actor_dashboard,
                group_metrics=minibatch_metrics,
            )
            tensorboard.add_scalar("perf/update_seconds", update_seconds, step)
            write_actor_tensorboard_metrics(
                tensorboard, actor_dashboard, behavior_age, step
            )

        # No operation below consumes behavior groups. Release the emptied
        # container before
        # BPB/evaluation allocates its own full-context activations and KV
        # caches, and before the next collect evaluates its RHS.
        optimizers["actor"].zero_grad(set_to_none=True)
        optimizers["critic"].zero_grad(set_to_none=True)
        del groups

        if crossed_interval(previous_step, step, args.bpb_every):
            eval_started = time.perf_counter()
            bpb = teacher_forced_bpb()
            eval_seconds = time.perf_counter() - eval_started
            logger.log(type="bpb", step=step, val_bpb=bpb, seconds=eval_seconds)
            tensorboard.add_scalar("guard/val_bpb", bpb, step)
            tensorboard.add_scalar("perf/bpb_eval_seconds", eval_seconds, step)
        if aime_rows and crossed_interval(previous_step, step, args.aime_every):
            aime_eval(step)
        if bench_rows and crossed_interval(previous_step, step, args.bench_every):
            bench_eval(step)
        if crossed_interval(previous_step, step, args.save_every):
            save_started = time.perf_counter()
            save_checkpoint(
                output / "latent_vapo_checkpoint.pt", wrapper, critic,
                optimizers, step, args, sampler, warmup_step,
                actor_init_provenance,
            )
            save_seconds = time.perf_counter() - save_started
            logger.log(type="checkpoint", step=step, seconds=save_seconds)
            tensorboard.add_scalar("perf/checkpoint_seconds", save_seconds, step)
    if args.consume_all_prompts and sampler.cursor != len(math_rows):
        raise RuntimeError(
            "--consume-all-prompts completed without exhausting the target "
            f"dataset: cursor {sampler.cursor}/{len(math_rows)}"
        )
    save_checkpoint(
        output / "latent_vapo_checkpoint.pt", wrapper, critic,
        optimizers, step, args, sampler, warmup_step,
        actor_init_provenance,
    )
    if (
        aime_rows
        and args.aime_every > 0
        and step % args.aime_every != 0
    ):
        # Persist terminal weights before the optional expensive eval. The
        # periodic cadence measures 100, 200, ...; this additionally measures
        # a one-pass dataset ending between cadence boundaries.
        aime_eval(step)
    tensorboard.close()


if __name__ == "__main__":
    main()
