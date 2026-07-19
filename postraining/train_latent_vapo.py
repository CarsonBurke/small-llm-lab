"""Latent-thought VAPO: full-model PPO over THINK/EMIT gates, tokens, thoughts.

The WHOLE policy model trains at RL time (user prescription, Jul 18) — no
frozen trunk.  The world model's objective changes here: pretraining taught
it to predict what the next latent WILL be; the policy gradient retrains the
same prediction path toward what the next latent SHOULD be.

- PPO trains everything through one differentiable teacher-forced replay:
  the gate (Bernoulli THINK/EMIT, scalar clipped surrogate), emitted tokens
  (clipped surrogate from a belief-reading renderer — the standard VAPO actor
  loss),
  and thought CONTENT (per-dim clipped surrogate on the fixed-sigma Gaussian
  around the model's own predicted latent; a joint 512-dim ratio saturates
  the clip after one Adam step, so coordinates are bounded individually).
- The critic is a SEPARATE from-scratch model (same architecture class,
  fresh weights, fully trainable, no SIGReg or latent prediction) trained
  purely by HL-Gauss cross-entropy on [0, 1] value targets.  It values
  every stream position — vocab AND latent — which is what gives thinking
  its training signal.
- No entropy bonus, no KL penalty, no beta-NLL: sigma is a fixed constant,
  the trust region is the only policy constraint (paper-faithful — VAPO's
  optimized loss is L_PPO + mu*L_NLL and nothing else).
- No pretraining anchor (user prescription, Jul 18): SIGReg and the latent
  target-prediction objective are dropped at RL time — keeping them would
  pit "predict what the next latent WILL be" against the policy gradient's
  "predict what it SHOULD be" on the same prediction path.  The model
  trains purely on its ability to think; the teacher-forced val-BPB guard
  is the drift detector.

Data and rewards follow the VAPO paper exactly: prompts are DAPO-Math-17K,
the terminal reward is the binary Minerva-style verifier on the EOS-truncated
generation, positives for the auxiliary LM loss are verifier-correct
trajectories, and AIME 2024 avg@k is the eval.  Each prompt group rolls out
as its own batch (prompts vary in length), and each group is one PPO
minibatch.  ``--rollout-only`` is the learnability gate: it reports whether
rewards vary within prompt groups at the initial 50/50 gate before any
update is attempted.

    python3 -m postraining.train_latent_vapo \
        --checkpoint ablation_results/<run>/pretraining_checkpoint.pt \
        --output postraining/runs/<name> [--rollout-only]
"""

from __future__ import annotations

import argparse
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
    module_answer_baselines,
    generalized_advantage_estimate,
    length_adaptive_lambda,
    load_unique_math_rows,
    masked_token_mean,
    per_dim_clipped_policy_loss,
    positive_example_lm_loss,
    validate_posttraining_context_budget,
)
from postraining.latent_rollout import (
    THOUGHT_SLOT,
    LatentRolloutBatch,
    assign_terminal_rewards,
    emitted_token_rows,
    half_forced_group_members,
    iter_length_aware_microbatches,
    refresh_old_statistics,
    replay_head_inputs,
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
    LatentThoughtModel,
    validate_renderer_checkpoint,
)
from postraining.model_io import fresh_trunk, load_model
from postraining.train_vapo import prompt_text
from postraining.value_model import SeparateCritic


EXECUTION_SCHEMA = "bf16_compiled_rollout_default_dynamic_replay/v3"
DEFAULT_BPB_GUARD_TOKENS = 2 * 1024 * 1024
# v2: the grading style follows each row's reward_model.style (Minerva for
# DAPO/AIME lineage data, official exact match for mathematics_dataset rows)
# instead of Minerva-normalizing everything.
REWARD_SCHEMA = POSTTRAIN_REWARD_SCHEMA


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
    """Epoch-shuffled DAPO prompt stream with a resumable cursor."""

    def __init__(self, rows: list[dict], seed: int):
        if not rows:
            raise ValueError("no math prompts loaded")
        self.rows = rows
        self.seed = seed
        self.cursor = 0
        self._epoch = -1
        self._order: list[int] = []

    def next_rows(self, count: int) -> list[dict]:
        picked = []
        while len(picked) < count:
            epoch, offset = divmod(self.cursor, len(self.rows))
            if epoch != self._epoch:
                self._epoch = epoch
                self._order = list(range(len(self.rows)))
                random.Random(self.seed * 1_000_003 + epoch).shuffle(self._order)
            picked.append(self.rows[self._order[offset]])
            self.cursor += 1
        return picked


def score_math_rollout(
    batch: LatentRolloutBatch,
    truth: str,
    tokenizer,
    stop_ids: tuple[int, ...],
    style: str = "minerva",
) -> None:
    """Binary verifier rewards only for explicitly terminated emissions."""
    scores = []
    for emitted in emitted_token_rows(batch):
        correct, _ = verify_terminated_answer(
            emitted, truth, tokenizer, stop_ids, style
        )
        scores.append(float(correct))
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

    def mean(key: str) -> float:
        return sum(metric[key] for metric in metrics) / len(metrics)

    def mean_product(left: str, right: str) -> float:
        return sum(
            metric[left] * metric[right] for metric in metrics
        ) / len(metrics)

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
    last = metrics[-1]
    gate_contribution = mean_product("gate_loss", "gate_pg_coef")
    renderer_contribution = mean("renderer_loss")
    thought_contribution = mean_product("thought_loss", "thought_pg_coef")
    positive_lm_contribution = mean_product(
        "positive_lm_loss", "positive_lm_weight"
    )
    return {
        "loss/gate_weighted": gate_contribution,
        "loss/renderer": renderer_contribution,
        "loss/thought_weighted": thought_contribution,
        "loss/positive_lm_weighted": positive_lm_contribution,
        "loss/actor_total": (
            gate_contribution
            + renderer_contribution
            + thought_contribution
            + positive_lm_contribution
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
        "behavior/gate_entropy": _weighted_metric_mean(
            metrics, "gate_entropy", "gate_action_count"
        ),
        "kl/gate_behavior": _weighted_metric_mean(
            metrics, "gate_behavior_kl", "gate_action_count"
        ),
        "kl/renderer_behavior": _weighted_metric_mean(
            metrics, "renderer_behavior_kl", "emit_action_count"
        ),
        "kl/thought_behavior_joint": _weighted_metric_mean(
            metrics, "thought_behavior_kl_joint", "thought_action_count"
        ),
        "kl/policy_behavior_per_action": _weighted_metric_mean(
            metrics, "policy_behavior_kl_per_action", "action_count"
        ),
        "clip/gate": _weighted_metric_mean(
            metrics, "gate_clip_fraction", "gate_action_count"
        ),
        "clip/renderer": _weighted_metric_mean(
            metrics, "renderer_clip_fraction", "emit_action_count"
        ),
        "clip/thought_coordinate_fraction": _weighted_metric_mean(
            metrics, "thought_clip_fraction", "thought_action_count"
        ),
        # Actor gradients accumulate across groups; only the final group has
        # the actual full pre-step norm. The critic steps per group instead.
        "grad/trunk": last["trunk_grad_norm"],
        "grad/renderer": last["renderer_grad_norm"],
        "grad/adapter": last["adapter_grad_norm"],
        "grad/gate": last["gate_grad_norm"],
        "grad/critic_mean": mean("critic_grad_norm"),
    }


def rollout_tensorboard_metrics(metrics: dict[str, float | int]) -> dict[str, float]:
    """Select semantic rollout signals while leaving rich telemetry in JSON."""
    return {
        "reward/mean": float(metrics["reward_mean"]),
        "reward/within_group_std": float(metrics["within_group_reward_std"]),
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


def bounded_epoch_order(group_count: int, remaining_steps: int) -> list[int]:
    """Shuffle one PPO epoch without advancing past the requested step cap."""
    if group_count < 1:
        raise ValueError("group_count must be positive")
    update_count = min(group_count, max(remaining_steps, 0))
    return torch.randperm(group_count).tolist()[:update_count]


def write_actor_tensorboard_metrics(
    tensorboard, dashboard: dict[str, float], epoch: int, step: int
) -> None:
    """Write one actor-step row without deterministic epoch-zero clutter."""
    if epoch == 0:
        refresh_drift = max(
            dashboard[tag]
            for tag in (
                "kl/gate_behavior",
                "kl/renderer_behavior",
                "kl/thought_behavior_joint",
                "clip/gate",
                "clip/renderer",
                "clip/thought_coordinate_fraction",
            )
        )
        tensorboard.add_scalar(
            "debug/behavior_refresh_max_drift", refresh_drift, step
        )
    for tag, value in dashboard.items():
        if epoch == 0 and (
            tag.startswith("kl/") or tag.startswith("clip/")
        ):
            continue
        # Advantages are frozen with the behavior rollout and are identical
        # across PPO epochs; chart them once.
        if epoch > 0 and tag.startswith("advantage/"):
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
    actor_lr: float,
    head_lr: float,
    renderer_lr: float,
    critic_lr: float,
    fused: bool = True,
) -> dict[str, torch.optim.Optimizer]:
    """The actor/critic optimizer layout.

    One actor AdamW in three groups: the pretrained trunk at ``actor_lr``,
    the fresh zero-init heads (gate, adapter) at ``head_lr``, and the
    renderer probe at ``renderer_lr`` (it drives token-PPO ratios directly
    and cannot ride the hotter trunk rate).  The probes live under
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
    head_parameters = list(wrapper.gate.parameters()) + list(
        wrapper.adapter.parameters()
    )
    return {
        "actor": torch.optim.AdamW(
            [
                {"params": trunk_parameters, "lr": actor_lr},
                {"params": head_parameters, "lr": head_lr},
                {"params": list(backbone.policy_probe.parameters()), "lr": renderer_lr},
            ],
            weight_decay=0.0,
            fused=fused,
        ),
        "critic": torch.optim.AdamW(
            critic.parameters(), lr=critic_lr, weight_decay=0.0, fused=fused,
        ),
    }


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
    actor_step: bool = True,
    loss_scale: float = 1.0,
    gae_lambda_alpha: float = 0.05,
    replay_max_trajectories: int = 32,
    replay_attention_budget: int = 4 * 1024 * 1024,
    replay_bucket: int = 1,
) -> dict[str, float]:
    """One minibatch update.

    The critic is a per-call supervised regression (no trust region) and
    always steps here.  The actor is trust-region-bound: with 16 groups x 2
    epochs, stepping it per minibatch would take 32 full-trunk steps against
    one frozen behavior policy and drive every ratio outside the clip
    (red-teamed) — so the trainer accumulates actor gradients across a whole
    epoch (``actor_step=False``, ``loss_scale=1/len(groups)``) and steps the
    actor once per epoch at the call site.  Standalone callers (tests) keep
    ``actor_step=True`` for the self-contained single-step behavior.

    The group is replayed in stable length-sorted shards because causal
    attention is quadratic in stream length. The planner bounds B*L^2 and
    trims every shard independently. Every objective keeps its denominator
    from the FULL prompt group, so sharding changes only the floating-point
    reduction order, not the loss or optimizer cadence.
    """
    backbone = wrapper.backbone
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
        "positive": positive.sum().clamp_min(1),
    }
    zero = batch.action_mask.new_zeros(())
    totals = {
        key: zero.clone()
        for key in (
            "value_loss", "value_sum", "value_target_sum", "target_entropy_sum",
            "gate_loss", "gate_clip", "gate_kl_sum", "gate_entropy_sum",
            "emit_probability_sum",
            "renderer_loss", "renderer_clip", "positive_lm", "thought_loss",
            "renderer_kl_sum", "thought_clip", "thought_kl_sum",
            "advantage_sum", "advantage_square_sum",
            "optional_think_advantage_sum", "forced_initial_think_advantage_sum",
            "thought_advantage_sum", "emit_advantage_sum", "target_square_sum",
            "residual_sum", "residual_square_sum",
        )
    }

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
        value_weight = local_action_count / denominators["action"]

        # The critic has no trust region, but it still takes exactly one step
        # per prompt group. Backward each weighted shard now and step below.
        value_logits = critic.value_logits(microbatch)
        value_ce = critic.support.cross_entropy(value_logits, value_targets)
        local_value_loss = masked_token_mean(value_ce, microbatch.action_mask)
        weighted_value_loss = value_weight * local_value_loss
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
            totals["value_loss"] += weighted_value_loss.detach()
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

        # The differentiable actor replay is separate from the critic graph,
        # lowering peak memory further. Gate/renderer losses backprop through
        # beliefs into the trunk; only the THINK-masked term consumes the
        # densely computed prediction projector.
        beliefs, predicted, stream_inputs, token_targets = replay_head_inputs(
            wrapper, microbatch
        )
        local_gate_count = microbatch.gate_mask.sum()
        gate_weight = local_gate_count / denominators["gate"]
        new_gate_logprobs = wrapper.gate.log_prob(
            microbatch.gate_actions.float(), beliefs
        )
        local_gate_loss, local_gate_clip, local_gate_kl = clipped_policy_loss(
            new_gate_logprobs,
            microbatch.old_gate_logprobs,
            advantages,
            microbatch.gate_mask,
        )
        weighted_gate_loss = gate_weight * local_gate_loss

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
        local_emit_count = microbatch.emit_mask.sum()
        renderer_weight = local_emit_count / denominators["emit"]
        (
            local_renderer_loss,
            local_renderer_clip,
            local_renderer_kl,
        ) = clipped_policy_loss(
            new_token_logprobs,
            microbatch.old_token_logprobs,
            advantages,
            microbatch.emit_mask,
        )
        weighted_renderer_loss = renderer_weight * local_renderer_loss

        micro_positive = microbatch.reward_scalar >= positive_reward_threshold
        positive_weight = micro_positive.sum() / denominators["positive"]
        local_positive_lm = positive_example_lm_loss(
            new_token_logprobs, microbatch.emit_mask, micro_positive
        )
        weighted_positive_lm = positive_weight * local_positive_lm

        think_mask = (
            (microbatch.gate_actions == THINK)
            & microbatch.action_mask.bool()
        )
        local_thought_count = think_mask.sum()
        thought_weight = local_thought_count / denominators["thought"]
        if bool(think_mask.any()):
            thought_means, thought_targets, _ = select_thought_actions(
                microbatch, predicted
            )
            compact_advantages = advantages[think_mask]
            if thought_pg_coef != 0.0:
                new_thought_logprobs = wrapper.transition.per_dim_log_prob(
                    thought_targets, thought_means
                )
                (
                    local_thought_loss,
                    local_thought_clip,
                    local_thought_joint_kl,
                ) = per_dim_clipped_policy_loss(
                    new_thought_logprobs,
                    microbatch.old_thought_logprobs[think_mask],
                    compact_advantages,
                    torch.ones_like(compact_advantages),
                )
                weighted_thought_loss = thought_weight * local_thought_loss
            else:
                # The renderer/trunk objectives can move the predicted
                # thought mean even when thought policy gradients are off.
                # Measure that drift without creating projector gradients.
                with torch.no_grad():
                    new_thought_logprobs = wrapper.transition.per_dim_log_prob(
                        thought_targets, thought_means.detach()
                    )
                    (
                        _,
                        local_thought_clip,
                        local_thought_joint_kl,
                    ) = per_dim_clipped_policy_loss(
                        new_thought_logprobs,
                        microbatch.old_thought_logprobs[think_mask],
                        torch.zeros_like(compact_advantages),
                        torch.ones_like(compact_advantages),
                    )
                weighted_thought_loss = predicted.detach().new_zeros(())
        else:
            # Preserve grad=None on the prediction projector when its policy
            # objective is disabled; a zero grad would let Adam apply stale
            # momentum despite there being no thought-content signal.
            weighted_thought_loss = predicted.detach().new_zeros(())
            local_thought_clip = predicted.detach().new_zeros(())
            local_thought_joint_kl = predicted.detach().new_zeros(())

        actor_total = (
            gate_pg_coef * weighted_gate_loss
            + weighted_renderer_loss
            + positive_lm_weight * weighted_positive_lm
            + thought_pg_coef * weighted_thought_loss
        )
        if not torch.isfinite(actor_total):
            raise RuntimeError(
                "non-finite actor loss before optimizer step: "
                f"gate={float(weighted_gate_loss)} "
                f"renderer={float(weighted_renderer_loss)} "
                f"positive_lm={float(weighted_positive_lm)} "
                f"thought={float(weighted_thought_loss)}"
            )
        (loss_scale * actor_total).backward()
        with torch.no_grad():
            totals["gate_loss"] += weighted_gate_loss.detach()
            totals["gate_clip"] += gate_weight * local_gate_clip
            totals["gate_kl_sum"] += local_gate_kl * local_gate_count
            totals["gate_entropy_sum"] += (
                wrapper.gate.entropy(beliefs) * microbatch.gate_mask
            ).sum()
            totals["emit_probability_sum"] += (
                wrapper.gate.emit_logit(beliefs).sigmoid()
                * microbatch.gate_mask
            ).sum()
            totals["renderer_loss"] += weighted_renderer_loss.detach()
            totals["renderer_clip"] += (
                renderer_weight * local_renderer_clip
            )
            totals["renderer_kl_sum"] += local_renderer_kl * local_emit_count
            totals["positive_lm"] += weighted_positive_lm.detach()
            totals["thought_loss"] += weighted_thought_loss.detach()
            totals["thought_clip"] += thought_weight * local_thought_clip
            totals["thought_kl_sum"] += (
                local_thought_joint_kl * local_thought_count
            )

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
        "value_loss": totals["value_loss"],
        "value_mean": totals["value_sum"] / action_denom,
        "value_target_mean": value_target_mean,
        "value_target_variance": target_variance,
        "value_residual_mean": residual_mean,
        "value_residual_variance": residual_variance,
        "explained_variance": explained_variance,
        "value_excess_ce": (
            totals["value_loss"]
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
        optimizers["critic"].step()
        return scalar_tensors_to_floats(metric_tensors)

    # Norms are cumulative across an accumulation epoch; the last minibatch
    # of an epoch reports the full pre-step norm.  Probes are registered
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
    }
    optimizers["critic"].step()
    if actor_step and "actor" in optimizers:
        optimizers["actor"].step()
    metric_tensors.update(
        gate_loss=totals["gate_loss"],
        gate_clip_fraction=totals["gate_clip"],
        gate_entropy=totals["gate_entropy_sum"] / denominators["gate"],
        emit_probability=totals["emit_probability_sum"] / denominators["gate"],
        renderer_loss=totals["renderer_loss"],
        renderer_clip_fraction=totals["renderer_clip"],
        positive_lm_loss=totals["positive_lm"],
        positive_fraction=positive.float().mean(),
        thought_loss=totals["thought_loss"],
        thought_clip_fraction=totals["thought_clip"],
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
            (
                totals["gate_kl_sum"]
                + totals["renderer_kl_sum"]
                + totals["thought_kl_sum"]
            )
            / denominators["action"]
        ),
        advantage_mean=advantage_mean,
        advantage_std=advantage_variance.sqrt(),
        reward=batch.reward_scalar.mean(),
        **grad_norms,
    )
    metrics = scalar_tensors_to_floats(metric_tensors)
    metrics.update(
        gate_pg_coef=float(gate_pg_coef),
        thought_pg_coef=float(thought_pg_coef),
        positive_lm_weight=float(positive_lm_weight),
    )
    return metrics


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
        "reward_schema": REWARD_SCHEMA,
        "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
        "rollout_policy_schema": ROLLOUT_POLICY_SCHEMA,
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
    parser.add_argument("--prompts-per-rollout", type=int, default=16)
    # Each prompt group is one PPO minibatch (prompts vary in length, so
    # groups are separate batches end to end).
    parser.add_argument("--samples-per-prompt", type=int, default=32)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    # Total generated-slot budget per trajectory (thinks + emits).  Thinking
    # is never forcibly interrupted; overthinking costs emitted tokens and
    # therefore reward. 0 means 4x the emit cap; the default is the explicit
    # 4096-slot side of the 1024-prompt + 4096-stream context contract.
    parser.add_argument(
        "--max-stream-steps", type=int, default=POSTTRAIN_STREAM_TOKENS
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    # Pretrained trunk (embeddings, blocks, projector): a conservative RL
    # fine-tuning rate — pretraining ran Muon at 0.04, but AdamW at equal lr
    # is far hotter in spectral norm, and PPO ratios must stay inside the
    # trust region across an accumulation epoch.
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    # Fresh zero-init heads (gate, thought adapter) start from nothing and
    # tolerate a hotter rate.
    parser.add_argument("--head-lr", type=float, default=1e-4)
    # The renderer probe drives token-PPO ratios directly; keep it at the
    # v1 rate rather than folding it into the 10x-hotter trunk group.
    parser.add_argument("--renderer-lr", type=float, default=1e-6)
    # The VAPO paper's 2e-6 presumes a value model initialized from pretrained
    # weights; this critic trains from scratch and needs a scratch-training lr.
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--value-bins", type=int, default=101)
    # HL-Gauss projection sigma as a fraction of bin width (cleanrl v215 /
    # Dreamer4 default).
    parser.add_argument("--value-sigma-ratio", type=float, default=2.0)
    # Head bias starts at the projected prior; binary verifier rewards start
    # near-zero for a small model, and a prior near the expected reward mean
    # removes the early decode transient a far-off prior causes.
    parser.add_argument("--value-prior", type=float, default=0.05)
    # Fixed thought-noise scale (log-sigma).  Never trained — no beta-NLL,
    # no entropy; the trust region is the only constraint on the thought
    # policy.  The prediction-error-matched -0.5 (sigma 0.61) gives a
    # 512-dim offset of norm ~13.7 against thought norms ~22.6 — hot enough
    # to corrupt long think runs and (red-teamed) turn the gate against
    # thinking; -1.5 (sigma 0.22) keeps exploration without drowning the
    # signal.  Smaller sigma also tightens the effective trust region on
    # the mean (ratio ~ delta-mean/sigma), pairing with the low actor lr.
    parser.add_argument("--thought-log-sigma", type=float, default=-1.5)
    # Weight of the per-dim clipped PPO term on thought content.  The
    # gradient flows through the prediction path into the whole trunk; 0
    # disables reward-training of thought content (control arm — the trunk
    # still trains through the token PPO term).
    parser.add_argument("--thought-pg-coef", type=float, default=1.0)
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
    # VAPO paper: 50 value-pretraining steps before policy updates.
    parser.add_argument("--value-warmup-steps", type=int, default=50)
    # The BPB guard is a do-no-harm regression check, not an optimization
    # target. A deterministic 2M-token prefix takes ~2.5s on the 5090 versus
    # ~75s for all 62M validation tokens, while still giving the guard ample
    # precision to detect renderer drift. Use 0 explicitly for a final,
    # challenge-comparable full-validation measurement.
    parser.add_argument("--bpb-every", type=int, default=320)
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
    parser.add_argument("--bench-every", type=int, default=80)
    parser.add_argument("--bench-samples", type=int, default=8)
    parser.add_argument(
        "--bench-max-tokens", type=int, default=POSTTRAIN_RESPONSE_TOKENS
    )
    # Batch multiple problem groups into the same left-padded GPU rollout.
    # This is a trajectory rather than prompt count so avg@8 and avg@32 use
    # comparable memory; replay-free eval storage makes 128 rows practical.
    parser.add_argument("--eval-batch-trajectories", type=int, default=128)
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
    # One PPO outer rollout advances 32 logical group updates by default;
    # checkpoint every completed rollout so a later collect cannot erase the
    # previous one. Warmup checkpoints are smaller (actor Adam has no state).
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
    # Prompt groups rolled out together as one left-padded batch (measured:
    # the sequential per-group rollout is launch-bound at ~140 W, so stepping
    # groups*samples rows per launch is the utilization lever.
    parser.add_argument("--rollout-groups", type=int, default=8)
    parser.add_argument("--rollout-only", action="store_true")
    parser.add_argument("--gate-min-within-group-reward-std", type=float, default=0.01)
    parser.add_argument(
        "--actor-init", default=None,
        help="initialize only the actor from a latent-VAPO checkpoint, while "
        "resetting critic/optimizers and continuing its unused prompt stream",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.replay_max_trajectories < 1:
        parser.error("--replay-max-trajectories must be positive")
    if args.replay_attention_budget < 1:
        parser.error("--replay-attention-budget must be positive")
    if args.replay_bucket < 1:
        parser.error("--replay-bucket must be positive")
    if args.warmup_save_every < 1:
        parser.error("--warmup-save-every must be positive")
    if args.eval_batch_trajectories < 1:
        parser.error("--eval-batch-trajectories must be positive")
    if args.bpb_val_tokens < 0:
        parser.error("--bpb-val-tokens must be nonnegative")
    if args.actor_init and args.resume:
        parser.error("--actor-init and --resume are mutually exclusive")
    if args.samples_per_prompt < 2 or args.samples_per_prompt % 2:
        parser.error("--samples-per-prompt must be even for the 50/50 forced split")
    for name, samples in (
        ("--aime-samples", args.aime_samples),
        ("--bench-samples", args.bench_samples),
    ):
        if samples < 2 or samples % 2:
            parser.error(f"{name} must be even for the 50/50 forced split")
    if (
        args.bench_every > 0
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
    # Full-model RL: every policy parameter trains — trunk, embeddings,
    # projector, renderer probe, gate, and adapter.  Only the backbone's
    # critic probe stays frozen (unused — the critic is a separate model).
    wrapper = LatentThoughtModel(backbone).to(device)
    # No module here behaves differently under train(): pin eval mode once so
    # the training flag (a dynamo guard) never flips between the step-0 evals
    # and the training loop and re-specializes the compiled step.
    wrapper.eval()
    wrapper.transition.log_sigma.fill_(args.thought_log_sigma)
    if not 0.0 < args.init_think_probability < 1.0:
        raise SystemExit("--init-think-probability must be strictly inside (0, 1)")
    with torch.no_grad():
        # P(EMIT) = sigmoid(bias) while the zero-init weights ignore the belief.
        wrapper.gate.head.bias.fill_(
            math.log((1.0 - args.init_think_probability) / args.init_think_probability)
        )
    actor_init_payload = None
    actor_init_provenance = None
    if args.actor_init:
        actor_init_payload = torch.load(
            args.actor_init, map_location="cpu", weights_only=False
        )
        validate_renderer_checkpoint(actor_init_payload, args.actor_init)
        wrapper.load_state_dict(actor_init_payload["model"], strict=True)
        # The run controls exploration noise even when actor weights are
        # initialized from an earlier reward schema.
        wrapper.transition.log_sigma.fill_(args.thought_log_sigma)
        actor_init_provenance = {
            "checkpoint": str(args.actor_init),
            "source_step": actor_init_payload.get("step"),
            "source_reward_schema": actor_init_payload.get("reward_schema"),
            "sampler_cursor": int(actor_init_payload["sampler_cursor"]),
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

    optimizers = build_optimizers(
        wrapper, critic,
        actor_lr=args.actor_lr, head_lr=args.head_lr,
        renderer_lr=args.renderer_lr, critic_lr=args.critic_lr,
    )

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
    bench_rows = (
        load_unique_math_rows(args.bench_data)
        if args.bench_every > 0 and not args.rollout_only
        else []
    )
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
    sampler = MathPromptSampler(math_rows, args.seed)
    if actor_init_payload is not None:
        source_args = actor_init_payload.get("args", {})
        if hasattr(source_args, "__dict__"):
            source_args = vars(source_args)
        source_math_data = source_args.get("math_data")
        if (
            source_math_data is None
            or Path(source_math_data).resolve() != Path(args.math_data).resolve()
        ):
            raise ValueError(
                "--actor-init must use the same math dataset so its sampler "
                "cursor denotes the remaining unused prompts"
            )
        if int(source_args.get("seed", -1)) != args.seed:
            raise ValueError(
                "--actor-init must use the source seed so its deterministic "
                "prompt order can continue without reuse"
            )
        source_exclusions = source_args.get("exclude_modules") or ""
        if source_exclusions != args.exclude_modules:
            raise ValueError(
                "--actor-init must use the same --exclude-modules setting"
            )
        actor_cursor = int(actor_init_payload.get("sampler_cursor", -1))
        if not 0 <= actor_cursor < len(math_rows):
            raise ValueError(
                "--actor-init source has no valid unused first-epoch prompt "
                f"cursor: {actor_cursor} for {len(math_rows)} rows"
            )
        sampler.cursor = actor_cursor
        if args.bpb_only:
            planned_prompt_count = 0
        elif args.rollout_only:
            planned_prompt_count = args.prompts_per_rollout
        else:
            actor_collections = math.ceil(
                args.steps / (args.ppo_epochs * args.prompts_per_rollout)
            )
            planned_prompt_count = args.prompts_per_rollout * (
                args.value_warmup_steps + actor_collections
            )
        if actor_cursor + planned_prompt_count > len(math_rows):
            raise ValueError(
                "--actor-init run would cross the source dataset's first "
                "epoch and reuse prompts: cursor "
                f"{actor_cursor} + planned {planned_prompt_count} > "
                f"{len(math_rows)} rows"
            )
        print(
            f"actor initialized from {args.actor_init} at source step "
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
    warmup_step = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_renderer_checkpoint(payload, args.resume)
        if payload.get("reward_schema") != REWARD_SCHEMA:
            raise ValueError(
                f"resume checkpoint reward schema must be {REWARD_SCHEMA!r}; "
                f"got {payload.get('reward_schema')!r}"
            )
        wrapper.load_state_dict(payload["model"], strict=True)
        critic.load_state_dict(payload["critic"], strict=True)
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][name])
        # log_sigma is a buffer (inside "model") and per-group lr rides along
        # in the optimizer states, so the loads above just clobbered both with
        # the checkpoint's values.  Reassert the CLI: these are the documented
        # cross-run knobs (external sigma annealing, lr changes on resume).
        wrapper.transition.log_sigma.fill_(args.thought_log_sigma)
        for group, lr in zip(
            optimizers["actor"].param_groups,
            (args.actor_lr, args.head_lr, args.renderer_lr),
            strict=True,
        ):
            group["lr"] = lr
        for group in optimizers["critic"].param_groups:
            group["lr"] = args.critic_lr
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

    # Rollout and evaluation both keep narrow prefix attention while making
    # prefix and batch dimensions symbolic. They use separate compiled bound
    # methods so eval's eager-fallback latch cannot disable training rollout.
    rollout_step_core = None
    if args.rollout_compile:
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, 64
        )
        rollout_step_core = torch.compile(
            wrapper.step_core,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
            dynamic=True,
        )
    eval_step_core = None
    if args.eval_compile:
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, 64
        )
        eval_step_core = torch.compile(
            wrapper.step_core,
            mode="max-autotune-no-cudagraphs",
            fullgraph=True,
            dynamic=True,
        )

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
        # compiled artifact — the property that makes epoch-0 PPO ratios
        # exactly one (see refresh_old_statistics on why it also runs
        # grad-enabled).
        globals()["replay_head_inputs"] = compiled_replay
        postraining.latent_rollout.replay_head_inputs = compiled_replay

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output / "metrics.jsonl")
    warmup_purge_step = None
    if args.resume and start_step == 0 and warmup_step > 0:
        removed = logger.purge_value_warmup_after(warmup_step)
        if removed:
            print(
                f"purged {removed} stale warmup metrics after step "
                f"{warmup_step}",
                flush=True,
            )
        warmup_purge_step = warmup_step + 1
    tensorboard = SummaryWriter(
        output / "tensorboard", purge_step=warmup_purge_step
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "phase": "latent_vapo_dapo",
                "execution_schema": EXECUTION_SCHEMA,
                "reward_schema": REWARD_SCHEMA,
                "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
                "rollout_policy_schema": ROLLOUT_POLICY_SCHEMA,
                "args": vars(args),
                "base": {
                    "checkpoint": str(args.checkpoint),
                    "architecture": backbone.architecture,
                },
                "actor_init": actor_init_provenance,
                "critic": {
                    "init": "scratch",
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

    def finish_group(
        batch: LatentRolloutBatch,
        row: dict,
        refresh_statistics: bool,
    ) -> LatentRolloutBatch:
        batch = trim_stream(batch, multiple=trim_multiple)
        score_math_rollout(
            batch, row["reward_model"]["ground_truth"], tokenizer, stop_ids,
            answer_style(row),
        )
        # Stepwise rollout and parallel replay disagree numerically at
        # bf16 scale; recompute the stored PPO statistics through the
        # update-step replay path so epoch-0 ratios are exactly one.
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

    def _collect(refresh_statistics: bool) -> list[LatentRolloutBatch]:
        """One rollout: a scored prompt group per sampled DAPO problem."""
        rollout_rows = sampler.next_rows(args.prompts_per_rollout)
        groups = []
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
                groups.append(finish_group(batch, row, refresh_statistics))
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
            split_groups = split_rollout_groups(batched, samples, prompt_lengths)
            # Every split owns cloned storage; drop the much larger
            # groups*samples rollout before the first full-stream replay.
            del batched
            for group, row in zip(split_groups, chunk, strict=True):
                groups.append(finish_group(group, row, refresh_statistics))
        return groups

    def training_autocast():
        return torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        )

    def collect(
        refresh_statistics: bool = True,
    ) -> list[LatentRolloutBatch]:
        original_step_core = wrapper.step_core
        if rollout_step_core is not None:
            wrapper.step_core = rollout_step_core
        try:
            # rollout_continuations itself is no-grad, while the optional
            # refresh below must remain grad-enabled so refresh/update share
            # one compiled replay specialization and identical PPO numerics.
            with training_autocast():
                return _collect(refresh_statistics)
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
        wrapper.eval()
        eval_started = time.perf_counter()
        metrics = evaluate_aime_latent(
            wrapper, tokenizer, aime_rows, args.aime_samples, args.aime_max_tokens,
            aime_stream_steps, args.aime_chunk, args.seed, device,
            prompt_tokens=args.prompt_tokens,
            batch_trajectories=args.eval_batch_trajectories,
            compiled_step_core=eval_step_core,
            answer_style_override="aime",
        )
        eval_seconds = time.perf_counter() - eval_started
        tensorboard.add_scalar("perf/aime_eval_seconds", eval_seconds, step)
        logger.log(type="aime", step=step, seconds=eval_seconds, **metrics)
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

    def bench_eval(step: int) -> None:
        wrapper.eval()
        captured_attempts: list[dict[str, object]] = []
        eval_started = time.perf_counter()
        metrics = evaluate_aime_latent(
            wrapper, tokenizer, bench_rows, args.bench_samples,
            args.bench_max_tokens, bench_stream_steps, args.bench_samples,
            args.seed, device, prompt_tokens=args.prompt_tokens,
            batch_trajectories=args.eval_batch_trajectories,
            compiled_step_core=eval_step_core,
            captured_attempts=captured_attempts,
        )
        eval_seconds = time.perf_counter() - eval_started
        tensorboard.add_scalar("perf/bench_eval_seconds", eval_seconds, step)
        write_benchmark_report(
            output,
            step,
            metrics,
            captured_attempts,
            reward_schema=REWARD_SCHEMA,
        )
        logger.log(type="bench", step=step, seconds=eval_seconds, **metrics)
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

    if start_step == 0 and warmup_step == 0:
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
            groups = collect(refresh_statistics=False)
            torch.cuda.synchronize()
            collect_seconds = time.perf_counter() - collect_started
            update_started = time.perf_counter()
            group_metrics = [
                training_update(
                    wrapper,
                    critic,
                    group,
                    optimizers,
                    value_only=True,
                    replay_max_trajectories=args.replay_max_trajectories,
                    replay_attention_budget=args.replay_attention_budget,
                    replay_bucket=args.replay_bucket,
                )
                for group in groups
            ]
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
                "value_warmup/critic_grad_norm_mean": metrics[
                    "critic_grad_norm"
                ],
            }
            for tag, value in warmup_dashboard.items():
                tensorboard.add_scalar(tag, value, warmup)
            # The optimizer state is sufficient for the next update and for
            # checkpoints. Keeping the last group's gradient buffers alive
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
        groups = collect()
        torch.cuda.synchronize()
        collect_seconds = time.perf_counter() - collect_started
        rollout_metrics = aggregate_diagnostics(
            groups, args.samples_per_prompt, stop_ids
        )
        logger.log(
            type="rollout", step=step,
            collect_seconds=collect_seconds,
            **rollout_metrics,
        )
        tensorboard.add_scalar("perf/collect_seconds", collect_seconds, step)
        rollout_dashboard = rollout_tensorboard_metrics(rollout_metrics)
        for tag, value in rollout_dashboard.items():
            tensorboard.add_scalar(tag, value, step)

        for epoch in range(args.ppo_epochs):
            # One accumulated actor step per epoch: stepping the full trunk
            # per minibatch would take ppo_epochs x len(groups) trust-region
            # steps against a single frozen behavior policy and saturate the
            # clip (red-teamed).  The critic has no trust region and still
            # steps per minibatch inside update_minibatch.
            epoch_order = bounded_epoch_order(len(groups), args.steps - step)
            if not epoch_order:
                break
            optimizers["actor"].zero_grad(set_to_none=True)
            epoch_metrics: list[dict[str, float]] = []
            for index in epoch_order:
                step += 1
                metrics = training_update(
                    wrapper, critic, groups[index], optimizers,
                    positive_lm_weight=args.positive_lm_weight,
                    positive_reward_threshold=args.positive_reward_threshold,
                    thought_pg_coef=args.thought_pg_coef,
                    gate_pg_coef=0.0 if step <= args.gate_freeze_steps else 1.0,
                    actor_step=False,
                    loss_scale=1.0 / len(epoch_order),
                    gae_lambda_alpha=args.gae_lambda_alpha,
                    replay_max_trajectories=args.replay_max_trajectories,
                    replay_attention_budget=args.replay_attention_budget,
                    replay_bucket=args.replay_bucket,
                )
                # Epoch 0 runs against the refresh-computed statistics with
                # the actor untouched, so every PPO ratio is exactly one and
                # every clip fraction exactly zero.  A nonzero here means the
                # refresh and update forwards diverged (e.g. compiled
                # artifacts split) — the drift refresh exists to remove.
                if epoch == 0:
                    for guard in (
                        "gate_clip_fraction",
                        "renderer_clip_fraction",
                        "thought_clip_fraction",
                    ):
                        if metrics[guard] > 1e-6:
                            print(
                                f"WARNING step {step}: epoch-0 {guard}="
                                f"{metrics[guard]:.3e} (expected exactly 0; "
                                "refresh/update code paths diverged)",
                                flush=True,
                            )
                logger.log(
                    type="train", step=step,
                    seconds=time.perf_counter() - started,
                    peak_vram_bytes=torch.cuda.max_memory_allocated(),
                    **metrics,
                )
                epoch_metrics.append(metrics)
            actor_dashboard = aggregate_actor_tensorboard_metrics(epoch_metrics)
            optimizers["actor"].step()
            write_actor_tensorboard_metrics(
                tensorboard, actor_dashboard, epoch, step
            )

        # No operation below consumes rollout tensors. Release them before
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
    save_checkpoint(
        output / "latent_vapo_checkpoint.pt", wrapper, critic,
        optimizers, step, args, sampler, warmup_step,
        actor_init_provenance,
    )
    tensorboard.close()


if __name__ == "__main__":
    main()
