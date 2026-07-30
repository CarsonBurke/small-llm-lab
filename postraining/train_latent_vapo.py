"""Latent-thought VAPO: full-model PPO over THINK/EMIT gates, tokens, thoughts.

The WHOLE deployed policy path trains at RL time — no frozen trunk. A fresh
linear head owns the Gaussian thought mean instead of reusing pretraining's
next-token latent predictor, so reward can shape a scratch representation
without inheriting that discrete-token target.

- PPO trains everything through one differentiable teacher-forced replay.
  Gate and content log probabilities form one joint action probability at
  each stream position. EMIT clips its joint gate+token ratio. THINK (v24)
  trains under a Dreamer4-style sampled reverse-KL penalty on the thought
  policy — ``KL(behavior || current)`` via the k3 estimator, weighted by
  ``--thought-reverse-kl-coef`` — and NO surrogate-side trust region. That
  is deliberate: the v23 projection bounded only the projected mean inside
  the surrogate, while the raw acting policy drifted through the
  Muon-owned trunk under the LM/gate/renderer losses (job 363: median raw
  KL 0.0424 nats, 68% of updates above 0.03). The penalty is the only term
  that prices that channel directly.

  EXACTLY ONE THINK trust mechanism is active per run. The surrogate-side
  modes (``--thought-clip-mode projected``, the v23 TRPL-style Mahalanobis
  mean projection; ``joint`` and ``per_dim``, the v21/v22 ratio clips) and
  the reverse-KL penalty are alternative SOLUTIONS to the same problem,
  never layers of one — mixing them makes neither term's contribution
  attributable. The rule is a biconditional enforced in both
  ``run_latent_vapo`` and argv parsing: nonzero coefficient <=> clip mode
  ``none``. The surrogate-side modes remain available as ablation arms by
  pairing them with ``--thought-reverse-kl-coef 0``.

  Known gap either way: sigma is not trust-regioned beyond the
  tanh-bounded log-sigma range and the +/-2 ratio guard (a numerical bound
  on the 512-D Gaussian tail that bounds by zeroing a saturated action's
  gradient). Acceptable while sigma is effectively pinned.
  Optional THINK clips its shared gate once as a separate 1-D factor.
- The critic is a SEPARATE from-scratch model (same architecture class,
  fresh weights, fully trainable, no SIGReg or latent prediction) trained
  purely by HL-Gauss cross-entropy on [0, 1] value targets.  It values
  every stream position — vocab AND latent — which is what gives thinking
  its training signal. The support (v22) anchors bin CENTERS at exactly 0
  and 1 with margin bins beyond each — Dreamer3's exact-zero bucket applied
  to both ends of the unit range — so the dominant exact-0/exact-1 verifier
  targets project symmetrically instead of decoding a truncation bias
  inward; labels smooth at sigma_ratio 1.0.
- An optional Bernoulli gate-entropy bonus can preserve THINK exploration;
  there is no continuous-policy entropy bonus or beta-NLL. A sampled reverse
  KL constrains aggregate drift of the 512-D Gaussian against the frozen
  rollout behavior policy, complementing factorwise PPO clipping.
  A weak orthogonal belief-conditioned head learns diagonal per-dimension
  thought log-sigma around its configured initial level.
- No pretraining anchor: SIGReg and the latent target-prediction objective are
  dropped at RL time. The fresh mean and recurrent policy train purely on
  their ability to think; the teacher-forced val-BPB guard is the drift
  detector.

Prompts are DAPO-Math-17K. An EOS-terminated, verifier-correct final Answer:
field receives reward 1; a wrong but strictly numeric final field receives
bounded distance shaping of at most 0.1. Malformed or unterminated responses
receive zero. AIME 2024 avg@k is the eval. Prompt groups roll out and replay
separately because their lengths differ, then accumulate into the shared
actor/critic optimizer minibatch. ``--rollout-only`` reports whether rewards
vary within prompt groups at the initial 50/50 gate before any update is
attempted.

    python3 -m postraining.train_latent_vapo \
        --checkpoint ablation_results/<run>/pretraining_checkpoint.pt \
        --output postraining/runs/<name> [--rollout-only]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
import torch.nn.functional as F
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
    write_benchmark_report,
)
from postraining.core import (
    POSTTRAIN_REWARD_SCHEMA,
    JsonlLogger,
    POSTTRAIN_CONTEXT_TOKENS,
    POSTTRAIN_PROMPT_TOKENS,
    POSTTRAIN_RESPONSE_TOKENS,
    answer_style,
    clipped_policy_loss,
    encode_prompt,
    extract_final_answer,
    deterministic_math_subset,
    generalized_advantage_and_return_targets,
    length_adaptive_lambda,
    load_posttraining_tokenizer,
    load_unique_math_rows,
    modal_answer_baseline,
    module_answer_baselines,
    nearby_numeric_reward,
    parse_numeric_answer,
    validate_posttraining_context_budget,
    verify_answer,
)
from postraining.hl_gauss import anchored_unit_geometry
from postraining.latent_rollout import (
    PAD_SLOT,
    THOUGHT_SLOT,
    LatentRolloutBatch,
    ReplayPlan,
    assign_terminal_rewards,
    build_replay_plan,
    compact_emit_token_logprobs,
    compact_next_slots,
    compact_slots,
    compact_stream_to_device,
    compact_thought_actions,
    emitted_token_rows,
    pack_rollout_groups_for_replay,
    iter_planned_replay_microbatches,
    refresh_old_statistics,
    replay_head_inputs,
    scatter_replay_statistics,
    scatter_slots,
    rollout_continuations,
    split_rollout_groups,
    trim_stream,
)
from postraining.latent_thought import (
    EMIT,
    THINK,
    RENDERER_FEATURES_SCHEMA,
    THOUGHT_DISTRIBUTION_SCHEMA,
    THOUGHT_MEAN_SCHEMA,
    DecodeRangeMask,
    LatentThoughtModel,
    migrate_legacy_wrapper_checkpoint,
    rollout_policy_schema_for_mode,
    transform_thought_action,
    validate_renderer_checkpoint,
)
from postraining.rollout_scheduler import (
    ContinuousScheduleStats,
    rollout_continuous_refill_groups,
)
from postraining.model_io import fresh_trunk, load_model
from postraining.muon import Muon
from postraining.reasoning_modes import (
    mode_rollout_budget,
    training_rollout_budget,
)
from postraining.provenance import capture_source_provenance
from postraining.runtime.profiling import (
    PROFILE_SCHEMA,
    DisabledProfiler,
    RunProfiler,
    device_timed,
    elapsed_seconds,
)
from postraining.train_vapo import prompt_text
from postraining.value_model import SeparateCritic
from postraining.vapo.objectives import (
    exact_bernoulli_behavior_kl,
    joint_action_logprobs,
    joint_thought_policy_loss,
    per_dimension_thought_policy_loss,
    project_thought_means,
    projected_thought_policy_loss,
    sampled_reverse_kl,
)
from postraining.vapo.config import build_arg_parser, validate_args
from postraining.vapo.schemas import (
    ACTOR_OBJECTIVE_SCHEMA,
    PERFORMANCE_COMPATIBLE_EXECUTION_SCHEMA,
    PREVIOUS_EXECUTION_SCHEMA,
    PROMPT_ORDER_SCHEMA,
    REPLAY_NUMERICS_SCHEMA,
    ZERO_AFFINE_EXECUTION_SCHEMA,
    execution_schema_for_adapter,
    migrate_anchored_value_resume,
    optimizer_schema_for_trunk_optimizer,
    resume_execution_schema_compatible,
    resume_replay_schema_compatible,
    value_support_geometry_matches,
)


# Refreshed device minibatches held for their update instead of the
# scatter-to-CPU/repack/re-upload round trip. Bounded because packed batch
# bytes track the pool's longest stream (a think-heavy pool can quadruple
# them); past the budget the refresh loop falls back to the scatter path
# for the remaining minibatches. VRAM-resident only between a pool's
# refresh and its last update — never across a collection.
RETAINED_MINIBATCH_BUDGET_BYTES = 8 << 30
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
    """Sequential prompt stream with a resumable monotonic cursor.

    The first epoch walks ``rows`` in their given order, so existing one-pass
    runs, resumes, and curriculum cursors keep their exact historical stream.
    When a run's step budget outlasts the dataset, later epochs reuse every
    prompt exactly once per epoch in a deterministic per-epoch shuffle derived
    from ``seed`` — the cursor keeps counting total prompts consumed, so
    checkpoints resume mid-epoch without replaying or skipping prompts.
    """

    def __init__(
        self,
        rows: list[dict],
        seed: int,
        dataset_identity: str | None = None,
    ):
        if not rows:
            raise ValueError("no math prompts loaded")
        self.rows = rows
        self.seed = seed
        self.cursor = 0
        self.dataset_identity = dataset_identity
        self._order_epoch: int | None = None
        self._order: list[int] | None = None

    @property
    def epoch(self) -> int:
        return self.cursor // len(self.rows)

    def _epoch_order(self, epoch: int) -> list[int]:
        if epoch != self._order_epoch:
            order = list(range(len(self.rows)))
            if epoch > 0:
                random.Random(self.seed * 1_000_003 + epoch).shuffle(order)
            self._order_epoch, self._order = epoch, order
        assert self._order is not None
        return self._order

    def next_rows(self, count: int) -> list[dict]:
        if count < 0:
            raise ValueError("prompt count must be nonnegative")
        picked: list[dict] = []
        while count:
            epoch, offset = divmod(self.cursor, len(self.rows))
            take = min(count, len(self.rows) - offset)
            order = self._epoch_order(epoch)
            picked.extend(self.rows[i] for i in order[offset:offset + take])
            self.cursor += take
            count -= take
        return picked


def answer_prefix_token_ids(tokenizer) -> tuple[int, ...]:
    """The ``"\\nAnswer:"`` ids as they tokenize inside a QA document.

    SentencePiece is boundary-sensitive: the sp1024 model encodes standalone
    ``"\\nAnswer:"`` as ``A n s w er :`` with the newline dropped, while the
    same substring inside the mathmix documents the backbone pretrained on
    (``"{question}\\nAnswer: {answer}"``) tokenizes as ``▁An s w er :``.
    Teacher-forcing the standalone ids would place the policy in a prompt
    state it never saw.  Deriving the ids from document-shaped encodings —
    and requiring two different question endings to agree — keeps the forced
    prefix on the pretraining distribution.
    """
    tails = []
    for context_text in ("?", "."):
        context = tokenizer.encode(context_text)
        combined = tokenizer.encode(context_text + "\nAnswer:")
        if combined[: len(context)] != context:
            raise RuntimeError(
                "tokenizer merged across the Answer: prefix boundary; "
                "cannot derive stable in-context prefix ids"
            )
        tails.append(tuple(combined[len(context):]))
    if tails[0] != tails[1]:
        raise RuntimeError(
            "the in-context Answer: prefix tokenizes differently after "
            f"different question endings: {tails[0]} vs {tails[1]}"
        )
    return tails[0]


def score_math_rollout(
    batch: LatentRolloutBatch,
    truth: str,
    tokenizer,
    stop_ids: tuple[int, ...],
    style: str = "minerva",
    nearby_reward_max: float = 0.1,
    solution_prefix_ids: tuple[int, ...] = (),
) -> None:
    """Exact verifier reward plus bounded final-answer numeric proximity.

    ``solution_prefix_ids`` are teacher-forced solution tokens that live at
    the end of the prompt (the none-mode ``Answer:`` prefix): the emitted
    continuation alone never contains them, so they rejoin the decode before
    the verifier parses a final answer.
    """
    scores = []
    for emitted in emitted_token_rows(batch):
        stop_cut = next(
            (index for index, token in enumerate(emitted) if token in stop_ids),
            None,
        )
        if stop_cut is None:
            scores.append(0.0)
            continue
        solution = tokenizer.decode(
            list(solution_prefix_ids) + emitted[: stop_cut + 1]
        )
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
    *,
    refreshed_statistics: bool = True,
) -> dict[str, float | int]:
    """Pool-level rollout metrics.

    ``refreshed_statistics`` says whether ``refresh_old_statistics`` filled
    ``batch.old_values``. It has no default meaning worth guessing: an
    unrefreshed batch carries zeros there, and reporting their mean as
    ``old_value_mean`` would publish a hard 0.0 that reads exactly like a
    critic collapsed to zero. The key is omitted instead.
    """
    stop_decisions = batch.stop_mask.sum().clamp_min(1)
    stop_set = set(stop_ids)
    # Fraction of rows that terminated themselves (emitted BOS or EOS)
    # rather than exhausting the token/stream budget.
    ended = [
        float(any(token in stop_set for token in row))
        for row in emitted_token_rows(batch)
    ]
    generated = batch.action_mask.bool()
    continue_fraction = float(
        ((batch.actions == THINK).float() * batch.stop_mask).sum()
        / stop_decisions
    )
    runs = think_run_lengths(batch.kind)
    grouped = batch.reward_scalar.reshape(-1, samples_per_prompt)
    exact = batch.reward_scalar == 1.0
    grouped_exact = exact.reshape(-1, samples_per_prompt).float()
    # Did thinking pay off THIS group?  Within-group Pearson correlation
    # between per-trajectory think counts and rewards (0.0 when either side
    # has no variance — undefined groups dilute the aggregate toward zero,
    # which is the honest prior for "no evidence either way").
    # Reward comparisons isolate learned continuation decisions; every latent
    # trajectory already contains the same mandatory initial thought.
    continued_thought_counts = (
        (batch.actions == THINK).float() * batch.stop_mask
    ).sum(1)
    initial_thought_counts = (
        (batch.actions == THINK).float()
        * (batch.action_mask - batch.stop_mask)
    ).sum(1)
    continued = continued_thought_counts > 0
    stopped = (
        ((batch.actions == EMIT).float() * batch.stop_mask).sum(1) > 0
    )

    def think_reward_correlation(rows: torch.Tensor) -> float:
        counts = continued_thought_counts[rows]
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
        "continue_thinking_fraction": continue_fraction,
        "stop_thinking_fraction": float(
            ((batch.actions == EMIT).float() * batch.stop_mask).sum()
            / stop_decisions
        ),
        "initial_thoughts_per_trajectory": float(
            initial_thought_counts.mean()
        ),
        "continued_thoughts_per_trajectory": float(
            continued_thought_counts.mean()
        ),
        "stopped_thinking_trajectory_fraction": float(stopped.float().mean()),
        "thoughts_per_trajectory": float(
            ((batch.actions == THINK).float() * batch.action_mask).sum(1).mean()
        ),
        "think_run_mean": float(runs.mean()) if runs.numel() else 0.0,
        "think_run_std": float(runs.std(unbiased=False)) if runs.numel() else 0.0,
        "think_runs_per_trajectory": runs.numel() / batch.reward_scalar.numel(),
        "emits_per_trajectory": float(batch.emit_mask.sum(1).mean()),
        "actions_per_trajectory": float(batch.action_mask.sum(1).mean()),
        "continued_thinking_trajectory_fraction": float(
            continued.float().mean()
        ),
        "reward_mean_continued_thinking": (
            float(batch.reward_scalar[continued].mean())
            if bool(continued.any())
            else 0.0
        ),
        "reward_mean_stopped_after_initial": (
            float(batch.reward_scalar[~continued].mean())
            if bool((~continued).any())
            else 0.0
        ),
        "think_reward_correlation": think_reward_correlation(
            torch.ones_like(continued)
        ),
        "ended_fraction": sum(ended) / max(len(ended), 1),
        **(
            {
                "old_value_mean": (
                    float(batch.old_values[generated].mean())
                    if generated.any()
                    else 0.0
                )
            }
            if refreshed_statistics
            else {}
        ),
    }


def aggregate_diagnostics(
    groups: list[LatentRolloutBatch],
    samples_per_prompt: int,
    stop_ids: tuple[int, ...] = (),
    *,
    refreshed_statistics: bool = True,
) -> dict[str, float | int]:
    """Mean of per-group rollout diagnostics; trajectory counts are summed."""
    per_group = [
        rollout_diagnostics(
            group,
            samples_per_prompt,
            stop_ids,
            refreshed_statistics=refreshed_statistics,
        )
        for group in groups
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


def lockstep_decode_metrics(
    groups: list[LatentRolloutBatch],
    chunk_groups: int,
) -> dict[str, float]:
    """How many decode steps a chunk paid for the actions it actually kept.

    A chunk's rows step together until its LAST row finishes, so the chunk
    costs one decode step per action of its longest trajectory while the mean
    trajectory stops far earlier. ``groups`` arrives in chunk order,
    ``chunk_groups`` per chunk, so the per-chunk maximum is the step count and
    the ratio against the mean action count is the share of decode work that
    produced nothing. Generation dominates pool wall time, so this ratio, not
    the batch width, is what bounds collection.

    Row length comes from ``action_mask``, NOT from ``stream_length``: groups
    reach this point through ``trim_stream``, which rounds the kept length UP
    to ``--replay-bucket`` and pads past the original stream to bound the set
    of compiled replay shapes. That bucketing would add up to a full bucket of
    pool-to-pool jitter to a metric whose whole job is to detect a change.
    The count still lands on the loop's SYNC_EVERY boundary, so it measures
    productive steps and omits the <=15 no-op steps after the last row ends.
    """
    if not groups:
        raise ValueError("at least one rollout group is required")
    # Both reductions cross the bus in ONE transfer. They used to be two
    # ``.tolist()``/``float()`` calls, and this runs inside the pool's own
    # timers at maximum queue depth, so each one drained the pipeline and was
    # billed to the collection it was measuring -- enough to swamp the effect
    # of anything being measured. Stacking them costs one sync instead.
    per_group = torch.stack(
        [
            torch.stack((group.action_mask.sum(1).max(), group.action_mask.sum()))
            for group in groups
        ]
    ).tolist()
    row_actions = [longest for longest, _ in per_group]
    chunk_steps = [
        max(row_actions[start : start + chunk_groups])
        for start in range(0, len(row_actions), chunk_groups)
    ]
    steps_mean = sum(chunk_steps) / len(chunk_steps)
    actions = sum(total for _, total in per_group)
    rows = sum(group.kind.size(0) for group in groups)
    return {
        "decode_steps_per_chunk_mean": steps_mean,
        "decode_steps_per_chunk_max": float(max(chunk_steps)),
        "decode_step_utilization": (
            (actions / rows) / steps_mean if steps_mean else 0.0
        ),
    }


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
    thought_reverse_kl_contribution = sum(
        metric["thought_reverse_kl_penalty"] for metric in metrics
    )
    gate_entropy_bonus = sum(
        metric["gate_entropy_bonus"] for metric in metrics
    )
    current_stop_probability = _weighted_metric_mean(
        metrics, "stop_probability", "gate_action_count"
    )
    behavior_stop_probability = _weighted_metric_mean(
        [
            {
                **metric,
                "behavior_stop_probability": metric.get(
                    "behavior_stop_probability",
                    metric["stop_probability"],
                ),
            }
            for metric in metrics
        ],
        "behavior_stop_probability",
        "gate_action_count",
    )
    exact_gate_kl = _weighted_metric_mean(
        [
            {
                **metric,
                "gate_behavior_kl_exact": metric.get(
                    "gate_behavior_kl_exact",
                    metric["gate_behavior_kl"],
                ),
            }
            for metric in metrics
        ],
        "gate_behavior_kl_exact",
        "gate_action_count",
    )
    action_count = sum(metric["action_count"] for metric in metrics)
    gate_action_count = sum(metric["gate_action_count"] for metric in metrics)
    return {
        "loss/policy": policy_contribution,
        "kl/thought_reverse_weighted": thought_reverse_kl_contribution,
        "bonus/gate_entropy_weighted": gate_entropy_bonus,
        "bonus/gate_entropy_gate_decision_fraction": (
            gate_action_count / max(action_count, 1.0)
        ),
        "bonus/gate_entropy_counterfactual_per_gate_amplification": (
            action_count / gate_action_count if gate_action_count else 0.0
        ),
        "loss/actor_total": (
            policy_contribution
            + thought_reverse_kl_contribution
            - gate_entropy_bonus
        ),
        "value/token_weighted_excess_ce": value["excess_ce"],
        "value/prediction_mean": value["prediction_mean"],
        "value/target_mean": value["target_mean"],
        "value/residual_mean": value["residual_mean"],
        "value/online_epoch_explained_variance": value["explained_variance"],
        "advantage/mean": advantage_mean,
        "advantage/std": advantage_std,
        "advantage/continued_think_mean": _weighted_metric_mean(
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
        # Historical name retained for dashboard continuity. This is the
        # CURRENT policy evaluated on replay states, not necessarily the
        # behavior policy once behavior_age > 0.
        "behavior/continue_thinking_probability": (
            1.0 - current_stop_probability
        ),
        "gate_same_state/behavior_continue_probability": (
            1.0 - behavior_stop_probability
        ),
        "gate_same_state/current_continue_probability": (
            1.0 - current_stop_probability
        ),
        "gate_same_state/continue_probability_delta": (
            behavior_stop_probability - current_stop_probability
        ),
        "gate_same_state/stop_probability_abs_delta": _weighted_metric_mean(
            [
                {
                    **metric,
                    "stop_probability_abs_delta": metric.get(
                        "stop_probability_abs_delta",
                        0.0,
                    ),
                }
                for metric in metrics
            ],
            "stop_probability_abs_delta",
            "gate_action_count",
        ),
        "gate_same_state/stop_probability_abs_delta_max": max(
            metric.get("stop_probability_abs_delta_max", 0.0)
            for metric in metrics
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
        "thought_input/raw_abs_max": max(
            metric["thought_raw_abs_max"] for metric in metrics
        ),
        "thought_input/raw_abs_gt_0_5_fraction": _weighted_metric_mean(
            metrics, "thought_raw_abs_gt_0_5_fraction", "thought_action_count"
        ),
        "thought_input/raw_abs_gt_0_8_fraction": _weighted_metric_mean(
            metrics, "thought_raw_abs_gt_0_8_fraction", "thought_action_count"
        ),
        "thought_input/raw_abs_gt_1_fraction": _weighted_metric_mean(
            metrics, "thought_raw_abs_gt_1_fraction", "thought_action_count"
        ),
        "thought_input/raw_abs_gt_2_fraction": _weighted_metric_mean(
            metrics, "thought_raw_abs_gt_2_fraction", "thought_action_count"
        ),
        "thought_input/transform_distortion_rms": math.sqrt(
            _weighted_metric_mean(
                [
                    {
                        **metric,
                        "thought_transform_distortion_square_mean": (
                            metric["thought_transform_distortion_rms"] ** 2
                        ),
                    }
                    for metric in metrics
                ],
                "thought_transform_distortion_square_mean",
                "thought_action_count",
            )
        ),
        "thought_input/squashed_raw_norm": _weighted_metric_mean(
            metrics, "thought_squashed_raw_norm", "thought_action_count"
        ),
        "thought_input/adapter_output_norm": _weighted_metric_mean(
            metrics, "thought_adapter_output_norm", "thought_action_count"
        ),
        "thought_input/post_thought_belief_norm": _weighted_metric_mean(
            metrics, "post_thought_belief_norm", "thought_action_count"
        ),
        "thought_input/mean_abs_max": max(
            metric["thought_mean_abs_max"] for metric in metrics
        ),
        "thought_input/mean_abs_gt_0_8_fraction": _weighted_metric_mean(
            metrics,
            "thought_mean_abs_gt_0_8_fraction",
            "thought_action_count",
        ),
        "kl/gate_behavior": _weighted_metric_mean(
            metrics, "gate_behavior_kl", "gate_action_count"
        ),
        "kl/gate_behavior_exact": exact_gate_kl,
        "kl/renderer_behavior": _weighted_metric_mean(
            metrics, "renderer_behavior_kl", "emit_action_count"
        ),
        "kl/thought_behavior_joint": _weighted_metric_mean(
            metrics, "thought_behavior_kl_joint", "thought_action_count"
        ),
        # Closed-form squared-Mahalanobis movement of the THINK mean from
        # stored behavior (the projected objective's trust statistic; the
        # sampled-ratio KL above stays as a cross-check).
        "trust/thought_d_mean": _weighted_metric_mean(
            metrics, "thought_trust_d_mean", "thought_action_count"
        ),
        "trust/thought_d_max": max(
            metric["thought_trust_d_max"] for metric in metrics
        ),
        "trust/projection_penalty": sum(
            metric["thought_projection_penalty"] for metric in metrics
        ),
        "kl/policy_behavior_per_action": sum(
            metric["policy_behavior_kl_per_action"] for metric in metrics
        ),
        "clip/policy": sum(
            metric["policy_clip_fraction"] for metric in metrics
        ),
        "clip/gate_all": _weighted_metric_mean(
            [
                {
                    **metric,
                    "gate_policy_clip_fraction": metric.get(
                        "gate_policy_clip_fraction",
                        0.0,
                    ),
                }
                for metric in metrics
            ],
            "gate_policy_clip_fraction",
            "gate_action_count",
        ),
        "clip/gate_continue": _weighted_metric_mean(
            metrics,
            "thought_gate_policy_clip_fraction",
            "think_action_count",
        ),
        "ratio/joint_abs_log_max": max(
            metric["joint_abs_log_ratio_max"] for metric in metrics
        ),
        "ratio/gate_abs_log_max": max(
            metric.get("gate_abs_log_ratio_max", 0.0)
            for metric in metrics
        ),
        "ratio/thought_dim_abs_log_max": max(
            metric["thought_dim_abs_log_ratio_max"] for metric in metrics
        ),
        "ratio/thought_joint_abs_log_max": max(
            metric["thought_joint_abs_log_ratio_max"] for metric in metrics
        ),
        "ratio/harmful_positive_log_max": max(
            metric["harmful_positive_log_ratio_max"] for metric in metrics
        ),
        # Share of THINK actions whose mean the trust region actually moved.
        # Computed since v23 but never surfaced, which left the projected
        # arm's headline diagnostic invisible; it reads identically 0 under
        # --thought-clip-mode none, so it also separates the two arms.
        "clip/thought_projection": sum(
            metric["thought_policy_clip_fraction"] for metric in metrics
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
        "reward/ended_fraction": float(metrics["ended_fraction"]),
        "reward/continued_thinking_mean": float(
            metrics["reward_mean_continued_thinking"]
        ),
        "reward/stopped_after_initial_mean": float(
            metrics["reward_mean_stopped_after_initial"]
        ),
        "behavior/continue_thinking_fraction": float(
            metrics["continue_thinking_fraction"]
        ),
        "behavior/stopped_thinking_trajectory_fraction": float(
            metrics["stopped_thinking_trajectory_fraction"]
        ),
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


def actor_minibatch_action_denominator(
    groups: list[LatentRolloutBatch],
    indices: list[int],
) -> torch.Tensor:
    """Global VAPO stream-action denominator for one actor step."""
    if not indices:
        raise ValueError("at least one prompt group is required")
    return torch.stack(
        [groups[index].action_mask.sum() for index in indices]
    ).sum()


def write_actor_tensorboard_metrics(
    tensorboard, dashboard: dict[str, float], behavior_age: int, step: int
) -> None:
    """Write one actor-step row, eliding exact fresh-behavior zeros."""
    if behavior_age == 0:
        refresh_drift = max(
            dashboard[tag]
            for tag in (
                "kl/gate_behavior",
                "kl/gate_behavior_exact",
                "kl/renderer_behavior",
                "kl/thought_behavior_joint",
                "kl/policy_behavior_per_action",
                "kl/thought_reverse_weighted",
                "clip/policy",
                "clip/gate_all",
                "clip/gate_continue",
                "clip/thought_projection",
                "ratio/joint_abs_log_max",
                "ratio/gate_abs_log_max",
                "ratio/thought_dim_abs_log_max",
                "ratio/thought_joint_abs_log_max",
                "ratio/harmful_positive_log_max",
                "trust/thought_d_mean",
                "trust/thought_d_max",
                "trust/projection_penalty",
            )
        )
        tensorboard.add_scalar(
            "debug/behavior_refresh_max_drift", refresh_drift, step
        )
    for tag, value in dashboard.items():
        if behavior_age == 0 and tag in {
            "kl/gate_behavior",
            "kl/gate_behavior_exact",
            "kl/renderer_behavior",
            "kl/thought_behavior_joint",
            "kl/policy_behavior_per_action",
            "kl/thought_reverse_weighted",
            "clip/policy",
            "clip/gate_all",
            "clip/gate_continue",
            "clip/thought_projection",
            "ratio/joint_abs_log_max",
            "ratio/gate_abs_log_max",
            "ratio/thought_dim_abs_log_max",
            "ratio/thought_joint_abs_log_max",
            "ratio/harmful_positive_log_max",
            "trust/thought_d_mean",
            "trust/thought_d_max",
            "trust/projection_penalty",
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


def renderer_parameters(backbone) -> list[torch.nn.Parameter]:
    """The trainable vocabulary-readout parameters, across backbone families.

    The fresh lineage renders through its pretrained ``policy_probe``; the
    nano backbones expose their native readout (untied projection, or the
    tied-dot scale/bias) via their own ``renderer_parameters`` method.
    """
    if hasattr(backbone, "policy_probe"):
        return list(backbone.policy_probe.parameters())
    return list(backbone.renderer_parameters())


def non_trunk_parameter_ids(backbone) -> set[int]:
    """Backbone parameters excluded from the trunk optimizer group and norm.

    The renderer readout is always its own semantic group. The fresh critic
    probe additionally exists but stays frozen and unused; nano backbones
    carry no probes at all.
    """
    excluded = {id(parameter) for parameter in renderer_parameters(backbone)}
    critic_probe = getattr(backbone, "critic_probe", None)
    if critic_probe is not None:
        excluded |= {id(parameter) for parameter in critic_probe.parameters()}
    return excluded


def muon_matrix_parameters(
    blocks: torch.nn.Module, excluded_parameter_ids: set[int]
) -> list[torch.nn.Parameter]:
    """The pretraining Muon partition: block matrices with ndim >= 2.

    Mirrors ``nanogpt_mini_gpt2vocab_train.py`` exactly — embeddings, the
    readout, biases, and norm gains stay under AdamW. Fresh-lineage probes
    live under ``blocks[-1]``, so the exclusion set must be honored here too.
    """
    return [
        parameter
        for parameter in blocks.parameters()
        if parameter.ndim >= 2 and id(parameter) not in excluded_parameter_ids
    ]


def step_optimizers(
    optimizers: dict[str, torch.optim.Optimizer], role: str
) -> None:
    """Step every optimizer belonging to ``role`` ("actor" or "critic").

    The Muon split registers trunk matrices under ``<role>_muon``; iterating
    by role prefix keeps every step/zero seam covering both optimizers.
    """
    for name in sorted(optimizers):
        if name == role or name.startswith(role + "_"):
            optimizers[name].step()


def zero_optimizers(
    optimizers: dict[str, torch.optim.Optimizer], role: str
) -> None:
    """zero_grad(set_to_none=True) for every optimizer of ``role``."""
    for name in sorted(optimizers):
        if name == role or name.startswith(role + "_"):
            optimizers[name].zero_grad(set_to_none=True)


def optimizer_learning_rates(args) -> dict[str, float]:
    """The per-optimizer learning rates the CLI owns across resumes."""
    return {
        "actor": args.learning_rate,
        "critic": args.critic_learning_rate,
        "actor_muon": args.muon_learning_rate,
        "critic_muon": args.critic_muon_learning_rate,
    }


def reassert_learning_rates(
    optimizers: dict[str, torch.optim.Optimizer], args
) -> None:
    """Reapply the CLI learning rates after loading optimizer state."""
    rates = optimizer_learning_rates(args)
    for name, optimizer in optimizers.items():
        for group in optimizer.param_groups:
            group["lr"] = rates[name]


def build_optimizers(
    wrapper: LatentThoughtModel,
    critic: SeparateCritic,
    learning_rate: float,
    critic_learning_rate: float | None = None,
    trunk_optimizer: str = "adamw",
    muon_learning_rate: float | None = None,
    critic_muon_learning_rate: float | None = None,
    fused: bool = True,
) -> dict[str, torch.optim.Optimizer]:
    """The actor/critic optimizer layout.

    One actor AdamW retains six semantic groups for exact telemetry and
    checkpoint validation, and every trainable policy component uses one
    general actor learning rate: trunk, Bernoulli gate, recurrent adapter,
    renderer, state-dependent log-sigma, and fresh mean. The critic may use
    its own constant rate so actor-rate ablations do not alter value warmup.
    The fresh probes live under
    ``blocks[-1]`` (so pretraining's optimizer saw them), which makes
    name-prefix filtering wrong — exclude them from the trunk by identity.
    Nano backbones keep the identical six-group layout with their native
    readout as the renderer group.

    ``trunk_optimizer="muon"`` restores the pretraining update geometry:
    block matrices (ndim >= 2) move from the AdamW trunk group into separate
    ``actor_muon``/``critic_muon`` Muon optimizers. Polar Express supplies
    the orthogonalized momentum update, while embeddings, readout, gains,
    and every RL-only head stay under AdamW. Weight decay stays 0 everywhere:
    pretraining's Muon decay (0.05) regularizes a from-scratch run, but over
    a long RL schedule it would only shrink the pretrained weights.
    """
    if trunk_optimizer not in ("adamw", "muon"):
        raise ValueError(f"unknown trunk optimizer {trunk_optimizer!r}")
    if critic_learning_rate is None:
        critic_learning_rate = learning_rate
    backbone = wrapper.backbone
    excluded_parameter_ids = non_trunk_parameter_ids(backbone)
    actor_muon_parameters: list[torch.nn.Parameter] = []
    critic_muon_parameters: list[torch.nn.Parameter] = []
    if trunk_optimizer == "muon":
        actor_muon_parameters = muon_matrix_parameters(
            backbone.blocks, excluded_parameter_ids
        )
        critic_muon_parameters = muon_matrix_parameters(critic.trunk.blocks, set())
    actor_muon_ids = {id(parameter) for parameter in actor_muon_parameters}
    critic_muon_ids = {id(parameter) for parameter in critic_muon_parameters}
    trunk_parameters = [
        parameter
        for parameter in backbone.parameters()
        if id(parameter) not in excluded_parameter_ids
        and id(parameter) not in actor_muon_ids
    ]
    critic_adamw_parameters = [
        parameter
        for parameter in critic.parameters()
        if id(parameter) not in critic_muon_ids
    ]
    optimizers = {
        "actor": torch.optim.AdamW(
            [
                {"params": trunk_parameters, "lr": learning_rate},
                {"params": list(wrapper.gate.parameters()), "lr": learning_rate},
                {"params": list(wrapper.adapter.parameters()), "lr": learning_rate},
                {"params": renderer_parameters(backbone), "lr": learning_rate},
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
            critic_adamw_parameters,
            lr=critic_learning_rate,
            weight_decay=0.0,
            fused=fused,
        ),
    }
    if trunk_optimizer == "muon":
        if muon_learning_rate is None or critic_muon_learning_rate is None:
            raise ValueError("the muon trunk optimizer requires its learning rates")
        optimizers["actor_muon"] = Muon(
            actor_muon_parameters, lr=muon_learning_rate, weight_decay=0.0
        )
        optimizers["critic_muon"] = Muon(
            critic_muon_parameters, lr=critic_muon_learning_rate, weight_decay=0.0
        )
    # Exact-partition guard (pretraining's optimizer assertion): every critic
    # parameter in exactly one optimizer, and the Muon split disjoint from
    # the actor's semantic groups.
    actor_registered = [
        parameter
        for name, optimizer in optimizers.items()
        if name.startswith("actor")
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(actor_registered) != len({id(p) for p in actor_registered}):
        raise AssertionError("actor parameter registered in two optimizers")
    actor_registered_ids = {id(parameter) for parameter in actor_registered}
    actor_expected_ids = {
        id(parameter)
        for parameter in wrapper.parameters()
        if parameter.requires_grad
    }
    if actor_registered_ids != actor_expected_ids:
        raise AssertionError(
            "every trainable actor parameter must belong to exactly one "
            "optimizer"
        )
    trunk_coverage = {id(p) for p in trunk_parameters} | actor_muon_ids
    trunk_expected = {
        id(parameter)
        for parameter in backbone.parameters()
        if id(parameter) not in excluded_parameter_ids
    }
    if trunk_coverage != trunk_expected:
        raise AssertionError(
            "the Muon split must exactly cover the AdamW trunk group it "
            "replaced"
        )
    critic_registered = [
        parameter
        for name, optimizer in optimizers.items()
        if name.startswith("critic")
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if {id(p) for p in critic_registered} != {
        id(p) for p in critic.parameters()
    } or len(critic_registered) != len({id(p) for p in critic_registered}):
        raise AssertionError("critic parameters must partition across optimizers")
    return optimizers


def update_minibatch(
    wrapper: LatentThoughtModel,
    critic: SeparateCritic,
    batch: LatentRolloutBatch,
    optimizers: dict[str, torch.optim.Optimizer],
    value_only: bool = False,
    thought_pg_coef: float = 1.0,
    # Kept in lockstep with the CLI defaults so callers that omit them
    # exercise the SHIPPED arm; the pair must satisfy the biconditional
    # enforced below (nonzero coefficient <=> clip mode 'none').
    thought_reverse_kl_coef: float = 0.5,
    thought_clip_mode: str = "none",
    thought_trust_epsilon: float = 0.03,
    thought_projection_penalty_coef: float = 1.0,
    gate_pg_coef: float = 1.0,
    gate_entropy_coef: float = 0.0,
    actor_step: bool = True,
    critic_step: bool = True,
    policy_action_denominator: torch.Tensor | None = None,
    value_action_denominator: torch.Tensor | None = None,
    gae_lambda_alpha: float = 0.05,
    replay_max_trajectories: int = 32,
    replay_attention_budget: int = 4 * 1024 * 1024,
    replay_bucket: int = 1,
    replay_slot_budget: int | None = None,
    replay_plan: ReplayPlan | None = None,
) -> dict[str, float]:
    """One minibatch update.

    Actor and critic gradients can be accumulated across prompt groups with
    ``actor_step=False`` and ``critic_step=False``. The trainer uses the same
    effective trajectory minibatch and one optimizer step for both; standalone
    callers keep the self-contained single-step defaults.

    The group is replayed in stable length-sorted shards because causal
    attention is quadratic in stream length. The planner bounds B*L^2, B*L
    (the vocabulary head's linear memory term), and trims every shard
    independently. The actor objectives keep denominators from the COMPLETE
    optimizer minibatch, which can span prompt groups, so sharding changes
    only floating-point reduction order.
    """
    backbone = wrapper.backbone
    if critic_step:
        zero_optimizers(optimizers, "critic")
    if actor_step and "actor" in optimizers:
        zero_optimizers(optimizers, "actor")
    if replay_max_trajectories < 1:
        raise ValueError("replay max trajectories must be positive")
    if replay_attention_budget < 1:
        raise ValueError("replay attention budget must be positive")
    if (
        not math.isfinite(thought_reverse_kl_coef)
        or thought_reverse_kl_coef < 0
    ):
        raise ValueError(
            "thought reverse KL coefficient must be finite and nonnegative"
        )
    if thought_clip_mode not in ("projected", "joint", "per_dim", "none"):
        raise ValueError(
            f"unknown thought clip mode {thought_clip_mode!r}; "
            "expected 'projected', 'joint', 'per_dim', or 'none'"
        )
    # Exactly one THINK trust mechanism per run. The surrogate-side modes
    # (projection / ratio clip) and the reverse-KL penalty are alternative
    # solutions to the same problem; mixing them makes neither term's
    # contribution attributable, so the two conditions below are the two
    # halves of one biconditional: coefficient nonzero <=> mode 'none'.
    if thought_clip_mode == "none" and thought_reverse_kl_coef == 0.0:
        raise ValueError(
            "thought clip mode 'none' removes the surrogate trust region, so "
            "--thought-reverse-kl-coef must be nonzero; an unconstrained "
            "512-D Gaussian surrogate is what the v22 joint-clip run showed "
            "cannot bound aggregate drift"
        )
    if thought_clip_mode != "none" and thought_reverse_kl_coef != 0.0:
        raise ValueError(
            f"thought clip mode {thought_clip_mode!r} and a nonzero "
            "--thought-reverse-kl-coef are alternative trust mechanisms, not "
            "layers of one, and must never be combined: the clip modes bound "
            "movement inside the surrogate while the penalty prices realized "
            "aggregate divergence of the acting policy. Use "
            f"--thought-clip-mode none with the penalty, or {thought_clip_mode!r} "
            "with --thought-reverse-kl-coef 0"
        )
    if (
        not math.isfinite(thought_projection_penalty_coef)
        or thought_projection_penalty_coef < 0.0
    ):
        raise ValueError(
            "thought projection penalty coefficient must be finite and "
            "nonnegative"
        )
    if not value_only and not batch.statistics_refreshed:
        # The explicit flag also covers pinned-EMIT batches, whose zero-width
        # thought tensors cannot express "not yet refreshed" as a width
        # mismatch.
        raise RuntimeError(
            "actor update requires refresh_old_statistics after rollout"
        )

    thought_actions = (batch.actions == THINK).float() * batch.action_mask
    optional_think_actions = (
        (batch.actions == THINK).float() * batch.stop_mask
    )
    forced_initial_think_actions = thought_actions - optional_think_actions
    emit_actions = (batch.actions == EMIT).float() * batch.action_mask
    denominators = {
        "action": batch.action_mask.sum().clamp_min(1),
        "gate": batch.stop_mask.sum().clamp_min(1),
        "emit": batch.emit_mask.sum().clamp_min(1),
        "thought": thought_actions.sum().clamp_min(1),
    }
    if policy_action_denominator is None:
        policy_action_denominator = batch.action_mask.sum()
    if value_action_denominator is None:
        value_action_denominator = batch.action_mask.sum()
    zero = batch.action_mask.new_zeros(())
    totals = {
        key: zero.clone()
        for key in (
            "value_loss", "value_sum", "value_target_sum", "target_entropy_sum",
            "policy_loss", "thought_reverse_kl_penalty",
            "policy_clip", "emit_policy_clip",
            "thought_policy_clip", "thought_gate_policy_clip", "gate_kl_sum",
            "gate_exact_kl_sum",
            "gate_policy_clip_count",
            "gate_entropy_sum", "gate_entropy_bonus",
            "stop_probability_sum",
            "behavior_stop_probability_sum",
            "stop_probability_delta_sum",
            "stop_probability_abs_delta_sum",
            "renderer_kl_sum", "thought_kl_sum",
            "advantage_sum", "advantage_square_sum",
            "optional_think_advantage_sum", "forced_initial_think_advantage_sum",
            "thought_advantage_sum", "emit_advantage_sum", "target_square_sum",
            "residual_sum", "residual_square_sum",
            "joint_abs_log_ratio_max", "thought_dim_abs_log_ratio_max",
            "thought_joint_abs_log_ratio_max",
            "gate_abs_log_ratio_max", "stop_probability_abs_delta_max",
            "harmful_positive_log_ratio_max",
            "thought_log_sigma_sum", "thought_log_sigma_square_sum",
            "thought_sigma_sum", "thought_expected_noise_norm_sum",
            "thought_realized_noise_norm_sum", "thought_normalized_noise_square_sum",
            "thought_mean_norm_sum",
            "thought_raw_abs_max", "thought_raw_abs_gt_0_5_count",
            "thought_raw_abs_gt_0_8_count",
            "thought_raw_abs_gt_1_count", "thought_raw_abs_gt_2_count",
            "thought_transform_distortion_square_sum",
            "thought_squashed_raw_norm_sum",
            "thought_adapter_output_norm_sum",
            "post_thought_belief_norm_sum",
            "thought_mean_abs_max", "thought_mean_abs_gt_0_8_count",
            "thought_trust_sum", "thought_trust_max",
            "thought_projection_penalty",
        )
    }
    totals["thought_log_sigma_min"] = zero.new_full((), float("inf"))
    totals["thought_log_sigma_max"] = zero.new_full((), float("-inf"))

    action_counts = batch.action_mask.sum(1)
    lambdas = (
        torch.ones_like(action_counts)
        if value_only
        else length_adaptive_lambda(action_counts, gae_lambda_alpha)
    )
    advantages, value_targets = generalized_advantage_and_return_targets(
        batch.rewards,
        batch.old_values,
        batch.action_mask,
        lambdas,
    )
    advantages = advantages.detach()
    value_targets = value_targets.detach()
    # Standardize THINK advantages (projected mode): the projected surrogate
    # has no ratio band making it scale-invariant, so the trust and penalty
    # coefficients are calibrated against unit-scale advantages. Centering
    # matters as much as scaling — without it a low-variance pool (all
    # trajectories sharing an outcome) turns mean/std into an unbounded
    # gradient amplifier; the subtracted per-minibatch constant is a
    # standard score-function baseline (the projected Gaussian normalizes
    # to one, so E[b * grad ratio] = 0). Computed over the complete
    # minibatch's actions; EMIT and the gate keep raw advantages (their
    # clip decisions never depended on this scale).
    advantage_action_count = batch.action_mask.sum().clamp_min(1)
    advantage_scale_mean = (
        advantages * batch.action_mask
    ).sum() / advantage_action_count
    thought_advantage_normalizer = (
        (advantages.square() * batch.action_mask).sum()
        / advantage_action_count
        - advantage_scale_mean.square()
    ).clamp_min(0).sqrt().clamp_min(1e-6)

    if replay_plan is None:
        replay_plan = build_replay_plan(
            batch,
            replay_max_trajectories,
            replay_attention_budget,
            replay_bucket,
            replay_slot_budget,
            include_action_indices=not value_only,
        ).to(batch.kind.device)
    replay_plan.validate_settings(
        replay_max_trajectories,
        replay_attention_budget,
        replay_bucket,
        replay_slot_budget,
        require_action_indices=not value_only,
    )

    # Non-finite losses must abort the run, but ``torch.isfinite`` on a live
    # loss is a host sync at the worst possible moment: maximum queue depth,
    # immediately before ``backward``, so the GPU drains and then idles
    # through the entire backward launch -- twice per shard, roughly a dozen
    # shards per minibatch. The tests are accumulated on device and resolved
    # in one sync after the shard loop instead. Every ``step_optimizers`` call
    # in this function comes later, so a poisoned shard still aborts before it
    # can reach the weights; only the wasted backward passes are new.
    finite_guards: list[
        tuple[int, str, torch.Tensor, dict[str, torch.Tensor]]
    ] = []

    for shard_index, (microbatch, shard) in enumerate(
        iter_planned_replay_microbatches(batch, replay_plan)
    ):
        rows = shard.rows
        stream_length = shard.stream_length
        micro_advantages = advantages[rows, :stream_length]
        micro_value_targets = value_targets[rows, :stream_length]
        local_action_count = microbatch.action_mask.sum()
        value_logits = critic.value_logits(microbatch)
        value_ce = critic.support.cross_entropy(value_logits, micro_value_targets)
        local_value_numerator = (value_ce * microbatch.action_mask).sum()
        weighted_value_loss = (
            local_value_numerator / value_action_denominator.clamp_min(1)
        )
        finite_guards.append(
            (
                shard_index,
                "value",
                weighted_value_loss.detach(),
                {},
            )
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
                micro_value_targets - values if value_only else micro_advantages
            )
            target_probs = critic.support.project(micro_value_targets)
            target_entropy = -(
                target_probs * target_probs.clamp_min(1e-20).log()
            ).sum(-1)
            micro_thought_actions = (
                (microbatch.actions == THINK).float()
                * microbatch.action_mask
            )
            micro_optional_thinks = (
                (microbatch.actions == THINK).float()
                * microbatch.stop_mask
            )
            micro_forced_initial_thinks = (
                micro_thought_actions - micro_optional_thinks
            )
            micro_emits = (
                (microbatch.actions == EMIT).float()
                * microbatch.action_mask
            )
            totals["value_loss"] += local_value_numerator.detach()
            totals["value_sum"] += (values * microbatch.action_mask).sum()
            totals["value_target_sum"] += (
                micro_value_targets * microbatch.action_mask
            ).sum()
            totals["target_square_sum"] += (
                micro_value_targets.square() * microbatch.action_mask
            ).sum()
            residuals = micro_value_targets - values
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
        beliefs, stream_inputs = replay_head_inputs(
            wrapper, microbatch
        )
        new_stop_logits = wrapper.gate.stop_logit(beliefs)
        new_stop_logprobs = -F.binary_cross_entropy_with_logits(
            new_stop_logits,
            microbatch.actions.float(),
            reduction="none",
        )

        # The CPU replay plan has already found every compact action slot.
        # Every compaction below therefore has a host-known output shape and
        # uses index_select/index_add without a data-dependent CUDA sync.
        emit_index = shard.emit_index
        # Shared with refresh_old_statistics: one compiled artifact for both
        # keeps the two forwards bit-identical (the age-0 zero-clip canary).
        compact_token_logprobs = compact_emit_token_logprobs(
            wrapper,
            compact_slots(stream_inputs, emit_index),
            compact_slots(beliefs, emit_index),
            compact_next_slots(microbatch.token_ids, emit_index),
        )
        new_token_logprobs = torch.zeros_like(microbatch.old_token_logprobs)
        scatter_slots(new_token_logprobs, emit_index, compact_token_logprobs)

        new_thought_joint = torch.zeros_like(new_token_logprobs)
        old_thought_joint = torch.zeros_like(new_token_logprobs)
        # Bound only on the has-THINK path, and read again further down
        # under ``policy_thought_logprobs is not None``, which is set in the
        # same branch. Reset here with the rest so a previous shard's index
        # can never survive into this one.
        think_index = None
        policy_thought_logprobs = None
        compact_old_thought_logprobs = None
        thought_reverse_kl_factors = None
        weighted_thought_reverse_kl = zero
        projected_policy_thought_logprobs = None
        thought_trust_scale = None
        weighted_projection_penalty = zero
        if shard.think_index.numel():
            think_index = shard.think_index
            thought_means, thought_targets = compact_thought_actions(
                wrapper, microbatch, beliefs, think_index
            )
            thought_log_sigma = wrapper.transition.predict_log_sigma(
                compact_slots(beliefs, think_index)
            )
            if thought_pg_coef != 0.0 or thought_reverse_kl_coef != 0.0:
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
            compact_old_thought_logprobs = compact_slots(
                microbatch.old_thought_logprobs, think_index
            ).float()
            if thought_reverse_kl_coef != 0.0:
                thought_reverse_kl_factors = sampled_reverse_kl(
                    new_thought_logprobs,
                    compact_old_thought_logprobs,
                )
                weighted_thought_reverse_kl = (
                    thought_reverse_kl_coef
                    * thought_reverse_kl_factors.sum()
                    / policy_action_denominator.clamp_min(1)
                )
            else:
                # Preserve behavior-KL telemetry for the coefficient-zero
                # ablation without retaining k3 activations or traversing its
                # backward graph.
                with torch.no_grad():
                    thought_reverse_kl_factors = sampled_reverse_kl(
                        new_thought_logprobs.detach(),
                        compact_old_thought_logprobs,
                    )
            if thought_pg_coef == 0.0:
                policy_thought_logprobs = new_thought_logprobs.detach()
            else:
                policy_thought_logprobs = (
                    new_thought_logprobs.detach()
                    + thought_pg_coef
                    * (new_thought_logprobs - new_thought_logprobs.detach())
                )
            # index_copy_ rather than masked_scatter: both place the compact
            # values at the mask's true slots in order, but masked_scatter's
            # gradient is masked_select, whose output shape is data
            # dependent, so the backward pass stalls the host.
            scatter_slots(
                new_thought_joint, think_index, policy_thought_logprobs.sum(-1)
            )
            scatter_slots(
                old_thought_joint,
                think_index,
                compact_old_thought_logprobs.sum(-1),
            )
            if thought_clip_mode == "projected":
                behavior_thought_means = compact_slots(
                    microbatch.old_thought_means, think_index
                ).float()
                behavior_thought_log_sigmas = compact_slots(
                    microbatch.old_thought_log_sigmas, think_index
                ).float()
                if thought_pg_coef == 0.0:
                    with torch.no_grad():
                        (
                            projected_thought_means,
                            thought_trust_scale,
                            thought_trust_sq,
                        ) = project_thought_means(
                            thought_means.detach().float(),
                            behavior_thought_means,
                            behavior_thought_log_sigmas,
                            thought_trust_epsilon,
                        )
                        projected_policy_thought_logprobs = (
                            wrapper.transition.per_dim_log_prob(
                                thought_targets,
                                projected_thought_means,
                                thought_log_sigma.detach(),
                            ).float()
                        )
                else:
                    (
                        projected_thought_means,
                        thought_trust_scale,
                        thought_trust_sq,
                    ) = project_thought_means(
                        thought_means.float(),
                        behavior_thought_means,
                        behavior_thought_log_sigmas,
                        thought_trust_epsilon,
                    )
                    raw_projected_logprobs = (
                        wrapper.transition.per_dim_log_prob(
                            thought_targets,
                            projected_thought_means,
                            thought_log_sigma,
                        ).float()
                    )
                    projected_policy_thought_logprobs = (
                        raw_projected_logprobs.detach()
                        + thought_pg_coef
                        * (
                            raw_projected_logprobs
                            - raw_projected_logprobs.detach()
                        )
                    )
                    # TRPL projection penalty: pull the raw mean toward its
                    # detached projection so the rollout policy tracks the
                    # trained one. The gap is zero wherever no projection
                    # occurred, so this term never acts inside the region.
                    projection_gap = (
                        thought_means.float()
                        - projected_thought_means.detach()
                    ) * (-behavior_thought_log_sigmas).exp()
                    weighted_projection_penalty = (
                        thought_projection_penalty_coef
                        * thought_pg_coef
                        * projection_gap.square().sum()
                        / policy_action_denominator.clamp_min(1)
                    )
                with torch.no_grad():
                    totals["thought_trust_sum"] += thought_trust_sq.sum()
                    totals["thought_trust_max"] = torch.maximum(
                        totals["thought_trust_max"], thought_trust_sq.max()
                    )
                    totals["thought_projection_penalty"] += (
                        weighted_projection_penalty.detach()
                    )
            elif thought_clip_mode == "none":
                # The reverse-KL penalty is the whole trust mechanism here, so
                # the surrogate scores the RAW Gaussian: no projection, an
                # all-ones trust scale (nothing is ever projected, so the
                # reported projection fraction is identically 0), and no
                # tracking penalty, which exists only to close a raw/projected
                # gap that cannot open.
                projected_policy_thought_logprobs = policy_thought_logprobs
                thought_trust_scale = torch.ones_like(
                    policy_thought_logprobs[..., 0]
                )
                with torch.no_grad():
                    # Closed-form drift telemetry stays live and comparable
                    # with the projected arm; only its use as a constraint is
                    # dropped. thought_trust_epsilon is unused in this mode.
                    _, _, thought_trust_sq = project_thought_means(
                        thought_means.detach().float(),
                        compact_slots(
                            microbatch.old_thought_means, think_index
                        ).float(),
                        compact_slots(
                            microbatch.old_thought_log_sigmas, think_index
                        ).float(),
                        thought_trust_epsilon,
                    )
                    totals["thought_trust_sum"] += thought_trust_sq.sum()
                    totals["thought_trust_max"] = torch.maximum(
                        totals["thought_trust_max"], thought_trust_sq.max()
                    )

        if gate_pg_coef == 0.0:
            policy_stop_logprobs = new_stop_logprobs.detach()
        else:
            policy_stop_logprobs = (
                new_stop_logprobs.detach()
                + gate_pg_coef
                * (new_stop_logprobs - new_stop_logprobs.detach())
            )
        new_joint_logprobs, old_joint_logprobs = joint_action_logprobs(
            policy_stop_logprobs,
            microbatch.old_stop_logprobs,
            new_token_logprobs,
            microbatch.old_token_logprobs,
            new_thought_joint,
            old_thought_joint,
            microbatch.stop_mask,
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
                (micro_advantages < 0) & microbatch.action_mask.bool(),
                joint_log_ratio.clamp_min(0),
                torch.zeros_like(joint_log_ratio),
            )
            totals["harmful_positive_log_ratio_max"] = torch.maximum(
                totals["harmful_positive_log_ratio_max"],
                harmful_positive_log_ratio.max(),
            )
        weighted_emit_policy_loss, weighted_emit_policy_clip, _ = (
            clipped_policy_loss(
                policy_stop_logprobs * microbatch.stop_mask
                + new_token_logprobs,
                microbatch.old_stop_logprobs * microbatch.stop_mask
                + microbatch.old_token_logprobs,
                micro_advantages,
                microbatch.emit_mask,
                denominator=policy_action_denominator,
                estimate_kl=False,
            )
        )
        weighted_policy_loss = weighted_emit_policy_loss
        weighted_policy_clip = weighted_emit_policy_clip
        if policy_thought_logprobs is not None:
            compact_stop_logprobs = compact_slots(
                policy_stop_logprobs, think_index
            )
            compact_old_stop_logprobs = compact_slots(
                microbatch.old_stop_logprobs, think_index
            )
            compact_thought_advantages = compact_slots(
                micro_advantages, think_index
            )
            compact_thought_stop_mask = compact_slots(
                microbatch.stop_mask, think_index
            )
            # 'none' shares this surrogate: with an identity projection and an
            # all-ones trust scale it reduces to the same unclipped IS form
            # under the +/-2 numerical guard, which is exactly the arm we want
            # against 'projected' -- one term differs, not the objective shape.
            if thought_clip_mode in ("projected", "none"):
                (
                    weighted_thought_policy_loss,
                    weighted_thought_policy_clip,
                    weighted_thought_gate_clip,
                ) = projected_thought_policy_loss(
                    compact_stop_logprobs,
                    compact_old_stop_logprobs,
                    projected_policy_thought_logprobs,
                    compact_old_thought_logprobs,
                    thought_trust_scale,
                    (compact_thought_advantages - advantage_scale_mean)
                    / thought_advantage_normalizer,
                    compact_thought_stop_mask,
                    policy_action_denominator,
                    gate_advantages=compact_thought_advantages,
                )
            else:
                thought_loss_function = (
                    joint_thought_policy_loss
                    if thought_clip_mode == "joint"
                    else per_dimension_thought_policy_loss
                )
                (
                    weighted_thought_policy_loss,
                    weighted_thought_policy_clip,
                    weighted_thought_gate_clip,
                ) = thought_loss_function(
                    compact_stop_logprobs,
                    compact_old_stop_logprobs,
                    policy_thought_logprobs,
                    compact_old_thought_logprobs,
                    compact_thought_advantages,
                    compact_thought_stop_mask,
                    policy_action_denominator,
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

        # The gate is one factor of an optional stream action, so its entropy
        # contribution uses the same token-level VAPO denominator as the
        # reward-driven surrogate. A per-gate mean would amplify the bonus by
        # action_count/gate_count precisely when gate decisions become rare.
        # The auxiliary objective deliberately updates only the Bernoulli
        # head. Sending its gradient into the shared belief trunk could erase
        # useful state features instead of making the gate use those features
        # less deterministically.
        # gate_pg_coef=0 remains a true gate freeze, including this term.
        weighted_gate_entropy_bonus = (
            gate_entropy_coef
            * gate_pg_coef
            * (
                wrapper.gate.entropy(beliefs.detach())
                * microbatch.stop_mask
            ).sum()
            / policy_action_denominator.clamp_min(1)
        )

        actor_total = (
            weighted_policy_loss
            + weighted_thought_reverse_kl
            + weighted_projection_penalty
            - weighted_gate_entropy_bonus
        )
        finite_guards.append(
            (
                shard_index,
                "actor",
                actor_total.detach(),
                {
                    "policy": weighted_policy_loss.detach(),
                    "thought_reverse_kl": weighted_thought_reverse_kl.detach(),
                    "projection_penalty": (
                        weighted_projection_penalty.detach()
                    ),
                    "gate_entropy_bonus": (
                        weighted_gate_entropy_bonus.detach()
                    ),
                },
            )
        )
        actor_total.backward()
        with torch.no_grad():
            gate_log_ratio = (
                new_stop_logprobs - microbatch.old_stop_logprobs
            )
            (
                behavior_stop_probability,
                current_stop_probability,
                exact_gate_kl,
            ) = exact_bernoulli_behavior_kl(
                new_stop_logits,
                microbatch.actions,
                microbatch.old_stop_logprobs,
            )
            stop_probability_delta = (
                current_stop_probability - behavior_stop_probability
            )
            renderer_log_ratio = (
                new_token_logprobs - microbatch.old_token_logprobs
            )
            totals["policy_loss"] += weighted_policy_loss.detach()
            totals["thought_reverse_kl_penalty"] += (
                weighted_thought_reverse_kl.detach()
            )
            totals["policy_clip"] += weighted_policy_clip
            totals["gate_entropy_bonus"] += weighted_gate_entropy_bonus.detach()
            totals["gate_kl_sum"] += (
                (torch.expm1(gate_log_ratio) - gate_log_ratio)
                * microbatch.stop_mask
            ).sum()
            totals["gate_exact_kl_sum"] += (
                exact_gate_kl * microbatch.stop_mask
            ).sum()
            totals["gate_abs_log_ratio_max"] = torch.maximum(
                totals["gate_abs_log_ratio_max"],
                (gate_log_ratio * microbatch.stop_mask).abs().max(),
            )
            gate_clip_low = gate_log_ratio.new_full((), math.log(0.8))
            gate_clip_high = gate_log_ratio.new_full((), math.log(1.28))
            totals["gate_policy_clip_count"] += (
                (
                    (gate_log_ratio < gate_clip_low)
                    | (gate_log_ratio > gate_clip_high)
                )
                * microbatch.stop_mask
            ).sum()
            totals["gate_entropy_sum"] += (
                wrapper.gate.entropy(beliefs) * microbatch.stop_mask
            ).sum()
            totals["stop_probability_sum"] += (
                current_stop_probability
                * microbatch.stop_mask
            ).sum()
            totals["behavior_stop_probability_sum"] += (
                behavior_stop_probability * microbatch.stop_mask
            ).sum()
            totals["stop_probability_delta_sum"] += (
                stop_probability_delta * microbatch.stop_mask
            ).sum()
            totals["stop_probability_abs_delta_sum"] += (
                stop_probability_delta.abs() * microbatch.stop_mask
            ).sum()
            totals["stop_probability_abs_delta_max"] = torch.maximum(
                totals["stop_probability_abs_delta_max"],
                (
                    stop_probability_delta
                    * microbatch.stop_mask
                ).abs().max(),
            )
            totals["renderer_kl_sum"] += (
                (torch.expm1(renderer_log_ratio) - renderer_log_ratio)
                * microbatch.emit_mask
            ).sum()
            if policy_thought_logprobs is not None:
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
                raw_thought = thought_targets.detach().float()
                transformed_thought = transform_thought_action(
                    raw_thought, wrapper.thought_action_transform
                )
                raw_abs = raw_thought.abs()
                totals["thought_raw_abs_max"] = torch.maximum(
                    totals["thought_raw_abs_max"], raw_abs.max()
                )
                totals["thought_raw_abs_gt_0_5_count"] += (
                    raw_abs > 0.5
                ).sum()
                totals["thought_raw_abs_gt_0_8_count"] += (
                    raw_abs > 0.8
                ).sum()
                totals["thought_raw_abs_gt_1_count"] += (raw_abs > 1.0).sum()
                totals["thought_raw_abs_gt_2_count"] += (raw_abs > 2.0).sum()
                totals["thought_transform_distortion_square_sum"] += (
                    transformed_thought - raw_thought
                ).square().sum()
                totals["thought_squashed_raw_norm_sum"] += (
                    transformed_thought.norm(dim=-1).sum()
                )
                thought_input_index = think_index + 1
                totals["thought_adapter_output_norm_sum"] += compact_slots(
                    stream_inputs.detach(), thought_input_index
                ).float().norm(dim=-1).sum()
                totals["post_thought_belief_norm_sum"] += compact_slots(
                    beliefs.detach(), thought_input_index
                ).float().norm(dim=-1).sum()
                mean_abs = thought_means.detach().float().abs()
                totals["thought_mean_abs_max"] = torch.maximum(
                    totals["thought_mean_abs_max"], mean_abs.max()
                )
                totals["thought_mean_abs_gt_0_8_count"] += (
                    mean_abs > 0.8
                ).sum()
                thought_log_ratio = (
                    new_thought_logprobs
                    - compact_slots(
                        microbatch.old_thought_logprobs, think_index
                    )
                )
                totals["thought_dim_abs_log_ratio_max"] = torch.maximum(
                    totals["thought_dim_abs_log_ratio_max"],
                    thought_log_ratio.abs().max(),
                )
                totals["thought_joint_abs_log_ratio_max"] = torch.maximum(
                    totals["thought_joint_abs_log_ratio_max"],
                    thought_log_ratio.sum(-1).abs().max(),
                )
                totals["thought_kl_sum"] += (
                    thought_reverse_kl_factors.detach().sum()
                )

    if finite_guards:
        # One host sync resolves every shard's loss guard, after all backward
        # passes are queued. This must stay ABOVE every ``step_optimizers``
        # call in this function -- including the ``value_only`` return path --
        # because ``actor_step``/``critic_step`` default to True and standalone
        # callers do step here.
        guarded = torch.stack(
            [value.reshape(()).float() for _, _, value, _ in finite_guards]
        )
        if not torch.isfinite(guarded).all():
            offenders = []
            for (shard, label, _, components), finite in zip(
                finite_guards, torch.isfinite(guarded).tolist(), strict=True
            ):
                if finite:
                    continue
                detail = " ".join(
                    f"{name}={float(component)}"
                    for name, component in components.items()
                )
                offenders.append(
                    f"shard={shard} {label}" + (f" {detail}" if detail else "")
                )
            raise RuntimeError(
                "non-finite loss before optimizer step: "
                + "; ".join(offenders)
            )

    action_denom = denominators["action"]
    # Pinned-EMIT batches store zero-width thoughts: keep the per-dimension
    # thought telemetry finite (zero) instead of dividing by a zero width or
    # reporting the untouched +/-inf extrema.
    thought_dim = max(batch.old_thought_logprobs.size(-1), 1)
    for extremum in ("thought_log_sigma_min", "thought_log_sigma_max"):
        totals[extremum] = torch.where(
            torch.isfinite(totals[extremum]),
            totals[extremum],
            torch.zeros_like(totals[extremum]),
        )
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
        "gate_action_count": batch.stop_mask.sum(),
        "emit_action_count": emit_actions.sum(),
        "forced_initial_think_action_count": forced_initial_think_actions.sum(),
    }
    if value_only:
        metric_tensors["critic_grad_norm"] = gradient_norm_tensor(
            critic.parameters()
        )
        if critic_step:
            step_optimizers(optimizers, "critic")
        return scalar_tensors_to_floats(metric_tensors)

    # Norms are cumulative across prompt groups; the final group reports the
    # complete pre-step norm. Fresh probes are registered
    # under blocks[-1], so exclude them from the trunk norm by identity.
    excluded_parameter_ids = non_trunk_parameter_ids(backbone)
    grad_norms = {
        "trunk_grad_norm": gradient_norm_tensor(
            parameter
            for parameter in backbone.parameters()
            if id(parameter) not in excluded_parameter_ids
        ),
        "renderer_grad_norm": gradient_norm_tensor(
            renderer_parameters(backbone)
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
        step_optimizers(optimizers, "critic")
    if actor_step and "actor" in optimizers:
        step_optimizers(optimizers, "actor")
    metric_tensors.update(
        policy_loss=totals["policy_loss"],
        thought_reverse_kl_penalty=totals["thought_reverse_kl_penalty"],
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
        stop_probability=(
            totals["stop_probability_sum"] / denominators["gate"]
        ),
        behavior_stop_probability=(
            totals["behavior_stop_probability_sum"] / denominators["gate"]
        ),
        stop_probability_delta=(
            totals["stop_probability_delta_sum"] / denominators["gate"]
        ),
        stop_probability_abs_delta=(
            totals["stop_probability_abs_delta_sum"] / denominators["gate"]
        ),
        stop_probability_abs_delta_max=totals[
            "stop_probability_abs_delta_max"
        ],
        gate_behavior_kl=totals["gate_kl_sum"] / denominators["gate"],
        gate_behavior_kl_exact=(
            totals["gate_exact_kl_sum"] / denominators["gate"]
        ),
        gate_abs_log_ratio_max=totals["gate_abs_log_ratio_max"],
        gate_policy_clip_fraction=(
            totals["gate_policy_clip_count"] / denominators["gate"]
        ),
        renderer_behavior_kl=totals["renderer_kl_sum"] / denominators["emit"],
        thought_behavior_kl_joint=(
            totals["thought_kl_sum"] / denominators["thought"]
        ),
        thought_behavior_kl_per_dim=(
            totals["thought_kl_sum"]
            / (denominators["thought"] * thought_dim)
        ),
        thought_trust_d_mean=(
            totals["thought_trust_sum"] / denominators["thought"]
        ),
        thought_trust_d_max=totals["thought_trust_max"],
        thought_projection_penalty=totals["thought_projection_penalty"],
        policy_behavior_kl_per_action=(
            totals["gate_kl_sum"]
            + totals["renderer_kl_sum"]
            + totals["thought_kl_sum"]
        ) / policy_action_denominator.clamp_min(1),
        joint_abs_log_ratio_max=totals["joint_abs_log_ratio_max"],
        thought_dim_abs_log_ratio_max=(
            totals["thought_dim_abs_log_ratio_max"]
        ),
        thought_joint_abs_log_ratio_max=(
            totals["thought_joint_abs_log_ratio_max"]
        ),
        harmful_positive_log_ratio_max=(
            totals["harmful_positive_log_ratio_max"]
        ),
        advantage_mean=advantage_mean,
        advantage_std=advantage_variance.sqrt(),
        reward=batch.reward_scalar.mean(),
        thought_log_sigma_mean=(
            totals["thought_log_sigma_sum"]
            / (denominators["thought"] * thought_dim)
        ),
        thought_log_sigma_std=(
            totals["thought_log_sigma_square_sum"]
            / (denominators["thought"] * thought_dim)
            - (
                totals["thought_log_sigma_sum"]
                / (denominators["thought"] * thought_dim)
            ).square()
        ).clamp_min(0).sqrt(),
        thought_log_sigma_min=totals["thought_log_sigma_min"],
        thought_log_sigma_max=totals["thought_log_sigma_max"],
        thought_sigma_mean=(
            totals["thought_sigma_sum"]
            / (denominators["thought"] * thought_dim)
        ),
        thought_expected_noise_norm=(
            totals["thought_expected_noise_norm_sum"] / denominators["thought"]
        ),
        thought_realized_noise_norm=(
            totals["thought_realized_noise_norm_sum"] / denominators["thought"]
        ),
        thought_normalized_noise_rms=(
            totals["thought_normalized_noise_square_sum"]
            / (denominators["thought"] * thought_dim)
        ).sqrt(),
        thought_mean_norm=(
            totals["thought_mean_norm_sum"] / denominators["thought"]
        ),
        thought_raw_abs_max=totals["thought_raw_abs_max"],
        thought_raw_abs_gt_0_5_fraction=(
            totals["thought_raw_abs_gt_0_5_count"]
            / (denominators["thought"] * thought_dim)
        ),
        thought_raw_abs_gt_0_8_fraction=(
            totals["thought_raw_abs_gt_0_8_count"]
            / (denominators["thought"] * thought_dim)
        ),
        thought_raw_abs_gt_1_fraction=(
            totals["thought_raw_abs_gt_1_count"]
            / (denominators["thought"] * thought_dim)
        ),
        thought_raw_abs_gt_2_fraction=(
            totals["thought_raw_abs_gt_2_count"]
            / (denominators["thought"] * thought_dim)
        ),
        thought_transform_distortion_rms=(
            totals["thought_transform_distortion_square_sum"]
            / (denominators["thought"] * thought_dim)
        ).sqrt(),
        thought_squashed_raw_norm=(
            totals["thought_squashed_raw_norm_sum"] / denominators["thought"]
        ),
        thought_adapter_output_norm=(
            totals["thought_adapter_output_norm_sum"]
            / denominators["thought"]
        ),
        post_thought_belief_norm=(
            totals["post_thought_belief_norm_sum"]
            / denominators["thought"]
        ),
        thought_mean_abs_max=totals["thought_mean_abs_max"],
        thought_mean_abs_gt_0_8_fraction=(
            totals["thought_mean_abs_gt_0_8_count"]
            / (denominators["thought"] * thought_dim)
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
        thought_reverse_kl_coef=float(thought_reverse_kl_coef),
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
    replay_slot_budget: int | None = None,
    replay_function: Callable = replay_head_inputs,
    emit_logprob_function: Callable = compact_emit_token_logprobs,
    replay_plans: list[ReplayPlan] | None = None,
) -> dict[str, float]:
    """Evaluate the just-updated policy on its behavior trajectories.

    On behavior-age 0, optimized ratios are exactly one before the first actor
    step; clipping therefore cannot reveal how far that step moved the deployed
    policy. This read-only replay measures the actual post-step drift
    without reusing trajectories for a gradient. It runs only at the explicit
    diagnostic cadence because it costs one additional policy forward.
    ``replay_function`` and ``emit_logprob_function`` deliberately stay eager
    in the trainer: invoking a grad-enabled training artifact under this
    function's no-grad context would force Dynamo/AOTAutograd to compile a
    second guarded graph, while enabling gradients here retains a full replay
    graph and can exceed peak memory. Both are parameters rather than module
    lookups precisely because the trainer rebinds those globals to compiled
    artifacts.
    """
    if not batches:
        raise ValueError("post-update drift requires at least one rollout batch")
    if replay_plans is not None and len(replay_plans) != len(batches):
        raise ValueError("post-update drift plan count must match batches")
    device = next(wrapper.parameters()).device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    totals = {
        "gate_kl": zero.clone(),
        "gate_exact_kl": zero.clone(),
        "renderer_kl": zero.clone(),
        "thought_kl": zero.clone(),
        "joint_abs_log_ratio_max": zero.clone(),
        "thought_joint_abs_log_ratio_max": zero.clone(),
        "gate_abs_log_ratio_max": zero.clone(),
        "behavior_stop_probability": zero.clone(),
        "current_stop_probability": zero.clone(),
        "stop_probability_abs_delta": zero.clone(),
        "stop_probability_abs_delta_max": zero.clone(),
        "gate_count": zero.clone(),
        "emit_count": zero.clone(),
        "thought_count": zero.clone(),
        "action_count": zero.clone(),
    }
    for batch_index, stored_batch in enumerate(batches):
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
        replay_plan = (
            replay_plans[batch_index]
            if replay_plans is not None
            else build_replay_plan(
                stored_batch,
                replay_max_trajectories,
                replay_attention_budget,
                replay_bucket,
                replay_slot_budget,
            ).to(device)
        )
        replay_plan.validate_settings(
            replay_max_trajectories,
            replay_attention_budget,
            replay_bucket,
            replay_slot_budget,
            require_action_indices=True,
        )
        for microbatch, shard in iter_planned_replay_microbatches(
            batch, replay_plan
        ):
            beliefs, stream_inputs = replay_function(
                wrapper, microbatch
            )
            stop_logits = wrapper.gate.stop_logit(beliefs)
            stop_logprobs = -F.binary_cross_entropy_with_logits(
                stop_logits,
                microbatch.actions.float(),
                reduction="none",
            )
            gate_log_ratio = stop_logprobs - microbatch.old_stop_logprobs.float()
            (
                behavior_stop_probability,
                current_stop_probability,
                exact_gate_kl,
            ) = exact_bernoulli_behavior_kl(
                stop_logits,
                microbatch.actions,
                microbatch.old_stop_logprobs,
            )
            stop_probability_abs_delta = (
                current_stop_probability - behavior_stop_probability
            ).abs()

            token_logprobs = torch.zeros_like(microbatch.old_token_logprobs).float()
            if shard.emit_index.numel():
                emit_index = shard.emit_index
                compact_token_logprobs = emit_logprob_function(
                    wrapper,
                    compact_slots(stream_inputs, emit_index),
                    compact_slots(beliefs, emit_index),
                    compact_next_slots(
                        microbatch.token_ids, emit_index
                    ),
                )
                scatter_slots(token_logprobs, emit_index, compact_token_logprobs)
            token_log_ratio = (
                token_logprobs - microbatch.old_token_logprobs.float()
            )

            thought_joint = torch.zeros_like(token_logprobs)
            old_thought_joint = torch.zeros_like(token_logprobs)
            thought_log_ratio = None
            if shard.think_index.numel():
                think_index = shard.think_index
                thought_means, thought_targets = compact_thought_actions(
                    wrapper, microbatch, beliefs, think_index
                )
                thought_log_sigma = wrapper.transition.predict_log_sigma(
                    compact_slots(beliefs, think_index)
                )
                thought_logprobs = wrapper.transition.per_dim_log_prob(
                    thought_targets, thought_means, thought_log_sigma
                ).float()
                old_thought_logprobs = compact_slots(
                    microbatch.old_thought_logprobs, think_index
                ).float()
                thought_log_ratio = thought_logprobs - old_thought_logprobs
                scatter_slots(
                    thought_joint, think_index, thought_logprobs.sum(-1)
                )
                scatter_slots(
                    old_thought_joint,
                    think_index,
                    old_thought_logprobs.sum(-1),
                )

            new_joint, old_joint = joint_action_logprobs(
                stop_logprobs,
                microbatch.old_stop_logprobs.float(),
                token_logprobs,
                microbatch.old_token_logprobs.float(),
                thought_joint,
                old_thought_joint,
                microbatch.stop_mask,
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
                * microbatch.stop_mask
            ).sum()
            totals["gate_exact_kl"] += (
                exact_gate_kl * microbatch.stop_mask
            ).sum()
            totals["gate_abs_log_ratio_max"] = torch.maximum(
                totals["gate_abs_log_ratio_max"],
                (gate_log_ratio * microbatch.stop_mask).abs().max(),
            )
            totals["behavior_stop_probability"] += (
                behavior_stop_probability * microbatch.stop_mask
            ).sum()
            totals["current_stop_probability"] += (
                current_stop_probability * microbatch.stop_mask
            ).sum()
            totals["stop_probability_abs_delta"] += (
                stop_probability_abs_delta * microbatch.stop_mask
            ).sum()
            totals["stop_probability_abs_delta_max"] = torch.maximum(
                totals["stop_probability_abs_delta_max"],
                (
                    stop_probability_abs_delta
                    * microbatch.stop_mask
                ).max(),
            )
            totals["renderer_kl"] += (
                (torch.expm1(token_log_ratio) - token_log_ratio)
                * microbatch.emit_mask
            ).sum()
            if thought_log_ratio is not None:
                totals["thought_kl"] += (
                    torch.expm1(thought_log_ratio) - thought_log_ratio
                ).sum()
                totals["thought_joint_abs_log_ratio_max"] = torch.maximum(
                    totals["thought_joint_abs_log_ratio_max"],
                    thought_log_ratio.sum(-1).abs().max(),
                )
            totals["gate_count"] += microbatch.stop_mask.sum()
            totals["emit_count"] += microbatch.emit_mask.sum()
            totals["thought_count"] += shard.think_index.numel()
            totals["action_count"] += action_mask.sum()
        del batch

    return scalar_tensors_to_floats(
        {
            "kl/post_update_gate_behavior": (
                totals["gate_kl"] / totals["gate_count"].clamp_min(1)
            ),
            "kl/post_update_gate_behavior_exact": (
                totals["gate_exact_kl"] / totals["gate_count"].clamp_min(1)
            ),
            "gate_same_state/post_update_behavior_continue_probability": (
                1.0
                - totals["behavior_stop_probability"]
                / totals["gate_count"].clamp_min(1)
            ),
            "gate_same_state/post_update_current_continue_probability": (
                1.0
                - totals["current_stop_probability"]
                / totals["gate_count"].clamp_min(1)
            ),
            "gate_same_state/post_update_stop_probability_abs_delta": (
                totals["stop_probability_abs_delta"]
                / totals["gate_count"].clamp_min(1)
            ),
            "gate_same_state/post_update_stop_probability_abs_delta_max": (
                totals["stop_probability_abs_delta_max"]
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
            "ratio/post_update_gate_abs_log_max": totals[
                "gate_abs_log_ratio_max"
            ],
            "ratio/post_update_thought_joint_abs_log_max": totals[
                "thought_joint_abs_log_ratio_max"
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
    reasoning_mode = getattr(args, "reasoning_mode", "latent")
    payload = {
        "step": step,
        "value_warmup_step": warmup_step,
        "execution_schema": execution_schema_for_adapter(
            wrapper.thought_adapter_kind,
            wrapper.thought_action_transform,
            getattr(args, "rollout_scheduler", "lockstep"),
        ),
        "actor_objective_schema": ACTOR_OBJECTIVE_SCHEMA,
        "replay_numerics_schema": REPLAY_NUMERICS_SCHEMA,
        "source_provenance": getattr(args, "source_provenance", None),
        "prompt_order_schema": PROMPT_ORDER_SCHEMA,
        "math_data_identity": sampler.dataset_identity,
        "reward_schema": REWARD_SCHEMA,
        "reasoning_mode": reasoning_mode,
        "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
        "rollout_policy_schema": rollout_policy_schema_for_mode(reasoning_mode),
        "thought_input_schema": wrapper.thought_input_schema,
        "thought_action_transform_schema": (
            wrapper.thought_action_transform_schema
        ),
        "thought_distribution_schema": THOUGHT_DISTRIBUTION_SCHEMA,
        "thought_mean_schema": THOUGHT_MEAN_SCHEMA,
        "thought_sigma_init_schema": wrapper.sigma_state_init_schema,
        "critic_adapter_init_schema": critic.adapter_init_schema,
        "optimizer_schema": optimizer_schema_for_trunk_optimizer(
            getattr(args, "trunk_optimizer", "adamw")
        ),
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
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    backbone = load_model(args.checkpoint, device)
    is_nano = backbone.architecture.startswith("nanogpt_mini")
    if not is_nano and not backbone.architecture.endswith(
        "probes_pope_belief_attached_ce_onepass_2k"
    ):
        raise ValueError(
            "belief-renderer VAPO requires a nanogpt_mini checkpoint or one "
            "pretrained with [token_latent, raw_belief] CE; old "
            "predicted-latent probe "
            f"architecture {backbone.architecture!r} is incompatible"
        )

    # Backbone-derived context contract: fresh PoPE executes 5x its pretrained
    # window, so RL keeps the full 1024-token prompt plus a 4096-slot stream.
    # Nano's half-truncate RoPE (base 1024) has no extrapolation story, so RL
    # stays strictly inside the pretraining window the checkpoint records
    # (train_seq_len; 1024 for checkpoints predating the field). The response
    # budget targets the standard ~1k tokens for DAPO math as soon as the
    # window affords it after the prompt.
    if is_nano:
        context_tokens = getattr(backbone, "train_context_tokens", 1024)
        default_prompt_tokens = 512
        default_response_tokens = min(
            1024, (context_tokens - default_prompt_tokens) // 2
        )
    else:
        context_tokens = POSTTRAIN_CONTEXT_TOKENS
        default_prompt_tokens = POSTTRAIN_PROMPT_TOKENS
        default_response_tokens = POSTTRAIN_RESPONSE_TOKENS
    if args.prompt_tokens is None:
        args.prompt_tokens = default_prompt_tokens
    if args.continuation_tokens is None:
        args.continuation_tokens = default_response_tokens
    if args.aime_max_tokens is None:
        args.aime_max_tokens = default_response_tokens
    if args.bench_max_tokens is None:
        args.bench_max_tokens = default_response_tokens
    if args.answer_tokens < 2:
        parser.error("--answer-tokens must fit an answer plus its terminator")

    pin_emit = args.reasoning_mode != "latent"
    rollout_policy_schema = rollout_policy_schema_for_mode(args.reasoning_mode)
    if pin_emit and args.max_stream_steps is not None:
        parser.error(
            "--max-stream-steps is a latent-mode knob; pinned-EMIT modes "
            "always use exactly one stream slot per emitted token"
        )

    def mode_budgets(max_tokens: int) -> tuple[int, int]:
        return mode_rollout_budget(
            args.reasoning_mode,
            max_tokens,
            answer_tokens=args.answer_tokens,
            prompt_tokens=args.prompt_tokens,
            context_tokens=context_tokens,
        )

    train_max_new_tokens, max_stream_steps = training_rollout_budget(
        args.reasoning_mode,
        args.continuation_tokens,
        answer_tokens=args.answer_tokens,
        prompt_tokens=args.prompt_tokens,
        context_tokens=context_tokens,
        max_stream_steps=args.max_stream_steps,
    )
    # The eval budget always scales with its own emit cap; an explicit
    # --max-stream-steps is a training-rollout knob.
    aime_max_new_tokens, aime_stream_steps = mode_budgets(args.aime_max_tokens)
    bench_max_new_tokens, bench_stream_steps = mode_budgets(
        args.bench_max_tokens
    )
    # Persist effective values as well as the user's raw override. Readers
    # should not have to reconstruct backbone-dependent defaults from a later
    # checkout merely to reproduce a checkpoint's rollout policy.
    args.resolved_train_max_new_tokens = train_max_new_tokens
    args.resolved_train_max_stream_steps = max_stream_steps
    args.resolved_aime_max_new_tokens = aime_max_new_tokens
    args.resolved_aime_max_stream_steps = aime_stream_steps
    args.resolved_bench_max_new_tokens = bench_max_new_tokens
    args.resolved_bench_max_stream_steps = bench_stream_steps
    validate_posttraining_context_budget(
        args.prompt_tokens, max_stream_steps, context_tokens
    )
    validate_posttraining_context_budget(
        args.prompt_tokens, aime_stream_steps, context_tokens
    )
    validate_posttraining_context_budget(
        args.prompt_tokens, bench_stream_steps, context_tokens
    )

    backbone.eval()
    # Full-model RL: every parameter on a deployed policy path trains — trunk,
    # embeddings, fresh thought mean/sigma, renderer, gate, and adapter. The
    # retired pretrained prediction projector remains checkpointed but has no
    # graph edge; the backbone critic probe is frozen and unused.
    wrapper = LatentThoughtModel(
        backbone,
        thought_adapter=args.thought_adapter,
        sigma_state_init=args.thought_sigma_state_init,
        thought_action_transform=args.thought_action_transform,
    ).to(device)
    # No module here behaves differently under train(): pin eval mode once so
    # the training flag (a dynamo guard) never flips between the step-0 evals
    # and the training loop and re-specializes the compiled step.
    wrapper.eval()
    if not 0.0 < args.init_stop_thinking_probability < 1.0:
        raise SystemExit(
            "--init-stop-thinking-probability must be strictly inside (0, 1)"
        )
    if not math.isfinite(args.thought_mean_gain_init) or (
        args.thought_mean_gain_init <= 0.0
    ):
        raise SystemExit("--thought-mean-gain-init must be finite and positive")
    # The CLI owns both fresh continuous-policy scales. Ordinary resume loads
    # learned values over these; explicit actor restart copies them into the
    # critic-warm payload before loading it.
    wrapper.transition.mean_head.reset_output_gain(
        args.thought_mean_gain_init
    )
    wrapper.transition.set_noise_level(args.thought_log_sigma_init)
    with torch.no_grad():
        # P(STOP) = sigmoid(bias) while zero-init weights ignore the belief.
        wrapper.gate.head.bias.fill_(
            math.log(
                args.init_stop_thinking_probability
                / (1.0 - args.init_stop_thinking_probability)
            )
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
        else:
            init_args = actor_init_payload.get("args", {})
            source_sigma_state_init = init_args.get(
                "thought_sigma_state_init", "constant"
            )
            if source_sigma_state_init != args.thought_sigma_state_init:
                raise ValueError(
                    f"{initialization_path} used --thought-sigma-state-init "
                    f"{source_sigma_state_init!r}, not "
                    f"{args.thought_sigma_state_init!r}; a trained actor must "
                    "retain its initialization lineage"
                )
        migrate_legacy_wrapper_checkpoint(
            actor_init_payload,
            wrapper,
            initialize_fresh_mean=bool(args.actor_critic_init),
            initialize_fresh_adapter=bool(args.actor_critic_init),
            initialize_fresh_gate=bool(args.actor_critic_init),
        )
        validate_renderer_checkpoint(
            actor_init_payload,
            initialization_path,
            # A critic-warmup checkpoint has not updated the actor; the
            # explicit migration above installed this run's fresh transition.
            allow_transition_reset=bool(args.actor_critic_init),
            expected_rollout_policy_schema=rollout_policy_schema,
            expected_thought_input_schema=wrapper.thought_input_schema,
            expected_thought_action_transform_schema=(
                wrapper.thought_action_transform_schema
            ),
        )
        wrapper.load_state_dict(actor_init_payload["model"], strict=True)
        actor_init_provenance = {
            "checkpoint": str(initialization_path),
            "critic_initialized": bool(args.actor_critic_init),
            "curriculum_transition": bool(args.curriculum_init),
            "source_step": actor_init_payload.get("step"),
            "source_reward_schema": actor_init_payload.get("reward_schema"),
            "source_execution_schema": actor_init_payload.get("execution_schema"),
            "source_actor_objective_schema": actor_init_payload.get(
                "actor_objective_schema"
            ),
            "source_value_warmup_step": actor_init_payload.get(
                "value_warmup_step"
            ),
            "source_sampler_cursor": int(actor_init_payload["sampler_cursor"]),
            "target_sampler_cursor": 0 if args.curriculum_init else int(
                actor_init_payload["sampler_cursor"]
            ),
            "fresh_mean_initialized": bool(args.actor_critic_init),
            "fresh_gate_initialized": bool(args.actor_critic_init),
            "fresh_gate_stop_probability": (
                args.init_stop_thinking_probability
                if args.actor_critic_init
                else None
            ),
            "fresh_mean_output_gain": (
                args.thought_mean_gain_init if args.actor_critic_init else None
            ),
            "fresh_log_sigma": (
                args.thought_log_sigma_init if args.actor_critic_init else None
            ),
            "fresh_sigma_state_init": (
                args.thought_sigma_state_init if args.actor_critic_init else None
            ),
            "fresh_adapter_initialized": bool(args.actor_critic_init),
            "fresh_adapter_kind": (
                args.thought_adapter if args.actor_critic_init else None
            ),
        }
    for parameter in wrapper.parameters():
        parameter.requires_grad_(True)
    if hasattr(backbone, "critic_probe"):
        # Fresh lineage only: the pretrained critic probe stays checkpointed
        # but has no graph edge. Nano backbones carry no probes.
        for parameter in backbone.critic_probe.parameters():
            parameter.requires_grad_(False)

    if args.value_anchored_support:
        value_num_bins, value_v_min, value_v_max = anchored_unit_geometry(
            args.value_bins, args.value_margin_bins
        )
    else:
        value_num_bins, value_v_min, value_v_max = args.value_bins, 0.0, 1.0
    critic = SeparateCritic(
        fresh_trunk(backbone, device),
        num_bins=value_num_bins,
        sigma_ratio=args.value_sigma_ratio,
        v_min=value_v_min,
        v_max=value_v_max,
        prior_value=args.value_prior,
        adapter_init=args.critic_adapter_init,
        thought_action_transform=args.thought_action_transform,
    ).to(device)
    critic.eval()  # no dropout in this architecture; keep norms deterministic
    if args.actor_critic_init:
        init_args = actor_init_payload.get("args", {})
        if not value_support_geometry_matches(init_args, args):
            raise ValueError(
                "--actor-critic-init warm critic was trained on a different "
                "value support geometry (anchored/bins/margin/sigma_ratio "
                f"{init_args.get('value_anchored_support', False)}/"
                f"{init_args.get('value_bins')}/"
                f"{init_args.get('value_margin_bins')}/"
                f"{init_args.get('value_sigma_ratio')}); rerun critic warmup "
                "under the current flags or start with --actor-init"
            )
        source_critic_adapter_init = init_args.get(
            "critic_adapter_init", "identity"
        )
        if source_critic_adapter_init != args.critic_adapter_init:
            raise ValueError(
                "--actor-critic-init warm critic used "
                f"--critic-adapter-init {source_critic_adapter_init!r}, not "
                f"{args.critic_adapter_init!r}; rerun critic warmup under the "
                "current adapter initialization"
            )
        expected_optimizer_schema = optimizer_schema_for_trunk_optimizer(
            args.trunk_optimizer
        )
        if (
            actor_init_payload.get("optimizer_schema")
            != expected_optimizer_schema
        ):
            raise ValueError(
                "--actor-critic-init optimizer schema is "
                f"{actor_init_payload.get('optimizer_schema')!r}, expected "
                f"{expected_optimizer_schema!r}; rerun critic warmup under "
                "the current optimizer implementation"
            )
        critic.load_state_dict(actor_init_payload["critic"], strict=True)

    optimizers = build_optimizers(
        wrapper,
        critic,
        learning_rate=args.learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        trunk_optimizer=args.trunk_optimizer,
        muon_learning_rate=args.muon_learning_rate,
        critic_muon_learning_rate=args.critic_muon_learning_rate,
    )
    if args.actor_critic_init:
        # A step-0 critic-warm checkpoint has never stepped its actor. Its
        # optimizer state is therefore empty and carries no momentum to
        # preserve; loading its obsolete three-group layout would only couple
        # the gate and recurrent adapter again. Preserve the trained critic's
        # optimizer state and start the untouched actor optimizer in the
        # current six-group layout.
        source_optimizers = actor_init_payload["optimizers"]
        for name, source in source_optimizers.items():
            if name.startswith("actor") and source["state"]:
                raise ValueError(
                    "--actor-critic-init requires a pristine actor optimizer"
                )
        critic_names = [name for name in optimizers if name.startswith("critic")]
        missing = [name for name in critic_names if name not in source_optimizers]
        extra = [
            name
            for name in source_optimizers
            if name.startswith("critic") and name not in optimizers
        ]
        if missing or extra:
            raise ValueError(
                "--actor-critic-init checkpoint optimizer layout "
                f"{sorted(n for n in source_optimizers if n.startswith('critic'))} "
                f"does not match the --trunk-optimizer {args.trunk_optimizer} "
                f"layout {sorted(critic_names)}; rerun the critic warmup or "
                "pass the matching --trunk-optimizer"
            )
        for name in critic_names:
            optimizers[name].load_state_dict(source_optimizers[name])
        reassert_learning_rates(
            {name: optimizers[name] for name in critic_names}, args
        )

    tokenizer = load_posttraining_tokenizer(
        backbone.architecture, FreshHyperparameters.tokenizer_path
    )
    # dict.fromkeys dedupes while keeping order: GPT-2's single
    # <|endoftext|> token reports as both EOS and BOS.
    stop_ids = tuple(
        dict.fromkeys(
            t for t in (tokenizer.eos_id(), tokenizer.bos_id()) if t >= 0
        )
    )
    if not stop_ids:
        raise RuntimeError(
            "posttraining requires a valid BOS or EOS token for explicit "
            "trajectory termination"
        )
    answer_prefix_ids: tuple[int, ...] = ()
    if args.reasoning_mode == "none":
        # Mathmix QA documents close with "\nAnswer: <x>" before EOS.
        # Teacher-forcing the prefix onto the prompt leaves the policy only
        # the answer value and its terminator to emit; scoring and decoding
        # prepend the same ids before parsing.
        answer_prefix_ids = answer_prefix_token_ids(tokenizer)
        if not answer_prefix_ids:
            raise RuntimeError(
                "the none-mode Answer: prefix encoded to zero tokens"
            )
        if len(answer_prefix_ids) >= args.prompt_tokens:
            raise RuntimeError(
                "the encoded Answer: prefix must leave room in the prompt "
                "budget"
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
            if actor_cursor < 0:
                raise ValueError(
                    "initialization source has no valid prompt cursor: "
                    f"{actor_cursor}"
                )
            sampler.cursor = actor_cursor
            print(
                f"actor{' and critic' if args.actor_critic_init else ''} initialized "
                f"from {initialization_path} at source step "
                f"{actor_init_payload.get('step')} and sampler cursor {actor_cursor}",
                flush=True,
            )
    seq_len = FreshHyperparameters.train_seq_len
    is_gpt2_vocab = is_nano and "gpt2vocab" in backbone.architecture
    if is_gpt2_vocab:
        # GPT-2 tokens map to fixed byte strings, so the guard reduces to a
        # direct LUT sum: zero leading-space/boundary tables make eval_val's
        # SentencePiece correction term vanish.
        gpt2_bytes = torch.load(
            "data/tokenizers/gpt2_byte_lut.pt", weights_only=True
        ).to(device)
        no_correction = torch.zeros(
            gpt2_bytes.size(0), dtype=torch.bool, device=device
        )
        luts = (gpt2_bytes, no_correction, no_correction)
    else:
        luts = baseline.build_sentencepiece_luts(
            tokenizer, FreshHyperparameters.vocab_size, device
        )
    # The BPB guard shares the checkpoint's own tokenizer family, and nano
    # pretrains against its own shard family; keep the guard on the same
    # validation bytes as nano's own pretraining val_bpb (DATA_PATH overrides).
    bpb_val_files = FreshHyperparameters.val_files
    if is_nano:
        default_val_dataset = (
            "data/datasets/fineweb10B_gpt2"
            if is_gpt2_vocab
            else "data/datasets/fineweb_onepass_sp1024"
        )
        bpb_val_files = os.path.join(
            os.environ.get("DATA_PATH", default_val_dataset),
            "fineweb_val_*.bin",
        )
    val_tokens = baseline.load_validation_tokens(bpb_val_files, seq_len)
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
        target_execution_schema = execution_schema_for_adapter(
            args.thought_adapter,
            args.thought_action_transform,
            args.rollout_scheduler,
        )
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_args = payload.get("args", {})
        source_sigma_state_init = resume_args.get(
            "thought_sigma_state_init", "constant"
        )
        if source_sigma_state_init != args.thought_sigma_state_init:
            raise ValueError(
                "resume checkpoint used --thought-sigma-state-init "
                f"{source_sigma_state_init!r}, not "
                f"{args.thought_sigma_state_init!r}"
            )
        source_critic_adapter_init = resume_args.get(
            "critic_adapter_init", "identity"
        )
        if source_critic_adapter_init != args.critic_adapter_init:
            raise ValueError(
                "resume checkpoint used --critic-adapter-init "
                f"{source_critic_adapter_init!r}, not "
                f"{args.critic_adapter_init!r}"
            )
        migrate_legacy_wrapper_checkpoint(payload, wrapper)
        validate_renderer_checkpoint(
            payload,
            args.resume,
            expected_rollout_policy_schema=rollout_policy_schema,
            expected_thought_input_schema=wrapper.thought_input_schema,
            expected_thought_action_transform_schema=(
                wrapper.thought_action_transform_schema
            ),
        )
        if not resume_execution_schema_compatible(
            payload,
            expected_execution_schema=target_execution_schema,
            allow_reverse_kl_migration=args.migrate_reverse_kl_resume,
            allow_performance_migration=args.migrate_v20_execution_resume,
            allow_joint_clip_migration=args.migrate_joint_clip_resume,
            allow_anchored_value_migration=args.migrate_anchored_value_resume,
            allow_projected_thought_migration=(
                args.migrate_projected_thought_resume
            ),
            allow_thought_reverse_kl_migration=(
                args.migrate_thought_reverse_kl_resume
            ),
            allow_rollout_scheduler_migration=(
                args.migrate_rollout_scheduler_resume
            ),
        ):
            raise ValueError(
                "resume checkpoint execution schema must be "
                f"{target_execution_schema!r}; "
                f"got {payload.get('execution_schema')!r}. Use --actor-init or "
                "--actor-critic-init for an initialization restart, or "
                "the migration flags matching the source: v23 requires "
                "--migrate-thought-reverse-kl-resume; v22 requires that plus "
                "--migrate-projected-thought-resume; v21 additionally requires "
                "--migrate-anchored-value-resume; v20 additionally requires "
                "--migrate-joint-clip-resume; v19 additionally requires "
                "--migrate-v20-execution-resume; v18 requires all of those "
                "plus --migrate-reverse-kl-resume. A scheduler change also "
                "requires --migrate-rollout-scheduler-resume."
            )
        if not resume_replay_schema_compatible(
            payload,
            allow_compact_replay_migration=(
                args.migrate_compact_replay_resume
            ),
        ):
            raise ValueError(
                "resume checkpoint replay numerics schema must be "
                f"{REPLAY_NUMERICS_SCHEMA!r}; got "
                f"{payload.get('replay_numerics_schema')!r}. Pre-compact "
                "checkpoints require --migrate-compact-replay-resume."
            )
        if payload.get("reward_schema") != REWARD_SCHEMA:
            raise ValueError(
                f"resume checkpoint reward schema must be {REWARD_SCHEMA!r}; "
                f"got {payload.get('reward_schema')!r}"
            )
        if payload.get("actor_objective_schema") != ACTOR_OBJECTIVE_SCHEMA:
            raise ValueError(
                "resume checkpoint actor objective schema must be "
                f"{ACTOR_OBJECTIVE_SCHEMA!r}; got "
                f"{payload.get('actor_objective_schema')!r}. Use "
                "--actor-init for an initialization restart under the "
                "current objective."
            )
        if payload.get("math_data_identity") != data_identity:
            raise ValueError(
                "resume checkpoint's prompt cursor belongs to different dataset "
                "bytes, exclusions, or ordering"
            )
        resume_seed = payload.get("args", {}).get("seed")
        if resume_seed is not None and int(resume_seed) != args.seed:
            raise ValueError(
                "resume requires the checkpoint's --seed: prompt epochs after "
                f"the first reshuffle from it (checkpoint seed {resume_seed}, "
                f"got {args.seed})"
            )
        wrapper.load_state_dict(payload["model"], strict=True)
        anchored_value_migration = None
        if args.migrate_anchored_value_resume:
            anchored_value_migration = migrate_anchored_value_resume(
                payload, critic
            )
        else:
            resume_args = payload.get("args", {})
            if not value_support_geometry_matches(resume_args, args):
                raise ValueError(
                    "resume checkpoint critic support geometry "
                    "(anchored/bins/margin/sigma_ratio "
                    f"{resume_args.get('value_anchored_support', False)}/"
                    f"{resume_args.get('value_bins')}/"
                    f"{resume_args.get('value_margin_bins')}/"
                    f"{resume_args.get('value_sigma_ratio')}) does not match "
                    "the current flags; resume with the checkpoint's value "
                    "support flags"
                )
            critic.load_state_dict(payload["critic"], strict=True)
        expected_optimizer_schema = optimizer_schema_for_trunk_optimizer(
            args.trunk_optimizer
        )
        if (
            not args.reset_optimizers_on_resume
            and payload.get("optimizer_schema") != expected_optimizer_schema
        ):
            raise ValueError(
                "resume checkpoint optimizer schema is "
                f"{payload.get('optimizer_schema')!r}, expected "
                f"{expected_optimizer_schema!r}; pass "
                "--reset-optimizers-on-resume to keep model/cursor state "
                "while discarding incompatible optimizer state"
            )
        if args.reset_optimizers_on_resume:
            # Trunk-optimizer migration: keep model, critic, step, and prompt
            # cursor; every optimizer starts with empty state under the
            # freshly built layout. Adam second moments rebuild over the next
            # few hundred updates — a transient, not a schedule change.
            print(
                "resume with reset optimizers: discarding "
                f"{sorted(payload['optimizers'])} state for the "
                f"--trunk-optimizer {args.trunk_optimizer} layout "
                f"{sorted(optimizers)}",
                flush=True,
            )
        elif set(payload["optimizers"]) != set(optimizers):
            raise ValueError(
                "resume checkpoint optimizer layout "
                f"{sorted(payload['optimizers'])} does not match the "
                f"--trunk-optimizer {args.trunk_optimizer} layout "
                f"{sorted(optimizers)}; checkpoints from before the Muon "
                "trunk split resume with --trunk-optimizer adamw, or migrate "
                "with --reset-optimizers-on-resume"
            )
        else:
            for name, optimizer in optimizers.items():
                if anchored_value_migration is not None and name == "critic":
                    # The saved critic AdamW moments are positionally keyed
                    # over param groups that include the OLD grid's head
                    # parameters; loading them against the rebuilt head would
                    # misalign shapes. Trunk-matrix Muon state (critic_muon)
                    # is grid-agnostic and loads normally.
                    print(
                        "anchored-value migration: fresh critic AdamW state "
                        "(saved moments belong to the old value head)",
                        flush=True,
                    )
                    continue
                optimizer.load_state_dict(payload["optimizers"][name])
        # The sigma head is learned: the model load above restored it, and
        # (unlike the old fixed-buffer scheme) the CLI must NOT reassert it on
        # resume — --thought-log-sigma-init is an initialization, not a
        # schedule.  Learning rates remain the documented cross-run knobs and
        # are reasserted below.
        reassert_learning_rates(optimizers, args)
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
        source_execution_schema = payload.get("execution_schema")
        if args.migrate_compact_replay_resume:
            actor_init_provenance = dict(actor_init_provenance or {})
            actor_init_provenance["compact_replay_resume_migration"] = {
                "source_replay_numerics_schema": payload.get(
                    "replay_numerics_schema"
                ),
                "target_replay_numerics_schema": REPLAY_NUMERICS_SCHEMA,
            }
        if source_execution_schema == ZERO_AFFINE_EXECUTION_SCHEMA:
            actor_init_provenance = dict(actor_init_provenance or {})
            actor_init_provenance["identity_affine_execution_relabel"] = {
                "source_execution_schema": source_execution_schema,
                "target_execution_schema": target_execution_schema,
                "adapter_state": "preserved",
            }
        if source_execution_schema == PREVIOUS_EXECUTION_SCHEMA:
            actor_init_provenance = dict(actor_init_provenance or {})
            actor_init_provenance["reverse_kl_resume_migration"] = {
                "source_execution_schema": PREVIOUS_EXECUTION_SCHEMA,
                "thought_reverse_kl_coef": args.thought_reverse_kl_coef,
            }
        if source_execution_schema in {
            PREVIOUS_EXECUTION_SCHEMA,
            PERFORMANCE_COMPATIBLE_EXECUTION_SCHEMA,
        }:
            actor_init_provenance = dict(actor_init_provenance or {})
            actor_init_provenance["performance_resume_migration"] = {
                "source_execution_schema": source_execution_schema,
                "target_execution_schema": target_execution_schema,
            }
        if anchored_value_migration is not None:
            actor_init_provenance = dict(actor_init_provenance or {})
            actor_init_provenance["anchored_value_resume_migration"] = {
                "source_execution_schema": source_execution_schema,
                "target_execution_schema": target_execution_schema,
                **anchored_value_migration,
            }
        if args.migrate_projected_thought_resume:
            actor_init_provenance = dict(actor_init_provenance or {})
            actor_init_provenance["projected_thought_resume_migration"] = {
                "source_execution_schema": source_execution_schema,
                "target_execution_schema": target_execution_schema,
                "thought_trust_epsilon": args.thought_trust_epsilon,
                "thought_projection_penalty_coef": (
                    args.thought_projection_penalty_coef
                ),
            }
        if args.migrate_thought_reverse_kl_resume:
            actor_init_provenance = dict(actor_init_provenance or {})
            actor_init_provenance["thought_reverse_kl_resume_migration"] = {
                "source_execution_schema": source_execution_schema,
                "target_execution_schema": target_execution_schema,
                "thought_reverse_kl_coef": args.thought_reverse_kl_coef,
                "thought_clip_mode": args.thought_clip_mode,
            }

    if args.actor_critic_init:
        torch.set_rng_state(actor_init_payload["cpu_rng"])
        torch.cuda.set_rng_state_all(actor_init_payload["cuda_rng"])
        random.setstate(actor_init_payload["python_rng"])

    if args.bpb_only or args.bench_only:
        planned_prompt_count = 0
    elif args.rollout_only:
        planned_prompt_count = (
            args.prompts_per_rollout * args.rollout_only_repeats
        )
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
        if args.consume_all_prompts:
            raise ValueError(
                "one-pass run would cross the dataset's first epoch and "
                "reuse prompts: cursor "
                f"{sampler.cursor} + planned {planned_prompt_count} > "
                f"{len(math_rows)} rows"
            )
        final_epoch = (sampler.cursor + planned_prompt_count - 1) // len(
            math_rows
        )
        print(
            f"prompt plan spans {final_epoch + 1} epochs of the "
            f"{len(math_rows)}-row dataset ({planned_prompt_count} prompts); "
            "epochs after the first reshuffle deterministically from --seed",
            flush=True,
        )

    # Built before the first torch.compile so compile accounting starts from
    # an empty metrics deque, and so every compiled artifact below can be
    # wrapped in a call counter. When --profile is absent this is the
    # do-nothing profiler and every profiling call site below is inert.
    profiler = (
        RunProfiler(Path(args.output), device, args)
        if args.profile
        else DisabledProfiler()
    )

    # Every compiled graph in this process reaches a fp32 head region —
    # FreshThoughtMeanHead, StateDependentLogSigmaHead, or
    # SeparateCritic.value_logits — so all of them carry _enter_autocast
    # nodes, and AOTAutogradCache refuses to key on an unrecognized
    # call_function target. It therefore BYPASSED on every process start
    # (measured: "Bypassing autograd cache due to: Unsupported call_function
    # target _enter_autocast", 8 bypasses in an 8-step smoke). Inductor's FX
    # graph cache was already hitting, which is why cold compilation was
    # ~93% AOTAutograd and only 0.28 s of codegen per graph. Declaring the
    # two autocast markers cacheable lets the AOT artifact persist across
    # processes: warm pool-1 replay refresh 18.1 s -> 11.0 s, per-shard
    # steady state unchanged, and the one-off single-row shard
    # specialization 7.2 s -> 1.7 s. Sound because the autocast arguments
    # are constants baked into the graph nodes and hashed with them, and
    # because bf16 is the only ambient autocast dtype here — the ambient
    # DTYPE is not part of the cache key, only whether autocast is on.
    # Declared before the first torch.compile and outside every flag, since
    # this value is serialized into the on-disk key of EVERY graph: gating
    # it would make an artifact's key depend on an unrelated flag. Bump the
    # version string to invalidate every cached artifact.
    for autocast_marker in (
        "torch.amp.autocast_mode._enter_autocast",
        "torch.amp.autocast_mode._exit_autocast",
    ):
        torch._inductor.config.unsafe_marked_cacheable_functions[
            autocast_marker
        ] = "v1"

    # See the --duck-shape help. Set before the first compile because it
    # decides how the FIRST trace allocates symbols, which is the only trace
    # that matters here.
    torch.fx.experimental._config.use_duck_shape = args.duck_shape
    # Note for whoever meets a recompile-limit failure here: cache_size_limit
    # (raised to 64 below) is not the only ceiling. exceeds_recompile_limit
    # checks accumulated_recompile_limit (default 256) FIRST, and then again
    # as a backstop on the frame's compile id. Under fullgraph=True either
    # raises rather than falling back to eager, so it would kill a long run
    # mid-flight. Deliberately NOT raised: compute_cache_size walks the live
    # entry list, so invalidated entries do not accumulate there.
    #
    # The frame that gets close is step_core, and only under
    # --rollout-flex-decode: eval's dynamic artifact, the tail graph, and the
    # dynamic=False rollout artifact all compile that one code object, and the
    # last of them specializes per (bucketed row count, kv width). That is why
    # decode_width_grid below caps the width axis at two -- at ~9 row counts
    # it keeps the total near 20 rather than the ~36 a block-fine width grid
    # would produce. Without the flag the measured maximum for any frame is 3.

    # Rollout and evaluation use the identical dynamic narrow-prefix step.
    # Compile it once: separate wrappers paid the same large cold compilation
    # cost at step-0 eval and again at the first training collect. If eval ever
    # latches an eager fallback, the eval closure below also disables this
    # shared artifact for training before it can be called again.
    compiled_generation_step = None
    if (
        args.eval_compile
        or (
            args.rollout_compile
            and args.rollout_scheduler == "lockstep"
        )
    ):
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, 64
        )
        compiled_generation_step = profiler.register_artifact(
            "generation_step",
            torch.compile(
                wrapper.step_core,
                mode="max-autotune-no-cudagraphs",
                fullgraph=True,
                dynamic=True,
            ),
        )
    rollout_step_core = (
        compiled_generation_step
        if args.rollout_compile and args.rollout_scheduler == "lockstep"
        else None
    )
    if rollout_step_core is not None and args.rollout_flex_decode:
        # Flex decoding needs its OWN artifact: the shared one above is
        # dynamic=True for eval's arbitrary batches, and under symbolic shapes
        # Inductor has no flex decode choice to pick at all (measured -- the
        # lowering fails outright with "no choices exist for backend", and a
        # dynamic batch alone is enough to do it). The rollout can afford
        # dynamic=False because it now compacts to bucketed row counts, so
        # the row count takes a bounded number of values and the KV width is
        # pinned by the mask. Separate from the paged artifact for the same
        # reason that one exists.
        # NOT max-autotune: each bucket specialization compiles lazily, the
        # first time that row count appears, which is mid-rollout with the
        # full KV cache set already resident. Exhaustive benchmarking there
        # allocates candidate workspaces on top of it and spikes the peak --
        # measured at 24.68 GiB against a 15.0 GiB steady state on the same
        # config, which is what OOMs the production shape.
        rollout_step_core = profiler.register_artifact(
            "flex_generation_step",
            torch.compile(
                wrapper.step_core,
                fullgraph=True,
                dynamic=False,
                # Under --rollout-graph-decode the loop holds one row count and
                # one KV width for the whole run, which is the precondition
                # capture needs: a single recording, replayed every step. The
                # caches are marked static and the mask's closed-over buffers
                # are refilled in place, so no address the recording captured
                # ever moves.
                **(
                    {"mode": "reduce-overhead"}
                    if args.rollout_graph_decode
                    else {}
                ),
            ),
        )
    eval_step_core = compiled_generation_step if args.eval_compile else None
    rollout_paged_step_core = None
    if args.rollout_scheduler == "continuous_refill":
        rollout_paged_step_core = profiler.register_artifact(
            "rollout_paged_step",
            torch.compile(
                wrapper.paged_step_core,
                mode="max-autotune-no-cudagraphs",
                fullgraph=True,
                # FlexDecoding requires a concrete batch dimension. The
                # scheduler supplies one full-capacity main bucket plus
                # power-of-two tail buckets, avoiding a specialization for
                # every possible survivor count.
                dynamic=False,
            ),
        )

    # Static-tail CUDA graph for the fixed-size compacted training tail.
    # The caches are allocated ONCE and live for the whole run: a fresh
    # allocation would move the static addresses and force a graph
    # re-record. The artifact wraps the ORIGINAL eager step_core (collect's
    # temporary rollout_step_core patch never reaches it), and it only ever
    # runs under the training autocast at one shape, so it stays a single
    # cudagraph specialization.
    # Flex decoding wants a block-aligned KV width. The boolean tail mask does
    # not, and rounding for it would only widen the range it scores, so the
    # alignment follows the flag that needs it.
    decode_block_size = DecodeRangeMask.DEFAULT_BLOCK_SIZE
    rollout_cache_width = args.prompt_tokens + max_stream_steps
    if args.rollout_flex_decode:
        rollout_cache_width = (
            -(-rollout_cache_width // decode_block_size) * decode_block_size
        )
    rollout_tail_caches = None
    rollout_tail_step_core = None
    if args.rollout_tail_graph:
        rollout_tail_caches = wrapper.make_static_generation_cache(
            args.rollout_tail_batch,
            rollout_cache_width,
            device,
            dtype=torch.bfloat16,
        )
        rollout_tail_step_core = profiler.register_artifact(
            "rollout_tail_step",
            torch.compile(
                wrapper.step_core,
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=False,
            ),
        )

    # Flex decoding. A boolean attn_mask disqualifies every fused SDPA backend,
    # and the v25 kernel profile put the resulting memory-efficient kernel at
    # 32% of pool device time with 12672 of its 12684 calls coming from this
    # step. The target is the lockstep MAIN loop, not the tail: at
    # --rollout-groups 32 the direct call count in NOTES.md:1285 is 2080 main
    # vs 32 tail, so a tail-only change reaches 1.52% of the decode iterations
    # and the tail does not even engage until 96.9% of rows have ended.
    # Inductor lowers flex decoding for fully static shapes alone, so the main
    # loop rounds its live row count up to a bucket multiple and gives filler
    # rows an EMPTY key range -- measured to return exactly 0, finite, reading
    # no KV -- which makes the padding free.
    #
    # Masks are owned by the caller, not rebuilt inside the rollout, because
    # they hold persistent buffers whose addresses a cudagraph would capture.
    # (They do NOT cost extra compiles: rebuilding per step was measured at
    # zero additional specializations, so the mask_mod closure is not guarded
    # on by identity.)
    rollout_tail_decode_mask = None
    if args.rollout_flex_decode and not is_nano:
        # validate_args cannot see the checkpoint. The other accepted family is
        # PoPE, whose _attention_step scores a complex inner product over a
        # k_real/k_imag pair rather than one QK product and raises on a block
        # mask -- at the first tail switch, which is many pools in. Fail here.
        raise ValueError(
            f"--rollout-flex-decode needs a single-QK decode step; "
            f"architecture {backbone.architecture!r} does not have one"
        )
    # The KV width is what makes flex decoding expensive in memory. The
    # boolean path narrows the cache to the written prefix every step; the flex
    # step reads the whole allocated width, so the width is fixed at
    # allocation. Pinning it to the args upper bound (prompt_tokens +
    # max_stream_steps = 2560) instead of the pool's actual padded prompt costs
    # ~2.2 GiB at the production shape, which is more than a 32 GB card has
    # spare once two rollout groups' cache sets briefly coexist -- measured as
    # an OOM at make_generation_cache with 27.09 GiB already allocated. So the
    # mask is built per chunk width instead, and a pool of short prompts skips
    # the widest cache entirely.
    #
    # The grid is deliberately much coarser than the KV block size. Every
    # distinct (row count, kv width) pair is its own dynamic=False
    # specialization of step_core, and eval's artifact and the tail graph
    # compile the SAME code object, so all three share one cache_size_limit.
    # Rounding to the 128-wide block would give four widths against the ~9
    # bucketed row counts -- 36 entries against a limit of 64, and an overflow
    # under fullgraph=True RAISES rather than falling back to eager, killing a
    # long run mid-flight. Half the prompt bound caps the width axis at two
    # while keeping the adaptivity that pays.
    decode_width_grid = max(
        decode_block_size,
        -(-args.prompt_tokens // (2 * decode_block_size)) * decode_block_size,
    )
    rollout_decode_masks: dict[int, DecodeRangeMask] = {}
    rollout_graph_rows = max(args.rollout_groups, 1) * args.samples_per_prompt

    def decode_mask_for(prompt_width: int) -> DecodeRangeMask | None:
        """Main-loop mask sized for a chunk left-padded to ``prompt_width``."""
        if not args.rollout_flex_decode or rollout_step_core is None:
            return None
        if args.rollout_graph_decode:
            # One recording for the run means one width for the run, so the
            # per-chunk adaptivity below is not available: the mask must match
            # the static cache exactly.
            width = rollout_cache_width
        else:
            width = prompt_width + max_stream_steps
            width = -(-width // decode_width_grid) * decode_width_grid
        mask = rollout_decode_masks.get(width)
        if mask is None:
            mask = DecodeRangeMask(
                # --rollout-groups 0 selects the sequential single-prompt
                # branch, which still rolls out samples_per_prompt rows.
                rollout_graph_rows,
                width,
                device,
                block_size=decode_block_size,
            )
            rollout_decode_masks[width] = mask
        return mask

    # The captured main loop's arena. Allocated ONCE for the whole run, at the
    # full row count and the full KV width, because a re-allocation moves the
    # addresses the recording captured and forces a re-record. It is also the
    # reason the per-chunk cache churn disappears under this flag: the old path
    # allocated a fresh cache per chunk and relied on the cyclic collector to
    # reclaim the previous one before the next expanded.
    #
    # That permanence is also its cost: unlike the per-chunk cache it replaces,
    # it stays resident through the update, so it is charged against peak VRAM
    # rather than overlapping it. ``planned_prompt_count`` is zero exactly when
    # no rollout will run (--bpb-only, --bench-only), and those modes must not
    # pay multiple GiB for an arena they never touch.
    rollout_graph_caches = None
    if args.rollout_graph_decode and planned_prompt_count > 0:
        rollout_graph_caches = wrapper.make_static_generation_cache(
            rollout_graph_rows,
            rollout_cache_width,
            device,
            dtype=torch.bfloat16,
        )

    if args.rollout_flex_decode and rollout_tail_caches is not None:
        rollout_tail_decode_mask = DecodeRangeMask(
            args.rollout_tail_batch,
            rollout_cache_width,
            device,
            block_size=decode_block_size,
        )

    # Preserve the eager functions for infrequent no-grad diagnostics. Calling
    # a compiled training artifact under no-grad would create a distinct
    # AOTAutograd specialization solely because grad mode is a Dynamo guard.
    # Both the replay head and the readout tail get rebound below, so both
    # need capturing here.
    diagnostic_replay_head_inputs = replay_head_inputs
    diagnostic_emit_token_logprobs = compact_emit_token_logprobs

    # Replay compilation is independent of rollout. Dynamic B/L plus bounded
    # 64-token buckets lets one artifact cover the length-aware shard plan;
    # no CUDA graph owns outputs that remain live through eager losses/backward.
    # Measured, not assumed: one artifact each covers 23 distinct (B, L) shard
    # shapes and every unseen shape costs 3-26 ms, so the bucket exists to keep
    # the packed batches compact, NOT to limit compilations. The only shape
    # that can still recompile is a single-row shard — the framework's 0/1
    # specialization, guard "2 <= batch.token_ids.size()[0]" — which the
    # planner emits whenever the row remainder is one. That is one extra
    # compile per process and torch offers no way to avoid it here:
    # mark_unbacked makes Inductor's constant folder raise
    # GuardOnDataDependentSymNode on the row dimension.
    trim_multiple = args.replay_bucket if args.compile_replay else 1
    if args.compile_replay:
        torch._dynamo.config.cache_size_limit = 64
        critic.value_logits = profiler.register_artifact(
            "critic_value_logits",
            torch.compile(
                critic.value_logits,
                # Length buckets still span many GEMM shapes. Max-autotune
                # repeatedly stalls training to benchmark each new regime;
                # default Inductor dispatches them to stable cuBLAS kernels.
                mode="default",
                fullgraph=True,
                dynamic=True,
            ),
        )
        compiled_replay = profiler.register_artifact(
            "replay_head_inputs",
            torch.compile(
                replay_head_inputs,
                mode="default",
                fullgraph=True,
                dynamic=True,
            ),
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
        # The vocabulary readout tail, for the same reason and by the same
        # rebinding. Eagerly it is the readout GEMM followed by ~7 separate
        # passes over a (slots, 50257) fp32 tensor — softcap pow/add/rsqrt/
        # mul/mul, then log-softmax — to produce one scalar per slot. Fused,
        # the vocabulary axis stays in registers and only the (slots,) result
        # is written. Both consumers must land on THIS object: two artifacts
        # would be free to pick different reduction orders, and refresh minus
        # update is precisely the age-0 canary.
        compiled_emit_logprobs = profiler.register_artifact(
            "compact_emit_token_logprobs",
            torch.compile(
                compact_emit_token_logprobs,
                mode="default",
                fullgraph=True,
                dynamic=True,
            ),
        )
        globals()["compact_emit_token_logprobs"] = compiled_emit_logprobs
        postraining.latent_rollout.compact_emit_token_logprobs = (
            compiled_emit_logprobs
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source_provenance = capture_source_provenance(
        output / "provenance", Path(__file__).resolve().parents[1]
    )
    args.source_provenance = source_provenance
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
                # Top level, not buried in args: a profiled run's timings are
                # perturbed by the profiler and must not be quoted as this
                # configuration's cost.
                "profiled": bool(args.profile),
                "profile_schema": PROFILE_SCHEMA if args.profile else None,
                "source_provenance": source_provenance,
                "execution_schema": execution_schema_for_adapter(
                    args.thought_adapter,
                    args.thought_action_transform,
                    args.rollout_scheduler,
                ),
                "actor_objective_schema": ACTOR_OBJECTIVE_SCHEMA,
                "replay_numerics_schema": REPLAY_NUMERICS_SCHEMA,
                "prompt_order_schema": PROMPT_ORDER_SCHEMA,
                "math_data_identity": data_identity,
                "reward_schema": REWARD_SCHEMA,
                "reasoning_mode": args.reasoning_mode,
                "context_tokens": context_tokens,
                "math_modal_answer_baseline": math_modal_baseline,
                "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
                "rollout_policy_schema": rollout_policy_schema,
                "thought_input_schema": wrapper.thought_input_schema,
                "thought_action_transform_schema": (
                    wrapper.thought_action_transform_schema
                ),
                "thought_distribution_schema": THOUGHT_DISTRIBUTION_SCHEMA,
                "thought_mean_schema": THOUGHT_MEAN_SCHEMA,
                "thought_sigma_init_schema": wrapper.sigma_state_init_schema,
                "optimizer_schema": optimizer_schema_for_trunk_optimizer(
                    args.trunk_optimizer
                ),
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
                    "value_anchored_support": args.value_anchored_support,
                    "value_margin_bins": args.value_margin_bins,
                    "value_num_bins_total": value_num_bins,
                    "value_v_min": value_v_min,
                    "value_v_max": value_v_max,
                    "value_sigma_ratio": args.value_sigma_ratio,
                    "value_prior": args.value_prior,
                    "adapter_init_schema": critic.adapter_init_schema,
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
            solution_prefix_ids=answer_prefix_ids,
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
                slot_budget=args.replay_slot_budget,
            )
        return batch

    last_decode_schedule_metrics: dict[str, float] | None = None
    rollout_paged_cache = None

    def _collect(
        refresh_statistics: bool,
        prompt_count: int,
        offload_to_cpu: bool,
        scoring_pool: ThreadPoolExecutor | None,
    ) -> list[LatentRolloutBatch]:
        """One rollout: a scored prompt group per sampled DAPO problem."""
        nonlocal last_decode_schedule_metrics, rollout_paged_cache
        last_decode_schedule_metrics = None
        pool_start_cursor = sampler.cursor
        rollout_rows = sampler.next_rows(prompt_count)
        prompt_budget = args.prompt_tokens - len(answer_prefix_ids)
        with profiler.phase("prompt_encode"):
            encoded_rows = [
                (
                    row,
                    encode_prompt(tokenizer, prompt_text(row), prompt_budget)
                    + list(answer_prefix_ids),
                )
                for row in rollout_rows
            ]
        # The sequential sampler has already consumed these rows. Stable
        # length sorting only reduces left-padding inside this pool; it does
        # not alter dataset coverage, reuse, or use RNG to order data.
        encoded_rows.sort(key=lambda item: len(item[1]))
        groups = []
        cpu = torch.device("cpu")

        def retain_group(
            batch: LatentRolloutBatch,
            row: dict,
        ) -> LatentRolloutBatch:
            return finish_group(batch, row, refresh_statistics)

        if args.rollout_groups <= 1:
            for row, encoded in encoded_rows:
                prompt_ids = torch.tensor(
                    encoded,
                    dtype=torch.long, device=device,
                )
                with profiler.phase("decode"):
                    batch = rollout_continuations(
                        wrapper, prompt_ids[None],
                        train_max_new_tokens, max_stream_steps,
                        args.temperature, args.top_p,
                        stop_ids=stop_ids or None,
                        pin_emit=pin_emit,
                        record_likelihoods=False,
                        cache_dtype=torch.bfloat16,
                        sync_every=args.rollout_sync_every,
                        compact_dead_ratio=args.rollout_compact_dead_ratio,
                        tensor_positions=rollout_step_core is not None,
                        compact_finished=(
                            rollout_step_core is None
                            or args.rollout_tail_batch > 0
                        ),
                        finished_batch_size=(
                            args.rollout_tail_batch
                            if rollout_step_core is not None
                            and args.rollout_tail_batch > 0
                            else None
                        ),
                        prompt_repeats=args.samples_per_prompt,
                        caches=(
                            rollout_graph_caches
                            if rollout_step_core is not None
                            else None
                        ),
                        tail_caches=rollout_tail_caches,
                        tail_step_core=rollout_tail_step_core,
                        decode_mask=decode_mask_for(len(encoded)),
                        tail_decode_mask=rollout_tail_decode_mask,
                    )
                if offload_to_cpu:
                    with profiler.phase("stream_d2h"):
                        batch = compact_stream_to_device(batch, cpu)
                with profiler.phase("score_inline"):
                    groups.append(retain_group(batch, row))
                del batch
            return groups
        # Left-padded batched rollout: all chunk groups step together, so
        # each launch carries chunk*samples rows instead of samples — the
        # sequential per-group loop is launch-bound, not compute-bound.
        samples = args.samples_per_prompt
        pending_scored: list = []

        def drain_scored() -> None:
            # Each future carries one chunk's worth of retained groups, in
            # chunk order. Timed as a phase because this is where a scoring
            # worker that has fallen behind the GPU shows up: the wait is
            # host time inside collect that no rollout kernel covers.
            with profiler.phase("score_wait"):
                for future in pending_scored:
                    groups.extend(future.result())
                pending_scored.clear()

        def split_and_retain_chunk(batched, lengths, chunk_rows) -> list:
            """Split one host-resident chunk into groups and retain each.

            Runs on the scoring worker so the whole CPU tail of a chunk
            overlaps the next chunk's rollout instead of stalling the GPU.
            """
            # Timed as a worker span, not a phase: this runs concurrently
            # with the next chunk's decode, and decode is launch-bound, so
            # worker CPU held here is time the host cannot spend launching.
            with profiler.worker_span("score"):
                split_groups = split_rollout_groups(batched, samples, lengths)
                retained = []
                for index, (group, row) in enumerate(
                    zip(split_groups, chunk_rows, strict=True)
                ):
                    retained.append(retain_group(group, row))
                    # Release each source as soon as it is retained so the
                    # chunk's cloned storage does not stay live to the end.
                    split_groups[index] = None
                    del group
            return retained

        def complete_chunk(
            batched: LatentRolloutBatch,
            prompt_lengths_cpu: torch.Tensor,
            chunk: list[dict],
            *,
            already_offloaded: bool = False,
        ) -> None:
            if offload_to_cpu and not already_offloaded:
                # One chunk-level D2H transfer, then all variable-length
                # splitting, trimming, decoding, and scoring stay on CPU.
                with profiler.phase("stream_d2h"):
                    batched = compact_stream_to_device(batched, cpu)
            expanded_prompt_lengths = (
                prompt_lengths_cpu.repeat_interleave(samples)
            )
            if not offload_to_cpu:
                expanded_prompt_lengths = expanded_prompt_lengths.to(device)
            if scoring_pool is None:
                with profiler.phase("score_inline"):
                    split_groups = split_rollout_groups(
                        batched, samples, expanded_prompt_lengths
                    )
                    del batched
                    for group_index, (group, row) in enumerate(
                        zip(split_groups, chunk, strict=True)
                    ):
                        groups.append(retain_group(group, row))
                        split_groups[group_index] = None
                        del group
                    del split_groups
                return

            # Draining the previous chunk before submitting this one preserves
            # chunk order and bounds host storage at one pending chunk.
            drain_scored()
            pending_scored.append(
                scoring_pool.submit(
                    split_and_retain_chunk,
                    batched,
                    expanded_prompt_lengths,
                    chunk,
                )
            )

        encoded_chunks = [
            encoded_rows[start : start + args.rollout_groups]
            for start in range(0, len(encoded_rows), args.rollout_groups)
        ]

        def upload_chunk(
            encoded_chunk: list[tuple[dict, list[int]]],
            width: int,
        ) -> tuple[list[dict], torch.Tensor, torch.Tensor]:
            chunk = [row for row, _ in encoded_chunk]
            encoded = [ids for _, ids in encoded_chunk]
            prompt_ids = torch.zeros(
                (len(chunk), width), dtype=torch.long, device=device
            )
            for index, ids in enumerate(encoded):
                prompt_ids[index, width - len(ids):] = torch.tensor(
                    ids, dtype=torch.long, device=device
                )
            prompt_lengths_cpu = torch.tensor(
                [len(ids) for ids in encoded], dtype=torch.long
            )
            return chunk, prompt_ids, prompt_lengths_cpu

        if args.rollout_scheduler == "continuous_refill":
            # One physical lane pool remains full while later prompt groups
            # are pending. Every admitted request keeps its own logical
            # position, request-local RNG stream, and paged live KV prefix.
            # A common prompt width retains the ordinary rollout's left-pad
            # position convention. One pool-wide prefill bank traverses each
            # unique prompt once; refill only fans its cached KV/state into
            # sample lanes.
            common_width = max(len(ids) for _, ids in encoded_rows)
            with profiler.phase("prompt_upload"):
                uploaded = [
                    upload_chunk(encoded_chunk, common_width)
                    for encoded_chunk in encoded_chunks
                ]
            schedule_stats = ContinuousScheduleStats()
            if rollout_paged_cache is None:
                # Static-batch FlexDecoding specializations retain their
                # first cache inputs. Own that arena explicitly and reuse it
                # across pools instead of attempting another multi-GiB
                # allocation. Stale suffixes are unreachable through each
                # request's live-page mask; every admitted prefix and future
                # write overwrites the reachable locations.
                rollout_paged_cache = wrapper.make_paged_generation_cache(
                    args.rollout_groups * samples,
                    args.prompt_tokens + max_stream_steps,
                    device,
                    dtype=torch.bfloat16,
                )
            with profiler.phase("decode"):
                scheduled_batches = rollout_continuous_refill_groups(
                    wrapper,
                    [item[1] for item in uploaded],
                    [item[2] for item in uploaded],
                    prompt_repeats=samples,
                    capacity_rows=args.rollout_groups * samples,
                    max_new_tokens=train_max_new_tokens,
                    max_stream_steps=max_stream_steps,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    # The sampler cursor is monotonic and checkpointed. Mixed
                    # with the user seed it gives each pool a reproducible,
                    # resume-stable request-key namespace.
                    seed=(args.seed << 32) ^ pool_start_cursor,
                    stop_ids=stop_ids or None,
                    cache_dtype=torch.bfloat16,
                    pin_emit=pin_emit,
                    offload_device=cpu if offload_to_cpu else device,
                    schedule_stats=schedule_stats,
                    paged_cache=rollout_paged_cache,
                )
            last_decode_schedule_metrics = schedule_stats.metrics()
            for batch_index, (chunk, _, prompt_lengths_cpu) in enumerate(
                uploaded
            ):
                batched = scheduled_batches[batch_index]
                complete_chunk(
                    batched,
                    prompt_lengths_cpu,
                    chunk,
                    already_offloaded=offload_to_cpu,
                )
                # complete_chunk's scoring future owns the batch until it has
                # split and retained its groups. Drop the scheduler list's
                # second reference now, otherwise every dense source chunk
                # survives alongside all split pool groups.
                scheduled_batches[batch_index] = None
                del batched
            drain_scored()
            return groups

        for encoded_chunk in encoded_chunks:
            with profiler.phase("prompt_upload"):
                chunk_width = max(len(ids) for _, ids in encoded_chunk)
                chunk, prompt_ids, prompt_lengths_cpu = upload_chunk(
                    encoded_chunk, chunk_width
                )
                prompt_lengths = prompt_lengths_cpu.to(device)
            with profiler.phase("decode"):
                batched = rollout_continuations(
                    wrapper, prompt_ids, train_max_new_tokens, max_stream_steps,
                    args.temperature, args.top_p, stop_ids=stop_ids or None,
                    prompt_lengths=prompt_lengths,
                    pin_emit=pin_emit,
                    record_likelihoods=False,
                    cache_dtype=torch.bfloat16,
                    sync_every=args.rollout_sync_every,
                    compact_dead_ratio=args.rollout_compact_dead_ratio,
                    tensor_positions=rollout_step_core is not None,
                    compact_finished=(
                        rollout_step_core is None
                        or args.rollout_tail_batch > 0
                    ),
                    finished_batch_size=(
                        args.rollout_tail_batch
                        if rollout_step_core is not None
                        and args.rollout_tail_batch > 0
                        else None
                    ),
                    prompt_repeats=samples,
                    caches=(
                        rollout_graph_caches
                        if rollout_step_core is not None
                        else None
                    ),
                    tail_caches=rollout_tail_caches,
                    tail_step_core=rollout_tail_step_core,
                    decode_mask=decode_mask_for(chunk_width),
                    tail_decode_mask=rollout_tail_decode_mask,
                )
            complete_chunk(batched, prompt_lengths_cpu, chunk)
            del batched
        drain_scored()
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
        original_paged_step_core = wrapper.paged_step_core
        if rollout_step_core is not None:
            wrapper.step_core = rollout_step_core
        if rollout_paged_step_core is not None:
            wrapper.paged_step_core = rollout_paged_step_core
        # Offloaded chunks are host tensors after compact_stream_to_device's
        # barrier, so one worker thread can trim and score chunk N (the
        # tokenizer's decode releases the GIL) while the launch-bound rollout
        # loop feeds chunk N+1 to the GPU. On-device chunks stay sequential:
        # worker tensor ops would enqueue D2H syncs into the rollout stream.
        scoring_pool = (
            ThreadPoolExecutor(max_workers=1) if offload_to_cpu else None
        )
        try:
            # rollout_continuations itself is no-grad, while the optional
            # refresh below must remain grad-enabled so refresh/update share
            # one compiled replay specialization and identical PPO numerics.
            with profiler.phase("collect"), training_autocast():
                return _collect(
                    refresh_statistics,
                    args.prompts_per_rollout
                    if prompt_count is None
                    else prompt_count,
                    offload_to_cpu,
                    scoring_pool,
                )
        finally:
            if scoring_pool is not None:
                scoring_pool.shutdown(wait=True)
            wrapper.step_core = original_step_core
            wrapper.paged_step_core = original_paged_step_core

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
            wrapper, tokenizer, aime_rows, args.aime_samples,
            aime_max_new_tokens,
            aime_stream_steps, args.aime_chunk, args.seed, device,
            prompt_tokens=args.prompt_tokens,
            batch_trajectories=args.eval_batch_trajectories,
            compiled_step_core=eval_step_core,
            compiled_tail_batch=args.eval_tail_batch or None,
            answer_style_override="aime",
            captured_attempts=captured_attempts,
            pin_emit=pin_emit,
            prompt_suffix_ids=answer_prefix_ids,
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
            "aime/policy_accuracy", metrics["policy_accuracy"], step
        )
        tensorboard.add_scalar(
            "aime/continue_thinking_fraction",
            metrics["continue_thinking_fraction"],
            step,
        )
        print(
            f"step:{step} "
            f"aime_policy_avg@{metrics['policy_samples']}:"
            f"{metrics['policy_accuracy']:.4f}",
            flush=True,
        )

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
            bench_max_new_tokens, bench_stream_steps, args.bench_samples,
            eval_seed, device, prompt_tokens=args.prompt_tokens,
            batch_trajectories=args.eval_batch_trajectories,
            compiled_step_core=eval_step_core,
            compiled_tail_batch=args.eval_tail_batch or None,
            captured_attempts=captured_attempts,
            pin_emit=pin_emit,
            prompt_suffix_ids=answer_prefix_ids,
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
            "bench/policy_accuracy", metrics["policy_accuracy"], step
        )
        tensorboard.add_scalar(
            "bench/continue_thinking_fraction",
            metrics["continue_thinking_fraction"],
            step,
        )
        print(
            f"step:{step} "
            f"bench_policy_avg@{metrics['policy_samples']}:"
            f"{metrics['policy_accuracy']:.4f}",
            flush=True,
        )

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
        all_passed = True
        for repeat in range(args.rollout_only_repeats):
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            groups = collect(
                refresh_statistics=False,
                offload_to_cpu=True,
            )
            torch.cuda.synchronize()
            # collect(refresh_statistics=False) above: no old_values exist.
            metrics = aggregate_diagnostics(
                groups,
                args.samples_per_prompt,
                stop_ids,
                refreshed_statistics=False,
            )
            if last_decode_schedule_metrics is None:
                metrics.update(
                    lockstep_decode_metrics(
                        groups, max(args.rollout_groups, 1)
                    )
                )
            else:
                metrics.update(last_decode_schedule_metrics)
            metrics["collect_seconds"] = time.perf_counter() - started
            metrics["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
            metrics["useful_actions_per_second"] = (
                metrics["trajectories"]
                * metrics["actions_per_trajectory"]
                / metrics["collect_seconds"]
            )
            passed = (
                metrics["within_group_reward_std"]
                >= args.gate_min_within_group_reward_std
            )
            all_passed &= passed
            logger.log(
                type="rollout_gate",
                repeat=repeat,
                passed=passed,
                **metrics,
            )
            print(
                json.dumps(
                    {
                        "type": "rollout_gate",
                        "repeat": repeat,
                        "passed": bool(passed),
                        **metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del groups
        tensorboard.close()
        raise SystemExit(0 if all_passed else 2)

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
            zero_optimizers(optimizers, "critic")
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
                    replay_slot_budget=args.replay_slot_budget,
                )
                for group in groups
            ]
            step_optimizers(optimizers, "critic")
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
            zero_optimizers(optimizers, "critic")
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
        # Before the clock starts: attaching the profiler's own machinery
        # would otherwise land inside this pool's wall time and inside its
        # unaccounted remainder, making traced and untraced pools
        # incomparable on the one row that matters.
        profiler.pool_started(step)
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
        with profiler.phase("rollout_diagnostics"):
            rollout_metrics = aggregate_diagnostics(
                groups, args.samples_per_prompt, stop_ids
            )
            if last_decode_schedule_metrics is None:
                rollout_metrics.update(
                    lockstep_decode_metrics(groups, max(args.rollout_groups, 1))
                )
            else:
                rollout_metrics.update(last_decode_schedule_metrics)
            minibatch_orders = optimizer_minibatch_orders(
                len(groups),
                args.prompts_per_minibatch,
                allow_partial_final=(
                    args.consume_all_prompts
                    and sampler.cursor == len(math_rows)
                ),
            )
        if len(minibatch_orders) != pool_updates:
            raise RuntimeError("rollout pool did not produce the planned updates")

        # Assemble each shuffled optimizer minibatch on CPU with RIGHT tail
        # padding, then refresh all old statistics under the still-frozen
        # behavior policy. The refreshed DEVICE batch is retained and consumed
        # directly by the matching update below — repacking and re-uploading
        # the identical order and width, plus the d2h stat scatter, was pure
        # round-trip overhead. Retention is bounded: past the budget (streams
        # lengthen when thinking grows) the pre-retention path takes over —
        # scatter stats to the CPU groups and let update repack.
        retained_device_batches: dict[int, LatentRolloutBatch] = {}
        device_replay_plans: dict[int, ReplayPlan] = {}
        replay_layout_tokens = {
            tuple(minibatch_order): object()
            for minibatch_order in minibatch_orders
        }
        retained_bytes_total = 0
        packed_batch_bytes_max = 0
        packed_real_slot_counts = []
        packed_capacity_slots = 0
        pool_cpu_pack_seconds = 0.0
        # Transfer and replay are asynchronous, so these are event pairs read
        # after the single barrier at the end of collection. Wall clocks here
        # used to need a synchronize per minibatch, which drained the pipeline
        # eight times a pool purely to read a clock.
        h2d_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        refresh_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        d2h_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        old_value_sums = []
        old_value_counts = []
        def pack_minibatch(
            minibatch_order: list[int],
        ) -> tuple[
            LatentRolloutBatch,
            ReplayPlan,
            list[LatentRolloutBatch],
            float,
        ]:
            selected = [groups[index] for index in minibatch_order]
            pack_started = time.perf_counter()
            with profiler.worker_span("pack"):
                packed = pack_rollout_groups_for_replay(
                    selected, pin_memory=True
                )
                packed.replay_layout_token = replay_layout_tokens[
                    tuple(minibatch_order)
                ]
                replay_plan = build_replay_plan(
                    packed,
                    args.replay_max_trajectories,
                    args.replay_attention_budget,
                    args.replay_bucket,
                    args.replay_slot_budget,
                )
            return (
                packed,
                replay_plan,
                selected,
                time.perf_counter() - pack_started,
            )

        # Minibatch orders are a disjoint partition of the pool, so a single
        # worker can pack minibatch N+1's pinned host batch while N's H2D
        # and refresh occupy the GPU — the worker only reads groups this
        # loop's scatter fallback never writes in the same iteration.
        pack_pool = ThreadPoolExecutor(max_workers=1)
        try:
            with profiler.phase("refresh_pipeline"):
                pack_future = pack_pool.submit(
                    pack_minibatch, minibatch_orders[0]
                )
                for age, minibatch_order in enumerate(minibatch_orders):
                    with profiler.phase("pack_wait"):
                        (
                            cpu_batch,
                            cpu_replay_plan,
                            selected_groups,
                            pack_elapsed,
                        ) = pack_future.result()
                    pool_cpu_pack_seconds += pack_elapsed
                    if age + 1 < len(minibatch_orders):
                        pack_future = pack_pool.submit(
                            pack_minibatch, minibatch_orders[age + 1]
                        )
                    with profiler.phase("h2d"), device_timed(h2d_events):
                        device_batch = cpu_batch.to(device, non_blocking=True)
                        device_replay_plan = cpu_replay_plan.to(
                            device, non_blocking=True
                        )
                    # Safe without a barrier: the source is pinned, so it came
                    # from the caching host allocator, which records a CUDA
                    # event when the block is freed and withholds it from
                    # reuse until the copy retires.
                    del cpu_batch
                    del cpu_replay_plan
                    device_replay_plans[age] = device_replay_plan
                    with (
                        profiler.phase("refresh"),
                        device_timed(refresh_events),
                        training_autocast(),
                    ):
                        refresh_old_statistics(
                            wrapper,
                            critic,
                            device_batch,
                            max_trajectories=args.replay_max_trajectories,
                            attention_budget=args.replay_attention_budget,
                            bucket_multiple=args.replay_bucket,
                            slot_budget=args.replay_slot_budget,
                            replay_plan=device_replay_plan,
                        )
                    packed_batch_bytes = sum(
                        value.numel() * value.element_size()
                        for field in fields(device_batch)
                        if isinstance(
                            (value := getattr(device_batch, field.name)),
                            torch.Tensor,
                        )
                    )
                    packed_batch_bytes_max = max(
                        packed_batch_bytes_max, packed_batch_bytes
                    )
                    # Reduced on device and read once after the pool's single
                    # barrier; int() here was a blocking sync per minibatch.
                    packed_real_slot_counts.append(
                        (device_batch.kind != PAD_SLOT).sum()
                    )
                    packed_capacity_slots += device_batch.kind.numel()
                    generated = device_batch.action_mask.bool()
                    # masked_fill, not boolean indexing: the latter lowers to
                    # masked_select, whose output size is data dependent, so
                    # it blocks the host on every minibatch. masked_fill also
                    # keeps a non-finite PAD slot out of the sum, which
                    # multiplying by the mask would not.
                    old_value_sums.append(
                        device_batch.old_values.masked_fill(
                            ~generated, 0.0
                        ).sum()
                    )
                    old_value_counts.append(generated.sum())
                    if (
                        retained_bytes_total + packed_batch_bytes
                        <= RETAINED_MINIBATCH_BUDGET_BYTES
                    ):
                        retained_device_batches[age] = device_batch
                        retained_bytes_total += packed_batch_bytes
                    else:
                        with (
                            profiler.phase("stat_scatter_d2h"),
                            device_timed(d2h_events),
                        ):
                            scatter_replay_statistics(
                                device_batch, selected_groups
                            )
                    del device_batch
                    del device_replay_plan
                    del selected_groups
        finally:
            pack_pool.shutdown(wait=True)
        # A phase of its own because this is where every kernel the pool
        # enqueued and has not retired actually drains. Folded into the
        # remainder it would be the largest anonymous entry in the table.
        with profiler.phase("pool_barrier"):
            # The only barrier this loop takes on its own account. Every
            # device reduction and event pair below is read after it, so no
            # per-minibatch metric drains the pipeline. Whether refresh
            # itself still blocks inside the shard planner is a separate
            # question, and one --profile answers rather than assumes.
            torch.cuda.synchronize()
        collect_seconds = time.perf_counter() - collect_started
        with profiler.phase("pool_reductions"):
            old_value_sum = float(torch.stack(old_value_sums).sum())
            old_value_count = int(torch.stack(old_value_counts).sum())
            rollout_metrics["old_value_mean"] = (
                old_value_sum / old_value_count if old_value_count else 0.0
            )
            rollout_metrics["retained_minibatches"] = len(
                retained_device_batches
            )
            rollout_metrics["packed_batch_gib_max"] = (
                packed_batch_bytes_max / 2**30
            )
            packed_real_slots = int(torch.stack(packed_real_slot_counts).sum())
            rollout_metrics["packed_padding_utilization"] = (
                packed_real_slots / packed_capacity_slots
                if packed_capacity_slots
                else 0.0
            )
            rollout_metrics["pool_cpu_pack_seconds"] = pool_cpu_pack_seconds
            # Device time on the transfer and replay streams, not host wall
            # time.
            rollout_metrics["pool_h2d_seconds"] = elapsed_seconds(h2d_events)
            rollout_metrics["pool_refresh_seconds"] = elapsed_seconds(
                refresh_events
            )
            rollout_metrics["pool_d2h_seconds"] = elapsed_seconds(d2h_events)
        rollout_peak_vram_bytes = torch.cuda.max_memory_allocated()
        # Restart the peak counter here so the train rows below report the
        # UPDATE-phase peak (packed minibatch + replay-shard activations),
        # not the collection-phase KV-cache peak that always dominates it.
        # Replay memory knobs (--replay-attention-budget and retained device
        # batches) are tuned against this number.
        torch.cuda.reset_peak_memory_stats()
        rollout_step = step + pool_updates
        with profiler.phase("pool_logging"):
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
                "perf/pool_cpu_pack_seconds",
                pool_cpu_pack_seconds,
                rollout_step,
            )
            for tag in (
                "packed_batch_gib_max",
                "packed_padding_utilization",
                "decode_steps_per_chunk_mean",
                "decode_step_utilization",
                "decode_step_savings_fraction",
                "pool_h2d_seconds",
                "pool_refresh_seconds",
                "pool_d2h_seconds",
            ):
                if tag in rollout_metrics:
                    tensorboard.add_scalar(
                        f"perf/{tag}", rollout_metrics[tag], rollout_step
                    )
            rollout_dashboard = rollout_tensorboard_metrics(rollout_metrics)
            # Horizontal anti-collapse calibration: on DAPO, a terminated
            # constant modal answer already earns nontrivial exact and shaped
            # reward. On-policy improvement is meaningful only relative to
            # both.
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
            with profiler.phase("update"):
                update_started = time.perf_counter()
                next_step = step + 1
                zero_optimizers(optimizers, "actor")
                zero_optimizers(optimizers, "critic")
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
                retained_minibatch = retained_device_batches.pop(
                    behavior_age, None
                )
                device_replay_plan = device_replay_plans.pop(behavior_age)
                minibatch_h2d_events = []
                if retained_minibatch is not None:
                    # Refreshed on-device in the pool loop above; identical to
                    # what repacking selected_groups would rebuild.
                    device_minibatch = retained_minibatch
                    minibatch_pack_seconds = 0.0
                else:
                    with profiler.phase("minibatch_pack"):
                        minibatch_pack_started = time.perf_counter()
                        cpu_minibatch = pack_rollout_groups_for_replay(
                            selected_groups, pin_memory=True
                        )
                        cpu_minibatch.replay_layout_token = (
                            replay_layout_tokens[tuple(minibatch_order)]
                        )
                        minibatch_pack_seconds = (
                            time.perf_counter() - minibatch_pack_started
                        )
                    with (
                        profiler.phase("minibatch_h2d"),
                        device_timed(minibatch_h2d_events),
                    ):
                        device_minibatch = cpu_minibatch.to(
                            device, non_blocking=True
                        )
                    # See the pool-loop transfer: freeing pinned host storage
                    # with the copy in flight is safe, the caching host
                    # allocator defers reuse behind a recorded event.
                    del cpu_minibatch
                policy_action_denominator = actor_minibatch_action_denominator(
                    [device_minibatch], [0]
                )
                with profiler.phase("forward_backward"):
                    metrics = training_update(
                        wrapper, critic, device_minibatch, optimizers,
                        thought_pg_coef=args.thought_pg_coef,
                        thought_reverse_kl_coef=args.thought_reverse_kl_coef,
                        thought_clip_mode=args.thought_clip_mode,
                        thought_trust_epsilon=args.thought_trust_epsilon,
                        thought_projection_penalty_coef=(
                            args.thought_projection_penalty_coef
                        ),
                        gate_entropy_coef=args.gate_entropy_coef,
                        gate_pg_coef=1.0,
                        actor_step=False,
                        critic_step=False,
                        policy_action_denominator=policy_action_denominator,
                        value_action_denominator=policy_action_denominator,
                        gae_lambda_alpha=args.gae_lambda_alpha,
                        replay_max_trajectories=args.replay_max_trajectories,
                        replay_attention_budget=args.replay_attention_budget,
                        replay_bucket=args.replay_bucket,
                        replay_slot_budget=args.replay_slot_budget,
                        replay_plan=device_replay_plan,
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
                # Reading the loss and gradient scalars is where the host
                # first waits on the backward, so this phase carries the
                # whole minibatch's device time that forward_backward only
                # enqueued.
                with profiler.phase("grad_telemetry"):
                    actor_dashboard = aggregate_actor_tensorboard_metrics(
                        minibatch_metrics
                    )
                    nonfinite_gradients = {
                        name: actor_dashboard[name]
                        for name in (
                            "grad/trunk", "grad/gate", "grad/adapter",
                            "grad/renderer", "grad/sigma",
                            "grad/thought_mean", "grad/critic",
                        )
                        if not math.isfinite(actor_dashboard[name])
                    }
                if nonfinite_gradients:
                    raise RuntimeError(
                        "non-finite gradients before optimizer step: "
                        f"{nonfinite_gradients}"
                    )
                with profiler.phase("optimizer_step"):
                    step_optimizers(optimizers, "actor")
                    step_optimizers(optimizers, "critic")
                    # Gradient buffers have already been reduced to scalar
                    # telemetry. Release them before the optional second
                    # replay so this read-only diagnostic cannot stack an
                    # inference forward on top of the training step's peak
                    # allocation.
                    zero_optimizers(optimizers, "actor")
                    zero_optimizers(optimizers, "critic")
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
                    elapsed_seconds(minibatch_h2d_events)
                )
                actor_dashboard["perf/minibatch_cpu_pack_seconds"] = (
                    minibatch_pack_seconds
                )
                if next_step == 1 or (
                    args.post_update_kl_every > 0
                    and next_step % args.post_update_kl_every == 0
                ):
                    drift_started = time.perf_counter()
                    with profiler.phase("post_update_drift"), training_autocast():
                        post_update_drift = measure_post_update_policy_drift(
                            wrapper,
                            [device_minibatch],
                            replay_max_trajectories=args.replay_max_trajectories,
                            replay_attention_budget=args.replay_attention_budget,
                            replay_bucket=args.replay_bucket,
                            replay_slot_budget=args.replay_slot_budget,
                            replay_function=diagnostic_replay_head_inputs,
                            emit_logprob_function=diagnostic_emit_token_logprobs,
                            replay_plans=[device_replay_plan],
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
                del device_replay_plan
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
        # Freeing a pool's host storage is tens of thousands of refcount
        # drops and allocator returns, so it gets a name rather than
        # inflating the remainder.
        with profiler.phase("pool_teardown"):
            zero_optimizers(optimizers, "actor")
            zero_optimizers(optimizers, "critic")
            del groups

        if crossed_interval(previous_step, step, args.bpb_every):
            with profiler.phase("bpb_eval"):
                eval_started = time.perf_counter()
                bpb = teacher_forced_bpb()
                eval_seconds = time.perf_counter() - eval_started
                logger.log(
                    type="bpb", step=step, val_bpb=bpb, seconds=eval_seconds
                )
                tensorboard.add_scalar("guard/val_bpb", bpb, step)
                tensorboard.add_scalar(
                    "perf/bpb_eval_seconds", eval_seconds, step
                )
        if aime_rows and crossed_interval(previous_step, step, args.aime_every):
            with profiler.phase("aime_eval"):
                aime_eval(step)
        if bench_rows and crossed_interval(previous_step, step, args.bench_every):
            with profiler.phase("bench_eval"):
                bench_eval(step)
        if crossed_interval(previous_step, step, args.save_every):
            with profiler.phase("checkpoint_save"):
                save_started = time.perf_counter()
                save_checkpoint(
                    output / "latent_vapo_checkpoint.pt", wrapper, critic,
                    optimizers, step, args, sampler, warmup_step,
                    actor_init_provenance,
                )
                save_seconds = time.perf_counter() - save_started
                logger.log(type="checkpoint", step=step, seconds=save_seconds)
                tensorboard.add_scalar(
                    "perf/checkpoint_seconds", save_seconds, step
                )
        profiler.pool_finished(step, time.perf_counter() - started)
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
    profiler.close()


if __name__ == "__main__":
    main()
