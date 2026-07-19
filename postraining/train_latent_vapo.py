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
from postraining.core import (
    JsonlLogger,
    POSTTRAIN_PROMPT_TOKENS,
    POSTTRAIN_RESPONSE_TOKENS,
    POSTTRAIN_STREAM_TOKENS,
    clipped_policy_loss,
    encode_prompt,
    generalized_advantage_estimate,
    length_adaptive_lambda,
    load_unique_math_rows,
    masked_token_mean,
    per_dim_clipped_policy_loss,
    positive_example_lm_loss,
    validate_posttraining_context_budget,
    verify_answer,
)
from postraining.latent_rollout import (
    THOUGHT_SLOT,
    LatentRolloutBatch,
    assign_terminal_rewards,
    emitted_token_rows,
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
    LatentThoughtModel,
    validate_renderer_checkpoint,
)
from postraining.model_io import fresh_trunk, load_model
from postraining.train_vapo import prompt_text
from postraining.value_model import SeparateCritic


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
    batch: LatentRolloutBatch, truth: str, tokenizer, stop_ids: tuple[int, ...]
) -> None:
    """Binary verifier rewards on stop-truncated decoded emissions."""
    scores = []
    stop_set = set(stop_ids)
    for emitted in emitted_token_rows(batch):
        cut = next((i for i, t in enumerate(emitted) if t in stop_set), None)
        if cut is not None:
            emitted = emitted[: cut + 1]
        correct, _ = verify_answer(tokenizer.decode(emitted), truth)
        scores.append(float(correct))
    assign_terminal_rewards(
        batch, torch.tensor(scores, dtype=torch.float32, device=batch.rewards.device)
    )


@torch.no_grad()
def evaluate_aime_latent(
    wrapper: LatentThoughtModel,
    tokenizer,
    rows: list[dict],
    samples: int,
    max_new_tokens: int,
    max_stream_steps: int,
    chunk: int,
    seed: int,
    device: torch.device,
    prompt_tokens: int,
    caches: list[tuple[torch.Tensor, ...]] | None = None,
) -> dict[str, float | int]:
    """AIME avg@k through the latent THINK/EMIT policy itself.

    Matches the VAPO paper's protocol (average pass rate over ``samples``
    generations at temperature 1.0 / top-p 0.7) but generation runs the
    gate-conditioned rollout, so the evaluated policy is exactly the trained
    one — including its latent thinking.  RNG state is saved and restored so
    the eval never perturbs training reproducibility.

    Prompts keep their TAIL ``prompt_tokens`` exactly like the DAPO training
    path: the question and answer-format instruction sit at the end, and the
    truncation keeps prompt + stream budget inside the training context
    (PoPE extrapolates poorly beyond it).
    """
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state()
    python_state = random.getstate()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    stop_ids = tuple(
        t for t in (tokenizer.eos_id(), tokenizer.bos_id()) if t >= 0
    )
    correct = 0
    total = 0
    think_actions = 0.0
    actions = 0.0
    try:
        for row in rows:
            prompt_ids = torch.tensor(
                encode_prompt(tokenizer, prompt_text(row), prompt_tokens),
                dtype=torch.long,
                device=device,
            )[None]
            truth = row["reward_model"]["ground_truth"]
            for start in range(0, samples, chunk):
                width = min(chunk, samples - start)
                # A remainder chunk narrower than the preallocated caches is
                # PADDED up to their batch width and the extra rows discarded:
                # routing a dynamic-shape fallback through the compiled
                # reduce-overhead step would recompile per position and can
                # LRU-evict the training loop's own graphs (review finding).
                rollout_width = caches[0][0].size(0) if caches is not None else width
                batch = trim_stream(
                    rollout_continuations(
                        wrapper, prompt_ids.expand(rollout_width, -1),
                        max_new_tokens, max_stream_steps, 1.0, 0.7,
                        stop_ids=stop_ids or None,
                        caches=caches,
                    )
                )
                think_actions += float(
                    (
                        (batch.gate_actions[:width] == 0).float()
                        * batch.action_mask[:width]
                    ).sum()
                )
                actions += float(batch.action_mask[:width].sum())
                stop_set = set(stop_ids)
                for emitted in emitted_token_rows(batch)[:width]:
                    cut = next(
                        (i for i, t in enumerate(emitted) if t in stop_set), None
                    )
                    if cut is not None:
                        emitted = emitted[: cut + 1]
                    is_correct, _ = verify_answer(tokenizer.decode(emitted), truth)
                    correct += int(is_correct)
                    total += 1
    finally:
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state)
        random.setstate(python_state)
    return {
        "accuracy": correct / max(total, 1),
        "samples": total,
        "think_fraction": think_actions / max(actions, 1.0),
    }


def think_run_lengths(kind: torch.Tensor) -> torch.Tensor:
    """Lengths of every consecutive-THOUGHT run in a (batch, stream) kind map.

    Runs are per-row (a row never opens with a thought — prompts are token
    slots), so flattening start/end indices row-major keeps them paired.
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
    actions = batch.action_mask.sum().clamp_min(1)
    stop_set = set(stop_ids)
    # Fraction of rows that terminated themselves (emitted BOS or EOS)
    # rather than exhausting the token/stream budget.
    ended = [
        float(any(token in stop_set for token in row))
        for row in emitted_token_rows(batch)
    ]
    generated = batch.action_mask.bool()
    think_fraction = float(
        ((batch.gate_actions == 0).float() * batch.action_mask).sum() / actions
    )
    runs = think_run_lengths(batch.kind)
    grouped = batch.reward_scalar.reshape(-1, samples_per_prompt)
    # Did thinking pay off THIS group?  Within-group Pearson correlation
    # between per-trajectory think counts and rewards (0.0 when either side
    # has no variance — undefined groups dilute the aggregate toward zero,
    # which is the honest prior for "no evidence either way").
    think_counts = (
        (batch.gate_actions == THINK).float() * batch.action_mask
    ).sum(1)
    thinkers = think_counts > 0
    centered_thinks = think_counts - think_counts.mean()
    centered_rewards = batch.reward_scalar - batch.reward_scalar.mean()
    scale = centered_thinks.square().mean().sqrt() * centered_rewards.square().mean().sqrt()
    think_reward_correlation = (
        float((centered_thinks * centered_rewards).mean() / scale)
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
        "think_run_mean": float(runs.mean()) if runs.numel() else 0.0,
        "think_run_std": float(runs.std(unbiased=False)) if runs.numel() else 0.0,
        "think_runs_per_trajectory": runs.numel() / batch.reward_scalar.numel(),
        "emits_per_trajectory": float(batch.emit_mask.sum(1).mean()),
        "actions_per_trajectory": float(batch.action_mask.sum(1).mean()),
        "old_value_mean": float(batch.old_values[generated].mean()) if generated.any() else 0.0,
        "thinking_trajectory_fraction": float(thinkers.float().mean()),
        "reward_mean_thinking": (
            float(batch.reward_scalar[thinkers].mean()) if bool(thinkers.any()) else 0.0
        ),
        "reward_mean_pure_emit": (
            float(batch.reward_scalar[~thinkers].mean())
            if bool((~thinkers).any())
            else 0.0
        ),
        "think_reward_correlation": think_reward_correlation,
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
    return aggregated


def gradient_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum())
    return total**0.5


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
    """
    backbone = wrapper.backbone
    optimizers["critic"].zero_grad(set_to_none=True)
    if actor_step and "actor" in optimizers:
        optimizers["actor"].zero_grad(set_to_none=True)

    # The replay pass is the differentiable forward.  Gate and renderer losses
    # backprop through beliefs into the trunk. The projector runs densely in
    # replay, but only the THINK-masked thought-content loss consumes it, so
    # token and gate losses cannot train the prediction projector.
    # value_only warmup skips it entirely (the critic reads the raw batch).
    if not value_only:
        beliefs, predicted, features, token_targets = replay_head_inputs(wrapper, batch)

    action_counts = batch.action_mask.sum(1)
    # length_adaptive_lambda floors VAPO's alpha*l credit horizon at
    # min(l, 1/alpha): raw alpha=0.05 hit lambda = 0 (TD(0)) at this run's
    # 12-23-action trajectories, structurally starving THINK decisions of
    # direct reward credit (audited: the think-fraction ratchet).
    lambdas = (
        torch.ones_like(action_counts)
        if value_only
        else length_adaptive_lambda(action_counts, gae_lambda_alpha)
    )
    advantages, _ = generalized_advantage_estimate(
        batch.rewards, batch.old_values, batch.action_mask, lambdas
    )
    _, value_targets = generalized_advantage_estimate(
        batch.rewards, batch.old_values, batch.action_mask,
        torch.ones_like(action_counts),
    )
    advantages = advantages.detach()
    value_targets = value_targets.detach()

    # Separate critic, HL-Gauss (cleanrl v215): softmax-CE against the
    # Gaussian-smoothed projection of the scalar targets, no value clipping.
    value_logits = critic.value_logits(batch)
    value_ce = critic.support.cross_entropy(value_logits, value_targets)
    value_loss = masked_token_mean(value_ce, batch.action_mask)
    with torch.no_grad():
        values = critic.support.to_expected_scalar(value_logits)
    metrics: dict[str, float] = {
        "value_loss": float(value_loss.detach()),
        "value_mean": float(masked_token_mean(values, batch.action_mask)),
        "value_target_mean": float(masked_token_mean(value_targets, batch.action_mask)),
    }
    with torch.no_grad():
        # HL-Gauss CE is floored at the smoothed target's own entropy (~1.4-2.1
        # nats depending on boundary truncation); the excess over that floor is
        # the true fit gap (a KL), which converges to 0 while value_loss
        # plateaus at the floor.
        target_probs = critic.support.project(value_targets)
        target_entropy = -(target_probs * target_probs.clamp_min(1e-20).log()).sum(-1)
        metrics["value_excess_ce"] = float(
            value_loss - masked_token_mean(target_entropy, batch.action_mask)
        )
    with torch.no_grad():
        # "Did the credit assignment favor thinking?" — mean GAE advantage
        # conditioned on the gate action actually taken.  A persistently
        # negative think_advantage_mean is the optimizer voting against the
        # THINK branch; 0.0 with think_action_count 0 means no evidence.
        think_actions = (batch.gate_actions == THINK).float() * batch.action_mask
        emit_actions = (batch.gate_actions == EMIT).float() * batch.action_mask
        metrics["think_advantage_mean"] = float(
            masked_token_mean(advantages, think_actions)
        )
        metrics["emit_advantage_mean"] = float(
            masked_token_mean(advantages, emit_actions)
        )
        metrics["think_action_count"] = float(think_actions.sum())
    if value_only:
        if not torch.isfinite(value_loss):
            raise RuntimeError(f"non-finite value loss: {metrics}")
        value_loss.backward()
        metrics["critic_grad_norm"] = gradient_norm(critic.parameters())
        optimizers["critic"].step()
        return metrics

    gate_mask = batch.action_mask
    new_gate_logprobs = wrapper.gate.log_prob(batch.gate_actions.float(), beliefs)
    gate_loss, gate_clip = clipped_policy_loss(
        new_gate_logprobs, batch.old_gate_logprobs, advantages, gate_mask
    )
    with torch.no_grad():
        # Diagnostic only — there is no entropy bonus in the objective.
        gate_entropy = masked_token_mean(wrapper.gate.entropy(beliefs), gate_mask)

    logits = backbone.logits_from_features(features)
    new_token_logprobs = (
        logits.float()
        .log_softmax(-1)
        .gather(-1, token_targets[..., None])
        .squeeze(-1)
    )
    renderer_loss, renderer_clip = clipped_policy_loss(
        new_token_logprobs, batch.old_token_logprobs, advantages, batch.emit_mask
    )

    # VAPO modification #6: NLL on positive trajectories' emitted tokens,
    # via the same helper the token trainer uses.  The paper flags positives
    # by verifier correctness; Tier 0 has no binary verifier, so a reward
    # threshold stands in until Tier 2 rewards land.
    positive = batch.reward_scalar >= positive_reward_threshold
    positive_lm = positive_example_lm_loss(
        new_token_logprobs, batch.emit_mask, positive
    )

    # Thought-content policy gradient (user prescription, Jul 18): the
    # thought decided at gate position p (stored at p+1, the token-target
    # shift) is a continuous action trained by per-dim clipped PPO against
    # the critic's THINK-position advantages.  The gradient flows through
    # projected thought mean into the entire trunk — this is the term that retrains
    # the world model toward what the next latent SHOULD be.
    think_mask = (batch.gate_actions == THINK) & batch.action_mask.bool()
    if thought_pg_coef != 0.0 and bool(think_mask.any()):
        thought_means, thought_targets, _ = select_thought_actions(
            batch, predicted
        )
        new_thought_logprobs = wrapper.transition.per_dim_log_prob(
            thought_targets, thought_means
        )
        compact_advantages = advantages[think_mask]
        thought_loss, thought_clip = per_dim_clipped_policy_loss(
            new_thought_logprobs,
            batch.old_thought_logprobs[think_mask],
            compact_advantages,
            torch.ones_like(compact_advantages),
        )
    else:
        # A differentiable empty selection would materialize zero-valued
        # projector grads. Adam would then keep applying stale momentum even
        # though this update contains no thought objective. Detached zeros
        # preserve a true ``grad is None`` contract for the projector.
        thought_loss = predicted.detach().new_zeros(())
        thought_clip = predicted.detach().new_zeros(())

    actor_total = (
        gate_pg_coef * gate_loss
        + renderer_loss
        + positive_lm_weight * positive_lm
        + thought_pg_coef * thought_loss
    )
    if not torch.isfinite(actor_total + value_loss):
        raise RuntimeError(
            "non-finite loss before optimizer step: "
            f"gate={float(gate_loss)} renderer={float(renderer_loss)} "
            f"positive_lm={float(positive_lm)} thought={float(thought_loss)} "
            f"value={float(value_loss)}"
        )
    (loss_scale * actor_total).backward()
    value_loss.backward()
    # Norms are cumulative across an accumulation epoch; the last minibatch
    # of an epoch reports the full pre-step norm.  Probes are registered
    # under blocks[-1], so exclude them from the trunk norm by identity.
    probe_parameter_ids = {
        id(parameter)
        for probe in (backbone.policy_probe, backbone.critic_probe)
        for parameter in probe.parameters()
    }
    grad_norms = {
        "trunk_grad_norm": gradient_norm(
            parameter
            for parameter in backbone.parameters()
            if id(parameter) not in probe_parameter_ids
        ),
        "renderer_grad_norm": gradient_norm(backbone.policy_probe.parameters()),
        "adapter_grad_norm": gradient_norm(wrapper.adapter.parameters()),
        "critic_grad_norm": gradient_norm(critic.parameters()),
        "gate_grad_norm": gradient_norm(wrapper.gate.parameters()),
    }
    optimizers["critic"].step()
    if actor_step and "actor" in optimizers:
        optimizers["actor"].step()

    with torch.no_grad():
        emit_probability = masked_token_mean(
            wrapper.gate.emit_logit(beliefs).sigmoid(), batch.action_mask
        )
    metrics.update(
        gate_loss=float(gate_loss.detach()),
        gate_clip_fraction=float(gate_clip),
        gate_entropy=float(gate_entropy),
        emit_probability=float(emit_probability),
        renderer_loss=float(renderer_loss.detach()),
        renderer_clip_fraction=float(renderer_clip),
        positive_lm_loss=float(positive_lm.detach()),
        positive_fraction=float(positive.float().mean()),
        thought_loss=float(thought_loss.detach()),
        thought_clip_fraction=float(thought_clip),
        advantage_mean=float(masked_token_mean(advantages, batch.action_mask)),
        advantage_std=float(
            masked_token_mean(
                (advantages - masked_token_mean(advantages, batch.action_mask)).square(),
                batch.action_mask,
            )
            ** 0.5
        ),
        reward=float(batch.reward_scalar.mean()),
        **grad_norms,
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
) -> None:
    payload = {
        "step": step,
        "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
        "model": wrapper.state_dict(),
        "critic": critic.state_dict(),
        "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
        "args": vars(args),
        "sampler_cursor": sampler.cursor,
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(),
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
    parser.add_argument("--bpb-every", type=int, default=80)
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
    parser.add_argument("--save-every", type=int, default=50)
    # Pretraining's recipe (train_gpt.py: torch.compile(model, dynamic=False,
    # fullgraph=True)) with reduce-overhead on top: CUDA-graph the stepwise
    # rollout and the batched replay forwards of BOTH models.  Off by
    # default: measured 2.6x SLOWER than eager at this model size (the
    # static-cache step pays masked SDPA over the full cache every step),
    # and the batched left-padded rollout — the measured utilization win —
    # is eager-only.  Revisit if the model grows.
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=False
    )
    # Round replayed stream lengths up to this bucket so the compiled replay
    # functions see a bounded set of shapes (each shape is its own CUDA graph).
    parser.add_argument("--replay-bucket", type=int, default=64)
    # Prompt groups rolled out together as one left-padded batch (measured:
    # the sequential per-group rollout is launch-bound at ~140 W, so stepping
    # groups*samples rows per launch is the utilization lever).  Eager-only;
    # under --compile the static-cache path keeps the per-group rollout.
    parser.add_argument("--rollout-groups", type=int, default=4)
    parser.add_argument("--rollout-only", action="store_true")
    parser.add_argument("--gate-min-within-group-reward-std", type=float, default=0.01)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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
    sampler = MathPromptSampler(load_unique_math_rows(args.math_data), args.seed)
    seq_len = FreshHyperparameters.train_seq_len
    luts = baseline.build_sentencepiece_luts(tokenizer, FreshHyperparameters.vocab_size, device)
    val_tokens = baseline.load_validation_tokens(FreshHyperparameters.val_files, seq_len)

    start_step = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_renderer_checkpoint(payload, args.resume)
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
        torch.set_rng_state(payload["cpu_rng"])
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
        random.setstate(payload["python_rng"])
        sampler.cursor = int(payload["sampler_cursor"])

    # --- torch.compile wiring: pretraining's recipe (dynamic=False) plus
    # reduce-overhead, so the launch-bound stepwise rollout replays as one
    # CUDA graph per shape and the batched replay forwards of both models
    # are graphed per stream bucket.  Patching happens after any resume load
    # and never touches state_dicts, so checkpoints stay format-identical.
    trim_multiple = args.replay_bucket if args.compile else 1
    static_caches: dict[tuple[int, int], list[tuple[torch.Tensor, ...]]] = {}

    def rollout_caches(batch_size: int, stream_steps: int):
        """One persistent zeroed cache set per (batch, cache_length) shape.

        Persistence matters: the cache tensors are marked static for CUDA
        graphs, and a fresh allocation per rollout would force a graph
        re-record every time.
        """
        if not args.compile:
            return None
        key = (batch_size, args.prompt_tokens + stream_steps)
        if key not in static_caches:
            static_caches[key] = wrapper.make_static_generation_cache(
                batch_size, key[1], device
            )
        return static_caches[key]

    if args.compile:
        # Every replay bucket and rollout shape guards its own compiled
        # graph; the default cache limit (8) is too tight for that set.
        torch._dynamo.config.cache_size_limit = 64
        # step_core is the pure tensor surface (no dataclass, no cache
        # aliasing across the boundary) — the one place reduce-overhead's
        # CUDA graphs pay: ~6.5k sequential launch-bound calls per collect.
        wrapper.step_core = torch.compile(
            wrapper.step_core, mode="reduce-overhead", fullgraph=True, dynamic=False
        )
        # The batched replay/critic forwards get plain compile (fusion, no
        # CUDA graphs): their outputs stay live across the whole eager loss
        # region and its backward — the classic graph-output-overwritten
        # hazard (red-teamed) — and at 32 x ~400-token minibatches they are
        # compute-bound, so graphs would buy little.
        critic.value_logits = torch.compile(critic.value_logits, dynamic=False)
        compiled_replay = torch.compile(replay_head_inputs, dynamic=False)
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
    tensorboard = SummaryWriter(output / "tensorboard")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "phase": "latent_vapo_dapo",
                "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
                "args": vars(args),
                "base": {
                    "checkpoint": str(args.checkpoint),
                    "architecture": backbone.architecture,
                },
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

    def finish_group(batch: LatentRolloutBatch, row: dict) -> LatentRolloutBatch:
        batch = trim_stream(batch, multiple=trim_multiple)
        score_math_rollout(
            batch, row["reward_model"]["ground_truth"], tokenizer, stop_ids
        )
        # Stepwise rollout and parallel replay disagree numerically at
        # bf16 scale; recompute the stored PPO statistics through the
        # update-step replay path so epoch-0 ratios are exactly one.
        # This also fills old_values from the separate critic (the
        # rollout never values).
        refresh_old_statistics(wrapper, critic, batch)
        return batch

    def collect() -> list[LatentRolloutBatch]:
        """One rollout: a scored prompt group per sampled DAPO problem."""
        rollout_rows = sampler.next_rows(args.prompts_per_rollout)
        groups = []
        if args.compile or args.rollout_groups <= 1:
            for row in rollout_rows:
                prompt_ids = torch.tensor(
                    encode_prompt(tokenizer, prompt_text(row), args.prompt_tokens),
                    dtype=torch.long, device=device,
                )
                batch = rollout_continuations(
                    wrapper, prompt_ids[None].expand(args.samples_per_prompt, -1),
                    args.continuation_tokens, max_stream_steps,
                    args.temperature, args.top_p, stop_ids=stop_ids or None,
                    caches=rollout_caches(args.samples_per_prompt, max_stream_steps),
                )
                groups.append(finish_group(batch, row))
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
            )
            for group, row in zip(
                split_rollout_groups(batched, samples, prompt_lengths), chunk,
                strict=True,
            ):
                groups.append(finish_group(group, row))
        return groups

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
        metrics = evaluate_aime_latent(
            wrapper, tokenizer, aime_rows, args.aime_samples, args.aime_max_tokens,
            aime_stream_steps, args.aime_chunk, args.seed, device,
            prompt_tokens=args.prompt_tokens,
            caches=rollout_caches(
                min(args.aime_chunk, args.aime_samples), aime_stream_steps
            ),
        )
        logger.log(type="aime", step=step, **metrics)
        tensorboard.add_scalar("aime/accuracy", metrics["accuracy"], step)
        tensorboard.add_scalar("aime/think_fraction", metrics["think_fraction"], step)
        print(f"step:{step} aime_avg@{args.aime_samples}:{metrics['accuracy']:.4f}", flush=True)

    def bench_eval(step: int) -> None:
        wrapper.eval()
        metrics = evaluate_aime_latent(
            wrapper, tokenizer, bench_rows, args.bench_samples,
            args.bench_max_tokens, bench_stream_steps, args.bench_samples,
            args.seed, device, prompt_tokens=args.prompt_tokens,
            caches=rollout_caches(args.bench_samples, bench_stream_steps),
        )
        logger.log(type="bench", step=step, **metrics)
        tensorboard.add_scalar("bench/accuracy", metrics["accuracy"], step)
        tensorboard.add_scalar("bench/think_fraction", metrics["think_fraction"], step)
        print(f"step:{step} bench_avg@{args.bench_samples}:{metrics['accuracy']:.4f}", flush=True)

    if args.rollout_only:
        metrics = aggregate_diagnostics(collect(), args.samples_per_prompt, stop_ids)
        passed = metrics["within_group_reward_std"] >= args.gate_min_within_group_reward_std
        logger.log(type="rollout_gate", passed=passed, **metrics)
        print(json.dumps({"passed": bool(passed), **metrics}, sort_keys=True))
        tensorboard.close()
        raise SystemExit(0 if passed else 2)

    if start_step == 0:
        bpb = teacher_forced_bpb()
        logger.log(type="bpb", step=0, val_bpb=bpb)
        tensorboard.add_scalar("guard/val_bpb", bpb, 0)
        print(f"step:0 teacher-forced val_bpb:{bpb:.4f}", flush=True)
        if aime_rows:
            aime_eval(0)
        if bench_rows:
            bench_eval(0)
        for warmup in range(1, args.value_warmup_steps + 1):
            group_metrics = [
                update_minibatch(wrapper, critic, group, optimizers, value_only=True)
                for group in collect()
            ]
            metrics = {
                key: float(sum(m[key] for m in group_metrics) / len(group_metrics))
                for key in group_metrics[0]
            }
            logger.log(type="value_warmup", step=warmup, **metrics)
            for key, value in metrics.items():
                tensorboard.add_scalar(f"value_warmup/{key}", value, warmup)

    def crossed_interval(previous: int, current: int, interval: int) -> bool:
        return interval > 0 and current // interval > previous // interval

    step = start_step
    while step < args.steps:
        previous_step = step
        started = time.perf_counter()
        groups = collect()
        rollout_metrics = aggregate_diagnostics(
            groups, args.samples_per_prompt, stop_ids
        )
        logger.log(type="rollout", step=step, **rollout_metrics)
        for key, value in rollout_metrics.items():
            tensorboard.add_scalar(f"rollout/{key}", value, step)

        for epoch in range(args.ppo_epochs):
            # One accumulated actor step per epoch: stepping the full trunk
            # per minibatch would take ppo_epochs x len(groups) trust-region
            # steps against a single frozen behavior policy and saturate the
            # clip (red-teamed).  The critic has no trust region and still
            # steps per minibatch inside update_minibatch.
            optimizers["actor"].zero_grad(set_to_none=True)
            for index in torch.randperm(len(groups)).tolist():
                step += 1
                metrics = update_minibatch(
                    wrapper, critic, groups[index], optimizers,
                    positive_lm_weight=args.positive_lm_weight,
                    positive_reward_threshold=args.positive_reward_threshold,
                    thought_pg_coef=args.thought_pg_coef,
                    gate_pg_coef=0.0 if step <= args.gate_freeze_steps else 1.0,
                    actor_step=False,
                    loss_scale=1.0 / len(groups),
                    gae_lambda_alpha=args.gae_lambda_alpha,
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
                for key, value in metrics.items():
                    tensorboard.add_scalar(f"train/{key}", value, step)
            optimizers["actor"].step()

        if crossed_interval(previous_step, step, args.bpb_every):
            bpb = teacher_forced_bpb()
            logger.log(type="bpb", step=step, val_bpb=bpb)
            tensorboard.add_scalar("guard/val_bpb", bpb, step)
        if aime_rows and crossed_interval(previous_step, step, args.aime_every):
            aime_eval(step)
        if bench_rows and crossed_interval(previous_step, step, args.bench_every):
            bench_eval(step)
        if crossed_interval(previous_step, step, args.save_every):
            save_checkpoint(
                output / "latent_vapo_checkpoint.pt", wrapper, critic,
                optimizers, step, args, sampler,
            )
    save_checkpoint(
        output / "latent_vapo_checkpoint.pt", wrapper, critic,
        optimizers, step, args, sampler,
    )
    tensorboard.close()


if __name__ == "__main__":
    main()
