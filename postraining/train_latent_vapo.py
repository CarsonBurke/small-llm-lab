"""Latent-thought VAPO: PPO over THINK/EMIT gates and the token renderer.

D4-faithful separation of concerns on top of a frozen pretrained trunk:

- PPO trains the gate (Bernoulli THINK/EMIT) and the renderer (policy probe)
  from stored-stream rollouts.
- The critic is a SEPARATE from-scratch model (same architecture class,
  fresh weights, fully trainable, no SIGReg or latent prediction) trained
  purely by HL-Gauss cross-entropy on [0, 1] value targets.
- The transition head stays a predictive world model: its log-std trains by
  beta-NLL on grounded transitions (stream positions whose next input was a
  real token), never by policy gradient.
- The policy trunk, embeddings, projectors, and adapter do not train at all.

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
import os
import random
import time
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.tensorboard import SummaryWriter

import train_gpt as baseline
from fresh_lejepa_train import FreshHyperparameters
from postraining.core import (
    JsonlLogger,
    clipped_policy_loss,
    generalized_advantage_estimate,
    length_adaptive_lambda,
    load_unique_math_rows,
    masked_token_mean,
    positive_example_lm_loss,
    verify_answer,
)
from postraining.latent_rollout import (
    THOUGHT_SLOT,
    LatentRolloutBatch,
    assign_terminal_rewards,
    emitted_token_rows,
    grounded_transition_mask,
    refresh_old_statistics,
    replay_head_inputs,
    rollout_continuations,
    trim_stream,
)
from postraining.latent_thought import LatentThoughtModel
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
    batch: LatentRolloutBatch, truth: str, tokenizer, eos: int
) -> None:
    """Binary verifier rewards on EOS-truncated decoded emissions."""
    scores = []
    for emitted in emitted_token_rows(batch):
        if eos >= 0 and eos in emitted:
            emitted = emitted[: emitted.index(eos) + 1]
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
) -> dict[str, float | int]:
    """AIME avg@k through the latent THINK/EMIT policy itself.

    Matches the VAPO paper's protocol (average pass rate over ``samples``
    generations at temperature 1.0 / top-p 0.7) but generation runs the
    gate-conditioned rollout, so the evaluated policy is exactly the trained
    one — including its latent thinking.  RNG state is saved and restored so
    the eval never perturbs training reproducibility.
    """
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state()
    python_state = random.getstate()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    eos = tokenizer.eos_id()
    correct = 0
    total = 0
    think_actions = 0.0
    actions = 0.0
    try:
        for row in rows:
            prompt_ids = torch.tensor(
                tokenizer.encode(prompt_text(row)), dtype=torch.long, device=device
            )[None]
            truth = row["reward_model"]["ground_truth"]
            for start in range(0, samples, chunk):
                width = min(chunk, samples - start)
                batch = trim_stream(
                    rollout_continuations(
                        wrapper, prompt_ids.expand(width, -1), max_new_tokens,
                        max_stream_steps, 1.0, 0.7,
                        eos_id=eos if eos >= 0 else None,
                    )
                )
                think_actions += float(
                    ((batch.gate_actions == 0).float() * batch.action_mask).sum()
                )
                actions += float(batch.action_mask.sum())
                for emitted in emitted_token_rows(batch):
                    if eos >= 0 and eos in emitted:
                        emitted = emitted[: emitted.index(eos) + 1]
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
    batch: LatentRolloutBatch, samples_per_prompt: int
) -> dict[str, float | int]:
    actions = batch.action_mask.sum().clamp_min(1)
    generated = batch.action_mask.bool()
    think_fraction = float(
        ((batch.gate_actions == 0).float() * batch.action_mask).sum() / actions
    )
    runs = think_run_lengths(batch.kind)
    grouped = batch.reward_scalar.reshape(-1, samples_per_prompt)
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
    }


def aggregate_diagnostics(
    groups: list[LatentRolloutBatch], samples_per_prompt: int
) -> dict[str, float | int]:
    """Mean of per-group rollout diagnostics; trajectory counts are summed."""
    per_group = [rollout_diagnostics(group, samples_per_prompt) for group in groups]
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


def update_minibatch(
    wrapper: LatentThoughtModel,
    critic: SeparateCritic,
    batch: LatentRolloutBatch,
    optimizers: dict[str, torch.optim.Optimizer],
    gate_entropy_coef: float,
    beta: float,
    value_only: bool = False,
    positive_lm_weight: float = 0.0,
    positive_reward_threshold: float = 0.5,
) -> dict[str, float]:
    backbone = wrapper.backbone
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        # Trunk, embeddings, and adapter are frozen: the replay pass carries
        # no trainable path, so beliefs are plain data for the heads below.
        beliefs, predicted, features, token_targets = replay_head_inputs(wrapper, batch)

    action_counts = batch.action_mask.sum(1)
    lambdas = torch.ones_like(action_counts) if value_only else length_adaptive_lambda(action_counts)
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

    grounded = grounded_transition_mask(batch)
    latent_targets = wrapper.embed_tokens(token_targets)
    transition_nll = wrapper.transition.beta_nll(
        latent_targets, predicted, beliefs, beta=beta, weights=grounded
    )

    total = (
        gate_loss
        - gate_entropy_coef * gate_entropy
        + renderer_loss
        + positive_lm_weight * positive_lm
        + value_loss
        + transition_nll
    )
    if not torch.isfinite(total):
        raise RuntimeError(
            "non-finite loss before optimizer step: "
            f"gate={float(gate_loss)} renderer={float(renderer_loss)} "
            f"positive_lm={float(positive_lm)} "
            f"value={float(value_loss)} transition={float(transition_nll)}"
        )
    total.backward()
    grad_norms = {
        "transition_grad_norm": gradient_norm(wrapper.transition.parameters()),
        "renderer_grad_norm": gradient_norm(backbone.policy_probe.parameters()),
        "critic_grad_norm": gradient_norm(critic.parameters()),
        "gate_grad_norm": gradient_norm(wrapper.gate.parameters()),
    }
    for optimizer in optimizers.values():
        optimizer.step()

    with torch.no_grad():
        emit_probability = masked_token_mean(
            wrapper.gate.emit_logit(beliefs).sigmoid(), batch.action_mask
        )
        log_std = wrapper.transition.log_std(beliefs[batch.action_mask.bool()])
    metrics.update(
        gate_loss=float(gate_loss.detach()),
        gate_clip_fraction=float(gate_clip),
        gate_entropy=float(gate_entropy.detach()),
        emit_probability=float(emit_probability),
        renderer_loss=float(renderer_loss.detach()),
        renderer_clip_fraction=float(renderer_clip),
        positive_lm_loss=float(positive_lm.detach()),
        positive_fraction=float(positive.float().mean()),
        transition_nll=float(transition_nll.detach()),
        log_std_mean=float(log_std.mean()) if log_std.numel() else 0.0,
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
    parser.add_argument("--prompt-tokens", type=int, default=384)
    parser.add_argument("--continuation-tokens", type=int, default=128)
    parser.add_argument("--prompts-per-rollout", type=int, default=16)
    # Each prompt group is one PPO minibatch (prompts vary in length, so
    # groups are separate batches end to end).
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    # Total generated-slot budget per trajectory (thinks + emits).  Thinking
    # is never forcibly interrupted; overthinking costs emitted tokens and
    # therefore reward.  0 means the default of 4x the emit cap.
    parser.add_argument("--max-stream-steps", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gate-lr", type=float, default=1e-4)
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
    parser.add_argument("--transition-lr", type=float, default=1e-4)
    parser.add_argument("--gate-entropy-coef", type=float, default=1e-3)
    parser.add_argument("--beta-nll-beta", type=float, default=0.5)
    parser.add_argument("--positive-lm-weight", type=float, default=0.1)
    parser.add_argument("--positive-reward-threshold", type=float, default=0.5)
    # VAPO paper: 50 value-pretraining steps before policy updates.
    parser.add_argument("--value-warmup-steps", type=int, default=50)
    parser.add_argument("--bpb-every", type=int, default=80)
    parser.add_argument("--aime-every", type=int, default=80)
    parser.add_argument("--aime-data", default="postraining/data/aime-2024.parquet")
    parser.add_argument("--aime-samples", type=int, default=32)
    parser.add_argument("--aime-max-tokens", type=int, default=512)
    # Rollout positions are sequential, so batching all samples of a problem
    # into one rollout is nearly free parallelism; lower this only if VRAM
    # becomes the constraint.
    parser.add_argument("--aime-chunk", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=50)
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

    backbone = load_model(args.checkpoint, device)
    backbone.eval()
    # load_model freezes everything except the probes; that is exactly the
    # RL trainable surface plus the new heads enabled below.  The backbone's
    # critic probe is unused here (the critic is a separate model) — freeze it.
    wrapper = LatentThoughtModel(backbone).to(device)
    for parameter in wrapper.new_parameters():
        parameter.requires_grad_(False)
    for parameter in backbone.critic_probe.parameters():
        parameter.requires_grad_(False)
    for module in (wrapper.gate, wrapper.transition):
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    critic = SeparateCritic(
        fresh_trunk(backbone, device),
        num_bins=args.value_bins,
        sigma_ratio=args.value_sigma_ratio,
        prior_value=args.value_prior,
    ).to(device)
    critic.eval()  # no dropout in this architecture; keep norms deterministic

    optimizers = {
        "gate": torch.optim.AdamW(
            wrapper.gate.parameters(), lr=args.gate_lr, weight_decay=0.0
        ),
        "renderer": torch.optim.AdamW(
            backbone.policy_probe.parameters(), lr=args.renderer_lr,
            weight_decay=0.0, fused=True,
        ),
        "critic": torch.optim.AdamW(
            critic.parameters(), lr=args.critic_lr, weight_decay=0.0, fused=True,
        ),
        "transition": torch.optim.AdamW(
            wrapper.transition.parameters(), lr=args.transition_lr, weight_decay=0.0
        ),
    }

    tokenizer = spm.SentencePieceProcessor(model_file=FreshHyperparameters.tokenizer_path)
    eos = tokenizer.eos_id()
    aime_rows = (
        load_unique_math_rows(args.aime_data)
        if args.aime_every > 0 and not args.rollout_only
        else []
    )
    sampler = MathPromptSampler(load_unique_math_rows(args.math_data), args.seed)
    seq_len = FreshHyperparameters.train_seq_len
    luts = baseline.build_sentencepiece_luts(tokenizer, FreshHyperparameters.vocab_size, device)
    val_tokens = baseline.load_validation_tokens(FreshHyperparameters.val_files, seq_len)

    start_step = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        wrapper.load_state_dict(payload["model"], strict=True)
        critic.load_state_dict(payload["critic"], strict=True)
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][name])
        start_step = int(payload["step"])
        torch.set_rng_state(payload["cpu_rng"])
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
        random.setstate(payload["python_rng"])
        sampler.cursor = int(payload["sampler_cursor"])

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(output / "metrics.jsonl")
    tensorboard = SummaryWriter(output / "tensorboard")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "phase": "latent_vapo_dapo",
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

    def collect() -> list[LatentRolloutBatch]:
        """One rollout: a scored prompt group per sampled DAPO problem."""
        groups = []
        for row in sampler.next_rows(args.prompts_per_rollout):
            prompt_ids = torch.tensor(
                tokenizer.encode(prompt_text(row)), dtype=torch.long, device=device
            )[-args.prompt_tokens:]
            batch = rollout_continuations(
                wrapper, prompt_ids[None].expand(args.samples_per_prompt, -1),
                args.continuation_tokens, max_stream_steps,
                args.temperature, args.top_p, eos_id=eos,
            )
            batch = trim_stream(batch)
            score_math_rollout(batch, row["reward_model"]["ground_truth"], tokenizer, eos)
            # Stepwise rollout and parallel replay disagree numerically at
            # bf16 scale; recompute the stored PPO statistics through the
            # update-step replay path so epoch-0 ratios are exactly one.
            # This also fills old_values from the separate critic (the
            # rollout never values).
            refresh_old_statistics(wrapper, critic, batch)
            groups.append(batch)
        return groups

    def teacher_forced_bpb() -> float:
        """Pretraining-style val BPB (teacher-forced): the do-no-harm guard."""
        wrapper.eval()
        _, bpb = baseline.eval_val(
            FreshHyperparameters, backbone, 0, 1, device, 8, val_tokens, *luts
        )
        return bpb

    def aime_eval(step: int) -> None:
        wrapper.eval()
        metrics = evaluate_aime_latent(
            wrapper, tokenizer, aime_rows, args.aime_samples, args.aime_max_tokens,
            aime_stream_steps, args.aime_chunk, args.seed, device,
        )
        logger.log(type="aime", step=step, **metrics)
        tensorboard.add_scalar("aime/accuracy", metrics["accuracy"], step)
        tensorboard.add_scalar("aime/think_fraction", metrics["think_fraction"], step)
        print(f"step:{step} aime_avg@{args.aime_samples}:{metrics['accuracy']:.4f}", flush=True)

    if args.rollout_only:
        metrics = aggregate_diagnostics(collect(), args.samples_per_prompt)
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
        for warmup in range(1, args.value_warmup_steps + 1):
            group_metrics = [
                update_minibatch(
                    wrapper, critic, group, optimizers, args.gate_entropy_coef,
                    args.beta_nll_beta, value_only=True,
                )
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
        rollout_metrics = aggregate_diagnostics(groups, args.samples_per_prompt)
        logger.log(type="rollout", step=step, **rollout_metrics)
        for key, value in rollout_metrics.items():
            tensorboard.add_scalar(f"rollout/{key}", value, step)

        for _ in range(args.ppo_epochs):
            for index in torch.randperm(len(groups)).tolist():
                step += 1
                metrics = update_minibatch(
                    wrapper, critic, groups[index], optimizers,
                    args.gate_entropy_coef, args.beta_nll_beta,
                    positive_lm_weight=args.positive_lm_weight,
                    positive_reward_threshold=args.positive_reward_threshold,
                )
                logger.log(
                    type="train", step=step,
                    seconds=time.perf_counter() - started,
                    peak_vram_bytes=torch.cuda.max_memory_allocated(),
                    **metrics,
                )
                for key, value in metrics.items():
                    tensorboard.add_scalar(f"train/{key}", value, step)

        if crossed_interval(previous_step, step, args.bpb_every):
            bpb = teacher_forced_bpb()
            logger.log(type="bpb", step=step, val_bpb=bpb)
            tensorboard.add_scalar("guard/val_bpb", bpb, step)
        if aime_rows and crossed_interval(previous_step, step, args.aime_every):
            aime_eval(step)
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
