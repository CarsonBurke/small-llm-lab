"""Single-GPU VAPO post-training for the fresh LeJEPA checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.tensorboard import SummaryWriter
from checkpointing import (
    RecoveryCheckpointPolicy,
    atomic_link_or_copy,
    atomic_torch_save,
)

from postraining.core import (
    JsonlLogger,
    TrajectoryBatch,
    generalized_advantage_estimate,
    length_adaptive_lambda,
    load_unique_math_rows,
    top_p_sample,
    verify_answer,
)
from postraining.model_io import DEFAULT_MODEL_CONFIG, load_model


def prompt_text(row: dict) -> str:
    return "\n".join(message["content"] for message in row["prompt"])


@torch.inference_mode()
def evaluate_aime(model, tokenizer, rows: list[dict], samples: int, max_tokens: int) -> float:
    correct = 0
    total = 0
    eos = tokenizer.eos_id()
    for row in rows:
        _, responses, _, _ = generate_group(
            model,
            tokenizer,
            prompt_text(row),
            samples,
            max_tokens,
            1.0,
            0.7,
            capture_stats=False,
        )
        truth = row["reward_model"]["ground_truth"]
        for response in responses:
            if eos >= 0 and (response == eos).any():
                response = response[: int((response == eos).nonzero()[0]) + 1]
            is_correct, _ = verify_answer(tokenizer.decode(response.tolist()), truth)
            correct += int(is_correct)
            total += 1
    return correct / total


def deterministic_aime(model, tokenizer, rows, samples: int, max_tokens: int, seed: int) -> float:
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state()
    python_state = random.getstate()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    try:
        return evaluate_aime(model, tokenizer, rows, samples, max_tokens)
    finally:
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state)
        random.setstate(python_state)


def slice_batch(batch: TrajectoryBatch, indices: torch.Tensor) -> TrajectoryBatch:
    return TrajectoryBatch(
        input_ids=batch.input_ids[indices],
        target_ids=batch.target_ids[indices],
        response_mask=batch.response_mask[indices],
        old_logprobs=batch.old_logprobs[indices],
        old_values=batch.old_values[indices],
        rewards=batch.rewards[indices],
        correct=batch.correct[indices],
        texts=[batch.texts[int(index)] for index in indices.cpu()],
    )


def concatenate_batches(batches: list[TrajectoryBatch], pad_id: int) -> TrajectoryBatch:
    """Pad and concatenate rollout shards, normally after moving them to CPU."""
    if not batches:
        raise ValueError("at least one rollout shard is required")
    maximum = max(batch.input_ids.size(1) for batch in batches)
    total = sum(batch.input_ids.size(0) for batch in batches)

    def combined(name: str, value: float | int) -> torch.Tensor:
        exemplar = getattr(batches[0], name)
        result = torch.full(
            (total, maximum), value, dtype=exemplar.dtype, device=exemplar.device
        )
        offset = 0
        for batch in batches:
            tensor = getattr(batch, name)
            result[offset : offset + tensor.size(0), : tensor.size(1)].copy_(tensor)
            offset += tensor.size(0)
        return result

    return TrajectoryBatch(
        input_ids=combined("input_ids", pad_id),
        target_ids=combined("target_ids", pad_id),
        response_mask=combined("response_mask", 0),
        old_logprobs=combined("old_logprobs", 0),
        old_values=combined("old_values", 0),
        rewards=combined("rewards", 0),
        correct=torch.cat([batch.correct for batch in batches]),
        texts=[text for batch in batches for text in batch.texts],
    )


def rollout_diagnostics(
    batch: TrajectoryBatch,
    samples_per_prompt: int,
    eos_id: int,
    include_learning_stats: bool = True,
) -> dict[str, float | int]:
    """Metrics used to decide whether sparse-reward RL is learnable at all."""
    lengths = batch.response_mask.sum(1)
    valid = batch.response_mask.bool()
    lambdas = length_adaptive_lambda(lengths)
    last = (batch.response_mask.size(1) - 1 - batch.response_mask.flip(1).argmax(1)).long()
    terminal_ids = batch.target_ids[torch.arange(batch.target_ids.size(0)), last]
    eos_fraction = (terminal_ids == eos_id).float().mean() if eos_id >= 0 else torch.tensor(0.0)
    if batch.correct.numel() % samples_per_prompt:
        raise ValueError("rollout trajectories are not divisible into prompt groups")
    positive_groups = batch.correct.reshape(-1, samples_per_prompt).any(1)

    def mean_std(values: torch.Tensor) -> tuple[float, float]:
        if values.numel() == 0:
            return 0.0, 0.0
        return float(values.mean()), float(values.std(unbiased=False))

    metrics: dict[str, float | int] = {
        "trajectories": batch.correct.numel(),
        "prompt_groups": positive_groups.numel(),
        "positive_trajectories": int(batch.correct.sum()),
        "positive_groups": int(positive_groups.sum()),
        "positive_group_fraction": float(positive_groups.float().mean()),
        "accuracy": float(batch.correct.float().mean()),
        "reward": float(torch.where(batch.correct, 1.0, -1.0).mean()),
        "eos_fraction": float(eos_fraction),
        "truncation_fraction": float(1.0 - eos_fraction),
        "response_length_mean": float(lengths.float().mean()),
        "response_length_p50": float(torch.quantile(lengths.float(), 0.50)),
        "response_length_p95": float(torch.quantile(lengths.float(), 0.95)),
        "response_length_max": int(lengths.max()),
        "lambda_mean": float(lambdas.mean()),
    }
    if include_learning_stats:
        selected_values = batch.old_values[valid].float()
        advantages, value_targets = generalized_advantage_estimate(
            batch.rewards, batch.old_values, batch.response_mask, lambdas
        )
        selected_advantages = advantages[valid].float()
        selected_targets = value_targets[valid].float()
        value_mean, value_std = mean_std(selected_values)
        advantage_mean, advantage_std = mean_std(selected_advantages)
        target_mean, target_std = mean_std(selected_targets)
        metrics.update(
            value_mean=value_mean,
            value_std=value_std,
            value_target_mean=target_mean,
            value_target_std=target_std,
            advantage_mean=advantage_mean,
            advantage_std=advantage_std,
            advantage_min=float(selected_advantages.min()) if selected_advantages.numel() else 0.0,
            advantage_max=float(selected_advantages.max()) if selected_advantages.numel() else 0.0,
        )
    return metrics


def frozen_features(model, input_ids: torch.Tensor) -> torch.Tensor:
    compiled = getattr(model, "_vapo_compiled_features", None)
    if compiled is None:
        fn = model.detached_probe_features
        compiled = torch.compile(fn, dynamic=True, fullgraph=False) if input_ids.is_cuda else fn
        object.__setattr__(model, "_vapo_compiled_features", compiled)
    return compiled(input_ids)


def probe_outputs(model, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    compiled = getattr(model, "_vapo_compiled_probes", None)
    if compiled is None:
        def forward(selected: torch.Tensor):
            return model.logits_from_features(selected), model.values_from_features(selected)

        compiled = torch.compile(forward, dynamic=True, fullgraph=False) if features.is_cuda else forward
        object.__setattr__(model, "_vapo_compiled_probes", compiled)
    return compiled(features)


def critic_outputs(model, features: torch.Tensor) -> torch.Tensor:
    compiled = getattr(model, "_vapo_compiled_critic", None)
    if compiled is None:
        fn = model.values_from_features
        compiled = torch.compile(fn, dynamic=True, fullgraph=False) if features.is_cuda else fn
        object.__setattr__(model, "_vapo_compiled_critic", compiled)
    return compiled(features)


def generation_outputs(model, token_ids, caches, position: int):
    compiled = getattr(model, "_vapo_compiled_generation", None)
    position_tensor = torch.tensor(position, device=token_ids.device)
    if compiled is None:
        fn = model.generation_step
        compiled = torch.compile(fn, dynamic=True, fullgraph=False) if token_ids.is_cuda else fn
        object.__setattr__(model, "_vapo_compiled_generation", compiled)
    return compiled(token_ids, caches, position_tensor)


@torch.inference_mode()
def generate_group(
    model,
    tokenizer,
    prompt: str,
    samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    capture_stats: bool = True,
):
    device = next(model.parameters()).device
    prompt_ids = [tokenizer.bos_id()] + tokenizer.encode(prompt)
    tokens = torch.tensor(prompt_ids, device=device).repeat(samples, 1)
    caches = model.make_generation_cache(samples, len(prompt_ids) + max_new_tokens, device)
    logits = None
    state_values = None
    for position in range(tokens.size(1)):
        logits, state_values, caches = generation_outputs(
            model, tokens[:, position], caches, position
        )
    if logits is None:
        raise ValueError("empty prompt")
    finished = torch.zeros(samples, dtype=torch.bool, device=device)
    generated: list[torch.Tensor] = []
    sampled_logprobs: list[torch.Tensor] = []
    sampled_values: list[torch.Tensor] = []
    eos = tokenizer.eos_id()
    for offset in range(max_new_tokens):
        next_token = top_p_sample(logits, temperature, top_p)
        selected_logprobs = None
        if capture_stats:
            selected_logprobs = logits.float().log_softmax(-1).gather(
                -1, next_token[:, None]
            ).squeeze(-1)
        if eos >= 0:
            next_token = torch.where(finished, torch.full_like(next_token, eos), next_token)
            finished |= next_token == eos
        generated.append(next_token)
        if capture_stats:
            sampled_logprobs.append(selected_logprobs)
            sampled_values.append(state_values.float())
        if finished.all():
            break
        logits, state_values, caches = generation_outputs(
            model, next_token, caches, len(prompt_ids) + offset
        )
    responses = torch.stack(generated, dim=1)
    return (
        tokens,
        responses,
        torch.stack(sampled_logprobs, dim=1) if capture_stats else None,
        torch.stack(sampled_values, dim=1) if capture_stats else None,
    )


def collect_rollouts(
    model, tokenizer, rows: list[dict], samples: int, max_new_tokens: int,
    temperature: float, top_p: float, token_chunk_size: int = 8192,
) -> TrajectoryBatch:
    device = next(model.parameters()).device
    sequences: list[torch.Tensor] = []
    prompt_lengths: list[int] = []
    correct: list[bool] = []
    texts: list[str] = []
    response_logprobs: list[torch.Tensor] = []
    response_values: list[torch.Tensor] = []
    eos = tokenizer.eos_id()
    for row in rows:
        prompts, responses, sampled_logprobs, sampled_values = generate_group(
            model, tokenizer, prompt_text(row), samples, max_new_tokens, temperature, top_p
        )
        if sampled_logprobs is None or sampled_values is None:
            raise RuntimeError("training rollout did not capture policy statistics")
        ground_truth = row["reward_model"]["ground_truth"]
        for index in range(samples):
            response = responses[index]
            if eos >= 0 and (response == eos).any():
                response = response[: int((response == eos).nonzero()[0]) + 1]
            sequence = torch.cat((prompts[index], response))
            text = tokenizer.decode(response.tolist())
            is_correct, _ = verify_answer(text, ground_truth)
            sequences.append(sequence)
            prompt_lengths.append(prompts.size(1))
            correct.append(is_correct)
            texts.append(text)
            response_logprobs.append(sampled_logprobs[index, : response.numel()])
            response_values.append(sampled_values[index, : response.numel()])

    maximum = max(sequence.numel() for sequence in sequences)
    pad_id = tokenizer.pad_id() if tokenizer.pad_id() >= 0 else tokenizer.eos_id()
    padded = torch.full((len(sequences), maximum), pad_id, dtype=torch.long, device=device)
    response_mask = torch.zeros((len(sequences), maximum - 1), dtype=torch.float32, device=device)
    logprobs = torch.zeros_like(response_mask)
    values = torch.zeros_like(response_mask)
    for index, (sequence, prompt_length, action_logprobs, action_values) in enumerate(
        zip(
            sequences,
            prompt_lengths,
            response_logprobs,
            response_values,
            strict=True,
        )
    ):
        padded[index, : sequence.numel()] = sequence
        response_slice = slice(prompt_length - 1, sequence.numel() - 1)
        response_mask[index, response_slice] = 1
        logprobs[index, response_slice] = action_logprobs
        values[index, response_slice] = action_values
    input_ids, target_ids = padded[:, :-1], padded[:, 1:]
    rewards = torch.zeros_like(values)
    correct_tensor = torch.tensor(correct, dtype=torch.bool, device=device)
    lengths = response_mask.sum(1).long()
    last = (response_mask.size(1) - 1 - response_mask.flip(1).argmax(1)).long()
    # Match the official DAPO verifier used by VAPO's data lineage: +1 for an
    # exact answer and -1 otherwise, with reward only on the terminal token.
    terminal_rewards = torch.where(correct_tensor, 1.0, -1.0)
    rewards[torch.arange(rewards.size(0), device=device), last] = terminal_rewards
    return TrajectoryBatch(
        input_ids, target_ids, response_mask, logprobs, values, rewards, correct_tensor, texts
    )


def collect_rollouts_to_cpu(
    model,
    tokenizer,
    rows: list[dict],
    samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    prompt_chunk_size: int,
    token_chunk_size: int,
) -> TrajectoryBatch:
    """Collect a paper-sized rollout as bounded GPU shards backed by CPU RAM."""
    if prompt_chunk_size < 1:
        raise ValueError("rollout prompt chunk size must be positive")
    shards: list[TrajectoryBatch] = []
    for start in range(0, len(rows), prompt_chunk_size):
        shard = collect_rollouts(
            model,
            tokenizer,
            rows[start : start + prompt_chunk_size],
            samples,
            max_new_tokens,
            temperature,
            top_p,
            token_chunk_size,
        )
        shards.append(shard.to(torch.device("cpu")))
        del shard
    pad_id = tokenizer.pad_id() if tokenizer.pad_id() >= 0 else tokenizer.eos_id()
    return concatenate_batches(shards, pad_id)


def _gradient_norm(parameters) -> float:
    norms = [parameter.grad.detach().float().norm() for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(norms).norm()) if norms else 0.0


def _backward_microbatch(
    model,
    batch: TrajectoryBatch,
    value_only: bool,
    token_chunk_size: int,
    token_denominator: int,
    correct_denominator: int,
) -> dict[str, torch.Tensor]:
    lengths = batch.response_mask.sum(1)
    if value_only:
        lambdas = torch.ones_like(lengths)
    else:
        lambdas = length_adaptive_lambda(lengths)
    advantages, _ = generalized_advantage_estimate(
        batch.rewards, batch.old_values, batch.response_mask, lambdas
    )
    _, value_targets = generalized_advantage_estimate(
        batch.rewards, batch.old_values, batch.response_mask, torch.ones_like(lengths)
    )

    selected = batch.response_mask.bool().flatten()
    with torch.no_grad():
        features = frozen_features(model, batch.input_ids).flatten(0, 1)[selected]
    targets = batch.target_ids.flatten()[selected]
    old_logprobs = batch.old_logprobs.flatten()[selected]
    selected_advantages = advantages.flatten()[selected].detach()
    selected_value_targets = value_targets.flatten()[selected].detach()
    count = int(selected.sum().item())
    if correct_denominator:
        trajectory_weights = batch.correct.float() / (correct_denominator * lengths.clamp_min(1))
        positive_weights = (batch.response_mask * trajectory_weights[:, None]).flatten()[selected]
    else:
        positive_weights = torch.zeros(count, device=features.device)

    sums = {
        key: torch.zeros((), device=features.device, dtype=torch.float64)
        for key in (
            "policy", "value", "positive", "entropy", "kl", "clipped", "residual",
            "target", "target_sq", "advantage", "advantage_sq", "ratio", "ratio_sq",
            "value_pred",
        )
    }
    sums["target"] = selected_value_targets.double().sum()
    sums["target_sq"] = selected_value_targets.double().square().sum()
    sums["advantage"] = selected_advantages.double().sum()
    sums["advantage_sq"] = selected_advantages.double().square().sum()
    sums["advantage_min"] = selected_advantages.min().double()
    sums["advantage_max"] = selected_advantages.max().double()
    sums["ratio_min"] = torch.full((), torch.inf, device=features.device, dtype=torch.float64)
    sums["ratio_max"] = torch.full((), -torch.inf, device=features.device, dtype=torch.float64)
    for start in range(0, count, token_chunk_size):
        end = min(start + token_chunk_size, count)
        chunk_features = features[start:end]
        if value_only:
            chunk_values = critic_outputs(model, chunk_features).float()
            value_loss = (
                chunk_values - selected_value_targets[start:end]
            ).square().sum() / token_denominator
            value_loss.backward()
            sums["value"] += value_loss.detach().double()
            sums["value_pred"] += chunk_values.detach().double().sum()
            sums["residual"] += (
                chunk_values.detach() - selected_value_targets[start:end]
            ).square().sum().double()
            continue

        logits, chunk_values = probe_outputs(model, chunk_features)
        all_logprobs = logits.float().log_softmax(-1)
        new_logprobs = all_logprobs.gather(-1, targets[start:end, None]).squeeze(-1)
        ratio = (new_logprobs - old_logprobs[start:end]).exp()
        clipped_ratio = ratio.clamp(0.80, 1.28)
        objective = torch.minimum(
            ratio * selected_advantages[start:end],
            clipped_ratio * selected_advantages[start:end],
        )
        policy_loss = -objective.sum() / token_denominator
        value_loss = (
            chunk_values.float() - selected_value_targets[start:end]
        ).square().sum() / token_denominator
        positive_loss = -(new_logprobs * positive_weights[start:end]).sum()
        total_chunk = policy_loss + value_loss + 0.1 * positive_loss
        total_chunk.backward()
        sums["policy"] += policy_loss.detach().double()
        sums["value"] += value_loss.detach().double()
        sums["positive"] += positive_loss.detach().double()
        sums["entropy"] += (
            -(all_logprobs.exp() * all_logprobs).sum(-1)
        ).sum().detach().double()
        sums["kl"] += (
            old_logprobs[start:end] - new_logprobs
        ).sum().detach().double()
        sums["clipped"] += ((ratio < 0.80) | (ratio > 1.28)).sum().double()
        sums["ratio"] += ratio.detach().double().sum()
        sums["ratio_sq"] += ratio.detach().double().square().sum()
        sums["ratio_min"] = torch.minimum(sums["ratio_min"], ratio.detach().double().min())
        sums["ratio_max"] = torch.maximum(sums["ratio_max"], ratio.detach().double().max())
        sums["value_pred"] += chunk_values.detach().double().sum()
        sums["residual"] += (
            chunk_values.detach().float() - selected_value_targets[start:end]
        ).square().sum().double()
    sums["tokens"] = torch.tensor(count, device=features.device, dtype=torch.float64)
    sums["trajectories"] = torch.tensor(
        batch.correct.numel(), device=features.device, dtype=torch.float64
    )
    sums["correct"] = batch.correct.double().sum()
    sums["length"] = lengths.double().sum()
    return sums


def update_step_accumulated(
    model,
    batch: TrajectoryBatch,
    actor_optimizer,
    critic_optimizer,
    value_only: bool = False,
    token_chunk_size: int = 8192,
    microbatch_trajectories: int = 8,
):
    """Update one optimizer minibatch while staging bounded trajectory shards on GPU."""
    if microbatch_trajectories < 1:
        raise ValueError("microbatch trajectories must be positive")
    device = next(model.parameters()).device
    token_denominator = max(int(batch.response_mask.sum().item()), 1)
    correct_denominator = int(batch.correct.sum().item())
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    totals: dict[str, torch.Tensor] | None = None
    order = torch.arange(batch.correct.numel())
    for start in range(0, order.numel(), microbatch_trajectories):
        micro = slice_batch(batch, order[start : start + microbatch_trajectories]).to(device)
        sums = _backward_microbatch(
            model,
            micro,
            value_only,
            token_chunk_size,
            token_denominator,
            correct_denominator,
        )
        if totals is None:
            totals = sums
        else:
            for key, value in sums.items():
                if key.endswith("_min"):
                    totals[key] = torch.minimum(totals[key], value)
                elif key.endswith("_max"):
                    totals[key] = torch.maximum(totals[key], value)
                else:
                    totals[key] += value
        del micro
    if totals is None:
        raise ValueError("empty optimizer minibatch")

    actor_grad_norm = _gradient_norm(model.policy_probe.parameters())
    critic_grad_norm = _gradient_norm(model.critic_probe.parameters())

    actor_optimizer.step()
    critic_optimizer.step()
    count = totals["tokens"].clamp_min(1)
    target_mean = totals["target"] / count
    target_variance = totals["target_sq"] / count - target_mean.square()
    explained_variance = torch.where(
        target_variance > 1e-8,
        1.0 - (totals["residual"] / count) / target_variance.clamp_min(1e-8),
        torch.zeros_like(target_variance),
    )
    if value_only:
        return {
            "value_loss": float(totals["value"]),
            "explained_variance": float(explained_variance),
            "value_mean": float(totals["value_pred"] / count),
            "value_target_mean": float(target_mean),
            "critic_grad_norm": critic_grad_norm,
        }
    advantage_mean = totals["advantage"] / count
    advantage_variance = totals["advantage_sq"] / count - advantage_mean.square()
    ratio_mean = totals["ratio"] / count
    ratio_variance = totals["ratio_sq"] / count - ratio_mean.square()
    total = totals["policy"] + totals["value"] + 0.1 * totals["positive"]
    return {
        "loss": float(total),
        "policy_loss": float(totals["policy"]),
        "value_loss": float(totals["value"]),
        "positive_lm_loss": float(totals["positive"]),
        "clip_fraction": float(totals["clipped"] / count),
        "entropy": float(totals["entropy"] / count),
        "approx_kl": float(totals["kl"] / count),
        "explained_variance": float(explained_variance),
        "reward": float(2.0 * totals["correct"] / totals["trajectories"] - 1.0),
        "accuracy": batch.correct.float().mean().item(),
        "positive_trajectories": int(totals["correct"]),
        "response_length": float(totals["length"] / totals["trajectories"]),
        "advantage_mean": float(advantage_mean),
        "advantage_std": float(advantage_variance.clamp_min(0).sqrt()),
        "advantage_min": float(totals["advantage_min"]),
        "advantage_max": float(totals["advantage_max"]),
        "value_mean": float(totals["value_pred"] / count),
        "value_target_mean": float(target_mean),
        "ratio_mean": float(ratio_mean),
        "ratio_std": float(ratio_variance.clamp_min(0).sqrt()),
        "ratio_min": float(totals["ratio_min"]),
        "ratio_max": float(totals["ratio_max"]),
        "actor_grad_norm": actor_grad_norm,
        "critic_grad_norm": critic_grad_norm,
    }


def update_step(
    model, batch: TrajectoryBatch, actor_optimizer, critic_optimizer,
    value_only: bool = False, token_chunk_size: int = 8192,
):
    """Compatibility wrapper for a single in-memory optimizer minibatch."""
    return update_step_accumulated(
        model,
        batch,
        actor_optimizer,
        critic_optimizer,
        value_only=value_only,
        token_chunk_size=token_chunk_size,
        microbatch_trajectories=max(batch.correct.numel(), 1),
    )


def save_checkpoint(path: Path, model, actor_optimizer, critic_optimizer, step: int, cursor: int, args) -> None:
    atomic_torch_save(
        {
            "step": step,
            "cursor": cursor,
            "model": model.state_dict(),
            "model_config": getattr(model, "model_config", DEFAULT_MODEL_CONFIG),
            "architecture": getattr(model, "architecture", None),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "args": vars(args),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "python_rng": random.getstate(),
        },
        path,
    )




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="postraining/data/dapo-math-17k.parquet")
    parser.add_argument("--tokenizer", default="data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--output", default="postraining/runs/fresh_lejepa_vapo")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--value-warmup-steps", type=int, default=50)
    parser.add_argument("--prompts-per-rollout", type=int, default=512)
    parser.add_argument("--rollout-prompt-chunk", type=int, default=1)
    parser.add_argument("--samples-per-prompt", type=int, default=16)
    parser.add_argument("--minibatch-trajectories", type=int, default=512)
    parser.add_argument("--microbatch-trajectories", type=int, default=8)
    parser.add_argument("--loss-chunk-tokens", type=int, default=8192)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=20480)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--actor-lr", type=float, default=1e-6)
    parser.add_argument("--critic-lr", type=float, default=2e-6)
    parser.add_argument(
        "--checkpoint-interval-seconds", type=float, default=480.0
    )
    parser.add_argument("--aime-every", type=int, default=100)
    parser.add_argument("--aime-data", default="postraining/data/aime-2024.parquet")
    parser.add_argument("--aime-samples", type=int, default=32)
    parser.add_argument("--aime-max-tokens", type=int, default=20480)
    parser.add_argument("--lr-warmup-steps", type=int, default=10)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--rollout-only", action="store_true")
    parser.add_argument("--gate-prompts", type=int, default=128)
    parser.add_argument("--gate-min-positive-trajectories", type=int, default=1)
    parser.add_argument("--gate-min-positive-groups", type=int, default=1)
    parser.add_argument("--gate-max-truncation-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    checkpoint_policy = RecoveryCheckpointPolicy(
        args.checkpoint_interval_seconds
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    rows = load_unique_math_rows(args.data)
    aime_rows = load_unique_math_rows(args.aime_data) if args.aime_every > 0 and not args.rollout_only else []
    random.shuffle(rows)
    model = load_model(args.resume or args.checkpoint, device)
    model.eval()
    if hasattr(model, "fold_input_projector_for_inference"):
        model.fold_input_projector_for_inference()
    actor = torch.optim.AdamW(model.policy_probe.parameters(), lr=args.actor_lr, fused=True)
    critic = torch.optim.AdamW(model.critic_probe.parameters(), lr=args.critic_lr, fused=True)
    output = Path(args.output)
    logger = JsonlLogger(output / "metrics.jsonl")
    tensorboard = SummaryWriter(output / "tensorboard")

    if args.prompts_per_rollout < 1:
        raise ValueError("prompts per rollout must be positive")
    if args.minibatch_trajectories < 1 or args.microbatch_trajectories < 1:
        raise ValueError("optimizer batch sizes must be positive")
    rollout_trajectories = args.prompts_per_rollout * args.samples_per_prompt
    if not args.rollout_only and rollout_trajectories % args.minibatch_trajectories:
        raise ValueError(
            "prompts-per-rollout * samples-per-prompt must be divisible by "
            "minibatch-trajectories; partial optimizer minibatches are not paper-equivalent"
        )
    if not args.rollout_only and args.minibatch_trajectories % args.samples_per_prompt:
        raise ValueError(
            "minibatch-trajectories must be divisible by samples-per-prompt so value "
            "warmup uses complete prompt groups"
        )

    if args.rollout_only:
        gate_rows = rows[: min(args.gate_prompts, len(rows))]
        gate = collect_rollouts_to_cpu(
            model,
            tokenizer,
            gate_rows,
            args.samples_per_prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.rollout_prompt_chunk,
            args.loss_chunk_tokens,
        )
        metrics = rollout_diagnostics(gate, args.samples_per_prompt, tokenizer.eos_id())
        passed = (
            metrics["positive_trajectories"] >= args.gate_min_positive_trajectories
            and metrics["positive_groups"] >= args.gate_min_positive_groups
            and metrics["truncation_fraction"] <= args.gate_max_truncation_fraction
        )
        logger.log(type="rollout_gate", passed=passed, **metrics)
        for key, value in metrics.items():
            tensorboard.add_scalar(f"rollout_gate/{key}", value, 0)
        tensorboard.add_scalar("rollout_gate/passed", int(passed), 0)
        tensorboard.close()
        print(json.dumps({"passed": passed, **metrics}, sort_keys=True))
        raise SystemExit(0 if passed else 2)

    cursor = 0
    start_step = 0
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        prior_args = resume["args"]
        immutable = (
            "data", "tokenizer", "prompts_per_rollout", "rollout_prompt_chunk",
            "samples_per_prompt", "minibatch_trajectories", "microbatch_trajectories",
            "loss_chunk_tokens", "ppo_epochs", "max_new_tokens", "temperature", "top_p", "actor_lr", "critic_lr",
            "seed", "lr_warmup_steps", "aime_data", "aime_samples", "aime_max_tokens",
        )
        mismatches = [key for key in immutable if getattr(args, key) != prior_args[key]]
        if mismatches:
            raise ValueError(f"resume configuration differs for: {', '.join(mismatches)}")
        model.load_state_dict(resume["model"], strict=True)
        actor.load_state_dict(resume["actor_optimizer"])
        critic.load_state_dict(resume["critic_optimizer"])
        cursor = int(resume.get("cursor", 0))
        start_step = int(resume["step"])
        torch.set_rng_state(resume["cpu_rng"])
        torch.cuda.set_rng_state(resume["cuda_rng"])
        random.setstate(resume["python_rng"])
    logger.log(type="config", **vars(args))
    torch.cuda.reset_peak_memory_stats()

    for warmup in range(1, args.value_warmup_steps + 1 if start_step == 0 and not args.resume else 1):
        critic.param_groups[0]["lr"] = args.critic_lr * min(warmup / max(args.lr_warmup_steps, 1), 1.0)
        # A warmup "step" is one paper-sized optimizer minibatch from a fixed
        # policy, not an entire 8192-trajectory PPO rollout.
        warmup_prompts = args.minibatch_trajectories // args.samples_per_prompt
        selected = [rows[(cursor + i) % len(rows)] for i in range(warmup_prompts)]
        cursor += warmup_prompts
        batch = collect_rollouts_to_cpu(
            model, tokenizer, selected, args.samples_per_prompt, args.max_new_tokens,
            args.temperature, args.top_p, args.rollout_prompt_chunk, args.loss_chunk_tokens,
        )
        metrics = update_step_accumulated(
            model, batch, actor, critic, value_only=True,
            token_chunk_size=args.loss_chunk_tokens,
            microbatch_trajectories=args.microbatch_trajectories,
        )
        logger.log(type="value_warmup", step=warmup, **metrics)
        for key, value in metrics.items():
            tensorboard.add_scalar(f"value_warmup/{key}", value, warmup)
    if not args.resume:
        value_checkpoint = output / "checkpoints/value_pretrained.pt"
        save_checkpoint(
            value_checkpoint, model, actor, critic, 0, cursor, args
        )
        atomic_link_or_copy(value_checkpoint, output / "vapo_checkpoint.pt")
        checkpoint_policy.committed((0, cursor))
        if args.aime_every > 0:
            accuracy = deterministic_aime(
                model, tokenizer, aime_rows, args.aime_samples, args.aime_max_tokens, args.seed
            )
            logger.log(type="aime", step=0, accuracy=accuracy, samples=len(aime_rows) * args.aime_samples)
            tensorboard.add_scalar("aime/accuracy", accuracy, 0)

    step = start_step
    while step < args.steps:
        rollout_started = time.perf_counter()
        selected = [rows[(cursor + i) % len(rows)] for i in range(args.prompts_per_rollout)]
        cursor += args.prompts_per_rollout
        batch = collect_rollouts_to_cpu(
            model, tokenizer, selected, args.samples_per_prompt, args.max_new_tokens,
            args.temperature, args.top_p, args.rollout_prompt_chunk, args.loss_chunk_tokens,
        )
        rollout_seconds = time.perf_counter() - rollout_started
        rollout_tokens = int(batch.response_mask.sum().item())
        rollout_metrics = rollout_diagnostics(
            batch, args.samples_per_prompt, tokenizer.eos_id(), include_learning_stats=False
        )
        logger.log(type="rollout", step=step, **rollout_metrics)
        for key, value in rollout_metrics.items():
            tensorboard.add_scalar(f"rollout/{key}", value, step)
        previous_step = step
        for epoch in range(args.ppo_epochs):
            order = torch.randperm(batch.input_ids.size(0))
            for start in range(0, order.numel(), args.minibatch_trajectories):
                step += 1
                actor.param_groups[0]["lr"] = args.actor_lr * min(step / max(args.lr_warmup_steps, 1), 1.0)
                critic.param_groups[0]["lr"] = args.critic_lr
                mini = slice_batch(batch, order[start : start + args.minibatch_trajectories])
                metrics = update_step_accumulated(
                    model, mini, actor, critic,
                    token_chunk_size=args.loss_chunk_tokens,
                    microbatch_trajectories=args.microbatch_trajectories,
                )
                logger.log(
                    type="train", step=step, ppo_epoch=epoch,
                    seconds=time.perf_counter() - rollout_started,
                    rollout_seconds=rollout_seconds,
                    rollout_tokens=rollout_tokens,
                    rollout_tokens_per_second=rollout_tokens / max(rollout_seconds, 1e-9),
                    peak_vram_bytes=torch.cuda.max_memory_allocated(), **metrics,
                )
                for key, value in metrics.items():
                    tensorboard.add_scalar(f"train/{key}", value, step)
                if args.aime_every > 0 and step % args.aime_every == 0:
                    accuracy = deterministic_aime(
                        model,
                        tokenizer,
                        aime_rows,
                        args.aime_samples,
                        args.aime_max_tokens,
                        args.seed,
                    )
                    logger.log(
                        type="aime",
                        step=step,
                        accuracy=accuracy,
                        samples=len(aime_rows) * args.aime_samples,
                    )
                    tensorboard.add_scalar("aime/accuracy", accuracy, step)
        # The complete rollout and every epoch/minibatch derived from it have
        # finished, so this is an exact restart boundary.
        checkpoint_state = (step, cursor)
        if checkpoint_policy.due():
            save_checkpoint(
                output / "vapo_checkpoint.pt",
                model,
                actor,
                critic,
                step,
                cursor,
                args,
            )
            checkpoint_policy.committed(checkpoint_state)
    terminal_checkpoint_state = (step, cursor)
    if checkpoint_policy.terminal_due(terminal_checkpoint_state):
        save_checkpoint(
            output / "vapo_checkpoint.pt",
            model,
            actor,
            critic,
            step,
            cursor,
            args,
        )
        checkpoint_policy.committed(terminal_checkpoint_state)
    tensorboard.close()


if __name__ == "__main__":
    main()
