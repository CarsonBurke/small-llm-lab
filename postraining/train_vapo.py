"""Single-GPU VAPO post-training for the fresh LeJEPA checkpoint."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from postraining.core import (
    JsonlLogger,
    TrajectoryBatch,
    clipped_policy_loss,
    generalized_advantage_estimate,
    length_adaptive_lambda,
    load_unique_math_rows,
    masked_token_mean,
    positive_example_lm_loss,
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
        _, responses = generate_group(model, tokenizer, prompt_text(row), samples, max_tokens, 1.0, 0.7)
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


@torch.inference_mode()
def generate_group(model, tokenizer, prompt: str, samples: int, max_new_tokens: int, temperature: float, top_p: float):
    device = next(model.parameters()).device
    prompt_ids = [tokenizer.bos_id()] + tokenizer.encode(prompt)
    tokens = torch.tensor(prompt_ids, device=device).repeat(samples, 1)
    caches = model.make_generation_cache(samples, len(prompt_ids) + max_new_tokens, device)
    logits = None
    for position in range(tokens.size(1)):
        logits, _, caches = model.generation_step(tokens[:, position], caches, position)
    if logits is None:
        raise ValueError("empty prompt")
    finished = torch.zeros(samples, dtype=torch.bool, device=device)
    generated: list[torch.Tensor] = []
    eos = tokenizer.eos_id()
    for offset in range(max_new_tokens):
        next_token = top_p_sample(logits, temperature, top_p)
        if eos >= 0:
            next_token = torch.where(finished, torch.full_like(next_token, eos), next_token)
            finished |= next_token == eos
        generated.append(next_token)
        if finished.all():
            break
        logits, _, caches = model.generation_step(next_token, caches, len(prompt_ids) + offset)
    responses = torch.stack(generated, dim=1)
    return tokens, responses


def collect_rollouts(model, tokenizer, rows: list[dict], samples: int, max_new_tokens: int, temperature: float, top_p: float) -> TrajectoryBatch:
    device = next(model.parameters()).device
    sequences: list[torch.Tensor] = []
    prompt_lengths: list[int] = []
    correct: list[bool] = []
    texts: list[str] = []
    eos = tokenizer.eos_id()
    for row in rows:
        prompts, responses = generate_group(
            model, tokenizer, prompt_text(row), samples, max_new_tokens, temperature, top_p
        )
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

    maximum = max(sequence.numel() for sequence in sequences)
    pad_id = tokenizer.pad_id() if tokenizer.pad_id() >= 0 else tokenizer.eos_id()
    padded = torch.full((len(sequences), maximum), pad_id, dtype=torch.long, device=device)
    response_mask = torch.zeros((len(sequences), maximum - 1), dtype=torch.float32, device=device)
    for index, (sequence, prompt_length) in enumerate(zip(sequences, prompt_lengths, strict=True)):
        padded[index, : sequence.numel()] = sequence
        response_mask[index, prompt_length - 1 : sequence.numel() - 1] = 1
    input_ids, target_ids = padded[:, :-1], padded[:, 1:]
    with torch.no_grad():
        features = model.detached_probe_features(input_ids)
        raw_logits = model.policy_probe(features)
        logits = model.logit_softcap * torch.tanh(raw_logits / model.logit_softcap)
        logprobs = logits.float().log_softmax(-1).gather(-1, target_ids[..., None]).squeeze(-1)
        values = model.critic_probe(features).squeeze(-1).float()
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


def update_step(model, batch: TrajectoryBatch, actor_optimizer, critic_optimizer, value_only: bool = False):
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

    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    features = model.detached_probe_features(batch.input_ids)
    values = model.critic_probe(features).squeeze(-1).float()
    value_loss = masked_token_mean((values - value_targets.detach()).square(), batch.response_mask)
    target_values = value_targets.detach()
    target_mean = masked_token_mean(target_values, batch.response_mask)
    target_variance = masked_token_mean((target_values - target_mean).square(), batch.response_mask)
    residual_variance = masked_token_mean((target_values - values.detach()).square(), batch.response_mask)
    explained_variance = torch.where(
        target_variance > 1e-8,
        1.0 - residual_variance / target_variance.clamp_min(1e-8),
        torch.zeros_like(target_variance),
    )
    if value_only:
        value_loss.backward()
        critic_optimizer.step()
        return {"value_loss": value_loss.item(), "explained_variance": explained_variance.item()}

    raw_logits = model.policy_probe(features)
    logits = (model.logit_softcap * torch.tanh(raw_logits / model.logit_softcap)).float()
    all_logprobs = logits.log_softmax(-1)
    new_logprobs = all_logprobs.gather(-1, batch.target_ids[..., None]).squeeze(-1)
    policy_loss, clip_fraction = clipped_policy_loss(
        new_logprobs, batch.old_logprobs, advantages.detach(), batch.response_mask
    )
    positive_mask = batch.response_mask * batch.correct[:, None]
    positive_lm_loss = positive_example_lm_loss(
        new_logprobs, batch.response_mask, batch.correct
    )
    entropy = masked_token_mean(-(all_logprobs.exp() * all_logprobs).sum(-1), batch.response_mask)
    approx_kl = masked_token_mean(batch.old_logprobs - new_logprobs, batch.response_mask)
    total = policy_loss + value_loss + 0.1 * positive_lm_loss
    total.backward()
    actor_optimizer.step()
    critic_optimizer.step()
    return {
        "loss": total.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "positive_lm_loss": positive_lm_loss.item(),
        "clip_fraction": clip_fraction.item(),
        "entropy": entropy.item(),
        "approx_kl": approx_kl.item(),
        "explained_variance": explained_variance.item(),
        "reward": torch.where(batch.correct, 1.0, -1.0).mean().item(),
        "accuracy": batch.correct.float().mean().item(),
        "response_length": lengths.float().mean().item(),
    }


def calibrate_prompts(model, tokenizer, row, args) -> int:
    target = int(torch.cuda.get_device_properties(0).total_memory * 0.95)
    best = 1
    calibration_tokens = args.calibration_tokens or args.max_new_tokens
    for candidate in (1, 2, 4, 8, 16, 32, 64):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            batch = collect_rollouts(
                model, tokenizer, [row] * candidate, args.samples_per_prompt,
                min(args.max_new_tokens, calibration_tokens), args.temperature, args.top_p,
            )
            # Include probe backward memory, then restore weights through no step.
            actor = torch.optim.AdamW(model.policy_probe.parameters(), lr=0.0)
            critic = torch.optim.AdamW(model.critic_probe.parameters(), lr=0.0)
            update_step(model, batch, actor, critic)
            peak = torch.cuda.max_memory_allocated()
            del batch, actor, critic
            if peak >= target:
                if candidate == 1:
                    raise RuntimeError("one prompt group exceeds the 95% VRAM calibration target")
                break
            best = candidate
        except torch.cuda.OutOfMemoryError:
            if candidate == 1:
                raise RuntimeError("one prompt group does not fit in VRAM at the configured response length")
            torch.cuda.empty_cache()
            break
    return best


def save_checkpoint(path: Path, model, actor_optimizer, critic_optimizer, step: int, cursor: int, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "cursor": cursor,
            "model": model.state_dict(),
            "model_config": getattr(model, "model_config", DEFAULT_MODEL_CONFIG),
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
    parser.add_argument("--prompts-per-step", type=int, default=0)
    parser.add_argument("--samples-per-prompt", type=int, default=16)
    parser.add_argument("--minibatch-trajectories", type=int, default=512)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=20480)
    parser.add_argument("--calibration-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--actor-lr", type=float, default=1e-6)
    parser.add_argument("--critic-lr", type=float, default=2e-6)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--aime-every", type=int, default=100)
    parser.add_argument("--aime-data", default="postraining/data/aime-2024.parquet")
    parser.add_argument("--aime-samples", type=int, default=32)
    parser.add_argument("--aime-max-tokens", type=int, default=20480)
    parser.add_argument("--lr-warmup-steps", type=int, default=10)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    rows = load_unique_math_rows(args.data)
    aime_rows = load_unique_math_rows(args.aime_data)
    random.shuffle(rows)
    model = load_model(args.checkpoint, device)
    model.eval()
    actor = torch.optim.AdamW(model.policy_probe.parameters(), lr=args.actor_lr)
    critic = torch.optim.AdamW(model.critic_probe.parameters(), lr=args.critic_lr)
    output = Path(args.output)
    logger = JsonlLogger(output / "metrics.jsonl")
    tensorboard = SummaryWriter(output / "tensorboard")

    cursor = 0
    start_step = 0
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        prior_args = resume["args"]
        immutable = (
            "data", "tokenizer", "samples_per_prompt", "minibatch_trajectories",
            "ppo_epochs", "max_new_tokens", "temperature", "top_p", "actor_lr", "critic_lr",
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
        prompts_per_step = int(prior_args["prompts_per_step"])
    else:
        prompts_per_step = args.prompts_per_step or calibrate_prompts(model, tokenizer, rows[0], args)
    args.prompts_per_step = prompts_per_step
    logger.log(type="config", **vars(args))
    torch.cuda.reset_peak_memory_stats()

    for warmup in range(1, args.value_warmup_steps + 1 if start_step == 0 and not args.resume else 1):
        critic.param_groups[0]["lr"] = args.critic_lr * min(warmup / max(args.lr_warmup_steps, 1), 1.0)
        selected = [rows[(cursor + i) % len(rows)] for i in range(prompts_per_step)]
        cursor += prompts_per_step
        batch = collect_rollouts(
            model, tokenizer, selected, args.samples_per_prompt, args.max_new_tokens,
            args.temperature, args.top_p,
        )
        metrics = update_step(model, batch, actor, critic, value_only=True)
        logger.log(type="value_warmup", step=warmup, **metrics)
        for key, value in metrics.items():
            tensorboard.add_scalar(f"value_warmup/{key}", value, warmup)
    if not args.resume:
        save_checkpoint(output / "checkpoints/value_pretrained.pt", model, actor, critic, 0, cursor, args)
        if args.aime_every > 0:
            accuracy = deterministic_aime(
                model, tokenizer, aime_rows, args.aime_samples, args.aime_max_tokens, args.seed
            )
            logger.log(type="aime", step=0, accuracy=accuracy, samples=len(aime_rows) * args.aime_samples)
            tensorboard.add_scalar("aime/accuracy", accuracy, 0)

    step = start_step
    while step < args.steps:
        rollout_started = time.perf_counter()
        selected = [rows[(cursor + i) % len(rows)] for i in range(prompts_per_step)]
        cursor += prompts_per_step
        batch = collect_rollouts(
            model, tokenizer, selected, args.samples_per_prompt, args.max_new_tokens,
            args.temperature, args.top_p,
        )
        rollout_seconds = time.perf_counter() - rollout_started
        rollout_tokens = int(batch.response_mask.sum().item())
        for epoch in range(args.ppo_epochs):
            order = torch.randperm(batch.input_ids.size(0), device=device)
            for start in range(0, order.numel(), args.minibatch_trajectories):
                if step >= args.steps:
                    break
                step += 1
                actor.param_groups[0]["lr"] = args.actor_lr * min(step / max(args.lr_warmup_steps, 1), 1.0)
                critic.param_groups[0]["lr"] = args.critic_lr
                mini = slice_batch(batch, order[start : start + args.minibatch_trajectories])
                metrics = update_step(model, mini, actor, critic)
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
                should_evaluate = args.aime_every > 0 and (step % args.aime_every == 0 or step == args.steps)
                if step % args.save_every == 0 or should_evaluate or step == args.steps:
                    save_checkpoint(output / f"checkpoints/step_{step:05d}.pt", model, actor, critic, step, cursor, args)
                if should_evaluate:
                    accuracy = deterministic_aime(
                        model, tokenizer, aime_rows, args.aime_samples,
                        args.aime_max_tokens, args.seed,
                    )
                    logger.log(
                        type="aime", step=step, accuracy=accuracy,
                        samples=len(aime_rows) * args.aime_samples,
                    )
                    tensorboard.add_scalar("aime/accuracy", accuracy, step)
    tensorboard.close()


if __name__ == "__main__":
    main()
