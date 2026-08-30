"""Single-GPU LoRA VAPO for native MiniCPM5-1B.

The final RL+OPD checkpoint is the default because its math success rate gives
sparse-reward VAPO useful within-prompt variation immediately. The SFT checkpoint
can still be selected explicitly, but no tokenizer, chat-template, or attention
block is replaced. Rollout stores only token ids, selected-token log-probabilities,
and critic values; replay reconstructs vocabulary logits in bounded chunks.
"""

from __future__ import annotations

import argparse
import atexit
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from checkpointing import RecoveryCheckpointPolicy, atomic_torch_save
from postraining.core import JsonlLogger, answer_style, load_unique_math_rows, verify_answer
from postraining.eval_hf_math import prepare_prompt_ids, resolved_eos_ids
from postraining.hf_vapo import (
    MINICPM5_MODEL_ID,
    MINICPM5_REVISION,
    LoRAConfig,
    MiniCPMVAPOPolicy,
    StaticCachePool,
    TrajectoryRecord,
    chunked_frozen_head_logprobs,
    collate_replay_microbatch,
    exact_top_p_sample,
    load_adapter_state_dict,
    plan_replay_microbatches,
    replay_storage_bytes,
)
from postraining.runtime.profiling import DEVICE_SAMPLE_FIELDS, DeviceSampler
from postraining.train_vapo import prompt_text


@dataclass(frozen=True)
class DecodeStats:
    target_decode_calls: int = 0
    target_decode_positions: int = 0

@dataclass(frozen=True)
class NextLatTrainingLoss:
    loss: Tensor
    smooth_l1: Tensor
    categorical_kl: Tensor
    transitions: int


@dataclass(frozen=True)
class RolloutResult:
    records: list[TrajectoryRecord]
    generated_tokens: int
    elapsed_seconds: float
    sampling_scanned_vocabulary: int
    sampling_nucleus_mass_lower_bound: float
    decoding: DecodeStats

def static_kv_cache_bytes(
    model_config: Any,
    *,
    batch_size: int,
    cache_length: int,
    element_size: int = 2,
) -> int:
    """Exact dense Llama K/V storage for one static rollout cache."""
    if batch_size < 1 or cache_length < 1 or element_size < 1:
        raise ValueError("cache dimensions must be positive")
    return (
        batch_size
        * cache_length
        * int(model_config.num_hidden_layers)
        * 2
        * int(model_config.num_key_value_heads)
        * int(model_config.head_dim)
        * element_size
    )


def _stop_ids(policy: MiniCPMVAPOPolicy, tokenizer) -> tuple[int, ...]:
    return resolved_eos_ids(policy.causal_lm, tokenizer, "chat")


def encode_math_prompt(
    tokenizer,
    row: dict,
    *,
    prompt_tokens: int,
    enable_thinking: bool,
) -> Tensor:
    ids = prepare_prompt_ids(
        tokenizer,
        prompt_text(row),
        prompt_mode="chat",
        prompt_tokens=prompt_tokens,
        enable_thinking=enable_thinking,
    )
    return torch.tensor(ids, dtype=torch.long)


def _first_stop_length(tokens: Tensor, stop_ids: tuple[int, ...]) -> int:
    stop = torch.zeros_like(tokens, dtype=torch.bool)
    for token_id in stop_ids:
        stop |= tokens == token_id
    positions = stop.nonzero(as_tuple=False)
    return int(positions[0]) + 1 if positions.numel() else tokens.numel()


def _decode_text(tokenizer, token_ids: Tensor) -> str:
    return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)



class RolloutEngine:
    """Fully batched multi-prompt generation over one reusable static KV cache."""

    def __init__(
        self,
        policy: MiniCPMVAPOPolicy,
        tokenizer,
        *,
        prompts_per_rollout: int,
        samples_per_prompt: int,
        cache_length: int,
        temperature: float,
        top_p: float,
        compile_decode: bool,
    ) -> None:
        from transformers import StaticCache

        if prompts_per_rollout < 1 or samples_per_prompt < 1:
            raise ValueError("rollout batch dimensions must be positive")
        self.policy = policy
        self.tokenizer = tokenizer
        self.prompts_per_rollout = prompts_per_rollout
        self.samples_per_prompt = samples_per_prompt
        self.batch_size = prompts_per_rollout * samples_per_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.cache_length = cache_length
        self.stop_ids = _stop_ids(policy, tokenizer)
        self.primary_stop = self.stop_ids[0]
        model_config: Any = getattr(policy.causal_lm, "config")
        device = next(policy.parameters()).device
        self.estimated_cache_bytes = static_kv_cache_bytes(
            model_config,
            batch_size=self.batch_size,
            cache_length=cache_length,
        )
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if self.estimated_cache_bytes > int(0.7 * free_bytes):
            raise MemoryError(
                "fully parallel rollout cache requires "
                f"{self.estimated_cache_bytes / 2**30:.2f} GiB with only "
                f"{free_bytes / 2**30:.2f} GiB free; reduce the response "
                "context or use a larger GPU"
            )
        self.cache_pool = StaticCachePool(
            lambda: StaticCache(
                config=model_config,
                max_cache_len=cache_length,
            ),
            batch_size=self.batch_size,
        )
        self.cache_positions = torch.arange(cache_length, device=device)
        self.stop_tensor = torch.tensor(self.stop_ids, device=device)
        self.primary_stop_tensor = torch.tensor(self.primary_stop, device=device)
        self.generated_buffer = torch.empty(
            (self.batch_size, cache_length), dtype=torch.long, device=device
        )
        self.logprob_buffer = torch.empty(
            (self.batch_size, cache_length), dtype=torch.float32, device=device
        )
        self.value_buffer = torch.empty_like(self.logprob_buffer)
        self.attention_mask_buffer = torch.zeros(
            (self.batch_size, cache_length), dtype=torch.bool, device=device
        )
        self.prompt_lengths_buffer = torch.empty(
            self.batch_size, dtype=torch.long, device=device
        )
        self.position_id_buffer = torch.empty(
            (self.batch_size, 1), dtype=torch.long, device=device
        )
        if hasattr(torch, "_dynamo"):
            torch._dynamo.mark_static_address(self.attention_mask_buffer)
            torch._dynamo.mark_static_address(self.position_id_buffer)

        def decode(
            token_ids: Tensor,
            cache_position: Tensor,
            cache,
            attention_mask: Tensor,
            position_ids: Tensor,
        ):
            hidden = policy.cached_hidden(
                token_ids[:, None],
                past_key_values=cache,
                cache_position=cache_position,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[:, -1]
            return hidden, policy.logits(hidden), policy.critic(hidden)

        self.decode = (
            torch.compile(decode, mode="reduce-overhead", fullgraph=False)
            if compile_decode
            else decode
        )

    def release_cache(self) -> None:
        self.cache_pool.clear()

    def _prepare_prompts(
        self, prompt_ids_cpu: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor, int]:
        if len(prompt_ids_cpu) != self.prompts_per_rollout:
            raise ValueError(
                "rollout prompt count differs from the configured batch"
            )
        lengths = [prompt.numel() for prompt in prompt_ids_cpu]
        if not lengths or min(lengths) < 1:
            raise ValueError("rollout prompts cannot be empty")
        maximum = max(lengths)
        if maximum >= self.cache_length:
            raise ValueError("prompt exhausts the rollout cache")

        device = self.generated_buffer.device
        pad_token_id = int(getattr(self.policy.causal_lm, "config").pad_token_id)
        prompt_batch = torch.full(
            (self.batch_size, maximum),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        position_ids = torch.zeros_like(prompt_batch)
        attention_mask = self.attention_mask_buffer
        attention_mask.zero_()
        for group, (prompt, length) in enumerate(zip(prompt_ids_cpu, lengths)):
            row_start = group * self.samples_per_prompt
            row_stop = row_start + self.samples_per_prompt
            token_start = maximum - length
            prompt_batch[row_start:row_stop, token_start:].copy_(
                prompt.to(device)[None].expand(self.samples_per_prompt, -1)
            )
            attention_mask[row_start:row_stop, token_start:maximum] = True
            position_ids[row_start:row_stop, token_start:].copy_(
                torch.arange(length, device=device)[None].expand(
                    self.samples_per_prompt, -1
                )
            )
        repeated_lengths = torch.tensor(
            lengths, dtype=torch.long, device=device
        ).repeat_interleave(self.samples_per_prompt)
        self.prompt_lengths_buffer.copy_(repeated_lengths)
        return prompt_batch, position_ids, maximum

    @torch.inference_mode()
    def generate_prompts(
        self,
        prompt_ids_cpu: Sequence[Tensor],
        *,
        max_new_tokens: int,
        progress_callback: (
            Callable[[int, dict[str, float | int]], None] | None
        ) = None,
    ) -> tuple[Tensor, Tensor, Tensor, int, float, DecodeStats]:
        """Generate every prompt/sample row in one synchronized GPU batch."""
        started = time.perf_counter()
        device = next(self.policy.parameters()).device
        prompt_batch, prompt_position_ids, prompt_width = self._prepare_prompts(
            prompt_ids_cpu
        )
        if prompt_width + max_new_tokens > self.cache_length:
            raise ValueError("prompt and response exceed the rollout cache")
        cache = self.cache_pool.acquire(self.batch_size)
        generated = self.generated_buffer[:, :max_new_tokens]
        logprobs = self.logprob_buffer[:, :max_new_tokens]
        values = self.value_buffer[:, :max_new_tokens]
        scanned_vocabulary = 0
        nucleus_mass_lower_bound = 1.0
        generated_count = 0
        next_progress = 256
        target_decode_calls = 0
        target_decode_positions = 0

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            hidden = self.policy.cached_hidden(
                prompt_batch,
                past_key_values=cache,
                cache_position=self.cache_positions[:prompt_width],
                attention_mask=self.attention_mask_buffer,
                position_ids=prompt_position_ids,
            )[:, -1]
            logits = self.policy.logits(hidden)
            state_values = self.policy.critic(hidden)
            finished = torch.zeros(
                self.batch_size, dtype=torch.bool, device=device
            )

            def sample_target(
                target_logits: Tensor, target_values: Tensor
            ) -> Tensor:
                nonlocal generated_count
                nonlocal scanned_vocabulary
                nonlocal nucleus_mass_lower_bound
                sampled, sampled_logprobs, sampling = exact_top_p_sample(
                    target_logits,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                active = ~finished
                actual = torch.where(active, sampled, self.primary_stop_tensor)
                selected_logprobs = torch.where(
                    active, sampled_logprobs, torch.zeros_like(sampled_logprobs)
                )
                generated[:, generated_count].copy_(actual)
                logprobs[:, generated_count].copy_(selected_logprobs)
                values[:, generated_count].copy_(target_values)
                scanned_vocabulary = max(
                    scanned_vocabulary, sampling.scanned_vocabulary
                )
                nucleus_mass_lower_bound = min(
                    nucleus_mass_lower_bound, self.top_p
                )
                finished.logical_or_(
                    (actual[:, None] == self.stop_tensor[None]).any(dim=1)
                )
                generated_count += 1
                return actual

            def report_progress(force: bool = False) -> None:
                nonlocal next_progress
                crossed = generated_count >= next_progress
                while generated_count >= next_progress:
                    next_progress += 256
                if progress_callback is None or not (crossed or force):
                    return
                wall_seconds = time.perf_counter() - started
                scheduled_tokens = generated_count * self.batch_size
                progress_callback(
                    generated_count,
                    {
                        "decode_steps": generated_count,
                        "batch_rows": self.batch_size,
                        "prompt_groups": self.prompts_per_rollout,
                        "scheduled_tokens": scheduled_tokens,
                        "wall_seconds": wall_seconds,
                        "scheduled_tokens_per_second": scheduled_tokens
                        / max(wall_seconds, 1e-9),
                        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
                    },
                )

            while generated_count < max_new_tokens:
                response_offset = generated_count
                token = sample_target(logits, state_values)
                all_finished = (
                    generated_count % 64 == 0 and bool(finished.all())
                )
                if generated_count == max_new_tokens or all_finished:
                    report_progress(force=True)
                    break

                physical_position = prompt_width + response_offset
                self.attention_mask_buffer[:, physical_position] = True
                self.position_id_buffer[:, 0].copy_(
                    self.prompt_lengths_buffer + response_offset
                )
                hidden, logits, state_values = self.decode(
                    token,
                    self.cache_positions[
                        physical_position : physical_position + 1
                    ],
                    cache,
                    self.attention_mask_buffer,
                    self.position_id_buffer,
                )
                target_decode_calls += 1
                target_decode_positions += 1
                report_progress()

        return (
            generated[:, :generated_count].cpu(),
            logprobs[:, :generated_count].cpu(),
            values[:, :generated_count].cpu(),
            scanned_vocabulary,
            nucleus_mass_lower_bound,
            DecodeStats(
                target_decode_calls=target_decode_calls,
                target_decode_positions=target_decode_positions,
            ),
        )


def _build_group_records(
    tokenizer,
    row: dict,
    prompt_ids: Tensor,
    responses: Tensor,
    logprobs: Tensor,
    values: Tensor,
    *,
    samples_per_prompt: int,
    stop_ids: tuple[int, ...],
) -> tuple[list[TrajectoryRecord], int]:
    """Decode, score, and compact one host-resident prompt group."""
    truth = row["reward_model"]["ground_truth"]
    style = answer_style(row)
    records = []
    generated_tokens = 0
    for sample in range(samples_per_prompt):
        response_length = _first_stop_length(responses[sample], stop_ids)
        response = responses[sample, :response_length]
        text = _decode_text(tokenizer, response)
        correct, _ = verify_answer(text, truth, style)
        records.append(
            TrajectoryRecord.from_device(
                token_ids=torch.cat((prompt_ids, response)),
                prompt_length=prompt_ids.numel(),
                old_logprobs=logprobs[sample, :response_length],
                old_values=values[sample, :response_length],
                correct=correct,
                text=text,
            )
        )
        generated_tokens += response_length
    return records, generated_tokens


def collect_rollouts(
    engine: RolloutEngine,
    tokenizer,
    rows: list[dict],
    *,
    prompt_tokens: int,
    max_new_tokens: int,
    enable_thinking: bool,
    progress_callback: (
        Callable[[int, dict[str, float | int]], None] | None
    ) = None,
) -> RolloutResult:
    started = time.perf_counter()
    encoded_rows = [
        (
            row,
            encode_math_prompt(
                tokenizer,
                row,
                prompt_tokens=prompt_tokens,
                enable_thinking=enable_thinking,
            ),
        )
        for row in rows
    ]
    (
        responses,
        logprobs,
        values,
        scanned_vocabulary,
        nucleus_mass_lower_bound,
        decoding,
    ) = engine.generate_prompts(
        [prompt_ids for _, prompt_ids in encoded_rows],
        max_new_tokens=max_new_tokens,
        progress_callback=progress_callback,
    )

    records: list[TrajectoryRecord] = []
    generated_tokens = 0
    with ThreadPoolExecutor(
        max_workers=len(encoded_rows),
        thread_name_prefix="minicpm-vapo-score",
    ) as scoring_pool:
        pending = []
        for group, (row, prompt_ids) in enumerate(encoded_rows):
            start = group * engine.samples_per_prompt
            stop = start + engine.samples_per_prompt
            pending.append(
                scoring_pool.submit(
                    _build_group_records,
                    tokenizer,
                    row,
                    prompt_ids,
                    responses[start:stop],
                    logprobs[start:stop],
                    values[start:stop],
                    samples_per_prompt=engine.samples_per_prompt,
                    stop_ids=engine.stop_ids,
                )
            )
        for future in pending:
            group_records, group_tokens = future.result()
            records.extend(group_records)
            generated_tokens += group_tokens
    return RolloutResult(
        records=records,
        generated_tokens=generated_tokens,
        elapsed_seconds=time.perf_counter() - started,
        sampling_scanned_vocabulary=scanned_vocabulary,
        sampling_nucleus_mass_lower_bound=nucleus_mass_lower_bound,
        decoding=decoding,
    )


def rollout_diagnostics(
    result: RolloutResult,
    *,
    samples_per_prompt: int,
    stop_ids: tuple[int, ...],
) -> dict[str, float | int]:
    records = result.records
    if not records or len(records) % samples_per_prompt:
        raise ValueError("rollout records must contain complete prompt groups")
    correct = torch.tensor([record.correct for record in records], dtype=torch.bool)
    grouped = correct.view(-1, samples_per_prompt)
    lengths = torch.tensor([record.response_length for record in records])
    terminated = torch.tensor(
        [int(record.token_ids[-1]) in stop_ids for record in records], dtype=torch.float32
    )
    metrics: dict[str, float | int] = {
        "trajectories": len(records),
        "prompt_groups": grouped.shape[0],
        "positive_trajectories": int(correct.sum()),
        "positive_groups": int(grouped.any(dim=1).sum()),
        "mixed_groups": int((grouped.any(dim=1) & ~grouped.all(dim=1)).sum()),
        "accuracy": float(correct.float().mean()),
        "truncation_fraction": float(1.0 - terminated.mean()),
        "response_length_mean": float(lengths.float().mean()),
        "response_length_p95": float(torch.quantile(lengths.float(), 0.95)),
        "response_length_max": int(lengths.max()),
        "generated_tokens": result.generated_tokens,
        "rollout_seconds": result.elapsed_seconds,
        "rollout_tokens_per_second": result.generated_tokens
        / max(result.elapsed_seconds, 1e-9),
        "replay_storage_bytes": replay_storage_bytes(records),
        "replay_bytes_per_token": replay_storage_bytes(records)
        / max(result.generated_tokens, 1),
        "sampling_scanned_vocabulary": result.sampling_scanned_vocabulary,
        "sampling_nucleus_mass_lower_bound": (
            result.sampling_nucleus_mass_lower_bound
        ),
        "target_decode_calls": result.decoding.target_decode_calls,
        "target_decode_positions": result.decoding.target_decode_positions,
        "target_positions_per_decode_call": (
            result.decoding.target_decode_positions
            / max(result.decoding.target_decode_calls, 1)
        ),
    }
    return metrics


def device_phase_metrics(
    sampler: DeviceSampler | None,
    *,
    started: float,
    ended: float,
    power_floor: float,
    prefix: str = "",
) -> dict[str, float | int]:
    """Attach trustworthy NVML samples formed entirely inside one phase."""
    if sampler is None:
        return {}
    stats = sampler.window([(started, ended)], power_floor)
    return {
        f"{prefix}device_{name}": value
        for name, value in stats.items()
        if isinstance(value, (int, float))
    }

def _nextlat_training_loss(
    policy: MiniCPMVAPOPolicy,
    hidden: Tensor,
    batch,
    *,
    horizon: int,
    global_transitions: int,
    kl_tokens: int,
    kl_weight: float,
    mse_coefficient: float,
    kl_coefficient: float,
) -> NextLatTrainingLoss:
    """Reference-style recursive future-state prediction on replay features."""
    if (
        horizon < 1
        or global_transitions < 1
        or kl_tokens < 0
        or not 0.0 <= kl_weight <= 1.0
    ):
        raise ValueError("NextLat training dimensions are invalid")
    target_hidden = hidden.detach()
    predicted = target_hidden
    token_embeddings = policy.token_embeddings(batch.input_ids).detach()
    attention = (
        batch.attention_mask
        if batch.attention_mask is not None
        else torch.ones_like(batch.input_ids, dtype=torch.bool)
    )
    smooth_l1_sum = hidden.new_zeros((), dtype=torch.float32)
    categorical_kl = hidden.new_zeros((), dtype=torch.float32)
    transitions = 0

    for offset in range(1, horizon + 1):
        if predicted.shape[1] <= 1:
            break
        predicted = policy.nextlat_head(
            predicted[:, :-1],
            token_embeddings[:, offset:],
        )
        target = target_hidden[:, offset:]
        valid = attention[:, :-offset] & attention[:, offset:]
        count = int(valid.sum())
        if count == 0:
            continue
        predicted_valid = predicted[valid]
        target_valid = target[valid]
        smooth_l1_sum += F.smooth_l1_loss(
            predicted_valid.float(),
            target_valid.float(),
            reduction="none",
        ).mean(dim=-1).sum()
        if kl_tokens:
            selected = torch.linspace(
                0,
                count - 1,
                steps=min(count, kl_tokens),
                device=hidden.device,
            ).long()
            student_logits = policy.logits(predicted_valid[selected]).float()
            with torch.no_grad():
                teacher_logits = policy.logits(target_valid[selected]).float()
            categorical_kl += (
                F.kl_div(
                    student_logits.log_softmax(dim=-1),
                    teacher_logits.log_softmax(dim=-1),
                    log_target=True,
                    reduction="batchmean",
                )
                / horizon
                * kl_weight
            )
        transitions += count

    smooth_l1 = smooth_l1_sum / global_transitions
    total = mse_coefficient * smooth_l1 + kl_coefficient * categorical_kl
    return NextLatTrainingLoss(
        loss=total,
        smooth_l1=smooth_l1,
        categorical_kl=categorical_kl,
        transitions=transitions,
    )




def tensorboard_scalars(
    writer: SummaryWriter,
    namespace: str,
    metrics: dict[str, Any],
    step: int,
) -> None:
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{namespace}/{name}", value, step)
    writer.flush()


def _approximate_kl_terms(log_ratio: Tensor) -> Tensor:
    """Pointwise non-negative PPO KL approximation."""
    return log_ratio.exp() - 1.0 - log_ratio


def _gradient_norm(parameters) -> float:
    gradients = [
        parameter.grad.detach().float().norm()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.stack(gradients).norm()) if gradients else 0.0


def update_step(
    policy: MiniCPMVAPOPolicy,
    records: list[TrajectoryRecord],
    actor_optimizer,
    critic_optimizer,
    nextlat_optimizer,
    *,
    replay_token_budget: int,
    replay_max_trajectories: int,
    logit_chunk_tokens: int,
    clip_low: float,
    clip_high: float,
    positive_coefficient: float,
    value_coefficient: float,
    nextlat_horizon: int,
    nextlat_kl_tokens: int,
    nextlat_mse_coefficient: float,
    nextlat_kl_coefficient: float,
    value_only: bool = False,
) -> dict[str, float | int]:
    if not records:
        raise ValueError("cannot update from an empty rollout")
    device = next(policy.parameters()).device
    policy.train()
    pad_token_id = int(getattr(policy.causal_lm, "config").pad_token_id)
    order = torch.randperm(len(records)).tolist()
    plan = plan_replay_microbatches(
        records,
        order,
        token_budget=replay_token_budget,
        max_trajectories=replay_max_trajectories,
    )
    total_actions = sum(record.response_length for record in records)
    correct_denominator = sum(record.correct for record in records)
    total_nextlat_transitions = sum(
        sum(max(record.input_length - offset, 0) for offset in range(1, nextlat_horizon + 1))
        for record in records
    )
    actor_parameters = list(policy.actor_parameters())
    critic_parameters = list(policy.critic.parameters())
    nextlat_parameters = list(policy.nextlat_head.parameters())
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    nextlat_optimizer.zero_grad(set_to_none=True)
    totals = {
        name: torch.zeros((), device=device, dtype=torch.float64)
        for name in (
            "policy",
            "value",
            "positive",
            "kl",
            "sampled_forward_kl",
            "clipped",
            "ratio",
            "ratio_sq",
            "value_prediction",
            "value_target",
            "value_target_sq",
            "value_residual_sq",
            "nextlat_loss",
            "nextlat_smooth_l1",
            "nextlat_categorical_kl",
            "nextlat_transitions",
        )
    }
    kl_batch_count = min(len(plan), nextlat_kl_tokens, 4)
    kl_tokens_by_batch: dict[int, int] = {}
    if kl_batch_count:
        kl_base, kl_extra = divmod(nextlat_kl_tokens, kl_batch_count)
        length_order = sorted(
            range(len(plan)),
            key=lambda plan_index: max(
                records[record_index].input_length
                for record_index in plan[plan_index]
            ),
            reverse=True,
        )
        kl_tokens_by_batch = {
            length_order[rank * len(plan) // kl_batch_count]:
            kl_base + int(rank < kl_extra)
            for rank in range(kl_batch_count)
        }

    for plan_index, indices in enumerate(plan):
        batch = collate_replay_microbatch(
            records,
            indices,
            pad_token_id=pad_token_id,
            correct_denominator=correct_denominator,
            device=device,
        )
        if value_only:
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                hidden = policy.replay_hidden(batch.input_ids, batch.attention_mask)
                action_hidden = hidden[
                    batch.action_batch_indices, batch.action_positions
                ]
            predictions = policy.critic(action_hidden.detach()).float()
            value_loss = F.mse_loss(
                predictions, batch.value_targets, reduction="sum"
            ) / total_actions
            (value_coefficient * value_loss).backward()
            totals["value"] += value_loss.detach().double()
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = policy.replay_hidden(batch.input_ids, batch.attention_mask)
                action_hidden = hidden[
                    batch.action_batch_indices, batch.action_positions
                ]
                new_logprobs = chunked_frozen_head_logprobs(
                    action_hidden,
                    batch.targets,
                    policy.lm_head_weight,
                    chunk_tokens=logit_chunk_tokens,
                )
                predictions = policy.critic(action_hidden.detach()).float()
                log_ratio = new_logprobs - batch.old_logprobs
                ratio = log_ratio.exp()
                clipped_ratio = ratio.clamp(1.0 - clip_low, 1.0 + clip_high)
                objective = torch.minimum(
                    ratio * batch.advantages,
                    clipped_ratio * batch.advantages,
                )
                policy_loss = -objective.sum() / total_actions
                value_loss = F.mse_loss(
                    predictions, batch.value_targets, reduction="sum"
                ) / total_actions
                positive_loss = -(
                    new_logprobs * batch.positive_weights
                ).sum()
                loss = (
                    policy_loss
                    + value_coefficient * value_loss
                    + positive_coefficient * positive_loss
                )
            loss.backward()
            totals["policy"] += policy_loss.detach().double()
            totals["value"] += value_loss.detach().double()
            totals["positive"] += positive_loss.detach().double()
            totals["sampled_forward_kl"] += (
                batch.old_logprobs - new_logprobs
            ).sum().detach().double()
            totals["kl"] += _approximate_kl_terms(
                log_ratio
            ).sum().detach().double()
            totals["clipped"] += (
                (ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high)
            ).sum().double()
            totals["ratio"] += ratio.detach().double().sum()
            totals["ratio_sq"] += ratio.detach().double().square().sum()
            del (
                new_logprobs,
                log_ratio,
                ratio,
                clipped_ratio,
                objective,
                policy_loss,
                positive_loss,
                loss,
            )
        nextlat_kl_count = kl_tokens_by_batch.get(plan_index, 0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            nextlat_loss = _nextlat_training_loss(
                policy,
                hidden,
                batch,
                horizon=nextlat_horizon,
                global_transitions=total_nextlat_transitions,
                kl_tokens=nextlat_kl_count,
                kl_weight=(
                    nextlat_kl_count / nextlat_kl_tokens
                    if nextlat_kl_tokens
                    else 0.0
                ),
                mse_coefficient=nextlat_mse_coefficient,
                kl_coefficient=nextlat_kl_coefficient,
            )
        nextlat_loss.loss.backward()
        totals["nextlat_loss"] += nextlat_loss.loss.detach().double()
        totals["nextlat_smooth_l1"] += nextlat_loss.smooth_l1.detach().double()
        totals["nextlat_categorical_kl"] += (
            nextlat_loss.categorical_kl.detach().double()
        )
        totals["nextlat_transitions"] += nextlat_loss.transitions


        totals["value_prediction"] += predictions.detach().double().sum()
        totals["value_target"] += batch.value_targets.detach().double().sum()
        totals["value_target_sq"] += (
            batch.value_targets.detach().double().square().sum()
        )
        totals["value_residual_sq"] += (
            predictions.detach() - batch.value_targets
        ).double().square().sum()
        del value_loss
        del batch, hidden, action_hidden, predictions

    actor_grad_norm = 0.0 if value_only else _gradient_norm(actor_parameters)
    critic_grad_norm = _gradient_norm(critic_parameters)
    nextlat_grad_norm = _gradient_norm(nextlat_parameters)
    if not value_only:
        actor_optimizer.step()
    critic_optimizer.step()
    nextlat_optimizer.step()
    count = float(total_actions)
    target_mean = totals["value_target"] / count
    target_variance = totals["value_target_sq"] / count - target_mean.square()
    explained_variance = (
        1.0 - totals["value_residual_sq"] / count / target_variance
        if target_variance > 1e-8
        else torch.zeros_like(target_variance)
    )
    metrics: dict[str, float | int] = {
        "value_loss": float(totals["value"]),
        "value_mean": float(totals["value_prediction"] / count),
        "value_target_mean": float(target_mean),
        "explained_variance": float(explained_variance),
        "critic_grad_norm": critic_grad_norm,
        "replay_microbatches": len(plan),
        "replay_actions": total_actions,
        "nextlat_loss": float(totals["nextlat_loss"]),
        "nextlat_smooth_l1": float(totals["nextlat_smooth_l1"]),
        "nextlat_categorical_kl": float(totals["nextlat_categorical_kl"]),
        "nextlat_transitions": int(totals["nextlat_transitions"]),
        "nextlat_grad_norm": nextlat_grad_norm,
    }
    policy.eval()
    if value_only:
        return metrics
    ratio_mean = totals["ratio"] / count
    ratio_variance = totals["ratio_sq"] / count - ratio_mean.square()
    metrics.update(
        loss=float(
            totals["policy"]
            + value_coefficient * totals["value"]
            + positive_coefficient * totals["positive"]
        ),
        policy_loss=float(totals["policy"]),
        positive_lm_loss=float(totals["positive"]),
        approximate_kl=float(totals["kl"] / count),
        sampled_forward_kl=float(totals["sampled_forward_kl"] / count),
        clip_fraction=float(totals["clipped"] / count),
        ratio_mean=float(ratio_mean),
        ratio_std=float(ratio_variance.clamp_min(0).sqrt()),
        actor_grad_norm=actor_grad_norm,
    )
    return metrics


def save_checkpoint(
    path: Path,
    policy: MiniCPMVAPOPolicy,
    actor_optimizer,
    critic_optimizer,
    nextlat_optimizer,
    *,
    step: int,
    cursor: int,
    warmup_step: int,
    pending_records: list[TrajectoryRecord] | None,
    pending_epoch: int,
    args,
) -> None:
    atomic_torch_save(
        {
            "policy": policy.checkpoint_payload(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "nextlat_optimizer": nextlat_optimizer.state_dict(),
            "step": step,
            "cursor": cursor,
            "warmup_step": warmup_step,
            "pending_records": pending_records,
            "pending_epoch": pending_epoch,
            "args": vars(args),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "python_rng": random.getstate(),
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MINICPM5_MODEL_ID)
    parser.add_argument("--revision", default=MINICPM5_REVISION)
    parser.add_argument("--data", default="postraining/data/dapo-math-17k.parquet")
    parser.add_argument("--output", default="postraining/runs/minicpm5_hf_vapo")
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--value-warmup-steps", type=int, default=10)
    parser.add_argument("--prompts-per-rollout", type=int, default=4)
    parser.add_argument("--samples-per-prompt", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=1_024)
    parser.add_argument("--max-new-tokens", type=int, default=4_096)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.set_defaults(thinking=True)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--critic-width", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=2e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--nextlat-lr", type=float, default=4e-4)
    parser.add_argument("--nextlat-horizon", type=int, default=2)
    parser.add_argument("--nextlat-projection-factor", type=float, default=1.6)
    parser.add_argument("--nextlat-kl-tokens", type=int, default=64)
    parser.add_argument("--nextlat-mse-coefficient", type=float, default=1.0)
    parser.add_argument("--nextlat-kl-coefficient", type=float, default=1.0)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--clip-low", type=float, default=0.20)
    parser.add_argument("--clip-high", type=float, default=0.28)
    parser.add_argument("--positive-coefficient", type=float, default=0.1)
    parser.add_argument("--value-coefficient", type=float, default=1.0)
    parser.add_argument("--replay-token-budget", type=int, default=17_408)
    parser.add_argument("--replay-max-trajectories", type=int, default=1)
    parser.add_argument("--logit-chunk-tokens", type=int, default=128)
    parser.add_argument(
        "--compile-rollout", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--device-telemetry", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--device-telemetry-interval-ms", type=int, default=250)
    parser.add_argument("--device-power-floor", type=float, default=400.0)
    parser.add_argument("--checkpoint-interval-seconds", type=float, default=480)
    parser.add_argument("--resume")
    parser.add_argument("--rollout-only", action="store_true")
    parser.add_argument("--gate-min-positive-trajectories", type=int, default=1)
    parser.add_argument("--gate-min-positive-groups", type=int, default=1)
    parser.add_argument("--gate-max-truncation-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def _validate_args(args) -> None:
    positive_integer_names = (
        "prompts_per_rollout",
        "samples_per_prompt",
        "prompt_tokens",
        "max_new_tokens",
        "lora_rank",
        "critic_width",
        "replay_token_budget",
        "replay_max_trajectories",
        "logit_chunk_tokens",
        "nextlat_horizon",
        "nextlat_kl_tokens",
    )
    invalid = [name for name in positive_integer_names if getattr(args, name) < 1]
    if invalid:
        raise ValueError(f"positive integer options required: {', '.join(invalid)}")
    if args.steps < 0 or args.value_warmup_steps < 0 or args.ppo_epochs < 1:
        raise ValueError("training step counts must be nonnegative and epochs positive")
    if not 0 < args.top_p <= 1 or args.temperature <= 0:
        raise ValueError("sampling temperature/top-p are invalid")
    if not 0 < args.clip_low < 1 or args.clip_high <= 0:
        raise ValueError("VAPO clipping bounds are invalid")
    if (
        not math.isfinite(args.nextlat_projection_factor)
        or args.nextlat_projection_factor <= 0
        or args.nextlat_lr <= 0
    ):
        raise ValueError("NextLat projection factor and learning rate must be positive")
    if (
        args.nextlat_mse_coefficient < 0
        or args.nextlat_kl_coefficient < 0
    ):
        raise ValueError("NextLat loss coefficients must be nonnegative")
    if args.prompt_tokens + args.max_new_tokens > 20_480:
        raise ValueError(
            "configured context exceeds the validated single-GPU VAPO envelope"
        )
    maximum_replay_length = args.prompt_tokens + args.max_new_tokens - 1
    if not args.rollout_only and maximum_replay_length > args.replay_token_budget:
        raise ValueError(
            "replay token budget must fit one maximum-length trajectory"
        )
    if args.device_telemetry_interval_ms < 1:
        raise ValueError("device telemetry interval must be positive")
    if args.device_power_floor < 0:
        raise ValueError("device power floor must be nonnegative")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    output = Path(args.output)
    logger = JsonlLogger(output / "metrics.jsonl")
    checkpoint_policy = RecoveryCheckpointPolicy(args.checkpoint_interval_seconds)
    rows = load_unique_math_rows(args.data)
    random.shuffle(rows)

    lora_config = LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha)
    policy, tokenizer = MiniCPMVAPOPolicy.from_pretrained(
        model_id=args.model,
        revision=args.revision,
        device=device,
        lora_config=lora_config,
        critic_width=args.critic_width,
        nextlat_projection_factor=args.nextlat_projection_factor,
    )
    actor_parameters = list(policy.actor_parameters())
    actor_optimizer = torch.optim.AdamW(
        actor_parameters, lr=args.actor_lr, weight_decay=0.0, fused=True
    )
    critic_optimizer = torch.optim.AdamW(
        policy.critic.parameters(), lr=args.critic_lr, weight_decay=0.0, fused=True
    )
    nextlat_optimizer = torch.optim.AdamW(
        policy.nextlat_head.parameters(),
        lr=args.nextlat_lr,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        fused=True,
    )
    start_step = 0
    cursor = 0
    completed_warmup = 0
    pending_records: list[TrajectoryRecord] | None = None
    pending_epoch = 0
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        prior_args = resume["args"]
        mutable_resume_options = {
            "steps",
            "output",
            "checkpoint_interval_seconds",
            "resume",
            "device_telemetry",
            "device_telemetry_interval_ms",
            "device_power_floor",
            "rollout_only",
            "gate_min_positive_trajectories",
            "gate_min_positive_groups",
            "gate_max_truncation_fraction",
        }
        mismatches = [
            name
            for name, value in vars(args).items()
            if name not in mutable_resume_options and prior_args.get(name) != value
        ]
        if mismatches:
            raise ValueError(
                "resume configuration differs for: " + ", ".join(mismatches)
            )
        payload = resume["policy"]
        if payload.get("schema") != "minicpm5_hf_vapo_adapter/v4":
            raise ValueError("resume checkpoint uses an obsolete policy schema")
        if payload["model_id"] != args.model or payload["revision"] != args.revision:
            raise ValueError("resume base checkpoint differs")
        if payload["lora_config"] != {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "targets": tuple(lora_config.targets),
        }:
            raise ValueError("resume LoRA configuration differs")
        if payload["nextlat_projection_factor"] != args.nextlat_projection_factor:
            raise ValueError("resume NextLat projection factor differs")
        load_adapter_state_dict(policy.causal_lm, payload["adapter"])
        policy.critic.load_state_dict(payload["critic"], strict=True)
        policy.nextlat_head.load_state_dict(payload["nextlat_head"], strict=True)
        actor_optimizer.load_state_dict(resume["actor_optimizer"])
        critic_optimizer.load_state_dict(resume["critic_optimizer"])
        nextlat_optimizer.load_state_dict(resume["nextlat_optimizer"])
        start_step = int(resume["step"])
        cursor = int(resume["cursor"])
        torch.set_rng_state(resume["cpu_rng"])
        torch.cuda.set_rng_state(resume["cuda_rng"])
        random.setstate(resume["python_rng"])
        completed_warmup = int(resume["warmup_step"])
        pending_records = resume["pending_records"]
        pending_epoch = int(resume["pending_epoch"])
        if not 0 <= completed_warmup <= args.value_warmup_steps:
            raise ValueError("resume warmup progress is invalid")
        if pending_records is None and pending_epoch != 0:
            raise ValueError("resume has a pending epoch without replay records")
        if pending_records is not None and not 0 <= pending_epoch < args.ppo_epochs:
            raise ValueError("resume pending PPO epoch is invalid")
    tensorboard = SummaryWriter(
        output / "tensorboard",
        purge_step=start_step + 1 if args.resume else None,
    )

    engine = RolloutEngine(
        policy,
        tokenizer,
        prompts_per_rollout=args.prompts_per_rollout,
        samples_per_prompt=args.samples_per_prompt,
        cache_length=args.prompt_tokens + args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        compile_decode=args.compile_rollout,
    )
    device_sampler = None
    if args.device_telemetry:
        device_sampler = DeviceSampler(args.device_telemetry_interval_ms, device)
        device_sampler.start()
        atexit.register(device_sampler.stop)
    parameter_metrics = {
        "trainable_actor_parameters": sum(
            parameter.numel() for parameter in actor_parameters
        ),
        "critic_parameters": sum(
            parameter.numel() for parameter in policy.critic.parameters()
        ),
        "rollout_batch_rows": engine.batch_size,
        "nextlat_head_parameters": sum(
            parameter.numel() for parameter in policy.nextlat_head.parameters()
        ),
        "estimated_static_kv_cache_bytes": engine.estimated_cache_bytes,
        "frozen_base_parameters": sum(
            parameter.numel()
            for parameter in policy.causal_lm.parameters()
            if not parameter.requires_grad
        ),
        "total_policy_parameters": sum(
            parameter.numel() for parameter in policy.parameters()
        ),
    }
    logger.log(type="config", **parameter_metrics, **vars(args))
    tensorboard_scalars(
        tensorboard, "config", {**parameter_metrics, **vars(args)}, 0
    )
    tensorboard.add_text(
        "config/arguments", json.dumps(vars(args), sort_keys=True), 0
    )
    tensorboard.flush()

    def live_rollout_metrics(
        progress_step: int,
        metrics: dict[str, float | int],
        *,
        namespace: str = "rollout_live",
    ) -> None:
        live_metrics = dict(metrics)
        if device_sampler is not None and device_sampler.samples:
            _, device_values = device_sampler.samples[-1]
            live_metrics.update(
                {
                    f"device_{name}": value
                    for name, value in zip(DEVICE_SAMPLE_FIELDS, device_values)
                }
            )
        tensorboard_scalars(
            tensorboard, namespace, live_metrics, progress_step
        )
    policy.eval()
    torch.cuda.reset_peak_memory_stats(device)

    if args.rollout_only:
        selected = rows[: args.prompts_per_rollout]
        rollout_started = time.perf_counter()
        result = collect_rollouts(
            engine,
            tokenizer,
            selected,
            prompt_tokens=args.prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.thinking,
            progress_callback=live_rollout_metrics,
        )
        rollout_ended = time.perf_counter()
        metrics = rollout_diagnostics(
            result,
            samples_per_prompt=args.samples_per_prompt,
            stop_ids=engine.stop_ids,
        )
        metrics.update(
            device_phase_metrics(
                device_sampler,
                started=rollout_started,
                ended=rollout_ended,
                power_floor=args.device_power_floor,
            )
        )
        metrics["peak_vram_bytes"] = torch.cuda.max_memory_allocated(device)
        passed = (
            metrics["positive_trajectories"]
            >= args.gate_min_positive_trajectories
            and metrics["positive_groups"] >= args.gate_min_positive_groups
            and metrics["truncation_fraction"] <= args.gate_max_truncation_fraction
        )
        logger.log(type="rollout_gate", passed=passed, **metrics)
        tensorboard_scalars(
            tensorboard,
            "rollout_gate",
            {"passed": int(passed), **metrics},
            0,
        )
        print(json.dumps({"passed": passed, **metrics}, sort_keys=True), flush=True)
        tensorboard.close()
        raise SystemExit(0 if passed else 2)

    for warmup in range(completed_warmup, args.value_warmup_steps):
        selected = [
            rows[(cursor + index) % len(rows)]
            for index in range(args.prompts_per_rollout)
        ]
        cursor += args.prompts_per_rollout
        torch.cuda.reset_peak_memory_stats(device)
        rollout_started = time.perf_counter()
        result = collect_rollouts(
            engine,
            tokenizer,
            selected,
            prompt_tokens=args.prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.thinking,
            progress_callback=lambda progress_step, live_metrics: live_rollout_metrics(
                warmup * args.max_new_tokens + progress_step,
                live_metrics,
                namespace="value_warmup_live",
            ),
        )
        rollout_ended = time.perf_counter()
        metrics = rollout_diagnostics(
            result,
            samples_per_prompt=args.samples_per_prompt,
            stop_ids=engine.stop_ids,
        )
        metrics.update(
            device_phase_metrics(
                device_sampler,
                started=rollout_started,
                ended=rollout_ended,
                power_floor=args.device_power_floor,
                prefix="rollout_",
            )
        )
        metrics["rollout_peak_vram_bytes"] = torch.cuda.max_memory_allocated(
            device
        )
        engine.release_cache()

        torch.cuda.reset_peak_memory_stats(device)
        update_started = time.perf_counter()
        update_metrics = update_step(
            policy,
            result.records,
            actor_optimizer,
            critic_optimizer,
            nextlat_optimizer,
            replay_token_budget=args.replay_token_budget,
            replay_max_trajectories=args.replay_max_trajectories,
            logit_chunk_tokens=args.logit_chunk_tokens,
            clip_low=args.clip_low,
            clip_high=args.clip_high,
            positive_coefficient=args.positive_coefficient,
            value_coefficient=args.value_coefficient,
            nextlat_horizon=args.nextlat_horizon,
            nextlat_kl_tokens=args.nextlat_kl_tokens,
            nextlat_mse_coefficient=args.nextlat_mse_coefficient,
            nextlat_kl_coefficient=args.nextlat_kl_coefficient,
            value_only=True,
        )
        update_ended = time.perf_counter()
        metrics.update(update_metrics)
        metrics["update_seconds"] = update_ended - update_started
        metrics["update_peak_vram_bytes"] = torch.cuda.max_memory_allocated(
            device
        )
        metrics.update(
            device_phase_metrics(
                device_sampler,
                started=update_started,
                ended=update_ended,
                power_floor=args.device_power_floor,
                prefix="update_",
            )
        )
        completed_warmup = warmup + 1
        logger.log(type="value_warmup", step=completed_warmup, **metrics)
        tensorboard_scalars(
            tensorboard, "value_warmup", metrics, completed_warmup
        )
        if checkpoint_policy.due():
            save_checkpoint(
                output / "vapo_adapter_checkpoint.pt",
                policy,
                actor_optimizer,
                critic_optimizer,
                nextlat_optimizer,
                step=start_step,
                cursor=cursor,
                warmup_step=completed_warmup,
                pending_records=pending_records,
                pending_epoch=pending_epoch,
                args=args,
            )
            checkpoint_policy.committed(
                (start_step, cursor, completed_warmup, pending_epoch)
            )
    step = start_step
    while step < args.steps:
        if pending_records is None:
            selected = [
                rows[(cursor + index) % len(rows)]
                for index in range(args.prompts_per_rollout)
            ]
            cursor += args.prompts_per_rollout
            torch.cuda.reset_peak_memory_stats(device)
            rollout_started = time.perf_counter()
            result = collect_rollouts(
                engine,
                tokenizer,
                selected,
                prompt_tokens=args.prompt_tokens,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=args.thinking,
                progress_callback=lambda progress_step, live_metrics: live_rollout_metrics(
                    step
                    * args.prompts_per_rollout
                    * args.max_new_tokens
                    + progress_step,
                    live_metrics,
                ),
            )
            rollout_ended = time.perf_counter()
            rollout_metrics = rollout_diagnostics(
                result,
                samples_per_prompt=args.samples_per_prompt,
                stop_ids=engine.stop_ids,
            )
            rollout_metrics.update(
                device_phase_metrics(
                    device_sampler,
                    started=rollout_started,
                    ended=rollout_ended,
                    power_floor=args.device_power_floor,
                )
            )
            rollout_metrics["peak_vram_bytes"] = torch.cuda.max_memory_allocated(
                device
            )
            logger.log(type="rollout", step=step, **rollout_metrics)
            tensorboard_scalars(
                tensorboard, "rollout", rollout_metrics, step
            )
            engine.release_cache()
            records = result.records
            epoch_start = 0
        else:
            records = pending_records
            epoch_start = pending_epoch
            pending_records = None
            pending_epoch = 0

        for epoch in range(epoch_start, args.ppo_epochs):
            if step >= args.steps:
                pending_records = records
                pending_epoch = epoch
                break
            torch.cuda.reset_peak_memory_stats(device)
            update_started = time.perf_counter()
            metrics = update_step(
                policy,
                records,
                actor_optimizer,
                critic_optimizer,
                nextlat_optimizer,
                replay_token_budget=args.replay_token_budget,
                replay_max_trajectories=args.replay_max_trajectories,
                logit_chunk_tokens=args.logit_chunk_tokens,
                clip_low=args.clip_low,
                clip_high=args.clip_high,
                positive_coefficient=args.positive_coefficient,
                value_coefficient=args.value_coefficient,
                nextlat_horizon=args.nextlat_horizon,
                nextlat_kl_tokens=args.nextlat_kl_tokens,
                nextlat_mse_coefficient=args.nextlat_mse_coefficient,
                nextlat_kl_coefficient=args.nextlat_kl_coefficient,
            )
            update_ended = time.perf_counter()
            metrics["update_seconds"] = update_ended - update_started
            metrics.update(
                device_phase_metrics(
                    device_sampler,
                    started=update_started,
                    ended=update_ended,
                    power_floor=args.device_power_floor,
                )
            )
            step += 1
            metrics["peak_vram_bytes"] = torch.cuda.max_memory_allocated(device)
            logger.log(type="train", step=step, ppo_epoch=epoch, **metrics)
            tensorboard_scalars(tensorboard, "train", metrics, step)
        if checkpoint_policy.due():
            save_checkpoint(
                output / "vapo_adapter_checkpoint.pt",
                policy,
                actor_optimizer,
                critic_optimizer,
                nextlat_optimizer,
                step=step,
                cursor=cursor,
                warmup_step=completed_warmup,
                pending_records=pending_records,
                pending_epoch=pending_epoch,
                args=args,
            )
            checkpoint_policy.committed(
                (step, cursor, completed_warmup, pending_epoch)
            )

    save_checkpoint(
        output / "vapo_adapter_checkpoint.pt",
        policy,
        actor_optimizer,
        critic_optimizer,
        nextlat_optimizer,
        step=step,
        cursor=cursor,
        warmup_step=completed_warmup,
        pending_records=pending_records,
        pending_epoch=pending_epoch,
        args=args,
    )
    tensorboard.close()


if __name__ == "__main__":
    main()
