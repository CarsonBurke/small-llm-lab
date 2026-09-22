"""Exact per-row NextLat speculative decoding with a ragged static KV cache."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch import Tensor

from postraining.vapo.policy import (
    VAPOPolicy,
    dense_top_p_probabilities,
    maximal_coupling_verify,
)
from postraining.thinking_budget import force_thinking_end_, validate_thinking_budget


@dataclass(frozen=True)
class NextLatDecodeStats:
    target_decode_calls: int = 0
    target_decode_positions: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    accepted_tokens_pos2plus: int = 0
    speculative_cycles: int = 0

    proposed_tokens_pos2plus: int = 0
    speculative_row_cycles: int = 0
    proposed_by_position: tuple[int, ...] = ()
    accepted_by_position: tuple[int, ...] = ()
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0

class RaggedStaticCacheLayer:
    """Dense static K/V tensors updated at independent positions per row."""

    is_compileable = True
    is_sliding = False

    def __init__(
        self,
        *,
        batch_size: int,
        num_heads: int,
        max_cache_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.max_cache_len = max_cache_len
        self.keys = torch.zeros(
            (batch_size, num_heads, max_cache_len, head_dim),
            dtype=dtype,
            device=device,
        )
        self.values = torch.zeros_like(self.keys)
        self.write_positions: Tensor | None = None
        torch._dynamo.mark_static_address(self.keys)
        torch._dynamo.mark_static_address(self.values)

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        *args,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        del args, kwargs
        if self.write_positions is None:
            raise RuntimeError("ragged cache write positions are not set")
        positions = self.write_positions
        if positions.shape != key_states.shape[:1] + key_states.shape[-2:-1]:
            raise ValueError("ragged cache positions do not match key states")
        indices = positions[:, None, :, None].expand_as(key_states)
        self.keys.scatter_(2, indices, key_states)
        self.values.scatter_(2, indices, value_states)
        return self.keys, self.values

    def reset(self) -> None:
        self.keys.zero_()
        self.values.zero_()
        self.write_positions = None

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        del query_length
        return self.max_cache_len, 0

    def get_seq_length(self) -> int:
        return self.max_cache_len

    def get_max_length(self) -> int:
        return self.max_cache_len


class RaggedStaticCache:
    """Transformers-compatible cache container with per-row write positions."""

    is_compileable = True

    def __init__(
        self,
        model_config: Any,
        *,
        batch_size: int,
        max_cache_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.layers = [
            RaggedStaticCacheLayer(
                batch_size=batch_size,
                num_heads=int(model_config.num_key_value_heads),
                max_cache_len=max_cache_len,
                head_dim=int(model_config.head_dim),
                dtype=dtype,
                device=device,
            )
            for _ in range(int(model_config.num_hidden_layers))
        ]

    def set_write_positions(self, positions: Tensor) -> None:
        for layer in self.layers:
            layer.write_positions = positions

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        return self.layers[layer_idx].update(
            key_states, value_states, *args, **kwargs
        )

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()


def ragged_causal_mask(valid_keys: Tensor, query_positions: Tensor) -> Tensor:
    """Build a boolean SDPA mask for independent logical row cursors."""
    if valid_keys.ndim != 2 or query_positions.ndim != 2:
        raise ValueError("ragged masks require [B,L] keys and [B,Q] positions")
    if valid_keys.shape[0] != query_positions.shape[0]:
        raise ValueError("ragged mask batch dimensions differ")
    key_positions = torch.arange(valid_keys.shape[1], device=valid_keys.device)
    causal = key_positions[None, None, :] <= query_positions[:, :, None]
    return causal[:, None] & valid_keys[:, None, None]


class NextLatSpeculativeEngine:
    """Batched exact speculative decoder with independent row cursors."""

    def __init__(
        self,
        policy: VAPOPolicy,
        *,
        stop_ids: tuple[int, ...],
        prompts_per_rollout: int,
        samples_per_prompt: int,
        cache_length: int,
        draft_length: int,
        temperature: float,
        top_p: float,
        compile_decode: bool,
        answer_reserve_tokens: int = 0,
        thinking_end_token_id: int | None = None,
    ) -> None:
        if getattr(policy, "token_carry", False):
            raise ValueError("token carry cannot use token-only NextLat proposals")
        validate_thinking_budget(answer_reserve_tokens, thinking_end_token_id)
        self.answer_reserve_tokens = answer_reserve_tokens
        self.thinking_end_token_id = thinking_end_token_id
        if (
            prompts_per_rollout < 1
            or samples_per_prompt < 1
            or cache_length < 2
            or draft_length < 2
        ):
            raise ValueError("NextLat engine dimensions are invalid")
        self.policy = policy
        self.prompts_per_rollout = prompts_per_rollout
        self.samples_per_prompt = samples_per_prompt
        self.batch_size = prompts_per_rollout * samples_per_prompt
        self.cache_length = cache_length
        self.draft_length = draft_length
        self.temperature = temperature
        self.top_p = top_p
        self.stop_ids = stop_ids
        self.primary_stop = stop_ids[0]
        device = next(policy.parameters()).device
        self.device_type = device.type
        config: Any = policy.causal_lm.config
        self.estimated_cache_bytes = (
            self.batch_size
            * cache_length
            * int(config.num_hidden_layers)
            * 2
            * int(config.num_key_value_heads)
            * int(config.head_dim)
            * torch.tensor([], dtype=torch.bfloat16).element_size()
        )
        if device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(device)
            if self.estimated_cache_bytes > int(0.7 * free_bytes):
                raise MemoryError(
                    "ragged NextLat cache requires "
                    f"{self.estimated_cache_bytes / 2**30:.2f} GiB with only "
                    f"{free_bytes / 2**30:.2f} GiB free"
                )
        self.cache = RaggedStaticCache(
            config,
            batch_size=self.batch_size,
            max_cache_len=cache_length,
            dtype=torch.bfloat16,
            device=device,
        )
        self.valid_keys = torch.zeros(
            (self.batch_size, cache_length), dtype=torch.bool, device=device
        )
        self.stop_tensor = torch.tensor(stop_ids, device=device)
        self.inactive_token = int(config.pad_token_id)
        self.inactive_token_tensor = torch.tensor(
            self.inactive_token, device=device
        )

        def cached_forward(
            token_ids: Tensor,
            positions: Tensor,
            attention_mask: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor]:
            self.cache.set_write_positions(positions)
            hidden = policy.cached_hidden(
                token_ids,
                past_key_values=self.cache,
                cache_position=positions[0],
                attention_mask=attention_mask,
                position_ids=positions,
            )
            return hidden, policy.logits(hidden), policy.rollout_values(hidden)

        target_forward = (
            torch.compile(
                cached_forward, mode="reduce-overhead", fullgraph=False
            )
            if compile_decode
            else cached_forward
        )
        self.verify_forward = target_forward
        self.commit_forward = target_forward

    def release_cache(self) -> None:
        self.cache.reset()
        self.valid_keys.zero_()

    def _behavior_probabilities(self, logits: Tensor) -> Tensor:
        return dense_top_p_probabilities(
            logits,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    @staticmethod
    def _raw_logprob(logits: Tensor, tokens: Tensor) -> Tensor:
        logits = logits.float()
        return (
            logits.gather(1, tokens[:, None]).squeeze(1)
            - logits.logsumexp(dim=-1)
        )

    def _prefill(
        self, prompt_ids_cpu: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if len(prompt_ids_cpu) != self.prompts_per_rollout:
            raise ValueError("prompt count differs from speculative batch size")
        expanded_prompts = [
            prompt
            for prompt in prompt_ids_cpu
            for _ in range(self.samples_per_prompt)
        ]
        lengths = torch.tensor(
            [prompt.numel() for prompt in expanded_prompts], dtype=torch.long
        )
        if int(lengths.min()) < 1 or int(lengths.max()) >= self.cache_length:
            raise ValueError("speculative prompt lengths are invalid")
        device = self.valid_keys.device
        lengths = lengths.to(device)
        width = int(lengths.max())
        pad = int(self.policy.causal_lm.config.pad_token_id)
        inputs = torch.full(
            (self.batch_size, width), pad, dtype=torch.long, device=device
        )
        for row, prompt in enumerate(expanded_prompts):
            inputs[row, : prompt.numel()].copy_(prompt.to(device))
        positions = torch.arange(width, device=device)[None].expand(
            self.batch_size, -1
        )
        self.valid_keys.zero_()
        self.valid_keys[:, :width] = positions < lengths[:, None]
        mask = ragged_causal_mask(self.valid_keys, positions)
        with torch.autocast(
            device_type=self.device_type,
            dtype=torch.bfloat16,
            enabled=self.device_type == "cuda",
        ):
            hidden, logits, values = self.commit_forward(inputs, positions, mask)
        rows = torch.arange(self.batch_size, device=device)
        last = lengths - 1
        return hidden[rows, last], logits[rows, last], values[rows, last], lengths

    @torch.inference_mode()
    def generate_prompts(
        self,
        prompt_ids_cpu: Sequence[Tensor],
        *,
        max_new_tokens: int,
        progress_callback: Callable[[int, dict[str, float | int]], None] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, int, float, NextLatDecodeStats]:
        validate_thinking_budget(
            self.answer_reserve_tokens, self.thinking_end_token_id, max_new_tokens
        )
        thinking_boundary = max_new_tokens - self.answer_reserve_tokens - 1
        if max_new_tokens < 1:
            raise ValueError("generation length must be positive")
        longest_prompt = max(
            (int(prompt.numel()) for prompt in prompt_ids_cpu),
            default=0,
        )
        if longest_prompt + max_new_tokens > self.cache_length:
            raise ValueError(
                "prompt plus generation length exceeds the NextLat cache"
            )
        started = time.perf_counter()
        device = self.valid_keys.device
        gpu_started = (
            torch.cuda.Event(enable_timing=True)
            if self.device_type == "cuda"
            else None
        )
        prefill_complete = (
            torch.cuda.Event(enable_timing=True)
            if self.device_type == "cuda"
            else None
        )
        decode_complete = (
            torch.cuda.Event(enable_timing=True)
            if self.device_type == "cuda"
            else None
        )
        if gpu_started is not None:
            gpu_started.record()
        hidden, logits, state_values, cursors = self._prefill(prompt_ids_cpu)
        if prefill_complete is not None:
            prefill_complete.record()
        responses = torch.full(
            (self.batch_size, max_new_tokens),
            self.primary_stop,
            dtype=torch.long,
            device=device,
        )
        logprobs = torch.zeros(
            (self.batch_size, max_new_tokens), dtype=torch.float32, device=device
        )
        values = torch.zeros_like(logprobs)
        response_lengths = torch.zeros(
            self.batch_size, dtype=torch.long, device=device
        )
        finished = torch.zeros(
            self.batch_size, dtype=torch.bool, device=device
        )
        thinking_closed = torch.zeros_like(finished)
        target_calls = 0
        target_positions = 0
        row_cycles = torch.zeros((), dtype=torch.long, device=device)
        proposed_by_position = torch.zeros(
            self.draft_length, dtype=torch.long, device=device
        )
        accepted_by_position = torch.zeros_like(proposed_by_position)
        proposed = torch.zeros((), dtype=torch.long, device=device)
        accepted = torch.zeros_like(proposed)
        accepted_pos2plus = torch.zeros_like(proposed)
        proposed_pos2plus = torch.zeros_like(proposed)
        cycles = 0
        pending_tokens = torch.full_like(
            response_lengths, self.inactive_token
        )
        pending_base_hidden = hidden.clone()
        has_pending = False

        while True:
            remaining = max_new_tokens - response_lengths
            active, minimum_remaining = torch.stack(
                (
                    (~finished).any().to(remaining.dtype),
                    remaining.masked_fill(finished, self.draft_length + 1).min(),
                )
            ).tolist()
            if not active:
                break
            row_cycles += (~finished).sum()
            steps = min(self.draft_length, minimum_remaining)
            cycles += 1

            draft_tokens: list[Tensor] = []
            draft_probabilities: list[Tensor] = []
            with torch.autocast(
                device_type=self.device_type,
                dtype=torch.bfloat16,
                enabled=self.device_type == "cuda",
            ):
                if has_pending:
                    draft_state = self.policy.nextlat_hidden(
                        pending_base_hidden, pending_tokens
                    )
                    draft_logits = self.policy.logits(draft_state)
                else:
                    draft_state = hidden
                    draft_logits = logits
                for _ in range(steps):
                    q = self._behavior_probabilities(draft_logits)
                    token = torch.multinomial(q, 1).squeeze(1)
                    token = torch.where(
                        finished, self.inactive_token_tensor, token
                    )
                    draft_tokens.append(token)
                    draft_probabilities.append(q)
                    draft_state = self.policy.nextlat_hidden(draft_state, token)
                    draft_logits = self.policy.logits(draft_state)
            drafts = torch.stack(draft_tokens, dim=1)
            offsets = torch.arange(steps, device=device)
            verify_positions = cursors[:, None] + offsets[None]
            active_drafts = (
                (~finished)[:, None] & (offsets[None] < remaining[:, None])
            )
            if has_pending:
                target_tokens = torch.cat((pending_tokens[:, None], drafts), dim=1)
                target_positions_tensor = torch.cat(
                    ((cursors - 1)[:, None], verify_positions), dim=1
                )
                active_inputs = torch.cat(
                    ((~finished)[:, None], active_drafts), dim=1
                )
                prefix_steps = 1
            else:
                target_tokens = drafts
                target_positions_tensor = verify_positions
                active_inputs = active_drafts
                prefix_steps = 0
            safe_positions = torch.where(
                active_inputs,
                target_positions_tensor,
                (cursors - 1).clamp_min(0)[:, None],
            )
            self.valid_keys.scatter_(1, safe_positions, active_inputs)
            verify_mask = ragged_causal_mask(self.valid_keys, safe_positions)
            with torch.autocast(
                device_type=self.device_type,
                dtype=torch.bfloat16,
                enabled=self.device_type == "cuda",
            ):
                verify_hidden, verify_logits, verify_values = self.verify_forward(
                    target_tokens, safe_positions, verify_mask
                )
            target_calls += 1
            target_positions += steps + prefix_steps

            target_logits: list[Tensor] = []
            target_state_values: list[Tensor] = []
            target_hidden_states: list[Tensor] = []
            target_probabilities: list[Tensor] = []
            for offset in range(steps):
                if prefix_steps == 0 and offset == 0:
                    step_hidden = hidden
                    step_logits = logits
                    step_values = state_values
                    step_probabilities = draft_probabilities[0]
                else:
                    target_index = prefix_steps + offset - 1
                    step_hidden = verify_hidden[:, target_index]
                    step_logits = verify_logits[:, target_index]
                    step_values = verify_values[:, target_index]
                    step_probabilities = self._behavior_probabilities(step_logits)
                target_hidden_states.append(step_hidden)
                target_logits.append(step_logits)
                target_state_values.append(step_values)
                target_probabilities.append(step_probabilities)
            bonus_index = prefix_steps + steps - 1
            bonus_logits = verify_logits[:, bonus_index]
            bonus_probabilities = self._behavior_probabilities(bonus_logits)

            cycle_count = torch.zeros_like(response_lengths)
            rejected = torch.zeros_like(finished)
            cycle_finished = finished.clone()

            for offset in range(steps):
                active = (
                    ~cycle_finished
                    & ~rejected
                    & (cycle_count < remaining)
                )
                committed, accepted_mask = maximal_coupling_verify(
                    target_probabilities[offset],
                    draft_probabilities[offset],
                    drafts[:, offset],
                    active,
                )
                if self.answer_reserve_tokens:
                    committed, forced = force_thinking_end_(
                        committed,
                        response_lengths + cycle_count,
                        thinking_closed,
                        active,
                        thinking_boundary,
                        self.thinking_end_token_id,
                    )
                    # The replacement is pending, not cached: all later verified
                    # states still depend on the original proposal.
                    accepted_mask = accepted_mask & ~forced
                proposed_at_position = active.sum()
                accepted_at_position = accepted_mask.sum()
                proposed_by_position[offset] += proposed_at_position
                accepted_by_position[offset] += accepted_at_position
                proposed += proposed_at_position
                accepted += accepted_at_position
                if offset > 0:
                    proposed_pos2plus += proposed_at_position
                    accepted_pos2plus += accepted_at_position
                output_positions = response_lengths + cycle_count
                safe_output_positions = output_positions.clamp_max(
                    max_new_tokens - 1
                )[:, None]
                responses.scatter_(
                    1,
                    safe_output_positions,
                    torch.where(
                        active,
                        committed,
                        responses.gather(1, safe_output_positions).squeeze(1),
                    )[:, None],
                )
                committed_logprobs = self._raw_logprob(
                    target_logits[offset], committed
                )
                logprobs.scatter_(
                    1,
                    safe_output_positions,
                    torch.where(
                        active,
                        committed_logprobs,
                        logprobs.gather(1, safe_output_positions).squeeze(1),
                    )[:, None],
                )
                committed_values = target_state_values[offset].float()
                values.scatter_(
                    1,
                    safe_output_positions,
                    torch.where(
                        active,
                        committed_values,
                        values.gather(1, safe_output_positions).squeeze(1),
                    )[:, None],
                )
                cycle_count += active.long()
                newly_rejected = active & ~accepted_mask
                pending_tokens = torch.where(
                    newly_rejected, committed, pending_tokens
                )
                pending_base_hidden = torch.where(
                    newly_rejected[:, None],
                    target_hidden_states[offset],
                    pending_base_hidden,
                )
                rejected |= newly_rejected
                emitted_stop = active & (
                    committed[:, None] == self.stop_tensor[None]
                ).any(dim=1)
                cycle_finished |= emitted_stop

            bonus_active = (
                ~cycle_finished
                & ~rejected
                & (cycle_count < remaining)
            )
            bonus = torch.multinomial(bonus_probabilities, 1).squeeze(1)
            output_positions = response_lengths + cycle_count
            if self.answer_reserve_tokens:
                bonus, _ = force_thinking_end_(
                    bonus,
                    output_positions,
                    thinking_closed,
                    bonus_active,
                    thinking_boundary,
                    self.thinking_end_token_id,
                )
            safe_output_positions = output_positions.clamp_max(
                max_new_tokens - 1
            )[:, None]
            responses.scatter_(
                1,
                safe_output_positions,
                torch.where(
                    bonus_active,
                    bonus,
                    responses.gather(1, safe_output_positions).squeeze(1),
                )[:, None],
            )
            bonus_logprobs = self._raw_logprob(bonus_logits, bonus)
            logprobs.scatter_(
                1,
                safe_output_positions,
                torch.where(
                    bonus_active,
                    bonus_logprobs,
                    logprobs.gather(1, safe_output_positions).squeeze(1),
                )[:, None],
            )
            bonus_values = verify_values[:, bonus_index].float()
            values.scatter_(
                1,
                safe_output_positions,
                torch.where(
                    bonus_active,
                    bonus_values,
                    values.gather(1, safe_output_positions).squeeze(1),
                )[:, None],
            )
            cycle_count += bonus_active.long()
            pending_tokens = torch.where(
                bonus_active, bonus, pending_tokens
            )
            pending_base_hidden = torch.where(
                bonus_active[:, None],
                verify_hidden[:, bonus_index],
                pending_base_hidden,
            )
            cycle_finished |= bonus_active & (
                bonus[:, None] == self.stop_tensor[None]
            ).any(dim=1)

            tentative = verify_positions.clamp_max(self.cache_length - 1)
            self.valid_keys.scatter_(
                1, tentative, torch.zeros_like(tentative, dtype=torch.bool)
            )
            committed_offsets = torch.arange(steps + 1, device=device)
            committed_positions = cursors[:, None] + committed_offsets[None]
            committed_mask = committed_offsets[None] < cycle_count[:, None]
            safe_committed = committed_positions.clamp_max(self.cache_length - 1)
            self.valid_keys.scatter_(1, safe_committed, committed_mask)

            cursors += cycle_count
            response_lengths += cycle_count
            finished = cycle_finished | (response_lengths >= max_new_tokens)
            has_pending = True

            if progress_callback is not None:
                elapsed = time.perf_counter() - started
                progress_callback(
                    int(response_lengths.max()),
                    {
                        "decode_steps": int(response_lengths.max()),
                        "batch_rows": self.batch_size,
                        "generated_tokens": int(response_lengths.sum()),
                        "wall_seconds": elapsed,
                        "generated_tokens_per_second": float(
                            response_lengths.sum()
                        )
                        / max(elapsed, 1e-9),
                        "nextlat_cycles": cycles,
                        "nextlat_proposed_tokens": int(proposed),
                        "nextlat_accepted_tokens": int(accepted),
                    },
                )

        width = int(response_lengths.max())
        if decode_complete is not None:
            decode_complete.record()
        response_output = responses[:, :width].cpu()
        logprob_output = logprobs[:, :width].cpu()
        value_output = values[:, :width].cpu()
        prefill_seconds = (
            gpu_started.elapsed_time(prefill_complete) / 1_000.0
            if gpu_started is not None and prefill_complete is not None
            else 0.0
        )
        decode_seconds = (
            prefill_complete.elapsed_time(decode_complete) / 1_000.0
            if prefill_complete is not None and decode_complete is not None
            else 0.0
        )
        return (
            response_output,
            logprob_output,
            value_output,
            int(self.policy.causal_lm.config.vocab_size),
            self.top_p,
            NextLatDecodeStats(
                target_decode_calls=target_calls,
                target_decode_positions=target_positions,
                proposed_tokens=int(proposed),
                accepted_tokens=int(accepted),
                accepted_tokens_pos2plus=int(accepted_pos2plus),
                proposed_tokens_pos2plus=int(proposed_pos2plus),
                speculative_cycles=cycles,
                speculative_row_cycles=int(row_cycles),
                proposed_by_position=tuple(proposed_by_position.tolist()),
                accepted_by_position=tuple(accepted_by_position.tolist()),
                prefill_seconds=prefill_seconds,
                decode_seconds=decode_seconds,
            ),
        )
