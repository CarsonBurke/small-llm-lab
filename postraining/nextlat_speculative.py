"""Exact per-row NextLat speculative decoding with a ragged static KV cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence
import time

import torch
from torch import Tensor

from postraining.hf_vapo import (
    MiniCPMVAPOPolicy,
    dense_top_p_probabilities,
    maximal_coupling_verify,
)


@dataclass(frozen=True)
class NextLatDecodeStats:
    target_decode_calls: int = 0
    target_decode_positions: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    accepted_tokens_pos2plus: int = 0
    speculative_cycles: int = 0


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
        policy: MiniCPMVAPOPolicy,
        *,
        stop_ids: tuple[int, ...],
        prompts_per_rollout: int,
        samples_per_prompt: int,
        cache_length: int,
        draft_length: int,
        temperature: float,
        top_p: float,
        compile_decode: bool,
    ) -> None:
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
        self.draft_length = draft_length
        self.temperature = temperature
        self.top_p = top_p
        self.stop_ids = stop_ids
        self.primary_stop = stop_ids[0]
        device = next(policy.parameters()).device
        config: Any = policy.causal_lm.config
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
            return hidden, policy.logits(hidden), policy.critic(hidden)

        if compile_decode:
            self.verify_forward = torch.compile(
                cached_forward, mode="reduce-overhead", fullgraph=False
            )
            self.commit_forward = torch.compile(
                cached_forward, mode="reduce-overhead", fullgraph=False
            )
        else:
            self.verify_forward = cached_forward
            self.commit_forward = cached_forward

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
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
        started = time.perf_counter()
        device = self.valid_keys.device
        hidden, logits, state_values, cursors = self._prefill(prompt_ids_cpu)
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
        target_calls = 0
        target_positions = 0
        proposed = 0
        accepted = 0
        accepted_pos2plus = 0
        cycles = 0

        while not bool(finished.all()):
            remaining = max_new_tokens - response_lengths
            if int(remaining.max()) <= 0:
                break
            steps = min(
                self.draft_length,
                int(remaining[~finished].min()),
            )
            cycles += 1

            draft_tokens: list[Tensor] = []
            draft_probabilities: list[Tensor] = []
            draft_state = hidden
            draft_logits = logits
            for _ in range(steps):
                q = self._behavior_probabilities(draft_logits)
                token = torch.multinomial(q, 1).squeeze(1)
                token = torch.where(finished, self.stop_tensor[0], token)
                draft_tokens.append(token)
                draft_probabilities.append(q)
                draft_state = self.policy.nextlat_hidden(draft_state, token)
                draft_logits = self.policy.logits(draft_state)
            drafts = torch.stack(draft_tokens, dim=1)
            offsets = torch.arange(steps, device=device)
            verify_positions = cursors[:, None] + offsets[None]
            active_drafts = (~finished)[:, None] & (offsets[None] < remaining[:, None])
            safe_positions = torch.where(
                active_drafts,
                verify_positions,
                (cursors - 1).clamp_min(0)[:, None],
            )
            self.valid_keys.scatter_(1, safe_positions, active_drafts)
            verify_mask = ragged_causal_mask(self.valid_keys, safe_positions)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                verify_hidden, verify_logits, verify_values = self.verify_forward(
                    drafts, safe_positions, verify_mask
                )
            target_calls += 1
            target_positions += steps

            target_logits = [logits]
            target_state_values = [state_values]
            for offset in range(1, steps):
                target_logits.append(verify_logits[:, offset - 1])
                target_state_values.append(verify_values[:, offset - 1])
            target_probabilities = [
                self._behavior_probabilities(step_logits)
                for step_logits in target_logits
            ]
            bonus_probabilities = self._behavior_probabilities(
                verify_logits[:, steps - 1]
            )

            cycle_count = torch.zeros_like(response_lengths)
            rejected = torch.zeros_like(finished)
            cycle_finished = finished.clone()
            final_tokens = torch.full_like(response_lengths, self.primary_stop)
            final_positions = (cursors - 1).clamp_min(0)
            final_required = torch.zeros_like(finished)

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
                proposed += int(active.sum())
                accepted += int(accepted_mask.sum())
                if offset > 0:
                    accepted_pos2plus += int(accepted_mask.sum())
                output_positions = response_lengths + cycle_count
                rows = active.nonzero(as_tuple=False).squeeze(1)
                if rows.numel():
                    responses[rows, output_positions[rows]] = committed[rows]
                    logprobs[rows, output_positions[rows]] = self._raw_logprob(
                        target_logits[offset][rows], committed[rows]
                    )
                    values[rows, output_positions[rows]] = target_state_values[
                        offset
                    ][rows]
                cycle_count += active.long()
                newly_rejected = active & ~accepted_mask
                final_tokens[newly_rejected] = committed[newly_rejected]
                final_positions[newly_rejected] = cursors[newly_rejected] + offset
                final_required |= newly_rejected
                rejected |= newly_rejected
                emitted_stop = active & (
                    committed[:, None] == self.stop_tensor[None]
                ).any(dim=1)
                cycle_finished |= emitted_stop
                if not bool((~cycle_finished & ~rejected).any()):
                    break

            bonus_active = (
                ~cycle_finished
                & ~rejected
                & (cycle_count < remaining)
            )
            if bool(bonus_active.any()):
                bonus = torch.multinomial(bonus_probabilities, 1).squeeze(1)
                output_positions = response_lengths + cycle_count
                rows = bonus_active.nonzero(as_tuple=False).squeeze(1)
                responses[rows, output_positions[rows]] = bonus[rows]
                logprobs[rows, output_positions[rows]] = self._raw_logprob(
                    verify_logits[rows, steps - 1], bonus[rows]
                )
                values[rows, output_positions[rows]] = verify_values[
                    rows, steps - 1
                ]
                cycle_count += bonus_active.long()
                final_tokens[bonus_active] = bonus[bonus_active]
                final_positions[bonus_active] = cursors[bonus_active] + steps
                final_required |= bonus_active
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

            safe_final_positions = torch.where(
                final_required,
                final_positions,
                (cursors + cycle_count - 1).clamp_min(0),
            )
            safe_final_tokens = torch.where(
                final_required,
                final_tokens,
                responses[
                    torch.arange(self.batch_size, device=device),
                    (response_lengths + cycle_count - 1).clamp_min(0),
                ],
            )
            commit_mask = ragged_causal_mask(
                self.valid_keys, safe_final_positions[:, None]
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                committed_hidden, committed_logits, committed_values = (
                    self.commit_forward(
                        safe_final_tokens[:, None],
                        safe_final_positions[:, None],
                        commit_mask,
                    )
                )
            target_calls += 1
            target_positions += 1
            hidden = committed_hidden[:, 0]
            logits = committed_logits[:, 0]
            state_values = committed_values[:, 0]
            cursors += cycle_count
            response_lengths += cycle_count
            finished = cycle_finished | (response_lengths >= max_new_tokens)

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
                        "nextlat_proposed_tokens": proposed,
                        "nextlat_accepted_tokens": accepted,
                    },
                )

        width = int(response_lengths.max())
        return (
            responses[:, :width].cpu(),
            logprobs[:, :width].cpu(),
            values[:, :width].cpu(),
            int(self.policy.causal_lm.config.vocab_size),
            self.top_p,
            NextLatDecodeStats(
                target_decode_calls=target_calls,
                target_decode_positions=target_positions,
                proposed_tokens=proposed,
                accepted_tokens=accepted,
                accepted_tokens_pos2plus=accepted_pos2plus,
                speculative_cycles=cycles,
            ),
        )
