from __future__ import annotations

import hashlib
from typing import cast

import torch
from torch import Tensor
from transformers.cache_utils import Cache

from postraining.fast_inference import (
    CapturedTrainingRolloutEngine,
    ContinuousTrainingGeneration,
    PromptPrefixBank,
    _CompactStaticLayer,
)
from postraining.vapo.policy import VAPOPolicy
from postraining.invariant_linear import compile_invariant
from postraining.uno import attach_uno_adapters, load_uno_adapter


def sparse_sampling_support(
    logits: Tensor, *, temperature: float, top_k: int, top_p: float
) -> tuple[Tensor, Tensor]:
    """The production top-k-then-nucleus law, without a dense probability tensor."""
    values, ids = logits.topk(top_k, dim=-1, sorted=True)
    probabilities = (values.float() / temperature).softmax(-1)
    if top_p < 1.0:
        before = probabilities.cumsum(-1) - probabilities
        probabilities = torch.where(before < top_p, probabilities, 0.0)
        probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    return ids, probabilities


def support_probability(ids: Tensor, probabilities: Tensor, tokens: Tensor) -> Tensor:
    """Look up arbitrary tokens in a bounded support in O(k log k), not O(k²)."""
    ordered, order = ids.sort(dim=-1)
    weights = probabilities.gather(-1, order)
    indices = torch.searchsorted(ordered.contiguous(), tokens.contiguous())
    safe_indices = indices.clamp_max(ids.size(-1) - 1)
    found = ordered.gather(-1, safe_indices) == tokens
    return torch.where(found, weights.gather(-1, safe_indices), 0.0)


def sample_support(ids: Tensor, probabilities: Tensor) -> tuple[Tensor, Tensor]:
    shape = probabilities.shape[:-1]
    selected = torch.multinomial(probabilities.reshape(-1, probabilities.size(-1)), 1)
    selected = selected.reshape(*shape, 1)
    return ids.gather(-1, selected).squeeze(-1), probabilities.gather(
        -1, selected
    ).squeeze(-1)


def couple_proposals(
    target_ids: Tensor,
    target_probabilities: Tensor,
    proposal_ids: Tensor,
    proposal_probabilities: Tensor,
    proposals: Tensor,
    proposal_mass: Tensor,
    uniforms: Tensor,
) -> tuple[Tensor, Tensor]:
    """Algorithm 1 prefix acceptance and exact residual/bonus support.

    The final target position is the bonus law. Rejection residuals live only
    on target support: tokens outside it have max(p-q, 0) == 0.
    """
    p_at_proposal = support_probability(
        target_ids[:, :-1], target_probabilities[:, :-1], proposals[..., None]
    ).squeeze(-1)
    accepted = uniforms * proposal_mass < p_at_proposal
    prefix = accepted.long().cumprod(dim=-1)
    count = prefix.sum(-1)
    row = torch.arange(proposals.size(0), device=proposals.device)
    chosen_ids = target_ids[row, count]
    chosen_p = target_probabilities[row, count]
    rejection_position = count.clamp_max(proposals.size(1) - 1)
    chosen_q = support_probability(
        proposal_ids[row, rejection_position],
        proposal_probabilities[row, rejection_position],
        chosen_ids,
    )
    residual = (chosen_p - chosen_q).clamp_min(0)
    bonus = count == proposals.size(1)
    # Unused residuals may have zero mass (notably p == q); bonus uses p.
    weights = torch.where(bonus[:, None], chosen_p, residual)
    weights = weights / weights.sum(-1, keepdim=True)
    correction, _ = sample_support(chosen_ids, weights)
    return prefix, correction


def commit_cycle_(
    generated: Tensor,
    output_position: Tensor,
    sequence_lengths: Tensor,
    pending: Tensor,
    active: Tensor,
    response_limit: Tensor,
    stop_ids: Tensor,
    free_token: Tensor,
    proposals: Tensor,
    accepted_prefix: Tensor,
    correction: Tensor,
    *,
    thinking_closed: Tensor | None = None,
    answer_reserve_tokens: int = 0,
    thinking_end_token_id: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Commit a clean emitted prefix, leaving its final token pending.

    sequence_lengths counts clean KV *excluding* pending both on entry and
    exit. Draft/verification scratch beyond the new cursor is never committed.
    Fixed-width masked writes cannot alias a truncated token onto a valid one.
    """
    count = accepted_prefix.sum(-1)
    candidates = torch.cat((free_token[:, None], proposals, correction[:, None]), dim=1)
    candidates.scatter_(1, (count + 1)[:, None], correction[:, None])
    offsets = torch.arange(candidates.size(1), device=candidates.device)
    eligible = offsets[None] < (count + 2)[:, None]
    eligible &= offsets[None] < (response_limit - output_position)[:, None]
    eligible &= active[:, None]
    if answer_reserve_tokens:
        if thinking_closed is None or thinking_end_token_id is None:
            raise ValueError("thinking budget requires per-lane state and token id")
        closes = (candidates == thinking_end_token_id) & eligible
        prior_close = closes.long().cumsum(-1) - closes.long()
        boundary = response_limit - answer_reserve_tokens - 1
        forced = (
            eligible
            & ~thinking_closed[:, None]
            & (prior_close == 0)
            & (output_position[:, None] + offsets[None] == boundary[:, None])
        )
        candidates.masked_fill_(forced, thinking_end_token_id)
        # Verification after a replacement used the wrong prefix. Retain only
        # the clean prefix and leave the delimiter pending for the next cycle.
        prior_forced = forced.long().cumsum(-1) - forced.long()
        eligible &= prior_forced == 0
    stop = (candidates[..., None] == stop_ids).any(-1) & eligible
    # Keep the first EOS, but no position after it.
    before_stop = stop.long().cumsum(-1) - stop.long()
    emit = eligible & (before_stop == 0)
    if answer_reserve_tokens:
        thinking_closed.logical_or_(
            ((candidates == thinking_end_token_id) & emit).any(-1)
        )
    emitted = emit.sum(-1)
    for offset in range(candidates.size(1)):
        destination = (output_position + offset).clamp_max(generated.size(1) - 1)[
            :, None
        ]
        previous = generated.gather(1, destination)
        value = torch.where(
            emit[:, offset, None], candidates[:, offset, None], previous
        )
        generated.scatter_(1, destination, value)
    last = candidates.gather(1, (emitted - 1).clamp_min(0)[:, None]).squeeze(1)
    pending.copy_(torch.where(active, last, pending))
    sequence_lengths.add_(emitted)
    output_position.add_(emitted)
    active.logical_and_(~stop.any(-1) & (output_position < response_limit))
    accepted = emit[:, 1:-1] & accepted_prefix.bool()
    if answer_reserve_tokens:
        accepted &= ~forced[:, 1:-1]
    return emitted, accepted


class _UnoStaticLayer(_CompactStaticLayer):
    """Explicit cached suffix append; prefill uses separate ordinary caches."""

    def update(self, key_states: Tensor, value_states: Tensor, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        width = key_states.size(-2)
        positions = self.sequence_lengths[:, None] + torch.arange(
            width, device=key_states.device
        )
        keys, values = self._append_suffix(key_states, value_states, positions)
        self.cumulative_length.copy_(self.sequence_lengths.max() + width)
        return keys, values


class UnoTrainingRolloutEngine(CapturedTrainingRolloutEngine):
    """Opt-in Uno Algorithm 1 over the current merged actor, with frozen draft LoRA.

    Each graph cycle drafts [pending, uniform noise], then verifies
    [free target sample, independent proposals] with the adapter disabled.
    Only verified clean KV is retained. The inherited continuous-refill pool
    remains the sole scheduler; its unit is a cycle rather than an AR token.
    """

    invariant_decode = True

    def __init__(
        self,
        source_policy: VAPOPolicy,
        *,
        uno_checkpoint: str,
        uno_block_size: int = 4,
        **kwargs,
    ) -> None:
        if uno_block_size < 2:
            raise ValueError("Uno block size must be at least two")
        if not kwargs.get("compile_decode", False):
            raise ValueError("Uno requires compiled, graph-captured rollout")
        if not uno_checkpoint:
            raise ValueError("Uno requires an explicitly trained adapter checkpoint")
        self.uno_checkpoint = str(uno_checkpoint)
        self.uno_block_size = int(uno_block_size)
        self._output_slots_per_cycle = self.uno_block_size + 1
        kwargs["invariant_decode"] = True
        super().__init__(source_policy, **kwargs)
        self.estimated_cache_bytes = (
            self.estimated_cache_bytes * self._scratch_cache_length // self.cache_length
        )
        bank, metadata = load_uno_adapter(
            uno_checkpoint,
            self.policy.causal_lm,
            model_id=source_policy.model_id,
            revision=source_policy.revision,
        )
        bank.requires_grad_(False)
        bank.eval()
        self.uno_router = attach_uno_adapters(self.policy.causal_lm, bank)
        self.uno_router.set_gate(None)
        with open(uno_checkpoint, "rb") as checkpoint:
            digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
        self.uno_metadata = {
            key: metadata[key]
            for key in (
                "schema",
                "model_id",
                "revision",
                "model_config",
                "uno_config",
                "trained_tokens",
                "step",
                "teacher_sha256",
                "training",
            )
        }
        self.uno_metadata["adapter_sha256"] = digest
        self.uno_metadata["rollout_arithmetic"] = self.arithmetic
        self.last_uno_metrics: dict[str, float | int] = {}
        self._block_offsets = torch.arange(
            self.uno_block_size, device=self._runtime_device
        )
        self._draft_gate = torch.ones(
            (1, self.uno_block_size, 1), device=self._runtime_device
        )
        self._draft_gate[:, 0] = 0
        self._cycle_metrics = torch.zeros(
            2 + 5 * (self.uno_block_size - 1),
            dtype=torch.long,
            device=self._runtime_device,
        )
        self._capture_forward_calls = 0
        # Gate-zero positions must preserve the clean actor's bf16 rounding.
        # Otherwise Inductor's fusion changes their logits and retained KV.
        self._draft_forward = compile_invariant(self._block_forward)
        self._verify_forward = compile_invariant(self._block_forward)
        self._support = torch.compile(self._sampling_support, fullgraph=True)
        self._couple = torch.compile(couple_proposals, fullgraph=True)
        self._sample_support = torch.compile(sample_support, fullgraph=True)
        self._commit = torch.compile(commit_cycle_, fullgraph=True)

    @property
    def _scratch_cache_length(self) -> int:
        return self.cache_length + self.uno_block_size + 1

    def _new_cache(self) -> Cache:
        return Cache(
            layers=[
                _UnoStaticLayer(
                    self._scratch_cache_length,
                    self.sequence_lengths,
                    self.attention_mask,
                )
                for _ in range(self._num_hidden_layers)
            ]
        )

    def _bind_flash_cache(self) -> None:
        super()._bind_flash_cache()
        for layer in self.policy.causal_lm.model.layers:
            layer.self_attn._rollout_max_cache_len = self._scratch_cache_length

    def _build_prompt_prefix_bank(self, prompt_ids_cpu, **kwargs) -> PromptPrefixBank:
        self.uno_router.set_gate(None)
        return super()._build_prompt_prefix_bank(prompt_ids_cpu, **kwargs)

    def _sampling_support(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        return sparse_sampling_support(
            logits, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p
        )

    def _block_forward(self, tokens: Tensor) -> Tensor:
        positions = self.sequence_lengths[:, None] + self._block_offsets[None]
        hidden = self.policy.cached_hidden(
            tokens,
            past_key_values=self.cache,
            cache_position=self._block_offsets,
            attention_mask=None,
            position_ids=positions,
        )
        return self.policy.logits(hidden)

    def _continuous_split_decode_step(self) -> None:
        # Inactive lanes start at zero and write every KV position they can read.
        self.sequence_lengths.masked_fill_(~self.active, 0)
        original_cursor = self.sequence_lengths.clone()
        active_before = self.active.clone()
        noise = torch.randint(
            int(self.policy.causal_lm.config.vocab_size),
            (self.batch_size, self.uno_block_size - 1),
            device=self._runtime_device,
        )
        draft_tokens = torch.cat((self._pending[:, None], noise), dim=1)
        self.flash_sequence_lengths.copy_(
            (self.sequence_lengths + self.uno_block_size).to(torch.int32)
        )
        self.uno_router.set_gate(self._draft_gate)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            draft_ids, draft_p = self._support(self._draft_forward(draft_tokens))
        draft, draft_mass = self._sample_support(draft_ids, draft_p)
        free, proposals = draft[:, 0], draft[:, 1:]
        # First draft position is exactly AR: gate=0 and no future attention.
        # Keep its KV, discard the noisy suffix by overwriting it in verification.
        self.sequence_lengths.add_(1)
        self.flash_sequence_lengths.add_(1)
        self.uno_router.set_gate(None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            target_ids, target_p = self._support(self._verify_forward(draft))
        prefix, correction = self._couple(
            target_ids,
            target_p,
            draft_ids[:, 1:],
            draft_p[:, 1:],
            proposals,
            draft_mass[:, 1:],
            torch.rand_like(draft_mass[:, 1:]),
        )
        self.sequence_lengths.copy_(original_cursor)
        emitted, accepted = self._commit(
            self.generated,
            self.output_position,
            self.sequence_lengths,
            self._pending,
            self.active,
            self.response_limit,
            self.stop_tensor,
            free,
            proposals,
            prefix,
            correction,
            thinking_closed=self.thinking_closed,
            answer_reserve_tokens=self.answer_reserve_tokens,
            thinking_end_token_id=self.thinking_end_token_id,
        )
        self.position_ids[:, 0].copy_(self.sequence_lengths)
        self._cycle_metrics[0].add_(active_before.sum())
        self._cycle_metrics[1].add_(emitted.sum())
        positions = self.uno_block_size - 1
        accepted_prefix = prefix.bool() & active_before[:, None]
        reached = torch.cat((active_before[:, None], accepted_prefix[:, :-1]), dim=1)
        useful_reached = reached & (self._block_offsets[None, 1:] < emitted[:, None])
        self._cycle_metrics[2 : 2 + positions].add_(accepted_prefix.sum(0))
        self._cycle_metrics[2 + positions : 2 + 2 * positions].add_(active_before.sum())
        self._cycle_metrics[2 + 2 * positions : 2 + 3 * positions].add_(reached.sum(0))
        self._cycle_metrics[2 + 3 * positions : 2 + 4 * positions].add_(accepted.sum(0))
        self._cycle_metrics[2 + 4 * positions :].add_(useful_reached.sum(0))

    def _capture_continuous_decode_schedule(self) -> None:
        # Capture is deliberately confined to the pre-admission boundary.
        if bool(self.active.any()):
            raise RuntimeError("Uno graph capture requires inactive lanes")
        state = [
            self.output_position,
            self.sequence_lengths,
            self.flash_sequence_lengths,
            self.position_ids,
            self.active,
            self.thinking_closed,
            self.generated,
            self._pending,
            self._cycle_metrics,
        ]
        state.extend(
            cast(_CompactStaticLayer, layer).cumulative_length
            for layer in self.cache.layers
        )
        snapshots = [tensor.clone() for tensor in state]
        # Capture uses exclusively initialized inactive scratch, never live prefix.
        self.active.zero_()
        try:
            super()._capture_continuous_decode_schedule()
        finally:
            self.uno_router.set_gate(None)
            for tensor, snapshot in zip(state, snapshots, strict=True):
                tensor.copy_(snapshot)
        self._capture_forward_calls += 4

    @torch.inference_mode()
    def generate_prompt_pool(
        self, prompt_ids_cpu, **kwargs
    ) -> ContinuousTrainingGeneration:
        self._cycle_metrics.zero_()
        self._capture_forward_calls = 0
        result = super().generate_prompt_pool(prompt_ids_cpu, **kwargs)
        counts = self._cycle_metrics.cpu().tolist()
        block = self.uno_block_size
        calls = result.decode_steps
        metrics: dict[str, float | int] = {
            "uno_cycles": calls,
            "uno_active_row_cycles": counts[0],
            "uno_tau": counts[1] / max(counts[0], 1),
            "uno_target_decode_calls": calls,
            "uno_target_decode_positions": calls * self.batch_size * block,
            "uno_draft_decode_calls": calls,
            "uno_draft_decode_positions": calls * self.batch_size * block,
            "uno_model_forward_calls": 2 * calls + self._capture_forward_calls,
            "uno_capture_model_forward_calls": self._capture_forward_calls,
            "uno_attempted_output_tokens": result.capacity_row_steps,
            "uno_useful_output_tokens": result.useful_tokens,
            "uno_decode_seconds": result.decode_seconds,
        }
        # Draft positions 1..B-1 correspond to emitted positions 2..B.
        positions = block - 1
        for position in range(positions):
            accepted = counts[2 + position]
            reached = counts[2 + 2 * positions + position]
            metrics[f"uno_accepted_position_{position + 1}"] = accepted
            metrics[f"uno_proposed_position_{position + 1}"] = counts[
                2 + positions + position
            ]
            metrics[f"uno_reached_position_{position + 1}"] = reached
            metrics[f"uno_conditional_acceptance_position_{position + 1}"] = (
                accepted / max(reached, 1)
            )
            metrics[f"uno_emitted_accepted_position_{position + 1}"] = counts[
                2 + 3 * positions + position
            ]
            metrics[f"uno_useful_reached_position_{position + 1}"] = counts[
                2 + 4 * positions + position
            ]
        self.last_uno_metrics = metrics
        return result

    def generate_prompts(self, *args, **kwargs):
        raise ValueError(
            "Uno uses generate_prompt_pool; exact behavior statistics are refreshed by replay"
        )

    def release_cache(self) -> None:
        self.uno_router.set_gate(None)
        super().release_cache()
