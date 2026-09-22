"""Lockstep rollout backend for the nano/KDA trunks.

The Hugging Face engine in ``postraining.fast_inference`` is a continuous
batching machine built on ``transformers`` static caches, fused Llama
projections, and measured FA4 split-KV decode kernels. None of that transfers
to the nano backbones, which own their own paged cache and attention step. So
this is a second backend behind the same interface rather than a port: it
produces the identical :class:`ContinuousTrainingGeneration` currency, and the
trainer cannot tell the two apart.

What it does:

* one left-padded prefill for the whole pool, filling every layer cache;
* lockstep decode over a fixed physical batch, attending only the live cache
  prefix through the backbone's own 2-D key-mask step;
* exact log-probabilities of each sampled token under the untruncated model
  distribution, so a step-zero importance ratio is one by construction;
* per-row response limits and stop tokens, with finished rows frozen.

What it deliberately does not do: continuous refill of finished lanes. The
pool decodes in one wave, so a pool whose rows finish at very different
lengths wastes the tail. That is a throughput property, not a correctness
one, and it has not been measured here — do not quote a speed for this engine
without running one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import time
from typing import Any

import torch
from torch import Tensor

from postraining.core import top_p_sample
from postraining.thinking_budget import validate_thinking_budget
from postraining.vapo.model.nano import NanoTrunk
from postraining.vapo.model.protocols import Capability
from postraining.vapo.rollout.results import (
    ContinuousTrainingGeneration,
    response_token_limits,
)


class NanoRolloutEngine:
    """Sampling backend over a :class:`NanoTrunk`.

    The engine holds the trunk it samples from; there is no separate merged
    inference replica, because a nano actor has no adapters to merge. It does
    hold the decode cache, which :meth:`release_cache` frees before the
    trainer allocates update activations.
    """

    def __init__(
        self,
        trunk: NanoTrunk,
        *,
        batch_size: int,
        cache_length: int,
        stop_ids: Sequence[int],
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        answer_reserve_tokens: int = 0,
        thinking_end_token_id: int | None = None,
        seed: int | None = None,
        cache_dtype: torch.dtype | None = None,
    ) -> None:
        trunk.require(Capability.PAGED_KV_CACHE)
        if batch_size < 1 or cache_length < 2:
            raise ValueError("rollout geometry must be positive")
        if top_k > trunk.vocab_size:
            raise ValueError("top-k exceeds the model vocabulary")
        self._trunk = trunk
        self.batch_size = int(batch_size)
        self.cache_length = int(cache_length)
        self.stop_ids = tuple(int(token) for token in stop_ids)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.answer_reserve_tokens = int(answer_reserve_tokens)
        self.thinking_end_token_id = thinking_end_token_id
        self.cache_dtype = cache_dtype
        self._device = trunk.device
        self._generator = None
        if seed is not None:
            self._generator = torch.Generator(device=self._device).manual_seed(int(seed))
        self._caches: list[tuple[Tensor, Tensor]] | None = None
        self.estimated_cache_bytes = trunk.geometry.kv_cache_bytes(
            batch_size=self.batch_size,
            cache_length=self.cache_length,
            element_size=torch.empty((), dtype=cache_dtype or torch.bfloat16).element_size(),
        )

    # -- engine interface --------------------------------------------------

    @property
    def trunk(self) -> NanoTrunk:
        return self._trunk

    def prepare_generation(self) -> None:
        """Put the trunk in evaluation mode and allocate the decode cache.

        A nano actor samples from its own weights, so there is nothing to
        refresh from; the call exists so the trainer's rollout phase is the
        same for both backends.
        """
        self._trunk.module.eval()
        if self._caches is None:
            self._caches = self._new_cache()

    def release_cache(self) -> None:
        self._caches = None

    def _new_cache(self) -> list[tuple[Tensor, Tensor]]:
        caches = self._trunk.new_kv_cache(
            batch_size=self.batch_size,
            cache_length=self.cache_length,
            device=self._device,
            dtype=self.cache_dtype,
        )
        # make_generation_cache allocates uninitialized storage. Masked slots
        # are never read by the narrowed decode step, but a prefill row's
        # padded prefix is, so start from finite zeros.
        for keys, values in caches:
            keys.zero_()
            values.zero_()
        return caches

    # -- generation --------------------------------------------------------

    @torch.inference_mode()
    def generate_prompt_pool(
        self,
        prompt_ids_cpu: Sequence[Tensor],
        *,
        max_new_tokens: int,
        context_tokens: int | None = None,
        progress_callback: Callable[[int, dict[str, float | int]], None] | None = None,
        **unsupported: Any,
    ) -> ContinuousTrainingGeneration:
        """Sample one response per prompt, in prompt order."""
        if unsupported:
            raise TypeError(
                "the nano engine does not implement "
                f"{', '.join(sorted(unsupported))}; it decodes one wave without refill"
            )
        validate_thinking_budget(
            self.answer_reserve_tokens, self.thinking_end_token_id, max_new_tokens
        )
        if not prompt_ids_cpu:
            raise ValueError("continuous prompt pool cannot be empty")
        if len(prompt_ids_cpu) > self.batch_size:
            raise ValueError(
                f"{len(prompt_ids_cpu)} prompts exceed the {self.batch_size} "
                "physical rows this engine decodes in one wave"
            )
        prompt_lengths = [int(prompt.numel()) for prompt in prompt_ids_cpu]
        limits = response_token_limits(
            prompt_lengths,
            max_new_tokens=max_new_tokens,
            context_tokens=context_tokens,
            answer_reserve_tokens=self.answer_reserve_tokens,
            thinking_end_token_id=self.thinking_end_token_id,
        )
        prompt_width = max(prompt_lengths)
        if prompt_width + max(limits) > self.cache_length:
            raise ValueError("prompt pool and responses exceed the rollout cache")
        if context_tokens is not None and context_tokens > self.cache_length:
            raise ValueError("context_tokens exceeds the allocated rollout cache")

        self.prepare_generation()
        caches = self._caches
        assert caches is not None
        for keys, values in caches:
            keys.zero_()
            values.zero_()

        rows = len(prompt_ids_cpu)
        lanes = self.batch_size
        device = self._device
        # The cache is allocated once for a fixed lane count, so a short pool
        # decodes on filler lanes rather than a reshaped cache. Left padding
        # keeps every lane's write head at the same absolute cache slot, which
        # is what makes one lockstep position tensor correct for the batch.
        padded = torch.zeros((lanes, prompt_width), dtype=torch.long, device=device)
        key_valid = torch.zeros((lanes, self.cache_length), dtype=torch.bool, device=device)
        for index, prompt in enumerate(prompt_ids_cpu):
            length = prompt_lengths[index]
            padded[index, prompt_width - length :] = prompt.to(
                device=device, dtype=torch.long
            )
            key_valid[index, prompt_width - length : prompt_width] = True
        # A fully masked attention row produces NaN that would contaminate
        # every later layer, so an unused lane keeps exactly one valid key and
        # is simply never read.
        key_valid[rows:, prompt_width - 1] = True

        started = time.perf_counter()
        belief = self._trunk.module.prefill_belief(
            self._trunk.embed_tokens(padded), caches, key_valid[:, :prompt_width]
        )
        logits = self._trunk.readout.logits(belief[:, -1])
        prefill_seconds = time.perf_counter() - started

        limit_row = torch.ones(lanes, dtype=torch.long, device=device)
        limit_row[:rows] = torch.tensor(limits, device=device)
        active = torch.zeros(lanes, dtype=torch.bool, device=device)
        active[:rows] = True
        produced = torch.zeros(lanes, dtype=torch.long, device=device)
        stop_row = (
            torch.tensor(self.stop_ids, device=device)
            if self.stop_ids
            else torch.zeros(0, dtype=torch.long, device=device)
        )
        tokens = torch.zeros((lanes, max(limits)), dtype=torch.long, device=device)
        token_logprobs = torch.zeros((lanes, max(limits)), dtype=torch.float32, device=device)

        started = time.perf_counter()
        steps = 0
        for step in range(max(limits)):
            sampled = top_p_sample(
                logits, self.temperature, self.top_p,
                generator=self._generator,
                top_k=self.top_k if self.top_k > 0 else None,
            ).reshape(lanes)
            exact = torch.log_softmax(logits.float(), dim=-1).gather(
                1, sampled[:, None]
            ).squeeze(1)
            # A finished row keeps emitting into a frozen column it never
            # reads back; masking here is what makes its response length the
            # step it stopped at, not the pool's longest row.
            tokens[:, step] = torch.where(active, sampled, torch.zeros_like(sampled))
            token_logprobs[:, step] = torch.where(active, exact, torch.zeros_like(exact))
            produced = produced + active.long()
            hit_stop = (
                (sampled[:, None] == stop_row[None, :]).any(dim=1)
                if stop_row.numel()
                else torch.zeros_like(active)
            )
            active = active & ~hit_stop & (produced < limit_row)
            steps += 1
            if progress_callback is not None:
                progress_callback(
                    step, {"active_rows": int(active.sum()), "step": step}
                )
            if not bool(active.any()):
                break
            position = torch.tensor(prompt_width + step, device=device)
            key_valid[:, prompt_width + step] = active
            live = key_valid[:, : prompt_width + step + 1]
            hidden = self._trunk.cached_hidden_states(
                input_ids=sampled[:, None],
                past_key_values=caches,
                cache_position=position,
                attention_mask=live,
            )
            logits = self._trunk.readout.logits(hidden[:, -1])
        decode_seconds = time.perf_counter() - started

        lengths = produced.clamp(max=limit_row).tolist()[:rows]
        responses = tuple(
            tokens[index, : lengths[index]].to(dtype=torch.int32, device="cpu")
            for index in range(rows)
        )
        logprob_rows = tuple(
            token_logprobs[index, : lengths[index]].to("cpu") for index in range(rows)
        )
        useful_tokens = int(sum(lengths))
        return ContinuousTrainingGeneration(
            responses=responses,
            logprobs=logprob_rows,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            decode_steps=steps,
            useful_tokens=useful_tokens,
            # Utilization is measured against the lanes the cache pays for,
            # not against the rows this pool happened to fill.
            capacity_row_steps=lanes * steps,
            admission_events=1,
            minimum_active_rows_with_backlog=rows,
            response_limits=tuple(limits),
        )


__all__ = ["NanoRolloutEngine"]
