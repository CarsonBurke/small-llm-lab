"""Counter-coupled B64 controller rollout, with ordinary stock heldout sampling.

Training is exactly four prompts times sixteen samples, without lane refill.
Neighboring logical lanes share gate/answer uniforms and opposite Gaussian
innovations; the independent diagnostic comparator uses distinct lane keys.
Only the categorical inverse-CDF ordering changes: stock filtering is performed
first, then probabilities are traversed in ascending token-ID order.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import Tensor

from postraining import minicpm_latent_rollout as rollout
from postraining.fast_inference import selected_token_logprobs
from postraining.runtime.coupled_rng import coupled_normal, coupled_uniform


def _fixed_label_categorical(
    logits: Tensor, uniform: Tensor, *, temperature: float, top_k: int, top_p: float,
) -> Tensor:
    """Apply stock filtering, then invert its CDF in fixed token-label order.

    Common uniforms must not use probability-rank order: nearly tied token
    probabilities would otherwise swap intervals even for nearly equal laws.
    Excluded/underflowed zeros may occur anywhere after the token-ID sort.
    """
    values, token_ids = logits.topk(top_k, dim=-1, sorted=True)
    probabilities = (values.float() / temperature).softmax(dim=-1)
    if top_p < 1.0:
        preceding_mass = probabilities.cumsum(dim=-1) - probabilities
        probabilities = torch.where(
            preceding_mass < top_p, probabilities,
            torch.zeros((), device=probabilities.device),
        )
        probabilities /= probabilities.sum(dim=-1, keepdim=True)
    token_ids, order = token_ids.sort(dim=-1)
    probabilities = probabilities.gather(1, order)
    # FP64 accumulation and an exact endpoint prevent rounded-up excluded tails.
    # >= implements the half-open intervals and skips every zero-mass plateau.
    mass = probabilities.double().cumsum(dim=-1)
    mass = mass / mass[:, -1:]
    ranks = torch.arange(top_k, device=logits.device)
    last_positive = torch.where(probabilities > 0, ranks, -1).amax(dim=-1)
    mass = torch.where(ranks[None] >= last_positive[:, None], 1.0, mass)
    selected = (uniform.double()[:, None] >= mass).sum(dim=-1)
    selected = torch.minimum(selected, last_positive)
    return token_ids.gather(1, selected[:, None]).squeeze(1)


class _PairedChunks(rollout._LatentChunks):
    def run(self, thinking, answering, steps):
        engine = self.engine
        if set(thinking).intersection(answering):
            raise ValueError("a lane cannot belong to both head phases")
        for lanes, destination in (
            (thinking, engine.thought_indices), (answering, engine.answer_indices),
        ):
            if len(lanes) != len(set(lanes)) or any(not 0 <= lane < 64 for lane in lanes):
                raise ValueError("invalid or duplicate real-lane membership")
            width = min(64, (len(lanes) + engine.head_bucket_size - 1)
                        // engine.head_bucket_size * engine.head_bucket_size)
            # Match the stock scheduler's ordered membership and inert scratch.
            destination.fill_(-1)
            if width:
                destination[:width].copy_(torch.tensor(
                    [*lanes, *range(64, 64 + width - len(lanes))], dtype=torch.int64,
                ))
        super().run(thinking, answering, steps)


class PairedControllerRolloutEngine(rollout.MiniCPMLatentRolloutEngine):
    """Compiled/captured paired training without a second policy replica.

    ``counter_seed=None`` draws a fresh scalar into persistent CUDA storage once
    per training generation, using the existing checkpointed CUDA RNG stream.
    An explicit signed-int64 seed is fixed across generations for diagnostics.
    ``stock_sampling()`` temporarily selects the original stock heads, permits
    ordinary configured pool sizes/refill, and draws no counter seed.
    """

    def __init__(self, policy, *, pair_samples: bool = True,
                 counter_seed: int | None = None, **kwargs):
        if type(pair_samples) is not bool:
            raise ValueError("pair_samples must be boolean")
        if counter_seed is not None and (
            type(counter_seed) is not int or not -(2**63) <= counter_seed < 2**63
        ):
            raise ValueError("counter_seed must be a signed int64 integer or None")
        if kwargs.get("compile_decode", True) is not True:
            raise ValueError("compiled CUDA rollout is mandatory")
        if next(policy.parameters()).device.type != "cuda":
            raise ValueError("paired rollout requires CUDA; there is no CPU fallback")
        super().__init__(policy, **kwargs)
        self.pair_samples = pair_samples
        self.counter_seed = counter_seed
        self._stock_heads = self._thought, self._gate, self._answer
        self._stock_sampling = False
        self._generating = False
        # Never rebind these graph inputs: only their contents change at boundaries.
        self.coupling_seed = torch.tensor(
            0 if counter_seed is None else counter_seed, dtype=torch.int64, device=self._device,
        )
        self.thought_indices = torch.full((64,), -1, dtype=torch.int64, device=self._device)
        self.answer_indices = torch.full_like(self.thought_indices, -1)
        self.lane_prompt_lengths = torch.zeros_like(self.thought_indices)
        self.admission_count = 0
        compile_head = lambda fn: torch.compile(
            fn, fullgraph=True, dynamic=True, options={"emulate_precision_casts": True},
        )
        self._thought, self._gate, self._answer = map(
            compile_head, (self._coupled_thought, self._coupled_gate, self._coupled_answer),
        )

    def _counters(self, indices, rows):
        lanes = indices[:rows]
        real = (lanes >= 0) & (lanes < 64)
        safe = lanes.clamp(0, 63)
        # WAIT/retired lanes do not advance decoder lengths or live-peer counters.
        positions = self._decoder.lengths[safe] - self.lane_prompt_lengths[safe]
        keys = safe // 2 if self.pair_samples else safe
        return lanes, torch.where(real, keys, -1), torch.where(real, positions, -1)

    def _coupled_thought(self, hidden):
        lanes, keys, positions = self._counters(self.thought_indices, hidden.shape[0])
        noise = coupled_normal(self.coupling_seed, keys, positions,
                               dimension=hidden.shape[1], role=1)
        if self.pair_samples:
            sign = torch.where(lanes.remainder(2) == 0, 1.0, -1.0)
            noise = noise * sign[:, None]
        mean = self.policy.transition.predict_mean(hidden)
        sigma = self.policy.transition.predict_log_sigma(hidden)
        raw = self.policy.transition.sample_latent(mean, sigma, noise=noise)
        return (raw, self.policy.transition.log_prob(raw, mean, sigma),
                self.policy.thought_embeddings(raw))

    def _coupled_gate(self, hidden):
        _, keys, positions = self._counters(self.thought_indices, hidden.shape[0])
        uniform = coupled_uniform(self.coupling_seed, keys, positions, role=2)
        probability = self.policy.thinking_gate.stop_logit(hidden).sigmoid()
        decision = (uniform < probability).long()
        return decision, self.policy.thinking_gate.log_prob(decision, hidden)

    def _coupled_answer(self, hidden):
        _, keys, positions = self._counters(self.answer_indices, hidden.shape[0])
        uniform = coupled_uniform(self.coupling_seed, keys, positions, role=3)
        logits = self.policy.logits(hidden)
        delimiters = [self.thinking_start_token_id, self.thinking_end_token_id]
        original = logits[:, delimiters].clone()
        logits[:, delimiters] = -torch.inf
        tokens = _fixed_label_categorical(
            logits, uniform, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
        )
        # Filtering changes sampling, never the original untempered policy score.
        logits[:, delimiters] = original
        return tokens, selected_token_logprobs(logits, tokens), self.policy.token_embeddings(tokens)

    @contextmanager
    def stock_sampling(self):
        """Temporarily use ordinary stock generation; restore mode even on failure."""
        if self._generating:
            raise RuntimeError("cannot switch sampling during an active generation")
        previous_mode = self._stock_sampling
        previous_heads = self._thought, self._gate, self._answer
        self._stock_sampling = True
        self._thought, self._gate, self._answer = self._stock_heads
        try:
            yield self
        finally:
            self._thought, self._gate, self._answer = previous_heads
            self._stock_sampling = previous_mode

    def _prepare_paired_generation(self, prompt_ids, max_new_tokens):
        if (self.prompts_per_rollout, self.samples_per_prompt, self.batch_size) != (4, 16, 64):
            raise ValueError("paired training requires exact B64=4x16/physical64 without refill")
        if self.cache_length != 11024 or self.answer_reserve_tokens != 1024 or max_new_tokens != 10000:
            raise ValueError("paired training requires cache11024/response10000/reserve1024")
        if len(prompt_ids) != 4:
            raise ValueError("paired training requires exactly four prompts")
        lengths = [prompt.numel() for prompt in prompt_ids]
        if any(not 1 <= length <= 1024 for length in lengths):
            raise ValueError("paired prompt length must be in [1, 1024]")
        self.admission_count = 0
        self.thought_indices.fill_(-1)
        self.answer_indices.fill_(-1)
        self.lane_prompt_lengths.copy_(torch.tensor(
            [length for length in lengths for _ in range(16)], dtype=torch.int64,
        ))
        if self.counter_seed is None:
            # One device-side seed draw, no scalar extraction or separate generator.
            self.coupling_seed.random_()

    @contextmanager
    def _sampling_overrides(self):
        previous_chunks = rollout._LatentChunks
        decoder = self._decoder
        owned_admit = "admit" in vars(decoder)
        previous_admit = decoder.admit

        def admit(lanes, prompt_indices):
            if self.admission_count or list(lanes) != list(range(64)):
                raise ValueError("paired training forbids refill or remapped logical lanes")
            if list(prompt_indices) != [lane // 16 for lane in range(64)]:
                raise ValueError("prompt membership does not match neighboring within-prompt pairs")
            self.admission_count += 1
            return previous_admit(lanes, prompt_indices)

        rollout._LatentChunks = _PairedChunks
        decoder.admit = admit
        try:
            yield
        finally:
            rollout._LatentChunks = previous_chunks
            if owned_admit:
                decoder.admit = previous_admit
            else:
                del decoder.admit

    @torch.inference_mode()
    def generate_prompt_pool(self, prompt_ids, max_new_tokens, progress_callback=None):
        if self._generating:
            raise RuntimeError("a paired engine cannot run overlapping generations")
        self._generating = True
        try:
            if self._stock_sampling:
                return super().generate_prompt_pool(prompt_ids, max_new_tokens, progress_callback)
            self._prepare_paired_generation(prompt_ids, max_new_tokens)
            with self._sampling_overrides():
                result = super().generate_prompt_pool(prompt_ids, max_new_tokens, progress_callback)
            if result.admission_events != 1 or self.admission_count != 1 or len(result.responses) != 64:
                self.release_cache()
                raise ValueError("paired generation violated exact64 no-refill contract")
            return result
        finally:
            self._generating = False
