"""Graph-chunk MiniCPM rollout with raw Gaussian thoughts and real head bypass.

Only boundary-known answer lanes and inert head padding enter the vocabulary head.
A bounded GPU action buffer is drained to pinned CPU storage once per chunk.
"""

from __future__ import annotations

import gc
import time
from collections import OrderedDict
from contextlib import contextmanager
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers.cache_utils import Cache

from postraining.fast_inference import (
    ROLLOUT_WORKSPACE_RESERVE_BYTES,
    _CompactStaticLayer,
    _FrozenParameterStash,
    _cuda_allocatable_bytes,
    build_fused_rollout_replica,
    selected_token_logprobs,
    synchronize_fused_lora_policy_,
    top_k_top_p_sample,
)
from postraining.invariant_attention import INVARIANT_ATTENTION
from postraining.invariant_linear import (
    LEGACY_ARITHMETIC,
    OPTIMIZED_ARITHMETIC,
    compile_invariant,
)
from postraining.minicpm_vapo import (
    CONTINUE_THOUGHT,
    FIRST_THOUGHT,
    FORCED_STOP_THINKING,
    STOP_THINKING,
    TOKEN_ACTION,
    MiniCPMVAPOPolicy,
)


@dataclass(frozen=True)
class LatentTrainingGeneration:
    responses: tuple[Tensor, ...]
    action_kinds: tuple[Tensor, ...]
    latent_vectors: tuple[Tensor, ...]
    controller_observations: tuple[Tensor, ...]
    logprobs: tuple[Tensor, ...]
    prefill_seconds: float
    decode_seconds: float
    decode_steps: int
    capacity_row_steps: int
    admission_events: int
    minimum_active_rows_with_backlog: int

    @property
    def useful_tokens(self) -> int:
        return sum(response.numel() for response in self.responses)

    @property
    def productive_utilization(self) -> float:
        return (
            self.useful_tokens / self.capacity_row_steps
            if self.capacity_row_steps
            else 0.0
        )


class _LatentStateDecoder:
    """Compact per-lane KV and hidden-state prefix bank, with no LM projection."""

    def __init__(
        self,
        policy: MiniCPMVAPOPolicy,
        batch_size: int,
        cache_length: int,
        compile_decode: bool,
    ) -> None:
        self.policy = policy
        self.batch_size = batch_size
        self.cache_length = cache_length
        self.device = next(policy.parameters()).device
        if self.device.type != "cuda":
            raise ValueError(
                "MiniCPM latent rollout requires CUDA and FlashAttention 4"
            )
        self.compile_decode = compile_decode
        self.lengths = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        self.flash_lengths = torch.ones(
            batch_size, device=self.device, dtype=torch.int32
        )
        self.mask = torch.zeros(
            (batch_size, cache_length), device=self.device, dtype=torch.bool
        )
        self.cache_positions = torch.arange(cache_length, device=self.device)
        self.cache: Cache | None = None
        self.prefixes: list[list[tuple[Tensor, Tensor]]] = []
        self.prefix_hidden: Tensor | None = None
        self.prompt_lengths: list[int] = []
        self._advance = (
            compile_invariant(self._advance_hidden)
            if compile_decode
            else self._advance_hidden
        )

    def _advance_hidden(self, embeddings: Tensor, active: Tensor) -> Tensor:
        # Paused lanes keep their valid prefix. The unconditional fixed-lane
        # append writes only the scratch entry at lengths[lane], never position
        # zero of a paused prefix. Its hidden output is discarded by the scheduler.
        self.flash_lengths.copy_((self.lengths + 1).to(torch.int32))
        hidden = self.policy.cached_hidden(
            None,
            inputs_embeds=embeddings[:, None],
            past_key_values=self.cache,
            cache_position=self.cache_positions[:1],
            position_ids=self.lengths[:, None],
        )[:, 0]
        self.lengths.add_(active.long())
        return hidden

    def advance(self, embeddings: Tensor, active: Tensor) -> Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self._advance(embeddings, active)

    def prefill(self, prompts: Sequence[Tensor]) -> None:
        self.release_cache()
        self.prompt_lengths = [prompt.numel() for prompt in prompts]
        self.policy.causal_lm.config._attn_implementation = "parameter_golf_fa4_decode"
        for layer in self.policy.causal_lm.model.layers:
            layer.self_attn._rollout_cached_append = False
        hidden_bank = []
        layer_count = len(self.policy.causal_lm.model.layers)
        # Each unique prompt is prefetched once, regardless of sample count.
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start : start + self.batch_size]
            lengths = torch.tensor([p.numel() for p in chunk], device=self.device)
            width = max(p.numel() for p in chunk)
            ids = torch.full(
                (len(chunk), width),
                int(self.policy.causal_lm.config.pad_token_id),
                device=self.device,
                dtype=torch.long,
            )
            mask = torch.zeros_like(ids, dtype=torch.bool)
            positions = torch.zeros_like(ids)
            for row, prompt in enumerate(chunk):
                length = prompt.numel()
                ids[row, -length:] = prompt.to(self.device)
                mask[row, -length:] = True
                positions[row, -length:] = self.cache_positions[:length]
            cache = Cache(
                layers=[
                    _CompactStaticLayer(width, lengths, mask)
                    for _ in range(layer_count)
                ]
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = self.policy.cached_hidden(
                    ids,
                    past_key_values=cache,
                    cache_position=self.cache_positions[:width],
                    attention_mask=mask,
                    position_ids=positions,
                )[:, -1].clone()
            hidden_bank.append(hidden)
            for row, prompt in enumerate(chunk):
                self.prefixes.append(
                    [
                        (
                            layer.key_backing[row, : prompt.numel()].cpu(),
                            layer.value_backing[row, : prompt.numel()].cpu(),
                        )
                        for layer in cache.layers
                    ]
                )
            cache.layers.clear()
        self.prefix_hidden = torch.cat(hidden_bank)
        self.cache = Cache(
            layers=[
                _CompactStaticLayer(
                    self.cache_length,
                    self.lengths,
                    self.mask,
                    indexed_decode=self.compile_decode,
                )
                for _ in range(layer_count)
            ]
        )
        for index, layer in enumerate(self.cache.layers):
            key, value = self.prefixes[0][index]
            layer.lazy_initialization(
                torch.empty(
                    (self.batch_size, key.size(1), 1, key.size(2)),
                    dtype=key.dtype,
                    device=self.device,
                ),
                torch.empty(
                    (self.batch_size, value.size(1), 1, value.size(2)),
                    dtype=value.dtype,
                    device=self.device,
                ),
            )
            layer.prefilling = False
        # The opaque FA4 op supports fullgraph compilation of the embedding trunk.
        self.policy.causal_lm.config._attn_implementation = INVARIANT_ATTENTION
        for layer in self.policy.causal_lm.model.layers:
            attention = layer.self_attn
            attention._rollout_cached_append = True
            attention._rollout_sequence_lengths = self.flash_lengths
            attention._rollout_max_cache_len = self.cache_length
            attention._rollout_optimized_decode = self.compile_decode

    def admit(self, lanes: Sequence[int], prompt_indices: Sequence[int]) -> Tensor:
        if self.cache is None or self.prefix_hidden is None:
            raise RuntimeError("latent decoder has no prompt prefix bank")
        for lane, prompt_index in zip(lanes, prompt_indices, strict=True):
            length = self.prompt_lengths[prompt_index]
            self.lengths[lane] = length
            for layer, (key, value) in zip(
                self.cache.layers, self.prefixes[prompt_index], strict=True
            ):
                layer.key_backing[lane, :length].copy_(key)
                layer.value_backing[lane, :length].copy_(value)
        return self.prefix_hidden[torch.tensor(prompt_indices, device=self.device)]

    def release_cache(self) -> None:
        if self.cache is not None:
            self.cache.layers.clear()
            self.cache = None
        self.prefixes.clear()
        self.prefix_hidden = None
        for layer in self.policy.causal_lm.model.layers:
            layer.self_attn._rollout_sequence_lengths = None
        if self.compile_decode:
            gc.collect()


_IDLE, _THINK, _WAIT, _ANSWER = range(4)


@contextmanager
def _preserve_decode_state(tensors: Sequence[Tensor], device: torch.device):
    """Discard warmup/capture mutations and draws without copying the KV bank.

    Appends only touch the suffix at or beyond the saved valid length. Restoring
    that length invalidates them; copying an entire KV cache would be wasteful.
    """
    saved = [tensor.clone() for tensor in tensors]
    rng = (
        torch.cuda.get_rng_state(device)
        if device.type == "cuda"
        else torch.get_rng_state()
    )
    try:
        yield
    finally:
        for tensor, original in zip(tensors, saved, strict=True):
            tensor.copy_(original)
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng, device)
        else:
            torch.set_rng_state(rng)


@dataclass
class _CapturedChunk:
    thought_indices: Tensor
    answer_indices: Tensor
    graph: torch.cuda.CUDAGraph | None = None


class _LatentChunks:
    """Fixed-storage GPU scheduler; Python sees its state only when draining."""

    def __init__(self, engine: MiniCPMLatentRolloutEngine, budget: int) -> None:
        self.engine = engine
        self.device = engine._device
        if engine.compile_decode and self.device.type != "cuda":
            raise RuntimeError("compiled latent graph execution requires CUDA")
        self.budget = budget
        self.force_at = budget - max(1, engine.answer_reserve_tokens) - 1
        batch, chunk = engine.batch_size, engine.chunk_steps
        head_rows = batch + engine.head_bucket_size - 1
        dimension = int(engine.policy.causal_lm.config.hidden_size)
        self.close_embedding = engine.policy.token_embeddings(
            torch.tensor([engine.thinking_end_token_id], device=self.device)
        )[0]
        self.hidden = torch.zeros(
            (head_rows, dimension), dtype=self.close_embedding.dtype, device=self.device
        )
        self.embeddings = torch.empty_like(self.hidden)
        self.active = torch.zeros(head_rows, dtype=torch.bool, device=self.device)
        self.positions = torch.zeros(head_rows, dtype=torch.long, device=self.device)
        self.phase = torch.zeros(head_rows, dtype=torch.int8, device=self.device)
        self.stop_ids = torch.tensor(engine.stop_ids, device=self.device)
        # Head padding has distinct scratch lanes, never real-lane duplicates.
        # Scratch stays IDLE and is excluded from decoder/KV and host output.
        # Metadata rows: phase, response length, tokens[chunk], kinds[chunk].
        self.metadata = torch.empty(
            (2 + 2 * chunk, head_rows), dtype=torch.long, device=self.device
        )
        self.scores = torch.empty(
            (chunk, head_rows), dtype=torch.float32, device=self.device
        )
        self.raw = torch.empty(
            (chunk, head_rows, dimension), dtype=torch.float32, device=self.device
        )
        self.observations = torch.empty(
            (chunk, batch, dimension), dtype=self.hidden.dtype, device=self.device
        )
        pinned = self.device.type == "cuda"
        self.host_metadata = torch.empty_like(
            self.metadata[:, :batch], device="cpu", pin_memory=pinned
        )
        self.host_scores = torch.empty_like(
            self.scores[:, :batch], device="cpu", pin_memory=pinned
        )
        self.host_raw = torch.empty_like(
            self.raw[:, :batch], device="cpu", pin_memory=pinned
        )
        self.host_observations = torch.empty_like(
            self.observations, device="cpu", pin_memory=pinned
        )
        engine.telemetry["staging_bytes"] = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.metadata,
                self.scores,
                self.raw,
                self.observations,
                self.host_metadata,
                self.host_scores,
                self.host_raw,
                self.host_observations,
            )
        )
        engine.telemetry["controller_observation_staging_bytes"] = (
            2 * self.observations.numel() * self.observations.element_size()
        )
        engine.telemetry["head_bucket_size"] = engine.head_bucket_size
        engine.telemetry["head_scratch_rows"] = head_rows - batch
        self.graphs: OrderedDict[tuple[int, int, int], _CapturedChunk] = OrderedDict()
        self.capture_stream = (
            torch.cuda.Stream(device=self.device) if engine.compile_decode else None
        )
        self.transfer_done = torch.cuda.Event() if pinned else None

    def _step(self, step: int, thought: Tensor, answer: Tensor) -> None:
        engine = self.engine
        # Preserve the actual state that scored this action, before close/WAIT
        # transitions or trunk advancement. Padding slots are discarded at drain.
        self.observations[step].copy_(self.hidden[:engine.batch_size])
        tokens = self.metadata[2 + step]
        kinds = self.metadata[2 + engine.chunk_steps + step]
        kinds.fill_(-1)
        self.embeddings.zero_()
        self.active.zero_()
        if thought.numel():
            hidden = self.hidden[thought]
            live = self.phase[thought] == _THINK
            first = self.positions[thought] == 0
            forced = (self.positions[thought] >= self.force_at) & ~first
            gated = live & ~first & ~forced
            decision, gate_score = engine._gate(hidden)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                raw, score, embedding = engine._thought(hidden)
            if raw.dtype != torch.float32:
                raise RuntimeError("latent transition must return raw fp32 vectors")
            # Static buckets draw padding noise, including for forced/first gates.
            # Extra scratch draws can change RNG sequences versus exact buckets;
            # only native learned decisions contribute to the stored likelihood.
            stop = forced | (gated & decision.bool())
            kind = torch.where(first, FIRST_THOUGHT, CONTINUE_THOUGHT)
            kind = torch.where(
                stop, torch.where(forced, FORCED_STOP_THINKING, STOP_THINKING), kind
            )
            score = score + torch.where(gated, gate_score, 0.0)
            score = torch.where(stop, torch.where(forced, 0.0, gate_score), score)
            kinds[thought] = torch.where(live, kind, -1)
            tokens[thought] = torch.where(
                stop, engine.thinking_end_token_id, engine.thinking_start_token_id
            )
            self.scores[step, thought] = score
            self.raw[step, thought] = raw
            self.embeddings[thought] = torch.where(
                stop[:, None], self.close_embedding, embedding
            )
            self.active[thought] = live
            self.positions[thought] += live.long()
            self.phase[thought] = torch.where(
                live & stop, _WAIT, self.phase[thought]
            ).to(self.phase.dtype)
        if answer.numel():
            # Shape is fixed for the entire captured chunk. This gather contains
            # only admitted answer lanes, retired lanes, and inert scratch.
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                token, score, embedding = engine._answer(self.hidden[answer])
            live = self.phase[answer] == _ANSWER
            complete = (self.positions[answer] + 1 >= self.budget) | (
                token[:, None] == self.stop_ids[None]
            ).any(-1)
            tokens[answer] = token
            kinds[answer] = torch.where(live, TOKEN_ACTION, -1)
            self.scores[step, answer] = score
            self.embeddings[answer] = embedding
            self.active[answer] = live & ~complete
            self.positions[answer] += live.long()
            self.phase[answer] = torch.where(
                live & complete, _IDLE, self.phase[answer]
            ).to(self.phase.dtype)
        # Even when all lanes stop inside a chunk, the fixed-lane trunk remains
        # in the graph. Paused prefixes and hidden states must remain unchanged.
        active = self.active[:engine.batch_size]
        real_hidden = self.hidden[:engine.batch_size]
        hidden = engine._decoder.advance(
            self.embeddings[:engine.batch_size], active
        )
        real_hidden.copy_(torch.where(active[:, None], hidden, real_hidden))

    def _execute(self, entry: _CapturedChunk, steps: int) -> None:
        for step in range(steps):
            self._step(step, entry.thought_indices, entry.answer_indices)
        self.metadata[0].copy_(self.phase)
        self.metadata[1].copy_(self.positions)

    def _mutable_state(self) -> list[Tensor]:
        decoder = self.engine._decoder
        tensors = [
            self.hidden,
            self.positions,
            self.phase,
            decoder.lengths,
            decoder.flash_lengths,
        ]
        if decoder.cache is not None:
            tensors.extend(layer.cumulative_length for layer in decoder.cache.layers)
        return tensors

    def _capture(self, entry: _CapturedChunk, steps: int) -> None:
        if self.device.type != "cuda":
            raise RuntimeError("compiled latent graph execution requires CUDA")
        current = torch.cuda.current_stream(self.device)
        stream = self.capture_stream
        # Compilation and allocator warmup happen before capture on a side
        # stream. Restore between attempts so none can overwrite a valid prefix.
        for _ in range(2):
            with _preserve_decode_state(self._mutable_state(), self.device):
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    self._execute(entry, steps)
                current.wait_stream(stream)
                stream.synchronize()
        with _preserve_decode_state(self._mutable_state(), self.device):
            stream.wait_stream(current)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self._execute(entry, steps)
            current.wait_stream(stream)
        entry.graph = graph
        self.engine.telemetry["graph_captures"] += 1

    def run(
        self, thinking: Sequence[int], answering: Sequence[int], steps: int
    ) -> None:
        engine = self.engine
        bucket = engine.head_bucket_size
        thought_count, answer_count = (
            min(engine.batch_size, (len(lanes) + bucket - 1) // bucket * bucket)
            for lanes in (thinking, answering)
        )
        key = thought_count, answer_count, steps
        entry = self.graphs.pop(key, None)
        if entry is None:
            # Evict before capture: graph-private workspace is bounded too.
            if len(self.graphs) >= engine.max_cached_graphs:
                _, retired = self.graphs.popitem(last=False)
                if retired.graph is not None:
                    retired.graph.reset()
                del retired
            entry = _CapturedChunk(
                torch.empty(thought_count, dtype=torch.long, device=self.device),
                torch.empty(answer_count, dtype=torch.long, device=self.device),
            )
        # Host-known membership only: no GPU nonzero or scalar extraction.
        # Each phase uses distinct scratch indices internally; phases may share
        # scratch because it is never live and their scatters execute in order.
        for indices, lanes in (
            (entry.thought_indices, thinking),
            (entry.answer_indices, answering),
        ):
            padding = indices.numel() - len(lanes)
            indices.copy_(torch.tensor(
                [*lanes, *range(engine.batch_size, engine.batch_size + padding)],
                dtype=torch.long,
            ))
        if engine.compile_decode:
            if entry.graph is None:
                self._capture(entry, steps)
            entry.graph.replay()
            engine.telemetry["graph_replays"] += 1
        else:
            self._execute(entry, steps)
        self.graphs[key] = entry
        engine.telemetry["peak_cached_graphs"] = max(
            engine.telemetry["peak_cached_graphs"], len(self.graphs)
        )
        engine.telemetry["trunk_steps"] += steps
        engine.telemetry["chunk_capacity_steps"] += steps * engine.batch_size
        engine.telemetry["thought_head_rows"] += steps * thought_count
        engine.telemetry["thought_head_padding_rows"] += (
            steps * (thought_count - len(thinking))
        )
        if answering:
            engine.telemetry["answer_projection_calls"] += steps
            engine.telemetry["answer_projection_rows"] += steps * answer_count
            engine.telemetry["answer_projection_padding_rows"] += (
                steps * (answer_count - len(answering))
            )

    def drain(self, steps: int, has_thoughts: bool) -> None:
        batch = self.engine.batch_size
        transfers = [
            (self.host_metadata, self.metadata[:, :batch]),
            (self.host_scores[:steps], self.scores[:steps, :batch]),
            (self.host_observations[:steps], self.observations[:steps]),
        ]
        if has_thoughts:
            transfers.append((self.host_raw[:steps], self.raw[:steps, :batch]))
        for destination, source in transfers:
            destination.copy_(source, non_blocking=self.transfer_done is not None)
            self.engine.telemetry["host_transfer_calls"] += 1
            self.engine.telemetry["host_transfer_bytes"] += (
                source.numel() * source.element_size()
            )
        self.engine.telemetry["controller_observation_transfer_bytes"] += (
            self.observations[:steps].numel() * self.observations.element_size()
        )
        if self.transfer_done is not None:
            self.transfer_done.record(torch.cuda.current_stream(self.device))
            self.transfer_done.synchronize()

    def release(self) -> None:
        for entry in self.graphs.values():
            if entry.graph is not None:
                entry.graph.reset()
        self.graphs.clear()


class MiniCPMLatentRolloutEngine:
    """Continuous-refill latent actor with explicit compiled CUDA graph chunks.

    Membership changes only at chunk boundaries. Closing thoughts consume the
    delimiter immediately, then wait before answer-head admission. Retired answer
    lanes may remain padding until that boundary, but no thinking lane is ever
    projected. Head buckets append IDLE scratch lanes, without widening decoder
    or KV storage. Padding draws can change RNG sequences; head_bucket_size=1
    selects exact membership sizes. Eager decoding requires compile_decode=False.
    """

    def __init__(
        self,
        policy: MiniCPMVAPOPolicy,
        *,
        stop_ids: Sequence[int],
        thinking_start_token_id: int,
        thinking_end_token_id: int,
        prompts_per_rollout: int,
        samples_per_prompt: int,
        cache_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        compile_decode: bool = True,
        physical_batch_size: int | None = None,
        answer_reserve_tokens: int = 1000,
        chunk_steps: int = 8,
        max_cached_graphs: int = 8,
        head_bucket_size: int = 16,
    ) -> None:
        if not getattr(policy, "latent_thinking", False):
            raise ValueError("latent rollout requires a latent-thinking policy")
        logical_batch = prompts_per_rollout * samples_per_prompt
        if prompts_per_rollout < 1 or samples_per_prompt < 1:
            raise ValueError("rollout batch dimensions must be positive")
        if physical_batch_size is None:
            physical_batch_size = logical_batch
        if not 1 <= physical_batch_size <= logical_batch:
            raise ValueError(
                "physical rollout batch must fit the logical rollout batch"
            )
        if (
            not stop_ids
            or thinking_end_token_id in stop_ids
            or thinking_start_token_id in stop_ids
        ):
            raise ValueError("thinking delimiters must not be rollout stop ids")
        if thinking_start_token_id == thinking_end_token_id:
            raise ValueError("thinking delimiters must differ")
        vocab_size = int(policy.causal_lm.config.vocab_size)
        if any(
            not 0 <= token < vocab_size
            for token in (*stop_ids, thinking_start_token_id, thinking_end_token_id)
        ):
            raise ValueError("rollout token id is outside the vocabulary")
        if temperature <= 0 or not 1 <= top_k <= vocab_size or not 0 < top_p <= 1:
            raise ValueError("sampling dimensions are invalid")
        if answer_reserve_tokens < 0 or cache_length < 4:
            raise ValueError("invalid answer reserve or cache length")
        if chunk_steps < 1 or max_cached_graphs < 1:
            raise ValueError("chunk and graph cache sizes must be positive")
        if not isinstance(head_bucket_size, int) or head_bucket_size < 1:
            raise ValueError("head bucket size must be a positive integer")
        self.source_policy = policy
        self.policy, self.fused_projection_groups = build_fused_rollout_replica(policy)
        self.batch_size = physical_batch_size
        self.prompts_per_rollout = prompts_per_rollout
        self.samples_per_prompt = samples_per_prompt
        self.cache_length = cache_length
        self.answer_reserve_tokens = answer_reserve_tokens
        self.stop_ids = tuple(int(token) for token in stop_ids)
        self.thinking_start_token_id = thinking_start_token_id
        self.thinking_end_token_id = thinking_end_token_id
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.chunk_steps = chunk_steps
        self.max_cached_graphs = max_cached_graphs
        self.head_bucket_size = min(head_bucket_size, physical_batch_size)
        self.compile_decode = compile_decode
        self.arithmetic = (
            f"latent/v3:chunk{chunk_steps}:head{self.head_bucket_size}:"
            + (OPTIMIZED_ARITHMETIC if compile_decode else LEGACY_ARITHMETIC)
        )
        self.telemetry: dict[str, int | float] = {}
        self._chunks: _LatentChunks | None = None
        self._device = next(self.policy.parameters()).device
        config = self.policy.causal_lm.config
        self.estimated_cache_bytes = (
            physical_batch_size
            * cache_length
            * int(config.num_hidden_layers)
            * 2
            * int(config.num_key_value_heads)
            * int(config.head_dim)
            * 2
        )
        self._source_stash = _FrozenParameterStash(policy.causal_lm)
        self.offloaded_source_bytes = self._source_stash.bytes
        self._decoder = _LatentStateDecoder(
            self.policy, physical_batch_size, cache_length, compile_decode
        )

        def thought(hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            mean = self.policy.transition.predict_mean(hidden)
            sigma = self.policy.transition.predict_log_sigma(hidden)
            raw = self.policy.transition.sample_latent(mean, sigma)
            return (
                raw,
                self.policy.transition.log_prob(raw, mean, sigma),
                self.policy.thought_embeddings(raw),
            )

        def gate(hidden: Tensor) -> tuple[Tensor, Tensor]:
            return self.policy.thinking_gate.sample(hidden)

        def answer(hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            logits = self.policy.logits(hidden)
            delimiters = [self.thinking_start_token_id, self.thinking_end_token_id]
            original = logits[:, delimiters].clone()
            logits[:, delimiters] = -torch.inf
            tokens = top_k_top_p_sample(
                logits, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p
            )
            # Filtering changes sampling, not the untempered replay likelihood.
            logits[:, delimiters] = original
            return (
                tokens,
                selected_token_logprobs(logits, tokens),
                self.policy.token_embeddings(tokens),
            )

        # Dynamic rows cover the bounded membership buckets of each phase.
        compile_head = lambda fn: torch.compile(
            fn, fullgraph=True, dynamic=True, options={"emulate_precision_casts": True}
        )
        self._thought = compile_head(thought) if compile_decode else thought
        self._gate = compile_head(gate) if compile_decode else gate
        self._answer = compile_head(answer) if compile_decode else answer

    @torch.no_grad()
    def synchronize_from(self, policy: MiniCPMVAPOPolicy) -> None:
        if not getattr(policy, "latent_thinking", False):
            raise ValueError("cannot refresh latent rollout from a token-only policy")
        self._release_decode_storage()
        self._source_stash.restore()
        if policy is not self.source_policy:
            self.source_policy = policy
            self._source_stash = _FrozenParameterStash(policy.causal_lm)
            self.offloaded_source_bytes = self._source_stash.bytes
        synchronize_fused_lora_policy_(self.policy, policy)
        for name in ("transition", "thinking_gate", "thought_adapter"):
            getattr(self.policy, name).load_state_dict(
                getattr(policy, name).state_dict(), strict=True
            )
        # Sigma is distribution geometry, not a parameter or state_dict buffer.
        self.policy.transition.vector_sigma = policy.transition.vector_sigma
        self._source_stash.offload()
        if self._device.type == "cuda":
            available = _cuda_allocatable_bytes(self._device)
            if self.estimated_cache_bytes + ROLLOUT_WORKSPACE_RESERVE_BYTES > available:
                self._source_stash.restore()
                raise MemoryError(
                    "latent rollout KV plus workspace exceeds available CUDA memory"
                )

    def _release_decode_storage(self) -> None:
        # Captures own cache/storage addresses: destroy them before releasing KV
        # and before restoring the offloaded source actor.
        if self._chunks is not None:
            self._chunks.release()
            self._chunks = None
        self._decoder.release_cache()

    def release_cache(self) -> None:
        self._release_decode_storage()
        self._source_stash.restore()

    def _synchronize_device(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    @torch.inference_mode()
    def generate_prompt_pool(
        self,
        prompt_ids: Sequence[Tensor],
        max_new_tokens: int,
        progress_callback: Callable[[int, dict[str, float | int]], None] | None = None,
    ) -> LatentTrainingGeneration:
        if len(prompt_ids) != self.prompts_per_rollout:
            raise ValueError("rollout prompt count differs from configured batch")
        if max_new_tokens < max(3, self.answer_reserve_tokens + 2):
            raise ValueError(
                "response budget must fit first thought, close, and answer reserve"
            )
        for prompt in prompt_ids:
            if prompt.ndim != 1 or prompt.numel() < 1 or prompt.dtype != torch.long:
                raise ValueError(
                    "rollout prompts must be nonempty one-dimensional int64 tokens"
                )
            if not bool((prompt == self.thinking_start_token_id).any()):
                raise ValueError("latent rollout requires a native thinking prefix")
            if prompt.numel() + max_new_tokens > self.cache_length:
                raise ValueError("prompt plus latent response exceeds rollout cache")
        # The trainer validates the exact native suffix including tokenizer whitespace.
        self.synchronize_from(self.source_policy)
        try:
            return self._generate(prompt_ids, max_new_tokens, progress_callback)
        except BaseException:
            self.release_cache()
            raise

    def _generate(
        self,
        prompt_ids: Sequence[Tensor],
        max_new_tokens: int,
        progress_callback: Callable[[int, dict[str, float | int]], None] | None,
    ) -> LatentTrainingGeneration:
        self.telemetry = dict.fromkeys(
            (
                "graph_captures",
                "graph_replays",
                "scheduling_boundaries",
                "host_transfer_calls",
                "host_transfer_bytes",
                "answer_projection_calls",
                "answer_projection_rows",
                "answer_projection_padding_rows",
                "thought_head_rows",
                "thought_head_padding_rows",
                "thought_actions",
                "trunk_steps",
                "chunk_capacity_steps",
                "staging_bytes",
                "controller_observation_staging_bytes",
                "controller_observation_transfer_bytes",
                "peak_cached_graphs",
            ),
            0,
        )
        self.telemetry.update(
            host_scheduling_seconds=0.0,
            host_finalization_seconds=0.0,
            host_tensor_assembly_seconds=0.0,
        )
        self._synchronize_device()
        started = time.perf_counter()
        self._decoder.prefill(prompt_ids)
        chunks = self._chunks = _LatentChunks(self, max_new_tokens)
        self._synchronize_device()
        prefilled = time.perf_counter()
        count = len(prompt_ids) * self.samples_per_prompt
        responses: list[list[int]] = [[] for _ in range(count)]
        kinds: list[list[int]] = [[] for _ in range(count)]
        scores: list[list[float]] = [[] for _ in range(count)]
        vectors: list[list[Tensor]] = [[] for _ in range(count)]
        observations: list[list[Tensor]] = [[] for _ in range(count)]
        lanes = [-1] * self.batch_size
        phases = [_IDLE] * self.batch_size
        next_row = steps = admissions = 0
        minimum_active = self.batch_size
        chunk_steps = 1
        stable_boundaries = 0
        # Wall-time from synchronized drain return to the next run submission,
        # not exact GPU-idle time: refill/promotion may enqueue small kernels.
        last_drained: float | None = None
        host_scheduling_seconds = 0.0
        while next_row < count or any(row >= 0 for row in lanes):
            free = [lane for lane, row in enumerate(lanes) if row < 0]
            admitted = free[: count - next_row]
            if admitted:
                rows = list(range(next_row, next_row + len(admitted)))
                indices = torch.tensor(admitted, device=self._device)
                chunks.hidden[indices] = self._decoder.admit(
                    admitted, [row // self.samples_per_prompt for row in rows]
                )
                chunks.positions[indices] = 0
                chunks.phase[indices] = _THINK
                for lane, row in zip(admitted, rows, strict=True):
                    lanes[lane], phases[lane] = row, _THINK
                next_row += len(admitted)
                admissions += 1
                chunk_steps, stable_boundaries = 1, 0
            if next_row < count:
                minimum_active = min(minimum_active, sum(row >= 0 for row in lanes))
            thinking = [lane for lane, phase in enumerate(phases) if phase == _THINK]
            answering = [lane for lane, phase in enumerate(phases) if phase == _ANSWER]
            before = phases
            if last_drained is not None:
                host_scheduling_seconds += time.perf_counter() - last_drained
            chunks.run(thinking, answering, chunk_steps)
            chunks.drain(chunk_steps, bool(thinking))
            last_drained = time.perf_counter()
            phases = chunks.host_metadata[0].tolist()
            for step in range(chunk_steps):
                for lane, row in enumerate(lanes):
                    kind = int(chunks.host_metadata[2 + self.chunk_steps + step, lane])
                    if kind < 0:
                        continue
                    responses[row].append(int(chunks.host_metadata[2 + step, lane]))
                    kinds[row].append(kind)
                    scores[row].append(float(chunks.host_scores[step, lane]))
                    observations[row].append(
                        chunks.host_observations[step, lane].clone()
                    )
                    if kind in (FIRST_THOUGHT, CONTINUE_THOUGHT):
                        # Clone before reusing the bounded staging slab. This is
                        # the exact fp32 sample, not its bf16 embedding/mean.
                        vectors[row].append(chunks.host_raw[step, lane].clone())
                        self.telemetry["thought_actions"] += 1
            for lane, phase in enumerate(phases):
                if phase == _IDLE:
                    lanes[lane] = -1
                elif phase == _WAIT:
                    phases[lane] = _ANSWER
            # This tensor-only promotion happens after close was consumed, and
            # before the next boundary-known answer bucket is constructed.
            chunks.phase.masked_fill_(chunks.phase == _WAIT, _ANSWER)
            steps += chunk_steps
            self.telemetry["scheduling_boundaries"] += 1
            if phases != before:
                chunk_steps, stable_boundaries = 1, 0
            else:
                stable_boundaries += 1
                if stable_boundaries >= 2:
                    chunk_steps = min(self.chunk_steps, chunk_steps * 2)
            if progress_callback is not None:
                progress_callback(
                    steps,
                    {
                        "scheduled_tokens": steps * self.batch_size,
                        "active_rows": sum(row >= 0 for row in lanes),
                        "completed_rows": next_row - sum(row >= 0 for row in lanes),
                    },
                )
        self._synchronize_device()
        finished = time.perf_counter()
        result = LatentTrainingGeneration(
            responses=tuple(torch.tensor(row, dtype=torch.long) for row in responses),
            action_kinds=tuple(torch.tensor(row, dtype=torch.int8) for row in kinds),
            latent_vectors=tuple(torch.stack(row) for row in vectors),
            controller_observations=tuple(torch.stack(row) for row in observations),
            logprobs=tuple(torch.tensor(row, dtype=torch.float32) for row in scores),
            prefill_seconds=prefilled - started,
            decode_seconds=finished - prefilled,
            decode_steps=steps,
            capacity_row_steps=steps * self.batch_size,
            admission_events=admissions,
            minimum_active_rows_with_backlog=minimum_active,
        )
        assembled = time.perf_counter()
        self.telemetry["host_scheduling_seconds"] = host_scheduling_seconds
        # Final drain processing (including the existing final synchronization)
        # and CPU output assembly have no subsequent run; report them separately.
        if last_drained is not None:
            self.telemetry["host_finalization_seconds"] = assembled - last_drained
        self.telemetry["host_tensor_assembly_seconds"] = assembled - finished
        return result
