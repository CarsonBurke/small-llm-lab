from __future__ import annotations

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import postraining.minicpm_latent_rollout as rollout
from postraining.minicpm_vapo import (
    CONTINUE_THOUGHT,
    FIRST_THOUGHT,
    FORCED_STOP_THINKING,
    STOP_THINKING,
    TOKEN_ACTION,
)


class FakeTransition(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor([0.000123, 0.000789]))
        self.vector_sigma = 1.0
        self.draws = []

    def predict_mean(self, hidden):
        return hidden.float() + self.offset

    def predict_log_sigma(self, hidden):
        return torch.zeros_like(hidden, dtype=torch.float32)

    def sample_latent(self, mean, sigma):
        raw = mean + torch.tensor([0.125, -0.25])
        self.draws.extend(raw.detach().clone().unbind())
        return raw

    def log_prob(self, raw, mean, sigma):
        return torch.full((raw.size(0),), -2.0)


class FakeGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("never_stop", torch.tensor(False))
        self.calls = []

    def sample(self, hidden):
        self.calls.extend(hidden.clone().unbind())
        # Prompt 10 stops after its first thought; 11 takes two thoughts.
        decision = (hidden[:, 1] >= hidden[:, 0] - 9).long()
        if self.never_stop:
            decision.zero_()
        return decision, torch.where(decision.bool(), -0.25, -0.75)


class FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.causal_lm = nn.Linear(1, 1)
        self.causal_lm.config = SimpleNamespace(
            vocab_size=32,
            hidden_size=2,
            num_hidden_layers=1,
            num_key_value_heads=1,
            head_dim=2,
        )
        self.transition = FakeTransition()
        self.thinking_gate = FakeGate()
        self.thought_adapter = nn.Identity()
        self.latent_thinking = True
        self.lm_states = []
        self.thought_inputs = []
        self.decoder = None
        self.force_answers = False

    def token_embeddings(self, ids):
        return torch.stack((ids, torch.full_like(ids, -100)), -1).to(torch.bfloat16)

    def thought_embeddings(self, raw):
        self.thought_inputs.extend(raw.detach().clone().unbind())
        return self.thought_adapter(raw.to(torch.bfloat16))

    def logits(self, hidden):
        # Zero states are head-only scratch; every real projection still must
        # follow a consumed close. Native fixture prompts never have zero state.
        for state in hidden:
            if torch.equal(state, torch.zeros_like(state)):
                continue
            assert any(
                torch.equal(state, self.decoder.hidden[lane])
                and self.decoder.closed[lane]
                for lane in range(self.decoder.batch_size)
            )
        self.lm_states.extend(hidden.clone().unbind())
        logits = torch.full((hidden.size(0), 32), -10.0)
        for row, state in enumerate(hidden):
            # Prompt 10 ends immediately, 11 emits one answer token before EOS.
            token = 20 if int(state[0]) == 11 and int(state[1]) == 3 else 2
            if self.force_answers:
                token = 20
            logits[row, token] = 10.0
        # Even an overwhelming native delimiter logit cannot reopen thinking.
        logits[:, 8] = 100.0
        logits[:, 9] = 99.0
        return logits


class FakeStateDecoder:
    """Independent lane histories; no transformer, CUDA, or model workload."""

    def __init__(self, policy, batch_size, cache_length, compile_decode):
        self.policy = policy
        policy.decoder = self
        self.batch_size = batch_size
        self.hidden = torch.zeros((batch_size, 2), dtype=torch.bfloat16)
        self.closed = [False] * batch_size
        self.lane_rows = [-1] * batch_size
        self.stream_inputs = []
        self.prefix_requests = []
        self.release_count = 0

    def prefill(self, prompts):
        self.prompts = prompts
        self.prefix_requests.append(tuple(prompt.tolist() for prompt in prompts))
        self.stream_inputs = []

    def admit(self, lanes, prompt_indices):
        for lane, prompt_index in zip(lanes, prompt_indices, strict=True):
            self.hidden[lane] = torch.tensor([self.prompts[prompt_index][0], 0])
            self.closed[lane] = False
            self.lane_rows[lane] = len(self.stream_inputs)
            self.stream_inputs.append([])
        return self.hidden[lanes]

    def advance(self, embeddings, active):
        assert embeddings.shape == (self.batch_size, 2)
        assert active.shape == (self.batch_size,)
        for lane in range(self.batch_size):
            if not active[lane]:
                continue
            vector = embeddings[lane].clone()
            self.stream_inputs[self.lane_rows[lane]].append(vector)
            if vector[1] == -100 and vector[0] == 9:
                assert not self.closed[lane]
                self.closed[lane] = True
            self.hidden[lane, 1] += 1
        return self.hidden.clone()

    def release_cache(self):
        self.release_count += 1


class FakeStash:
    bytes = 0

    def __init__(self, policy):
        self.resident = True

    def restore(self):
        self.resident = True

    def offload(self):
        self.resident = False


@pytest.fixture
def make_engine(monkeypatch):
    monkeypatch.setattr(
        rollout,
        "build_fused_rollout_replica",
        lambda policy: (copy.deepcopy(policy), ()),
    )
    monkeypatch.setattr(
        rollout, "synchronize_fused_lora_policy_", lambda destination, source: 0
    )
    monkeypatch.setattr(rollout, "_FrozenParameterStash", FakeStash)
    monkeypatch.setattr(rollout, "_LatentStateDecoder", FakeStateDecoder)

    def make(*, reserve=1, batch=2, samples=2, prompts=2, chunk=1, head_bucket_size=16):
        source = FakePolicy()
        engine = rollout.MiniCPMLatentRolloutEngine(
            source,
            stop_ids=(2,),
            thinking_start_token_id=8,
            thinking_end_token_id=9,
            prompts_per_rollout=prompts,
            samples_per_prompt=samples,
            cache_length=32,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
            compile_decode=False,
            physical_batch_size=batch,
            answer_reserve_tokens=reserve,
            chunk_steps=chunk,
            head_bucket_size=head_bucket_size,
        )
        return source, engine

    return make


def test_phase_heads_close_consumption_refill_and_exact_raw_actions(make_engine):
    _, engine = make_engine()
    result = engine.generate_prompt_pool(
        [torch.tensor([10, 8]), torch.tensor([11, 7, 8, 7])], 7
    )
    assert [row.tolist() for row in result.responses] == [
        [8, 9, 2],
        [8, 9, 2],
        [8, 8, 9, 20, 2],
        [8, 8, 9, 20, 2],
    ]
    assert [row.tolist() for row in result.action_kinds] == [
        [FIRST_THOUGHT, STOP_THINKING, TOKEN_ACTION],
        [FIRST_THOUGHT, STOP_THINKING, TOKEN_ACTION],
        [FIRST_THOUGHT, CONTINUE_THOUGHT, STOP_THINKING, TOKEN_ACTION, TOKEN_ACTION],
        [FIRST_THOUGHT, CONTINUE_THOUGHT, STOP_THINKING, TOKEN_ACTION, TOKEN_ACTION],
    ]
    assert result.logprobs[0][:2].tolist() == [-2.0, -0.25]
    assert result.logprobs[2][:3].tolist() == [-2.0, -2.75, -0.25]
    assert (
        len(engine.policy.lm_states) == 6
    )  # only six answer actions, no thought/close projections
    draws = sorted(tuple(raw.tolist()) for raw in engine.policy.transition.draws)
    stored = sorted(
        tuple(raw.tolist()) for rows in result.latent_vectors for raw in rows
    )
    assert all(raw in draws for raw in stored)
    assert any(
        not torch.equal(raw, raw.bfloat16().float())
        for row in result.latent_vectors
        for raw in row
    )
    for row, raw in zip(
        engine._decoder.stream_inputs, result.latent_vectors, strict=True
    ):
        assert torch.equal(torch.stack(row[: raw.size(0)]), raw.bfloat16())
        assert row[raw.size(0)].tolist() == [9.0, -100.0]
    assert all(
        row.dtype == torch.float32
        and row.device.type == "cpu"
        and not row.requires_grad
        for row in result.latent_vectors
    )
    assert result.minimum_active_rows_with_backlog == 2
    assert result.admission_events == 2
    assert result.useful_tokens == 16
    assert result.capacity_row_steps == 16


@pytest.mark.parametrize("reserve,budget,thoughts", [(0, 5, 3), (2, 6, 3)])
def test_budget_forces_close_without_gate_or_vocab_action(
    make_engine, reserve, budget, thoughts
):
    source, engine = make_engine(reserve=reserve, batch=1, samples=1, prompts=1)
    source.thinking_gate.never_stop.fill_(True)
    result = engine.generate_prompt_pool([torch.tensor([10, 8])], budget)
    assert result.action_kinds[0].tolist() == [FIRST_THOUGHT] + [CONTINUE_THOUGHT] * (
        thoughts - 1
    ) + [FORCED_STOP_THINKING, TOKEN_ACTION]
    assert result.responses[0].tolist() == [8] * thoughts + [9, 2]
    assert result.logprobs[0][thoughts].item() == 0.0
    assert len(engine.policy.lm_states) == 1


def test_refill_does_not_wait_for_other_lanes_and_keeps_prompt_order(make_engine):
    _, engine = make_engine(batch=3, samples=2)
    result = engine.generate_prompt_pool(
        [torch.tensor([10, 8]), torch.tensor([11, 8])], 7
    )
    assert [row.tolist() for row in result.responses] == [
        [8, 9, 2],
        [8, 9, 2],
        [8, 8, 9, 20, 2],
        [8, 8, 9, 20, 2],
    ]
    # A fourth row enters while the third lane is still answering.
    assert result.admission_events == 2
    assert result.decode_steps == 8
    assert len(engine._decoder.prefix_requests) == 1


def test_release_and_generation_refresh_all_latent_actor_state(make_engine):
    source, engine = make_engine(batch=1, samples=1, prompts=1)
    prompt = [torch.tensor([10, 8])]
    first = engine.generate_prompt_pool(prompt, 6)
    engine.release_cache()
    assert engine._source_stash.resident
    with torch.no_grad():
        source.transition.offset.add_(0.5)
        source.thinking_gate.never_stop.fill_(True)
    second = engine.generate_prompt_pool(prompt, 6)
    assert first.action_kinds[0].tolist() == [
        FIRST_THOUGHT,
        STOP_THINKING,
        TOKEN_ACTION,
    ]
    assert FORCED_STOP_THINKING in second.action_kinds[0].tolist()
    torch.testing.assert_close(
        second.latent_vectors[0][0], first.latent_vectors[0][0] + 0.5
    )
    assert not engine._source_stash.resident


def test_repeated_generation_frees_cache_before_restoring_offloaded_weights(
    make_engine,
    monkeypatch,
):
    _, engine = make_engine(batch=1, samples=1, prompts=1)
    cache_live = False
    prefill = engine._decoder.prefill
    release = engine._decoder.release_cache
    restore = engine._source_stash.restore

    def allocate(prompts):
        nonlocal cache_live
        cache_live = True
        return prefill(prompts)

    def free():
        nonlocal cache_live
        cache_live = False
        release()

    def restore_with_memory_limit():
        if cache_live:
            raise MemoryError("source weights and live KV cannot coexist")
        restore()

    monkeypatch.setattr(engine._decoder, "prefill", allocate)
    monkeypatch.setattr(engine._decoder, "release_cache", free)
    monkeypatch.setattr(engine._source_stash, "restore", restore_with_memory_limit)
    prompt = [torch.tensor([10, 8])]
    first = engine.generate_prompt_pool(prompt, 6)
    second = engine.generate_prompt_pool(prompt, 6)
    assert first.responses[0].tolist() == second.responses[0].tolist()


def test_budget_exhaustion_keeps_full_answer_reserve_and_final_token(make_engine):
    source, engine = make_engine(reserve=2, batch=1, samples=1, prompts=1)
    source.thinking_gate.never_stop.fill_(True)
    engine.policy.force_answers = True
    result = engine.generate_prompt_pool([torch.tensor([10, 8])], 6)
    assert result.responses[0].tolist() == [8, 8, 8, 9, 20, 20]
    assert result.action_kinds[0].tolist() == [
        FIRST_THOUGHT,
        CONTINUE_THOUGHT,
        CONTINUE_THOUGHT,
        FORCED_STOP_THINKING,
        TOKEN_ACTION,
        TOKEN_ACTION,
    ]
    assert len(engine._decoder.stream_inputs[0]) == 5
    assert result.latent_vectors[0].size(0) == 3
    assert len(engine.policy.lm_states) == 2


@pytest.mark.parametrize(
    "prompt,budget,reserve,match",
    [
        ([10, 8], 2, 0, "budget"),
        ([10, 8], 4, 3, "budget"),
        ([10, 8] + [7] * 28, 3, 0, "cache"),
        ([10, 7], 4, 0, "prefix"),
    ],
)
def test_invalid_stream_geometry_is_rejected_before_actor_refresh(
    make_engine, prompt, budget, reserve, match
):
    _, engine = make_engine(reserve=reserve, batch=1, samples=1, prompts=1)
    with pytest.raises(ValueError, match=match):
        engine.generate_prompt_pool([torch.tensor(prompt)], budget)
    assert engine._source_stash.resident
    assert not engine._decoder.prefix_requests


def test_paused_decoder_preserves_valid_kv_prefix_and_length():
    decoder = rollout._LatentStateDecoder.__new__(rollout._LatentStateDecoder)
    decoder.lengths = torch.tensor([2, 3])
    decoder.flash_lengths = torch.ones(2, dtype=torch.int32)
    decoder.cache_positions = torch.arange(8)
    backing = torch.arange(16, dtype=torch.float32).reshape(2, 8).clone()
    prefix = backing[1, :3].clone()
    decoder.cache = backing

    def cached_hidden(_, *, inputs_embeds, **kwargs):
        backing[torch.arange(2), decoder.lengths] = inputs_embeds[:, 0, 0]
        return inputs_embeds

    decoder.policy = SimpleNamespace(cached_hidden=cached_hidden)
    for _ in range(2):
        decoder._advance_hidden(
            torch.tensor([[40.0], [99.0]]), torch.tensor([True, False])
        )
    assert decoder.lengths.tolist() == [4, 3]
    assert torch.equal(backing[1, :3], prefix)
    # Only scratch beyond the paused valid prefix may be overwritten.
    assert backing[1, 3] == 99
    assert backing[0, 2:4].tolist() == [40, 40]


@pytest.mark.parametrize("fail", [False, True])
def test_capture_warmup_restores_hidden_lengths_and_rng(fail):
    hidden = torch.tensor([[1.25, 2.5]])
    lengths = torch.tensor([2])
    phase = torch.tensor([rollout._THINK], dtype=torch.int8)
    position = torch.tensor([1])
    tensors = [hidden, lengths, phase, position]
    originals = [tensor.clone() for tensor in tensors]
    state = torch.get_rng_state().clone()
    try:
        with rollout._preserve_decode_state(tensors, torch.device("cpu")):
            hidden.add_(torch.randn_like(hidden))
            lengths.add_(4)
            position.add_(3)
            phase.fill_(rollout._WAIT)
            if fail:
                raise RuntimeError("warmup failure")
    except RuntimeError:
        assert fail
    assert all(torch.equal(tensor, saved) for tensor, saved in zip(tensors, originals))
    assert torch.equal(torch.get_rng_state(), state)


def test_chunk_close_waits_without_thought_projection_or_hidden_overwrite(make_engine):
    _, engine = make_engine(batch=2, samples=1, chunk=8)
    engine._decoder.prefill([torch.tensor([10, 8]), torch.tensor([11, 8])])
    engine.telemetry = {
        key: 0
        for key in (
            "staging_bytes",
            "graph_captures",
            "graph_replays",
            "peak_cached_graphs",
            "trunk_steps",
            "chunk_capacity_steps",
            "answer_projection_calls",
            "answer_projection_rows",
            "answer_projection_padding_rows",
            "thought_head_rows",
            "thought_head_padding_rows",
            "host_transfer_calls",
            "host_transfer_bytes",
            "controller_observation_transfer_bytes",
        )
    }
    chunks = rollout._LatentChunks(engine, 7)
    chunks.hidden[:2].copy_(engine._decoder.admit([0, 1], [0, 1]))
    chunks.phase[:2].fill_(rollout._THINK)
    chunks.run([0, 1], [], 4)
    chunks.drain(4, True)
    assert not engine.policy.lm_states
    assert chunks.phase.tolist() == [rollout._WAIT, rollout._WAIT, rollout._IDLE]
    assert chunks.positions.tolist() == [2, 3, 0]
    assert chunks.hidden.tolist() == [[10, 2], [11, 3], [0, 0]]
    assert chunks.host_metadata[2 + 8 : 2 + 8 + 4].tolist() == [
        [FIRST_THOUGHT, FIRST_THOUGHT],
        [STOP_THINKING, CONTINUE_THOUGHT],
        [-1, STOP_THINKING],
        [-1, -1],
    ]
    assert chunks.host_observations[:4].tolist() == [
        [[10, 0], [11, 0]],
        [[10, 1], [11, 1]],
        [[10, 2], [11, 2]],
        [[10, 2], [11, 3]],
    ]
    # The first answer is only legal after explicit boundary promotion.
    chunks.phase[:2].fill_(rollout._ANSWER)
    chunks.run([], [0, 1], 2)
    chunks.drain(2, False)
    assert chunks.host_metadata[2 + 8 : 2 + 8 + 2].tolist() == [
        [TOKEN_ACTION, TOKEN_ACTION],
        [-1, TOKEN_ACTION],
    ]
    assert chunks.host_metadata[2:4].tolist() == [[2, 20], [2, 2]]
    assert chunks.phase.tolist() == [rollout._IDLE] * 3
    assert chunks.host_observations[:2].tolist() == [
        [[10, 2], [11, 3]],
        [[10, 2], [11, 4]],
    ]
    # Actual bf16 pre-action states transfer even in answer-only chunks.
    assert engine.telemetry["controller_observation_transfer_bytes"] == 6 * 2 * 2 * 2
    assert engine.telemetry["graph_replays"] == 0


@pytest.mark.parametrize("head_bucket_size", [1, 16])
def test_chunk_staging_is_budget_independent_and_refill_keeps_raw_actions(
    make_engine, head_bucket_size
):
    source, engine = make_engine(
        batch=3, samples=2, chunk=4, head_bucket_size=head_bucket_size
    )
    prompt_ids = [torch.tensor([10, 8]), torch.tensor([11, 8])]
    first = engine.generate_prompt_pool(prompt_ids, 7)
    staging = engine.telemetry["staging_bytes"]
    assert [row.tolist() for row in first.responses] == [
        [8, 9, 2],
        [8, 9, 2],
        [8, 8, 9, 20, 2],
        [8, 8, 9, 20, 2],
    ]
    assert first.admission_events == 2
    first_raw = tuple(row.clone() for row in first.latent_vectors)
    first_observations = tuple(row.clone() for row in first.controller_observations)
    for index, (response, observations) in enumerate(zip(
        first.responses, first.controller_observations, strict=True,
    )):
        expected = torch.stack((
            torch.full((response.numel(),), 10 + index // 2),
            torch.arange(response.numel()),
        ), -1).bfloat16()
        assert torch.equal(observations, expected)
        assert observations.dtype == torch.bfloat16
        assert observations.device.type == "cpu"
        assert not observations.requires_grad
        kinds = first.action_kinds[index]
        thought_mask = (kinds == FIRST_THOUGHT) | (kinds == CONTINUE_THOUGHT)
        expected_raw = observations[thought_mask].float() + engine.policy.transition.offset
        expected_raw = expected_raw + torch.tensor([0.125, -0.25])
        assert torch.equal(first.latent_vectors[index], expected_raw)
    source.thinking_gate.never_stop.fill_(True)
    engine.policy.force_answers = True
    second = engine.generate_prompt_pool(prompt_ids, 20)
    assert engine.telemetry["staging_bytes"] == staging
    assert engine._chunks.host_raw.shape == (4, 3, 2)
    assert engine.telemetry["scheduling_boundaries"] < second.decode_steps
    assert all(row.numel() == 20 for row in second.responses)
    for row, saved in zip(first.latent_vectors, first_raw, strict=True):
        assert torch.equal(row, saved)
    for row, saved in zip(
        first.controller_observations, first_observations, strict=True,
    ):
        assert torch.equal(row, saved)
    for index, observations in enumerate(second.controller_observations):
        # Refilled lanes restart at the new prompt, not the retired row's state.
        # Forced close and the final answer also retain their pre-action states.
        expected = torch.stack((
            torch.full((20,), 10 + index // 2),
            torch.arange(20),
        ), -1).bfloat16()
        assert torch.equal(observations, expected)
        kinds = second.action_kinds[index]
        thought_mask = (kinds == FIRST_THOUGHT) | (kinds == CONTINUE_THOUGHT)
        expected_raw = observations[thought_mask].float() + engine.policy.transition.offset
        expected_raw = expected_raw + torch.tensor([0.125, -0.25])
        assert torch.equal(second.latent_vectors[index], expected_raw)
    assert engine.telemetry["controller_observation_staging_bytes"] == 2 * 4 * 3 * 2 * 2
    assert engine.telemetry["controller_observation_transfer_bytes"] == (
        second.capacity_row_steps * 2 * 2
    )
    for inputs, raw in zip(
        engine._decoder.stream_inputs, second.latent_vectors, strict=True
    ):
        assert torch.equal(torch.stack(inputs[: raw.size(0)]), raw.bfloat16())
        assert inputs[raw.size(0)].tolist() == [9, -100]


def test_padded_refill_always_emits_first_thought_despite_stop_gate(
    make_engine, monkeypatch
):
    _, engine = make_engine(batch=3, samples=2, chunk=4)

    def always_stop(hidden):
        return torch.ones(hidden.size(0), dtype=torch.long), torch.full(
            (hidden.size(0),), -0.25
        )

    monkeypatch.setattr(engine.policy.thinking_gate, "sample", always_stop)
    result = engine.generate_prompt_pool(
        [torch.tensor([10, 8]), torch.tensor([11, 8])], 7
    )
    assert [row.tolist() for row in result.responses] == [[8, 9, 2]] * 4
    assert [row.tolist() for row in result.action_kinds] == [
        [FIRST_THOUGHT, STOP_THINKING, TOKEN_ACTION]
    ] * 4
    assert [row[:2].tolist() for row in result.logprobs] == [[-2.0, -0.25]] * 4
    assert result.admission_events == 2
    assert [row.tolist() for row in result.controller_observations] == [
        [[10, 0], [10, 1], [10, 2]],
        [[10, 0], [10, 1], [10, 2]],
        [[11, 0], [11, 1], [11, 2]],
        [[11, 0], [11, 1], [11, 2]],
    ]


def test_short_lived_rows_do_not_pay_full_chunk_before_polling(make_engine):
    _, engine = make_engine(batch=1, samples=1, prompts=1, chunk=32)
    result = engine.generate_prompt_pool([torch.tensor([10, 8])], 7)
    assert result.responses[0].tolist() == [8, 9, 2]
    assert result.decode_steps == 3
    assert engine.telemetry["answer_projection_rows"] == 1


def test_host_scheduling_timing_excludes_chunk_work_and_splits_final_assembly(
    make_engine, monkeypatch
):
    _, engine = make_engine(batch=1, samples=1, prompts=1, chunk=8)
    clock = 0.0
    run = rollout._LatentChunks.run
    drain = rollout._LatentChunks.drain
    assemble = rollout.LatentTrainingGeneration

    def timed_run(*args):
        nonlocal clock
        run(*args)
        clock += 100.0

    def timed_drain(*args):
        nonlocal clock
        drain(*args)
        clock += 50.0

    def timed_assemble(**kwargs):
        nonlocal clock
        result = assemble(**kwargs)
        clock += 11.0
        return result

    def on_progress(*args):
        nonlocal clock
        clock += 7.0

    monkeypatch.setattr(rollout.time, "perf_counter", lambda: clock)
    monkeypatch.setattr(rollout._LatentChunks, "run", timed_run)
    monkeypatch.setattr(rollout._LatentChunks, "drain", timed_drain)
    monkeypatch.setattr(rollout, "LatentTrainingGeneration", timed_assemble)
    result = engine.generate_prompt_pool(
        [torch.tensor([10, 8])], 7, progress_callback=on_progress
    )
    assert result.responses[0].tolist() == [8, 9, 2]
    assert engine.telemetry["scheduling_boundaries"] == 3
    assert engine.telemetry["host_scheduling_seconds"] == 14.0
    assert engine.telemetry["host_finalization_seconds"] == 18.0
    assert engine.telemetry["host_tensor_assembly_seconds"] == 11.0
    assert result.decode_seconds == 471.0


def test_fake_cuda_capture_restores_each_warmup_and_discards_rng_draws(monkeypatch):
    class Stream:
        def wait_stream(self, other):
            pass

        def synchronize(self):
            pass

    chunks = rollout._LatentChunks.__new__(rollout._LatentChunks)
    chunks.engine = SimpleNamespace(telemetry={"graph_captures": 0})
    chunks.device = torch.device("cuda")
    chunks.capture_stream = Stream()
    hidden = torch.tensor([[1.0, 2.0]])
    lengths = torch.tensor([2])
    backing = torch.arange(12, dtype=torch.float32)
    prefix = backing[:2].clone()
    starts, draws = [], []
    chunks._mutable_state = lambda: [hidden, lengths]

    def execute(entry, steps):
        starts.append(int(lengths[0]))
        draws.append(torch.randn(2))
        hidden.add_(draws[-1])
        for _ in range(steps):
            backing[lengths[0]] = hidden.sum()
            lengths.add_(1)

    chunks._execute = execute
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: Stream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "CUDAGraph", SimpleNamespace)
    monkeypatch.setattr(torch.cuda, "graph", lambda graph, stream: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "get_rng_state", lambda device: torch.get_rng_state()
    )
    monkeypatch.setattr(
        torch.cuda, "set_rng_state", lambda state, device: torch.set_rng_state(state)
    )
    state = torch.get_rng_state().clone()
    entry = rollout._CapturedChunk(torch.tensor([0]), torch.empty(0, dtype=torch.long))
    chunks._capture(entry, 2)
    assert starts == [2, 2, 2]
    assert all(torch.equal(draws[0], draw) for draw in draws[1:])
    assert torch.equal(torch.get_rng_state(), state)
    assert hidden.tolist() == [[1, 2]]
    assert lengths.tolist() == [2]
    assert torch.equal(backing[:2], prefix)
    assert entry.graph is not None


@torch.inference_mode()
def test_graph_entries_keep_index_addresses_and_bound_cache_variants(make_engine):
    _, engine = make_engine(batch=2, samples=1, chunk=4, head_bucket_size=1)
    engine.max_cached_graphs = 2
    engine.generate_prompt_pool([torch.tensor([10, 8]), torch.tensor([11, 8])], 7)
    chunks = engine._chunks
    # Exercise scheduler storage reuse with native first-thought states.
    chunks.hidden.copy_(engine._decoder.admit([0, 1], [0, 1]))
    chunks.positions.zero_()
    chunks.phase.fill_(rollout._THINK)
    chunks.run([0], [], 1)
    entry = chunks.graphs[(1, 0, 1)]
    pointer = entry.thought_indices.data_ptr()
    chunks.run([1], [], 1)
    assert chunks.graphs[(1, 0, 1)] is entry
    assert entry.thought_indices.data_ptr() == pointer
    assert entry.thought_indices.tolist() == [1]
    chunks.run([0, 1], [], 2)
    chunks.run([0, 1], [], 3)
    assert len(chunks.graphs) == 2
    assert (1, 0, 1) not in chunks.graphs


@torch.inference_mode()
def test_padded_heads_share_only_idle_scratch_and_keep_physical_kv(
    make_engine, monkeypatch
):
    _, engine = make_engine(batch=3, samples=1, prompts=3, chunk=4)
    engine.generate_prompt_pool(
        [torch.tensor([token, 8]) for token in (10, 11, 12)], 7
    )
    chunks = engine._chunks
    decoder = engine._decoder
    chunks.hidden[:3].copy_(decoder.admit([0, 1, 2], [0, 1, 2]))
    chunks.phase.zero_()
    chunks.positions.zero_()
    chunks.phase[0] = rollout._THINK
    chunks.phase[1] = rollout._ANSWER
    chunks.positions[1] = 3
    decoder.closed[1] = True
    decoder.hidden[1] = torch.tensor([11, 3])
    chunks.hidden[1].copy_(decoder.hidden[1])

    # Allocate the production compact KV from the scheduler's actual trunk
    # inputs, not a claimed batch-size field or the padded head storage.
    lengths = torch.ones(3, dtype=torch.long)
    cache = rollout._CompactStaticLayer(
        16, lengths, torch.zeros((3, 16), dtype=torch.bool)
    )
    cache.prefilling = False
    advance = decoder.advance

    def cached_advance(embeddings, active):
        states = embeddings[:, None, None]
        cache.update(states, states)
        lengths.add_(active.long())
        return advance(embeddings, active)

    monkeypatch.setattr(decoder, "advance", cached_advance)
    chunks.run([0], [1], 4)
    chunks.drain(4, True)
    entry = chunks.graphs[(3, 3, 4)]
    assert entry.thought_indices.tolist() == [0, 3, 4]
    assert entry.answer_indices.tolist() == [1, 3, 4]
    assert chunks.host_metadata[2 + 4 : 2 + 4 + 4].tolist() == [
        [FIRST_THOUGHT, TOKEN_ACTION, -1],
        [STOP_THINKING, TOKEN_ACTION, -1],
        [-1, -1, -1],
        [-1, -1, -1],
    ]
    assert chunks.host_metadata[2:4, :2].tolist() == [[8, 20], [9, 2]]
    expected_raw = torch.tensor([10.0, 0.0]) + engine.policy.transition.offset
    expected_raw = expected_raw + torch.tensor([0.125, -0.25])
    assert torch.equal(chunks.host_raw[0, 0], expected_raw)
    assert chunks.host_scores[:2, 0].tolist() == [-2.0, -0.25]
    assert chunks.host_observations[:4].tolist() == [
        [[10, 0], [11, 3], [12, 0]],
        [[10, 1], [11, 4], [12, 0]],
        [[10, 2], [11, 4], [12, 0]],
        [[10, 2], [11, 4], [12, 0]],
    ]
    assert chunks.hidden.tolist() == [[10, 2], [11, 4], [12, 0], [0, 0], [0, 0]]
    assert chunks.positions.tolist() == [2, 5, 0, 0, 0]
    assert chunks.phase.tolist() == [rollout._WAIT] + [rollout._IDLE] * 4
    assert not chunks.active[3:].any()
    assert chunks.metadata[2 + 4 : 2 + 4 + 4, 3:].eq(-1).all()
    assert cache.key_backing.shape == cache.value_backing.shape == (3, 16, 1, 2)
    assert lengths.tolist() == [3, 2, 1]
    assert torch.equal(cache.key_backing[0, 1, 0], expected_raw.bfloat16())
    assert cache.key_backing[0, 2, 0].tolist() == [9, -100]
    assert cache.key_backing[1, 1, 0].tolist() == [20, -100]


@torch.inference_mode()
def test_padded_memberships_reuse_entries_without_any_thought_vocab_projection(
    make_engine
):
    _, engine = make_engine(
        batch=5, samples=1, prompts=5, chunk=4, head_bucket_size=4
    )
    engine.generate_prompt_pool(
        [torch.tensor([token, 8]) for token in range(10, 15)], 7
    )
    chunks = engine._chunks
    chunks.release()
    chunks.hidden[:5].copy_(engine._decoder.admit(range(5), range(5)))
    chunks.positions.zero_()
    chunks.phase[:5].fill_(rollout._THINK)
    engine.policy.lm_states.clear()
    before_rows = engine.telemetry["thought_head_rows"]
    before_padding = engine.telemetry["thought_head_padding_rows"]
    before_answers = engine.telemetry["answer_projection_calls"]
    chunks.run([0], [], 1)
    entry = chunks.graphs[(4, 0, 1)]
    pointer = entry.thought_indices.data_ptr()
    assert entry.thought_indices.tolist() == [0, 5, 6, 7]
    chunks.run([1, 2], [], 1)
    assert chunks.graphs[(4, 0, 1)] is entry
    assert entry.thought_indices.data_ptr() == pointer
    assert entry.thought_indices.tolist() == [1, 2, 5, 6]
    chunks.run([0, 3, 4], [], 1)
    assert chunks.graphs[(4, 0, 1)] is entry
    assert entry.thought_indices.tolist() == [0, 3, 4, 5]
    # The final partial bucket is capped at physical B, not rounded up to 8.
    chunks.run([1, 2, 3, 4], [], 1)
    chunks.run([0, 1, 2, 3, 4], [], 1)
    assert chunks.graphs[(5, 0, 1)].thought_indices.tolist() == list(range(5))
    assert set(chunks.graphs) == {(4, 0, 1), (5, 0, 1)}
    assert not engine.policy.lm_states
    assert engine.telemetry["answer_projection_calls"] == before_answers
    assert engine.telemetry["thought_head_rows"] - before_rows == 21
    assert engine.telemetry["thought_head_padding_rows"] - before_padding == 6
    assert chunks.phase[5:].tolist() == [rollout._IDLE] * 3
    assert chunks.positions[5:].tolist() == [0] * 3
    assert chunks.hidden[5:].eq(0).all()


@pytest.mark.parametrize("head_bucket_size", [0, -1, 1.5])
def test_invalid_head_bucket_size_is_rejected(make_engine, head_bucket_size):
    with pytest.raises(ValueError, match="head bucket size"):
        make_engine(head_bucket_size=head_bucket_size)


def test_compiled_execution_never_falls_back_to_fake_cpu(make_engine):
    _, engine = make_engine(batch=1, samples=1, prompts=1)
    engine.compile_decode = True
    with pytest.raises(RuntimeError, match="requires CUDA"):
        engine.generate_prompt_pool([torch.tensor([10, 8])], 7)
    assert engine._source_stash.resident
    assert not engine.policy.transition.draws
