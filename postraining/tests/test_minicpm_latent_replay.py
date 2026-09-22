from __future__ import annotations

import copy
import pickle
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import postraining.vapo.policy as vapo
from postraining.vapo.model import readout as readout_module
from postraining.vapo.model.hf import HFCausalTrunk, HFModelSpec
from postraining.vapo.policy import (
    CONTINUE_THOUGHT,
    FIRST_THOUGHT,
    FORCED_STOP_THINKING,
    VAPOCritic,
    VAPOPolicy,
    STOP_THINKING,
    TOKEN_ACTION,
    TrajectoryRecord,
    collate_replay_microbatch,
)
from postraining.vapo.model.lora import LoRAConfig


class _Trunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(17, 4)
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.layers = nn.ModuleList()

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None, **kwargs):
        inputs = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        # A small causal trunk: changing a thought changes subsequent states,
        # while each packed trajectory starts a fresh prefix sum.
        transformed = inputs + self.q_proj(inputs)
        if position_ids is None:
            hidden = transformed.cumsum(dim=1)
        else:
            starts = (position_ids[0] == 0).nonzero(as_tuple=True)[0].tolist()
            bounds = starts + [transformed.shape[1]]
            hidden = torch.cat(
                [
                    transformed[:, start:stop].cumsum(dim=1)
                    for start, stop in zip(bounds[:-1], bounds[1:])
                ],
                dim=1,
            )
        return SimpleNamespace(last_hidden_state=hidden)


class _LM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="llama", vocab_size=17, hidden_size=4)
        self.model = _Trunk()
        self.lm_head = nn.Linear(4, 17, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens



TEST_SPEC = HFModelSpec(
    key="test",
    model_id="test/fixture",
    revision="0" * 40,
    vocab_size=17,
    requires_chat_template=False,
)


def _trunk(model):
    """Wrap a fixture causal LM in the Hugging Face trunk adapter."""
    return HFCausalTrunk(model, TEST_SPEC)


@pytest.fixture
def models(monkeypatch):
    torch.manual_seed(83)
    base = _LM()
    config = LoRAConfig(rank=2, alpha=4, targets=("q_proj",))
    actor = VAPOPolicy(_trunk(copy.deepcopy(base)), config, latent_thinking=True)
    critic = VAPOCritic(
        _trunk(copy.deepcopy(base)), config, critic_width=5, latent_thinking=True
    )
    # ValueHead intentionally starts at zero; train its readout away from zero
    # so this test measures the gradient path into the independent adapter.
    with torch.no_grad():
        critic.value_head.output.weight.normal_()
    return actor, critic


def _record(*, thoughts=2, forced=False, prompt_length=2, answers=2):
    kinds = [FIRST_THOUGHT] + [CONTINUE_THOUGHT] * (thoughts - 1)
    kinds += [FORCED_STOP_THINKING if forced else STOP_THINKING]
    kinds += [TOKEN_ACTION] * answers
    tokens = [3] * prompt_length + [4] * thoughts + [5] + [6] * answers
    return TrajectoryRecord(
        token_ids=torch.tensor(tokens, dtype=torch.int32),
        prompt_length=prompt_length,
        old_logprobs=torch.zeros(len(kinds)),
        advantages=torch.arange(len(kinds), dtype=torch.float32),
        correct=True,
        text="answer",
        forced_token_index=thoughts if forced else -1,
        action_kinds=torch.tensor(kinds, dtype=torch.int8),
        latent_vectors=torch.arange(thoughts * 4, dtype=torch.float32).view(thoughts, 4)
        / 7,
    )


def _batch(*records):
    return collate_replay_microbatch(
        records, range(len(records)), pad_token_id=0, device=torch.device("cpu")
    )


def _hidden(side, batch):
    return side.replay_hidden(
        batch.input_ids,
        None,
        position_ids=batch.position_ids,
        latent_vectors=batch.latent_vectors,
        latent_input_positions=batch.latent_input_positions,
    )


def test_latent_replay_preserves_adapter_gradients_with_input_gradient_hooks(models):
    batch = _batch(_record())
    for side in models:
        expected = _hidden(side, batch)
        expected.sum().backward()
        expected_grad = side.thought_adapter.carry.weight.grad.clone()
        side.zero_grad(set_to_none=True)

        def require_input_grad(_module, _inputs, output):
            output.requires_grad_(True)

        handle = side.causal_lm.get_input_embeddings().register_forward_hook(
            require_input_grad
        )
        try:
            actual = _hidden(side, batch)
            actual.sum().backward()
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(
                side.thought_adapter.carry.weight.grad, expected_grad
            )
            assert side.causal_lm.get_input_embeddings().weight.grad is None
        finally:
            handle.remove()


@pytest.mark.parametrize(
    "kinds",
    [
        [TOKEN_ACTION, CONTINUE_THOUGHT, STOP_THINKING, TOKEN_ACTION, TOKEN_ACTION],
        [FIRST_THOUGHT, FIRST_THOUGHT, STOP_THINKING, TOKEN_ACTION, TOKEN_ACTION],
        [FIRST_THOUGHT, STOP_THINKING, CONTINUE_THOUGHT, TOKEN_ACTION, TOKEN_ACTION],
        [FIRST_THOUGHT, CONTINUE_THOUGHT, STOP_THINKING, STOP_THINKING, TOKEN_ACTION],
        [
            FIRST_THOUGHT,
            CONTINUE_THOUGHT,
            STOP_THINKING,
            TOKEN_ACTION,
            CONTINUE_THOUGHT,
        ],
    ],
)
def test_record_rejects_invalid_latent_phase_order(kinds):
    with pytest.raises(ValueError, match="action kinds"):
        replace(_record(), action_kinds=torch.tensor(kinds, dtype=torch.int8))


def test_record_requires_exact_vectors_close_and_answer():
    record = _record()
    with pytest.raises(ValueError, match="both"):
        replace(record, latent_vectors=None)
    with pytest.raises(ValueError, match="action kinds"):
        replace(record, latent_vectors=torch.zeros(1, 4))
    with pytest.raises(ValueError, match="answer"):
        _record(thoughts=2, answers=0)
    with pytest.raises(ValueError, match="forced token"):
        replace(_record(forced=True), forced_token_index=-1)
    with pytest.raises(ValueError, match="zero policy"):
        replace(_record(forced=True), old_logprobs=torch.ones(5))
    with pytest.raises(ValueError, match="detached finite"):
        replace(record, latent_vectors=torch.full((2, 4), float("nan")))


@pytest.mark.parametrize("observation_dtype", [torch.bfloat16, torch.float64])
def test_from_device_preserves_detached_vectors_metadata_and_storage(observation_dtype):
    record = _record(forced=True)
    raw = record.latent_vectors.clone().requires_grad_()
    observations = (
        torch.arange(record.response_length * 4, dtype=observation_dtype)
        .view(record.response_length, 4).div(7).requires_grad_()
    )
    restored = TrajectoryRecord.from_device(
        token_ids=record.token_ids.long(),
        prompt_length=record.prompt_length,
        old_logprobs=record.old_logprobs,
        old_values=torch.zeros(record.response_length),
        correct=record.correct,
        text=record.text,
        forced_token_index=record.forced_token_index,
        action_kinds=record.action_kinds.long(),
        latent_vectors=raw,
        controller_observations=observations,
    )
    assert torch.equal(restored.action_kinds, record.action_kinds)
    assert torch.equal(restored.latent_vectors, raw.detach())
    assert not restored.latent_vectors.requires_grad
    assert torch.equal(restored.controller_observations, observations.detach())
    assert restored.controller_observations.dtype == observation_dtype
    assert restored.controller_observations.device.type == "cpu"
    assert not restored.controller_observations.requires_grad
    checkpoint_record = pickle.loads(pickle.dumps(restored))
    assert torch.equal(checkpoint_record.controller_observations, observations.detach())
    assert checkpoint_record.controller_observations.dtype == observation_dtype
    assert restored.storage_bytes == sum(
        t.numel() * t.element_size()
        for t in (
            restored.token_ids,
            restored.old_logprobs,
            restored.advantages,
            restored.action_kinds,
            restored.latent_vectors,
            restored.controller_observations,
        )
    )


def test_packing_aligns_latent_inputs_with_next_action_states(models):
    actor, _ = models
    records = (_record(), _record(thoughts=1, forced=True, prompt_length=3))
    packed = _batch(*records)
    assert packed.latent_input_positions.tolist() == [2, 3, 9]
    assert packed.latent_action_indices.tolist() == [0, 1, 5]
    assert packed.action_positions.tolist() == [1, 2, 3, 4, 5, 8, 9, 10, 11]
    assert packed.policy_mask.tolist() == [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
    ]
    actual = _hidden(actor, packed)
    offset = 0
    for record in records:
        separate = _hidden(actor, _batch(record))
        torch.testing.assert_close(
            actual[:, offset : offset + record.input_length], separate
        )
        offset += record.input_length
    native = replace(records[0], action_kinds=None, latent_vectors=None)
    with pytest.raises(ValueError, match="cannot mix"):
        _batch(records[0], native)


def test_cached_and_replay_embeddings_reuse_exact_raw_vectors(models):
    actor, _ = models
    batch = _batch(_record())
    raw = batch.latent_vectors
    assert torch.equal(actor.thought_embeddings(raw), raw)
    explicit = actor.token_embeddings(batch.input_ids).clone()
    explicit[0, batch.latent_input_positions] = actor.thought_embeddings(raw)
    replay = _hidden(actor, batch)
    cached = actor.cached_hidden(
        None,
        inputs_embeds=explicit,
        past_key_values=None,
        cache_position=torch.arange(explicit.shape[1]),
    )
    assert torch.equal(replay, cached)
    altered = replace(batch, latent_vectors=raw + torch.tensor([1.0, -2.0, 0.5, 3.0]))
    changed = _hidden(actor, altered)
    assert torch.equal(replay[:, :2], changed[:, :2])
    assert not torch.equal(replay[:, 2:], changed[:, 2:])


@pytest.mark.parametrize("invalid_first", [257, 1.5])
def test_from_device_rejects_action_codes_before_narrowing(invalid_first):
    record = _record()
    kinds = torch.tensor(
        [invalid_first, CONTINUE_THOUGHT, STOP_THINKING, TOKEN_ACTION, TOKEN_ACTION]
    )
    with pytest.raises(ValueError, match="integer latent"):
        TrajectoryRecord.from_device(
            token_ids=record.token_ids,
            prompt_length=record.prompt_length,
            old_logprobs=record.old_logprobs,
            old_values=torch.zeros(record.response_length),
            correct=True,
            text=record.text,
            action_kinds=kinds,
            latent_vectors=record.latent_vectors,
        )


def test_thought_input_rounding_preserves_raw_action_precision(models):
    actor, critic = models
    actor.causal_lm.to(dtype=torch.bfloat16)
    critic.causal_lm.to(dtype=torch.bfloat16)
    raw = torch.tensor([[0.1234567, -0.9876543, 1.234567, -2.345678]])
    for side in (actor, critic):
        adapted = side.thought_embeddings(raw)
        assert adapted.dtype == torch.bfloat16
        assert torch.equal(adapted, raw.bfloat16())
    assert not torch.equal(raw, raw.bfloat16().float())


def test_action_likelihood_scores_only_phase_distribution_and_trains_heads(
    models, monkeypatch
):
    actor, _ = models
    batch = _batch(_record())
    hidden = torch.randn(batch.action_count, 4, requires_grad=True)
    projected = []
    original = readout_module.chunked_frozen_head_logprobs

    def lexical(states, targets, weight, *, chunk_tokens):
        projected.append(states.shape[0])
        return original(states, targets, weight, chunk_tokens=chunk_tokens)

    monkeypatch.setattr(readout_module, "chunked_frozen_head_logprobs", lexical)
    actual = actor.action_logprobs(hidden, batch, chunk_tokens=2)
    mean = actor.transition.predict_mean(hidden[:2])
    gaussian = actor.transition.log_prob(
        batch.latent_vectors, mean, actor.transition.predict_log_sigma(hidden[:2])
    )
    expected = torch.cat(
        (
            gaussian[:1],
            gaussian[1:] + actor.thinking_gate.log_prob(torch.zeros(1), hidden[1:2]),
            actor.thinking_gate.log_prob(torch.ones(1), hidden[2:3]),
            original(
                hidden[3:], batch.targets[3:], actor.lm_head_weight, chunk_tokens=2
            ),
        )
    )
    torch.testing.assert_close(actual, expected)
    assert projected == [2]
    (-actual.sum()).backward()
    assert actor.transition.mean_head.weight.grad.abs().sum() > 0
    assert actor.thinking_gate.head.weight.grad.abs().sum() > 0
    assert actor.causal_lm.lm_head.weight.grad is None


def test_forced_close_has_no_actor_or_gate_gradient(models):
    actor, _ = models
    batch = _batch(_record(thoughts=1, forced=True, answers=1))
    hidden = torch.randn(batch.action_count, 4, requires_grad=True)
    likelihood = actor.action_logprobs(hidden, batch, chunk_tokens=2)
    assert likelihood[1].item() == 0
    likelihood.sum().backward()
    assert torch.equal(hidden.grad[1], torch.zeros(4))
    assert actor.thinking_gate.head.weight.grad is None
    assert actor.transition.mean_head.weight.grad.abs().sum() > 0


def test_critic_values_train_independent_adapter_without_actor_gradients(models):
    actor, critic = models
    batch = _batch(_record())
    source = torch.randn(2, 4, requires_grad=True)
    raw = actor.transition.predict_mean(source)
    batch = replace(batch, latent_vectors=raw)
    hidden = _hidden(critic, batch)
    values = critic.values(hidden[batch.action_batch_indices, batch.action_positions])
    changed = critic.values(
        _hidden(critic, replace(batch, latent_vectors=raw + 0.7))[
            batch.action_batch_indices, batch.action_positions
        ]
    )
    assert not torch.equal(values[1:], changed[1:])
    optimizer = torch.optim.SGD(critic.backbone_parameters(), lr=0.01)
    before = critic.thought_adapter.carry.weight.detach().clone()
    values.sum().backward()
    optimizer.step()
    assert not torch.equal(before, critic.thought_adapter.carry.weight)
    assert source.grad is None
    assert all(parameter.grad is None for parameter in actor.parameters())
    assert (
        actor.thought_adapter.carry.weight.data_ptr()
        != critic.thought_adapter.carry.weight.data_ptr()
    )


def test_actor_optimizer_trains_thought_adapter_through_replayed_actions(models):
    actor, _ = models
    batch = _batch(_record())
    hidden = _hidden(actor, batch)
    optimizer = torch.optim.SGD(actor.actor_parameters(), lr=0.001)
    before = actor.thought_adapter.carry.weight.detach().clone()
    loss = -actor.action_logprobs(
        hidden[batch.action_batch_indices, batch.action_positions],
        batch,
        chunk_tokens=2,
    ).sum()
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, actor.thought_adapter.carry.weight)


def test_latent_checkpoint_restores_behavior_and_rejects_mode_config_or_head_mismatch(
    models,
):
    actor, critic = models
    raw = torch.randn(2, 4)
    for side in (actor, critic):
        restored = copy.deepcopy(side)
        with torch.no_grad():
            side.thought_adapter.carry.weight.normal_(std=0.2)
            if side is actor:
                side.transition.mean_head.bias.add_(0.3)
                side.thinking_gate.head.bias.add_(0.4)
        payload = side.checkpoint_payload()
        restored.load_latent_state_dict(payload)
        torch.testing.assert_close(
            restored.thought_embeddings(raw), side.thought_embeddings(raw)
        )
        if side is actor:
            torch.testing.assert_close(
                restored.transition.predict_mean(raw), side.transition.predict_mean(raw)
            )
            torch.testing.assert_close(
                restored.thinking_gate.stop_logit(raw),
                side.thinking_gate.stop_logit(raw),
            )
            with pytest.raises(ValueError, match="thought_sigma"):
                restored.load_latent_state_dict(dict(payload, thought_sigma=2.0))
        with pytest.raises(ValueError, match="mode"):
            restored.load_latent_state_dict(dict(payload, latent_thinking=False))
        with pytest.raises(RuntimeError):
            restored.load_latent_state_dict(dict(payload, thought_adapter={}))


def test_native_payload_and_replay_remain_separate(models):
    actor, _ = models
    native = VAPOPolicy(_trunk(_LM()), actor.lora_config)
    payload = native.checkpoint_payload()
    assert set(payload) == {
        "trunk",
        "lora_config",
        "lora_modules",
        "nextlat_projection_factor",
        "adapter",
        "nextlat",
    }
    native.load_latent_state_dict(payload)
    with pytest.raises(ValueError, match="mode"):
        native.load_latent_state_dict(actor.checkpoint_payload())
    record = replace(_record(), action_kinds=None, latent_vectors=None)
    batch = _batch(record)
    hidden = torch.randn(batch.action_count, 4)
    expected = vapo.chunked_frozen_head_logprobs(
        hidden, batch.targets, native.lm_head_weight, chunk_tokens=2
    )
    torch.testing.assert_close(
        native.action_logprobs(hidden, batch, chunk_tokens=2), expected
    )
    with pytest.raises(ValueError, match="native-token"):
        actor.action_logprobs(hidden, batch)
    with pytest.raises(ValueError, match="native policy"):
        native.action_logprobs(hidden, _batch(_record()))
