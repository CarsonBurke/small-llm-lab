from __future__ import annotations

import copy
import io
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import postraining.minicpm_vapo as vapo
from postraining.token_carry import TokenCarryCombiner
from postraining.minicpm_vapo import (
    LoRAConfig,
    MiniCPMVAPOCritic,
    MiniCPMVAPOPolicy,
    TrajectoryRecord,
    collate_replay_microbatch,
)


class _CausalTrunk(nn.Module):
    """Packed causal prefix sum with a normalized readout."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(17, 4)
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.layers = nn.ModuleList()
        self.calls = 0

    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None, use_cache=False, **kwargs):
        assert not use_cache
        self.calls += 1
        inputs = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        transformed = inputs + self.q_proj(inputs)
        starts = (position_ids[0] == 0).nonzero(as_tuple=True)[0].tolist()
        stops = starts[1:] + [transformed.shape[1]]
        running = torch.cat([
            transformed[:, start:stop].cumsum(dim=1)
            for start, stop in zip(starts, stops)
        ], dim=1)
        return SimpleNamespace(last_hidden_state=F.layer_norm(running, (4,)))


class _LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="llama", vocab_size=17, hidden_size=4)
        self.model = _CausalTrunk()
        self.lm_head = nn.Linear(4, 17, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens


@pytest.fixture
def models(monkeypatch):
    monkeypatch.setattr(vapo, "MINICPM5_VOCAB_SIZE", 17)
    torch.manual_seed(19)
    base = _LM()
    lora = LoRAConfig(rank=2, alpha=4, targets=("q_proj",))
    return (
        MiniCPMVAPOPolicy(copy.deepcopy(base), lora, token_carry=True),
        MiniCPMVAPOCritic(copy.deepcopy(base), lora, token_carry=True, critic_width=5),
    )


def _record(tokens, prompt):
    actions = len(tokens) - prompt
    return TrajectoryRecord(
        token_ids=torch.tensor(tokens, dtype=torch.int32), prompt_length=prompt,
        old_logprobs=torch.zeros(actions), advantages=torch.zeros(actions),
        correct=True, text="",
        carry_hiddens=(torch.arange(actions * 4).reshape(actions, 4).float() / 7 + tokens[0]).bfloat16(),
    )


def _batch(*records):
    return collate_replay_microbatch(records, range(len(records)), pad_token_id=0, device=torch.device("cpu"))


def _oracle(side, records):
    """Independently form each fixed observation stream, including plain prompts."""
    outputs = []
    for record in records:
        ids = record.token_ids[:-1].long().unsqueeze(0)
        plain = side.token_embeddings(ids)
        mixed = side.token_combiner(plain[:, record.prompt_length:], record.carry_hiddens[:-1])
        embeddings = torch.cat((plain[:, :record.prompt_length], mixed), dim=1)
        outputs.append(side.causal_lm.model(
            inputs_embeds=embeddings, position_ids=torch.arange(record.input_length).unsqueeze(0),
        ).last_hidden_state)
    return torch.cat(outputs, dim=1)


def test_combiner_starts_identity_then_learns_gate_without_training_carry_producer():
    combiner = TokenCarryCombiner(4)
    embedding = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4).requires_grad_()
    producer = torch.full((2, 4), 2.0, requires_grad=True)
    previous = producer.square()
    saved = previous.detach().clone()
    optimizer = torch.optim.SGD(combiner.parameters(), lr=0.01)
    initial_gate = combiner.gate_logit.detach().clone()
    torch.testing.assert_close(initial_gate.sigmoid(), torch.tensor(0.01))
    torch.testing.assert_close(combiner(embedding, previous), embedding, rtol=0, atol=0)
    combiner(embedding, previous).square().sum().backward()
    assert producer.grad is None
    assert embedding.grad.abs().sum() > 0
    assert combiner.token_delta.weight.grad.abs().sum() > 0
    assert combiner.carry.weight.grad.abs().sum() > 0
    assert combiner.gate_logit.grad.item() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert not torch.allclose(combiner(embedding, previous), embedding)
    combiner(embedding, previous).square().sum().backward()
    assert combiner.gate_logit.grad.abs().item() > 0
    optimizer.step()
    assert combiner.gate_logit.item() != initial_gate.item()
    assert producer.grad is None
    torch.testing.assert_close(previous, saved, rtol=0, atol=0)


@pytest.mark.parametrize("gate_logit,gate", [(-100.0, 0.0), (None, 0.01), (100.0, 1.0)])
def test_gate_scales_entire_residual_not_pretrained_embedding(gate_logit, gate):
    combiner = TokenCarryCombiner(2)
    with torch.no_grad():
        combiner.token_delta.weight.copy_(2 * torch.eye(2))
        combiner.carry.weight.copy_(3 * torch.eye(2))
        if gate_logit is not None:
            combiner.gate_logit.fill_(gate_logit)
    embedding = torch.tensor([[1.0, -2.0]])
    previous = torch.tensor([[4.0, 5.0]])
    torch.testing.assert_close(
        combiner(embedding, previous), embedding + gate * (2 * embedding + 3 * previous),
    )


@pytest.mark.parametrize("side_index", [0, 1])
def test_parallel_fixed_carry_matches_independent_streams_and_gradients(models, side_index):
    side = models[side_index]
    with torch.no_grad():
        side.token_combiner.token_delta.weight.normal_(std=0.3)
        side.token_combiner.carry.weight.normal_(std=0.2)
        side.causal_lm.model.q_proj.lora_b.normal_(std=0.1)
    reference = copy.deepcopy(side)
    records = [_record([1, 2, 3, 4, 5, 6], 2), _record([7, 8, 9, 10], 3), _record([3, 4, 5, 6], 1)]
    batch = _batch(*records)
    batch.carry_hiddens.requires_grad_()
    actual = side.token_carry_replay_hidden(batch)
    assert side.causal_lm.model.calls == 1
    expected = _oracle(reference, records)
    torch.testing.assert_close(actual, expected)
    weights = torch.linspace(-0.7, 1.3, actual.numel()).view_as(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    for name, parameter in side.named_parameters():
        expected_parameter = dict(reference.named_parameters())[name]
        if expected_parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=3e-5, atol=3e-6)
    assert batch.carry_hiddens.grad is None
    assert side.token_combiner.token_delta.weight.grad.abs().sum() > 0
    assert side.token_combiner.carry.weight.grad.abs().sum() > 0
    assert side.token_combiner.gate_logit.grad.abs().item() > 0
    assert side.causal_lm.model.q_proj.lora_b.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in models[1 - side_index].parameters())


def test_actor_update_changes_predictions_not_shared_history_or_critic(models):
    actor, critic = models
    batch = _batch(_record([1, 2, 3, 4, 5, 6], 2))
    saved = batch.carry_hiddens.clone()
    critic_before = critic.token_carry_replay_hidden(batch).detach()
    actor_before = actor.token_carry_replay_hidden(batch)
    optimizer = torch.optim.SGD(actor.actor_parameters(), lr=0.07)
    (actor_before * torch.arange(actor_before.numel()).view_as(actor_before)).sum().backward()
    optimizer.step()
    actor_after = actor.token_carry_replay_hidden(batch)
    assert not torch.allclose(actor_before, actor_after)
    torch.testing.assert_close(batch.carry_hiddens, saved, rtol=0, atol=0)
    torch.testing.assert_close(critic.token_carry_replay_hidden(batch), critic_before, rtol=0, atol=0)
    assert actor.token_combiner.token_delta.weight.data_ptr() != critic.token_combiner.token_delta.weight.data_ptr()
    assert actor.token_combiner.carry.weight.data_ptr() != critic.token_combiner.carry.weight.data_ptr()
    assert actor.token_combiner.gate_logit.data_ptr() != critic.token_combiner.gate_logit.data_ptr()


def test_prompt_boundary_and_single_response_use_plain_embeddings(models):
    actor, _ = models
    records = [_record([1, 2, 3, 4, 5], 2), _record([6, 7, 8], 2)]
    with torch.no_grad():
        actor.token_combiner.token_delta.weight.copy_(1.5 * torch.eye(4))
        actor.token_combiner.carry.weight.normal_(std=0.4)
    batch = _batch(*records)
    actual = actor.token_carry_replay_hidden(batch)
    torch.testing.assert_close(actual, _oracle(actor, records))
    for offset, record in zip(batch.sequence_boundaries[:-1], records):
        plain = actor.causal_lm.model(
            input_ids=record.token_ids[:record.prompt_length].long().unsqueeze(0),
            position_ids=torch.arange(record.prompt_length).unsqueeze(0),
        ).last_hidden_state
        torch.testing.assert_close(actual[:, offset:offset + record.prompt_length], plain)
    single = actor.token_carry_replay_hidden(_batch(records[1]))
    torch.testing.assert_close(single, actual[:, -records[1].input_length:])


def test_record_owns_ordinary_bf16_carries_and_roundtrips_exactly():
    with torch.inference_mode():
        source = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        record = TrajectoryRecord.from_device(
            token_ids=torch.tensor([1, 2, 3, 4, 5]), prompt_length=2,
            old_logprobs=torch.zeros(3), old_values=torch.zeros(3),
            correct=True, text="answer", carry_hiddens=source,
        )
    saved = record.carry_hiddens.clone()
    assert not record.carry_hiddens.is_inference()
    assert not record.carry_hiddens.requires_grad
    with torch.inference_mode():
        source.zero_()
    torch.testing.assert_close(record.carry_hiddens, saved, rtol=0, atol=0)
    assert record.storage_bytes - replace(record, carry_hiddens=None).storage_bytes == 24
    stream = io.BytesIO()
    torch.save(record, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    restored.__post_init__()
    torch.testing.assert_close(restored.carry_hiddens, saved, rtol=0, atol=0)
    batch = _batch(restored, _record([6, 7, 8], 1))
    assert batch.carry_input_positions.tolist() == [2, 3, 5]
    torch.testing.assert_close(batch.carry_hiddens[:2], saved[:-1], rtol=0, atol=0)


@pytest.mark.parametrize("invalid", [torch.zeros(2, 4, dtype=torch.bfloat16), torch.zeros(3, 4), torch.full((3, 4), float("nan"), dtype=torch.bfloat16)])
def test_record_rejects_invalid_carry_history(invalid):
    with pytest.raises(ValueError, match="carry hiddens"):
        replace(_record([1, 2, 3, 4, 5], 2), carry_hiddens=invalid)


def test_missing_and_mixed_carry_history_cannot_fall_back_to_tokens(models):
    record = _record([1, 2, 3, 4], 1)
    native = replace(record, carry_hiddens=None)
    with pytest.raises(ValueError, match="cannot mix"):
        _batch(record, native)
    for side in models:
        with pytest.raises(ValueError, match="stored carry"):
            side.token_carry_replay_hidden(_batch(native))
        with pytest.raises(ValueError, match="stored carries"):
            side.replay_hidden(torch.tensor([[1, 2, 3]]), None)


@pytest.mark.parametrize("side_index", [0, 1])
def test_checkpoint_restores_carry_behavior_and_rejects_mode_mismatch(models, side_index):
    side = models[side_index]
    restored = copy.deepcopy(side)
    with torch.no_grad():
        side.token_combiner.token_delta.weight.normal_()
        side.token_combiner.carry.weight.normal_()
        side.token_combiner.gate_logit.fill_(-0.7)
    payload = side.checkpoint_payload()
    restored.load_token_carry_state_dict(payload)
    batch = _batch(_record([1, 2, 3, 4], 1))
    torch.testing.assert_close(restored.token_carry_replay_hidden(batch), side.token_carry_replay_hidden(batch))
    for mode in (False, 1, "true"):
        with pytest.raises(ValueError, match="mode"):
            restored.load_token_carry_state_dict(dict(payload, token_carry=mode))
    with pytest.raises(ValueError, match="missing token_combiner"):
        restored.load_token_carry_state_dict({"token_carry": True})
    with pytest.raises(ValueError, match="latent thinking"):
        restored.load_token_carry_state_dict(dict(payload, latent_thinking=True))
    with pytest.raises(RuntimeError):
        restored.load_token_carry_state_dict(dict(payload, token_combiner={}))
    with pytest.raises(RuntimeError):
        restored.load_token_carry_state_dict(dict(
            payload, token_combiner={"token.weight": torch.eye(4), "carry.weight": torch.zeros(4, 4)},
        ))
    native = type(side)(_LM(), side.lora_config)
    with pytest.raises(ValueError, match="mode"):
        native.load_token_carry_state_dict(payload)
    with pytest.raises(ValueError, match="token-carry state"):
        native.load_token_carry_state_dict({"token_combiner": payload["token_combiner"]})


@pytest.mark.parametrize("model_type", [MiniCPMVAPOPolicy, MiniCPMVAPOCritic])
def test_gaussian_latent_actions_cannot_be_enabled_with_token_carry(models, model_type):
    with pytest.raises(ValueError, match="cannot be combined"):
        model_type(_LM(), models[0].lora_config, token_carry=True, latent_thinking=True)
