from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from postraining.minicpm_vapo import TrajectoryRecord
from postraining.token_carry import TokenCarryCombiner, token_carry_replay_hidden
import postraining.train_minicpm_vapo as trainer


class _CarrySide(nn.Module):
    token_carry = True
    latent_thinking = False

    def __init__(self, offset):
        super().__init__()
        self.hidden = nn.Parameter(torch.arange(64).reshape(8, 8).float() / 100 + offset)
        self.token_combiner = TokenCarryCombiner(8)
        self.nextlat_head = nn.Linear(8, 8)
        self.register_buffer("lm_head_weight", torch.eye(8))
        self.causal_lm = SimpleNamespace(config=SimpleNamespace(pad_token_id=0, hidden_size=8))

    def token_embeddings(self, ids):
        return self.hidden[ids]

    def token_carry_replay_hidden(self, batch):
        return token_carry_replay_hidden(self, batch)

    def replay_hidden(self, input_ids, attention_mask, *, inputs_embeds, sequence_boundaries, **kwargs):
        return torch.cat([
            inputs_embeds[:, start:stop].cumsum(dim=1)
            for start, stop in zip(sequence_boundaries[:-1], sequence_boundaries[1:])
        ], dim=1)

    def values(self, hidden):
        return hidden[..., 0]


@pytest.fixture
def sides(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())
    return _CarrySide(0.25), _CarrySide(-0.5)


def _records(count):
    records = []
    for index in range(count):
        length = 4 + index % 3
        prompt = 1 + index % 2
        response = length - prompt
        records.append(TrajectoryRecord(
            token_ids=(torch.arange(length, dtype=torch.int32) + index) % 8,
            prompt_length=prompt,
            old_logprobs=torch.full((response,), -2.0),
            advantages=torch.linspace(0.3, 1.0, response),
            correct=index % 2 == 0,
            text=str(index),
            carry_hiddens=(torch.arange(response * 8).reshape(response, 8) / 50 + index).bfloat16(),
        ))
    return records


_REPLAY_OPTIONS = dict(replay_token_budget=5, replay_max_trajectories=16, logit_chunk_tokens=3)


def test_refresh_preserves_carries_and_pricing_is_independent_of_packing(sides):
    actor, critic = sides
    records = _records(7)
    saved = [record.carry_hiddens.clone() for record in records]
    refreshed, _ = trainer.refresh_behavior_statistics(actor, critic, records, **_REPLAY_OPTIONS)
    wide, _ = trainer.refresh_behavior_statistics(
        actor, critic, records, **{**_REPLAY_OPTIONS, "replay_token_budget": 64},
    )
    for small_record, wide_record, original in zip(refreshed, wide, saved):
        torch.testing.assert_close(small_record.carry_hiddens, original, rtol=0, atol=0)
        torch.testing.assert_close(wide_record.carry_hiddens, original, rtol=0, atol=0)
        torch.testing.assert_close(small_record.old_logprobs, wide_record.old_logprobs)
        torch.testing.assert_close(small_record.advantages, wide_record.advantages)
    before = trainer.measure_post_update_behavior_kl(actor, refreshed, **_REPLAY_OPTIONS)
    assert before["post_update_ratio_abs_log_max"] == pytest.approx(0.0, abs=1e-6)
    with torch.no_grad():
        actor.token_combiner.carry.weight[0, 0].add_(0.2)
    after = trainer.measure_post_update_behavior_kl(actor, refreshed, **_REPLAY_OPTIONS)
    assert after["post_update_approximate_kl"] > 0
    for record, original in zip(refreshed, saved):
        torch.testing.assert_close(record.carry_hiddens, original, rtol=0, atol=0)


@pytest.mark.parametrize("value_only", [False, True])
def test_optimizer_epochs_learn_from_fixed_carries_without_changing_history(sides, value_only):
    actor, critic = sides
    records = _records(8)
    saved = [record.carry_hiddens.clone() for record in records]
    actor_before = actor.token_combiner.carry.weight.detach().clone()
    critic_before = critic.token_combiner.carry.weight.detach().clone()
    options = dict(
        **_REPLAY_OPTIONS,
        optimizer_minibatches=2, clip_low=0.2, clip_high=0.2,
        value_coefficient=1.0, nextlat_horizon=1, nextlat_samples=1,
        nextlat_mse_coefficient=1.0, nextlat_kl_coefficient=1.0,
        nextlat_kl_chunk_tokens=3, train_nextlat=False,
        grad_clip_norm=100.0, value_only=value_only,
    )
    actor_optimizer = torch.optim.SGD(actor.parameters(), lr=0.005)
    critic_optimizer = torch.optim.SGD(critic.parameters(), lr=0.005)
    for _ in range(2):
        trainer.update_step(actor, critic, records, actor_optimizer, critic_optimizer, **options)
    assert not torch.equal(critic.token_combiner.carry.weight, critic_before)
    assert torch.equal(actor.token_combiner.carry.weight, actor_before) == value_only
    for record, original in zip(records, saved):
        torch.testing.assert_close(record.carry_hiddens, original, rtol=0, atol=0)
