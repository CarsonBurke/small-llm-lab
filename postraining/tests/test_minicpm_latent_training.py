from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from postraining.latent_thought import GaussianTransitionHead, StopThinkingGate
from postraining.minicpm_vapo import (
    MiniCPMVAPOPolicy,
    NextLatAuxiliaryHead,
    TrajectoryRecord,
    FIRST_THOUGHT,
    CONTINUE_THOUGHT,
    STOP_THINKING,
    FORCED_STOP_THINKING,
    TOKEN_ACTION,
    collate_replay_microbatch,
)
from postraining.train_minicpm_vapo import (
    _replay_hidden,
    _nextlat_training_loss,
    _nextlat_record_capacity,
    _nextlat_shard_samples,
    _validate_args,
    build_parser,
    refresh_behavior_statistics,
    update_step,
    validate_resume_configuration,
    measure_post_update_behavior_kl,
)


class ReplaySide(nn.Module):
    """Small causal state accumulator for testing the real trainer objectives."""

    action_logprobs = MiniCPMVAPOPolicy.action_logprobs

    def __init__(self):
        super().__init__()
        self.latent_thinking = True
        self.embedding = nn.Embedding(16, 4)
        nn.init.constant_(self.embedding.weight, 0.05)
        self.thought_adapter = nn.Linear(4, 4, bias=False)
        nn.init.eye_(self.thought_adapter.weight)
        self.transition = GaussianTransitionHead(4, 1.0)
        self.thinking_gate = StopThinkingGate(4, 0.5)
        self.value_head = nn.Linear(4, 1, bias=False)
        nn.init.constant_(self.value_head.weight, 0.1)
        self.nextlat_head = NextLatAuxiliaryHead(4, 1.0)
        self.register_buffer(
            "lm_head_weight", torch.arange(64).view(16, 4).float() / 100
        )
        self.causal_lm = SimpleNamespace(config=SimpleNamespace(pad_token_id=0))

    def replay_hidden(
        self,
        input_ids,
        attention_mask,
        *,
        latent_vectors=None,
        latent_input_positions=None,
        sequence_boundaries=None,
        **kwargs,
    ):
        embedded = self.embedding(input_ids)
        assert sequence_boundaries is not None
        if latent_vectors is not None:
            embedded = embedded.clone()
            embedded[0, latent_input_positions] = self.thought_adapter(
                latent_vectors.detach()
            )
        return torch.cat(
            [
                embedded[:, start:stop].cumsum(dim=1)
                for start, stop in zip(
                    sequence_boundaries[:-1], sequence_boundaries[1:]
                )
            ],
            dim=1,
        )

    def token_embeddings(self, ids):
        return self.embedding(ids)

    def values(self, hidden):
        return self.value_head(hidden).squeeze(-1)


def record(*, forced=False):
    return TrajectoryRecord.from_device(
        token_ids=torch.tensor([1, 8, 8, 8, 9, 3, 4, 2]),
        prompt_length=2,
        old_logprobs=torch.zeros(6),
        old_values=torch.zeros(6),
        correct=True,
        text="</think> Answer: 42",
        forced_token_index=2 if forced else -1,
        action_kinds=torch.tensor(
            [
                FIRST_THOUGHT,
                CONTINUE_THOUGHT,
                FORCED_STOP_THINKING if forced else STOP_THINKING,
                TOKEN_ACTION,
                TOKEN_ACTION,
                TOKEN_ACTION,
            ],
            dtype=torch.int8,
        ),
        latent_vectors=torch.tensor([[0.2, -0.1, 0.3, 0.1], [0.1, 0.2, -0.2, 0.4]]),
    )


@pytest.mark.parametrize("forced", [False, True])
def test_latent_replay_trains_actor_and_independent_critic(monkeypatch, forced):
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())
    actor: Any = ReplaySide()
    critic: Any = ReplaySide()
    original = record(forced=forced)
    assert original.latent_vectors is not None
    refreshed, _ = refresh_behavior_statistics(
        actor,
        critic,
        [original],
        replay_token_budget=16,
        replay_max_trajectories=1,
        logit_chunk_tokens=2,
    )
    torch.testing.assert_close(refreshed[0].latent_vectors, original.latent_vectors)
    torch.testing.assert_close(refreshed[0].action_kinds, original.action_kinds)
    kl = measure_post_update_behavior_kl(
        actor,
        refreshed,
        replay_token_budget=16,
        replay_max_trajectories=1,
        logit_chunk_tokens=2,
    )
    assert kl["post_update_ratio_abs_log_max"] == pytest.approx(0.0)
    actor_mean_before = actor.transition.mean_head.weight.detach().clone()
    gate_before = actor.thinking_gate.head.weight.detach().clone()
    critic_adapter_before = critic.thought_adapter.weight.detach().clone()
    batch = collate_replay_microbatch(
        refreshed, [0], pad_token_id=0, device=torch.device("cpu")
    )
    with torch.no_grad():
        before = critic.values(_replay_hidden(critic, batch))
        changed = replace(original, latent_vectors=original.latent_vectors + 0.5)
        changed_batch = collate_replay_microbatch(
            [changed], [0], pad_token_id=0, device=torch.device("cpu")
        )
        after = critic.values(_replay_hidden(critic, changed_batch))
    torch.testing.assert_close(before[:, :2], after[:, :2])
    assert not torch.allclose(before[:, 2:], after[:, 2:])
    metrics = update_step(
        actor,
        critic,
        refreshed,
        torch.optim.SGD(actor.parameters(), lr=0.01),
        torch.optim.SGD(critic.parameters(), lr=0.01),
        optimizer_minibatches=1,
        replay_token_budget=16,
        replay_max_trajectories=1,
        logit_chunk_tokens=2,
        clip_low=0.2,
        clip_high=0.28,
        value_coefficient=1.0,
        nextlat_horizon=1,
        nextlat_samples=2,
        nextlat_mse_coefficient=1.0,
        nextlat_kl_coefficient=1.0,
        nextlat_kl_chunk_tokens=2,
        train_nextlat=True,
        grad_clip_norm=1.0,
    )
    assert not torch.equal(actor.transition.mean_head.weight, actor_mean_before)
    assert not torch.equal(actor.thinking_gate.head.weight, gate_before)
    assert not torch.equal(critic.thought_adapter.weight, critic_adapter_before)
    assert metrics["ratio_mean"] == pytest.approx(1.0)
    assert metrics["replay_policy_actions"] == 6 - int(forced)
    assert all(torch.isfinite(parameter).all() for parameter in actor.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in critic.parameters())


def test_latent_nextlat_does_not_project_thought_states(monkeypatch):
    actor: Any = ReplaySide()
    batch = collate_replay_microbatch(
        [record()], [0], pad_token_id=0, device=torch.device("cpu")
    )
    hidden = _replay_hidden(actor, batch)
    seen = []
    original = actor.token_embeddings

    def capture(ids):
        seen.extend(ids.tolist())
        return original(ids)

    monkeypatch.setattr(actor, "token_embeddings", capture)
    loss = _nextlat_training_loss(
        actor,
        hidden,
        batch,
        max_samples=8,
        horizon=1,
        mse_coefficient=1.0,
        kl_coefficient=1.0,
        kl_chunk_tokens=2,
    )
    assert loss.transitions == 8
    assert set(seen) <= {3, 4}
    assert 8 not in seen and 9 not in seen


def test_native_resume_cannot_silently_enable_latent_thinking():
    native = build_parser().parse_args([])
    latent = build_parser().parse_args(["--latent-thinking"])
    _validate_args(latent)
    with pytest.raises(ValueError, match="latent_thinking"):
        validate_resume_configuration(
            {"args": vars(native), "pending_records": None}, latent
        )
    assert not native.latent_thinking
    assert latent.latent_thinking


def test_thought_only_shard_cannot_steal_answer_transition_budget():
    long_thinking = TrajectoryRecord.from_device(
        token_ids=torch.tensor([1, 8] + [8] * 8 + [9, 2]),
        prompt_length=2,
        old_logprobs=torch.zeros(10),
        old_values=torch.zeros(10),
        correct=False,
        text="</think>",
        action_kinds=torch.tensor(
            [FIRST_THOUGHT] + [CONTINUE_THOUGHT] * 7 + [STOP_THINKING, TOKEN_ACTION],
            dtype=torch.int8,
        ),
        latent_vectors=torch.ones(8, 4),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        allocation = _nextlat_shard_samples(
            [_nextlat_record_capacity(item, 1) for item in (long_thinking, record())],
            max_samples=8,
        )
    assert allocation == (0, 8)


def test_compact_nextlat_sampling_preserves_seeded_sequence_boundaries(monkeypatch):
    actor: Any = ReplaySide()
    batch = collate_replay_microbatch(
        [record(), record()],
        [0, 1],
        pad_token_id=0,
        device=torch.device("cpu"),
    )
    hidden = torch.arange(14, dtype=torch.float32)[None, :, None].expand(1, 14, 4)
    observed = []
    original = actor.nextlat_head.forward

    def capture(states, embeddings):
        observed.append(states[:, 0].detach().clone())
        return original(states, embeddings)

    monkeypatch.setattr(actor.nextlat_head, "forward", capture)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        # At horizon two, each three-token answer has exactly one valid start.
        expected = torch.tensor([4.0, 11.0])[torch.randint(2, (16,))]
        torch.manual_seed(11)
        result = _nextlat_training_loss(
            actor,
            hidden,
            batch,
            max_samples=16,
            horizon=2,
            mse_coefficient=1.0,
            kl_coefficient=1.0,
            kl_chunk_tokens=2,
        )
    torch.testing.assert_close(observed[0], expected, rtol=0, atol=0)
    assert result.transitions == 32
