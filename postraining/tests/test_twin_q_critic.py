"""Twin token Q heads (TD3/TD7 clipped double Q): CPU contracts on tiny models."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch

from postraining.core import generalized_value_estimate_targets
from postraining.core import actor_gae_lambdas
from postraining.tests.test_hidden_carry import BACKBONES
from postraining.tests.test_mac_update import (
    CLOSE,
    MAC_ARGS,
    _actor_parameters,
    _cot_batch,
    _cot_wrapper,
    _gradients,
    _oracle_actor_gradients,
    _parse,
)
from postraining.tests.test_q_critic import HIDDEN, _oracle_action_offsets
from postraining.train_latent_vapo import (
    STUDY_CRITIC_SCHEMA,
    belief_gain_dashboard,
    build_optimizers,
    critic_run_schema,
    twin_dashboard,
    update_minibatch,
    validate_study_critic_init,
)
from postraining.value_model import SeparateCritic
from postraining.vapo.schemas import (
    CRITIC_Q_SCHEMA,
    CRITIC_TWIN_Q_SCHEMA,
    resume_critic_schema_compatible,
)

LAMBDA = 0.5


def _critic(*, twin: bool, live: bool = True, seed: int = 0):
    factory, _ = BACKBONES["nano"]
    torch.manual_seed(seed)
    critic = SeparateCritic(factory(11), q_head_hidden=HIDDEN, q_twin=twin).eval()
    if live:
        with torch.no_grad():
            for head in critic.q_heads:
                head.state.weight.normal_(std=0.5)
                head.direction.weight.normal_(std=0.5)
    return critic


def _twin_update(wrapper, critic, batch, optimizers, **kwargs):
    return update_minibatch(
        wrapper, critic, batch, optimizers, mean_actor_critic=True,
        mac_estimator="all_action", fused_behavior_statistics=True,
        critic_lambda=LAMBDA, think_close_id=CLOSE, actor_step=False,
        critic_step=False, **kwargs,
    )


def test_twin_heads_start_independent_at_zero_and_leave_the_first_head_unchanged():
    single = _critic(twin=False, live=False)
    twin = _critic(twin=True, live=False)
    # Built last: the first head is the single-head critic's own draw.
    for left, right in zip(single.q_head.parameters(), twin.q_head.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert not torch.equal(twin.q_head.belief.weight, twin.q_twin.belief.weight)
    wrapper = _cot_wrapper()
    batch = _cot_batch(wrapper)
    values, directions = twin.token_value_members(batch, _oracle_action_offsets(wrapper, twin, batch))
    assert values.shape == (2, *batch.token_ids.shape)
    assert directions.shape == (2, *batch.token_ids.shape, twin.trunk.tok_emb.embedding_dim)
    assert not values.abs().any() and not directions.abs().any()


def test_the_actor_reads_the_twins_mean():
    critic = _critic(twin=True)
    wrapper = _cot_wrapper()
    batch = _cot_batch(wrapper)
    mean_rows = _oracle_action_offsets(wrapper, critic, batch)
    values, directions = critic.token_value_members(batch, mean_rows)
    assert not torch.allclose(values[0], values[1])
    mean_values, mean_directions = critic.token_values(batch, mean_rows)
    torch.testing.assert_close(mean_values, values.mean(0))
    torch.testing.assert_close(mean_directions, directions.mean(0))
    # Q is linear in (c, u): the mean direction values every token as the mean Q.
    latents = critic.action_latents()
    with torch.no_grad():
        member_scores = directions @ latents.T
    # fp32 dot products over the table: rounding is relative to the largest score.
    torch.testing.assert_close(mean_directions @ latents.T, member_scores.mean(0), rtol=1e-5,
                               atol=1e-6 * max(1.0, float(member_scores.abs().max())))


def test_each_twin_regresses_the_clipped_double_q_target():
    wrapper = _cot_wrapper()
    critic = _critic(twin=True)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    batch = _cot_batch(wrapper)
    oracle_batch = copy.deepcopy(batch)
    parameters = [p for p in critic.parameters() if p.requires_grad]
    values, _ = critic.token_value_members(
        oracle_batch, _oracle_action_offsets(wrapper, critic, oracle_batch))
    mask = oracle_batch.action_mask
    lambdas = actor_gae_lambdas(mask.sum(1), 0.05, LAMBDA)
    targets = generalized_value_estimate_targets(
        oracle_batch.rewards, values.detach().amin(0), mask, lambdas)
    loss = ((values - targets).square().sum(0) * mask).sum() / mask.sum()
    expected = torch.autograd.grad(loss, parameters, allow_unused=True)
    critic.zero_grad(set_to_none=True)
    metrics = _twin_update(wrapper, critic, batch, optimizers)
    actual = _gradients(parameters)
    assert any(g is not None and bool(g.abs().sum()) for g in expected)
    # fp32 rounding of Q, which the live fixture puts near 50 (a real Q
    # lies in [0, 1]), carries into every entry; fitting Q uncentred would
    # move them by their own size.
    rounding = 2 * torch.finfo(torch.float32).eps * max(1.0, float(values.detach().abs().max()))
    for got, want in zip(actual, expected):
        want = torch.zeros_like(got) if want is None else want
        torch.testing.assert_close(got, want, rtol=1e-4,
                                   atol=max(1e-6, rounding * float(want.abs().max())))
    gap = (values[0] - values[1]).detach()
    count = float(mask.sum())
    assert metrics["twin/action_count"] == count
    torch.testing.assert_close(
        metrics["twin/gap_square_sum"], float((gap.square() * mask).sum()), rtol=1e-5, atol=1e-8)
    shift = (values.mean(0) - values.amin(0)).detach()
    torch.testing.assert_close(
        metrics["twin/min_shift_sum"], float((shift * mask).sum()), rtol=1e-5, atol=1e-8)
    assert metrics["twin/min_shift_sum"] > 0


def test_the_twin_mac_actor_gradient_is_the_mean_critics():
    wrapper = _cot_wrapper()
    critic = _critic(twin=True)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    batch = _cot_batch(wrapper)
    expected = _oracle_actor_gradients(wrapper, critic, copy.deepcopy(batch))
    wrapper.zero_grad(set_to_none=True)
    _twin_update(wrapper, critic, batch, optimizers, mac_vocab_chunk=3)
    actual = _gradients(_actor_parameters(wrapper))
    assert any(bool(g.abs().sum()) for g in expected)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.double(), want.double(), atol=1e-6, rtol=1e-4)


def test_both_directions_share_the_direction_rate():
    wrapper = _cot_wrapper()
    critic = _critic(twin=True)
    optimizers = build_optimizers(wrapper, critic, learning_rate=1e-3, fused=False,
                                  critic_action_learning_rate=1e-4,
                                  critic_q_direction_learning_rate=2e-7)
    groups = optimizers["critic_action"].param_groups
    (direction,) = [g for g in groups if g.get("token_q_direction")]
    assert [id(p) for p in direction["params"]] == [
        id(weight) for head in critic.q_heads for weight in head.direction_weights]
    assert direction["lr"] == 2e-7
    rest = {id(p) for g in groups if not g.get("token_q_direction") for p in g["params"]}
    assert {id(p) for p in critic.q_twin.parameters()} - rest == {
        id(weight) for weight in critic.q_twin.direction_weights}


def test_the_twin_critic_has_its_own_strict_schema():
    assert critic_run_schema(_critic(twin=True)) == CRITIC_TWIN_Q_SCHEMA
    assert critic_run_schema(_critic(twin=False)) == CRITIC_Q_SCHEMA
    assert resume_critic_schema_compatible({"critic_schema": CRITIC_TWIN_Q_SCHEMA}, True, twin_q=True)
    assert not resume_critic_schema_compatible({"critic_schema": CRITIC_Q_SCHEMA}, True, twin_q=True)
    assert not resume_critic_schema_compatible({"critic_schema": CRITIC_TWIN_Q_SCHEMA}, True)
    with pytest.raises(ValueError, match="twin Q heads"):
        resume_critic_schema_compatible({}, False, twin_q=True)
    with pytest.raises(ValueError, match="pairs with"):
        SeparateCritic(BACKBONES["nano"][0](11), q_twin=True)


def test_twin_cli_needs_a_token_q_head(capsys):
    assert _parse(*MAC_ARGS, "--critic-q-twin").critic_q_twin
    assert not _parse(*MAC_ARGS).critic_q_twin
    with pytest.raises(SystemExit):
        _parse("--critic-q-twin")
    assert "pairs a second head" in capsys.readouterr().err


def test_a_study_critic_initializes_only_its_own_head_count():
    critic = _critic(twin=True)
    payload = {"schema": STUDY_CRITIC_SCHEMA, "checkpoint_sha256": "h",
               "critic_schema": CRITIC_Q_SCHEMA,
               "arm": {"context": "prompt", "critic_init": "actor",
                       "direction_scale": "inverse_dim", "weighting": "token",
                       "train_belief_gain": True}}
    args = _parse(*MAC_ARGS, "--critic-q-twin")
    args.base_checkpoint_sha256 = "h"
    with pytest.raises(ValueError, match="twin study critic"):
        validate_study_critic_init(payload, critic, args)


def test_belief_gain_dashboard_reads_every_token_q_head():
    critic = _critic(twin=True)
    with torch.no_grad():
        critic.q_head.belief_gain.fill_(0.25)
        critic.q_twin.belief_gain.fill_(-0.5)
    assert belief_gain_dashboard(critic) == {
        "critic_q/belief_gain": 0.25, "critic_q/belief_gain_twin": -0.5}
    assert belief_gain_dashboard(_critic(twin=False)) == {"critic_q/belief_gain": 0.0}
    assert belief_gain_dashboard(SimpleNamespace(q_head=None)) == {}


def test_twin_dashboard_pools_disagreement_over_groups():
    metrics = [
        {"twin/gap_square_sum": 4.0, "twin/min_shift_sum": 1.0, "twin/action_count": 2.0},
        {"twin/gap_square_sum": 5.0, "twin/min_shift_sum": 2.0, "twin/action_count": 1.0},
    ]
    assert twin_dashboard(metrics) == {"twin/gap_rms": math.sqrt(3.0), "twin/min_shift_mean": 1.0}
    assert twin_dashboard([{"value_loss": 0.0}]) == {}
