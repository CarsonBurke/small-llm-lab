"""The pessimistic prior on the bilinear token Q critic: CPU contracts on tiny models."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from postraining.core import actor_gae_lambdas, generalized_value_estimate_targets
from postraining.critic_study import support_census_summary
from postraining.latent_rollout import replay_head_inputs
from postraining.mean_actor_critic import SUPPORT_CENSUS_FIELDS
from postraining.nano_backbone import rational_softcap
from postraining.pessimistic_q import pessimistic_q_prior
from postraining.tests.test_mac_update import CLOSE, MAC_ARGS, _cot_batch, _cot_wrapper, _gradients, _parse
from postraining.tests.test_twin_q_critic import LAMBDA, _critic
from postraining.tests.test_q_critic import _oracle_action_offsets
from postraining.train_latent_vapo import (
    CARRIED_CRITIC_VALUE_ARGS,
    build_optimizers,
    q_prior_dashboard,
    support_dashboard,
    update_minibatch,
    validate_critic_only_init,
    validate_resume_arg_contract,
)
from postraining.value_model import SeparateCritic

BETA = 0.05
CAP = 15.0


def _penalty(raw, directions, latents, weights):
    probabilities = rational_softcap(raw, CAP).softmax(-1)
    total = 0.0
    for head in directions:
        scores = head @ latents.T
        advantages = scores - (probabilities * scores).sum(-1, keepdim=True)
        total = total + (weights[:, None] * advantages.clamp_min(0).square()).sum()
    return total


def test_the_prior_returns_the_exact_gradient_of_its_penalty():
    generator = torch.Generator().manual_seed(0)
    raw = torch.randn(7, 13, generator=generator, dtype=torch.float64) * 4
    directions = torch.randn(2, 7, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    latents = torch.randn(13, 5, generator=generator, dtype=torch.float64)
    weights = torch.rand(7, generator=generator, dtype=torch.float64)
    grad_directions, penalty, positive = pessimistic_q_prior(
        raw, CAP, directions.detach(), latents.detach(), weights)
    expected = _penalty(raw, directions, latents.detach(), weights)
    torch.testing.assert_close(grad_directions, torch.autograd.grad(expected, directions)[0])
    unweighted = _penalty(raw, directions.detach(), latents.detach(), torch.ones(7, dtype=torch.float64))
    assert float(penalty.sum()) == pytest.approx(float(unweighted), rel=1e-5)
    probabilities = rational_softcap(raw, CAP).softmax(-1)
    scores = directions.detach() @ latents.detach().T
    centered = scores - (probabilities * scores).sum(-1, keepdim=True)
    assert positive.tolist() == (centered > 0).sum((1, 2)).double().tolist()


def test_inflating_q_never_escapes_the_prior():
    # CQL(H) was met by inflating the score scale (job 14195); here that
    # scales the penalty by k^2, and a critic with no contrasts pays nothing.
    generator = torch.Generator().manual_seed(1)
    raw = torch.randn(4, 9, generator=generator) * 3
    directions = torch.randn(1, 4, 3, generator=generator)
    latents = torch.randn(9, 3, generator=generator)
    _, penalty, positive = pessimistic_q_prior(raw, CAP, directions, latents, torch.ones(4))
    _, scaled, scaled_positive = pessimistic_q_prior(raw, CAP, 10 * directions, latents, torch.ones(4))
    assert float(scaled) == pytest.approx(100 * float(penalty), rel=1e-4)
    assert scaled_positive.tolist() == positive.tolist()
    grad_directions, flat, _ = pessimistic_q_prior(
        raw, CAP, torch.zeros_like(directions), latents, torch.ones(4))
    assert float(flat) == 0 and not grad_directions.any()


def _oracle(wrapper, critic, batch):
    """The value loss plus beta * sum_a relu(A_h)^2 per head at every emit
    slot, both over the action count, with pi the actor's own readout; the
    prior reaches the critic through its directions only."""
    parameters = [p for p in critic.parameters() if p.requires_grad]
    values, directions = critic.token_value_members(batch, _oracle_action_offsets(wrapper, critic, batch))
    mask = batch.action_mask
    lambdas = actor_gae_lambdas(mask.sum(1), 0.05, LAMBDA)
    targets = generalized_value_estimate_targets(batch.rewards, values.detach().amin(0), mask, lambdas)
    count = mask.sum()
    loss = ((values - targets).square().sum(0) * mask).sum() / count
    emit = batch.emit_mask.bool()
    with torch.no_grad():
        beliefs, _ = replay_head_inputs(wrapper, batch)
        backbone = wrapper.backbone
        probabilities = rational_softcap(
            backbone._raw_logits(beliefs[emit]).float(), backbone.logit_softcap).softmax(-1)
    latents = critic.action_latents().detach()
    for head in directions:
        scores = head[emit] @ latents.T
        advantages = scores - (probabilities * scores).sum(-1, keepdim=True)
        loss = loss + BETA * advantages.clamp_min(0).square().sum() / count
    return parameters, torch.autograd.grad(loss, parameters, allow_unused=True)


def _update(wrapper, critic, batch, pessimism, max_trajectories=None):
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    wrapper.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    shards = {} if max_trajectories is None else {"replay_max_trajectories": max_trajectories}
    return update_minibatch(
        wrapper, critic, batch, optimizers, actor_step=False, critic_step=False,
        mean_actor_critic=True, mac_estimator="all_action", fused_behavior_statistics=True,
        critic_lambda=LAMBDA, think_close_id=CLOSE, mac_vocab_chunk=3,
        critic_q_pessimism=pessimism, **shards)


@pytest.mark.parametrize("max_trajectories", [None, 1])
@pytest.mark.parametrize("twin", [False, True])
def test_each_head_fits_its_target_plus_the_prior_and_the_actor_is_unchanged(twin, max_trajectories):
    wrapper = _cot_wrapper()
    critic = _critic(twin=twin)
    batch = _cot_batch(wrapper)
    assert torch.equal(batch.emit_mask.bool(), batch.action_mask.bool())
    actor_parameters = [p for p in wrapper.parameters() if p.requires_grad]
    plain_batch, prior_batch = copy.deepcopy(batch), copy.deepcopy(batch)
    plain = _update(wrapper, critic, plain_batch, 0.0, max_trajectories)
    actor_plain = _gradients(actor_parameters)
    parameters, expected = _oracle(wrapper, critic, copy.deepcopy(batch))
    metrics = _update(wrapper, critic, prior_batch, BETA, max_trajectories)
    # The critic's backward waits for the actor pass; what that pass records does not change.
    for name in ("old_values", "old_token_logprobs"):
        assert torch.equal(getattr(prior_batch, name), getattr(plain_batch, name)), name
    for got, want in zip(_gradients(parameters), expected):
        want = torch.zeros_like(got) if want is None else want
        # Gradients reach the hundreds; fp32 rounding leaves ~2e-7 of the norm.
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-6 * float(want.abs().max()))
    # The prior trains the critic only; MAC reads the same Q either way.
    for got, want in zip(_gradients(actor_parameters), actor_plain):
        torch.testing.assert_close(got, want)
    assert metrics["policy_loss"] == pytest.approx(plain["policy_loss"])
    heads = 2 if twin else 1
    assert metrics["q_prior/count"] == heads * float(batch.emit_mask.sum())
    dashboard = q_prior_dashboard([metrics])
    assert dashboard["q_prior/penalty_per_state"] > 0
    assert 0 < dashboard["q_prior/positive_tokens_per_state"] < critic.trunk.tok_emb.weight.size(0)
    assert q_prior_dashboard([plain]) == {}


def test_the_prior_refuses_updates_without_the_policy_it_reads():
    from postraining.tests.test_hidden_carry import BACKBONES

    wrapper = _cot_wrapper()
    critic = _critic(twin=True)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    batch = _cot_batch(wrapper)
    with pytest.raises(ValueError, match="MAC actor update"):
        update_minibatch(wrapper, critic, batch, optimizers, value_only=True, critic_step=False,
                         actor_step=False, critic_q_pessimism=BETA)
    with pytest.raises(ValueError, match="nonnegative"):
        update_minibatch(wrapper, critic, batch, optimizers, value_only=True, critic_step=False,
                         actor_step=False, critic_q_pessimism=-1.0)
    plain = SeparateCritic(BACKBONES["nano"][0](11)).eval()
    with pytest.raises(ValueError):
        update_minibatch(wrapper, plain, batch, build_optimizers(wrapper, plain, learning_rate=0.0, fused=False),
                         mean_actor_critic=True, mac_estimator="all_action", fused_behavior_statistics=True,
                         think_close_id=CLOSE, actor_step=False, critic_step=False, critic_q_pessimism=BETA)


def test_cli_binds_the_weight(capsys):
    assert _parse(*MAC_ARGS).critic_q_pessimism == 0.0
    assert _parse(*MAC_ARGS, "--critic-q-pessimism", "1e-3").critic_q_pessimism == 1e-3
    no_mac = tuple(arg for arg in MAC_ARGS if arg not in ("--mean-actor-critic", "--mac-estimator", "all_action"))
    for extra, message in (
        (("--critic-q-pessimism", "0.1"), "bilinear token Q"),
        ((*no_mac, "--critic-q-pessimism", "0.1"), "--mean-actor-critic"),
        ((*MAC_ARGS, "--critic-q-pessimism", "-0.1"), "nonnegative"),
        ((*MAC_ARGS, "--critic-q-pessimism", "nan"), "nonnegative"),
    ):
        with pytest.raises(SystemExit):
            _parse(*extra)
        assert message in capsys.readouterr().err


def test_resume_and_every_critic_carry_bind_the_weight():
    current = _parse(*MAC_ARGS, "--critic-q-twin", "--critic-q-pessimism", "1e-3")
    saved = {name: getattr(current, name) for name in vars(current)}
    validate_resume_arg_contract(saved, current)
    with pytest.raises(ValueError, match="critic_q_pessimism"):
        validate_resume_arg_contract(saved | {"critic_q_pessimism": 1e-4}, current)
    with pytest.raises(ValueError, match="critic_q_pessimism"):
        validate_resume_arg_contract({k: v for k, v in saved.items() if k != "critic_q_pessimism"}, current)
    # --actor-init refits or explicitly accepts a critic fit under another weight.
    assert "critic_q_pessimism" in CARRIED_CRITIC_VALUE_ARGS
    current.value_warmup_steps = 0
    with pytest.raises(ValueError, match="at least that --critic-q-pessimism"):
        validate_critic_only_init({"args": {"critic_q_pessimism": 1e-2}}, current, "data")


def test_support_census_summary_means():
    sums = np.array([10.0, 4.0, 0.5, 7.0, 7.0, 0.7, 1.0, -0.2])
    assert support_census_summary(sums) == {
        "offsupport_tokens": 10.0, "offsupport_positive_fraction": 0.4,
        "offsupport_advantage_mean": 0.05, "rank2to8_positive_fraction": 1.0,
        "rank2to8_advantage_mean": pytest.approx(0.1), "top_advantage_mean": -0.2,
    }


def test_support_dashboard_pools_the_census_over_groups():
    def group(*values):
        return {f"support/{name}_sum": value for name, value in zip(SUPPORT_CENSUS_FIELDS, values, strict=True)}

    dashboard = support_dashboard([group(10, 4, 1.0, 7, 7, 0.7, 1, -0.1),
                                   group(30, 6, -1.0, 7, 0, -0.7, 1, 0.3)])
    assert dashboard == {
        "support/offsupport_advantage_mean": 0.0, "support/offsupport_positive_fraction": 0.25,
        "support/rank2to8_advantage_mean": 0.0, "support/rank2to8_positive_fraction": 0.5,
        "support/offsupport_tokens_per_state": 20.0,
        "support/top_advantage_mean": pytest.approx(0.1),
    }
    assert support_dashboard([{"value_loss": 0.0}]) == {}
