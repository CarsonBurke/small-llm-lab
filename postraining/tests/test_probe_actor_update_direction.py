"""CPU contracts of ``postraining.probe_actor_update_direction``."""

from __future__ import annotations

import copy

import pytest
import torch

import postraining.train_latent_vapo as tlv
from postraining.latent_rollout import assign_terminal_rewards, refresh_old_statistics
from postraining.probe_actor_update_direction import (
    compare,
    estimator_weight,
    evaluate,
    patched_actor_loss,
    within_row_variance,
)
from postraining.tests.test_hidden_carry import _prompts, _roll
from postraining.tests.test_memory_carry import _memory_critic, _memory_wrapper


def test_estimator_weights():
    advantages = torch.tensor([0.5, -0.2, 0.0, 1.0])
    surprisal = torch.tensor([1.0, 8.0, 3.0, 20.0])
    noise = torch.tensor(0.01)
    pg = estimator_weight("pg").weight
    dg = estimator_weight("dg").weight
    tilt = estimator_weight("tilt").weight
    torch.testing.assert_close(pg(advantages, surprisal, noise), advantages)
    torch.testing.assert_close(dg(advantages, surprisal, noise), torch.sigmoid(advantages * surprisal) * advantages)
    torch.testing.assert_close(tilt(advantages, surprisal, noise) + 0.5 * advantages, dg(advantages, surprisal, noise))
    cap = estimator_weight("dg_cap4").weight
    torch.testing.assert_close(
        cap(advantages, surprisal, noise), torch.sigmoid(advantages * surprisal.clamp(max=4)) * advantages
    )
    eta = estimator_weight("dg_eta4_perm_td0")
    assert eta.permute and eta.td0 and not eta.row
    torch.testing.assert_close(
        eta.weight(advantages, surprisal, noise), torch.sigmoid(advantages * surprisal / 4) * advantages
    )
    debias = estimator_weight("dg_eta2_debias_row")
    assert debias.row
    torch.testing.assert_close(
        debias.weight(advantages, surprisal, noise),
        torch.sigmoid(advantages * surprisal / 2) * advantages - surprisal * noise / 8,
    )
    for bad in ("pg_eta2", "pg_cap3", "pg_debias", "dg_eta0", "foo"):
        with pytest.raises(ValueError):
            estimator_weight(bad)


def test_debias_removes_dg_noise_rectification():
    # Zero-mean noise at fixed surprisal: DG's mean weight is ~ l sigma^2 / 4
    # above PG's; the debiased weight's mean returns to ~0.
    generator = torch.Generator().manual_seed(0)
    sigma = 0.02  # A·l ~ 0.2: the second-order expansion holds
    advantages = sigma * torch.randn(200_000, generator=generator)
    surprisal = torch.full_like(advantages, 9.0)
    variance = advantages.var()
    dg = estimator_weight("dg").weight(advantages, surprisal, variance)
    debiased = estimator_weight("dg_debias").weight(advantages, surprisal, variance)
    assert dg.mean() == pytest.approx(9.0 * sigma**2 / 4, rel=0.05)
    assert abs(float(debiased.mean())) < 0.05 * 9.0 * sigma**2 / 4


def test_within_row_variance_and_row_mean_advantages():
    advantages = torch.tensor([[1.0, 3.0, 0.0], [2.0, 2.0, 2.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    # Row 0 deviations +-1 (2 tokens, 1 dof), row 1 none (2 dof): 2 / 3.
    assert float(within_row_variance(advantages, mask)) == pytest.approx(2 / 3)
    captured = {}

    def spy(a, l, noise):
        captured["a"] = a
        return a

    estimator = estimator_weight("pg_row")._replace(weight=spy)
    patched_actor_loss(estimator, torch.Generator())(torch.zeros(2, 3), advantages, mask)
    torch.testing.assert_close(captured["a"], torch.tensor([[2.0, 2.0, 0.0], [2.0, 2.0, 2.0]]))


def test_patched_loss_gradient_and_permutation():
    logits = torch.randn(2, 5, requires_grad=True)
    logprobs = logits.log_softmax(-1)[:, 0]
    advantages = torch.tensor([0.3, -0.4])
    mask = torch.ones(2)
    estimator = estimator_weight("dg")
    loss, diagnostics = patched_actor_loss(estimator, torch.Generator())(logprobs[None], advantages[None], mask[None])
    assert "gate_sum" in diagnostics
    expected = -(estimator.weight(advantages, -logprobs.detach(), None) * logprobs).sum() / 2
    torch.testing.assert_close(loss, expected)

    values = torch.arange(1.0, 7.0).reshape(2, 3)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    captured = {}

    def spy(advantages, surprisal, noise):
        captured["a"] = advantages
        return advantages

    permuted_estimator = estimator_weight("pg_perm")._replace(weight=spy)
    patched_actor_loss(permuted_estimator, torch.Generator().manual_seed(0))(torch.zeros(2, 3), values, mask)
    permuted = captured["a"]
    assert permuted[0, 2] == values[0, 2]
    assert sorted(permuted[mask.bool()].tolist()) == sorted(values[mask.bool()].tolist())


def test_patched_update_moves_the_actor_and_evaluation_is_deterministic():
    wrapper = _memory_wrapper()
    critic = _memory_critic(wrapper)
    batch = _roll(wrapper, _prompts("nano", 4, 5), new_tokens=8)
    assign_terminal_rewards(batch, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    refresh_old_statistics(wrapper, critic, batch)
    eval_batch = _roll(wrapper, _prompts("nano", 3, 5, seed=9), new_tokens=8, seed=3)
    base = evaluate(wrapper, [eval_batch], (1, 2))
    again = evaluate(wrapper, [eval_batch], (1, 2))
    for key in base:
        assert torch.equal(base[key], again[key]), key
    pristine = copy.deepcopy(wrapper.state_dict())
    original = tlv.delightful_policy_loss
    tlv.delightful_policy_loss = patched_actor_loss(estimator_weight("pg"), torch.Generator())
    try:
        optimizers = tlv.build_optimizers(wrapper, critic, learning_rate=1e-2, fused=False)
        tlv.update_minibatch(
            wrapper, critic, batch, optimizers, critic_step=False,
            policy_action_denominator=batch.emit_mask.sum(), delightful_policy_gradient=True,
        )
    finally:
        tlv.delightful_policy_loss = original
    assert tlv.delightful_policy_loss is original
    stepped = evaluate(wrapper, [eval_batch], (1, 2))
    record = compare(base, stepped)
    assert record["entropy_delta"] != 0.0
    assert set(record) >= {"junk_mass_delta", "stop_logprob_delta", "kl_base_to_stepped"}
    wrapper.load_state_dict(pristine)
    restored = evaluate(wrapper, [eval_batch], (1, 2))
    assert torch.equal(restored["sampled"], base["sampled"])


def test_make_actor_trainable_restores_the_trainer_optimizer_partition():
    # load_run freezes every parameter (model_io.load_model); the probe must
    # restore the trainer's trainable set before building the Muon layout.
    wrapper = _memory_wrapper()
    critic = _memory_critic(wrapper)
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError, match="empty parameter list"):
        tlv.build_optimizers(
            wrapper, critic, 1e-5, trunk_optimizer="muon",
            muon_learning_rate=1e-5, critic_muon_learning_rate=1e-5, fused=False,
        )
    tlv.make_actor_trainable(wrapper)
    assert all(parameter.requires_grad for parameter in wrapper.parameters())
    optimizers = tlv.build_optimizers(
        wrapper, critic, 1e-5, trunk_optimizer="muon",
        muon_learning_rate=1e-5, critic_muon_learning_rate=1e-5, fused=False,
    )
    assert {"actor", "actor_muon", "critic", "critic_muon"} <= optimizers.keys()
