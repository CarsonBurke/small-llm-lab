import torch
import torch.nn.functional as F

from postraining.sim_dg_sequence_credit import (
    TASKS,
    gae_advantages,
    initial_policy,
    oracle_values,
)


def _sample(task, seed, samples):
    init = initial_policy(task, seed)
    probs = F.softmax(init["logits"], -1)[None]  # one run
    generator = torch.Generator().manual_seed(seed + 100)
    drawn = torch.multinomial(probs.flatten(0, 2), samples, replacement=True, generator=generator)
    tokens = drawn.view(1, task.prompts, -1, samples).transpose(2, 3)
    stack = lambda key: init[key][None]
    noncausal = torch.ones(task.filler, dtype=torch.bool)
    context = {"target": stack("target"), "coherent": stack("coherent"), "noncausal": noncausal}
    if task.causal:
        noncausal[init["causal_positions"]] = False
        context.update(causal_positions=init["causal_positions"], causal_target=stack("causal_target"))
    return probs, tokens, context


def _expected_reward(task, tokens, context):
    """Reward with the coherence Bernoulli replaced by its success probability."""
    filler = tokens[..., : task.filler]
    if task.answer_success is None:
        correct = (filler == context["target"][:, :, None]).float()
        return (correct.cumprod(-1) if task.name == "sequential" else correct).mean(-1)
    reward = (tokens[..., -1] == context["target"][:, :, None]).float()
    if task.causal:
        positions = context["causal_positions"]
        reward = reward * (filler[..., positions] == context["causal_target"][:, :, None]).all(-1)
    if task.junk_penalty:
        in_set = (filler[..., None] == context["coherent"][:, :, None]).any(-1)
        reward = reward * in_set[..., context["noncausal"]].float().mean(-1)
    return reward


def test_oracle_values_are_exact_conditional_expectations():
    """V(s_t) equals the Monte Carlo mean reward over continuations, per state.

    Checked through the martingale property: averaging V(s_{t+1}) and the
    terminal reward over samples recovers the initial value V(s_0) for every
    prefix length, and V(s_0) matches the mean reward.
    """
    for name, task in TASKS.items():
        probs, tokens, context = _sample(task, seed=3, samples=20_000)
        values = oracle_values(task, probs, tokens, context)
        reward = _expected_reward(task, tokens, context)
        assert values.shape == tokens.shape, name
        start = values[..., 0]
        # V(s_0) does not depend on the sample.
        torch.testing.assert_close(start, start[..., :1].expand_as(start), msg=name)
        chain = torch.cat((values, reward[..., None]), -1).mean(2)  # (1, prompts, length + 1)
        tolerance = 4 * (reward.std(2).amax() / 20_000**0.5).item() + 1e-6
        assert (chain - start[..., :1]).abs().amax() < tolerance, name


def test_lambda_zero_oracle_gives_non_causal_tokens_zero_advantage():
    task = TASKS["coherence"]
    probs, tokens, context = _sample(task, seed=4, samples=64)
    # Make one filler position non-causal by removing it from the junk set.
    context["noncausal"][5] = False
    values = oracle_values(task, probs, tokens, context)
    reward = _expected_reward(task, tokens, context)
    advantages = gae_advantages(values, reward, 0.0)
    assert advantages[..., 5].abs().max() == 0
    assert advantages[..., 6].abs().max() > 0


def test_gate_temperature_scales_per_gate_and_advantage_kind():
    from postraining.sim_dg_sequence_credit import GATES, gate_temperature

    # runs: dg_group, dgrms_gae, dgbase_group, dgbase_gae, pg_gae, dgbase_group with no reward
    gates = ["dg", "dgrms", "dgbase", "dgbase", "pg", "dgbase"]
    kinds = ["group", "gae", "group", "gae", "gae", "group"]
    column = lambda flags: torch.tensor(flags)[:, None, None, None]
    gate_mask = {g: column([x == g for x in gates]) for g in GATES}
    group_mask = column([k == "group" for k in kinds])
    generator = torch.Generator().manual_seed(0)
    rewards = (torch.rand(6, 3, 4, generator=generator) < 0.3).float()
    rewards[5] = 0
    advantage = torch.randn(6, 3, 4, 5, generator=generator)
    running_mean = torch.tensor([0.1, 0.2, 0.3, 0.05, 0.5, 0.4])
    eta = 2.0
    temperature = gate_temperature(advantage, rewards, running_mean, gate_mask, group_mask, eta)
    temperature = temperature.expand(6, 3, 1, 1)
    full = lambda value: torch.full((3, 1, 1), float(value))
    torch.testing.assert_close(temperature[0], full(eta))
    torch.testing.assert_close(temperature[1], full(eta * advantage[1].square().mean().sqrt()))
    prompt_mean = rewards[2].mean(-1)[:, None, None]
    torch.testing.assert_close(temperature[2], eta * prompt_mean.where(prompt_mean > 0, 1.0))
    torch.testing.assert_close(temperature[3], full(eta * 0.05))  # the critic's level, not the pool's
    torch.testing.assert_close(temperature[4], full(eta))
    torch.testing.assert_close(temperature[5], full(eta))  # zero scale falls back to eta


def test_unknown_estimator_is_rejected():
    import argparse

    import pytest

    from postraining.sim_dg_sequence_credit import simulate

    with pytest.raises(ValueError, match="unknown estimator"):
        simulate(TASKS["lottery"], ["dgfoo_group"], argparse.Namespace())
