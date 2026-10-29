import math

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


def test_oracle_action_values_match_state_values_and_substitution():
    """Q(s_t, a) = V(s_{t+1}) with token t set to a; V(s_t) = pi_t . Q(s_t, .)."""
    from postraining.sim_dg_sequence_credit import oracle_action_values

    for name, task in TASKS.items():
        probs, tokens, context = _sample(task, seed=5, samples=32)
        values = oracle_values(task, probs, tokens, context)
        alpha, beta, feature = oracle_action_values(task, probs, tokens, context)
        expected_pi = (probs * feature).sum(-1)[:, :, None]
        torch.testing.assert_close(alpha + beta * expected_pi, values, msg=name)
        # Substitute a random token at every position in turn.
        generator = torch.Generator().manual_seed(6)
        length = tokens.shape[-1]
        for t in range(length):
            swapped = tokens.clone()
            swapped[..., t] = torch.randint(0, task.vocab, tokens.shape[:-1], generator=generator)
            q = alpha[..., t] + beta[..., t] * feature[:, :, t].gather(-1, swapped[..., t])
            if t + 1 < length:
                target = oracle_values(task, probs, swapped, context)[..., t + 1]
            else:
                target = _expected_reward(task, swapped, context)
            torch.testing.assert_close(q, target, msg=f"{name} position {t}")


def test_exact_token_q_advantage_is_lambda_zero_oracle():
    """Q(s, a_t) - pi . Q(s, .) equals TD(0) on exact values against the
    expected reward (the realized reward differs only by the coherence draw)."""
    from postraining.sim_dg_sequence_credit import oracle_action_values

    for name, task in TASKS.items():
        probs, tokens, context = _sample(task, seed=7, samples=16)
        alpha, beta, feature = oracle_action_values(task, probs, tokens, context)
        taken = feature.gather(-1, tokens.transpose(2, 3)).transpose(2, 3)
        token_q = beta * (taken - (probs * feature).sum(-1)[:, :, None])
        values = oracle_values(task, probs, tokens, context)
        td0 = gae_advantages(values, _expected_reward(task, tokens, context), 0.0)
        torch.testing.assert_close(token_q, td0, msg=name)


def test_entropy_drift_is_first_order_entropy_change():
    from postraining.sim_dg_sequence_credit import entropy, entropy_drift

    task = TASKS["lottery"]
    generator = torch.Generator().manual_seed(8)
    logits = torch.randn(2, 3, task.filler + 1, task.vocab, generator=generator, dtype=torch.float64)
    grad = torch.randn(logits.shape, generator=generator, dtype=torch.float64)
    noncausal = torch.ones(task.filler, dtype=torch.bool)
    drift = entropy_drift(F.log_softmax(logits, -1), grad, task, noncausal)
    eps = 1e-6
    change = (entropy(logits - eps * grad) - entropy(logits + eps * grad)) / (2 * eps)
    torch.testing.assert_close(drift, change[:, :, : task.filler].mean((1, 2)))


def test_new_estimators_run_and_existing_ones_are_unchanged():
    import argparse

    from postraining.sim_dg_sequence_credit import simulate

    args = argparse.Namespace(
        policy="mlp", features=8, hidden=16, optimizer="adam", lr=1e-3, eta=1.0, value_noise=0.01,
        oracle_lambda=0.0, oracle_value_noise=0.01, oracle_shrink=0.3, steps=2, samples=4, seeds=1, log_every=1,
    )
    task = TASKS["chain_coherence"]
    old = ["pg_group", "dg_oracle"]
    new = ["pg_qprop", "dg_qprop", "pg_tokenq", "dg_tokenq", "dgquad_group", "dgexact_group", "dgprefix_group", "dg_loo", "pg_groupnorm", "dg_split", "pg_split"]
    alone = simulate(task, old, args)["history"]
    mixed = simulate(task, new, args)["history"]
    for row in mixed:
        for key, values in row.items():
            if key not in ("step", "gate_pos_mean", "gate_neg_mean"):
                assert all(math.isfinite(v) for v in values), key
    # Same estimators, same draws: rerunning reproduces the history exactly.
    assert simulate(task, old, args)["history"] == alone


def test_exact_centring_zeroes_a_null_token_within_its_group():
    """Given the group's rewards, a token whose surprisal is the same in every
    sample gets mean DG coefficient exactly the null coefficient at p_hat."""
    from postraining.sim_dg_sequence_credit import null_coefficient

    rewards = torch.tensor([1.0] + [0.0] * 15, dtype=torch.float64)
    p_hat = rewards.mean()
    for scale in (0.5, 3.0, 10.0):
        u = rewards - p_hat
        mean = (u * torch.sigmoid(u * scale)).mean()
        torch.testing.assert_close(null_coefficient(p_hat, p_hat, torch.tensor(scale, dtype=torch.float64)), mean)
    # Small delight: the quadratic approximation s Var(U) / 4.
    s = torch.tensor(1e-3, dtype=torch.float64)
    torch.testing.assert_close(null_coefficient(p_hat, p_hat, s), s * p_hat * (1 - p_hat) / 4, rtol=1e-3, atol=0)
