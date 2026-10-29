"""The token Q's action table is the critic trunk's fixed readout, and its
fit is centred under the policy (critic schema v7).

At determined answer starts the v4 critic, whose table was the trunk's input
latents, ranked the correct token first at 14-26% of states where the actor
did at 94% (NOTES 2026-10-20). These tests pin what the readout table, the
linear direction path and the belief gain are for, the one configuration they
rule out, and why the fit reads Q = c + u.(x - xbar) rather than c + u.x.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from postraining.latent_rollout import compact_slots, replay_head_inputs, slot_index
from postraining.latent_thought import LatentThoughtModel
from postraining.tests.test_hidden_carry import NANO_KWARGS
from postraining.tests.test_mac_update import _cot_batch, _cot_wrapper
from postraining.tests.test_q_critic import HIDDEN, _oracle_action_offsets, _q_critic
from postraining.value_model import SeparateCritic, TokenQHead, next_slot_tokens, policy_centered_q
from postraining.nano_backbone import NanoGPTBackbone


def _oracle_policy_mean_rows(wrapper, critic, batch):
    """sum_a pi(a | s_t) x(a) at every emit slot, fp64 brute force."""
    with torch.no_grad():
        beliefs, inputs = replay_head_inputs(wrapper, batch)
        index = slot_index(batch.emit_mask.bool())
        logits = wrapper.backbone.logits_from_features(wrapper.renderer_features(
            compact_slots(inputs, index), compact_slots(beliefs, index))).double()
        return (logits.softmax(-1) @ critic.action_latents().double()).float()


def test_the_table_is_the_readout_over_one_scale():
    critic = _q_critic(_cot_wrapper())
    weight = critic.trunk.proj.weight
    table = critic.action_latents()
    eps = torch.finfo(torch.float32).eps
    rms = (weight.detach().float().square().mean() + eps).sqrt()
    torch.testing.assert_close(table, weight.float() / rms)
    assert not table.requires_grad
    assert float(table.square().mean()) == pytest.approx(1.0, rel=1e-4)
    # One scale for every row: relative row norms, how strongly a belief
    # selects each token, are the readout's.
    torch.testing.assert_close(table.norm(dim=1) / table.norm(dim=1).max(),
                               weight.float().norm(dim=1) / weight.float().norm(dim=1).max())


def test_the_taken_tokens_q_reads_the_same_table():
    wrapper = _cot_wrapper()
    critic = _q_critic(wrapper)
    batch = _cot_batch(wrapper)
    offsets = _oracle_action_offsets(wrapper, critic, batch)
    with torch.no_grad():
        values, _ = critic.token_values(batch, offsets)
        beliefs = critic.trunk.temporal_belief_from_token_latent(critic.assemble_inputs(batch))
        states, directions = critic.q_head(beliefs.float(), None)
        rows = critic.action_latents()[next_slot_tokens(batch.token_ids)]
    torch.testing.assert_close(values, states + (directions * offsets).sum(-1))
    # The offsets are the taken token's own row against the policy's mean.
    emit = batch.emit_mask.bool()
    mean_rows = rows[emit] - offsets[emit]
    torch.testing.assert_close(mean_rows, _oracle_policy_mean_rows(wrapper, critic, batch),
                               rtol=1e-5, atol=1e-6)


def test_the_table_takes_no_gradient_and_the_heads_do():
    """A trained table moved a row about 3.16 lr on its first touch under
    AdamW and gained nothing in the toy (NOTES 2026-10-20): the table is
    the readout, fixed; only the head learns where to read it."""
    wrapper = _cot_wrapper()
    critic = _q_critic(wrapper)
    batch = _cot_batch(wrapper)
    values, _ = critic.token_values(batch, _oracle_action_offsets(wrapper, critic, batch))
    values[batch.action_mask.bool()].sum().backward()
    assert critic.trunk.proj.weight.grad is None
    assert critic.trunk.proj.bias.grad is None
    assert critic.q_head.state.weight.grad.any()
    assert any(weight.grad.any() for weight in critic.q_head.direction_weights)


def test_u_as_the_belief_makes_q_the_trunks_logit():
    """u = s b, with s the table's RMS, gives u . x(a) = b . w(a), the state-
    dependent part of the trunk's own logit, through the linear path alone."""
    critic = _q_critic(_cot_wrapper(), live=False)
    head = critic.q_head
    weight = critic.trunk.proj.weight.detach().float()
    rms = (weight.square().mean() + torch.finfo(torch.float32).eps).sqrt()
    with torch.no_grad():
        head.belief_direction.weight.copy_(rms * torch.eye(weight.size(1)))
        beliefs = torch.randn(32, weight.size(1))
        _, directions = head(beliefs, None)
        scores = directions @ critic.action_latents().T
        logits = critic.trunk._raw_logits(beliefs) - critic.trunk.proj.bias
    torch.testing.assert_close(scores, logits, rtol=1e-5, atol=1e-5)


def test_the_belief_gain_scales_the_trunks_centred_logit():
    """u = g b gives u . (x(a) - xbar) = (g / r)(z(a) - zbar): one scalar
    reads the trunk's own ranking, centred under the policy."""
    critic = _q_critic(_cot_wrapper(), live=False)
    head = critic.q_head
    weight = critic.trunk.proj.weight.detach().float()
    rms = (weight.square().mean() + torch.finfo(torch.float32).eps).sqrt()
    gain = 0.37
    with torch.no_grad():
        head.belief_gain.fill_(gain)
        beliefs = torch.randn(32, weight.size(1))
        _, directions = head(beliefs, None)
        table = critic.action_latents()
        logits = critic.trunk._raw_logits(beliefs) - critic.trunk.proj.bias
        pi = torch.randn(32, table.size(0)).softmax(-1)
        scores = torch.einsum("sd,sad->sa", directions, table[None] - (pi @ table)[:, None])
    centred = logits - (pi * logits).sum(-1, keepdim=True)
    torch.testing.assert_close(scores, gain / rms * centred, rtol=1e-4, atol=1e-4)


def _expected_gain_gradient(returns: Tensor, seed: int = 11):
    """d/dg of the fresh head's Monte Carlo loss in expectation over the
    policy, through the head itself; the policy's logits are the beliefs'
    scores on the table."""
    generator = torch.Generator().manual_seed(seed)
    vocab, dim, states = 128, 16, 24
    table = torch.randn(vocab, dim, generator=generator)
    table = table / table.square().mean().sqrt()
    beliefs = 2.0 * torch.randn(states, dim, generator=generator)
    logits = beliefs @ table.T
    pi = logits.softmax(-1)
    head = TokenQHead(dim, 32, False)
    c, u = head(beliefs, None)
    q = policy_centered_q(c[None, :, None].expand(1, states, vocab),
                          u[None, :, None].expand(1, states, vocab, dim),
                          table[None] - (pi @ table)[:, None])[0]
    loss = (pi * (q - returns(logits)).square()).sum()
    (gradient,) = torch.autograd.grad(loss, head.belief_gain)
    return float(gradient), logits, pi


def test_the_belief_gain_moves_only_where_returns_follow_the_trunks_ranking():
    """From a fresh head, on-policy, E_pi[d/dg] = -2 sum_s Cov_pi(z, G):
    zero when returns ignore the token, and negative (g grows) when they
    reward the trunk's argmax. This is the expectation at initialization
    only: per-sample gradients are nonzero under blind returns, so Adam
    still walks g, and once D, L or c are nonzero, or the sampler is not
    the policy whose mean row centres x, g's gradient is no longer this
    covariance alone."""
    def blind(logits):
        return torch.rand(logits.size(0), 1, generator=torch.Generator().manual_seed(2)).expand_as(logits)

    def argmax(logits):
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.size(-1)).float()

    blind_gradient, _, _ = _expected_gain_gradient(blind)
    argmax_gradient, logits, pi = _expected_gain_gradient(argmax)
    top = logits.argmax(-1, keepdim=True)
    covariance = pi.gather(-1, top)[:, 0] * (logits.gather(-1, top)[:, 0] - (pi * logits).sum(-1))
    want = -2 * float(covariance.sum())
    assert want < -1.0
    assert abs(argmax_gradient - want) < 1e-4 * abs(want)
    # Centring cancels a token-blind return exactly, up to fp32 rounding.
    assert abs(blind_gradient) < 1e-5 * abs(want)


def test_a_per_row_norm_would_change_which_token_a_belief_selects():
    # Rows of norm 1 and 4.2: the belief scores the second higher (3.3
    # against 1), a per-row unit norm the first (1 against 0.78).
    rows = torch.tensor([[1.0, 0.0], [3.0, 3.0]])
    belief = torch.tensor([1.0, 0.1])
    per_row = torch.nn.functional.rms_norm(rows, (2,))
    one_scale = rows / rows.square().mean().sqrt()
    assert int((rows @ belief).argmax()) == 1
    assert int((one_scale @ belief).argmax()) == 1
    assert int((per_row @ belief).argmax()) == 0


def test_a_fresh_trunks_zero_readout_is_an_exact_saddle():
    """Why a token Q critic must start from the actor's trunk: a fresh
    trunk zeroes its readout, so x = 0 and xbar = 0, Q has no token
    contrast, the direction's gradient carries x - xbar = 0, and the fixed
    table never moves off zero."""
    torch.manual_seed(0)
    trunk = NanoGPTBackbone(**NANO_KWARGS).float().eval()
    assert not trunk.proj.weight.any()
    critic = SeparateCritic(trunk, q_head_hidden=HIDDEN).eval()
    with torch.no_grad():
        critic.q_head.state.weight.normal_()
    wrapper = LatentThoughtModel(trunk).eval()
    batch = _cot_batch(wrapper)
    offsets = _oracle_action_offsets(wrapper, critic, batch)
    assert not offsets.any()
    values, directions = critic.token_values(batch, offsets)
    assert not critic.action_latents().any()
    assert not directions.any()
    mask = batch.action_mask.bool()
    (values[mask] - 1.0).square().sum().backward()
    assert critic.q_head.state.weight.grad.any()
    for weight in critic.q_head.direction_weights:
        assert weight.grad is None or not weight.grad.any()
    assert critic.trunk.proj.weight.grad is None


def _expected_fit_gradients(table: Tensor, logits: Tensor, returns: Tensor, centred: bool):
    """The Monte Carlo regression's gradient on a head's state value c and
    direction u, in expectation over the policy's action: every action
    enumerated and weighted by pi(a | s), no sampling noise.

    ``table`` [vocab, dim], ``logits`` and ``returns`` (E[G | s, a]) [states,
    vocab]. Returns d/dc [states], d/du [states, dim] and the head's c, u.
    """
    torch.manual_seed(0)
    states, dim = logits.size(0), table.size(1)
    head = TokenQHead(dim, 32, False)
    with torch.no_grad():
        head.state.weight.normal_(std=0.5)
        for weight in head.direction_weights:
            weight.normal_(std=0.5)
    beliefs = torch.randn(states, dim)
    pi = logits.softmax(-1)
    mean_rows = pi @ table if centred else torch.zeros(states, dim)
    c, u = head(beliefs, None)
    c, u = c.detach().requires_grad_(), u.detach().requires_grad_()
    vocab = table.size(0)
    # The production read with actions on the time axis: one state row each.
    q = policy_centered_q(c[None, :, None].expand(1, states, vocab),
                          u[None, :, None].expand(1, states, vocab, dim),
                          table[None] - mean_rows[:, None])[0]
    loss = (pi * (q - returns).square()).sum()
    grad_c, grad_u = torch.autograd.grad(loss, (c, u))
    return grad_c, grad_u, c.detach(), u.detach()


def test_the_centred_fits_direction_gradient_ignores_the_state_values_error():
    """E_pi[d/du (c + u.(x - xbar) - G)^2] = 2 (Cov_pi(x) u - Cov_pi(x, G)):
    c's error, which dominates a fresh critic, never reaches u. Uncentred,
    u also takes 2 xbar (c + u.xbar - V), a push along the policy's mean
    row whose MAC step rewards the tokens that make it up."""
    generator = torch.Generator().manual_seed(3)
    vocab, dim, states = 64, 8, 16
    table = torch.randn(vocab, dim, generator=generator)
    table = table / table.square().mean().sqrt()
    logits = 3.0 * torch.randn(states, vocab, generator=generator)
    pi = logits.softmax(-1).double()
    # Returns that depend on the state alone: there is no token contrast.
    returns = torch.rand(states, 1, generator=generator).expand(states, vocab)
    _, grad_u, c, u = _expected_fit_gradients(table, logits, returns, centred=True)
    x = table.double()
    xbar = pi @ x
    covariance = torch.einsum("sa,sad,sae->sde", pi, x[None] - xbar[:, None], x[None] - xbar[:, None])
    torch.testing.assert_close(grad_u.double(), 2 * torch.einsum("sde,se->sd", covariance, u.double()),
                               rtol=1e-4, atol=1e-5)
    # The same u with c moved by any amount: the direction gradient is unchanged.
    shifted_returns = returns + 5.0
    _, shifted_u, _, _ = _expected_fit_gradients(table, logits, shifted_returns, centred=True)
    torch.testing.assert_close(shifted_u, grad_u, rtol=1e-4, atol=1e-5)
    _, leaking_u, leak_c, leak_u = _expected_fit_gradients(table, logits, shifted_returns, centred=False)
    _, leaking_base, _, _ = _expected_fit_gradients(table, logits, returns, centred=False)
    # Uncentred, the 5.0 shift in V enters u's gradient as -2 * 5.0 * xbar.
    torch.testing.assert_close(leaking_u.double() - leaking_base.double(), -10.0 * xbar, rtol=1e-4, atol=1e-4)


def test_an_uncentred_fit_sharpens_the_policy_where_returns_ignore_the_token():
    """From u = 0 with c below the return, one uncentred step moves u along
    xbar, and all-action MAC (step pi(a) A(a) on the logits) then lowers the
    policy's entropy at every state; the centred step leaves A = 0."""
    generator = torch.Generator().manual_seed(5)
    vocab, dim, states = 256, 16, 32
    table = torch.randn(vocab, dim, generator=generator)
    table = table / table.square().mean().sqrt()
    logits = 4.0 * torch.randn(states, dim, generator=generator) @ table.T / dim ** 0.5
    pi = logits.softmax(-1)
    log_pi = logits.log_softmax(-1)
    for centred in (True, False):
        mean_rows = pi @ table if centred else torch.zeros(states, dim)
        u = torch.zeros(states, dim, requires_grad=True)
        c = torch.zeros(states)
        q = c[:, None] + (u[:, None] * (table[None] - mean_rows[:, None])).sum(-1)
        # A fresh critic's c = 0 under a return of 1 that ignores the token.
        loss = (pi * (q - 1.0).square()).sum()
        (grad,) = torch.autograd.grad(loss, u)
        stepped = -1e-2 * grad
        scores = stepped @ table.T
        advantages = scores - (pi * scores).sum(-1, keepdim=True)
        step = pi * advantages
        # dH/dz_a = -pi(a) (log pi(a) + H).
        entropy = -(pi * log_pi).sum(-1, keepdim=True)
        entropy_change = (-pi * (log_pi + entropy) * step).sum(-1)
        if centred:
            # sum_a pi(a) (x(a) - xbar) = 0, to fp32 rounding of a step
            # whose uncentred entries reach 5e-2 here.
            assert float(stepped.abs().max()) < 1e-7
            assert float(advantages.abs().max()) < 1e-6
        else:
            torch.testing.assert_close(stepped, 2e-2 * (pi @ table))
            assert entropy_change.lt(0).all()


def _ranking_accuracy(table_kind: str, seed: int = 0, dim: int = 16, vocab: int = 256,
                      hidden: int = 64, steps: int = 1500) -> float:
    """Train a fresh head on returns 1 for the readout's argmax token, sampled
    from the readout's own policy, and report how often its highest-Q token
    among the policy's top 8 is that argmax on held-out beliefs."""
    generator = torch.Generator().manual_seed(seed)
    readout = torch.randn(vocab, dim, generator=generator)
    unrelated = torch.randn(vocab, dim, generator=generator)
    beliefs = torch.randn(4096, dim, generator=generator)
    heldout = torch.randn(1024, dim, generator=generator)
    table = {"readout": readout, "unrelated": unrelated}[table_kind]
    table = table / table.square().mean().sqrt()
    torch.manual_seed(seed)
    head = TokenQHead(dim, hidden, False)
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-3)
    logits = 2.0 * beliefs @ readout.T
    correct = logits.argmax(1)
    for _ in range(steps):
        index = torch.randint(beliefs.size(0), (256,), generator=generator)
        actions = torch.multinomial(logits[index].softmax(-1), 1, generator=generator).squeeze(1)
        policy = logits[index].softmax(-1)
        states, directions = head(beliefs[index], None)
        values = states + (directions * (table[actions] - policy @ table)).sum(-1)
        loss = (values - (actions == correct[index]).float()).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        policy = heldout @ readout.T
        top = policy.topk(8, dim=1).indices
        _, directions = head(heldout, None)
        preferred = (directions @ table.T).gather(1, top).argmax(1)
        return float((top.gather(1, preferred[:, None]).squeeze(1) == policy.argmax(1)).float().mean())


def test_the_readout_table_learns_the_contrast_an_unrelated_table_cannot():
    # Chance among the top 8 is 0.125; this toy measured 0.76-0.78 for the
    # readout table and 0.16-0.19 for an unrelated one over seeds 0-2 under
    # the centred fit (0.60-0.65 and 0.21-0.23 uncentred).
    assert _ranking_accuracy("readout") > 0.7
    assert _ranking_accuracy("unrelated") < 0.25
