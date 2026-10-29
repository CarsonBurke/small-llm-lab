"""The token Q critic, its policy-centered actor advantage, and REPPO's
actor stabilizers."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from postraining.core import (
    generalized_advantage_and_return_targets,
    generalized_value_estimate_targets,
)
from postraining.latent_rollout import (
    assign_terminal_rewards,
    carried_hiddens,
    compact_emit_token_logprobs_entropy_and_advantages,
    build_replay_plan,
    compact_next_slots,
    compact_slots,
    policy_action_offsets_at_actions,
    replay_head_inputs,
    replay_policy_action_offsets,
    slot_index,
    compact_emit_token_logprobs,
    compact_emit_token_logprobs_and_entropy,
)
from postraining.nano_backbone import (
    SoftcappedTargetLogprobs,
    SoftcappedTargetLogprobsEntropy,
    SoftcappedTargetLogprobsEntropyAdvantages,
    policy_centered_token_values,
    rational_softcap,
)
from postraining.tests.test_critic_action_head import (
    HIDDEN,
    _changed,
    _scored_batch,
    _snapshot,
)
from postraining.tests.test_hidden_carry import (
    BACKBONES,
    _carry_wrapper,
    _cot_args,
    _prompts,
    _roll,
)
from postraining.tests.test_memory_carry import HEADS, HEAD_DIM, _memory_wrapper
from postraining.train_latent_vapo import (
    MathPromptSampler,
    advantage_group_dashboard,
    aggregate_actor_tensorboard_metrics,
    build_optimizers,
    rewinds_warmup_prompts,
    run_actor_objective_schema,
    save_checkpoint,
    step_token_kl,
    trust_region_token_logprobs,
    update_minibatch,
    validate_resume_arg_contract,
)
from postraining.value_model import (
    SeparateCritic,
    TokenQHead,
    next_slot_inputs,
    next_slot_tokens,
)
from postraining.vapo.config import build_arg_parser, validate_args
from postraining.vapo.reppo import (
    EntropyTemperature,
    backtrack_parameters,
    enforce_kl_bound,
    kl_backtrack_fraction,
    snapshot_parameters,
)

from postraining.vapo.schemas import (
    CRITIC_Q_SCHEMA,
    CRITIC_SCHEMA,
    resume_critic_schema_compatible,
)


def _q_critic(wrapper, *, live: bool = True):
    factory, _ = BACKBONES["nano"]
    critic = SeparateCritic(
        factory(11),
        actor_memory=wrapper.combiner if wrapper.memory_carry else None,
        q_head_hidden=HIDDEN,
        q_head_reads_actor_belief=wrapper.hidden_carry,
    ).eval()
    with torch.no_grad():
        if wrapper.memory_carry:
            critic.combiner.out.weight.normal_(std=0.1)
        if live:
            critic.q_head.state.weight.normal_(std=0.5)
            critic.q_head.direction.weight.normal_(std=0.5)
    return critic


def _token_values(wrapper, critic, batch):
    """The critic's Q of each taken token and its direction, centred under
    the wrapper's policy, as the trainer reads them."""
    return critic.token_values(batch, policy_action_offsets_at_actions(
        wrapper, batch, critic.action_latents(), 512))


def _oracle_action_offsets(wrapper, critic, batch):
    """x(a_t) - sum_a pi(a | s_t) x(a) at every emit slot by brute force,
    in fp64: the policy's full softmax over the renderer's own logits."""
    with torch.no_grad():
        beliefs, inputs = replay_head_inputs(wrapper, batch)
        emit_index = slot_index(batch.emit_mask.bool())
        logits = wrapper.backbone.logits_from_features(wrapper.renderer_features(
            compact_slots(inputs, emit_index), compact_slots(beliefs, emit_index))).double()
        latents = critic.action_latents().double()
        targets = compact_next_slots(batch.token_ids, emit_index)
        dense = torch.zeros((*batch.emit_mask.shape, latents.size(-1)), dtype=torch.float64)
        dense.view(-1, latents.size(-1))[emit_index] = latents[targets] - logits.softmax(-1) @ latents
    return dense.float()


def _expected_advantages(wrapper, critic, batch):
    """A_t by brute force: every token's Q, then its mean under the
    policy's full softmax, at every emit slot."""
    with torch.no_grad():
        beliefs, inputs = replay_head_inputs(wrapper, batch)
        _, directions = _token_values(wrapper, critic, batch)
        latents = critic.action_latents()
        emit_index = slot_index(batch.emit_mask.bool())
        targets = compact_next_slots(batch.token_ids, emit_index)
        logits = wrapper.backbone.logits_from_features(
            wrapper.renderer_features(
                compact_slots(inputs, emit_index),
                compact_slots(beliefs, emit_index),
            )
        ).double()
        q = compact_slots(directions, emit_index).double() @ latents.double().T
        expected = q.gather(-1, targets[:, None]).squeeze(-1) - (
            logits.softmax(-1) * q
        ).sum(-1)
    dense = torch.zeros_like(batch.old_values, dtype=torch.float64)
    dense.view(-1)[emit_index] = expected
    return dense


def _rows():
    rewards = torch.zeros(3, 7)
    mask = torch.zeros(3, 7)
    mask[0, 1:6] = 1
    rewards[0, 5] = 1.0
    mask[1] = 1
    rewards[1, 6] = 0.3
    mask[2, 2:4] = 1
    rewards[2, 3] = -1.0
    return rewards, mask


# --- generalized value estimates -------------------------------------------


def test_gve_matches_the_reppo_recursion():
    rewards, mask = _rows()
    q = torch.randn(3, 7, generator=torch.Generator().manual_seed(0))
    lambdas = torch.tensor([0.9, 0.0, 0.5])
    targets = generalized_value_estimate_targets(rewards, q, mask, lambdas)
    expected = torch.zeros(3, 7)
    for row in range(3):
        following, next_q, next_valid = 0.0, 0.0, 0.0
        for t in reversed(range(7)):
            bootstrap = (1 - lambdas[row]) * next_q + lambdas[row] * following
            expected[row, t] = (rewards[row, t] + next_valid * bootstrap) * mask[row, t]
            following, next_q, next_valid = expected[row, t], q[row, t], mask[row, t]
    torch.testing.assert_close(targets, expected)


def test_gve_endpoints_are_monte_carlo_and_sarsa():
    rewards, mask = _rows()
    q = torch.randn(3, 7, generator=torch.Generator().manual_seed(1))
    _, mc = generalized_advantage_and_return_targets(
        rewards, torch.zeros_like(q), mask, torch.ones(3)
    )
    torch.testing.assert_close(
        generalized_value_estimate_targets(rewards, q, mask, torch.ones(3)), mc
    )
    sarsa = generalized_value_estimate_targets(rewards, q, mask, torch.zeros(3))
    next_q = torch.cat((q[:, 1:], torch.zeros(3, 1)), dim=1)
    next_valid = torch.cat((mask[:, 1:], torch.zeros(3, 1)), dim=1)
    torch.testing.assert_close(sarsa, (rewards + next_q * next_valid) * mask)
    # A row's last action is valued by its reward alone at any lambda.
    assert sarsa[0, 5] == 1.0 and sarsa[1, 6] == pytest.approx(0.3)


def test_gve_targets_carry_no_gradient():
    rewards, mask = _rows()
    q = torch.randn(3, 7, requires_grad=True)
    assert not generalized_value_estimate_targets(
        rewards, q, mask, torch.full((3,), 0.5)
    ).requires_grad


# --- exact entropy readout ---------------------------------------------------


def test_entropy_readout_matches_autograd_and_keeps_the_logprobs():
    generator = torch.Generator().manual_seed(2)
    raw = (torch.randn(6, 41, generator=generator, dtype=torch.float64) * 20)
    raw.requires_grad_()
    targets = torch.randint(0, 41, (6,), generator=generator)
    logprobs, entropy = SoftcappedTargetLogprobsEntropy.apply(raw, targets, 15.0)
    assert torch.equal(logprobs, SoftcappedTargetLogprobs.apply(raw, targets, 15.0))
    reference = rational_softcap(raw, 15.0).log_softmax(-1)
    reference_entropy = -(reference.exp() * reference).sum(-1)
    reference_logprobs = reference.gather(-1, targets[:, None]).squeeze(-1)
    torch.testing.assert_close(entropy, reference_entropy)
    weights = torch.randn(2, 6, generator=generator, dtype=torch.float64)
    gradient, = torch.autograd.grad(
        (logprobs * weights[0] + entropy * weights[1]).sum(), raw
    )
    reference_gradient, = torch.autograd.grad(
        (reference_logprobs * weights[0] + reference_entropy * weights[1]).sum(),
        raw,
    )
    torch.testing.assert_close(gradient, reference_gradient)


def test_emit_entropy_helper_shares_the_logprob_helpers_values():
    wrapper = _memory_wrapper()
    generator = torch.Generator().manual_seed(3)
    dim = wrapper.backbone.model_dim
    inputs = torch.randn(5, dim, generator=generator)
    beliefs = torch.randn(5, dim, generator=generator)
    targets = torch.randint(0, 50, (5,), generator=generator)
    logprobs, entropy = compact_emit_token_logprobs_and_entropy(
        wrapper, inputs, beliefs, targets
    )
    assert torch.equal(
        logprobs, compact_emit_token_logprobs(wrapper, inputs, beliefs, targets)
    )
    assert bool((entropy > 0).all())


# --- the standalone Q critic -------------------------------------------------


def test_q_critic_has_no_v_and_starts_at_zero():
    wrapper = _memory_wrapper()
    critic = _q_critic(wrapper, live=False)
    assert critic.values_actions and critic.head is None
    assert not any(name.startswith("head.") for name in critic.state_dict())
    batch = _scored_batch(wrapper)
    # Q is centred under the policy, which only the actor's readout has.
    with pytest.raises(ValueError, match="centered under the policy"):
        critic.estimates(batch)
    values, directions = _token_values(wrapper, critic, batch)
    assert torch.equal(values, torch.zeros_like(values))
    assert torch.equal(directions, torch.zeros_like(directions))
    with pytest.raises(ValueError, match="only the token Q head"):
        SeparateCritic(
            BACKBONES["nano"][0](11), q_head_reads_actor_belief=True
        )
    with pytest.raises(ValueError, match="no V"):
        SeparateCritic(
            BACKBONES["nano"][0](11), actor_memory=wrapper.combiner,
            action_head_hidden=HIDDEN, q_head_hidden=HIDDEN,
        )


@pytest.mark.parametrize("reads_actor_belief", [False, True])
def test_token_q_direction_and_gradient_have_no_dimension_multiplier(reads_actor_belief):
    torch.manual_seed(1337)
    head = TokenQHead(model_dim=16, hidden=8, reads_actor_belief=reads_actor_belief)
    with torch.no_grad():
        head.direction.weight.normal_(std=0.1)
        head.belief_direction.weight.normal_(std=0.1)
        head.belief_gain.fill_(0.3)
        head.state.weight.normal_(std=0.1)
    beliefs = torch.randn(4, 16)
    actor_beliefs = torch.randn(4, 16) if reads_actor_belief else None
    hidden = head.belief(beliefs)
    if reads_actor_belief:
        hidden = hidden + head.actor_belief(
            torch.nn.functional.rms_norm(actor_beliefs, (16,))
        )
    hidden = hidden.relu().square()
    states, directions = head(beliefs, actor_beliefs)
    torch.testing.assert_close(states, head.state(hidden).squeeze(-1), rtol=0, atol=0)
    torch.testing.assert_close(
        directions, head.direction(hidden) + head.belief_direction(beliefs) + 0.3 * beliefs
    )
    assert bool(directions.abs().sum())
    directions.sum().backward()
    expected_gradient = hidden.detach().sum(0)[None, :].expand_as(
        head.direction.weight
    )
    torch.testing.assert_close(head.direction.weight.grad, expected_gradient)
    torch.testing.assert_close(
        head.belief_direction.weight.grad, beliefs.sum(0)[None, :].expand(16, 16)
    )
    torch.testing.assert_close(head.belief_gain.grad, beliefs.sum())


def test_earlier_q_checkpoints_are_incompatible_with_the_belief_gain_critic():
    inverse_dimension = (
        "standalone_token_bilinear_q_unit_rms_latent_inverse_dim_readout_"
        "actor_belief_gve_mse/v3"
    )
    input_latent = (
        "standalone_token_bilinear_q_unit_rms_latent_unscaled_direction_readout_"
        "actor_belief_gve_mse/v4"
    )
    uncentred = (
        "standalone_token_bilinear_q_table_rms_readout_row_unscaled_direction_readout_"
        "actor_belief_gve_mse/v5"
    )
    gainless = (
        "standalone_token_bilinear_q_policy_centered_fixed_rms_readout_row_"
        "unscaled_direction_readout_actor_belief_gve_mse/v6"
    )
    assert CRITIC_Q_SCHEMA.endswith("/v7")
    assert "policy_centered" in CRITIC_Q_SCHEMA and "belief_gain" in CRITIC_Q_SCHEMA
    for old in (inverse_dimension, input_latent, uncentred, gainless):
        assert not resume_critic_schema_compatible({"critic_schema": old}, True)
    assert resume_critic_schema_compatible({"critic_schema": CRITIC_Q_SCHEMA}, True)


def test_q_reads_the_sampled_action_and_trains_end_to_end():
    wrapper = _memory_wrapper()
    critic = _q_critic(wrapper)
    batch = _scored_batch(wrapper)
    with torch.no_grad():
        values, directions = _token_values(wrapper, critic, batch)
        slot = int(batch.action_mask[0].nonzero()[1])
        token = int(batch.token_ids[0, slot + 1])
        other = (token + 1) % 50
        changed = copy.deepcopy(batch)
        changed.token_ids[0, slot + 1] = other
        changed_values, _ = _token_values(wrapper, critic, changed)
        latents = critic.action_latents()
    assert torch.equal(changed_values[0, :slot], values[0, :slot])
    # Q(s_t, .) is one bilinear readout: another token at t moves Q_t by
    # u_t . (x(other) - x(token)) and nothing else.
    torch.testing.assert_close(
        changed_values[0, slot] - values[0, slot],
        directions[0, slot] @ (latents[other] - latents[token]),
    )
    # The actor's belief that sampled a_t is state at t.
    with torch.no_grad():
        moved = copy.deepcopy(batch)
        moved.hiddens[0, slot + 1] += 1.0
        moved_values, _ = _token_values(wrapper, critic, moved)
    assert torch.equal(moved_values[0, :slot], values[0, :slot])
    assert not torch.equal(moved_values[0, slot], values[0, slot])
    (_token_values(wrapper, critic, batch)[0] * batch.action_mask).square().sum().backward()
    trained = {
        name.split(".")[0]
        for name, parameter in critic.named_parameters()
        if parameter.grad is not None and bool(parameter.grad.abs().sum())
    }
    # The loss reaches the trunk and the memory reader; only the action
    # table, the trunk's readout, is fixed.
    assert {"q_head", "trunk", "combiner"} <= trained
    assert bool(critic.trunk.tok_emb.weight.grad.abs().sum())
    assert critic.trunk.proj.weight.grad is None


def test_policy_centered_values_are_exact_and_zero_mean_under_the_policy():
    generator = torch.Generator().manual_seed(0)
    rows, vocab, dim = 5, 37, 8
    logits = 3.0 * torch.randn(rows, vocab, generator=generator)
    directions = torch.randn(rows, dim, generator=generator)
    latents = torch.randn(vocab, dim, generator=generator)
    targets = torch.randint(vocab, (rows,), generator=generator)
    advantages = policy_centered_token_values(
        logits, targets, directions, latents
    )
    q = (directions.double() @ latents.double().T) + 7.0
    expected = q.gather(-1, targets[:, None]).squeeze(-1) - (
        logits.double().softmax(-1) * q
    ).sum(-1)
    torch.testing.assert_close(
        advantages.double(), expected, rtol=1e-5, atol=1e-5
    )
    every = torch.stack(
        [
            policy_centered_token_values(
                logits, torch.full((rows,), token), directions, latents
            )
            for token in range(vocab)
        ],
        dim=-1,
    )
    torch.testing.assert_close(
        (logits.softmax(-1) * every).sum(-1), torch.zeros(rows),
        atol=1e-5, rtol=0,
    )
    assert not policy_centered_token_values(
        logits.requires_grad_(), targets, directions.requires_grad_(), latents
    ).requires_grad


def test_optimizers_route_the_q_readout_to_the_action_rate():
    wrapper = _memory_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(
        wrapper, critic, learning_rate=1e-3, fused=False,
        critic_action_learning_rate=5e-3,
    )
    q_head = {id(p) for p in critic.q_head.parameters()}
    assert {
        id(p) for g in optimizers["critic_action"].param_groups for p in g["params"]
    } == q_head
    assert optimizers["critic_action"].param_groups[0]["lr"] == 5e-3


def test_q_warmup_fits_the_monte_carlo_return_through_the_whole_critic():
    wrapper = _memory_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=1e-3, fused=False)
    critic_before, actor_before = _snapshot(critic), _snapshot(wrapper)
    metrics = update_minibatch(
        wrapper, critic, _scored_batch(wrapper), optimizers, value_only=True
    )
    changed = _changed(critic_before, critic)
    assert any(name.startswith("trunk.") for name in changed)
    assert "q_head.direction.weight" in changed
    assert not _changed(actor_before, wrapper)
    # At lambda one the generalized value estimates are the return itself.
    assert metrics["explained_variance"] == pytest.approx(
        metrics["mc_explained_variance"]
    )


def test_q_actor_advantage_is_policy_centered_q_and_targets_bootstrap():
    wrapper = _memory_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    batch = _scored_batch(wrapper)
    batch.source_id = torch.zeros(batch.kind.size(0), dtype=torch.long)
    with torch.no_grad():
        q, _ = _token_values(wrapper, critic, batch)
    expected = _expected_advantages(wrapper, critic, batch)
    # A sentinel the update must overwrite at every action slot.
    batch.old_values.fill_(123.0)
    lambdas = torch.full((batch.kind.size(0),), 0.5)
    expected_targets = generalized_value_estimate_targets(
        batch.rewards, q, batch.action_mask, lambdas
    )
    metrics = update_minibatch(
        wrapper, critic, batch, optimizers,
        delightful_policy_gradient=True, fused_behavior_statistics=True,
        critic_lambda=0.5, actor_step=False, critic_step=False,
        advantage_source_count=1,
    )
    action = batch.action_mask.bool()
    torch.testing.assert_close(batch.old_values[action], q[action])
    count = batch.action_mask.sum()
    assert bool(expected.abs().sum())
    assert metrics["advantage_mean"] == pytest.approx(
        float((expected * batch.action_mask).sum() / count), rel=1e-4
    )
    assert metrics["advantage_group/source_0/square_sum"] == pytest.approx(
        float((expected.square() * batch.emit_mask).sum()), rel=1e-4
    )
    assert metrics["value_target_mean"] == pytest.approx(
        float((expected_targets * batch.action_mask).sum() / count), rel=1e-5
    )
    dashboard = aggregate_actor_tensorboard_metrics([metrics])
    assert "value/mc_explained_variance" in dashboard
    assert "credit/td0_calibration_slope" in dashboard


def test_token_q_readout_compiles_like_the_trainer_and_matches_eager():
    """The trainer compiles this readout fullgraph with dynamic shapes; a
    softcap computed outside its autograd function once broke the trace."""
    wrapper = _memory_wrapper()
    critic = _q_critic(wrapper)
    batch = _scored_batch(wrapper)
    with torch.no_grad():
        _, directions = _token_values(wrapper, critic, batch)
        latents = critic.action_latents()
    beliefs, inputs = replay_head_inputs(wrapper, batch)
    emit_index = slot_index(batch.emit_mask.bool())
    arguments = (
        wrapper,
        compact_slots(inputs, emit_index),
        compact_slots(beliefs, emit_index),
        compact_next_slots(batch.token_ids, emit_index),
        compact_slots(directions, emit_index),
        latents,
    )
    torch._dynamo.reset()
    compiled = torch.compile(
        compact_emit_token_logprobs_entropy_and_advantages,
        fullgraph=True, dynamic=True, backend="aot_eager",
    )
    logprobs, entropy, advantages, offsets = compiled(*arguments)
    eager = compact_emit_token_logprobs_entropy_and_advantages(*arguments)
    torch.testing.assert_close(logprobs, eager[0], rtol=0, atol=0)
    torch.testing.assert_close(advantages, eager[2], rtol=0, atol=0)
    torch.testing.assert_close(offsets, eager[3], rtol=0, atol=0)
    # The readout's offsets are the replay helper's, which centre a
    # value-only fit, and the fp64 brute force's.
    torch.testing.assert_close(offsets, replay_policy_action_offsets(
        wrapper, batch, latents, 512))
    torch.testing.assert_close(offsets, _oracle_action_offsets(wrapper, critic, batch)[
        batch.emit_mask.bool()], rtol=1e-4, atol=1e-5)
    # The step KL's zero stand-ins leave the log-probabilities unchanged.
    placeholders = compiled(
        *arguments[:4], torch.zeros_like(arguments[4]), torch.zeros_like(latents)
    )
    assert torch.equal(placeholders[0], logprobs)
    assert not advantages.requires_grad
    # Its gradient is the entropy readout's.
    raw = torch.randn(6, 40, requires_grad=True)
    targets = torch.randint(40, (6,))
    weights = torch.randn(2, 6)
    gradients = []
    for function, extra in (
        (SoftcappedTargetLogprobsEntropy, ()),
        (
            SoftcappedTargetLogprobsEntropyAdvantages,
            (torch.randn(6, 3), torch.randn(40, 3)),
        ),
    ):
        outputs = function.apply(raw, targets, 15.0, *extra)
        (weights[0] @ outputs[0] + weights[1] @ outputs[1]).backward()
        gradients.append(raw.grad.clone())
        raw.grad = None
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)


def test_fresh_token_q_leaves_its_zero_init_and_then_moves_the_actor():
    wrapper = _memory_wrapper()
    critic = _q_critic(wrapper, live=False)
    optimizers = build_optimizers(wrapper, critic, learning_rate=1e-2, fused=False)
    batch = _scored_batch(wrapper)
    # Step 0: Q is zero for every token, so the advantage is exactly zero.
    metrics = update_minibatch(
        wrapper, critic, copy.deepcopy(batch), optimizers,
        delightful_policy_gradient=True, fused_behavior_statistics=True,
        actor_step=False, critic_step=False,
    )
    assert metrics["advantage_mean"] == 0.0 and metrics["advantage_std"] == 0.0
    # Only the readouts have a gradient at the zero point; neither factor
    # of u . x is stuck at a saddle.
    zero_grads = {
        name for name, parameter in critic.named_parameters()
        if parameter.grad is not None and bool(parameter.grad.abs().sum())
    }
    assert {"q_head.direction.weight", "q_head.state.bias"} <= zero_grads
    wrapper.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    update_minibatch(
        wrapper, critic, copy.deepcopy(batch), optimizers, value_only=True
    )
    critic.zero_grad(set_to_none=True)
    (_token_values(wrapper, critic, batch)[0] * batch.action_mask).square().sum().backward()
    assert bool(critic.trunk.tok_emb.weight.grad.abs().sum())
    assert any(
        bool(parameter.grad.abs().sum())
        for name, parameter in critic.named_parameters()
        if name.startswith("trunk.blocks") and parameter.grad is not None
    )
    critic.zero_grad(set_to_none=True)
    before = _snapshot(wrapper)
    metrics = update_minibatch(
        wrapper, critic, copy.deepcopy(batch), optimizers,
        delightful_policy_gradient=True, fused_behavior_statistics=True,
    )
    assert metrics["advantage_std"] > 0.0
    assert _changed(before, wrapper)


def test_advantages_break_down_by_source_and_taken_token_class():
    wrapper = _memory_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    batch = _scored_batch(wrapper)
    batch.source_id = torch.tensor([0, 1, 0, 1])
    emit = batch.emit_mask.bool()
    taken = next_slot_tokens(batch.token_ids)[emit]
    previous = batch.token_ids[emit]
    classes = {
        "first": (taken[:1].clone(), None),
        # The second taken token only where the first precedes it.
        "second": (taken[1:2].clone(), previous[1:2].clone()),
    }
    metrics = update_minibatch(
        wrapper, critic, batch, optimizers,
        delightful_policy_gradient=True, fused_behavior_statistics=True,
        actor_step=False, critic_step=False,
        advantage_source_count=2, advantage_token_classes=classes,
    )
    emitted = float(batch.emit_mask.sum())
    total = metrics["advantage_mean"] * float(batch.action_mask.sum())
    assert taken[0] != taken[1]  # disjoint classes
    expected_second = float(
        ((taken == taken[1]) & (previous == previous[1])).sum()
    )
    assert metrics["advantage_group/second/count"] == expected_second >= 1
    for names in (("source_0", "source_1"), ("first", "second", "other")):
        assert sum(
            metrics[f"advantage_group/{name}/count"] for name in names
        ) == emitted
        assert sum(
            metrics[f"advantage_group/{name}/sum"] for name in names
        ) == pytest.approx(total, rel=1e-5, abs=1e-7)
    dashboard = advantage_group_dashboard([metrics, metrics], ["math", "mc"])
    assert dashboard["advantage/source/mc/tokens"] == 2 * metrics[
        "advantage_group/source_1/count"
    ]
    assert dashboard["advantage/token/other/mean"] == pytest.approx(
        metrics["advantage_group/other/sum"]
        / metrics["advantage_group/other/count"]
    )
    batch.source_id = None
    with pytest.raises(ValueError, match="source ids"):
        update_minibatch(
            wrapper, critic, batch, optimizers,
            delightful_policy_gradient=True, fused_behavior_statistics=True,
            actor_step=False, critic_step=False, advantage_source_count=2,
        )


def test_q_actor_refuses_the_unfused_and_non_delightful_paths():
    wrapper = _memory_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=1e-3, fused=False)
    with pytest.raises(ValueError, match="fused behavior statistics"):
        update_minibatch(
            wrapper, critic, _scored_batch(wrapper), optimizers,
            delightful_policy_gradient=True,
        )
    with pytest.raises(ValueError, match="Delightful PG only"):
        update_minibatch(
            wrapper, critic, _scored_batch(wrapper), optimizers,
            fused_behavior_statistics=True,
        )


def test_q_targets_and_advantages_do_not_depend_on_sharding():
    wrapper = _memory_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    batch = _scored_batch(wrapper)
    metrics = {}
    for shard_rows in (32, 1):
        copied = copy.deepcopy(batch)
        metrics[shard_rows] = update_minibatch(
            wrapper, critic, copied, optimizers,
            delightful_policy_gradient=True, fused_behavior_statistics=True,
            critic_lambda=0.5, replay_max_trajectories=shard_rows,
            actor_step=False, critic_step=False,
        )
        wrapper.zero_grad(set_to_none=True)
        critic.zero_grad(set_to_none=True)
    for name in (
        "advantage_mean", "value_target_mean", "value_loss",
        "mc_target_mean", "mc_explained_variance",
    ):
        assert metrics[1][name] == pytest.approx(
            metrics[32][name], rel=1e-4, abs=1e-7
        ), name


def test_q_critic_drives_the_carry_mode():
    wrapper = _carry_wrapper()
    critic = _q_critic(wrapper)
    assert critic.q_head.actor_belief is not None
    optimizers = build_optimizers(wrapper, critic, learning_rate=1e-3, fused=False)
    batch = _roll(wrapper, _prompts("nano", 4, 5), seed=5)
    assign_terminal_rewards(batch, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    with torch.no_grad():
        q, _ = _token_values(wrapper, critic, batch)
    batch.old_values.fill_(123.0)
    before = _snapshot(wrapper)
    update_minibatch(
        wrapper, critic, batch, optimizers,
        delightful_policy_gradient=True, fused_behavior_statistics=True,
    )
    action = batch.action_mask.bool()
    torch.testing.assert_close(batch.old_values[action], q[action])
    assert _changed(before, wrapper)


def test_entropy_term_moves_the_actor_and_reports_the_entropy():
    wrapper = _memory_wrapper()
    critic = _q_critic(wrapper, live=False)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    batch = _scored_batch(wrapper)
    # A fresh Q is zero, so the Delightful term has no gradient: whatever the
    # actor gets comes from the entropy term alone.
    update_minibatch(
        wrapper, critic, copy.deepcopy(batch), optimizers,
        delightful_policy_gradient=True, fused_behavior_statistics=True,
        actor_step=False, critic_step=False,
    )
    assert not any(
        bool(p.grad.abs().sum()) for p in wrapper.parameters() if p.grad is not None
    )
    wrapper.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    metrics = update_minibatch(
        wrapper, critic, batch, optimizers,
        delightful_policy_gradient=True, fused_behavior_statistics=True,
        entropy_temperature=0.1, actor_step=False, critic_step=False,
    )
    assert any(
        bool(p.grad.abs().sum()) for p in wrapper.parameters() if p.grad is not None
    )
    dashboard = aggregate_actor_tensorboard_metrics([metrics])
    entropy = dashboard["entropy/policy_mean"]
    vocabulary = wrapper.backbone.tok_emb.num_embeddings
    assert 0.0 < entropy <= math.log(vocabulary) + 1e-4
    assert dashboard["loss/entropy"] == pytest.approx(
        -0.1 * metrics["policy_entropy_sum"] / float(batch.emit_mask.sum()),
        rel=1e-5,
    )
    assert dashboard["loss/actor_total"] == pytest.approx(
        dashboard["loss/policy"] + dashboard["loss/entropy"]
    )
    with pytest.raises(ValueError, match="Delightful PG term"):
        update_minibatch(
            wrapper, _q_critic(wrapper), batch, optimizers,
            value_only=True, entropy_temperature=0.1,
        )


def test_q_critic_reports_the_entropy_without_an_entropy_term():
    wrapper = _memory_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    batch = _scored_batch(wrapper)
    dashboards = [
        aggregate_actor_tensorboard_metrics([
            update_minibatch(
                wrapper, critic, copy.deepcopy(batch), optimizers,
                delightful_policy_gradient=True, fused_behavior_statistics=True,
                entropy_temperature=temperature, actor_step=False,
                critic_step=False,
            )
        ])
        for temperature in (None, 0.1)
    ]
    bare, termed = dashboards
    assert "loss/entropy" not in bare
    assert bare["entropy/policy_mean"] == pytest.approx(
        termed["entropy/policy_mean"], rel=1e-6
    )
    assert bare["loss/actor_total"] == pytest.approx(bare["loss/policy"])


def test_entropy_mean_is_omitted_without_an_active_emit_slot():
    metrics = {
        "policy_entropy_sum": 0.0, "policy_entropy_count": 0.0,
    }
    wrapper = _memory_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    full = update_minibatch(
        wrapper, critic, _scored_batch(wrapper), optimizers,
        delightful_policy_gradient=True, fused_behavior_statistics=True,
        entropy_temperature=0.1, actor_step=False, critic_step=False,
    )
    full.update(metrics)
    dashboard = aggregate_actor_tensorboard_metrics([full])
    assert "entropy/policy_mean" not in dashboard


# --- REPPO's stabilizers -----------------------------------------------------


def test_entropy_temperature_rises_below_target_and_decays_above():
    controller = EntropyTemperature(target=2.0, initial=0.01, learning_rate=0.1)
    for _ in range(5):
        controller.update(1.0)
    assert controller.temperature > 0.01 * math.exp(0.4)
    risen = controller.temperature
    for _ in range(20):
        controller.update(3.0)
    assert controller.temperature < risen
    state = controller.state_dict()
    restored = EntropyTemperature(target=2.0, initial=0.5, learning_rate=0.1)
    restored.load_state_dict(state)
    assert restored.temperature == controller.temperature
    controller.update(1.0)
    restored.update(1.0)
    assert restored.temperature == controller.temperature
    with pytest.raises(ValueError, match="targets"):
        EntropyTemperature(target=1.0, initial=0.5, learning_rate=0.1).load_state_dict(
            state
        )


def test_backtracking_keeps_the_fraction_of_the_step():
    parameters = [torch.zeros(3), torch.ones(2)]
    before = snapshot_parameters(parameters)
    parameters[0].add_(4.0)
    parameters[1].sub_(2.0)
    backtrack_parameters(parameters, before, 0.25)
    torch.testing.assert_close(parameters[0], torch.ones(3))
    torch.testing.assert_close(parameters[1], torch.full((2,), 0.5))
    assert kl_backtrack_fraction(4e-4, 1e-4) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        kl_backtrack_fraction(1e-5, 1e-4)


def test_step_kl_is_an_fp32_self_reference_quadratic_in_the_step():
    wrapper = _memory_wrapper()
    batch = _scored_batch(wrapper)
    plan = build_replay_plan(
        batch, 1, 4 * 1024 * 1024, 1, include_action_indices=True
    )

    def readout(stride=1):
        return trust_region_token_logprobs(
            wrapper, batch, plan, shard_stride=stride
        )

    reference = readout()
    unmoved = step_token_kl(reference, readout())
    assert unmoved["kl/post_update_token_pre_step"] == 0.0
    assert unmoved["ratio/post_update_token_pre_step_abs_log_max"] == 0.0
    assert unmoved["trust/measured_tokens"] == float(batch.emit_mask.sum())
    # The readout is fp32 whatever autocast surrounds it.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(readout(), reference)
    parameters = [p for p in wrapper.parameters() if p.requires_grad]
    before = snapshot_parameters(parameters)
    with torch.no_grad():
        for parameter in parameters:
            parameter.add_(1e-3 * torch.randn_like(parameter))
    full = step_token_kl(reference, readout())["kl/post_update_token_pre_step"]
    backtrack_parameters(parameters, before, 0.5)
    half = step_token_kl(reference, readout())["kl/post_update_token_pre_step"]
    assert full > 0.0
    assert half / full == pytest.approx(0.25, rel=0.05)
    subset = readout(2)
    assert 0 < subset.numel() < reference.numel()
    with pytest.raises(ValueError, match="same tokens"):
        step_token_kl(reference, subset)
    # Reading never writes gradients.
    assert all(p.grad is None for p in wrapper.parameters())


def _quadratic_drift(parameters, before, curvature=1.0, exponent=2):
    def measure():
        step = sum(
            float((now - then).square().sum())
            for now, then in zip(parameters, before)
        )
        return {"kl": curvature * step ** (exponent / 2)}
    return measure


def test_trust_region_lands_a_quadratic_kl_on_the_bound():
    parameters = [torch.zeros(4)]
    before = snapshot_parameters(parameters)
    parameters[0].add_(1.0)
    measure = _quadratic_drift(parameters, before)
    drift, trust = enforce_kl_bound(
        parameters, before, measure(), measure,
        bound=1.0, max_backtracks=3, kl_key="kl",
    )
    assert trust["trust/unbounded_kl"] == pytest.approx(4.0)
    assert trust["trust/step_fraction"] == pytest.approx(0.5)
    assert trust["trust/backtracks"] == 1.0
    assert drift["kl"] == pytest.approx(1.0)
    torch.testing.assert_close(parameters[0], torch.full((4,), 0.5))


def test_trust_region_leaves_a_step_inside_the_bound_and_caps_passes():
    parameters = [torch.zeros(4)]
    before = snapshot_parameters(parameters)
    parameters[0].add_(0.1)
    measure = _quadratic_drift(parameters, before)
    drift, trust = enforce_kl_bound(
        parameters, before, measure(), measure,
        bound=1.0, max_backtracks=3, kl_key="kl",
    )
    assert trust["trust/backtracks"] == 0.0
    assert trust["trust/step_fraction"] == 1.0
    torch.testing.assert_close(parameters[0], torch.full((4,), 0.1))
    # A KL flatter than quadratic stays above the bound after each pass;
    # the cap holds.
    parameters[0].fill_(1.0)
    measure = _quadratic_drift(parameters, before, curvature=4.0, exponent=1)
    drift, trust = enforce_kl_bound(
        parameters, before, measure(), measure,
        bound=1.0, max_backtracks=2, kl_key="kl",
    )
    assert trust["trust/backtracks"] == 2.0
    assert drift["kl"] > 1.0
    assert trust["trust/step_fraction"] == pytest.approx(
        float(parameters[0][0])
    )


# --- contracts ---------------------------------------------------------------


def _parse(*extra):
    parser = build_arg_parser()
    args = parser.parse_args(["--checkpoint", "c", "--output", "o", *extra])
    validate_args(parser, args)
    return args


Q_ARGS = (
    "--critic-q-head-hidden", "8", "--critic-init", "actor", "--delightful-policy-gradient",
    "--reasoning-mode", "memory",
    "--prompts-per-rollout", "64", "--prompts-per-minibatch", "64",
)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--critic-q-head-hidden", "8"), "pass --delightful-policy-gradient"),
        (
            (*Q_ARGS, "--critic-action-head-hidden", "8"),
            "no V for --critic-action-head-hidden",
        ),
        ((*Q_ARGS, "--actor-gae-lambda", "0"), "pass --q-critic-lambda"),
        ((*Q_ARGS, "--q-critic-lambda", "1.5"), "must lie in [0, 1]"),
        (("--q-critic-lambda", "0.5"), "needs a standalone or post-action Q critic"),
        ((*Q_ARGS, "--actor-init", "a.pt"), "add --fresh-critic"),
        # The action table is the trunk's readout, zero in a fresh trunk.
        (tuple(arg for arg in Q_ARGS if arg not in ("--critic-init", "actor")),
         "requires --critic-init actor"),
        (
            ("--entropy-target", "1.0", "--entropy-temperature-init", "1e-6"),
            "term of the Delightful objective",
        ),
        (("--entropy-target", "1.0"), "relative to the scale of Q"),
        (("--actor-kl-bound", "0"), "finite and positive"),
        (
            ("--actor-kl-bound", "1e-4", "--reasoning-mode", "latent",
             "--prompts-per-rollout", "64", "--prompts-per-minibatch", "64"),
            "token KL only",
        ),
        (("--actor-kl-backtracks", "2"), "needs --actor-kl-bound"),
        (("--actor-kl-shard-stride", "2"), "needs --actor-kl-bound"),
        (
            ("--actor-kl-bound", "1e-4", "--actor-kl-shard-stride", "0",
             "--prompts-per-rollout", "64", "--prompts-per-minibatch", "64"),
            "--actor-kl-shard-stride must be positive",
        ),
        (("--entropy-temperature-init", "0.1"), "needs --entropy-target"),
        (
            ("--entropy-temperature-learning-rate", "0.1"),
            "needs --entropy-target",
        ),
        (("--value-warmup-rewind",), "returns to an --actor-init"),
        (
            ("--value-warmup-rewind", "--actor-init", "a.pt", "--value-warmup-steps", "0"),
            "whole-critic warmup",
        ),
    ],
)
def test_q_critic_cli_refusals(extra, message, capsys):
    with pytest.raises(SystemExit):
        _parse(*extra)
    assert message in capsys.readouterr().err


def test_trust_region_takes_any_minibatching_and_objective():
    # Each step reads its own fp32 reference, so neither the pool's
    # minibatch split nor the objective's recorded statistics matter.
    split = _parse(
        "--actor-kl-bound", "1e-4", "--reasoning-mode", "memory",
        "--prompts-per-rollout", "64", "--prompts-per-minibatch", "32",
    )
    assert split.actor_kl_shard_stride == 4
    _parse(
        "--actor-kl-bound", "1e-4", "--target-policy-optimization",
        "--reasoning-mode", "memory",
    )


def test_warmup_rewind_flag_rewinds_a_whole_critic_warmup():
    args = _parse(
        *Q_ARGS, "--actor-init", "a.pt", "--fresh-critic",
        "--value-warmup-steps", "50", "--value-warmup-rewind",
    )
    assert rewinds_warmup_prompts(args)
    args.value_warmup_rewind = False
    assert not rewinds_warmup_prompts(args)
    # Its resume drops --actor-init and keeps the flag the manifest binds:
    # the checkpoint carries the cursor to rewind to.
    resumed = _parse(
        *Q_ARGS, "--resume", "o/latent_vapo_checkpoint.pt",
        "--value-warmup-steps", "50", "--value-warmup-rewind",
    )
    assert rewinds_warmup_prompts(resumed)


def test_schemas_name_every_component():
    plain = _parse("--delightful-policy-gradient", "--reasoning-mode", "memory")
    reppo = _parse(
        *Q_ARGS, "--entropy-target", "1.5", "--entropy-temperature-init", "1e-6",
        "--actor-kl-bound", "3e-4"
    )
    plain_schema = run_actor_objective_schema(plain)
    reppo_schema = run_actor_objective_schema(reppo)
    assert reppo_schema.startswith(plain_schema)
    for component in (
        "q_minus_exact_policy_mean", "entropy", "fp32_shard_subset"
    ):
        assert component in reppo_schema and component not in plain_schema
    assert resume_critic_schema_compatible({"critic_schema": CRITIC_Q_SCHEMA}, True)
    assert not resume_critic_schema_compatible({"critic_schema": CRITIC_SCHEMA}, True)
    assert not resume_critic_schema_compatible({"critic_schema": CRITIC_Q_SCHEMA})


def test_resume_binds_the_new_contracts():
    reppo = _parse(*Q_ARGS, "--entropy-target", "1.5", "--entropy-temperature-init", "1e-6",
        "--actor-kl-bound", "3e-4")
    saved = vars(copy.deepcopy(reppo))
    validate_resume_arg_contract(saved, reppo)
    for name, value in (
        ("q_critic_lambda", 0.5),
        ("entropy_target", 2.0),
        ("actor_kl_bound", 1e-3),
        ("actor_kl_shard_stride", 1),
        ("value_warmup_rewind", True),
    ):
        changed = copy.deepcopy(reppo)
        setattr(changed, name, value)
        with pytest.raises(ValueError, match=name):
            validate_resume_arg_contract(saved, changed)


def test_checkpoint_carries_the_q_schema_and_temperature(tmp_path):
    wrapper = _memory_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=1e-3, fused=False)
    args = _cot_args(
        reasoning_mode="memory", thought_sigma=None,
        init_stop_thinking_probability=None, memory_heads=HEADS,
        memory_head_dim=HEAD_DIM, trunk_optimizer="adamw",
        learning_rate=1e-3, critic_learning_rate=1e-3,
        muon_learning_rate=None, critic_muon_learning_rate=None,
        delightful_policy_gradient=True, critic_q_head_hidden=HIDDEN,
        entropy_target=1.5,
    )
    controller = EntropyTemperature(1.5, 0.01, 0.05)
    controller.update(0.5)
    path = tmp_path / "q.pt"
    save_checkpoint(
        path, wrapper, critic, optimizers, step=0, args=args,
        sampler=MathPromptSampler([{"id": index} for index in range(4)], seed=3),
        warmup_step=1, entropy_temperature=controller,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["critic_schema"] == CRITIC_Q_SCHEMA
    assert "entropy" in payload["actor_objective_schema"]
    restored = EntropyTemperature(1.5, 0.01, 0.05)
    restored.load_state_dict(payload["entropy_temperature_state"])
    assert restored.temperature == controller.temperature
