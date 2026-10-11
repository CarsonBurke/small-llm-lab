"""The critic's dueling action head and the Q-bootstrapped actor advantages."""

from __future__ import annotations

import copy

import pytest
import torch

from postraining.latent_rollout import (
    assign_terminal_rewards,
    pack_rollout_groups_for_replay,
    refresh_old_statistics,
)
from postraining.tests.test_hidden_carry import BACKBONES, _cot_args, _prompts, _roll
from postraining.tests.test_memory_carry import HEADS, HEAD_DIM, _memory_wrapper
from postraining.train_latent_vapo import (
    MathPromptSampler,
    aggregate_actor_tensorboard_metrics,
    build_optimizers,
    carry_init_critic,
    plan_one_pass_training,
    rewinds_warmup_prompts,
    save_checkpoint,
    update_minibatch,
)
from postraining.value_model import SeparateCritic
from postraining.vapo.config import build_arg_parser, validate_args
from postraining.vapo.schemas import CRITIC_ACTION_HEAD_SCHEMA

HIDDEN = 16


def _critic(wrapper, *, action_head: bool = True, live_head: bool = False):
    factory, _ = BACKBONES["nano"]
    critic = SeparateCritic(
        factory(11),
        actor_memory=wrapper.combiner,
        action_head_hidden=HIDDEN if action_head else None,
    ).eval()
    with torch.no_grad():
        critic.head.weight.normal_(std=0.05)
        critic.combiner.out.weight.normal_(std=0.1)
        if live_head:
            critic.action_head.out.weight.normal_(std=0.5)
    return critic


def _scored_batch(wrapper, seed: int = 5):
    batch = _roll(wrapper, _prompts("nano", 4, 5, seed=seed), seed=seed)
    assign_terminal_rewards(batch, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    return batch


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in module.state_dict().items()}


def _changed(before: dict[str, torch.Tensor], module: torch.nn.Module) -> set[str]:
    return {
        name
        for name, value in module.state_dict().items()
        if not torch.equal(value, before[name])
    }


def test_fresh_head_is_exactly_v_and_q_reads_the_sampled_action():
    wrapper = _memory_wrapper()
    critic = _critic(wrapper)
    batch = _scored_batch(wrapper)
    with torch.no_grad():
        values, action_values = critic.estimates(batch)
        assert torch.equal(action_values, values)
        assert torch.equal(critic.values(batch), values)
        critic.action_head.out.weight.normal_(std=0.5)
        values, action_values = critic.estimates(batch)
        # Change the token one action slot emitted: Q at that slot moves, V
        # there and everything before it do not.
        slot = int(batch.action_mask[0].nonzero()[1])
        changed = copy.deepcopy(batch)
        changed.token_ids[0, slot + 1] = (changed.token_ids[0, slot + 1] + 1) % 50
        changed_values, changed_action_values = critic.estimates(changed)
    assert torch.equal(changed_values[0, : slot + 1], values[0, : slot + 1])
    assert torch.equal(
        changed_action_values[0, :slot], action_values[0, :slot]
    )
    assert not torch.equal(
        changed_action_values[0, slot], action_values[0, slot]
    )


def test_q_loss_trains_only_the_head():
    wrapper = _memory_wrapper()
    critic = _critic(wrapper)
    batch = _scored_batch(wrapper)
    values, action_values = critic.estimates(batch)
    (action_values * batch.action_mask).square().sum().backward()
    # The output layer is zero, so only it has a first-step gradient; V's
    # trunk, head and reader get none through the detached inputs.
    trained = {
        name
        for name, parameter in critic.named_parameters()
        if parameter.grad is not None and bool(parameter.grad.abs().sum())
    }
    assert trained == {"action_head.out.weight", "action_head.out.bias"}


def test_optimizers_give_the_head_its_own_adamw():
    wrapper = _memory_wrapper()
    critic = _critic(wrapper)
    optimizers = build_optimizers(
        wrapper, critic, learning_rate=1e-3, fused=False,
        critic_action_learning_rate=5e-3,
    )
    head = {id(p) for p in critic.action_head.parameters()}
    assert {id(p) for g in optimizers["critic_action"].param_groups for p in g["params"]} == head
    assert not head & {
        id(p) for g in optimizers["critic"].param_groups for p in g["params"]
    }
    assert optimizers["critic_action"].param_groups[0]["lr"] == 5e-3
    assert "critic_action" not in build_optimizers(
        wrapper, _critic(wrapper, action_head=False), learning_rate=1e-3, fused=False
    )


def test_action_head_warmup_moves_only_the_head():
    wrapper = _memory_wrapper()
    critic = _critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=1e-3, fused=False)
    critic_before, actor_before = _snapshot(critic), _snapshot(wrapper)
    metrics = update_minibatch(
        wrapper, critic, _scored_batch(wrapper), optimizers,
        value_only=True, value_scope="action_head",
    )
    assert _changed(critic_before, critic) == {
        "action_head.out.weight", "action_head.out.bias",
    }
    assert not _changed(actor_before, wrapper)
    assert metrics["action_value_loss"] == pytest.approx(metrics["value_loss"])
    # The ordinary warmup fits both.
    critic_before = _snapshot(critic)
    update_minibatch(
        wrapper, critic, _scored_batch(wrapper, seed=6), optimizers, value_only=True
    )
    changed = _changed(critic_before, critic)
    assert "head.weight" in changed and "action_head.belief.weight" in changed
    with pytest.raises(ValueError, match="action-head-only"):
        update_minibatch(
            wrapper, critic, _scored_batch(wrapper), optimizers,
            value_scope="action_head",
        )


def test_lambda_zero_actor_follows_q_minus_v():
    """At actor lambda 0 the advantage is the head's residual: zero for a
    fresh head, so the policy does not move; nonzero once it has learned."""
    wrapper = _memory_wrapper()
    critic = _critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=1e-3, fused=False)
    batch = _scored_batch(wrapper)
    refresh_old_statistics(wrapper, critic, batch)
    assert torch.equal(batch.old_action_values, batch.old_values)
    actor_before = _snapshot(wrapper)
    metrics = update_minibatch(
        wrapper, critic, batch, optimizers,
        delightful_policy_gradient=True, actor_gae_lambda=0.0,
    )
    assert metrics["advantage_std"] == 0.0 and metrics["advantage_mean"] == 0.0
    assert not _changed(actor_before, wrapper)

    with torch.no_grad():
        critic.action_head.out.weight.normal_(std=0.5)
    batch = _scored_batch(wrapper, seed=8)
    refresh_old_statistics(wrapper, critic, batch)
    residual = (batch.old_action_values - batch.old_values) * batch.action_mask
    assert bool(residual.abs().sum())
    metrics = update_minibatch(
        wrapper, critic, batch, optimizers,
        delightful_policy_gradient=True, actor_gae_lambda=0.0,
    )
    assert _changed(actor_before, wrapper)
    count = batch.action_mask.sum()
    assert metrics["advantage_mean"] == pytest.approx(
        float(residual.sum() / count), rel=1e-5, abs=1e-7
    )
    dashboard = aggregate_actor_tensorboard_metrics([metrics])
    assert dashboard["credit/action_residual_std"] > 0.0
    assert "credit/action_residual_calibration_slope" in dashboard
    assert "value/action_value_online_explained_variance" in dashboard


def test_fused_statistics_fill_q_as_the_refresh_does():
    wrapper = _memory_wrapper()
    critic = _critic(wrapper, live_head=True)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    refreshed = _scored_batch(wrapper)
    fused = copy.deepcopy(refreshed)
    refresh_old_statistics(wrapper, critic, refreshed)
    fused_metrics = update_minibatch(
        wrapper, critic, fused, optimizers,
        delightful_policy_gradient=True, actor_gae_lambda=0.0,
        fused_behavior_statistics=True, actor_step=False, critic_step=False,
    )
    torch.testing.assert_close(fused.old_action_values, refreshed.old_action_values)
    critic.zero_grad(set_to_none=True)
    wrapper.zero_grad(set_to_none=True)
    refreshed_metrics = update_minibatch(
        wrapper, critic, refreshed, optimizers,
        delightful_policy_gradient=True, actor_gae_lambda=0.0,
        actor_step=False, critic_step=False,
    )
    assert fused_metrics["advantage_mean"] == pytest.approx(
        refreshed_metrics["advantage_mean"], rel=1e-5, abs=1e-8
    )


def test_q_statistics_survive_packing():
    wrapper = _memory_wrapper()
    critic = _critic(wrapper, live_head=True)
    groups = [_scored_batch(wrapper, seed=seed) for seed in (4, 9)]
    packed = pack_rollout_groups_for_replay(groups)
    refresh_old_statistics(wrapper, critic, packed)
    for group in groups:
        refresh_old_statistics(wrapper, critic, group)
    row = 0
    for group in groups:
        rows = group.kind.size(0)
        torch.testing.assert_close(
            packed.old_action_values[row : row + rows, : group.stream_length]
            * group.action_mask,
            group.old_action_values * group.action_mask,
        )
        row += rows


def test_carried_critic_without_a_head_gets_a_fresh_one(tmp_path):
    wrapper = _memory_wrapper()
    source = _critic(wrapper, action_head=False)
    source_optimizers = build_optimizers(wrapper, source, learning_rate=1e-3, fused=False)
    update_minibatch(wrapper, source, _scored_batch(wrapper), source_optimizers, value_only=True)
    sampler = MathPromptSampler([{"id": index} for index in range(4)], seed=3)
    args = _cot_args(
        reasoning_mode="memory", thought_sigma=None,
        init_stop_thinking_probability=None, memory_heads=HEADS,
        memory_head_dim=HEAD_DIM, trunk_optimizer="adamw",
        learning_rate=1e-3, critic_learning_rate=1e-3,
        muon_learning_rate=None, critic_muon_learning_rate=None,
    )
    path = tmp_path / "source.pt"
    save_checkpoint(
        path, wrapper, source, source_optimizers, step=0, args=args,
        sampler=sampler, warmup_step=1,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["critic_action_head_schema"] is None

    target = _critic(wrapper)
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.add_(1.0)
    optimizers = build_optimizers(wrapper, target, learning_rate=1e-3, fused=False)
    carry_init_critic(target, optimizers, payload, args)
    for name, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[name], value), name
    assert not optimizers["critic_action"].state
    assert optimizers["critic"].state
    head_out = target.action_head.out.weight
    assert torch.equal(head_out, torch.ones_like(head_out))  # left as built

    # A run with a head saves it, and a head-less run refuses to carry it.
    path = tmp_path / "headed.pt"
    save_checkpoint(
        path, wrapper, target, optimizers, step=0, args=args,
        sampler=sampler, warmup_step=1,
    )
    headed = torch.load(path, map_location="cpu", weights_only=False)
    assert headed["critic_action_head_schema"] == CRITIC_ACTION_HEAD_SCHEMA
    plain = _critic(wrapper, action_head=False)
    with pytest.raises(ValueError, match="action head"):
        carry_init_critic(
            plain,
            build_optimizers(wrapper, plain, learning_rate=1e-3, fused=False),
            headed, args,
        )


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--critic-action-head-hidden", "0"], "must be positive"),
        (["--critic-action-head-learning-rate", "1e-4"], "needs --critic-action-head-hidden"),
        (["--value-warmup-scope", "action-head"], "needs --critic-action-head-hidden"),
        (
            ["--critic-action-head-hidden", "8", "--target-policy-optimization"],
            "TPO does not",
        ),
        (
            ["--critic-action-head-hidden", "8", "--value-warmup-scope", "action-head"],
            "carry a critic",
        ),
    ],
)
def test_action_head_cli_refusals(extra, message, capsys):
    parser = build_arg_parser()
    args = parser.parse_args(["--checkpoint", "c", "--output", "o", *extra])
    with pytest.raises(SystemExit):
        validate_args(parser, args)
    assert message in capsys.readouterr().err


def test_action_head_cli_defaults_its_rate_to_the_critic_rate():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--checkpoint", "c", "--output", "o",
        "--critic-action-head-hidden", "8", "--critic-learning-rate", "3e-5",
    ])
    validate_args(parser, args)
    assert args.critic_action_head_learning_rate == 3e-5
    assert args.value_warmup_scope == "critic"


def test_only_an_action_head_warmup_rewinds_its_prompts():
    parser = build_arg_parser()
    base = ["--checkpoint", "c", "--output", "o"]
    assert not rewinds_warmup_prompts(
        parser.parse_args([*base, "--value-warmup-steps", "50"])
    )
    headed = [*base, "--critic-action-head-hidden", "8", "--value-warmup-scope", "action-head"]
    assert rewinds_warmup_prompts(parser.parse_args([*headed, "--value-warmup-steps", "50"]))
    assert not rewinds_warmup_prompts(parser.parse_args([*headed, "--value-warmup-steps", "0"]))


def test_rewound_warmup_leaves_every_row_to_the_actor():
    # 50 head-warmup minibatches overlap the actor's first 50, so the actor
    # still owns all 17,917 rows: 1,120 steps, the last one partial.
    fresh = plan_one_pass_training(
        dataset_rows=17_917,
        sampler_cursor=0,
        warmup_updates=50,
        remaining_actor_steps=1_120,
        prompts_per_minibatch=16,
        actor_start_cursor=0,
    )
    assert fresh == (17_917, 1_120)
    # Resumed 20 updates into the warmup: the cursor has advanced, but the
    # actor still starts from the rewind target.
    resumed = plan_one_pass_training(
        dataset_rows=17_917,
        sampler_cursor=20 * 16,
        warmup_updates=30,
        remaining_actor_steps=1_120,
        prompts_per_minibatch=16,
        actor_start_cursor=0,
    )
    assert resumed == (17_917 - 20 * 16, 1_120)
    with pytest.raises(ValueError, match="cannot start past"):
        plan_one_pass_training(
            dataset_rows=17_917,
            sampler_cursor=0,
            warmup_updates=50,
            remaining_actor_steps=1_120,
            prompts_per_minibatch=16,
            actor_start_cursor=16,
        )
