from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from postraining.train_latent_vapo import (
    aggregate_actor_tensorboard_metrics,
    aggregate_value_diagnostics,
    actor_minibatch_action_denominator,
    optimizer_minibatch_orders,
    plan_one_pass_training,
    rollout_tensorboard_metrics,
    source_actor_signal_mask,
    stratified_optimizer_minibatch_orders,
    validate_resume_arg_contract,
    write_actor_tensorboard_metrics,
)
from postraining.train_latent_vapo import RESUME_EXACT_ARG_FIELDS


def test_optimizer_minibatch_orders_are_disjoint_and_complete() -> None:
    generator = torch.Generator().manual_seed(7)
    minibatches = optimizer_minibatch_orders(64, 16, generator)
    assert len(minibatches) == 4
    assert all(len(minibatch) == 16 for minibatch in minibatches)
    assert sorted(index for minibatch in minibatches for index in minibatch) == list(
        range(64)
    )
    assert minibatches[0] != list(range(16))
    assert minibatches == optimizer_minibatch_orders(
        64, 16, torch.Generator().manual_seed(7)
    )
    with pytest.raises(ValueError, match="complete minibatches"):
        optimizer_minibatch_orders(63, 16)


def test_optimizer_minibatch_orders_allow_one_shuffled_terminal_partial() -> None:
    minibatches = optimizer_minibatch_orders(
        29,
        16,
        torch.Generator().manual_seed(11),
        allow_partial_final=True,
    )
    assert [len(minibatch) for minibatch in minibatches] == [16, 13]
    assert sorted(index for batch in minibatches for index in batch) == list(
        range(29)
    )
    assert minibatches == optimizer_minibatch_orders(
        29,
        16,
        torch.Generator().manual_seed(11),
        allow_partial_final=True,
    )


def test_stratified_optimizer_minibatches_preserve_every_source_quota() -> None:
    source_quotas = [28, 20, 8, 8]
    groups = []
    for source_id, quota in enumerate(source_quotas):
        groups.extend(
            SimpleNamespace(
                kind=torch.zeros(16, 1),
                source_id=torch.full((16,), source_id, dtype=torch.long),
            )
            for _ in range(quota)
        )
    batches = stratified_optimizer_minibatch_orders(
        groups, 16, source_quotas, torch.Generator().manual_seed(9)
    )
    assert len(batches) == 4
    for batch in batches:
        assert Counter(int(groups[index].source_id[0]) for index in batch) == {
            0: 7,
            1: 5,
            2: 2,
            3: 2,
        }
    assert sorted(index for batch in batches for index in batch) == list(range(64))


def test_source_actor_mask_fails_closed_only_for_zero_reward_source() -> None:
    sources = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    rewards = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0])
    assert source_actor_signal_mask(sources, rewards).tolist() == [
        True,
        True,
        False,
        False,
        True,
    ]
    assert source_actor_signal_mask(None, rewards).all()


def test_exact_resume_contract_rejects_reward_change_but_allows_step_ceiling() -> None:
    values = {name: 1 for name in RESUME_EXACT_ARG_FIELDS}
    current = SimpleNamespace(**values, steps=40_000)
    validate_resume_arg_contract(values, current)
    current.steps = 80_000
    validate_resume_arg_contract(values, current)
    current.nearby_reward_max = 0.1
    with pytest.raises(ValueError, match="nearby_reward_max"):
        validate_resume_arg_contract(values, current)


def test_one_pass_plan_consumes_all_dapo_rows_once_including_tail() -> None:
    planned_prompts, actor_steps = plan_one_pass_training(
        dataset_rows=17_917,
        sampler_cursor=0,
        warmup_updates=50,
        remaining_actor_steps=1_070,
        prompts_per_minibatch=16,
    )
    assert planned_prompts == 17_917
    assert actor_steps == 1_070
    assert 17_917 - 50 * 16 == 17_117
    assert divmod(17_117, 16) == (1_069, 13)

    # A checkpoint after 400 complete actor updates resumes with the same
    # exact final cursor and one terminal partial minibatch.
    resumed_prompts, resumed_steps = plan_one_pass_training(
        dataset_rows=17_917,
        sampler_cursor=50 * 16 + 400 * 16,
        warmup_updates=0,
        remaining_actor_steps=670,
        prompts_per_minibatch=16,
    )
    assert resumed_prompts == 10_717
    assert resumed_steps == 670

    with pytest.raises(ValueError, match="requires exactly 1070"):
        plan_one_pass_training(
            dataset_rows=17_917,
            sampler_cursor=0,
            warmup_updates=50,
            remaining_actor_steps=1_069,
            prompts_per_minibatch=16,
        )


def test_actor_denominator_covers_only_selected_groups() -> None:
    class Group:
        def __init__(self, actions):
            self.action_mask = torch.tensor(actions, dtype=torch.float32)
            self.stop_mask = torch.tensor(actions, dtype=torch.float32)

    groups = [
        Group([[1, 1], [1, 0]]),
        Group([[1, 1]]),
        Group([[1, 1]]),
    ]
    assert actor_minibatch_action_denominator(groups, [0, 2]) == 5


def _actor_metrics(**overrides: float) -> dict[str, float]:
    metrics = {
        "action_count": 10.0,
        "value_mean": 0.2,
        "value_target_mean": 0.3,
        "value_target_variance": 0.1,
        "value_residual_mean": 0.1,
        "value_residual_variance": 0.05,
        "value_excess_ce": 0.4,
        "advantage_mean": 0.1,
        "advantage_std": 0.2,
        "policy_loss": 1.0,
        "token_behavior_kl": 0.02,
        "policy_behavior_kl_per_action": 0.04,
        "policy_clip_fraction": 0.1,
        "token_abs_log_ratio_max": 0.7,
        "harmful_positive_log_ratio_max": 0.4,
        "trunk_grad_norm": 1.0,
        "renderer_grad_norm": 2.0,
        "combiner_grad_norm": 3.0,
        "critic_grad_norm": 5.0,
    }
    metrics.update(overrides)
    return metrics


def test_value_diagnostics_reconstruct_global_explained_variance() -> None:
    groups = [
        _actor_metrics(
            action_count=2.0,
            value_target_mean=0.0,
            value_target_variance=0.0,
            value_residual_mean=0.0,
            value_residual_variance=0.0,
        ),
        _actor_metrics(
            action_count=2.0,
            value_target_mean=0.5,
            value_target_variance=0.25,
            value_residual_mean=0.0,
            value_residual_variance=0.0625,
        ),
    ]
    aggregated = aggregate_value_diagnostics(groups)
    assert aggregated["explained_variance"] == pytest.approx(5.0 / 6.0)
    # Averaging per-group EV would instead report (0 + .75) / 2 = .375.
    assert aggregated["explained_variance"] != pytest.approx(0.375)


def test_actor_dashboard_is_compact_and_uses_correct_weights() -> None:
    first = _actor_metrics(
        action_count=10.0,
        token_behavior_kl=1.0,
        policy_loss=2.0,
        token_abs_log_ratio_max=0.5,
        harmful_positive_log_ratio_max=0.2,
        trunk_grad_norm=10.0,
        critic_grad_norm=11.0,
    )
    last = _actor_metrics(
        action_count=30.0,
        token_behavior_kl=3.0,
        policy_loss=4.0,
        token_abs_log_ratio_max=0.9,
        harmful_positive_log_ratio_max=0.6,
        trunk_grad_norm=20.0,
        critic_grad_norm=21.0,
    )
    dashboard = aggregate_actor_tensorboard_metrics([first, last])

    # Losses are contributions to be summed; KLs are action-weighted means;
    # extrema are maxima; grad norms come from the final (complete) group.
    assert dashboard["loss/policy"] == pytest.approx(6.0)
    assert dashboard["loss/actor_total"] == pytest.approx(6.0)
    assert dashboard["kl/token_behavior"] == pytest.approx(2.5)
    assert dashboard["ratio/token_abs_log_max"] == pytest.approx(0.9)
    assert dashboard["ratio/harmful_positive_log_max"] == pytest.approx(0.6)
    assert dashboard["grad/trunk"] == pytest.approx(20.0)
    assert dashboard["grad/renderer"] == pytest.approx(2.0)
    assert dashboard["grad/combiner"] == pytest.approx(3.0)
    assert dashboard["grad/critic"] == pytest.approx(21.0)
    assert "grad/critic_mean" not in dashboard
    assert all(
        not tag.startswith("train/") and not tag.startswith("rollout/")
        for tag in dashboard
    )


def test_actor_dashboard_has_only_the_authoritative_policy_loss() -> None:
    metrics = _actor_metrics()
    dashboard = aggregate_actor_tensorboard_metrics([metrics])
    assert dashboard["loss/policy"] == 1.0
    assert dashboard["loss/actor_total"] == pytest.approx(1.0)
    assert "loss/gate_weighted" not in dashboard
    assert "loss/renderer" not in dashboard
    assert "loss/thought_weighted" not in dashboard


def test_rollout_dashboard_drops_duplicate_and_constant_plumbing() -> None:
    raw = {
        "reward_mean": 0.2,
        "reward_std": 0.4,
        "within_group_reward_std": 0.4,
        "exact_accuracy": 0.05,
        "exact_within_group_reward_std": 0.1,
        "partial_reward_fraction": 0.6,
        "ended_fraction": 0.8,
        "actions_per_trajectory": 16.0,
        "trajectories": 512,
    }
    dashboard = rollout_tensorboard_metrics(raw)
    assert dashboard["reward/mean"] == pytest.approx(0.2)
    assert "rollout/reward_std" not in dashboard
    assert "rollout/trajectories" not in dashboard
    assert set(dashboard) == {
        "reward/mean",
        "reward/within_group_std",
        "reward/exact_accuracy",
        "reward/exact_within_group_std",
        "reward/partial_fraction",
        "reward/ended_fraction",
        "behavior/actions_per_trajectory",
    }


def test_actor_writer_logs_one_compact_row_per_optimizer_step() -> None:
    class Writer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, float, int]] = []

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            self.calls.append((tag, value, step))

    dashboard = aggregate_actor_tensorboard_metrics([_actor_metrics()])
    dashboard.update(
        {
            "kl/post_update_token_behavior": 0.2,
            "ratio/post_update_token_abs_log_max": 0.4,
        }
    )
    behavior_age_zero = Writer()
    write_actor_tensorboard_metrics(
        behavior_age_zero, dashboard, behavior_age=0, step=16
    )
    behavior_age_zero_tags = {tag for tag, _, _ in behavior_age_zero.calls}
    assert "debug/behavior_refresh_max_drift" in behavior_age_zero_tags
    # Fresh-behavior rows elide the exactly-zero drift statistics...
    assert "kl/token_behavior" not in behavior_age_zero_tags
    assert not any(tag.startswith("clip/") for tag in behavior_age_zero_tags)
    assert "ratio/token_abs_log_max" not in behavior_age_zero_tags
    # ...but keep the post-update diagnostics, which measure the real move.
    assert "kl/post_update_token_behavior" in behavior_age_zero_tags
    assert "ratio/post_update_token_abs_log_max" in behavior_age_zero_tags
    assert "advantage/mean" in behavior_age_zero_tags

    behavior_age_one = Writer()
    write_actor_tensorboard_metrics(
        behavior_age_one, dashboard, behavior_age=1, step=32
    )
    behavior_age_one_tags = {tag for tag, _, _ in behavior_age_one.calls}
    assert "debug/behavior_refresh_max_drift" not in behavior_age_one_tags
    assert "kl/token_behavior" in behavior_age_one_tags
    assert "clip/policy" in behavior_age_one_tags
    assert "ratio/harmful_positive_log_max" in behavior_age_one_tags
    assert "advantage/mean" in behavior_age_one_tags
