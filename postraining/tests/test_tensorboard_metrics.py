from __future__ import annotations

import pytest
import torch

from postraining.train_latent_vapo import (
    aggregate_actor_tensorboard_metrics,
    aggregate_value_diagnostics,
    bounded_epoch_order,
    rollout_tensorboard_metrics,
    write_actor_tensorboard_metrics,
)


def test_bounded_epoch_order_stops_exactly_at_training_step_cap() -> None:
    torch.manual_seed(7)
    assert len(bounded_epoch_order(16, 100)) == 16
    partial = bounded_epoch_order(16, 7)
    assert len(partial) == 7
    assert len(set(partial)) == 7
    assert bounded_epoch_order(16, 0) == []


def _actor_metrics(**overrides: float) -> dict[str, float]:
    metrics = {
        "action_count": 10.0,
        "gate_action_count": 9.0,
        "emit_action_count": 6.0,
        "think_action_count": 3.0,
        "forced_initial_think_action_count": 1.0,
        "thought_action_count": 4.0,
        "value_mean": 0.2,
        "value_target_mean": 0.3,
        "value_target_variance": 0.1,
        "value_residual_mean": 0.1,
        "value_residual_variance": 0.05,
        "value_excess_ce": 0.4,
        "advantage_mean": 0.1,
        "advantage_std": 0.2,
        "think_advantage_mean": 0.3,
        "forced_initial_think_advantage_mean": 0.4,
        "thought_advantage_mean": 0.325,
        "emit_advantage_mean": -0.1,
        "gate_loss": 1.0,
        "renderer_loss": 2.0,
        "thought_loss": 3.0,
        "positive_lm_loss": 4.0,
        "gate_pg_coef": 1.0,
        "thought_pg_coef": 1.0,
        "positive_lm_weight": 0.1,
        "emit_probability": 0.8,
        "gate_entropy": 0.5,
        "gate_behavior_kl": 0.01,
        "renderer_behavior_kl": 0.02,
        "thought_behavior_kl_joint": 0.03,
        "policy_behavior_kl_per_action": 0.04,
        "gate_clip_fraction": 0.1,
        "renderer_clip_fraction": 0.2,
        "thought_clip_fraction": 0.3,
        "trunk_grad_norm": 1.0,
        "renderer_grad_norm": 2.0,
        "adapter_grad_norm": 3.0,
        "gate_grad_norm": 4.0,
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
        gate_action_count=2.0,
        gate_behavior_kl=1.0,
        gate_loss=2.0,
        trunk_grad_norm=10.0,
    )
    last = _actor_metrics(
        action_count=30.0,
        gate_action_count=6.0,
        gate_behavior_kl=3.0,
        gate_loss=4.0,
        trunk_grad_norm=20.0,
    )
    dashboard = aggregate_actor_tensorboard_metrics([first, last])

    assert dashboard["loss/gate_weighted"] == pytest.approx(3.0)
    assert dashboard["kl/gate_behavior"] == pytest.approx(2.5)
    assert dashboard["grad/trunk"] == pytest.approx(20.0)
    assert dashboard["loss/positive_lm_weighted"] == pytest.approx(0.4)
    assert "thought_behavior_kl_per_dim" not in dashboard
    assert all(
        not tag.startswith("train/") and not tag.startswith("rollout/")
        for tag in dashboard
    )


def test_actor_dashboard_losses_respect_disabled_policy_components() -> None:
    metrics = _actor_metrics(gate_pg_coef=0.0, thought_pg_coef=0.0)
    dashboard = aggregate_actor_tensorboard_metrics([metrics])
    assert dashboard["loss/gate_weighted"] == 0.0
    assert dashboard["loss/thought_weighted"] == 0.0
    assert dashboard["loss/actor_total"] == pytest.approx(2.4)


def test_actor_dashboard_ignores_empty_conditional_components() -> None:
    metrics = _actor_metrics(
        think_action_count=0.0,
        think_advantage_mean=999.0,
        emit_action_count=0.0,
        renderer_behavior_kl=999.0,
    )
    dashboard = aggregate_actor_tensorboard_metrics([metrics])
    assert dashboard["advantage/optional_think_mean"] == 0.0
    assert dashboard["kl/renderer_behavior"] == 0.0


def test_rollout_dashboard_drops_duplicate_and_constant_plumbing() -> None:
    raw = {
        "reward_mean": 0.2,
        "reward_std": 0.4,
        "within_group_reward_std": 0.4,
        "reward_mean_forced_initial": 0.1,
        "reward_mean_unforced_initial": 0.3,
        "ended_fraction": 0.8,
        "think_fraction": 0.25,
        "thoughts_per_trajectory": 4.0,
        "emits_per_trajectory": 12.0,
        "actions_per_trajectory": 16.0,
        "think_run_p95": 2.0,
        "think_run_max": 3.0,
        "trajectories": 512,
        "forced_initial_trajectory_fraction": 0.5,
    }
    dashboard = rollout_tensorboard_metrics(raw)
    assert dashboard["reward/forced_initial_delta"] == pytest.approx(-0.2)
    assert "rollout/reward_std" not in dashboard
    assert "rollout/actions_per_trajectory" not in dashboard
    assert "rollout/trajectories" not in dashboard
    assert set(dashboard) == {
        "reward/mean",
        "reward/within_group_std",
        "reward/forced_initial_mean",
        "reward/unforced_initial_mean",
        "reward/forced_initial_delta",
        "reward/ended_fraction",
        "behavior/optional_think_fraction",
        "behavior/thoughts_per_trajectory",
        "behavior/emits_per_trajectory",
        "behavior/think_run_p95",
        "behavior/think_run_max",
    }


def test_actor_writer_logs_one_compact_row_per_optimizer_step() -> None:
    class Writer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, float, int]] = []

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            self.calls.append((tag, value, step))

    dashboard = aggregate_actor_tensorboard_metrics([_actor_metrics()])
    epoch_zero = Writer()
    write_actor_tensorboard_metrics(epoch_zero, dashboard, epoch=0, step=16)
    epoch_zero_tags = {tag for tag, _, _ in epoch_zero.calls}
    assert "debug/behavior_refresh_max_drift" in epoch_zero_tags
    assert not any(tag.startswith("kl/") for tag in epoch_zero_tags)
    assert not any(tag.startswith("clip/") for tag in epoch_zero_tags)
    assert "advantage/mean" in epoch_zero_tags

    epoch_one = Writer()
    write_actor_tensorboard_metrics(epoch_one, dashboard, epoch=1, step=32)
    epoch_one_tags = {tag for tag, _, _ in epoch_one.calls}
    assert "debug/behavior_refresh_max_drift" not in epoch_one_tags
    assert "kl/gate_behavior" in epoch_one_tags
    assert "clip/gate" in epoch_one_tags
    assert not any(tag.startswith("advantage/") for tag in epoch_one_tags)
