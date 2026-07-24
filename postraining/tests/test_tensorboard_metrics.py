from __future__ import annotations

import pytest
import torch

from postraining.train_latent_vapo import (
    aggregate_actor_tensorboard_metrics,
    aggregate_value_diagnostics,
    actor_minibatch_denominators,
    optimizer_minibatch_orders,
    plan_one_pass_training,
    rollout_tensorboard_metrics,
    write_actor_tensorboard_metrics,
)


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


def test_actor_denominators_cover_only_selected_groups_and_positive_tokens() -> None:
    class Group:
        def __init__(self, actions, emits, rewards):
            self.action_mask = torch.tensor(actions, dtype=torch.float32)
            self.gate_mask = torch.tensor(actions, dtype=torch.float32)
            self.emit_mask = torch.tensor(emits, dtype=torch.float32)
            self.reward_scalar = torch.tensor(rewards, dtype=torch.float32)

    groups = [
        Group([[1, 1], [1, 0]], [[1, 1], [1, 0]], [1.0, 0.0]),
        Group([[1, 1]], [[1, 0]], [1.0]),
        Group([[1, 1]], [[1, 1]], [1.0]),
    ]
    actions, gate_actions, positive_tokens = actor_minibatch_denominators(
        groups, [0, 2], positive_reward_threshold=0.5
    )
    assert actions == 5
    assert gate_actions == 5
    assert positive_tokens == 4


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
        "policy_loss": 1.0,
        "thought_reverse_kl_penalty": 0.06,
        "positive_lm_loss": 4.0,
        "gate_entropy_bonus": 0.25,
        "gate_pg_coef": 1.0,
        "thought_pg_coef": 1.0,
        "thought_reverse_kl_coef": 0.3,
        "positive_lm_weight": 0.1,
        "gate_entropy_coef": 0.5,
        "emit_probability": 0.8,
        "thought_adapter_weight_rms": 1e-4,
        "thought_adapter_bias_rms": 2e-4,
        "gate_entropy": 0.5,
        "gate_behavior_kl": 0.01,
        "renderer_behavior_kl": 0.02,
        "thought_behavior_kl_joint": 0.03,
        "policy_behavior_kl_per_action": 0.04,
        "policy_clip_fraction": 0.1,
        "emit_policy_clip_fraction": 0.05,
        "thought_policy_clip_fraction": 0.15,
        "thought_gate_policy_clip_fraction": 0.1,
        "joint_abs_log_ratio_max": 0.7,
        "thought_dim_abs_log_ratio_max": 0.08,
        "thought_joint_abs_log_ratio_max": 0.3,
        "harmful_positive_log_ratio_max": 0.4,
        "thought_trust_d_mean": 0.05,
        "thought_trust_d_max": 0.2,
        "thought_projection_penalty": 0.01,
        "trunk_grad_norm": 1.0,
        "renderer_grad_norm": 2.0,
        "adapter_grad_norm": 3.0,
        "gate_grad_norm": 4.0,
        "sigma_grad_norm": 4.5,
        "thought_mean_grad_norm": 4.75,
        "critic_grad_norm": 5.0,
        "thought_log_sigma_mean": -2.0,
        "thought_log_sigma_std": 0.2,
        "thought_log_sigma_min": -2.5,
        "thought_log_sigma_max": -1.5,
        "thought_sigma_mean": 0.14,
        "thought_expected_noise_norm": 3.1,
        "thought_realized_noise_norm": 3.0,
        "thought_normalized_noise_rms": 1.0,
        "thought_log_sigma_raw_bias_mean": -2.0,
        "thought_log_sigma_weight_rms": 0.01,
        "thought_log_sigma_residual_gain": 0.01,
        "thought_mean_norm": 0.2,
        "thought_mean_weight_rms": 0.00044,
        "thought_mean_bias_rms": 0.0,
        "thought_mean_output_gain": 0.01,
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
        policy_loss=2.0,
        joint_abs_log_ratio_max=0.5,
        thought_dim_abs_log_ratio_max=0.05,
        thought_joint_abs_log_ratio_max=0.25,
        harmful_positive_log_ratio_max=0.2,
        thought_trust_d_mean=0.02,
        thought_trust_d_max=0.1,
        thought_projection_penalty=0.01,
        trunk_grad_norm=10.0,
        critic_grad_norm=11.0,
    )
    last = _actor_metrics(
        action_count=30.0,
        gate_action_count=6.0,
        gate_behavior_kl=3.0,
        policy_loss=4.0,
        joint_abs_log_ratio_max=0.9,
        thought_dim_abs_log_ratio_max=0.09,
        thought_joint_abs_log_ratio_max=0.45,
        harmful_positive_log_ratio_max=0.6,
        thought_trust_d_mean=0.06,
        thought_trust_d_max=0.3,
        thought_projection_penalty=0.02,
        trunk_grad_norm=20.0,
        critic_grad_norm=21.0,
    )
    dashboard = aggregate_actor_tensorboard_metrics([first, last])

    assert dashboard["loss/policy"] == pytest.approx(6.0)
    assert dashboard["kl/thought_reverse_weighted"] == pytest.approx(0.12)
    assert dashboard["kl/gate_behavior"] == pytest.approx(2.5)
    assert dashboard["grad/trunk"] == pytest.approx(20.0)
    assert dashboard["grad/critic"] == pytest.approx(21.0)
    assert dashboard["sigma/log_std_mean"] == pytest.approx(-2.0)
    assert dashboard["sigma/log_std_std"] == pytest.approx(0.2)
    assert dashboard["sigma/expected_noise_norm"] == pytest.approx(3.1)
    assert dashboard["grad/sigma"] == pytest.approx(4.5)
    assert dashboard["grad/thought_mean"] == pytest.approx(4.75)
    assert dashboard["sigma/mean_norm"] == pytest.approx(0.2)
    assert dashboard["sigma/noise_mean_norm_ratio"] == pytest.approx(0.2 / 3.1)
    assert dashboard["sigma/state_residual_gain"] == pytest.approx(0.01)
    assert dashboard["sigma/mean_output_gain"] == pytest.approx(0.01)
    assert dashboard["behavior/thought_adapter_weight_rms"] == pytest.approx(1e-4)
    assert dashboard["behavior/thought_adapter_bias_rms"] == pytest.approx(2e-4)
    assert dashboard["ratio/joint_abs_log_max"] == pytest.approx(0.9)
    assert dashboard["ratio/thought_dim_abs_log_max"] == pytest.approx(0.09)
    assert dashboard["ratio/thought_joint_abs_log_max"] == pytest.approx(0.45)
    assert dashboard["ratio/harmful_positive_log_max"] == pytest.approx(0.6)
    # Trust statistics: d_mean weighted by thought actions (equal counts
    # here, so plain mean), d_max as max, penalty summed like the losses.
    assert dashboard["trust/thought_d_mean"] == pytest.approx(0.04)
    assert dashboard["trust/thought_d_max"] == pytest.approx(0.3)
    assert dashboard["trust/projection_penalty"] == pytest.approx(0.03)
    assert "grad/critic_mean" not in dashboard
    assert dashboard["loss/positive_lm_weighted"] == pytest.approx(0.8)
    assert "thought_behavior_kl_per_dim" not in dashboard
    assert all(
        not tag.startswith("train/") and not tag.startswith("rollout/")
        for tag in dashboard
    )


def test_actor_dashboard_has_only_the_authoritative_joint_policy_loss() -> None:
    metrics = _actor_metrics()
    dashboard = aggregate_actor_tensorboard_metrics([metrics])
    assert dashboard["loss/policy"] == 1.0
    assert dashboard["kl/thought_reverse_weighted"] == pytest.approx(0.06)
    assert dashboard["bonus/gate_entropy_weighted"] == pytest.approx(0.25)
    assert dashboard["loss/actor_total"] == pytest.approx(1.21)
    assert "loss/gate_weighted" not in dashboard
    assert "loss/renderer" not in dashboard
    assert "loss/thought_weighted" not in dashboard


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
        "exact_accuracy": 0.05,
        "exact_within_group_reward_std": 0.1,
        "partial_reward_fraction": 0.6,
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
        "reward/exact_accuracy",
        "reward/exact_within_group_std",
        "reward/partial_fraction",
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
    dashboard.update(
        {
            "kl/post_update_policy_behavior_per_action": 0.2,
            "ratio/post_update_joint_abs_log_max": 0.4,
            "ratio/post_update_thought_joint_abs_log_max": 0.3,
        }
    )
    behavior_age_zero = Writer()
    write_actor_tensorboard_metrics(
        behavior_age_zero, dashboard, behavior_age=0, step=16
    )
    behavior_age_zero_tags = {tag for tag, _, _ in behavior_age_zero.calls}
    assert "debug/behavior_refresh_max_drift" in behavior_age_zero_tags
    assert "kl/gate_behavior" not in behavior_age_zero_tags
    assert "kl/thought_reverse_weighted" not in behavior_age_zero_tags
    assert not any(tag.startswith("clip/") for tag in behavior_age_zero_tags)
    assert "ratio/joint_abs_log_max" not in behavior_age_zero_tags
    assert "ratio/thought_joint_abs_log_max" not in behavior_age_zero_tags
    assert "kl/post_update_policy_behavior_per_action" in behavior_age_zero_tags
    assert "ratio/post_update_joint_abs_log_max" in behavior_age_zero_tags
    assert (
        "ratio/post_update_thought_joint_abs_log_max"
        in behavior_age_zero_tags
    )
    assert "advantage/mean" in behavior_age_zero_tags

    behavior_age_one = Writer()
    write_actor_tensorboard_metrics(
        behavior_age_one, dashboard, behavior_age=1, step=32
    )
    behavior_age_one_tags = {tag for tag, _, _ in behavior_age_one.calls}
    assert "debug/behavior_refresh_max_drift" not in behavior_age_one_tags
    assert "ratio/thought_joint_abs_log_max" in behavior_age_one_tags
    assert "kl/gate_behavior" in behavior_age_one_tags
    assert "kl/thought_reverse_weighted" in behavior_age_one_tags
    assert "clip/policy" in behavior_age_one_tags
    assert "ratio/harmful_positive_log_max" in behavior_age_one_tags
    assert "advantage/mean" in behavior_age_one_tags
