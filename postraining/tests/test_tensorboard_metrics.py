from __future__ import annotations

from collections import Counter
import math
from types import SimpleNamespace

import pytest
import torch

from postraining.train_latent_vapo import (
    aggregate_actor_tensorboard_metrics,
    aggregate_value_diagnostics,
    actor_minibatch_action_denominator,
    optimizer_minibatch_orders,
    plan_one_pass_training,
    resolve_critic_init_provenance,
    rollout_tensorboard_metrics,
    resume_topology_history,
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


def test_source_actor_mask_is_disabled_by_default_and_opt_in() -> None:
    sources = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    rewards = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0])
    assert source_actor_signal_mask(sources, rewards).all()
    assert source_actor_signal_mask(
        sources, rewards, enabled=True
    ).tolist() == [
        True,
        True,
        False,
        False,
        True,
    ]
    assert source_actor_signal_mask(None, rewards, enabled=True).all()


def test_exact_resume_contract_rejects_reward_change_but_allows_step_ceiling() -> None:
    values = {name: 1 for name in RESUME_EXACT_ARG_FIELDS}
    saved = {
        **values,
        "prompts_per_rollout": 16,
        "prompts_per_minibatch": 16,
        "delightful_policy_gradient": True,
        "zero_reward_stop_pools": 8,
    }
    current = SimpleNamespace(
        **saved, steps=40_000, allow_dg_topology_migration=False
    )
    validate_resume_arg_contract(saved, current)
    current.steps = 80_000
    validate_resume_arg_contract(saved, current)
    current.prompts_per_rollout = current.prompts_per_minibatch = 24
    current.zero_reward_stop_pools = 0
    with pytest.raises(ValueError, match="--allow-dg-topology-migration"):
        validate_resume_arg_contract(saved, current)
    current.allow_dg_topology_migration = True
    validate_resume_arg_contract(saved, current)
    current.nearby_reward_max = 0.1
    with pytest.raises(ValueError, match="nearby_reward_max"):
        validate_resume_arg_contract(saved, current)


def test_exact_resume_rejects_vapo_rollout_topology_change() -> None:
    values = {name: 1 for name in RESUME_EXACT_ARG_FIELDS}
    saved = {
        **values,
        "prompts_per_rollout": 64,
        "prompts_per_minibatch": 16,
        "delightful_policy_gradient": False,
    }
    current = SimpleNamespace(
        **values,
        prompts_per_rollout=32,
        prompts_per_minibatch=16,
        delightful_policy_gradient=False,
    )
    with pytest.raises(ValueError, match="outside the one-fresh-batch"):
        validate_resume_arg_contract(saved, current)


def test_resume_topology_history_preserves_and_appends_transition() -> None:
    prior = {"topology_history": [{"source_step": 100}]}
    saved = {"prompts_per_rollout": 16, "prompts_per_minibatch": 16}
    current = SimpleNamespace(
        prompts_per_rollout=24, prompts_per_minibatch=24
    )
    history = resume_topology_history(
        prior,
        saved,
        current,
        checkpoint="run/latent_vapo_checkpoint.pt",
        checkpoint_sha256="sha256:checkpoint",
        step=774,
        sampler_cursor=13_184,
    )
    assert history[0] == {"source_step": 100}
    assert history[1] == {
        "source_checkpoint": "run/latent_vapo_checkpoint.pt",
        "source_checkpoint_sha256": "sha256:checkpoint",
        "source_step": 774,
        "source_sampler_cursor": 13_184,
        "before": {
            "prompts_per_rollout": 16,
            "prompts_per_minibatch": 16,
        },
        "after": {
            "prompts_per_rollout": 24,
            "prompts_per_minibatch": 24,
        },
    }
    assert resume_topology_history(
        {"topology_history": history},
        saved,
        current,
        checkpoint="copied-run/latent_vapo_checkpoint.pt",
        checkpoint_sha256="sha256:checkpoint",
        step=774,
        sampler_cursor=13_184,
    ) == history


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
        "value_loss": 0.4,
        "advantage_mean": 0.1,
        "advantage_std": 0.2,
        "policy_loss": 1.0,
        "token_behavior_kl": 0.02,
        "policy_behavior_kl_per_action": 0.04,
        "policy_clip_fraction": 0.1,
        "stochastic_policy_loss": 0.25,
        "stochastic_policy_clip_fraction": 0.05,
        "token_abs_log_ratio_max": 0.7,
        "harmful_positive_log_ratio_max": 0.4,
        "trunk_grad_norm": 1.0,
        "renderer_grad_norm": 2.0,
        "combiner_grad_norm": 3.0,
        "gate_grad_norm": 3.5,
        "thought_mean_grad_norm": 4.0,
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
    assert dashboard["loss/stochastic_policy"] == pytest.approx(0.5)
    assert dashboard["clip/stochastic_policy"] == pytest.approx(0.1)
    assert dashboard["grad/gate"] == pytest.approx(3.5)
    assert dashboard["grad/thought_mean"] == pytest.approx(4.0)
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


def test_delightful_dashboard_reports_gates_and_drops_ppo_only_metrics() -> None:
    first = _actor_metrics(
        action_count=10.0,
        delightful_gate_mean=0.4,
        delightful_positive_gate_mean=0.8,
        delightful_negative_gate_mean=0.2,
        delightful_positive_count=2.0,
        delightful_negative_count=8.0,
        delightful_delight_mean=-0.3,
        delightful_surprisal_mean=3.0,
        delightful_positive_loss_contribution=1.3,
        delightful_negative_loss_contribution=-0.3,
    )
    last = _actor_metrics(
        action_count=30.0,
        delightful_gate_mean=0.6,
        delightful_positive_gate_mean=0.9,
        delightful_negative_gate_mean=0.1,
        delightful_positive_count=18.0,
        delightful_negative_count=12.0,
        delightful_delight_mean=0.1,
        delightful_surprisal_mean=5.0,
        delightful_positive_loss_contribution=1.2,
        delightful_negative_loss_contribution=-0.2,
    )
    dashboard = aggregate_actor_tensorboard_metrics([first, last])
    assert dashboard["delightful/gate_mean"] == pytest.approx(0.55)
    assert dashboard["delightful/positive_gate_mean"] == pytest.approx(0.89)
    assert dashboard["delightful/negative_gate_mean"] == pytest.approx(0.14)
    assert dashboard["delightful/delight_mean"] == pytest.approx(0.0)
    assert dashboard["delightful/surprisal_mean"] == pytest.approx(4.5)
    assert dashboard["delightful/reinforce_positive_advantage"] == pytest.approx(
        2.5
    )
    assert dashboard["delightful/suppress_negative_advantage"] == pytest.approx(
        -0.5
    )
    assert "clip/policy" not in dashboard
    assert "ratio/harmful_positive_log_max" not in dashboard

    class Writer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, float, int]] = []

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            self.calls.append((tag, value, step))

    writer = Writer()
    write_actor_tensorboard_metrics(writer, dashboard, behavior_age=0, step=1)
    tags = {tag for tag, _, _ in writer.calls}
    assert "debug/behavior_refresh_max_drift" in tags
    assert "delightful/gate_mean" in tags
    assert "clip/policy" not in tags


def test_tpo_dashboard_reports_discrete_target_fit_without_ppo_metrics() -> None:
    first = _actor_metrics(
        action_count=10.0,
        tpo_active_count=2.0,
        tpo_loss=0.3,
        tpo_pre_update_fit_kl=0.03,
        tpo_target_behavior_kl=0.03,
        tpo_old_probability_mean=0.4,
        tpo_pre_update_probability_mean=0.4,
        tpo_target_probability_mean=0.45,
        tpo_target_move_abs_mean=0.1,
        tpo_target_log_odds_shift_abs_mean=0.08,
        tpo_target_log_odds_shift_square_mean=0.01,
        tpo_pre_update_probability_residual_mean=-0.05,
        tpo_pre_update_probability_residual_abs_mean=0.05,
        tpo_pre_update_probability_residual_square_mean=0.004,
        tpo_positive_target_probability_mean=0.5,
        tpo_negative_target_probability_mean=0.3,
        tpo_positive_count=1.0,
        tpo_negative_count=1.0,
    )
    last = _actor_metrics(
        action_count=30.0,
        tpo_active_count=6.0,
        tpo_loss=0.5,
        tpo_pre_update_fit_kl=0.07,
        tpo_target_behavior_kl=0.07,
        tpo_old_probability_mean=0.5,
        tpo_pre_update_probability_mean=0.5,
        tpo_target_probability_mean=0.55,
        tpo_target_move_abs_mean=0.3,
        tpo_target_log_odds_shift_abs_mean=0.24,
        tpo_target_log_odds_shift_square_mean=0.09,
        tpo_pre_update_probability_residual_mean=-0.1,
        tpo_pre_update_probability_residual_abs_mean=0.1,
        tpo_pre_update_probability_residual_square_mean=0.016,
        tpo_positive_target_probability_mean=0.6,
        tpo_negative_target_probability_mean=0.4,
        tpo_positive_count=3.0,
        tpo_negative_count=3.0,
    )
    dashboard = aggregate_actor_tensorboard_metrics([first, last])
    assert dashboard["tpo/loss"] == pytest.approx(0.45)
    assert dashboard["tpo/target_probability_mean"] == pytest.approx(0.525)
    assert dashboard["tpo/target_log_odds_shift_rms"] == pytest.approx(
        math.sqrt(0.07)
    )
    assert dashboard["tpo/pre_update_probability_residual_rms"] == pytest.approx(
        math.sqrt(0.013)
    )
    assert dashboard["tpo/positive_target_probability_mean"] == pytest.approx(
        0.575
    )
    assert dashboard["tpo/negative_target_probability_mean"] == pytest.approx(
        0.375
    )
    assert "clip/policy" not in dashboard
    assert "ratio/harmful_positive_log_max" not in dashboard


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
        "thoughts_per_trajectory": 2.0,
        "continued_thoughts": 512,
        "forced_first_thought_fraction": 1.0,
        "continuation_fraction": 0.5,
        "gate_stop_fraction": 0.5,
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
        "behavior/thoughts_per_trajectory",
        "behavior/continued_thoughts",
        "behavior/forced_first_thought_fraction",
        "behavior/continuation_fraction",
        "behavior/gate_stop_fraction",
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


def _critic_init_args(**overrides) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            "critic_init": "scratch",
            "actor_critic_init": None,
            "resume": None,
            **overrides,
        }
    )


def test_fresh_critic_init_provenance_names_its_start() -> None:
    assert resolve_critic_init_provenance(_critic_init_args(), None, None) == {
        "init": "scratch",
        "checkpoint": None,
        "source_execution_schema": None,
    }
    assert resolve_critic_init_provenance(
        _critic_init_args(critic_init="actor"), None, None
    )["init"] == "actor_copy"
    assert resolve_critic_init_provenance(
        _critic_init_args(actor_critic_init="warm.pt"),
        {"execution_schema": "exec/v30"},
        None,
    ) == {
        "init": "warm_checkpoint",
        "checkpoint": "warm.pt",
        "source_execution_schema": "exec/v30",
    }


def test_resume_keeps_the_recorded_critic_init_and_refuses_a_contradiction() -> None:
    warm = {
        "critic": {
            "init": "warm_checkpoint",
            "checkpoint": "warm.pt",
            "source_execution_schema": "exec/v30",
            "parameters": 7,
        }
    }
    resumed = _critic_init_args(resume="run/latent_vapo_checkpoint.pt")
    # A resume carries no --actor-critic-init, yet the run's critic still
    # started from that warm checkpoint: the record survives the resume.
    assert resolve_critic_init_provenance(resumed, None, warm) == {
        "init": "warm_checkpoint",
        "checkpoint": "warm.pt",
        "source_execution_schema": "exec/v30",
    }
    actor_copy = {"critic": {"init": "actor_copy"}}
    with pytest.raises(ValueError, match="--critic-init actor"):
        resolve_critic_init_provenance(resumed, None, actor_copy)
    resumed.critic_init = "actor"
    assert resolve_critic_init_provenance(resumed, None, actor_copy)[
        "init"
    ] == "actor_copy"
    with pytest.raises(ValueError, match="--critic-init scratch"):
        resolve_critic_init_provenance(
            resumed, None, {"critic": {"init": "scratch"}}
        )
    with pytest.raises(ValueError, match="unknown critic init"):
        resolve_critic_init_provenance(resumed, None, {"critic": {}})
    # A resume into a new directory reads the parent run's manifest, and its
    # own manifest then carries the same record, so a resume chain of any
    # length keeps the original start.
    with pytest.raises(ValueError, match="resumed run's manifest"):
        resolve_critic_init_provenance(resumed, None, None)
    chained = {"critic": resolve_critic_init_provenance(resumed, None, actor_copy)}
    assert resolve_critic_init_provenance(resumed, None, chained)[
        "init"
    ] == "actor_copy"
