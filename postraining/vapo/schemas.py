"""Versioned execution schemas gating checkpoint and resume compatibility."""

from __future__ import annotations

from postraining.muon import MUON_ALGORITHM_SCHEMA


EXECUTION_SCHEMA = (
    "unique_prefix_compact_tail_shuffled_pool1024_disjoint_b256_deterministic_hidden_carry_token_clip_anchored_value_general_lr_sequential_data/v28"
)
CONTINUOUS_REFILL_EXECUTION_SCHEMA_SUFFIX = (
    "+request_stable_continuous_refill_paged_flex_attention_cdf_right/v2"
)
PROMPT_ORDER_SCHEMA = "sequential_one_pass/v1"
ACTOR_OBJECTIVE_SCHEMA = (
    "vapo_token_clip_no_positive_example_lm_token_denominator/v3"
)
DELIGHTFUL_ACTOR_OBJECTIVE_SCHEMA = (
    "delightful_policy_gradient_current_token_surprisal_eta1_"
    "detached_gate_no_importance_ratio/v1"
)
TARGET_POLICY_ACTOR_OBJECTIVE_SCHEMA = (
    "target_policy_intra_trajectory_executed_token_raw_gae_"
    "old_policy_odds_target_global_token_mean_eta_controlled_no_pg_aux/v6"
)
REPLAY_NUMERICS_SCHEMA = (
    "compact_token_logprob_and_target_vs_rest_logodds_next_slot_targets/v3"
)
ADAMW_ALGORITHM_SCHEMA = (
    "torch_adamw_betas0.9_0.999_eps1e-8_amsgrad_false_weight_decay0/v1"
)


def execution_schema_for_rollout_scheduler(kind: str = "lockstep") -> str:
    """Execution schema for the deployed rollout scheduler."""
    if kind == "lockstep":
        return EXECUTION_SCHEMA
    if kind == "continuous_refill":
        return EXECUTION_SCHEMA + CONTINUOUS_REFILL_EXECUTION_SCHEMA_SUFFIX
    raise ValueError(f"unknown rollout scheduler {kind!r}")


def optimizer_schema_for_trunk_optimizer(kind: str) -> str:
    """Exact update algorithm whose state a checkpoint may restore."""
    if kind == "adamw":
        return ADAMW_ALGORITHM_SCHEMA
    if kind == "muon":
        return f"{MUON_ALGORITHM_SCHEMA}+{ADAMW_ALGORITHM_SCHEMA}"
    raise ValueError(f"unknown trunk optimizer {kind!r}")


def actor_objective_schema(
    delightful_policy_gradient: bool = False,
    target_policy_optimization: bool = False,
) -> str:
    """Exact actor estimator selected for checkpoint provenance."""
    if delightful_policy_gradient and target_policy_optimization:
        raise ValueError("actor objectives are mutually exclusive")
    if target_policy_optimization:
        return TARGET_POLICY_ACTOR_OBJECTIVE_SCHEMA
    return (
        DELIGHTFUL_ACTOR_OBJECTIVE_SCHEMA
        if delightful_policy_gradient
        else ACTOR_OBJECTIVE_SCHEMA
    )


def resume_execution_schema_compatible(
    payload: dict,
    *,
    expected_execution_schema: str = EXECUTION_SCHEMA,
) -> bool:
    """Resume compatible policy state at a complete rollout-pool boundary.

    v28 replaced the stochastic thought/gate policy with the deterministic
    hidden carry; no saved state from any earlier schema has a sound
    interpretation under it, so compatibility is strict equality — there
    are no migrations.
    """
    return payload.get("execution_schema") == expected_execution_schema


def resume_replay_schema_compatible(payload: dict) -> bool:
    """Strict replay-numerics match; pre-v28 layouts are never migrated."""
    return payload.get("replay_numerics_schema") == REPLAY_NUMERICS_SCHEMA


def value_support_geometry_matches(saved_args: dict, args) -> bool:
    """Whether a checkpoint's critic support geometry matches the CLI's.

    Distinct (bins, margin) pairs can collide on the same total head width
    (e.g. anchored 101/4 and 103/3 both build 110 bins), in which case a
    strict state-dict load succeeds while sigma and the projection width
    silently change — so geometry must be compared as ARGS, not shapes.
    Margin only shapes the grid when the support is anchored.
    """
    return (
        bool(saved_args.get("value_anchored_support", False))
        == args.value_anchored_support
        and saved_args.get("value_bins") == args.value_bins
        and saved_args.get("value_sigma_ratio") == args.value_sigma_ratio
        and (
            not args.value_anchored_support
            or saved_args.get("value_margin_bins") == args.value_margin_bins
        )
    )
