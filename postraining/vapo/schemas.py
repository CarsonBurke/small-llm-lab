"""Versioned execution schemas gating checkpoint and resume compatibility."""

from __future__ import annotations

from postraining.muon import MUON_ALGORITHM_SCHEMA


EXECUTION_SCHEMA = (
    "unique_prefix_compact_tail_broad_mixture_forced_initial_think_"
    "one_way_stop_vector_sigma_isotropic_trajectory_position_rng/v29"
)
CONTINUOUS_REFILL_EXECUTION_SCHEMA_SUFFIX = (
    "+request_stable_gate_token_gaussian_refill_paged_flex/v3"
)
PROMPT_ORDER_SCHEMA = "sequential_one_pass/v1"
# The separate critic's value head and loss: one zero-initialized fp32 linear
# readout of the critic trunk's belief, trained by unclipped token-weighted
# squared error against the lambda-one (Monte Carlo) return. Replaces the HL-Gauss
# categorical head, whose point prior pinned the decoded value near 0.
CRITIC_SCHEMA = "scalar_zero_init_linear_unclipped_mse/v1"
ACTOR_OBJECTIVE_SCHEMA = (
    "vapo_joint_gate_token_gaussian_clip_token_denominator/v4"
)
DELIGHTFUL_ACTOR_OBJECTIVE_SCHEMA = (
    "delightful_token_policy_plus_vapo_gate_gaussian/v2"
)
TARGET_POLICY_ACTOR_OBJECTIVE_SCHEMA = (
    "target_policy_token_odds_plus_vapo_gate_gaussian/v7"
)
REPLAY_NUMERICS_SCHEMA = (
    "compact_emit_gate_raw_gaussian_next_slot_exact_replay/v4"
)
# Deterministic hidden carry (``--reasoning-mode carry``) restores the v28
# contract on the v29 trainer: tokens are the only actions, and each generated
# token's input adds the detached belief that produced it. It shares neither
# stream semantics nor objective terms with the gate/Gaussian family above,
# so every schema is its own string. Its tokens are drawn from the ordinary
# generator lane exactly like cot, which is what makes a zero-init carry
# rollout bitwise the cot rollout.
CARRY_EXECUTION_SCHEMA = (
    "unique_prefix_compact_tail_broad_mixture_deterministic_hidden_carry_"
    "token_only_generator_token_rng/v30"
)
CARRY_REPLAY_NUMERICS_SCHEMA = (
    "compact_token_logprob_next_slot_hidden_carry_exact_replay/v5"
)
CARRY_ACTOR_OBJECTIVE_SCHEMA = "vapo_token_clip_hidden_carry_token_denominator/v4"
CARRY_DELIGHTFUL_ACTOR_OBJECTIVE_SCHEMA = (
    "delightful_token_policy_hidden_carry/v2"
)
CARRY_TARGET_POLICY_ACTOR_OBJECTIVE_SCHEMA = (
    "target_policy_token_odds_hidden_carry/v7"
)
ADAMW_ALGORITHM_SCHEMA = (
    "torch_adamw_betas0.9_0.999_eps1e-8_amsgrad_false_weight_decay0/v1"
)


def execution_schema_for(
    reasoning_mode: str, rollout_scheduler: str = "lockstep"
) -> str:
    """Execution schema for a reasoning mode under a rollout scheduler.

    latent, cot, and none share the v29 string (its bytes are a resume
    invariant of in-flight runs). Carry has no continuous-refill variant:
    the paged scheduler does not implement the hidden carry.
    """
    if reasoning_mode == "carry":
        if rollout_scheduler != "lockstep":
            raise ValueError(
                "hidden carry is implemented by the lockstep scheduler only"
            )
        return CARRY_EXECUTION_SCHEMA
    if reasoning_mode not in ("latent", "cot", "none"):
        raise ValueError(f"unknown reasoning mode {reasoning_mode!r}")
    if rollout_scheduler == "lockstep":
        return EXECUTION_SCHEMA
    if rollout_scheduler == "continuous_refill":
        return EXECUTION_SCHEMA + CONTINUOUS_REFILL_EXECUTION_SCHEMA_SUFFIX
    raise ValueError(f"unknown rollout scheduler {rollout_scheduler!r}")


def replay_numerics_schema_for(reasoning_mode: str) -> str:
    """Replay-numerics schema a reasoning mode refreshes and updates under."""
    if reasoning_mode == "carry":
        return CARRY_REPLAY_NUMERICS_SCHEMA
    if reasoning_mode not in ("latent", "cot", "none"):
        raise ValueError(f"unknown reasoning mode {reasoning_mode!r}")
    return REPLAY_NUMERICS_SCHEMA


def optimizer_schema_for_trunk_optimizer(kind: str) -> str:
    """Exact update algorithm whose state a checkpoint may restore."""
    if kind == "adamw":
        return ADAMW_ALGORITHM_SCHEMA
    if kind == "muon":
        return f"{MUON_ALGORITHM_SCHEMA}+{ADAMW_ALGORITHM_SCHEMA}"
    raise ValueError(f"unknown trunk optimizer {kind!r}")


def actor_objective_schema(
    reasoning_mode: str,
    delightful_policy_gradient: bool = False,
    target_policy_optimization: bool = False,
) -> str:
    """Exact actor estimator selected for checkpoint provenance."""
    if delightful_policy_gradient and target_policy_optimization:
        raise ValueError("actor objectives are mutually exclusive")
    if reasoning_mode == "carry":
        vapo, delightful, target = (
            CARRY_ACTOR_OBJECTIVE_SCHEMA,
            CARRY_DELIGHTFUL_ACTOR_OBJECTIVE_SCHEMA,
            CARRY_TARGET_POLICY_ACTOR_OBJECTIVE_SCHEMA,
        )
    elif reasoning_mode in ("latent", "cot", "none"):
        vapo, delightful, target = (
            ACTOR_OBJECTIVE_SCHEMA,
            DELIGHTFUL_ACTOR_OBJECTIVE_SCHEMA,
            TARGET_POLICY_ACTOR_OBJECTIVE_SCHEMA,
        )
    else:
        raise ValueError(f"unknown reasoning mode {reasoning_mode!r}")
    if target_policy_optimization:
        return target
    return delightful if delightful_policy_gradient else vapo


def resume_execution_schema_compatible(
    payload: dict, *, expected_execution_schema: str
) -> bool:
    """Require the exact execution contract; there are no migrations.

    The v29 stream, RNG, and policy semantics are incompatible with the v28
    deterministic-carry checkpoints, and the v30 carry contract with both.
    """
    return payload.get("execution_schema") == expected_execution_schema


def resume_replay_schema_compatible(payload: dict, reasoning_mode: str) -> bool:
    """Strict replay-numerics match; old stream layouts are never migrated."""
    return payload.get("replay_numerics_schema") == replay_numerics_schema_for(
        reasoning_mode
    )


def resume_critic_schema_compatible(payload: dict) -> bool:
    """Strict critic head/loss match; categorical critics are never migrated."""
    return payload.get("critic_schema") == CRITIC_SCHEMA
