"""Versioned execution schemas gating checkpoint and resume compatibility."""

from __future__ import annotations

from postraining.muon import MUON_ALGORITHM_SCHEMA


EXECUTION_SCHEMA = (
    "unique_prefix_compact_tail_broad_mixture_forced_initial_think_"
    "one_way_stop_vector_sigma_isotropic_trajectory_position_rng/v29"
)
# Lockstep cot/none: tokens are the only actions, drawn by a counter-keyed
# Gumbel race whose key is (pool seed, request, sample, stream slot) rather
# than the global generator's position. That makes a trajectory's tokens
# independent of batch shape and compaction, which is what lets the
# CUDA-graph decode arena replay them; it also changes every sampled token
# for a given seed, so v29 checkpoints of these modes cannot resume here.
# v32 keys the slot within the trajectory's own unpadded stream instead of
# the chunk's left-padded one, so the draws no longer depend on which prompts
# share a chunk; every draw changes, so a v31 checkpoint resumes exactly only
# under v31.
TOKEN_EXECUTION_SCHEMA = (
    "unique_prefix_bucketed_graph_decode_broad_mixture_pinned_emit_"
    "counter_gumbel_request_row_slot_token_rng/v32"
)
# v31 differs from v32 only in the slot its draws are keyed on. Its policy,
# critic, optimizer state and prompt cursor mean exactly what v32's do and it
# carries no sampler state, so an explicitly acknowledged resume
# (--allow-token-rng-migration) may continue a v31 run under v32: every later
# draw is a v32 draw, and nothing earlier is replayed.
PADDED_SLOT_TOKEN_EXECUTION_SCHEMA = (
    "unique_prefix_bucketed_graph_decode_broad_mixture_pinned_emit_"
    "counter_gumbel_request_slot_token_rng/v31"
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
# The nano readout tail is one fused op (softcapped target log-prob with a
# single-pass backward), so replay log-probs round differently from v4.
REPLAY_NUMERICS_SCHEMA = (
    "compact_emit_gate_raw_gaussian_next_slot_fused_softcap_readout_"
    "exact_replay/v5"
)
# Deterministic hidden carry (``--reasoning-mode carry``) restores the v28
# contract on the v29 trainer: tokens are the only actions, and each generated
# token's input adds the detached belief that produced it. It shares neither
# stream semantics nor objective terms with the gate/Gaussian family above,
# so every schema is its own string. Its tokens are drawn by the same
# counter-keyed race as lockstep cot, which is what makes a zero-init carry
# rollout bitwise the cot rollout.
CARRY_EXECUTION_SCHEMA = (
    "unique_prefix_bucketed_graph_decode_broad_mixture_deterministic_"
    "hidden_carry_token_only_counter_gumbel_request_row_slot_token_rng/v32"
)
CARRY_REPLAY_NUMERICS_SCHEMA = (
    "compact_token_logprob_next_slot_hidden_carry_fused_softcap_readout_"
    "exact_replay/v6"
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

    Lockstep latent keeps the v29 string (its bytes are a resume invariant
    of in-flight runs); lockstep cot/none decode through the graph arena
    under ``TOKEN_EXECUTION_SCHEMA``. Continuous refill keeps its own
    request-stable sampler for every non-carry mode. Carry has no
    continuous-refill variant: the paged scheduler does not implement the
    hidden carry.
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
        return EXECUTION_SCHEMA if reasoning_mode == "latent" else (
            TOKEN_EXECUTION_SCHEMA
        )
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
    payload: dict,
    *,
    expected_execution_schema: str,
    allow_token_rng_migration: bool = False,
) -> bool:
    """Require the exact execution contract, bar one acknowledged migration.

    The v29 latent stream, RNG, and policy semantics are incompatible with
    the v28 deterministic-carry checkpoints, and the v32 counter-keyed token
    contracts with both and with the generator-drawn v29/v30 cot and carry
    runs they replace. The single migration is lockstep cot/none from the
    padded-slot-keyed v31 draw to v32, and only when the caller allows it.
    """
    saved = payload.get("execution_schema")
    if saved == expected_execution_schema:
        return True
    return (
        allow_token_rng_migration
        and saved == PADDED_SLOT_TOKEN_EXECUTION_SCHEMA
        and expected_execution_schema == TOKEN_EXECUTION_SCHEMA
    )


def resume_replay_schema_compatible(payload: dict, reasoning_mode: str) -> bool:
    """Strict replay-numerics match; old stream layouts are never migrated."""
    return payload.get("replay_numerics_schema") == replay_numerics_schema_for(
        reasoning_mode
    )


def resume_critic_schema_compatible(payload: dict) -> bool:
    """Strict critic head/loss match; categorical critics are never migrated."""
    return payload.get("critic_schema") == CRITIC_SCHEMA
