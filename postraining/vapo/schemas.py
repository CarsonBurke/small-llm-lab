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
# The optional dueling action head: Q(s_t, a_t) = V(s_t) + a zero-initialized
# relu^2 MLP of the critic belief at t and its input at t + 1, with V and both
# inputs detached, trained by unclipped squared error against the same
# lambda-one return; the actor bootstraps its advantages from Q.
CRITIC_ACTION_HEAD_SCHEMA = "dueling_detached_relu2_mlp_q_lambda_one_mse/v1"
# The standalone Q critic (REPPO's): no V head. Q over the whole vocabulary,
# Q(s_t, a) = c(f_t) + u(f_t) . (x(a) - sum_b pi(b | s_t) x(b)), centred
# under the sampling policy so c is its value (v5 fit c + u . x(a)), with x
# the critic trunk's own fixed readout row of each token over the table's
# RMS (v4 read its input latent at unit RMS), u an unscaled direction
# readout plus a linear path from the belief plus a scalar gain on the
# belief at the head rate (v6 had no gain), and f_t a zero-readout relu^2
# layer over the critic belief at t and, under a carry, the unit-RMS actor
# belief that sampled a_t; trained end to end by unclipped squared error
# against undiscounted generalized value estimates bootstrapped from its
# own Q of the next taken token.
CRITIC_Q_SCHEMA = (
    "standalone_token_bilinear_q_policy_centered_fixed_rms_readout_row_"
    "unscaled_direction_readout_belief_gain_actor_belief_gve_mse/v7"
)
# TD3/TD7's twin critics on that readout: two independently initialized Q
# heads over one trunk and one action table, each regressing the same
# generalized value estimates bootstrapped from the smaller of their Q of
# the next taken token (clipped double Q); the actor reads their mean.
CRITIC_TWIN_Q_SCHEMA = (
    "twin_standalone_token_bilinear_q_policy_centered_fixed_rms_readout_row_"
    "unscaled_direction_readout_belief_gain_actor_belief_clipped_double_q_gve_mse/v4"
)
CRITIC_POST_ACTION_Q_SCHEMA = (
    "independent_post_action_binary_success_logits_mc_bce_no_actor_memory/v1"
)
POST_ACTION_BINARY_DG_SUFFIX = (
    "+conditional_binary_dg_post_action_q_no_baseline_no_q_gate/v1"
)
# Actor-objective components appended to the base estimator's schema.
# A Q critic's actor advantage is Q of the taken token centered exactly
# under the policy over the full vocabulary, in raw units.
Q_ADVANTAGE_ACTOR_OBJECTIVE_SUFFIX = "+q_minus_exact_policy_mean_q_raw_units/v2"
# REPPO's entropy term: the exact vocabulary entropy over the actor's emit
# slots, weighted by a temperature that Adam tunes in log space toward a
# target entropy.
ENTROPY_TEMPERATURE_ACTOR_OBJECTIVE_SUFFIX = (
    "+exact_entropy_bonus_adam_log_temperature_target/v1"
)
# The per-step KL bound: the realized mean token KL(old || new) on the
# step's own minibatch, read in fp32 before and after the step on a fixed
# subset of its shards, and held under the bound by backtracking the step.
# v3 measured in bf16, whose rounding floor (2-3e-4) exceeded the step.
KL_TRUST_REGION_ACTOR_OBJECTIVE_SUFFIX = (
    "+post_step_token_kl_fp32_shard_subset_backtrack/v4"
)
ACTOR_OBJECTIVE_SCHEMA = (
    "vapo_joint_gate_token_gaussian_clip_token_denominator/v4"
)
# Mean Actor-Critic: at every generated state, minus the policy expectation
# of the token Q critic's detached policy-centered Q over the full
# vocabulary, each state weighted equally. v1 of the think-only variant
# (10-15) excluded every state from the closing think fence on.
MEAN_ACTOR_CRITIC_ACTOR_OBJECTIVE_SCHEMA = (
    "mean_actor_critic_all_generated_states_all_actions_token_denominator/v1"
)
# The same states and weights with the sampled token's Monte Carlo residual
# G - Q(s, a_t) as a score-function control variate (unbiased for any
# critic), and its control: the sampled term alone with G - E_pi Q; and that
# term under the Delightful Policy Gradient gate sigmoid((G - E_pi Q) *
# surprisal), eta = 1, or under that gate on the critic's TD residual or on
# its action head's advantage of the sampled token.
MAC_ESTIMATOR_ACTOR_OBJECTIVE_SCHEMAS = {
    "all_action": MEAN_ACTOR_CRITIC_ACTOR_OBJECTIVE_SCHEMA,
    "control_variate": (
        "mean_actor_critic_all_generated_states_all_actions_plus_sampled_"
        "mc_residual_token_denominator/v1"
    ),
    "sampled_baseline": (
        "token_q_all_generated_states_sampled_mc_return_minus_policy_q_"
        "token_denominator/v1"
    ),
    "delightful_sampled_baseline": (
        "token_q_all_generated_states_sampled_mc_return_minus_policy_q_"
        "delight_gate_eta1_token_denominator/v1"
    ),
    "delightful_td_gate": (
        "token_q_all_generated_states_sampled_mc_return_minus_policy_q_"
        "td_advantage_delight_gate_eta1_token_denominator/v1"
    ),
    "delightful_head_gate": (
        "token_q_all_generated_states_sampled_mc_return_minus_policy_q_"
        "head_advantage_delight_gate_eta1_token_denominator/v1"
    ),
    # One coefficient per trajectory on every generated token, from its
    # prompt group alone: the leave-one-out reward advantage (RLOO), and
    # that under the answer-level delight gate sigmoid(U * -log(n_k / G)),
    # eta = 1, n_k the group's count of its graded-answer cluster, rows
    # without a graded answer one shared cluster.
    "rloo_sequence": (
        "sequence_leave_one_out_group_reward_all_generated_states_token_denominator/v1"
    ),
    "delightful_answer_gate": (
        "sequence_leave_one_out_group_reward_answer_reward_cluster_delight_gate_eta1_"
        "shared_ungraded_cluster_all_generated_states_token_denominator/v1"
    ),
}
# The same estimators over a gated-delta memory policy: each generated
# token's input reads the memory of the beliefs before it, and the actor
# gradient reaches the memory's writer and reader through one replay scan.
MEMORY_MAC_ESTIMATOR_ACTOR_OBJECTIVE_SCHEMAS = {
    "all_action": (
        "mean_actor_critic_all_generated_states_all_actions_gated_delta_"
        "memory_token_denominator/v1"
    ),
    "control_variate": (
        "mean_actor_critic_all_generated_states_all_actions_plus_sampled_"
        "mc_residual_gated_delta_memory_token_denominator/v1"
    ),
    "sampled_baseline": (
        "token_q_all_generated_states_sampled_mc_return_minus_policy_q_"
        "gated_delta_memory_token_denominator/v1"
    ),
    "delightful_sampled_baseline": (
        "token_q_all_generated_states_sampled_mc_return_minus_policy_q_"
        "delight_gate_eta1_gated_delta_memory_token_denominator/v1"
    ),
    "delightful_td_gate": (
        "token_q_all_generated_states_sampled_mc_return_minus_policy_q_"
        "td_advantage_delight_gate_eta1_gated_delta_memory_token_denominator/v1"
    ),
    "delightful_head_gate": (
        "token_q_all_generated_states_sampled_mc_return_minus_policy_q_"
        "head_advantage_delight_gate_eta1_gated_delta_memory_token_denominator/v1"
    ),
    "rloo_sequence": (
        "sequence_leave_one_out_group_reward_all_generated_states_gated_delta_memory_"
        "token_denominator/v1"
    ),
    "delightful_answer_gate": (
        "sequence_leave_one_out_group_reward_answer_reward_cluster_delight_gate_eta1_"
        "shared_ungraded_cluster_all_generated_states_gated_delta_memory_"
        "token_denominator/v1"
    ),
}
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
# The gated-delta memory carry (``postraining.memory_carry``): the carry's
# token-only action space and counter-keyed draw, with the carried beliefs
# (prompt beliefs included) written into a per-row fp32 memory that each
# generated token's input reads. Replay adds one chunked memory scan per
# actor and critic pass.
MEMORY_EXECUTION_SCHEMA = (
    "unique_prefix_bucketed_graph_decode_broad_mixture_gated_delta_memory_"
    "carry_token_only_counter_gumbel_request_row_slot_token_rng/v1"
)
MEMORY_REPLAY_NUMERICS_SCHEMA = (
    "compact_token_logprob_next_slot_gated_delta_memory_chunk_scan_"
    "fp32_state_tf32_dot_fused_softcap_readout_exact_replay/v1"
)
MEMORY_ACTOR_OBJECTIVE_SCHEMA = (
    "vapo_token_clip_gated_delta_memory_token_denominator/v1"
)
MEMORY_DELIGHTFUL_ACTOR_OBJECTIVE_SCHEMA = (
    "delightful_token_policy_gated_delta_memory/v1"
)
MEMORY_TARGET_POLICY_ACTOR_OBJECTIVE_SCHEMA = (
    "target_policy_token_odds_gated_delta_memory/v1"
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
    if reasoning_mode in ("carry", "memory"):
        if rollout_scheduler != "lockstep":
            raise ValueError(
                "hidden carry is implemented by the lockstep scheduler only"
            )
        return (
            CARRY_EXECUTION_SCHEMA
            if reasoning_mode == "carry"
            else MEMORY_EXECUTION_SCHEMA
        )
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
    if reasoning_mode == "memory":
        return MEMORY_REPLAY_NUMERICS_SCHEMA
    if reasoning_mode not in ("latent", "cot", "none"):
        raise ValueError(f"unknown reasoning mode {reasoning_mode!r}")
    return REPLAY_NUMERICS_SCHEMA


def optimizer_schema_for_trunk_optimizer(kind: str, *, fresh_muon: bool = False) -> str:
    """Exact update algorithm whose state a checkpoint may restore.

    ``fresh_muon``: the fresh modules' matrices step under Muon
    (``--fresh-learning-rate``) whatever the trunk's optimizer.
    """
    if kind not in ("adamw", "muon"):
        raise ValueError(f"unknown trunk optimizer {kind!r}")
    if kind == "adamw" and not fresh_muon:
        return ADAMW_ALGORITHM_SCHEMA
    return f"{MUON_ALGORITHM_SCHEMA}+{ADAMW_ALGORITHM_SCHEMA}"


def actor_objective_schema(
    reasoning_mode: str,
    delightful_policy_gradient: bool = False,
    target_policy_optimization: bool = False,
    *,
    q_advantage: bool = False,
    post_action_binary_q: bool = False,
    entropy_temperature: bool = False,
    kl_trust_region: bool = False,
    mac_estimator: str | None = None,
) -> str:
    """Exact actor estimator selected for checkpoint provenance."""
    if mac_estimator is not None:
        if (reasoning_mode not in ("cot", "memory") or delightful_policy_gradient
                or target_policy_optimization or not q_advantage
                or post_action_binary_q or entropy_temperature or kl_trust_region):
            raise ValueError("MAC is a standalone cot or memory token-Q actor objective")
        schemas = (MEMORY_MAC_ESTIMATOR_ACTOR_OBJECTIVE_SCHEMAS if reasoning_mode == "memory"
                   else MAC_ESTIMATOR_ACTOR_OBJECTIVE_SCHEMAS)
        if mac_estimator not in schemas:
            raise ValueError(f"unknown MAC estimator {mac_estimator!r}")
        return schemas[mac_estimator]
    if post_action_binary_q and (q_advantage or not delightful_policy_gradient
                                 or target_policy_optimization):
        raise ValueError("post-action binary DG cannot be combined with another Q estimator")
    return _base_actor_objective_schema(
        reasoning_mode, delightful_policy_gradient, target_policy_optimization
    ) + "".join(
        suffix
        for enabled, suffix in (
            (q_advantage, Q_ADVANTAGE_ACTOR_OBJECTIVE_SUFFIX),
            (post_action_binary_q, POST_ACTION_BINARY_DG_SUFFIX),
            (entropy_temperature, ENTROPY_TEMPERATURE_ACTOR_OBJECTIVE_SUFFIX),
            (kl_trust_region, KL_TRUST_REGION_ACTOR_OBJECTIVE_SUFFIX),
        )
        if enabled
    )


def _base_actor_objective_schema(
    reasoning_mode: str,
    delightful_policy_gradient: bool,
    target_policy_optimization: bool,
) -> str:
    if delightful_policy_gradient and target_policy_optimization:
        raise ValueError("actor objectives are mutually exclusive")
    if reasoning_mode == "carry":
        vapo, delightful, target = (
            CARRY_ACTOR_OBJECTIVE_SCHEMA,
            CARRY_DELIGHTFUL_ACTOR_OBJECTIVE_SCHEMA,
            CARRY_TARGET_POLICY_ACTOR_OBJECTIVE_SCHEMA,
        )
    elif reasoning_mode == "memory":
        vapo, delightful, target = (
            MEMORY_ACTOR_OBJECTIVE_SCHEMA,
            MEMORY_DELIGHTFUL_ACTOR_OBJECTIVE_SCHEMA,
            MEMORY_TARGET_POLICY_ACTOR_OBJECTIVE_SCHEMA,
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


def critic_schema_for(
    q_critic: bool, *, post_action_q: bool = False, twin_q: bool = False
) -> str:
    """The value critic's schema, or the standalone (twin) Q critic's."""
    if twin_q and (not q_critic or post_action_q):
        raise ValueError("twin Q heads belong to the standalone token Q critic")
    if post_action_q:
        return CRITIC_POST_ACTION_Q_SCHEMA
    if twin_q:
        return CRITIC_TWIN_Q_SCHEMA
    return CRITIC_Q_SCHEMA if q_critic else CRITIC_SCHEMA


def resume_critic_schema_compatible(
    payload: dict, q_critic: bool = False, *, post_action_q: bool = False,
    twin_q: bool = False,
) -> bool:
    """Strict critic head/loss match; categorical critics are never migrated,
    and neither a V nor a Q critic ever becomes the other, nor one Q head
    a twin pair."""
    return payload.get("critic_schema") == critic_schema_for(
        q_critic, post_action_q=post_action_q, twin_q=twin_q
    )
