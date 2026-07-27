"""Versioned execution schemas and explicit checkpoint migrations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from postraining.muon import MUON_ALGORITHM_SCHEMA

if TYPE_CHECKING:
    from postraining.value_model import SeparateCritic


EXECUTION_SCHEMA = (
    "forced_initial_one_way_stop_unique_prefix_compact_tail_shuffled_pool1024_disjoint_b256_reverse_kl_thought_trust_anchored_value_orthogonal_silu_adapter_general_lr_sequential_data/v27"
)
IDENTITY_AFFINE_EXECUTION_SCHEMA = (
    "forced_initial_one_way_stop_unique_prefix_compact_tail_shuffled_pool1024_disjoint_b256_reverse_kl_thought_trust_anchored_value_identity_affine_general_lr_sequential_data/v27"
)
TANH_ACTION_EXECUTION_SCHEMA_SUFFIX = "+tanh_raw_gaussian_recurrent_input/v1"
TAIL_MERGE_EXECUTION_SCHEMA_SUFFIX = "+scalar_aligned_rolling_tail_merge/v1"
ZERO_AFFINE_EXECUTION_SCHEMA = (
    "unique_prefix_compact_tail_shuffled_pool1024_disjoint_b256_reverse_kl_thought_trust_anchored_value_zero_affine_general_lr_sequential_data/v24"
)
NO_THOUGHT_KL_EXECUTION_SCHEMA = (
    "unique_prefix_compact_tail_shuffled_pool1024_disjoint_b256_projected_thought_trust_no_kl_anchored_value_zero_affine_general_lr_sequential_data/v23"
)
JOINT_CLIP_EXECUTION_SCHEMA = (
    "unique_prefix_compact_tail_shuffled_pool1024_disjoint_b256_joint_thought_clip_no_kl_anchored_value_zero_affine_general_lr_sequential_data/v22"
)
UNANCHORED_VALUE_EXECUTION_SCHEMA = (
    "unique_prefix_compact_tail_shuffled_pool1024_disjoint_b256_joint_thought_clip_no_kl_zero_affine_general_lr_sequential_data/v21"
)
PER_DIM_REVERSE_KL_EXECUTION_SCHEMA = (
    "unique_prefix_compact_tail_shuffled_pool1024_disjoint_b256_per_dim_clip_reverse_kl_zero_affine_general_lr_sequential_data/v20"
)
PERFORMANCE_COMPATIBLE_EXECUTION_SCHEMA = (
    "shuffled_pool1024_disjoint_b256_per_dim_clip_reverse_kl_zero_affine_general_lr_sequential_data/v19"
)
PREVIOUS_EXECUTION_SCHEMA = (
    "shuffled_pool1024_disjoint_b256_per_dim_thought_clip_zero_affine_general_lr_sequential_data/v18"
)
PROMPT_ORDER_SCHEMA = "sequential_one_pass/v1"
ACTOR_OBJECTIVE_SCHEMA = (
    "vapo_policy_no_positive_example_lm_gate_entropy_per_action/v2"
)
REPLAY_NUMERICS_SCHEMA = "compact_think_head_next_slot_targets/v1"
ADAMW_ALGORITHM_SCHEMA = (
    "torch_adamw_betas0.9_0.999_eps1e-8_amsgrad_false_weight_decay0/v1"
)


def execution_schema_for_adapter(
    kind: str,
    thought_action_transform: str = "identity",
    rollout_scheduler: str = "lockstep",
) -> str:
    """Execution schema for the selected deployed recurrent thought path."""
    if kind == "orthogonal_silu":
        schema = EXECUTION_SCHEMA
    elif kind == "identity_affine":
        schema = IDENTITY_AFFINE_EXECUTION_SCHEMA
    else:
        raise ValueError(f"unknown thought adapter kind {kind!r}")
    if thought_action_transform == "tanh":
        schema += TANH_ACTION_EXECUTION_SCHEMA_SUFFIX
    elif thought_action_transform != "identity":
        raise ValueError(
            f"unknown thought action transform {thought_action_transform!r}"
        )
    if rollout_scheduler == "tail_merge":
        schema += TAIL_MERGE_EXECUTION_SCHEMA_SUFFIX
    elif rollout_scheduler != "lockstep":
        raise ValueError(f"unknown rollout scheduler {rollout_scheduler!r}")
    return schema


def optimizer_schema_for_trunk_optimizer(kind: str) -> str:
    """Exact update algorithm whose state a checkpoint may restore."""
    if kind == "adamw":
        return ADAMW_ALGORITHM_SCHEMA
    if kind == "muon":
        return f"{MUON_ALGORITHM_SCHEMA}+{ADAMW_ALGORITHM_SCHEMA}"
    raise ValueError(f"unknown trunk optimizer {kind!r}")


def resume_execution_schema_compatible(
    payload: dict,
    *,
    expected_execution_schema: str = EXECUTION_SCHEMA,
    allow_reverse_kl_migration: bool = False,
    allow_performance_migration: bool = False,
    allow_joint_clip_migration: bool = False,
    allow_anchored_value_migration: bool = False,
    allow_projected_thought_migration: bool = False,
    allow_thought_reverse_kl_migration: bool = False,
    allow_tail_merge_migration: bool = False,
) -> bool:
    """Resume compatible policy state at a complete rollout-pool boundary."""
    execution_schema = payload.get("execution_schema")
    if execution_schema == expected_execution_schema:
        return not (
            allow_reverse_kl_migration
            or allow_performance_migration
            or allow_joint_clip_migration
            or allow_anchored_value_migration
            or allow_projected_thought_migration
            or allow_thought_reverse_kl_migration
            or allow_tail_merge_migration
        )
    if allow_tail_merge_migration:
        # Scheduling changes only future RNG attribution and batch-shaped
        # numerics. It is a sound migration exclusively at a completed pool
        # boundary from the otherwise-identical lockstep schema.
        return (
            expected_execution_schema.endswith(TAIL_MERGE_EXECUTION_SCHEMA_SUFFIX)
            and execution_schema
            == expected_execution_schema[: -len(TAIL_MERGE_EXECUTION_SCHEMA_SUFFIX)]
            and not (
                allow_reverse_kl_migration
                or allow_performance_migration
                or allow_joint_clip_migration
                or allow_anchored_value_migration
                or allow_projected_thought_migration
                or allow_thought_reverse_kl_migration
            )
        )
    # Every older policy used an affine thought interface. Its saved matrices
    # have the same shapes as v26 but acquire different semantics under
    # 2*SiLU, so no objective flag can make a nonlinear resume sound.
    if expected_execution_schema != IDENTITY_AFFINE_EXECUTION_SCHEMA:
        return False
    # v25 changes only the fresh adapter initialization. A v24 resume restores
    # its learned affine and optimizer state exactly.
    if execution_schema == ZERO_AFFINE_EXECUTION_SCHEMA:
        return not (
            allow_reverse_kl_migration
            or allow_performance_migration
            or allow_joint_clip_migration
            or allow_anchored_value_migration
            or allow_projected_thought_migration
            or allow_thought_reverse_kl_migration
        )
    if execution_schema == NO_THOUGHT_KL_EXECUTION_SCHEMA:
        return (
            allow_thought_reverse_kl_migration
            and not allow_projected_thought_migration
            and not allow_reverse_kl_migration
            and not allow_performance_migration
            and not allow_joint_clip_migration
            and not allow_anchored_value_migration
        )
    if execution_schema == JOINT_CLIP_EXECUTION_SCHEMA:
        return (
            allow_thought_reverse_kl_migration
            and allow_projected_thought_migration
            and not allow_reverse_kl_migration
            and not allow_performance_migration
            and not allow_joint_clip_migration
            and not allow_anchored_value_migration
        )
    if execution_schema == UNANCHORED_VALUE_EXECUTION_SCHEMA:
        return (
            allow_thought_reverse_kl_migration
            and allow_projected_thought_migration
            and allow_anchored_value_migration
            and not allow_reverse_kl_migration
            and not allow_performance_migration
            and not allow_joint_clip_migration
        )
    if execution_schema == PER_DIM_REVERSE_KL_EXECUTION_SCHEMA:
        return (
            allow_thought_reverse_kl_migration
            and allow_projected_thought_migration
            and allow_anchored_value_migration
            and allow_joint_clip_migration
            and not allow_reverse_kl_migration
            and not allow_performance_migration
        )
    if execution_schema == PERFORMANCE_COMPATIBLE_EXECUTION_SCHEMA:
        return (
            allow_thought_reverse_kl_migration
            and allow_projected_thought_migration
            and allow_anchored_value_migration
            and allow_joint_clip_migration
            and allow_performance_migration
            and not allow_reverse_kl_migration
        )
    return (
        allow_reverse_kl_migration
        and allow_performance_migration
        and allow_joint_clip_migration
        and allow_anchored_value_migration
        and allow_projected_thought_migration
        and allow_thought_reverse_kl_migration
        and execution_schema == PREVIOUS_EXECUTION_SCHEMA
    )


def resume_replay_schema_compatible(
    payload: dict,
    *,
    allow_compact_replay_migration: bool = False,
) -> bool:
    """Require an explicit boundary migration from dense replay numerics."""
    source = payload.get("replay_numerics_schema")
    if source == REPLAY_NUMERICS_SCHEMA:
        return not allow_compact_replay_migration
    # v26 and earlier did not label replay numerics and evaluated the mean
    # densely. Unknown labeled schemas are never guessed compatible.
    return source is None and allow_compact_replay_migration


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


def migrate_anchored_value_resume(
    payload: dict, critic: SeparateCritic
) -> dict[str, object]:
    """Load a pre-anchored critic into the anchored-support geometry.

    The trunk and adapter transfer verbatim — they are the critic's learned
    capacity and are grid-agnostic. The value head and support buffers belong
    to the source's [0, 1]-edge grid (a different bin count), so they keep
    the freshly constructed state: zero head weights with the prior projected
    into the bias. Decoded values collapse to the prior until the head
    relearns from the transferred trunk features; the caller must also start
    the critic AdamW optimizer fresh, because its saved moments include the
    old head parameters.
    """
    source = payload["critic"]
    rebuilt_prefixes = ("head.", "support.")
    transferred = {
        key: value
        for key, value in source.items()
        if not key.startswith(rebuilt_prefixes)
    }
    result = critic.load_state_dict(transferred, strict=False)
    expected_missing = {
        key
        for key in critic.state_dict()
        if key.startswith(rebuilt_prefixes)
    }
    if result.unexpected_keys or set(result.missing_keys) != expected_missing:
        raise ValueError(
            "anchored-value migration expects the source critic to differ "
            "only in head/support state; got unexpected keys "
            f"{sorted(result.unexpected_keys)} and missing keys "
            f"{sorted(result.missing_keys)}"
        )
    return {
        "transferred_parameters": len(transferred),
        "rebuilt_state": sorted(expected_missing),
    }
