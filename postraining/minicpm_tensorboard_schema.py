"""Canonical TensorBoard tag organization for the stopped MiniCPM run."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

MAX_SCALAR_TAGS_PER_CATEGORY = 12
EXPECTED_LEGACY_SCALAR_TAGS = 143
EXPECTED_LEGACY_TEXT_TAGS = 3
TEXT_SUMMARY_SUFFIX = "/text_summary"


def _prefix_map(
    source_prefix: str,
    destination_prefix: str,
    suffixes: tuple[str, ...],
) -> dict[str, str]:
    return {
        f"{source_prefix}/{suffix}": f"{destination_prefix}/{suffix}"
        for suffix in suffixes
    }


LEGACY_SCALAR_TAG_MAP: dict[str, str] = {}
LEGACY_SCALAR_TAG_MAP.update(
    _prefix_map(
        "config",
        "configuration_model",
        (
            "trainable_actor_parameters",
            "critic_parameters",
            "critic_width",
            "lora_rank",
            "lora_alpha",
            "frozen_base_parameters",
            "total_policy_parameters",
            "rollout_replica_parameters",
            "rollout_fused_projection_groups",
            "estimated_static_kv_cache_bytes",
        ),
    )
)
LEGACY_SCALAR_TAG_MAP.update(
    _prefix_map(
        "config",
        "configuration_nextlat",
        (
            "nextlat_head_parameters",
            "nextlat_lr",
            "nextlat_horizon",
            "nextlat_projection_factor",
            "nextlat_draft_length",
            "train_nextlat",
            "nextlat_rollout",
            "nextlat_kl_tokens",
            "nextlat_mse_coefficient",
            "nextlat_kl_coefficient",
        ),
    )
)
LEGACY_SCALAR_TAG_MAP.update(
    _prefix_map(
        "config",
        "configuration_rollout",
        (
            "rollout_batch_rows",
            "prompts_per_rollout",
            "samples_per_prompt",
            "prompt_tokens",
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "compile_rollout",
            "fast_rollout",
            "min_rollout_tokens_per_second",
            "thinking",
        ),
    )
)
LEGACY_SCALAR_TAG_MAP.update(
    _prefix_map(
        "config",
        "configuration_optimization",
        (
            "steps",
            "value_warmup_steps",
            "actor_lr",
            "critic_lr",
            "ppo_epochs",
            "clip_low",
            "clip_high",
            "positive_coefficient",
            "value_coefficient",
            "replay_token_budget",
            "replay_max_trajectories",
            "logit_chunk_tokens",
        ),
    )
)
LEGACY_SCALAR_TAG_MAP.update(
    _prefix_map(
        "config",
        "configuration_runtime",
        (
            "device_telemetry",
            "device_telemetry_interval_ms",
            "device_power_floor",
            "checkpoint_interval_seconds",
            "rollout_only",
            "gate_min_positive_trajectories",
            "gate_min_positive_groups",
            "gate_max_truncation_fraction",
            "seed",
        ),
    )
)
LEGACY_SCALAR_TAG_MAP.update(
    _prefix_map(
        "rollout",
        "rollout_quality",
        (
            "trajectories",
            "prompt_groups",
            "positive_trajectories",
            "positive_groups",
            "mixed_groups",
            "accuracy",
            "truncation_fraction",
            "response_length_mean",
            "response_length_p95",
            "response_length_max",
        ),
    )
)
LEGACY_SCALAR_TAG_MAP.update(
    _prefix_map(
        "rollout",
        "rollout_performance",
        (
            "generated_tokens",
            "rollout_seconds",
            "rollout_tokens_per_second",
            "scheduled_rollout_tokens_per_second",
            "prefill_seconds",
            "decode_seconds",
            "decode_tokens_per_second",
            "target_decode_calls",
            "target_decode_positions",
            "scheduled_decode_tokens_per_second",
            "target_positions_per_decode_call",
        ),
    )
)
LEGACY_SCALAR_TAG_MAP.update(
    _prefix_map(
        "rollout",
        "rollout_sampling",
        (
            "sampling_scanned_vocabulary",
            "sampling_candidate_support",
            "sampling_full_policy_mass_lower_bound",
            "sampling_conditional_mass_lower_bound",
        ),
    )
)
LEGACY_SCALAR_TAG_MAP.update(
    {
        "rollout/replay_storage_bytes": "replay/storage_bytes",
        "rollout/replay_bytes_per_token": "replay/bytes_per_token",
        "train/replay_microbatches": "replay/microbatches",
        "train/replay_actions": "replay/actions",
    }
)
LEGACY_SCALAR_TAG_MAP.update(
    {
        "train/policy_loss": "actor/policy_loss",
        "train/positive_lm_loss": "actor/positive_lm_loss",
        "train/approximate_kl": "actor/approximate_kl",
        "train/sampled_forward_kl": "actor/sampled_forward_kl",
        "train/clip_fraction": "actor/clip_fraction",
        "train/ratio_mean": "actor/ratio_mean",
        "train/ratio_std": "actor/ratio_std",
        "train/actor_grad_norm": "actor/grad_norm",
    }
)
LEGACY_SCALAR_TAG_MAP.update(
    {
        "train/value_loss": "critic/loss",
        "train/value_mean": "critic/prediction_mean",
        "train/value_target_mean": "critic/target_mean",
        "train/explained_variance": "critic/explained_variance",
        "train/critic_grad_norm": "critic/grad_norm",
    }
)
LEGACY_SCALAR_TAG_MAP.update(
    _prefix_map(
        "train",
        "auxiliary_nextlat",
        (
            "nextlat_loss",
            "nextlat_smooth_l1",
            "nextlat_categorical_kl",
            "nextlat_transitions",
            "nextlat_grad_norm",
            "nextlat_optimizer_steps",
        ),
    )
)
LEGACY_SCALAR_TAG_MAP.update(
    {
        "train/loss": "optimization/total_loss",
        "train/update_seconds": "optimization/update_seconds",
    }
)
LEGACY_SCALAR_TAG_MAP.update(
    _prefix_map(
        "rollout_live",
        "system_live",
        (
            "decode_steps",
            "batch_rows",
            "prompt_groups",
            "scheduled_tokens",
            "wall_seconds",
            "scheduled_tokens_per_second",
            "peak_vram_bytes",
            "device_utilization_gpu_percent",
            "device_utilization_memory_percent",
            "device_power_draw_watts",
            "device_clocks_sm_mhz",
            "device_memory_used_mib",
        ),
    )
)

_DEVICE_AGGREGATES = (
    "device_power_draw_watts_mean",
    "device_power_draw_watts_min",
    "device_power_draw_watts_max",
    "device_seconds_below_power_floor",
    "device_clocks_sm_mhz_mean",
    "device_clocks_sm_mhz_min",
    "device_utilization_gpu_percent_mean",
    "peak_vram_bytes",
)
_DEVICE_METADATA = (
    "device_power_draw_watts_period_ms",
    "device_power_draw_watts_readings",
    "device_clocks_sm_mhz_period_ms",
    "device_clocks_sm_mhz_readings",
    "device_utilization_gpu_percent_period_ms",
    "device_utilization_gpu_percent_readings",
)
LEGACY_SCALAR_TAG_MAP.update(
    _prefix_map("rollout", "system_rollout", _DEVICE_AGGREGATES)
)
LEGACY_SCALAR_TAG_MAP.update(
    _prefix_map("rollout", "system_rollout_metadata", _DEVICE_METADATA)
)
LEGACY_SCALAR_TAG_MAP.update(
    _prefix_map("train", "system_update", _DEVICE_AGGREGATES)
)
LEGACY_SCALAR_TAG_MAP.update(
    _prefix_map("train", "system_update_metadata", _DEVICE_METADATA)
)

LEGACY_TEXT_TAG_MAP: dict[str, str] = {
    "config/arguments": "samples/configuration",
    "rollout_samples/correct": "samples/rollout_correct",
    "rollout_samples/incorrect": "samples/rollout_incorrect",
}
LEGACY_TAG_MAP: dict[str, str] = {
    **LEGACY_SCALAR_TAG_MAP,
    **LEGACY_TEXT_TAG_MAP,
}


def scalar_category_counts(
    tag_map: Mapping[str, str] = LEGACY_SCALAR_TAG_MAP,
) -> dict[str, int]:
    """Count destination scalar tags by their TensorBoard category."""
    counts = Counter()
    for destination in tag_map.values():
        category, separator, leaf = destination.partition("/")
        if not separator or not category or not leaf:
            raise ValueError(f"invalid organized TensorBoard tag: {destination!r}")
        counts[category] += 1
    return dict(sorted(counts.items()))


def validate_scalar_category_cap(
    tag_map: Mapping[str, str] = LEGACY_SCALAR_TAG_MAP,
    *,
    maximum: int = MAX_SCALAR_TAGS_PER_CATEGORY,
) -> dict[str, int]:
    """Return category sizes, raising if a category is too broad."""
    if maximum < 1:
        raise ValueError("maximum category size must be positive")
    counts = scalar_category_counts(tag_map)
    oversized = {
        category: count for category, count in counts.items() if count > maximum
    }
    if oversized:
        details = ", ".join(
            f"{category}={count}" for category, count in oversized.items()
        )
        raise ValueError(
            f"TensorBoard scalar categories exceed the {maximum}-tag cap: {details}"
        )
    return counts


def organized_tag(tag: str) -> str:
    """Strictly translate one legacy logical or serialized text tag."""
    try:
        return LEGACY_TAG_MAP[tag]
    except KeyError:
        if tag.endswith(TEXT_SUMMARY_SUFFIX):
            logical_tag = tag[: -len(TEXT_SUMMARY_SUFFIX)]
            try:
                return LEGACY_TEXT_TAG_MAP[logical_tag] + TEXT_SUMMARY_SUFFIX
            except KeyError:
                pass
    raise KeyError(f"unrecognized legacy TensorBoard tag: {tag!r}") from None


def _validate_schema() -> None:
    if len(LEGACY_SCALAR_TAG_MAP) != EXPECTED_LEGACY_SCALAR_TAGS:
        raise RuntimeError(
            "MiniCPM TensorBoard scalar mapping is not total: "
            f"expected {EXPECTED_LEGACY_SCALAR_TAGS}, got "
            f"{len(LEGACY_SCALAR_TAG_MAP)}"
        )
    if len(LEGACY_TEXT_TAG_MAP) != EXPECTED_LEGACY_TEXT_TAGS:
        raise RuntimeError(
            "MiniCPM TensorBoard text mapping is not total: "
            f"expected {EXPECTED_LEGACY_TEXT_TAGS}, got "
            f"{len(LEGACY_TEXT_TAG_MAP)}"
        )
    if set(LEGACY_SCALAR_TAG_MAP) & set(LEGACY_TEXT_TAG_MAP):
        raise RuntimeError("scalar and text legacy tag maps overlap")
    destinations = tuple(LEGACY_TAG_MAP.values())
    if len(destinations) != len(set(destinations)):
        duplicates = sorted(
            tag for tag, count in Counter(destinations).items() if count > 1
        )
        raise RuntimeError(
            f"multiple legacy TensorBoard tags map to {duplicates!r}"
        )
    validate_scalar_category_cap()


_validate_schema()

__all__ = [
    "EXPECTED_LEGACY_SCALAR_TAGS",
    "EXPECTED_LEGACY_TEXT_TAGS",
    "LEGACY_SCALAR_TAG_MAP",
    "LEGACY_TAG_MAP",
    "LEGACY_TEXT_TAG_MAP",
    "MAX_SCALAR_TAGS_PER_CATEGORY",
    "TEXT_SUMMARY_SUFFIX",
    "organized_tag",
    "scalar_category_counts",
    "validate_scalar_category_cap",
]
