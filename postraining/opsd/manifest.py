"""Shared immutable-data contracts for final-answer OPSD arms."""

from __future__ import annotations

from postraining.core import math_corpus_policy_sha256
from postraining.math_prompt import ANSWER_FENCE_PROMPT_SCHEMA
from postraining.opsd.data import TEACHER_PROMPT_SCHEMA
from postraining.opsd.schemas import OPSD_PROMPT_SCHEMA


DAPO_OPSD_DATA_SCHEMA = "dapo_opsd_final_answer_privilege/v2"
DAPO_OPSD_SPLIT_SCHEMA = (
    "sha256_clean_gate_then_split_local_token_length_answer_derangement/v3"
)
MATH_MIXTURE_DATA_SCHEMA = "opsd_answer_only_math_mixture/v1"
MATH_MIXTURE_SPLIT_SCHEMA = (
    "source_stratified_clean_gate_split_local_position_derangement/v1"
)
MATH_MIXTURE_SOURCE_QUOTAS = {
    "deepmind_math": 24,
    "gsm8k": 18,
    "dapo_math_17k": 6,
}
MATH_MIXTURE_GROUPS_PER_CYCLE = sum(MATH_MIXTURE_SOURCE_QUOTAS.values())

_SPLIT_SCHEMA_BY_DATA_SCHEMA = {
    DAPO_OPSD_DATA_SCHEMA: DAPO_OPSD_SPLIT_SCHEMA,
    MATH_MIXTURE_DATA_SCHEMA: MATH_MIXTURE_SPLIT_SCHEMA,
}


def validate_final_answer_manifest(manifest: dict) -> None:
    """Reject manifests outside the audited final-answer data contracts."""
    schema = manifest.get("schema")
    if schema not in _SPLIT_SCHEMA_BY_DATA_SCHEMA:
        raise ValueError(f"unsupported OPSD final-answer manifest schema {schema!r}")
    if manifest.get("math_corpus_policy_sha256") != math_corpus_policy_sha256():
        raise ValueError(
            "OPSD manifest lacks the current reviewed math corpus policy; "
            "prepare new immutable data and start a new run"
        )
    if schema == DAPO_OPSD_DATA_SCHEMA:
        identities = [manifest.get("math_corpus_identity")]
    else:
        sources = manifest.get("sources") or {}
        identities = [
            (sources.get(source) or {}).get("math_corpus_identity")
            for source in MATH_MIXTURE_SOURCE_QUOTAS
        ]
    if any(not isinstance(identity, str) or not identity for identity in identities):
        raise ValueError(
            "OPSD manifest lacks effective source corpus identities; "
            "prepare new immutable data and start a new run"
        )
    expected_split = _SPLIT_SCHEMA_BY_DATA_SCHEMA[schema]
    if manifest.get("split_schema") != expected_split:
        raise ValueError(
            "OPSD data manifest lacks its split-local permutation contract"
        )
    if manifest.get("teacher_prompt_schema") != TEACHER_PROMPT_SCHEMA:
        raise ValueError("OPSD data manifest has a different teacher prompt")
    if manifest.get("opsd_prompt_schema") != OPSD_PROMPT_SCHEMA:
        raise ValueError("OPSD data manifest has a different OPSD prompt schema")
    if (
        manifest.get("answer_fence_prompt_schema")
        != ANSWER_FENCE_PROMPT_SCHEMA
    ):
        raise ValueError(
            "OPSD data manifest has a different answer-fence prompt schema"
        )
    if schema == MATH_MIXTURE_DATA_SCHEMA:
        quotas = manifest.get("source_quotas")
        if quotas != MATH_MIXTURE_SOURCE_QUOTAS:
            raise ValueError(
                "OPSD math mixture does not use the audited 24:18:6 quotas"
            )
        if (
            int(manifest.get("groups_per_cycle", -1))
            != MATH_MIXTURE_GROUPS_PER_CYCLE
        ):
            raise ValueError("OPSD math mixture quotas do not match groups_per_cycle")
        for field in ("train_source_rows", "gate_source_rows"):
            rows = manifest.get(field)
            if not isinstance(rows, dict) or set(rows) != set(quotas):
                raise ValueError(f"OPSD math mixture has invalid {field}")
            if any(not isinstance(count, int) or count < 1 for count in rows.values()):
                raise ValueError(f"OPSD math mixture has nonpositive {field}")
        if sum(manifest["train_source_rows"].values()) != int(
            manifest.get("train_rows", -1)
        ):
            raise ValueError("OPSD math mixture train source counts do not add up")
        if sum(manifest["gate_source_rows"].values()) != int(
            manifest.get("gate_rows", -1)
        ):
            raise ValueError("OPSD math mixture gate source counts do not add up")
        gate_rows = int(manifest["gate_rows"])
        if any(
            manifest["gate_source_rows"][source]
            * MATH_MIXTURE_GROUPS_PER_CYCLE
            != gate_rows * quota
            for source, quota in quotas.items()
        ):
            raise ValueError("OPSD math mixture gate does not match source quotas")


def source_quotas(manifest: dict) -> dict[str, int] | None:
    """Return an immutable source schedule for mixture manifests."""
    validate_final_answer_manifest(manifest)
    if manifest.get("schema") != MATH_MIXTURE_DATA_SCHEMA:
        return None
    return {str(name): int(quota) for name, quota in manifest["source_quotas"].items()}
