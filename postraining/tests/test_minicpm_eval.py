from __future__ import annotations

from copy import deepcopy

import pytest

from postraining.minicpm_eval import load_evaluation_policy, resolve_evaluation_config


def _checkpoint(*, carry: bool = True, slot: bool = False) -> dict:
    # State values are deliberately opaque: dry-run validation must not execute
    # model code or inspect/copy the checkpoint's tensors.
    actor = {
        "model_id": "openbmb/MiniCPM5-1B",
        "revision": "pinned-revision",
        "lora_config": {
            "rank": 16, "alpha": 32.0, "targets": ("q_proj",),
            "initialization": "nora",
        },
        "lora_modules": ["model.layers.0.self_attn.q_proj"],
        "adapter": {
            "model.layers.0.self_attn.q_proj.lora_a": None,
            "model.layers.0.self_attn.q_proj.lora_b": None,
        },
        "nextlat_projection_factor": 1.6,
        "nextlat": {"weight": None},
    }
    if slot:
        assert carry
        actor.update(
            token_carry=True,
            token_combiner={
                name: None for name in (
                    "token_delta.weight", "query_token.weight", "query_carry.weight",
                    "key.weight", "value.weight", "output.weight", "null_key", "scale",
                )
            },
            slot_memory={"slots": 64, "heads": 1, "head_dim": 128, "rope_base": 10000.0},
            slot_head={"projection.weight": None, "projection.bias": None},
        )
    elif carry:
        actor.update(token_carry=True, token_combiner={
            "token_delta.weight": None, "carry.weight": None, "scale": None,
        })
    schema = "minicpm5_vapo_adapter/v6"
    if slot:
        schema = "minicpm5_vapo_slot_memory/v1"
    elif carry:
        schema = "minicpm5_vapo_token_carry/v4"
    return {
        "policy": {"schema": schema, "actor": actor},
        "args": {
            "model": actor["model_id"], "revision": actor["revision"],
            "thinking": True, "prompt_tokens": 1024, "max_new_tokens": 10000,
            "samples_per_prompt": 16, "prompts_per_rollout": 4,
            "rollout_physical_batch_size": 8, "temperature": 0.9,
            "top_k": 20, "top_p": 0.95, "prompt_suffix": "Use boxed answers.",
            "answer_reserve_tokens": 1000, "seed": 42,
            "token_carry": carry, "latent_thinking": False, "uno_rollout": False,
            "slot_memory": slot, "slot_memory_slots": 64, "slot_memory_heads": 1,
            "slot_memory_head_dim": 128,
            "lora_rank": 16, "lora_alpha": 32.0, "lora_initialization": "nora",
            "nextlat_projection_factor": 1.6,
        },
    }


@pytest.mark.parametrize("carry", [False, True])
def test_inherited_protocol_and_explicit_overrides_do_not_mutate_checkpoint(carry) -> None:
    checkpoint = _checkpoint(carry=carry)
    original = deepcopy(checkpoint)
    inherited = resolve_evaluation_config(checkpoint, {})
    assert inherited == {name: checkpoint["args"].get(name) for name in inherited}
    assert resolve_evaluation_config(checkpoint, dict.fromkeys(inherited)) == inherited
    overrides = {
        "thinking": False, "answer_reserve_tokens": 0, "prompt_suffix": "",
        "max_new_tokens": 512, "temperature": 0.7, "top_k": -1, "top_p": 1,
        "samples_per_prompt": 2, "prompts_per_rollout": 3,
        "rollout_physical_batch_size": 0, "seed": 0,
    }
    assert resolve_evaluation_config(checkpoint, overrides) == inherited | overrides
    assert checkpoint == original


@pytest.mark.parametrize("schema", [
    "minicpm5_vapo_token_carry/v1", "minicpm5_vapo_token_carry/v3",
    "minicpm5_vapo_adapter/v4", "minicpm5_vapo_latent/v1", None,
])
def test_unsupported_schema_cannot_be_treated_as_native(schema) -> None:
    checkpoint = _checkpoint()
    checkpoint["policy"]["schema"] = schema
    with pytest.raises(ValueError, match="unsupported.*schema"):
        resolve_evaluation_config(checkpoint, {})


@pytest.mark.parametrize("location,field,value", [
    ("args", "model", "different-model"),
    ("args", "revision", "different-revision"),
    ("actor", "revision", ""),
    ("args", "token_carry", False),
    ("actor", "token_carry", False),
    ("actor", "token_carry", 1),
    ("args", "latent_thinking", True),
    ("actor", "latent_thinking", True),
    ("args", "uno_rollout", True),
    ("args", "uno_checkpoint", "uno.pt"),
    ("actor", "transition", {}),
    ("actor", "token_combiner", {"old_projection.weight": None}),
    ("args", "lora_rank", 8),
    ("args", "lora_initialization", "standard"),
    ("args", "nextlat_projection_factor", 2.0),
    ("actor", "adapter", {}),
])
def test_actor_metadata_and_saved_training_mode_must_agree(location, field, value) -> None:
    checkpoint = _checkpoint()
    source = checkpoint["args"] if location == "args" else checkpoint["policy"]["actor"]
    source[field] = value
    with pytest.raises(ValueError):
        resolve_evaluation_config(checkpoint, {})


def test_native_schema_rejects_carry_state_even_when_mode_is_disabled() -> None:
    checkpoint = _checkpoint(carry=False)
    checkpoint["policy"]["actor"]["token_combiner"] = {}
    with pytest.raises(ValueError, match="native checkpoint contains"):
        resolve_evaluation_config(checkpoint, {})


def test_missing_protocol_cannot_be_replaced_with_new_defaults_or_overrides() -> None:
    checkpoint = _checkpoint()
    del checkpoint["args"]["answer_reserve_tokens"]
    with pytest.raises(ValueError, match="missing evaluation protocol"):
        resolve_evaluation_config(checkpoint, {"answer_reserve_tokens": 0})


@pytest.mark.parametrize("overrides", [
    {"unknown": None}, {"model": "other-model"}, {"revision": "other-revision"},
])
def test_unknown_or_base_identity_overrides_are_rejected(overrides) -> None:
    with pytest.raises(ValueError):
        resolve_evaluation_config(_checkpoint(), overrides)


@pytest.mark.parametrize("overrides", [
    {"temperature": float("nan")}, {"temperature": float("inf")}, {"temperature": 0},
    {"top_p": 0}, {"top_p": 1.01}, {"top_p": float("nan")},
    {"top_k": 0}, {"top_k": -2}, {"top_k": 130561}, {"top_k": 2.5},
    {"top_k": -1, "top_p": 0.95},
    {"prompt_tokens": 0}, {"max_new_tokens": -1}, {"samples_per_prompt": True},
    {"prompts_per_rollout": 1.5}, {"samples_per_prompt": 0},
    {"rollout_physical_batch_size": -1}, {"rollout_physical_batch_size": 65},
    {"answer_reserve_tokens": -1}, {"answer_reserve_tokens": 9999},
    {"thinking": False}, {"thinking": "true"}, {"prompt_suffix": 1},
    {"seed": 2**64}, {"seed": -(2**63) - 1}, {"seed": False},
])
def test_malformed_sampling_and_budgets_are_rejected(overrides) -> None:
    with pytest.raises(ValueError):
        resolve_evaluation_config(_checkpoint(), overrides)
    # The same validation applies to inherited metadata, not just CLI values.
    checkpoint = _checkpoint()
    checkpoint["args"].update(overrides)
    with pytest.raises(ValueError):
        resolve_evaluation_config(checkpoint, {})


def test_answer_reserve_leaves_one_thinking_token_and_close_delimiter() -> None:
    config = resolve_evaluation_config(_checkpoint(), {"max_new_tokens": 1002})
    assert config["answer_reserve_tokens"] == 1000
    with pytest.raises(ValueError, match="room for thinking"):
        resolve_evaluation_config(_checkpoint(), {"max_new_tokens": 1001})


def test_cpu_loader_fails_before_attempting_model_loading() -> None:
    import torch

    with pytest.raises(RuntimeError, match="requires CUDA"):
        load_evaluation_policy(_checkpoint(), stock=True, device=torch.device("cpu"))


def test_total_context_inheritance_and_override_preserve_historical_actor_loading() -> None:
    checkpoint = _checkpoint()
    assert resolve_evaluation_config(checkpoint, {})["context_tokens"] is None
    checkpoint["args"]["context_tokens"] = 10000
    assert resolve_evaluation_config(checkpoint, {})["context_tokens"] == 10000
    assert resolve_evaluation_config(
        checkpoint, {"context_tokens": 12000},
    )["context_tokens"] == 12000
    with pytest.raises(ValueError, match="context_tokens"):
        resolve_evaluation_config(checkpoint, {"context_tokens": 1002})


def test_slot_memory_schema_resolves_and_inherits_protocol() -> None:
    checkpoint = _checkpoint(slot=True)
    original = deepcopy(checkpoint)
    config = resolve_evaluation_config(checkpoint, {})
    assert config["max_new_tokens"] == 10000 and config["answer_reserve_tokens"] == 1000
    assert checkpoint == original


@pytest.mark.parametrize("location,field,value,message", [
    ("args", "slot_memory", False, "args slot_memory differs"),
    ("actor", "slot_memory", None, "slot-memory state differs"),
    ("actor", "slot_head", {"projection.weight": None}, "slot_head does not match"),
    ("args", "slot_memory_slots", 32, "slots differs from saved args"),
    ("args", "slot_memory_head_dim", 64, "head_dim differs from saved args"),
    ("actor", "token_combiner", {"token_delta.weight": None, "carry.weight": None, "scale": None},
     "does not match slot memory v1"),
])
def test_slot_memory_metadata_must_agree(location, field, value, message) -> None:
    checkpoint = _checkpoint(slot=True)
    source = checkpoint["args"] if location == "args" else checkpoint["policy"]["actor"]
    if value is None:
        del source[field]
    else:
        source[field] = value
    with pytest.raises(ValueError, match=message):
        resolve_evaluation_config(checkpoint, {})


def test_plain_carry_schema_rejects_slot_state() -> None:
    checkpoint = _checkpoint(carry=True)
    checkpoint["policy"]["actor"]["slot_memory"] = {"slots": 64, "heads": 1, "head_dim": 128, "rope_base": 10000.0}
    with pytest.raises(ValueError, match="slot-memory state differs"):
        resolve_evaluation_config(checkpoint, {})
    checkpoint = _checkpoint(carry=True)
    checkpoint["args"]["slot_memory"] = True
    with pytest.raises(ValueError, match="args slot_memory differs"):
        resolve_evaluation_config(checkpoint, {})
