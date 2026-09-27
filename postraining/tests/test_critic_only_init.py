"""Critic transfer preserves value-task contracts without inheriting DG actors."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from postraining.train_latent_vapo import (
    ANSWER_FENCE_PROMPT_SCHEMA, CRITIC_SCHEMA, REWARD_SCHEMA,
    resolve_critic_init_provenance, validate_critic_only_init,
)


def contract():
    args = SimpleNamespace(
        value_warmup_steps=0, base_checkpoint_sha256="base", python_reward_schema="partial/v2",
        reasoning_mode="cot", think_tokens=True, answer_fence=True,
        think_min_tokens=1, nearby_reward_max=0.1, combined_mlp_hidden=64,
        combined_mlp_blocks=2, prompt_tokens=256, continuation_tokens=2048,
        resolved_train_max_new_tokens=2048, resolved_train_max_stream_steps=2048,
        delightful_policy_gradient=False,
    )
    payload = dict(
        base_checkpoint_sha256="base", python_reward_schema="partial/v2",
        critic_schema=CRITIC_SCHEMA, reward_schema=REWARD_SCHEMA,
        math_data_identity="corpus", answer_fence_prompt_schema=ANSWER_FENCE_PROMPT_SCHEMA,
        value_warmup_step=50, step=12, critic={}, args=vars(args).copy(),
        actor_objective_schema="historical DG objective",
    )
    payload["args"]["delightful_policy_gradient"] = True
    return payload, args


def test_critic_transfer_accepts_different_actor_objective_without_mutating_source():
    payload, args = contract()
    original = deepcopy(payload)
    validate_critic_only_init(payload, args, "corpus")
    assert payload == original
    assert args.delightful_policy_gradient is False


@pytest.mark.parametrize("key", ["base_checkpoint_sha256", "python_reward_schema",
    "critic_schema", "reward_schema", "math_data_identity", "answer_fence_prompt_schema"])
def test_critic_transfer_rejects_changed_value_task(key):
    payload, args = contract()
    payload[key] = "different"
    with pytest.raises(ValueError):
        validate_critic_only_init(payload, args, "corpus")


@pytest.mark.parametrize("key", ["reasoning_mode", "think_tokens", "answer_fence",
    "think_min_tokens", "nearby_reward_max", "combined_mlp_hidden", "combined_mlp_blocks",
    "prompt_tokens", "continuation_tokens", "resolved_train_max_new_tokens",
    "resolved_train_max_stream_steps"])
def test_critic_transfer_rejects_geometry_fence_and_budget_mismatches(key):
    payload, args = contract()
    payload["args"][key] = "different"
    with pytest.raises(ValueError, match=key):
        validate_critic_only_init(payload, args, "corpus")


def test_critic_transfer_requires_warmed_weights_and_no_new_warmup():
    payload, args = contract()
    args.value_warmup_steps = 50
    with pytest.raises(ValueError, match="warmup-steps 0"):
        validate_critic_only_init(payload, args, "corpus")
    args.value_warmup_steps = 0
    payload["value_warmup_step"] = 0
    with pytest.raises(ValueError, match="completed critic warmup"):
        validate_critic_only_init(payload, args, "corpus")


def test_critic_transfer_provenance_survives_later_resumes():
    record = dict(init="critic_only_checkpoint", checkpoint="source.pt",
        checkpoint_sha256="hash", source_execution_schema="execution",
        source_actor_step=12, source_value_warmup_step=50, optimizer_state="fresh")
    args = SimpleNamespace(resume=None, critic_only_init="source.pt",
        critic_only_init_provenance=record, actor_critic_init=None, critic_init="scratch")
    assert resolve_critic_init_provenance(args, None, None) == record
    args.resume = "later.pt"
    args.critic_only_init = None
    assert resolve_critic_init_provenance(args, None, {"critic": record}) == record
