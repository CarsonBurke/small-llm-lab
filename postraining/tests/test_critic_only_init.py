"""Critic transfer preserves value-task contracts without inheriting DG actors."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

import torch

from postraining.latent_thought import LatentThoughtModel
from postraining.nano_backbone import NanoGPTBackbone
from postraining.train_latent_vapo import (
    ANSWER_FENCE_PROMPT_SCHEMA, CRITIC_SCHEMA, REWARD_SCHEMA,
    build_optimizers, carry_init_critic, reward_schema_for_abstention,
    resolve_critic_init_provenance, validate_carried_critic,
    validate_critic_only_init,
)
from postraining.value_model import SeparateCritic
from postraining.vapo.config import build_arg_parser, validate_args
from postraining.vapo.schemas import optimizer_schema_for_trunk_optimizer


def contract():
    args = SimpleNamespace(
        value_warmup_steps=0, base_checkpoint_sha256="base", python_reward_schema="partial/v2",
        reasoning_mode="cot", think_tokens=True, answer_fence=True,
        think_min_tokens=1, nearby_reward_max=0.1, abstention_penalty=None,
        combined_mlp_hidden=64,
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


def test_signed_abstention_critic_cannot_initialize_normalized_reward_run():
    payload, args = contract()
    args.abstention_penalty = 1.0
    payload["args"]["abstention_penalty"] = 1.0
    assert reward_schema_for_abstention(None) == REWARD_SCHEMA
    assert reward_schema_for_abstention(1.0) != REWARD_SCHEMA
    with pytest.raises(ValueError, match="reward_schema"):
        validate_critic_only_init(payload, args, "corpus")
    payload["reward_schema"] = reward_schema_for_abstention(1.0)
    validate_critic_only_init(payload, args, "corpus")


@pytest.mark.parametrize("key", ["base_checkpoint_sha256", "python_reward_schema",
    "critic_schema", "reward_schema", "math_data_identity", "answer_fence_prompt_schema"])
def test_critic_transfer_rejects_changed_value_task(key):
    payload, args = contract()
    payload[key] = "different"
    with pytest.raises(ValueError):
        validate_critic_only_init(payload, args, "corpus")


@pytest.mark.parametrize("key", ["reasoning_mode", "think_tokens", "answer_fence", "abstention_penalty",
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
        critic_only_init_provenance=record, critic_init="scratch")
    assert resolve_critic_init_provenance(args, None, None) == record
    args.resume = "later.pt"
    args.critic_only_init = None
    assert resolve_critic_init_provenance(args, None, {"critic": record}) == record


def branch(**source_overrides):
    """An --actor-init branch from a mid-run checkpoint of the same run."""
    payload, args = contract()
    args.trunk_optimizer = "muon"
    args.temperature, args.top_p, args.answer_tokens = 1.0, 1.0, 24
    payload["optimizer_schema"] = optimizer_schema_for_trunk_optimizer("muon")
    payload.update(step=166, value_warmup_step=50)
    payload["args"].update({
        "temperature": 1.0, "top_p": 1.0, "answer_tokens": 24,
        "value_warmup_steps": 50, **source_overrides,
    })
    return payload, args


def test_branch_carries_an_unchanged_critic_without_a_refit():
    payload, args = branch()
    original = deepcopy(payload)
    assert validate_carried_critic(payload, args) == []
    assert payload == original
    # The actor objective is not part of the critic's value task.
    args.delightful_policy_gradient = True
    assert validate_carried_critic(payload, args) == []


def test_branch_that_changes_the_reward_must_refit_the_carried_critic():
    # The step-166 abstention branch: the critic fit the penalized reward.
    payload, args = branch(abstention_penalty=1.0)
    payload["reward_schema"] = reward_schema_for_abstention(1.0)
    with pytest.raises(ValueError, match="reward_schema, abstention_penalty"):
        validate_carried_critic(payload, args)
    args.value_warmup_steps = 50
    assert validate_carried_critic(payload, args) == [
        "reward_schema", "abstention_penalty"
    ]
    payload, args = branch(nearby_reward_max=0.0)
    with pytest.raises(ValueError, match="nearby_reward_max changed"):
        validate_carried_critic(payload, args)


def test_a_changed_value_contract_can_be_accepted_without_a_refit():
    # r2's warm critic under a reward that only differs on abstentions,
    # which its warmup pools essentially never sampled.
    payload, args = branch(abstention_penalty=1.0)
    payload["reward_schema"] = reward_schema_for_abstention(1.0)
    args.accept_critic_value_change = True
    assert validate_carried_critic(payload, args) == [
        "reward_schema", "abstention_penalty"
    ]


@pytest.mark.parametrize("name, value", [
    ("temperature", 0.8), ("top_p", 0.95), ("answer_tokens", 16),
    ("continuation_tokens", 1024), ("resolved_train_max_stream_steps", 1024),
])
def test_branch_that_changes_the_policy_or_horizon_must_refit(name, value):
    payload, args = branch(**{name: value})
    with pytest.raises(ValueError, match=f"{name} changed"):
        validate_carried_critic(payload, args)
    args.value_warmup_steps = 10
    assert validate_carried_critic(payload, args) == [name]


def test_a_field_an_older_checkpoint_never_recorded_reads_as_its_default():
    payload, args = branch()
    del payload["args"]["temperature"]
    assert validate_carried_critic(payload, args) == []
    args.temperature = 0.7
    with pytest.raises(ValueError, match="temperature changed"):
        validate_carried_critic(payload, args)


def test_branch_refuses_a_critic_it_cannot_carry():
    for edit, message in (
        (lambda payload: payload.pop("critic"), "no critic"),
        (lambda payload: payload.update(critic_schema="hl_gauss"), "critic schema"),
        (lambda payload: payload.update(optimizer_schema="other"), "optimizer schema"),
    ):
        payload, args = branch()
        edit(payload)
        with pytest.raises(ValueError, match=message):
            validate_carried_critic(payload, args)


def test_a_critic_stopped_mid_warmup_is_carried_only_into_a_warmup():
    payload, args = branch()
    payload.update(step=0, value_warmup_step=20)
    with pytest.raises(ValueError, match="before its warmup finished"):
        validate_carried_critic(payload, args)
    args.value_warmup_steps = 30
    assert validate_carried_critic(payload, args) == []
    # A finished warmup checkpoint carries as-is.
    payload["value_warmup_step"] = 50
    args.value_warmup_steps = 0
    assert validate_carried_critic(payload, args) == []
    # A step-0 critic that never warmed is fit only if its own run loaded
    # it trained; a run that started from scratch has nothing to carry.
    payload["value_warmup_step"] = 0
    payload["args"]["value_warmup_steps"] = 0
    with pytest.raises(ValueError, match="before its warmup finished"):
        validate_carried_critic(payload, args)
    for loaded in ({"critic_only_init": "c.pt"}, {"actor_init": "a.pt"}):
        payload["args"].update(loaded)
        assert validate_carried_critic(payload, args) == []
    payload["args"]["fresh_critic"] = True
    payload["args"].pop("critic_only_init")
    with pytest.raises(ValueError, match="before its warmup finished"):
        validate_carried_critic(payload, args)


def _actor_and_critic(seed):
    torch.manual_seed(seed)
    kwargs = dict(vocab_size=64, num_layers=2, model_dim=64)
    wrapper = LatentThoughtModel(NanoGPTBackbone(**kwargs).float())
    critic = SeparateCritic(NanoGPTBackbone(**kwargs).float())
    return wrapper, critic


@pytest.mark.parametrize("trunk_optimizer", ["adamw", "muon"])
def test_carry_init_critic_loads_critic_state_and_leaves_the_actor_fresh(
    trunk_optimizer,
):
    args = SimpleNamespace(
        trunk_optimizer=trunk_optimizer, learning_rate=2e-5,
        critic_learning_rate=3e-5, muon_learning_rate=4e-6,
        critic_muon_learning_rate=5e-6,
    )

    def optimizers_for(wrapper, critic, critic_rate):
        return build_optimizers(
            wrapper, critic, learning_rate=args.learning_rate,
            critic_learning_rate=critic_rate, trunk_optimizer=trunk_optimizer,
            muon_learning_rate=args.muon_learning_rate,
            critic_muon_learning_rate=args.critic_muon_learning_rate,
            fused=False,
        )

    # The source run stepped both actor and critic under other rates.
    source_wrapper, source_critic = _actor_and_critic(1)
    source = optimizers_for(source_wrapper, source_critic, 1e-3)
    for optimizer in source.values():
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.randn_like(parameter)
        optimizer.step()
    payload = {
        "critic": source_critic.state_dict(),
        "optimizers": {name: opt.state_dict() for name, opt in source.items()},
    }

    wrapper, critic = _actor_and_critic(2)
    optimizers = optimizers_for(wrapper, critic, args.critic_learning_rate)
    carry_init_critic(critic, optimizers, payload, args)
    for name, value in source_critic.state_dict().items():
        assert torch.equal(critic.state_dict()[name], value), name
    for name, optimizer in optimizers.items():
        if name.startswith("critic"):
            loaded = optimizer.state_dict()
            assert loaded["state"], name
            for index, state in payload["optimizers"][name]["state"].items():
                for key, value in state.items():
                    assert torch.equal(
                        torch.as_tensor(loaded["state"][index][key]),
                        torch.as_tensor(value),
                    ), (name, key)
            rate = (
                args.critic_learning_rate if name == "critic"
                else args.critic_muon_learning_rate
            )
            assert {group["lr"] for group in optimizer.param_groups} == {rate}
        else:
            assert not optimizer.state_dict()["state"], name

    # A layout from the other trunk optimizer is refused by name.
    other = "adamw" if trunk_optimizer == "muon" else "muon"
    mismatched = dict(payload, optimizers={
        name: state for name, state in payload["optimizers"].items()
        if name != "critic_muon"
    } if trunk_optimizer == "muon" else {
        **payload["optimizers"], "critic_muon": {"state": {}, "param_groups": []},
    })
    wrapper, critic = _actor_and_critic(2)
    with pytest.raises(ValueError, match="critic optimizer layout"):
        carry_init_critic(
            critic, optimizers_for(wrapper, critic, 3e-5), mismatched,
            SimpleNamespace(**{**vars(args), "trunk_optimizer": other}),
        )


def test_accepting_a_value_change_needs_a_carried_critic(capsys):
    parser = build_arg_parser()
    for flags in ((), ("--actor-init", "a.pt", "--fresh-critic"),
                  ("--critic-only-init", "c.pt")):
        args = parser.parse_args(["--checkpoint", "c", "--output", "o",
                                  "--accept-critic-value-change", *flags])
        with pytest.raises(SystemExit):
            validate_args(parser, args)
        assert "applies only to a critic carried" in capsys.readouterr().err
    args = parser.parse_args(["--checkpoint", "c", "--output", "o",
                              "--accept-critic-value-change", "--actor-init", "a.pt"])
    validate_args(parser, args)
    assert args.value_warmup_steps == 0


@pytest.mark.parametrize("flags, warmup", [
    ((), 50),
    (("--actor-init", "a.pt"), 0),
    (("--actor-init", "a.pt", "--fresh-critic"), 50),
    (("--critic-only-init", "c.pt"), 0),
    (("--actor-init", "a.pt", "--value-warmup-steps", "20"), 20),
])
def test_value_warmup_defaults_to_zero_only_for_a_loaded_critic(flags, warmup):
    parser = build_arg_parser()
    args = parser.parse_args(["--checkpoint", "c", "--output", "o", *flags])
    validate_args(parser, args)
    assert args.value_warmup_steps == warmup
