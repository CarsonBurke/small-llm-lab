"""Configuration and scalar-tensor integration; no model workloads."""

import math
from types import SimpleNamespace

import pytest
import torch

import postraining.train_latent_vapo as trainer
from postraining.vapo.config import build_arg_parser, validate_args
from postraining.vapo.schemas import (
    CRITIC_POST_ACTION_Q_SCHEMA,
    CRITIC_Q_SCHEMA,
    CRITIC_SCHEMA,
    POST_ACTION_BINARY_DG_SUFFIX,
    Q_ADVANTAGE_ACTOR_OBJECTIVE_SUFFIX,
    critic_schema_for,
    resume_critic_schema_compatible,
)


MODE_ARGS = [
    "--critic-post-action-q-hidden", "8", "--critic-init", "actor",
    "--reasoning-mode", "memory", "--delightful-policy-gradient",
    "--answer-fence", "--python-reward-mode", "binary",
    "--no-zero-reward-actor-freeze",
]


def test_excluded_python_source_does_not_define_binary_q_reward_task():
    args = parse("--exclude-rl-sources", "code")
    sources = [SimpleNamespace(name="code", verifier="python_mbpp"),
               SimpleNamespace(name="math", verifier="math")]
    assert trainer.mixture_python_reward_schema(args, sources) is None
    trainer.validate_actor_init_python_reward(
        {"python_reward_schema": "older_fraction_objective"}, None, carries_critic=False,
    )
    with pytest.raises(ValueError, match="Python reward objective"):
        trainer.validate_actor_init_python_reward(
            {"python_reward_schema": "older_fraction_objective"}, None, carries_critic=True,
        )


def test_active_python_source_retains_objective_guard():
    args = parse()
    sources = [SimpleNamespace(name="code", verifier="python_mbpp")]
    expected = trainer.mixture_python_reward_schema(args, sources)
    assert expected == trainer.PYTHON_REWARD_SCHEMA
    with pytest.raises(ValueError, match="Python reward objective"):
        trainer.validate_actor_init_python_reward(
            {"python_reward_schema": "older_fraction_objective"}, expected, carries_critic=False,
        )


def test_historical_mode_keeps_manifest_wide_reward_schema():
    args = parse("--exclude-rl-sources", "code")
    args.critic_post_action_q_hidden = None
    sources = [SimpleNamespace(name="code", verifier="python_mbpp")]
    assert trainer.mixture_python_reward_schema(args, sources) == trainer.PYTHON_REWARD_SCHEMA


def parse(*extra):
    parser = build_arg_parser()
    args = parser.parse_args(["--checkpoint", "c", "--output", "o", *MODE_ARGS, *extra])
    validate_args(parser, args)
    return args


def test_binary_q_contract_is_explicit_and_not_policy_centered():
    args = parse()
    assert args.actor_gae_lambda == 0
    assert args.critic_post_action_q_prior == 0.02
    assert args.q_critic_lambda == 1
    assert args.critic_action_head_learning_rate == args.critic_learning_rate
    assert not trainer.uses_q_critic(args)
    assert trainer.uses_post_action_q(args)
    schema = trainer.run_actor_objective_schema(args)
    assert POST_ACTION_BINARY_DG_SUFFIX in schema
    assert Q_ADVANTAGE_ACTOR_OBJECTIVE_SUFFIX not in schema


@pytest.mark.parametrize("extra", [
    ["--critic-q-head-hidden", "8"],
    ["--critic-action-head-hidden", "8"],
    ["--critic-init", "scratch"],
    ["--actor-gae-lambda", "0.1"],
    ["--q-critic-lambda", "0.95"],
    ["--python-reward-mode", "test-fraction"],
    ["--nearby-reward-max", "0.1"],
    ["--abstention-penalty", "1"],
    ["--source-success-actor-gate"],
    ["--zero-reward-actor-freeze"],
    ["--no-think-tokens"],
    ["--no-answer-fence"],
    ["--reasoning-mode", "latent"],
    ["--critic-post-action-q-prior", "0"],
    ["--critic-post-action-q-prior", "nan"],
])
def test_incompatible_estimators_or_reward_contracts_are_rejected(extra):
    with pytest.raises(SystemExit):
        parse(*extra)


def test_old_critic_contracts_are_rejected_in_both_directions():
    assert critic_schema_for(True, post_action_q=True) == CRITIC_POST_ACTION_Q_SCHEMA
    for old_schema in (CRITIC_Q_SCHEMA, CRITIC_SCHEMA):
        assert not resume_critic_schema_compatible(
            {"critic_schema": old_schema}, True, post_action_q=True
        )
    payload = {"critic_schema": CRITIC_POST_ACTION_Q_SCHEMA}
    assert resume_critic_schema_compatible(payload, True, post_action_q=True)
    assert not resume_critic_schema_compatible(payload, True)
    assert not resume_critic_schema_compatible(payload)


class LogitStub(torch.nn.Module):
    """Stored scalar logits only; never invokes a learned network."""

    def __init__(self, shape):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(shape))
        self.post_action_q_head = torch.nn.Identity()
        self.action_head = None
        self.q_head = None
        self.values_actions = True

    def action_success_logits(self, batch):
        return self.logits


CONTRACT = dict(
    think_fence_ids=(10, 11), answer_fence_ids=(12, 13),
    eos_id=1, stop_ids=(1, 2), min_think_tokens=2,
)


def batch_stub():
    # Early think close is irreversibly invalid after only one think token.
    return SimpleNamespace(
        token_ids=torch.tensor([[99, 10, 42, 11, 1]]),
        action_mask=torch.tensor([[1., 1., 1., 1., 0.]]),
        emit_mask=torch.tensor([[1., 1., 1., 1., 0.]]),
        reward_scalar=torch.tensor([0.]),
        rewards=torch.zeros(1, 5), old_values=torch.full((1, 5), 123.),
        source_id=None, kind=torch.zeros(1, 5, dtype=torch.long),
    )


def test_bce_raw_calibration_is_distinct_from_exact_actor_override():
    batch = batch_stub()
    critic = LogitStub(batch.action_mask.shape)
    raw, actor_q, errors = trainer.post_action_q_loss_and_predictions(
        critic, batch, torch.zeros_like(batch.action_mask), CONTRACT
    )
    assert torch.equal(raw, torch.full_like(raw, 0.5))
    assert torch.equal(actor_q, torch.tensor([[0.5, 0.5, 0., 0., 0.5]]))
    assert not actor_q.requires_grad
    loss = (errors * batch.action_mask).sum() / batch.action_mask.sum()
    assert loss.item() == pytest.approx(math.log(2))
    loss.backward()
    assert torch.equal(critic.logits.grad, torch.tensor([[0.125, 0.125, 0.125, 0.125, 0.]]))


def test_full_critic_warmup_branch_uses_bce_raw_probability_metrics(monkeypatch):
    batch = batch_stub()
    critic = LogitStub(batch.action_mask.shape)
    wrapper = SimpleNamespace(backbone=None, hidden_carry=False)
    plan = SimpleNamespace(validate_settings=lambda *args, **kwargs: None)
    shard = SimpleNamespace(rows=torch.tensor([0]), stream_length=5)
    monkeypatch.setattr(trainer, "validate_carry_policy", lambda *args: None)
    monkeypatch.setattr(trainer, "iter_planned_replay_microbatches", lambda *args: [(batch, shard)])
    metrics = trainer.update_minibatch(
        wrapper, critic, batch, {}, value_only=True, actor_step=False,
        critic_step=False, replay_plan=plan, post_action_q_contract=CONTRACT,
    )
    assert metrics["binary_success_bce"] == pytest.approx(math.log(2))
    assert metrics["binary_success_brier"] == pytest.approx(0.25)
    assert metrics["value_mean"] == pytest.approx(0.5)
    assert metrics["value_residual_mean"] == pytest.approx(-0.5)
    assert torch.equal(critic.logits.grad, torch.tensor([[0.125, 0.125, 0.125, 0.125, 0.]]))


@pytest.mark.parametrize("terminal_reward", [0., 1.])
@pytest.mark.parametrize("bad_score", [None, 1e-6, .1, float("nan"), float("inf"), -float("inf")])
def test_full_fused_actor_branch_uses_current_q_and_one_binary_gate(monkeypatch, terminal_reward, bad_score):
    # Only stored scalar parameters and scalar-tensor replay stubs; no model.
    tokens = torch.tensor([[99, 10, 42, 43, 11, 12, 50, 13, 1]])
    mask = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1., 0.]])
    batch = SimpleNamespace(
        token_ids=tokens, action_mask=mask, emit_mask=mask,
        reward_scalar=torch.tensor([terminal_reward]),
        rewards=torch.zeros_like(mask), old_values=torch.full_like(mask, 123.),
        old_token_logprobs=torch.zeros_like(mask),
        source_id=None, kind=torch.zeros_like(tokens), statistics_refreshed=False,
        tpo_statistics_refreshed=False,
    )
    batch.rewards[0, 7] = terminal_reward
    critic = LogitStub(mask.shape)
    backbone = torch.nn.Module()
    backbone.register_parameter("logp", torch.nn.Parameter(torch.full((8,), -math.log(2))))
    wrapper = SimpleNamespace(
        backbone=backbone, hidden_carry=True, memory_carry=False,
        combiner=torch.nn.Identity(), embed_tokens=lambda ids: torch.ones(ids.numel(), 1),
    )
    plan = SimpleNamespace(validate_settings=lambda *args, **kwargs: None)
    shard = SimpleNamespace(rows=torch.tensor([0]), stream_length=9, emit_index=torch.arange(8))
    monkeypatch.setattr(trainer, "validate_carry_policy", lambda *args: None)
    monkeypatch.setattr(trainer, "iter_planned_replay_microbatches", lambda *args: [(batch, shard)])
    monkeypatch.setattr(trainer, "replay_head_inputs", lambda *args: (torch.zeros(1, 9, 1), torch.ones(1, 9, 1)))
    monkeypatch.setattr(trainer, "compact_emit_token_logprobs", lambda *args: backbone.logp)
    monkeypatch.setattr(trainer, "non_trunk_parameter_ids", lambda *args: set())
    monkeypatch.setattr(trainer, "renderer_parameters", lambda *args: [])
    def update(optimizers, step):
        return trainer.update_minibatch(
            wrapper, critic, batch, optimizers, actor_step=step, critic_step=step,
            replay_plan=plan, fused_behavior_statistics=True,
            delightful_policy_gradient=True, post_action_q_contract=CONTRACT,
            actor_gae_lambda=0, critic_lambda=1,
        )
    if bad_score is not None:
        with torch.no_grad():
            backbone.logp[0] = bad_score
        def forbidden_step():
            pytest.fail("invalid scores reached an optimizer step")
        optimizer = SimpleNamespace(step=forbidden_step, zero_grad=lambda **kwargs: None)
        with pytest.raises(RuntimeError, match="binary_q_input_domain"):
            update({"actor": optimizer, "critic": optimizer}, True)
        return
    metrics = update({}, False)
    # pi=.5 ⇒ binary gate=2/3, detached Q=.5 at every live prefix.
    # A failed sampled continuation must not zero its preceding actor credit.
    expected = torch.full((8,), -0.5 * (2 / 3) / 8)
    expected[-1] = -terminal_reward * (2 / 3) / 8
    assert torch.allclose(backbone.logp.grad, expected)
    assert torch.equal(batch.old_values, torch.full_like(mask, 0.5))
    assert metrics["actor_success_probability_mean"] == pytest.approx((7 * 0.5 + terminal_reward) / 8)
    assert metrics["binary_dg_coefficient_mean"] == pytest.approx((7 * 0.5 + terminal_reward) * (2 / 3) / 8)
    assert metrics["binary_success_bce"] == pytest.approx(math.log(2))
    assert metrics["binary_success_brier"] == pytest.approx(0.25)
    assert not any(key.startswith("credit/") for key in metrics)
    assert torch.allclose(critic.logits.grad[:, :8], torch.full((1, 8), (0.5 - terminal_reward) / 8))
