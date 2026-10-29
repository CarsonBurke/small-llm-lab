"""Mean Actor-Critic through the trainer's update: CPU oracles on tiny models."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from postraining.core import generalized_advantage_and_return_targets
from postraining.latent_rollout import (
    assign_terminal_rewards,
    build_replay_plan,
    compact_next_slots,
    compact_slots,
    replay_head_inputs,
    scatter_slots,
    slot_index,
)
from postraining.latent_thought import LatentThoughtModel
from postraining.value_model import SeparateCritic
from postraining.tests.test_hidden_carry import BACKBONES, _prompts, _roll
from postraining.tests.test_memory_carry import _memory_wrapper
from postraining.tests.test_q_critic import (
    HIDDEN, _expected_advantages, _oracle_action_offsets, _q_critic, _token_values,
)
from postraining.rollout_report import RowVerdict
from postraining.train_latent_vapo import (
    STUDY_CRITIC_SCHEMA,
    build_optimizers,
    graded_answer_keys,
    stamp_group_terms,
    load_study_critic,
    mac_dashboard,
    post_think_state_mask,
    reassert_learning_rates,
    run_actor_objective_schema,
    update_minibatch,
    validate_study_critic_init,
)
from postraining.mean_actor_critic import (
    DELIGHTFUL_MAC_ESTIMATORS, MAC_ESTIMATORS, SAMPLED_ONLY_ESTIMATORS, SEQUENCE_ESTIMATORS,
    gate_shuffle_strata, shuffle_within_strata,
)
from postraining.vapo.config import build_arg_parser, validate_args
from postraining.vapo.schemas import (
    CRITIC_Q_SCHEMA, MAC_ESTIMATOR_ACTOR_OBJECTIVE_SCHEMAS,
    MEAN_ACTOR_CRITIC_ACTOR_OBJECTIVE_SCHEMA, MEMORY_MAC_ESTIMATOR_ACTOR_OBJECTIVE_SCHEMAS,
    critic_schema_for,
)

CLOSE = 40


def _cot_wrapper(seed: int = 3):
    factory, _ = BACKBONES["nano"]
    return LatentThoughtModel(factory(seed)).eval()


def _cot_batch(wrapper, seed: int = 5):
    batch = _roll(wrapper, _prompts("nano", 4, 5, seed=seed), seed=seed,
                  hidden_carry=False)
    assign_terminal_rewards(batch, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    return batch


# The scorer's verdicts for ``_cot_batch``'s rewards [1, 0, 1, 0]: two rows
# agree on the right answer, one is wrong, and one was gate-zeroed (its
# counterfactual "7" earned nothing, so it is no cluster member of "7").
VERDICTS = (RowVerdict(True, "7"), RowVerdict(True, "3"), RowVerdict(True, "7"),
            RowVerdict(False, "7"))


def _stamped(batch, verdicts=VERDICTS):
    stamp_group_terms(batch, verdicts)
    return batch


def _sequence_rows(batch, estimator):
    """Each row's coefficient, written out from the stamped group terms."""
    advantages = batch.group_advantage.double()
    if estimator == "rloo_sequence":
        return advantages
    return advantages * torch.sigmoid(advantages * batch.answer_surprisal.double())


def _mac(wrapper, critic, batch, optimizers, **kwargs):
    kwargs.setdefault("think_close_id", CLOSE)
    kwargs.setdefault("mac_estimator", "all_action")
    return update_minibatch(
        wrapper, critic, batch, optimizers, mean_actor_critic=True,
        fused_behavior_statistics=True, critic_lambda=1.0,
        actor_step=False, critic_step=False, **kwargs,
    )


def _actor_parameters(wrapper):
    return [p for p in wrapper.parameters() if p.requires_grad]


def _oracle_actor_gradients(wrapper, critic, batch, estimator="all_action", gate_seed=None,
                            plan=None):
    """-mean over emit states of the estimator's loss, by autograd through the
    renderer's own logits: sum_a pi(a) stopgrad[Q(a) - E_pi Q], plus or
    instead the sampled log-probability times stopgrad[G - Q(a_t)] or
    stopgrad[G - E_pi Q]. The shuffled head gate's A is the oracle's own,
    shuffled shard by shard of ``plan``, in each shard's compact order, from
    one generator seeded ``gate_seed``."""
    with torch.no_grad():
        q_values, directions = _token_values(wrapper, critic, batch)
        latents = critic.action_latents()
        _, returns = generalized_advantage_and_return_targets(
            batch.rewards, torch.zeros_like(batch.old_values), batch.action_mask,
            torch.ones(batch.action_mask.size(0)))
    beliefs, inputs = replay_head_inputs(wrapper, batch)
    emit_index = slot_index(batch.emit_mask.bool())
    logits = wrapper.backbone.logits_from_features(
        wrapper.renderer_features(compact_slots(inputs, emit_index),
                                  compact_slots(beliefs, emit_index))
    ).double()
    log_probabilities = logits.log_softmax(-1)
    probabilities = log_probabilities.exp()
    scores = compact_slots(directions, emit_index).double() @ latents.double().T
    centered = (scores - (probabilities.detach() * scores).sum(-1, keepdim=True)).detach()
    targets = compact_next_slots(batch.token_ids, emit_index)
    sampled = log_probabilities.gather(-1, targets[:, None]).squeeze(-1)
    residuals = compact_slots(returns - q_values, emit_index).double()
    # G - E_pi Q = G - Q(a_t) + A(a_t).
    head_advantages = centered.gather(-1, targets[:, None]).squeeze(-1)
    baseline_residuals = residuals + head_advantages
    state_values = (compact_slots(q_values, emit_index).double()
                    - centered.gather(-1, targets[:, None]).squeeze(-1))
    sequence = None
    if estimator in SEQUENCE_ESTIMATORS:
        sequence = _sequence_rows(batch, estimator)[
            torch.div(emit_index, batch.stream_length, rounding_mode="floor")]
    shuffled = None
    if gate_seed is not None:
        def dense(compact):
            out = torch.zeros_like(batch.old_values, dtype=compact.dtype)
            scatter_slots(out, emit_index, compact)
            return out

        head_dense, logprob_dense = dense(head_advantages), dense(sampled.detach())
        answer_dense = post_think_state_mask(batch.token_ids, batch.emit_mask, CLOSE)
        shuffled_dense = torch.zeros_like(head_dense)
        generator = torch.Generator().manual_seed(gate_seed)
        for shard in plan.shards():
            rows, length = shard.rows, shard.stream_length

            def pick(tensor, rows=rows, length=length, shard=shard):
                return compact_slots(tensor[rows, :length], shard.emit_index)

            local = torch.zeros_like(head_dense[rows, :length])
            scatter_slots(local, shard.emit_index, shuffle_within_strata(
                pick(head_dense), gate_shuffle_strata(pick(logprob_dense), pick(answer_dense)),
                generator))
            shuffled_dense[rows, :length] = local
        shuffled = compact_slots(shuffled_dense, emit_index)
    coefficient = {
        "all_action": torch.zeros_like(residuals),
        "control_variate": residuals,
        "sampled_baseline": baseline_residuals,
        "delightful_sampled_baseline":
            baseline_residuals * torch.sigmoid(baseline_residuals * -sampled.detach()),
        "delightful_td_gate":
            baseline_residuals * torch.sigmoid(
                _td_advantages(batch, emit_index, state_values) * -sampled.detach()),
        "delightful_head_gate":
            baseline_residuals * torch.sigmoid(head_advantages * -sampled.detach()),
        "delightful_shuffled_head_gate": None if shuffled is None else
            baseline_residuals * torch.sigmoid(shuffled * -sampled.detach()),
        # One constant per trajectory; no critic value enters it.
        "rloo_sequence": sequence,
        "delightful_answer_gate": sequence,
    }[estimator].detach()
    loss = -(coefficient * sampled).sum()
    if estimator not in SAMPLED_ONLY_ESTIMATORS:
        loss = loss - (probabilities * centered).sum()
    loss = loss / emit_index.numel()
    parameters = _actor_parameters(wrapper)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, gradients)]


def _td_advantages(batch, emit_index, values):
    """The TD residual rew_t + c(s_{t+1}) - c(s_t) at each emit state, from
    the batch's rewards and the compact state values c, with no next value
    after a row's last action (in a cot batch every action is an emit)."""
    dense = torch.zeros_like(batch.old_values, dtype=values.dtype)
    scatter_slots(dense, emit_index, values)
    following = compact_next_slots(dense * batch.action_mask.to(values.dtype), emit_index)
    return compact_slots(batch.rewards.to(values.dtype), emit_index) + following - values


def _score_rounding(wrapper, critic, batch) -> float:
    """Relative fp32 rounding of a MAC gradient entry: the trainer forms each
    coefficient from fp32 token scores, which the live fixture puts near 60
    (a real Q lies in [0, 1]), and their rounding, about eps * |score|,
    carries into every entry."""
    with torch.no_grad():
        _, directions = _token_values(wrapper, critic, batch)
        scale = float((directions.flatten(0, -2) @ critic.action_latents().T).abs().max())
    return 2 * torch.finfo(torch.float32).eps * max(1.0, scale)


def _gradients(module_parameters):
    return [torch.zeros_like(p) if p.grad is None else p.grad.clone() for p in module_parameters]


@pytest.mark.parametrize("estimator", MAC_ESTIMATORS)
def test_mac_actor_gradient_is_the_estimators_expectation(estimator):
    wrapper = _cot_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    batch = _stamped(_cot_batch(wrapper))
    if estimator == "delightful_td_gate":
        # A reward before the last action: the trainer's G_t - G_{t+1}
        # difference must still be the TD residual rew_t + c_{t+1} - c_t.
        batch.rewards[0, int(batch.action_mask[0].nonzero()[1])] = 0.25
    expected = _oracle_actor_gradients(wrapper, critic, copy.deepcopy(batch), estimator)
    wrapper.zero_grad(set_to_none=True)
    metrics = _mac(wrapper, critic, batch, optimizers, mac_vocab_chunk=3,
                   mac_estimator=estimator)
    actual = _gradients(_actor_parameters(wrapper))
    assert any(bool(g.abs().sum()) for g in expected)
    rounding = _score_rounding(wrapper, critic, batch)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.double(), want.double(), rtol=1e-4,
                                   atol=max(1e-6, rounding * float(want.abs().max())))
    # Behavior statistics and the reported advantages come from the same pass:
    # the critic's centered Q of the taken token, or for the sampled baseline
    # the coefficient its policy gradient weights, G - E_pi Q.
    sampled = head = _expected_advantages(wrapper, critic, batch)
    emit = batch.emit_mask.bool()
    if estimator in SEQUENCE_ESTIMATORS:
        # The trajectory's coefficient on each of its tokens.
        sampled = _sequence_rows(batch, estimator)[:, None].expand_as(sampled)
    elif estimator in SAMPLED_ONLY_ESTIMATORS:
        with torch.no_grad():
            q_values, _ = _token_values(wrapper, critic, batch)
            _, returns = generalized_advantage_and_return_targets(
                batch.rewards, torch.zeros_like(batch.old_values), batch.action_mask,
                torch.ones(batch.action_mask.size(0)))
        sampled = returns - q_values + sampled
        if estimator in DELIGHTFUL_MAC_ESTIMATORS:
            # The gate at the log-probs the same pass recorded.
            delight = sampled
            critic_gated = estimator in ("delightful_td_gate", "delightful_head_gate")
            if critic_gated:
                emit_index = slot_index(emit)
                if estimator == "delightful_td_gate":
                    source = "gate"
                    compact_delight = _td_advantages(
                        batch, emit_index, compact_slots(returns - sampled, emit_index))
                else:
                    # The head's own A(a_t), with no next-state value.
                    source = "head"
                    compact_delight = compact_slots(head, emit_index)
                delight = torch.zeros_like(sampled)
                scatter_slots(delight, emit_index, compact_delight)
                # Its moments skip each row's last action, where A = r.
                nonterminal = compact_next_slots(batch.action_mask, emit_index).double()
                residual = compact_slots(sampled, emit_index).double()
                for name, value in (
                        ("states", nonterminal),
                        ("square", compact_delight.double().square() * nonterminal),
                        ("residual", compact_delight.double() * residual * nonterminal)):
                    # The trainer's c is fp32 from scores near 60.
                    torch.testing.assert_close(
                        metrics[f"mac/{source}_advantage_{name}_sum"], float(value.sum()),
                        rtol=1e-4, atol=max(1e-6, rounding * float(value.abs().sum())))
                assert 0 < metrics[f"mac/{source}_advantage_states_sum"] < emit_index.numel()
                assert not any(key.startswith("mac/shuffle") for key in metrics)
            gate = torch.sigmoid(delight * -batch.old_token_logprobs)
            torch.testing.assert_close(
                metrics["mac/think/gate_sum"] + metrics["mac/answer/gate_sum"],
                float((gate * batch.emit_mask).sum()), rtol=1e-4, atol=1e-6)
            assert metrics["mac/gate_positive_count"] + metrics["mac/gate_negative_count"] > 0
            # The gate weighted by the term it scales, |r| (1 - pi(a_t)).
            mass = sampled.abs() * (1 - batch.old_token_logprobs.exp())
            active = (delight * batch.old_token_logprobs).abs() > 1
            for sign, side in (("positive", sampled > 0), ("negative", sampled < 0)):
                side = side & emit
                for name, value in (("mass", mass), ("weighted", gate * mass),
                                    ("active_mass", mass * active)):
                    torch.testing.assert_close(
                        metrics[f"mac/gate_{sign}_{name}_sum"],
                        float(value[side].sum()), rtol=1e-4, atol=1e-6)
            if critic_gated:
                # The same gate mass where A is the critic's alone.
                dense_nonterminal = torch.zeros_like(sampled)
                scatter_slots(dense_nonterminal, emit_index, nonterminal.to(sampled.dtype))
                for name, value in (("mass", mass), ("weighted", gate * mass),
                                    ("active_mass", mass * active)):
                    torch.testing.assert_close(
                        metrics[f"mac/gate_nonterminal_{name}_sum"],
                        float((value * dense_nonterminal).sum()), rtol=1e-4, atol=1e-6)
            sampled = sampled * gate
    torch.testing.assert_close(
        metrics["advantage_group/span_think/square_sum"]
        + metrics["advantage_group/span_answer/square_sum"],
        float((sampled.square() * batch.emit_mask).sum()), rtol=1e-4, atol=1e-8,
    )
    assert bool(batch.old_token_logprobs[emit].lt(0).all())
    assert metrics["policy_entropy_count"] == float(emit.sum())


def test_mac_critic_trains_on_the_same_pass():
    wrapper = _cot_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    _mac(wrapper, critic, _cot_batch(wrapper), optimizers)
    assert critic.q_head.direction.weight.grad.abs().sum() > 0
    assert critic.q_head.state.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("option", [{"mac_vocab_chunk": 1}, {"replay_max_trajectories": 1}])
def test_mac_gradient_does_not_depend_on_chunking_or_sharding(option):
    wrapper = _cot_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    batch = _cot_batch(wrapper)
    runs = []
    for kwargs in ({}, option):
        wrapper.zero_grad(set_to_none=True)
        critic.zero_grad(set_to_none=True)
        _mac(wrapper, critic, copy.deepcopy(batch), optimizers, **kwargs)
        runs.append(_gradients(_actor_parameters(wrapper)))
    rounding = _score_rounding(wrapper, critic, batch)
    for left, right in zip(*runs):
        torch.testing.assert_close(left, right, rtol=1e-5,
                                   atol=max(1e-7, rounding * float(left.abs().max())))


def test_mac_refuses_other_objectives_and_carried_policies():
    wrapper = _cot_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    with pytest.raises(ValueError, match="standalone actor objective"):
        _mac(wrapper, critic, _cot_batch(wrapper), optimizers,
             delightful_policy_gradient=True)
    with pytest.raises(ValueError, match="think fence"):
        _mac(wrapper, critic, _cot_batch(wrapper), optimizers, think_close_id=None)
    with pytest.raises(ValueError, match="unknown MAC estimator"):
        _mac(wrapper, critic, _cot_batch(wrapper), optimizers, mac_estimator=None)


def test_post_think_states_start_at_the_slot_holding_the_fence():
    # Prompt [7, 40] holds a fence that is not the completion's.
    tokens = torch.tensor([[7, 40, 3, 40, 5, 6, 0]])
    emit = torch.tensor([[0, 1, 1, 1, 1, 1, 0]])
    mask = post_think_state_mask(tokens, emit, CLOSE)
    # The slot emitting the fence (2) is a think state; slot 3 holds it.
    assert mask.tolist() == [[False, False, False, True, True, True, False]]


def _parse(*extra):
    parser = build_arg_parser()
    args = parser.parse_args(["--checkpoint", "c", "--output", "o", *extra])
    validate_args(parser, args)
    return args


MAC_ARGS = (
    "--mean-actor-critic", "--mac-estimator", "all_action",
    "--critic-q-head-hidden", "8", "--critic-init", "actor", "--reasoning-mode", "cot",
    "--think-tokens", "--answer-fence", "--no-zero-reward-actor-freeze",
    "--prompts-per-rollout", "64", "--prompts-per-minibatch", "64",
)


def test_mac_cli_selects_its_schema_and_any_q_lambda():
    args = _parse(*MAC_ARGS)
    # Length-adaptive generalized value estimates, as for every token Q
    # critic; Monte Carlo targets are --q-critic-lambda 1.
    assert args.q_critic_lambda is None
    assert _parse(*MAC_ARGS, "--q-critic-lambda", "0.5").q_critic_lambda == 0.5
    assert run_actor_objective_schema(args) == MEAN_ACTOR_CRITIC_ACTOR_OBJECTIVE_SCHEMA
    schemas = {
        estimator: run_actor_objective_schema(_parse(*MAC_ARGS, "--mac-estimator", estimator))
        for estimator in MAC_ESTIMATORS
    }
    assert schemas == MAC_ESTIMATOR_ACTOR_OBJECTIVE_SCHEMAS
    assert len(set(schemas.values())) == len(MAC_ESTIMATORS)


def test_a_memory_mac_run_records_its_own_schema():
    memory = tuple("memory" if arg == "cot" else arg for arg in MAC_ARGS)
    schemas = {
        estimator: run_actor_objective_schema(_parse(*memory, "--mac-estimator", estimator))
        for estimator in MAC_ESTIMATORS
    }
    assert schemas == MEMORY_MAC_ESTIMATOR_ACTOR_OBJECTIVE_SCHEMAS
    # A cot MAC checkpoint cannot resume as a memory run, nor the reverse.
    assert not set(schemas.values()) & set(MAC_ESTIMATOR_ACTOR_OBJECTIVE_SCHEMAS.values())
    assert len(set(schemas.values())) == len(MAC_ESTIMATORS)


def test_mac_estimator_is_required_and_only_with_mac(capsys):
    without = tuple(arg for arg in MAC_ARGS if arg not in ("--mac-estimator", "all_action"))
    with pytest.raises(SystemExit):
        _parse(*without)
    assert "needs --mac-estimator" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _parse("--mac-estimator", "control_variate")
    assert "selects a --mean-actor-critic estimator" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--delightful-policy-gradient",), "replaces the DG and TPO"),
        (("--zero-reward-actor-freeze",), "uses no sampled outcome"),
        (("--reasoning-mode", "carry"), "supports --reasoning-mode cot"),
        (("--mac-vocab-chunk", "0"), "must be positive"),
        (("--study-critic-init", "s.pt", "--resume", "r.pt"), "excludes"),
        (("--critic-q-direction-learning-rate", "-1"), "positive rate"),
        (("--bench-stop-floor", "1.5"), "[0, 1] floor"),
        (("--eval-top-p", "0"), "--eval-top-p"),
    ],
)
def test_mac_cli_refusals(extra, message, capsys):
    with pytest.raises(SystemExit):
        _parse(*MAC_ARGS, *extra)
    assert message in capsys.readouterr().err


def test_direction_readout_has_its_own_reasserted_rate():
    wrapper = _cot_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=1e-3, fused=False,
                                  critic_action_learning_rate=1e-4,
                                  critic_q_direction_learning_rate=2e-7)
    groups = optimizers["critic_action"].param_groups
    direction = [g for g in groups if g.get("token_q_direction")]
    assert len(direction) == 1 and [id(p) for p in direction[0]["params"]] == [
        id(weight) for weight in critic.q_head.direction_weights]
    assert direction[0]["lr"] == 2e-7
    rest = [p for g in groups if not g.get("token_q_direction") for p in g["params"]]
    assert rest and all(p is not critic.q_head.direction.weight for p in rest)
    for group in groups:
        group["lr"] = 1.0
    reassert_learning_rates(optimizers, SimpleNamespace(
        learning_rate=1e-3, critic_learning_rate=1e-3, muon_learning_rate=None,
        critic_muon_learning_rate=None, critic_action_head_learning_rate=1e-4,
        critic_q_direction_learning_rate=2e-7,
    ))
    assert sorted(g["lr"] for g in groups) == [2e-7, 1e-4]


def _study_payload(critic, collection=Path("unused"), **arm):
    return {
        "schema": STUDY_CRITIC_SCHEMA, "checkpoint_sha256": "base", "step": 640,
        "critic_schema": critic_schema_for(True, twin_q=bool(arm.get("q_twin", False))),
        "arm": {"context": "prompt", "critic_init": "actor",
                "direction_scale": "inverse_dim", "weighting": "token",
                "head_learning_rate": 1e-4, "trunk_learning_rate": 2e-5,
                "muon_learning_rate": 5e-5, "train_belief_gain": True, **arm},
        "config": {"q_head_hidden": 8, "train_dataset": str(collection / "rollout_dataset")},
        "critic": critic.state_dict(),
    }


def _study_args(**overrides):
    return SimpleNamespace(**{
        "base_checkpoint_sha256": "base", "critic_init": "actor", "critic_q_head_hidden": 8, "critic_q_twin": False, "critic_q_pessimism": 0.0,
        "critic_action_head_learning_rate": 1e-4, "critic_learning_rate": 2e-5,
        "critic_muon_learning_rate": 5e-5, "critic_q_direction_learning_rate": None,
        "reasoning_mode": "cot", "temperature": 1.0, "top_p": 1.0, **overrides,
    })


def test_study_critic_must_continue_its_base_and_geometry():
    critic = _q_critic(_cot_wrapper())
    rate = 1e-4 / critic.q_head.direction.out_features
    validate_study_critic_init(_study_payload(critic), critic,
                               _study_args(critic_q_direction_learning_rate=rate))
    for payload, args, message in (
        (_study_payload(critic), _study_args(), "rates"),
        (_study_payload(critic), _study_args(base_checkpoint_sha256="other",
                                             critic_q_direction_learning_rate=rate),
         "another base"),
        (_study_payload(critic, context="gold"), _study_args(critic_q_direction_learning_rate=rate),
         "prompt-context"),
        # The frozen-gain ablation arm, which this run would train past.
        (_study_payload(critic, train_belief_gain=False),
         _study_args(critic_q_direction_learning_rate=rate), "froze"),
        # A study critic without the gain (v6), fit uncentred (v5), or against the input-latent
        # action table (v4).
        ({**_study_payload(critic), "critic_schema":
          "standalone_token_bilinear_q_table_rms_readout_row_unscaled_direction_readout_"
          "actor_belief_gve_mse/v5"},
         _study_args(critic_q_direction_learning_rate=rate), "another token Q contract"),
        # A real v6 study checkpoint's arm has no gain field at all.
        ({**_study_payload(critic), "critic_schema":
          "standalone_token_bilinear_q_policy_centered_fixed_rms_readout_row_"
          "unscaled_direction_readout_actor_belief_gve_mse/v6",
          "arm": {key: value for key, value in _study_payload(critic)["arm"].items()
                  if key != "train_belief_gain"}},
         _study_args(critic_q_direction_learning_rate=rate), "another token Q contract"),
        ({**_study_payload(critic), "critic_schema":
          "standalone_token_bilinear_q_unit_rms_latent_unscaled_direction_readout_"
          "actor_belief_gve_mse/v4"},
         _study_args(critic_q_direction_learning_rate=rate), "another token Q contract"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_study_critic_init(payload, critic, args)


def test_study_critic_must_value_this_runs_prompts_and_contract(tmp_path):
    critic = _q_critic(_cot_wrapper())
    rate = 1e-4 / critic.q_head.direction.out_features
    (tmp_path / "manifest.json").write_text(json.dumps({
        "math_data_identity": "drills",
        "args": {"reasoning_mode": "cot", "temperature": 1.0, "top_p": 1.0},
    }))
    payload = _study_payload(critic, tmp_path)
    args = _study_args(critic_q_direction_learning_rate=rate)
    validate_study_critic_init(payload, critic, args, "drills")
    with pytest.raises(ValueError, match="other prompts"):
        validate_study_critic_init(payload, critic, args, "mixture")
    with pytest.raises(ValueError, match="top_p"):
        validate_study_critic_init(payload, critic, _study_args(
            critic_q_direction_learning_rate=rate, top_p=0.7), "drills")
    with pytest.raises(ValueError, match="rates"):
        validate_study_critic_init(payload, critic, _study_args(
            critic_q_direction_learning_rate=rate, critic_learning_rate=1e-5))


@pytest.mark.parametrize("shard_rows", [32, 1])
def test_mac_value_only_warmup_fits_the_critic_alone(shard_rows):
    """The warmup's critic gradient is the centred Monte Carlo regression's,
    centred under an fp64 softmax of the policy, on any sharding."""
    wrapper = _cot_wrapper()
    critic = _q_critic(wrapper)
    with torch.no_grad():
        critic.q_head.state.weight.normal_(std=0.5)
        critic.q_head.direction.weight.normal_(std=0.5)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    batch = _cot_batch(wrapper)
    mask = batch.action_mask
    _, returns = generalized_advantage_and_return_targets(
        batch.rewards, torch.zeros_like(batch.old_values), mask, torch.ones(mask.size(0)))
    values, _ = critic.token_values(batch, _oracle_action_offsets(wrapper, critic, batch))
    loss = ((values - returns).square() * mask).sum() / mask.sum()
    parameters = [p for p in critic.parameters() if p.requires_grad]
    expected = torch.autograd.grad(loss, parameters, allow_unused=True)
    metrics = update_minibatch(wrapper, critic, batch, optimizers, value_only=True,
                               mean_actor_critic=True, mac_estimator="control_variate",
                               think_close_id=CLOSE, critic_step=False,
                               replay_max_trajectories=shard_rows)
    assert metrics["value_loss"] == pytest.approx(float(loss), rel=1e-5)
    assert critic.q_head.direction.weight.grad.abs().sum() > 0
    for parameter, want in zip(parameters, expected, strict=True):
        if want is None:
            assert parameter.grad is None or not parameter.grad.any()
            continue
        torch.testing.assert_close(parameter.grad, want, rtol=1e-4,
                                   atol=1e-6 * float(want.abs().max().clamp_min(1)))
    assert all(p.grad is None for p in _actor_parameters(wrapper))


def test_mac_dashboard_reports_span_q_variance_and_sampled_coefficients():
    wrapper = _cot_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    metrics = _mac(wrapper, critic, _cot_batch(wrapper), optimizers)
    dashboard = mac_dashboard([metrics])
    assert dashboard["mac/think_states"] + dashboard["mac/answer_states"] == metrics["emit_action_count"]
    assert dashboard["mac/think_policy_q_variance"] > 0
    assert dashboard["mac/think_coefficient_rms"] == 0
    batch = _cot_batch(wrapper)
    with torch.no_grad():
        q_values, _ = _token_values(wrapper, critic, batch)
        _, returns = generalized_advantage_and_return_targets(
            batch.rewards, torch.zeros_like(batch.old_values), batch.action_mask,
            torch.ones(batch.action_mask.size(0)))
    emit = batch.emit_mask.bool()
    expected = (returns - q_values)[emit].square().mean().sqrt()
    metrics = _mac(wrapper, critic, batch, optimizers, mac_estimator="control_variate")
    dashboard = mac_dashboard([metrics])
    states = dashboard["mac/think_states"] + dashboard["mac/answer_states"]
    rms = ((dashboard["mac/think_coefficient_rms"] ** 2 * dashboard["mac/think_states"]
            + dashboard["mac/answer_coefficient_rms"] ** 2 * dashboard["mac/answer_states"])
           / states) ** 0.5
    assert rms == pytest.approx(float(expected), rel=1e-5)
    assert not any("delight" in key for key in dashboard)
    metrics = _mac(wrapper, critic, _cot_batch(wrapper), optimizers,
                   mac_estimator="delightful_sampled_baseline")
    dashboard = mac_dashboard([metrics])
    # sigmoid(r * surprisal) with surprisal >= 0 sits above 1/2 where the
    # continuation beat c and below it where it fell short.
    assert dashboard["mac/delight_gate_positive_mean"] > 0.5 > dashboard["mac/delight_gate_negative_mean"]
    assert 0 < dashboard["mac/think_delight_gate_mean"] < 1
    assert dashboard["mac/delight_gate_positive_mass_mean"] > 0.5 > dashboard[
        "mac/delight_gate_negative_mass_mean"]
    for sign in ("positive", "negative"):
        assert 0 <= dashboard[f"mac/delight_{sign}_active_mass_share"] <= 1
    assert not any("td_advantage" in key or "nonterminal" in key for key in dashboard)
    metrics = _mac(wrapper, critic, _cot_batch(wrapper), optimizers,
                   mac_estimator="delightful_td_gate")
    dashboard = mac_dashboard([metrics])
    assert dashboard["mac/td_advantage_rms"] > 0
    assert -1 <= dashboard["mac/td_advantage_residual_correlation"] <= 1
    assert math.isfinite(dashboard["mac/td_advantage_calibration_slope"])
    assert 0 < dashboard["mac/delight_gate_nonterminal_mass_mean"] < 1
    assert 0 <= dashboard["mac/delight_nonterminal_active_mass_share"] <= 1


def test_the_shuffled_head_gate_permutes_the_heads_advantages_within_strata():
    """The control gates on the head's A(a_t), permuted within surprisal-and-
    span strata by one generator from the call's seed: the oracle shuffles
    its own A the same way. The same seed replays the gradient exactly, and
    another seed moves it (the gate is not vacuous)."""
    wrapper = _cot_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    batch = _cot_batch(wrapper)
    estimator = "delightful_shuffled_head_gate"
    # Two shards of two rows, the plan the update forms itself: one
    # generator stream runs across both.
    plan = build_replay_plan(batch, 2, 4 * 1024 * 1024)
    assert len(list(plan.shards())) == 2
    expected = _oracle_actor_gradients(wrapper, critic, copy.deepcopy(batch), estimator,
                                       gate_seed=17, plan=plan)
    rounding = _score_rounding(wrapper, critic, batch)
    runs = []
    for seed in (17, 17, 18):
        wrapper.zero_grad(set_to_none=True)
        critic.zero_grad(set_to_none=True)
        metrics = _mac(wrapper, critic, copy.deepcopy(batch), optimizers, mac_vocab_chunk=3,
                       mac_estimator=estimator, mac_gate_shuffle_seed=seed,
                       replay_max_trajectories=2)
        runs.append((metrics, _gradients(_actor_parameters(wrapper))))
    for got, want in zip(runs[0][1], expected):
        torch.testing.assert_close(got.double(), want.double(), rtol=1e-4,
                                   atol=max(1e-6, rounding * float(want.abs().max())))
    assert all(torch.equal(a, b) for a, b in zip(runs[0][1], runs[1][1]))
    assert not all(torch.equal(a, b) for a, b in zip(runs[0][1], runs[2][1]))
    metrics = runs[0][0]
    # Both A's moments, over the same non-terminal states.
    for source in ("head", "shuffled"):
        assert metrics[f"mac/{source}_advantage_square_sum"] > 0
    assert metrics["mac/head_advantage_states_sum"] == metrics["mac/shuffled_advantage_states_sum"]
    dashboard = mac_dashboard([metrics])
    for key in ("mac/head_advantage_calibration_slope",
                "mac/shuffled_head_advantage_calibration_slope"):
        assert math.isfinite(dashboard[key])
    assert -1 <= dashboard["mac/shuffle_head_correlation"] <= 1
    assert 0 <= dashboard["mac/shuffle_fixed_mass_share"] < 1
    assert "mac/td_advantage_rms" not in dashboard


def test_the_gate_shuffle_seed_comes_with_and_only_with_the_shuffled_gate():
    wrapper = _cot_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    with pytest.raises(ValueError, match="gate shuffle seed"):
        _mac(wrapper, critic, _cot_batch(wrapper), optimizers,
             mac_estimator="delightful_shuffled_head_gate")
    with pytest.raises(ValueError, match="gate shuffle seed"):
        _mac(wrapper, critic, _cot_batch(wrapper), optimizers,
             mac_estimator="delightful_head_gate", mac_gate_shuffle_seed=1)


def test_the_shuffled_head_gate_is_no_cli_estimator():
    with pytest.raises(SystemExit):
        _parse(*MAC_ARGS, "--mac-estimator", "delightful_shuffled_head_gate")
    head = _parse(*MAC_ARGS, "--mac-estimator", "delightful_head_gate")
    assert head.mac_estimator == "delightful_head_gate"


def test_bench_capture_flags_default_to_the_periodic_grid_and_refuse_overreach():
    args = _parse()
    assert (args.bench_capture_problems, args.bench_capture_samples) == (4, 4)
    args = _parse("--bench-only", "--bench-samples", "8", "--bench-max-rows", "512",
                  "--bench-capture-problems", "512", "--bench-capture-samples", "8")
    assert (args.bench_capture_problems, args.bench_capture_samples) == (512, 8)
    for extra in (("--bench-capture-problems", "0"),
                  ("--bench-max-rows", "16", "--bench-capture-problems", "17"),
                  ("--bench-only", "--bench-samples", "4", "--bench-capture-samples", "5")):
        with pytest.raises(SystemExit):
            _parse(*extra)


def test_policy_and_critic_views_must_agree_on_every_trajectorys_action_count():
    """Equal totals are not enough: with 3 + 1 actions against 1 + 3, every
    policy row would centre the wrong trajectory's actions. And equal counts
    are not enough for the offsets, which carry the taken token's row."""
    from postraining.latent_rollout import policy_action_offsets_at_actions
    from postraining.rollout_dataset import group_batch

    wrapper = _cot_wrapper()
    critic = _q_critic(wrapper)
    policy = group_batch([5, 6], [(7, 8, 9), (10,)], [1.0, 0.0])
    shifted = group_batch([5, 6, 11], [(7,), (8, 9, 10)], [1.0, 0.0])
    assert int(policy.emit_mask.sum()) == int(shifted.action_mask.sum())
    with pytest.raises(ValueError, match="a trajectory's action count"):
        policy_action_offsets_at_actions(wrapper, policy, critic.action_latents(), 512, shifted)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    with pytest.raises(RuntimeError, match="a trajectory's action count"):
        update_minibatch(wrapper, critic, shifted, optimizers, value_only=True,
                         critic_step=False, actor_step=False, policy_batch=policy)
    retokened = group_batch([5, 6], [(7, 8, 12), (10,)], [1.0, 0.0])
    with pytest.raises(ValueError, match="took different tokens"):
        policy_action_offsets_at_actions(wrapper, policy, critic.action_latents(), 512, retokened)
    with pytest.raises(RuntimeError, match="took different tokens"):
        update_minibatch(wrapper, critic, retokened, optimizers, value_only=True,
                         critic_step=False, actor_step=False, policy_batch=policy)


def _memory_batch(wrapper, seed: int = 5):
    batch = _roll(wrapper, _prompts("nano", 4, 5, seed=seed), seed=seed)
    assign_terminal_rewards(batch, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    return batch


@pytest.mark.parametrize("estimator", ["all_action", "sampled_baseline", "delightful_sampled_baseline",
                                       "delightful_td_gate", "delightful_answer_gate"])
def test_memory_mac_actor_gradient_is_the_estimators_expectation(estimator):
    # A live memory: every read and write is load-bearing, so the gradient
    # through the replay scan reaches the reader and the writer.
    wrapper = _memory_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    batch = _stamped(_memory_batch(wrapper))
    expected = _oracle_actor_gradients(wrapper, critic, copy.deepcopy(batch), estimator)
    wrapper.zero_grad(set_to_none=True)
    metrics = _mac(wrapper, critic, batch, optimizers, mac_vocab_chunk=3,
                   mac_estimator=estimator)
    actual = _gradients(_actor_parameters(wrapper))
    rounding = _score_rounding(wrapper, critic, batch)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.double(), want.double(), rtol=1e-4,
                                   atol=max(1e-6, rounding * float(want.abs().max())))
    memory = wrapper.combiner
    for parameter in (memory.reader.out.weight, memory.reader.query.weight,
                      memory.writer.key.weight, memory.writer.value.weight):
        assert parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
    # The memory telemetry runs on the MAC branch too.
    assert 0 < metrics["memory_write_beta_mean"] < 1
    assert metrics["carry_input_delta_rms_ratio"] > 0


@pytest.mark.parametrize("twin", [False, True])
def test_a_cot_study_critic_warm_starts_a_memory_critic_exactly(twin):
    factory, _ = BACKBONES["nano"]
    cot = _cot_wrapper()
    study = _q_critic(cot)
    if twin:
        single = study
        study = SeparateCritic(factory(11), q_head_hidden=HIDDEN, q_twin=True).eval()
        study.load_state_dict(single.state_dict(), strict=False)
        # Twins that disagree, so each head's warm start is checked.
        with torch.no_grad():
            study.q_twin.state.weight.normal_(std=0.5)
            study.q_twin.direction.weight.normal_(std=0.5)
    # A fresh memory policy reads nothing: it is the cot policy at step 0.
    wrapper = _memory_wrapper(live=False)
    wrapper.backbone.load_state_dict(cot.backbone.state_dict())
    critic = SeparateCritic(factory(11), actor_memory=wrapper.combiner, q_head_hidden=HIDDEN,
                            q_head_reads_actor_belief=True, q_twin=twin).eval()
    with torch.no_grad():
        for head in critic.q_heads:
            head.actor_belief.weight.normal_(std=0.5)
    load_study_critic(critic, study.state_dict())
    memory_batch = _memory_batch(wrapper)
    cot_batch = _cot_batch(cot)
    assert torch.equal(memory_batch.token_ids, cot_batch.token_ids)
    with torch.no_grad():
        for got, want in zip(critic.token_q_parts(memory_batch), study.token_q_parts(cot_batch)):
            torch.testing.assert_close(got, want)
    # Anything else the study critic lacks is a different critic.
    with pytest.raises(ValueError, match="outside its memory inputs"):
        load_study_critic(critic, {name: value for name, value in study.state_dict().items()
                                   if not name.startswith("q_head.state")})


def test_a_memory_run_accepts_a_study_critic_of_cot_rollouts_only(tmp_path):
    critic = _q_critic(_cot_wrapper())
    rate = 1e-4 / critic.q_head.direction.out_features
    for mode in ("cot", "memory"):
        (tmp_path / "manifest.json").write_text(json.dumps({
            "math_data_identity": "drills",
            "args": {"reasoning_mode": mode, "temperature": 1.0, "top_p": 1.0},
        }))
        payload = _study_payload(critic, tmp_path)
        args = _study_args(critic_q_direction_learning_rate=rate, reasoning_mode="memory")
        if mode == "cot":
            validate_study_critic_init(payload, critic, args, "drills")
        else:
            with pytest.raises(ValueError, match="reasoning_mode"):
                validate_study_critic_init(payload, critic, args, "drills")


# --- Answer-level sequence estimators through the update -------------------


def test_group_terms_are_the_leave_one_out_advantage_and_answer_surprisal():
    wrapper = _cot_wrapper()
    batch = _stamped(_cot_batch(wrapper))
    assert graded_answer_keys(VERDICTS, [1.0, 0.0, 1.0, 0.0]) == [
        ("7", 1.0), ("3", 0.0), ("7", 1.0), None]
    assert graded_answer_keys(
        [RowVerdict(True, "[INVALID]"), RowVerdict(True, None)], [0.0, 0.0]) == [None, None]
    with pytest.raises(ValueError, match="one verdict per group row"):
        graded_answer_keys(VERDICTS, [1.0, 0.0])
    torch.testing.assert_close(batch.group_advantage,
                               torch.tensor([2 / 3, -2 / 3, 2 / 3, -2 / 3]))
    torch.testing.assert_close(batch.answer_surprisal, torch.tensor(
        [math.log(2), math.log(4), math.log(2), math.log(4)]))
    with pytest.raises(ValueError, match="one verdict per group row"):
        stamp_group_terms(batch, VERDICTS[:3])


def test_answer_clusters_split_one_canonical_string_graded_two_ways():
    """verify_answer canonicalizes "3 4" and "34" to one string but grades
    them differently: a wrong row must not join the correct row's cluster
    (latent_eval.majority_vote_score keys the same (answer, correct) pair)."""
    wrapper = _cot_wrapper()
    verdicts = [RowVerdict(True, "34")] * 3 + [RowVerdict(True, "5")]
    batch = _cot_batch(wrapper)
    assign_terminal_rewards(batch, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    stamp_group_terms(batch, verdicts)
    # Clusters {0, 2} ("34", correct), {1} ("34", wrong), {3} ("5", wrong).
    torch.testing.assert_close(batch.answer_surprisal, torch.tensor(
        [math.log(2), math.log(4), math.log(2), math.log(4)]))
    # Keyed on the string alone, row 1 would sit in a cluster of 3.
    assert float(batch.answer_surprisal[1]) != pytest.approx(math.log(4 / 3))


def test_a_sequence_estimator_refuses_an_unstamped_batch():
    wrapper = _cot_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    for estimator in SEQUENCE_ESTIMATORS:
        with pytest.raises(ValueError, match="group terms"):
            _mac(wrapper, critic, _cot_batch(wrapper), optimizers, mac_estimator=estimator)


def test_the_answer_gate_at_one_is_rloo(monkeypatch):
    """With the gate forced to 1 the answer-gate update is RLOO's to the
    bit, and the critic's gradient is the same in all three arms: the
    coefficient is the only difference, and the critic still trains."""
    import postraining.mean_actor_critic as mean_actor_critic

    wrapper = _cot_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    batch = _stamped(_cot_batch(wrapper))

    def run(estimator):
        wrapper.zero_grad(set_to_none=True)
        critic.zero_grad(set_to_none=True)
        metrics = _mac(wrapper, critic, copy.deepcopy(batch), optimizers,
                       mac_estimator=estimator)
        return (metrics, _gradients(_actor_parameters(wrapper)),
                _gradients(list(critic.parameters())))

    rloo, gated = run("rloo_sequence"), run("delightful_answer_gate")
    with monkeypatch.context() as patch:
        patch.setattr(mean_actor_critic, "answer_gate",
                      lambda advantages, surprisal: torch.ones_like(advantages))
        ungated = run("delightful_answer_gate")
    assert all(torch.equal(a, b) for a, b in zip(rloo[1], ungated[1]))
    assert not all(torch.equal(a, b) for a, b in zip(rloo[1], gated[1]))
    for left, right in ((rloo, gated), (rloo, ungated)):
        assert all(torch.equal(a, b) for a, b in zip(left[2], right[2]))
    assert any(bool(g.abs().sum()) for g in rloo[2])


def test_sequence_estimator_telemetry():
    wrapper = _cot_wrapper()
    critic = _q_critic(wrapper)
    optimizers = build_optimizers(wrapper, critic, learning_rate=0.0, fused=False)
    batch = _stamped(_cot_batch(wrapper))
    advantages, surprisal = batch.group_advantage.double(), batch.answer_surprisal.double()
    gate = torch.sigmoid(advantages * surprisal)
    tokens = batch.emit_mask.sum(1).double()
    with torch.no_grad():
        q_values, _ = _token_values(wrapper, critic, batch)
        _, returns = generalized_advantage_and_return_targets(
            batch.rewards, torch.zeros_like(batch.old_values), batch.action_mask,
            torch.ones(batch.action_mask.size(0)))
    emit = batch.emit_mask.bool()
    # SB's coefficient G - E_pi Q, which the policy-centred critic's state
    # value c is: Q(a_t) plus the head's own A(a_t) taken back out.
    baseline = (returns - q_values + _expected_advantages(wrapper, critic, batch))[emit]
    rloo = mac_dashboard([_mac(wrapper, critic, copy.deepcopy(batch), optimizers,
                               mac_estimator="rloo_sequence")])
    metrics = _mac(wrapper, critic, copy.deepcopy(batch), optimizers,
                   mac_estimator="delightful_answer_gate")
    dashboard = mac_dashboard([metrics])
    for board in (rloo, dashboard):
        # Mean per-group answer entropy: clusters of 2, 1 and the ungraded 1.
        assert board["mac/answer_cluster_entropy_mean"] == pytest.approx(
            0.5 * math.log(2) + 0.5 * math.log(4))
        assert board["mac/answer_unanimous_share"] == 0
        assert board["mac/sequence_advantage_rms"] == pytest.approx(2 / 3, rel=1e-6)
        sb = math.sqrt((board["mac/think_sb_coefficient_rms"] ** 2 * board["mac/think_states"]
                        + board["mac/answer_sb_coefficient_rms"] ** 2
                        * board["mac/answer_states"]) / emit.sum())
        assert sb == pytest.approx(float(baseline.square().mean().sqrt()), rel=1e-4)
    assert rloo["mac/answer_gate_token_mass_mean"] == pytest.approx(1.0)
    assert not any(key.startswith("mac/answer_gate_q") for key in rloo)
    assert dashboard["mac/sequence_coefficient_rms"] == pytest.approx(
        float((advantages * gate).square().mean().sqrt()), rel=1e-6)
    assert dashboard["mac/answer_gate_token_mass_mean"] == pytest.approx(
        float((gate * advantages.abs() * tokens).sum() / (advantages.abs() * tokens).sum()),
        rel=1e-6)
    assert dashboard["mac/answer_gate_positive_mean"] == pytest.approx(
        float(gate[advantages > 0].mean()), rel=1e-6)
    assert dashboard["mac/answer_gate_negative_mean"] == pytest.approx(
        float(gate[advantages < 0].mean()), rel=1e-6)
    assert dashboard["mac/answer_gate_q50"] == pytest.approx(
        float(gate.quantile(0.5)), rel=1e-6)
    # The token coefficient RMS is the rows' weighted by their tokens.
    states = dashboard["mac/think_states"] + dashboard["mac/answer_states"]
    rms = math.sqrt((dashboard["mac/think_coefficient_rms"] ** 2 * dashboard["mac/think_states"]
                     + dashboard["mac/answer_coefficient_rms"] ** 2
                     * dashboard["mac/answer_states"]) / states)
    assert rms == pytest.approx(float(
        ((advantages * gate).square() * tokens).sum().div(tokens.sum()).sqrt()), rel=1e-5)
    # A unanimous group: every row l = 0, U = 0, no gate quantiles.
    unanimous = _stamped(_cot_batch(wrapper), [RowVerdict(True, "7")] * 4)
    assign_terminal_rewards(unanimous, torch.ones(4))
    stamp_group_terms(unanimous, [RowVerdict(True, "7")] * 4)
    board = mac_dashboard([_mac(wrapper, critic, unanimous, optimizers,
                                mac_estimator="delightful_answer_gate")])
    assert board["mac/answer_unanimous_share"] == 1
    assert board["mac/sequence_coefficient_rms"] == 0
    assert not any(key.startswith("mac/answer_gate_q") for key in board)


@pytest.mark.parametrize(("extra", "message"), [
    (("--samples-per-prompt", "1"), "at least 2"),
    (("--no-answer-fence",), "--answer-fence"),
    (("--nearby-reward-max", "0.1"), "pure verifier reward"),
    (("--abstention-penalty", "0.5"), "pure verifier reward"),
])
def test_sequence_estimator_cli_refusals(extra, message, capsys):
    for estimator in SEQUENCE_ESTIMATORS:
        with pytest.raises(SystemExit):
            _parse(*MAC_ARGS, "--mac-estimator", estimator, *extra)
        assert message in capsys.readouterr().err
