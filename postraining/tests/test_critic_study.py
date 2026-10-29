from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from postraining.branch_rollouts import (
    BranchPair,
    branch_arena_width,
    continuation_budget,
    roll_branches,
    sample_alternative,
    think_positions,
    truncate_continuation,
)
from postraining.critic_study import (
    STUDY_ARMS,
    CriticArm,
    Study,
    gate_verdicts,
    paired_agreement_difference,
    resolved_gap_mask,
    run_arm,
    select_arms,
    sign_agreement,
    think_action_mask,
    wilson_interval,
    within_prompt_slope,
)
from postraining.latent_rollout import (
    compact_next_slots,
    compact_slots,
    emitted_token_rows,
    replay_head_inputs,
    rollout_continuations,
    slot_index,
    split_rollout_groups,
    trim_stream,
)
from postraining.latent_thought import LatentThoughtModel
from postraining.rollout_dataset import RolloutGroup, group_batch, prompt_sha256
from postraining.tests.test_hidden_carry import NANO_KWARGS, _nano_backbone
from postraining.tests.test_q_critic import _expected_advantages, _oracle_action_offsets
from postraining.train_latent_vapo import build_optimizers, update_minibatch
from postraining.vapo.schemas import CRITIC_Q_SCHEMA

OPEN, CLOSE, STOP = 5, 6, 3


@pytest.mark.parametrize("arms", [",".join(arm.name for arm in STUDY_ARMS), "actor_trunk"])
def test_entrypoint_freezes_sources_and_applies_family_alpha(tmp_path, monkeypatch, arms):
    from postraining import critic_study, provenance

    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"not loaded by this orchestration test")
    calls = []
    snapshots = []
    monkeypatch.setattr(provenance, "capture_source_provenance", lambda output, repo: snapshots.append(output) or {"snapshot": "fixture"})
    monkeypatch.setattr(critic_study, "load_study", lambda args: SimpleNamespace(
        args=args, rollout_dataset=SimpleNamespace(dataset_digest=lambda m: m),
        train_manifest="train", heldout_manifest="heldout",
        torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
    ))
    final = dict.fromkeys(("gates", "slope", "slope_early_think", "branch", "ev_think_actions"))

    def fit(study, arm, output):
        calls.append((arm.name, study.args.ci_alpha, study.args.epochs))
        return {"final": final}

    monkeypatch.setattr(critic_study, "run_arm", fit)
    output = tmp_path / "fit"
    critic_study.main([
        "--checkpoint", str(checkpoint), "--train-dataset", "train",
        "--heldout-dataset", "heldout", "--output", str(output), "--arms", arms,
        "--family-alpha", "0.05",
    ])
    expected = select_arms(arms)
    assert calls == [(arm.name, pytest.approx(0.05 / len(expected)), 4) for arm in expected]
    assert snapshots == [output / "provenance"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source_provenance"] == {"snapshot": "fixture"}
    assert set(json.loads((output / "results.json").read_text())["arms"]) == {arm.name for arm in expected}


# --- statistics ----------------------------------------------------------------


def test_think_action_mask_selects_think_states_not_think_tokens():
    tokens = [OPEN, 9, 9, 9, CLOSE, 7, 7, STOP]
    # The closing action is taken from a think state, so it is credited;
    # the answer span's states follow the close and are not.
    assert think_action_mask(tokens, OPEN, CLOSE).tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    # Unclosed: every action after the opening fence, a stop taken from a
    # think state included; never-opened: nothing.
    assert think_action_mask([OPEN, 9, 9], OPEN, CLOSE).tolist() == [0, 1, 1]
    assert think_action_mask([OPEN, 9, 9, STOP], OPEN, CLOSE).tolist() == [0, 1, 1, 1]
    assert not think_action_mask([9, CLOSE, 9], OPEN, CLOSE).any()
    # Only the first span counts.
    assert think_action_mask([OPEN, 9, CLOSE, OPEN, 9], OPEN, CLOSE).tolist() == [0, 1, 1, 0, 0]
    # The branch tool selects from exactly the same states.
    assert think_positions(tokens, OPEN, CLOSE).tolist() == [1, 2, 3, 4]
    assert think_positions([OPEN, 9, 9, STOP], OPEN, CLOSE).tolist() == [1, 2, 3]


def test_think_mask_keeps_the_summed_advantage_centred_under_the_policy():
    # A two-step think MDP: at each think state the policy continues or
    # closes. With A = Q - sum_a pi(a) Q(s, a), the expected advantage
    # summed over the masked actions is zero only when the mask is a
    # function of the state, i.e. includes the closing action.
    rng = np.random.default_rng(0)
    p_close, q_close, q_continue = 0.3, -1.0, 0.5
    baseline = p_close * q_close + (1 - p_close) * q_continue
    totals = {True: [], False: []}
    for _ in range(20000):
        tokens, advantages = [OPEN], [0.0]
        for _ in range(2):
            if rng.random() < p_close:
                tokens.append(CLOSE)
                advantages.append(q_close - baseline)
                break
            tokens.append(9)
            advantages.append(q_continue - baseline)
        mask = think_action_mask(tokens, OPEN, CLOSE)
        advantages = np.array(advantages)
        totals[True].append(advantages[mask].sum())
        token_mask = mask & (np.array(tokens) != CLOSE)
        totals[False].append(advantages[token_mask].sum())
    assert abs(np.mean(totals[True])) < 0.02
    # Dropping the close leaves -pi(close) A(close) behind at every state.
    assert np.mean(totals[False]) > 0.3


def test_within_prompt_slope_removes_prompt_effects_and_recovers_the_slope():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(200), 8)
    offsets = rng.normal(scale=5.0, size=200)[groups]
    x = rng.normal(size=groups.size) + offsets  # x correlates with the prompt too
    y = 0.5 * x - 3.0 * offsets + rng.normal(scale=0.5, size=groups.size)
    result = within_prompt_slope(groups, x, y, bootstrap=400)
    # Pooled OLS would be dominated by the -3 * offset confounder.
    assert result["slope"] == pytest.approx(0.5, abs=0.05)
    assert result["slope_ci_low"] < 0.5 < result["slope_ci_high"]
    assert result["slope_ci_low"] > 0
    null = within_prompt_slope(groups, rng.normal(size=groups.size), y, bootstrap=400)
    assert null["slope_ci_low"] < 0 < null["slope_ci_high"]


def test_within_prompt_slope_counts_informative_prompts():
    groups = np.array([0, 0, 1, 1, 2, 2])
    x = np.array([1.0, 2.0, 1.0, 3.0, 0.0, 1.0])
    y = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    result = within_prompt_slope(groups, x, y, bootstrap=50)
    assert result["informative_prompts"] == 1
    assert result["prompts"] == 3
    # sxy = 0.5 from prompt 0 alone; sxx = 0.5 + 2 + 0.5 over every prompt.
    assert result["slope"] == pytest.approx(0.5 / 3.0)


def test_sign_agreement_scores_ties_as_half_and_resamples_prompts():
    observed = np.array([1.0, -1.0, 1.0, -1.0])
    groups = np.array(["a", "a", "b", "b"])
    assert sign_agreement(np.zeros(4), observed, groups, bootstrap=50)["agreement"] == 0.5
    perfect = sign_agreement(observed * 3, observed, groups, bootstrap=50)
    assert perfect["agreement"] == 1.0 and perfect["cluster_ci_low"] == 1.0
    wrong = sign_agreement(-observed, observed, groups, bootstrap=50)
    assert wrong["agreement"] == 0.0
    # Zero observed gaps carry no sign to agree with.
    assert sign_agreement(np.ones(2), np.array([0.0, 1.0]), np.array([0, 1]), bootstrap=10)["pairs"] == 1
    low, high = wilson_interval(8, 10)
    assert low < 0.8 < high


def test_resolved_gap_mask_is_fishers_exact_test_on_success_counts():
    taken = np.array([29, 18, 0, 16, 32])
    alternative = np.array([3, 14, 0, 6, 28])
    samples = np.array([32, 32, 32, 32, 32])
    assert resolved_gap_mask(taken, alternative, samples).tolist() == [True, False, False, True, False]
    # 32/32 against 28/32 passes a plug-in Wald test but not Fisher (p = 0.11).
    assert not resolved_gap_mask(np.array([32]), np.array([28]), 32).any()
    with pytest.raises(ValueError, match="success counts"):
        resolved_gap_mask(np.array([-1]), np.array([5]), np.array([32]))
    with pytest.raises(ValueError, match="success counts"):
        resolved_gap_mask(np.array([33]), np.array([5]), np.array([32]))


def test_paired_agreement_difference_scores_the_same_pairs():
    observed = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 0.0])
    critic = np.array([1.0, 1.0, -1.0, 1.0, 0.0, 5.0])  # 1, 1, 1, 0, 0.5 on resolved pairs
    prior = np.array([1.0, -1.0, -1.0, 1.0, 1.0, 5.0])  # 1, 0, 1, 0, 1
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    result = paired_agreement_difference(critic, prior, observed, groups, bootstrap=200)
    assert result["pairs"] == 5
    assert result["difference"] == pytest.approx((0 + 1 + 0 + 0 - 0.5) / 5)
    assert result["cluster_ci_low"] <= result["difference"] <= result["cluster_ci_high"]
    # Identical predictors differ by exactly zero in every resample.
    same = paired_agreement_difference(critic, critic, observed, groups, bootstrap=50)
    assert same["difference"] == same["cluster_ci_low"] == same["cluster_ci_high"] == 0.0

def test_gates_need_the_cluster_interval_and_the_policy_prior_beaten():
    slope = {"slope": 0.2, "slope_ci_low": 0.05, "informative_prompts": 30}
    branch = {
        "critic": {"agreement": 0.7, "cluster_ci_low": 0.55, "pairs": 40},
        "critic_minus_prior": {"difference": 0.05, "cluster_ci_low": 0.01},
        "oracle_beats_prior": True,
    }
    assert gate_verdicts(slope, branch) == {"slope": True, "branch": True}
    # The oracle is the weaker test (half the continuations), so a critic
    # that beats the prior passes even where the oracle does not.
    unpowered = {**branch, "oracle_beats_prior": False}
    assert gate_verdicts(slope, unpowered)["branch"] is True
    # Not beating the prior is undecided only where the oracle could not
    # either; a critic failing agreement fails regardless.
    tied = {**branch, "critic_minus_prior": {"difference": 0.05, "cluster_ci_low": -0.02}}
    assert gate_verdicts(slope, {**tied, "oracle_beats_prior": False})["branch"] is None
    disagreeing = {**unpowered, "critic": {"agreement": 0.55, "cluster_ci_low": 0.45, "pairs": 40}}
    assert gate_verdicts(slope, disagreeing)["branch"] is False
    assert gate_verdicts({**slope, "slope_ci_low": -0.01}, branch)["slope"] is False
    # Beating the prior on the point estimate alone is not enough.
    noisy = {**branch, "critic_minus_prior": {"difference": 0.05, "cluster_ci_low": -0.02}}
    assert gate_verdicts(slope, noisy)["branch"] is False
    # A critic whose summed advantage never varies within a prompt fails;
    # data without a single mixed prompt cannot decide.
    degenerate = {"slope": float("nan"), "slope_ci_low": float("nan"), "informative_prompts": 30}
    assert gate_verdicts(degenerate, None) == {"slope": False, "branch": None}
    assert gate_verdicts({**degenerate, "informative_prompts": 0}, None)["slope"] is None


def test_arm_registry_changes_one_factor_per_arm():
    base = STUDY_ARMS[0]
    assert base.name == "actor_trunk" and base.critic_init == "actor"
    single = [arm for arm in STUDY_ARMS if not arm.q_twin]
    for arm in single[1:]:
        differing = [key for key, value in vars(arm).items() if key != "name" and value != getattr(base, key)]
        assert len(differing) == 1, (arm.name, differing)
    # A twin arm is one single-head arm plus the twin head.
    twins = [arm for arm in STUDY_ARMS if arm.q_twin]
    assert [arm.name for arm in twins] == ["actor_trunk_twin", "frozen_belief_gain_twin"]
    for arm in twins:
        assert replace(arm, name=arm.name.removesuffix("_twin"), q_twin=False) in single
    assert [arm.name for arm in select_arms("gold_context,actor_trunk")] == ["gold_context", "actor_trunk"]
    with pytest.raises(ValueError):
        select_arms("actor_trunk,actor_trunk")
    # A fresh trunk's readout, the token Q's action table, is zero.
    with pytest.raises(ValueError, match="trunk's readout"):
        CriticArm("scratch", critic_init="scratch")
    with pytest.raises(ValueError):
        select_arms("nope")
    with pytest.raises(ValueError):
        CriticArm("bad", weighting="prompt")
    # The trainer's value loss has no per-trajectory weights.
    with pytest.raises(ValueError, match="token-weighted"):
        CriticArm("bad", weighting="trajectory")


# --- branch helpers --------------------------------------------------------------


def test_alternative_excludes_the_taken_token():
    rng = np.random.default_rng(0)
    probabilities = np.array([0.7, 0.2, 0.1, 0.0])
    draws = [sample_alternative(probabilities, 0, rng) for _ in range(3000)]
    assert 0 not in draws and 3 not in draws
    assert np.mean(np.array(draws) == 1) == pytest.approx(2 / 3, abs=0.03)
    with pytest.raises(ValueError):
        sample_alternative(np.array([1.0, 0.0]), 0, rng)


def test_branch_budget_and_truncation_match_the_original_rollout():
    assert continuation_budget(position=10, max_new_tokens=64) == 53
    assert truncate_continuation([1, 2, STOP, 4], 10, frozenset({STOP})) == [1, 2, STOP]
    assert truncate_continuation([1, 2, 4, 4], 3, frozenset({STOP})) == [1, 2, 4]


# --- the arm mechanics on a tiny token policy ----------------------------------


SAMPLES = 4


def _tokenizer():
    return SimpleNamespace(think_open_id=OPEN, think_close_id=CLOSE, encode=lambda text: [11, 12, 13])


def _wrapper():
    backbone = _nano_backbone(3)
    backbone.model_config = dict(NANO_KWARGS)
    backbone.architecture = "nanogpt_mini_test"
    return LatentThoughtModel(backbone).eval()


def _pool(wrapper, pool: int, prompts: list[list[int]], seed: int) -> list[RolloutGroup]:
    width = max(len(prompt) for prompt in prompts)
    prompt_ids = torch.zeros((len(prompts), width), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        prompt_ids[row, width - len(prompt):] = torch.tensor(prompt)
    lengths = torch.tensor([len(prompt) for prompt in prompts])
    batch = rollout_continuations(
        wrapper, prompt_ids, 10, 10, 1.0, 1.0, generator=torch.Generator().manual_seed(seed),
        stop_ids=(STOP,), prompt_lengths=lengths, prompt_repeats=SAMPLES, pin_emit=True,
    )
    groups = []
    for number, (group, prompt) in enumerate(
        zip(split_rollout_groups(batch, SAMPLES, lengths.repeat_interleave(SAMPLES)), prompts, strict=True)
    ):
        # Think spans by construction: the tiny policy never learned fences.
        completions = tuple(
            (OPEN, *(token for token in row if token != STOP), CLOSE, STOP)
            for row in emitted_token_rows(trim_stream(group))
        )
        rewards = tuple(float((number + sample) % 2) for sample in range(SAMPLES))
        text = f"prompt {prompt}"
        groups.append(RolloutGroup(
            pool=pool, group=number, source="drills", prompt_index=str(number), prompt_text=text,
            prompt_sha256=prompt_sha256(text), ground_truth="1", answer_style="rule", module=None,
            prompt_tokens=tuple(prompt), completions=completions, rewards=rewards,
            structural_format_ok=(True,) * SAMPLES, terminated=(True,) * SAMPLES,
        ))
    return groups


def _args(**overrides):
    defaults = {
        "replay_bucket": 4, "replay_max_trajectories": 64, "replay_attention_budget": 1 << 20,
        "replay_slot_budget": None, "q_head_hidden": 16, "seed": 7, "bootstrap": 50,
        "min_branch_agreement": 0.6, "ci_alpha": 0.05, "probe_groups": 2, "vocab_chunk": 7, "epochs": 2, "eval_every": 2,
        "combined_mlp_hidden": 8, "combined_mlp_blocks": 1, "compile": True,
    }
    return SimpleNamespace(**{**defaults, **overrides})


def _study(branches=None, **overrides):
    wrapper = _wrapper()
    train = [_pool(wrapper, 0, [[1, 2, 4], [7, 8]], seed=1), _pool(wrapper, 1, [[9, 10, 4, 4]], seed=2)]
    heldout = [_pool(wrapper, 0, [[20, 21, 22], [23, 24]], seed=3)]
    return Study(_args(**overrides), wrapper, _tokenizer(), train, heldout, branches)


@pytest.mark.parametrize("arm", STUDY_ARMS, ids=lambda arm: arm.name)
def test_study_fits_with_a_fully_frozen_loaded_actor(arm):
    study = _study()
    # load_model freezes a real checkpoint; the original tiny-model fixture
    # left the actor trainable and missed Muon(empty actor parameters).
    study.wrapper.requires_grad_(False)
    actor_before = study.actor_fingerprint()
    critic, optimizers = study.build_critic(arm)
    assert set(optimizers) == {"critic", "critic_action", "critic_muon"}
    critic_before = [parameter.detach().clone() for parameter in critic.parameters()]
    metrics = study.fit_pool(critic, optimizers, study.train_pools[0], arm)
    assert np.isfinite(metrics["objective_loss"])
    assert any(not torch.equal(before, after) for before, after in zip(critic_before, critic.parameters(), strict=True))
    assert study.actor_fingerprint() == actor_before
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in study.wrapper.parameters())


def test_study_refuses_prompts_shared_by_train_and_heldout():
    wrapper = _wrapper()
    pool = _pool(wrapper, 0, [[1, 2, 4]], seed=1)
    with pytest.raises(ValueError, match="both train and held-out"):
        Study(_args(), wrapper, _tokenizer(), [pool], [pool])


def test_study_refuses_heldout_cells_without_training_trajectories():
    wrapper = _wrapper()
    train = [_pool(wrapper, 0, [[1, 2, 4]], seed=1)]
    (group,) = _pool(wrapper, 0, [[20, 21, 22]], seed=3)
    heldout = [[replace(group, module="mul_integer_d4")]]
    with pytest.raises(ValueError, match=r"held-out cells \['mul_integer_d4'\]"):
        Study(_args(), wrapper, _tokenizer(), train, heldout)


def test_every_arm_starts_from_the_same_critic_and_splits_the_direction_rate():
    study = _study()
    base, _ = study.build_critic(CriticArm("base"))
    unscaled, unscaled_optimizers = study.build_critic(CriticArm("u", direction_scale="unscaled"))
    for (name, left), right in zip(base.state_dict().items(), unscaled.state_dict().values(), strict=True):
        assert torch.equal(left, right), name
    _, optimizers = study.build_critic(CriticArm("base"))
    assert not any(name.startswith("actor") for name in optimizers)
    rest, direction = optimizers["critic_action"].param_groups
    dim = base.q_head.direction.out_features
    assert direction["lr"] == pytest.approx(rest["lr"] / dim)
    assert len(unscaled_optimizers["critic_action"].param_groups) == 1


def test_scored_advantages_are_the_trainers_policy_centred_token_q():
    study = _study()
    critic, _ = study.build_critic(CriticArm("base"))
    with torch.no_grad():
        critic.q_head.state.weight.normal_(std=0.5)
        critic.q_head.direction.weight.normal_(std=0.5)
    group = study.heldout_pools[0][0]
    scored = study.score_group(critic, group, CriticArm("base"))
    batch = group_batch(group.prompt_tokens, group.completions, group.rewards, 4)
    expected = _expected_advantages(study.wrapper, critic, batch)
    expected = expected[batch.emit_mask.bool()].numpy()
    np.testing.assert_allclose(scored["advantage"], expected, rtol=1e-4, atol=1e-6)


def test_gold_context_moves_only_the_critic_stream():
    study = _study()
    group = study.heldout_pools[0][0]
    plain = study.critic_batch(group, CriticArm("base"))
    gold = study.critic_batch(group, CriticArm("g", context="gold"))
    assert gold.prompt_length == plain.prompt_length + 3
    assert gold.token_ids[0, plain.prompt_length : gold.prompt_length].tolist() == [11, 12, 13]
    assert gold.action_mask.sum() == plain.action_mask.sum()
    assert torch.equal(gold.reward_scalar, plain.reward_scalar)


def test_gold_context_scores_the_actors_policy_against_the_gold_critic():
    """A for the gold arm: Q from the critic's gold stream, pi from the
    actor's plain stream, aligned action by action."""
    study = _study()
    gold_arm = CriticArm("g", context="gold")
    critic, _ = study.build_critic(gold_arm)
    with torch.no_grad():
        critic.q_head.state.weight.normal_(std=0.5)
        critic.q_head.direction.weight.normal_(std=0.5)
    group = study.heldout_pools[0][0]
    scored = study.score_group(critic, group, gold_arm)
    plain = group_batch(group.prompt_tokens, group.completions, group.rewards, 4)
    gold = study.critic_batch(group, gold_arm)
    with torch.no_grad():
        beliefs, inputs = replay_head_inputs(study.wrapper, plain)
        actor_index = slot_index(plain.emit_mask.bool())
        logits = study.wrapper.backbone.logits_from_features(
            study.wrapper.renderer_features(compact_slots(inputs, actor_index), compact_slots(beliefs, actor_index))
        ).double()
        # A reads only the direction: the centring row cancels in Q(a) - E_pi Q.
        _, member_directions = critic.token_q_parts(gold)
        q = compact_slots(member_directions.mean(0), slot_index(gold.action_mask.bool())).double() @ critic.action_latents().double().T
        targets = compact_next_slots(plain.token_ids, actor_index)
        expected = q.gather(-1, targets[:, None]).squeeze(-1) - (logits.softmax(-1) * q).sum(-1)
    np.testing.assert_allclose(scored["advantage"], expected.numpy(), rtol=1e-4, atol=1e-6)
    # The reference answer reached the critic: the plain stream scores differently.
    assert not np.allclose(scored["advantage"], study.score_group(critic, group, CriticArm("base"))["advantage"])



@pytest.mark.parametrize("trajectories, bucket", [(64, 4), (1, 1), (2, 4)])
def test_gold_context_fit_centres_q_under_the_actors_plain_policy(trajectories, bucket):
    """The gold critic regresses Q = c + u.(x - xbar) with xbar the policy's
    mean row where the policy acts, from its plain prompt: the gold context
    is the critic's information, never the policy's. Shards and buckets
    reorder and pad the two views differently; the rows must still meet."""
    study = _study(replay_max_trajectories=trajectories, replay_bucket=bucket)
    gold_arm = CriticArm("g", context="gold")
    critic, optimizers = study.build_critic(gold_arm)
    _randomized(critic)
    pool = study.train_pools[0]  # two prompts of different lengths, packed

    def loss(view):
        total, actions = 0.0, 0
        with torch.no_grad():
            for group in pool:
                gold = study.critic_batch(group, gold_arm)
                actor = view(group, gold)
                mask = gold.action_mask.bool()
                rows = torch.zeros((*mask.shape, critic.action_latents().size(-1)))
                rows[mask] = _oracle_action_offsets(study.wrapper, critic, actor)[actor.emit_mask.bool()]
                values, _ = critic.token_values(gold, rows)
                targets = gold.reward_scalar[:, None].expand_as(values)
                total += float(((values - targets).square() * mask).sum())
                actions += int(mask.sum())
        return total / actions

    want = loss(lambda group, gold: study.actor_batch(group))
    tolerance = 1e-5
    # Centring under the gold stream instead reads a different policy, by
    # well past the fit's tolerance (6.5e-4 relative here).
    assert abs(loss(lambda group, gold: gold) - want) > 10 * tolerance * want
    fitted = study.fit_pool(critic, optimizers, pool, gold_arm)
    assert fitted["value_loss"] == pytest.approx(want, rel=tolerance)


TWIN = next(arm for arm in STUDY_ARMS if arm.q_twin)


def _randomized(critic, seed=1):
    """Nonzero readouts on every head, so each head predicts its own Q."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for head in critic.q_heads:
            head.state.weight.normal_(generator=generator, std=0.5)
            head.direction.weight.normal_(generator=generator, std=0.5)
    return critic


def test_twin_arm_adds_a_second_head_without_moving_the_single_head_draws():
    study = _study()
    single, _ = study.build_critic(replace(TWIN, q_twin=False))
    twin, _ = study.build_critic(TWIN)
    assert single.q_twin is None and len(twin.q_heads) == 2
    # Built last: the trunk, combiner and first head draw exactly what the
    # single-head arm draws.
    twin_state = twin.state_dict()
    for name, value in single.state_dict().items():
        assert torch.equal(value, twin_state[name]), name
    assert set(twin_state) - set(single.state_dict()) == {
        f"q_twin.{name}" for name in twin.q_twin.state_dict()
    }
    # Independently initialized hidden layers, both readouts at zero: a
    # fresh twin critic predicts Q = 0 for every token, like a single head.
    first, second = twin.q_heads
    assert not torch.equal(first.belief.weight, second.belief.weight)
    for head in twin.q_heads:
        assert not head.state.weight.any() and not head.state.bias.any()
        assert not head.direction.weight.any()


def test_twin_loss_is_the_sum_of_each_heads_monte_carlo_regression():
    study = _study()
    group = study.train_pools[1][0]
    batch = study.critic_batch(group, TWIN)
    critic, optimizers = study.build_critic(TWIN)
    _randomized(critic)
    actions = batch.action_mask.sum()
    # Every action's target is its trajectory's terminal reward: the
    # value-only update's lambda-one, undiscounted Monte Carlo return.
    targets = batch.reward_scalar[:, None].expand_as(batch.action_mask)
    members, _ = critic.token_value_members(batch, _oracle_action_offsets(study.wrapper, critic, batch))
    losses = [((values - targets).square() * batch.action_mask).sum() / actions for values in members]
    expected = []
    for head, loss in zip(critic.q_heads, losses, strict=True):
        expected.append(torch.autograd.grad(loss, list(head.parameters()), retain_graph=True))
    study.trainer.zero_optimizers(optimizers, "critic")
    metrics = update_minibatch(study.wrapper, critic, batch, optimizers, value_only=True,
                               critic_step=False, actor_step=False)
    assert metrics["value_loss"] == pytest.approx(float(sum(losses).detach()), rel=1e-5)
    # Each head's gradient is its own loss's alone: the twins share no
    # target bootstrap and no readout.
    for head, grads in zip(critic.q_heads, expected, strict=True):
        for parameter, grad in zip(head.parameters(), grads, strict=True):
            # fp32 rounding of the summed contributions, about 1e-7 of the
            # largest entry: readout-row latents put unit-RMS rows under
            # randomized heads, so these gradients reach O(100).
            torch.testing.assert_close(parameter.grad, grad, rtol=1e-4,
                                       atol=1e-6 * float(grad.abs().max()))
    # The fit reports the heads' taken-token disagreement.
    gap = (members[0] - members[1]).detach()
    rms = float(((gap.square() * batch.action_mask).sum() / actions).sqrt())
    assert rms > 0
    assert study.trainer.twin_dashboard([metrics])["twin/gap_rms"] == pytest.approx(rms, rel=1e-5)
    fitted = study.fit_pool(*study.build_critic(TWIN), study.train_pools[1], TWIN)
    assert fitted["objective_loss"] == fitted["value_loss"]
    assert fitted["twin/gap_rms"] == 0.0  # fresh heads: both predict zero


def test_twin_direction_rate_split_covers_both_readouts():
    study = _study()
    critic, optimizers = study.build_critic(TWIN)
    rest, directions = optimizers["critic_action"].param_groups
    assert directions["token_q_direction"]
    assert [id(p) for p in directions["params"]] == [
        id(weight) for head in critic.q_heads for weight in head.direction_weights]
    assert directions["lr"] == pytest.approx(rest["lr"] / critic.q_head.direction.out_features)
    head_ids = {id(p) for head in critic.q_heads for p in head.parameters()}
    assert {id(p) for p in rest["params"]} == head_ids - {id(p) for p in directions["params"]}


@pytest.mark.parametrize("twin", [False, True])
def test_a_frozen_gain_arm_is_the_trained_arm_with_the_gain_held_at_zero(twin):
    """The ablation changes nothing but whether g trains: the same initial
    critic, g in no optimizer, and still exactly zero after a fit, while
    the trained arm's g leaves zero at the head rate."""
    study = _study()
    trained_arm = CriticArm("trained", q_twin=twin)
    frozen_arm = replace(trained_arm, name="frozen", train_belief_gain=False)
    trained, trained_optimizers = study.build_critic(trained_arm)
    frozen, frozen_optimizers = study.build_critic(frozen_arm)
    for (name, left), right in zip(trained.state_dict().items(), frozen.state_dict().values(), strict=True):
        assert torch.equal(left, right), name

    def registered(optimizers):
        return {id(p) for optimizer in optimizers.values() for group in optimizer.param_groups
                for p in group["params"]}

    gains = {id(head.belief_gain) for head in frozen.q_heads}
    assert len(gains) == len(frozen.q_heads) == 1 + twin
    assert not gains & registered(frozen_optimizers)
    trained_gains = {id(head.belief_gain): group["lr"]
                     for group in trained_optimizers["critic_action"].param_groups
                     for p in group["params"] for head in trained.q_heads if p is head.belief_gain}
    assert len(trained_gains) == 1 + twin and set(trained_gains.values()) == {trained_arm.head_learning_rate}
    for critic, optimizers, arm in ((trained, trained_optimizers, trained_arm),
                                    (frozen, frozen_optimizers, frozen_arm)):
        study.fit_pool(critic, optimizers, study.train_pools[0], arm)
    assert all(float(head.belief_gain) == 0.0 for head in frozen.q_heads)
    assert all(float(head.belief_gain.detach()) != 0.0 for head in trained.q_heads)


@pytest.mark.parametrize("arm", [STUDY_ARMS[0], TWIN], ids=lambda arm: arm.name)
def test_critic_optimizers_are_the_trainers_critic_layout(arm):
    """The study's critic optimizers are exactly the critic half of
    ``build_optimizers``, which needs a trainable actor to build at all."""
    study = _study()
    critic, optimizers = study.build_critic(arm)
    trainer_optimizers = build_optimizers(
        study.wrapper, critic, learning_rate=1e-5, critic_learning_rate=arm.trunk_learning_rate,
        trunk_optimizer="muon", muon_learning_rate=1e-5, critic_muon_learning_rate=arm.muon_learning_rate,
        critic_action_learning_rate=arm.head_learning_rate, fused=False,
        critic_q_direction_learning_rate=arm.head_learning_rate / critic.q_head.direction.out_features,
    )
    critic_names = sorted(name for name in trainer_optimizers if name.startswith("critic"))
    assert sorted(optimizers) == critic_names

    def layout(optimizer):
        return [
            ([id(p) for p in group["params"]], group["lr"], group.get("token_q_direction", False))
            for group in optimizer.param_groups
        ]

    for name in critic_names:
        ours, theirs = layout(optimizers[name]), layout(trainer_optimizers[name])
        assert [(params, flag) for params, _, flag in ours] == [(params, flag) for params, _, flag in theirs], name
        assert [lr for _, lr, _ in ours] == pytest.approx([lr for _, lr, _ in theirs]), name


def test_twin_scoring_reads_the_heads_mean():
    study = _study()
    critic = _randomized(study.build_critic(TWIN)[0])
    group = study.heldout_pools[0][0]
    scored = study.score_group(critic, group, TWIN)
    batch = study.critic_batch(group, TWIN)
    with torch.no_grad():
        rows = _oracle_action_offsets(study.wrapper, critic, batch)
        members, member_directions = critic.token_value_members(batch, rows)
        mean_q, mean_directions = critic.token_values(batch, rows)
    index = slot_index(batch.action_mask.bool())
    # The study centres with chunked fp32 policy rows, the oracle with an
    # fp64 softmax: Q = c + u.(x - xbar) differs by the rounding of its
    # largest terms, not of the (possibly near-zero) result.
    rounding = 4 * torch.finfo(torch.float32).eps * float(members.abs().max())
    np.testing.assert_allclose(scored["q_taken"], compact_slots(mean_q, index).numpy(), rtol=1e-6, atol=rounding)
    np.testing.assert_allclose(scored["member_q_taken"], torch.stack(
        [compact_slots(values, index) for values in members]).numpy(), rtol=1e-6, atol=rounding)
    torch.testing.assert_close(scored["directions"], compact_slots(mean_directions, index))
    # The actor's centred advantage under the mean direction; the heads'
    # gap is each head's centred advantage, computed head by head.
    plain = group_batch(group.prompt_tokens, group.completions, group.rewards, 4)
    emit = plain.emit_mask.bool()
    np.testing.assert_allclose(
        scored["advantage"], _expected_advantages(study.wrapper, critic, plain)[emit].numpy(), rtol=1e-4, atol=1e-6)
    per_head = []
    for head in range(2):
        def head_values(stream, action_offsets, head=head):
            return members[head], member_directions[head]
        view = SimpleNamespace(token_values=head_values, action_latents=critic.action_latents)
        per_head.append(_expected_advantages(study.wrapper, view, plain)[emit].numpy())
    np.testing.assert_allclose(scored["advantage_gap"], per_head[0] - per_head[1], rtol=1e-4, atol=1e-6)


def test_twin_evaluation_gates_the_mean_and_reports_each_head():
    branches = {
        "pool": [0, 0], "group": [0, 1], "sample": [0, 2], "position": [1, 2],
        "alternative_token": [30, 31], "prompt_sha256": ["a", "b"],
        "taken_rewards": [[1.0] * 32, [0.0] * 32], "alternative_rewards": [[0.0] * 32, [1.0] * 32],
        "continuations": [32, 32], "taken_logprob": [-0.1, -2.0], "alternative_logprob": [-3.0, -0.5],
    }
    study = _study(branches)
    critic = _randomized(study.build_critic(TWIN)[0])
    result = study.evaluate(critic, TWIN, step=3)
    scored = [study.score_group(critic, group, TWIN) for pool in study.heldout_pools for group in pool]
    targets = np.concatenate([
        np.full(len(tokens), reward)
        for pool in study.heldout_pools for group in pool
        for tokens, reward in zip(group.completions, group.rewards, strict=True)
    ])
    mean_q = np.concatenate([entry["q_taken"] for entry in scored])
    members = np.concatenate([entry["member_q_taken"] for entry in scored], axis=1)
    from postraining.critic_study import explained_variance
    assert result["ev_all_actions"] == pytest.approx(explained_variance(mean_q, targets))
    twin = result["twin"]
    assert twin["ev_all_actions"] == pytest.approx([explained_variance(q, targets) for q in members])
    assert twin["q_gap_rms_all_actions"] == pytest.approx(float(np.sqrt(np.mean((members[0] - members[1]) ** 2))))
    assert twin["branch_pairs"] == 2
    for key in ("q_gap_rms_think_actions", "advantage_gap_rms_think_actions",
                "advantage_gap_rms_branch_states", "branch_pair_gap_difference_rms"):
        assert np.isfinite(twin[key]) and twin[key] > 0, key
    # The single-head arm's evaluation carries no twin record.
    assert "twin" not in study.evaluate(study.build_critic(STUDY_ARMS[0])[0], STUDY_ARMS[0], step=0)


def test_twin_statistics_measure_head_disagreement():
    from postraining.critic_study import twin_statistics
    targets = np.array([0.0, 1.0, 0.0, 1.0])
    members = np.stack((targets, targets + np.array([0.0, 0.0, 2.0, 0.0])))
    stats = twin_statistics(members, targets, members[:, :2], targets[:2], np.array([3.0, -4.0]),
                            np.zeros(0), np.array([1.0]))
    assert stats["ev_all_actions"][0] == pytest.approx(1.0)
    assert stats["q_gap_rms_all_actions"] == pytest.approx(1.0)
    assert stats["q_gap_rms_think_actions"] == 0.0
    assert stats["advantage_gap_rms_think_actions"] == pytest.approx(np.sqrt(12.5))
    assert np.isnan(stats["advantage_gap_rms_branch_states"])
    assert stats["branch_pairs"] == 1
    with pytest.raises(ValueError, match="two heads"):
        twin_statistics(members[:1], targets, members[:1], targets, np.zeros(1), np.zeros(1), np.zeros(1))


def test_roll_branches_decodes_through_the_trainer_arena():
    from postraining.graph_decode import PinnedDecodeArena

    wrapper = _wrapper()
    arena = PinnedDecodeArena(
        wrapper, rows=4 * 2 * 3, kv_width=24, temperature=1.0, stop_ids=(STOP,),
        device=torch.device("cpu"), hidden_carry=False, cache_dtype=torch.float32,
        autocast_dtype=None, compile=False, cuda_graphs=False,
    )
    # One pair (two branch prompts) in an arena sized for four.
    prompts = [[1, 2, OPEN, 9], [1, 2, OPEN, 8, 7]]
    rows = roll_branches(arena, prompts, budget=6, continuations=3, seed=5, first_request=10)
    assert len(rows) == 2 * 3
    for row in rows:
        assert 1 <= len(row) <= 6
        assert STOP not in row[:-1]
    # Seeded by request identity: the same requests replay exactly, other
    # requests draw other noise.
    assert roll_branches(arena, prompts, 6, 3, 5, 10) == rows
    assert roll_branches(arena, prompts, 6, 3, 5, 12) != rows
    # A chunk with no budget left decodes nothing.
    assert roll_branches(arena, prompts, 0, 3, 5, 10) == [[]] * 6


def test_branch_capacity_covers_shared_prefix_and_budget_from_different_pairs():
    from postraining.graph_decode import PinnedDecodeArena

    # The longer prefix has the shorter remaining budget. A shared batch
    # needs max(prefix) + max(budget), rather than max(prefix + own_budget).
    pairs = [
        (BranchPair(0, 0, 0, 1, 9, 8, -1.0, -2.0), SimpleNamespace(prompt_tokens=[1, 2])),
        (BranchPair(0, 1, 0, 4, 9, 8, -1.0, -2.0), SimpleNamespace(prompt_tokens=[1, 2, 7, 8])),
    ]
    width = branch_arena_width(pairs, pairs_per_batch=2, max_new_tokens=8)
    assert width == 15  # longest prefix 9 + largest remaining budget 6
    assert branch_arena_width(pairs, pairs_per_batch=1, max_new_tokens=8) == 12
    prompts = [[1, 2, OPEN, 9], [1, 2, 7, 8, OPEN, 7, 8, 9, 9]]
    budgets = [continuation_budget(pair.position, 8) for pair, _ in pairs]
    assert budgets == [6, 3]

    def arena(capacity):
        return PinnedDecodeArena(
            _wrapper(), rows=4, kv_width=capacity, temperature=1.0, stop_ids=(STOP,),
            device=torch.device("cpu"), hidden_carry=False, cache_dtype=torch.float32,
            autocast_dtype=None, compile=False, cuda_graphs=False,
        )

    old_width = max(len(group.prompt_tokens) for _, group in pairs) + 8 + 1
    with pytest.raises(ValueError, match="exceeds the arena width"):
        roll_branches(arena(old_width), prompts, max(budgets), 1, 5, 0)
    rows = roll_branches(arena(width), prompts, max(budgets), 1, 5, 0)
    assert len(rows) == len(prompts)
    assert all(0 < len(row) <= max(budgets) for row in rows)
    assert len(truncate_continuation(rows[1], budgets[1], {STOP})) <= budgets[1]


def _branch_columns(observed_sign, prior_sign, successes=(30, 2), samples=32, prompts=20):
    """Branch records over ``prompts`` prompts whose taken arm wins when
    observed_sign is +1, with the policy prior pointing prior_sign."""
    rows = len(observed_sign)
    high, low = successes
    win = [[1.0] * high + [0.0] * (samples - high), [1.0] * low + [0.0] * (samples - low)]
    return {
        "prompt_sha256": [f"p{row % prompts}" for row in range(rows)],
        "taken_rewards": [win[0] if sign > 0 else win[1] for sign in observed_sign],
        "alternative_rewards": [win[1] if sign > 0 else win[0] for sign in observed_sign],
        "taken_logprob": [0.0] * rows,
        "alternative_logprob": [-float(sign) for sign in prior_sign],
    }


def test_branch_metrics_power_rule_reads_the_oracle_against_the_prior():
    rng = np.random.default_rng(0)
    observed = rng.choice([-1, 1], size=120)
    statistics = {"bootstrap": 400, "seed": 0, "alpha": 0.05}
    records = list(range(observed.size))

    def metrics(prior, predicted):
        holder = SimpleNamespace(branches=_branch_columns(observed, prior))
        return Study._branch_metrics(holder, np.asarray(predicted, dtype=float), records, statistics)

    # A prior right on half the pairs: the split-half oracle beats it, and
    # so does a critic that reads every gap.
    coin = np.where(np.arange(observed.size) % 2 == 0, observed, -observed)
    informative = metrics(coin, observed)
    assert informative["pairs_resolved"] == observed.size
    assert informative["oracle_beats_prior"]
    assert informative["critic_minus_prior"]["cluster_ci_low"] > 0
    assert gate_verdicts({"slope": 0.0, "slope_ci_low": 0.0, "informative_prompts": 0}, informative)["branch"]
    # A prior already right on every pair leaves nothing to beat: the
    # oracle ties it, and a perfect critic reads unpowered, not failed.
    saturated = metrics(observed, observed)
    assert not saturated["oracle_beats_prior"]
    assert saturated["critic_minus_prior"]["difference"] == 0.0
    assert gate_verdicts({"slope": 0.0, "slope_ci_low": 0.0, "informative_prompts": 0}, saturated)["branch"] is None
    # Gaps of 18 vs 14 of 32 do not resolve, and neither do their halves.
    unresolved = SimpleNamespace(branches=_branch_columns(observed, coin, successes=(18, 14)))
    weak = Study._branch_metrics(unresolved, observed.astype(float), records, statistics)
    assert weak["pairs_resolved"] == 0


def test_early_think_slope_reads_only_the_first_think_actions(tmp_path, monkeypatch):
    from postraining import critic_study

    # Fixture think spans run up to 11 actions; a cutoff of 2 must change
    # the early sum, and with it the slope.
    full = run_arm(_study(), STUDY_ARMS[0], tmp_path / "full")["final"]
    assert full["slope_early_think"]["slope"] == pytest.approx(full["slope"]["slope"])
    monkeypatch.setattr(critic_study, "EARLY_THINK_ACTIONS", 2)
    early = run_arm(_study(), STUDY_ARMS[0], tmp_path / "early")["final"]
    assert early["slope"]["slope"] == pytest.approx(full["slope"]["slope"])
    assert early["slope_early_think"]["slope"] != pytest.approx(early["slope"]["slope"])


@pytest.mark.parametrize("arm", STUDY_ARMS, ids=lambda arm: arm.name)
def test_run_arm_fits_and_reports_every_gate(tmp_path, arm):
    branches = {
        "pool": [0, 0], "group": [0, 1], "sample": [0, 2], "position": [1, 2],
        "alternative_token": [30, 31], "prompt_sha256": ["a", "b"],
        "taken_rewards": [[1.0] * 32, [0.0] * 32], "alternative_rewards": [[0.0] * 32, [1.0] * 32],
        "continuations": [32, 32], "taken_logprob": [-0.1, -2.0], "alternative_logprob": [-3.0, -0.5],
    }
    study = _study(branches)
    result = run_arm(study, arm, tmp_path / arm.name)
    records = [json.loads(line) for line in (tmp_path / arm.name / "metrics.jsonl").read_text().splitlines()]
    fits = [record for record in records if record["type"] == "fit"]
    # Two epochs over two pools, in the same seeded order for every arm.
    assert [record["step"] for record in fits] == [1, 2, 3, 4]
    order = np.random.default_rng(7)
    assert [record["pool"] for record in fits] == [int(pool) for _ in range(2) for pool in order.permutation(2)]
    heldout = [record for record in records if record["type"] == "heldout"]
    assert [record["step"] for record in heldout] == [0, 2, 4]
    (first,) = [record for record in records if record["type"] == "first_step_change"]
    assert first["q_change_rms"] > 0
    final = result["final"]
    assert final["branch"]["pairs_total"] == 2
    assert final["branch"]["pairs_resolved"] == 2
    assert set(final["gates"]) == {"slope", "branch"}
    assert final["think_actions"] > 0
    # Not gating: the early-think slope is reported beside the gate's.
    assert final["slope_early_think"]["trajectories"] == final["slope"]["trajectories"]
    assert np.isfinite(final["think_advantage_std"])
    assert (tmp_path / arm.name / "tensorboard").is_dir()

# CPU-only orchestration contracts: no model construction or execution.

def test_update_budget_caps_updates_without_rounding_to_epochs():
    from postraining.critic_study import update_budget
    assert update_budget(SimpleNamespace(max_updates=640, epochs=4), 32) == 640
    assert update_budget(SimpleNamespace(max_updates=129, epochs=4), 32) == 129
    assert update_budget(SimpleNamespace(max_updates=None, epochs=4), 32) == 128
    for budget, pools in ((0, 32), (-1, 32), (640, 0)):
        with pytest.raises(ValueError, match="positive"):
            update_budget(SimpleNamespace(max_updates=budget, epochs=4), pools)


def _futility_rows(evs, agreements, *, end=300):
    return [
        {"step": end - 180 + i * 60, "ev_think_actions": ev,
         "branch": {"critic": {"agreement": agreement}}}
        for i, (ev, agreement) in enumerate(zip(evs, agreements, strict=True))
    ]


def test_plateau_futility_requires_three_complete_windows_and_floor():
    from postraining.critic_study import plateau_futility
    rows = _futility_rows([.1, .101, .102, .103], [.6, .601, .602, .603])
    assert plateau_futility(rows)
    assert not plateau_futility(rows[1:])
    assert not plateau_futility(_futility_rows([.1] * 4, [.6] * 4, end=240))
    assert not plateau_futility([*rows, {**rows[-1], "step": 320}])
    assert not plateau_futility(_futility_rows([.1, .11, .111, .112], [.6] * 4))
    assert not plateau_futility(_futility_rows([.1] * 4, [.6, .62, .621, .622]))
    assert plateau_futility(_futility_rows([.2, .19, .18, .17], [.7, .69, .68, .67]))
    assert not plateau_futility(_futility_rows([.1, .1, float("nan"), .1], [.6] * 4))
    assert not plateau_futility([{**row, "branch": None} for row in rows])


def test_checkpoint_payload_preserves_optimizer_groups_bindings_and_cursor():
    from pathlib import Path
    from postraining.critic_study import checkpoint_payload
    optimizer = SimpleNamespace(
        state_dict=lambda: {"state": {17: {"momentum": 4}}},
        param_groups=[{"params": [17], "lr": 1e-4 / 512, "betas": (.9, .999)}],
    )
    study = SimpleNamespace(
        torch=SimpleNamespace(get_rng_state=lambda: "cpu-rng", cuda=SimpleNamespace(
            is_available=lambda: True, get_rng_state_all=lambda: ["cuda-rng"])),
        args=SimpleNamespace(max_updates=640, checkpoint=Path("base.pt")),
        checkpoint_bindings={"checkpoint_sha256": "base", "source_sha256": {"code.py": "source"}},
    )
    rng = np.random.default_rng(1337)
    order = rng.permutation(32)
    evaluation = {"step": 60, "gates": {"branch": False}}
    payload = checkpoint_payload(
        study, STUDY_ARMS[0], SimpleNamespace(state_dict=lambda: {"weights": "opaque"}),
        {"critic_action": optimizer}, step=60, epoch=1, order=order,
        next_pool=28, order_rng=rng, evaluations=[evaluation], stop_reason=None,
    )
    assert payload["schema"] == "token_q_critic_study_checkpoint/v2"
    assert payload["critic_schema"] == CRITIC_Q_SCHEMA
    assert payload["arm"]["q_twin"] is False
    twin = checkpoint_payload(
        study, next(arm for arm in STUDY_ARMS if arm.q_twin), SimpleNamespace(state_dict=lambda: {}),
        {}, step=60, epoch=1, order=order, next_pool=28, order_rng=rng, evaluations=[], stop_reason=None,
    )
    # One schema: the arm distinguishes a twin critic.
    assert twin["schema"] == payload["schema"]
    assert twin["arm"]["q_twin"] is True and twin["arm"]["critic_init"] == "actor"
    assert payload["critic"] == {"weights": "opaque"}
    assert payload["optimizers"]["critic_action"]["state"][17]["momentum"] == 4
    assert payload["optimizer_groups"]["critic_action"] == [{"lr": 1e-4 / 512, "betas": (.9, .999)}]
    assert payload["checkpoint_sha256"] == "base"
    assert payload["source_sha256"] == {"code.py": "source"}
    assert payload["config"]["checkpoint"] == "base.pt"
    assert payload["pool_order"] == order.tolist()
    assert payload["next_pool"] == 28
    assert payload["order_rng"] == rng.bit_generator.state
    assert payload["final_evaluation"] == evaluation
    assert payload["cpu_rng"] == "cpu-rng"
    assert payload["cuda_rng"] == ["cuda-rng"]
    assert {"python_rng", "numpy_rng"} <= payload.keys()


def test_atomic_checkpoint_failure_keeps_previous_complete_state(tmp_path):
    from postraining.critic_study import atomic_checkpoint
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("previous")
    def fail(payload, stream):
        stream.write(b"partial")
        raise OSError("disk full")
    with pytest.raises(OSError, match="disk full"):
        atomic_checkpoint(SimpleNamespace(save=fail), {}, checkpoint)
    assert checkpoint.read_text() == "previous"
    assert list(tmp_path.iterdir()) == [checkpoint]
    atomic_checkpoint(SimpleNamespace(save=lambda payload, stream: stream.write(b"complete")), {}, checkpoint)
    assert checkpoint.read_bytes() == b"complete"


def test_update_cap_keeps_historical_pool_order_and_immutable_recovery(tmp_path, monkeypatch):
    """Exercise scheduling and durable state with opaque fake models only."""
    import pickle
    from postraining import critic_study
    from torch.utils import tensorboard

    class Writer:
        def __init__(self, *args): pass
        def add_scalar(self, *args): pass
        def flush(self): pass
        def close(self): pass
    monkeypatch.setattr(tensorboard, "SummaryWriter", Writer)
    monkeypatch.setattr(critic_study, "_log_evaluation", lambda *args: None)
    seen = []
    state = SimpleNamespace(state_dict=lambda: {"opaque": "weights"})
    optimizer = SimpleNamespace(state_dict=lambda: {"opaque": "momentum"}, param_groups=[{"lr": 1e-7}])
    fake_torch = SimpleNamespace(
        get_rng_state=lambda: "rng", cuda=SimpleNamespace(is_available=lambda: False),
        save=lambda payload, stream: pickle.dump(payload, stream),
    )
    def evaluate(critic, arm, step):
        return {"step": step, "ev_think_actions": step / 1000,
                "branch": {"critic": {"agreement": .6}}, "gates": {"branch": False}}
    study = SimpleNamespace(
        args=SimpleNamespace(max_updates=129, epochs=4, eval_every=20, seed=1337),
        train_pools=list(range(32)), torch=fake_torch,
        actor_fingerprint=lambda: [(1., 2.)], build_critic=lambda arm: (state, {"critic": optimizer}),
        evaluate=evaluate, training_probe=lambda *args: {"q_mse": .1},
        probe_values=lambda *args: np.zeros((2, 3)),
        fit_pool=lambda critic, opts, pool, arm: seen.append(pool) or {"objective_loss": .2},
        last_branch_predictions=[],
    )
    result = critic_study.run_arm(study, STUDY_ARMS[0], tmp_path / "arm")
    rng = np.random.default_rng(1337)
    expected = np.concatenate([rng.permutation(32) for _ in range(5)]).tolist()
    assert seen == expected[:129]
    assert result["steps"] == 129
    assert result["stop_reason"] == "max_updates"
    checkpoints = sorted((tmp_path / "arm" / "checkpoints").glob("*.pt"))
    assert [int(path.stem.split("_")[1]) for path in checkpoints] == [0, 20, 40, 60, 80, 100, 120, 129]
    with (tmp_path / "arm" / "checkpoint.pt").open("rb") as stream:
        payload = pickle.load(stream)
    assert payload["epoch"] == 4
    assert payload["next_pool"] == 1
    assert payload["step"] == 129
    assert payload["stop_reason"] == "max_updates"
    assert payload["pool_order"] == expected[128:160]
    assert payload["frozen_actor_fingerprint"] == [(1., 2.)]
    assert (tmp_path / "arm" / "checkpoint.pt").stat().st_ino == checkpoints[-1].stat().st_ino
