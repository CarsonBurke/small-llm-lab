"""Statistical contract checks; no model workloads."""

import pytest

from postraining.frozen_critic_gates import evaluate_predictions


def row(prompt, state=0, sign=1, rewards=None):
    rewards = rewards or ([0] * 32, [1] * 32)
    return {"prompt_id": prompt, "state_id": state, "reference_rewards": [0, 1] * 16, "actions": [
        {"token_id": 10, "q": .1 * sign, "advantage": -.1 * sign, "rewards": rewards[0]},
        {"token_id": 20, "q": .3 * sign, "advantage": .1 * sign, "rewards": rewards[1]},
    ]}


def evaluate(rows):
    return evaluate_predictions(rows, bootstrap_samples=200, permutation_samples=200)


def test_aligned_branch_predictions_pass_with_independent_prompts():
    report = evaluate([row(index) for index in range(20)])
    assert report["passed"]
    assert report["sign_agreement_ci95"] == [1., 1.]
    assert report["prompt_fixed_slope"] == pytest.approx(5.)
    assert report["state_fixed_slope"] == pytest.approx(5.)
    assert report["arm_swap_pvalue_one_sided"] < .05
    assert report["resolved_pair_fraction_descriptive"] == 1


def test_reversed_predictions_and_ties_do_not_pass():
    reversed_report = evaluate([row(index, sign=-1) for index in range(20)])
    assert not reversed_report["passed"]
    assert reversed_report["sign_agreement"] == 0
    tied_report = evaluate([row(index, rewards=([0, 1], [1, 0])) for index in range(20)])
    assert tied_report["sign_agreement"] == .5
    assert not tied_report["passed"]


def test_pair_ranking_alone_cannot_pass_centered_sign_gate():
    rows = [row(index) for index in range(20)]
    for entry in rows:
        del entry["reference_rewards"]
    report = evaluate(rows)
    assert report["pairwise_sign_agreement"] == 1
    assert report["sign_agreement"] is None
    assert report["sign_agreement_ci95"] is None
    assert not report["passed"]


def test_centered_sign_differs_from_pairwise_ranking():
    rows = [row(index) for index in range(20)]
    for entry in rows:
        # Pairwise ranking is perfect, but both actions have positive A;
        # the lower action empirically falls below the policy reference.
        for action in entry["actions"]:
            action["advantage"] += .2
    report = evaluate(rows)
    assert report["pairwise_sign_agreement"] == 1
    assert report["sign_agreement"] == .5
    assert not report["passed"]


def test_missing_one_reference_or_invalid_reference_cannot_pass():
    rows = [row(index) for index in range(20)]
    del rows[0]["reference_rewards"]
    assert evaluate(rows)["sign_agreement"] is None
    rows[0]["reference_rewards"] = []
    with pytest.raises(ValueError, match="reference rewards"):
        evaluate(rows)


def test_many_states_of_one_prompt_are_not_independent_prompts():
    report = evaluate([row("same", state=index) for index in range(40)])
    assert report["prompts"] == 1
    assert report["status"] == "inconclusive"


def test_prompt_baselines_cannot_create_fixed_effect_slope():
    rows = []
    for index in range(20):
        entry = row(index, rewards=([index % 2] * 4, [index % 2] * 4))
        for action in entry["actions"]:
            action["q"] += index % 2
            action["advantage"] += index % 2
        rows.append(entry)
    assert evaluate(rows)["prompt_fixed_slope"] == 0


def test_duplicate_tokens_and_empty_input_are_inconclusive():
    entry = row(0)
    entry["actions"][1]["token_id"] = 10
    entry["actions"][1]["q"] = entry["actions"][0]["q"]
    entry["actions"][1]["advantage"] = entry["actions"][0]["advantage"]
    report = evaluate([entry])
    assert report["pairs"] == 1
    assert report["prompts"] == 1
    assert report["duplicate_action_pairs"] == 1
    assert report["pairwise_sign_agreement"] == .5
    assert report["sign_agreement"] == .5
    assert report["state_fixed_slope"] is None
    assert report["status"] == "inconclusive"
    assert evaluate([])["sign_agreement"] is None


def test_identical_actions_retain_centered_sign_information():
    rows = [row(index, rewards=([0] * 32, [0] * 32)) for index in range(20)]
    for entry in rows:
        entry["actions"][1].update(token_id=10, q=.1, advantage=-.1)
    report = evaluate(rows)
    assert report["prompts"] == 20
    assert report["sign_agreement"] == 1
    assert report["state_fixed_slope_ci95"] is None
    assert not report["passed"]


def test_undefined_slope_is_not_a_zero_or_pass():
    rows = [row(index) for index in range(20)]
    for entry in rows:
        for action in entry["actions"]:
            action["q"], action["advantage"] = .2, 0.
    report = evaluate(rows)
    assert report["sign_agreement"] == .5
    assert report["prompt_fixed_slope"] is None
    assert report["prompt_fixed_slope_ci95"] is None
    assert not report["passed"]


def test_validation_rejects_nonbinary_labels_and_inconsistent_baselines():
    entry = row(0)
    entry["actions"][0]["rewards"] = [.5]
    with pytest.raises(ValueError, match="binary"):
        evaluate([entry])
    entry = row(0)
    entry["actions"][0]["advantage"] += 1
    with pytest.raises(ValueError, match="baseline"):
        evaluate([entry])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate([row(0), row(0)])


def test_reproducible_and_power_is_reported_honestly():
    rows = [row(index, sign=1 if index % 3 else -1) for index in range(20)]
    assert evaluate(rows) == evaluate(rows)
    report = evaluate(rows)
    assert report["power"]["median_worst_case_pair_se"] == pytest.approx(.125)
    assert report["power"]["continuations_per_arm_for_0_001_se_at_p_half"] == 500000
