"""Pure data and reduction checks; no model execution or GPU workload."""
import pytest
import torch

from postraining.frozen_critic_fit import (
    FitConfig, group_branch_states, group_prediction_rows, masked_regression_loss, normalize_branches,
    prepare_branch, prepare_rollout,
    truncate_journal,
)


def test_rollout_gold_prefix_shifts_only_state_positions():
    record = {"token_ids": [10, 11, 12, 13, 14], "reasoning_positions": [1, 2],
              "gold_context_ids": [8, 9], "reward": 1}
    prepared = prepare_rollout(record, True)
    assert prepared == {"tokens": [8, 9, 10, 11, 12, 13],
                        "positions": [3, 4], "actions": [12, 13], "reward": 1.0}
    plain = prepare_rollout(record, False)
    assert plain["positions"] == [1, 2]
    assert plain["actions"] == prepared["actions"]


@pytest.mark.parametrize("positions", [[], [-1], [3]])
def test_invalid_action_alignment_rejected(positions):
    with pytest.raises(ValueError, match="reasoning pre-action"):
        prepare_rollout({"token_ids": [10, 11, 12, 13],
                         "reasoning_positions": positions, "reward": 1}, False)


def test_gold_context_cannot_silently_be_missing():
    with pytest.raises(ValueError, match="gold_context_ids"):
        prepare_rollout({"token_ids": [10, 11], "reasoning_positions": [0],
                         "reward": 0}, True)


def test_branch_action_stays_out_of_causal_state():
    prepared = prepare_branch({"prefix_tokens": [10, 11], "forced_token": 12,
                               "gold_context_ids": [9]}, True)
    assert prepared["tokens"] == [9, 10, 11]
    assert prepared["positions"] == [2]
    assert prepared["actions"] == [12]


def test_pair_normalization_keeps_full_policy_for_both_actions():
    pair = {"pair_id": "x:2", "prompt_id": "x", "prefix_ids": [10, 11],
            "policy_probabilities": [.2, .8], "gold_context_ids": [9],
            "reference_rewards": [0, 1, 1],
            "actions": [{"token_id": 0, "rewards": [1, 0]},
                        {"token_id": 1, "rewards": [0, 0]}]}
    flat = normalize_branches([pair])
    assert [r["arm"] for r in flat] == [0, 1]
    assert [r["forced_token"] for r in flat] == [0, 1]
    assert all(r["policy_probabilities"] == [.2, .8] for r in flat)
    assert flat[0]["prefix_tokens"] == [10, 11]
    assert flat[0]["rewards"] == [1, 0]
    assert all(r["reference_rewards"] == [0, 1, 1] for r in flat)


def test_shared_state_groups_distinct_actions_once():
    shared = {"pair_id": "x:2", "prompt_id": "x", "prefix_tokens": [10, 11],
              "policy_probabilities": [.2, .8], "gold_context_ids": [9],
              "reference_rewards": [1, 0]}
    first = {**shared, "arm": 0, "forced_token": 0, "rewards": [1, 1]}
    second = {**shared, "arm": 1, "forced_token": 1, "rewards": [0, 0]}
    states = group_branch_states([second, first])
    assert len(states) == 1
    assert [r["forced_token"] for r in states[0]] == [0, 1]
    assert prepare_branch(states[0][0], True)["tokens"] == prepare_branch(states[0][1], True)["tokens"]


@pytest.mark.parametrize("field,value", [("prefix_tokens", [10, 12]),
                                         ("policy_probabilities", [.3, .7]),
                                         ("gold_context_ids", [8]),
                                         ("reference_rewards", [0, 0])])
def test_shared_state_rejects_inconsistent_metadata(field, value):
    shared = {"pair_id": "x:2", "prompt_id": "x", "prefix_tokens": [10, 11],
              "policy_probabilities": [.2, .8], "gold_context_ids": [9],
              "reference_rewards": [1, 0], "forced_token": 0, "rewards": [1, 1]}
    with pytest.raises(ValueError, match="shared state field"):
        group_branch_states([{**shared, "arm": 0}, {**shared, "arm": 1, field: value}])


def test_shared_state_rejects_missing_or_duplicate_arm():
    record = {"pair_id": "x:2", "prompt_id": "x", "arm": 0}
    with pytest.raises(ValueError, match="both action arms"):
        group_branch_states([record])
    with pytest.raises(ValueError, match="duplicate"):
        group_branch_states([record, record])


def test_group_predictions_requires_shared_reference_outcomes():
    base = {"pair_id": "x:2", "prompt_id": "x", "reference_rewards": [1, 0],
            "forced_token": 4, "q": .5, "advantage": .1, "rewards": [1, 1]}
    flat = [{**base, "arm": 0}, {**base, "arm": 1, "forced_token": 5}]
    grouped = group_prediction_rows(flat)
    assert grouped[0]["reference_rewards"] == [1, 0]
    assert len(grouped[0]["actions"]) == 2
    flat[1]["reference_rewards"] = [0, 0]
    with pytest.raises(ValueError, match="share"):
        group_prediction_rows(flat)


def test_loss_weighting_changes_only_rollout_length_influence():
    values = torch.tensor([[1., 99.], [3., 3.]])
    targets = torch.tensor([0., 0.])
    mask = torch.tensor([[1., 0.], [1., 1.]])
    assert masked_regression_loss(values, targets, mask, "rollout").item() == 5
    assert masked_regression_loss(values, targets, mask, "token").item() == pytest.approx(19 / 3)


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"loss_weighting": "unknown"},
                                     {"head_learning_rate": float("nan")}])
def test_invalid_study_config_rejected(kwargs):
    with pytest.raises(ValueError):
        FitConfig(**kwargs)


def test_resume_truncates_only_uncommitted_journal_rows(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 20, "loss": 1}\n{"step": 40, "loss": 2}\n{"step": 60')
    truncate_journal(path, 20)
    assert path.read_text() == '{"step": 20, "loss": 1}\n'


def test_resume_rejects_nonmonotonic_and_middle_corruption(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 20}\n{"step": 20}\n')
    with pytest.raises(ValueError, match="nonmonotonic"):
        truncate_journal(path, 20)
    path.write_text('{"step": 20}\nno json\n{"step": 40}\n')
    with pytest.raises(ValueError, match="corrupt"):
        truncate_journal(path, 20)
