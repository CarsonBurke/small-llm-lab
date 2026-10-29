"""Static fixtures: never load a checkpoint or execute a model."""

import json

from scripts.dg_qtarget_study import build_report, read_metrics, train_diagnostics


def test_partial_jsonl_preserves_complete_observations(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"type":"bpb","step":0,"val_bpb":2.0}\n{"step":')
    rows, warnings = read_metrics(path)
    assert rows == [{"type": "bpb", "step": 0, "val_bpb": 2.0}]
    assert len(warnings) == 1


def test_group_moments_are_weighted_and_dashboard_is_authoritative():
    groups = [
        {"emit_action_count": 10, "action_count": 10, "delightful_gate_mean": 0.6,
         "advantage_mean": 0, "advantage_std": 1, "policy_entropy_count": 10, "policy_entropy_sum": 20},
        {"emit_action_count": 30, "action_count": 30, "delightful_gate_mean": 0.8,
         "advantage_mean": 2, "advantage_std": 1, "policy_entropy_count": 30, "policy_entropy_sum": 120},
    ]
    result = train_diagnostics({"group_metrics": groups})
    assert result["gate_mean"] == 0.75
    assert result["entropy"] == 3.5
    assert result["advantage_std"] == (1.75)**0.5
    assert "q_mc_ev" not in result
    assert train_diagnostics({"group_metrics": groups, "dashboard": {"delightful/gate_mean": 0.7}})["gate_mean"] == 0.7


def test_matched_steps_do_not_compare_unmatched_latest_or_count_warmup_as_joint(tmp_path):
    arms = {}
    for name, last in (("adaptive", 20), ("mc", 40)):
        directory = tmp_path / name
        directory.mkdir()
        rows = [{"type": "value_warmup", "step": 50},
                {"type": "bpb", "step": 0, "val_bpb": 2.0},
                {"type": "train", "step": last, "dashboard": {}},
                {"type": "bench", "step": 20, "accuracy": 0.1}]
        if last == 40:
            rows.append({"type": "bench", "step": 40, "accuracy": 0.2})
        (directory / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        arms[name] = {"output": str(directory), "q_critic_lambda": None if name == "adaptive" else 1.0}
    study = tmp_path / "study.json"
    study.write_text(json.dumps({"arms": arms, "critic_warmup_updates": 50, "joint_updates": 950, "total_updates": 1000}))
    report = build_report(study)
    assert report["arms"]["adaptive"]["observed_total_updates"] == 70
    assert report["arms"]["mc"]["observed_total_updates"] == 90
    assert [r["joint_step"] for r in report["matched_steps"]["bench"]] == [20]
    assert report["matched_steps"]["bench"][0]["total_updates"] == 70
    assert report["matched_steps"]["bpb"][0]["total_updates"] == 0
    assert not report["matched_steps"]["train"]


def test_missing_arms_are_reported_without_fabricated_zero_scores(tmp_path):
    study = tmp_path / "study.json"
    study.write_text(json.dumps({"arms": {"mc": {"output": "missing"}}, "critic_warmup_updates": 50, "joint_updates": 950, "total_updates": 1000}))
    report = build_report(study)
    assert report["arms"]["mc"]["warnings"]
    assert all(not rows for rows in report["matched_steps"].values())


def test_contract_scores_and_source_confounds_are_preserved(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    rows = [
        {"type": "bench", "step": 20, "accuracy": 0.4, "contract_accuracy": 0.1,
         "structural_format_fraction": 0.5, "module_accuracy": {"arithmetic": 0.4},
         "prompt_correct_counts": [2, 0], "contract_prompt_correct_counts": [0, 0]},
        {"type": "rollout", "step": 20, "reward_mean": 0.03,
         "source_metrics": {"science_mc": {"reward_mean": 0.02,
             "think_gate_zeroed_correct_fraction": 0.2, "think_tokens_mean": 20}}},
    ]
    (output / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    study = tmp_path / "study.json"
    study.write_text(json.dumps({"arms": {"mc": {"output": str(output)}}, "critic_warmup_updates": 50, "joint_updates": 950, "total_updates": 1000}))
    report = build_report(study)
    assert report["matched_steps"]["bench_contract"][0]["arms"]["mc"] == 0.1
    assert report["arms"]["mc"]["evaluation_context"]["bench"][20]["prompt_correct_counts"] == [2, 0]
    assert report["matched_steps"]["rollout_sources"][0]["arms"]["mc"]["science_mc"]["think_gate_zeroed_correct_fraction"] == 0.2


def test_empty_action_groups_produce_no_variance():
    assert "advantage_std" not in train_diagnostics({"group_metrics": [{"action_count": 0, "advantage_mean": 1, "advantage_std": 2}]})


def test_final_evaluation_merges_without_advancing_training_budget(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "metrics.jsonl").write_text(json.dumps({"type": "bench", "step": 940, "accuracy": 0.1}) + "\n")
    final = output / "final_eval" / "bench"
    final.mkdir(parents=True)
    (final / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [
        {"type": "bench", "step": 950, "accuracy": 0.2, "contract_accuracy": 0.15},
        {"type": "train", "step": 950, "dashboard": {}},
    ]))
    aime = output / "final_eval" / "aime"
    aime.mkdir(parents=True)
    (aime / "metrics.jsonl").write_text(json.dumps({"type": "aime", "step": 950, "accuracy": 0.05, "contract_accuracy": 0.04}) + "\n")
    study = tmp_path / "study.json"
    study.write_text(json.dumps({"arms": {"mc": {"output": str(output)}}, "critic_warmup_updates": 50, "joint_updates": 950, "total_updates": 1000}))
    report = build_report(study)
    assert report["arms"]["mc"]["observed_joint_updates"] == 0
    assert report["matched_steps"]["bench"][-1]["joint_step"] == 950
    assert report["final_budget_comparison"]["matched_evaluations_available"]["bench_contract"]
    assert report["matched_steps"]["aime_contract"][0]["arms"]["mc"] == 0.04
    assert not report["final_budget_comparison"]["ready"]
    assert not report["arms"]["mc"]["warnings"]
