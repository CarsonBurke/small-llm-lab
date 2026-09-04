from copy import deepcopy

import pytest
import torch

from postraining.critic_pretraining import (
    POSITION_BUCKET_NAMES,
    build_output_checkpoint,
    deterministic_stratified_positions,
    evaluate_acceptance_gate,
    position_bucket_metrics,
    split_records_by_prompt,
    stratified_response_position_samples,
    validate_critic_corpus,
    validate_corpus_checkpoint_identity,
    weighted_explained_variance,
    weighted_mse,
)


def _record(prompt_index: int, *, correct: bool = False) -> dict:
    return {
        "prompt_index": prompt_index,
        "prompt_ids": torch.tensor([10, 11], dtype=torch.int32),
        "response_ids": torch.tensor([20, 21, 22], dtype=torch.int32),
        "correct": correct,
    }


def _corpus() -> dict:
    return {
        "schema": "minicpm_critic_corpus/v1",
        "model_id": "openbmb/MiniCPM5-1B",
        "revision": "revision",
        "source_checkpoint": "/checkpoints/source.pt",
        "source_checkpoint_step": 12,
        "data_sha256": "a" * 64,
        "cursor_start": 48,
        "cursor_end": 50,
        "prompt_tokens": 8,
        "max_new_tokens": 16,
        "samples_per_prompt": 2,
        "seed": 7,
        "records": [
            _record(4, correct=False),
            _record(4, correct=True),
            _record(9, correct=False),
            _record(9, correct=True),
        ],
    }


def test_stratified_sampling_is_deterministic_and_respects_every_stratum_boundary():
    first = stratified_response_position_samples(17, 5, seed=91)
    second = stratified_response_position_samples(17, 5, seed=91)

    assert first == second
    assert deterministic_stratified_positions(17, 5, 91) == tuple(
        sample.position for sample in first
    )
    for stratum, sample in enumerate(first):
        start = stratum * 17 // 5
        stop = (stratum + 1) * 17 // 5
        assert start <= sample.position < stop
        assert sample.weight == stop - start


def test_stratified_sampling_keeps_all_boundary_states_when_budget_allows():
    samples = stratified_response_position_samples(4, 256, seed=3)

    assert [sample.position for sample in samples] == [0, 1, 2, 3]
    assert [sample.weight for sample in samples] == [1.0] * 4


def test_prompt_split_is_deterministic_and_has_no_prompt_leakage():
    records = [
        {"prompt_index": prompt_index, "trajectory": trajectory}
        for prompt_index in range(10)
        for trajectory in range(3)
    ]

    train, validation = split_records_by_prompt(records, 0.2, seed=123)
    train_again, validation_again = split_records_by_prompt(records, 0.2, seed=123)

    train_prompts = {record["prompt_index"] for record in train}
    validation_prompts = {record["prompt_index"] for record in validation}
    assert train_prompts.isdisjoint(validation_prompts)
    assert train_prompts | validation_prompts == set(range(10))
    assert len(validation_prompts) == 2
    assert train == train_again
    assert validation == validation_again
    for prompt_index in range(10):
        assert sum(record["prompt_index"] == prompt_index for record in train + validation) == 3


def test_weighted_mse_and_explained_variance_use_weighted_residual_variance():
    targets = torch.tensor([-1.0, 1.0])
    predictions = torch.tensor([-0.5, 0.5])
    weights = torch.tensor([1.0, 3.0])

    assert weighted_mse(predictions, targets, weights) == pytest.approx(0.25)
    assert weighted_explained_variance(predictions, targets, weights) == pytest.approx(
        0.75
    )


def test_position_bucket_metrics_report_all_four_boundary_buckets():
    relative_positions = torch.tensor(
        [0.0, 0.2499, 0.25, 0.4999, 0.5, 0.7499, 0.75, 1.0]
    )
    targets = torch.tensor([-1.0, 1.0] * 4)
    predictions = targets.clone()
    weights = torch.ones(8)

    metrics = position_bucket_metrics(
        predictions, targets, weights, relative_positions
    )

    assert tuple(metrics) == POSITION_BUCKET_NAMES
    for bucket in metrics.values():
        assert bucket["count"] == 2
        assert bucket["weight"] == 2.0
        assert bucket["weighted_mse"] == 0.0
        assert bucket["explained_variance"] == 1.0
        assert bucket["class_margin"] == 2.0


def test_acceptance_gate_requires_aggregate_signal_and_stable_position_buckets():
    metrics = {
        "weighted_mse": 0.4,
        "target_variance": 0.9,
        "explained_variance": 0.25,
        "class_margin": 0.5,
        "buckets": {
            name: {"explained_variance": 0.1} for name in POSITION_BUCKET_NAMES
        },
    }

    assert evaluate_acceptance_gate(metrics).passed

    unstable = deepcopy(metrics)
    unstable["buckets"]["75-100%"]["explained_variance"] = -0.2
    result = evaluate_acceptance_gate(unstable)
    assert not result.passed
    assert any("75-100%" in reason for reason in result.reasons)


def test_corpus_schema_validation_accepts_contract_and_rejects_bad_records():
    corpus = _corpus()
    assert validate_critic_corpus(corpus) is corpus

    bad_dtype = deepcopy(corpus)
    bad_dtype["records"][0]["response_ids"] = torch.tensor(
        [20, 21], dtype=torch.int64
    )
    with pytest.raises(ValueError, match="CPU int32 vector"):
        validate_critic_corpus(bad_dtype)

    bad_count = deepcopy(corpus)
    bad_count["records"].pop()
    with pytest.raises(ValueError, match="sample counts"):
        validate_critic_corpus(bad_count)

    missing = deepcopy(corpus)
    del missing["revision"]
    with pytest.raises(ValueError, match="missing fields: revision"):
        validate_critic_corpus(missing)


def test_checkpoint_identity_and_output_advance_only_consumed_cursor():
    corpus = _corpus()
    source = {
        "policy": {
            "schema": "minicpm5_vapo_adapter/v6",
            "actor": {
                "adapter": {"actor": torch.tensor([1.0])},
                "nextlat": {"actor": torch.tensor([3.0])},
            },
            "critic": {
                "model_id": corpus["model_id"],
                "revision": corpus["revision"],
                "adapter": {"critic": torch.tensor([1.0])},
                "value_head": {"old": torch.tensor([2.0])},
                "nextlat": {"critic": torch.tensor([3.0])},
            },
        },
        "step": corpus["source_checkpoint_step"],
        "cursor": corpus["cursor_start"],
        "pending_records": None,
        "pending_epoch": 0,
        "data_sha256": corpus["data_sha256"],
        "cpu_rng": torch.tensor([4], dtype=torch.uint8),
        "cuda_rng": torch.tensor([5], dtype=torch.uint8),
        "python_rng": (3, (6,), None),
        "critic_optimizer": {
            "state": {"old": torch.tensor([8.0])},
            "param_groups": [{"params": [0], "lr": 1e-4, "weight_decay": 0.0}],
        },
    }
    validate_corpus_checkpoint_identity(
        corpus, source, corpus["source_checkpoint"]
    )

    output = build_output_checkpoint(
        source,
        {"new": torch.tensor([7.0])},
        {
            "state": {0: {"step": torch.tensor(2.0)}},
            "param_groups": [{"params": [0], "lr": 3e-4, "weight_decay": 0.1}],
        },
        {"schema": "metadata"},
        cursor_end=corpus["cursor_end"],
    )

    assert output["cursor"] == corpus["cursor_end"]
    assert output["step"] == source["step"]
    assert output["policy"]["actor"] is source["policy"]["actor"]
    assert output["policy"]["critic"]["nextlat"] is source["policy"]["critic"]["nextlat"]
    assert output["cpu_rng"] is source["cpu_rng"]
    assert output["cuda_rng"] is source["cuda_rng"]
    assert output["python_rng"] is source["python_rng"]
    assert output["critic_optimizer"]["param_groups"] == [
        {"params": [0], "lr": 1e-4, "weight_decay": 0.0}
    ]
    assert output["critic_optimizer"]["state"] == {}
    assert output["policy"]["critic"]["value_head"]["new"].item() == 7.0
    assert source["cursor"] == corpus["cursor_start"]
