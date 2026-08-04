from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "probe_qwen3_answer_embeddings.py"
SPEC = importlib.util.spec_from_file_location("probe_qwen3_answer_embeddings", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_last_token_pool_handles_left_and_right_padding():
    hidden = torch.tensor(
        [
            [[10.0], [11.0], [12.0], [13.0]],
            [[20.0], [21.0], [22.0], [23.0]],
        ]
    )
    left_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    right_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

    assert MODULE.last_token_pool(hidden, left_mask).tolist() == [[13.0], [23.0]]
    assert MODULE.last_token_pool(hidden, right_mask).tolist() == [[11.0], [22.0]]


def test_truncate_and_normalize_implements_mrl_readout():
    embeddings = torch.tensor([[3.0, 4.0, 100.0], [0.0, 2.0, -50.0]])

    actual = MODULE.truncate_and_normalize(embeddings, 2)

    torch.testing.assert_close(actual, torch.tensor([[0.6, 0.8], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        MODULE.truncate_and_normalize(embeddings, 4)


def test_dimension_metadata_labels_below_official_mrl_minimum_as_extrapolation():
    assert MODULE.dimension_metadata(16, 4096) == {
        "value": 16,
        "official_mrl_supported": False,
        "status": "below_official_minimum_extrapolation",
    }
    assert MODULE.dimension_metadata(32, 4096)["official_mrl_supported"] is True
    with pytest.raises(ValueError):
        MODULE.dimension_metadata(4097, 4096)


def test_cached_scorer_can_use_asymmetric_target_embeddings():
    plain = {
        "target": torch.tensor([1.0, 0.0]),
        "same": torch.tensor([1.0, 0.0]),
        "other": torch.tensor([0.0, 1.0]),
    }
    instructed = {"target": torch.tensor([0.0, 1.0])}
    scorer = MODULE.CachedQwenScorer(
        plain, instructed, dimension=2, mode="retrieval_target_instructed"
    )

    target = scorer.preencode_target("target")
    cosine = scorer.cosine(["same", "other"], target)
    rewards = scorer.score(["same", "other"], target)

    torch.testing.assert_close(cosine, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(
        rewards,
        torch.tensor([torch.exp(torch.tensor(-10.0)), 1.0]),
    )


def test_case_metrics_rewards_graded_order_and_penalizes_inversion():
    ordered = MODULE._case_metrics([1.0, 0.5, 0.0], [0.9, 0.4, -0.2])
    inverted = MODULE._case_metrics([1.0, 0.5, 0.0], [-0.2, 0.4, 0.9])

    assert ordered["spearman"] == pytest.approx(1.0)
    assert ordered["pairwise_correct"] == pytest.approx(1.0)
    assert ordered["correct_min_minus_incorrect_max"] == pytest.approx(0.5)
    assert inverted["spearman"] == pytest.approx(-1.0)
    assert inverted["pairwise_correct"] == pytest.approx(0.0)


def test_graded_suite_includes_systematic_numeric_and_code_cases():
    cases = MODULE.build_graded_cases()
    names = {case.name for case in cases}

    assert len([name for name in names if name.startswith("numeric_sweep_")]) == 41
    assert "independent_functions" in names
    assert "factorial_program" in names
    assert "linear_equation" in names
