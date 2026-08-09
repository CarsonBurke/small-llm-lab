"""Hand-computed CPU tests for byte-diffusion targets and reductions."""

from __future__ import annotations

import math

import pytest
import torch

from pretraining.byte_diffusion.objectives import (
    IGNORE_INDEX,
    blt_masked_loss,
    canvas_cross_entropy,
    cross_entropy_per_row,
    introspection_balanced_loss,
    joint_canvas_ar_loss,
    same_position_targets,
    shifted_ar_targets,
)


def _logits_with_true_nll(nll: torch.Tensor) -> torch.Tensor:
    """Return two-way normalized logits whose class-zero NLL is exact."""

    probability = torch.exp(-nll)
    return torch.stack((probability.log(), (1.0 - probability).log()), dim=-1)


def test_shifted_ar_targets_match_byte_and_special_alignment() -> None:
    ids = torch.tensor([[256, 65, 195, 169, 256, 262]])
    valid = torch.tensor([[True, True, True, True, True, False]])
    observed = shifted_ar_targets(ids, valid)
    torch.testing.assert_close(
        observed,
        torch.tensor([[65, 195, 169, 256, IGNORE_INDEX, IGNORE_INDEX]]),
    )


def test_shifted_targets_do_not_bridge_internal_document_padding() -> None:
    ids = torch.tensor([[256, 65, 262, 256, 66]])
    valid = torch.tensor([[True, True, False, True, True]])
    score = torch.tensor([[False, True, False, False, True]])
    observed = shifted_ar_targets(ids, valid, score_mask=score)
    torch.testing.assert_close(
        observed,
        torch.tensor([[65, IGNORE_INDEX, IGNORE_INDEX, 66, IGNORE_INDEX]]),
    )


def test_same_position_targets_select_only_corrupted_bytes() -> None:
    clean = torch.tensor([[65, 195, 169, 256, 262]])
    active = torch.tensor([[False, True, False, True, False]])
    observed = same_position_targets(clean, active)
    torch.testing.assert_close(
        observed,
        torch.tensor([[IGNORE_INDEX, 195, IGNORE_INDEX, 256, IGNORE_INDEX]]),
    )


def test_uniform_261_way_logits_have_log_vocab_cross_entropy() -> None:
    logits = torch.zeros(1, 4, 261, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[65, 195, 256, IGNORE_INDEX]])
    rows = cross_entropy_per_row(logits, targets)
    torch.testing.assert_close(
        rows.mean, torch.tensor([math.log(261)], dtype=torch.float64)
    )
    assert rows.count.item() == 3
    rows.mean.sum().backward()
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad[:, -1], torch.zeros_like(logits.grad[:, -1]))


def test_bfloat16_cross_entropy_is_reduced_in_float32() -> None:
    logits = torch.zeros(1, 8192, 261, dtype=torch.bfloat16)
    targets = torch.zeros(1, 8192, dtype=torch.long)
    rows = cross_entropy_per_row(logits, targets)
    assert rows.total.dtype == torch.float32
    assert float(rows.mean) == pytest.approx(math.log(261), rel=1e-6)


def test_canvas_loss_weights_canvases_equally_instead_of_weighting_by_k() -> None:
    desired_nll = torch.tensor(
        [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]], dtype=torch.float64
    )
    logits = _logits_with_true_nll(desired_nll)
    targets = torch.zeros(2, 3, dtype=torch.long)
    active = torch.tensor([[True, False, False], [True, True, True]])
    objective = canvas_cross_entropy(logits, targets, active)
    torch.testing.assert_close(
        objective.per_canvas, torch.tensor([1.0, 3.0], dtype=torch.float64)
    )
    torch.testing.assert_close(objective.loss, torch.tensor(2.0, dtype=torch.float64))
    # A global active-token mean would be 2.5 and is a different estimator.
    assert objective.loss != torch.tensor(2.5, dtype=torch.float64)


def test_joint_objective_adds_mean_canvas_and_mean_ar_losses() -> None:
    canvas_logits = _logits_with_true_nll(
        torch.tensor([[1.0], [3.0]], dtype=torch.float64)
    )
    canvas_targets = torch.zeros(2, 1, dtype=torch.long)
    canvas_active = torch.ones(2, 1, dtype=torch.bool)
    ar_logits = _logits_with_true_nll(torch.tensor([[2.0, 4.0]], dtype=torch.float64))
    ar_targets = torch.zeros(1, 2, dtype=torch.long)
    objective = joint_canvas_ar_loss(
        canvas_logits,
        canvas_targets,
        canvas_active,
        ar_logits,
        ar_targets,
        lambda_ar=0.5,
    )
    torch.testing.assert_close(
        objective.diffusion, torch.tensor(2.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        objective.ar, torch.tensor(3.0, dtype=torch.float64)
    )
    torch.testing.assert_close(objective.total, torch.tensor(3.5, dtype=torch.float64))
    assert objective.diffusion_targets.item() == 2
    assert objective.ar_targets.item() == 2


def test_blt_masked_loss_is_zero_safe_and_uses_inverse_t_on_sums() -> None:
    logits = _logits_with_true_nll(
        torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float64)
    )
    targets = torch.zeros(2, 2, dtype=torch.long)
    active = torch.tensor([[False, False], [True, False]])
    objective = blt_masked_loss(logits, targets, active, t=torch.tensor(0.5))
    torch.testing.assert_close(
        objective.masked_total, torch.tensor([0.0, 2.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        objective.per_row, torch.tensor([0.0, 4.0], dtype=torch.float64)
    )
    torch.testing.assert_close(objective.loss, torch.tensor(2.0, dtype=torch.float64))
    torch.testing.assert_close(objective.masked_count, torch.tensor([0, 1]))


def test_cross_entropy_empty_rows_have_exact_zero_value_and_gradient() -> None:
    logits = torch.randn(1, 3, 5, dtype=torch.float64, requires_grad=True)
    targets = torch.zeros(1, 3, dtype=torch.long)
    active = torch.zeros(1, 3, dtype=torch.bool)
    rows = cross_entropy_per_row(logits, targets, active=active)
    torch.testing.assert_close(rows.mean, torch.zeros_like(rows.mean))
    rows.mean.sum().backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))
    with pytest.raises(ValueError, match="at least one active"):
        canvas_cross_entropy(logits.detach(), targets, active)


def test_introspection_balance_detaches_the_ratio() -> None:
    masked = torch.tensor(2.0, requires_grad=True)
    clean = torch.tensor(4.0, requires_grad=True)
    total, coefficient = introspection_balanced_loss(masked, clean)
    torch.testing.assert_close(coefficient, torch.tensor(0.5))
    torch.testing.assert_close(total, torch.tensor(4.0))
    total.backward()
    torch.testing.assert_close(masked.grad, torch.tensor(1.0))
    torch.testing.assert_close(clean.grad, torch.tensor(0.5))
    assert not coefficient.requires_grad
