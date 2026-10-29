"""Toy readout tensors only; no backbone, model, training, or GPU workload."""

import pytest
import torch

from postraining.nano_backbone import (
    SoftcappedTargetLogprobs,
    SoftcappedTargetLogprobsEntropy,
    SoftcappedTargetLogprobsEntropyAdvantages,
)


CAP = 15.0


def reference(raw, targets):
    # Float64 native autograd is independent of the custom backward.
    capped = CAP * raw / torch.sqrt(raw.square() + CAP**2)
    logp = capped.log_softmax(-1)
    selected = logp.gather(-1, targets[:, None]).squeeze(-1)
    entropy = -(logp.exp() * logp).sum(-1)
    return selected, entropy


def rows(dtype):
    return torch.tensor([
        [1000., -1000., -1000., -1000., -1000., -1000., -1000., -1000.],
        [9., 9., 9., -9., 0., 1., -2., 3.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [5., -3., 2., -1., 0., 7., -8., 4.],
    ], dtype=dtype)


def outputs(raw, targets, variant):
    if variant == 'target':
        return (SoftcappedTargetLogprobs.apply(raw, targets, CAP),)
    if variant == 'entropy':
        return SoftcappedTargetLogprobsEntropy.apply(raw, targets, CAP)
    # Deterministic, small Q factors exercise the third readout's additional
    # detached statistic without constructing a model or optimizer.
    directions = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12
    latents = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 24
    return SoftcappedTargetLogprobsEntropyAdvantages.apply(
        raw, targets, CAP, directions, latents
    )


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize('variant', ['target', 'entropy', 'advantages'])
def test_forward_and_backward_match_float64_reference(dtype, variant):
    raw = rows(dtype).requires_grad_()
    targets = torch.tensor([0, 2, 3, 5])
    actual = outputs(raw, targets, variant)
    ref_raw = raw.detach().double().requires_grad_()
    ref_selected, ref_entropy = reference(ref_raw, targets)
    forward_rtol = 1e-11 if dtype == torch.float64 else 2e-6
    forward_atol = 1e-12 if dtype == torch.float64 else 5e-7
    selected_weights = torch.tensor([.7, -.2, .5, 1.3], dtype=torch.float64)
    entropy_weights = torch.tensor([.3, .1, -.4, .2], dtype=torch.float64)
    actual_loss = (actual[0] * selected_weights.to(actual[0].dtype)).sum()
    ref_loss = (ref_selected * selected_weights).sum()
    if variant != 'target':
        torch.testing.assert_close(
            actual[1].double(), ref_entropy, rtol=forward_rtol, atol=forward_atol
        )
        actual_loss = actual_loss + (actual[1] * entropy_weights.to(actual[1].dtype)).sum()
        ref_loss = ref_loss + (ref_entropy * entropy_weights).sum()
    if variant == 'advantages':
        assert not actual[2].requires_grad
    actual_loss.backward()
    ref_loss.backward()
    assert actual[0].dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert torch.isfinite(actual[0]).all()
    assert (actual[0] <= 0).all()
    torch.testing.assert_close(
        actual[0].double(), ref_selected, rtol=forward_rtol, atol=forward_atol
    )
    tolerance = 8e-3 if dtype == torch.bfloat16 else forward_rtol
    torch.testing.assert_close(
        raw.grad.double(), ref_raw.grad, rtol=tolerance, atol=forward_atol
    )


@pytest.mark.parametrize('dtype', [torch.float64, torch.float32, torch.bfloat16])
def test_all_three_selected_outputs_and_both_entropies_agree_bitwise(dtype):
    raw = rows(dtype)
    targets = torch.tensor([0, 2, 3, 5])
    values = [outputs(raw, targets, mode) for mode in ['target', 'entropy', 'advantages']]
    assert torch.equal(values[0][0], values[1][0])
    assert torch.equal(values[0][0], values[2][0])
    assert torch.equal(values[1][1], values[2][1])


@pytest.mark.parametrize('variant', ['target', 'entropy', 'advantages'])
def test_peaked_entropy_has_no_large_score_subtraction(variant):
    raw = rows(torch.float32)
    targets = torch.tensor([0, 2, 3, 5])
    result = outputs(raw, targets, variant)
    if variant != 'target':
        # The old logsumexp - E[capped] expression can round to zero here.
        assert result[1][0] > 0
        expected = reference(raw.double(), targets)[1][0]
        torch.testing.assert_close(result[1][0].double(), expected, rtol=3e-4, atol=1e-12)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('variant', ['target', 'entropy', 'advantages'])
def test_nonfinite_rows_are_not_silently_repaired(bad, variant):
    raw = rows(torch.float32)
    raw[0, 1] = bad
    targets = torch.tensor([0, 2, 3, 5])
    result = outputs(raw, targets, variant)
    assert torch.isnan(result[0][0])
    assert torch.isfinite(result[0][1:]).all()
    # Row masking belongs to the caller; invalid active rows stay observable.
    masked = torch.where(torch.tensor([False, True, True, True]), result[0], 0.)
    assert torch.isfinite(masked).all()
