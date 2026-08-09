from __future__ import annotations

import pytest
import torch

from pretraining.byte_diffusion.sampling import (
    isd_accept_or_resample,
    remaining_schedule,
    reveal_low_entropy,
    reveal_quotas,
    sample_absorbing_canvas,
)


def test_isd_exact_acceptance_and_residual() -> None:
    p = torch.tensor([0.5, 0.3, 0.2])
    q = torch.tensor([0.25, 0.5, 0.25])
    assert isd_accept_or_resample(p, q, 0, accept_uniform=0.999).accepted
    assert isd_accept_or_resample(p, q, 1, accept_uniform=0.59).accepted
    rejected = isd_accept_or_resample(
        p, q, 1, accept_uniform=0.61, residual_uniform=0.99
    )
    assert not rejected.accepted
    assert rejected.token == 0
    assert rejected.acceptance_probability == pytest.approx(0.6)


def test_integer_canvas_schedules() -> None:
    assert remaining_schedule(10, 4) == [10, 8, 5, 3, 0]
    assert reveal_quotas(10, 4) == [2, 3, 2, 3]
    assert remaining_schedule(3, 5) == [3, 3, 2, 2, 1, 0]
    assert reveal_quotas(3, 5) == [0, 1, 0, 1, 1]


def test_reveal_is_stable_and_eot_is_gated() -> None:
    ids = torch.full((4,), 261, dtype=torch.long)
    unresolved = torch.ones(4, dtype=torch.bool)
    logits = torch.full((4, 261), -10.0)
    logits[0, 65] = 10
    logits[1, 256] = 20  # Confident EOT is illegal while byte zero is unresolved.
    logits[2, 66] = 5
    logits[3, 67] = 5
    next_ids, next_unresolved = reveal_low_entropy(
        ids, unresolved, logits, 2, mask_id=261, eot_id=256
    )
    assert next_ids[0] == 65
    assert next_ids[1] == 261
    assert not next_unresolved[0]
    assert next_unresolved[1]


def test_eot_and_confidence_can_end_before_maximum_nfe() -> None:
    initial = torch.full((4,), 261, dtype=torch.long)

    def terminal(_: torch.Tensor) -> torch.Tensor:
        logits = torch.full((4, 261), -20.0)
        logits[:, 256] = 20.0
        return logits

    ended = sample_absorbing_canvas(
        initial,
        terminal,
        steps=8,
        mask_id=261,
        eot_id=256,
    )
    assert ended.executed_nfe == 1
    assert ended.useful_nfe == 1

    def confident_bytes(_: torch.Tensor) -> torch.Tensor:
        logits = torch.full((4, 261), -20.0)
        logits[:, 65] = 20.0
        return logits

    adaptive = sample_absorbing_canvas(
        initial,
        confident_bytes,
        steps=8,
        mask_id=261,
        eot_id=256,
        adaptive_confidence=0.99,
        min_steps=2,
    )
    assert adaptive.executed_nfe == 2
    assert adaptive.ids.tolist() == [65, 65, 65, 65]
