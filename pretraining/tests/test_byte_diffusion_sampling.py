from __future__ import annotations

import pytest
import torch

from pretraining.byte_diffusion.sampling import (
    isd_accept_or_resample,
    remaining_schedule,
    reveal_low_entropy,
    reveal_quotas,
    sample_absorbing_canvas,
    sample_absorbing_canvas_batched,
)
from pretraining.eval_byte_diffusion_gsm8k import (
    sample_absorbing_canvas_batched_traced,
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
        strategy="confidence",
        confidence_threshold=0.99,
    )
    assert adaptive.executed_nfe == 1
    assert adaptive.ids.tolist() == [65, 65, 65, 65]


def test_batched_absorbing_sampler_has_independent_rows_and_inactive_lanes() -> None:
    initial = torch.full((3, 4), 261, dtype=torch.long)

    def denoise(_: torch.Tensor) -> torch.Tensor:
        logits = torch.full((3, 4, 261), -20.0)
        logits[0, :, 65] = 20.0
        logits[1, :, 66] = 20.0
        logits[1, 0, 256] = 40.0
        logits[2, :, 67] = 20.0
        return logits

    sample = sample_absorbing_canvas_batched(
        initial,
        denoise,
        steps=4,
        mask_id=261,
        eot_id=256,
        strategy="fixed_quota",
        row_active=torch.tensor([True, True, False]),
    )
    assert sample.ids[0].tolist() == [65, 65, 65, 65]
    assert sample.ids[1, 0] == 256
    assert not bool(sample.active[1, 1:].any())
    assert sample.ids[2].tolist() == [261, 261, 261, 261]
    assert sample.useful_nfe[0] == 4
    assert 1 <= sample.useful_nfe[1] <= 4
    assert sample.useful_nfe[2] == 0
    assert sample.executed_nfe == 4


def _binary_logits(probabilities: torch.Tensor) -> torch.Tensor:
    logits = torch.full((*probabilities.shape, 261), -torch.inf)
    logits[..., 65] = probabilities.log()
    logits[..., 66] = (1.0 - probabilities).log()
    return logits


def test_confidence_strategy_reveals_every_position_above_threshold() -> None:
    initial = torch.full((1, 3), 261, dtype=torch.long)
    calls: list[torch.Tensor] = []

    def denoise(ids: torch.Tensor) -> torch.Tensor:
        calls.append(ids.clone())
        return _binary_logits(torch.tensor([[0.95, 0.80, 0.60]]))

    sample_absorbing_canvas_batched(
        initial,
        denoise,
        steps=3,
        mask_id=261,
        eot_id=256,
        strategy="confidence",
        confidence_threshold=0.7,
        stochastic=False,
    )
    assert calls[1].eq(261).tolist() == [[False, False, True]]


def test_entropy_bounded_strategy_uses_largest_cumulative_entropy_prefix() -> None:
    initial = torch.full((1, 3), 261, dtype=torch.long)
    calls: list[torch.Tensor] = []

    def denoise(ids: torch.Tensor) -> torch.Tensor:
        calls.append(ids.clone())
        return _binary_logits(torch.tensor([[0.99, 0.80, 0.50]]))

    sample_absorbing_canvas_batched(
        initial,
        denoise,
        steps=3,
        mask_id=261,
        eot_id=256,
        strategy="entropy_bounded",
        entropy_budget=0.6,
        stochastic=False,
    )
    assert calls[1].eq(261).tolist() == [[False, False, True]]


def test_gated_eot_does_not_inflate_confidence_quota() -> None:
    initial = torch.full((1, 3), 261, dtype=torch.long)
    calls: list[torch.Tensor] = []

    def denoise(ids: torch.Tensor) -> torch.Tensor:
        calls.append(ids.clone())
        logits = _binary_logits(torch.tensor([[0.95, 0.60, 0.60]]))
        logits[0, 1] = -torch.inf
        logits[0, 1, 256] = 0.0
        return logits

    sample_absorbing_canvas_batched(
        initial,
        denoise,
        steps=3,
        mask_id=261,
        eot_id=256,
        strategy="confidence",
        confidence_threshold=0.9,
        stochastic=False,
    )
    assert calls[1].eq(261).tolist() == [[False, True, True]]


def test_fixed_one_step_allows_simultaneous_prefix_resolution_before_eot() -> None:
    initial = torch.full((1, 4), 261, dtype=torch.long)

    def denoise(_: torch.Tensor) -> torch.Tensor:
        logits = torch.full((1, 4, 261), -torch.inf)
        logits[..., 65] = 0.0
        # Make the later EOT the highest-entropy proposal so a conservative
        # pre-update quota would incorrectly leave it unresolved.
        logits[0, 1] = 0.0
        logits[0, 1, 256] = 0.01
        return logits

    sample = sample_absorbing_canvas_batched(
        initial,
        denoise,
        steps=1,
        mask_id=261,
        eot_id=256,
        strategy="fixed_quota",
        stochastic=False,
    )
    assert sample.ids[0, :2].tolist() == [65, 256]
    assert not sample.active[0, 2:].any()


def test_traced_batched_sampler_matches_production_greedy_states() -> None:
    initial = torch.full((3, 4), 261, dtype=torch.long)
    row_active = torch.tensor([True, True, False])

    def denoise(_: torch.Tensor) -> torch.Tensor:
        logits = torch.full((3, 4, 261), -20.0)
        logits[0, :, 65] = torch.tensor([20.0, 19.0, 18.0, 17.0])
        logits[1, :, 66] = 20.0
        logits[1, 0, 256] = 40.0
        logits[2, :, 67] = 20.0
        return logits

    production = sample_absorbing_canvas_batched(
        initial,
        denoise,
        steps=4,
        mask_id=261,
        eot_id=256,
        strategy="fixed_quota",
        row_active=row_active,
        stochastic=False,
    )
    traced, traces = sample_absorbing_canvas_batched_traced(
        initial,
        denoise,
        steps=4,
        mask_id=261,
        eot_id=256,
        generator=None,
        strategy="fixed_quota",
        confidence_threshold=0.7,
        entropy_budget=1.0,
        row_active=row_active,
        stochastic=False,
    )

    assert torch.equal(traced.ids, production.ids)
    assert torch.equal(traced.active, production.active)
    assert torch.equal(traced.useful_nfe, production.useful_nfe)
    assert traced.executed_nfe == production.executed_nfe
    assert traces[2] is None
    assert traces[0] is not None
    steps = traces[0]["steps"]
    assert isinstance(steps, list)
    assert [step["step"] for step in steps] == [1, 2, 3, 4]
    assert [len(step["revealed_positions"]) for step in steps] == [1, 1, 1, 1]
    assert steps[-1]["output_ids"] == [65, 65, 65, 65]


def test_traced_fixed_sampler_preserves_rng_across_zero_quota_steps() -> None:
    initial = torch.full((1, 3), 261, dtype=torch.long)
    row_active = torch.tensor([True])

    def denoise(_: torch.Tensor) -> torch.Tensor:
        logits = torch.full((1, 3, 261), -5.0)
        logits[..., 65] = 0.0
        logits[..., 66] = 0.0
        return logits

    production_generator = torch.Generator().manual_seed(0)
    traced_generator = torch.Generator().manual_seed(0)
    production = sample_absorbing_canvas_batched(
        initial,
        denoise,
        steps=5,
        mask_id=261,
        eot_id=256,
        generator=production_generator,
        strategy="fixed_quota",
        row_active=row_active,
        stochastic=True,
    )
    traced, traces = sample_absorbing_canvas_batched_traced(
        initial,
        denoise,
        steps=5,
        mask_id=261,
        eot_id=256,
        generator=traced_generator,
        strategy="fixed_quota",
        confidence_threshold=0.7,
        entropy_budget=1.0,
        row_active=row_active,
        stochastic=True,
    )

    assert torch.equal(traced.ids, production.ids)
    assert torch.equal(traced_generator.get_state(), production_generator.get_state())
    assert traces[0] is not None
    steps = traces[0]["steps"]
    assert isinstance(steps, list)
    assert steps[0]["denoiser_executed"] is False
    assert steps[0]["sampler_executed"] is False
