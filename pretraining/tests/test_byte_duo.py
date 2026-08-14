"""Independent CPU reference tests for Byte-Duo's diffusion mathematics."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from pretraining.byte_diffusion.duo import (
    DuoSchedule,
    duo_nelbo_token_loss,
    duo_reverse_posterior,
    sample_antithetic_times,
    sample_branch_antithetic_times,
    uniform_state_corruption,
)


def test_reference_linear_schedule_matches_scaling_dllms_order_of_operations() -> None:
    schedule = DuoSchedule(eps=1e-3)
    t = torch.tensor([0.0, 0.5, 1.0])
    expected = 1.0 - (1.0 - schedule.eps) * t
    torch.testing.assert_close(schedule.alpha(t), expected)
    torch.testing.assert_close(
        schedule.derivative(t), torch.full_like(t, -(1.0 - schedule.eps))
    )
    torch.testing.assert_close(schedule.alpha(t)[-1], torch.tensor(schedule.eps))


def test_uniform_forward_process_matches_exact_marginal_and_has_no_mask() -> None:
    rows = 120_000
    clean_atoms = 5
    clean = torch.zeros((rows, 1), dtype=torch.long)
    active = torch.ones_like(clean, dtype=torch.bool)
    schedule = DuoSchedule(eps=0.01)
    t = torch.full((rows,), 0.6)
    result = uniform_state_corruption(
        clean,
        active,
        t,
        clean_atoms=clean_atoms,
        pad_id=6,
        schedule=schedule,
        generator=torch.Generator().manual_seed(11),
    )
    alpha = float(schedule.alpha(torch.tensor(0.6)))
    expected = torch.full((clean_atoms,), (1 - alpha) / clean_atoms)
    expected[0] += alpha
    observed = torch.bincount(result.ids[:, 0], minlength=clean_atoms).float() / rows
    torch.testing.assert_close(observed, expected, atol=0.004, rtol=0)
    assert not bool(result.ids.eq(5).any())  # id 5 is MASK in this toy layout.


def test_terminal_forward_marginal_matches_reference_nearly_uniform_schedule() -> None:
    rows = 100_000
    clean = torch.zeros((rows, 1), dtype=torch.long)
    result = uniform_state_corruption(
        clean,
        torch.ones_like(clean, dtype=torch.bool),
        torch.ones(rows),
        clean_atoms=7,
        pad_id=8,
        generator=torch.Generator().manual_seed(13),
    )
    observed = torch.bincount(result.ids[:, 0], minlength=7).float() / rows
    alpha = DuoSchedule().eps
    expected = torch.full((7,), (1 - alpha) / 7)
    expected[0] += alpha
    torch.testing.assert_close(observed, expected, atol=0.004, rtol=0)
    assert float(result.replaced.float().mean()) > 0.995


def _literal_duo_eq11(
    logits: torch.Tensor,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    alpha: torch.Tensor,
    dalpha: float,
) -> torch.Tensor:
    """Literal scalar transcription independent of the vectorized routine."""

    batch, length, classes = logits.shape
    answer = torch.empty((batch, length), dtype=torch.float64)
    probabilities = logits.double().softmax(-1)
    for row in range(batch):
        a = float(alpha[row])
        kappa = (1 - a) / (classes * a + 1 - a)
        for position in range(length):
            x = int(clean[row, position])
            z = int(noisy[row, position])
            equal = float(x == z)
            xbar_theta = classes * a * probabilities[row, position] + 1 - a
            xbar_z = 1 - a + classes * a * equal
            theta_z = float(xbar_theta[z])
            theta_x = float(xbar_theta[x])
            term1 = classes * (1 / xbar_z - 1 / theta_z)
            term2_coefficient = kappa * equal + (1 - equal)
            term2_offset = (
                (classes - 1) * kappa * equal
                - (1 / kappa) * (1 - equal)
            ) * math.log(kappa)
            term2_theta = -term2_coefficient * (
                float(xbar_theta.log().sum()) - classes * math.log(theta_z)
            )
            term2_theta -= (
                classes
                * a
                / (1 - a)
                * (math.log(theta_x) - math.log(theta_z))
                * (1 - equal)
            )
            answer[row, position] = dalpha / (classes * a) * (
                term1 - term2_theta - term2_offset
            )
    return answer


def test_vectorized_duo_nelbo_is_exactly_eq11_literal_reference() -> None:
    logits = torch.tensor(
        [
            [[0.1, -0.3, 0.7, 0.2], [1.2, -0.5, 0.0, 0.4]],
            [[-0.4, 0.2, 0.3, 0.8], [0.1, 0.9, -0.1, -0.8]],
        ],
        dtype=torch.float64,
    )
    noisy = torch.tensor([[0, 1], [3, 0]])
    clean = torch.tensor([[0, 2], [1, 0]])
    alpha = torch.tensor([0.73, 0.21], dtype=torch.float64)
    expected = _literal_duo_eq11(logits, noisy, clean, alpha, -0.999)
    observed = duo_nelbo_token_loss(
        logits,
        noisy,
        clean,
        alpha,
        -0.999,
        clean_atoms=4,
    )
    torch.testing.assert_close(observed.double(), expected, atol=3e-5, rtol=3e-5)


def test_duo_objective_is_finite_at_reference_terminal_time_with_finite_gradients() -> None:
    logits = torch.randn(2, 3, 5, requires_grad=True)
    loss = duo_nelbo_token_loss(
        logits,
        torch.tensor([[0, 1, 2], [3, 4, 0]]),
        torch.tensor([[1, 1, 2], [0, 4, 3]]),
        DuoSchedule().alpha(torch.ones(2)),
        -0.999,
        clean_atoms=5,
    ).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_antithetic_global_times_cover_one_sample_per_stratum() -> None:
    rows = 249
    times = sample_antithetic_times(
        rows,
        device="cpu",
        generator=torch.Generator().manual_seed(17),
    )
    assert bool(((times > 0) & (times < 1)).all())
    assert times.unique().numel() == rows
    # Determinism and microbatching independence are the critical contract.
    again = sample_antithetic_times(
        rows,
        device="cpu",
        generator=torch.Generator().manual_seed(17),
    )
    torch.testing.assert_close(times, again)
    torch.testing.assert_close(torch.cat(times.split((24,) * 10 + (9,))), times)


def test_branch_antithetic_times_spread_each_page_across_time_bands() -> None:
    rows, branches = 249, 8
    times = sample_branch_antithetic_times(
        rows,
        branches,
        device="cpu",
        generator=torch.Generator().manual_seed(19),
    )

    assert times.shape == (rows, branches)
    band = torch.arange(branches) / branches
    assert bool((times >= band[None]).all())
    assert bool((times < (band + 1 / branches)[None]).all())
    strata = (times.flatten().sort().values * (rows * branches)).floor()
    torch.testing.assert_close(strata, torch.arange(rows * branches).float())


def test_exact_reverse_posterior_normalizes_and_identity_transition_is_delta() -> None:
    probabilities = torch.tensor(
        [[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]]
    )
    noisy = torch.tensor([[2, 0]])
    posterior = duo_reverse_posterior(probabilities, noisy, 0.6, 0.2)
    torch.testing.assert_close(posterior.sum(-1), torch.ones_like(noisy, dtype=torch.float32))
    assert bool((posterior >= 0).all())
    identity = duo_reverse_posterior(probabilities, noisy, 0.4, 0.4)
    torch.testing.assert_close(identity, F.one_hot(noisy, 4).float(), atol=1e-6, rtol=1e-6)


def test_reverse_posterior_float32_matches_float64_oracle() -> None:
    probabilities = torch.rand(3, 5, 7, generator=torch.Generator().manual_seed(19))
    probabilities /= probabilities.sum(-1, keepdim=True)
    noisy = torch.randint(7, (3, 5), generator=torch.Generator().manual_seed(23))
    fp32 = duo_reverse_posterior(probabilities, noisy, 0.77, 0.11)
    fp64 = duo_reverse_posterior(
        probabilities, noisy, 0.77, 0.11, use_float64=True
    )
    torch.testing.assert_close(fp32.double(), fp64, atol=2e-7, rtol=2e-6)
