"""CPU contracts for byte-native corruption distributions."""

from __future__ import annotations

import math

import pytest
import torch

from pretraining.byte_diffusion.config import AtomicVocabulary
from pretraining.byte_diffusion.corruption import (
    CorruptedBatch,
    absorbing_rb,
    allmask_50,
    blt_bernoulli,
    blt_exact_k,
    exact_k_mask,
    uniform_replacement,
    whole_patch,
)


VOCAB = AtomicVocabulary()


def test_exact_k_mask_selects_only_eligible_positions_without_replacement() -> None:
    eligible = torch.tensor(
        [[True, False, True, True, False], [False, True, True, False, True]]
    )
    selected = exact_k_mask(
        eligible,
        torch.tensor([2, 1]),
        generator=torch.Generator().manual_seed(7),
    )
    torch.testing.assert_close(selected.sum(1), torch.tensor([2, 1]))
    assert not bool((selected & ~eligible).any())
    with pytest.raises(ValueError, match="outside the eligible count"):
        exact_k_mask(eligible, torch.tensor([4, 1]))


def test_absorbing_rb_masks_exact_k_and_never_touches_padding() -> None:
    clean = torch.tensor([[65, 195, 169, VOCAB.eot_id, VOCAB.pad_id]])
    eligible = torch.tensor([[True, True, True, True, False]])
    result = absorbing_rb(
        clean,
        eligible,
        VOCAB,
        k=torch.tensor([2]),
        generator=torch.Generator().manual_seed(11),
    )
    assert int(result.active.sum()) == 2
    assert bool((result.ids[result.active] == VOCAB.mask_id).all())
    torch.testing.assert_close(result.ids[~result.active], clean[~result.active])
    assert result.ids[0, -1] == VOCAB.pad_id
    assert result.targets[0, -1] == VOCAB.pad_id
    torch.testing.assert_close(result.noise_fraction, torch.tensor([0.5]))


def test_blt_uses_one_t_per_example_and_allows_a_zero_mask_row() -> None:
    clean = torch.tensor([[1, 2, 3], [4, 5, VOCAB.pad_id]])
    eligible = torch.tensor([[True, True, True], [True, True, False]])
    fully_masked = blt_bernoulli(
        clean,
        eligible,
        VOCAB,
        t=torch.tensor(1.0),
        generator=torch.Generator().manual_seed(13),
    )
    torch.testing.assert_close(fully_masked.active, eligible)
    assert fully_masked.t is not None and fully_masked.t.ndim == 0

    per_example = blt_bernoulli(
        clean,
        eligible,
        VOCAB,
        t=torch.tensor([1.0, 1e-6]),
        generator=torch.Generator().manual_seed(14),
    )
    torch.testing.assert_close(per_example.active[0], eligible[0])
    assert not bool(per_example.active[1].any())

    # A very small t may select no bytes.  The corruption object itself
    # remains valid; the objective owns the zero-safe reduction.
    zero_found = False
    for seed in range(64):
        sparse = blt_bernoulli(
            clean,
            eligible,
            VOCAB,
            t=torch.tensor(1e-6),
            generator=torch.Generator().manual_seed(seed),
        )
        if not bool(sparse.active.any()):
            zero_found = True
            break
    assert zero_found


def test_blt_exact_k_has_nonempty_masks_and_inverse_fraction_sum_weight() -> None:
    clean = torch.arange(12, dtype=torch.long).view(2, 6)
    eligible = torch.tensor(
        [[True, True, True, True, False, False], [True] * 6]
    )
    result = blt_exact_k(
        clean,
        eligible,
        VOCAB,
        k=torch.tensor([2, 3]),
        generator=torch.Generator().manual_seed(15),
    )
    torch.testing.assert_close(result.active.sum(1), torch.tensor([2, 3]))
    assert result.t is not None
    torch.testing.assert_close(result.t, torch.tensor([0.5, 0.5]))
    # For constant per-position loss c, sum(masked loss)/(K/S) is exactly S*c.
    constant_nll = torch.full_like(clean, 1.25, dtype=torch.float32)
    estimated_full_sum = (
        (constant_nll * result.active).sum(1) / result.t
    )
    torch.testing.assert_close(
        estimated_full_sum,
        eligible.sum(1).to(torch.float32) * 1.25,
    )


@pytest.mark.parametrize("eligible_count", range(1, 8))
def test_exact_k_coefficients_equal_integrated_bernoulli_objective(
    eligible_count: int,
) -> None:
    """Prove equality for every subset size in a finite eligible canvas."""

    for selected_count in range(1, eligible_count + 1):
        bernoulli_coefficient = (
            math.factorial(selected_count - 1)
            * math.factorial(eligible_count - selected_count)
            / math.factorial(eligible_count)
        )
        exact_k_coefficient = (
            (1 / eligible_count)
            * (1 / math.comb(eligible_count, selected_count))
            * (eligible_count / selected_count)
        )
        assert exact_k_coefficient == pytest.approx(
            bernoulli_coefficient, rel=1e-15, abs=1e-15
        )


def test_allmask_50_keeps_exact_nonzero_counts_and_valid_targets() -> None:
    clean = torch.arange(24, dtype=torch.long).view(4, 6)
    eligible = torch.ones_like(clean, dtype=torch.bool)
    result = allmask_50(
        clean,
        eligible,
        VOCAB,
        generator=torch.Generator().manual_seed(17),
    )
    counts = result.active.sum(1)
    assert bool(((counts >= 1) & (counts <= 6)).all())
    assert bool((result.ids[result.active] == VOCAB.mask_id).all())
    torch.testing.assert_close(result.targets, clean)


def test_whole_patch_masks_only_valid_bytes_in_a_partial_eot_patch() -> None:
    clean = torch.tensor(
        [[65, 66, 67, 68, 69, VOCAB.eot_id, VOCAB.pad_id, VOCAB.pad_id]]
    )
    eligible = torch.tensor(
        [[True, True, True, True, True, True, False, False]]
    )

    observed = None
    for seed in range(128):
        candidate = whole_patch(
            clean,
            eligible,
            VOCAB,
            generator=torch.Generator().manual_seed(seed),
        )
        expected = torch.tensor(
            [[False, False, False, False, True, True, False, False]]
        )
        if torch.equal(candidate.active, expected):
            observed = candidate
            break
    assert observed is not None, (
        "deterministic seed sweep never selected the short patch"
    )
    assert int(observed.active.sum()) == 2
    assert observed.ids[0, -1] == VOCAB.pad_id
    assert observed.ids[0, -2] == VOCAB.pad_id


def test_uniform_replacement_never_emits_mask_or_padding_as_noise() -> None:
    clean = torch.tensor([[65, 66, VOCAB.eot_id, VOCAB.pad_id]])
    eligible = torch.tensor([[True, True, True, False]])
    result = uniform_replacement(
        clean,
        eligible,
        VOCAB,
        t=torch.tensor(1.0),
        generator=torch.Generator().manual_seed(23),
    )
    assert bool((result.ids[eligible] < VOCAB.output_size).all())
    assert not bool((result.ids[eligible] == VOCAB.mask_id).any())
    assert result.ids[0, -1] == VOCAB.pad_id
    # Every eligible clean byte is a target even if a replacement happens to
    # equal EOT.  Corrupted EOT is noise metadata, not a truncation signal.
    torch.testing.assert_close(result.active, eligible)
    torch.testing.assert_close(result.targets, clean)


def test_corrupted_batch_rejects_input_only_active_targets() -> None:
    invalid = CorruptedBatch(
        ids=torch.tensor([[VOCAB.mask_id]]),
        targets=torch.tensor([[VOCAB.pad_id]]),
        active=torch.tensor([[True]]),
        noise_fraction=torch.tensor([1.0]),
        kind="invalid",
    )
    with pytest.raises(ValueError, match="not predictable"):
        invalid.validate(VOCAB)
