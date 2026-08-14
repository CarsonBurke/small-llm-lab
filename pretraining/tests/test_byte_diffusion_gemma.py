"""CPU reference tests for the byte-native DiffusionGemma primitives."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from pretraining.byte_diffusion.diffusion_gemma import (
    EntropyBudgetSamplerConfig,
    STOP_MAX_STEPS,
    STOP_STABLE_ENTROPY,
    final_eot_valid_mask,
    initialize_revisable_entropy_sampler,
    integrated_exact_k_uniform_replacement,
    revisable_entropy_sampler_step,
    sample_static_half_batch_mask,
    stopgrad_probability_embedding_projection,
    uniform_replacement_corruption,
)


def test_integrated_exact_k_matches_shared_uniform_t_marginal() -> None:
    rows = 60_000
    clean = torch.zeros((rows, 2), dtype=torch.long)
    eligible = torch.ones_like(clean, dtype=torch.bool)
    bernoulli = uniform_replacement_corruption(
        clean,
        eligible,
        output_size=4,
        generator=torch.Generator().manual_seed(101),
    )
    integrated = integrated_exact_k_uniform_replacement(
        clean,
        eligible,
        output_size=4,
        generator=torch.Generator().manual_seed(103),
    )

    expected_counts = torch.full((3,), 1.0 / 3)
    bernoulli_counts = torch.bincount(bernoulli.k, minlength=3).float() / rows
    integrated_counts = torch.bincount(integrated.k, minlength=3).float() / rows
    torch.testing.assert_close(bernoulli_counts, expected_counts, atol=0.01, rtol=0)
    torch.testing.assert_close(integrated_counts, expected_counts, atol=0.01, rtol=0)

    # For U=2, integrating t gives P(00)=P(11)=1/3 and each singleton 1/6.
    pattern = integrated.replaced[:, 0].long() + 2 * integrated.replaced[:, 1].long()
    observed_patterns = torch.bincount(pattern, minlength=4).float() / rows
    torch.testing.assert_close(
        observed_patterns,
        torch.tensor([1 / 3, 1 / 6, 1 / 6, 1 / 3]),
        atol=0.01,
        rtol=0,
    )


def test_integrated_exact_k_respects_explicit_counts_and_deterministic_rng() -> None:
    clean = torch.arange(12, dtype=torch.long).view(3, 4)
    eligible = torch.tensor(
        [
            [True, True, True, True],
            [True, False, True, False],
            [False, False, False, False],
        ]
    )
    k = torch.tensor([3, 1, 0])
    first = integrated_exact_k_uniform_replacement(
        clean,
        eligible,
        output_size=16,
        k=k,
        generator=torch.Generator().manual_seed(107),
    )
    second = integrated_exact_k_uniform_replacement(
        clean,
        eligible,
        output_size=16,
        k=k,
        generator=torch.Generator().manual_seed(107),
    )

    torch.testing.assert_close(first.k, k)
    assert not bool((first.replaced & ~eligible).any())
    torch.testing.assert_close(first.ids, second.ids)
    torch.testing.assert_close(first.replaced, second.replaced)
    with pytest.raises(RuntimeError, match="outside the eligible count"):
        integrated_exact_k_uniform_replacement(
            clean,
            eligible,
            output_size=16,
            k=torch.tensor([5, 1, 0]),
        )


def test_uniform_replacement_supervises_changed_and_unchanged_targets() -> None:
    clean = torch.tensor([[0, 0, 0, 2]])  # Final input-only PAD is not supervised.
    eligible = torch.tensor([[True, True, True, False]])
    result = uniform_replacement_corruption(
        clean,
        eligible,
        output_size=1,
        t=torch.tensor(1.0),
        generator=torch.Generator().manual_seed(109),
    )

    # All positions were selected by the forward process, but replacement from
    # a one-id vocabulary leaves their visible values unchanged. They remain
    # dense denoising labels rather than being discarded as easy targets.
    torch.testing.assert_close(result.replaced, eligible)
    assert not bool(result.changed.any())
    torch.testing.assert_close(result.unchanged, eligible)
    torch.testing.assert_close(result.active, eligible)
    torch.testing.assert_close(result.supervised_targets, torch.tensor([3]))
    torch.testing.assert_close(result.changed_targets, torch.tensor([0]))
    torch.testing.assert_close(result.unchanged_targets, torch.tensor([3]))
    torch.testing.assert_close(result.targets, clean)
    assert result.ids[0, -1] == 2


def test_uniform_replacement_accepts_clean_and_terminal_noise_endpoints() -> None:
    clean = torch.tensor([[0, 1, 2]])
    eligible = torch.ones_like(clean, dtype=torch.bool)
    clean_state = uniform_replacement_corruption(
        clean,
        eligible,
        output_size=3,
        t=torch.tensor(0.0),
        generator=torch.Generator().manual_seed(113),
    )
    noisy_state = uniform_replacement_corruption(
        clean,
        eligible,
        output_size=3,
        t=torch.tensor(1.0),
        generator=torch.Generator().manual_seed(127),
    )
    assert not bool(clean_state.replaced.any())
    torch.testing.assert_close(noisy_state.replaced, eligible)
    assert bool((noisy_state.ids >= 0).all())
    assert bool((noisy_state.ids < 3).all())
    with pytest.raises(RuntimeError, match=r"lie in \[0, 1\]"):
        uniform_replacement_corruption(
            clean,
            eligible,
            output_size=3,
            t=torch.tensor(1.01),
        )


def test_static_half_batch_selection_is_exact_and_seeded() -> None:
    first = sample_static_half_batch_mask(
        10,
        device="cpu",
        generator=torch.Generator().manual_seed(131),
    )
    second = sample_static_half_batch_mask(
        10,
        device="cpu",
        generator=torch.Generator().manual_seed(131),
    )
    assert int(first.sum()) == 5
    torch.testing.assert_close(first, second)
    odd = sample_static_half_batch_mask(
        3, device="cpu", generator=torch.Generator().manual_seed(137)
    )
    assert int(odd.sum()) == 1


def test_probability_embedding_stops_prior_gradients_but_trains_projection() -> None:
    source_logits = torch.randn(2, 3, 5, requires_grad=True)
    probabilities = source_logits.softmax(-1)
    embeddings = torch.randn(5, 4, requires_grad=True)
    projection = nn.Linear(4, 2)
    selected = torch.tensor([True, False])

    conditioning = stopgrad_probability_embedding_projection(
        probabilities,
        embeddings,
        projection,
        selected_rows=selected,
    )
    torch.testing.assert_close(conditioning[1], torch.zeros_like(conditioning[1]))
    conditioning.square().sum().backward()

    assert source_logits.grad is None
    assert embeddings.grad is not None and bool((embeddings.grad != 0).any())
    assert projection.weight.grad is not None
    assert bool((projection.weight.grad != 0).any())

    with pytest.raises(ValueError, match=r"return \[B,C,D\]"):
        stopgrad_probability_embedding_projection(
            probabilities,
            embeddings,
            lambda value: value.sum(-1),
        )


def _confident_logits(tokens: torch.Tensor, vocabulary: int) -> torch.Tensor:
    logits = torch.full((*tokens.shape, vocabulary), -30.0)
    return logits.scatter(-1, tokens[..., None], 30.0)


def test_entropy_budget_selects_low_entropy_prefix_with_stable_ties() -> None:
    state = initialize_revisable_entropy_sampler(
        1,
        3,
        output_size=3,
        device="cpu",
        generator=torch.Generator().manual_seed(137),
    )
    logits = torch.tensor(
        [
            [
                [20.0, -20.0, -20.0],
                [0.0, 0.0, -20.0],
                [math.log(0.8), math.log(0.2), -20.0],
            ]
        ]
    )
    step = revisable_entropy_sampler_step(
        state,
        logits,
        EntropyBudgetSamplerConfig(
            max_steps=4,
            entropy_budget=0.1,
            stop_entropy=0.0,
            temperature_max=1.0,
            temperature_min=1.0,
        ),
        eot_id=2,
        generator=torch.Generator().manual_seed(139),
    )

    # Algorithm 1 admits position m when entropy before m remains in budget.
    # Therefore the near-zero-entropy position and the next-lowest position
    # survive, while the highest-entropy position is uniformly renoised.
    torch.testing.assert_close(step.selected, torch.tensor([[True, False, True]]))


def test_revisable_sampler_can_change_every_prior_canvas_decision() -> None:
    config = EntropyBudgetSamplerConfig(
        max_steps=4,
        entropy_budget=100.0,
        stop_entropy=0.0,
        temperature_max=1.0,
        temperature_min=1.0,
    )
    state = initialize_revisable_entropy_sampler(
        1,
        4,
        output_size=3,
        device="cpu",
        generator=torch.Generator().manual_seed(149),
    )
    first = revisable_entropy_sampler_step(
        state,
        _confident_logits(torch.zeros((1, 4), dtype=torch.long), 3),
        config,
        eot_id=2,
        generator=torch.Generator().manual_seed(151),
    )
    torch.testing.assert_close(first.state.ids, torch.zeros((1, 4), dtype=torch.long))
    second = revisable_entropy_sampler_step(
        first.state,
        _confident_logits(torch.ones((1, 4), dtype=torch.long), 3),
        config,
        eot_id=2,
        generator=torch.Generator().manual_seed(157),
    )
    torch.testing.assert_close(second.state.ids, torch.ones((1, 4), dtype=torch.long))
    assert bool(second.revised.all())
    assert not bool(second.state.finished.any())


def test_stable_low_entropy_argmax_stops_after_two_steps_and_truncates_eot() -> None:
    config = EntropyBudgetSamplerConfig(
        max_steps=5,
        entropy_budget=0.1,
        stop_entropy=0.005,
        temperature_max=0.8,
        temperature_min=0.4,
    )
    state = initialize_revisable_entropy_sampler(
        1,
        4,
        output_size=3,
        device="cpu",
        generator=torch.Generator().manual_seed(163),
    )
    prediction = torch.tensor([[0, 2, 1, 0]])
    logits = _confident_logits(prediction, 3)
    first = revisable_entropy_sampler_step(
        state,
        logits,
        config,
        eot_id=2,
        generator=torch.Generator().manual_seed(167),
    )
    assert not bool(first.stopped.any())
    # An intermediate EOT is merely revisable canvas state and never stops.
    assert first.state.ids[0, 1] == 2
    second_generator = torch.Generator().manual_seed(173)
    rng_before_stop = second_generator.get_state().clone()
    second = revisable_entropy_sampler_step(
        first.state,
        logits,
        config,
        eot_id=2,
        generator=second_generator,
    )
    torch.testing.assert_close(second_generator.get_state(), rng_before_stop)
    assert bool(second.stopped.all())
    assert second.state.stop_reason.item() == STOP_STABLE_ENTROPY
    torch.testing.assert_close(second.state.final_ids, prediction)
    torch.testing.assert_close(
        second.state.final_valid,
        torch.tensor([[True, True, False, False]]),
    )


def test_inactive_short_canvas_suffix_does_not_block_adaptive_stopping() -> None:
    active = torch.tensor([[True, True, False, False]])
    state = initialize_revisable_entropy_sampler(
        1,
        4,
        output_size=3,
        device="cpu",
        generator=torch.Generator().manual_seed(227),
        active=active,
    )
    config = EntropyBudgetSamplerConfig(max_steps=4)
    # Invalid positions deliberately stay maximally uncertain. They are PAD
    # storage, not part of the canvas entropy or convergence predicate.
    logits = torch.tensor(
        [[[30.0, -30.0, -30.0], [-30.0, 30.0, -30.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    first = revisable_entropy_sampler_step(state, logits, config, eot_id=2)
    second = revisable_entropy_sampler_step(first.state, logits, config, eot_id=2)
    assert bool(second.stopped.all())
    torch.testing.assert_close(second.state.final_valid, active)


def test_stable_but_high_entropy_predictions_wait_until_max_steps() -> None:
    config = EntropyBudgetSamplerConfig(
        max_steps=3,
        entropy_budget=0.1,
        stop_entropy=0.01,
        temperature_max=1.0,
        temperature_min=1.0,
    )
    state = initialize_revisable_entropy_sampler(
        1,
        2,
        output_size=3,
        device="cpu",
        generator=torch.Generator().manual_seed(179),
    )
    logits = torch.zeros((1, 2, 3))
    first = revisable_entropy_sampler_step(
        state,
        logits,
        config,
        eot_id=2,
        generator=torch.Generator().manual_seed(181),
    )
    second = revisable_entropy_sampler_step(
        first.state,
        logits,
        config,
        eot_id=2,
        generator=torch.Generator().manual_seed(191),
    )
    assert not bool(second.stopped.any())
    third = revisable_entropy_sampler_step(
        second.state,
        logits,
        config,
        eot_id=2,
        generator=torch.Generator().manual_seed(193),
    )
    assert bool(third.stopped.all())
    assert third.state.stop_reason.item() == STOP_MAX_STEPS


def test_temperature_schedule_matches_algorithm_one_time_grid() -> None:
    config = EntropyBudgetSamplerConfig(
        max_steps=4,
        temperature_max=0.8,
        temperature_min=0.4,
    )
    assert config.temperature(1) == pytest.approx(0.8)
    assert config.temperature(2) == pytest.approx(0.7)
    assert config.temperature(4) == pytest.approx(0.5)


def test_maximum_step_returns_argmax_without_advancing_transition_rng() -> None:
    generator = torch.Generator().manual_seed(211)
    state = initialize_revisable_entropy_sampler(
        1,
        3,
        output_size=4,
        device="cpu",
        generator=generator,
    )
    rng_before = generator.get_state().clone()
    prediction = torch.tensor([[1, 2, 3]])
    step = revisable_entropy_sampler_step(
        state,
        _confident_logits(prediction, 4),
        EntropyBudgetSamplerConfig(max_steps=1),
        eot_id=3,
        generator=generator,
    )

    torch.testing.assert_close(generator.get_state(), rng_before)
    torch.testing.assert_close(step.state.final_ids, prediction)
    assert step.state.stop_reason.item() == STOP_MAX_STEPS


def test_final_eot_mask_keeps_eot_and_all_tokens_when_absent() -> None:
    ids = torch.tensor([[1, 2, 0, 2], [1, 0, 1, 0]])
    torch.testing.assert_close(
        final_eot_valid_mask(ids, eot_id=2),
        torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        ),
    )


def test_sampler_rng_is_deterministic_across_initialization_and_updates() -> None:
    config = EntropyBudgetSamplerConfig(
        max_steps=4,
        entropy_budget=0.2,
        stop_entropy=0.0,
    )
    first_generator = torch.Generator().manual_seed(197)
    second_generator = torch.Generator().manual_seed(197)
    first_state = initialize_revisable_entropy_sampler(
        2, 5, output_size=7, device="cpu", generator=first_generator
    )
    second_state = initialize_revisable_entropy_sampler(
        2, 5, output_size=7, device="cpu", generator=second_generator
    )
    logits = torch.randn(2, 5, 7, generator=torch.Generator().manual_seed(199))
    first = revisable_entropy_sampler_step(
        first_state,
        logits,
        config,
        eot_id=6,
        generator=first_generator,
    )
    second = revisable_entropy_sampler_step(
        second_state,
        logits,
        config,
        eot_id=6,
        generator=second_generator,
    )

    torch.testing.assert_close(first_state.ids, second_state.ids)
    torch.testing.assert_close(first.state.ids, second.state.ids)
    torch.testing.assert_close(first.sampled_ids, second.sampled_ids)
    torch.testing.assert_close(first.selected, second.selected)
