"""Source removal must preserve the original prompt positions, including MC RNG."""
from collections import Counter
from pathlib import Path

import pytest

from postraining.vapo import mixture
from postraining.vapo.mixture import MixtureSource, MixedPromptSampler


def sources():
    return [MixtureSource(name, Path(name), "math", tuple(
        {"source": name, "row": index} for index in range(size)
    )) for name, size in [("math", 7), ("code", 3), ("science", 4)]]


def test_exclusion_preserves_original_stream_and_choice_rng(monkeypatch):
    monkeypatch.setattr(mixture, "randomize_choice_row", lambda row, seed, cursor: {
        **row, "choice_cursor": cursor,
    })
    original = MixedPromptSampler(sources(), 13, "same")
    stream = original.next_rows(42)
    start = 5
    expected = [row for row in stream[start:] if row["source"] != "code"]
    filtered = MixedPromptSampler(sources(), 13, "same", cursor=start,
                                  excluded_sources=("code",))
    positions = list(filtered.positions(start, len(expected)))
    assert filtered.cursor_after(start, len(expected)) == positions[-1] + 1
    assert filtered.source_counts(start, len(expected)) == Counter(
        row["source"] for row in expected)
    assert filtered.next_rows(len(expected)) == expected
    assert filtered.cursor == positions[-1] + 1
    resumed = MixedPromptSampler(sources(), 13, "same", cursor=filtered.cursor,
                                 excluded_sources=("code",))
    assert resumed.next_rows(15) == filtered.next_rows(15)


def test_exclusion_validation_and_remaining_counts():
    with pytest.raises(ValueError, match="unknown"):
        MixedPromptSampler(sources(), 1, "same", excluded_sources=("typo",))
    with pytest.raises(ValueError, match="every"):
        MixedPromptSampler(sources(), 1, "same",
                           excluded_sources=("math", "code", "science"))
    sampler = MixedPromptSampler(sources(), 1, "same", cursor=5,
                                 excluded_sources=("code",))
    assert sampler.remaining_in_cycle() == sum(
        source != "code" for source in sampler.schedule[5:])
    assert sampler.next_rows(0) == []
    assert sampler.cursor == 5
    assert sampler.cursor_after(5, 0) == 5
    with pytest.raises(ValueError, match="nonnegative"):
        sampler.next_rows(-1)


def test_resume_only_allows_additive_exclusions():
    from types import SimpleNamespace
    from postraining.train_latent_vapo import (
        RESUME_EXACT_ARG_FIELDS, validate_resume_arg_contract,
    )
    saved = {name: 1 for name in RESUME_EXACT_ARG_FIELDS}
    saved.update(prompts_per_rollout=64, prompts_per_minibatch=64,
                 delightful_policy_gradient=True, exclude_rl_sources="")
    current = SimpleNamespace(**{**saved, "exclude_rl_sources": "code"})
    validate_resume_arg_contract(saved, current)
    saved["exclude_rl_sources"] = "code"
    validate_resume_arg_contract(saved, current)
    current.exclude_rl_sources = ""
    with pytest.raises(ValueError, match="re-enable"):
        validate_resume_arg_contract(saved, current)
