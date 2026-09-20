from __future__ import annotations

import pytest

from scripts.render_minicpm_responses import parse_sample


_CURRENT_SAMPLE = (
    """Reward: 1

Response tokens: 42

Response limit: 100

Prompt tokens: 7

Repetition: {"repetition_3gram_fraction": 0.0}

Prompt:
What is 2 + 2?

Ground truth:
4

Model response:
The answer is 4."""
)


_LEGACY_SAMPLE = (
    """Reward: -1

Response tokens: 12

Prompt:
What is 2 + 2?

Ground truth:
4

Model response:
5"""
)


_TRUNCATED_SAMPLE = (
    """Reward: -1

Response tokens: 4000

Response limit: 4644

Prompt tokens: 5356

Repetition: {}

Prompt:
long prompt prefix

[... middle truncated ...]

response suffix"""
)


def test_parse_sample_accepts_middle_truncated_tensorboard_text() -> None:
    sample = parse_sample(_TRUNCATED_SAMPLE, step=40, label="incorrect", wall_time=1.0)

    assert sample.middle_truncated
    assert sample.prompt.endswith("[... middle truncated ...]")
    assert sample.truth == "[omitted from truncated TensorBoard sample]"
    assert sample.response == "response suffix"


def test_parse_sample_accepts_current_metadata_headers() -> None:
    sample = parse_sample(_CURRENT_SAMPLE, step=8, label="correct", wall_time=1.0)

    assert sample.tokens == 42
    assert sample.prompt == "What is 2 + 2?"
    assert sample.truth == "4"
    assert sample.response == "The answer is 4."


def test_parse_sample_keeps_legacy_header_format() -> None:
    sample = parse_sample(_LEGACY_SAMPLE, step=8, label="incorrect", wall_time=1.0)

    assert sample.tokens == 12
    assert sample.response == "5"


def test_parse_sample_rejects_missing_structural_headers() -> None:
    with pytest.raises(ValueError, match="unrecognized saved sample headers"):
        parse_sample("Reward: 1\n\nResponse tokens: 3\n\nPrompt:\nmissing", step=0, label="correct", wall_time=1.0)
