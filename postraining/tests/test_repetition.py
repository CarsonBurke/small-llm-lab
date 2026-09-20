import pytest

from postraining.repetition import repetition_metrics


@pytest.mark.parametrize(
    ("text", "longest"),
    [("", 0), (" \t\n", 0), ("one", 1), ("one two", 1)],
)
def test_empty_and_short_text(text, longest):
    assert repetition_metrics(text) == {
        "repetition_3gram_fraction": 0.0,
        "repetition_16gram_fraction": 0.0,
        "repetition_max_identical_word_run": longest,
    }


def test_exact_multiword_loop_counts_overlapping_ngrams():
    metrics = repetition_metrics("red green blue " * 8)
    # There are three distinct rotations, not eight disjoint phrase samples.
    assert metrics["repetition_3gram_fraction"] == pytest.approx(1 - 3 / 22)
    assert metrics["repetition_16gram_fraction"] == pytest.approx(1 - 3 / 9)
    assert metrics["repetition_max_identical_word_run"] == 1


def test_repeated_syntax_does_not_imply_long_span_repetition():
    metrics = repetition_metrics(
        "if x is None: return 1\nif y is None: return 2\nif z is None: return 3"
    )
    # Only "is None: return" repeats: two duplicates among sixteen trigrams.
    assert metrics["repetition_3gram_fraction"] == pytest.approx(2 / 16)
    assert metrics["repetition_16gram_fraction"] == 0.0

    passage = " ".join(f"word{i}" for i in range(16))
    assert repetition_metrics(passage)["repetition_16gram_fraction"] == 0.0
    repeated = repetition_metrics(f"{passage}\n{passage}")
    assert repeated["repetition_3gram_fraction"] == pytest.approx(1 - 16 / 30)
    assert repeated["repetition_16gram_fraction"] == pytest.approx(1 / 17)


def test_unicode_casefold_and_whitespace_define_word_identity():
    metrics = repetition_metrics("Straße\tx Y\nSTRASSE x y")
    assert metrics["repetition_3gram_fraction"] == pytest.approx(1 / 4)
    assert metrics["repetition_16gram_fraction"] == 0.0
    assert metrics["repetition_max_identical_word_run"] == 1
    assert (
        repetition_metrics("Straße STRASSE straße")["repetition_max_identical_word_run"]
        == 3
    )


def test_identical_word_runs_reset_and_keep_the_longest_run():
    metrics = repetition_metrics("go GO go stop STOP end end END end done")
    assert metrics["repetition_max_identical_word_run"] == 4
    assert repetition_metrics("a a b a a")["repetition_max_identical_word_run"] == 2
    assert repetition_metrics("yes yes! yes")["repetition_max_identical_word_run"] == 1


def test_identical_words_respect_ngram_size_boundaries():
    below = repetition_metrics("same " * 15)
    boundary = repetition_metrics("same " * 16)
    above = repetition_metrics("same " * 17)
    assert below["repetition_16gram_fraction"] == 0.0
    assert boundary["repetition_16gram_fraction"] == 0.0
    assert above["repetition_16gram_fraction"] == 0.5
    assert above["repetition_3gram_fraction"] == pytest.approx(1 - 1 / 15)
    assert above["repetition_max_identical_word_run"] == 17
