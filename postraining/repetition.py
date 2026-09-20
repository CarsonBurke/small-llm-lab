"""Observational word repetition metrics; never a reward or stopping rule.

The repeated n-gram fraction follows Open-R1's whitespace-word measurement:
https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
We use Unicode casefolding instead of lowercasing. TRL uses the same fraction
on token IDs: https://huggingface.co/docs/trl/main/en/rewards#get_repetition_penalty_reward
"""


def repetition_metrics(text: str) -> dict[str, float | int]:
    """Measure overlapping 3/16-word repeats and the longest identical-word run.

    Words are casefolded and whitespace-separated; punctuation is retained.
    Fractions are ``1 - unique / total``, or zero when no n-gram fits. Empty
    text has a longest run of zero. Work and storage are linear in word count
    for these fixed n-gram sizes.
    """
    words = tuple(text.casefold().split())
    metrics: dict[str, float | int] = {}
    for size in (3, 16):
        total = len(words) - size + 1
        fraction = 0.0
        if total > 0:
            unique = len({words[start : start + size] for start in range(total)})
            fraction = 1.0 - unique / total
        metrics[f"repetition_{size}gram_fraction"] = fraction

    longest = run = 0
    previous = None
    for word in words:
        run = run + 1 if word == previous else 1
        if run > longest:
            longest = run
        previous = word
    metrics["repetition_max_identical_word_run"] = longest
    return metrics
