"""Behavioral probes for answer-embedding geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from torch import Tensor

from postraining.answer_encoder.model import DEFAULT_REWARD_TEMPERATURE


class ProbeScorer(Protocol):
    space: str

    def preencode_target(self, target: str) -> object: ...

    def score(self, answers: Sequence[str], target_embedding: object) -> Tensor: ...

    def cosine(self, answers: Sequence[str], target_embedding: object) -> Tensor: ...


@dataclass(frozen=True)
class ProbeCase:
    name: str
    target: str
    candidates: tuple[tuple[str, str], ...]


FUNCTIONS_XY = """\
def x(value):
    return value + 1

def y(value):
    return value * 2
"""

FUNCTIONS_YX = """\
def y(value):
    return value * 2

def x(value):
    return value + 1
"""

FUNCTIONS_WRONG = """\
def x(value):
    return value - 1

def y(value):
    return value * 2
"""


DEFAULT_PROBES = (
    ProbeCase(
        name="numeric_10",
        target="10",
        candidates=(
            ("exact", "10"),
            ("decimal_equivalent", "10.0"),
            ("word_equivalent", "ten"),
            ("near_integer", "9"),
            ("near_decimal", "10.1"),
            ("far_50", "50"),
            ("far_100", "100"),
        ),
    ),
    ProbeCase(
        name="taxonomy_cat",
        target="cat",
        candidates=(
            ("exact", "cat"),
            ("subtype", "kitten"),
            ("coordinate", "dog"),
            ("unrelated", "airplane"),
        ),
    ),
    ProbeCase(
        name="negation",
        target="The value is greater than ten.",
        candidates=(
            ("exact", "The value is greater than ten."),
            ("paraphrase", "The value exceeds 10."),
            ("negated", "The value is not greater than ten."),
            ("reversed", "The value is less than ten."),
        ),
    ),
    ProbeCase(
        name="function_permutation",
        target=FUNCTIONS_XY,
        candidates=(
            ("exact", FUNCTIONS_XY),
            ("reordered", FUNCTIONS_YX),
            ("semantic_bug", FUNCTIONS_WRONG),
        ),
    ),
)


def run_behavioral_probes(
    scorer: ProbeScorer,
    cases: tuple[ProbeCase, ...] = DEFAULT_PROBES,
) -> dict:
    report: dict[str, object] = {
        "embedding_space": scorer.space,
        "reward_mapping": "exp((cosine - 1) / temperature)",
        "reward_temperature": getattr(
            scorer, "reward_temperature", DEFAULT_REWARD_TEMPERATURE
        ),
        "cases": {},
    }
    aggregate: dict[str, float] = {}
    for case in cases:
        target = scorer.preencode_target(case.target)
        labels = [label for label, _ in case.candidates]
        texts = [text for _, text in case.candidates]
        rewards = scorer.score(texts, target)
        cosine = scorer.cosine(texts, target)
        values = {
            label: {
                "cosine": float(cosine[index]),
                "reward": float(rewards[index]),
            }
            for index, label in enumerate(labels)
        }
        report["cases"][case.name] = values
        for label, metrics in values.items():
            aggregate[f"probe/{case.name}/{label}_cosine"] = metrics["cosine"]

    numeric = report["cases"]["numeric_10"]
    aggregate["probe/numeric_exact_min_margin"] = numeric["exact"]["cosine"] - max(
        metrics["cosine"] for label, metrics in numeric.items() if label != "exact"
    )
    equivalent_numeric = min(
        numeric["decimal_equivalent"]["cosine"],
        numeric["word_equivalent"]["cosine"],
    )
    nonequivalent_numeric = max(
        numeric["near_integer"]["cosine"],
        numeric["near_decimal"]["cosine"],
        numeric["far_50"]["cosine"],
        numeric["far_100"]["cosine"],
    )
    aggregate["probe/numeric_equivalent_margin"] = (
        equivalent_numeric - nonequivalent_numeric
    )
    related = report["cases"]["taxonomy_cat"]
    aggregate["probe/taxonomy_related_margin"] = min(
        related["subtype"]["cosine"],
        related["coordinate"]["cosine"],
    ) - related["unrelated"]["cosine"]
    functions = report["cases"]["function_permutation"]
    aggregate["probe/function_reorder_bug_margin"] = (
        functions["reordered"]["cosine"] - functions["semantic_bug"]["cosine"]
    )
    negation = report["cases"]["negation"]
    aggregate["probe/paraphrase_negation_margin"] = (
        negation["paraphrase"]["cosine"] - negation["negated"]["cosine"]
    )
    report["metrics"] = aggregate
    checks = {
        "numeric_equivalents_outrank_nonequivalents": (
            aggregate["probe/numeric_equivalent_margin"] > 0.0
        ),
        "taxonomy_related_outranks_unrelated": (
            aggregate["probe/taxonomy_related_margin"] > 0.0
        ),
        "function_reorder_outranks_bug": aggregate["probe/function_reorder_bug_margin"] > 0.0,
        "paraphrase_outranks_negation": aggregate["probe/paraphrase_negation_margin"] > 0.0,
    }
    report["gate"] = {"passed": all(checks.values()), "checks": checks}
    return report
