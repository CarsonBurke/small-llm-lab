"""A held-out arithmetic panel and its exact-match grader.

Aggregate math accuracy hides the thing this project most needs to know. A
model can score respectably on word problems by pattern-matching their prose
while being unable to add two four-digit numbers, and no benchmark in the
repository would say so: the frozen gate reports one number per source, and
the sources mix skills freely.

This probe is deliberately narrow. Each item is one arithmetic fact with one
canonical answer, and results are reported per family and per digit count, so
"can it add?" and "can it add *five*-digit numbers?" are separate questions
with separate answers.

Two properties make it trustworthy:

*It is disjoint from training by construction.* The panel is generated from a
seed reserved for evaluation, then filtered against the drill training seed's
key set and against the global problem registry. An item that a training
drill could produce is dropped, not merely unlikely.

*It grades the answer, not the reasoning.* The model may work the problem
however it likes; only the final answer span is compared, after the same
numeric canonicalization the drill generator uses to write it. That keeps the
probe honest about capability rather than about formatting.

Building the panel is pure CPU. Running a model against it is not, and goes
through mlq like every other model execution.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from postraining.math_drills import (
    DEFAULT_CURRICULUM,
    FamilySpec,
    generate,
    strip_decimal,
)
from postraining.problem_registry import problem_key

PROBE_SCHEMA = "arithmetic_capability_probe/v1"

# Seeds are part of the contract, not a tuning knob. The panel seed must never
# be used to generate training drills, and `assert_disjoint` enforces it.
PROBE_SEED = 20260806
TRAINING_SEED = 11

# Below three digits a family's problem space is small enough that any useful
# training set exhausts it -- there are only a hundred one-digit additions --
# so a held-out measurement there is not merely hard but impossible. That is
# not a gap worth mourning: memorized single-digit facts are exactly what the
# model should have. The probe therefore measures the multi-digit regime,
# where generalization is the question, and says so rather than quietly
# reporting memorized items as held-out accuracy.
PROBE_MIN_DIGITS = 3

PROBE_CURRICULUM = tuple(
    FamilySpec(
        spec.name,
        spec.weight,
        min(max(spec.min_digits, PROBE_MIN_DIGITS), spec.max_digits),
        spec.max_digits,
    )
    for spec in DEFAULT_CURRICULUM
)

_COMPARISON_ANSWERS = {"<", ">", "="}
_REMAINDER_RE = re.compile(r"^(-?\d+)\s+remainder\s+(-?\d+)$")
_FRACTION_RE = re.compile(r"^(-?\d+)\s*/\s*(-?\d+)$")
_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


@dataclass(frozen=True)
class ProbeItem:
    family: str
    digits: int
    problem: str
    answer: str

    @property
    def key(self) -> bytes:
        return problem_key(self.problem)


def canonical_answer(text: str) -> str | None:
    """One normal form per answer value, or None if ``text`` is not an answer.

    The probe must not mark ``2.50`` wrong when the reference says ``2.5``, or
    ``6 remainder 0`` wrong against ``6 remainder 0``. It must equally not
    accept ``6`` for ``6 remainder 4``. Every accepted shape is listed here;
    anything else is a non-answer rather than a lenient match.
    """
    text = text.strip()
    # A leading plus is decoration, not a different number; strip it before
    # any shape test so "+7" and "7" cannot grade differently.
    if text.startswith("+"):
        text = text[1:].lstrip()
    if not text:
        return None
    if text in _COMPARISON_ANSWERS:
        return text
    if match := _REMAINDER_RE.match(text):
        return f"{int(match.group(1))} remainder {int(match.group(2))}"
    if match := _FRACTION_RE.match(text):
        denominator = int(match.group(2))
        if denominator == 0:
            return None
        return str(Fraction(int(match.group(1)), denominator))
    if _NUMBER_RE.match(text):
        return strip_decimal(text)
    return None


def graded(prediction: str, reference: str) -> bool:
    """Exact match after canonicalization; an ungradable prediction is wrong."""
    expected = canonical_answer(reference)
    if expected is None:
        raise ValueError(f"reference answer {reference!r} is not canonical")
    return canonical_answer(prediction) == expected


def build_panel(
    *,
    per_family: int = 64,
    curriculum: Sequence[FamilySpec] = PROBE_CURRICULUM,
    seed: int = PROBE_SEED,
    excluded_keys: frozenset[bytes] = frozenset(),
    oversample: int = 60,
) -> list[ProbeItem]:
    """A balanced panel: ``per_family`` items for each family, digit-stratified.

    The generator is weighted, so drawing a balanced panel means drawing more
    than needed and keeping the first ``per_family`` of each. ``oversample``
    bounds that draw; too small a value raises rather than silently returning
    a short, unbalanced panel.

    Pass ``excluded_keys`` from the training drill stream. `assert_disjoint`
    verifies the result independently, but excluding up front is what makes
    the panel buildable at all rather than merely checkable.
    """
    wanted = {spec.name: per_family for spec in curriculum}
    taken: dict[str, list[ProbeItem]] = {spec.name: [] for spec in curriculum}
    budget = per_family * len(curriculum) * oversample
    for drill in generate(
        seed=seed, count=budget, curriculum=curriculum, excluded_keys=excluded_keys
    ):
        bucket = taken[drill.family]
        if len(bucket) >= wanted[drill.family]:
            if all(len(rows) >= wanted[name] for name, rows in taken.items()):
                break
            continue
        bucket.append(
            ProbeItem(
                family=drill.family,
                digits=int(drill.difficulty.get("digits", 0)),
                problem=drill.problem,
                answer=drill.answer,
            )
        )
    short = {name: len(rows) for name, rows in taken.items() if len(rows) < wanted[name]}
    if short:
        raise RuntimeError(
            f"panel is short for {short}; raise oversample above {oversample} "
            "or lower per_family"
        )
    return [item for name in sorted(taken) for item in taken[name]]


def assert_disjoint(
    panel: Sequence[ProbeItem],
    *,
    training_seed: int = TRAINING_SEED,
    training_count: int,
    curriculum: Sequence[FamilySpec] = DEFAULT_CURRICULUM,
) -> None:
    """Fail if any panel item is reachable from the training drill stream.

    This is the check that makes the probe a held-out measurement rather than
    a training-set report. It regenerates the training keys rather than
    trusting a recorded list, so it cannot pass because a manifest was stale.
    """
    training = training_keys(
        seed=training_seed, count=training_count, curriculum=curriculum
    )
    overlap = [item for item in panel if item.key in training]
    if overlap:
        raise AssertionError(
            f"{len(overlap)} probe items also appear in the first "
            f"{training_count:,} training drills, first: "
            f"{overlap[0].problem!r}"
        )


def score(
    panel: Sequence[ProbeItem], predictions: Iterable[str]
) -> dict:
    """Exact-match accuracy overall, per family, and per family and digit."""
    predictions = list(predictions)
    if len(predictions) != len(panel):
        raise ValueError(
            f"{len(predictions)} predictions for {len(panel)} panel items"
        )
    correct = 0
    ungradable = 0
    by_family: dict[str, list[int]] = {}
    by_digits: dict[str, dict[int, list[int]]] = {}
    for item, prediction in zip(panel, predictions, strict=True):
        hit = graded(prediction, item.answer)
        correct += hit
        ungradable += canonical_answer(prediction) is None
        family = by_family.setdefault(item.family, [0, 0])
        family[0] += hit
        family[1] += 1
        digits = by_digits.setdefault(item.family, {}).setdefault(item.digits, [0, 0])
        digits[0] += hit
        digits[1] += 1
    return {
        "schema": PROBE_SCHEMA,
        "items": len(panel),
        "correct": correct,
        "accuracy": correct / max(len(panel), 1),
        "ungradable_predictions": ungradable,
        "by_family": {
            name: {"correct": hit, "items": total, "accuracy": hit / total}
            for name, (hit, total) in sorted(by_family.items())
        },
        "by_family_digits": {
            name: {
                str(size): {
                    "correct": hit,
                    "items": total,
                    "accuracy": hit / total,
                }
                for size, (hit, total) in sorted(sizes.items())
            }
            for name, sizes in sorted(by_digits.items())
        },
    }


def training_keys(
    *,
    seed: int = TRAINING_SEED,
    count: int,
    curriculum: Sequence[FamilySpec] = DEFAULT_CURRICULUM,
) -> frozenset[bytes]:
    """Every problem key the training drill stream will emit."""
    return frozenset(
        drill.key
        for drill in generate(seed=seed, count=count, curriculum=curriculum)
    )


def write_panel(panel: Sequence[ProbeItem], path: Path, provenance: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(json.dumps({"schema": PROBE_SCHEMA, **provenance}) + "\n")
        for item in panel:
            handle.write(
                json.dumps(
                    {
                        "family": item.family,
                        "digits": item.digits,
                        "problem": item.problem,
                        "answer": item.answer,
                    }
                )
                + "\n"
            )


def read_panel(path: Path) -> tuple[list[ProbeItem], dict]:
    lines = Path(path).read_text().splitlines()
    if not lines:
        raise ValueError(f"{path} is empty")
    provenance = json.loads(lines[0])
    if provenance.get("schema") != PROBE_SCHEMA:
        raise ValueError(
            f"probe schema {provenance.get('schema')!r} is not {PROBE_SCHEMA!r}"
        )
    panel = [
        ProbeItem(
            family=row["family"],
            digits=row["digits"],
            problem=row["problem"],
            answer=row["answer"],
        )
        for row in (json.loads(line) for line in lines[1:])
    ]
    return panel, provenance
