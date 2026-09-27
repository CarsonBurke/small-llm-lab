"""Which problems two math RL pools share, so each can be owned by one source.

A prompt reachable through two mixture sources is drawn under two names,
counted twice against the pass, and splits its own learnability telemetry.
``prepare_vapo_mixture`` refuses exact canonical-prompt collisions, but pools
assembled from the same upstream competitions (UltraData-RL-2609 and
DAPO-Math-17K both draw on AoPS/AMC/AIME/CN-olympiad pools) mostly share
problems *up to formatting*: ``$\\frac mn$`` against ``$\\frac{m}{n}$``, a
``[asy]`` figure kept by one copy and dropped by the other, or DAPO's
integer-answer rewrite appended to an otherwise identical statement ("The
answer is in the form \\frac{m}{n} ... Please provide the value of m + n").

Three matchers run, strongest first, and a pair is reported under the first
that fires:

``whitespace``
    The mixture builder's own rule: whitespace-collapsed, lowercased
    canonical prompt.
``skeleton``
    NFKC, LaTeX delimiters/spacing/text wrappers removed, ``\\dfrac``,
    ``\\tfrac`` and the ``\\df`` macro folded to ``\\frac``, then every
    character dropped except letters, digits, the operator characters
    ``+ - = < > ^ /`` and a decimal point between digits. Non-Latin letters
    survive, so two different Chinese problems cannot collapse onto their
    shared digits; signs survive because ``(x+y)^2dx - (x^2+y^2)dy`` and the
    same integral with ``+`` are different problems with different answers;
    ``^`` and the decimal point keep ``2^{10}`` from ``210`` and ``1.5`` from
    ``15``. Braces and parentheses still go, since ``\\frac12`` against
    ``\\frac{1}{2}`` is the formatting this key exists to ignore, so
    ``\\frac{1}{2}+3`` and ``\\frac{1}{2+3}`` share a key; a same-text
    group whose targets disagree is quarantined, not merged.
``shingle``
    Near-duplicate containment over 8-token shingles. Containment is measured
    against the *smaller* problem's own shingle count, so a restatement
    embedded in a longer prompt (an appended answer-form clause, a kept
    figure) cannot dilute away, and the digit multiset of the smaller problem
    must be contained in the larger one's. The digit rule is what separates a
    reformatted copy from a *sibling variant* -- "Compute the number of
    ordered pairs with 1 <= x < y <= 200" against the same sentence with 100
    -- which shares nearly every shingle but is a different problem with a
    different answer.

CJK text has no word boundaries, so each non-Latin letter is its own token;
treating a Chinese sentence as one "word" would make every shingle depend on
the whole sentence and miss reformatted copies, while dropping it (the ASCII
``[a-z0-9]+`` rule) leaves only subscript runs such as ``x 1 x 2 x 3`` to
match, which is how unrelated Chinese problems collided. Windows dominated by
one token are not indexed, for the same reason ``decontaminate`` drops them.

Template text is not evidence either. DAPO appends "The answer is in the form
\\frac{m}{n}, where gcd(m, n) = 1. Please provide the value of m + n." to
over 600 problems, and UltraData carries a Chinese equivalent on 300; a short
problem's shingles are dominated by such a clause, so it "contains" every
other problem with the same clause. ``template_shingles`` collects the
windows that occur in more than ``MAX_SHINGLE_DOCUMENTS`` problems of any
pool, and both sides of every comparison ignore them.

Calibration (2026-09-23, UltraData-RL-2609 Math vs DAPO-Math-17K) and the
thresholds' false-positive/false-negative evidence are recorded in NOTES.md.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from postraining.decontaminate import informative

OVERLAP_SCHEMA = "math_problem_overlap/v1"
SHINGLE_SIZE = 8
# Calibrated on UltraData-RL-2609 Math vs DAPO-Math-17K; see NOTES.md.
MIN_CONTAINMENT = 0.3
MIN_SHARED_SHINGLES = 6
# A window shared by more problems than this is template text, not a problem.
MAX_SHINGLE_DOCUMENTS = 10
MATCHERS = ("whitespace", "skeleton", "shingle")

_WHITESPACE = re.compile(r"\s+")
_LATEX_DELIMITERS = re.compile(r"\\[()\[\]]|\$")
_LATEX_SPACING = re.compile(
    r"\\(?:left|right|big|Big|bigg|Bigg|displaystyle|textstyle|quad|qquad"
    r"|[,;:! ])(?![A-Za-z])"
)
_LATEX_TEXT_WRAPPER = re.compile(
    r"\\(?:text|textbf|textit|textrm|mathrm|mathbf|mathit|operatorname|mbox|emph)"
    r"\s*\{([^{}]*)\}"
)
_LATEX_FRACTION = re.compile(r"\\(?:[dt]frac|df)(?![A-Za-z])")
_SKELETON_DROPPED = re.compile(r"(?!(?<=\d)\.(?=\d))[^\w+\-=<>^/]|_")
_TOKEN = re.compile(r"[0-9]+|[a-z]+|[^\W\d_a-z]")
_NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def whitespace_key(text: str) -> str:
    """The identity ``prepare_vapo_mixture`` refuses to see in two sources."""
    return _WHITESPACE.sub(" ", text).strip().lower()


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\u2212", "-")
    text = _LATEX_DELIMITERS.sub(" ", text)
    text = _LATEX_FRACTION.sub(r"\\frac", text)
    text = _LATEX_SPACING.sub(" ", text)
    # Nested wrappers (\text{\textbf{x}}) unwrap from the inside out.
    for _ in range(3):
        text = _LATEX_TEXT_WRAPPER.sub(r" \1 ", text)
    return text.casefold()


def skeleton_key(text: str) -> str:
    """Letters, digits, operators and decimal points, markup normalized away."""
    return _SKELETON_DROPPED.sub("", _clean(text))


def overlap_tokens(text: str) -> list[str]:
    """Digit runs, Latin letter runs, and single non-Latin letters."""
    return _TOKEN.findall(_clean(text))


def shingles(
    text: str,
    size: int = SHINGLE_SIZE,
    ignore: frozenset[tuple[str, ...]] = frozenset(),
) -> frozenset[tuple[str, ...]]:
    """Distinct informative ``size``-token windows of ``text``, less ``ignore``."""
    tokens = overlap_tokens(text)
    return frozenset(
        window
        for start in range(len(tokens) - size + 1)
        if informative(window := tuple(tokens[start : start + size]), size)
        and window not in ignore
    )


def template_shingles(
    pools: Iterable[Iterable[str]],
    *,
    size: int = SHINGLE_SIZE,
    max_documents: int = MAX_SHINGLE_DOCUMENTS,
) -> frozenset[tuple[str, ...]]:
    """Windows that occur in more than ``max_documents`` problems of a pool.

    Counted per pool rather than over the union, so a problem that genuinely
    appears in both pools is not mistaken for a template by its own copies.
    """
    template: set[tuple[str, ...]] = set()
    for pool in pools:
        documents: Counter = Counter()
        for problem in pool:
            documents.update(shingles(problem, size))
        template.update(
            gram for gram, count in documents.items() if count > max_documents
        )
    return frozenset(template)


def number_multiset(text: str) -> Counter:
    return Counter(_NUMBER.findall(unicodedata.normalize("NFKC", text)))


def numbers_contained(left: Counter, right: Counter) -> bool:
    """Whether the smaller problem's numbers all occur in the larger one."""
    smaller, larger = sorted((left, right), key=lambda counts: sum(counts.values()))
    return not (smaller - larger)


@dataclass(frozen=True)
class OverlapMatch:
    """``candidate`` (a query position) restates ``reference`` (an index row)."""

    candidate: int
    reference: int
    matcher: str
    shared_shingles: int
    containment: float


class ProblemOverlapIndex:
    """All three matchers over one reference pool of canonical problems."""

    def __init__(
        self,
        problems: Sequence[str],
        *,
        size: int = SHINGLE_SIZE,
        min_containment: float = MIN_CONTAINMENT,
        min_shared: int = MIN_SHARED_SHINGLES,
        template: frozenset[tuple[str, ...]] = frozenset(),
    ):
        if not 0.0 < min_containment <= 1.0:
            raise ValueError("min_containment must lie in (0, 1]")
        if min_shared < 1:
            raise ValueError("min_shared must be positive")
        self.size = size
        self.min_containment = min_containment
        self.min_shared = min_shared
        self.template = template
        self._whitespace: dict[str, list[int]] = {}
        self._skeleton: dict[str, list[int]] = {}
        self._postings: dict[tuple[str, ...], list[int]] = {}
        self._sizes: list[int] = []
        self._numbers: list[Counter] = []
        for position, problem in enumerate(problems):
            self._whitespace.setdefault(whitespace_key(problem), []).append(position)
            skeleton = skeleton_key(problem)
            if skeleton:
                self._skeleton.setdefault(skeleton, []).append(position)
            grams = shingles(problem, size, template)
            self._sizes.append(len(grams))
            self._numbers.append(number_multiset(problem))
            for gram in grams:
                self._postings.setdefault(gram, []).append(position)

    def __len__(self) -> int:
        return len(self._sizes)

    def matches(
        self, problem: str, *, candidate: int = -1, exclude: int | None = None
    ) -> list[OverlapMatch]:
        """Every reference row ``problem`` restates, strongest matcher first.

        ``exclude`` skips one reference position, so an index can be queried
        with its own rows to find near-duplicates *within* a pool.
        """
        found: dict[int, OverlapMatch] = {}
        for matcher, table, key in (
            ("whitespace", self._whitespace, whitespace_key(problem)),
            ("skeleton", self._skeleton, skeleton_key(problem)),
        ):
            for reference in table.get(key, ()) if key else ():
                if reference != exclude and reference not in found:
                    found[reference] = OverlapMatch(
                        candidate, reference, matcher, 0, 1.0
                    )
        grams = shingles(problem, self.size, self.template)
        hits: Counter = Counter()
        for gram in grams:
            for reference in self._postings.get(gram, ()):
                hits[reference] += 1
        numbers = None
        for reference, shared in hits.items():
            if reference == exclude or reference in found:
                continue
            containment = shared / min(len(grams), self._sizes[reference])
            if shared < self.min_shared or containment < self.min_containment:
                continue
            if numbers is None:
                numbers = number_multiset(problem)
            if not numbers_contained(numbers, self._numbers[reference]):
                continue
            found[reference] = OverlapMatch(
                candidate, reference, "shingle", shared, containment
            )
        order = {name: rank for rank, name in enumerate(MATCHERS)}
        return sorted(
            found.values(),
            key=lambda match: (order[match.matcher], -match.containment, match.reference),
        )

    def provenance(self) -> dict:
        return {
            "schema": OVERLAP_SCHEMA,
            "matchers": list(MATCHERS),
            "shingle_size": self.size,
            "min_containment": self.min_containment,
            "min_shared_shingles": self.min_shared,
            "template_shingles_ignored": len(self.template),
            "containment_denominator": "smaller problem's informative shingles",
            "sibling_variant_rule": "smaller problem's digit multiset must be "
            "contained in the larger one's",
            "reference_rows": len(self),
        }


def cross_matches(
    candidates: Iterable[str], index: ProblemOverlapIndex
) -> list[OverlapMatch]:
    """The strongest match of each candidate that restates an index row."""
    found = []
    for position, problem in enumerate(candidates):
        matches = index.matches(problem, candidate=position)
        if matches:
            found.append(matches[0])
    return found
