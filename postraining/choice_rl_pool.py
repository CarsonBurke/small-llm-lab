"""Single-choice RL prompt pools: rendering, answer balancing and screens.

Every single-choice pool is rendered the same way -- question, blank line,
``A. option`` lines -- and graded the same way: the target is one upper-case
letter under the ``rule`` style, which ``core.verify_answer`` compares
exactly against the stripped ``<answer>`` span. Two properties are fixed at
build time because a reward cannot fix them later:

* **Letter priors.** Source answers are rarely uniform over positions, so a
  constant letter would out-earn chance. Each row's options are permuted by a
  content-seeded RNG that places the correct option at a uniformly drawn
  position; a letter prior then earns exactly chance.
* **Chance.** Guessing among k options scores 1/k, so ``extra_info.module``
  names the option count and the trainer's per-module guess baselines report
  the chance line beside the accuracy it calibrates.

Rows whose options point at other options by label or position ("all of the
above", "options 1 and 3") cannot be permuted without changing their meaning
and are dropped. Screens, in order: canonicalization, the math evaluation
index, the benchmark-question screen (``prepare_sft_corpus``), an optional
owner index of problems another pool already serves, and the RL prompt
budget.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from postraining.choice_prompt import (
    RENDERED_LABELS,
    bare_label_list,
    render_choice_problem,
    split_options,
)
from postraining.core import GPT2BPETokenizer, encode_prompt
from postraining.math_prompt import strip_math_prompt_framing
from postraining.prepare_sft_corpus import (
    QUESTION_TEMPLATE_DOCUMENTS,
    QuestionIndex,
    build_question_index,
    contains_benchmark_question,
    matching_questions,
    read_reference_texts,
    word_string,
)
from postraining.prepare_sft_traces import build_decontamination_index, contaminated

# ``exact``: the stripped answer span equals the letter (core.verify_answer).
CHOICE_STYLE = "rule"
# An option or question that points at other options by label or position.
CROSS_REFERENCE = re.compile(
    r"(?i)\b(?:all|none|both|neither|any)\s+of\s+the\s+(?:above|following|"
    r"options|choices|answers)\b|\b(?:options?|choices?|answers?|statements?)"
    r"\s*\(?(?:[a-j]|[1-9]\d?)\)?(?:\s*(?:,|and|or|&|-)\s*\(?(?:[a-j]|[1-9]\d?)\)?)*\b"
    r"|^\(?[A-Ja-j1-9]\)?\s*(?:and|&|,)\s*\(?[A-Ja-j1-9]\)?\b"
)

# Order-dependent references the pattern above misses: "All the above",
# "Neither of the previous two", "the option below". "All of these" and
# "none of these" name the whole set and survive a shuffle, so they stay.
POSITIONAL_REFERENCE = re.compile(
    r"(?i)\b(?:all|none|both|neither|any|each|either)(?:[ \t]+of)?(?:[ \t]+the)?"
    r"[ \t]+(?:above|below|previous|preceding|former|latter)\b"
    r"|\bthe[ \t]+(?:above|previous|preceding|former|latter)[ \t]+"
    r"(?:two|three|four|options?|choices?|answers?|statements?)\b"
    r"|\b(?:options?|choices?|answers?)[ \t]+(?:above|below)\b"
)


def normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def identity(text: str) -> str:
    return hashlib.sha256(normalized(text).encode("utf-8")).hexdigest()


def balanced_order(count: int, target: int, seed: bytes) -> list[int]:
    """A permutation of option indices placing ``target`` uniformly at random."""

    rng = random.Random(int.from_bytes(seed[:8], "big"))
    position = rng.randrange(count)
    others = [index for index in range(count) if index != target]
    rng.shuffle(others)
    return others[:position] + [target] + others[position:]


def load_reference_texts(
    targets, *, allow_missing: bool
) -> tuple[list[str], dict[str, int], list[str]]:
    """Texts of every present target; missing ones refuse unless allowed."""

    texts: list[str] = []
    counts: dict[str, int] = {}
    missing: list[str] = []
    for path, column in targets:
        if not path.exists():
            if not allow_missing:
                raise SystemExit(
                    f"{path} is missing; the pool would enter training "
                    "unscreened against it"
                )
            missing.append(str(path))
            continue
        values = read_reference_texts(path, column)
        texts.extend(values)
        counts[f"{path}:{column}"] = len(values)
    return texts, counts, missing


@dataclass(frozen=True)
class Screens:
    tokenizer: GPT2BPETokenizer
    exact: set[str]
    ngrams: set[tuple[str, ...]]
    benchmarks: QuestionIndex
    # Problems another pool or corpus already owns, screened with the same
    # rule so a restated copy cannot be served twice.
    owners: QuestionIndex | None
    max_prompt_tokens: int

    @classmethod
    def build(
        cls, benchmark_texts: list[str], owner_texts: list[str],
        max_prompt_tokens: int,
    ) -> "Screens":
        exact, ngrams = build_decontamination_index()
        return cls(
            tokenizer=GPT2BPETokenizer(think_tokens=True, answer_tokens=True),
            exact=exact,
            ngrams=ngrams,
            benchmarks=build_question_index(benchmark_texts),
            owners=build_question_index(owner_texts) if owner_texts else None,
            max_prompt_tokens=max_prompt_tokens,
        )


def refers_to_labels(text: str, labels: tuple[str, ...]) -> bool:
    """Whether ``text`` names options by bare label: "All of A, C and D",
    "Both 1 and 3", or an option that is a label alone ("C")."""

    return bare_label_list(text, labels) or bool(
        re.fullmatch(r"\(?(?:%s)\)?\.?" % "|".join(map(re.escape, labels)),
                     text.strip())
    )


def choice_problem(
    question: str, options: tuple[str, ...], target: int, seed: bytes,
    min_options: int, source_labels: tuple[str, ...] = (),
) -> tuple[str, str, list[int]] | str:
    """(rendered problem, letter, option order), or a drop reason.

    ``source_labels`` are the labels the source printed, when it printed
    any; a bare reference to them or to the rendered letters in an option or
    the question would point at a different option after the shuffle.
    """

    count = len(options)
    if count < min_options:
        return "too_few_options"
    if count > len(RENDERED_LABELS):
        return "too_many_options"
    if any("\n" in option or not option.strip() for option in options):
        return "multiline_or_empty_option"
    if len({normalized(option) for option in options}) < count:
        # Two options with one text: whichever letter is graded, the other
        # is marked wrong for the same answer.
        return "duplicate_option_text"
    labels = tuple(dict.fromkeys(RENDERED_LABELS[:count] + tuple(source_labels)))
    if any(
        CROSS_REFERENCE.search(text) or POSITIONAL_REFERENCE.search(text)
        or refers_to_labels(text, labels)
        for text in (question, *options)
    ):
        return "option_cross_reference"
    order = balanced_order(count, target, seed)
    rendered = tuple(options[i] for i in order)
    problem = render_choice_problem(question, rendered)
    parsed = split_options(problem)
    if parsed is None or parsed.options != rendered:
        # The rendering must parse back to exactly these options, or the
        # screens and every audit read a different problem than the policy
        # sees. A question line that reads as an option ("E. coli ...")
        # merges into the block.
        return "ambiguous_rendering"
    return problem, RENDERED_LABELS[order.index(target)], order


def screen_problem(problem: str, screens: Screens) -> tuple[str, int] | str:
    """(canonical problem, prompt tokens), or a drop reason."""

    try:
        # Canonicalized here, so the stored prompt is already what the
        # mixture's own canonicalization pass leaves unchanged.
        problem, _ = strip_math_prompt_framing(problem)
    except ValueError:
        return "uncanonicalizable"
    if not problem:
        return "empty_problem"
    if contaminated(problem, screens.exact, screens.ngrams):
        return "contaminated_math_eval"
    if contains_benchmark_question(problem, screens.benchmarks):
        return "contains_benchmark_question"
    if screens.owners is not None and contains_benchmark_question(
        problem, screens.owners
    ):
        return "owned_by_another_pool"
    tokens = len(encode_prompt(screens.tokenizer, problem))
    if tokens > screens.max_prompt_tokens:
        return "over_prompt_budget"
    return problem, tokens


# A pool can be split into a teacher-trace (SFT) partition and an RL
# partition. The unit of the split is not a row but a question component:
# rows whose problems restate one another's question stem under
# ``QUESTION_RULE`` (the owner screen's rule), or share it outright, go to one
# side together, so the partitions share no question the owner screen would
# catch. Generic stems ("Which of these is a mixture?") recur with different
# options as different rows and are one component.
#
# The rule's "distinctive" grams are those in at most a few indexed
# questions, so whether two stems match depends on the rest of the index: a
# pair whose shared grams are too common across the whole pool to count can
# match once one side is indexed without the rest, which is how an owner
# screen sees the other partition. Components are therefore built under
# three indexes -- every stem, every stem with all its grams distinctive (a
# question indexed alone), and each partition's own stems, repeated until
# nothing crosses -- and the built partitions are checked against each other
# and against the SFT corpus distilled from one of them
# (``tests/test_choice_traces.py``).
#
# The n-gram rule misses a reworded question ("Cellular respiration that
# proceeds in the presence of oxygen is known as what?" / "What is cellular
# respiration that proceeds in the presence of oxygen known as?"), so rows
# with the same set of option texts and the same correct option text are one
# component as well. In science-mc-rl-v2 that joined 147 cross-partition
# pairs of the v2 split, at least six of them reworded copies.
#
# Rewordings with different distractors escape both ("the change in
# velocity" / "the change in the velocity", "How does bacteria reproduce?" /
# "How do bacteria reproduce?"): the v4 split left about 2.5% of SFT rows
# with an RL row of the same correct option text whose stem shared half its
# content words (NOTES 2026-09-24). Rows with the same correct option text
# are therefore also one component when their stems are close in words or
# in characters (``link_reworded``). A first dry build still let 16 pairs
# cross whose correct option texts differed by an article or a plural
# ("stomach" / "the stomach", "kidney" / "the kidneys") and about 30 where
# one contained the other ("local" / "local winds", "sun" / "sunlight"), so
# the answers are compared as article-free, singular word sets, equal or
# nested, or as single words one of which begins the other. Paraphrases of
# one question with the same answer still crossed below those thresholds
# ("What is defined as the amount of force acting on a given area?" / "What
# property is the result of force acting on a given area?", about 200 pairs
# at Jaccard 0.4-0.6 in a second dry build, nearly all one fact asked twice),
# and a fifth red team still found 292 pairs at Jaccard 0.3-0.4 with the same
# answer (about 70% one fact: "What is the distance north or south of the
# equator called?" / "What is used to measure, in degrees, the distance north
# or south of the equator?") and 86 at 0.4-0.6 with related answers ("radio"
# / "radio waves", about two thirds one fact). Joining a pair that is not one
# question only moves both rows to one side, so the thresholds sit below
# those bands: Jaccard 0.3 for the same answer word set, 0.4 for related
# ones.
SPLIT_SCHEMA = "question_component_hash_split/v5"
# Stems this close, given related correct option texts, are one question:
# the Jaccard similarity of their content-word sets, or difflib's ratio over
# their normalized word strings in either argument order (a first v5 build
# split "What protects reptiles from injury and loss of water?" from "What
# protects reptiles from drying out?", ratio 0.733 one way and 0.756 the
# other).
REWORDED_MIN_JACCARD = 0.4
REWORDED_MIN_RATIO = 0.85
SAME_ANSWER_MIN_JACCARD = 0.3
SAME_ANSWER_MIN_RATIO = 0.75
_STOPWORDS = frozenset(
    "a an the of to in on at by for with from as and or is are was were be "
    "been it its this that these those what which who whom whose how why "
    "when where do does did can could would should will may might most "
    "called known".split()
)
SPLIT_RULE = (
    "rows are grouped into components by union-find over (a) the question "
    "rule (a row's rendered problem restating another row's question stem, "
    "or sharing it), over an index of every stem, over the same index with "
    "every gram distinctive, and then, until no row matches across, over "
    "each partition's own stem index, (b) identical normalized option "
    "sets with the same normalized correct option text, and (c) correct "
    "option texts whose article-free singular word sets are equal or nested "
    "(or are single words, one a prefix of the other, of 3+ characters), "
    "with stems whose stopword-stripped word "
    f"sets have Jaccard similarity >= {REWORDED_MIN_JACCARD} or whose "
    "normalized word strings have difflib ratio (the larger over both "
    f"argument orders) >= {REWORDED_MIN_RATIO} "
    f"(>= {SAME_ANSWER_MIN_JACCARD} or >= {SAME_ANSWER_MIN_RATIO} when the "
    "two word sets are equal); a component goes "
    "to the SFT partition when int(sha256(salt + NUL + min "
    "original_query_sha256 of its rows)[:16], 16) / 2**64 < sft_fraction, and "
    "to the RL partition otherwise"
)


class _Components:
    """Union-find over rows."""

    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, row: int) -> int:
        while self.parent[row] != row:
            self.parent[row] = self.parent[self.parent[row]]
            row = self.parent[row]
        return row

    def union(self, first: int, second: int) -> bool:
        first, second = self.find(first), self.find(second)
        if first == second:
            return False
        self.parent[max(first, second)] = min(first, second)
        return True

    def link(
        self, problems: list[str], stems: list[str],
        candidates: list[int], references: list[int],
        template_documents: int | None = QUESTION_TEMPLATE_DOCUMENTS,
    ) -> int:
        """Join every candidate row to the reference rows it restates, under
        an index of the reference stems alone; returns the joins made."""

        index = build_question_index(
            [stems[row] for row in references], template_documents
        )
        # The index deduplicates stems, so an id names every row with a stem.
        holders: dict[int, list[int]] = {}
        for row in references:
            question = index.exact.get(word_string(stems[row]))
            if question is not None:
                holders.setdefault(question, []).append(row)
        joins = 0
        for rows in holders.values():
            joins += sum(self.union(rows[0], row) for row in rows[1:])
        for row in candidates:
            for question in matching_questions(problems[row], index):
                joins += self.union(row, holders[question][0])
        return joins


    def link_answers(self, problems: list[str], answers: list[str]) -> int:
        """Join rows with the same option texts and correct option text."""

        holders: dict[tuple, int] = {}
        joins = 0
        for row, (problem, answer) in enumerate(zip(problems, answers)):
            parsed = split_options(problem)
            if parsed is None or answer not in parsed.labels:
                raise ValueError(f"row {row} is not a single-choice problem")
            options = tuple(normalized(option) for option in parsed.options)
            key = (
                tuple(sorted(options)), options[parsed.labels.index(answer)]
            )
            first = holders.setdefault(key, row)
            joins += self.union(first, row)
        return joins

    def link_reworded(
        self, problems: list[str], stems: list[str], answers: list[str]
    ) -> int:
        """Join rows with related correct option texts and close stems.

        Two correct option texts are related when their normalized word sets
        (``_answer_words``) are equal or one contains the other ("stomach" /
        "the stomach", "bases" / "base", "bacteria" / "beneficial
        bacteria"), or both are one word and one begins the other ("sun" /
        "sunlight"). Stems must be closer for related answers than for
        answers with equal word sets. Candidate pairs share a word's first
        three characters.
        """

        answer_words = []
        for row, (problem, answer) in enumerate(zip(problems, answers)):
            parsed = split_options(problem)
            if parsed is None or answer not in parsed.labels:
                raise ValueError(f"row {row} is not a single-choice problem")
            answer_words.append(
                _answer_words(parsed.options[parsed.labels.index(answer)])
            )
        by_prefix: dict[str, list[int]] = {}
        for row, words in enumerate(answer_words):
            for prefix in {word[:3] for word in words}:
                by_prefix.setdefault(prefix, []).append(row)
        words = [_words(stem) for stem in stems]
        content = [
            frozenset(word for word in stem if word not in _STOPWORDS)
            for stem in words
        ]
        strings = [" ".join(stem) for stem in words]
        joins = 0
        for first in range(len(problems)):
            candidates = sorted({
                second
                for prefix in {word[:3] for word in answer_words[first]}
                for second in by_prefix[prefix]
                if second > first
            })
            # SequenceMatcher caches its second sequence.
            matcher = difflib.SequenceMatcher(None, autojunk=False)
            matcher.set_seq2(strings[first])
            for second in candidates:
                if self.find(first) == self.find(second):
                    continue
                if answer_words[first] == answer_words[second]:
                    min_jaccard, min_ratio = SAME_ANSWER_MIN_JACCARD, SAME_ANSWER_MIN_RATIO
                elif _related_answers(answer_words[first], answer_words[second]):
                    min_jaccard, min_ratio = REWORDED_MIN_JACCARD, REWORDED_MIN_RATIO
                else:
                    continue
                union = content[first] | content[second]
                close = bool(union) and (
                    len(content[first] & content[second]) / len(union)
                    >= min_jaccard
                )
                if not close:
                    # ratio() depends on argument order; the rule takes the
                    # larger, so it cannot depend on row order. Both quick
                    # bounds are symmetric.
                    matcher.set_seq1(strings[second])
                    close = (
                        matcher.real_quick_ratio() >= min_ratio
                        and matcher.quick_ratio() >= min_ratio
                        and (
                            matcher.ratio() >= min_ratio
                            or difflib.SequenceMatcher(
                                None, strings[first], strings[second], autojunk=False
                            ).ratio() >= min_ratio
                        )
                    )
                joins += close and self.union(first, second)
        return joins


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _singular(word: str) -> str:
    """A crude singular: enough to equate "kidneys"/"kidney", "bases"/"base"."""

    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _related_answers(first: frozenset[str], second: frozenset[str]) -> bool:
    if first <= second or second <= first:
        return True
    if len(first) == len(second) == 1:
        (one,), (other,) = first, second
        return min(len(one), len(other)) >= 3 and (
            one.startswith(other) or other.startswith(one)
        )
    return False


def _answer_words(text: str) -> frozenset[str]:
    """An option text's words, singular, without articles; never empty."""

    words = [_singular(word) for word in _words(text)]
    return frozenset(
        [word for word in words if word not in {"a", "an", "the"}] or words or [text]
    )


def split_draw(key: str, salt: str) -> float:
    """A uniform draw in [0, 1) fixed by content."""

    digest = hashlib.sha256(f"{salt}\x00{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / 2**64


def partition_questions(
    keys: list[str], problems: list[str], stems: list[str], answers: list[str],
    sft_fraction: float, salt: str,
) -> tuple[list[bool], dict]:
    """Whether each row goes to the SFT partition, plus a component report.

    ``answers`` are the correct labels of the rendered ``problems``. Depends
    only on row content: components and their minimal key are independent
    of row order.
    """

    if not 0.0 < sft_fraction < 1.0:
        raise ValueError(f"sft_fraction must be in (0, 1), got {sft_fraction}")
    components = _Components(len(problems))
    everything = list(range(len(problems)))
    components.link(problems, stems, everything, everything)
    solo_joins = components.link(problems, stems, everything, everything, None)
    answer_joins = components.link_answers(problems, answers)
    reworded_joins = components.link_reworded(problems, stems, answers)
    rounds = 0
    while True:
        members: dict[int, list[int]] = {}
        for row in everything:
            members.setdefault(components.find(row), []).append(row)
        sft = [False] * len(keys)
        for rows in members.values():
            side = split_draw(min(keys[row] for row in rows), salt) < sft_fraction
            for row in rows:
                sft[row] = side
        sft_rows = [row for row in everything if sft[row]]
        rl_rows = [row for row in everything if not sft[row]]
        joins = components.link(problems, stems, sft_rows, rl_rows)
        joins += components.link(problems, stems, rl_rows, sft_rows)
        if not joins:
            break
        rounds += 1
    sizes = Counter(len(rows) for rows in members.values())
    return sft, {
        "components": len(members),
        "component_sizes": {str(size): count for size, count in sorted(sizes.items())},
        "rows_in_multi_row_components": sum(
            len(rows) for rows in members.values() if len(rows) > 1
        ),
        "joins_with_every_gram_distinctive": solo_joins,
        "joins_by_option_set_and_answer": answer_joins,
        "joins_by_reworded_stem_and_answer": reworded_joins,
        "cross_partition_rounds": rounds,
    }


def choice_report(rows: list[dict]) -> dict:
    """Module sizes, chance, letter balance and prompt lengths of a pool."""

    choice = [row for row in rows if row["extra_info"]["option_order"] is not None]
    letters = Counter(row["reward_model"]["ground_truth"] for row in choice)
    source = Counter(
        RENDERED_LABELS[row["extra_info"]["source_position"]] for row in choice
    )
    lengths = sorted(row["extra_info"]["prompt_token_count"] for row in rows)
    return {
        "rows_by_module": dict(
            sorted(Counter(row["extra_info"]["module"] for row in rows).items())
        ),
        "chance_accuracy_single_choice": (
            sum(row["extra_info"]["chance"] for row in choice) / len(choice)
            if choice else None
        ),
        "target_letters_after_balancing": dict(sorted(letters.items())),
        "target_letters_in_source_order": dict(sorted(source.items())),
        "modal_letter_share_after_balancing": (
            max(letters.values()) / len(choice) if choice else None
        ),
        "prompt_tokens": {
            "median": lengths[len(lengths) // 2],
            "p90": lengths[int(0.9 * (len(lengths) - 1))],
            "max": lengths[-1],
        } if lengths else None,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def refuse_existing_outputs(output: Path) -> str | None:
    """Why ``output`` cannot be written, or ``None``. Pools are immutable."""

    if "." in output.stem:
        return "--output stem must not contain a dot"
    for path in (output, output.with_suffix(".manifest.json")):
        if path.exists():
            return f"{path} exists; prompt pools are immutable"
    return None


def write_pool(rows: list[dict], manifest: dict, output: Path) -> str:
    """Write the pool and its manifest (with the pool's sha256) together."""

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".manifest.json")
    staging = output.with_name(output.name + f".{os.getpid()}.tmp")
    manifest_staging = manifest_path.with_name(
        manifest_path.name + f".{os.getpid()}.tmp"
    )
    pq.write_table(pa.Table.from_pylist(rows), staging)
    digest = sha256_file(staging)
    manifest_staging.write_text(
        json.dumps({**manifest, "output_sha256": digest}, indent=2) + "\n"
    )
    staging.replace(output)
    manifest_staging.replace(manifest_path)
    return digest
