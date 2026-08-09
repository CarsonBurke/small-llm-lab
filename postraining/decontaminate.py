"""Substring contamination detection for held-out problems in web text.

Exact problem-key matching (`postraining.problem_registry`) catches a document
whose *whole* text is a problem statement. That is the only case the corpus
builder's `quality_keys` mechanism ever caught, and it is nearly useless for
web text: a GSM8K item quoted inside a tutoring page, a forum thread, or a
scraped worksheet is a fragment of a much larger document and hashes to
something entirely different.

This module closes that hole with the standard n-gram overlap test used by
GPT-3 and Llama: a document is contaminated if it contains any word n-gram
(default 13) that also occurs in a protected problem statement. False
positives at n=13 are rare, and the cost of one is a discarded web document.

Coverage is not universal and the index says so. A protected problem shorter
than the n-gram width -- which is most of the DeepMind arithmetic pool, whose
items read like "What is the difference between -221017 and -1429.06?" --
contributes no n-grams at all. Such problems are template-generated, so exact
key matching is the appropriate and sufficient tool for them. Both tests run;
neither alone is enough.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from postraining.problem_registry import (
    REGISTRY_SCHEMA,
    SPLIT_PRIORITY,
    ProblemRegistry,
    problem_key,
    strip_framing,
)

INDEX_SCHEMA = "math_contamination_index/v1"
DEFAULT_NGRAM_SIZE = 13
DIGEST_PERSON = b"pgolf-ngram"

# Hashes buffered before being folded into a deduplicated uint64 block.
_BLOCK_HASHES = 8_000_000

# Odd 64-bit multiplier (the FNV-1a prime) for the rolling n-gram hash.
_MULTIPLIER = np.uint64(1099511628211)

_WORD_RE = re.compile(r"[0-9]+|[^\W\d_]+", re.UNICODE)
_WORD_HASH_CACHE: dict[str, int] = {}
_WORD_CACHE_LIMIT = 2_000_000

# An n-gram dominated by one repeated word identifies a notation, not a
# problem. `a_1 a_2 a_3 ... a_7` reduces to the window
# "a 1 a 2 a 3 a 4 a 5 a 6 a", in which "a" fills 7 of 13 slots; indexing it
# would flag every LaTeX subscript run in the corpus. English prose never
# concentrates like that -- even "of the" repeated puts "the" at 4 of 13 -- so
# the rule costs nothing on real problem statements. A problem made entirely
# of such patterns loses n-gram protection and keeps exact key matching;
# `low_diversity_ngrams` reports how many windows were dropped.
MAX_REPEATED_WORD_FRACTION = 0.5


def index_words(text: str) -> list[str]:
    """Word tokens for n-gram matching.

    Punctuation, markup, and spacing differ between a scraped copy of a
    problem and its dataset original, so only alphanumeric runs survive.
    Digit runs stay separate from letter runs, which keeps "5 apples" and
    "5apples" identical -- a normalization difference should not let a quoted
    problem escape.
    """
    return _WORD_RE.findall(unicodedata.normalize("NFKC", text).casefold())


def word_hash(word: str) -> int:
    """Deterministic 64-bit hash of a single word.

    Python's built-in `hash` is salted per process and would make the index
    unreadable by the run that consumes it, so this is blake2b. It is cached
    because web text reuses a small word set relentlessly.
    """
    cached = _WORD_HASH_CACHE.get(word)
    if cached is not None:
        return cached
    if len(_WORD_HASH_CACHE) >= _WORD_CACHE_LIMIT:
        # Web text has an unbounded vocabulary once URLs, identifiers and
        # scanner noise are counted. Drop the table rather than let it grow
        # without limit; the common words repopulate it immediately.
        _WORD_HASH_CACHE.clear()
    digest = int.from_bytes(
        hashlib.blake2b(
            word.encode("utf-8"), digest_size=8, person=DIGEST_PERSON
        ).digest(),
        "little",
    )
    _WORD_HASH_CACHE[word] = digest
    return digest


def informative(window: Sequence[str], size: int) -> bool:
    """Whether an n-gram is varied enough to identify anything."""
    counts = Counter(window)
    return counts.most_common(1)[0][1] <= size * MAX_REPEATED_WORD_FRACTION


def ngram_hashes(words: list[str], size: int) -> np.ndarray:
    """64-bit rolling hashes of every contiguous ``size``-word window.

    Hashing each window's joined text directly would be simpler but costs a
    blake2b call per window, and the corpus builder must test tens of millions
    of web documents. Words are hashed once each and folded into a polynomial
    over `size` vectorized steps instead, which is the same total work per
    document as a single pass over its words.
    """
    if len(words) < size:
        return np.empty(0, dtype=np.uint64)
    ids = np.fromiter(
        (word_hash(word) for word in words), dtype=np.uint64, count=len(words)
    )
    windows = np.lib.stride_tricks.sliding_window_view(ids, size)
    accumulator = np.zeros(windows.shape[0], dtype=np.uint64)
    for offset in range(size):
        # Unsigned integer arithmetic wraps modulo 2**64 by definition, which
        # is exactly the polynomial hash wanted here.
        accumulator = accumulator * _MULTIPLIER + windows[:, offset]
    return accumulator


def informative_ngram_hashes(words: list[str], size: int) -> np.ndarray:
    """`ngram_hashes` with degenerate windows dropped.

    Only the index side filters. A document window that would have been
    dropped has no counterpart in the index to match, so filtering the query
    side too would cost work without changing any answer.
    """
    hashes = ngram_hashes(words, size)
    if hashes.size == 0:
        return hashes
    keep = np.fromiter(
        (
            informative(words[start : start + size], size)
            for start in range(hashes.size)
        ),
        dtype=bool,
        count=hashes.size,
    )
    return hashes[keep]


def protected_splits(split: str) -> frozenset[str]:
    """The splits whose problems a builder for ``split`` must refuse."""
    if split not in SPLIT_PRIORITY:
        raise ValueError(f"unknown split {split!r}")
    return frozenset(SPLIT_PRIORITY[: SPLIT_PRIORITY.index(split)])


class ContaminationIndex:
    """Sorted array of protected n-gram hashes, searched by binary search.

    A `set` would be faster per lookup but costs roughly 60 bytes per entry;
    a sorted `uint64` array costs 8 and lets a whole document be tested with
    one vectorized `searchsorted`, which is what the corpus builder needs
    when it is streaming millions of web documents.
    """

    def __init__(self, hashes: np.ndarray, provenance: dict):
        if hashes.dtype != np.uint64:
            raise TypeError(f"hash array must be uint64, got {hashes.dtype}")
        # `hit_count` binary-searches this array; an unsorted one would return
        # false negatives in silence rather than failing.
        if hashes.size > 1 and not np.all(hashes[:-1] <= hashes[1:]):
            raise ValueError("hash array must be sorted ascending")
        self._hashes = hashes
        self.provenance = provenance
        self.ngram_size = int(provenance["ngram_size"])

    def __len__(self) -> int:
        return int(self._hashes.size)

    @property
    def splits(self) -> frozenset[str]:
        """Which registry splits this index protects."""
        return frozenset(self.provenance.get("splits", ()))

    @property
    def covered_rows(self) -> int:
        """Protected rows that contributed at least one usable n-gram."""
        return int(self.provenance["covered_rows"])

    @property
    def short_rows(self) -> int:
        """Protected rows that yielded no informative n-gram, so are invisible
        to this index and rest on exact-key matching alone.

        Shortness is only one of the two reasons. A row is also counted here
        when it is long but repetitive enough that `informative` rejects every
        window -- `Simplify (d*d*((d*d**3)/d)/d)/(d/d**6)` is 18 words of
        which 12 are `d`. The name is narrower than what it counts and is kept
        because `v1` stores it; a `v2` registry should call it
        `uncoverable_rows`. Anything deciding coverage must use the builder's
        predicate (`informative_ngram_hashes(...).size > 0`) rather than
        re-deriving one from this name.
        """
        return int(self.provenance["short_rows"])

    # -- construction ----------------------------------------------------

    @classmethod
    def build(
        cls,
        problems: Iterable[str],
        *,
        ngram_size: int = DEFAULT_NGRAM_SIZE,
        splits: Iterable[str] = (),
    ) -> ContaminationIndex:
        if ngram_size < 2:
            raise ValueError(f"ngram_size must be at least 2, got {ngram_size}")
        # Hashes are folded into `uint64` blocks as they accumulate. A flat
        # Python list would cost ~30 bytes an entry, and a large protected
        # pool produces tens of millions of them.
        blocks: list[np.ndarray] = []
        pending: list[np.ndarray] = []
        pending_size = 0
        total = 0
        covered = 0
        degenerate = 0
        for text in problems:
            total += 1
            # The same canonical view `problem_key` hashes. Indexing the raw
            # prompt would put its instruction wrapper into the index: the
            # DAPO preamble alone is 31 words, so 19 pure-boilerplate 13-grams
            # would be protected, and every QA document the corpus builder
            # renders with that template would be refused as contaminated.
            words = index_words(strip_framing(text))
            hashes = informative_ngram_hashes(words, ngram_size)
            raw = ngram_hashes(words, ngram_size)
            degenerate += int(raw.size - hashes.size)
            if hashes.size == 0:
                continue
            covered += 1
            pending.append(hashes)
            pending_size += hashes.size
            if pending_size >= _BLOCK_HASHES:
                blocks.append(np.unique(np.concatenate(pending)))
                pending.clear()
                pending_size = 0
        if pending:
            blocks.append(np.unique(np.concatenate(pending)))
        array = (
            np.unique(np.concatenate(blocks))
            if blocks
            else np.empty(0, dtype=np.uint64)
        )
        provenance = {
            "schema": INDEX_SCHEMA,
            "ngram_size": ngram_size,
            "splits": sorted(splits),
            # Rows, not distinct problems: a pool that repeats each problem
            # for rollout batching contributes one row per repetition, and
            # calling that a problem count would overstate coverage.
            "protected_rows": total,
            "covered_rows": covered,
            "short_rows": total - covered,
            "low_diversity_ngrams": degenerate,
            "max_repeated_word_fraction": MAX_REPEATED_WORD_FRACTION,
            "unique_ngrams": int(array.size),
        }
        return cls(array, provenance)

    # -- querying --------------------------------------------------------

    def hit_count(self, text: str) -> int:
        """Number of distinct protected n-grams the document reproduces."""
        probe = ngram_hashes(index_words(text), self.ngram_size)
        if probe.size == 0 or self._hashes.size == 0:
            return 0
        probe = np.unique(probe)
        position = np.searchsorted(self._hashes, probe)
        position = np.minimum(position, self._hashes.size - 1)
        return int(np.count_nonzero(self._hashes[position] == probe))

    def contaminated(self, text: str, *, min_hits: int = 1) -> bool:
        return self.hit_count(text) >= min_hits

    # -- storage ---------------------------------------------------------

    def write(self, directory: Path) -> str:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "ngrams.npy"
        # `save` on a C-contiguous sorted array is byte-deterministic, so the
        # digest below identifies the index content, not the moment it ran.
        np.save(path, self._hashes, allow_pickle=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        # Record the digest on the live object too, so provenance read off an
        # index that was just built matches provenance read off disk.
        self.provenance["ngrams_sha256"] = digest
        (directory / "index.json").write_text(
            json.dumps(self.provenance, indent=2, sort_keys=True) + "\n"
        )
        return digest

    @classmethod
    def read(cls, directory: Path) -> ContaminationIndex:
        directory = Path(directory)
        path = directory / "ngrams.npy"
        manifest = json.loads((directory / "index.json").read_text())
        if manifest["schema"] != INDEX_SCHEMA:
            raise ValueError(
                f"contamination index schema {manifest['schema']!r} is not "
                f"{INDEX_SCHEMA!r}"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest["ngrams_sha256"]:
            raise ValueError(
                f"{path} hashes to {digest}, index.json records "
                f"{manifest['ngrams_sha256']}; the index is not the one its "
                "manifest describes"
            )
        return cls(np.load(path, allow_pickle=False), manifest)


class ProblemGuard:
    """The single admission test a corpus builder applies to a document.

    Both halves of the check live here so no builder can accidentally wire up
    one and forget the other: the exact test alone was the state of the world
    that let 97% of the GSM8K SFT pool sit inside the pretraining corpus, and
    the n-gram test alone cannot see the short template problems that make up
    most of the RL pool.
    """

    def __init__(
        self,
        registry: ProblemRegistry,
        index: ContaminationIndex,
        *,
        split: str = "pretrain",
        min_ngram_hits: int = 1,
    ):
        self.excluded = registry.excluded_from(split)
        self.index = index
        self.split = split
        self.min_ngram_hits = min_ngram_hits
        # Only splits the registry actually populates need index coverage; an
        # empty split is protected vacuously. Checking this rather than
        # trusting the caller is what stops an index built over `eval` alone
        # from being used as if it protected the RL and SFT pools too, which
        # would leave 99% of the claimed problems on exact matching only.
        self.protected_splits = frozenset(
            name for name in protected_splits(split) if registry.split_counts.get(name)
        )
        missing = self.protected_splits - index.splits
        if missing:
            raise ValueError(
                f"the contamination index covers {sorted(index.splits)}, but a "
                f"guard for {split!r} must refuse problems claimed by "
                f"{sorted(self.protected_splits)}; {sorted(missing)} would be "
                "checked by exact key only. Rebuild the index with "
                f"--index-splits {' '.join(sorted(self.protected_splits))}"
            )

    @classmethod
    def from_problems(
        cls,
        problems: Iterable[str],
        *,
        protected_split: str = "eval",
        ngram_size: int = DEFAULT_NGRAM_SIZE,
        **kwargs,
    ) -> ProblemGuard:
        """An in-memory guard over a literal problem list.

        For tests and one-off checks. Real builds load a written, hashed
        registry so the manifest can record which decontamination rule the
        corpus was actually built under.
        """
        problems = list(problems)
        registry = ProblemRegistry(
            {problem_key(text): protected_split for text in problems},
            {"schema": REGISTRY_SCHEMA, "split_counts": {protected_split: len(problems)}},
        )
        index = ContaminationIndex.build(
            problems, ngram_size=ngram_size, splits=(protected_split,)
        )
        return cls(registry, index, **kwargs)

    def reason(self, text: str, quality_keys: Iterable[str] = ()) -> str | None:
        """Why this document may not enter ``split``, or None to admit it.

        ``quality_keys`` are the document's problem statements when the source
        knows them -- a QA pool yields one per item. Web sources have none, so
        the whole document text is tested as a key too; that costs one hash
        and covers the case where a page is nothing but a problem.
        """
        for key_text in (*quality_keys, text):
            if problem_key(key_text) in self.excluded:
                return "registry_exact"
        if self.index.hit_count(text) >= self.min_ngram_hits:
            return "registry_ngram"
        return None

    def provenance(self, registry: ProblemRegistry) -> dict:
        """What the corpus manifest must record about this admission rule.

        Deliberately without rejection counts. The guard is stateless, and its
        caller already tallies the reasons it returns -- a caller that resumes
        from a journal restores that tally, which a counter living here could
        not do without silently under-reporting the resumed portion of a build.
        """
        return {
            "split": self.split,
            "protected_splits": sorted(self.protected_splits),
            "excluded_problem_keys": len(self.excluded),
            "registry_sha256": registry.provenance.get("registry_sha256"),
            "registry_split_counts": registry.provenance.get("split_counts"),
            "index_sha256": self.index.provenance.get("ngrams_sha256"),
            "index_splits": sorted(self.index.splits),
            "index_ngram_size": self.index.ngram_size,
            "index_unique_ngrams": len(self.index),
            "index_covered_rows": self.index.covered_rows,
            "index_short_rows": self.index.short_rows,
            "min_ngram_hits": self.min_ngram_hits,
        }


def load_guard(
    directory: Path, *, split: str = "pretrain", min_ngram_hits: int = 1
) -> tuple[ProblemGuard, ProblemRegistry]:
    """Read a versioned registry directory and build its admission test."""
    registry = ProblemRegistry.read(directory)
    index = ContaminationIndex.read(directory)
    return (
        ProblemGuard(
            registry, index, split=split, min_ngram_hits=min_ngram_hits
        ),
        registry,
    )
