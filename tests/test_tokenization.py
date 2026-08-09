"""Deterministic contracts for the combined ToaST + TST tokenizer.

These are CPU/static tests: no model executes, so they run directly rather than
through mlq.
"""

from __future__ import annotations

import itertools
import json
import random

import pytest

from tokenization import ngram_store
from tokenization.spec import (
    DEFAULT_SPECIALS,
    PRETOKEN_PATTERN,
    NgramReference,
    TokenizerSpec,
)
from tokenization.split_tree import (
    best_split,
    build_split_tree,
    candidate_tokens,
    count_ngrams,
    most_known_split,
    tree_tokens,
)
from tokenization.tokenizer import SplitTreeNumericTokenizer
from tokenization.train import BYTE_ALPHABET, build_tokenizer
from tokenization.tst import NumericOverflowError, NumericScheme
from tokenization.vocab_lp import select_vocabulary


# --------------------------------------------------------------------------
# Triadic Suffix Tokenization
# --------------------------------------------------------------------------


def numeric_cases(seed: int = 0, count: int = 400) -> list[str]:
    rng = random.Random(seed)
    cases = [
        "0",
        "00",
        "000",
        "1",
        "1234",
        "01234",
        "110911",
        "100400",
        "1234567",
        "123456789012345",
        "0.1",
        "0.10",
        "0.100",
        "0.0045",
        "1.12345678",
        "9.11",
        "9.9",
        "0.00450",
        "1000000",
    ]
    for _ in range(count):
        digits = rng.randint(1, 19)
        cases.append(str(rng.randrange(10**digits)).zfill(rng.randint(1, digits)))
    for _ in range(count):
        integer = str(rng.randrange(10**6))
        fraction = str(rng.randrange(10**12)).zfill(rng.randint(1, 15))[:15]
        cases.append(f"{integer}.{fraction}")
    return cases


@pytest.mark.parametrize("group_size", [1, 2, 3, 4])
def test_numeric_round_trip_is_exact(group_size: int) -> None:
    scheme = NumericScheme(group_size=group_size)
    for span in numeric_cases():
        groups = scheme.groups_for_span(span)
        assert scheme.span_for_groups(groups) == span, span


@pytest.mark.parametrize("group_size", [1, 2, 3])
def test_every_emitted_group_is_in_the_enumerated_vocabulary(
    group_size: int,
) -> None:
    scheme = NumericScheme(group_size=group_size)
    vocabulary = set(scheme.group_vocabulary())
    for span in numeric_cases():
        for group in scheme.groups_for_span(span):
            assert group in vocabulary, (span, group)


def test_triadic_examples_match_the_paper() -> None:
    scheme = NumericScheme(group_size=3)
    rendered = lambda span: [  # noqa: E731 - table-driven readability
        scheme.render_group(*group) for group in scheme.groups_for_span(span)
    ]
    assert rendered("100400") == ["100k", "400"]
    assert rendered("1234567") == ["1m", "234k", "567"]
    assert rendered("123456789012345") == [
        "123t",
        "456b",
        "789m",
        "012k",
        "345",
    ]
    # The paper pads the trailing fractional group to `780ppp`; this scheme
    # keeps `78ppp` so that `0.1`, `0.10` and `0.100` stay distinguishable.
    assert rendered("1.12345678") == ["1", "123p", "456pp", "78ppp"]


def test_group_value_is_digits_times_ten_to_the_power() -> None:
    scheme = NumericScheme(group_size=3)
    for span in ["1234567", "0.0045", "100400", "12.5"]:
        total = sum(
            int(digits) * 10**power
            for digits, power in scheme.groups_for_span(span)
        )
        assert abs(total - float(span)) < 1e-9, span


def test_padded_mode_is_the_papers_lossy_canonical_form() -> None:
    scheme = NumericScheme(group_size=3, leading_zero_padding=True)
    padded = scheme.groups_for_span("0.1")
    assert [scheme.render_group(*g) for g in padded] == ["000", "100p"]
    # Exactly the collision the default mode exists to avoid.
    assert padded == scheme.groups_for_span("0.100")


def test_out_of_range_spans_are_refused_not_truncated() -> None:
    scheme = NumericScheme(group_size=1, max_int_digits=4, max_frac_digits=2)
    assert scheme.accepts("1234", "12")
    assert not scheme.accepts("12345", "")
    assert not scheme.accepts("1", "234")
    with pytest.raises(NumericOverflowError):
        scheme.groups_for_span("12345")


def test_decoder_rejects_group_sequences_the_encoder_cannot_produce() -> None:
    scheme = NumericScheme(group_size=3)
    with pytest.raises(ValueError):
        scheme.span_for_groups([("1", 0), ("2", 0)])
    with pytest.raises(ValueError):
        scheme.span_for_groups([("1", 6), ("2", 0)])
    with pytest.raises(ValueError):
        scheme.span_for_groups([])


def test_option_a_vocabulary_is_far_smaller_than_option_b() -> None:
    compound = NumericScheme(group_size=3, compound=True)
    separate = NumericScheme(group_size=3, compound=False)
    spec_b = _spec_for(compound, ())
    spec_a = _spec_for(separate, ())
    assert spec_a.numeric_count < spec_b.numeric_count
    # The N=1 setting the paper hypothesises for small models is cheap enough
    # to carry in a small embedding table.
    assert _spec_for(NumericScheme(group_size=1), ()).numeric_count == 340


# --------------------------------------------------------------------------
# Split trees
# --------------------------------------------------------------------------


def test_best_split_maximises_the_minimum_and_breaks_ties_leftmost() -> None:
    counts = {b"ab": 5, b"cd": 5, b"a": 9, b"bcd": 4, b"abc": 5, b"d": 5}
    # Split 2 scores min(5, 5) = 5; split 1 scores min(9, 4) = 4; split 3
    # scores min(5, 5) = 5 as well, so the leftmost of the two wins.
    assert best_split(b"abcd", counts) == 2


def test_best_split_falls_back_when_no_split_has_both_parts() -> None:
    counts = {b"a": 3, b"ab": 3}
    assert best_split(b"abc", counts) == most_known_split(b"abc", counts)


def test_most_known_split_always_makes_progress() -> None:
    # No prefix is known at all: the listing would return 0 and recurse
    # forever on the same span.
    assert most_known_split(b"xy", {}) == 1
    assert 1 <= most_known_split(b"xyz", {}) <= 2
    # Every prefix known: the listing would return None.
    counts = {b"x": 1, b"xy": 1, b"xyz": 1}
    assert most_known_split(b"xyz", counts) == 2


def test_split_tree_is_a_full_binary_tree_over_bytes() -> None:
    counts = count_ngrams([(b"kentucky", 100), (b"kentish", 40)], min_count=1, max_length=8)
    tree = build_split_tree(b"kentucky", counts)
    assert len(tree.nodes) == 2 * len(b"kentucky") - 1
    leaves = tree.leaves()
    assert len(leaves) == len(b"kentucky")
    assert [tree.segment(leaf) for leaf in leaves] == [
        bytes([byte]) for byte in b"kentucky"
    ]
    for index, node in enumerate(tree.nodes):
        if node.is_leaf:
            continue
        left, right = tree.nodes[node.left], tree.nodes[node.right]
        assert left.start == node.start and right.end == node.end
        assert left.end == right.start
        assert tree.nodes[node.left].parent == index


def test_removing_a_token_only_expands_that_node() -> None:
    """The paper's key structural property: no cascading effects."""
    counts = count_ngrams(
        [(b"kentucky", 100), (b"kentish", 40), (b"lucky", 70)],
        min_count=1,
        max_length=8,
    )
    tree = build_split_tree(b"kentucky", counts)
    full = frozenset(candidate_tokens(tree))
    with_all = tree_tokens(tree, full)
    for removed in with_all:
        if len(removed) == 1:
            continue
        reduced = tree_tokens(tree, full - {removed})
        # Every other emitted token is untouched, and the removed one is
        # replaced by a contiguous expansion of itself.
        assert b"".join(reduced) == b"".join(with_all)
        assert len(reduced) > len(with_all)
        for token in with_all:
            if token != removed:
                assert token in reduced


def test_inference_is_total_with_only_the_byte_alphabet() -> None:
    counts = count_ngrams([(b"hello world", 5)], min_count=1, max_length=6)
    tree = build_split_tree(b"hello", counts)
    alphabet = frozenset(bytes([b]) for b in range(256))
    assert tree_tokens(tree, alphabet) == [bytes([b]) for b in b"hello"]


def test_count_ngrams_respects_boundaries_and_thresholds() -> None:
    counts = count_ngrams([(b"aba", 3), (b"bab", 2)], min_count=5, max_length=3)
    assert counts[b"a"] == 3 * 2 + 2 * 1
    assert counts[b"b"] == 3 * 1 + 2 * 2
    assert counts[b"ab"] == 3 + 2
    assert counts[b"ba"] == 3 + 2
    # Below the threshold, and never spanning two pretokens.
    assert b"aba" not in counts
    assert b"abab" not in counts


# --------------------------------------------------------------------------
# Vocabulary selection
# --------------------------------------------------------------------------


def test_lp_selection_matches_brute_force_on_a_small_instance() -> None:
    corpus = [(b"aab", 40), (b"aac", 25), (b"bab", 12), (b"cab", 7)]
    counts = count_ngrams(corpus, min_count=1, max_length=3)
    trees = [build_split_tree(pretoken, counts) for pretoken, _ in corpus]
    weights = [count for _, count in corpus]
    alphabet = tuple(sorted({bytes([b]) for pretoken, _ in corpus for b in pretoken}))

    candidates = sorted(set().union(*(candidate_tokens(t) for t in trees)) - set(alphabet))
    for size in (len(alphabet) + 1, len(alphabet) + 2):
        solution = select_vocabulary(
            trees, weights, size=size, forced=alphabet
        )
        best = None
        for extra in itertools.combinations(candidates, size - len(alphabet)):
            vocabulary = frozenset(alphabet) | frozenset(extra)
            total = sum(
                len(tree_tokens(tree, vocabulary)) * weight
                for tree, weight in zip(trees, weights, strict=True)
            )
            best = total if best is None else min(best, total)
        assert solution.rounded_objective == best, size
        assert len(solution.vocabulary) == size
        assert set(alphabet) <= set(solution.vocabulary)


def test_lp_rejects_a_budget_below_the_forced_alphabet() -> None:
    counts = count_ngrams([(b"ab", 2)], min_count=1, max_length=2)
    trees = [build_split_tree(b"ab", counts)]
    with pytest.raises(ValueError, match="cannot hold"):
        select_vocabulary(trees, [2], size=1, forced=(b"a", b"b"))


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def test_ngram_store_round_trips_and_hashes_deterministically(tmp_path) -> None:
    counts = {b"a": 3, b"ab": 12, b"\xff\x00": 7}
    first = ngram_store.write_counts(tmp_path / "a.bin", counts)
    second = ngram_store.write_counts(
        tmp_path / "b.bin", dict(reversed(list(counts.items())))
    )
    assert first == second, "insertion order must not change the file image"
    assert ngram_store.read_counts(tmp_path / "a.bin") == counts
    assert ngram_store.file_sha256(tmp_path / "a.bin") == first


def test_ngram_store_rejects_a_truncated_file(tmp_path) -> None:
    path = tmp_path / "a.bin"
    ngram_store.write_counts(path, {b"a": 3, b"ab": 12})
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError):
        ngram_store.read_counts(path)


# --------------------------------------------------------------------------
# End-to-end tokenizer
# --------------------------------------------------------------------------


def _spec_for(scheme: NumericScheme, text_tokens: tuple[bytes, ...]) -> TokenizerSpec:
    return TokenizerSpec(
        numeric=scheme,
        text_tokens=text_tokens,
        ngrams=NgramReference(
            filename="ngrams.bin",
            sha256="0" * 64,
            entries=0,
            min_count=1,
            max_length=1,
        ),
        specials=DEFAULT_SPECIALS,
        pretoken_pattern=PRETOKEN_PATTERN,
    )


@pytest.fixture(scope="module")
def trained() -> tuple[TokenizerSpec, dict[bytes, int]]:
    rng = random.Random(11)
    stems = [
        "token",
        "compress",
        "vocabul",
        "arithmet",
        "magnitud",
        "recursive",
        "corpus",
        "inference",
        "boundary",
        "fraction",
    ]
    endings = ["ation", "ary", "ic", "ed", "ing", "es", "al", "ism", "ist", "ously"]
    documents = []
    for index in range(600):
        a, b = rng.randrange(10**5), rng.randrange(10**3)
        documents.append(f"What is {a} + {b}?\nAnswer: {a + b}")
        words = " ".join(
            rng.choice(stems) + rng.choice(endings) for _ in range(12)
        )
        documents.append(
            f"Tokenization is an important first step: {words}, "
            f"discussed in section {index}."
        )
        documents.append("Naïve café — résumé, 数学、そして φ = 1.618033988749895.")
    spec, counts, _ = build_tokenizer(
        documents=iter(documents),
        scheme=NumericScheme(group_size=1),
        vocab_size=845,
        specials=DEFAULT_SPECIALS,
        min_count=2,
        max_ngram=12,
        max_trees=4000,
        pattern=PRETOKEN_PATTERN,
        log=lambda message: None,
    )
    return spec, counts


ROUND_TRIP_TEXTS = [
    "",
    "hello",
    "What is -0.188 + -0.814?\nAnswer: -1.002",
    "Work out 4 * 4.45.\nAnswer: 17.8",
    "The value 9.11 is less than 9.9.",
    "0.1 0.10 0.100 are three different strings",
    "id=12345678901234567890123456789 exceeds the numeric range",
    "1.2.3 and 1,234 and 007 and 1e9",
    "Naïve café — résumé, 数学、そして φ = 1.618033988749895.",
    "<think>2 + 2 = 4</think><answer>4</answer>",
    "trailing whitespace   \n\n\ttabbed\n",
    "emoji 🙂 and a surrogate-free string",
    "-5 - 110911 = -110916",
]


@pytest.mark.parametrize("text", ROUND_TRIP_TEXTS)
def test_encode_decode_round_trip(trained, text: str) -> None:
    tokenizer = SplitTreeNumericTokenizer(*trained)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_round_trip_on_random_mixed_text(trained) -> None:
    tokenizer = SplitTreeNumericTokenizer(*trained)
    rng = random.Random(5)
    alphabet = "abc XYZ 0123456789.,-\n\t é数🙂"
    for _ in range(300):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        assert tokenizer.decode(tokenizer.encode(text)) == text, repr(text)


def test_numbers_never_reach_the_text_path(trained) -> None:
    spec, counts = trained
    tokenizer = SplitTreeNumericTokenizer(spec, counts)
    ids = tokenizer.encode("value 4.45 here")
    kinds = [
        "numeric" if spec.numeric_base <= i < spec.text_base else "other"
        for i in ids
    ]
    assert kinds.count("numeric") == len(
        spec.numeric.groups_for_span("4.45")
    )


def test_out_of_range_numbers_fall_back_to_text_losslessly(trained) -> None:
    spec, counts = trained
    tokenizer = SplitTreeNumericTokenizer(spec, counts)
    long_number = "9" * (spec.numeric.max_int_digits + 5)
    ids = tokenizer.encode(f"x{long_number}y")
    assert all(not (spec.numeric_base <= i < spec.text_base) for i in ids)
    assert tokenizer.decode(ids) == f"x{long_number}y"


def test_special_tokens_are_atomic_and_stable(trained) -> None:
    spec, counts = trained
    tokenizer = SplitTreeNumericTokenizer(spec, counts)
    for index, special in enumerate(spec.specials):
        assert tokenizer.encode(special) == [index]
    # Special ids must not move when the text vocabulary changes.
    smaller = TokenizerSpec(
        numeric=spec.numeric,
        text_tokens=spec.text_tokens[: len(spec.text_tokens) // 2],
        ngrams=spec.ngrams,
        specials=spec.specials,
        pretoken_pattern=spec.pretoken_pattern,
    )
    assert smaller.eot_id() == spec.eot_id()
    assert smaller.numeric_base == spec.numeric_base


def test_encoding_is_deterministic(trained) -> None:
    spec, counts = trained
    text = "Calculate -0.2*0.57 and then 12345 + 6789."
    first = SplitTreeNumericTokenizer(spec, counts).encode(text)
    second = SplitTreeNumericTokenizer(spec, counts).encode(text)
    assert first == second


def test_spec_serialization_is_stable(trained, tmp_path) -> None:
    spec, _ = trained
    path = tmp_path / "tokenizer.json"
    spec.write(path)
    reloaded = TokenizerSpec.read(path)
    assert reloaded.sha256() == spec.sha256()
    assert reloaded.text_tokens == spec.text_tokens
    assert reloaded.vocab_size == spec.vocab_size


def test_spec_rejects_a_layout_that_does_not_reconstruct(trained) -> None:
    spec, _ = trained
    payload = json.loads(spec.dumps())
    payload["layout"]["vocab_size"] += 1
    with pytest.raises(ValueError, match="reconstructs"):
        TokenizerSpec.from_json(payload)


def test_vocabulary_contains_the_whole_byte_alphabet(trained) -> None:
    spec, _ = trained
    assert set(BYTE_ALPHABET) <= set(spec.text_tokens)


def test_trained_tokenizer_beats_the_byte_alphabet_on_compression(trained) -> None:
    spec, counts = trained
    tokenizer = SplitTreeNumericTokenizer(spec, counts)
    text = "Tokenization is an important first step in many natural language tasks."
    assert len(tokenizer.encode(text)) < len(text.encode("utf-8"))


def write_directory(spec: TokenizerSpec, counts, directory) -> None:
    """Lay a trained tokenizer out the way `tokenization/train.py` does."""
    directory.mkdir(parents=True, exist_ok=True)
    digest = ngram_store.write_counts(directory / spec.ngrams.filename, counts)
    bound = TokenizerSpec(
        numeric=spec.numeric,
        text_tokens=spec.text_tokens,
        ngrams=NgramReference(
            filename=spec.ngrams.filename,
            sha256=digest,
            entries=len(counts),
            min_count=spec.ngrams.min_count,
            max_length=spec.ngrams.max_length,
        ),
        specials=spec.specials,
        pretoken_pattern=spec.pretoken_pattern,
        provenance=spec.provenance,
    )
    bound.write(directory / "tokenizer.json")


def test_from_directory_round_trips_a_written_tokenizer(trained, tmp_path) -> None:
    spec, counts = trained
    write_directory(spec, counts, tmp_path / "tok")
    loaded = SplitTreeNumericTokenizer.from_directory(tmp_path / "tok")
    text = "Tokenization is an important first step: 41277 + 8163 = 49440."
    assert loaded.decode(loaded.encode(text)) == text
    assert loaded.encode(text) == SplitTreeNumericTokenizer(spec, counts).encode(text)


def test_from_directory_refuses_counts_the_spec_does_not_declare(
    trained, tmp_path
) -> None:
    """The counts are part of the encoding, not a discardable training artefact.

    Split trees are rebuilt from them at encode time, so a directory whose
    counts were swapped underneath its `tokenizer.json` produces different ids
    while still hashing to the `spec_sha256` that binds a corpus to its
    tokenizer. Loading it would silently reinterpret every shard.
    """
    spec, counts = trained
    directory = tmp_path / "tok"
    write_directory(spec, counts, directory)
    swapped = dict(counts)
    swapped[next(iter(sorted(swapped)))] += 1
    ngram_store.write_counts(directory / spec.ngrams.filename, swapped)
    with pytest.raises(ValueError, match="would not reproduce its own"):
        SplitTreeNumericTokenizer.from_directory(directory)
