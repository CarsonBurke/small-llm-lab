"""Byte accounting and vocabulary identity for the tokenizer ablation.

Bits per byte is the only metric that survives a tokenizer swap, so the four
arms of the corpus-v8 ablation are compared through it. That comparison is
only meaningful if the byte denominator is the real byte length of the
validation text under whichever tokenizer built the corpus, and if a corpus is
never trained under a vocabulary other than its own.

These are CPU/static tests: no model executes, so they run directly rather
than through mlq.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from pretraining.byte_accounting import (
    GPT2_BYTE_LUT,
    GPT2_EOT_ID,
    ByteCounter,
    padded_vocab_size,
    read_dataset_manifest,
    require_matching_vocab_size,
    tokenizer_identity,
)
from tokenization import ngram_store
from tokenization.spec import (
    DEFAULT_SPECIALS,
    END_OF_TEXT,
    PRETOKEN_PATTERN,
    NgramReference,
    TokenizerSpec,
)
from tokenization.tokenizer import SplitTreeNumericTokenizer
from tokenization.train import BYTE_ALPHABET
from tokenization.tst import NumericScheme

EXTRA_TOKENS = (b"the", b"ing", b"hello", b"wor")


def write_tiny_tokenizer(directory) -> TokenizerSpec:
    """A real, loadable tokenizer directory without paying for the LP solve.

    The vocabulary integer program is what makes a *good* tokenizer; nothing
    here depends on the vocabulary being good, only on it being complete, so
    the byte alphabet plus a few n-grams is enough to encode and decode
    anything.
    """
    directory.mkdir(parents=True, exist_ok=True)
    counts = {gram: 8 for gram in BYTE_ALPHABET}
    counts.update({token: 64 for token in EXTRA_TOKENS})
    digest = ngram_store.write_counts(directory / "ngrams.bin", counts)
    spec = TokenizerSpec(
        numeric=NumericScheme(group_size=1),
        text_tokens=BYTE_ALPHABET + EXTRA_TOKENS,
        ngrams=NgramReference(
            filename="ngrams.bin",
            sha256=digest,
            entries=len(counts),
            min_count=1,
            max_length=8,
        ),
        specials=DEFAULT_SPECIALS,
        pretoken_pattern=PRETOKEN_PATTERN,
    )
    spec.write(directory / "tokenizer.json")
    return spec


@pytest.fixture(scope="module")
def tiny_tokenizer_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tokenizers") / "tiny_toast_tst"
    write_tiny_tokenizer(directory)
    return directory


def toast_manifest(directory, spec: TokenizerSpec) -> dict:
    return {
        "tokenizer": directory.name,
        "tokenizer_provenance": {
            "kind": "toast_tst",
            "name": directory.name,
            "vocab_size": spec.vocab_size,
            "eot_id": spec.eot_id(),
            "directory": str(directory),
            "spec_sha256": hashlib.sha256(
                (directory / "tokenizer.json").read_bytes()
            ).hexdigest(),
            "ngrams_sha256": spec.ngrams.sha256,
        },
    }


def test_bound_tokenizer_refuses_a_replaced_spec(tiny_tokenizer_dir) -> None:
    spec_path = tiny_tokenizer_dir / "tokenizer.json"
    spec = TokenizerSpec.read(spec_path)
    manifest = toast_manifest(tiny_tokenizer_dir, spec)
    original = spec_path.read_bytes()
    try:
        spec_path.write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="token ids would be reinterpreted"):
            ByteCounter(manifest)
    finally:
        spec_path.write_bytes(original)


# --------------------------------------------------------------------------
# Vocabulary width
# --------------------------------------------------------------------------


def test_padded_vocab_size_comes_from_the_manifest() -> None:
    assert padded_vocab_size({}) == 50_304
    assert padded_vocab_size({"tokenizer_provenance": {"vocab_size": 50257}}) == 50_304
    assert padded_vocab_size({"tokenizer_provenance": {"vocab_size": 16384}}) == 16_384
    assert padded_vocab_size({"tokenizer_provenance": {"vocab_size": 16385}}) == 16_512


def test_read_dataset_manifest_names_the_legacy_tokenizer(tmp_path) -> None:
    """The pre-manifest FineWeb shards are GPT-2 by construction."""
    assert read_dataset_manifest(tmp_path) == {"tokenizer": "gpt2"}
    (tmp_path / "mix_manifest.json").write_text(json.dumps({"tokenizer": "tiny"}))
    assert read_dataset_manifest(tmp_path)["tokenizer"] == "tiny"


def test_require_matching_vocab_size_refuses_either_mismatch() -> None:
    manifest = {"tokenizer_provenance": {"vocab_size": 16384}}
    require_matching_vocab_size(manifest, 16_384)
    # Too small indexes out of range, or worse, silently does not.
    with pytest.raises(ValueError, match="does not match the corpus"):
        require_matching_vocab_size(manifest, 8_192)
    # Too large leaves the tail of the embedding table unreachable.
    with pytest.raises(ValueError, match="does not match the corpus"):
        require_matching_vocab_size(manifest, 50_304)
    require_matching_vocab_size({"tokenizer": "gpt2"}, 50_304)


# --------------------------------------------------------------------------
# Vocabulary identity
# --------------------------------------------------------------------------


def test_identity_separates_vocabularies_that_share_a_padding_bucket() -> None:
    """The regression this function exists for.

    Padded size is a 128-wide bucket. A trained tokenizer sized anywhere in
    50,177..50,304 pads to exactly GPT-2's 50,304 while meaning something
    different by every single id, so a padded-size comparison would wave a
    catastrophic mismatch through as if it were a match.
    """
    gpt2 = {"tokenizer": "gpt2"}
    imposter = {
        "tokenizer": "gpt2",
        "tokenizer_provenance": {
            "kind": "toast_tst",
            "name": "gpt2",
            "vocab_size": 50_257,
            "eot_id": 0,
            "directory": "data/tokenizers/toast_tst_50k",
            "spec_sha256": "a" * 64,
            "ngrams_sha256": "b" * 64,
        },
    }
    assert padded_vocab_size(imposter) == padded_vocab_size(gpt2)
    assert tokenizer_identity(imposter) != tokenizer_identity(gpt2)


def test_identity_ignores_where_a_tokenizer_is_stored() -> None:
    """A copy or a rename produces identical ids and must compare equal."""
    provenance = {
        "kind": "toast_tst",
        "name": "toast_tst_32k",
        "vocab_size": 32_768,
        "eot_id": 0,
        "directory": "data/tokenizers/toast_tst_32k",
        "spec_sha256": "c" * 64,
        "ngrams_sha256": "d" * 64,
    }
    moved = dict(provenance, name="copy", directory="/elsewhere/copy")
    assert tokenizer_identity({"tokenizer_provenance": provenance}) == (
        tokenizer_identity({"tokenizer_provenance": moved})
    )


def test_identity_tracks_the_counts_as_well_as_the_spec() -> None:
    """Split trees are rebuilt from the counts, so the counts are the encoding."""
    provenance = {
        "kind": "toast_tst",
        "name": "toast_tst_32k",
        "vocab_size": 32_768,
        "eot_id": 0,
        "directory": "data/tokenizers/toast_tst_32k",
        "spec_sha256": "c" * 64,
        "ngrams_sha256": "d" * 64,
    }
    other = dict(provenance, ngrams_sha256="e" * 64)
    assert tokenizer_identity({"tokenizer_provenance": provenance}) != (
        tokenizer_identity({"tokenizer_provenance": other})
    )


def test_legacy_shards_are_the_same_tokenizer_as_a_declared_gpt2_corpus() -> None:
    declared = {
        "tokenizer": "gpt2",
        "tokenizer_provenance": {
            "kind": "gpt2",
            "name": "gpt2",
            "vocab_size": 50_257,
            "eot_id": GPT2_EOT_ID,
            "directory": None,
            "spec_sha256": None,
            "ngrams_sha256": None,
        },
    }
    assert tokenizer_identity(declared) == tokenizer_identity({"tokenizer": "gpt2"})


# --------------------------------------------------------------------------
# Byte counting
# --------------------------------------------------------------------------


def test_gpt2_counter_uses_the_lookup_table() -> None:
    counter = ByteCounter({"tokenizer": "gpt2"})
    assert counter.is_gpt2
    assert counter.expected_lut_size() == 50_304
    lut = torch.load(GPT2_BYTE_LUT, weights_only=True)
    ids = [15496, 995, 13, 50256]
    expected = int(sum(int(lut[token]) for token in ids))
    assert counter.count(torch.tensor(ids, dtype=torch.int64)) == expected
    assert counter.count_ids(ids) == expected
    # Document separators are structure, not content.
    assert int(lut[50256]) == 1


def test_gpt2_counter_sums_in_int64() -> None:
    """A validation split is millions of tokens; int32 accumulation wraps."""
    counter = ByteCounter({"tokenizer": "gpt2"})
    ids = torch.full((4_000_000,), 15496, dtype=torch.int64)
    lut = torch.load(GPT2_BYTE_LUT, weights_only=True)
    assert counter.count(ids) == 4_000_000 * int(lut[15496])


def test_gpt2_counter_refuses_ids_from_another_vocabulary() -> None:
    counter = ByteCounter({"tokenizer": "gpt2"})
    with pytest.raises(IndexError, match="outside the"):
        counter.count(torch.tensor([1, 60_000], dtype=torch.int64))


def test_the_counter_discriminates_on_the_directory_not_the_name(
    tiny_tokenizer_dir,
) -> None:
    """A trained tokenizer's name is a directory basename and can be "gpt2"."""
    spec = TokenizerSpec.read(tiny_tokenizer_dir / "tokenizer.json")
    manifest = toast_manifest(tiny_tokenizer_dir, spec)
    manifest["tokenizer"] = "gpt2"
    manifest["tokenizer_provenance"]["name"] = "gpt2"
    counter = ByteCounter(manifest)
    assert not counter.is_gpt2
    assert counter.expected_lut_size() is None


@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "What is 41277 + 8163?\nAnswer: 49440",
        "Naïve café — 数学 φ = 1.618033988749895.",
        "0.1 0.10 0.100 are three different strings",
    ],
)
def test_decoding_counter_is_exact_for_the_numeric_scheme(
    tiny_tokenizer_dir, text: str
) -> None:
    """TST token bytes depend on their neighbours, so no per-token table exists.

    A magnitude-carrying token renders differently according to what follows
    it, which is exactly why this path decodes instead of summing a table.
    """
    tokenizer = SplitTreeNumericTokenizer.from_directory(tiny_tokenizer_dir)
    spec = TokenizerSpec.read(tiny_tokenizer_dir / "tokenizer.json")
    counter = ByteCounter(toast_manifest(tiny_tokenizer_dir, spec))
    ids = tokenizer.encode(text)
    assert counter.count_ids(ids) == len(text.encode("utf-8"))
    assert counter.count(torch.tensor(ids, dtype=torch.int64)) == len(
        text.encode("utf-8")
    )


def test_specials_cost_one_byte_each_under_both_counters(tiny_tokenizer_dir) -> None:
    """The convention is arbitrary; what matters is that it is arm-neutral.

    A separator is structure, not content, so any fixed charge for it is
    defensible -- one byte matches the existing GPT-2 table, and that is the
    reason for the number, not the reason the convention is sound.

    It is sound because both counters charge the true UTF-8 length for content
    and the same constant for the separator, and both arms' validation sets
    hold the same documents: the validation split is keyed on
    `stable_digest(document.text)`, a hash of TEXT, so it selects identically
    under every tokenizer. The constant then contributes an identical offset
    to both arms and cancels in the comparison. If the split ever moves to
    hashing token ids, that cancellation goes away and this convention stops
    being neutral.
    """
    tokenizer = SplitTreeNumericTokenizer.from_directory(tiny_tokenizer_dir)
    spec = TokenizerSpec.read(tiny_tokenizer_dir / "tokenizer.json")
    counter = ByteCounter(toast_manifest(tiny_tokenizer_dir, spec))
    ids = tokenizer.encode(f"ab{END_OF_TEXT}cd{END_OF_TEXT}")
    assert counter.count_ids(ids) == 2 + 1 + 2 + 1


def test_the_decoding_counter_handles_a_leading_and_only_special(
    tiny_tokenizer_dir,
) -> None:
    spec = TokenizerSpec.read(tiny_tokenizer_dir / "tokenizer.json")
    counter = ByteCounter(toast_manifest(tiny_tokenizer_dir, spec))
    assert counter.count_ids([]) == 0
    assert counter.count_ids([spec.eot_id()]) == 1


def test_the_decoding_counter_flushes_runs_around_multi_byte_neighbours(
    tiny_tokenizer_dir,
) -> None:
    """The run-flush boundary is where a decode-based counter would go wrong.

    `count_ids` decodes runs of ordinary tokens together because a TST token's
    bytes depend on its neighbours. A special in the middle of multi-byte text
    forces a flush, and a flush that dropped or double-counted the partial run
    would only show up when the surrounding characters are wider than one
    byte.
    """
    tokenizer = SplitTreeNumericTokenizer.from_directory(tiny_tokenizer_dir)
    spec = TokenizerSpec.read(tiny_tokenizer_dir / "tokenizer.json")
    counter = ByteCounter(toast_manifest(tiny_tokenizer_dir, spec))
    for text in (
        f"数学{END_OF_TEXT}φ",
        f"{END_OF_TEXT}café{END_OF_TEXT}",
        f"🙂{END_OF_TEXT}{END_OF_TEXT}🙂",
        f"1.618{END_OF_TEXT}0.100",
    ):
        ids = tokenizer.encode(text)
        specials = text.count(END_OF_TEXT)
        content = len(text.replace(END_OF_TEXT, "").encode("utf-8"))
        assert counter.count_ids(ids) == content + specials, text
