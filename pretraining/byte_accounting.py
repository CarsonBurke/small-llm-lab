"""Exact byte counts for bits-per-byte, under whichever tokenizer built a corpus.

Bits per byte is the one metric that stays comparable when the tokenizer
changes, which is the whole reason the tokenizer ablation can be read at all.
It is only comparable if the byte count in its denominator is the real byte
length of the validation text -- and that is exactly where a tokenizer swap
breaks silently.

The GPT-2 path uses a precomputed per-token byte-length table, which is valid
because every GPT-2 BPE token maps to a fixed byte string. That assumption
does not survive the numeric scheme in `tokenization/`: a TST token carries
(digits, power), and the bytes it renders to depend on the tokens beside it,
so no per-token table exists. Non-GPT-2 corpora are therefore counted by
decoding the validation sequence itself.

Both paths follow the same convention for registered special tokens -- one
byte each, matching `data/tokenizers/gpt2_byte_lut.pt`, where the end-of-text
token is 1 rather than the 13 bytes of its printed name. Document separators
are structure, not content, and charging them their spelling would make a
corpus with shorter documents look worse for a reason that has nothing to do
with the model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

GPT2_BYTE_LUT = Path("data/tokenizers/gpt2_byte_lut.pt")
SPECIAL_TOKEN_BYTES = 1
GPT2_VOCAB_SIZE = 50_257
GPT2_EOT_ID = 50_256


def padded_vocab_size(manifest: dict, multiple: int = 128) -> int:
    """The embedding width a dataset's token ids require, padded for matmuls."""
    provenance = manifest.get("tokenizer_provenance")
    real = int(provenance["vocab_size"]) if provenance else GPT2_VOCAB_SIZE
    return -(-real // multiple) * multiple


def tokenizer_identity(manifest: dict) -> tuple:
    """What has to match for two corpora to share one embedding table.

    Padded vocabulary size is far too coarse to answer that: it is a 128-wide
    bucket, so GPT-2 and any trained tokenizer within 47 tokens of it compare
    equal while meaning entirely different things by every id.

    The directory and the human-readable name are deliberately excluded. A
    tokenizer copied elsewhere or renamed still produces identical ids, and
    the two content hashes pin the encoding exactly: `spec_sha256` covers the
    vocabulary and numeric scheme, `ngrams_sha256` the counts that split-tree
    inference is rebuilt from.
    """
    provenance = manifest.get("tokenizer_provenance")
    if not provenance:
        # Pre-manifest shards. GPT-2 by construction; see read_dataset_manifest.
        return ("gpt2", GPT2_VOCAB_SIZE, GPT2_EOT_ID, None, None)
    return (
        provenance["kind"],
        int(provenance["vocab_size"]),
        int(provenance["eot_id"]),
        provenance.get("spec_sha256"),
        provenance.get("ngrams_sha256"),
    )


def read_dataset_manifest(data_path: str | Path) -> dict:
    path = Path(data_path) / "mix_manifest.json"
    if not path.exists():
        # The legacy FineWeb shards predate the manifest. They are GPT-2 by
        # construction, and saying so explicitly beats inferring it later.
        return {"tokenizer": "gpt2"}
    return json.loads(path.read_text())


def require_matching_vocab_size(manifest: dict, vocab_size: int) -> None:
    """Fail if the trainer's embedding width disagrees with the corpus.

    A width larger than the corpus needs makes the tail of the embedding table
    unreachable and quietly wastes parameters; a width smaller than the corpus
    needs indexes out of range, or -- worse, if the corpus is small enough --
    does not, and trains on ids that mean something else. Neither is a
    degradation to warn about.
    """
    expected = padded_vocab_size(manifest)
    if vocab_size != expected:
        provenance = manifest.get("tokenizer_provenance") or {}
        raise ValueError(
            f"VOCAB_SIZE={vocab_size} does not match the corpus, which was "
            f"built under tokenizer {manifest.get('tokenizer')!r} with "
            f"{provenance.get('vocab_size', 50257)} tokens (padded: "
            f"{expected}). Token ids mean nothing under the wrong vocabulary."
        )


def load_bound_tokenizer(provenance: dict):
    """Load and verify the exact tokenizer identity recorded by a corpus.

    ``SplitTreeNumericTokenizer.from_directory`` proves that the n-gram file
    matches the tokenizer spec currently beside it. The manifest hashes prove
    that those two internally-consistent files are the *same* files that
    produced the corpus, preventing a retrained directory from silently
    reinterpreting old token IDs.
    """

    if provenance.get("kind") != "toast_tst":
        raise ValueError(
            "a bound custom tokenizer requires kind='toast_tst', got "
            f"{provenance.get('kind')!r}"
        )
    directory = Path(provenance["directory"])
    spec_path = directory / "tokenizer.json"
    spec_digest = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    expected_spec = provenance.get("spec_sha256")
    if spec_digest != expected_spec:
        raise ValueError(
            f"{spec_path} hashes to {spec_digest}, but the corpus/checkpoint "
            f"requires {expected_spec}; token ids would be reinterpreted"
        )
    from tokenization.spec import TokenizerSpec
    from tokenization.tokenizer import SplitTreeNumericTokenizer

    spec = TokenizerSpec.read(spec_path)
    ngrams_path = directory / spec.ngrams.filename
    ngrams_digest = hashlib.sha256(ngrams_path.read_bytes()).hexdigest()
    expected_ngrams = provenance.get("ngrams_sha256")
    if ngrams_digest != expected_ngrams:
        raise ValueError(
            f"{ngrams_path} hashes to {ngrams_digest}, but the "
            f"corpus/checkpoint requires {expected_ngrams}; token ids would "
            "be reinterpreted"
        )
    tokenizer = SplitTreeNumericTokenizer.from_directory(directory)
    if tokenizer.vocab_size != int(provenance["vocab_size"]):
        raise ValueError(
            f"{directory} has {tokenizer.vocab_size} tokens, but provenance "
            f"requires {provenance['vocab_size']}"
        )
    if tokenizer.eot_id != int(provenance["eot_id"]):
        raise ValueError(
            f"{directory} has end-of-text id {tokenizer.eot_id}, but "
            f"provenance requires {provenance['eot_id']}"
        )
    return tokenizer


class ByteCounter:
    """Counts the bytes a stretch of token ids represents.

    Constructed from the dataset manifest rather than from a flag, because the
    corpus is the only thing that knows which vocabulary produced its ids.
    """

    def __init__(self, manifest: dict, device=None):
        self.manifest = manifest
        provenance = manifest.get("tokenizer_provenance")
        # `tokenizer_provenance.directory` is the discriminator, not the name:
        # a trained tokenizer's name is its directory basename and can be
        # anything at all, including "gpt2".
        self.is_gpt2 = not provenance or provenance.get("directory") is None
        self.lut = None
        self.tokenizer = None
        self.special_ids: frozenset[int] = frozenset()
        if self.is_gpt2:
            import torch

            lut = torch.load(GPT2_BYTE_LUT, weights_only=True)
            if device is not None:
                lut = lut.to(device)
            self.lut = lut
            return
        self.tokenizer = load_bound_tokenizer(provenance)
        self.special_ids = frozenset(range(len(self.tokenizer.spec.specials)))

    def expected_lut_size(self) -> int | None:
        return None if self.lut is None else int(self.lut.numel())

    def count(self, targets) -> int:
        """UTF-8 bytes represented by a flat tensor of target token ids."""
        if self.lut is not None:
            flat = targets.reshape(-1)
            if self.lut.numel() <= int(flat.max()):
                raise IndexError(
                    f"token id {int(flat.max())} is outside the "
                    f"{self.lut.numel()}-entry GPT-2 byte table; this corpus "
                    "was not built with the GPT-2 tokenizer"
                )
            import torch

            # int64 before summing: the table is int32 and a validation split
            # is millions of tokens.
            return int(self.lut[flat].to(torch.int64).sum())
        return self.count_ids(targets.reshape(-1).tolist())

    def count_ids(self, ids: list[int]) -> int:
        """Decode-based counting, exact for a context-dependent numeric scheme.

        Runs of ordinary tokens are decoded together, because a TST token's
        bytes depend on its neighbours and decoding it alone would not give
        the length it actually contributes.
        """
        if self.lut is not None:
            return int(sum(int(self.lut[token]) for token in ids))
        total = 0
        run: list[int] = []
        for token in ids:
            if token in self.special_ids:
                if run:
                    total += len(self.tokenizer.decode(run).encode("utf-8"))
                    run = []
                total += SPECIAL_TOKEN_BYTES
            else:
                run.append(token)
        if run:
            total += len(self.tokenizer.decode(run).encode("utf-8"))
        return total
