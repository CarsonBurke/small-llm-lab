"""Serializable definition of a combined ToaST + TST tokenizer.

The spec is the tokenizer contract: identical spec bytes and identical n-gram
counts mean identical token ids for identical text. Both hashes are recorded so
a corpus manifest, a checkpoint, and a resume can all be bound to one exact
tokenizer, the way this repository binds every other input.

Token id layout is fixed and stable:

1. special tokens, so their ids never move when the text vocabulary changes;
2. numeric tokens, enumerated deterministically from the numeric scheme;
3. text tokens, sorted, as chosen by the vocabulary integer program.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from tokenization.tst import NumericScheme

SPEC_SCHEMA = "toast_tst_tokenizer/v1"

# Length-limited variant of the GPT-4o pre-tokenization regex, following the
# paper's Appendix B.3: unbounded repeats become bounded so that a long run of
# whitespace or symbols cannot produce one enormous split tree. Digits stay in
# the pattern only as a fallback for spans the numeric scheme refuses; ordinary
# numbers never reach it.
PRETOKEN_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]{0,32}"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]{1,32}(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]{1,32}"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]{0,32}(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]{1,16}[\r\n/]{0,16}"
    r"|\s{0,16}[\r\n]{1,16}"
    r"|\s{1,16}(?!\S)"
    r"|\s{1,16}"
    r"|[\s\S]"
)

END_OF_TEXT = "<|endoftext|>"
DEFAULT_SPECIALS = (
    END_OF_TEXT,
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
)


@dataclass(frozen=True)
class NgramReference:
    """Pointer to the count dictionary that split-tree inference needs.

    Split trees are rebuilt at encode time from these counts, so the counts are
    part of the tokenizer, not a training artefact that can be discarded.
    """

    filename: str
    sha256: str
    entries: int
    min_count: int
    max_length: int

    def to_json(self) -> dict:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "entries": self.entries,
            "min_count": self.min_count,
            "max_length": self.max_length,
        }

    @classmethod
    def from_json(cls, payload: dict) -> NgramReference:
        return cls(
            filename=payload["filename"],
            sha256=payload["sha256"],
            entries=int(payload["entries"]),
            min_count=int(payload["min_count"]),
            max_length=int(payload["max_length"]),
        )


@dataclass(frozen=True)
class TokenizerSpec:
    """Everything needed to reproduce token ids exactly."""

    numeric: NumericScheme
    text_tokens: tuple[bytes, ...]
    ngrams: NgramReference
    specials: tuple[str, ...] = DEFAULT_SPECIALS
    pretoken_pattern: str = PRETOKEN_PATTERN
    schema: str = SPEC_SCHEMA
    provenance: dict = field(default_factory=dict)

    # -- derived layout --------------------------------------------------

    @property
    def numeric_tokens(self) -> tuple[tuple[str, int], ...]:
        return self.numeric.group_vocabulary()

    @property
    def numeric_markers(self) -> tuple[int, ...]:
        return self.numeric.marker_vocabulary()

    @property
    def special_base(self) -> int:
        return 0

    @property
    def numeric_base(self) -> int:
        return len(self.specials)

    @property
    def numeric_count(self) -> int:
        if self.numeric.compound:
            # One token per (digits, power) pair, plus nothing else: the
            # decimal point is implied by the sign of the power.
            return len(self.numeric_tokens)
        # Option A: digit groups and magnitude markers are separate tokens.
        digit_groups = {digits for digits, _ in self.numeric_tokens}
        return len(digit_groups) + len(self.numeric_markers)

    @property
    def text_base(self) -> int:
        return self.numeric_base + self.numeric_count

    @property
    def vocab_size(self) -> int:
        return self.text_base + len(self.text_tokens)

    def eot_id(self) -> int:
        return self.specials.index(END_OF_TEXT)

    # -- serialization ---------------------------------------------------

    def to_json(self) -> dict:
        return {
            "schema": self.schema,
            "pretoken_pattern": self.pretoken_pattern,
            "specials": list(self.specials),
            "numeric": {
                "group_size": self.numeric.group_size,
                "compound": self.numeric.compound,
                "max_int_digits": self.numeric.max_int_digits,
                "max_frac_digits": self.numeric.max_frac_digits,
                "leading_zero_padding": self.numeric.leading_zero_padding,
            },
            "ngrams": self.ngrams.to_json(),
            "layout": {
                "special_base": self.special_base,
                "numeric_base": self.numeric_base,
                "numeric_count": self.numeric_count,
                "text_base": self.text_base,
                "vocab_size": self.vocab_size,
            },
            "text_tokens": [token.hex() for token in self.text_tokens],
            "provenance": self.provenance,
        }

    @classmethod
    def from_json(cls, payload: dict) -> TokenizerSpec:
        if payload["schema"] != SPEC_SCHEMA:
            raise ValueError(
                f"unsupported tokenizer schema {payload['schema']!r}, "
                f"expected {SPEC_SCHEMA!r}"
            )
        numeric = NumericScheme(**payload["numeric"])
        spec = cls(
            numeric=numeric,
            text_tokens=tuple(bytes.fromhex(t) for t in payload["text_tokens"]),
            ngrams=NgramReference.from_json(payload["ngrams"]),
            specials=tuple(payload["specials"]),
            pretoken_pattern=payload["pretoken_pattern"],
            schema=payload["schema"],
            provenance=payload.get("provenance", {}),
        )
        layout = payload["layout"]
        if spec.vocab_size != layout["vocab_size"]:
            raise ValueError(
                f"spec reconstructs {spec.vocab_size} tokens but the recorded "
                f"layout says {layout['vocab_size']}"
            )
        return spec

    def dumps(self) -> str:
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.dumps().encode()).hexdigest()

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: Path) -> TokenizerSpec:
        return cls.from_json(json.loads(Path(path).read_text()))
