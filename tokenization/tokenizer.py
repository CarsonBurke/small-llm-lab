"""The combined encoder and decoder.

Pre-tokenization routes each span to one of three paths:

* a registered special token, matched first so it can never be split;
* a numeric span, mapped by :mod:`tokenization.tst` to magnitude-carrying
  tokens without ever entering the n-gram statistics;
* everything else, mapped by split-tree inference over the text vocabulary.

Keeping numbers off the split-tree path is the point of combining the two
methods. The vocabulary integer program then spends its entire budget on text
instead of on the arbitrary digit fragments that byte-level BPE accumulates,
and place value stops depending on how often a particular digit run happened
to appear in the training corpus.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import regex

from tokenization.ngram_store import file_sha256, read_counts
from tokenization.spec import TokenizerSpec
from tokenization.split_tree import build_split_tree, tree_tokens
from tokenization.tst import NUMERIC_SPAN_RE, NumericScheme

# Token kinds in the reverse table.
_SPECIAL = 0
_TEXT = 1
_GROUP = 2
_DIGITS = 3
_MARKER = 4


class SplitTreeNumericTokenizer:
    """Encode and decode under a fixed :class:`TokenizerSpec`."""

    def __init__(self, spec: TokenizerSpec, counts: Mapping[bytes, int]):
        self.spec = spec
        self.counts = counts
        self.scheme: NumericScheme = spec.numeric
        self._pretoken_re = regex.compile(spec.pretoken_pattern)
        self._special_re = (
            regex.compile("|".join(regex.escape(s) for s in spec.specials))
            if spec.specials
            else None
        )
        self._build_tables()
        self._cache: dict[bytes, tuple[int, ...]] = {}

    # -- construction ----------------------------------------------------

    def _build_tables(self) -> None:
        spec = self.spec
        self._reverse: list[tuple[int, object]] = []
        self._special_ids: dict[str, int] = {}
        self._group_ids: dict[tuple[str, int], int] = {}
        self._digit_ids: dict[str, int] = {}
        self._marker_ids: dict[int, int] = {}
        self._text_ids: dict[bytes, int] = {}

        for special in spec.specials:
            self._special_ids[special] = len(self._reverse)
            self._reverse.append((_SPECIAL, special))

        if self.scheme.compound:
            for digits, power in spec.numeric_tokens:
                self._group_ids[(digits, power)] = len(self._reverse)
                self._reverse.append((_GROUP, (digits, power)))
        else:
            for digits in dict.fromkeys(d for d, _ in spec.numeric_tokens):
                self._digit_ids[digits] = len(self._reverse)
                self._reverse.append((_DIGITS, digits))
            for power in spec.numeric_markers:
                self._marker_ids[power] = len(self._reverse)
                self._reverse.append((_MARKER, power))

        if len(self._reverse) != spec.text_base:
            raise AssertionError(
                f"numeric table ended at {len(self._reverse)}, spec says "
                f"{spec.text_base}"
            )
        for token in spec.text_tokens:
            self._text_ids[token] = len(self._reverse)
            self._reverse.append((_TEXT, token))
        self._text_vocabulary = frozenset(spec.text_tokens)

    @classmethod
    def from_directory(cls, directory) -> SplitTreeNumericTokenizer:
        from pathlib import Path

        directory = Path(directory)
        spec = TokenizerSpec.read(directory / "tokenizer.json")
        counts_path = directory / spec.ngrams.filename
        # Split trees are rebuilt from these counts at encode time, so the
        # counts decide the encoding just as much as the vocabulary does. The
        # spec hash that binds a corpus to its tokenizer covers only
        # tokenizer.json, and would happily accept a directory whose counts
        # had been swapped underneath it.
        digest = file_sha256(counts_path)
        if digest != spec.ngrams.sha256:
            raise ValueError(
                f"{counts_path} hashes to {digest}, but "
                f"{directory / 'tokenizer.json'} declares "
                f"{spec.ngrams.sha256}; the counts decide split-tree "
                "inference, so this directory would not reproduce its own "
                "encoding"
            )
        counts = read_counts(counts_path)
        if len(counts) != spec.ngrams.entries:
            raise ValueError(
                f"{counts_path} holds {len(counts)} distinct n-grams, but "
                f"the spec declares {spec.ngrams.entries}"
            )
        return cls(spec, counts)

    @property
    def vocab_size(self) -> int:
        return self.spec.vocab_size

    @property
    def eot_id(self) -> int:
        return self.spec.eot_id()

    # -- encoding --------------------------------------------------------

    def encode(self, text: str, *, allow_specials: bool = True) -> list[int]:
        ids: list[int] = []
        for is_special, chunk in self._split_specials(text, allow_specials):
            if is_special:
                ids.append(self._special_ids[chunk])
            else:
                self._encode_plain(chunk, ids)
        return ids

    def _split_specials(
        self, text: str, allow_specials: bool
    ) -> Iterable[tuple[bool, str]]:
        if not allow_specials or self._special_re is None:
            yield False, text
            return
        position = 0
        for match in self._special_re.finditer(text):
            if match.start() > position:
                yield False, text[position : match.start()]
            yield True, match.group()
            position = match.end()
        if position < len(text):
            yield False, text[position:]

    def _encode_plain(self, text: str, ids: list[int]) -> None:
        for is_numeric, chunk in self._split_numeric(text):
            if is_numeric:
                self._encode_numeric(chunk, ids)
            else:
                self._encode_text(chunk, ids)

    def _split_numeric(self, text: str) -> list[tuple[bool, str]]:
        """Partition into numeric spans and text, merging rejected spans.

        A span the scheme cannot represent stays on the text path rather than
        being escaped, which keeps encoding total without an escape token.
        """
        pieces: list[tuple[bool, str]] = []
        position = 0
        for match in NUMERIC_SPAN_RE.finditer(text):
            span = match.group()
            integer, fraction = self.scheme.split_span(span)
            if not self.scheme.accepts(integer, fraction):
                continue
            if match.start() > position:
                pieces.append((False, text[position : match.start()]))
            pieces.append((True, span))
            position = match.end()
        if position < len(text):
            pieces.append((False, text[position:]))
        return pieces

    def _encode_numeric(self, span: str, ids: list[int]) -> None:
        groups = self.scheme.groups_for_span(span)
        if self.scheme.compound:
            for group in groups:
                ids.append(self._group_ids[group])
        else:
            for digits, power in groups:
                ids.append(self._digit_ids[digits])
                ids.append(self._marker_ids[power])

    def _encode_text(self, text: str, ids: list[int]) -> None:
        position = 0
        for match in self._pretoken_re.finditer(text):
            if match.start() != position:
                raise ValueError(
                    f"pre-tokenization left a gap at {position} in {text!r}; "
                    "the pattern must tile its input"
                )
            position = match.end()
            ids.extend(self._encode_pretoken(match.group().encode("utf-8")))
        if position != len(text):
            raise ValueError(
                f"pre-tokenization stopped at {position} of {len(text)} in "
                f"{text!r}"
            )

    def _encode_pretoken(self, pretoken: bytes) -> tuple[int, ...]:
        cached = self._cache.get(pretoken)
        if cached is not None:
            return cached
        tree = build_split_tree(pretoken, self.counts)
        pieces = tree_tokens(tree, self._text_vocabulary)
        ids: list[int] = []
        for piece in pieces:
            token_id = self._text_ids.get(piece)
            if token_id is None:
                # Split-tree inference emits single bytes unconditionally, so a
                # miss here means the byte alphabet is incomplete.
                raise KeyError(
                    f"text vocabulary is missing {piece!r}; the byte alphabet "
                    "must be forced into every vocabulary"
                )
            ids.append(token_id)
        result = tuple(ids)
        self._cache[pretoken] = result
        return result

    # -- decoding --------------------------------------------------------

    def decode(self, ids: Iterable[int]) -> str:
        out: list[str] = []
        pending_bytes = bytearray()
        pending_groups: list[tuple[str, int]] = []
        pending_digits: str | None = None

        def flush_bytes() -> None:
            if pending_bytes:
                out.append(pending_bytes.decode("utf-8", errors="replace"))
                pending_bytes.clear()

        def flush_groups() -> None:
            if pending_groups:
                out.append(_render_numeric_run(pending_groups, self.scheme))
                pending_groups.clear()

        for token_id in ids:
            if not 0 <= token_id < len(self._reverse):
                raise ValueError(
                    f"token id {token_id} is outside 0..{len(self._reverse) - 1}"
                )
            kind, payload = self._reverse[token_id]
            if kind == _TEXT:
                flush_groups()
                pending_bytes.extend(payload)
                continue
            if kind == _GROUP:
                flush_bytes()
                pending_groups.append(payload)
                continue
            if kind == _DIGITS:
                flush_bytes()
                if pending_digits is not None:
                    # A digit group with no marker: emit it at power zero so a
                    # malformed generation still decodes to something readable.
                    pending_groups.append((pending_digits, 0))
                pending_digits = payload
                continue
            if kind == _MARKER:
                flush_bytes()
                if pending_digits is None:
                    continue
                pending_groups.append((pending_digits, payload))
                pending_digits = None
                continue
            flush_bytes()
            flush_groups()
            out.append(payload)
        if pending_digits is not None:
            pending_groups.append((pending_digits, 0))
        flush_bytes()
        flush_groups()
        return "".join(out)


def _render_numeric_run(
    groups: list[tuple[str, int]], scheme: NumericScheme
) -> str:
    """Reassemble one or more numbers from a run of numeric tokens.

    Encoder output always forms a single number with strictly decreasing
    powers, and :meth:`NumericScheme.span_for_groups` verifies that exactly.
    Model output need not, so a run is cut wherever the power stops decreasing
    and each piece is reassembled positionally. That keeps generation-time
    decoding total while still inverting the encoder exactly.
    """
    numbers: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    for group in groups:
        if current and group[1] >= current[-1][1]:
            numbers.append(current)
            current = []
        current.append(group)
    if current:
        numbers.append(current)

    rendered: list[str] = []
    for number in numbers:
        try:
            rendered.append(scheme.span_for_groups(number))
        except ValueError:
            integer = "".join(d for d, p in number if p >= 0)
            fraction = "".join(d for d, p in number if p < 0)
            rendered.append(f"{integer}.{fraction}" if fraction else integer)
    return "".join(rendered)


class BatchEncoder:
    """`SplitTreeNumericTokenizer` behind the corpus builder's batch signature.

    The builders were written against `GPT2BatchEncoder`, whose contract is a
    list of texts in and a list of id lists out. Matching that contract here,
    rather than teaching the builders about two tokenizer APIs, is what lets a
    corpus be rebuilt under this tokenizer by changing one flag -- which is the
    only way the tokenizer ablation is a controlled comparison rather than two
    differently-built datasets.
    """

    def __init__(self, tokenizer: SplitTreeNumericTokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_directory(cls, directory) -> BatchEncoder:
        return cls(SplitTreeNumericTokenizer.from_directory(directory))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def eot_id(self) -> int:
        return self.tokenizer.eot_id

    def encode(
        self, texts: list[str], out_type: type = int, num_threads: int | None = None
    ) -> list[list[int]]:
        if out_type is not int:
            raise ValueError("BatchEncoder only encodes to int ids")
        # Corpus text is data, not markup: a document that happens to contain
        # "<|endoftext|>" must encode as those characters rather than as a
        # document boundary the builder never inserted.
        return [
            self.tokenizer.encode(text, allow_specials=False) for text in texts
        ]

    def decode(self, token_batches: list[list[int]]) -> list[str]:
        return [self.tokenizer.decode(tokens) for tokens in token_batches]
