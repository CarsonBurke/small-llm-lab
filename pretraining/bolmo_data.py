"""Lossless source-token to byte-patch data for Bolmo byteification.

Bolmo keeps the pretrained subword Transformer while replacing its external
interface with bytes.  This module supplies the exact alignment needed by the
stitching and distillation stages. Every source token owns a non-empty atomic
span; following the released recipe, the boundary immediately before EOT is
removed so EOT joins the preceding patch.

Ordinary UTF-8 bytes occupy ids ``0..255``.  Source-tokenizer specials are
kept atomic at ``256 + special_index``; in particular, the document-boundary
EOT is id 256.  Padding follows the specials and is never valid data.

TST compound tokens require care because their decimal point is implicit.
With the required N=1 scheme, the first negative-power token in each numeric
run owns ``b"."`` followed by its digit. Numeric-run state is retained only
within one independent model row and reset at every synthetic BOS, as well as
by text or specials. This matches the source tokenizer's total decoder when a
row begins at a fractional token below power -1.

The optional retained source-token embedding is also causal.  At each byte it
names the longest fixed-surface source token ending at that byte.  Only text
tokens and atomic specials are candidates.  TST numeric tokens are semantic
``(digit, magnitude)`` pairs rather than fixed byte strings, so admitting them
would reveal magnitude that is not present in the causal byte prefix.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

from tokenization.tokenizer import SplitTreeNumericTokenizer


BOLMO_DATA_SCHEMA = "bolmo_byte_dataset/v3"
BYTE_VOCAB_SIZE = 256


@dataclass(frozen=True)
class ByteifiedTokens:
    """Atomic symbols and their exact source-token patch lengths."""

    atomic_ids: tuple[int, ...]
    patch_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(length <= 0 for length in self.patch_lengths):
            raise ValueError("every source token must own a non-empty patch")
        if sum(self.patch_lengths) != len(self.atomic_ids):
            raise ValueError(
                "patch lengths do not cover the atomic sequence: "
                f"{sum(self.patch_lengths)} != {len(self.atomic_ids)}"
            )

    @property
    def boundary_mask(self) -> tuple[bool, ...]:
        mask = [False] * len(self.atomic_ids)
        position = 0
        for length in self.patch_lengths:
            position += length
            mask[position - 1] = True
        return tuple(mask)


def truncate_to_complete_patches(
    source_ids: Sequence[int],
    byteified: ByteifiedTokens,
    *,
    max_atomic_tokens: int,
) -> tuple[Sequence[int], ByteifiedTokens]:
    """Truncate to the longest complete source-patch prefix.

    ``max_atomic_tokens`` includes the synthetic leading EOT/BOS atom added by
    :func:`make_bolmo_example`. Bolmo's Table-8 byte limit must never cut a
    source patch because Stage-1 teacher alignment is defined patchwise.
    """

    if max_atomic_tokens <= 1:
        raise ValueError("max_atomic_tokens must leave room after BOS")
    if len(source_ids) != len(byteified.patch_lengths):
        raise ValueError("source ids and byte patches disagree in length")
    byte_budget = max_atomic_tokens - 1
    used = 0
    keep = 0
    for patch_length in byteified.patch_lengths:
        if used + patch_length > byte_budget:
            break
        used += patch_length
        keep += 1
    if keep == 0 and source_ids:
        raise ValueError(
            "one source patch exceeds the configured maximum byte length"
        )
    return (
        source_ids[:keep],
        ByteifiedTokens(
            atomic_ids=byteified.atomic_ids[:used],
            patch_lengths=byteified.patch_lengths[:keep],
        ),
    )


def eos_aware_boundary_mask(
    atomic_ids: Sequence[int],
    patch_lengths: Sequence[int],
    *,
    eot_id: int,
) -> tuple[bool, ...]:
    """Return source-token ends with the boundary immediately before EOT removed.

    This is ``skip_boundary_before_eos=True`` in AI2's released Bolmo recipe.
    Position zero is the synthetic BOS/EOT and is always retained as a patch.
    The original ``patch_lengths`` remain unchanged for source-token
    distillation; only the byte/global routing boundaries are coalesced.
    """

    boundary = list(ByteifiedTokens(tuple(atomic_ids), tuple(patch_lengths)).boundary_mask)
    for position in range(len(atomic_ids) - 1):
        if atomic_ids[position + 1] == eot_id:
            boundary[position] = False
    if boundary:
        boundary[0] = True
    return tuple(boundary)


@dataclass(frozen=True)
class BolmoExample:
    """One unpadded example before bounded shard-level collation."""

    source_ids: tuple[int, ...]
    source_valid_mask: tuple[bool, ...]
    byte_ids: tuple[int, ...]
    expanded_ids: tuple[int, ...]
    boundary_mask: tuple[bool, ...]
    valid_mask: tuple[bool, ...]
    score_mask: tuple[bool, ...]
    patch_lens: tuple[int, ...]

    def __post_init__(self) -> None:
        source_length = len(self.source_ids)
        if not (
            len(self.source_valid_mask) == source_length
            and len(self.patch_lens) == source_length
        ):
            raise ValueError("source fields must have identical lengths")
        byte_length = len(self.byte_ids)
        if not (
            len(self.expanded_ids) == byte_length
            and len(self.boundary_mask) == byte_length
            and len(self.valid_mask) == byte_length
            and len(self.score_mask) == byte_length
        ):
            raise ValueError("byte fields must have identical lengths")
        if any(score and not valid for score, valid in zip(
            self.score_mask, self.valid_mask, strict=True
        )):
            raise ValueError("score mask cannot include invalid byte positions")
        valid_patch_atoms = sum(
            length
            for length, valid in zip(
                self.patch_lens, self.source_valid_mask, strict=True
            )
            if valid
        )
        if valid_patch_atoms != sum(self.valid_mask):
            raise ValueError("valid patch lengths do not cover valid byte ids")


def validate_source_tokenizer(tokenizer: SplitTreeNumericTokenizer) -> None:
    """Require the exact ToaST + compound TST N=1 source representation."""

    scheme = tokenizer.scheme
    if scheme.group_size != 1 or not scheme.compound:
        raise ValueError(
            "Bolmo byteification requires compound TST with group_size=1; "
            f"got group_size={scheme.group_size}, compound={scheme.compound}"
        )
    if scheme.leading_zero_padding:
        raise ValueError(
            "lossy TST leading-zero padding cannot be byteified exactly"
        )
    if tokenizer.eot_id != 0:
        raise ValueError(
            "the Bolmo dataset schema requires source EOT id 0 so its atomic "
            f"id is 256, got source EOT id {tokenizer.eot_id}"
        )


def atomic_special_id(source_special_id: int) -> int:
    return BYTE_VOCAB_SIZE + source_special_id


def atomic_pad_id(tokenizer: SplitTreeNumericTokenizer) -> int:
    return BYTE_VOCAB_SIZE + len(tokenizer.spec.specials)


class SourceTokenByteifier:
    """Stateful, lossless conversion of source ids into atomic byte patches.

    Instances retain numeric-run rendering state across :meth:`byteify` calls
    until :meth:`reset`. Dataset builders reset at every synthetic-BOS model
    row so semantic TST tokens render exactly as that independent source row
    decodes; callers may retain state only when deliberately splitting one
    logical row into streaming chunks.
    """

    def __init__(self, tokenizer: SplitTreeNumericTokenizer):
        validate_source_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        spec = tokenizer.spec
        self._special_count = len(spec.specials)
        self._numeric_base = spec.numeric_base
        self._text_base = spec.text_base
        self._vocab_size = spec.vocab_size
        self._numeric_payloads = spec.numeric_tokens
        self._text_surfaces = spec.text_tokens
        self._previous_numeric_power: int | None = None

    def reset(self) -> None:
        self._previous_numeric_power = None

    def byteify(self, source_ids: Iterable[int]) -> ByteifiedTokens:
        atoms: list[int] = []
        lengths: list[int] = []
        for source_id_value in source_ids:
            source_id = int(source_id_value)
            patch = self._patch_for_token(source_id)
            atoms.extend(patch)
            lengths.append(len(patch))
        return ByteifiedTokens(tuple(atoms), tuple(lengths))

    def _patch_for_token(self, source_id: int) -> tuple[int, ...]:
        if not 0 <= source_id < self._vocab_size:
            raise ValueError(
                f"source token id {source_id} is outside 0..{self._vocab_size - 1}"
            )
        if source_id < self._special_count:
            self._previous_numeric_power = None
            return (atomic_special_id(source_id),)
        if source_id < self._text_base:
            digits, power = self._numeric_payloads[source_id - self._numeric_base]
            if len(digits) != 1:
                raise AssertionError("validated TST N=1 emitted a non-unit group")
            previous = self._previous_numeric_power
            # The tokenizer cuts a generated/malformed numeric run whenever
            # power stops decreasing. Its fallback surface is still total:
            # concatenate integer/fraction digits and insert one decimal point
            # before the first negative-power token in each run.
            starts_new_run = previous is not None and power >= previous
            run_previous = None if starts_new_run else previous
            self._previous_numeric_power = power
            prefix = (
                (ord("."),)
                if power < 0 and (run_previous is None or run_previous >= 0)
                else ()
            )
            return (*prefix, ord(digits))

        self._previous_numeric_power = None
        surface = self._text_surfaces[source_id - self._text_base]
        if not surface:
            raise ValueError(f"source text token {source_id} has an empty surface")
        return tuple(surface)


class ExpandedSuffixMatcher:
    """Longest fixed source-token surface ending at each causal byte."""

    def __init__(
        self,
        tokenizer: SplitTreeNumericTokenizer,
        *,
        source_model_vocab_size: int | None = None,
    ):
        validate_source_tokenizer(tokenizer)
        self.tokenizer = tokenizer
        self.pad_id = (
            tokenizer.vocab_size
            if source_model_vocab_size is None
            else source_model_vocab_size
        )
        if self.pad_id < tokenizer.vocab_size:
            raise ValueError(
                "source model vocabulary cannot be smaller than the logical "
                f"tokenizer vocabulary ({self.pad_id} < {tokenizer.vocab_size})"
            )
        # Aho-Corasick failure links make matching linear in emitted bytes.
        # This matters for the full Arm-C build: rescanning a reversed history
        # at every byte would multiply hundreds of millions of atoms by the
        # maximum source-token surface length.
        transitions: list[dict[int, int]] = [{}]
        failure = [0]
        own_token: list[int | None] = [None]
        surface_length = [0]
        for offset, surface in enumerate(tokenizer.spec.text_tokens):
            if not surface:
                continue
            token_id = tokenizer.spec.text_base + offset
            state = 0
            for byte in surface:
                child = transitions[state].get(byte)
                if child is None:
                    child = len(transitions)
                    transitions[state][byte] = child
                    transitions.append({})
                    failure.append(0)
                    own_token.append(None)
                    surface_length.append(surface_length[state] + 1)
                state = child
            if own_token[state] is not None:
                raise ValueError(f"duplicate fixed token surface {surface!r}")
            own_token[state] = token_id

        best_token = own_token.copy()
        best_length = [
            length if token is not None else 0
            for token, length in zip(own_token, surface_length, strict=True)
        ]
        queue: deque[int] = deque(transitions[0].values())
        while queue:
            state = queue.popleft()
            fallback = failure[state]
            if best_length[fallback] > best_length[state]:
                best_token[state] = best_token[fallback]
                best_length[state] = best_length[fallback]
            for byte, child in transitions[state].items():
                fallback = failure[state]
                while fallback and byte not in transitions[fallback]:
                    fallback = failure[fallback]
                failure[child] = transitions[fallback].get(byte, 0)
                queue.append(child)

        self._transitions = transitions
        self._failure = failure
        self._best_token = best_token

    def match(self, atomic_ids: Sequence[int]) -> tuple[int, ...]:
        result: list[int] = []
        state = 0
        special_count = len(self.tokenizer.spec.specials)
        for atom_value in atomic_ids:
            atom = int(atom_value)
            if 0 <= atom < BYTE_VOCAB_SIZE:
                while state and atom not in self._transitions[state]:
                    state = self._failure[state]
                state = self._transitions[state].get(atom, 0)
                token_id = self._best_token[state]
                result.append(self.pad_id if token_id is None else token_id)
                continue
            special_index = atom - BYTE_VOCAB_SIZE
            if not 0 <= special_index < special_count:
                raise ValueError(f"invalid atomic id {atom} in an unpadded sequence")
            state = 0
            result.append(special_index)
        return tuple(result)


def make_bolmo_example(
    real_source_ids: Sequence[int],
    byteified: ByteifiedTokens,
    tokenizer: SplitTreeNumericTokenizer,
    suffix_matcher: ExpandedSuffixMatcher,
    *,
    expected_real_source_tokens: int,
    source_model_vocab_size: int | None = None,
    context_source_tokens: int = 0,
) -> BolmoExample:
    """Prepend an atomic EOT/BOS patch and pad only the source-token axis."""

    if len(real_source_ids) > expected_real_source_tokens:
        raise ValueError("example exceeds its fixed source-token width")
    if len(byteified.patch_lengths) != len(real_source_ids):
        raise ValueError("source ids and byte patches disagree in length")
    if not 0 <= context_source_tokens <= len(real_source_ids):
        raise ValueError("context_source_tokens is outside the real source prefix")
    source_pad = (
        tokenizer.vocab_size
        if source_model_vocab_size is None
        else source_model_vocab_size
    )
    if source_pad < tokenizer.vocab_size:
        raise ValueError("source model vocabulary is smaller than the tokenizer")
    if suffix_matcher.pad_id != source_pad:
        raise ValueError("suffix matcher and example use different source pad ids")
    missing = expected_real_source_tokens - len(real_source_ids)
    source_ids = (
        tokenizer.eot_id,
        *(int(value) for value in real_source_ids),
        *((source_pad,) * missing),
    )
    source_valid = (True,) * (1 + len(real_source_ids)) + (False,) * missing
    patch_lens = (1, *byteified.patch_lengths, *((0,) * missing))
    atoms = (atomic_special_id(tokenizer.eot_id), *byteified.atomic_ids)
    valid_patch_lens = tuple(patch_lens[: 1 + len(real_source_ids)])
    boundary = eos_aware_boundary_mask(
        atoms,
        valid_patch_lens,
        eot_id=atomic_special_id(tokenizer.eot_id),
    )
    expanded = suffix_matcher.match(atoms)
    context_atoms = 1 + sum(byteified.patch_lengths[:context_source_tokens])
    return BolmoExample(
        source_ids=tuple(source_ids),
        source_valid_mask=tuple(source_valid),
        byte_ids=tuple(atoms),
        expanded_ids=expanded,
        boundary_mask=boundary,
        valid_mask=(True,) * len(atoms),
        score_mask=(False,) * context_atoms
        + (True,) * (len(atoms) - context_atoms),
        patch_lens=tuple(patch_lens),
    )


def collate_examples(
    examples: Sequence[BolmoExample],
    tokenizer: SplitTreeNumericTokenizer,
    *,
    source_model_vocab_size: int | None = None,
    byte_length_multiple: int = 128,
) -> dict[str, torch.Tensor]:
    """Pad a bounded example shard to its local maximum byte length."""

    if not examples:
        raise ValueError("cannot collate an empty Bolmo shard")
    source_width = len(examples[0].source_ids)
    if any(len(example.source_ids) != source_width for example in examples):
        raise ValueError("all examples in a shard need one source-token width")
    if byte_length_multiple <= 0:
        raise ValueError("byte_length_multiple must be positive")
    unpadded_byte_width = max(len(example.byte_ids) for example in examples)
    byte_width = (
        (unpadded_byte_width + byte_length_multiple - 1)
        // byte_length_multiple
        * byte_length_multiple
    )
    count = len(examples)
    atom_pad = atomic_pad_id(tokenizer)
    source_pad = (
        tokenizer.vocab_size
        if source_model_vocab_size is None
        else source_model_vocab_size
    )
    if source_pad < tokenizer.vocab_size:
        raise ValueError("source model vocabulary is smaller than the tokenizer")

    source_ids = torch.full((count, source_width), source_pad, dtype=torch.int32)
    source_valid = torch.zeros((count, source_width), dtype=torch.bool)
    byte_ids = torch.full((count, byte_width), atom_pad, dtype=torch.int16)
    expanded_ids = torch.full((count, byte_width), source_pad, dtype=torch.int32)
    boundary = torch.zeros((count, byte_width), dtype=torch.bool)
    valid = torch.zeros((count, byte_width), dtype=torch.bool)
    score = torch.zeros((count, byte_width), dtype=torch.bool)
    patch_lens = torch.zeros((count, source_width), dtype=torch.int32)

    for row, example in enumerate(examples):
        source_ids[row] = torch.tensor(example.source_ids, dtype=torch.int32)
        source_valid[row] = torch.tensor(
            example.source_valid_mask, dtype=torch.bool
        )
        patch_lens[row] = torch.tensor(example.patch_lens, dtype=torch.int32)
        length = len(example.byte_ids)
        byte_ids[row, :length] = torch.tensor(example.byte_ids, dtype=torch.int16)
        expanded_ids[row, :length] = torch.tensor(
            example.expanded_ids, dtype=torch.int32
        )
        boundary[row, :length] = torch.tensor(
            example.boundary_mask, dtype=torch.bool
        )
        valid[row, :length] = torch.tensor(example.valid_mask, dtype=torch.bool)
        score[row, :length] = torch.tensor(example.score_mask, dtype=torch.bool)

    return {
        "source_ids": source_ids,
        "source_valid_mask": source_valid,
        "byte_ids": byte_ids,
        "expanded_ids": expanded_ids,
        "boundary_mask": boundary,
        "valid_mask": valid,
        "score_mask": score,
        "patch_lens": patch_lens,
    }


def save_example_shard(
    path: Path,
    examples: Sequence[BolmoExample],
    tokenizer: SplitTreeNumericTokenizer,
    *,
    source_model_vocab_size: int | None = None,
    byte_length_multiple: int = 128,
) -> dict[str, object]:
    """Atomically save one bounded CPU tensor shard and describe its schema."""

    payload = collate_examples(
        examples,
        tokenizer,
        source_model_vocab_size=source_model_vocab_size,
        byte_length_multiple=byte_length_multiple,
    )
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".working")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "examples": len(examples),
        "source_width": int(payload["source_ids"].shape[1]),
        "byte_width": int(payload["byte_ids"].shape[1]),
        "valid_source_tokens": int(payload["source_valid_mask"].sum()),
        "valid_atomic_tokens": int(payload["valid_mask"].sum()),
        "dtypes": {name: str(tensor.dtype) for name, tensor in payload.items()},
    }


def load_example_shard(
    path: Path,
    *,
    source_vocab_size: int | None = None,
    source_pad_id: int | None = None,
    atomic_vocab_size: int | None = None,
    context_source_tokens: int | None = None,
) -> dict[str, torch.Tensor]:
    """Load and structurally validate a saved tensor shard."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {
        "source_ids",
        "source_valid_mask",
        "byte_ids",
        "expanded_ids",
        "boundary_mask",
        "valid_mask",
        "score_mask",
        "patch_lens",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        observed = set(payload) if isinstance(payload, dict) else type(payload)
        raise ValueError(f"invalid Bolmo shard fields: {observed}")
    expected_dtypes = {
        "source_ids": torch.int32,
        "source_valid_mask": torch.bool,
        "byte_ids": torch.int16,
        "expanded_ids": torch.int32,
        "boundary_mask": torch.bool,
        "valid_mask": torch.bool,
        "score_mask": torch.bool,
        "patch_lens": torch.int32,
    }
    for name, dtype in expected_dtypes.items():
        tensor = payload[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Bolmo shard field {name} is not a tensor")
        if tensor.ndim != 2:
            raise ValueError(f"Bolmo shard field {name} must have rank 2")
        if tensor.dtype != dtype:
            raise ValueError(
                f"Bolmo shard field {name} has dtype {tensor.dtype}, "
                f"expected {dtype}"
            )
    if payload["source_ids"].shape != payload["source_valid_mask"].shape:
        raise ValueError("source ids and validity mask shapes differ")
    if payload["source_ids"].shape != payload["patch_lens"].shape:
        raise ValueError("source ids and patch lengths shapes differ")
    byte_shape = payload["byte_ids"].shape
    for name in ("expanded_ids", "boundary_mask", "valid_mask", "score_mask"):
        if payload[name].shape != byte_shape:
            raise ValueError(f"byte field {name} has the wrong shape")
    if payload["source_ids"].shape[0] != byte_shape[0]:
        raise ValueError("source and byte fields have different batch sizes")

    source_valid = payload["source_valid_mask"]
    byte_valid = payload["valid_mask"]
    if source_valid.shape[1] and not bool(source_valid[:, 0].all()):
        raise ValueError("every example must begin with a valid EOT/BOS patch")
    if byte_valid.shape[1] and not bool(byte_valid[:, 0].all()):
        raise ValueError("every example must begin with a valid EOT/BOS atom")
    if source_valid.shape[1] and not bool((payload["source_ids"][:, 0] == 0).all()):
        raise ValueError("every source example must begin with synthetic EOT/BOS id 0")
    if byte_valid.shape[1] and not bool(
        (payload["byte_ids"][:, 0] == BYTE_VOCAB_SIZE).all()
    ):
        raise ValueError("every atomic example must begin with synthetic EOT/BOS id 256")
    if byte_valid.shape[1] and bool(payload["score_mask"][:, 0].any()):
        raise ValueError("synthetic EOT/BOS cannot be a scored target")
    if bool((payload["score_mask"] & ~byte_valid).any()):
        raise ValueError("score mask includes padded byte positions")
    for name, mask in (
        ("source_valid_mask", source_valid),
        ("valid_mask", byte_valid),
    ):
        if mask.shape[1] > 1 and bool(((~mask[:, :-1]) & mask[:, 1:]).any()):
            raise ValueError(f"{name} must be a contiguous valid prefix")

    patch_lens = payload["patch_lens"]
    if bool((patch_lens < 0).any()):
        raise ValueError("patch lengths cannot be negative")
    if bool((patch_lens[~source_valid] != 0).any()):
        raise ValueError("source-padding positions must have zero patch length")
    if bool((patch_lens[source_valid] <= 0).any()):
        raise ValueError("valid source positions must have non-empty patches")
    if not torch.equal(patch_lens.sum(dim=1), byte_valid.sum(dim=1)):
        raise ValueError("patch lengths do not cover the valid byte prefix")
    for row in range(patch_lens.shape[0]):
        scored_source_token_count(
            source_valid[row],
            patch_lens[row],
            byte_valid[row],
            payload["score_mask"][row],
            context_source_tokens=context_source_tokens,
        )
    if bool((payload["boundary_mask"] & ~byte_valid).any()):
        raise ValueError("a padded byte position is marked as a patch boundary")
    if source_vocab_size is not None:
        if source_pad_id is None or source_pad_id < source_vocab_size:
            raise ValueError(
                "source_pad_id must accompany and cover source_vocab_size"
            )
        source_ids = payload["source_ids"]
        expanded_ids = payload["expanded_ids"]
        if bool(
            ((source_ids[source_valid] < 0) | (source_ids[source_valid] >= source_vocab_size)).any()
        ):
            raise ValueError("a valid source id is outside the source vocabulary")
        if bool((source_ids[~source_valid] != source_pad_id).any()):
            raise ValueError("a padded source id differs from the source pad id")
        valid_expanded = expanded_ids[byte_valid]
        if bool(
            (
                (valid_expanded < 0)
                | ((valid_expanded >= source_vocab_size) & (valid_expanded != source_pad_id))
            ).any()
        ):
            raise ValueError("a valid expanded id is neither a source id nor null")
    if atomic_vocab_size is not None:
        byte_ids = payload["byte_ids"]
        if bool(
            ((byte_ids[byte_valid] < 0) | (byte_ids[byte_valid] >= atomic_vocab_size)).any()
        ):
            raise ValueError("a valid atomic id is outside the atomic vocabulary")
        if bool((byte_ids[~byte_valid] != atomic_vocab_size).any()):
            raise ValueError("a padded atomic id differs from the atomic pad id")

    expected_boundary = torch.zeros_like(payload["boundary_mask"])
    for row in range(patch_lens.shape[0]):
        valid_lengths = patch_lens[row][source_valid[row]]
        ends = valid_lengths.cumsum(dim=0) - 1
        expected_boundary[row, ends.to(torch.int64)] = True
        valid_count = int(byte_valid[row].sum())
        atoms = payload["byte_ids"][row, :valid_count]
        before_eot = torch.nonzero(
            atoms[1:] == BYTE_VOCAB_SIZE, as_tuple=False
        ).flatten()
        expected_boundary[row, before_eot] = False
        expected_boundary[row, 0] = True
    if not torch.equal(payload["boundary_mask"], expected_boundary):
        raise ValueError("boundary mask does not match EOS-aware source patching")
    return payload


def scored_source_token_count(
    source_valid_mask: torch.Tensor,
    patch_lens: torch.Tensor,
    valid_mask: torch.Tensor,
    score_mask: torch.Tensor,
    *,
    context_source_tokens: int | None = None,
) -> int:
    """Validate a byte score suffix and return its complete source patches.

    The synthetic BOS and any explicit left-context source tokens must be an
    unscored prefix of complete patches. Every later valid atom is scored.
    This makes validation accounting fail closed instead of inferring the
    target span from a hard-coded source-axis offset.
    """

    tensors = (source_valid_mask, patch_lens, valid_mask, score_mask)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("score accounting expects one-dimensional rows")
    if source_valid_mask.shape != patch_lens.shape:
        raise ValueError("source validity and patch lengths differ in shape")
    if valid_mask.shape != score_mask.shape:
        raise ValueError("byte validity and score masks differ in shape")
    valid_atoms = int(valid_mask.sum())
    valid_patches = int(source_valid_mask.sum())
    if valid_atoms <= 1 or valid_patches <= 1:
        raise ValueError("Bolmo row has no scoreable source target")
    score = score_mask[:valid_atoms]
    scored_positions = torch.nonzero(score, as_tuple=False).flatten()
    if not len(scored_positions):
        raise ValueError("Bolmo row has no scored byte targets")
    score_start = int(scored_positions[0])
    if bool(score[:score_start].any()) or not bool(score[score_start:].all()):
        raise ValueError("score mask must be one contiguous suffix of valid atoms")
    valid_lengths = patch_lens[:valid_patches]
    patch_ends = valid_lengths.cumsum(0)
    matching_boundaries = torch.nonzero(
        patch_ends == score_start, as_tuple=False
    ).flatten()
    if len(matching_boundaries) != 1:
        raise ValueError("score mask must start at a complete source-patch boundary")
    unscored_patches = int(matching_boundaries[0]) + 1
    if context_source_tokens is not None:
        expected_unscored = 1 + context_source_tokens
        if unscored_patches != expected_unscored:
            raise ValueError(
                "score mask context differs from its split contract: "
                f"{unscored_patches - 1} != {context_source_tokens}"
            )
    scored_patches = valid_patches - unscored_patches
    if scored_patches <= 0:
        raise ValueError("Bolmo row has no scored source-token patches")
    return scored_patches


def decode_atomic_ids(
    atomic_ids: Iterable[int], tokenizer: SplitTreeNumericTokenizer
) -> str:
    """Render an unpadded atomic stream for round-trip checks and diagnostics."""

    pieces: list[str] = []
    pending = bytearray()

    def flush() -> None:
        if pending:
            pieces.append(pending.decode("utf-8"))
            pending.clear()

    for atom_value in atomic_ids:
        atom = int(atom_value)
        if 0 <= atom < BYTE_VOCAB_SIZE:
            pending.append(atom)
            continue
        special_index = atom - BYTE_VOCAB_SIZE
        if not 0 <= special_index < len(tokenizer.spec.specials):
            raise ValueError(f"cannot decode padding or invalid atomic id {atom}")
        flush()
        pieces.append(tokenizer.spec.specials[special_index])
    flush()
    return "".join(pieces)
