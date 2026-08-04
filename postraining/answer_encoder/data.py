"""Corpora and stochastic text views for LeJEPA answer training."""

from __future__ import annotations

import glob
import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor

from postraining.answer_encoder.model import GPT2_EOT_ID, GPT2_MASK_ID, GPT2_PAD_ID


SHARD_MAGIC = 20240520
SHARD_VERSION = 1
SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize


@dataclass(frozen=True)
class ViewConfig:
    global_views: int = 2
    local_views: int = 6
    global_scale_min: float = 0.3
    global_scale_max: float = 1.0
    local_scale_min: float = 0.05
    local_scale_max: float = 0.3
    mask_probability: float = 0.1
    block_shuffle_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.global_views < 1 or self.local_views < 0:
            raise ValueError("view counts require at least one global view")
        if self.global_views + self.local_views < 2:
            raise ValueError("LeJEPA training requires at least two views")
        for low, high, name in (
            (self.global_scale_min, self.global_scale_max, "global"),
            (self.local_scale_min, self.local_scale_max, "local"),
        ):
            if not 0.0 < low <= high <= 1.0:
                raise ValueError(f"{name} view scales must satisfy 0 < min <= max <= 1")
        if not 0.0 <= self.mask_probability <= 1.0:
            raise ValueError("mask_probability must be in [0, 1]")
        if not 0.0 <= self.block_shuffle_probability <= 1.0:
            raise ValueError("block_shuffle_probability must be in [0, 1]")

    @property
    def count(self) -> int:
        return self.global_views + self.local_views


@dataclass(frozen=True)
class SpanMaskConfig:
    """Contiguous token spans deleted from the compact training context."""

    probability: float = 0.3
    mean_span_length: float = 3.0

    def __post_init__(self) -> None:
        if not 0.0 < self.probability <= 1.0:
            raise ValueError("span mask probability must be in (0, 1]")
        if self.mean_span_length <= 0.0:
            raise ValueError("mean span length must be positive")


@dataclass(frozen=True)
class TokenExample:
    tokens: Tensor | tuple[int, ...]
    blocks: tuple[Tensor | tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        tokens = torch.as_tensor(self.tokens, dtype=torch.uint16).clone()
        blocks = tuple(
            torch.as_tensor(block, dtype=torch.uint16).clone()
            for block in self.blocks
        )
        if tokens.numel() == 0:
            raise ValueError("text examples must contain at least one token")
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "blocks", blocks)


class TokenShard:
    """Memory-mapped modded-nanogpt token shard."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        header = np.fromfile(self.path, dtype="<i4", count=256)
        if header.size != 256 or int(header[0]) != SHARD_MAGIC:
            raise ValueError(f"invalid token shard magic in {self.path}")
        if int(header[1]) != SHARD_VERSION:
            raise ValueError(f"unsupported token shard version in {self.path}")
        self.num_tokens = int(header[2])
        expected_bytes = SHARD_HEADER_BYTES + 2 * self.num_tokens
        if self.path.stat().st_size != expected_bytes:
            raise ValueError(
                f"token shard {self.path} has {self.path.stat().st_size} bytes, expected {expected_bytes}"
            )
        self.tokens = np.memmap(
            self.path,
            dtype="<u2",
            mode="r",
            offset=SHARD_HEADER_BYTES,
            shape=(self.num_tokens,),
        )

    def sample(self, rng: random.Random, min_tokens: int, max_tokens: int) -> TokenExample:
        if self.num_tokens < min_tokens:
            raise ValueError(
                f"token shard {self.path} has {self.num_tokens} tokens, fewer than requested {min_tokens}"
            )
        upper = min(max_tokens, self.num_tokens)
        for _ in range(16):
            length = rng.randint(min_tokens, upper)
            start = rng.randrange(self.num_tokens - length + 1)
            tokens = np.asarray(self.tokens[start : start + length])
            if GPT2_EOT_ID not in tokens:
                return TokenExample(tokens=torch.from_numpy(tokens.astype(np.int32)))
        # Fall back to the longest document-bounded fragment in the sampled
        # window rather than silently joining two unrelated documents.
        boundaries = np.flatnonzero(tokens == GPT2_EOT_ID)
        starts = np.concatenate(([0], boundaries + 1))
        stops = np.concatenate((boundaries, [len(tokens)]))
        fragments = [tokens[start:stop] for start, stop in zip(starts, stops, strict=True)]
        fragment = max(fragments, key=len)
        if len(fragment) < min_tokens:
            raise RuntimeError(f"could not sample {min_tokens} document-bounded tokens from {self.path}")
        return TokenExample(tokens=torch.from_numpy(fragment[:upper].astype(np.int32)))


class TokenShardCorpus:
    def __init__(self, pattern: str):
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no token shards matched {pattern!r}")
        self.shards = tuple(TokenShard(path) for path in paths)
        self.weights = tuple(shard.num_tokens for shard in self.shards)

    def sample(self, rng: random.Random, min_tokens: int, max_tokens: int) -> TokenExample:
        shard = rng.choices(self.shards, weights=self.weights, k=1)[0]
        return shard.sample(rng, min_tokens, max_tokens)


def parse_parquet_source(specification: str) -> tuple[Path, str]:
    try:
        path, column = specification.rsplit(":", 1)
    except ValueError as error:
        raise ValueError("parquet sources must use PATH:COLUMN syntax") from error
    if not path or not column:
        raise ValueError("parquet sources must use nonempty PATH:COLUMN values")
    return Path(path), column


def _nested_value(row: object, path: Iterable[str]) -> object:
    value = row
    for part in path:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def load_parquet_texts(
    specification: str,
    *,
    max_texts: int | None = None,
    deduplicate: bool = False,
    max_characters: int | None = None,
) -> list[str]:
    """Stream one string column without materializing a large parquet table."""
    path, column = parse_parquet_source(specification)
    parts = column.split(".")
    parquet = pq.ParquetFile(path)
    texts: list[str] = []
    if max_characters is not None and max_characters < 1:
        raise ValueError("max_characters must be positive")
    seen: set[bytes] = set()
    for batch in parquet.iter_batches(columns=[parts[0]], batch_size=8192):
        for row in batch.column(0).to_pylist():
            value = _nested_value(row, parts[1:]) if len(parts) > 1 else row
            if not isinstance(value, str):
                continue
            text = value.strip()
            digest = hashlib.blake2b(text.encode(), digest_size=16).digest()
            if not text or (deduplicate and digest in seen):
                continue
            if deduplicate:
                seen.add(digest)
            texts.append(text if max_characters is None else text[:max_characters])
            if max_texts is not None and len(texts) >= max_texts:
                return texts
    return texts


_BLOCK_SEPARATOR = re.compile(r"\n\s*\n+")


def tokenize_texts(
    texts: Iterable[str],
    encode: Callable[[str], list[int]],
    *,
    block_separator_tokens: tuple[int, ...],
    max_source_tokens: int,
    preserve_blocks: bool,
) -> list[TokenExample]:
    examples = []
    for text in texts:
        token_ids = tuple(encode(text)[:max_source_tokens])
        if not token_ids:
            continue
        blocks: tuple[tuple[int, ...], ...] = ()
        if preserve_blocks:
            remaining = max_source_tokens
            collected = []
            for block in _BLOCK_SEPARATOR.split(text):
                if not block.strip() or remaining <= 0:
                    continue
                block_tokens = tuple(encode(block)[:remaining])
                if block_tokens:
                    collected.append(block_tokens)
                    remaining -= len(block_tokens) + len(block_separator_tokens)
            blocks = tuple(collected)
        if len(blocks) < 2:
            blocks = ()
        examples.append(TokenExample(tokens=token_ids, blocks=blocks))
    return examples


class MixedCorpus:
    """Mixture of broad token shards and explicitly supplied answer text."""

    def __init__(
        self,
        token_corpus: TokenShardCorpus,
        answer_examples: list[TokenExample],
        answer_probability: float,
    ):
        if not 0.0 <= answer_probability <= 1.0:
            raise ValueError("answer_probability must be in [0, 1]")
        if answer_probability > 0.0 and not answer_examples:
            raise ValueError("answer_probability is positive but no answer text was loaded")
        self.token_corpus = token_corpus
        self.answer_examples = tuple(answer_examples)
        self.answer_probability = answer_probability
        self._answer_order = list(range(len(self.answer_examples)))
        self._answer_cursor = len(self._answer_order)
        self.answer_draws = 0

    def _next_answer(self, rng: random.Random) -> TokenExample:
        if self._answer_cursor >= len(self._answer_order):
            rng.shuffle(self._answer_order)
            self._answer_cursor = 0
        index = self._answer_order[self._answer_cursor]
        self._answer_cursor += 1
        self.answer_draws += 1
        return self.answer_examples[index]

    def sample(self, rng: random.Random, min_tokens: int, max_tokens: int) -> TokenExample:
        if self.answer_examples and rng.random() < self.answer_probability:
            return self._next_answer(rng)
        return self.token_corpus.sample(rng, min_tokens, max_tokens)


def _join_blocks(
    blocks: tuple[Tensor, ...],
    separator_tokens: tuple[int, ...],
    rng: random.Random,
) -> list[int]:
    order = list(range(len(blocks)))
    rng.shuffle(order)
    joined: list[int] = []
    for index in order:
        if joined:
            joined.extend(separator_tokens)
        joined.extend(map(int, blocks[index].tolist()))
    return joined


def make_view(
    example: TokenExample,
    *,
    scale: tuple[float, float],
    max_view_tokens: int,
    mask_probability: float,
    block_shuffle_probability: float,
    block_separator_tokens: tuple[int, ...],
    mask_token_id: int,
    rng: random.Random,
) -> list[int]:
    tokens = list(map(int, example.tokens.tolist()))
    if (
        example.blocks
        and rng.random() < block_shuffle_probability
    ):
        tokens = _join_blocks(example.blocks, block_separator_tokens, rng)
    fraction = rng.uniform(*scale)
    crop_length = min(max_view_tokens, max(1, round(len(tokens) * fraction)))
    start = rng.randrange(len(tokens) - crop_length + 1)
    cropped = tokens[start : start + crop_length]
    if mask_probability:
        original = cropped
        cropped = [
            mask_token_id if rng.random() < mask_probability else token
            for token in cropped
        ]
        # A fully masked short answer is not a semantics-preserving view:
        # the same [MASK] observation cannot identify both "9" and "10".
        if all(token == mask_token_id for token in cropped):
            restore = rng.randrange(len(cropped))
            cropped[restore] = original[restore]
    return cropped


def make_example_views(
    example: TokenExample,
    config: ViewConfig,
    *,
    max_view_tokens: int,
    block_separator_tokens: tuple[int, ...],
    mask_token_id: int,
    rng: random.Random,
) -> list[list[int]]:
    views = [
        make_view(
            example,
            scale=(config.global_scale_min, config.global_scale_max),
            max_view_tokens=max_view_tokens,
            mask_probability=config.mask_probability,
            block_shuffle_probability=config.block_shuffle_probability,
            block_separator_tokens=block_separator_tokens,
            mask_token_id=mask_token_id,
            rng=rng,
        )
        for _ in range(config.global_views)
    ]
    views.extend(
        make_view(
            example,
            scale=(config.local_scale_min, config.local_scale_max),
            max_view_tokens=max_view_tokens,
            mask_probability=config.mask_probability,
            block_shuffle_probability=config.block_shuffle_probability,
            block_separator_tokens=block_separator_tokens,
            mask_token_id=mask_token_id,
            rng=rng,
        )
        for _ in range(config.local_views)
    )
    return views


def make_span_mask(
    length: int,
    config: SpanMaskConfig,
    *,
    rng: random.Random,
) -> list[bool]:
    """Select an exact-size union of contiguous, geometrically sized spans."""
    if length < 1:
        raise ValueError("span masking requires a nonempty sequence")
    target_count = min(length, max(1, round(length * config.probability)))
    selected = [False] * length
    selected_count = 0
    while selected_count < target_count:
        candidates = [index for index, value in enumerate(selected) if not value]
        start = rng.choice(candidates)
        span_length = max(1, round(rng.expovariate(1.0 / config.mean_span_length)))
        stop = min(length, start + span_length)
        for index in range(start, stop):
            if selected[index]:
                continue
            selected[index] = True
            selected_count += 1
            if selected_count == target_count:
                break
    return selected


@dataclass(frozen=True)
class MaskedPredictionBatch:
    """A complete target and its compacted, position-preserving context."""

    target_ids: Tensor
    target_attention_mask: Tensor
    prediction_mask: Tensor
    context_ids: Tensor
    context_attention_mask: Tensor
    context_position_ids: Tensor

    def to(self, device: torch.device | str) -> MaskedPredictionBatch:
        return MaskedPredictionBatch(
            **{
                field: getattr(self, field).to(device)
                for field in self.__dataclass_fields__
            }
        )


class MaskedPredictionBatchSampler:
    """Sample complete targets and compacted variable-cardinality contexts."""

    def __init__(
        self,
        corpus: MixedCorpus | TokenShardCorpus,
        mask_config: SpanMaskConfig,
        *,
        max_tokens: int,
        min_source_tokens: int,
        pad_token_id: int = GPT2_PAD_ID,
        seed: int = 0,
    ):
        if not 1 <= min_source_tokens <= max_tokens:
            raise ValueError("min_source_tokens must be in [1, max_tokens]")
        self.corpus = corpus
        self.mask_config = mask_config
        self.max_tokens = max_tokens
        self.min_source_tokens = min_source_tokens
        self.pad_token_id = pad_token_id
        self.rng = random.Random(seed)

    def state_dict(self) -> dict:
        state = {"rng": self.rng.getstate()}
        if isinstance(self.corpus, MixedCorpus):
            state["answer_order"] = list(self.corpus._answer_order)
            state["answer_cursor"] = self.corpus._answer_cursor
            state["answer_draws"] = self.corpus.answer_draws
        return state

    def load_state_dict(self, state: dict) -> None:
        self.rng.setstate(state["rng"])
        if isinstance(self.corpus, MixedCorpus):
            self.corpus._answer_order = list(state["answer_order"])
            self.corpus._answer_cursor = int(state["answer_cursor"])
            self.corpus.answer_draws = int(state["answer_draws"])

    def batch(self, batch_size: int) -> MaskedPredictionBatch:
        """Return targets plus visible context tokens at original positions."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        rows: list[list[int]] = []
        masks: list[list[bool]] = []
        context_rows: list[list[int]] = []
        context_positions: list[list[int]] = []
        max_target_length = 1
        max_context_length = 1
        for _ in range(batch_size):
            example = self.corpus.sample(
                self.rng, self.min_source_tokens, self.max_tokens
            )
            tokens = list(map(int, example.tokens[: self.max_tokens].tolist()))
            mask = make_span_mask(len(tokens), self.mask_config, rng=self.rng)
            visible = [token for token, missing in zip(tokens, mask, strict=True) if not missing]
            visible_positions = [
                index for index, missing in enumerate(mask) if not missing
            ]
            rows.append(tokens)
            masks.append(mask)
            context_rows.append(visible)
            context_positions.append(visible_positions)
            max_target_length = max(max_target_length, len(tokens))
            max_context_length = max(max_context_length, len(visible))

        target_ids = torch.full(
            (batch_size, max_target_length), self.pad_token_id, dtype=torch.long
        )
        target_attention_mask = torch.zeros_like(target_ids, dtype=torch.bool)
        prediction_mask = torch.zeros_like(target_ids, dtype=torch.bool)
        context_ids = torch.full(
            (batch_size, max_context_length), self.pad_token_id, dtype=torch.long
        )
        context_attention_mask = torch.zeros_like(context_ids, dtype=torch.bool)
        context_position_ids = torch.zeros_like(context_ids)
        for batch_index, (tokens, mask, visible, visible_positions) in enumerate(
            zip(rows, masks, context_rows, context_positions, strict=True)
        ):
            length = len(tokens)
            target_ids[batch_index, :length] = torch.tensor(tokens)
            target_attention_mask[batch_index, :length] = True
            prediction_mask[batch_index, :length] = torch.tensor(mask)
            context_length = len(visible)
            if context_length:
                context_ids[batch_index, :context_length] = torch.tensor(visible)
                context_attention_mask[batch_index, :context_length] = True
                context_position_ids[batch_index, :context_length] = torch.tensor(
                    visible_positions
                )
        return MaskedPredictionBatch(
            target_ids=target_ids,
            target_attention_mask=target_attention_mask,
            prediction_mask=prediction_mask,
            context_ids=context_ids,
            context_attention_mask=context_attention_mask,
            context_position_ids=context_position_ids,
        )


class ViewBatchSampler:
    def __init__(
        self,
        corpus: MixedCorpus | TokenShardCorpus,
        view_config: ViewConfig,
        *,
        max_view_tokens: int,
        min_source_tokens: int,
        block_separator_tokens: tuple[int, ...],
        pad_token_id: int = GPT2_PAD_ID,
        mask_token_id: int = GPT2_MASK_ID,
        seed: int = 0,
    ):
        if not 1 <= min_source_tokens <= max_view_tokens:
            raise ValueError("min_source_tokens must be in [1, max_view_tokens]")
        self.corpus = corpus
        self.view_config = view_config
        self.max_view_tokens = max_view_tokens
        self.min_source_tokens = min_source_tokens
        self.block_separator_tokens = block_separator_tokens
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.rng = random.Random(seed)

    def state_dict(self) -> dict:
        state = {"rng": self.rng.getstate()}
        if isinstance(self.corpus, MixedCorpus):
            state["answer_order"] = list(self.corpus._answer_order)
            state["answer_cursor"] = self.corpus._answer_cursor
            state["answer_draws"] = self.corpus.answer_draws
        return state

    def load_state_dict(self, state: dict) -> None:
        self.rng.setstate(state["rng"])
        if isinstance(self.corpus, MixedCorpus):
            self.corpus._answer_order = list(state["answer_order"])
            self.corpus._answer_cursor = int(state["answer_cursor"])
            self.corpus.answer_draws = int(state["answer_draws"])

    def batch(self, batch_size: int) -> tuple[Tensor, Tensor]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        all_views: list[list[list[int]]] = []
        max_length = 1
        for batch_index in range(batch_size):
            example = self.corpus.sample(
                self.rng, self.min_source_tokens, self.max_view_tokens
            )
            views = make_example_views(
                example,
                self.view_config,
                max_view_tokens=self.max_view_tokens,
                block_separator_tokens=self.block_separator_tokens,
                mask_token_id=self.mask_token_id,
                rng=self.rng,
            )
            max_length = max(max_length, *(len(view) for view in views))
            all_views.append(views)

        token_ids = torch.full(
            (batch_size, self.view_config.count, max_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for batch_index, views in enumerate(all_views):
            for view_index, view in enumerate(views):
                length = len(view)
                token_ids[batch_index, view_index, :length] = torch.tensor(view)
                attention_mask[batch_index, view_index, :length] = True
        return token_ids, attention_mask
