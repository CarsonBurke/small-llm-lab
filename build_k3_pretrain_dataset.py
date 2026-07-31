"""Build the balanced four-domain corpus for KDA pretraining.

The mix is pinned by ``pretraining/k3_sources.json``. Remote shards must be
prepared first with ``prepare_k3_pretrain_sources.py``. The builder applies:

* exact token-budget balancing by source and domain;
* domain-aware quality filters and paragraph-preserving long-doc chunking;
* exact cross-source deduplication plus formatting-insensitive prose dedup;
* stable document-level validation splitting;
* exact DAPO/AIME exclusion where a source exposes problem-level keys;
* loader-aligned shards with a one-token overlap, so no shard tails are lost.

This is a preprocessing workload and must be run through mlq.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from build_math_mix_dataset import (
    DAPO_PREAMBLE,
    DAPO_REMINDER,
    DEEPMIND_EASY_MODULES,
    GPT2BatchEncoder,
    GPT2_EOT_ID,
    OPENMATH_SOURCES,
    allocate_token_budgets,
    deepmind_qa_pairs,
    write_shard,
)

ENCODE_BATCH = 1024
DEFAULT_MANIFEST = Path("pretraining/k3_sources.json")
DEFAULT_REMOTE_ROOT = Path("data/pretraining_sources")
DEFAULT_CONTEXT_CHARS = 32_768
WHITESPACE_RE = re.compile(r"\s+")
PROSE_PUNCTUATION_RE = re.compile(r"[^\w]+", re.UNICODE)


@dataclass(frozen=True)
class RawDocument:
    source: str
    domain: str
    segments: tuple[str, ...]
    quality_keys: tuple[str, ...]
    qa: bool = False

    @property
    def text(self) -> str:
        return "\n".join(self.segments)


def stable_digest(text: str, *, person: bytes = b"pgolf-v5") -> bytes:
    return hashlib.blake2b(
        text.encode("utf-8", errors="ignore"), digest_size=16, person=person
    ).digest()


def normalize_exact(text: str) -> str:
    return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def normalize_formatting(text: str) -> str:
    normalized = normalize_exact(text).casefold()
    return PROSE_PUNCTUATION_RE.sub(" ", normalized).strip()


def normalize_problem(text: str) -> str:
    """Case/spacing normalization that preserves mathematical operators."""
    return normalize_exact(text).casefold()


class DocumentDeduplicator:
    """Deterministic exact plus conservative formatting-near deduplication."""

    def __init__(self, heldout_keys: set[bytes] | None = None):
        self.exact: set[bytes] = set()
        self.formatting: set[bytes] = set()
        self.heldout_keys = heldout_keys or set()
        self.counts = Counter()

    def accept(self, document: RawDocument) -> bool:
        for quality_key in document.quality_keys:
            heldout = stable_digest(
                normalize_problem(quality_key),
                person=b"pgolf-holdout",
            )
            if heldout in self.heldout_keys:
                self.counts["heldout_overlap"] += 1
                return False

        exact = stable_digest(normalize_exact(document.text), person=b"pgolf-exact")
        if exact in self.exact:
            self.counts["exact_duplicate"] += 1
            return False
        formatting = None
        if document.domain in {"web", "knowledge"}:
            formatting = stable_digest(
                normalize_formatting(document.text), person=b"pgolf-format"
            )
            if formatting in self.formatting:
                self.counts["formatting_duplicate"] += 1
                return False
        self.exact.add(exact)
        if formatting is not None:
            self.formatting.add(formatting)
        self.counts["accepted"] += 1
        return True


def repeated_line_fraction(text: str) -> float:
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if len(line) >= 20]
    if not lines:
        return 0.0
    counts = Counter(lines)
    repeated_chars = sum(len(line) * (count - 1) for line, count in counts.items())
    total_chars = sum(len(line) for line in lines)
    return repeated_chars / max(total_chars, 1)


def quality_reason(text: str, domain: str) -> str | None:
    stripped = text.strip()
    minimum = 100 if domain in {"code", "math"} else 200
    if len(stripped) < minimum:
        return "too_short"
    printable = sum(char.isprintable() or char in "\n\t" for char in stripped)
    if printable / len(stripped) < 0.90:
        return "low_printable_fraction"
    alpha = sum(char.isalpha() for char in stripped)
    domain_alpha_min = 0.03 if domain == "code" else 0.08
    if alpha / len(stripped) < domain_alpha_min:
        return "low_alpha_fraction"
    if repeated_line_fraction(stripped) > 0.35:
        return "repeated_lines"
    if "\x00" in stripped:
        return "nul_byte"
    return None


def paragraph_chunks(text: str, max_chars: int) -> Iterator[str]:
    """Split pathological long documents without cutting normal paragraphs."""
    text = text.strip()
    if len(text) <= max_chars:
        yield text
        return
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    current: list[str] = []
    current_chars = 0
    for paragraph in paragraphs:
        pieces = (
            [paragraph[start : start + max_chars] for start in range(0, len(paragraph), max_chars)]
            if len(paragraph) > max_chars
            else [paragraph]
        )
        for piece in pieces:
            added = len(piece) + (2 if current else 0)
            if current and current_chars + added > max_chars:
                yield "\n\n".join(current)
                current, current_chars = [], 0
            current.append(piece)
            current_chars += len(piece) + (2 if len(current) > 1 else 0)
    if current:
        yield "\n\n".join(current)


def logical_source_path(path: Path, source: dict) -> str:
    if source.get("repo_id"):
        for filename in source["files"]:
            if path.as_posix().endswith(filename):
                return filename
        raise ValueError(f"{path} is not listed in source {source['name']}")
    root = Path(source["path"])
    if root.is_file():
        return root.name
    return path.relative_to(root).as_posix()


def parquet_documents(
    files: list[Path],
    source: dict,
    max_chars: int,
    rejection_counts: Counter,
    metadata_counts: dict[str, Counter],
) -> Iterator[RawDocument]:
    columns = [source["text_column"]]
    title_column = source.get("title_column")
    if title_column:
        columns.append(title_column)
    for column in source.get("metadata_columns", []):
        if column not in columns:
            columns.append(column)
    for column in source.get("allowed_values", {}):
        if column not in columns:
            columns.append(column)
    parquet_files = {path: pq.ParquetFile(path) for path in files}
    row_groups = [
        (path, row_group)
        for path, parquet in parquet_files.items()
        for row_group in range(parquet.num_row_groups)
    ]
    row_groups.sort(
        key=lambda item: stable_digest(
            f"{source['name']}:{logical_source_path(item[0], source)}:{item[1]}",
            person=b"pgolf-rowgroup",
        )
    )
    for path, row_group in row_groups:
        parquet = parquet_files[path]
        missing = set(columns) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        for batch in parquet.iter_batches(
            batch_size=ENCODE_BATCH,
            row_groups=[row_group],
            columns=columns,
        ):
            rows = batch.to_pylist()
            rows.sort(
                key=lambda row: stable_digest(
                    str(row.get(source["text_column"], "")),
                    person=b"pgolf-row",
                )
            )
            for row in rows:
                rejected = False
                for column, allowed in source.get("allowed_values", {}).items():
                    if str(row.get(column, "")).casefold() not in allowed:
                        rejection_counts[
                            f"{source['name']}:disallowed_{column}"
                        ] += 1
                        rejected = True
                        break
                if rejected:
                    continue
                for column in source.get("metadata_columns", []):
                    metadata_counts[
                        f"{source['name']}:{column}"
                    ][str(row.get(column))] += 1
                text = row[source["text_column"]]
                if not isinstance(text, str):
                    continue
                if title_column and row.get(title_column):
                    text = f"# {row[title_column]}\n\n{text}"
                for chunk in paragraph_chunks(text, max_chars):
                    yield RawDocument(
                        source=source["name"],
                        domain=source["domain"],
                        segments=(chunk,),
                        quality_keys=(chunk,),
                    )


def jsonl_documents(
    files: list[Path],
    source: dict,
    max_chars: int,
) -> Iterator[RawDocument]:
    opener: Callable = gzip.open if any(path.suffix == ".gz" for path in files) else open
    for path in files:
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                text = row.get(source["text_column"])
                if not isinstance(text, str):
                    continue
                for chunk in paragraph_chunks(text, max_chars):
                    yield RawDocument(
                        source=source["name"],
                        domain=source["domain"],
                        segments=(chunk,),
                        quality_keys=(chunk,),
                    )


def stable_fraction(text: str, numerator: int, denominator: int) -> bool:
    value = int.from_bytes(stable_digest(text, person=b"pgolf-split")[:8], "little")
    return value % denominator < numerator


def dapo_wrap(problem: str, response: str) -> str:
    return f"{DAPO_PREAMBLE}\n\n{problem}\n\n{DAPO_REMINDER}\n{response}"


def openmath_documents(
    files: list[Path], source: dict, template_fraction: float
) -> Iterator[RawDocument]:
    columns = ["problem", "generated_solution", "expected_answer", "problem_source"]
    seen_problems: set[bytes] = set()
    numerator = round(template_fraction * 10_000)
    parquet_files = {path: pq.ParquetFile(path) for path in files}
    row_groups = [
        (path, row_group)
        for path, parquet in parquet_files.items()
        for row_group in range(parquet.num_row_groups)
    ]
    row_groups.sort(
        key=lambda item: stable_digest(
            f"{source['name']}:{logical_source_path(item[0], source)}:{item[1]}",
            person=b"pgolf-rowgroup",
        )
    )
    for path, row_group in row_groups:
        parquet = parquet_files[path]
        for batch in parquet.iter_batches(
            batch_size=ENCODE_BATCH,
            row_groups=[row_group],
            columns=columns,
        ):
            rows = batch.to_pylist()
            rows.sort(
                key=lambda row: stable_digest(
                    str(row.get("problem", "")),
                    person=b"pgolf-row",
                )
            )
            for row in rows:
                if row["problem_source"] not in OPENMATH_SOURCES:
                    continue
                if any(not isinstance(row.get(key), str) for key in columns[:3]):
                    continue
                problem = normalize_exact(row["problem"])
                solution = row["generated_solution"].strip()
                answer = row["expected_answer"].strip()
                if not problem or not solution or not answer:
                    continue
                problem_key = stable_digest(
                    normalize_problem(problem), person=b"pgolf-problem"
                )
                if problem_key in seen_problems:
                    continue
                seen_problems.add(problem_key)
                response = f"{solution}\nAnswer: {answer}"
                text = (
                    dapo_wrap(problem, response)
                    if stable_fraction(problem, numerator, 10_000)
                    else f"{problem}\n{response}"
                )
                yield RawDocument(
                    source=source["name"],
                    domain=source["domain"],
                    segments=(text,),
                    quality_keys=(problem,),
                    qa=True,
                )


def deepmind_documents(
    source: dict, template_fraction: float
) -> Iterator[RawDocument]:
    easy_dir = Path(source["path"])
    modules = sorted(path.stem for path in easy_dir.glob("*.txt"))
    if not modules:
        raise FileNotFoundError(f"no DeepMind modules under {easy_dir}")
    # Reuse the verified parser but expand its module set for breadth.
    original_modules = DEEPMIND_EASY_MODULES[:]
    DEEPMIND_EASY_MODULES[:] = modules
    numerator = round(template_fraction * 10_000)
    try:
        pairs = deepmind_qa_pairs(easy_dir)
        document_index = 0
        while True:
            count = 8 + document_index % 17
            rows: list[tuple[str, str]] = []
            for _ in range(count):
                pair = next(pairs, None)
                if pair is None:
                    break
                rows.append(pair)
            if not rows:
                return
            key = "\n".join(question for question, _ in rows)
            if stable_fraction(key, numerator, 10_000):
                segments = tuple(
                    dapo_wrap(question, f"Answer: {answer}")
                    for question, answer in rows
                )
            else:
                segments = tuple(
                    f"{question}\nAnswer: {answer}" for question, answer in rows
                )
            yield RawDocument(
                source=source["name"],
                domain=source["domain"],
                segments=segments,
                quality_keys=tuple(question for question, _ in rows),
                qa=True,
            )
            document_index += 1
    finally:
        DEEPMIND_EASY_MODULES[:] = original_modules


def prompt_content(row: dict) -> str | None:
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        return "\n".join(
            message.get("content", "")
            for message in prompt
            if isinstance(message, dict) and isinstance(message.get("content"), str)
        )
    return prompt if isinstance(prompt, str) else None


def strip_dapo_wrapper(text: str) -> str:
    text = text.strip()
    if text.startswith(DAPO_PREAMBLE):
        text = text[len(DAPO_PREAMBLE) :].lstrip()
    if text.endswith(DAPO_REMINDER):
        text = text[: -len(DAPO_REMINDER)].rstrip()
    return text


def heldout_problem_keys(paths: list[Path]) -> set[bytes]:
    keys: set[bytes] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"required heldout dataset is missing: {path}")
        parquet = pq.ParquetFile(path)
        if "prompt" not in parquet.schema_arrow.names:
            raise ValueError(
                f"required heldout dataset has no prompt column: {path}"
            )
        for batch in parquet.iter_batches(batch_size=2048, columns=["prompt"]):
            for row in batch.to_pylist():
                content = prompt_content(row)
                if content:
                    problem = strip_dapo_wrapper(content)
                    keys.add(
                        stable_digest(
                            normalize_problem(problem), person=b"pgolf-holdout"
                        )
                    )
    return keys


def source_files(source: dict, remote_root: Path) -> list[Path]:
    if source.get("repo_id"):
        paths = [remote_root / source["name"] / filename for filename in source["files"]]
    else:
        root = Path(source["path"])
        pattern = source.get("glob")
        paths = sorted(root.glob(pattern)) if pattern else [root]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{source['name']} is missing {len(missing)} input(s), first: {missing[0]}"
        )
    return paths


class ValidationCollector:
    def __init__(self, domains, tokens_per_domain: int):
        self.target = tokens_per_domain + 1
        self.tokens = {domain: [] for domain in domains}
        self.counts = Counter()

    def add(self, domain: str, tokens: np.ndarray) -> None:
        current = sum(part.size for part in self.tokens[domain])
        if current >= self.target:
            return
        keep = min(tokens.size, self.target - current)
        self.tokens[domain].append(tokens[:keep].copy())
        self.counts[domain] += 1

    def write(self, output_dir: Path) -> dict[str, int]:
        written = {}
        for domain, parts in self.tokens.items():
            count = sum(part.size for part in parts)
            if count < self.target:
                raise RuntimeError(
                    f"validation domain {domain} has {count:,} tokens, "
                    f"requires {self.target:,}"
                )
            payload = np.concatenate(parts)[: self.target]
            write_shard(output_dir / f"domainval_{domain}_000000.bin", payload)
            written[domain] = int(payload.size)
        return written


def encode_raw_documents(
    documents: Iterator[RawDocument],
    tokenizer: GPT2BatchEncoder,
    deduplicator: DocumentDeduplicator,
    validation: ValidationCollector,
    validation_permille: int,
    rejection_counts: Counter,
    max_document_tokens: int,
) -> Iterator[np.ndarray]:
    pending: list[RawDocument] = []

    def flush(batch: list[RawDocument]) -> Iterator[np.ndarray]:
        flat = [segment for document in batch for segment in document.segments]
        encoded = tokenizer.encode(flat, out_type=int)
        cursor = 0
        for document in batch:
            tokens = [GPT2_EOT_ID]
            for _ in document.segments:
                tokens.extend(encoded[cursor])
                cursor += 1
                if document.qa:
                    tokens.append(GPT2_EOT_ID)
            if document.qa:
                tokens.pop()
            array = np.asarray(tokens, dtype=np.int32)
            split_key = int.from_bytes(
                stable_digest(document.text, person=b"pgolf-val")[:8],
                "little",
            )
            if split_key % 1000 < validation_permille:
                validation.add(document.domain, array)
            else:
                yield from split_token_document(array, max_document_tokens)

    for document in documents:
        reason = quality_reason(document.text, document.domain)
        if reason:
            rejection_counts[f"{document.source}:{reason}"] += 1
            continue
        if not deduplicator.accept(document):
            rejection_counts[f"{document.source}:deduplicated"] += 1
            continue
        pending.append(document)
        if len(pending) == ENCODE_BATCH:
            yield from flush(pending)
            pending = []
    if pending:
        yield from flush(pending)


def filter_token_documents(
    documents: Iterator[np.ndarray],
    source: dict,
    tokenizer: GPT2BatchEncoder,
    deduplicator: DocumentDeduplicator,
    validation: ValidationCollector,
    validation_permille: int,
    rejection_counts: Counter,
    max_document_tokens: int,
) -> Iterator[np.ndarray]:
    pending: list[np.ndarray] = []

    def flush(batch: list[np.ndarray]) -> Iterator[np.ndarray]:
        texts = tokenizer.decode(
            [
                document[1:].astype(np.int64, copy=False).tolist()
                for document in batch
            ]
        )
        for tokens, text in zip(batch, texts, strict=True):
            document = RawDocument(
                source=source["name"],
                domain=source["domain"],
                segments=(text,),
                quality_keys=(text,),
            )
            reason = quality_reason(text, source["domain"])
            if reason:
                rejection_counts[f"{source['name']}:{reason}"] += 1
                continue
            if not deduplicator.accept(document):
                rejection_counts[f"{source['name']}:deduplicated"] += 1
                continue
            split_key = int.from_bytes(
                stable_digest(text, person=b"pgolf-val")[:8],
                "little",
            )
            if split_key % 1000 < validation_permille:
                validation.add(source["domain"], tokens)
            else:
                yield from split_token_document(tokens, max_document_tokens)

    for document in documents:
        pending.append(document)
        if len(pending) == ENCODE_BATCH:
            yield from flush(pending)
            pending = []
    if pending:
        yield from flush(pending)


def split_token_document(
    tokens: np.ndarray,
    max_document_tokens: int,
) -> Iterator[np.ndarray]:
    if tokens.size <= max_document_tokens:
        yield tokens
        return
    if tokens[0] != GPT2_EOT_ID:
        raise ValueError("token document does not begin with GPT-2 EOT")
    cursor = 1
    while cursor < tokens.size:
        take = min(max_document_tokens - 1, tokens.size - cursor)
        yield np.concatenate(
            (
                np.asarray([GPT2_EOT_ID], dtype=np.int32),
                tokens[cursor : cursor + take],
            )
        )
        cursor += take


def token_shard_documents(path: Path) -> Iterator[np.ndarray]:
    from build_math_mix_dataset import fineweb_documents

    yield from fineweb_documents(path, GPT2_EOT_ID)


def logical_interleave(
    sources: dict[str, Iterator[np.ndarray]],
    budgets: dict[str, int],
    on_exhausted: str = "error",
) -> Iterator[tuple[str, np.ndarray]]:
    """Interleave sources toward per-source budgets, balanced by fill ratio.

    ``on_exhausted="redistribute"`` moves an exhausted source's unfilled
    remainder onto the still-active sources, proportional to their remaining
    budgets with largest-remainder rounding, so the stream always delivers
    exactly ``sum(budgets)`` tokens (the loader-aligned writer requires the
    exact total). Sources that already completed their budget are not
    revisited, and the last active source exhausting still raises. The
    realized per-source totals land in the dataset manifest as
    ``source_tokens_written`` alongside the requested budgets.
    """
    if on_exhausted not in ("error", "redistribute"):
        raise ValueError(f"unsupported exhaustion policy {on_exhausted!r}")
    budgets = dict(budgets)
    written = {name: 0 for name in sources}
    active = list(sources)
    while active:
        name = min(active, key=lambda item: written[item] / budgets[item])
        if written[name] >= budgets[name]:
            active.remove(name)
            continue
        document = next(sources[name], None)
        if document is None:
            deficit = budgets[name] - written[name]
            if on_exhausted == "error" or len(active) == 1:
                raise RuntimeError(
                    f"source {name!r} exhausted at {written[name]:,} / "
                    f"{budgets[name]:,} tokens with no active source to "
                    "absorb the remainder"
                    if on_exhausted == "redistribute"
                    else f"source {name!r} exhausted at {written[name]:,} / "
                    f"{budgets[name]:,} tokens"
                )
            budgets[name] = written[name]
            active.remove(name)
            remaining = {
                other: budgets[other] - written[other] for other in active
            }
            total_remaining = sum(remaining.values())
            if total_remaining <= 0:
                raise RuntimeError(
                    f"source {name!r} exhausted with {deficit:,} tokens "
                    "unfilled and every active source already at budget"
                )
            shares = {
                other: deficit * remaining[other] // total_remaining
                for other in active
            }
            leftover = deficit - sum(shares.values())
            for other in sorted(
                active,
                key=lambda item: (
                    (deficit * remaining[item]) % total_remaining,
                    item,
                ),
                reverse=True,
            )[:leftover]:
                shares[other] += 1
            for other, share in shares.items():
                budgets[other] += share
            print(
                f"source {name!r} exhausted at {written[name]:,} tokens; "
                f"redistributed {deficit:,} tokens across "
                f"{len(active)} active sources",
                flush=True,
            )
            continue
        keep = min(document.size, budgets[name] - written[name])
        if keep:
            written[name] += keep
            yield name, document[:keep]


class LoaderAlignedShardWriter:
    """Write N*batch+1 shards, overlapping one boundary token."""

    def __init__(
        self,
        output_dir: Path,
        batch_tokens: int,
        total_steps: int,
        steps_per_shard: int,
    ):
        self.output_dir = output_dir
        self.batch_tokens = batch_tokens
        self.total_steps = total_steps
        self.steps_per_shard = steps_per_shard
        self.shard_index = 0
        self.steps_written = 0
        self.capacity = self._next_capacity()
        self.buffer = np.empty(self.capacity, dtype=np.uint16)
        self.fill = 0
        self.source_tokens = Counter()
        self.shards: list[dict] = []

    def _next_capacity(self) -> int:
        steps = min(self.steps_per_shard, self.total_steps - self.steps_written)
        return steps * self.batch_tokens + 1

    def append(self, tokens: np.ndarray, source: str) -> None:
        cursor = 0
        while cursor < tokens.size:
            take = min(self.capacity - self.fill, tokens.size - cursor)
            self.buffer[self.fill : self.fill + take] = tokens[cursor : cursor + take]
            self.fill += take
            self.source_tokens[source] += take
            cursor += take
            if self.fill == self.capacity:
                self._flush(source)

    def _flush(self, boundary_source: str) -> None:
        steps = (self.capacity - 1) // self.batch_tokens
        path = self.output_dir / f"fineweb_train_{self.shard_index:06d}.bin"
        write_shard(path, self.buffer)
        self.shards.append(
            {
                "index": self.shard_index,
                "tokens": self.capacity,
                "steps": steps,
                "source_tokens": dict(self.source_tokens),
            }
        )
        self.shard_index += 1
        self.steps_written += steps
        if self.steps_written == self.total_steps:
            self.fill = 0
            return
        boundary = int(self.buffer[-1])
        self.capacity = self._next_capacity()
        self.buffer = np.empty(self.capacity, dtype=np.uint16)
        self.buffer[0] = boundary
        self.fill = 1
        self.source_tokens = Counter({boundary_source: 1})

    def finish(self) -> None:
        if self.steps_written != self.total_steps or self.fill != 0:
            raise RuntimeError(
                f"incomplete aligned stream: {self.steps_written}/{self.total_steps} "
                f"steps, trailing fill={self.fill}"
            )


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance_files(source: dict, resolved: list[Path]) -> list[Path]:
    if source["kind"] == "token_shards":
        root = resolved[0]
        return sorted(root.glob("fineweb_*.bin"))
    if source["kind"] == "deepmind_qa":
        root = resolved[0]
        return sorted(root.glob("*.txt"))
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--weights",
        help="optional JSON profile overriding every source and domain weight",
    )
    parser.add_argument("--remote-root", default=str(DEFAULT_REMOTE_ROOT))
    parser.add_argument("--full-source-set", action="store_true")
    parser.add_argument("--output", default="data/datasets/k3mix_v5_gpt2_2k")
    parser.add_argument("--training-steps", type=int, default=2000)
    parser.add_argument("--train-batch-tokens", type=int, default=524_288)
    parser.add_argument("--steps-per-shard", type=int, default=200)
    parser.add_argument("--validation-tokens", type=int, default=1_048_576)
    parser.add_argument("--validation-permille", type=int, default=20)
    parser.add_argument("--max-document-chars", type=int, default=DEFAULT_CONTEXT_CHARS)
    parser.add_argument("--max-document-tokens", type=int, default=8192)
    parser.add_argument("--qa-template-fraction", type=float, default=0.20)
    parser.add_argument(
        "--on-exhausted",
        choices=("error", "redistribute"),
        default="error",
        help="what to do when a source runs dry before its budget: fail the "
        "build, or move the remainder onto still-active sources so a long "
        "build cannot die at the finish line (realized totals are recorded "
        "in the manifest)",
    )
    parser.add_argument(
        "--heldout",
        action="append",
        default=[
            "postraining/data/dapo-math-17k.parquet",
            "postraining/data/aime-2024.parquet",
            "postraining/data/aime-2026.parquet",
        ],
    )
    args = parser.parse_args()

    positive = {
        "training_steps": args.training_steps,
        "train_batch_tokens": args.train_batch_tokens,
        "steps_per_shard": args.steps_per_shard,
        "validation_tokens": args.validation_tokens,
        "max_document_chars": args.max_document_chars,
        "max_document_tokens": args.max_document_tokens,
    }
    if invalid := {name: value for name, value in positive.items() if value <= 0}:
        parser.error(f"positive values required: {invalid}")
    if args.max_document_tokens < 2:
        parser.error("--max-document-tokens must be at least 2")
    if not 1 <= args.validation_permille < 1000:
        parser.error("--validation-permille must be in [1, 1000)")
    if not 0 <= args.qa_template_fraction <= 1:
        parser.error("--qa-template-fraction must be in [0, 1]")

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    sources_config = manifest["sources"]
    if args.full_source_set:
        for source in sources_config:
            if "full_files" in source:
                source["files"] = source["full_files"]
    weight_profile_path = Path(args.weights) if args.weights else None
    if weight_profile_path is not None:
        profile = json.loads(weight_profile_path.read_text())
        profile_weights = profile["sources"]
        source_names = {source["name"] for source in sources_config}
        if set(profile_weights) != source_names:
            parser.error(
                "weight profile sources differ from the source manifest: "
                f"missing={sorted(source_names - set(profile_weights))}, "
                f"extra={sorted(set(profile_weights) - source_names)}"
            )
        for source in sources_config:
            source["weight"] = profile_weights[source["name"]]
        manifest["domains"] = profile["domains"]
        manifest["weight_profile"] = {
            "name": profile["name"],
            "path": weight_profile_path.as_posix(),
        }
    weights = {source["name"]: source["weight"] for source in sources_config}
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12):
        parser.error(f"source weights sum to {sum(weights.values())}, not 1")
    domain_weights = Counter()
    for source in sources_config:
        domain_weights[source["domain"]] += source["weight"]
    domains_match = set(domain_weights) == set(manifest["domains"]) and all(
        math.isclose(
            domain_weights[domain],
            manifest["domains"][domain],
            abs_tol=1e-12,
        )
        for domain in domain_weights
    )
    if not domains_match:
        parser.error(
            f"source-derived domain weights {dict(domain_weights)} disagree with "
            f"manifest {manifest['domains']}"
        )
    unique_tokens = args.training_steps * args.train_batch_tokens + 1
    validation_odds = args.validation_permille / (
        1000 - args.validation_permille
    )
    estimated_validation_tokens = {
        domain: unique_tokens * weight * validation_odds
        for domain, weight in domain_weights.items()
    }
    validation_safety_margin = 1.25
    undersized_validation = {
        domain: round(tokens)
        for domain, tokens in estimated_validation_tokens.items()
        if tokens < validation_safety_margin * (args.validation_tokens + 1)
    }
    if undersized_validation:
        parser.error(
            "validation split is too small for the requested domain weights; "
            f"estimated tokens with a {validation_safety_margin:.2f}x safety "
            f"margin: {undersized_validation}. Increase --validation-permille "
            "or reduce --validation-tokens."
        )

    final_output_dir = Path(args.output)
    output_dir = final_output_dir.with_name(final_output_dir.name + ".building")
    if final_output_dir.exists():
        parser.error(
            f"{final_output_dir} already exists; use a fresh --output"
        )
    if output_dir.exists():
        parser.error(
            f"incomplete build directory {output_dir} already exists; "
            "inspect it, then remove it or choose a fresh --output"
        )
    output_dir.mkdir(parents=True)

    heldout = heldout_problem_keys([Path(path) for path in args.heldout])
    deduplicator = DocumentDeduplicator(heldout)
    validation = ValidationCollector(
        tuple(manifest["domains"]),
        args.validation_tokens,
    )
    tokenizer = GPT2BatchEncoder()
    rejection_counts = Counter()
    metadata_counts: dict[str, Counter] = defaultdict(Counter)
    remote_root = Path(args.remote_root)

    source_iterators: dict[str, Iterator[np.ndarray]] = {}
    resolved_files: dict[str, list[str]] = {}
    input_fingerprints: dict[str, list[dict]] = {}
    for source in sources_config:
        files = source_files(source, remote_root)
        resolved_files[source["name"]] = [
            logical_source_path(path, source) for path in files
        ]
        provenance_files = source_provenance_files(source, files)
        input_fingerprints[source["name"]] = [
            {
                "path": (
                    path.name
                    if source["kind"] in {"token_shards", "deepmind_qa"}
                    else logical_source_path(path, source)
                ),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in provenance_files
        ]
        kind = source["kind"]
        if kind == "token_shards":
            source_iterators[source["name"]] = filter_token_documents(
                token_shard_documents(files[0]),
                source,
                tokenizer,
                deduplicator,
                validation,
                args.validation_permille,
                rejection_counts,
                args.max_document_tokens,
            )
            continue
        if kind == "parquet_text":
            raw = parquet_documents(
                files,
                source,
                args.max_document_chars,
                rejection_counts,
                metadata_counts,
            )
        elif kind == "jsonl_text":
            raw = jsonl_documents(files, source, args.max_document_chars)
        elif kind == "openmath_qa":
            raw = openmath_documents(files, source, args.qa_template_fraction)
        elif kind == "deepmind_qa":
            raw = deepmind_documents(source, args.qa_template_fraction)
        else:
            raise ValueError(f"unsupported source kind {kind!r}")
        source_iterators[source["name"]] = encode_raw_documents(
            raw,
            tokenizer,
            deduplicator,
            validation,
            args.validation_permille,
            rejection_counts,
            args.max_document_tokens,
        )

    budgets = allocate_token_budgets(unique_tokens, weights)
    writer = LoaderAlignedShardWriter(
        output_dir,
        args.train_batch_tokens,
        args.training_steps,
        args.steps_per_shard,
    )
    written = Counter()
    for source_name, tokens in logical_interleave(
        source_iterators, budgets, args.on_exhausted
    ):
        writer.append(tokens, source_name)
        written[source_name] += tokens.size
    writer.finish()
    if args.on_exhausted == "error":
        if dict(written) != budgets:
            raise AssertionError(
                f"written source budgets {dict(written)} != {budgets}"
            )
    elif sum(written.values()) != sum(budgets.values()):
        raise AssertionError(
            f"redistributed stream wrote {sum(written.values()):,} tokens "
            f"against a {sum(budgets.values()):,} token budget"
        )

    validation_sizes = validation.write(output_dir)
    fineweb_config = next(
        source for source in sources_config if source["name"] == "fineweb"
    )
    fineweb_val_shards = sorted(
        Path(fineweb_config["path"]).glob("fineweb_val_*.bin")
    )
    if not fineweb_val_shards:
        raise FileNotFoundError(
            f"no FineWeb validation shards under {fineweb_config['path']}"
        )
    for path in fineweb_val_shards:
        shutil.copy2(path, output_dir / path.name)
    source_manifest_copy = output_dir / "source_manifest.json"
    source_manifest_copy.write_text(json.dumps(manifest, indent=2) + "\n")
    result = {
        "version": 5,
        "builder": str(Path(__file__).resolve()),
        "builder_git_revision": git_revision(),
        "builder_sha256": sha256_file(Path(__file__)),
        "source_manifest_sha256": sha256_file(manifest_path),
        "weight_profile_sha256": (
            sha256_file(weight_profile_path)
            if weight_profile_path is not None
            else None
        ),
        "command": sys.argv,
        "source_manifest": manifest_path.as_posix(),
        "resolved_files": resolved_files,
        "input_fingerprints": input_fingerprints,
        "tokenizer": "gpt2",
        "source_set": "full" if args.full_source_set else "sampled",
        "bos_id": GPT2_EOT_ID,
        "eos_id": GPT2_EOT_ID,
        "training_steps": args.training_steps,
        "train_batch_tokens": args.train_batch_tokens,
        "unique_stream_tokens": unique_tokens,
        "physical_shard_tokens": sum(shard["tokens"] for shard in writer.shards),
        "boundary_overlap_tokens": max(len(writer.shards) - 1, 0),
        "loader_aligned": True,
        "steps_per_shard": args.steps_per_shard,
        "shards": writer.shards,
        "source_weights": weights,
        "source_token_budgets": budgets,
        "source_tokens_written": dict(written),
        "domain_weights": manifest["domains"],
        "validation_tokens_per_domain": args.validation_tokens,
        "validation_shard_sizes": validation_sizes,
        "challenge_validation_shards": [
            path.name for path in fineweb_val_shards
        ],
        "validation_documents": dict(validation.counts),
        "validation_split": f"stable hash < {args.validation_permille}/1000",
        "max_document_chars": args.max_document_chars,
        "max_document_tokens": args.max_document_tokens,
        "qa_template_fraction": args.qa_template_fraction,
        "heldout_problem_keys": len(heldout),
        "deduplication": {
            "exact_normalized": True,
            "formatting_insensitive": True,
            "upstream_fuzzy_filters": {
                source["name"]: source["quality"]
                for source in sources_config
                if "dedup" in source["quality"].lower()
                or "simhash" in source["quality"].lower()
            },
            "counts": dict(deduplicator.counts),
        },
        "rejection_counts": dict(rejection_counts),
        "source_metadata_candidate_counts": {
            name: dict(counts)
            for name, counts in sorted(metadata_counts.items())
        },
        "ordering": "deterministic least-completed source token budget",
    }
    (output_dir / "mix_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    output_dir.rename(final_output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
