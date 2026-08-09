"""Build the balanced four-domain corpus for KDA pretraining.

The mix is pinned by ``pretraining/k3_sources.json``. Remote shards must be
prepared first with ``scripts/prepare_k3_pretrain_sources.py``. The builder applies:

* exact token-budget balancing by source and domain;
* domain-aware quality filters and paragraph-preserving long-doc chunking;
* exact cross-source deduplication plus formatting-insensitive prose dedup;
* stable document-level validation splitting;
* global problem-registry exclusion: every document is refused if its problem
  key belongs to an evaluation, RL, or SFT split, or if it reproduces a
  protected problem's 13-gram anywhere in its text;
* loader-aligned shards with a one-token overlap, so no shard tails are lost.

This is a preprocessing workload and must be run through mlq.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postraining.decontaminate import ProblemGuard, load_guard
from scripts.build_math_mix_dataset import (
    DAPO_PREAMBLE,
    DAPO_REMINDER,
    DEEPMIND_EASY_MODULES,
    GPT2BatchEncoder,
    GPT2_EOT_ID,
    OPENMATH_SOURCES,
    SHARD_MAGIC,
    SHARD_VERSION,
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

    def __init__(self, guard: ProblemGuard):
        self.exact: set[bytes] = set()
        self.formatting: set[bytes] = set()
        self.guard = guard
        self.counts = Counter()

    def rejection(self, document: RawDocument) -> str | None:
        """Why this document is refused, or None if it is admitted.

        Callers want the reason, not a boolean: "held-out problem" and
        "duplicate of an earlier page" are different facts about the corpus,
        and folding them together in the per-source report hides exactly the
        number a decontamination change needs to be judged by.
        """
        reason = self.guard.reason(document.text, document.quality_keys)
        if reason is not None:
            self.counts[reason] += 1
            return reason

        exact = stable_digest(normalize_exact(document.text), person=b"pgolf-exact")
        if exact in self.exact:
            self.counts["exact_duplicate"] += 1
            return "exact_duplicate"
        formatting = None
        if document.domain in {"web", "knowledge"}:
            formatting = stable_digest(
                normalize_formatting(document.text), person=b"pgolf-format"
            )
            if formatting in self.formatting:
                self.counts["formatting_duplicate"] += 1
                return "formatting_duplicate"
        self.exact.add(exact)
        if formatting is not None:
            self.formatting.add(formatting)
        self.counts["accepted"] += 1
        return None

    def accept(self, document: RawDocument) -> bool:
        return self.rejection(document) is None


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


def weighted_deepmind_pairs(
    easy_dir: Path, modules: list[str], weights: dict[str, float]
) -> Iterator[tuple[str, str]]:
    """QA pairs drawn from modules in proportion to declared weights.

    `deepmind_qa_pairs` round-robins, which gives all 56 modules equal share.
    That is the wrong prior for a corpus whose measured weakness is
    arithmetic: ``arithmetic__add_or_sub`` and ``polynomials__compose`` are
    not equally worth the model's tokens. Weights are relative, a module named
    but absent is an error, and an unnamed module gets weight 1.0 so widening
    the module set never silently drops it.
    """
    unknown = sorted(set(weights) - set(modules))
    if unknown:
        raise ValueError(
            f"module weights name {unknown} which are not under {easy_dir}"
        )
    handles = {
        module: (easy_dir / f"{module}.txt").open(encoding="utf-8")
        for module in modules
    }
    share = {module: float(weights.get(module, 1.0)) for module in modules}
    if any(value <= 0 for value in share.values()):
        raise ValueError("DeepMind module weights must all be positive")
    emitted = dict.fromkeys(modules, 0)
    try:
        while handles:
            # Least completed share first, the same interleaving rule the
            # source scheduler uses, so the realized mix tracks the weights
            # rather than the file lengths.
            module = min(handles, key=lambda name: emitted[name] / share[name])
            handle = handles[module]
            question = handle.readline()
            answer = handle.readline()
            if not answer:
                if question.strip():
                    print(f"warning: dropping unpaired trailing question {question!r}")
                handles.pop(module).close()
                continue
            emitted[module] += 1
            yield question.strip(), answer.strip()
    finally:
        for handle in handles.values():
            handle.close()


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
    module_weights = source.get("module_weights")
    try:
        pairs = (
            weighted_deepmind_pairs(easy_dir, modules, module_weights)
            if module_weights
            else deepmind_qa_pairs(easy_dir)
        )
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
        pairs.close()
        DEEPMIND_EASY_MODULES[:] = original_modules


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


def load_corpus_tokenizer(name: str, gpt2: GPT2BatchEncoder) -> tuple[object, dict]:
    """Resolve --tokenizer to an encoder and the provenance that identifies it.

    The provenance is what makes a corpus interpretable later: token ids are
    meaningless without the vocabulary that produced them, and a shard stream
    carries no record of its own. `vocab_size` is checked against the shard
    dtype here rather than at the first overflow, which would otherwise
    surface as silently wrapped ids.
    """
    if name == "gpt2":
        return gpt2, {
            "kind": "gpt2",
            "name": "gpt2",
            "vocab_size": 50257,
            "eot_id": GPT2_EOT_ID,
            "directory": None,
            "spec_sha256": None,
            "ngrams_sha256": None,
        }
    from tokenization.spec import TokenizerSpec
    from tokenization.tokenizer import BatchEncoder

    directory = Path(name)
    spec_path = directory / "tokenizer.json"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"--tokenizer {name!r} is neither 'gpt2' nor a directory holding "
            "tokenizer.json; train one with tokenization/train.py"
        )
    encoder = BatchEncoder.from_directory(directory)
    spec = TokenizerSpec.read(spec_path)
    if encoder.vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError(
            f"tokenizer {name!r} has {encoder.vocab_size:,} tokens, which does "
            "not fit the uint16 shard format"
        )
    return encoder, {
        # `kind` is the discriminator every consumer branches on. `name` is a
        # directory basename and can be anything at all, including "gpt2":
        # discriminating on it would let a trained tokenizer be mistaken for
        # GPT-2 by any code that only reads the name.
        "kind": "toast_tst",
        "name": directory.name,
        "vocab_size": encoder.vocab_size,
        "eot_id": encoder.eot_id,
        "directory": directory.as_posix(),
        "spec_sha256": sha256_file(spec_path),
        # The spec alone does not pin the encoding: inference reads the n-gram
        # counts, and rewriting them changes the ids for the same text without
        # changing the spec hash.
        "ngrams_sha256": sha256_file(directory / spec.ngrams.filename),
    }


ITEM_REJECTION_SUFFIX = "registry_items_dropped"


def partition_rejection_counts(counts: Counter) -> tuple[dict, dict]:
    """Split document-unit rejections from item-unit ones for the manifest.

    Almost every rejection reason refuses a whole document, but item-level
    decontamination refuses a question/answer pair inside one. Reporting both
    in a single map invites a reader -- or a later script -- to sum them, and
    at 8..24 items per DeepMind document the total would be wrong by an order
    of magnitude for exactly one key.
    """
    items = {
        name: value
        for name, value in counts.items()
        if name.endswith(f":{ITEM_REJECTION_SUFFIX}")
    }
    documents = {
        name: value for name, value in counts.items() if name not in items
    }
    return documents, items


def decontaminated(
    document: RawDocument, guard: ProblemGuard, rejection_counts: Counter
) -> RawDocument | None:
    """Drop the contaminated items of a packed QA document, not the document.

    A DeepMind document carries 8..24 independent question/answer pairs, one
    per segment, each with its own quality key. Refusing all of them because
    one is a held-out problem is a real cost, not a rounding error: at the
    measured 5.2% document-level rejection rate, roughly fifteen clean pairs
    go with every contaminated one.

    Sources whose segments are not item-aligned -- a web page, an OpenMath row
    -- have nothing to trim, and fall through to the whole-document test.
    """
    if not document.qa or len(document.segments) != len(document.quality_keys):
        return document
    if len(document.segments) < 2:
        return document
    kept = [
        (segment, key)
        for segment, key in zip(document.segments, document.quality_keys, strict=True)
        if guard.reason(segment, (key,)) is None
    ]
    dropped = len(document.segments) - len(kept)
    if not dropped:
        return document
    # Counted in ITEMS, unlike every other key in this counter, which counts
    # documents. `partition_rejection_counts` splits the two apart before they
    # reach a manifest so nothing can add them together; the suffix is what it
    # keys on.
    rejection_counts[f"{document.source}:{ITEM_REJECTION_SUFFIX}"] += dropped
    if not kept:
        rejection_counts[f"{document.source}:registry_all_items_dropped"] += 1
        return None
    return replace(
        document,
        segments=tuple(segment for segment, _ in kept),
        quality_keys=tuple(key for _, key in kept),
    )


def encode_raw_documents(
    documents: Iterator[RawDocument],
    tokenizer,
    deduplicator: DocumentDeduplicator,
    validation: ValidationCollector,
    validation_permille: int,
    rejection_counts: Counter,
    max_document_tokens: int,
    eot_id: int = GPT2_EOT_ID,
) -> Iterator[np.ndarray]:
    pending: list[RawDocument] = []

    def flush(batch: list[RawDocument]) -> Iterator[np.ndarray]:
        flat = [segment for document in batch for segment in document.segments]
        encoded = tokenizer.encode(flat, out_type=int)
        cursor = 0
        for document in batch:
            tokens = [eot_id]
            for _ in document.segments:
                tokens.extend(encoded[cursor])
                cursor += 1
                if document.qa:
                    tokens.append(eot_id)
            if document.qa:
                tokens.pop()
            array = np.asarray(tokens, dtype=np.int32)
            # Keyed on the document TEXT, never on its token ids. Every arm
            # of the tokenizer ablation therefore holds out the identical
            # document set, which is what makes bits-per-byte comparable
            # across arms: the byte denominators cover the same bytes, and
            # the fixed one-byte charge for a separator (see
            # `pretraining/byte_accounting.py`) contributes the same offset
            # to each. Hashing token ids here would silently break both.
            split_key = int.from_bytes(
                stable_digest(document.text, person=b"pgolf-val")[:8],
                "little",
            )
            if split_key % 1000 < validation_permille:
                validation.add(document.domain, array)
            else:
                yield from split_token_document(
                    array, max_document_tokens, eot_id
                )

    for document in documents:
        document = decontaminated(document, deduplicator.guard, rejection_counts)
        if document is None:
            continue
        reason = quality_reason(document.text, document.domain)
        if reason:
            rejection_counts[f"{document.source}:{reason}"] += 1
            continue
        if rejected := deduplicator.rejection(document):
            rejection_counts[f"{document.source}:{rejected}"] += 1
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
    target: object = None,
    eot_id: int = GPT2_EOT_ID,
) -> Iterator[np.ndarray]:
    """Filter pre-tokenized shards, re-encoding them if the vocabulary changed.

    The fineweb shards are stored under GPT-2 ids, so ``tokenizer`` always
    decodes them. When ``target`` is a different tokenizer the decoded text is
    re-encoded under it: a corpus is only a controlled comparison if every
    source passed through the same vocabulary, and silently passing GPT-2 ids
    into a stream the model will read under another one would corrupt the
    largest source in the mix rather than merely bias it.
    """
    pending: list[np.ndarray] = []
    reencode = target is not None and target is not tokenizer

    def flush(batch: list[np.ndarray]) -> Iterator[np.ndarray]:
        texts = tokenizer.decode(
            [
                document[1:].astype(np.int64, copy=False).tolist()
                for document in batch
            ]
        )
        if reencode:
            batch = [
                np.asarray([eot_id, *ids], dtype=np.int32)
                for ids in target.encode(texts, out_type=int)
            ]
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
            if rejected := deduplicator.rejection(document):
                rejection_counts[f"{source['name']}:{rejected}"] += 1
                continue
            split_key = int.from_bytes(
                stable_digest(text, person=b"pgolf-val")[:8],
                "little",
            )
            if split_key % 1000 < validation_permille:
                validation.add(source["domain"], tokens)
            else:
                yield from split_token_document(
                    tokens, max_document_tokens, eot_id
                )

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
    eot_id: int = GPT2_EOT_ID,
) -> Iterator[np.ndarray]:
    if tokens.size <= max_document_tokens:
        yield tokens
        return
    if tokens[0] != eot_id:
        raise ValueError(
            f"token document does not begin with end-of-text id {eot_id}"
        )
    cursor = 1
    while cursor < tokens.size:
        take = min(max_document_tokens - 1, tokens.size - cursor)
        yield np.concatenate(
            (
                np.asarray([eot_id], dtype=np.int32),
                tokens[cursor : cursor + take],
            )
        )
        cursor += take


def token_shard_documents(path: Path) -> Iterator[np.ndarray]:
    from scripts.build_math_mix_dataset import fineweb_documents

    yield from fineweb_documents(path, GPT2_EOT_ID)


def _documents_from_shards(
    shards: list[Path], boundary_id: int
) -> Iterator[np.ndarray]:
    """Recover boundary-prefixed documents from an ordered shard list.

    Unlike :func:`token_shard_documents`, this accepts explicit shard paths so
    it can be used for the challenge validation stream as well as training.
    A document may cross a physical shard boundary, hence the carried tail.
    """

    carry = np.empty(0, dtype=np.int32)
    first_shard = True
    for shard in shards:
        header = np.fromfile(shard, dtype="<i4", count=256)
        if header.size != 256 or tuple(header[:2]) != (SHARD_MAGIC, SHARD_VERSION):
            raise ValueError(f"invalid token-shard header: {shard}")
        tokens = np.fromfile(shard, dtype="<u2", offset=256 * 4).astype(np.int32)
        if int(header[2]) != tokens.size:
            raise ValueError(
                f"token-shard payload length mismatch for {shard}: "
                f"header={int(header[2]):,}, payload={tokens.size:,}"
            )
        stream = np.concatenate((carry, tokens)) if carry.size else tokens
        starts = np.flatnonzero(stream == boundary_id)
        if first_shard and starts.size == 0:
            raise ValueError(
                f"{shard} contains no document boundary id {boundary_id}"
            )
        first_shard = False
        if starts.size == 0:
            carry = stream
            continue
        stream = stream[starts[0] :]
        starts = starts - starts[0]
        for begin, end in zip(starts[:-1], starts[1:], strict=True):
            yield stream[begin:end]
        carry = stream[starts[-1] :]
    if carry.size:
        yield carry


class StreamingTokenShardWriter:
    """Write one challenge-format shard without materializing it in RAM."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("w+b")
        self.handle.write(bytes(256 * 4))
        self.tokens = 0

    def append(self, tokens: np.ndarray) -> None:
        if tokens.ndim != 1 or not tokens.size:
            raise ValueError("streamed token documents must be non-empty vectors")
        if int(tokens.min()) < 0 or int(tokens.max()) > np.iinfo(np.uint16).max:
            raise ValueError("streamed token id does not fit uint16")
        self.handle.write(tokens.astype("<u2", copy=False).tobytes())
        self.tokens += int(tokens.size)
        if self.tokens > np.iinfo(np.int32).max:
            raise ValueError("challenge shard token count does not fit its int32 header")

    def close(self) -> None:
        if self.handle.closed:
            return
        header = np.zeros(256, dtype="<i4")
        header[0] = SHARD_MAGIC
        header[1] = SHARD_VERSION
        header[2] = self.tokens
        self.handle.seek(0)
        self.handle.write(header.tobytes())
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()

    def __enter__(self) -> StreamingTokenShardWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def materialize_challenge_validation(
    source_shards: list[Path],
    output_dir: Path,
    source_tokenizer: GPT2BatchEncoder,
    target_tokenizer,
    target_eot_id: int,
) -> tuple[list[Path], dict[str, int | bool]]:
    """Write FineWeb validation under the corpus tokenizer.

    The distributed trainer discovers ``fineweb_val_*.bin`` by filename and
    interprets every id using the corpus manifest. Copying GPT-2 validation
    ids into a ToaST+TST corpus therefore produces a plausible but meaningless
    BPB. The custom-tokenizer path decodes complete GPT-2 documents and
    re-encodes them losslessly, preserving one structural separator per
    document. GPT-2 corpora retain the byte-identical copy path.
    """

    if not source_shards:
        raise ValueError("challenge validation requires at least one source shard")
    source_shards = sorted(source_shards)
    source_tokens = 0
    for path in source_shards:
        header = np.fromfile(path, dtype="<i4", count=3)
        if header.size != 3 or tuple(header[:2]) != (SHARD_MAGIC, SHARD_VERSION):
            raise ValueError(f"invalid token-shard header: {path}")
        source_tokens += int(header[2])

    if target_tokenizer is source_tokenizer:
        outputs = []
        for path in source_shards:
            output = output_dir / path.name
            shutil.copy2(path, output)
            outputs.append(output)
        return outputs, {
            "reencoded": False,
            "documents": 0,
            "source_tokens": source_tokens,
            "tokens": source_tokens,
        }

    output = output_dir / "fineweb_val_000000.bin"
    documents = 0
    pending: list[np.ndarray] = []
    with StreamingTokenShardWriter(output) as writer:

        def flush(batch: list[np.ndarray]) -> None:
            nonlocal documents
            texts = source_tokenizer.decode(
                [document[1:].astype(np.int64, copy=False).tolist() for document in batch]
            )
            encoded = target_tokenizer.encode(texts, out_type=int)
            for ids in encoded:
                writer.append(np.asarray([target_eot_id, *ids], dtype=np.int32))
            documents += len(batch)

        for document in _documents_from_shards(source_shards, GPT2_EOT_ID):
            pending.append(document)
            if len(pending) == ENCODE_BATCH:
                flush(pending)
                pending = []
        if pending:
            flush(pending)
        target_tokens = writer.tokens
    return [output], {
        "reencoded": True,
        "documents": documents,
        "source_tokens": source_tokens,
        "tokens": target_tokens,
    }


class SourceStream:
    """A document iterator that can hand back the tail of a partial draw.

    Without this, a stage boundary silently destroys tokens: the interleaver
    truncates the document that overshoots a budget, and the generator has
    already advanced past it, so the remainder is gone. That is harmless at the
    very end of a build and wrong in the middle of one -- the anneal stage
    would find its source short by part of a document and fail, or redistribute
    around a shortfall that never really existed.
    """

    def __init__(self, documents: Iterator[np.ndarray]):
        self.documents = documents
        self.pending: np.ndarray | None = None

    def draw(self) -> np.ndarray | None:
        """The next document, or None once the underlying stream is spent.

        Deliberately not the iterator protocol: every caller has to be able to
        tell exhaustion from an empty draw and act on it, and a sentinel makes
        that impossible to skip past.
        """
        if self.pending is not None:
            document, self.pending = self.pending, None
            return document
        return next(self.documents, None)

    def pushback(self, tail: np.ndarray) -> None:
        if self.pending is not None:
            raise RuntimeError("SourceStream already holds a pending remainder")
        self.pending = tail


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
    streams = {
        name: source if isinstance(source, SourceStream) else SourceStream(source)
        for name, source in sources.items()
    }
    written = {name: 0 for name in sources}
    active = list(sources)
    while active:
        name = min(active, key=lambda item: written[item] / budgets[item])
        if written[name] >= budgets[name]:
            active.remove(name)
            continue
        document = streams[name].draw()
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
        if keep < document.size:
            # Keep the unused tail for whoever draws from this source next --
            # the next stage, or nobody if this was the last one.
            streams[name].pushback(document[keep:])
        if keep:
            written[name] += keep
            yield name, document[:keep]


def staged_interleave(
    sources: dict[str, Iterator[np.ndarray]],
    stages: list[tuple[str, dict[str, int]]],
    on_exhausted: str = "error",
) -> Iterator[tuple[str, str, np.ndarray]]:
    """Run several weight profiles back to back over one set of sources.

    This is how a mid-training anneal is expressed here: the final fraction of
    the stream is drawn under a different mixture, typically one that shifts
    weight onto mathematics and worked solutions while the learning rate is
    already decaying. Doing it in the corpus rather than in the trainer means
    the anneal survives every mechanism that already works on a token stream
    -- exact resume, shard alignment, provenance hashing -- and adds nothing
    to the training loop that could disagree with a checkpoint.

    The source iterators are shared across stages and never restarted, so a
    document read in the bulk stage cannot reappear in the anneal. Each stage
    yields its own realized totals, since a redistribution in one stage says
    nothing about the next.
    """
    # Wrapped once, outside the stage loop, so a document partially consumed
    # by one stage resumes in the next instead of being dropped at the seam.
    streams = {
        name: source if isinstance(source, SourceStream) else SourceStream(source)
        for name, source in sources.items()
    }
    for name, budgets in stages:
        active = {
            source: stream
            for source, stream in streams.items()
            if budgets.get(source, 0) > 0
        }
        stage_budgets = {source: budgets[source] for source in active}
        for source, tokens in logical_interleave(
            active, stage_budgets, on_exhausted
        ):
            yield name, source, tokens


def stage_budgets(
    total_tokens: int,
    weights: dict[str, float],
    anneal_weights: dict[str, float] | None,
    anneal_fraction: float,
) -> list[tuple[str, dict[str, int]]]:
    """Split the token budget into a bulk stage and an optional anneal stage.

    Every source keeps a key in both stages, zero included, so the manifest
    records what each stage deliberately excluded rather than leaving it to be
    inferred from an absence.
    """
    def allocate(tokens: int, fractions: dict[str, float]) -> dict[str, int]:
        used = {name: value for name, value in fractions.items() if value > 0}
        budgets = allocate_token_budgets(tokens, used)
        return {name: budgets.get(name, 0) for name in fractions}

    if anneal_weights is None:
        return [("bulk", allocate(total_tokens, weights))]
    if not 0 < anneal_fraction < 1:
        raise ValueError(
            f"anneal fraction must be strictly between 0 and 1, got "
            f"{anneal_fraction}"
        )
    missing = set(weights) ^ set(anneal_weights)
    if missing:
        raise ValueError(
            f"the anneal profile must weight exactly the same sources as the "
            f"bulk profile; {sorted(missing)} appear in only one"
        )
    anneal_tokens = int(total_tokens * anneal_fraction)
    return [
        ("bulk", allocate(total_tokens - anneal_tokens, weights)),
        ("anneal", allocate(anneal_tokens, anneal_weights)),
    ]


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
    parser.add_argument(
        "--tokenizer",
        default="gpt2",
        help="'gpt2', or a directory holding a trained ToaST+TST tokenizer. "
        "Changing this changes every token id in the corpus, so a dataset "
        "built under one may never be resumed or extended under another; the "
        "resolved identity and its hash go into the dataset manifest",
    )
    parser.add_argument(
        "--anneal-weights",
        help="optional second weight profile for the tail of the stream. The "
        "final --anneal-fraction of the token budget is drawn under this "
        "mixture instead, which is where a math-heavy mid-training anneal "
        "goes. Source iterators are shared, so the anneal never repeats a "
        "document the bulk stage already used",
    )
    parser.add_argument(
        "--anneal-fraction",
        type=float,
        default=0.15,
        help="share of the token budget drawn under --anneal-weights",
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
        "--problem-registry",
        type=Path,
        default=Path("data/problem_registry/v1"),
        help="versioned registry directory from scripts/build_problem_registry.py; "
        "every problem it assigns to eval, rl, or sft is refused here",
    )
    parser.add_argument(
        "--min-ngram-hits",
        type=int,
        default=1,
        help="protected n-grams a document must reproduce to be refused; "
        "1 is the GPT-3/Llama convention and the only value that has been "
        "measured on this corpus",
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
    all_weights = {source["name"]: source["weight"] for source in sources_config}
    if not math.isclose(sum(all_weights.values()), 1.0, abs_tol=1e-12):
        parser.error(f"source weights sum to {sum(all_weights.values())}, not 1")
    # A zero weight is a deliberate exclusion, not a source to allocate zero
    # tokens to; keeping it would make the budget allocator reject the profile.
    weights = {name: weight for name, weight in all_weights.items() if weight > 0}

    anneal_profile_path = Path(args.anneal_weights) if args.anneal_weights else None
    anneal_weights = None
    if anneal_profile_path is not None:
        anneal_profile = json.loads(anneal_profile_path.read_text())
        if set(anneal_profile["sources"]) != set(all_weights):
            parser.error(
                "anneal profile sources differ from the source manifest: "
                f"missing={sorted(set(all_weights) - set(anneal_profile['sources']))}, "
                f"extra={sorted(set(anneal_profile['sources']) - set(all_weights))}"
            )
        if not math.isclose(
            sum(anneal_profile["sources"].values()), 1.0, abs_tol=1e-12
        ):
            parser.error("anneal profile source weights do not sum to 1")
        # Union of both stages' sources, so a source the bulk stage excludes
        # can still carry the anneal (and the reverse), with an explicit zero
        # recorded in whichever stage does not use it.
        anneal_weights = {
            name: weight
            for name, weight in anneal_profile["sources"].items()
            if weight > 0
        }
        for name in anneal_weights:
            weights.setdefault(name, 0.0)
        for name in weights:
            anneal_weights.setdefault(name, 0.0)
        manifest["anneal_weight_profile"] = {
            "name": anneal_profile["name"],
            "path": anneal_profile_path.as_posix(),
            "fraction": args.anneal_fraction,
        }
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

    if not (args.problem_registry / "registry.parquet").exists():
        parser.error(
            f"no problem registry at {args.problem_registry}; build one with "
            "scripts/build_problem_registry.py. Pretraining without it would "
            "silently repeat evaluation and post-training problems."
        )
    guard, registry = load_guard(
        args.problem_registry, split="pretrain", min_ngram_hits=args.min_ngram_hits
    )
    print(
        f"problem registry: {len(guard.excluded):,} problems excluded from "
        f"pretraining, {len(guard.index):,} protected "
        f"{guard.index.ngram_size}-grams",
        flush=True,
    )
    deduplicator = DocumentDeduplicator(guard)
    validation = ValidationCollector(
        tuple(manifest["domains"]),
        args.validation_tokens,
    )
    gpt2 = GPT2BatchEncoder()
    tokenizer, tokenizer_provenance = load_corpus_tokenizer(args.tokenizer, gpt2)
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
                gpt2,
                deduplicator,
                validation,
                args.validation_permille,
                rejection_counts,
                args.max_document_tokens,
                tokenizer,
                tokenizer_provenance["eot_id"],
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
            tokenizer_provenance["eot_id"],
        )

    stages = stage_budgets(
        unique_tokens, weights, anneal_weights, args.anneal_fraction
    )
    budgets = Counter()
    for _, stage in stages:
        budgets.update(stage)
    budgets = dict(budgets)
    writer = LoaderAlignedShardWriter(
        output_dir,
        args.train_batch_tokens,
        args.training_steps,
        args.steps_per_shard,
    )
    written = Counter()
    stage_written: dict[str, Counter] = {name: Counter() for name, _ in stages}
    for stage_name, source_name, tokens in staged_interleave(
        source_iterators, stages, args.on_exhausted
    ):
        writer.append(tokens, source_name)
        written[source_name] += tokens.size
        stage_written[stage_name][source_name] += tokens.size
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
    challenge_validation_shards, challenge_validation = (
        materialize_challenge_validation(
            fineweb_val_shards,
            output_dir,
            gpt2,
            tokenizer,
            tokenizer_provenance["eot_id"],
        )
    )
    source_manifest_copy = output_dir / "source_manifest.json"
    source_manifest_copy.write_text(json.dumps(manifest, indent=2) + "\n")
    document_rejections, item_rejections = partition_rejection_counts(
        rejection_counts
    )
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
        "anneal_weight_profile_sha256": (
            sha256_file(anneal_profile_path)
            if anneal_profile_path is not None
            else None
        ),
        "command": sys.argv,
        "source_manifest": manifest_path.as_posix(),
        "resolved_files": resolved_files,
        "input_fingerprints": input_fingerprints,
        "tokenizer": tokenizer_provenance["name"],
        "tokenizer_provenance": tokenizer_provenance,
        "source_set": "full" if args.full_source_set else "sampled",
        "bos_id": tokenizer_provenance["eot_id"],
        "eos_id": tokenizer_provenance["eot_id"],
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
        "stages": [
            {
                "name": name,
                "weight_profile": (
                    args.anneal_weights if name == "anneal" else args.weights
                ),
                "requested_tokens": sum(stage.values()),
                "source_token_budgets": stage,
                "source_tokens_written": dict(stage_written[name]),
            }
            for name, stage in stages
        ],
        "source_tokens_written": dict(written),
        "domain_weights": manifest["domains"],
        "validation_tokens_per_domain": args.validation_tokens,
        "validation_shard_sizes": validation_sizes,
        "challenge_validation_shards": [
            path.name for path in challenge_validation_shards
        ],
        "challenge_validation": challenge_validation,
        "validation_documents": dict(validation.counts),
        "validation_split": f"stable hash < {args.validation_permille}/1000",
        "max_document_chars": args.max_document_chars,
        "max_document_tokens": args.max_document_tokens,
        "qa_template_fraction": args.qa_template_fraction,
        "problem_registry": {
            "directory": str(args.problem_registry),
            **guard.provenance(registry),
        },
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
        # Documents refused whole. Items refused inside a packed QA
        # document are a different unit and are reported separately.
        "rejection_counts": document_rejections,
        "item_rejection_counts": item_rejections,
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
