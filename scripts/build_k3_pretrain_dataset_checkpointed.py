"""Build the K3 corpus with durable, source-granular preprocessing checkpoints.

The original builder interleaves filtering, tokenization, and final shard
assembly. That is economical for small corpora, but a late source-capacity
failure forces the entire preprocessing pass to be repeated. This builder
materializes one exact-budget token cache per source first. Each completed
source is an atomic checkpoint containing:

* its tokenized training documents;
* the accepted-document deduplication journal;
* cumulative validation data and filtering counters;
* pinned input fingerprints and preprocessing settings.

On restart, completed sources are verified and reused. Only the source that
was being processed when the job stopped is rebuilt. Final loader-aligned
shard assembly is a sequential binary copy from those caches.

This is a preprocessing workload and must be run through mlq.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postraining import decontaminate, problem_registry
from scripts import build_k3_pretrain_dataset as base
from scripts.build_math_mix_dataset import GPT2BatchEncoder, allocate_token_budgets

CACHE_FORMAT_VERSION = 1
TOKEN_LENGTH = struct.Struct("<I")
DEDUP_RECORD_BYTES = 32
ZERO_DIGEST = b"\0" * 16


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def counter_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.items()}


def write_json_fsync(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2) + "\n"
    with path.open("w") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def fsync_path(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def artifact_fingerprint(path: Path) -> dict:
    return {
        "size_bytes": path.stat().st_size,
        "sha256": base.sha256_file(path),
    }


def token_cache_stats(path: Path) -> dict:
    digest = hashlib.sha256()
    tokens = 0
    documents = 0
    size_bytes = 0
    with path.open("rb") as handle:
        while header := handle.read(TOKEN_LENGTH.size):
            digest.update(header)
            size_bytes += len(header)
            if len(header) != TOKEN_LENGTH.size:
                raise ValueError(f"truncated token-cache header: {path}")
            (length,) = TOKEN_LENGTH.unpack(header)
            payload = handle.read(length * np.dtype("<u2").itemsize)
            digest.update(payload)
            size_bytes += len(payload)
            if len(payload) != length * np.dtype("<u2").itemsize:
                raise ValueError(f"truncated token-cache payload: {path}")
            tokens += length
            documents += 1
    return {
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
        "tokens": tokens,
        "documents": documents,
    }


class JournaledDeduplicator(base.DocumentDeduplicator):
    """Document deduplicator that durably records every accepted digest."""

    def __init__(
        self,
        guard: base.ProblemGuard,
        journal: BinaryIO | None = None,
    ):
        super().__init__(guard)
        self.journal = journal

    def rejection(self, document: base.RawDocument) -> str | None:
        rejected = super().rejection(document)
        if rejected is not None or self.journal is None:
            return rejected
        exact = base.stable_digest(
            base.normalize_exact(document.text), person=b"pgolf-exact"
        )
        formatting = (
            base.stable_digest(
                base.normalize_formatting(document.text),
                person=b"pgolf-format",
            )
            if document.domain in {"web", "knowledge"}
            else ZERO_DIGEST
        )
        self.journal.write(exact)
        self.journal.write(formatting)
        return None

    def load_journal(self, path: Path) -> None:
        with path.open("rb") as handle:
            while record := handle.read(DEDUP_RECORD_BYTES):
                if len(record) != DEDUP_RECORD_BYTES:
                    raise ValueError(f"truncated dedup journal: {path}")
                self.exact.add(record[:16])
                if record[16:] != ZERO_DIGEST:
                    self.formatting.add(record[16:])


class TokenCacheWriter:
    """Length-prefixed uint16 document stream."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("wb")
        self.tokens = 0
        self.documents = 0

    def append(self, tokens: np.ndarray) -> None:
        if tokens.ndim != 1 or not tokens.size:
            raise ValueError("cached token documents must be non-empty vectors")
        if int(tokens.min()) < 0 or int(tokens.max()) > np.iinfo(np.uint16).max:
            raise ValueError("cached token id does not fit uint16")
        payload = tokens.astype("<u2", copy=False)
        self.handle.write(TOKEN_LENGTH.pack(payload.size))
        self.handle.write(payload.tobytes())
        self.tokens += int(payload.size)
        self.documents += 1

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()

    def __enter__(self) -> TokenCacheWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def token_cache_documents(path: Path) -> Iterator[np.ndarray]:
    with path.open("rb") as handle:
        while header := handle.read(TOKEN_LENGTH.size):
            if len(header) != TOKEN_LENGTH.size:
                raise ValueError(f"truncated token-cache header: {path}")
            (length,) = TOKEN_LENGTH.unpack(header)
            payload = handle.read(length * np.dtype("<u2").itemsize)
            if len(payload) != length * np.dtype("<u2").itemsize:
                raise ValueError(f"truncated token-cache payload: {path}")
            yield np.frombuffer(payload, dtype="<u2").astype(np.int32)


def save_validation(
    directory: Path,
    validation: base.ValidationCollector,
) -> dict[str, int]:
    sizes = {}
    for domain, parts in validation.tokens.items():
        payload = (
            np.concatenate(parts).astype(np.int32, copy=False)
            if parts
            else np.empty(0, dtype=np.int32)
        )
        np.save(directory / f"validation_{domain}.npy", payload)
        sizes[domain] = int(payload.size)
    return sizes


def load_validation(
    directory: Path,
    domains: tuple[str, ...],
    tokens_per_domain: int,
    counts: dict[str, int],
) -> base.ValidationCollector:
    validation = base.ValidationCollector(domains, tokens_per_domain)
    for domain in domains:
        path = directory / f"validation_{domain}.npy"
        payload = np.load(path, allow_pickle=False)
        if payload.ndim != 1:
            raise ValueError(f"invalid validation checkpoint: {path}")
        if payload.size:
            validation.tokens[domain] = [payload.astype(np.int32, copy=False)]
    validation.counts.update(counts)
    return validation


def source_input_fingerprints(source: dict, files: list[Path]) -> list[dict]:
    return [
        {
            "path": (
                path.name
                if source["kind"] in {"token_shards", "deepmind_qa"}
                else base.logical_source_path(path, source)
            ),
            "size_bytes": path.stat().st_size,
            "sha256": base.sha256_file(path),
        }
        for path in base.source_provenance_files(source, files)
    ]


def source_iterator(
    source: dict,
    files: list[Path],
    tokenizer,
    deduplicator: JournaledDeduplicator,
    validation: base.ValidationCollector,
    validation_permille: int,
    rejection_counts: Counter,
    metadata_counts: dict[str, Counter],
    max_document_chars: int,
    max_document_tokens: int,
    qa_template_fraction: float,
    gpt2: GPT2BatchEncoder | None = None,
    eot_id: int = base.GPT2_EOT_ID,
) -> Iterator[np.ndarray]:
    kind = source["kind"]
    if kind == "token_shards":
        return base.filter_token_documents(
            base.token_shard_documents(files[0]),
            source,
            gpt2 if gpt2 is not None else tokenizer,
            deduplicator,
            validation,
            validation_permille,
            rejection_counts,
            max_document_tokens,
            tokenizer,
            eot_id,
        )
    if kind == "parquet_text":
        raw = base.parquet_documents(
            files,
            source,
            max_document_chars,
            rejection_counts,
            metadata_counts,
        )
    elif kind == "jsonl_text":
        raw = base.jsonl_documents(files, source, max_document_chars)
    elif kind == "openmath_qa":
        raw = base.openmath_documents(files, source, qa_template_fraction)
    elif kind == "deepmind_qa":
        raw = base.deepmind_documents(source, qa_template_fraction)
    else:
        raise ValueError(f"unsupported source kind {kind!r}")
    return base.encode_raw_documents(
        raw,
        tokenizer,
        deduplicator,
        validation,
        validation_permille,
        rejection_counts,
        max_document_tokens,
        eot_id,
    )


def checkpoint_signature(args: argparse.Namespace, registry: dict) -> dict:
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "checkpoint_builder_sha256": base.sha256_file(Path(__file__)),
        "base_builder_sha256": base.sha256_file(Path(base.__file__)),
        # The admission rule lives in these two modules, not in the builder
        # that calls them. Hashing only the builders would let a changed guard
        # resume a cache whose documents were admitted under the old rule.
        "decontaminate_sha256": base.sha256_file(Path(decontaminate.__file__)),
        "problem_registry_sha256": base.sha256_file(Path(problem_registry.__file__)),
        "validation_tokens": args.validation_tokens,
        "validation_permille": args.validation_permille,
        "max_document_chars": args.max_document_chars,
        "max_document_tokens": args.max_document_tokens,
        "qa_template_fraction": args.qa_template_fraction,
        "min_ngram_hits": args.min_ngram_hits,
        # The anneal changes how much each source must yield, so a cache built
        # without it cannot be resumed into a build that has one.
        "anneal_weights_sha256": (
            base.sha256_file(Path(args.anneal_weights))
            if args.anneal_weights
            else None
        ),
        "anneal_fraction": args.anneal_fraction if args.anneal_weights else None,
        # Binding both artifact digests makes a rebuilt or widened registry a
        # cache-invalidating change, so a resumed build can never mix
        # documents admitted under two different decontamination rules.
        "problem_registry": registry,
    }


def tokenizer_signature(tokenizer, provenance: dict) -> dict:
    """What the resume check must see to know two runs share a vocabulary.

    For GPT-2 that means the serialized backend and the library versions that
    produced it; for a trained ToaST+TST tokenizer it is the spec hash, which
    already covers the vocabulary, the numeric scheme, and the n-gram
    reference. Either way a cache built under one vocabulary can never be
    resumed under another, because every cached token id would change meaning.
    """
    if provenance["kind"] != "gpt2":
        return dict(provenance)
    import tokenizers
    import transformers

    backend = tokenizer._tokenizer.backend_tokenizer.to_str()
    return {
        "name": "gpt2",
        "backend_sha256": hashlib.sha256(backend.encode()).hexdigest(),
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
    }


def validation_source_budgets(
    sources: list[dict],
    tokens_per_domain: int,
) -> dict[str, int]:
    target = tokens_per_domain + 1
    result = {}
    # A zero-weight source is a deliberate exclusion: it contributes no
    # training tokens, so demanding a positive validation budget for it would
    # reject an otherwise valid profile.
    sources = [source for source in sources if source["weight"] > 0]
    domains = sorted({source["domain"] for source in sources})
    for domain in domains:
        members = [source for source in sources if source["domain"] == domain]
        rational_weights = {
            source["name"]: Fraction(str(source["weight"])) for source in members
        }
        total_weight = sum(rational_weights.values())
        exact = {
            name: target * weight / total_weight
            for name, weight in rational_weights.items()
        }
        budgets = {
            name: value.numerator // value.denominator
            for name, value in exact.items()
        }
        remainder = target - sum(budgets.values())
        for name in sorted(
            budgets,
            key=lambda item: (exact[item] - budgets[item], item),
            reverse=True,
        )[:remainder]:
            budgets[name] += 1
        if any(value <= 0 for value in budgets.values()):
            raise ValueError(
                f"validation target is too small for domain {domain!r}: "
                f"{budgets}"
            )
        result.update(budgets)
    return result


def validate_completed_checkpoint(
    directory: Path,
    source: dict,
    source_index: int,
    budget: int,
    signature: dict,
    fingerprints: list[dict],
) -> dict:
    metadata_path = directory / "checkpoint.json"
    if not metadata_path.exists():
        raise ValueError(f"incomplete source checkpoint: {directory}")
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "source": source["name"],
        "source_index": source_index,
        "source_spec_sha256": canonical_sha256(source),
        "budget": budget,
        "signature": signature,
        "input_fingerprints": fingerprints,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"checkpoint {directory} is incompatible with this build: "
            f"{mismatches}"
        )
    if metadata.get("tokens") != budget:
        raise ValueError(
            f"checkpoint {directory} has {metadata.get('tokens')} tokens, "
            f"expected {budget}"
        )
    expected_artifacts = metadata.get("artifacts")
    if not isinstance(expected_artifacts, dict):
        raise ValueError(f"checkpoint {directory} has no artifact fingerprints")
    artifact_names = {
        "tokens.bin",
        "dedup.bin",
        "state.json",
        *(f"validation_{domain}.npy" for domain in metadata["domains"]),
    }
    if set(expected_artifacts) != artifact_names:
        raise ValueError(
            f"checkpoint {directory} artifact list differs: "
            f"{set(expected_artifacts) ^ artifact_names}"
        )
    for name in sorted(artifact_names):
        path = directory / name
        if not path.is_file():
            raise ValueError(f"checkpoint artifact is missing: {path}")
        actual = (
            token_cache_stats(path)
            if name == "tokens.bin"
            else artifact_fingerprint(path)
        )
        if actual != expected_artifacts[name]:
            raise ValueError(
                f"checkpoint artifact failed verification: {path}; "
                f"actual={actual}, expected={expected_artifacts[name]}"
            )
    dedup_size = (directory / "dedup.bin").stat().st_size
    if dedup_size % DEDUP_RECORD_BYTES:
        raise ValueError(f"truncated dedup journal: {directory / 'dedup.bin'}")
    return metadata


def load_state(
    directory: Path,
    domains: tuple[str, ...],
    validation_tokens: int,
) -> tuple[base.ValidationCollector, Counter, dict[str, Counter], Counter]:
    state = json.loads((directory / "state.json").read_text())
    validation = load_validation(
        directory,
        domains,
        validation_tokens,
        state["validation_counts"],
    )
    rejection_counts = Counter(state["rejection_counts"])
    metadata_counts = defaultdict(
        Counter,
        {
            name: Counter(counts)
            for name, counts in state["metadata_counts"].items()
        },
    )
    dedup_counts = Counter(state["dedup_counts"])
    return validation, rejection_counts, metadata_counts, dedup_counts


def write_state(
    directory: Path,
    validation: base.ValidationCollector,
    rejection_counts: Counter,
    metadata_counts: dict[str, Counter],
    dedup_counts: Counter,
) -> None:
    save_validation(directory, validation)
    state = {
        "validation_counts": counter_dict(validation.counts),
        "rejection_counts": counter_dict(rejection_counts),
        "metadata_counts": {
            name: counter_dict(counts)
            for name, counts in sorted(metadata_counts.items())
        },
        "dedup_counts": counter_dict(dedup_counts),
    }
    (directory / "state.json").write_text(json.dumps(state, indent=2) + "\n")


def build_source_caches(
    *,
    args: argparse.Namespace,
    sources: list[dict],
    budgets: dict[str, int],
    domain_names: tuple[str, ...],
    cache_root: Path,
    guard: base.ProblemGuard,
    signature: dict,
    tokenizer,
    gpt2: GPT2BatchEncoder,
    eot_id: int,
    source_validation_budgets: dict[str, int],
) -> tuple[
    dict[str, list[str]],
    dict[str, list[dict]],
    base.ValidationCollector,
    Counter,
    dict[str, Counter],
    JournaledDeduplicator,
]:
    cache_root.mkdir(parents=True, exist_ok=True)
    validation = base.ValidationCollector(domain_names, args.validation_tokens)
    rejection_counts: Counter = Counter()
    metadata_counts: dict[str, Counter] = defaultdict(Counter)
    deduplicator = JournaledDeduplicator(guard)
    resolved_files: dict[str, list[str]] = {}
    input_fingerprints: dict[str, list[dict]] = {}
    found_gap = False

    for source_index, source in enumerate(sources):
        name = source["name"]
        if not budgets.get(name):
            # Excluded by both stages of the weight profile. Nothing to cache,
            # and nothing to hash: touching its files would make an unused
            # source cost build time and fail a build if it is absent.
            print(f"[{source_index:02d}] {name}: weight 0, skipped", flush=True)
            continue
        files = base.source_files(source, Path(args.remote_root))
        resolved_files[name] = [
            base.logical_source_path(path, source) for path in files
        ]
        fingerprints = source_input_fingerprints(source, files)
        input_fingerprints[name] = fingerprints
        completed = cache_root / f"{source_index:02d}_{name}"
        working = cache_root / f"{source_index:02d}_{name}.working"

        if completed.exists():
            if found_gap:
                raise ValueError(
                    f"checkpoint {completed} exists after a missing earlier "
                    "source; use a fresh --cache-dir"
                )
            validate_completed_checkpoint(
                completed,
                source,
                source_index,
                budgets[name],
                signature,
                fingerprints,
            )
            deduplicator.load_journal(completed / "dedup.bin")
            validation, rejection_counts, metadata_counts, dedup_counts = (
                load_state(completed, domain_names, args.validation_tokens)
            )
            deduplicator.counts = dedup_counts
            print(f"checkpoint reused: {name}", flush=True)
            continue

        found_gap = True
        if working.exists():
            shutil.rmtree(working)
        working.mkdir()
        journal_path = working / "dedup.bin"
        source_validation = base.ValidationCollector(
            (source["domain"],),
            source_validation_budgets[name] - 1,
        )
        with journal_path.open("wb") as journal, TokenCacheWriter(
            working / "tokens.bin"
        ) as writer:
            deduplicator.journal = journal
            documents = source_iterator(
                source,
                files,
                tokenizer,
                deduplicator,
                source_validation,
                args.validation_permille,
                rejection_counts,
                metadata_counts,
                args.max_document_chars,
                args.max_document_tokens,
                args.qa_template_fraction,
                gpt2,
                eot_id,
            )
            target = budgets[name]
            for tokens in documents:
                keep = min(tokens.size, target - writer.tokens)
                if keep:
                    writer.append(tokens[:keep])
                if writer.tokens == target:
                    break
            else:
                raise RuntimeError(
                    f"source {name!r} exhausted at {writer.tokens:,} / "
                    f"{target:,} tokens"
                )
            deduplicator.journal = None
            token_count = writer.tokens
            document_count = writer.documents

        domain = source["domain"]
        source_validation_count = sum(
            part.size for part in source_validation.tokens[domain]
        )
        if source_validation_count != source_validation_budgets[name]:
            raise RuntimeError(
                f"source {name!r} produced {source_validation_count:,} "
                f"validation tokens, requires "
                f"{source_validation_budgets[name]:,}"
            )
        validation.tokens[domain].extend(source_validation.tokens[domain])
        validation.counts[domain] += source_validation.counts[domain]
        fsync_path(journal_path)
        write_state(
            working,
            validation,
            rejection_counts,
            metadata_counts,
            deduplicator.counts,
        )
        artifact_names = {
            "tokens.bin",
            "dedup.bin",
            "state.json",
            *(f"validation_{domain}.npy" for domain in domain_names),
        }
        for artifact_name in artifact_names:
            fsync_path(working / artifact_name)
        artifacts = {
            artifact_name: (
                token_cache_stats(working / artifact_name)
                if artifact_name == "tokens.bin"
                else artifact_fingerprint(working / artifact_name)
            )
            for artifact_name in sorted(artifact_names)
        }
        checkpoint = {
            "source": name,
            "source_index": source_index,
            "source_spec_sha256": canonical_sha256(source),
            "budget": budgets[name],
            "tokens": token_count,
            "documents": document_count,
            "domains": list(domain_names),
            "signature": signature,
            "input_fingerprints": fingerprints,
            "artifacts": artifacts,
        }
        write_json_fsync(working / "checkpoint.json", checkpoint)
        fsync_directory(working)
        working.rename(completed)
        fsync_directory(cache_root)
        print(
            f"checkpoint completed: {name} "
            f"({token_count:,} tokens, {document_count:,} records)",
            flush=True,
        )

    return (
        resolved_files,
        input_fingerprints,
        validation,
        rejection_counts,
        metadata_counts,
        deduplicator,
    )


def assemble_dataset(
    *,
    args: argparse.Namespace,
    sources: list[dict],
    stages: list[tuple[str, dict[str, int]]],
    cache_root: Path,
    output_dir: Path,
) -> tuple[base.LoaderAlignedShardWriter, Counter, dict[str, Counter]]:
    # Only sources some stage draws from have a cache directory on disk; a
    # zero-weight source was skipped at cache time and has nothing to open.
    drawn = {name for _, stage in stages for name, count in stage.items() if count}
    source_iterators = {
        source["name"]: token_cache_documents(
            cache_root / f"{index:02d}_{source['name']}" / "tokens.bin"
        )
        for index, source in enumerate(sources)
        if source["name"] in drawn
    }
    writer = base.LoaderAlignedShardWriter(
        output_dir,
        args.train_batch_tokens,
        args.training_steps,
        args.steps_per_shard,
    )
    written = Counter()
    stage_written = {name: Counter() for name, _ in stages}
    for stage_name, source_name, tokens in base.staged_interleave(
        source_iterators, stages, args.on_exhausted
    ):
        writer.append(tokens, source_name)
        written[source_name] += tokens.size
        stage_written[stage_name][source_name] += tokens.size
    writer.finish()
    expected = Counter()
    for _, stage in stages:
        expected.update({k: v for k, v in stage.items() if v})
    if args.on_exhausted == "error":
        if dict(written) != dict(expected):
            raise AssertionError(
                f"written source budgets {dict(written)} != {dict(expected)}"
            )
    elif sum(written.values()) != sum(expected.values()):
        raise AssertionError(
            f"redistributed stream wrote {sum(written.values()):,} tokens "
            f"against a {sum(expected.values()):,} token budget"
        )
    return writer, written, stage_written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(base.DEFAULT_MANIFEST))
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--anneal-weights",
        help="optional second weight profile for the tail of the stream; the "
        "final --anneal-fraction of the token budget is drawn under it",
    )
    parser.add_argument("--anneal-fraction", type=float, default=0.15)
    parser.add_argument(
        "--on-exhausted",
        choices=("error", "redistribute"),
        default="error",
        help="what to do when a source runs dry before its budget: fail the "
        "build, or move the remainder onto still-active sources. An anneal "
        "stage draws from sources the bulk stage already consumed, so a "
        "profile that weights a small source heavily in the anneal wants "
        "'redistribute' (realized totals are recorded per stage in the "
        "manifest)",
    )
    parser.add_argument(
        "--tokenizer",
        default="gpt2",
        help="'gpt2', or a directory holding a trained ToaST+TST tokenizer",
    )
    parser.add_argument("--remote-root", default=str(base.DEFAULT_REMOTE_ROOT))
    parser.add_argument("--full-source-set", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--training-steps", type=int, default=20_000)
    parser.add_argument("--train-batch-tokens", type=int, default=524_288)
    parser.add_argument("--steps-per-shard", type=int, default=200)
    parser.add_argument("--validation-tokens", type=int, default=1_048_576)
    parser.add_argument("--validation-permille", type=int, default=20)
    parser.add_argument(
        "--max-document-chars", type=int, default=base.DEFAULT_CONTEXT_CHARS
    )
    parser.add_argument("--max-document-tokens", type=int, default=8192)
    parser.add_argument("--qa-template-fraction", type=float, default=0.20)
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
    if invalid := {key: value for key, value in positive.items() if value <= 0}:
        parser.error(f"positive values required: {invalid}")
    if args.max_document_tokens < 2:
        parser.error("--max-document-tokens must be at least 2")
    if not 1 <= args.validation_permille < 1000:
        parser.error("--validation-permille must be in [1, 1000)")
    if not 0 <= args.qa_template_fraction <= 1:
        parser.error("--qa-template-fraction must be in [0, 1]")

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    sources = manifest["sources"]
    if args.full_source_set:
        for source in sources:
            if "full_files" in source:
                source["files"] = source["full_files"]

    weights_path = Path(args.weights)
    profile = json.loads(weights_path.read_text())
    weights = profile["sources"]
    profile_names = set(weights)
    source_names = {source["name"] for source in sources}
    if profile_names != source_names:
        parser.error(
            f"weight profile sources differ from manifest: "
            f"{sorted(profile_names ^ source_names)}"
        )
    for source in sources:
        source["weight"] = weights[source["name"]]
    manifest["domains"] = profile["domains"]
    manifest["weight_profile"] = {
        "path": weights_path.as_posix(),
        "description": profile.get("description"),
    }
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12):
        parser.error("source weights must sum to one")

    anneal_weights = None
    anneal_path = Path(args.anneal_weights) if args.anneal_weights else None
    if anneal_path is not None:
        anneal_profile = json.loads(anneal_path.read_text())
        if set(anneal_profile["sources"]) != source_names:
            parser.error(
                "anneal profile sources differ from manifest: "
                f"{sorted(set(anneal_profile['sources']) ^ source_names)}"
            )
        if not math.isclose(
            sum(anneal_profile["sources"].values()), 1.0, abs_tol=1e-12
        ):
            parser.error("anneal profile source weights must sum to one")
        anneal_weights = dict(anneal_profile["sources"])
        manifest["anneal_weight_profile"] = {
            "path": anneal_path.as_posix(),
            "fraction": args.anneal_fraction,
            "description": anneal_profile.get("description"),
        }
    # The validation split measures the corpus as a whole, so it is drawn
    # under the blend of the two stages rather than under either one.
    share = args.anneal_fraction if anneal_weights else 0.0
    validation_sources = [
        {
            **source,
            "weight": (1 - share) * weights[source["name"]]
            + share * (anneal_weights or weights)[source["name"]],
        }
        for source in sources
    ]

    domain_weights = defaultdict(float)
    for source in sources:
        domain_weights[source["domain"]] += source["weight"]
    if set(domain_weights) != set(manifest["domains"]) or any(
        not math.isclose(
            domain_weights[domain],
            manifest["domains"][domain],
            abs_tol=1e-12,
        )
        for domain in domain_weights
    ):
        parser.error("source-derived domain weights disagree with manifest")

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
            f"margin: {undersized_validation}"
        )

    final_output_dir = Path(args.output)
    output_dir = final_output_dir.with_name(final_output_dir.name + ".building")
    if final_output_dir.exists():
        parser.error(f"{final_output_dir} already exists; use a fresh --output")

    stages = base.stage_budgets(
        unique_tokens, weights, anneal_weights, args.anneal_fraction
    )
    # Caching is per source and stage-agnostic: a source must yield whatever
    # both stages together ask of it, drawn once, in order.
    budgets = Counter()
    for _, stage in stages:
        budgets.update(stage)
    budgets = {name: count for name, count in budgets.items() if count}
    source_validation_budgets = validation_source_budgets(
        validation_sources, args.validation_tokens
    )
    if not (args.problem_registry / "registry.parquet").exists():
        parser.error(
            f"no problem registry at {args.problem_registry}; build one with "
            "scripts/build_problem_registry.py. Pretraining without it would "
            "silently repeat evaluation and post-training problems."
        )
    guard, registry = base.load_guard(
        args.problem_registry,
        split="pretrain",
        min_ngram_hits=args.min_ngram_hits,
    )
    print(
        f"problem registry: {len(guard.excluded):,} problems excluded from "
        f"pretraining, {len(guard.index):,} protected "
        f"{guard.index.ngram_size}-grams",
        flush=True,
    )
    gpt2 = GPT2BatchEncoder()
    tokenizer, tokenizer_provenance = base.load_corpus_tokenizer(
        args.tokenizer, gpt2
    )
    signature = checkpoint_signature(
        args,
        {
            "directory": str(args.problem_registry),
            "registry_sha256": registry.provenance["registry_sha256"],
            "index_sha256": guard.index.provenance["ngrams_sha256"],
        },
    )
    signature["tokenizer"] = tokenizer_signature(tokenizer, tokenizer_provenance)
    signature["validation_source_budgets"] = source_validation_budgets
    cache_root = Path(args.cache_dir)
    domain_names = tuple(manifest["domains"])
    (
        resolved_files,
        input_fingerprints,
        validation,
        rejection_counts,
        metadata_counts,
        deduplicator,
    ) = build_source_caches(
        args=args,
        sources=sources,
        budgets=budgets,
        domain_names=domain_names,
        cache_root=cache_root,
        guard=guard,
        signature=signature,
        tokenizer=tokenizer,
        gpt2=gpt2,
        eot_id=tokenizer_provenance["eot_id"],
        source_validation_budgets=source_validation_budgets,
    )

    validation_shortfalls = {
        domain: args.validation_tokens + 1 - sum(
            part.size for part in validation.tokens[domain]
        )
        for domain in domain_names
        if sum(part.size for part in validation.tokens[domain])
        < args.validation_tokens + 1
    }
    if validation_shortfalls:
        raise RuntimeError(
            f"validation checkpoints are incomplete: {validation_shortfalls}"
        )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    writer, written, stage_written = assemble_dataset(
        args=args,
        sources=sources,
        stages=stages,
        cache_root=cache_root,
        output_dir=output_dir,
    )
    validation_sizes = validation.write(output_dir)
    fineweb = next(source for source in sources if source["name"] == "fineweb")
    fineweb_val_shards = sorted(
        Path(fineweb["path"]).glob("fineweb_val_*.bin")
    )
    if not fineweb_val_shards:
        raise FileNotFoundError(
            f"no FineWeb validation shards under {fineweb['path']}"
        )
    challenge_validation_shards, challenge_validation = (
        base.materialize_challenge_validation(
            fineweb_val_shards,
            output_dir,
            gpt2,
            tokenizer,
            tokenizer_provenance["eot_id"],
        )
    )
    effective_manifest_path = output_dir / "source_manifest.json"
    effective_manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    document_rejections, item_rejections = base.partition_rejection_counts(
        rejection_counts
    )
    result = {
        "version": 6,
        "builder": str(Path(__file__).resolve()),
        "builder_git_revision": base.git_revision(),
        "builder_sha256": base.sha256_file(Path(__file__)),
        "base_builder_sha256": base.sha256_file(Path(base.__file__)),
        "decontaminate_sha256": base.sha256_file(Path(decontaminate.__file__)),
        "problem_registry_sha256": base.sha256_file(Path(problem_registry.__file__)),
        "source_manifest_sha256": base.sha256_file(effective_manifest_path),
        "source_manifest_input_sha256": base.sha256_file(manifest_path),
        "weight_profile_sha256": base.sha256_file(weights_path),
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
                "source_token_budgets": {k: v for k, v in stage.items() if v},
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
        "validation_source_token_budgets": source_validation_budgets,
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
        "checkpointing": {
            "format_version": CACHE_FORMAT_VERSION,
            "cache_dir": cache_root.as_posix(),
            "granularity": "completed source",
            "completed_sources": [source["name"] for source in sources],
        },
    }
    (output_dir / "mix_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    output_dir.rename(final_output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
