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

import build_k3_pretrain_dataset as base
from build_math_mix_dataset import GPT2BatchEncoder, allocate_token_budgets

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
        heldout_keys: set[bytes],
        journal: BinaryIO | None = None,
    ):
        super().__init__(heldout_keys)
        self.journal = journal

    def accept(self, document: base.RawDocument) -> bool:
        accepted = super().accept(document)
        if not accepted or self.journal is None:
            return accepted
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
        return accepted

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
    tokenizer: GPT2BatchEncoder,
    deduplicator: JournaledDeduplicator,
    validation: base.ValidationCollector,
    validation_permille: int,
    rejection_counts: Counter,
    metadata_counts: dict[str, Counter],
    max_document_chars: int,
    max_document_tokens: int,
    qa_template_fraction: float,
) -> Iterator[np.ndarray]:
    kind = source["kind"]
    if kind == "token_shards":
        return base.filter_token_documents(
            base.token_shard_documents(files[0]),
            source,
            tokenizer,
            deduplicator,
            validation,
            validation_permille,
            rejection_counts,
            max_document_tokens,
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
    )


def checkpoint_signature(args: argparse.Namespace, heldout: list[dict]) -> dict:
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "checkpoint_builder_sha256": base.sha256_file(Path(__file__)),
        "base_builder_sha256": base.sha256_file(Path(base.__file__)),
        "validation_tokens": args.validation_tokens,
        "validation_permille": args.validation_permille,
        "max_document_chars": args.max_document_chars,
        "max_document_tokens": args.max_document_tokens,
        "qa_template_fraction": args.qa_template_fraction,
        "heldout": heldout,
    }


def tokenizer_signature(tokenizer: GPT2BatchEncoder) -> dict:
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
    heldout_keys: set[bytes],
    signature: dict,
    tokenizer: GPT2BatchEncoder,
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
    deduplicator = JournaledDeduplicator(heldout_keys)
    resolved_files: dict[str, list[str]] = {}
    input_fingerprints: dict[str, list[dict]] = {}
    found_gap = False

    for source_index, source in enumerate(sources):
        name = source["name"]
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
    budgets: dict[str, int],
    cache_root: Path,
    output_dir: Path,
) -> tuple[base.LoaderAlignedShardWriter, Counter]:
    source_iterators = {
        source["name"]: token_cache_documents(
            cache_root / f"{index:02d}_{source['name']}" / "tokens.bin"
        )
        for index, source in enumerate(sources)
    }
    writer = base.LoaderAlignedShardWriter(
        output_dir,
        args.train_batch_tokens,
        args.training_steps,
        args.steps_per_shard,
    )
    written = Counter()
    for source_name, tokens in base.logical_interleave(
        source_iterators, budgets
    ):
        writer.append(tokens, source_name)
        written[source_name] += tokens.size
    writer.finish()
    if dict(written) != budgets:
        raise AssertionError(f"written source budgets {dict(written)} != {budgets}")
    return writer, written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(base.DEFAULT_MANIFEST))
    parser.add_argument("--weights", required=True)
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

    budgets = allocate_token_budgets(unique_tokens, weights)
    source_validation_budgets = validation_source_budgets(
        sources, args.validation_tokens
    )
    heldout_paths = [Path(path) for path in args.heldout]
    heldout_fingerprints = [
        {
            "path": path.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": base.sha256_file(path),
        }
        for path in heldout_paths
    ]
    heldout_keys = base.heldout_problem_keys(heldout_paths)
    tokenizer = GPT2BatchEncoder()
    signature = checkpoint_signature(args, heldout_fingerprints)
    signature["tokenizer"] = tokenizer_signature(tokenizer)
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
        heldout_keys=heldout_keys,
        signature=signature,
        tokenizer=tokenizer,
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
    writer, written = assemble_dataset(
        args=args,
        sources=sources,
        budgets=budgets,
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
    for path in fineweb_val_shards:
        shutil.copy2(path, output_dir / path.name)
    effective_manifest_path = output_dir / "source_manifest.json"
    effective_manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    result = {
        "version": 6,
        "builder": str(Path(__file__).resolve()),
        "builder_git_revision": base.git_revision(),
        "builder_sha256": base.sha256_file(Path(__file__)),
        "base_builder_sha256": base.sha256_file(Path(base.__file__)),
        "source_manifest_sha256": base.sha256_file(effective_manifest_path),
        "source_manifest_input_sha256": base.sha256_file(manifest_path),
        "weight_profile_sha256": base.sha256_file(weights_path),
        "command": sys.argv,
        "source_manifest": manifest_path.as_posix(),
        "resolved_files": resolved_files,
        "input_fingerprints": input_fingerprints,
        "tokenizer": "gpt2",
        "source_set": "full" if args.full_source_set else "sampled",
        "bos_id": base.GPT2_EOT_ID,
        "eos_id": base.GPT2_EOT_ID,
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
        "validation_source_token_budgets": source_validation_budgets,
        "validation_split": f"stable hash < {args.validation_permille}/1000",
        "max_document_chars": args.max_document_chars,
        "max_document_tokens": args.max_document_tokens,
        "qa_template_fraction": args.qa_template_fraction,
        "heldout_problem_keys": len(heldout_keys),
        "deduplication": {
            "exact_normalized": True,
            "formatting_insensitive": True,
            "counts": dict(deduplicator.counts),
        },
        "rejection_counts": dict(rejection_counts),
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
