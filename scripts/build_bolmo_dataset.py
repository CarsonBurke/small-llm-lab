#!/usr/bin/env python3
"""Stream ToaST+TST challenge shards into paper-aligned Bolmo byte data.

This is CPU preprocessing, but project policy still requires invoking it via
``mlq``.  It never materializes a corpus: source shards are memory-mapped,
examples are formed at a fixed source-token width, and only a bounded number
of examples is collated before each ``.pt`` artifact is written.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pretraining.bolmo_data import (
    BOLMO_DATA_SCHEMA,
    BYTE_VOCAB_SIZE,
    BolmoExample,
    ByteifiedTokens,
    ExpandedSuffixMatcher,
    SourceTokenByteifier,
    atomic_pad_id,
    atomic_special_id,
    make_bolmo_example,
    save_example_shard,
    truncate_to_complete_patches,
    validate_source_tokenizer,
)
from tokenization.tokenizer import SplitTreeNumericTokenizer


CHALLENGE_MAGIC = 20240520
CHALLENGE_VERSION = 1
CHALLENGE_HEADER_INTS = 256


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def fingerprint_file(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def expand_patterns(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = sorted(Path(path) for path in glob.glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        if not matches:
            raise FileNotFoundError(f"input pattern matched no shards: {pattern}")
        for path in matches:
            resolved = path.resolve()
            if resolved not in seen:
                paths.append(path)
                seen.add(resolved)
    return paths


def read_challenge_shard(path: Path) -> np.memmap:
    header = np.fromfile(path, dtype="<i4", count=CHALLENGE_HEADER_INTS)
    if header.size != CHALLENGE_HEADER_INTS:
        raise ValueError(f"truncated challenge shard header: {path}")
    if tuple(int(value) for value in header[:2]) != (
        CHALLENGE_MAGIC,
        CHALLENGE_VERSION,
    ):
        raise ValueError(f"invalid challenge shard header: {path}")
    token_count = int(header[2])
    if token_count < 0:
        raise ValueError(f"negative token count in {path}")
    expected = CHALLENGE_HEADER_INTS * 4 + token_count * 2
    if path.stat().st_size != expected:
        raise ValueError(
            f"challenge shard size mismatch for {path}: "
            f"expected {expected}, got {path.stat().st_size}"
        )
    return np.memmap(
        path,
        mode="r",
        dtype="<u2",
        offset=CHALLENGE_HEADER_INTS * 4,
        shape=(token_count,),
    )


def iter_source_examples(
    paths: Sequence[Path],
    source_tokens_per_example: int,
    *,
    overlap_tokens: int = 0,
    include_tail: bool = True,
) -> Iterator[np.ndarray]:
    """Yield fixed source-token windows without loading any complete shard."""

    if source_tokens_per_example <= 0:
        raise ValueError("source_tokens_per_example must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative")
    pending = np.empty(source_tokens_per_example, dtype=np.int32)
    filled = 0
    previous_tail: np.ndarray | None = None
    for shard_index, path in enumerate(paths):
        tokens = read_challenge_shard(path)
        start = 0
        if shard_index and overlap_tokens:
            if len(tokens) < overlap_tokens or previous_tail is None:
                raise ValueError(f"shard is too short for overlap check: {path}")
            observed = np.asarray(tokens[:overlap_tokens], dtype=np.int32)
            if not np.array_equal(observed, previous_tail):
                raise ValueError(
                    f"declared {overlap_tokens}-token overlap does not match "
                    f"at {path}"
                )
            start = overlap_tokens
        previous_tail = (
            np.asarray(tokens[-overlap_tokens:], dtype=np.int32).copy()
            if overlap_tokens and len(tokens) >= overlap_tokens
            else None
        )
        while start < len(tokens):
            take = min(source_tokens_per_example - filled, len(tokens) - start)
            pending[filled : filled + take] = tokens[start : start + take]
            filled += take
            start += take
            if filled == source_tokens_per_example:
                yield pending.copy()
                filled = 0
    if filled and include_tail:
        yield pending[:filled].copy()


def iter_context_target_examples(
    paths: Sequence[Path],
    target_tokens_per_example: int,
) -> Iterator[np.ndarray]:
    """Yield source-validation windows with one left-context token.

    This exactly mirrors the source trainer's contiguous ``inputs=buf[:-1]``
    and ``targets=buf[1:]`` windows: adjacent examples share the token at the
    input/target boundary, while each row scores ``target_tokens_per_example``
    tokens. Validation shards are independent streams, so partial tails do not
    cross shard boundaries.
    """

    if target_tokens_per_example <= 0:
        raise ValueError("target_tokens_per_example must be positive")
    width = target_tokens_per_example + 1
    for path in paths:
        tokens = read_challenge_shard(path)
        for start in range(0, len(tokens) - width + 1, target_tokens_per_example):
            yield np.asarray(tokens[start : start + width], dtype=np.int32)


def discover_source_manifest(paths: Sequence[Path]) -> tuple[Path, dict] | None:
    parents = {path.resolve().parent for path in paths}
    if len(parents) != 1:
        return None
    candidate = next(iter(parents)) / "mix_manifest.json"
    if not candidate.is_file():
        return None
    return candidate, json.loads(candidate.read_text())


def validate_manifest_tokenizer(
    discovered: tuple[Path, dict] | None,
    tokenizer_dir: Path,
    tokenizer: SplitTreeNumericTokenizer,
) -> dict | None:
    if discovered is None:
        return None
    path, manifest = discovered
    provenance = manifest.get("tokenizer_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{path} does not bind its source tokenizer")
    expected = {
        "kind": "toast_tst",
        "vocab_size": tokenizer.vocab_size,
        "eot_id": tokenizer.eot_id,
        "spec_sha256": sha256_file(tokenizer_dir / "tokenizer.json"),
        "ngrams_sha256": tokenizer.spec.ngrams.sha256,
    }
    mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"source manifest {path} was built under another tokenizer: "
            f"{mismatches}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "payload_sha256": canonical_sha256(manifest),
    }


def infer_train_overlap(
    discovered: tuple[Path, dict] | None,
    train_paths: Sequence[Path],
) -> int:
    if discovered is None:
        return 0
    _, manifest = discovered
    if not manifest.get("loader_aligned") or len(train_paths) <= 1:
        return 0
    physical = manifest.get("physical_shard_tokens")
    unique = manifest.get("unique_stream_tokens")
    if not isinstance(physical, int) or not isinstance(unique, int):
        raise ValueError("loader-aligned manifest lacks physical/unique counts")
    duplicated = physical - unique
    declared_shards = manifest.get("shards")
    declared_count = (
        len(declared_shards)
        if isinstance(declared_shards, list) and len(declared_shards) > 1
        else len(train_paths)
    )
    transitions = declared_count - 1
    if duplicated < 0 or duplicated % transitions:
        raise ValueError(
            "cannot infer a uniform source-shard overlap from manifest: "
            f"duplicated={duplicated}, transitions={transitions}"
        )
    return duplicated // transitions


@dataclass(frozen=True)
class SplitBuild:
    name: str
    paths: tuple[Path, ...]
    overlap_tokens: int = 0
    max_examples: int | None = None
    context_source_tokens: int = 0
    skip_last_source_token: bool = False


def _flush_examples(
    *,
    output_dir: Path,
    split: str,
    shard_index: int,
    examples: list[BolmoExample],
    tokenizer: SplitTreeNumericTokenizer,
    source_model_vocab_size: int,
    byte_length_multiple: int,
) -> dict:
    filename = f"{split}-{shard_index:05d}.pt"
    path = output_dir / filename
    stats = save_example_shard(
        path,
        examples,
        tokenizer,
        source_model_vocab_size=source_model_vocab_size,
        byte_length_multiple=byte_length_multiple,
    )
    return {
        "path": filename,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **stats,
    }


def build_split(
    split: SplitBuild,
    *,
    output_dir: Path,
    tokenizer: SplitTreeNumericTokenizer,
    source_tokens_per_example: int,
    examples_per_shard: int,
    include_tail: bool,
    source_model_vocab_size: int,
    byte_length_multiple: int,
    max_atomic_tokens: int | None,
) -> dict:
    if split.max_examples is not None and split.max_examples <= 0:
        raise ValueError(f"split {split.name!r} max_examples must be positive")
    byteifier = SourceTokenByteifier(tokenizer)
    suffix_matcher = ExpandedSuffixMatcher(
        tokenizer, source_model_vocab_size=source_model_vocab_size
    )
    buffered: list[BolmoExample] = []
    artifacts: list[dict] = []
    real_source_tokens = 0
    input_source_tokens = 0
    paper_skipped_source_tokens = 0
    scored_source_tokens = 0
    atomic_tokens = 0
    examples_built = 0
    byte_truncated_examples = 0
    byte_truncated_source_tokens = 0
    truncated = False
    source_examples = (
        iter_context_target_examples(split.paths, source_tokens_per_example)
        if split.context_source_tokens == 1
        else iter_source_examples(
            split.paths,
            source_tokens_per_example,
            overlap_tokens=split.overlap_tokens,
            include_tail=include_tail,
        )
    )
    for real_ids in source_examples:
        if split.max_examples is not None and examples_built == split.max_examples:
            truncated = True
            break
        # Each source row is an independent teacher sequence beginning at the
        # synthetic EOT/BOS. Resetting is essential for semantic TST numeric
        # tokens: the byte row must decode exactly like this source row, not
        # like an invisible continuation of the prior row.
        byteifier.reset()
        byteified = byteifier.byteify(real_ids)
        input_source_tokens += len(real_ids)
        retained_ids: Sequence[int] = real_ids
        if split.skip_last_source_token:
            if len(real_ids) < 2:
                continue
            retained_ids = real_ids[:-1]
            retained_atoms = sum(byteified.patch_lengths[:-1])
            byteified = ByteifiedTokens(
                atomic_ids=byteified.atomic_ids[:retained_atoms],
                patch_lengths=byteified.patch_lengths[:-1],
            )
            paper_skipped_source_tokens += 1
        if max_atomic_tokens is not None:
            retained_ids, retained_byteified = truncate_to_complete_patches(
                retained_ids,
                byteified,
                max_atomic_tokens=max_atomic_tokens,
            )
            paper_dropped = int(split.skip_last_source_token)
            dropped = len(real_ids) - len(retained_ids) - paper_dropped
            if dropped:
                byte_truncated_examples += 1
                byte_truncated_source_tokens += dropped
            byteified = retained_byteified
        example = make_bolmo_example(
            retained_ids,
            byteified,
            tokenizer,
            suffix_matcher,
            expected_real_source_tokens=(
                source_tokens_per_example
                + split.context_source_tokens
                - int(split.skip_last_source_token)
            ),
            source_model_vocab_size=source_model_vocab_size,
            context_source_tokens=split.context_source_tokens,
        )
        buffered.append(example)
        real_source_tokens += len(retained_ids)
        scored_source_tokens += max(
            0, len(retained_ids) - split.context_source_tokens
        )
        atomic_tokens += len(example.byte_ids)
        examples_built += 1
        if len(buffered) == examples_per_shard:
            artifacts.append(
                _flush_examples(
                    output_dir=output_dir,
                    split=split.name,
                    shard_index=len(artifacts),
                    examples=buffered,
                    tokenizer=tokenizer,
                    source_model_vocab_size=source_model_vocab_size,
                    byte_length_multiple=byte_length_multiple,
                )
            )
            buffered = []
    if buffered:
        artifacts.append(
            _flush_examples(
                output_dir=output_dir,
                split=split.name,
                shard_index=len(artifacts),
                examples=buffered,
                tokenizer=tokenizer,
                source_model_vocab_size=source_model_vocab_size,
                byte_length_multiple=byte_length_multiple,
            )
        )
    if not artifacts:
        raise ValueError(f"split {split.name!r} produced no examples")
    return {
        "input_shards": [str(path) for path in split.paths],
        "overlap_tokens_per_transition": split.overlap_tokens,
        "context_source_tokens": split.context_source_tokens,
        "skip_last_source_token": split.skip_last_source_token,
        "max_examples": split.max_examples,
        "truncated_by_max_examples": truncated,
        "examples": sum(int(item["examples"]) for item in artifacts),
        "input_source_tokens": input_source_tokens,
        "real_source_tokens": real_source_tokens,
        "paper_skipped_source_tokens": paper_skipped_source_tokens,
        "scored_source_tokens": scored_source_tokens,
        "atomic_tokens_including_prepended_eot": atomic_tokens,
        "byte_truncated_examples": byte_truncated_examples,
        "byte_truncated_source_tokens": byte_truncated_source_tokens,
        "artifacts": artifacts,
    }


def build_dataset(
    *,
    tokenizer_dir: Path,
    output_dir: Path,
    train_paths: Sequence[Path],
    validation_paths: Sequence[Path],
    domain_validation_paths: dict[str, Sequence[Path]],
    source_tokens_per_example: int,
    examples_per_shard: int,
    include_tail: bool = True,
    train_overlap_tokens: int | None = None,
    source_model_vocab_size: int | None = None,
    byte_length_multiple: int = 128,
    max_atomic_tokens: int | None = None,
    max_train_examples: int | None = None,
    max_validation_examples: int | None = 256,
    max_domain_validation_examples: int | None = 256,
    command: Sequence[str] | None = None,
) -> dict:
    """Build all splits and return the exact manifest written to disk."""

    if examples_per_shard <= 0:
        raise ValueError("examples_per_shard must be positive")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = Path(tokenizer_dir)
    tokenizer = SplitTreeNumericTokenizer.from_directory(tokenizer_dir)
    validate_source_tokenizer(tokenizer)
    if source_model_vocab_size is None:
        source_model_vocab_size = (tokenizer.vocab_size + 127) // 128 * 128
    if source_model_vocab_size < tokenizer.vocab_size:
        raise ValueError(
            "source model vocabulary cannot be smaller than logical tokenizer "
            f"vocabulary ({source_model_vocab_size} < {tokenizer.vocab_size})"
        )
    if byte_length_multiple <= 0:
        raise ValueError("byte_length_multiple must be positive")
    if max_atomic_tokens is not None:
        if max_atomic_tokens <= 1:
            raise ValueError("max_atomic_tokens must leave room after BOS")
        if max_atomic_tokens % byte_length_multiple:
            raise ValueError(
                "max_atomic_tokens must be divisible by byte_length_multiple"
            )
    train_paths = tuple(Path(path) for path in train_paths)
    validation_paths = tuple(Path(path) for path in validation_paths)
    if not train_paths or not validation_paths:
        raise ValueError("both train and validation source shards are required")
    all_paths = (*train_paths, *validation_paths)
    for values in domain_validation_paths.values():
        all_paths += tuple(Path(path) for path in values)
    unique_input_paths = tuple(dict.fromkeys(all_paths))

    discovered = discover_source_manifest(train_paths)
    if discovered is None:
        raise ValueError(
            "training shards must share a parent with a tokenizer-bound "
            "mix_manifest.json"
        )
    source_discoveries: list[tuple[Path, dict]] = []
    for parent in sorted({path.resolve().parent for path in unique_input_paths}):
        manifest_path = parent / "mix_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                f"source shard parent {parent} has no tokenizer-bound mix_manifest.json"
            )
        source_discoveries.append(
            (manifest_path, json.loads(manifest_path.read_text()))
        )

    implementation_paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1] / "pretraining" / "bolmo_data.py",
        tokenizer_dir / "tokenizer.json",
        tokenizer_dir / tokenizer.spec.ngrams.filename,
        *(item[0] for item in source_discoveries),
    )
    consumed_before = {
        str(path): fingerprint_file(path)
        for path in (*implementation_paths, *unique_input_paths)
    }
    source_manifests = [
        validate_manifest_tokenizer(item, tokenizer_dir, tokenizer)
        for item in source_discoveries
    ]
    overlap = (
        infer_train_overlap(discovered, train_paths)
        if train_overlap_tokens is None
        else train_overlap_tokens
    )

    splits = [
        SplitBuild("train", train_paths, overlap, max_train_examples, 0, True),
        SplitBuild(
            "validation", validation_paths, 0, max_validation_examples, 1
        ),
        *(
            SplitBuild(
                f"domainval_{name}",
                tuple(paths),
                0,
                max_domain_validation_examples,
                1,
            )
            for name, paths in sorted(domain_validation_paths.items())
        ),
    ]
    split_results = {
        split.name: build_split(
            split,
            output_dir=output_dir,
            tokenizer=tokenizer,
            source_tokens_per_example=source_tokens_per_example,
            examples_per_shard=examples_per_shard,
            include_tail=include_tail,
            source_model_vocab_size=source_model_vocab_size,
            byte_length_multiple=byte_length_multiple,
            max_atomic_tokens=max_atomic_tokens,
        )
        for split in splits
    }
    consumed_after = {
        str(path): fingerprint_file(path)
        for path in (*implementation_paths, *unique_input_paths)
    }
    if consumed_after != consumed_before:
        changed = [
            path
            for path in consumed_before
            if consumed_before[path] != consumed_after.get(path)
        ]
        raise RuntimeError(
            "a consumed input changed while the dataset was being built; "
            f"refusing to publish mixed provenance: {changed}"
        )
    input_fingerprints = [consumed_before[str(path)] for path in unique_input_paths]
    spec_path = tokenizer_dir / "tokenizer.json"
    ngrams_path = tokenizer_dir / tokenizer.spec.ngrams.filename
    manifest = {
        "schema": BOLMO_DATA_SCHEMA,
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": consumed_before[str(Path(__file__).resolve())]["sha256"],
        "library_sha256": consumed_before[
            str(Path(__file__).resolve().parents[1] / "pretraining" / "bolmo_data.py")
        ]["sha256"],
        "command": list(command) if command is not None else None,
        "source_tokenizer": {
            "directory": str(tokenizer_dir),
            "kind": "toast_tst",
            "logical_vocab_size": tokenizer.vocab_size,
            "source_model_vocab_size": source_model_vocab_size,
            "eot_id": tokenizer.eot_id,
            "source_pad_id": source_model_vocab_size,
            "spec_file_sha256": consumed_before[str(spec_path)]["sha256"],
            "spec_canonical_sha256": tokenizer.spec.sha256(),
            "ngrams_sha256": consumed_before[str(ngrams_path)]["sha256"],
            "tst_group_size": tokenizer.scheme.group_size,
            "tst_compound": tokenizer.scheme.compound,
        },
        "atomic_vocabulary": {
            "byte_ids": [0, BYTE_VOCAB_SIZE - 1],
            "special_base": BYTE_VOCAB_SIZE,
            "specials": {
                value: atomic_special_id(index)
                for index, value in enumerate(tokenizer.spec.specials)
            },
            "eot_id": atomic_special_id(tokenizer.eot_id),
            "pad_id": atomic_pad_id(tokenizer),
            "vocab_size_including_pad": atomic_pad_id(tokenizer) + 1,
        },
        "examples": {
            "source_sequence_length": source_tokens_per_example,
            "training_real_source_tokens": source_tokens_per_example - 1,
            "training_stored_source_width": source_tokens_per_example,
            "validation_stored_source_width": source_tokens_per_example + 2,
            "prepended_source_id": tokenizer.eot_id,
            "prepended_atomic_id": atomic_special_id(tokenizer.eot_id),
            "examples_per_artifact": examples_per_shard,
            "byte_length_multiple": byte_length_multiple,
            "max_atomic_tokens": max_atomic_tokens,
            "max_train_examples": max_train_examples,
            "max_validation_examples": max_validation_examples,
            "max_domain_validation_examples": max_domain_validation_examples,
            "tail_examples_included_and_source_padded": include_tail,
            "byte_padding": "local maximum within each bounded artifact",
            "expanded_embedding": (
                "longest causal fixed-surface text token ending at each byte; "
                "atomic specials map to their source ids; TST numeric semantic "
                "tokens are excluded"
            ),
        },
        "tensor_fields": {
            "source_ids": "int32 [N,S] train; int32 [N,S+2] validation",
            "source_valid_mask": "bool with the split's source_ids shape",
            "byte_ids": "int16 [N,L_local]",
            "expanded_ids": "int32 [N,L_local]",
            "boundary_mask": "bool [N,L_local]",
            "valid_mask": "bool [N,L_local]",
            "score_mask": "bool [N,L_local]",
            "patch_lens": "int32 with the split's source_ids shape",
        },
        "source_manifests": source_manifests,
        "input_fingerprints": input_fingerprints,
        "splits": split_results,
    }
    manifest["payload_sha256"] = canonical_sha256(manifest)
    temporary = output_dir / "manifest.json.working"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_dir / "manifest.json")
    return manifest


def parse_domain_values(values: Sequence[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"domain validation must be NAME=PATTERN, got {value!r}"
            )
        name, pattern = value.split("=", 1)
        if not name or not pattern:
            raise ValueError(
                f"domain validation must be NAME=PATTERN, got {value!r}"
            )
        result.setdefault(name, []).append(pattern)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", action="append", required=True, metavar="GLOB")
    parser.add_argument(
        "--validation", action="append", required=True, metavar="GLOB"
    )
    parser.add_argument(
        "--domain-validation",
        action="append",
        default=[],
        metavar="NAME=GLOB",
    )
    parser.add_argument("--source-tokens", type=int, default=2048)
    parser.add_argument("--examples-per-shard", type=int, default=256)
    parser.add_argument(
        "--source-model-vocab-size",
        type=int,
        default=None,
        help="checkpoint embedding vocabulary; default: logical vocab padded to 128",
    )
    parser.add_argument(
        "--byte-length-multiple",
        type=int,
        default=128,
        help="pad each artifact's byte axis for xLSTM/TFLA chunkwise kernels",
    )
    parser.add_argument(
        "--max-atomic-tokens",
        type=int,
        default=12288,
        help=(
            "maximum byte/special atoms including prepended BOS; truncate only "
            "at complete source-patch boundaries"
        ),
    )
    parser.add_argument(
        "--train-overlap-tokens",
        type=int,
        default=None,
        help="default: infer loader-aligned overlap from mix_manifest.json",
    )
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
        help="stop after this many train examples (default: consume train stream)",
    )
    parser.add_argument(
        "--max-validation-examples",
        type=int,
        default=1024,
        help="source-matched canonical validation examples (default: 1024)",
    )
    parser.add_argument(
        "--max-domain-validation-examples",
        type=int,
        default=256,
        help="bound each domain-validation split separately (default: 256)",
    )
    parser.add_argument(
        "--drop-tail",
        action="store_true",
        help="drop each split's final partial source-token example",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    domain_patterns = parse_domain_values(args.domain_validation)
    manifest = build_dataset(
        tokenizer_dir=args.tokenizer,
        output_dir=args.output,
        train_paths=expand_patterns(args.train),
        validation_paths=expand_patterns(args.validation),
        domain_validation_paths={
            name: expand_patterns(patterns)
            for name, patterns in domain_patterns.items()
        },
        source_tokens_per_example=args.source_tokens,
        examples_per_shard=args.examples_per_shard,
        include_tail=not args.drop_tail,
        train_overlap_tokens=args.train_overlap_tokens,
        source_model_vocab_size=args.source_model_vocab_size,
        byte_length_multiple=args.byte_length_multiple,
        max_atomic_tokens=args.max_atomic_tokens,
        max_train_examples=args.max_train_examples,
        max_validation_examples=args.max_validation_examples,
        max_domain_validation_examples=args.max_domain_validation_examples,
        command=sys.argv if argv is None else [str(Path(__file__)), *argv],
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_sha256": manifest["payload_sha256"],
                "splits": {
                    name: {
                        "examples": details["examples"],
                        "source_tokens": details["real_source_tokens"],
                        "atomic_tokens": details[
                            "atomic_tokens_including_prepended_eot"
                        ],
                    }
                    for name, details in manifest["splits"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
