#!/usr/bin/env python3
"""Materialize a GPT-2 view of an already-selected byte-native corpus.

The byte corpus remains the document-selection ledger. This script changes
only its representation, so AR and byte-diffusion comparisons see identical
complete documents and raw UTF-8 bytes. It never independently fills a GPT-2
token budget, which would select a different corpus.

This is corpus preprocessing and should be submitted through ``mlq``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from pretraining.byte_diffusion.data import AtomicIdManifest
from scripts.build_bolmo_dataset import canonical_sha256, sha256_file
from scripts.build_byte_diffusion_dataset import ChallengeDocumentReader
from scripts.build_k3_pretrain_dataset import StreamingTokenShardWriter
from scripts.build_k3_pretrain_dataset_checkpointed import tokenizer_signature
from scripts.build_math_mix_dataset import GPT2BatchEncoder, GPT2_EOT_ID


def _fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _overlap(source_manifest: dict[str, Any], paths: Sequence[Path]) -> int:
    if len(paths) <= 1 or not source_manifest.get("loader_aligned"):
        return 0
    physical = source_manifest.get("physical_shard_tokens")
    unique = source_manifest.get("unique_stream_tokens")
    if not isinstance(physical, int) or not isinstance(unique, int):
        raise ValueError("source manifest lacks physical/unique stream sizes")
    duplicated = physical - unique
    transitions = len(paths) - 1
    if duplicated < 0 or duplicated % transitions:
        raise ValueError("source manifest does not declare uniform shard overlap")
    return duplicated // transitions


def materialize_split(
    *,
    name: str,
    source_paths: Sequence[Path],
    output_path: Path,
    source_eot_id: int,
    source_overlap: int,
    target_encoder: Any,
    target_eot_id: int,
    batch_size: int = 1_024,
) -> dict[str, Any]:
    """Re-encode identical complete documents and record the dropped tail."""

    reader = ChallengeDocumentReader(
        source_paths,
        eot_id=source_eot_id,
        overlap_tokens=source_overlap,
    )
    pending: list[str] = []
    documents = 0
    literal_bytes = 0
    with StreamingTokenShardWriter(output_path) as writer:
        writer.append(np.asarray([target_eot_id], dtype=np.int32))

        def flush() -> None:
            nonlocal documents
            encoded = target_encoder.encode(pending, out_type=int)
            for ids in encoded:
                writer.append(np.asarray([*ids, target_eot_id], dtype=np.int32))
            documents += len(pending)
            pending.clear()

        for atomic_document in reader.iter_documents():
            if atomic_document == (source_eot_id,):
                continue
            raw = bytes(atomic_document[:-1])
            text = raw.decode("utf-8", errors="strict")
            literal_bytes += len(raw)
            pending.append(text)
            if len(pending) == batch_size:
                flush()
        if pending:
            flush()
        target_tokens = writer.tokens

    return {
        "name": name,
        "input_shards": [str(path) for path in source_paths],
        "output_shard": str(output_path),
        "documents": documents,
        "literal_utf8_bytes": literal_bytes,
        "target_tokens": target_tokens,
        "source_physical_tokens": reader.stats.physical_tokens,
        "source_unique_tokens": reader.stats.unique_tokens,
        "source_empty_boundaries": reader.stats.empty_documents,
        "source_incomplete_tail_tokens": reader.stats.incomplete_tail_tokens,
    }


def materialize_view(source: Path, output: Path) -> dict[str, Any]:
    source = Path(source)
    output = Path(output)
    source_manifest_path = source / "mix_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"missing source manifest: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text())
    atomic_manifest = AtomicIdManifest.reference()
    provenance = source_manifest.get("tokenizer_provenance", {})
    expected = {
        "kind": "utf8_bytes",
        "vocab_size": atomic_manifest.output_size,
        "eot_id": atomic_manifest.eot_id,
        "spec_sha256": atomic_manifest.sha256,
    }
    mismatch = {
        key: (provenance.get(key), value)
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatch:
        raise ValueError(f"source is not the canonical byte encoding: {mismatch}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    train_paths = tuple(sorted(source.glob("fineweb_train_*.bin")))
    validation_paths = tuple(sorted(source.glob("fineweb_val_*.bin")))
    if not train_paths or not validation_paths:
        raise ValueError("source needs train and challenge-validation shards")

    encoder = GPT2BatchEncoder()
    target_provenance = {
        "kind": "gpt2",
        "name": "gpt2",
        "vocab_size": 50_257,
        "eot_id": GPT2_EOT_ID,
        "signature": tokenizer_signature(
            encoder,
            {"kind": "gpt2"},
        ),
    }
    source_inputs = (source_manifest_path, *train_paths, *validation_paths)
    before = {str(path): _fingerprint(path) for path in source_inputs}
    train = materialize_split(
        name="train",
        source_paths=train_paths,
        output_path=output / "fineweb_train_000000.bin",
        source_eot_id=atomic_manifest.eot_id,
        source_overlap=_overlap(source_manifest, train_paths),
        target_encoder=encoder,
        target_eot_id=GPT2_EOT_ID,
    )
    validation = materialize_split(
        name="validation",
        source_paths=validation_paths,
        output_path=output / "fineweb_val_000000.bin",
        source_eot_id=atomic_manifest.eot_id,
        source_overlap=0,
        target_encoder=encoder,
        target_eot_id=GPT2_EOT_ID,
    )
    after = {str(path): _fingerprint(path) for path in source_inputs}
    if before != after:
        raise RuntimeError("a source artifact changed during view materialization")

    manifest = {
        "schema": "byte_corpus_token_view/v1",
        "view": "gpt2",
        "selection_unit": "utf8_bytes",
        "document_selection_changed": False,
        "source_manifest": _fingerprint(source_manifest_path),
        "source_manifest_payload_sha256": canonical_sha256(source_manifest),
        "tokenizer_provenance": target_provenance,
        "input_fingerprints": list(before.values()),
        "splits": {"train": train, "validation": validation},
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    manifest["payload_sha256"] = canonical_sha256(manifest)
    temporary = output / "mix_manifest.json.working"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / "mix_manifest.json")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = materialize_view(args.source, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
