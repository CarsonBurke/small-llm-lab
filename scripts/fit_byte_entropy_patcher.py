#!/usr/bin/env python3
"""Fit and serialize the compact causal entropy patcher on training bytes.

This is corpus preprocessing and must run through ``mlq``.  Input shards are
the canonical byte-native selection ledger; batches end only at EOT so the
estimator never invents a document reset at an implementation chunk boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterator

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from pretraining.byte_diffusion.patching import (
    CausalEntropyPatcher,
    EntropyPatchConfig,
    HashedNgramEntropyConfig,
    HashedNgramEntropyModel,
    document_ids_from_eot,
    entropy_patch_start_mask,
)
from scripts.build_bolmo_dataset import (
    discover_source_manifest,
    infer_train_overlap,
    read_challenge_shard,
)
from scripts.build_byte_diffusion_dataset import ChallengeDocumentReader


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def logical_training_chunks(
    paths: tuple[Path, ...],
    *,
    overlap_atoms: int,
    eot_id: int,
    target_atoms: int,
    max_document_atoms: int,
) -> Iterator[np.ndarray]:
    """Yield bounded int64 batches of complete documents from challenge shards."""

    if (
        not paths
        or overlap_atoms < 0
        or target_atoms <= 0
        or max_document_atoms <= 0
    ):
        raise ValueError("training shard geometry is invalid")
    # Reuse the dataset reader's exact overlap-chain authentication, but scan
    # payload windows here. Its document-batch iterator first locates every EOT
    # in a complete ~419M-atom shard, defeating this fitter's batch memory cap.
    ChallengeDocumentReader(
        paths,
        eot_id=eot_id,
        overlap_tokens=overlap_atoms,
        require_terminal_eot=False,
    )
    pending = np.empty(0, dtype="<u2")
    removed_stream_bos = False
    for shard_index, path in enumerate(paths):
        payload = read_challenge_shard(path)
        cursor = overlap_atoms if shard_index else 0
        while cursor < len(payload):
            stop = min(cursor + target_atoms, len(payload))
            window = payload[cursor:stop]
            terminal = np.flatnonzero(window == eot_id)
            if not terminal.size:
                pending = np.concatenate((pending, np.asarray(window)))
                if pending.size > max_document_atoms:
                    raise ValueError(
                        "source document exceeds authenticated max_document_tokens"
                    )
                cursor = stop
                continue

            document_lengths = np.diff(
                np.r_[
                    -pending.size - 1,
                    terminal,
                ]
            )
            if bool((document_lengths > max_document_atoms).any()):
                raise ValueError(
                    "source document exceeds authenticated max_document_tokens"
                )
            complete_stop = int(terminal[-1]) + 1
            complete_window = np.asarray(window[:complete_stop])
            complete = (
                np.concatenate((pending, complete_window))
                if pending.size
                else complete_window
            )
            pending = np.asarray(window[complete_stop:]).copy()
            if pending.size > max_document_atoms:
                raise ValueError(
                    "source document exceeds authenticated max_document_tokens"
                )
            if not removed_stream_bos:
                if complete.size == 0 or complete[0] != eot_id:
                    raise ValueError("byte stream lacks its leading EOT/BOS sentinel")
                complete = complete[1:]
                removed_stream_bos = True
            if complete.size:
                # One bounded conversion gives the entropy model its native
                # arithmetic dtype. The payload stays memory-mapped uint16.
                yield np.asarray(complete, dtype=np.int64)
            cursor = stop
        del payload
    if not removed_stream_bos:
        raise ValueError("byte stream contains no complete source documents")
    # ``pending`` is the allowed unterminated exact-budget tail. It is omitted
    # exactly as it is by the document-aligned byte dataset builder.


def publish_patcher_artifact(
    output: Path,
    artifact: bytes,
    record: dict[str, object],
) -> Path:
    """Publish metadata first and use the artifact as the completion marker."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = output.with_suffix(output.suffix + ".json")
    artifact_working = output.with_suffix(output.suffix + ".working")
    metadata_working = metadata.with_suffix(metadata.suffix + ".working")
    metadata_payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    with artifact_working.open("wb") as handle:
        handle.write(artifact)
        handle.flush()
        os.fsync(handle.fileno())
    with metadata_working.open("w") as handle:
        handle.write(metadata_payload)
        handle.flush()
        os.fsync(handle.fileno())
    # If interrupted between these replacements, the absent artifact permits
    # a safe retry, which replaces the already-published metadata.
    os.replace(metadata_working, metadata)
    os.replace(artifact_working, output)
    return metadata


def authenticated_training_overlap(
    manifest_path: Path,
    manifest: dict[str, object],
    paths: tuple[Path, ...],
) -> int:
    """Bind the selected challenge shards to loader-aligned manifest geometry."""

    discovered = discover_source_manifest(paths)
    if discovered is None or discovered[0].resolve() != manifest_path.resolve():
        raise ValueError("training shards do not share the selected mix manifest")
    if manifest.get("loader_aligned") is not True:
        raise ValueError("byte entropy fitting requires loader-aligned shards")
    declared_shards = manifest.get("shards")
    if not isinstance(declared_shards, list) or len(declared_shards) != len(paths):
        raise ValueError("mix manifest shard inventory does not match training files")

    physical_atoms = 0
    for index, (path, record) in enumerate(zip(paths, declared_shards, strict=True)):
        if not isinstance(record, dict):
            raise ValueError(f"invalid shard record {index} in mix manifest")
        expected_name = f"fineweb_train_{index:06d}.bin"
        if path.name != expected_name or record.get("index") != index:
            raise ValueError(
                f"training shard order differs at index {index}: {path.name}"
            )
        payload = read_challenge_shard(path)
        payload_atoms = len(payload)
        del payload
        if record.get("tokens") != payload_atoms:
            raise ValueError(
                f"manifest token count differs for {path}: "
                f"expected {record.get('tokens')}, observed {payload_atoms}"
            )
        physical_atoms += payload_atoms

    overlap = infer_train_overlap(discovered, paths)
    transitions = max(len(paths) - 1, 0)
    unique_atoms = physical_atoms - transitions * overlap
    expected_counts = {
        "physical_shard_tokens": physical_atoms,
        "unique_stream_tokens": unique_atoms,
        "boundary_overlap_tokens": transitions * overlap,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected_counts.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"mix manifest stream geometry mismatch: {mismatches}")
    return overlap


def calibrated_threshold(
    entropies: np.ndarray,
    document_ids: np.ndarray,
    *,
    target_mean_patch: float,
    max_patch_size: int,
) -> tuple[float, float, int]:
    """Choose the observed entropy threshold nearest the target mean length."""

    if entropies.size == 0 or not 1 <= target_mean_patch <= max_patch_size:
        raise ValueError("entropy calibration geometry is invalid")
    candidates = np.unique(entropies)
    candidates = np.r_[
        np.nextafter(candidates[0], -np.inf),
        candidates,
        np.nextafter(candidates[-1], np.inf),
    ]
    low, high = 0, candidates.size - 1

    def evaluate(index: int) -> tuple[float, int]:
        starts = entropy_patch_start_mask(
            entropies,
            document_ids,
            EntropyPatchConfig(
                mode="threshold",
                threshold=float(candidates[index]),
                max_patch_size=max_patch_size,
            ),
        )
        patches = int(starts.sum())
        return entropies.size / patches, patches

    while low < high:
        middle = (low + high) // 2
        mean_patch, _ = evaluate(middle)
        if mean_patch < target_mean_patch:
            low = middle + 1
        else:
            high = middle
    indices = range(max(0, low - 2), min(candidates.size, low + 3))
    best = min(
        (
            (abs(evaluate(index)[0] - target_mean_patch), index, *evaluate(index))
            for index in indices
        ),
        key=lambda item: (item[0], item[1]),
    )
    _, index, mean_patch, patches = best
    return float(candidates[index]), float(mean_patch), int(patches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-order", type=int, default=4)
    parser.add_argument("--table-size", type=int, default=1_024)
    parser.add_argument("--smoothing", type=float, default=0.5)
    parser.add_argument("--max-patch-size", type=int, default=8)
    parser.add_argument("--target-mean-patch", type=float, default=4.0)
    parser.add_argument("--batch-atoms", type=int, default=64 << 20)
    parser.add_argument("--calibration-atoms", type=int, default=16 << 20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.source / "mix_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    provenance = manifest.get("tokenizer_provenance", {})
    expected = {"kind": "utf8_bytes", "vocab_size": 261, "eot_id": 256}
    mismatch = {
        name: (provenance.get(name), value)
        for name, value in expected.items()
        if provenance.get(name) != value
    }
    if mismatch:
        raise ValueError(f"source is not the canonical byte corpus: {mismatch}")
    paths = tuple(sorted(args.source.glob("fineweb_train_*.bin")))
    if not paths:
        raise FileNotFoundError("byte corpus has no training shards")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    overlap = authenticated_training_overlap(manifest_path, manifest, paths)
    max_document_atoms = manifest.get("max_document_tokens")
    if not isinstance(max_document_atoms, int) or max_document_atoms <= 0:
        raise ValueError("mix manifest lacks a positive max_document_tokens")
    fingerprints = tuple(
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    )
    model = HashedNgramEntropyModel(
        HashedNgramEntropyConfig(
            vocab_size=261,
            context_order=args.context_order,
            table_size=args.table_size,
            additive_smoothing=args.smoothing,
        )
    )
    calibration = np.empty(args.calibration_atoms, dtype=np.int64)
    calibration_size = 0
    atoms = 0
    chunks = 0
    for ids in logical_training_chunks(
        paths,
        overlap_atoms=overlap,
        eot_id=256,
        target_atoms=args.batch_atoms,
        max_document_atoms=max_document_atoms,
    ):
        documents = document_ids_from_eot(ids, eot_id=256)
        model.update(ids, documents)
        take = min(ids.size, calibration.size - calibration_size)
        if take:
            calibration[calibration_size : calibration_size + take] = ids[:take]
            calibration_size += take
        atoms += ids.size
        chunks += 1
        print(f"fit chunks={chunks} atoms={atoms}", flush=True)
    expected_atoms = manifest.get("unique_stream_tokens")
    if not isinstance(expected_atoms, int) or not 0 < atoms <= expected_atoms:
        raise RuntimeError(
            "entropy fitter consumed an invalid amount of the authenticated "
            f"logical stream: expected at most {expected_atoms}, observed {atoms}"
        )
    excluded_atoms = expected_atoms - atoms
    if excluded_atoms < 1:
        raise RuntimeError("entropy fitter did not exclude the stream BOS sentinel")
    incomplete_tail_atoms = excluded_atoms - 1
    calibration = calibration[:calibration_size]
    calibration_documents = document_ids_from_eot(calibration, eot_id=256)
    entropies = model.predict_entropies_numpy(
        calibration, calibration_documents
    )
    threshold, mean_patch, calibration_patches = calibrated_threshold(
        entropies,
        calibration_documents,
        target_mean_patch=args.target_mean_patch,
        max_patch_size=args.max_patch_size,
    )
    patcher = CausalEntropyPatcher(
        model,
        EntropyPatchConfig(
            mode="threshold",
            threshold=threshold,
            max_patch_size=args.max_patch_size,
        ),
    )
    artifact = patcher.to_bytes()
    record = {
        "schema": "byte_entropy_patcher_fit/v1",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_fingerprints": fingerprints,
        "training_atoms": atoms,
        "excluded_stream_bos_atoms": 1,
        "excluded_incomplete_tail_atoms": incomplete_tail_atoms,
        "fit_chunks": chunks,
        "calibration_atoms": calibration_size,
        "calibration_patches": calibration_patches,
        "calibration_mean_patch_size": mean_patch,
        "target_mean_patch_size": args.target_mean_patch,
        "patcher_sha256": patcher.sha256,
        "patcher_bytes": len(artifact),
        "model_config": patcher.model.config.__dict__,
        "patch_config": patcher.config.__dict__,
    }
    publish_patcher_artifact(args.output, artifact, record)
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
