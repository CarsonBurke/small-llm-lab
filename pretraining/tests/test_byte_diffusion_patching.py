from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

import scripts.fit_byte_entropy_patcher as fitter_module
from pretraining.byte_diffusion.patching import (
    CausalEntropyPatcher,
    EntropyPatchConfig,
    HashedNgramEntropyConfig,
    HashedNgramEntropyModel,
    context_hashes_numpy,
    context_hashes_torch,
    document_ids_from_eot,
    entropy_patch_start_mask,
    fixed_stride_patch_layout,
)
from scripts.fit_byte_entropy_patcher import (
    authenticated_training_overlap,
    calibrated_threshold,
    logical_training_chunks,
    publish_patcher_artifact,
)
from scripts.build_bolmo_dataset import (
    CHALLENGE_HEADER_INTS,
    CHALLENGE_MAGIC,
    CHALLENGE_VERSION,
)


def _fitted_patcher(*, mode: str = "threshold", threshold: float = 2.0):
    # UTF-8 is intentionally represented only by its bytes. Multibyte code
    # points receive no special treatment in fitting or patch routing.
    documents = tuple(
        np.concatenate(
            (
                np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64),
                np.asarray([256], dtype=np.int64),
            )
        )
        for text in ("café 🙂\n", "naïve λ\n")
    )
    ids = np.concatenate(documents)
    document_ids = np.repeat(np.arange(len(documents)), [doc.size for doc in documents])
    estimator = HashedNgramEntropyModel.fit(
        ids,
        document_ids,
        HashedNgramEntropyConfig(
            vocab_size=261,
            context_order=3,
            table_size=64,
            additive_smoothing=0.5,
        ),
    )
    return CausalEntropyPatcher(
        estimator,
        EntropyPatchConfig(mode=mode, threshold=threshold, max_patch_size=4),
    )


def test_utf8_sequences_are_routed_as_variable_raw_bytes() -> None:
    patcher = _fitted_patcher(threshold=100.0)
    ids = np.frombuffer("é🙂x".encode("utf-8"), dtype=np.uint8).astype(np.int64)
    plan = patcher.patch(ids, np.zeros(ids.size, dtype=np.int64))

    assert ids.tolist() == [195, 169, 240, 159, 153, 130, 120]
    assert plan.layout.patch_lengths.tolist() == [4, 3]
    assert plan.layout.byte_to_patch.tolist() == [0, 0, 0, 0, 1, 1, 1]


@pytest.mark.parametrize("mode", ["threshold", "cumulative"])
def test_context_hashes_and_boundaries_are_document_isolated(mode: str) -> None:
    patcher = _fitted_patcher(mode=mode, threshold=6.0)
    first = np.asarray([10, 11, 12, 13, 20, 21, 22, 23], dtype=np.int64)
    changed = first.copy()
    changed[:4] = np.asarray([90, 91, 92, 93])
    documents = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

    first_plan = patcher.patch(first, documents)
    changed_plan = patcher.patch(changed, documents)

    np.testing.assert_array_equal(first_plan.entropies[4:], changed_plan.entropies[4:])
    np.testing.assert_array_equal(first_plan.starts[4:], changed_plan.starts[4:])
    assert first_plan.layout.patch_ordinals[first_plan.layout.patch_cu_seqlens[1]] == 0


def test_threshold_and_cumulative_modes_reset_at_document_boundaries() -> None:
    entropies = np.asarray([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    documents = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    threshold = entropy_patch_start_mask(
        entropies,
        documents,
        EntropyPatchConfig(mode="threshold", threshold=1.5, max_patch_size=8),
    )
    cumulative = entropy_patch_start_mask(
        entropies,
        documents,
        EntropyPatchConfig(mode="cumulative", threshold=3.0, max_patch_size=8),
    )

    assert np.flatnonzero(threshold).tolist() == [0, 1, 2, 3, 4]
    assert np.flatnonzero(cumulative).tolist() == [0, 1, 3, 4]


@pytest.mark.parametrize("mode", ["threshold", "cumulative"])
def test_maximum_patch_size_is_a_hard_bound(mode: str) -> None:
    entropies = np.zeros(19, dtype=np.float64)
    documents = np.zeros(19, dtype=np.int64)
    starts = entropy_patch_start_mask(
        entropies,
        documents,
        EntropyPatchConfig(mode=mode, threshold=100.0, max_patch_size=4),
    )
    layout = fixed_stride_patch_layout(documents, stride=4)

    assert np.flatnonzero(starts).tolist() == [0, 4, 8, 12, 16]
    assert int(layout.patch_lengths.max()) <= 4


@pytest.mark.parametrize("mode,threshold", [("threshold", 2.0), ("cumulative", 6.0)])
def test_patching_is_prefix_consistent_and_cannot_read_future_bytes(
    mode: str, threshold: float
) -> None:
    patcher = _fitted_patcher(mode=mode, threshold=threshold)
    ids = np.asarray([40, 41, 42, 43, 44, 45, 46, 47], dtype=np.int64)
    documents = np.zeros(ids.size, dtype=np.int64)
    full = patcher.patch(ids, documents)

    for stop in range(1, ids.size + 1):
        prefix = patcher.patch(ids[:stop], documents[:stop])
        np.testing.assert_array_equal(prefix.entropies, full.entropies[:stop])
        np.testing.assert_array_equal(prefix.starts, full.starts[:stop])

    changed = ids.copy()
    changed[4:] = np.asarray([80, 81, 82, 83])
    changed_plan = patcher.patch(changed, documents)
    np.testing.assert_array_equal(full.entropies[:5], changed_plan.entropies[:5])
    np.testing.assert_array_equal(full.starts[:5], changed_plan.starts[:5])


def test_fixed_stride_layout_matches_document_local_reference() -> None:
    documents = np.asarray([4] * 5 + [9] * 3 + [12] * 8, dtype=np.int64)
    layout = fixed_stride_patch_layout(documents, stride=4)

    assert layout.patch_starts.tolist() == [0, 4, 5, 8, 12]
    assert layout.patch_stops.tolist() == [4, 5, 8, 12, 16]
    assert layout.patch_lengths.tolist() == [4, 1, 3, 4, 4]
    assert layout.patch_ordinals.tolist() == [0, 1, 0, 0, 1]
    assert layout.byte_patch_ordinals.tolist() == [0] * 4 + [1] + [0] * 7 + [1] * 4
    assert layout.byte_cu_seqlens.tolist() == [0, 5, 8, 16]
    assert layout.patch_cu_seqlens.tolist() == [0, 2, 3, 5]
    assert layout.patch_byte_cu_seqlens.tolist() == [0, 4, 5, 8, 12, 16]
    # A byte can consume its current patch latent only after all bytes in that
    # patch are visible; earlier positions route to the prior patch/BOS.
    assert layout.condition_patch_indices.tolist() == [
        -1,
        -1,
        -1,
        0,
        1,
        -1,
        -1,
        2,
        -1,
        -1,
        -1,
        3,
        3,
        3,
        3,
        4,
    ]


def test_numpy_and_torch_hash_and_entropy_inference_are_exactly_reproducible() -> None:
    patcher = _fitted_patcher()
    ids = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64)
    documents = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)

    numpy_hashes = context_hashes_numpy(ids, documents, patcher.model.config)
    torch_hashes = context_hashes_torch(
        torch.from_numpy(ids), torch.from_numpy(documents), patcher.model.config
    )
    np.testing.assert_array_equal(numpy_hashes, torch_hashes.numpy())
    torch.testing.assert_close(
        patcher.model.predict_entropies_torch(
            torch.from_numpy(ids), torch.from_numpy(documents)
        ),
        torch.from_numpy(patcher.model.predict_entropies_numpy(ids, documents)),
        rtol=0,
        atol=0,
    )


def test_serialized_patcher_is_deterministic_hash_bound_and_rejects_tampering() -> None:
    patcher = _fitted_patcher(mode="cumulative", threshold=5.0)
    first = patcher.to_bytes()
    second = patcher.to_bytes()

    assert first == second
    assert patcher.sha256 == hashlib.sha256(first).hexdigest()
    restored = CausalEntropyPatcher.from_bytes(first)
    assert restored.sha256 == patcher.sha256
    np.testing.assert_array_equal(restored.model.counts, patcher.model.counts)
    assert restored.config == patcher.config

    corrupted = bytearray(first)
    corrupted[-1] ^= 1
    with pytest.raises(ValueError, match="hash mismatch"):
        CausalEntropyPatcher.from_bytes(bytes(corrupted))


def test_default_estimator_count_table_fits_the_declared_compact_budget() -> None:
    model = HashedNgramEntropyModel()

    assert model.parameter_bytes == 1_024 * 261 * 4
    assert len(model.to_bytes()) < 1_100_000


def test_eot_document_ids_reset_after_the_terminal_byte() -> None:
    ids = np.asarray([65, 66, 256, 70, 71, 256, 80], dtype=np.int64)
    assert document_ids_from_eot(ids).tolist() == [0, 0, 0, 1, 1, 1, 2]


def test_fitter_preserves_documents_across_overlapped_physical_shards(
    tmp_path,
    monkeypatch,
) -> None:
    def write_shard(path, values) -> None:
        header = np.zeros(CHALLENGE_HEADER_INTS, dtype="<i4")
        header[0] = CHALLENGE_MAGIC
        header[1] = CHALLENGE_VERSION
        header[2] = len(values)
        with path.open("wb") as handle:
            handle.write(header.tobytes())
            handle.write(np.asarray(values, dtype="<u2").tobytes())

    first = tmp_path / "fineweb_train_000000.bin"
    second = tmp_path / "fineweb_train_000001.bin"
    write_shard(first, [256, 1, 2, 3])
    # The leading 3 is the one-atom loader overlap, not a second corpus atom.
    # The final two atoms are an allowed unterminated exact-budget tail.
    write_shard(second, [3, 4, 256, 9, 256, 10, 11])
    manifest = {
        "loader_aligned": True,
        "physical_shard_tokens": 11,
        "unique_stream_tokens": 10,
        "boundary_overlap_tokens": 1,
        "shards": [
            {"index": 0, "tokens": 4},
            {"index": 1, "tokens": 7},
        ],
    }
    manifest_path = tmp_path / "mix_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    overlap = authenticated_training_overlap(
        manifest_path, manifest, (first, second)
    )
    assert overlap == 1

    class TrackingPayload:
        def __init__(self, values) -> None:
            self.values = np.asarray(values, dtype="<u2")
            self.max_slice = 0

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, item):
            if isinstance(item, slice):
                start, stop, step = item.indices(len(self.values))
                assert step == 1
                self.max_slice = max(self.max_slice, stop - start)
            return self.values[item]

    payloads = {
        first: TrackingPayload([256, 1, 2, 3]),
        second: TrackingPayload([3, 4, 256, 9, 256, 10, 11]),
    }
    monkeypatch.setattr(
        fitter_module,
        "read_challenge_shard",
        lambda path: payloads[path],
    )

    chunks = tuple(
        logical_training_chunks(
            (first, second),
            overlap_atoms=overlap,
            eot_id=256,
            target_atoms=3,
            max_document_atoms=8,
        )
    )

    np.testing.assert_array_equal(
        np.concatenate(chunks), np.asarray([1, 2, 3, 4, 256, 9, 256])
    )
    assert all(chunk[-1] == 256 for chunk in chunks)
    assert max(payload.max_slice for payload in payloads.values()) <= 3


def test_patcher_publication_recovers_from_metadata_only_partial_commit(
    tmp_path,
) -> None:
    output = tmp_path / "entropy.bdpatch"
    metadata = output.with_suffix(output.suffix + ".json")
    metadata.write_text("stale partial publication")
    record = {"schema": "test", "patcher_bytes": 3}

    observed_metadata = publish_patcher_artifact(output, b"new", record)

    assert output.read_bytes() == b"new"
    assert observed_metadata == metadata
    assert json.loads(metadata.read_text()) == record


def test_fitter_main_records_excluded_boundaries_and_publishes_pair(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()

    def write_shard(path, values) -> None:
        header = np.zeros(CHALLENGE_HEADER_INTS, dtype="<i4")
        header[:3] = (CHALLENGE_MAGIC, CHALLENGE_VERSION, len(values))
        with path.open("wb") as handle:
            handle.write(header.tobytes())
            handle.write(np.asarray(values, dtype="<u2").tobytes())

    write_shard(source / "fineweb_train_000000.bin", [256, 1, 2, 3])
    write_shard(
        source / "fineweb_train_000001.bin",
        [3, 4, 256, 9, 256, 10, 11],
    )
    (source / "mix_manifest.json").write_text(
        json.dumps(
            {
                "tokenizer_provenance": {
                    "kind": "utf8_bytes",
                    "vocab_size": 261,
                    "eot_id": 256,
                },
                "loader_aligned": True,
                "physical_shard_tokens": 11,
                "unique_stream_tokens": 10,
                "boundary_overlap_tokens": 1,
                "max_document_tokens": 8,
                "shards": [
                    {"index": 0, "tokens": 4},
                    {"index": 1, "tokens": 7},
                ],
            }
        )
    )
    output = tmp_path / "entropy.bdpatch"
    monkeypatch.setattr(
        "sys.argv",
        [
            "fit_byte_entropy_patcher.py",
            "--source",
            str(source),
            "--output",
            str(output),
            "--context-order",
            "2",
            "--table-size",
            "8",
            "--batch-atoms",
            "3",
            "--calibration-atoms",
            "7",
        ],
    )

    fitter_module.main()

    record = json.loads(output.with_suffix(".bdpatch.json").read_text())
    assert record["training_atoms"] == 7
    assert record["excluded_stream_bos_atoms"] == 1
    assert record["excluded_incomplete_tail_atoms"] == 2
    assert record["calibration_atoms"] == 7
    assert record["patcher_bytes"] == output.stat().st_size
    CausalEntropyPatcher.from_bytes(output.read_bytes())
    assert not output.with_suffix(".bdpatch.working").exists()
    assert not output.with_suffix(".bdpatch.json.working").exists()


def test_threshold_calibration_selects_nearest_observed_routing_cell() -> None:
    entropies = np.asarray([0.5, 4.0, 0.5, 0.5, 4.0, 0.5, 0.5, 0.5])
    documents = np.zeros(entropies.size, dtype=np.int64)

    threshold, mean_patch, patches = calibrated_threshold(
        entropies,
        documents,
        target_mean_patch=4.0,
        max_patch_size=8,
    )

    assert threshold > 0
    assert mean_patch == pytest.approx(entropies.size / patches)
    # A global threshold can only choose all high-entropy boundaries or none
    # for this two-level sample. The former is the nearest realizable cell.
    assert mean_patch == pytest.approx(8 / 3)
