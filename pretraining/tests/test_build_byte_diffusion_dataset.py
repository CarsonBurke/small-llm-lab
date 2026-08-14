from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pretraining.bolmo_data import SourceTokenByteifier
from pretraining.byte_diffusion.data import AtomicDocument, AtomicIdManifest, pack_documents
from pretraining.byte_diffusion.patching import (
    CausalEntropyPatcher,
    EntropyPatchConfig,
    HashedNgramEntropyConfig,
    HashedNgramEntropyModel,
)
from pretraining.byte_diffusion.training import chunks_to_batch, load_data_directory
from scripts.build_bolmo_dataset import (
    CHALLENGE_HEADER_INTS,
    CHALLENGE_MAGIC,
    CHALLENGE_VERSION,
    canonical_sha256,
)
from scripts.build_byte_diffusion_dataset import (
    ARTIFACT_SCHEMA,
    ARTIFACT_ALIGNMENT,
    DATASET_SCHEMA,
    ENTROPY_DATASET_SCHEMA,
    PATCHING_POLICY_SCHEMA,
    ChallengeDocumentReader,
    DatasetPatchingPolicy,
    StreamingDocumentPacker,
    VectorizedDocumentPagePacker,
    _chunk_arrays,
    build_dataset,
    build_split,
    parse_args,
    resolve_patching_policy,
    validate_atomic_utf8,
    validate_atomic_utf8_batch,
    write_deterministic_npz,
    write_deterministic_mapped_artifact,
)
from tokenization import ngram_store
from tokenization.spec import (
    DEFAULT_SPECIALS,
    PRETOKEN_PATTERN,
    NgramReference,
    TokenizerSpec,
)
from tokenization.tokenizer import SplitTreeNumericTokenizer
from tokenization.train import BYTE_ALPHABET
from tokenization.tst import NumericScheme


EXTRA_TEXT_TOKENS = (b"hello", b"world", b"cafe", "🙂".encode())


def _write_tokenizer(directory: Path) -> SplitTreeNumericTokenizer:
    directory.mkdir(parents=True)
    counts = {surface: 8 for surface in BYTE_ALPHABET}
    counts.update({surface: 64 for surface in EXTRA_TEXT_TOKENS})
    digest = ngram_store.write_counts(directory / "ngrams.bin", counts)
    spec = TokenizerSpec(
        numeric=NumericScheme(group_size=1, compound=True),
        text_tokens=BYTE_ALPHABET + EXTRA_TEXT_TOKENS,
        ngrams=NgramReference(
            filename="ngrams.bin",
            sha256=digest,
            entries=len(counts),
            min_count=1,
            max_length=8,
        ),
        specials=DEFAULT_SPECIALS,
        pretoken_pattern=PRETOKEN_PATTERN,
    )
    spec.write(directory / "tokenizer.json")
    return SplitTreeNumericTokenizer.from_directory(directory)


def _write_challenge_shard(path: Path, tokens: list[int]) -> None:
    header = np.zeros(CHALLENGE_HEADER_INTS, dtype="<i4")
    header[0] = CHALLENGE_MAGIC
    header[1] = CHALLENGE_VERSION
    header[2] = len(tokens)
    with path.open("wb") as handle:
        handle.write(header.tobytes())
        handle.write(np.asarray(tokens, dtype="<u2").tobytes())


def _source_manifest(
    tokenizer_dir: Path,
    tokenizer: SplitTreeNumericTokenizer,
    *,
    physical_tokens: int,
    unique_tokens: int,
) -> dict:
    return {
        "tokenizer_provenance": {
            "kind": "toast_tst",
            "vocab_size": tokenizer.vocab_size,
            "eot_id": tokenizer.eot_id,
            "spec_sha256": hashlib.sha256(
                (tokenizer_dir / "tokenizer.json").read_bytes()
            ).hexdigest(),
            "ngrams_sha256": tokenizer.spec.ngrams.sha256,
        },
        "loader_aligned": True,
        "physical_shard_tokens": physical_tokens,
        "unique_stream_tokens": unique_tokens,
    }


def _write_entropy_patcher(path: Path, *, max_patch_size: int = 5) -> bytes:
    patcher = CausalEntropyPatcher(
        HashedNgramEntropyModel(
            HashedNgramEntropyConfig(
                vocab_size=261,
                context_order=2,
                table_size=16,
                additive_smoothing=0.5,
            )
        ),
        EntropyPatchConfig(
            mode="threshold",
            threshold=100.0,
            max_patch_size=max_patch_size,
        ),
    )
    artifact = patcher.to_bytes()
    path.write_bytes(artifact)
    return artifact


@pytest.fixture
def fixture_source(tmp_path: Path):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer = _write_tokenizer(tokenizer_dir)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    documents = [
        [*tokenizer.encode("A"), tokenizer.eot_id],
        [*tokenizer.encode("abcdefghijkl"), tokenizer.eot_id],
        [*tokenizer.encode("value 12.034"), tokenizer.eot_id],
    ]
    train = [token for document in documents for token in document]
    cut = max(2, len(train) // 2)
    train_a_tokens = train[:cut]
    train_b_tokens = train[cut - 1 :]
    train_a = source_dir / "train-000.bin"
    train_b = source_dir / "train-001.bin"
    validation = source_dir / "validation.bin"
    validation_documents = [
        [*tokenizer.encode("hello"), tokenizer.eot_id],
        [*tokenizer.encode("🙂"), tokenizer.eot_id],
    ]
    validation_tokens = [
        token for document in validation_documents for token in document
    ]
    _write_challenge_shard(train_a, train_a_tokens)
    _write_challenge_shard(train_b, train_b_tokens)
    _write_challenge_shard(validation, validation_tokens)
    manifest = _source_manifest(
        tokenizer_dir,
        tokenizer,
        physical_tokens=len(train_a_tokens) + len(train_b_tokens),
        unique_tokens=len(train),
    )
    (source_dir / "mix_manifest.json").write_text(json.dumps(manifest))
    return {
        "tokenizer_dir": tokenizer_dir,
        "tokenizer": tokenizer,
        "train_paths": (train_a, train_b),
        "validation_paths": (validation,),
        "train_documents": documents,
        "validation_documents": validation_documents,
        "train_tokens": train,
    }


def _load_split_arrays(output: Path, split: dict) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {}
    for artifact in split["artifacts"]:
        path = output / artifact["path"]
        for name, descriptor in artifact["arrays"].items():
            array = np.memmap(
                path,
                mode="r",
                dtype=np.dtype(descriptor["dtype"]),
                offset=int(descriptor["byte_offset"]),
                shape=tuple(descriptor["shape"]),
            )
            rows.setdefault(name, []).append(np.asarray(array).copy())
    return {name: np.concatenate(values, axis=0) for name, values in rows.items()}


def test_document_reader_deduplicates_overlap_and_keeps_eot_boundaries(
    fixture_source,
) -> None:
    reader = ChallengeDocumentReader(
        fixture_source["train_paths"], eot_id=0, overlap_tokens=1
    )

    documents = list(reader.iter_documents())

    assert documents == [tuple(document) for document in fixture_source["train_documents"]]
    assert reader.stats.unique_tokens == len(fixture_source["train_tokens"])
    assert reader.stats.complete_documents == 3
    assert reader.stats.incomplete_tail_tokens == 0


def test_streaming_packer_exactly_matches_reference_contract(fixture_source) -> None:
    tokenizer = fixture_source["tokenizer"]
    manifest = AtomicIdManifest.reference()
    byteifier = SourceTokenByteifier(tokenizer)
    documents: list[AtomicDocument] = []
    for index, source_ids in enumerate(fixture_source["train_documents"]):
        byteifier.reset()
        documents.append(
            AtomicDocument(str(index), byteifier.byteify(source_ids).atomic_ids)
        )

    streaming = StreamingDocumentPacker(manifest, chunk_size=8)
    actual = []
    for index, document in enumerate(documents):
        actual.extend(streaming.add_document(index, document))
    actual.extend(streaming.finish())

    expected_ids = tuple(
        atomic_id for document in documents for atomic_id in document.atomic_ids
    )
    observed_ids = tuple(
        atomic_id
        for chunk in actual
        for atomic_id, valid in zip(chunk.input_ids, chunk.valid_mask, strict=True)
        if valid
    )
    observed_targets = tuple(
        target
        for chunk in actual
        for target, score in zip(chunk.target_ids, chunk.score_mask, strict=True)
        if score
    )
    assert observed_ids == expected_ids
    assert observed_targets == expected_ids[1:]
    assert len(actual) == -(-len(expected_ids) // 8)
    assert any(
        len({value for value in chunk.document_indices if value >= 0}) > 1
        for chunk in actual
    )


def test_vectorized_document_packer_exactly_matches_streaming_reference() -> None:
    manifest = AtomicIdManifest.reference()
    documents = (
        AtomicDocument("empty", (manifest.eot_id,)),
        AtomicDocument("short", (65, 66, manifest.eot_id)),
        AtomicDocument("aligned", tuple(range(70, 77)) + (manifest.eot_id,)),
        AtomicDocument(
            "cross-page",
            tuple((index % 95) + 32 for index in range(9_000))
            + (manifest.eot_id,),
        ),
        AtomicDocument(
            "utf8", tuple("hé🙂".encode("utf-8")) + (manifest.eot_id,)
        ),
    )
    reference_packer = StreamingDocumentPacker(
        manifest, chunk_size=128, document_aligned_pages=True
    )
    reference_chunks = []
    for index, document in enumerate(documents):
        reference_chunks.extend(reference_packer.add_document(index, document))
    reference_chunks.extend(reference_packer.finish())
    reference = _chunk_arrays(reference_chunks)

    source = np.asarray(
        [value for document in documents for value in document.atomic_ids],
        dtype="<u2",
    )
    lengths = np.asarray([len(document.atomic_ids) for document in documents])
    vectorized = VectorizedDocumentPagePacker(manifest, chunk_size=128)
    first_tokens = int(lengths[:2].sum())
    blocks = [
        vectorized.add_batch(
            source[:first_tokens], lengths[:2], first_document_index=0
        ),
        vectorized.add_batch(
            source[first_tokens:], lengths[2:], first_document_index=2
        ),
        vectorized.finish(),
    ]
    blocks = [block for block in blocks if block is not None]
    observed = {
        name: np.concatenate([block[name] for block in blocks])
        for name in reference
    }

    assert vectorized.alignment_padding == reference_packer.alignment_padding
    for name, expected in reference.items():
        np.testing.assert_array_equal(observed[name], expected, err_msg=name)


def test_vectorized_document_reader_exactly_matches_scalar_reader(
    fixture_source,
) -> None:
    scalar = ChallengeDocumentReader(
        fixture_source["train_paths"], eot_id=0, overlap_tokens=1
    )
    expected = list(scalar.iter_documents())
    expected_stats = scalar.stats
    vectorized = ChallengeDocumentReader(
        fixture_source["train_paths"], eot_id=0, overlap_tokens=1
    )
    observed = []
    for source, lengths in vectorized.iter_document_batches(target_tokens=5):
        cursor = 0
        for length in lengths:
            stop = cursor + int(length)
            observed.append(tuple(int(value) for value in source[cursor:stop]))
            cursor = stop

    assert observed == expected
    assert vectorized.stats == expected_stats


def test_vectorized_byte_native_split_is_byte_identical_to_reference(
    tmp_path: Path,
) -> None:
    manifest = AtomicIdManifest.reference()
    tokens = np.asarray(
        [
            manifest.eot_id,
            *b"a",
            manifest.eot_id,
            *b"bcdef",
            manifest.eot_id,
            *(b"x" * 37),
            manifest.eot_id,
            *"hé🙂".encode("utf-8"),
            manifest.eot_id,
        ],
        dtype="<u2",
    )
    source = tmp_path / "source.bin"
    _write_challenge_shard(source, tokens)
    fast_dir = tmp_path / "fast"
    reference_dir = tmp_path / "reference"
    fast_dir.mkdir()
    reference_dir.mkdir()
    common = {
        "name": "train",
        "paths": (source,),
        "tokenizer": None,
        "atomic_manifest": manifest,
        "overlap_tokens": 0,
        "chunk_size": 16,
        "chunks_per_shard": 2,
        "max_documents": None,
        "require_one_chunk_per_document": False,
        "require_terminal_eot": True,
        "close_rows_at_document": False,
        "document_aligned_pages": True,
    }

    fast = build_split(output_dir=fast_dir, **common)
    reference = build_split(
        output_dir=reference_dir,
        vectorized_byte_native=False,
        **common,
    )

    assert fast == reference
    for artifact in fast["artifacts"]:
        assert (fast_dir / artifact["path"]).read_bytes() == (
            reference_dir / artifact["path"]
        ).read_bytes()


def test_one_pass_builder_rejects_documents_that_expand_to_multiple_rows(
    fixture_source, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="exactly one row per document"):
        build_dataset(
            tokenizer_dir=fixture_source["tokenizer_dir"],
            output_dir=tmp_path / "output",
            train_paths=fixture_source["train_paths"],
            validation_paths=fixture_source["validation_paths"],
            chunk_size=8,
            chunks_per_shard=2,
            require_one_train_chunk_per_document=True,
        )


def test_document_aligned_pages_pack_multiple_isolated_documents() -> None:
    manifest = AtomicIdManifest.reference()
    documents = (
        AtomicDocument("a", (65, 66, manifest.eot_id)),
        AtomicDocument("b", (70, 71, 72, 73, manifest.eot_id)),
        AtomicDocument(
            "c", (80, 81, 82, 83, 84, 85, 86, 87, manifest.eot_id)
        ),
    )
    packer = StreamingDocumentPacker(
        manifest,
        chunk_size=16,
        document_aligned_pages=True,
    )
    chunks = []
    for index, document in enumerate(documents):
        chunks.extend(packer.add_document(index, document))
    chunks.extend(packer.finish())

    assert len(chunks) == 2
    first = chunks[0]
    assert first.valid_mask == (
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
    )
    assert first.document_offsets == (
        0,
        1,
        2,
        -1,
        0,
        1,
        2,
        3,
        4,
        -1,
        -1,
        -1,
        0,
        1,
        2,
        3,
    )
    assert first.label_halo_valid
    assert first.label_halo_id == 84
    batch = chunks_to_batch(chunks)
    assert batch.bos_targets.tolist() == [65, 70, 80]
    assert batch.bos_row_indices is not None
    assert batch.bos_row_indices.tolist() == [0, 0, 0]
    assert int(batch.ar_targets.ne(-100).sum() + batch.bos_targets.numel()) == 17


def test_builder_writes_versioned_reproducible_document_aligned_artifacts(
    fixture_source, tmp_path: Path
) -> None:
    first_output = tmp_path / "built-a"
    second_output = tmp_path / "built-b"
    kwargs = {
        "tokenizer_dir": fixture_source["tokenizer_dir"],
        "train_paths": fixture_source["train_paths"],
        "validation_paths": fixture_source["validation_paths"],
        "chunk_size": 8,
        "chunks_per_shard": 1,
    }

    manifest = build_dataset(output_dir=first_output, **kwargs)
    repeated = build_dataset(output_dir=second_output, **kwargs)

    assert manifest["schema"] == DATASET_SCHEMA
    assert manifest["artifact_schema"] == ARTIFACT_SCHEMA
    atomic = manifest["atomic_vocabulary"]
    assert [item["atomic_id"] for item in atomic["specials"]] == list(
        range(256, 261)
    )
    assert atomic["eot_id"] == 256
    assert atomic["mask_id"] == 261
    assert atomic["pad_id"] == 262
    assert manifest["packing"]["patch_stride"] == 4
    assert manifest["splits"]["train"]["complete_documents"] == 3
    assert manifest["splits"]["train"]["overlap_tokens_per_transition"] == 1
    assert manifest["splits"]["train"]["incomplete_tail_source_tokens"] == 0

    train_arrays = _load_split_arrays(first_output, manifest["splits"]["train"])
    assert train_arrays["input_ids"].shape[1] == 8
    assert np.all(train_arrays["input_ids"][~train_arrays["valid_mask"]] == 262)
    assert np.all(train_arrays["target_ids"][~train_arrays["score_mask"]] == 262)
    assert np.all(train_arrays["document_indices"][~train_arrays["valid_mask"]] == -1)
    assert np.all(train_arrays["patch_offsets"][~train_arrays["valid_mask"]] == -1)
    # Patch phase follows the dense physical stream rather than restarting and
    # wasting the tail of every short document.
    for document_index in range(3):
        positions = train_arrays["document_indices"] == document_index
        physical = np.flatnonzero(positions.reshape(-1))
        assert train_arrays["patch_offsets"][positions].tolist() == [
            int(index % 4) for index in physical
        ]
        assert train_arrays["document_offsets"][positions].tolist() == list(
            range(int(positions.sum()))
        )
    assert train_arrays["label_halo_valid"].any()
    for row in np.flatnonzero(train_arrays["label_halo_valid"]):
        assert train_arrays["target_ids"][row, -1] == train_arrays[
            "label_halo_id"
        ][row]

    saved = json.loads((first_output / "manifest.json").read_text())
    claimed_hash = saved.pop("payload_sha256")
    assert canonical_sha256(saved) == claimed_hash

    loaded_manifest, train_chunks, validation_chunks = load_data_directory(
        first_output,
        chunk_size=8,
        recipe="causal_only",
    )
    assert loaded_manifest.to_dict() == atomic
    assert train_chunks
    assert train_chunks.exposure_summary["rows"] == len(train_chunks)
    assert (
        train_chunks.exposure_summary["literal_atomic_tokens"]
        + train_chunks.exposure_summary["special_atomic_tokens"]
        == train_chunks.exposure_summary["valid_atomic_tokens"]
    )
    assert validation_chunks
    _, pinned_train_chunks, _ = load_data_directory(
        first_output,
        chunk_size=8,
        recipe="causal_only",
        expected_payload_sha256=claimed_hash,
    )
    assert pinned_train_chunks.trust_pinned_row_index
    pinned_train_chunks.training_batch([0])
    assert not pinned_train_chunks._semantically_validated_artifacts
    assert manifest["packing"]["layout"] == "dense_eot_delimited_stream"
    assert any(
        len({value for value in chunk.document_indices if value >= 0}) > 1
        for chunk in train_chunks
    )
    native_batch = train_chunks.training_batch([0, min(1, len(train_chunks) - 1)])
    reference_batch = chunks_to_batch(
        [train_chunks[0], train_chunks[min(1, len(train_chunks) - 1)]],
        dense_stream=True,
    )
    width = native_batch.ids.shape[1]
    reference_batch = type(reference_batch)(
        ids=reference_batch.ids[:, :width],
        valid=reference_batch.valid[:, :width],
        ar_targets=reference_batch.ar_targets[:, :width],
        bos_targets=reference_batch.bos_targets,
        positions=reference_batch.positions[:, :width],
        full_valid=bool(reference_batch.valid[:, :width].all()),
    )
    assert native_batch.full_valid == reference_batch.full_valid
    for field in ("ids", "valid", "ar_targets", "bos_targets"):
        torch.testing.assert_close(
            getattr(native_batch, field), getattr(reference_batch, field)
        )
    torch.testing.assert_close(
        native_batch.positions,
        torch.arange(width)[None].expand(native_batch.ids.shape[0], -1),
    )
    validation_batch, identities = validation_chunks.validation_batch([0])
    assert validation_batch.ids.shape[0] == 1
    assert identities[0].chunk_index == validation_chunks[0].chunk_index
    assert identities[0].stream_start == validation_chunks[0].stream_start
    with pytest.raises(ValueError, match="document-local patch-aligned"):
        load_data_directory(
            first_output,
            chunk_size=8,
            recipe="canvas",
            required_branch_bytes=8,
        )
    for fingerprint in manifest["input_fingerprints"]:
        path = Path(fingerprint["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == fingerprint["sha256"]

    assert manifest["payload_sha256"] == repeated["payload_sha256"]
    first_artifacts = manifest["splits"]["train"]["artifacts"]
    repeated_artifacts = repeated["splits"]["train"]["artifacts"]
    assert [item["sha256"] for item in first_artifacts] == [
        item["sha256"] for item in repeated_artifacts
    ]
    for first, second in zip(first_artifacts, repeated_artifacts, strict=True):
        assert (first_output / first["path"]).read_bytes() == (
            second_output / second["path"]
        ).read_bytes()


def test_builder_consumes_byte_native_source_without_subword_tokenizer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    manifest = AtomicIdManifest.reference()
    # Loader-aligned streams begin with EOT/BOS. It must not become an empty
    # model document; the later EOTs terminate the two literal documents.
    tokens = [
        manifest.eot_id,
        *"héllo".encode("utf-8"),
        manifest.eot_id,
        *"🙂".encode("utf-8"),
        manifest.eot_id,
    ]
    train = source / "train.bin"
    validation = source / "validation.bin"
    _write_challenge_shard(train, tokens)
    _write_challenge_shard(validation, tokens)
    source_manifest = {
        "tokenizer_provenance": {
            "kind": "utf8_bytes",
            "name": "utf8_bytes",
            "vocab_size": manifest.output_size,
            "eot_id": manifest.eot_id,
            "spec_sha256": manifest.sha256,
            "ngrams_sha256": None,
        },
        "loader_aligned": True,
        "physical_shard_tokens": len(tokens),
        "unique_stream_tokens": len(tokens),
        "challenge_validation_shards": [validation.name],
        "challenge_validation": {"documents": 2, "tokens": len(tokens)},
    }
    (source / "mix_manifest.json").write_text(json.dumps(source_manifest))

    built = build_dataset(
        output_dir=output,
        train_paths=(train,),
        validation_paths=(validation,),
        chunk_size=8,
        chunks_per_shard=2,
    )

    assert built["source_encoding"] == {
        "kind": "utf8_bytes",
        "logical_vocab_size": manifest.output_size,
        "eot_id": manifest.eot_id,
        "atomic_manifest_sha256": manifest.sha256,
    }
    assert built["splits"]["train"]["complete_documents"] == 2
    assert built["splits"]["train"]["stream_bos_boundaries"] == 1
    assert built["splits"]["train"]["empty_documents"] == 0
    arrays = _load_split_arrays(output, built["splits"]["train"])
    valid = arrays["input_ids"][arrays["valid_mask"]]
    expected = np.asarray(
        [*"héllo".encode("utf-8"), 256, *"🙂".encode("utf-8"), 256]
    )
    np.testing.assert_array_equal(valid, expected)
    loaded_manifest, _, loaded_validation = load_data_directory(
        output,
        chunk_size=8,
        recipe="causal_only",
        require_challenge_validation=True,
    )
    assert loaded_manifest.sha256 == manifest.sha256
    assert loaded_validation


def test_byte_native_builder_skips_only_the_initial_stream_bos(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    manifest = AtomicIdManifest.reference()
    tokens = [manifest.eot_id, ord("A"), manifest.eot_id, manifest.eot_id]
    train = source / "train.bin"
    validation = source / "validation.bin"
    _write_challenge_shard(train, tokens)
    _write_challenge_shard(validation, tokens)
    source_manifest = {
        "tokenizer_provenance": {
            "kind": "utf8_bytes",
            "name": "utf8_bytes",
            "vocab_size": manifest.output_size,
            "eot_id": manifest.eot_id,
            "spec_sha256": manifest.sha256,
            "ngrams_sha256": None,
        },
        "loader_aligned": True,
        "physical_shard_tokens": len(tokens),
        "unique_stream_tokens": len(tokens),
    }
    (source / "mix_manifest.json").write_text(json.dumps(source_manifest))

    built = build_dataset(
        output_dir=output,
        train_paths=(train,),
        validation_paths=(validation,),
        chunk_size=8,
        chunks_per_shard=2,
    )

    split = built["splits"]["train"]
    assert split["complete_documents"] == 2
    assert split["stream_bos_boundaries"] == 1
    assert split["empty_documents"] == 1
    arrays = _load_split_arrays(output, split)
    np.testing.assert_array_equal(
        arrays["input_ids"][arrays["valid_mask"]],
        np.asarray([ord("A"), manifest.eot_id, manifest.eot_id]),
    )


@pytest.mark.parametrize(
    "atomic_ids",
    [
        (0xC3, 256),
        (0xE2, 0x82, 256),
        (0xF0, 0x9F, 0x99, 256),
        (0xC3, 257, 0xA9, 256),
    ],
)
def test_byte_native_ingestion_rejects_invalid_utf8_segments(atomic_ids) -> None:
    with pytest.raises(ValueError, match="invalid UTF-8"):
        validate_atomic_utf8(atomic_ids, document_key="bad")


def test_reader_reports_or_rejects_unterminated_document(tmp_path: Path) -> None:
    path = tmp_path / "tail.bin"
    _write_challenge_shard(path, [7, 8, 0, 9, 10])
    permissive = ChallengeDocumentReader((path,), eot_id=0)

    assert list(permissive.iter_documents()) == [(7, 8, 0)]
    assert permissive.stats.incomplete_tail_tokens == 2

    strict = ChallengeDocumentReader((path,), eot_id=0, require_terminal_eot=True)
    with pytest.raises(ValueError, match="ends inside a document"):
        list(strict.iter_documents())


def test_reader_rejects_false_declared_overlap(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    _write_challenge_shard(first, [1, 2, 3])
    _write_challenge_shard(second, [9, 4, 0])

    with pytest.raises(ValueError, match="overlap does not match"):
        ChallengeDocumentReader((first, second), eot_id=0, overlap_tokens=1)


def test_deterministic_npz_is_byte_identical(tmp_path: Path) -> None:
    arrays = {
        "z": np.asarray([[1, 2], [3, 4]], dtype="<u2"),
        "a": np.asarray([True, False], dtype=np.bool_),
    }
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"

    write_deterministic_npz(first, arrays)
    write_deterministic_npz(second, dict(reversed(tuple(arrays.items()))))

    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as restored:
        assert restored.files == ["a", "z"]
        assert np.array_equal(restored["z"], arrays["z"])


def test_mapped_artifact_is_deterministic_aligned_and_row_addressable(
    tmp_path: Path,
) -> None:
    arrays = {
        "z": np.arange(64 * 8, dtype="<u2").reshape(64, 8),
        "a": np.arange(64, dtype="<i8"),
    }
    first = tmp_path / "first.bdm"
    second = tmp_path / "second.bdm"
    first_index = write_deterministic_mapped_artifact(first, arrays)
    second_index = write_deterministic_mapped_artifact(
        second, dict(reversed(tuple(arrays.items())))
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_index == second_index
    assert list(first_index) == ["a", "z"]
    assert all(
        descriptor["byte_offset"] % ARTIFACT_ALIGNMENT == 0
        for descriptor in first_index.values()
    )
    z = first_index["z"]
    mapped = np.memmap(
        first,
        mode="r",
        dtype=np.dtype(z["dtype"]),
        offset=z["byte_offset"],
        shape=tuple(z["shape"]),
    )
    selected = mapped[np.arange(32)]
    assert isinstance(mapped, np.memmap)
    assert selected.shape == (32, 8)
    assert selected.nbytes == arrays["z"][:32].nbytes
    np.testing.assert_array_equal(selected, arrays["z"][:32])


def test_entropy_patcher_binding_rejects_tampering_and_wrong_policy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "entropy.patcher"
    artifact = _write_entropy_patcher(path)

    policy = resolve_patching_policy(
        "causal_entropy_v1", entropy_patcher_path=path
    )

    assert policy.name == "causal_entropy_v1"
    assert policy.artifact_sha256 == hashlib.sha256(artifact).hexdigest()
    assert policy.to_manifest()["patcher_artifact"]["parameter_bytes"] == 16 * 261 * 4
    with pytest.raises(ValueError, match="requires --entropy-patcher"):
        resolve_patching_policy("causal_entropy_v1")
    with pytest.raises(ValueError, match="only valid"):
        resolve_patching_policy("fixed_stride_v1", entropy_patcher_path=path)

    corrupted = bytearray(artifact)
    corrupted[-1] ^= 1
    path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="hash mismatch"):
        DatasetPatchingPolicy.causal_entropy(path)


def test_entropy_dataset_packs_whole_patches_and_binds_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    atomic = AtomicIdManifest.reference()
    # The first EOT is the loader's stream BOS. The real document has forced
    # max-five patches [5, 5, 2], including EOT in the final short tail.
    tokens = [atomic.eot_id, *b"abcdefghijk", atomic.eot_id]
    train = source / "train.bin"
    validation = source / "validation.bin"
    _write_challenge_shard(train, tokens)
    _write_challenge_shard(validation, tokens)
    source_manifest = {
        "tokenizer_provenance": {
            "kind": "utf8_bytes",
            "name": "utf8_bytes",
            "vocab_size": atomic.output_size,
            "eot_id": atomic.eot_id,
            "spec_sha256": atomic.sha256,
            "ngrams_sha256": None,
        },
        "loader_aligned": True,
        "physical_shard_tokens": len(tokens),
        "unique_stream_tokens": len(tokens),
    }
    (source / "mix_manifest.json").write_text(json.dumps(source_manifest))
    patcher_path = tmp_path / "entropy.patcher"
    patcher_artifact = _write_entropy_patcher(patcher_path, max_patch_size=5)

    built = build_dataset(
        output_dir=output,
        train_paths=(train,),
        validation_paths=(validation,),
        chunk_size=8,
        chunks_per_shard=2,
        document_aligned_pages=True,
        patching_policy_name="causal_entropy_v1",
        entropy_patcher_path=patcher_path,
    )

    assert built["schema"] == ENTROPY_DATASET_SCHEMA
    patching = built["patching"]
    assert patching["schema"] == PATCHING_POLICY_SCHEMA
    assert patching["name"] == "causal_entropy_v1"
    assert patching["patcher_artifact"] == {
        "path": "entropy-patcher.bdpatch",
        "input_path": str(patcher_path),
        "sha256": hashlib.sha256(patcher_artifact).hexdigest(),
        "bytes": len(patcher_artifact),
        "parameter_bytes": 16 * 261 * 4,
    }
    assert (output / "entropy-patcher.bdpatch").read_bytes() == patcher_artifact
    assert patching["boundary_config"]["max_patch_size"] == 5
    assert built["packing"]["patch_stride"] is None
    assert built["packing"]["chunk_alignment"].startswith("whole variable patches")
    assert built["patching_input_fingerprints"][0]["sha256"] == patching[
        "patcher_artifact"
    ]["sha256"]

    split = built["splits"]["train"]
    assert split["patches"] == 3
    assert split["mean_patch_size"] == pytest.approx(4.0)
    assert split["maximum_observed_patch_size"] == 5
    assert split["document_padding_tokens"] == 0
    assert split["patch_boundary_padding_tokens"] == 3
    arrays = _load_split_arrays(output, split)
    np.testing.assert_array_equal(
        arrays["valid_mask"][0],
        np.asarray([True] * 5 + [False] * 3),
    )
    np.testing.assert_array_equal(
        arrays["patch_offsets"],
        np.asarray(
            [
                [0, 1, 2, 3, 4, -1, -1, -1],
                [0, 1, 2, 3, 4, 0, 1, -1],
            ],
            dtype=np.int8,
        ),
    )
    # No row can start with a continuation offset, which would prove an
    # artificial chunk boundary had split a logical entropy patch.
    for row_offsets, row_valid in zip(
        arrays["patch_offsets"], arrays["valid_mask"], strict=True
    ):
        valid_offsets = row_offsets[row_valid]
        assert valid_offsets[0] == 0
    eot = arrays["input_ids"] == atomic.eot_id
    assert arrays["patch_offsets"][eot].tolist() == [1]


def test_entropy_policy_requires_document_aligned_byte_native_build(
    tmp_path: Path,
) -> None:
    path = tmp_path / "entropy.patcher"
    _write_entropy_patcher(path)
    with pytest.raises(ValueError, match="byte-native document_aligned_pages"):
        build_dataset(
            output_dir=tmp_path / "output",
            train_paths=(tmp_path / "missing-train.bin",),
            validation_paths=(tmp_path / "missing-val.bin",),
            chunk_size=8,
            chunks_per_shard=1,
            patching_policy_name="causal_entropy_v1",
            entropy_patcher_path=path,
        )


def test_cli_contract_parses_required_inputs() -> None:
    args = parse_args(
        [
            "--tokenizer",
            "tok",
            "--output",
            "out",
            "--train",
            "train-*.bin",
            "--validation",
            "val.bin",
            "--chunk-size",
            "512",
            "--chunks-per-shard",
            "4",
            "--require-terminal-eot",
        ]
    )

    assert args.tokenizer == Path("tok")
    assert args.chunk_size == 512
    assert args.chunks_per_shard == 4
    assert args.require_terminal_eot
    assert args.patching_policy == "fixed_stride_v1"
    assert args.entropy_patcher is None
