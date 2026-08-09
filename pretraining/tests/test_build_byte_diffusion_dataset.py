from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pretraining.bolmo_data import SourceTokenByteifier
from pretraining.byte_diffusion.data import AtomicDocument, AtomicIdManifest, pack_documents
from pretraining.byte_diffusion.training import chunks_to_batch, load_data_directory
from scripts.build_bolmo_dataset import (
    CHALLENGE_HEADER_INTS,
    CHALLENGE_MAGIC,
    CHALLENGE_VERSION,
    canonical_sha256,
)
from scripts.build_byte_diffusion_dataset import (
    ARTIFACT_SCHEMA,
    DATASET_SCHEMA,
    ChallengeDocumentReader,
    StreamingDocumentPacker,
    build_dataset,
    parse_args,
    validate_atomic_utf8,
    write_deterministic_npz,
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
        with np.load(output / artifact["path"], allow_pickle=False) as payload:
            for name in payload.files:
                rows.setdefault(name, []).append(payload[name].copy())
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

    expected = tuple(
        chunk
        for document in documents
        for chunk in pack_documents((document,), manifest, chunk_size=8)
    )
    streaming = StreamingDocumentPacker(manifest, chunk_size=8)
    actual = []
    for index, document in enumerate(documents):
        actual.extend(streaming.add_document(index, document))
    actual.extend(streaming.finish())

    assert len(actual) == len(expected)
    for observed, reference in zip(actual, expected, strict=True):
        assert observed.input_ids == reference.input_ids
        assert observed.target_ids == reference.target_ids
        assert observed.valid_mask == reference.valid_mask
        assert observed.score_mask == reference.score_mask
        assert observed.document_offsets == reference.document_offsets
        assert len({value for value in observed.document_indices if value >= 0}) == 1


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
    # Every new document resets its patch offset in its independent row.
    for document_index in range(3):
        positions = train_arrays["document_indices"] == document_index
        assert train_arrays["patch_offsets"][positions][0] == 0
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
    assert all(
        len({value for value in chunk.document_indices if value >= 0}) == 1
        for chunk in (*train_chunks, *validation_chunks)
    )
    native_batch = train_chunks.training_batch([0, min(1, len(train_chunks) - 1)])
    reference_batch = chunks_to_batch(
        [train_chunks[0], train_chunks[min(1, len(train_chunks) - 1)]]
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
    for field in ("ids", "valid", "ar_targets", "bos_targets", "positions"):
        torch.testing.assert_close(
            getattr(native_batch, field), getattr(reference_batch, field)
        )
    validation_batch, identities = validation_chunks.validation_batch([0])
    assert validation_batch.ids.shape[0] == 1
    assert identities[0].chunk_index == validation_chunks[0].chunk_index
    assert identities[0].stream_start == validation_chunks[0].stream_start
    _, diffusion_train, diffusion_validation = load_data_directory(
        first_output,
        chunk_size=8,
        recipe="canvas",
        required_branch_bytes=8,
    )
    # Diffusion eligibility must never filter AR/BPB rows. The two-atom "A"
    # document and short validation documents remain present and use padded
    # short canvases at objective construction time.
    assert len(diffusion_train) == len(train_chunks)
    assert len(diffusion_validation) == len(validation_chunks)
    assert min(sum(chunk.valid_mask) for chunk in diffusion_train) < 8
    adaptive = diffusion_train.training_batches(
        list(range(len(diffusion_train))),
        max_batch_size=2,
        physical_token_budget=16,
    )
    assert sum(batch.ids.shape[0] for batch in adaptive) == len(diffusion_train)
    assert all(
        batch.ids.shape[0] * (batch.ids.shape[1] + 8) <= 16
        for batch in adaptive
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
