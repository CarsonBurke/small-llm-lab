from __future__ import annotations

import copy

import pytest

from pretraining.byte_diffusion.data import (
    AtomicCodec,
    AtomicDocument,
    AtomicIdManifest,
    DeterministicChunkCursor,
    SpecialAtom,
    pack_documents,
    packed_chunks_sha256,
)


@pytest.fixture
def manifest() -> AtomicIdManifest:
    return AtomicIdManifest.reference()


def test_manifest_roundtrip_preserves_explicit_special_order(
    manifest: AtomicIdManifest,
) -> None:
    payload = manifest.to_dict()
    restored = AtomicIdManifest.from_dict(payload)

    assert restored == manifest
    assert [record["atomic_id"] for record in payload["specials"]] == list(
        range(256, 261)
    )
    assert restored.eot_id == 256
    assert restored.mask_id == 261
    assert restored.pad_id == 262
    assert restored.output_size == 261
    assert restored.input_size == 263
    assert len(restored.sha256) == 64


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda payload: payload.update(specials={"eot": 256}), "ordered JSON list"),
        (
            lambda payload: payload["specials"].reverse(),
            "explicit ordered list",
        ),
        (
            lambda payload: payload["specials"][1].update(
                name=payload["specials"][0]["name"]
            ),
            "names must be unique",
        ),
        (lambda payload: payload.update(eot_id=261), "clean specials"),
        (lambda payload: payload.update(mask_id=262), "MASK"),
    ],
)
def test_manifest_rejects_ambiguous_or_incompatible_ids(
    manifest: AtomicIdManifest, mutate, match: str
) -> None:
    payload = copy.deepcopy(manifest.to_dict())
    mutate(payload)
    with pytest.raises(ValueError, match=match):
        AtomicIdManifest.from_dict(payload)


def test_manifest_does_not_coerce_fractional_ids(
    manifest: AtomicIdManifest,
) -> None:
    payload = manifest.to_dict()
    payload["specials"][0]["atomic_id"] = 256.5

    with pytest.raises(TypeError, match="integer"):
        AtomicIdManifest.from_dict(payload)


def test_atomic_codec_roundtrips_bytes_and_typed_specials(
    manifest: AtomicIdManifest,
) -> None:
    codec = AtomicCodec(manifest)
    parts = (
        "héllo".encode(),
        SpecialAtom("<think>"),
        b"\x00\xff",
        SpecialAtom("<|endoftext|>"),
    )

    encoded = codec.encode(parts)

    assert encoded == (
        *"héllo".encode(),
        257,
        0,
        255,
        256,
    )
    assert codec.decode(encoded) == parts
    assert codec.decode_bytes(b"raw") == b"raw"

    with pytest.raises(TypeError, match="bytes or SpecialAtom"):
        codec.encode(["implicit unicode"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="unknown atomic special"):
        codec.encode([SpecialAtom("<missing>")])
    with pytest.raises(ValueError, match="input-only"):
        codec.decode([manifest.mask_id])
    with pytest.raises(ValueError, match="clean specials"):
        codec.decode_bytes([manifest.eot_id])


def test_pack_documents_pads_locally_resets_phase_and_carries_halo(
    manifest: AtomicIdManifest,
) -> None:
    documents = (
        AtomicDocument("short", (ord("a"), ord("b"), manifest.eot_id)),
        AtomicDocument(
            "long",
            (
                ord("c"),
                ord("d"),
                ord("e"),
                ord("f"),
                ord("g"),
                manifest.eot_id,
            ),
        ),
    )

    chunks = pack_documents(documents, manifest, chunk_size=4)

    assert len(chunks) == 3
    first, middle, last = chunks

    # EOT closes the three-atom short document and its remaining patch slot is
    # storage PAD, not a target or a key.
    assert first.input_ids == (ord("a"), ord("b"), 256, 262)
    assert first.target_ids == (ord("b"), 256, 262, 262)
    assert first.valid_mask == (True, True, True, False)
    assert first.score_mask == (True, True, False, False)
    assert first.patch_offsets == (0, 1, 2, -1)
    assert not first.label_halo_valid

    # The next document restarts at patch offset zero.  Its artificial chunk
    # edge is aligned and carries exactly one shifted target as a halo.
    assert middle.stream_start == 4
    assert middle.input_ids == tuple(map(ord, "cdef"))
    assert middle.target_ids == tuple(map(ord, "defg"))
    assert middle.patch_offsets == (0, 1, 2, 3)
    assert middle.document_offsets == (0, 1, 2, 3)
    assert middle.label_halo_valid
    assert middle.label_halo_id == ord("g")
    assert middle.target_ids[-1] == middle.label_halo_id

    assert last.input_ids == (ord("g"), 256, 262, 262)
    assert last.target_ids == (256, 262, 262, 262)
    assert last.score_mask == (True, False, False, False)
    assert not last.label_halo_valid
    assert [span.document_key for chunk in chunks for span in chunk.document_spans] == [
        "short",
        "long",
        "long",
    ]


def test_packing_never_scores_across_document_or_storage_padding(
    manifest: AtomicIdManifest,
) -> None:
    documents = (
        AtomicDocument("one", (1, manifest.eot_id)),
        AtomicDocument("two", (2, manifest.eot_id)),
        AtomicDocument("three", (3, 4, manifest.eot_id)),
    )
    (chunk, second) = pack_documents(documents, manifest, chunk_size=8)

    assert chunk.input_ids == (1, 256, 262, 262, 2, 256, 262, 262)
    assert chunk.target_ids == (256, 262, 262, 262, 256, 262, 262, 262)
    assert chunk.score_mask == (True, False, False, False, True, False, False, False)
    assert not chunk.label_halo_valid
    assert second.input_ids == (3, 4, 256, 262, 262, 262, 262, 262)
    assert all(
        token == manifest.pad_id
        for token, valid in zip(second.input_ids, second.valid_mask, strict=True)
        if not valid
    )


@pytest.mark.parametrize(
    "document,match",
    [
        (AtomicDocument("missing", (1, 2)), "one EOT at its end"),
        (AtomicDocument("early", (256, 1, 256)), "one EOT at its end"),
        (AtomicDocument("input-only", (261, 256)), "clean atomic id"),
    ],
)
def test_packing_rejects_malformed_documents(
    manifest: AtomicIdManifest, document: AtomicDocument, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        pack_documents((document,), manifest, chunk_size=4)


def test_cursor_resume_is_exact_and_dataset_checked(
    manifest: AtomicIdManifest,
) -> None:
    documents = tuple(
        AtomicDocument(str(index), (index, index + 1, index + 2, 256))
        for index in range(12)
    )
    chunks = pack_documents(documents, manifest, chunk_size=4)
    uninterrupted = DeterministicChunkCursor(chunks, seed=716, shuffle=True)
    prefix = [uninterrupted.next_index() for _ in range(7)]
    state = uninterrupted.state_dict()
    expected_suffix = [uninterrupted.next_index() for _ in range(20)]

    resumed = DeterministicChunkCursor(chunks, seed=716, shuffle=True)
    resumed.load_state_dict(state)

    independent = DeterministicChunkCursor(chunks, seed=716, shuffle=True)
    assert [independent.next_index() for _ in range(7)] == prefix
    assert [resumed.next_index() for _ in range(20)] == expected_suffix

    altered_documents = documents[:-1] + (
        AtomicDocument("changed", (9, 8, 7, 256)),
    )
    altered_chunks = pack_documents(altered_documents, manifest, chunk_size=4)
    assert packed_chunks_sha256(chunks) != packed_chunks_sha256(altered_chunks)
    altered = DeterministicChunkCursor(altered_chunks, seed=716, shuffle=True)
    with pytest.raises(ValueError, match="dataset_sha256 mismatch"):
        altered.load_state_dict(state)


def test_cursor_state_at_epoch_end_resumes_before_next_shuffle(
    manifest: AtomicIdManifest,
) -> None:
    chunks = pack_documents(
        (
            AtomicDocument("a", (1, 2, 3, 256)),
            AtomicDocument("b", (4, 5, 6, 256)),
        ),
        manifest,
        chunk_size=4,
    )
    cursor = DeterministicChunkCursor(chunks, seed=5)
    first_epoch = [cursor.next_index() for _ in chunks]
    state = cursor.state_dict()
    expected = cursor.next_index()

    resumed = DeterministicChunkCursor(chunks, seed=5)
    resumed.load_state_dict(state)

    assert sorted(first_epoch) == [0, 1]
    assert resumed.next_index() == expected
    assert resumed.epoch == 1


def test_pack_requires_patch_aligned_chunk_size(manifest: AtomicIdManifest) -> None:
    document = AtomicDocument("doc", (1, manifest.eot_id))
    with pytest.raises(ValueError, match="multiple of four"):
        pack_documents((document,), manifest, chunk_size=6)
