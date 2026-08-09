"""CPU contracts for lossless Bolmo byteification and its streaming builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pretraining.bolmo_data import (
    BYTE_VOCAB_SIZE,
    ByteifiedTokens,
    ExpandedSuffixMatcher,
    SourceTokenByteifier,
    atomic_pad_id,
    atomic_special_id,
    collate_examples,
    decode_atomic_ids,
    load_example_shard,
    make_bolmo_example,
    truncate_to_complete_patches,
)
from scripts.build_bolmo_dataset import (
    CHALLENGE_HEADER_INTS,
    CHALLENGE_MAGIC,
    CHALLENGE_VERSION,
    build_dataset,
    canonical_sha256,
    iter_context_target_examples,
    iter_source_examples,
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


EXTRA_TEXT_TOKENS = (b"lo", b"hello", b"caf\xc3\xa9", "🙂".encode())


def test_byte_limit_truncates_only_at_complete_source_patches() -> None:
    source = [10, 11, 12]
    byteified = ByteifiedTokens(
        atomic_ids=(1, 2, 3, 4, 5, 6),
        patch_lengths=(2, 3, 1),
    )
    retained_source, retained_bytes = truncate_to_complete_patches(
        source,
        byteified,
        max_atomic_tokens=6,  # one BOS + exactly the first two patches
    )
    assert list(retained_source) == [10, 11]
    assert retained_bytes.atomic_ids == (1, 2, 3, 4, 5)
    assert retained_bytes.patch_lengths == (2, 3)


def test_validation_windows_share_one_context_token(tmp_path: Path) -> None:
    path = tmp_path / "validation.bin"
    write_challenge_shard(path, list(range(9)))
    windows = list(iter_context_target_examples([path], 4))
    assert [window.tolist() for window in windows] == [
        [0, 1, 2, 3, 4],
        [4, 5, 6, 7, 8],
    ]


def write_tiny_tokenizer(directory: Path) -> TokenizerSpec:
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
    return spec


@pytest.fixture
def tiny_tokenizer(tmp_path) -> tuple[Path, SplitTreeNumericTokenizer]:
    directory = tmp_path / "tokenizer"
    write_tiny_tokenizer(directory)
    return directory, SplitTreeNumericTokenizer.from_directory(directory)


def test_implicit_decimal_is_owned_by_first_fractional_patch(tiny_tokenizer) -> None:
    _, tokenizer = tiny_tokenizer
    text = "values: 12.034 and 9.10"
    source_ids = tokenizer.encode(text)
    byteified = SourceTokenByteifier(tokenizer).byteify(source_ids)

    assert decode_atomic_ids(byteified.atomic_ids, tokenizer) == text
    numeric = [
        (index, tokenizer.spec.numeric_tokens[token_id - tokenizer.spec.numeric_base])
        for index, token_id in enumerate(source_ids)
        if tokenizer.spec.numeric_base <= token_id < tokenizer.spec.text_base
    ]
    for source_index, (_, power) in numeric:
        if power != -1:
            continue
        start = sum(byteified.patch_lengths[:source_index])
        length = byteified.patch_lengths[source_index]
        assert byteified.atomic_ids[start : start + length][0] == ord(".")


def test_multibyte_utf8_and_all_specials_round_trip_atomically(
    tiny_tokenizer,
) -> None:
    _, tokenizer = tiny_tokenizer
    text = "café🙂<think>数学</think>"
    source_ids = tokenizer.encode(text)
    byteified = SourceTokenByteifier(tokenizer).byteify(source_ids)

    assert decode_atomic_ids(byteified.atomic_ids, tokenizer) == text
    assert len("café🙂数学".encode("utf-8")) < len(byteified.atomic_ids)
    for index, special in enumerate(tokenizer.spec.specials):
        special_bytes = SourceTokenByteifier(tokenizer).byteify(
            tokenizer.encode(special)
        )
        assert special_bytes.atomic_ids == (BYTE_VOCAB_SIZE + index,)
        assert special_bytes.patch_lengths == (1,)
    assert atomic_special_id(tokenizer.eot_id) == 256
    assert atomic_pad_id(tokenizer) == 256 + len(tokenizer.spec.specials)


def test_streaming_chunks_preserve_numeric_state_and_boundaries(
    tiny_tokenizer,
) -> None:
    _, tokenizer = tiny_tokenizer
    text = "before 123.0456 after"
    source_ids = tokenizer.encode(text)
    numeric_indices = [
        index
        for index, token_id in enumerate(source_ids)
        if tokenizer.spec.numeric_base <= token_id < tokenizer.spec.text_base
    ]
    split = numeric_indices[len(numeric_indices) // 2]

    one_shot = SourceTokenByteifier(tokenizer).byteify(source_ids)
    streaming = SourceTokenByteifier(tokenizer)
    first = streaming.byteify(source_ids[:split])
    second = streaming.byteify(source_ids[split:])
    reset_second = SourceTokenByteifier(tokenizer).byteify(source_ids[split:])

    assert first.atomic_ids + second.atomic_ids == one_shot.atomic_ids
    assert first.patch_lengths + second.patch_lengths == one_shot.patch_lengths
    assert sum(one_shot.boundary_mask) == len(source_ids)
    assert decode_atomic_ids(one_shot.atomic_ids, tokenizer) == text
    assert reset_second.atomic_ids[0] == ord(".")


def test_fractional_token_after_text_uses_total_decode_fallback(
    tiny_tokenizer,
) -> None:
    _, tokenizer = tiny_tokenizer
    text_id = tokenizer.encode("x")[0]
    fractional_id = tokenizer.spec.numeric_base + tokenizer.spec.numeric_tokens.index(
        ("5", -2)
    )
    byteified = SourceTokenByteifier(tokenizer).byteify([text_id, fractional_id])
    assert decode_atomic_ids(byteified.atomic_ids, tokenizer) == tokenizer.decode(
        [text_id, fractional_id]
    )
    assert byteified.atomic_ids[-2:] == (ord("."), ord("5"))


def test_eot_inside_numeric_run_matches_source_total_decoder(
    tiny_tokenizer,
) -> None:
    _, tokenizer = tiny_tokenizer
    numeric = tokenizer.spec.numeric_tokens
    source_ids = [
        tokenizer.spec.numeric_base + numeric.index(("6", -2)),
        tokenizer.eot_id,
        tokenizer.spec.numeric_base + numeric.index(("0", -3)),
        tokenizer.spec.numeric_base + numeric.index(("5", -4)),
    ]
    byteified = SourceTokenByteifier(tokenizer).byteify(source_ids)
    assert decode_atomic_ids(byteified.atomic_ids, tokenizer) == tokenizer.decode(
        source_ids
    )
    assert byteified.atomic_ids[-3:] == (ord("."), ord("0"), ord("5"))


def test_expanded_ids_are_longest_causal_fixed_surfaces_only(
    tiny_tokenizer,
) -> None:
    _, tokenizer = tiny_tokenizer
    source_ids = tokenizer.encode("hello 12.30")
    byteified = SourceTokenByteifier(tokenizer).byteify(source_ids)
    source_model_vocab = (tokenizer.vocab_size + 127) // 128 * 128
    matcher = ExpandedSuffixMatcher(
        tokenizer, source_model_vocab_size=source_model_vocab
    )
    expanded = matcher.match(byteified.atomic_ids)

    hello_id = tokenizer.spec.text_base + tokenizer.spec.text_tokens.index(b"hello")
    hello_end = bytes(byteified.atomic_ids).index(b"hello") + len(b"hello") - 1
    assert expanded[hello_end] == hello_id
    assert all(
        value == source_model_vocab
        or value < tokenizer.spec.numeric_base
        or value >= tokenizer.spec.text_base
        for value in expanded
    ), "semantic numeric ids must never enter the expanded embedding channel"

    lo_id = tokenizer.spec.text_base + tokenizer.spec.text_tokens.index(b"lo")
    separated = (*b"hel", atomic_special_id(tokenizer.eot_id), *b"lo")
    separated_expanded = matcher.match(separated)
    assert separated_expanded[3] == tokenizer.eot_id
    assert separated_expanded[-1] == lo_id
    assert separated_expanded[-1] != hello_id


def test_suffix_automaton_matches_longest_surface_oracle_with_resets(
    tiny_tokenizer,
) -> None:
    _, tokenizer = tiny_tokenizer
    source_model_vocab = (tokenizer.vocab_size + 127) // 128 * 128
    matcher = ExpandedSuffixMatcher(
        tokenizer, source_model_vocab_size=source_model_vocab
    )
    atoms = (*b"hellohello caf\xc3\xa9", 256, *b"hello")
    observed = matcher.match(atoms)

    history = bytearray()
    expected = []
    fixed = tuple(enumerate(tokenizer.spec.text_tokens, tokenizer.spec.text_base))
    for atom in atoms:
        if atom >= BYTE_VOCAB_SIZE:
            history.clear()
            expected.append(atom - BYTE_VOCAB_SIZE)
            continue
        history.append(atom)
        candidates = [
            (len(surface), token_id)
            for token_id, surface in fixed
            if surface and history.endswith(surface)
        ]
        expected.append(
            max(candidates)[1] if candidates else source_model_vocab
        )
    assert observed == tuple(expected)


def test_collation_prepends_eot_and_pads_byte_axis_for_tfla(
    tiny_tokenizer,
) -> None:
    _, tokenizer = tiny_tokenizer
    source = tokenizer.encode("café 0.125")
    byteified = SourceTokenByteifier(tokenizer).byteify(source)
    source_model_vocab = (tokenizer.vocab_size + 127) // 128 * 128
    matcher = ExpandedSuffixMatcher(
        tokenizer, source_model_vocab_size=source_model_vocab
    )
    example = make_bolmo_example(
        source,
        byteified,
        tokenizer,
        matcher,
        expected_real_source_tokens=len(source) + 3,
        source_model_vocab_size=source_model_vocab,
    )
    payload = collate_examples(
        [example],
        tokenizer,
        source_model_vocab_size=source_model_vocab,
        byte_length_multiple=128,
    )

    assert payload["source_ids"].shape == (1, len(source) + 4)
    assert payload["source_ids"][0, 0].item() == tokenizer.eot_id
    assert payload["byte_ids"][0, 0].item() == 256
    assert payload["byte_ids"].shape[1] % 128 == 0
    assert payload["source_ids"][0, -1].item() == source_model_vocab
    assert payload["expanded_ids"][0, -1].item() == source_model_vocab
    assert not payload["source_valid_mask"][0, -1]
    assert not payload["valid_mask"][0, -1]
    assert payload["boundary_mask"].sum() == len(source) + 1


def test_boundary_before_internal_eot_is_coalesced(tiny_tokenizer) -> None:
    _, tokenizer = tiny_tokenizer
    source = [*tokenizer.encode("left"), tokenizer.eot_id, *tokenizer.encode("right")]
    byteifier = SourceTokenByteifier(tokenizer)
    byteified = byteifier.byteify(source)
    source_model_vocab = (tokenizer.vocab_size + 127) // 128 * 128
    matcher = ExpandedSuffixMatcher(
        tokenizer, source_model_vocab_size=source_model_vocab
    )
    example = make_bolmo_example(
        source,
        byteified,
        tokenizer,
        matcher,
        expected_real_source_tokens=len(source),
        source_model_vocab_size=source_model_vocab,
    )
    eot_position = example.byte_ids.index(256, 1)
    assert not example.boundary_mask[eot_position - 1]
    assert example.boundary_mask[eot_position]
    assert sum(example.boundary_mask) == len(source)


def write_challenge_shard(path: Path, tokens: list[int]) -> None:
    header = np.zeros(CHALLENGE_HEADER_INTS, dtype="<i4")
    header[0] = CHALLENGE_MAGIC
    header[1] = CHALLENGE_VERSION
    header[2] = len(tokens)
    with path.open("wb") as handle:
        handle.write(header.tobytes())
        handle.write(np.asarray(tokens, dtype="<u2").tobytes())


def tokenizer_manifest(tokenizer_dir: Path, tokenizer) -> dict:
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
    }


def test_builder_streams_overlapping_shards_and_hashes_every_artifact(
    tiny_tokenizer, tmp_path
) -> None:
    tokenizer_dir, tokenizer = tiny_tokenizer
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    train = [
        tokenizer.eot_id,
        *tokenizer.encode("A 123.045 café🙂"),
        tokenizer.eot_id,
        *tokenizer.encode("tail"),
    ]
    powers = [
        tokenizer.spec.numeric_tokens[token - tokenizer.spec.numeric_base][1]
        if tokenizer.spec.numeric_base <= token < tokenizer.spec.text_base
        else None
        for token in train
    ]
    cut = powers.index(-1) + 1
    first, second = train[:cut], train[cut - 1 :]
    train_a = source_dir / "fineweb_train_000000.bin"
    train_b = source_dir / "fineweb_train_000001.bin"
    validation = source_dir / "fineweb_val_000000.bin"
    write_challenge_shard(train_a, first)
    write_challenge_shard(train_b, second)
    write_challenge_shard(
        validation, [tokenizer.eot_id, *tokenizer.encode("validation 0.50")]
    )
    source_manifest = tokenizer_manifest(tokenizer_dir, tokenizer)
    source_manifest.update(
        {
            "physical_shard_tokens": len(first) + len(second),
            "unique_stream_tokens": len(train),
        }
    )
    (source_dir / "mix_manifest.json").write_text(json.dumps(source_manifest))

    output = tmp_path / "bolmo"
    source_model_vocab = (tokenizer.vocab_size + 127) // 128 * 128
    manifest = build_dataset(
        tokenizer_dir=tokenizer_dir,
        output_dir=output,
        train_paths=[train_a, train_b],
        validation_paths=[validation],
        domain_validation_paths={},
        source_tokens_per_example=4,
        examples_per_shard=2,
        source_model_vocab_size=source_model_vocab,
        byte_length_multiple=128,
    )

    assert manifest["schema"] == "bolmo_byte_dataset/v3"
    raw_windows = list(
        iter_source_examples(
            [train_a, train_b], 4, overlap_tokens=1, include_tail=True
        )
    )
    expected_rows = [window[:-1].tolist() for window in raw_windows if len(window) >= 2]
    expected_source = [token for row in expected_rows for token in row]
    assert manifest["splits"]["train"]["input_source_tokens"] == sum(
        len(window) for window in raw_windows
    )
    assert manifest["splits"]["train"]["real_source_tokens"] == len(
        expected_source
    )
    assert manifest["splits"]["train"]["paper_skipped_source_tokens"] == len(
        expected_rows
    )
    assert manifest["splits"]["train"]["overlap_tokens_per_transition"] == 1
    assert manifest["source_tokenizer"]["logical_vocab_size"] == tokenizer.vocab_size
    assert manifest["source_tokenizer"]["source_model_vocab_size"] == source_model_vocab
    assert manifest["examples"]["byte_length_multiple"] == 128

    recovered_source: list[int] = []
    recovered_rows: list[list[int]] = []
    for artifact in manifest["splits"]["train"]["artifacts"]:
        path = output / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["byte_ids"].shape[1] % 128 == 0
        for row in range(payload["source_ids"].shape[0]):
            source_valid = payload["source_valid_mask"][row]
            recovered_source.extend(
                payload["source_ids"][row, 1:][source_valid[1:]].tolist()
            )
            byte_valid = payload["valid_mask"][row]
            atoms = payload["byte_ids"][row][byte_valid].tolist()[1:]
            source_row = payload["source_ids"][row, 1:][source_valid[1:]].tolist()
            assert decode_atomic_ids(atoms, tokenizer) == tokenizer.decode(source_row)
            recovered_rows.append(source_row)
    assert recovered_source == expected_source
    assert recovered_rows == expected_rows
    saved_manifest = json.loads((output / "manifest.json").read_text())
    assert saved_manifest["payload_sha256"] == manifest["payload_sha256"]
    claimed_hash = saved_manifest.pop("payload_sha256")
    assert canonical_sha256(saved_manifest) == claimed_hash
    for fingerprint in manifest["input_fingerprints"]:
        path = Path(fingerprint["path"])
        assert path.stat().st_size == fingerprint["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == fingerprint["sha256"]


def test_shard_loader_rejects_a_boundary_mask_inconsistent_with_patches(
    tiny_tokenizer, tmp_path
) -> None:
    _, tokenizer = tiny_tokenizer
    source = tokenizer.encode("hello 1.25")
    byteified = SourceTokenByteifier(tokenizer).byteify(source)
    source_model_vocab = (tokenizer.vocab_size + 127) // 128 * 128
    matcher = ExpandedSuffixMatcher(
        tokenizer, source_model_vocab_size=source_model_vocab
    )
    example = make_bolmo_example(
        source,
        byteified,
        tokenizer,
        matcher,
        expected_real_source_tokens=len(source),
        source_model_vocab_size=source_model_vocab,
    )
    payload = collate_examples(
        [example],
        tokenizer,
        source_model_vocab_size=source_model_vocab,
    )
    payload["boundary_mask"].zero_()
    path = tmp_path / "corrupt.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="boundary mask"):
        load_example_shard(path)


def test_shard_loader_rejects_padded_vocab_gap_and_wrong_bos(
    tiny_tokenizer, tmp_path
) -> None:
    _, tokenizer = tiny_tokenizer
    source = tokenizer.encode("hello")
    source_model_vocab = (tokenizer.vocab_size + 127) // 128 * 128
    example = make_bolmo_example(
        source,
        SourceTokenByteifier(tokenizer).byteify(source),
        tokenizer,
        ExpandedSuffixMatcher(
            tokenizer, source_model_vocab_size=source_model_vocab
        ),
        expected_real_source_tokens=len(source),
        source_model_vocab_size=source_model_vocab,
    )
    payload = collate_examples(
        [example], tokenizer, source_model_vocab_size=source_model_vocab
    )
    payload["source_ids"][0, 1] = tokenizer.vocab_size
    gap_path = tmp_path / "gap.pt"
    torch.save(payload, gap_path)
    with pytest.raises(ValueError, match="outside the source vocabulary"):
        load_example_shard(
            gap_path,
            source_vocab_size=tokenizer.vocab_size,
            source_pad_id=source_model_vocab,
            atomic_vocab_size=atomic_pad_id(tokenizer),
        )

    payload["source_ids"][0, 1] = source[0]
    payload["byte_ids"][0, 0] = ord("x")
    bos_path = tmp_path / "bos.pt"
    torch.save(payload, bos_path)
    with pytest.raises(ValueError, match="synthetic EOT/BOS id 256"):
        load_example_shard(bos_path)


def test_builder_bounds_each_split_exactly_and_resets_numeric_windows(
    tiny_tokenizer, tmp_path
) -> None:
    tokenizer_dir, tokenizer = tiny_tokenizer
    source_dir = tmp_path / "bounded_source"
    source_dir.mkdir()
    train = [
        tokenizer.eot_id,
        *tokenizer.encode("123.45 carries into another sufficiently long window"),
    ]
    validation = [
        tokenizer.eot_id,
        *tokenizer.encode("validation has more than one source window"),
    ]
    domain = [
        tokenizer.eot_id,
        *tokenizer.encode("domain validation also has multiple windows"),
    ]
    train_path = source_dir / "train.bin"
    validation_path = source_dir / "validation.bin"
    domain_path = source_dir / "domain_math.bin"
    write_challenge_shard(train_path, train)
    write_challenge_shard(validation_path, validation)
    write_challenge_shard(domain_path, domain)
    (source_dir / "mix_manifest.json").write_text(
        json.dumps(tokenizer_manifest(tokenizer_dir, tokenizer))
    )

    output = tmp_path / "bounded_bolmo"
    source_model_vocab = (tokenizer.vocab_size + 127) // 128 * 128
    manifest = build_dataset(
        tokenizer_dir=tokenizer_dir,
        output_dir=output,
        train_paths=[train_path],
        validation_paths=[validation_path],
        domain_validation_paths={"math": [domain_path]},
        source_tokens_per_example=4,
        examples_per_shard=2,
        source_model_vocab_size=source_model_vocab,
        max_train_examples=2,
        max_validation_examples=1,
        max_domain_validation_examples=1,
    )

    assert manifest["examples"]["max_train_examples"] == 2
    assert manifest["examples"]["max_validation_examples"] == 1
    assert manifest["examples"]["max_domain_validation_examples"] == 1
    assert manifest["splits"]["train"]["examples"] == 2
    assert manifest["splits"]["train"]["input_source_tokens"] == 8
    assert manifest["splits"]["train"]["real_source_tokens"] == 6
    assert manifest["splits"]["train"]["paper_skipped_source_tokens"] == 2
    assert manifest["splits"]["train"]["truncated_by_max_examples"]
    assert manifest["splits"]["validation"]["examples"] == 1
    assert manifest["splits"]["validation"]["real_source_tokens"] == 5
    assert manifest["splits"]["validation"]["scored_source_tokens"] == 4
    assert manifest["splits"]["validation"]["truncated_by_max_examples"]
    assert manifest["splits"]["domainval_math"]["examples"] == 1
    assert manifest["splits"]["domainval_math"]["truncated_by_max_examples"]

    train_artifact = manifest["splits"]["train"]["artifacts"][0]
    payload = load_example_shard(output / train_artifact["path"])
    recovered_source: list[int] = []
    for row in range(2):
        source_valid = payload["source_valid_mask"][row]
        recovered_source.extend(
            payload["source_ids"][row, 1:][source_valid[1:]].tolist()
        )
        byte_valid = payload["valid_mask"][row]
        atoms = payload["byte_ids"][row][byte_valid].tolist()[1:]
        source_row = payload["source_ids"][row, 1:][source_valid[1:]].tolist()
        assert decode_atomic_ids(atoms, tokenizer) == tokenizer.decode(source_row)
    assert recovered_source == [*train[:3], *train[4:7]]


def test_builder_resets_fractional_tst_state_at_every_bos_row(
    tiny_tokenizer, tmp_path
) -> None:
    tokenizer_dir, tokenizer = tiny_tokenizer
    numeric = tokenizer.spec.numeric_tokens
    source_ids = [
        tokenizer.spec.numeric_base + numeric.index((digit, power))
        for digit, power in zip("1234567", range(1, -6, -1), strict=True)
    ]
    source_dir = tmp_path / "numeric_source"
    source_dir.mkdir()
    train_path = source_dir / "train.bin"
    validation_path = source_dir / "validation.bin"
    write_challenge_shard(train_path, source_ids)
    write_challenge_shard(validation_path, source_ids)
    (source_dir / "mix_manifest.json").write_text(
        json.dumps(tokenizer_manifest(tokenizer_dir, tokenizer))
    )
    output = tmp_path / "numeric_bolmo"
    source_model_vocab = (tokenizer.vocab_size + 127) // 128 * 128
    manifest = build_dataset(
        tokenizer_dir=tokenizer_dir,
        output_dir=output,
        train_paths=[train_path],
        validation_paths=[validation_path],
        domain_validation_paths={},
        source_tokens_per_example=3,
        examples_per_shard=4,
        source_model_vocab_size=source_model_vocab,
        max_validation_examples=2,
    )
    validation_artifact = manifest["splits"]["validation"]["artifacts"][0]
    payload = load_example_shard(
        output / validation_artifact["path"], context_source_tokens=1
    )
    assert payload["source_ids"].shape[0] == 2
    for row in range(2):
        source_valid = payload["source_valid_mask"][row]
        source_row = payload["source_ids"][row, 1:][source_valid[1:]].tolist()
        byte_valid = payload["valid_mask"][row]
        atoms = payload["byte_ids"][row][byte_valid].tolist()[1:]
        assert decode_atomic_ids(atoms, tokenizer) == tokenizer.decode(source_row)
    second_atoms = payload["byte_ids"][1][payload["valid_mask"][1]].tolist()
    assert second_atoms[1] == ord(".")


def test_shard_loader_rejects_score_masks_that_split_patches_or_context(
    tiny_tokenizer, tmp_path
) -> None:
    _, tokenizer = tiny_tokenizer
    source = tokenizer.encode("hello 1.25")
    byteified = SourceTokenByteifier(tokenizer).byteify(source)
    source_model_vocab = (tokenizer.vocab_size + 127) // 128 * 128
    matcher = ExpandedSuffixMatcher(
        tokenizer, source_model_vocab_size=source_model_vocab
    )
    example = make_bolmo_example(
        source,
        byteified,
        tokenizer,
        matcher,
        expected_real_source_tokens=len(source),
        source_model_vocab_size=source_model_vocab,
        context_source_tokens=1,
    )
    payload = collate_examples(
        [example], tokenizer, source_model_vocab_size=source_model_vocab
    )
    path = tmp_path / "context.pt"
    torch.save(payload, path)
    load_example_shard(path, context_source_tokens=1)
    with pytest.raises(ValueError, match="split contract"):
        load_example_shard(path, context_source_tokens=0)

    score_start = int(torch.nonzero(payload["score_mask"][0])[0])
    payload["score_mask"][0, score_start - 1] = True
    corrupt_path = tmp_path / "split_patch.pt"
    torch.save(payload, corrupt_path)
    with pytest.raises(ValueError, match="patch boundary"):
        load_example_shard(corrupt_path, context_source_tokens=1)


def test_source_window_iterator_rejects_a_false_overlap(
    tiny_tokenizer, tmp_path
) -> None:
    _, tokenizer = tiny_tokenizer
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    write_challenge_shard(first, [tokenizer.eot_id, *tokenizer.encode("first")])
    write_challenge_shard(second, [tokenizer.eot_id, *tokenizer.encode("second")])
    with pytest.raises(ValueError, match="overlap does not match"):
        list(
            iter_source_examples(
                [first, second], 4, overlap_tokens=1, include_tail=True
            )
        )
