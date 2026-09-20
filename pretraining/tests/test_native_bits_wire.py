"""Observable raw-bit packet contracts; no neural models or GPU dependencies."""

import struct
from hashlib import sha256
from itertools import product

import pytest

from pretraining.nanogpt_mini.native_bits_wire import (
    HEADER_BYTES,
    MAGIC,
    MAX_COUNT,
    MAX_PACKET_BYTES,
    pack_packet,
    payload_bit_count,
    unpack_packet,
)

CHECKPOINT = sha256(b"checkpoint").hexdigest()
SOURCE = sha256("\0cafe\u0301\n🙂".encode("utf-8")).hexdigest()


def header(count=0, latent_bits=3, identity_bits=2, magic=MAGIC):
    return struct.pack(
        ">4s32s32sQHH",
        magic,
        bytes.fromhex(CHECKPOINT),
        bytes.fromhex(SOURCE),
        count,
        latent_bits,
        identity_bits,
    )


def pack(latents, residual, latent_bits=3, identity_bits=2, **kwargs):
    return pack_packet(
        latents,
        residual,
        kwargs.get("checkpoint_sha256", CHECKPOINT),
        kwargs.get("source_sha256", SOURCE),
        latent_bits=latent_bits,
        identity_bits=identity_bits,
    )


def test_exhaustive_small_bit_matrices_roundtrip():
    for count in range(3):
        for latent_bits, identity_bits in product(range(1, 4), range(4)):
            total_bits = count * (latent_bits + identity_bits)
            for bits in product((0, 1), repeat=total_bits):
                boundary = count * latent_bits
                latents = [
                    list(bits[start : start + latent_bits])
                    for start in range(0, boundary, latent_bits)
                ]
                residual = [
                    list(
                        bits[
                            boundary + row * identity_bits : boundary
                            + (row + 1) * identity_bits
                        ]
                    )
                    for row in range(count)
                ]
                payload = pack(latents, residual, latent_bits, identity_bits)
                assert unpack_packet(payload) == {
                    "latents": latents,
                    "residual": residual,
                    "checkpoint_sha256": CHECKPOINT,
                    "source_sha256": SOURCE,
                    "latent_bits": latent_bits,
                    "identity_bits": identity_bits,
                }
                assert payload_bit_count(payload) == total_bits
                assert len(payload) == HEADER_BYTES + (total_bits + 7) // 8


def test_independent_wire_fixture_msb_order_and_no_row_padding():
    # 101,011 latent rows followed by 10,01 residual rows -> 10101110 01000000.
    expected = header(count=2) + b"\xae\x40"
    assert pack([[1, 0, 1], [0, 1, 1]], [[1, 0], [0, 1]]) == expected
    decoded = unpack_packet(expected)
    assert decoded["latents"] == [[1, 0, 1], [0, 1, 1]]
    assert decoded["residual"] == [[1, 0], [0, 1]]


def test_pure_latent_packet_has_no_dummy_residual_bits():
    latents = [[1, 0, 1], [0, 1, 1], [1, 0, 0]]
    expected = header(count=3, identity_bits=0) + b"\xae\x00"
    assert pack(latents, [[], [], []], identity_bits=0) == expected
    decoded = unpack_packet(expected)
    assert decoded["latents"] == latents
    assert decoded["residual"] == [[], [], []]
    assert decoded["identity_bits"] == 0
    assert payload_bit_count(expected) == 9


def test_pure_latent_packet_rejects_truncation_padding_and_trailing_bytes():
    payload = pack([[1, 0, 1], [0, 1, 1], [1, 0, 0]], [[], [], []], identity_bits=0)
    malformed = [
        *(payload[:size] for size in range(len(payload))),
        payload[:-1] + b"\x01",
        payload + b"\x00",
    ]
    for frame in malformed:
        for operation in (unpack_packet, payload_bit_count):
            with pytest.raises(ValueError):
                operation(frame)


@pytest.mark.parametrize("residual", [[], [[]], [[], [0]], [None, []]])
def test_pure_latent_packet_requires_an_empty_residual_row_per_symbol(residual):
    with pytest.raises(ValueError):
        pack([[1, 0, 1], [0, 1, 1]], residual, identity_bits=0)


@pytest.mark.parametrize(
    "latent_bits,identity_bits", [(1, 0), (1, 1), (256, 0), (256, 256)]
)
def test_width_boundaries_and_non_byte_aligned_channels(latent_bits, identity_bits):
    latents = [
        [i % 2 for i in range(latent_bits)],
        [1] * latent_bits,
        [0] * latent_bits,
    ]
    residual = [
        [int(i % 3 == 0) for i in range(identity_bits)],
        [0] * identity_bits,
        [1] * identity_bits,
    ]
    payload = pack(latents, residual, latent_bits, identity_bits)
    recovered = unpack_packet(payload)
    assert recovered["latents"] == latents
    assert recovered["residual"] == residual
    assert payload_bit_count(payload) == 3 * (latent_bits + identity_bits)


@pytest.mark.parametrize("identity_bits", [0, 17])
def test_empty_source_preserves_explicit_widths_and_digests(identity_bits):
    payload = pack([], [], 8, identity_bits)
    assert payload == header(0, 8, identity_bits)
    decoded = unpack_packet(payload)
    assert decoded["latents"] == decoded["residual"] == []
    assert decoded["latent_bits"] == 8
    assert decoded["identity_bits"] == identity_bits
    assert decoded["checkpoint_sha256"] == CHECKPOINT
    assert decoded["source_sha256"] == SOURCE
    assert payload_bit_count(payload) == 0


def test_bool_bits_are_exact_and_digests_accept_hex_case():
    payload = pack(
        [[True, False, True]], [[False, True]], checkpoint_sha256=CHECKPOINT.upper()
    )
    decoded = unpack_packet(payload)
    assert decoded["latents"] == [[1, 0, 1]]
    assert decoded["residual"] == [[0, 1]]
    assert decoded["checkpoint_sha256"] == CHECKPOINT


def test_xor_residuals_recover_opaque_id_bits_despite_wrong_predictions():
    original = [[0, 0, 1], [1, 1, 0], [1, 0, 1]]
    predictions = [[1, 1, 0], [0, 0, 0], [1, 0, 1]]
    residual = [
        [actual ^ guess for actual, guess in zip(row, guessed)]
        for row, guessed in zip(original, predictions)
    ]
    payload = pack([[0], [0], [0]], residual, 1, 3)
    decoded = unpack_packet(payload)
    recovered = [
        [guess ^ error for guess, error in zip(row, errors)]
        for row, errors in zip(predictions, decoded["residual"])
    ]
    assert recovered == original


@pytest.mark.parametrize("bit", [2, 1.0])
@pytest.mark.parametrize("channel", ["latents", "residual"])
def test_nonbinary_and_lossy_bit_casts_are_rejected(bit, channel):
    latents = [[0, 0, 0]]
    residual = [[0, 0]]
    (latents if channel == "latents" else residual)[0][0] = bit
    with pytest.raises(ValueError):
        pack(latents, residual)


@pytest.mark.parametrize(
    "latents,residual",
    [
        ([[0, 1, 0]], []),
        ([[0, 1]], [[0, 1]]),
        ([[0, 1, 0]], [[0]]),
        ([None], [[0, 1]]),
        (None, []),
    ],
)
def test_mismatched_counts_and_ragged_rows_are_rejected(latents, residual):
    with pytest.raises((TypeError, ValueError)):
        pack(latents, residual)


@pytest.mark.parametrize(
    "name,width",
    [
        ("latent_bits", 0),
        ("latent_bits", -1),
        ("latent_bits", 257),
        ("latent_bits", 1.0),
        ("latent_bits", True),
        ("identity_bits", -1),
        ("identity_bits", 257),
        ("identity_bits", 1.0),
        ("identity_bits", True),
    ],
)
def test_encoder_rejects_widths_outside_integer_range(width, name):
    kwargs = {"latent_bits": 3, "identity_bits": 2, name: width}
    with pytest.raises(ValueError):
        pack([], [], **kwargs)


@pytest.mark.parametrize("digest", ["00" * 31, "g0" * 32, " " * 64])
@pytest.mark.parametrize("name", ["checkpoint_sha256", "source_sha256"])
def test_encoder_rejects_malformed_digest(digest, name):
    with pytest.raises(ValueError):
        pack([], [], **{name: digest})


@pytest.mark.parametrize(
    "payload",
    [
        header(magic=b"NB02"),
        header(count=MAX_COUNT + 1),
        header(latent_bits=0),
        header(latent_bits=257),
        header(identity_bits=257),
        header(count=2) + b"\xae",  # missing final source bits
        header(count=2) + b"\xae\x41",  # nonzero alignment padding
        header() + b"\x00",  # trailing byte, even if all zero
        header(count=2) + b"\xae\x40\x00",  # trailing byte after valid payload
    ],
)
def test_malformed_frames_are_rejected_by_unpack_and_bit_counter(payload):
    for operation in (unpack_packet, payload_bit_count):
        with pytest.raises(ValueError):
            operation(payload)


def test_every_byte_truncation_is_rejected():
    payload = pack([[1, 0, 1], [0, 1, 1]], [[1, 0], [0, 1]])
    for size in range(len(payload)):
        with pytest.raises(ValueError):
            unpack_packet(payload[:size])


def test_maximum_count_header_is_supported_without_allocating_decoded_rows():
    payload = header(MAX_COUNT, 1, 1) + bytes((MAX_COUNT * 2 + 7) // 8)
    assert payload_bit_count(payload) == MAX_COUNT * 2


def test_encoder_rejects_oversized_count_before_accessing_rows():
    rows = [None] * (MAX_COUNT + 1)
    with pytest.raises(ValueError):
        pack(rows, rows)


def test_decoder_rejects_input_larger_than_resource_limit():
    oversized = bytes(MAX_PACKET_BYTES + 1)
    for operation in (unpack_packet, payload_bit_count):
        with pytest.raises(ValueError):
            operation(oversized)
