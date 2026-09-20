"""Lossless, raw bit transport for native-bit latents and identity residuals.

This is bit packing, not entropy coding. NB01 binds opaque bits to a checkpoint
and original UTF-8 source through SHA-256 digests; digest verification belongs
to the caller that has the checkpoint and reconstructed source bytes.
"""

from __future__ import annotations

import struct

MAGIC = b"NB01"
MAX_COUNT = 1_000_000
MAX_PACKET_BYTES = 64 * 1024 * 1024
_HEADER = struct.Struct(">4s32s32sQHH")
HEADER_BYTES = _HEADER.size


def _width(value: int, name: str, minimum: int = 1) -> int:
    if type(value) is not int or not minimum <= value <= 256:
        raise ValueError(f"{name} must be an integer in {minimum}..256")
    return value


def _digest(value: str, name: str) -> bytes:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 hex digest") from exc
    if len(raw) != 32:
        raise ValueError(f"{name} must be a SHA-256 hex digest without whitespace")
    return raw


def _pack_rows(
    output: bytearray,
    rows: list[list[int]],
    width: int,
    position: int,
    name: str,
) -> int:
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise ValueError(f"each {name} row must have exactly {width} bits")
        for bit in row:
            if not isinstance(bit, (bool, int)) or bit not in (0, 1):
                raise ValueError(f"{name} must contain only bool or integer 0/1 bits")
            output[position // 8] |= bit << (7 - position % 8)
            position += 1
    return position


def pack_packet(
    latents: list[list[int]],
    residual: list[list[int]],
    checkpoint_sha256: str,
    source_sha256: str,
    *,
    latent_bits: int,
    identity_bits: int,
) -> bytes:
    """Pack all latent rows, then all XOR-residual rows, MSB first.

    Widths are explicit even for empty sources. Residual rows correspond to
    opaque identity-bit predictions, not to UTF-8 byte or Unicode codepoint bits.
    Pure-latent packets use identity_bits=0 and one empty residual row per symbol.
    """
    latent_bits = _width(latent_bits, "latent_bits")
    identity_bits = _width(identity_bits, "identity_bits", minimum=0)
    checkpoint_digest = _digest(checkpoint_sha256, "checkpoint_sha256")
    source_digest = _digest(source_sha256, "source_sha256")
    if not isinstance(latents, (list, tuple)) or not isinstance(
        residual, (list, tuple)
    ):
        raise TypeError("latents and residual must be row sequences")
    count = len(latents)
    if count != len(residual):
        raise ValueError("latent and residual row counts must agree")
    if count > MAX_COUNT:
        raise ValueError(f"row count exceeds {MAX_COUNT}")
    meaningful_bits = count * (latent_bits + identity_bits)
    size = HEADER_BYTES + (meaningful_bits + 7) // 8
    if size > MAX_PACKET_BYTES:
        raise ValueError("packet exceeds 64 MiB")
    output = bytearray(size)
    _HEADER.pack_into(
        output,
        0,
        MAGIC,
        checkpoint_digest,
        source_digest,
        count,
        latent_bits,
        identity_bits,
    )
    position = _pack_rows(output, latents, latent_bits, HEADER_BYTES * 8, "latents")
    _pack_rows(output, residual, identity_bits, position, "residual")
    return bytes(output)


def _read_header(payload: bytes) -> tuple[bytes, bytes, int, int, int]:
    if not isinstance(payload, bytes):
        raise TypeError("packet must be bytes")
    if len(payload) < HEADER_BYTES:
        raise ValueError("truncated packet header")
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError("packet exceeds 64 MiB")
    magic, checkpoint, source, count, latent_bits, identity_bits = _HEADER.unpack_from(
        payload
    )
    if magic != MAGIC:
        raise ValueError("invalid packet magic or version")
    _width(latent_bits, "latent_bits")
    _width(identity_bits, "identity_bits", minimum=0)
    if count > MAX_COUNT:
        raise ValueError(f"row count exceeds {MAX_COUNT}")
    meaningful_bits = count * (latent_bits + identity_bits)
    if len(payload) != HEADER_BYTES + (meaningful_bits + 7) // 8:
        raise ValueError("packet is truncated or has trailing bytes")
    padding = -meaningful_bits % 8
    if padding and payload[-1] & ((1 << padding) - 1):
        raise ValueError("nonzero packet padding")
    return checkpoint, source, count, latent_bits, identity_bits


def payload_bit_count(payload: bytes) -> int:
    """Return meaningful raw payload bits, excluding header and zero padding."""
    _, _, count, latent_bits, identity_bits = _read_header(payload)
    return count * (latent_bits + identity_bits)


def _unpack_rows(
    payload: bytes, count: int, width: int, position: int
) -> list[list[int]]:
    rows = []
    for _ in range(count):
        rows.append(
            [
                (payload[bit // 8] >> (7 - bit % 8)) & 1
                for bit in range(position, position + width)
            ]
        )
        position += width
    return rows


def unpack_packet(payload: bytes) -> dict:
    """Validate framing and recover bit rows and digest metadata exactly."""
    checkpoint, source, count, latent_bits, identity_bits = _read_header(payload)
    start = HEADER_BYTES * 8
    return {
        "latents": _unpack_rows(payload, count, latent_bits, start),
        "residual": _unpack_rows(
            payload, count, identity_bits, start + count * latent_bits
        ),
        "checkpoint_sha256": checkpoint.hex(),
        "source_sha256": source.hex(),
        "latent_bits": latent_bits,
        "identity_bits": identity_bits,
    }
