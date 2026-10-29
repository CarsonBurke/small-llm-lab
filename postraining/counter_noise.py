"""Counter-keyed noise: hash lanes of (trajectory seed, stream slot, lane).

A trajectory's sampling noise is a function of its identity alone, so a
static-shape CUDA-graph decode, the eager loop and a later replay all draw
the same values whatever the batch shape, row order, left-pad width or
compaction. The token race (``latent_rollout.counter_gumbel_tokens``) keys
the vocabulary lanes; the memory writer's exploration noise
(``memory_carry.write_noise``) keys its own lanes under a separate domain.

A domain derives a sub-seed from each 32-bit half of the trajectory seed
through a bijective mix before the slot is hashed in, so two domains'
streams coincide only when both halves of two seeds line up (about 2^-64
per pair) -- unlike XOR-salting the slot key, which a pair of slots can
cancel for every seed at once.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

MASK32 = (1 << 32) - 1
# Multipliers stay below 2**31 so every int64 product of a 32-bit lane is
# exact (no signed overflow) on CPU and in generated Triton code alike.
_MIX32 = 0x045D9F3B
_SLOT_STRIDE = 0x2545F491
_HIGH_LANE_SALT = 0x3C6EF372
# fp32 resolution of ``counter_normals``' uniform: 23 bits make every
# centered value an odd multiple of 2^-23 strictly inside (-1, 1), exact in
# fp32, so ``erfinv`` is always finite (|z| <= 5.4).
_NORMAL_BITS = 23


def mix32(value: Tensor | int) -> Tensor | int:
    """Bijective 32-bit avalanche mix (Mueller's hash) on int64 lanes."""
    value = (((value >> 16) ^ value) * _MIX32) & MASK32
    value = (((value >> 16) ^ value) * _MIX32) & MASK32
    return (value >> 16) ^ value


def counter_bits(
    row_seeds: Tensor,
    slot: int | Tensor,
    lanes: Tensor,
    domain: tuple[int, int] | None = None,
) -> Tensor:
    """32-bit hashes ``(..., L)`` of ``(row_seeds, slot, lane)``, as int64.

    ``row_seeds`` (int64) broadcasts against ``slot``; ``lanes`` is a 1-D
    int64 lane index. ``domain`` is a pair of 32-bit salts for the seed's
    low and high halves; ``None`` is the token race's domain.

    Both 32-bit halves of the int64 seed key the hash through separate
    lanes, so two trajectories share noise only if their whole seeds match.
    """
    low_seed = row_seeds & MASK32
    high_seed = (row_seeds >> 32) & MASK32
    if domain is not None:
        low_seed = mix32(low_seed ^ domain[0])
        high_seed = mix32(high_seed ^ domain[1])
    slot_key = mix32((slot * _SLOT_STRIDE) & MASK32)
    low_lane = mix32(low_seed ^ slot_key)
    high_lane = mix32(high_seed ^ mix32(slot_key ^ _HIGH_LANE_SALT))
    return mix32(mix32(low_lane[..., None] ^ lanes) ^ high_lane[..., None])


def counter_normals(
    row_seeds: Tensor,
    slot: int | Tensor,
    count: int,
    domain: tuple[int, int],
) -> Tensor:
    """fp32 standard normals ``(..., count)`` keyed like ``counter_bits``.

    One hash per normal through the inverse CDF, so lane ``i`` is the same
    value however many lanes are drawn.
    """
    lanes = torch.arange(count, dtype=torch.int64, device=row_seeds.device)
    bits = counter_bits(row_seeds, slot, lanes, domain) >> (32 - _NORMAL_BITS)
    centered = (2 * bits + 1 - (1 << _NORMAL_BITS)).float() * 2.0**-_NORMAL_BITS
    return math.sqrt(2.0) * torch.erfinv(centered)
