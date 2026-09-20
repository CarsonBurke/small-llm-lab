"""CPU contract for the per-step split-KV partition plan."""

from __future__ import annotations

import pytest
import torch

from postraining.split_kv_plan import (
    SPLITS,
    TILE_N,
    plan_split_kv,
    split_kv_metadata,
    split_kv_plan_shapes,
)


def _reference_plan(lengths: list[int], capacity: int) -> tuple[list[int], list[int]]:
    offsets: list[int] = []
    live: list[int] = []
    for row, length in enumerate(lengths):
        span = -(-length // (TILE_N * SPLITS)) * TILE_N
        for split in range(SPLITS):
            start = min(span * split, length)
            offsets.append(row * capacity + start)
            live.append(min(span, length - start))
    offsets.append(len(lengths) * capacity)
    return offsets, live


@pytest.mark.parametrize(
    ("lengths", "capacity"),
    [
        ([257, 65, 2, 1], 257),
        ([8192, 513, 65, 16], 8192),
        ([1], 128),
        ([129, 1, 8192, 65, 4096, 1, 1, 8191], 8192),
    ],
)
def test_plan_partitions_every_live_prefix_exactly(lengths, capacity) -> None:
    device_lengths = torch.tensor(lengths, dtype=torch.int32)
    metadata = split_kv_metadata(len(lengths), capacity, "cpu")
    offsets, live = plan_split_kv(device_lengths, *metadata)
    expected_offsets, expected_live = _reference_plan(lengths, capacity)
    assert offsets.dtype == live.dtype == torch.int32
    assert (offsets.numel(), live.numel()) == split_kv_plan_shapes(len(lengths))
    assert offsets.tolist() == expected_offsets
    assert live.tolist() == expected_live
    # Partitions tile each row contiguously and cover exactly the live prefix.
    for row, length in enumerate(lengths):
        row_offsets = offsets[row * SPLITS : (row + 1) * SPLITS].tolist()
        row_live = live[row * SPLITS : (row + 1) * SPLITS].tolist()
        assert row_offsets[0] == row * capacity
        assert sum(row_live) == length
        for start, size, next_start in zip(row_offsets, row_live, row_offsets[1:]):
            assert size >= 0
            assert size == 0 or start + size == next_start
        assert all(start % TILE_N == 0 for start in (o - row * capacity for o in row_offsets) if start < length)


def test_plan_is_a_pure_function_of_lengths() -> None:
    metadata = split_kv_metadata(3, 512, "cpu")
    lengths = torch.tensor([300, 1, 512], dtype=torch.int32)
    first = plan_split_kv(lengths, *metadata)
    second = plan_split_kv(lengths.clone(), *metadata)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    lengths.add_(1).clamp_max_(512)
    changed = plan_split_kv(lengths, *metadata)
    assert not torch.equal(changed[1], first[1])


def test_plan_rejects_non_int32_lengths() -> None:
    metadata = split_kv_metadata(2, 64, "cpu")
    with pytest.raises(ValueError, match="int32"):
        plan_split_kv(torch.tensor([3, 4]), *metadata)
    with pytest.raises(ValueError, match="positive"):
        split_kv_metadata(0, 64, "cpu")
