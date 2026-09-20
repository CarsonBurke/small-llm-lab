"""KV partition plan for the SM120 split-4 FA4 decode kernel.

The plan depends only on per-row cache lengths, never on layer activations,
so the rollout engine computes it once per decode step and every layer's
attention launch reuses the same device buffers. This module stays free of
CUTLASS and flash-attention imports so the engine and CPU tests can use it.
"""

from __future__ import annotations

import torch
from torch import Tensor


SPLITS = 4
TILE_N = 32


def split_kv_metadata(
    batch: int, capacity: int, device: torch.device | str
) -> tuple[Tensor, Tensor, Tensor]:
    """Static plan operands: split ids, row base offsets, and the end sentinel."""
    if batch < 1 or capacity < 1:
        raise ValueError("split-KV plan needs a positive batch and capacity")
    return (
        torch.arange(SPLITS, device=device, dtype=torch.int32)[None, :],
        torch.arange(batch, device=device, dtype=torch.int32)[:, None] * capacity,
        torch.full((1,), batch * capacity, device=device, dtype=torch.int32),
    )


def plan_split_kv(
    lengths: Tensor, split_ids: Tensor, batch_offsets: Tensor, sentinel: Tensor
) -> tuple[Tensor, Tensor]:
    """Partition each row's live KV prefix into ``SPLITS`` tile-aligned ranges.

    Starts align to K tiles and clamp to the row length, so trailing
    partitions of short rows are truly empty. Retired lanes keep length one.
    Returns flat int32 ``offsets`` (``batch * SPLITS + 1`` entries, the last
    one the sentinel) and ``live`` lengths (``batch * SPLITS`` entries).
    """
    if lengths.dtype != torch.int32 or lengths.dim() != 1:
        raise ValueError("split-KV lengths must be a flat int32 tensor")
    span = ((lengths + TILE_N * SPLITS - 1) // (TILE_N * SPLITS)) * TILE_N
    starts = torch.minimum(span[:, None] * split_ids, lengths[:, None])
    live = torch.minimum(span[:, None], lengths[:, None] - starts).reshape(-1)
    offsets = torch.cat(((batch_offsets + starts).reshape(-1), sentinel))
    return offsets, live


def split_kv_plan_shapes(batch: int) -> tuple[int, int]:
    """Flat sizes of the ``offsets`` and ``live`` plan buffers."""
    return batch * SPLITS + 1, batch * SPLITS
