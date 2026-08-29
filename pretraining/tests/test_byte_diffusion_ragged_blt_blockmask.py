from __future__ import annotations

import math

import torch

from pretraining.byte_diffusion.attention import (
    RaggedBltBlockMaskMetadata,
    RaggedBltLayout,
    build_ragged_blt_block_mask,
    ragged_blt_allowed,
    ragged_blt_block_mask_metadata,
)


def _boundary_layout() -> RaggedBltLayout:
    clean_valid = torch.tensor(
        [
            [True, True, True, True, True, True, True, True, False],
            [True, True, True, True, True, True, True, True, True],
        ]
    )
    clean_positions = torch.tensor(
        [[0, 1, 2, 3, 4, 0, 1, 2, -1], [0, 1, 2, 3, 4, 5, 0, 1, 2]]
    )
    clean_segment_ids = torch.tensor(
        [[10, 10, 10, 10, 10, 11, 11, 11, -1], [20, 20, 20, 20, 20, 20, 21, 21, 21]]
    )
    block_rows = torch.tensor([0, 0, 0, 1, 1])
    block_starts = torch.tensor([0, 3, 6, 2, 7])
    byte_offsets = torch.arange(4)
    columns = block_starts[:, None] + byte_offsets
    safe_columns = columns.clamp_max(clean_valid.shape[1] - 1)
    origin_segments = clean_segment_ids[block_rows, block_starts]
    origin_positions = clean_positions[block_rows, block_starts]
    block_valid = (
        (columns < clean_valid.shape[1])
        & clean_valid[block_rows[:, None], safe_columns]
        & clean_segment_ids[block_rows[:, None], safe_columns].eq(
            origin_segments[:, None]
        )
        & clean_positions[block_rows[:, None], safe_columns].eq(
            origin_positions[:, None] + byte_offsets[None]
        )
    )
    layout = RaggedBltLayout(
        clean_valid=clean_valid,
        clean_positions=clean_positions,
        clean_segment_ids=clean_segment_ids,
        block_valid=block_valid,
        block_rows=block_rows,
        block_starts=block_starts,
        row_cu_offsets=torch.tensor([0, 3, 5], dtype=torch.int32),
    )
    layout.validate_values()
    return layout


def _topology(counts: torch.Tensor, indices: torch.Tensor, columns: int) -> torch.Tensor:
    rows = counts.shape[-1]
    slots = torch.arange(indices.shape[-1])
    populated = slots[None, :] < counts[0, 0, :, None]
    topology = torch.zeros(rows, columns, dtype=torch.bool)
    row_ids = torch.arange(rows)[:, None].expand_as(populated)[populated]
    topology[row_ids, indices[0, 0][populated].long()] = True
    return topology


def _dense_block_oracle(
    layout: RaggedBltLayout, block_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    q_block, kv_block = block_size
    query_blocks = math.ceil(layout.query_length / q_block)
    kv_blocks = math.ceil(layout.kv_length / kv_block)
    allowed = ragged_blt_allowed(layout)
    padded = torch.nn.functional.pad(
        allowed,
        (
            0,
            kv_blocks * kv_block - layout.kv_length,
            0,
            query_blocks * q_block - layout.query_length,
        ),
    )
    counts = (
        padded.view(1, query_blocks, q_block, kv_blocks, kv_block)
        .permute(0, 1, 3, 2, 4)
        .sum((-2, -1))[0]
    )
    full = counts == q_block * kv_block
    partial = (counts > 0) & ~full
    return partial, full


@torch.no_grad()
def test_direct_ragged_metadata_exactly_matches_dense_block_oracle() -> None:
    layout = _boundary_layout()
    for block_size in ((8, 4), (4, 4), (7, 5), (128, 128)):
        metadata = ragged_blt_block_mask_metadata(layout, block_size=block_size)
        expected_partial, expected_full = _dense_block_oracle(layout, block_size)
        kv_blocks = expected_partial.shape[-1]
        partial = _topology(
            metadata.kv_num_blocks, metadata.kv_indices, kv_blocks
        )
        full = _topology(
            metadata.full_kv_num_blocks, metadata.full_kv_indices, kv_blocks
        )
        assert torch.equal(partial, expected_partial)
        assert torch.equal(full, expected_full)
        assert not bool((partial & full).any())

        # The explicitly constructed backward transpose is exact as well.
        assert torch.equal(
            _topology(
                metadata.q_num_blocks,
                metadata.q_indices,
                expected_partial.shape[0],
            ),
            expected_partial.T,
        )
        assert torch.equal(
            _topology(
                metadata.full_q_num_blocks,
                metadata.full_q_indices,
                expected_full.shape[0],
            ),
            expected_full.T,
        )


def test_direct_builder_never_calls_flex_mask_materializer(monkeypatch) -> None:
    import torch.nn.attention.flex_attention as flex

    def fail(*args, **kwargs):
        raise AssertionError("create_block_mask must not be called")

    monkeypatch.setattr(flex, "create_block_mask", fail)
    layout = _boundary_layout()
    metadata = ragged_blt_block_mask_metadata(layout, block_size=128)
    block_mask = build_ragged_blt_block_mask(
        layout, block_size=128, metadata=metadata
    )
    assert block_mask.kv_num_blocks.data_ptr() == metadata.kv_num_blocks.data_ptr()
    assert block_mask.q_num_blocks.data_ptr() == metadata.q_num_blocks.data_ptr()


@torch.no_grad()
def test_continuation_page_prefix_starts_at_physical_row_boundary() -> None:
    # This row begins midway through document 44.  Position 10 is therefore at
    # physical column zero; an origin at physical column two must see columns
    # zero and one rather than deriving negative clean-bank block indices.
    layout = RaggedBltLayout(
        clean_valid=torch.ones(1, 8, dtype=torch.bool),
        clean_positions=torch.arange(10, 18)[None],
        clean_segment_ids=torch.full((1, 8), 44, dtype=torch.long),
        block_valid=torch.ones(1, 4, dtype=torch.bool),
        block_rows=torch.tensor([0]),
        block_starts=torch.tensor([2]),
        row_cu_offsets=torch.tensor([0, 1], dtype=torch.int32),
    )
    layout.validate_values()
    metadata = ragged_blt_block_mask_metadata(layout, block_size=(4, 4))
    expected_partial, expected_full = _dense_block_oracle(layout, (4, 4))
    kv_blocks = expected_partial.shape[-1]
    assert torch.equal(
        _topology(metadata.kv_num_blocks, metadata.kv_indices, kv_blocks),
        expected_partial,
    )
    assert torch.equal(
        _topology(
            metadata.full_kv_num_blocks, metadata.full_kv_indices, kv_blocks
        ),
        expected_full,
    )


def _production_scale_layout() -> RaggedBltLayout:
    rows, clean_length = 4, 8192
    starts_per_row = torch.arange(4, clean_length, 16)
    block_rows = torch.arange(rows).repeat_interleave(starts_per_row.numel())
    block_starts = starts_per_row.repeat(rows)
    blocks_per_row = starts_per_row.numel()
    return RaggedBltLayout(
        clean_valid=torch.ones(rows, clean_length, dtype=torch.bool),
        clean_positions=torch.arange(clean_length)[None].expand(rows, -1),
        clean_segment_ids=torch.arange(rows)[:, None].expand(-1, clean_length),
        block_valid=torch.ones(block_rows.numel(), 4, dtype=torch.bool),
        block_rows=block_rows,
        block_starts=block_starts,
        row_cu_offsets=(
            torch.arange(rows + 1, dtype=torch.int32) * blocks_per_row
        ),
    )


def test_production_block_size_uses_sparse_forward_capacity() -> None:
    layout = _production_scale_layout()
    metadata = ragged_blt_block_mask_metadata(layout, block_size=128)
    query_blocks = math.ceil(layout.query_length / 128)
    kv_blocks = math.ceil(layout.kv_length / 128)
    dense_block_grid = query_blocks * kv_blocks
    forward_index_storage = (
        metadata.kv_indices.numel() + metadata.full_kv_indices.numel()
    )

    assert metadata.kv_indices.shape[-2] == query_blocks
    assert metadata.full_kv_indices.shape[-2] == query_blocks
    assert max(
        metadata.kv_indices.shape[-1], metadata.full_kv_indices.shape[-1]
    ) <= math.ceil(layout.clean_length / 128) + 2
    assert forward_index_storage < dense_block_grid // 2


def test_metadata_device_round_trip_preserves_sparse_contract() -> None:
    metadata = ragged_blt_block_mask_metadata(_boundary_layout(), block_size=(8, 4))
    moved = metadata.to("cpu")
    assert isinstance(moved, RaggedBltBlockMaskMetadata)
    assert moved.block_size == metadata.block_size
    assert moved.query_length == metadata.query_length
    assert moved.kv_length == metadata.kv_length
    for actual, expected in zip(
        (
            moved.kv_num_blocks,
            moved.kv_indices,
            moved.q_num_blocks,
            moved.q_indices,
            moved.full_kv_num_blocks,
            moved.full_kv_indices,
            moved.full_q_num_blocks,
            moved.full_q_indices,
        ),
        (
            metadata.kv_num_blocks,
            metadata.kv_indices,
            metadata.q_num_blocks,
            metadata.q_indices,
            metadata.full_kv_num_blocks,
            metadata.full_kv_indices,
            metadata.full_q_num_blocks,
            metadata.full_q_indices,
        ),
    ):
        assert torch.equal(actual, expected)
