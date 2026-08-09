"""Exhaustive small-tensor information-flow tests."""

from __future__ import annotations

import pytest
import torch

from pretraining.byte_diffusion.masks import (
    ar_patch_conditioning_indices,
    canvas_branch_mask,
    causal_window_mask,
    denoising_patch_conditioning_indices,
    introspection_mask,
)


def test_causal_window_matches_hand_enumerated_matrix() -> None:
    valid = torch.ones(1, 6, dtype=torch.bool)
    observed = causal_window_mask(valid, window=3)[0]
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(observed, expected)


def test_causal_window_excludes_invalid_queries_and_keys() -> None:
    valid = torch.tensor([[True, True, False, True]])
    observed = causal_window_mask(valid, window=4)[0]
    assert not bool(observed[2].any())
    assert not bool(observed[:, 2].any())
    torch.testing.assert_close(observed[3], torch.tensor([True, True, False, True]))
    with pytest.raises(ValueError, match="positive"):
        causal_window_mask(valid, window=0)


def test_canvas_branches_read_only_their_prefix_and_themselves() -> None:
    clean = torch.ones(1, 3, dtype=torch.bool)
    branches = torch.ones(1, 2, 2, dtype=torch.bool)
    observed = canvas_branch_mask(clean, branches, torch.tensor([[2, 3]]))[0]
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0],
            [1, 1, 0, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 1, 1],
            [1, 1, 1, 0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(observed, expected)


def test_canvas_mask_respects_internal_padding_and_branch_padding() -> None:
    clean = torch.tensor([[True, False, True]])
    branches = torch.tensor([[[True, False]]])
    observed = canvas_branch_mask(clean, branches, torch.tensor([[3]]))[0]
    assert not bool(observed[:, 1].any())
    assert not bool(observed[4].any())
    torch.testing.assert_close(
        observed[3], torch.tensor([True, False, True, True, False])
    )


def test_introspection_edges_are_exhaustive_for_two_adjacent_blocks() -> None:
    clean = torch.ones(1, 8, dtype=torch.bool)
    proposal = torch.ones_like(clean)
    observed = introspection_mask(clean, proposal, block_size=4)[0]
    expected = torch.zeros(16, 16, dtype=torch.bool)
    for query in range(8):
        expected[query, : query + 1] = True
    for offset in range(4):
        expected[8 + offset, 8 : 8 + offset + 1] = True
    for offset in range(4, 8):
        expected[8 + offset, :4] = True
        expected[8 + offset, 12 : 8 + offset + 1] = True
    torch.testing.assert_close(observed, expected)

    # Explicitly name the highest-risk negative edges.
    assert not bool(observed[:8, 8:].any())
    assert not bool(observed[8:12, :8].any())
    assert not bool(observed[12:, 4:8].any())
    assert not bool(observed[8:12, 12:].any())


def test_introspection_invalid_positions_are_neither_queries_nor_keys() -> None:
    clean = torch.tensor([[True, True, False, True]])
    proposal = torch.tensor([[True, False, True, True]])
    observed = introspection_mask(clean, proposal, block_size=2)[0]
    assert not bool(observed[2].any())
    assert not bool(observed[:, 2].any())
    assert not bool(observed[5].any())
    assert not bool(observed[:, 5].any())


def test_clean_and_denoising_patch_conditioning_have_distinct_alignment() -> None:
    torch.testing.assert_close(
        ar_patch_conditioning_indices(8),
        torch.tensor([-1, -1, -1, 0, 0, 0, 0, 1]),
    )
    torch.testing.assert_close(
        denoising_patch_conditioning_indices(8),
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
    )
    with pytest.raises(ValueError, match="negative"):
        ar_patch_conditioning_indices(-1)

