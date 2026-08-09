"""Small-tensor attention-mask oracles for byte diffusion.

The functions in this module intentionally materialize dense boolean masks.
They define information flow unambiguously and are suitable for CPU tests and
for checking optimized Flash/Flex attention masks.  Production calls over long
sequences must use a sparse implementation and compare it against these
oracles on small tensors.

All masks use ``True`` to mean that a query may read a key.  The final two
dimensions are ordered ``[query, key]``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _require_bool_mask(name: str, value: Tensor, rank: int) -> None:
    if value.dtype != torch.bool or value.ndim != rank:
        raise ValueError(f"{name} must be a rank-{rank} boolean tensor")


def causal_window_mask(valid: Tensor, window: int) -> Tensor:
    """Return a dense causal sliding-window mask.

    A query at position ``q`` may read a valid key ``k`` exactly when
    ``q - window < k <= q``.  Invalid positions are neither queries nor keys.
    """

    _require_bool_mask("valid", valid, 2)
    if window <= 0:
        raise ValueError("window must be positive")
    length = valid.shape[1]
    positions = torch.arange(length, device=valid.device)
    query = positions[:, None]
    key = positions[None, :]
    geometry = (key <= query) & (key > query - window)
    return valid[:, :, None] & valid[:, None, :] & geometry


def canvas_branch_mask(
    clean_valid: Tensor,
    branch_valid: Tensor,
    prefix_lengths: Tensor,
    *,
    window: int | None = None,
) -> Tensor:
    """Return the clean-plus-independent-canvas information-flow oracle.

    The physical order is the complete clean row followed by each flattened
    branch in branch-index order.  Clean queries are ordinary causal queries
    over the clean row.  A branch query reads valid clean positions strictly
    before its exclusive ``prefix_lengths`` entry and all valid positions in
    its own branch bidirectionally.  It reads no other branch.

    ``branch_valid`` has shape ``[batch, branches, canvas_length]`` and
    ``prefix_lengths`` has shape ``[batch, branches]``.  Prefix lengths are
    physical clean-row indices, so internal document-alignment padding remains
    excluded by ``clean_valid``.
    """

    _require_bool_mask("clean_valid", clean_valid, 2)
    _require_bool_mask("branch_valid", branch_valid, 3)
    if branch_valid.shape[0] != clean_valid.shape[0]:
        raise ValueError("clean and branch batch sizes differ")
    if (
        prefix_lengths.dtype != torch.long
        or prefix_lengths.shape != branch_valid.shape[:2]
    ):
        raise ValueError("prefix_lengths must be one int64 value per branch")

    batch, clean_length = clean_valid.shape
    _, branches, canvas_length = branch_valid.shape
    if bool(((prefix_lengths < 0) | (prefix_lengths > clean_length)).any()):
        raise ValueError("a branch prefix length is outside the clean row")
    if window is not None and window <= 0:
        raise ValueError("window must be positive when supplied")

    total_length = clean_length + branches * canvas_length
    allowed = torch.zeros(
        (batch, total_length, total_length),
        dtype=torch.bool,
        device=clean_valid.device,
    )

    positions = torch.arange(clean_length, device=clean_valid.device)
    distance = positions[:, None] - positions[None, :]
    causal = distance >= 0
    if window is not None:
        causal &= distance < window
    allowed[:, :clean_length, :clean_length] = (
        clean_valid[:, :, None] & clean_valid[:, None, :] & causal
    )

    for branch in range(branches):
        start = clean_length + branch * canvas_length
        stop = start + canvas_length
        query_valid = branch_valid[:, branch]
        prefix_keys = clean_valid & (positions[None, :] < prefix_lengths[:, branch, None])
        prefix_allowed = query_valid[:, :, None] & prefix_keys[:, None, :]
        if window is not None:
            offsets = torch.arange(canvas_length, device=clean_valid.device)
            absolute_query = prefix_lengths[:, branch, None] + offsets[None, :]
            prefix_allowed &= (
                absolute_query[:, :, None] - positions[None, None, :] < window
            )
        allowed[:, start:stop, :clean_length] = prefix_allowed
        allowed[:, start:stop, start:stop] = (
            query_valid[:, :, None] & query_valid[:, None, :]
        )
    return allowed


def introspection_mask(
    clean_valid: Tensor,
    proposal_valid: Tensor,
    block_size: int,
) -> Tensor:
    """Return the dense I-DLM clean/proposal-copy attention mask.

    Physical order is ``clean[0:L]`` then ``proposal[0:L]``.  Clean queries
    remain causal and never read proposals.  A proposal query reads causal
    proposal positions in its own block and clean positions from strictly
    earlier blocks.  In particular, it cannot read its current clean block.
    """

    _require_bool_mask("clean_valid", clean_valid, 2)
    _require_bool_mask("proposal_valid", proposal_valid, 2)
    if clean_valid.shape != proposal_valid.shape:
        raise ValueError("clean and proposal masks must have identical shapes")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    batch, length = clean_valid.shape
    total_length = 2 * length
    allowed = torch.zeros(
        (batch, total_length, total_length),
        dtype=torch.bool,
        device=clean_valid.device,
    )
    positions = torch.arange(length, device=clean_valid.device)
    causal = positions[None, :] <= positions[:, None]
    allowed[:, :length, :length] = (
        clean_valid[:, :, None] & clean_valid[:, None, :] & causal
    )

    query_block = torch.div(positions, block_size, rounding_mode="floor")[:, None]
    key_block = torch.div(positions, block_size, rounding_mode="floor")[None, :]
    earlier_clean_block = key_block < query_block
    same_proposal_block = key_block == query_block
    proposal_causal = same_proposal_block & causal

    allowed[:, length:, :length] = (
        proposal_valid[:, :, None]
        & clean_valid[:, None, :]
        & earlier_clean_block
    )
    allowed[:, length:, length:] = (
        proposal_valid[:, :, None]
        & proposal_valid[:, None, :]
        & proposal_causal
    )
    return allowed


def ar_patch_conditioning_indices(
    length: int,
    patch_stride: int = 4,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Patch-latent index visible to each shifted clean prediction.

    A byte position that closes patch ``p`` may use latent ``p`` to predict
    the next byte.  Earlier positions in the patch use ``p - 1``.  ``-1`` is
    the synthetic initial global state.
    """

    if length < 0:
        raise ValueError("length cannot be negative")
    if patch_stride <= 0:
        raise ValueError("patch_stride must be positive")
    positions = torch.arange(length, dtype=torch.long, device=device)
    return torch.div(positions + 1, patch_stride, rounding_mode="floor") - 1


def denoising_patch_conditioning_indices(
    length: int,
    patch_stride: int = 4,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Patch-latent index for same-position denoising queries."""

    if length < 0:
        raise ValueError("length cannot be negative")
    if patch_stride <= 0:
        raise ValueError("patch_stride must be positive")
    positions = torch.arange(length, dtype=torch.long, device=device)
    return torch.div(positions, patch_stride, rounding_mode="floor")
