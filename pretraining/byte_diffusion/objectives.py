"""Reference target construction and losses for byte diffusion.

These routines favor explicit reductions over fused-kernel convenience.  They
are the CPU correctness oracle for later optimized loss gathers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


IGNORE_INDEX = -100


@dataclass(frozen=True)
class RowCrossEntropy:
    """Cross entropy reduced over the final position axis, one row at a time."""

    mean: Tensor
    total: Tensor
    count: Tensor


@dataclass(frozen=True)
class CanvasCrossEntropy:
    """Per-canvas means and their equally weighted aggregate."""

    loss: Tensor
    per_canvas: Tensor
    total: Tensor
    count: Tensor


@dataclass(frozen=True)
class JointObjective:
    total: Tensor
    diffusion: Tensor
    ar: Tensor
    diffusion_targets: Tensor
    ar_targets: Tensor


@dataclass(frozen=True)
class BltMaskedObjective:
    """Fast-BLT masked sums weighted by the shared sampled ``1/t``."""

    loss: Tensor
    per_row: Tensor
    masked_total: Tensor
    masked_count: Tensor


def _require_aligned_ids_mask(ids: Tensor, mask: Tensor) -> None:
    if ids.dtype != torch.long or ids.ndim != 2:
        raise ValueError("ids must be a rank-2 int64 tensor")
    if mask.dtype != torch.bool or mask.shape != ids.shape:
        raise ValueError("mask must be boolean and aligned with ids")


def shifted_ar_targets(
    clean_ids: Tensor,
    valid: Tensor,
    *,
    score_mask: Tensor | None = None,
    output_size: int = 261,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """Build hidden-position-aligned next-atomic-id labels.

    Position ``i`` receives target ``clean_ids[i + 1]`` only when both the
    current hidden position and the target position are valid.  Requiring both
    sides prevents a label from bridging internal document-alignment padding.
    ``score_mask`` is expressed on target positions, matching the repository's
    existing byte-accounting convention.
    """

    _require_aligned_ids_mask(clean_ids, valid)
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    if score_mask is None:
        score_mask = valid
    elif score_mask.dtype != torch.bool or score_mask.shape != clean_ids.shape:
        raise ValueError("score_mask must be boolean and aligned with clean_ids")

    targets = torch.full_like(clean_ids, ignore_index)
    if clean_ids.shape[1] < 2:
        return targets
    active = valid[:, :-1] & valid[:, 1:] & score_mask[:, 1:]
    next_ids = clean_ids[:, 1:]
    if not clean_ids.is_cuda and bool(
        ((next_ids[active] < 0) | (next_ids[active] >= output_size)).any()
    ):
        raise ValueError("an active AR target is outside the output vocabulary")
    targets[:, :-1] = torch.where(active, next_ids, ignore_index)
    return targets


def same_position_targets(
    clean_ids: Tensor,
    active: Tensor,
    *,
    output_size: int = 261,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """Build same-position denoising labels for active corrupted bytes."""

    _require_aligned_ids_mask(clean_ids, active)
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    selected = clean_ids[active]
    if not clean_ids.is_cuda and bool(
        ((selected < 0) | (selected >= output_size)).any()
    ):
        raise ValueError("an active denoising target is outside the output vocabulary")
    return torch.where(active, clean_ids, ignore_index)


def cross_entropy_per_row(
    logits: Tensor,
    targets: Tensor,
    *,
    active: Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> RowCrossEntropy:
    """Compute zero-safe CE sums and means over the last position dimension.

    The returned row shape is ``targets.shape[:-1]``.  Empty rows have exact
    zero total, mean, and gradient; callers decide whether an empty row is
    valid for their objective.
    """

    if logits.ndim < 2 or logits.shape[:-1] != targets.shape:
        raise ValueError("logits and targets have incompatible shapes")
    if targets.dtype != torch.long:
        raise ValueError("targets must be int64")
    if active is None:
        active = targets != ignore_index
    elif active.dtype != torch.bool or active.shape != targets.shape:
        raise ValueError("active must be boolean and aligned with targets")
    if not logits.is_cuda and bool(
        ((targets[active] < 0) | (targets[active] >= logits.shape[-1])).any()
    ):
        raise ValueError("an active target is outside the logit vocabulary")

    safe_targets = torch.where(active, targets, 0)
    ce_logits = (
        logits.float()
        if logits.dtype in {torch.float16, torch.bfloat16}
        else logits
    )
    nll = F.cross_entropy(
        ce_logits.reshape(-1, logits.shape[-1]),
        safe_targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    masked_nll = torch.where(active, nll, 0.0)
    total = masked_nll.sum(dim=-1)
    count = active.sum(dim=-1)
    mean = total / count.clamp_min(1).to(total.dtype)
    return RowCrossEntropy(mean=mean, total=total, count=count)


def canvas_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    active: Tensor,
) -> CanvasCrossEntropy:
    """Average active bytes within each canvas, then canvases equally.

    Unlike a global masked-token mean, this does not give a high-noise canvas
    more weight merely because it sampled a larger ``K``.
    """

    rows = cross_entropy_per_row(logits, targets, active=active)
    nonempty = rows.count > 0
    if rows.count.numel() == 0 or (
        not logits.is_cuda and not bool(nonempty.any())
    ):
        raise ValueError("at least one active canvas target is required")
    return CanvasCrossEntropy(
        loss=(rows.mean * nonempty).sum()
        / nonempty.sum().clamp_min(1).to(rows.mean.dtype),
        per_canvas=rows.mean,
        total=rows.total,
        count=rows.count,
    )


def ar_cross_entropy(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    """Return mean clean AR CE and its active-target count."""

    rows = cross_entropy_per_row(logits, targets)
    count = rows.count.sum()
    if int(count) == 0:
        raise ValueError("AR objective needs at least one active target")
    return rows.total.sum() / count.to(rows.total.dtype), count


def joint_canvas_ar_loss(
    canvas_logits: Tensor,
    canvas_targets: Tensor,
    canvas_active: Tensor,
    ar_logits: Tensor,
    ar_targets: Tensor,
    *,
    lambda_ar: float = 1.0,
) -> JointObjective:
    """Return the named ``mean(canvas means) + lambda_ar * mean(AR)`` loss."""

    if lambda_ar < 0:
        raise ValueError("lambda_ar cannot be negative")
    diffusion = canvas_cross_entropy(canvas_logits, canvas_targets, canvas_active)
    ar, ar_count = ar_cross_entropy(ar_logits, ar_targets)
    return JointObjective(
        total=diffusion.loss + lambda_ar * ar,
        diffusion=diffusion.loss,
        ar=ar,
        diffusion_targets=diffusion.count.sum(),
        ar_targets=ar_count,
    )


def blt_masked_loss(
    logits: Tensor,
    targets: Tensor,
    active: Tensor,
    t: Tensor | float,
) -> BltMaskedObjective:
    """Fast-BLT masked summed CE with sampled ``1/t`` weighting.

    One ``t`` may be shared globally or supplied per leading row.  Rows with
    zero Bernoulli-selected positions contribute exactly zero rather than a
    NaN from an empty mean.  This is the masked component of ``paper_sum``;
    the caller may add the clean per-row summed CE before the batch mean.
    """

    rows = cross_entropy_per_row(logits, targets, active=active)
    noise = torch.as_tensor(t, dtype=rows.total.dtype, device=rows.total.device)
    if noise.ndim == 0:
        pass
    elif noise.shape != rows.total.shape:
        raise ValueError("non-scalar t must match the loss row shape")
    if not logits.is_cuda and bool(((noise <= 0) | (noise > 1)).any()):
        raise ValueError("t must lie in (0, 1]")
    per_row = rows.total / noise
    return BltMaskedObjective(
        loss=per_row.mean(),
        per_row=per_row,
        masked_total=rows.total,
        masked_count=rows.count,
    )


def introspection_balanced_loss(
    mask_ce: Tensor, clean_ce: Tensor
) -> tuple[Tensor, Tensor]:
    """Balance I-DLM masked and clean CE with a detached magnitude ratio."""

    if mask_ce.numel() != 1 or clean_ce.numel() != 1:
        raise ValueError("balanced introspection losses must be scalar")
    if not bool(torch.isfinite(mask_ce)) or not bool(torch.isfinite(clean_ce)):
        raise ValueError("balanced introspection losses must be finite")
    if float(clean_ce.detach()) <= 0:
        raise ValueError("clean_ce must be positive")
    coefficient = (mask_ce.detach() / clean_ce.detach())
    return mask_ce + coefficient * clean_ce, coefficient
