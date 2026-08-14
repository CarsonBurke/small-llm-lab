"""Exact reference samplers for autoregressive, ISD, and absorbing canvases."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class ISDDecision:
    token: int
    accepted: bool
    acceptance_probability: float


def positive_residual_distribution(p: Tensor, q: Tensor) -> Tensor:
    """Normalized positive part used after an ISD rejection."""

    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("p and q must be aligned categorical vectors")
    if bool((p < 0).any()) or bool((q < 0).any()):
        raise ValueError("probabilities cannot be negative")
    residual = (p - q).clamp_min(0)
    total = residual.sum()
    if float(total) <= 0:
        # This only occurs up to numerical precision when p == q. The caller
        # should accept in that case, but returning p is a safe total function.
        return p / p.sum()
    return residual / total


def _categorical_from_uniform(probabilities: Tensor, uniform: float) -> int:
    if not 0.0 <= uniform < 1.0:
        raise ValueError("categorical uniform must lie in [0, 1)")
    cdf = probabilities.float().cumsum(-1)
    cdf[-1] = 1.0
    return int(torch.searchsorted(cdf, torch.tensor(uniform, device=cdf.device)).item())


def isd_accept_or_resample(
    p: Tensor,
    q: Tensor,
    proposal: int,
    *,
    accept_uniform: float,
    residual_uniform: float = 0.0,
) -> ISDDecision:
    """One exact speculative acceptance correction from I-DLM."""

    if not 0 <= proposal < p.numel() or p.shape != q.shape:
        raise ValueError("invalid proposal or categorical shape")
    p = p.float() / p.float().sum()
    q = q.float() / q.float().sum()
    q_value = float(q[proposal])
    p_value = float(p[proposal])
    acceptance = 1.0 if q_value == 0.0 else min(1.0, p_value / q_value)
    if accept_uniform < acceptance:
        return ISDDecision(proposal, True, acceptance)
    residual = positive_residual_distribution(p, q)
    token = _categorical_from_uniform(residual, residual_uniform)
    return ISDDecision(token, False, acceptance)


def remaining_schedule(valid_count: int, steps: int, gamma: float = 1.0) -> list[int]:
    """Integer absorbing schedule including initial and final remaining count."""

    if valid_count < 0 or steps <= 0 or gamma <= 0:
        raise ValueError("schedule requires count >= 0, steps > 0, gamma > 0")
    remaining = [valid_count]
    for step in range(1, steps):
        value = math.ceil(valid_count * (1.0 - step / steps) ** gamma)
        remaining.append(min(remaining[-1], value))
    remaining.append(0)
    return remaining


def reveal_quotas(valid_count: int, steps: int, gamma: float = 1.0) -> list[int]:
    remaining = remaining_schedule(valid_count, steps, gamma)
    return [before - after for before, after in zip(remaining, remaining[1:])]


def entropy_from_logits(logits: Tensor) -> Tensor:
    probabilities = logits.float().softmax(-1)
    return -(probabilities * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1)


def reveal_low_entropy(
    ids: Tensor,
    unresolved: Tensor,
    logits: Tensor,
    quota: int,
    *,
    mask_id: int,
    eot_id: int,
    uniforms: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Reveal confident positions, freezing previous decisions and gating EOT."""

    if ids.ndim != 1 or unresolved.shape != ids.shape or logits.shape[:1] != ids.shape:
        raise ValueError("canvas tensors must align")
    if unresolved.dtype != torch.bool or not 0 <= quota <= int(unresolved.sum()):
        raise ValueError("invalid unresolved mask or reveal quota")
    if bool((ids[unresolved] != mask_id).any()):
        raise ValueError("unresolved positions must store MASK")
    if quota == 0:
        return ids.clone(), unresolved.clone()
    if uniforms is None:
        proposed = logits.argmax(-1)
        entropy = entropy_from_logits(logits)
    else:
        if uniforms.shape != ids.shape:
            raise ValueError("one supplied uniform is required per position")
        from .kernels import categorical_sample_entropy_argmax_confidence

        proposed, entropy, _, _ = categorical_sample_entropy_argmax_confidence(
            logits, uniforms
        )
    from .kernels import reveal_low_entropy as fused_reveal

    output, next_unresolved, _, _ = fused_reveal(
        ids[None],
        proposed[None],
        entropy[None],
        unresolved[None],
        torch.ones_like(unresolved)[None],
        quota,
        eot_id=eot_id,
    )
    return output[0], next_unresolved[0]


@dataclass(frozen=True)
class CanvasSample:
    ids: Tensor
    active: Tensor
    useful_nfe: int
    executed_nfe: int


@dataclass(frozen=True)
class BatchedCanvasSample:
    ids: Tensor
    active: Tensor
    useful_nfe: Tensor
    executed_nfe: int


UnmaskingStrategy = Literal["confidence", "entropy_bounded", "fixed_quota"]


def unmasking_quota_and_priority(
    *,
    strategy: UnmaskingStrategy,
    entropy: Tensor,
    confidence: Tensor,
    samples: Tensor,
    live: Tensor,
    fixed_quota: Tensor,
    force_resolve: bool,
    confidence_threshold: float,
    entropy_budget: float,
    eot_id: int,
) -> tuple[Tensor, Tensor]:
    """Select the exact Fast-BLT reveal count and ranking on device."""

    earlier_unresolved = live.to(torch.int32).cumsum(-1) - live.to(torch.int32)
    eligible = live & ((samples != eot_id) | earlier_unresolved.eq(0))
    eligible_count = eligible.sum(-1)
    if strategy == "fixed_quota":
        if force_resolve:
            return live.sum(-1), entropy
        return torch.minimum(fixed_quota, eligible_count), entropy
    if strategy == "confidence":
        qualifying = (eligible & confidence.ge(confidence_threshold)).sum(-1)
        quota = torch.where(eligible_count > 0, qualifying.clamp_min(1), 0)
        # The reveal kernel selects the lowest priority, so negative
        # confidence reveals precisely the highest-confidence positions.
        return quota, -confidence
    if strategy == "entropy_bounded":
        eligible_entropy = torch.where(
            eligible, entropy, torch.full_like(entropy, torch.inf)
        )
        cumulative = eligible_entropy.sort(-1).values.cumsum(-1)
        quota = cumulative.le(entropy_budget).sum(-1)
        quota = torch.where(eligible_count > 0, quota.clamp_min(1), 0)
        return torch.minimum(quota, eligible_count), entropy
    raise ValueError(f"unknown unmasking strategy {strategy!r}")


@torch.no_grad()
def sample_absorbing_canvas_batched(
    initial_ids: Tensor,
    denoise: Callable[[Tensor], Tensor],
    *,
    steps: int,
    mask_id: int,
    eot_id: int,
    gamma: float = 1.0,
    generator: torch.Generator | None = None,
    strategy: UnmaskingStrategy = "confidence",
    confidence_threshold: float = 0.7,
    entropy_budget: float = 1.0,
    row_active: Tensor | None = None,
    stochastic: bool = True,
) -> BatchedCanvasSample:
    """Vectorized Fast-BLT sampling with an independent decision per row.

    Confidence and entropy-bounded modes implement Section 3.1.2 directly:
    every qualifying position is revealed, with a one-position fallback. The
    fixed polynomial quota is retained only as an explicitly named ablation.
    ``steps`` is a hard NFE cap, so its final step resolves every live slot.
    """

    if initial_ids.ndim != 2 or initial_ids.dtype != torch.long:
        raise ValueError("batched canvases must be int64 [batch, width]")
    if strategy not in {"confidence", "entropy_bounded", "fixed_quota"}:
        raise ValueError(f"unknown unmasking strategy {strategy!r}")
    if not 0.0 < confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must lie in (0, 1]")
    if entropy_budget < 0.0 or not math.isfinite(entropy_budget):
        raise ValueError("entropy budget must be finite and nonnegative")
    if row_active is None:
        row_active = torch.ones(
            initial_ids.shape[0], dtype=torch.bool, device=initial_ids.device
        )
    elif row_active.shape != initial_ids.shape[:1] or row_active.dtype != torch.bool:
        raise ValueError("row_active must be one boolean per canvas")

    ids = initial_ids.clone()
    unresolved = ids.eq(mask_id) & row_active[:, None]
    if strategy != "fixed_quota" and steps < initial_ids.shape[1]:
        raise ValueError(
            "paper-exact confidence/entropy sampling requires steps >= canvas width"
        )
    active = torch.ones_like(unresolved) & row_active[:, None]
    useful = torch.zeros(
        initial_ids.shape[0], dtype=torch.int64, device=initial_ids.device
    )
    executed = 0
    target_remaining = remaining_schedule(initial_ids.shape[1], steps, gamma)[1:]
    for step_index, target in enumerate(target_remaining):
        live_before = unresolved & active
        live_rows = live_before.any(-1)
        if not bool(live_rows.any()):
            break
        base_quota = (live_before.sum(-1) - target).clamp_min(0)
        if strategy == "fixed_quota" and not bool((base_quota > 0).any()):
            continue
        logits = denoise(ids)
        if logits.shape[:2] != ids.shape:
            raise ValueError("batched denoiser logits must align with canvases")
        executed += 1
        from .kernels import (
            categorical_entropy_argmax_confidence,
            categorical_sample_entropy_argmax_confidence,
            reveal_low_entropy as fused_reveal,
        )

        if stochastic:
            uniforms = torch.rand(
                ids.shape,
                device=ids.device,
                generator=generator,
                dtype=torch.float32,
            )
            samples, entropy, _, confidence = (
                categorical_sample_entropy_argmax_confidence(logits, uniforms)
            )
        else:
            entropy, samples, confidence = categorical_entropy_argmax_confidence(
                logits
            )
        quota, priority = unmasking_quota_and_priority(
            strategy=strategy,
            entropy=entropy,
            confidence=confidence,
            samples=samples,
            live=live_before,
            fixed_quota=base_quota,
            force_resolve=step_index + 1 == steps,
            confidence_threshold=confidence_threshold,
            entropy_budget=entropy_budget,
            eot_id=eot_id,
        )
        ids, unresolved, active, revealed = fused_reveal(
            ids,
            samples,
            priority,
            unresolved,
            active,
            quota,
            eot_id=eot_id,
            allow_simultaneous_eot=(
                strategy == "fixed_quota" and step_index + 1 == steps
            ),
        )
        useful += (revealed & live_before).any(-1)
    if bool((unresolved & active).any()):
        raise RuntimeError("absorbing schedule failed to resolve a live canvas")
    return BatchedCanvasSample(
        ids=ids,
        active=active,
        useful_nfe=useful,
        executed_nfe=executed,
    )


@torch.no_grad()
def sample_absorbing_canvas(
    initial_ids: Tensor,
    denoise: Callable[[Tensor], Tensor],
    *,
    steps: int,
    mask_id: int,
    eot_id: int,
    gamma: float = 1.0,
    generator: torch.Generator | None = None,
    strategy: UnmaskingStrategy = "confidence",
    confidence_threshold: float = 0.7,
    entropy_budget: float = 1.0,
) -> CanvasSample:
    """Single-canvas adapter around the production batched sampler."""

    if initial_ids.ndim != 1 or initial_ids.dtype != torch.long:
        raise ValueError("canvas must be int64 [width]")

    def batched_denoise(ids: Tensor) -> Tensor:
        logits = denoise(ids[0])
        if logits.shape[:1] != initial_ids.shape:
            raise ValueError("denoiser logits must align with the canvas")
        return logits[None]

    sample = sample_absorbing_canvas_batched(
        initial_ids[None],
        batched_denoise,
        steps=steps,
        mask_id=mask_id,
        eot_id=eot_id,
        gamma=gamma,
        generator=generator,
        strategy=strategy,
        confidence_threshold=confidence_threshold,
        entropy_budget=entropy_budget,
        stochastic=True,
    )
    return CanvasSample(
        ids=sample.ids[0],
        active=sample.active[0],
        useful_nfe=int(sample.useful_nfe[0]),
        executed_nfe=sample.executed_nfe,
    )
