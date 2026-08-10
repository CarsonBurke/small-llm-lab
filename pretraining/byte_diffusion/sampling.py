"""Exact reference samplers for autoregressive, ISD, and absorbing canvases."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

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
    adaptive_confidence: float | None = None,
    min_steps: int = 1,
) -> CanvasSample:
    """Absorbing sampler with fixed quotas and optional adaptive final reveal."""

    if adaptive_confidence is not None and not 0.0 < adaptive_confidence <= 1.0:
        raise ValueError("adaptive confidence must lie in (0, 1]")
    if not 1 <= min_steps <= steps:
        raise ValueError("min_steps must lie in [1, steps]")

    ids = initial_ids.clone()
    unresolved = ids.eq(mask_id)
    active = torch.ones_like(unresolved)
    quotas = reveal_quotas(int(unresolved.sum()), steps, gamma)
    useful = 0
    executed = 0
    for step_index, quota in enumerate(quotas):
        live_before = unresolved & active
        if not bool(live_before.any()):
            break
        if quota == 0 and adaptive_confidence is None:
            continue
        logits = denoise(ids)
        executed += 1
        if adaptive_confidence is not None and step_index + 1 >= min_steps:
            confidence = logits.float().softmax(-1).amax(-1)
            if bool((confidence[live_before] >= adaptive_confidence).all()):
                quota = int(live_before.sum())
        if quota and bool(live_before.any()):
            uniforms = torch.rand(
                ids.shape,
                device=ids.device,
                generator=generator,
                dtype=torch.float32,
            )
            from .kernels import (
                categorical_sample_entropy_argmax_confidence,
                reveal_low_entropy as fused_reveal,
            )

            samples, entropy, _, _ = categorical_sample_entropy_argmax_confidence(
                logits, uniforms
            )
            canvas, next_unresolved, next_active, _ = fused_reveal(
                ids[None],
                samples[None],
                entropy[None],
                unresolved[None],
                active[None],
                quota,
                eot_id=eot_id,
            )
            ids, unresolved, active = (
                canvas[0],
                next_unresolved[0],
                next_active[0],
            )
            if bool((live_before & ~unresolved).any()):
                useful += 1
    if bool(unresolved.any()):
        raise RuntimeError("absorbing schedule failed to resolve the canvas")
    return CanvasSample(
        ids=ids,
        active=active,
        useful_nfe=useful,
        executed_nfe=executed,
    )
