"""Byte-native reference primitives derived from DiffusionGemma.

This module deliberately contains correctness oracles rather than trainer
integration.  Uniform replacement has dense clean-canvas supervision, the
integrated exact-K arm marginalizes one shared ``t ~ Uniform[0, 1]``, and the
sampler follows DiffusionGemma Algorithm 1 without absorbing commitments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class UniformReplacementBatch:
    """One uniformly corrupted canvas with dense same-position labels."""

    ids: Tensor
    targets: Tensor
    active: Tensor
    replaced: Tensor
    changed: Tensor
    unchanged: Tensor
    noise_fraction: Tensor
    changed_fraction: Tensor
    t: Tensor | None
    k: Tensor
    integrated_exact_k: bool

    def validate(self, output_size: int) -> None:
        shape = self.ids.shape
        aligned = (
            self.targets,
            self.active,
            self.replaced,
            self.changed,
            self.unchanged,
        )
        if self.ids.ndim != 2 or any(value.shape != shape for value in aligned):
            raise ValueError("uniform-replacement tensors must be aligned rank-2")
        if self.ids.dtype != torch.long or self.targets.dtype != torch.long:
            raise TypeError("uniform-replacement ids and targets must be int64")
        if any(
            value.dtype != torch.bool
            for value in (
                self.active,
                self.replaced,
                self.changed,
                self.unchanged,
            )
        ):
            raise TypeError("uniform-replacement masks must be boolean")
        if self.k.shape != shape[:1] or self.k.dtype != torch.long:
            raise ValueError("uniform-replacement K must be one int64 per row")
        if (
            self.noise_fraction.shape != shape[:1]
            or self.changed_fraction.shape != shape[:1]
        ):
            raise ValueError("uniform-replacement fractions must be one value per row")
        torch._assert_async(
            (
                ~self.active
                | (self.ids.ge(0) & self.ids.lt(output_size))
            ).all(),
            "uniform replacement emitted an invalid output id",
        )
        torch._assert_async(
            (
                ~self.active
                | (self.targets.ge(0) & self.targets.lt(output_size))
            ).all(),
            "dense supervision contains an invalid target",
        )
        torch._assert_async(
            (~self.replaced | self.active).all(),
            "replacement selected an ineligible position",
        )
        torch._assert_async(
            self.changed.eq(self.active & self.ids.ne(self.targets)).all(),
            "changed-target accounting is inconsistent",
        )
        torch._assert_async(
            self.unchanged.eq(self.active & ~self.changed).all(),
            "changed and unchanged targets must partition supervision",
        )
        torch._assert_async(
            self.k.eq(self.replaced.sum(1)).all(),
            "replacement K does not match the sampled mask",
        )

    @property
    def changed_targets(self) -> Tensor:
        return self.changed.sum(1)

    @property
    def unchanged_targets(self) -> Tensor:
        return self.unchanged.sum(1)

    @property
    def supervised_targets(self) -> Tensor:
        return self.active.sum(1)


def _validate_uniform_inputs(
    clean_ids: Tensor,
    eligible: Tensor,
    output_size: int,
) -> None:
    if clean_ids.ndim != 2 or clean_ids.shape != eligible.shape:
        raise ValueError("clean ids and eligibility must be aligned rank-2 tensors")
    if clean_ids.dtype != torch.long or eligible.dtype != torch.bool:
        raise TypeError("clean ids must be int64 and eligibility must be boolean")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    torch._assert_async(
        (
            ~eligible
            | (clean_ids.ge(0) & clean_ids.lt(output_size))
        ).all(),
        "eligible clean ids must lie in the output vocabulary",
    )


def _exact_k_mask(
    eligible: Tensor,
    k: Tensor,
    *,
    generator: torch.Generator | None,
) -> Tensor:
    counts = eligible.sum(1)
    if k.shape != counts.shape or k.dtype != torch.long:
        raise ValueError("K must be one int64 value per row")
    torch._assert_async(
        ((k >= 0) & (k <= counts)).all(),
        "K lies outside the eligible count",
    )
    width = eligible.shape[1]
    if width == 0:
        return eligible.clone()

    # Sampling a categorical permutation avoids float-score ties.  The dummy
    # item gives all-empty rows positive mass; removing it preserves the
    # uniform relative permutation of eligible positions.
    weights = torch.cat(
        (
            eligible.to(torch.float32),
            torch.ones(
                (eligible.shape[0], 1),
                dtype=torch.float32,
                device=eligible.device,
            ),
        ),
        dim=1,
    )
    order = torch.multinomial(
        weights,
        width + 1,
        replacement=False,
        generator=generator,
    )
    original = order.clamp_max(width - 1)
    eligible_draw = order.lt(width) & eligible.gather(1, original)
    relative_rank = eligible_draw.cumsum(1) - 1
    sentinel = torch.full_like(relative_rank, width + 1)
    ranks = torch.full_like(order, width + 1)
    ranks.scatter_(
        1,
        order,
        torch.where(eligible_draw, relative_rank, sentinel),
    )
    return eligible & ranks[:, :width].lt(k[:, None])


def _materialize_uniform_replacement(
    clean_ids: Tensor,
    eligible: Tensor,
    replaced: Tensor,
    *,
    output_size: int,
    generator: torch.Generator | None,
    t: Tensor | None,
    integrated_exact_k: bool,
) -> UniformReplacementBatch:
    random_ids = torch.randint(
        output_size,
        clean_ids.shape,
        dtype=torch.long,
        device=clean_ids.device,
        generator=generator,
    )
    ids = torch.where(replaced, random_ids, clean_ids)
    changed = eligible & ids.ne(clean_ids)
    unchanged = eligible & ~changed
    eligible_count = eligible.sum(1)
    k = replaced.sum(1)
    denominator = eligible_count.clamp_min(1).to(torch.float32)
    result = UniformReplacementBatch(
        ids=ids,
        targets=clean_ids,
        active=eligible,
        replaced=replaced,
        changed=changed,
        unchanged=unchanged,
        noise_fraction=k.to(torch.float32) / denominator,
        changed_fraction=changed.sum(1).to(torch.float32) / denominator,
        t=t,
        k=k,
        integrated_exact_k=integrated_exact_k,
    )
    result.validate(output_size)
    return result


def uniform_replacement_corruption(
    clean_ids: Tensor,
    eligible: Tensor,
    *,
    output_size: int,
    generator: torch.Generator | None = None,
    t: Tensor | None = None,
) -> UniformReplacementBatch:
    """DiffusionGemma corruption with one shared Bernoulli rate per row.

    The sampled replacement id may equal the clean id.  ``replaced`` records
    forward-process events, while ``changed`` records visible value changes.
    Every eligible position remains an active clean target in either case.
    """

    _validate_uniform_inputs(clean_ids, eligible, output_size)
    batch = clean_ids.shape[0]
    if t is None:
        t = torch.rand(
            (batch,),
            dtype=torch.float32,
            device=clean_ids.device,
            generator=generator,
        )
    elif t.shape not in {(), (batch,)}:
        raise ValueError("t must be scalar or one value per row")
    elif not t.is_floating_point():
        raise TypeError("t must be floating point")
    if t.device != clean_ids.device:
        raise ValueError("t and clean ids must be on the same device")
    torch._assert_async(((t >= 0) & (t <= 1)).all(), "t must lie in [0, 1]")
    row_t = t.expand(batch) if t.ndim == 0 else t
    replaced = eligible & torch.rand(
        clean_ids.shape,
        dtype=torch.float32,
        device=clean_ids.device,
        generator=generator,
    ).lt(row_t[:, None])
    return _materialize_uniform_replacement(
        clean_ids,
        eligible,
        replaced,
        output_size=output_size,
        generator=generator,
        t=t,
        integrated_exact_k=False,
    )


def integrated_exact_k_uniform_replacement(
    clean_ids: Tensor,
    eligible: Tensor,
    *,
    output_size: int,
    generator: torch.Generator | None = None,
    k: Tensor | None = None,
) -> UniformReplacementBatch:
    """Exactly integrate out a shared ``t ~ Uniform[0, 1]``.

    With ``U`` eligible positions, beta-binomial integration makes
    ``K ~ Uniform{0, ..., U}``; conditional on K, every K-subset is uniform.
    This arm samples that integrated law directly without changing the
    marginal forward process.
    """

    _validate_uniform_inputs(clean_ids, eligible, output_size)
    counts = eligible.sum(1)
    if k is None:
        choices = torch.arange(
            eligible.shape[1] + 1,
            device=clean_ids.device,
        )[None]
        weights = choices.le(counts[:, None]).to(torch.float32)
        k = torch.multinomial(
            weights,
            1,
            replacement=True,
            generator=generator,
        ).squeeze(1)
    elif k.device != clean_ids.device:
        raise ValueError("K and clean ids must be on the same device")
    replaced = _exact_k_mask(eligible, k, generator=generator)
    return _materialize_uniform_replacement(
        clean_ids,
        eligible,
        replaced,
        output_size=output_size,
        generator=generator,
        t=None,
        integrated_exact_k=True,
    )


def sample_static_half_batch_mask(
    batch_size: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Select the nearest representable 50% of a static batch.

    Odd tail microbatches are unavoidable when the matched update contains
    249 rows.  Selecting ``floor(B / 2)`` rows preserves the paper's 50%
    mixture to within one row without dropping or duplicating supervision.
    """

    if batch_size <= 0:
        raise ValueError("static half-batch selection requires a positive batch")
    order = torch.randperm(batch_size, device=device, generator=generator)
    ranks = torch.empty_like(order)
    ranks.scatter_(0, order, torch.arange(batch_size, device=order.device))
    return ranks.lt(batch_size // 2)


def stopgrad_probability_embedding_projection(
    probabilities: Tensor,
    embedding_weight: Tensor,
    projection: Callable[[Tensor], Tensor],
    *,
    selected_rows: Tensor | None = None,
) -> Tensor:
    """Project ``stopgrad(probabilities) @ embedding_weight``.

    Gradients intentionally reach the current embedding table and projection,
    but never the prior-pass probabilities.  ``selected_rows`` applies the
    paper's 50% conditioning dropout with static tensor shapes.
    """

    if probabilities.ndim != 3 or embedding_weight.ndim != 2:
        raise ValueError("probabilities must be [B,C,V] and embeddings [V,D]")
    if probabilities.shape[-1] != embedding_weight.shape[0]:
        raise ValueError("probability vocabulary and embedding rows do not match")
    if not probabilities.is_floating_point() or not embedding_weight.is_floating_point():
        raise TypeError("probabilities and embeddings must be floating point")
    if probabilities.device != embedding_weight.device:
        raise ValueError("probabilities and embeddings must be on the same device")
    expected = probabilities.detach().to(embedding_weight.dtype) @ embedding_weight
    projected = projection(expected)
    if projected.ndim != 3 or projected.shape[:2] != probabilities.shape[:2]:
        raise ValueError("self-conditioning projection must return [B,C,D]")
    if selected_rows is not None:
        if selected_rows.shape != probabilities.shape[:1] or selected_rows.dtype != torch.bool:
            raise ValueError("selected_rows must be one boolean per batch row")
        if selected_rows.device != probabilities.device:
            raise ValueError("selected rows and probabilities must share a device")
        projected = torch.where(
            selected_rows[:, None, None],
            projected,
            torch.zeros_like(projected),
        )
    return projected


@dataclass(frozen=True)
class EntropyBudgetSamplerConfig:
    max_steps: int = 48
    entropy_budget: float = 0.1
    stop_entropy: float = 0.005
    temperature_max: float = 0.8
    temperature_min: float = 0.4

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        scalars = (
            self.entropy_budget,
            self.stop_entropy,
            self.temperature_max,
            self.temperature_min,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("sampler scalar configuration must be finite")
        if self.entropy_budget < 0 or self.stop_entropy < 0:
            raise ValueError("entropy thresholds cannot be negative")
        if self.temperature_max <= 0 or self.temperature_min <= 0:
            raise ValueError("temperatures must be positive")
        if self.temperature_max < self.temperature_min:
            raise ValueError("temperature_max must not be below temperature_min")

    def temperature(self, step: int) -> float:
        """Algorithm-1 temperature at one-indexed denoising ``step``."""

        if not 1 <= step <= self.max_steps:
            raise ValueError("step lies outside the denoising schedule")
        t = 1.0 - (step - 1) / self.max_steps
        return (self.temperature_max - self.temperature_min) * t + self.temperature_min


STOP_NONE = 0
STOP_STABLE_ENTROPY = 1
STOP_MAX_STEPS = 2


@dataclass(frozen=True)
class RevisableEntropySamplerState:
    """Static-shape state for a batch of independently stopping canvases."""

    ids: Tensor
    active: Tensor
    previous_argmax: Tensor
    finished: Tensor
    final_ids: Tensor
    final_valid: Tensor
    stop_reason: Tensor
    step: int = 0


@dataclass(frozen=True)
class RevisableEntropyStep:
    state: RevisableEntropySamplerState
    probabilities: Tensor
    entropy: Tensor
    mean_entropy: Tensor
    argmax_ids: Tensor
    sampled_ids: Tensor
    selected: Tensor
    revised: Tensor
    stopped: Tensor
    temperature: float


def initialize_revisable_entropy_sampler(
    batch_size: int,
    canvas_length: int,
    *,
    output_size: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    active: Tensor | None = None,
) -> RevisableEntropySamplerState:
    """Initialize every canvas position from the uniform output prior."""

    if min(batch_size, canvas_length, output_size) <= 0:
        raise ValueError("batch, canvas length, and output size must be positive")
    ids = torch.randint(
        output_size,
        (batch_size, canvas_length),
        dtype=torch.long,
        device=device,
        generator=generator,
    )
    if active is None:
        active = torch.ones_like(ids, dtype=torch.bool)
    elif active.shape != ids.shape or active.dtype != torch.bool:
        raise ValueError("active canvas positions must be aligned boolean values")
    elif active.device != ids.device:
        raise ValueError("active canvas positions must share the sampler device")
    if not torch.compiler.is_compiling() and bool(~active.any(1).all()):
        raise ValueError("every sampler row needs at least one active canvas position")
    return RevisableEntropySamplerState(
        ids=ids,
        active=active,
        previous_argmax=torch.zeros_like(ids),
        finished=torch.zeros(batch_size, dtype=torch.bool, device=ids.device),
        final_ids=torch.zeros_like(ids),
        final_valid=torch.zeros_like(ids, dtype=torch.bool),
        stop_reason=torch.full(
            (batch_size,), STOP_NONE, dtype=torch.long, device=ids.device
        ),
    )


def final_eot_valid_mask(ids: Tensor, eot_id: int) -> Tensor:
    """Keep a final canvas through its first EOT, or all positions if absent."""

    if ids.ndim != 2 or ids.dtype != torch.long:
        raise ValueError("final ids must be rank-2 int64")
    positions = torch.arange(ids.shape[1], device=ids.device)[None]
    eot_positions = torch.where(ids.eq(eot_id), positions, ids.shape[1])
    first_eot = eot_positions.amin(-1)
    return positions <= first_eot[:, None]


def _entropy_bounded_selection(
    entropy: Tensor, budget: float, active: Tensor
) -> Tensor:
    """Algorithm-1 selection with stable position-index tie breaking."""

    ranked_entropy = torch.where(active, entropy, torch.inf)
    order = ranked_entropy.argsort(dim=-1, stable=True)
    sorted_entropy = ranked_entropy.gather(-1, order)
    exclusive = sorted_entropy.cumsum(-1) - sorted_entropy
    selected_sorted = exclusive.le(budget)
    selected = torch.zeros_like(selected_sorted)
    selected.scatter_(1, order, selected_sorted)
    return selected & active


@torch.no_grad()
def revisable_entropy_sampler_step(
    state: RevisableEntropySamplerState,
    logits: Tensor,
    config: EntropyBudgetSamplerConfig,
    *,
    eot_id: int,
    generator: torch.Generator | None = None,
) -> RevisableEntropyStep:
    """Advance one revisable multinomial denoising step.

    Intermediate EOT values are ordinary revisable noise.  EOT truncation is
    applied only to the deterministic argmax canvas when a row stops.
    """

    if state.ids.ndim != 2 or state.ids.dtype != torch.long:
        raise ValueError("sampler state ids must be rank-2 int64")
    if logits.shape[:2] != state.ids.shape or logits.ndim != 3:
        raise ValueError("sampler logits must align as [B,C,V]")
    if not logits.is_floating_point():
        raise TypeError("sampler logits must be floating point")
    if logits.device != state.ids.device:
        raise ValueError("sampler logits and state must be on the same device")
    if logits.shape[-1] <= 0 or not 0 <= eot_id < logits.shape[-1]:
        raise ValueError("eot_id must lie in the output vocabulary")
    batch = state.ids.shape[0]
    state_vectors = (
        state.active,
        state.previous_argmax,
        state.final_ids,
        state.final_valid,
    )
    if any(value.shape != state.ids.shape for value in state_vectors):
        raise ValueError("sampler canvas state tensors do not align")
    if (
        state.active.dtype != torch.bool
        or state.previous_argmax.dtype != torch.long
        or state.final_ids.dtype != torch.long
        or state.final_valid.dtype != torch.bool
    ):
        raise TypeError("sampler canvas state dtypes are invalid")
    if (
        state.finished.shape != (batch,)
        or state.stop_reason.shape != (batch,)
        or state.finished.dtype != torch.bool
        or state.stop_reason.dtype != torch.long
    ):
        raise ValueError("sampler row state tensors do not align")
    if not 0 <= state.step < config.max_steps:
        raise RuntimeError("sampler has exhausted its maximum step count")
    torch._assert_async(
        ~state.finished.all(),
        "every sampler row has already stopped",
    )

    step = state.step + 1
    temperature = config.temperature(step)
    probabilities = (logits.float() / temperature).softmax(-1)
    log_probabilities = probabilities.clamp_min(
        torch.finfo(torch.float32).tiny
    ).log()
    entropy = -(probabilities * log_probabilities).sum(-1)
    active_count = state.active.sum(-1).clamp_min(1)
    mean_entropy = torch.where(state.active, entropy, 0.0).sum(-1) / active_count
    argmax_ids = logits.argmax(-1)
    stable = (
        (argmax_ids.eq(state.previous_argmax) | ~state.active).all(-1)
        if state.step > 0
        else torch.zeros_like(state.finished)
    )
    live = ~state.finished
    stable_stop = live & stable & mean_entropy.le(config.stop_entropy)
    max_stop = live & ~stable_stop & torch.full_like(live, step >= config.max_steps)
    stopped = stable_stop | max_stop
    finished = state.finished | stopped
    reason = torch.where(
        stable_stop,
        torch.full_like(state.stop_reason, STOP_STABLE_ENTROPY),
        torch.where(
            max_stop,
            torch.full_like(state.stop_reason, STOP_MAX_STEPS),
            state.stop_reason,
        ),
    )
    # Fixed replay/padding slots are outside the generated canvas. Their
    # predictions must not terminate a ragged-phase continuation.
    canvas_positions = torch.arange(state.ids.shape[1], device=state.ids.device)[None]
    generated_eot = state.active & argmax_ids.eq(eot_id)
    first_generated_eot = torch.where(
        generated_eot, canvas_positions, state.ids.shape[1]
    ).amin(-1)
    candidate_valid = state.active & (
        canvas_positions <= first_generated_eot[:, None]
    )
    final_ids = torch.where(stopped[:, None], argmax_ids, state.final_ids)
    final_valid = torch.where(stopped[:, None], candidate_valid, state.final_valid)

    continuing = live & ~stopped
    if step < config.max_steps and bool(continuing.any()):
        flattened = probabilities.reshape(-1, probabilities.shape[-1])
        sampled_ids = torch.multinomial(
            flattened,
            1,
            replacement=True,
            generator=generator,
        ).reshape_as(state.ids)
        selected = _entropy_bounded_selection(
            entropy, config.entropy_budget, state.active
        )
        uniform_ids = torch.randint(
            probabilities.shape[-1],
            state.ids.shape,
            dtype=torch.long,
            device=state.ids.device,
            generator=generator,
        )
        proposal = torch.where(selected, sampled_ids, uniform_ids)
    else:
        # Algorithm 1 returns the deterministic prediction at N without a
        # transition, so the exhausted step must not advance the RNG stream.
        sampled_ids = argmax_ids
        selected = torch.zeros_like(state.ids, dtype=torch.bool)
        proposal = argmax_ids
    next_ids = torch.where(
        continuing[:, None] & state.active,
        proposal,
        torch.where(stopped[:, None], argmax_ids, state.ids),
    )
    selected = selected & continuing[:, None]
    revised = continuing[:, None] & next_ids.ne(state.ids)
    previous_argmax = torch.where(
        live[:, None], argmax_ids, state.previous_argmax
    )
    next_state = RevisableEntropySamplerState(
        ids=next_ids,
        active=state.active,
        previous_argmax=previous_argmax,
        finished=finished,
        final_ids=final_ids,
        final_valid=final_valid,
        stop_reason=reason,
        step=step,
    )
    return RevisableEntropyStep(
        state=next_state,
        probabilities=probabilities,
        entropy=entropy,
        mean_entropy=mean_entropy,
        argmax_ids=argmax_ids,
        sampled_ids=sampled_ids,
        selected=selected,
        revised=revised,
        stopped=stopped,
        temperature=temperature,
    )


__all__ = (
    "EntropyBudgetSamplerConfig",
    "RevisableEntropySamplerState",
    "RevisableEntropyStep",
    "STOP_MAX_STEPS",
    "STOP_NONE",
    "STOP_STABLE_ENTROPY",
    "UniformReplacementBatch",
    "final_eot_valid_mask",
    "initialize_revisable_entropy_sampler",
    "integrated_exact_k_uniform_replacement",
    "revisable_entropy_sampler_step",
    "sample_static_half_batch_mask",
    "stopgrad_probability_embedding_projection",
    "uniform_replacement_corruption",
)
