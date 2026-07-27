"""Pure policy-objective functions used by latent VAPO updates."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from postraining.core import clipped_policy_loss


def exact_bernoulli_behavior_kl(
    new_logits: torch.Tensor,
    actions: torch.Tensor,
    old_selected_logprobs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover behavior probabilities and compute ``KL(old || new)``.

    Rollout storage contains the behavior log-probability of the sampled
    Bernoulli action rather than both class probabilities. For a Bernoulli,
    that selected probability and the action recover the full behavior
    distribution exactly.
    """
    if not (
        new_logits.shape == actions.shape == old_selected_logprobs.shape
    ):
        raise ValueError("Bernoulli logits, actions, and log-probabilities must match")
    old_selected_logprobs = old_selected_logprobs.float()
    selected_probability = old_selected_logprobs.exp()
    old_stop_probability = torch.where(
        actions.bool(),
        selected_probability,
        -torch.expm1(old_selected_logprobs),
    ).clamp(0.0, 1.0)
    new_logits = new_logits.float()
    new_stop_probability = new_logits.sigmoid()
    old_continue_probability = 1.0 - old_stop_probability
    kl = (
        torch.xlogy(old_stop_probability, old_stop_probability)
        - old_stop_probability * F.logsigmoid(new_logits)
        + torch.xlogy(old_continue_probability, old_continue_probability)
        - old_continue_probability * F.logsigmoid(-new_logits)
    )
    return old_stop_probability, new_stop_probability, kl


def joint_action_logprobs(
    new_stop_logprobs: torch.Tensor,
    old_stop_logprobs: torch.Tensor,
    new_token_logprobs: torch.Tensor,
    old_token_logprobs: torch.Tensor,
    new_thought_logprobs: torch.Tensor,
    old_thought_logprobs: torch.Tensor,
    stop_mask: torch.Tensor,
    emit_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine conditional policy factors into one log-probability/action."""
    return (
        new_stop_logprobs * stop_mask
        + new_token_logprobs * emit_mask
        + new_thought_logprobs,
        old_stop_logprobs * stop_mask
        + old_token_logprobs * emit_mask
        + old_thought_logprobs,
    )


def per_dimension_thought_policy_loss(
    new_stop_logprobs: torch.Tensor,
    old_stop_logprobs: torch.Tensor,
    new_thought_logprobs: torch.Tensor,
    old_thought_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    stop_mask: torch.Tensor,
    action_denominator: torch.Tensor,
    epsilon_low: float = 0.20,
    epsilon_high: float = 0.28,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clip a diagonal-Gaussian THINK action one latent dimension at a time.

    Factorwise clipping must retain the score-function gradient of the joint
    diagonal Gaussian: ``A * sum_d grad(log p_d)``. Merely averaging the
    dimensional losses would weaken mean/sigma learning by ``latent_dim``.
    The straight-through rescaling below reports their mean surrogate value
    while preserving their summed gradient.

    A CONTINUE stop-gate decision is clipped exactly once as its own factor. A
    detached baseline correction makes an unchanged continued THINK report one
    action's loss rather than two without altering either factor's gradient.
    The mandatory first THINK has ``stop_mask == 0`` and trains only Gaussian
    factors.

    The returned clip fractions are action-normalized. A thought whose every
    dimension clips contributes one to the first; a clipped stop gate
    contributes one to the second.
    """
    if new_thought_logprobs.ndim != 2:
        raise ValueError("thought log-probabilities must have shape [N, D]")
    if new_thought_logprobs.shape != old_thought_logprobs.shape:
        raise ValueError("new and old thought log-probabilities must match")
    action_count, latent_dim = new_thought_logprobs.shape
    if latent_dim < 1:
        raise ValueError("thought log-probabilities need at least one dimension")
    expected_vector_shape = (action_count,)
    for name, value in (
        ("new gate log-probabilities", new_stop_logprobs),
        ("old gate log-probabilities", old_stop_logprobs),
        ("advantages", advantages),
        ("gate mask", stop_mask),
    ):
        if tuple(value.shape) != expected_vector_shape:
            raise ValueError(f"{name} must have shape {expected_vector_shape}")

    dimension_log_ratio = new_thought_logprobs - old_thought_logprobs
    # new_full, not new_tensor: see clipped_policy_loss -- the host-built
    # constant costs a blocking copy every call.
    log_lower = torch.log(
        dimension_log_ratio.new_full((), 1.0 - epsilon_low)
    )
    log_upper = torch.log(
        dimension_log_ratio.new_full((), 1.0 + epsilon_high)
    )
    dimension_advantages = advantages[:, None]
    effective_dimension_log_ratio = torch.where(
        dimension_advantages >= 0,
        torch.minimum(dimension_log_ratio, log_upper),
        torch.maximum(dimension_log_ratio, log_lower),
    )
    dimension_denominator = (
        action_denominator.to(dimension_log_ratio.device) * latent_dim
    ).clamp_min(1)
    dimension_mean_loss = -(
        effective_dimension_log_ratio.exp() * dimension_advantages
    ).sum() / dimension_denominator
    dimension_clip_fraction = (
        (dimension_log_ratio < log_lower)
        | (dimension_log_ratio > log_upper)
    ).sum() / dimension_denominator
    dimension_loss = (
        dimension_mean_loss.detach()
        + latent_dim * (dimension_mean_loss - dimension_mean_loss.detach())
    )
    gate_loss, gate_clip_fraction, _ = clipped_policy_loss(
        new_stop_logprobs,
        old_stop_logprobs,
        advantages,
        stop_mask,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        denominator=action_denominator,
        estimate_kl=False,
    )
    optional_gate_baseline = (
        advantages.detach() * stop_mask
    ).sum() / action_denominator.clamp_min(1)
    return (
        dimension_loss + gate_loss + optional_gate_baseline,
        dimension_clip_fraction,
        gate_clip_fraction,
    )


def joint_thought_policy_loss(
    new_stop_logprobs: torch.Tensor,
    old_stop_logprobs: torch.Tensor,
    new_thought_logprobs: torch.Tensor,
    old_thought_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    stop_mask: torch.Tensor,
    action_denominator: torch.Tensor,
    epsilon_low: float = 0.20,
    epsilon_high: float = 0.28,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clip a diagonal-Gaussian THINK action's JOINT ratio exactly once.

    The thought is one action: its per-dimension log-ratios sum into a single
    joint ratio and the clip-higher band applies to that ratio, exactly like
    EMIT's joint gate+token clip. Unlike the per-dimension objective this
    imposes the trust region on the actual sampled action, at the cost of an
    all-or-nothing gradient per thought — one clipped joint ratio silences
    every dimension of that action's favorable-direction gradient, and the
    harmful direction carries the raw ``exp(sum)`` ratio (watch
    ``harmful_positive_log_ratio_max``; the log-space clip inside
    ``clipped_policy_loss`` bounds only the favorable side).

    A CONTINUE stop-gate decision is clipped exactly once as its own factor. A
    detached baseline correction makes an unchanged continued THINK report one
    action's loss rather than two, matching the per-dimension objective's
    reporting. The mandatory first THINK has ``stop_mask == 0`` and trains only the
    Gaussian factor. Clip fractions are action-normalized, like the
    per-dimension objective's.
    """
    if new_thought_logprobs.ndim != 2:
        raise ValueError("thought log-probabilities must have shape [N, D]")
    if new_thought_logprobs.shape != old_thought_logprobs.shape:
        raise ValueError("new and old thought log-probabilities must match")
    if new_thought_logprobs.size(1) < 1:
        raise ValueError("thought log-probabilities need at least one dimension")
    expected_vector_shape = (new_thought_logprobs.size(0),)
    for name, value in (
        ("new gate log-probabilities", new_stop_logprobs),
        ("old gate log-probabilities", old_stop_logprobs),
        ("advantages", advantages),
        ("gate mask", stop_mask),
    ):
        if tuple(value.shape) != expected_vector_shape:
            raise ValueError(f"{name} must have shape {expected_vector_shape}")

    thought_loss, thought_clip_fraction, _ = clipped_policy_loss(
        new_thought_logprobs.sum(-1),
        old_thought_logprobs.sum(-1),
        advantages,
        torch.ones_like(advantages),
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        denominator=action_denominator,
        estimate_kl=False,
    )
    gate_loss, gate_clip_fraction, _ = clipped_policy_loss(
        new_stop_logprobs,
        old_stop_logprobs,
        advantages,
        stop_mask,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        denominator=action_denominator,
        estimate_kl=False,
    )
    optional_gate_baseline = (
        advantages.detach() * stop_mask
    ).sum() / action_denominator.clamp_min(1)
    return (
        thought_loss + gate_loss + optional_gate_baseline,
        thought_clip_fraction,
        gate_clip_fraction,
    )


def project_thought_means(
    new_means: torch.Tensor,
    old_means: torch.Tensor,
    old_log_sigmas: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TRPL-style mean projection onto the behavior trust region.

    Distances are squared Mahalanobis in behavior-sigma units (= twice the
    Gaussian KL while sigma is frozen), computed in closed form from the
    stored behavior means — no sampled ratio enters the trust decision, so
    unlike the joint clip it cannot fire on rollout noise. Inside the region
    the mean passes through untouched. Outside, the excess is scaled back
    onto the boundary WITH the gradient flowing through the scale: the
    Jacobian ``s * (I - u u^T)`` annihilates the radial component (nothing
    keeps pushing outward) while tangential learning survives.

    Returns ``(projected_means, scale, mahalanobis_sq)`` with ``scale == 1``
    exactly where no projection occurred.
    """
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("trust epsilon must be finite and positive")
    if new_means.shape != old_means.shape:
        raise ValueError("new and old thought means must match")
    if old_log_sigmas.shape != old_means.shape:
        raise ValueError("behavior log sigmas must match the means")
    normalized_shift = (new_means - old_means) * (-old_log_sigmas).exp()
    mahalanobis_sq = normalized_shift.square().sum(-1)
    # Both torch.where branches evaluate; the clamp keeps the unselected
    # rsqrt finite at zero shift (age-0 rows) so its zeroed gradient cannot
    # poison the backward pass with 0 * inf.
    scale = torch.where(
        mahalanobis_sq > epsilon,
        (epsilon / mahalanobis_sq.clamp_min(1e-12)).sqrt(),
        torch.ones_like(mahalanobis_sq),
    )
    projected = old_means + scale[:, None] * (new_means - old_means)
    return projected, scale, mahalanobis_sq


def projected_thought_policy_loss(
    new_stop_logprobs: torch.Tensor,
    old_stop_logprobs: torch.Tensor,
    projected_thought_logprobs: torch.Tensor,
    old_thought_logprobs: torch.Tensor,
    trust_scale: torch.Tensor,
    advantages: torch.Tensor,
    stop_mask: torch.Tensor,
    action_denominator: torch.Tensor,
    epsilon_low: float = 0.20,
    epsilon_high: float = 0.28,
    ratio_guard: float = 2.0,
    gate_advantages: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Surrogate for THINK actions scored under the PROJECTED Gaussian.

    ``exp(log pi_proj - log pi_behavior) * A`` with the gradient through the
    ratio, exactly the unclipped PPO surrogate — the trust region lives in
    the projection, not in a ratio band. With the mean projected, the
    MEAN-driven part of the joint log-ratio concentrates near zero
    (its sampling noise has standard deviation ~sqrt(2 * d_M) <=
    sqrt(2 * epsilon)); the sigma-driven part is NOT constrained by the
    projection (see the module docstring on the sigma gap), so
    ``ratio_guard`` — a symmetric log-space clamp — bounds the exponential
    against Gaussian-tail and sigma-drift excursions. It is a numerical
    guard, not a trust mechanism: a saturated clamp zeroes that action's
    gradient rather than steering it.

    The CONTINUE stop-gate factor keeps its own 1-D textbook clip, and the
    detached baseline makes an unchanged continued THINK report one action's
    loss, both exactly as in the joint/per-dimension objectives. The gate
    factor uses ``gate_advantages`` (raw scale, matching the EMIT gate)
    while ``advantages`` are the caller's unit-normalized values that
    calibrate the thought surrogate. The middle return value is the
    action-normalized PROJECTION fraction (reported through the
    thought-clip-fraction channel).
    """
    if projected_thought_logprobs.ndim != 2:
        raise ValueError("thought log-probabilities must have shape [N, D]")
    if projected_thought_logprobs.shape != old_thought_logprobs.shape:
        raise ValueError("new and old thought log-probabilities must match")
    if ratio_guard <= 0.0:
        raise ValueError("ratio guard must be positive")
    if gate_advantages is None:
        gate_advantages = advantages
    expected_vector_shape = (projected_thought_logprobs.size(0),)
    for name, value in (
        ("new gate log-probabilities", new_stop_logprobs),
        ("old gate log-probabilities", old_stop_logprobs),
        ("trust scale", trust_scale),
        ("advantages", advantages),
        ("gate advantages", gate_advantages),
        ("gate mask", stop_mask),
    ):
        if tuple(value.shape) != expected_vector_shape:
            raise ValueError(f"{name} must have shape {expected_vector_shape}")

    joint_log_ratio = (
        projected_thought_logprobs.sum(-1) - old_thought_logprobs.sum(-1)
    )
    guarded_log_ratio = joint_log_ratio.clamp(-ratio_guard, ratio_guard)
    denominator = action_denominator.to(
        joint_log_ratio.device
    ).clamp_min(1)
    thought_loss = -(guarded_log_ratio.exp() * advantages).sum() / denominator
    projection_fraction = (trust_scale < 1.0).sum() / denominator
    gate_loss, gate_clip_fraction, _ = clipped_policy_loss(
        new_stop_logprobs,
        old_stop_logprobs,
        gate_advantages,
        stop_mask,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        denominator=action_denominator,
        estimate_kl=False,
    )
    optional_gate_baseline = (
        gate_advantages.detach() * stop_mask
    ).sum() / action_denominator.clamp_min(1)
    return (
        thought_loss + gate_loss + optional_gate_baseline,
        projection_fraction,
        gate_clip_fraction,
    )


def sampled_reverse_kl(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
) -> torch.Tensor:
    """Per-factor k3 estimate of ``KL(old behavior || current policy)``.

    Rollout actions are sampled from the old policy. For
    ``log_ratio = log(new) - log(old)``, the expectation under those actions
    of ``exp(log_ratio) - 1 - log_ratio`` is the reverse KL. Keeping the
    factors separate until after k3 avoids exponentiating the potentially
    enormous joint ratio of the 512-D diagonal Gaussian.
    """
    if new_logprobs.shape != old_logprobs.shape:
        raise ValueError("new and old log-probabilities must match")
    log_ratio = new_logprobs - old_logprobs
    return torch.expm1(log_ratio) - log_ratio
