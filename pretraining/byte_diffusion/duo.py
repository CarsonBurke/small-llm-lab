"""Exact uniform-state diffusion primitives for the Byte-Duo recipe.

This is the non-absorbing process used by Duo: every active latent is always
one of the 261 clean atomic values.  MASK is neither a prior state nor a
prediction class.  The implementation follows Eq. 9--11 of *Scaling Beyond
Masked Diffusion Language Models* and the authors' ``DUO_BASE`` reference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class DuoSchedule:
    """Reference linear-in-alpha continuous-time schedule.

    Time is sampled uniformly on ``[0, 1]`` and
    ``alpha(t) = 1 - (1 - eps) * t``. This is the reference implementation's
    ``scaled_t = (1-eps)*t; alpha = 1-scaled_t`` sequence: alpha remains in
    ``[eps, 1]`` without truncating and renormalizing the NELBO's time
    integral. Sampling still starts from the explicitly defined uniform prior,
    as in ``UniformState``.
    """

    eps: float = 1e-3

    def __post_init__(self) -> None:
        if not math.isfinite(self.eps) or not 0.0 < self.eps < 0.5:
            raise ValueError("Duo schedule eps must lie strictly in (0, 0.5)")

    def alpha(self, t: Tensor) -> Tensor:
        if not t.is_floating_point():
            raise TypeError("diffusion time must be floating point")
        if not torch.compiler.is_compiling():
            if bool(((t < 0) | (t > 1)).any()):
                raise ValueError("diffusion time must lie in [0, 1]")
        return 1.0 - (1.0 - self.eps) * t

    def derivative(self, t: Tensor) -> Tensor:
        return torch.full_like(t, -(1.0 - self.eps))


@dataclass(frozen=True)
class DuoCorruption:
    ids: Tensor
    targets: Tensor
    active: Tensor
    replaced: Tensor
    changed: Tensor
    t: Tensor
    alpha: Tensor
    fixed_clean: Tensor | None = None

    def validate(self, *, clean_atoms: int, pad_id: int) -> None:
        shape = self.ids.shape
        fixed_clean = (
            torch.zeros_like(self.active)
            if self.fixed_clean is None
            else self.fixed_clean
        )
        if self.ids.ndim != 2 or any(
            value.shape != shape
            for value in (
                self.targets,
                self.active,
                self.replaced,
                self.changed,
                fixed_clean,
            )
        ):
            raise ValueError("Duo corruption tensors must be aligned rank-2")
        if self.ids.dtype != torch.long or self.targets.dtype != torch.long:
            raise TypeError("Duo ids and targets must be int64")
        if any(
            value.dtype != torch.bool
            for value in (self.active, self.replaced, self.changed, fixed_clean)
        ):
            raise TypeError("Duo masks must be boolean")
        if bool((self.active & fixed_clean).any()):
            raise ValueError("Duo diffused and fixed-clean positions must be disjoint")
        if self.t.shape != shape[:1] or self.alpha.shape != shape[:1]:
            raise ValueError("Duo time and alpha must have one value per row")
        active_ids = self.ids.masked_select(self.active)
        if not torch.compiler.is_compiling() and bool(
            ((active_ids < 0) | (active_ids >= clean_atoms)).any()
        ):
            raise ValueError("Duo corruption emitted a non-clean atom")
        visible = self.active | fixed_clean
        fixed_ids = self.ids.masked_select(fixed_clean)
        if not torch.compiler.is_compiling() and bool(
            fixed_ids.ne(self.targets.masked_select(fixed_clean)).any()
        ):
            raise ValueError("fixed-clean Duo positions must retain their targets")
        if not torch.compiler.is_compiling() and bool(
            self.ids.masked_select(~visible).ne(pad_id).any()
        ):
            raise ValueError("inactive Duo storage must contain PAD")
        if not torch.equal(self.changed, self.active & self.ids.ne(self.targets)):
            raise ValueError("Duo changed accounting is inconsistent")


def sample_antithetic_times(
    rows: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Stratified antithetic times for one complete optimizer update.

    Call once for the complete optimizer-update sample count and slice the
    returned tensor only afterward. This makes the sampled objective
    independent of how that update is partitioned.
    """

    if rows <= 0:
        raise ValueError("antithetic sampling needs a positive row count")
    jitter = torch.rand(rows, device=device, generator=generator)
    strata = torch.arange(rows, device=device, dtype=jitter.dtype)
    # Exact reference construction: rand(N)/N + arange(N)/N.
    # The mathematical endpoint t=0 has measure zero, but float RNGs can emit
    # it exactly and alpha would then round to one in Eq. 11 denominators.
    # Moving only that sub-ULP endpoint to the first positive float-resolution
    # time preserves the uniform integral to machine precision.
    return ((strata + jitter) / rows).clamp_min(torch.finfo(jitter.dtype).eps)


def sample_branch_antithetic_times(
    rows: int,
    branches: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Stratify all samples while spreading each page's branches over time.

    A direct ``view(rows, branches)`` assigns adjacent, highly correlated time
    strata to branches that share a clean page. Branch-major layout preserves
    the exact global antithetic set while giving each page one sample from
    every branch-sized band of the unit interval.
    """

    if rows <= 0 or branches <= 0:
        raise ValueError("branch-antithetic sampling needs positive geometry")
    return sample_antithetic_times(
        rows * branches,
        device=device,
        generator=generator,
    ).view(branches, rows).transpose(0, 1).contiguous()


def uniform_state_corruption(
    clean_ids: Tensor,
    active: Tensor,
    t: Tensor,
    *,
    clean_atoms: int,
    pad_id: int,
    schedule: DuoSchedule = DuoSchedule(),
    generator: torch.Generator | None = None,
) -> DuoCorruption:
    """Sample ``q_t = alpha*x + (1-alpha)/K`` without a MASK state."""

    if clean_ids.ndim != 2 or active.shape != clean_ids.shape:
        raise ValueError("clean ids and active mask must be aligned rank-2")
    if clean_ids.dtype != torch.long or active.dtype != torch.bool:
        raise TypeError("clean ids must be int64 and active must be boolean")
    if t.shape != clean_ids.shape[:1] or not t.is_floating_point():
        raise ValueError("t must be floating point with one value per row")
    if clean_atoms <= 1 or 0 <= pad_id < clean_atoms:
        raise ValueError("clean atom count and PAD id are inconsistent")
    if not torch.compiler.is_compiling():
        clean = clean_ids.masked_select(active)
        if bool(((clean < 0) | (clean >= clean_atoms)).any()):
            raise ValueError("active targets must be clean atomic ids")
    alpha = schedule.alpha(t)
    replaced = active & torch.rand(
        clean_ids.shape,
        device=clean_ids.device,
        generator=generator,
    ).ge(alpha[:, None])
    uniform = torch.randint(
        clean_atoms,
        clean_ids.shape,
        device=clean_ids.device,
        dtype=torch.long,
        generator=generator,
    )
    ids = torch.where(active, torch.where(replaced, uniform, clean_ids), pad_id)
    result = DuoCorruption(
        ids=ids,
        targets=clean_ids,
        active=active,
        replaced=replaced,
        changed=active & ids.ne(clean_ids),
        t=t,
        alpha=alpha,
    )
    result.validate(clean_atoms=clean_atoms, pad_id=pad_id)
    return result


def duo_nelbo_token_loss(
    logits: Tensor,
    noisy_ids: Tensor,
    clean_ids: Tensor,
    alpha: Tensor,
    dalpha: Tensor | float,
    *,
    clean_atoms: int,
) -> Tensor:
    """Lower-variance exact Duo NELBO integrand from paper Eq. 11.

    The returned tensor has one NELBO contribution per token.  Computation is
    deliberately float32 even under BF16 autocast.  The reference schedule
    keeps alpha strictly between zero and one, so non-finite results are bugs
    and fail closed rather than being clamped or silently replaced.
    """

    if logits.shape[:-1] != noisy_ids.shape or noisy_ids.shape != clean_ids.shape:
        raise ValueError("Duo logits, noisy ids, and clean ids do not align")
    if logits.shape[-1] != clean_atoms:
        raise ValueError("Duo logits must cover exactly the clean atomic vocabulary")
    if alpha.shape not in {(logits.shape[0],), (logits.shape[0], 1)}:
        raise ValueError("alpha must have one scalar per row")
    if noisy_ids.dtype != torch.long or clean_ids.dtype != torch.long:
        raise TypeError("Duo targets must be int64")

    work_logits = logits.float()
    log_x_theta = F.log_softmax(work_logits, dim=-1)
    x_theta = log_x_theta.exp()
    row_alpha = alpha.reshape(logits.shape[0], 1).float()
    if not torch.compiler.is_compiling() and bool(
        ((row_alpha <= 0) | (row_alpha >= 1)).any()
    ):
        raise ValueError("Duo Eq. 11 requires alpha strictly inside (0, 1)")
    derivative = torch.as_tensor(dalpha, device=logits.device, dtype=torch.float32)
    if derivative.ndim == 0:
        derivative = derivative.expand(logits.shape[0]).reshape(-1, 1)
    elif derivative.shape in {(logits.shape[0],), (logits.shape[0], 1)}:
        derivative = derivative.reshape(-1, 1)
    else:
        raise ValueError("dalpha must be scalar or have one value per row")

    x_bar_theta = clean_atoms * row_alpha[..., None] * x_theta + 1.0 - row_alpha[..., None]
    equal = clean_ids.eq(noisy_ids).float()
    unequal = 1.0 - equal
    xbar_noisy = (1.0 - row_alpha) + clean_atoms * row_alpha * equal
    xbar_theta_noisy = x_bar_theta.gather(-1, noisy_ids[..., None]).squeeze(-1)
    xbar_theta_clean = x_bar_theta.gather(-1, clean_ids[..., None]).squeeze(-1)

    coefficient = derivative / (clean_atoms * row_alpha)
    term1 = clean_atoms * (xbar_noisy.reciprocal() - xbar_theta_noisy.reciprocal())
    kappa = (1.0 - row_alpha) / (
        clean_atoms * row_alpha + 1.0 - row_alpha
    )
    log_kappa = kappa.log()
    term2_coefficients = equal * kappa + unequal
    term2_offset = (
        (clean_atoms - 1.0) * kappa * equal - kappa.reciprocal() * unequal
    ) * log_kappa
    term2_theta = -term2_coefficients * (
        x_bar_theta.log().sum(-1)
        - clean_atoms * xbar_theta_noisy.log()
    )
    term2_theta = term2_theta - (
        clean_atoms
        * row_alpha
        / (1.0 - row_alpha)
        * (xbar_theta_clean.log() - xbar_theta_noisy.log())
        * unequal
    )
    loss = coefficient * (term1 - (term2_theta + term2_offset))
    if not torch.compiler.is_compiling() and not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("non-finite exact Duo NELBO")
    return loss


def duo_reverse_posterior(
    clean_probabilities: Tensor,
    noisy_ids: Tensor,
    alpha_s: Tensor | float,
    alpha_t: Tensor | float,
    *,
    use_float64: bool = False,
) -> Tensor:
    """Exact ancestral ``q(z_s | z_t, x_theta)`` from paper Eq. 10."""

    if clean_probabilities.shape[:-1] != noisy_ids.shape:
        raise ValueError("posterior probabilities and noisy ids do not align")
    if noisy_ids.dtype != torch.long:
        raise TypeError("noisy ids must be int64")
    clean_atoms = clean_probabilities.shape[-1]
    if clean_atoms <= 1:
        raise ValueError("posterior needs at least two states")
    dtype = torch.float64 if use_float64 else torch.float32
    x = clean_probabilities.to(dtype)
    # Normalize model predictions defensively; this preserves exact formulas
    # while admitting probabilities produced by either softmax precision.
    x = x / x.sum(-1, keepdim=True)
    shape = (*([1] * (x.ndim - 1)),)

    def broadcast_alpha(value: Tensor | float) -> Tensor:
        tensor = torch.as_tensor(value, device=x.device, dtype=dtype)
        if tensor.ndim == 0:
            return tensor
        if tensor.shape in {(x.shape[0],), (x.shape[0], 1)}:
            return tensor.reshape(x.shape[0], *([1] * (x.ndim - 1)))
        if tensor.shape == x.shape[:-1]:
            return tensor[..., None]
        raise ValueError("alpha must be scalar, per-row, or per-token")

    a_s = broadcast_alpha(alpha_s)
    a_t = broadcast_alpha(alpha_t)
    if not torch.compiler.is_compiling():
        if bool(((a_t < 0) | (a_s > 1) | (a_t > a_s) | (a_s <= 0)).any()):
            raise ValueError("reverse transition requires 0 <= alpha_t <= alpha_s <= 1")
    ratio = a_t / a_s
    delta = a_s - a_t
    observed = F.one_hot(noisy_ids, clean_atoms).to(dtype)
    numerator = (
        a_t * clean_atoms * x * observed
        + (ratio - a_t) * observed
        + delta * x
        + (1.0 - ratio) * (1.0 - a_s) / clean_atoms
    )
    denominator = (
        a_t * clean_atoms * x.gather(-1, noisy_ids[..., None])
        + 1.0
        - a_t
    )
    posterior = numerator / denominator
    mass = posterior.sum(-1, keepdim=True)
    if not torch.compiler.is_compiling():
        if not bool(torch.isfinite(posterior).all()) or not bool(
            torch.isfinite(mass).all()
        ):
            raise FloatingPointError("non-finite exact Duo posterior")
        if bool(posterior.lt(-1e-6).any()) or bool(mass.le(0).any()):
            raise FloatingPointError("invalid exact Duo posterior probability mass")
    # Roundoff at K=261 can produce sub-ulp negatives at schedule endpoints.
    # Correct only that bounded numerical error; material failures above abort.
    posterior = posterior.clamp_min(0)
    return posterior / posterior.sum(-1, keepdim=True)


def sample_categorical(
    probabilities: Tensor, *, generator: torch.Generator | None = None
) -> Tensor:
    """Vectorized categorical sampling over the final dimension."""

    if probabilities.ndim < 2:
        raise ValueError("categorical probabilities need a class dimension")
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    return torch.multinomial(flat, 1, replacement=True, generator=generator).reshape(
        probabilities.shape[:-1]
    )


# Explicit fusion boundaries for the two vocabulary-dense elementwise kernels.
# They are invoked only by CUDA entry points; CPU property tests retain the
# transparent eager implementations above.
compiled_duo_nelbo_token_loss = torch.compile(
    duo_nelbo_token_loss, dynamic=False, fullgraph=False
)
compiled_duo_reverse_posterior = torch.compile(
    duo_reverse_posterior, dynamic=True, fullgraph=False
)


__all__ = (
    "DuoCorruption",
    "DuoSchedule",
    "duo_nelbo_token_loss",
    "duo_reverse_posterior",
    "compiled_duo_nelbo_token_loss",
    "compiled_duo_reverse_posterior",
    "sample_branch_antithetic_times",
    "sample_antithetic_times",
    "sample_categorical",
    "uniform_state_corruption",
)
