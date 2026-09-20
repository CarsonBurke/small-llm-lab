"""Cola's staged objectives with explicit, checkpointed scaled-control coefficients.

The public paper supplies these terms, but not numerical beta/lambdas or an
executable masking recipe. Defaults below are control choices, not recovered
paper hyperparameters. Stage two uses negative entropy, NOT KL to a fixed base.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from pretraining.cola.model import ColaModel


@dataclass(frozen=True)
class LossConfig:
    beta: float = 0.001
    mask_weight: float = 1.0
    vae_weight: float = 1.0
    flow_weight: float = 1.0
    reference_weight: float = 0.1
    mask_probability: float = 0.15
    time_loc: float = 1.0
    time_scale: float = 1.0

    def __post_init__(self):
        for name in (
            "beta",
            "mask_weight",
            "vae_weight",
            "flow_weight",
            "reference_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")
        if not 0 < self.mask_probability < 1:
            raise ValueError("mask_probability must be strictly between zero and one")
        if (
            not math.isfinite(self.time_loc)
            or not math.isfinite(self.time_scale)
            or self.time_scale <= 0
        ):
            raise ValueError("invalid logit-normal time distribution")


STAT_NAMES = (
    "objective_sum",
    "reconstruction_nats",
    "correct_positions",
    "masked_nats",
    "correct_masked_positions",
    "masked_positions",
    "base_kl_nats",
    "negative_entropy_nats",
    "reference_kl_nats",
    "flow_squared_error",
    "mean_power_sum",
    "variance_sum",
    "positions",
    "latent_coordinates",
    "source_bytes",
    "low_clamp_coordinates",
    "high_clamp_coordinates",
)


def gaussian_terms(
    mean: Tensor, logvar: Tensor, reference_mean: Tensor, reference_logvar: Tensor
):
    """Coordinate-summed Gaussian terms; reference direction is current || frozen."""
    variance = logvar.exp()
    base_kl = 0.5 * (mean.square() + variance - 1 - logvar).sum()
    negative_entropy = -0.5 * (1 + math.log(2 * math.pi) + logvar).sum()
    reference_kl = (
        0.5
        * (
            reference_logvar
            - logvar
            + (variance + (mean - reference_mean).square()) * (-reference_logvar).exp()
            - 1
        ).sum()
    )
    return base_kl, negative_entropy, reference_kl


@torch.compile(fullgraph=True, dynamic=False)
def corrupt_inputs(ids: Tensor, masked: Tensor, mask_token_id: int):
    return torch.cat((ids, torch.where(masked, mask_token_id, ids)), dim=0)


@torch.compile(fullgraph=True, dynamic=False)
def sample_posterior(mean: Tensor, logvar: Tensor, noise: Tensor):
    return mean + (0.5 * logvar).exp() * noise


@torch.compile(fullgraph=True, dynamic=False)
def make_flow_inputs(latents: Tensor, noise: Tensor, times: Tensor):
    fraction = times[:, None, None]
    return (1 - fraction) * latents + fraction * noise, noise - latents


@torch.compile(fullgraph=True, dynamic=False)
def loss_tail(
    logits: Tensor,
    ids: Tensor,
    byte_lengths: Tensor,
    masked: Tensor,
    mean: Tensor,
    logvar: Tensor,
    reference_mean: Tensor,
    reference_logvar: Tensor,
    velocity: Tensor,
    target: Tensor,
    joint: bool,
    beta: float,
    mask_weight: float,
    vae_weight: float,
    flow_weight: float,
    reference_weight: float,
):
    batch = ids.shape[0]
    targets = torch.cat((ids, ids), dim=0)
    nll = F.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), reduction="none"
    ).view_as(targets)
    reconstruction = nll[:batch].sum()
    masked_nll = (nll[batch:] * masked).sum()
    masked_count = masked.sum()
    mean, logvar = mean[:batch], logvar[:batch]
    base_kl, negative_entropy, reference_kl = gaussian_terms(
        mean, logvar, reference_mean, reference_logvar
    )
    position_count = ids.numel()
    flow_error = (velocity - target).square().mean(dim=-1).sum()
    vae_objective = (
        reconstruction / position_count
        + mask_weight * masked_nll / masked_count.clamp_min(1)
    )
    if joint:
        objective = (
            vae_weight * (vae_objective + beta * negative_entropy / position_count)
            + flow_weight * flow_error / position_count
            + reference_weight * reference_kl / position_count
        )
    else:
        objective = vae_objective + beta * base_kl / position_count
    statistics = torch.stack(
        (
            objective.detach() * position_count,
            reconstruction.detach(),
            (logits[:batch].argmax(-1) == ids).sum(),
            masked_nll.detach(),
            ((logits[batch:].argmax(-1) == ids) * masked).sum(),
            masked_count,
            base_kl.detach(),
            negative_entropy.detach(),
            reference_kl.detach(),
            flow_error.detach(),
            mean.detach().square().sum(),
            logvar.detach().exp().sum(),
            mean.new_tensor(position_count),
            mean.new_tensor(mean.numel()),
            byte_lengths[ids].sum(),
            (logvar <= -30).sum(),
            (logvar >= 20).sum(),
        )
    ).float()
    return objective, statistics


class Objective:
    """Orchestrate separately compiled neural and reduction kernels.

    There is one clean and one masked VAE branch, batched together. Only clean
    history is detached for flow matching; the current reparameterized sample
    and the literal target (epsilon-z) both remain differentiable in stage two.
    """

    def __init__(
        self,
        model: ColaModel,
        config: LossConfig,
        byte_lengths: Tensor,
        reference: nn.Module | None = None,
    ):
        self.model, self.config, self.reference = model, config, reference
        self.byte_lengths = byte_lengths

    def __call__(self, ids: Tensor, generator: torch.Generator | None = None):
        shape = (*ids.shape, self.model.config.latent_dim)
        masked = (
            torch.rand(ids.shape, device=ids.device, generator=generator)
            < self.config.mask_probability
        )
        posterior_noise = torch.randn(
            (2 * ids.shape[0], *shape[1:]), device=ids.device, generator=generator
        )
        combined = corrupt_inputs(ids, masked, self.model.config.mask_token_id)
        mean, logvar = self.model.vae.encode(combined)
        latents = sample_posterior(mean, logvar, posterior_noise)
        logits = self.model.vae.decode(latents)
        clean = latents[: ids.shape[0]]
        joint = self.reference is not None
        if self.reference is not None:
            with torch.no_grad():
                reference_mean, reference_logvar = self.reference(ids)
            noise = torch.randn(shape, device=ids.device, generator=generator)
            times = torch.sigmoid(
                self.config.time_loc
                + self.config.time_scale
                * torch.randn((ids.shape[0],), device=ids.device, generator=generator)
            )
            noisy, target = make_flow_inputs(clean, noise, times)
            velocity = self.model.prior(
                clean.detach(), noisy, times[:, None].expand_as(ids)
            )
        else:
            reference_mean, reference_logvar = (
                mean[: ids.shape[0]].detach(),
                logvar[: ids.shape[0]].detach(),
            )
            velocity = target = torch.zeros_like(clean)
        return loss_tail(
            logits,
            ids,
            self.byte_lengths,
            masked,
            mean,
            logvar,
            reference_mean,
            reference_logvar,
            velocity,
            target,
            joint,
            self.config.beta,
            self.config.mask_weight,
            self.config.vae_weight,
            self.config.flow_weight,
            self.config.reference_weight,
        )


def summarize_statistics(values) -> dict[str, float]:
    totals = dict(zip(STAT_NAMES, map(float, values), strict=True))
    count, source_bytes = totals["positions"], totals["source_bytes"]
    if min(count, source_bytes) <= 0 or not all(
        math.isfinite(value) for value in totals.values()
    ):
        raise FloatingPointError("invalid Cola objective statistics")
    masked = max(1, totals["masked_positions"])
    return {
        "objective": totals["objective_sum"] / count,
        "reconstruction_nats_per_position": totals["reconstruction_nats"] / count,
        "reconstruction_bits_per_byte": totals["reconstruction_nats"]
        / (source_bytes * math.log(2)),
        "reconstruction_accuracy": totals["correct_positions"] / count,
        "masked_nats_per_masked_position": totals["masked_nats"] / masked,
        "masked_accuracy": totals["correct_masked_positions"] / masked,
        "base_kl_nats_per_position": totals["base_kl_nats"] / count,
        "negative_entropy_nats_per_position": totals["negative_entropy_nats"] / count,
        "reference_kl_nats_per_position": totals["reference_kl_nats"] / count,
        "flow_mse": totals["flow_squared_error"] / count,
        "latent_mean_power": totals["mean_power_sum"] / totals["latent_coordinates"],
        "latent_variance": totals["variance_sum"] / totals["latent_coordinates"],
        "latent_log_snr": math.log(
            max(totals["mean_power_sum"], 1e-30) / max(totals["variance_sum"], 1e-30)
        ),
        "latent_low_clamp_fraction": totals["low_clamp_coordinates"]
        / totals["latent_coordinates"],
        "latent_high_clamp_fraction": totals["high_clamp_coordinates"]
        / totals["latent_coordinates"],
        "source_bytes": source_bytes,
        "positions": count,
        "masked_positions": totals["masked_positions"],
    }
