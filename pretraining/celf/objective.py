"""CELF objective: decoder CE, masked CE, and flow matching.

No KL, entropy, reference-encoder, EMA, or isotropy term exists. Cross-entropy
from a decoder that reads only anchored latents is the anti-collapse anchor.
With ``attachment="all"`` the flow loss reaches the encoder through the history
stream, the noisy block, and the velocity target alike; with ``"history"`` only
through the history stream. With ``flow=False`` (codec stage) the prior is
not executed and the objective is the two CE terms alone.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import LossConfig

if TYPE_CHECKING:
    from .model import CelfModel

STAT_NAMES = (
    "objective_sum",
    "reconstruction_nats",
    "correct_bytes",
    "masked_nats",
    "correct_masked_bytes",
    "masked_bytes",
    "flow_squared_error",
    "patches",
    "bytes",
    "effective_rank_weighted",
    "sequences",
)


@torch.compile(fullgraph=True, dynamic=False)
def sample_times(normal: Tensor, decode_time: float, loc: float, scale: float) -> Tensor:
    """Logit-normal flow times restricted to the used path segment [decode_time, 1]."""
    return decode_time + (1.0 - decode_time) * torch.sigmoid(loc + scale * normal)


@torch.compile(fullgraph=True, dynamic=False)
def make_flow_inputs(latents: Tensor, noise: Tensor, times: Tensor):
    fraction = times[:, None, None]
    return (1 - fraction) * latents + fraction * noise, noise - latents


def effective_rank(z: Tensor) -> Tensor:
    """exp(entropy of normalized covariance eigenvalues) over all positions."""
    flat = z.detach().float().flatten(0, -2)
    centered = flat - flat.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(1, flat.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum()
    return entropy.exp()


@torch.compile(fullgraph=True, dynamic=False)
def loss_tail(
    logits: Tensor,
    patches: Tensor,
    masked: Tensor,
    velocity: Tensor,
    target: Tensor,
    mask_weight: float,
    flow_weight: float,
):
    batch = patches.shape[0]
    targets = torch.cat((patches, patches), dim=0)
    nll = F.cross_entropy(
        logits.flatten(0, 2), targets.flatten(), reduction="none"
    ).view_as(targets)
    reconstruction = nll[:batch].sum()
    masked_bytes_mask = masked[..., None].expand_as(patches)
    masked_nll = (nll[batch:] * masked_bytes_mask).sum()
    masked_count = masked_bytes_mask.sum()
    byte_count = patches.numel()
    patch_count = patches.shape[0] * patches.shape[1]
    flow_error = (velocity - target).square().mean(dim=-1).sum()
    objective = (
        reconstruction / byte_count
        + mask_weight * masked_nll / masked_count.clamp_min(1)
        + flow_weight * flow_error / patch_count
    )
    predictions = logits.argmax(-1)
    statistics = torch.stack(
        (
            objective.detach() * byte_count,
            reconstruction.detach(),
            (predictions[:batch] == patches).sum(),
            masked_nll.detach(),
            ((predictions[batch:] == patches) & masked_bytes_mask).sum(),
            masked_count,
            flow_error.detach(),
            velocity.new_tensor(patch_count),
            velocity.new_tensor(byte_count),
        )
    ).float()
    return objective, statistics


class Objective:
    """One clean branch and one masked branch, batched through the codec.

    Only the clean branch feeds the prior: its anchored latents are the
    history stream and its unit-power latents the flow endpoints.
    """

    def __init__(self, model: CelfModel, config: LossConfig):
        self.model, self.config = model, config

    def __call__(
        self, ids: Tensor, generator: torch.Generator | None = None, *, flow: bool = True
    ):
        """With ``flow=False`` (codec stage) the prior is not executed at all."""
        model_config = self.model.config
        batch = ids.shape[0]
        if ids.ndim != 2 or ids.shape[1] != model_config.seq_bytes:
            raise ValueError("objective consumes complete [B, seq_bytes] contexts")
        patches = ids.view(batch, model_config.seq_patches, model_config.patch_size)
        device = ids.device
        masked = (
            torch.rand(patches.shape[:2], device=device, generator=generator)
            < self.config.mask_probability
        )
        both = torch.cat((patches, patches), dim=0)
        both_masked = torch.cat((torch.zeros_like(masked), masked), dim=0)
        z = self.model.encoder(both, both_masked)
        anchor_noise = torch.randn(z.shape, device=device, generator=generator)
        w = self.model.anchor(z, anchor_noise)
        logits = self.model.decoder(w)
        z_clean, w_clean = z[:batch], w[:batch]
        if flow:
            times = sample_times(
                torch.randn((batch,), device=device, generator=generator),
                model_config.decode_time,
                self.config.time_loc,
                self.config.time_scale,
            )
            flow_noise = torch.randn(z_clean.shape, device=device, generator=generator)
            # 'history' keeps the encoder attached only through the conditioning
            # stream; the flow endpoint and target are then fixed for the encoder.
            endpoint = z_clean if self.config.attachment == "all" else z_clean.detach()
            noisy, target = make_flow_inputs(endpoint, flow_noise, times)
            velocity = self.model.prior(
                w_clean, noisy, times[:, None].expand(batch, model_config.seq_patches)
            )
        else:
            velocity = target = torch.zeros_like(z_clean)
        objective, statistics = loss_tail(
            logits,
            patches,
            masked,
            velocity,
            target,
            self.config.mask_weight,
            self.config.flow_weight,
        )
        rank = effective_rank(z_clean)
        statistics = torch.cat(
            (statistics, torch.stack((rank * batch, rank.new_tensor(batch))))
        )
        return objective, statistics


def summarize_statistics(values) -> dict[str, float]:
    totals = dict(zip(STAT_NAMES, map(float, values), strict=True))
    byte_count, patch_count = totals["bytes"], totals["patches"]
    if min(byte_count, patch_count, totals["sequences"]) <= 0 or not all(
        math.isfinite(value) for value in totals.values()
    ):
        raise FloatingPointError("invalid CELF objective statistics")
    masked = max(1.0, totals["masked_bytes"])
    return {
        "objective": totals["objective_sum"] / byte_count,
        "reconstruction_nats_per_byte": totals["reconstruction_nats"] / byte_count,
        "reconstruction_bpb": totals["reconstruction_nats"] / (byte_count * math.log(2)),
        "reconstruction_accuracy": totals["correct_bytes"] / byte_count,
        "masked_nats_per_byte": totals["masked_nats"] / masked,
        "masked_accuracy": totals["correct_masked_bytes"] / masked,
        "flow_mse": totals["flow_squared_error"] / patch_count,
        "latent_effective_rank": totals["effective_rank_weighted"] / totals["sequences"],
        "bytes": byte_count,
        "patches": patch_count,
        "masked_bytes": totals["masked_bytes"],
    }
