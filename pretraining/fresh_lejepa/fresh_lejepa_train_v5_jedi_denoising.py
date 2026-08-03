"""V5 control with JEDI's conditional EDM objective in place of latent MSE.

This is deliberately a pretraining-only ablation.  The denoiser predicts the
next *token-local* projected embedding from the existing causal context; it is
not called by policy evaluation, cached token generation, or VAPO.  A genuine
latent-imagination rollout would instead need a recursively usable transition
whose target is the next causal belief.

Reference: JEDI, arXiv:2605.13013, equations (7)-(8).  As in JEDI, the future
target is stop-gradient while the conditioning context remains attached.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pretraining.fresh_lejepa import fresh_lejepa_train as v1
from pretraining.fresh_lejepa.fresh_lejepa_train_v4 import (
    SIGREG_POSITION_CHUNK,
    SIGREG_PROJECTION_CHUNK,
    _install_configurable_accumulation,
)
from pretraining.fresh_lejepa.fresh_lejepa_train_v5_swiglu import FreshLeJEPAV5SwiGLU
import train_gpt as baseline
from train_gpt import CastedLinear


ARCHITECTURE = "fresh_lejepa_v5_rms_jedi_edm_token_target"

# JEDI inherits DIAMOND's bounded log-normal training distribution and changes
# sigma_data to 1.  Environment overrides make follow-up noise ablations
# possible without silently changing the default paper-faithful setup.
EDM_P_MEAN = float(os.environ.get("FRESH_LEJEPA_EDM_P_MEAN", "-0.4"))
EDM_P_STD = float(os.environ.get("FRESH_LEJEPA_EDM_P_STD", "1.2"))
EDM_SIGMA_MIN = float(os.environ.get("FRESH_LEJEPA_EDM_SIGMA_MIN", "0.002"))
EDM_SIGMA_MAX = float(os.environ.get("FRESH_LEJEPA_EDM_SIGMA_MAX", "20.0"))
EDM_SIGMA_DATA = float(os.environ.get("FRESH_LEJEPA_EDM_SIGMA_DATA", "1.0"))


def edm_coefficients(
    sigma: Tensor, sigma_data: float = EDM_SIGMA_DATA
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Standard EDM preconditioning coefficients (Karras et al., 2022)."""
    sigma_data_tensor = sigma.new_tensor(sigma_data)
    denominator = (sigma.square() + sigma_data_tensor.square()).sqrt()
    c_skip = sigma_data_tensor.square() / denominator.square()
    c_out = sigma * sigma_data_tensor / denominator
    c_in = denominator.reciprocal()
    c_noise = sigma.log() / 4
    return c_skip, c_out, c_in, c_noise


def sinusoidal_noise_embedding(noise_level: Tensor, dim: int) -> Tensor:
    """Parameter-free per-example embedding of EDM's log-noise coordinate."""
    half = dim // 2
    if half == 0:
        return noise_level[:, None]
    frequencies = torch.exp(
        torch.linspace(0, math.log(1000), half, device=noise_level.device)
    )
    angles = noise_level[:, None] * frequencies[None]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if embedding.size(-1) < dim:
        embedding = F.pad(embedding, (0, dim - embedding.size(-1)))
    return embedding


class ConditionalEDMDenoiser(nn.Module):
    """Compact D-wide denoiser conditioned on one causal context per token.

    The context is the V5 projected causal belief, so its existing prediction
    projector remains trained and the rest of V5 is held constant.  Two simple
    residual layers keep this substantially cheaper than an image diffusion
    U-Net while retaining explicit noisy-target, context, and time paths.
    """

    def __init__(self, model_dim: int):
        super().__init__()
        self.model_dim = model_dim
        self.noisy_input = CastedLinear(model_dim, model_dim)
        self.context_input = CastedLinear(model_dim, model_dim)
        self.noise_input = CastedLinear(model_dim, model_dim)
        self.hidden1 = CastedLinear(model_dim, model_dim)
        self.hidden2 = CastedLinear(model_dim, model_dim)
        self.output = CastedLinear(model_dim, model_dim)
        self.output._zero_init = True
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def network_output(
        self, noisy_target: Tensor, context: Tensor, sigma: Tensor
    ) -> Tensor:
        _, _, c_in, c_noise = edm_coefficients(sigma)
        shape = (sigma.size(0),) + (1,) * (noisy_target.ndim - 1)
        scaled_noisy = noisy_target * c_in.view(shape).to(noisy_target.dtype)
        noise_embedding = sinusoidal_noise_embedding(c_noise, self.model_dim)
        noise_embedding = noise_embedding[:, None, :].to(noisy_target.dtype)
        hidden = (
            self.noisy_input(scaled_noisy)
            + self.context_input(context)
            + self.noise_input(noise_embedding)
        )
        hidden = hidden + F.silu(
            self.hidden1(F.rms_norm(hidden, (hidden.size(-1),)))
        )
        hidden = hidden + F.silu(
            self.hidden2(F.rms_norm(hidden, (hidden.size(-1),)))
        )
        return self.output(F.rms_norm(hidden, (hidden.size(-1),)))

    def forward(
        self, noisy_target: Tensor, context: Tensor, sigma: Tensor
    ) -> Tensor:
        """Return the EDM-preconditioned estimate of the clean target."""
        c_skip, c_out, _, _ = edm_coefficients(sigma)
        shape = (sigma.size(0),) + (1,) * (noisy_target.ndim - 1)
        residual = self.network_output(noisy_target, context, sigma)
        return (
            c_skip.view(shape).to(noisy_target.dtype) * noisy_target
            + c_out.view(shape).to(noisy_target.dtype) * residual
        )


class FreshLeJEPAV5JEDIDenoising(FreshLeJEPAV5SwiGLU):
    """Replace only V5's direct next-latent MSE with conditional EDM."""

    edm_p_mean = EDM_P_MEAN
    edm_p_std = EDM_P_STD
    edm_sigma_min = EDM_SIGMA_MIN
    edm_sigma_max = EDM_SIGMA_MAX

    def __init__(self, *args, **kwargs):
        model_dim = kwargs.get("model_dim", args[2] if len(args) > 2 else None)
        super().__init__(*args, **kwargs)
        if model_dim is None:
            raise ValueError("model_dim is required")
        # Preserve the control's global RNG lineage: adding the auxiliary must
        # not perturb any later stochastic setup outside model construction.
        with torch.random.fork_rng(devices=[]):
            self.blocks[-1].latent_denoiser = ConditionalEDMDenoiser(model_dim)

    @property
    def latent_denoiser(self) -> ConditionalEDMDenoiser:
        return self.blocks[-1].latent_denoiser

    def sample_edm_sigma(self, batch_size: int, device: torch.device) -> Tensor:
        log_sigma = torch.randn(batch_size, device=device, dtype=torch.float32)
        return (log_sigma * self.edm_p_std + self.edm_p_mean).exp().clamp(
            self.edm_sigma_min, self.edm_sigma_max
        )

    def latent_denoising_loss(
        self,
        context: Tensor,
        target_latent: Tensor,
        sigma: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """JEDI/EDM F-space loss with an attached context and detached target."""
        clean = target_latent.detach()
        if sigma is None:
            sigma = self.sample_edm_sigma(clean.size(0), clean.device)
        else:
            sigma = sigma.to(device=clean.device, dtype=torch.float32)
        # Construct the perturbed target and F-space regression target in
        # FP32.  At EDM's low-sigma tail, BF16 can round the perturbation away
        # before the subtraction below and turn an O(1) score target into an
        # unrelated near-zero value.
        clean_fp32 = clean.float()
        if noise is None:
            noise_fp32 = torch.randn(
                clean.shape, device=clean.device, dtype=torch.float32
            )
        else:
            noise_fp32 = noise.to(device=clean.device, dtype=torch.float32)
        broadcast = (sigma.size(0),) + (1,) * (clean.ndim - 1)
        sigma_view = sigma.view(broadcast)
        noisy_fp32 = clean_fp32 + sigma_view * noise_fp32

        c_skip, c_out, _, _ = edm_coefficients(sigma)
        # Equation (8): predicting this preconditioned residual is equivalent
        # to EDM-weighted MSE on the reconstructed clean target.
        network_target = (
            clean_fp32 - c_skip.view(broadcast) * noisy_fp32
        ) / c_out.view(broadcast).clamp_min(1e-12)
        network_prediction = self.latent_denoiser.network_output(
            noisy_fp32.to(context.dtype), context, sigma
        )
        return F.mse_loss(network_prediction.float(), network_target)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        token_latent, predicted, target_latent = self.training_latents(
            input_ids, target_ids
        )
        logits = self.logits_from_features(
            self.probe_features(token_latent, predicted)
        )
        policy_loss = F.cross_entropy(
            logits.float().flatten(0, 1), target_ids.flatten()
        )
        if not self.training:
            return policy_loss

        latent_loss = self.latent_denoising_loss(predicted, target_latent)
        if self.defer_sigreg:
            sigreg_loss = policy_loss.detach().new_zeros(())
        else:
            sigreg_loss = self.sigreg(
                self.training_sigreg_features(token_latent, target_latent)
            )
        total_loss = (
            policy_loss
            + v1.FreshHyperparameters.latent_loss_weight * latent_loss
            + v1.FreshHyperparameters.sigreg_weight * sigreg_loss
        )
        if self.return_loss_components:
            return total_loss, torch.stack(
                (policy_loss.detach(), latent_loss.detach(), sigreg_loss.detach())
            )
        return total_loss


def main() -> None:
    original_main = _install_configurable_accumulation(default_steps=8)
    FreshLeJEPAV5JEDIDenoising.return_loss_components = True
    v1.FreshHyperparameters.sigreg_proj_chunk = SIGREG_PROJECTION_CHUNK
    v1.FreshHyperparameters.sigreg_position_chunk = SIGREG_POSITION_CHUNK
    v1.FreshLeJEPAGPT = FreshLeJEPAV5JEDIDenoising
    v1.EXPERIMENT_ARCHITECTURE = ARCHITECTURE
    v1.EXPERIMENT_SOURCE = Path(__file__)
    try:
        v1.main()
    finally:
        baseline.main = original_main


if __name__ == "__main__":
    main()
