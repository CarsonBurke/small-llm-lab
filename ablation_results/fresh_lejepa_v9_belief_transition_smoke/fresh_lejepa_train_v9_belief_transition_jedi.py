"""V9: V5 plus an action-conditioned JEDI transition between causal beliefs.

The proven V5 next-token MSE remains intact.  The additional transition learns

    (belief_t, token_action_t+1) -> belief_t+1

with an EDM denoising objective whose future teacher belief is stop-gradient.
This is the transition required for later latent-imagination rollouts: emitted
tokens remain actions, while the state can advance recursively in belief space.
The current BPB path remains teacher-forced and exactly V5 at inference.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import fresh_lejepa_train as v1
from fresh_lejepa_train_v4 import (
    SIGREG_POSITION_CHUNK,
    SIGREG_PROJECTION_CHUNK,
    _install_configurable_accumulation,
)
from fresh_lejepa_train_v5_jedi_denoising import (
    EDM_P_MEAN,
    EDM_P_STD,
    EDM_SIGMA_DATA,
    EDM_SIGMA_MAX,
    EDM_SIGMA_MIN,
    edm_coefficients,
    sinusoidal_noise_embedding,
)
from fresh_lejepa_train_v5_swiglu import FreshLeJEPAV5SwiGLU
import train_gpt as baseline
from train_gpt import CastedLinear


ARCHITECTURE = "fresh_lejepa_v9_action_conditioned_belief_transition_edm"
TRANSITION_LOSS_WEIGHT = float(
    os.environ.get("FRESH_LEJEPA_TRANSITION_WEIGHT", "0.1")
)
CONDITION_DIAGNOSTIC_STRIDE = int(
    os.environ.get("FRESH_LEJEPA_CONDITION_DIAG_STRIDE", "64")
)


class BeliefTransitionEDM(nn.Module):
    """D-wide EDM denoiser conditioned on current belief and token action."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.model_dim = model_dim
        self.noisy_input = CastedLinear(model_dim, model_dim)
        self.belief_input = CastedLinear(model_dim, model_dim)
        self.action_input = CastedLinear(model_dim, model_dim)
        self.noise_input = CastedLinear(model_dim, model_dim)
        self.hidden1 = CastedLinear(model_dim, model_dim)
        self.hidden2 = CastedLinear(model_dim, model_dim)
        self.output = CastedLinear(model_dim, model_dim)
        self.output._zero_init = True
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def network_output(
        self,
        noisy_next_belief: Tensor,
        current_belief: Tensor,
        token_action: Tensor,
        sigma: Tensor,
    ) -> Tensor:
        _, _, c_in, c_noise = edm_coefficients(sigma)
        shape = (sigma.size(0),) + (1,) * (noisy_next_belief.ndim - 1)
        scaled_noisy = noisy_next_belief * c_in.view(shape).to(
            noisy_next_belief.dtype
        )
        noise_embedding = sinusoidal_noise_embedding(c_noise, self.model_dim)
        noise_embedding = noise_embedding[:, None, :].to(noisy_next_belief.dtype)
        hidden = (
            self.noisy_input(scaled_noisy)
            + self.belief_input(current_belief)
            + self.action_input(token_action)
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
        self,
        noisy_next_belief: Tensor,
        current_belief: Tensor,
        token_action: Tensor,
        sigma: Tensor,
    ) -> Tensor:
        c_skip, c_out, _, _ = edm_coefficients(sigma)
        shape = (sigma.size(0),) + (1,) * (noisy_next_belief.ndim - 1)
        residual = self.network_output(
            noisy_next_belief, current_belief, token_action, sigma
        )
        return (
            c_skip.view(shape).to(noisy_next_belief.dtype) * noisy_next_belief
            + c_out.view(shape).to(noisy_next_belief.dtype) * residual
        )


class FreshLeJEPAV9BeliefTransition(FreshLeJEPAV5SwiGLU):
    """Preserve V5 BPB training and add a recursively usable state transition."""

    transition_loss_weight = TRANSITION_LOSS_WEIGHT
    transition_condition_gain_weight = 0.0
    edm_p_mean = EDM_P_MEAN
    edm_p_std = EDM_P_STD
    edm_sigma_min = EDM_SIGMA_MIN
    edm_sigma_max = EDM_SIGMA_MAX

    def __init__(self, *args, **kwargs):
        model_dim = kwargs.get("model_dim", args[2] if len(args) > 2 else None)
        super().__init__(*args, **kwargs)
        if model_dim is None:
            raise ValueError("model_dim is required")
        # Preserve all V5 initialization and the caller's subsequent RNG state.
        with torch.random.fork_rng(devices=[]):
            self.blocks[-1].belief_transition = BeliefTransitionEDM(model_dim)

    @property
    def belief_transition(self) -> BeliefTransitionEDM:
        return self.blocks[-1].belief_transition

    def sample_edm_sigma(self, batch_size: int, device: torch.device) -> Tensor:
        log_sigma = torch.randn(batch_size, device=device, dtype=torch.float32)
        return (log_sigma * self.edm_p_std + self.edm_p_mean).exp().clamp(
            self.edm_sigma_min, self.edm_sigma_max
        )

    def teacher_transition_latents(
        self, input_ids: Tensor, target_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Encode T+1 tokens once and return causal state/action transitions."""
        trajectory_ids = torch.cat((input_ids, target_ids[:, -1:]), dim=1)
        trajectory_tokens = self.embed_tokens(trajectory_ids)
        trajectory_beliefs = self.temporal_belief_from_token_latent(
            trajectory_tokens
        )
        current_tokens = trajectory_tokens[:, :-1]
        token_actions = trajectory_tokens[:, 1:]
        current_beliefs = trajectory_beliefs[:, :-1]
        next_beliefs = trajectory_beliefs[:, 1:]
        predicted_tokens = self.prediction_latent(current_beliefs)
        return (
            current_tokens,
            predicted_tokens,
            token_actions,
            current_beliefs,
            next_beliefs,
        )

    def transition_denoising_loss(
        self,
        current_belief: Tensor,
        token_action: Tensor,
        next_teacher_belief: Tensor,
        sigma: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return EDM loss and a cheap shuffled-conditioning usefulness metric."""
        clean = next_teacher_belief.detach()
        if sigma is None:
            sigma = self.sample_edm_sigma(clean.size(0), clean.device)
        else:
            sigma = sigma.to(device=clean.device, dtype=torch.float32)
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
        c_skip, c_out, _, _ = edm_coefficients(sigma, EDM_SIGMA_DATA)
        network_target = (
            clean_fp32 - c_skip.view(broadcast) * noisy_fp32
        ) / c_out.view(broadcast).clamp_min(1e-12)
        noisy = noisy_fp32.to(current_belief.dtype)
        prediction = self.belief_transition.network_output(
            noisy, current_belief, token_action, sigma
        )
        loss = F.mse_loss(prediction.float(), network_target)

        # One position per stride makes conditioning-use observable at little
        # cost.  Positive gain means correct state/action conditioning beats a
        # batch-shuffled transition under identical target noise.
        stride = max(CONDITION_DIAGNOSTIC_STRIDE, 1)
        selected = slice(None, None, stride)
        if current_belief.size(0) > 1:
            shuffled_belief = current_belief.roll(1, dims=0)
            shuffled_action = token_action.roll(1, dims=0)
        else:
            shuffled_belief = torch.zeros_like(current_belief)
            shuffled_action = torch.zeros_like(token_action)
        shuffled_prediction = self.belief_transition.network_output(
            noisy[:, selected],
            shuffled_belief[:, selected],
            shuffled_action[:, selected],
            sigma,
        )
        selected_target = network_target[:, selected]
        correct_selected = prediction[:, selected].float()
        condition_gain = (
            F.mse_loss(shuffled_prediction.float(), selected_target)
            - F.mse_loss(correct_selected, selected_target)
        ).detach()
        return loss, condition_gain

    @torch.no_grad()
    def imagine_next_belief(
        self,
        current_belief: Tensor,
        token_action: Tensor,
        noise: Tensor | None = None,
        steps: int = 3,
        rho: float = 7.0,
    ) -> Tensor:
        """Advance one latent state with the paper's three-step Euler solver."""
        if steps < 1:
            raise ValueError("diffusion steps must be positive")
        if noise is None:
            noise = torch.randn_like(current_belief)
        ramp = torch.linspace(0, 1, steps, device=current_belief.device)
        maximum = self.edm_sigma_max ** (1 / rho)
        minimum = self.edm_sigma_min ** (1 / rho)
        sigmas = (maximum + ramp * (minimum - maximum)).pow(rho)
        sigmas = torch.cat((sigmas, sigmas.new_zeros(1)))
        state = noise * sigmas[0].to(noise.dtype)
        for index in range(steps):
            sigma = sigmas[index].expand(current_belief.size(0))
            denoised = self.belief_transition(
                state, current_belief, token_action, sigma
            )
            derivative = (state - denoised) / sigma.view(-1, 1, 1).to(state.dtype)
            state = state + (
                sigmas[index + 1] - sigmas[index]
            ).to(state.dtype) * derivative
        return state

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        (
            token_latent,
            predicted,
            target_token_latent,
            current_belief,
            next_teacher_belief,
        ) = self.teacher_transition_latents(input_ids, target_ids)
        logits = self.logits_from_features(
            self.probe_features(token_latent, predicted)
        )
        policy_loss = F.cross_entropy(
            logits.float().flatten(0, 1), target_ids.flatten()
        )
        if not self.training:
            return policy_loss

        # Preserve V5's attached direct prediction target exactly.
        latent_loss = F.mse_loss(predicted.float(), target_token_latent.float())
        transition_loss, condition_gain = self.transition_denoising_loss(
            current_belief, target_token_latent, next_teacher_belief
        )
        if self.defer_sigreg:
            sigreg_loss = policy_loss.detach().new_zeros(())
        else:
            sigreg_loss = self.sigreg(
                self.training_sigreg_features(token_latent, target_token_latent)
            )
        total_loss = (
            policy_loss
            + self.latent_loss_weight * latent_loss
            + self.sigreg_loss_weight * sigreg_loss
            + self.transition_loss_weight * transition_loss
        )
        if self.return_loss_components:
            return total_loss, torch.stack(
                (
                    policy_loss.detach(),
                    latent_loss.detach(),
                    sigreg_loss.detach(),
                    transition_loss.detach(),
                    condition_gain,
                )
            )
        return total_loss


def main() -> None:
    original_main = _install_configurable_accumulation(
        default_steps=8,
        extra_components=(
            ("transition_loss", "transition_loss_weight"),
            ("transition_condition_gain", "transition_condition_gain_weight"),
        ),
    )
    FreshLeJEPAV9BeliefTransition.return_loss_components = True
    v1.FreshHyperparameters.sigreg_proj_chunk = SIGREG_PROJECTION_CHUNK
    v1.FreshHyperparameters.sigreg_position_chunk = SIGREG_POSITION_CHUNK
    v1.FreshLeJEPAGPT = FreshLeJEPAV9BeliefTransition
    v1.EXPERIMENT_ARCHITECTURE = ARCHITECTURE
    v1.EXPERIMENT_SOURCE = Path(__file__)
    try:
        v1.main()
    finally:
        baseline.main = original_main


if __name__ == "__main__":
    main()
