"""Mixture of prototype energies (MoS-style head) on the stripped base.

A single softmax over codebook energies is log-linear in ``z_hat``: the
softmax-bottleneck line (Yang et al.) shows the resulting log-prob matrix has
rank <= d+1, which a mixture breaks.  This arm keeps ONE codebook and ONE
per-dim metric (the geometry stays shared) and mixes M=4 components that
differ only in temperature and bias tilt, gated by the prediction:

    p(y | z) = sum_m pi_m(z) * softmax_k( exp(delta_m) * L_k + b_{k,m} )
    pi(z)    = softmax(W_g z + g_b)

where ``L`` is the base head's logit vector (per-dim scaled distance + bias).
At init delta_m = 0 and b_m = 0, so every component equals the base head and
the output is EXACTLY the base distribution regardless of the gate; the gate
weight is given small noise so component posteriors differ across samples,
which breaks the symmetry through the per-component bias gradients after the
first step.  New parameters (M*(V+1) + M*d + M ~ 6.7k) are registered flat
under the last block -> fused-Adam scalar group.

The mixture IS the model's predictive distribution, so validation loss (and
BPB) is the mixture NLL — this arm changes capacity, not just training.

Base: ``fresh_lejepa_train_energy_readout_perdim_nosigreg`` (per-dim scale,
BN projector, SIGReg apparatus removed).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import energy_readout.fresh_lejepa_train_energy_readout as energy
import energy_readout.fresh_lejepa_train_energy_readout_perdim_nosigreg as nosig

ARCHITECTURE = (
    "energy_readout_lejepa_tied_codebook_perdim_nosigreg_mixture4_onepass_2k"
)

MIXTURE_COMPONENTS = 4


class EnergyReadoutMixtureLeJEPA(nosig.EnergyReadoutPerDimNoSigregLeJEPA):
    """Stripped no-sigreg energy readout with an M=4 mixture-of-energies head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        owner = self.blocks[-1]
        vocab = self.tok_emb.num_embeddings
        model_dim = self.tok_emb.embedding_dim
        m = MIXTURE_COMPONENTS
        # Registered after the full base init: trunk RNG stream unchanged.
        # All flat (ndim 1) -> fused-Adam scalar group with fp32 masters.
        owner.energy_mix_bias = nn.Parameter(
            torch.zeros(m * vocab, dtype=torch.float32)
        )
        owner.energy_mix_logtemp = nn.Parameter(torch.zeros(m, dtype=torch.float32))
        owner.energy_gate_weight = nn.Parameter(
            torch.randn(m * model_dim, dtype=torch.float32) * 0.02
        )
        owner.energy_gate_bias = nn.Parameter(torch.zeros(m, dtype=torch.float32))

    def mixture_log_probs(
        self, predicted: Tensor, input_ids: Tensor | None
    ) -> Tensor:
        base_logits = self.energy_logits(predicted, input_ids)  # (B, T, V) fp32
        owner = self.blocks[-1]
        vocab = self.tok_emb.num_embeddings
        model_dim = self.tok_emb.embedding_dim
        m = MIXTURE_COMPONENTS
        gate_logits = predicted.float() @ owner.energy_gate_weight.view(
            m, model_dim
        ).transpose(0, 1) + owner.energy_gate_bias
        log_gate = F.log_softmax(gate_logits, dim=-1)  # (B, T, M)
        mix_bias = owner.energy_mix_bias.view(m, vocab)
        temps = torch.exp(owner.energy_mix_logtemp)
        # Accumulate log sum_m exp(log_pi_m + log_softmax_k(component_m)) one
        # component at a time to keep the peak at 2x(B,T,V) instead of Mx.
        log_probs: Tensor | None = None
        for index in range(m):
            component = F.log_softmax(
                temps[index] * base_logits + mix_bias[index], dim=-1
            )
            term = log_gate[..., index : index + 1] + component
            log_probs = term if log_probs is None else torch.logaddexp(
                log_probs, term
            )
        return log_probs

    def forward(self, input_ids: Tensor, target_ids: Tensor):
        # Replicates the base forward with the CE swapped for the mixture NLL
        # (train AND eval: the mixture is the model's true likelihood).
        token_latent, _belief, predicted, target_latent = (
            self.training_latents_with_belief(input_ids, target_ids)
        )
        log_probs = self.mixture_log_probs(
            predicted, input_ids if energy.BIGRAM_TABLE else None
        )
        policy_loss = F.nll_loss(
            log_probs.flatten(0, 1), target_ids.flatten()
        )
        if not self.training:
            return policy_loss
        total_loss = policy_loss
        latent_weight = type(self).latent_loss_weight
        if latent_weight != 0.0:
            latent_loss = F.mse_loss(predicted.float(), target_latent.float())
            total_loss = total_loss + latent_weight * latent_loss
            latent_component = latent_loss.detach()
        else:
            with torch.no_grad():
                latent_component = F.mse_loss(
                    predicted.float(), target_latent.float()
                )
        if self.defer_sigreg:
            sigreg_loss = policy_loss.detach().new_zeros(())
        else:
            sigreg_loss = self.sigreg(
                self.training_sigreg_features(token_latent, target_latent)
            )
            total_loss = total_loss + type(self).sigreg_loss_weight * sigreg_loss
        if self.return_loss_components:
            return total_loss, torch.stack(
                (policy_loss.detach(), latent_component, sigreg_loss.detach())
            )
        return total_loss

    def logits_from_features(self, features: Tensor) -> Tensor:
        # Mixture log-probs are valid sampling logits (softmax recovers p).
        return self.mixture_log_probs(features, None)

    @classmethod
    def experiment_metadata(cls) -> dict[str, str | int | float]:
        metadata = super().experiment_metadata()
        metadata.update(
            {
                "head_mixture": "shared_codebook_temp_bias_gate",
                "head_mixture_components": MIXTURE_COMPONENTS,
            }
        )
        return metadata


def main() -> None:
    original_class = nosig.EnergyReadoutPerDimNoSigregLeJEPA
    original_architecture = nosig.ARCHITECTURE
    original_file = nosig.__file__
    nosig.EnergyReadoutPerDimNoSigregLeJEPA = EnergyReadoutMixtureLeJEPA
    nosig.ARCHITECTURE = ARCHITECTURE
    nosig.__file__ = str(Path(__file__).resolve())
    try:
        nosig.main()
    finally:
        nosig.EnergyReadoutPerDimNoSigregLeJEPA = original_class
        nosig.ARCHITECTURE = original_architecture
        nosig.__file__ = original_file


if __name__ == "__main__":
    main()
