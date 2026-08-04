"""Shared-point categorical energy and JEPA pretraining.

One causal prediction ``r_t`` serves both representation learning and token
generation.  The live target embedding table is also the vocabulary energy
codebook ``c_k``::

    p_t(k) = softmax_k(b_k - ||r_t - c_k||^2 / (2 sigma_t^2))

Exact token NLL supplies the full-vocabulary contrast: it attracts the
observed target and repels competing tokens according to their energy.  JEPA
MSE aligns that same energy center with the observed target code, while paired
SIGReg shapes the code geometry.  No probe, detached decoder representation,
categorical barycenter, or independently predicted latent is involved.

Run the canonical ablation through the shared GPU queue::

    mlq submit --name shared_point_energy_2k --cwd "$PWD" \
      --max-parallel-runs 1 -- python3 scripts/ablation.py --steps 2000 \
      --name shared_point_energy_2k \
      --script energy_readout/fresh_lejepa_train_shared_point_energy.py \
      --env DATA_PATH=data/datasets/fineweb_onepass_sp1024 \
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch import Tensor

from energy_readout import fresh_lejepa_train_jepa_rbf_decoder as rbf
from energy_readout.fresh_lejepa_train_unified_energy_barycenter import (
    UnifiedEnergyBarycenterLeJEPA,
)


ARCHITECTURE = "fresh_lejepa_pope_shared_point_energy_onepass_2k"


class SharedPointEnergyLeJEPA(UnifiedEnergyBarycenterLeJEPA):
    """Use one predicted point for both JEPA alignment and token energies."""

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        token_latent, belief, predicted, target_latent = (
            self.training_latents_with_belief(input_ids, target_ids)
        )
        logits = self.energy_logits(predicted, belief)
        policy_loss = F.cross_entropy(
            logits.flatten(0, 1),
            target_ids.flatten(),
            reduction="mean",
        )
        if not self.training:
            return policy_loss

        # The JEPA target is exactly the target token's entry in the same live
        # codebook used by the categorical energy.  Unlike expected distance
        # over the entire categorical law, this does not pull every competing
        # vocabulary code toward every observed target.
        latent_loss = F.mse_loss(predicted.float(), target_latent.float())
        if self.defer_sigreg:
            sigreg_loss = policy_loss.detach().new_zeros(())
        else:
            sigreg_loss = self.sigreg(
                self.training_sigreg_features(token_latent, target_latent)
            )
        total_loss = (
            policy_loss
            + self.latent_loss_weight * latent_loss
            + self.sigreg_loss_weight * sigreg_loss
        )
        if self.return_loss_components:
            return total_loss, torch.stack(
                (policy_loss.detach(), latent_loss.detach(), sigreg_loss.detach())
            )
        return total_loss

    @classmethod
    def experiment_metadata(cls) -> dict[str, str | int | float]:
        metadata = super().experiment_metadata()
        metadata.update(
            {
                "emission_head": "shared_point_categorical_energy",
                "jepa_prediction": "categorical_energy_center",
                "categorical_objective": "exact_token_nll",
                "geometric_objective": "point_to_observed_target_code_mse",
                "representation_distribution_link": (
                    "same_energy_center_and_target_codebook"
                ),
                "uncertainty_gradient": "attached_to_causal_belief",
                "prediction_gradient_from_token_nll": "attached",
                "codebook_gradient_from_token_nll": "attached",
                "geometry_owner": "joint_nll_point_mse_and_paired_sigreg",
                "distance_accumulation": "fp32_dot_and_code_norm",
                "continuous_latent_sampling": "none",
                "semantic_partial_credit": "target_code_distance_in_shared_geometry",
            }
        )
        return metadata


def main() -> None:
    rbf.run_experiment(
        SharedPointEnergyLeJEPA,
        ARCHITECTURE,
        Path(__file__),
    )


if __name__ == "__main__":
    main()
