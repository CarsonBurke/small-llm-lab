"""Unified categorical-energy and JEPA mean-embedding pretraining.

The categorical distribution and JEPA prediction are the same mathematical
object.  A causal natural parameter and one learned log-radius induce a full
vocabulary distribution through the shared target codebook::

    p_t(k) = softmax_k(b_k - ||r_t - c_k||^2 / (2 sigma_t^2))
    m_t = sum_k p_t(k) c_k

Categorical NLL makes ``p_t`` useful for exact token sampling.  The geometric
loss is its expected squared code distance to the observed target.  This is
exactly barycenter MSE plus predictive code variance, so it retains JEPA mean
alignment without allowing probability on opposite distant codes to cancel.
There is no independent vocabulary probe or detached emission representation.

Run the canonical ablation through the shared GPU queue::

    mlq submit --name unified_energy_barycenter_2k --cwd "$PWD" \
      --max-parallel-runs 1 -- python3 scripts/ablation.py --steps 2000 \
      --name unified_energy_barycenter_2k \
      --script energy_readout/fresh_lejepa_train_unified_energy_barycenter.py \
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


ARCHITECTURE = "fresh_lejepa_pope_unified_energy_barycenter_onepass_2k"


class UnifiedEnergyBarycenterLeJEPA(rbf.GeometryPreservingRBFLeJEPA):
    """One energy distribution jointly serving token NLL and JEPA MSE."""

    def semantic_codebook(self) -> Tensor:
        """Return the live target/emission codebook with gradients attached."""
        raw = F.rms_norm(
            self.tok_emb.weight,
            (self.tok_emb.embedding_dim,),
        )
        return self.latent_projector(raw)

    def position_log_sigma(self, belief: Tensor) -> Tensor:
        """Predict one radius while retaining its connection to belief."""
        owner = self.rbf_owner
        weight = owner.rbf_log_sigma_weight.to(dtype=belief.dtype)
        projected = F.linear(belief, weight.unsqueeze(0)).squeeze(-1)
        return projected.float() + owner.rbf_log_sigma_bias

    def bounded_precision(self, log_sigma: Tensor) -> Tensor:
        """Exponentiate log precision inside the established safe domain."""
        log_precision = -2.0 * log_sigma
        log_precision_min, log_precision_max = self.precision_log_bounds(
            log_precision
        )
        bounded = log_precision.clamp(
            min=log_precision_min,
            max=log_precision_max,
        )
        safe = log_precision + (bounded - log_precision).detach()
        return safe.exp()

    def energy_logits_and_codebook(
        self,
        predicted: Tensor,
        belief: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return fast distance-equivalent logits and their live codebook."""
        codebook = self.semantic_codebook()
        precision = self.bounded_precision(
            self.position_log_sigma(belief)
        ).unsqueeze(-1)

        # Expanding squared distance gives
        #   precision * (r @ c - ||c||^2 / 2)
        # plus a per-position constant.  Softmax removes that constant, so we
        # avoid both a redundant norm and the cancellation-prone cdist path.
        with torch.autocast(device_type=predicted.device.type, enabled=False):
            predicted_fp32 = predicted.float()
            codebook_fp32 = codebook.float()
            dots = predicted_fp32 @ codebook_fp32.transpose(0, 1)
            codebook_sq = codebook_fp32.square().sum(dim=-1)
        logits = self.rbf_owner.rbf_token_bias + precision * (
            dots - 0.5 * codebook_sq
        )
        return logits, codebook

    def energy_logits(self, predicted: Tensor, belief: Tensor) -> Tensor:
        logits, _codebook = self.energy_logits_and_codebook(predicted, belief)
        return logits

    @staticmethod
    def categorical_statistics(
        logits: Tensor,
        codebook: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return the semantic mean and total variance of a token law."""
        probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            codebook_fp32 = codebook.float()
            predicted_mean = probabilities @ codebook_fp32
            expected_norm_sq = probabilities @ codebook_fp32.square().sum(
                dim=-1
            )
            variance = expected_norm_sq - predicted_mean.square().sum(dim=-1)
        return predicted_mean, variance.clamp_min(0.0)

    @classmethod
    def categorical_barycenter(
        cls,
        logits: Tensor,
        codebook: Tensor,
    ) -> Tensor:
        """Map the sampled categorical law to its semantic mean embedding."""
        predicted_mean, _variance = cls.categorical_statistics(logits, codebook)
        return predicted_mean

    @classmethod
    def semantic_distribution_loss(
        cls,
        logits: Tensor,
        codebook: Tensor,
        target_code: Tensor,
    ) -> Tensor:
        """Expected per-dimension squared distance from tokens to target."""
        predicted_mean, variance = cls.categorical_statistics(logits, codebook)
        mean_distance_sq = (predicted_mean - target_code.float()).square().sum(
            dim=-1
        )
        return ((mean_distance_sq + variance) / codebook.size(-1)).mean()

    def distribution_outputs(
        self,
        predicted: Tensor,
        belief: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return token logits and the JEPA prediction they imply."""
        logits, codebook = self.energy_logits_and_codebook(predicted, belief)
        return logits, self.categorical_barycenter(logits, codebook)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        token_latent, belief, predicted, target_latent = (
            self.training_latents_with_belief(input_ids, target_ids)
        )
        logits, codebook = self.energy_logits_and_codebook(predicted, belief)
        policy_loss = F.cross_entropy(
            logits.flatten(0, 1),
            target_ids.flatten(),
            reduction="mean",
        )
        if not self.training:
            return policy_loss

        latent_loss = self.semantic_distribution_loss(
            logits,
            codebook,
            target_latent,
        )
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
                "emission_head": "unified_categorical_energy_barycenter",
                "jepa_prediction": "categorical_codebook_mean_embedding",
                "categorical_objective": "exact_token_nll",
                "geometric_objective": "expected_squared_target_code_distance",
                "representation_distribution_link": "identity_via_probability_barycenter",
                "uncertainty_gradient": "attached_to_causal_belief",
                "prediction_gradient_from_token_nll": "attached",
                "codebook_gradient_from_token_nll": "attached",
                "geometry_owner": "joint_nll_barycenter_mse_and_paired_sigreg",
                "distance_accumulation": "fp32_dot_and_code_norm",
                "continuous_latent_sampling": "none",
                "semantic_partial_credit": "tokenwise_expected_code_distance",
            }
        )
        return metadata


def main() -> None:
    rbf.run_experiment(
        UnifiedEnergyBarycenterLeJEPA,
        ARCHITECTURE,
        Path(__file__),
    )


if __name__ == "__main__":
    main()
