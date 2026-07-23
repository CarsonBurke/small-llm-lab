"""Low-rank current-token residual on the stripped no-sigreg energy head.

The belief-attached probe consumed ``cat(token_latent, belief)``; the energy
head reads only the prediction ``z_hat``, deleting the current-token channel —
flagged by the red-team as the family's strongest BPB risk.  This arm restores
the channel in the code geometry instead of as a raw 1M-param bigram table:

    delta_logit_k = (U e_{x_t})^T (V c_k)        rank r = 32

``U`` is a (vocab, r) table addressed by the current token; ``V`` is an
(r, model_dim) projection of the live codebook, so the residual stays tied to
the same code geometry the head scores against.  LoRA-style init (U small
normal, V zero) makes the head numerically identical to the base at step 0
while giving V a nonzero gradient from the start.  Both factors are
registered flat (ndim 1) under the last block so the optimizer split routes
them to the fused-Adam scalar group with fp32 masters — the same routing as
``energy_bias`` and the lineage's bigram table.

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

import energy_readout.fresh_lejepa_train_energy_readout_perdim_nosigreg as nosig

ARCHITECTURE = (
    "energy_readout_lejepa_tied_codebook_perdim_nosigreg_lowrank_token_onepass_2k"
)

TOKEN_RESIDUAL_RANK = 32


class EnergyReadoutLowRankTokenLeJEPA(nosig.EnergyReadoutPerDimNoSigregLeJEPA):
    """Stripped no-sigreg energy readout plus a rank-32 current-token residual."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        owner = self.blocks[-1]
        vocab = self.tok_emb.num_embeddings
        model_dim = self.tok_emb.embedding_dim
        rank = TOKEN_RESIDUAL_RANK
        # Registered after the full base init so the trunk's RNG stream (and
        # therefore every shared parameter) is bit-identical to the base run
        # at the same seed.  Flat registration -> fused-Adam scalar group.
        owner.energy_token_factor = nn.Parameter(
            torch.randn(vocab * rank, dtype=torch.float32) * 0.02
        )
        owner.energy_token_proj = nn.Parameter(
            torch.zeros(rank * model_dim, dtype=torch.float32)
        )

    def energy_logits(self, predicted: Tensor, input_ids: Tensor | None) -> Tensor:
        logits = super().energy_logits(predicted, input_ids)
        if input_ids is None:
            raise RuntimeError(
                "the low-rank token residual requires current-token ids at the readout"
            )
        owner = self.blocks[-1]
        vocab = self.tok_emb.num_embeddings
        model_dim = self.tok_emb.embedding_dim
        rank = TOKEN_RESIDUAL_RANK
        codebook = self.energy_codebook()
        projected_codes = owner.energy_token_proj.view(rank, model_dim) @ (
            codebook.float().transpose(0, 1)
        )
        token_factors = owner.energy_token_factor.view(vocab, rank)[input_ids]
        return logits + token_factors @ projected_codes

    def forward(self, input_ids: Tensor, target_ids: Tensor):
        # Replicates the base forward with one change: current-token ids are
        # always passed to the readout (the base only passes them for the
        # bigram flag).  Latent-MSE and deferred-sigreg branches are kept
        # verbatim so loss components stay schema-identical.
        token_latent, _belief, predicted, target_latent = (
            self.training_latents_with_belief(input_ids, target_ids)
        )
        logits = self.energy_logits(predicted, input_ids)
        policy_loss = F.cross_entropy(
            logits.float().flatten(0, 1), target_ids.flatten()
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

    @classmethod
    def experiment_metadata(cls) -> dict[str, str | int | float]:
        metadata = super().experiment_metadata()
        metadata.update(
            {
                "token_residual": "low_rank_code_projected",
                "token_residual_rank": TOKEN_RESIDUAL_RANK,
            }
        )
        return metadata


def main() -> None:
    # nosig.main() resolves its class, architecture, and __file__ as module
    # globals at call time; patch all three and delegate so the stripped
    # installer, canaries, and run wiring carry over verbatim.
    original_class = nosig.EnergyReadoutPerDimNoSigregLeJEPA
    original_architecture = nosig.ARCHITECTURE
    original_file = nosig.__file__
    nosig.EnergyReadoutPerDimNoSigregLeJEPA = EnergyReadoutLowRankTokenLeJEPA
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
