"""Geometry-aware soft targets on the stripped no-sigreg energy head.

Same head, smarter CE: instead of a one-hot target, train against

    target = (1 - eps) * onehot(y)  +  eps * q
    q_k    = softmax_k( -1/2 * sum_d s_d (c_{y,d} - c_{k,d})^2 + b_k )

i.e. ``q`` is the head's own emission distribution evaluated at the true
token's code — "what the head would predict from a perfect prediction
z_hat = c_y".  This reuses the live per-dim metric and bias (all detached),
so there is no new temperature hyperparameter and the target sharpens as the
codebook spreads.  The codebook acts as a similarity graph: mass leaks only
toward geometrically confusable tokens, unlike uniform label smoothing.

eps = 0.1.  Applied in TRAINING ONLY: the eval path returns plain one-hot CE
so val_loss/val_bpb stay directly comparable with every sibling run (and with
the challenge metric).  Known risk, registered: soft targets usually cost
val likelihood (the reason plain label smoothing was rejected); the win
condition is that geometry-aware leakage acts as a better-than-nothing
regularizer at 2k.  A win here is provisional until it also beats plain
label smoothing at the same eps.

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
from torch import Tensor

import energy_readout.fresh_lejepa_train_energy_readout as energy
import energy_readout.fresh_lejepa_train_energy_readout_perdim_nosigreg as nosig

ARCHITECTURE = (
    "energy_readout_lejepa_tied_codebook_perdim_nosigreg_softgeo_onepass_2k"
)

SOFT_TARGET_EPS = 0.1


class EnergyReadoutSoftGeoLeJEPA(nosig.EnergyReadoutPerDimNoSigregLeJEPA):
    """Stripped no-sigreg energy readout trained on geometry-aware soft targets."""

    @torch.no_grad()
    def geometry_soft_targets(self, target_ids: Tensor) -> Tensor:
        """The head's own distribution at the true code, fully detached."""
        codebook = self.energy_codebook().detach()
        target_codes = codebook[target_ids]  # (B, T, d)
        q_logits = self.energy_logits(target_codes, None)
        return F.softmax(q_logits.float(), dim=-1)

    def forward(self, input_ids: Tensor, target_ids: Tensor):
        # Replicates the base forward; only the TRAINING policy loss changes
        # (eval returns plain CE for comparable BPB).
        token_latent, _belief, predicted, target_latent = (
            self.training_latents_with_belief(input_ids, target_ids)
        )
        logits = self.energy_logits(
            predicted, input_ids if energy.BIGRAM_TABLE else None
        )
        if not self.training:
            return F.cross_entropy(
                logits.float().flatten(0, 1), target_ids.flatten()
            )
        target_dist = self.geometry_soft_targets(target_ids).mul_(SOFT_TARGET_EPS)
        index = target_ids.unsqueeze(-1)
        target_dist.scatter_add_(
            -1,
            index,
            torch.full(
                index.shape,
                1.0 - SOFT_TARGET_EPS,
                dtype=target_dist.dtype,
                device=target_dist.device,
            ),
        )
        policy_loss = F.cross_entropy(
            logits.float().flatten(0, 1), target_dist.flatten(0, 1)
        )
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
                "policy_targets": "geometry_soft_own_head",
                "policy_target_eps": SOFT_TARGET_EPS,
            }
        )
        return metadata


def main() -> None:
    original_class = nosig.EnergyReadoutPerDimNoSigregLeJEPA
    original_architecture = nosig.ARCHITECTURE
    original_file = nosig.__file__
    nosig.EnergyReadoutPerDimNoSigregLeJEPA = EnergyReadoutSoftGeoLeJEPA
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
