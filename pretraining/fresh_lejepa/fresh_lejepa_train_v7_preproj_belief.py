"""V7: actor and critic consume the pre-pred_proj contextual belief."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from pretraining.fresh_lejepa import fresh_lejepa_train as v1
from pretraining.fresh_lejepa.fresh_lejepa_train_v4 import (
    SIGREG_POSITION_CHUNK,
    SIGREG_PROJECTION_CHUNK,
    _install_configurable_accumulation,
)
from pretraining.fresh_lejepa.fresh_lejepa_train_v6_belief_dropout import FreshLeJEPAV6BeliefDropout
import train_gpt as baseline


ARCHITECTURE = "fresh_lejepa_v7_pre_predproj_belief_only_dropout01"


class FreshLeJEPAV7PreProjBelief(FreshLeJEPAV6BeliefDropout):
    def latent_features(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        token_latent = self.embed_tokens(input_ids)
        belief = self.temporal_belief_from_token_latent(token_latent)
        return token_latent, belief

    def generation_probe_features(
        self, token_latent: Tensor, belief: Tensor, predicted: Tensor
    ) -> Tensor:
        return belief.detach()

    def forward(self, input_ids: Tensor, target_ids: Tensor):
        trajectory_ids = torch.cat((input_ids, target_ids[:, -1:]), dim=1)
        trajectory_latent = self.embed_tokens(trajectory_ids)
        token_latent = trajectory_latent[:, :-1]
        target_latent = trajectory_latent[:, 1:]
        belief = self.temporal_belief_from_token_latent(token_latent)
        predicted = self.prediction_latent(belief)

        logits = self.logits_from_features(belief.detach())
        policy_loss = F.cross_entropy(
            logits.float().flatten(0, 1), target_ids.flatten()
        )
        if not self.training:
            return policy_loss

        latent_loss = F.mse_loss(predicted.float(), target_latent.float())
        sigreg_loss = policy_loss.detach().new_zeros(())
        total_loss = (
            policy_loss
            + self.latent_loss_weight * latent_loss
            + self.sigreg_loss_weight * sigreg_loss
        )
        if self.return_loss_components:
            return total_loss, torch.stack(
                (policy_loss.detach(), latent_loss.detach(), sigreg_loss)
            )
        return total_loss


def main() -> None:
    original_main = _install_configurable_accumulation(default_steps=8)
    FreshLeJEPAV7PreProjBelief.return_loss_components = True
    v1.FreshHyperparameters.sigreg_proj_chunk = SIGREG_PROJECTION_CHUNK
    v1.FreshHyperparameters.sigreg_position_chunk = SIGREG_POSITION_CHUNK
    v1.FreshLeJEPAGPT = FreshLeJEPAV7PreProjBelief
    v1.EXPERIMENT_ARCHITECTURE = ARCHITECTURE
    v1.EXPERIMENT_SOURCE = Path(__file__)
    try:
        v1.main()
    finally:
        baseline.main = original_main


if __name__ == "__main__":
    main()
