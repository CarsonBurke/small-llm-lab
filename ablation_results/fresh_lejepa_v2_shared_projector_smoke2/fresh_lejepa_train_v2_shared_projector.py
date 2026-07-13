"""V2 ablation: intra-token projector shared by SIGReg, predictor, target, and probe."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

import fresh_lejepa_train as v1
from fresh_lejepa_train_v2_sigreg_projector import (
    FreshLeJEPAV2SIGRegProjector,
)


ARCHITECTURE = "fresh_lejepa_v2_shared_token_projector"


class FreshLeJEPAV2SharedProjector(FreshLeJEPAV2SIGRegProjector):
    def __init__(self, *args, **kwargs):
        model_dim = kwargs.get("model_dim", args[2] if len(args) > 2 else None)
        super().__init__(*args, **kwargs)
        if model_dim is None:
            raise ValueError("model_dim is required")
        self.blocks[-1].prediction_projector = type(self.latent_projector)(model_dim)

    @property
    def prediction_projector(self):
        return self.blocks[-1].prediction_projector

    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        raw = super().embed_tokens(input_ids)
        return self.latent_projector(raw)

    def sigreg_features(self, token_latent: Tensor) -> Tensor:
        return token_latent

    def prediction_latent(self, predicted: Tensor) -> Tensor:
        return self.prediction_projector(predicted)

    def training_latents(
        self, input_ids: Tensor, target_ids: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # The language-model batch is one trajectory shifted by a token.  As in
        # LeWM, project that trajectory once and slice context/target latents
        # from the same tensor so BatchNorm statistics and overlapping tokens
        # are shared exactly.
        trajectory_ids = torch.cat((input_ids, target_ids[:, -1:]), dim=1)
        trajectory_latent = self.embed_tokens(trajectory_ids)
        token_latent = trajectory_latent[:, :-1]
        target_latent = trajectory_latent[:, 1:]
        predicted = self.predict_from_token_latent(token_latent)
        return token_latent, predicted, target_latent

    def policy_codebook(self) -> Tensor:
        raw = F.rms_norm(self.tok_emb.weight, (self.tok_emb.embedding_dim,))
        return self.latent_projector.inference(raw).detach()


def main() -> None:
    v1.FreshLeJEPAGPT = FreshLeJEPAV2SharedProjector
    v1.EXPERIMENT_ARCHITECTURE = ARCHITECTURE
    v1.EXPERIMENT_SOURCE = Path(__file__)
    v1.main()


if __name__ == "__main__":
    main()
