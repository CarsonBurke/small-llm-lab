"""V4 ablation: probes receive only the post-temporal predicted latent."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

import fresh_lejepa_train as v1
from fresh_lejepa_train_v4 import (
    FreshLeJEPAV4,
    SIGREG_PROJECTION_CHUNK,
    SIGREG_POSITION_CHUNK,
    _install_configurable_accumulation,
)
import train_gpt as baseline


ARCHITECTURE = "fresh_lejepa_v4_predicted_only_probe"


class FreshLeJEPAV4PredictedOnly(FreshLeJEPAV4):
    def probe_features(self, token_latent: Tensor, predicted: Tensor) -> Tensor:
        return torch.cat((torch.zeros_like(token_latent), predicted.detach()), dim=-1)


def main() -> None:
    original_main = _install_configurable_accumulation(default_steps=8)
    FreshLeJEPAV4PredictedOnly.return_loss_components = True
    v1.FreshHyperparameters.sigreg_proj_chunk = SIGREG_PROJECTION_CHUNK
    v1.FreshHyperparameters.sigreg_position_chunk = SIGREG_POSITION_CHUNK
    v1.FreshLeJEPAGPT = FreshLeJEPAV4PredictedOnly
    v1.EXPERIMENT_ARCHITECTURE = ARCHITECTURE
    v1.EXPERIMENT_SOURCE = Path(__file__)
    try:
        v1.main()
    finally:
        baseline.main = original_main


if __name__ == "__main__":
    main()
