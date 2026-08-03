"""V2 ablation: compact probes receive only the post-transformer predicted latent."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from pathlib import Path

import torch
from torch import Tensor

from pretraining.fresh_lejepa import fresh_lejepa_train as v1
from pretraining.fresh_lejepa import fresh_lejepa_train_v2 as v2


ARCHITECTURE = "fresh_lejepa_v2_predicted_only_probe"


class FreshLeJEPAV2PredictedOnly(v2.FreshLeJEPAGPTV2):
    def probe_features(self, token_latent: Tensor, predicted: Tensor) -> Tensor:
        # Preserve the exact 2D probe shape and parameter count while removing
        # all sample-specific information from the pre-temporal branch.
        return torch.cat((torch.zeros_like(token_latent), predicted.detach()), dim=-1)


def main() -> None:
    v1.FreshLeJEPAGPT = FreshLeJEPAV2PredictedOnly
    v1.EXPERIMENT_ARCHITECTURE = ARCHITECTURE
    v1.EXPERIMENT_SOURCE = Path(__file__)
    v1.main()


if __name__ == "__main__":
    main()
