"""V1 large-probe shared-projector ablation replacing only BN with RMSNorm."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from pathlib import Path

import torch.nn.functional as F
from torch import Tensor, nn

from pretraining.fresh_lejepa import fresh_lejepa_train as v1
from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_projector import (
    FreshLeJEPASharedProjectorV1Probes,
)
from train_gpt import CastedLinear, RMSNorm


ARCHITECTURE = "fresh_lejepa_shared_rms_predproj_v1_large_probes"


class RMSTokenProjector(nn.Module):
    """Linear -> per-token RMSNorm -> GELU -> Linear."""

    def __init__(self, model_dim: int, hidden_dim: int = 2048):
        super().__init__()
        self.input = CastedLinear(model_dim, hidden_dim)
        self.norm = RMSNorm()
        self.output = CastedLinear(hidden_dim, model_dim)

    def forward(self, latent: Tensor) -> Tensor:
        return self.output(F.gelu(self.norm(self.input(latent))))

    def inference(self, latent: Tensor) -> Tensor:
        return self(latent)


class FreshLeJEPASharedRMSProjectorV1Probes(
    FreshLeJEPASharedProjectorV1Probes
):
    projector_class = RMSTokenProjector


def main() -> None:
    v1.FreshLeJEPAGPT = FreshLeJEPASharedRMSProjectorV1Probes
    v1.EXPERIMENT_ARCHITECTURE = ARCHITECTURE
    v1.EXPERIMENT_SOURCE = Path(__file__)
    v1.main()


if __name__ == "__main__":
    main()
