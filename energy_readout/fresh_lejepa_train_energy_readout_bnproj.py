"""Energy readout with the lineage's original FP32 BatchNorm projectors.

The shared-RMS fork replaced the BN ``TokenProjector`` with ``RMSTokenProjector``,
but the recorded 1k head-to-head favored BN (1.5719 vs 1.5817 BPB), and the
LeJEPA/LeWM references both use BatchNorm MLP projectors.  This arm restores
``TokenProjector`` (Linear -> FP32BatchNorm1d -> GELU -> Linear) for both the
latent and prediction projectors.

The codebook is the one place BN needs care: the vocab table is not a training
batch, so projecting it in training mode would normalize with (and pollute)
statistics from the wrong distribution.  ``TokenProjector.inference`` already
solves this — running statistics, no stat update — and gradients still flow
through the linears and BN affine parameters, so the codebook stays attached.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch.nn.functional as F
from torch import Tensor

import energy_readout.fresh_lejepa_train_energy_readout as energy
from pretraining.fresh_lejepa.fresh_lejepa_train_v2_sigreg_projector import TokenProjector

ARCHITECTURE = "energy_readout_lejepa_tied_codebook_bnproj_onepass_2k"


class EnergyReadoutBNProjLeJEPA(energy.EnergyReadoutLeJEPA):
    """Energy readout over BN projectors; codebook uses eval-statistics BN."""

    projector_class = TokenProjector

    def energy_codebook(self) -> Tensor:
        raw = F.rms_norm(self.tok_emb.weight, (self.tok_emb.embedding_dim,))
        codebook = self.latent_projector.inference(raw)
        return codebook.detach() if energy.DETACH_CODEBOOK else codebook

    @classmethod
    def experiment_metadata(cls) -> dict[str, str | int | float]:
        metadata = super().experiment_metadata()
        metadata.update(
            {
                "projector_norm": "fp32_batchnorm",
                "energy_codebook_bn_stats": "running_eval_mode",
            }
        )
        return metadata


def main() -> None:
    energy.EnergyReadoutLeJEPA = EnergyReadoutBNProjLeJEPA
    energy.ARCHITECTURE = ARCHITECTURE
    energy.main()


if __name__ == "__main__":
    main()
