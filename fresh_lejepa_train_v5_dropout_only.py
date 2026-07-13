"""V5 controlled ablation: add only temporal-predictor dropout 0.1."""

from __future__ import annotations

from pathlib import Path

import fresh_lejepa_train as v1
from fresh_lejepa_train_v4 import (
    SIGREG_POSITION_CHUNK,
    SIGREG_PROJECTION_CHUNK,
    _install_configurable_accumulation,
)
from fresh_lejepa_train_v4_predictor_dropout import PredictorDropoutMixin
from fresh_lejepa_train_v5_swiglu import FreshLeJEPAV5SwiGLU
import train_gpt as baseline


ARCHITECTURE = "fresh_lejepa_v5_prenorm_swiglu_predictor_dropout01_only"


class FreshLeJEPAV5DropoutOnly(PredictorDropoutMixin, FreshLeJEPAV5SwiGLU):
    pass


def main() -> None:
    original_main = _install_configurable_accumulation(default_steps=8)
    FreshLeJEPAV5DropoutOnly.return_loss_components = True
    v1.FreshHyperparameters.sigreg_proj_chunk = SIGREG_PROJECTION_CHUNK
    v1.FreshHyperparameters.sigreg_position_chunk = SIGREG_POSITION_CHUNK
    v1.FreshLeJEPAGPT = FreshLeJEPAV5DropoutOnly
    v1.EXPERIMENT_ARCHITECTURE = ARCHITECTURE
    v1.EXPERIMENT_SOURCE = Path(__file__)
    try:
        v1.main()
    finally:
        baseline.main = original_main


if __name__ == "__main__":
    main()
