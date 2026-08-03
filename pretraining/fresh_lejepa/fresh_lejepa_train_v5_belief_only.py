"""V5 controlled ablation: remove only the token branch from both probes."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from pathlib import Path

from torch import Tensor, nn

from pretraining.fresh_lejepa import fresh_lejepa_train as v1
from pretraining.fresh_lejepa.fresh_lejepa_train_v4 import (
    SIGREG_POSITION_CHUNK,
    SIGREG_PROJECTION_CHUNK,
    _install_configurable_accumulation,
)
from pretraining.fresh_lejepa.fresh_lejepa_train_v5_swiglu import FreshLeJEPAV5SwiGLU
from pretraining.fresh_lejepa.fresh_lejepa_train_v6_belief_dropout import (
    BeliefOnlyCriticProbe,
    BeliefOnlyPolicyProbe,
)
import train_gpt as baseline


ARCHITECTURE = "fresh_lejepa_v5_belief_only_no_dropout"


class FreshLeJEPAV5BeliefOnly(FreshLeJEPAV5SwiGLU):
    def make_policy_probe(self, model_dim: int, vocab_size: int) -> nn.Module:
        return BeliefOnlyPolicyProbe(model_dim, vocab_size)

    def make_critic_probe(self, model_dim: int) -> nn.Module:
        return BeliefOnlyCriticProbe(model_dim)

    def probe_features(self, token_latent: Tensor, predicted: Tensor) -> Tensor:
        return predicted.detach()


def main() -> None:
    original_main = _install_configurable_accumulation(default_steps=8)
    FreshLeJEPAV5BeliefOnly.return_loss_components = True
    v1.FreshHyperparameters.sigreg_proj_chunk = SIGREG_PROJECTION_CHUNK
    v1.FreshHyperparameters.sigreg_position_chunk = SIGREG_POSITION_CHUNK
    v1.FreshLeJEPAGPT = FreshLeJEPAV5BeliefOnly
    v1.EXPERIMENT_ARCHITECTURE = ARCHITECTURE
    v1.EXPERIMENT_SOURCE = Path(__file__)
    try:
        v1.main()
    finally:
        baseline.main = original_main


if __name__ == "__main__":
    main()
