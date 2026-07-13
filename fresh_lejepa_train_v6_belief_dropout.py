"""V6: belief-only pre-norm SwiGLU probes plus LeWM predictor dropout."""

from __future__ import annotations

from pathlib import Path

import torch.nn.functional as F
from torch import Tensor, nn

import fresh_lejepa_train as v1
from fresh_lejepa_train_v4 import (
    SIGREG_POSITION_CHUNK,
    SIGREG_PROJECTION_CHUNK,
    _install_configurable_accumulation,
)
from fresh_lejepa_train_v4_predictor_dropout import PredictorDropoutMixin
from fresh_lejepa_train_v5_swiglu import FreshLeJEPAV5SwiGLU, PreNormSwiGLUBlock
import train_gpt as baseline
from train_gpt import CastedLinear


ARCHITECTURE = "fresh_lejepa_v6_belief_only_prenorm_swiglu_predictor_dropout01"


class BeliefOnlySwiGLUTrunk(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.belief = CastedLinear(model_dim, model_dim)
        self.block1 = PreNormSwiGLUBlock(model_dim)
        self.block2 = PreNormSwiGLUBlock(model_dim)

    def forward(self, belief: Tensor) -> Tensor:
        latent = self.belief(belief)
        latent = self.block1(latent)
        latent = self.block2(latent)
        return F.rms_norm(latent, (latent.size(-1),))


class BeliefOnlyPolicyProbe(BeliefOnlySwiGLUTrunk):
    def __init__(self, model_dim: int, vocab_size: int):
        super().__init__(model_dim)
        self.output = CastedLinear(model_dim, vocab_size, bias=False)
        self.output._zero_init = True
        nn.init.zeros_(self.output.weight)

    def forward(self, belief: Tensor) -> Tensor:
        return self.output(super().forward(belief))


class BeliefOnlyCriticProbe(BeliefOnlySwiGLUTrunk):
    def __init__(self, model_dim: int):
        super().__init__(model_dim)
        self.output = CastedLinear(model_dim, 1, bias=False)
        self.output._zero_init = True
        nn.init.zeros_(self.output.weight)

    def forward(self, belief: Tensor) -> Tensor:
        return self.output(super().forward(belief))


class FreshLeJEPAV6BeliefDropout(PredictorDropoutMixin, FreshLeJEPAV5SwiGLU):
    def make_policy_probe(self, model_dim: int, vocab_size: int) -> nn.Module:
        return BeliefOnlyPolicyProbe(model_dim, vocab_size)

    def make_critic_probe(self, model_dim: int) -> nn.Module:
        return BeliefOnlyCriticProbe(model_dim)

    def probe_features(self, token_latent: Tensor, predicted: Tensor) -> Tensor:
        return predicted.detach()


def main() -> None:
    original_main = _install_configurable_accumulation(default_steps=8)
    FreshLeJEPAV6BeliefDropout.return_loss_components = True
    v1.FreshHyperparameters.sigreg_proj_chunk = SIGREG_PROJECTION_CHUNK
    v1.FreshHyperparameters.sigreg_position_chunk = SIGREG_POSITION_CHUNK
    v1.FreshLeJEPAGPT = FreshLeJEPAV6BeliefDropout
    v1.EXPERIMENT_ARCHITECTURE = ARCHITECTURE
    v1.EXPERIMENT_SOURCE = Path(__file__)
    try:
        v1.main()
    finally:
        baseline.main = original_main


if __name__ == "__main__":
    main()
