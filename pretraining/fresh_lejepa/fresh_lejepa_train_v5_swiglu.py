"""V5: V4 world model with a D-wide pre-norm SwiGLU policy and critic."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pretraining.fresh_lejepa import fresh_lejepa_train as v1
from pretraining.fresh_lejepa.fresh_lejepa_train_v4 import (
    FreshLeJEPAV4,
    SIGREG_POSITION_CHUNK,
    SIGREG_PROJECTION_CHUNK,
    _install_configurable_accumulation,
)
import train_gpt as baseline
from train_gpt import CastedLinear


ARCHITECTURE = "fresh_lejepa_v5_shared_rms_paired_b128_prenorm_swiglu_probes"


class PreNormSwiGLUBlock(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.gate = CastedLinear(model_dim, model_dim)
        self.value = CastedLinear(model_dim, model_dim)
        self.output = CastedLinear(model_dim, model_dim)

    def forward(self, latent: Tensor) -> Tensor:
        normalized = F.rms_norm(latent, (latent.size(-1),))
        update = self.output(F.silu(self.gate(normalized)) * self.value(normalized))
        return latent + update


class PreNormSwiGLUFusion(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.token = CastedLinear(model_dim, model_dim)
        self.predicted = CastedLinear(model_dim, model_dim)
        self.block1 = PreNormSwiGLUBlock(model_dim)
        self.block2 = PreNormSwiGLUBlock(model_dim)

    def forward(self, features: Tensor) -> Tensor:
        token, predicted = features.chunk(2, dim=-1)
        latent = self.token(token) + self.predicted(predicted)
        latent = self.block1(latent)
        latent = self.block2(latent)
        return F.rms_norm(latent, (latent.size(-1),))


class SwiGLUPolicyProbe(PreNormSwiGLUFusion):
    def __init__(self, model_dim: int, vocab_size: int):
        super().__init__(model_dim)
        self.output = CastedLinear(model_dim, vocab_size, bias=False)
        self.output._zero_init = True
        nn.init.zeros_(self.output.weight)

    def forward(self, features: Tensor) -> Tensor:
        return self.output(super().forward(features))


class SwiGLUCriticProbe(PreNormSwiGLUFusion):
    def __init__(self, model_dim: int):
        super().__init__(model_dim)
        self.output = CastedLinear(model_dim, 1, bias=False)
        self.output._zero_init = True
        nn.init.zeros_(self.output.weight)

    def forward(self, features: Tensor) -> Tensor:
        return self.output(super().forward(features))


class FreshLeJEPAV5SwiGLU(FreshLeJEPAV4):
    def make_policy_probe(self, model_dim: int, vocab_size: int) -> nn.Module:
        return SwiGLUPolicyProbe(model_dim, vocab_size)

    def make_critic_probe(self, model_dim: int) -> nn.Module:
        return SwiGLUCriticProbe(model_dim)


def main() -> None:
    original_main = _install_configurable_accumulation(default_steps=8)
    FreshLeJEPAV5SwiGLU.return_loss_components = True
    v1.FreshHyperparameters.sigreg_proj_chunk = SIGREG_PROJECTION_CHUNK
    v1.FreshHyperparameters.sigreg_position_chunk = SIGREG_POSITION_CHUNK
    v1.FreshLeJEPAGPT = FreshLeJEPAV5SwiGLU
    v1.EXPERIMENT_ARCHITECTURE = ARCHITECTURE
    v1.EXPERIMENT_SOURCE = Path(__file__)
    try:
        v1.main()
    finally:
        baseline.main = original_main


if __name__ == "__main__":
    main()
