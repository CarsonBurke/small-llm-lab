"""V8: V7 plus a shared intra-token pre-norm SwiGLU encoder."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import fresh_lejepa_train as v1
from fresh_lejepa_train_v4 import (
    SIGREG_POSITION_CHUNK,
    SIGREG_PROJECTION_CHUNK,
    _install_configurable_accumulation,
)
from fresh_lejepa_train_v7_preproj_belief import FreshLeJEPAV7PreProjBelief
import train_gpt as baseline
from train_gpt import CastedLinear


ARCHITECTURE = "fresh_lejepa_v8_shared_token_prenorm_swiglu_encoder"


class ZeroInitPreNormSwiGLUBlock(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.gate = CastedLinear(model_dim, model_dim)
        self.value = CastedLinear(model_dim, model_dim)
        self.output = CastedLinear(model_dim, model_dim)
        self.output._zero_init = True
        nn.init.zeros_(self.output.weight)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(self, latent: Tensor) -> Tensor:
        normalized = F.rms_norm(latent, (latent.size(-1),))
        update = self.output(F.silu(self.gate(normalized)) * self.value(normalized))
        return latent + update


class SharedTokenEncoder(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.block1 = ZeroInitPreNormSwiGLUBlock(model_dim)
        self.block2 = ZeroInitPreNormSwiGLUBlock(model_dim)

    def forward(self, token_embedding: Tensor) -> Tensor:
        latent = self.block1(token_embedding)
        latent = self.block2(latent)
        return F.rms_norm(latent, (latent.size(-1),))


class FreshLeJEPAV8TokenSwiGLU(FreshLeJEPAV7PreProjBelief):
    def __init__(self, *args, **kwargs):
        model_dim = kwargs.get("model_dim", args[2] if len(args) > 2 else None)
        super().__init__(*args, **kwargs)
        if model_dim is None:
            raise ValueError("model_dim is required")
        with torch.random.fork_rng(devices=[]):
            self.blocks[-1].token_encoder = SharedTokenEncoder(model_dim)

    @property
    def token_encoder(self) -> SharedTokenEncoder:
        return self.blocks[-1].token_encoder

    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        folded = getattr(self, "_folded_token_latents", None)
        if folded is not None:
            return F.embedding(input_ids, folded)
        encoded = self.token_encoder(self.tok_emb(input_ids))
        return self.latent_projector(encoded)

    @torch.no_grad()
    def fold_input_projector_for_inference(self) -> None:
        encoded = self.token_encoder(self.tok_emb.weight)
        table = self.latent_projector.inference(encoded).detach()
        object.__setattr__(self, "_folded_token_latents", table)


def main() -> None:
    original_main = _install_configurable_accumulation(default_steps=8)
    FreshLeJEPAV8TokenSwiGLU.return_loss_components = True
    v1.FreshHyperparameters.sigreg_proj_chunk = SIGREG_PROJECTION_CHUNK
    v1.FreshHyperparameters.sigreg_position_chunk = SIGREG_POSITION_CHUNK
    v1.FreshLeJEPAGPT = FreshLeJEPAV8TokenSwiGLU
    v1.EXPERIMENT_ARCHITECTURE = ARCHITECTURE
    v1.EXPERIMENT_SOURCE = Path(__file__)
    try:
        v1.main()
    finally:
        baseline.main = original_main


if __name__ == "__main__":
    main()
