"""Fresh LeJEPA V2 with compact additive-fusion policy and critic probes."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import fresh_lejepa_train as v1
from train_gpt import CastedLinear


ARCHITECTURE = "fresh_lejepa_attached_target_additive_codebook_probes_v2"


class AdditiveFusion(nn.Module):
    """Four D->D matrices fuse both latents without a width bottleneck."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.token = CastedLinear(model_dim, model_dim)
        self.predicted = CastedLinear(model_dim, model_dim)
        self.hidden1 = CastedLinear(model_dim, model_dim)
        self.hidden2 = CastedLinear(model_dim, model_dim)

    def forward(self, features: Tensor) -> Tensor:
        token, predicted = features.chunk(2, dim=-1)
        hidden = F.rms_norm(self.token(token) + self.predicted(predicted), (token.size(-1),))
        hidden = hidden + F.silu(self.hidden1(hidden))
        hidden = hidden + F.silu(self.hidden2(F.rms_norm(hidden, (hidden.size(-1),))))
        return F.rms_norm(hidden, (hidden.size(-1),))


class AdditivePolicyProbe(AdditiveFusion):
    """Produces a codebook-space latent; vocabulary decoding lives on the model."""


class AdditiveCriticProbe(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.fusion = AdditiveFusion(model_dim)
        self.output = CastedLinear(model_dim, 1, bias=False)
        self.output._zero_init = True

    def forward(self, features: Tensor) -> Tensor:
        return self.output(self.fusion(features))


class FreshLeJEPAGPTV2(v1.FreshLeJEPAGPT):
    def __init__(self, *args, **kwargs):
        model_dim = kwargs.get("model_dim", args[2] if len(args) > 2 else None)
        super().__init__(*args, **kwargs)
        if model_dim is None:
            raise ValueError("model_dim is required")
        owner = self.blocks[-1]
        owner.policy_probe = AdditivePolicyProbe(model_dim)
        owner.critic_probe = AdditiveCriticProbe(model_dim)
        self._init_weights()

    def logits_from_features(self, features: Tensor) -> Tensor:
        hidden = self.policy_probe(features)
        codebook = F.rms_norm(self.tok_emb.weight.detach(), (self.tok_emb.embedding_dim,))
        raw = F.linear(hidden, codebook) / math.sqrt(self.tok_emb.embedding_dim)
        return self.logit_softcap * torch.tanh(raw / self.logit_softcap)


def main() -> None:
    v1.FreshLeJEPAGPT = FreshLeJEPAGPTV2
    v1.EXPERIMENT_ARCHITECTURE = ARCHITECTURE
    v1.EXPERIMENT_SOURCE = Path(__file__)
    v1.main()


if __name__ == "__main__":
    main()
