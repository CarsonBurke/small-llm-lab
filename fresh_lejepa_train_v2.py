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

    def __init__(self, model_dim: int, dropout: float = 0.0):
        super().__init__()
        self.token = CastedLinear(model_dim, model_dim)
        self.predicted = CastedLinear(model_dim, model_dim)
        self.hidden1 = CastedLinear(model_dim, model_dim)
        self.hidden2 = CastedLinear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: Tensor) -> Tensor:
        token, predicted = features.chunk(2, dim=-1)
        hidden = F.rms_norm(self.token(token) + self.predicted(predicted), (token.size(-1),))
        hidden = hidden + self.dropout(F.silu(self.hidden1(hidden)))
        hidden = hidden + self.dropout(F.silu(self.hidden2(F.rms_norm(hidden, (hidden.size(-1),)))))
        return F.rms_norm(hidden, (hidden.size(-1),))


class AdditivePolicyProbe(AdditiveFusion):
    """Produces a calibrated codebook-space latent."""

    def __init__(
        self, model_dim: int, vocab_size: int, embedding_init_std: float,
        dropout: float = 0.0,
    ):
        super().__init__(model_dim, dropout=dropout)
        initial_scale = math.sqrt(model_dim) * embedding_init_std
        self.logit_scale = nn.Parameter(torch.tensor(initial_scale, dtype=torch.float32))
        self.vocab_bias = nn.Parameter(torch.zeros(vocab_size, dtype=torch.float32))


class AdditiveCriticProbe(nn.Module):
    def __init__(self, model_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fusion = AdditiveFusion(model_dim, dropout=dropout)
        self.output = CastedLinear(model_dim, 1, bias=False)
        self.output._zero_init = True
        nn.init.zeros_(self.output.weight)

    def forward(self, features: Tensor) -> Tensor:
        return self.output(self.fusion(features))


class FreshLeJEPAGPTV2(v1.FreshLeJEPAGPT):
    probe_dropout = 0.0

    def make_policy_probe(self, model_dim: int, vocab_size: int) -> nn.Module:
        return AdditivePolicyProbe(
            model_dim,
            vocab_size,
            self.tied_embed_init_std,
            dropout=self.probe_dropout,
        )

    def make_critic_probe(self, model_dim: int) -> nn.Module:
        return AdditiveCriticProbe(model_dim, dropout=self.probe_dropout)

    def logits_from_features(self, features: Tensor) -> Tensor:
        hidden = self.policy_probe(features)
        codebook = self.policy_codebook()
        raw = F.linear(hidden, codebook) / math.sqrt(self.tok_emb.embedding_dim)
        raw = raw * self.policy_probe.logit_scale.to(raw.dtype)
        raw = raw + self.policy_probe.vocab_bias.to(raw.dtype)
        return self.logit_softcap * torch.tanh(raw / self.logit_softcap)

    def policy_codebook(self) -> Tensor:
        return F.rms_norm(self.tok_emb.weight.detach(), (self.tok_emb.embedding_dim,))


def main() -> None:
    v1.FreshLeJEPAGPT = FreshLeJEPAGPTV2
    v1.EXPERIMENT_ARCHITECTURE = ARCHITECTURE
    v1.EXPERIMENT_SOURCE = Path(__file__)
    v1.main()


if __name__ == "__main__":
    main()
