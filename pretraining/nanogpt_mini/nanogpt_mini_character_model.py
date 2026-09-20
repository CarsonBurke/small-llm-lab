"""Matched Mini character softmax with a zero-feature BOS and exact ID transport."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pretraining.nanogpt_mini.nanogpt_mini_model import GPT


@dataclass(frozen=True)
class CharacterConfig:
    vocab_size: int
    num_layers: int = 6
    model_dim: int = 512

    def __post_init__(self):
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.model_dim < 128 or self.model_dim % 128:
            raise ValueError("Mini model_dim must be a positive multiple of 128")


class CharacterGPT(nn.Module):
    sum_stat_names = ("rate_nats",)
    mean_stat_names = ("symbol_accuracy",)
    # Preserve the original Mini embedding optimizer convention, not CODE_LR.
    embedding_lr = 0.7

    def __init__(self, config: CharacterConfig):
        super().__init__()
        self.config = config
        self.prior = GPT(config.vocab_size, config.num_layers, config.model_dim)
        self.reset_parameters()

    @property
    def table(self) -> nn.Embedding:
        return self.prior.embed

    @property
    def transport_widths(self) -> tuple[int, int]:
        return max(1, (self.config.vocab_size - 1).bit_length()), 0

    @torch.no_grad()
    def reset_parameters(self):
        for name, parameter in self.prior.named_parameters():
            if name.endswith("weight"):
                if "proj" in name:
                    parameter.zero_()
                elif "embed" in name:
                    parameter.normal_()
                else:
                    parameter.normal_(
                        std=math.sqrt(0.33) / math.sqrt(parameter.size(-1))
                    )
            elif name.endswith("bias"):
                parameter.zero_()
            elif name.endswith("gains"):
                parameter.normal_(mean=1, std=0)
            else:
                raise ValueError(f"uninitialized parameter: {name}")

    def compile_components(self):
        self.logits = torch.compile(self.logits, dynamic=True, fullgraph=True)

    def logits(self, ids: Tensor) -> Tensor:
        # Shift features, not labels: position zero has no learned BOS token,
        # and the last input character is still a scored target.
        features = self.prior.embed(ids[:, :-1].long())
        bos = features.new_zeros((ids.shape[0], 1, self.config.model_dim))
        x = self.prior.norm1(torch.cat((bos, features), dim=1))
        for block in self.prior.blocks:
            x = block(x)
        logits = self.prior.proj(self.prior.norm2(x)).float()
        return 15 * logits * (logits.square() + 15**2).rsqrt()

    def forward(self, ids: Tensor, codec_weight: float = 1.0):
        logits = self.logits(ids)
        rate_nats = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            ids.long().reshape(-1),
            reduction="sum",
        )
        stats = {
            "rate_nats": rate_nats.detach(),
            "symbol_accuracy": (logits.argmax(-1) == ids).float().mean().detach(),
        }
        return rate_nats, stats

    def export(self, ids: Tensor) -> dict[str, list[list[int]]]:
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("export requires one row")
        if ids.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("source identities must be integers")
        # Transport is deliberately independent of embeddings and model weights.
        originals = ids[0].tolist()
        if any(value < 0 or value >= self.config.vocab_size for value in originals):
            raise ValueError("source identity is outside the checkpoint alphabet")
        width, _ = self.transport_widths
        return {
            "latents": [
                [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]
                for value in originals
            ],
            "residual": [[] for _ in originals],
        }

    def recover(self, latents: list[list[int]], residual: list[list[int]]) -> list[int]:
        if len(latents) != len(residual):
            raise ValueError("latent and residual stream lengths differ")
        width, _ = self.transport_widths
        originals = []
        for row, correction in zip(latents, residual):
            if len(correction) != 0:
                raise ValueError("character transport has no residual bits")
            if len(row) != width or any(
                type(bit) not in (bool, int) or bit not in (0, 1) for bit in row
            ):
                raise ValueError("invalid binary stream or width")
            value = 0
            for bit in row:
                value = (value << 1) | bit
            if value >= self.config.vocab_size:
                raise ValueError("decoded identity is outside the checkpoint alphabet")
            originals.append(value)
        return originals
