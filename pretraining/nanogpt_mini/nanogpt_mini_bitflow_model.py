"""Exactly bijective opaque-ID coding with a causal Mini binary-vector prior.

Alternating XOR couplings permute the entire code-bit domain. Their algebraic
inverse needs neither a reconstruction network nor a residual stream. The
likelihood charges all binary-domain mass, including unused identity patterns:
its rate is a code-space coding bound, not a vocabulary-renormalized NLL.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pretraining.nanogpt_mini.bit_density import (
    LatentPrior,
    binary_nll,
    binary_st,
    bit_entropy,
)
from pretraining.nanogpt_mini.nanogpt_mini_model import Linear


@dataclass(frozen=True)
class BitFlowConfig:
    vocab_size: int
    code_bits: int = 0
    codec_dim: int = 64
    flow_layers: int = 4
    num_layers: int = 6
    model_dim: int = 512
    mixture_components: int = 8
    learn_codec: bool = True
    mask_surrogate: str = "sigmoid"
    density_head: str = "mixture"
    prefix_width: int = 128

    def __post_init__(self):
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.code_bits == 0:
            object.__setattr__(
                self, "code_bits", max(2, (self.vocab_size - 1).bit_length())
            )
        if not 2 <= self.code_bits <= 32 or self.vocab_size > 1 << self.code_bits:
            raise ValueError(
                "code_bits must cover the alphabet and be between 2 and 32"
            )
        if self.codec_dim < 1 or self.flow_layers < 1 or self.num_layers < 1:
            raise ValueError("codec_dim, flow_layers and num_layers must be positive")
        if self.model_dim < 128 or self.model_dim % 128:
            raise ValueError("Mini model_dim must be a positive multiple of 128")
        if type(self.mixture_components) is not int or self.mixture_components < 1:
            raise ValueError("mixture_components must be a positive integer")
        if self.mask_surrogate not in {"sigmoid", "identity"}:
            raise ValueError("mask_surrogate must be sigmoid or identity")
        if self.density_head not in {"mixture", "prefix"}:
            raise ValueError("density_head must be mixture or prefix")
        if type(self.prefix_width) is not int or self.prefix_width < 1:
            raise ValueError("prefix_width must be a positive integer")
        if self.density_head == "prefix" and (
            self.mixture_components != 1 or self.learn_codec
        ):
            raise ValueError(
                "prefix density requires mixture_components=1 and a fixed codec"
            )

    @property
    def latent_bits(self) -> int:
        return self.code_bits


class XORCoupling(nn.Module):
    """A self-inverse binary coupling; the mask sees only the unchanged half."""

    def __init__(
        self, bits: int, width: int, update_left: bool, mask_surrogate: str = "sigmoid"
    ):
        super().__init__()
        self.split = bits // 2
        self.update_left = update_left
        self.mask_surrogate = mask_surrogate
        updated = self.split if update_left else bits - self.split
        conditioning = bits - updated
        self.hidden = Linear(conditioning, width)
        self.output = Linear(width, updated)

    def forward(self, bits: Tensor) -> Tensor:
        left, right = bits[..., : self.split], bits[..., self.split :]
        unchanged, updated = (right, left) if self.update_left else (left, right)
        hidden = self.hidden((2 * unchanged - 1).to(torch.bfloat16)).relu().square()
        logits = self.output(hidden).float()
        # Surrogates change backward credit only; both preserve exact hard bits.
        if self.mask_surrogate == "identity":
            mask = (logits > 0).to(logits.dtype) + (logits - logits.detach())
        else:
            mask = binary_st(logits)
        updated = updated + mask - 2 * updated * mask
        return (
            torch.cat((updated, unchanged), dim=-1)
            if self.update_left
            else torch.cat((unchanged, updated), dim=-1)
        )


class BitFlowCodec(nn.Module):
    def __init__(self, config: BitFlowConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            XORCoupling(
                config.code_bits,
                config.codec_dim,
                update_left=bool(index % 2),
                mask_surrogate=config.mask_surrogate,
            )
            for index in range(config.flow_layers)
        )

    def forward(self, bits: Tensor) -> Tensor:
        for layer in self.layers:
            bits = layer(bits)
        return bits

    def inverse(self, bits: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            bits = layer(bits)
        return bits

    def compile_components(self):
        self.compile(dynamic=True, fullgraph=True)
        self.inverse = torch.compile(self.inverse, dynamic=True, fullgraph=True)


class BitFlowGPT(nn.Module):
    identity_shifts: Tensor
    sum_stat_names = ("rate_nats", "latent_nats")
    mean_stat_names = ("latent_bit_mean", "latent_bit_entropy")

    def __init__(self, config: BitFlowConfig):
        super().__init__()
        self.config = config
        self.codec = BitFlowCodec(config)
        self.prior = LatentPrior(config)
        self.register_buffer(
            "identity_shifts", torch.arange(config.code_bits - 1, -1, -1)
        )
        self.reset_parameters()
        if config.mask_surrogate == "identity":
            # Finite identity-STE gradients can exceed sqrt(FP32_MAX).
            # Keep the tiny codec's master parameters and Adam moments in FP64;
            # Linear still casts to BF16 for every neural operation.
            self.codec.double()
        # Freeze only after identical construction and initialization, so fixed
        # and learned controls have exactly the same initial codec AND prior.
        self.codec.requires_grad_(config.learn_codec)

    @property
    def transport_widths(self) -> tuple[int, int]:
        return self.config.code_bits, 0

    @torch.no_grad()
    def reset_parameters(self):
        for name, parameter in self.named_parameters():
            if name.endswith("weight"):
                if (
                    name in {"prior.proj.weight", "prior.proj.output.weight"}
                    or (".proj." in name and not name.startswith("prior.proj."))
                    or (name.startswith("codec.") and ".output." in name)
                ):
                    parameter.zero_()
                else:
                    parameter.normal_(std=math.sqrt(0.33 / parameter.shape[1]))
            elif name.endswith("bias"):
                parameter.zero_()
            elif name.endswith("gains"):
                parameter.fill_(1)
            else:
                raise ValueError(f"uninitialized parameter: {name}")
        self.prior.reset_mixture_head()

    def compile_components(self):
        self.codec.compile_components()
        self.prior.compile(dynamic=True, fullgraph=True)

    def identity_bits(self, ids: Tensor) -> Tensor:
        return ((ids.long().unsqueeze(-1) >> self.identity_shifts) & 1).float()

    def bits_to_ids(self, bits: Tensor) -> Tensor:
        return (bits.long() << self.identity_shifts).sum(-1)

    def encode_latents(self, ids: Tensor) -> Tensor:
        return self.codec(self.identity_bits(ids))

    def forward(self, ids: Tensor, codec_weight: float = 1.0):
        codes = self.encode_latents(ids)
        # Both target and context credit reach the encoder. There is no
        # reconstruction loss, balance term, residual, or codec-weight schedule.
        latent_nats = binary_nll(
            self.prior(codes), codes, self.config.mixture_components
        ).sum()
        detached_codes = codes.detach()
        rate = latent_nats.detach()
        stats = {
            "rate_nats": rate,
            "latent_nats": rate,
            "latent_bit_mean": detached_codes.mean(),
            "latent_bit_entropy": bit_entropy(detached_codes),
        }
        return latent_nats, stats

    @torch.no_grad()
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
        if bool(((ids < 0) | (ids >= self.config.vocab_size)).any()):
            raise ValueError("source identity is outside the checkpoint alphabet")
        if ids.shape[1] == 0:
            return {"latents": [], "residual": []}
        codes = self.encode_latents(ids)
        return {
            "latents": codes[0].to(torch.int64).tolist(),
            "residual": [[] for _ in range(ids.shape[1])],
        }

    @torch.no_grad()
    def recover(self, latents: list[list[int]], residual: list[list[int]]) -> list[int]:
        if len(latents) != len(residual):
            raise ValueError("latent and residual stream lengths differ")
        if any(len(row) != 0 for row in residual):
            raise ValueError("BitFlow has no residual bits")
        if any(
            len(row) != self.config.code_bits
            or any(
                type(value) not in (bool, int) or value not in (0, 1) for value in row
            )
            for row in latents
        ):
            raise ValueError("invalid binary stream or width")
        if not latents:
            return []
        codes = torch.tensor(
            latents, dtype=torch.float32, device=self.identity_shifts.device
        ).unsqueeze(0)
        ids = self.bits_to_ids(self.codec.inverse(codes))
        if bool((ids >= self.config.vocab_size).any()):
            raise ValueError(
                "decoded identity is outside the checkpoint alphabet (reserved ID)"
            )
        return ids[0].tolist()
