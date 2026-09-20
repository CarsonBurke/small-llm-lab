"""Unicode-boundary, model-native binary channel for NanoGPT Mini.

The only vocabulary-sized tensor is an INPUT lookup of learned bit logits.
Opaque IDs have no Unicode structure. No network outputs vocabulary logits:
all losses are binary. A causal compressor emits K latent bits per character;
Mini predicts only those bits. A causal local decoder reconstructs character
code bits, then a small decoder predicts opaque identity bits.

A codebook need not be injective during learning. XOR corrections on identity
bits make the internal representation exactly reversible anyway. The modeled
rate pays for BOTH latent and identity-residual information. Reconstruction
auxiliaries train the codebook directly but are excluded from reported rate.
The two-part rate is a coding bound, not the exact marginal character NLL;
physical packets currently bit-pack rather than entropy-code these streams.
"""

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pretraining.nanogpt_mini.bit_density import (
    LatentPrior,
    binary_nll,
    binary_st,
    bit_entropy,
)
from pretraining.nanogpt_mini.nanogpt_mini_model import Linear, RMSNorm


@dataclass(frozen=True)
class NativeBitsConfig:
    vocab_size: int
    code_bits: int = 32
    latent_bits: int = 8
    codec_dim: int = 64
    num_layers: int = 6
    model_dim: int = 512
    learn_codes: bool = True
    mixture_components: int = 1

    def __post_init__(self):
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if not max(1, (self.vocab_size - 1).bit_length()) <= self.code_bits <= 256:
            raise ValueError("code_bits must cover the alphabet and be at most 256")
        if not 1 <= self.latent_bits < self.code_bits:
            raise ValueError("latent_bits must be positive and smaller than code_bits")
        if self.codec_dim < 1 or self.num_layers < 1:
            raise ValueError("codec_dim and num_layers must be positive")
        if self.model_dim < 128 or self.model_dim % 128:
            raise ValueError("Mini model_dim must be a positive multiple of 128")
        if self.mixture_components < 1:
            raise ValueError("mixture_components must be positive")


class LocalBitNetwork(nn.Module):
    """15-character causal receptive field, no byte/code-point features."""

    def __init__(self, in_bits: int, out_bits: int, width: int):
        super().__init__()
        self.input = Linear(in_bits, width)
        self.layers = nn.ModuleList(
            nn.Conv1d(width, width, 3, dilation=2**i) for i in range(3)
        )
        self.norm = RMSNorm(width)
        self.output = Linear(width, out_bits)

    def forward(self, bits):
        h = self.input((2 * bits - 1).to(torch.bfloat16))
        for index, layer in enumerate(self.layers):
            conv = cast(nn.Conv1d, layer)
            dilation = 2**index
            x = self.norm(h).transpose(1, 2)
            bias = conv.bias
            assert bias is not None
            x = F.conv1d(
                F.pad(x, (2 * dilation, 0)),
                conv.weight.to(x.dtype),
                bias.to(x.dtype),
                dilation=dilation,
            )
            h = h + F.silu(x.transpose(1, 2))
        return self.output(self.norm(h)).float()


class IdentityDecoder(nn.Module):
    """Direct binary reconstruction of an opaque identity, O(code_bits*log V)."""

    def __init__(self, code_bits, identity_bits, width):
        super().__init__()
        self.hidden = Linear(code_bits, width)
        self.output = Linear(width, identity_bits)

    def forward(self, codes):
        h = self.hidden((2 * codes - 1).to(torch.bfloat16)).relu().square()
        return self.output(h).float()


class NativeBitsGPT(nn.Module):
    identity_shifts: Tensor
    sum_stat_names = ("rate_nats", "latent_nats", "residual_nats", "codec_nats")
    mean_stat_names = (
        "code_reconstruction_accuracy",
        "identity_reconstruction_accuracy",
        "latent_bit_mean",
        "latent_bit_entropy",
        "direct_identity_reconstruction_accuracy",
    )

    def __init__(self, config: NativeBitsConfig):
        super().__init__()
        self.config = config
        self.id_bits = max(1, (config.vocab_size - 1).bit_length())
        self.table = nn.Embedding(config.vocab_size, config.code_bits)
        self.table.weight.requires_grad_(config.learn_codes)
        self.compressor = LocalBitNetwork(
            config.code_bits, config.latent_bits, config.codec_dim
        )
        self.prior = LatentPrior(config)
        self.decoder = LocalBitNetwork(
            config.latent_bits, config.code_bits, config.codec_dim
        )
        self.identity_decoder = IdentityDecoder(
            config.code_bits, self.id_bits, config.codec_dim
        )
        self.register_buffer("identity_shifts", torch.arange(self.id_bits - 1, -1, -1))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        for name, p in self.named_parameters():
            if name == "table.weight":
                p.normal_(std=0.1)
            elif name.endswith("weight"):
                if ".proj." in name:
                    p.zero_()
                else:
                    fan_in = p.shape[1] * (p.shape[2] if p.ndim == 3 else 1)
                    p.normal_(std=math.sqrt(0.33 / fan_in))
            elif name.endswith("bias"):
                p.zero_()
            elif name.endswith("gains"):
                p.fill_(1)
            else:
                raise ValueError(f"uninitialized parameter: {name}")
        self.prior.reset_mixture_head()

    @property
    def transport_widths(self) -> tuple[int, int]:
        return self.config.latent_bits, self.id_bits

    def compile_components(self):
        for module in (
            self.compressor,
            self.prior,
            self.decoder,
            self.identity_decoder,
        ):
            module.compile(dynamic=True, fullgraph=True)

    def character_codes(self, ids: Tensor) -> Tensor:
        # Gather BEFORE binarizing: work is proportional to observed characters,
        # never to the number of possible output identities.
        return binary_st(self.table(ids.long()))

    def identity_bits(self, ids: Tensor) -> Tensor:
        return ((ids.long().unsqueeze(-1) >> self.identity_shifts) & 1).float()

    def bits_to_ids(self, bits: Tensor) -> Tensor:
        return (bits.long() << self.identity_shifts).sum(-1)

    def encode_latents(self, ids: Tensor) -> Tensor:
        return binary_st(self.compressor(self.character_codes(ids)))

    def residual_logits(self, latents: Tensor) -> Tensor:
        reconstructed_codes = binary_st(self.decoder(latents))
        return self.identity_decoder(reconstructed_codes)

    def forward(self, ids: Tensor, codec_weight: float = 1.0):
        codes = self.character_codes(ids)
        latents = binary_st(self.compressor(codes))
        latent_logits = self.prior(latents)
        decoded_logits = self.decoder(latents)
        decoded_codes = binary_st(decoded_logits)
        source_logits = self.identity_decoder(decoded_codes)
        source_bits = self.identity_bits(ids)
        # The encoder must not lower prediction loss by changing the answer
        # toward the prior's current guess. Reconstruction and prior-context
        # paths still train the encoder; only the current label is detached.
        latent_nats = binary_nll(
            latent_logits, latents.detach(), self.config.mixture_components
        ).sum()
        # For r = source XOR prediction, P(r=1) is P(source=1) or its
        # complement according to prediction. This BCE is exactly its NLL.
        residual_nats = F.binary_cross_entropy_with_logits(
            source_logits, source_bits, reduction="sum"
        )
        code_aux = F.binary_cross_entropy_with_logits(
            decoded_logits, codes, reduction="none"
        ).mean(-1)
        direct_identity_logits = self.identity_decoder(codes)
        identity_aux = F.binary_cross_entropy_with_logits(
            direct_identity_logits, source_bits, reduction="none"
        ).mean(-1)
        codec_nats = (code_aux + identity_aux).sum()
        rate_nats = latent_nats + residual_nats
        total = rate_nats + codec_weight * codec_nats
        stats = {
            "rate_nats": rate_nats,
            "latent_nats": latent_nats,
            "residual_nats": residual_nats,
            "codec_nats": codec_nats,
            "code_reconstruction_accuracy": ((decoded_codes > 0.5) == (codes > 0.5))
            .all(-1)
            .float()
            .mean(),
            "identity_reconstruction_accuracy": (
                (source_logits > 0) == source_bits.bool()
            )
            .all(-1)
            .float()
            .mean(),
            "latent_bit_mean": latents.mean(),
            "latent_bit_entropy": bit_entropy(latents),
            "direct_identity_reconstruction_accuracy": (
                (direct_identity_logits > 0) == source_bits.bool()
            )
            .all(-1)
            .float()
            .mean(),
        }
        return total, {name: value.detach() for name, value in stats.items()}

    @torch.no_grad()
    def export(self, ids: Tensor) -> dict[str, list[list[int]]]:
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError("export requires one nonempty row")
        if bool(((ids < 0) | (ids >= self.config.vocab_size)).any()):
            raise ValueError("source identity is outside the checkpoint alphabet")
        latents = self.encode_latents(ids)
        prediction = self.residual_logits(latents) > 0
        residual = self.identity_bits(ids).bool() ^ prediction
        return {
            "latents": latents[0].to(torch.int64).tolist(),
            "residual": residual[0].to(torch.int64).tolist(),
        }

    @torch.no_grad()
    def recover(self, latents: list[list[int]], residual: list[list[int]]) -> list[int]:
        if len(latents) != len(residual):
            raise ValueError("latent and residual stream lengths differ")
        if not latents:
            return []
        for stream, width in (
            (latents, self.config.latent_bits),
            (residual, self.id_bits),
        ):
            if any(
                len(row) != width
                or any(type(v) not in (bool, int) or v not in (0, 1) for v in row)
                for row in stream
            ):
                raise ValueError("invalid binary stream or width")
        device = self.table.weight.device
        z = torch.tensor(latents, dtype=torch.float32, device=device).unsqueeze(0)
        r = torch.tensor(residual, dtype=torch.bool, device=device).unsqueeze(0)
        prediction = self.residual_logits(z) > 0
        ids = self.bits_to_ids(prediction ^ r)
        if bool((ids >= self.config.vocab_size).any()):
            raise ValueError("decoded identity is outside the checkpoint alphabet")
        return ids[0].tolist()
