"""Model and objective contracts for the CE-anchored latent flow.

There is no variational machinery: the encoder is deterministic and every
consumer of a latent (decoder, prior history, evaluation posterior) sees it at
the single fixed noise level ``decode_time`` on the flow path. That noise level
is the only rate control; cross-entropy is the only anti-collapse term.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

ARCHITECTURE = "celf_byte_v1"
VOCAB_SIZE = 256


@dataclass(frozen=True)
class ModelConfig:
    patch_size: int = 4
    latent_dim: int = 32
    byte_dim: int = 128
    codec_dim: int = 512
    codec_layers: int = 4
    codec_heads: int = 8
    codec_ffn_dim: int = 1536
    prior_dim: int = 640
    prior_layers: int = 13
    prior_heads: int = 10
    prior_mlp_ratio: int = 4
    prior_rope_dim: int = 48
    block_patches: int = 8
    seq_bytes: int = 2560
    decode_time: float = 0.25
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6

    def __post_init__(self):
        for name in (
            "patch_size",
            "latent_dim",
            "byte_dim",
            "codec_dim",
            "codec_layers",
            "codec_heads",
            "codec_ffn_dim",
            "prior_dim",
            "prior_layers",
            "prior_heads",
            "prior_mlp_ratio",
            "prior_rope_dim",
            "block_patches",
            "seq_bytes",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.seq_bytes % (self.patch_size * self.block_patches):
            raise ValueError("seq_bytes must hold complete blocks of complete patches")
        if self.codec_dim % self.codec_heads or self.prior_dim % self.prior_heads:
            raise ValueError("attention width must be divisible by head count")
        if self.codec_dim // self.codec_heads % 2:
            raise ValueError("codec attention head width must be even")
        if self.prior_rope_dim % 2 or self.prior_rope_dim > self.prior_dim // self.prior_heads:
            raise ValueError("prior rotary width must be even and fit an attention head")
        if not 0.0 < self.decode_time < 1.0:
            raise ValueError("decode_time must lie strictly inside the flow path (0,1)")
        if self.rope_theta <= 1 or self.norm_eps <= 0:
            raise ValueError("invalid rotary base or normalization epsilon")

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    @property
    def seq_patches(self) -> int:
        return self.seq_bytes // self.patch_size

    @property
    def block_bytes(self) -> int:
        return self.patch_size * self.block_patches

    @property
    def blocks(self) -> int:
        return self.seq_patches // self.block_patches

    @property
    def signal_scale(self) -> float:
        """Latent signal amplitude at the anchoring noise level."""
        return 1.0 - self.decode_time

    @property
    def anchor_capacity_bits_per_patch(self) -> float:
        """Gaussian-channel capacity of one anchored latent, a rate ceiling."""
        snr = (self.signal_scale / self.decode_time) ** 2
        return 0.5 * self.latent_dim * math.log2(1.0 + snr)


ATTACHMENTS = ("all", "history")


@dataclass(frozen=True)
class LossConfig:
    mask_weight: float = 1.0
    flow_weight: float = 1.0
    mask_probability: float = 0.15
    time_loc: float = 0.0
    time_scale: float = 1.0
    attachment: str = "all"

    def __post_init__(self):
        if self.attachment not in ATTACHMENTS:
            raise ValueError(
                "attachment must be 'all' (flow gradient into the encoder through "
                "history, noisy block, and target) or 'history' (history only)"
            )
        for name in ("mask_weight", "flow_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")
        if not 0 < self.mask_probability < 1:
            raise ValueError("mask_probability must be strictly between zero and one")
        if (
            not math.isfinite(self.time_loc)
            or not math.isfinite(self.time_scale)
            or self.time_scale <= 0
        ):
            raise ValueError("invalid logit-normal time distribution")
