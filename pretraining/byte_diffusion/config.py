"""Versioned, validated configuration for the byte-diffusion family."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Literal


class ModelMode(IntEnum):
    AR = 0
    BLT_D = 1
    CANVAS = 2


@dataclass(frozen=True)
class AtomicVocabulary:
    """Atomic output and input-only ids.

    Clean ids 0..260 include literal octets and five typed controls. MASK and
    PAD are input-only. Their exact ordering is bound by the shared tokenizer,
    data, checkpoint, and export manifest.
    """

    byte_values: int = 256
    clean_specials: int = 5
    mask_id: int = 261
    pad_id: int = 262
    eot_id: int = 256

    def __post_init__(self) -> None:
        if self.byte_values != 256:
            raise ValueError("byte_values must be exactly 256")
        if self.clean_specials <= 0:
            raise ValueError("clean_specials must be positive")
        if self.mask_id != self.output_size:
            raise ValueError("MASK must immediately follow predictable ids")
        if self.pad_id != self.mask_id + 1:
            raise ValueError("PAD must immediately follow MASK")
        if not 0 <= self.eot_id < self.output_size:
            raise ValueError("EOT must be a predictable atomic id")

    @property
    def output_size(self) -> int:
        return self.byte_values + self.clean_specials

    @property
    def input_size(self) -> int:
        return self.output_size + 2


@dataclass(frozen=True)
class CorruptionConfig:
    kind: Literal[
        "blt_bernoulli",
        "absorbing_rb",
        "allmask_50",
        "whole_patch",
        "contiguous_patch_span",
        "uniform_replacement",
    ] = "absorbing_rb"
    canvas_length: int = 128
    branches_per_row: int = 1
    patch_stride: int = 4

    def __post_init__(self) -> None:
        if self.canvas_length <= 0 or self.canvas_length % self.patch_stride:
            raise ValueError("canvas_length must be a positive patch multiple")
        if self.branches_per_row <= 0:
            raise ValueError("branches_per_row must be positive")

    @property
    def corrupted_positions_per_row(self) -> int:
        return self.canvas_length * self.branches_per_row

    @classmethod
    def canvas512(cls, **overrides: Any) -> "CorruptionConfig":
        values = {
            "kind": "absorbing_rb",
            "canvas_length": 512,
            "branches_per_row": 1,
            **overrides,
        }
        return cls(**values)

    @classmethod
    def blt16_reference(cls, **overrides: Any) -> "CorruptionConfig":
        values = {
            "kind": "blt_bernoulli",
            "canvas_length": 16,
            "branches_per_row": 32,
            **overrides,
        }
        return cls(**values)


@dataclass(frozen=True)
class ByteDiffusionConfig:
    schema_version: int = 1
    vocab: AtomicVocabulary = AtomicVocabulary()
    local_dim: int = 256
    global_dim: int = 512
    local_heads: int = 4
    global_heads: int = 8
    encoder_layers: int = 1
    global_layers: int = 9
    decoder_layers: int = 2
    encoder_ffn_dim: int = 512
    global_ffn_dim: int = 704
    decoder_ffn_dim: int = 512
    patch_stride: int = 4
    local_window: int = 512
    rope_theta: float = 500_000.0
    ngram_enabled: bool = True
    ngram_table_size: int = 32_768
    ngram_rank: int = 16
    ngram_orders: tuple[int, ...] = (3, 4, 5, 6, 7, 8)
    output_tied: bool = False
    explicit_timestep: bool = False
    self_conditioning: bool = False
    self_conditioning_hidden: int = 64
    validate_production_layout: bool = True

    def __post_init__(self) -> None:
        positive = {
            "local_dim": self.local_dim,
            "global_dim": self.global_dim,
            "local_heads": self.local_heads,
            "global_heads": self.global_heads,
            "encoder_layers": self.encoder_layers,
            "global_layers": self.global_layers,
            "decoder_layers": self.decoder_layers,
            "patch_stride": self.patch_stride,
            "local_window": self.local_window,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"architecture values must be positive: {positive}")
        if self.local_dim % self.local_heads:
            raise ValueError("local_dim must divide local_heads")
        if self.global_dim % self.global_heads:
            raise ValueError("global_dim must divide global_heads")
        if self.validate_production_layout and self.local_dim // self.local_heads != 64:
            raise ValueError("the optimized local head dimension must be 64")
        if self.validate_production_layout and self.global_dim // self.global_heads != 64:
            raise ValueError("the optimized global head dimension must be 64")
        if self.patch_stride != 4:
            raise ValueError("version 1 implements fixed stride four")
        if tuple(sorted(set(self.ngram_orders))) != self.ngram_orders:
            raise ValueError("ngram_orders must be sorted and unique")
        if self.output_tied and self.local_dim <= 0:
            raise ValueError("invalid tied output width")

    @classmethod
    def tiny(cls, **overrides: Any) -> "ByteDiffusionConfig":
        values = {
            "local_dim": 32,
            "global_dim": 64,
            "local_heads": 1,
            "global_heads": 1,
            "global_layers": 2,
            "encoder_ffn_dim": 64,
            "global_ffn_dim": 96,
            "decoder_ffn_dim": 64,
            "local_window": 16,
            "ngram_enabled": False,
            "validate_production_layout": False,
            **overrides,
        }
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def production_parameter_target(self) -> int:
        if self != ByteDiffusionConfig():
            raise ValueError("the closed parameter target applies only to the default config")
        return 23_011_584
