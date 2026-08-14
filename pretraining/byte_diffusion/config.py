"""Versioned, validated configuration for the byte-diffusion family."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace as dataclass_replace
from enum import IntEnum
import math
import os
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
        "blt_exact_k",
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

    @classmethod
    def blt_entropy_reference(
        cls,
        *,
        block_length: int = 8,
        branch_byte_budget: int = 8_192,
        **overrides: Any,
    ) -> "CorruptionConfig":
        """Fast-BLT entropy geometry with a block-size-independent patcher.

        ``block_length`` is the fixed diffusion horizon from Fast-BLT.  It is
        deliberately unrelated to the entropy patcher's maximum physical
        patch size.  Keeping the byte-slot budget explicit makes B4/B8/B16
        recipe comparisons preserve their total branch supervision.
        """

        if block_length <= 0 or branch_byte_budget <= 0:
            raise ValueError("entropy block length and branch budget must be positive")
        if branch_byte_budget % block_length:
            raise ValueError("entropy branch budget must be divisible by block length")
        values = {
            "kind": "blt_exact_k",
            "canvas_length": block_length,
            "branches_per_row": branch_byte_budget // block_length,
            **overrides,
        }
        return cls(**values)


@dataclass(frozen=True)
class ByteDiffusionConfig:
    schema_version: int = 5
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
    global_ffn_kind: Literal["swiglu", "relu_squared"] = "swiglu"
    patch_stride: int = 4
    local_window: int = 512
    global_window: int | None = 512
    decoder_prefix_window: int | None = 512
    decoder_branch_attention: Literal[
        "shared_flex", "duplicated_varlen"
    ] = "shared_flex"
    rope_theta: float = 500_000.0
    ngram_enabled: bool = True
    ngram_table_size: int = 32_768
    ngram_rank: int = 16
    ngram_orders: tuple[int, ...] = (3, 4, 5, 6, 7, 8)
    ngram_hash: Literal["blt_prime", "legacy257"] = "blt_prime"
    ngram_factor_init: Literal["weak", "scale_matched"] = "weak"
    ngram_aggregation: Literal["sum", "mean"] = "sum"
    ngram_table_sharing: Literal["shared", "per_order"] = "shared"
    decoder_conditioning: Literal[
        "split_cross_attention", "gated_projection", "rmsnorm_projection"
    ] = "gated_projection"
    decoder_split_residual_scale: float = 1.0
    output_tied: bool = False
    # Fast-BLT distinguishes clean/noisy streams through token identity and
    # attention topology; a learned recipe-type embedding is a separate
    # architecture ablation, not part of the reference-aligned default.
    mode_embeddings: bool = False
    explicit_timestep: bool = False
    self_conditioning: bool = False
    self_conditioning_hidden: int = 64
    # Scaling-DLLMs uses 256 Fourier features and a 128-wide condition MLP.
    # The smaller defaults preserve the measured reference cell; both widths
    # are explicit so the paper-sized conditioner can be isolated in a 2k
    # ablation instead of silently changing the architecture.
    duo_time_features: int = 64
    duo_time_condition_dim: int = 32
    # Scratch data contains bytes plus EOT only. Reserved typed controls stay
    # in the head for post-training but are outside diffusion support until
    # the post-training corpus supplies positive targets for them.
    duo_diffusion_atoms: int = 257
    duo_variable_length_probability: float = 0.01
    duo_noisy_ngrams: bool = True
    duo_random_phase_training: bool = False
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
            "encoder_ffn_dim": self.encoder_ffn_dim,
            "global_ffn_dim": self.global_ffn_dim,
            "decoder_ffn_dim": self.decoder_ffn_dim,
            "patch_stride": self.patch_stride,
            "local_window": self.local_window,
            "ngram_table_size": self.ngram_table_size,
            "ngram_rank": self.ngram_rank,
            "self_conditioning_hidden": self.self_conditioning_hidden,
            "duo_time_features": self.duo_time_features,
            "duo_time_condition_dim": self.duo_time_condition_dim,
            "duo_diffusion_atoms": self.duo_diffusion_atoms,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"architecture values must be positive: {positive}")
        if self.local_dim % self.local_heads:
            raise ValueError("local_dim must divide local_heads")
        if self.global_dim % self.global_heads:
            raise ValueError("global_dim must divide global_heads")
        if self.duo_time_features % 2:
            raise ValueError("duo_time_features must be even")
        if self.duo_diffusion_atoms not in {257, 261}:
            raise ValueError(
                "duo_diffusion_atoms must match a supported posterior size: "
                "257 or 261"
            )
        if not self.vocab.eot_id < self.duo_diffusion_atoms <= self.vocab.output_size:
            raise ValueError("duo_diffusion_atoms must include EOT and fit the output head")
        if not 0.0 <= self.duo_variable_length_probability <= 1.0:
            raise ValueError("duo_variable_length_probability must lie in [0, 1]")
        if self.validate_production_layout and self.local_dim // self.local_heads != 64:
            raise ValueError("the optimized local head dimension must be 64")
        if self.validate_production_layout and self.global_dim // self.global_heads != 64:
            raise ValueError("the optimized global head dimension must be 64")
        if self.patch_stride != 4:
            raise ValueError("version 1 implements fixed stride four")
        if tuple(sorted(set(self.ngram_orders))) != self.ngram_orders:
            raise ValueError("ngram_orders must be sorted and unique")
        if self.ngram_hash not in {"blt_prime", "legacy257"}:
            raise ValueError(f"unsupported ngram_hash {self.ngram_hash!r}")
        if self.ngram_factor_init not in {"weak", "scale_matched"}:
            raise ValueError(
                f"unsupported ngram_factor_init {self.ngram_factor_init!r}"
            )
        if self.ngram_aggregation not in {"sum", "mean"}:
            raise ValueError(
                f"unsupported ngram_aggregation {self.ngram_aggregation!r}"
            )
        if self.ngram_table_sharing not in {"shared", "per_order"}:
            raise ValueError(
                f"unsupported ngram_table_sharing {self.ngram_table_sharing!r}"
            )
        if self.decoder_conditioning not in {
            "split_cross_attention",
            "gated_projection",
            "rmsnorm_projection",
        }:
            raise ValueError(
                f"unsupported decoder_conditioning {self.decoder_conditioning!r}"
            )
        if (
            not math.isfinite(self.decoder_split_residual_scale)
            or self.decoder_split_residual_scale <= 0
        ):
            raise ValueError("decoder_split_residual_scale must be finite and positive")
        if self.global_ffn_kind not in {"swiglu", "relu_squared"}:
            raise ValueError(f"unsupported global_ffn_kind {self.global_ffn_kind!r}")
        if self.decoder_prefix_window is not None and self.decoder_prefix_window <= 0:
            raise ValueError("decoder_prefix_window must be positive or None")
        if self.global_window is not None and self.global_window <= 0:
            raise ValueError("global_window must be positive or None")
        if self.decoder_branch_attention not in {
            "shared_flex",
            "duplicated_varlen",
        }:
            raise ValueError(
                "decoder_branch_attention must be shared_flex or duplicated_varlen"
            )
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
            "global_window": 16,
            "decoder_prefix_window": 16,
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
        return 23_010_306


def model_config_from_env(*, tiny: bool = False) -> ByteDiffusionConfig:
    """Build one explicit architecture cell from the ablation environment."""

    base = ByteDiffusionConfig.tiny() if tiny else ByteDiffusionConfig()
    overrides: dict[str, object] = {}
    string_fields = {
        "BYTE_DIFFUSION_NGRAM_HASH": "ngram_hash",
        "BYTE_DIFFUSION_NGRAM_FACTOR_INIT": "ngram_factor_init",
        "BYTE_DIFFUSION_NGRAM_AGGREGATION": "ngram_aggregation",
        "BYTE_DIFFUSION_NGRAM_TABLE_SHARING": "ngram_table_sharing",
        "BYTE_DIFFUSION_DECODER_CONDITIONING": "decoder_conditioning",
        "BYTE_DIFFUSION_GLOBAL_FFN_KIND": "global_ffn_kind",
        "BYTE_DIFFUSION_DECODER_BRANCH_ATTENTION": "decoder_branch_attention",
    }
    for environment, field in string_fields.items():
        if environment in os.environ:
            overrides[field] = os.environ[environment]
    if "BYTE_DIFFUSION_NGRAM_ENABLED" in os.environ:
        value = os.environ["BYTE_DIFFUSION_NGRAM_ENABLED"]
        if value not in {"0", "1"}:
            raise ValueError("BYTE_DIFFUSION_NGRAM_ENABLED must be 0 or 1")
        overrides["ngram_enabled"] = value == "1"
    positive_integer_fields = {
        "BYTE_DIFFUSION_ENCODER_LAYERS": "encoder_layers",
        "BYTE_DIFFUSION_GLOBAL_LAYERS": "global_layers",
        "BYTE_DIFFUSION_DECODER_LAYERS": "decoder_layers",
        "BYTE_DIFFUSION_ENCODER_FFN_DIM": "encoder_ffn_dim",
        "BYTE_DIFFUSION_GLOBAL_FFN_DIM": "global_ffn_dim",
        "BYTE_DIFFUSION_DECODER_FFN_DIM": "decoder_ffn_dim",
        "BYTE_DIFFUSION_NGRAM_TABLE_SIZE": "ngram_table_size",
        "BYTE_DIFFUSION_NGRAM_RANK": "ngram_rank",
        "BYTE_DIFFUSION_SELF_CONDITIONING_HIDDEN": "self_conditioning_hidden",
        "BYTE_DUO_TIME_FEATURES": "duo_time_features",
        "BYTE_DUO_TIME_CONDITION_DIM": "duo_time_condition_dim",
        "BYTE_DUO_DIFFUSION_ATOMS": "duo_diffusion_atoms",
    }
    for environment, field in positive_integer_fields.items():
        if environment not in os.environ:
            continue
        value = int(os.environ[environment])
        if value <= 0:
            raise ValueError(f"{environment} must be positive")
        overrides[field] = value
    boolean_fields = {
        "BYTE_DIFFUSION_OUTPUT_TIED": "output_tied",
        "BYTE_DIFFUSION_MODE_EMBEDDINGS": "mode_embeddings",
        "BYTE_DIFFUSION_EXPLICIT_TIMESTEP": "explicit_timestep",
        "BYTE_DIFFUSION_SELF_CONDITIONING": "self_conditioning",
        "BYTE_DUO_NOISY_NGRAMS": "duo_noisy_ngrams",
        "BYTE_DUO_RANDOM_PHASE_TRAINING": "duo_random_phase_training",
    }
    for environment, field in boolean_fields.items():
        if environment not in os.environ:
            continue
        value = os.environ[environment]
        if value not in {"0", "1"}:
            raise ValueError(f"{environment} must be 0 or 1")
        overrides[field] = value == "1"
    if "BYTE_DIFFUSION_DECODER_SPLIT_RESIDUAL_SCALE" in os.environ:
        overrides["decoder_split_residual_scale"] = float(
            os.environ["BYTE_DIFFUSION_DECODER_SPLIT_RESIDUAL_SCALE"]
        )
    if "BYTE_DUO_VARIABLE_LENGTH_PROBABILITY" in os.environ:
        overrides["duo_variable_length_probability"] = float(
            os.environ["BYTE_DUO_VARIABLE_LENGTH_PROBABILITY"]
        )
    if "BYTE_DIFFUSION_DECODER_PREFIX_WINDOW" in os.environ:
        raw_window = os.environ["BYTE_DIFFUSION_DECODER_PREFIX_WINDOW"].strip().lower()
        if raw_window in {"none", "unbounded", "exact"}:
            overrides["decoder_prefix_window"] = None
        else:
            window = int(raw_window)
            if window <= 0:
                raise ValueError(
                    "BYTE_DIFFUSION_DECODER_PREFIX_WINDOW must be positive or unbounded"
                )
            overrides["decoder_prefix_window"] = window
    if "BYTE_DIFFUSION_GLOBAL_WINDOW" in os.environ:
        raw_window = os.environ["BYTE_DIFFUSION_GLOBAL_WINDOW"].strip().lower()
        if raw_window in {"none", "unbounded", "exact"}:
            overrides["global_window"] = None
        else:
            window = int(raw_window)
            if window <= 0:
                raise ValueError(
                    "BYTE_DIFFUSION_GLOBAL_WINDOW must be positive or unbounded"
                )
            overrides["global_window"] = window
    if "BYTE_DIFFUSION_NGRAM_ORDERS" in os.environ:
        try:
            orders = tuple(
                int(value.strip())
                for value in os.environ["BYTE_DIFFUSION_NGRAM_ORDERS"].split(",")
                if value.strip()
            )
        except ValueError as error:
            raise ValueError(
                "BYTE_DIFFUSION_NGRAM_ORDERS must be comma-separated integers"
            ) from error
        if not orders:
            raise ValueError("BYTE_DIFFUSION_NGRAM_ORDERS cannot be empty")
        overrides["ngram_orders"] = orders
    return dataclass_replace(base, **overrides)
