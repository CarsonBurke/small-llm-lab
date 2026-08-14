"""Scratch mixed-objective training and evaluation for byte diffusion.

The production entrypoint is deliberately strict: it starts from random
weights, computes the clean AR and denoising objectives in the same optimizer
update, and resumes only from a checkpoint with an identical model, data,
optimizer, distributed, and schedule contract.  The named ``causal_only``
recipe is an experimental control, never an implicit warm-up phase.

GPU callers must invoke the scripts through ``mlq``.  This module contains no
implicit job launch and CPU tests never initialize CUDA.
"""

from __future__ import annotations

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .attention import (
    CanvasBlockMaskMetadata,
    CanvasBranchLayout,
    build_canvas_block_mask,
    canvas_block_mask_metadata,
)
from .config import AtomicVocabulary, ByteDiffusionConfig, CorruptionConfig, ModelMode
from .corruption import (
    CorruptedBatch,
    absorbing_rb,
    allmask_50,
    blt_bernoulli,
    blt_exact_k,
    uniform_replacement,
    whole_patch,
)
from .data import (
    AtomicDocument,
    AtomicIdManifest,
    DeterministicChunkCursor,
    MASK_ID,
    PAD_ID,
    PackedChunk,
    pack_documents,
    packed_chunks_sha256,
)
from .kernels import flash_sdpa_only
from .layers import pack_valid
from .model import ByteDiffusionModel
from .variable_patching import (
    DatasetPatchingSpec,
    build_variable_patch_layout,
    load_dataset_patching_spec,
)
from .objectives import (
    IGNORE_INDEX,
    ar_cross_entropy,
    blt_masked_loss,
    canvas_cross_entropy,
    cross_entropy_per_target,
    cross_entropy_per_row,
    masked_cross_entropy_per_target,
    same_position_targets,
)


CHECKPOINT_SCHEMA = "byte_diffusion_training/v10"
CANONICAL_PRESET = "fast_blt_b4_scratch_v1"
CAUSAL_CONTROL_PRESET = "causal_v5_scratch_v1"
DENSE_FAST_BLT_PRESET = "fast_blt_dense_exactk_scratch_v1"
ENTROPY_FAST_BLT_PRESET = "fast_blt_entropy_exactk_scratch_v1"
PRODUCTION_PRESETS = frozenset(
    {
        CANONICAL_PRESET,
        CAUSAL_CONTROL_PRESET,
        DENSE_FAST_BLT_PRESET,
        ENTROPY_FAST_BLT_PRESET,
    }
)
Recipe = Literal["canvas", "blt_d", "causal_only"]
PatchingPolicy = Literal["fixed_stride_v1", "causal_entropy_v1"]
AttentionPolicy = Literal["flash_sdpa", "dense_reference"]
ObjectiveReduction = Literal["equal_mean", "paper_sum", "row_normalized_sum"]
CompileMode = Literal["default", "reduce-overhead", "max-autotune-no-cudagraphs"]
OptimizerKind = Literal["adamw", "nanogpt_muon"]


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class TrainingRunConfig:
    """Optimizer/run contract, including the standard ablation environment."""

    iterations: int = 2_000
    val_loss_every: int = 20
    train_log_every: int = 10
    validation_chunks: int = 2_048
    validation_microbatch_per_rank: int = 64
    ar_validation_microbatch_per_rank: int = 128
    diffusion_validation_chunks: int = 256
    warmdown_iters: int = 1_200
    run_id: str = "byte_diffusion"
    preset: str | None = None
    initialization_kind: Literal["scratch"] = "scratch"
    seed: int = 1_337
    recipe: Recipe = "canvas"
    patching_policy: PatchingPolicy = "fixed_stride_v1"
    corruption: CorruptionConfig = CorruptionConfig.canvas512()
    microbatch_per_rank: int = 32
    microbatch_token_budget: int = 278_528
    gradient_accumulation: int = 1
    global_batch_size: int | None = None
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    optimizer_kind: OptimizerKind = "adamw"
    embedding_learning_rate: float = 0.6
    output_learning_rate: float = 0.008
    matrix_learning_rate: float = 0.04
    scalar_learning_rate: float = 0.04
    muon_momentum: float = 0.95
    muon_momentum_warmup_start: float = 0.85
    muon_momentum_warmup_steps: int = 500
    muon_backend_steps: int = 5
    lambda_ar: float = 1.0
    objective_reduction: ObjectiveReduction = "equal_mean"
    max_grad_norm: float | None = 1.0
    attention_policy: AttentionPolicy = "flash_sdpa"
    allow_cpu_reference: bool = False
    compile_model: bool = True
    compile_dynamic_shapes: bool = True
    activation_checkpointing: bool = False
    compile_mode: CompileMode = "default"
    cuda_graphs: bool = False
    length_sorted_microbatches: bool = False

    def __post_init__(self) -> None:
        positive = {
            "iterations": self.iterations,
            "val_loss_every": self.val_loss_every,
            "train_log_every": self.train_log_every,
            "validation_chunks": self.validation_chunks,
            "validation_microbatch_per_rank": self.validation_microbatch_per_rank,
            "ar_validation_microbatch_per_rank": (
                self.ar_validation_microbatch_per_rank
            ),
            "diffusion_validation_chunks": self.diffusion_validation_chunks,
            "microbatch_per_rank": self.microbatch_per_rank,
            "microbatch_token_budget": self.microbatch_token_budget,
            "gradient_accumulation": self.gradient_accumulation,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"run dimensions must be positive: {positive}")
        if self.global_batch_size is not None and self.global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive or None")
        if self.diffusion_validation_chunks > self.validation_chunks:
            raise ValueError(
                "diffusion_validation_chunks cannot exceed validation_chunks"
            )
        if not self.run_id:
            raise ValueError("RUN_ID must be non-empty")
        if self.preset is not None and self.preset not in PRODUCTION_PRESETS:
            raise ValueError(f"unknown byte-diffusion preset {self.preset!r}")
        if self.initialization_kind != "scratch":
            raise ValueError("byte diffusion currently supports scratch lineages only")
        if not 0 <= self.warmdown_iters <= self.iterations:
            raise ValueError("WARMDOWN_ITERS must be in [0, ITERATIONS]")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.epsilon <= 0:
            raise ValueError("optimizer values are outside their valid range")
        if self.optimizer_kind not in {"adamw", "nanogpt_muon"}:
            raise ValueError(f"unknown optimizer kind {self.optimizer_kind!r}")
        hybrid_rates = {
            "embedding_learning_rate": self.embedding_learning_rate,
            "output_learning_rate": self.output_learning_rate,
            "matrix_learning_rate": self.matrix_learning_rate,
            "scalar_learning_rate": self.scalar_learning_rate,
        }
        if any(value <= 0 for value in hybrid_rates.values()):
            raise ValueError(f"hybrid optimizer rates must be positive: {hybrid_rates}")
        if not 0 <= self.muon_momentum < 1 or not 0 <= self.muon_momentum_warmup_start < 1:
            raise ValueError("Muon momentum values must lie in [0, 1)")
        if self.muon_momentum_warmup_steps < 0 or self.muon_backend_steps <= 0:
            raise ValueError("Muon warmup must be nonnegative and backend steps positive")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("AdamW betas must lie in [0, 1)")
        if self.lambda_ar < 0:
            raise ValueError("lambda_ar cannot be negative")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None")
        if self.recipe not in {"canvas", "blt_d", "causal_only"}:
            raise ValueError(f"unknown training recipe {self.recipe!r}")
        if self.patching_policy not in {"fixed_stride_v1", "causal_entropy_v1"}:
            raise ValueError(f"unknown patching policy {self.patching_policy!r}")
        if not isinstance(self.corruption, CorruptionConfig):
            raise TypeError("corruption must be a CorruptionConfig")
        if self.objective_reduction not in {
            "equal_mean",
            "paper_sum",
            "row_normalized_sum",
        }:
            raise ValueError(
                f"unknown objective reduction {self.objective_reduction!r}"
            )
        if self.attention_policy not in {"flash_sdpa", "dense_reference"}:
            raise ValueError(f"unknown attention policy {self.attention_policy!r}")
        if self.compile_mode not in {
            "default",
            "reduce-overhead",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError(f"unknown compile mode {self.compile_mode!r}")
        if self.recipe == "causal_only" and self.lambda_ar == 0:
            raise ValueError("causal_only requires a nonzero AR coefficient")
        if (
            self.activation_checkpointing
            and self.compile_model
            and self.recipe in {"canvas", "blt_d"}
        ):
            raise ValueError(
                "activation checkpointing is incompatible with compiled branch "
                "training because backward recomputation escapes compiled FlexAttention"
            )
        if (
            self.objective_reduction in {"paper_sum", "row_normalized_sum"}
            and self.recipe != "blt_d"
        ):
            raise ValueError(
                "summed BLT reductions are defined only for the BLT-D recipe"
            )
        if (
            self.objective_reduction in {"paper_sum", "row_normalized_sum"}
            and self.corruption.kind not in {"blt_bernoulli", "blt_exact_k"}
        ):
            raise ValueError("summed BLT reductions require a Fast-BLT estimator")
        if self.corruption.kind == "uniform_replacement":
            raise ValueError(
                "uniform replacement requires the unimplemented reference "
                "self-conditioning forward and is not a production recipe"
            )
        if self.preset in PRODUCTION_PRESETS:
            expected = {
                "iterations": 2_000,
                "val_loss_every": 20,
                "train_log_every": 10,
                "validation_chunks": 2_048,
                "validation_microbatch_per_rank": 64,
                "ar_validation_microbatch_per_rank": 128,
                "diffusion_validation_chunks": 256,
                "warmdown_iters": 1_200,
                "seed": 1_337,
                "initialization_kind": "scratch",
                "recipe": "blt_d",
                "patching_policy": "fixed_stride_v1",
                "corruption": CorruptionConfig(
                    kind="blt_bernoulli",
                    canvas_length=4,
                    branches_per_row=128,
                ),
                "global_batch_size": 249,
                "microbatch_per_rank": 32,
                "microbatch_token_budget": 278_528,
                "learning_rate": 3e-4,
                "weight_decay": 0.1,
                "beta1": 0.9,
                "beta2": 0.95,
                "epsilon": 1e-8,
                "optimizer_kind": "adamw",
                "embedding_learning_rate": 0.6,
                "output_learning_rate": 0.008,
                "matrix_learning_rate": 0.04,
                "scalar_learning_rate": 0.04,
                "muon_momentum": 0.95,
                "muon_momentum_warmup_start": 0.85,
                "muon_momentum_warmup_steps": 500,
                "muon_backend_steps": 5,
                "lambda_ar": 1.0,
                "objective_reduction": "paper_sum",
                "max_grad_norm": 1.0,
                "attention_policy": "flash_sdpa",
                "compile_model": True,
                "compile_dynamic_shapes": True,
                "activation_checkpointing": False,
                "compile_mode": "default",
                "cuda_graphs": False,
                "length_sorted_microbatches": False,
            }
            if self.preset == CAUSAL_CONTROL_PRESET:
                expected.update(
                    recipe="causal_only",
                    corruption=CorruptionConfig.canvas512(),
                    objective_reduction="equal_mean",
                )
            elif self.preset == DENSE_FAST_BLT_PRESET:
                expected.update(
                    corruption=CorruptionConfig(
                        kind="blt_exact_k",
                        canvas_length=4,
                        branches_per_row=2_048,
                    ),
                    objective_reduction="row_normalized_sum",
                )
            elif self.preset == ENTROPY_FAST_BLT_PRESET:
                expected.update(
                    patching_policy="causal_entropy_v1",
                    corruption=CorruptionConfig.blt_entropy_reference(),
                    objective_reduction="row_normalized_sum",
                )
            observed = {name: getattr(self, name) for name in expected}
            mismatches = {
                name: (observed[name], value)
                for name, value in expected.items()
                if observed[name] != value
            }
            if mismatches:
                raise ValueError(
                    f"preset {self.preset!r} contract mismatch: {mismatches}"
                )

    @classmethod
    def from_env(cls, **overrides: Any) -> "TrainingRunConfig":
        iterations = _positive_int_env("ITERATIONS", 2_000)
        default_warmdown = 1_200 if iterations >= 2_000 else 0
        preset = os.environ.get("BYTE_DIFFUSION_PRESET") or None
        default_recipe = (
            "causal_only" if preset == CAUSAL_CONTROL_PRESET else "blt_d"
        )
        recipe = os.environ.get("BYTE_DIFFUSION_RECIPE", default_recipe)
        microbatch = int(os.environ.get("BYTE_DIFFUSION_MICROBATCH", "32"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        # 249 full 8192-byte pages matches nanoGPT's measured 2.035M-byte
        # update exposure. Canvas keeps its frozen 256-row ablation contract.
        default_global_batch = (
            249 if recipe == "blt_d" or preset in PRODUCTION_PRESETS else 256
        )
        global_batch = int(
            os.environ.get(
                "BYTE_DIFFUSION_GLOBAL_BATCH", str(default_global_batch)
            )
        )
        if microbatch <= 0 or world_size <= 0 or global_batch <= 0:
            raise ValueError(
                "microbatch, WORLD_SIZE, and BYTE_DIFFUSION_GLOBAL_BATCH "
                "must be positive"
            )
        local_counts = distributed_local_counts(global_batch, world_size)
        local_microsteps = tuple(-(-count // microbatch) for count in local_counts)
        if len(set(local_microsteps)) != 1:
            raise ValueError(
                "uneven global batch would give ranks different backward-call "
                f"counts: rows={local_counts}, microsteps={local_microsteps}"
            )
        default_accumulation = local_microsteps[0]
        values: dict[str, Any] = {
            "iterations": iterations,
            "val_loss_every": _positive_int_env("VAL_LOSS_EVERY", 20),
            "train_log_every": _positive_int_env("TRAIN_LOG_EVERY", 10),
            "validation_chunks": _positive_int_env(
                "BYTE_DIFFUSION_VALIDATION_CHUNKS", 2_048
            ),
            "validation_microbatch_per_rank": _positive_int_env(
                "BYTE_DIFFUSION_VALIDATION_MICROBATCH", 64
            ),
            "ar_validation_microbatch_per_rank": _positive_int_env(
                "BYTE_DIFFUSION_AR_VALIDATION_MICROBATCH", 128
            ),
            "diffusion_validation_chunks": _positive_int_env(
                "BYTE_DIFFUSION_DIFFUSION_VALIDATION_CHUNKS", 256
            ),
            "warmdown_iters": int(
                os.environ.get("WARMDOWN_ITERS", str(default_warmdown))
            ),
            "run_id": os.environ.get("RUN_ID", "byte_diffusion"),
            "preset": preset,
            "seed": int(os.environ.get("SEED", "1337")),
            "recipe": recipe,
            "patching_policy": os.environ.get(
                "BYTE_DIFFUSION_PATCHING_POLICY",
                (
                    "causal_entropy_v1"
                    if preset == ENTROPY_FAST_BLT_PRESET
                    else "fixed_stride_v1"
                ),
            ),
            "microbatch_per_rank": microbatch,
            "microbatch_token_budget": _positive_int_env(
                "BYTE_DIFFUSION_MICROBATCH_TOKEN_BUDGET", 278_528
            ),
            "gradient_accumulation": int(
                os.environ.get(
                    "BYTE_DIFFUSION_GRAD_ACCUM", str(default_accumulation)
                )
            ),
            "global_batch_size": global_batch,
            "learning_rate": float(os.environ.get("BYTE_DIFFUSION_LR", "3e-4")),
            "weight_decay": float(
                os.environ.get("BYTE_DIFFUSION_WEIGHT_DECAY", "0.1")
            ),
            "lambda_ar": float(os.environ.get("BYTE_DIFFUSION_LAMBDA_AR", "1")),
            "optimizer_kind": os.environ.get(
                "BYTE_DIFFUSION_OPTIMIZER", "adamw"
            ),
            "embedding_learning_rate": float(
                os.environ.get("BYTE_DIFFUSION_EMBED_LR", "0.6")
            ),
            "output_learning_rate": float(
                os.environ.get("BYTE_DIFFUSION_OUTPUT_LR", "0.008")
            ),
            "matrix_learning_rate": float(
                os.environ.get("BYTE_DIFFUSION_MATRIX_LR", "0.04")
            ),
            "scalar_learning_rate": float(
                os.environ.get("BYTE_DIFFUSION_SCALAR_LR", "0.04")
            ),
            "muon_momentum": float(
                os.environ.get("BYTE_DIFFUSION_MUON_MOMENTUM", "0.95")
            ),
            "muon_momentum_warmup_start": float(
                os.environ.get("BYTE_DIFFUSION_MUON_MOMENTUM_START", "0.85")
            ),
            "muon_momentum_warmup_steps": int(
                os.environ.get("BYTE_DIFFUSION_MUON_MOMENTUM_WARMUP", "500")
            ),
            "muon_backend_steps": _positive_int_env(
                "BYTE_DIFFUSION_MUON_BACKEND_STEPS", 5
            ),
            "objective_reduction": os.environ.get(
                "BYTE_DIFFUSION_OBJECTIVE_REDUCTION",
                (
                    "row_normalized_sum"
                    if preset in {
                        DENSE_FAST_BLT_PRESET,
                        ENTROPY_FAST_BLT_PRESET,
                    }
                    else "paper_sum" if recipe == "blt_d" else "equal_mean"
                ),
            ),
            "attention_policy": os.environ.get(
                "BYTE_DIFFUSION_ATTENTION_BACKEND", "flash_sdpa"
            ),
            "allow_cpu_reference": os.environ.get(
                "BYTE_DIFFUSION_ALLOW_CPU_REFERENCE", "0"
            )
            == "1",
            "compile_model": os.environ.get("BYTE_DIFFUSION_COMPILE", "1") == "1",
            "compile_dynamic_shapes": os.environ.get(
                "BYTE_DIFFUSION_DYNAMIC_SHAPES", "1"
            )
            == "1",
            "activation_checkpointing": os.environ.get(
                "BYTE_DIFFUSION_ACTIVATION_CHECKPOINTING", "0"
            )
            == "1",
            "compile_mode": os.environ.get(
                "BYTE_DIFFUSION_COMPILE_MODE", "default"
            ),
            "cuda_graphs": os.environ.get("BYTE_DIFFUSION_CUDA_GRAPHS", "0") == "1",
            "length_sorted_microbatches": os.environ.get(
                "BYTE_DIFFUSION_SORT_MICROBATCHES", "0"
            )
            == "1",
        }
        default_kind = (
            "blt_exact_k"
            if preset in {DENSE_FAST_BLT_PRESET, ENTROPY_FAST_BLT_PRESET}
            else "blt_bernoulli" if recipe == "blt_d" else "absorbing_rb"
        )
        # The diffusion horizon is independent of entropy patch geometry.
        # The pinned entropy preset retains B8; B4/B16 are explicit recipe
        # variants with the same branch-byte budget.
        default_length = (
            8
            if preset == ENTROPY_FAST_BLT_PRESET
            else 4 if recipe == "blt_d" else 512
        )
        # B4 x 2048 and B8 x 1024 both expose 8192 branch byte-slots per row.
        # Holding that budget fixed isolates patch geometry from supervision
        # and decoder-work changes in the entropy reference.
        default_branches = (
            1_024
            if preset == ENTROPY_FAST_BLT_PRESET
            else 2_048
            if preset == DENSE_FAST_BLT_PRESET
            else 128 if recipe == "blt_d" else 1
        )
        corruption_kind = os.environ.get(
            "BYTE_DIFFUSION_CORRUPTION", default_kind
        )
        values["corruption"] = CorruptionConfig(
            kind=corruption_kind,
            canvas_length=int(
                os.environ.get(
                    "BYTE_DIFFUSION_CANVAS_LENGTH", str(default_length)
                )
            ),
            branches_per_row=int(
                os.environ.get(
                    "BYTE_DIFFUSION_BRANCHES", str(default_branches)
                )
            ),
        )
        values.update(overrides)
        result = cls(**values)
        if result.preset in PRODUCTION_PRESETS and result.global_batch_size != 249:
            raise ValueError(
                f"preset {result.preset!r} requires global batch 249, "
                f"observed {result.global_batch_size}"
            )
        if result.global_batch_size is None:
            raise ValueError("configured global batch cannot be None")
        result_counts = distributed_local_counts(
            result.global_batch_size, world_size
        )
        result_microsteps = tuple(
            -(-count // result.microbatch_per_rank) for count in result_counts
        )
        if len(set(result_microsteps)) != 1:
            raise ValueError(
                "uneven global batch would give ranks different backward-call "
                f"counts: rows={result_counts}, microsteps={result_microsteps}"
            )
        required_accumulation = result_microsteps[0]
        if result.gradient_accumulation != required_accumulation:
            raise ValueError(
                "BYTE_DIFFUSION_GRAD_ACCUM must equal ceil(local global-batch "
                f"share / microbatch) = {required_accumulation}"
            )
        return result

    def contract_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    initialized_here: bool = False

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    @classmethod
    def from_environment(cls, device_type: str) -> "DistributedContext":
        launched = "RANK" in os.environ or "WORLD_SIZE" in os.environ
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        initialized_here = False
        if launched and not dist.is_initialized():
            backend = "nccl" if device_type == "cuda" else "gloo"
            dist.init_process_group(backend=backend)
            initialized_here = True
        if dist.is_initialized():
            rank, world_size = dist.get_rank(), dist.get_world_size()
        return cls(rank, local_rank, world_size, initialized_here)

    def barrier(self) -> None:
        if dist.is_initialized():
            dist.barrier()

    def close(self) -> None:
        if self.initialized_here and dist.is_initialized():
            dist.destroy_process_group()


@dataclass(frozen=True)
class TrainingBatch:
    ids: Tensor
    valid: Tensor
    ar_targets: Tensor
    bos_targets: Tensor
    positions: Tensor
    full_valid: bool
    document_ids: Tensor | None = None
    # ``bos_targets`` is packed: it contains one target per true document
    # start, not one padded slot per physical row.  This mapping is needed by
    # Fast-BLT's per-row objective reduction when a packed page contains more
    # than one document.
    bos_row_indices: Tensor | None = None
    isolate_documents: bool = False
    byte_indices: Tensor | None = None
    byte_cu_seqlens: Tensor | None = None
    patch_indices: Tensor | None = None
    patch_cu_seqlens: Tensor | None = None
    condition_patch_indices: Tensor | None = None
    global_patch_sources: Tensor | None = None
    global_patch_positions: Tensor | None = None
    physical_to_global_patch_indices: Tensor | None = None
    bos_condition_indices: Tensor | None = None
    prior_condition_indices: Tensor | None = None
    patch_offsets: Tensor | None = None
    patch_byte_cu_seqlens: Tensor | None = None
    max_patch_size: int | None = None
    physical_patch_row_indices: Tensor | None = None
    physical_patch_start_columns: Tensor | None = None
    physical_patch_lengths: Tensor | None = None
    physical_patch_prior_condition_indices: Tensor | None = None

    def _map_tensors(self, transform) -> "TrainingBatch":
        return TrainingBatch(
            **{
                field.name: (
                    transform(value)
                    if isinstance(value := getattr(self, field.name), Tensor)
                    else value
                )
                for field in fields(self)
            }
        )

    def to(
        self, device: torch.device, *, non_blocking: bool = False
    ) -> "TrainingBatch":
        return self._map_tensors(
            lambda tensor: tensor.to(device, non_blocking=non_blocking)
        )

    def pin_memory(self) -> "TrainingBatch":
        return self._map_tensors(Tensor.pin_memory)

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                value.record_stream(stream)


@dataclass(frozen=True)
class ValidationChunkIdentity:
    chunk_index: int
    stream_start: int


@dataclass(frozen=True)
class StepLoss:
    total: Tensor
    ar: Tensor
    diffusion: Tensor
    ar_targets: Tensor
    diffusion_targets: Tensor
    noise_masked: Tensor
    noise_eligible: Tensor
    noise_nll: Tensor
    noise_correct: Tensor


@dataclass(frozen=True)
class CanvasCorruptionPlan:
    clean_branches: Tensor
    noisy_branches: Tensor
    branch_valid: Tensor
    branch_starts: Tensor
    active: Tensor


@dataclass(frozen=True)
class BltCorruptionPlan:
    clean_blocks: Tensor
    noisy_blocks: Tensor
    branch_valid: Tensor
    active: Tensor
    block_starts: Tensor
    condition_indices: Tensor
    block_length: int
    t: Tensor | None
    sampling_weight: Tensor | None = None
    branch_query_indices: Tensor | None = None
    branch_kv_indices: Tensor | None = None
    branch_query_cu_seqlens: Tensor | None = None
    branch_kv_cu_seqlens: Tensor | None = None

    def to(self, device: torch.device) -> "BltCorruptionPlan":
        return BltCorruptionPlan(
            **{
                field.name: (
                    value.to(device) if isinstance(value := getattr(self, field.name), Tensor) else value
                )
                for field in fields(self)
            }
        )


@dataclass(frozen=True)
class BltSamplingPlan:
    """CPU-resolved origins and sparse branch topology for one BLT batch."""

    block_starts: Tensor
    sampling_weight: Tensor
    selected: Tensor
    condition_indices: Tensor
    branch_valid: Tensor
    block_mask_metadata: CanvasBlockMaskMetadata | None = None
    branch_query_indices: Tensor | None = None
    branch_kv_indices: Tensor | None = None
    branch_query_cu_seqlens: Tensor | None = None
    branch_kv_cu_seqlens: Tensor | None = None
    validation_active: Tensor | None = None
    validation_t: Tensor | None = None

    def _map_tensors(self, transform) -> "BltSamplingPlan":
        return BltSamplingPlan(
            **{
                field.name: (
                    transform(value)
                    if (value := getattr(self, field.name)) is not None
                    else None
                )
                for field in fields(self)
            }
        )

    def to(
        self, device: torch.device, *, non_blocking: bool = False
    ) -> "BltSamplingPlan":
        return self._map_tensors(
            lambda value: value.to(device, non_blocking=non_blocking)
            if isinstance(value, Tensor)
            else value.to(device, non_blocking=non_blocking)
        )

    def pin_memory(self) -> "BltSamplingPlan":
        return self._map_tensors(
            lambda value: value.pin_memory()
            if isinstance(value, Tensor)
            else value.pin_memory()
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                value.record_stream(stream)
            elif isinstance(value, CanvasBlockMaskMetadata):
                value.record_stream(stream)


@dataclass(frozen=True)
class StepMetrics:
    step: int
    total: float
    ar: float
    diffusion: float
    learning_rate: float
    elapsed_ms: float
    ar_targets: int
    diffusion_targets: int
    microsteps: int
    max_microbatch: int
    max_physical_positions: int
    preclip_grad_norm: float
    grad_clip_scale: float
    mean_noise_fraction: float
    all_mask_fraction: float
    noise_bucket_counts: tuple[int, int, int, int, int, int, int]
    noise_bucket_nll: tuple[float, float, float, float, float, float, float]
    noise_bucket_accuracy: tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class ValidationMetrics:
    ar_loss: float
    bpb: float
    atomic_bpb: float
    diffusion_loss: float
    diffusion_elbo_proxy_bpb: float | None
    diffusion_elbo_proxy_nats_per_block_atom: float | None
    diffusion_elbo_atoms: float
    ar_targets: int
    literal_bytes: int
    special_targets: int
    diffusion_targets: int
    diffusion_chunks: int
    elapsed_ms: float
    diffusion_role_nll: tuple[float, float, float, float, float, float]
    diffusion_role_counts: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class GradientInterferenceMetrics:
    ar_norm: float
    diffusion_norm: float
    cosine: float


def gradient_interference_metrics(
    ar_loss: Tensor,
    diffusion_loss: Tensor,
    parameters: Sequence[nn.Parameter],
) -> GradientInterferenceMetrics:
    """Measure two objective gradients without mutating optimizer gradients."""

    if not parameters:
        raise ValueError("gradient diagnostics require shared parameters")
    ar_gradients = torch.autograd.grad(
        ar_loss, parameters, retain_graph=True, allow_unused=True
    )
    diffusion_gradients = torch.autograd.grad(
        diffusion_loss, parameters, allow_unused=True
    )
    dot = ar_loss.new_zeros((), dtype=torch.float64)
    ar_squared = dot.clone()
    diffusion_squared = dot.clone()
    for ar_gradient, diffusion_gradient in zip(
        ar_gradients, diffusion_gradients, strict=True
    ):
        if ar_gradient is not None:
            ar_squared += ar_gradient.detach().double().square().sum()
        if diffusion_gradient is not None:
            diffusion_squared += diffusion_gradient.detach().double().square().sum()
        if ar_gradient is not None and diffusion_gradient is not None:
            dot += (
                ar_gradient.detach().double() * diffusion_gradient.detach().double()
            ).sum()
    ar_norm = ar_squared.sqrt()
    diffusion_norm = diffusion_squared.sqrt()
    denominator = ar_norm * diffusion_norm
    cosine = dot / denominator if float(denominator) > 0 else dot.new_zeros(())
    return GradientInterferenceMetrics(
        ar_norm=float(ar_norm),
        diffusion_norm=float(diffusion_norm),
        cosine=float(cosine),
    )


def _packed_document_layout(
    valid: Tensor,
    document_ids: Tensor,
    positions: Tensor,
    *,
    patch_stride: int = 4,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build document-local byte and virtual-BOS global-patch metadata."""

    if valid.device.type != "cpu":
        raise ValueError("packed document layout must be collated on CPU")
    if valid.shape != document_ids.shape or valid.shape != positions.shape:
        raise ValueError("document layout tensors must align")
    if valid.shape[1] % patch_stride:
        raise ValueError("document pages must end on a patch boundary")

    byte_indices = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
    flat_rows = torch.div(byte_indices, valid.shape[1], rounding_mode="floor")
    flat_documents = document_ids.reshape(-1).index_select(0, byte_indices)

    def cu_seqlens(rows: Tensor, documents: Tensor) -> Tensor:
        boundary = torch.ones(documents.shape[0], dtype=torch.bool)
        boundary[1:] = (rows[1:] != rows[:-1]) | (
            documents[1:] != documents[:-1]
        )
        starts = torch.nonzero(boundary, as_tuple=False).flatten()
        lengths = torch.diff(
            torch.cat((starts, starts.new_tensor([documents.numel()])))
        )
        return torch.cat(
            (
                torch.zeros(1, dtype=torch.int32),
                lengths.to(torch.int32).cumsum(0, dtype=torch.int32),
            )
        )

    byte_cu = cu_seqlens(flat_rows, flat_documents)
    patch_valid = valid.view(valid.shape[0], -1, patch_stride).any(-1)
    patch_documents = document_ids.view(
        document_ids.shape[0], -1, patch_stride
    ).amax(-1)
    patch_indices = torch.nonzero(
        patch_valid.reshape(-1), as_tuple=False
    ).flatten()
    patches_per_row = patch_valid.shape[1]
    patch_rows = torch.div(patch_indices, patches_per_row, rounding_mode="floor")
    packed_patch_documents = patch_documents.reshape(-1).index_select(
        0, patch_indices
    )
    physical_patch_cu = cu_seqlens(patch_rows, packed_patch_documents)
    packed_patch_positions = (
        positions[:, ::patch_stride]
        .div(patch_stride, rounding_mode="floor")
        .reshape(-1)
        .index_select(0, patch_indices)
    )

    # Fast BLT's first patch is a one-byte BOS patch. Insert its patch input in
    # the packed global sequence without changing the corpus byte stream. A
    # page that starts in the middle of a document deliberately receives no
    # synthetic BOS: its unavailable preceding global state makes physical
    # origin zero ineligible for diffusion.
    segment_starts = physical_patch_cu[:-1].to(torch.long)
    segment_lengths = torch.diff(physical_patch_cu).to(torch.long)
    begins_document = packed_patch_positions.index_select(0, segment_starts).eq(0)
    segment_ids = torch.repeat_interleave(
        torch.arange(segment_lengths.numel()), segment_lengths
    )
    bos_prefix = begins_document.to(torch.long).cumsum(0)
    physical_to_global_packed = (
        torch.arange(patch_indices.numel(), dtype=torch.long)
        + bos_prefix.index_select(0, segment_ids)
    )
    total_global_patches = int(
        patch_indices.numel() + begins_document.sum()
    )
    global_patch_sources = torch.full(
        (total_global_patches,), -1, dtype=torch.long
    )
    global_patch_sources[physical_to_global_packed] = torch.arange(
        patch_indices.numel(), dtype=torch.long
    )
    global_patch_positions = torch.zeros(total_global_patches, dtype=torch.long)
    global_patch_positions[physical_to_global_packed] = (
        packed_patch_positions
        + begins_document.index_select(0, segment_ids).to(torch.long)
    )
    segment_global_lengths = segment_lengths + begins_document.to(torch.long)
    global_patch_cu = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32),
            segment_global_lengths.to(torch.int32).cumsum(0, dtype=torch.int32),
        )
    )
    bos_condition_indices = (
        segment_starts
        + torch.cat(
            (
                torch.zeros(1, dtype=torch.long),
                bos_prefix[:-1],
            )
        )
    )[begins_document]

    within_segment = torch.arange(patch_indices.numel()) - torch.repeat_interleave(
        segment_starts, segment_lengths
    )
    prior_packed = torch.where(
        within_segment.gt(0),
        physical_to_global_packed - 1,
        torch.where(
            begins_document.index_select(0, segment_ids),
            physical_to_global_packed - 1,
            -1,
        ),
    )
    prior_by_physical = torch.full(
        (patch_valid.numel(),), -1, dtype=torch.long
    )
    prior_by_physical[patch_indices] = prior_packed

    # Map every packed byte to either its current patch (the patch-final-byte
    # exception) or the explicit previous same-document global state. The
    # latter is the virtual BOS state at true document starts and -1 at a
    # contextless continuation-page boundary.
    physical_to_packed = torch.full(
        (patch_valid.numel(),), -1, dtype=torch.long
    )
    physical_to_packed[patch_indices] = torch.arange(
        patch_indices.numel(), dtype=torch.long
    )
    byte_columns = byte_indices.remainder(valid.shape[1])
    physical_patches = flat_rows * patches_per_row + torch.div(
        byte_columns, patch_stride, rounding_mode="floor"
    )
    packed_positions = positions.reshape(-1).index_select(0, byte_indices)
    current_packed = physical_to_packed.index_select(0, physical_patches)
    current_global = physical_to_global_packed.index_select(0, current_packed)
    prior_global = prior_by_physical.index_select(0, physical_patches)
    condition_patch_indices = torch.where(
        packed_positions.remainder(patch_stride).eq(patch_stride - 1),
        current_global,
        prior_global,
    )
    return (
        byte_indices,
        byte_cu,
        patch_indices,
        global_patch_cu,
        condition_patch_indices,
        global_patch_sources,
        global_patch_positions,
        physical_to_global_packed,
        bos_condition_indices,
        prior_by_physical.view_as(patch_valid),
    )


def chunks_to_batch(
    chunks: Sequence[PackedChunk], *, dense_stream: bool = False
) -> TrainingBatch:
    if not chunks:
        raise ValueError("cannot collate an empty chunk batch")
    width = len(chunks[0].input_ids)
    if any(len(chunk.input_ids) != width for chunk in chunks):
        raise ValueError("training chunks must have one fixed width")
    ids = torch.tensor([chunk.input_ids for chunk in chunks], dtype=torch.long)
    valid = torch.tensor([chunk.valid_mask for chunk in chunks], dtype=torch.bool)
    score = torch.tensor([chunk.score_mask for chunk in chunks], dtype=torch.bool)
    stored_targets = torch.tensor(
        [chunk.target_ids for chunk in chunks], dtype=torch.long
    )
    ar_targets = torch.where(score, stored_targets, IGNORE_INDEX)
    positions = torch.tensor(
        [chunk.document_offsets for chunk in chunks], dtype=torch.long
    ).clamp_min(0)
    if dense_stream:
        positions = torch.arange(width, dtype=torch.long)[None].expand(
            len(chunks), -1
        )
    document_ids = torch.tensor(
        [chunk.document_indices for chunk in chunks], dtype=torch.long
    )
    patch_offsets = torch.tensor(
        [chunk.patch_offsets for chunk in chunks], dtype=torch.long
    )
    bos_targets: list[int] = []
    bos_row_indices: list[int] = []
    for row, chunk in enumerate(chunks):
        if dense_stream:
            if chunk.stream_start == 0:
                first_valid = next(
                    index
                    for index, is_valid in enumerate(chunk.valid_mask)
                    if is_valid
                )
                bos_targets.append(chunk.input_ids[first_valid])
                bos_row_indices.append(row)
            continue
        for column, (is_valid, document_offset) in enumerate(
            zip(chunk.valid_mask, chunk.document_offsets, strict=True)
        ):
            if is_valid and document_offset == 0:
                bos_targets.append(chunk.input_ids[column])
                bos_row_indices.append(row)
    layout = (
        (None,) * 10
        if dense_stream
        else _packed_document_layout(valid, document_ids, positions)
    )
    return TrainingBatch(
        ids=ids,
        valid=valid,
        ar_targets=ar_targets,
        bos_targets=torch.tensor(bos_targets, dtype=torch.long),
        positions=positions,
        full_valid=bool(valid.all()) if dense_stream else False,
        document_ids=document_ids,
        bos_row_indices=torch.tensor(bos_row_indices, dtype=torch.long),
        isolate_documents=not dense_stream,
        byte_indices=layout[0],
        byte_cu_seqlens=layout[1],
        patch_indices=layout[2],
        patch_cu_seqlens=layout[3],
        condition_patch_indices=layout[4],
        global_patch_sources=layout[5],
        global_patch_positions=layout[6],
        physical_to_global_patch_indices=layout[7],
        bos_condition_indices=layout[8],
        prior_condition_indices=layout[9],
        patch_offsets=patch_offsets,
    )


def chunks_supported_by_current_model(
    chunks: Iterable[PackedChunk], *, require_single_document: bool
) -> tuple[PackedChunk, ...]:
    """Keep contiguous-prefix rows and reject cross-document diffusion rows."""

    supported: list[PackedChunk] = []
    for chunk in chunks:
        valid = torch.tensor(chunk.valid_mask)
        if bool(((~valid[:-1]) & valid[1:]).any()):
            continue
        document_ids = {
            index for index in chunk.document_indices if index >= 0
        }
        if require_single_document and len(document_ids) != 1:
            continue
        supported.append(chunk)
    if not supported:
        raise ValueError("no packed chunks satisfy the current model contract")
    return tuple(supported)


def take_distributed_chunks(
    cursor: DeterministicChunkCursor,
    local_count: int,
    context: DistributedContext,
) -> list[PackedChunk]:
    """Advance every rank through the same global order and select its shard."""

    selected: list[PackedChunk] = []
    for global_offset in range(local_count * context.world_size):
        index = cursor.next_index()
        if global_offset % context.world_size == context.rank:
            selected.append(cursor.chunk_at(index))
    if len(selected) != local_count:
        raise AssertionError("distributed chunk sharding produced the wrong size")
    return selected


def take_distributed_batch(
    cursor: DeterministicChunkCursor,
    local_count: int,
    context: DistributedContext,
) -> TrainingBatch:
    """Advance the global order and use native tensor collation when available."""

    global_indices = cursor.next_indices(local_count * context.world_size)
    indices = global_indices[context.rank :: context.world_size]
    if indices.size != local_count:
        raise AssertionError("distributed batch sharding produced the wrong size")
    native = getattr(cursor.chunks, "training_batch", None)
    if native is not None:
        return native(indices)
    return chunks_to_batch([cursor.chunk_at(index) for index in indices])


def take_distributed_indices(
    cursor: DeterministicChunkCursor,
    local_count: int,
    context: DistributedContext,
) -> np.ndarray:
    """Advance one global sample order and return this rank's row indices."""

    global_indices = cursor.next_indices(local_count * context.world_size)
    indices = global_indices[context.rank :: context.world_size]
    if indices.size != local_count:
        raise AssertionError("distributed row sharding produced the wrong size")
    return indices


def distributed_local_counts(global_count: int, world_size: int) -> tuple[int, ...]:
    """Return the exact, maximally balanced number of rows assigned per rank."""

    if global_count <= 0 or world_size <= 0:
        raise ValueError("global count and world size must be positive")
    quotient, remainder = divmod(global_count, world_size)
    return tuple(
        quotient + int(rank < remainder) for rank in range(world_size)
    )


def take_uneven_distributed_indices(
    cursor: DeterministicChunkCursor,
    global_count: int,
    context: DistributedContext,
    *,
    rotation: int = 0,
) -> np.ndarray:
    """Shard an exact global row count while keeping every cursor identical.

    The remainder rotates between ranks across updates.  In particular, 249
    pages on eight ranks assigns 32 pages to one rank and 31 to the other
    seven, without padding or silently increasing exposure to 256 pages.
    """

    if not 0 <= context.rank < context.world_size:
        raise ValueError("distributed rank is outside WORLD_SIZE")
    counts = distributed_local_counts(global_count, context.world_size)
    rotation %= context.world_size
    global_indices = cursor.next_indices(global_count)
    global_offsets = np.arange(global_count, dtype=np.int64)
    selected = global_indices[
        (global_offsets + rotation) % context.world_size == context.rank
    ]
    expected = counts[(context.rank - rotation) % context.world_size]
    if selected.size != expected:
        raise AssertionError("uneven distributed row sharding produced the wrong size")
    return selected


def sample_nonoverlapping_patch_starts(
    valid: Tensor,
    *,
    span_length: int,
    count: int,
    patch_stride: int,
    generator: torch.Generator,
) -> Tensor:
    """Sample nonoverlapping aligned spans from one valid-prefix document row."""

    if valid.dtype != torch.bool or valid.ndim != 2:
        raise ValueError("valid must be a rank-2 boolean tensor")
    if span_length <= 0 or span_length % patch_stride:
        raise ValueError("span length must be a positive patch multiple")
    if count <= 0:
        raise ValueError("span count must be positive")
    if count * span_length > valid.shape[1]:
        raise ValueError("physical row cannot store the requested branch bank")
    if not valid.is_cuda and bool(((~valid[:, :-1]) & valid[:, 1:]).any()):
        raise ValueError("branch sampling requires one valid prefix per row")
    span_patches = span_length // patch_stride
    # Only starts whose complete span is valid participate in random sampling.
    # Rows too short for all requested branches fall back to deterministic
    # consecutive spans; the final span may be partial and all later spans are
    # PAD-only. This retains every document tail for AR and short-block
    # diffusion without ever choosing a PAD-crossing span when a full one exists.
    physical_patches = torch.div(valid.sum(1), patch_stride, rounding_mode="floor")
    compressed_counts = physical_patches - count * span_patches + count
    # Draw a uniform subset without replacement for every row in one device
    # operation.  Invalid tail candidates receive +inf and cannot be selected.
    max_candidates = valid.shape[1] // patch_stride - count * span_patches + count
    candidate = torch.arange(max_candidates, device=valid.device)
    scores = torch.rand(
        (valid.shape[0], max_candidates),
        device=valid.device,
        generator=generator,
        dtype=torch.float32,
    ).masked_fill(candidate[None] >= compressed_counts[:, None], torch.inf)
    compressed = scores.topk(count, dim=1, largest=False, sorted=False).indices
    compressed = compressed.sort(1).values
    separation = torch.arange(count, device=valid.device) * (span_patches - 1)
    sampled = (compressed + separation) * patch_stride
    fallback = (
        torch.arange(count, device=valid.device)[None] * span_length
    ).expand(valid.shape[0], -1)
    return torch.where((compressed_counts >= count)[:, None], sampled, fallback)


def _gather_spans(
    ids: Tensor,
    starts: Tensor,
    span_length: int,
    *,
    fill_value: int | bool | None = None,
) -> Tensor:
    offsets = torch.arange(span_length, device=ids.device)
    indices = starts[:, :, None] + offsets[None, None, :]
    in_range = indices < ids.shape[1]
    gathered = ids[:, None, :].expand(-1, starts.shape[1], -1).gather(
        2, indices.clamp_max(ids.shape[1] - 1)
    )
    if fill_value is None:
        if (
            not ids.is_cuda
            and not torch.compiler.is_compiling()
            and not bool(in_range.all())
        ):
            raise ValueError("span lies outside physical row storage")
        return gathered
    return torch.where(in_range, gathered, torch.as_tensor(fill_value, device=ids.device))


def sample_blt_patch_starts(
    valid: Tensor,
    *,
    count: int,
    patch_stride: int,
    generator: torch.Generator,
    document_ids: Tensor,
    clean_ids: Tensor,
    prior_condition_indices: Tensor,
    block_length: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample document-local starts and explicit packed prior indices."""

    if valid.dtype != torch.bool or valid.ndim != 2 or count <= 0:
        raise ValueError("BLT start sampling requires boolean rows and count > 0")
    if (
        document_ids.shape != valid.shape
        or document_ids.dtype != torch.long
        or clean_ids.shape != valid.shape
    ):
        raise ValueError("clean ids and document ids must align with BLT validity")
    if block_length <= 0:
        raise ValueError("document-aware BLT sampling needs a positive block")
    expected_prior_shape = (valid.shape[0], valid.shape[1] // patch_stride)
    if (
        prior_condition_indices.shape != expected_prior_shape
        or prior_condition_indices.dtype != torch.long
    ):
        raise ValueError("prior conditions must align with physical patches")
    patch_starts = torch.arange(
        0, valid.shape[1], patch_stride, device=valid.device
    )
    if patch_starts.numel() < count:
        padding = count - patch_starts.numel()
        patch_starts = torch.cat(
            (patch_starts, torch.zeros(padding, device=valid.device, dtype=torch.long))
        )
        real_candidates = torch.arange(
            patch_starts.numel(), device=valid.device
        ) < patch_starts.numel() - padding
    else:
        real_candidates = torch.ones_like(patch_starts, dtype=torch.bool)
    start_docs = document_ids[:, patch_starts]
    candidate_priors = prior_condition_indices[:, patch_starts // patch_stride]
    positions = torch.arange(valid.shape[1], device=valid.device)
    last_indices = torch.where(valid, positions[None], -1).amax(1)
    last_ids = clean_ids.gather(1, last_indices[:, None]).squeeze(1)
    last_documents = document_ids.gather(
        1, last_indices[:, None]
    ).squeeze(1)
    terminates_in_storage = last_ids.eq(256)[:, None] & start_docs.eq(
        last_documents[:, None]
    )
    complete_block = (patch_starts + block_length <= valid.shape[1])[None]
    eligible_mask = (
        real_candidates[None]
        & (complete_block | terminates_in_storage)
        & valid[:, patch_starts]
        & start_docs.ge(0)
        & candidate_priors.ge(0)
    )
    eligible = eligible_mask.sum(1)
    selected_count = eligible.clamp_max(count)
    if count >= patch_starts.numel():
        # The dense estimator requests every physical patch origin.  Drawing
        # random scores, top-k sorting them, and sorting the selected origins
        # again is distributionally a no-op in this exhaustive case.  Keep
        # the already ordered candidate bank and mark its true per-row subset
        # directly.  This also makes exhaustive topology independent of the
        # sampling RNG, as it must be.
        candidates = patch_starts[:count][None].expand(valid.shape[0], -1)
        candidate_selected = eligible_mask[:, :count]
        # Stable linear-time compaction keeps active origins first without a
        # random top-k or comparison sort.  Active and inactive prefix ranks
        # form one exact permutation of every candidate column.
        active_rank = candidate_selected.to(torch.long).cumsum(1) - 1
        inactive_rank = (~candidate_selected).to(torch.long).cumsum(1) - 1
        destination = torch.where(
            candidate_selected,
            active_rank,
            selected_count[:, None] + inactive_rank,
        )
        starts = torch.zeros_like(candidates).scatter(1, destination, candidates)
        selected = torch.zeros_like(candidate_selected).scatter(
            1, destination, candidate_selected
        )
    else:
        # A random permutation followed by a prefix samples without
        # replacement.  Keep sampled origins in physical order so Q/K/V tiles
        # and sparse-mask metadata walk the shared clean bank monotonically.
        scores = torch.rand(
            (valid.shape[0], patch_starts.numel()),
            device=valid.device,
            generator=generator,
            dtype=torch.float32,
        ).masked_fill(~eligible_mask, torch.inf)
        chosen = scores.topk(count, dim=1, largest=False, sorted=True).indices
        starts = patch_starts[chosen]
        selected = torch.arange(count, device=valid.device)[None] < selected_count[:, None]
        starts = torch.where(selected, starts, torch.zeros_like(starts))
        physical_order = torch.where(
            selected, starts, torch.full_like(starts, valid.shape[1])
        ).argsort(dim=1)
        starts = torch.gather(starts, 1, physical_order)
        selected = torch.gather(selected, 1, physical_order)
    condition_indices = torch.gather(
        prior_condition_indices, 1, starts // patch_stride
    )
    condition_indices = torch.where(
        selected, condition_indices, torch.full_like(condition_indices, -1)
    )
    weight = eligible.to(torch.float32) / selected_count.clamp_min(1)
    return starts, weight, selected, condition_indices


def _variable_blt_candidate_matrices(
    batch: TrainingBatch,
    *,
    block_length: int,
    eot_id: int = 256,
) -> tuple[Tensor, Tensor, Tensor]:
    """Materialize the authenticated ragged patch population by physical row.

    Entropy patch starts are data, not an implicit fixed-stride arithmetic
    progression.  This converts the packed patch metadata to one dense CPU
    candidate bank per page while preserving physical order.  The bank is
    temporary topology metadata; model execution remains PAD-free and ragged.
    """

    metadata = (
        batch.physical_patch_row_indices,
        batch.physical_patch_start_columns,
        batch.physical_patch_prior_condition_indices,
    )
    if any(value is None for value in metadata):
        raise ValueError("variable BLT sampling requires physical patch metadata")
    if batch.document_ids is None:
        raise ValueError("variable BLT sampling requires document ids")
    rows, starts, priors = metadata
    assert rows is not None and starts is not None and priors is not None
    tensors = (batch.ids, batch.valid, batch.document_ids, rows, starts, priors)
    if any(tensor.device.type != "cpu" for tensor in tensors):
        raise ValueError("variable BLT topology must be prepared on CPU")
    if rows.ndim != 1 or starts.shape != rows.shape or priors.shape != rows.shape:
        raise ValueError("physical patch metadata must be aligned vectors")
    if any(tensor.dtype != torch.long for tensor in (rows, starts, priors)):
        raise TypeError("physical patch metadata must be int64")
    if block_length <= 0:
        raise ValueError("variable BLT block length must be positive")
    batch_size, width = batch.ids.shape
    if rows.numel() == 0 or bool(((rows < 0) | (rows >= batch_size)).any()):
        raise ValueError("physical patch rows are empty or out of range")
    if bool(((starts < 0) | (starts >= width)).any()):
        raise ValueError("physical patch starts are out of range")
    if bool((rows[1:] < rows[:-1]).any()):
        raise ValueError("physical patches are not grouped by row")
    same_row = rows[1:].eq(rows[:-1])
    if bool((same_row & starts[1:].le(starts[:-1])).any()):
        raise ValueError("physical patch starts are not strictly ordered")

    row_counts = torch.bincount(rows, minlength=batch_size)
    if bool(row_counts.eq(0).any()):
        raise ValueError("every physical row needs at least one patch")
    row_offsets = row_counts.cumsum(0) - row_counts
    slots = torch.arange(rows.numel()) - row_offsets.index_select(0, rows)
    candidate_width = int(row_counts.max())
    dense_starts = torch.zeros((batch_size, candidate_width), dtype=torch.long)
    dense_priors = torch.full_like(dense_starts, -1)
    present = torch.zeros_like(dense_starts, dtype=torch.bool)
    dense_starts[rows, slots] = starts
    dense_priors[rows, slots] = priors
    present[rows, slots] = True

    start_documents = batch.document_ids[rows, starts]
    offsets = torch.arange(block_length, dtype=torch.long)
    columns = starts[:, None] + offsets[None]
    in_storage = columns.lt(width)
    safe_columns = columns.clamp_max(width - 1)
    candidate_rows = rows[:, None].expand_as(safe_columns)
    suffix_valid = batch.valid[candidate_rows, safe_columns] & in_storage
    suffix_same_document = suffix_valid & batch.document_ids[
        candidate_rows, safe_columns
    ].eq(start_documents[:, None])
    complete_same_document = suffix_same_document.all(1)
    # A short block is semantically valid only when its contiguous suffix
    # reaches this document's EOT. Physical page width and PAD storage are not
    # evidence that a mid-document continuation has a complete target block.
    same_document_prefix = suffix_same_document.cumprod(1).to(torch.bool)
    terminates_at_eot = (
        same_document_prefix
        & batch.ids[candidate_rows, safe_columns].eq(eot_id)
    ).any(1)
    flat_eligible = (
        batch.valid[rows, starts]
        & start_documents.ge(0)
        & priors.ge(0)
        & (complete_same_document | terminates_at_eot)
    )
    eligible = torch.zeros_like(present)
    eligible[rows, slots] = flat_eligible
    return dense_starts, dense_priors, eligible


def _select_variable_blt_candidates(
    starts: Tensor,
    priors: Tensor,
    eligible: Tensor,
    *,
    count: int,
    scores: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Select a uniform origin subset and retain exact HT row weights."""

    if count <= 0:
        raise ValueError("variable BLT branch count must be positive")
    if (
        starts.ndim != 2
        or priors.shape != starts.shape
        or eligible.shape != starts.shape
    ):
        raise ValueError("variable BLT candidate matrices must align")
    if eligible.dtype != torch.bool:
        raise TypeError("variable BLT eligibility must be boolean")
    _, candidate_width = starts.shape
    eligible_count = eligible.sum(1)
    selected_count = eligible_count.clamp_max(count)
    if bool(selected_count.eq(0).any()):
        raise ValueError("every BLT row needs at least one eligible entropy patch")

    if count < candidate_width:
        if scores is None or scores.shape != starts.shape:
            raise ValueError("sampled variable BLT selection requires aligned scores")
        ranked_scores = scores.masked_fill(~eligible, torch.inf)
        chosen = ranked_scores.topk(
            count, dim=1, largest=False, sorted=True
        ).indices
        chosen_starts = starts.gather(1, chosen)
        chosen_priors = priors.gather(1, chosen)
        selected = eligible.gather(1, chosen)
        physical_order = torch.where(
            selected,
            chosen_starts,
            torch.full_like(chosen_starts, starts.shape[1]),
        ).argsort(dim=1, stable=True)
        chosen_starts = chosen_starts.gather(1, physical_order)
        chosen_priors = chosen_priors.gather(1, physical_order)
        selected = selected.gather(1, physical_order)
    else:
        # Stable compaction is a linear-time exhaustive path and consumes no
        # random numbers.  This makes the dense arm independent of an otherwise
        # distributionally irrelevant topology RNG draw.
        active_rank = eligible.to(torch.long).cumsum(1) - 1
        inactive = ~eligible
        inactive_rank = inactive.to(torch.long).cumsum(1) - 1
        destination = torch.where(
            eligible,
            active_rank,
            selected_count[:, None] + inactive_rank,
        )
        chosen_starts = torch.zeros_like(starts).scatter(1, destination, starts)
        chosen_priors = torch.full_like(priors, -1).scatter(
            1, destination, priors
        )
        selected = torch.zeros_like(eligible).scatter(1, destination, eligible)
        if count > candidate_width:
            padding = count - candidate_width
            chosen_starts = F.pad(chosen_starts, (0, padding))
            chosen_priors = F.pad(chosen_priors, (0, padding), value=-1)
            selected = F.pad(selected, (0, padding))
        else:
            chosen_starts = chosen_starts[:, :count]
            chosen_priors = chosen_priors[:, :count]
            selected = selected[:, :count]

    chosen_starts = torch.where(selected, chosen_starts, 0)
    chosen_priors = torch.where(selected, chosen_priors, -1)
    weight = eligible_count.to(torch.float32) / selected_count.to(torch.float32)
    return chosen_starts, weight, selected, chosen_priors


def sample_variable_blt_patch_starts(
    batch: TrainingBatch,
    *,
    count: int,
    block_length: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Uniformly sample authenticated entropy-patch origins without replacement."""

    if generator.device.type != "cpu":
        raise ValueError("variable BLT topology sampling requires a CPU generator")
    starts, priors, eligible = _variable_blt_candidate_matrices(
        batch, block_length=block_length
    )
    scores = (
        torch.rand(starts.shape, generator=generator, dtype=torch.float32)
        if count < starts.shape[1]
        else None
    )
    return _select_variable_blt_candidates(
        starts, priors, eligible, count=count, scores=scores
    )


def _blt_branch_valid(
    batch: TrainingBatch,
    starts: Tensor,
    selected: Tensor,
    *,
    block_length: int,
) -> Tensor:
    """Resolve exact same-document/PAD branch validity on CPU."""

    if batch.ids.device.type != "cpu" or starts.device.type != "cpu":
        raise ValueError("packed BLT branch topology must be resolved on CPU")
    if batch.document_ids is None:
        raise ValueError("packed BLT branch topology requires document ids")
    if starts.ndim != 2 or selected.shape != starts.shape:
        raise ValueError("BLT starts and selections must be aligned matrices")
    if selected.dtype != torch.bool:
        raise ValueError("BLT branch selections must be boolean")
    _, clean_length = batch.ids.shape
    branches = starts.shape[1]
    start_documents = torch.gather(batch.document_ids, 1, starts)

    branch_offsets = torch.arange(block_length)[None, None, :]
    branch_columns = starts[:, :, None] + branch_offsets
    branch_in_range = branch_columns.lt(clean_length)
    safe_branch_columns = branch_columns.clamp_max(clean_length - 1)
    expanded_valid = batch.valid[:, None, :].expand(-1, branches, -1)
    expanded_documents = batch.document_ids[:, None, :].expand(
        -1, branches, -1
    )
    branch_valid = (
        selected[:, :, None]
        & branch_in_range
        & torch.gather(expanded_valid, 2, safe_branch_columns)
        & torch.gather(expanded_documents, 2, safe_branch_columns).eq(
            start_documents[:, :, None]
        )
    )

    return branch_valid


def _packed_blt_branch_indices(
    batch: TrainingBatch,
    starts: Tensor,
    selected: Tensor,
    *,
    block_length: int,
    clean_window: int | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build the small duplicated-varlen benchmark/oracle topology."""

    branch_valid = _blt_branch_valid(
        batch, starts, selected, block_length=block_length
    )
    batch_size, clean_length = batch.ids.shape
    branches = starts.shape[1]
    total_length = clean_length + branches * block_length
    assert batch.document_ids is not None
    start_documents = torch.gather(batch.document_ids, 1, starts)
    expanded_valid = batch.valid[:, None, :].expand(-1, branches, -1)
    expanded_documents = batch.document_ids[:, None, :].expand(
        -1, branches, -1
    )
    branch_offsets = torch.arange(block_length)[None, None, :]
    prefix_width = clean_length if clean_window is None else clean_window
    prefix_offsets = torch.arange(prefix_width)[None, None, :]
    prefix_columns = starts[:, :, None] - prefix_width + prefix_offsets
    prefix_in_range = prefix_columns.ge(0)
    safe_prefix_columns = prefix_columns.clamp_min(0)
    prefix_valid = (
        selected[:, :, None]
        & prefix_in_range
        & torch.gather(expanded_valid, 2, safe_prefix_columns)
        & torch.gather(expanded_documents, 2, safe_prefix_columns).eq(
            start_documents[:, :, None]
        )
    )

    row_bases = (
        torch.arange(batch_size, dtype=torch.long)[:, None, None] * total_length
    )
    branch_bases = (
        clean_length
        + torch.arange(branches, dtype=torch.long)[None, :, None]
        * block_length
    )
    query_grid = row_bases + branch_bases + branch_offsets
    prefix_grid = row_bases + safe_prefix_columns
    query_indices = query_grid[branch_valid]
    kv_grid = torch.cat((prefix_grid, query_grid), dim=2)
    kv_valid = torch.cat((prefix_valid, branch_valid), dim=2)
    kv_indices = kv_grid[kv_valid]
    query_lengths = branch_valid.sum(-1).reshape(-1)
    kv_lengths = kv_valid.sum(-1).reshape(-1)
    active_branches = query_lengths.gt(0)
    query_lengths = query_lengths[active_branches].to(torch.int32)
    kv_lengths = kv_lengths[active_branches].to(torch.int32)

    def cumulative(lengths: Tensor) -> Tensor:
        return torch.cat(
            (lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32))
        )

    return (
        branch_valid,
        query_indices,
        kv_indices,
        cumulative(query_lengths),
        cumulative(kv_lengths),
    )


def prepare_blt_sampling(
    batch: TrainingBatch,
    config: CorruptionConfig,
    generator: torch.Generator,
    *,
    clean_window: int | None,
    branch_attention: Literal["shared_flex", "duplicated_varlen"] = "shared_flex",
) -> BltSamplingPlan:
    """Sample BLT origins and attention metadata entirely on CPU."""

    if batch.document_ids is None:
        raise ValueError("BLT sampling requires document-local prior metadata")
    if generator.device.type != "cpu":
        raise ValueError("BLT topology sampling requires a CPU generator")
    if batch.physical_patch_start_columns is not None:
        starts, weight, selected, condition_indices = (
            sample_variable_blt_patch_starts(
                batch,
                count=config.branches_per_row,
                block_length=config.canvas_length,
                generator=generator,
            )
        )
    else:
        if batch.prior_condition_indices is None:
            raise ValueError("fixed-stride BLT sampling omitted prior metadata")
        starts, weight, selected, condition_indices = sample_blt_patch_starts(
            batch.valid,
            count=config.branches_per_row,
            patch_stride=config.patch_stride,
            generator=generator,
            document_ids=batch.document_ids,
            clean_ids=batch.ids,
            prior_condition_indices=batch.prior_condition_indices,
            block_length=config.canvas_length,
        )
    branch_valid = _blt_branch_valid(
        batch, starts, selected, block_length=config.canvas_length
    )
    metadata = None
    query = kv = cu_query = cu_kv = None
    if branch_attention == "shared_flex":
        branch_positions = _gather_spans(
            batch.positions, starts, config.canvas_length, fill_value=0
        )
        assert batch.document_ids is not None
        layout = CanvasBranchLayout(
            clean_valid=batch.valid,
            branch_valid=branch_valid,
            prefix_lengths=starts,
            prefix_window=clean_window,
            # Positions also identify the start of the current document. Keep
            # them for an unbounded prefix so sparse metadata can prune prior
            # documents instead of relying on the token mask to reject them.
            clean_positions=batch.positions,
            branch_positions=branch_positions,
            clean_segment_ids=batch.document_ids,
            branch_segment_ids=torch.gather(batch.document_ids, 1, starts),
        )
        metadata = canvas_block_mask_metadata(layout)
    else:
        branch_valid, query, kv, cu_query, cu_kv = _packed_blt_branch_indices(
            batch,
            starts,
            selected,
            block_length=config.canvas_length,
            clean_window=clean_window,
        )
    return BltSamplingPlan(
        starts,
        weight,
        selected,
        condition_indices,
        branch_valid,
        metadata,
        query,
        kv,
        cu_query,
        cu_kv,
    )


def sample_validation_starts(
    valid: Tensor,
    chunks: Sequence[PackedChunk | ValidationChunkIdentity],
    *,
    span_length: int,
    count: int,
    patch_stride: int,
    seed: int,
) -> Tensor:
    """Sample per-chunk plans invariant to batching, rank, and world size."""

    if valid.shape[0] != len(chunks):
        raise ValueError("validation chunks and validity rows must align")
    rows: list[Tensor] = []
    for row, chunk in enumerate(chunks):
        identity = (
            seed
            + 1_000_003 * int(chunk.chunk_index)
            + 97_409 * int(chunk.stream_start)
        ) % (2**63 - 1)
        generator = torch.Generator(device=valid.device).manual_seed(identity)
        rows.append(
            sample_nonoverlapping_patch_starts(
                valid[row : row + 1],
                span_length=span_length,
                count=count,
                patch_stride=patch_stride,
                generator=generator,
            )
        )
    return torch.cat(rows)


def sample_validation_blt_starts(
    batch: TrainingBatch,
    chunks: Sequence[PackedChunk | ValidationChunkIdentity],
    *,
    block_length: int,
    count: int,
    patch_stride: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Document-local BLT plans invariant to validation batch composition."""

    if batch.document_ids is None or batch.ids.shape[0] != len(chunks):
        raise ValueError("BLT validation requires aligned document metadata")
    if batch.valid.device.type != "cpu":
        raise ValueError("validation BLT topology must be prepared on CPU")
    if count <= 0 or patch_stride <= 0 or block_length <= 0:
        raise ValueError("validation BLT dimensions must be positive")
    if batch.physical_patch_start_columns is not None:
        candidate_starts, candidate_priors, eligible_tensor = (
            _variable_blt_candidate_matrices(batch, block_length=block_length)
        )
        chunk_indices = np.fromiter(
            (int(chunk.chunk_index) for chunk in chunks),
            dtype=np.uint64,
            count=len(chunks),
        )
        stream_starts = np.fromiter(
            (int(chunk.stream_start) for chunk in chunks),
            dtype=np.uint64,
            count=len(chunks),
        )
        row_keys = (
            np.uint64(seed)
            ^ (chunk_indices * np.uint64(0xD2B74407B1CE6E93))
            ^ (stream_starts * np.uint64(0xCA5A826395121157))
        )
        columns = np.arange(candidate_starts.shape[1], dtype=np.uint64)[None]
        priorities = row_keys[:, None] + columns * np.uint64(0x9E3779B97F4A7C15)
        priorities ^= priorities >> np.uint64(30)
        priorities *= np.uint64(0xBF58476D1CE4E5B9)
        priorities ^= priorities >> np.uint64(27)
        priorities *= np.uint64(0x94D049BB133111EB)
        priorities ^= priorities >> np.uint64(31)
        eligible = eligible_tensor.numpy()
        priorities = np.where(eligible, priorities, np.iinfo(np.uint64).max)
        order = np.argsort(priorities, axis=1, kind="stable")
        ranks = np.empty_like(order)
        np.put_along_axis(
            ranks,
            order,
            np.broadcast_to(np.arange(order.shape[1]), order.shape),
            axis=1,
        )
        # Exact integer priority ranks fit losslessly in float64 and let the
        # shared selector retain its invalid-candidate handling.
        scores = torch.from_numpy(ranks.astype(np.float64))
        return _select_variable_blt_candidates(
            candidate_starts,
            candidate_priors,
            eligible_tensor,
            count=count,
            scores=scores,
        )
    if batch.prior_condition_indices is None:
        raise ValueError("fixed-stride BLT validation omitted prior metadata")
    width = batch.valid.shape[1]
    candidate_count = max(count, -(-width // patch_stride))
    patch_starts = np.zeros(candidate_count, dtype=np.int64)
    real_candidates = np.arange(candidate_count) < -(-width // patch_stride)
    patch_starts[real_candidates] = (
        np.arange(int(real_candidates.sum()), dtype=np.int64) * patch_stride
    )
    valid = batch.valid.numpy()
    clean_ids = batch.ids.numpy()
    documents = batch.document_ids.numpy()
    priors = batch.prior_condition_indices.numpy()
    last_indices = np.where(
        valid, np.arange(width, dtype=np.int64)[None], -1
    ).max(axis=1)
    last_ids = np.take_along_axis(
        clean_ids, last_indices[:, None], axis=1
    ).squeeze(1)
    last_documents = np.take_along_axis(
        documents, last_indices[:, None], axis=1
    ).squeeze(1)
    start_documents = documents[:, patch_starts]
    terminates_in_storage = (
        (last_ids == 256)[:, None]
        & (start_documents == last_documents[:, None])
    )
    complete_block = (patch_starts + block_length <= width)[None]
    eligible = (
        real_candidates[None]
        & (complete_block | terminates_in_storage)
        & valid[:, patch_starts]
        & (documents[:, patch_starts] >= 0)
        & (priors[:, patch_starts // patch_stride] >= 0)
    )

    # SplitMix64 gives every row/candidate a stateless pseudorandom priority.
    # Unlike one Python Generator/top-k call per row, this is one vectorized
    # operation and remains invariant to batching, rank, and world size.
    chunk_indices = np.fromiter(
        (int(chunk.chunk_index) for chunk in chunks),
        dtype=np.uint64,
        count=len(chunks),
    )
    stream_starts = np.fromiter(
        (int(chunk.stream_start) for chunk in chunks),
        dtype=np.uint64,
        count=len(chunks),
    )
    row_keys = (
        np.uint64(seed)
        ^ (chunk_indices * np.uint64(0xD2B74407B1CE6E93))
        ^ (stream_starts * np.uint64(0xCA5A826395121157))
    )
    priorities = (
        row_keys[:, None]
        + np.arange(candidate_count, dtype=np.uint64)[None]
        * np.uint64(0x9E3779B97F4A7C15)
    )
    priorities ^= priorities >> np.uint64(30)
    priorities *= np.uint64(0xBF58476D1CE4E5B9)
    priorities ^= priorities >> np.uint64(27)
    priorities *= np.uint64(0x94D049BB133111EB)
    priorities ^= priorities >> np.uint64(31)
    priorities = np.where(eligible, priorities, np.iinfo(np.uint64).max)
    chosen = np.argpartition(priorities, count - 1, axis=1)[:, :count]
    chosen_starts = patch_starts[chosen]
    selected_count = eligible.sum(axis=1).clip(max=count)
    selected = np.arange(count)[None] < selected_count[:, None]
    chosen_starts = np.where(selected, chosen_starts, 0)
    order = np.argsort(
        np.where(selected, chosen_starts, width), axis=1, kind="stable"
    )
    chosen_starts = np.take_along_axis(chosen_starts, order, axis=1)
    selected = np.take_along_axis(selected, order, axis=1)
    condition_indices = np.take_along_axis(
        priors, chosen_starts // patch_stride, axis=1
    )
    condition_indices = np.where(selected, condition_indices, -1)
    sampling_weight = eligible.sum(axis=1) / np.maximum(selected_count, 1)
    return (
        torch.from_numpy(chosen_starts),
        torch.from_numpy(sampling_weight.astype(np.float32)),
        torch.from_numpy(selected),
        torch.from_numpy(condition_indices),
    )


def validation_exact_k_mask(
    eligible: Tensor,
    chunks: Sequence[PackedChunk | ValidationChunkIdentity],
    *,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Deterministic exact-K sample of Fast-BLT's absorbing ELBO.

    A row samples K uniformly from 1..S and a uniform K-subset. Therefore
    ``sum(masked CE)/(K/S)`` is an unbiased, finite-variance estimator of the
    continuous-time 1/t objective. Stateless priorities keep this validation
    draw invariant to microbatching, rank count, and earlier validation calls.
    """

    if eligible.device.type != "cpu" or eligible.dtype != torch.bool:
        raise ValueError("validation exact-K eligibility must be a CPU bool tensor")
    if eligible.ndim != 3 or eligible.shape[0] != len(chunks):
        raise ValueError("validation exact-K rows and chunk identities must align")
    shape = tuple(eligible.shape)
    flat = eligible.numpy().reshape(shape[0], -1)
    counts = flat.sum(axis=1, dtype=np.int64)
    if np.any(counts <= 0):
        raise ValueError("every diffusion validation row needs an eligible atom")

    chunk_indices = np.fromiter(
        (int(chunk.chunk_index) for chunk in chunks),
        dtype=np.uint64,
        count=len(chunks),
    )
    stream_starts = np.fromiter(
        (int(chunk.stream_start) for chunk in chunks),
        dtype=np.uint64,
        count=len(chunks),
    )
    row_keys = (
        np.uint64(seed)
        ^ (chunk_indices * np.uint64(0xD2B74407B1CE6E93))
        ^ (stream_starts * np.uint64(0xCA5A826395121157))
    )

    def splitmix64(values: np.ndarray) -> np.ndarray:
        mixed = values.copy()
        mixed ^= mixed >> np.uint64(30)
        mixed *= np.uint64(0xBF58476D1CE4E5B9)
        mixed ^= mixed >> np.uint64(27)
        mixed *= np.uint64(0x94D049BB133111EB)
        mixed ^= mixed >> np.uint64(31)
        return mixed

    k_hash = splitmix64(row_keys ^ np.uint64(0xA0761D6478BD642F))
    chosen_k = (k_hash % counts.astype(np.uint64)).astype(np.int64) + 1
    columns = np.arange(flat.shape[1], dtype=np.uint64)[None]
    priorities = splitmix64(
        row_keys[:, None] + columns * np.uint64(0x9E3779B97F4A7C15)
    )
    priorities = np.where(flat, priorities, np.iinfo(np.uint64).max)
    order = np.argsort(priorities, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks,
        order,
        np.broadcast_to(np.arange(flat.shape[1]), order.shape),
        axis=1,
    )
    active = flat & (ranks < chosen_k[:, None])
    t = chosen_k.astype(np.float32) / counts.astype(np.float32)
    return torch.from_numpy(active.reshape(shape)), torch.from_numpy(t)


def canvas_objective_units(valid: Tensor, config: CorruptionConfig) -> Tensor:
    """Count nonempty physical canvases under the short-row fallback layout."""

    spans = torch.div(
        valid.sum(1) + config.canvas_length - 1,
        config.canvas_length,
        rounding_mode="floor",
    )
    return spans.clamp_max(config.branches_per_row).sum()


def prepare_canvas_corruption(
    batch: TrainingBatch,
    config: CorruptionConfig,
    vocab: AtomicVocabulary,
    generator: torch.Generator,
) -> CanvasCorruptionPlan:
    starts = sample_nonoverlapping_patch_starts(
        batch.valid,
        span_length=config.canvas_length,
        count=config.branches_per_row,
        patch_stride=config.patch_stride,
        generator=generator,
    )
    clean = _gather_spans(batch.ids, starts, config.canvas_length)
    branch_valid = _gather_spans(batch.valid, starts, config.canvas_length)
    flat_clean = clean.flatten(0, 1)
    flat_valid = branch_valid.flatten(0, 1)
    corrupted = _corrupt(flat_clean, flat_valid, config, vocab, generator)
    shape = clean.shape
    return CanvasCorruptionPlan(
        clean_branches=clean,
        noisy_branches=corrupted.ids.view(shape),
        branch_valid=branch_valid,
        branch_starts=starts,
        active=corrupted.active.view(shape),
    )


def prepare_blt_corruption(
    batch: TrainingBatch,
    config: CorruptionConfig,
    vocab: AtomicVocabulary,
    generator: torch.Generator,
    *,
    sampling: BltSamplingPlan | None = None,
) -> BltCorruptionPlan:
    if batch.document_ids is None:
        raise ValueError(
            "BLT corruption requires document-local explicit prior metadata"
        )
    block_length = config.canvas_length
    blocks = config.branches_per_row
    if sampling is None:
        if batch.physical_patch_start_columns is not None:
            starts, sampling_weight, selected, condition_indices = (
                sample_variable_blt_patch_starts(
                    batch,
                    count=blocks,
                    block_length=block_length,
                    generator=generator,
                )
            )
        else:
            if batch.prior_condition_indices is None:
                raise ValueError("fixed-stride BLT corruption omitted prior metadata")
            starts, sampling_weight, selected, condition_indices = (
                sample_blt_patch_starts(
                    batch.valid,
                    count=blocks,
                    patch_stride=config.patch_stride,
                    generator=generator,
                    document_ids=batch.document_ids,
                    clean_ids=batch.ids,
                    prior_condition_indices=batch.prior_condition_indices,
                    block_length=block_length,
                )
            )
        sampled_branch_valid = None
    else:
        expected = (batch.ids.shape[0], blocks)
        if sampling.block_starts.shape != expected:
            raise ValueError("precomputed BLT sampling plan does not align with batch")
        starts = sampling.block_starts
        sampling_weight = sampling.sampling_weight
        selected = sampling.selected
        condition_indices = sampling.condition_indices
        sampled_branch_valid = sampling.branch_valid
    clean_blocks = _gather_spans(
        batch.ids,
        starts,
        block_length,
        fill_value=vocab.pad_id,
    )
    block_valid = _gather_spans(
        batch.valid, starts, block_length, fill_value=False
    )
    block_documents = _gather_spans(
        batch.document_ids, starts, block_length, fill_value=-1
    )
    start_documents = torch.gather(batch.document_ids, 1, starts)
    block_valid &= block_documents.eq(start_documents[:, :, None])
    block_valid &= selected[:, :, None]
    if sampled_branch_valid is not None:
        if sampled_branch_valid.shape != block_valid.shape:
            raise ValueError("precomputed BLT validity does not align with blocks")
        if batch.ids.device.type == "cpu" and not torch.equal(
            sampled_branch_valid, block_valid
        ):
            raise ValueError("precomputed BLT topology disagrees with batch metadata")
    clean_blocks = torch.where(block_valid, clean_blocks, vocab.pad_id)
    flattened = clean_blocks.flatten(1, 2)
    flattened_valid = block_valid.flatten(1, 2)
    corrupted = _corrupt(flattened, flattened_valid, config, vocab, generator)
    corrupted_blocks = corrupted.ids.view_as(clean_blocks)
    active_blocks = corrupted.active.view_as(block_valid)
    return BltCorruptionPlan(
        clean_blocks,
        corrupted_blocks,
        block_valid,
        active_blocks,
        starts,
        condition_indices,
        block_length,
        corrupted.t,
        sampling_weight,
        None if sampling is None else sampling.branch_query_indices,
        None if sampling is None else sampling.branch_kv_indices,
        None if sampling is None else sampling.branch_query_cu_seqlens,
        None if sampling is None else sampling.branch_kv_cu_seqlens,
    )


def learning_rate_multiplier(
    update_index: int, iterations: int, warmdown_iters: int
) -> float:
    """Flat schedule followed by a linear final-update warmdown."""

    if not 0 <= update_index < iterations:
        raise ValueError("update_index must identify a planned update")
    if not 0 <= warmdown_iters <= iterations:
        raise ValueError("warmdown_iters must lie in [0, iterations]")
    if warmdown_iters == 0 or update_index < iterations - warmdown_iters:
        return 1.0
    return (iterations - update_index) / warmdown_iters


def zeropower_via_newtonschulz5(
    gradient: Tensor, *, steps: int, epsilon: float = 1e-7
) -> Tensor:
    """Latest nanoGPT's BF16 quintic orthogonalization."""

    if gradient.ndim != 2 or steps <= 0:
        raise ValueError("Muon requires a rank-two gradient and positive steps")
    a, b, c = 3.4445, -4.7750, 2.0315
    update = gradient.bfloat16()
    update = update / (update.norm() + epsilon)
    transposed = gradient.shape[0] > gradient.shape[1]
    if transposed:
        update = update.T
    for _ in range(steps):
        gram = update @ update.T
        correction = b * gram + c * gram @ gram
        update = a * update + correction @ update
    return update.T if transposed else update


# Newton--Schulz is a fixed chain of matrix and pointwise kernels.  Leaving the
# Python loop eager launches the chain independently for every Muon matrix on
# every update.  The wrapper is lazy (no compilation happens at import time)
# and CPU reference tests continue to use the plain implementation.
_compiled_zeropower_via_newtonschulz5 = torch.compile(
    zeropower_via_newtonschulz5,
    fullgraph=True,
)


class NanoGPTMuon(torch.optim.Optimizer):
    """Matrix optimizer matching the repository's current AR baseline."""

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        *,
        learning_rate: float,
        momentum: float,
        backend_steps: int,
    ) -> None:
        parameters = list(parameters)
        if not parameters or any(parameter.ndim != 2 for parameter in parameters):
            raise ValueError("nanoGPT Muon needs at least one rank-two parameter")
        super().__init__(
            parameters,
            {
                "lr": learning_rate,
                "base_lr": learning_rate,
                "momentum": momentum,
                "backend_steps": backend_steps,
            },
        )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            parameters = group["params"]
            total = sum(parameter.numel() for parameter in parameters)
            updates = torch.zeros(
                total,
                device=parameters[0].device,
                dtype=torch.bfloat16,
            )
            cursor = 0
            for index, parameter in enumerate(parameters):
                if index % world_size == rank and parameter.grad is not None:
                    state = self.state[parameter]
                    momentum_buffer = state.get("momentum_buffer")
                    if momentum_buffer is None:
                        momentum_buffer = torch.zeros_like(parameter)
                        state["momentum_buffer"] = momentum_buffer
                    momentum_buffer.mul_(group["momentum"]).add_(parameter.grad)
                    gradient = parameter.grad.add(
                        momentum_buffer, alpha=group["momentum"]
                    )
                    orthogonalize = (
                        _compiled_zeropower_via_newtonschulz5
                        if gradient.is_cuda
                        else zeropower_via_newtonschulz5
                    )
                    update = orthogonalize(
                        gradient, steps=group["backend_steps"]
                    )
                    update *= max(
                        1.0, parameter.shape[0] / parameter.shape[1]
                    ) ** 0.5
                    updates[cursor : cursor + parameter.numel()] = update.reshape(-1)
                cursor += parameter.numel()
            if distributed:
                dist.all_reduce(updates)
            cursor = 0
            for parameter in parameters:
                update = updates[cursor : cursor + parameter.numel()].view_as(parameter)
                parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
                cursor += parameter.numel()
        return loss


class OptimizerBundle:
    """Drive Adam and Muon groups through one checkpointable handle."""

    SCHEMA = "byte_diffusion_optimizer_bundle/v1"

    def __init__(
        self, optimizers: Sequence[torch.optim.Optimizer], *, kind: OptimizerKind
    ) -> None:
        if not optimizers:
            raise ValueError("optimizer bundle cannot be empty")
        self.optimizers = tuple(optimizers)
        self.kind = kind
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                group.setdefault("base_lr", group["lr"])

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [
            group
            for optimizer in self.optimizers
            for group in optimizer.param_groups
        ]

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "kind": self.kind,
            "optimizers": [
                optimizer.state_dict() for optimizer in self.optimizers
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != self.SCHEMA or state.get("kind") != self.kind:
            raise ValueError("optimizer bundle checkpoint contract mismatch")
        optimizer_states = state.get("optimizers")
        if not isinstance(optimizer_states, list) or len(optimizer_states) != len(
            self.optimizers
        ):
            raise ValueError("optimizer bundle checkpoint arity mismatch")
        for optimizer, optimizer_state in zip(
            self.optimizers, optimizer_states, strict=True
        ):
            optimizer.load_state_dict(optimizer_state)


def create_optimizer(
    parameters: Iterable[nn.Parameter],
    config: TrainingRunConfig,
    device: torch.device,
) -> torch.optim.AdamW:
    """Create fused AdamW on CUDA and the equation-identical CPU reference."""

    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )


def create_model_optimizer(
    model: ByteDiffusionModel,
    config: TrainingRunConfig,
    device: torch.device,
) -> OptimizerBundle:
    """Create the named optimizer cell without ambiguous parameter overlap."""

    if config.optimizer_kind == "adamw":
        optimizer = create_optimizer(model.parameters(), config, device)
        optimizer.param_groups[0]["base_lr"] = config.learning_rate
        optimizer.param_groups[0]["tag"] = "adamw"
        return OptimizerBundle([optimizer], kind="adamw")

    embedding_names = {
        "embedding.weight",
        "mode_embedding.weight",
        "ngrams.table.weight",
    }
    embedding_parameters: list[nn.Parameter] = []
    output_parameters: list[nn.Parameter] = []
    scalar_parameters: list[nn.Parameter] = []
    matrix_parameters: list[nn.Parameter] = []
    classified: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in embedding_names or name.startswith("ngrams.tables."):
            destination = embedding_parameters
        elif name == "output.weight":
            destination = output_parameters
        elif parameter.ndim < 2:
            destination = scalar_parameters
        elif parameter.ndim == 2:
            destination = matrix_parameters
        else:
            raise ValueError(
                f"nanoGPT optimizer has no rule for {name} shape {tuple(parameter.shape)}"
            )
        if id(parameter) in classified:
            raise AssertionError(f"optimizer parameter classified twice: {name}")
        classified.add(id(parameter))
        destination.append(parameter)
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if classified != expected:
        raise AssertionError("optimizer parameter classification is incomplete")
    adam_groups = [
        {
            "params": embedding_parameters,
            "lr": config.embedding_learning_rate,
            "base_lr": config.embedding_learning_rate,
            "tag": "embedding",
        },
        {
            "params": output_parameters,
            "lr": config.output_learning_rate,
            "base_lr": config.output_learning_rate,
            "tag": "output",
        },
        {
            "params": scalar_parameters,
            "lr": config.scalar_learning_rate,
            "base_lr": config.scalar_learning_rate,
            "tag": "scalar",
        },
    ]
    adam_groups = [group for group in adam_groups if group["params"]]
    if not adam_groups or not matrix_parameters:
        raise ValueError("nanoGPT optimizer requires both Adam and Muon groups")
    adam = torch.optim.Adam(
        adam_groups,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        fused=device.type == "cuda",
    )
    muon = NanoGPTMuon(
        matrix_parameters,
        learning_rate=config.matrix_learning_rate,
        momentum=config.muon_momentum_warmup_start,
        backend_steps=config.muon_backend_steps,
    )
    for group in muon.param_groups:
        group["tag"] = "matrix"
    return OptimizerBundle([adam, muon], kind="nanogpt_muon")


def attention_context(
    policy: AttentionPolicy,
    device: torch.device,
    *,
    allow_cpu_reference: bool,
):
    """Return an explicit backend context; never silently select dense CUDA."""

    if device.type == "cuda":
        if policy != "flash_sdpa":
            raise RuntimeError(
                "CUDA training requires BYTE_DIFFUSION_ATTENTION_BACKEND=flash_sdpa; "
                "dense fallback is forbidden"
            )
        return flash_sdpa_only()
    if policy != "dense_reference" or not allow_cpu_reference:
        raise RuntimeError(
            "CPU execution is a correctness reference and requires "
            "attention_policy='dense_reference' plus allow_cpu_reference=True"
        )
    return nullcontext()


class JointForward(nn.Module):
    """One DDP-visible forward containing both clean and noisy passes."""

    def __init__(self, model: ByteDiffusionModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        clean_ids: Tensor,
        valid: Tensor,
        positions: Tensor,
        noisy_ids: Tensor | None,
        diffusion_mode: int,
        noisy_valid: Tensor | None = None,
        starts: Tensor | None = None,
        block_length: int = 0,
        assume_full_clean: bool = False,
        packed_causal_logits: bool = False,
        bos_count: int | None = None,
        document_ids: Tensor | None = None,
        byte_indices: Tensor | None = None,
        byte_cu_seqlens: Tensor | None = None,
        patch_indices: Tensor | None = None,
        patch_cu_seqlens: Tensor | None = None,
        condition_patch_indices: Tensor | None = None,
        global_patch_sources: Tensor | None = None,
        global_patch_positions: Tensor | None = None,
        physical_to_global_patch_indices: Tensor | None = None,
        bos_condition_indices: Tensor | None = None,
        branch_condition_indices: Tensor | None = None,
        branch_query_indices: Tensor | None = None,
        branch_kv_indices: Tensor | None = None,
        branch_query_cu_seqlens: Tensor | None = None,
        branch_kv_cu_seqlens: Tensor | None = None,
        branch_block_mask=None,
        patch_byte_cu_seqlens: Tensor | None = None,
        max_patch_size: int | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        del block_length
        if bos_count is None:
            # Compatibility for direct callers. Training passes the exact
            # packed count so ordinary dense-stream batches do no synthetic
            # BOS decoder work at all.
            bos_count = clean_ids.shape[0]
        def bos_logits_from(states: Tensor | None) -> Tensor:
            if bos_count == 0:
                return self.model.embedding.weight.new_empty(
                    (0, self.model.config.vocab.output_size),
                    device=clean_ids.device,
                )
            if states is None:
                states = self.model.virtual_bos_global_states(
                    bos_count,
                    device=clean_ids.device,
                    allow_dense_reference=not clean_ids.is_cuda,
                )
            if states.shape[0] != bos_count:
                raise ValueError("virtual BOS states and targets must align")
            return self.model.forward_bos_logits(
                states,
                allow_dense_reference=not clean_ids.is_cuda,
            )
        if noisy_ids is None:
            output = self.model.forward_ar_varlen(
                clean_ids,
                valid,
                positions=positions,
                allow_dense_reference=not clean_ids.is_cuda,
                assume_full_clean=assume_full_clean,
                return_padded_logits=not packed_causal_logits,
                document_ids=document_ids,
                byte_indices=byte_indices,
                byte_cu_seqlens=byte_cu_seqlens,
                patch_indices=patch_indices,
                patch_cu_seqlens=patch_cu_seqlens,
                condition_patch_indices=condition_patch_indices,
                global_patch_sources=global_patch_sources,
                global_patch_positions=global_patch_positions,
                physical_to_global_patch_indices=physical_to_global_patch_indices,
                bos_condition_indices=bos_condition_indices,
                patch_byte_cu_seqlens=patch_byte_cu_seqlens,
                max_patch_size=max_patch_size,
            )
            return output.logits, None, bos_logits_from(output.bos_patch_states)
        if diffusion_mode == int(ModelMode.CANVAS):
            if noisy_valid is None or starts is None:
                raise ValueError("canvas forward requires branch validity and starts")
            output = self.model.forward_canvas_branches(
                clean_ids,
                valid,
                noisy_ids,
                noisy_valid,
                starts,
                positions=positions,
                assume_full_clean=assume_full_clean,
                document_ids=document_ids,
            )
            return output.clean_logits, output.branch_logits, bos_logits_from(None)
        if diffusion_mode == int(ModelMode.BLT_D):
            if noisy_valid is None or starts is None:
                raise ValueError("BLT-D forward requires block validity and starts")
            output = self.model.forward_blt_d_branches(
                clean_ids,
                valid,
                noisy_ids,
                noisy_valid,
                starts,
                positions=positions,
                assume_full_clean=assume_full_clean,
                document_ids=document_ids,
                byte_indices=byte_indices,
                byte_cu_seqlens=byte_cu_seqlens,
                patch_indices=patch_indices,
                patch_cu_seqlens=patch_cu_seqlens,
                condition_patch_indices=condition_patch_indices,
                global_patch_sources=global_patch_sources,
                global_patch_positions=global_patch_positions,
                physical_to_global_patch_indices=physical_to_global_patch_indices,
                bos_condition_indices=bos_condition_indices,
                branch_condition_indices=branch_condition_indices,
                branch_query_indices=branch_query_indices,
                branch_kv_indices=branch_kv_indices,
                branch_query_cu_seqlens=branch_query_cu_seqlens,
                branch_kv_cu_seqlens=branch_kv_cu_seqlens,
                branch_block_mask=branch_block_mask,
                return_clean_patch_states=not (
                    document_ids is not None
                    and self.model.config.decoder_branch_attention
                    == "shared_flex"
                ),
                patch_byte_cu_seqlens=patch_byte_cu_seqlens,
                max_patch_size=max_patch_size,
            )
            return (
                output.clean_logits,
                output.branch_logits,
                bos_logits_from(output.bos_patch_states),
            )
        else:
            raise ValueError("unknown diffusion mode")


def _corrupt(
    ids: Tensor,
    eligible: Tensor,
    config: CorruptionConfig,
    vocab: AtomicVocabulary,
    generator: torch.Generator,
) -> CorruptedBatch:
    if config.kind == "absorbing_rb":
        return absorbing_rb(ids, eligible, vocab, generator=generator)
    if config.kind == "allmask_50":
        return allmask_50(ids, eligible, vocab, generator=generator)
    if config.kind == "blt_bernoulli":
        return blt_bernoulli(ids, eligible, vocab, generator=generator)
    if config.kind == "blt_exact_k":
        return blt_exact_k(ids, eligible, vocab, generator=generator)
    if config.kind == "uniform_replacement":
        return uniform_replacement(ids, eligible, vocab, generator=generator)
    if config.kind in {"whole_patch", "contiguous_patch_span"}:
        return whole_patch(
            ids,
            eligible,
            vocab,
            patch_stride=config.patch_stride,
            generator=generator,
            contiguous=config.kind == "contiguous_patch_span",
        )
    raise ValueError(f"unsupported corruption kind {config.kind!r}")


class ByteDiffusionTrainer:
    """Update-boundary-exact scratch trainer."""

    def __init__(
        self,
        model: ByteDiffusionModel,
        train_cursor: DeterministicChunkCursor,
        validation_chunks: Sequence[PackedChunk],
        run_config: TrainingRunConfig,
        *,
        device: torch.device,
        distributed: DistributedContext = DistributedContext(),
        atomic_manifest: AtomicIdManifest | None = None,
        dataset_provenance: Mapping[str, Any] | None = None,
        source_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        if device.type == "cuda":
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(device)
        elif not run_config.allow_cpu_reference:
            raise RuntimeError("CPU trainer is available only as an explicit reference")
        if device.type == "cuda" and run_config.recipe == "canvas":
            raise ValueError(
                "the legacy canvas trainer is not safe for packed multi-document "
                "pages; use the standalone DiffusionGemma recipe"
            )
        self.model_config = model.config
        self.run_config = run_config
        self.train_cursor = train_cursor
        self.validation_chunks = validation_chunks
        dataset_patching = getattr(self.train_cursor.chunks, "patching", None)
        if run_config.patching_policy == "causal_entropy_v1":
            if (
                not isinstance(dataset_patching, DatasetPatchingSpec)
                or dataset_patching.name != "causal_entropy_v1"
                or not dataset_patching.artifact_sha256
            ):
                raise ValueError(
                    "entropy training requires authenticated patcher provenance"
                )
        lazy_validation_digest = getattr(validation_chunks, "dataset_sha256", None)
        self.validation_sha256 = (
            str(lazy_validation_digest)
            if lazy_validation_digest is not None
            else packed_chunks_sha256(validation_chunks)
        )
        self._validation_starts_cache: dict[
            tuple[tuple[int, int], ...], Tensor
        ] = {}
        self._validation_blt_plan_cache: dict[
            tuple[tuple[int, int], ...], BltSamplingPlan
        ] = {}
        self.device = device
        self.transfer_stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self.distributed = distributed
        self.atomic_manifest = atomic_manifest or AtomicIdManifest.reference()
        self.dataset_provenance = dict(dataset_provenance or {})
        self.source_provenance = dict(source_provenance or {})
        if (
            self.atomic_manifest.output_size != model.config.vocab.output_size
            or self.atomic_manifest.mask_id != model.config.vocab.mask_id
            or self.atomic_manifest.pad_id != model.config.vocab.pad_id
            or self.atomic_manifest.eot_id != model.config.vocab.eot_id
        ):
            raise ValueError("trainer atomic manifest does not match model vocabulary")
        if model.config.explicit_timestep or model.config.self_conditioning:
            raise ValueError(
                "the optimized shared-branch trainer does not silently ignore "
                "timestep/self-conditioning arms; use the standalone model "
                "ablation path until a branch-conditioned implementation is selected"
            )
        self.joint = JointForward(model).to(device)
        self.joint.model.activation_checkpointing = (
            run_config.activation_checkpointing and device.type == "cuda"
        )
        self.joint.model.require_compiled_training = (
            run_config.compile_model and device.type == "cuda"
        )
        forward_core: nn.Module = self.joint
        self.validation_model: nn.Module = self.joint
        if device.type == "cuda" and run_config.compile_model:
            # Inductor's dynamic scheduler currently fails on singleton batches
            # in both cumsum lowering and fused concat projections. A one-row
            # diagnostic run cannot have a smaller accumulation tail, so static
            # compilation loses no shape reuse. Production microbatches remain
            # dynamic to share graphs with their exact-global-batch tail.
            compile_dynamic = (
                run_config.compile_dynamic_shapes
                and run_config.microbatch_per_rank > 1
            )
            forward_core = torch.compile(
                forward_core,
                # Production must never silently fall back to eager between
                # decoder layers. Prepacked document/branch topology removes
                # the former data-dependent graph breaks, so fail closed if a
                # regression reintroduces one.
                fullgraph=True,
                dynamic=compile_dynamic,
                mode=(
                    "reduce-overhead"
                    if run_config.cuda_graphs
                    else run_config.compile_mode
                ),
            )
            # Grad-enabled and no-grad AOT graphs have different aliasing
            # contracts. Separate wrappers avoid an upstream Inductor view-
            # replay failure when validation compiles before the first update.
            self.validation_model = torch.compile(
                self.joint,
                fullgraph=True,
                dynamic=compile_dynamic,
                mode=run_config.compile_mode,
            )
        if dist.is_initialized():
            kwargs: dict[str, Any] = {}
            if device.type == "cuda":
                kwargs = {"device_ids": [device.index], "output_device": device.index}
            self.forward_model: nn.Module = DDP(forward_core, **kwargs)
        else:
            self.forward_model = forward_core
        self.optimizer = create_model_optimizer(
            self.joint.model, run_config, device
        )
        if run_config.cuda_graphs:
            if device.type != "cuda" or not run_config.compile_model:
                raise ValueError("CUDA graphs require CUDA plus compiled execution")
            for parameter in self.joint.parameters():
                parameter.grad = torch.zeros_like(parameter)
        self.corruption_generator = torch.Generator(device=device)
        self.corruption_generator.manual_seed(
            run_config.seed + 1_000_003 * distributed.rank
        )
        # Sampling BLT origins on CPU lets us resolve the complete varlen
        # attention topology before entering the compiled CUDA forward.  Keep
        # its RNG independent and checkpointed so resume remains exact.
        self.blt_sampling_generator = torch.Generator(device="cpu")
        self.blt_sampling_generator.manual_seed(
            run_config.seed + 31_415_927 + 1_000_003 * distributed.rank
        )
        self.completed_steps = 0
        self.training_time_ms = 0.0
        self._training_window_started_at: float | None = None

    def _set_learning_rate(self) -> float:
        multiplier = learning_rate_multiplier(
            self.completed_steps,
            self.run_config.iterations,
            self.run_config.warmdown_iters,
        )
        learning_rate = 0.0
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * multiplier
            learning_rate = max(learning_rate, float(group["lr"]))
            if "momentum" in group:
                warmup_steps = self.run_config.muon_momentum_warmup_steps
                progress = (
                    min(self.completed_steps / warmup_steps, 1.0)
                    if warmup_steps
                    else 1.0
                )
                group["momentum"] = (
                    (1.0 - progress)
                    * self.run_config.muon_momentum_warmup_start
                    + progress * self.run_config.muon_momentum
                )
        return learning_rate

    def measure_gradient_interference(
        self, chunks: Sequence[PackedChunk]
    ) -> GradientInterferenceMetrics:
        """Run the named fixed-batch AR/diffusion shared-trunk diagnostic."""

        if self.run_config.recipe == "causal_only":
            raise ValueError("gradient interference requires a diffusion objective")
        generator_state = self.corruption_generator.get_state().clone()
        sampling_generator_state = self.blt_sampling_generator.get_state().clone()
        self.corruption_generator.manual_seed(self.run_config.seed + 7_919)
        self.blt_sampling_generator.manual_seed(self.run_config.seed + 7_919)
        cpu_batch = chunks_to_batch(chunks)
        blt_sampling = (
            prepare_blt_sampling(
                cpu_batch,
                self.run_config.corruption,
                (
                    self.corruption_generator
                    if self.device.type == "cpu"
                    else self.blt_sampling_generator
                ),
                clean_window=self.model_config.decoder_prefix_window,
                branch_attention=self.model_config.decoder_branch_attention,
            )
            if self.run_config.recipe == "blt_d"
            else None
        )
        batch = cpu_batch.to(self.device)
        if blt_sampling is not None:
            blt_sampling = blt_sampling.to(self.device)
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        try:
            with attention_context(
                self.run_config.attention_policy,
                self.device,
                allow_cpu_reference=self.run_config.allow_cpu_reference,
            ), autocast:
                losses = self._compute_loss(batch, blt_sampling=blt_sampling)
            shared = tuple(self.joint.model.global_blocks.parameters())
            # Compare the actual additive optimizer components. In paper-sum
            # mode the clean term is a mean of per-row CE sums, not the
            # token-mean diagnostic exposed as ``losses.ar``.
            ar_component = losses.total - losses.diffusion
            metrics = gradient_interference_metrics(
                ar_component, losses.diffusion, shared
            )
        finally:
            self.corruption_generator.set_state(generator_state)
            self.blt_sampling_generator.set_state(sampling_generator_state)
        return metrics

    def _compute_loss(
        self,
        batch: TrainingBatch,
        *,
        blt_sampling: BltSamplingPlan | None = None,
        collect_diagnostics: bool = True,
    ) -> StepLoss:
        noisy_ids = noisy_valid = starts = None
        block_length = 0
        canvas_plan: CanvasCorruptionPlan | None = None
        blt_plan: BltCorruptionPlan | None = None
        if self.run_config.recipe == "causal_only":
            diffusion_mode = -1
        elif self.run_config.recipe == "canvas":
            canvas_plan = prepare_canvas_corruption(
                batch,
                self.run_config.corruption,
                self.model_config.vocab,
                self.corruption_generator,
            )
            noisy_ids = canvas_plan.noisy_branches
            noisy_valid = canvas_plan.branch_valid
            starts = canvas_plan.branch_starts
            diffusion_mode = int(ModelMode.CANVAS)
        else:
            if blt_sampling is None:
                if batch.ids.device.type != "cpu":
                    raise ValueError(
                        "CUDA BLT loss requires CPU-precomputed branch topology"
                    )
                blt_sampling = prepare_blt_sampling(
                    batch,
                    self.run_config.corruption,
                    (
                        self.corruption_generator
                        if self.device.type == "cpu"
                        else self.blt_sampling_generator
                    ),
                    clean_window=self.model_config.decoder_prefix_window,
                    branch_attention=self.model_config.decoder_branch_attention,
                )
            blt_plan = prepare_blt_corruption(
                batch,
                self.run_config.corruption,
                self.model_config.vocab,
                self.corruption_generator,
                sampling=blt_sampling,
            )
            noisy_ids = blt_plan.noisy_blocks
            noisy_valid = blt_plan.branch_valid
            starts = blt_plan.block_starts
            block_length = blt_plan.block_length
            diffusion_mode = int(ModelMode.BLT_D)

        branch_block_mask = None
        if (
            blt_plan is not None
            and self.model_config.decoder_branch_attention == "shared_flex"
            and batch.ids.is_cuda
        ):
            if blt_sampling is None or blt_sampling.block_mask_metadata is None:
                raise ValueError("shared Flex BLT requires precomputed sparse metadata")
            if batch.document_ids is None:
                raise ValueError("shared Flex BLT requires document ids")
            branch_positions = _gather_spans(
                batch.positions,
                blt_plan.block_starts,
                blt_plan.block_length,
                fill_value=0,
            )
            branch_layout = CanvasBranchLayout(
                clean_valid=batch.valid,
                branch_valid=blt_plan.branch_valid,
                prefix_lengths=blt_plan.block_starts,
                prefix_window=self.model_config.decoder_prefix_window,
                clean_positions=batch.positions,
                branch_positions=branch_positions,
                clean_segment_ids=batch.document_ids,
                branch_segment_ids=torch.gather(
                    batch.document_ids, 1, blt_plan.block_starts
                ),
            )
            branch_block_mask = build_canvas_block_mask(
                branch_layout, metadata=blt_sampling.block_mask_metadata
            )

        ar_logits_expanded, diffusion_logits, bos_logits = self.forward_model(
            batch.ids,
            batch.valid,
            batch.positions,
            noisy_ids,
            diffusion_mode,
            noisy_valid,
            starts,
            block_length,
            batch.full_valid and not batch.isolate_documents,
            self.run_config.recipe == "causal_only",
            batch.bos_targets.shape[0],
            (
                batch.document_ids
                if batch.isolate_documents
                and self.run_config.recipe != "causal_only"
                else None
            ),
            batch.byte_indices,
            batch.byte_cu_seqlens,
            batch.patch_indices,
            batch.patch_cu_seqlens,
            batch.condition_patch_indices,
            batch.global_patch_sources,
            batch.global_patch_positions,
            batch.physical_to_global_patch_indices,
            batch.bos_condition_indices,
            (
                blt_plan.condition_indices
                if blt_plan is not None
                else None
            ),
            None if blt_plan is None else blt_plan.branch_query_indices,
            None if blt_plan is None else blt_plan.branch_kv_indices,
            None if blt_plan is None else blt_plan.branch_query_cu_seqlens,
            None if blt_plan is None else blt_plan.branch_kv_cu_seqlens,
            branch_block_mask,
            **(
                {}
                if batch.patch_byte_cu_seqlens is None
                else {
                    "patch_byte_cu_seqlens": batch.patch_byte_cu_seqlens,
                    "max_patch_size": batch.max_patch_size,
                }
            ),
        )
        if batch.bos_targets.numel():
            bos_nll = F.cross_entropy(
                bos_logits,
                batch.bos_targets,
                reduction="none",
            )
        else:
            bos_nll = bos_logits.new_empty((0,))
        if self.run_config.recipe == "causal_only":
            packed_targets = pack_valid(batch.ar_targets, batch.valid)
            ar_total = F.cross_entropy(
                ar_logits_expanded,
                packed_targets,
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            ar_count = packed_targets.ne(IGNORE_INDEX).sum()
            ar_total = ar_total + bos_nll.sum()
            ar_count = ar_count + batch.bos_targets.numel()
            ar = ar_total / ar_count.clamp_min(1).to(ar_total.dtype)
        else:
            ar_rows = cross_entropy_per_row(ar_logits_expanded, batch.ar_targets)
            if batch.bos_row_indices is None:
                raise ValueError("mixed training requires BOS-to-row indices")
            bos_row_totals = ar_rows.total.new_zeros(ar_rows.total.shape)
            bos_row_counts = ar_rows.count.new_zeros(ar_rows.count.shape)
            bos_row_totals.scatter_add_(0, batch.bos_row_indices, bos_nll)
            bos_row_counts.scatter_add_(
                0,
                batch.bos_row_indices,
                torch.ones_like(batch.bos_row_indices, dtype=ar_rows.count.dtype),
            )
            ar_row_totals = ar_rows.total + bos_row_totals
            ar_row_counts = ar_rows.count + bos_row_counts
            ar_count = ar_row_counts.sum()
            ar = ar_row_totals.sum() / ar_count.clamp_min(1).to(
                ar_row_totals.dtype
            )
        if self.run_config.recipe == "causal_only":
            zero = ar.detach().new_zeros(())
            return StepLoss(
                total=self.run_config.lambda_ar * ar,
                ar=ar,
                diffusion=zero,
                ar_targets=ar_count,
                diffusion_targets=ar_count.new_zeros(()),
                noise_masked=ar_count.new_empty((0,)),
                noise_eligible=ar_count.new_empty((0,)),
                noise_nll=ar.new_empty((0,)),
                noise_correct=ar_count.new_empty((0,)),
            )
        if diffusion_logits is None:
            raise AssertionError("mixed objective omitted diffusion logits")
        if canvas_plan is not None:
            diffusion_targets = same_position_targets(
                canvas_plan.clean_branches.flatten(0, 1),
                canvas_plan.active.flatten(0, 1),
                output_size=self.model_config.vocab.output_size,
            ).view_as(canvas_plan.clean_branches)
            canvas_objective = canvas_cross_entropy(
                diffusion_logits, diffusion_targets, canvas_plan.active
            )
            diffusion = canvas_objective.loss
            diffusion_count = canvas_plan.active.sum()
            noise_masked = canvas_plan.active.flatten(0, 1).sum(1)
            noise_eligible = canvas_plan.branch_valid.flatten(0, 1).sum(1)
            noise_nll = canvas_objective.total.flatten()
            noise_correct = (
                (
                    diffusion_logits.argmax(-1).eq(diffusion_targets)
                    & canvas_plan.active
                )
                .flatten(0, 1)
                .sum(1)
                if collect_diagnostics
                else diffusion_count.new_empty((0,))
            )
        elif blt_plan is not None:
            diffusion_targets = same_position_targets(
                blt_plan.clean_blocks.flatten(0, 1),
                blt_plan.active.flatten(0, 1),
                output_size=self.model_config.vocab.output_size,
            ).view_as(blt_plan.clean_blocks)
            if self.run_config.corruption.kind in {"blt_bernoulli", "blt_exact_k"}:
                if blt_plan.t is None:
                    raise AssertionError("BLT corruption omitted t")
                blt_objective = blt_masked_loss(
                    diffusion_logits,
                    diffusion_targets,
                    blt_plan.active,
                    blt_plan.t,
                )
                diffusion = (
                    blt_objective.per_row
                    * (
                        blt_plan.sampling_weight
                        if blt_plan.sampling_weight is not None
                        else 1.0
                    )
                ).mean()
            else:
                diffusion = canvas_cross_entropy(
                    diffusion_logits, diffusion_targets, blt_plan.active
                ).loss
            diffusion_count = blt_plan.active.sum()
            noise_masked = blt_plan.active.flatten(0, 1).sum(1)
            noise_eligible = blt_plan.branch_valid.flatten(0, 1).sum(1)
            noise_nll = (
                blt_objective.group_total.flatten()
                if "blt_objective" in locals()
                else cross_entropy_per_row(
                    diffusion_logits, diffusion_targets, active=blt_plan.active
                ).total.flatten()
            )
            noise_correct = (
                (
                    diffusion_logits.argmax(-1).eq(diffusion_targets)
                    & blt_plan.active
                )
                .flatten(0, 1)
                .sum(1)
                if collect_diagnostics
                else diffusion_count.new_empty((0,))
            )
        else:
            raise AssertionError("mixed objective omitted its corruption plan")
        total_loss = diffusion + self.run_config.lambda_ar * ar
        if (
            blt_plan is not None
            and self.run_config.corruption.kind in {"blt_bernoulli", "blt_exact_k"}
            and self.run_config.objective_reduction
            in {"paper_sum", "row_normalized_sum"}
        ):
            # Fast-BLT equations 5--7: summed clean CE plus the 1/t-weighted
            # masked sum for each clean row, followed by one batch mean.
            if "blt_objective" not in locals():
                raise AssertionError("paper_sum omitted the BLT objective")
            combined_rows = (
                blt_objective.per_row
                * (
                    blt_plan.sampling_weight
                    if blt_plan.sampling_weight is not None
                    else 1.0
                )
                + self.run_config.lambda_ar * ar_row_totals
            )
            if self.run_config.objective_reduction == "row_normalized_sum":
                row_denominator = ar_row_counts.clamp_min(1).to(
                    combined_rows.dtype
                )
                total_loss = (combined_rows / row_denominator).mean()
                diffusion = (
                    blt_objective.per_row
                    * (
                        blt_plan.sampling_weight
                        if blt_plan.sampling_weight is not None
                        else 1.0
                    )
                    / row_denominator
                ).mean()
            else:
                total_loss = combined_rows.mean()
        return StepLoss(
            total=total_loss,
            ar=ar,
            diffusion=diffusion,
            ar_targets=ar_count,
            diffusion_targets=diffusion_count,
            noise_masked=noise_masked,
            noise_eligible=noise_eligible,
            noise_nll=noise_nll,
            noise_correct=noise_correct,
        )

    def run_update(self, *, materialize_metrics: bool = True) -> StepMetrics | None:
        if self.completed_steps >= self.run_config.iterations:
            raise RuntimeError("planned training updates are already complete")
        if self._training_window_started_at is None:
            self._training_window_started_at = time.perf_counter()
        learning_rate = self._set_learning_rate()
        self.optimizer.zero_grad(set_to_none=not self.run_config.cuda_graphs)
        if self.run_config.global_batch_size is None:
            local_rows = (
                self.run_config.microbatch_per_rank
                * self.run_config.gradient_accumulation
            )
            indices = take_distributed_indices(
                self.train_cursor, local_rows, self.distributed
            )
        else:
            indices = take_uneven_distributed_indices(
                self.train_cursor,
                self.run_config.global_batch_size,
                self.distributed,
                rotation=self.completed_steps,
            )
        native_batches = getattr(self.train_cursor.chunks, "training_batches", None)
        native_groups = getattr(
            self.train_cursor.chunks, "training_batch_groups", None
        )
        native_ar_units = getattr(
            self.train_cursor.chunks, "training_ar_units", None
        )
        stream_cpu_preparation = (
            self.device.type == "cuda"
            and native_groups is not None
            and native_ar_units is not None
        )
        batch_groups: list[tuple[int, ...]] | None = None
        if stream_cpu_preparation:
            batch_groups = native_groups(
                indices,
                max_batch_size=self.run_config.microbatch_per_rank,
                physical_token_budget=self.run_config.microbatch_token_budget,
                sort_by_length=self.run_config.length_sorted_microbatches,
            )
            cpu_batches: Sequence[TrainingBatch] = ()
        elif native_batches is None:
            cpu_batches = [
                chunks_to_batch(
                    [self.train_cursor.chunk_at(index) for index in indices[start:stop]]
                )
                for start in range(0, len(indices), self.run_config.microbatch_per_rank)
                for stop in (
                    min(start + self.run_config.microbatch_per_rank, len(indices)),
                )
            ]
        else:
            cpu_batches = native_batches(
                indices,
                max_batch_size=self.run_config.microbatch_per_rank,
                physical_token_budget=self.run_config.microbatch_token_budget,
                sort_by_length=self.run_config.length_sorted_microbatches,
            )
        microstep_count = (
            len(batch_groups) if batch_groups is not None else len(cpu_batches)
        )
        dataset_chunk_size = getattr(self.train_cursor.chunks, "chunk_size", None)
        dataset_branch_bytes = getattr(
            self.train_cursor.chunks, "required_branch_bytes", None
        )
        fixed_microbatch_bound = (
            dataset_chunk_size is not None
            and dataset_branch_bytes is not None
            and self.run_config.microbatch_per_rank
            * (
                int(dataset_chunk_size)
                + int(dataset_branch_bytes)
            )
            <= self.run_config.microbatch_token_budget
        )
        fixed_equal_microsteps = False
        if fixed_microbatch_bound:
            if self.run_config.global_batch_size is None:
                fixed_equal_microsteps = True
            else:
                rank_rows = np.asarray(
                    distributed_local_counts(
                        self.run_config.global_batch_size,
                        self.distributed.world_size,
                    ),
                    dtype=np.int64,
                )
                rank_microsteps = (
                    rank_rows + self.run_config.microbatch_per_rank - 1
                ) // self.run_config.microbatch_per_rank
                fixed_equal_microsteps = bool(
                    np.all(rank_microsteps == rank_microsteps[0])
                )
        if self.distributed.world_size > 1 and not fixed_equal_microsteps:
            # Adaptive row-width packing can produce a different backward-call
            # count per rank, which DDP cannot accept. Fixed production pages
            # satisfy the bound above by construction, so they avoid two
            # redundant collectives and host synchronizations every update.
            microsteps_min = torch.tensor(microstep_count, device=self.device)
            microsteps_max = microsteps_min.clone()
            dist.all_reduce(microsteps_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(microsteps_max, op=dist.ReduceOp.MAX)
            if int(microsteps_min) != int(microsteps_max):
                raise ValueError(
                    "adaptive batching gave ranks different backward-call counts: "
                    f"min={int(microsteps_min)}, max={int(microsteps_max)}"
                )
        cpu_blt_sampling = (
            [
                prepare_blt_sampling(
                    batch,
                    self.run_config.corruption,
                    (
                        self.corruption_generator
                        if self.device.type == "cpu"
                        else self.blt_sampling_generator
                    ),
                    clean_window=self.model_config.decoder_prefix_window,
                    branch_attention=self.model_config.decoder_branch_attention,
                )
                for batch in cpu_batches
            ]
            if self.run_config.recipe == "blt_d" and not stream_cpu_preparation
            else [None] * len(cpu_batches)
        )
        if self.device.type == "cuda" and not stream_cpu_preparation:
            cpu_batches = tuple(batch.pin_memory() for batch in cpu_batches)
            cpu_blt_sampling = tuple(
                None if sampling is None else sampling.pin_memory()
                for sampling in cpu_blt_sampling
            )
        local_ar_units = (
            int(native_ar_units(indices))
            if stream_cpu_preparation
            else sum(
                int(batch.ar_targets.ne(IGNORE_INDEX).sum())
                + int(batch.bos_targets.ne(IGNORE_INDEX).sum())
                for batch in cpu_batches
            )
        )
        local_objective_units_by_batch: list[int]
        if self.run_config.recipe == "canvas":
            local_objective_units_by_batch = [
                int(canvas_objective_units(batch.valid, self.run_config.corruption))
                for batch in cpu_batches
            ]
        elif self.run_config.recipe == "blt_d" and batch_groups is not None:
            local_objective_units_by_batch = [len(group) for group in batch_groups]
        elif self.run_config.recipe == "blt_d":
            local_objective_units_by_batch = [
                batch.ids.shape[0] for batch in cpu_batches
            ]
        else:
            local_objective_units_by_batch = [0] * microstep_count
        local_diffusion_units = sum(local_objective_units_by_batch)
        global_units = torch.tensor(
            [local_ar_units, local_diffusion_units],
            dtype=torch.float32,
            device=self.device,
        )
        if dist.is_initialized():
            dist.all_reduce(global_units)
        world_scale = float(self.distributed.world_size)
        if materialize_metrics:
            objective_total = torch.zeros((), dtype=torch.float64, device=self.device)
            ar_numerator = torch.zeros_like(objective_total)
            diffusion_numerator = torch.zeros_like(objective_total)
            ar_target_tensor = torch.zeros((), dtype=torch.long, device=self.device)
            diffusion_target_tensor = torch.zeros_like(ar_target_tensor)
            masked_total = torch.zeros_like(ar_target_tensor)
            eligible_total = torch.zeros_like(ar_target_tensor)
            all_mask_canvases = torch.zeros_like(ar_target_tensor)
            canvas_count = torch.zeros_like(ar_target_tensor)
            noise_bucket_counts = torch.zeros(7, dtype=torch.long, device=self.device)
            noise_bucket_targets = torch.zeros_like(noise_bucket_counts)
            noise_bucket_correct = torch.zeros_like(noise_bucket_counts)
            noise_bucket_nll = torch.zeros(7, dtype=torch.float64, device=self.device)
        if self.device.type == "cuda" and materialize_metrics:
            torch.cuda.synchronize(self.device)
        def prepare_cpu_group(
            group: tuple[int, ...]
        ) -> tuple[TrainingBatch, BltSamplingPlan | None]:
            training_batch = getattr(
                self.train_cursor.chunks, "training_batch", None
            )
            if training_batch is None:
                raise AssertionError("streaming source lost native batch collation")
            cpu_batch = training_batch(group)
            cpu_sampling = (
                prepare_blt_sampling(
                    cpu_batch,
                    self.run_config.corruption,
                    self.blt_sampling_generator,
                    clean_window=self.model_config.decoder_prefix_window,
                    branch_attention=self.model_config.decoder_branch_attention,
                )
                if self.run_config.recipe == "blt_d"
                else None
            )
            return (
                cpu_batch.pin_memory(),
                None if cpu_sampling is None else cpu_sampling.pin_memory(),
            )

        def transfer_pair(
            pair: tuple[TrainingBatch, BltSamplingPlan | None]
        ) -> tuple[TrainingBatch, BltSamplingPlan | None, torch.cuda.Event | None]:
            cpu_batch, cpu_sampling = pair
            if self.transfer_stream is None:
                return (
                    cpu_batch.to(self.device),
                    None if cpu_sampling is None else cpu_sampling.to(self.device),
                    None,
                )
            with torch.cuda.stream(self.transfer_stream):
                batch = cpu_batch.to(self.device, non_blocking=True)
                sampling = (
                    None
                    if cpu_sampling is None
                    else cpu_sampling.to(self.device, non_blocking=True)
                )
                ready = torch.cuda.Event()
                ready.record(self.transfer_stream)
            return batch, sampling, ready

        producer: ThreadPoolExecutor | None = None
        pending_cpu = None
        if batch_groups is not None:
            producer = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="byte-training-prefetch"
            )
            pending_cpu = producer.submit(prepare_cpu_group, batch_groups[0])
            first_cpu_pair = pending_cpu.result()
            pending_cpu = (
                producer.submit(prepare_cpu_group, batch_groups[1])
                if len(batch_groups) > 1
                else None
            )
        else:
            first_cpu_pair = (cpu_batches[0], cpu_blt_sampling[0])
        current_batch, current_sampling, current_ready = transfer_pair(first_cpu_pair)
        max_microbatch_observed = 0
        max_physical_positions_observed = 0
        try:
            for microstep, local_objective_units in enumerate(
                local_objective_units_by_batch
            ):
                if self.run_config.cuda_graphs:
                    torch.compiler.cudagraph_mark_step_begin()
                execution_stream = (
                    torch.cuda.current_stream(self.device)
                    if self.transfer_stream is not None
                    else None
                )
                if execution_stream is not None:
                    if current_ready is None:
                        raise AssertionError("CUDA transfer omitted its ready event")
                    execution_stream.wait_event(current_ready)
                    current_batch.record_stream(execution_stream)
                    if current_sampling is not None:
                        current_sampling.record_stream(execution_stream)
                batch = current_batch
                blt_sampling = current_sampling
                batch_rows, batch_width = batch.ids.shape
                max_microbatch_observed = max(
                    max_microbatch_observed, batch_rows
                )
                max_physical_positions_observed = max(
                    max_physical_positions_observed,
                    batch_rows
                    * (
                        batch_width
                        + (
                            self.run_config.corruption.corrupted_positions_per_row
                            if self.run_config.recipe != "causal_only"
                            else 0
                        )
                    ),
                )
                synchronize = microstep + 1 == microstep_count
                sync_context = (
                    self.forward_model.no_sync()  # type: ignore[attr-defined]
                    if isinstance(self.forward_model, DDP) and not synchronize
                    else nullcontext()
                )
                autocast = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.device.type == "cuda"
                    else nullcontext()
                )
                with sync_context, attention_context(
                    self.run_config.attention_policy,
                    self.device,
                    allow_cpu_reference=self.run_config.allow_cpu_reference,
                ), autocast:
                    losses = self._compute_loss(
                        batch,
                        blt_sampling=blt_sampling,
                        collect_diagnostics=materialize_metrics,
                    )
                    if (
                        self.run_config.recipe == "blt_d"
                        and self.run_config.objective_reduction
                        in {"paper_sum", "row_normalized_sum"}
                    ):
                        scaled = losses.total * (
                            batch.ids.shape[0] * world_scale / global_units[1]
                        )
                    else:
                        scaled = self.run_config.lambda_ar * losses.ar * (
                            losses.ar_targets.to(losses.ar.dtype)
                            * world_scale
                            / global_units[0]
                        )
                        if self.run_config.recipe != "causal_only":
                            scaled = scaled + losses.diffusion * (
                                local_objective_units * world_scale / global_units[1]
                            )
                scaled.backward()
                if materialize_metrics:
                    objective_total += scaled.detach().to(torch.float64) / world_scale
                    ar_numerator += (
                        losses.ar.detach().to(torch.float64)
                        * losses.ar_targets.to(torch.float64)
                    )
                    if self.run_config.recipe != "causal_only":
                        diffusion_numerator += (
                            losses.diffusion.detach().to(torch.float64)
                            * local_objective_units
                        )
                    ar_target_tensor += losses.ar_targets
                    diffusion_target_tensor += losses.diffusion_targets
                    masked_total += losses.noise_masked.sum()
                    eligible_total += losses.noise_eligible.sum()
                    all_mask_canvases += (
                        (losses.noise_eligible > 0)
                        & (losses.noise_masked == losses.noise_eligible)
                    ).sum()
                    canvas_count += losses.noise_eligible.gt(0).sum()
                if materialize_metrics and losses.noise_eligible.numel():
                    masked = losses.noise_masked
                    eligible = losses.noise_eligible
                    bucket = torch.full_like(masked, 5)
                    bucket = torch.where(4 * masked <= 3 * eligible, 4, bucket)
                    bucket = torch.where(2 * masked <= eligible, 3, bucket)
                    bucket = torch.where(4 * masked <= eligible, 2, bucket)
                    bucket = torch.where(masked == 0, 1, bucket)
                    bucket = torch.where(eligible == 0, 0, bucket)
                    bucket = torch.where(
                        (eligible > 0) & (masked == eligible), 6, bucket
                    )
                    noise_bucket_counts += torch.bincount(bucket, minlength=7)
                    noise_bucket_targets.scatter_add_(0, bucket, masked)
                    noise_bucket_correct.scatter_add_(
                        0, bucket, losses.noise_correct
                    )
                    noise_bucket_nll.scatter_add_(
                        0, bucket, losses.noise_nll.detach().to(torch.float64)
                    )
                # Resolve/build the following host batch only after all kernels
                # for the current backward have been launched.  CPU collation
                # and the non-default-stream copy then overlap actual GPU work
                # instead of stalling the host before the current forward.
                if batch_groups is not None and pending_cpu is not None:
                    next_pair = transfer_pair(pending_cpu.result())
                    following = microstep + 2
                    pending_cpu = (
                        producer.submit(prepare_cpu_group, batch_groups[following])
                        if producer is not None and following < len(batch_groups)
                        else None
                    )
                else:
                    next_pair = (
                        transfer_pair(
                            (
                                cpu_batches[microstep + 1],
                                cpu_blt_sampling[microstep + 1],
                            )
                        )
                        if batch_groups is None
                        and microstep + 1 < len(cpu_batches)
                        else None
                    )
                if next_pair is not None:
                    current_batch, current_sampling, current_ready = next_pair
        finally:
            if producer is not None:
                producer.shutdown(wait=True)

        clip_limit = (
            self.run_config.max_grad_norm
            if self.run_config.max_grad_norm is not None
            else math.inf
        )
        observed_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.joint.parameters(), clip_limit
        )
        self.optimizer.step()
        self.joint.model.enforce_padding_invariant()
        if self.device.type == "cuda" and materialize_metrics:
            torch.cuda.synchronize(self.device)
        self.completed_steps += 1
        if not materialize_metrics:
            return None
        if self._training_window_started_at is None:
            raise AssertionError("training timing window was not initialized")
        elapsed_ms = (time.perf_counter() - self._training_window_started_at) * 1_000
        self.training_time_ms += elapsed_ms
        self._training_window_started_at = None
        preclip_grad_norm = float(observed_grad_norm)
        grad_clip_scale = 1.0
        if self.run_config.max_grad_norm is not None:
            grad_clip_scale = min(
                1.0,
                self.run_config.max_grad_norm / max(preclip_grad_norm, 1e-12),
            )
        float_values = torch.cat(
            (
                torch.stack(
                    (objective_total, ar_numerator, diffusion_numerator)
                ),
                noise_bucket_nll,
            )
        )
        integer_values = torch.cat(
            (
                torch.stack(
                    (
                        ar_target_tensor,
                        diffusion_target_tensor,
                        masked_total,
                        eligible_total,
                        all_mask_canvases,
                        canvas_count,
                    )
                ),
                noise_bucket_counts,
                noise_bucket_targets,
                noise_bucket_correct,
            )
        )
        if dist.is_initialized():
            dist.all_reduce(float_values)
            dist.all_reduce(integer_values)
        values = float_values[:3]
        noise_bucket_nll = float_values[3:]
        counts = integer_values[:6]
        noise_bucket_counts = integer_values[6:13]
        noise_bucket_targets = integer_values[13:20]
        noise_bucket_correct = integer_values[20:27]
        ar_metric = values[1] / counts[0].clamp_min(1)
        diffusion_metric = values[2] / global_units[1].clamp_min(1)
        return StepMetrics(
            step=self.completed_steps,
            total=float(values[0]),
            ar=float(ar_metric),
            diffusion=float(diffusion_metric),
            learning_rate=learning_rate,
            elapsed_ms=self.training_time_ms,
            ar_targets=int(counts[0]),
            diffusion_targets=int(counts[1]),
            microsteps=microstep_count,
            max_microbatch=max_microbatch_observed,
            max_physical_positions=max_physical_positions_observed,
            preclip_grad_norm=preclip_grad_norm,
            grad_clip_scale=grad_clip_scale,
            mean_noise_fraction=float(counts[2] / counts[3].clamp_min(1)),
            all_mask_fraction=float(counts[4] / counts[5].clamp_min(1)),
            noise_bucket_counts=tuple(
                int(value) for value in noise_bucket_counts.cpu().tolist()
            ),
            noise_bucket_nll=tuple(
                float(noise_bucket_nll[index] / noise_bucket_targets[index].clamp_min(1))
                for index in range(7)
            ),
            noise_bucket_accuracy=tuple(
                float(
                    noise_bucket_correct[index]
                    / noise_bucket_targets[index].clamp_min(1)
                )
                for index in range(7)
            ),
        )

    @torch.no_grad()
    def validate(self) -> ValidationMetrics:
        started_at = time.perf_counter()
        was_training = self.joint.training
        self.joint.eval()
        # [AR NLL, masked diagnostic NLL, AR count, literal count, special
        # count, masked count, diffusion rows, absorbing ELBO NLL, ELBO atom
        # coverage]. Keep the reduction on device and synchronize once.
        totals = torch.zeros(9, dtype=torch.float64, device=self.device)
        role_nll = torch.zeros(6, dtype=torch.float64, device=self.device)
        role_counts = torch.zeros(6, dtype=torch.long, device=self.device)

        def selected_indices(limit: int | None) -> np.ndarray:
            total = len(self.validation_chunks)
            if limit is None or limit >= total:
                global_indices = np.arange(total, dtype=np.int64)
            else:
                sample_positions = np.arange(limit, dtype=np.int64)
                global_indices = (
                    (2 * sample_positions + 1) * total // (2 * limit)
                )
            # All validation objectives assign a row to the same rank. This
            # lets a joint diffusion forward also provide that row's exact AR
            # statistics instead of redundantly running its clean path twice.
            local = global_indices[
                global_indices % self.distributed.world_size
                == self.distributed.rank
            ]
            return local

        def batches(
            indices: Sequence[int],
            *,
            batch_size: int,
        ) -> Iterable[
            tuple[TrainingBatch, Sequence[PackedChunk | ValidationChunkIdentity]]
        ]:
            native_batch = getattr(
                self.validation_chunks, "validation_batch", None
            )
            for start in range(0, len(indices), batch_size):
                positions = indices[start : start + batch_size]
                if native_batch is None:
                    chunks: Sequence[PackedChunk | ValidationChunkIdentity] = [
                        self.validation_chunks[index] for index in positions
                    ]
                    item = chunks_to_batch(chunks), chunks  # type: ignore[arg-type]
                else:
                    item = native_batch(positions)
                if self.device.type == "cuda" and not item[0].ids.is_pinned():
                    item = (item[0].pin_memory(), item[1])
                yield item

        def prefetched_batches(
            indices: Sequence[int],
            *,
            batch_size: int,
        ) -> Iterable[
            tuple[TrainingBatch, Sequence[PackedChunk | ValidationChunkIdentity]]
        ]:
            source = iter(batches(indices, batch_size=batch_size))
            if self.device.type != "cuda":
                yield from source
                return
            # Native collation copies/crops memory-mapped rows and pins the
            # resulting batch. Do that one batch ahead while the GPU consumes
            # the current batch, bounding retained staging memory to two
            # batches instead of caching the whole ~946 MiB proxy.
            sentinel = object()
            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="byte-validation-prefetch"
            ) as executor:
                pending = executor.submit(next, source, sentinel)
                while True:
                    item = pending.result()
                    if item is sentinel:
                        break
                    pending = executor.submit(next, source, sentinel)
                    yield item  # type: ignore[misc]

        def autocast_context():
            return (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if self.device.type == "cuda"
                else nullcontext()
            )

        def transfer_validation(
            cpu_batch: TrainingBatch,
            cpu_sampling: BltSamplingPlan | None = None,
        ) -> tuple[TrainingBatch, BltSamplingPlan | None]:
            if self.transfer_stream is None:
                return (
                    cpu_batch.to(self.device),
                    None
                    if cpu_sampling is None
                    else cpu_sampling.to(self.device),
                )
            with torch.cuda.stream(self.transfer_stream):
                batch = cpu_batch.to(self.device, non_blocking=True)
                sampling = (
                    None
                    if cpu_sampling is None
                    else cpu_sampling.to(self.device, non_blocking=True)
                )
            execution_stream = torch.cuda.current_stream(self.device)
            execution_stream.wait_stream(self.transfer_stream)
            batch.record_stream(execution_stream)
            if sampling is not None:
                sampling.record_stream(execution_stream)
            return batch, sampling

        def accumulate_ar(
            batch: TrainingBatch,
            ar_logits: Tensor,
            bos_logits: Tensor,
        ) -> None:
            packed_targets = pack_valid(batch.ar_targets, batch.valid)
            if ar_logits.ndim == 3:
                ar_logits = pack_valid(ar_logits, batch.valid)
            ar_active = packed_targets.ne(IGNORE_INDEX)
            safe_targets = torch.where(ar_active, packed_targets, 0)
            ar_nll = cross_entropy_per_target(ar_logits, safe_targets)
            bos_nll = (
                F.cross_entropy(
                    bos_logits.float(), batch.bos_targets, reduction="sum"
                )
                if batch.bos_targets.numel()
                else bos_logits.new_zeros((), dtype=torch.float32)
            )
            totals[0] += (
                ar_nll.masked_fill(~ar_active, 0).double().sum()
                + bos_nll.double()
            )
            totals[2] += ar_active.sum() + batch.bos_targets.numel()
            selected_bos = batch.bos_targets
            totals[3] += ((packed_targets < 256) & ar_active).sum() + (
                selected_bos < 256
            ).sum()
            totals[4] += ((packed_targets >= 256) & ar_active).sum() + (
                selected_bos >= 256
            ).sum()

        all_indices = selected_indices(None)
        diffusion_indices = (
            selected_indices(
                min(
                    self.run_config.diffusion_validation_chunks,
                    len(self.validation_chunks),
                )
            )
            if self.run_config.recipe != "causal_only"
            else np.empty(0, dtype=np.int64)
        )
        ar_only_indices = np.setdiff1d(
            all_indices,
            diffusion_indices,
            assume_unique=True,
        )

        try:
            # Clean causal likelihood is a retained-language anchor. Score it
            # over every declared row, but diffusion recipes are promoted by
            # the absorbing-ELBO proxy computed below, not by this AR head.
            for cpu_batch, _ in prefetched_batches(
                ar_only_indices,
                batch_size=self.run_config.ar_validation_microbatch_per_rank,
            ):
                batch, _ = transfer_validation(cpu_batch)
                with attention_context(
                    self.run_config.attention_policy,
                    self.device,
                    allow_cpu_reference=self.run_config.allow_cpu_reference,
                ), autocast_context():
                    ar_logits, _, bos_logits = self.validation_model(
                        batch.ids,
                        batch.valid,
                        batch.positions,
                        None,
                        -1,
                        None,
                        None,
                        0,
                        batch.full_valid and not batch.isolate_documents,
                        True,
                        batch.bos_targets.shape[0],
                        None,
                        batch.byte_indices,
                        batch.byte_cu_seqlens,
                        batch.patch_indices,
                        batch.patch_cu_seqlens,
                        batch.condition_patch_indices,
                        batch.global_patch_sources,
                        batch.global_patch_positions,
                        batch.physical_to_global_patch_indices,
                        batch.bos_condition_indices,
                        None,
                        **(
                            {}
                            if batch.patch_byte_cu_seqlens is None
                            else {
                                "patch_byte_cu_seqlens": batch.patch_byte_cu_seqlens,
                                "max_patch_size": batch.max_patch_size,
                            }
                        ),
                    )
                accumulate_ar(batch, ar_logits, bos_logits)

            # All-mask diagnostics do not define BPB and are substantially more
            # expensive because each batch constructs two data-dependent Flex
            # masks. Evaluate an evenly spaced, hash-bound subset instead of
            # recomputing the diagnostic over the full BPB proxy.
            if self.run_config.recipe != "causal_only":
                for cpu_batch, chunks in prefetched_batches(
                    diffusion_indices,
                    batch_size=self.run_config.validation_microbatch_per_rank,
                ):
                    block_length = 0
                    if self.run_config.recipe == "canvas":
                        block_length = self.run_config.corruption.canvas_length
                        diffusion_mode = int(ModelMode.CANVAS)
                    else:
                        block_length = self.run_config.corruption.canvas_length
                        diffusion_mode = int(ModelMode.BLT_D)
                    identity_key = tuple(
                        (int(chunk.chunk_index), int(chunk.stream_start))
                        for chunk in chunks
                    )
                    selected_cpu: Tensor | None = None
                    condition_cpu: Tensor | None = None
                    branch_query_cpu = branch_kv_cpu = None
                    branch_query_cu_cpu = branch_kv_cu_cpu = None
                    block_mask_metadata_cpu: CanvasBlockMaskMetadata | None = None
                    blt_sampling_cpu: BltSamplingPlan | None = None
                    starts_cpu = (
                        self._validation_starts_cache.get(identity_key)
                        if self.run_config.recipe == "canvas"
                        else None
                    )
                    if self.run_config.recipe == "blt_d":
                        blt_sampling_cpu = self._validation_blt_plan_cache.get(
                            identity_key
                        )
                        if blt_sampling_cpu is None:
                            (
                                starts_cpu,
                                sampling_weight_cpu,
                                selected_cpu,
                                condition_cpu,
                            ) = sample_validation_blt_starts(
                                cpu_batch,
                                chunks,
                                block_length=block_length,
                                count=self.run_config.corruption.branches_per_row,
                                patch_stride=self.run_config.corruption.patch_stride,
                                seed=self.run_config.seed,
                            )
                            branch_valid_cpu = _blt_branch_valid(
                                cpu_batch,
                                starts_cpu,
                                selected_cpu,
                                block_length=block_length,
                            )
                            validation_active_cpu, validation_t_cpu = (
                                validation_exact_k_mask(
                                    branch_valid_cpu,
                                    chunks,
                                    seed=self.run_config.seed + 104_729,
                                )
                            )
                            if (
                                self.model_config.decoder_branch_attention
                                == "shared_flex"
                            ):
                                if cpu_batch.document_ids is None:
                                    raise AssertionError(
                                        "BLT validation omitted documents"
                                    )
                                branch_positions_cpu = _gather_spans(
                                    cpu_batch.positions,
                                    starts_cpu,
                                    block_length,
                                    fill_value=0,
                                )
                                cpu_layout = CanvasBranchLayout(
                                    clean_valid=cpu_batch.valid,
                                    branch_valid=branch_valid_cpu,
                                    prefix_lengths=starts_cpu,
                                    prefix_window=(
                                        self.model_config.decoder_prefix_window
                                    ),
                                    clean_positions=cpu_batch.positions,
                                    branch_positions=branch_positions_cpu,
                                    clean_segment_ids=cpu_batch.document_ids,
                                    branch_segment_ids=torch.gather(
                                        cpu_batch.document_ids, 1, starts_cpu
                                    ),
                                )
                                block_mask_metadata_cpu = (
                                    canvas_block_mask_metadata(cpu_layout)
                                )
                            else:
                                (
                                    _,
                                    branch_query_cpu,
                                    branch_kv_cpu,
                                    branch_query_cu_cpu,
                                    branch_kv_cu_cpu,
                                ) = _packed_blt_branch_indices(
                                    cpu_batch,
                                    starts_cpu,
                                    selected_cpu,
                                    block_length=block_length,
                                    clean_window=(
                                        self.model_config.decoder_prefix_window
                                    ),
                                )
                            blt_sampling_cpu = BltSamplingPlan(
                                block_starts=starts_cpu,
                                sampling_weight=sampling_weight_cpu,
                                selected=selected_cpu,
                                condition_indices=condition_cpu,
                                branch_valid=branch_valid_cpu,
                                block_mask_metadata=block_mask_metadata_cpu,
                                branch_query_indices=branch_query_cpu,
                                branch_kv_indices=branch_kv_cpu,
                                branch_query_cu_seqlens=branch_query_cu_cpu,
                                branch_kv_cu_seqlens=branch_kv_cu_cpu,
                                validation_active=validation_active_cpu,
                                validation_t=validation_t_cpu,
                            )
                            # Only the recurring proxy is retained. Full-split
                            # evaluation streams plans once and releases them.
                            if len(self.validation_chunks) <= 4_096:
                                if self.device.type == "cuda":
                                    blt_sampling_cpu = (
                                        blt_sampling_cpu.pin_memory()
                                    )
                                self._validation_blt_plan_cache[
                                    identity_key
                                ] = blt_sampling_cpu
                        starts_cpu = blt_sampling_cpu.block_starts
                        selected_cpu = blt_sampling_cpu.selected
                        condition_cpu = blt_sampling_cpu.condition_indices
                        branch_query_cpu = blt_sampling_cpu.branch_query_indices
                        branch_kv_cpu = blt_sampling_cpu.branch_kv_indices
                        branch_query_cu_cpu = (
                            blt_sampling_cpu.branch_query_cu_seqlens
                        )
                        branch_kv_cu_cpu = blt_sampling_cpu.branch_kv_cu_seqlens
                        block_mask_metadata_cpu = (
                            blt_sampling_cpu.block_mask_metadata
                        )
                    elif starts_cpu is None:
                        starts_cpu = sample_validation_starts(
                            cpu_batch.valid,
                            chunks,
                            span_length=block_length,
                            count=self.run_config.corruption.branches_per_row,
                            patch_stride=self.run_config.corruption.patch_stride,
                            seed=self.run_config.seed,
                        )
                        self._validation_starts_cache[identity_key] = starts_cpu
                    batch, blt_sampling_device = transfer_validation(
                        cpu_batch,
                        blt_sampling_cpu,
                    )
                    starts_tensor = (
                        blt_sampling_device.block_starts
                        if blt_sampling_device is not None
                        else starts_cpu.to(self.device)
                    )
                    diffusion_clean = _gather_spans(
                        batch.ids,
                        starts_tensor,
                        block_length,
                        fill_value=self.model_config.vocab.pad_id,
                    )
                    noisy_valid = _gather_spans(
                        batch.valid, starts_tensor, block_length, fill_value=False
                    )
                    if self.run_config.recipe == "blt_d":
                        if batch.document_ids is None or blt_sampling_device is None:
                            raise AssertionError("BLT validation omitted document plan")
                        noisy_valid = blt_sampling_device.branch_valid
                        diffusion_clean = torch.where(
                            noisy_valid,
                            diffusion_clean,
                            self.model_config.vocab.pad_id,
                        )
                    if self.run_config.recipe == "blt_d":
                        if (
                            blt_sampling_device is None
                            or blt_sampling_device.validation_active is None
                            or blt_sampling_device.validation_t is None
                        ):
                            raise AssertionError(
                                "BLT validation omitted its exact-K ELBO sample"
                            )
                        diffusion_active = (
                            blt_sampling_device.validation_active & noisy_valid
                        )
                    else:
                        # Canvas objectives do not define Fast-BLT's absorbing
                        # ELBO. Retain their all-mask reconstruction diagnostic
                        # without presenting it as a likelihood or BPB.
                        diffusion_active = noisy_valid
                    noisy = torch.where(
                        diffusion_active,
                        self.model_config.vocab.mask_id,
                        diffusion_clean,
                    )
                    branch_block_mask = None
                    if (
                        blt_sampling_device is not None
                        and blt_sampling_device.block_mask_metadata is not None
                        and batch.ids.is_cuda
                    ):
                        if batch.document_ids is None:
                            raise AssertionError("BLT validation omitted documents")
                        branch_positions = _gather_spans(
                            batch.positions, starts_tensor, block_length, fill_value=0
                        )
                        device_layout = CanvasBranchLayout(
                            clean_valid=batch.valid,
                            branch_valid=noisy_valid,
                            prefix_lengths=starts_tensor,
                            prefix_window=self.model_config.decoder_prefix_window,
                            clean_positions=batch.positions,
                            branch_positions=branch_positions,
                            clean_segment_ids=batch.document_ids,
                            branch_segment_ids=torch.gather(
                                batch.document_ids, 1, starts_tensor
                            ),
                        )
                        branch_block_mask = build_canvas_block_mask(
                            device_layout,
                            metadata=blt_sampling_device.block_mask_metadata,
                        )
                    with attention_context(
                        self.run_config.attention_policy,
                        self.device,
                        allow_cpu_reference=self.run_config.allow_cpu_reference,
                    ), autocast_context():
                        ar_logits, diffusion_logits, bos_logits = self.validation_model(
                            batch.ids,
                            batch.valid,
                            batch.positions,
                            noisy,
                            diffusion_mode,
                            noisy_valid,
                            starts_tensor,
                            block_length,
                            batch.full_valid and not batch.isolate_documents,
                            True,
                            batch.bos_targets.shape[0],
                            batch.document_ids if batch.isolate_documents else None,
                            batch.byte_indices,
                            batch.byte_cu_seqlens,
                            batch.patch_indices,
                            batch.patch_cu_seqlens,
                            batch.condition_patch_indices,
                            batch.global_patch_sources,
                            batch.global_patch_positions,
                            batch.physical_to_global_patch_indices,
                            batch.bos_condition_indices,
                            (
                                None
                                if blt_sampling_device is None
                                else blt_sampling_device.condition_indices
                            ),
                            (
                                None
                                if self.run_config.recipe != "blt_d"
                                else (
                                    None
                                    if blt_sampling_device is None
                                    else blt_sampling_device.branch_query_indices
                                )
                            ),
                            (
                                None
                                if self.run_config.recipe != "blt_d"
                                else (
                                    None
                                    if blt_sampling_device is None
                                    else blt_sampling_device.branch_kv_indices
                                )
                            ),
                            (
                                None
                                if self.run_config.recipe != "blt_d"
                                else (
                                    None
                                    if blt_sampling_device is None
                                    else blt_sampling_device.branch_query_cu_seqlens
                                )
                            ),
                            (
                                None
                                if self.run_config.recipe != "blt_d"
                                else (
                                    None
                                    if blt_sampling_device is None
                                    else blt_sampling_device.branch_kv_cu_seqlens
                                )
                            ),
                            branch_block_mask,
                            **(
                                {}
                                if batch.patch_byte_cu_seqlens is None
                                else {
                                    "patch_byte_cu_seqlens": batch.patch_byte_cu_seqlens,
                                    "max_patch_size": batch.max_patch_size,
                                }
                            ),
                        )
                    if diffusion_logits is None:
                        raise AssertionError("validation diffusion logits are missing")
                    accumulate_ar(batch, ar_logits, bos_logits)
                    targets = same_position_targets(
                        diffusion_clean.flatten(0, 1)
                        if diffusion_clean.ndim == 3
                        else diffusion_clean,
                        diffusion_active.flatten(0, 1)
                        if diffusion_active.ndim == 3
                        else diffusion_active,
                    )
                    if diffusion_clean.ndim == 3:
                        targets = targets.view_as(diffusion_clean)
                    per_target_nll = masked_cross_entropy_per_target(
                        diffusion_logits, targets, diffusion_active
                    ).to(torch.float64)
                    totals[1] += per_target_nll.sum()
                    totals[5] += diffusion_active.sum()
                    totals[6] += batch.ids.shape[0]
                    if self.run_config.recipe == "blt_d":
                        assert blt_sampling_device is not None
                        assert blt_sampling_device.validation_t is not None
                        row_nll = per_target_nll.flatten(1).sum(1)
                        inverse_t = blt_sampling_device.validation_t.reciprocal()
                        origin_weight = blt_sampling_device.sampling_weight
                        totals[7] += (row_nll * inverse_t * origin_weight).sum()
                        totals[8] += (
                            noisy_valid.flatten(1).sum(1) * origin_weight
                        ).sum()
                    safe_targets = torch.where(diffusion_active, targets, 0)
                    roles = torch.full_like(safe_targets, 3)
                    roles = torch.where(safe_targets <= 0x7F, 0, roles)
                    roles = torch.where(
                        (safe_targets >= 0xC2) & (safe_targets <= 0xF4),
                        1,
                        roles,
                    )
                    roles = torch.where(
                        (safe_targets >= 0x80) & (safe_targets <= 0xBF),
                        2,
                        roles,
                    )
                    roles = torch.where(
                        safe_targets == self.atomic_manifest.eot_id, 4, roles
                    )
                    roles = torch.where(
                        (safe_targets >= 256)
                        & (safe_targets != self.atomic_manifest.eot_id),
                        5,
                        roles,
                    )
                    flat_roles = roles.flatten()
                    role_nll.scatter_add_(0, flat_roles, per_target_nll.flatten())
                    role_counts.scatter_add_(
                        0,
                        flat_roles,
                        diffusion_active.flatten().to(role_counts.dtype),
                    )
        finally:
            self.joint.train(was_training)
        if dist.is_initialized():
            dist.all_reduce(totals)
            dist.all_reduce(role_nll)
            dist.all_reduce(role_counts)
        ar_count = int(totals[2])
        if ar_count == 0:
            raise ValueError("validation has no AR targets")
        ar_loss = float(totals[0] / ar_count)
        literal_count = int(totals[3])
        if literal_count == 0:
            raise ValueError("validation has no literal-byte targets")
        special_count = int(totals[4])
        diff_count = int(totals[5])
        diffusion_chunks = int(totals[6])
        diffusion_loss = float(totals[1] / diff_count) if diff_count else 0.0
        diffusion_elbo_atoms = float(totals[8])
        diffusion_elbo_proxy_nats_per_block_atom = (
            float(totals[7] / totals[8]) if diffusion_elbo_atoms else None
        )
        diffusion_elbo_proxy_bpb = (
            diffusion_elbo_proxy_nats_per_block_atom / math.log(2.0)
            if diffusion_elbo_proxy_nats_per_block_atom is not None
            else None
        )
        return ValidationMetrics(
            ar_loss=ar_loss,
            # The challenge byte LUT charges every registered special atom,
            # including EOT, as one byte. Atomic byte targets therefore use
            # the complete scored-target count as the BPB denominator.
            bpb=float(totals[0]) / ar_count / math.log(2.0),
            atomic_bpb=ar_loss / math.log(2.0),
            diffusion_loss=diffusion_loss,
            diffusion_elbo_proxy_bpb=diffusion_elbo_proxy_bpb,
            diffusion_elbo_proxy_nats_per_block_atom=(
                diffusion_elbo_proxy_nats_per_block_atom
            ),
            diffusion_elbo_atoms=diffusion_elbo_atoms,
            ar_targets=ar_count,
            literal_bytes=literal_count,
            special_targets=special_count,
            diffusion_targets=diff_count,
            diffusion_chunks=diffusion_chunks,
            elapsed_ms=(time.perf_counter() - started_at) * 1_000,
            diffusion_role_nll=tuple(
                float(role_nll[index] / role_counts[index].clamp_min(1))
                for index in range(6)
            ),
            diffusion_role_counts=tuple(
                int(value) for value in role_counts.cpu().tolist()
            ),
        )

    def _rank_state(self) -> dict[str, Any]:
        return {
            "cursor": self.train_cursor.state_dict(),
            "corruption_generator": self.corruption_generator.get_state(),
            "blt_sampling_generator": self.blt_sampling_generator.get_state(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if self.device.type == "cuda"
            else None,
        }

    def save_checkpoint(self, path: Path) -> None:
        rank_state = self._rank_state()
        rank_states: list[dict[str, Any] | None]
        if dist.is_initialized():
            gathered: list[dict[str, Any] | None] = [None] * self.distributed.world_size
            dist.all_gather_object(gathered, rank_state)
            rank_states = gathered
        else:
            rank_states = [rank_state]
        if self.distributed.is_primary:
            payload = {
                "schema": CHECKPOINT_SCHEMA,
                "completed_steps": self.completed_steps,
                "training_time_ms": self.training_time_ms,
                "model_config": self.model_config.to_dict(),
                "run_contract": self.run_config.contract_dict(),
                "initialization_kind": self.run_config.initialization_kind,
                "dataset_sha256": self.train_cursor.dataset_sha256,
                "validation_sha256": self.validation_sha256,
                "atomic_manifest": self.atomic_manifest.to_dict(),
                "atomic_manifest_sha256": self.atomic_manifest.sha256,
                "dataset_provenance": self.dataset_provenance,
                "source_provenance": self.source_provenance,
                "world_size": self.distributed.world_size,
                "model": self.joint.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "rank_states": rank_states,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".working")
            torch.save(payload, temporary)
            temporary.replace(path)
        self.distributed.barrier()

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        expected = {
            "schema": CHECKPOINT_SCHEMA,
            "model_config": self.model_config.to_dict(),
            "run_contract": self.run_config.contract_dict(),
            "initialization_kind": self.run_config.initialization_kind,
            "dataset_sha256": self.train_cursor.dataset_sha256,
            "validation_sha256": self.validation_sha256,
            "atomic_manifest": self.atomic_manifest.to_dict(),
            "atomic_manifest_sha256": self.atomic_manifest.sha256,
            "dataset_provenance": self.dataset_provenance,
            "source_provenance": self.source_provenance,
            "world_size": self.distributed.world_size,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"checkpoint {key} does not match the active run")
        completed_steps = int(payload.get("completed_steps", -1))
        if not 0 <= completed_steps <= self.run_config.iterations:
            raise ValueError("checkpoint step is outside the planned schedule")
        rank_states = payload.get("rank_states")
        if (
            not isinstance(rank_states, list)
            or len(rank_states) != self.distributed.world_size
        ):
            raise ValueError("checkpoint rank RNG state count is invalid")
        rank_state = rank_states[self.distributed.rank]
        if not isinstance(rank_state, Mapping):
            raise ValueError("checkpoint rank state is invalid")
        self.joint.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.train_cursor.load_state_dict(rank_state["cursor"])
        # ``map_location=self.device`` also relocates serialized RNG byte
        # tensors. Generator APIs require their state tensors on CPU even when
        # the generator itself is CUDA-backed.
        self.corruption_generator.set_state(
            rank_state["corruption_generator"].cpu()
        )
        self.blt_sampling_generator.set_state(
            rank_state["blt_sampling_generator"].cpu()
        )
        random.setstate(rank_state["python_rng"])
        torch.set_rng_state(rank_state["torch_rng"].cpu())
        if self.device.type == "cuda":
            cuda_rng = rank_state.get("cuda_rng")
            if cuda_rng is None:
                raise ValueError("CUDA checkpoint omitted CUDA RNG states")
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng])
        self.completed_steps = completed_steps
        self.training_time_ms = float(payload.get("training_time_ms", 0.0))


def model_config_from_dict(value: Mapping[str, Any]) -> ByteDiffusionConfig:
    fields = dict(value)
    fields["vocab"] = AtomicVocabulary(**fields["vocab"])
    if "ngram_orders" in fields:
        fields["ngram_orders"] = tuple(fields["ngram_orders"])
    return ByteDiffusionConfig(**fields)


def run_config_from_dict(value: Mapping[str, Any]) -> TrainingRunConfig:
    fields = dict(value)
    fields["corruption"] = CorruptionConfig(**fields["corruption"])
    return TrainingRunConfig(**fields)


def load_atomic_documents(
    path: Path, manifest: AtomicIdManifest
) -> tuple[AtomicDocument, ...]:
    """Load JSONL records with either explicit ``atomic_ids`` or UTF-8 ``text``."""

    documents: list[AtomicDocument] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = str(record.get("key", f"{path.stem}:{line_number}"))
            if "atomic_ids" in record:
                atomic_ids = tuple(int(value) for value in record["atomic_ids"])
            elif "text" in record:
                atomic_ids = (*str(record["text"]).encode("utf-8"), manifest.eot_id)
            else:
                raise ValueError(
                    f"{path}:{line_number} needs 'atomic_ids' or 'text'"
                )
            document = AtomicDocument(key=key, atomic_ids=atomic_ids)
            document.validate(manifest)
            documents.append(document)
    if not documents:
        raise ValueError(f"no documents found in {path}")
    return tuple(documents)


class _MappedArtifactPayload:
    """Small mapping facade over independently memory-mapped artifact arrays."""

    def __init__(
        self,
        path: Path,
        descriptors: Mapping[str, Mapping[str, Any]],
        names: Iterable[str],
    ) -> None:
        self.files = tuple(names)
        self._arrays = {
            name: np.memmap(
                path,
                mode="r",
                dtype=np.dtype(str(descriptors[name]["dtype"])),
                offset=int(descriptors[name]["byte_offset"]),
                shape=tuple(int(value) for value in descriptors[name]["shape"]),
                order="C",
            )
            for name in self.files
        }

    def __getitem__(self, name: str) -> np.ndarray:
        return self._arrays[name]

    def __enter__(self) -> "_MappedArtifactPayload":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


def _validate_variable_patch_row(
    columns: np.ndarray,
    documents: np.ndarray,
    patch_offsets: np.ndarray,
    *,
    max_patch_size: int,
) -> None:
    """Validate one v6 row without assuming a fixed physical patch grid."""

    values = patch_offsets[columns].astype(np.int64, copy=False)
    if values.size == 0 or int(values[0]) != 0:
        raise ValueError("entropy-patched rows must begin on a patch boundary")
    starts = np.flatnonzero(values == 0)
    lengths = np.diff(np.r_[starts, values.size])
    if bool(((lengths <= 0) | (lengths > max_patch_size)).any()):
        raise ValueError("entropy patch length exceeds its manifest-bound maximum")
    expected = np.arange(values.size) - np.repeat(starts, lengths)
    if not np.array_equal(values, expected):
        raise ValueError("entropy patch offsets are not contiguous zero-based runs")
    patch_documents = documents[columns[starts]]
    if not np.array_equal(documents[columns], np.repeat(patch_documents, lengths)):
        raise ValueError("one entropy patch mixes multiple documents")


class MappedPackedChunkDataset(Sequence[PackedChunk]):
    """Hash-verified view over deterministic row-addressable artifacts.

    The manifest supplies compact scheduling metadata and byte descriptors for
    one aligned raw-array container per shard. Array objects are read-only
    memory maps; native collation faults in and copies only selected rows.
    """

    _REQUIRED_FIELDS = frozenset(
        {
            "chunk_index",
            "stream_start",
            "stream_stop",
            "input_ids",
            "target_ids",
            "valid_mask",
            "score_mask",
            "document_indices",
            "document_offsets",
            "patch_offsets",
            "label_halo_id",
            "label_halo_valid",
        }
    )

    def __init__(
        self,
        directory: Path,
        artifacts: Sequence[Mapping[str, Any]],
        *,
        artifact_schema: str,
        chunk_size: int,
        required_branch_bytes: int,
        branch_span_length: int,
        split: str,
        dense_stream: bool,
        document_aligned_pages: bool = False,
        patching: DatasetPatchingSpec = DatasetPatchingSpec(
            "fixed_stride_v1", 4, 4
        ),
        eot_id: int = 256,
        defer_payload_validation: bool = False,
        trust_pinned_row_index: bool = False,
    ) -> None:
        self.directory = directory
        self.artifacts = tuple(dict(artifact) for artifact in artifacts)
        self.artifact_schema = artifact_schema
        self.chunk_size = chunk_size
        self.required_branch_bytes = required_branch_bytes
        self.branch_span_length = int(branch_span_length)
        if self.branch_span_length < 0 or self.branch_span_length % 4:
            raise ValueError("branch span length must be a nonnegative patch multiple")
        self.branch_tail_halo = max(0, self.branch_span_length - 4)
        self.split = split
        self.dense_stream = bool(dense_stream)
        self.document_aligned_pages = bool(document_aligned_pages)
        self.patching = patching
        self.eot_id = int(eot_id)
        self.defer_payload_validation = bool(defer_payload_validation)
        self.trust_pinned_row_index = bool(trust_pinned_row_index)
        if self.trust_pinned_row_index and not self.defer_payload_validation:
            raise ValueError("trusted row-index loading requires deferred validation")
        artifact_offsets = [0]
        valid_count_blocks: list[np.ndarray] = []
        physical_extent_blocks: list[np.ndarray] = []
        document_start_blocks: list[np.ndarray] = []
        row_digest_blocks: list[np.ndarray] = []
        identity_records: list[dict[str, Any]] = []
        exposure = {
            "rows": 0,
            "valid_atomic_tokens": 0,
            "literal_atomic_tokens": 0,
            "special_atomic_tokens": 0,
            "eot_atomic_tokens": 0,
            "physical_storage_positions": 0,
            "storage_padding_tokens": 0,
            "canvas512_eligible_positions": 0,
            "scored_ar_targets": 0,
        }
        observed_document_starts = 0
        expected_continuation: tuple[int, int] | None = None
        for artifact_index, artifact in enumerate(self.artifacts):
            if artifact.get("schema") != artifact_schema:
                raise ValueError(f"{split} artifact schema mismatch")
            relative = Path(str(artifact["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("dataset artifact path escapes its directory")
            path = directory / relative
            if path.stat().st_size != int(artifact["size_bytes"]):
                raise ValueError(f"dataset artifact size mismatch: {relative}")
            self._validate_mapped_index(artifact, path.stat().st_size)
            row_count = int(artifact.get("chunks", -1))
            raw_row_digests = artifact.get("row_sha256")
            if (
                row_count <= 0
                or not isinstance(raw_row_digests, list)
                or len(raw_row_digests) != row_count
                or any(
                    type(value) is not str or len(value) != 64
                    for value in raw_row_digests
                )
            ):
                raise ValueError(f"artifact {relative} lacks its row digests")
            try:
                row_digest_blocks.append(
                    np.frombuffer(
                        b"".join(bytes.fromhex(value) for value in raw_row_digests),
                        dtype=np.uint8,
                    ).reshape(row_count, 32).copy()
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"artifact {relative} has malformed row digests"
                ) from error
            artifact.pop("row_sha256")
            digest = str(artifact["sha256"])
            if not self.defer_payload_validation:
                observed_digest = self._sha256_file(path)
                if observed_digest != digest:
                    raise ValueError(f"dataset artifact sha256 mismatch: {relative}")
            if self.defer_payload_validation:
                row_valid_counts = artifact.get("row_valid_counts")
                row_physical_extents = artifact.get("row_physical_extents")
                row_document_starts = artifact.get("row_document_starts")
                row_fields = (
                    row_valid_counts,
                    row_physical_extents,
                    row_document_starts,
                )
                if row_count <= 0 or any(
                    not isinstance(field, list) or len(field) != row_count
                    for field in row_fields
                ):
                    raise ValueError(
                        f"artifact {relative} lacks its hash-bound row index"
                    )
                if any(
                    not isinstance(value, int)
                    for field in row_fields
                    for value in field
                ):
                    raise ValueError(f"artifact {relative} row index is not integral")
                if any(
                    count <= 0 or count > chunk_size
                    for count in row_valid_counts
                ) or any(
                    extent <= 0 or extent > chunk_size
                    for extent in row_physical_extents
                ):
                    raise ValueError(f"artifact {relative} row index is out of range")
                if any(
                    extent < count
                    for count, extent in zip(
                        row_valid_counts, row_physical_extents, strict=True
                    )
                ):
                    raise ValueError(f"artifact {relative} row extents are impossible")
                if sum(row_valid_counts) != int(
                    artifact.get("valid_atomic_tokens", -1)
                ):
                    raise ValueError(f"artifact {relative} row counts are false")
                if sum(row_document_starts) > int(
                    artifact.get("eot_atomic_tokens", -1)
                ) + row_count:
                    raise ValueError(f"artifact {relative} document index is impossible")
                observed_fields = {
                    "rows": row_count,
                    "valid_atomic_tokens": int(artifact["valid_atomic_tokens"]),
                    "literal_atomic_tokens": int(artifact["literal_atomic_tokens"]),
                    "special_atomic_tokens": int(artifact["special_atomic_tokens"]),
                    "eot_atomic_tokens": int(artifact["eot_atomic_tokens"]),
                    "physical_storage_positions": int(
                        artifact["physical_storage_positions"]
                    ),
                    "storage_padding_tokens": int(
                        artifact["storage_padding_tokens"]
                    ),
                    "canvas512_eligible_positions": int(
                        artifact["canvas512_eligible_positions"]
                    ),
                    "scored_ar_targets": int(artifact["scored_ar_targets"]),
                }
                if observed_fields["physical_storage_positions"] != (
                    row_count * chunk_size
                ):
                    raise ValueError(f"artifact {relative} physical size is false")
                if observed_fields["storage_padding_tokens"] != (
                    observed_fields["physical_storage_positions"]
                    - observed_fields["valid_atomic_tokens"]
                ):
                    raise ValueError(f"artifact {relative} PAD count is false")
                for name, value in observed_fields.items():
                    exposure[name] += value
                valid_count_blocks.append(np.asarray(row_valid_counts, dtype=np.int32))
                physical_extent_blocks.append(
                    np.asarray(row_physical_extents, dtype=np.int32)
                )
                document_start_blocks.append(
                    np.asarray(row_document_starts, dtype=np.int32)
                )
                # The JSON lists and boxed Python integers dominate resident
                # loader memory at production scale.  The compact arrays above
                # retain the same hash-bound metadata for first-use checks.
                artifact.pop("row_valid_counts")
                artifact.pop("row_physical_extents")
                artifact.pop("row_document_starts")
                artifact_offsets.append(artifact_offsets[-1] + row_count)
                observed_document_starts += sum(row_document_starts)
                identity_records.append(
                    {
                        "path": str(relative),
                        "sha256": digest,
                        "rows": row_count,
                    }
                )
                continue
            with self._mapped_payload(artifact_index) as payload:
                required = {
                    "chunk_index",
                    "stream_start",
                    "stream_stop",
                    "input_ids",
                    "target_ids",
                    "valid_mask",
                    "score_mask",
                    "document_indices",
                    "document_offsets",
                    "patch_offsets",
                    "label_halo_id",
                    "label_halo_valid",
                }
                if not required.issubset(payload.files):
                    missing = sorted(required.difference(payload.files))
                    raise ValueError(f"dataset artifact omitted fields: {missing}")
                valid = payload["valid_mask"]
                ids = payload["input_ids"]
                score = payload["score_mask"]
                documents = payload["document_indices"]
                offsets = payload["document_offsets"]
                patch_offsets = payload["patch_offsets"]
                targets = payload["target_ids"]
                halo_valid = payload["label_halo_valid"]
                halo_ids = payload["label_halo_id"]
                if valid.ndim != 2 or valid.shape[1] != chunk_size:
                    raise ValueError("dataset artifact input shape mismatch")
                aligned_arrays = {
                    "input ids": ids,
                    "target ids": targets,
                    "score mask": score,
                    "document ids": documents,
                    "document offsets": offsets,
                    "patch offsets": patch_offsets,
                }
                for name, array in aligned_arrays.items():
                    if array.shape != valid.shape:
                        raise ValueError(f"artifact {name} do not align")
                if halo_valid.shape != (valid.shape[0],) or halo_ids.shape != (
                    valid.shape[0],
                ):
                    raise ValueError("artifact label halos do not align")
                invalid = ~valid
                if bool(score[invalid].any()):
                    raise ValueError("invalid storage positions cannot be scored")
                if bool((ids[invalid] != PAD_ID).any()) or bool(
                    (targets[invalid] != PAD_ID).any()
                ):
                    raise ValueError("invalid storage positions must contain PAD")
                if (
                    bool((documents[invalid] != -1).any())
                    or bool((offsets[invalid] != -1).any())
                    or bool((patch_offsets[invalid] != -1).any())
                ):
                    raise ValueError("invalid storage positions cannot carry metadata")
                if bool((ids[valid] < 0).any()) or bool((ids[valid] >= MASK_ID).any()):
                    raise ValueError("valid inputs must be clean atomic ids")
                if bool((targets[score] < 0).any()) or bool(
                    (targets[score] >= MASK_ID).any()
                ):
                    raise ValueError("scored targets must be clean atomic ids")
                if bool((targets[valid & ~score] != PAD_ID).any()):
                    raise ValueError("unscored positions must contain PAD targets")
                valid_ids = ids[valid]
                observed_exposure = {
                    "rows": int(valid.shape[0]),
                    "valid_atomic_tokens": int(valid.sum()),
                    "literal_atomic_tokens": int((valid_ids < 256).sum()),
                    "special_atomic_tokens": int((valid_ids >= 256).sum()),
                    "eot_atomic_tokens": int((valid_ids == 256).sum()),
                    "physical_storage_positions": int(ids.size),
                    "storage_padding_tokens": int(ids.size - valid.sum()),
                    "canvas512_eligible_positions": int(
                        np.minimum(valid.sum(axis=1), 512).sum()
                    ),
                    "scored_ar_targets": int(score.sum()),
                }
                for name, value in observed_exposure.items():
                    claimed = artifact.get(name if name != "rows" else "chunks")
                    if claimed is None:
                        raise ValueError(
                            f"artifact {relative} omitted required exposure {name}"
                        )
                    if int(claimed) != value:
                        raise ValueError(
                            f"artifact {relative} has false {name}: "
                            f"claimed {claimed}, observed {value}"
                        )
                    exposure[name] += value
                artifact_valid_counts = np.empty(valid.shape[0], dtype=np.int32)
                artifact_physical_extents = np.empty(valid.shape[0], dtype=np.int32)
                for row in range(valid.shape[0]):
                    row_valid = valid[row]
                    valid_count = int(row_valid.sum())
                    if (
                        not self.document_aligned_pages
                        and bool(np.any((~row_valid[:-1]) & row_valid[1:]))
                    ):
                        raise ValueError("v2 artifact row is not one valid prefix")
                    document_set = set(int(value) for value in documents[row, row_valid])
                    if not document_set or min(document_set) < 0:
                        raise ValueError("artifact row has invalid document metadata")
                    if (
                        not self.dense_stream
                        and not self.document_aligned_pages
                        and len(document_set) != 1
                    ):
                        raise ValueError(
                            "isolated artifact row must contain one document segment"
                        )
                    if self.document_aligned_pages:
                        columns = np.flatnonzero(row_valid)
                        if columns[0] != 0:
                            raise ValueError("aligned page must start at physical column zero")
                        row_documents = documents[row]
                        row_offsets = offsets[row]
                        row_patch_offsets = patch_offsets[row]
                        starts = columns[
                            np.r_[
                                True,
                                row_documents[columns[1:]]
                                != row_documents[columns[:-1]],
                            ]
                        ]
                        if not self.patching.variable and np.any(
                            starts % self.patching.max_patch_size
                        ):
                            raise ValueError(
                                "document segment does not start on a patch boundary"
                            )
                        continuation = expected_continuation
                        first_document = int(row_documents[columns[0]])
                        first_offset = int(row_offsets[columns[0]])
                        if continuation is None:
                            if first_offset != 0:
                                raise ValueError(
                                    "aligned page starts with an orphan continuation"
                                )
                        elif (first_document, first_offset) != continuation:
                            raise ValueError(
                                "aligned page does not continue the preceding document"
                            )
                        start_offsets = row_offsets[starts]
                        if np.any(start_offsets[1:] != 0):
                            raise ValueError(
                                "mid-page document segment must start at offset zero"
                            )
                        observed_document_starts += int((start_offsets == 0).sum())
                        if self.patching.variable:
                            _validate_variable_patch_row(
                                columns,
                                row_documents,
                                row_patch_offsets,
                                max_patch_size=self.patching.max_patch_size,
                            )
                            if bool((row_patch_offsets[starts] != 0).any()):
                                raise ValueError(
                                    "document segment must begin on an entropy boundary"
                                )
                        else:
                            stride = self.patching.max_patch_size
                            if np.any(
                                row_patch_offsets[columns]
                                != row_offsets[columns] % stride
                            ):
                                raise ValueError("document patch phase did not reset")
                            for patch_start in range(0, chunk_size, stride):
                                patch_documents = set(
                                    int(value)
                                    for value in row_documents[
                                        patch_start : patch_start + stride
                                    ]
                                    if value >= 0
                                )
                                if len(patch_documents) > 1:
                                    raise ValueError("one patch mixes two documents")
                        for left, right in zip(columns[:-1], columns[1:], strict=True):
                            same_document = (
                                row_documents[left] == row_documents[right]
                            )
                            if same_document:
                                if right != left + 1 or (
                                    row_offsets[right] != row_offsets[left] + 1
                                ):
                                    raise ValueError("document atoms are not contiguous")
                            elif right - left - 1 not in range(
                                0,
                                1 if self.patching.variable else self.patching.max_patch_size,
                            ):
                                raise ValueError(
                                    "inter-document alignment gap is not zero to three"
                                )
                            elif ids[row, left] != self.eot_id or score[row, left]:
                                raise ValueError(
                                    "document boundary does not follow an unscored EOT"
                                )
                        eot = row_valid & (ids[row] == self.eot_id)
                        if bool(score[row, eot].any()):
                            raise ValueError("terminal EOT cannot have an AR target")
                        if bool((~score[row, row_valid & ~eot]).any()):
                            raise ValueError(
                                "every non-EOT atom must carry its shifted AR target"
                            )
                        scored_columns = np.flatnonzero(score[row])
                        for column in scored_columns:
                            if column + 1 < chunk_size and (
                                row_valid[column + 1]
                                and row_documents[column + 1]
                                == row_documents[column]
                            ):
                                if targets[row, column] != ids[row, column + 1]:
                                    raise ValueError("stored AR target is not next atom")
                            elif not (
                                column == columns[-1]
                                and halo_valid[row]
                                and targets[row, column] == halo_ids[row]
                            ):
                                raise ValueError("scored page edge omitted its halo")
                        start_positions = np.searchsorted(columns, starts)
                        stop_positions = np.r_[start_positions[1:], len(columns)]
                        for segment_index, stop_position in enumerate(stop_positions):
                            final_column = int(columns[stop_position - 1])
                            final_is_eot = ids[row, final_column] == self.eot_id
                            is_final_segment = segment_index == len(stop_positions) - 1
                            if not is_final_segment and not final_is_eot:
                                raise ValueError(
                                    "mid-page document segment must end with EOT"
                                )
                            if final_is_eot and (
                                score[row, final_column]
                                or (is_final_segment and bool(halo_valid[row]))
                            ):
                                raise ValueError(
                                    "terminal EOT cannot be scored or carry a halo"
                                )
                        final_column = int(columns[-1])
                        if ids[row, final_column] == self.eot_id:
                            expected_continuation = None
                        else:
                            if (
                                (
                                    not self.patching.variable
                                    and final_column != chunk_size - 1
                                )
                                or not bool(halo_valid[row])
                            ):
                                raise ValueError(
                                    "unterminated document segment must end at an "
                                    "allowed row boundary and carry a halo"
                                )
                            expected_continuation = (
                                int(row_documents[final_column]),
                                int(row_offsets[final_column]) + 1,
                            )
                    artifact_valid_counts[row] = valid_count
                    artifact_physical_extents[row] = (
                        int(np.flatnonzero(row_valid)[-1]) + 1
                    )
                valid_count_blocks.append(artifact_valid_counts)
                physical_extent_blocks.append(artifact_physical_extents)
                document_start_blocks.append(
                    (valid & (offsets == 0)).sum(axis=1, dtype=np.int32)
                )
                artifact_offsets.append(artifact_offsets[-1] + valid.shape[0])
            identity_records.append(
                {
                    "path": str(relative),
                    "sha256": digest,
                    "rows": int(valid.shape[0]),
                }
            )
        if self.document_aligned_pages:
            if expected_continuation is not None:
                raise ValueError("dataset ends with an unterminated document")
            if observed_document_starts != exposure["eot_atomic_tokens"]:
                raise ValueError(
                    "document start and terminal EOT counts disagree: "
                    f"{observed_document_starts} starts, "
                    f"{exposure['eot_atomic_tokens']} EOTs"
                )
            exposure["bos_ar_targets"] = observed_document_starts
            exposure["total_ar_targets"] = (
                exposure["scored_ar_targets"] + observed_document_starts
            )
        if artifact_offsets[-1] == 0:
            raise ValueError(f"{split} has no rows satisfying the model contract")
        self._artifact_offsets = np.asarray(artifact_offsets, dtype=np.int64)
        self._valid_counts = np.concatenate(valid_count_blocks)
        self._physical_extents = np.concatenate(physical_extent_blocks)
        self._row_document_starts = tuple(document_start_blocks)
        self._row_digests = tuple(row_digest_blocks)
        self._verified_rows = tuple(
            np.zeros(block.shape[0], dtype=np.bool_) for block in row_digest_blocks
        )
        self._verified_artifacts: set[int] = (
            set() if self.defer_payload_validation else set(range(len(self.artifacts)))
        )
        self._verified_artifact_stats: dict[int, tuple[int, ...]] = {}
        if not self.defer_payload_validation:
            for artifact_index, artifact in enumerate(self.artifacts):
                self._verified_artifact_stats[artifact_index] = self._file_identity(
                    self.directory / str(artifact["path"])
                )
        self._semantically_validated_artifacts: set[int] = (
            set() if self.defer_payload_validation else set(range(len(self.artifacts)))
        )
        self.exposure_summary = dict(exposure)
        self.dataset_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "schema": artifact_schema,
                    "split": split,
                    "chunk_size": chunk_size,
                    "required_branch_bytes": required_branch_bytes,
                    "branch_span_length": self.branch_span_length,
                    "artifacts": identity_records,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def __len__(self) -> int:
        return int(self._artifact_offsets[-1])

    @staticmethod
    def _sha256_file(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(block)
        return hasher.hexdigest()

    @staticmethod
    def _file_identity(path: Path) -> tuple[int, ...]:
        stat = path.stat()
        return (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
        )

    @classmethod
    def _validate_mapped_index(
        cls, artifact: Mapping[str, Any], size_bytes: int
    ) -> None:
        if artifact.get("format") != "aligned_raw_arrays/v1":
            raise ValueError("artifact does not use the mapped-array format")
        alignment = int(artifact.get("alignment", -1))
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("artifact alignment must be a positive power of two")
        descriptors = artifact.get("arrays")
        if not isinstance(descriptors, Mapping):
            raise ValueError("artifact omitted its mapped-array descriptors")
        if not cls._REQUIRED_FIELDS.issubset(descriptors):
            missing = sorted(cls._REQUIRED_FIELDS.difference(descriptors))
            raise ValueError(f"dataset artifact omitted fields: {missing}")
        rows = int(artifact.get("chunks", -1))
        width = int(artifact.get("chunk_size", -1))
        expected_layouts = {
            "chunk_index": (np.dtype("<i8"), (rows,)),
            "stream_start": (np.dtype("<i8"), (rows,)),
            "stream_stop": (np.dtype("<i8"), (rows,)),
            "input_ids": (np.dtype("<u2"), (rows, width)),
            "target_ids": (np.dtype("<u2"), (rows, width)),
            "valid_mask": (np.dtype(np.bool_), (rows, width)),
            "score_mask": (np.dtype(np.bool_), (rows, width)),
            "document_indices": (np.dtype("<i8"), (rows, width)),
            "document_offsets": (np.dtype("<i4"), (rows, width)),
            "patch_offsets": (np.dtype(np.int8), (rows, width)),
            "label_halo_id": (np.dtype("<u2"), (rows,)),
            "label_halo_valid": (np.dtype(np.bool_), (rows,)),
        }
        if rows <= 0 or width <= 0:
            raise ValueError("artifact row geometry is invalid")
        intervals: list[tuple[int, int, str]] = []
        for name, descriptor in descriptors.items():
            if not isinstance(name, str) or not isinstance(descriptor, Mapping):
                raise ValueError("artifact array descriptor is malformed")
            try:
                dtype = np.dtype(str(descriptor["dtype"]))
                raw_shape = descriptor["shape"]
                offset = int(descriptor["byte_offset"])
                byte_length = int(descriptor["byte_length"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"artifact array descriptor {name!r} is malformed"
                ) from error
            if dtype.hasobject or not isinstance(raw_shape, list) or any(
                type(value) is not int or value <= 0 for value in raw_shape
            ):
                raise ValueError(f"artifact array descriptor {name!r} is unsafe")
            expected_layout = expected_layouts.get(name)
            if expected_layout is not None and (
                dtype != expected_layout[0] or tuple(raw_shape) != expected_layout[1]
            ):
                raise ValueError(
                    f"artifact array descriptor {name!r} has the wrong layout"
                )
            expected_length = math.prod(raw_shape) * dtype.itemsize
            if (
                offset < 0
                or offset % alignment
                or byte_length != expected_length
                or offset + byte_length > size_bytes
            ):
                raise ValueError(f"artifact array descriptor {name!r} is out of bounds")
            intervals.append((offset, offset + byte_length, name))
        intervals.sort()
        for (_, left_stop, left_name), (right_start, _, right_name) in zip(
            intervals[:-1], intervals[1:], strict=True
        ):
            if left_stop > right_start:
                raise ValueError(
                    f"artifact arrays {left_name!r} and {right_name!r} overlap"
                )

    def _mapped_payload(
        self, artifact_index: int, names: Iterable[str] | None = None
    ) -> _MappedArtifactPayload:
        artifact = self.artifacts[artifact_index]
        descriptors = artifact["arrays"]
        selected_names = tuple(descriptors) if names is None else tuple(names)
        missing = set(selected_names).difference(descriptors)
        if missing:
            raise ValueError(f"dataset artifact omitted fields: {sorted(missing)}")
        return _MappedArtifactPayload(
            self.directory / str(artifact["path"]),
            descriptors,
            selected_names,
        )

    def _resolve_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        artifact_index = int(
            np.searchsorted(self._artifact_offsets, index, side="right") - 1
        )
        return artifact_index, index - int(self._artifact_offsets[artifact_index])

    def valid_count(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return int(self._valid_counts[index])

    def _verify_artifact_hash(self, artifact_index: int) -> None:
        artifact = self.artifacts[artifact_index]
        relative = Path(str(artifact["path"]))
        path = self.directory / relative
        identity = self._file_identity(path)
        prior_identity = self._verified_artifact_stats.get(artifact_index)
        if prior_identity is not None:
            if identity != prior_identity:
                raise ValueError(
                    f"dataset artifact changed after verification: {relative}"
                )
            if artifact_index in self._verified_artifacts:
                return
        if self._sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"dataset artifact sha256 mismatch: {relative}")
        if self._file_identity(path) != identity:
            raise ValueError(f"dataset artifact changed while hashing: {relative}")
        self._verified_artifacts.add(artifact_index)
        self._verified_artifact_stats[artifact_index] = identity

    def _verify_artifact_identity(self, artifact_index: int) -> None:
        """Pin an opened mmap to one cheap filesystem identity."""

        artifact = self.artifacts[artifact_index]
        relative = Path(str(artifact["path"]))
        identity = self._file_identity(self.directory / relative)
        prior_identity = self._verified_artifact_stats.get(artifact_index)
        if prior_identity is not None and identity != prior_identity:
            raise ValueError(f"dataset artifact changed after verification: {relative}")
        self._verified_artifact_stats[artifact_index] = identity

    def _verify_selected_rows(
        self, artifact_index: int, rows: Iterable[int]
    ) -> None:
        """Verify only mmap rows consumed by a trusted production batch."""

        if not self.trust_pinned_row_index:
            return
        self._verify_artifact_identity(artifact_index)
        pending = sorted(
            {
                int(row)
                for row in rows
                if not self._verified_rows[artifact_index][int(row)]
            }
        )
        if not pending:
            return
        arrays = self._arrays(artifact_index)
        for row in pending:
            digest = hashlib.sha256(b"byte_diffusion_mapped_row/v1\0")
            for name in sorted(arrays):
                digest.update(name.encode("ascii"))
                digest.update(b"\0")
                digest.update(memoryview(arrays[name][row : row + 1]).cast("B"))
            if digest.digest() != self._row_digests[artifact_index][row].tobytes():
                raise ValueError(
                    f"dataset artifact row sha256 mismatch: "
                    f"{self.artifacts[artifact_index]['path']} row {row}"
                )
            self._verified_rows[artifact_index][row] = True

    def _preceding_continuation(
        self, artifact_index: int
    ) -> tuple[int, int] | None:
        """Read the prior shard boundary needed to validate a shuffled shard."""

        if artifact_index == 0:
            return None
        previous = artifact_index - 1
        self._verify_artifact_hash(previous)
        with self._mapped_payload(
            previous,
            ("valid_mask", "input_ids", "document_indices", "document_offsets"),
        ) as payload:
            valid = payload["valid_mask"][-1]
            ids = payload["input_ids"][-1]
            documents = payload["document_indices"][-1]
            offsets = payload["document_offsets"][-1]
        columns = np.flatnonzero(valid)
        if columns.size == 0:
            raise ValueError("preceding artifact ends with an empty row")
        final = int(columns[-1])
        if int(ids[final]) == self.eot_id:
            return None
        return int(documents[final]), int(offsets[final]) + 1

    def _validate_deferred_artifact(
        self, artifact_index: int, arrays: Mapping[str, np.ndarray]
    ) -> None:
        """Perform the full semantic checks before a deferred shard is consumed."""

        if artifact_index in self._semantically_validated_artifacts:
            return
        artifact = self.artifacts[artifact_index]
        relative = artifact["path"]
        required = {
            "chunk_index",
            "stream_start",
            "stream_stop",
            "input_ids",
            "target_ids",
            "valid_mask",
            "score_mask",
            "document_indices",
            "document_offsets",
            "patch_offsets",
            "label_halo_id",
            "label_halo_valid",
        }
        if not required.issubset(arrays):
            missing = sorted(required.difference(arrays))
            raise ValueError(f"dataset artifact omitted fields: {missing}")
        valid = arrays["valid_mask"]
        ids = arrays["input_ids"]
        targets = arrays["target_ids"]
        score = arrays["score_mask"]
        documents = arrays["document_indices"]
        offsets = arrays["document_offsets"]
        patch_offsets = arrays["patch_offsets"]
        halo_ids = arrays["label_halo_id"]
        halo_valid = arrays["label_halo_valid"]
        if valid.ndim != 2 or valid.shape[1] != self.chunk_size:
            raise ValueError("dataset artifact input shape mismatch")
        for name, array in {
            "input ids": ids,
            "target ids": targets,
            "score mask": score,
            "document ids": documents,
            "document offsets": offsets,
            "patch offsets": patch_offsets,
        }.items():
            if array.shape != valid.shape:
                raise ValueError(f"artifact {name} do not align")
        if halo_valid.shape != (valid.shape[0],) or halo_ids.shape != (
            valid.shape[0],
        ):
            raise ValueError("artifact label halos do not align")
        row_vectors = {
            "chunk indices": arrays["chunk_index"],
            "stream starts": arrays["stream_start"],
            "stream stops": arrays["stream_stop"],
        }
        for name, array in row_vectors.items():
            if array.shape != (valid.shape[0],) or not np.issubdtype(
                array.dtype, np.integer
            ):
                raise ValueError(f"artifact {name} must be one integer per row")
        chunk_indices = arrays["chunk_index"]
        if chunk_indices.size > 1 and bool((np.diff(chunk_indices) != 1).any()):
            raise ValueError("artifact chunk indices are not contiguous")
        if int(chunk_indices[0]) != int(artifact.get("first_chunk_index", -1)) or int(
            chunk_indices[-1]
        ) != int(artifact.get("last_chunk_index", -1)):
            raise ValueError("artifact chunk-index bounds are false")
        if bool((arrays["stream_stop"] < arrays["stream_start"]).any()):
            raise ValueError("artifact stream interval is negative")
        invalid = ~valid
        if bool(score[invalid].any()):
            raise ValueError("invalid storage positions cannot be scored")
        if bool((ids[invalid] != PAD_ID).any()) or bool(
            (targets[invalid] != PAD_ID).any()
        ):
            raise ValueError("invalid storage positions must contain PAD")
        if (
            bool((documents[invalid] != -1).any())
            or bool((offsets[invalid] != -1).any())
            or bool((patch_offsets[invalid] != -1).any())
        ):
            raise ValueError("invalid storage positions cannot carry metadata")
        if bool((ids[valid] < 0).any()) or bool((ids[valid] >= MASK_ID).any()):
            raise ValueError("valid inputs must be clean atomic ids")
        if bool((targets[score] < 0).any()) or bool(
            (targets[score] >= MASK_ID).any()
        ):
            raise ValueError("scored targets must be clean atomic ids")
        if bool((targets[valid & ~score] != PAD_ID).any()):
            raise ValueError("unscored positions must contain PAD targets")

        valid_ids = ids[valid]
        observed_exposure = {
            "chunks": int(valid.shape[0]),
            "valid_atomic_tokens": int(valid.sum()),
            "literal_atomic_tokens": int((valid_ids < 256).sum()),
            "special_atomic_tokens": int((valid_ids >= 256).sum()),
            "eot_atomic_tokens": int((valid_ids == self.eot_id).sum()),
            "physical_storage_positions": int(ids.size),
            "storage_padding_tokens": int(ids.size - valid.sum()),
            "canvas512_eligible_positions": int(
                np.minimum(valid.sum(axis=1), 512).sum()
            ),
            "scored_ar_targets": int(score.sum()),
        }
        for name, observed in observed_exposure.items():
            if int(artifact.get(name, -1)) != observed:
                raise ValueError(
                    f"artifact {relative} has false {name}: claimed "
                    f"{artifact.get(name)}, observed {observed}"
                )
        row_valid_counts = valid.sum(axis=1, dtype=np.int64)
        row_physical_extents = np.where(
            valid,
            np.arange(self.chunk_size, dtype=np.int64)[None] + 1,
            0,
        ).max(axis=1)
        start = int(self._artifact_offsets[artifact_index])
        stop = int(self._artifact_offsets[artifact_index + 1])
        if not np.array_equal(
            row_valid_counts, self._valid_counts[start:stop]
        ):
            raise ValueError(f"artifact {relative} has false row valid counts")
        if not np.array_equal(
            row_physical_extents, self._physical_extents[start:stop]
        ):
            raise ValueError(f"artifact {relative} has false row physical extents")

        expected_continuation = (
            self._preceding_continuation(artifact_index)
            if self.document_aligned_pages
            else None
        )
        observed_document_starts: list[int] = []
        for row in range(valid.shape[0]):
            row_valid = valid[row]
            if bool(np.any((~row_valid[:-1]) & row_valid[1:])) and not (
                self.document_aligned_pages
            ):
                raise ValueError("artifact row is not one valid prefix")
            columns = np.flatnonzero(row_valid)
            if columns.size == 0:
                raise ValueError("artifact row has no valid atoms")
            document_set = set(int(value) for value in documents[row, row_valid])
            if not document_set or min(document_set) < 0:
                raise ValueError("artifact row has invalid document metadata")
            if (
                not self.dense_stream
                and not self.document_aligned_pages
                and len(document_set) != 1
            ):
                raise ValueError("isolated artifact row must contain one document")
            observed_document_starts.append(
                int((row_valid & (offsets[row] == 0)).sum())
            )
            if not self.document_aligned_pages:
                continue
            if int(columns[0]) != 0:
                raise ValueError("aligned page must start at physical column zero")
            row_documents = documents[row]
            row_offsets = offsets[row]
            row_patch_offsets = patch_offsets[row]
            starts = columns[
                np.r_[
                    True,
                    row_documents[columns[1:]] != row_documents[columns[:-1]],
                ]
            ]
            if not self.patching.variable and np.any(
                starts % self.patching.max_patch_size
            ):
                raise ValueError("document segment does not start on a patch boundary")
            first = int(row_documents[0]), int(row_offsets[0])
            if expected_continuation is None:
                if first[1] != 0:
                    raise ValueError("aligned page starts with an orphan continuation")
            elif first != expected_continuation:
                raise ValueError("aligned page does not continue the preceding document")
            start_offsets = row_offsets[starts]
            if np.any(start_offsets[1:] != 0):
                raise ValueError("mid-page document segment must start at offset zero")
            if self.patching.variable:
                _validate_variable_patch_row(
                    columns,
                    row_documents,
                    row_patch_offsets,
                    max_patch_size=self.patching.max_patch_size,
                )
                if bool((row_patch_offsets[starts] != 0).any()):
                    raise ValueError(
                        "document segment must begin on an entropy boundary"
                    )
            else:
                stride = self.patching.max_patch_size
                if np.any(
                    row_patch_offsets[columns] != row_offsets[columns] % stride
                ):
                    raise ValueError("document patch phase did not reset")
                patch_documents = row_documents.reshape(-1, stride)
                patch_present = patch_documents >= 0
                patch_min = np.where(
                    patch_present, patch_documents, np.iinfo(np.int64).max
                ).min(1)
                patch_max = np.where(patch_present, patch_documents, -1).max(1)
                if bool((patch_present.any(1) & (patch_min != patch_max)).any()):
                    raise ValueError("one patch mixes two documents")
            left = columns[:-1]
            right = columns[1:]
            same_document = row_documents[left] == row_documents[right]
            if bool(
                (
                    same_document
                    & (
                        (right != left + 1)
                        | (row_offsets[right] != row_offsets[left] + 1)
                    )
                ).any()
            ):
                raise ValueError("document atoms are not contiguous")
            changed_document = ~same_document
            maximum_gap = 0 if self.patching.variable else self.patching.max_patch_size - 1
            if bool((changed_document & ((right - left - 1) > maximum_gap)).any()):
                raise ValueError("inter-document alignment gap is not zero to three")
            if bool(
                (
                    changed_document
                    & ((ids[row, left] != self.eot_id) | score[row, left])
                ).any()
            ):
                raise ValueError("document boundary does not follow an unscored EOT")
            eot = row_valid & (ids[row] == self.eot_id)
            if bool(score[row, eot].any()):
                raise ValueError("terminal EOT cannot have an AR target")
            if bool((~score[row, row_valid & ~eot]).any()):
                raise ValueError("every non-EOT atom must carry its shifted AR target")
            scored_columns = np.flatnonzero(score[row])
            next_columns = np.minimum(scored_columns + 1, self.chunk_size - 1)
            has_next = scored_columns + 1 < self.chunk_size
            same_document_next = (
                has_next
                & row_valid[next_columns]
                & (row_documents[next_columns] == row_documents[scored_columns])
            )
            wrong_shift = same_document_next & (
                targets[row, scored_columns] != ids[row, next_columns]
            )
            if bool(wrong_shift.any()):
                raise ValueError("stored AR target is not next atom")
            valid_halo = (
                (scored_columns == columns[-1])
                & bool(halo_valid[row])
                & (targets[row, scored_columns] == halo_ids[row])
            )
            if bool((~same_document_next & ~valid_halo).any()):
                raise ValueError("scored page edge omitted its halo")
            start_positions = np.searchsorted(columns, starts)
            stop_positions = np.r_[start_positions[1:], len(columns)]
            for segment_index, stop_position in enumerate(stop_positions):
                final_column = int(columns[stop_position - 1])
                final_is_eot = ids[row, final_column] == self.eot_id
                is_final_segment = segment_index == len(stop_positions) - 1
                if not is_final_segment and not final_is_eot:
                    raise ValueError("mid-page document segment must end with EOT")
                if final_is_eot and (
                    score[row, final_column]
                    or (is_final_segment and bool(halo_valid[row]))
                ):
                    raise ValueError("terminal EOT cannot be scored or carry a halo")
            final_column = int(columns[-1])
            if ids[row, final_column] == self.eot_id:
                expected_continuation = None
            else:
                if (
                    (
                        not self.patching.variable
                        and final_column != self.chunk_size - 1
                    )
                    or not bool(halo_valid[row])
                ):
                    raise ValueError(
                        "unterminated document segment must end at an allowed row "
                        "boundary and carry a halo"
                    )
                expected_continuation = (
                    int(row_documents[final_column]),
                    int(row_offsets[final_column]) + 1,
                )
        if not np.array_equal(
            np.asarray(observed_document_starts, dtype=np.int32),
            self._row_document_starts[artifact_index],
        ):
            raise ValueError(f"artifact {relative} has false row document starts")
        if artifact_index == len(self.artifacts) - 1 and expected_continuation is not None:
            raise ValueError("dataset ends with an unterminated document")
        self._semantically_validated_artifacts.add(artifact_index)

    def _validate_batch_identity(
        self, artifact_index: int, arrays: Mapping[str, np.ndarray]
    ) -> None:
        """Cheap identity checks retained by the pinned production fast path."""

        artifact = self.artifacts[artifact_index]
        rows = int(artifact["chunks"])
        for name in ("chunk_index", "stream_start", "stream_stop"):
            array = arrays[name]
            if array.shape != (rows,) or not np.issubdtype(array.dtype, np.integer):
                raise ValueError(f"artifact {name} must be one integer per row")
        chunk_indices = arrays["chunk_index"]
        if rows > 1 and bool((np.diff(chunk_indices) != 1).any()):
            raise ValueError("artifact chunk indices are not contiguous")
        if int(chunk_indices[0]) != int(artifact.get("first_chunk_index", -1)) or int(
            chunk_indices[-1]
        ) != int(artifact.get("last_chunk_index", -1)):
            raise ValueError("artifact chunk-index bounds are false")
        if bool((arrays["stream_stop"] < arrays["stream_start"]).any()):
            raise ValueError("artifact stream interval is negative")

    @lru_cache(maxsize=4)
    def _arrays(self, artifact_index: int) -> dict[str, np.ndarray]:
        if self.trust_pinned_row_index:
            self._verify_artifact_identity(artifact_index)
        else:
            self._verify_artifact_hash(artifact_index)
        with self._mapped_payload(artifact_index) as payload:
            arrays = {name: payload[name] for name in payload.files}
        if self.defer_payload_validation and not self.trust_pinned_row_index:
            self._validate_deferred_artifact(artifact_index, arrays)
        return arrays

    @lru_cache(maxsize=4)
    def _batch_arrays(self, artifact_index: int) -> dict[str, np.ndarray]:
        """Load only fields used by native train/validation collation."""

        names = (
            "input_ids",
            "target_ids",
            "valid_mask",
            "score_mask",
            "document_offsets",
            "document_indices",
            "patch_offsets",
            "chunk_index",
            "stream_start",
            "stream_stop",
        )
        if (
            not self.trust_pinned_row_index
            and artifact_index not in self._semantically_validated_artifacts
        ):
            arrays = self._arrays(artifact_index)
            return {name: arrays[name] for name in names}
        if self.trust_pinned_row_index:
            all_arrays = self._arrays(artifact_index)
            arrays = {name: all_arrays[name] for name in names}
        else:
            self._verify_artifact_hash(artifact_index)
            with self._mapped_payload(artifact_index, names) as payload:
                arrays = {name: payload[name] for name in names}
        if self.trust_pinned_row_index:
            self._validate_batch_identity(artifact_index, arrays)
        return arrays

    def __getitem__(self, index: int | slice) -> PackedChunk | tuple[PackedChunk, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        artifact_index, row = self._resolve_index(index)
        self._verify_selected_rows(artifact_index, (row,))
        arrays = self._arrays(artifact_index)

        def values(name: str) -> tuple[Any, ...]:
            return tuple(arrays[name][row].tolist())

        return PackedChunk(
            chunk_index=int(arrays["chunk_index"][row]),
            stream_start=int(arrays["stream_start"][row]),
            stream_stop=int(arrays["stream_stop"][row]),
            input_ids=tuple(int(value) for value in values("input_ids")),
            target_ids=tuple(int(value) for value in values("target_ids")),
            valid_mask=tuple(bool(value) for value in values("valid_mask")),
            score_mask=tuple(bool(value) for value in values("score_mask")),
            document_indices=tuple(
                int(value) for value in values("document_indices")
            ),
            document_offsets=tuple(
                int(value) for value in values("document_offsets")
            ),
            patch_offsets=tuple(int(value) for value in values("patch_offsets")),
            label_halo_id=int(arrays["label_halo_id"][row]),
            label_halo_valid=bool(arrays["label_halo_valid"][row]),
            document_spans=(),
        )

    def _tensor_batch(
        self,
        indices: Sequence[int],
        *,
        include_identities: bool = True,
        integrity_scope: Literal["rows", "artifact"] = "rows",
    ) -> tuple[TrainingBatch, tuple[ValidationChunkIdentity, ...]]:
        if len(indices) == 0:
            raise ValueError("cannot materialize an empty training batch")
        global_rows = np.asarray(indices, dtype=np.int64)
        global_rows = np.where(global_rows < 0, global_rows + len(self), global_rows)
        if bool(((global_rows < 0) | (global_rows >= len(self))).any()):
            raise IndexError("mapped batch index is out of range")
        artifact_ids = np.searchsorted(
            self._artifact_offsets, global_rows, side="right"
        ).astype(np.int64, copy=False) - 1
        local_rows = global_rows - self._artifact_offsets[artifact_ids]
        unique_artifacts = np.unique(artifact_ids)
        for artifact in unique_artifacts:
            artifact_index = int(artifact)
            if integrity_scope == "artifact":
                # Training consumes an artifact's shuffled rows densely and
                # consecutively.  One OpenSSL-backed sequential file hash is
                # substantially cheaper than hashing the same payload through
                # a Python rows x fields loop, while enforcing the stronger
                # complete-artifact contract from the signed manifest.
                self._verify_artifact_hash(artifact_index)
            elif integrity_scope == "rows":
                selected_rows = local_rows[artifact_ids == artifact]
                self._verify_selected_rows(artifact_index, selected_rows)
            else:
                raise ValueError(f"unknown integrity scope {integrity_scope!r}")
        # Stage one mapped shard at a time into contiguous batch arrays. A
        # validation batch can intentionally sample one row from hundreds of
        # artifacts; retaining every artifact's eleven memmaps at once exceeds
        # ordinary process file-descriptor limits. Resolving one shard once and
        # copying all required fields preserves vectorized collation while the
        # small LRU bounds live mappings independently of validation size.
        collated_names = (
            "input_ids",
            "target_ids",
            "valid_mask",
            "score_mask",
            "document_offsets",
            "document_indices",
            "patch_offsets",
            "chunk_index",
            "stream_start",
        )
        collated: dict[str, np.ndarray] = {}
        for artifact in unique_artifacts:
            artifact_index = int(artifact)
            selected_positions = np.flatnonzero(artifact_ids == artifact)
            selected_rows = local_rows[selected_positions]
            arrays = self._batch_arrays(artifact_index)
            if not collated:
                collated = {
                    name: np.empty(
                        (global_rows.size, *arrays[name].shape[1:]),
                        dtype=arrays[name].dtype,
                    )
                    for name in collated_names
                }
            for name, result in collated.items():
                result[selected_positions] = arrays[name][selected_rows]
        if not collated:
            raise AssertionError("nonempty mapped batch produced no artifact rows")

        def rows(name: str) -> np.ndarray:
            return collated[name]

        valid_rows = rows("valid_mask")
        max_extent = int(self._physical_extents[global_rows].max())
        minimum_width = max(
            max_extent + self.branch_tail_halo,
            self.required_branch_bytes,
            1,
        )
        width = min(self.chunk_size, -(-minimum_width // 4) * 4)
        ids = torch.from_numpy(rows("input_ids")[:, :width]).to(torch.long)
        valid = torch.from_numpy(valid_rows[:, :width]).to(torch.bool)
        score = torch.from_numpy(rows("score_mask")[:, :width]).to(torch.bool)
        stored_targets = torch.from_numpy(rows("target_ids")[:, :width]).to(
            torch.long
        )
        ar_targets = torch.where(score, stored_targets, IGNORE_INDEX)
        document_offsets = torch.from_numpy(
            rows("document_offsets")[:, :width]
        ).to(torch.long)
        document_ids = torch.from_numpy(
            rows("document_indices")[:, :width]
        ).to(torch.long)
        patch_offsets = torch.from_numpy(
            rows("patch_offsets")[:, :width]
        ).to(torch.long)
        positions = (
            torch.arange(width, dtype=torch.long)[None].expand(global_rows.size, -1)
            if self.dense_stream
            else document_offsets.clamp_min(0)
        )
        stream_starts = rows("stream_start").reshape(-1)
        if self.dense_stream:
            # Interior document starts are already scored by the preceding
            # EOT's bridge target. Only the first atom in the whole stream has
            # no physical predecessor and therefore needs the virtual BOS.
            starts_stream = torch.from_numpy(stream_starts).eq(0)
            bos_row_indices = starts_stream.nonzero(as_tuple=False).flatten()
            bos_targets = ids[bos_row_indices, 0]
        else:
            starts_document = document_offsets.eq(0) & valid
            bos_locations = starts_document.nonzero(as_tuple=False)
            bos_row_indices = bos_locations[:, 0]
            bos_targets = ids[bos_locations[:, 0], bos_locations[:, 1]]
        fixed_layout = None
        variable_layout = None
        if not self.dense_stream:
            if self.patching.variable:
                variable_layout = build_variable_patch_layout(
                    valid,
                    document_ids,
                    document_offsets,
                    patch_offsets,
                    max_patch_size=self.patching.max_patch_size,
                )
            else:
                fixed_layout = _packed_document_layout(
                    valid,
                    document_ids,
                    positions,
                    patch_stride=self.patching.max_patch_size,
                )

        def fixed(index: int) -> Tensor | None:
            return None if fixed_layout is None else fixed_layout[index]

        def variable(name: str) -> Tensor | None:
            return (
                None
                if variable_layout is None
                else getattr(variable_layout, name)
            )

        def layout_value(name: str, fixed_index: int) -> Tensor | None:
            value = variable(name)
            return fixed(fixed_index) if value is None else value

        batch = TrainingBatch(
            ids=ids,
            valid=valid,
            ar_targets=ar_targets,
            bos_targets=bos_targets,
            positions=positions,
            full_valid=bool(valid.all()) and self.dense_stream,
            document_ids=document_ids,
            bos_row_indices=bos_row_indices,
            isolate_documents=not self.dense_stream,
            byte_indices=layout_value("byte_indices", 0),
            byte_cu_seqlens=layout_value("byte_cu_seqlens", 1),
            patch_indices=fixed(2),
            patch_cu_seqlens=layout_value("patch_cu_seqlens", 3),
            condition_patch_indices=layout_value("condition_patch_indices", 4),
            global_patch_sources=layout_value("global_patch_sources", 5),
            global_patch_positions=layout_value("global_patch_positions", 6),
            physical_to_global_patch_indices=layout_value(
                "physical_to_global_patch_indices", 7
            ),
            bos_condition_indices=layout_value("bos_condition_indices", 8),
            prior_condition_indices=fixed(9),
            patch_offsets=patch_offsets,
            patch_byte_cu_seqlens=variable("patch_byte_cu_seqlens"),
            max_patch_size=(
                self.patching.max_patch_size if self.patching.variable else None
            ),
            physical_patch_row_indices=variable("physical_patch_row_indices"),
            physical_patch_start_columns=variable(
                "physical_patch_start_columns"
            ),
            physical_patch_lengths=variable("physical_patch_lengths"),
            physical_patch_prior_condition_indices=variable(
                "physical_patch_prior_condition_indices"
            ),
        )
        identities = ()
        if include_identities:
            chunk_indices = rows("chunk_index").reshape(-1)
            identities = tuple(
                ValidationChunkIdentity(int(chunk_index), int(stream_start))
                for chunk_index, stream_start in zip(
                    chunk_indices, stream_starts, strict=True
                )
            )
        return batch, identities

    def training_batch(self, indices: Sequence[int]) -> TrainingBatch:
        """Materialize a tensor batch without a NumPy→Python→Torch round trip."""

        return self._tensor_batch(
            indices,
            include_identities=False,
            integrity_scope="artifact",
        )[0]

    def training_ar_units(self, indices: Sequence[int]) -> int:
        """Count exact scored AR and virtual-BOS targets without collation."""

        if len(indices) == 0:
            return 0
        global_rows = np.asarray(indices, dtype=np.int64)
        global_rows = np.where(global_rows < 0, global_rows + len(self), global_rows)
        if bool(((global_rows < 0) | (global_rows >= len(self))).any()):
            raise IndexError("mapped batch index is out of range")
        artifact_ids = np.searchsorted(
            self._artifact_offsets, global_rows, side="right"
        ).astype(np.int64, copy=False) - 1
        local_rows = global_rows - self._artifact_offsets[artifact_ids]
        total = 0
        for artifact in np.unique(artifact_ids):
            selected = local_rows[artifact_ids == artifact]
            arrays = self._batch_arrays(int(artifact))
            score = arrays["score_mask"][selected]
            total += int(score.sum(dtype=np.int64))
            if self.dense_stream:
                total += int((arrays["stream_start"][selected] == 0).sum())
            else:
                valid = arrays["valid_mask"][selected]
                offsets = arrays["document_offsets"][selected]
                total += int((valid & (offsets == 0)).sum(dtype=np.int64))
        return total

    def training_batch_groups(
        self,
        indices: Sequence[int],
        *,
        max_batch_size: int,
        physical_token_budget: int,
        sort_by_length: bool = False,
    ) -> list[tuple[int, ...]]:
        """Partition row ids by memory workload without loading row payloads."""

        if len(indices) == 0 or max_batch_size <= 0 or physical_token_budget <= 0:
            raise ValueError("adaptive batching requires positive rows and limits")
        if (
            not sort_by_length
            and max_batch_size
            * (self.chunk_size + self.required_branch_bytes)
            <= physical_token_budget
        ):
            return [
                tuple(indices[start : start + max_batch_size])
                for start in range(0, len(indices), max_batch_size)
            ]
        ordered = tuple(indices)
        if sort_by_length:
            # Rows selected for the optimizer update are unchanged.  Only
            # their microbatch grouping is reordered to avoid mixing the
            # shortest and longest cropped banks in one padded launch.
            ordered = tuple(
                sorted(ordered, key=self._valid_counts.__getitem__)
            )
        result: list[tuple[int, ...]] = []
        start = 0
        while start < len(ordered):
            count = min(max_batch_size, len(ordered) - start)
            while count > 1:
                candidate = ordered[start : start + count]
                max_extent = max(
                    self._physical_extents[index] for index in candidate
                )
                clean_width = max(
                    max_extent + self.branch_tail_halo,
                    self.required_branch_bytes,
                    1,
                )
                clean_width = min(self.chunk_size, -(-clean_width // 4) * 4)
                physical_width = clean_width + self.required_branch_bytes
                if count * physical_width <= physical_token_budget:
                    break
                count -= 1
            candidate = tuple(ordered[start : start + count])
            max_valid = max(self._physical_extents[index] for index in candidate)
            clean_width = min(
                self.chunk_size,
                -(
                    -max(
                        max_valid + self.branch_tail_halo,
                        self.required_branch_bytes,
                        1,
                    )
                    // 4
                )
                * 4,
            )
            if count * (clean_width + self.required_branch_bytes) > physical_token_budget:
                raise ValueError(
                    "one row exceeds the configured physical microbatch token budget"
                )
            result.append(candidate)
            start += count
        return result

    def training_batches(
        self,
        indices: Sequence[int],
        *,
        max_batch_size: int,
        physical_token_budget: int,
        sort_by_length: bool = False,
    ) -> list[TrainingBatch]:
        """Partition rows by their exact cropped-bank memory workload."""

        return [
            self.training_batch(group)
            for group in self.training_batch_groups(
                indices,
                max_batch_size=max_batch_size,
                physical_token_budget=physical_token_budget,
                sort_by_length=sort_by_length,
            )
        ]

    def validation_batch(
        self, indices: Sequence[int]
    ) -> tuple[TrainingBatch, tuple[ValidationChunkIdentity, ...]]:
        return self._tensor_batch(indices)

    def shuffled_indices(self, epoch_seed: int) -> np.ndarray:
        generator = np.random.default_rng(epoch_seed)
        group_order = generator.permutation(len(self.artifacts))

        def shuffled_group(group_index: int) -> np.ndarray:
            start = int(self._artifact_offsets[group_index])
            stop = int(self._artifact_offsets[group_index + 1])
            rows = np.arange(start, stop, dtype=np.int64)
            buckets = (self._valid_counts[start:stop] - 1) // 512
            _, inverse = np.unique(buckets, return_inverse=True)
            bucket_priority = generator.random(int(inverse.max()) + 1)
            row_priority = generator.random(rows.size)
            order = np.lexsort((row_priority, bucket_priority[inverse]))
            return rows[order]

        return np.concatenate(
            tuple(shuffled_group(int(group)) for group in group_order)
        )


class DeterministicSubsetChunkDataset(Sequence[PackedChunk]):
    """Evenly cover a large immutable split with a hash-bound fixed subset."""

    def __init__(self, source: Sequence[PackedChunk], limit: int) -> None:
        if limit <= 0 or limit >= len(source):
            raise ValueError("subset limit must lie strictly inside the source")
        self.source = source
        positions = np.arange(limit, dtype=np.int64)
        self.indices = (
            (2 * positions + 1) * len(source) // (2 * limit)
        )
        if np.unique(self.indices).size != limit:
            raise AssertionError("deterministic validation subset repeated a row")
        source_sha256 = getattr(source, "dataset_sha256", None)
        if source_sha256 is None:
            source_sha256 = packed_chunks_sha256(source)
        self.dataset_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "schema": "byte_diffusion_validation_subset/v1",
                    "source_sha256": source_sha256,
                    "source_rows": len(source),
                    "indices": self.indices.tolist(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def __len__(self) -> int:
        return len(self.indices)

    def valid_count(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        source_valid_count = getattr(self.source, "valid_count", None)
        if source_valid_count is None:
            return sum(self[index].valid_mask)
        return int(source_valid_count(self.indices[index]))

    def __getitem__(self, index: int | slice) -> PackedChunk | tuple[PackedChunk, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.source[int(self.indices[index])]

    def training_batch(self, indices: Sequence[int]) -> TrainingBatch:
        native = getattr(self.source, "training_batch", None)
        if native is None:
            return chunks_to_batch([self[index] for index in indices])
        mapped = np.take(self.indices, np.asarray(indices, dtype=np.int64))
        return native(mapped)

    def validation_batch(
        self, indices: Sequence[int]
    ) -> tuple[TrainingBatch, Sequence[PackedChunk | ValidationChunkIdentity]]:
        native = getattr(self.source, "validation_batch", None)
        if native is None:
            chunks = tuple(self[index] for index in indices)
            return chunks_to_batch(chunks), chunks
        mapped = np.take(self.indices, np.asarray(indices, dtype=np.int64))
        return native(mapped)


def load_data_directory(
    directory: Path,
    *,
    chunk_size: int,
    recipe: Recipe,
    required_branch_bytes: int = 0,
    branch_span_length: int = 0,
    validation_chunk_limit: int | None = None,
    require_challenge_validation: bool = False,
    expected_payload_sha256: str | None = None,
    expected_source_manifest_sha256: str | None = None,
    expected_patching_policy: PatchingPolicy | None = None,
    audit_payload_semantics: bool = False,
) -> tuple[
    AtomicIdManifest,
    Sequence[PackedChunk],
    Sequence[PackedChunk],
]:
    dataset_manifest_path = directory / "manifest.json"
    if not dataset_manifest_path.is_file():
        raise FileNotFoundError(
            f"canonical byte-diffusion dataset manifest is missing: "
            f"{dataset_manifest_path}"
        )
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    claimed_payload = dataset_manifest.get("payload_sha256")
    unsigned_manifest = dict(dataset_manifest)
    unsigned_manifest.pop("payload_sha256", None)
    observed_payload = hashlib.sha256(
        json.dumps(
            unsigned_manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if claimed_payload != observed_payload:
        raise ValueError("dataset manifest payload_sha256 mismatch")
    if (
        expected_payload_sha256 is not None
        and observed_payload != expected_payload_sha256
    ):
        raise ValueError(
            "dataset manifest differs from the pinned run contract: "
            f"expected {expected_payload_sha256}, observed {observed_payload}"
        )
    manifest = AtomicIdManifest.from_dict(dataset_manifest["atomic_vocabulary"])
    packing = dataset_manifest.get("packing", {})
    if int(packing.get("chunk_size", -1)) != chunk_size:
        raise ValueError(
            "requested chunk size does not match the built dataset manifest"
        )
    layout = packing.get("layout")
    if layout not in {
        "dense_eot_delimited_stream",
        "one_document_per_row",
        "document_aligned_pages",
    }:
        raise ValueError("dataset manifest has an unsupported packing layout")
    dense_stream = layout == "dense_eot_delimited_stream"
    document_aligned_pages = layout == "document_aligned_pages"
    if recipe != "causal_only":
        if required_branch_bytes <= 0:
            raise ValueError("diffusion data loading requires positive branch bytes")
        if dense_stream:
            raise ValueError(
                "diffusion requires document-local patch-aligned packing; "
                "dense EOT streams are a causal-only packing control"
            )
        if recipe == "canvas" and document_aligned_pages:
            raise ValueError(
                "document-aligned production pages require the BLT-D varlen "
                "branch path; legacy Canvas/Flex is a one-document control"
            )
    minimum = required_branch_bytes if recipe != "causal_only" else 0
    try:
        train_record = dataset_manifest["splits"]["train"]
        validation_record = dataset_manifest["splits"]["validation"]
    except (KeyError, TypeError) as error:
        raise ValueError("dataset manifest is missing train/validation splits") from error
    verify_source_manifest = (
        require_challenge_validation
        or expected_source_manifest_sha256 is not None
    )
    source_manifest: Mapping[str, Any] | None = None
    source_manifest_path: Path | None = None
    if verify_source_manifest:
        source_records = dataset_manifest.get("source_manifests")
        if not isinstance(source_records, list) or len(source_records) != 1:
            raise ValueError(
                "pinned data loading requires exactly one bound source manifest"
            )
        source_record = source_records[0]
        recorded_source_sha256 = str(source_record.get("sha256", ""))
        if (
            expected_source_manifest_sha256 is not None
            and recorded_source_sha256 != expected_source_manifest_sha256
        ):
            raise ValueError(
                "source manifest differs from the pinned run contract: "
                f"expected {expected_source_manifest_sha256}, "
                f"observed {recorded_source_sha256}"
            )
        source_manifest_path = Path(str(source_record.get("path", "")))
        if not source_manifest_path.is_absolute():
            source_manifest_path = Path(__file__).resolve().parents[2] / source_manifest_path
        if not source_manifest_path.is_file():
            raise ValueError("bound challenge source manifest is unavailable")
        source_bytes = source_manifest_path.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != recorded_source_sha256:
            raise ValueError("bound challenge source manifest changed after dataset build")
        source_manifest = json.loads(source_bytes)
    if require_challenge_validation:
        if source_manifest is None or source_manifest_path is None:
            raise AssertionError("challenge source manifest was not verified")
        declared_shards = source_manifest.get("challenge_validation_shards")
        challenge = source_manifest.get("challenge_validation")
        if not isinstance(declared_shards, list) or not declared_shards:
            raise ValueError("source manifest does not declare challenge validation shards")
        if not isinstance(challenge, dict):
            raise ValueError("source manifest does not declare challenge validation totals")

        def resolved(path: str, *, base: Path) -> Path:
            candidate = Path(path)
            return (candidate if candidate.is_absolute() else base / candidate).resolve()

        expected_paths = tuple(
            resolved(str(path), base=source_manifest_path.parent)
            for path in declared_shards
        )
        repository = Path(__file__).resolve().parents[2]
        observed_paths = tuple(
            resolved(str(path), base=repository)
            for path in validation_record.get("input_shards", ())
        )
        if observed_paths != expected_paths:
            raise ValueError(
                "byte validation shards differ from the source manifest's "
                "canonical challenge validation declaration"
            )
        if int(validation_record.get("complete_documents", -1)) != int(
            challenge.get("documents", -2)
        ):
            raise ValueError("byte validation document count is not challenge-canonical")
        if int(validation_record.get("unique_source_tokens_read", -1)) != int(
            challenge.get("tokens", -2)
        ):
            raise ValueError("byte validation source length is not challenge-canonical")
    patching = load_dataset_patching_spec(directory, dataset_manifest)
    if (
        expected_patching_policy is not None
        and patching.name != expected_patching_policy
    ):
        raise ValueError(
            "dataset patching policy differs from the run contract: "
            f"expected {expected_patching_policy}, observed {patching.name}"
        )
    artifact_schema = str(dataset_manifest.get("artifact_schema"))
    train = MappedPackedChunkDataset(
        directory,
        train_record["artifacts"],
        artifact_schema=artifact_schema,
        chunk_size=chunk_size,
        required_branch_bytes=minimum,
        branch_span_length=branch_span_length if recipe != "causal_only" else 0,
        split="train",
        dense_stream=dense_stream,
        document_aligned_pages=document_aligned_pages,
        patching=patching,
        eot_id=manifest.eot_id,
        defer_payload_validation=not audit_payload_semantics,
        trust_pinned_row_index=(
            not audit_payload_semantics and expected_payload_sha256 is not None
        ),
    )
    validation = MappedPackedChunkDataset(
        directory,
        validation_record["artifacts"],
        artifact_schema=artifact_schema,
        chunk_size=chunk_size,
        required_branch_bytes=minimum,
        branch_span_length=branch_span_length if recipe != "causal_only" else 0,
        split="validation",
        dense_stream=dense_stream,
        document_aligned_pages=document_aligned_pages,
        patching=patching,
        eot_id=manifest.eot_id,
        defer_payload_validation=not audit_payload_semantics,
        trust_pinned_row_index=(
            not audit_payload_semantics and expected_payload_sha256 is not None
        ),
    )
    for split_name, record, dataset in (
        ("train", train_record, train),
        ("validation", validation_record, validation),
    ):
        expected = {
            **dataset.exposure_summary,
            "chunks": dataset.exposure_summary["rows"],
        }
        claimed_bos_targets = (
            1 if dense_stream else int(record.get("complete_documents", -1))
        )
        observed_bos_targets = expected.get("bos_ar_targets")
        if (
            observed_bos_targets is not None
            and int(observed_bos_targets) != claimed_bos_targets
        ):
            raise ValueError(
                f"dataset split {split_name!r} document count disagrees with "
                f"observed BOS starts: claimed {claimed_bos_targets}, "
                f"observed {observed_bos_targets}"
            )
        expected["bos_ar_targets"] = claimed_bos_targets
        expected["total_ar_targets"] = (
            expected["scored_ar_targets"] + expected["bos_ar_targets"]
        )
        for name, value in expected.items():
            if name == "rows":
                continue
            if name not in record:
                raise ValueError(
                    f"dataset split {split_name!r} omitted exposure {name}"
                )
            if int(record[name]) != int(value):
                raise ValueError(
                    f"dataset split {split_name!r} has false {name}: "
                    f"claimed {record[name]}, observed {value}"
                )
        if expected["physical_storage_positions"] != len(dataset) * chunk_size:
            raise ValueError(f"dataset split {split_name!r} physical size is invalid")
        if expected["valid_atomic_tokens"] != (
            expected["literal_atomic_tokens"]
            + expected["special_atomic_tokens"]
        ):
            raise ValueError(f"dataset split {split_name!r} atom totals disagree")
        if expected["storage_padding_tokens"] != (
            expected["physical_storage_positions"]
            - expected["valid_atomic_tokens"]
        ):
            raise ValueError(f"dataset split {split_name!r} PAD totals disagree")
        if expected["total_ar_targets"] != expected["valid_atomic_tokens"]:
            raise ValueError(f"dataset split {split_name!r} AR coverage is incomplete")
    selected_validation: Sequence[PackedChunk] = validation
    if (
        validation_chunk_limit is not None
        and validation_chunk_limit < len(validation)
    ):
        selected_validation = DeterministicSubsetChunkDataset(
            validation, validation_chunk_limit
        )
    return manifest, train, selected_validation


def format_train_metric(metrics: StepMetrics, iterations: int) -> str:
    buckets = metrics.noise_bucket_counts
    bucket_nll = metrics.noise_bucket_nll
    bucket_accuracy = metrics.noise_bucket_accuracy
    return (
        f"step:{metrics.step}/{iterations} train_loss:{metrics.total:.6f} "
        f"train_time:{metrics.elapsed_ms:.3f}ms "
        f"ar_loss:{metrics.ar:.6f} diffusion_loss:{metrics.diffusion:.6f} "
        f"lr:{metrics.learning_rate:.8g} ar_targets:{metrics.ar_targets} "
        f"diffusion_targets:{metrics.diffusion_targets} "
        f"microsteps:{metrics.microsteps} max_microbatch:{metrics.max_microbatch} "
        f"max_physical_positions:{metrics.max_physical_positions} "
        f"preclip_grad_norm:{metrics.preclip_grad_norm:.6f} "
        f"grad_clip_scale:{metrics.grad_clip_scale:.6f} "
        f"noise_fraction:{metrics.mean_noise_fraction:.6f} "
        f"all_mask_fraction:{metrics.all_mask_fraction:.6f} "
        f"noise_q0:{buckets[1]} noise_q1:{buckets[2]} noise_q2:{buckets[3]} "
        f"noise_q3:{buckets[4]} noise_q4:{buckets[5]} noise_all:{buckets[6]} "
        f"noise_nll_q0:{bucket_nll[1]:.6f} "
        f"noise_nll_q1:{bucket_nll[2]:.6f} "
        f"noise_nll_q2:{bucket_nll[3]:.6f} "
        f"noise_nll_q3:{bucket_nll[4]:.6f} "
        f"noise_nll_q4:{bucket_nll[5]:.6f} "
        f"noise_nll_all:{bucket_nll[6]:.6f} "
        f"noise_acc_q0:{bucket_accuracy[1]:.6f} "
        f"noise_acc_q1:{bucket_accuracy[2]:.6f} "
        f"noise_acc_q2:{bucket_accuracy[3]:.6f} "
        f"noise_acc_q3:{bucket_accuracy[4]:.6f} "
        f"noise_acc_q4:{bucket_accuracy[5]:.6f} "
        f"noise_acc_all:{bucket_accuracy[6]:.6f}"
    )


def format_validation_metric(
    step: int,
    iterations: int,
    metrics: ValidationMetrics,
    training_time_ms: float,
    *,
    scope: Literal["proxy", "challenge"] = "proxy",
) -> str:
    if scope not in {"proxy", "challenge"}:
        raise ValueError(f"unknown validation scope {scope!r}")
    role_nll = metrics.diffusion_role_nll
    role_counts = metrics.diffusion_role_counts
    diffusion_primary = metrics.diffusion_elbo_proxy_bpb is not None
    primary_bpb = (
        metrics.diffusion_elbo_proxy_bpb if diffusion_primary else metrics.bpb
    )
    primary_loss = (
        metrics.diffusion_elbo_proxy_nats_per_block_atom
        if diffusion_primary
        else metrics.ar_loss
    )
    assert primary_bpb is not None and primary_loss is not None
    scoped_bpb = f"val_{scope}_bpb:{primary_bpb:.6f}"
    diffusion_elbo = (
        ""
        if metrics.diffusion_elbo_proxy_bpb is None
        else (
            "val_diffusion_elbo_proxy_bpb:"
            f"{metrics.diffusion_elbo_proxy_bpb:.6f} "
            "val_diffusion_elbo_proxy_nats_per_block_atom:"
            f"{metrics.diffusion_elbo_proxy_nats_per_block_atom:.6f} "
        )
    )
    return (
        f"step:{step}/{iterations} val_loss: {primary_loss:.6f} "
        f"val_bpb: {primary_bpb:.6f} train_time: {training_time_ms:.3f}ms "
        f"{scoped_bpb} "
        f"val_diffusion_primary:{int(diffusion_primary)} "
        f"val_ar_anchor_loss:{metrics.ar_loss:.6f} "
        f"val_ar_anchor_bpb:{metrics.atomic_bpb:.6f} "
        f"val_diffusion_loss:{metrics.diffusion_loss:.6f} "
        f"{diffusion_elbo}"
        f"val_diffusion_elbo_atoms:{metrics.diffusion_elbo_atoms:.3f} "
        f"val_ar_targets:{metrics.ar_targets} "
        f"val_literal_bytes:{metrics.literal_bytes} "
        f"val_special_targets:{metrics.special_targets} "
        f"val_diffusion_targets:{metrics.diffusion_targets} "
        f"val_diffusion_chunks:{metrics.diffusion_chunks} "
        f"val_eval_seconds:{metrics.elapsed_ms / 1_000:.6f} "
        f"diff_ascii_nll:{role_nll[0]:.6f} diff_lead_nll:{role_nll[1]:.6f} "
        f"diff_cont_nll:{role_nll[2]:.6f} diff_invalid_nll:{role_nll[3]:.6f} "
        f"diff_eot_nll:{role_nll[4]:.6f} diff_special_nll:{role_nll[5]:.6f} "
        f"diff_ascii_n:{role_counts[0]} diff_lead_n:{role_counts[1]} "
        f"diff_cont_n:{role_counts[2]} diff_eot_n:{role_counts[4]}"
    )
