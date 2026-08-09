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
from dataclasses import asdict, dataclass
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

from .config import AtomicVocabulary, ByteDiffusionConfig, CorruptionConfig, ModelMode
from .corruption import (
    CorruptedBatch,
    absorbing_rb,
    allmask_50,
    blt_bernoulli,
    uniform_replacement,
    whole_patch,
)
from .data import (
    AtomicDocument,
    AtomicIdManifest,
    DeterministicChunkCursor,
    PackedChunk,
    pack_documents,
    packed_chunks_sha256,
)
from .kernels import flash_sdpa_only
from .model import ByteDiffusionModel
from .objectives import (
    IGNORE_INDEX,
    ar_cross_entropy,
    blt_masked_loss,
    canvas_cross_entropy,
    cross_entropy_per_row,
    same_position_targets,
)


CHECKPOINT_SCHEMA = "byte_diffusion_training/v6"
CANONICAL_PRESET = "canvas512_scratch_v1"
Recipe = Literal["canvas", "blt_d", "causal_only"]
AttentionPolicy = Literal["flash_sdpa", "dense_reference"]
ObjectiveReduction = Literal["equal_mean", "paper_sum"]
CompileMode = Literal["default", "reduce-overhead", "max-autotune-no-cudagraphs"]


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
    warmdown_iters: int = 1_200
    run_id: str = "byte_diffusion"
    preset: str | None = None
    initialization_kind: Literal["scratch"] = "scratch"
    seed: int = 1_337
    recipe: Recipe = "canvas"
    corruption: CorruptionConfig = CorruptionConfig.canvas512()
    microbatch_per_rank: int = 8
    microbatch_token_budget: int = 278_528
    gradient_accumulation: int = 1
    global_batch_size: int | None = None
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
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

    def __post_init__(self) -> None:
        positive = {
            "iterations": self.iterations,
            "val_loss_every": self.val_loss_every,
            "train_log_every": self.train_log_every,
            "validation_chunks": self.validation_chunks,
            "microbatch_per_rank": self.microbatch_per_rank,
            "microbatch_token_budget": self.microbatch_token_budget,
            "gradient_accumulation": self.gradient_accumulation,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"run dimensions must be positive: {positive}")
        if self.global_batch_size is not None and self.global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive or None")
        if not self.run_id:
            raise ValueError("RUN_ID must be non-empty")
        if self.preset not in {None, CANONICAL_PRESET}:
            raise ValueError(f"unknown byte-diffusion preset {self.preset!r}")
        if self.initialization_kind != "scratch":
            raise ValueError("byte diffusion currently supports scratch lineages only")
        if not 0 <= self.warmdown_iters <= self.iterations:
            raise ValueError("WARMDOWN_ITERS must be in [0, ITERATIONS]")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.epsilon <= 0:
            raise ValueError("optimizer values are outside their valid range")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("AdamW betas must lie in [0, 1)")
        if self.lambda_ar < 0:
            raise ValueError("lambda_ar cannot be negative")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None")
        if self.recipe not in {"canvas", "blt_d", "causal_only"}:
            raise ValueError(f"unknown training recipe {self.recipe!r}")
        if not isinstance(self.corruption, CorruptionConfig):
            raise TypeError("corruption must be a CorruptionConfig")
        if self.objective_reduction not in {"equal_mean", "paper_sum"}:
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
        if self.objective_reduction == "paper_sum" and self.recipe != "blt_d":
            raise ValueError("paper_sum is defined only for the BLT-D recipe")
        if (
            self.objective_reduction == "paper_sum"
            and self.corruption.kind != "blt_bernoulli"
        ):
            raise ValueError("paper_sum requires BLT Bernoulli corruption")
        if self.corruption.kind == "uniform_replacement":
            raise ValueError(
                "uniform replacement requires the unimplemented reference "
                "self-conditioning forward and is not a production recipe"
            )
        if self.preset == CANONICAL_PRESET:
            expected = {
                "iterations": 2_000,
                "val_loss_every": 20,
                "train_log_every": 10,
                "validation_chunks": 2_048,
                "warmdown_iters": 1_200,
                "seed": 1_337,
                "initialization_kind": "scratch",
                "recipe": "canvas",
                "corruption": CorruptionConfig.canvas512(),
                "global_batch_size": 256,
                "learning_rate": 3e-4,
                "weight_decay": 0.1,
                "beta1": 0.9,
                "beta2": 0.95,
                "epsilon": 1e-8,
                "lambda_ar": 1.0,
                "objective_reduction": "equal_mean",
                "max_grad_norm": 1.0,
                "attention_policy": "flash_sdpa",
                "compile_model": True,
                "compile_dynamic_shapes": True,
                "activation_checkpointing": False,
                "compile_mode": "default",
                "cuda_graphs": False,
            }
            observed = {name: getattr(self, name) for name in expected}
            mismatches = {
                name: (observed[name], value)
                for name, value in expected.items()
                if observed[name] != value
            }
            if mismatches:
                raise ValueError(
                    f"preset {CANONICAL_PRESET!r} contract mismatch: {mismatches}"
                )

    @classmethod
    def from_env(cls, **overrides: Any) -> "TrainingRunConfig":
        iterations = _positive_int_env("ITERATIONS", 2_000)
        default_warmdown = 1_200 if iterations >= 2_000 else 0
        recipe = os.environ.get("BYTE_DIFFUSION_RECIPE", "canvas")
        microbatch = int(os.environ.get("BYTE_DIFFUSION_MICROBATCH", "8"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        global_batch = int(os.environ.get("BYTE_DIFFUSION_GLOBAL_BATCH", "256"))
        if microbatch <= 0 or world_size <= 0 or global_batch % world_size:
            raise ValueError(
                "WORLD_SIZE must be positive and divide BYTE_DIFFUSION_GLOBAL_BATCH"
            )
        local_batch = global_batch // world_size
        default_accumulation = -(-local_batch // microbatch)
        values: dict[str, Any] = {
            "iterations": iterations,
            "val_loss_every": _positive_int_env("VAL_LOSS_EVERY", 20),
            "train_log_every": _positive_int_env("TRAIN_LOG_EVERY", 10),
            "validation_chunks": _positive_int_env(
                "BYTE_DIFFUSION_VALIDATION_CHUNKS", 2_048
            ),
            "warmdown_iters": int(
                os.environ.get("WARMDOWN_ITERS", str(default_warmdown))
            ),
            "run_id": os.environ.get("RUN_ID", "byte_diffusion"),
            "preset": os.environ.get("BYTE_DIFFUSION_PRESET") or None,
            "seed": int(os.environ.get("SEED", "1337")),
            "recipe": recipe,
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
            "objective_reduction": os.environ.get(
                "BYTE_DIFFUSION_OBJECTIVE_REDUCTION",
                "paper_sum" if recipe == "blt_d" else "equal_mean",
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
        }
        default_kind = "blt_bernoulli" if recipe == "blt_d" else "absorbing_rb"
        default_length = 16 if recipe == "blt_d" else 512
        default_branches = 32 if recipe == "blt_d" else 1
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
        if result.preset == CANONICAL_PRESET and result.global_batch_size != 256:
            raise ValueError(
                f"preset {CANONICAL_PRESET!r} requires global batch 256, "
                f"observed {result.global_batch_size}"
            )
        if (
            result.global_batch_size is None
            or result.global_batch_size % world_size
        ):
            raise ValueError("configured global batch must divide WORLD_SIZE")
        result_local_batch = result.global_batch_size // world_size
        required_accumulation = -(
            -result_local_batch // result.microbatch_per_rank
        )
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

    def to(self, device: torch.device) -> "TrainingBatch":
        return TrainingBatch(
            ids=self.ids.to(device),
            valid=self.valid.to(device),
            ar_targets=self.ar_targets.to(device),
            bos_targets=self.bos_targets.to(device),
            positions=self.positions.to(device),
            full_valid=self.full_valid,
        )


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
    block_length: int
    t: Tensor | None
    sampling_weight: Tensor | None = None


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
    ar_targets: int
    literal_bytes: int
    special_targets: int
    diffusion_targets: int
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


def chunks_to_batch(chunks: Sequence[PackedChunk]) -> TrainingBatch:
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
    bos_targets = torch.full((len(chunks),), IGNORE_INDEX, dtype=torch.long)
    for row, chunk in enumerate(chunks):
        first_valid = next(
            (index for index, is_valid in enumerate(chunk.valid_mask) if is_valid),
            None,
        )
        if first_valid is not None and chunk.document_offsets[first_valid] == 0:
            bos_targets[row] = chunk.input_ids[first_valid]
    return TrainingBatch(
        ids,
        valid,
        ar_targets,
        bos_targets,
        positions,
        all(all(chunk.valid_mask) for chunk in chunks),
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

    indices: list[int] = []
    for global_offset in range(local_count * context.world_size):
        index = cursor.next_index()
        if global_offset % context.world_size == context.rank:
            indices.append(index)
    if len(indices) != local_count:
        raise AssertionError("distributed batch sharding produced the wrong size")
    native = getattr(cursor.chunks, "training_batch", None)
    if native is not None:
        return native(indices)
    return chunks_to_batch([cursor.chunk_at(index) for index in indices])


def take_distributed_indices(
    cursor: DeterministicChunkCursor,
    local_count: int,
    context: DistributedContext,
) -> list[int]:
    """Advance one global sample order and return this rank's row indices."""

    indices: list[int] = []
    for global_offset in range(local_count * context.world_size):
        index = cursor.next_index()
        if global_offset % context.world_size == context.rank:
            indices.append(index)
    if len(indices) != local_count:
        raise AssertionError("distributed row sharding produced the wrong size")
    return indices


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
) -> tuple[Tensor, Tensor]:
    """Sample fixed-patch BLT starts iid and return Horvitz--Thompson weights."""

    if valid.dtype != torch.bool or valid.ndim != 2 or count <= 0:
        raise ValueError("BLT start sampling requires boolean rows and count > 0")
    # Fixed patch zero plays Fast BLT's excluded first patch. Every later patch
    # containing at least one atom is an eligible block origin; end blocks may
    # extend into PAD, exactly as in the reference construction.
    eligible = torch.div(
        valid.sum(1).sub(1).clamp_min(0), patch_stride, rounding_mode="floor"
    )
    uniforms = torch.rand(
        (valid.shape[0], count),
        device=valid.device,
        generator=generator,
        dtype=torch.float32,
    )
    patch_index = 1 + (uniforms * eligible[:, None].clamp_min(1)).floor().long()
    starts = torch.where(eligible[:, None] > 0, patch_index, torch.zeros_like(patch_index))
    return starts * patch_stride, eligible.to(torch.float32) / count


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
) -> BltCorruptionPlan:
    block_length = config.canvas_length
    blocks = config.branches_per_row
    starts, sampling_weight = sample_blt_patch_starts(
        batch.valid,
        count=blocks,
        patch_stride=config.patch_stride,
        generator=generator,
    )
    clean_blocks = _gather_spans(
        batch.ids,
        starts,
        block_length,
        fill_value=vocab.pad_id,
    )
    block_valid = _gather_spans(
        batch.valid, starts, block_length, fill_value=False
    )
    block_valid &= sampling_weight[:, None, None] > 0
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
        block_length,
        corrupted.t,
        sampling_weight,
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
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        del block_length
        bos_logits = self.model.forward_bos_logits(
            clean_ids.shape[0],
            device=clean_ids.device,
            allow_dense_reference=not clean_ids.is_cuda,
        )
        if noisy_ids is None:
            ar_logits = self.model.forward_ar_varlen(
                clean_ids,
                valid,
                positions=positions,
                allow_dense_reference=not clean_ids.is_cuda,
                assume_full_clean=assume_full_clean,
            ).logits
            return ar_logits, None, bos_logits
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
            )
            return output.clean_logits, output.branch_logits, bos_logits
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
            )
            return output.clean_logits, output.branch_logits, bos_logits
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
    ) -> None:
        if device.type == "cuda":
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(device)
        elif not run_config.allow_cpu_reference:
            raise RuntimeError("CPU trainer is available only as an explicit reference")
        self.model_config = model.config
        self.run_config = run_config
        self.train_cursor = train_cursor
        self.validation_chunks = validation_chunks
        lazy_validation_digest = getattr(validation_chunks, "dataset_sha256", None)
        self.validation_sha256 = (
            str(lazy_validation_digest)
            if lazy_validation_digest is not None
            else packed_chunks_sha256(validation_chunks)
        )
        self.device = device
        self.distributed = distributed
        self.atomic_manifest = atomic_manifest or AtomicIdManifest.reference()
        self.dataset_provenance = dict(dataset_provenance or {})
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
                fullgraph=False,
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
                fullgraph=False,
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
        self.optimizer = create_optimizer(self.joint.parameters(), run_config, device)
        if run_config.cuda_graphs:
            if device.type != "cuda" or not run_config.compile_model:
                raise ValueError("CUDA graphs require CUDA plus compiled execution")
            for parameter in self.joint.parameters():
                parameter.grad = torch.zeros_like(parameter)
        self.corruption_generator = torch.Generator(device=device)
        self.corruption_generator.manual_seed(
            run_config.seed + 1_000_003 * distributed.rank
        )
        self.completed_steps = 0
        self.training_time_ms = 0.0

    def _set_learning_rate(self) -> float:
        multiplier = learning_rate_multiplier(
            self.completed_steps,
            self.run_config.iterations,
            self.run_config.warmdown_iters,
        )
        learning_rate = self.run_config.learning_rate * multiplier
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def measure_gradient_interference(
        self, chunks: Sequence[PackedChunk]
    ) -> GradientInterferenceMetrics:
        """Run the named fixed-batch AR/diffusion shared-trunk diagnostic."""

        if self.run_config.recipe == "causal_only":
            raise ValueError("gradient interference requires a diffusion objective")
        generator_state = self.corruption_generator.get_state().clone()
        self.corruption_generator.manual_seed(self.run_config.seed + 7_919)
        batch = chunks_to_batch(chunks).to(self.device)
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
                losses = self._compute_loss(batch)
            shared = tuple(self.joint.model.global_blocks.parameters())
            metrics = gradient_interference_metrics(
                losses.ar, losses.diffusion, shared
            )
        finally:
            self.corruption_generator.set_state(generator_state)
        return metrics

    def _compute_loss(self, batch: TrainingBatch) -> StepLoss:
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
            blt_plan = prepare_blt_corruption(
                batch,
                self.run_config.corruption,
                self.model_config.vocab,
                self.corruption_generator,
            )
            noisy_ids = blt_plan.noisy_blocks
            noisy_valid = blt_plan.branch_valid
            starts = blt_plan.block_starts
            block_length = blt_plan.block_length
            diffusion_mode = int(ModelMode.BLT_D)

        ar_logits_expanded, diffusion_logits, bos_logits = self.forward_model(
            batch.ids,
            batch.valid,
            batch.positions,
            noisy_ids,
            diffusion_mode,
            noisy_valid,
            starts,
            block_length,
            batch.full_valid,
        )
        ar_rows = cross_entropy_per_row(ar_logits_expanded, batch.ar_targets)
        bos_rows = cross_entropy_per_row(
            bos_logits[:, None, :], batch.bos_targets[:, None]
        )
        ar_row_totals = ar_rows.total + bos_rows.total
        ar_row_counts = ar_rows.count + bos_rows.count
        ar_count = ar_row_counts.sum()
        ar = ar_row_totals.sum() / ar_count.clamp_min(1).to(ar_row_totals.dtype)
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
                diffusion_logits.argmax(-1).eq(diffusion_targets)
                & canvas_plan.active
            ).flatten(0, 1).sum(1)
        elif blt_plan is not None:
            diffusion_targets = same_position_targets(
                blt_plan.clean_blocks.flatten(0, 1),
                blt_plan.active.flatten(0, 1),
                output_size=self.model_config.vocab.output_size,
            ).view_as(blt_plan.clean_blocks)
            if self.run_config.corruption.kind == "blt_bernoulli":
                if blt_plan.t is None:
                    raise AssertionError("BLT corruption omitted t")
                blt_objective = blt_masked_loss(
                    diffusion_logits.flatten(1, 2),
                    diffusion_targets.flatten(1, 2),
                    blt_plan.active.flatten(1, 2),
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
            diagnostic_rows = cross_entropy_per_row(
                diffusion_logits, diffusion_targets, active=blt_plan.active
            )
            noise_nll = diagnostic_rows.total.flatten()
            noise_correct = (
                diffusion_logits.argmax(-1).eq(diffusion_targets)
                & blt_plan.active
            ).flatten(0, 1).sum(1)
        else:
            raise AssertionError("mixed objective omitted its corruption plan")
        total_loss = diffusion + self.run_config.lambda_ar * ar
        if (
            blt_plan is not None
            and self.run_config.corruption.kind == "blt_bernoulli"
            and self.run_config.objective_reduction == "paper_sum"
        ):
            # Fast-BLT equations 5--7: summed clean CE plus the 1/t-weighted
            # masked sum for each clean row, followed by one batch mean.
            if "blt_objective" not in locals():
                raise AssertionError("paper_sum omitted the BLT objective")
            total_loss = (
                blt_objective.per_row
                * (
                    blt_plan.sampling_weight
                    if blt_plan.sampling_weight is not None
                    else 1.0
                )
                + self.run_config.lambda_ar * ar_row_totals
            ).mean()
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

    def run_update(self) -> StepMetrics:
        if self.completed_steps >= self.run_config.iterations:
            raise RuntimeError("planned training updates are already complete")
        learning_rate = self._set_learning_rate()
        self.optimizer.zero_grad(set_to_none=not self.run_config.cuda_graphs)
        if self.run_config.global_batch_size is None:
            local_rows = (
                self.run_config.microbatch_per_rank
                * self.run_config.gradient_accumulation
            )
        else:
            if self.run_config.global_batch_size % self.distributed.world_size:
                raise ValueError("global batch must divide evenly across ranks")
            local_rows = (
                self.run_config.global_batch_size // self.distributed.world_size
            )
        indices = take_distributed_indices(
            self.train_cursor, local_rows, self.distributed
        )
        native_batches = getattr(self.train_cursor.chunks, "training_batches", None)
        if native_batches is None:
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
            )
        local_ar_units = sum(
            int(batch.ar_targets.ne(IGNORE_INDEX).sum())
            + int(batch.bos_targets.ne(IGNORE_INDEX).sum())
            for batch in cpu_batches
        )
        local_objective_units_by_batch: list[int]
        if self.run_config.recipe == "canvas":
            local_objective_units_by_batch = [
                int(canvas_objective_units(batch.valid, self.run_config.corruption))
                for batch in cpu_batches
            ]
        elif self.run_config.recipe == "blt_d":
            local_objective_units_by_batch = [
                batch.ids.shape[0] for batch in cpu_batches
            ]
        else:
            local_objective_units_by_batch = [0] * len(cpu_batches)
        local_diffusion_units = sum(local_objective_units_by_batch)
        global_units = torch.tensor(
            [local_ar_units, local_diffusion_units],
            dtype=torch.float32,
            device=self.device,
        )
        if dist.is_initialized():
            dist.all_reduce(global_units)
        world_scale = float(self.distributed.world_size)
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
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        for microstep, (cpu_batch, local_objective_units) in enumerate(
            zip(cpu_batches, local_objective_units_by_batch, strict=True)
        ):
            if self.run_config.cuda_graphs:
                torch.compiler.cudagraph_mark_step_begin()
            batch = cpu_batch.to(self.device)
            synchronize = microstep + 1 == len(cpu_batches)
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
                losses = self._compute_loss(batch)
                if (
                    self.run_config.recipe == "blt_d"
                    and self.run_config.objective_reduction == "paper_sum"
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
            canvas_count += losses.noise_eligible.numel()
            if losses.noise_eligible.numel():
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

        if self.run_config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.joint.parameters(), self.run_config.max_grad_norm
            )
        self.optimizer.step()
        self.joint.model.enforce_padding_invariant()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        self.training_time_ms += elapsed_ms
        self.completed_steps += 1
        values = torch.stack((objective_total, ar_numerator, diffusion_numerator))
        counts = torch.stack(
            (
                ar_target_tensor,
                diffusion_target_tensor,
                masked_total,
                eligible_total,
                all_mask_canvases,
                canvas_count,
            )
        )
        if dist.is_initialized():
            dist.all_reduce(values)
            dist.all_reduce(counts)
            dist.all_reduce(noise_bucket_counts)
            dist.all_reduce(noise_bucket_targets)
            dist.all_reduce(noise_bucket_correct)
            dist.all_reduce(noise_bucket_nll)
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
            microsteps=len(cpu_batches),
            max_microbatch=max(batch.ids.shape[0] for batch in cpu_batches),
            max_physical_positions=max(
                batch.ids.shape[0]
                * (
                    batch.ids.shape[1]
                    + (
                        self.run_config.corruption.corrupted_positions_per_row
                        if self.run_config.recipe != "causal_only"
                        else 0
                    )
                )
                for batch in cpu_batches
            ),
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
        was_training = self.joint.training
        self.joint.eval()
        # [AR NLL, diffusion NLL, AR count, literal count, special count,
        # diffusion count]. Keep the whole validation reduction on device and
        # synchronize exactly once after optional DDP aggregation.
        totals = torch.zeros(6, dtype=torch.float64, device=self.device)
        role_nll = torch.zeros(6, dtype=torch.float64, device=self.device)
        role_counts = torch.zeros(6, dtype=torch.long, device=self.device)
        try:
            local_indices = range(
                self.distributed.rank,
                len(self.validation_chunks),
                self.distributed.world_size,
            )
            for start in range(
                0, len(local_indices), self.run_config.microbatch_per_rank
            ):
                positions = local_indices[
                    start : start + self.run_config.microbatch_per_rank
                ]
                native_batch = getattr(
                    self.validation_chunks, "validation_batch", None
                )
                if native_batch is None:
                    chunks: Sequence[PackedChunk | ValidationChunkIdentity] = [
                        self.validation_chunks[index] for index in positions
                    ]
                    cpu_batch = chunks_to_batch(chunks)  # type: ignore[arg-type]
                else:
                    cpu_batch, chunks = native_batch(list(positions))
                batch = cpu_batch.to(self.device)
                noisy = noisy_valid = starts_tensor = None
                diffusion_active = diffusion_clean = None
                block_length = 0
                diffusion_mode = -1
                if self.run_config.recipe == "canvas":
                    starts_tensor = sample_validation_starts(
                        batch.valid,
                        chunks,
                        span_length=self.run_config.corruption.canvas_length,
                        count=self.run_config.corruption.branches_per_row,
                        patch_stride=self.run_config.corruption.patch_stride,
                        seed=self.run_config.seed,
                    )
                    diffusion_clean = _gather_spans(
                        batch.ids,
                        starts_tensor,
                        self.run_config.corruption.canvas_length,
                    )
                    noisy_valid = _gather_spans(
                        batch.valid,
                        starts_tensor,
                        self.run_config.corruption.canvas_length,
                    )
                    noisy = torch.where(
                        noisy_valid,
                        self.model_config.vocab.mask_id,
                        diffusion_clean,
                    )
                    diffusion_active = noisy_valid
                    diffusion_mode = int(ModelMode.CANVAS)
                elif self.run_config.recipe == "blt_d":
                    block_length = self.run_config.corruption.canvas_length
                    starts_tensor = sample_validation_starts(
                        batch.valid,
                        chunks,
                        span_length=block_length,
                        count=self.run_config.corruption.branches_per_row,
                        patch_stride=self.run_config.corruption.patch_stride,
                        seed=self.run_config.seed,
                    )
                    diffusion_clean = _gather_spans(
                        batch.ids, starts_tensor, block_length
                    )
                    noisy_valid = _gather_spans(
                        batch.valid, starts_tensor, block_length
                    )
                    noisy = torch.where(
                        noisy_valid,
                        self.model_config.vocab.mask_id,
                        diffusion_clean,
                    )
                    diffusion_active = noisy_valid
                    diffusion_mode = int(ModelMode.BLT_D)
                autocast = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.device.type == "cuda"
                    else nullcontext()
                )
                with attention_context(
                    self.run_config.attention_policy,
                    self.device,
                    allow_cpu_reference=self.run_config.allow_cpu_reference,
                ), autocast:
                    ar_logits, diffusion_logits, bos_logits = self.validation_model(
                        batch.ids,
                        batch.valid,
                        batch.positions,
                        noisy,
                        diffusion_mode,
                        noisy_valid,
                        starts_tensor,
                        block_length,
                        batch.full_valid,
                    )
                ar_rows = cross_entropy_per_row(ar_logits, batch.ar_targets)
                bos_rows = cross_entropy_per_row(
                    bos_logits[:, None, :], batch.bos_targets[:, None]
                )
                totals[0] += (
                    ar_rows.total.double().sum() + bos_rows.total.double().sum()
                )
                totals[2] += ar_rows.count.sum() + bos_rows.count.sum()
                selected = batch.ar_targets[batch.ar_targets != IGNORE_INDEX]
                selected_bos = batch.bos_targets[
                    batch.bos_targets != IGNORE_INDEX
                ]
                totals[3] += (selected < 256).sum() + (selected_bos < 256).sum()
                totals[4] += (selected >= 256).sum() + (selected_bos >= 256).sum()
                if diffusion_logits is not None:
                    if diffusion_active is None or diffusion_clean is None:
                        raise AssertionError("validation diffusion metadata is missing")
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
                    diff_rows = cross_entropy_per_row(
                        diffusion_logits, targets, active=diffusion_active
                    )
                    totals[1] += diff_rows.total.double().sum()
                    totals[5] += diff_rows.count.sum()
                    active_targets = targets[diffusion_active]
                    active_logits = diffusion_logits[diffusion_active].float()
                    per_target_nll = F.cross_entropy(
                        active_logits, active_targets, reduction="none"
                    ).to(torch.float64)
                    roles = torch.full_like(active_targets, 3)
                    roles = torch.where(active_targets <= 0x7F, 0, roles)
                    roles = torch.where(
                        (active_targets >= 0xC2) & (active_targets <= 0xF4),
                        1,
                        roles,
                    )
                    roles = torch.where(
                        (active_targets >= 0x80) & (active_targets <= 0xBF),
                        2,
                        roles,
                    )
                    roles = torch.where(
                        active_targets == self.atomic_manifest.eot_id, 4, roles
                    )
                    roles = torch.where(
                        (active_targets >= 256)
                        & (active_targets != self.atomic_manifest.eot_id),
                        5,
                        roles,
                    )
                    role_nll.scatter_add_(0, roles, per_target_nll)
                    role_counts += torch.bincount(roles, minlength=6)
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
        diffusion_loss = float(totals[1] / diff_count) if diff_count else 0.0
        return ValidationMetrics(
            ar_loss=ar_loss,
            # The challenge byte LUT charges every registered special atom,
            # including EOT, as one byte. Atomic byte targets therefore use
            # the complete scored-target count as the BPB denominator.
            bpb=float(totals[0]) / ar_count / math.log(2.0),
            atomic_bpb=ar_loss / math.log(2.0),
            diffusion_loss=diffusion_loss,
            ar_targets=ar_count,
            literal_bytes=literal_count,
            special_targets=special_count,
            diffusion_targets=diff_count,
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


class NpzPackedChunkDataset(Sequence[PackedChunk]):
    """Hash-verified, shard-cached view over canonical v2 NPZ artifacts.

    Only compact ``(artifact,row)`` indices are resident.  Array payloads are
    decompressed on demand into a small LRU, and epoch shuffles keep rows from
    one artifact adjacent so random training does not thrash compressed shards.
    """

    def __init__(
        self,
        directory: Path,
        artifacts: Sequence[Mapping[str, Any]],
        *,
        artifact_schema: str,
        chunk_size: int,
        required_branch_bytes: int,
        split: str,
    ) -> None:
        self.directory = directory
        self.artifacts = tuple(dict(artifact) for artifact in artifacts)
        self.artifact_schema = artifact_schema
        self.chunk_size = chunk_size
        self.required_branch_bytes = required_branch_bytes
        self.split = split
        selected: list[tuple[int, int]] = []
        valid_counts: list[int] = []
        groups: list[list[int]] = []
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
        for artifact_index, artifact in enumerate(self.artifacts):
            if artifact.get("schema") != artifact_schema:
                raise ValueError(f"{split} artifact schema mismatch")
            relative = Path(str(artifact["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("dataset artifact path escapes its directory")
            path = directory / relative
            if path.stat().st_size != int(artifact["size_bytes"]):
                raise ValueError(f"dataset artifact size mismatch: {relative}")
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    hasher.update(block)
            digest = hasher.hexdigest()
            if digest != artifact["sha256"]:
                raise ValueError(f"dataset artifact sha256 mismatch: {relative}")
            with np.load(path, allow_pickle=False) as payload:
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
                if valid.ndim != 2 or valid.shape[1] != chunk_size:
                    raise ValueError("dataset artifact input shape mismatch")
                if documents.shape != valid.shape:
                    raise ValueError("artifact document ids do not align")
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
                group: list[int] = []
                for row in range(valid.shape[0]):
                    row_valid = valid[row]
                    valid_count = int(row_valid.sum())
                    if bool(np.any((~row_valid[:-1]) & row_valid[1:])):
                        raise ValueError("v2 artifact row is not one valid prefix")
                    document_set = set(int(value) for value in documents[row, row_valid])
                    if len(document_set) != 1 or next(iter(document_set)) < 0:
                        raise ValueError("v2 artifact row must contain one document segment")
                    global_index = len(selected)
                    selected.append((artifact_index, row))
                    valid_counts.append(valid_count)
                    group.append(global_index)
                groups.append(group)
            identity_records.append(
                {
                    "path": str(relative),
                    "sha256": digest,
                    "rows": group,
                }
            )
        if not selected:
            raise ValueError(f"{split} has no rows satisfying the model contract")
        self._selected = tuple(selected)
        self._valid_counts = tuple(valid_counts)
        self._groups = tuple(tuple(group) for group in groups if group)
        self.exposure_summary = dict(exposure)
        self.dataset_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "schema": artifact_schema,
                    "split": split,
                    "chunk_size": chunk_size,
                    "required_branch_bytes": required_branch_bytes,
                    "artifacts": identity_records,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def __len__(self) -> int:
        return len(self._selected)

    @lru_cache(maxsize=4)
    def _arrays(self, artifact_index: int) -> dict[str, np.ndarray]:
        path = self.directory / str(self.artifacts[artifact_index]["path"])
        with np.load(path, allow_pickle=False) as payload:
            return {name: payload[name].copy() for name in payload.files}

    def __getitem__(self, index: int | slice) -> PackedChunk | tuple[PackedChunk, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        artifact_index, row = self._selected[index]
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
        self, indices: Sequence[int]
    ) -> tuple[TrainingBatch, tuple[ValidationChunkIdentity, ...]]:
        if not indices:
            raise ValueError("cannot materialize an empty training batch")
        selected = [self._selected[index] for index in indices]

        def rows(name: str) -> np.ndarray:
            artifact_ids = {artifact for artifact, _ in selected}
            if len(artifact_ids) == 1:
                artifact = selected[0][0]
                row_ids = np.fromiter(
                    (row for _, row in selected), dtype=np.int64, count=len(selected)
                )
                return np.ascontiguousarray(self._arrays(artifact)[name][row_ids])
            return np.stack(
                [self._arrays(artifact)[name][row] for artifact, row in selected]
            )

        valid_rows = rows("valid_mask")
        max_valid = int(valid_rows.sum(axis=1).max())
        minimum_width = max(max_valid, self.required_branch_bytes, 1)
        width = min(self.chunk_size, -(-minimum_width // 4) * 4)
        ids = torch.from_numpy(rows("input_ids")[:, :width]).to(torch.long)
        valid = torch.from_numpy(valid_rows[:, :width]).to(torch.bool)
        score = torch.from_numpy(rows("score_mask")[:, :width]).to(torch.bool)
        stored_targets = torch.from_numpy(rows("target_ids")[:, :width]).to(
            torch.long
        )
        ar_targets = torch.where(score, stored_targets, IGNORE_INDEX)
        positions = (
            torch.from_numpy(rows("document_offsets")[:, :width])
            .to(torch.long)
            .clamp_min_(0)
        )
        bos_targets = torch.full((len(indices),), IGNORE_INDEX, dtype=torch.long)
        starts_document = positions[:, 0].eq(0)
        bos_targets = torch.where(starts_document, ids[:, 0], bos_targets)
        batch = TrainingBatch(
            ids=ids,
            valid=valid,
            ar_targets=ar_targets,
            bos_targets=bos_targets,
            positions=positions,
            full_valid=bool(valid.all()),
        )
        chunk_indices = rows("chunk_index").reshape(-1)
        stream_starts = rows("stream_start").reshape(-1)
        identities = tuple(
            ValidationChunkIdentity(int(chunk_index), int(stream_start))
            for chunk_index, stream_start in zip(
                chunk_indices, stream_starts, strict=True
            )
        )
        return batch, identities

    def training_batch(self, indices: Sequence[int]) -> TrainingBatch:
        """Materialize a tensor batch without a NumPy→Python→Torch round trip."""

        return self._tensor_batch(indices)[0]

    def training_batches(
        self,
        indices: Sequence[int],
        *,
        max_batch_size: int,
        physical_token_budget: int,
    ) -> list[TrainingBatch]:
        """Partition rows by their exact cropped-bank memory workload."""

        if not indices or max_batch_size <= 0 or physical_token_budget <= 0:
            raise ValueError("adaptive batching requires positive rows and limits")
        result: list[TrainingBatch] = []
        start = 0
        while start < len(indices):
            count = min(max_batch_size, len(indices) - start)
            while count > 1:
                candidate = indices[start : start + count]
                max_valid = max(self._valid_counts[index] for index in candidate)
                clean_width = max(max_valid, self.required_branch_bytes, 1)
                clean_width = min(self.chunk_size, -(-clean_width // 4) * 4)
                physical_width = clean_width + self.required_branch_bytes
                if count * physical_width <= physical_token_budget:
                    break
                count -= 1
            candidate = indices[start : start + count]
            max_valid = max(self._valid_counts[index] for index in candidate)
            clean_width = min(
                self.chunk_size,
                -(-max(max_valid, self.required_branch_bytes, 1) // 4) * 4,
            )
            if count * (clean_width + self.required_branch_bytes) > physical_token_budget:
                raise ValueError(
                    "one row exceeds the configured physical microbatch token budget"
                )
            result.append(self.training_batch(candidate))
            start += count
        return result

    def validation_batch(
        self, indices: Sequence[int]
    ) -> tuple[TrainingBatch, tuple[ValidationChunkIdentity, ...]]:
        return self._tensor_batch(indices)

    def shuffled_indices(self, epoch_seed: int) -> tuple[int, ...]:
        generator = random.Random(epoch_seed)
        group_order = list(range(len(self._groups)))
        generator.shuffle(group_order)
        result: list[int] = []
        for group_index in group_order:
            buckets: dict[int, list[int]] = {}
            for row in self._groups[group_index]:
                bucket = (self._valid_counts[row] - 1) // 512
                buckets.setdefault(bucket, []).append(row)
            bucket_order = list(buckets)
            generator.shuffle(bucket_order)
            for bucket in bucket_order:
                rows = buckets[bucket]
                generator.shuffle(rows)
                result.extend(rows)
        return tuple(result)


class DeterministicSubsetChunkDataset(Sequence[PackedChunk]):
    """Evenly cover a large immutable split with a hash-bound fixed subset."""

    def __init__(self, source: Sequence[PackedChunk], limit: int) -> None:
        if limit <= 0 or limit >= len(source):
            raise ValueError("subset limit must lie strictly inside the source")
        self.source = source
        self.indices = tuple(
            ((2 * index + 1) * len(source)) // (2 * limit)
            for index in range(limit)
        )
        if len(set(self.indices)) != limit:
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
                    "indices": self.indices,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int | slice) -> PackedChunk | tuple[PackedChunk, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.source[self.indices[index]]

    def training_batch(self, indices: Sequence[int]) -> TrainingBatch:
        native = getattr(self.source, "training_batch", None)
        if native is None:
            return chunks_to_batch([self[index] for index in indices])
        return native([self.indices[index] for index in indices])

    def validation_batch(
        self, indices: Sequence[int]
    ) -> tuple[TrainingBatch, Sequence[PackedChunk | ValidationChunkIdentity]]:
        native = getattr(self.source, "validation_batch", None)
        if native is None:
            chunks = tuple(self[index] for index in indices)
            return chunks_to_batch(chunks), chunks
        return native([self.indices[index] for index in indices])


def load_data_directory(
    directory: Path,
    *,
    chunk_size: int,
    recipe: Recipe,
    required_branch_bytes: int = 0,
    validation_chunk_limit: int | None = None,
    require_challenge_validation: bool = False,
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
    if dataset_manifest.get("schema") != "byte_diffusion_dataset/v3":
        raise ValueError("training requires the byte-native v3 dataset")
    manifest = AtomicIdManifest.from_dict(dataset_manifest["atomic_vocabulary"])
    packing = dataset_manifest.get("packing", {})
    if int(packing.get("chunk_size", -1)) != chunk_size:
        raise ValueError(
            "requested chunk size does not match the built dataset manifest"
        )
    if int(packing.get("row_document_segments", -1)) != 1:
        raise ValueError("dataset rows are not isolated to one document segment")
    if recipe != "causal_only":
        if required_branch_bytes <= 0:
            raise ValueError("diffusion data loading requires positive branch bytes")
    minimum = required_branch_bytes if recipe != "causal_only" else 0
    try:
        train_record = dataset_manifest["splits"]["train"]
        validation_record = dataset_manifest["splits"]["validation"]
    except (KeyError, TypeError) as error:
        raise ValueError("dataset manifest is missing train/validation splits") from error
    if require_challenge_validation:
        source_records = dataset_manifest.get("source_manifests")
        if not isinstance(source_records, list) or len(source_records) != 1:
            raise ValueError(
                "challenge evaluation requires exactly one bound source manifest"
            )
        source_record = source_records[0]
        source_manifest_path = Path(str(source_record.get("path", "")))
        if not source_manifest_path.is_absolute():
            source_manifest_path = Path(__file__).resolve().parents[2] / source_manifest_path
        if not source_manifest_path.is_file():
            raise ValueError("bound challenge source manifest is unavailable")
        source_bytes = source_manifest_path.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != source_record.get("sha256"):
            raise ValueError("bound challenge source manifest changed after dataset build")
        source_manifest = json.loads(source_bytes)
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
    artifact_schema = str(dataset_manifest.get("artifact_schema"))
    train = NpzPackedChunkDataset(
        directory,
        train_record["artifacts"],
        artifact_schema=artifact_schema,
        chunk_size=chunk_size,
        required_branch_bytes=minimum,
        split="train",
    )
    validation = NpzPackedChunkDataset(
        directory,
        validation_record["artifacts"],
        artifact_schema=artifact_schema,
        chunk_size=chunk_size,
        required_branch_bytes=minimum,
        split="validation",
    )
    for split_name, record, dataset in (
        ("train", train_record, train),
        ("validation", validation_record, validation),
    ):
        expected = {
            **dataset.exposure_summary,
            "chunks": dataset.exposure_summary["rows"],
            "bos_ar_targets": int(record.get("complete_documents", -1)),
        }
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
    scoped_bpb = f"val_{scope}_bpb:{metrics.bpb:.6f}"
    return (
        f"step:{step}/{iterations} val_loss: {metrics.ar_loss:.6f} "
        f"val_bpb: {metrics.bpb:.6f} train_time: {training_time_ms:.3f}ms "
        f"{scoped_bpb} "
        f"val_atomic_bpb:{metrics.atomic_bpb:.6f} "
        f"val_diffusion_loss:{metrics.diffusion_loss:.6f} "
        f"val_ar_targets:{metrics.ar_targets} "
        f"val_literal_bytes:{metrics.literal_bytes} "
        f"val_special_targets:{metrics.special_targets} "
        f"val_diffusion_targets:{metrics.diffusion_targets} "
        f"diff_ascii_nll:{role_nll[0]:.6f} diff_lead_nll:{role_nll[1]:.6f} "
        f"diff_cont_nll:{role_nll[2]:.6f} diff_invalid_nll:{role_nll[3]:.6f} "
        f"diff_eot_nll:{role_nll[4]:.6f} diff_special_nll:{role_nll[5]:.6f} "
        f"diff_ascii_n:{role_counts[0]} diff_lead_n:{role_counts[1]} "
        f"diff_cont_n:{role_counts[2]} diff_eot_n:{role_counts[4]}"
    )
