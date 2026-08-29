"""Standalone training and validation for the reference-aligned I-DLM cell."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import nullcontext
import math
from pathlib import Path
import time
from typing import Mapping, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from checkpointing import atomic_torch_save

from .data import AtomicIdManifest, DeterministicChunkCursor
from .idlm import IDLMShiftedTargets, IDLMTrainingLayout, auto_balanced_loss, make_training_layout
from .idlm_model import IDLMForwardMetadata, IDLMModel, IDLMModelOutput
from .pipeline import DeviceBatchPrefetcher
from .training import TrainingBatch


IGNORE_INDEX = -100
CHECKPOINT_SCHEMA = "byte_idlm_training/v4"


class NativeBatchDataset(Protocol):
    """The zero-Python-row-roundtrip dataset interface used by this recipe."""

    def __len__(self) -> int: ...

    def training_batch(self, indices: Sequence[int]) -> TrainingBatch: ...


@dataclass(frozen=True)
class IDLMTrainingConfig:
    iterations: int = 2_000
    batch_size: int = 1
    global_batch_size: int = 249
    validation_batch_size: int = 2
    validation_rows: int = 2_048
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0
    warmdown_iters: int = 1_200
    seed: int = 1_337
    compile_model: bool = True
    # (optimizer-step start, stride). The final stride is the only production
    # serving stride; earlier entries are an explicit curriculum, never an
    # inference-time topology switch hidden from the checkpoint contract.
    stride_curriculum: tuple[tuple[int, int], ...] = ((0, 4),)

    def __post_init__(self) -> None:
        positive = {
            "iterations": self.iterations,
            "batch_size": self.batch_size,
            "global_batch_size": self.global_batch_size,
            "validation_batch_size": self.validation_batch_size,
            "validation_rows": self.validation_rows,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"I-DLM run dimensions must be positive: {positive}")
        if not 0 <= self.warmdown_iters <= self.iterations:
            raise ValueError("warmdown_iters must be in [0, iterations]")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.epsilon <= 0:
            raise ValueError("I-DLM optimizer values are invalid")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("I-DLM Adam betas must lie in [0, 1)")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None")
        if not self.stride_curriculum or self.stride_curriculum[0][0] != 0:
            raise ValueError("I-DLM stride curriculum must begin at optimizer step zero")
        starts = tuple(item[0] for item in self.stride_curriculum)
        strides = tuple(item[1] for item in self.stride_curriculum)
        if any(start < 0 or start >= self.iterations for start in starts):
            raise ValueError("I-DLM stride curriculum starts must lie inside the run")
        if any(right <= left for left, right in zip(starts, starts[1:])):
            raise ValueError("I-DLM stride curriculum starts must increase strictly")
        if any(stride <= 0 for stride in strides):
            raise ValueError("I-DLM curriculum strides must be positive")


@dataclass(frozen=True)
class IDLMBatch:
    layout: IDLMTrainingLayout
    targets: IDLMShiftedTargets
    bos_targets: Tensor
    metadata: IDLMForwardMetadata | None = None

    @classmethod
    def from_native(
        cls,
        batch: TrainingBatch,
        *,
        mask_id: int,
        pad_id: int,
        block_size: int,
    ) -> "IDLMBatch":
        if batch.document_ids is None:
            segment_ids = torch.zeros_like(batch.ids)
        else:
            segment_ids = batch.document_ids
        # Passing positions=None intentionally resets positions at every
        # visible packed-document segment.  Continuation pages have no cached
        # prefix in this standalone cell, so treating the page boundary as a
        # fresh RoPE origin is the exact full-replay behavior for that sample.
        layout = make_training_layout(
            batch.ids,
            batch.valid,
            mask_id=mask_id,
            pad_id=pad_id,
            block_size=block_size,
            segment_ids=segment_ids,
            positions=None,
        )
        # The canonical mapped corpus already carries exact shifted labels,
        # including the one-id halo at an artificial page boundary.  Reusing
        # those targets preserves complete exposure without peeking at input.
        targets = IDLMShiftedTargets(
            proposal=batch.ar_targets,
            clean=batch.ar_targets.clone(),
            ignore_index=IGNORE_INDEX,
        )
        metadata = None
        if batch.byte_indices is not None and batch.byte_cu_seqlens is not None:
            metadata = IDLMForwardMetadata(
                clean_indices=batch.byte_indices,
                clean_cu_seqlens=batch.byte_cu_seqlens,
                # Keep the FlashAttention launch geometry static across
                # batches.  The exact per-document lengths remain encoded in
                # ``clean_cu_seqlens``; this value is only a safe upper bound.
                # Reading ``lengths.max()`` into Python synchronized CUDA and
                # specialized compiled graphs to corpus-dependent values.
                max_clean_seqlen=layout.sequence_length,
            )
        return cls(
            layout=layout,
            targets=targets,
            bos_targets=batch.bos_targets,
            metadata=metadata,
        )

    def to(self, device: torch.device, *, non_blocking: bool = False) -> "IDLMBatch":
        layout = IDLMTrainingLayout(
            proposal_ids=self.layout.proposal_ids.to(device, non_blocking=non_blocking),
            clean_ids=self.layout.clean_ids.to(device, non_blocking=non_blocking),
            valid=self.layout.valid.to(device, non_blocking=non_blocking),
            positions=self.layout.positions.to(device, non_blocking=non_blocking),
            block_size=self.layout.block_size,
            segment_ids=(
                None
                if self.layout.segment_ids is None
                else self.layout.segment_ids.to(device, non_blocking=non_blocking)
            ),
        )
        return IDLMBatch(
            layout=layout,
            targets=IDLMShiftedTargets(
                self.targets.proposal.to(device, non_blocking=non_blocking),
                self.targets.clean.to(device, non_blocking=non_blocking),
                self.targets.ignore_index,
            ),
            bos_targets=self.bos_targets.to(device, non_blocking=non_blocking),
            metadata=(
                None
                if self.metadata is None
                else IDLMForwardMetadata(
                    clean_indices=self.metadata.clean_indices.to(
                        device, non_blocking=non_blocking
                    ),
                    clean_cu_seqlens=self.metadata.clean_cu_seqlens.to(
                        device, non_blocking=non_blocking
                    ),
                    max_clean_seqlen=self.metadata.max_clean_seqlen,
                    block_mask=self.metadata.block_mask,
                )
            ),
        )

    def pin_memory(self) -> "IDLMBatch":
        def pin(value: Tensor) -> Tensor:
            return value if value.is_pinned() else value.pin_memory()

        metadata = self.metadata
        return IDLMBatch(
            layout=IDLMTrainingLayout(
                proposal_ids=pin(self.layout.proposal_ids),
                clean_ids=pin(self.layout.clean_ids),
                valid=pin(self.layout.valid),
                positions=pin(self.layout.positions),
                block_size=self.layout.block_size,
                segment_ids=(
                    None
                    if self.layout.segment_ids is None
                    else pin(self.layout.segment_ids)
                ),
            ),
            targets=IDLMShiftedTargets(
                pin(self.targets.proposal),
                pin(self.targets.clean),
                self.targets.ignore_index,
            ),
            bos_targets=pin(self.bos_targets),
            metadata=(
                None
                if metadata is None
                else IDLMForwardMetadata(
                    clean_indices=pin(metadata.clean_indices),
                    clean_cu_seqlens=pin(metadata.clean_cu_seqlens),
                    max_clean_seqlen=metadata.max_clean_seqlen,
                    block_mask=metadata.block_mask,
                )
            ),
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        values = (
            self.layout.proposal_ids,
            self.layout.clean_ids,
            self.layout.valid,
            self.layout.positions,
            self.layout.segment_ids,
            self.targets.proposal,
            self.targets.clean,
            self.bos_targets,
            None if self.metadata is None else self.metadata.clean_indices,
            None if self.metadata is None else self.metadata.clean_cu_seqlens,
        )
        for value in values:
            if value is not None:
                value.record_stream(stream)

    def with_metadata(self, metadata: IDLMForwardMetadata) -> "IDLMBatch":
        return IDLMBatch(
            layout=self.layout,
            targets=self.targets,
            bos_targets=self.bos_targets,
            metadata=metadata,
        )


@dataclass(frozen=True)
class IDLMObjective:
    total: Tensor
    proposal_mean: Tensor
    clean_mean: Tensor
    clean_scale: Tensor
    proposal_nll_sum: Tensor
    clean_nll_sum: Tensor
    proposal_targets: Tensor
    clean_targets: Tensor
    proposal_offset_correct: Tensor
    proposal_offset_targets: Tensor

    @property
    def clean_bpb(self) -> Tensor:
        return self.clean_nll_sum / self.clean_targets.clamp_min(1) / math.log(2.0)


@dataclass(frozen=True)
class IDLMStepMetrics:
    step: int
    total: float
    proposal_loss: float
    clean_loss: float
    clean_scale: float
    clean_bpb: float
    proposal_targets: int
    clean_targets: int
    learning_rate: float
    grad_norm: float
    elapsed_ms: float


@dataclass(frozen=True)
class IDLMValidationMetrics:
    clean_loss: float
    clean_bpb: float
    proposal_loss: float
    clean_scale: float
    proposal_targets: int
    clean_targets: int
    proposal_offset_accuracy: tuple[float, ...]
    proposal_offset_targets: tuple[int, ...]
    elapsed_ms: float


def _cross_entropy_sum(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    if logits.shape[:-1] != targets.shape or targets.dtype != torch.long:
        raise ValueError("I-DLM logits and targets must align")
    active = targets.ne(IGNORE_INDEX)
    safe = torch.where(active, targets, 0)
    ce_logits = logits.float() if logits.dtype in {torch.float16, torch.bfloat16} else logits
    nll = F.cross_entropy(
        ce_logits.reshape(-1, logits.shape[-1]),
        safe.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    return torch.where(active, nll, 0).sum(), active.sum()


def idlm_objective(
    output: IDLMModelOutput,
    batch: IDLMBatch,
    *,
    proposal_denominator: Tensor | int | None = None,
    clean_denominator: Tensor | int | None = None,
    balance_scale: Tensor | None = None,
) -> IDLMObjective:
    """Dense proposal CE plus clean shifted/BOS CE and detached balancing."""

    proposal_sum, proposal_count = _cross_entropy_sum(
        output.proposal_logits, batch.targets.proposal
    )
    clean_shifted_sum, clean_shifted_count = _cross_entropy_sum(
        output.clean_logits, batch.targets.clean
    )
    if batch.bos_targets.numel():
        bos_sum = F.cross_entropy(
            output.bos_logits.float(), batch.bos_targets, reduction="sum"
        )
    else:
        bos_sum = clean_shifted_sum.new_zeros(())
    clean_sum = clean_shifted_sum + bos_sum
    clean_count = clean_shifted_count + batch.bos_targets.numel()
    proposal_mean = proposal_sum / proposal_count.clamp_min(1)
    clean_mean = clean_sum / clean_count.clamp_min(1)
    if balance_scale is None:
        _, clean_scale = auto_balanced_loss(proposal_mean, clean_mean)
    else:
        if balance_scale.numel() != 1 or balance_scale.requires_grad:
            raise ValueError("I-DLM balance scale must be one detached scalar")
        clean_scale = balance_scale.to(clean_mean)
    proposal_normalizer = (
        proposal_count if proposal_denominator is None else proposal_denominator
    )
    clean_normalizer = clean_count if clean_denominator is None else clean_denominator
    proposal_contribution = proposal_sum / torch.as_tensor(
        proposal_normalizer, device=proposal_sum.device
    ).clamp_min(1).to(proposal_sum.dtype)
    clean_contribution = clean_sum / torch.as_tensor(
        clean_normalizer, device=clean_sum.device
    ).clamp_min(1).to(clean_sum.dtype)
    total = proposal_contribution + clean_scale * clean_contribution

    active = batch.targets.proposal.ne(IGNORE_INDEX)
    predictions = output.proposal_logits.argmax(-1)
    offsets = batch.layout.positions.remainder(batch.layout.block_size)
    offset_targets = torch.zeros(
        batch.layout.block_size, dtype=torch.long, device=active.device
    )
    offset_correct = torch.zeros_like(offset_targets)
    offset_targets.scatter_add_(
        0, offsets.reshape(-1), active.reshape(-1).to(torch.long)
    )
    offset_correct.scatter_add_(
        0,
        offsets.reshape(-1),
        (active & predictions.eq(batch.targets.proposal)).reshape(-1).to(torch.long),
    )
    return IDLMObjective(
        total=total,
        proposal_mean=proposal_mean,
        clean_mean=clean_mean,
        clean_scale=clean_scale,
        proposal_nll_sum=proposal_sum,
        clean_nll_sum=clean_sum,
        proposal_targets=proposal_count,
        clean_targets=clean_count,
        proposal_offset_correct=offset_correct,
        proposal_offset_targets=offset_targets,
    )


class IDLMTrainer:
    """Single-device trainer; GPU execution is submitted externally via mlq."""

    def __init__(
        self,
        model: IDLMModel,
        train_dataset: NativeBatchDataset,
        validation_dataset: NativeBatchDataset,
        cursor: DeterministicChunkCursor,
        config: IDLMTrainingConfig,
        *,
        device: torch.device,
        atomic_manifest: AtomicIdManifest,
        batch_prefetcher: DeviceBatchPrefetcher | None = None,
    ) -> None:
        if len(validation_dataset) == 0:
            raise ValueError("I-DLM validation dataset cannot be empty")
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.cursor = cursor
        self.config = config
        self.device = device
        self.atomic_manifest = atomic_manifest
        if config.stride_curriculum[-1][1] != model.config.block_size:
            raise ValueError(
                "the final curriculum stride must equal the checkpoint serving stride"
            )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
            fused=device.type == "cuda",
        )
        self._forward_layout = self.model.forward_layout
        if device.type == "cuda" and config.compile_model:
            self._forward_layout = torch.compile(
                self.model.forward_layout,
                dynamic=True,
                fullgraph=False,
            )
        self.completed_steps = 0
        self.training_time_ms = 0.0
        self._training_window_started_at: float | None = None
        self.last_update_counts = torch.zeros(2, dtype=torch.long, device=device)
        self._microbatch_keys = tuple(
            (start, min(start + config.batch_size, config.global_batch_size))
            for start in range(0, config.global_batch_size, config.batch_size)
        )
        self._trainable_parameters = tuple(
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        )
        self._owns_batch_prefetcher = batch_prefetcher is None
        self.batch_prefetcher = (
            DeviceBatchPrefetcher(device=device)
            if batch_prefetcher is None
            else batch_prefetcher
        )
        try:
            self._validation_ledgers = {
                stride: self._build_validation_ledger(stride)
                for stride in dict.fromkeys(item[1] for item in config.stride_curriculum)
            }
        except BaseException:
            if self._owns_batch_prefetcher:
                self.batch_prefetcher.close()
            raise

    def close(self) -> None:
        if self._owns_batch_prefetcher:
            self.batch_prefetcher.close()

    @property
    def allow_dense_reference(self) -> bool:
        return self.device.type == "cpu"

    def _stride_for_update(self, update_index: int) -> int:
        stride = self.config.stride_curriculum[0][1]
        for start, candidate in self.config.stride_curriculum[1:]:
            if update_index < start:
                break
            stride = candidate
        return stride

    def _cpu_batch(
        self,
        dataset: NativeBatchDataset,
        indices: Sequence[int],
        *,
        block_size: int | None = None,
    ) -> IDLMBatch:
        return IDLMBatch.from_native(
            dataset.training_batch(indices),
            mask_id=self.model.config.mask_id,
            pad_id=self.model.config.pad_id,
            block_size=(
                self.model.config.block_size if block_size is None else block_size
            ),
        )

    def _prepare_device_metadata(self, batch: IDLMBatch) -> IDLMBatch:
        if self.device.type == "cuda":
            return batch.with_metadata(
                self.model.prepare_forward_metadata(batch.layout, batch.metadata)
            )
        return batch

    def _batch(
        self,
        dataset: NativeBatchDataset,
        indices: Sequence[int],
        *,
        block_size: int | None = None,
    ) -> IDLMBatch:
        batch = self._cpu_batch(dataset, indices, block_size=block_size)
        if self.device.type == "cuda":
            batch = batch.pin_memory()
        batch = batch.to(self.device, non_blocking=self.device.type == "cuda")
        return self._prepare_device_metadata(batch)

    def _build_validation_ledger(self, block_size: int) -> tuple[IDLMBatch, ...]:
        rows = min(self.config.validation_rows, len(self.validation_dataset))
        keys = tuple(
            np.arange(start, min(rows, start + self.config.validation_batch_size))
            for start in range(0, rows, self.config.validation_batch_size)
        )

        def prepare(indices: np.ndarray) -> IDLMBatch:
            return self._cpu_batch(
                self.validation_dataset, indices, block_size=block_size
            )

        ledger = tuple(
            self._prepare_device_metadata(batch)
            for batch in self.batch_prefetcher.batches(keys, prepare)
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return ledger

    def _autocast(self):
        return (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )

    def _set_learning_rate(self, update_index: int) -> float:
        if self.config.warmdown_iters == 0:
            scale = 1.0
        else:
            start = self.config.iterations - self.config.warmdown_iters
            scale = (
                1.0
                if update_index < start
                else (self.config.iterations - update_index)
                / self.config.warmdown_iters
            )
        learning_rate = self.config.learning_rate * scale
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def run_update(
        self, *, materialize_metrics: bool = True
    ) -> IDLMStepMetrics | None:
        self.model.train()
        if self._training_window_started_at is None:
            self._training_window_started_at = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(4, dtype=torch.float64, device=self.device)
        indices = self.cursor.next_indices(self.config.global_batch_size)
        training_stride = self._stride_for_update(self.completed_steps)
        global_native = self.train_dataset.training_batch(indices)
        proposal_denominator = global_native.ar_targets.ne(IGNORE_INDEX).sum().to(
            self.device
        )
        clean_denominator = proposal_denominator + global_native.bos_targets.numel()
        microbatch_keys = self._microbatch_keys

        def prepare_microbatch(key: tuple[int, int]) -> IDLMBatch:
            start, stop = key
            return self._cpu_batch(
                self.train_dataset,
                indices[start:stop],
                block_size=training_stride,
            )

        parameters = self._trainable_parameters
        proposal_gradients: dict[Tensor, Tensor] = {}
        batches = self.batch_prefetcher.batches(microbatch_keys, prepare_microbatch)
        for batch in batches:
            batch = self._prepare_device_metadata(batch)
            with self._autocast():
                output = self._forward_layout(
                    batch.layout,
                    metadata=batch.metadata,
                    allow_dense_reference=self.allow_dense_reference,
                    bos_count=batch.bos_targets.numel(),
                )
                objective = idlm_objective(
                    output,
                    batch,
                    proposal_denominator=proposal_denominator,
                    clean_denominator=clean_denominator,
                    balance_scale=output.clean_logits.new_ones(()),
                )
            proposal_contribution = objective.proposal_nll_sum / torch.as_tensor(
                proposal_denominator, device=self.device
            ).clamp_min(1).to(objective.proposal_nll_sum)
            clean_contribution = objective.clean_nll_sum / torch.as_tensor(
                clean_denominator, device=self.device
            ).clamp_min(1).to(objective.clean_nll_sum)

            # Eq. 2's detached scale is defined over the whole optimizer step,
            # but becomes known only after all accumulation microbatches. Keep
            # one auxiliary proposal-gradient bank while ordinary ``.grad``
            # accumulates the clean pathway, then combine them exactly once.
            clean_gradients = tuple(parameter.grad for parameter in parameters)
            for parameter in parameters:
                parameter.grad = None
            proposal_contribution.backward(retain_graph=True)
            for parameter in parameters:
                if parameter.grad is not None:
                    accumulated = proposal_gradients.get(parameter)
                    if accumulated is None:
                        proposal_gradients[parameter] = parameter.grad.detach().clone()
                    else:
                        accumulated.add_(parameter.grad)
                parameter.grad = None
            for parameter, clean_gradient in zip(
                parameters, clean_gradients, strict=True
            ):
                parameter.grad = clean_gradient
            clean_contribution.backward()
            totals += torch.stack(
                (
                    objective.proposal_nll_sum.detach().double(),
                    objective.clean_nll_sum.detach().double(),
                    objective.proposal_targets.detach().double(),
                    objective.clean_targets.detach().double(),
                )
            )
        proposal_mean = totals[0] / proposal_denominator.clamp_min(1)
        clean_mean = totals[1] / clean_denominator.clamp_min(1)
        clean_scale = (
            proposal_mean.detach()
            / clean_mean.detach().clamp_min(torch.finfo(clean_mean.dtype).tiny)
        ).to(torch.float32)
        for parameter in parameters:
            proposal_gradient = proposal_gradients.get(parameter)
            if parameter.grad is None:
                if proposal_gradient is not None:
                    parameter.grad = proposal_gradient
            else:
                parameter.grad.mul_(clean_scale)
                if proposal_gradient is not None:
                    parameter.grad.add_(proposal_gradient)
        if self.config.max_grad_norm is None:
            squared = sum(
                parameter.grad.float().square().sum()
                for parameter in self.model.parameters()
                if parameter.grad is not None
            )
            grad_norm = squared.sqrt()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
        next_step = self.completed_steps + 1
        learning_rate = self._set_learning_rate(self.completed_steps)
        self.optimizer.step()
        self.model.enforce_padding_invariant()
        self.completed_steps = next_step
        self.last_update_counts = totals[2:4].to(torch.long)
        if not materialize_metrics:
            return None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if self._training_window_started_at is None:
            raise AssertionError("I-DLM training timer was not initialized")
        elapsed_ms = (time.perf_counter() - self._training_window_started_at) * 1_000
        self.training_time_ms += elapsed_ms
        self._training_window_started_at = None
        proposal_count = int(totals[2])
        clean_count = int(totals[3])
        return IDLMStepMetrics(
            step=next_step,
            total=float(proposal_mean + clean_scale.double() * clean_mean),
            proposal_loss=float(proposal_mean),
            clean_loss=float(clean_mean),
            clean_scale=float(clean_scale),
            clean_bpb=float(clean_mean / math.log(2.0)),
            proposal_targets=proposal_count,
            clean_targets=clean_count,
            learning_rate=learning_rate,
            grad_norm=float(grad_norm),
            elapsed_ms=self.training_time_ms,
        )

    @torch.no_grad()
    def validate(self) -> IDLMValidationMetrics:
        was_training = self.model.training
        self.model.eval()
        started = time.perf_counter()
        totals = torch.zeros(4, dtype=torch.float64, device=self.device)
        validation_stride = self._stride_for_update(
            0 if self.completed_steps == 0 else self.completed_steps - 1
        )
        offset_correct = torch.zeros(
            validation_stride, dtype=torch.long, device=self.device
        )
        offset_targets = torch.zeros_like(offset_correct)
        for batch in self._validation_ledgers[validation_stride]:
            with self._autocast():
                output = self._forward_layout(
                    batch.layout,
                    metadata=batch.metadata,
                    allow_dense_reference=self.allow_dense_reference,
                    bos_count=batch.bos_targets.numel(),
                )
                objective = idlm_objective(output, batch)
            totals += torch.stack(
                (
                    objective.proposal_nll_sum.double(),
                    objective.clean_nll_sum.double(),
                    objective.proposal_targets.double(),
                    objective.clean_targets.double(),
                )
            )
            offset_correct += objective.proposal_offset_correct
            offset_targets += objective.proposal_offset_targets
        if was_training:
            self.model.train()
        proposal_count = int(totals[2])
        clean_count = int(totals[3])
        counts = tuple(int(value) for value in offset_targets.cpu().tolist())
        correct = tuple(int(value) for value in offset_correct.cpu().tolist())
        return IDLMValidationMetrics(
            clean_loss=float(totals[1] / max(clean_count, 1)),
            clean_bpb=float(totals[1] / max(clean_count, 1) / math.log(2.0)),
            proposal_loss=float(totals[0] / max(proposal_count, 1)),
            clean_scale=float(
                (totals[0] / max(proposal_count, 1))
                / (totals[1] / max(clean_count, 1)).clamp_min(
                    torch.finfo(totals.dtype).tiny
                )
            ),
            proposal_targets=proposal_count,
            clean_targets=clean_count,
            proposal_offset_accuracy=tuple(
                hits / count if count else 0.0
                for hits, count in zip(correct, counts, strict=True)
            ),
            proposal_offset_targets=counts,
            elapsed_ms=(time.perf_counter() - started) * 1_000,
        )

    def save_checkpoint(self, path: Path, *, extra: Mapping[str, object] | None = None) -> None:
        atomic_torch_save(
            {
                "schema": CHECKPOINT_SCHEMA,
                "model_config": asdict(self.model.config),
                "training_config": asdict(self.config),
                "atomic_manifest_sha256": self.atomic_manifest.sha256,
                "completed_steps": self.completed_steps,
                "training_time_ms": self.training_time_ms,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "cursor": self.cursor.state_dict(),
                "extra": dict(extra or {}),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        expected = {
            "schema": CHECKPOINT_SCHEMA,
            "model_config": asdict(self.model.config),
            "training_config": asdict(self.config),
            "atomic_manifest_sha256": self.atomic_manifest.sha256,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"I-DLM checkpoint {key} mismatch")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.cursor.load_state_dict(payload["cursor"])
        self.completed_steps = int(payload["completed_steps"])
        self.training_time_ms = float(payload["training_time_ms"])


def format_train_metric(metric: IDLMStepMetrics, iterations: int) -> str:
    return (
        f"step:{metric.step}/{iterations} train_loss:{metric.total:.6f} "
        f"proposal_loss:{metric.proposal_loss:.6f} clean_loss:{metric.clean_loss:.6f} "
        f"clean_bpb:{metric.clean_bpb:.6f} clean_scale:{metric.clean_scale:.6f} "
        f"proposal_targets:{metric.proposal_targets} clean_targets:{metric.clean_targets} "
        f"lr:{metric.learning_rate:.8g} grad_norm:{metric.grad_norm:.6f} "
        f"train_time:{metric.elapsed_ms:.3f}ms"
    )


def format_validation_metric(
    step: int, iterations: int, metric: IDLMValidationMetrics, training_time_ms: float
) -> str:
    offsets = " ".join(
        f"proposal_acc_o{index}:{accuracy:.6f}"
        for index, accuracy in enumerate(metric.proposal_offset_accuracy)
    )
    return (
        f"step:{step}/{iterations} val_loss:{metric.clean_loss:.6f} "
        f"val_bpb:{metric.clean_bpb:.6f} "
        f"val_ar_anchor_bpb:{metric.clean_bpb:.6f} generation_primary:1 "
        f"val_proposal_loss:{metric.proposal_loss:.6f} "
        f"val_clean_scale:{metric.clean_scale:.6f} "
        f"val_clean_targets:{metric.clean_targets} "
        f"val_proposal_targets:{metric.proposal_targets} {offsets} "
        f"val_time_ms:{metric.elapsed_ms:.3f} train_time:{training_time_ms:.3f}ms"
    )


__all__ = [
    "IGNORE_INDEX",
    "CHECKPOINT_SCHEMA",
    "NativeBatchDataset",
    "IDLMTrainingConfig",
    "IDLMBatch",
    "IDLMObjective",
    "IDLMStepMetrics",
    "IDLMValidationMetrics",
    "idlm_objective",
    "IDLMTrainer",
    "format_train_metric",
    "format_validation_metric",
]
