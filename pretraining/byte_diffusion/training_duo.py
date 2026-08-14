"""Prepared corruption, exact objective, and validation for Byte-Duo."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Iterable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from .attention import (
    CanvasBlockMaskMetadata,
    CanvasBranchLayout,
    build_canvas_block_mask,
    canvas_block_mask_metadata,
)
from .config import AtomicVocabulary
from .diffusion_gemma_model import (
    DiffusionGemmaAttentionMetadata,
    compile_stable_document_metadata,
)
from .duo import (
    DuoCorruption,
    DuoSchedule,
    duo_nelbo_token_loss,
    sample_branch_antithetic_times,
    uniform_state_corruption,
)
from .duo_model import DuoModel
from .training_diffusion_gemma import DocumentCanvasSelection, select_document_canvases


IGNORE_INDEX = -100

PURE_DUO_OBJECTIVE = "pure_nelbo"
JOINT_DUO_CLEAN_AR_OBJECTIVE = "joint_nelbo_clean_ar"
DuoTrainingObjective = Literal["pure_nelbo", "joint_nelbo_clean_ar"]
DUO_TRAINING_OBJECTIVES: tuple[DuoTrainingObjective, ...] = (
    PURE_DUO_OBJECTIVE,
    JOINT_DUO_CLEAN_AR_OBJECTIVE,
)


def duo_clean_ar_weight(objective: str) -> float:
    """Return the fixed clean-AR coefficient for a named Duo objective."""

    if objective == PURE_DUO_OBJECTIVE:
        return 0.0
    if objective == JOINT_DUO_CLEAN_AR_OBJECTIVE:
        return 1.0
    raise ValueError(f"unknown Byte-Duo training objective {objective!r}")


def duo_objective_contract(objective: str) -> dict[str, object]:
    """Describe the exact independently normalized optimizer objective."""

    clean_ar_weight = duo_clean_ar_weight(objective)
    return {
        "objective": objective,
        "duo_nelbo_reduction": "global_mean_over_active_diffusion_atoms",
        "clean_ar_reduction": "global_mean_over_next_atom_and_document_bos_targets",
        "clean_ar_weight": clean_ar_weight,
        "combination": "duo_nelbo_plus_weighted_clean_causal_ar",
    }


def _pin_cpu_tensor(value: Tensor, *, owner: str) -> Tensor:
    """Materialize CPU views before registering their storage as pinned."""

    if value.device.type != "cpu":
        raise ValueError(f"only CPU {owner} can be pinned")
    contiguous = value.contiguous()
    return contiguous if contiguous.is_pinned() else contiguous.pin_memory()


@dataclass(frozen=True)
class DuoBatch:
    clean_ids: Tensor
    clean_valid: Tensor
    document_ids: Tensor
    positions: Tensor
    ar_targets: Tensor
    bos_targets: Tensor
    attention_metadata: DiffusionGemmaAttentionMetadata | None = None
    row_ids: Tensor | None = None
    bos_row_indices: Tensor | None = None

    def validate(
        self, vocab: AtomicVocabulary, *, check_values: bool = True
    ) -> None:
        shape = self.clean_ids.shape
        if self.clean_ids.ndim != 2 or any(
            value.shape != shape
            for value in (
                self.clean_valid,
                self.document_ids,
                self.positions,
                self.ar_targets,
            )
        ):
            raise ValueError("Duo batch fields must align as [B,L]")
        if self.clean_ids.dtype != torch.long or self.clean_valid.dtype != torch.bool:
            raise TypeError("Duo clean ids must be int64 and validity boolean")
        if any(
            value.dtype != torch.long
            for value in (self.document_ids, self.positions, self.ar_targets, self.bos_targets)
        ):
            raise TypeError("Duo metadata and targets must be int64")
        if self.row_ids is not None and (
            self.row_ids.shape != shape[:1] or self.row_ids.dtype != torch.long
        ):
            raise ValueError("Duo validation row ids must be aligned int64")
        if self.bos_row_indices is not None:
            if (
                self.bos_row_indices.shape != self.bos_targets.shape
                or self.bos_row_indices.dtype != torch.long
            ):
                raise ValueError("Duo BOS row indices must align with BOS targets")
            if not torch.compiler.is_compiling() and (
                bool(
                    (
                        (self.bos_row_indices < 0)
                        | (self.bos_row_indices >= shape[0])
                    ).any()
                )
                or bool(
                    (self.bos_row_indices[1:] < self.bos_row_indices[:-1]).any()
                )
            ):
                raise ValueError("Duo BOS row indices must be ordered and in range")
        if check_values:
            active = self.clean_ids.masked_select(self.clean_valid)
            if not torch.compiler.is_compiling() and bool(
                ((active < 0) | (active >= vocab.output_size)).any()
            ):
                raise ValueError("Duo clean data includes a non-clean atom")
            if not torch.compiler.is_compiling() and bool(
                self.clean_ids.masked_select(~self.clean_valid).ne(vocab.pad_id).any()
            ):
                raise ValueError("inactive Duo clean storage must contain PAD")

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "DuoBatch":
        metadata = (
            None
            if self.attention_metadata is None
            else self.attention_metadata.to(device, non_blocking=non_blocking)
        )
        return DuoBatch(
            clean_ids=self.clean_ids.to(device, non_blocking=non_blocking),
            clean_valid=self.clean_valid.to(device, non_blocking=non_blocking),
            document_ids=self.document_ids.to(device, non_blocking=non_blocking),
            positions=self.positions.to(device, non_blocking=non_blocking),
            ar_targets=self.ar_targets.to(device, non_blocking=non_blocking),
            bos_targets=self.bos_targets.to(device, non_blocking=non_blocking),
            attention_metadata=metadata,
            # Authentication metadata stays on the host so validation's
            # uniqueness checks do not synchronize CUDA every microbatch.
            row_ids=self.row_ids,
            # BOS rows are needed only while partitioning a host update. The
            # selected compact labels are already attached to this slice.
            bos_row_indices=None,
        )

    def pin_memory(self) -> "DuoBatch":
        def pin(value: Tensor) -> Tensor:
            return _pin_cpu_tensor(value, owner="Duo batches")

        metadata = self.attention_metadata
        return DuoBatch(
            clean_ids=pin(self.clean_ids),
            clean_valid=pin(self.clean_valid),
            document_ids=pin(self.document_ids),
            positions=pin(self.positions),
            ar_targets=pin(self.ar_targets),
            bos_targets=pin(self.bos_targets),
            attention_metadata=(
                None
                if metadata is None
                else DiffusionGemmaAttentionMetadata(
                    pin(metadata.byte_indices),
                    pin(metadata.byte_cu_seqlens),
                    pin(metadata.patch_indices),
                    pin(metadata.patch_cu_seqlens),
                    metadata.physical_layout,
                )
            ),
            row_ids=None if self.row_ids is None else pin(self.row_ids),
            bos_row_indices=(
                None if self.bos_row_indices is None else pin(self.bos_row_indices)
            ),
        )

    def slice_rows(
        self, start: int, stop: int, *, include_ar_bos: bool = False
    ) -> "DuoBatch":
        """Return a contiguous microbatch without rereading the dataset.

        Byte-Duo's production path uses compile-stable attention metadata. Its
        unused sequence capacity is a global tail, so a strict subset gets a
        fresh fixed-shape cumulative-offset table rather than an invalid slice
        through that tail. A full-batch slice reuses the original metadata.
        BOS labels are compact, so joint-objective callers explicitly request
        the labels whose recorded physical row belongs to this slice. Pure
        diffusion updates intentionally receive an empty tensor and preserve
        their former no-clean-logits execution path.
        """

        rows, width = self.clean_ids.shape
        if not 0 <= start < stop <= rows:
            raise ValueError("invalid Duo batch row slice")
        if start == 0 and stop == rows:
            return self
        if include_ar_bos and self.bos_targets.numel() and self.bos_row_indices is None:
            raise ValueError("joint Duo row slicing requires BOS row indices")
        metadata = self.attention_metadata
        if metadata is None:
            raise ValueError("Duo row slicing requires compile-stable attention metadata")
        if metadata.byte_indices.numel() != rows * width:
            raise ValueError("Duo row slicing requires physical byte metadata")
        if metadata.patch_indices.numel() % rows:
            raise ValueError("Duo row slicing requires physical patch metadata")
        patch_width = metadata.patch_indices.numel() // rows
        patch_stride, remainder = divmod(width, patch_width)
        if remainder or patch_stride <= 0:
            raise ValueError("Duo byte and patch metadata widths disagree")
        clean_valid = self.clean_valid[start:stop]
        document_ids = self.document_ids[start:stop]
        if include_ar_bos and self.bos_row_indices is not None:
            selected_bos = self.bos_row_indices.ge(start) & self.bos_row_indices.lt(stop)
            bos_targets = self.bos_targets.masked_select(selected_bos)
            bos_row_indices = self.bos_row_indices.masked_select(selected_bos) - start
        else:
            bos_targets = self.bos_targets.new_empty((0,))
            bos_row_indices = (
                None
                if self.bos_row_indices is None
                else self.bos_row_indices.new_empty((0,))
            )
        return DuoBatch(
            clean_ids=self.clean_ids[start:stop],
            clean_valid=clean_valid,
            document_ids=document_ids,
            positions=self.positions[start:stop],
            ar_targets=self.ar_targets[start:stop],
            bos_targets=bos_targets,
            attention_metadata=compile_stable_document_metadata(
                clean_valid,
                document_ids,
                patch_stride=patch_stride,
                max_segments_per_row=(metadata.byte_cu_seqlens.numel() - 1) // rows,
            ),
            row_ids=None if self.row_ids is None else self.row_ids[start:stop],
            bos_row_indices=bos_row_indices,
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        for value in (
            self.clean_ids,
            self.clean_valid,
            self.document_ids,
            self.positions,
            self.ar_targets,
            self.bos_targets,
        ):
            value.record_stream(stream)
        if self.attention_metadata is not None:
            for value in (
                self.attention_metadata.byte_indices,
                self.attention_metadata.byte_cu_seqlens,
                self.attention_metadata.patch_indices,
                self.attention_metadata.patch_cu_seqlens,
            ):
                value.record_stream(stream)
        if self.bos_row_indices is not None:
            self.bos_row_indices.record_stream(stream)


@dataclass(frozen=True)
class PreparedDuoInputs:
    selection: DocumentCanvasSelection
    corruption: DuoCorruption
    branches: int
    local_block_mask_metadata: CanvasBlockMaskMetadata | None = None
    global_block_mask_metadata: CanvasBlockMaskMetadata | None = None

    def pin_memory(self) -> "PreparedDuoInputs":
        """Pin the complete CPU ledger for an asynchronous device transfer."""

        def pin(value: Tensor) -> Tensor:
            return _pin_cpu_tensor(value, owner="PreparedDuoInputs")

        def pin_metadata(
            metadata: CanvasBlockMaskMetadata | None,
        ) -> CanvasBlockMaskMetadata | None:
            if metadata is None:
                return None
            return CanvasBlockMaskMetadata(
                pin(metadata.kv_num_blocks),
                pin(metadata.kv_indices),
                pin(metadata.q_num_blocks),
                pin(metadata.q_indices),
                metadata.block_size,
                metadata.query_length,
                metadata.kv_length,
                None
                if metadata.full_kv_num_blocks is None
                else pin(metadata.full_kv_num_blocks),
                None
                if metadata.full_kv_indices is None
                else pin(metadata.full_kv_indices),
                None
                if metadata.full_q_num_blocks is None
                else pin(metadata.full_q_num_blocks),
                None
                if metadata.full_q_indices is None
                else pin(metadata.full_q_indices),
            )

        return PreparedDuoInputs(
            DocumentCanvasSelection(
                starts=pin(self.selection.starts),
                valid=pin(self.selection.valid),
                targets=pin(self.selection.targets),
                document_ids=pin(self.selection.document_ids),
                positions=pin(self.selection.positions),
            ),
            DuoCorruption(
                ids=pin(self.corruption.ids),
                targets=pin(self.corruption.targets),
                active=pin(self.corruption.active),
                replaced=pin(self.corruption.replaced),
                changed=pin(self.corruption.changed),
                t=pin(self.corruption.t),
                alpha=pin(self.corruption.alpha),
                fixed_clean=(
                    None
                    if self.corruption.fixed_clean is None
                    else pin(self.corruption.fixed_clean)
                ),
            ),
            self.branches,
            pin_metadata(self.local_block_mask_metadata),
            pin_metadata(self.global_block_mask_metadata),
        )

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "PreparedDuoInputs":
        def move(value: Tensor) -> Tensor:
            return value.to(device, non_blocking=non_blocking)

        def move_metadata(
            metadata: CanvasBlockMaskMetadata | None,
        ) -> CanvasBlockMaskMetadata | None:
            return (
                None
                if metadata is None
                else metadata.to(device, non_blocking=non_blocking)
            )

        return PreparedDuoInputs(
            DocumentCanvasSelection(
                starts=move(self.selection.starts),
                valid=move(self.selection.valid),
                targets=move(self.selection.targets),
                document_ids=move(self.selection.document_ids),
                positions=move(self.selection.positions),
            ),
            DuoCorruption(
                ids=move(self.corruption.ids),
                targets=move(self.corruption.targets),
                active=move(self.corruption.active),
                replaced=move(self.corruption.replaced),
                changed=move(self.corruption.changed),
                t=move(self.corruption.t),
                alpha=move(self.corruption.alpha),
                fixed_clean=(
                    None
                    if self.corruption.fixed_clean is None
                    else move(self.corruption.fixed_clean)
                ),
            ),
            self.branches,
            move_metadata(self.local_block_mask_metadata),
            move_metadata(self.global_block_mask_metadata),
        )

    def slice_rows(self, start: int, stop: int) -> "PreparedDuoInputs":
        if not 0 <= start <= stop <= self.selection.starts.shape[0]:
            raise ValueError("invalid prepared Duo row slice")
        branches = self.branches
        flat_start, flat_stop = start * branches, stop * branches
        selection = DocumentCanvasSelection(
            starts=self.selection.starts[start:stop],
            valid=self.selection.valid[start:stop],
            targets=self.selection.targets[start:stop],
            document_ids=self.selection.document_ids[start:stop],
            positions=self.selection.positions[start:stop],
        )
        corruption = DuoCorruption(
            ids=self.corruption.ids[flat_start:flat_stop],
            targets=self.corruption.targets[flat_start:flat_stop],
            active=self.corruption.active[flat_start:flat_stop],
            replaced=self.corruption.replaced[flat_start:flat_stop],
            changed=self.corruption.changed[flat_start:flat_stop],
            t=self.corruption.t[flat_start:flat_stop],
            alpha=self.corruption.alpha[flat_start:flat_stop],
            fixed_clean=(
                None
                if self.corruption.fixed_clean is None
                else self.corruption.fixed_clean[flat_start:flat_stop]
            ),
        )

        def slice_metadata(
            metadata: CanvasBlockMaskMetadata | None,
        ) -> CanvasBlockMaskMetadata | None:
            if metadata is None:
                return None
            return CanvasBlockMaskMetadata(
                metadata.kv_num_blocks[start:stop],
                metadata.kv_indices[start:stop],
                metadata.q_num_blocks[start:stop],
                metadata.q_indices[start:stop],
                metadata.block_size,
                metadata.query_length,
                metadata.kv_length,
                None
                if metadata.full_kv_num_blocks is None
                else metadata.full_kv_num_blocks[start:stop],
                None
                if metadata.full_kv_indices is None
                else metadata.full_kv_indices[start:stop],
                None
                if metadata.full_q_num_blocks is None
                else metadata.full_q_num_blocks[start:stop],
                None
                if metadata.full_q_indices is None
                else metadata.full_q_indices[start:stop],
            )

        return PreparedDuoInputs(
            selection,
            corruption,
            branches,
            slice_metadata(self.local_block_mask_metadata),
            slice_metadata(self.global_block_mask_metadata),
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        for value in (
            self.selection.starts,
            self.selection.valid,
            self.selection.targets,
            self.selection.document_ids,
            self.selection.positions,
            self.corruption.ids,
            self.corruption.targets,
            self.corruption.active,
            self.corruption.replaced,
            self.corruption.changed,
            self.corruption.t,
            self.corruption.alpha,
        ):
            value.record_stream(stream)
        for metadata in (
            self.local_block_mask_metadata,
            self.global_block_mask_metadata,
        ):
            if metadata is not None:
                metadata.record_stream(stream)


@dataclass(frozen=True)
class PreparedDuoMicrobatch:
    """One row slice of a fully prepared pure-diffusion optimizer update."""

    batch: DuoBatch
    prepared: PreparedDuoInputs

    def pin_memory(self) -> "PreparedDuoMicrobatch":
        return PreparedDuoMicrobatch(
            self.batch.pin_memory(),
            self.prepared.pin_memory(),
        )

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "PreparedDuoMicrobatch":
        return PreparedDuoMicrobatch(
            self.batch.to(device, non_blocking=non_blocking),
            self.prepared.to(device, non_blocking=non_blocking),
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        self.batch.record_stream(stream)
        self.prepared.record_stream(stream)


@dataclass(frozen=True)
class PreparedDuoUpdate:
    """A complete update that can be prefetched as one host/device payload."""

    microbatches: tuple[PreparedDuoMicrobatch, ...]
    nelbo_denominator: Tensor
    ar_denominator: Tensor
    host_state: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.microbatches:
            raise ValueError("a prepared Duo update needs at least one microbatch")
        for name, value in (
            ("NELBO", self.nelbo_denominator),
            ("AR", self.ar_denominator),
        ):
            if value.shape != () or value.dtype != torch.long:
                raise ValueError(f"{name} denominator must be a scalar int64 tensor")

    def pin_memory(self) -> "PreparedDuoUpdate":
        return PreparedDuoUpdate(
            tuple(item.pin_memory() for item in self.microbatches),
            _pin_cpu_tensor(self.nelbo_denominator, owner="Duo update denominators"),
            _pin_cpu_tensor(self.ar_denominator, owner="Duo update denominators"),
            self.host_state,
        )

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "PreparedDuoUpdate":
        return PreparedDuoUpdate(
            tuple(
                item.to(device, non_blocking=non_blocking)
                for item in self.microbatches
            ),
            self.nelbo_denominator.to(device, non_blocking=non_blocking),
            self.ar_denominator.to(device, non_blocking=non_blocking),
            self.host_state,
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        for item in self.microbatches:
            item.record_stream(stream)
        self.nelbo_denominator.record_stream(stream)
        self.ar_denominator.record_stream(stream)


def prepare_duo_update(
    model: DuoModel,
    batch: DuoBatch,
    *,
    microbatch_size: int,
    canvas_length: int,
    branches: int,
    schedule: DuoSchedule = DuoSchedule(),
    times: Tensor | None = None,
    generator: torch.Generator | None = None,
    include_clean_ar: bool = False,
    host_state: dict[str, object] | None = None,
) -> PreparedDuoUpdate:
    """Prepare, partition, and count one exact Duo optimizer update on CPU."""

    rows = batch.clean_ids.shape[0]
    if batch.clean_ids.device.type != "cpu":
        raise ValueError("prefetched Duo updates must be prepared on CPU")
    if microbatch_size <= 0:
        raise ValueError("Duo microbatch size must be positive")
    prepared = prepare_duo_inputs(
        model,
        batch,
        canvas_length=canvas_length,
        branches=branches,
        schedule=schedule,
        times=times,
        generator=generator,
    )
    microbatches = tuple(
        PreparedDuoMicrobatch(
            batch.slice_rows(
                start,
                min(start + microbatch_size, rows),
                include_ar_bos=include_clean_ar,
            ),
            prepared.slice_rows(start, min(start + microbatch_size, rows)),
        )
        for start in range(0, rows, microbatch_size)
    )
    ar_denominator = batch.ar_targets.ne(IGNORE_INDEX).sum() + batch.bos_targets.numel()
    return PreparedDuoUpdate(
        microbatches,
        prepared.corruption.active.sum(),
        ar_denominator,
        host_state,
    )


@dataclass(frozen=True)
class PreparedDuoValidationBatch:
    """Authenticated validation batch whose deterministic ledger is cached."""

    batch: DuoBatch
    prepared: PreparedDuoInputs

    def pin_memory(self) -> "PreparedDuoValidationBatch":
        batch = self.batch

        def pin(value: Tensor) -> Tensor:
            return _pin_cpu_tensor(value, owner="validation batches")

        metadata = batch.attention_metadata
        pinned_metadata = (
            None
            if metadata is None
            else DiffusionGemmaAttentionMetadata(
                pin(metadata.byte_indices),
                pin(metadata.byte_cu_seqlens),
                pin(metadata.patch_indices),
                pin(metadata.patch_cu_seqlens),
                metadata.physical_layout,
            )
        )
        pinned_batch = DuoBatch(
            clean_ids=pin(batch.clean_ids),
            clean_valid=pin(batch.clean_valid),
            document_ids=pin(batch.document_ids),
            positions=pin(batch.positions),
            ar_targets=pin(batch.ar_targets),
            bos_targets=pin(batch.bos_targets),
            attention_metadata=pinned_metadata,
            row_ids=batch.row_ids,
            bos_row_indices=(
                None if batch.bos_row_indices is None else pin(batch.bos_row_indices)
            ),
        )
        return PreparedDuoValidationBatch(
            pinned_batch, self.prepared.pin_memory()
        )

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "PreparedDuoValidationBatch":
        return PreparedDuoValidationBatch(
            self.batch.to(device, non_blocking=non_blocking),
            self.prepared.to(device, non_blocking=non_blocking),
        )


@dataclass(frozen=True)
class DuoLoss:
    total: Tensor
    nelbo_sum: Tensor
    nelbo_targets: Tensor
    ar_nll_sum: Tensor
    ar_targets: Tensor
    denoising_correct: Tensor
    corruption: DuoCorruption


@dataclass(frozen=True)
class DuoValidation:
    conditional_canvas_nelbo_nats_per_atom: float
    clean_ar_nats_per_atom: float | None
    ar_anchor_bpb: float | None
    denoising_accuracy: float
    targets: int
    changed_targets: int
    ar_targets: int


def _counter_hash(counter: Tensor, seed: int, stream: int) -> Tensor:
    """Deterministic vectorized 63-bit validation-ledger hash."""

    value = counter.to(torch.int64)
    value = value * 6_364_136_223_846_793_005
    mask = (1 << 63) - 1
    seed_term = ((int(seed) + 1) * 1_442_695_040_888_963_407) & mask
    stream_term = ((int(stream) + 1) * 2_862_933_555_777_941_757) & mask
    value = value + seed_term
    value = value + stream_term
    value = value ^ (value >> 21)
    value = value ^ (value << 37)
    value = value ^ (value >> 4)
    return value & mask


def _counter_uniform(counter: Tensor, seed: int, stream: int) -> Tensor:
    # Retain 52 high-quality mantissa bits, then cast to the model's FP32
    # schedule arithmetic. No mutable RNG state depends on validation batching.
    hashed = _counter_hash(counter, seed, stream)
    return ((hashed >> 11).to(torch.float64) / float(1 << 52)).to(torch.float32)


def _duo_canvas_layouts(
    model: DuoModel,
    batch: DuoBatch,
    selection: DocumentCanvasSelection,
) -> tuple[CanvasBranchLayout, CanvasBranchLayout]:
    """Construct the exact byte and patch branch layouts on any device."""

    branches = selection.valid.shape[1]
    canvas_length = selection.valid.shape[2]
    branch_segments = selection.document_ids[:, :, 0]
    local_layout = CanvasBranchLayout(
        batch.clean_valid,
        selection.valid,
        selection.starts,
        model.config.local_window,
        batch.positions,
        selection.positions,
        batch.document_ids,
        branch_segments,
    )
    stride = model.config.patch_stride
    rows, length = batch.clean_valid.shape
    clean_patch_valid = batch.clean_valid.view(
        rows, length // stride, stride
    ).any(-1)
    clean_patch_documents = batch.document_ids.view(
        rows, length // stride, stride
    )[:, :, 0]
    clean_patch_positions = batch.positions[:, ::stride].div(
        stride, rounding_mode="floor"
    )
    branch_patch_valid = selection.valid.view(
        rows, branches, canvas_length // stride, stride
    ).any(-1)
    branch_patch_positions = selection.positions[:, :, ::stride].div(
        stride, rounding_mode="floor"
    )
    global_layout = CanvasBranchLayout(
        clean_patch_valid,
        branch_patch_valid,
        selection.starts // stride,
        model.config.global_window,
        clean_patch_positions,
        branch_patch_positions,
        clean_patch_documents,
        branch_segments,
    )
    return local_layout, global_layout


def _duo_block_mask_metadata(
    model: DuoModel,
    batch: DuoBatch,
    selection: DocumentCanvasSelection,
) -> tuple[CanvasBlockMaskMetadata | None, CanvasBlockMaskMetadata | None]:
    """Prepare byte/patch sparse topology next to the CPU canvas ledger."""

    if batch.clean_ids.device.type != "cpu":
        return None, None
    local_layout, global_layout = _duo_canvas_layouts(model, batch, selection)
    return (
        canvas_block_mask_metadata(local_layout),
        canvas_block_mask_metadata(global_layout),
    )


def prepare_duo_validation_inputs(
    model: DuoModel,
    batch: DuoBatch,
    *,
    row_ids: Tensor,
    total_rows: int,
    canvas_length: int,
    branches: int,
    seed: int,
    schedule: DuoSchedule = DuoSchedule(),
    expose_random_phase: bool = False,
) -> PreparedDuoInputs:
    """Build a batch/order-invariant authenticated validation ledger slice.

    The canonical within-family metric always uses an all-active canvas so its
    numerator, denominator, and conditioning distribution remain identical
    across architecture cells. ``expose_random_phase`` is reserved for a
    separately named serving-robustness diagnostic.
    """

    # The data loader validates atom ranges once. Rechecking values here would
    # add two host/device synchronizations to every validation microbatch.
    batch.validate(model.config.vocab, check_values=False)
    rows, length = batch.clean_ids.shape
    if row_ids.shape != (rows,) or row_ids.dtype != torch.long:
        raise ValueError("validation ledger row ids must be aligned int64")
    if total_rows <= 0 or not torch.compiler.is_compiling() and bool(
        ((row_ids < 0) | (row_ids >= total_rows)).any()
    ):
        raise ValueError("validation ledger row ids lie outside its declared extent")
    device = batch.clean_ids.device
    branch_ids = row_ids[:, None] * branches + torch.arange(branches, device=device)
    total_branches = total_rows * branches

    # Globally stratified time coordinates with a deterministic affine
    # permutation. The multiplier is selected once from values coprime to the
    # ledger extent, so every stratum is used exactly once.
    multiplier = 2 * (abs(int(seed)) % max(total_branches, 1)) + 1
    while math.gcd(multiplier, total_branches) != 1:
        multiplier += 2
    strata = (multiplier * branch_ids + abs(int(seed)) + 17) % total_branches
    jitter = _counter_uniform(branch_ids, seed, 0)
    unit_times = (strata.to(torch.float32) + jitter) / total_branches
    branch_times = unit_times.clamp_min(torch.finfo(unit_times.dtype).eps)

    candidates = batch.clean_valid & batch.positions.remainder(
        model.config.patch_stride
    ).eq(0)
    candidate_counts = candidates.sum(1)
    if not torch.compiler.is_compiling() and bool(candidate_counts.eq(0).any()):
        raise ValueError("every validation row needs a canvas origin")
    origin_rank = _counter_hash(branch_ids, seed, 1) % candidate_counts[:, None]
    candidate_rank = candidates.cumsum(1) - 1
    starts = (
        candidates[:, None, :]
        & candidate_rank[:, None, :].eq(origin_rank[:, :, None])
    ).to(torch.int64).argmax(-1)

    offsets = torch.arange(canvas_length, device=device)
    indices = starts[:, :, None] + offsets
    exists = indices.lt(length)
    safe = indices.clamp_max(length - 1)
    expanded_ids = batch.clean_ids[:, None, :].expand(-1, branches, -1)
    expanded_valid = batch.clean_valid[:, None, :].expand(-1, branches, -1)
    expanded_documents = batch.document_ids[:, None, :].expand(-1, branches, -1)
    targets = torch.gather(expanded_ids, 2, safe)
    gathered_valid = torch.gather(expanded_valid, 2, safe)
    gathered_documents = torch.gather(expanded_documents, 2, safe)
    origin_documents = torch.gather(batch.document_ids, 1, starts)[:, :, None]
    origin_positions = torch.gather(batch.positions, 1, starts)[:, :, None]
    valid = exists & gathered_valid & gathered_documents.eq(origin_documents)
    targets = torch.where(valid, targets, model.config.vocab.pad_id)
    selection = DocumentCanvasSelection(
        starts=starts,
        valid=valid,
        targets=targets,
        document_ids=origin_documents.expand_as(indices),
        positions=origin_positions + offsets,
    )
    selection.validate(batch.clean_ids, batch.clean_valid)

    flat_targets = targets.flatten(0, 1)
    flat_active = valid.flatten(0, 1)
    flat_times = branch_times.flatten()
    alpha = schedule.alpha(flat_times)
    token_counters = branch_ids.flatten()[:, None] * canvas_length + offsets
    fixed_clean = torch.zeros_like(flat_active)
    if expose_random_phase:
        phase = (
            _counter_hash(branch_ids.flatten(), seed, 4)
            % model.config.patch_stride
        )[:, None]
        fixed_clean = flat_active & offsets[None].lt(phase)
        flat_active = flat_active & ~fixed_clean
    replaced = flat_active & _counter_uniform(token_counters, seed, 2).ge(
        alpha[:, None]
    )
    uniform = _counter_hash(token_counters, seed, 3) % model.config.duo_diffusion_atoms
    noisy = torch.where(
        flat_active,
        torch.where(replaced, uniform, flat_targets),
        model.config.vocab.pad_id,
    )
    noisy = torch.where(fixed_clean, flat_targets, noisy)
    corruption = DuoCorruption(
        ids=noisy,
        targets=flat_targets,
        active=flat_active,
        replaced=replaced,
        changed=flat_active & noisy.ne(flat_targets),
        t=flat_times,
        alpha=alpha,
        fixed_clean=fixed_clean,
    )
    corruption.validate(
        clean_atoms=model.config.duo_diffusion_atoms,
        pad_id=model.config.vocab.pad_id,
    )
    local_metadata, global_metadata = _duo_block_mask_metadata(
        model, batch, selection
    )
    return PreparedDuoInputs(
        selection,
        corruption,
        branches,
        local_metadata,
        global_metadata,
    )


def prepare_duo_validation_batch(
    model: DuoModel,
    batch: DuoBatch,
    *,
    total_rows: int,
    canvas_length: int,
    branches: int,
    seed: int,
    schedule: DuoSchedule = DuoSchedule(),
    expose_random_phase: bool = False,
) -> PreparedDuoValidationBatch:
    """Authenticate and freeze one deterministic ledger batch on the host.

    This checked constructor is intentionally CPU-only. All scalar range,
    alignment, schedule, and corruption invariants are proved once before the
    batch enters the repeated GPU validation path.
    """

    if batch.clean_ids.device.type != "cpu":
        raise ValueError("Duo validation ledgers must be prepared on CPU")
    if batch.row_ids is None or batch.row_ids.device.type != "cpu":
        raise ValueError("Duo validation ledger rows require CPU row ids")
    prepared = prepare_duo_validation_inputs(
        model,
        batch,
        row_ids=batch.row_ids,
        total_rows=total_rows,
        canvas_length=canvas_length,
        branches=branches,
        seed=seed,
        schedule=schedule,
        expose_random_phase=expose_random_phase,
    )
    return PreparedDuoValidationBatch(batch, prepared)


def prepare_duo_inputs(
    model: DuoModel,
    batch: DuoBatch,
    *,
    canvas_length: int,
    branches: int,
    schedule: DuoSchedule = DuoSchedule(),
    times: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> PreparedDuoInputs:
    """Prepare all stochastic state once for a complete global update."""

    batch.validate(model.config.vocab)
    if model.schedule_eps != schedule.eps:
        raise ValueError("Duo model and corruption schedules disagree")
    selection = select_document_canvases(
        batch.clean_ids,
        batch.clean_valid,
        batch.document_ids,
        batch.positions,
        canvas_length=canvas_length,
        branches=branches,
        patch_stride=model.config.patch_stride,
        pad_id=model.config.vocab.pad_id,
        generator=generator,
    )
    rows = batch.clean_ids.shape[0]
    if times is None:
        times = sample_branch_antithetic_times(
            rows,
            branches,
            device=batch.clean_ids.device,
            generator=generator,
        )
    if times.shape == (rows,):
        branch_times = times[:, None].expand(rows, branches).reshape(-1)
    elif times.shape == (rows, branches):
        branch_times = times.reshape(-1)
    else:
        raise ValueError("prepared Duo times must be [B] or [B,M]")
    targets = selection.targets.flatten(0, 1)
    active = selection.valid.flatten(0, 1)
    offsets = torch.arange(canvas_length, device=active.device)[None]
    fixed_clean = torch.zeros_like(active)
    if model.config.duo_random_phase_training:
        phases = torch.randint(
            model.config.patch_stride,
            (active.shape[0], 1),
            device=active.device,
            generator=generator,
        )
        fixed_clean = active & offsets.lt(phases)
        active = active & ~fixed_clean
    if model.config.duo_variable_length_probability:
        shortened = torch.rand(
            (active.shape[0], 1), device=active.device, generator=generator
        ).lt(model.config.duo_variable_length_probability)
        sampled_widths = torch.randint(
            1,
            canvas_length + 1,
            (active.shape[0], 1),
            device=active.device,
            generator=generator,
        )
        widths = torch.where(
            shortened, sampled_widths, torch.full_like(sampled_widths, canvas_length)
        )
        active &= active.cumsum(1).le(widths)
    visible = active | fixed_clean
    targets = torch.where(visible, targets, model.config.vocab.pad_id)
    selection = replace(
        selection,
        valid=visible.view_as(selection.valid),
        targets=targets.view_as(selection.targets),
    )
    corruption = uniform_state_corruption(
        targets,
        active,
        branch_times,
        clean_atoms=model.config.duo_diffusion_atoms,
        pad_id=model.config.vocab.pad_id,
        schedule=schedule,
        generator=generator,
    )
    if bool(fixed_clean.any()):
        corruption = DuoCorruption(
            ids=torch.where(fixed_clean, targets, corruption.ids),
            targets=corruption.targets,
            active=corruption.active,
            replaced=corruption.replaced,
            changed=corruption.changed,
            t=corruption.t,
            alpha=corruption.alpha,
            fixed_clean=fixed_clean,
        )
    local_metadata, global_metadata = _duo_block_mask_metadata(
        model, batch, selection
    )
    return PreparedDuoInputs(
        selection,
        corruption,
        branches,
        local_metadata,
        global_metadata,
    )


def _projection_owner(model: DuoModel) -> DuoModel:
    """Return the registered Duo module behind a possible DDP wrapper.

    DDP forwards ``forward`` but intentionally does not proxy arbitrary module
    attributes.  Accessing the wrapped module here keeps the output parameters
    registered exactly once while preserving DDP's gradient hooks.
    """

    owner = getattr(model, "module", model)
    owner = getattr(owner, "_orig_mod", owner)
    if not isinstance(owner, DuoModel):
        raise TypeError("clean AR projection requires a DuoModel or DDP(DuoModel)")
    return owner


def _ar_sum_and_count(model: DuoModel, states: Tensor, batch: DuoBatch) -> tuple[Tensor, Tensor]:
    """Clean causal NLL without materializing the full byte-logit tensor."""

    if states.shape != batch.clean_ids.shape + (model.config.local_dim,):
        raise ValueError("clean decoder states must align with the clean batch")
    projection_owner = _projection_owner(model)
    output_weight = (
        projection_owner.embedding.weight[: model.config.vocab.output_size]
        if projection_owner.output is None
        else projection_owner.output.weight
    )
    output_bias = (
        None if projection_owner.output is None else projection_owner.output.bias
    )
    options = (
        torch.nn.LinearCrossEntropyOptions(
            # One 16-row page microbatch is 131,072 clean atoms. Keeping that
            # as one fused tile maximizes GEMM efficiency while avoiding the
            # update-wide [2M, 261] logit tensor.
            batch_chunk_size=131_072,
            acc_policy="accurate",
        )
        if states.device.type == "cuda"
        else None
    )
    clean_total = F.linear_cross_entropy(
        states.reshape(-1, states.shape[-1]),
        output_weight,
        batch.ar_targets.reshape(-1),
        linear_bias=output_bias,
        ignore_index=IGNORE_INDEX,
        reduction="sum",
        options=options,
    )
    bos_logits = model.forward_bos_logits(
        batch.bos_targets.numel(), device=batch.clean_ids.device
    )
    bos_total = F.cross_entropy(bos_logits, batch.bos_targets, reduction="sum")
    count = batch.ar_targets.ne(IGNORE_INDEX).sum() + batch.bos_targets.numel()
    return clean_total + bos_total, count


def duo_loss(
    model: DuoModel,
    batch: DuoBatch,
    *,
    canvas_length: int,
    branches: int,
    prepared: PreparedDuoInputs | None = None,
    schedule: DuoSchedule = DuoSchedule(),
    generator: torch.Generator | None = None,
    nelbo_denominator: Tensor | int | None = None,
    ar_denominator: Tensor | int | None = None,
    clean_ar_weight: float = 0.0,
    compute_ar_diagnostic: bool = False,
    compute_denoising_accuracy: bool = False,
    objective_fn: Callable[..., Tensor] = duo_nelbo_token_loss,
) -> DuoLoss:
    """Duo NELBO, optionally plus independently normalized clean causal AR."""

    if not math.isfinite(clean_ar_weight) or clean_ar_weight < 0:
        raise ValueError("clean AR weight must be finite and nonnegative")
    batch.validate(model.config.vocab, check_values=prepared is None)
    if model.schedule_eps != schedule.eps:
        raise ValueError("Duo model and objective schedules disagree")
    if prepared is None:
        prepared = prepare_duo_inputs(
            model,
            batch,
            canvas_length=canvas_length,
            branches=branches,
            schedule=schedule,
            generator=generator,
        )
    selection, corruption = prepared.selection, prepared.corruption
    batch_rows = batch.clean_ids.shape[0]
    noisy = corruption.ids.view(batch_rows, branches, canvas_length)
    branch_times = corruption.t.view(batch_rows, branches)
    need_clean_states = compute_ar_diagnostic or clean_ar_weight != 0.0
    local_block_mask = None
    global_block_mask = None
    if (
        batch.clean_ids.device.type == "cuda"
        and prepared.local_block_mask_metadata is not None
        and prepared.global_block_mask_metadata is not None
    ):
        local_layout, global_layout = _duo_canvas_layouts(
            model, batch, selection
        )
        local_block_mask = build_canvas_block_mask(
            local_layout, metadata=prepared.local_block_mask_metadata
        )
        global_block_mask = build_canvas_block_mask(
            global_layout, metadata=prepared.global_block_mask_metadata
        )
    output = model(
        batch.clean_ids,
        batch.clean_valid,
        batch.document_ids,
        batch.positions,
        noisy,
        selection.valid,
        selection.starts,
        branch_times,
        attention_metadata=batch.attention_metadata,
        local_block_mask_metadata=prepared.local_block_mask_metadata,
        global_block_mask_metadata=prepared.global_block_mask_metadata,
        local_block_mask=local_block_mask,
        global_block_mask=global_block_mask,
        return_clean_logits=False,
        return_clean_states=need_clean_states,
    )
    token_loss = objective_fn(
        output.branch_logits.flatten(0, 1)[..., : model.config.duo_diffusion_atoms],
        torch.where(corruption.active, corruption.ids, 0),
        torch.where(corruption.active, corruption.targets, 0),
        corruption.alpha,
        -(1.0 - schedule.eps),
        clean_atoms=model.config.duo_diffusion_atoms,
    )
    nelbo_sum = token_loss.masked_select(corruption.active).sum()
    nelbo_targets = corruption.active.sum()
    denominator = (
        nelbo_targets if nelbo_denominator is None else torch.as_tensor(
            nelbo_denominator, device=nelbo_sum.device
        )
    )
    diffusion = nelbo_sum / denominator.clamp_min(1).to(nelbo_sum.dtype)

    # Pure Duo training neither unembeds nor computes CE over the 8,192-byte
    # clean bank. The clean path is materialized only for validation or the
    # explicitly named joint-objective ablation.
    if need_clean_states:
        if output.clean_decoder_states is None:
            raise AssertionError("Duo clean-AR objective omitted decoder states")
        ar_nll_sum, ar_targets = _ar_sum_and_count(
            model, output.clean_decoder_states, batch
        )
    else:
        ar_nll_sum = nelbo_sum.new_zeros(())
        ar_targets = nelbo_targets.new_zeros(())
    denoising_correct = (
        (
            output.branch_logits.flatten(0, 1)[..., : model.config.duo_diffusion_atoms]
            .argmax(-1)
            .eq(corruption.targets)
            & corruption.active
        ).sum()
        if compute_denoising_accuracy
        else nelbo_targets.new_zeros(())
    )
    ar_normalizer = (
        ar_targets if ar_denominator is None else torch.as_tensor(
            ar_denominator, device=ar_nll_sum.device
        )
    )
    total = diffusion + clean_ar_weight * (
        ar_nll_sum / ar_normalizer.clamp_min(1).to(ar_nll_sum.dtype)
    )
    return DuoLoss(
        total,
        nelbo_sum,
        nelbo_targets,
        ar_nll_sum,
        ar_targets,
        denoising_correct,
        corruption,
    )


@torch.no_grad()
def accumulate_duo_validation_stats(
    model: DuoModel,
    batches: Iterable[DuoBatch | PreparedDuoValidationBatch],
    *,
    canvas_length: int,
    branches: int,
    seed: int,
    schedule: DuoSchedule = DuoSchedule(),
    objective_fn: Callable[..., Tensor] = duo_nelbo_token_loss,
    total_rows: int | None = None,
    expected_row_ids: Tensor | None = None,
    compute_ar_diagnostic: bool = False,
) -> Tensor:
    """Accumulate six additive validation statistics for one ledger shard.

    ``row_ids`` always refer to the global validation ledger.  Supplying
    ``expected_row_ids`` authenticates a disjoint distributed shard without
    requiring each rank to materialize or falsely claim every validation row.
    The returned FP64 tensor is directly all-reducible.
    """

    model.eval()
    device = next(model.parameters()).device
    materialized = tuple(batches) if total_rows is None else None
    stream = iter(materialized) if materialized is not None else iter(batches)
    if materialized is not None:
        if not materialized:
            raise ValueError("Duo validation needs at least one batch")
        declared_row_ids = tuple(
            (item.batch if isinstance(item, PreparedDuoValidationBatch) else item).row_ids
            for item in materialized
            if (item.batch if isinstance(item, PreparedDuoValidationBatch) else item).row_ids
            is not None
        )
        total_rows = (
            int(torch.cat(declared_row_ids).max()) + 1
            if declared_row_ids
            else sum(
                (item.batch if isinstance(item, PreparedDuoValidationBatch) else item)
                .clean_ids.shape[0]
                for item in materialized
            )
        )
    if total_rows is None or total_rows <= 0:
        raise ValueError("Duo validation needs a positive declared row count")
    if expected_row_ids is None:
        expected = torch.ones(total_rows, dtype=torch.bool)
    else:
        if expected_row_ids.device.type != "cpu":
            expected_row_ids = expected_row_ids.cpu()
        if expected_row_ids.dtype != torch.long or expected_row_ids.ndim != 1:
            raise ValueError("expected validation row ids must be rank-1 int64")
        if bool(
            ((expected_row_ids < 0) | (expected_row_ids >= total_rows)).any()
        ):
            raise ValueError("expected validation row ids lie outside the ledger")
        if expected_row_ids.unique().numel() != expected_row_ids.numel():
            raise ValueError("expected validation row ids must be unique")
        expected = torch.zeros(total_rows, dtype=torch.bool)
        expected[expected_row_ids] = True
    fallback_offset = 0
    seen = torch.zeros(total_rows, dtype=torch.bool)
    totals = torch.zeros(6, dtype=torch.float64, device=device)
    observed_batches = 0
    for item in stream:
        if isinstance(item, PreparedDuoValidationBatch):
            batch = item.batch
            prepared = item.prepared
        else:
            batch = item
            prepared = None
        row_ids = batch.row_ids
        if row_ids is None:
            row_ids = torch.arange(
                fallback_offset,
                fallback_offset + batch.clean_ids.shape[0],
            )
        elif row_ids.device.type != "cpu":
            row_ids = row_ids.cpu()
        if bool(((row_ids < 0) | (row_ids >= total_rows)).any()):
            raise ValueError("Duo validation row ids lie outside the ledger")
        if row_ids.unique().numel() != row_ids.numel():
            raise ValueError("Duo validation row ids must be unique")
        if bool((~expected.index_select(0, row_ids)).any()):
            raise ValueError("Duo validation batches included an unassigned row")
        if bool(seen.index_select(0, row_ids).any()):
            raise ValueError("Duo validation ledger row ids must be unique")
        seen[row_ids] = True
        fallback_offset += batch.clean_ids.shape[0]
        observed_batches += 1
        if prepared is None:
            prepared = prepare_duo_validation_inputs(
                model,
                batch,
                row_ids=row_ids.to(device, non_blocking=True),
                total_rows=total_rows,
                canvas_length=canvas_length,
                branches=branches,
                seed=seed,
                schedule=schedule,
            )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            loss = duo_loss(
                model,
                batch,
                canvas_length=canvas_length,
                branches=branches,
                prepared=prepared,
                schedule=schedule,
                compute_ar_diagnostic=compute_ar_diagnostic,
                compute_denoising_accuracy=True,
                objective_fn=objective_fn,
            )
        totals += torch.stack(
            (
                loss.nelbo_sum.double(),
                loss.nelbo_targets.double(),
                loss.ar_nll_sum.double(),
                loss.ar_targets.double(),
                loss.denoising_correct.double(),
                prepared.corruption.changed.sum().double(),
            )
        )
    if not torch.equal(seen, expected):
        raise ValueError("Duo validation batches did not cover the assigned ledger shard")
    if observed_batches == 0 and bool(expected.any()):
        raise ValueError("Duo validation needs at least one assigned batch")
    return totals


def duo_validation_from_stats(totals: Tensor) -> DuoValidation:
    """Convert six globally summed sufficient statistics into user metrics."""

    if totals.shape != (6,) or not totals.is_floating_point():
        raise ValueError("Duo validation statistics must be six floating values")
    targets = int(totals[1])
    ar_targets = int(totals[3])
    return DuoValidation(
        conditional_canvas_nelbo_nats_per_atom=float(
            totals[0] / max(targets, 1)
        ),
        clean_ar_nats_per_atom=(
            float(totals[2] / ar_targets) if ar_targets else None
        ),
        ar_anchor_bpb=(
            float(totals[2] / ar_targets / math.log(2.0))
            if ar_targets
            else None
        ),
        denoising_accuracy=float(totals[4] / max(targets, 1)),
        targets=targets,
        changed_targets=int(totals[5]),
        ar_targets=ar_targets,
    )


@torch.no_grad()
def validate_duo(
    model: DuoModel,
    batches: Iterable[DuoBatch | PreparedDuoValidationBatch],
    *,
    canvas_length: int,
    branches: int,
    seed: int,
    schedule: DuoSchedule = DuoSchedule(),
    objective_fn: Callable[..., Tensor] = duo_nelbo_token_loss,
    total_rows: int | None = None,
    compute_ar_diagnostic: bool = False,
) -> DuoValidation:
    totals = accumulate_duo_validation_stats(
        model,
        batches,
        canvas_length=canvas_length,
        branches=branches,
        seed=seed,
        schedule=schedule,
        objective_fn=objective_fn,
        total_rows=total_rows,
        compute_ar_diagnostic=compute_ar_diagnostic,
    )
    return duo_validation_from_stats(totals)


__all__ = (
    "DUO_TRAINING_OBJECTIVES",
    "DuoBatch",
    "DuoLoss",
    "DuoValidation",
    "JOINT_DUO_CLEAN_AR_OBJECTIVE",
    "PURE_DUO_OBJECTIVE",
    "PreparedDuoInputs",
    "PreparedDuoMicrobatch",
    "PreparedDuoUpdate",
    "PreparedDuoValidationBatch",
    "accumulate_duo_validation_stats",
    "duo_validation_from_stats",
    "duo_loss",
    "duo_clean_ar_weight",
    "duo_objective_contract",
    "prepare_duo_inputs",
    "prepare_duo_update",
    "prepare_duo_validation_batch",
    "prepare_duo_validation_inputs",
    "validate_duo",
)
