"""Complete loss and optimizer-step path for the standalone DiffusionGemma cell."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import AtomicVocabulary
from .diffusion_gemma import (
    UniformReplacementBatch,
    integrated_exact_k_uniform_replacement,
    uniform_replacement_corruption,
)
from .diffusion_gemma_model import (
    DiffusionGemmaAttentionMetadata,
    DiffusionGemmaModel,
    DiffusionGemmaOutput,
)


IGNORE_INDEX = -100


@dataclass(frozen=True)
class DocumentCanvasSelection:
    starts: Tensor
    valid: Tensor
    targets: Tensor
    document_ids: Tensor
    positions: Tensor

    def validate(self, clean_ids: Tensor, clean_valid: Tensor) -> None:
        if self.starts.ndim != 2 or self.starts.dtype != torch.long:
            raise ValueError("canvas starts must be [B,M] int64")
        expected = (self.starts.shape[0], self.starts.shape[1], self.valid.shape[-1])
        aligned = (self.valid, self.targets, self.document_ids, self.positions)
        if any(value.shape != expected for value in aligned):
            raise ValueError("canvas selection tensors do not align")
        if self.valid.dtype != torch.bool:
            raise TypeError("canvas validity must be boolean")
        if any(value.dtype != torch.long for value in aligned[1:]):
            raise TypeError("canvas targets/metadata must be int64")
        if clean_ids.shape != clean_valid.shape or clean_ids.shape[0] != expected[0]:
            raise ValueError("clean bank and selected canvases do not align")


@dataclass(frozen=True)
class DiffusionGemmaBatch:
    clean_ids: Tensor
    clean_valid: Tensor
    document_ids: Tensor
    positions: Tensor
    ar_targets: Tensor
    bos_targets: Tensor
    attention_metadata: DiffusionGemmaAttentionMetadata | None = None

    def validate(self, vocab: AtomicVocabulary) -> None:
        shape = self.clean_ids.shape
        if self.clean_ids.ndim != 2:
            raise ValueError("clean ids must be rank-2")
        if any(
            value.shape != shape
            for value in (
                self.clean_valid,
                self.document_ids,
                self.positions,
                self.ar_targets,
            )
        ):
            raise ValueError("DiffusionGemma batch tensors must align")
        if self.clean_ids.dtype != torch.long or self.clean_valid.dtype != torch.bool:
            raise TypeError("clean ids must be int64 and validity boolean")
        if any(
            value.dtype != torch.long
            for value in (
                self.document_ids,
                self.positions,
                self.ar_targets,
                self.bos_targets,
            )
        ):
            raise TypeError("targets and document metadata must be int64")
        torch._assert_async(
            (
                ~self.clean_valid
                | self.clean_ids.ge(0) & self.clean_ids.lt(vocab.output_size)
            ).all(),
            "valid clean ids lie outside the output vocabulary",
        )
        if self.attention_metadata is not None:
            metadata = self.attention_metadata
            if metadata.byte_indices.device != self.clean_ids.device:
                raise ValueError("attention metadata and batch must share a device")

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "DiffusionGemmaBatch":
        def move(tensor: Tensor) -> Tensor:
            return tensor.to(device, non_blocking=non_blocking)

        return DiffusionGemmaBatch(
            clean_ids=move(self.clean_ids),
            clean_valid=move(self.clean_valid),
            document_ids=move(self.document_ids),
            positions=move(self.positions),
            ar_targets=move(self.ar_targets),
            bos_targets=move(self.bos_targets),
            attention_metadata=(
                None
                if self.attention_metadata is None
                else self.attention_metadata.to(device, non_blocking=non_blocking)
            ),
        )

    def pin_memory(self) -> "DiffusionGemmaBatch":
        def pin(tensor: Tensor) -> Tensor:
            return tensor if tensor.is_pinned() else tensor.pin_memory()

        metadata = self.attention_metadata
        return DiffusionGemmaBatch(
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


@dataclass(frozen=True)
class DiffusionGemmaLoss:
    total: Tensor
    diffusion: Tensor
    clean_ar: Tensor
    headline_bpb: Tensor
    diffusion_targets: Tensor
    ar_targets: Tensor
    changed_targets: Tensor
    unchanged_targets: Tensor
    noise_fraction: Tensor
    output: DiffusionGemmaOutput
    corruption: UniformReplacementBatch
    selection: DocumentCanvasSelection
    diffusion_nll_sum: Tensor
    ar_nll_sum: Tensor


@dataclass(frozen=True)
class PreparedDiffusionGemmaInputs:
    """Stochastic canvas inputs sampled once for an optimizer update."""

    selection: DocumentCanvasSelection
    corruption: UniformReplacementBatch

    def slice_rows(self, start: int, stop: int) -> "PreparedDiffusionGemmaInputs":
        branches = self.selection.starts.shape[1]
        flat_start = start * branches
        flat_stop = stop * branches
        selection = DocumentCanvasSelection(
            starts=self.selection.starts[start:stop],
            valid=self.selection.valid[start:stop],
            targets=self.selection.targets[start:stop],
            document_ids=self.selection.document_ids[start:stop],
            positions=self.selection.positions[start:stop],
        )
        corruption = UniformReplacementBatch(
            ids=self.corruption.ids[flat_start:flat_stop],
            targets=self.corruption.targets[flat_start:flat_stop],
            active=self.corruption.active[flat_start:flat_stop],
            replaced=self.corruption.replaced[flat_start:flat_stop],
            changed=self.corruption.changed[flat_start:flat_stop],
            unchanged=self.corruption.unchanged[flat_start:flat_stop],
            noise_fraction=self.corruption.noise_fraction[flat_start:flat_stop],
            changed_fraction=self.corruption.changed_fraction[flat_start:flat_stop],
            t=(
                None
                if self.corruption.t is None
                else self.corruption.t[flat_start:flat_stop]
            ),
            k=self.corruption.k[flat_start:flat_stop],
            integrated_exact_k=self.corruption.integrated_exact_k,
        )
        return PreparedDiffusionGemmaInputs(selection, corruption)

    def pin_memory(self) -> "PreparedDiffusionGemmaInputs":
        def pin(value: Tensor) -> Tensor:
            return value if value.is_pinned() else value.pin_memory()

        selection = self.selection
        corruption = self.corruption
        return PreparedDiffusionGemmaInputs(
            DocumentCanvasSelection(
                starts=pin(selection.starts),
                valid=pin(selection.valid),
                targets=pin(selection.targets),
                document_ids=pin(selection.document_ids),
                positions=pin(selection.positions),
            ),
            UniformReplacementBatch(
                ids=pin(corruption.ids),
                targets=pin(corruption.targets),
                active=pin(corruption.active),
                replaced=pin(corruption.replaced),
                changed=pin(corruption.changed),
                unchanged=pin(corruption.unchanged),
                noise_fraction=pin(corruption.noise_fraction),
                changed_fraction=pin(corruption.changed_fraction),
                t=None if corruption.t is None else pin(corruption.t),
                k=pin(corruption.k),
                integrated_exact_k=corruption.integrated_exact_k,
            ),
        )

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "PreparedDiffusionGemmaInputs":
        def move(value: Tensor) -> Tensor:
            return value.to(device, non_blocking=non_blocking)

        selection = self.selection
        corruption = self.corruption
        return PreparedDiffusionGemmaInputs(
            DocumentCanvasSelection(
                starts=move(selection.starts),
                valid=move(selection.valid),
                targets=move(selection.targets),
                document_ids=move(selection.document_ids),
                positions=move(selection.positions),
            ),
            UniformReplacementBatch(
                ids=move(corruption.ids),
                targets=move(corruption.targets),
                active=move(corruption.active),
                replaced=move(corruption.replaced),
                changed=move(corruption.changed),
                unchanged=move(corruption.unchanged),
                noise_fraction=move(corruption.noise_fraction),
                changed_fraction=move(corruption.changed_fraction),
                t=None if corruption.t is None else move(corruption.t),
                k=move(corruption.k),
                integrated_exact_k=corruption.integrated_exact_k,
            ),
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
            self.corruption.unchanged,
            self.corruption.noise_fraction,
            self.corruption.changed_fraction,
            self.corruption.k,
        ):
            value.record_stream(stream)
        if self.corruption.t is not None:
            self.corruption.t.record_stream(stream)


@dataclass(frozen=True)
class PreparedDiffusionGemmaValidationBatch:
    batch: DiffusionGemmaBatch
    prepared: PreparedDiffusionGemmaInputs

    def pin_memory(self) -> "PreparedDiffusionGemmaValidationBatch":
        return PreparedDiffusionGemmaValidationBatch(
            self.batch.pin_memory(), self.prepared.pin_memory()
        )

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "PreparedDiffusionGemmaValidationBatch":
        return PreparedDiffusionGemmaValidationBatch(
            self.batch.to(device, non_blocking=non_blocking),
            self.prepared.to(device, non_blocking=non_blocking),
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        self.batch.record_stream(stream)
        self.prepared.record_stream(stream)


@dataclass(frozen=True)
class DiffusionGemmaValidation:
    """Held-row diagnostics; none is presented as a diffusion likelihood."""

    denoising_ce_nats_per_atom: float
    ar_anchor_bpb: float
    denoising_accuracy: float
    changed_accuracy: float
    unchanged_accuracy: float
    diffusion_targets: int
    ar_targets: int
    changed_targets: int
    unchanged_targets: int


def select_document_canvases(
    clean_ids: Tensor,
    clean_valid: Tensor,
    document_ids: Tensor,
    positions: Tensor,
    *,
    canvas_length: int,
    branches: int,
    patch_stride: int,
    pad_id: int,
    generator: torch.Generator | None = None,
    validate_candidates: bool = True,
) -> DocumentCanvasSelection:
    """Uniformly sample document-relative physical-patch origins.

    A start is eligible exactly when it is valid and its document-relative
    position is patch aligned.  Canvas storage is always fixed shape, while
    validity stops at the first document boundary.  This includes short tails
    without ever borrowing supervision from the next packed document.
    """

    if clean_ids.ndim != 2 or any(
        value.shape != clean_ids.shape
        for value in (clean_valid, document_ids, positions)
    ):
        raise ValueError("clean canvas source tensors must align as [B,L]")
    if clean_ids.dtype != torch.long or clean_valid.dtype != torch.bool:
        raise TypeError("clean ids must be int64 and valid boolean")
    if document_ids.dtype != torch.long or positions.dtype != torch.long:
        raise TypeError("document metadata must be int64")
    if canvas_length <= 0 or canvas_length % patch_stride:
        raise ValueError("canvas length must be a positive patch multiple")
    if branches <= 0:
        raise ValueError("branches must be positive")

    length = clean_ids.shape[1]
    candidates = (
        clean_valid
        & positions.remainder(patch_stride).eq(0)
    )
    if (
        validate_candidates
        and not torch.compiler.is_compiling()
        and bool(~candidates.any(1).all())
    ):
        raise ValueError("every row needs at least one document-relative canvas origin")
    # A single random phase plus evenly spaced rotations gives every branch a
    # uniform marginal origin while covering the row far more evenly than
    # independent replacement draws. Short rows remain valid: rotations then
    # repeat origins deterministically instead of failing without replacement.
    candidate_counts = candidates.sum(1)
    phase = torch.rand(
        (clean_ids.shape[0], 1), device=clean_ids.device, generator=generator
    )
    rotations = torch.arange(
        branches, dtype=torch.float32, device=clean_ids.device
    )[None] / branches
    origin_ranks = torch.floor(
        (phase + rotations).remainder(1.0) * candidate_counts[:, None]
    ).to(torch.long)
    candidate_ranks = candidates.cumsum(1) - 1
    starts = (
        candidates[:, None]
        & candidate_ranks[:, None].eq(origin_ranks[:, :, None])
    ).to(torch.long).argmax(-1)
    offsets = torch.arange(canvas_length, device=clean_ids.device)
    indices = starts[:, :, None] + offsets
    exists = indices.lt(length)
    safe_indices = indices.clamp_max(length - 1)
    expanded_ids = clean_ids[:, None, :].expand(-1, branches, -1)
    expanded_valid = clean_valid[:, None, :].expand(-1, branches, -1)
    expanded_documents = document_ids[:, None, :].expand(-1, branches, -1)
    targets = torch.gather(expanded_ids, 2, safe_indices)
    gathered_valid = torch.gather(expanded_valid, 2, safe_indices)
    gathered_documents = torch.gather(expanded_documents, 2, safe_indices)
    origin_documents = torch.gather(document_ids, 1, starts)[:, :, None]
    origin_positions = torch.gather(positions, 1, starts)[:, :, None]
    valid = (
        exists
        & gathered_valid
        & gathered_documents.eq(origin_documents)
    )
    targets = torch.where(valid, targets, pad_id)
    # Metadata for the synthetic PAD suffix remains document-relative and
    # monotone. Invalid entries never enter attention, but keeping their
    # geometry well formed makes CPU and compiled CUDA lowerings identical.
    gathered_documents = origin_documents.expand_as(indices)
    gathered_positions = origin_positions + offsets
    result = DocumentCanvasSelection(
        starts=starts,
        valid=valid,
        targets=targets,
        document_ids=gathered_documents,
        positions=gathered_positions,
    )
    result.validate(clean_ids, clean_valid)
    return result


def _cross_entropy_sum_and_count(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    flat_targets = targets.reshape(-1)
    active = flat_targets.ne(IGNORE_INDEX)
    total = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        flat_targets,
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return total, active.sum()


def diffusion_gemma_loss(
    model: DiffusionGemmaModel,
    batch: DiffusionGemmaBatch,
    *,
    canvas_length: int,
    branches: int,
    clean_ar_weight: float = 1.0,
    self_condition: bool = True,
    integrated_exact_k: bool = False,
    generator: torch.Generator | None = None,
    prepared: PreparedDiffusionGemmaInputs | None = None,
    self_conditioned_rows: Tensor | None = None,
    diffusion_denominator: Tensor | int | None = None,
    ar_denominator: Tensor | int | None = None,
) -> DiffusionGemmaLoss:
    """Dense unchanged-inclusive diffusion loss plus clean AR/BOS likelihood."""

    batch.validate(model.config.vocab)
    if not math.isfinite(clean_ar_weight) or clean_ar_weight < 0:
        raise ValueError("clean AR weight must be finite and nonnegative")
    if prepared is None:
        prepared = prepare_diffusion_gemma_inputs(
            model,
            batch,
            canvas_length=canvas_length,
            branches=branches,
            integrated_exact_k=integrated_exact_k,
            generator=generator,
        )
    selection = prepared.selection
    corruption = prepared.corruption
    selection.validate(batch.clean_ids, batch.clean_valid)
    noisy = corruption.ids.view_as(selection.targets)
    noisy = torch.where(selection.valid, noisy, model.config.vocab.pad_id)
    output = model(
        batch.clean_ids,
        batch.clean_valid,
        batch.document_ids,
        batch.positions,
        noisy,
        selection.valid,
        selection.starts,
        self_condition=self_condition,
        self_conditioned_rows=self_conditioned_rows,
        generator=generator,
        attention_metadata=batch.attention_metadata,
    )
    dense_targets = torch.where(
        selection.valid, selection.targets, IGNORE_INDEX
    )
    diffusion_total, diffusion_count = _cross_entropy_sum_and_count(
        output.branch_logits, dense_targets
    )
    diffusion = diffusion_total / diffusion_count.clamp_min(1).to(diffusion_total.dtype)

    clean_total, clean_count = _cross_entropy_sum_and_count(
        output.clean_logits, batch.ar_targets
    )
    bos_logits = model.forward_bos_logits(
        batch.bos_targets.numel(), device=batch.clean_ids.device
    )
    bos_total = F.cross_entropy(bos_logits, batch.bos_targets, reduction="sum")
    ar_total = clean_total + bos_total
    ar_count = clean_count + batch.bos_targets.numel()
    clean_ar = ar_total / ar_count.clamp_min(1).to(ar_total.dtype)
    headline_bpb = clean_ar / math.log(2.0)
    diffusion_normalizer = (
        diffusion_count if diffusion_denominator is None else diffusion_denominator
    )
    ar_normalizer = ar_count if ar_denominator is None else ar_denominator
    diffusion_contribution = diffusion_total / torch.as_tensor(
        diffusion_normalizer, device=diffusion_total.device
    ).clamp_min(1).to(diffusion_total.dtype)
    ar_contribution = ar_total / torch.as_tensor(
        ar_normalizer, device=ar_total.device
    ).clamp_min(1).to(ar_total.dtype)
    total = diffusion_contribution + clean_ar_weight * ar_contribution
    return DiffusionGemmaLoss(
        total=total,
        diffusion=diffusion,
        clean_ar=clean_ar,
        headline_bpb=headline_bpb,
        diffusion_targets=diffusion_count,
        ar_targets=ar_count,
        changed_targets=corruption.changed_targets.sum(),
        unchanged_targets=corruption.unchanged_targets.sum(),
        noise_fraction=corruption.noise_fraction.mean(),
        output=output,
        corruption=corruption,
        selection=selection,
        diffusion_nll_sum=diffusion_total,
        ar_nll_sum=ar_total,
    )


def prepare_diffusion_gemma_inputs(
    model: DiffusionGemmaModel,
    batch: DiffusionGemmaBatch,
    *,
    canvas_length: int,
    branches: int,
    integrated_exact_k: bool,
    generator: torch.Generator | None,
    validate_candidates: bool = True,
) -> PreparedDiffusionGemmaInputs:
    """Sample origins and corruption before update-global normalization."""

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
        validate_candidates=validate_candidates,
    )
    flat_targets = selection.targets.flatten(0, 1)
    flat_valid = selection.valid.flatten(0, 1)
    corruption = (
        integrated_exact_k_uniform_replacement(
            flat_targets,
            flat_valid,
            output_size=model.config.vocab.output_size,
            generator=generator,
        )
        if integrated_exact_k
        else uniform_replacement_corruption(
            flat_targets,
            flat_valid,
            output_size=model.config.vocab.output_size,
            generator=generator,
        )
    )
    return PreparedDiffusionGemmaInputs(selection, corruption)


def diffusion_gemma_optimizer_step(
    model: DiffusionGemmaModel,
    optimizer: torch.optim.Optimizer,
    batch: DiffusionGemmaBatch,
    **loss_kwargs: object,
) -> DiffusionGemmaLoss:
    """One ordinary differentiable optimizer step, useful as the CPU oracle."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = diffusion_gemma_loss(model, batch, **loss_kwargs)
    losses.total.backward()
    optimizer.step()
    return losses


def validate_diffusion_gemma(
    model: DiffusionGemmaModel,
    batches: Iterable[DiffusionGemmaBatch | PreparedDiffusionGemmaValidationBatch],
    *,
    canvas_length: int,
    branches: int,
    seed: int,
    self_condition: bool = True,
    integrated_exact_k: bool = True,
) -> DiffusionGemmaValidation:
    """Evaluate deterministic denoising CE and a separate clean-AR anchor.

    Uniform-replacement denoising CE is not the absorbing-diffusion ELBO used
    by Fast-BLT and therefore is deliberately not converted to bits-per-byte.
    It is a within-recipe learning diagnostic; downstream generation accuracy
    is the promotion metric for this cell.
    """

    was_training = model.training
    model.eval()
    generator = torch.Generator(device=model.embedding.weight.device).manual_seed(seed)
    # Keep the complete validation reduction resident on device.  The old
    # implementation converted ten CUDA scalars to Python for every batch,
    # serializing the launch stream and leaving the GPU idle between otherwise
    # large forwards.  One final transfer authenticates the aggregate without
    # changing any metric semantics.
    totals = torch.zeros(8, dtype=torch.float64, device=model.embedding.weight.device)
    with torch.inference_mode():
        for item in batches:
            if isinstance(item, PreparedDiffusionGemmaValidationBatch):
                batch, prepared = item.batch, item.prepared
            else:
                batch, prepared = item, None
            loss = diffusion_gemma_loss(
                model,
                batch,
                canvas_length=canvas_length,
                branches=branches,
                clean_ar_weight=0.0,
                self_condition=self_condition,
                integrated_exact_k=integrated_exact_k,
                generator=generator,
                prepared=prepared,
            )
            predictions = loss.output.branch_logits.argmax(-1).flatten(0, 1)
            targets = loss.selection.targets.flatten(0, 1)
            correct = predictions.eq(targets)
            changed = loss.corruption.changed
            unchanged = loss.corruption.unchanged
            totals += torch.stack(
                (
                    loss.diffusion_nll_sum.detach().double(),
                    loss.ar_nll_sum.detach().double(),
                    loss.diffusion_targets.detach().double(),
                    loss.ar_targets.detach().double(),
                    (correct & changed).sum().double(),
                    (correct & unchanged).sum().double(),
                    changed.sum().double(),
                    unchanged.sum().double(),
                )
            )
    model.train(was_training)
    observed = totals.cpu().numpy()
    diffusion_nll, ar_nll = map(float, observed[:2])
    (
        diffusion_targets,
        ar_targets,
        changed_correct,
        unchanged_correct,
        changed_targets,
        unchanged_targets,
    ) = map(int, observed[2:])
    if diffusion_targets <= 0 or ar_targets <= 0:
        raise ValueError("DiffusionGemma validation requires nonempty supervision")
    return DiffusionGemmaValidation(
        denoising_ce_nats_per_atom=diffusion_nll / diffusion_targets,
        ar_anchor_bpb=ar_nll / ar_targets / math.log(2.0),
        denoising_accuracy=(changed_correct + unchanged_correct) / diffusion_targets,
        changed_accuracy=changed_correct / max(1, changed_targets),
        unchanged_accuracy=unchanged_correct / max(1, unchanged_targets),
        diffusion_targets=diffusion_targets,
        ar_targets=ar_targets,
        changed_targets=changed_targets,
        unchanged_targets=unchanged_targets,
    )


__all__ = (
    "DiffusionGemmaBatch",
    "DiffusionGemmaLoss",
    "DiffusionGemmaValidation",
    "DocumentCanvasSelection",
    "PreparedDiffusionGemmaInputs",
    "PreparedDiffusionGemmaValidationBatch",
    "diffusion_gemma_loss",
    "diffusion_gemma_optimizer_step",
    "prepare_diffusion_gemma_inputs",
    "select_document_canvases",
    "validate_diffusion_gemma",
)
