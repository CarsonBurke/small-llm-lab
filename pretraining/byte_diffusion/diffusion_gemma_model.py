"""Standalone BLT-shaped DiffusionGemma cell.

The shared byte model intentionally carries objective mode and optional time
embeddings.  DiffusionGemma does neither, so this module owns a separate cell
instead of hiding those parameters behind zero-valued inputs.  Clean tokens
and every noisy canvas live in one physical bank: the causal clean computation
is performed once and canvas queries see only their own document prefix and
their own bidirectional canvas.

Clean documents are packed into native variable-length causal attention and
canvas queries use the repository's shared-bank FlexAttention kernel.  The
only dense path is the explicit CPU correctness oracle; CUDA fails closed if
its precomputed document layout is omitted and never materializes an O(L^2)
mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import (
    CanvasBlockMaskMetadata,
    CanvasBranchLayout,
    build_canvas_block_mask,
)
from .config import ByteDiffusionConfig
from .diffusion_gemma import sample_static_half_batch_mask
from .layers import (
    ConditionedTransformerBlock,
    PatchPool,
    RMSNorm,
    TransformerBlock,
    packed_sequence_offsets,
    valid_row_indices,
)
from .model import CausalNgramEmbedding


DIFFUSION_GEMMA_PARAMETER_COUNT = 23_043_330


@dataclass(frozen=True)
class DiffusionGemmaOutput:
    clean_logits: Tensor
    branch_logits: Tensor
    clean_patch_states: Tensor
    branch_patch_states: Tensor
    self_conditioned_rows: Tensor | None = None
    prior_batch_size: int = 0
    # Byte-Duo can feed these states directly to fused linear cross-entropy,
    # avoiding a full [batch, 8192, 261] clean-logit materialization. Other
    # cells retain their existing public logits contract and leave this unset.
    clean_decoder_states: Tensor | None = None


@dataclass(frozen=True)
class DiffusionGemmaAttentionMetadata:
    """Document-packed indices shared by every layer in one model forward."""

    byte_indices: Tensor
    byte_cu_seqlens: Tensor
    patch_indices: Tensor
    patch_cu_seqlens: Tensor
    physical_layout: bool = False

    def __post_init__(self) -> None:
        if self.byte_indices.ndim != 1 or self.byte_indices.dtype != torch.long:
            raise ValueError("byte indices must be rank-1 int64")
        if self.patch_indices.ndim != 1 or self.patch_indices.dtype != torch.long:
            raise ValueError("patch indices must be rank-1 int64")
        for name, offsets in (
            ("byte", self.byte_cu_seqlens),
            ("patch", self.patch_cu_seqlens),
        ):
            if offsets.ndim != 1 or offsets.dtype != torch.int32:
                raise ValueError(f"{name} cumulative lengths must be rank-1 int32")
            if offsets.numel() < 2:
                raise ValueError(f"{name} cumulative lengths cannot be empty")
        devices = {
            self.byte_indices.device,
            self.byte_cu_seqlens.device,
            self.patch_indices.device,
            self.patch_cu_seqlens.device,
        }
        if len(devices) != 1:
            raise ValueError("DiffusionGemma attention metadata must share a device")
        if not isinstance(self.physical_layout, bool):
            raise TypeError("physical-layout marker must be boolean")

    def to(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> "DiffusionGemmaAttentionMetadata":
        return DiffusionGemmaAttentionMetadata(
            self.byte_indices.to(device, non_blocking=non_blocking),
            self.byte_cu_seqlens.to(device, non_blocking=non_blocking),
            self.patch_indices.to(device, non_blocking=non_blocking),
            self.patch_cu_seqlens.to(device, non_blocking=non_blocking),
            self.physical_layout,
        )

def compile_stable_document_metadata(
    valid: Tensor,
    document_ids: Tensor,
    *,
    patch_stride: int,
    max_segments_per_row: int,
) -> DiffusionGemmaAttentionMetadata:
    """Build fixed-capacity packed metadata for compiled document attention.

    Document-aligned pages place at most ``patch_stride - 1`` invalid PAD
    slots after a document's EOT.  Packing those physical slots with the
    preceding document cannot affect a real token under causal attention, and
    it makes both byte and patch token counts independent of batch contents.
    Unused sequence slots are represented by repeated terminal cumulative
    offsets; native varlen attention supports these empty tail sequences.
    """

    if valid.device.type != "cpu" or document_ids.device.type != "cpu":
        raise ValueError("compile-stable document metadata must be built on CPU")
    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError("valid must be rank-2 boolean")
    if document_ids.shape != valid.shape or document_ids.dtype != torch.long:
        raise ValueError("document ids must be aligned int64")
    if patch_stride <= 0 or valid.shape[1] % patch_stride:
        raise ValueError("physical rows must be patch aligned")
    if max_segments_per_row <= 0:
        raise ValueError("max segments per row must be positive")
    if bool((document_ids.masked_select(valid) < 0).any()):
        raise ValueError("valid positions require nonnegative document ids")
    adjacent_valid = valid[:, 1:] & valid[:, :-1]
    if bool(
        (
            adjacent_valid
            & document_ids[:, 1:].lt(document_ids[:, :-1])
        ).any()
    ):
        raise ValueError("document ids must be nondecreasing within physical rows")

    physical_documents = torch.where(valid, document_ids, -1).cummax(1).values
    if bool(physical_documents[:, 0].lt(0).any()):
        raise ValueError("a physical row cannot begin with alignment padding")
    grouped_documents = physical_documents.view(
        valid.shape[0], -1, patch_stride
    )
    if bool(grouped_documents.ne(grouped_documents[:, :, :1]).any()):
        raise ValueError("a physical patch cannot straddle documents")

    def fixed_offsets(level_documents: Tensor) -> Tensor:
        rows, width = level_documents.shape
        row_boundary = torch.ones_like(level_documents, dtype=torch.bool)
        row_boundary[:, 1:] = level_documents[:, 1:] != level_documents[:, :-1]
        row_segment_counts = row_boundary.sum(1)
        if bool(row_segment_counts.gt(max_segments_per_row).any()):
            raise ValueError(
                "one physical row exceeds the fixed packed-document capacity"
            )
        flat_documents = level_documents.reshape(-1)
        flat_rows = torch.arange(rows)[:, None].expand(rows, width).reshape(-1)
        boundary = torch.ones_like(flat_documents, dtype=torch.bool)
        boundary[1:] = (flat_rows[1:] != flat_rows[:-1]) | (
            flat_documents[1:] != flat_documents[:-1]
        )
        starts = torch.nonzero(boundary, as_tuple=False).flatten()
        segment_count = starts.numel()
        capacity = rows * max_segments_per_row
        if segment_count > capacity:
            raise ValueError(
                f"packed document count {segment_count} exceeds fixed capacity {capacity}"
            )
        lengths = torch.diff(
            torch.cat((starts, starts.new_tensor([flat_documents.numel()])))
        )
        offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32),
                lengths.to(torch.int32).cumsum(0, dtype=torch.int32),
            )
        )
        empty_tail = offsets.new_full(
            (capacity - segment_count,), int(flat_documents.numel())
        )
        return torch.cat((offsets, empty_tail))

    batch, length = valid.shape
    byte_indices = torch.arange(batch * length, dtype=torch.long)
    patch_documents = grouped_documents[:, :, 0]
    patches_per_row = patch_documents.shape[1]
    patch_indices = torch.arange(batch * patches_per_row, dtype=torch.long)
    return DiffusionGemmaAttentionMetadata(
        byte_indices,
        fixed_offsets(physical_documents),
        patch_indices,
        fixed_offsets(patch_documents),
        True,
    )


def _packed_document_metadata(
    valid: Tensor,
    document_ids: Tensor,
    *,
    patch_stride: int,
) -> DiffusionGemmaAttentionMetadata:
    """Build exact packed byte/patch document boundaries once per batch."""

    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError("valid must be rank-2 boolean")
    if document_ids.shape != valid.shape or document_ids.dtype != torch.long:
        raise ValueError("document ids must be aligned int64")
    if valid.shape[1] % patch_stride:
        raise ValueError("physical rows must be patch aligned")

    def indices_and_cu(
        level_valid: Tensor, level_documents: Tensor
    ) -> tuple[Tensor, Tensor]:
        indices = valid_row_indices(level_valid)
        flat_documents = level_documents.reshape(-1).index_select(0, indices)
        rows = torch.div(indices, level_valid.shape[1], rounding_mode="floor")
        boundary = torch.ones_like(flat_documents, dtype=torch.bool)
        boundary[1:] = (rows[1:] != rows[:-1]) | (
            flat_documents[1:] != flat_documents[:-1]
        )
        starts = torch.nonzero(boundary, as_tuple=False).flatten()
        stops = torch.cat((starts[1:], starts.new_tensor([indices.numel()])))
        return indices, packed_sequence_offsets(stops - starts)

    byte_indices, byte_cu = indices_and_cu(valid, document_ids)
    batch, length = valid.shape
    patch_valid = valid.view(batch, length // patch_stride, patch_stride).any(-1)
    grouped_documents = document_ids.view(
        batch, length // patch_stride, patch_stride
    )
    patch_documents = grouped_documents[:, :, 0]
    if not torch.compiler.is_compiling():
        mismatch = valid.view_as(grouped_documents) & grouped_documents.ne(
            patch_documents[:, :, None]
        )
        if bool(mismatch.any()):
            raise ValueError("a physical patch cannot straddle documents")
    patch_indices, patch_cu = indices_and_cu(patch_valid, patch_documents)
    return DiffusionGemmaAttentionMetadata(
        byte_indices, byte_cu, patch_indices, patch_cu
    )


def packed_patch_cu_seqlens(
    patch_indices: Tensor,
    valid: Tensor,
    document_ids: Tensor,
    *,
    patch_stride: int,
) -> Tensor:
    """Document offsets for the exact prepacked physical patch bank.

    Valid byte lengths are not generally divisible by the patch stride: the
    document-aligned dataset inserts up to three *invalid* physical PAD slots
    after EOT. Consequently ``byte_cu_seqlens // stride`` is wrong whenever a
    UTF-8 document has a short final patch. Derive patch boundaries from the
    physical patch indices and their valid document owner instead.
    """

    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError("valid must be rank-2 boolean")
    if document_ids.shape != valid.shape or document_ids.dtype != torch.long:
        raise ValueError("document ids must be aligned int64")
    if valid.shape[1] % patch_stride:
        raise ValueError("physical rows must be patch aligned")
    patches_per_row = valid.shape[1] // patch_stride
    grouped_valid = valid.view(valid.shape[0], patches_per_row, patch_stride)
    grouped_documents = document_ids.view_as(grouped_valid)
    patch_documents = torch.where(
        grouped_valid,
        grouped_documents,
        torch.full_like(grouped_documents, -1),
    ).amax(-1)
    patch_rows = torch.div(patch_indices, patches_per_row, rounding_mode="floor")
    packed_documents = patch_documents.reshape(-1).index_select(0, patch_indices)
    boundary = torch.ones_like(packed_documents, dtype=torch.bool)
    boundary[1:] = (patch_rows[1:] != patch_rows[:-1]) | (
        packed_documents[1:] != packed_documents[:-1]
    )
    starts = boundary.nonzero(as_tuple=False).flatten()
    lengths = torch.diff(
        torch.cat((starts, starts.new_tensor([patch_indices.numel()])))
    )
    return packed_sequence_offsets(lengths)


def _patch_metadata(
    valid: Tensor,
    segment_ids: Tensor,
    positions: Tensor,
    stride: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if valid.shape[1] % stride:
        raise ValueError("physical length must be patch aligned")
    batch, length = valid.shape
    patch_valid = valid.view(batch, length // stride, stride).any(-1)
    grouped_segments = segment_ids.view(batch, length // stride, stride)
    grouped_valid = valid.view(batch, length // stride, stride)
    first = grouped_segments[:, :, 0]
    if not torch.compiler.is_compiling():
        mismatch = grouped_valid & grouped_segments.ne(first[:, :, None])
        if bool(mismatch.any()):
            raise ValueError("a patch cannot straddle document segments")
    patch_positions = positions[:, ::stride] // stride
    return patch_valid, first, patch_positions


class DiffusionGemmaModel(nn.Module):
    """Mode-free, time-free byte DiffusionGemma over a shared BLT backbone."""

    def __init__(self, config: ByteDiffusionConfig = ByteDiffusionConfig()) -> None:
        super().__init__()
        if config.explicit_timestep:
            raise ValueError("DiffusionGemma is explicitly timestep-free")
        if config.self_conditioning:
            raise ValueError(
                "self-conditioning belongs to this cell, not the shared model flag"
            )
        self.config = config
        self.embedding = nn.Embedding(
            config.vocab.input_size,
            config.local_dim,
            padding_idx=config.vocab.pad_id,
        )
        self.ngrams = CausalNgramEmbedding(config) if config.ngram_enabled else None
        self.encoder = nn.ModuleList(
            TransformerBlock(
                config.local_dim,
                config.local_heads,
                config.encoder_ffn_dim,
                config.rope_theta,
            )
            for _ in range(config.encoder_layers)
        )
        self.pool = PatchPool(
            config.local_dim, config.global_dim, config.global_heads, config.patch_stride
        )
        self.global_blocks = nn.ModuleList(
            TransformerBlock(
                config.global_dim,
                config.global_heads,
                config.global_ffn_dim,
                config.rope_theta,
                config.global_ffn_kind,
            )
            for _ in range(config.global_layers)
        )
        self.global_norm = RMSNorm(config.global_dim)
        self.decoder = nn.ModuleList(
            ConditionedTransformerBlock(
                config.local_dim,
                config.global_dim,
                config.local_heads,
                config.decoder_ffn_dim,
                config.rope_theta,
                config.decoder_conditioning,
                config.decoder_split_residual_scale,
            )
            for _ in range(config.decoder_layers)
        )
        self.decoder_norm = RMSNorm(config.local_dim)
        self.output = (
            None
            if config.output_tied
            else nn.Linear(config.local_dim, config.vocab.output_size, bias=False)
        )
        # The paper applies one FFW to E[p(token)]. This budgeted BLT cell keeps
        # that shared projection, returns to full local width, and injects the
        # same prediction feature into both noisy encoder and decoder streams.
        self.self_conditioner = nn.Sequential(
            RMSNorm(config.local_dim),
            nn.Linear(
                config.local_dim, config.self_conditioning_hidden, bias=False
            ),
            nn.SiLU(),
            nn.Linear(
                config.self_conditioning_hidden, config.local_dim, bias=False
            ),
        )
        self.reset_parameters()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def validate_production_parameterization(self) -> None:
        if self.config != ByteDiffusionConfig():
            raise ValueError("the closed parameter count applies only to the default cell")
        if self.parameter_count != DIFFUSION_GEMMA_PARAMETER_COUNT:
            raise AssertionError(
                "DiffusionGemma parameterization drifted: "
                f"expected {DIFFUSION_GEMMA_PARAMETER_COUNT:,}, "
                f"observed {self.parameter_count:,}"
            )

    @torch.compiler.disable
    def prepare_attention_metadata(
        self, clean_valid: Tensor, document_ids: Tensor
    ) -> DiffusionGemmaAttentionMetadata:
        """Prepare document packing once for repeated denoising forwards."""

        return _packed_document_metadata(
            clean_valid,
            document_ids,
            patch_stride=self.config.patch_stride,
        )

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[self.config.vocab.pad_id].zero_()
            if self.ngrams is not None:
                nn.init.normal_(self.ngrams.projection_weight, mean=0.0, std=0.02)
                if self.config.ngram_factor_init == "scale_matched":
                    self.ngrams.projection_weight.mul_(
                        self.config.ngram_rank**-0.5 / 0.02
                    )

    def _logits(self, states: Tensor) -> Tensor:
        normalized = self.decoder_norm(states)
        if self.output is None:
            return F.linear(
                normalized, self.embedding.weight[: self.config.vocab.output_size]
            )
        return self.output(normalized)

    def project_self_conditioning(self, probabilities: Tensor) -> Tensor:
        """Paper-style ``FFW(stopgrad(p) @ E)`` at full local width."""

        if probabilities.ndim != 4:
            raise ValueError("self-conditioning probabilities must be [B,M,C,V]")
        if probabilities.shape[-1] != self.config.vocab.output_size:
            raise ValueError("self-conditioning probability vocabulary is incorrect")
        if not probabilities.is_floating_point():
            raise TypeError("self-conditioning probabilities must be floating point")
        expected = probabilities.detach().to(self.embedding.weight.dtype) @ self.embedding.weight[
            : self.config.vocab.output_size
        ]
        return self.self_conditioner(expected)

    def _isolated_ngram_features(
        self, ids: Tensor, valid: Tensor, segment_ids: Tensor
    ) -> Tensor:
        """Evaluate the shared n-gram tables with exact segment resets."""

        if self.ngrams is None:
            raise AssertionError("isolated n-grams requested while disabled")
        return self.ngrams.forward_segmented(ids, valid, segment_ids)

    def _ngram_features(
        self,
        clean_ids: Tensor,
        clean_valid: Tensor,
        document_ids: Tensor,
        noisy_ids: Tensor,
        branch_valid: Tensor,
        starts: Tensor,
    ) -> Tensor | None:
        if self.ngrams is None:
            return None
        batch, branches, canvas = noisy_ids.shape
        halo = max(self.config.ngram_orders) - 1
        offsets = torch.arange(halo, device=clean_ids.device)
        prefix_indices = starts[:, :, None] - halo + offsets
        exists = prefix_indices.ge(0)
        prefix = torch.gather(
            clean_ids[:, None, :].expand(-1, branches, -1),
            2,
            prefix_indices.clamp_min(0),
        )
        prefix_documents = torch.gather(
            document_ids[:, None, :].expand(-1, branches, -1),
            2,
            prefix_indices.clamp_min(0),
        )
        prefix_valid = torch.gather(
            clean_valid[:, None, :].expand(-1, branches, -1),
            2,
            prefix_indices.clamp_min(0),
        )
        origin_documents = torch.gather(document_ids, 1, starts)[:, :, None]
        prefix_valid &= exists & prefix_documents.eq(origin_documents)
        prefix = torch.where(prefix_valid, prefix, self.config.vocab.pad_id)
        visible = torch.cat((prefix, noisy_ids), dim=2).flatten(0, 1)
        visible_valid = torch.cat((prefix_valid, branch_valid), dim=2).flatten(0, 1)
        visible_documents = torch.cat(
            (
                prefix_documents,
                origin_documents.expand(-1, -1, canvas),
            ),
            dim=2,
        ).flatten(0, 1)
        branch = self._isolated_ngram_features(
            visible, visible_valid, visible_documents
        )[:, -canvas:].reshape(
            batch, branches * canvas, self.config.local_dim
        )
        clean = self._isolated_ngram_features(
            clean_ids, clean_valid, document_ids
        )
        return torch.cat((clean, branch), dim=1)

    def _aligned_clean_condition(
        self,
        patches: Tensor,
        document_ids: Tensor,
        positions: Tensor,
        patch_document_ids: Tensor,
    ) -> Tensor:
        """Most recent closed patch, reset exactly at every document boundary."""

        batch, length = document_ids.shape
        stride = self.config.patch_stride
        physical_patch = torch.arange(length, device=patches.device) // stride
        chosen = physical_patch[None].expand(batch, -1) - 1
        chosen = torch.where(
            positions.remainder(stride).eq(stride - 1),
            physical_patch[None],
            chosen,
        )
        safe = chosen.clamp_min(0)
        condition = torch.gather(
            patches,
            1,
            safe[:, :, None].expand(-1, -1, patches.shape[-1]),
        )
        chosen_document = torch.gather(patch_document_ids, 1, safe)
        available = chosen.ge(0) & chosen_document.eq(document_ids)
        return torch.where(available[:, :, None], condition, 0)

    def _forward_branches(
        self,
        clean_ids: Tensor,
        clean_valid: Tensor,
        document_ids: Tensor,
        positions: Tensor,
        noisy_ids: Tensor,
        branch_valid: Tensor,
        branch_starts: Tensor,
        *,
        local_self_condition: Tensor | None = None,
        decoder_self_condition: Tensor | None = None,
        attention_metadata: DiffusionGemmaAttentionMetadata | None = None,
        local_block_mask_metadata: CanvasBlockMaskMetadata | None = None,
        global_block_mask_metadata: CanvasBlockMaskMetadata | None = None,
        local_block_mask=None,
        global_block_mask=None,
        return_clean_logits: bool = True,
        return_clean_states: bool = False,
        allow_synthetic_branch_suffix: bool = False,
        encoder_modulations: tuple[Tensor, ...] | None = None,
        global_modulations: tuple[Tensor, ...] | None = None,
        decoder_modulations: tuple[Tensor, ...] | None = None,
        final_modulation: Tensor | None = None,
    ) -> DiffusionGemmaOutput:
        if clean_ids.ndim != 2 or clean_ids.shape != clean_valid.shape:
            raise ValueError("clean ids and valid must be aligned rank-2 tensors")
        if document_ids.shape != clean_ids.shape or positions.shape != clean_ids.shape:
            raise ValueError("document ids and positions must align with clean ids")
        if noisy_ids.ndim != 3 or noisy_ids.shape != branch_valid.shape:
            raise ValueError("noisy ids and validity must be aligned [B,M,C]")
        batch, branches, canvas = noisy_ids.shape
        if branch_starts.shape != (batch, branches):
            raise ValueError("branch starts must be [B,M]")
        if canvas % self.config.patch_stride:
            raise ValueError("canvas length must be patch aligned")
        if clean_ids.shape[1] % self.config.patch_stride:
            raise ValueError("clean length must be patch aligned")
        if not torch.compiler.is_compiling():
            if bool((branch_starts % self.config.patch_stride).any()):
                raise ValueError("branch starts must be physical-patch aligned")
            if bool(((branch_starts < 0) | (branch_starts >= clean_ids.shape[1])).any()):
                raise ValueError("canvas origins must lie inside the clean row")
            if bool(noisy_ids.masked_select(~branch_valid).ne(self.config.vocab.pad_id).any()):
                raise ValueError("inactive branch positions must contain PAD")
        modulation_groups = (
            encoder_modulations,
            global_modulations,
            decoder_modulations,
            final_modulation,
        )
        uses_adaln = any(value is not None for value in modulation_groups)
        if uses_adaln:
            if any(value is None for value in modulation_groups):
                raise ValueError("branch AdaLN requires every block and final modulation")
            assert encoder_modulations is not None
            assert global_modulations is not None
            assert decoder_modulations is not None
            assert final_modulation is not None
            if (
                len(encoder_modulations) != len(self.encoder)
                or len(global_modulations) != len(self.global_blocks)
                or len(decoder_modulations) != len(self.decoder)
            ):
                raise ValueError("branch AdaLN module counts must match transformer depth")
            if any(
                value.shape[:2] != (batch, branches)
                for value in (
                    *encoder_modulations,
                    *global_modulations,
                    *decoder_modulations,
                    final_modulation,
                )
            ):
                raise ValueError("branch AdaLN values must align [B,M]")

        if attention_metadata is None:
            attention_metadata = _packed_document_metadata(
                clean_valid,
                document_ids,
                patch_stride=self.config.patch_stride,
            )
        if attention_metadata.byte_indices.device != clean_ids.device:
            raise ValueError("attention metadata and model inputs must share a device")
        if not torch.compiler.is_compiling():
            if int(attention_metadata.byte_cu_seqlens[-1]) != int(
                attention_metadata.byte_indices.numel()
            ):
                raise ValueError("byte cumulative lengths do not cover packed bytes")
            if int(attention_metadata.patch_cu_seqlens[-1]) != int(
                attention_metadata.patch_indices.numel()
            ):
                raise ValueError("patch cumulative lengths do not cover packed patches")

        length = clean_ids.shape[1]
        offsets = torch.arange(canvas, device=clean_ids.device)
        indices = branch_starts[:, :, None] + offsets
        exists = indices.lt(length)
        safe_indices = indices.clamp_max(length - 1)
        source_document = torch.gather(
            document_ids[:, None, :].expand(-1, branches, -1), 2, safe_indices
        )
        branch_segments = torch.gather(document_ids, 1, branch_starts)
        origin_positions = torch.gather(positions, 1, branch_starts)
        source_positions = origin_positions[:, :, None] + offsets
        expected_valid = torch.gather(
            clean_valid[:, None, :].expand(-1, branches, -1), 2, safe_indices
        ) & exists & source_document.eq(branch_segments[:, :, None])
        source_document = branch_segments[:, :, None].expand_as(indices)
        if not torch.compiler.is_compiling():
            if allow_synthetic_branch_suffix:
                branch_offsets = torch.arange(canvas, device=clean_ids.device)
                contiguous = branch_offsets < branch_valid.sum(-1, keepdim=True)
                valid_documents = torch.where(
                    clean_valid, document_ids, branch_segments[:, :1]
                )
                one_document = valid_documents.eq(branch_segments[:, :1]).all(1)
                if (
                    not torch.equal(branch_valid, contiguous)
                    or bool((expected_valid & ~branch_valid).any())
                    or not bool(one_document.all())
                ):
                    raise ValueError(
                        "synthetic inference suffix must follow one contiguous clean document"
                    )
            elif not torch.equal(branch_valid, expected_valid):
                raise ValueError(
                    "branch validity crosses a document or disagrees with clean data"
                )

        physical_ids = torch.cat((clean_ids, noisy_ids.flatten(1, 2)), dim=1)
        physical_valid = torch.cat((clean_valid, branch_valid.flatten(1, 2)), dim=1)
        physical_positions = torch.cat((positions, source_positions.flatten(1, 2)), dim=1)
        local = self.embedding(physical_ids)
        ngrams = self._ngram_features(
            clean_ids,
            clean_valid,
            document_ids,
            noisy_ids,
            branch_valid,
            branch_starts,
        )
        if ngrams is not None:
            local = local + ngrams
            if self.config.ngram_aggregation == "mean":
                local = local / (len(self.config.ngram_orders) + 1)
        if local_self_condition is not None:
            if local_self_condition.shape != noisy_ids.shape + (self.config.local_dim,):
                raise ValueError("local self-conditioning must be [B,M,C,D]")
            local = local + F.pad(local_self_condition.flatten(1, 2), (0, 0, length, 0))

        local_layout = CanvasBranchLayout(
            clean_valid,
            branch_valid,
            branch_starts,
            self.config.local_window,
            positions,
            source_positions,
            document_ids,
            branch_segments,
        )
        if local_block_mask is None and clean_ids.device.type != "cpu":
            local_block_mask = build_canvas_block_mask(
                local_layout, metadata=local_block_mask_metadata
            )
        for index, block in enumerate(self.encoder):
            kwargs = dict(
                clean_length=length,
                clean_indices=attention_metadata.byte_indices,
                clean_cu_seqlens=attention_metadata.byte_cu_seqlens,
                assume_physical_clean=attention_metadata.physical_layout,
                positions=physical_positions,
                layout=local_layout,
                clean_window=self.config.local_window,
                allow_dense_reference=clean_ids.device.type == "cpu",
                block_mask=local_block_mask,
            )
            if encoder_modulations is None:
                local = block.forward_shared_document_branches(local, **kwargs)
            else:
                local = block.forward_shared_document_branches_adaln(
                    local, encoder_modulations[index], **kwargs
                )
        patches = self.pool(local, physical_valid)

        clean_patch_valid, clean_patch_documents, clean_patch_positions = _patch_metadata(
            clean_valid, document_ids, positions, self.config.patch_stride
        )
        canvas_patches = canvas // self.config.patch_stride
        branch_patch_valid = branch_valid.view(
            batch, branches, canvas_patches, self.config.patch_stride
        ).any(-1)
        patch_starts = branch_starts // self.config.patch_stride
        branch_patch_positions = source_positions[:, :, ::self.config.patch_stride] // self.config.patch_stride
        global_layout = CanvasBranchLayout(
            clean_patch_valid,
            branch_patch_valid,
            patch_starts,
            self.config.global_window,
            clean_patch_positions,
            branch_patch_positions,
            clean_patch_documents,
            branch_segments,
        )
        if global_block_mask is None and clean_ids.device.type != "cpu":
            global_block_mask = build_canvas_block_mask(
                global_layout, metadata=global_block_mask_metadata
            )
        global_positions = torch.cat(
            (clean_patch_positions, branch_patch_positions.flatten(1, 2)), dim=1
        )
        for index, block in enumerate(self.global_blocks):
            kwargs = dict(
                clean_length=clean_patch_valid.shape[1],
                clean_indices=attention_metadata.patch_indices,
                clean_cu_seqlens=attention_metadata.patch_cu_seqlens,
                assume_physical_clean=attention_metadata.physical_layout,
                positions=global_positions,
                layout=global_layout,
                clean_window=self.config.global_window,
                allow_dense_reference=clean_ids.device.type == "cpu",
                block_mask=global_block_mask,
            )
            if global_modulations is None:
                patches = block.forward_shared_document_branches(patches, **kwargs)
            else:
                patches = block.forward_shared_document_branches_adaln(
                    patches, global_modulations[index], **kwargs
                )
        patches = self.global_norm(patches)

        clean_patches = length // self.config.patch_stride
        clean_patch_states = patches[:, :clean_patches]
        branch_patch_states = patches[:, clean_patches:].view(
            batch, branches, canvas_patches, self.config.global_dim
        )
        condition = torch.cat(
            (
                self._aligned_clean_condition(
                    clean_patch_states,
                    document_ids,
                    positions,
                    clean_patch_documents,
                ),
                branch_patch_states.repeat_interleave(
                    self.config.patch_stride, dim=2
                ).flatten(1, 2),
            ),
            dim=1,
        )
        decoder_states = self.embedding(physical_ids)
        if decoder_self_condition is not None:
            if decoder_self_condition.shape != noisy_ids.shape + (self.config.local_dim,):
                raise ValueError("decoder self-conditioning must be [B,M,C,D]")
            decoder_states = decoder_states + F.pad(
                decoder_self_condition.flatten(1, 2), (0, 0, length, 0)
            )
        if self.config.decoder_prefix_window == self.config.local_window:
            # The byte-resolution topology is identical. Reusing the exact
            # BlockMask avoids a second data-dependent Flex setup per pass.
            decoder_layout = local_layout
            decoder_block_mask = local_block_mask
        else:
            decoder_layout = CanvasBranchLayout(
                clean_valid,
                branch_valid,
                branch_starts,
                self.config.decoder_prefix_window,
                positions,
                source_positions,
                document_ids,
                branch_segments,
            )
            decoder_block_mask = (
                None
                if clean_ids.device.type == "cpu"
                else build_canvas_block_mask(decoder_layout)
            )
        for index, block in enumerate(self.decoder):
            kwargs = dict(
                positions=physical_positions,
                clean_length=length,
                clean_indices=attention_metadata.byte_indices,
                clean_cu_seqlens=attention_metadata.byte_cu_seqlens,
                assume_physical_clean=attention_metadata.physical_layout,
                layout=decoder_layout,
                clean_window=self.config.decoder_prefix_window,
                allow_dense_reference=clean_ids.device.type == "cpu",
                block_mask=decoder_block_mask,
            )
            if decoder_modulations is None:
                decoder_states = block.forward_shared_document_branches(
                    decoder_states, condition, **kwargs
                )
            else:
                decoder_states = block.forward_shared_document_branches_adaln(
                    decoder_states,
                    condition,
                    decoder_modulations[index],
                    **kwargs,
                )

        branch_states = decoder_states[:, length:]
        if final_modulation is not None:
            final_shift, final_scale = final_modulation.chunk(2, dim=-1)
            branch_states = branch_states.view(
                batch, branches, canvas, self.config.local_dim
            )
            branch_states = self.decoder_norm(branch_states)
            branch_states = branch_states * (1 + final_scale[:, :, None]) + final_shift[:, :, None]
            branch_states = branch_states.flatten(1, 2)
            branch_logits = (
                F.linear(
                    branch_states,
                    self.embedding.weight[: self.config.vocab.output_size],
                )
                if self.output is None
                else self.output(branch_states)
            )
        else:
            branch_logits = self._logits(branch_states)
        if return_clean_logits:
            clean_logits = self._logits(decoder_states[:, :length])
        else:
            # Pure branch objectives must not materialize [B,L,V] logits for
            # the 8,192-byte clean bank.  This is a major memory-bandwidth and
            # VRAM cost at byte vocabulary width, not a harmless diagnostic.
            clean_logits = decoder_states.new_empty(
                (batch, 0, self.config.vocab.output_size)
            )
        return DiffusionGemmaOutput(
            clean_logits=clean_logits,
            branch_logits=branch_logits.view(
                batch, branches, canvas, self.config.vocab.output_size
            ),
            clean_patch_states=clean_patch_states,
            branch_patch_states=branch_patch_states,
            clean_decoder_states=(
                self.decoder_norm(decoder_states[:, :length])
                if return_clean_states
                else None
            ),
        )

    def forward(
        self,
        clean_ids: Tensor,
        clean_valid: Tensor,
        document_ids: Tensor,
        positions: Tensor,
        noisy_ids: Tensor,
        branch_valid: Tensor,
        branch_starts: Tensor,
        *,
        self_condition: bool = True,
        prior_probabilities: Tensor | None = None,
        self_conditioned_rows: Tensor | None = None,
        generator: torch.Generator | None = None,
        attention_metadata: DiffusionGemmaAttentionMetadata | None = None,
        allow_synthetic_branch_suffix: bool = False,
    ) -> DiffusionGemmaOutput:
        """Run the final dense objective pass, optionally with compact 50% prior."""

        batch = clean_ids.shape[0]
        if self_conditioned_rows is not None:
            if not self_condition or prior_probabilities is not None:
                raise ValueError(
                    "an explicit self-conditioning mask requires the auxiliary prior path"
                )
            if (
                self_conditioned_rows.shape != (batch,)
                or self_conditioned_rows.dtype != torch.bool
                or self_conditioned_rows.device != clean_ids.device
            ):
                raise ValueError(
                    "self-conditioned rows must be one aligned boolean per row"
                )
        if prior_probabilities is not None:
            if self_condition:
                raise ValueError(
                    "external prior probabilities and auxiliary prior pass are exclusive"
                )
            if prior_probabilities.shape != noisy_ids.shape + (
                self.config.vocab.output_size,
            ):
                raise ValueError("external prior probabilities do not align with canvas")
            projected = self.project_self_conditioning(prior_probabilities)
            output = self._forward_branches(
                clean_ids,
                clean_valid,
                document_ids,
                positions,
                noisy_ids,
                branch_valid,
                branch_starts,
                local_self_condition=projected,
                decoder_self_condition=projected,
                attention_metadata=attention_metadata,
                allow_synthetic_branch_suffix=allow_synthetic_branch_suffix,
            )
            selected = torch.ones(batch, dtype=torch.bool, device=clean_ids.device)
            return DiffusionGemmaOutput(
                **{
                    **output.__dict__,
                    "self_conditioned_rows": selected,
                    "prior_batch_size": 0,
                }
            )
        if not self_condition:
            return self._forward_branches(
                clean_ids,
                clean_valid,
                document_ids,
                positions,
                noisy_ids,
                branch_valid,
                branch_starts,
                attention_metadata=attention_metadata,
                allow_synthetic_branch_suffix=allow_synthetic_branch_suffix,
            )
        selected = (
            sample_static_half_batch_mask(
                batch, device=clean_ids.device, generator=generator
            )
            if self_conditioned_rows is None
            else self_conditioned_rows
        )
        selected_indices = selected.nonzero(as_tuple=False).flatten()
        if selected_indices.numel() == 0:
            output = self._forward_branches(
                clean_ids,
                clean_valid,
                document_ids,
                positions,
                noisy_ids,
                branch_valid,
                branch_starts,
                attention_metadata=attention_metadata,
                allow_synthetic_branch_suffix=allow_synthetic_branch_suffix,
            )
            return DiffusionGemmaOutput(
                **{
                    **output.__dict__,
                    "self_conditioned_rows": selected,
                    "prior_batch_size": 0,
                }
            )
        with torch.no_grad():
            prior = self._forward_branches(
                clean_ids.index_select(0, selected_indices),
                clean_valid.index_select(0, selected_indices),
                document_ids.index_select(0, selected_indices),
                positions.index_select(0, selected_indices),
                noisy_ids.index_select(0, selected_indices),
                branch_valid.index_select(0, selected_indices),
                branch_starts.index_select(0, selected_indices),
                attention_metadata=_packed_document_metadata(
                    clean_valid.index_select(0, selected_indices),
                    document_ids.index_select(0, selected_indices),
                    patch_stride=self.config.patch_stride,
                ),
                allow_synthetic_branch_suffix=allow_synthetic_branch_suffix,
            ).branch_logits.softmax(-1)
        projected_selected = self.project_self_conditioning(prior)
        shape = noisy_ids.shape + (self.config.local_dim,)
        projected = projected_selected.new_zeros(shape).index_copy(
            0, selected_indices, projected_selected
        )
        output = self._forward_branches(
            clean_ids,
            clean_valid,
            document_ids,
            positions,
            noisy_ids,
            branch_valid,
            branch_starts,
            local_self_condition=projected,
            decoder_self_condition=projected,
            attention_metadata=attention_metadata,
            allow_synthetic_branch_suffix=allow_synthetic_branch_suffix,
        )
        return DiffusionGemmaOutput(
            **{
                **output.__dict__,
                "self_conditioned_rows": selected,
                "prior_batch_size": selected_indices.numel(),
            }
        )

    def forward_bos_logits(self, count: int, *, device: torch.device) -> Tensor:
        """Atomic first-byte prediction from DiffusionGemma's virtual EOT/BOS."""

        if count < 0:
            raise ValueError("BOS count cannot be negative")
        if count == 0:
            return self.embedding.weight.new_empty((0, self.config.vocab.output_size))
        # The virtual EOT is the first byte of an incomplete patch.  Under the
        # clean shifted geometry it therefore has no closed global patch to
        # condition on; encoding a singleton patch here leaked information
        # unavailable to the ordinary clean path.
        condition = self.embedding.weight.new_zeros((1, self.config.global_dim))
        states = self.embedding(
            torch.full((1, 1), self.config.vocab.eot_id, dtype=torch.long, device=device)
        )
        decoder_positions = torch.zeros((1, 1), dtype=torch.long, device=device)
        singleton = torch.ones((1, 1, 1), dtype=torch.bool, device=device)
        for block in self.decoder:
            states = block(
                states,
                condition[:, None],
                positions=decoder_positions,
                allowed=singleton,
            )
        return self._logits(states)[:, 0].expand(count, -1)


__all__ = (
    "compile_stable_document_metadata",
    "DIFFUSION_GEMMA_PARAMETER_COUNT",
    "DiffusionGemmaAttentionMetadata",
    "DiffusionGemmaModel",
    "DiffusionGemmaOutput",
    "packed_patch_cu_seqlens",
)
