"""Time-conditioned BLT backbone for the standalone Byte-Duo recipe.

The topology deliberately reuses the repository's optimized shared-bank BLT
attention implementation: clean pages are strictly causal and document
isolated, while each 512-byte noisy canvas is bidirectional and can see only
its own clean prefix. Duo adds mandatory scale-shift-gate time conditioning to
noisy byte states. The time-independent clean bank and its rotated K/V are
cached once per committed canvas and reused across every reverse NFE.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import (
    CanvasBlockMaskMetadata,
    CanvasBranchLayout,
    build_canvas_block_mask,
)
from .config import ByteDiffusionConfig
from .diffusion_gemma_model import (
    DiffusionGemmaAttentionMetadata,
    DiffusionGemmaModel,
    DiffusionGemmaOutput,
    _packed_document_metadata,
    _patch_metadata,
)
from .layers import CleanAttentionBank, pack_rows, unpack_rows
from .variable_patching import DuoCleanPatchMetadata


DUO_TIME_FEATURES = 64
DUO_TIME_CONDITION_DIM = 32
DUO_PARAMETER_COUNT = 24_094_791


@dataclass(frozen=True)
class DuoCleanBank:
    """Time-independent clean pathway and per-layer rotated attention K/V."""

    clean_ids: Tensor
    clean_valid: Tensor
    document_ids: Tensor
    positions: Tensor
    attention_metadata: DiffusionGemmaAttentionMetadata | DuoCleanPatchMetadata
    encoder: tuple[CleanAttentionBank, ...]
    global_blocks: tuple[CleanAttentionBank, ...]
    decoder: tuple[CleanAttentionBank, ...]
    clean_patch_states: Tensor
    decoder_condition: Tensor | None
    clean_decoder_states: Tensor

    @property
    def batch_size(self) -> int:
        return self.clean_ids.shape[0]

    @property
    def length(self) -> int:
        return self.clean_ids.shape[1]


@dataclass(frozen=True)
class DuoCanvasCache:
    """Clean bank plus canvas-static layouts reused across every Duo NFE."""

    clean: DuoCleanBank
    branch_valid: Tensor
    branch_starts: Tensor
    source_positions: Tensor
    branch_segments: Tensor
    local_layout: CanvasBranchLayout | None
    local_block_mask: object | None
    global_layout: CanvasBranchLayout | None
    global_block_mask: object | None
    decoder_layout: CanvasBranchLayout
    decoder_block_mask: object | None
    decoder_projected_branch_conditions: tuple[Tensor, ...] | None

    @property
    def canvas_length(self) -> int:
        return self.branch_valid.shape[-1]


class CleanAtomEmbedding(nn.Module):
    """Embedding with 261 clean rows and one private PAD storage row.

    External PAD remains id 262 for dataset compatibility, but is remapped to
    private row 261.  External id 261 (MASK in other recipes) is rejected and
    never acquires a learned representation in this model.
    """

    def __init__(self, clean_atoms: int, dim: int, *, external_pad_id: int) -> None:
        super().__init__()
        if external_pad_id <= clean_atoms:
            raise ValueError("external PAD must follow the forbidden MASK slot")
        self.clean_atoms = clean_atoms
        self.external_pad_id = external_pad_id
        self.weight = nn.Parameter(torch.empty(clean_atoms + 1, dim))

    @property
    def padding_idx(self) -> int:
        return self.clean_atoms

    def forward(self, ids: Tensor) -> Tensor:
        if ids.dtype != torch.long:
            raise TypeError("atomic ids must be int64")
        if torch.compiler.is_compiling():
            torch._assert_async(
                ~ids.eq(self.clean_atoms).any(),
                "Byte-Duo forbids the absorbing MASK id",
            )
            torch._assert_async(
                (ids.lt(self.clean_atoms) | ids.eq(self.external_pad_id)).all(),
                "Byte-Duo received an unknown atomic id",
            )
        else:
            if bool(ids.eq(self.clean_atoms).any()):
                raise ValueError("Byte-Duo forbids the absorbing MASK id")
            valid = ids.lt(self.clean_atoms) | ids.eq(self.external_pad_id)
            if bool(~valid.all()):
                raise ValueError("Byte-Duo received an unknown atomic id")
        internal = torch.where(ids.eq(self.external_pad_id), self.clean_atoms, ids)
        return F.embedding(internal, self.weight, padding_idx=self.clean_atoms)


def sinusoidal_time_features(t: Tensor, width: int = DUO_TIME_FEATURES) -> Tensor:
    if t.ndim not in {1, 2} or not t.is_floating_point():
        raise ValueError("Duo time must be floating point [B] or [B,M]")
    if width <= 0 or width % 2:
        raise ValueError("time feature width must be a positive even integer")
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(width // 2, device=t.device, dtype=torch.float32)
        / (width // 2)
    )
    angles = t.float()[..., None] * frequencies
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


class DuoModel(DiffusionGemmaModel):
    """Byte-Duo's non-absorbing, AdaLN-Zero time-conditioned BLT cell."""

    def __init__(
        self,
        config: ByteDiffusionConfig = ByteDiffusionConfig(),
        *,
        schedule_eps: float = 1e-3,
    ) -> None:
        if config.explicit_timestep or config.self_conditioning:
            raise ValueError("Duo owns mandatory time conditioning and no self-conditioning")
        if not math.isfinite(schedule_eps) or not 0.0 < schedule_eps < 0.5:
            raise ValueError("Duo model schedule eps must lie in (0, 0.5)")
        super().__init__(config)
        self.schedule_eps = float(schedule_eps)
        self.embedding = CleanAtomEmbedding(
            config.vocab.output_size,
            config.local_dim,
            external_pad_id=config.vocab.pad_id,
        )
        self.output = nn.Linear(
            config.local_dim, config.vocab.output_size, bias=True
        )
        # Scaling-DLLMs maps sigma=-log(alpha) through a time MLP, then gives
        # every transformer block an independent zero-initialized 6D AdaLN
        # projection (attention/FFN shift, scale, and residual gate). The final
        # branch norm has its own 2D projection. Clean prefix states remain
        # time-independent so they can eventually be cached across NFEs.
        self.time_embedding = nn.Sequential(
            nn.Linear(
                config.duo_time_features,
                config.duo_time_condition_dim,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                config.duo_time_condition_dim,
                config.duo_time_condition_dim,
                bias=True,
            ),
        )
        # All block modulations consume one shared [B,M,C] condition.
        # One packed projection replaces thirteen launch-bound GEMMs while
        # retaining independent output parameters for every block.
        if config.duo_mutable_topology == "full_resolution_decoder":
            # The clean encoder/global hierarchy is time independent and the
            # mutable stream bypasses both stacks.  Do not reserve modulation
            # rows that can never participate in a forward or receive a
            # gradient; those parameters are material under a 16 MB budget.
            self._adaln_widths = (
                *((6 * config.local_dim,) * config.decoder_layers),
                2 * config.local_dim,
            )
        else:
            self._adaln_widths = (
                *((6 * config.local_dim,) * config.encoder_layers),
                *((6 * config.global_dim,) * config.global_layers),
                *((6 * config.local_dim,) * config.decoder_layers),
                2 * config.local_dim,
            )
        self.time_adaln = nn.Linear(
            config.duo_time_condition_dim, sum(self._adaln_widths), bias=True
        )
        del self.self_conditioner
        self.reset_parameters()
        nn.init.zeros_(self.time_adaln.weight)
        nn.init.zeros_(self.time_adaln.bias)
        if self.output is None:
            raise ValueError("Byte-Duo requires an independent zero-init output head")
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding, CleanAtomEmbedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        embedding = getattr(self, "embedding", None)
        if isinstance(embedding, CleanAtomEmbedding):
            with torch.no_grad():
                embedding.weight[embedding.padding_idx].zero_()
        elif isinstance(embedding, nn.Embedding) and embedding.padding_idx is not None:
            with torch.no_grad():
                embedding.weight[embedding.padding_idx].zero_()
        ngrams = getattr(self, "ngrams", None)
        if ngrams is not None:
            nn.init.normal_(ngrams.projection_weight, mean=0.0, std=0.02)
            if self.config.ngram_factor_init == "scale_matched":
                ngrams.projection_weight.data.mul_(
                    self.config.ngram_rank**-0.5 / 0.02
                )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def estimated_quantized_artifact_bytes(self) -> int:
        """Conservative model-only estimate from the measured family export.

        The sibling 23,043,330-parameter BLT cell exports to 12,721,373 bytes
        with the actual group quantizer.  Scale that measured density by Duo's
        parameter count and add one percent for its different matrix shapes.
        The eventual exporter audit remains authoritative.
        """

        reference_parameters = 23_043_330
        reference_bytes = 12_721_373
        return math.ceil(
            self.parameter_count
            * reference_bytes
            / reference_parameters
            * 1.01
        )

    def validate_production_parameterization(self) -> None:
        if self.config != ByteDiffusionConfig():
            raise ValueError("the production budget applies only to the default Duo cell")
        if self.schedule_eps != 1e-3:
            raise ValueError("the production Duo cell uses reference schedule eps=1e-3")
        if self.parameter_count != DUO_PARAMETER_COUNT:
            raise AssertionError(
                "Byte-Duo parameterization drifted: "
                f"expected {DUO_PARAMETER_COUNT:,}, observed {self.parameter_count:,}"
            )
        if self.estimated_quantized_artifact_bytes >= 16_000_000:
            raise AssertionError("Byte-Duo's estimated artifact exceeds 16 MB")
        if not isinstance(self.embedding, CleanAtomEmbedding):
            raise AssertionError("Byte-Duo embedding unexpectedly admits MASK")

    def _time_condition(self, noisy_ids: Tensor, t: Tensor) -> Tensor:
        batch, branches, canvas = noisy_ids.shape
        if t.shape == (batch,):
            row_time = t[:, None].expand(batch, branches)
        elif t.shape == (batch, branches):
            row_time = t
        else:
            raise ValueError("Duo time must be [B] or [B,M]")
        # Scaling-DLLMs names this argument ``sigma``: the DiT receives
        # -log(alpha(t)), a monotone continuous-time coordinate.
        alpha = 1.0 - (1.0 - self.schedule_eps) * row_time
        sigma = -alpha.log()
        # Trigonometric features are evaluated in FP32 even when the model is
        # explicitly stored in BF16, then cast only at the MLP boundary.  This
        # avoids quantizing high-frequency phases before sin/cos while keeping
        # the conditioner usable outside autocast as well.
        features = sinusoidal_time_features(
            sigma, self.config.duo_time_features
        ).to(self.time_embedding[0].weight.dtype)
        return F.silu(self.time_embedding(features))

    def _adaln_modulations(
        self, time_condition: Tensor
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...], Tensor]:
        values = self.time_adaln(time_condition).split(self._adaln_widths, dim=-1)
        if self.config.duo_mutable_topology == "full_resolution_decoder":
            return (), (), values[:-1], values[-1]
        encoder_stop = self.config.encoder_layers
        global_stop = encoder_stop + self.config.global_layers
        decoder_stop = global_stop + self.config.decoder_layers
        return (
            values[:encoder_stop],
            values[encoder_stop:global_stop],
            values[global_stop:decoder_stop],
            values[decoder_stop],
        )

    def _branch_ngram_features(
        self,
        clean: DuoCleanBank,
        noisy_ids: Tensor,
        branch_valid: Tensor,
        branch_starts: Tensor,
    ) -> Tensor | None:
        """Evaluate only canvas n-grams; clean n-grams live in the cache."""

        if self.ngrams is None or not self.config.duo_noisy_ngrams:
            return None
        batch, branches, canvas = noisy_ids.shape
        halo = max(self.config.ngram_orders) - 1
        offsets = torch.arange(halo, device=noisy_ids.device)
        prefix_indices = branch_starts[:, :, None] - halo + offsets
        exists = prefix_indices.ge(0)
        safe = prefix_indices.clamp_min(0)
        prefix = torch.gather(
            clean.clean_ids[:, None, :].expand(-1, branches, -1), 2, safe
        )
        prefix_documents = torch.gather(
            clean.document_ids[:, None, :].expand(-1, branches, -1), 2, safe
        )
        prefix_valid = torch.gather(
            clean.clean_valid[:, None, :].expand(-1, branches, -1), 2, safe
        )
        origin_documents = torch.gather(
            clean.document_ids, 1, branch_starts
        )[:, :, None]
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
        return self._isolated_ngram_features(
            visible, visible_valid, visible_documents
        )[:, -canvas:].reshape(batch, branches * canvas, self.config.local_dim)

    def prepare_clean_bank(
        self,
        clean_ids: Tensor,
        clean_valid: Tensor,
        document_ids: Tensor,
        positions: Tensor,
        *,
        attention_metadata: DiffusionGemmaAttentionMetadata | None = None,
        clean_patch_metadata: DuoCleanPatchMetadata | None = None,
    ) -> DuoCleanBank:
        """Run the time-independent clean encoder/global/decoder exactly once."""

        if torch.is_grad_enabled():
            raise RuntimeError("Duo clean-bank caching is inference-only")
        if clean_ids.ndim != 2 or clean_ids.shape != clean_valid.shape:
            raise ValueError("clean ids and valid must be aligned rank-2 tensors")
        if document_ids.shape != clean_ids.shape or positions.shape != clean_ids.shape:
            raise ValueError("document ids and positions must align with clean ids")
        expects_entropy = self.config.duo_clean_patching == "causal_entropy_v1"
        if expects_entropy != (clean_patch_metadata is not None):
            raise ValueError(
                "Duo model config and supplied clean patch metadata disagree"
            )
        if clean_patch_metadata is None and clean_ids.shape[1] % self.config.patch_stride:
            raise ValueError("clean length must be patch aligned")
        if attention_metadata is not None and clean_patch_metadata is not None:
            raise ValueError("Duo clean patch policies are mutually exclusive")
        if attention_metadata is None and clean_patch_metadata is None:
            attention_metadata = _packed_document_metadata(
                clean_valid,
                document_ids,
                patch_stride=self.config.patch_stride,
            )

        effective_metadata = (
            clean_patch_metadata
            if clean_patch_metadata is not None
            else attention_metadata
        )
        if effective_metadata is None:
            raise AssertionError("clean attention metadata disappeared")
        if (
            clean_patch_metadata is not None
            and self.config.duo_mutable_topology != "full_resolution_decoder"
        ):
            raise ValueError(
                "entropy clean patches require the full-resolution Duo decoder"
            )
        encoder_banks: list[CleanAttentionBank] = []
        global_banks: list[CleanAttentionBank] = []
        if self.config.duo_mutable_topology == "full_resolution_decoder":
            # Mutable bytes never query encoder/global K/V. Run the causal
            # clean hierarchy without constructing or retaining those banks;
            # only decoder clean K/V is reused by reverse NFEs.
            clean_patch_states, decoder_condition = self._clean_hierarchy(
                clean_ids,
                clean_valid,
                document_ids,
                positions,
                attention_metadata,
                clean_patch_metadata,
            )
        else:
            local = self.embedding(clean_ids)
            if self.ngrams is not None:
                local = local + self._isolated_ngram_features(
                    clean_ids, clean_valid, document_ids
                )
                if self.config.ngram_aggregation == "mean":
                    local = local / (len(self.config.ngram_orders) + 1)
            for block in self.encoder:
                local, bank = block.prepare_clean_bank(
                    local,
                    clean_indices=attention_metadata.byte_indices,
                    clean_cu_seqlens=attention_metadata.byte_cu_seqlens,
                    assume_physical_clean=attention_metadata.physical_layout,
                    positions=positions,
                    clean_window=self.config.local_window,
                    allow_dense_reference=clean_ids.device.type == "cpu",
                )
                encoder_banks.append(bank)

            patches = self.pool(local, clean_valid)
            _, clean_patch_documents, clean_patch_positions = _patch_metadata(
                clean_valid,
                document_ids,
                positions,
                self.config.patch_stride,
            )
            for block in self.global_blocks:
                patches, bank = block.prepare_clean_bank(
                    patches,
                    clean_indices=attention_metadata.patch_indices,
                    clean_cu_seqlens=attention_metadata.patch_cu_seqlens,
                    assume_physical_clean=attention_metadata.physical_layout,
                    positions=clean_patch_positions,
                    clean_window=self.config.global_window,
                    allow_dense_reference=clean_ids.device.type == "cpu",
                )
                global_banks.append(bank)
            clean_patch_states = self.global_norm(patches)
            decoder_condition = self._aligned_clean_condition(
                clean_patch_states,
                document_ids,
                positions,
                clean_patch_documents,
            )

        decoder_states = self.embedding(clean_ids)
        decoder_banks: list[CleanAttentionBank] = []
        assume_physical_clean = (
            clean_patch_metadata is not None
            or (
                attention_metadata.physical_layout
                if attention_metadata is not None
                else False
            )
        )
        for block in self.decoder:
            kwargs = {
                "clean_indices": effective_metadata.byte_indices,
                "clean_cu_seqlens": effective_metadata.byte_cu_seqlens,
                "assume_physical_clean": assume_physical_clean,
                "positions": positions,
                "clean_window": self.config.decoder_prefix_window,
                "allow_dense_reference": clean_ids.device.type == "cpu",
            }
            if (
                self.config.duo_mutable_topology == "full_resolution_decoder"
                and block.conditioning != "split_cross_attention"
            ):
                projected_patches = block.project_reusable_condition(
                    clean_patch_states
                )
                projected_condition = self._clean_byte_condition(
                    projected_patches,
                    clean_valid,
                    document_ids,
                    positions,
                    clean_patch_metadata,
                )
                decoder_states, bank = block.prepare_clean_bank_projected(
                    decoder_states, projected_condition, **kwargs
                )
            else:
                if decoder_condition is None:
                    raise AssertionError("raw decoder condition is unavailable")
                decoder_states, bank = block.prepare_clean_bank(
                    decoder_states, decoder_condition, **kwargs
                )
            decoder_banks.append(bank)
        return DuoCleanBank(
            clean_ids=clean_ids,
            clean_valid=clean_valid,
            document_ids=document_ids,
            positions=positions,
            attention_metadata=effective_metadata,
            encoder=tuple(encoder_banks),
            global_blocks=tuple(global_banks),
            decoder=tuple(decoder_banks),
            clean_patch_states=clean_patch_states,
            decoder_condition=decoder_condition,
            clean_decoder_states=decoder_states,
        )

    def _ngram_features(
        self,
        clean_ids: Tensor,
        clean_valid: Tensor,
        document_ids: Tensor,
        noisy_ids: Tensor,
        branch_valid: Tensor,
        starts: Tensor,
    ) -> Tensor | None:
        """Build full-forward n-grams with the same policy as cached serving."""

        if self.ngrams is None:
            return None
        clean = self._isolated_ngram_features(
            clean_ids, clean_valid, document_ids
        )
        if not self.config.duo_noisy_ngrams:
            batch, branches, canvas = noisy_ids.shape
            branch = clean.new_zeros(
                (batch, branches * canvas, self.config.local_dim)
            )
            return torch.cat((clean, branch), dim=1)
        return super()._ngram_features(
            clean_ids,
            clean_valid,
            document_ids,
            noisy_ids,
            branch_valid,
            starts,
        )

    def _clean_hierarchy(
        self,
        clean_ids: Tensor,
        clean_valid: Tensor,
        document_ids: Tensor,
        positions: Tensor,
        attention_metadata: DiffusionGemmaAttentionMetadata | None,
        clean_patch_metadata: DuoCleanPatchMetadata | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Run the selected immutable clean encoder/global hierarchy.

        Mutable states deliberately never enter this helper.  Clean bytes and
        patches use the existing document-packed causal kernels, so the full-
        resolution control changes only the revisable topology.
        """

        batch, length = clean_ids.shape
        local_padded = self.embedding(clean_ids)
        if self.ngrams is not None:
            local_padded = local_padded + self._isolated_ngram_features(
                clean_ids, clean_valid, document_ids
            )
            if self.config.ngram_aggregation == "mean":
                local_padded = local_padded / (len(self.config.ngram_orders) + 1)
        if (attention_metadata is None) == (clean_patch_metadata is None):
            raise ValueError("exactly one Duo clean patch policy is required")
        metadata = (
            clean_patch_metadata
            if clean_patch_metadata is not None
            else attention_metadata
        )
        if metadata is None:
            raise AssertionError("clean hierarchy metadata disappeared")
        physical = (
            clean_patch_metadata is not None
            or (
                attention_metadata.physical_layout
                if attention_metadata is not None
                else False
            )
        )
        local = (
            local_padded.reshape(-1, self.config.local_dim)
            if physical
            else pack_rows(local_padded, metadata.byte_indices)
        )
        packed_positions = (
            positions.reshape(-1)
            if physical
            else pack_rows(positions, metadata.byte_indices)
        )
        for block in self.encoder:
            local = block.forward_packed(
                local,
                cu_seqlens=metadata.byte_cu_seqlens,
                positions=packed_positions,
                max_seqlen=length,
                window=self.config.local_window,
                allow_dense_reference=clean_ids.device.type == "cpu",
            )
        local_padded = (
            local.view(batch, length, self.config.local_dim)
            if physical
            else unpack_rows(local, metadata.byte_indices, local_padded)
        )
        if clean_patch_metadata is not None:
            pool_local = local.index_select(
                0, clean_patch_metadata.pool_byte_indices
            )
            packed_patches = self.pool.forward_packed(
                pool_local,
                clean_patch_metadata.patch_byte_cu_seqlens,
                max_patch_size=clean_patch_metadata.max_patch_size,
            )
            packed_patch_positions = clean_patch_metadata.patch_ordinals
            patch_cu_seqlens = clean_patch_metadata.patch_cu_seqlens
            max_patch_seqlen = clean_patch_metadata.max_patch_seqlen
            patches = packed_patches
            patch_documents = None
        else:
            if attention_metadata is None:
                raise AssertionError("fixed clean metadata disappeared")
            patches = self.pool(local_padded, clean_valid)
            _, patch_documents, patch_positions = _patch_metadata(
                clean_valid, document_ids, positions, self.config.patch_stride
            )
            packed_patches = (
                patches.reshape(-1, self.config.global_dim)
                if physical
                else pack_rows(patches, attention_metadata.patch_indices)
            )
            packed_patch_positions = (
                patch_positions.reshape(-1)
                if physical
                else pack_rows(patch_positions, attention_metadata.patch_indices)
            )
            patch_cu_seqlens = attention_metadata.patch_cu_seqlens
            max_patch_seqlen = patches.shape[1]
        for block in self.global_blocks:
            packed_patches = block.forward_packed(
                packed_patches,
                cu_seqlens=patch_cu_seqlens,
                positions=packed_patch_positions,
                max_seqlen=max_patch_seqlen,
                window=self.config.global_window,
                allow_dense_reference=clean_ids.device.type == "cpu",
            )
        packed_patches = self.global_norm(packed_patches)
        if clean_patch_metadata is not None:
            clean_patch_states = packed_patches
        elif physical:
            clean_patch_states = packed_patches.view_as(patches)
        else:
            if attention_metadata is None:
                raise AssertionError("fixed clean metadata disappeared")
            clean_patch_states = unpack_rows(
                packed_patches, attention_metadata.patch_indices, patches
            )
        decoder_condition = (
            self._clean_byte_condition(
                clean_patch_states,
                clean_valid,
                document_ids,
                positions,
                clean_patch_metadata,
            )
            if any(
                block.conditioning == "split_cross_attention"
                for block in self.decoder
            )
            else None
        )
        return clean_patch_states, decoder_condition

    def _clean_byte_condition(
        self,
        patch_states: Tensor,
        clean_valid: Tensor,
        document_ids: Tensor,
        positions: Tensor,
        clean_patch_metadata: DuoCleanPatchMetadata | None,
    ) -> Tensor:
        """Align a latent only after its complete physical patch is visible."""

        if clean_patch_metadata is None:
            _, patch_documents, _ = _patch_metadata(
                clean_valid,
                document_ids,
                positions,
                self.config.patch_stride,
            )
            return self._aligned_clean_condition(
                patch_states, document_ids, positions, patch_documents
            )
        indices = clean_patch_metadata.byte_condition_indices
        safe = indices.clamp_min(0)
        selected = patch_states.index_select(0, safe)
        selected = torch.where(indices[:, None].ge(0), selected, 0)
        padded = selected.new_zeros(
            (*clean_valid.shape, selected.shape[-1])
        )
        return unpack_rows(
            selected,
            clean_patch_metadata.pool_byte_indices,
            padded,
        )

    def _preceding_patch_condition(
        self,
        clean_patch_states: Tensor,
        clean_valid: Tensor,
        document_ids: Tensor,
        positions: Tensor,
        branch_starts: Tensor,
        clean_patch_metadata: DuoCleanPatchMetadata | None = None,
    ) -> Tensor:
        """Select the last fully closed clean patch for each branch origin.

        An origin inside a physical patch cannot use that patch's latent: it
        contains bytes at or after the origin.  Thus every phase in physical
        patch ``p`` uses ``p-1``.  Starts in a document's first patch receive
        the exact zero virtual-BOS condition.
        """

        if clean_patch_metadata is not None:
            chosen = torch.gather(
                clean_patch_metadata.origin_condition_indices,
                1,
                branch_starts,
            )
            safe = chosen.clamp_min(0)
            gathered = clean_patch_states.index_select(
                0, safe.reshape(-1)
            ).view(*safe.shape, clean_patch_states.shape[-1])
            return torch.where(chosen[..., None].ge(0), gathered, 0)
        stride = self.config.patch_stride
        chosen = torch.div(branch_starts, stride, rounding_mode="floor") - 1
        safe = chosen.clamp_min(0)
        gathered = torch.gather(
            clean_patch_states,
            1,
            safe[..., None].expand(-1, -1, clean_patch_states.shape[-1]),
        )
        patch_valid = clean_valid.view(
            clean_valid.shape[0], -1, stride
        ).any(-1)
        patch_documents = document_ids.view(
            document_ids.shape[0], -1, stride
        )[:, :, 0]
        chosen_valid = torch.gather(patch_valid, 1, safe)
        chosen_documents = torch.gather(patch_documents, 1, safe)
        branch_documents = torch.gather(document_ids, 1, branch_starts)
        branch_positions = torch.gather(positions, 1, branch_starts)
        available = (
            chosen.ge(0)
            & chosen_valid
            & chosen_documents.eq(branch_documents)
            & branch_positions.ge(stride)
        )
        return torch.where(available[..., None], gathered, 0)

    def prepare_canvas_cache(
        self,
        clean: DuoCleanBank,
        branch_valid: Tensor,
        branch_starts: Tensor,
        *,
        allow_synthetic_branch_suffix: bool = False,
        validate_inputs: bool = True,
    ) -> DuoCanvasCache:
        """Prepare canvas-static layouts and masks once for all reverse NFEs."""

        if torch.is_grad_enabled():
            raise RuntimeError("Duo canvas caching is inference-only")
        if branch_valid.ndim != 3:
            raise ValueError("branch validity must be [B,M,C]")
        batch, branches, canvas = branch_valid.shape
        if batch != clean.batch_size or branch_starts.shape != (batch, branches):
            raise ValueError("branch starts must align with the cached clean batch")
        full_resolution = (
            self.config.duo_mutable_topology == "full_resolution_decoder"
        )
        if not full_resolution and canvas % self.config.patch_stride:
            raise ValueError("canvas length must be patch aligned")
        if validate_inputs and not torch.compiler.is_compiling():
            if not full_resolution and bool(
                (branch_starts % self.config.patch_stride).any()
            ):
                raise ValueError("branch starts must be physical-patch aligned")
            if bool(((branch_starts < 0) | (branch_starts >= clean.length)).any()):
                raise ValueError("canvas origins must lie inside the clean row")

        offsets = torch.arange(canvas, device=clean.clean_ids.device)
        indices = branch_starts[:, :, None] + offsets
        exists = indices.lt(clean.length)
        safe = indices.clamp_max(clean.length - 1)
        source_document = torch.gather(
            clean.document_ids[:, None, :].expand(-1, branches, -1), 2, safe
        )
        branch_segments = torch.gather(clean.document_ids, 1, branch_starts)
        source_positions = (
            torch.gather(clean.positions, 1, branch_starts)[:, :, None] + offsets
        )
        expected_valid = torch.gather(
            clean.clean_valid[:, None, :].expand(-1, branches, -1), 2, safe
        ) & exists & source_document.eq(branch_segments[:, :, None])
        if validate_inputs and not torch.compiler.is_compiling():
            contiguous = offsets < branch_valid.sum(-1, keepdim=True)
            if allow_synthetic_branch_suffix:
                valid_documents = torch.where(
                    clean.clean_valid,
                    clean.document_ids,
                    branch_segments[:, :1],
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
            elif not torch.equal(branch_valid, contiguous) or bool(
                (branch_valid & ~expected_valid).any()
            ):
                raise ValueError(
                    "branch validity must be a contiguous same-document source prefix"
                )

        if full_resolution:
            # Mutable bytes enter only the full-resolution decoder.  Do not
            # construct dead local/global branch layouts or their BlockMasks.
            local_layout = None
            local_mask = None
            global_layout = None
            global_mask = None
            decoder_layout = CanvasBranchLayout(
                clean.clean_valid,
                branch_valid,
                branch_starts,
                self.config.decoder_prefix_window,
                clean.positions,
                source_positions,
                clean.document_ids,
                branch_segments,
            )
            decoder_mask = (
                None
                if clean.clean_ids.device.type == "cpu"
                else build_canvas_block_mask(decoder_layout)
            )
            preceding = self._preceding_patch_condition(
                clean.clean_patch_states,
                clean.clean_valid,
                clean.document_ids,
                clean.positions,
                branch_starts,
                (
                    clean.attention_metadata
                    if isinstance(clean.attention_metadata, DuoCleanPatchMetadata)
                    else None
                ),
            )
            decoder_projected_branch_conditions = tuple(
                (
                    preceding
                    if block.conditioning == "split_cross_attention"
                    else block.project_reusable_condition(preceding)
                )
                for block in self.decoder
            )
        else:
            decoder_projected_branch_conditions = None
            local_layout = CanvasBranchLayout(
                clean.clean_valid,
                branch_valid,
                branch_starts,
                self.config.local_window,
                clean.positions,
                source_positions,
                clean.document_ids,
                branch_segments,
            )
            local_mask = (
                None
                if clean.clean_ids.device.type == "cpu"
                else build_canvas_block_mask(local_layout)
            )
            clean_patch_valid, clean_patch_documents, clean_patch_positions = (
                _patch_metadata(
                    clean.clean_valid,
                    clean.document_ids,
                    clean.positions,
                    self.config.patch_stride,
                )
            )
            canvas_patches = canvas // self.config.patch_stride
            branch_patch_valid = branch_valid.view(
                batch, branches, canvas_patches, self.config.patch_stride
            ).any(-1)
            branch_patch_positions = (
                source_positions[:, :, :: self.config.patch_stride]
                // self.config.patch_stride
            )
            global_layout = CanvasBranchLayout(
                clean_patch_valid,
                branch_patch_valid,
                branch_starts // self.config.patch_stride,
                self.config.global_window,
                clean_patch_positions,
                branch_patch_positions,
                clean_patch_documents,
                branch_segments,
            )
            global_mask = (
                None
                if clean.clean_ids.device.type == "cpu"
                else build_canvas_block_mask(global_layout)
            )
            if self.config.decoder_prefix_window == self.config.local_window:
                decoder_layout = local_layout
                decoder_mask = local_mask
            else:
                decoder_layout = CanvasBranchLayout(
                    clean.clean_valid,
                    branch_valid,
                    branch_starts,
                    self.config.decoder_prefix_window,
                    clean.positions,
                    source_positions,
                    clean.document_ids,
                    branch_segments,
                )
                decoder_mask = (
                    None
                    if clean.clean_ids.device.type == "cpu"
                    else build_canvas_block_mask(decoder_layout)
                )
        return DuoCanvasCache(
            clean,
            branch_valid,
            branch_starts,
            source_positions,
            branch_segments,
            local_layout,
            local_mask,
            global_layout,
            global_mask,
            decoder_layout,
            decoder_mask,
            decoder_projected_branch_conditions,
        )

    def forward_prepared(
        self,
        cache: DuoCanvasCache,
        noisy_ids: Tensor,
        t: Tensor,
        *,
        return_clean_logits: bool = False,
        return_clean_states: bool = False,
    ) -> DiffusionGemmaOutput:
        """Run only revisable branches against a prepared clean/canvas cache."""

        if torch.is_grad_enabled():
            raise RuntimeError("prepared Duo forward is inference-only")
        if noisy_ids.shape != cache.branch_valid.shape:
            raise ValueError("noisy ids must align with the prepared canvas")
        if noisy_ids.device.type == "cpu" and not torch.compiler.is_compiling():
            inactive = noisy_ids.masked_select(~cache.branch_valid)
            if bool(inactive.ne(self.config.vocab.pad_id).any()):
                raise ValueError("inactive branch positions must contain PAD")
        clean = cache.clean
        batch, branches, canvas = noisy_ids.shape
        time_condition = self._time_condition(noisy_ids, t)
        encoder_modulations, global_modulations, decoder_modulations, final_modulation = (
            self._adaln_modulations(time_condition)
        )
        if self.config.duo_mutable_topology == "full_resolution_decoder":
            del encoder_modulations, global_modulations
            decoder = self.embedding(noisy_ids).flatten(1, 2)
            branch_positions = cache.source_positions.flatten(1, 2)
            projected_conditions = cache.decoder_projected_branch_conditions
            if projected_conditions is None or len(projected_conditions) != len(
                self.decoder
            ):
                raise AssertionError("full-resolution condition cache is incomplete")
            for index, (block, bank, projected) in enumerate(
                zip(self.decoder, clean.decoder, projected_conditions)
            ):
                projected = projected[:, :, None, :].expand(
                    -1, -1, canvas, -1
                ).flatten(1, 2)
                if block.conditioning == "split_cross_attention":
                    decoder = block.forward_branch_from_clean_bank_adaln(
                        decoder,
                        projected,
                        decoder_modulations[index],
                        bank,
                        positions=branch_positions,
                        layout=cache.decoder_layout,
                        allow_dense_reference=noisy_ids.device.type == "cpu",
                        block_mask=cache.decoder_block_mask,
                    )
                else:
                    decoder = block.forward_branch_from_clean_bank_projected_adaln(
                        decoder,
                        projected,
                        decoder_modulations[index],
                        bank,
                        positions=branch_positions,
                        layout=cache.decoder_layout,
                        allow_dense_reference=noisy_ids.device.type == "cpu",
                        block_mask=cache.decoder_block_mask,
                    )
            final_shift, final_scale = final_modulation.chunk(2, dim=-1)
            branch_states = self.decoder_norm(
                decoder.view(batch, branches, canvas, self.config.local_dim)
            )
            branch_states = (
                branch_states * (1 + final_scale[:, :, None])
                + final_shift[:, :, None]
            ).flatten(1, 2)
            branch_logits = (
                F.linear(
                    branch_states,
                    self.embedding.weight[: self.config.vocab.output_size],
                )
                if self.output is None
                else self.output(branch_states)
            )
            clean_logits = (
                self._logits(clean.clean_decoder_states)
                if return_clean_logits
                else branch_logits.new_empty(
                    (batch, 0, self.config.vocab.output_size)
                )
            )
            return DiffusionGemmaOutput(
                clean_logits=clean_logits,
                branch_logits=branch_logits.view(
                    batch, branches, canvas, self.config.vocab.output_size
                ),
                clean_patch_states=clean.clean_patch_states,
                branch_patch_states=clean.clean_patch_states.new_empty(
                    (batch, branches, 0, self.config.global_dim)
                ),
                clean_decoder_states=(
                    self.decoder_norm(clean.clean_decoder_states)
                    if return_clean_states
                    else None
                ),
            )
        branch = self.embedding(noisy_ids).flatten(1, 2)
        ngrams = self._branch_ngram_features(
            clean, noisy_ids, cache.branch_valid, cache.branch_starts
        )
        if ngrams is not None:
            branch = branch + ngrams
            if self.config.ngram_aggregation == "mean":
                branch = branch / (len(self.config.ngram_orders) + 1)
        branch_positions = cache.source_positions.flatten(1, 2)
        for index, (block, bank) in enumerate(zip(self.encoder, clean.encoder)):
            branch = block.forward_branch_from_clean_bank_adaln(
                branch,
                encoder_modulations[index],
                bank,
                positions=branch_positions,
                layout=cache.local_layout,
                allow_dense_reference=noisy_ids.device.type == "cpu",
                block_mask=cache.local_block_mask,
            )

        branch_patches = self.pool(branch, cache.branch_valid.flatten(1, 2))
        patch_positions = (
            cache.source_positions[:, :, :: self.config.patch_stride]
            // self.config.patch_stride
        ).flatten(1, 2)
        for index, (block, bank) in enumerate(
            zip(self.global_blocks, clean.global_blocks)
        ):
            branch_patches = block.forward_branch_from_clean_bank_adaln(
                branch_patches,
                global_modulations[index],
                bank,
                positions=patch_positions,
                layout=cache.global_layout,
                allow_dense_reference=noisy_ids.device.type == "cpu",
                block_mask=cache.global_block_mask,
            )
        branch_patch_states = self.global_norm(branch_patches).view(
            batch,
            branches,
            canvas // self.config.patch_stride,
            self.config.global_dim,
        )
        branch_condition = branch_patch_states.repeat_interleave(
            self.config.patch_stride, dim=2
        ).flatten(1, 2)
        decoder = self.embedding(noisy_ids).flatten(1, 2)
        for index, (block, bank) in enumerate(zip(self.decoder, clean.decoder)):
            decoder = block.forward_branch_from_clean_bank_adaln(
                decoder,
                branch_condition,
                decoder_modulations[index],
                bank,
                positions=branch_positions,
                layout=cache.decoder_layout,
                allow_dense_reference=noisy_ids.device.type == "cpu",
                block_mask=cache.decoder_block_mask,
            )
        final_shift, final_scale = final_modulation.chunk(2, dim=-1)
        branch_states = self.decoder_norm(
            decoder.view(batch, branches, canvas, self.config.local_dim)
        )
        branch_states = (
            branch_states * (1 + final_scale[:, :, None])
            + final_shift[:, :, None]
        ).flatten(1, 2)
        branch_logits = (
            F.linear(
                branch_states,
                self.embedding.weight[: self.config.vocab.output_size],
            )
            if self.output is None
            else self.output(branch_states)
        )
        clean_logits = (
            self._logits(clean.clean_decoder_states)
            if return_clean_logits
            else branch_logits.new_empty((batch, 0, self.config.vocab.output_size))
        )
        return DiffusionGemmaOutput(
            clean_logits=clean_logits,
            branch_logits=branch_logits.view(
                batch, branches, canvas, self.config.vocab.output_size
            ),
            clean_patch_states=clean.clean_patch_states,
            branch_patch_states=branch_patch_states,
            clean_decoder_states=(
                self.decoder_norm(clean.clean_decoder_states)
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
        t: Tensor,
        *,
        attention_metadata=None,
        clean_patch_metadata: DuoCleanPatchMetadata | None = None,
        local_block_mask_metadata: CanvasBlockMaskMetadata | None = None,
        global_block_mask_metadata: CanvasBlockMaskMetadata | None = None,
        local_block_mask=None,
        global_block_mask=None,
        return_clean_logits: bool = False,
        return_clean_states: bool = False,
        allow_synthetic_branch_suffix: bool = False,
    ) -> DiffusionGemmaOutput:
        """Denoise clean atomic states; diffusion time is not optional."""

        if not t.is_floating_point():
            raise TypeError("Duo time conditioning must be floating point")
        time_condition = self._time_condition(noisy_ids, t)
        encoder_modulations, global_modulations, decoder_modulations, final_modulation = (
            self._adaln_modulations(time_condition)
        )
        if self.config.duo_mutable_topology == "full_resolution_decoder":
            return self._forward_full_resolution_decoder(
                clean_ids,
                clean_valid,
                document_ids,
                positions,
                noisy_ids,
                branch_valid,
                branch_starts,
                attention_metadata=attention_metadata,
                clean_patch_metadata=clean_patch_metadata,
                decoder_block_mask_metadata=local_block_mask_metadata,
                decoder_block_mask=local_block_mask,
                return_clean_logits=return_clean_logits,
                return_clean_states=return_clean_states,
                allow_synthetic_branch_suffix=allow_synthetic_branch_suffix,
                decoder_modulations=decoder_modulations,
                final_modulation=final_modulation,
            )
        return self._forward_branches(
            clean_ids,
            clean_valid,
            document_ids,
            positions,
            noisy_ids,
            branch_valid,
            branch_starts,
            attention_metadata=attention_metadata,
            local_block_mask_metadata=local_block_mask_metadata,
            global_block_mask_metadata=global_block_mask_metadata,
            local_block_mask=local_block_mask,
            global_block_mask=global_block_mask,
            return_clean_logits=return_clean_logits,
            return_clean_states=return_clean_states,
            allow_synthetic_branch_suffix=allow_synthetic_branch_suffix,
            encoder_modulations=encoder_modulations,
            global_modulations=global_modulations,
            decoder_modulations=decoder_modulations,
            final_modulation=final_modulation,
        )

    def _forward_full_resolution_decoder(
        self,
        clean_ids: Tensor,
        clean_valid: Tensor,
        document_ids: Tensor,
        positions: Tensor,
        noisy_ids: Tensor,
        branch_valid: Tensor,
        branch_starts: Tensor,
        *,
        attention_metadata: DiffusionGemmaAttentionMetadata | None,
        clean_patch_metadata: DuoCleanPatchMetadata | None,
        decoder_block_mask_metadata: CanvasBlockMaskMetadata | None,
        decoder_block_mask,
        return_clean_logits: bool,
        return_clean_states: bool,
        allow_synthetic_branch_suffix: bool,
        decoder_modulations: tuple[Tensor, ...],
        final_modulation: Tensor,
    ) -> DiffusionGemmaOutput:
        """Causal clean hierarchy plus an unpatched mutable byte decoder."""

        if clean_ids.ndim != 2 or clean_ids.shape != clean_valid.shape:
            raise ValueError("clean ids and valid must be aligned rank-2 tensors")
        if document_ids.shape != clean_ids.shape or positions.shape != clean_ids.shape:
            raise ValueError("document ids and positions must align with clean ids")
        if noisy_ids.ndim != 3 or noisy_ids.shape != branch_valid.shape:
            raise ValueError("noisy ids and validity must be aligned [B,M,C]")
        expects_entropy = self.config.duo_clean_patching == "causal_entropy_v1"
        if expects_entropy != (clean_patch_metadata is not None):
            raise ValueError(
                "Duo model config and supplied clean patch metadata disagree"
            )
        batch, branches, canvas = noisy_ids.shape
        length = clean_ids.shape[1]
        if canvas <= 0:
            raise ValueError("full-resolution canvas length must be positive")
        if clean_patch_metadata is None and length % self.config.patch_stride:
            raise ValueError("clean length must remain patch aligned")
        if branch_starts.shape != (batch, branches):
            raise ValueError("branch starts must align [B,M]")
        if len(decoder_modulations) != len(self.decoder):
            raise ValueError("decoder AdaLN module count differs from decoder depth")
        if not torch.compiler.is_compiling():
            if bool(((branch_starts < 0) | (branch_starts >= length)).any()):
                raise ValueError("canvas origins must lie inside the clean row")
            if bool(
                noisy_ids.masked_select(~branch_valid)
                .ne(self.config.vocab.pad_id)
                .any()
            ):
                raise ValueError("inactive branch positions must contain PAD")

        if attention_metadata is not None and clean_patch_metadata is not None:
            raise ValueError("Duo clean patch policies are mutually exclusive")
        if attention_metadata is None and clean_patch_metadata is None:
            attention_metadata = _packed_document_metadata(
                clean_valid,
                document_ids,
                patch_stride=self.config.patch_stride,
            )
        clean_metadata = (
            clean_patch_metadata
            if clean_patch_metadata is not None
            else attention_metadata
        )
        if clean_metadata is None:
            raise AssertionError("clean metadata disappeared")
        if clean_metadata.byte_indices.device != clean_ids.device:
            raise ValueError("attention metadata and inputs must share a device")

        offsets = torch.arange(canvas, device=clean_ids.device)
        indices = branch_starts[:, :, None] + offsets
        exists = indices.lt(length)
        safe = indices.clamp_max(length - 1)
        source_documents = torch.gather(
            document_ids[:, None, :].expand(-1, branches, -1), 2, safe
        )
        branch_segments = torch.gather(document_ids, 1, branch_starts)
        origin_positions = torch.gather(positions, 1, branch_starts)
        source_positions = origin_positions[:, :, None] + offsets
        expected_valid = (
            torch.gather(
                clean_valid[:, None, :].expand(-1, branches, -1), 2, safe
            )
            & exists
            & source_documents.eq(branch_segments[:, :, None])
        )
        if not torch.compiler.is_compiling():
            contiguous = offsets < branch_valid.sum(-1, keepdim=True)
            if allow_synthetic_branch_suffix:
                # Synthetic continuation is permitted only for rows already
                # authenticated as a single clean document.  This flag never
                # relaxes the attention layout's document or strict-prefix rule.
                valid_documents = torch.where(
                    clean_valid, document_ids, branch_segments[:, :1]
                )
                single_document = valid_documents.eq(
                    branch_segments[:, :1]
                ).all(1)
                if (
                    not torch.equal(branch_valid, contiguous)
                    or bool((expected_valid & ~branch_valid).any())
                    or not bool(single_document.all())
                ):
                    raise ValueError(
                        "synthetic inference suffix must follow one contiguous clean document"
                    )
            elif not torch.equal(branch_valid, contiguous) or bool(
                (branch_valid & ~expected_valid).any()
            ):
                raise ValueError(
                    "branch validity must be a contiguous same-document source prefix"
                )

        clean_patch_states, clean_condition = self._clean_hierarchy(
            clean_ids,
            clean_valid,
            document_ids,
            positions,
            attention_metadata,
            clean_patch_metadata,
        )
        preceding = self._preceding_patch_condition(
            clean_patch_states,
            clean_valid,
            document_ids,
            positions,
            branch_starts,
            clean_patch_metadata,
        )
        physical_ids = torch.cat((clean_ids, noisy_ids.flatten(1, 2)), dim=1)
        physical_positions = torch.cat(
            (positions, source_positions.flatten(1, 2)), dim=1
        )
        decoder_states = self.embedding(physical_ids)
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
        if decoder_block_mask is None and clean_ids.device.type != "cpu":
            decoder_block_mask = build_canvas_block_mask(
                decoder_layout, metadata=decoder_block_mask_metadata
            )
        for index, block in enumerate(self.decoder):
            if block.conditioning == "split_cross_attention":
                if clean_condition is None:
                    raise AssertionError("split decoder condition is unavailable")
                branch_condition = preceding[:, :, None, :].expand(
                    -1, -1, canvas, -1
                ).flatten(1, 2)
                condition = torch.cat((clean_condition, branch_condition), dim=1)
                decoder_states = block.forward_shared_document_branches_adaln(
                    decoder_states,
                    condition,
                    decoder_modulations[index],
                    positions=physical_positions,
                    clean_length=length,
                    clean_indices=clean_metadata.byte_indices,
                    clean_cu_seqlens=clean_metadata.byte_cu_seqlens,
                    assume_physical_clean=(
                        clean_patch_metadata is not None
                        or (
                            attention_metadata.physical_layout
                            if attention_metadata is not None
                            else False
                        )
                    ),
                    layout=decoder_layout,
                    clean_window=self.config.decoder_prefix_window,
                    allow_dense_reference=clean_ids.device.type == "cpu",
                    block_mask=decoder_block_mask,
                )
                continue
            projected_patches = block.project_reusable_condition(
                clean_patch_states
            )
            clean_projected = self._clean_byte_condition(
                projected_patches,
                clean_valid,
                document_ids,
                positions,
                clean_patch_metadata,
            )
            branch_projected = block.project_reusable_condition(preceding)
            branch_projected = branch_projected[:, :, None, :].expand(
                -1, -1, canvas, -1
            ).flatten(1, 2)
            projected_condition = torch.cat(
                (clean_projected, branch_projected), dim=1
            )
            decoder_states = block.forward_shared_document_branches_projected_adaln(
                decoder_states,
                projected_condition,
                decoder_modulations[index],
                positions=physical_positions,
                clean_length=length,
                clean_indices=clean_metadata.byte_indices,
                clean_cu_seqlens=clean_metadata.byte_cu_seqlens,
                assume_physical_clean=(
                    clean_patch_metadata is not None
                    or (
                        attention_metadata.physical_layout
                        if attention_metadata is not None
                        else False
                    )
                ),
                layout=decoder_layout,
                clean_window=self.config.decoder_prefix_window,
                allow_dense_reference=clean_ids.device.type == "cpu",
                block_mask=decoder_block_mask,
            )

        branch_states = decoder_states[:, length:].view(
            batch, branches, canvas, self.config.local_dim
        )
        final_shift, final_scale = final_modulation.chunk(2, dim=-1)
        branch_states = self.decoder_norm(branch_states)
        branch_states = (
            branch_states * (1 + final_scale[:, :, None])
            + final_shift[:, :, None]
        ).flatten(1, 2)
        branch_logits = (
            F.linear(
                branch_states,
                self.embedding.weight[: self.config.vocab.output_size],
            )
            if self.output is None
            else self.output(branch_states)
        )
        clean_decoder_states = decoder_states[:, :length]
        clean_logits = (
            self._logits(clean_decoder_states)
            if return_clean_logits
            else decoder_states.new_empty(
                (batch, 0, self.config.vocab.output_size)
            )
        )
        return DiffusionGemmaOutput(
            clean_logits=clean_logits,
            branch_logits=branch_logits.view(
                batch, branches, canvas, self.config.vocab.output_size
            ),
            clean_patch_states=clean_patch_states,
            branch_patch_states=clean_patch_states.new_empty(
                (batch, branches, 0, self.config.global_dim)
            ),
            clean_decoder_states=(
                self.decoder_norm(clean_decoder_states)
                if return_clean_states
                else None
            ),
        )


__all__ = (
    "CleanAtomEmbedding",
    "DUO_PARAMETER_COUNT",
    "DUO_TIME_CONDITION_DIM",
    "DUO_TIME_FEATURES",
    "DuoCanvasCache",
    "DuoCleanBank",
    "DuoModel",
    "sinusoidal_time_features",
)
