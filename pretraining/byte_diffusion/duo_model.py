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
from .layers import CleanAttentionBank


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
    attention_metadata: DiffusionGemmaAttentionMetadata
    encoder: tuple[CleanAttentionBank, ...]
    global_blocks: tuple[CleanAttentionBank, ...]
    decoder: tuple[CleanAttentionBank, ...]
    clean_patch_states: Tensor
    decoder_condition: Tensor
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
    local_layout: CanvasBranchLayout
    local_block_mask: object | None
    global_layout: CanvasBranchLayout
    global_block_mask: object | None
    decoder_layout: CanvasBranchLayout
    decoder_block_mask: object | None

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
        return F.silu(
            self.time_embedding(
                sinusoidal_time_features(sigma, self.config.duo_time_features)
            )
        )

    def _adaln_modulations(
        self, time_condition: Tensor
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...], Tensor]:
        values = self.time_adaln(time_condition).split(self._adaln_widths, dim=-1)
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
    ) -> DuoCleanBank:
        """Run the time-independent clean encoder/global/decoder exactly once."""

        if torch.is_grad_enabled():
            raise RuntimeError("Duo clean-bank caching is inference-only")
        if clean_ids.ndim != 2 or clean_ids.shape != clean_valid.shape:
            raise ValueError("clean ids and valid must be aligned rank-2 tensors")
        if document_ids.shape != clean_ids.shape or positions.shape != clean_ids.shape:
            raise ValueError("document ids and positions must align with clean ids")
        if clean_ids.shape[1] % self.config.patch_stride:
            raise ValueError("clean length must be patch aligned")
        if attention_metadata is None:
            attention_metadata = _packed_document_metadata(
                clean_valid,
                document_ids,
                patch_stride=self.config.patch_stride,
            )

        local = self.embedding(clean_ids)
        if self.ngrams is not None:
            local = local + self._isolated_ngram_features(
                clean_ids, clean_valid, document_ids
            )
            if self.config.ngram_aggregation == "mean":
                local = local / (len(self.config.ngram_orders) + 1)
        encoder_banks: list[CleanAttentionBank] = []
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
        clean_patch_valid, clean_patch_documents, clean_patch_positions = (
            _patch_metadata(
                clean_valid,
                document_ids,
                positions,
                self.config.patch_stride,
            )
        )

        global_banks: list[CleanAttentionBank] = []
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
        for block in self.decoder:
            decoder_states, bank = block.prepare_clean_bank(
                decoder_states,
                decoder_condition,
                clean_indices=attention_metadata.byte_indices,
                clean_cu_seqlens=attention_metadata.byte_cu_seqlens,
                assume_physical_clean=attention_metadata.physical_layout,
                positions=positions,
                clean_window=self.config.decoder_prefix_window,
                allow_dense_reference=clean_ids.device.type == "cpu",
            )
            decoder_banks.append(bank)
        return DuoCleanBank(
            clean_ids,
            clean_valid,
            document_ids,
            positions,
            attention_metadata,
            tuple(encoder_banks),
            tuple(global_banks),
            tuple(decoder_banks),
            clean_patch_states,
            decoder_condition,
            decoder_states,
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
        if canvas % self.config.patch_stride:
            raise ValueError("canvas length must be patch aligned")
        if validate_inputs and not torch.compiler.is_compiling():
            if bool((branch_starts % self.config.patch_stride).any()):
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
            if allow_synthetic_branch_suffix:
                contiguous = offsets < branch_valid.sum(-1, keepdim=True)
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
            elif not torch.equal(branch_valid, expected_valid):
                raise ValueError(
                    "branch validity crosses a document or disagrees with clean data"
                )

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
