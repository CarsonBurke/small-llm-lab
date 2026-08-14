"""Scratch BLT-shaped byte model shared by AR and diffusion objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import ByteDiffusionConfig, ModelMode
from .attention import CanvasBranchLayout, build_canvas_block_mask
from .layers import (
    ConditionedTransformerBlock,
    PatchPool,
    RMSNorm,
    TransformerBlock,
    pack_valid,
    pack_rows,
    packed_sequence_offsets,
    prefix_row_indices,
    valid_row_indices,
    unpack_valid,
    unpack_rows,
)


@dataclass(frozen=True)
class ModelOutput:
    logits: Tensor
    byte_states: Tensor
    patch_states: Tensor
    bos_patch_states: Tensor | None = None


@dataclass(frozen=True)
class CanvasBranchOutput:
    clean_logits: Tensor
    branch_logits: Tensor
    clean_patch_states: Tensor
    branch_patch_states: Tensor


@dataclass(frozen=True)
class BltBranchOutput:
    clean_logits: Tensor
    branch_logits: Tensor
    clean_patch_states: Tensor
    bos_patch_states: Tensor | None = None


class CausalNgramEmbedding(nn.Module):
    """Factorized visible-id n-grams ending at each absolute byte position."""

    def __init__(self, config: ByteDiffusionConfig) -> None:
        super().__init__()
        self.orders = config.ngram_orders
        self.table_size = config.ngram_table_size
        self.table_mask = (
            config.ngram_table_size - 1
            if config.ngram_table_size & (config.ngram_table_size - 1) == 0
            else None
        )
        self.eot_id = config.vocab.eot_id
        self.pad_id = config.vocab.pad_id
        self.hash_kind = config.ngram_hash
        self.table = (
            nn.Embedding(config.ngram_table_size, config.ngram_rank)
            if config.ngram_table_sharing == "shared"
            else None
        )
        self.tables = (
            None
            if config.ngram_table_sharing == "shared"
            else nn.ModuleList(
                nn.Embedding(config.ngram_table_size, config.ngram_rank)
                for _ in self.orders
            )
        )
        self.projection_weight = nn.Parameter(
            torch.empty(
                len(self.orders) * config.local_dim,
                config.ngram_rank,
            )
        )

    @property
    def projection_weights(self) -> Tensor:
        return self.projection_weight.view(
            len(self.orders), -1, self.projection_weight.shape[-1]
        )

    def _hash_ids(self, ids: Tensor, order: int) -> tuple[Tensor, Tensor]:
        """Return configured polynomial hashes and their causal validity mask."""

        rolling = torch.zeros_like(ids)
        context_ok = torch.ones_like(ids, dtype=torch.bool)
        if self.hash_kind == "blt_prime":
            # Appendix C puts the current byte at exponent zero, so consume
            # oldest-to-current in the Horner recurrence. The 10-digit prime
            # avoids the legacy base-257 collapse modulo a power-of-two table.
            base = 1_000_000_007
            offsets = reversed(range(order))
            addend = 0
        else:
            base = 257
            offsets = range(order)
            addend = 1
        for offset in offsets:
            shifted = torch.zeros_like(ids)
            if offset == 0:
                shifted = ids
            elif offset < ids.shape[1]:
                shifted[:, offset:] = ids[:, :-offset]
            rolling = rolling * base + shifted + addend
            rolling = (
                rolling & self.table_mask
                if self.table_mask is not None
                else rolling % self.table_size
            )
            if offset:
                context_ok &= (shifted != self.eot_id) & (shifted != self.pad_id)
        enough_context = torch.arange(ids.shape[1], device=ids.device) >= order - 1
        return rolling, context_ok & enough_context[None, :]

    def _forward_impl(
        self,
        ids: Tensor,
        *,
        valid: Tensor | None = None,
        segment_ids: Tensor | None = None,
    ) -> Tensor:
        """Embed all orders with one recurrence and optional segment resets."""

        if (valid is None) != (segment_ids is None):
            raise ValueError("valid and segment ids must be supplied together")
        if valid is not None:
            if valid.shape != ids.shape or valid.dtype != torch.bool:
                raise ValueError("n-gram validity must be aligned boolean storage")
            if segment_ids is None or segment_ids.shape != ids.shape:
                raise ValueError("n-gram segment ids must align with input ids")

        features: list[Tensor] = []
        maximum_order = max(self.orders)
        base = 1_000_000_007 if self.hash_kind == "blt_prime" else 257
        addend = 0 if self.hash_kind == "blt_prime" else 1
        rolling = ids + addend
        rolling = (
            rolling & self.table_mask
            if self.table_mask is not None
            else rolling % self.table_size
        )
        context_ok = torch.ones_like(ids, dtype=torch.bool)
        position_indices = torch.arange(ids.shape[1], device=ids.device)
        power = 1
        feature_index = 0
        for order in range(1, maximum_order + 1):
            if order > 1:
                offset = order - 1
                shifted = torch.zeros_like(ids)
                shifted[:, offset:] = ids[:, :-offset]
                if self.hash_kind == "blt_prime":
                    power = (power * base) % self.table_size
                    rolling = rolling + shifted * power
                else:
                    # legacy257 consumes current-to-oldest in its Horner
                    # recurrence; extending the order therefore appends the
                    # newly exposed older byte at exponent zero.
                    rolling = rolling * base + shifted + addend
                rolling = (
                    rolling & self.table_mask
                    if self.table_mask is not None
                    else rolling % self.table_size
                )
                context_ok &= (shifted != self.eot_id) & (
                    shifted != self.pad_id
                )
                if valid is not None:
                    prior_valid = torch.zeros_like(valid)
                    prior_valid[:, offset:] = valid[:, :-offset]
                    prior_segment = torch.zeros_like(segment_ids)
                    prior_segment[:, offset:] = segment_ids[:, :-offset]
                    context_ok &= prior_valid & prior_segment.eq(segment_ids)
            if order not in self.orders:
                continue
            enough_context = position_indices >= order - 1
            active = context_ok & enough_context[None]
            if valid is not None:
                active &= valid
            table = (
                self.table
                if self.table is not None
                else self.tables[feature_index]
            )
            features.append(table(rolling) * active[:, :, None])
            feature_index += 1
        stacked = torch.stack(features, dim=2)
        # Flatten order and rank into one tensor-core-friendly projection.
        # The former four-index einsum lowered to several small contractions
        # and intermediate transposes on the byte hot path.  This is exactly
        # the same per-order low-rank sum expressed as one dense GEMM.
        projection = self.projection_weights.permute(0, 2, 1).reshape(
            len(self.orders) * self.projection_weight.shape[-1], -1
        )
        return stacked.flatten(2) @ projection

    def forward(self, ids: Tensor) -> Tensor:
        """Embed all configured orders with one shared rolling recurrence."""

        return self._forward_impl(ids)

    def forward_segmented(
        self, ids: Tensor, valid: Tensor, segment_ids: Tensor
    ) -> Tensor:
        """Embed n-grams while resetting context at packed-document edges."""

        return self._forward_impl(ids, valid=valid, segment_ids=segment_ids)


def _causal_window_mask(length: int, window: int, device: torch.device) -> Tensor:
    positions = torch.arange(length, device=device)
    distance = positions[:, None] - positions[None, :]
    return (distance >= 0) & (distance < window)


def _bidirectional_valid_mask(valid: Tensor) -> Tensor:
    return valid[:, :, None] & valid[:, None, :]


class ByteDiffusionModel(nn.Module):
    def __init__(self, config: ByteDiffusionConfig = ByteDiffusionConfig()) -> None:
        super().__init__()
        self.config = config
        self.activation_checkpointing = False
        self.require_compiled_training = False
        self.embedding = nn.Embedding(
            config.vocab.input_size, config.local_dim, padding_idx=config.vocab.pad_id
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
        self.mode_embedding = (
            nn.Embedding(len(ModelMode), config.local_dim)
            if config.mode_embeddings
            else None
        )
        self.output = (
            None
            if config.output_tied
            else nn.Linear(config.local_dim, config.vocab.output_size, bias=False)
        )
        self.timestep_vector = (
            nn.Parameter(torch.empty(config.local_dim)) if config.explicit_timestep else None
        )
        self.self_conditioner = (
            nn.Sequential(
                nn.Linear(config.local_dim, config.self_conditioning_hidden, bias=False),
                nn.SiLU(),
                nn.Linear(config.self_conditioning_hidden, config.local_dim, bias=False),
            )
            if config.self_conditioning
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            if self.ngrams is not None:
                nn.init.normal_(
                    self.ngrams.projection_weight,
                    mean=0.0,
                    std=0.02,
                )
            self.embedding.weight[self.config.vocab.pad_id].zero_()
            if (
                self.ngrams is not None
                and self.config.ngram_factor_init == "scale_matched"
            ):
                self.ngrams.projection_weight.mul_(
                    self.config.ngram_rank**-0.5 / 0.02
                )
            if self.timestep_vector is not None:
                nn.init.normal_(self.timestep_vector, mean=0.0, std=0.02)

    def combine_ngram_features(self, base: Tensor, features: Tensor) -> Tensor:
        combined = base + features
        if self.config.ngram_aggregation == "mean":
            if self.ngrams is None:
                raise AssertionError("ngram mean requested without ngrams")
            combined = combined / (len(self.ngrams.orders) + 1)
        return combined

    def encoder_embeddings(self, ids: Tensor) -> Tensor:
        states = self.embedding(ids)
        if self.ngrams is not None:
            states = self.combine_ngram_features(states, self.ngrams(ids))
        return states

    def decoder_embeddings(
        self,
        ids: Tensor,
        mode: ModelMode,
        *,
        mode_ids: Tensor | None = None,
    ) -> Tensor:
        """Return fresh decoder lookups with an optional recipe-type cue."""

        states = self.embedding(ids)
        if mode_ids is not None and (
            mode_ids.shape != ids.shape or mode_ids.dtype != torch.long
        ):
            raise ValueError("mode ids must be aligned int64 values")
        if self.mode_embedding is None:
            return states
        return states + (
            self.mode_embedding.weight[int(mode)]
            if mode_ids is None
            else self.mode_embedding(mode_ids)
        )

    def virtual_bos_patch_input(
        self,
        device: torch.device,
        *,
        allow_dense_reference: bool,
    ) -> Tensor:
        """Encode Fast BLT's one-byte BOS patch through the shared local path."""

        stride = self.config.patch_stride
        ids = torch.full(
            (1, stride),
            self.config.vocab.pad_id,
            dtype=torch.long,
            device=device,
        )
        ids[:, 0] = self.config.vocab.eot_id
        valid = torch.zeros((1, stride), dtype=torch.bool, device=device)
        valid[:, 0] = True
        padded = self.encoder_embeddings(ids)
        local = padded[:, :1].reshape(1, self.config.local_dim)
        cu = torch.tensor([0, 1], dtype=torch.int32, device=device)
        positions = torch.zeros(1, dtype=torch.long, device=device)
        for block in self.encoder:
            local = block.forward_packed(
                local,
                cu_seqlens=cu,
                positions=positions,
                max_seqlen=1,
                window=1,
                allow_dense_reference=allow_dense_reference,
            )
        local_padded = padded.clone()
        local_padded[:, 0] = local
        return self.pool(local_padded, valid)[0, 0]

    def virtual_bos_global_states(
        self,
        count: int,
        *,
        device: torch.device,
        allow_dense_reference: bool,
    ) -> Tensor:
        """Return document-isolated normalized outputs of the virtual BOS patch."""

        if count < 0:
            raise ValueError("virtual BOS count cannot be negative")
        if count == 0:
            return self.embedding.weight.new_empty((0, self.config.global_dim))
        states = self.virtual_bos_patch_input(
            device, allow_dense_reference=allow_dense_reference
        )[None]
        cu = torch.tensor([0, 1], dtype=torch.int32, device=device)
        positions = torch.zeros(1, dtype=torch.long, device=device)
        for block in self.global_blocks:
            states = block.forward_packed(
                states,
                cu_seqlens=cu,
                positions=positions,
                max_seqlen=1,
                window=self.config.global_window,
                allow_dense_reference=allow_dense_reference,
            )
        return self.global_norm(states).expand(count, -1)

    def enforce_padding_invariant(self) -> None:
        """Restore the fixed PAD row after optimizers that apply decoupled decay."""

        with torch.no_grad():
            self.embedding.weight[self.config.vocab.pad_id].zero_()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate(self, ids: Tensor, valid: Tensor) -> None:
        if ids.ndim != 2 or ids.dtype != torch.long or ids.shape != valid.shape:
            raise ValueError("ids must be rank-2 int64 and align with valid")
        if valid.dtype != torch.bool:
            raise TypeError("valid must be boolean")
        if ids.shape[1] % self.config.patch_stride:
            raise ValueError("physical rows must be padded to a patch boundary")
        if not torch.compiler.is_compiling():
            if bool((~valid.any(1)).any()):
                raise ValueError("every row must contain a valid atomic id")
            if bool((ids[valid] >= self.config.vocab.input_size).any()) or bool(
                (ids[valid] < 0).any()
            ):
                raise ValueError("valid input id out of range")
            if bool((ids[~valid] != self.config.vocab.pad_id).any()):
                raise ValueError("invalid positions must store PAD")

    @staticmethod
    def _prefix_lengths(valid: Tensor) -> Tensor:
        lengths = valid.sum(1)
        expected = torch.arange(valid.shape[1], device=valid.device)[None] < lengths[:, None]
        if not torch.compiler.is_compiling() and not torch.equal(valid, expected):
            raise ValueError("varlen production path requires one contiguous document per row")
        return lengths

    @staticmethod
    def _cu_seqlens(lengths: Tensor) -> Tensor:
        return packed_sequence_offsets(lengths)

    @staticmethod
    def _fixed_cu_seqlens(
        batch_size: int, sequence_length: int, device: torch.device
    ) -> Tensor:
        """Offsets for fixed-width rows without compiling a constant cumsum."""

        return torch.arange(
            batch_size + 1, dtype=torch.int32, device=device
        ) * sequence_length

    @staticmethod
    def _segment_lengths(
        valid: Tensor, document_ids: Tensor, indices: Tensor
    ) -> Tensor:
        """Lengths of row- and document-isolated sequences in packed order."""

        if document_ids.shape != valid.shape or document_ids.dtype != torch.long:
            raise ValueError("document ids must be aligned int64 values")
        flat_documents = document_ids.reshape(-1).index_select(0, indices)
        flat_rows = torch.div(indices, valid.shape[1], rounding_mode="floor")
        if not torch.compiler.is_compiling() and bool((flat_documents < 0).any()):
            raise ValueError("valid atoms need nonnegative document ids")
        boundary = torch.ones_like(flat_documents, dtype=torch.bool)
        boundary[1:] = (flat_rows[1:] != flat_rows[:-1]) | (
            flat_documents[1:] != flat_documents[:-1]
        )
        starts = torch.nonzero(boundary, as_tuple=False).flatten()
        stops = torch.cat((starts[1:], starts.new_tensor([indices.numel()])))
        return stops - starts

    def encode(
        self,
        ids: Tensor,
        valid: Tensor,
        *,
        bidirectional: bool = False,
        positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        self._validate(ids, valid)
        length = ids.shape[1]
        local = self.encoder_embeddings(ids)
        if bidirectional:
            local_allowed = _bidirectional_valid_mask(valid)
        else:
            base = _causal_window_mask(length, self.config.local_window, ids.device)
            local_allowed = base[None] & valid[:, :, None] & valid[:, None, :]
        for block in self.encoder:
            local = block(local, positions=positions, allowed=local_allowed)
        patches = self.pool(local, valid)
        patch_valid = valid.view(valid.shape[0], -1, self.config.patch_stride).any(-1)
        if bidirectional:
            global_allowed = _bidirectional_valid_mask(patch_valid)
        else:
            patch_length = patches.shape[1]
            causal = torch.ones(
                patch_length, patch_length, dtype=torch.bool, device=ids.device
            ).tril()
            global_allowed = causal[None] & patch_valid[:, :, None] & patch_valid[:, None, :]
        patch_positions = None
        if positions is not None:
            patch_positions = (
                positions[..., :: self.config.patch_stride]
                // self.config.patch_stride
            )
        for block in self.global_blocks:
            patches = block(patches, positions=patch_positions, allowed=global_allowed)
        return local, self.global_norm(patches)

    def _aligned_condition(self, patches: Tensor, length: int, mode: ModelMode) -> Tensor:
        if mode is ModelMode.AR:
            # Position i reads the most recent fully closed patch.  Expressing
            # this as padding plus repeat_interleave avoids a PyTorch 2.13
            # Inductor/SymPy failure on the equivalent symbolic floor divide.
            initial = patches.new_zeros(
                (patches.shape[0], self.config.patch_stride - 1, patches.shape[-1])
            )
            repeated = patches.repeat_interleave(self.config.patch_stride, dim=1)
            return torch.cat((initial, repeated), dim=1)[:, :length]
        return patches.repeat_interleave(self.config.patch_stride, dim=1)[:, :length]

    def _document_aligned_condition(
        self,
        patches: Tensor,
        valid: Tensor,
        document_ids: Tensor,
        positions: Tensor,
        mode: ModelMode,
    ) -> Tensor:
        """Align patch states without leaking across packed documents."""

        batch, length = valid.shape
        stride = self.config.patch_stride
        patch_documents = document_ids.view(batch, -1, stride).amax(-1)
        physical_patch = torch.arange(length, device=valid.device) // stride
        if mode is ModelMode.AR:
            chosen_patch = physical_patch[None].expand(batch, -1) - 1
            chosen_patch = torch.where(
                positions.remainder(stride).eq(stride - 1),
                physical_patch[None],
                chosen_patch,
            )
        else:
            chosen_patch = physical_patch[None].expand(batch, -1)
        gather_patch = chosen_patch.clamp_min(0)
        gathered_states = torch.gather(
            patches,
            1,
            gather_patch[..., None].expand(-1, -1, patches.shape[-1]),
        )
        gathered_documents = torch.gather(patch_documents, 1, gather_patch)
        active = (
            valid
            & chosen_patch.ge(0)
            & gathered_documents.eq(document_ids)
        )
        return torch.where(active[..., None], gathered_states, 0)

    def decode(
        self,
        ids: Tensor,
        valid: Tensor,
        patches: Tensor,
        mode: ModelMode,
        *,
        positions: Tensor | None = None,
        allowed_override: Tensor | None = None,
        condition_override: Tensor | None = None,
        mode_ids_override: Tensor | None = None,
        timestep: Tensor | None = None,
        self_condition_probs: Tensor | None = None,
    ) -> Tensor:
        states = self.decoder_embeddings(ids, mode, mode_ids=mode_ids_override)
        if self.timestep_vector is not None:
            if timestep is None:
                raise ValueError("explicit-timestep model requires timestep values")
            timestep = timestep.to(states.dtype)
            if timestep.ndim == 0:
                timestep = timestep.expand(ids.shape[0])
            if timestep.shape != (ids.shape[0],):
                raise ValueError("timestep must be scalar or one value per row")
            states = states + timestep[:, None, None] * self.timestep_vector
        elif timestep is not None:
            raise ValueError("timestep supplied to a timestep-free model")
        if self.self_conditioner is not None:
            if self_condition_probs is None or self_condition_probs.shape != (
                *ids.shape,
                self.config.vocab.output_size,
            ):
                raise ValueError("self-conditioning probabilities must align with ids")
            expected = self_condition_probs.detach().to(states.dtype) @ self.embedding.weight[
                : self.config.vocab.output_size
            ]
            states = states + self.self_conditioner(expected)
        elif self_condition_probs is not None:
            raise ValueError("self-conditioning supplied to a model without that arm")
        length = ids.shape[1]
        if allowed_override is not None:
            allowed = allowed_override
        elif mode is ModelMode.AR:
            base = _causal_window_mask(length, self.config.local_window, ids.device)
            allowed = base[None] & valid[:, :, None] & valid[:, None, :]
        else:
            allowed = _bidirectional_valid_mask(valid)
        condition = (
            condition_override
            if condition_override is not None
            else self._aligned_condition(patches, length, mode)
        )
        for block in self.decoder:
            states = block(
                states,
                condition,
                positions=positions,
                allowed=allowed,
            )
        states = self.decoder_norm(states)
        if self.output is None:
            return F.linear(states, self.embedding.weight[: self.config.vocab.output_size])
        return self.output(states)

    def forward(
        self,
        ids: Tensor,
        valid: Tensor,
        *,
        mode: ModelMode = ModelMode.AR,
        positions: Tensor | None = None,
        timestep: Tensor | None = None,
        self_condition_probs: Tensor | None = None,
    ) -> ModelOutput:
        # BLT-D deliberately uses the same clean-prefix encoder in training via
        # ``forward_blt_d``. A standalone denoising row is encoded bidirectionally.
        bidirectional = mode is ModelMode.CANVAS
        byte_states, patch_states = self.encode(
            ids, valid, bidirectional=bidirectional, positions=positions
        )
        logits = self.decode(
            ids,
            valid,
            patch_states,
            mode,
            positions=positions,
            timestep=timestep,
            self_condition_probs=self_condition_probs,
        )
        return ModelOutput(logits=logits, byte_states=byte_states, patch_states=patch_states)

    def _encode_clean_varlen(
        self,
        ids: Tensor,
        valid: Tensor,
        *,
        positions: Tensor | None = None,
        allow_dense_reference: bool = False,
        assume_full_clean: bool = False,
        document_ids: Tensor | None = None,
        byte_indices: Tensor | None = None,
        byte_cu_seqlens: Tensor | None = None,
        patch_indices_override: Tensor | None = None,
        patch_cu_seqlens: Tensor | None = None,
        global_patch_sources: Tensor | None = None,
        global_patch_positions: Tensor | None = None,
        physical_to_global_patch_indices: Tensor | None = None,
        patch_byte_cu_seqlens: Tensor | None = None,
        max_patch_size: int | None = None,
        materialize_padded_patches: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        self._validate(ids, valid)
        if (
            assume_full_clean
            and not torch.compiler.is_compiling()
            and not bool(valid.all())
        ):
            raise ValueError("assume_full_clean requires every clean slot to be valid")
        prepacked = byte_indices is not None
        if prepacked != (byte_cu_seqlens is not None):
            raise ValueError("byte packed indices and offsets must be supplied together")
        virtual_bos_values = (
            global_patch_sources,
            global_patch_positions,
            physical_to_global_patch_indices,
        )
        has_virtual_bos = global_patch_sources is not None
        if sum(value is not None for value in virtual_bos_values) not in {0, 3}:
            raise ValueError("virtual BOS patch metadata must be supplied together")
        variable_patches = patch_byte_cu_seqlens is not None
        if variable_patches != (max_patch_size is not None):
            raise ValueError(
                "variable patch byte offsets and maximum size must be supplied together"
            )
        if variable_patches and (not prepacked or not has_virtual_bos):
            raise ValueError(
                "variable patches require packed bytes and explicit global topology"
            )
        if not variable_patches and (
            (patch_indices_override is None) != (patch_cu_seqlens is None)
        ):
            raise ValueError("patch packed indices and offsets must be supplied together")
        if has_virtual_bos and patch_indices_override is None and not variable_patches:
            raise ValueError("virtual BOS patches require packed physical patch indices")
        document_packed = document_ids is not None or prepacked
        lengths = (
            valid.sum(1)
            if document_packed
            else self._prefix_lengths(valid)
        )
        if not torch.compiler.is_compiling() and bool((lengths == 0).any()):
            raise ValueError("varlen rows must be nonempty")
        if positions is None:
            positions = torch.arange(ids.shape[1], device=ids.device)[None].expand_as(ids)
        if positions.shape != ids.shape:
            raise ValueError("absolute positions must align with ids")
        max_length = ids.shape[1]
        padded_local = self.encoder_embeddings(ids)
        if assume_full_clean and document_packed:
            raise ValueError("document-packed pages cannot assume one full sequence per row")
        if assume_full_clean:
            lengths = torch.full_like(lengths, ids.shape[1])
            cu = self._fixed_cu_seqlens(
                ids.shape[0], ids.shape[1], ids.device
            )
            local = padded_local.reshape(-1, self.config.local_dim)
            packed_positions = positions.reshape(-1)
            clean_indices = torch.arange(
                ids.numel(), dtype=torch.long, device=ids.device
            )
        else:
            clean_indices = (
                byte_indices
                if byte_indices is not None
                else (
                    valid_row_indices(valid)
                    if document_packed
                    else prefix_row_indices(valid)
                )
            )
            if byte_cu_seqlens is not None:
                cu = byte_cu_seqlens
                lengths = cu[1:] - cu[:-1]
            elif document_ids is not None:
                lengths = self._segment_lengths(valid, document_ids, clean_indices)
                cu = self._cu_seqlens(lengths)
            else:
                cu = self._cu_seqlens(lengths)
            local = pack_rows(padded_local, clean_indices)
            packed_positions = pack_rows(positions, clean_indices)
        for block in self.encoder:
            local = block.forward_packed(
                local,
                cu_seqlens=cu,
                positions=packed_positions,
                max_seqlen=max_length,
                window=self.config.local_window,
                allow_dense_reference=allow_dense_reference,
            )
        if assume_full_clean:
            local_padded = local.view_as(padded_local)
        else:
            local_padded = unpack_rows(local, clean_indices, padded_local)
        if variable_patches:
            if patch_byte_cu_seqlens is None or max_patch_size is None:
                raise AssertionError("variable patch metadata disappeared")
            physical_packed_patches = self.pool.forward_packed(
                local,
                patch_byte_cu_seqlens,
                max_patch_size=max_patch_size,
            )
            patches = physical_packed_patches.new_empty(
                (ids.shape[0], 0, self.config.global_dim)
            )
            patch_valid = None
            patch_lengths = None
        else:
            patches = self.pool(local_padded, valid)
            patch_valid = valid.view(
                valid.shape[0], -1, self.config.patch_stride
            ).any(-1)
            patch_lengths = patch_valid.sum(1)
        # Global RoPE is expressed in patch units everywhere.  Using raw byte
        # offsets here would make the clean bank rotate four times faster than
        # canvas/BLT branches despite representing the same physical patches.
        patch_positions_padded = (
            None
            if variable_patches
            else positions[:, :: self.config.patch_stride]
            // self.config.patch_stride
        )
        if variable_patches:
            if global_patch_sources is None or global_patch_positions is None:
                raise AssertionError("variable global topology disappeared")
            safe_sources = global_patch_sources.clamp_min(0)
            packed_patches = physical_packed_patches.index_select(0, safe_sources)
            bos_patch = self.virtual_bos_patch_input(
                ids.device, allow_dense_reference=allow_dense_reference
            )
            packed_patches = torch.where(
                global_patch_sources[:, None].ge(0),
                packed_patches,
                bos_patch[None],
            )
            packed_patch_positions = global_patch_positions
            patch_indices = None
        elif assume_full_clean:
            patch_indices = torch.arange(
                patch_valid.numel(), dtype=torch.long, device=ids.device
            )
            packed_patches = patches.reshape(-1, self.config.global_dim)
            packed_patch_positions = patch_positions_padded.reshape(-1)
        else:
            patch_indices = (
                patch_indices_override
                if patch_indices_override is not None
                else (
                    valid_row_indices(patch_valid)
                    if document_packed
                    else prefix_row_indices(patch_valid)
                )
            )
            physical_packed_patches = pack_rows(patches, patch_indices)
            physical_packed_positions = pack_rows(
                patch_positions_padded, patch_indices
            )
            if has_virtual_bos:
                if global_patch_sources is None or global_patch_positions is None:
                    raise AssertionError("virtual BOS metadata disappeared")
                safe_sources = global_patch_sources.clamp_min(0)
                packed_patches = physical_packed_patches.index_select(
                    0, safe_sources
                )
                bos_patch = self.virtual_bos_patch_input(
                    ids.device, allow_dense_reference=allow_dense_reference
                )
                packed_patches = torch.where(
                    global_patch_sources[:, None].ge(0),
                    packed_patches,
                    bos_patch[None],
                )
                packed_patch_positions = global_patch_positions
            else:
                packed_patches = physical_packed_patches
                packed_patch_positions = physical_packed_positions
        if variable_patches:
            if patch_cu_seqlens is None:
                raise ValueError("variable patches require global patch sequence offsets")
            patch_cu = patch_cu_seqlens
        elif assume_full_clean:
            patch_lengths = torch.full_like(patch_lengths, patches.shape[1])
            patch_cu = self._fixed_cu_seqlens(
                patches.shape[0], patches.shape[1], patches.device
            )
        else:
            if patch_cu_seqlens is not None:
                patch_cu = patch_cu_seqlens
                patch_lengths = patch_cu[1:] - patch_cu[:-1]
            elif document_ids is not None:
                if patch_valid is None:
                    raise AssertionError("fixed patch validity disappeared")
                patch_documents = document_ids.view(
                    document_ids.shape[0], -1, self.config.patch_stride
                ).amax(-1)
                patch_lengths = self._segment_lengths(
                    patch_valid, patch_documents, patch_indices
                )
                patch_cu = self._cu_seqlens(patch_lengths)
            else:
                patch_cu = self._cu_seqlens(patch_lengths)
        for block in self.global_blocks:
            packed_patches = block.forward_packed(
                packed_patches,
                cu_seqlens=patch_cu,
                positions=packed_patch_positions,
                max_seqlen=(
                    ids.shape[1]
                    if variable_patches
                    else patches.shape[1] + (1 if has_virtual_bos else 0)
                ),
                window=self.config.global_window,
                allow_dense_reference=allow_dense_reference,
            )
        normalized_packed_patches = self.global_norm(packed_patches)
        if not materialize_padded_patches:
            normalized_patches = normalized_packed_patches.new_empty(
                (ids.shape[0], 0, self.config.global_dim)
            )
        elif assume_full_clean:
            normalized_patches = normalized_packed_patches.view_as(patches)
        elif variable_patches:
            normalized_patches = patches
        else:
            physical_normalized = normalized_packed_patches
            if has_virtual_bos:
                if physical_to_global_patch_indices is None:
                    raise AssertionError("physical/global patch map disappeared")
                physical_normalized = normalized_packed_patches.index_select(
                    0, physical_to_global_patch_indices
                )
            normalized_patches = unpack_rows(
                physical_normalized, patch_indices, patches
            )
        return (
            local_padded,
            normalized_patches,
            cu,
            lengths,
            packed_positions,
            clean_indices,
            normalized_packed_patches,
        )

    def forward_ar_varlen(
        self,
        ids: Tensor,
        valid: Tensor,
        *,
        positions: Tensor | None = None,
        allow_dense_reference: bool = False,
        assume_full_clean: bool = False,
        return_padded_logits: bool = True,
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
        patch_byte_cu_seqlens: Tensor | None = None,
        max_patch_size: int | None = None,
    ) -> ModelOutput:
        """Production PAD-free native-varlen Flash path for clean AR rows."""

        if positions is None:
            positions = torch.arange(ids.shape[1], device=ids.device)[None].expand_as(ids)
        (
            local_padded,
            normalized_patches,
            cu,
            lengths,
            packed_positions,
            clean_indices,
            normalized_packed_patches,
        ) = (
            self._encode_clean_varlen(
                ids,
                valid,
                positions=positions,
                allow_dense_reference=allow_dense_reference,
                assume_full_clean=assume_full_clean,
                document_ids=document_ids,
                byte_indices=byte_indices,
                byte_cu_seqlens=byte_cu_seqlens,
                patch_indices_override=patch_indices,
                patch_cu_seqlens=patch_cu_seqlens,
                global_patch_sources=global_patch_sources,
                global_patch_positions=global_patch_positions,
                physical_to_global_patch_indices=physical_to_global_patch_indices,
                patch_byte_cu_seqlens=patch_byte_cu_seqlens,
                max_patch_size=max_patch_size,
            )
        )

        bos_patch_states = (
            None
            if bos_condition_indices is None
            else normalized_packed_patches.index_select(0, bos_condition_indices)
        )

        if condition_patch_indices is not None:
            safe_condition = condition_patch_indices.clamp_min(0)
            packed_condition = normalized_packed_patches.index_select(
                0, safe_condition
            )
            packed_condition = torch.where(
                condition_patch_indices[:, None].ge(0),
                packed_condition,
                0,
            )
            condition = None
        else:
            packed_condition = None
            condition = (
            self._document_aligned_condition(
                normalized_patches,
                valid,
                document_ids,
                positions,
                ModelMode.AR,
            )
            if document_ids is not None
            else self._aligned_condition(
                normalized_patches, ids.shape[1], ModelMode.AR
            )
            )
        padded_states = self.decoder_embeddings(ids, ModelMode.AR)
        states = (
            padded_states.reshape(-1, self.config.local_dim)
            if assume_full_clean
            else pack_rows(padded_states, clean_indices)
        )
        if packed_condition is None:
            if condition is None:
                raise AssertionError("AR condition was not constructed")
            packed_condition = (
                condition.reshape(-1, self.config.global_dim)
                if assume_full_clean
                else pack_rows(condition, clean_indices)
            )
        for block in self.decoder:
            states = block.forward_packed(
                states,
                packed_condition,
                cu_seqlens=cu,
                positions=packed_positions,
                max_seqlen=ids.shape[1],
                window=self.config.decoder_prefix_window,
                allow_dense_reference=allow_dense_reference,
            )
        normalized_states = self.decoder_norm(states)
        if self.output is None:
            packed_logits = F.linear(
                normalized_states, self.embedding.weight[: self.config.vocab.output_size]
            )
        else:
            packed_logits = self.output(normalized_states)
        if not return_padded_logits:
            logits = packed_logits
        elif assume_full_clean:
            logits = packed_logits.view(*ids.shape, self.config.vocab.output_size)
        else:
            logits = unpack_rows(
                packed_logits,
                clean_indices,
                local_padded.new_empty(
                    (*ids.shape, self.config.vocab.output_size)
                ),
            )
        return ModelOutput(
            logits=logits,
            byte_states=local_padded,
            patch_states=normalized_patches,
            bos_patch_states=bos_patch_states,
        )

    def forward_bos_logits(
        self,
        condition: Tensor,
        *,
        allow_dense_reference: bool = False,
    ) -> Tensor:
        """Predict document-first atoms from their own virtual BOS latents."""

        if condition.ndim != 2 or condition.shape[1] != self.config.global_dim:
            raise ValueError("BOS condition must be [documents, global_dim]")
        batch_size = condition.shape[0]
        device = condition.device
        if batch_size == 0:
            return self.embedding.weight.new_empty(
                (0, self.config.vocab.output_size), device=device
            )
        ids = torch.full(
            (batch_size,), self.config.vocab.eot_id, dtype=torch.long, device=device
        )
        states = self.decoder_embeddings(ids, ModelMode.AR)
        cu = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        positions = torch.zeros(batch_size, dtype=torch.long, device=device)
        for block in self.decoder:
            states = block.forward_packed(
                states,
                condition,
                cu_seqlens=cu,
                positions=positions,
                max_seqlen=1,
                window=1,
                allow_dense_reference=allow_dense_reference,
            )
        states = self.decoder_norm(states)
        if self.output is None:
            return F.linear(
                states, self.embedding.weight[: self.config.vocab.output_size]
            )
        return self.output(states)

    def forward_canvas_branches(
        self,
        clean_ids: Tensor,
        clean_valid: Tensor,
        noisy_ids: Tensor,
        branch_valid: Tensor,
        branch_starts: Tensor,
        *,
        positions: Tensor | None = None,
        assume_full_clean: bool = False,
        document_ids: Tensor | None = None,
    ) -> CanvasBranchOutput:
        """Encode one clean row plus M isolated latent-canvas branches once."""

        if (
            self.require_compiled_training
            and self.training
            and clean_ids.is_cuda
            and not torch.compiler.is_compiling()
        ):
            raise RuntimeError(
                "production canvas training escaped torch.compile before FlexAttention"
            )
        self._validate(clean_ids, clean_valid)
        if (
            assume_full_clean
            and not torch.compiler.is_compiling()
            and not bool(clean_valid.all())
        ):
            raise ValueError("assume_full_clean requires every clean slot to be valid")
        if noisy_ids.ndim != 3 or noisy_ids.shape != branch_valid.shape:
            raise ValueError("noisy_ids and branch_valid must be [batch, branches, canvas]")
        if noisy_ids.dtype != torch.long or branch_valid.dtype != torch.bool:
            raise TypeError("canvas ids must be int64 and validity must be boolean")
        batch, branches, canvas_length = noisy_ids.shape
        if batch != clean_ids.shape[0] or branch_starts.shape != (batch, branches):
            raise ValueError("canvas metadata and clean batch do not align")
        if branch_starts.dtype != torch.long:
            raise TypeError("branch starts must be int64")
        if canvas_length % self.config.patch_stride:
            raise ValueError("canvas length must be patch aligned")
        if not torch.compiler.is_compiling():
            if bool((branch_starts % self.config.patch_stride).any()):
                raise ValueError("canvas branches must start on patch boundaries")
            if bool(
                (
                    (branch_starts < 0)
                    | (branch_starts + canvas_length > clean_ids.shape[1])
                ).any()
            ):
                raise ValueError("canvas branch lies outside its clean row")
            if bool((noisy_ids[~branch_valid] != self.config.vocab.pad_id).any()):
                raise ValueError("invalid branch positions must store PAD")

        clean_length = clean_ids.shape[1]
        if positions is None:
            positions = torch.arange(clean_length, device=clean_ids.device)[None].expand_as(clean_ids)
        if positions.shape != clean_ids.shape:
            raise ValueError("absolute clean positions must align with clean ids")
        if document_ids is not None and (
            document_ids.shape != clean_ids.shape
            or document_ids.dtype != torch.long
        ):
            raise ValueError("canvas document ids must be aligned int64 values")
        if document_ids is not None and not torch.compiler.is_compiling():
            first_document = document_ids[:, :1]
            multiple_documents = clean_valid & document_ids.ne(first_document)
            if bool(multiple_documents.any()):
                raise ValueError(
                    "legacy canvas branches support one document per row; use "
                    "the standalone DiffusionGemma document-canvas path for "
                    "packed multi-document pages"
                )
        branch_documents = (
            None
            if document_ids is None
            else torch.gather(document_ids, 1, branch_starts)
        )
        physical_ids = torch.cat((clean_ids, noisy_ids.flatten(1, 2)), dim=1)
        physical_valid = torch.cat((clean_valid, branch_valid.flatten(1, 2)), dim=1)
        allow_dense_reference = clean_ids.device.type != "cuda"
        local = self.embedding(physical_ids)
        if self.ngrams is not None:
            halo_size = max(self.config.ngram_orders) - 1
            prefix_indices = (
                branch_starts[:, :, None]
                - halo_size
                + torch.arange(halo_size, device=clean_ids.device)[None, None, :]
            )
            prefix_exists = prefix_indices >= 0
            prefix = torch.gather(
                clean_ids[:, None, :].expand(-1, branches, -1),
                2,
                prefix_indices.clamp_min(0),
            )
            prefix = torch.where(prefix_exists, prefix, self.config.vocab.pad_id)
            visible = torch.cat((prefix, noisy_ids), dim=2).flatten(0, 1)
            branch_features = self.ngrams(visible)[:, -canvas_length:].reshape(
                batch, branches * canvas_length, self.config.local_dim
            )
            local = self.combine_ngram_features(
                local,
                torch.cat((self.ngrams(clean_ids), branch_features), dim=1),
            )
        offsets = torch.arange(canvas_length, device=clean_ids.device)
        branch_indices = branch_starts[:, :, None] + offsets[None, None, :]
        branch_positions = torch.gather(
            positions[:, None, :].expand(-1, branches, -1), 2, branch_indices
        )
        physical_positions = torch.cat((positions, branch_positions.flatten(1, 2)), dim=1)
        local_layout = CanvasBranchLayout(
            clean_valid=clean_valid,
            branch_valid=branch_valid,
            prefix_lengths=branch_starts,
            prefix_window=self.config.local_window,
            clean_positions=positions,
            branch_positions=branch_positions,
            clean_segment_ids=document_ids,
            branch_segment_ids=branch_documents,
        )
        if not torch.compiler.is_compiling():
            local_layout.validate_prefix_lengths()
        local_block_mask = (
            None
            if allow_dense_reference
            else build_canvas_block_mask(local_layout)
        )
        decoder_layout = CanvasBranchLayout(
            clean_valid=clean_valid,
            branch_valid=branch_valid,
            prefix_lengths=branch_starts,
            prefix_window=self.config.decoder_prefix_window,
            clean_positions=(
                None if self.config.decoder_prefix_window is None else positions
            ),
            branch_positions=(
                None
                if self.config.decoder_prefix_window is None
                else branch_positions
            ),
            clean_segment_ids=document_ids,
            branch_segment_ids=branch_documents,
        )
        decoder_block_mask = (
            None
            if allow_dense_reference
            else build_canvas_block_mask(decoder_layout)
        )
        for block in self.encoder:
            local = block.forward_clean_and_branches(
                local,
                clean_valid=clean_valid,
                positions=physical_positions,
                layout=local_layout,
                clean_window=self.config.local_window,
                allow_dense_reference=allow_dense_reference,
                assume_full_clean=assume_full_clean,
                block_mask=local_block_mask,
            )
        patches = self.pool(local, physical_valid)

        clean_patches = clean_length // self.config.patch_stride
        canvas_patches = canvas_length // self.config.patch_stride
        clean_patch_valid = clean_valid.view(batch, clean_patches, self.config.patch_stride).any(-1)
        branch_patch_valid = branch_valid.view(
            batch, branches, canvas_patches, self.config.patch_stride
        ).any(-1)
        patch_starts = branch_starts // self.config.patch_stride
        clean_patch_documents = (
            None
            if document_ids is None
            else document_ids.view(
                batch, clean_patches, self.config.patch_stride
            ).amax(-1)
        )
        clean_patch_positions = (
            positions[:, :: self.config.patch_stride] // self.config.patch_stride
        )
        patch_offsets = torch.arange(canvas_patches, device=clean_ids.device)
        branch_patch_indices = patch_starts[:, :, None] + patch_offsets[None, None, :]
        branch_patch_positions = torch.gather(
            clean_patch_positions[:, None, :].expand(-1, branches, -1),
            2,
            branch_patch_indices,
        )
        global_layout = CanvasBranchLayout(
            clean_valid=clean_patch_valid,
            branch_valid=branch_patch_valid,
            prefix_lengths=patch_starts,
            prefix_window=self.config.global_window,
            clean_positions=clean_patch_positions,
            branch_positions=branch_patch_positions,
            clean_segment_ids=clean_patch_documents,
            branch_segment_ids=branch_documents,
        )
        global_block_mask = (
            None
            if allow_dense_reference
            else build_canvas_block_mask(global_layout)
        )
        global_positions = torch.cat(
            (clean_patch_positions, branch_patch_positions.flatten(1, 2)), dim=1
        )
        def global_forward(states: Tensor) -> Tensor:
            for block in self.global_blocks:
                states = block.forward_clean_and_branches(
                    states,
                    clean_valid=clean_patch_valid,
                    positions=global_positions,
                    layout=global_layout,
                    clean_window=self.config.global_window,
                    allow_dense_reference=allow_dense_reference,
                    assume_full_clean=assume_full_clean,
                    block_mask=global_block_mask,
                )
            return states

        if self.activation_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            patches = checkpoint(
                global_forward,
                patches,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            patches = global_forward(patches)
        patches = self.global_norm(patches)

        clean_patch_states = patches[:, :clean_patches]
        branch_patch_states = patches[:, clean_patches:].view(
            batch, branches, canvas_patches, self.config.global_dim
        )
        clean_condition = self._aligned_condition(
            clean_patch_states, clean_length, ModelMode.AR
        )
        branch_condition = branch_patch_states.repeat_interleave(
            self.config.patch_stride, dim=2
        ).flatten(1, 2)
        condition = torch.cat((clean_condition, branch_condition), dim=1)
        mode_ids = torch.full_like(physical_ids, int(ModelMode.CANVAS))
        mode_ids[:, :clean_length] = int(ModelMode.AR)
        states = self.decoder_embeddings(
            physical_ids, ModelMode.CANVAS, mode_ids=mode_ids
        )
        for block in self.decoder:
            states = block.forward_clean_and_branches(
                states,
                condition,
                clean_valid=clean_valid,
                positions=physical_positions,
                layout=decoder_layout,
                clean_window=self.config.decoder_prefix_window,
                allow_dense_reference=allow_dense_reference,
                assume_full_clean=assume_full_clean,
                block_mask=decoder_block_mask,
            )
        states = self.decoder_norm(states)
        logits = (
            F.linear(states, self.embedding.weight[: self.config.vocab.output_size])
            if self.output is None
            else self.output(states)
        )
        return CanvasBranchOutput(
            # The Flex mask builder creates a graph boundary. Materialize both
            # disjoint logit banks before crossing it so AOTAutograd does not
            # have to replay slice/view aliases with corrupted symbolic shapes
            # on Python 3.14 / torch 2.13.
            clean_logits=logits[:, :clean_length].clone(),
            branch_logits=logits[:, clean_length:].reshape(
                batch, branches, canvas_length, self.config.vocab.output_size
            ).clone(),
            clean_patch_states=clean_patch_states,
            branch_patch_states=branch_patch_states,
        )

    def forward_blt_d(
        self,
        clean_ids: Tensor,
        noisy_ids: Tensor,
        valid: Tensor,
        *,
        block_conditions: Tensor,
        block_starts: Tensor | None = None,
        block_length: int | None = None,
        positions: Tensor | None = None,
    ) -> ModelOutput:
        """Fast-BLT reference: clean backbone, noisy fresh decoder inputs."""

        byte_states, patch_states = self.encode(
            clean_ids, valid, bidirectional=False, positions=positions
        )
        if block_starts is None:
            block_starts = torch.zeros(
                (clean_ids.shape[0], 1), dtype=torch.long, device=clean_ids.device
            )
            block_length = clean_ids.shape[1]
        if block_length is None or block_length <= 0 or block_length % self.config.patch_stride:
            raise ValueError("BLT-D block length must be a positive patch multiple")
        if block_starts.dtype != torch.long or block_starts.ndim != 2:
            raise ValueError("block_starts must be [batch, blocks] int64")
        if block_starts.shape[0] != clean_ids.shape[0]:
            raise ValueError("block starts and clean rows must align")
        if block_conditions.shape != (
            clean_ids.shape[0],
            block_starts.shape[1],
            self.config.global_dim,
        ):
            raise ValueError(
                "BLT-D requires one resolved global condition per block"
            )
        if bool((block_starts % self.config.patch_stride).any()):
            raise ValueError("BLT-D blocks must start on patch boundaries")
        batch, length = clean_ids.shape
        if bool(((block_starts < 0) | (block_starts + block_length > length)).any()):
            raise ValueError("BLT-D block lies outside its clean row")
        byte_positions = torch.arange(length, device=clean_ids.device)
        allowed = torch.zeros((batch, length, length), dtype=torch.bool, device=clean_ids.device)
        condition = patch_states.new_zeros((batch, length, self.config.global_dim))
        query_claimed = torch.zeros((batch, length), dtype=torch.bool, device=clean_ids.device)
        for row in range(batch):
            for branch, start_tensor in enumerate(block_starts[row]):
                start = int(start_tensor)
                stop = start + block_length
                if bool(query_claimed[row, start:stop].any()):
                    raise ValueError("BLT-D blocks cannot overlap")
                query_claimed[row, start:stop] = True
                query_positions = byte_positions[start:stop, None]
                prefix_keys = (byte_positions[None, :] < start) & (
                    (
                        self.config.decoder_prefix_window is None
                        or query_positions - byte_positions[None, :]
                        < self.config.decoder_prefix_window
                    )
                )
                own_block = (byte_positions[None, :] >= start) & (byte_positions[None, :] < stop)
                keys = (prefix_keys | own_block) & valid[row, None, :]
                allowed[row, start:stop] = keys & valid[row, start:stop, None]
                condition[row, start:stop] = block_conditions[row, branch]
        logits = self.decode(
            noisy_ids,
            valid,
            patch_states,
            ModelMode.BLT_D,
            positions=positions,
            allowed_override=allowed,
            condition_override=condition,
        )
        return ModelOutput(logits=logits, byte_states=byte_states, patch_states=patch_states)

    def forward_blt_d_branches(
        self,
        clean_ids: Tensor,
        clean_valid: Tensor,
        noisy_blocks: Tensor,
        branch_valid: Tensor,
        block_starts: Tensor,
        *,
        positions: Tensor | None = None,
        assume_full_clean: bool = False,
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
        return_clean_patch_states: bool = True,
        patch_byte_cu_seqlens: Tensor | None = None,
        max_patch_size: int | None = None,
    ) -> BltBranchOutput:
        """Fast-BLT sampled blocks with one shared clean encoder/decoder bank."""

        if (
            self.require_compiled_training
            and self.training
            and clean_ids.is_cuda
            and not torch.compiler.is_compiling()
        ):
            raise RuntimeError(
                "production BLT-D training escaped torch.compile before FlexAttention"
            )
        if noisy_blocks.ndim != 3 or noisy_blocks.shape != branch_valid.shape:
            raise ValueError("noisy BLT blocks and validity must be [batch, blocks, length]")
        batch, branches, block_length = noisy_blocks.shape
        variable_patches = patch_byte_cu_seqlens is not None
        if block_length <= 0 or (
            not variable_patches and block_length % self.config.patch_stride
        ):
            raise ValueError("BLT block length must be patch aligned")
        if block_starts.shape != (batch, branches) or block_starts.dtype != torch.long:
            raise ValueError("one int64 block start is required per sampled block")
        if (
            branch_condition_indices is None
            or branch_condition_indices.shape != block_starts.shape
            or branch_condition_indices.dtype != torch.long
        ):
            raise ValueError("BLT blocks require explicit int64 prior conditions")
        if not torch.compiler.is_compiling():
            if not variable_patches and bool(
                (block_starts % self.config.patch_stride).any()
            ):
                raise ValueError("BLT blocks must start on patch boundaries")
            if bool(((block_starts < 0) | (block_starts >= clean_ids.shape[1])).any()):
                raise ValueError("BLT block start lies outside the clean row")
            if variable_patches:
                if (
                    byte_indices is None
                    or patch_byte_cu_seqlens is None
                    or patch_cu_seqlens is None
                    or physical_to_global_patch_indices is None
                ):
                    raise ValueError(
                        "variable BLT starts require packed byte and patch metadata"
                    )
                packed_patch_starts = patch_byte_cu_seqlens[:-1].to(torch.long)
                physical_flat = byte_indices.index_select(
                    0, packed_patch_starts
                )
                physical_rows = torch.div(
                    physical_flat,
                    clean_ids.shape[1],
                    rounding_mode="floor",
                )
                physical_columns = physical_flat.remainder(clean_ids.shape[1])
                global_segment_starts = torch.repeat_interleave(
                    patch_cu_seqlens[:-1].to(torch.long),
                    torch.diff(patch_cu_seqlens).to(torch.long),
                )
                physical_global = physical_to_global_patch_indices.to(torch.long)
                physical_segment_starts = global_segment_starts.index_select(
                    0, physical_global
                )
                physical_priors = torch.where(
                    physical_global.gt(physical_segment_starts),
                    physical_global - 1,
                    torch.full_like(physical_global, -1),
                )
                requested_rows = torch.arange(
                    batch, device=block_starts.device
                )[:, None, None]
                matches = physical_rows[None, None].eq(requested_rows) & (
                    physical_columns[None, None] == block_starts[:, :, None]
                )
                active = branch_valid.any(-1)
                found = matches.any(-1)
                if bool((active & ~found).any()):
                    raise ValueError(
                        "variable BLT block start is not an authenticated physical patch"
                    )
                selected = matches.to(torch.long).argmax(-1)
                expected_priors = physical_priors.index_select(
                    0, selected.reshape(-1)
                ).view_as(block_starts)
                if bool(
                    (
                        active
                        & branch_condition_indices.ne(expected_priors)
                    ).any()
                ):
                    raise ValueError(
                        "variable BLT branch condition does not match its patch origin"
                    )
        if positions is None:
            positions = torch.arange(clean_ids.shape[1], device=clean_ids.device)[None].expand_as(clean_ids)
        allow_dense_reference = clean_ids.device.type != "cuda"
        (
            _,
            patch_states,
            _,
            _,
            _,
            clean_indices,
            packed_patch_states,
        ) = self._encode_clean_varlen(
            clean_ids,
            clean_valid,
            positions=positions,
            allow_dense_reference=allow_dense_reference,
            assume_full_clean=assume_full_clean,
            document_ids=document_ids,
            byte_indices=byte_indices,
            byte_cu_seqlens=byte_cu_seqlens,
            patch_indices_override=patch_indices,
            patch_cu_seqlens=patch_cu_seqlens,
            global_patch_sources=global_patch_sources,
            global_patch_positions=global_patch_positions,
            physical_to_global_patch_indices=physical_to_global_patch_indices,
            patch_byte_cu_seqlens=patch_byte_cu_seqlens,
            max_patch_size=max_patch_size,
            materialize_padded_patches=return_clean_patch_states,
        )
        clean_length = clean_ids.shape[1]
        branch_ids = noisy_blocks.flatten(1, 2)
        offsets = torch.arange(block_length, device=clean_ids.device)
        start_positions = torch.gather(positions, 1, block_starts)
        branch_positions = start_positions[:, :, None] + offsets[None, None, :]
        physical_positions = torch.cat((positions, branch_positions.flatten(1, 2)), dim=1)
        document_packed = document_ids is not None
        shared_document_attention = (
            document_packed
            and self.config.decoder_branch_attention == "shared_flex"
        )
        projected_shared_conditioning = (
            shared_document_attention
            and self.config.decoder_conditioning != "split_cross_attention"
        )
        if document_packed:
            if (
                byte_indices is None
                or byte_cu_seqlens is None
                or condition_patch_indices is None
            ):
                raise ValueError("document-packed BLT-D requires packed clean metadata")
            if not projected_shared_conditioning:
                selected_condition = packed_patch_states.index_select(
                    0, condition_patch_indices.clamp_min(0)
                )
                selected_condition = torch.where(
                    condition_patch_indices[:, None].ge(0), selected_condition, 0
                )
                clean_condition = patch_states.new_zeros(
                    (batch, clean_length, self.config.global_dim)
                )
                clean_condition.reshape(-1, self.config.global_dim).index_copy_(
                    0, byte_indices, selected_condition
                )
            else:
                clean_condition = None
        else:
            clean_condition = self._aligned_condition(
                patch_states, clean_length, ModelMode.AR
            )
        prior_states = None
        if document_packed and not projected_shared_conditioning:
            prior_states = packed_patch_states.index_select(
                0, branch_condition_indices.clamp_min(0).reshape(-1)
            ).view(batch, branches, self.config.global_dim)
        elif not document_packed:
            gather_index = branch_condition_indices.clamp_min(0)[..., None].expand(
                -1, -1, self.config.global_dim
            )
            prior_states = torch.gather(patch_states, 1, gather_index)
        if prior_states is not None:
            prior_states = torch.where(
                branch_condition_indices[..., None].ge(0),
                prior_states,
                torch.zeros_like(prior_states),
            )
        condition = None
        if not projected_shared_conditioning:
            if clean_condition is None or prior_states is None:
                raise AssertionError("decoder clean condition disappeared")
            branch_condition = prior_states[:, :, None, :].expand(
                -1, -1, block_length, -1
            )
            condition = torch.cat(
                (clean_condition, branch_condition.flatten(1, 2)), dim=1
        )
        if shared_document_attention:
            clean_states = self.decoder_embeddings(clean_ids, ModelMode.AR)
            branch_states = self.decoder_embeddings(branch_ids, ModelMode.BLT_D)
            states = torch.cat((clean_states, branch_states), dim=1)
        else:
            physical_ids = torch.cat((clean_ids, branch_ids), dim=1)
            mode_ids = torch.full_like(physical_ids, int(ModelMode.BLT_D))
            mode_ids[:, :clean_length] = int(ModelMode.AR)
            states = self.decoder_embeddings(
                physical_ids, ModelMode.BLT_D, mode_ids=mode_ids
            )
        if document_packed:
            if document_ids is None or byte_cu_seqlens is None:
                raise AssertionError("document-packed BLT-D metadata disappeared")
            branch_documents = torch.gather(document_ids, 1, block_starts)
            layout = CanvasBranchLayout(
                clean_valid=clean_valid,
                branch_valid=branch_valid,
                prefix_lengths=block_starts,
                prefix_window=self.config.decoder_prefix_window,
                clean_positions=positions,
                branch_positions=branch_positions,
                clean_segment_ids=document_ids,
                branch_segment_ids=branch_documents,
            )
            if not torch.compiler.is_compiling():
                layout.validate_prefix_lengths()
            if (
                self.config.decoder_branch_attention == "shared_flex"
                and not allow_dense_reference
                and branch_block_mask is None
            ):
                raise ValueError(
                    "shared document branch attention requires a prebuilt BlockMask"
                )
            packed_branch_metadata = (
                branch_query_indices,
                branch_kv_indices,
                branch_query_cu_seqlens,
                branch_kv_cu_seqlens,
            )
            prepacked_branches = not any(
                value is None for value in packed_branch_metadata
            )
            if (
                self.config.decoder_branch_attention == "duplicated_varlen"
                and not prepacked_branches
                and clean_ids.is_cuda
            ):
                raise ValueError(
                    "document-packed BLT-D requires precomputed branch attention metadata"
                )
            for block in self.decoder:
                if self.config.decoder_branch_attention == "shared_flex":
                    if not projected_shared_conditioning:
                        if condition is None:
                            raise AssertionError(
                                "shared decoder condition disappeared"
                            )
                        states = block.forward_shared_document_branches(
                            states,
                            condition,
                            clean_length=clean_length,
                            clean_indices=clean_indices,
                            clean_cu_seqlens=byte_cu_seqlens,
                            positions=physical_positions,
                            layout=layout,
                            clean_window=self.config.decoder_prefix_window,
                            allow_dense_reference=allow_dense_reference,
                            block_mask=branch_block_mask,
                        )
                        continue
                    # One global patch state conditions four neighboring
                    # bytes. Project each unique 512D patch/origin state once
                    # per decoder layer, then gather/repeat the 256D result.
                    # This removes the dominant redundant conditioning GEMM.
                    projected_patches = block.condition(packed_patch_states)
                    selected_projected = projected_patches.index_select(
                        0, condition_patch_indices.clamp_min(0)
                    )
                    selected_projected = torch.where(
                        condition_patch_indices[:, None].ge(0),
                        selected_projected,
                        0,
                    )
                    clean_projected = projected_patches.new_zeros(
                        (batch, clean_length, self.config.local_dim)
                    )
                    clean_projected.reshape(
                        -1, self.config.local_dim
                    ).index_copy_(0, byte_indices, selected_projected)
                    branch_projected = projected_patches.index_select(
                        0, branch_condition_indices.clamp_min(0).reshape(-1)
                    ).view(batch, branches, self.config.local_dim)
                    branch_projected = torch.where(
                        branch_condition_indices[..., None].ge(0),
                        branch_projected,
                        0,
                    )[:, :, None, :].expand(-1, -1, block_length, -1)
                    projected_condition = torch.cat(
                        (
                            clean_projected,
                            branch_projected.flatten(1, 2),
                        ),
                        dim=1,
                    )
                    states = block.forward_shared_document_branches_projected(
                        states,
                        projected_condition,
                        clean_length=clean_length,
                        clean_indices=clean_indices,
                        clean_cu_seqlens=byte_cu_seqlens,
                        positions=physical_positions,
                        layout=layout,
                        clean_window=self.config.decoder_prefix_window,
                        allow_dense_reference=allow_dense_reference,
                        block_mask=branch_block_mask,
                    )
                elif prepacked_branches:
                    if condition is None:
                        raise AssertionError("packed decoder condition disappeared")
                    states = block.forward_packed_document_branches(
                        states,
                        condition,
                        clean_length=clean_length,
                        clean_indices=clean_indices,
                        clean_cu_seqlens=byte_cu_seqlens,
                        positions=physical_positions,
                        branch_query_indices=branch_query_indices,
                        branch_kv_indices=branch_kv_indices,
                        branch_query_cu_seqlens=branch_query_cu_seqlens,
                        branch_kv_cu_seqlens=branch_kv_cu_seqlens,
                        max_branch_query_length=block_length,
                        max_branch_kv_length=(
                            clean_length
                            if self.config.decoder_prefix_window is None
                            else self.config.decoder_prefix_window
                        )
                        + block_length,
                        clean_window=self.config.decoder_prefix_window,
                        allow_dense_reference=allow_dense_reference,
                    )
                else:
                    if condition is None:
                        raise AssertionError("decoder condition disappeared")
                    states = block.forward_document_branches(
                        states,
                        condition,
                        clean_valid=clean_valid,
                        clean_indices=clean_indices,
                        clean_cu_seqlens=byte_cu_seqlens,
                        clean_document_ids=document_ids,
                        positions=physical_positions,
                        branch_valid=branch_valid,
                        branch_starts=block_starts,
                        clean_window=self.config.decoder_prefix_window,
                        allow_dense_reference=True,
                    )
        else:
            layout = CanvasBranchLayout(
                clean_valid=clean_valid,
                branch_valid=branch_valid,
                prefix_lengths=block_starts,
                prefix_window=self.config.decoder_prefix_window,
                clean_positions=(
                    None
                    if self.config.decoder_prefix_window is None
                    else positions
                ),
                branch_positions=(
                    None
                    if self.config.decoder_prefix_window is None
                    else branch_positions
                ),
            )
            if not torch.compiler.is_compiling():
                layout.validate_prefix_lengths()
            block_mask = (
                None if allow_dense_reference else build_canvas_block_mask(layout)
            )
            for block in self.decoder:
                states = block.forward_clean_and_branches(
                    states,
                    condition,
                    clean_valid=clean_valid,
                    positions=physical_positions,
                    layout=layout,
                    clean_window=self.config.decoder_prefix_window,
                    allow_dense_reference=allow_dense_reference,
                    assume_full_clean=assume_full_clean,
                    block_mask=block_mask,
                )
        if shared_document_attention:
            clean_states = self.decoder_norm(states[:, :clean_length])
            branch_states = self.decoder_norm(states[:, clean_length:])
            if self.output is None:
                output_weight = self.embedding.weight[: self.config.vocab.output_size]
                clean_logits = F.linear(clean_states, output_weight)
                branch_logits = F.linear(branch_states, output_weight)
            else:
                clean_logits = self.output(clean_states)
                branch_logits = self.output(branch_states)
        else:
            states = self.decoder_norm(states)
            logits = (
                F.linear(states, self.embedding.weight[: self.config.vocab.output_size])
                if self.output is None
                else self.output(states)
            )
            clean_logits = logits[:, :clean_length]
            branch_logits = logits[:, clean_length:]
        return BltBranchOutput(
            clean_logits=clean_logits,
            branch_logits=branch_logits.reshape(
                batch, branches, block_length, self.config.vocab.output_size
            ),
            clean_patch_states=patch_states,
            bos_patch_states=(
                None
                if bos_condition_indices is None
                else packed_patch_states.index_select(0, bos_condition_indices)
            ),
        )
