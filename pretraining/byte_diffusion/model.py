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
    unpack_valid,
)


@dataclass(frozen=True)
class ModelOutput:
    logits: Tensor
    byte_states: Tensor
    patch_states: Tensor


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


class CausalNgramEmbedding(nn.Module):
    """Factorized visible-id n-grams ending at each absolute byte position."""

    def __init__(self, config: ByteDiffusionConfig) -> None:
        super().__init__()
        self.orders = config.ngram_orders
        self.table_size = config.ngram_table_size
        self.eot_id = config.vocab.eot_id
        self.pad_id = config.vocab.pad_id
        self.table = nn.Embedding(config.ngram_table_size, config.ngram_rank)
        self.projections = nn.ModuleList(
            nn.Linear(config.ngram_rank, config.local_dim, bias=False)
            for _ in self.orders
        )

    def forward(self, ids: Tensor) -> Tensor:
        batch, length = ids.shape
        result = self.table.weight.new_zeros((batch, length, self.projections[0].out_features))
        modulus = self.table_size
        for order, projection in zip(self.orders, self.projections, strict=True):
            rolling = torch.zeros_like(ids)
            context_ok = torch.ones_like(ids, dtype=torch.bool)
            for offset in range(order):
                shifted = torch.zeros_like(ids)
                if offset == 0:
                    shifted = ids
                elif offset < length:
                    shifted[:, offset:] = ids[:, :-offset]
                rolling = (rolling * 257 + shifted + 1) % modulus
                if offset:
                    context_ok &= (shifted != self.eot_id) & (shifted != self.pad_id)
            enough_context = torch.arange(length, device=ids.device) >= order - 1
            features = projection(self.table(rolling))
            active = context_ok & enough_context[None, :]
            result = result + features * active[:, :, None]
        return result


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
            )
            for _ in range(config.decoder_layers)
        )
        self.decoder_norm = RMSNorm(config.local_dim)
        self.mode_embedding = nn.Embedding(len(ModelMode), config.local_dim)
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
            self.embedding.weight[self.config.vocab.pad_id].zero_()
            if self.timestep_vector is not None:
                nn.init.normal_(self.timestep_vector, mean=0.0, std=0.02)

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
        return torch.cat(
            (torch.zeros(1, dtype=torch.int32, device=lengths.device), lengths.cumsum(0).to(torch.int32))
        )

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
        local = self.embedding(ids)
        if self.ngrams is not None:
            local = local + self.ngrams(ids)
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
        if mode_ids_override is None:
            states = self.embedding(ids) + self.mode_embedding.weight[int(mode)]
        else:
            if mode_ids_override.shape != ids.shape or mode_ids_override.dtype != torch.long:
                raise ValueError("mode_ids_override must be aligned int64 mode ids")
            states = self.embedding(ids) + self.mode_embedding(mode_ids_override)
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        self._validate(ids, valid)
        if (
            assume_full_clean
            and not torch.compiler.is_compiling()
            and not bool(valid.all())
        ):
            raise ValueError("assume_full_clean requires every clean slot to be valid")
        lengths = self._prefix_lengths(valid)
        if not torch.compiler.is_compiling() and bool((lengths == 0).any()):
            raise ValueError("varlen rows must be nonempty")
        if positions is None:
            positions = torch.arange(ids.shape[1], device=ids.device)[None].expand_as(ids)
        if positions.shape != ids.shape:
            raise ValueError("absolute positions must align with ids")
        cu = self._cu_seqlens(lengths)
        max_length = ids.shape[1]
        padded_local = self.embedding(ids)
        if self.ngrams is not None:
            padded_local = padded_local + self.ngrams(ids)
        if assume_full_clean:
            lengths = torch.full_like(lengths, ids.shape[1])
            cu = self._cu_seqlens(lengths)
            local = padded_local.reshape(-1, self.config.local_dim)
            packed_positions = positions.reshape(-1)
        else:
            local = pack_valid(padded_local, valid)
            packed_positions = pack_valid(positions, valid)
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
            local_padded = unpack_valid(local, valid, padded_local)
        patches = self.pool(local_padded, valid)
        patch_valid = valid.view(valid.shape[0], -1, self.config.patch_stride).any(-1)
        patch_lengths = patch_valid.sum(1)
        patch_cu = self._cu_seqlens(patch_lengths)
        packed_patches = (
            patches.reshape(-1, self.config.global_dim)
            if assume_full_clean
            else pack_valid(patches, patch_valid)
        )
        # Global RoPE is expressed in patch units everywhere.  Using raw byte
        # offsets here would make the clean bank rotate four times faster than
        # canvas/BLT branches despite representing the same physical patches.
        patch_positions_padded = (
            positions[:, :: self.config.patch_stride]
            // self.config.patch_stride
        )
        packed_patch_positions = (
            patch_positions_padded.reshape(-1)
            if assume_full_clean
            else pack_valid(patch_positions_padded, patch_valid)
        )
        if assume_full_clean:
            patch_lengths = torch.full_like(patch_lengths, patches.shape[1])
            patch_cu = self._cu_seqlens(patch_lengths)
        for block in self.global_blocks:
            packed_patches = block.forward_packed(
                packed_patches,
                cu_seqlens=patch_cu,
                positions=packed_patch_positions,
                max_seqlen=patches.shape[1],
                window=None,
                allow_dense_reference=allow_dense_reference,
            )
        if assume_full_clean:
            normalized_patches = self.global_norm(packed_patches).view_as(patches)
        else:
            normalized_patches = unpack_valid(
                self.global_norm(packed_patches), patch_valid, patches
            )
        return local_padded, normalized_patches, cu, lengths, packed_positions

    def forward_ar_varlen(
        self,
        ids: Tensor,
        valid: Tensor,
        *,
        positions: Tensor | None = None,
        allow_dense_reference: bool = False,
        assume_full_clean: bool = False,
    ) -> ModelOutput:
        """Production PAD-free native-varlen Flash path for clean AR rows."""

        if positions is None:
            positions = torch.arange(ids.shape[1], device=ids.device)[None].expand_as(ids)
        local_padded, normalized_patches, cu, lengths, packed_positions = (
            self._encode_clean_varlen(
                ids,
                valid,
                positions=positions,
                allow_dense_reference=allow_dense_reference,
                assume_full_clean=assume_full_clean,
            )
        )

        condition = self._aligned_condition(normalized_patches, ids.shape[1], ModelMode.AR)
        padded_states = self.embedding(ids) + self.mode_embedding.weight[int(ModelMode.AR)]
        states = (
            padded_states.reshape(-1, self.config.local_dim)
            if assume_full_clean
            else pack_valid(padded_states, valid)
        )
        packed_condition = (
            condition.reshape(-1, self.config.global_dim)
            if assume_full_clean
            else pack_valid(condition, valid)
        )
        for block in self.decoder:
            states = block.forward_packed(
                states,
                packed_condition,
                cu_seqlens=cu,
                positions=packed_positions,
                max_seqlen=ids.shape[1],
                window=self.config.local_window,
                allow_dense_reference=allow_dense_reference,
            )
        logits = local_padded.new_zeros(
            (*ids.shape, self.config.vocab.output_size)
        )
        normalized_states = self.decoder_norm(states)
        if self.output is None:
            packed_logits = F.linear(
                normalized_states, self.embedding.weight[: self.config.vocab.output_size]
            )
        else:
            packed_logits = self.output(normalized_states)
        if assume_full_clean:
            logits = packed_logits.view(*ids.shape, self.config.vocab.output_size)
        else:
            logits = unpack_valid(packed_logits, valid, logits)
        return ModelOutput(
            logits=logits,
            byte_states=local_padded,
            patch_states=normalized_patches,
        )

    def forward_bos_logits(
        self,
        batch_size: int,
        *,
        device: torch.device,
        allow_dense_reference: bool = False,
    ) -> Tensor:
        """Predict each document's first atom from a synthetic EOT/BOS.

        The virtual BOS is deliberately outside the fixed four-byte patch
        stream, so real byte zero remains at patch offset zero.  At this first
        AR position global conditioning is exactly zero; consequently only the
        shared decoder needs to run.  Packing the ``batch_size`` one-token
        sequences keeps CUDA on the native varlen Flash path.
        """

        if batch_size <= 0:
            raise ValueError("BOS batch size must be positive")
        ids = torch.full(
            (batch_size,), self.config.vocab.eot_id, dtype=torch.long, device=device
        )
        states = self.embedding(ids) + self.mode_embedding.weight[int(ModelMode.AR)]
        condition = states.new_zeros((batch_size, self.config.global_dim))
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
            local = local + torch.cat((self.ngrams(clean_ids), branch_features), dim=1)
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
        )
        if not torch.compiler.is_compiling():
            local_layout.validate_prefix_lengths()
        local_block_mask = (
            None
            if allow_dense_reference
            else build_canvas_block_mask(local_layout)
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
        global_layout = CanvasBranchLayout(
            clean_valid=clean_patch_valid,
            branch_valid=branch_patch_valid,
            prefix_lengths=patch_starts,
        )
        global_block_mask = (
            None
            if allow_dense_reference
            else build_canvas_block_mask(global_layout)
        )
        clean_patch_positions = positions[:, :: self.config.patch_stride] // self.config.patch_stride
        patch_offsets = torch.arange(canvas_patches, device=clean_ids.device)
        branch_patch_indices = patch_starts[:, :, None] + patch_offsets[None, None, :]
        branch_patch_positions = torch.gather(
            clean_patch_positions[:, None, :].expand(-1, branches, -1),
            2,
            branch_patch_indices,
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
                    clean_window=None,
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
        states = self.embedding(physical_ids) + self.mode_embedding(mode_ids)
        for block in self.decoder:
            states = block.forward_clean_and_branches(
                states,
                condition,
                clean_valid=clean_valid,
                positions=physical_positions,
                layout=local_layout,
                clean_window=self.config.local_window,
                allow_dense_reference=allow_dense_reference,
                assume_full_clean=assume_full_clean,
                block_mask=local_block_mask,
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
            for start_tensor in block_starts[row]:
                start = int(start_tensor)
                stop = start + block_length
                if bool(query_claimed[row, start:stop].any()):
                    raise ValueError("BLT-D blocks cannot overlap")
                query_claimed[row, start:stop] = True
                query_positions = byte_positions[start:stop, None]
                prefix_keys = (byte_positions[None, :] < start) & (
                    query_positions - byte_positions[None, :] < self.config.local_window
                )
                own_block = (byte_positions[None, :] >= start) & (byte_positions[None, :] < stop)
                keys = (prefix_keys | own_block) & valid[row, None, :]
                allowed[row, start:stop] = keys & valid[row, start:stop, None]
                prior_patch = start // self.config.patch_stride - 1
                if prior_patch >= 0:
                    condition[row, start:stop] = patch_states[row, prior_patch]
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
        if block_length % self.config.patch_stride:
            raise ValueError("BLT block length must be patch aligned")
        if block_starts.shape != (batch, branches) or block_starts.dtype != torch.long:
            raise ValueError("one int64 block start is required per sampled block")
        if not torch.compiler.is_compiling():
            if bool((block_starts % self.config.patch_stride).any()):
                raise ValueError("BLT blocks must start on patch boundaries")
            if bool(((block_starts < 0) | (block_starts >= clean_ids.shape[1])).any()):
                raise ValueError("BLT block start lies outside the clean row")
        if positions is None:
            positions = torch.arange(clean_ids.shape[1], device=clean_ids.device)[None].expand_as(clean_ids)
        allow_dense_reference = clean_ids.device.type != "cuda"
        _, patch_states, _, _, _ = self._encode_clean_varlen(
            clean_ids,
            clean_valid,
            positions=positions,
            allow_dense_reference=allow_dense_reference,
            assume_full_clean=assume_full_clean,
        )
        clean_length = clean_ids.shape[1]
        physical_ids = torch.cat((clean_ids, noisy_blocks.flatten(1, 2)), dim=1)
        offsets = torch.arange(block_length, device=clean_ids.device)
        branch_positions = block_starts[:, :, None] + offsets[None, None, :]
        physical_positions = torch.cat((positions, branch_positions.flatten(1, 2)), dim=1)
        layout = CanvasBranchLayout(
            clean_valid=clean_valid,
            branch_valid=branch_valid,
            prefix_lengths=block_starts,
            prefix_window=self.config.local_window,
            clean_positions=positions,
            branch_positions=branch_positions,
        )
        if not torch.compiler.is_compiling():
            layout.validate_prefix_lengths()
        block_mask = (
            None if allow_dense_reference else build_canvas_block_mask(layout)
        )
        clean_condition = self._aligned_condition(patch_states, clean_length, ModelMode.AR)
        prior_patch = block_starts // self.config.patch_stride - 1
        gather_index = prior_patch.clamp_min(0)[..., None].expand(
            -1, -1, self.config.global_dim
        )
        prior_states = torch.gather(patch_states, 1, gather_index)
        prior_states = torch.where(
            (prior_patch >= 0)[..., None], prior_states, torch.zeros_like(prior_states)
        )
        branch_condition = prior_states[:, :, None, :].expand(
            -1, -1, block_length, -1
        )
        condition = torch.cat((clean_condition, branch_condition.flatten(1, 2)), dim=1)
        mode_ids = torch.full_like(physical_ids, int(ModelMode.BLT_D))
        mode_ids[:, :clean_length] = int(ModelMode.AR)
        states = self.embedding(physical_ids) + self.mode_embedding(mode_ids)
        for block in self.decoder:
            states = block.forward_clean_and_branches(
                states,
                condition,
                clean_valid=clean_valid,
                positions=physical_positions,
                layout=layout,
                clean_window=self.config.local_window,
                allow_dense_reference=allow_dense_reference,
                assume_full_clean=assume_full_clean,
                block_mask=block_mask,
            )
        states = self.decoder_norm(states)
        logits = (
            F.linear(states, self.embedding.weight[: self.config.vocab.output_size])
            if self.output is None
            else self.output(states)
        )
        return BltBranchOutput(
            clean_logits=logits[:, :clean_length].clone(),
            branch_logits=logits[:, clean_length:].reshape(
                batch, branches, block_length, self.config.vocab.output_size
            ).clone(),
            clean_patch_states=patch_states,
        )
