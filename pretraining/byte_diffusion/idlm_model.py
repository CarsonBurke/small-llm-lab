"""Standalone reference-aligned Introspective Diffusion LM.

This cell is intentionally separate from :mod:`model`.  It is a shared flat
Transformer with one input cue for the proposal pathway: every proposal id is
the atomic ``MASK`` id.  There are no mode, timestep, or self-conditioning
embeddings.  Clean and proposal states share every learned parameter.

The reference cell uses ``clean_prefix_window=None``.  It implements the full
prefix mask specified by I-DLM: clean queries are ordinarily causal, while a
proposal query reads causal proposal states in its current block and clean
states only from strictly earlier blocks.  ``blt_window512`` is a named byte
adaptation which limits visible clean history to 512 atomic ids.  It is not
silently treated as the paper mask.  Both cells have production
Varlen-Flash/Flex lowerings and dense correctness oracles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import (
    IntrospectionBranchLayout,
    PackedCleanQKV,
    branch_attention,
    packed_clean_attention,
)
from .idlm import IDLMTrainingLayout, strict_attention_mask
from .layers import RMSNorm, TransformerBlock, apply_rotary, pack_rows, unpack_rows


PhysicalOrder = Literal["clean_proposal", "proposal_clean"]


@dataclass(frozen=True)
class IDLMModelConfig:
    """Closed architecture for the standalone I-DLM ablation.

    The default has 23,017,984 trainable parameters, only 6,910 above the
    byte-family 23,011,074 reference target and safely below the 16 MiB
    challenge artifact budget after the same quantization assumptions.
    """

    input_size: int = 263
    output_size: int = 261
    mask_id: int = 261
    pad_id: int = 262
    eot_id: int = 256
    dim: int = 512
    heads: int = 8
    layers: int = 8
    ffn_dim: int = 1_168
    rope_theta: float = 500_000.0
    block_size: int = 4
    clean_prefix_window: int | None = None

    def __post_init__(self) -> None:
        positive = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "dim": self.dim,
            "heads": self.heads,
            "layers": self.layers,
            "ffn_dim": self.ffn_dim,
            "block_size": self.block_size,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"I-DLM architecture values must be positive: {positive}")
        if self.dim % self.heads:
            raise ValueError("I-DLM width must divide its attention heads")
        if self.mask_id != self.output_size or self.pad_id != self.mask_id + 1:
            raise ValueError("I-DLM MASK/PAD ids must follow the output vocabulary")
        if self.input_size != self.pad_id + 1:
            raise ValueError("I-DLM input vocabulary must end at PAD")
        if not 0 <= self.eot_id < self.output_size:
            raise ValueError("I-DLM EOT must be a predictable atomic id")
        if self.clean_prefix_window is not None and self.clean_prefix_window <= 0:
            raise ValueError("clean_prefix_window must be positive or None")

    @classmethod
    def tiny(cls, **overrides: object) -> "IDLMModelConfig":
        values: dict[str, object] = {
            "dim": 32,
            "heads": 1,
            "layers": 2,
            "ffn_dim": 48,
            **overrides,
        }
        return cls(**values)

    @classmethod
    def blt_window512(cls, **overrides: object) -> "IDLMModelConfig":
        """Return the explicitly non-reference 512-byte-prefix adaptation."""

        return cls(clean_prefix_window=512, **overrides)


@dataclass(frozen=True)
class IDLMForwardMetadata:
    """Precomputed document packing and optional Flex mask for one batch."""

    clean_indices: Tensor
    clean_cu_seqlens: Tensor
    max_clean_seqlen: int
    block_mask: object | None = None

    def __post_init__(self) -> None:
        if self.clean_indices.dtype != torch.long or self.clean_indices.ndim != 1:
            raise ValueError("clean_indices must be rank-1 int64")
        if (
            self.clean_cu_seqlens.dtype != torch.int32
            or self.clean_cu_seqlens.ndim != 1
        ):
            raise ValueError("clean_cu_seqlens must be rank-1 int32")
        if self.max_clean_seqlen <= 0:
            raise ValueError("max_clean_seqlen must be positive")


@dataclass(frozen=True)
class IDLMModelOutput:
    proposal_logits: Tensor
    clean_logits: Tensor
    bos_logits: Tensor


def _same_segment(layout: IDLMTrainingLayout) -> Tensor:
    if layout.segment_ids is None:
        return torch.ones(
            (layout.batch_size, layout.sequence_length, layout.sequence_length),
            dtype=torch.bool,
            device=layout.clean_ids.device,
        )
    return layout.segment_ids[:, :, None].eq(layout.segment_ids[:, None, :])


def idlm_dense_attention_mask(
    layout: IDLMTrainingLayout,
    *,
    clean_prefix_window: int | None,
) -> Tensor:
    """Return the exact mask, optionally limiting only clean-prefix history."""

    allowed = strict_attention_mask(layout)
    if clean_prefix_window is None:
        return allowed
    length = layout.sequence_length
    query_positions = layout.positions[:, :, None]
    key_positions = layout.positions[:, None, :]
    within_window = key_positions > query_positions - clean_prefix_window
    same_segment = _same_segment(layout)

    # In [proposal | clean] order, only keys from the clean half are windowed.
    # Proposal-to-proposal current-block causality is unchanged.
    allowed[:, :length, length:] &= within_window & same_segment
    allowed[:, length:, length:] &= within_window & same_segment
    return allowed


def _packed_document_metadata(layout: IDLMTrainingLayout) -> IDLMForwardMetadata:
    """Build segment offsets with tensor operations, never a per-row scan."""

    flat_valid = layout.valid.reshape(-1)
    clean_indices = flat_valid.nonzero(as_tuple=False).flatten()
    flat_rows = torch.div(
        clean_indices, layout.sequence_length, rounding_mode="floor"
    )
    if layout.segment_ids is None:
        flat_segments = flat_rows
    else:
        flat_segments = layout.segment_ids.reshape(-1).index_select(0, clean_indices)
    boundary = torch.ones_like(clean_indices, dtype=torch.bool)
    boundary[1:] = (flat_rows[1:] != flat_rows[:-1]) | (
        flat_segments[1:] != flat_segments[:-1]
    )
    starts = boundary.nonzero(as_tuple=False).flatten()
    stops = torch.cat((starts[1:], starts.new_tensor([clean_indices.numel()])))
    lengths = stops - starts
    clean_cu = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=clean_indices.device),
            lengths.to(torch.int32).cumsum(0, dtype=torch.int32),
        )
    )
    # ``max_clean_seqlen`` is a launch bound, not semantic metadata.  The
    # physical row width is exact, safely bounds every packed document, and
    # stays constant across batches for stable CUDA graph compilation.
    return IDLMForwardMetadata(clean_indices, clean_cu, layout.sequence_length)


@torch.compiler.disable
def _build_idlm_introspection_block_mask(
    layout: IDLMTrainingLayout,
    *,
    clean_prefix_window: int | None,
    flex_block_size: int = 128,
):
    """Lower the document-relative I-DLM oracle to an exact Flex mask."""

    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except Exception as error:
        raise RuntimeError("FlexAttention is unavailable") from error
    length = layout.sequence_length
    proposal_block_size = layout.block_size

    def mask_mod(batch: Tensor, head: Tensor, query: Tensor, key: Tensor) -> Tensor:
        del head
        clean_key = key < length
        key_offset = torch.where(clean_key, key, key - length).clamp(0, length - 1)
        query_offset = query.clamp(0, length - 1)
        query_block = torch.div(
            layout.positions[batch, query_offset],
            proposal_block_size,
            rounding_mode="floor",
        )
        key_position = layout.positions[batch, key_offset]
        query_position = layout.positions[batch, query_offset]
        key_block = torch.div(
            key_position, proposal_block_size, rounding_mode="floor"
        )
        query_valid = layout.valid[batch, query_offset]
        same_segment = torch.ones_like(query_valid)
        if layout.segment_ids is not None:
            same_segment = layout.segment_ids[batch, key_offset].eq(
                layout.segment_ids[batch, query_offset]
            )
        from_clean = (
            clean_key
            & layout.valid[batch, key_offset]
            & same_segment
            & (key_block < query_block)
        )
        if clean_prefix_window is not None:
            from_clean = from_clean & (
                key_position > query_position - clean_prefix_window
            )
        from_proposal = (
            ~clean_key
            & layout.valid[batch, key_offset]
            & same_segment
            & (key_block == query_block)
            & (key_position <= query_position)
        )
        return query_valid & (from_clean | from_proposal)

    return create_block_mask(
        mask_mod,
        B=layout.batch_size,
        H=None,
        Q_LEN=length,
        KV_LEN=2 * length,
        device=layout.clean_ids.device,
        BLOCK_SIZE=flex_block_size,
        _compile=False,
        separate_full_blocks=True,
    )


class IDLMModel(nn.Module):
    """One shared Transformer used as both causal anchor and MASK proposer."""

    def __init__(self, config: IDLMModelConfig = IDLMModelConfig()) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.input_size, config.dim, padding_idx=config.pad_id
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(
                config.dim,
                config.heads,
                config.ffn_dim,
                config.rope_theta,
            )
            for _ in range(config.layers)
        )
        self.norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, config.output_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[self.config.pad_id].zero_()

    def enforce_padding_invariant(self) -> None:
        with torch.no_grad():
            self.embedding.weight[self.config.pad_id].zero_()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _project(self, states: Tensor) -> Tensor:
        return self.output(self.norm(states))

    @torch.compiler.disable
    def prepare_forward_metadata(
        self,
        layout: IDLMTrainingLayout,
        metadata: IDLMForwardMetadata | None = None,
    ) -> IDLMForwardMetadata:
        """Prepare data-dependent packing/Flex setup outside the model forward."""

        metadata = metadata or _packed_document_metadata(layout)
        if metadata.block_mask is not None:
            return metadata
        block_mask = _build_idlm_introspection_block_mask(
            layout,
            clean_prefix_window=self.config.clean_prefix_window,
        )
        return IDLMForwardMetadata(
            clean_indices=metadata.clean_indices,
            clean_cu_seqlens=metadata.clean_cu_seqlens,
            max_clean_seqlen=metadata.max_clean_seqlen,
            block_mask=block_mask,
        )

    def _forward_dense(
        self,
        layout: IDLMTrainingLayout,
        *,
        physical_order: PhysicalOrder,
    ) -> tuple[Tensor, Tensor]:
        length = layout.sequence_length
        proposal_clean_mask = idlm_dense_attention_mask(
            layout, clean_prefix_window=self.config.clean_prefix_window
        )
        proposal_clean_ids = torch.cat((layout.proposal_ids, layout.clean_ids), dim=1)
        positions = torch.cat((layout.positions, layout.positions), dim=1)
        if physical_order == "proposal_clean":
            ids = proposal_clean_ids
            allowed = proposal_clean_mask
        elif physical_order == "clean_proposal":
            permutation = torch.cat(
                (
                    torch.arange(length, 2 * length, device=layout.clean_ids.device),
                    torch.arange(length, device=layout.clean_ids.device),
                )
            )
            ids = proposal_clean_ids.index_select(1, permutation)
            positions = positions.index_select(1, permutation)
            allowed = proposal_clean_mask.index_select(1, permutation).index_select(
                2, permutation
            )
        else:
            raise ValueError(f"unsupported physical order {physical_order!r}")
        states = self.embedding(ids)
        for block in self.blocks:
            states = block(states, positions=positions, allowed=allowed)
        logits = self._project(states)
        if physical_order == "proposal_clean":
            return logits[:, :length], logits[:, length:]
        return logits[:, length:], logits[:, :length]

    def _forward_production(
        self,
        layout: IDLMTrainingLayout,
        metadata: IDLMForwardMetadata | None,
    ) -> tuple[Tensor, Tensor]:
        metadata = metadata or _packed_document_metadata(layout)
        branch_layout = IntrospectionBranchLayout(
            clean_valid=layout.valid,
            proposal_valid=layout.valid,
            block_size=layout.block_size,
            clean_segment_ids=layout.segment_ids,
            proposal_segment_ids=layout.segment_ids,
        )
        block_mask = metadata.block_mask
        if block_mask is None:
            metadata = self.prepare_forward_metadata(layout, metadata)
            block_mask = metadata.block_mask
        if block_mask is None:
            raise AssertionError("I-DLM Flex block mask preparation failed")
        states = self.embedding(torch.cat((layout.clean_ids, layout.proposal_ids), dim=1))
        positions = torch.cat((layout.positions, layout.positions), dim=1)
        for block in self.blocks:
            states = self._forward_production_block(
                block,
                states,
                positions=positions,
                layout=branch_layout,
                metadata=metadata,
                block_mask=block_mask,
            )
        logits = self._project(states)
        return logits[:, layout.sequence_length :], logits[:, : layout.sequence_length]

    def _forward_production_block(
        self,
        block: TransformerBlock,
        states: Tensor,
        *,
        positions: Tensor,
        layout: IntrospectionBranchLayout,
        metadata: IDLMForwardMetadata,
        block_mask: object,
    ) -> Tensor:
        """Varlen clean plus shared proposal attention with an exact max length."""

        batch = states.shape[0]
        clean_length = layout.clean_length
        normalized = block.attention_norm(states)
        clean_states = normalized[:, :clean_length]
        proposal_states = normalized[:, clean_length:]
        attention = block.attention
        clean_qkv = attention.qkv(clean_states).view(
            batch, clean_length, 3, attention.heads, attention.head_dim
        )
        proposal_qkv = attention.qkv(proposal_states).view(
            batch, clean_length, 3, attention.heads, attention.head_dim
        )
        clean_q, clean_k, clean_v = (
            value.transpose(1, 2) for value in clean_qkv.unbind(2)
        )
        proposal_q, proposal_k, proposal_v = (
            value.transpose(1, 2) for value in proposal_qkv.unbind(2)
        )
        clean_q, clean_k = apply_rotary(
            clean_q, clean_k, positions[:, :clean_length], attention.rope_theta
        )
        proposal_q, proposal_k = apply_rotary(
            proposal_q,
            proposal_k,
            positions[:, clean_length:],
            attention.rope_theta,
        )
        packed = PackedCleanQKV(
            query=pack_rows(clean_q.transpose(1, 2), metadata.clean_indices),
            key=pack_rows(clean_k.transpose(1, 2), metadata.clean_indices),
            value=pack_rows(clean_v.transpose(1, 2), metadata.clean_indices),
            cu_seqlens=metadata.clean_cu_seqlens,
            absolute_positions=pack_rows(
                positions[:, :clean_length], metadata.clean_indices
            ),
            max_seqlen=metadata.max_clean_seqlen,
        )
        clean_attended = packed_clean_attention(
            packed,
            window=self.config.clean_prefix_window,
            backend="varlen_flash",
        )
        clean_output = unpack_rows(
            clean_attended,
            metadata.clean_indices,
            clean_attended.new_empty(
                (batch, clean_length, attention.heads, attention.head_dim)
            ),
        )
        proposal_output = branch_attention(
            proposal_q,
            clean_k,
            clean_v,
            proposal_k,
            proposal_v,
            layout,
            backend="flex",
            block_mask=block_mask,
        ).transpose(1, 2)
        attended = torch.cat(
            (
                attention.output(clean_output.flatten(-2)),
                attention.output(proposal_output.flatten(-2)),
            ),
            dim=1,
        )
        states = states + attended
        return states + block.ffn(block.ffn_norm(states))

    def forward_bos(self, count: int, *, device: torch.device) -> Tensor:
        """Score document-first atoms from the ordinary EOT-as-BOS anchor."""

        if count < 0:
            raise ValueError("BOS count cannot be negative")
        if count == 0:
            return self.output.weight.new_empty((0, self.config.output_size))
        ids = torch.full((1, 1), self.config.eot_id, dtype=torch.long, device=device)
        states = self.embedding(ids)
        for block in self.blocks:
            states = block(states, causal=True)
        return self._project(states)[:, 0].expand(count, -1)

    def forward_layout(
        self,
        layout: IDLMTrainingLayout,
        *,
        metadata: IDLMForwardMetadata | None = None,
        allow_dense_reference: bool = False,
        physical_order: PhysicalOrder = "clean_proposal",
        bos_count: int = 0,
    ) -> IDLMModelOutput:
        if allow_dense_reference:
            proposal, clean = self._forward_dense(
                layout, physical_order=physical_order
            )
        else:
            if physical_order != "clean_proposal":
                raise ValueError("production lowering uses physical [clean | proposal]")
            proposal, clean = self._forward_production(layout, metadata)
        return IDLMModelOutput(
            proposal_logits=proposal,
            clean_logits=clean,
            bos_logits=self.forward_bos(bos_count, device=layout.clean_ids.device),
        )

    def forward_sequence(
        self,
        ids: Tensor,
        valid: Tensor,
        *,
        positions: Tensor | None = None,
    ) -> Tensor:
        """Full-replay causal oracle used by lossless ISD inference."""

        if ids.dtype != torch.long or ids.ndim != 2 or ids.shape != valid.shape:
            raise ValueError("causal ids must be rank-2 int64 aligned with valid")
        if valid.dtype != torch.bool:
            raise ValueError("causal validity must be boolean")
        if positions is None:
            positions = torch.arange(ids.shape[1], device=ids.device)[None].expand_as(ids)
        elif positions.shape != ids.shape:
            raise ValueError("causal positions must align with ids")
        if ids.device.type == "cuda":
            return self._forward_sequence_packed(
                ids, valid, positions, allow_dense_reference=False
            )
        query = positions[:, :, None]
        key = positions[:, None, :]
        allowed = valid[:, :, None] & valid[:, None, :] & (key <= query)
        if self.config.clean_prefix_window is not None:
            allowed &= key > query - self.config.clean_prefix_window
        states = self.embedding(ids)
        for block in self.blocks:
            states = block(states, positions=positions, allowed=allowed)
        return self._project(states)

    def _forward_sequence_packed(
        self,
        ids: Tensor,
        valid: Tensor,
        positions: Tensor,
        *,
        allow_dense_reference: bool,
    ) -> Tensor:
        """Varlen-Flash causal sequence path used by the ISD sampler."""

        lengths = valid.sum(1)
        if not torch.compiler.is_compiling() and bool((lengths <= 0).any()):
            raise ValueError("packed I-DLM sequences must be nonempty")
        indices = valid.reshape(-1).nonzero(as_tuple=False).flatten()
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=ids.device),
                lengths.to(torch.int32).cumsum(0, dtype=torch.int32),
            )
        )
        packed_ids = pack_rows(ids, indices)
        packed_positions = pack_rows(positions, indices)
        states = self.embedding(packed_ids)
        max_seqlen = int(lengths.max())
        for block in self.blocks:
            states = block.forward_packed(
                states,
                cu_seqlens=cu_seqlens,
                positions=packed_positions,
                max_seqlen=max_seqlen,
                window=self.config.clean_prefix_window,
                allow_dense_reference=allow_dense_reference,
            )
        packed_logits = self._project(states)
        return unpack_rows(
            packed_logits,
            indices,
            packed_logits.new_zeros(
                (ids.shape[0], ids.shape[1], self.config.output_size)
            ),
        )


__all__ = [
    "PhysicalOrder",
    "IDLMModelConfig",
    "IDLMForwardMetadata",
    "IDLMModelOutput",
    "IDLMModel",
    "idlm_dense_attention_mask",
]
