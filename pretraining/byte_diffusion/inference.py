"""Typed hierarchical K/V caches and cached absorbing-canvas generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .attention import CanvasBranchLayout, branch_attention, build_canvas_block_mask
from .config import ModelMode
from .layers import PackedSelfAttention, apply_rotary, unpack_valid
from .model import ByteDiffusionModel
from .patching import CausalEntropyPatcher, EntropyPatchPlan
from .sampling import (
    CanvasSample,
    UnmaskingStrategy,
    sample_absorbing_canvas,
    sample_absorbing_canvas_batched,
)
from .state import TransactionalDecodeState
from .variable_patching import VariablePatchLayout, build_variable_patch_layout


@dataclass(frozen=True)
class LayerKV:
    """RoPE-applied K and raw V for one clean-prefix attention layer."""

    key: Tensor
    value: Tensor
    storage: _LayerKVStorage | None = None


@dataclass
class _LayerKVStorage:
    """Shared geometric storage with a copy-on-write append frontier."""

    key: Tensor
    value: Tensor
    used: int


def _append_layer_kv(cached: LayerKV, key: Tensor, value: Tensor) -> LayerKV:
    """Append K/V with geometric storage growth rather than quadratic cats."""

    old_length = cached.key.shape[2]
    required = old_length + key.shape[2]
    storage = cached.storage
    capacity = 0 if storage is None else storage.key.shape[2]
    # Extending the current frontier cannot alter the visible prefix. If this
    # cache is an older branch (storage.used > old_length), writing its tail
    # would corrupt a newer descendant and must first copy on write.
    if storage is None or storage.used != old_length or capacity < required:
        growth = max(512, max(1, old_length // 4))
        capacity = max(required, old_length + growth)
        shape = (*cached.key.shape[:2], capacity, cached.key.shape[-1])
        storage_key = torch.empty(
            shape, dtype=cached.key.dtype, device=cached.key.device
        )
        storage_value = torch.empty(
            shape, dtype=cached.value.dtype, device=cached.value.device
        )
        storage_key[:, :, :old_length].copy_(cached.key)
        storage_value[:, :, :old_length].copy_(cached.value)
        storage = _LayerKVStorage(storage_key, storage_value, old_length)
    storage.key[:, :, old_length:required].copy_(key)
    storage.value[:, :, old_length:required].copy_(value)
    storage.used = required
    return LayerKV(
        storage.key[:, :, :required],
        storage.value[:, :, :required],
        storage,
    )


@dataclass(frozen=True)
class HierarchicalPrefixCache:
    """Immutable clean-prefix state reused by every canvas denoising NFE."""

    ids: Tensor
    valid: Tensor
    patch_valid: Tensor
    positions: Tensor
    patch_positions: Tensor
    patch_states: Tensor
    local: tuple[LayerKV, ...]
    global_: tuple[LayerKV, ...]
    decoder: tuple[LayerKV, ...]
    has_virtual_bos: bool = True

    @property
    def batch_size(self) -> int:
        return self.ids.shape[0]

    @property
    def length(self) -> int:
        return self.ids.shape[1]


@dataclass(frozen=True)
class CachedCanvasPlan:
    starts: Tensor
    canvas_length: int
    branch_positions: Tensor
    branch_patch_positions: Tensor | None
    local_layout: CanvasBranchLayout
    global_layout: CanvasBranchLayout | None
    local_block_mask: object | None
    global_block_mask: object | None


@dataclass(frozen=True)
class CanvasGeneration:
    alignment_ids: Tensor
    canvas: CanvasSample | None
    committed_canvas_ids: Tensor


@dataclass(frozen=True)
class EntropyCanvasGeneration:
    """One entropy-topology proposal, truncated only by semantic EOT."""

    alignment_ids: Tensor
    canvas: CanvasSample | None
    committed_canvas_ids: Tensor
    overflow_canvas_ids: Tensor


def _entropy_patch_plan(
    patcher: CausalEntropyPatcher, ids: Tensor
) -> EntropyPatchPlan:
    """Apply the authenticated CPU patcher to one committed document."""

    if ids.ndim != 1 or ids.dtype != torch.long or ids.numel() == 0:
        raise ValueError("entropy patching requires a nonempty int64 document")
    host_ids = ids.detach().to(device="cpu", dtype=torch.long).numpy()
    documents = torch.zeros(ids.numel(), dtype=torch.long).numpy()
    return patcher.patch(host_ids, documents)


def _variable_prefix_metadata(
    patcher: CausalEntropyPatcher, ids: Tensor
) -> tuple[dict[str, Tensor], VariablePatchLayout]:
    """Construct the same ragged pooling/global topology used in training."""

    plan = _entropy_patch_plan(patcher, ids)
    length = ids.numel()
    valid = torch.ones((1, length), dtype=torch.bool)
    document_ids = torch.zeros((1, length), dtype=torch.long)
    document_offsets = torch.arange(length, dtype=torch.long)[None]
    byte_indices = torch.arange(length, dtype=torch.long)
    starts = torch.from_numpy(plan.layout.patch_starts)
    byte_to_patch = torch.from_numpy(plan.layout.byte_to_patch)
    patch_offsets = (
        byte_indices - starts.index_select(0, byte_to_patch)
    )[None]
    layout = build_variable_patch_layout(
        valid,
        document_ids,
        document_offsets,
        patch_offsets,
        max_patch_size=patcher.config.max_patch_size,
    )
    device = ids.device
    metadata = {
        "valid": valid.to(device),
        "positions": document_offsets.to(device),
        "document_ids": document_ids.to(device),
        "byte_indices": layout.byte_indices.to(device),
        "byte_cu_seqlens": layout.byte_cu_seqlens.to(device),
        "patch_cu_seqlens": layout.patch_cu_seqlens.to(device),
        "condition_patch_indices": layout.condition_patch_indices.to(device),
        "global_patch_sources": layout.global_patch_sources.to(device),
        "global_patch_positions": layout.global_patch_positions.to(device),
        "physical_to_global_patch_indices": (
            layout.physical_to_global_patch_indices.to(device)
        ),
        "bos_condition_indices": layout.bos_condition_indices.to(device),
        "patch_byte_cu_seqlens": layout.patch_byte_cu_seqlens.to(device),
    }
    return metadata, layout


def entropy_next_byte_starts_patch(
    patcher: CausalEntropyPatcher, committed_ids: Tensor
) -> bool:
    """Return the causal boundary decision immediately after a prefix.

    The appended byte value is deliberately arbitrary: the patcher's entropy
    for position ``i`` hashes only positions before ``i``.  Keeping this probe
    here makes that no-lookahead invariant explicit in serving.
    """

    probe = torch.cat((committed_ids, committed_ids.new_zeros(1)))
    return bool(_entropy_patch_plan(patcher, probe).starts[-1])


@torch.no_grad()
def entropy_ar_next_logits(
    model: ByteDiffusionModel,
    patcher: CausalEntropyPatcher,
    committed_ids: Tensor,
) -> Tensor:
    """Score the next byte with the authenticated variable clean topology."""

    metadata, _ = _variable_prefix_metadata(patcher, committed_ids)
    output = model.forward_ar_varlen(
        committed_ids[None],
        metadata.pop("valid"),
        allow_dense_reference=committed_ids.device.type != "cuda",
        document_ids=metadata.pop("document_ids"),
        max_patch_size=patcher.config.max_patch_size,
        **metadata,
    )
    return output.logits[0, -1]


@torch.no_grad()
def denoise_entropy_blt_reference(
    model: ByteDiffusionModel,
    patcher: CausalEntropyPatcher,
    committed_ids: Tensor,
    noisy_ids: Tensor,
) -> Tensor:
    """Uncached Fast-BLT denoising with the trained variable-patch topology.

    A clean dummy atom materializes the next patch origin for the existing
    clean-plus-branch model API.  The decoder prefix cutoff is immediately
    before that slot, so neither it nor any noisy candidate can enter the
    immutable clean context.
    """

    if noisy_ids.ndim != 1 or noisy_ids.dtype != torch.long:
        raise ValueError("entropy BLT canvas must be rank-1 int64")
    canvas_length = noisy_ids.numel()
    if canvas_length <= 0:
        raise ValueError("entropy BLT canvas must be nonempty")
    if not entropy_next_byte_starts_patch(patcher, committed_ids):
        raise ValueError("entropy BLT branch requires a patch-aligned prefix")

    start = committed_ids.numel()
    clean_ids = torch.cat((committed_ids, committed_ids.new_zeros(1)))
    metadata, layout = _variable_prefix_metadata(patcher, clean_ids)
    origins = layout.physical_patch_start_columns.eq(start).nonzero().flatten()
    if origins.numel() != 1:
        raise AssertionError("next entropy patch origin was not unique")
    prior = layout.physical_patch_prior_condition_indices.index_select(
        0, origins
    ).to(committed_ids.device)[None]
    starts = torch.tensor(
        [[start]], dtype=torch.long, device=committed_ids.device
    )
    branch_valid = torch.ones(
        (1, 1, canvas_length), dtype=torch.bool, device=committed_ids.device
    )
    branch_mask = None
    if (
        committed_ids.device.type == "cuda"
        and model.config.decoder_branch_attention == "shared_flex"
    ):
        positions = metadata["positions"]
        branch_positions = starts[:, :, None] + torch.arange(
            canvas_length, device=committed_ids.device
        )[None, None]
        branch_layout = CanvasBranchLayout(
            clean_valid=metadata["valid"],
            branch_valid=branch_valid,
            prefix_lengths=starts,
            prefix_window=model.config.decoder_prefix_window,
            clean_positions=positions,
            branch_positions=branch_positions,
            clean_segment_ids=metadata["document_ids"],
            branch_segment_ids=torch.zeros_like(starts),
        )
        branch_mask = build_canvas_block_mask(branch_layout)

    output = model.forward_blt_d_branches(
        clean_ids[None],
        metadata.pop("valid"),
        noisy_ids[None, None],
        branch_valid,
        starts,
        document_ids=metadata.pop("document_ids"),
        branch_condition_indices=prior,
        branch_block_mask=branch_mask,
        max_patch_size=patcher.config.max_patch_size,
        **metadata,
    )
    return output.branch_logits[0, 0]


class EntropyPatchedCanvasGenerator:
    """Exact reference serving for a causal-entropy Fast-BLT checkpoint.

    This deliberately recomputes the clean hierarchy.  Variable patch lengths
    make the fixed-stride K/V cache invalid, and silently routing through it
    would change both pooling and global positions.
    """

    def __init__(
        self,
        model: ByteDiffusionModel,
        patcher: CausalEntropyPatcher,
        *,
        block_length: int,
        seed: int,
        blocked_output_ids: Tensor | None = None,
    ) -> None:
        if patcher.model.config.vocab_size != model.config.vocab.output_size:
            raise ValueError("entropy patcher vocabulary differs from the model")
        self.model = model
        self.patcher = patcher
        if block_length <= 0:
            raise ValueError("entropy BLT block length must be positive")
        self.block_length = block_length
        self.generator = torch.Generator(device=model.embedding.weight.device)
        self.generator.manual_seed(seed)
        self.blocked_output_ids = (
            None
            if blocked_output_ids is None
            else blocked_output_ids.to(
                device=model.embedding.weight.device, dtype=torch.long
            )
        )
        if self.blocked_output_ids is not None:
            if self.blocked_output_ids.ndim != 1 or bool(
                (
                    (self.blocked_output_ids < 0)
                    | (
                        self.blocked_output_ids
                        >= model.config.vocab.output_size
                    )
                ).any()
            ):
                raise ValueError("blocked output ids must be valid clean ids")
            if bool(
                self.blocked_output_ids.eq(model.config.vocab.eot_id).any()
            ):
                raise ValueError("EOT cannot be blocked during generation")
        self.ids = torch.empty(
            0, dtype=torch.long, device=model.embedding.weight.device
        )
        self.forwards = 0
        self.denoising_forwards = 0
        self.causal_forwards = 0
        self.rejected_bytes = 0

    def _masked_logits(self, logits: Tensor) -> Tensor:
        if self.blocked_output_ids is None or not self.blocked_output_ids.numel():
            return logits
        logits = logits.clone()
        logits[..., self.blocked_output_ids] = -torch.inf
        return logits

    def prefill(self, prefix: Tensor) -> None:
        if prefix.ndim != 1 or prefix.dtype != torch.long or not prefix.numel():
            raise ValueError("entropy generation prefix must be nonempty int64")
        if bool(
            ((prefix < 0) | (prefix >= self.model.config.vocab.output_size)).any()
        ):
            raise ValueError("entropy generation prefix contains an invalid id")
        eot = prefix.eq(self.model.config.vocab.eot_id).nonzero().flatten()
        if eot.numel():
            raise ValueError("entropy generation prompt cannot contain EOT")
        self.ids = prefix.to(self.ids.device).clone()

    def _choose(self, logits: Tensor, *, stochastic: bool) -> Tensor:
        logits = self._masked_logits(logits.float())
        if not stochastic:
            return logits.argmax(-1, keepdim=True).to(torch.long)
        return torch.multinomial(
            logits.softmax(-1), 1, generator=self.generator
        ).to(torch.long)

    @torch.no_grad()
    def align_prefix_ar(
        self,
        *,
        stochastic: bool = False,
        max_forwards: int | None = None,
    ) -> Tensor:
        """Close an incomplete variable patch without noising prompt bytes."""

        if not self.ids.numel():
            raise RuntimeError("prefill must run before entropy generation")
        if max_forwards is not None and max_forwards < 0:
            raise ValueError("AR alignment forward budget cannot be negative")
        generated: list[Tensor] = []
        for _ in range(self.patcher.config.max_patch_size):
            if entropy_next_byte_starts_patch(self.patcher, self.ids):
                break
            if max_forwards is not None and len(generated) >= max_forwards:
                break
            logits = entropy_ar_next_logits(self.model, self.patcher, self.ids)
            self.forwards += 1
            self.causal_forwards += 1
            chosen = self._choose(logits, stochastic=stochastic)
            self.ids = torch.cat((self.ids, chosen))
            generated.append(chosen)
            if int(chosen) == self.model.config.vocab.eot_id:
                break
        else:
            raise AssertionError("entropy patcher exceeded its maximum patch size")
        return torch.cat(generated) if generated else self.ids.new_empty((0,))

    @torch.no_grad()
    def generate_blt(
        self,
        steps: int,
        *,
        strategy: UnmaskingStrategy = "confidence",
        confidence_threshold: float = 0.7,
        entropy_budget: float = 1.0,
        stochastic: bool = False,
    ) -> EntropyCanvasGeneration:
        """Generate and commit one fixed-size Fast-BLT block."""

        if steps <= 0:
            raise ValueError("diffusion steps must be positive")
        alignment = self.align_prefix_ar(stochastic=stochastic)
        if int(self.ids[-1]) == self.model.config.vocab.eot_id:
            empty = self.ids.new_empty((0,))
            return EntropyCanvasGeneration(alignment, None, empty, empty)
        if not entropy_next_byte_starts_patch(self.patcher, self.ids):
            raise AssertionError("AR alignment did not close the variable patch")

        canvas_length = self.block_length
        initial = torch.full(
            (1, canvas_length),
            self.model.config.vocab.mask_id,
            dtype=torch.long,
            device=self.ids.device,
        )

        def denoise(canvas: Tensor) -> Tensor:
            return self._masked_logits(
                denoise_entropy_blt_reference(
                    self.model, self.patcher, self.ids, canvas[0]
                )
            )[None]

        sampled = sample_absorbing_canvas_batched(
            initial,
            denoise,
            steps=steps,
            mask_id=self.model.config.vocab.mask_id,
            eot_id=self.model.config.vocab.eot_id,
            generator=self.generator,
            strategy=strategy,
            confidence_threshold=confidence_threshold,
            entropy_budget=entropy_budget,
            stochastic=stochastic,
        )
        self.forwards += sampled.executed_nfe
        self.denoising_forwards += sampled.executed_nfe
        canvas = CanvasSample(
            ids=sampled.ids[0],
            active=sampled.active[0],
            useful_nfe=int(sampled.useful_nfe[0]),
            executed_nfe=sampled.executed_nfe,
        )
        eot = canvas.ids.eq(self.model.config.vocab.eot_id).nonzero().flatten()
        semantic_length = int(eot[0]) + 1 if eot.numel() else canvas_length
        semantic = canvas.ids[:semantic_length]
        if bool(semantic.eq(self.model.config.vocab.mask_id).any()):
            raise AssertionError("live entropy canvas retained MASK after sampling")

        if not 1 <= semantic_length <= canvas_length:
            raise AssertionError("entropy serving selected an invalid commit length")
        committed = semantic
        overflow = canvas.ids[semantic_length:]
        self.ids = torch.cat((self.ids, committed))
        self.rejected_bytes += overflow.numel()
        return EntropyCanvasGeneration(
            alignment_ids=alignment,
            canvas=canvas,
            committed_canvas_ids=committed,
            overflow_canvas_ids=overflow,
        )


def _project_kv(
    attention: PackedSelfAttention,
    states: Tensor,
    positions: Tensor,
) -> LayerKV:
    batch, length, _ = states.shape
    qkv = attention.qkv(states).view(
        batch, length, 3, attention.heads, attention.head_dim
    )
    query, key, value = (item.transpose(1, 2) for item in qkv.unbind(2))
    _, key = apply_rotary(query, key, positions, attention.rope_theta)
    return LayerKV(key=key, value=value)


def _full_cu(batch: int, length: int, device: torch.device) -> Tensor:
    return torch.arange(
        0, (batch + 1) * length, length, dtype=torch.int32, device=device
    )


def _gather_absolute_positions(
    values: Tensor,
    valid: Tensor,
    positions: Tensor,
    targets: Tensor,
    *,
    fill_value: float | int = 0,
) -> Tensor:
    """Gather ragged cache values by semantic position, never storage slot."""

    if values.shape[:2] != valid.shape or positions.shape != valid.shape:
        raise ValueError("cache values, validity, and positions must align")
    if targets.ndim != 2 or targets.shape[0] != values.shape[0]:
        raise ValueError("position targets must be int64 [batch, count]")
    matches = valid[:, None] & positions[:, None].eq(targets[:, :, None])
    found = matches.any(-1)
    indices = matches.to(torch.int64).argmax(-1)
    gather_shape = (*indices.shape, *values.shape[2:])
    expanded = indices.view(*indices.shape, *([1] * (values.ndim - 2))).expand(
        gather_shape
    )
    selected = torch.gather(
        values[:, None].expand(-1, targets.shape[1], *values.shape[1:]),
        2,
        expanded.unsqueeze(2),
    ).squeeze(2)
    found_shape = (*found.shape, *([1] * (values.ndim - 2)))
    return torch.where(
        found.view(found_shape), selected, torch.full_like(selected, fill_value)
    )


def document_start_ar_metadata(valid: Tensor, patch_stride: int) -> dict[str, Tensor]:
    """Build virtual-BOS maps for one prefix document in every padded row."""

    if valid.ndim != 2 or valid.shape[0] == 0 or valid.shape[1] == 0:
        raise ValueError("AR alignment requires a nonempty [batch, length] mask")
    if patch_stride <= 0 or valid.shape[1] % patch_stride:
        raise ValueError("AR alignment storage must contain complete patch slots")
    lengths = valid.sum(1).to(torch.long)
    if bool((lengths == 0).any()) or bool(((~valid[:, :-1]) & valid[:, 1:]).any()):
        raise ValueError("AR alignment requires one nonempty valid prefix per row")
    batch, storage_length = valid.shape
    patches_per_row = storage_length // patch_stride
    physical_patch_counts = torch.div(
        lengths + patch_stride - 1, patch_stride, rounding_mode="floor"
    )
    device = valid.device
    byte_indices = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
    byte_cu = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=device),
            lengths.cumsum(0).to(torch.int32),
        )
    )
    patch_valid = valid.view(batch, patches_per_row, patch_stride).any(-1)
    patch_indices = torch.nonzero(
        patch_valid.reshape(-1), as_tuple=False
    ).flatten()
    physical_patches = int(patch_indices.numel())
    physical_rows = torch.repeat_interleave(
        torch.arange(batch, dtype=torch.long, device=device),
        physical_patch_counts,
    )
    physical_starts = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            physical_patch_counts.cumsum(0)[:-1],
        )
    )
    within_patch = torch.arange(
        physical_patches, dtype=torch.long, device=device
    ) - torch.repeat_interleave(physical_starts, physical_patch_counts)
    # Each row inserts one virtual BOS before its packed physical patches.
    physical_to_global = (
        torch.arange(physical_patches, dtype=torch.long, device=device)
        + physical_rows
        + 1
    )
    global_patch_counts = physical_patch_counts + 1
    patch_cu = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=device),
            global_patch_counts.cumsum(0).to(torch.int32),
        )
    )
    total_global_patches = physical_patches + batch
    global_patch_sources = torch.full(
        (total_global_patches,), -1, dtype=torch.long, device=device
    )
    global_patch_sources[physical_to_global] = torch.arange(
        physical_patches, dtype=torch.long, device=device
    )
    global_patch_positions = torch.zeros(
        total_global_patches, dtype=torch.long, device=device
    )
    global_patch_positions[physical_to_global] = within_patch + 1

    byte_rows = torch.div(byte_indices, storage_length, rounding_mode="floor")
    byte_positions = byte_indices.remainder(storage_length)
    physical_patch = physical_starts.index_select(0, byte_rows) + torch.div(
        byte_positions, patch_stride, rounding_mode="floor"
    )
    current_global = physical_to_global.index_select(0, physical_patch)
    condition_patch_indices = torch.where(
        byte_positions.remainder(patch_stride).eq(patch_stride - 1),
        current_global,
        current_global - 1,
    )
    return {
        "byte_indices": byte_indices,
        "byte_cu_seqlens": byte_cu,
        "patch_indices": patch_indices,
        "patch_cu_seqlens": patch_cu,
        "condition_patch_indices": condition_patch_indices,
        "global_patch_sources": global_patch_sources,
        "global_patch_positions": global_patch_positions,
        "physical_to_global_patch_indices": physical_to_global,
        "bos_condition_indices": patch_cu[:-1].to(torch.long),
    }


@torch.no_grad()
def prefill_prefix(
    model: ByteDiffusionModel,
    ids: Tensor,
    *,
    valid: Tensor | None = None,
    positions: Tensor | None = None,
    allow_dense_reference: bool = False,
    document_start: bool = True,
) -> HierarchicalPrefixCache:
    """Causally prefill a full, patch-aligned prefix and retain typed K/V."""

    if ids.dtype != torch.long or ids.ndim != 2 or ids.shape[1] == 0:
        raise ValueError("prefill ids must be nonempty int64 [batch, length]")
    if ids.shape[1] % model.config.patch_stride:
        raise ValueError("cached prefix length must be patch aligned")
    batch, length = ids.shape
    device = ids.device
    if positions is None:
        positions = torch.arange(length, device=device)[None].expand_as(ids)
    if positions.shape != ids.shape or positions.dtype != torch.long:
        raise ValueError("prefill positions must be int64 and align with ids")
    if valid is None:
        valid = torch.ones_like(ids, dtype=torch.bool)
    elif valid.shape != ids.shape or valid.dtype != torch.bool:
        raise ValueError("prefill validity must be boolean and align with ids")
    if bool(((~valid[:, :-1]) & valid[:, 1:]).any()) or bool(
        (valid.sum(1) == 0).any()
    ):
        raise ValueError("prefill requires one nonempty valid prefix per row")
    if bool((ids[~valid] != model.config.vocab.pad_id).any()):
        raise ValueError("invalid prefill storage slots must contain PAD")
    if bool(
        ((ids[valid] < 0) | (ids[valid] >= model.config.vocab.output_size)).any()
    ):
        raise ValueError("valid prefix contains an input-only or invalid id")
    lengths = valid.sum(1)
    cu = model._cu_seqlens(lengths)

    local_states = model.encoder_embeddings(ids)
    local_cache: list[LayerKV] = []
    for block in model.encoder:
        normalized = block.attention_norm(local_states)
        local_cache.append(_project_kv(block.attention, normalized, positions))
        packed = block.forward_packed(
            local_states[valid],
            cu_seqlens=cu,
            positions=positions[valid],
            max_seqlen=length,
            window=model.config.local_window,
            allow_dense_reference=allow_dense_reference,
        )
        local_states = unpack_valid(packed, valid, local_states)

    physical_patches = model.pool(local_states, valid)
    physical_patch_length = physical_patches.shape[1]
    physical_patch_valid = valid.view(
        batch, physical_patch_length, model.config.patch_stride
    ).any(-1)
    physical_patch_positions = (
        positions[:, :: model.config.patch_stride] // model.config.patch_stride
    )
    if document_start:
        bos_patch = model.virtual_bos_patch_input(
            device, allow_dense_reference=allow_dense_reference
        )
        patches = torch.cat(
            (
                bos_patch.view(1, 1, -1).expand(batch, -1, -1),
                physical_patches,
            ),
            dim=1,
        )
        patch_valid = torch.cat(
            (
                torch.ones((batch, 1), dtype=torch.bool, device=device),
                physical_patch_valid,
            ),
            dim=1,
        )
        patch_positions = torch.cat(
            (
                torch.zeros((batch, 1), dtype=torch.long, device=device),
                physical_patch_positions + 1,
            ),
            dim=1,
        )
    else:
        patches = physical_patches
        patch_valid = physical_patch_valid
        patch_positions = physical_patch_positions
    patch_length = patches.shape[1]
    patch_cu = model._cu_seqlens(patch_valid.sum(1))
    global_cache: list[LayerKV] = []
    for block in model.global_blocks:
        normalized = block.attention_norm(patches)
        global_cache.append(_project_kv(block.attention, normalized, patch_positions))
        packed = block.forward_packed(
            patches[patch_valid],
            cu_seqlens=patch_cu,
            positions=patch_positions[patch_valid],
            max_seqlen=patch_length,
            window=model.config.global_window,
            allow_dense_reference=allow_dense_reference,
        )
        patches = unpack_valid(packed, patch_valid, patches)
    normalized_patches = patches.new_zeros(patches.shape)
    normalized_patches[patch_valid] = model.global_norm(patches[patch_valid])

    if document_start:
        byte_offsets = torch.arange(length, device=device)
        physical_patch = torch.div(
            byte_offsets, model.config.patch_stride, rounding_mode="floor"
        )
        condition_patch = torch.where(
            byte_offsets.remainder(model.config.patch_stride).eq(
                model.config.patch_stride - 1
            ),
            physical_patch + 1,
            physical_patch,
        )
        condition = torch.gather(
            normalized_patches,
            1,
            condition_patch[None, :, None].expand(
                batch, -1, model.config.global_dim
            ),
        )
    else:
        condition = model._aligned_condition(
            normalized_patches, length, ModelMode.AR
        )
    decoder_states = model.decoder_embeddings(ids, ModelMode.AR)
    decoder_cache: list[LayerKV] = []
    for conditioned in model.decoder:
        decoder_states = conditioned._add_condition(decoder_states, condition)
        block = conditioned.block
        normalized = block.attention_norm(decoder_states)
        decoder_cache.append(_project_kv(block.attention, normalized, positions))
        packed_states = block.forward_packed(
            decoder_states[valid],
            cu_seqlens=cu,
            positions=positions[valid],
            max_seqlen=length,
            window=model.config.local_window,
            allow_dense_reference=allow_dense_reference,
        )
        decoder_states = unpack_valid(packed_states, valid, decoder_states)

    return HierarchicalPrefixCache(
        ids=ids.clone(),
        valid=valid,
        patch_valid=patch_valid,
        positions=positions.clone(),
        patch_positions=patch_positions.clone(),
        patch_states=normalized_patches.clone(),
        local=tuple(local_cache),
        global_=tuple(global_cache),
        decoder=tuple(decoder_cache),
        has_virtual_bos=document_start,
    )


def _cached_branch_block(
    states: Tensor,
    clean_kv: LayerKV,
    conditioned_block,
    condition: Tensor | None,
    positions: Tensor,
    layout: CanvasBranchLayout,
    block_mask,
    *,
    allow_dense_reference: bool,
) -> Tensor:
    if condition is not None:
        states = conditioned_block._add_condition(states, condition)
        block = conditioned_block.block
    else:
        block = conditioned_block
    normalized = block.attention_norm(states)
    attention = block.attention
    batch, length, _ = normalized.shape
    qkv = attention.qkv(normalized).view(
        batch, length, 3, attention.heads, attention.head_dim
    )
    query, key, value = (item.transpose(1, 2) for item in qkv.unbind(2))
    query, key = apply_rotary(query, key, positions, attention.rope_theta)
    attended = branch_attention(
        query,
        clean_kv.key,
        clean_kv.value,
        key,
        value,
        layout,
        backend="dense_reference" if allow_dense_reference else "flex",
        block_mask=block_mask,
        allow_dense_reference=allow_dense_reference,
    )
    states = states + attention.output(
        attended.transpose(1, 2).reshape(batch, length, attention.dim)
    )
    return states + block.ffn(block.ffn_norm(states))


@torch.no_grad()
def prepare_cached_canvas(
    model: ByteDiffusionModel,
    cache: HierarchicalPrefixCache,
    starts: Tensor,
    canvas_length: int,
    *,
    allow_dense_reference: bool = False,
    include_global: bool = True,
) -> CachedCanvasPlan:
    """Build reusable Flex block masks and absolute positions once per canvas."""

    if starts.ndim != 2 or starts.dtype != torch.long:
        raise ValueError("cached canvas starts must be int64 [batch, branches]")
    batch, branches = starts.shape
    if batch != cache.batch_size:
        raise ValueError("cached canvas geometry does not align with the prefix")
    if branches != 1:
        raise ValueError("cached inference currently accepts one canvas per row")
    if canvas_length % model.config.patch_stride:
        raise ValueError("cached canvas must be patch aligned")
    if not torch.compiler.is_compiling():
        if bool((starts % model.config.patch_stride).any()):
            raise ValueError("cached canvas starts must be patch aligned")
        prefix_lengths = cache.valid.sum(1, keepdim=True)
        if bool(((starts < 0) | (starts > prefix_lengths)).any()):
            raise ValueError("cached canvas start lies beyond the clean prefix")
    branch_valid = torch.ones(
        (batch, branches, canvas_length),
        dtype=torch.bool,
        device=starts.device,
    )
    offsets = torch.arange(canvas_length, device=starts.device)
    branch_positions = starts[:, :, None] + offsets[None, None, :]
    local_layout = CanvasBranchLayout(
        clean_valid=cache.valid,
        branch_valid=branch_valid,
        # Cached storage may contain right-padding holes from the ragged
        # prefill, followed by valid appended K/V.  ``prefix_lengths`` is a
        # physical storage cutoff, not a semantic token count.  Every valid
        # cache entry is already in the generated prefix, so expose the full
        # storage extent and let ``clean_valid`` (plus absolute positions for
        # the local window) select the semantic keys.
        prefix_lengths=torch.full_like(starts, cache.length),
        prefix_window=model.config.local_window,
        clean_positions=cache.positions,
        branch_positions=branch_positions,
    )
    local_mask = None if allow_dense_reference else build_canvas_block_mask(local_layout)
    branch_patch_positions = None
    global_layout = None
    global_mask = None
    if include_global:
        canvas_patches = canvas_length // model.config.patch_stride
        patch_starts = starts // model.config.patch_stride
        global_patch_starts = patch_starts + int(cache.has_virtual_bos)
        branch_patch_positions = global_patch_starts[:, :, None] + torch.arange(
            canvas_patches, device=starts.device
        )[None, None, :]
        global_layout = CanvasBranchLayout(
            clean_valid=cache.patch_valid,
            branch_valid=torch.ones(
                batch, 1, canvas_patches, dtype=torch.bool, device=starts.device
            ),
            prefix_lengths=torch.full_like(
                global_patch_starts, cache.patch_valid.shape[1]
            ),
            prefix_window=model.config.global_window,
            clean_positions=cache.patch_positions,
            branch_positions=branch_patch_positions,
        )
        global_mask = (
            None
            if allow_dense_reference
            else build_canvas_block_mask(global_layout)
        )
    return CachedCanvasPlan(
        starts=starts,
        canvas_length=canvas_length,
        branch_positions=branch_positions,
        branch_patch_positions=branch_patch_positions,
        local_layout=local_layout,
        global_layout=global_layout,
        local_block_mask=local_mask,
        global_block_mask=global_mask,
    )


@torch.no_grad()
def denoise_canvas_cached(
    model: ByteDiffusionModel,
    cache: HierarchicalPrefixCache,
    noisy_ids: Tensor,
    starts: Tensor,
    *,
    allow_dense_reference: bool = False,
    plan: CachedCanvasPlan | None = None,
) -> Tensor:
    """Denoise branches while reusing all clean local/global/decoder K/V."""

    if noisy_ids.dtype != torch.long or noisy_ids.ndim != 3:
        raise ValueError("noisy canvas ids must be int64 [batch, branches, length]")
    batch, branches, canvas_length = noisy_ids.shape
    if plan is None:
        plan = prepare_cached_canvas(
            model,
            cache,
            starts,
            canvas_length,
            allow_dense_reference=allow_dense_reference,
        )
    elif not torch.compiler.is_compiling() and (
        plan.starts.data_ptr() != starts.data_ptr()
        or plan.canvas_length != canvas_length
    ):
        raise ValueError("cached canvas plan does not match starts/length")
    branch_valid = plan.local_layout.branch_valid
    flat_positions = plan.branch_positions[:, 0]
    states = model.embedding(noisy_ids[:, 0])
    if model.ngrams is not None:
        halo = max(model.config.ngram_orders) - 1
        indices = starts[:, 0, None] - halo + torch.arange(halo, device=noisy_ids.device)
        prefix = _gather_absolute_positions(
            cache.ids,
            cache.valid,
            cache.positions,
            indices,
            fill_value=model.config.vocab.pad_id,
        )
        visible = torch.cat((prefix, noisy_ids[:, 0]), dim=1)
        states = model.combine_ngram_features(
            states, model.ngrams(visible)[:, -canvas_length:]
        )
    for block, clean_kv in zip(model.encoder, cache.local, strict=True):
        states = _cached_branch_block(
            states,
            clean_kv,
            block,
            None,
            flat_positions,
            plan.local_layout,
            plan.local_block_mask,
            allow_dense_reference=allow_dense_reference,
        )

    if plan.branch_patch_positions is None or plan.global_layout is None:
        raise ValueError("canvas denoising requires a global cached plan")
    patches = model.pool(states, branch_valid[:, 0])
    for block, clean_kv in zip(model.global_blocks, cache.global_, strict=True):
        patches = _cached_branch_block(
            patches,
            clean_kv,
            block,
            None,
            plan.branch_patch_positions[:, 0],
            plan.global_layout,
            plan.global_block_mask,
            allow_dense_reference=allow_dense_reference,
        )
    patches = model.global_norm(patches)

    condition = patches.repeat_interleave(model.config.patch_stride, dim=1)
    states = model.decoder_embeddings(noisy_ids[:, 0], ModelMode.CANVAS)
    for block, clean_kv in zip(model.decoder, cache.decoder, strict=True):
        states = _cached_branch_block(
            states,
            clean_kv,
            block,
            condition,
            flat_positions,
            plan.local_layout,
            plan.local_block_mask,
            allow_dense_reference=allow_dense_reference,
        )
    states = model.decoder_norm(states)
    if model.output is None:
        return F.linear(states, model.embedding.weight[: model.config.vocab.output_size])
    return model.output(states)


@torch.no_grad()
def denoise_blt_cached(
    model: ByteDiffusionModel,
    cache: HierarchicalPrefixCache,
    noisy_ids: Tensor,
    starts: Tensor,
    *,
    allow_dense_reference: bool = False,
    plan: CachedCanvasPlan | None = None,
) -> Tensor:
    """Fast-BLT decoder-only denoising against immutable clean-prefix K/V."""

    if noisy_ids.dtype != torch.long or noisy_ids.ndim != 3:
        raise ValueError("noisy BLT ids must be int64 [batch, branches, length]")
    batch, branches, block_length = noisy_ids.shape
    if branches != 1:
        raise ValueError("cached BLT inference currently accepts one branch per row")
    if plan is None:
        plan = prepare_cached_canvas(
            model,
            cache,
            starts,
            block_length,
            allow_dense_reference=allow_dense_reference,
            include_global=False,
        )
    flat_positions = plan.branch_positions[:, 0]
    prior_patch = (
        starts[:, 0] // model.config.patch_stride
        if cache.has_virtual_bos
        else starts[:, 0] // model.config.patch_stride - 1
    )
    prior_state = _gather_absolute_positions(
        cache.patch_states,
        cache.patch_valid,
        cache.patch_positions,
        prior_patch[:, None],
    )[:, 0]
    prior_state = torch.where(
        prior_patch[:, None].ge(0), prior_state, torch.zeros_like(prior_state)
    )
    condition = prior_state[:, None, :].expand(-1, block_length, -1)
    states = model.decoder_embeddings(noisy_ids[:, 0], ModelMode.BLT_D)
    for block, clean_kv in zip(model.decoder, cache.decoder, strict=True):
        states = _cached_branch_block(
            states,
            clean_kv,
            block,
            condition,
            flat_positions,
            plan.local_layout,
            plan.local_block_mask,
            allow_dense_reference=allow_dense_reference,
        )
    states = model.decoder_norm(states)
    if model.output is None:
        return F.linear(
            states, model.embedding.weight[: model.config.vocab.output_size]
        )
    return model.output(states)


def _append_causal_block(
    states: Tensor,
    cached: LayerKV,
    block,
    *,
    cached_positions: Tensor,
    cached_valid: Tensor,
    query_positions: Tensor,
    window: int | None,
) -> tuple[Tensor, LayerKV]:
    """Append a small committed block to one causal Transformer cache."""

    normalized = block.attention_norm(states)
    attention = block.attention
    batch, query_length, _ = normalized.shape
    qkv = attention.qkv(normalized).view(
        batch, query_length, 3, attention.heads, attention.head_dim
    )
    query, key, value = (item.transpose(1, 2) for item in qkv.unbind(2))
    query, key = apply_rotary(
        query, key, query_positions, attention.rope_theta
    )
    updated_cache = _append_layer_kv(cached, key, value)
    full_key = updated_cache.key
    full_value = updated_cache.value
    key_positions = torch.cat((cached_positions, query_positions), dim=1)
    distance = query_positions[:, :, None] - key_positions[:, None, :]
    allowed = distance.ge(0)
    full_valid = torch.cat(
        (cached_valid, torch.ones_like(query_positions, dtype=torch.bool)), dim=1
    )
    allowed &= full_valid[:, None]
    if window is not None:
        allowed &= distance.lt(window)
    attended = F.scaled_dot_product_attention(
        query,
        full_key,
        full_value,
        attn_mask=allowed[:, None],
    )
    states = states + attention.output(
        attended.transpose(1, 2).reshape(batch, query_length, attention.dim)
    )
    states = states + block.ffn(block.ffn_norm(states))
    return states, updated_cache


@torch.no_grad()
def append_clean_block(
    model: ByteDiffusionModel,
    cache: HierarchicalPrefixCache,
    committed_ids: Tensor,
) -> HierarchicalPrefixCache:
    """Incrementally make one full committed patch block causal context."""

    if committed_ids.ndim != 2 or committed_ids.dtype != torch.long:
        raise ValueError("committed ids must be int64 [batch, length]")
    batch, length = committed_ids.shape
    if batch != cache.batch_size or length <= 0 or (
        length % model.config.patch_stride
    ):
        raise ValueError("committed block must be batch-aligned and patch complete")
    if bool(
        ((committed_ids < 0) | (committed_ids >= model.config.vocab.output_size)).any()
    ):
        raise ValueError("committed block contains an invalid model output id")
    starts = cache.valid.sum(1).to(torch.long)
    positions = starts[:, None] + torch.arange(
        length, device=committed_ids.device, dtype=torch.long
    )[None]
    valid = torch.ones_like(committed_ids, dtype=torch.bool)

    local_states = model.embedding(committed_ids)
    if model.ngrams is not None:
        halo = max(model.config.ngram_orders) - 1
        prefix_positions = starts[:, None] - halo + torch.arange(
            halo, device=committed_ids.device
        )[None]
        prefix = _gather_absolute_positions(
            cache.ids,
            cache.valid,
            cache.positions,
            prefix_positions,
            fill_value=model.config.vocab.pad_id,
        )
        visible = torch.cat((prefix, committed_ids), dim=1)
        local_states = model.combine_ngram_features(
            local_states, model.ngrams(visible)[:, -length:]
        )
    local_cache: list[LayerKV] = []
    for block, cached in zip(model.encoder, cache.local, strict=True):
        local_states, updated = _append_causal_block(
            local_states,
            cached,
            block,
            cached_positions=cache.positions,
            cached_valid=cache.valid,
            query_positions=positions,
            window=model.config.local_window,
        )
        local_cache.append(updated)

    patches = model.pool(local_states, valid)
    patch_starts = cache.patch_valid.sum(1).to(torch.long)
    patch_positions = patch_starts[:, None] + torch.arange(
        patches.shape[1], device=committed_ids.device, dtype=torch.long
    )[None]
    global_cache: list[LayerKV] = []
    for block, cached in zip(model.global_blocks, cache.global_, strict=True):
        patches, updated = _append_causal_block(
            patches,
            cached,
            block,
            cached_positions=cache.patch_positions,
            cached_valid=cache.patch_valid,
            query_positions=patch_positions,
            window=model.config.global_window,
        )
        global_cache.append(updated)
    normalized_new_patches = model.global_norm(patches)
    all_patch_states = torch.cat(
        (cache.patch_states, normalized_new_patches), dim=1
    )

    physical_patch = torch.div(positions, model.config.patch_stride, rounding_mode="floor")
    if cache.has_virtual_bos:
        condition_patch = physical_patch
        condition_patch = torch.where(
            positions.remainder(model.config.patch_stride).eq(
                model.config.patch_stride - 1
            ),
            physical_patch + 1,
            condition_patch,
        )
    else:
        condition_patch = physical_patch - 1
        condition_patch = torch.where(
            positions.remainder(model.config.patch_stride).eq(
                model.config.patch_stride - 1
            ),
            physical_patch,
            condition_patch,
        )
    all_patch_valid = torch.cat(
        (
            cache.patch_valid,
            torch.ones(
                (batch, normalized_new_patches.shape[1]),
                dtype=torch.bool,
                device=committed_ids.device,
            ),
        ),
        dim=1,
    )
    all_patch_positions = torch.cat(
        (cache.patch_positions, patch_positions), dim=1
    )
    condition = _gather_absolute_positions(
        all_patch_states,
        all_patch_valid,
        all_patch_positions,
        condition_patch,
    )
    decoder_states = model.decoder_embeddings(committed_ids, ModelMode.AR)
    decoder_cache: list[LayerKV] = []
    for conditioned, cached in zip(model.decoder, cache.decoder, strict=True):
        decoder_states = conditioned._add_condition(decoder_states, condition)
        decoder_states, updated = _append_causal_block(
            decoder_states,
            cached,
            conditioned.block,
            cached_positions=cache.positions,
            cached_valid=cache.valid,
            query_positions=positions,
            window=model.config.local_window,
        )
        decoder_cache.append(updated)

    return HierarchicalPrefixCache(
        ids=torch.cat((cache.ids, committed_ids), dim=1),
        valid=torch.cat((cache.valid, valid), dim=1),
        patch_valid=all_patch_valid,
        positions=torch.cat((cache.positions, positions), dim=1),
        patch_positions=all_patch_positions,
        patch_states=all_patch_states,
        local=tuple(local_cache),
        global_=tuple(global_cache),
        decoder=tuple(decoder_cache),
        has_virtual_bos=cache.has_virtual_bos,
    )


class CachedCanvasGenerator:
    """End-to-end transactional decoder with state-owned RNG and K/V reuse."""

    def __init__(
        self,
        model: ByteDiffusionModel,
        state: TransactionalDecodeState,
        *,
        allow_dense_reference: bool = False,
        blocked_output_ids: Tensor | None = None,
    ) -> None:
        if state.eot_id != model.config.vocab.eot_id:
            raise ValueError("decode-state EOT id does not match the model vocabulary")
        if state.patch_stride != model.config.patch_stride:
            raise ValueError("decode-state patch stride does not match the model")
        self.model = model
        self.state = state
        self.allow_dense_reference = allow_dense_reference
        if blocked_output_ids is not None:
            blocked_output_ids = blocked_output_ids.to(
                device=state.device, dtype=torch.long
            )
            if blocked_output_ids.ndim != 1 or bool(
                (
                    (blocked_output_ids < 0)
                    | (blocked_output_ids >= model.config.vocab.output_size)
                ).any()
            ):
                raise ValueError("blocked output ids must be valid rank-1 clean ids")
            if bool(blocked_output_ids.eq(state.eot_id).any()):
                raise ValueError("EOT cannot be blocked during generation")
        self.blocked_output_ids = blocked_output_ids
        self.cache: HierarchicalPrefixCache | None = None

    def _apply_output_mask(self, logits: Tensor) -> Tensor:
        if self.blocked_output_ids is None or self.blocked_output_ids.numel() == 0:
            return logits
        # Decoder logits are ephemeral and never reused after sampling.
        logits[..., self.blocked_output_ids] = -torch.inf
        return logits

    def prefill(self, prefix: Tensor) -> None:
        if prefix.ndim != 1 or prefix.dtype != torch.long or prefix.numel() == 0:
            raise ValueError("generation prefix must be nonempty rank-1 int64")
        if bool(((prefix < 0) | (prefix >= self.model.config.vocab.output_size)).any()):
            raise ValueError("generation prefix contains an input-only or invalid id")
        eot = prefix.eq(self.state.eot_id).nonzero()
        if eot.numel() and int(eot[0]) != prefix.numel() - 1:
            raise ValueError("EOT may appear only as the terminal prefix atom")
        self.state.replace_prefix(prefix)
        self.cache = None
        if self.state.patch_phase == 0 and int(self.state.ids[-1]) != self.state.eot_id:
            self.cache = prefill_prefix(
                self.model,
                self.state.ids[None],
                allow_dense_reference=self.allow_dense_reference,
            )
            self.state.note_prefill(replay=False)

    @torch.no_grad()
    def align_prefix_ar(self) -> Tensor:
        """Take at most three exact AR steps before opening a canvas."""

        generated: list[Tensor] = []
        while self.state.patch_phase:
            length = self.state.ids.numel()
            width = length + (-length) % self.model.config.patch_stride
            storage = torch.full(
                (1, width),
                self.model.config.vocab.pad_id,
                dtype=torch.long,
                device=self.state.ids.device,
            )
            storage[0, :length] = self.state.ids
            valid = torch.arange(width, device=storage.device)[None] < length
            positions = torch.arange(width, device=storage.device)[None]
            output = self.model.forward_ar_varlen(
                storage,
                valid,
                positions=positions,
                allow_dense_reference=self.allow_dense_reference,
                **document_start_ar_metadata(
                    valid, self.model.config.patch_stride
                ),
            )
            probabilities = self._apply_output_mask(
                output.logits[0, length - 1].float()
            ).softmax(-1)
            sampled = torch.multinomial(
                probabilities, 1, generator=self.state.generator
            ).to(torch.long)
            self.state.begin()
            self.state.note_work(1)
            accepted = self.state.commit_ids(sampled)
            generated.append(accepted)
            if int(accepted[-1]) == self.state.eot_id:
                self.cache = None
                break
        if (
            self.cache is None
            and self.state.ids.numel()
            and int(self.state.ids[-1]) != self.state.eot_id
        ):
            self.cache = prefill_prefix(
                self.model,
                self.state.ids[None],
                allow_dense_reference=self.allow_dense_reference,
            )
            self.state.note_prefill(replay=True)
        return (
            torch.cat(generated)
            if generated
            else self.state.ids.new_empty((0,))
        )

    def generate(self, canvas_length: int, steps: int) -> CanvasGeneration:
        if canvas_length <= 0 or canvas_length % self.model.config.patch_stride:
            raise ValueError("canvas length must be a positive patch multiple")
        if steps <= 0:
            raise ValueError("diffusion steps must be positive")
        if self.state.ids.numel() == 0:
            raise RuntimeError("prefill must run before cached generation")
        alignment = self.align_prefix_ar()
        if self.state.ids.numel() and int(self.state.ids[-1]) == self.state.eot_id:
            return CanvasGeneration(
                alignment_ids=alignment,
                canvas=None,
                committed_canvas_ids=self.state.ids.new_empty((0,)),
            )
        if self.cache is None:
            raise RuntimeError("prefill must run before cached generation")
        start = self.state.ids.numel()
        # Branch positions extend beyond the clean K/V bank directly; no fake
        # future clean atoms are embedded or pooled.
        cache = self.cache
        starts = torch.tensor([[start]], dtype=torch.long, device=self.state.ids.device)
        plan = prepare_cached_canvas(
            self.model,
            cache,
            starts,
            canvas_length,
            allow_dense_reference=self.allow_dense_reference,
        )
        initial = torch.full(
            (canvas_length,),
            self.model.config.vocab.mask_id,
            dtype=torch.long,
            device=self.state.ids.device,
        )
        self.state.begin()
        try:
            sample = sample_absorbing_canvas(
                initial,
                lambda ids: self._apply_output_mask(
                    denoise_canvas_cached(
                        self.model,
                        cache,
                        ids[None, None],
                        starts,
                        allow_dense_reference=self.allow_dense_reference,
                        plan=plan,
                    )[0]
                ),
                steps=steps,
                mask_id=self.model.config.vocab.mask_id,
                eot_id=self.model.config.vocab.eot_id,
                generator=self.state.generator,
                strategy="fixed_quota",
            )
            self.state.note_work(
                sample.executed_nfe * canvas_length,
                forwards=sample.executed_nfe,
                denoise=True,
            )
            accepted = self.state.commit_ids(sample.ids)
        except Exception:
            self.state.abort()
            raise
        # Explicit causal replay: scratch branch K/V is discarded and the
        # accepted semantic prefix is re-prefilled into committed typed caches.
        replay_padding = (-self.state.ids.numel()) % self.model.config.patch_stride
        replay_ids = self.state.ids
        if replay_padding:
            replay_ids = torch.cat(
                (
                    replay_ids,
                    torch.full(
                        (replay_padding,),
                        self.model.config.vocab.pad_id,
                        dtype=torch.long,
                        device=replay_ids.device,
                    ),
                )
            )
        replay_valid = torch.arange(
            replay_ids.numel(), device=replay_ids.device
        )[None] < self.state.ids.numel()
        self.cache = prefill_prefix(
            self.model,
            replay_ids[None],
            valid=replay_valid,
            allow_dense_reference=self.allow_dense_reference,
        )
        self.state.note_prefill(replay=True)
        self.state.counters.rejected_bytes += canvas_length - accepted.numel()
        return CanvasGeneration(
            alignment_ids=alignment,
            canvas=sample,
            committed_canvas_ids=accepted,
        )

    def generate_blt(
        self,
        block_length: int,
        steps: int,
        *,
        strategy: UnmaskingStrategy = "confidence",
        confidence_threshold: float = 0.7,
        entropy_budget: float = 1.0,
    ) -> CanvasGeneration:
        """Generate and incrementally commit one Fast-BLT block."""

        if block_length <= 0 or block_length % self.model.config.patch_stride:
            raise ValueError("BLT block length must be a positive patch multiple")
        if steps <= 0:
            raise ValueError("diffusion steps must be positive")
        if self.state.ids.numel() == 0:
            raise RuntimeError("prefill must run before cached generation")
        alignment = self.align_prefix_ar()
        if self.state.ids.numel() and int(self.state.ids[-1]) == self.state.eot_id:
            return CanvasGeneration(
                alignment_ids=alignment,
                canvas=None,
                committed_canvas_ids=self.state.ids.new_empty((0,)),
            )
        if self.cache is None:
            raise RuntimeError("prefill must establish a patch-aligned cache")
        cache = self.cache
        starts = torch.tensor(
            [[cache.length]], dtype=torch.long, device=self.state.ids.device
        )
        plan = prepare_cached_canvas(
            self.model,
            cache,
            starts,
            block_length,
            allow_dense_reference=self.allow_dense_reference,
            include_global=False,
        )
        initial = torch.full(
            (block_length,),
            self.model.config.vocab.mask_id,
            dtype=torch.long,
            device=self.state.ids.device,
        )
        self.state.begin()
        try:
            sample = sample_absorbing_canvas(
                initial,
                lambda ids: self._apply_output_mask(
                    denoise_blt_cached(
                        self.model,
                        cache,
                        ids[None, None],
                        starts,
                        allow_dense_reference=self.allow_dense_reference,
                        plan=plan,
                    )[0]
                ),
                steps=steps,
                mask_id=self.model.config.vocab.mask_id,
                eot_id=self.model.config.vocab.eot_id,
                generator=self.state.generator,
                strategy=strategy,
                confidence_threshold=confidence_threshold,
                entropy_budget=entropy_budget,
            )
            self.state.note_work(
                sample.executed_nfe * block_length,
                forwards=sample.executed_nfe,
                denoise=True,
            )
            accepted = self.state.commit_ids(sample.ids)
        except Exception:
            self.state.abort()
            raise
        if accepted.numel() == block_length and int(accepted[-1]) != self.state.eot_id:
            self.cache = append_clean_block(
                self.model, cache, accepted[None]
            )
            self.state.note_cache_append()
        else:
            # EOT truncates the proposal and terminates this trajectory; no
            # partially committed cache is exposed.
            self.cache = None
        return CanvasGeneration(
            alignment_ids=alignment,
            canvas=sample,
            committed_canvas_ids=accepted,
        )


__all__ = [
    "CachedCanvasGenerator",
    "CachedCanvasPlan",
    "CanvasGeneration",
    "EntropyCanvasGeneration",
    "EntropyPatchedCanvasGenerator",
    "HierarchicalPrefixCache",
    "LayerKV",
    "append_clean_block",
    "denoise_blt_cached",
    "denoise_canvas_cached",
    "denoise_entropy_blt_reference",
    "document_start_ar_metadata",
    "entropy_ar_next_logits",
    "entropy_next_byte_starts_patch",
    "prefill_prefix",
    "prepare_cached_canvas",
]
