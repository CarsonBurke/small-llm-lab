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
from .sampling import CanvasSample, sample_absorbing_canvas
from .state import TransactionalDecodeState


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
    branch_patch_positions: Tensor
    local_layout: CanvasBranchLayout
    global_layout: CanvasBranchLayout
    local_block_mask: object | None
    global_block_mask: object | None


@dataclass(frozen=True)
class CanvasGeneration:
    alignment_ids: Tensor
    canvas: CanvasSample | None
    committed_canvas_ids: Tensor


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


def _document_start_ar_metadata(valid: Tensor, patch_stride: int) -> dict[str, Tensor]:
    """Build the single-document virtual-BOS map used by AR alignment."""

    if valid.ndim != 2 or valid.shape[0] != 1:
        raise ValueError("cached AR alignment requires exactly one document")
    length = int(valid.sum())
    physical_patches = (length + patch_stride - 1) // patch_stride
    device = valid.device
    byte_indices = torch.arange(length, dtype=torch.long, device=device)
    patch_indices = torch.arange(
        physical_patches, dtype=torch.long, device=device
    )
    byte_positions = byte_indices
    physical_patch = torch.div(
        byte_positions, patch_stride, rounding_mode="floor"
    )
    # Global packed index zero is virtual BOS; physical patch p is p + 1.
    condition_patch_indices = torch.where(
        byte_positions.remainder(patch_stride).eq(patch_stride - 1),
        physical_patch + 1,
        physical_patch,
    )
    return {
        "byte_indices": byte_indices,
        "byte_cu_seqlens": torch.tensor(
            [0, length], dtype=torch.int32, device=device
        ),
        "patch_indices": patch_indices,
        "patch_cu_seqlens": torch.tensor(
            [0, physical_patches + 1], dtype=torch.int32, device=device
        ),
        "condition_patch_indices": condition_patch_indices,
        "global_patch_sources": torch.cat(
            (
                torch.full((1,), -1, dtype=torch.long, device=device),
                patch_indices,
            )
        ),
        "global_patch_positions": torch.arange(
            physical_patches + 1, dtype=torch.long, device=device
        ),
        "physical_to_global_patch_indices": patch_indices + 1,
        "bos_condition_indices": torch.zeros(1, dtype=torch.long, device=device),
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
            window=None,
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
    decoder_states = model.embedding(ids) + model.mode_embedding.weight[int(ModelMode.AR)]
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
        prefix_lengths=starts,
        prefix_window=model.config.local_window,
        clean_positions=cache.positions,
        branch_positions=branch_positions,
    )
    local_mask = None if allow_dense_reference else build_canvas_block_mask(local_layout)
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
        prefix_lengths=global_patch_starts,
    )
    global_mask = None if allow_dense_reference else build_canvas_block_mask(global_layout)
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
        prefix = torch.gather(cache.ids, 1, indices.clamp_min(0))
        prefix = torch.where(indices >= 0, prefix, model.config.vocab.pad_id)
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
    states = model.embedding(noisy_ids[:, 0]) + model.mode_embedding.weight[
        int(ModelMode.CANVAS)
    ]
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
        )
    flat_positions = plan.branch_positions[:, 0]
    prior_patch = (
        starts[:, 0] // model.config.patch_stride
        if cache.has_virtual_bos
        else starts[:, 0] // model.config.patch_stride - 1
    )
    safe_patch = prior_patch.clamp_min(0)
    prior_state = cache.patch_states[
        torch.arange(batch, device=noisy_ids.device), safe_patch
    ]
    prior_state = torch.where(
        prior_patch[:, None].ge(0), prior_state, torch.zeros_like(prior_state)
    )
    condition = prior_state[:, None, :].expand(-1, block_length, -1)
    states = model.embedding(noisy_ids[:, 0]) + model.mode_embedding.weight[
        int(ModelMode.BLT_D)
    ]
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
    start = cache.length
    positions = start + torch.arange(
        length, device=committed_ids.device, dtype=torch.long
    )[None].expand(batch, -1)
    valid = torch.ones_like(committed_ids, dtype=torch.bool)

    local_states = model.embedding(committed_ids)
    if model.ngrams is not None:
        halo = max(model.config.ngram_orders) - 1
        visible = torch.cat((cache.ids[:, -halo:], committed_ids), dim=1)
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
            query_positions=positions,
            window=model.config.local_window,
        )
        local_cache.append(updated)

    patches = model.pool(local_states, valid)
    patch_start = cache.patch_positions.shape[1]
    patch_positions = patch_start + torch.arange(
        patches.shape[1], device=committed_ids.device, dtype=torch.long
    )[None].expand(batch, -1)
    global_cache: list[LayerKV] = []
    for block, cached in zip(model.global_blocks, cache.global_, strict=True):
        patches, updated = _append_causal_block(
            patches,
            cached,
            block,
            cached_positions=cache.patch_positions,
            query_positions=patch_positions,
            window=None,
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
    condition = torch.gather(
        all_patch_states,
        1,
        condition_patch[..., None].expand(-1, -1, model.config.global_dim),
    )
    decoder_states = model.embedding(committed_ids) + model.mode_embedding.weight[
        int(ModelMode.AR)
    ]
    decoder_cache: list[LayerKV] = []
    for conditioned, cached in zip(model.decoder, cache.decoder, strict=True):
        decoder_states = conditioned._add_condition(decoder_states, condition)
        decoder_states, updated = _append_causal_block(
            decoder_states,
            cached,
            conditioned.block,
            cached_positions=cache.positions,
            query_positions=positions,
            window=model.config.local_window,
        )
        decoder_cache.append(updated)

    return HierarchicalPrefixCache(
        ids=torch.cat((cache.ids, committed_ids), dim=1),
        valid=torch.cat((cache.valid, valid), dim=1),
        patch_valid=torch.cat(
            (
                cache.patch_valid,
                torch.ones(
                    (batch, normalized_new_patches.shape[1]),
                    dtype=torch.bool,
                    device=committed_ids.device,
                ),
            ),
            dim=1,
        ),
        positions=torch.cat((cache.positions, positions), dim=1),
        patch_positions=torch.cat(
            (cache.patch_positions, patch_positions), dim=1
        ),
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
                **_document_start_ar_metadata(
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
        adaptive_confidence: float | None = None,
        min_steps: int = 1,
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
                adaptive_confidence=adaptive_confidence,
                min_steps=min_steps,
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
    "HierarchicalPrefixCache",
    "LayerKV",
    "append_clean_block",
    "denoise_blt_cached",
    "denoise_canvas_cached",
    "prefill_prefix",
    "prepare_cached_canvas",
]
