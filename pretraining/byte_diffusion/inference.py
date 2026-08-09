"""Typed hierarchical K/V caches and cached absorbing-canvas generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .attention import CanvasBranchLayout, branch_attention, build_canvas_block_mask
from .config import ModelMode
from .layers import PackedSelfAttention, apply_rotary
from .model import ByteDiffusionModel
from .sampling import CanvasSample, sample_absorbing_canvas
from .state import TransactionalDecodeState


@dataclass(frozen=True)
class LayerKV:
    """RoPE-applied K and raw V for one clean-prefix attention layer."""

    key: Tensor
    value: Tensor


@dataclass(frozen=True)
class HierarchicalPrefixCache:
    """Immutable clean-prefix state reused by every canvas denoising NFE."""

    ids: Tensor
    valid: Tensor
    patch_valid: Tensor
    positions: Tensor
    patch_positions: Tensor
    local: tuple[LayerKV, ...]
    global_: tuple[LayerKV, ...]
    decoder: tuple[LayerKV, ...]

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


@torch.no_grad()
def prefill_prefix(
    model: ByteDiffusionModel,
    ids: Tensor,
    *,
    valid: Tensor | None = None,
    positions: Tensor | None = None,
    allow_dense_reference: bool = False,
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

    local_states = model.embedding(ids)
    if model.ngrams is not None:
        local_states = local_states + model.ngrams(ids)
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
        local_states = local_states.new_zeros(local_states.shape).masked_scatter(
            valid[..., None], packed
        )

    patches = model.pool(local_states, valid)
    patch_length = patches.shape[1]
    patch_valid = valid.view(
        batch, patch_length, model.config.patch_stride
    ).any(-1)
    patch_positions = (
        positions[:, :: model.config.patch_stride] // model.config.patch_stride
    )
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
        patches = patches.new_zeros(patches.shape).masked_scatter(
            patch_valid[..., None], packed
        )
    normalized_patches = patches.new_zeros(patches.shape)
    normalized_patches[patch_valid] = model.global_norm(patches[patch_valid])

    condition = model._aligned_condition(normalized_patches, length, ModelMode.AR)
    decoder_states = model.embedding(ids) + model.mode_embedding.weight[int(ModelMode.AR)]
    decoder_cache: list[LayerKV] = []
    for conditioned in model.decoder:
        decoder_states = decoder_states + conditioned.condition_norm(
            conditioned.condition(condition)
        )
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
        decoder_states = decoder_states.new_zeros(decoder_states.shape).masked_scatter(
            valid[..., None], packed_states
        )

    return HierarchicalPrefixCache(
        ids=ids.clone(),
        valid=valid,
        patch_valid=patch_valid,
        positions=positions.clone(),
        patch_positions=patch_positions.clone(),
        local=tuple(local_cache),
        global_=tuple(global_cache),
        decoder=tuple(decoder_cache),
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
        states = states + conditioned_block.condition_norm(
            conditioned_block.condition(condition)
        )
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
    branch_patch_positions = patch_starts[:, :, None] + torch.arange(
        canvas_patches, device=starts.device
    )[None, None, :]
    global_layout = CanvasBranchLayout(
        clean_valid=cache.patch_valid,
        branch_valid=torch.ones(
            batch, 1, canvas_patches, dtype=torch.bool, device=starts.device
        ),
        prefix_lengths=patch_starts,
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
        states = states + model.ngrams(visible)[:, -canvas_length:]
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


class CachedCanvasGenerator:
    """End-to-end transactional decoder with state-owned RNG and K/V reuse."""

    def __init__(
        self,
        model: ByteDiffusionModel,
        state: TransactionalDecodeState,
        *,
        allow_dense_reference: bool = False,
    ) -> None:
        if state.eot_id != model.config.vocab.eot_id:
            raise ValueError("decode-state EOT id does not match the model vocabulary")
        if state.patch_stride != model.config.patch_stride:
            raise ValueError("decode-state patch stride does not match the model")
        self.model = model
        self.state = state
        self.allow_dense_reference = allow_dense_reference
        self.cache: HierarchicalPrefixCache | None = None

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
            )
            probabilities = output.logits[0, length - 1].float().softmax(-1)
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
                lambda ids: denoise_canvas_cached(
                    self.model,
                    cache,
                    ids[None, None],
                    starts,
                    allow_dense_reference=self.allow_dense_reference,
                    plan=plan,
                )[0],
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


__all__ = [
    "CachedCanvasGenerator",
    "CachedCanvasPlan",
    "CanvasGeneration",
    "HierarchicalPrefixCache",
    "LayerKV",
    "denoise_canvas_cached",
    "prefill_prefix",
    "prepare_cached_canvas",
]
