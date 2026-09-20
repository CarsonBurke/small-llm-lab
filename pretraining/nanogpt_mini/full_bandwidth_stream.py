"""Token-sequential, document-batched FullBandwidthGPT execution on CUDA.

Every call performs one real transformer step per document lane. Incoming carry
and historical K/V are detached; the complete current-token computation,
including its current K/V, is trainable. The caller compiles ``stream_forward``
or an enclosing loss function with ``fullgraph=True, dynamic=False`` and calls
``StreamState.commit`` only after that step's backward has completed.

There is no producer/Jacobi pass, history concatenation, or history-sized cache
update. Persistent BF16 K/V storage costs ``4 * layers * batch * capacity * kv_dim``
bytes, plus ``2 * batch * model_dim`` bytes of carry and ``8 * batch`` of positions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor

from pretraining.nanogpt_mini.full_bandwidth_stream_attention import cached_attention
from pretraining.nanogpt_mini.nanogpt_mini_full_bandwidth_model import FullBandwidthGPT


@triton.jit
def _commit_state(
    KEY_CACHE,
    VALUE_CACHE,
    PREVIOUS,
    POSITIONS,
    HIDDEN,
    NEW_KEYS,
    NEW_VALUES,
    EFFECTIVE_POSITIONS,
    BATCH: tl.constexpr,
    HEADS: tl.constexpr,
    WIDTH: tl.constexpr,
    CAPACITY: tl.constexpr,
    CARRY_WIDTH: tl.constexpr,
    BLOCK_CARRY: tl.constexpr,
):
    bh = tl.program_id(0)
    layer = tl.program_id(1)
    lane = bh // HEADS
    head = bh % HEADS
    position = tl.load(EFFECTIVE_POSITIONS + lane)
    slot = position % CAPACITY
    d = tl.arange(0, WIDTH)
    current_offset = (layer.to(tl.int64) * BATCH * HEADS + bh) * WIDTH + d
    cache_offset = (
        (layer.to(tl.int64) * BATCH * HEADS + bh) * CAPACITY + slot
    ) * WIDTH + d
    tl.store(KEY_CACHE + cache_offset, tl.load(NEW_KEYS + current_offset))
    tl.store(VALUE_CACHE + cache_offset, tl.load(NEW_VALUES + current_offset))
    if layer == 0:
        # A KV head owns its query group's carry channels, not just head_dim.
        # These widths coincide for MHA; shared-KV still commits every D channel.
        carry_d = tl.arange(0, BLOCK_CARRY)
        carry_offset = bh.to(tl.int64) * CARRY_WIDTH + carry_d
        carry = tl.load(HIDDEN + carry_offset, mask=carry_d < CARRY_WIDTH, other=0.0)
        tl.store(PREVIOUS + carry_offset, carry, mask=carry_d < CARRY_WIDTH)
        if head == 0:
            # All programs read the separate effective-position snapshot, never
            # POSITIONS, so this write cannot race another layer/head's reads.
            tl.store(POSITIONS + lane, position + 1)


class StreamState:
    """Fixed-address recurrent carry and ring caches for independent documents.

    ``positions[b]`` is the number of committed tokens since the latest BOS in
    lane b. A BOS is supplied explicitly through ``resets[b]`` at both forward
    and commit. Resetting a lane changes its logical history length and RoPE
    origin, not its cache allocation. Stale slots remain physically present but
    cannot be read until replaced with tokens from the new document.

    ``keys``/``values`` are per-layer tuple views of contiguous backing buffers.
    The commit kernel receives those backing buffers directly, allowing compiled
    mutation to be reinplaced without materializing or copying a whole history.
    State is single-stream: do not overlap forward/backward/commit on one instance.
    """

    def __init__(
        self,
        model: FullBandwidthGPT,
        batch_size: int,
        capacity: int,
        device: torch.device | str,
    ) -> None:
        for name, value in (("batch_size", batch_size), ("capacity", capacity)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("StreamState requires CUDA; there is no CPU fallback")
        self.batch_size = batch_size
        self.capacity = capacity
        self.num_layers = len(model.blocks)
        self.model_dim = model.model_dim
        self.num_heads = model.blocks[0].attn.num_heads
        self.num_kv_heads = model.blocks[0].attn.num_kv_heads
        self.head_dim = model.blocks[0].attn.head_dim
        if self.num_heads * self.head_dim != self.model_dim:
            raise ValueError("Mini attention width must equal model_dim")
        if self.num_kv_heads < 1 or self.num_heads % self.num_kv_heads:
            raise ValueError("Mini K/V heads must divide query heads")
        self.kv_dim = self.num_kv_heads * self.head_dim
        for block in model.blocks:
            if (
                block.attn.num_heads != self.num_heads
                or block.attn.head_dim != self.head_dim
                or block.attn.num_kv_heads != self.num_kv_heads
            ):
                raise ValueError("StreamState requires uniform Mini attention heads")
        shape = (
            self.num_layers,
            batch_size,
            self.num_kv_heads,
            capacity,
            self.head_dim,
        )
        self._keys = torch.empty(shape, dtype=torch.bfloat16, device=device)
        self._values = torch.empty(shape, dtype=torch.bfloat16, device=device)
        self.keys = self._keys.unbind(0)
        self.values = self._values.unbind(0)
        self.previous = torch.zeros(
            (batch_size, self.model_dim), dtype=torch.bfloat16, device=device
        )
        self.positions = torch.zeros(batch_size, dtype=torch.int64, device=device)
        # The training objective and commit are separate compiled graphs. These
        # buffers survive both and must never be treated as ephemeral graph input
        # copies, particularly when CUDA graph replay includes input mutation.
        for tensor in (
            self._keys,
            self._values,
            self.previous,
            self.positions,
            *self.keys,
            *self.values,
        ):
            torch._dynamo.mark_static_address(tensor)

    @torch.no_grad()
    @torch.compile(fullgraph=True, dynamic=False)
    def reset(self) -> None:
        """Start an empty stream without clearing or reallocating K/V storage.

        As with commit, no outstanding backward may still refer to this state.
        The next input in each lane must be its actual BOS (``resets=True``).
        """
        self.previous.zero_()
        self.positions.zero_()

    @torch.no_grad()
    @torch.compile(fullgraph=True, dynamic=False)
    def commit(
        self,
        hidden: Tensor,
        new_keys: Tensor,
        new_values: Tensor,
        resets: Tensor,
    ) -> None:
        """Commit the completed step IN PLACE, strictly AFTER local backward.

        Only one K/V slot per layer/lane/KV head is written. Current hidden and K/V
        are detached here, so no graph survives into the next token. ``resets``
        must be the same BOS mask used for the corresponding ``stream_forward``.
        """
        current_shape = (
            self.num_layers,
            self.batch_size,
            self.num_kv_heads,
            self.head_dim,
        )
        if hidden.shape != (self.batch_size, self.model_dim):
            raise ValueError("hidden must have shape [batch, model_dim]")
        if new_keys.shape != current_shape or new_values.shape != current_shape:
            raise ValueError(
                "new K/V must have shape [layers, batch, kv_heads, head_dim]"
            )
        if resets.shape != (self.batch_size,) or resets.dtype != torch.bool:
            raise ValueError("resets must be bool [batch]")
        for tensor in (hidden, new_keys, new_values):
            if tensor.dtype != torch.bfloat16 or tensor.device != self.previous.device:
                raise ValueError("hidden and new K/V must be BF16 on the state device")
        if resets.device != self.positions.device:
            raise ValueError("resets must be on the state device")
        effective_positions = torch.where(resets, 0, self.positions)
        _commit_state[(self.batch_size * self.num_kv_heads, self.num_layers)](
            self._keys,
            self._values,
            self.previous,
            self.positions,
            hidden.detach().contiguous(),
            new_keys.detach().contiguous(),
            new_values.detach().contiguous(),
            effective_positions,
            BATCH=self.batch_size,
            HEADS=self.num_kv_heads,
            WIDTH=self.head_dim,
            CAPACITY=self.capacity,
            CARRY_WIDTH=self.model_dim // self.num_kv_heads,
            BLOCK_CARRY=triton.next_power_of_2(self.model_dim // self.num_kv_heads),
            num_warps=4,
        )


def _rotary_at_positions(x: Tensor, angular_freq: Tensor, positions: Tensor) -> Tensor:
    """The Mini split-half, half-truncated RoPE at per-document positions."""
    theta = positions.to(torch.float32)[:, None, None] * angular_freq[None, None, :]
    cosine, sine = theta.cos(), theta.sin()
    first, second = x.float().chunk(2, dim=-1)
    return torch.cat(
        (first * cosine + second * sine, first * (-sine) + second * cosine), dim=-1
    ).type_as(x)


def stream_forward(
    model: FullBandwidthGPT,
    tokens: Tensor,
    previous: Tensor,
    keys: tuple[Tensor, ...],
    values: tuple[Tensor, ...],
    positions: Tensor,
    resets: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """One differentiable current-token step with detached temporal boundaries.

    Inputs: tokens ``[B]``, previous ``[B,D]``, per-layer K/V ``[B,Hkv,C,d]``,
    positions ``int64[B]``, and actual-document-BOS resets ``bool[B]``.
    Returns FP32 logits ``[B,V]``, BF16 final-normalized hidden ``[B,D]``,
    and attached current keys/values, each BF16 ``[L,B,Hkv,d]``.

    This function never mutates state. Cache tensors and current outputs must
    remain alive and unmodified through backward. ``capacity`` counts the current
    token: attention uses at most ``capacity-1`` previous tokens even though the
    ring stores ``capacity`` committed tokens between steps. Per-lane positions
    stay as tensor values, including after wrap or reset; they are never Python
    guards or compilation specializations.
    """
    if tokens.ndim != 1 or tokens.shape[0] < 1:
        raise ValueError("tokens must have nonempty shape [batch]")
    batch = tokens.shape[0]
    if tokens.device.type != "cuda":
        raise ValueError("stream_forward requires CUDA; there is no CPU fallback")
    if tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("tokens must be int32 or int64")
    if previous.shape != (batch, model.model_dim) or previous.dtype != torch.bfloat16:
        raise ValueError("previous must be BF16 [batch, model_dim]")
    if positions.shape != (batch,) or positions.dtype != torch.int64:
        raise ValueError("positions must be int64 [batch]")
    if resets.shape != (batch,) or resets.dtype != torch.bool:
        raise ValueError("resets must be bool [batch]")
    if any(tensor.device != tokens.device for tensor in (previous, positions, resets)):
        raise ValueError("stream inputs must be on the same CUDA device")
    if len(keys) != len(model.blocks) or len(values) != len(model.blocks):
        raise ValueError("history must contain one key/value tensor per layer")

    effective_positions = torch.where(resets, 0, positions.detach())
    x = model.recurrent_input(tokens, previous.detach(), resets)
    current_keys, current_values = [], []
    for block, key_cache, value_cache in zip(model.blocks, keys, values):
        attention = block.attn
        normalized = block.norm1(x)
        query_shape = (batch, attention.num_heads, attention.head_dim)
        kv_shape = (batch, attention.num_kv_heads, attention.head_dim)
        q = F.rms_norm(attention.q(normalized).view(query_shape), (attention.head_dim,))
        k = F.rms_norm(attention.k(normalized).view(kv_shape), (attention.head_dim,))
        v = attention.v(normalized).view(kv_shape)
        # SDPA autocasts Q/K/V to BF16 after FP32 normalization/RoPE.
        # Our custom kernel has no autocast dispatcher, so make that boundary explicit.
        q = _rotary_at_positions(
            q, attention.rotary.angular_freq, effective_positions
        ).bfloat16()
        k = _rotary_at_positions(
            k, attention.rotary.angular_freq, effective_positions
        ).bfloat16()
        attended = cached_attention(
            q, k, v, key_cache, value_cache, effective_positions, scale=0.12
        )
        x = x + model.residual_scale * attention.proj(
            attended.reshape(batch, 1, model.model_dim)
        )
        x = x + model.residual_scale * block.mlp(block.norm2(x))
        current_keys.append(k)
        current_values.append(v)
    hidden = model.norm2(x).squeeze(1).bfloat16()
    return (
        model.logits(hidden),
        hidden,
        torch.stack(current_keys),
        torch.stack(current_values),
    )


__all__ = ["StreamState", "stream_forward"]
