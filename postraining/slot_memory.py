"""Slot-choice latent memory over the stored token-carry producer stream.

The policy keeps ``M`` overwrite slots per rollout lane. At response step ``t``
it samples a joint action ``(x_t, sigma_t)``: the vocabulary token and a slot
index in ``{0..M-1}`` or the null choice ``NO_WRITE`` (``-1`` on the host,
logit index ``M`` in the head). The detached post-final-norm producer ``m_t``
is written into slot ``sigma_t`` by pure overwrite, then the consumed token
``x_t`` reads a softmax over the alive slot entries. Keys receive rotary
position at write time and queries at read time, so relative age is native to
the score. A learned null key with a zero value lets every read attend to
"nothing", which also makes the empty memory at step zero well defined.

Replay never re-simulates the memory: the recorded ``sigma`` sequence defines
``alive[t, s]`` (the response index of the producer occupying slot ``s`` after
write ``t``) and key ``i`` is visible to query ``t`` iff
``alive[t, sigma_i] == i``. That rule is a FlexAttention ``mask_mod`` over the
``(T, M)`` table with block structure derived from the same table, so replay
stays block-sparse with at most ``M`` keys per query. Actor and critic consume
the identical recorded contract through independent combiners.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention.flex_attention import (
    AuxRequest,
    BlockMask,
    flex_attention,
)

NO_WRITE = -1
MAX_SLOTS = 32_767  # trajectory records store slot choices as int16
SLOT_MEMORY_SCHEMA = "minicpm5_vapo_slot_memory/v1"
DEFAULT_SLOT_BLOCK_SIZE = 128


@dataclass(frozen=True)
class SlotMemoryConfig:
    """Checkpoint- and resume-invariant slot memory geometry."""

    slots: int = 64
    heads: int = 1
    head_dim: int = 128
    rope_base: float = 10_000.0

    def __post_init__(self) -> None:
        if type(self.slots) is not int or self.slots < 1:
            raise ValueError("slot memory needs a positive integer slot count")
        if self.slots > MAX_SLOTS:
            raise ValueError(f"slot memory stores choices as int16; at most {MAX_SLOTS} slots")
        if type(self.heads) is not int or self.heads < 1:
            raise ValueError("slot memory needs a positive integer head count")
        if type(self.head_dim) is not int or self.head_dim < 2 or self.head_dim % 2:
            raise ValueError("slot memory head dimension must be a positive even integer")
        if not (type(self.rope_base) is float and math.isfinite(self.rope_base) and self.rope_base > 1):
            raise ValueError("slot memory rotary base must be a finite float above one")

    @property
    def width(self) -> int:
        return self.heads * self.head_dim

    @property
    def null_index(self) -> int:
        return self.slots

    def payload(self) -> dict[str, Any]:
        return {
            "slots": self.slots,
            "heads": self.heads,
            "head_dim": self.head_dim,
            "rope_base": self.rope_base,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> "SlotMemoryConfig":
        if not isinstance(payload, dict) or set(payload) != {"slots", "heads", "head_dim", "rope_base"}:
            raise ValueError("slot memory payload must carry slots, heads, head_dim, rope_base")
        return cls(
            slots=payload["slots"], heads=payload["heads"],
            head_dim=payload["head_dim"], rope_base=payload["rope_base"],
        )


def rotary_cos_sin(positions: Tensor, head_dim: int, base: float) -> tuple[Tensor, Tensor]:
    """FP32 rotate-half tables for integer write/read positions."""
    if positions.dtype != torch.long or positions.ndim != 1:
        raise ValueError("rotary positions must be a long vector")
    half = head_dim // 2
    inverse = base ** (
        -torch.arange(0, half, device=positions.device, dtype=torch.float32) / half
    )
    angles = positions.float()[:, None] * inverse[None]
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate ``[N, heads, D]`` by per-row tables ``[N, D]`` in FP32."""
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    result = x.float() * cos[:, None] + rotated.float() * sin[:, None]
    return result.to(x.dtype)


def build_alive_table(slot_choices: Tensor, slots: int, *, offset: int = 0) -> Tensor:
    """``alive[t, s]`` is ``offset`` + the latest write index ``<= t`` into ``s``.

    Empty slots hold ``NO_WRITE``. ``offset`` shifts occupant indices into a
    packed global key space so a packed table never crosses trajectories.
    """
    if slot_choices.ndim != 1 or slot_choices.dtype.is_floating_point:
        raise ValueError("slot choices must be an integer vector")
    if slot_choices.numel() and (
        int(slot_choices.min()) < NO_WRITE or int(slot_choices.max()) >= slots
    ):
        raise ValueError("slot choices must lie in [-1, slots)")
    length = slot_choices.numel()
    device = slot_choices.device
    steps = torch.arange(length, device=device, dtype=torch.long)
    lanes = torch.arange(slots, device=device, dtype=torch.long)
    writes = torch.where(
        slot_choices.long()[:, None] == lanes[None], steps[:, None], torch.full((), NO_WRITE, device=device)
    )
    if not length:
        return writes
    alive = writes.cummax(dim=0).values
    return torch.where(alive >= 0, alive + offset, alive)


def visibility_reference(alive_table: Tensor, key_choices: Tensor) -> Tensor:
    """Dense ``[Q, K]`` visibility: key ``k`` visible to ``q`` iff ``alive[q, sigma_k] == k``.

    Explicit loop-free reference used by tests and the dense read.
    """
    if alive_table.ndim != 2 or key_choices.ndim != 1:
        raise ValueError("alive table must be [Q, M] and key choices [K]")
    written = key_choices >= 0
    occupant = alive_table[:, key_choices.clamp_min(0)]
    keys = torch.arange(key_choices.numel(), device=key_choices.device)
    return written[None] & (occupant == keys[None])


def slot_mask_mod(alive_table: Tensor, key_choices: Tensor) -> Callable[..., Tensor]:
    """FlexAttention ``mask_mod`` over the flattened ``(Q, M)`` table."""
    if alive_table.ndim != 2 or key_choices.ndim != 1:
        raise ValueError("alive table must be [Q, M] and key choices [K]")
    slots = alive_table.shape[1]
    alive_flat = alive_table.reshape(-1)

    def mask_mod(batch: Tensor, head: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        del batch, head
        choice = key_choices[kv_idx]
        occupant = alive_flat[q_idx * slots + choice.clamp_min(0)]
        return (choice >= 0) & (occupant == kv_idx)

    return mask_mod


def slot_block_mask(
    alive_table: Tensor,
    key_choices: Tensor,
    *,
    block_size: int = DEFAULT_SLOT_BLOCK_SIZE,
) -> BlockMask:
    """Block-sparse mask whose structure comes from the table, not a dense scan.

    A query block needs exactly the key blocks that hold any occupant of any of
    its rows, so the structure costs ``O(Q * M)`` rather than ``O(Q * K)``.
    """
    queries, slots = alive_table.shape
    keys = key_choices.numel()
    if queries != keys:
        raise ValueError("slot replay aligns one query per stored key row")
    if queries < 1:
        raise ValueError("cannot build a block mask without rows")
    device = alive_table.device
    query_blocks = (queries + block_size - 1) // block_size
    key_blocks = (keys + block_size - 1) // block_size
    presence = torch.zeros((query_blocks, key_blocks), dtype=torch.bool, device=device)
    row_blocks = torch.arange(queries, device=device) // block_size
    occupied = alive_table >= 0
    presence[row_blocks[:, None].expand_as(alive_table)[occupied], alive_table[occupied] // block_size] = True
    kv_num_blocks = presence.sum(dim=-1, dtype=torch.int32)
    kv_indices = torch.argsort(~presence, dim=-1, stable=True).to(torch.int32)
    return BlockMask.from_kv_blocks(
        kv_num_blocks[None, None],
        kv_indices[None, None],
        None,
        None,
        BLOCK_SIZE=block_size,
        mask_mod=slot_mask_mod(alive_table, key_choices),
        seq_lengths=(queries, keys),
    )


_compiled_flex_attention: Callable[..., Any] | None = None


def flex_attention_kernel(device: torch.device) -> Callable[..., Any]:
    """Compiled FlexAttention on CUDA; the CPU math path exists only for host tests."""
    global _compiled_flex_attention
    if device.type != "cuda":
        return flex_attention
    if _compiled_flex_attention is None:
        _compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
    return _compiled_flex_attention


class SlotChoiceHead(nn.Module):
    """Actor-only categorical over ``M`` slots plus the null write."""

    def __init__(self, hidden_size: int, slots: int) -> None:
        super().__init__()
        self.slots = slots
        self.projection = nn.Linear(hidden_size, slots + 1, bias=True, dtype=torch.float32)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def logits(self, hidden: Tensor) -> Tensor:
        return F.linear(
            hidden.float(), self.projection.weight, self.projection.bias
        )

    def log_prob(self, hidden: Tensor, choices: Tensor) -> Tensor:
        """Exact log-probability of recorded choices; ``NO_WRITE`` maps to the null logit."""
        if choices.ndim != 1 or choices.numel() != hidden.shape[0]:
            raise ValueError("one slot choice is required per hidden state")
        index = torch.where(choices < 0, torch.full_like(choices, self.slots), choices)
        return self.logits(hidden).log_softmax(dim=-1).gather(1, index[:, None]).squeeze(1)

    def sample(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Sample head indices in ``[0, M]`` (``M`` is null) with their log-probabilities."""
        log_probabilities = self.logits(hidden).log_softmax(dim=-1)
        choice = torch.multinomial(log_probabilities.exp(), 1).squeeze(1)
        return choice, log_probabilities.gather(1, choice[:, None]).squeeze(1)


class SlotMemoryCombiner(nn.Module):
    """Token embedding plus a scaled residual of a memory read.

    The pretrained embedding path is untouched: ``token_delta`` and ``output``
    start at zero, so the initial policy equals the native model exactly, and
    the FP32 per-channel scale is multiplied before any BF16 rounding, as in
    :class:`postraining.token_carry.TokenCarryCombiner`.
    """

    def __init__(self, hidden_size: int, config: SlotMemoryConfig) -> None:
        super().__init__()
        self.config = config
        width = config.width
        self.token_delta = nn.Linear(hidden_size, hidden_size, bias=False, dtype=torch.float32)
        self.query_token = nn.Linear(hidden_size, width, bias=False, dtype=torch.float32)
        self.query_carry = nn.Linear(hidden_size, width, bias=False, dtype=torch.float32)
        self.key = nn.Linear(hidden_size, width, bias=False, dtype=torch.float32)
        self.value = nn.Linear(hidden_size, width, bias=False, dtype=torch.float32)
        self.output = nn.Linear(width, hidden_size, bias=False, dtype=torch.float32)
        self.null_key = nn.Parameter(torch.zeros((config.heads, config.head_dim), dtype=torch.float32))
        self.scale = nn.Parameter(torch.full((hidden_size,), 0.01, dtype=torch.float32))
        nn.init.zeros_(self.token_delta.weight)
        nn.init.zeros_(self.output.weight)
        for module in (self.query_token, self.query_carry, self.key, self.value):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def score_scale(self) -> float:
        return 1.0 / math.sqrt(self.config.head_dim)

    def _heads(self, x: Tensor) -> Tensor:
        return x.view(x.shape[0], self.config.heads, self.config.head_dim)

    def queries(self, token_embedding: Tensor, producer_hidden: Tensor, positions: Tensor) -> Tensor:
        """Rotated read queries ``[N, heads, D]`` at read positions."""
        dtype = token_embedding.dtype
        raw = F.linear(token_embedding, self.query_token.weight.to(dtype)) + F.linear(
            producer_hidden.detach().to(dtype), self.query_carry.weight.to(dtype)
        )
        cos, sin = rotary_cos_sin(positions, self.config.head_dim, self.config.rope_base)
        return apply_rotary(self._heads(raw), cos, sin)

    def keys_values(self, producer_hidden: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Write-time keys (rotated at the write position) and values ``[N, heads, D]``."""
        hidden = producer_hidden.detach()
        dtype = hidden.dtype
        keys = self._heads(F.linear(hidden, self.key.weight.to(dtype)))
        values = self._heads(F.linear(hidden, self.value.weight.to(dtype)))
        cos, sin = rotary_cos_sin(positions, self.config.head_dim, self.config.rope_base)
        return apply_rotary(keys, cos, sin), values

    def null_logits(self, queries: Tensor) -> Tensor:
        """``[N, heads]`` logits of the zero-valued null entry."""
        return (queries.float() * self.null_key[None]).sum(dim=-1) * self.score_scale

    def combine(self, token_embedding: Tensor, read: Tensor) -> Tensor:
        dtype = token_embedding.dtype
        residual = F.linear(token_embedding, self.token_delta.weight.to(dtype)) + F.linear(
            read.to(dtype), self.output.weight.to(dtype)
        )
        # Do not quantize the learned scale to BF16 before multiplication.
        return token_embedding + (residual.float() * self.scale).to(dtype)

    def read_dense(self, queries: Tensor, keys: Tensor, values: Tensor, visible: Tensor) -> Tensor:
        """Reference read over explicit ``[N, K]`` visibility, including the null entry."""
        if visible.shape != (queries.shape[0], keys.shape[0]):
            raise ValueError("dense visibility must be [queries, keys]")
        scores = torch.einsum("nhd,khd->nhk", queries.float(), keys.float()) * self.score_scale
        scores = scores.masked_fill(~visible[:, None, :], float("-inf"))
        full = torch.cat((self.null_logits(queries)[..., None], scores), dim=-1)
        weights = full.softmax(dim=-1)[..., 1:]
        read = torch.einsum("nhk,khd->nhd", weights, values.float())
        return read.reshape(queries.shape[0], self.config.width).to(queries.dtype)

    def read_flex(self, queries: Tensor, keys: Tensor, values: Tensor, block_mask: BlockMask) -> Tensor:
        """Block-sparse read; the null entry mixes in through the returned log-sum-exp."""
        kernel = flex_attention_kernel(queries.device)
        layout = lambda x: x.transpose(0, 1)[None].contiguous()  # noqa: E731
        out, aux = kernel(
            layout(queries), layout(keys), layout(values),
            block_mask=block_mask, scale=self.score_scale,
            return_aux=AuxRequest(lse=True),
        )
        lse = aux.lse[0].transpose(0, 1)  # [N, heads]
        # Fully masked rows return zero output and -inf lse: the null takes all mass.
        null_mass = torch.sigmoid(self.null_logits(queries) - lse)
        # Mix in FP32: near-empty memories have 1 - null_mass close to zero and
        # a BF16 factor would quantize the whole read.
        mixed = out[0].transpose(0, 1).float() * (1.0 - null_mass)[..., None]
        return mixed.reshape(queries.shape[0], self.config.width).to(queries.dtype)

    def read_slots(
        self, queries: Tensor, slot_keys: Tensor, slot_values: Tensor, slot_alive: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Rollout read over ``[B, M]`` slot banks; returns the read and null mass."""
        scores = torch.einsum("bhd,bmhd->bhm", queries.float(), slot_keys.float()) * self.score_scale
        scores = scores.masked_fill((slot_alive < 0)[:, None, :], float("-inf"))
        full = torch.cat((self.null_logits(queries)[..., None], scores), dim=-1)
        weights = full.softmax(dim=-1)
        read = torch.einsum("bhm,bmhd->bhd", weights[..., 1:], slot_values.float())
        return read.reshape(queries.shape[0], self.config.width).to(queries.dtype), weights[..., 0]


class SlotMemoryRolloutState:
    """Per-lane slot banks plus the per-token slot history for replay export.

    Every mutation is a fixed-shape gather/where/index_put over static buffers so
    the decode step stays CUDA-graph capturable.
    """

    def __init__(
        self,
        config: SlotMemoryConfig,
        *,
        batch_size: int,
        capacity: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.config = config
        shape = (batch_size, config.slots, config.heads, config.head_dim)
        self.keys = torch.zeros(shape, dtype=dtype, device=device)
        self.values = torch.zeros(shape, dtype=dtype, device=device)
        self.alive = torch.full((batch_size, config.slots), NO_WRITE, dtype=torch.long, device=device)
        self.history = torch.full((batch_size, capacity), NO_WRITE, dtype=torch.long, device=device)
        self.rows = torch.arange(batch_size, dtype=torch.long, device=device)

    def buffers(self) -> tuple[Tensor, ...]:
        return (self.keys, self.values, self.alive, self.history, self.rows)

    def reset_all(self) -> None:
        self.alive.fill_(NO_WRITE)
        self.history.fill_(NO_WRITE)

    def reset_lanes(self, lanes: Tensor) -> None:
        """Episode boundary: an admitted lane starts with every slot empty."""
        self.alive.index_fill_(0, lanes, NO_WRITE)
        self.history.index_fill_(0, lanes, NO_WRITE)

    def step(
        self,
        combiner: SlotMemoryCombiner,
        head: SlotChoiceHead,
        *,
        token_embedding: Tensor,
        producer_hidden: Tensor,
        position: Tensor,
        active: Tensor,
        forced: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Sample the slot, overwrite, read, and return the combined embedding.

        Forced environment tokens never write (their joint action is not a
        policy action). Inactive lanes keep their banks and history untouched.
        Returns the mixed ``[B, H]`` embedding and the slot log-probability
        (zero where the step is forced or inactive).
        """
        slots = self.config.slots
        choice, choice_logprob = head.sample(producer_hidden)
        choice = torch.where(forced, torch.full_like(choice, slots), choice)
        write = active & (choice < slots)
        target = choice.clamp_max(slots - 1)
        recorded = torch.where(write, target, torch.full_like(target, NO_WRITE))
        safe_position = position.clamp_max(self.history.shape[1] - 1)
        previous = self.history[self.rows, safe_position]
        self.history.index_put_(
            (self.rows, safe_position), torch.where(active, recorded, previous)
        )
        keys, values = combiner.keys_values(producer_hidden, position)
        lanes = (self.rows, target)
        self.keys.index_put_(lanes, torch.where(write[:, None, None], keys, self.keys[lanes]))
        self.values.index_put_(lanes, torch.where(write[:, None, None], values, self.values[lanes]))
        self.alive.index_put_(lanes, torch.where(write, position, self.alive[lanes]))
        queries = combiner.queries(token_embedding, producer_hidden, position)
        read, _ = combiner.read_slots(queries, self.keys, self.values, self.alive)
        logprob = torch.where(active & ~forced, choice_logprob, torch.zeros_like(choice_logprob))
        return combiner.combine(token_embedding, read), logprob


def replay_read(
    combiner: SlotMemoryCombiner,
    *,
    token_embeddings: Tensor,
    carries: Tensor,
    positions: Tensor,
    alive_table: Tensor,
    key_choices: Tensor,
    backend: str,
) -> Tensor:
    """Read for every stored key row from the recorded table (``flex`` or ``dense``)."""
    queries = combiner.queries(token_embeddings, carries, positions)
    keys, values = combiner.keys_values(carries, positions)
    if backend == "flex":
        return combiner.read_flex(queries, keys, values, slot_block_mask(alive_table, key_choices))
    if backend == "dense":
        return combiner.read_dense(queries, keys, values, visibility_reference(alive_table, key_choices))
    raise ValueError(f"unknown slot replay backend {backend!r}")


def slot_memory_replay_hidden(side: Any, batch: Any, *, backend: str = "flex") -> Tensor:
    """One packed forward with memory reads on the fixed behavior-time stream."""
    config = getattr(side, "slot_memory", None)
    if config is None:
        raise ValueError("slot replay requires a slot-memory model")
    carries = batch.carry_hiddens
    positions = batch.carry_input_positions
    alive_table = batch.slot_alive_table
    key_choices = batch.slot_key_choices
    slot_positions = batch.slot_positions
    if carries is None or positions is None or alive_table is None or key_choices is None or slot_positions is None:
        raise ValueError("slot replay requires stored carries, slot choices, and the alive table")
    if (
        batch.input_ids.ndim != 2 or batch.input_ids.shape[0] != 1
        or carries.ndim != 2 or carries.dtype != torch.bfloat16
        or carries.shape[1] != side.causal_lm.config.hidden_size
        or positions.ndim != 1 or positions.dtype != torch.long
        or positions.numel() != carries.shape[0]
        or alive_table.shape != (carries.shape[0], config.slots)
        or key_choices.shape != (carries.shape[0],) or slot_positions.shape != (carries.shape[0],)
        or alive_table.dtype != torch.long or key_choices.dtype != torch.long
        or slot_positions.dtype != torch.long
        or any(t.device != batch.input_ids.device for t in (carries, positions, alive_table, key_choices, slot_positions))
    ):
        raise ValueError("stored slot replay inputs must match the packed input")
    differentiable = torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    with torch.inference_mode(False), torch.set_grad_enabled(differentiable):
        carries = carries.detach()
        if carries.is_inference():
            carries = carries.clone()
        embeddings = side.token_embeddings(batch.input_ids)
        if positions.numel():
            consumed = embeddings[0, positions]
            read = replay_read(
                side.token_combiner,
                token_embeddings=consumed, carries=carries, positions=slot_positions,
                alive_table=alive_table, key_choices=key_choices, backend=backend,
            )
            mixed = side.token_combiner.combine(consumed, read)
            embeddings = embeddings.index_copy(1, positions, mixed.unsqueeze(0))
        return side.replay_hidden(
            None, batch.attention_mask,
            inputs_embeds=embeddings,
            position_ids=batch.position_ids,
            cu_seqlens=batch.cu_seqlens,
            sequence_boundaries=batch.sequence_boundaries,
            max_sequence_length=batch.max_sequence_length,
        )


def slot_choice_statistics(choice_vectors: list[Tensor], slots: int) -> dict[str, float]:
    """Host-side write statistics of recorded slot choices (diagnostics only)."""
    if not choice_vectors:
        return {}
    choices = torch.cat([vector.long() for vector in choice_vectors])
    total = max(choices.numel(), 1)
    written = choices[choices >= 0]
    counts = torch.bincount(written, minlength=slots).float()
    probabilities = counts / counts.sum().clamp_min(1.0)
    entropy = -(probabilities[probabilities > 0] * probabilities[probabilities > 0].log()).sum()
    distinct = [int(torch.unique(vector[vector >= 0]).numel()) for vector in choice_vectors]
    return {
        "null_fraction": float((choices < 0).sum()) / total,
        "write_entropy": float(entropy),
        "write_entropy_ratio": float(entropy) / math.log(slots) if slots > 1 else 0.0,
        "distinct_slots_per_trajectory": sum(distinct) / len(distinct),
    }
