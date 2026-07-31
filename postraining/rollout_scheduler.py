"""Request-stable continuous batching for on-policy latent rollouts.

The scheduler owns a fixed set of physical decode lanes. After every decode
iteration it evicts completed trajectories and admits whole prompt groups
into the freed lanes. Each lane has an independent logical position and each
trajectory has an RNG key derived only from the pool seed and stable request
identity, so changing capacity or refill order does not change that request's
sampled actions.

The model supplies paged KV storage and row-indexed prefill/decode operations.
This keeps cache placement out of policy logic and lets the scheduler preserve
group attribution without constraining trajectory lengths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
from torch import Tensor
from torch.func import _random as stateless_random

from postraining.latent_rollout import (
    PAD_SLOT,
    TOKEN_SLOT,
    LatentRolloutBatch,
)
from postraining.latent_thought import StepOutput


_STREAM_FIELD_NAMES = (
    "kind",
    "token_ids",
    "hiddens",
    "action_mask",
    "old_token_logprobs",
    "old_values",
)
_UINT64_MASK = (1 << 64) - 1


class ContinuousRefillModel(Protocol):
    """Model-side paged-cache operations required by the scheduler."""

    backbone: object

    def make_paged_generation_cache(
        self,
        capacity: int,
        max_length: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> object: ...

    def build_prompt_prefix_bank(
        self,
        unique_prompt_ids: Tensor,
        unique_prompt_lengths: Tensor,
        *,
        dtype: torch.dtype | None = None,
    ) -> object: ...

    def admit_prompt_prefixes(
        self,
        bank: object,
        group_indices: Tensor,
        group_slot_ids: Tensor,
        paged_cache: object,
    ) -> StepOutput: ...

    def paged_step(
        self,
        input_latent: Tensor,
        paged_cache: object,
        *,
        slot_ids: Tensor,
        positions: Tensor,
        live: Tensor | None = None,
    ) -> StepOutput: ...

    def embed_tokens(self, token_ids: Tensor) -> Tensor: ...

    def combined_input(self, token_ids: Tensor, hidden: Tensor) -> Tensor: ...


@dataclass
class ContinuousScheduleStats:
    """Physical work and lifecycle events for one continuous rollout pool."""

    decode_steps: int = 0
    row_steps: int = 0
    occupied_row_steps: int = 0
    useful_actions: int = 0
    lockstep_decode_steps: int = 0
    admission_events: int = 0
    admitted_groups: int = 0
    admitted_rows: int = 0
    eviction_events: int = 0
    evicted_rows: int = 0
    active_rows_max: int = 0
    free_rows_min: int = 0
    capacity_rows: int = 0
    chunks: int = 0

    def metrics(self) -> dict[str, float]:
        productive_utilization = (
            self.useful_actions / self.row_steps if self.row_steps else 0.0
        )
        capacity_row_steps = self.decode_steps * self.capacity_rows
        capacity_utilization = (
            self.useful_actions / capacity_row_steps
            if capacity_row_steps
            else 0.0
        )
        occupancy = (
            self.occupied_row_steps / capacity_row_steps
            if capacity_row_steps
            else 0.0
        )
        savings = (
            1.0 - self.decode_steps / self.lockstep_decode_steps
            if self.lockstep_decode_steps
            else 0.0
        )
        return {
            "decode_steps_per_chunk_mean": (
                self.decode_steps / self.chunks if self.chunks else 0.0
            ),
            "decode_step_utilization": productive_utilization,
            "decode_capacity_utilization": capacity_utilization,
            "decode_slot_occupancy": occupancy,
            "decode_steps_total": float(self.decode_steps),
            "decode_lockstep_steps_total": float(self.lockstep_decode_steps),
            "decode_step_savings_fraction": savings,
            "decode_row_steps_total": float(self.row_steps),
            "decode_occupied_row_steps_total": float(
                self.occupied_row_steps
            ),
            "decode_useful_actions_total": float(self.useful_actions),
            "decode_refill_admission_events": float(self.admission_events),
            "decode_refill_admitted_groups": float(self.admitted_groups),
            "decode_refill_admitted_rows": float(self.admitted_rows),
            "decode_refill_eviction_events": float(self.eviction_events),
            "decode_refill_evicted_rows": float(self.evicted_rows),
            "decode_refill_active_rows_max": float(self.active_rows_max),
            "decode_refill_free_rows_min": float(self.free_rows_min),
        }


@dataclass(frozen=True)
class _Origin:
    origin_id: int
    rows: int
    prompt_length: int


@dataclass(frozen=True)
class _RecordSegment:
    origin_ids: tuple[int, ...]
    origin_rows: tuple[int, ...]
    through_positions: tuple[int, ...]
    values: dict[str, Tensor]


class _RecordLedger:
    """Completed trajectory records with stable request attribution."""

    def __init__(self, origins: Sequence[_Origin]):
        self.origins = {origin.origin_id: origin for origin in origins}
        self.segments: list[_RecordSegment] = []

    def append_completed(
        self,
        values: dict[str, Tensor],
        slots: Tensor,
        through_positions: Sequence[int],
        origin_ids: Sequence[int],
        origin_rows: Sequence[int],
        *,
        target_device: torch.device,
    ) -> None:
        if slots.numel() == 0:
            return
        completed_rows = slots.numel()
        if not (
            len(through_positions)
            == len(origin_ids)
            == len(origin_rows)
            == completed_rows
        ):
            raise ValueError("completed record metadata must align with slots")
        width = max(through_positions) + 1
        async_cuda_to_cpu = (
            slots.device.type == "cuda" and target_device.type == "cpu"
        )

        def select(value: Tensor) -> Tensor:
            selected = value.index_select(0, slots)
            if async_cuda_to_cpu:
                target = torch.empty_like(
                    selected, device=target_device, pin_memory=True
                )
                target.copy_(selected, non_blocking=True)
                return target
            return selected.to(target_device).clone()

        segment_values = {
            name: select(value[:, :width])
            for name, value in values.items()
        }
        self.segments.append(
            _RecordSegment(
                origin_ids=tuple(origin_ids),
                origin_rows=tuple(origin_rows),
                through_positions=tuple(through_positions),
                values=segment_values,
            )
        )

    def assemble(
        self, *, target_device: torch.device, carry_injected: bool = False
    ) -> dict[int, LatentRolloutBatch]:
        result: dict[int, LatentRolloutBatch] = {}
        for origin_id, origin in sorted(self.origins.items()):
            relevant = [
                segment
                for segment in self.segments
                if origin_id in segment.origin_ids
            ]
            selected_rows = sum(
                segment.origin_ids.count(origin_id)
                for segment in relevant
            )
            if selected_rows != origin.rows:
                raise RuntimeError(
                    f"origin {origin_id} completed {selected_rows}/"
                    f"{origin.rows} trajectories"
                )
            used = origin.prompt_length
            for segment in relevant:
                used = max(
                    used,
                    max(
                        position + 1
                        for segment_origin, position in zip(
                            segment.origin_ids,
                            segment.through_positions,
                            strict=True,
                        )
                        if segment_origin == origin_id
                    ),
                )

            exemplar = relevant[0].values
            assembled: dict[str, Tensor] = {}
            for name in _STREAM_FIELD_NAMES:
                value = exemplar[name]
                shape = (origin.rows, used, *value.shape[2:])
                if name == "kind":
                    target = torch.full(
                        shape,
                        PAD_SLOT,
                        dtype=value.dtype,
                        device=target_device,
                    )
                else:
                    target = torch.zeros(
                        shape, dtype=value.dtype, device=target_device
                    )
                assembled[name] = target
            for segment in relevant:
                selected_indices = [
                    index
                    for index, segment_origin in enumerate(segment.origin_ids)
                    if segment_origin == origin_id
                ]
                selected = torch.tensor(
                    selected_indices,
                    dtype=torch.long,
                    device=target_device,
                )
                rows = torch.tensor(
                    [segment.origin_rows[index] for index in selected_indices],
                    dtype=torch.long,
                    device=target_device,
                )
                for name in _STREAM_FIELD_NAMES:
                    source = segment.values[name].index_select(0, selected)
                    kept = min(source.size(1), used)
                    assembled[name][rows, :kept] = source[:, :kept]

            action_mask = assembled["action_mask"]
            result[origin_id] = LatentRolloutBatch(
                **assembled,
                rewards=torch.zeros_like(action_mask),
                reward_scalar=torch.zeros(
                    origin.rows,
                    dtype=torch.float32,
                    device=target_device,
                ),
                prompt_length=origin.prompt_length,
                carry_injected=carry_injected,
            )
        return result


@dataclass(frozen=True)
class _PendingGroup:
    request_id: int
    origin_id: int
    group_row: int
    prompt_ids: Tensor
    prompt_length: int


@dataclass
class _PolicyBuffers:
    belief: Tensor
    logits: Tensor

    @classmethod
    def from_output(
        cls, output: StepOutput, capacity: int
    ) -> "_PolicyBuffers":
        def empty_like(value: Tensor) -> Tensor:
            return torch.empty(
                (capacity, *value.shape[1:]),
                dtype=value.dtype,
                device=value.device,
            )

        return cls(
            belief=empty_like(output.belief),
            logits=empty_like(output.logits),
        )

    def scatter(self, slots: Tensor, output: StepOutput) -> None:
        if output.belief.size(0) != slots.numel():
            raise ValueError("model output rows do not match requested slots")
        for name in ("belief", "logits"):
            getattr(self, name).index_copy_(0, slots, getattr(output, name))

    def scatter_prefix(self, slots: Tensor, output: StepOutput) -> None:
        """Scatter the leading real rows from a bucket-padded model step."""
        rows = slots.numel()
        if output.belief.size(0) < rows:
            raise ValueError("model output has fewer rows than requested slots")
        for name in ("belief", "logits"):
            getattr(self, name).index_copy_(
                0, slots, getattr(output, name)[:rows]
            )


def _empty_record_values(
    capacity: int,
    width: int,
    hidden_dim: int,
    *,
    device: torch.device,
) -> dict[str, Tensor]:
    action_mask = torch.zeros(
        (capacity, width), dtype=torch.float32, device=device
    )
    return {
        "kind": torch.full(
            (capacity, width), PAD_SLOT, dtype=torch.long, device=device
        ),
        "token_ids": torch.zeros(
            (capacity, width), dtype=torch.long, device=device
        ),
        "hiddens": torch.zeros(
            (capacity, width, hidden_dim), dtype=torch.float32, device=device
        ),
        "action_mask": action_mask,
        "old_token_logprobs": torch.zeros_like(action_mask),
        "old_values": torch.zeros_like(action_mask),
    }


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def _request_seed(pool_seed: int, request_id: int, sample_index: int) -> int:
    seed = _splitmix64(pool_seed & _UINT64_MASK)
    seed = _splitmix64(seed ^ (request_id & _UINT64_MASK))
    return _splitmix64(seed ^ (sample_index & _UINT64_MASK))


def _signed64(value: int) -> int:
    return value if value < (1 << 63) else value - (1 << 64)


def _decode_execution_width(
    active_rows: int,
    capacity_rows: int,
    *,
    pending_groups: bool,
) -> int:
    """Choose a static CUDA batch bucket for the FlexDecoding compiler.

    Current FlexDecoding requires a concrete batch dimension. While refill
    work remains, full capacity gives the long main phase one specialization
    and pads at most one incomplete prompt group. After all requests have
    entered, power-of-two tail buckets bound padding below 2x without
    compiling every survivor count.
    """
    if not 0 < active_rows <= capacity_rows:
        raise ValueError("active rows must lie within decode capacity")
    if pending_groups:
        return capacity_rows
    return min(capacity_rows, 1 << (active_rows - 1).bit_length())


@torch.no_grad()
def warmup_decode_width_buckets(
    model: ContinuousRefillModel,
    paged_cache,
    capacity_rows: int,
    *,
    passes: int = 3,
) -> list[int]:
    """Drive every declared execution-width bucket through the paged step.

    The width set is exactly the range of ``_decode_execution_width`` — full
    capacity for the refill main phase plus the power-of-two tail buckets —
    so after this returns, no live pool can present the compiled paged step
    with a shape it has not already seen. Under ``mode="reduce-overhead"``
    that means every CUDA graph is captured before the first real token:
    shapes are declared up front, never discovered mid-rollout. Largest
    width first, so later captures reuse the largest activation slab in the
    shared cudagraph memory pool (the vLLM capture-order rule).

    Every warmup row is dead (``live`` all false): KV reads see an empty
    range, KV writes land in the scratch pages no slot owns, and recurrent
    (KDA) lane writes are redirected to the scratch lane rows in
    ``[capacity, 2 * capacity)`` — so the arena is untouched everywhere a
    real request can reach. ``no_grad`` is
    LOAD-BEARING twice over: grad state is a compile guard (a mismatched
    warmup would specialize a second, never-replayed artifact per width),
    and inference mode is what lets cudagraph trees end the current
    execution generation between replays — a grad-enabled artifact with
    the previous step's output still alive would re-record a fresh child
    graph every decode step, unbounded. The input latent comes from the
    model's own token-embedding producer rather than a hardcoded dtype so
    the warmup call is guard-identical (dtype, stride) to the decode
    loop's; ``passes`` defaults to 3 because inductor's cudagraph trees
    run eager warmup calls before recording a shape's graph.
    """
    device = paged_cache.page_table.device
    widths = sorted(
        {
            min(capacity_rows, 1 << shift)
            for shift in range(capacity_rows.bit_length())
        }
        | {capacity_rows},
        reverse=True,
    )
    for width in widths:
        input_latent = model.embed_tokens(
            torch.zeros((width, 1), dtype=torch.long, device=device)
        )
        slot_ids = torch.zeros(width, dtype=torch.long, device=device)
        positions = torch.zeros(width, dtype=torch.long, device=device)
        live = torch.zeros(width, dtype=torch.bool, device=device)
        for _ in range(passes):
            model.paged_step(
                input_latent,
                paged_cache,
                slot_ids=slot_ids,
                positions=positions,
                live=live,
            )
    return widths


def _cpu_request_random(key_bits: Tensor) -> tuple[Tensor, Tensor]:
    """Isolated CPU reference for tests; production CUDA stays vectorized."""
    rows = key_bits.size(0)
    token = torch.empty(rows, dtype=torch.float32)
    next_key_bits = key_bits.clone()
    for row in range(rows):
        seed = int(key_bits[row, 0]) & _UINT64_MASK
        counter = int(key_bits[row, 1]) & _UINT64_MASK
        decision = _splitmix64(seed ^ _splitmix64(counter))
        generator = torch.Generator().manual_seed(
            decision & ((1 << 63) - 1)
        )
        token[row] = torch.rand((), generator=generator)
        next_key_bits[row, 1] = _signed64(
            (counter + 1) & _UINT64_MASK
        )
    return token, next_key_bits


def _request_random(key_bits: Tensor) -> tuple[Tensor, Tensor]:
    """One token-uniform draw and successor key per request.

    The deterministic hidden-carry policy samples nothing but the next
    token, so this draw order does not reproduce rollouts recorded under
    the old gate/thought scheme.
    """
    if key_bits.device.type == "cpu":
        return _cpu_request_random(key_bits)
    if key_bits.device.type != "cuda":
        raise ValueError("request-stable Philox supports only CPU and CUDA")
    keys = key_bits.view(torch.uint64)
    token_key, next_keys = stateless_random.split(keys, 2)
    rows = keys.size(0)
    token = stateless_random.uniform(token_key, (rows,), dtype=torch.float32)
    return token, next_keys.view(torch.int64)


def _top_p_from_uniform(
    logits: Tensor,
    uniform: Tensor,
    temperature: float,
    top_p: float,
) -> Tensor:
    """Inverse-CDF token sampling driven by one request-local uniform."""
    logits = logits.float()
    if temperature != 1.0:
        logits = logits / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
        probabilities = sorted_logits.softmax(dim=-1)
        remove = probabilities.cumsum(dim=-1) - probabilities > top_p
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        probabilities = sorted_logits.softmax(dim=-1)
        indices = sorted_indices
    else:
        probabilities = logits.softmax(dim=-1)
        indices = None
    cdf = probabilities.cumsum(dim=-1)
    sampled = torch.searchsorted(
        cdf.contiguous(), uniform[:, None].contiguous(), right=False
    ).squeeze(-1)
    sampled.clamp_max_(logits.size(-1) - 1)
    if indices is not None:
        sampled = indices.gather(-1, sampled[:, None]).squeeze(-1)
    return sampled


def _reset_record_slots(values: dict[str, Tensor], slots: Tensor) -> None:
    for name, value in values.items():
        selected = value.index_select(0, slots)
        if name == "kind":
            selected.fill_(PAD_SLOT)
        else:
            selected.zero_()
        value.index_copy_(0, slots, selected)


@torch.no_grad()
def rollout_continuous_refill_groups(
    model: ContinuousRefillModel,
    prompt_chunks: Sequence[Tensor],
    prompt_length_chunks: Sequence[Tensor],
    *,
    prompt_repeats: int,
    capacity_rows: int,
    max_new_tokens: int,
    max_stream_steps: int,
    temperature: float,
    top_p: float,
    seed: int,
    stop_ids: int | Sequence[int] | None = None,
    cache_dtype: torch.dtype | None = None,
    replay_storage: bool = True,
    pin_emit: bool = False,
    offload_device: torch.device = torch.device("cpu"),
    schedule_stats: ContinuousScheduleStats | None = None,
    paged_cache: object | None = None,
    pad_decode_width: bool | None = None,
) -> list[LatentRolloutBatch]:
    """Continuously refill fixed physical lanes with whole prompt groups.

    Every request runs until its environment stop, emitted-token cap, or
    stream cap. Admission never truncates or cancels a live trajectory.
    Refill occurs after every decode iteration. Returned batches correspond
    one-for-one with ``prompt_chunks``.
    """
    if not prompt_chunks:
        return []
    if len(prompt_chunks) != len(prompt_length_chunks):
        raise ValueError("prompt chunks and length chunks must align")
    if prompt_repeats < 1:
        raise ValueError("prompt_repeats must be positive")
    if capacity_rows < prompt_repeats:
        raise ValueError("capacity_rows must fit one whole prompt group")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if max_stream_steps < max_new_tokens:
        raise ValueError("stream budget cannot fit the requested actions")
    widths = {chunk.size(1) for chunk in prompt_chunks}
    if len(widths) != 1:
        raise ValueError("all chunks must use one common prompt width")
    prompt_width = widths.pop()
    device = prompt_chunks[0].device
    if any(chunk.device != device for chunk in prompt_chunks):
        raise ValueError("all prompt chunks must share one device")
    # Bucketing the decode batch exists for the FlexDecoding compiler, so it
    # defaults on where that compiler runs. It stays overridable because the
    # padded step mutates nothing a test can observe indirectly -- the only
    # way to assert that is to run it.
    if pad_decode_width is None:
        pad_decode_width = device.type == "cuda"

    pending: list[_PendingGroup] = []
    origins: list[_Origin] = []
    next_request_id = 0
    for origin_id, (prompts, lengths) in enumerate(
        zip(prompt_chunks, prompt_length_chunks, strict=True)
    ):
        if prompts.dim() != 2 or prompts.size(1) != prompt_width:
            raise ValueError("prompt chunks must be rank-two")
        if prompts.size(0) == 0:
            raise ValueError("prompt chunks must contain at least one group")
        if lengths.shape != (prompts.size(0),):
            raise ValueError("prompt lengths must contain one length per group")
        lengths = lengths.to(device="cpu", dtype=torch.long)
        if bool((lengths < 1).any()) or bool((lengths > prompt_width).any()):
            raise ValueError("prompt lengths must lie within the common width")
        origins.append(
            _Origin(origin_id, prompts.size(0) * prompt_repeats, prompt_width)
        )
        for group_row in range(prompts.size(0)):
            pending.append(
                _PendingGroup(
                    request_id=next_request_id,
                    origin_id=origin_id,
                    group_row=group_row,
                    prompt_ids=prompts[group_row],
                    prompt_length=int(lengths[group_row]),
                )
            )
            next_request_id += 1

    if schedule_stats is not None:
        if any(vars(schedule_stats).values()):
            raise ValueError("schedule_stats must be empty")
        schedule_stats.capacity_rows = capacity_rows
        schedule_stats.free_rows_min = capacity_rows
        schedule_stats.chunks = len(prompt_chunks)

    max_stream = prompt_width + max_stream_steps
    stored_hidden_dim = (
        model.backbone.tok_emb.embedding_dim
        if replay_storage and not pin_emit
        else 0
    )
    bank_prompt_ids = torch.stack([group.prompt_ids for group in pending])
    bank_prompt_lengths = torch.tensor(
        [group.prompt_length for group in pending],
        dtype=torch.long,
        device=device,
    )
    prompt_bank = model.build_prompt_prefix_bank(
        bank_prompt_ids, bank_prompt_lengths, dtype=cache_dtype
    )
    if paged_cache is None:
        cache = model.make_paged_generation_cache(
            capacity_rows, max_stream, device, dtype=cache_dtype
        )
    else:
        cache = paged_cache
        if getattr(cache, "capacity", None) != capacity_rows:
            raise ValueError("paged cache capacity does not match capacity_rows")
        if getattr(cache, "max_length", -1) < max_stream:
            raise ValueError("paged cache is shorter than the rollout stream")
        cache_kv_starts = getattr(cache, "kv_starts", None)
        if not isinstance(cache_kv_starts, Tensor):
            raise TypeError("paged cache must expose a kv_starts tensor")
        if cache_kv_starts.device != device:
            raise ValueError("paged cache and prompts must share a device")
    values = _empty_record_values(
        capacity_rows, max_stream, stored_hidden_dim, device=device
    )
    ledger = _RecordLedger(origins)
    ended = torch.zeros(capacity_rows, dtype=torch.bool, device=device)
    emitted = torch.zeros(capacity_rows, dtype=torch.long, device=device)
    positions = torch.zeros(capacity_rows, dtype=torch.long, device=device)
    origin_ids = [-1] * capacity_rows
    origin_rows = [-1] * capacity_rows
    # Torch does not implement indexed writes for uint64. Store identical raw
    # bits in int64 and reinterpret only at the stateless Philox boundary.
    request_key_bits = torch.zeros(
        (capacity_rows, 2), dtype=torch.int64, device=device
    )
    policy: _PolicyBuffers | None = None
    pending_index = 0
    occupied_slots: list[int] = []
    free_slots = list(range(capacity_rows))
    ids = (
        ()
        if stop_ids is None
        else ((stop_ids,) if isinstance(stop_ids, int) else tuple(stop_ids))
    )
    stop_tensor = (
        torch.tensor(ids, dtype=torch.long, device=device) if ids else None
    )

    def admit() -> None:
        nonlocal pending_index, policy, occupied_slots, free_slots
        available_groups = len(free_slots) // prompt_repeats
        remaining_groups = len(pending) - pending_index
        count = min(available_groups, remaining_groups)
        if count == 0:
            return
        admitted = pending[pending_index : pending_index + count]
        pending_index += count
        admitted_slot_count = count * prompt_repeats
        selected_slots = free_slots[:admitted_slot_count]
        free_slots = free_slots[admitted_slot_count:]
        occupied_slots = sorted((*occupied_slots, *selected_slots))
        slot_ids = torch.tensor(
            selected_slots,
            dtype=torch.long,
            device=device,
        ).view(count, prompt_repeats)
        flat_slots = slot_ids.flatten()
        group_indices = torch.tensor(
            [group.request_id for group in admitted],
            dtype=torch.long,
            device=device,
        )
        prompts = bank_prompt_ids.index_select(0, group_indices)
        lengths = bank_prompt_lengths.index_select(0, group_indices)
        output = model.admit_prompt_prefixes(
            prompt_bank, group_indices, slot_ids, cache
        )
        if policy is None:
            policy = _PolicyBuffers.from_output(output, capacity_rows)
        policy.scatter(flat_slots, output)

        _reset_record_slots(values, flat_slots)
        repeated_prompts = prompts.repeat_interleave(prompt_repeats, dim=0)
        repeated_lengths = lengths.repeat_interleave(prompt_repeats)
        pad_lengths = prompt_width - repeated_lengths
        prompt_valid = (
            torch.arange(prompt_width, device=device)[None, :]
            >= pad_lengths[:, None]
        )
        values["kind"][flat_slots, :prompt_width] = torch.where(
            prompt_valid,
            values["kind"].new_full((), TOKEN_SLOT),
            values["kind"].new_full((), PAD_SLOT),
        )
        values["token_ids"][flat_slots, :prompt_width] = (
            repeated_prompts * prompt_valid
        )
        ended[flat_slots] = False
        emitted[flat_slots] = 0
        positions[flat_slots] = prompt_width - 1

        admitted_origin_ids = []
        admitted_origin_rows = []
        admitted_keys = []
        for group in admitted:
            for sample in range(prompt_repeats):
                row = group.group_row * prompt_repeats + sample
                admitted_origin_ids.append(group.origin_id)
                admitted_origin_rows.append(row)
                admitted_keys.append(
                    [
                        _signed64(
                            _request_seed(seed, group.request_id, sample)
                        ),
                        0,
                    ]
                )
        for slot, origin_id, origin_row in zip(
            selected_slots,
            admitted_origin_ids,
            admitted_origin_rows,
            strict=True,
        ):
            origin_ids[slot] = origin_id
            origin_rows[slot] = origin_row
        request_key_bits[flat_slots] = torch.tensor(
            admitted_keys, dtype=torch.int64, device=device
        )
        if schedule_stats is not None:
            schedule_stats.admission_events += 1
            schedule_stats.admitted_groups += count
            schedule_stats.admitted_rows += flat_slots.numel()
            active_rows = len(occupied_slots)
            schedule_stats.active_rows_max = max(
                schedule_stats.active_rows_max, active_rows
            )
            schedule_stats.free_rows_min = min(
                schedule_stats.free_rows_min,
                len(free_slots),
            )

    def active_mask(slots: Tensor) -> Tensor:
        # Every decode step emits one token, so the emitted-token cap alone
        # bounds the trajectory; the entry validation guarantees the stream
        # budget can hold max_new_tokens consequence slots.
        return ~ended.index_select(0, slots) & (
            emitted.index_select(0, slots) < max_new_tokens
        )

    while pending_index < len(pending) or occupied_slots:
        admit()
        if not occupied_slots:
            raise RuntimeError("pending groups cannot fit the lane capacity")
        slots = torch.tensor(
            occupied_slots, dtype=torch.long, device=device
        )
        if policy is None:
            raise RuntimeError("admission did not initialize policy state")

        slot_positions = positions.index_select(0, slots)
        belief = policy.belief.index_select(0, slots)
        logits = policy.logits.index_select(0, slots)
        keys = request_key_bits.index_select(0, slots)
        token_uniform, next_keys = _request_random(keys)
        request_key_bits.index_copy_(0, slots, next_keys)

        token = _top_p_from_uniform(
            logits, token_uniform, temperature, top_p
        )

        row_slots = (slots, slot_positions)
        values["action_mask"][row_slots] = values["action_mask"].new_ones(())
        next_positions = slot_positions + 1
        next_slots = (slots, next_positions)
        values["kind"][next_slots] = TOKEN_SLOT
        values["token_ids"][next_slots] = token
        if stored_hidden_dim:
            # The +1 shift: the consequence slot stores the belief that
            # decided its token, exactly the carry replay will inject there.
            values["hiddens"][next_slots] = belief.float()
        emitted.index_add_(0, slots, torch.ones_like(slots))
        if stop_tensor is not None:
            ended[slots] |= torch.isin(token, stop_tensor)
        positions.index_copy_(0, slots, next_positions)

        if pin_emit:
            next_input = model.embed_tokens(token[:, None])
        else:
            # Every fed-back token here was just generated, so the
            # hasThought flag is implicitly all-ones (combined_input's
            # contract); prompt tokens enter only through prefix admission.
            next_input = model.combined_input(token, belief)
        model_slots = slots
        model_positions = next_positions
        model_input = next_input
        model_live = None
        if pad_decode_width:
            execution_width = _decode_execution_width(
                slots.numel(),
                capacity_rows,
                pending_groups=pending_index < len(pending),
            )
            padding_rows = execution_width - slots.numel()
            if padding_rows:
                # Padding exists to hold the batch dimension at a bucket the
                # FlexDecoding compiler has already specialized; it must own
                # no cache state at all. `live` gives these rows an empty KV
                # range and a scratch write sink, so which slot id they carry
                # is immaterial and no free lane is reserved or mutated.
                padding_slots = torch.zeros(
                    padding_rows, dtype=torch.long, device=device
                )
                model_slots = torch.cat((slots, padding_slots))
                model_positions = torch.cat(
                    (next_positions, torch.zeros_like(padding_slots))
                )
                model_input = torch.cat(
                    (
                        next_input,
                        next_input.new_zeros(
                            padding_rows, 1, next_input.size(2)
                        ),
                    )
                )
                model_live = torch.arange(
                    execution_width, device=device
                ) < slots.numel()
        output = model.paged_step(
            model_input,
            cache,
            slot_ids=model_slots,
            positions=model_positions,
            live=model_live,
        )
        policy.scatter_prefix(slots, output)
        if schedule_stats is not None:
            schedule_stats.decode_steps += 1
            schedule_stats.row_steps += model_slots.numel()
            schedule_stats.occupied_row_steps += slots.numel()

        # One status transfer is the only per-iteration device/host boundary.
        # Host-owned lane lists then drive eviction and the next admission
        # without rediscovering occupied/free slots through CUDA shapes.
        continuing = active_mask(slots)
        status = torch.stack(
            (
                continuing.long(),
                positions.index_select(0, slots),
            ),
            dim=1,
        ).to("cpu").tolist()
        completed_slots = [
            slot
            for slot, (is_continuing, _) in zip(
                occupied_slots, status, strict=True
            )
            if not is_continuing
        ]
        if completed_slots:
            completed_positions = [
                position
                for is_continuing, position in status
                if not is_continuing
            ]
            completed_tensor = torch.tensor(
                completed_slots, dtype=torch.long, device=device
            )
            ledger.append_completed(
                values,
                completed_tensor,
                completed_positions,
                [origin_ids[slot] for slot in completed_slots],
                [origin_rows[slot] for slot in completed_slots],
                target_device=offload_device,
            )
            for slot in completed_slots:
                origin_ids[slot] = -1
                origin_rows[slot] = -1
            completed = set(completed_slots)
            occupied_slots = [
                slot for slot in occupied_slots if slot not in completed
            ]
            free_slots = sorted((*free_slots, *completed_slots))
            if schedule_stats is not None:
                schedule_stats.eviction_events += 1
                schedule_stats.evicted_rows += len(completed_slots)

    if device.type == "cuda" and offload_device.type == "cpu":
        # Copies and subsequent slot resets share this stream, so recycling
        # needs no per-eviction barrier. Host reconstruction starts only
        # after this single pool-level completion.
        torch.cuda.current_stream(device).synchronize()
    results = ledger.assemble(
        target_device=offload_device, carry_injected=not pin_emit
    )
    ordered = [results[index] for index in range(len(prompt_chunks))]
    if schedule_stats is not None:
        per_chunk_actions = [
            batch.action_mask.sum(dim=1) for batch in ordered
        ]
        schedule_stats.useful_actions = sum(
            int(actions.sum()) for actions in per_chunk_actions
        )
        schedule_stats.lockstep_decode_steps = sum(
            min(int(actions.max()), max_stream_steps)
            for actions in per_chunk_actions
        )
    return ordered
