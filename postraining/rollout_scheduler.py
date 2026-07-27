"""Bounded scalar-position cohort scheduling for latent-policy rollouts.

The ordinary :func:`postraining.latent_rollout.rollout_continuations` surface
deliberately remains the simple one-chunk implementation.  This module owns
the pool-level optimization: chunks with one common left-padded prompt width
share an absolute position grid, so completed rows can be replaced by live
rows from later chunks at synchronization boundaries. Aligned states remain
parked as one bounded cohort and advance only when the cohort reaches its row
capacity or the pool has no more chunks. This avoids repeatedly stepping an
ever-growing rolling tail while retaining the dense scalar-position attention
kernel.

The scheduler never changes an MDP transition or a trajectory's group
attribution.  Scheduling does change which global RNG draws map to which
trajectory, so callers must version that execution schema.  It also returns
compact CPU batches: record prefixes are moved off device at pause boundaries
and the live continuation uses a survivor-sized slab, avoiding two dense
chunk-sized thought buffers coexisting on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from postraining.core import top_p_sample
from postraining.latent_rollout import (
    PAD_SLOT,
    THOUGHT_SLOT,
    TOKEN_SLOT,
    LatentRolloutBatch,
)
from postraining.latent_thought import EMIT, THINK, LatentThoughtModel, StepOutput


_STREAM_FIELD_NAMES = (
    "kind",
    "token_ids",
    "thoughts",
    "actions",
    "action_mask",
    "stop_mask",
    "old_stop_logprobs",
    "old_token_logprobs",
    "old_thought_logprobs",
    "old_thought_means",
    "old_thought_log_sigmas",
    "old_values",
)


@dataclass(frozen=True)
class _Origin:
    origin_id: int
    rows: int
    prompt_length: int


@dataclass
class TailScheduleStats:
    """Actual recurrent work paid by one scalar-tail rollout pool."""

    decode_steps: int = 0
    row_steps: int = 0
    useful_actions: int = 0
    lockstep_decode_steps: int = 0
    catchup_decode_steps: int = 0
    parks: int = 0
    merges: int = 0
    merge_events: int = 0
    cohort_admissions: int = 0
    cohort_rollovers: int = 0
    cohort_rows_max: int = 0
    chunks: int = 0

    def metrics(self) -> dict[str, float]:
        utilization = (
            self.useful_actions / self.row_steps if self.row_steps else 0.0
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
            "decode_step_utilization": utilization,
            "decode_steps_total": float(self.decode_steps),
            "decode_lockstep_steps_total": float(self.lockstep_decode_steps),
            "decode_step_savings_fraction": savings,
            "decode_row_steps_total": float(self.row_steps),
            "decode_useful_actions_total": float(self.useful_actions),
            "decode_tail_catchup_steps": float(self.catchup_decode_steps),
            "decode_tail_parks": float(self.parks),
            "decode_tail_merges": float(self.merges),
            "decode_tail_merge_events": float(self.merge_events),
            "decode_cohort_admissions": float(self.cohort_admissions),
            "decode_cohort_rollovers": float(self.cohort_rollovers),
            "decode_cohort_rows_max": float(self.cohort_rows_max),
        }


@dataclass
class _RecordSlab:
    """Device record storage aligned to the state's current decode rows."""

    base_position: int
    origin_ids: Tensor
    origin_rows: Tensor
    values: dict[str, Tensor]

    @property
    def rows(self) -> int:
        return self.origin_rows.numel()

    def input_at(self, slab_rows: Tensor, position: int) -> tuple[Tensor, Tensor, Tensor]:
        column = position - self.base_position
        return (
            self.values["kind"][slab_rows, column],
            self.values["token_ids"][slab_rows, column],
            self.values["thoughts"][slab_rows, column],
        )


@dataclass(frozen=True)
class _RecordSegment:
    """One compact host-resident section of one or more trajectory records."""

    base_position: int
    origin_ids: Tensor
    origin_rows: Tensor
    values: dict[str, Tensor]


class _RecordLedger:
    """Sparse-in-time record ownership across rolling pause boundaries."""

    def __init__(self, origins: Sequence[_Origin]):
        self.origins = {origin.origin_id: origin for origin in origins}
        self.segments: list[_RecordSegment] = []

    def absorb(self, other: "_RecordLedger") -> None:
        overlap = self.origins.keys() & other.origins.keys()
        if overlap:
            raise ValueError(f"record origins overlap: {sorted(overlap)}")
        self.origins.update(other.origins)
        self.segments.extend(other.segments)
        other.segments.clear()

    def flush(
        self,
        slab: _RecordSlab,
        through_position: int,
        *,
        cpu: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Offload a used slab prefix and return its current input seam."""
        width = through_position - slab.base_position + 1
        if width < 1 or width > next(iter(slab.values.values())).size(1):
            raise ValueError("record flush boundary is outside the live slab")
        seam = slab.input_at(
            torch.arange(slab.rows, device=slab.origin_rows.device),
            through_position,
        )
        async_cuda_to_cpu = (
            next(iter(slab.values.values())).device.type == "cuda"
            and cpu.type == "cpu"
        )

        def move(value: Tensor) -> Tensor:
            if async_cuda_to_cpu:
                target = torch.empty_like(value, device=cpu, pin_memory=True)
                target.copy_(value, non_blocking=True)
                return target
            if value.device == cpu:
                return value.clone()
            return value.to(cpu)

        segment_values = {
            name: move(value[:, :width])
            for name, value in slab.values.items()
        }
        origin_ids = move(slab.origin_ids)
        origin_rows = move(slab.origin_rows)
        if async_cuda_to_cpu:
            # All compact prefix copies share the current stream.  One
            # barrier makes the segment host-readable without serializing
            # once per field.
            torch.cuda.current_stream(
                next(iter(slab.values.values())).device
            ).synchronize()
        self.segments.append(
            _RecordSegment(
                base_position=slab.base_position,
                origin_ids=origin_ids,
                origin_rows=origin_rows,
                values=segment_values,
            )
        )
        return seam

    def finalize(
        self,
        slab: _RecordSlab,
        through_position: int,
        *,
        cpu: torch.device,
    ) -> dict[int, LatentRolloutBatch]:
        self.flush(slab, through_position, cpu=cpu)
        result: dict[int, LatentRolloutBatch] = {}
        for origin_id, origin in sorted(self.origins.items()):
            relevant = [
                segment
                for segment in self.segments
                if bool((segment.origin_ids == origin_id).any())
            ]
            if not relevant:
                raise RuntimeError(f"origin {origin_id} has no record segments")
            used = origin.prompt_length
            for segment in relevant:
                selected = (
                    segment.origin_ids == origin_id
                ).nonzero().squeeze(-1)
                semantic = (
                    (
                        segment.values["kind"].index_select(0, selected)
                        != PAD_SLOT
                    )
                    | segment.values["action_mask"]
                    .index_select(0, selected)
                    .bool()
                ).any(0)
                if bool(semantic.any()):
                    used = max(
                        used,
                        segment.base_position
                        + int(semantic.nonzero().max())
                        + 1,
                    )
            exemplar = relevant[0].values
            assembled: dict[str, Tensor] = {}
            for name in _STREAM_FIELD_NAMES:
                value = exemplar[name]
                shape = (origin.rows, used, *value.shape[2:])
                if name == "kind":
                    target = torch.full(
                        shape, PAD_SLOT, dtype=value.dtype, device=cpu
                    )
                else:
                    target = torch.zeros(shape, dtype=value.dtype, device=cpu)
                assembled[name] = target

            # Segments are chronological.  Their one-column seam overlaps by
            # design; the later segment owns the action at that seam while its
            # input was copied exactly from the earlier segment.
            for segment in relevant:
                selected = (segment.origin_ids == origin_id).nonzero().squeeze(-1)
                rows = segment.origin_rows.index_select(0, selected)
                start = segment.base_position
                for name in _STREAM_FIELD_NAMES:
                    source = segment.values[name].index_select(0, selected)
                    kept = min(source.size(1), used - start)
                    if kept > 0:
                        assembled[name][rows, start : start + kept] = source[
                            :, :kept
                        ]
            action_mask = assembled["action_mask"]
            result[origin_id] = LatentRolloutBatch(
                **assembled,
                emit_mask=(
                    (assembled["actions"] == EMIT).float() * action_mask
                ),
                rewards=torch.zeros_like(action_mask),
                reward_scalar=torch.zeros(
                    origin.rows, dtype=torch.float32, device=cpu
                ),
                prompt_length=origin.prompt_length,
            )
        return result


@dataclass
class DecodeState:
    """Pausable actor state at one scalar absolute stream position."""

    wrapper: LatentThoughtModel
    generator: torch.Generator | None
    temperature: float
    top_p: float
    max_new_tokens: int
    max_stream: int
    prompt_length: int
    first_position: int
    position: int
    pin_emit: bool
    replay_storage: bool
    stop_tensor: Tensor | None
    cache_dtype: torch.dtype | None
    tensor_positions: bool
    compact_dead_ratio: float
    sync_every: int
    caches: list[tuple[Tensor, ...]]
    output: StepOutput
    emitted: Tensor
    ended: Tensor
    thinking_active: Tensor
    valid_slots: Tensor | None
    pad_lengths: Tensor | None
    position_index: Tensor | None
    slab: _RecordSlab
    slab_rows: Tensor
    ledger: _RecordLedger
    schedule_stats: TailScheduleStats | None = None
    completed: bool = False

    @property
    def rows(self) -> int:
        return self.emitted.numel()

    def active_mask(self) -> Tensor:
        initial = self.position == self.first_position
        return ~self.ended & (
            (self.emitted < self.max_new_tokens)
            | (self.thinking_active if initial else False)
        )

    def step_position(self, position: int) -> tuple[int | Tensor, Tensor | None]:
        if self.position_index is not None:
            self.position_index.fill_(position)
            step_position: int | Tensor = self.position_index
        else:
            step_position = position
        if self.valid_slots is None:
            return step_position, None
        mask = self.valid_slots[:, : position + 1]
        if position < self.prompt_length:
            mask = mask | (self.pad_lengths[:, None] > position)
        return step_position, mask


def _empty_record_values(
    rows: int,
    width: int,
    thought_dim: int,
    *,
    device: torch.device,
) -> dict[str, Tensor]:
    action_mask = torch.zeros((rows, width), dtype=torch.float32, device=device)
    thoughts = torch.zeros(
        (rows, width, thought_dim), dtype=torch.float32, device=device
    )
    return {
        "kind": torch.full(
            (rows, width), PAD_SLOT, dtype=torch.long, device=device
        ),
        "token_ids": torch.zeros(
            (rows, width), dtype=torch.long, device=device
        ),
        "thoughts": thoughts,
        "actions": torch.zeros(
            (rows, width), dtype=torch.long, device=device
        ),
        "action_mask": action_mask,
        "stop_mask": torch.zeros_like(action_mask),
        "old_stop_logprobs": torch.zeros_like(action_mask),
        "old_token_logprobs": torch.zeros_like(action_mask),
        "old_thought_logprobs": thoughts.new_zeros((rows, width, 0)),
        "old_thought_means": thoughts.new_zeros((rows, width, 0)),
        "old_thought_log_sigmas": thoughts.new_zeros((rows, width, 0)),
        "old_values": torch.zeros_like(action_mask),
    }


@torch.no_grad()
def start_decode_state(
    wrapper: LatentThoughtModel,
    prompt_ids: Tensor,
    prompt_lengths: Tensor,
    *,
    origin_id: int,
    prompt_repeats: int,
    max_new_tokens: int,
    max_stream_steps: int,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None = None,
    stop_ids: int | Sequence[int] | None = None,
    cache_dtype: torch.dtype | None = None,
    tensor_positions: bool = False,
    replay_storage: bool = True,
    pin_emit: bool = False,
    sync_every: int = 16,
    compact_dead_ratio: float = 0.25,
    schedule_stats: TailScheduleStats | None = None,
) -> DecodeState:
    """Prefill one common-width chunk and expose its first action state."""
    if prompt_ids.dim() != 2 or prompt_ids.size(1) < 1:
        raise ValueError("prompt_ids must be (batch, length>=1)")
    if prompt_lengths.shape != (prompt_ids.size(0),):
        raise ValueError("prompt_lengths must contain one length per prompt")
    if prompt_repeats < 1:
        raise ValueError("prompt_repeats must be positive")
    if sync_every < 1:
        raise ValueError("sync_every must be positive")
    if not 0.0 < compact_dead_ratio <= 1.0:
        raise ValueError("compact_dead_ratio must be in (0, 1]")
    if max_stream_steps < max_new_tokens + (0 if pin_emit else 1):
        raise ValueError("stream budget cannot fit the requested actions")

    device = prompt_ids.device
    prefix_rows, prompt_length = prompt_ids.shape
    rows = prefix_rows * prompt_repeats
    max_stream = prompt_length + max_stream_steps
    prompt_lengths = prompt_lengths.to(device=device, dtype=torch.long)
    if bool((prompt_lengths < 1).any()) or bool(
        (prompt_lengths > prompt_length).any()
    ):
        raise ValueError("prompt lengths must lie within the common width")
    pad_lengths = (
        prompt_length - prompt_lengths
    ).repeat_interleave(prompt_repeats)
    valid_slots = (
        torch.arange(max_stream, device=device)[None, :] >= pad_lengths[:, None]
    )
    stop_tensor = None
    if stop_ids is not None:
        ids = (stop_ids,) if isinstance(stop_ids, int) else tuple(stop_ids)
        if ids:
            stop_tensor = torch.tensor(ids, dtype=torch.long, device=device)

    prefix_caches = wrapper.make_generation_cache(
        prefix_rows, prompt_length, device, dtype=cache_dtype
    )
    prefix_valid = valid_slots[::prompt_repeats, :prompt_length]
    output = wrapper.prefill(prompt_ids, prefix_caches, prefix_valid)
    caches = wrapper.make_generation_cache(
        rows, max_stream, device, dtype=cache_dtype
    )
    for source_layer, target_layer in zip(
        output.caches, caches, strict=True
    ):
        for source, target in zip(source_layer, target_layer, strict=True):
            grouped = target.view(
                prefix_rows, prompt_repeats, *target.shape[1:]
            )
            grouped[:, :, :, :prompt_length].copy_(
                source[:, None].expand(
                    prefix_rows, prompt_repeats, *source.shape[1:]
                )
            )

    def expand(value: Tensor) -> Tensor:
        return value.repeat_interleave(prompt_repeats, dim=0)

    output = StepOutput(
        belief=expand(output.belief),
        predicted=expand(output.predicted),
        thought_log_sigma=expand(output.thought_log_sigma),
        input_latent=expand(output.input_latent),
        logits=expand(output.logits),
        caches=caches,
    )
    thought_dim = (
        wrapper.backbone.tok_emb.embedding_dim
        if replay_storage and not pin_emit
        else 0
    )
    values = _empty_record_values(
        rows, max_stream, thought_dim, device=device
    )
    prompt_valid = valid_slots[:, :prompt_length]
    values["kind"][:, :prompt_length] = torch.where(
        prompt_valid,
        values["kind"].new_full((), TOKEN_SLOT),
        values["kind"].new_full((), PAD_SLOT),
    )
    values["token_ids"][:, :prompt_length] = (
        prompt_ids.repeat_interleave(prompt_repeats, dim=0) * prompt_valid
    )
    slab = _RecordSlab(
        base_position=0,
        origin_ids=torch.full(
            (rows,), origin_id, dtype=torch.long, device=device
        ),
        origin_rows=torch.arange(rows, device=device),
        values=values,
    )
    # This is the shared LEFT-padded width.  ``split_rollout_groups`` later
    # subtracts each external true prompt length from it.
    origin = _Origin(origin_id, rows, prompt_length)
    return DecodeState(
        wrapper=wrapper,
        generator=generator,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        max_stream=max_stream,
        prompt_length=prompt_length,
        first_position=prompt_length - 1,
        position=prompt_length - 1,
        pin_emit=pin_emit,
        replay_storage=replay_storage,
        stop_tensor=stop_tensor,
        cache_dtype=cache_dtype,
        tensor_positions=tensor_positions,
        compact_dead_ratio=compact_dead_ratio,
        sync_every=sync_every,
        caches=caches,
        output=output,
        emitted=torch.zeros(rows, dtype=torch.long, device=device),
        ended=torch.zeros(rows, dtype=torch.bool, device=device),
        thinking_active=torch.full(
            (rows,), not pin_emit, dtype=torch.bool, device=device
        ),
        valid_slots=valid_slots,
        pad_lengths=pad_lengths,
        position_index=(
            torch.zeros((), dtype=torch.long, device=device)
            if tensor_positions
            else None
        ),
        slab=slab,
        slab_rows=torch.arange(rows, device=device),
        ledger=_RecordLedger((origin,)),
        schedule_stats=schedule_stats,
    )


def _compact_decode_rows(
    state: DecodeState,
    keep: Tensor,
    *,
    owning_cache: bool,
) -> None:
    """Compact model state while record rows retain stable attribution."""
    count = keep.numel()
    live_prefix = state.position + 1
    for layer, cache in enumerate(state.caches):
        compacted = []
        for tensor in cache:
            selected = tensor[:, :, :live_prefix].index_select(0, keep)
            if owning_cache:
                target = torch.empty(
                    (count, *tensor.shape[1:]),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                target[:, :, :live_prefix].copy_(selected)
            else:
                tensor[:count, :, :live_prefix].copy_(selected)
                target = tensor[:count]
            compacted.append(target)
        state.caches[layer] = tuple(compacted)
    state.output = StepOutput(
        belief=state.output.belief.index_select(0, keep),
        predicted=state.output.predicted.index_select(0, keep),
        thought_log_sigma=state.output.thought_log_sigma.index_select(0, keep),
        input_latent=state.output.input_latent.index_select(0, keep),
        logits=state.output.logits.index_select(0, keep),
        caches=state.caches,
    )
    state.emitted = state.emitted.index_select(0, keep)
    state.ended = state.ended.index_select(0, keep)
    state.thinking_active = state.thinking_active.index_select(0, keep)
    state.slab_rows = state.slab_rows.index_select(0, keep)
    if state.valid_slots is not None:
        state.valid_slots = state.valid_slots.index_select(0, keep)
        state.pad_lengths = state.pad_lengths.index_select(0, keep)


def _rebase_records(state: DecodeState, cpu: torch.device) -> None:
    """Flush a chunk slab and replace it by a survivor-sized continuation."""
    seam_kind, seam_token_ids, seam_thoughts = state.ledger.flush(
        state.slab, state.position, cpu=cpu
    )
    seam_kind = seam_kind.index_select(0, state.slab_rows)
    seam_token_ids = seam_token_ids.index_select(0, state.slab_rows)
    seam_thoughts = seam_thoughts.index_select(0, state.slab_rows)
    origin_ids = state.slab.origin_ids.index_select(0, state.slab_rows)
    origin_rows = state.slab.origin_rows.index_select(0, state.slab_rows)
    width = state.max_stream - state.position
    thought_dim = state.slab.values["thoughts"].size(-1)
    values = _empty_record_values(
        state.rows, width, thought_dim, device=state.emitted.device
    )
    values["kind"][:, 0].copy_(seam_kind)
    values["token_ids"][:, 0].copy_(seam_token_ids)
    if thought_dim:
        values["thoughts"][:, 0].copy_(seam_thoughts)
    state.slab = _RecordSlab(
        base_position=state.position,
        origin_ids=origin_ids,
        origin_rows=origin_rows,
        values=values,
    )
    state.slab_rows = torch.arange(state.rows, device=state.emitted.device)


@torch.no_grad()
def advance_decode_state(
    state: DecodeState,
    *,
    park_live_rows: int | None = None,
    stop_position: int | None = None,
    offload_device: torch.device = torch.device("cpu"),
) -> bool:
    """Advance until completion or pause before an aligned scalar action.

    Returns ``True`` only when the state completed.  A paused state has exact
    active survivors, owns compact KV storage, and has a host-resident record
    prefix plus a survivor-sized device continuation slab.
    """
    if state.completed:
        return True
    if park_live_rows is not None and park_live_rows < 1:
        raise ValueError("park_live_rows must be positive")
    if stop_position is not None:
        if stop_position < state.position:
            raise ValueError("cannot advance backward to a stop position")
        if (
            stop_position - state.first_position
        ) % state.sync_every:
            raise ValueError("stop position must lie on the synchronization grid")

    while state.position < state.max_stream - 1:
        initial = state.position == state.first_position
        active = state.active_mask()
        at_sync = (
            (state.position - state.first_position) % state.sync_every == 0
        )
        if at_sync:
            active_count = int(active.sum())
            if active_count == 0:
                state.completed = True
                break
            pause_for_position = (
                stop_position is not None
                and state.position == stop_position
            )
            pause_for_tail = (
                park_live_rows is not None
                and not initial
                and active_count <= park_live_rows
            )
            if pause_for_position or pause_for_tail:
                keep = active.nonzero().squeeze(-1)
                _compact_decode_rows(state, keep, owning_cache=True)
                _rebase_records(state, offload_device)
                return False
            dead = active.numel() - active_count
            if dead >= max(1, int(active.numel() * state.compact_dead_ratio)):
                keep = active.nonzero().squeeze(-1)
                _compact_decode_rows(state, keep, owning_cache=False)
                active = active.index_select(0, keep)

        belief = state.output.belief
        if state.pin_emit or initial:
            action = torch.full(
                (state.rows,),
                THINK if initial and not state.pin_emit else EMIT,
                dtype=torch.long,
                device=belief.device,
            )
            stop_logprob = None
        else:
            sampled_action = state.wrapper.gate.sample_action(
                belief, generator=state.generator
            )
            action = torch.where(
                state.thinking_active,
                sampled_action,
                sampled_action.new_full((), EMIT),
            )
            stop_logprob = None

        token = top_p_sample(
            state.output.logits,
            state.temperature,
            state.top_p,
            generator=state.generator,
        )
        thought = (
            None
            if state.pin_emit
            else state.wrapper.transition.sample_latent(
                state.output.predicted,
                state.output.thought_log_sigma,
                generator=state.generator,
            )
        )
        record = active
        column = state.position - state.slab.base_position
        next_column = column + 1
        row_slots = (state.slab_rows, column)
        next_slots = (state.slab_rows, next_column)
        values = state.slab.values
        ones = values["action_mask"].new_ones(())
        values["action_mask"][row_slots] = torch.where(
            record, ones, values["action_mask"][row_slots]
        )
        sampled_stop = None
        if not state.pin_emit and not initial:
            sampled_stop = record & state.thinking_active
            values["stop_mask"][row_slots] = torch.where(
                sampled_stop, ones, values["stop_mask"][row_slots]
            )
            if stop_logprob is not None:
                values["old_stop_logprobs"][row_slots] = torch.where(
                    sampled_stop,
                    stop_logprob.float(),
                    values["old_stop_logprobs"][row_slots],
                )
        values["actions"][row_slots] = torch.where(
            record, action, values["actions"][row_slots]
        )
        emits = record & (action == EMIT)
        thinks = record & (action == THINK)
        if sampled_stop is not None:
            state.thinking_active &= ~(emits & sampled_stop)

        next_kind = torch.where(
            emits,
            values["kind"].new_full((state.rows,), TOKEN_SLOT),
            torch.where(
                thinks,
                values["kind"].new_full((state.rows,), THOUGHT_SLOT),
                values["kind"].new_full((state.rows,), PAD_SLOT),
            ),
        )
        next_token_ids = torch.where(
            emits, token, token.new_zeros((state.rows,))
        )
        values["kind"][next_slots] = next_kind
        values["token_ids"][next_slots] = next_token_ids
        if state.replay_storage and thought is not None:
            values["thoughts"][next_slots] = torch.where(
                thinks[:, None], thought, values["thoughts"][next_slots]
            )
        state.emitted += emits.long()
        if state.stop_tensor is not None:
            state.ended |= emits & torch.isin(token, state.stop_tensor)

        if thought is None:
            next_input = state.wrapper.embed_tokens(next_token_ids[:, None])
        else:
            next_input = torch.where(
                (next_kind == THOUGHT_SLOT)[:, None, None],
                state.wrapper.thought_input(thought),
                state.wrapper.embed_tokens(next_token_ids[:, None]),
            )
        next_position = state.position + 1
        step_position, key_mask = state.step_position(next_position)
        if state.schedule_stats is not None:
            state.schedule_stats.decode_steps += 1
            state.schedule_stats.row_steps += state.rows
            if stop_position is not None:
                state.schedule_stats.catchup_decode_steps += 1
        state.output = state.wrapper.step(
            next_input, state.caches, step_position, key_mask
        )
        state.caches = state.output.caches
        state.position = next_position

    state.completed = True
    return True


@torch.no_grad()
def merge_aligned_decode_states(
    first: DecodeState,
    second: DecodeState,
    *additional: DecodeState,
) -> DecodeState:
    """Merge aligned survivors with one destination allocation per tensor."""
    states = (first, second, *additional)
    if any(state.completed for state in states):
        raise ValueError("cannot merge a completed decode state")
    comparable = (
        "wrapper",
        "generator",
        "temperature",
        "top_p",
        "max_new_tokens",
        "max_stream",
        "prompt_length",
        "first_position",
        "position",
        "pin_emit",
        "replay_storage",
        "cache_dtype",
        "tensor_positions",
        "compact_dead_ratio",
        "sync_every",
        "schedule_stats",
    )
    for state in states[1:]:
        for name in comparable:
            left, right = getattr(first, name), getattr(state, name)
            if name in {"wrapper", "generator", "schedule_stats"}:
                equal = left is right
            else:
                equal = left == right
            if not equal:
                raise ValueError(f"decode states differ in {name}")
        if first.stop_tensor is None or state.stop_tensor is None:
            if first.stop_tensor is not state.stop_tensor:
                raise ValueError("decode states differ in stop ids")
        elif not torch.equal(first.stop_tensor, state.stop_tensor):
            raise ValueError("decode states differ in stop ids")
        if (first.valid_slots is None) != (state.valid_slots is None):
            raise ValueError("decode states differ in key-mask storage")
        if (first.pad_lengths is None) != (state.pad_lengths is None):
            raise ValueError("decode states differ in pad-length storage")
    for state in states:
        if state.slab.base_position != state.position:
            raise ValueError("decode state records were not rebased at the seam")
        if len(state.caches) != len(first.caches):
            raise ValueError("decode cache layer counts differ")
        for state_layer, first_layer in zip(
            state.caches, first.caches, strict=True
        ):
            if len(state_layer) != len(first_layer):
                raise ValueError("decode cache tensor counts differ")
    seen_origins: set[int] = set()
    thought_dim = first.slab.values["thoughts"].size(-1)
    for state in states:
        overlap = seen_origins & state.ledger.origins.keys()
        if overlap:
            raise ValueError(f"record origins overlap: {sorted(overlap)}")
        seen_origins.update(state.ledger.origins)
        if thought_dim != state.slab.values["thoughts"].size(-1):
            raise ValueError("decode record thought dimensions differ")
        for state_layer, first_layer in zip(
            state.caches, first.caches, strict=True
        ):
            for source, exemplar in zip(
                state_layer, first_layer, strict=True
            ):
                if source.shape[1:] != exemplar.shape[1:]:
                    raise ValueError("decode cache layouts differ")
                if (
                    source.device != exemplar.device
                    or source.dtype != exemplar.dtype
                ):
                    raise ValueError("decode cache devices or dtypes differ")

    device = first.emitted.device
    rows = sum(state.rows for state in states)
    live_prefix = first.position + 1

    def cat_output(name: str) -> Tensor:
        return torch.cat(
            tuple(getattr(state.output, name) for state in states), dim=0
        )

    output_values = {
        name: cat_output(name)
        for name in (
            "belief",
            "predicted",
            "thought_log_sigma",
            "input_latent",
            "logits",
        )
    }
    caches: list[tuple[Tensor, ...]] = []
    for layer_index, first_layer in enumerate(first.caches):
        layer = []
        for tensor_index, exemplar in enumerate(first_layer):
            sources = [
                state.caches[layer_index][tensor_index]
                for state in states
            ]
            target = torch.empty(
                (rows, *exemplar.shape[1:]),
                dtype=exemplar.dtype,
                device=device,
            )
            offset = 0
            for state, source in zip(states, sources, strict=True):
                target[
                    offset : offset + state.rows, :, :live_prefix
                ].copy_(source[:, :, :live_prefix])
                offset += state.rows
            layer.append(target)
        caches.append(tuple(layer))
        # Validation is complete before allocation starts, so source ownership
        # can be released one layer at a time. This bounds the merge transient
        # to the source caches plus one destination layer instead of retaining
        # complete source and destination cache sets together.
        for state in states:
            state.caches[layer_index] = ()

    output = StepOutput(
        belief=output_values["belief"],
        predicted=output_values["predicted"],
        thought_log_sigma=output_values["thought_log_sigma"],
        input_latent=output_values["input_latent"],
        logits=output_values["logits"],
        caches=caches,
    )
    for state in states[1:]:
        first.ledger.absorb(state.ledger)
    width = first.max_stream - first.position
    values = _empty_record_values(rows, width, thought_dim, device=device)
    for name in ("kind", "token_ids", "thoughts"):
        values[name][:, 0] = torch.cat(
            tuple(state.slab.values[name][:, 0] for state in states), dim=0
        )
    slab = _RecordSlab(
        base_position=first.position,
        origin_ids=torch.cat(
            tuple(state.slab.origin_ids for state in states), dim=0
        ),
        origin_rows=torch.cat(
            tuple(state.slab.origin_rows for state in states), dim=0
        ),
        values=values,
    )
    first.caches = caches
    first.output = output
    first.emitted = torch.cat(tuple(state.emitted for state in states))
    first.ended = torch.cat(tuple(state.ended for state in states))
    first.thinking_active = torch.cat(
        tuple(state.thinking_active for state in states)
    )
    first.valid_slots = (
        None
        if first.valid_slots is None
        else torch.cat(
            tuple(state.valid_slots for state in states), dim=0
        )
    )
    first.pad_lengths = (
        None
        if first.pad_lengths is None
        else torch.cat(
            tuple(state.pad_lengths for state in states), dim=0
        )
    )
    first.position_index = (
        torch.zeros((), dtype=torch.long, device=device)
        if first.tensor_positions
        else None
    )
    first.slab = slab
    first.slab_rows = torch.arange(rows, device=device)
    return first


@torch.no_grad()
def rollout_scalar_tail_chunks(
    wrapper: LatentThoughtModel,
    prompt_chunks: Sequence[Tensor],
    prompt_length_chunks: Sequence[Tensor],
    *,
    prompt_repeats: int,
    max_new_tokens: int,
    max_stream_steps: int,
    temperature: float,
    top_p: float,
    tail_rows: int = 16,
    generator: torch.Generator | None = None,
    stop_ids: int | Sequence[int] | None = None,
    cache_dtype: torch.dtype | None = None,
    tensor_positions: bool = False,
    replay_storage: bool = True,
    pin_emit: bool = False,
    sync_every: int = 16,
    compact_dead_ratio: float = 0.25,
    offload_device: torch.device = torch.device("cpu"),
    schedule_stats: TailScheduleStats | None = None,
) -> list[LatentRolloutBatch]:
    """Roll common-width chunks with a bounded scalar-position tail cohort.

    Returned batches correspond one-for-one with ``prompt_chunks`` and retain
    each chunk's original repeated-prompt row order.
    """
    if not prompt_chunks:
        return []
    if len(prompt_chunks) != len(prompt_length_chunks):
        raise ValueError("prompt chunks and length chunks must align")
    widths = {chunk.size(1) for chunk in prompt_chunks}
    if len(widths) != 1:
        raise ValueError("all chunks must use one common prompt width")
    if tail_rows < 1:
        raise ValueError("tail_rows must be positive")
    if schedule_stats is not None:
        if any(value for value in vars(schedule_stats).values()):
            raise ValueError("schedule_stats must be empty")
        schedule_stats.chunks = len(prompt_chunks)

    # Prefix parking always targets host memory, even when the caller needs
    # device batches back (critic warmup does).  Keeping prefixes on CUDA in
    # that case would make chunk N's dense thought slab coexist with chunk
    # N+1 and defeat the scheduler's memory bound.  Final compact batches are
    # moved to ``offload_device`` only after recurrent decode has finished.
    record_device = torch.device("cpu")
    finished: dict[int, LatentRolloutBatch] = {}
    cohort: list[DecodeState] = []
    cohort_rows = 0
    merged: DecodeState | None = None

    def finish(state: DecodeState) -> None:
        finished.update(
            state.ledger.finalize(
                state.slab, state.position, cpu=record_device
            )
        )

    def note_cohort_rows() -> None:
        if schedule_stats is not None:
            schedule_stats.cohort_rows_max = max(
                schedule_stats.cohort_rows_max, cohort_rows
            )

    def merge_cohort() -> DecodeState:
        nonlocal cohort, cohort_rows
        if not cohort:
            raise RuntimeError("cannot merge an empty rollout cohort")
        if len(cohort) == 1:
            result = cohort[0]
        else:
            result = merge_aligned_decode_states(
                cohort[0], cohort[1], *cohort[2:]
            )
            if schedule_stats is not None:
                schedule_stats.merges += len(cohort) - 1
                schedule_stats.merge_events += 1
        cohort.clear()
        cohort_rows = 0
        return result

    for origin_id, (prompts, lengths) in enumerate(
        zip(prompt_chunks, prompt_length_chunks, strict=True)
    ):
        state = start_decode_state(
            wrapper,
            prompts,
            lengths,
            origin_id=origin_id,
            prompt_repeats=prompt_repeats,
            max_new_tokens=max_new_tokens,
            max_stream_steps=max_stream_steps,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
            stop_ids=stop_ids,
            cache_dtype=cache_dtype,
            tensor_positions=tensor_positions,
            replay_storage=replay_storage,
            pin_emit=pin_emit,
            sync_every=sync_every,
            compact_dead_ratio=compact_dead_ratio,
            schedule_stats=schedule_stats,
        )
        if not cohort:
            completed = advance_decode_state(
                state,
                park_live_rows=tail_rows,
                offload_device=record_device,
            )
            if completed:
                finish(state)
                continue
            cohort = [state]
            cohort_rows = state.rows
            if schedule_stats is not None:
                schedule_stats.parks += 1
            note_cohort_rows()
            continue

        # At every admission boundary the parked cohort contains at most
        # ``tail_rows`` rows. A fresh dense chunk can therefore coexist with
        # it under the same B+tail steady-state bound as the original rolling
        # scheduler. States whose union still fits the tail remain separate:
        # merging them now would allocate/copy KV and flush a new record seam,
        # then immediately pause again without executing a model step.
        completed = advance_decode_state(
            state,
            stop_position=cohort[0].position,
            offload_device=record_device,
        )
        if completed:
            finish(state)
            continue
        cohort.append(state)
        cohort_rows += state.rows
        if schedule_stats is not None:
            schedule_stats.cohort_admissions += 1
        note_cohort_rows()
        if cohort_rows <= tail_rows:
            continue

        # The admitted union outgrew the parked-row budget. Merge it once,
        # advance until it is sparse again, and only then prefill another
        # dense chunk. This has the same model-step/RNG schedule and row order
        # as eager rolling pairwise merges, but eliminates zero-step merges.
        merged = merge_cohort()
        if schedule_stats is not None:
            schedule_stats.cohort_rollovers += 1
        completed = advance_decode_state(
            merged,
            park_live_rows=tail_rows,
            offload_device=record_device,
        )
        if completed:
            finish(merged)
            merged = None
            continue
        cohort = [merged]
        cohort_rows = merged.rows
        if schedule_stats is not None:
            schedule_stats.parks += 1
        note_cohort_rows()
        merged = None

    if cohort:
        merged = merge_cohort()
        advance_decode_state(merged, offload_device=record_device)
        finish(merged)
        merged = None
    if set(finished) != set(range(len(prompt_chunks))):
        raise RuntimeError("scalar tail scheduler lost a rollout chunk")
    ordered = [finished[index] for index in range(len(prompt_chunks))]
    if schedule_stats is not None:
        per_chunk_actions = [
            batch.action_mask.sum(dim=1) for batch in ordered
        ]
        schedule_stats.useful_actions = sum(
            int(actions.sum()) for actions in per_chunk_actions
        )
        schedule_stats.lockstep_decode_steps = sum(
            min(
                (int(actions.max()) + sync_every - 1)
                // sync_every
                * sync_every,
                max_stream_steps,
            )
            for actions in per_chunk_actions
        )
    # Do not retain the final recurrent state and its full KV allocation while
    # optional finalized batches move back to the GPU (critic warmup).
    cohort.clear()
    merged = None
    state = None
    if offload_device.type != "cpu":
        ordered = [batch.to(offload_device) for batch in ordered]
    return ordered
