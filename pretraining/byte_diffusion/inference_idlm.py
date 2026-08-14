"""Lossless I-DLM serving oracles and the fused 2N-1 replay schedule.

The replay implementation deliberately performs two model forwards per
iteration: one MASK stride and one clean verification pass.  It is slower than
the paper's fused cached schedule, but it is distributionally exact and never
promotes proposal K/V as clean state.  A production cache must implement the
transactional protocol below before replacing this oracle; arbitrary partial
patch promotion is not approximated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from .idlm import categorical_from_uniform, isd_commit_prefix


@runtime_checkable
class CausalIDLM(Protocol):
    """Minimum model surface required by the full-replay oracle."""

    config: object

    def forward_sequence(
        self, ids: Tensor, valid: Tensor, *, positions: Tensor | None = None
    ) -> Tensor: ...


@dataclass(frozen=True)
class IDLMScratchHandle:
    """Typed identity for unresolved proposal K/V owned by a serving cache."""

    generation: int
    prefix_lengths: Tensor
    proposal_slots: int


@runtime_checkable
class IDLMServingCache(Protocol):
    """Requirements for a future cached implementation.

    ``begin_isd`` must fork immutable clean-prefix K/V.  ``commit`` may promote
    only K/V recomputed from the corrected clean ids and only through the valid
    accepted prefix.  Proposal/MASK K/V and entries after a rejection must be
    discarded.  Implementations with patch-local state must replay the zero to
    three incomplete committed bytes before closing a patch.
    """

    def prefill(self, ids: Tensor, valid: Tensor) -> None: ...

    def begin_isd(self, proposal_slots: int) -> IDLMScratchHandle: ...

    def commit(
        self, handle: IDLMScratchHandle, clean_ids: Tensor, valid: Tensor
    ) -> None: ...

    def abort(self, handle: IDLMScratchHandle) -> None: ...


@dataclass(frozen=True)
class IDLMReplayResult:
    ids: Tensor
    valid: Tensor
    prompt_lengths: Tensor
    lengths: Tensor
    finished: Tensor
    iterations: int
    model_forwards: int
    physical_proposal_slots: int
    eligible_proposals: int
    accepted_proposals: int
    model_query_tokens: int
    fused_2n_minus_1_forwards: int = 0
    cache_backed: bool = False

    @property
    def proposed_tokens(self) -> int:
        """Compatibility spelling for physical proposal compute."""

        return self.physical_proposal_slots

    @property
    def acceptance_rate(self) -> float:
        return (
            self.accepted_proposals / self.eligible_proposals
            if self.eligible_proposals
            else 0.0
        )

    @property
    def model_extend_forwards(self) -> int:
        return self.model_forwards

    @property
    def clean_commit_forwards(self) -> int:
        # Strict causal ISD commits corrected clean K/V in the same extend;
        # unlike block diffusion it has no separate zero-output commit pass.
        return 0

    @property
    def generated_valid(self) -> Tensor:
        positions = torch.arange(self.ids.shape[1], device=self.ids.device)
        return self.valid & (positions[None] >= self.prompt_lengths[:, None])


def _gather_logits(logits: Tensor, positions: Tensor) -> Tensor:
    if logits.ndim != 3 or positions.dtype != torch.long:
        raise ValueError("logit gather requires [batch,length,vocab] and int64 positions")
    if positions.ndim == 1:
        return logits.gather(
            1,
            positions[:, None, None].expand(-1, 1, logits.shape[-1]),
        )[:, 0]
    if positions.ndim == 2:
        return logits.gather(
            1, positions[..., None].expand(-1, -1, logits.shape[-1])
        )
    raise ValueError("logit positions must have rank one or two")


def _probabilities(logits: Tensor, temperature: float) -> Tensor:
    if not temperature > 0:
        raise ValueError("temperature must be positive")
    return (logits.float() / temperature).softmax(-1)


def _trained_stride(model: CausalIDLM) -> int:
    try:
        stride = int(model.config.block_size)
    except AttributeError as error:
        raise ValueError("I-DLM model config must bind its trained block_size") from error
    if stride <= 0:
        raise ValueError("trained I-DLM stride must be positive")
    return stride


def _require_trained_stride(model: CausalIDLM, stride: int) -> None:
    trained = _trained_stride(model)
    if stride != trained:
        raise ValueError(
            f"serving stride {stride} differs from trained stride {trained}; "
            "run a declared stride curriculum and checkpoint the final topology"
        )


@torch.no_grad()
def generate_idlm_full_replay(
    model: CausalIDLM,
    prompt_ids: Tensor,
    prompt_valid: Tensor,
    *,
    stride: int,
    max_new_tokens: int,
    generator: torch.Generator | None = None,
    temperature: float = 1.0,
) -> IDLMReplayResult:
    """Generate exactly from the model's clean causal distribution via ISD.

    Tensor scatter/gather handles every batch row together.  The only Python
    loop is over decoding iterations, never over rows or proposal positions.
    """

    if stride <= 0 or max_new_tokens <= 0:
        raise ValueError("stride and max_new_tokens must be positive")
    _require_trained_stride(model, stride)
    if (
        prompt_ids.dtype != torch.long
        or prompt_ids.ndim != 2
        or prompt_ids.shape != prompt_valid.shape
        or prompt_valid.dtype != torch.bool
    ):
        raise ValueError("prompt ids/validity must be aligned rank-2 tensors")
    prompt_lengths = prompt_valid.sum(1)
    if bool((prompt_lengths <= 0).any()):
        raise ValueError("every I-DLM prompt needs at least one committed atom")
    prompt_columns = torch.arange(prompt_ids.shape[1], device=prompt_ids.device)
    if not torch.equal(
        prompt_valid,
        prompt_columns[None] < prompt_lengths[:, None],
    ):
        raise ValueError("I-DLM replay prompts must be contiguous prefixes")

    config = model.config
    try:
        pad_id = int(config.pad_id)
        mask_id = int(config.mask_id)
        eot_id = int(config.eot_id)
    except AttributeError as error:
        raise ValueError("I-DLM model config must define pad_id, mask_id, and eot_id") from error

    batch = prompt_ids.shape[0]
    capacity = prompt_ids.shape[1] + max_new_tokens
    ids = torch.full(
        (batch, capacity), pad_id, dtype=torch.long, device=prompt_ids.device
    )
    ids[:, : prompt_ids.shape[1]] = torch.where(
        prompt_valid, prompt_ids, torch.as_tensor(pad_id, device=prompt_ids.device)
    )
    lengths = prompt_lengths.clone()
    generation_limits = prompt_lengths + max_new_tokens
    finished = torch.zeros(batch, dtype=torch.bool, device=prompt_ids.device)
    storage_positions = torch.arange(capacity, device=prompt_ids.device)[None]
    proposal_count = stride - 1
    iterations = 0
    model_forwards = 0
    proposed_tokens = torch.zeros((), dtype=torch.long, device=ids.device)
    eligible_proposals = torch.zeros((), dtype=torch.long, device=ids.device)
    accepted_proposals = torch.zeros((), dtype=torch.long, device=ids.device)
    model_query_tokens = torch.zeros((), dtype=torch.long, device=ids.device)

    while bool((~finished & (lengths < generation_limits)).any()):
        active = ~finished & (lengths < generation_limits)
        iteration_lengths = lengths.clone()
        work_width = capacity + stride
        work_positions = torch.arange(work_width, device=ids.device)[None]
        committed = torch.full(
            (batch, work_width), pad_id, dtype=torch.long, device=ids.device
        )
        committed[:, :capacity] = ids

        draft_stop = lengths + stride - 1
        draft_valid = work_positions < torch.where(active, draft_stop, lengths)[:, None]
        draft_mask = (
            active[:, None]
            & (work_positions >= lengths[:, None])
            & (work_positions < draft_stop[:, None])
        )
        draft_ids = torch.where(
            draft_mask,
            torch.as_tensor(mask_id, device=ids.device),
            committed,
        )
        draft_logits = model.forward_sequence(draft_ids, draft_valid)
        model_forwards += 1
        model_query_tokens += draft_valid.sum()

        anchor_p = _probabilities(
            _gather_logits(draft_logits, lengths - 1), temperature
        )
        anchor_ids = categorical_from_uniform(
            anchor_p,
            torch.rand(batch, device=ids.device, generator=generator),
        )
        proposal_positions = lengths[:, None] + torch.arange(
            proposal_count, device=ids.device
        )[None]
        q = _probabilities(
            _gather_logits(draft_logits, proposal_positions), temperature
        )
        proposal_ids = categorical_from_uniform(
            q,
            torch.rand(
                (batch, proposal_count), device=ids.device, generator=generator
            ),
        )
        candidates = torch.cat((anchor_ids[:, None], proposal_ids), dim=1)

        candidate_offsets = torch.arange(stride, device=ids.device)
        candidate_positions = lengths[:, None] + candidate_offsets[None]
        candidate_source = (work_positions - lengths[:, None]).clamp(0, stride - 1)
        candidate_values = candidates.gather(1, candidate_source)
        candidate_active = (
            active[:, None]
            & (work_positions >= lengths[:, None])
            & (work_positions < (lengths + stride)[:, None])
        )
        verify_ids = torch.where(candidate_active, candidate_values, committed)
        verify_valid = work_positions < torch.where(
            active, lengths + stride, lengths
        )[:, None]
        verify_logits = model.forward_sequence(verify_ids, verify_valid)
        model_forwards += 1
        model_query_tokens += verify_valid.sum()

        p_positions = lengths[:, None] + torch.arange(
            proposal_count, device=ids.device
        )[None]
        p = _probabilities(_gather_logits(verify_logits, p_positions), temperature)
        bonus_p = _probabilities(
            _gather_logits(verify_logits, lengths + stride - 1), temperature
        )
        commit = isd_commit_prefix(
            anchor_ids,
            p,
            q,
            proposal_ids,
            accept_uniforms=torch.rand(
                (batch, proposal_count), device=ids.device, generator=generator
            ),
            residual_uniforms=torch.rand(
                (batch, proposal_count), device=ids.device, generator=generator
            ),
            bonus_p=bonus_p,
            bonus_uniforms=torch.rand(batch, device=ids.device, generator=generator),
        )

        commit_width = commit.ids.shape[1]
        commit_columns = torch.arange(commit_width, device=ids.device)
        eot = commit.valid & commit.ids.eq(eot_id)
        first_eot = torch.where(
            eot,
            commit_columns[None],
            torch.full_like(commit_columns[None], commit_width),
        ).amin(1)
        commit_valid = (
            commit.valid
            & active[:, None]
            & (commit_columns[None] <= first_eot[:, None])
            & (
                lengths[:, None] + commit_columns[None]
                < generation_limits[:, None]
            )
        )

        relative = storage_positions - lengths[:, None]
        gather_relative = relative.clamp(0, commit_width - 1)
        write_values = commit.ids.gather(1, gather_relative)
        write = (
            (relative >= 0)
            & (relative < commit_width)
            & commit_valid.gather(1, gather_relative)
        )
        ids = torch.where(write, write_values, ids)
        written_eot = (write & write_values.eq(eot_id)).any(1)
        lengths = lengths + commit_valid.sum(1)
        finished |= written_eot

        iterations += 1
        proposed_tokens += active.sum() * proposal_count
        semantic_proposals = active[:, None] & (
            iteration_lengths[:, None]
            + 1
            + torch.arange(proposal_count, device=ids.device)[None]
            < generation_limits[:, None]
        ) & ~anchor_ids.eq(eot_id)[:, None]
        inspected_proposals = (
            semantic_proposals
            & commit.valid[:, 1 : 1 + proposal_count]
        )
        if proposal_count:
            proposal_commit_ids = commit.ids[:, 1 : 1 + proposal_count]
            proposal_columns = torch.arange(proposal_count, device=ids.device)[None]
            first_proposal_eot = torch.where(
                inspected_proposals & proposal_commit_ids.eq(eot_id),
                proposal_columns,
                proposal_count,
            ).amin(1)
            inspected_proposals &= proposal_columns <= first_proposal_eot[:, None]
        eligible_proposals += inspected_proposals.sum()
        accepted_proposals += (
            commit.proposal_accepted
            & inspected_proposals
        ).sum()

    valid = storage_positions < lengths[:, None]
    return IDLMReplayResult(
        ids=ids,
        valid=valid,
        prompt_lengths=prompt_lengths,
        lengths=lengths,
        finished=finished,
        iterations=iterations,
        model_forwards=model_forwards,
        physical_proposal_slots=int(proposed_tokens),
        eligible_proposals=int(eligible_proposals),
        accepted_proposals=int(accepted_proposals),
        model_query_tokens=int(model_query_tokens),
    )


@torch.no_grad()
def generate_idlm_fused_replay(
    model: CausalIDLM,
    prompt_ids: Tensor,
    prompt_valid: Tensor,
    *,
    stride: int,
    max_new_tokens: int,
    generator: torch.Generator | None = None,
    temperature: float = 1.0,
) -> IDLMReplayResult:
    """Exact ISD with the paper's fused ``2N-1`` extend schedule.

    This implementation fuses introspection of ``N-1`` pending proposals with
    ``N`` physical MASK queries for the next stride. The final MASK query is
    systems padding and is never counted in the semantic acceptance
    denominator. The current standalone model has no persistent KV API, so
    every extend is honestly replayed over the clean prefix; ``cache_backed``
    is therefore false. The state transitions (bootstrap, verify, trim, and
    corrected-prefix commit) are nevertheless the exact transactional
    protocol a paged-KV lowering must preserve.
    """

    if stride <= 0 or max_new_tokens <= 0:
        raise ValueError("stride and max_new_tokens must be positive")
    _require_trained_stride(model, stride)
    if (
        prompt_ids.dtype != torch.long
        or prompt_ids.ndim != 2
        or prompt_ids.shape != prompt_valid.shape
        or prompt_valid.dtype != torch.bool
    ):
        raise ValueError("prompt ids/validity must be aligned rank-2 tensors")
    prompt_lengths = prompt_valid.sum(1)
    if bool((prompt_lengths <= 0).any()):
        raise ValueError("every I-DLM prompt needs at least one committed atom")
    prompt_columns = torch.arange(prompt_ids.shape[1], device=prompt_ids.device)
    if not torch.equal(
        prompt_valid, prompt_columns[None] < prompt_lengths[:, None]
    ):
        raise ValueError("I-DLM prompts must be contiguous prefixes")
    try:
        pad_id = int(model.config.pad_id)
        mask_id = int(model.config.mask_id)
        eot_id = int(model.config.eot_id)
    except AttributeError as error:
        raise ValueError("I-DLM config must define PAD, MASK, and EOT") from error

    batch = prompt_ids.shape[0]
    capacity = prompt_ids.shape[1] + max_new_tokens
    ids = torch.full(
        (batch, capacity), pad_id, dtype=torch.long, device=prompt_ids.device
    )
    ids[:, : prompt_ids.shape[1]] = torch.where(
        prompt_valid,
        prompt_ids,
        torch.as_tensor(pad_id, device=prompt_ids.device),
    )
    lengths = prompt_lengths.clone()
    limits = prompt_lengths + max_new_tokens
    finished = torch.zeros(batch, dtype=torch.bool, device=ids.device)
    storage = torch.arange(capacity, device=ids.device)[None]
    proposal_count = stride - 1
    pending_ids = torch.full(
        (batch, proposal_count), pad_id, dtype=torch.long, device=ids.device
    )
    pending_q = torch.empty(
        (batch, proposal_count, int(model.config.output_size)),
        dtype=torch.float32,
        device=ids.device,
    )
    has_pending = torch.zeros(batch, dtype=torch.bool, device=ids.device)
    iterations = 0
    model_forwards = 0
    physical_slots = torch.zeros((), dtype=torch.long, device=ids.device)
    eligible_proposals = torch.zeros((), dtype=torch.long, device=ids.device)
    accepted_proposals = torch.zeros((), dtype=torch.long, device=ids.device)
    model_query_tokens = torch.zeros((), dtype=torch.long, device=ids.device)
    fused_forwards = 0

    def append_committed(commit_ids: Tensor, commit_valid: Tensor, rows: Tensor) -> None:
        nonlocal ids, lengths, finished
        if rows.numel() == 0:
            return
        row_lengths = lengths.index_select(0, rows)
        width = commit_ids.shape[1]
        offsets = torch.arange(width, device=ids.device)[None]
        eot = commit_valid & commit_ids.eq(eot_id)
        first_eot = torch.where(eot, offsets, width).amin(1)
        allowed = (
            commit_valid
            & (offsets <= first_eot[:, None])
            & (row_lengths[:, None] + offsets < limits.index_select(0, rows)[:, None])
        )
        absolute = row_lengths[:, None] + offsets
        target_rows = rows[:, None].expand_as(absolute)
        ids[target_rows[allowed], absolute[allowed]] = commit_ids[allowed]
        lengths.index_add_(0, rows, allowed.sum(1))
        finished[rows] |= (allowed & commit_ids.eq(eot_id)).any(1)

    while bool((~finished & (lengths < limits)).any()):
        live = ~finished & (lengths < limits)

        # A ragged prompt/rejection tail is advanced with exact AR anchors
        # until the trained proposal-block boundary. No MASK is issued in this
        # phase, so serving never exposes a proposal topology absent in train.
        ragged = live & ~has_pending & lengths.remainder(stride).ne(0)
        if bool(ragged.any()):
            rows = ragged.nonzero(as_tuple=False).flatten()
            row_ids = ids.index_select(0, rows)
            row_lengths = lengths.index_select(0, rows)
            valid = torch.arange(capacity, device=ids.device)[None] < row_lengths[:, None]
            logits = model.forward_sequence(row_ids, valid)
            model_forwards += 1
            model_query_tokens += valid.sum()
            p = _probabilities(
                _gather_logits(logits, row_lengths - 1),
                temperature,
            )
            anchors = categorical_from_uniform(
                p,
                torch.rand(rows.numel(), device=ids.device, generator=generator),
            )
            append_committed(
                anchors[:, None],
                torch.ones((rows.numel(), 1), dtype=torch.bool, device=ids.device),
                rows,
            )
            iterations += 1
            continue

        live = ~finished & (lengths < limits)
        live_rows = live.nonzero(as_tuple=False).flatten()
        local_pending = has_pending.index_select(0, live_rows)
        bootstrap = ~local_pending
        introspect = local_pending
        live_lengths = lengths.index_select(0, live_rows)
        live_ids = ids.index_select(0, live_rows)
        live_pending_ids = pending_ids.index_select(0, live_rows)
        work_width = capacity + 2 * stride - 1
        work_positions = torch.arange(work_width, device=ids.device)[None]
        committed = torch.full(
            (live_rows.numel(), work_width),
            pad_id,
            dtype=torch.long,
            device=ids.device,
        )
        committed[:, :capacity] = live_ids
        relative = work_positions - live_lengths[:, None]
        pending_source = relative.clamp(0, max(proposal_count - 1, 0))
        if proposal_count:
            pending_values = live_pending_ids.gather(1, pending_source)
        else:
            pending_values = committed
        introspect_clean = (
            introspect[:, None]
            & relative.ge(0)
            & relative.lt(proposal_count)
        )
        mask_region = (
            bootstrap[:, None]
            & relative.ge(0)
            & relative.lt(proposal_count)
        ) | (
            introspect[:, None]
            & relative.ge(proposal_count)
            & relative.lt(2 * stride - 1)
        )
        work_ids = torch.where(
            introspect_clean,
            pending_values,
            torch.where(mask_region, mask_id, committed),
        )
        append_lengths = torch.where(
            introspect,
            torch.full_like(live_lengths, 2 * stride - 1),
            torch.where(
                bootstrap,
                torch.full_like(live_lengths, proposal_count),
                torch.zeros_like(live_lengths),
            ),
        )
        work_valid = work_positions < (live_lengths + append_lengths)[:, None]
        logits = model.forward_sequence(work_ids, work_valid)
        model_forwards += 1
        model_query_tokens += work_valid.sum()

        if bool(bootstrap.any()):
            local_rows = bootstrap.nonzero(as_tuple=False).flatten()
            rows = live_rows.index_select(0, local_rows)
            row_logits = logits.index_select(0, local_rows)
            row_lengths = lengths.index_select(0, rows)
            anchor_p = _probabilities(
                _gather_logits(row_logits, row_lengths - 1), temperature
            )
            anchors = categorical_from_uniform(
                anchor_p,
                torch.rand(rows.numel(), device=ids.device, generator=generator),
            )
            if proposal_count:
                q_positions = row_lengths[:, None] + torch.arange(
                    proposal_count, device=ids.device
                )[None]
                q = _probabilities(_gather_logits(row_logits, q_positions), temperature)
                proposals = categorical_from_uniform(
                    q,
                    torch.rand(
                        (rows.numel(), proposal_count),
                        device=ids.device,
                        generator=generator,
                    ),
                )
                pending_ids[rows] = proposals
                pending_q[rows] = q
            append_committed(
                anchors[:, None],
                torch.ones((rows.numel(), 1), dtype=torch.bool, device=ids.device),
                rows,
            )
            can_continue = ~finished.index_select(0, rows) & (
                lengths.index_select(0, rows) < limits.index_select(0, rows)
            )
            has_pending[rows] = can_continue & (proposal_count > 0)
            physical_slots += rows.numel() * proposal_count

        if bool(introspect.any()):
            local_rows = introspect.nonzero(as_tuple=False).flatten()
            rows = live_rows.index_select(0, local_rows)
            row_logits = logits.index_select(0, local_rows)
            row_lengths = lengths.index_select(0, rows)
            p_positions = row_lengths[:, None] - 1 + torch.arange(
                proposal_count, device=ids.device
            )[None]
            p = _probabilities(_gather_logits(row_logits, p_positions), temperature)
            bonus_p = _probabilities(
                _gather_logits(row_logits, row_lengths + proposal_count - 1),
                temperature,
            )
            next_q_positions = row_lengths[:, None] + proposal_count + torch.arange(
                proposal_count, device=ids.device
            )[None]
            next_q = _probabilities(
                _gather_logits(row_logits, next_q_positions), temperature
            )
            next_proposals = categorical_from_uniform(
                next_q,
                torch.rand(
                    (rows.numel(), proposal_count),
                    device=ids.device,
                    generator=generator,
                ),
            )
            correction = isd_commit_prefix(
                torch.full((rows.numel(),), pad_id, dtype=torch.long, device=ids.device),
                p,
                pending_q.index_select(0, rows),
                pending_ids.index_select(0, rows),
                accept_uniforms=torch.rand(
                    (rows.numel(), proposal_count), device=ids.device, generator=generator
                ),
                residual_uniforms=torch.rand(
                    (rows.numel(), proposal_count), device=ids.device, generator=generator
                ),
                bonus_p=bonus_p,
                bonus_uniforms=torch.rand(rows.numel(), device=ids.device, generator=generator),
            )
            semantic = torch.arange(proposal_count, device=ids.device)[None] < (
                limits.index_select(0, rows) - row_lengths
            )[:, None]
            inspected = semantic & correction.valid[:, 1 : 1 + proposal_count]
            corrected_proposals = correction.ids[:, 1 : 1 + proposal_count]
            proposal_columns = torch.arange(
                proposal_count, device=ids.device
            )[None]
            first_proposal_eot = torch.where(
                inspected & corrected_proposals.eq(eot_id),
                proposal_columns,
                proposal_count,
            ).amin(1)
            inspected &= proposal_columns <= first_proposal_eot[:, None]
            eligible_proposals += inspected.sum()
            accepted_proposals += (
                correction.proposal_accepted & inspected
            ).sum()
            append_committed(correction.ids[:, 1:], correction.valid[:, 1:], rows)
            continue_pending = (
                correction.all_proposals_accepted
                & ~finished.index_select(0, rows)
                & (lengths.index_select(0, rows) < limits.index_select(0, rows))
            )
            pending_ids[rows] = next_proposals
            pending_q[rows] = next_q
            has_pending[rows] = continue_pending
            physical_slots += rows.numel() * stride
            fused_forwards += 1

        iterations += 1

    valid = storage < lengths[:, None]
    return IDLMReplayResult(
        ids=ids,
        valid=valid,
        prompt_lengths=prompt_lengths,
        lengths=lengths,
        finished=finished,
        iterations=iterations,
        model_forwards=model_forwards,
        physical_proposal_slots=int(physical_slots),
        eligible_proposals=int(eligible_proposals),
        accepted_proposals=int(accepted_proposals),
        model_query_tokens=int(model_query_tokens),
        fused_2n_minus_1_forwards=fused_forwards,
    )


__all__ = [
    "CausalIDLM",
    "IDLMScratchHandle",
    "IDLMServingCache",
    "IDLMReplayResult",
    "generate_idlm_fused_replay",
    "generate_idlm_full_replay",
]
