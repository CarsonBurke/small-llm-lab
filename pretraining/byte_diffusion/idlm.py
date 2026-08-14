"""Reference-faithful primitives for Introspective Diffusion LMs.

The paper's training input is physically ordered as ``[proposal | clean]``.
Every valid proposal id is ``MASK``; proposal and clean hidden positions both
predict the next clean id.  The proposal copy uses causal attention within its
current block and may read clean tokens only from strictly earlier blocks.
Clean queries use the ordinary token-causal mask and never read proposals.

These primitives deliberately contain no mode embedding or model-specific
forward logic.  ``MASK`` is the only proposal-path input cue, which lets an
integration preserve exact clean/AR anchor parity.

Reference: ``papers/introspective_diffusion_language_models_2604.11035v1.pdf``,
Section 3.1 and Eqs. 1--2 (p. 5), Algorithm 1 (p. 7), and Appendix E (p. 18).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


IGNORE_INDEX = -100


@dataclass(frozen=True)
class IDLMTrainingLayout:
    """Typed ``[all-MASK proposal | clean]`` training geometry."""

    proposal_ids: Tensor
    clean_ids: Tensor
    valid: Tensor
    positions: Tensor
    block_size: int
    segment_ids: Tensor | None = None

    def __post_init__(self) -> None:
        if self.clean_ids.dtype != torch.long or self.clean_ids.ndim != 2:
            raise ValueError("clean_ids must be rank-2 int64")
        if (
            self.proposal_ids.dtype != torch.long
            or self.proposal_ids.shape != self.clean_ids.shape
        ):
            raise ValueError("proposal_ids must be int64 and aligned with clean_ids")
        if self.valid.dtype != torch.bool or self.valid.shape != self.clean_ids.shape:
            raise ValueError("valid must be boolean and aligned with clean_ids")
        if self.positions.dtype != torch.long or self.positions.shape != self.clean_ids.shape:
            raise ValueError("positions must be aligned document-relative int64 ids")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.segment_ids is not None and (
            self.segment_ids.dtype != torch.long
            or self.segment_ids.shape != self.clean_ids.shape
        ):
            raise ValueError("segment_ids must be aligned int64 ids")
        self.validate_positions()

    @property
    def batch_size(self) -> int:
        return self.clean_ids.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.clean_ids.shape[1]

    @property
    def input_ids(self) -> Tensor:
        return torch.cat((self.proposal_ids, self.clean_ids), dim=1)

    def validate_positions(self) -> None:
        """Require zero-based, consecutive positions within each document."""

        # Layouts are semantically validated while still on CPU.  Repeating
        # Python ``bool`` reductions after the batch transfer would add three
        # host synchronizations to every GPU microstep.
        if self.positions.device.type != "cpu" or torch.compiler.is_compiling():
            return

        valid_positions = self.positions[self.valid]
        if bool((valid_positions < 0).any()):
            raise ValueError("valid document-relative positions must be nonnegative")
        if self.sequence_length == 0:
            return

        same_document_as_previous = self.valid[:, 1:] & self.valid[:, :-1]
        if self.segment_ids is not None:
            same_document_as_previous &= self.segment_ids[:, 1:].eq(
                self.segment_ids[:, :-1]
            )
        consecutive = self.positions[:, 1:].eq(self.positions[:, :-1] + 1)
        if bool((same_document_as_previous & ~consecutive).any()):
            raise ValueError("positions must increase by one within each document")

        document_start = self.valid.clone()
        document_start[:, 1:] &= ~same_document_as_previous
        if bool(self.positions[document_start].ne(0).any()):
            raise ValueError("every document-relative position sequence must start at zero")


@dataclass(frozen=True)
class IDLMShiftedTargets:
    """Next-id targets aligned to proposal and clean hidden positions."""

    proposal: Tensor
    clean: Tensor
    ignore_index: int = IGNORE_INDEX

    def __post_init__(self) -> None:
        if self.proposal.dtype != torch.long or self.proposal.ndim != 2:
            raise ValueError("proposal targets must be rank-2 int64")
        if self.clean.dtype != torch.long or self.clean.shape != self.proposal.shape:
            raise ValueError("clean targets must align with proposal targets")

    @property
    def combined(self) -> Tensor:
        return torch.cat((self.proposal, self.clean), dim=1)


@dataclass(frozen=True)
class IDLMLoss:
    """Paper Eq. 2 with a detached, compile-safe clean-loss scale."""

    total: Tensor
    proposal: Tensor
    clean: Tensor
    clean_scale: Tensor
    proposal_targets: Tensor
    clean_targets: Tensor


@dataclass(frozen=True)
class ISDCorrection:
    """Independent p/q corrections before prefix truncation."""

    ids: Tensor
    accepted: Tensor
    acceptance_probability: Tensor


@dataclass(frozen=True)
class ISDCommitBatch:
    """Maximum-width committed tokens from one ISD introspection step.

    ``ids`` and ``valid`` have width ``2 + proposal_count``: the guaranteed
    anchor, proposals through the first rejection (including its residual
    resample), and a bonus slot used only when every proposal is accepted.
    """

    ids: Tensor
    valid: Tensor
    proposal_accepted: Tensor
    acceptance_probability: Tensor
    all_proposals_accepted: Tensor


def make_training_layout(
    clean_ids: Tensor,
    valid: Tensor,
    *,
    mask_id: int,
    pad_id: int,
    block_size: int,
    segment_ids: Tensor | None = None,
    positions: Tensor | None = None,
) -> IDLMTrainingLayout:
    """Construct the paper's all-MASK proposal copy without target leakage."""

    if positions is None:
        columns = torch.arange(clean_ids.shape[1], device=clean_ids.device)[
            None
        ].expand_as(clean_ids)
        if segment_ids is None:
            positions = columns
        else:
            same_document_as_previous = valid[:, 1:] & valid[:, :-1]
            same_document_as_previous &= segment_ids[:, 1:].eq(
                segment_ids[:, :-1]
            )
            document_start = valid.clone()
            document_start[:, 1:] &= ~same_document_as_previous
            start_columns = torch.where(document_start, columns, 0).cummax(dim=1).values
            positions = torch.where(valid, columns - start_columns, 0)

    proposal_ids = torch.where(
        valid,
        torch.as_tensor(mask_id, dtype=clean_ids.dtype, device=clean_ids.device),
        torch.as_tensor(pad_id, dtype=clean_ids.dtype, device=clean_ids.device),
    )
    return IDLMTrainingLayout(
        proposal_ids=proposal_ids,
        clean_ids=clean_ids,
        valid=valid,
        positions=positions,
        block_size=block_size,
        segment_ids=segment_ids,
    )


def strict_attention_mask(layout: IDLMTrainingLayout) -> Tensor:
    """Return the exact dense attention oracle for ``[proposal | clean]``.

    Output shape is ``[batch, 2 * length, 2 * length]``.  This is intended as
    a correctness oracle and mask specification; production integrations may
    lower the same topology to Flash/Flex block metadata.
    """

    length = layout.sequence_length
    device = layout.clean_ids.device
    query_positions = layout.positions[:, :, None]
    key_positions = layout.positions[:, None, :]
    query_block = torch.div(
        query_positions, layout.block_size, rounding_mode="floor"
    )
    key_block = torch.div(
        key_positions, layout.block_size, rounding_mode="floor"
    )
    causal = key_positions <= query_positions

    query_valid = layout.valid[:, :, None]
    key_valid = layout.valid[:, None, :]
    same_segment = torch.ones(
        (layout.batch_size, length, length), dtype=torch.bool, device=device
    )
    if layout.segment_ids is not None:
        same_segment = layout.segment_ids[:, :, None].eq(
            layout.segment_ids[:, None, :]
        )

    proposal_to_proposal = (
        query_valid
        & key_valid
        & same_segment
        & (query_block == key_block)
        & causal
    )
    proposal_to_clean = (
        query_valid
        & key_valid
        & same_segment
        & (key_block < query_block)
    )
    clean_to_clean = query_valid & key_valid & same_segment & causal
    no_clean_to_proposal = torch.zeros_like(clean_to_clean)

    proposal_rows = torch.cat((proposal_to_proposal, proposal_to_clean), dim=-1)
    clean_rows = torch.cat((no_clean_to_proposal, clean_to_clean), dim=-1)
    return torch.cat((proposal_rows, clean_rows), dim=1)


def shifted_targets(
    layout: IDLMTrainingLayout,
    *,
    score_mask: Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> IDLMShiftedTargets:
    """Map every eligible hidden position to the next clean id.

    Requiring current and next positions to be valid and in the same segment
    prevents labels from crossing packed-document padding.  ``score_mask`` is
    expressed on target positions.
    """

    if score_mask is None:
        score_mask = layout.valid
    elif score_mask.dtype != torch.bool or score_mask.shape != layout.valid.shape:
        raise ValueError("score_mask must be boolean and aligned with the layout")

    targets = torch.full_like(layout.clean_ids, ignore_index)
    if layout.sequence_length > 1:
        active = layout.valid[:, :-1] & layout.valid[:, 1:] & score_mask[:, 1:]
        if layout.segment_ids is not None:
            active &= layout.segment_ids[:, :-1].eq(layout.segment_ids[:, 1:])
        targets[:, :-1] = torch.where(
            active,
            layout.clean_ids[:, 1:],
            torch.as_tensor(ignore_index, device=layout.clean_ids.device),
        )
    return IDLMShiftedTargets(
        proposal=targets,
        clean=targets.clone(),
        ignore_index=ignore_index,
    )


def auto_balanced_loss(
    proposal_loss: Tensor,
    clean_loss: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply I-DLM Eq. 2 without host scalar conversion or device sync."""

    if proposal_loss.numel() != 1 or clean_loss.numel() != 1:
        raise ValueError("I-DLM pathway losses must be scalar tensors")
    denominator = clean_loss.detach().clamp_min(
        torch.finfo(clean_loss.dtype).tiny
    )
    clean_scale = (proposal_loss.detach() / denominator).detach()
    return proposal_loss + clean_scale * clean_loss, clean_scale


def _mean_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    *,
    ignore_index: int,
) -> tuple[Tensor, Tensor]:
    if logits.ndim != 3 or logits.shape[:-1] != targets.shape:
        raise ValueError("logits and shifted targets must align")
    if targets.dtype != torch.long:
        raise ValueError("shifted targets must be int64")
    active = targets.ne(ignore_index)
    safe_targets = torch.where(active, targets, 0)
    ce_logits = logits.float() if logits.dtype in {torch.float16, torch.bfloat16} else logits
    nll = F.cross_entropy(
        ce_logits.reshape(-1, logits.shape[-1]),
        safe_targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    total = torch.where(active, nll, 0.0).sum()
    count = active.sum()
    return total / count.clamp_min(1).to(total.dtype), count


def idlm_loss(
    proposal_logits: Tensor,
    clean_logits: Tensor,
    targets: IDLMShiftedTargets,
) -> IDLMLoss:
    """Compute dense proposal/clean CE means and detached auto-balancing."""

    proposal_loss, proposal_count = _mean_cross_entropy(
        proposal_logits,
        targets.proposal,
        ignore_index=targets.ignore_index,
    )
    clean_loss, clean_count = _mean_cross_entropy(
        clean_logits,
        targets.clean,
        ignore_index=targets.ignore_index,
    )
    total, clean_scale = auto_balanced_loss(proposal_loss, clean_loss)
    return IDLMLoss(
        total=total,
        proposal=proposal_loss,
        clean=clean_loss,
        clean_scale=clean_scale,
        proposal_targets=proposal_count,
        clean_targets=clean_count,
    )


def _normalize_probabilities(probabilities: Tensor) -> Tensor:
    if not probabilities.is_floating_point():
        raise ValueError("categorical probabilities must be floating point")
    torch._assert_async(
        torch.isfinite(probabilities).all(),
        "categorical probabilities must be finite",
    )
    torch._assert_async(
        probabilities.ge(0).all(),
        "categorical probabilities must be nonnegative",
    )
    denominator = probabilities.sum(dim=-1, keepdim=True)
    torch._assert_async(
        denominator.gt(0).all(),
        "every categorical row must have positive mass",
    )
    return probabilities / denominator


def positive_residual_distribution(p: Tensor, q: Tensor) -> Tensor:
    """Batched ``normalize(max(0, p-q))`` used after ISD rejection."""

    if p.shape != q.shape or p.ndim < 1:
        raise ValueError("p and q must be aligned categorical tensors")
    p = _normalize_probabilities(p)
    q = _normalize_probabilities(q)
    residual = (p - q).clamp_min(0)
    residual_total = residual.sum(dim=-1, keepdim=True)
    normalized = residual / residual_total.clamp_min(
        torch.finfo(residual.dtype).tiny
    )
    # A rejection has zero probability when p == q.  Falling back to p keeps
    # the primitive total under roundoff or externally forced rejection.
    return torch.where(residual_total > 0, normalized, p)


def categorical_from_uniform(probabilities: Tensor, uniforms: Tensor) -> Tensor:
    """Vectorized inverse-CDF sampling with one uniform per categorical row."""

    if probabilities.shape[:-1] != uniforms.shape:
        raise ValueError("uniforms must align with categorical batch dimensions")
    probabilities = _normalize_probabilities(probabilities)
    upper = 1.0 - torch.finfo(probabilities.dtype).eps
    uniforms = uniforms.to(probabilities.dtype).clamp(0.0, upper)
    cdf = probabilities.cumsum(dim=-1)
    return (uniforms[..., None] >= cdf).sum(dim=-1).clamp_max(
        probabilities.shape[-1] - 1
    )


def isd_acceptance_probability(p: Tensor, q: Tensor, proposal_ids: Tensor) -> Tensor:
    """Return ``min(1, p(xhat)/q(xhat))`` for batched proposals."""

    if p.shape != q.shape or p.shape[:-1] != proposal_ids.shape:
        raise ValueError("p, q, and proposal ids must align")
    if proposal_ids.dtype != torch.long:
        raise ValueError("proposal ids must be int64")
    p = _normalize_probabilities(p)
    q = _normalize_probabilities(q)
    indices = proposal_ids[..., None]
    p_value = p.gather(-1, indices).squeeze(-1)
    q_value = q.gather(-1, indices).squeeze(-1)
    ratio = p_value / q_value.clamp_min(torch.finfo(q.dtype).tiny)
    return torch.where(q_value > 0, ratio.clamp_max(1), torch.ones_like(ratio))


def isd_accept_or_resample(
    p: Tensor,
    q: Tensor,
    proposal_ids: Tensor,
    *,
    accept_uniforms: Tensor,
    residual_uniforms: Tensor,
) -> ISDCorrection:
    """Apply exact batched speculative correction to independent proposals."""

    if (
        accept_uniforms.shape != proposal_ids.shape
        or residual_uniforms.shape != proposal_ids.shape
    ):
        raise ValueError("one acceptance and residual uniform is required per proposal")
    acceptance = isd_acceptance_probability(p, q, proposal_ids)
    accepted = accept_uniforms.to(acceptance.dtype) < acceptance
    residual = positive_residual_distribution(p, q)
    resampled = categorical_from_uniform(residual, residual_uniforms)
    ids = torch.where(accepted, proposal_ids, resampled)
    return ISDCorrection(ids=ids, accepted=accepted, acceptance_probability=acceptance)


def isd_commit_prefix(
    anchor_ids: Tensor,
    p: Tensor,
    q: Tensor,
    proposal_ids: Tensor,
    *,
    accept_uniforms: Tensor,
    residual_uniforms: Tensor,
    bonus_p: Tensor,
    bonus_uniforms: Tensor,
) -> ISDCommitBatch:
    """Correct one proposal stride, truncate at rejection, and add the bonus.

    The guaranteed anchor is always committed.  A rejected proposal is
    replaced from the positive residual and committed; proposals after the
    first rejection are invalid.  The bonus is committed iff every proposal
    was accepted.
    """

    if anchor_ids.dtype != torch.long or anchor_ids.ndim != 1:
        raise ValueError("anchor_ids must be rank-1 int64")
    if proposal_ids.ndim != 2 or proposal_ids.shape[0] != anchor_ids.shape[0]:
        raise ValueError("proposal ids must have shape [batch, proposals]")
    if bonus_p.shape[:-1] != anchor_ids.shape or bonus_uniforms.shape != anchor_ids.shape:
        raise ValueError("bonus distribution and uniforms must align with anchors")

    correction = isd_accept_or_resample(
        p,
        q,
        proposal_ids,
        accept_uniforms=accept_uniforms,
        residual_uniforms=residual_uniforms,
    )
    batch, proposals = proposal_ids.shape
    if proposals:
        accepted_int = correction.accepted.to(torch.int64)
        accepted_before = torch.cat(
            (
                torch.ones(
                    (batch, 1), dtype=torch.int64, device=proposal_ids.device
                ),
                accepted_int[:, :-1].cumprod(dim=1),
            ),
            dim=1,
        ).to(torch.bool)
    else:
        accepted_before = torch.empty(
            (batch, 0), dtype=torch.bool, device=proposal_ids.device
        )
    all_accepted = correction.accepted.all(dim=1)
    bonus_ids = categorical_from_uniform(bonus_p, bonus_uniforms)

    ids = torch.cat(
        (anchor_ids[:, None], correction.ids, bonus_ids[:, None]), dim=1
    )
    valid = torch.cat(
        (
            torch.ones((batch, 1), dtype=torch.bool, device=proposal_ids.device),
            accepted_before,
            all_accepted[:, None],
        ),
        dim=1,
    )
    if ids.shape != (batch, proposals + 2):
        raise AssertionError("ISD commit geometry is inconsistent")
    return ISDCommitBatch(
        ids=ids,
        valid=valid,
        proposal_accepted=correction.accepted,
        acceptance_probability=correction.acceptance_probability,
        all_proposals_accepted=all_accepted,
    )


__all__ = [
    "IGNORE_INDEX",
    "IDLMTrainingLayout",
    "IDLMShiftedTargets",
    "IDLMLoss",
    "ISDCorrection",
    "ISDCommitBatch",
    "make_training_layout",
    "strict_attention_mask",
    "shifted_targets",
    "auto_balanced_loss",
    "idlm_loss",
    "positive_residual_distribution",
    "categorical_from_uniform",
    "isd_acceptance_probability",
    "isd_accept_or_resample",
    "isd_commit_prefix",
]
