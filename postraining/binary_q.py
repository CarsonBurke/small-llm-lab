"""Binary terminal-Q credit, with no state-value or policy-average baseline.

Only the conditional expectation of the *whole* binary DG coefficient is
used: E[R sigmoid(R s / eta) | prefix, action] = Q sigmoid(s / eta).
The format oracle inspects generated tokens through the chosen action, never
the subsequently sampled continuation. Terminal EOS may use its verifier
label because that label is then a deterministic function of the prefix.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from postraining.latent_rollout import LatentRolloutBatch


def binary_q_delightful_policy_loss(
    logprobs: Tensor,
    success_probabilities: Tensor,
    mask: Tensor,
    temperature: float = 1.0,
    denominator: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Score surrogate for conditional binary-DG, not sigmoid(Q*s)*Q.

    Both Q and the gate are frozen for the actor gradient. The probabilities
    are predictions, not an advantage or a normalization statistic. Masked
    entries may contain NaNs; they must contribute neither loss nor gradient.
    """
    if logprobs.shape != success_probabilities.shape or mask.shape != logprobs.shape:
        raise ValueError("binary Q, log probabilities, and mask must align")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("delight temperature must be finite and positive")
    active = mask.bool()
    logp = torch.where(active, logprobs.float(), 0.0)
    q = torch.where(active, success_probabilities.detach().float(), 0.0)
    # Return validity to the trainer's host-side pre-step guards. Device-side
    # assertions poison the CUDA context and obscure the offending values.
    valid_q = torch.all((q >= 0) & (q <= 1))
    valid_logprobs = torch.isfinite(logp.detach()).all() & (logp.detach() <= 0).all()
    surprisal = -logp.detach()
    gate = torch.sigmoid(surprisal / temperature)
    terms = -q * gate * logp * mask.float()
    denom = (mask.sum() if denominator is None else denominator.to(mask.device)).clamp_min(1)
    positive = active & (q > 0)
    zero = terms.detach().new_zeros(())
    return terms.sum() / denom, {
        "gate_sum": gate.masked_fill(~active, 0).sum(),
        "positive_gate_sum": gate.masked_fill(~positive, 0).sum(),
        "positive_count": positive.sum(),
        "negative_gate_sum": zero,
        "negative_count": positive.new_zeros((), dtype=torch.long),
        # For this estimator delight is the success-conditioned surprisal;
        # Q is outside the gate. Record its own units, not Q*surprisal.
        "delight_sum": surprisal.masked_fill(~active, 0).sum(),
        "surprisal_sum": surprisal.masked_fill(~active, 0).sum(),
        "positive_loss_sum": terms.detach().masked_fill(~positive, 0).sum(),
        "negative_loss_sum": zero,
        "coefficient_sum": (q * gate * mask).sum(),
        "valid_q": valid_q,
        "valid_logprobs": valid_logprobs,
        "positive_logprob_count": (active & (logp.detach() > 0)).sum(),
        "nonfinite_logprob_count": (active & ~torch.isfinite(logp.detach())).sum(),
        "positive_logprob_max": logp.detach().clamp_min(0).amax(),
    }


def exact_prefix_success_constraints(
    batch: LatentRolloutBatch,
    *,
    think_fence_ids: tuple[int, int],
    answer_fence_ids: tuple[int, int],
    eos_id: int,
    min_think_tokens: int,
    max_new_tokens: int | None = None,
    stop_ids: tuple[int, ...] | None = None,
) -> tuple[Tensor, Tensor]:
    """Return (known, exact Q) at action slots for the strict binary contract.

    The contract requires one anchored think pair, then one nonempty answer
    pair immediately followed by EOS. Ordinary tokens between think-close
    and answer-open are allowed, matching structural_format_ok. Only emitted
    response tokens count: prompt text, padding and recorded actor hiddens do
    not. Once a prefix violates a necessary condition it cannot be repaired
    by appending tokens under the unique-fence contract.

    This is a vectorized prefix scan. No full-vocabulary Q calculation, host
    token decoding, critic pass, or data-dependent device synchronization is
    required. The terminal label must be the already verified binary reward.
    """
    terminals = (eos_id,) if stop_ids is None else tuple(dict.fromkeys(stop_ids))
    fences = (*think_fence_ids, *answer_fence_ids)
    if not terminals or len(set(fences)) != 4 or set(fences).intersection(terminals):
        raise ValueError("structural fences and terminal IDs must be distinct")
    if min_think_tokens < 1 or (max_new_tokens is not None and max_new_tokens < 1):
        raise ValueError("think floor and optional response budget must be positive")
    emit = batch.emit_mask.bool()
    tokens = torch.cat((batch.token_ids[:, 1:], torch.zeros_like(batch.token_ids[:, :1])), 1)
    ordinal = emit.long().cumsum(1)
    masks = [emit & (tokens == token) for token in (*think_fence_ids, *answer_fence_ids)]
    counts = [flag.long().cumsum(1) for flag in masks]
    # Ordinals, not padded stream positions, measure emitted-token distances.
    locations = [(ordinal * flag).cummax(1).values for flag in masks]
    to, tc, ao, ac = masks
    nto, ntc, nao, nac = counts
    pto, ptc, pao, pac = locations
    eos = torch.zeros_like(emit)
    for terminal in terminals:
        eos = eos | (emit & (tokens == terminal))
    violation = emit & (
        ((ordinal == 1) & ~to)
        | (nto > 1) | (ntc > 1) | (nao > 1) | (nac > 1)
        | (tc & ((nto != 1) | (ordinal - pto - 1 < min_think_tokens)))
        | (ao & ((ntc != 1) | (ptc >= ordinal)))
        | (ac & ((nao != 1) | (ordinal - pao < 2)))
        | ((nac > 0) & (ordinal > pac) & ~eos)
        | (eos & ((nac != 1) | (ordinal - pac != 1)))
    )
    if max_new_tokens is not None:
        violation = violation | (emit & (ordinal >= max_new_tokens) & ~eos)
    impossible = violation.long().cumsum(1) > 0
    known = emit & (impossible | eos)
    rewards = batch.reward_scalar.detach().float()
    torch._assert_async(torch.all((rewards == 0) | (rewards == 1)), "terminal rewards must be binary")
    # Reject a violated-prefix success label rather than silently treating a
    # different verifier contract as ours. No future label enters live Q.
    torch._assert_async(torch.all(~(eos & impossible & (rewards[:, None] == 1))), "success label contradicts strict prefix contract")
    exact = torch.where(eos & ~impossible, rewards[:, None], 0.0)
    return known, exact
