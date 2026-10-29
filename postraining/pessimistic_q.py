"""A pessimistic prior for a bilinear token Q critic at tokens it has no data on.

All-action MAC reads Q(s, a) at every token, while the critic regresses Q
only at the tokens the policy took. A bilinear Q extrapolates to the rest,
and at confident states it scored unsampled tokens above the taken one;
the actor followed (NOTES 2026-10-19). CQL(H) did not fix this here: its
temperature-one log-normalizer over 50,304 tokens only concentrates once
score gaps reach about log V = 10.8, which a value of a [0, 1] reward never
supports, so its push on unsampled tokens never stopped. The critic met it
by inflating its score scale until off-support advantages read -7 and the
value fit failed (job 14195, NOTES 2026-10-20).

This prior instead regresses an unsupported token's advantage toward the
policy's value, one-sided, the discrete form of mildly conservative
Q-learning's pseudo-target (Lyu et al. 2022). Per head h and state s,

    beta * sum_a relu(A_h(s, a))^2,   A_h(s, a) = u_h(s).(x(a) - sum_b pi(b|s) x(b)),

under the current policy pi. The state term c(s) cancels. Only the
direction u learns from it: the action table x is the critic trunk's
readout (``SeparateCritic.action_rows``), whose rows for unseen tokens
would otherwise take a dense, Adam-normalized step of their own every
update; their extrapolation is
u's to correct, state by state. Inflating the score scale multiplies the
prior by k^2, so the escape CQL(H) found is closed.

With the regression weighted by how often each token is taken, its tabular
fixed point (every Q(s, a) free) keeps the policy's value V = sum pi Q at
the fit's and, with R = sum_a relu(A(a)),

    A(a) = pi(a)/(pi(a) + beta) * (A_fit(a) + beta R)   where A(a) > 0,
    Q(a) = Q_fit(a) + beta R                             elsewhere:

a positive advantage keeps pi/(pi + beta) of itself, so tokens the policy
samples well past beta keep their fit, tokens it all but never samples
cannot read better than the policy, and once the actor makes a token likely
enough to be sampled, its data takes over. The beta R offset is small once
off-support advantages are shrunk; ``q_prior/penalty_per_state`` and
``q_prior/positive_tokens_per_state`` track how far from that the critic is.

The gradients are formed explicitly, a chunk of states at a time, from the
actor's own readout; no states*vocab tensor outlives a call.
"""

from __future__ import annotations

import torch
from torch import Tensor

from postraining.nano_backbone import _shifted_softcapped_logits


def pessimistic_q_prior(raw_logits: Tensor, cap: float, member_directions: Tensor,
                        action_latents: Tensor, row_weights: Tensor
                        ) -> tuple[Tensor, Tensor, Tensor]:
    """The prior's direction gradients for one chunk of states, and its statistics.

    ``raw_logits`` [states, vocab] is the actor's readout before its softcap
    ``cap``; ``member_directions`` [heads, states, dim] are each head's
    detached directions u and ``action_latents`` [vocab, dim] the detached
    action table x; ``row_weights`` [states] carry beta and the caller's
    normalization. Returns d/du [heads, states, dim] of
    sum_h sum_s w_s sum_a relu(A_h)^2, then per head the unweighted sum of
    relu(A_h)^2 and the count of tokens with A_h > 0 (fp64).
    """
    if raw_logits.ndim != 2 or member_directions.ndim != 3 or action_latents.ndim != 2:
        raise ValueError("the Q prior needs logits [states, vocab], directions [heads, states, dim] "
                         "and latents [vocab, dim]")
    states, vocab = raw_logits.shape
    if (member_directions.shape[1] != states or action_latents.shape[0] != vocab
            or member_directions.shape[2] != action_latents.shape[1]):
        raise ValueError("Q prior shapes disagree")
    if row_weights.shape != (states,):
        raise ValueError("Q prior row weights must be [states]")
    with torch.no_grad(), torch.autocast(device_type=raw_logits.device.type, enabled=False):
        _, _, shifted, log_norm = _shifted_softcapped_logits(raw_logits, cap)
        probabilities = (shifted - log_norm[:, None]).exp()
        latents = action_latents.to(probabilities.dtype)
        # A(a) = u.(x(a) - xbar), xbar the policy's mean action latent.
        mean_latents = probabilities @ latents
        weights = 2 * row_weights.to(probabilities.dtype)[:, None]
        grad_directions, penalties, positives = [], [], []
        for directions in member_directions.to(probabilities.dtype):
            excess = (directions @ latents.T
                      - (directions * mean_latents).sum(-1, keepdim=True)).clamp_min(0)
            penalties.append(excess.square().double().sum())
            positives.append((excess > 0).double().sum())
            # d/du of sum_a relu(A(a))^2 = sum_a 2 relu(A(a)) (x(a) - xbar).
            excess = weights * excess
            grad_directions.append(excess @ latents - excess.sum(-1, keepdim=True) * mean_latents)
        return (torch.stack(grad_directions).to(member_directions.dtype),
                torch.stack(penalties), torch.stack(positives))


def nano_pessimistic_q_prior(backbone, beliefs: Tensor, member_directions: Tensor,
                             action_latents: Tensor, row_weights: Tensor):
    """``pessimistic_q_prior`` at a nano renderer's own readout of compact
    emit beliefs, the policy all-action MAC reads (``nano_mean_actor_critic``)."""
    with torch.no_grad():
        raw = backbone._raw_logits(beliefs)
    return pessimistic_q_prior(raw, backbone.logit_softcap, member_directions,
                               action_latents, row_weights)
