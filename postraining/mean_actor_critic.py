"""Token Q actor estimators for the nano rational-softcap renderer.

For Q(s,a)=c(s)+u(s).x(a), c cancels from the policy advantage
A(a)=Q(a)-E_pi Q. The estimators of the policy gradient at a state share
one fused readout; ``MAC_ESTIMATORS`` names them:

* ``all_action`` is Mean Actor-Critic: -sum_a pi(a|s) stopgrad[A(a)]. Every
  vocabulary logit moves by pi(a|s) times its centered Q, so the gradient is
  only as good as the critic's Q at actions it has not seen; it follows the
  critic's errors (NOTES 2026-10-19).
* ``control_variate`` adds the sampled token's Monte Carlo residual,
  -stopgrad[G - Q(a_t)] log pi(a_t), with Q(a_t) = c + A(a_t). By the score-function (Stein) identity
  its expectation over a_t ~ pi is the gradient of E_pi[Q_true - Q], so the
  sum is the on-policy gradient for any critic: Q's errors cost variance,
  not bias. It is the discrete form of fitting the critic's action slope to
  returns along the policy's own score functions, as the TD7 latent-Q
  critic's Hermite tangent does for a Gaussian policy.
* ``sampled_baseline`` is the control without the all-action term,
  -stopgrad[G - E_pi Q] log pi(a_t): REINFORCE with the critic's state value,
  which for the policy-centred critic (``value_model.policy_centered_q``)
  is c itself.
* ``delightful_sampled_baseline`` gates that sampled term as Delightful
  Policy Gradient does (Osband 2026, ``core.delightful_policy_loss``): its
  coefficient r = G - E_pi Q is weighted by the detached
  sigmoid(r * -log pi(a_t)), eta = 1. A surprising token in a better-than-
  expected continuation is reinforced at nearly full weight, and one in a
  worse continuation is suppressed at nearly none; a confident token's
  gate stays near 1/2 either way. It is biased by design: r copies one
  sequence outcome to every token, so for a single token it is mostly the
  other tokens' sampling noise, and the nonlinear gate turns that noise into
  a push toward surprising tokens whatever their effect. The paper's
  sequence task broadcasts its outcome too, but over 10 tokens that all
  matter; here ~250 tokens share it, most of them barely (NOTES 2026-09-22,
  2026-10-20).
* ``delightful_td_gate`` takes the gate's advantage from the critic
  rather than the outcome: token transitions are deterministic, so
  Q(s_t, a_t) = V(s_{t+1}) and the critic's advantage of the sampled token
  is its TD residual A_t = rew_t + c(s_{t+1}) - c(s_t)
  (``gate_advantages``). The coefficient is sigmoid(A_t * -log pi(a_t))
  (G - c): a gate that is a function of the state and token alone,
  independent of the continuation's sampling, on the sampled term. Its
  expectation given the token is sigmoid(A l) (Q - c), so the gate can
  rescale but not flip a token's expected push. It is not unbiased: where
  c differs from V, the error enters A and the multiplier's baseline, and a
  gate that varies across tokens stops that baseline error from cancelling.
  It also rectifies critic noise as the outcome gate rectifies outcome
  noise, at the critic error's variance rather than the outcome's. Where
  the critic sees no effect (A ~ 0) it is the sampled baseline at half
  scale.
* ``delightful_head_gate`` gates the same term on the critic action head's
  own advantage of the sampled token, A(a_t) = u.(x(a_t) - sum_a pi(a) x(a)),
  the ``target_advantages`` this readout forms anyway: no TD residual and
  no next-state value enter. sigmoid(A(a_t) * -log pi(a_t)) (G - c). Like
  the TD gate it can rescale but not flip a token's expected push; its bias
  is the head's error at the taken token, not the value function's
  per-step wobble (NOTES 2026-10-21).

Two estimators read no critic at all; their sampled coefficient is one
constant per trajectory, formed from its prompt group (``SEQUENCE_ESTIMATORS``,
``sequence_coefficients``). The context is the prompt and the action is the
final answer:

* ``rloo_sequence`` is REINFORCE with the leave-one-out group baseline
  (RLOO): U_i = R_i - mean_{j != i} R_j on every token of trajectory i. The
  baseline is independent of trajectory i, so it is unbiased.
* ``delightful_answer_gate`` is Delightful Policy Gradient at the answer
  level (Osband 2026, eta = 1): sigmoid(U_i * l_i) U_i, with
  l_i = -log(n_k / G) the surprisal of trajectory i's answer cluster k under
  the group's empirical answer distribution (``answer_cluster_surprisal``).
  The trainer keys a cluster on the verifier's (canonical answer, reward)
  pair, so one canonical string graded two ways is two clusters
  (``train_latent_vapo.graded_answer_keys``). A rare correct answer is reinforced at nearly full weight and a rare wrong
  one is nearly unblamed; a unanimous group (l = 0) gates at 1/2. With RLOO
  as its control the gate is the only difference. The gate is a function of
  the whole group, not of a token, so a per-token critic baseline in the
  coefficient would no longer cancel in expectation; none is used.

``MAC_CONTROL_ESTIMATORS`` are gradient-measurement controls, not training
objectives (no schema, no CLI choice):

* ``delightful_shuffled_head_gate`` is the head gate with its advantages
  permuted across the update's states within strata of the sampled
  token's surprisal and span (``gate_shuffle_strata``,
  ``shuffle_within_strata``), passed as ``gate_advantages``. Each stratum
  keeps its multiset of A, so the gate's |A| * surprisal marginal is
  nearly kept while which token carries which A is destroyed: the placebo
  that separates DG's blame suppression from the information in A.

One custom function forms the loss together with the sampled token's
log-probability, the policy entropy, the sampled token's centered Q, a
census of the centered Q by the policy's support (off support, ranks 2-8,
top token: what the all-action term follows where the critic has no data),
the taken token's offset x(a_t) - sum_a pi(a) x(a), which centres the
critic's own fit of Q, and the policy variance of Q, sum_a pi(a) A(a)^2: how much the critic separates the
actions the policy samples. It is the first-order gain of a logit step along
A (the softmax natural gradient), not of this update's own logit step along
pi*A, which gains sum_a (pi(a) A(a))^2 before the softcap, shared parameters
and the optimizer reshape it. Those are the update's telemetry and behavior
statistics; producing them in the loss forward means the vocabulary table is
read once forward and once backward. The caller chooses the states, their
weights and the returns; this module knows no verifier, fences or update
schedule.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from postraining.nano_backbone import _shifted_softcapped_logits, policy_action_offsets_

MAC_ESTIMATORS = ("all_action", "control_variate", "sampled_baseline",
                  "delightful_sampled_baseline", "delightful_td_gate",
                  "delightful_head_gate", "rloo_sequence", "delightful_answer_gate")
# Measurement controls: valid in an update, never a training objective.
MAC_CONTROL_ESTIMATORS = ("delightful_shuffled_head_gate",)
# Estimators with no all-action term: only the sampled token's score moves.
SAMPLED_ONLY_ESTIMATORS = ("sampled_baseline", "delightful_sampled_baseline",
                           "delightful_td_gate", "delightful_head_gate",
                           "delightful_shuffled_head_gate", "rloo_sequence",
                           "delightful_answer_gate")
# Estimators whose sampled coefficient is one constant per trajectory, from
# its prompt group's rewards and answers (``sequence_coefficients``); the
# caller passes it as the readout's ``residuals``, which they use verbatim.
SEQUENCE_ESTIMATORS = ("rloo_sequence", "delightful_answer_gate")
# Estimators whose delight gate reads advantages the caller supplies
# (``gate_advantages``) rather than ones the readout forms.
GATE_ADVANTAGE_ESTIMATORS = ("delightful_td_gate", "delightful_shuffled_head_gate")
# Estimators that gate the sampled term by delight.
DELIGHTFUL_MAC_ESTIMATORS = ("delightful_sampled_baseline", "delightful_td_gate",
                             "delightful_head_gate", "delightful_shuffled_head_gate")
# Delight gates on a critic advantage rather than on the outcome G - c.
CRITIC_GATED_ESTIMATORS = ("delightful_td_gate", "delightful_head_gate",
                           "delightful_shuffled_head_gate")
# The shuffled head gate's surprisal strata: -log pi(a_t) binned at 2^-10
# (pi ~ 0.999) to 2^3 (pi ~ 3.4e-4), doubling, so the surprisal varies at
# most twofold inside a finite bin; each bin splits by think/answer span.
GATE_SHUFFLE_SURPRISAL_EDGES = tuple(2.0 ** k for k in range(-10, 4))
# Tokens the policy all but never samples at a state: the critic has no
# regression data there, and the all-action term still reads their Q.
OFF_SUPPORT_PROBABILITY = 1e-4
# The policy's top tokens a support census reports: the top one and ranks
# 2 through SUPPORT_RANKS.
SUPPORT_RANKS = 8
SUPPORT_CENSUS_FIELDS = (
    "offsupport_count", "offsupport_positive", "offsupport_advantage",
    "rank2to8_count", "rank2to8_positive", "rank2to8_advantage",
    "top_count", "top_advantage",
)


def support_census(probabilities: Tensor, centered: Tensor) -> Tensor:
    """fp64 sums over states of the critic's policy-centered Q by support:
    (count, A > 0, A) at tokens with pi below ``OFF_SUPPORT_PROBABILITY``,
    the same at the policy's ranks 2..``SUPPORT_RANKS`` whatever their
    mass, and (count, A) at its top token; ``SUPPORT_CENSUS_FIELDS`` names
    them.
    """
    off = probabilities < OFF_SUPPORT_PROBABILITY
    ranks = probabilities.topk(min(SUPPORT_RANKS, probabilities.size(-1)), dim=-1).indices
    near = centered.gather(-1, ranks[:, 1:])
    top = centered.gather(-1, ranks[:, :1])
    count = probabilities.new_tensor
    return torch.stack((
        off.sum().double(), (off & (centered > 0)).sum().double(),
        (centered * off).double().sum(),
        count(near.numel()).double(), (near > 0).sum().double(), near.double().sum(),
        count(top.numel()).double(), top.double().sum(),
    ))


def _centered_scores(probabilities: Tensor, directions: Tensor,
                     action_latents: Tensor) -> Tensor:
    """u.x(a) - sum_b pi(b) u.x(b), in the probabilities' fp32 dtype."""
    scores = directions.to(probabilities.dtype) @ action_latents.to(probabilities.dtype).T
    return scores - (probabilities * scores).sum(-1, keepdim=True)


def delight_inputs(estimator: str, residuals: Tensor, target_advantages: Tensor,
                   gate_advantages: Tensor | None = None) -> Tensor:
    """The U a delightful estimator's gate sigmoid(U * -log pi(a_t)) reads:
    the residual G - c itself, the readout's own A(a_t) for
    ``delightful_head_gate``, or the supplied ``gate_advantages``."""
    if estimator == "delightful_sampled_baseline":
        return residuals
    if estimator == "delightful_head_gate":
        return target_advantages
    if estimator in GATE_ADVANTAGE_ESTIMATORS:
        return gate_advantages.to(residuals.dtype)
    raise ValueError(f"{estimator!r} has no delight gate")


def sampled_coefficients(estimator: str, residuals: Tensor, target_advantages: Tensor,
                         target_logprobs: Tensor,
                         gate_advantages: Tensor | None = None) -> Tensor:
    """The detached return coefficient of the sampled token's log-probability,
    from ``residuals`` G - c: zero, G - Q(a_t) = G - c - A(a_t), G - E_pi Q =
    G - c, or that times its delight gate sigmoid(U * -log pi(a_t)), U from
    ``delight_inputs``. For ``SEQUENCE_ESTIMATORS`` the residuals already are
    the trajectory's coefficient (``sequence_coefficients``)."""
    if estimator == "all_action":
        return torch.zeros_like(target_advantages)
    residuals = residuals.to(target_advantages.dtype)
    if estimator == "control_variate":
        return residuals - target_advantages
    if estimator in DELIGHTFUL_MAC_ESTIMATORS:
        return residuals * delight_gate(
            delight_inputs(estimator, residuals, target_advantages, gate_advantages),
            target_logprobs)
    return residuals


def delight_gate(advantages: Tensor, target_logprobs: Tensor) -> Tensor:
    """sigmoid(U * surprisal), eta = 1, from the current policy's sampled
    log-probability; detached by construction (DG prescribes the gate)."""
    return torch.sigmoid(advantages * -target_logprobs.to(advantages.dtype))


def leave_one_out_advantages(rewards: Tensor) -> Tensor:
    """U_i = R_i - mean_{j != i} R_j within each group of ``rewards``
    [..., G], G >= 2: the RLOO advantage, whose baseline excludes the
    trajectory it scores."""
    size = rewards.size(-1)
    if size < 2:
        raise ValueError("a leave-one-out baseline needs at least two samples per group")
    others = (rewards.sum(-1, keepdim=True) - rewards) / (size - 1)
    return rewards - others


def answer_cluster_labels(answers) -> list[int]:
    """One group's answer-cluster labels: each answer's label is the index of
    its first occurrence, so equal answers share a label. ``None`` (no graded
    answer: unterminated, malformed or abstaining) is one shared cluster,
    the verifier's own equivalence class for a response it cannot grade."""
    first: dict[object, int] = {}
    return [first.setdefault(answer, index) for index, answer in enumerate(answers)]


def answer_cluster_surprisal(labels: Tensor) -> Tensor:
    """l_i = -log(n_k(i) / G) for group labels [..., G]: the surprisal of
    each sample's answer cluster under the group's empirical distribution,
    0 for a unanimous group and log G for a singleton. Its mean over a
    group is that group's plug-in answer entropy."""
    size = labels.size(-1)
    counts = (labels[..., :, None] == labels[..., None, :]).sum(-1)
    return math.log(size) - counts.double().log()


def sequence_coefficients(estimator: str, advantages: Tensor, surprisal: Tensor) -> Tensor:
    """Each trajectory's constant sampled coefficient from its group terms:
    the leave-one-out advantage U for ``rloo_sequence``, or
    sigmoid(U * l) U for ``delightful_answer_gate`` (eta = 1), with l the
    answer cluster's surprisal."""
    if estimator == "rloo_sequence":
        return advantages
    if estimator == "delightful_answer_gate":
        return advantages * answer_gate(advantages, surprisal)
    raise ValueError(f"{estimator!r} is not a sequence estimator")


def answer_gate(advantages: Tensor, surprisal: Tensor) -> Tensor:
    """sigmoid(U * l), eta = 1: ``delight_gate`` with the answer cluster's
    surprisal l = -log(n_k / G) in place of the token's."""
    return delight_gate(advantages, -surprisal.to(advantages.dtype))


def gate_shuffle_strata(target_logprobs: Tensor, answer: Tensor) -> Tensor:
    """Each state's shuffle stratum: the sampled token's surprisal bin under
    ``GATE_SHUFFLE_SURPRISAL_EDGES`` (bin i holds edges[i-1] <= l <
    edges[i]), offset by the bin count in the answer span (``answer`` is
    1 after the think fence, 0 before)."""
    edges = torch.tensor(GATE_SHUFFLE_SURPRISAL_EDGES, dtype=torch.float32,
                         device=target_logprobs.device)
    bins = torch.bucketize(-target_logprobs.float(), edges, right=True)
    return bins + answer.long() * (len(GATE_SHUFFLE_SURPRISAL_EDGES) + 1)


def shuffle_within_strata(values: Tensor, strata: Tensor,
                          generator: torch.Generator) -> Tensor:
    """``values`` permuted uniformly at random within each stratum.

    The states of each stratum, in index order, take that stratum's values
    in the order of i.i.d. uniform keys drawn from ``generator``: a uniform
    permutation per stratum, so each stratum keeps its multiset of values.
    Deterministic given the generator's state, the strata and the device.
    """
    if values.ndim != 1 or strata.shape != values.shape:
        raise ValueError("shuffle values and strata must be matching [states]")
    keys = torch.rand(values.shape, generator=generator, dtype=torch.float64,
                      device=values.device)
    # Stratum-major, key-minor: integers plus keys in [0, 1). Stable sorts,
    # so ties (measure zero) and the home order are deterministic.
    donors = torch.sort(strata.double() + keys, stable=True).indices
    homes = torch.sort(strata, stable=True).indices
    shuffled = torch.empty_like(values)
    shuffled[homes] = values[donors]
    return shuffled


def _readout_policy_and_scores(raw: Tensor, cap: float, directions: Tensor,
                               action_latents: Tensor) -> tuple[Tensor, ...]:
    """(row max, log-normalizer, log-probabilities, probabilities, centered
    Q scores) of a raw readout chunk: the one arithmetic the MAC forward and
    ``softcapped_target_logprobs_and_advantages`` share. Call with autocast
    off."""
    _, row_max, shifted, log_norm = _shifted_softcapped_logits(raw, cap)
    log_probabilities = shifted - log_norm[:, None]
    probabilities = log_probabilities.exp()
    centered = _centered_scores(probabilities, directions, action_latents)
    return row_max, log_norm, log_probabilities, probabilities, centered


def softcapped_target_logprobs_and_advantages(raw_logits: Tensor, targets: Tensor, cap: float,
                                              directions: Tensor,
                                              action_latents: Tensor) -> tuple[Tensor, Tensor]:
    """The sampled token's log-probability and policy-centered Q,
    ``softcapped_mean_actor_critic``'s second and fourth outputs from the
    same arithmetic, without its loss: detached, for a pass that needs every
    state's A(a_t) before the loss (the shuffled head gate)."""
    with torch.no_grad(), torch.autocast(device_type=raw_logits.device.type, enabled=False):
        _, _, log_probabilities, _, centered = _readout_policy_and_scores(
            raw_logits, cap, directions.detach(), action_latents.detach())
        index = targets[:, None]
        return (log_probabilities.gather(-1, index).squeeze(-1),
                centered.gather(-1, index).squeeze(-1))


class _SoftcappedMeanActorCritic(torch.autograd.Function):
    """Save the raw readout and two row statistics; recompute the rest.

    No states*vocab*critic_dim tensor or critic graph exists. The forward's
    normalization arithmetic is ``_shifted_softcapped_logits``, the one the
    other nano readouts share, so the returned log-probabilities agree with
    theirs. Backward is the analytic gradient of the categorical expectation
    with the advantage detached, fused with the softcap derivative.
    """

    @staticmethod
    def forward(ctx, raw: Tensor, targets: Tensor, cap: float, directions: Tensor,
                action_latents: Tensor, row_weights: Tensor, residuals: Tensor,
                estimator: str, gate_advantages: Tensor | None):
        with torch.autocast(device_type=raw.device.type, enabled=False):
            row_max, log_norm, log_probabilities, probabilities, centered = (
                _readout_policy_and_scores(raw, cap, directions, action_latents))
            entropy = -(probabilities * log_probabilities).sum(-1)
            target_logprobs = log_probabilities.gather(-1, targets[:, None]).squeeze(-1)
            target_advantages = centered.gather(-1, targets[:, None]).squeeze(-1)
            q_variance = (probabilities * centered.square()).sum(-1)
            census = support_census(probabilities, centered)
            weights = row_weights.to(probabilities.dtype)
            coefficients = sampled_coefficients(
                estimator, residuals, target_advantages, target_logprobs, gate_advantages)
            loss = -(weights * coefficients * target_logprobs).sum()
            if estimator not in SAMPLED_ONLY_ESTIMATORS:
                loss = loss - (weights[:, None] * probabilities * centered).sum()
            # Last use of the probabilities: the critic's offsets consume them.
            offsets = policy_action_offsets_(probabilities, targets, action_latents)
        ctx.save_for_backward(raw, targets, directions, action_latents, row_weights,
                              coefficients, row_max, log_norm)
        ctx.cap = cap
        ctx.estimator = estimator
        ctx.mark_non_differentiable(target_logprobs, entropy, target_advantages, q_variance,
                                    census, offsets)
        return loss, target_logprobs, entropy, target_advantages, q_variance, census, offsets

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad: Tensor, _logprobs, _entropy, _advantages, _variance, _census,
                 _offsets):
        (raw, targets, directions, action_latents, row_weights, coefficients,
         row_max, log_norm) = ctx.saved_tensors
        cap = ctx.cap
        with torch.autocast(device_type=raw.device.type, enabled=False):
            x = raw.to(log_norm.dtype)
            inverse_norm = (x.square() + cap * cap).rsqrt()
            probabilities = ((cap * x * inverse_norm - row_max[:, None]) - log_norm[:, None]).exp()
            weights = grad * row_weights.to(x.dtype)
            # The sampled score's gradient is c * (onehot(a_t) - pi).
            scores = -coefficients[:, None]
            if ctx.estimator not in SAMPLED_ONLY_ESTIMATORS:
                centered = _centered_scores(probabilities, directions, action_latents)
                # The second centering is the softmax Jacobian applied to the
                # detached advantage, including its fp32 normalization roundoff.
                scores = scores + centered - (probabilities * centered).sum(-1, keepdim=True)
            grad_capped = (-probabilities * scores * weights[:, None]).scatter_add(
                -1, targets[:, None], -(weights * coefficients)[:, None])
            grad_raw = grad_capped * cap**3 * inverse_norm**3
        return grad_raw.to(raw.dtype), None, None, None, None, None, None, None, None


def softcapped_mean_actor_critic(raw_logits: Tensor, targets: Tensor, cap: float,
                                 directions: Tensor, action_latents: Tensor,
                                 row_weights: Tensor, residuals: Tensor,
                                 estimator: str,
                                 gate_advantages: Tensor | None = None) -> tuple[Tensor, ...]:
    """A token Q estimator's loss at raw readout logits under
    ``cap*x/sqrt(x*x+cap*cap)``.

    ``residuals`` are each state's Monte Carlo return minus the critic's
    state value, G - c(s); ``all_action`` ignores them. ``gate_advantages``
    [states] are the critic's advantages of the sampled tokens, read only by
    (and required for) ``GATE_ADVANTAGE_ESTIMATORS``. Returns (loss,
    sampled-token log-probability, entropy, sampled-token centered Q, policy
    variance of Q, the chunk's support census sums ``SUPPORT_CENSUS_FIELDS``,
    the (states, critic_dim) fp32 offset x(a_t) - sum_a pi(a) x(a) that
    centres the critic's fit); only the loss is differentiable, and only in
    ``raw_logits``. Weights are used verbatim, so chunks keep their share of
    a global normalization. Validate ``cap`` with ``validate_softcap`` and
    ``estimator`` with ``validate_mac_estimator`` outside any compiled region.
    """
    if raw_logits.ndim != 2 or not raw_logits.size(0) or not raw_logits.size(1):
        raise ValueError("MAC requires nonempty raw logits [states, vocab]")
    if targets.shape != raw_logits.shape[:1] or row_weights.shape != raw_logits.shape[:1]:
        raise ValueError("MAC targets and row weights must be [states]")
    if residuals.shape != raw_logits.shape[:1]:
        raise ValueError("MAC residuals must be [states]")
    if (gate_advantages is not None) != (estimator in GATE_ADVANTAGE_ESTIMATORS):
        raise ValueError(
            "gate advantages come with, and only with, a supplied-gate estimator "
            f"{GATE_ADVANTAGE_ESTIMATORS}")
    if gate_advantages is not None and gate_advantages.shape != raw_logits.shape[:1]:
        raise ValueError("MAC gate advantages must be [states]")
    if directions.ndim != 2 or directions.size(0) != raw_logits.size(0):
        raise ValueError("critic directions must be [states, critic_dim]")
    if action_latents.shape != (raw_logits.size(1), directions.size(1)):
        raise ValueError("critic action latents must be [vocab, critic_dim]")
    return _SoftcappedMeanActorCritic.apply(
        raw_logits, targets, cap, directions.detach(), action_latents.detach(),
        row_weights.detach(), residuals.detach(), estimator,
        None if gate_advantages is None else gate_advantages.detach())


def validate_softcap(cap) -> float:
    """The renderer's softcap as a positive finite Python float."""
    if type(cap) not in (int, float):
        raise TypeError("rational softcap must be a Python scalar")
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("rational softcap must be positive and finite")
    return float(cap)


def validate_mac_estimator(estimator, *, controls: bool = False) -> str:
    """One of ``MAC_ESTIMATORS``, or with ``controls`` also of
    ``MAC_CONTROL_ESTIMATORS``."""
    known = MAC_ESTIMATORS + (MAC_CONTROL_ESTIMATORS if controls else ())
    if estimator not in known:
        raise ValueError(f"unknown MAC estimator {estimator!r}; expected one of {known}")
    return estimator


def nano_mean_actor_critic(backbone, beliefs: Tensor, targets: Tensor, directions: Tensor,
                           action_latents: Tensor, row_weights: Tensor, residuals: Tensor,
                           estimator: str, gate_advantages: Tensor | None = None):
    """A token Q estimator at a nano renderer's own readout of compact emit
    beliefs.

    The nano readout is belief-only (``logits_from_features`` slices the
    belief half of the renderer features), so ``_raw_logits`` plus the
    backbone's softcap is exactly the sampling policy. Call and backward
    each chunk immediately: accumulating chunk losses would keep every
    vocabulary readout alive.
    """
    return softcapped_mean_actor_critic(
        backbone._raw_logits(beliefs), targets, backbone.logit_softcap,
        directions, action_latents, row_weights, residuals, estimator, gate_advantages)


def nano_target_logprobs_and_advantages(backbone, beliefs: Tensor, targets: Tensor,
                                        directions: Tensor, action_latents: Tensor):
    """``softcapped_target_logprobs_and_advantages`` at a nano renderer's
    own readout of compact emit beliefs, as ``nano_mean_actor_critic``
    reads them (the readout GEMM in the caller's autocast, as there)."""
    with torch.no_grad():
        return softcapped_target_logprobs_and_advantages(
            backbone._raw_logits(beliefs), targets, backbone.logit_softcap,
            directions, action_latents)
