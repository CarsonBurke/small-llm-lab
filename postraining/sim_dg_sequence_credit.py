"""Toy simulation: token-level Delightful PG under sequence-level rewards.

Question: does DG's surprisal gate help or hurt when one sequence reward is
broadcast over many tokens, most of which do not cause success (the regime of
our math RL: binary reward, a few percent success, lucky guesses, and a long
low-probability token tail)?

A response is ``filler`` tokens, optionally followed by one answer token. The
estimators mirror the author's reference code (google-deepmind/egg,
``egg/losses/dg.py`` and ``reinforce.py``): sequence reward minus the prompt
group's mean reward, broadcast to every response token, coefficient
``sigmoid(U * l / eta) * U`` for DG (``U`` for PG) with ``l`` the current
policy's stop-gradient surprisal, and the loss normalized by the token count.
The ``gae`` advantage instead models our run: length-adaptive-lambda GAE over
a critic that knows only the running mean return, plus per-token value noise.

Two policy parameterizations share one initial distribution:

- ``tabular``: an independent logit delta for every (prompt, position). Every
  effect is attributable to the estimator, but nothing is shared, so DG's
  claimed benefit (reallocating a shared gradient budget across contexts, DG
  paper Prop 2) cannot appear.
- ``mlp``: the initial logits plus a residual MLP over fixed random context
  features, output layer zero-initialized, so step 0 matches ``tabular`` and
  every context competes for the same parameters.

Tasks:

- ``lottery``: reward = answer correct; filler has no causal effect at all.
- ``coherence``: reward = answer correct AND a Bernoulli draw whose success
  probability is the fraction of filler tokens inside each position's
  initial top-k "coherent" set, so junk filler is genuinely harmful but each
  junk token moves success only slightly (sigma/Delta >> 1, the gambling
  regime of Osband, "Does This Gradient Spark Joy?", section 4.2).
- ``chain``: reward = answer correct AND each of ``causal`` evenly spaced
  filler positions emits its own target token (rare but causal tokens, the
  "breakthrough" case DG is designed for); binary.
- ``chain_rare``: one causal position whose target starts at 1-10%.
- ``chain_coherence``: ``chain`` plus the junk penalty over the non-causal
  filler.
- ``graded``: egg's token-reversal analogue. Every position has a target
  token over a binary vocabulary and reward is the fraction correct; all
  tokens are causal.
- ``sequential``: the same with egg's ``reward_to_first_error`` (credit only
  up to the first wrong token) over 16 positions, the setting where the DG
  paper reports its larger gains.

Estimators are ``{gate}_{advantage}``. Besides ``pg`` and ``dg`` (fixed
temperature ``eta``), two gates set the temperature from the batch's own
advantage scale, keeping the raw advantage as the magnitude:

- ``dgrms``: ``eta * RMS(A)`` over the run's tokens.
- ``dgbase``: ``eta * b``, with ``b`` the mean reward the baseline stands for:
  the prompt's group mean for ``group``, and for ``gae`` and ``oracle`` the
  critics' running mean return (the ``gae`` critic's whole value level; a
  critic has no per-prompt constant baseline). For binary rewards a failure's
  group advantage is then ``-b``, the DG paper's ``b / eta = 1`` regime
  (Prop 1); fractional rewards (``graded``, ``sequential``) do not map onto it.

``dgrms`` includes the zero advantages of all-equal groups, so its scale
also tracks the fraction of mixed groups.

Three DG gates subtract the coefficient a token with no effect on R expects,
removing DG's rectification of advantage spread into a rarity bonus:
``dgquad`` its second-order value l Var(A) / (4 eta) (Var over the group's
samples at the position), ``dgexact`` the exact value for binary group
advantages at the group's success rate, and ``dgprefix`` the same with the
exact prefix success probability E[R | h_t] in place of the group's.

Advantages beyond ``group``, ``gae`` and ``oracle`` (lambda-``--oracle-
lambda`` GAE over exact values): ``loo`` (leave-one-out baseline),
``groupnorm`` (divided by the group's reward std), and three built on the
exact action values Q(s_t, a) = E[R | prefix, a] for every token, corrupted
like the oracle critic (shrink toward the running mean, iid error per
(state, action) entry):

- ``tokenq``: Q(s_t, a_t) - pi . Q(s_t, .), the production trainer's
  policy-centred token advantage; with an exact Q it is lambda-0 credit.
- ``qprop``: Q-Prop, (R - Q(s_t, a_t)) on the sampled token (gated for DG)
  plus the ungated analytic term sum_a pi(a) Q(s_t, a).
- ``split``: 0.5 (R - b_loo - A_t) + g A_t with A_t the ``tokenq`` advantage
  and g the DG gate, or 1/2 for PG, which then is (R - b_loo) / 2.

``expected_success`` is the exact expected reward for the answer tasks and
the mean per-position target probability for ``graded``/``sequential``.

Every (estimator, seed) run is one slice of a leading batch dimension: the
parameters are disjoint slices, each run's loss depends only on its own
slice, and Adam/SGD act elementwise, so batching is exact. Runs with the same
seed share their initial policy and features; all runs draw samples from one
generator.

Runs on the CPU; it executes no language model. The GPU is hidden before
torch loads because the optimizer's graph-capture health check would
otherwise open a CUDA context on the shared device.

    python -m postraining.sim_dg_sequence_credit --task coherence --policy mlp
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

ADVANTAGES = ("group", "loo", "groupnorm", "gae", "oracle", "tokenq", "qprop", "split")
GATES = ("pg", "dg", "dgrms", "dgbase", "dgquad", "dgexact", "dgprefix")
# DG gates that subtract an estimate of the coefficient a null token gets.
CENTRED = ("dgquad", "dgexact", "dgprefix")
GROUP_KINDS = ("group", "loo", "groupnorm")
ESTIMATORS = tuple(f"{g}_{a}" for a in ADVANTAGES for g in ("pg", "dg"))


@dataclass(frozen=True)
class Task:
    name: str
    prompts: int
    filler: int
    vocab: int
    coherent_top_k: int
    filler_entropy: float
    # Initial per-prompt answer success probability, log-uniform in [lo, hi];
    # None means the per-position target tasks (graded/sequential).
    answer_success: tuple[float, float] | None
    junk_penalty: bool = False
    causal: int = 0
    causal_success: tuple[float, float] = (0.2, 0.6)


TASKS = {
    "lottery": Task("lottery", 64, 48, 256, 5, 1.0, (0.002, 0.2)),
    "coherence": Task("coherence", 64, 48, 256, 5, 1.0, (0.004, 0.4), junk_penalty=True),
    "chain": Task("chain", 64, 48, 256, 5, 1.0, (0.05, 0.5), causal=3),
    # Our regime: about 3% initial success.
    "coherence_low": Task("coherence_low", 64, 48, 256, 5, 1.0, (0.003, 0.1), junk_penalty=True),
    "chain_rare": Task(
        "chain_rare", 64, 48, 256, 5, 1.0, (0.1, 0.6), causal=1, causal_success=(0.01, 0.1)
    ),
    "chain_coherence": Task(
        "chain_coherence", 64, 48, 256, 5, 1.0, (0.05, 0.5), junk_penalty=True, causal=3
    ),
    "graded": Task("graded", 64, 10, 2, 1, 0.6, None),
    "sequential": Task("sequential", 64, 16, 2, 1, 0.6, None),
}


def zipf_logits(shape: tuple[int, ...], vocab: int, entropy: float, generator) -> torch.Tensor:
    """Randomly permuted Zipf logits whose exponent is bisected to ``entropy``."""
    log_ranks = torch.arange(1, vocab + 1, dtype=torch.float64).log()
    low, high = 0.0, 20.0
    for _ in range(60):
        exponent = (low + high) / 2
        probs = torch.softmax(-exponent * log_ranks, 0)
        value = float(-(probs * probs.log()).sum())
        low, high = (exponent, high) if value > entropy else (low, exponent)
    base = (-exponent * log_ranks).float()
    return base[torch.rand((*shape, vocab), generator=generator).argsort(-1)]


def set_probability(logits: torch.Tensor, target: torch.Tensor, probability: torch.Tensor) -> None:
    """Set each row's ``target`` logit in place so its probability is exact."""
    rest = logits.scatter(-1, target[..., None], float("-inf")).logsumexp(-1)
    logits.scatter_(-1, target[..., None], (rest + torch.log(probability / (1 - probability)))[..., None])


def log_uniform(bounds: tuple[float, float], shape: tuple[int, ...], generator) -> torch.Tensor:
    low, high = (math.log(p) for p in bounds)
    return torch.exp(low + (high - low) * torch.rand(shape, generator=generator))


def initial_policy(task: Task, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    filler = zipf_logits((task.prompts, task.filler), task.vocab, task.filler_entropy, generator)
    policy = {"coherent": filler.topk(task.coherent_top_k, -1).indices}
    if task.answer_success is None:
        policy["target"] = torch.randint(0, task.vocab, (task.prompts, task.filler), generator=generator)
        policy["logits"] = filler
        return policy
    # Coherence is defined by the initial top-k, before causal targets move.
    if task.causal:
        positions = torch.linspace(0, task.filler - 1, task.causal + 2)[1:-1].round().long()
        causal_target = torch.randint(0, task.vocab, (task.prompts, task.causal), generator=generator)
        sub = filler[:, positions]
        set_probability(
            sub, causal_target, log_uniform(task.causal_success, (task.prompts, task.causal), generator)
        )
        filler[:, positions] = sub
        policy.update(causal_positions=positions, causal_target=causal_target)
    answer = zipf_logits((task.prompts,), task.vocab, 1.5, generator)
    target = torch.randint(0, task.vocab, (task.prompts,), generator=generator)
    set_probability(answer, target, log_uniform(task.answer_success, (task.prompts,), generator))
    policy.update(logits=torch.cat((filler, answer[:, None]), 1), target=target)
    return policy


class Policy:
    """Initial logits plus a trainable delta, batched over runs."""

    def __init__(self, base: torch.Tensor, kind: str, args, seeds: list[int]):
        self.base, self.kind = base, kind
        n_runs, prompts, positions, vocab = base.shape
        if kind == "tabular":
            self.delta = torch.zeros_like(base, requires_grad=True)
            self.params = [self.delta]
            return
        features, first = [], []
        for seed in seeds:
            generator = torch.Generator().manual_seed(20_000 + seed)
            features.append(torch.randn(prompts * positions, args.features, generator=generator))
            first.append(
                torch.randn(args.features, args.hidden, generator=generator) / math.sqrt(args.features)
            )
        self.features = torch.stack(features)
        self.first = torch.stack(first).requires_grad_()
        self.bias = torch.zeros(n_runs, 1, args.hidden, requires_grad=True)
        self.second = torch.zeros(n_runs, args.hidden, vocab, requires_grad=True)
        self.params = [self.first, self.bias, self.second]

    def logits(self) -> torch.Tensor:
        if self.kind == "tabular":
            return self.base + self.delta
        hidden = F.gelu(torch.bmm(self.features, self.first) + self.bias)
        return self.base + torch.bmm(hidden, self.second).view_as(self.base)


def gae_advantages(values: torch.Tensor, rewards: torch.Tensor, lam: float) -> torch.Tensor:
    """GAE (gamma 1) over per-token state values; the reward lands on the last token.

    ``values`` is (runs, prompts, samples, length), the value of the state
    before each token, and ``rewards`` (runs, prompts, samples).
    """
    next_values = torch.cat((values[..., 1:], rewards[..., None]), -1)
    deltas = next_values - values
    advantages = torch.empty_like(values)
    running = torch.zeros_like(rewards)
    for t in range(values.shape[-1] - 1, -1, -1):
        running = deltas[..., t] + lam * running
        advantages[..., t] = running
    return advantages


def exclusive_prefix(x: torch.Tensor, op: str) -> torch.Tensor:
    """Length n+1: entry t reduces x[..., :t] (empty reduction at t = 0)."""
    identity = torch.ones_like(x[..., :1]) if op == "prod" else torch.zeros_like(x[..., :1])
    return torch.cat((identity, x.cumprod(-1) if op == "prod" else x.cumsum(-1)), -1)


def inclusive_suffix(x: torch.Tensor, op: str) -> torch.Tensor:
    """Length n+1: entry t reduces x[..., t:] (empty reduction at t = n)."""
    identity = torch.ones_like(x[..., :1]) if op == "prod" else torch.zeros_like(x[..., :1])
    flipped = x.flip(-1)
    reduced = (flipped.cumprod(-1) if op == "prod" else flipped.cumsum(-1)).flip(-1)
    return torch.cat((reduced, identity), -1)


def oracle_values(task: Task, probs, tokens, context) -> torch.Tensor:
    """Exact expected reward of the state before each token, under ``probs``.

    With these values, lambda-0 GAE gives every token exactly its causal
    effect V(s_{t+1}) - V(s_t) plus, on the last token, the terminal noise;
    a non-causal token's advantage is exactly zero.
    """
    filler_probs = probs[:, :, : task.filler]
    filler_tokens = tokens[..., : task.filler]
    if task.answer_success is None:
        p = filler_probs.gather(-1, context["target"][..., None]).squeeze(-1)[:, :, None]
        correct = (filler_tokens == context["target"][:, :, None]).float()
        if task.name != "sequential":
            realized = exclusive_prefix(correct, "sum")
            return ((realized + inclusive_suffix(p, "sum").expand_as(realized)) / task.filler)[..., :-1]
        # Reward (1/F) sum_k prod_{j<=k} c_j: realized prefix credit, then the
        # expected credit of the remaining run while the prefix is still correct.
        alive = exclusive_prefix(correct, "prod")
        earned = exclusive_prefix(correct.cumprod(-1), "sum")
        remaining = torch.zeros_like(p[..., :1])
        tails = [remaining]
        for j in range(task.filler - 1, -1, -1):
            remaining = p[..., j : j + 1] * (1 + remaining)
            tails.append(remaining)
        tail = torch.cat(tails[::-1], -1)
        return ((earned + alive * tail) / task.filler)[..., :-1]
    answer = probs[:, :, -1].gather(-1, context["target"][..., None]).squeeze(-1)[:, :, None, None]
    value = answer.expand(*filler_tokens.shape[:-1], task.filler + 1)
    if task.causal:
        positions, causal_target = context["causal_positions"], context["causal_target"]
        realized = torch.ones_like(filler_tokens, dtype=probs.dtype)
        realized[..., positions] = (filler_tokens[..., positions] == causal_target[:, :, None]).to(probs.dtype)
        expected = torch.ones_like(filler_probs[..., 0])
        expected[..., positions] = filler_probs[:, :, positions].gather(-1, causal_target[..., None]).squeeze(-1)
        value = value * exclusive_prefix(realized, "prod") * inclusive_suffix(expected, "prod")[:, :, None]
    if task.junk_penalty:
        noncausal = context["noncausal"].to(probs.dtype)
        in_set = (filler_tokens[..., None] == context["coherent"][:, :, None]).any(-1).to(probs.dtype)
        mass = filler_probs.gather(-1, context["coherent"]).sum(-1)
        fraction = exclusive_prefix(in_set * noncausal, "sum") + inclusive_suffix(mass * noncausal, "sum")[:, :, None]
        value = value * fraction / noncausal.sum()
    return value


def oracle_action_values(task: Task, probs, tokens, context) -> tuple[torch.Tensor, ...]:
    """Exact Q(s_t, a) = E[R | prefix, a_t = a] for every token, in factored form.

    In every task the token at position t moves the expected reward through
    one 0/1 feature of that token (its target, the causal target, or
    coherent-set membership), so Q(s_t, a) = alpha_t + beta_t * f_t(a).
    Returns ``alpha`` and ``beta`` (runs, prompts, samples, length) and
    ``feature`` (runs, prompts, length, vocab); the feature depends only on
    the position, never on the sampled prefix. At the answer position Q is the
    expected reward of the complete response, without the coherence draw.
    Consistency with ``oracle_values``: V(s_t) = alpha_t + beta_t * pi_t . f_t,
    and Q(s_t, a_t) = V(s_{t+1}).
    """
    runs, prompts, samples, length = tokens.shape
    vocab = probs.shape[-1]
    dtype = probs.dtype
    filler_probs = probs[:, :, : task.filler]
    filler_tokens = tokens[..., : task.filler]
    feature = torch.zeros(runs, prompts, length, vocab, dtype=dtype)
    if task.answer_success is None:
        target = context["target"]
        feature.scatter_(-1, target[..., None], 1.0)
        p = filler_probs.gather(-1, target[..., None]).squeeze(-1)[:, :, None]
        correct = (filler_tokens == target[:, :, None]).to(dtype)
        if task.name != "sequential":
            alpha = (exclusive_prefix(correct, "sum")[..., :-1] + inclusive_suffix(p, "sum")[..., 1:]) / task.filler
            return alpha, torch.full_like(alpha, 1 / task.filler), feature
        tails = [torch.zeros_like(p[..., :1])]
        for j in range(task.filler - 1, -1, -1):
            tails.append(p[..., j : j + 1] * (1 + tails[-1]))
        tail = torch.cat(tails[::-1], -1)  # tail[j]: expected credit from position j on
        alpha = exclusive_prefix(correct.cumprod(-1), "sum")[..., :-1] / task.filler
        beta = exclusive_prefix(correct, "prod")[..., :-1] * (1 + tail[..., 1:]) / task.filler
        return alpha, beta, feature
    target = context["target"]
    answer = probs[:, :, -1].gather(-1, target[..., None]).squeeze(-1)[:, :, None, None]
    feature[:, :, -1].scatter_(-1, target[..., None], 1.0)
    # Causal product with position t left out: realized before t, expected after.
    realized = torch.ones_like(filler_tokens, dtype=dtype)
    expected = torch.ones_like(filler_probs[..., 0])
    causal = torch.zeros(task.filler, dtype=torch.bool)
    if task.causal:
        positions, causal_target = context["causal_positions"], context["causal_target"]
        causal[positions] = True
        realized[..., positions] = (filler_tokens[..., positions] == causal_target[:, :, None]).to(dtype)
        expected[..., positions] = filler_probs[:, :, positions].gather(-1, causal_target[..., None]).squeeze(-1)
        feature[:, :, positions] = feature[:, :, positions].scatter(-1, causal_target[..., None], 1.0)
    chain_out = exclusive_prefix(realized, "prod")[..., :-1] * inclusive_suffix(expected, "prod")[..., 1:][:, :, None]
    # Junk fraction with position t left out, and the weight of t's own token.
    junk_out = torch.ones_like(chain_out)
    junk_weight = torch.zeros(task.filler, dtype=dtype)
    junk_realized = torch.ones_like(filler_tokens[..., 0], dtype=dtype)
    if task.junk_penalty:
        noncausal = context["noncausal"].to(dtype)
        in_set = (filler_tokens[..., None] == context["coherent"][:, :, None]).any(-1).to(dtype)
        mass = filler_probs.gather(-1, context["coherent"]).sum(-1)
        junk_out = (
            exclusive_prefix(in_set * noncausal, "sum")[..., :-1]
            + inclusive_suffix(mass * noncausal, "sum")[..., 1:][:, :, None]
        ) / noncausal.sum()
        junk_weight = noncausal / noncausal.sum()
        junk_realized = (in_set * noncausal).sum(-1) / noncausal.sum()
        coherent = torch.zeros_like(feature[:, :, : task.filler]).scatter_(-1, context["coherent"], 1.0)
        feature[:, :, : task.filler] = torch.where(causal[:, None], feature[:, :, : task.filler], coherent)
    base = answer * chain_out
    alpha = torch.where(causal, 0.0, base * junk_out)
    beta = torch.where(causal, base * junk_out, base * junk_weight)
    # Answer position: Q = 1[a = target] * realized chain * realized junk fraction.
    final = realized.prod(-1) * junk_realized
    alpha = torch.cat((alpha, torch.zeros_like(final[..., None])), -1)
    beta = torch.cat((beta, final[..., None]), -1)
    return alpha, beta, feature


def gate_temperature(
    advantage: torch.Tensor,
    rewards: torch.Tensor,
    running_mean: torch.Tensor,
    gate_mask: dict[str, torch.Tensor],
    group_mask: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    """Per-run gate temperature, broadcastable to ``advantage``.

    ``advantage`` is (runs, prompts, samples, length), ``rewards`` (runs,
    prompts, samples), ``running_mean`` (runs,), the critics' value level
    before this step's update, and the masks (runs, 1, 1, 1). A zero scale
    means every advantage it covers is zero (group) or the critic has seen no
    reward yet; the temperature falls back to ``eta`` there.
    """
    rms = advantage.square().mean((1, 2, 3), keepdim=True).sqrt()
    prompt_base = rewards.mean(-1)[..., None, None]
    critic_base = running_mean[:, None, None, None]
    base = torch.where(group_mask, prompt_base, critic_base)
    one = torch.ones_like(base)
    scale = torch.where(gate_mask["dgrms"], rms, torch.where(gate_mask["dgbase"], base, one))
    return eta * torch.where(scale > 0, scale, one)


def null_coefficient(success, p_hat, delight_scale) -> torch.Tensor:
    """E[U sigmoid(U s)] for U = R - p_hat, R ~ Bernoulli(success), s = l / eta.

    The DG coefficient a token with no effect on R expects; with
    ``success = p_hat`` it is p(1-p)[sigmoid((1-p)s) - sigmoid(-p s)], whose
    second-order expansion is s p(1-p) / 4 = s Var(U) / 4.
    """
    rise = (1 - p_hat) * torch.sigmoid((1 - p_hat) * delight_scale)
    fall = -p_hat * torch.sigmoid(-p_hat * delight_scale)
    return success * rise + (1 - success) * fall


def entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, -1)
    return -(logp.exp() * logp).sum(-1)


def entropy_gradient(logp: torch.Tensor) -> torch.Tensor:
    """dH/dz for a softmax over logits z: -pi_k (log pi_k + H)."""
    probs = logp.exp()
    return -probs * (logp - (probs * logp).sum(-1, keepdim=True))


def entropy_drift(logp, logit_grad, task: Task, noncausal) -> torch.Tensor:
    """First-order entropy change of one plain gradient step on the logits.

    ``logit_grad`` is the loss gradient on the logits scaled to a per-sample
    mean, so the result is in coefficient units per state: what a tabular
    policy's entropy would do per unit learning rate. Averaged over prompts
    and the non-causal filler positions; returns (runs,).
    """
    drift = -(entropy_gradient(logp) * logit_grad).sum(-1)[:, :, : task.filler]
    return drift[..., noncausal].mean((1, 2))


def noncausal_credit(task: Task, logp, token_logp, advantage, coefficient, temperature, noncausal) -> dict:
    """Sampled-term credit on non-causal filler tokens, per run.

    ``coef_surprisal_corr``: Pearson correlation of the sampled coefficient
    with the token's surprisal. ``excess_drift``: the first-order (softmax,
    per token) entropy change of the coefficient's excess over half the
    advantage, the part by which DG differs from PG/2. ``rectified``: the
    second-order prediction of that excess's mean, l Var(A) / (4 eta), with
    Var(A) over the group's samples at the position.
    """
    pick = lambda x: x[..., : task.filler][..., noncausal]  # (runs, prompts, samples, n)
    surprisal = pick(-token_logp)
    # Per state: H and sum_k pi_k^2 (log pi_k + H), broadcast over samples.
    state_logp = logp[:, :, : task.filler][:, :, noncausal][:, :, None]
    state_probs = state_logp.exp()
    h = -(state_probs * state_logp).sum(-1)
    curvature = (state_probs.square() * (state_logp + h[..., None])).sum(-1)
    # <dH/dz, e_a - pi> = pi_a (l - H) + sum_k pi_k^2 (log pi_k + H).
    direction = (-surprisal).exp() * (surprisal - h) + curvature
    coef, adv = pick(coefficient), pick(advantage)
    excess = coef - adv / 2
    c, l = coef.flatten(1), surprisal.flatten(1)
    c, l = c - c.mean(1, keepdim=True), l - l.mean(1, keepdim=True)
    corr = (c * l).sum(1) / (c.square().sum(1) * l.square().sum(1)).sqrt().clamp_min(1e-30)
    spread = pick(advantage.var(2, unbiased=False, keepdim=True).expand_as(advantage))
    return {
        "noncausal_coef_surprisal_corr": corr.tolist(),
        "noncausal_excess_drift": (excess * direction).mean((1, 2, 3)).tolist(),
        "noncausal_rectified": (surprisal * spread / (4 * temperature)).mean((1, 2, 3)).tolist(),
        "noncausal_excess_mean": excess.mean((1, 2, 3)).tolist(),
        "noncausal_advantage_var": spread.mean((1, 2, 3)).tolist(),
    }


def simulate(task: Task, estimators: list[str], args) -> dict:
    for estimator in estimators:
        gate, _, kind = estimator.partition("_")
        if gate not in GATES or kind not in ADVANTAGES:
            raise ValueError(f"unknown estimator {estimator!r}")
        # The exact null coefficient assumes U = R - group mean with R in {0, 1}.
        if gate in ("dgexact", "dgprefix") and (kind != "group" or task.answer_success is None):
            raise ValueError(f"{estimator!r} needs group advantages over a binary-reward task")
        if kind == "split" and gate not in ("pg", "dg"):
            raise ValueError(f"{estimator!r}: split is defined for the pg and dg gates only")
    runs = [(estimator, seed) for estimator in estimators for seed in range(args.seeds)]
    inits = [initial_policy(task, seed) for seed in range(args.seeds)]
    stack = lambda key: torch.stack([inits[seed][key] for _, seed in runs])
    policy = Policy(stack("logits"), args.policy, args, [seed for _, seed in runs])
    coherent, target = stack("coherent"), stack("target")
    has_answer = task.answer_success is not None
    if task.causal:
        causal_positions = inits[0]["causal_positions"]
        causal_target = stack("causal_target")
        noncausal = torch.ones(task.filler, dtype=torch.bool)
        noncausal[causal_positions] = False
    else:
        noncausal = torch.ones(task.filler, dtype=torch.bool)
    optimizer = (
        torch.optim.Adam(policy.params, lr=args.lr)
        if args.optimizer == "adam"
        else torch.optim.SGD(policy.params, lr=args.lr)
    )
    gate_kind = [e.split("_", 1)[0] for e, _ in runs]
    gate_mask = {
        gate: torch.tensor([g == gate for g in gate_kind])[:, None, None, None] for gate in GATES
    }
    is_dg = ~gate_mask["pg"]
    advantage_kind = [e.split("_", 1)[1] for e, _ in runs]
    kind_mask = {
        kind: torch.tensor([k == kind for k in advantage_kind])[:, None, None, None]
        for kind in ADVANTAGES
    }
    context = {"target": target, "coherent": coherent, "noncausal": noncausal}
    if task.causal:
        context.update(causal_positions=causal_positions, causal_target=causal_target)
    n_runs, prompts, samples = len(runs), task.prompts, args.samples
    length = task.filler + has_answer
    # Horizon floor of the trainer's length_adaptive_lambda at alpha = 0.05.
    lam = 1 - 1 / max(0.05 * length, min(length, 20))
    running_mean = torch.zeros(n_runs)
    generator = torch.Generator().manual_seed(10_000)
    history = []
    for step in range(args.steps + 1):
        logits = policy.logits()
        logp = F.log_softmax(logits, -1)  # (runs, prompts, length, vocab)
        with torch.no_grad():
            drawn = torch.multinomial(
                logp.exp().flatten(0, 2), samples, replacement=True, generator=generator
            ).view(n_runs, prompts, length, samples)
        # (runs, prompts, samples, length)
        token_logp = logp.gather(-1, drawn).transpose(2, 3)
        tokens = drawn.transpose(2, 3)
        filler_tokens = tokens[..., : task.filler]
        with torch.no_grad():
            if not has_answer:
                correct = (filler_tokens == target[:, :, None]).float()
                # egg's reward_to_first_error: credit the correct prefix only.
                rewards = (correct.cumprod(-1) if task.name == "sequential" else correct).mean(-1)
            else:
                rewards = (tokens[..., -1] == target[:, :, None]).float()
                if task.causal:
                    hits = filler_tokens[..., causal_positions] == causal_target[:, :, None]
                    rewards = rewards * hits.all(-1).float()
                if task.junk_penalty:
                    in_set = (filler_tokens[..., None] == coherent[:, :, None]).any(-1)
                    fraction = in_set[..., noncausal].float().mean(-1)
                    rewards = rewards * (torch.rand(rewards.shape, generator=generator) < fraction)
            centred = rewards - rewards.mean(-1, keepdim=True)
            group = centred[..., None].expand_as(token_logp)
            # R minus the mean of the group's other samples.
            loo = group * samples / (samples - 1)
            noise = lambda std: std * torch.randn(token_logp.shape, generator=generator)
            advantage = torch.zeros_like(token_logp)
            if kind_mask["group"].any():
                advantage = torch.where(kind_mask["group"], group, advantage)
            if kind_mask["loo"].any():
                advantage = torch.where(kind_mask["loo"], loo, advantage)
            if kind_mask["groupnorm"].any():
                normalized = centred / (rewards.std(-1, keepdim=True) + 1e-4)
                advantage = torch.where(kind_mask["groupnorm"], normalized[..., None].expand_as(token_logp), advantage)
            if kind_mask["gae"].any():
                values = running_mean[:, None, None, None] + noise(args.value_noise)
                advantage = torch.where(kind_mask["gae"], gae_advantages(values, rewards, lam), advantage)
            if kind_mask["oracle"].any():
                # A partial critic: the running mean plus a fraction of the
                # exact value structure, plus iid per-state error.
                exact = oracle_values(task, logp.exp(), tokens, context)
                mean = running_mean[:, None, None, None]
                values = mean + args.oracle_shrink * (exact - mean) + noise(args.oracle_value_noise)
                advantage = torch.where(
                    kind_mask["oracle"], gae_advantages(values, rewards, args.oracle_lambda), advantage
                )
            expected_q = None
            token_q = kind_mask["tokenq"] | kind_mask["split"]
            if token_q.any() or kind_mask["qprop"].any():
                # The same partial critic over actions: Q~(s, a) = mean +
                # shrink * (Q(s, a) - mean) + an iid error per (state, action)
                # entry. Each sample's prefix is its own state.
                alpha, beta, feature = oracle_action_values(task, logp.exp(), tokens, context)
                mean = running_mean[:, None, None, None]
                taken = feature.gather(-1, drawn).transpose(2, 3)
                error = noise(args.oracle_value_noise)  # the error of the sampled entry
                q_taken = mean + args.oracle_shrink * (alpha + beta * taken - mean) + error
                if token_q.any():
                    # sum_a pi(a) Q~(s, a). Given the sampled entry's error,
                    # the other entries' errors are independent of it.
                    probs = logp.exp()
                    p_taken = token_logp.exp()
                    covered = (probs.square().sum(-1)[:, :, None] - p_taken.square()).clamp_min(0)
                    q_policy = (
                        mean
                        + args.oracle_shrink * (alpha + beta * (probs * feature).sum(-1)[:, :, None] - mean)
                        + p_taken * error
                        + covered.sqrt() * noise(args.oracle_value_noise)
                    )
                    advantage = torch.where(token_q, q_taken - q_policy, advantage)
                if kind_mask["qprop"].any():
                    advantage = torch.where(kind_mask["qprop"], rewards[..., None] - q_taken, advantage)
                    # Sum over samples of Q~_s(s_t, a) for every a, dropping
                    # a-independent terms (no gradient, since sum_a pi = 1).
                    # Sampled entries carry the errors drawn above; the
                    # samples - n_a unsampled ones are fresh.
                    counts = torch.zeros_like(feature).scatter_add_(-1, drawn, torch.ones_like(drawn, dtype=feature.dtype))
                    errors = torch.zeros_like(feature).scatter_add_(-1, drawn, error.transpose(2, 3))
                    fresh = args.oracle_value_noise * (samples - counts).sqrt()
                    fresh = fresh * torch.randn(feature.shape, generator=generator)
                    expected_q = args.oracle_shrink * beta.sum(2)[..., None] * feature + errors + fresh
                    expected_q = expected_q * kind_mask["qprop"]
            surprisal = -token_logp.detach()
            temperature = gate_temperature(
                advantage,
                rewards,
                running_mean,
                gate_mask,
                kind_mask["group"] | kind_mask["loo"] | kind_mask["groupnorm"],
                args.eta,
            )
            gate = torch.where(
                is_dg, torch.sigmoid(advantage * surprisal / temperature), torch.ones_like(advantage)
            )
            # ``split``: the token's own credit A_t through the gate, plus the
            # outcome luck left over, R - b_loo - A_t, ungated at the gate's
            # small-signal weight 1/2. PG then reduces to (R - b_loo) / 2.
            gate = torch.where(kind_mask["split"] & ~is_dg, torch.full_like(gate, 0.5), gate)
            coefficient = gate * advantage
            if kind_mask["split"].any():
                residual = 0.5 * (loo - advantage)
                coefficient = coefficient + torch.where(kind_mask["split"], residual, torch.zeros_like(residual))
            # Centring: subtract the expected coefficient of a token with no
            # effect on R, so DG's even-in-U term no longer pays for rarity.
            centring = torch.zeros_like(advantage)
            if gate_mask["dgquad"].any():
                # Second order: l * Var(A) / (4 eta), with Var(A) over the
                # group's samples at each position (for ``group``, the
                # group's reward variance).
                spread = advantage.var(2, unbiased=False, keepdim=True)
                centring = torch.where(gate_mask["dgquad"], surprisal * spread / (4 * temperature), centring)
            if gate_mask["dgexact"].any() or gate_mask["dgprefix"].any():
                # U = R - p_hat is 1 - p_hat or -p_hat. A null token at
                # prefix h succeeds with probability p(h): the prompt's p_hat
                # (``dgexact``) or the exact E[R | h] (``dgprefix``).
                p_hat = rewards.mean(-1)[..., None, None]
                null = lambda success: null_coefficient(success, p_hat, surprisal / temperature)
                centring = torch.where(gate_mask["dgexact"], null(p_hat), centring)
                if gate_mask["dgprefix"].any():
                    prefix = oracle_values(task, logp.exp(), tokens, context)
                    centring = torch.where(gate_mask["dgprefix"], null(prefix), centring)
            coefficient = coefficient - centring
            running_mean = 0.99 * running_mean + 0.01 * rewards.mean((1, 2))
        log_step = step % args.log_every == 0
        if log_step:
            logits.retain_grad()
        # Token-mean loss per run, summed across runs (disjoint parameters).
        loss = -(coefficient * token_logp).sum() / (prompts * samples * length)
        if expected_q is not None:
            # Q-Prop's analytic term sum_a pi(a|s_t) Q(s_t, a), ungated, over
            # every sampled state; the policy is prefix-independent, so the
            # sum over samples folds into ``expected_q``.
            loss = loss - (logp.exp() * expected_q).sum() / (prompts * samples * length)
        if log_step:
            with torch.no_grad():
                probs = logp.exp()
                filler_probs = probs[:, :, : task.filler]
                coherent_mass = filler_probs.gather(-1, coherent).sum(-1)
                if not has_answer:
                    success = filler_probs.gather(-1, target[..., None]).squeeze(-1).mean(-1)
                else:
                    success = probs[:, :, -1].gather(-1, target[..., None]).squeeze(-1)
                    if task.causal:
                        causal_probs = filler_probs[:, :, causal_positions]
                        success = success * causal_probs.gather(-1, causal_target[..., None]).squeeze(-1).prod(-1)
                    if task.junk_penalty:
                        success = success * coherent_mass[..., noncausal].mean(-1)
                mixed = (rewards.amax(-1) > rewards.amin(-1)).float().mean(1)
                history.append(
                    {
                        "step": step,
                        "expected_success": success.mean(1).tolist(),
                        "filler_entropy": entropy(logp[:, :, : task.filler][..., noncausal, :]).mean((1, 2)).tolist(),
                        "coherent_mass": coherent_mass[..., noncausal].mean((1, 2)).tolist(),
                        "gate_mean": gate.mean((1, 2, 3)).tolist(),
                        # NaN when a run has no tokens of that sign this step.
                        **{
                            f"gate_{name}_mean": ((gate * side).sum((1, 2, 3)) / side.sum((1, 2, 3))).tolist()
                            for name, side in (("pos", (advantage > 0).float()), ("neg", (advantage < 0).float()))
                        },
                        "filler_surprisal": surprisal[..., : task.filler].mean((1, 2, 3)).tolist(),
                        "mixed_group_fraction": mixed.tolist(),
                        "reward_mean": rewards.mean((1, 2)).tolist(),
                        **noncausal_credit(task, logp.detach(), token_logp.detach(), advantage, coefficient, temperature, noncausal),
                    }
                )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if log_step:
            history[-1]["noncausal_entropy_drift"] = entropy_drift(
                logp.detach(), logits.grad * (prompts * length), task, noncausal
            ).tolist()
        optimizer.step()
    return {"runs": runs, "history": history}


def finite_mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else math.nan


def summarize(result: dict, task: Task, args) -> None:
    runs, history = result["runs"], result["history"]
    estimators = list(dict.fromkeys(e for e, _ in runs))
    keys = ("expected_success", "filler_entropy", "coherent_mass", "gate_pos_mean", "gate_neg_mean")
    checkpoints = sorted({0, len(history) // 3, 2 * len(history) // 3, len(history) - 1})
    print(
        f"task {task.name}, {args.policy} policy, {args.optimizer} lr {args.lr:g}, "
        f"eta {args.eta:g}, {args.seeds} seeds"
    )
    for index in checkpoints:
        row = history[index]
        cells = []
        for estimator in estimators:
            slots = [i for i, (e, _) in enumerate(runs) if e == estimator]
            mean = {k: finite_mean([row[k][i] for i in slots]) for k in keys}
            cells.append(
                f"{estimator} acc {mean['expected_success']:.3f} "
                f"H {mean['filler_entropy']:.2f} coh {mean['coherent_mass']:.3f} "
                f"gate +{mean['gate_pos_mean']:.2f}/-{mean['gate_neg_mean']:.2f}"
            )
        print(f"  step {row['step']:5d} | " + " | ".join(cells))
    # Area under the accuracy curve separates speed from final accuracy.
    auc = []
    for estimator in estimators:
        slots = [i for i, (e, _) in enumerate(runs) if e == estimator]
        values = [sum(row["expected_success"][i] for i in slots) / len(slots) for row in history]
        auc.append(f"{estimator} {sum(values) / len(values):.3f}")
    print("  mean accuracy over training | " + " | ".join(auc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--estimators", default=",".join(ESTIMATORS))
    parser.add_argument("--policy", choices=("tabular", "mlp"), default="tabular")
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--value-noise", type=float, default=0.01)
    parser.add_argument("--oracle-lambda", type=float, default=0.0)
    parser.add_argument("--oracle-value-noise", type=float, default=0.0)
    parser.add_argument("--oracle-shrink", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    task = TASKS[args.task]
    result = simulate(task, args.estimators.split(","), args)
    summarize(result, task, args)
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"args": vars(args), **result}, handle)


if __name__ == "__main__":
    main()
