"""Offline learnability study of the direct token-Q critic.

The actor-facing quantity of the noise-free qmean direction is the policy-
centred token value

    A(s_t, a_t) = Q(s_t, a_t) - sum_a pi(a | s_t) Q(s_t, a)
                = u_t . (x(a_t) - E_pi[x])

of the standalone token-Q critic (``TokenQHead``: Q = c + u . x). The state
term c cancels, so the critic's explained variance, which c dominates, says
little about A. This study measures A itself, on think tokens, against a
frozen policy:

* every critic arm is fit, through the trainer's own value-only
  ``update_minibatch``, to the same recorded trajectories
  (``rollout_dataset``), in the same order, from the same initial weights;
* held-out trajectories give, per arm, the within-prompt slope of the
  terminal reward R on the trajectory's summed think-token advantage
  (prompt fixed effects, prompt-cluster bootstrap);
* branch rollouts (``branch_rollouts``) give ground-truth Q gaps
  Q(s, a) - Q(s, a') at sampled think positions; the arm's predicted gap
  u . (x(a) - x(a')) is scored by sign agreement against the gaps the
  branches resolved, next to the policy-prior predictor
  log pi(a) - log pi(a'), which a critic must beat to be more than a
  rarity detector.

A twin arm (``q_twin``) fits TD3/TD7's twin critics: two independently
initialized token-Q heads on one critic trunk, each regressing the same
Monte Carlo target, their losses summed. Every gate scores what the actor
reads, the twins' mean Q and direction, so a twin arm compares with its
single-head counterpart; the heads' own fits and their disagreement are
reported beside it.

Every evaluation also takes a support census of the actor's view of A over the whole vocabulary at every held-out
state: at tokens the policy all but never samples (pi < 1e-4) and at the
policy's ranks 2-8, the advantages the full-vocabulary actor would follow.

The actor is frozen throughout: nothing here updates the policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Pure statistics first: the tests import them without a model stack.

# v2: the token Q's action table is the trunk's readout, so every arm starts
# from the actor's trunk, and checkpoints bind the critic schema. Study v3:
# the critic's Q is centred under the frozen policy, which every fit and
# score reads from the policy's own view of the group.
# Arm v3: an arm may freeze the token Q heads' belief gain.
ARM_SCHEMA = "token_q_critic_study_arm/v3"
STUDY_SCHEMA = "token_q_critic_study/v3"
STUDY_CHECKPOINT_SCHEMA = "token_q_critic_study_checkpoint/v2"
DEFAULT_BOOTSTRAP = 10000
# Think actions in the early-credit slope: fewer than the 33-token think
# floor, so every format-valid row contributes the same count.
EARLY_THINK_ACTIONS = 16
# Fisher test level at which a branch pair's gap counts as resolved.
RESOLVE_ALPHA = 0.05


@dataclass(frozen=True)
class CriticArm:
    """One critic configuration; every arm sees identical data and order.

    ``direction_scale="inverse_dim"`` reproduces critic schema v3's scaled
    direction readout (u = W h / dim) as an Adam-equivalent learning-rate
    change: with zero initialization and no weight decay, Adam's step in W
    is invariant to the 1/dim gradient factor, so u moves dim times less per
    step, exactly as if W's rate were divided by dim. The equivalence is up
    to Adam's epsilon, which the 1/dim-smaller v3 gradients met sooner.

    ``weighting`` is fixed at ``"token"``: every action's squared error
    counts once. The trainer's ``update_minibatch`` has no per-trajectory
    loss weights, so trajectory weighting is not an arm here; the field
    stays because study checkpoints, and the trainer's
    ``--study-critic-init``, bind it.

    ``q_twin`` builds ``SeparateCritic(q_twin=True)``: a second token-Q head,
    constructed after every other module, so the trunk and the first head
    draw the same initialization as the single-head arm's.

    ``train_belief_gain=False`` freezes every head's belief gain g at its
    zero initialization, which is exactly critic schema v6's direction
    u = D f + L b: the arm measures what the gain adds. The trainer accepts
    only a trained-gain study critic, since it trains g.
    """

    name: str
    critic_init: str = "actor"
    head_learning_rate: float = 1e-4
    trunk_learning_rate: float = 2e-5
    muon_learning_rate: float = 4.8333333333333334e-05
    direction_scale: str = "inverse_dim"
    weighting: str = "token"
    context: str = "prompt"
    q_twin: bool = False
    train_belief_gain: bool = True

    def __post_init__(self):
        if self.critic_init != "actor":
            # The token Q's action table is the trunk's readout, which a
            # fresh trunk initializes to zero: Q = c + u . x has no token
            # contrast there, and u and x each get a gradient only through
            # the other, so neither ever leaves zero.
            raise ValueError(
                f"critic init {self.critic_init!r}: a token Q critic's action table is its "
                "trunk's readout, so it starts from the actor's trunk"
            )
        if self.direction_scale not in ("inverse_dim", "unscaled"):
            raise ValueError(f"unknown direction scale {self.direction_scale!r}")
        if self.weighting != "token":
            raise ValueError(f"unsupported weighting {self.weighting!r}; the trainer's value loss is token-weighted")
        if self.context not in ("prompt", "gold"):
            raise ValueError(f"unknown critic context {self.context!r}")
        if min(self.head_learning_rate, self.trunk_learning_rate, self.muon_learning_rate) <= 0:
            raise ValueError("learning rates must be positive")


# The base arm is the actor-trunk critic, the RL warm start's configuration;
# each other single-head arm changes one factor, and the twin arm adds
# TD3/TD7 twin heads to it. Scratch-trunk arms left with the input-latent
# action table: a fresh trunk's readout is zero (CriticArm).
STUDY_ARMS = (
    CriticArm("actor_trunk"),
    CriticArm("unscaled_direction", direction_scale="unscaled"),
    CriticArm("head_lr_1e-3", head_learning_rate=1e-3),
    CriticArm("gold_context", context="gold"),
    CriticArm("frozen_belief_gain", train_belief_gain=False),
    CriticArm("actor_trunk_twin", q_twin=True),
    CriticArm("frozen_belief_gain_twin", train_belief_gain=False, q_twin=True),
)


def select_arms(names: str) -> tuple[CriticArm, ...]:
    by_name = {arm.name: arm for arm in STUDY_ARMS}
    chosen = [name for name in names.split(",") if name]
    unknown = sorted(set(chosen) - set(by_name))
    if unknown:
        raise ValueError(f"unknown arms {unknown}; known: {sorted(by_name)}")
    if len(set(chosen)) != len(chosen):
        raise ValueError("an arm is listed twice")
    return tuple(by_name[name] for name in chosen)


def think_action_mask(tokens: Sequence[int], think_open: int, think_close: int) -> np.ndarray:
    """Every action taken in a think state: after the first opening fence,
    before any closing fence.

    Action i emits completion token i from the state its prefix defines.
    Membership depends on that state alone, never on the token taken there,
    so the action that closes the span (or stops inside it) is included.
    That is what keeps the policy-centred advantage summed over the mask at
    zero mean under the policy: dropping the closing action would leave
    -pi(close | s) A(s, close) behind at every think state, and a critic
    that only learned "closing too early fails" would pass the slope gate
    through it. The answer span is excluded because its states follow the
    close: a reference-reading critic could value it without valuing the
    reasoning.
    """
    tokens = list(tokens)
    mask = np.zeros(len(tokens), dtype=bool)
    if think_open not in tokens:
        return mask
    start = tokens.index(think_open) + 1
    end = tokens.index(think_close, start) + 1 if think_close in tokens[start:] else len(tokens)
    mask[start:end] = True
    return mask


def support_census_summary(sums: np.ndarray) -> dict:
    """Means from ``CriticStudy.support_census`` sums: the advantage the
    actor reads, and how often it is positive, off the policy's support
    (pi < 1e-4), at its ranks 2-8, and at its top token."""
    off, off_positive, off_sum, near, near_positive, near_sum, top, top_sum = (float(v) for v in sums)
    return {
        "offsupport_tokens": off,
        "offsupport_positive_fraction": off_positive / off if off else float("nan"),
        "offsupport_advantage_mean": off_sum / off if off else float("nan"),
        "rank2to8_positive_fraction": near_positive / near if near else float("nan"),
        "rank2to8_advantage_mean": near_sum / near if near else float("nan"),
        "top_advantage_mean": top_sum / top if top else float("nan"),
    }


def explained_variance(prediction: np.ndarray, target: np.ndarray) -> float:
    variance = float(np.var(target))
    if variance <= 1e-12:
        return float("nan")
    return 1.0 - float(np.var(target - prediction)) / variance


def _demean_within(groups: np.ndarray, values: np.ndarray) -> np.ndarray:
    sums = np.bincount(groups, weights=values)
    counts = np.bincount(groups)
    return values - (sums / np.maximum(counts, 1))[groups]


def within_prompt_slope(
    groups: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Slope of y on x with prompt fixed effects and a cluster bootstrap.

    Prompt fixed effects remove everything the prompt alone determines
    (difficulty, the critic's per-prompt offset), so the slope measures
    whether trajectories the critic credited more, against their own
    prompt's siblings, earned more. The bootstrap resamples whole prompts:
    samples of one prompt share a policy state and are not independent.
    Prompts whose rewards all agree contribute nothing to either sum.

    Demeaning is within a prompt, so the slope is a ratio of per-prompt
    sums, and a resample (a repeated prompt counts as a new prompt) sums
    the picked prompts' terms: the bootstrap is exact and vectorized.
    """
    groups = np.asarray(groups)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not (groups.shape == x.shape == y.shape) or groups.ndim != 1:
        raise ValueError("groups, x and y must be matching vectors")
    _, groups = np.unique(groups, return_inverse=True)
    count = int(groups.max()) + 1 if groups.size else 0
    x_within = _demean_within(groups, x)
    y_within = _demean_within(groups, y)
    sxx = np.bincount(groups, weights=x_within * x_within, minlength=count)
    sxy = np.bincount(groups, weights=x_within * y_within, minlength=count)
    syy = np.bincount(groups, weights=y_within * y_within, minlength=count)
    total_xx, total_xy, total_yy = float(sxx.sum()), float(sxy.sum()), float(syy.sum())
    slope = total_xy / total_xx if total_xx > 0.0 else float("nan")
    correlation = (
        total_xy / math.sqrt(total_xx * total_yy) if total_xx > 0.0 and total_yy > 0.0 else float("nan")
    )
    informative = int(
        np.count_nonzero(np.bincount(groups, weights=np.abs(y_within), minlength=count) > 1e-12)
    )
    low = high = float("nan")
    if count:
        picked = np.random.default_rng(seed).integers(0, count, size=(bootstrap, count))
        denominators = sxx[picked].sum(1)
        valid = denominators > 0.0
        if valid.any():
            draws = sxy[picked].sum(1)[valid] / denominators[valid]
            low, high = (float(value) for value in np.quantile(draws, (alpha / 2, 1 - alpha / 2)))
    return {
        "slope": slope,
        "slope_ci_low": low,
        "slope_ci_high": high,
        "within_correlation": correlation,
        "prompts": count,
        "informative_prompts": informative,
        "trajectories": int(x.size),
        "alpha": alpha,
    }


def wilson_interval(successes: float, trials: float, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return centre - half, centre + half


def resolved_gap_mask(
    taken_successes: np.ndarray,
    alternative_successes: np.ndarray,
    samples: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Branch pairs whose two success rates differ by Fisher's exact test.

    Each branch arm's success count over ``samples`` binary continuations
    estimates Q. A Wald test on the plug-in variance is anti-conservative
    near 0 and 1, where drill branches mostly live (32/32 against 28/32
    passes Wald; Fisher gives p = 0.11).
    """
    from scipy.stats import fisher_exact

    taken = np.asarray(taken_successes, dtype=np.int64)
    alternative = np.asarray(alternative_successes, dtype=np.int64)
    samples = np.broadcast_to(np.asarray(samples, dtype=np.int64), taken.shape)
    if taken.size and (min(taken.min(), alternative.min()) < 0 or bool((np.maximum(taken, alternative) > samples).any())):
        raise ValueError("success counts must lie in [0, samples]")
    return np.array(
        [
            fisher_exact([[a, n - a], [b, n - b]])[1] < alpha
            for a, b, n in zip(taken.tolist(), alternative.tolist(), samples.tolist(), strict=True)
        ],
        dtype=bool,
    )


def _agreement_scores(predicted_gap: np.ndarray, observed_gap: np.ndarray) -> np.ndarray:
    """Per-pair sign agreement; a tied prediction scores one half."""
    return np.where(
        predicted_gap == 0, 0.5, (np.sign(predicted_gap) == np.sign(observed_gap)).astype(np.float64)
    )


def _cluster_bootstrap_means(scores: np.ndarray, groups: np.ndarray, bootstrap: int, seed: int) -> np.ndarray:
    _, groups = np.unique(groups, return_inverse=True)
    count = int(groups.max()) + 1
    sums = np.bincount(groups, weights=scores, minlength=count)
    sizes = np.bincount(groups, minlength=count).astype(np.float64)
    picked = np.random.default_rng(seed).integers(0, count, size=(bootstrap, count))
    return sums[picked].sum(1) / sizes[picked].sum(1)


def sign_agreement(
    predicted_gap: np.ndarray,
    observed_gap: np.ndarray,
    groups: np.ndarray,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Fraction of pairs whose predicted gap has the observed gap's sign.

    Ties in the prediction score one half, so a constant critic scores
    exactly 0.5 rather than whichever side ties break to. The Wilson
    interval treats pairs as independent; the prompt-cluster bootstrap does
    not, and is the interval the gate reads.
    """
    predicted_gap = np.asarray(predicted_gap, dtype=np.float64)
    observed_gap = np.asarray(observed_gap, dtype=np.float64)
    groups = np.asarray(groups)
    keep = observed_gap != 0
    predicted_gap, observed_gap, groups = predicted_gap[keep], observed_gap[keep], groups[keep]
    scores = _agreement_scores(predicted_gap, observed_gap)
    pairs = int(scores.size)
    if not pairs:
        return {"agreement": float("nan"), "pairs": 0, "prompts": 0,
                "wilson_low": float("nan"), "wilson_high": float("nan"),
                "cluster_ci_low": float("nan"), "cluster_ci_high": float("nan")}
    agreement = float(scores.mean())
    wilson_low, wilson_high = wilson_interval(float(scores.sum()), pairs)
    draws = _cluster_bootstrap_means(scores, groups, bootstrap, seed)
    return {
        "agreement": agreement,
        "pairs": pairs,
        "prompts": int(np.unique(groups).size),
        "wilson_low": wilson_low,
        "wilson_high": wilson_high,
        "cluster_ci_low": float(np.quantile(draws, alpha / 2)),
        "cluster_ci_high": float(np.quantile(draws, 1 - alpha / 2)),
    }


def paired_agreement_difference(
    predicted_gap: np.ndarray,
    baseline_gap: np.ndarray,
    observed_gap: np.ndarray,
    groups: np.ndarray,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Sign agreement of one predictor minus another's on the same pairs.

    The pairs are shared, so the comparison is paired: each pair scores the
    difference of its two agreements, and whole prompts are resampled.
    """
    predicted_gap = np.asarray(predicted_gap, dtype=np.float64)
    baseline_gap = np.asarray(baseline_gap, dtype=np.float64)
    observed_gap = np.asarray(observed_gap, dtype=np.float64)
    groups = np.asarray(groups)
    keep = observed_gap != 0
    differences = (
        _agreement_scores(predicted_gap[keep], observed_gap[keep])
        - _agreement_scores(baseline_gap[keep], observed_gap[keep])
    )
    if not differences.size:
        return {"difference": float("nan"), "pairs": 0,
                "cluster_ci_low": float("nan"), "cluster_ci_high": float("nan")}
    draws = _cluster_bootstrap_means(differences, groups[keep], bootstrap, seed)
    return {
        "difference": float(differences.mean()),
        "pairs": int(differences.size),
        "cluster_ci_low": float(np.quantile(draws, alpha / 2)),
        "cluster_ci_high": float(np.quantile(draws, 1 - alpha / 2)),
    }


def gate_verdicts(slope: dict, branch: dict | None, min_agreement: float = 0.6) -> dict[str, bool | None]:
    """The study's pre-registered held-out gates for one arm.

    * slope: the prompt-fixed-effects slope of R on the summed think-token
      advantage is positive with its cluster CI above zero;
    * branch: sign agreement with resolved branch gaps is at least
      ``min_agreement`` with its cluster CI above one half, and its paired
      excess over the policy-prior predictor's agreement on the same pairs
      has a cluster CI above zero.
    None marks a gate without the data to decide it: no prompt whose
    rewards disagree, no branch data, or a third condition that is not met
    on a branch set where even the split-half Monte-Carlo oracle cannot
    beat the prior (unpowered, not failed). A critic that beats the prior
    passes whatever the oracle shows: the oracle is scored on half the
    continuations, so it is the weaker test. A critic failing the first two
    conditions fails. A critic whose summed advantage does not vary within
    prompts has an undefined slope and fails.
    """
    if not slope["informative_prompts"]:
        slope_pass = None
    else:
        slope_pass = bool(
            math.isfinite(slope["slope"]) and math.isfinite(slope["slope_ci_low"])
            and slope["slope"] > 0 and slope["slope_ci_low"] > 0
        )
    if branch is None or not branch["critic"]["pairs"]:
        branch_pass = None
    else:
        critic = branch["critic"]
        agrees = critic["agreement"] >= min_agreement and critic["cluster_ci_low"] > 0.5
        if not agrees:
            branch_pass = False
        elif branch["critic_minus_prior"]["cluster_ci_low"] > 0:
            branch_pass = True
        else:
            branch_pass = False if branch["oracle_beats_prior"] else None
    return {"slope": slope_pass, "branch": branch_pass}


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")


def twin_statistics(
    member_q_all: np.ndarray,
    target_all: np.ndarray,
    member_q_think: np.ndarray,
    target_think: np.ndarray,
    advantage_gap_think: np.ndarray,
    advantage_gap_branch: np.ndarray,
    pair_gap_difference: np.ndarray,
) -> dict:
    """Each twin head's own fit, and how far the two heads disagree.

    ``member_q_*`` are (heads, actions) Q of the taken token, the targets
    each action's Monte Carlo return. Disagreement is the RMS of head 0
    minus head 1: of the taken token's Q (all and think actions), of its
    policy-centred advantage A (think actions, which the slope gate sums,
    and branch states), and of the branch pairs' predicted gap
    u . (x(a) - x(a')). Not gating: the gates score the heads' mean.
    """
    if member_q_all.shape[0] != 2 or member_q_think.shape[0] != 2:
        raise ValueError("twin statistics need exactly two heads")
    return {
        "ev_all_actions": [explained_variance(q, target_all) for q in member_q_all],
        "ev_think_actions": [explained_variance(q, target_think) for q in member_q_think],
        "q_gap_rms_all_actions": _rms(member_q_all[0] - member_q_all[1]),
        "q_gap_rms_think_actions": _rms(member_q_think[0] - member_q_think[1]),
        "advantage_gap_rms_think_actions": _rms(advantage_gap_think),
        "advantage_gap_rms_branch_states": _rms(advantage_gap_branch),
        "branch_pair_gap_difference_rms": _rms(pair_gap_difference),
        "branch_pairs": int(pair_gap_difference.size),
    }


def strict_json(value) -> str:
    """JSON with undefined statistics (NaN, inf) as null, which every JSON
    reader parses; bare NaN is not JSON."""
    def finite(item):
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, dict):
            return {key: finite(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(entry) for entry in item]
        return item

    return json.dumps(finite(value), sort_keys=True, allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def update_budget(args, pools: int) -> int:
    """Explicit update caps override historical epoch budgets."""
    budget = getattr(args, "max_updates", None)
    budget = args.epochs * pools if budget is None else budget
    if pools < 1 or budget < 1:
        raise ValueError("training needs a positive pool count and update budget")
    return budget


def plateau_futility(evaluations: Sequence[dict], *, floor: int = 256) -> bool:
    """Operational futility, not proof of optimizer convergence.

    At 60-update boundaries compare three consecutive nonoverlapping windows.
    Negative gains count as futile; missing or nonfinite evidence does not.
    Gates never participate in this stopping rule.
    """
    if not evaluations or evaluations[-1]["step"] < floor or evaluations[-1]["step"] % 60:
        return False
    by_step = {row["step"]: row for row in evaluations}
    end = evaluations[-1]["step"]
    rows = [by_step.get(end - offset) for offset in (180, 120, 60, 0)]
    if any(row is None or row.get("branch") is None for row in rows):
        return False
    ev = [row["ev_think_actions"] for row in rows]
    agreement = [row["branch"]["critic"]["agreement"] for row in rows]
    if not all(value is not None and math.isfinite(value) for value in (*ev, *agreement)):
        return False
    return all(right - left < 0.005 for left, right in zip(ev, ev[1:])) and all(
        right - left < 0.01 for left, right in zip(agreement, agreement[1:])
    )


def checkpoint_payload(study, arm, critic, optimizers, *, step, epoch, order,
                       next_pool, order_rng, evaluations, stop_reason):
    """Full independent critic state with exact next-pool cursor and bindings."""
    from postraining.vapo.schemas import critic_schema_for

    torch = study.torch
    return {
        "schema": STUDY_CHECKPOINT_SCHEMA,
        # The critic's own contract: the study schema alone does not say
        # which action table a head's directions were fit against.
        "critic_schema": critic_schema_for(True, twin_q=arm.q_twin),
        "critic": critic.state_dict(),
        "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
        "optimizer_groups": {
            name: [{key: value for key, value in group.items() if key != "params"}
                   for group in opt.param_groups]
            for name, opt in optimizers.items()
        },
        "arm": asdict(arm),
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(study.args).items()},
        "bindings": getattr(study, "checkpoint_bindings", {}),
        **getattr(study, "checkpoint_bindings", {}),
        "checkpoint_sha256": getattr(study, "checkpoint_bindings", {}).get("checkpoint_sha256"),
        "final_evaluation": evaluations[-1] if evaluations else None,
        "frozen_actor_fingerprint": getattr(study, "frozen_actor_fingerprint", None),
        "step": step, "epoch": epoch, "pool_order": list(map(int, order)),
        "next_pool": next_pool, "order_rng": order_rng.bit_generator.state,
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "evaluations": evaluations, "stop_reason": stop_reason,
    }


def atomic_checkpoint(torch, payload: dict, destination: Path) -> None:
    """Durably replace the recovery state after a complete write."""
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


# --- model stack -------------------------------------------------------------


def _model_stack():
    # Deferred: train_gpt patches must load before the trainer's modules, and
    # the statistics above must stay importable without them.
    import torch

    import train_gpt as baseline  # noqa: F401
    from postraining import (
        latent_rollout,
        rollout_dataset,
        train_latent_vapo,
        value_model,
    )
    from postraining.latent_thought import LatentThoughtModel
    from postraining.model_io import copy_trunk, fresh_trunk, load_model

    return torch, latent_rollout, rollout_dataset, train_latent_vapo, value_model, LatentThoughtModel, copy_trunk, fresh_trunk, load_model


def critic_optimizers(trainer, critic, arm: CriticArm, fused: bool) -> dict:
    """The critic half of the trainer's ``build_optimizers`` muon layout.

    ``build_optimizers`` always builds the actor's optimizers too, and
    ``Muon`` refuses the empty parameter list of a frozen actor, so the
    study builds the critic's three directly, on the trainer's own Muon
    partition: ``critic_muon`` over the trunk's block matrices,
    ``critic_action`` over the token-Q head(s) at the head rate, and
    ``critic`` AdamW over everything else at the trunk rate. No actor
    optimizer exists to step; the frozen-actor fingerprint also checks that
    no in-place operation changes the policy.
    """
    import torch

    muon = trainer.muon_matrix_parameters(critic.trunk.blocks, set())
    heads = [
        parameter
        for head in (critic.action_head, critic.q_head, critic.q_twin, critic.post_action_q_head)
        if head is not None
        for parameter in head.parameters()
        if parameter.requires_grad
    ]
    excluded = {id(parameter) for parameter in (*muon, *heads)}
    rest = [
        parameter for parameter in critic.parameters()
        if parameter.requires_grad and id(parameter) not in excluded
    ]
    optimizers = {
        "critic": torch.optim.AdamW(rest, lr=arm.trunk_learning_rate, weight_decay=0.0, fused=fused),
        "critic_action": torch.optim.AdamW(heads, lr=arm.head_learning_rate, weight_decay=0.0, fused=fused),
        "critic_muon": trainer.Muon(muon, lr=arm.muon_learning_rate, weight_decay=0.0),
    }
    registered = [
        id(parameter) for optimizer in optimizers.values()
        for group in optimizer.param_groups for parameter in group["params"]
    ]
    if len(registered) != len(set(registered)) or set(registered) != {
        id(parameter) for parameter in critic.parameters() if parameter.requires_grad
    }:
        raise AssertionError("critic parameters must partition across optimizers")
    if arm.direction_scale == "inverse_dim":
        scale_direction_learning_rate(optimizers, critic, 1.0 / critic.q_head.direction.out_features, fused=fused)
    return optimizers


def scale_direction_learning_rate(optimizers: dict, critic, factor: float, fused: bool) -> None:
    """Give every token-Q head's direction readout one group at ``factor``
    times the head rate, before any step has built optimizer state.

    The group carries the ``token_q_direction`` tag of the group the
    trainer's ``build_optimizers(critic_q_direction_learning_rate=...)``
    forms over the same readouts.
    """
    import torch

    head = optimizers["critic_action"]
    if any(head.state.values()):
        raise RuntimeError("the direction rate must be split before the first step")
    (group,) = head.param_groups
    directions = [weight for q_head in critic.q_heads for weight in q_head.direction_weights]
    direction_ids = {id(direction) for direction in directions}
    rest = [parameter for parameter in group["params"] if id(parameter) not in direction_ids]
    if len(rest) != len(group["params"]) - len(directions):
        raise RuntimeError("the head optimizer does not hold every direction readout")
    optimizers["critic_action"] = torch.optim.AdamW(
        [
            {"params": rest, "lr": group["lr"]},
            {"params": directions, "lr": group["lr"] * factor, "token_q_direction": True},
        ],
        weight_decay=0.0,
        fused=fused,
    )


class Study:
    """The frozen actor, tokenizer and recorded data shared by every arm.

    ``load_study`` builds one from the CLI; tests inject small models.
    """

    def __init__(self, args, wrapper, tokenizer, train_pools, heldout_pools, branches=None):
        (torch, self.latent_rollout, self.rollout_dataset, self.trainer, self.value_model,
         _, self.copy_trunk, self.fresh_trunk, _) = _model_stack()
        self.torch = torch
        self.args = args
        self.wrapper = wrapper
        self.backbone = wrapper.backbone
        self.device = next(wrapper.parameters()).device
        if wrapper.hidden_carry or wrapper.memory_carry:
            raise ValueError("the study's recorded streams belong to a token policy")
        self.tokenizer = tokenizer
        self.think_open = tokenizer.think_open_id
        self.think_close = tokenizer.think_close_id
        self.train_pools = train_pools
        self.heldout_pools = heldout_pools
        train_prompts = {group.prompt_sha256 for pool in train_pools for group in pool}
        heldout_prompts = {group.prompt_sha256 for pool in heldout_pools for group in pool}
        if train_prompts & heldout_prompts:
            raise ValueError(f"{len(train_prompts & heldout_prompts)} prompts appear in both train and held-out data")
        # Pools built with different seeds can exhaust different cells; a
        # held-out cell the critic never trained on would test transfer,
        # not the fit the gates are about.
        unseen = {group.module for pool in heldout_pools for group in pool} - {
            group.module for pool in train_pools for group in pool
        }
        if unseen:
            raise ValueError(f"held-out cells {sorted(map(str, unseen))} have no training trajectories")
        self.branches = branches

    def actor_fingerprint(self) -> list[tuple[float, float]]:
        """Per-parameter sums and square sums: equal before and after an arm
        exactly when no actor step or in-place write happened, short of an
        adversarial change preserving both."""
        with self.torch.no_grad():
            return [
                (float(parameter.double().sum()), float(parameter.double().square().sum()))
                for parameter in self.wrapper.parameters()
            ]

    def autocast(self):
        # The trainer's replay and update numerics: bf16 autocast on CUDA.
        return self.torch.autocast(
            device_type=self.device.type, dtype=self.torch.bfloat16,
            enabled=self.device.type == "cuda",
        )

    def gold_suffix(self, truth: str) -> list[int]:
        return self.tokenizer.encode(f"\n\nReference answer: {truth}")

    def critic_batch(self, group, arm: CriticArm):
        prompt = list(group.prompt_tokens)
        if arm.context == "gold":
            prompt += self.gold_suffix(group.ground_truth)
        return self.rollout_dataset.group_batch(prompt, group.completions, group.rewards, self.args.replay_bucket)

    def build_critic(self, arm: CriticArm):
        torch = self.torch
        cuda = self.device.type == "cuda"
        with torch.random.fork_rng(devices=[self.device] if cuda else []):
            torch.manual_seed(self.args.seed + self.trainer.CRITIC_INIT_SEED_OFFSET)
            trunk = (
                self.copy_trunk(self.backbone, self.device)
                if arm.critic_init == "actor"
                else self.fresh_trunk(self.backbone, self.device)
            )
            # The trainer's construction (combiner MLP included, unused by a
            # token policy but drawn from the same RNG stream), so a seed
            # gives the q-head the trainer would build.
            critic = self.value_model.SeparateCritic(
                trunk,
                mlp_hidden=self.args.combined_mlp_hidden,
                num_blocks=self.args.combined_mlp_blocks,
                q_head_hidden=self.args.q_head_hidden,
                q_head_reads_actor_belief=False,
                q_twin=arm.q_twin,
            ).to(self.device)
        if not arm.train_belief_gain:
            for head in critic.q_heads:
                head.belief_gain.requires_grad_(False)
        critic.eval()
        if cuda and self.args.compile:
            # The trainer's --compile-replay settings. Its rebinding of
            # ``estimates`` misses a token-Q critic, whose update and scoring
            # read its trunk pass ``token_q_parts`` directly, so that is the
            # one compiled here, as the trainer does. A KDA trunk runs the
            # eager FLA chunk kernel, so its graph breaks around it.
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            critic.token_q_parts = torch.compile(
                critic.token_q_parts,
                mode="default",
                fullgraph="_kda_" not in self.backbone.architecture,
                dynamic=True,
            )
        return critic, critic_optimizers(self.trainer, critic, arm, fused=cuda)

    def actor_batch(self, group):
        """The policy's view of a group: its own prompt, never the gold one."""
        return self.rollout_dataset.group_batch(
            group.prompt_tokens, group.completions, group.rewards, self.args.replay_bucket)

    def action_offsets(self, critic, group, critic_batch):
        """The taken token's offset from the frozen policy's mean action row
        at each of ``critic_batch``'s action slots, from the policy's own
        view of the group."""
        return self.latent_rollout.policy_action_offsets_at_actions(
            self.wrapper, self.actor_batch(group).to(self.device), critic.action_latents(),
            self.args.vocab_chunk, critic_batch)

    def fit_pool(self, critic, optimizers, pool, arm: CriticArm) -> dict:
        lr, trainer = self.latent_rollout, self.trainer
        packed = lr.pack_rollout_groups_for_replay([self.critic_batch(group, arm) for group in pool])
        # The policy's view, which centres the critic's Q: the same rows,
        # without the gold context.
        policy_batch = (
            lr.pack_rollout_groups_for_replay([self.actor_batch(group) for group in pool])
            .to(self.device) if arm.context == "gold" else None
        )
        plan = lr.build_replay_plan(
            packed, self.args.replay_max_trajectories, self.args.replay_attention_budget,
            self.args.replay_bucket, self.args.replay_slot_budget,
        )
        batch = packed.to(self.device)
        trainer.zero_optimizers(optimizers, "critic")
        with self.autocast():
            metrics = trainer.update_minibatch(
                self.wrapper, critic, batch, optimizers,
                value_only=True, critic_step=False, actor_step=False,
                replay_max_trajectories=self.args.replay_max_trajectories,
                replay_attention_budget=self.args.replay_attention_budget,
                replay_bucket=self.args.replay_bucket,
                replay_slot_budget=self.args.replay_slot_budget,
                replay_plan=plan.to(self.device),
                mac_vocab_chunk=self.args.vocab_chunk,
                policy_batch=policy_batch,
            )
        trainer.step_optimizers(optimizers, "critic")
        # The trainer's value_loss is the squared-error sum over the action
        # count, summed over a twin critic's heads: each regresses the same
        # Monte Carlo target on its own, and TD7 adds their losses.
        return {
            **metrics,
            **trainer.twin_dashboard([metrics]),
            "objective_loss": float(metrics["value_loss"]),
        }

    def score_group(self, critic, group, arm: CriticArm) -> dict[str, np.ndarray]:
        """Per-action Q, MC target, policy-centred A, and direction for one
        held-out prompt group, in row-major completion order."""
        torch, lr = self.torch, self.latent_rollout
        actor_batch = self.actor_batch(group).to(self.device)
        critic_batch = self.critic_batch(group, arm).to(self.device)
        twin = critic.q_twin is not None
        with torch.no_grad(), self.autocast():
            # One trunk pass; the mean is exactly what ``token_values``
            # returns, the actor's view.
            member_q, member_directions = critic.token_value_members(
                critic_batch, self.action_offsets(critic, group, critic_batch))
            q_taken, directions = member_q.mean(0), member_directions.mean(0)
            critic_index = lr.slot_index(critic_batch.action_mask.bool())
            q_taken = lr.compact_slots(q_taken, critic_index).float()
            directions = lr.compact_slots(directions, critic_index).float()
            if twin:
                member_q = torch.stack([lr.compact_slots(q, critic_index).float() for q in member_q])
                # A is linear in u, so the heads' advantage gap A1 - A2 is
                # the advantage under their direction gap u1 - u2.
                direction_gap = lr.compact_slots(
                    member_directions[0] - member_directions[1], critic_index
                ).float()
            beliefs, inputs = lr.replay_head_inputs(self.wrapper, actor_batch)
            actor_index = lr.slot_index(actor_batch.emit_mask.bool())
            if actor_index.numel() != critic_index.numel():
                raise RuntimeError("actor and critic streams disagree on the action count")
            targets = lr.compact_next_slots(actor_batch.token_ids, actor_index)
            latents = critic.action_latents()

            def centred(directions):
                # The actor's own readout of A, as the trainer forms it, over
                # chunks of slots to bound the vocabulary-wide logits.
                advantages = []
                for start in range(0, actor_index.numel(), self.args.vocab_chunk):
                    index = actor_index[start : start + self.args.vocab_chunk]
                    chunk = slice(start, start + index.numel())
                    _, _, chunk_advantages, _ = lr.compact_emit_token_logprobs_entropy_and_advantages(
                        self.wrapper,
                        lr.compact_slots(inputs, index),
                        lr.compact_slots(beliefs, index),
                        targets[chunk],
                        directions[chunk],
                        latents,
                    )
                    advantages.append(chunk_advantages.float())
                return torch.cat(advantages).cpu().numpy()

            scored = {
                "q_taken": q_taken.cpu().numpy(),
                "advantage": centred(directions),
                "support_census": self.support_census(
                    lr.compact_slots(beliefs, actor_index), directions, latents),
                "directions": directions,
                "latents": latents,
                "targets": targets,
            }
            if twin:
                scored["member_q_taken"] = member_q.cpu().numpy()
                scored["advantage_gap"] = centred(direction_gap)
                scored["direction_gap"] = direction_gap
        return scored

    def support_census(self, beliefs, directions, latents) -> np.ndarray:
        """``mean_actor_critic.support_census`` sums of the actor's view of A
        over the whole vocabulary at each state."""
        torch = self.torch
        from postraining.mean_actor_critic import SUPPORT_CENSUS_FIELDS, support_census
        from postraining.nano_backbone import rational_softcap

        backbone = self.wrapper.backbone
        sums = torch.zeros(len(SUPPORT_CENSUS_FIELDS), dtype=torch.float64, device=self.device)
        for start in range(0, directions.size(0), self.args.vocab_chunk):
            chunk = slice(start, start + self.args.vocab_chunk)
            # The readout under the caller's autocast, as the actor samples.
            raw = backbone._raw_logits(beliefs[chunk])
            with torch.autocast(device_type=self.device.type, enabled=False):
                probabilities = rational_softcap(raw.float(), backbone.logit_softcap).softmax(-1)
                scores = directions[chunk].float() @ latents.float().T
                advantages = scores - (probabilities * scores).sum(-1, keepdim=True)
                sums += support_census(probabilities, advantages)
        return sums.cpu().numpy()

    def evaluate(self, critic, arm: CriticArm, step: int) -> dict:
        """Held-out gates for one arm at its current weights."""
        rows_reward, rows_group, rows_think_sum, rows_early_sum = [], [], [], []
        q_all, mc_all, q_think, mc_think, a_think = [], [], [], [], []
        branch_predicted, branch_keys = [], []
        twin = critic.q_twin is not None
        # Twin heads only: each head's Q of the taken token, the heads'
        # centred-advantage gap A1 - A2, and their branch pair-gap gap.
        member_all, member_think, gap_think, gap_branch_state, gap_branch_pair = [], [], [], [], []
        branch_lookup = self._branch_lookup()
        from postraining.mean_actor_critic import SUPPORT_CENSUS_FIELDS

        census = np.zeros(len(SUPPORT_CENSUS_FIELDS))
        group_id = 0
        for pool in self.heldout_pools:
            for group in pool:
                scored = self.score_group(critic, group, arm)
                census += scored["support_census"]
                offset = 0
                for sample, tokens in enumerate(group.completions):
                    span = slice(offset, offset + len(tokens))
                    offset += len(tokens)
                    think = think_action_mask(tokens, self.think_open, self.think_close)
                    reward = group.rewards[sample]
                    q_row = scored["q_taken"][span]
                    a_row = scored["advantage"][span]
                    q_all.append(q_row)
                    mc_all.append(np.full(len(tokens), reward))
                    q_think.append(q_row[think])
                    mc_think.append(np.full(int(think.sum()), reward))
                    a_think.append(a_row[think])
                    rows_reward.append(reward)
                    rows_group.append(group_id)
                    rows_think_sum.append(float(a_row[think].sum()))
                    rows_early_sum.append(float(a_row[np.flatnonzero(think)[:EARLY_THINK_ACTIONS]].sum()))
                    if twin:
                        member_row = scored["member_q_taken"][:, span]
                        member_all.append(member_row)
                        member_think.append(member_row[:, think])
                        gap_think.append(scored["advantage_gap"][span][think])
                    for position, alternative, record in branch_lookup.get((group.pool, group.group, sample), ()):
                        direction = scored["directions"][span.start + position]
                        latents = scored["latents"]
                        taken = int(tokens[position])
                        contrast = latents[taken] - latents[alternative]
                        gap = float(direction @ contrast)
                        branch_predicted.append(gap)
                        branch_keys.append(record)
                        if twin:
                            gap_branch_state.append(float(scored["advantage_gap"][span.start + position]))
                            gap_branch_pair.append(float(scored["direction_gap"][span.start + position] @ contrast))
                group_id += 1
        groups = np.array(rows_group)
        reward = np.array(rows_reward)
        think_sum = np.array(rows_think_sum)
        a_think = np.concatenate(a_think) if a_think else np.zeros(0)
        if self.branches is not None and len(branch_keys) != len(self.branches["pool"]):
            raise RuntimeError(
                f"{len(self.branches['pool']) - len(branch_keys)} branch records match no held-out trajectory"
            )
        statistics = {"bootstrap": self.args.bootstrap, "seed": self.args.seed, "alpha": self.args.ci_alpha}
        result = {
            "step": step,
            "arm": arm.name,
            "ev_all_actions": explained_variance(np.concatenate(q_all), np.concatenate(mc_all)),
            "ev_think_actions": explained_variance(np.concatenate(q_think), np.concatenate(mc_think)),
            "think_advantage_std": float(a_think.std()) if a_think.size else float("nan"),
            "think_advantage_abs_mean": float(np.abs(a_think).mean()) if a_think.size else float("nan"),
            "think_actions": int(a_think.size),
            "slope": within_prompt_slope(groups, think_sum, reward, **statistics),
            # Not gating. The summed advantage telescopes toward the last
            # think tokens, where the working has already produced the
            # result; a critic that only checks that result (the gold arm
            # can) passes the gate without crediting the steps. The first
            # EARLY_THINK_ACTIONS think actions are a state-determined set
            # (zero-mean advantage under the policy, like the full mask)
            # that comes before most results exist.
            "slope_early_think": within_prompt_slope(groups, np.array(rows_early_sum), reward, **statistics),
        }
        result["branch"] = (
            self._branch_metrics(np.array(branch_predicted), branch_keys, statistics)
            if self.branches is not None else None
        )
        self.last_branch_predictions = [
            {"arm": arm.name, "step": step, "record": int(record),
             "prompt_sha256": self.branches["prompt_sha256"][record],
             "predicted_gap": float(gap),
             "taken_rewards": self.branches["taken_rewards"][record],
             "alternative_rewards": self.branches["alternative_rewards"][record],
             "logprior_gap": float(self.branches["taken_logprob"][record]
                                   - self.branches["alternative_logprob"][record])}
            for gap, record in zip(branch_predicted, branch_keys, strict=True)
        ] if self.branches is not None else []
        result["gates"] = gate_verdicts(result["slope"], result["branch"], self.args.min_branch_agreement)
        result["support_census"] = support_census_summary(census)
        if twin:
            result["twin"] = twin_statistics(
                np.concatenate(member_all, axis=1), np.concatenate(mc_all),
                np.concatenate(member_think, axis=1), np.concatenate(mc_think),
                np.concatenate(gap_think), np.array(gap_branch_state), np.array(gap_branch_pair),
            )
        return result

    def _branch_lookup(self) -> dict:
        if self.branches is None:
            return {}
        lookup: dict = {}
        columns = self.branches
        for record in range(len(columns["pool"])):
            key = (columns["pool"][record], columns["group"][record], columns["sample"][record])
            lookup.setdefault(key, []).append((columns["position"][record], columns["alternative_token"][record], record))
        return lookup

    def _branch_metrics(self, predicted: np.ndarray, records: list[int], statistics: dict) -> dict:
        columns = self.branches
        index = np.array(records, dtype=np.int64)
        taken = np.array([columns["taken_rewards"][record] for record in index], dtype=np.float64).reshape(index.size, -1)
        alternative = np.array(
            [columns["alternative_rewards"][record] for record in index], dtype=np.float64
        ).reshape(taken.shape)
        if not np.isin(np.concatenate((taken, alternative)), (0.0, 1.0)).all():
            raise ValueError("branch gates need binary continuation rewards")
        samples = taken.shape[1]
        groups = np.array(columns["prompt_sha256"], dtype=object)[index]
        prior = np.array(columns["taken_logprob"])[index] - np.array(columns["alternative_logprob"])[index]
        # Resolution selects the pairs every arm is scored on; it does not
        # depend on the arm, so it takes no share of the arms' family alpha.
        resolved = resolved_gap_mask(taken.sum(1), alternative.sum(1), samples, RESOLVE_ALPHA)
        observed = taken.mean(1) - alternative.mean(1)
        # Power of the prior condition: one half of each pair's
        # continuations predicts the other half's resolved gap. If even
        # this Monte-Carlo oracle cannot beat the prior, a critic cannot be
        # shown to, and the condition is unpowered rather than failed.
        half = samples // 2
        oracle = taken[:, :half].mean(1) - alternative[:, :half].mean(1)
        held = taken[:, half:].mean(1) - alternative[:, half:].mean(1)
        held_resolved = resolved_gap_mask(
            taken[:, half:].sum(1), alternative[:, half:].sum(1), samples - half, RESOLVE_ALPHA
        )
        oracle_minus_prior = paired_agreement_difference(
            oracle[held_resolved], prior[held_resolved], held[held_resolved], groups[held_resolved], **statistics,
        )
        likelier_alternative = resolved & (prior < 0)
        return {
            "pairs_total": int(index.size),
            "pairs_resolved": int(resolved.sum()),
            "critic": sign_agreement(predicted[resolved], observed[resolved], groups[resolved], **statistics),
            "policy_prior": sign_agreement(prior[resolved], observed[resolved], groups[resolved], **statistics),
            "critic_minus_prior": paired_agreement_difference(
                predicted[resolved], prior[resolved], observed[resolved], groups[resolved], **statistics,
            ),
            "oracle_minus_prior": oracle_minus_prior,
            "oracle_beats_prior": bool(oracle_minus_prior["cluster_ci_low"] > 0),
            # Agreement where the alternative was the likelier token: a
            # rarity detector scores below one half here.
            "critic_alternative_likelier": sign_agreement(
                predicted[likelier_alternative], observed[likelier_alternative],
                groups[likelier_alternative], **statistics,
            ),
            "critic_all_pairs_correlation": (
                float(np.corrcoef(predicted, observed)[0, 1]) if index.size > 2 and np.std(predicted) > 0 and np.std(observed) > 0 else float("nan")
            ),
        }

    def training_probe(self, critic, arm: CriticArm) -> dict:
        """Fixed first-pool first-eight-group fit diagnostic."""
        predictions, targets = [], []
        with self.torch.no_grad(), self.autocast():
            for group in self.train_pools[0][:8]:
                batch = self.critic_batch(group, arm).to(self.device)
                values, _ = critic.token_values(batch, self.action_offsets(critic, group, batch))
                index = self.latent_rollout.slot_index(batch.action_mask.bool())
                predictions.append(self.latent_rollout.compact_slots(values, index).float().cpu().numpy())
                targets.append(np.concatenate([
                    np.full(len(tokens), reward)
                    for tokens, reward in zip(group.completions, group.rewards, strict=True)
                ]))
        predicted, target = np.concatenate(predictions), np.concatenate(targets)
        return {"actions": int(target.size),
                "q_mse": float(np.mean((predicted - target) ** 2)),
                "q_ev": explained_variance(predicted, target)}

    def probe_values(self, critic, arm: CriticArm) -> np.ndarray:
        """Q of the taken token over the first held-out pool's think actions."""
        values = []
        for group in self.heldout_pools[0][: self.args.probe_groups]:
            scored = self.score_group(critic, group, arm)
            offset = 0
            for tokens in group.completions:
                think = think_action_mask(tokens, self.think_open, self.think_close)
                values.append(np.stack((scored["q_taken"][offset : offset + len(tokens)][think],
                                        scored["advantage"][offset : offset + len(tokens)][think])))
                offset += len(tokens)
        return np.concatenate(values, axis=1)



# What must agree between the train and held-out collections for one critic
# to be fit on one and judged on the other.
CONTRACT_BINDING_KEYS = (
    "reward_schema", "temperature", "top_p", "max_new_tokens", "think_min_tokens",
    "nearby_reward_max", "stop_ids", "trim_multiple",
)


def load_study(args) -> Study:
    """The study over a checkpoint and its recorded datasets."""
    torch, _, rollout_dataset, _, _, LatentThoughtModel, _, _, load_model = _model_stack()
    from postraining.core import load_posttraining_tokenizer
    from pretraining.fresh_lejepa.fresh_lejepa_train import FreshHyperparameters

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    checkpoint_sha256 = sha256_file(args.checkpoint)
    manifests = {}
    for name, directory in (("train", args.train_dataset), ("heldout", args.heldout_dataset)):
        manifest = rollout_dataset.read_manifest(directory)
        binding = manifest["binding"]
        # A collection from a --curriculum-init policy binds that snapshot;
        # this study replays --checkpoint itself.
        if (binding["checkpoint_sha256"] != checkpoint_sha256
                or binding.get("policy_state_sha256") is not None):
            raise ValueError(f"the {name} dataset was not collected from --checkpoint")
        if binding["reasoning_mode"] != "cot":
            raise ValueError(f"the {name} dataset is not a token-policy collection")
        manifests[name] = manifest
    train_binding, heldout_binding = manifests["train"]["binding"], manifests["heldout"]["binding"]
    for key in CONTRACT_BINDING_KEYS:
        if train_binding.get(key) != heldout_binding.get(key):
            raise ValueError(
                f"train and held-out datasets differ in {key}: "
                f"{train_binding.get(key)!r} vs {heldout_binding.get(key)!r}"
            )
        if train_binding.get(key) is None:
            raise ValueError(f"the datasets do not bind {key}")
    if train_binding["temperature"] != 1.0 or train_binding["top_p"] != 1.0:
        # A centres under the temperature-1 policy; trajectories sampled
        # from any other sampler give the summed advantage a nonzero mean.
        raise ValueError("the study needs datasets sampled at temperature 1 and top_p 1")
    branches = None
    if args.branch_dataset is not None:
        import pyarrow.parquet as pq

        branch_manifest = json.loads((args.branch_dataset / "manifest.json").read_text())
        if branch_manifest["rollout_dataset_digest"] != rollout_dataset.dataset_digest(manifests["heldout"]):
            raise ValueError("the branch dataset was not built on the held-out dataset")
        path = args.branch_dataset / branch_manifest["file"]
        if sha256_file(path) != branch_manifest["sha256"]:
            raise ValueError(f"{path} does not match its manifest digest")
        branches = pq.read_table(path).to_pydict()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone = load_model(args.checkpoint, device, payload=payload)
    sft_args = ((payload.get("sft") or {}).get("args") or {}) if isinstance(payload, dict) else {}
    del payload
    if not sft_args.get("think_tokens"):
        raise ValueError("think-token metrics need a think-fence checkpoint")
    backbone.eval()
    wrapper = LatentThoughtModel(backbone).to(device).eval().requires_grad_(False)
    tokenizer = load_posttraining_tokenizer(
        backbone.architecture,
        FreshHyperparameters.tokenizer_path,
        think_tokens=True,
        answer_tokens=bool(sft_args.get("answer_fence")),
        tokenizer_provenance=backbone.model_config.get("tokenizer_provenance"),
    )
    study = Study(
        args,
        wrapper,
        tokenizer,
        [rollout_dataset.read_pool(args.train_dataset, entry) for entry in manifests["train"]["pools"]],
        [rollout_dataset.read_pool(args.heldout_dataset, entry) for entry in manifests["heldout"]["pools"]],
        branches,
    )
    study.train_manifest = manifests["train"]
    study.heldout_manifest = manifests["heldout"]
    return study


def run_arm(study: Study, arm: CriticArm, output: Path) -> dict:
    from torch.utils.tensorboard import SummaryWriter

    args = study.args
    output.mkdir(parents=True, exist_ok=False)
    (output / "arm.json").write_text(json.dumps({"schema": ARM_SCHEMA, **asdict(arm)}, indent=2) + "\n")
    tensorboard = SummaryWriter(output / "tensorboard")
    actor_before = study.actor_fingerprint()
    study.frozen_actor_fingerprint = actor_before
    critic, optimizers = study.build_critic(arm)
    evaluations = []
    budget = update_budget(args, len(study.train_pools))
    order_rng = np.random.default_rng(args.seed)
    order, epoch, next_pool = [], 0, 0
    stop_reason = None
    checkpoint = output / "checkpoint.pt"
    (output / "checkpoints").mkdir()
    saved_step = -1
    with (output / "metrics.jsonl").open("x") as log:
        def emit(record):
            log.write(strict_json(record) + "\n")
            log.flush()

        def evaluate(step):
            evaluation = study.evaluate(critic, arm, step)
            evaluation["training_probe"] = study.training_probe(critic, arm)
            evaluations.append(evaluation)
            emit({"type": "heldout", **evaluation})
            _log_evaluation(tensorboard, evaluation)
            with (output / "branch_predictions.jsonl").open("a") as stream:
                for row in study.last_branch_predictions:
                    stream.write(strict_json(row) + "\n")
            return evaluation

        def save(step):
            nonlocal saved_step
            if saved_step == step:
                return
            if study.actor_fingerprint() != actor_before:
                raise RuntimeError(f"arm {arm.name} changed the frozen actor at step {step}")
            tensorboard.flush()
            snapshot = output / "checkpoints" / f"step_{step:06d}.pt"
            atomic_checkpoint(study.torch, checkpoint_payload(
                study, arm, critic, optimizers, step=step, epoch=epoch,
                order=order, next_pool=next_pool, order_rng=order_rng,
                evaluations=evaluations, stop_reason=stop_reason,
            ), snapshot)
            # Hard links retain immutable trajectory checkpoints without
            # duplicating their storage for the actor's convenient latest path.
            temporary = checkpoint.with_name(f".{checkpoint.name}.{os.getpid()}.link")
            try:
                os.link(snapshot, temporary)
                os.replace(temporary, checkpoint)
            finally:
                temporary.unlink(missing_ok=True)
            saved_step = step

        evaluate(0)
        save(0)
        probe_before = study.probe_values(critic, arm)
        step = 0
        while step < budget:
            # One permutation per epoch from the study seed: every arm visits
            # the pools in the same order.
            order = order_rng.permutation(len(study.train_pools))
            for next_pool, pool_index in enumerate(order, start=1):
                started = time.perf_counter()
                metrics = study.fit_pool(critic, optimizers, study.train_pools[pool_index], arm)
                step += 1
                record = {
                    "type": "fit", "step": step, "epoch": epoch, "pool": int(pool_index),
                    "seconds": time.perf_counter() - started,
                    **{key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))},
                }
                emit(record)
                for key in ("objective_loss", "value_loss", "mc_explained_variance", "explained_variance",
                            "twin/gap_rms"):
                    if key in record:
                        tensorboard.add_scalar(f"fit/{key}", record[key], step)
                if step == 1:
                    probe_after = study.probe_values(critic, arm)
                    change = probe_after - probe_before
                    first = {
                        "type": "first_step_change", "step": 1,
                        "q_abs_change_mean": float(np.abs(change[0]).mean()),
                        "q_change_rms": float(np.sqrt(np.mean(change[0] ** 2))),
                        "advantage_change_rms": float(np.sqrt(np.mean(change[1] ** 2))),
                        "probe_actions": int(change.shape[1]),
                    }
                    emit(first)
                    for key in ("q_abs_change_mean", "q_change_rms", "advantage_change_rms"):
                        tensorboard.add_scalar(f"first_step/{key}", first[key], 1)
                if step == budget:
                    stop_reason = "max_updates"
                if step % args.eval_every == 0:
                    evaluate(step)
                    if step < budget and plateau_futility(evaluations):
                        stop_reason = "heldout_plateau_or_degradation_futility"
                    save(step)
                if stop_reason is not None or step == budget:
                    break
            if stop_reason is not None or step == budget:
                break
            epoch += 1
        if evaluations[-1]["step"] != step:
            evaluate(step)
        stop_reason = stop_reason or "max_updates"
        save(step)
        emit({"type": "termination", "step": step, "max_updates": budget,
              "reason": stop_reason})
    tensorboard.close()
    if study.actor_fingerprint() != actor_before:
        raise RuntimeError(f"arm {arm.name} changed the frozen actor")
    return {"arm": asdict(arm), "steps": step, "max_updates": budget,
            "stop_reason": stop_reason, "critic_checkpoint": str(checkpoint),
            "final": evaluations[-1], "evaluations": evaluations}


def _log_evaluation(tensorboard, evaluation: dict) -> None:
    step = evaluation["step"]
    for key in ("ev_all_actions", "ev_think_actions", "think_advantage_std"):
        tensorboard.add_scalar(f"heldout/{key}", evaluation[key], step)
    for key in ("slope", "slope_ci_low", "within_correlation"):
        tensorboard.add_scalar(f"heldout/think_sum_{key}", evaluation["slope"][key], step)
        tensorboard.add_scalar(f"heldout/early_think_{key}", evaluation["slope_early_think"][key], step)
    if "training_probe" in evaluation:
        for key in ("q_mse", "q_ev"):
            tensorboard.add_scalar(f"train_probe/{key}", evaluation["training_probe"][key], step)
    if evaluation["branch"] is not None:
        tensorboard.add_scalar("heldout/branch_agreement", evaluation["branch"]["critic"]["agreement"], step)
        tensorboard.add_scalar("heldout/branch_prior_agreement", evaluation["branch"]["policy_prior"]["agreement"], step)
    twin = evaluation.get("twin")
    if twin is not None:
        for head, (ev_all, ev_think) in enumerate(zip(twin["ev_all_actions"], twin["ev_think_actions"], strict=True)):
            tensorboard.add_scalar(f"heldout_twin/head{head}_ev_all_actions", ev_all, step)
            tensorboard.add_scalar(f"heldout_twin/head{head}_ev_think_actions", ev_think, step)
        for key in ("q_gap_rms_all_actions", "q_gap_rms_think_actions", "advantage_gap_rms_think_actions",
                    "advantage_gap_rms_branch_states", "branch_pair_gap_difference_rms"):
            tensorboard.add_scalar(f"heldout_twin/{key}", twin[key], step)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--heldout-dataset", type=Path, required=True)
    parser.add_argument("--branch-dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", default=",".join(arm.name for arm in STUDY_ARMS))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-updates", type=int,
                        help="total optimizer-update cap per arm; overrides --epochs")
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--q-head-hidden", type=int, default=1024)
    # The trainer's --combined-mlp-hidden/--combined-mlp-blocks defaults.
    parser.add_argument("--combined-mlp-hidden", type=int, default=2048)
    parser.add_argument("--combined-mlp-blocks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument(
        "--family-alpha", type=float, default=0.05,
        help="error rate across all arms; each arm's intervals use alpha / arms (Bonferroni)",
    )
    parser.add_argument("--min-branch-agreement", type=float, default=0.6)
    parser.add_argument("--probe-groups", type=int, default=8)
    parser.add_argument("--vocab-chunk", type=int, default=4096)
    parser.add_argument("--replay-max-trajectories", type=int, default=128)
    parser.add_argument("--replay-attention-budget", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--replay-bucket", type=int, default=64)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True,
        help="compile the critic as the trainer's --compile-replay does",
    )
    parser.add_argument("--replay-slot-budget", type=int, default=24576)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    from postraining.provenance import capture_source_provenance

    args = build_parser().parse_args(argv)
    arms = select_arms(args.arms)
    if args.max_updates is not None and args.max_updates < 1:
        raise SystemExit("--max-updates must be positive")
    if args.epochs < 1 or args.eval_every < 1:
        raise SystemExit("--epochs and --eval-every must be positive")
    if not 0.0 < args.family_alpha < 1.0:
        raise SystemExit("--family-alpha must lie in (0, 1)")
    # H1 is "some arm passes": one family of len(arms) tests.
    args.ci_alpha = args.family_alpha / len(arms)
    args.output.mkdir(parents=True, exist_ok=False)
    source_provenance = capture_source_provenance(
        args.output / "provenance", Path(__file__).resolve().parents[1]
    )
    study = load_study(args)
    source = Path(__file__).resolve().parent
    manifest = {
        "schema": STUDY_SCHEMA,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "train_dataset_digest": study.rollout_dataset.dataset_digest(study.train_manifest),
        "heldout_dataset_digest": study.rollout_dataset.dataset_digest(study.heldout_manifest),
        "branch_dataset_sha256": None if args.branch_dataset is None else sha256_file(args.branch_dataset / "manifest.json"),
        "arms": [asdict(arm) for arm in arms],
        "source_provenance": source_provenance,
        "source_sha256": {
            name: sha256_file(source / name)
            for name in (
                "critic_study.py", "branch_rollouts.py", "rollout_dataset.py",
                "train_latent_vapo.py", "value_model.py", "latent_rollout.py",
            )
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    study.checkpoint_bindings = {
        key: manifest[key] for key in (
            "checkpoint_sha256", "train_dataset_digest", "heldout_dataset_digest",
            "branch_dataset_sha256", "source_sha256", "source_provenance",
        )
    }
    results = {}
    for arm in arms:
        results[arm.name] = run_arm(study, arm, args.output / arm.name)
        study.torch.cuda.empty_cache()
    summary = {
        name: {"steps": result.get("steps"), "max_updates": result.get("max_updates"),
               "stop_reason": result.get("stop_reason"),
               "critic_checkpoint": result.get("critic_checkpoint"),
               "gates": result["final"]["gates"], "slope": result["final"]["slope"],
               "slope_early_think": result["final"]["slope_early_think"],
               "branch": result["final"]["branch"], "ev_think_actions": result["final"]["ev_think_actions"],
               **({"twin": result["final"]["twin"]} if "twin" in result["final"] else {})}
        for name, result in results.items()
    }
    (args.output / "results.json").write_text(strict_json({"schema": STUDY_SCHEMA, "arms": summary}) + "\n")
    print(strict_json(summary), flush=True)


if __name__ == "__main__":
    main()
