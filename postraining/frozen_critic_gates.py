"""CPU-only, prompt-clustered diagnostics for an untouched branch test set.

Each state has two independently policy-drawn actions, independent branch
continuations, and a separate policy-reference continuation group. The sign
gate tests centered A against forced-action reward minus policy-reference
reward. Without reference rewards, only pairwise diagnostics are available.
Duplicate action draws contribute centered tests; their pairwise sign is the
uninformative tie score 0.5. All inferential statistics weight prompts equally, then states
equally within prompts. Branch means are noisy observations, not ground truth.
"""

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence

import numpy as np


def _wilson(rewards: np.ndarray) -> tuple[float, float]:
    n = len(rewards)
    p = float(rewards.mean())
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - radius, center + radius


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def _interval(values: list[float | None]) -> list[float] | None:
    # Undefined resamples signal insufficient identifying variation. Do not
    # silently discard them and report an artificially precise interval.
    if not values or any(value is None or not math.isfinite(value) for value in values):
        return None
    return np.quantile(values, [.025, .975]).tolist()


def evaluate_predictions(
    rows: Sequence[Mapping[str, Any]], *, seed: int = 0,
    bootstrap_samples: int = 2000, permutation_samples: int = 2000,
    min_prompts: int = 16,
) -> dict[str, Any]:
    """Evaluate fixed predictions; callers must supply a held-out test split.

    ``advantage`` must use the exact sampling policy's all-action mean.
    This function cannot establish split provenance or distribution identity;
    passing its gate does not authorize an actor run on tuning data.
    """
    if bootstrap_samples < 100 or permutation_samples < 100 or min_prompts < 2:
        raise ValueError("require >=100 resamples and >=2 independent prompts")
    grouped = defaultdict(list)
    seen = set()
    duplicate_actions = 0
    for row in rows:
        prompt = str(row["prompt_id"])
        identity = (prompt, str(row["state_id"]))
        if identity in seen:
            raise ValueError("duplicate prompt/state identity")
        seen.add(identity)
        actions = row["actions"]
        if len(actions) != 2:
            raise ValueError("each state requires exactly two policy-drawn actions")
        parsed = []
        for action in actions:
            q, advantage = float(action["q"]), float(action["advantage"])
            rewards = np.asarray(action["rewards"], dtype=float)
            if not math.isfinite(q) or not math.isfinite(advantage):
                raise ValueError("Q and advantage must be finite")
            if rewards.ndim != 1 or not rewards.size or not np.isin(rewards, [0, 1]).all():
                raise ValueError("branch rewards must be nonempty binary lists")
            parsed.append((q, advantage, rewards))
        if actions[0]["token_id"] == actions[1]["token_id"]:
            duplicate_actions += 1
            if not (math.isclose(parsed[0][0], parsed[1][0], abs_tol=1e-7, rel_tol=1e-5)
                    and math.isclose(parsed[0][1], parsed[1][1], abs_tol=1e-7, rel_tol=1e-5)):
                raise ValueError("identical actions at one state must have identical predictions")
            # Canonicalize tolerated readout rounding: identical actions have
            # exactly zero predicted gap, regardless of separate GEMM rows.
            parsed[1] = (parsed[0][0], parsed[0][1], parsed[1][2])
        # A common state baseline must cancel identically between both arms.
        if not math.isclose(parsed[0][0] - parsed[1][0],
                            parsed[0][1] - parsed[1][1], rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError("action advantages must share the same policy baseline")
        reference = row.get("reference_rewards")
        if reference is not None:
            reference = np.asarray(reference, dtype=float)
            if reference.ndim != 1 or not reference.size or not np.isin(reference, [0, 1]).all():
                raise ValueError("policy reference rewards must be nonempty binary lists")
        grouped[prompt].append((parsed, reference))

    contributions = []
    resolved = 0
    pair_count = 0
    worst_case_ses = []
    reference_pairs = 0
    resolved_centered_actions = 0
    centered_ses = []
    for states in grouped.values():
        xs, ys = [], []
        signs, pair_covariances, pair_variances = [], [], []
        centered_signs = []
        for (first, second), reference in states:
            x0, x1 = first[1], second[1]
            y0, y1 = float(first[2].mean()), float(second[2].mean())
            xs.extend([x0, x1])
            ys.extend([y0, y1])
            dx, dy = x0 - x1, y0 - y1
            signs.append(.5 if dx == 0 or dy == 0 else float(dx * dy > 0))
            pair_covariances.append(dx * dy)
            pair_variances.append(dx * dx)
            interval0, interval1 = _wilson(first[2]), _wilson(second[2])
            if reference is None:
                centered_signs.append(float("nan"))
            else:
                reference_pairs += 1
                reference_mean = float(reference.mean())
                reference_interval = _wilson(reference)
                action_signs = []
                for action, observed, interval in ((first, y0, interval0), (second, y1, interval1)):
                    delta = observed - reference_mean
                    action_signs.append(.5 if action[1] == 0 or delta == 0
                                        else float(action[1] * delta > 0))
                    resolved_centered_actions += int(interval[0] > reference_interval[1]
                                                    or reference_interval[0] > interval[1])
                    centered_ses.append(math.sqrt(.25 / len(action[2]) + .25 / len(reference)))
                centered_signs.append(float(np.mean(action_signs)))
            resolved += int(interval0[0] > interval1[1] or interval1[0] > interval0[1])
            pair_count += 1
            worst_case_ses.append(math.sqrt(.25 / len(first[2]) + .25 / len(second[2])))
        x, y = np.asarray(xs), np.asarray(ys)
        x, y = x - x.mean(), y - y.mean()
        contributions.append([np.mean(signs), np.mean(x * y), np.mean(x * x),
                              np.mean(pair_covariances), np.mean(pair_variances),
                              np.mean(centered_signs)])

    count = len(contributions)
    result: dict[str, Any] = {
        "schema": "frozen_critic_branch_gates/v1", "seed": seed,
        "prompts": count, "pairs": pair_count, "duplicate_action_pairs": duplicate_actions,
        "min_prompts": min_prompts, "bootstrap_samples": bootstrap_samples,
        "permutation_samples": permutation_samples,
        "weighting": "equal prompts, equal states per prompt; identical-action pairs score 0.5 for pairwise sign",
        "status": "inconclusive", "passed": False,
        "sign_agreement": None, "sign_agreement_ci95": None,
        "pairwise_sign_agreement": None, "pairwise_sign_agreement_ci95": None,
        "reference_pairs": reference_pairs,
        "resolved_centered_actions_wilson": resolved_centered_actions,
        "resolved_centered_action_fraction_descriptive": resolved_centered_actions / (2 * reference_pairs) if reference_pairs else None,
        "prompt_fixed_slope": None, "prompt_fixed_slope_ci95": None,
        "state_fixed_slope": None, "state_fixed_slope_ci95": None,
        "arm_swap_pvalue_one_sided": None,
        "resolved_pairs_wilson": resolved,
        "resolved_pair_fraction_descriptive": resolved / pair_count if pair_count else None,
        "power": {
            "median_worst_case_pair_se": float(np.median(worst_case_ses)) if pair_count else None,
            "median_pair_difference_for_80pct_power": 2.801582 * float(np.median(worst_case_ses)) if pair_count else None,
            "continuations_per_arm_for_0_001_se_at_p_half": 500000,
            "median_worst_case_centered_advantage_se": float(np.median(centered_ses)) if centered_ses else None,
            "note": "Normal approximation, independent arms; not power of the aggregate gate.",
        },
        "gate_contract": "Untouched test prompts only; independent policy-reference rewards at every state; positive lower 95% CI for centered-A sign minus 0.5, prompt-fixed slope and state-fixed slope; no tuning-data actor authorization.",
    }
    if not count:
        return result
    data = np.asarray(contributions, dtype=float)
    means = data.mean(0)
    has_reference = reference_pairs == pair_count
    result.update(sign_agreement=float(means[5]) if has_reference else None,
                  pairwise_sign_agreement=float(means[0]),
                  prompt_fixed_slope=_ratio(means[1], means[2]),
                  state_fixed_slope=_ratio(means[3], means[4]))
    rng = np.random.default_rng(seed)
    sign_samples, pair_sign_samples, prompt_samples, state_samples = [], [], [], []
    for _ in range(bootstrap_samples):
        sample = data[rng.integers(count, size=count)].mean(0)
        sign_samples.append(float(sample[5]) if has_reference else None)
        pair_sign_samples.append(float(sample[0]))
        prompt_samples.append(_ratio(sample[1], sample[2]))
        state_samples.append(_ratio(sample[3], sample[4]))
    # A single common arm swap per prompt preserves arbitrary dependencies
    # among that prompt's states; this is a cluster sign-randomization test.
    observed = float(data[:, 3].sum())
    exceedances = 0
    for _ in range(permutation_samples):
        statistic = float((rng.choice([-1, 1], size=count) * data[:, 3]).sum())
        exceedances += int(statistic >= observed)
    result.update(
        sign_agreement_ci95=_interval(sign_samples),
        pairwise_sign_agreement_ci95=_interval(pair_sign_samples),
        prompt_fixed_slope_ci95=_interval(prompt_samples),
        state_fixed_slope_ci95=_interval(state_samples),
        arm_swap_pvalue_one_sided=(1 + exceedances) / (1 + permutation_samples),
    )
    sign_ci, slope_ci = result["sign_agreement_ci95"], result["prompt_fixed_slope_ci95"]
    state_ci = result["state_fixed_slope_ci95"]
    if count >= min_prompts and sign_ci is not None and slope_ci is not None and state_ci is not None:
        passed = sign_ci[0] > .5 and slope_ci[0] > 0 and state_ci[0] > 0
        result.update(status="pass" if passed else "inconclusive", passed=passed)
    return result
