"""Scalar identities for terminal binary-reward DG; never loads a model."""

import json
import math
import random
from pathlib import Path


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    z = math.exp(x)
    return z / (1 + z)


def dg_weight(utility, probability, eta=1.0):
    return utility * sigmoid(-utility * math.log(probability) / eta)


def integrated_weight(q, probability, magnitude=1.0, eta=1.0):
    return magnitude * (
        q - sigmoid(magnitude * math.log(probability) / eta)
    )


def integrated_binary_weight(q, probability, eta=1.0):
    return q * sigmoid(-math.log(probability) / eta)


def binary_optimum(q):
    """KKT optimum of sum q_i log(1+pi_i), for nonnegative q."""
    if max(q) == 0:
        return [1 / len(q)] * len(q), 0.0
    ordered = sorted(q, reverse=True)
    total = 0.0
    for k, value in enumerate(ordered, 1):
        total += value
        multiplier = total / (k + 1)
        if k == len(q) or ordered[k] <= multiplier:
            return [max(v / multiplier - 1, 0.0) for v in q], multiplier
    raise AssertionError("KKT support was not found")


def check_binary_optima():
    rng = random.Random(271828)
    maximum_simplex_error = 0.0
    maximum_active_stationarity_error = 0.0
    maximum_zero_action_gradient = -math.inf
    for _ in range(300):
        q = [rng.random() ** 3 for _ in range(rng.randrange(2, 13))]
        q[-1] = 0.0
        probabilities, multiplier = binary_optimum(q)
        maximum_simplex_error = max(
            maximum_simplex_error, abs(sum(probabilities) - 1)
        )
        for v, p in zip(q, probabilities):
            if p > 0:
                maximum_active_stationarity_error = max(
                    maximum_active_stationarity_error,
                    abs(v / (1 + p) - multiplier),
                )
            else:
                assert v <= multiplier + 1e-14
        probe = [rng.random() for _ in q]
        probe = [p / sum(probe) for p in probe]
        weights = [integrated_binary_weight(v, p) for v, p in zip(q, probe)]
        mean_weight = sum(p * w for p, w in zip(probe, weights))
        zero_gradient = probe[-1] * (weights[-1] - mean_weight)
        maximum_zero_action_gradient = max(maximum_zero_action_gradient, zero_gradient)
        assert zero_gradient <= 0
    return {
        "cases": 300,
        "maximum_simplex_error": maximum_simplex_error,
        "maximum_active_stationarity_error": maximum_active_stationarity_error,
        "maximum_zero_action_expected_logit_gradient": maximum_zero_action_gradient,
    }


def gradient(probabilities, weights):
    """Exact derivative with respect to action-one logit, frozen Q/weights."""
    p0, p1 = probabilities
    return p0 * p1 * (weights[1] - weights[0])


def check_identities():
    maximum_error = 0.0
    for p in (1e-8, 0.001, 0.1, 0.5, 0.99):
        for q in (0.0, 0.02, 0.5, 0.9, 1.0):
            for magnitude in (0.1, 0.5, 1.0, 2.0):
                for eta in (0.5, 1.0, 2.0):
                    expectation = q * dg_weight(magnitude, p, eta)
                    expectation += (1 - q) * dg_weight(-magnitude, p, eta)
                    calculated = integrated_weight(q, p, magnitude, eta)
                    maximum_error = max(maximum_error, abs(expectation - calculated))
    return maximum_error


def main():
    scenarios = []
    probabilities = (0.9, 0.1)
    for q in ((0.02, 0.02), (0.02, 0.1)):
        true_weights = [integrated_weight(v, p) for v, p in zip(q, probabilities)]
        naive_weights = [dg_weight(2 * v - 1, p) for v, p in zip(q, probabilities)]
        shifted_q = [v + 0.03 for v in q]
        shifted_weights = [
            integrated_weight(v, p) for v, p in zip(shifted_q, probabilities)
        ]
        shifted_naive_weights = [
            dg_weight(2 * v - 1, p) for v, p in zip(shifted_q, probabilities)
        ]
        binary_weights = [
            integrated_binary_weight(v, p) for v, p in zip(q, probabilities)
        ]
        shifted_binary_weights = [
            integrated_binary_weight(v, p) for v, p in zip(shifted_q, probabilities)
        ]
        # Actor gradient for action one: score is -p1 for action zero, p0 for one.
        score = (-probabilities[1], probabilities[0])
        conditional_outcome_variance = sum(
            p * v * (1 - v) * s * s
            for p, v, s in zip(probabilities, q, score)
        )
        scenarios.append(
            {
                "probabilities": probabilities,
                "success_probabilities": q,
                "ordinary_reward_gradient": gradient(probabilities, q),
                "integrated_signed_dg_gradient": gradient(probabilities, true_weights),
                "naive_expected_signed_utility_dg_gradient": gradient(probabilities, naive_weights),
                "integrated_gradient_change_under_common_q_error_0.03": gradient(probabilities, shifted_weights)
                - gradient(probabilities, true_weights),
                "naive_gradient_change_under_common_q_error_0.03": gradient(probabilities, shifted_naive_weights)
                - gradient(probabilities, naive_weights),
                "integrated_original_binary_dg_gradient": gradient(probabilities, binary_weights),
                "binary_gradient_change_under_common_q_error_0.03": gradient(probabilities, shifted_binary_weights)
                - gradient(probabilities, binary_weights),
                "sampled_outcome_actor_gradient_variance_removed_by_exact_conditioning": conditional_outcome_variance,
            }
        )
    # Same action probability, contrasting success predictions: naive plugging
    # expected signed utility into DG can reverse the intended action ranking.
    p = 1 / 50304
    ranking = {
        "action_probability": p,
        "success_probabilities": (0.02, 0.1),
        "naive_weights": [dg_weight(2 * q - 1, p) for q in (0.02, 0.1)],
        "integrated_weights": [integrated_weight(q, p) for q in (0.02, 0.1)],
    }
    report = {
        "scope": "Analytical finite-action scalar verification; no empirical model or training claim.",
        "assumptions": [
            "Terminal reward R is binary; Q is calibrated P(R=1 | current prefix, chosen action).",
            "Actor score and surprisal are measurable before future outcome; critic weights detached.",
            "Signed utility 2R-1 is a candidate objective change, not a user-approved reward mapping.",
            "No state value or policy-average Q baseline is computed.",
            "Conditional expectation removes outcome sampling noise, not Q prediction error or action sampling noise.",
        ],
        "identity_max_abs_error": check_identities(),
        "scenarios": scenarios,
        "uniform_probability_ranking_example": ranking,
        "eta1_magnitude1_regularizer": "sum_a log(1+pi_a), whose coefficient derivative is -pi_a/(1+pi_a) up to a constant score baseline",
        "original_binary_candidate": {
            "coefficient": "Q*sigmoid(s/eta); eta=1 gives Q/(1+pi)",
            "fixed_prefix_frozen_Q_objective_at_eta1": "sum_a Q_a*log(1+pi_a)",
            "KKT": "pi_a=max(Q_a/nu-1,0), nu=sum_active_Q/(number_active+1)",
            "zero_Q_property": "Expected direct-logit gradient is nonpositive for every exact zero-Q action; all-zero-Q states have zero gradient.",
            "caveat": "Direct-logit guarantees do not guarantee changes under shared neural parameters, inaccurate Q, or finite-sample updates.",
            "optima_checks": check_binary_optima(),
            "optima": [
                {"Q": q, "pi_optimum": binary_optimum(q)[0]}
                for q in ((0.02, 0.1), (0.09, 0.1), (0.02, 0.02), (0, 0))
            ],
        },
        "signed_candidate_counterexample": {
            "probabilities": (0.9, 0.1),
            "Q": (0.2, 0.0),
            "signed_expected_bad_action_logit_gradient": gradient(
                (0.9, 0.1), [integrated_weight(0.2, 0.9), integrated_weight(0, 0.1)]
            ),
            "binary_expected_bad_action_logit_gradient": gradient(
                (0.9, 0.1), [integrated_binary_weight(0.2, 0.9), integrated_binary_weight(0, 0.1)]
            ),
        },
    }
    assert report["identity_max_abs_error"] < 1e-14
    assert ranking["naive_weights"][0] > ranking["naive_weights"][1]
    assert ranking["integrated_weights"][0] < ranking["integrated_weights"][1]
    for scenario in scenarios:
        assert abs(scenario["integrated_gradient_change_under_common_q_error_0.03"]) < 1e-14
    destination = Path("postraining/runs/critic_design_math_20261013/results.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
