"""Exact scalar counterexamples for baseline-free action-Q policy credit.

No models, tensors, random samples, training, or resource-heavy workloads.
The expected update is the ascent gradient with respect to softmax logits,
holding critic predictions and gate weights detached as in the actor loss.
"""

import argparse
import json
import math
from pathlib import Path


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


def expected_gradient(probabilities, weights):
    mean_weight = sum(p * w for p, w in zip(probabilities, weights))
    return [p * (w - mean_weight) for p, w in zip(probabilities, weights)]


def dg_weights(probabilities, values):
    return [q * sigmoid(q * -math.log(p)) for p, q in zip(probabilities, values)]


def run():
    probabilities = [0.9, 0.1]
    equal_q = {}
    for utility in [0.2, -0.2, 0.0]:
        values = [utility, utility]
        weights = dg_weights(probabilities, values)
        equal_q[str(utility)] = {
            "weights": weights,
            "dg_expected_logit_gradient": expected_gradient(probabilities, weights),
            "raw_q_expected_logit_gradient": expected_gradient(probabilities, values),
            "surprisal_independent_gate_gradient": expected_gradient(
                probabilities, [q * sigmoid(q) for q in values]
            ),
        }
        assert abs(expected_gradient(probabilities, values)[0]) < 1e-15
        if utility:
            assert expected_gradient(probabilities, weights)[1] > 0

    # Rare action is actually worse, yet its surprisal advantage outweighs
    # its smaller Q under the original gate.
    unequal_q = [0.2, 0.19]
    original_gradient = expected_gradient(probabilities, dg_weights(probabilities, unequal_q))
    plain_gradient = expected_gradient(probabilities, unequal_q)
    assert original_gradient[1] > 0
    assert plain_gradient[1] < 0

    # Both estimated Q values are unbiased and in [0, 1]. Action 0's
    # estimate is exact; action 1 has independent symmetric error.
    # A nonlinear confidence/reward gate nevertheless prefers noisy action 1.
    equal_probabilities = [0.5, 0.5]
    noisy_estimates = [[0.2, 0.0], [0.2, 0.4]]
    nonlinear_gradients = [
        expected_gradient(equal_probabilities, [q * sigmoid(q) for q in values])
        for values in noisy_estimates
    ]
    noise_gradient = [sum(g[i] for g in nonlinear_gradients) / 2 for i in range(2)]
    linear_gradients = [expected_gradient(equal_probabilities, values) for values in noisy_estimates]
    linear_noise_gradient = [sum(g[i] for g in linear_gradients) / 2 for i in range(2)]
    assert noise_gradient[1] > 0
    assert abs(linear_noise_gradient[1]) < 1e-15

    # A true-zero action receives negative expected ordinary PG movement
    # through softmax competition even though its own sampled weight is zero.
    invalid_values = [0.2, 0.0]
    invalid_gradient = expected_gradient(probabilities, invalid_values)
    assert invalid_gradient[1] < 0

    # Conditional expectation of binary-outcome DG with baseline zero.
    # E[R sigmoid(R surprisal)] = Q sigmoid(surprisal) = Q / (1+p).
    rb_weights = [q / (1 + p) for p, q in zip(probabilities, unequal_q)]
    rb_gradient = expected_gradient(probabilities, rb_weights)
    assert rb_gradient[1] > 0
    for p, q in zip(probabilities, unequal_q):
        sampled_weight_expectation = q * sigmoid(-math.log(p))
        assert abs(sampled_weight_expectation - q / (1 + p)) < 1e-15
    rb_invalid_gradient = expected_gradient(
        probabilities, [q / (1 + p) for p, q in zip(probabilities, invalid_values)]
    )
    assert rb_invalid_gradient[1] < 0

    # Shared token parameters invalidate a per-prefix guarantee even when
    # exact-Q, independent-prefix expected gradients are all correct.
    # At A, closing early is invalid. At B, the same token is necessary.
    shared_probabilities = [0.5, 0.5]  # close, continue
    a_weights = [0.0, 0.1 / 1.5]
    b_weights = [1.0 / 1.5, 0.0]
    a_gradient = expected_gradient(shared_probabilities, a_weights)
    b_gradient = expected_gradient(shared_probabilities, b_weights)
    shared_gradient = [(a + b) / 2 for a, b in zip(a_gradient, b_gradient)]
    assert a_gradient[0] < 0
    assert shared_gradient[0] > 0

    # Interior optimum of the fixed-prefix surrogate differs from reward
    # maximization: Q_i/(1+p_i)=nu for each active action.
    nu = sum(unequal_q) / 3
    rb_optimum = [q / nu - 1 for q in unequal_q]
    assert abs(sum(rb_optimum) - 1) < 1e-15
    assert abs(rb_optimum[0] - 7 / 13) < 1e-15

    return {
        "scope": "Exact finite scalar expectations; no empirical model-performance claims",
        "equal_action_q": equal_q,
        "gate_can_promote_worse_action": {
            "probabilities": probabilities,
            "true_q": unequal_q,
            "original_dg_gradient": original_gradient,
            "raw_q_gradient": plain_gradient,
            "reward_directional_derivative_original_dg": sum(g * d for g, d in zip(plain_gradient, original_gradient)),
            "reward_directional_derivative_raw_q": sum(g * g for g in plain_gradient),
        },
        "unbiased_heteroskedastic_critic_noise": {
            "true_q": [0.2, 0.2],
            "estimates_equiprobable": noisy_estimates,
            "q_sigmoid_q_expected_gradient": noise_gradient,
            "raw_q_expected_gradient": linear_noise_gradient,
        },
        "known_invalid_zero_q": {
            "probabilities": probabilities,
            "true_q": invalid_values,
            "raw_q_gradient": invalid_gradient,
        },
        "binary_outcome_dg_rao_blackwell": {
            "weight": "Q/(1+pi)",
            "probabilities": probabilities,
            "true_q": unequal_q,
            "expected_logit_gradient": rb_gradient,
            "reward_directional_derivative": sum(g * d for g, d in zip(plain_gradient, rb_gradient)),
            "fixed_prefix_optimum": rb_optimum,
            "fixed_prefix_reward_at_optimum": sum(p * q for p, q in zip(rb_optimum, unequal_q)),
            "raw_reward_optimum": 0.2,
            "invalid_expected_logit_gradient": rb_invalid_gradient,
            "sampled_outcome_weight_variances": [q * (1 - q) / (1 + p) ** 2 for p, q in zip(probabilities, unequal_q)],
            "critic_estimate_noise_variance_multiplier": [1 / (1 + p) ** 2 for p in probabilities],
        },
        "shared_parameter_invalid_action_counterexample": {
            "prefix_a_q_close_continue": [0, 0.1],
            "prefix_b_q_close_continue": [1, 0],
            "prefix_probabilities": shared_probabilities,
            "prefix_a_local_gradient": a_gradient,
            "prefix_b_local_gradient": b_gradient,
            "equal_occupancy_shared_parameter_gradient": shared_gradient,
            "consequence": "Shared close-token logit increases, raising invalid premature closure at prefix A",
        },
        "limits": [
            "Exact original DG changes policy with equal nonzero action utilities even without critic error.",
            "A fixed utility threshold cannot eliminate this for every state.",
            "A surprisal-independent nonlinear Q gate removes equal-exact-Q drift but not heteroskedastic-noise bias.",
            "Raw action Q avoids these two effects in expectation when estimates are conditionally unbiased.",
            "Raw action Q still has sampling variance and cannot correct biased action-conditioned critic errors.",
            "Binary-outcome Rao-Blackwell DG is linear in Q estimates, but retains surprise exploration and does not universally ascend reward.",
            "Zero-Q suppression is a per-prefix independent-logit guarantee; shared parameters can increase invalid actions elsewhere.",
            "Policy centering and expected pairwise Q differences are algebraically baseline-like and do not satisfy a strict ban.",
            "No training optimum or causal diagnosis follows from these scalar counterexamples.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
