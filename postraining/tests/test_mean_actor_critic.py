"""Small CPU algebra oracles; these do not instantiate or execute models."""
import pytest
import torch

import itertools
import math

from postraining.mean_actor_critic import (
    GATE_ADVANTAGE_ESTIMATORS, GATE_SHUFFLE_SURPRISAL_EDGES, MAC_CONTROL_ESTIMATORS,
    MAC_ESTIMATORS, SAMPLED_ONLY_ESTIMATORS, SEQUENCE_ESTIMATORS, answer_cluster_labels,
    answer_cluster_surprisal, answer_gate, gate_shuffle_strata, leave_one_out_advantages,
    sequence_coefficients, shuffle_within_strata, softcapped_mean_actor_critic,
    softcapped_target_logprobs_and_advantages, validate_mac_estimator, validate_softcap,
)
from postraining.nano_backbone import (
    SoftcappedTargetLogprobsEntropyAdvantages, rational_softcap,
)


def _inputs():
    generator = torch.Generator().manual_seed(8)
    raw = torch.randn(5, 7, dtype=torch.float64, generator=generator, requires_grad=True)
    targets = torch.tensor([0, 3, 6, 2, 2])
    directions = torch.randn(5, 3, dtype=torch.float64, generator=generator, requires_grad=True)
    latents = torch.randn(7, 3, dtype=torch.float64, generator=generator, requires_grad=True)
    weights = torch.tensor([.03, .12, .21, 0, .04], dtype=torch.float64, requires_grad=True)
    return raw, targets, directions, latents, weights


def _mac(raw, targets, cap, directions, latents, weights, residuals=None, estimator="all_action",
         gate_advantages=None):
    if residuals is None:
        residuals = torch.zeros(raw.size(0), dtype=raw.dtype)
    if estimator in GATE_ADVANTAGE_ESTIMATORS and gate_advantages is None:
        gate_advantages = GATE_ADVANTAGES.to(raw.dtype)
    return softcapped_mean_actor_critic(raw, targets, cap, directions, latents, weights,
                                        residuals, estimator, gate_advantages)


# A critic's per-token advantages for the TD-gated estimator, of both signs.
GATE_ADVANTAGES = torch.tensor([.2, .5, -.6, .05, -.1], dtype=torch.float64)


@pytest.mark.parametrize("cap", [.7, 15.])
def test_analytic_gradient_matches_bruteforce_detached_centered_q(cap):
    raw, targets, directions, latents, weights = _inputs()
    probabilities = rational_softcap(raw, cap).softmax(-1)
    scores = directions.detach() @ latents.detach().T
    advantages = scores - (probabilities.detach() * scores).sum(-1, keepdim=True)
    oracle = -(weights.detach()[:, None] * probabilities * advantages.detach()).sum()
    expected = torch.autograd.grad(oracle, raw)[0]
    loss, *_ = _mac(raw, targets, cap, directions, latents, weights)
    loss.backward()
    torch.testing.assert_close(raw.grad, expected, atol=2e-14, rtol=2e-13)
    assert loss.item() == pytest.approx(oracle.item(), abs=1e-14)
    assert directions.grad is None and latents.grad is None and weights.grad is None


def test_statistics_match_the_shared_nano_readout():
    raw, targets, directions, latents, weights = _inputs()
    _, logprobs, entropy, advantages, variance, _census, offsets = _mac(
        raw, targets, 15., directions, latents, weights)
    probabilities = rational_softcap(raw.detach(), 15.).softmax(-1)
    scores = directions.detach() @ latents.detach().T
    centered = scores - (probabilities * scores).sum(-1, keepdim=True)
    torch.testing.assert_close(variance, (probabilities * centered.square()).sum(-1))
    # The offset that centres the critic's own fit: u . offset is the
    # taken token's advantage.
    torch.testing.assert_close(offsets, latents.detach()[targets] - probabilities @ latents.detach())
    torch.testing.assert_close(advantages, (directions.detach() * offsets).sum(-1))
    reference = SoftcappedTargetLogprobsEntropyAdvantages.apply(
        raw.detach(), targets, 15., directions.detach(), latents.detach())
    # The log-probability shares the readout's normalization arithmetic.
    assert torch.equal(logprobs, reference[0])
    torch.testing.assert_close(entropy, reference[1], atol=1e-14, rtol=1e-13)
    # The shared readout reads u against its offsets, in the readout's own
    # compute dtype; MAC centres the whole vocabulary's scores.
    torch.testing.assert_close(advantages, reference[2], atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(offsets, reference[3], atol=1e-12, rtol=1e-11)
    assert not logprobs.requires_grad and not entropy.requires_grad
    assert not advantages.requires_grad and not offsets.requires_grad


def test_unsampled_actions_receive_gradient_and_state_offset_cancels():
    # A huge cap makes the renderer the identity up to rounding.
    raw = torch.zeros(1, 4, dtype=torch.float64, requires_grad=True)
    directions = torch.tensor([[1.]], dtype=torch.float64)
    latents = torch.tensor([[-2.], [-1.], [1.], [2.]], dtype=torch.float64)
    weights = torch.ones(1, dtype=torch.float64)
    loss, *_ = _mac(raw, torch.tensor([1]), 1e9, directions, latents, weights)
    loss.backward()
    torch.testing.assert_close(raw.grad, torch.tensor([[.5, .25, -.25, -.5]], dtype=torch.float64))
    # A common action-embedding offset adds only a state term to Q.
    shifted = raw.detach().clone().requires_grad_(True)
    _mac(shifted, torch.tensor([1]), 1e9, directions, latents + 20,
                                 weights)[0].backward()
    torch.testing.assert_close(shifted.grad, raw.grad)


def test_globally_weighted_chunks_match_unsplit_gradient():
    raw, targets, directions, latents, weights = _inputs()
    full = torch.autograd.grad(
        _mac(raw, targets, 15., directions, latents, weights)[0], raw)[0]
    chunks = []
    for start in range(0, len(raw), 2):
        part = raw[start:start + 2].detach().requires_grad_(True)
        _mac(part, targets[start:start + 2], 15., directions[start:start + 2],
                                     latents, weights[start:start + 2])[0].backward()
        chunks.append(part.grad)
    torch.testing.assert_close(torch.cat(chunks), full)


def test_invalid_shapes_rejected():
    raw, targets, directions, latents, weights = _inputs()
    with pytest.raises(ValueError, match="action latents"):
        _mac(raw, targets, 15., directions, latents[:6], weights)
    with pytest.raises(ValueError, match="nonempty"):
        _mac(raw[:0], targets[:0], 15., directions[:0], latents, weights[:0])


@pytest.mark.parametrize("cap", [0, -1, float("inf"), float("nan")])
def test_invalid_softcap_rejected(cap):
    with pytest.raises(ValueError, match="softcap"):
        validate_softcap(cap)


def test_tensor_softcap_rejected():
    with pytest.raises(TypeError):
        validate_softcap(torch.tensor(15.))


def _oracle_loss(raw, targets, cap, directions, latents, weights, residuals, estimator):
    """The estimator's loss by autograd through the renderer, every
    coefficient detached; ``residuals`` are G - c(s)."""
    log_probabilities = rational_softcap(raw, cap).log_softmax(-1)
    probabilities = log_probabilities.exp()
    scores = directions.detach() @ latents.detach().T
    advantages = (scores - (probabilities.detach() * scores).sum(-1, keepdim=True)).detach()
    sampled = log_probabilities.gather(-1, targets[:, None]).squeeze(-1)
    coefficient = {
        "all_action": torch.zeros_like(residuals),
        "control_variate": residuals - advantages.gather(-1, targets[:, None]).squeeze(-1),
        "sampled_baseline": residuals,
        # DG's detached gate on the sampled term, from the current policy.
        "delightful_sampled_baseline": residuals * torch.sigmoid(residuals * -sampled.detach()),
        # The same gate on the critic's advantage, not the outcome.
        "delightful_td_gate": residuals * torch.sigmoid(GATE_ADVANTAGES * -sampled.detach()),
        # On the head's own advantage of the sampled token.
        "delightful_head_gate": residuals * torch.sigmoid(
            advantages.gather(-1, targets[:, None]).squeeze(-1) * -sampled.detach()),
        # On supplied advantages, as the TD gate.
        "delightful_shuffled_head_gate":
            residuals * torch.sigmoid(GATE_ADVANTAGES * -sampled.detach()),
        # A sequence estimator's caller supplies each trajectory's
        # coefficient as the residual; the readout uses it verbatim.
        "rloo_sequence": residuals,
        "delightful_answer_gate": residuals,
    }[estimator].detach()
    loss = -(weights.detach() * coefficient * sampled).sum()
    if estimator not in SAMPLED_ONLY_ESTIMATORS:
        loss = loss - (weights.detach()[:, None] * probabilities * advantages).sum()
    return loss


@pytest.mark.parametrize("estimator", MAC_ESTIMATORS + MAC_CONTROL_ESTIMATORS)
@pytest.mark.parametrize("cap", [.7, 15.])
def test_every_estimator_matches_its_autograd_oracle(estimator, cap):
    raw, targets, directions, latents, weights = _inputs()
    residuals = torch.tensor([.4, -.9, .1, .7, -.3], dtype=torch.float64, requires_grad=True)
    oracle = _oracle_loss(raw, targets, cap, directions, latents, weights, residuals, estimator)
    expected = torch.autograd.grad(oracle, raw)[0]
    loss, *_ = _mac(raw, targets, cap, directions, latents, weights, residuals, estimator)
    loss.backward()
    torch.testing.assert_close(raw.grad, expected, atol=2e-14, rtol=2e-13)
    assert loss.item() == pytest.approx(oracle.item(), abs=1e-13)
    assert residuals.grad is None and directions.grad is None


def test_all_action_ignores_residuals():
    raw, targets, directions, latents, weights = _inputs()
    gradients = []
    for residuals in (torch.zeros(5, dtype=torch.float64), torch.full((5,), 3., dtype=torch.float64)):
        part = raw.detach().clone().requires_grad_(True)
        _mac(part, targets, 15., directions, latents, weights, residuals)[0].backward()
        gradients.append(part.grad)
    assert torch.equal(*gradients)


def test_control_variate_is_unbiased_for_any_critic():
    """Exact expectation over the sampled token with the true return: the
    control variate and the sampled baseline both give the gradient of MAC
    under the true Q, however wrong the bilinear critic is; MAC under the
    critic does not."""
    generator = torch.Generator().manual_seed(3)
    vocab, dim, cap = 6, 2, 15.
    raw = torch.randn(1, vocab, dtype=torch.float64, generator=generator)
    directions = torch.randn(1, dim, dtype=torch.float64, generator=generator)
    latents = torch.randn(vocab, dim, dtype=torch.float64, generator=generator)
    critic_state = 0.37  # c(s), the policy-centred critic's E_pi Q
    probabilities = rational_softcap(raw, cap).softmax(-1)[0]
    scores = (directions @ latents.T)[0]
    q_critic = critic_state + scores - (probabilities * scores).sum()
    q_true = torch.rand(vocab, dtype=torch.float64, generator=generator)
    weights = torch.ones(1, dtype=torch.float64)

    def gradient(estimator, target, mac_directions=directions, mac_latents=latents):
        part = raw.clone().requires_grad_(True)
        # G - c; the readout forms G - Q(a_t) = G - c - A(a_t) from it.
        residual = (q_true[target] - critic_state).reshape(1)
        _mac(part, torch.tensor([target]), cap, mac_directions, mac_latents, weights,
             residual, estimator)[0].backward()
        return part.grad

    # MAC under the true Q: a one-dimensional "embedding" that is Q itself.
    truth = gradient("all_action", 0, torch.ones(1, 1, dtype=torch.float64), q_true[:, None])
    for estimator in ("control_variate", "sampled_baseline"):
        expectation = sum(probabilities[a] * gradient(estimator, a) for a in range(vocab))
        torch.testing.assert_close(expectation, truth, atol=1e-12, rtol=1e-10)
    biased = gradient("all_action", 0)
    assert (biased - truth).abs().max() > 1e-3


def test_the_delight_gate_spares_surprising_tokens_of_failed_continuations():
    """DG's asymmetry at one state: with G - c of either sign, a confident
    sampled token keeps about half its sampled-baseline weight, a surprising
    one nearly all of it when the continuation beat c and nearly none when
    it fell short."""
    vocab, cap = 400, 15.
    raw = torch.zeros(1, vocab, dtype=torch.float64)
    raw[0, 0] = 9.0  # token 0 is confident; the rest share the remaining mass
    directions = torch.zeros(1, 2, dtype=torch.float64)
    latents = torch.zeros(vocab, 2, dtype=torch.float64)
    weights = torch.ones(1, dtype=torch.float64)

    def ratio(target, residual):
        grads = []
        for estimator in ("sampled_baseline", "delightful_sampled_baseline"):
            part = raw.clone().requires_grad_(True)
            _mac(part, torch.tensor([target]), cap, directions, latents, weights,
                 torch.tensor([residual], dtype=torch.float64), estimator)[0].backward()
            grads.append(part.grad)
        return float(grads[1].norm() / grads[0].norm())

    confident, surprising = ratio(0, 1.0), ratio(1, 1.0)
    assert 0.5 < confident < 0.55 and surprising > 0.99
    assert 0.45 < ratio(0, -1.0) < 0.5 and ratio(1, -1.0) < 0.01


def test_the_td_gate_reads_the_critic_not_the_outcome():
    """With the gate on the critic's advantage A, the outcome's sign moves
    the coefficient but not the gate: a surprising token's weight is set by
    A alone (near one for A > 0, near zero for A < 0), and a token the
    critic sees no effect in keeps half of the sampled baseline either way.
    The outcome-gated DG instead keeps a surprising token of a successful
    continuation whatever its critic advantage."""
    vocab, cap = 400, 15.
    raw = torch.zeros(1, vocab, dtype=torch.float64)
    raw[0, 0] = 9.0
    directions = torch.zeros(1, 2, dtype=torch.float64)
    latents = torch.zeros(vocab, 2, dtype=torch.float64)
    weights = torch.ones(1, dtype=torch.float64)

    def gradient(estimator, target, residual, advantage=None):
        part = raw.clone().requires_grad_(True)
        _mac(part, torch.tensor([target]), cap, directions, latents, weights,
             torch.tensor([residual], dtype=torch.float64), estimator,
             None if advantage is None else torch.tensor([advantage], dtype=torch.float64)
             )[0].backward()
        return part.grad

    def ratio(target, residual, advantage):
        return float(gradient("delightful_td_gate", target, residual, advantage).norm()
                     / gradient("sampled_baseline", target, residual).norm())

    for residual in (1.0, -1.0):
        assert ratio(1, residual, 0.0) == pytest.approx(0.5)
        assert ratio(1, residual, 1.0) > 0.99 and ratio(1, residual, -1.0) < 0.01
    harmful = gradient("delightful_td_gate", 1, 1.0, -1.0)
    outcome_gated = gradient("delightful_sampled_baseline", 1, 1.0)
    assert float(outcome_gated.norm()) > 50 * float(harmful.norm())


def test_gate_advantages_come_only_with_the_td_gate():
    raw, targets, directions, latents, weights = _inputs()
    residuals = torch.zeros(raw.size(0), dtype=raw.dtype)
    with pytest.raises(ValueError, match="gate advantages"):
        softcapped_mean_actor_critic(raw, targets, 15., directions, latents, weights,
                                     residuals, "delightful_td_gate")
    with pytest.raises(ValueError, match="gate advantages"):
        softcapped_mean_actor_critic(raw, targets, 15., directions, latents, weights,
                                     residuals, "sampled_baseline", GATE_ADVANTAGES)


def test_unknown_estimator_rejected():
    assert validate_mac_estimator("control_variate") == "control_variate"
    with pytest.raises(ValueError, match="unknown MAC estimator"):
        validate_mac_estimator("q_prop")
    # A measurement control is no training objective.
    with pytest.raises(ValueError, match="unknown MAC estimator"):
        validate_mac_estimator("delightful_shuffled_head_gate")
    assert validate_mac_estimator("delightful_shuffled_head_gate", controls=True)


@pytest.mark.parametrize("cap", [.7, 15.])
def test_the_head_gate_reads_exactly_the_readouts_target_advantages(cap):
    """The head gate is the supplied-gate estimator fed the readout's own
    A(a_t), bit for bit; the loss-free pass the shuffled control reads
    returns that A and log-probability bit for bit too."""
    raw, targets, directions, latents, weights = _inputs()
    residuals = torch.tensor([.4, -.9, .1, .7, -.3], dtype=torch.float64)
    head = raw.detach().clone().requires_grad_(True)
    loss, logprobs, _, advantages, *_ = _mac(head, targets, cap, directions, latents, weights,
                                             residuals, "delightful_head_gate")
    loss.backward()
    supplied = raw.detach().clone().requires_grad_(True)
    supplied_loss, *_ = _mac(supplied, targets, cap, directions, latents, weights, residuals,
                             "delightful_shuffled_head_gate", advantages)
    supplied_loss.backward()
    assert torch.equal(loss, supplied_loss) and torch.equal(head.grad, supplied.grad)
    # A different A moves the supplied gate, so the equality is not vacuous.
    other = raw.detach().clone().requires_grad_(True)
    _mac(other, targets, cap, directions, latents, weights, residuals,
         "delightful_shuffled_head_gate", advantages.flip(0))[0].backward()
    assert not torch.equal(other.grad, head.grad)
    loss_free = softcapped_target_logprobs_and_advantages(raw, targets, cap, directions, latents)
    assert torch.equal(loss_free[0], logprobs) and torch.equal(loss_free[1], advantages)
    assert not any(t.requires_grad for t in loss_free)


def test_gate_shuffle_strata_bin_surprisal_by_the_recorded_edges_and_span():
    edges = GATE_SHUFFLE_SURPRISAL_EDGES
    assert edges[0] == 2.0 ** -10 and edges[-1] == 8.0
    assert all(b == 2 * a for a, b in zip(edges, edges[1:]))
    surprisal = torch.tensor([0.0, edges[0] / 2, edges[0], edges[3] * 1.5, 8.0, 50.0])
    bins = gate_shuffle_strata(-surprisal, torch.zeros(6))
    # Bin i holds edges[i-1] <= l < edges[i]; bin 0 below the first edge.
    assert bins.tolist() == [0, 0, 1, 4, 14, 14]
    answer = gate_shuffle_strata(-surprisal, torch.ones(6))
    assert (answer - bins).tolist() == [len(edges) + 1] * 6


def _strata_and_values(n=600, seed=0):
    generator = torch.Generator().manual_seed(seed)
    strata = torch.randint(0, 7, (n,), generator=generator)
    values = torch.randn(n, generator=generator)
    return strata, values


def test_the_shuffle_is_a_within_stratum_permutation_keeping_each_multiset():
    strata, values = _strata_and_values()
    shuffled = shuffle_within_strata(values, strata, torch.Generator().manual_seed(11))
    assert torch.equal(shuffled.sort().values, values.sort().values)
    for stratum in strata.unique():
        mask = strata == stratum
        assert torch.equal(shuffled[mask].sort().values, values[mask].sort().values)
    # It moves values: with ~86 states per stratum, few fixed points.
    assert (shuffled == values).float().mean() < 0.1
    # A value never leaves its stratum even when strata interleave.
    lone = torch.tensor([0, 1, 0, 1, 2])
    lone_values = torch.tensor([1., 10., 2., 20., 99.])
    moved = shuffle_within_strata(lone_values, lone, torch.Generator().manual_seed(3))
    assert set(moved[lone == 0].tolist()) == {1., 2.} and moved[4] == 99.


def test_the_shuffle_is_deterministic_in_its_generator_state():
    strata, values = _strata_and_values()
    first = shuffle_within_strata(values, strata, torch.Generator().manual_seed(5))
    again = shuffle_within_strata(values, strata, torch.Generator().manual_seed(5))
    other = shuffle_within_strata(values, strata, torch.Generator().manual_seed(6))
    assert torch.equal(first, again) and not torch.equal(first, other)
    # One generator across calls (shards): the second draw differs from the
    # first but replays with the generator.
    generator = torch.Generator().manual_seed(5)
    shards = [shuffle_within_strata(values, strata, generator) for _ in range(2)]
    replay = torch.Generator().manual_seed(5)
    assert torch.equal(shards[0], first) and not torch.equal(shards[1], first)
    assert all(torch.equal(shard, shuffle_within_strata(values, strata, replay))
               for shard in shards)


def test_the_shuffle_is_uniform_over_a_strata_permutations():
    values = torch.tensor([0., 1., 2.])
    strata = torch.zeros(3, dtype=torch.long)
    generator = torch.Generator().manual_seed(1)
    counts = {}
    draws = 6000
    for _ in range(draws):
        key = tuple(shuffle_within_strata(values, strata, generator).tolist())
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 6
    # Each permutation 1000 +- 4 SD (SD about 29).
    assert all(abs(count - draws / 6) < 4 * (draws / 6 * 5 / 6) ** 0.5 for count in counts.values())


def test_the_shuffle_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="matching"):
        shuffle_within_strata(torch.zeros(3), torch.zeros(4, dtype=torch.long),
                              torch.Generator().manual_seed(0))


def test_support_census_sums_the_centered_q_by_support():
    from postraining.mean_actor_critic import SUPPORT_CENSUS_FIELDS

    generator = torch.Generator().manual_seed(9)
    # A confident policy over 40 tokens: most of them fall below 1e-4.
    raw = torch.randn(6, 40, dtype=torch.float64, generator=generator) * 6
    targets = torch.zeros(6, dtype=torch.long)
    directions = torch.randn(6, 3, dtype=torch.float64, generator=generator)
    latents = torch.randn(40, 3, dtype=torch.float64, generator=generator)
    weights = torch.ones(6, dtype=torch.float64)
    census = dict(zip(SUPPORT_CENSUS_FIELDS, _mac(raw, targets, 15., directions, latents, weights)[5].tolist()))
    probabilities = rational_softcap(raw, 15.).softmax(-1)
    scores = directions @ latents.T
    centered = scores - (probabilities * scores).sum(-1, keepdim=True)
    off = probabilities < 1e-4
    order = probabilities.argsort(-1, descending=True)
    near = centered.gather(-1, order[:, 1:8])
    top = centered.gather(-1, order[:, :1])
    assert 0 < off.sum() < off.numel()
    assert census["offsupport_count"] == off.sum()
    assert census["offsupport_positive"] == (off & (centered > 0)).sum()
    assert census["offsupport_advantage"] == pytest.approx(float(centered[off].sum()))
    assert census["rank2to8_count"] == 42 and census["top_count"] == 6
    assert census["rank2to8_positive"] == (near > 0).sum()
    assert census["rank2to8_advantage"] == pytest.approx(float(near.sum()))
    assert census["top_advantage"] == pytest.approx(float(top.sum()))


# --- Answer-level sequence estimators -------------------------------------


def test_leave_one_out_advantage_excludes_the_scored_sample():
    rewards = torch.tensor([[1., 0., 1., 0.], [1., 1., 1., 1.], [.5, 0., 0., 0.]],
                           dtype=torch.float64)
    advantages = leave_one_out_advantages(rewards)
    for group in range(rewards.size(0)):
        for i in range(4):
            others = torch.cat((rewards[group, :i], rewards[group, i + 1:]))
            assert advantages[group, i] == pytest.approx(float(rewards[group, i] - others.mean()))
    # A uniform group carries no signal; U is G/(G-1) times the mean-centred reward.
    assert not advantages[1].any()
    torch.testing.assert_close(advantages, (rewards - rewards.mean(-1, keepdim=True)) * 4 / 3)
    with pytest.raises(ValueError, match="at least two"):
        leave_one_out_advantages(torch.ones(3, 1))


def test_answer_clusters_share_labels_and_ungraded_rows_share_one_cluster():
    answers = ["12", "7", "12", None, "7", None, "12", "-3"]
    labels = answer_cluster_labels(answers)
    assert labels == [0, 1, 0, 3, 1, 3, 0, 7]
    surprisal = answer_cluster_surprisal(torch.tensor(labels))
    counts = [3, 2, 3, 2, 2, 2, 3, 1]
    torch.testing.assert_close(
        surprisal, torch.tensor([-math.log(n / 8) for n in counts], dtype=torch.float64))
    # A singleton is log G; a unanimous group is exactly zero, including
    # one where no row has a graded answer.
    assert surprisal[7] == pytest.approx(math.log(8))
    for unanimous in (["5"] * 6, [None] * 6):
        surprisal = answer_cluster_surprisal(torch.tensor(answer_cluster_labels(unanimous)))
        assert torch.equal(surprisal, torch.zeros(6, dtype=torch.float64))
    # The row mean is the group's plug-in answer entropy.
    labels = torch.tensor(answer_cluster_labels(answers))
    shares = torch.tensor([3, 2, 2, 1], dtype=torch.float64) / 8
    assert float(answer_cluster_surprisal(labels).mean()) == pytest.approx(
        float(-(shares * shares.log()).sum()))
    # Batched over groups.
    batched = answer_cluster_surprisal(torch.tensor([[0, 0, 2, 3], [0, 1, 1, 1]]))
    torch.testing.assert_close(batched, torch.tensor(
        [[math.log(2), math.log(2), math.log(4), math.log(4)],
         [math.log(4), math.log(4 / 3), math.log(4 / 3), math.log(4 / 3)]],
        dtype=torch.float64))


def test_the_answer_gate_limits_and_its_rloo_control():
    advantages = torch.tensor([1., -1., .5, -.5, 0., 30., -30.], dtype=torch.float64)
    surprisal = torch.tensor([math.log(16)] * 4 + [1., 1., 1.], dtype=torch.float64)
    gate = answer_gate(advantages, surprisal)
    torch.testing.assert_close(gate, torch.sigmoid(advantages * surprisal))
    # A rare correct answer is reinforced nearly fully, a rare wrong one is
    # nearly unblamed; the gate saturates at 1 and 0.
    assert gate[0] > 0.94 and gate[1] < 0.06
    assert gate[5] == pytest.approx(1.0) and gate[6] == pytest.approx(0.0, abs=1e-12)
    # A unanimous group (l = 0) gates at exactly 1/2, whatever U.
    assert torch.equal(answer_gate(advantages, torch.zeros_like(advantages)),
                       torch.full_like(advantages, 0.5))
    rloo = sequence_coefficients("rloo_sequence", advantages, surprisal)
    dg = sequence_coefficients("delightful_answer_gate", advantages, surprisal)
    assert torch.equal(rloo, advantages)
    torch.testing.assert_close(dg, advantages * gate)
    # The gate rescales but never flips a trajectory's coefficient.
    assert bool((dg * rloo >= 0).all()) and bool((dg.abs() <= rloo.abs()).all())
    with pytest.raises(ValueError, match="not a sequence estimator"):
        sequence_coefficients("sampled_baseline", advantages, surprisal)
    assert set(SEQUENCE_ESTIMATORS) <= set(SAMPLED_ONLY_ESTIMATORS) & set(MAC_ESTIMATORS)


def _bandit_expectations(estimator, logits, rewards, answers, samples):
    """Exact expectations over every ordered group of ``samples`` draws from
    softmax(logits): of the estimator's per-sample mean of coefficient times
    the score, and the true gradient of J = E_pi r."""
    theta = logits.clone().requires_grad_(True)
    log_pi = theta.log_softmax(-1)
    pi = log_pi.exp()
    true_gradient = torch.autograd.grad((pi * rewards).sum(), theta, retain_graph=True)[0]
    expectation = torch.zeros_like(logits)
    for draw in itertools.product(range(logits.numel()), repeat=samples):
        index = torch.tensor(draw)
        advantages = leave_one_out_advantages(rewards[index])
        surprisal = answer_cluster_surprisal(torch.tensor(
            answer_cluster_labels([answers[a] for a in draw])))
        coefficients = sequence_coefficients(estimator, advantages, surprisal).detach()
        objective = (coefficients * log_pi[index]).sum() / samples
        gradient = torch.autograd.grad(objective, theta, retain_graph=True)[0]
        expectation += float(pi[index].prod()) * gradient
    return expectation, true_gradient


def test_rloo_sequence_is_unbiased_and_the_answer_gate_is_not():
    """A four-arm bandit, groups of three: arms 0 and 1 give the same
    answer, arm 3 none. Enumerating all 64 ordered groups, RLOO's mean
    estimate is grad J exactly; the gated estimate is not."""
    logits = torch.tensor([1.2, -0.3, 0.4, 0.0], dtype=torch.float64)
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    answers = ["4", "4", "5", None]
    expectation, truth = _bandit_expectations("rloo_sequence", logits, rewards, answers, 3)
    torch.testing.assert_close(expectation, truth, atol=1e-14, rtol=1e-12)
    biased, _ = _bandit_expectations("delightful_answer_gate", logits, rewards, answers, 3)
    assert (biased - truth).abs().max() > 1e-3
    # Non-binary rewards too: RLOO's baseline is any function of the others.
    rewards = torch.tensor([1.0, 0.3, 0.05, 0.0], dtype=torch.float64)
    expectation, truth = _bandit_expectations("rloo_sequence", logits, rewards, answers, 3)
    torch.testing.assert_close(expectation, truth, atol=1e-14, rtol=1e-12)
