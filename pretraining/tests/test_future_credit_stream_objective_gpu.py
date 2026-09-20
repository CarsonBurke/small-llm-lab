"""Future-bag objective mathematics on CUDA; execute through mlq.

An analytic two-coordinate recurrence gives independent oracles for the bag
target, its masks, and gradient ownership. It is not a replacement training
model.
"""

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from pretraining.future_credit_stream.objective import (
    CE_STATISTICS,
    FUTURE_BAG_STATISTICS,
    FutureBagObjective,
    StreamingObjective,
    TemporalReferenceObjective,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

BOS = 1
VOCAB = 4


class _AnalyticWriter(nn.Module):
    def __init__(self):
        super().__init__()
        self.write = nn.Parameter(torch.tensor([[1.0, 0.1], [-0.2, 0.9]], device="cuda"))
        self.gate_hidden = nn.Parameter(torch.tensor([0.3, -0.4], device="cuda"))
        self.gate_carry = nn.Parameter(torch.tensor([0.1, 0.2], device="cuda"))
        self.gate_bias = nn.Parameter(torch.tensor(0.5, device="cuda"))

    def forward(self, hidden, previous):
        logit = hidden @ self.gate_hidden + previous @ self.gate_carry + self.gate_bias
        gate = logit.sigmoid()[:, None]
        return gate * (hidden @ self.write) + (1 - gate) * previous, gate.expand_as(hidden)


def _scaled(parameter, weight):
    if weight == 0.0:
        return parameter.detach()
    return parameter.detach() + weight * (parameter - parameter.detach())


class _AnalyticModel(nn.Module):
    config = {"vocab_size": VOCAB}

    def __init__(self, use_writer=True):
        super().__init__()
        self.core = nn.Parameter(torch.tensor([[0.7, -0.2], [0.3, 0.6]], device="cuda"))
        self.readout = nn.Parameter(torch.tensor([[0.5, -0.1, 0.4, -0.3], [-0.3, 0.8, 0.2, 0.1]], device="cuda"))
        self.gains = nn.Parameter(torch.tensor([1.2, 0.8], device="cuda"))
        self.writer = _AnalyticWriter() if use_writer else None

    def forward(self, observed, carry, temporal_gradient=False):
        if not temporal_gradient:
            carry = carry.detach()
        features = F.one_hot(observed % 2, 2).float() + carry
        hidden = features @ self.core
        return hidden @ self.readout, hidden

    def carry_readout(self, carry, parameter_gradient=0.0):
        direction = F.rms_norm(carry, (2,), weight=_scaled(self.gains, parameter_gradient))
        return direction @ _scaled(self.readout, parameter_gradient)


def _transition():
    observed = torch.tensor([2, 1, 0, 3, 2], device="cuda")
    targets = torch.tensor([2, 2, BOS, 3, 0], device="cuda")
    # future[j] is x_{t+2+j}; each column is one lane.
    future = torch.tensor([[3, BOS, 2, 0, 2],
                           [0, 2, 3, BOS, 2],
                           [2, 3, 0, 3, BOS]], device="cuda")
    resets = torch.tensor([False, True, False, True, False], device="cuda")
    carry = torch.tensor([[0.2, -0.4], [2.0, -1.0], [-0.5, 0.8], [1.5, 0.3], [0.9, 0.7]],
                         device="cuda", requires_grad=True)
    return observed, targets, future, resets, carry


def _hand_bag(discount):
    """The bag each lane of ``_transition`` must receive, written out by hand.

    The current target ``x_{t+1}`` always leads with weight 1; ``future[j]``
    follows with weight ``discount^(j+1)`` until the closing BOS, inclusive.
    """
    g = discount
    rows = torch.zeros(5, VOCAB, device="cuda")
    rows[0, 2] += 1.0; rows[0, 3] += g; rows[0, 0] += g ** 2; rows[0, 2] += g ** 3   # 2 | 3, 0, 2 continue
    rows[1, 2] += 1.0; rows[1, BOS] += g                                             # 2 | closing BOS
    rows[2, BOS] += 1.0                                                              # closing BOS is the target
    rows[3, 3] += 1.0; rows[3, 0] += g; rows[3, BOS] += g ** 2                        # 3 | 0, closing BOS
    rows[4, 0] += 1.0; rows[4, 2] += g; rows[4, 2] += g ** 2; rows[4, BOS] += g ** 3   # 0 | 2, 2, closing BOS
    return rows / rows.sum(-1)[:, None]


def _oracle(model, observed, targets, future, resets, carry, discount, weight):
    """Independent equations for every loss term, keeping graphs live."""
    state = carry.detach().masked_fill(resets[:, None], 0)
    features = F.one_hot(observed % 2, 2).float() + state
    hidden = features @ model.core
    ce = F.cross_entropy(hidden @ model.readout, targets)
    write_input = hidden.detach() + weight * (hidden - hidden.detach())
    next_carry, gate = model.writer(write_input, state)
    bag = _hand_bag(discount)
    direction = F.rms_norm(next_carry, (2,), weight=_scaled(model.gains, weight))
    log_belief = (direction @ _scaled(model.readout, weight)).log_softmax(-1)
    bag_loss = -(bag * log_belief).sum(-1).mean()
    with torch.no_grad():
        hidden_direction = F.rms_norm(hidden, (2,), weight=model.gains)
        hidden_bag_loss = -(bag * (hidden_direction @ model.readout).log_softmax(-1)).sum(-1).mean()
    return {"hidden": hidden, "ce": ce, "next_carry": next_carry, "gate": gate,
            "bag": bag, "bag_loss": bag_loss, "hidden_bag_loss": hidden_bag_loss}


@pytest.mark.parametrize("discount", [0.0, 0.65, 1.0])
def test_bag_targets_masks_and_gradient_ownership(discount):
    model = _AnalyticModel()
    objective = FutureBagObjective(model, bos_id=BOS, discount=discount)
    observed, targets, future, resets, carry = _transition()
    bag = objective.future_bag(targets, future)
    expected_bag = _hand_bag(discount)
    torch.testing.assert_close(bag, expected_bag)
    torch.testing.assert_close(bag.sum(-1), torch.ones(5, device="cuda"))
    # A row whose target closes the document is judged on that BOS alone, and
    # nothing after a closing BOS ever enters a bag.
    assert torch.equal(bag[2], F.one_hot(torch.tensor(BOS, device="cuda"), VOCAB).float())
    assert bag[1, 3] == 0 and bag[3, 3] == 1 / (1 + discount + discount ** 2)
    if discount == 0.0:
        torch.testing.assert_close(bag, F.one_hot(targets, VOCAB).float())

    loss, next_carry, stats = objective(observed, carry, targets, future, resets)
    oracle = _oracle(model, observed, targets, future, resets, carry, discount, 0.0)
    torch.testing.assert_close(next_carry, oracle["next_carry"].detach())
    assert not next_carry.requires_grad
    torch.testing.assert_close(loss, oracle["ce"] + oracle["bag_loss"])
    entropy = -(torch.xlogy(expected_bag, expected_bag)).sum(-1).mean()
    expected_stats = torch.stack((
        oracle["ce"], oracle["bag_loss"], oracle["hidden_bag_loss"], entropy, oracle["gate"].mean(),
        F.cosine_similarity(oracle["next_carry"], oracle["hidden"], dim=-1).mean(),
    )).detach()
    assert stats.shape == (len(FUTURE_BAG_STATISTICS),) and stats.dtype == torch.float32
    torch.testing.assert_close(stats, expected_stats)

    expected_backbone = torch.autograd.grad(
        oracle["ce"], (model.core, model.readout), retain_graph=True, allow_unused=True)
    writer_parameters = list(model.writer.parameters())
    expected_writer = torch.autograd.grad(oracle["bag_loss"], writer_parameters)
    loss.backward()
    # CE owns the backbone; the frozen head's input gradient owns the writer;
    # the norm gains are only on the frozen path, and the hidden's reference
    # bag loss is gradient-free. Nothing reaches the incoming carry.
    torch.testing.assert_close(model.core.grad, expected_backbone[0])
    torch.testing.assert_close(model.readout.grad, expected_backbone[1])
    assert model.gains.grad is None
    for parameter, expected in zip(writer_parameters, expected_writer):
        assert torch.count_nonzero(expected) > 0
        torch.testing.assert_close(parameter.grad, expected)
    assert carry.grad is None


def test_backbone_future_weight_scales_the_bag_gradient_into_the_backbone():
    weight = 0.5
    model = _AnalyticModel()
    objective = FutureBagObjective(model, bos_id=BOS, discount=0.9, backbone_future_weight=weight)
    observed, targets, future, resets, carry = _transition()
    loss, _, _ = objective(observed, carry, targets, future, resets)
    live = _oracle(model, observed, targets, future, resets, carry, 0.9, 1.0)
    backbone = (model.core, model.readout, model.gains)
    expected_ce = torch.autograd.grad(live["ce"], backbone, retain_graph=True, allow_unused=True)
    expected_bag = torch.autograd.grad(live["bag_loss"], backbone)
    writer_parameters = list(model.writer.parameters())
    loss.backward()
    for parameter, ce_gradient, bag_gradient in zip(backbone, expected_ce, expected_bag):
        expected = weight * bag_gradient if ce_gradient is None else ce_gradient + weight * bag_gradient
        torch.testing.assert_close(parameter.grad, expected)
    # The writer's own gradient is never scaled.
    model.zero_grad(set_to_none=True)
    unscaled = _oracle(model, observed, targets, future, resets, carry, 0.9, 0.0)
    expected_writer = torch.autograd.grad(unscaled["bag_loss"], writer_parameters)
    loss, _, _ = objective(observed, carry, targets, future, resets)
    loss.backward()
    for parameter, expected in zip(writer_parameters, expected_writer):
        torch.testing.assert_close(parameter.grad, expected)
    assert carry.grad is None


def test_reset_lanes_read_a_zero_carry_and_the_writer_sees_the_reset_state():
    model = _AnalyticModel()
    objective = FutureBagObjective(model, bos_id=BOS, discount=0.5)
    observed, targets, future, resets, carry = _transition()
    loss, next_carry, _ = objective(observed, carry, targets, future, resets)
    perturbed = carry.detach().clone()
    perturbed[resets] *= 3.0
    perturbed[resets, 0] += 1.0
    with torch.no_grad():
        perturbed_loss, perturbed_carry, _ = objective(observed, perturbed, targets, future, resets)
    torch.testing.assert_close(perturbed_loss, loss.detach())
    torch.testing.assert_close(perturbed_carry, next_carry)


def test_plain_ce_objective_carries_the_hidden_and_reports_only_ce():
    model = _AnalyticModel(use_writer=False)
    objective = StreamingObjective(model)
    observed, targets, _, resets, carry = _transition()
    loss, next_carry, stats = objective(observed, carry, targets, resets)
    state = carry.detach().masked_fill(resets[:, None], 0)
    hidden = (F.one_hot(observed % 2, 2).float() + state) @ model.core
    torch.testing.assert_close(loss, F.cross_entropy(hidden @ model.readout, targets))
    torch.testing.assert_close(next_carry, hidden.detach())
    assert stats.shape == (len(CE_STATISTICS),)
    torch.testing.assert_close(stats[0], loss.detach())
    with pytest.raises(ValueError):
        StreamingObjective(_AnalyticModel())
    with pytest.raises(ValueError):
        FutureBagObjective(model)


def test_temporal_reference_backpropagates_through_the_page_and_truncates_at_its_edge():
    model = _AnalyticModel(use_writer=False)
    objective = TemporalReferenceObjective(model)
    inputs = torch.tensor([[2, 3], [0, 1], [2, 3]], device="cuda")
    targets = torch.tensor([[3, 0], [1, 2], [BOS, 2]], device="cuda")
    resets = torch.tensor([[False, False], [False, True], [False, False]], device="cuda")
    incoming = torch.tensor([[0.3, -0.2], [0.5, 0.1]], device="cuda", requires_grad=True)
    loss, final_carry, stats = objective(inputs, incoming, targets, resets)

    state = incoming.detach()
    total = 0.0
    for tick in range(3):
        state = state.masked_fill(resets[tick][:, None], 0)
        hidden = (F.one_hot(inputs[tick] % 2, 2).float() + state) @ model.core
        total = total + F.cross_entropy(hidden @ model.readout, targets[tick])
        state = hidden
    expected = total / 3
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(final_carry, state.detach())
    assert not final_carry.requires_grad
    assert stats.shape == (len(CE_STATISTICS),)
    expected_core = torch.autograd.grad(expected, model.core)[0]
    loss.backward()
    torch.testing.assert_close(model.core.grad, expected_core)
    assert incoming.grad is None
    with pytest.raises(ValueError):
        TemporalReferenceObjective(_AnalyticModel())
