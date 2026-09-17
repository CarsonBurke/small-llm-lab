"""Online scalar TD mathematics on CUDA; execute through mlq.

An analytic two-coordinate recurrence gives independent gradient oracles for
teacher detachment and document boundaries, not a replacement training model.
"""

import pytest
import torch
import torch.nn.functional as F

from pretraining.future_credit_stream.objective import StreamingObjective


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class _AnalyticModel(torch.nn.Module):
    def __init__(self, use_critic=True):
        super().__init__()
        self.core = torch.nn.Parameter(torch.tensor([[0.7, -0.2], [0.3, 0.6]], device="cuda"))
        self.readout = torch.nn.Parameter(torch.tensor([[0.5, -0.1, 0.4], [-0.3, 0.8, 0.2]], device="cuda"))
        self.critic = torch.nn.Linear(2, 1, device="cuda") if use_critic else None
        if self.critic is not None:
            with torch.no_grad():
                self.critic.weight.copy_(torch.tensor([[0.6, -0.2]], device="cuda"))
                self.critic.bias.fill_(0.4)

    def forward(self, observed, carry):
        features = F.one_hot(observed % 2, 2).float() + carry.detach()
        hidden = features @ self.core
        return hidden @ self.readout, hidden

    def value(self, hidden, detach_weights=False):
        weight, bias = self.critic.weight, self.critic.bias
        if detach_weights:
            weight, bias = weight.detach(), bias.detach()
        return F.linear(hidden.square(), weight, bias).squeeze(-1)


@pytest.mark.parametrize("discount", [0.0, 0.65, 1.0])
def test_online_td_per_lane_targets_and_gradient_ownership_at_both_boundaries(discount):
    model = _AnalyticModel()
    objective = StreamingObjective(model, discount=discount, bos_id=1)
    observed = torch.tensor([2, 1, 4, 1], device="cuda")
    targets = torch.tensor([2, 2, 1, 1], device="cuda")
    resets = torch.tensor([False, True, False, True], device="cuda")
    carry = torch.tensor([[0.2, -0.4], [2.0, -1.0], [-0.5, 0.8], [1.5, 0.3]],
                         device="cuda", requires_grad=True)
    loss, hidden, stats = objective(observed, carry, targets, resets, True)

    features = F.one_hot(observed % 2, 2).float() + carry.detach().masked_fill(resets[:, None], 0)
    expected_hidden = features @ model.core
    logits = expected_hidden @ model.readout
    ce = F.cross_entropy(logits, targets, reduction="none")
    actor = (expected_hidden.square() @ model.critic.weight.detach().squeeze(0)
             + model.critic.bias.detach()).masked_fill(targets == 1, 0)
    local = ce.mean() + discount * actor.mean()
    expected_core, expected_readout = torch.autograd.grad(local, (model.core, model.readout))

    # Fit the ORIGINAL state even on reset lanes: zeroing it first would stop
    # the critic learning that those preceding-document terminal states end.
    previous = carry.detach().square() @ model.critic.weight.detach().squeeze(0) + model.critic.bias.detach()
    teacher = (ce.detach() + discount * actor.detach()).masked_fill(resets, 0)
    residual = previous - teacher
    expected_weight = (residual[:, None] * carry.detach().square()).mean(0, keepdim=True)
    expected_bias = residual.mean().reshape(1)
    regression = 0.5 * residual.square().mean()
    torch.testing.assert_close(hidden, expected_hidden)
    torch.testing.assert_close(loss, local.detach() + regression)
    torch.testing.assert_close(stats, torch.stack((ce.detach().mean(), regression,
                                                  previous.mean(), teacher.mean())))
    assert stats.dtype == torch.float32 and not stats.requires_grad
    loss.backward()
    # These equalities fail if either TD teacher is live or either value branch
    # leaks gradients to the wrong owner. In particular CE is differentiated once.
    torch.testing.assert_close(model.core.grad, expected_core)
    torch.testing.assert_close(model.readout.grad, expected_readout)
    torch.testing.assert_close(model.critic.weight.grad, expected_weight)
    torch.testing.assert_close(model.critic.bias.grad, expected_bias)
    assert carry.grad is None


@pytest.mark.parametrize("use_critic", [True, False])
def test_first_observation_and_ce_arm_have_no_previous_state_fit(use_critic):
    model = _AnalyticModel(use_critic)
    objective = StreamingObjective(model, discount=0.7, bos_id=1)
    observed = torch.tensor([2, 1], device="cuda")
    targets = torch.tensor([2, 1], device="cuda")
    resets = observed == 1
    carry = torch.tensor([[0.2, -0.4], [1.0, -2.0]], device="cuda", requires_grad=True)
    # CE also ignores has_previous=True; no critic or teacher is needed.
    loss, hidden, stats = objective(observed, carry, targets, resets, not use_critic)
    ce = F.cross_entropy(hidden @ model.readout, targets, reduction="none")
    expected = ce.mean()
    if use_critic:
        actor = model.value(hidden, detach_weights=True).masked_fill(targets == 1, 0)
        expected = expected + 0.7 * actor.mean()
    expected_core, expected_readout = torch.autograd.grad(expected, (model.core, model.readout), retain_graph=True)
    loss.backward()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(model.core.grad, expected_core)
    torch.testing.assert_close(model.readout.grad, expected_readout)
    torch.testing.assert_close(stats, torch.cat((ce.detach().mean().reshape(1),
                                                torch.zeros(3, device="cuda"))), rtol=0, atol=0)
    assert carry.grad is None and not stats.requires_grad
    if use_critic:
        assert all(parameter.grad is None for parameter in model.critic.parameters())


def test_reset_terminal_fit_is_independent_of_new_document_target():
    model = _AnalyticModel()
    objective = StreamingObjective(model, discount=0.8, bos_id=1)
    observed = torch.ones(2, device="cuda", dtype=torch.long)
    resets = torch.ones(2, device="cuda", dtype=torch.bool)
    carry = torch.tensor([[0.7, -0.4], [1.0, 0.8]], device="cuda")

    def run(targets):
        model.zero_grad(set_to_none=True)
        loss, hidden, stats = objective(observed, carry, targets, resets, True)
        loss.backward()
        return (hidden.detach(), stats,
                {name: parameter.grad.clone() for name, parameter in model.named_parameters()})

    base = run(torch.tensor([2, 0], device="cuda"))
    # Change only one lane to avoid a permutation-invariant batch mean.
    changed = run(torch.tensor([0, 0], device="cuda"))
    torch.testing.assert_close(changed[0], base[0], rtol=0, atol=0)
    torch.testing.assert_close(changed[1][1:], base[1][1:], rtol=0, atol=0)
    for name in ("critic.weight", "critic.bias"):
        torch.testing.assert_close(changed[2][name], base[2][name], rtol=0, atol=0)
        assert torch.count_nonzero(base[2][name]) > 0
    assert (changed[2]["readout"] - base[2]["readout"]).norm() > 0.01
