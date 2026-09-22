"""Score-function boundary contracts for the explicit frozen-controller ablation."""
from types import SimpleNamespace
import copy
from contextlib import contextmanager

import torch
import pytest
import scripts.train_minicpm_latent_controller as controller_runtime

from postraining.vapo.policy import FIRST_THOUGHT, CONTINUE_THOUGHT, STOP_THINKING
from scripts.train_minicpm_latent_controller import (
    WhitenedGaussianController, NormalizedStopGate, controller_scores,
)


def test_controller_scores_only_observed_gate_and_gaussian_actions():
    torch.manual_seed(21)
    policy = SimpleNamespace(
        transition=WhitenedGaussianController(6, 1.0),
        thinking_gate=NormalizedStopGate(6, 0.25),
    )
    states = torch.randn(3, 6)
    raw = states[:2] + torch.randn(2, 6)/6**0.5
    kinds = torch.tensor([FIRST_THOUGHT, CONTINUE_THOUGHT, STOP_THINKING])
    noise_sum_squares = torch.zeros((), dtype=torch.float64)
    scores = controller_scores(policy, states, kinds, raw, noise_sum_squares=noise_sum_squares)
    torch.testing.assert_close(noise_sum_squares, (raw-states[:2]).square().sum(dtype=torch.float64))
    gaussian = torch.distributions.Independent(
        torch.distributions.Normal(states[:2], 1/6**0.5), 1,
    ).log_prob(raw)
    expected = torch.stack((gaussian[0], gaussian[1]+torch.tensor(0.75).log(),
                            torch.tensor(0.25).log()))
    torch.testing.assert_close(scores, expected)
    scores[0].backward(retain_graph=True)
    assert torch.count_nonzero(policy.thinking_gate.head.weight.grad) == 0
    assert torch.count_nonzero(policy.transition.mean_head.weight.grad) > 0
    for module in (policy.transition, policy.thinking_gate):
        module.zero_grad(set_to_none=True)
    scores[2].backward()
    assert torch.count_nonzero(policy.transition.mean_head.weight.grad) == 0
    assert torch.count_nonzero(policy.thinking_gate.head.weight.grad) > 0


def test_controller_low_noise_mean_parameterization_preserves_initial_distribution():
    controller = WhitenedGaussianController(12, 0.5)
    states = torch.randn(5, 12)
    torch.testing.assert_close(controller.predict_mean(states), states, rtol=0, atol=0)
    assert controller.component_std == 0.5/12**0.5
    with torch.no_grad():
        controller.mean_head.bias.fill_(0.125)
    expected = states + (0.5/12**0.5)*0.125
    torch.testing.assert_close(controller.predict_mean(states), expected)


def test_controller_score_accumulation_does_not_depend_on_replay_partition():
    torch.manual_seed(22)
    policy = SimpleNamespace(
        transition=WhitenedGaussianController(6, 1.0),
        thinking_gate=NormalizedStopGate(6, 0.25),
    )
    states = torch.randn(4, 6)
    raw = states[:3] + torch.randn(3, 6)/6**0.5
    kinds = torch.tensor([FIRST_THOUGHT, CONTINUE_THOUGHT, CONTINUE_THOUGHT, STOP_THINKING])
    advantages = torch.tensor([0.7, -0.2, 0.4, -0.1])
    parameters = tuple(policy.transition.parameters())+tuple(policy.thinking_gate.parameters())
    whole = -(controller_scores(policy, states, kinds, raw)*advantages).sum()
    reference = torch.autograd.grad(whole, parameters)
    first = -(controller_scores(policy, states[:2], kinds[:2], raw[:2])*advantages[:2]).sum()
    last = -(controller_scores(policy, states[2:], kinds[2:], raw[2:])*advantages[2:]).sum()
    partitioned = torch.autograd.grad(first+last, parameters)
    for actual, expected in zip(partitioned, reference, strict=True):
        torch.testing.assert_close(actual, expected)


def test_rejected_controller_step_restores_parameters_and_adam_state(monkeypatch):
    torch.manual_seed(24)
    policy = SimpleNamespace(
        transition=WhitenedGaussianController(6, 1.0),
        thinking_gate=NormalizedStopGate(6, 0.25),
    )
    parameters = tuple(policy.transition.parameters())+tuple(policy.thinking_gate.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=0.1, weight_decay=0)
    states = torch.randn(3, 6)
    raw = states[:2]+torch.randn(2, 6)/6**0.5
    kinds = torch.tensor([FIRST_THOUGHT, CONTINUE_THOUGHT, STOP_THINKING])
    # Populate moments so rollback must recover tensors and step counters.
    controller_scores(policy, states, kinds, raw).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    old_scores = controller_scores(policy, states, kinds, raw).detach()
    before_parameters = [parameter.detach().clone() for parameter in parameters]
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    monkeypatch.setattr(controller_runtime, "controller_batches", lambda records, batch_size: iter([
        (states, kinds, raw, torch.tensor([0.3, -0.7, 0.5]), old_scores),
    ]))
    monkeypatch.setattr(controller_runtime, "controller_kl", lambda *args: {
        "max_gaussian_kl": float("inf"), "max_joint_kl": float("inf"),
    })
    with pytest.raises(RuntimeError, match="state restored"):
        controller_runtime.controller_update(policy, optimizer, [object()], batch_size=4, max_kl=0.02)
    for actual, expected in zip(parameters, before_parameters, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    restored = optimizer.state_dict()
    assert restored["param_groups"] == before_optimizer["param_groups"]
    for parameter_id, old_state in before_optimizer["state"].items():
        for name, old_value in old_state.items():
            torch.testing.assert_close(restored["state"][parameter_id][name], old_value, rtol=0, atol=0)


@pytest.mark.parametrize('sampling', ['independent', 'antithetic'])
@pytest.mark.parametrize('eval_only', [False, True])
def test_recorded_sampling_cannot_switch_on_resume(sampling, eval_only):
    args = controller_runtime.parser().parse_args(['--output', 'unused', '--sampling', sampling])
    saved = {'args': vars(args).copy()}
    args.eval_only = eval_only
    controller_runtime.validate_resume_configuration(saved, args)
    args.sampling = 'antithetic' if sampling == 'independent' else 'independent'
    with pytest.raises(ValueError, match='sampling'):
        controller_runtime.validate_resume_configuration(saved, args)


@pytest.mark.parametrize('sampling', ['independent', 'antithetic'])
def test_legacy_checkpoint_can_bootstrap_sampling_without_mutating_other_contracts(sampling):
    args = controller_runtime.parser().parse_args(['--output', 'unused', '--sampling', sampling])
    saved = {'args': vars(args).copy()}
    del saved['args']['sampling']
    controller_runtime.validate_resume_configuration(saved, args)
    args.thought_sigma *= 2
    with pytest.raises(ValueError, match='thought_sigma'):
        controller_runtime.validate_resume_configuration(saved, args)


def test_topk_resume_preserves_training_policy_but_allows_isolated_evaluation():
    args = controller_runtime.parser().parse_args(['--output', 'unused'])
    saved = {'args': vars(args).copy()}
    del saved['args']['top_k']
    original = copy.deepcopy(saved)
    controller_runtime.validate_resume_configuration(saved, args)
    assert args.top_k == 20

    args.top_k = 50
    with pytest.raises(ValueError, match='checkpoint used top-k=20'):
        controller_runtime.validate_resume_configuration(saved, args)

    args.eval_only = True
    controller_runtime.validate_resume_configuration(saved, args)
    assert args.top_k == 50
    assert saved == original

    args.eval_only = False
    args.top_k = None
    saved['args']['top_k'] = 50
    controller_runtime.validate_resume_configuration(saved, args)
    assert args.top_k == 50


@pytest.mark.parametrize('sampling', ['independent', 'antithetic'])
@pytest.mark.parametrize('fail', [False, True])
def test_evaluation_restores_training_rng_and_sampling_after_success_or_failure(monkeypatch, sampling, fail):
    cuda_rng = torch.tensor([12], dtype=torch.uint8)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda: cuda_rng.clone())

    def set_cuda_rng(value):
        cuda_rng.copy_(value)

    monkeypatch.setattr(torch.cuda, 'set_rng_state', set_cuda_rng)

    class PairedEngine:
        paired = True

        @contextmanager
        def stock_sampling(self):
            previous = self.paired
            self.paired = False
            try:
                yield
            finally:
                self.paired = previous

    # The stock engine deliberately has no paired-only context-manager API.
    engine = PairedEngine() if sampling == 'antithetic' else object()
    cpu_before = torch.get_rng_state().clone()
    cuda_before = cuda_rng.clone()

    def evaluate():
        with controller_runtime.evaluation_sampling(engine, sampling):
            if sampling == 'antithetic':
                assert not engine.paired
            torch.rand(8)
            cuda_rng.fill_(99)
            if fail:
                raise RuntimeError('heldout failure')

    if fail:
        with pytest.raises(RuntimeError, match='heldout failure'):
            evaluate()
    else:
        evaluate()
    torch.testing.assert_close(torch.get_rng_state(), cpu_before, rtol=0, atol=0)
    torch.testing.assert_close(cuda_rng, cuda_before, rtol=0, atol=0)
    if sampling == 'antithetic':
        assert engine.paired


@pytest.mark.parametrize('batch_size', [1, 4])
def test_sampled_noise_metrics_use_preupdate_means_and_only_gaussian_actions(monkeypatch, batch_size):
    policy = SimpleNamespace(
        transition=WhitenedGaussianController(4, 0.5),
        thinking_gate=NormalizedStopGate(4, 0.25),
    )
    with torch.no_grad():
        policy.transition.mean_head.bias.fill_(0.4)
    states = torch.zeros(4, 4)
    epsilon = torch.tensor([[1., -1., 2., -2.], [0., 0., 0., 0.], [3., -3., 1., -1.]])
    mean_before = policy.transition.predict_mean(states[:3]).detach()
    raw = mean_before + policy.transition.component_std * epsilon
    kinds = torch.tensor([FIRST_THOUGHT, CONTINUE_THOUGHT, CONTINUE_THOUGHT, STOP_THINKING])
    old_scores = controller_scores(policy, states, kinds, raw).detach()
    advantages = torch.tensor([0.3, -0.7, 0.5, 0.2])

    def batches(records, size):
        for start in range(0, len(states), size):
            stop = min(start+size, len(states))
            yield (states[start:stop], kinds[start:stop], raw[start:min(stop, len(raw))],
                   advantages[start:stop], old_scores[start:stop])

    monkeypatch.setattr(controller_runtime, 'controller_batches', batches)
    monkeypatch.setattr(controller_runtime, 'controller_kl', lambda *args: {
        'max_gaussian_kl': 0.0, 'max_joint_kl': 0.0,
    })
    optimizer = torch.optim.AdamW(
        tuple(policy.transition.parameters())+tuple(policy.thinking_gate.parameters()),
        lr=0.1, weight_decay=0,
    )
    metrics = controller_runtime.controller_update(
        policy, optimizer, [object()], batch_size=batch_size, max_kl=0.02,
    )
    second_moment = epsilon.square().mean().item()
    assert metrics['sampled_gaussian_actions'] == 3
    assert metrics['sampled_epsilon_second_moment'] == pytest.approx(second_moment)
    assert metrics['sampled_epsilon_rms'] == pytest.approx(second_moment**0.5)
    assert metrics['sampled_noise_component_rms'] == pytest.approx(second_moment**0.5 * 0.25)
    assert metrics['sampled_noise_vector_rms'] == pytest.approx(second_moment**0.5 * 0.5)
    assert metrics['behavior_logprob_max_error'] == 0
    assert not torch.equal(policy.transition.predict_mean(states[:3]).detach(), mean_before)
