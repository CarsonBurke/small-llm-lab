from __future__ import annotations

import math

import torch

from fresh_lejepa_train_v1_probe_shared_rms_pope import FreshLeJEPASharedRMSV1PoPE
from postraining.latent_thought import (
    EMIT,
    THINK,
    GaussianTransitionHead,
    LatentThoughtModel,
    ThinkEmitGate,
    ThoughtAdapter,
)
from postraining.model_io import _pope_construction

KWARGS = dict(
    vocab_size=32, num_layers=3, model_dim=32, num_heads=4, num_kv_heads=2,
    mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.01,
    logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
)


def _pope_model() -> FreshLeJEPASharedRMSV1PoPE:
    with _pope_construction():
        return FreshLeJEPASharedRMSV1PoPE(**KWARGS).eval()


def test_token_step_matches_backbone_generation_step():
    torch.manual_seed(3)
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone).eval()
    ids = torch.randint(0, 32, (2, 9))
    caches_a = backbone.make_generation_cache(2, ids.size(1), torch.device("cpu"))
    caches_b = backbone.make_generation_cache(2, ids.size(1), torch.device("cpu"))
    with torch.no_grad():
        for position in range(ids.size(1)):
            logits, _, caches_a = backbone.generation_step(
                ids[:, position], caches_a, position
            )
            output = wrapper.token_step(ids[:, position], caches_b, position)
            caches_b = output.caches
            torch.testing.assert_close(output.logits, logits)


def test_cached_generation_matches_full_forward_beyond_pretraining_context():
    torch.manual_seed(5)
    length = 1050  # crosses the 1024-token pretraining context
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone).eval()
    ids = torch.randint(0, 32, (1, length))
    with torch.no_grad():
        full = backbone.policy_logits(ids)
        caches = backbone.make_generation_cache(1, length, torch.device("cpu"))
        stepped = []
        for position in range(length):
            output = wrapper.token_step(ids[:, position], caches, position)
            caches = output.caches
            stepped.append(output.logits)
    stepped = torch.stack(stepped, dim=1)
    torch.testing.assert_close(stepped, full, rtol=2e-4, atol=2e-4)


def test_gate_zero_init_is_exactly_uniform():
    gate = ThinkEmitGate(16)
    belief = torch.randn(4, 16)
    assert torch.all(gate.emit_logit(belief) == 0)
    log_prob = gate.log_prob(torch.tensor([THINK, EMIT, THINK, EMIT]), belief)
    torch.testing.assert_close(log_prob, torch.full((4,), math.log(0.5)))
    torch.testing.assert_close(gate.entropy(belief), torch.full((4,), math.log(2.0)))


def test_gate_sample_log_prob_recomputes_identically():
    torch.manual_seed(11)
    gate = ThinkEmitGate(16)
    with torch.no_grad():
        gate.head.weight.normal_(std=0.5)
        gate.head.bias.normal_()
    belief = torch.randn(64, 16)
    generator = torch.Generator().manual_seed(7)
    action, log_prob = gate.sample(belief, generator=generator)
    torch.testing.assert_close(gate.log_prob(action, belief), log_prob)
    assert set(action.unique().tolist()) <= {THINK, EMIT}


def test_transition_log_prob_matches_torch_distributions():
    torch.manual_seed(13)
    head = GaussianTransitionHead(8)
    with torch.no_grad():
        head.log_std_head.weight.normal_(std=0.1)
    belief = torch.randn(5, 8)
    mean = torch.randn(5, 8)
    generator = torch.Generator().manual_seed(21)
    sample, log_prob = head.sample(mean, belief, generator=generator)
    torch.testing.assert_close(head.log_prob(sample, mean, belief), log_prob)
    reference = torch.distributions.Normal(mean, head.log_std(belief).exp())
    torch.testing.assert_close(log_prob, reference.log_prob(sample).sum(-1))
    torch.testing.assert_close(head.entropy(belief), reference.entropy().sum(-1))


def test_transition_log_std_is_clamped_and_initialized_at_prediction_scale():
    torch.manual_seed(15)
    head = GaussianTransitionHead(8, log_std_init=-0.5)
    torch.testing.assert_close(
        head.log_std(torch.zeros(1, 8)), torch.full((1, 8), -0.5)
    )
    # The zero-init weight makes the head input-independent, so the clamp is
    # only exercised with a randomized weight.
    with torch.no_grad():
        head.log_std_head.weight.normal_(std=1.0)
    belief = 1000.0 * torch.randn(64, 8)
    log_std = head.log_std(belief)
    assert torch.all(log_std >= head.log_std_min)
    assert torch.all(log_std <= head.log_std_max)
    assert float(log_std.min()) == head.log_std_min
    assert float(log_std.max()) == head.log_std_max


def test_beta_nll_gradient_is_a_drop_in_for_the_mse_it_replaces():
    # With constant log-std s and beta=0.5, d(beta_nll)/d(mean) must equal
    # e^{-s}/2 times d(mse)/d(mean) per element — NOT model_dim/2 times,
    # which a sum-over-dims reduction would silently produce.
    torch.manual_seed(101)
    dim = 512
    head = GaussianTransitionHead(dim, log_std_init=-0.5)
    belief = torch.randn(4, 7, dim)
    target = torch.randn(4, 7, dim)
    mean_nll = torch.randn(4, 7, dim, requires_grad=True)
    head.beta_nll(target, mean_nll, belief, beta=0.5).backward()
    mean_mse = mean_nll.detach().clone().requires_grad_(True)
    torch.nn.functional.mse_loss(mean_mse, target).backward()
    ratio = mean_nll.grad / mean_mse.grad
    expected = math.exp(0.5) / 2.0
    torch.testing.assert_close(
        ratio, torch.full_like(ratio, expected), rtol=1e-4, atol=1e-4
    )


def test_beta_nll_scale_weight_is_detached_from_the_log_std_gradient():
    # Seitzer et al.: the sigma^(2*beta) factor must not contribute to the
    # log-std gradient. With beta=0.5 and error^2 == sigma^2 per dim, the
    # attached NLL derivative (1 - err^2/sigma^2) is exactly zero, so ANY
    # remaining gradient would come from a leaky scale factor.
    torch.manual_seed(103)
    head = GaussianTransitionHead(6, log_std_init=-0.5)
    belief = torch.randn(32, 6)
    mean = torch.randn(32, 6)
    sigma = math.exp(-0.5)
    target = mean + sigma * (2 * torch.randint(0, 2, mean.shape).float() - 1)
    head.beta_nll(target, mean, belief, beta=0.5).backward()
    torch.testing.assert_close(
        head.log_std_head.bias.grad, torch.zeros_like(head.log_std_head.bias)
    )


def test_transition_beta_nll_trains_mean_free_head_toward_target():
    torch.manual_seed(17)
    head = GaussianTransitionHead(4)
    belief = torch.randn(64, 4)
    mean = torch.zeros(64, 4)
    near = head.beta_nll(0.1 * torch.randn(64, 4), mean, belief)
    far = head.beta_nll(3.0 * torch.randn(64, 4), mean, belief)
    assert float(near) < float(far)
    far.backward()
    assert head.log_std_head.weight.grad is not None
    assert torch.isfinite(head.log_std_head.weight.grad).all()


def test_adapter_zero_init_passes_thought_through():
    torch.manual_seed(19)
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone)
    thought = torch.randn(2, 32)
    injected = wrapper.thought_input(thought)
    assert injected.shape == (2, 1, 32)
    torch.testing.assert_close(
        injected.squeeze(1), thought.to(injected.dtype), rtol=0, atol=0
    )


def test_thought_step_advances_state_without_rendering_machinery_changes():
    torch.manual_seed(23)
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone).eval()
    ids = torch.randint(0, 32, (1, 4))
    caches = backbone.make_generation_cache(1, 8, torch.device("cpu"))
    with torch.no_grad():
        output = None
        for position in range(ids.size(1)):
            output = wrapper.token_step(ids[:, position], caches, position)
            caches = output.caches
        assert output is not None
        sample, _ = wrapper.transition.sample(output.predicted, output.belief)
        thought_output = wrapper.step(
            wrapper.thought_input(sample), caches, ids.size(1)
        )
    assert thought_output.logits.shape == output.logits.shape
    assert not torch.allclose(thought_output.belief, output.belief)


def test_new_parameters_exclude_backbone_and_strict_load_round_trips():
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone)
    backbone_ids = {id(parameter) for parameter in backbone.parameters()}
    new = list(wrapper.new_parameters())
    assert new
    assert all(id(parameter) not in backbone_ids for parameter in new)
    wrapper.load_backbone_checkpoint(
        {key: value.clone() for key, value in backbone.state_dict().items()}
    )
