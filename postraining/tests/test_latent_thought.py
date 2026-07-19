from __future__ import annotations

import math

import pytest
import torch

from fresh_lejepa_train_v1_probe_shared_rms_pope import FreshLeJEPASharedRMSV1PoPE
from postraining.latent_thought import (
    EMIT,
    THINK,
    GaussianTransitionHead,
    LatentThoughtModel,
    RENDERER_FEATURES_SCHEMA,
    ThinkEmitGate,
    ThoughtAdapter,
    validate_renderer_checkpoint,
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


def test_token_step_matches_full_forward_with_belief_renderer():
    torch.manual_seed(3)
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone).eval()
    ids = torch.randint(0, 32, (2, 9))
    caches = backbone.make_generation_cache(2, ids.size(1), torch.device("cpu"))
    with torch.no_grad():
        token_latents = backbone.embed_tokens(ids)
        beliefs = backbone.temporal_belief_from_token_latent(token_latents)
        expected = backbone.logits_from_features(
            wrapper.renderer_features(token_latents, beliefs)
        )
        for position in range(ids.size(1)):
            output = wrapper.token_step(ids[:, position], caches, position)
            caches = output.caches
            torch.testing.assert_close(output.logits, expected[:, position])


def test_cached_generation_matches_full_forward_beyond_pretraining_context():
    torch.manual_seed(5)
    length = 1050  # crosses the 1024-token pretraining context
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone).eval()
    ids = torch.randint(0, 32, (1, length))
    with torch.no_grad():
        token_latents = backbone.embed_tokens(ids)
        beliefs = backbone.temporal_belief_from_token_latent(token_latents)
        full = backbone.logits_from_features(
            wrapper.renderer_features(token_latents, beliefs)
        )
        caches = backbone.make_generation_cache(1, length, torch.device("cpu"))
        stepped = []
        for position in range(length):
            output = wrapper.token_step(ids[:, position], caches, position)
            caches = output.caches
            stepped.append(output.logits)
    stepped = torch.stack(stepped, dim=1)
    torch.testing.assert_close(stepped, full, rtol=2e-4, atol=2e-4)


def test_renderer_logits_do_not_depend_on_prediction_projector():
    torch.manual_seed(7)
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone).eval()
    ids = torch.randint(0, 32, (2, 7))
    with torch.no_grad():
        backbone.policy_probe.output.weight.normal_(std=0.1)
        input_latent = backbone.embed_tokens(ids)
        belief = backbone.temporal_belief_from_token_latent(input_latent)
        predicted_before = backbone.prediction_latent(belief)
        logits_before = wrapper.policy_logits(ids)
        for parameter in backbone.prediction_projector.parameters():
            parameter.add_(torch.randn_like(parameter))
        predicted_after = backbone.prediction_latent(belief)
        logits_after = wrapper.policy_logits(ids)
    assert not torch.equal(predicted_before, predicted_after)
    torch.testing.assert_close(logits_after, logits_before)


def test_dense_step_projection_gets_no_renderer_gradient():
    torch.manual_seed(9)
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone).eval()
    with torch.no_grad():
        backbone.policy_probe.output.weight.normal_(std=0.1)
    ids = torch.randint(0, 32, (2,))
    caches = wrapper.make_generation_cache(2, 1, torch.device("cpu"))
    projected = []
    handle = backbone.prediction_projector.register_forward_pre_hook(
        lambda _module, inputs: projected.append(tuple(inputs[0].shape))
    )
    try:
        output = wrapper.token_step(ids, caches, 0)
    finally:
        handle.remove()
    output.logits.float().square().mean().backward()
    assert projected == [(2, 1, 32)]
    assert all(
        parameter.grad is None
        for parameter in backbone.prediction_projector.parameters()
    )
    assert float(backbone.blocks[0].attn.proj.weight.grad.abs().sum()) > 0.0


def test_renderer_checkpoint_schema_rejects_old_semantics():
    validate_renderer_checkpoint(
        {"renderer_features_schema": RENDERER_FEATURES_SCHEMA}, "current.pt"
    )
    with pytest.raises(ValueError, match="Old or untagged VAPO checkpoints"):
        validate_renderer_checkpoint({}, "old.pt")
    with pytest.raises(ValueError, match="predicted/v1"):
        validate_renderer_checkpoint(
            {"renderer_features_schema": "input_latent+predicted/v1"}, "old.pt"
        )


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
    head = GaussianTransitionHead(8, log_sigma=-0.5)
    mean = torch.randn(5, 8)
    generator = torch.Generator().manual_seed(21)
    sample, log_prob = head.sample(mean, generator=generator)
    torch.testing.assert_close(head.log_prob(sample, mean), log_prob)
    reference = torch.distributions.Normal(mean, math.exp(-0.5))
    torch.testing.assert_close(log_prob, reference.log_prob(sample).sum(-1))
    torch.testing.assert_close(
        head.per_dim_log_prob(sample, mean), reference.log_prob(sample)
    )


def test_transition_sigma_is_a_fixed_buffer_with_no_parameters():
    head = GaussianTransitionHead(8, log_sigma=-0.5)
    assert list(head.parameters()) == []
    torch.testing.assert_close(head.log_sigma, torch.tensor(-0.5))
    assert "log_sigma" in head.state_dict()


def test_thought_policy_gradient_flows_through_the_mean():
    # v2 (full-model RL): the policy gradient must reach the prediction
    # path — per_dim_log_prob differentiates through the passed mean.
    torch.manual_seed(15)
    head = GaussianTransitionHead(8)
    mean = torch.randn(5, 8, requires_grad=True)
    sample = (mean + 0.3).detach()
    head.per_dim_log_prob(sample, mean).sum().backward()
    assert mean.grad is not None
    # d/dmean of -0.5*((s-m)/sigma)^2 is (s-m)/sigma^2, positive here.
    assert torch.all(mean.grad > 0)


def test_adapter_zero_init_passes_thought_through():
    torch.manual_seed(19)
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone)
    thought = torch.randn(2, 32)
    injected = wrapper.thought_input(thought)
    assert injected.shape == (2, 1, 32)
    # Zero-init correction: the injection is exactly the raw thought — its
    # magnitude (the model's confidence) reaches the trunk unmodified.
    torch.testing.assert_close(
        injected.squeeze(1), thought.to(injected.dtype), rtol=1e-5, atol=1e-6
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
        sample, _ = wrapper.transition.sample(output.predicted)
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
