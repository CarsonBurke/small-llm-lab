from __future__ import annotations

import math

import pytest
import torch

from fresh_lejepa_train import FreshLeJEPAGPT
from fresh_lejepa_train_v1_probe_shared_rms_pope import FreshLeJEPASharedRMSV1PoPE
from postraining.latent_thought import (
    AffineThoughtAdapter,
    EMIT,
    IDENTITY_AFFINE_THOUGHT_INPUT_SCHEMA,
    THINK,
    GaussianTransitionHead,
    LatentThoughtModel,
    migrate_legacy_wrapper_checkpoint,
    migrate_scalar_log_sigma_state,
    RENDERER_FEATURES_SCHEMA,
    ROLLOUT_POLICY_SCHEMA,
    THOUGHT_ACTION_TRANSFORM_SCHEMAS,
    THOUGHT_DISTRIBUTION_SCHEMA,
    THOUGHT_INPUT_SCHEMA,
    THOUGHT_MEAN_SCHEMA,
    THOUGHT_LOG_SIGMA_MAX,
    THOUGHT_LOG_SIGMA_MIN,
    StopThinkingGate,
    ThoughtAdapter,
    validate_renderer_checkpoint,
    wrapper_init_kwargs_from_checkpoint,
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


def _assert_prefill_matches_steps(
    wrapper: LatentThoughtModel,
    ids: torch.Tensor,
    key_valid: torch.Tensor | None = None,
) -> None:
    length = ids.size(1)
    dense_caches = wrapper.make_generation_cache(
        ids.size(0), length, torch.device("cpu")
    )
    stepped_caches = wrapper.make_generation_cache(
        ids.size(0), length, torch.device("cpu")
    )
    with torch.no_grad():
        dense = wrapper.prefill(ids, dense_caches, key_valid)
        stepped = None
        pad_lengths = None if key_valid is None else length - key_valid.sum(1)
        for position in range(length):
            key_mask = None
            if key_valid is not None:
                key_mask = key_valid[:, : position + 1]
                key_mask = key_mask | (pad_lengths[:, None] > position)
            stepped = wrapper.token_step(
                ids[:, position], stepped_caches, position, key_mask
            )
    assert stepped is not None
    torch.testing.assert_close(dense.belief, stepped.belief)
    torch.testing.assert_close(dense.predicted, stepped.predicted)
    torch.testing.assert_close(
        dense.thought_log_sigma, stepped.thought_log_sigma
    )
    torch.testing.assert_close(dense.logits, stepped.logits)
    for dense_layer, stepped_layer in zip(
        dense_caches, stepped_caches, strict=True
    ):
        for dense_tensor, stepped_tensor in zip(
            dense_layer, stepped_layer, strict=True
        ):
            torch.testing.assert_close(dense_tensor, stepped_tensor)


def test_dense_pope_prefill_matches_incremental_with_left_padding():
    torch.manual_seed(41)
    wrapper = LatentThoughtModel(_pope_model()).eval()
    ids = torch.randint(1, 32, (3, 8))
    lengths = torch.tensor([4, 6, 8])
    key_valid = torch.arange(8)[None] >= (8 - lengths)[:, None]
    ids = ids * key_valid
    _assert_prefill_matches_steps(wrapper, ids, key_valid)


def test_dense_rope_prefill_matches_incremental():
    torch.manual_seed(43)
    wrapper = LatentThoughtModel(FreshLeJEPAGPT(**KWARGS).eval())
    ids = torch.randint(1, 32, (2, 7))
    _assert_prefill_matches_steps(wrapper, ids)


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


def test_dense_step_thought_mean_gets_no_renderer_gradient():
    torch.manual_seed(9)
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone).eval()
    with torch.no_grad():
        backbone.policy_probe.output.weight.normal_(std=0.1)
    ids = torch.randint(0, 32, (2,))
    caches = wrapper.make_generation_cache(2, 1, torch.device("cpu"))
    projected = []
    handle = wrapper.transition.mean_head.register_forward_pre_hook(
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
        for parameter in wrapper.transition.mean_head.parameters()
    )
    assert float(backbone.blocks[0].attn.proj.weight.grad.abs().sum()) > 0.0


def test_renderer_checkpoint_schema_rejects_old_semantics():
    validate_renderer_checkpoint(
        {
            "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
            "rollout_policy_schema": ROLLOUT_POLICY_SCHEMA,
            "thought_input_schema": THOUGHT_INPUT_SCHEMA,
            "thought_distribution_schema": THOUGHT_DISTRIBUTION_SCHEMA,
            "thought_mean_schema": THOUGHT_MEAN_SCHEMA,
        },
        "current.pt",
    )
    # Initial gain is fully represented by the learned scalar, so the v2
    # origin label remains functionally resumable for the live policy.
    validate_renderer_checkpoint(
        {
            "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
            "rollout_policy_schema": ROLLOUT_POLICY_SCHEMA,
            "thought_input_schema": THOUGHT_INPUT_SCHEMA,
            "thought_distribution_schema": THOUGHT_DISTRIBUTION_SCHEMA,
            "thought_mean_schema": (
                "fresh_linear_learned_output_gain_0.01_zero_bias/v2"
            ),
        },
        "live-v2.pt",
    )
    with pytest.raises(ValueError, match="Old or untagged VAPO checkpoints"):
        validate_renderer_checkpoint({}, "old.pt")
    with pytest.raises(ValueError, match="predicted/v1"):
        validate_renderer_checkpoint(
            {"renderer_features_schema": "input_latent+predicted/v1"}, "old.pt"
        )
    with pytest.raises(
        ValueError, match="reasoning mode or forced-initial assignment"
    ):
        validate_renderer_checkpoint(
            {"renderer_features_schema": RENDERER_FEATURES_SCHEMA},
            "old-policy.pt",
        )
    with pytest.raises(ValueError, match="different deployed thought adapter"):
        validate_renderer_checkpoint(
            {
                "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
                "rollout_policy_schema": ROLLOUT_POLICY_SCHEMA,
            },
            "old-adapter.pt",
        )
    with pytest.raises(ValueError, match="unbounded log-sigma"):
        validate_renderer_checkpoint(
            {
                "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
                "rollout_policy_schema": ROLLOUT_POLICY_SCHEMA,
                "thought_input_schema": THOUGHT_INPUT_SCHEMA,
            },
            "old-sigma.pt",
        )
    validate_renderer_checkpoint(
        {
            "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
            "rollout_policy_schema": ROLLOUT_POLICY_SCHEMA,
            "thought_input_schema": THOUGHT_INPUT_SCHEMA,
            "thought_mean_schema": THOUGHT_MEAN_SCHEMA,
        },
        "critic-warmup.pt",
        allow_transition_reset=True,
    )
    affine_payload = {
        "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
        "rollout_policy_schema": ROLLOUT_POLICY_SCHEMA,
        "thought_input_schema": "fresh_zero_affine/v5",
        "thought_distribution_schema": THOUGHT_DISTRIBUTION_SCHEMA,
        "thought_mean_schema": THOUGHT_MEAN_SCHEMA,
    }
    validate_renderer_checkpoint(
        affine_payload,
        "affine-v24.pt",
        expected_thought_input_schema=IDENTITY_AFFINE_THOUGHT_INPUT_SCHEMA,
    )
    with pytest.raises(ValueError, match="different deployed thought adapter"):
        validate_renderer_checkpoint(affine_payload, "nonlinear-v26.pt")
    tanh_payload = {
        "renderer_features_schema": RENDERER_FEATURES_SCHEMA,
        "rollout_policy_schema": ROLLOUT_POLICY_SCHEMA,
        "thought_input_schema": THOUGHT_INPUT_SCHEMA,
        "thought_action_transform_schema": (
            THOUGHT_ACTION_TRANSFORM_SCHEMAS["tanh"]
        ),
        "thought_distribution_schema": THOUGHT_DISTRIBUTION_SCHEMA,
        "thought_mean_schema": THOUGHT_MEAN_SCHEMA,
    }
    validate_renderer_checkpoint(
        tanh_payload,
        "tanh.pt",
        expected_thought_action_transform_schema=(
            THOUGHT_ACTION_TRANSFORM_SCHEMAS["tanh"]
        ),
    )
    with pytest.raises(ValueError, match="different action"):
        validate_renderer_checkpoint(tanh_payload, "tanh-as-raw.pt")


def test_wrapper_checkpoint_kwargs_preserve_legacy_policy_semantics():
    assert wrapper_init_kwargs_from_checkpoint({"args": {}}) == {
        "thought_adapter": "identity_affine",
        "sigma_state_init": "constant",
        "thought_action_transform": "identity",
    }
    assert wrapper_init_kwargs_from_checkpoint(
        {
            "args": {
                "thought_adapter": "orthogonal_silu",
                "thought_sigma_state_init": "orthogonal",
                "thought_action_transform": "tanh",
            }
        }
    ) == {
        "thought_adapter": "orthogonal_silu",
        "sigma_state_init": "orthogonal",
        "thought_action_transform": "tanh",
    }


def test_tanh_thought_input_transforms_once_before_the_adapter():
    wrapper = LatentThoughtModel(
        _pope_model(), thought_action_transform="tanh"
    ).eval()
    raw = torch.linspace(-3.0, 3.0, 64).view(2, 32)
    expected = wrapper.adapter(raw.tanh())[:, None]
    torch.testing.assert_close(wrapper.thought_input(raw), expected)


def test_gate_zero_init_is_exactly_uniform():
    gate = StopThinkingGate(16)
    belief = torch.randn(4, 16)
    assert torch.all(gate.stop_logit(belief) == 0)
    log_prob = gate.log_prob(torch.tensor([THINK, EMIT, THINK, EMIT]), belief)
    torch.testing.assert_close(log_prob, torch.full((4,), math.log(0.5)))
    torch.testing.assert_close(gate.entropy(belief), torch.full((4,), math.log(2.0)))


def test_gate_sample_log_prob_recomputes_identically():
    torch.manual_seed(11)
    gate = StopThinkingGate(16)
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
    belief = torch.randn(5, 8)
    log_sigma = head.predict_log_sigma(belief)
    generator = torch.Generator().manual_seed(21)
    sample, log_prob = head.sample(mean, log_sigma, generator=generator)
    torch.testing.assert_close(head.log_prob(sample, mean, log_sigma), log_prob)
    reference = torch.distributions.Normal(mean, log_sigma.exp())
    torch.testing.assert_close(log_prob, reference.log_prob(sample).sum(-1))
    torch.testing.assert_close(
        head.per_dim_log_prob(sample, mean, log_sigma), reference.log_prob(sample)
    )


def test_transition_mean_head_starts_small_orthogonal_and_zero_bias():
    torch.manual_seed(12)
    head = GaussianTransitionHead(8)
    gram = head.mean_head.weight @ head.mean_head.weight.T
    expected = torch.eye(8)
    torch.testing.assert_close(gram, expected, rtol=1e-5, atol=2e-7)
    assert torch.count_nonzero(head.mean_head.bias) == 0
    assert head.mean_head.output_gain.item() == pytest.approx(
        head.MEAN_INIT_GAIN
    )
    assert head.MEAN_INIT_GAIN == pytest.approx(0.1)
    torch.testing.assert_close(
        head.log_sigma_head.bias.detach(),
        torch.full((8,), head.raw_from_log_sigma(-2.0)),
    )
    log_sigma = head.predict_log_sigma(torch.randn(3, 8))
    assert not torch.allclose(log_sigma[0], log_sigma[1])
    assert float(
        (log_sigma.detach() + 2.0).square().mean().sqrt()
    ) < 0.1
    belief = torch.randn(5, 8)
    belief = torch.nn.functional.rms_norm(belief, (8,))
    mean = head.predict_mean(belief)
    torch.testing.assert_close(
        mean.norm(dim=-1),
        belief.norm(dim=-1) * head.MEAN_INIT_GAIN,
        rtol=1e-5,
        atol=1e-6,
    )


def test_transition_sigma_head_starts_orthogonal_and_mildly_state_dependent():
    head = GaussianTransitionHead(8, log_sigma=-0.5)
    parameters = list(head.log_sigma_head.parameters())
    assert len(parameters) == 3
    gram = head.log_sigma_head.weight @ head.log_sigma_head.weight.T
    torch.testing.assert_close(
        gram, torch.eye(8), rtol=1e-5, atol=2e-7
    )
    assert head.log_sigma_head.residual_gain.item() == pytest.approx(0.01)
    expected_raw_bias = head.raw_from_log_sigma(-0.5)
    torch.testing.assert_close(
        head.log_sigma_head.bias.detach(), torch.full((8,), expected_raw_bias)
    )
    beliefs = torch.nn.functional.rms_norm(torch.randn(128, 8), (8,))
    log_sigma = head.predict_log_sigma(beliefs)
    assert not torch.allclose(log_sigma[0], log_sigma[1])
    assert float(
        (log_sigma.detach() + 0.5).square().mean().sqrt()
    ) < 0.05

    weight = head.log_sigma_head.weight.detach().clone()
    head.set_noise_level(-2.5)
    torch.testing.assert_close(head.log_sigma_head.weight, weight)
    assert float(
        (head.predict_log_sigma(beliefs).detach() + 2.5)
        .square()
        .mean()
        .sqrt()
    ) < 0.05


def test_transition_sigma_head_has_first_step_gradients_for_all_parameters():
    head = GaussianTransitionHead(8, log_sigma=-2.0)
    beliefs = torch.nn.functional.rms_norm(torch.randn(32, 8), (8,))

    head.predict_log_sigma(beliefs).square().mean().backward()

    for name, parameter in head.log_sigma_head.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad), name


def test_transition_sigma_constant_control_is_exact():
    head = GaussianTransitionHead(
        8, log_sigma=-0.5, sigma_state_init="constant"
    )
    assert torch.count_nonzero(head.log_sigma_head.weight) == 0
    beliefs = torch.randn(6, 8)
    torch.testing.assert_close(
        head.predict_log_sigma(beliefs), torch.full((6, 8), -0.5)
    )


def test_transition_sigma_orthogonal_reset_preserves_global_rng_stream():
    torch.manual_seed(123)
    head = GaussianTransitionHead(
        8, log_sigma=-2.0, sigma_state_init="constant"
    )
    before = torch.get_rng_state()
    head.reset_noise(-2.0, sigma_state_init="orthogonal")
    after = torch.get_rng_state()
    torch.testing.assert_close(after, before)


def test_transition_sigma_preserves_small_residuals_under_bf16_autocast():
    head = GaussianTransitionHead(8, log_sigma=-2.0)
    belief = torch.ones(2, 8)
    with torch.no_grad():
        head.log_sigma_head.weight.zero_()
        head.log_sigma_head.weight[0, 0] = 1e-3
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        log_sigma = head.predict_log_sigma(belief)
    assert log_sigma.dtype == torch.float32
    # A bf16 affine output centered at -2 would round this residual away.
    assert float(log_sigma[0, 0].detach()) != -2.0
    expected = head.bound_raw_log_sigma(torch.tensor(
        head.raw_from_log_sigma(-2.0)
        + head.log_sigma_head.residual_gain.item() * 1e-3
    ))
    assert float(log_sigma[0, 0].detach()) == pytest.approx(
        float(expected), abs=2e-5
    )


def test_transition_sigma_is_smoothly_bounded_and_finite():
    head = GaussianTransitionHead(8, log_sigma=-2.0)
    beliefs = torch.ones(2, 8)
    with torch.no_grad():
        head.log_sigma_head.weight.fill_(1e6)
    upper = head.predict_log_sigma(beliefs)
    lower = head.predict_log_sigma(-beliefs)
    assert torch.isfinite(upper).all()
    assert torch.isfinite(lower).all()
    assert torch.all(upper <= THOUGHT_LOG_SIGMA_MAX)
    assert torch.all(lower >= THOUGHT_LOG_SIGMA_MIN)
    torch.testing.assert_close(upper, torch.full_like(upper, THOUGHT_LOG_SIGMA_MAX))
    torch.testing.assert_close(lower, torch.full_like(lower, THOUGHT_LOG_SIGMA_MIN))


def test_transition_sigma_init_must_be_strictly_inside_bounds():
    with pytest.raises(ValueError, match="strictly inside"):
        GaussianTransitionHead(8, log_sigma=THOUGHT_LOG_SIGMA_MIN)
    with pytest.raises(ValueError, match="strictly inside"):
        GaussianTransitionHead(8, log_sigma=THOUGHT_LOG_SIGMA_MAX)


def test_scalar_log_sigma_state_migrates_to_the_head():
    head = GaussianTransitionHead(8, log_sigma=-0.5)
    legacy = {"transition.log_sigma": torch.tensor(-1.5)}
    assert migrate_scalar_log_sigma_state(legacy, head)
    assert "transition.log_sigma" not in legacy
    torch.testing.assert_close(
        legacy["transition.log_sigma_head.bias"],
        torch.full((8,), head.raw_from_log_sigma(-1.5)),
    )
    assert torch.all(legacy["transition.log_sigma_head.weight"] == 0.0)
    torch.testing.assert_close(
        legacy["transition.log_sigma_head.residual_gain"],
        head.log_sigma_head.residual_gain,
    )
    # The migrated policy is the retired scalar policy exactly.
    head.log_sigma_head.load_state_dict(
        {
            "weight": legacy["transition.log_sigma_head.weight"],
            "bias": legacy["transition.log_sigma_head.bias"],
            "residual_gain": legacy[
                "transition.log_sigma_head.residual_gain"
            ],
        }
    )
    torch.testing.assert_close(
        head.predict_log_sigma(torch.randn(4, 8)), torch.full((4, 8), -1.5)
    )
    # New-format states pass through untouched.
    assert not migrate_scalar_log_sigma_state(legacy, head)


def test_fresh_mean_requires_explicit_legacy_branch_migration():
    torch.manual_seed(14)
    wrapper = LatentThoughtModel(_pope_model())
    state = {
        key: value.detach().clone()
        for key, value in wrapper.state_dict().items()
        if not key.startswith("transition.mean_head.")
    }
    payload = {"model": state}
    assert migrate_legacy_wrapper_checkpoint(payload, wrapper) == (
        False,
        False,
        False,
    )
    assert "transition.mean_head.weight" not in state
    _, migrated, _ = migrate_legacy_wrapper_checkpoint(
        payload, wrapper, initialize_fresh_mean=True
    )
    assert migrated
    assert payload["thought_mean_schema"] == THOUGHT_MEAN_SCHEMA
    torch.testing.assert_close(
        state["transition.mean_head.weight"],
        wrapper.transition.mean_head.weight,
    )
    torch.testing.assert_close(
        state["transition.mean_head.bias"],
        wrapper.transition.mean_head.bias,
    )
    torch.testing.assert_close(
        state["transition.mean_head.output_gain"],
        wrapper.transition.mean_head.output_gain,
    )


def test_explicit_actor_restart_replaces_mean_and_sigma_heads():
    wrapper = LatentThoughtModel(_pope_model())
    wrapper.transition.mean_head.reset_output_gain(0.1)
    state = {
        key: value.detach().clone()
        for key, value in wrapper.state_dict().items()
    }
    state["transition.mean_head.weight"].zero_()
    state["transition.mean_head.bias"].fill_(2.0)
    state["transition.mean_head.output_gain"].fill_(0.01)
    state["transition.log_sigma_head.weight"].zero_()
    state["transition.log_sigma_head.bias"].fill_(2.0)
    state["transition.log_sigma_head.residual_gain"].fill_(0.5)
    payload = {"model": state}

    _, migrated, _ = migrate_legacy_wrapper_checkpoint(
        payload, wrapper, initialize_fresh_mean=True
    )

    assert migrated
    torch.testing.assert_close(
        state["transition.mean_head.weight"],
        wrapper.transition.mean_head.weight,
    )
    torch.testing.assert_close(
        state["transition.mean_head.bias"],
        wrapper.transition.mean_head.bias,
    )
    assert state["transition.mean_head.output_gain"].item() == pytest.approx(0.1)
    torch.testing.assert_close(
        state["transition.log_sigma_head.weight"],
        wrapper.transition.log_sigma_head.weight,
    )
    torch.testing.assert_close(
        state["transition.log_sigma_head.bias"],
        wrapper.transition.log_sigma_head.bias,
    )
    torch.testing.assert_close(
        state["transition.log_sigma_head.residual_gain"],
        wrapper.transition.log_sigma_head.residual_gain,
    )


def test_explicit_actor_restart_replaces_the_complete_gate():
    wrapper = LatentThoughtModel(_pope_model())
    with torch.no_grad():
        wrapper.gate.head.weight.zero_()
        wrapper.gate.head.bias.fill_(1.25)
    expected_weight = wrapper.gate.head.weight.detach().clone()
    expected_bias = wrapper.gate.head.bias.detach().clone()
    state = {
        key: value.detach().clone()
        for key, value in wrapper.state_dict().items()
    }
    state["gate.head.weight"].fill_(0.5)
    state["gate.head.bias"].fill_(-2.0)
    payload = {"model": state}

    migrate_legacy_wrapper_checkpoint(
        payload, wrapper, initialize_fresh_gate=True
    )
    wrapper.load_state_dict(state, strict=True)

    torch.testing.assert_close(wrapper.gate.head.weight, expected_weight)
    torch.testing.assert_close(wrapper.gate.head.bias, expected_bias)
    belief = torch.randn(4, expected_weight.shape[1])
    expected_stop_probability = torch.sigmoid(expected_bias).expand(4)
    torch.testing.assert_close(
        wrapper.gate.stop_logit(belief).sigmoid(), expected_stop_probability
    )


def test_fresh_adapter_explicitly_replaces_critic_warm_identity_state():
    wrapper = LatentThoughtModel(_pope_model())
    state = {
        key: value.detach().clone()
        for key, value in wrapper.state_dict().items()
    }
    state["adapter.projection.weight"] = torch.eye(32)
    state["adapter.projection.bias"] = torch.full((32,), 0.5)
    payload = {
        "model": state,
        "thought_input_schema": "identity_init_affine/v1",
    }

    _, _, reset = migrate_legacy_wrapper_checkpoint(
        payload, wrapper, initialize_fresh_adapter=True
    )

    assert reset
    assert payload["thought_input_schema"] == THOUGHT_INPUT_SCHEMA
    torch.testing.assert_close(
        state["adapter.projection.weight"],
        wrapper.adapter.projection.weight,
    )
    torch.testing.assert_close(
        state["adapter.projection.bias"],
        wrapper.adapter.projection.bias,
    )
    assert "adapter.interpolation_strength" not in state


def test_thought_policy_gradient_flows_through_the_mean():
    # v2 (full-model RL): the policy gradient must reach the prediction
    # path — per_dim_log_prob differentiates through the passed mean.
    torch.manual_seed(15)
    head = GaussianTransitionHead(8)
    mean = torch.randn(5, 8, requires_grad=True)
    sample = (mean + 0.3).detach()
    log_sigma = torch.full((5, 8), -0.5)
    head.per_dim_log_prob(sample, mean, log_sigma).sum().backward()
    assert mean.grad is not None
    # d/dmean of -0.5*((s-m)/sigma)^2 is (s-m)/sigma^2, positive here.
    assert torch.all(mean.grad > 0)


def test_thought_policy_gradient_reaches_the_sigma_head():
    # State-dependent sigma: the log-prob path must differentiate through
    # predict_log_sigma so the joint-action PPO objective can move the head
    # — including its zero-init weight, via the belief.
    torch.manual_seed(16)
    head = GaussianTransitionHead(8)
    belief = torch.randn(5, 8)
    mean = torch.randn(5, 8)
    sample = mean + 0.3 * torch.randn(5, 8)
    log_sigma = head.predict_log_sigma(belief)
    head.log_prob(sample, mean, log_sigma).sum().backward()
    assert head.log_sigma_head.bias.grad is not None
    # d/dlog_sigma of the log-density is ((s-m)/sigma)^2 - 1 per dim,
    # generically nonzero for off-mean samples.
    assert head.log_sigma_head.bias.grad.abs().sum().item() > 0.0
    assert head.log_sigma_head.weight.grad is not None
    assert head.log_sigma_head.weight.grad.abs().sum().item() > 0.0


def test_actor_adapter_starts_orthogonal_nonlinear_and_critic_is_affine():
    torch.manual_seed(19)
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone)
    critic_adapter = AffineThoughtAdapter(32)
    thought = torch.randn(2, 32)
    injected = wrapper.thought_input(thought)
    assert injected.shape == (2, 1, 32)
    actor_gram = (
        wrapper.adapter.projection.weight
        @ wrapper.adapter.projection.weight.T
    )
    critic_gram = (
        critic_adapter.projection.weight
        @ critic_adapter.projection.weight.T
    )
    torch.testing.assert_close(actor_gram, torch.eye(32), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(critic_gram, torch.eye(32), rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(wrapper.adapter.projection.bias) == 0
    assert torch.count_nonzero(critic_adapter.projection.bias) == 0
    assert not torch.allclose(injected.squeeze(1), thought)
    torch.testing.assert_close(critic_adapter(thought).norm(dim=-1), thought.norm(dim=-1))


def test_nonlinear_adapter_learns_on_its_first_backward_pass():
    adapter = ThoughtAdapter(4)
    thought = torch.randn(3, 4, requires_grad=True)

    adapter(thought).sum().backward()

    assert float(adapter.projection.weight.grad.abs().sum()) > 0.0
    assert float(adapter.projection.bias.grad.abs().sum()) > 0.0
    assert float(thought.grad.abs().sum()) > 0.0


def test_identity_affine_adapter_remains_an_exact_control():
    adapter = ThoughtAdapter(4, kind="identity_affine")
    thought = torch.randn(3, 4)
    torch.testing.assert_close(adapter(thought), thought)


def test_adapter_bias_is_a_shared_thought_type_offset():
    adapter = ThoughtAdapter(4)
    marker = torch.tensor([0.25, -0.5, 1.0, 0.75])
    with torch.no_grad():
        adapter.projection.weight.zero_()
        adapter.projection.bias.copy_(marker)
    thoughts = torch.randn(3, 4)

    torch.testing.assert_close(
        adapter(thoughts),
        (2.0 * torch.nn.functional.silu(marker)).expand_as(thoughts),
    )


def test_chunked_teacher_forced_ce_matches_one_shot_cross_entropy():
    torch.manual_seed(29)
    backbone = _pope_model()
    wrapper = LatentThoughtModel(backbone).eval()
    ids = torch.randint(0, 32, (3, 10))
    targets = torch.randint(0, 32, (3, 10))
    with torch.no_grad():
        one_shot = torch.nn.functional.cross_entropy(
            wrapper.policy_logits(ids).float().flatten(0, 1), targets.flatten()
        )
        full = wrapper(ids, targets)
        # Force several uneven chunks; the token-weighted sum must reduce to
        # the identical mean.
        wrapper.BPB_EVAL_CHUNK_TOKENS = 7
        chunked = wrapper(ids, targets)
    torch.testing.assert_close(full, one_shot)
    torch.testing.assert_close(chunked, one_shot)


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
        sample, _ = wrapper.transition.sample(
            output.predicted,
            wrapper.transition.predict_log_sigma(output.belief),
        )
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
