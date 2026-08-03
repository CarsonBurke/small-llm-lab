from __future__ import annotations

import torch

from pretraining.fresh_lejepa import fresh_lejepa_train as v1_module
from pretraining.fresh_lejepa import fresh_lejepa_train_v1_probe_shared_rms_pope_belief_attached as belief_attached_module
from pretraining.fresh_lejepa import fresh_lejepa_train_v4_predicted_only as v4_predicted_module
from pretraining.fresh_lejepa import fresh_lejepa_train_v4_predictor_dropout as v4_dropout_module
from pretraining.fresh_lejepa.fresh_lejepa_train_v2_sigreg_projector import FreshLeJEPAV2SIGRegProjector
from pretraining.fresh_lejepa.fresh_lejepa_train_v2_shared_projector import FreshLeJEPAV2SharedProjector
from pretraining.fresh_lejepa.fresh_lejepa_train_v2_predicted_only import FreshLeJEPAV2PredictedOnly
from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_projector import (
    FreshLeJEPASharedProjectorV1Probes,
)
from pretraining.fresh_lejepa.fresh_lejepa_train import ResidualProbe
from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_rms_projector import (
    FreshLeJEPASharedRMSProjectorV1Probes,
    RMSTokenProjector,
)
from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_rms_pope import (
    FreshLeJEPASharedRMSV1PoPE,
)
from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_rms_pope_attached import (
    FreshLeJEPASharedRMSV1PoPEAttachedCE,
)
from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_rms_pope_belief_attached import (
    FreshLeJEPASharedRMSV1PoPEBeliefAttachedCE,
)
from pretraining.fresh_lejepa.fresh_lejepa_train_v4 import (
    FreshLeJEPAV4,
    LearnedOutputCriticProbe,
    LearnedOutputPolicyProbe,
    _install_configurable_accumulation,
)
from pretraining.fresh_lejepa.fresh_lejepa_train_v4_predicted_only import FreshLeJEPAV4PredictedOnly
from pretraining.fresh_lejepa.fresh_lejepa_train_v4_predictor_dropout import FreshLeJEPAV4PredictorDropout
from pretraining.fresh_lejepa.fresh_lejepa_train_v5_swiglu import (
    FreshLeJEPAV5SwiGLU,
    PreNormSwiGLUBlock,
    SwiGLUCriticProbe,
    SwiGLUPolicyProbe,
)
from pretraining.fresh_lejepa.fresh_lejepa_train_v5_dropout_only import FreshLeJEPAV5DropoutOnly
from pretraining.fresh_lejepa.fresh_lejepa_train_v5_belief_only import FreshLeJEPAV5BeliefOnly
from pretraining.fresh_lejepa.fresh_lejepa_train_v5_jedi_denoising import (
    FreshLeJEPAV5JEDIDenoising,
    edm_coefficients,
)
from pretraining.fresh_lejepa.fresh_lejepa_train_v6_belief_dropout import (
    BeliefOnlyCriticProbe,
    BeliefOnlyPolicyProbe,
    FreshLeJEPAV6BeliefDropout,
)
from pretraining.fresh_lejepa.fresh_lejepa_train_v7_preproj_belief import FreshLeJEPAV7PreProjBelief
from pretraining.fresh_lejepa.fresh_lejepa_train_v8_token_swiglu import (
    FreshLeJEPAV8TokenSwiGLU,
    SharedTokenEncoder,
)
from pretraining.fresh_lejepa.fresh_lejepa_train_v9_belief_transition_jedi import (
    FreshLeJEPAV9BeliefTransition,
)


KWARGS = dict(
    vocab_size=32, num_layers=3, model_dim=32, num_heads=4, num_kv_heads=2,
    mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.01,
    logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
)


def test_sigreg_only_projector_does_not_change_predictor_or_probe_latent():
    model = FreshLeJEPAV2SIGRegProjector(**KWARGS)
    ids = torch.randint(0, 32, (2, 5))
    raw = model.embed_tokens(ids)
    token_latent, _ = model.latent_features(ids)
    torch.testing.assert_close(token_latent, raw)
    assert not torch.allclose(model.sigreg_features(token_latent), token_latent)


def test_shared_projector_drives_all_latent_paths_but_ce_stays_detached():
    model = FreshLeJEPAV2SharedProjector(**KWARGS)
    ids = torch.randint(0, 32, (2, 5))
    raw = torch.nn.functional.rms_norm(model.tok_emb(ids), (32,))
    token_latent, _ = model.latent_features(ids)
    assert not torch.allclose(token_latent, raw)
    torch.testing.assert_close(model.sigreg_features(token_latent), token_latent)
    model.policy_logits(ids).sum().backward()
    assert model.tok_emb.weight.grad is None
    assert model.latent_projector.input.weight.grad is None


def test_shared_projector_projects_shifted_training_trajectory_once():
    model = FreshLeJEPAV2SharedProjector(**KWARGS).train()
    ids = torch.randint(0, 32, (2, 6))
    inputs, targets = ids[:, :-1], ids[:, 1:]
    before = model.latent_projector.norm.num_batches_tracked.clone()
    token_latent, _, target_latent = model.training_latents(inputs, targets)
    after = model.latent_projector.norm.num_batches_tracked
    assert (after - before).item() == 1
    torch.testing.assert_close(token_latent[:, 1:], target_latent[:, :-1])
    sigreg_latent = model.training_sigreg_features(token_latent, target_latent)
    assert sigreg_latent.shape == (2, 6, 32)
    torch.testing.assert_close(sigreg_latent, torch.cat((token_latent, target_latent[:, -1:]), 1))


def test_predicted_only_probe_zeros_raw_half_and_preserves_shape():
    model = FreshLeJEPAV2PredictedOnly(**KWARGS)
    token = torch.randn(2, 5, 32)
    predicted = torch.randn(2, 5, 32)
    features = model.probe_features(token, predicted)
    torch.testing.assert_close(features[..., :32], torch.zeros_like(token))
    torch.testing.assert_close(features[..., 32:], predicted)
    assert features.shape == (2, 5, 64)


def test_shared_projector_incremental_matches_full_sequence():
    model = FreshLeJEPAV2SharedProjector(**KWARGS).eval()
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.attn.proj.weight, std=0.05)
            torch.nn.init.normal_(block.mlp.proj.weight, std=0.05)
        torch.nn.init.normal_(model.critic_probe.output.weight, std=0.05)
    ids = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        full_logits, full_values = model.policy_logits(ids), model.values(ids)
        caches = model.make_generation_cache(2, 6, ids.device)
        logits_steps, value_steps = [], []
        for position in range(6):
            logits, values, caches = model.generation_step(ids[:, position], caches, position)
            logits_steps.append(logits)
            value_steps.append(values)
    torch.testing.assert_close(torch.stack(logits_steps, 1), full_logits, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(torch.stack(value_steps, 1), full_values, rtol=2e-4, atol=2e-4)


def test_shared_projector_v1_probe_ablation_uses_full_learned_outputs():
    model = FreshLeJEPASharedProjectorV1Probes(**KWARGS)
    assert isinstance(model.policy_probe, ResidualProbe)
    assert isinstance(model.critic_probe, ResidualProbe)
    assert model.policy_probe.input.in_features == 64
    assert model.policy_probe.input.out_features == 64
    assert model.policy_probe.output.in_features == 64
    assert model.policy_probe.output.out_features == 32
    ids = torch.randint(0, 32, (2, 5))
    assert model.policy_logits(ids).shape == (2, 5, 32)


def test_v1_probe_swap_preserves_control_parameters_and_rng_stream():
    torch.manual_seed(123)
    control = FreshLeJEPAV2SharedProjector(**KWARGS)
    control_next_random = torch.randn(8)
    torch.manual_seed(123)
    variant = FreshLeJEPASharedProjectorV1Probes(**KWARGS)
    variant_next_random = torch.randn(8)
    control_state = control.state_dict()
    variant_state = variant.state_dict()
    for name, value in control_state.items():
        if "policy_probe" not in name and "critic_probe" not in name:
            torch.testing.assert_close(variant_state[name], value)
    torch.testing.assert_close(variant_next_random, control_next_random)


def test_v1_probe_logits_survive_training_entrypoint_class_rebind():
    model = FreshLeJEPASharedProjectorV1Probes(**KWARGS)
    original = v1_module.FreshLeJEPAGPT
    v1_module.FreshLeJEPAGPT = FreshLeJEPASharedProjectorV1Probes
    try:
        ids = torch.randint(0, 32, (2, 5))
        assert model.policy_logits(ids).shape == (2, 5, 32)
    finally:
        v1_module.FreshLeJEPAGPT = original


def test_attached_ce_changes_no_initial_state_or_rng_stream():
    torch.manual_seed(789)
    control = FreshLeJEPASharedRMSV1PoPE(**KWARGS)
    control_next_random = torch.randn(8)
    torch.manual_seed(789)
    attached = FreshLeJEPASharedRMSV1PoPEAttachedCE(**KWARGS)
    attached_next_random = torch.randn(8)
    for name, value in control.state_dict().items():
        torch.testing.assert_close(attached.state_dict()[name], value)
    torch.testing.assert_close(attached_next_random, control_next_random)


def test_attached_ce_updates_both_latent_paths_but_not_critic_path():
    model = FreshLeJEPASharedRMSV1PoPEAttachedCE(**KWARGS)
    token = torch.randn(2, 5, 32, requires_grad=True)
    predicted = torch.randn(2, 5, 32, requires_grad=True)
    model.policy_loss_features(token, predicted).sum().backward()
    assert token.grad is not None and torch.count_nonzero(token.grad).item() > 0
    assert predicted.grad is not None and torch.count_nonzero(predicted.grad).item() > 0

    with torch.no_grad():
        torch.nn.init.normal_(model.policy_probe.output.weight, std=0.05)
    model.eval()
    ids = torch.randint(0, 32, (2, 5))
    targets = torch.randint(0, 32, ids.shape)
    loss = model(ids, targets)
    loss.backward()
    assert model.tok_emb.weight.grad is not None
    assert torch.count_nonzero(model.tok_emb.weight.grad).item() > 0
    assert model.latent_projector.input.weight.grad is not None
    assert torch.count_nonzero(model.latent_projector.input.weight.grad).item() > 0
    assert model.prediction_projector.input.weight.grad is not None
    assert torch.count_nonzero(model.prediction_projector.input.weight.grad).item() > 0

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        torch.nn.init.normal_(model.critic_probe.output.weight, std=0.05)
    model.values(ids).sum().backward()
    assert model.tok_emb.weight.grad is None
    assert model.latent_projector.input.weight.grad is None
    assert model.prediction_projector.input.weight.grad is None


def test_belief_attached_ce_changes_no_initial_state_or_rng_stream():
    torch.manual_seed(790)
    control = FreshLeJEPASharedRMSV1PoPE(**KWARGS)
    control_next_random = torch.randn(8)
    torch.manual_seed(790)
    belief_attached = FreshLeJEPASharedRMSV1PoPEBeliefAttachedCE(**KWARGS)
    belief_attached_next_random = torch.randn(8)
    for name, value in control.state_dict().items():
        torch.testing.assert_close(belief_attached.state_dict()[name], value)
    torch.testing.assert_close(belief_attached_next_random, control_next_random)


def test_belief_attached_entrypoint_disables_data_reusing_warmup(monkeypatch):
    delegated = []
    original_class = belief_attached_module.pope.FreshLeJEPASharedRMSV1PoPE
    original_architecture = belief_attached_module.pope.POPE_ARCHITECTURE
    original_file = belief_attached_module.pope.__file__
    monkeypatch.setattr(
        belief_attached_module.pope.v1.FreshHyperparameters,
        "warmup_steps",
        20,
    )
    monkeypatch.setattr(
        belief_attached_module.pope,
        "main",
        lambda: delegated.append(True),
    )
    try:
        belief_attached_module.main()
        assert delegated == [True]
        assert belief_attached_module.pope.v1.FreshHyperparameters.warmup_steps == 0
        assert (
            belief_attached_module.pope.FreshLeJEPASharedRMSV1PoPE
            is FreshLeJEPASharedRMSV1PoPEBeliefAttachedCE
        )
    finally:
        belief_attached_module.pope.FreshLeJEPASharedRMSV1PoPE = original_class
        belief_attached_module.pope.POPE_ARCHITECTURE = original_architecture
        belief_attached_module.pope.__file__ = original_file


def test_belief_attached_ce_bypasses_prediction_projector_gradients():
    model = FreshLeJEPASharedRMSV1PoPEBeliefAttachedCE(**KWARGS).eval()
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.attn.proj.weight, std=0.05)
            torch.nn.init.normal_(block.mlp.proj.weight, std=0.05)
        torch.nn.init.normal_(model.policy_probe.output.weight, std=0.05)
    ids = torch.randint(0, 32, (2, 5))
    targets = torch.randint(0, 32, ids.shape)
    model(ids, targets).backward()

    assert model.tok_emb.weight.grad is not None
    assert torch.count_nonzero(model.tok_emb.weight.grad).item() > 0
    assert model.latent_projector.input.weight.grad is not None
    assert torch.count_nonzero(model.latent_projector.input.weight.grad).item() > 0
    assert model.blocks[0].attn.c_qkv.weight.grad is not None
    assert torch.count_nonzero(model.blocks[0].attn.c_qkv.weight.grad).item() > 0
    assert all(
        parameter.grad is None
        for parameter in model.prediction_projector.parameters()
    )


def test_belief_attached_latent_loss_still_trains_prediction_projector():
    model = FreshLeJEPASharedRMSV1PoPEBeliefAttachedCE(**KWARGS).train()
    ids = torch.randint(0, 32, (2, 6))
    token, _, predicted, target = model.training_latents_with_belief(
        ids[:, :-1], ids[:, 1:]
    )
    torch.testing.assert_close(token[:, 1:], target[:, :-1])
    torch.nn.functional.mse_loss(predicted, target).backward()
    assert any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.prediction_projector.parameters()
    )


def test_belief_renderer_is_independent_of_prediction_projector():
    model = FreshLeJEPASharedRMSV1PoPEBeliefAttachedCE(**KWARGS).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.policy_probe.output.weight, std=0.05)
    ids = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        before = model.policy_logits(ids)
        for parameter in model.prediction_projector.parameters():
            parameter.add_(torch.randn_like(parameter) * 10)
        after = model.policy_logits(ids)
    torch.testing.assert_close(after, before)


def test_belief_renderer_incremental_matches_full_sequence():
    model = FreshLeJEPASharedRMSV1PoPEBeliefAttachedCE(**KWARGS).eval()
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.attn.proj.weight, std=0.05)
            torch.nn.init.normal_(block.mlp.proj.weight, std=0.05)
        torch.nn.init.normal_(model.policy_probe.output.weight, std=0.05)
        torch.nn.init.normal_(model.critic_probe.output.weight, std=0.05)
    ids = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        full_logits, full_values = model.policy_logits(ids), model.values(ids)
        caches = model.make_generation_cache(2, 6, ids.device)
        logits_steps, value_steps = [], []
        for position in range(6):
            logits, values, caches = model.generation_step(
                ids[:, position], caches, position
            )
            logits_steps.append(logits)
            value_steps.append(values)
    torch.testing.assert_close(
        torch.stack(logits_steps, 1), full_logits, rtol=2e-4, atol=2e-4
    )
    torch.testing.assert_close(
        torch.stack(value_steps, 1), full_values, rtol=2e-4, atol=2e-4
    )


def test_rms_ablation_changes_only_projector_norms_and_preserves_rng():
    torch.manual_seed(456)
    bn_model = FreshLeJEPASharedProjectorV1Probes(**KWARGS)
    bn_next_random = torch.randn(8)
    torch.manual_seed(456)
    rms_model = FreshLeJEPASharedRMSProjectorV1Probes(**KWARGS)
    rms_next_random = torch.randn(8)
    assert isinstance(rms_model.latent_projector, RMSTokenProjector)
    assert isinstance(rms_model.prediction_projector, RMSTokenProjector)
    for projector_name in ("latent_projector", "prediction_projector"):
        bn_projector = getattr(bn_model, projector_name)
        rms_projector = getattr(rms_model, projector_name)
        torch.testing.assert_close(rms_projector.input.weight, bn_projector.input.weight)
        torch.testing.assert_close(rms_projector.input.bias, bn_projector.input.bias)
        torch.testing.assert_close(rms_projector.output.weight, bn_projector.output.weight)
        torch.testing.assert_close(rms_projector.output.bias, bn_projector.output.bias)
    torch.testing.assert_close(rms_next_random, bn_next_random)


def test_v4_builds_compact_learned_probes_directly_and_detaches_ce():
    model = FreshLeJEPAV4(**KWARGS)
    assert isinstance(model.policy_probe, LearnedOutputPolicyProbe)
    assert isinstance(model.critic_probe, LearnedOutputCriticProbe)
    assert model.policy_probe.output.out_features == 32
    assert sum(p.numel() for p in model.policy_probe.parameters()) < sum(
        p.numel() for p in FreshLeJEPASharedProjectorV1Probes(**KWARGS).policy_probe.parameters()
    )
    ids = torch.randint(0, 32, (2, 5))
    model.policy_logits(ids).sum().backward()
    assert model.policy_probe.output.weight.grad is not None
    assert model.tok_emb.weight.grad is None
    assert model.latent_projector.input.weight.grad is None


def test_v4_sigreg_stays_and_computes_in_fp32_after_bfloat16_cast():
    model = FreshLeJEPAV4(**KWARGS).bfloat16()
    assert model.sigreg.t.dtype == torch.float32
    assert model.sigreg.phi.dtype == torch.float32
    assert model.sigreg.weights.dtype == torch.float32
    features = torch.randn(4, 3, 32, dtype=torch.bfloat16)
    loss = model.sigreg(features)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_sigreg_projection_checkpoint_preserves_loss_and_gradients():
    checkpointed = FreshLeJEPAV4(**KWARGS).sigreg
    eager = FreshLeJEPAV4(**KWARGS).sigreg
    checkpointed.num_proj = eager.num_proj = 8
    checkpointed.proj_chunk = eager.proj_chunk = 4
    checkpointed.position_chunk = eager.position_chunk = 2
    eager.checkpoint_projection_chunks = False
    first = torch.randn(4, 3, 32, requires_grad=True)
    second = first.detach().clone().requires_grad_(True)
    torch.manual_seed(321)
    first_loss = checkpointed(first)
    first_loss.backward()
    torch.manual_seed(321)
    second_loss = eager(second)
    second_loss.backward()
    torch.testing.assert_close(first_loss, second_loss)
    torch.testing.assert_close(first.grad, second.grad)


def test_v4_accumulation_fork_uses_paired_sigreg_without_editing_upstream():
    original = v1_module.baseline.main
    try:
        restored = _install_configurable_accumulation(default_steps=8)
        assert restored is original
        constants = v1_module.baseline.main.__code__.co_consts
        assert "GRAD_ACCUM_STEPS" in constants
        stripped = {value.strip() for value in constants if isinstance(value, str)}
        assert {"policy_loss:", "latent_loss:", "sigreg_loss:"} <= stripped
        assert "local grad accumulation must be even for paired B=128 SIGReg" in constants
        assert "pending_sigreg_batch" in v1_module.baseline.main.__code__.co_varnames
    finally:
        v1_module.baseline.main = original


def test_v4_accumulation_fork_logs_weighted_extra_components():
    original = v1_module.baseline.main
    try:
        _install_configurable_accumulation(
            default_steps=8,
            extra_components=(("transition_loss", "transition_loss_weight"),),
        )
        constants = v1_module.baseline.main.__code__.co_consts
        stripped = {value.strip() for value in constants if isinstance(value, str)}
        assert "transition_loss:" in stripped
        assert "transition_loss_weight" in v1_module.baseline.main.__code__.co_names
    finally:
        v1_module.baseline.main = original


def test_v4_optional_training_components_preserve_eval_scalar_api():
    model = FreshLeJEPAV4(**KWARGS)
    model.return_loss_components = True
    ids = torch.randint(0, 32, (2, 5))
    targets = torch.randint(0, 32, (2, 5))
    total, components = model(ids, targets)
    assert total.ndim == 0
    assert components.shape == (3,)
    model.eval()
    assert model(ids, targets).ndim == 0


def test_v4_derivative_entrypoints_enable_component_tuple_contract():
    original_v1_main = v1_module.main
    original_model_class = v1_module.FreshLeJEPAGPT
    original_baseline_main = v1_module.baseline.main
    observed = []

    def fake_v1_main():
        observed.append(v1_module.FreshLeJEPAGPT.return_loss_components)

    try:
        v1_module.main = fake_v1_main
        for module, model_class in (
            (v4_predicted_module, FreshLeJEPAV4PredictedOnly),
            (v4_dropout_module, FreshLeJEPAV4PredictorDropout),
        ):
            model_class.return_loss_components = False
            module.main()
        assert observed == [True, True]
    finally:
        v1_module.main = original_v1_main
        v1_module.FreshLeJEPAGPT = original_model_class
        v1_module.baseline.main = original_baseline_main


def test_v4_incremental_matches_full_sequence():
    model = FreshLeJEPAV4(**KWARGS).eval()
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.attn.proj.weight, std=0.05)
            torch.nn.init.normal_(block.mlp.proj.weight, std=0.05)
        torch.nn.init.normal_(model.critic_probe.output.weight, std=0.05)
    ids = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        full_logits, full_values = model.policy_logits(ids), model.values(ids)
        caches = model.make_generation_cache(2, 6, ids.device)
        logits_steps, value_steps = [], []
        for position in range(6):
            logits, values, caches = model.generation_step(ids[:, position], caches, position)
            logits_steps.append(logits)
            value_steps.append(values)
    torch.testing.assert_close(torch.stack(logits_steps, 1), full_logits, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(torch.stack(value_steps, 1), full_values, rtol=2e-4, atol=2e-4)


def test_v4_folded_input_projector_is_exact_and_nonpersistent():
    model = FreshLeJEPAV4(**KWARGS).eval()
    ids = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        expected = model.embed_tokens(ids)
        state_keys = set(model.state_dict())
        model.fold_input_projector_for_inference()
        actual = model.embed_tokens(ids)
    torch.testing.assert_close(actual, expected)
    assert set(model.state_dict()) == state_keys
    model.clear_folded_input_projector()
    assert not hasattr(model, "_folded_token_latents")


def test_v4_predicted_only_preserves_shape_and_removes_token_branch():
    model = FreshLeJEPAV4PredictedOnly(**KWARGS)
    token = torch.randn(2, 5, 32)
    predicted = torch.randn(2, 5, 32)
    features = model.probe_features(token, predicted)
    torch.testing.assert_close(features[..., :32], torch.zeros_like(token))
    torch.testing.assert_close(features[..., 32:], predicted)


def test_v4_predictor_dropout_changes_no_weights_and_is_eval_deterministic():
    torch.manual_seed(789)
    control = FreshLeJEPAV4(**KWARGS)
    torch.manual_seed(789)
    dropout = FreshLeJEPAV4PredictorDropout(**KWARGS)
    control_state, dropout_state = control.state_dict(), dropout.state_dict()
    assert control_state.keys() == dropout_state.keys()
    for name in control_state:
        torch.testing.assert_close(dropout_state[name], control_state[name])
    ids = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        for block in dropout.blocks:
            torch.nn.init.normal_(block.attn.proj.weight, std=0.05)
            torch.nn.init.normal_(block.mlp.proj.weight, std=0.05)
    dropout.train()
    train_first = dropout.latent_features(ids)[1]
    train_second = dropout.latent_features(ids)[1]
    assert not torch.allclose(train_first, train_second)
    dropout.eval()
    with torch.no_grad():
        first = dropout.latent_features(ids)[1]
        second = dropout.latent_features(ids)[1]
    torch.testing.assert_close(first, second)


def test_v5_uses_prenorm_swiglu_probes_and_detaches_world_model():
    model = FreshLeJEPAV5SwiGLU(**KWARGS)
    assert isinstance(model.policy_probe, SwiGLUPolicyProbe)
    assert isinstance(model.critic_probe, SwiGLUCriticProbe)
    assert isinstance(model.policy_probe.block1, PreNormSwiGLUBlock)
    ids = torch.randint(0, 32, (2, 5))
    model.policy_logits(ids).sum().backward()
    assert model.policy_probe.output.weight.grad is not None
    assert model.tok_emb.weight.grad is None
    assert model.latent_projector.input.weight.grad is None


def test_v5_probe_rng_does_not_change_world_model_initialization():
    torch.manual_seed(987)
    v4 = FreshLeJEPAV4(**KWARGS)
    v4_next_random = torch.randn(8)
    torch.manual_seed(987)
    v5 = FreshLeJEPAV5SwiGLU(**KWARGS)
    v5_next_random = torch.randn(8)
    v4_state, v5_state = v4.state_dict(), v5.state_dict()
    for name, value in v4_state.items():
        if "policy_probe" not in name and "critic_probe" not in name:
            assert torch.equal(v5_state[name], value), name
    assert torch.equal(v5_next_random, v4_next_random)


def test_v5_dropout_only_changes_no_state_or_probe_inputs():
    torch.manual_seed(111)
    control = FreshLeJEPAV5SwiGLU(**KWARGS)
    torch.manual_seed(111)
    dropout = FreshLeJEPAV5DropoutOnly(**KWARGS)
    for name, value in control.state_dict().items():
        assert torch.equal(dropout.state_dict()[name], value), name
    token, predicted = torch.randn(2, 5, 32), torch.randn(2, 5, 32)
    assert dropout.probe_features(token, predicted).shape[-1] == 64
    assert dropout.predictor_dropout == 0.1


def test_v5_belief_only_changes_only_probe_architecture_and_input():
    torch.manual_seed(112)
    control = FreshLeJEPAV5SwiGLU(**KWARGS)
    torch.manual_seed(112)
    belief_only = FreshLeJEPAV5BeliefOnly(**KWARGS)
    control_state, belief_state = control.state_dict(), belief_only.state_dict()
    for name, value in control_state.items():
        if "policy_probe" not in name and "critic_probe" not in name:
            assert torch.equal(belief_state[name], value), name
    token = torch.randn(2, 5, 32)
    predicted = torch.randn(2, 5, 32, requires_grad=True)
    features = belief_only.probe_features(token, predicted)
    assert features.shape[-1] == 32
    assert not features.requires_grad


def test_v6_probe_is_strictly_belief_only_and_has_predictor_dropout():
    model = FreshLeJEPAV6BeliefDropout(**KWARGS)
    assert isinstance(model.policy_probe, BeliefOnlyPolicyProbe)
    assert isinstance(model.critic_probe, BeliefOnlyCriticProbe)
    assert not hasattr(model.policy_probe, "token")
    assert not hasattr(model.critic_probe, "token")
    token = torch.randn(2, 5, 32)
    belief = torch.randn(2, 5, 32, requires_grad=True)
    features = model.probe_features(token, belief)
    assert features.shape == belief.shape
    torch.testing.assert_close(features, belief)
    assert features.data_ptr() == belief.data_ptr()
    assert not features.requires_grad
    assert model.predictor_dropout == 0.1


def test_v6_dropout_adds_no_world_parameters_and_eval_generation_matches_full():
    torch.manual_seed(654)
    v5 = FreshLeJEPAV5SwiGLU(**KWARGS)
    torch.manual_seed(654)
    v6 = FreshLeJEPAV6BeliefDropout(**KWARGS)
    v5_state, v6_state = v5.state_dict(), v6.state_dict()
    for name, value in v5_state.items():
        if "policy_probe" not in name and "critic_probe" not in name:
            assert torch.equal(v6_state[name], value), name
    with torch.no_grad():
        for block in v6.blocks:
            torch.nn.init.normal_(block.attn.proj.weight, std=0.05)
            torch.nn.init.normal_(block.mlp.proj.weight, std=0.05)
        torch.nn.init.normal_(v6.critic_probe.output.weight, std=0.05)
    v6.eval()
    ids = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        full_logits, full_values = v6.policy_logits(ids), v6.values(ids)
        caches = v6.make_generation_cache(2, 6, ids.device)
        logits_steps, value_steps = [], []
        for position in range(6):
            logits, values, caches = v6.generation_step(
                ids[:, position], caches, position
            )
            logits_steps.append(logits)
            value_steps.append(values)
    torch.testing.assert_close(torch.stack(logits_steps, 1), full_logits, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(torch.stack(value_steps, 1), full_values, rtol=2e-4, atol=2e-4)


def test_v7_policy_uses_pre_prediction_projector_belief():
    model = FreshLeJEPAV7PreProjBelief(**KWARGS).eval()
    ids = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        model.policy_probe.output.weight.normal_(std=0.05)
        before = model.policy_logits(ids)
        model.prediction_projector.output.weight.normal_(std=3.0)
        model.prediction_projector.output.bias.normal_(std=3.0)
        after = model.policy_logits(ids)
    torch.testing.assert_close(after, before)


def test_v7_generation_uses_same_preprojection_belief_as_full_forward():
    model = FreshLeJEPAV7PreProjBelief(**KWARGS)
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.attn.proj.weight, std=0.05)
            torch.nn.init.normal_(block.mlp.proj.weight, std=0.05)
        torch.nn.init.normal_(model.policy_probe.output.weight, std=0.05)
        torch.nn.init.normal_(model.critic_probe.output.weight, std=0.05)
    model.eval()
    ids = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        full_logits, full_values = model.policy_logits(ids), model.values(ids)
        caches = model.make_generation_cache(2, 6, ids.device)
        logits_steps, value_steps = [], []
        for position in range(6):
            logits, values, caches = model.generation_step(
                ids[:, position], caches, position
            )
            logits_steps.append(logits)
            value_steps.append(values)
    torch.testing.assert_close(torch.stack(logits_steps, 1), full_logits, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(torch.stack(value_steps, 1), full_values, rtol=2e-4, atol=2e-4)


def test_v8_shared_token_encoder_starts_as_exact_embedding_rmsnorm():
    model = FreshLeJEPAV8TokenSwiGLU(**KWARGS)
    assert isinstance(model.token_encoder, SharedTokenEncoder)
    ids = torch.randint(0, 32, (2, 6))
    raw = model.tok_emb(ids)
    expected = torch.nn.functional.rms_norm(raw, (32,))
    actual = model.token_encoder(raw)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert "token_encoder.block1.gate.weight" in dict(model.blocks[-1].named_parameters())


def test_v8_input_and_target_are_overlapping_views_of_one_shared_encoding():
    model = FreshLeJEPAV8TokenSwiGLU(**KWARGS).train()
    ids = torch.randint(0, 32, (2, 6))
    token, _, target = model.training_latents(ids[:, :-1], ids[:, 1:])
    torch.testing.assert_close(token[:, 1:], target[:, :-1])


def test_v8_folded_shared_encoder_and_projector_are_exact():
    model = FreshLeJEPAV8TokenSwiGLU(**KWARGS).eval()
    ids = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        expected = model.embed_tokens(ids)
        model.fold_input_projector_for_inference()
        actual = model.embed_tokens(ids)
    torch.testing.assert_close(actual, expected)


def test_v8_adds_only_token_encoder_parameters_to_v7_world_state():
    torch.manual_seed(246)
    v7 = FreshLeJEPAV7PreProjBelief(**KWARGS)
    v7_next_random = torch.randn(8)
    torch.manual_seed(246)
    v8 = FreshLeJEPAV8TokenSwiGLU(**KWARGS)
    v8_next_random = torch.randn(8)
    v7_state, v8_state = v7.state_dict(), v8.state_dict()
    for name, value in v7_state.items():
        assert torch.equal(v8_state[name], value), name
    assert torch.equal(v8_next_random, v7_next_random)


def test_jedi_variant_preserves_v5_and_adds_only_training_denoiser():
    torch.manual_seed(321)
    control = FreshLeJEPAV5SwiGLU(**KWARGS).eval()
    control_next_random = torch.randn(8)
    torch.manual_seed(321)
    jedi = FreshLeJEPAV5JEDIDenoising(**KWARGS).eval()
    jedi_next_random = torch.randn(8)
    jedi_state = jedi.state_dict()
    for name, value in control.state_dict().items():
        assert torch.equal(jedi_state[name], value), name
    assert torch.equal(jedi_next_random, control_next_random)

    ids = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        control_logits = control.policy_logits(ids)
        jedi_logits = jedi.policy_logits(ids)
        jedi.latent_denoiser.output.weight.normal_(std=10)
        changed_denoiser_logits = jedi.policy_logits(ids)
    torch.testing.assert_close(jedi_logits, control_logits)
    torch.testing.assert_close(changed_denoiser_logits, jedi_logits)


def test_jedi_denoising_detaches_target_but_trains_causal_context():
    model = FreshLeJEPAV5JEDIDenoising(**KWARGS).train()
    with torch.no_grad():
        model.latent_denoiser.output.weight.normal_(std=0.05)
    context = torch.randn(2, 5, 32, requires_grad=True)
    target = torch.randn(2, 5, 32, requires_grad=True)
    sigma = torch.tensor([0.25, 2.0])
    noise = torch.randn_like(target)
    loss = model.latent_denoising_loss(context, target, sigma, noise)
    loss.backward()
    assert context.grad is not None
    assert context.grad.abs().sum() > 0
    assert target.grad is None
    assert model.latent_denoiser.context_input.weight.grad is not None


def test_jedi_f_space_loss_matches_weighted_clean_reconstruction_loss():
    model = FreshLeJEPAV5JEDIDenoising(**KWARGS).train()
    context = torch.randn(2, 5, 32)
    clean = torch.randn(2, 5, 32)
    sigma = torch.tensor([0.1, 1.5])
    noise = torch.randn_like(clean)
    noisy = clean + sigma[:, None, None] * noise
    c_skip, c_out, _, _ = edm_coefficients(sigma)
    prediction = model.latent_denoiser.network_output(noisy, context, sigma)
    target = (
        clean - c_skip[:, None, None] * noisy
    ) / c_out[:, None, None]
    f_loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
    denoised = (
        c_skip[:, None, None] * noisy
        + c_out[:, None, None] * prediction.float()
    )
    weights = c_out.reciprocal().square()[:, None, None]
    weighted_clean_loss = ((denoised - clean).square() * weights).mean()
    torch.testing.assert_close(f_loss, weighted_clean_loss)


def test_jedi_denoiser_preserves_model_compute_dtype():
    model = FreshLeJEPAV5JEDIDenoising(**KWARGS).bfloat16()
    context = torch.randn(2, 5, 32, dtype=torch.bfloat16)
    noisy = torch.randn_like(context)
    sigma = torch.tensor([0.2, 1.0])
    prediction = model.latent_denoiser.network_output(noisy, context, sigma)
    assert prediction.dtype == torch.bfloat16


def test_jedi_sigma_sampler_uses_bounded_diamond_distribution():
    model = FreshLeJEPAV5JEDIDenoising(**KWARGS)
    model.edm_p_std = 100.0
    sigma = model.sample_edm_sigma(1024, torch.device("cpu"))
    assert sigma.min() >= model.edm_sigma_min
    assert sigma.max() <= model.edm_sigma_max


def test_jedi_low_sigma_target_is_constructed_in_fp32():
    model = FreshLeJEPAV5JEDIDenoising(**KWARGS).bfloat16().train()
    context = torch.zeros(1, 1, 32, dtype=torch.bfloat16)
    clean = torch.ones_like(context)
    noise = torch.ones_like(context)
    sigma = torch.tensor([0.002])

    # The denoiser output layer is zero initialized, so the loss is exactly
    # the squared FP32 F-space target.  Constructing `noisy` in BF16 would
    # round clean + sigma * noise back to clean and fail this comparison.
    loss = model.latent_denoising_loss(context, clean, sigma, noise)
    c_skip, c_out, _, _ = edm_coefficients(sigma)
    noisy_fp32 = clean.float() + sigma[:, None, None] * noise.float()
    expected_target = (
        clean.float() - c_skip[:, None, None] * noisy_fp32
    ) / c_out[:, None, None]
    torch.testing.assert_close(loss, expected_target.square().mean())


def test_v9_preserves_v5_policy_and_rng_lineage():
    torch.manual_seed(543)
    control = FreshLeJEPAV5SwiGLU(**KWARGS).eval()
    control_next_random = torch.randn(8)
    torch.manual_seed(543)
    transition = FreshLeJEPAV9BeliefTransition(**KWARGS).eval()
    transition_next_random = torch.randn(8)
    transition_state = transition.state_dict()
    for name, value in control.state_dict().items():
        assert torch.equal(transition_state[name], value), name
    assert torch.equal(transition_next_random, control_next_random)

    ids = torch.randint(0, 32, (2, 6))
    targets = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        torch.testing.assert_close(
            transition.policy_logits(ids), control.policy_logits(ids)
        )
        torch.testing.assert_close(
            transition(ids, targets), control(ids, targets)
        )


def test_v9_teacher_transition_is_causal_and_action_conditioned():
    model = FreshLeJEPAV9BeliefTransition(**KWARGS).eval()
    ids = torch.randint(0, 32, (2, 6))
    targets = torch.cat((ids[:, 1:], torch.randint(0, 32, (2, 1))), dim=1)
    token, predicted, action, belief, next_belief = model.teacher_transition_latents(
        ids, targets
    )
    trajectory_tokens = model.embed_tokens(
        torch.cat((ids, targets[:, -1:]), dim=1)
    )
    trajectory_beliefs = model.temporal_belief_from_token_latent(trajectory_tokens)
    torch.testing.assert_close(token, trajectory_tokens[:, :-1])
    torch.testing.assert_close(action, trajectory_tokens[:, 1:])
    torch.testing.assert_close(belief, trajectory_beliefs[:, :-1])
    torch.testing.assert_close(next_belief, trajectory_beliefs[:, 1:])
    torch.testing.assert_close(predicted, model.prediction_latent(belief))


def test_v9_transition_detaches_future_but_trains_state_and_action():
    model = FreshLeJEPAV9BeliefTransition(**KWARGS).train()
    with torch.no_grad():
        model.belief_transition.output.weight.normal_(std=0.05)
    belief = torch.randn(2, 5, 32, requires_grad=True)
    action = torch.randn(2, 5, 32, requires_grad=True)
    future = torch.randn(2, 5, 32, requires_grad=True)
    sigma = torch.tensor([0.25, 2.0])
    noise = torch.randn_like(future)
    loss, belief_gain, action_gain = model.transition_denoising_loss(
        belief, action, future, sigma, noise
    )
    loss.backward()
    assert belief.grad is not None and belief.grad.abs().sum() > 0
    assert action.grad is not None and action.grad.abs().sum() > 0
    assert future.grad is None
    assert belief_gain.ndim == 0 and not belief_gain.requires_grad
    assert action_gain.ndim == 0 and not action_gain.requires_grad


def test_v9_forward_keeps_direct_mse_and_logs_transition_separately():
    model = FreshLeJEPAV9BeliefTransition(**KWARGS).train()
    model.return_loss_components = True
    ids = torch.randint(0, 32, (2, 5))
    targets = torch.randint(0, 32, (2, 5))
    total, components = model(ids, targets)
    assert components.shape == (6,)
    expected = (
        components[0]
        + model.latent_loss_weight * components[1]
        + model.sigreg_loss_weight * components[2]
        + model.transition_loss_weight * components[3]
    )
    torch.testing.assert_close(total.detach(), expected)


def test_v9_three_step_imagination_is_deterministic_for_fixed_noise():
    model = FreshLeJEPAV9BeliefTransition(**KWARGS).eval()
    belief = torch.randn(2, 1, 32)
    action = torch.randn_like(belief)
    noise = torch.randn_like(belief)
    first = model.imagine_next_belief(belief, action, noise=noise, steps=3)
    second = model.imagine_next_belief(belief, action, noise=noise, steps=3)
    torch.testing.assert_close(first, second)
    assert first.shape == belief.shape
    assert torch.isfinite(first).all()
    torch.testing.assert_close(
        first.square().mean(-1), torch.ones_like(first[..., 0]), rtol=2e-3, atol=2e-3
    )

    flat_belief = belief[:, 0]
    flat_action = action[:, 0]
    flat_noise = noise[:, 0]
    flat = model.imagine_next_belief(
        flat_belief, flat_action, noise=flat_noise, steps=3
    )
    torch.testing.assert_close(flat, first[:, 0])
