"""CPU tests for the geometry-preserving LeJEPA RBF decoder.

Run directly: ``python3 energy_readout/test_jepa_rbf_decoder.py``.
These focused unit tests execute no GPU workload and are safe outside mlq.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from energy_readout import fresh_lejepa_train_jepa_rbf_decoder as rbf
from pretraining.fresh_lejepa import fresh_lejepa_train as v1
from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_rms_pope_belief_attached import (
    FreshLeJEPASharedRMSV1PoPEBeliefAttachedCE,
)


VOCAB, LAYERS, DIM, HEADS, KV_HEADS = 48, 2, 64, 2, 1
BATCH, SEQ = 2, 12


def model_kwargs() -> dict[str, int | float | bool]:
    return {
        "vocab_size": VOCAB,
        "num_layers": LAYERS,
        "model_dim": DIM,
        "num_heads": HEADS,
        "num_kv_heads": KV_HEADS,
        "mlp_mult": v1.FreshHyperparameters.mlp_mult,
        "tie_embeddings": True,
        "tied_embed_init_std": v1.FreshHyperparameters.tied_embed_init_std,
        "logit_softcap": v1.FreshHyperparameters.logit_softcap,
        "rope_base": v1.FreshHyperparameters.rope_base,
        "qk_gain_init": v1.FreshHyperparameters.qk_gain_init,
    }


def build_model() -> rbf.GeometryPreservingRBFLeJEPA:
    torch.manual_seed(0)
    model = rbf.GeometryPreservingRBFLeJEPA(**model_kwargs())
    model.return_loss_components = True
    return model


def batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    trajectory = torch.randint(0, VOCAB, (BATCH, SEQ + 1))
    return trajectory[:, :-1], trajectory[:, 1:]


def decoder_inputs(
    model: rbf.GeometryPreservingRBFLeJEPA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, target_ids = batch()
    _token, belief, predicted, _target = model.training_latents_with_belief(
        input_ids, target_ids
    )
    return belief, predicted, input_ids, target_ids


def test_decoder_replaces_probes_with_only_vector_parameters() -> None:
    model = build_model()
    names = dict(model.named_parameters())
    assert not any("probe" in name for name in names), tuple(names)
    assert names["blocks.1.rbf_log_sigma_weight"].shape == (DIM,)
    assert names["blocks.1.rbf_log_sigma_bias"].shape == ()
    assert names["blocks.1.rbf_token_bias"].shape == (VOCAB,)
    expected_log_sigma = 0.25 * math.log(DIM)
    assert math.isclose(
        float(model.rbf_owner.rbf_log_sigma_bias.detach()),
        expected_log_sigma,
        abs_tol=1e-6,
    )


def test_surviving_jepa_initialization_matches_parent_exactly() -> None:
    torch.manual_seed(0)
    parent = FreshLeJEPASharedRMSV1PoPEBeliefAttachedCE(**model_kwargs())
    child = build_model()
    parent_state = parent.state_dict()
    child_state = child.state_dict()
    shared = set(parent_state) & set(child_state)
    assert shared
    assert not any("probe" in name for name in child_state)
    for name in shared:
        assert torch.equal(parent_state[name], child_state[name]), name


def test_decoder_parameters_route_to_scalar_adam_group() -> None:
    model = build_model()
    block_named = list(model.blocks.named_parameters())
    rbf_params = [
        (name, parameter)
        for name, parameter in block_named
        if "rbf_" in name
    ]
    assert len(rbf_params) == 3
    assert all(parameter.ndim < 2 for _, parameter in rbf_params), rbf_params


def test_semantic_codebook_matches_target_projector() -> None:
    model = build_model().eval()
    token_ids = torch.arange(VOCAB)
    with torch.no_grad():
        expected = model.embed_tokens(token_ids)
        actual = model.semantic_codebook()
    assert actual.requires_grad is False
    assert torch.allclose(actual, expected, atol=1e-6)


def test_initial_uncertainty_is_position_independent() -> None:
    model = build_model().eval()
    belief, _predicted, _inputs, _targets = decoder_inputs(model)
    with torch.no_grad():
        log_sigma = model.position_log_sigma(belief)
    expected = torch.full_like(log_sigma, 0.25 * math.log(DIM))
    assert torch.allclose(log_sigma, expected)


def test_uncertainty_head_is_context_dependent_after_learning() -> None:
    model = build_model().eval()
    belief, _predicted, _inputs, _targets = decoder_inputs(model)
    with torch.no_grad():
        model.rbf_owner.rbf_log_sigma_weight.copy_(torch.linspace(-0.2, 0.2, DIM))
        log_sigma = model.position_log_sigma(belief)
    assert float(log_sigma.std()) > 1e-4


def test_smaller_sigma_sharpens_the_categorical_distribution() -> None:
    model = build_model().eval()
    belief, predicted, _inputs, _targets = decoder_inputs(model)
    with torch.no_grad():
        model.rbf_owner.rbf_log_sigma_weight.zero_()
        model.rbf_owner.rbf_token_bias.zero_()
        model.rbf_owner.rbf_log_sigma_bias.fill_(math.log(0.5))
        sharp = F.softmax(model.energy_logits(predicted, belief), dim=-1)
        model.rbf_owner.rbf_log_sigma_bias.fill_(math.log(5.0))
        broad = F.softmax(model.energy_logits(predicted, belief), dim=-1)
        sharp_entropy = -(sharp * sharp.clamp_min(1e-12).log()).sum(dim=-1).mean()
        broad_entropy = -(broad * broad.clamp_min(1e-12).log()).sum(dim=-1).mean()
    assert sharp_entropy < broad_entropy, (sharp_entropy, broad_entropy)


def test_nearer_code_receives_higher_probability_with_equal_bias() -> None:
    model = build_model().eval()
    belief = torch.zeros(1, 1, DIM)
    predicted = torch.zeros(1, 1, DIM)
    codebook = torch.zeros(VOCAB, DIM)
    codebook[0, 0] = 0.25
    codebook[1, 0] = 4.0
    original = model.semantic_codebook
    try:
        model.semantic_codebook = lambda: codebook
        with torch.no_grad():
            model.rbf_owner.rbf_token_bias.zero_()
            logits = model.energy_logits(predicted, belief)
        assert logits[0, 0, 0] > logits[0, 0, 1]
    finally:
        model.semantic_codebook = original


def test_bfloat16_nearby_codes_use_stable_fp32_distances() -> None:
    model = build_model().eval()
    torch.manual_seed(4)
    predicted = torch.randn(1, 1, DIM, dtype=torch.bfloat16)
    belief = torch.zeros_like(predicted)
    codebook = torch.randn(VOCAB, DIM, dtype=torch.bfloat16)
    codebook[0] = (predicted[0, 0].float() + 0.01).to(torch.bfloat16)
    codebook[1] = (predicted[0, 0].float() + 0.04).to(torch.bfloat16)
    original = model.semantic_codebook
    try:
        model.semantic_codebook = lambda: codebook
        with torch.no_grad():
            model.rbf_owner.rbf_log_sigma_weight.zero_()
            model.rbf_owner.rbf_log_sigma_bias.zero_()
            model.rbf_owner.rbf_token_bias.zero_()
            logits = model.energy_logits(predicted, belief)
        expected = (
            predicted.float() - codebook[:2].float()[None, None]
        ).square().sum(dim=-1)
        recovered = -2.0 * logits[..., :2]
        assert torch.allclose(recovered, expected, atol=2e-5), (
            recovered,
            expected,
        )
        assert logits[0, 0, 0] > logits[0, 0, 1]
    finally:
        model.semantic_codebook = original


def test_extreme_log_sigma_values_keep_nll_and_gradients_finite() -> None:
    model = build_model().train()
    predicted = torch.zeros(1, 1, DIM)
    belief = torch.ones_like(predicted)
    target_ids = torch.ones(1, 1, dtype=torch.long)
    codebook = torch.zeros(VOCAB, DIM)
    codebook[0, 0] = 0.1
    codebook[1, 0] = 2.0
    original = model.semantic_codebook
    try:
        model.semantic_codebook = lambda: codebook
        for raw_log_sigma in (-100.0, 100.0):
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                model.rbf_owner.rbf_log_sigma_weight.zero_()
                model.rbf_owner.rbf_log_sigma_bias.fill_(raw_log_sigma)
                model.rbf_owner.rbf_token_bias.zero_()
            loss = model.token_nll(predicted, belief, target_ids)
            assert torch.isfinite(loss), (raw_log_sigma, loss)
            loss.backward()
            gradient = model.rbf_owner.rbf_log_sigma_bias.grad
            assert gradient is not None
            assert torch.isfinite(gradient), (raw_log_sigma, gradient)
        assert model.rbf_owner.rbf_log_sigma_bias.grad != 0
    finally:
        model.semantic_codebook = original


def test_token_nll_gradient_firewall() -> None:
    """NLL must train calibration parameters and no JEPA geometry parameter."""
    model = build_model().train()
    belief, predicted, _inputs, target_ids = decoder_inputs(model)
    model.zero_grad(set_to_none=True)
    nll = model.token_nll(predicted, belief, target_ids)
    nll.backward()

    owner = model.rbf_owner
    assert owner.rbf_log_sigma_weight.grad is not None
    assert owner.rbf_log_sigma_weight.grad.abs().sum() > 0
    assert owner.rbf_log_sigma_bias.grad is not None
    assert owner.rbf_log_sigma_bias.grad.abs() > 0
    assert owner.rbf_token_bias.grad is not None
    assert owner.rbf_token_bias.grad.abs().sum() > 0

    geometry_names = (
        "tok_emb",
        "latent_projector",
        "prediction_projector",
        "attn",
        "mlp",
    )
    leaked = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if any(part in name for part in geometry_names) and parameter.grad is not None
    }
    assert not leaked, tuple(leaked)


def test_full_training_loss_keeps_jepa_mse_attached() -> None:
    model = build_model().train()
    input_ids, target_ids = batch()
    model.zero_grad(set_to_none=True)
    total, components = model(input_ids, target_ids)
    expected = components[0] + model.latent_loss_weight * components[1]
    assert components.shape == (3,)
    assert float(components[2]) == 0.0  # paired SIGReg is deferred by this lineage
    assert torch.isclose(total.detach(), expected, atol=1e-6), (total, components)
    total.backward()

    named = dict(model.named_parameters())
    for fragment in ("tok_emb.weight", "latent_projector", "prediction_projector"):
        gradients = [
            parameter.grad
            for name, parameter in named.items()
            if fragment in name
        ]
        assert gradients, fragment
        assert all(gradient is not None for gradient in gradients), fragment
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0, fragment


def test_eval_forward_is_exact_token_nll() -> None:
    model = build_model().eval()
    input_ids, target_ids = batch()
    with torch.no_grad():
        loss = model(input_ids, target_ids)
        _token, belief, predicted, _target = model.training_latents_with_belief(
            input_ids, target_ids
        )
        expected = model.token_nll(predicted, belief, target_ids)
    assert torch.isclose(loss, expected, atol=1e-6)


def test_policy_logits_use_same_decoder_as_training() -> None:
    model = build_model().eval()
    input_ids, _target_ids = batch()
    with torch.no_grad():
        public = model.policy_logits(input_ids)
        token_latent = model.embed_tokens(input_ids)
        belief = model.temporal_belief_from_token_latent(token_latent)
        predicted = model.prediction_latent(belief)
        expected = model.energy_logits(predicted, belief)
    assert public.shape == (BATCH, SEQ, VOCAB)
    assert torch.allclose(public, expected, atol=1e-6)


def test_cached_policy_generation_matches_full_forward() -> None:
    model = build_model().eval()
    input_ids, _target_ids = batch()
    with torch.no_grad():
        expected = model.policy_logits(input_ids)
        caches = model.make_generation_cache(
            BATCH, SEQ, torch.device("cpu"), dtype=model.tok_emb.weight.dtype
        )
        actual_steps = []
        for position in range(SEQ):
            logits, caches = model.generation_policy_step(
                input_ids[:, position], caches, position
            )
            actual_steps.append(logits)
        actual = torch.stack(actual_steps, dim=1)
    assert torch.allclose(actual, expected, atol=2e-5), (
        (actual - expected).abs().max(),
        actual.shape,
    )


def test_actor_critic_generation_rejects_before_cache_mutation() -> None:
    model = build_model().eval()
    input_ids, _target_ids = batch()
    caches = model.make_generation_cache(BATCH, SEQ, torch.device("cpu"))
    for cache in caches:
        for tensor in cache:
            tensor.zero_()
    snapshots = [tuple(tensor.clone() for tensor in cache) for cache in caches]
    try:
        model.generation_step(input_ids[:, 0], caches, 0)
    except RuntimeError as error:
        assert "generation_policy_step" in str(error)
    else:
        raise AssertionError("actor-critic generation silently omitted its value")
    for cache, snapshot in zip(caches, snapshots, strict=True):
        for tensor, original in zip(cache, snapshot, strict=True):
            assert torch.equal(tensor, original)


def test_decoder_feature_contract_rejects_wrong_width() -> None:
    model = build_model()
    try:
        model.logits_from_features(torch.zeros(1, 1, DIM))
    except ValueError:
        pass
    else:
        raise AssertionError("decoder accepted a feature tensor missing belief")


def test_value_path_is_explicitly_unsupported() -> None:
    model = build_model()
    try:
        model.values_from_features(torch.zeros(1, 1, 2 * DIM))
    except RuntimeError:
        pass
    else:
        raise AssertionError("decoder silently supplied an untrained value")


def test_checkpoint_loader_reconstructs_rbf_architecture() -> None:
    from postraining import model_io

    torch.manual_seed(0)
    with model_io._pope_construction():
        source = rbf.GeometryPreservingRBFLeJEPA(**model_kwargs())
    payload = {
        "model": source.state_dict(),
        "metadata": {
            "architecture": rbf.ARCHITECTURE,
            "model": model_kwargs(),
        },
    }
    loaded = model_io.load_model(
        "unused-with-explicit-payload.pt", torch.device("cpu"), payload=payload
    )
    assert isinstance(loaded, rbf.GeometryPreservingRBFLeJEPA)
    assert loaded.architecture == rbf.ARCHITECTURE
    for name, parameter in loaded.named_parameters():
        expected_trainable = name.startswith("blocks.1.rbf_")
        assert parameter.requires_grad is expected_trainable, name


def test_standard_renderer_features_match_direct_policy_logits() -> None:
    from postraining.latent_thought import LatentThoughtModel

    model = build_model().eval()
    wrapper = LatentThoughtModel(model).eval()
    input_ids, _target_ids = batch()
    with torch.no_grad():
        expected = model.policy_logits(input_ids)
        actual = wrapper.policy_logits(input_ids)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert list(model.renderer_parameters()) == [
        model.rbf_owner.rbf_log_sigma_weight,
        model.rbf_owner.rbf_log_sigma_bias,
        model.rbf_owner.rbf_token_bias,
    ]


def main() -> None:
    tests = [
        (name, function)
        for name, function in sorted(globals().items())
        if name.startswith("test_") and callable(function)
    ]
    for name, function in tests:
        function()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
