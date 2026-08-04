"""CPU tests for the unified categorical-energy/JEPA barycenter model."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from energy_readout import fresh_lejepa_train_unified_energy_barycenter as unified
from energy_readout.test_jepa_rbf_decoder import (
    BATCH,
    DIM,
    SEQ,
    VOCAB,
    batch,
    model_kwargs,
)


def build_model() -> unified.UnifiedEnergyBarycenterLeJEPA:
    torch.manual_seed(0)
    model = unified.UnifiedEnergyBarycenterLeJEPA(**model_kwargs())
    model.return_loss_components = True
    return model


def latent_inputs(
    model: unified.UnifiedEnergyBarycenterLeJEPA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, target_ids = batch()
    token, belief, predicted, target = model.training_latents_with_belief(
        input_ids, target_ids
    )
    return token, belief, predicted, target, target_ids


def test_distribution_and_jepa_prediction_share_exact_logits() -> None:
    model = build_model().eval()
    _token, belief, predicted, _target, _target_ids = latent_inputs(model)
    with torch.no_grad():
        logits, predicted_mean = model.distribution_outputs(predicted, belief)
        probabilities = F.softmax(logits, dim=-1)
        expected_mean = probabilities @ model.semantic_codebook().float()
    assert logits.shape == (BATCH, SEQ, VOCAB)
    assert predicted_mean.shape == (BATCH, SEQ, DIM)
    assert torch.allclose(predicted_mean, expected_mean, atol=1e-6)
    assert torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones(BATCH, SEQ),
        atol=1e-6,
    )


def test_fast_energy_is_exactly_distance_softmax_up_to_constant() -> None:
    model = build_model().eval()
    _token, belief, predicted, _target, _target_ids = latent_inputs(model)
    with torch.no_grad():
        logits, codebook = model.energy_logits_and_codebook(predicted, belief)
        precision = model.bounded_precision(
            model.position_log_sigma(belief)
        ).unsqueeze(-1)
        direct = model.rbf_owner.rbf_token_bias - 0.5 * precision * torch.cdist(
            predicted.float(), codebook.float()
        ).square()
    assert torch.allclose(
        F.softmax(logits, dim=-1),
        F.softmax(direct, dim=-1),
        atol=2e-5,
    )


def test_bfloat16_energy_preserves_near_code_ordering() -> None:
    model = build_model().eval()
    torch.manual_seed(7)
    predicted = torch.randn(1, 1, DIM, dtype=torch.bfloat16)
    belief = torch.zeros_like(predicted)
    codebook = torch.randn(VOCAB, DIM, dtype=torch.bfloat16)
    codebook[0] = (predicted[0, 0].float() + 0.01).to(torch.bfloat16)
    codebook[1] = (predicted[0, 0].float() + 0.05).to(torch.bfloat16)
    original = model.semantic_codebook
    try:
        model.semantic_codebook = lambda: codebook
        with torch.no_grad():
            model.rbf_owner.rbf_log_sigma_weight.zero_()
            model.rbf_owner.rbf_log_sigma_bias.zero_()
            model.rbf_owner.rbf_token_bias.zero_()
            logits = model.energy_logits(predicted, belief)
            direct = -0.5 * torch.cdist(
                predicted.float(), codebook.float()
            ).square()
        assert logits.argmax(dim=-1).item() == direct.argmax(dim=-1).item()
        assert logits[0, 0, 0] > logits[0, 0, 1]
    finally:
        model.semantic_codebook = original


def test_semantic_neighbor_gets_less_barycentric_error() -> None:
    target_code = torch.zeros(1, 1, 2)
    codebook = torch.tensor(
        [
            [0.0, 0.0],   # target: kitten
            [0.1, 0.0],   # nearby: cat
            [5.0, 0.0],   # distant: fire
        ]
    )
    neighbor_logits = torch.tensor([[[0.0, 12.0, -12.0]]])
    distant_logits = torch.tensor([[[0.0, -12.0, 12.0]]])
    neighbor_mean = unified.UnifiedEnergyBarycenterLeJEPA.categorical_barycenter(
        neighbor_logits, codebook
    )
    distant_mean = unified.UnifiedEnergyBarycenterLeJEPA.categorical_barycenter(
        distant_logits, codebook
    )
    neighbor_error = F.mse_loss(neighbor_mean, target_code)
    distant_error = F.mse_loss(distant_mean, target_code)
    assert neighbor_error < distant_error / 1000.0


def test_opposite_distant_mass_cannot_cancel_semantic_error() -> None:
    codebook = torch.tensor([[-10.0], [0.0], [10.0]])
    target_code = torch.zeros(1, 1, 1)
    distant_logits = torch.tensor([[[12.0, -20.0, 12.0]]])
    target_logits = torch.tensor([[[-20.0, 12.0, -20.0]]])
    distant_mean = unified.UnifiedEnergyBarycenterLeJEPA.categorical_barycenter(
        distant_logits, codebook
    )
    assert torch.allclose(distant_mean, target_code, atol=1e-5)
    distant_loss = (
        unified.UnifiedEnergyBarycenterLeJEPA.semantic_distribution_loss(
            distant_logits, codebook, target_code
        )
    )
    target_loss = (
        unified.UnifiedEnergyBarycenterLeJEPA.semantic_distribution_loss(
            target_logits, codebook, target_code
        )
    )
    assert distant_loss > 99.0
    assert target_loss < 1e-6


def test_semantic_loss_equals_full_vocab_expected_distance() -> None:
    torch.manual_seed(8)
    logits = torch.randn(2, 3, 7)
    codebook = torch.randn(7, 5)
    target = torch.randn(2, 3, 5)
    actual = unified.UnifiedEnergyBarycenterLeJEPA.semantic_distribution_loss(
        logits, codebook, target
    )
    probabilities = F.softmax(logits, dim=-1)
    distance_sq = (target.unsqueeze(-2) - codebook).square().sum(dim=-1)
    expected = (probabilities * distance_sq).sum(dim=-1).div(5).mean()
    assert torch.allclose(actual, expected, atol=1e-6)


def test_live_codebook_is_the_attached_target_encoder() -> None:
    model = build_model().train()
    token_ids = torch.arange(VOCAB)
    actual = model.semantic_codebook()
    expected = model.embed_tokens(token_ids)
    assert actual.requires_grad
    assert torch.allclose(actual, expected, atol=1e-6)


def test_token_nll_reaches_the_shared_geometry() -> None:
    model = build_model().train()
    _token, belief, predicted, _target, target_ids = latent_inputs(model)
    model.zero_grad(set_to_none=True)
    logits = model.energy_logits(predicted, belief)
    loss = F.cross_entropy(logits.flatten(0, 1), target_ids.flatten())
    loss.backward()

    required_fragments = (
        "tok_emb.weight",
        "latent_projector",
        "prediction_projector",
        "attn",
        "rbf_log_sigma_weight",
        "rbf_log_sigma_bias",
        "rbf_token_bias",
    )
    named = dict(model.named_parameters())
    for fragment in required_fragments:
        gradients = [
            parameter.grad
            for name, parameter in named.items()
            if fragment in name
        ]
        assert gradients, fragment
        assert any(
            gradient is not None and float(gradient.abs().sum()) > 0.0
            for gradient in gradients
        ), fragment


def test_semantic_distribution_loss_reaches_probabilities_and_target_codes() -> None:
    model = build_model().train()
    _token, belief, predicted, target, _target_ids = latent_inputs(model)
    model.zero_grad(set_to_none=True)
    logits, codebook = model.energy_logits_and_codebook(predicted, belief)
    loss = model.semantic_distribution_loss(logits, codebook, target)
    loss.backward()
    assert model.rbf_owner.rbf_log_sigma_weight.grad is not None
    assert model.rbf_owner.rbf_log_sigma_weight.grad.abs().sum() > 0
    assert model.tok_emb.weight.grad is not None
    assert model.tok_emb.weight.grad.abs().sum() > 0


def test_full_training_loss_uses_expected_semantic_distance() -> None:
    model = build_model().train()
    input_ids, target_ids = batch()
    total, components = model(input_ids, target_ids)
    with torch.no_grad():
        _token, belief, predicted, target = model.training_latents_with_belief(
            input_ids, target_ids
        )
        logits, codebook = model.energy_logits_and_codebook(predicted, belief)
        expected_policy = F.cross_entropy(
            logits.flatten(0, 1), target_ids.flatten()
        )
        expected_latent = model.semantic_distribution_loss(
            logits, codebook, target
        )
    assert components.shape == (3,)
    assert torch.allclose(components[0], expected_policy, atol=1e-6)
    assert torch.allclose(components[1], expected_latent, atol=1e-6)
    assert float(components[2]) == 0.0
    expected_total = expected_policy + model.latent_loss_weight * expected_latent
    assert torch.allclose(total.detach(), expected_total, atol=1e-6)


def test_eval_forward_remains_exact_token_nll() -> None:
    model = build_model().eval()
    input_ids, target_ids = batch()
    with torch.no_grad():
        actual = model(input_ids, target_ids)
        expected = F.cross_entropy(
            model.policy_logits(input_ids).flatten(0, 1),
            target_ids.flatten(),
        )
    assert torch.allclose(actual, expected, atol=1e-6)


def test_cached_sampling_logits_match_full_forward() -> None:
    model = build_model().eval()
    input_ids, _target_ids = batch()
    with torch.no_grad():
        expected = model.policy_logits(input_ids)
        caches = model.make_generation_cache(
            BATCH, SEQ, torch.device("cpu"), dtype=model.tok_emb.weight.dtype
        )
        steps = []
        for position in range(SEQ):
            logits, caches = model.generation_policy_step(
                input_ids[:, position], caches, position
            )
            steps.append(logits)
        actual = torch.stack(steps, dim=1)
    assert torch.allclose(actual, expected, atol=2e-5)


def test_checkpoint_loader_reconstructs_unified_architecture() -> None:
    from postraining import model_io

    torch.manual_seed(0)
    with model_io._pope_construction():
        source = unified.UnifiedEnergyBarycenterLeJEPA(**model_kwargs())
    payload = {
        "model": source.state_dict(),
        "metadata": {
            "architecture": unified.ARCHITECTURE,
            "model": model_kwargs(),
            "optimizer": {"train_seq_len": 2048},
        },
    }
    loaded = model_io.load_model(
        "unused-with-explicit-payload.pt",
        torch.device("cpu"),
        payload=payload,
    )
    assert isinstance(loaded, unified.UnifiedEnergyBarycenterLeJEPA)
    assert loaded.architecture == unified.ARCHITECTURE
    assert loaded.train_context_tokens == 2048


def test_checkpoint_restores_precision_domain_from_state() -> None:
    from postraining import model_io

    torch.manual_seed(0)
    with model_io._pope_construction():
        source = unified.UnifiedEnergyBarycenterLeJEPA(**model_kwargs())
    with torch.no_grad():
        source.rbf_owner.rbf_log_precision_bounds_nano.copy_(
            torch.tensor([-7_000_000_000, 3_000_000_000])
        )
    payload = {
        "model": source.state_dict(),
        "metadata": {
            "architecture": unified.ARCHITECTURE,
            "model": model_kwargs(),
        },
    }
    loaded = model_io.load_model(
        "unused-with-explicit-payload.pt",
        torch.device("cpu"),
        payload=payload,
    )
    bounds = loaded.rbf_owner.rbf_log_precision_bounds_nano
    assert torch.equal(bounds, torch.tensor([-7_000_000_000, 3_000_000_000]))


def test_precision_domain_survives_bfloat16_model_cast() -> None:
    model = build_model().bfloat16()
    assert model.rbf_owner.rbf_log_precision_bounds_nano.dtype == torch.int64
    reference = torch.zeros((), dtype=torch.float32)
    lower, upper = model.precision_log_bounds(reference)
    assert torch.allclose(lower, torch.tensor(-11.512925), atol=1e-6)
    assert torch.allclose(upper, torch.tensor(4.605170), atol=1e-6)


def test_metadata_records_one_shared_distribution() -> None:
    metadata = unified.UnifiedEnergyBarycenterLeJEPA.experiment_metadata()
    assert metadata["jepa_prediction"] == "categorical_codebook_mean_embedding"
    assert metadata["categorical_objective"] == "exact_token_nll"
    assert metadata["semantic_partial_credit"] == (
        "tokenwise_expected_code_distance"
    )
    assert metadata["prediction_gradient_from_token_nll"] == "attached"
    assert metadata["codebook_gradient_from_token_nll"] == "attached"


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
