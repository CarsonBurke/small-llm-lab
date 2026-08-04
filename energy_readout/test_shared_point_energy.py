"""CPU tests for the shared-point categorical-energy JEPA model."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from energy_readout import fresh_lejepa_train_shared_point_energy as shared
from energy_readout.test_jepa_rbf_decoder import (
    BATCH,
    DIM,
    SEQ,
    VOCAB,
    batch,
    model_kwargs,
)


def build_model() -> shared.SharedPointEnergyLeJEPA:
    from postraining import model_io

    torch.manual_seed(0)
    with model_io._pope_construction():
        model = shared.SharedPointEnergyLeJEPA(**model_kwargs())
    model.return_loss_components = True
    return model


def latent_inputs(
    model: shared.SharedPointEnergyLeJEPA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, target_ids = batch()
    token, belief, predicted, target = model.training_latents_with_belief(
        input_ids, target_ids
    )
    return token, belief, predicted, target, target_ids


def test_energy_center_is_the_exact_jepa_prediction() -> None:
    model = build_model().eval()
    _token, belief, predicted, _target, _target_ids = latent_inputs(model)
    with torch.no_grad():
        actual = model.energy_logits(predicted, belief)
        codebook = model.semantic_codebook()
        precision = model.bounded_precision(
            model.position_log_sigma(belief)
        ).unsqueeze(-1)
        direct = model.rbf_owner.rbf_token_bias - 0.5 * precision * torch.cdist(
            predicted.float(), codebook.float()
        ).square()
    assert torch.allclose(
        F.softmax(actual, dim=-1),
        F.softmax(direct, dim=-1),
        atol=2e-5,
    )


def test_observed_jepa_target_is_the_matching_live_codebook_entry() -> None:
    model = build_model().eval()
    _token, _belief, _predicted, target, target_ids = latent_inputs(model)
    with torch.no_grad():
        matching_codes = model.semantic_codebook()[target_ids]
    assert torch.allclose(target, matching_codes, atol=1e-6)


def test_full_training_loss_is_nll_plus_point_mse() -> None:
    model = build_model().train()
    input_ids, target_ids = batch()
    total, components = model(input_ids, target_ids)
    with torch.no_grad():
        _token, belief, predicted, target = model.training_latents_with_belief(
            input_ids, target_ids
        )
        expected_policy = F.cross_entropy(
            model.energy_logits(predicted, belief).flatten(0, 1),
            target_ids.flatten(),
        )
        expected_latent = F.mse_loss(predicted.float(), target.float())
    assert components.shape == (3,)
    assert torch.allclose(components[0], expected_policy, atol=1e-6)
    assert torch.allclose(components[1], expected_latent, atol=1e-6)
    assert float(components[2]) == 0.0
    expected_total = expected_policy + model.latent_loss_weight * expected_latent
    assert torch.allclose(total.detach(), expected_total, atol=1e-6)


def test_forward_does_not_use_categorical_expected_distance() -> None:
    model = build_model().train()
    input_ids, target_ids = batch()
    original = model.semantic_distribution_loss
    try:
        model.semantic_distribution_loss = lambda *_args: (_ for _ in ()).throw(
            AssertionError("expected-distance objective must not be called")
        )
        total, _components = model(input_ids, target_ids)
    finally:
        model.semantic_distribution_loss = original
    assert torch.isfinite(total)


def test_token_nll_updates_the_energy_center_codebook_and_uncertainty() -> None:
    model = build_model().train()
    _token, belief, predicted, _target, target_ids = latent_inputs(model)
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(
        model.energy_logits(predicted, belief).flatten(0, 1),
        target_ids.flatten(),
    )
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


def test_point_mse_updates_both_prediction_and_target_geometry() -> None:
    model = build_model().train()
    _token, _belief, predicted, target, _target_ids = latent_inputs(model)
    model.zero_grad(set_to_none=True)
    F.mse_loss(predicted.float(), target.float()).backward()
    named = dict(model.named_parameters())
    for fragment in (
        "tok_emb.weight",
        "latent_projector",
        "prediction_projector",
        "attn",
    ):
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


def test_geometry_gives_semantic_neighbor_partial_credit() -> None:
    target = torch.tensor([0.0, 0.0])       # kitten
    neighbor = torch.tensor([0.1, 0.0])     # cat
    unrelated = torch.tensor([5.0, 0.0])    # fire
    neighbor_error = F.mse_loss(neighbor, target)
    unrelated_error = F.mse_loss(unrelated, target)
    assert neighbor_error < unrelated_error / 1000.0


def test_eval_forward_is_exact_token_nll() -> None:
    model = build_model().eval()
    input_ids, target_ids = batch()
    with torch.no_grad():
        actual = model(input_ids, target_ids)
        expected = F.cross_entropy(
            model.policy_logits(input_ids).flatten(0, 1),
            target_ids.flatten(),
        )
    assert torch.allclose(actual, expected, atol=1e-6)


def test_cached_generation_logits_match_full_forward() -> None:
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


def test_checkpoint_loader_reconstructs_shared_point_architecture() -> None:
    from postraining import model_io

    torch.manual_seed(0)
    with model_io._pope_construction():
        source = shared.SharedPointEnergyLeJEPA(**model_kwargs())
    payload = {
        "model": source.state_dict(),
        "metadata": {
            "architecture": shared.ARCHITECTURE,
            "model": model_kwargs(),
            "optimizer": {"train_seq_len": 2048},
        },
    }
    loaded = model_io.load_model(
        "unused-with-explicit-payload.pt",
        torch.device("cpu"),
        payload=payload,
    )
    assert isinstance(loaded, shared.SharedPointEnergyLeJEPA)
    assert loaded.architecture == shared.ARCHITECTURE
    assert loaded.train_context_tokens == 2048


def test_metadata_describes_one_shared_point_without_barycenter() -> None:
    metadata = shared.SharedPointEnergyLeJEPA.experiment_metadata()
    assert metadata["jepa_prediction"] == "categorical_energy_center"
    assert metadata["geometric_objective"] == "point_to_observed_target_code_mse"
    assert metadata["representation_distribution_link"] == (
        "same_energy_center_and_target_codebook"
    )
    assert metadata["prediction_gradient_from_token_nll"] == "attached"
    assert metadata["codebook_gradient_from_token_nll"] == "attached"
    assert not any("barycenter" in str(value) for value in metadata.values())


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
