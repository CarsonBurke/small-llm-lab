"""CPU tests for distributional-kernel JEPA pretraining."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

import train_gpt as baseline
from energy_readout import fresh_lejepa_train_distributional_kernel_jepa as kernel
from energy_readout import fresh_lejepa_train_energy_readout as energy
from energy_readout import fresh_lejepa_train_energy_readout_perdim_nosigreg as nosig
from energy_readout.test_energy_readout_perdim_nosigreg import (
    BATCH,
    DIM,
    HEADS,
    KV_HEADS,
    LAYERS,
    SEQ,
    VOCAB,
    batch,
)
from pretraining.fresh_lejepa import fresh_lejepa_train as v1


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


def build_model() -> kernel.DistributionalKernelLeJEPA:
    from postraining import model_io

    torch.manual_seed(0)
    with model_io._pope_construction():
        model = kernel.DistributionalKernelLeJEPA(**model_kwargs())
    model.return_loss_components = True
    return model


def training_outputs(model: kernel.DistributionalKernelLeJEPA):
    input_ids, target_ids = batch()
    token, belief, predicted, target = model.training_latents_with_belief(
        input_ids, target_ids
    )
    logits = model.energy_logits(predicted, None)
    codebook = model.energy_codebook()
    return token, belief, predicted, target, logits, codebook, target_ids


def test_rff_buffer_adds_no_trainable_parameters_or_rng_drift() -> None:
    from postraining import model_io

    torch.manual_seed(0)
    with model_io._pope_construction():
        base = nosig.EnergyReadoutPerDimNoSigregLeJEPA(**model_kwargs())
    child = build_model()
    base_params = dict(base.named_parameters())
    child_params = dict(child.named_parameters())
    assert set(base_params) == set(child_params)
    for name in base_params:
        assert torch.equal(base_params[name], child_params[name]), name
    frequencies = child.blocks[-1].kernel_frequencies_q16
    assert frequencies.dtype == torch.int16
    assert frequencies.shape == (kernel.KERNEL_FEATURES // 2, DIM)


def test_kernel_features_are_unit_norm_and_cast_stable() -> None:
    model = build_model().eval()
    torch.manual_seed(4)
    codes = torch.randn(17, DIM)
    with torch.no_grad():
        features = model.kernel_features(codes)
        cast_model = build_model().bfloat16().eval()
        cast_features = cast_model.kernel_features(codes.bfloat16())
    assert features.shape == (17, kernel.KERNEL_FEATURES)
    assert torch.allclose(
        features.square().sum(dim=-1), torch.ones(17), atol=1e-6
    )
    assert cast_model.blocks[-1].kernel_frequencies_q16.dtype == torch.int16
    assert torch.allclose(features, cast_features, atol=2e-3)


def test_kernel_similarity_respects_code_geometry() -> None:
    model = build_model().eval()
    target = torch.zeros(3, DIM)
    target[0, 0] = 1.0
    target[1, 0] = 1.0
    target[1, 1] = 0.05
    target[2, 0] = -1.0
    with torch.no_grad():
        features = model.kernel_features(target)
        neighbor_similarity = features[0] @ features[1]
        distant_similarity = features[0] @ features[2]
    assert neighbor_similarity > distant_similarity + 0.25


def test_distributional_loss_matches_kernel_mean_definition() -> None:
    model = build_model().train()
    _token, _belief, _predicted, _target, logits, codebook, target_ids = (
        training_outputs(model)
    )
    actual = model.distributional_jepa_loss(logits, codebook, target_ids)
    probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
    features = model.kernel_features(codebook)
    expected = (
        (probabilities @ features - features[target_ids].detach())
        .square()
        .sum(dim=-1)
        .mean()
    )
    assert torch.allclose(actual, expected, atol=1e-6)


def test_semantic_neighbor_gets_less_kernel_error_than_distant_token() -> None:
    model = build_model().eval()
    codes = torch.zeros(VOCAB, DIM)
    codes[:, 0] = -1.0
    codes[0].zero_()
    codes[0, 0] = 1.0          # target: kitten
    codes[1] = codes[0]
    codes[1, 1] = 0.05         # nearby: cat
    codes[2, 0] = -1.0         # distant: fire
    target_ids = torch.zeros(1, 1, dtype=torch.long)
    neighbor_logits = torch.full((1, 1, VOCAB), -100.0)
    neighbor_logits[..., 1] = 0.0
    distant_logits = torch.full((1, 1, VOCAB), -100.0)
    distant_logits[..., 2] = 0.0
    neighbor_loss = model.distributional_jepa_loss(
        neighbor_logits, codes, target_ids
    )
    distant_loss = model.distributional_jepa_loss(
        distant_logits, codes, target_ids
    )
    assert neighbor_loss < distant_loss / 100.0


def test_nonlinear_mean_detects_linear_barycenter_cancellation() -> None:
    model = build_model().eval()
    codes = torch.zeros(VOCAB, DIM)
    codes[0, 0] = 1.0
    codes[1, 0] = 1.0
    codes[1, 1] = 1.0
    codes[2, 0] = 1.0
    codes[2, 1] = -1.0
    probabilities = torch.zeros(1, 1, VOCAB)
    probabilities[..., 1] = 0.5
    probabilities[..., 2] = 0.5
    linear_mean = probabilities @ codes
    assert torch.allclose(linear_mean[0, 0], codes[0])
    logits = probabilities.clamp_min(1e-30).log()
    loss = model.distributional_jepa_loss(
        logits, codes, torch.zeros(1, 1, dtype=torch.long)
    )
    assert loss > 0.01


def test_entire_kernel_feature_dictionary_is_stopped() -> None:
    model = build_model().eval()
    codes = torch.randn(VOCAB, DIM, requires_grad=True)
    logits = torch.full((1, 1, VOCAB), -100.0)
    logits[..., 1] = 0.0
    logits.requires_grad_()
    target_ids = torch.zeros(1, 1, dtype=torch.long)
    loss = model.distributional_jepa_loss(logits, codes, target_ids)
    loss.backward()
    assert codes.grad is None
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_full_training_loss_is_nll_plus_distributional_jepa() -> None:
    model = build_model().train()
    model.latent_projector.norm.momentum = 0.0
    model.prediction_projector.norm.momentum = 0.0
    input_ids, target_ids = batch()
    total, components = model(input_ids, target_ids)
    with torch.no_grad():
        _token, _belief, predicted, target = model.training_latents_with_belief(
            input_ids, target_ids
        )
        logits = model.energy_logits(predicted, None)
        expected_policy = F.cross_entropy(
            logits.flatten(0, 1), target_ids.flatten()
        )
        expected_latent = F.mse_loss(predicted.float(), target.float())
        expected_kernel = model.distributional_jepa_loss(
            logits, model.energy_codebook(), target_ids
        )
    assert components.shape == (4,)
    assert torch.allclose(components[0], expected_policy, atol=1e-6)
    assert torch.allclose(components[1], expected_latent, atol=1e-6)
    assert float(components[2]) == 0.0
    assert torch.allclose(components[3], expected_kernel, atol=1e-6)
    assert torch.allclose(
        total.detach(),
        expected_policy + model.kernel_jepa_loss_weight * expected_kernel,
        atol=1e-6,
    )


def test_kernel_score_updates_distribution_and_code_geometry() -> None:
    model = build_model().train()
    _token, _belief, _predicted, _target, logits, codebook, target_ids = (
        training_outputs(model)
    )
    model.zero_grad(set_to_none=True)
    model.distributional_jepa_loss(logits, codebook, target_ids).backward()
    named = dict(model.named_parameters())
    for fragment in (
        "tok_emb.weight",
        "latent_projector",
        "prediction_projector",
        "attn",
        "energy_log_scale",
        "energy_bias",
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


def test_cached_discrete_generation_matches_full_forward() -> None:
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


def test_optional_bigram_policy_and_cached_generation_match() -> None:
    original = energy.BIGRAM_TABLE
    energy.BIGRAM_TABLE = True
    try:
        model = build_model().eval()
        with torch.no_grad():
            model.blocks[-1].energy_bigram.copy_(
                torch.linspace(-0.2, 0.2, VOCAB * VOCAB)
            )
        input_ids, _target_ids = batch()
        with torch.no_grad():
            expected = model.policy_logits(input_ids)
            caches = model.make_generation_cache(
                BATCH,
                SEQ,
                torch.device("cpu"),
                dtype=model.tok_emb.weight.dtype,
            )
            steps = []
            for position in range(SEQ):
                logits, caches = model.generation_policy_step(
                    input_ids[:, position], caches, position
                )
                steps.append(logits)
            actual = torch.stack(steps, dim=1)
    finally:
        energy.BIGRAM_TABLE = original
    assert torch.allclose(actual, expected, atol=2e-5)


def test_checkpoint_loader_reconstructs_distributional_architecture() -> None:
    from postraining import model_io

    source = build_model().eval()
    payload = {
        "model": source.state_dict(),
        "metadata": {
            "architecture": kernel.ARCHITECTURE,
            "model": model_kwargs(),
            "optimizer": {"train_seq_len": 2048},
        },
    }
    loaded = model_io.load_model(
        "unused-with-explicit-payload.pt",
        torch.device("cpu"),
        payload=payload,
    ).eval()
    input_ids, _target_ids = batch()
    with torch.no_grad():
        assert torch.equal(
            source.blocks[-1].kernel_frequencies_q16,
            loaded.blocks[-1].kernel_frequencies_q16,
        )
        assert torch.allclose(
            source.policy_logits(input_ids),
            loaded.policy_logits(input_ids),
            atol=1e-7,
        )
    assert isinstance(loaded, kernel.DistributionalKernelLeJEPA)
    assert loaded.train_context_tokens == 2048


def test_loader_infers_nondefault_kernel_feature_count_from_state() -> None:
    from postraining import model_io

    torch.manual_seed(0)
    with model_io._pope_construction():
        source = kernel.DistributionalKernelLeJEPA(
            **model_kwargs(), kernel_features=32
        )
    payload = {
        "model": source.state_dict(),
        "metadata": {
            "architecture": kernel.ARCHITECTURE,
            "model": model_kwargs(),
        },
    }
    loaded = model_io.load_model(
        "unused-with-explicit-payload.pt",
        torch.device("cpu"),
        payload=payload,
    )
    assert loaded.blocks[-1].kernel_frequencies_q16.shape == (16, DIM)


def test_accumulation_schema_includes_kernel_loss() -> None:
    captured: list[str] = []
    real_compile = builtins.compile
    original_main = baseline.main

    def compile_spy(source, *args, **kwargs):
        if isinstance(source, str) and "train_components" in source:
            captured.append(source)
        return real_compile(source, *args, **kwargs)

    builtins.compile = compile_spy
    try:
        nosig._install_nosigreg_accumulation(
            default_steps=8,
            extra_components=(("kernel_jepa_loss", "kernel_jepa_loss_weight"),),
        )
    finally:
        builtins.compile = real_compile
        baseline.main = original_main
    assert len(captured) == 1
    source = captured[0]
    assert "train_components = torch.zeros(4" in source
    assert "base_model.kernel_jepa_loss_weight * train_components[3]" in source
    assert 'kernel_jepa_loss:{train_components[3].item():.4f}' in source


def test_main_installs_and_restores_distributional_harness() -> None:
    captured: list[str] = []
    real_compile = builtins.compile
    original_baseline_main = baseline.main
    original_nosig_main = nosig.main
    original_class = nosig.EnergyReadoutPerDimNoSigregLeJEPA
    original_architecture = nosig.ARCHITECTURE
    original_file = nosig.__file__

    def compile_spy(source, *args, **kwargs):
        if isinstance(source, str) and "train_components" in source:
            captured.append(source)
        return real_compile(source, *args, **kwargs)

    def fake_nosig_main() -> None:
        assert nosig.EnergyReadoutPerDimNoSigregLeJEPA is (
            kernel.DistributionalKernelLeJEPA
        )
        assert nosig.ARCHITECTURE == kernel.ARCHITECTURE
        restored = nosig._install_nosigreg_accumulation(default_steps=8)
        baseline.main = restored

    builtins.compile = compile_spy
    nosig.main = fake_nosig_main
    try:
        kernel.main()
    finally:
        builtins.compile = real_compile
        nosig.main = original_nosig_main
        baseline.main = original_baseline_main
    assert len(captured) == 1
    assert "train_components = torch.zeros(4" in captured[0]
    assert nosig.EnergyReadoutPerDimNoSigregLeJEPA is original_class
    assert nosig.ARCHITECTURE == original_architecture
    assert nosig.__file__ == original_file


def test_metadata_records_one_distributional_representation() -> None:
    metadata = kernel.DistributionalKernelLeJEPA.experiment_metadata()
    assert metadata["objective"] == "exact_nll_plus_distributional_kernel_jepa"
    assert metadata["jepa_prediction"] == "categorical_kernel_mean"
    assert metadata["proper_score"] == "log_score_plus_low_rank_kernel_score"
    assert metadata["uncertainty"] == "categorical_entropy_no_position_sigma"
    assert metadata["kernel_features"] == kernel.KERNEL_FEATURES


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
