"""CPU unit tests for the per-dimension-scale energy-readout arm.

Run directly: ``python3 energy_readout/test_energy_readout_perdim.py``
No GPU workload; safe outside mlq.
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

from pretraining.fresh_lejepa import fresh_lejepa_train as v1
import energy_readout.fresh_lejepa_train_energy_readout as energy
import energy_readout.fresh_lejepa_train_energy_readout_bnproj as bn
import energy_readout.fresh_lejepa_train_energy_readout_perdim as perdim
from pretraining.fresh_lejepa.fresh_lejepa_train_v2_sigreg_projector import TokenProjector

VOCAB, LAYERS, DIM, HEADS, KV_HEADS = 64, 2, 64, 2, 1
BATCH, SEQ = 2, 16


def _build(cls):
    torch.manual_seed(0)
    model = cls(
        vocab_size=VOCAB,
        num_layers=LAYERS,
        model_dim=DIM,
        num_heads=HEADS,
        num_kv_heads=KV_HEADS,
        mlp_mult=v1.FreshHyperparameters.mlp_mult,
        tie_embeddings=True,
        tied_embed_init_std=v1.FreshHyperparameters.tied_embed_init_std,
        logit_softcap=v1.FreshHyperparameters.logit_softcap,
        rope_base=v1.FreshHyperparameters.rope_base,
        qk_gain_init=v1.FreshHyperparameters.qk_gain_init,
    )
    model.return_loss_components = True
    return model


def build_model() -> perdim.EnergyReadoutPerDimScaleLeJEPA:
    return _build(perdim.EnergyReadoutPerDimScaleLeJEPA)


def build_scalar_model() -> bn.EnergyReadoutBNProjLeJEPA:
    return _build(bn.EnergyReadoutBNProjLeJEPA)


def batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    trajectory = torch.randint(0, VOCAB, (BATCH, SEQ + 1))
    return trajectory[:, :-1], trajectory[:, 1:]


def _predicted(model) -> torch.Tensor:
    input_ids, target_ids = batch()
    with torch.no_grad():
        _token, _belief, predicted, _target = model.training_latents_with_belief(
            input_ids, target_ids
        )
    return predicted


def test_shape_registration_and_optimizer_routing() -> None:
    """Per-dim scale is a (model_dim,) vector, Adam-side; BN projectors installed."""
    model = build_model()
    owner = model.blocks[-1]
    assert owner.energy_log_scale.shape == (DIM,), owner.energy_log_scale.shape
    assert owner.energy_log_scale.ndim == 1  # Adam scalar-group routing
    expected = -0.5 * math.log(DIM)
    assert torch.allclose(
        owner.energy_log_scale.detach(),
        torch.full((DIM,), expected, dtype=torch.float32),
        atol=1e-6,
    )
    # Probes deleted, BN projectors installed.
    names = [name for name, _ in model.named_parameters()]
    assert not any("probe" in name for name in names), names
    assert type(model.latent_projector) is TokenProjector
    assert type(model.prediction_projector) is TokenProjector
    # Optimizer split: energy params + BN affines land scalar-side, none in the
    # matrix (Muon) group.
    block_named = list(model.blocks.named_parameters())
    matrix = [name for name, p in block_named if p.ndim == 2]
    scalar = [name for name, p in block_named if p.ndim < 2]
    assert not any("energy" in name or ".norm." in name for name in matrix), matrix
    assert sum("energy" in name for name in scalar) == 2, scalar  # log_scale, bias
    assert sum(".norm." in name for name in scalar) == 4, scalar  # 2 projectors x (w, b)
    assert len(matrix) + len(scalar) == len(block_named)


def test_init_equivalence_with_scalar_variant() -> None:
    """At init all dims share the scalar value, so eval CE matches the scalar arm."""
    perdim_model = build_model()
    scalar_model = build_scalar_model()
    input_ids, target_ids = batch()
    perdim_model.eval()
    scalar_model.eval()
    with torch.no_grad():
        ce_perdim = perdim_model(input_ids, target_ids)
        ce_scalar = scalar_model(input_ids, target_ids)
    assert torch.isclose(ce_perdim, ce_scalar, atol=1e-5), (ce_perdim, ce_scalar)


def test_formula_matches_hand_computed_distance() -> None:
    """Eval logits equal b - 0.5 * Σ_d s_d (ẑ_d - c_kd)^2 on the actual latents."""
    model = build_model()
    model.eval()
    predicted = _predicted(model)
    with torch.no_grad():
        logits = model.energy_logits(predicted, None)
        codebook = model.energy_codebook().float()
        scale = torch.exp(model.blocks[-1].energy_log_scale.float())
        bias = model.blocks[-1].energy_bias.float()
        # Broadcast (B, T, 1, d) - (V, d) -> (B, T, V, d), weight by s, sum d.
        diff = predicted.float().unsqueeze(-2) - codebook
        weighted_sq = (diff.square() * scale).sum(dim=-1)
        expected = bias - 0.5 * weighted_sq
    assert torch.allclose(logits, expected, atol=1e-4), (
        (logits - expected).abs().max()
    )


def test_distance_equals_dot_up_to_position_constant() -> None:
    """The per-dim z_sq term is constant across k, so the two CEs coincide."""
    model = build_model()
    model.eval()
    input_ids, target_ids = batch()
    predicted = _predicted(model)
    with torch.no_grad():
        distance_logits = model.energy_logits(predicted, None)
        codebook = model.energy_codebook()
        scale = torch.exp(model.blocks[-1].energy_log_scale.float())
        scaled_predicted = predicted * scale.to(predicted.dtype)
        dots = (scaled_predicted @ codebook.transpose(0, 1)).float()
        c_sq = (codebook.float().square() * scale).sum(dim=-1)
        dot_equivalent = dots - 0.5 * c_sq + model.blocks[-1].energy_bias
        ce_distance = F.cross_entropy(
            distance_logits.flatten(0, 1), target_ids.flatten()
        )
        ce_dot = F.cross_entropy(
            dot_equivalent.flatten(0, 1), target_ids.flatten()
        )
    assert torch.isclose(ce_distance, ce_dot, atol=1e-5), (ce_distance, ce_dot)


def test_gradient_flow() -> None:
    """Backward reaches the scale vector, tok_emb, BN affines, and the bias."""
    model = build_model()
    model.train()
    input_ids, target_ids = batch()
    total, components = model(input_ids, target_ids)
    assert components.shape == (3,)
    total.backward()
    owner = model.blocks[-1]
    grad = owner.energy_log_scale.grad
    assert grad is not None and grad.shape == (DIM,)
    assert grad.abs().sum() > 0
    assert (grad != 0).any()
    named = dict(model.named_parameters())
    assert named["tok_emb.weight"].grad is not None
    assert named["tok_emb.weight"].grad.abs().sum() > 0
    assert owner.energy_bias.grad is not None and owner.energy_bias.grad.abs().sum() > 0
    for name, param in named.items():
        if ".norm." in name:
            assert param.grad is not None and param.grad.abs().sum() > 0, name


def test_elementwise_straight_through_clamp() -> None:
    """Half the vector past each bound: forward is clamped, both halves get grad."""
    model = build_model()
    owner = model.blocks[-1]
    half = DIM // 2
    with torch.no_grad():
        owner.energy_log_scale[:half].fill_(50.0)  # exp(50) >> SCALE_MAX
        owner.energy_log_scale[half:].fill_(-50.0)  # exp(-50) << SCALE_MIN

    # Forward matches the same head evaluated with the log scale already clamped.
    model.eval()
    predicted = _predicted(model)
    with torch.no_grad():
        clamped_logits = model.energy_logits(predicted, None)
        saved = owner.energy_log_scale.detach().clone()
        owner.energy_log_scale.copy_(
            saved.clamp(math.log(energy.SCALE_MIN), math.log(energy.SCALE_MAX))
        )
        at_bounds = model.energy_logits(predicted, None)
        owner.energy_log_scale.copy_(saved)
    assert torch.allclose(clamped_logits, at_bounds), "elementwise clamp not applied"

    # Backward: straight-through keeps the gradient alive in BOTH halves.
    model.train()
    input_ids, target_ids = batch()
    total, _ = model(input_ids, target_ids)
    total.backward()
    grad = owner.energy_log_scale.grad
    assert grad is not None
    assert grad[:half].abs().sum() > 0, "upper-clamped half lost its gradient"
    assert grad[half:].abs().sum() > 0, "lower-clamped half lost its gradient"


def test_dot_head_form_branch() -> None:
    """ENERGY_HEAD_FORM=dot must match Σ_d s_d ẑ_d c_kd + b_k exactly."""
    original = energy.HEAD_FORM
    try:
        energy.HEAD_FORM = "dot"
        model = build_model()
        model.eval()
        predicted = _predicted(model)
        with torch.no_grad():
            logits = model.energy_logits(predicted, None)
            codebook = model.energy_codebook()
            scale = torch.exp(model.blocks[-1].energy_log_scale.float())
            scaled_predicted = predicted * scale.to(predicted.dtype)
            expected = (
                (scaled_predicted @ codebook.transpose(0, 1)).float()
                + model.blocks[-1].energy_bias
            )
        assert torch.allclose(logits, expected, atol=1e-5)
        model.train()
        input_ids, target_ids = batch()
        total, _ = model(input_ids, target_ids)
        total.backward()
        assert model.tok_emb.weight.grad is not None
        assert model.tok_emb.weight.grad.abs().sum() > 0
    finally:
        energy.HEAD_FORM = original


def test_experiment_metadata() -> None:
    meta = perdim.EnergyReadoutPerDimScaleLeJEPA.experiment_metadata()
    assert meta["energy_scale_form"] == "per_dimension_diagonal"
    assert meta["energy_scale_dims"] == v1.FreshHyperparameters.model_dim
    assert meta["projector_norm"] == "fp32_batchnorm"


def main() -> None:
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
