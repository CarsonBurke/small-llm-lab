"""CPU unit tests for the energy-readout family.

Run directly: ``python3 energy_readout/test_energy_readout.py``
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

import fresh_lejepa_train as v1
import energy_readout.fresh_lejepa_train_energy_readout as energy

VOCAB, LAYERS, DIM, HEADS, KV_HEADS = 64, 2, 64, 2, 1
BATCH, SEQ = 2, 16


def build_model() -> energy.EnergyReadoutLeJEPA:
    torch.manual_seed(0)
    model = energy.EnergyReadoutLeJEPA(
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


def batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    trajectory = torch.randint(0, VOCAB, (BATCH, SEQ + 1))
    return trajectory[:, :-1], trajectory[:, 1:]


def test_probes_deleted_and_energy_params_registered() -> None:
    model = build_model()
    names = [name for name, _ in model.named_parameters()]
    assert not any("probe" in name for name in names), names
    owner_names = [name for name, _ in model.blocks[-1].named_parameters()]
    assert any("energy_log_scale" in name for name in owner_names)
    assert any("energy_bias" in name for name in owner_names)
    expected = -0.5 * math.log(DIM)
    actual = float(model.blocks[-1].energy_log_scale.detach())
    assert abs(actual - expected) < 1e-6, (actual, expected)


def test_optimizer_routing_split() -> None:
    """Replicate baseline.main's blocks split: energy params must be Adam-side."""
    model = build_model()
    block_named = list(model.blocks.named_parameters())
    matrix = [name for name, p in block_named if p.ndim == 2]
    scalar = [name for name, p in block_named if p.ndim < 2]
    assert not any("energy" in name for name in matrix), matrix
    assert sum("energy" in name for name in scalar) == 2, scalar
    assert len(matrix) + len(scalar) == len(block_named)


def test_forward_contract_and_gradients() -> None:
    model = build_model()
    input_ids, target_ids = batch()

    model.eval()
    with torch.no_grad():
        eval_loss = model(input_ids, target_ids)
    assert eval_loss.ndim == 0
    # Calibrated small init scale: starts near the uniform ln(V) baseline.
    assert abs(float(eval_loss) - math.log(VOCAB)) < 1.0, float(eval_loss)

    model.train()
    total, components = model(input_ids, target_ids)
    assert components.shape == (3,)
    assert float(components[2]) == 0.0  # sigreg deferred
    # With ENERGY_LATENT_MSE_WEIGHT=0 the MSE is diagnostic-only.
    assert torch.isclose(total, components[0]), (total, components)

    total.backward()
    for name in ("tok_emb.weight",):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and grad.abs().sum() > 0, name
    owner = model.blocks[-1]
    assert owner.energy_bias.grad is not None
    assert owner.energy_bias.grad.abs().sum() > 0
    assert owner.energy_log_scale.grad is not None
    projector_grads = [
        p.grad for n, p in model.named_parameters() if "latent_projector" in n
    ]
    assert projector_grads and all(
        g is not None and g.abs().sum() > 0 for g in projector_grads
    )
    prediction_grads = [
        p.grad for n, p in model.named_parameters() if "prediction_projector" in n
    ]
    assert prediction_grads and all(
        g is not None and g.abs().sum() > 0 for g in prediction_grads
    )


def test_distance_head_equals_dot_head_up_to_position_constant() -> None:
    """CE must be identical: the -s/2*||z||^2 term is constant per position."""
    model = build_model()
    model.eval()
    input_ids, target_ids = batch()
    with torch.no_grad():
        token_latent, _belief, predicted, _target = (
            model.training_latents_with_belief(input_ids, target_ids)
        )
        del token_latent
        distance_logits = model.energy_logits(predicted, None)
        codebook = model.energy_codebook()
        scale = torch.exp(model.blocks[-1].energy_log_scale)
        dot_equivalent = (
            scale * (predicted @ codebook.T).float()
            - 0.5 * scale * codebook.float().square().sum(-1)
            + model.blocks[-1].energy_bias
        )
        ce_distance = F.cross_entropy(
            distance_logits.flatten(0, 1), target_ids.flatten()
        )
        ce_dot = F.cross_entropy(
            dot_equivalent.flatten(0, 1), target_ids.flatten()
        )
    assert torch.isclose(ce_distance, ce_dot, atol=1e-5), (ce_distance, ce_dot)


def test_latent_mse_weight_knob_attaches_gradient() -> None:
    original = energy.EnergyReadoutLeJEPA.latent_loss_weight
    try:
        energy.EnergyReadoutLeJEPA.latent_loss_weight = 0.5
        model = build_model()
        model.train()
        input_ids, target_ids = batch()
        total, components = model(input_ids, target_ids)
        expected = components[0] + 0.5 * components[1]
        assert torch.isclose(total, expected, atol=1e-6), (total, expected)
    finally:
        energy.EnergyReadoutLeJEPA.latent_loss_weight = original


def test_bigram_table_variant() -> None:
    original = energy.BIGRAM_TABLE
    try:
        energy.BIGRAM_TABLE = True
        model = build_model()
        owner = model.blocks[-1]
        assert owner.energy_bigram.shape == (VOCAB * VOCAB,)
        assert owner.energy_bigram.ndim == 1  # Adam-side routing
        model.train()
        input_ids, target_ids = batch()
        total, _components = model(input_ids, target_ids)
        total.backward()
        assert owner.energy_bigram.grad is not None
        assert owner.energy_bigram.grad.abs().sum() > 0
        # The readout must refuse to run without current-token ids.
        try:
            model.energy_logits(torch.zeros(1, 1, DIM), None)
        except RuntimeError:
            pass
        else:
            raise AssertionError("bigram readout accepted missing ids")
    finally:
        energy.BIGRAM_TABLE = original


def test_detached_codebook_variant() -> None:
    original = energy.DETACH_CODEBOOK
    try:
        energy.DETACH_CODEBOOK = True
        model = build_model()
        model.train()
        input_ids, target_ids = batch()
        total, _components = model(input_ids, target_ids)
        total.backward()
        # The trunk-input path must still reach the embedding table.
        assert model.tok_emb.weight.grad is not None
        assert model.tok_emb.weight.grad.abs().sum() > 0
    finally:
        energy.DETACH_CODEBOOK = original


def test_scale_clamp_bounds_logits() -> None:
    model = build_model()
    model.eval()
    input_ids, target_ids = batch()
    with torch.no_grad():
        _token, _belief, predicted, _target = model.training_latents_with_belief(
            input_ids, target_ids
        )
        model.blocks[-1].energy_log_scale.fill_(50.0)  # exp(50) >> SCALE_MAX
        clamped_hi = model.energy_logits(predicted, None)
        model.blocks[-1].energy_log_scale.fill_(math.log(energy.SCALE_MAX))
        at_max = model.energy_logits(predicted, None)
        model.blocks[-1].energy_log_scale.fill_(-50.0)  # exp(-50) << SCALE_MIN
        clamped_lo = model.energy_logits(predicted, None)
        model.blocks[-1].energy_log_scale.fill_(math.log(energy.SCALE_MIN))
        at_min = model.energy_logits(predicted, None)
    assert torch.allclose(clamped_hi, at_max), "upper scale clamp not applied"
    assert torch.allclose(clamped_lo, at_min), "lower scale clamp not applied"


def test_scale_clamp_is_straight_through() -> None:
    """Beyond the bound the forward is clamped but the gradient stays alive."""
    model = build_model()
    model.train()
    input_ids, target_ids = batch()
    with torch.no_grad():
        model.blocks[-1].energy_log_scale.fill_(50.0)
    total, _components = model(input_ids, target_ids)
    total.backward()
    grad = model.blocks[-1].energy_log_scale.grad
    assert grad is not None and float(grad.abs()) > 0, grad


def test_dot_head_form_branch() -> None:
    """The ENERGY_HEAD_FORM=dot ablation arm must match s*(z.c) + b exactly."""
    original = energy.HEAD_FORM
    try:
        energy.HEAD_FORM = "dot"
        model = build_model()
        model.eval()
        input_ids, target_ids = batch()
        with torch.no_grad():
            _token, _belief, predicted, _target = (
                model.training_latents_with_belief(input_ids, target_ids)
            )
            logits = model.energy_logits(predicted, None)
            codebook = model.energy_codebook()
            scale = torch.exp(model.blocks[-1].energy_log_scale)
            expected = (
                scale * (predicted @ codebook.T).float()
                + model.blocks[-1].energy_bias
            )
        assert torch.allclose(logits, expected, atol=1e-6)
        model.train()
        total, _components = model(input_ids, target_ids)
        total.backward()
        assert model.tok_emb.weight.grad is not None
        assert model.tok_emb.weight.grad.abs().sum() > 0
    finally:
        energy.HEAD_FORM = original


def test_bias_init_from_counts(tmp_dir: Path | None = None) -> None:
    import json
    import tempfile

    counts = [0] * VOCAB
    counts[3] = 900
    counts[7] = 100
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(counts, fh)
        path = fh.name
    original = energy.BIAS_INIT_COUNTS
    try:
        energy.BIAS_INIT_COUNTS = path
        model = build_model()
        bias = model.blocks[-1].energy_bias.detach()
        total = sum(counts) + VOCAB
        expected_3 = math.log((900 + 1) / total)
        expected_0 = math.log(1 / total)
        assert abs(float(bias[3]) - expected_3) < 1e-5, float(bias[3])
        assert abs(float(bias[0]) - expected_0) < 1e-5, float(bias[0])
        assert float(bias[3]) > float(bias[7]) > float(bias[0])
    finally:
        energy.BIAS_INIT_COUNTS = original
        Path(path).unlink()


def test_values_path_raises() -> None:
    model = build_model()
    try:
        model.values_from_features(torch.zeros(1, 1, DIM))
    except RuntimeError:
        pass
    else:
        raise AssertionError("critic path should be unsupported")


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
