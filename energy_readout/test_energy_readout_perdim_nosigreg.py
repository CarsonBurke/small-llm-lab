"""CPU unit tests for the sigreg-removed energy-readout arm.

Run directly: ``python3 energy_readout/test_energy_readout_perdim_nosigreg.py``
No GPU workload; safe outside mlq.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import fresh_lejepa_train as v1
import train_gpt as baseline
import energy_readout.fresh_lejepa_train_energy_readout_perdim_nosigreg as nosig
from fresh_lejepa_train_v2_sigreg_projector import TokenProjector

VOCAB, LAYERS, DIM, HEADS, KV_HEADS = 64, 2, 64, 2, 1
BATCH, SEQ = 2, 16


def build_model() -> nosig.EnergyReadoutPerDimNoSigregLeJEPA:
    torch.manual_seed(0)
    model = nosig.EnergyReadoutPerDimNoSigregLeJEPA(
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


def test_inherits_perdim_bnproj_features_with_zero_sigreg_weight() -> None:
    model = build_model()
    assert type(model.latent_projector) is TokenProjector
    assert model.blocks[-1].energy_log_scale.shape == (DIM,)
    names = [name for name, _ in model.named_parameters()]
    assert not any("probe" in name for name in names), names
    assert type(model).sigreg_loss_weight == 0.0
    metadata = type(model).experiment_metadata()
    assert metadata["sigreg_batch"] == "removed"
    assert metadata["energy_scale_form"] == "per_dimension_diagonal"


def test_forward_backward_with_zero_sigreg_component() -> None:
    model = build_model()
    model.train()
    input_ids, target_ids = batch()
    total, components = model(input_ids, target_ids)
    assert components.shape == (3,)
    # The forward defers sigreg out of the loss; with the loop-side call
    # removed the component must be identically zero.
    assert float(components[2]) == 0.0
    total.backward()
    named = dict(model.named_parameters())
    for name in ("tok_emb.weight",):
        assert named[name].grad is not None and named[name].grad.abs().sum() > 0
    owner = dict(model.blocks[-1].named_parameters())
    for name, param in owner.items():
        if "energy" in name:
            assert param.grad is not None, name


def test_install_patched_source_has_no_sigreg_computation() -> None:
    """The loop must keep component bookkeeping but carry no sigreg blocks."""
    import builtins

    original = baseline.main
    captured: list[str] = []
    real_compile = builtins.compile

    def compile_spy(source, *args, **kwargs):
        if isinstance(source, str) and "train_components" in source:
            captured.append(source)
        return real_compile(source, *args, **kwargs)

    builtins.compile = compile_spy
    try:
        nosig._install_nosigreg_accumulation(default_steps=8)
    finally:
        builtins.compile = real_compile
        baseline.main = original
    assert len(captured) == 1, len(captured)
    source = captured[0]
    # No sigreg computation of any lineage may survive.
    assert "pooled_sigreg" not in source
    assert "pending_sigreg" not in source
    assert "paired_sigreg" not in source
    assert "deferred_sigreg" not in source
    # No batch-shape guards from the paired/pooled schemes.
    assert "must be even" not in source
    assert "512" not in source
    # Component bookkeeping and schema-compatible logging remain.
    assert "train_components += loss_components" in source
    assert "base_model.sigreg_loss_weight * train_components[2]" in source
    assert 'sigreg_loss:{train_components[2].item():.4f}' in source
    # Configurable accumulation guard remains.
    assert "GRAD_ACCUM_STEPS" in source


def test_install_patches_and_restores() -> None:
    original = baseline.main
    returned = nosig._install_nosigreg_accumulation(default_steps=8)
    try:
        assert returned is original
        assert baseline.main is not original
        assert callable(baseline.main)
    finally:
        baseline.main = original


def test_install_signature_matches_v4_installer() -> None:
    """pope.main calls the installer by keyword; signatures must match."""
    import inspect as _inspect

    import fresh_lejepa_train_v4 as v4

    v4_params = _inspect.signature(v4._install_configurable_accumulation).parameters
    nosig_params = _inspect.signature(nosig._install_nosigreg_accumulation).parameters
    assert list(v4_params) == list(nosig_params)
    for name in v4_params:
        assert v4_params[name].default == nosig_params[name].default, name


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
