"""CPU unit tests for the pooled-B512 SIGReg energy-readout arm.

Run directly: ``python3 energy_readout/test_energy_readout_perdim_b512sigreg.py``
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
import energy_readout.fresh_lejepa_train_energy_readout_perdim_b512sigreg as b512
from fresh_lejepa_train_v2_sigreg_projector import TokenProjector

VOCAB, LAYERS, DIM, HEADS, KV_HEADS = 64, 2, 64, 2, 1
BATCH, SEQ, MICROBATCHES = 2, 16, 4


def build_model() -> b512.EnergyReadoutPerDimB512SigregLeJEPA:
    torch.manual_seed(0)
    model = b512.EnergyReadoutPerDimB512SigregLeJEPA(
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


def batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
    torch.manual_seed(1)
    out = []
    for _ in range(MICROBATCHES):
        trajectory = torch.randint(0, VOCAB, (BATCH, SEQ + 1))
        out.append((trajectory[:, :-1], trajectory[:, 1:]))
    return out


def test_inherits_perdim_bnproj_features() -> None:
    model = build_model()
    assert type(model.latent_projector) is TokenProjector
    assert model.blocks[-1].energy_log_scale.shape == (DIM,)
    names = [name for name, _ in model.named_parameters()]
    assert not any("probe" in name for name in names), names
    metadata = type(model).experiment_metadata()
    assert metadata["sigreg_batch"] == "pooled_512_once_per_step"
    assert metadata["energy_scale_form"] == "per_dimension_diagonal"


def test_pooled_sigreg_matches_manual_computation() -> None:
    model = build_model()
    model.train()
    micro = batches()
    # SIGReg draws fresh random projections per call; seed identically so the
    # pooled call and the manual reference see the same directions.
    torch.manual_seed(7)
    pooled = model.pooled_sigreg_loss(micro)
    trajectories = torch.cat(
        [torch.cat((x, y[:, -1:]), dim=1) for x, y in micro], dim=0
    )
    torch.manual_seed(7)
    expected = model.sigreg(model.embed_tokens(trajectories))
    assert pooled.shape == ()
    assert torch.allclose(pooled, expected, atol=1e-6), (pooled, expected)


def test_pooled_sigreg_sees_full_step_batch() -> None:
    model = build_model()
    model.train()
    seen: list[torch.Size] = []
    handle = model.sigreg.register_forward_pre_hook(
        lambda module, inputs: seen.append(inputs[0].shape)
    )
    try:
        model.pooled_sigreg_loss(batches())
    finally:
        handle.remove()
    assert seen == [torch.Size((MICROBATCHES * BATCH, SEQ + 1, DIM))], seen


def test_pooled_sigreg_gradients_reach_embedding_and_projector() -> None:
    model = build_model()
    model.train()
    pooled = model.pooled_sigreg_loss(batches())
    pooled.backward()
    named = dict(model.named_parameters())
    # The projectors are registered under the last block; match by suffix.
    bn_weight = [n for n in named if n.endswith("latent_projector.norm.weight")]
    assert len(bn_weight) == 1, bn_weight
    for name in ("tok_emb.weight", bn_weight[0]):
        grad = named[name].grad
        assert grad is not None and grad.abs().sum() > 0, name
    # The pooled statistic regularizes the encoder side only: no gradient may
    # reach the prediction projector or the energy head.
    for name, param in named.items():
        if "prediction_projector" in name or "energy" in name:
            assert param.grad is None or param.grad.abs().sum() == 0, name


def test_install_pooled_accumulation_patches_and_restores() -> None:
    original = baseline.main
    returned = b512._install_pooled_accumulation(default_steps=8)
    try:
        assert returned is original
        assert baseline.main is not original
        assert callable(baseline.main)
    finally:
        baseline.main = original


def test_install_patched_source_contains_pooled_blocks() -> None:
    """The installer must inject the pooled block into BOTH loops, paired into neither."""
    import builtins

    original = baseline.main
    captured: list[str] = []
    real_compile = builtins.compile

    def compile_spy(source, *args, **kwargs):
        if isinstance(source, str) and "pooled_sigreg" in source:
            captured.append(source)
        return real_compile(source, *args, **kwargs)

    builtins.compile = compile_spy
    try:
        b512._install_pooled_accumulation(default_steps=8)
    finally:
        builtins.compile = real_compile
        baseline.main = original
    assert len(captured) == 1, len(captured)
    source = captured[0]
    # Once in the warmup loop, once in the training loop.
    assert source.count("pending_sigreg_batches = []") == 2
    assert source.count("pending_sigreg_batches.append((x, y))") == 2
    assert (
        source.count("base_model.pooled_sigreg_loss(pending_sigreg_batches)") == 2
    )
    assert (
        source.count(
            "* base_model.sigreg_loss_weight * grad_accum_steps * grad_scale"
            ").backward()"
        )
        == 2
    )
    # Logged contributions recover the per-step statistic after the later
    # division by grad_accum_steps.
    assert (
        "train_loss += grad_accum_steps * base_model.sigreg_loss_weight"
        " * pooled_sigreg.detach()" in source
    )
    assert (
        "train_components[2] += grad_accum_steps * pooled_sigreg.detach()"
        in source
    )
    # No paired-scheme leftovers may survive the fork.
    assert "pending_sigreg_batch = (x, y)" not in source
    assert "deferred_sigreg_loss" not in source
    assert "local_sequences * grad_accum_steps != 512" in source


def test_install_signature_matches_v4_installer() -> None:
    """pope.main calls the installer positionally-by-name; signatures must match."""
    import inspect as _inspect

    import fresh_lejepa_train_v4 as v4

    v4_params = _inspect.signature(v4._install_configurable_accumulation).parameters
    pooled_params = _inspect.signature(b512._install_pooled_accumulation).parameters
    assert list(v4_params) == list(pooled_params)
    for name in v4_params:
        assert v4_params[name].default == pooled_params[name].default, name


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
