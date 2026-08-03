"""CPU unit tests for the BN-projector energy-readout arm.

Run directly: ``python3 energy_readout/test_energy_readout_bnproj.py``
No GPU workload; safe outside mlq.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from pretraining.fresh_lejepa import fresh_lejepa_train as v1
import energy_readout.fresh_lejepa_train_energy_readout_bnproj as bn
from pretraining.fresh_lejepa.fresh_lejepa_train_v2_sigreg_projector import TokenProjector

VOCAB, LAYERS, DIM, HEADS, KV_HEADS = 64, 2, 64, 2, 1
BATCH, SEQ = 2, 16


def build_model() -> bn.EnergyReadoutBNProjLeJEPA:
    torch.manual_seed(0)
    model = bn.EnergyReadoutBNProjLeJEPA(
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


def test_bn_projectors_installed_and_probes_deleted() -> None:
    model = build_model()
    assert type(model.latent_projector) is TokenProjector
    assert type(model.prediction_projector) is TokenProjector
    names = [name for name, _ in model.named_parameters()]
    assert not any("probe" in name for name in names), names
    owner_names = [name for name, _ in model.blocks[-1].named_parameters()]
    assert any("energy_log_scale" in name for name in owner_names)
    assert any("energy_bias" in name for name in owner_names)


def test_optimizer_routing_split() -> None:
    """BN affine params (ndim 1) must land Adam-side; projector linears Muon-side."""
    model = build_model()
    block_named = list(model.blocks.named_parameters())
    matrix = [name for name, p in block_named if p.ndim == 2]
    scalar = [name for name, p in block_named if p.ndim < 2]
    assert not any("energy" in name or ".norm." in name for name in matrix), matrix
    assert sum(".norm." in name for name in scalar) == 4, scalar  # 2 projectors x (w, b)
    assert len(matrix) + len(scalar) == len(block_named)


def test_forward_backward_reaches_bn_and_energy_params() -> None:
    model = build_model()
    model.train()
    input_ids, target_ids = batch()
    total, components = model(input_ids, target_ids)
    assert components.shape == (3,)
    total.backward()
    named = dict(model.named_parameters())
    for name, param in named.items():
        if ".norm." in name or "energy_bias" in name or name == "tok_emb.weight":
            assert param.grad is not None and param.grad.abs().sum() > 0, name


def test_codebook_uses_running_stats_without_updating_them() -> None:
    model = build_model()
    model.train()
    input_ids, target_ids = batch()
    # A training forward moves the running statistics off their init.
    total, _ = model(input_ids, target_ids)
    total.backward()
    norm = model.latent_projector.norm
    running_mean_before = norm.running_mean.detach().clone()
    tracked_before = int(norm.num_batches_tracked)
    with torch.no_grad():
        codebook = model.energy_codebook()
        codebook_again = model.energy_codebook()
        raw = F.rms_norm(model.tok_emb.weight, (model.tok_emb.embedding_dim,))
        expected = model.latent_projector.inference(raw)
    assert torch.allclose(codebook, expected, atol=1e-6)
    assert torch.allclose(codebook, codebook_again, atol=1e-6)
    # The codebook passes must not have touched the running statistics.
    assert torch.allclose(norm.running_mean, running_mean_before)
    assert int(norm.num_batches_tracked) == tracked_before
    # Batch-stat projection of the vocab table must NOT be what the codebook
    # uses (this train-mode call does update stats, so it comes last).
    with torch.no_grad():
        train_mode = model.latent_projector(raw)
    assert not torch.allclose(codebook, train_mode, atol=1e-4)


def test_codebook_stays_attached() -> None:
    model = build_model()
    model.train()
    codebook = model.energy_codebook()
    grad = torch.autograd.grad(codebook.sum(), model.tok_emb.weight)[0]
    assert grad is not None and grad.abs().sum() > 0


def test_eval_mode_codebook_matches_eval_forward() -> None:
    model = build_model()
    model.eval()
    with torch.no_grad():
        raw = F.rms_norm(model.tok_emb.weight, (model.tok_emb.embedding_dim,))
        assert torch.allclose(
            model.energy_codebook(), model.latent_projector(raw), atol=1e-6
        )


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
