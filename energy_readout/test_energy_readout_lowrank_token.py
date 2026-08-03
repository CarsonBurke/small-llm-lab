"""CPU unit tests for the low-rank current-token residual arm."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from pretraining.fresh_lejepa import fresh_lejepa_train as v1
import energy_readout.fresh_lejepa_train_energy_readout_lowrank_token as arm
import energy_readout.fresh_lejepa_train_energy_readout_perdim_nosigreg as nosig

VOCAB, LAYERS, DIM, HEADS, KV_HEADS = 64, 2, 64, 2, 1
BATCH, SEQ = 2, 16

MODEL_KWARGS = dict(
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


def build(cls):
    torch.manual_seed(0)
    model = cls(**MODEL_KWARGS)
    model.return_loss_components = True
    return model


def batch():
    torch.manual_seed(1)
    trajectory = torch.randint(0, VOCAB, (BATCH, SEQ + 1))
    return trajectory[:, :-1], trajectory[:, 1:]


def test_trunk_rng_parity_with_base() -> None:
    base = build(nosig.EnergyReadoutPerDimNoSigregLeJEPA)
    lowrank = build(arm.EnergyReadoutLowRankTokenLeJEPA)
    assert torch.equal(base.tok_emb.weight, lowrank.tok_emb.weight)
    assert torch.equal(
        base.latent_projector.input.weight, lowrank.latent_projector.input.weight
    )


def test_step0_logits_equal_base() -> None:
    base = build(nosig.EnergyReadoutPerDimNoSigregLeJEPA)
    lowrank = build(arm.EnergyReadoutLowRankTokenLeJEPA)
    base.eval()
    lowrank.eval()
    input_ids, target_ids = batch()
    with torch.no_grad():
        assert torch.allclose(
            base(input_ids, target_ids), lowrank(input_ids, target_ids), atol=1e-6
        )


def test_new_params_flat_and_adam_routed() -> None:
    lowrank = build(arm.EnergyReadoutLowRankTokenLeJEPA)
    owner = dict(lowrank.blocks[-1].named_parameters())
    factor = owner["energy_token_factor"]
    proj = owner["energy_token_proj"]
    assert factor.ndim == 1 and factor.numel() == VOCAB * arm.TOKEN_RESIDUAL_RANK
    assert proj.ndim == 1 and proj.numel() == arm.TOKEN_RESIDUAL_RANK * DIM
    assert torch.all(proj == 0)


def test_zero_side_receives_gradient() -> None:
    lowrank = build(arm.EnergyReadoutLowRankTokenLeJEPA)
    lowrank.train()
    input_ids, target_ids = batch()
    total, components = lowrank(input_ids, target_ids)
    assert components.shape == (3,) and float(components[2]) == 0.0
    total.backward()
    owner = dict(lowrank.blocks[-1].named_parameters())
    # LoRA-style init: the zero side (proj) must get gradient through the
    # random side; the random side's gradient is exactly zero while proj is 0.
    assert owner["energy_token_proj"].grad.abs().sum() > 0
    assert owner["energy_token_factor"].grad.abs().sum() == 0


def test_residual_changes_logits_when_proj_nonzero() -> None:
    lowrank = build(arm.EnergyReadoutLowRankTokenLeJEPA)
    lowrank.eval()
    input_ids, target_ids = batch()
    with torch.no_grad():
        before = lowrank(input_ids, target_ids)
        lowrank.blocks[-1].energy_token_proj.add_(0.1)
        after = lowrank(input_ids, target_ids)
    assert not torch.allclose(before, after, atol=1e-6)


def test_metadata() -> None:
    metadata = arm.EnergyReadoutLowRankTokenLeJEPA.experiment_metadata()
    assert metadata["token_residual"] == "low_rank_code_projected"
    assert metadata["token_residual_rank"] == arm.TOKEN_RESIDUAL_RANK
    assert metadata["sigreg_batch"] == "removed"


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
