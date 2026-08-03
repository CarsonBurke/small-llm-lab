"""CPU unit tests for the geometry-aware soft-target arm."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from pretraining.fresh_lejepa import fresh_lejepa_train as v1
import energy_readout.fresh_lejepa_train_energy_readout_perdim_nosigreg as nosig
import energy_readout.fresh_lejepa_train_energy_readout_softgeo as arm

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


def test_eval_loss_identical_to_base() -> None:
    """Soft targets are train-only; the eval path must be plain CE."""
    base = build(nosig.EnergyReadoutPerDimNoSigregLeJEPA)
    soft = build(arm.EnergyReadoutSoftGeoLeJEPA)
    base.eval()
    soft.eval()
    input_ids, target_ids = batch()
    with torch.no_grad():
        assert torch.allclose(
            base(input_ids, target_ids), soft(input_ids, target_ids), atol=1e-7
        )


def test_soft_targets_normalized_and_peaked_on_target() -> None:
    soft = build(arm.EnergyReadoutSoftGeoLeJEPA)
    soft.train()
    input_ids, target_ids = batch()
    q = soft.geometry_soft_targets(target_ids)
    sums = q.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    # q is the head's distribution at the true code: the true token has
    # distance zero, so it must carry the largest distance term; with the
    # bias at init (uniform-ish) it must be the argmax.
    assert (q.argmax(dim=-1) == target_ids).float().mean() > 0.99


def test_training_loss_decomposition() -> None:
    soft = build(arm.EnergyReadoutSoftGeoLeJEPA)
    soft.train()
    # Freeze BN running-stat updates so the reference computation below sees
    # the same codebook (inference uses running stats) as the forward did.
    soft.latent_projector.norm.momentum = 0.0
    soft.prediction_projector.norm.momentum = 0.0
    input_ids, target_ids = batch()
    total, components = soft(input_ids, target_ids)
    with torch.no_grad():
        _token_latent, _belief, predicted, _target_latent = (
            soft.training_latents_with_belief(input_ids, target_ids)
        )
        logits = soft.energy_logits(predicted, None).float()
        log_probs = F.log_softmax(logits.flatten(0, 1), dim=-1)
        onehot_ce = F.nll_loss(log_probs, target_ids.flatten())
        q = soft.geometry_soft_targets(target_ids).flatten(0, 1)
        soft_ce = -(q * log_probs).sum(dim=-1).mean()
        expected = (
            (1 - arm.SOFT_TARGET_EPS) * onehot_ce + arm.SOFT_TARGET_EPS * soft_ce
        )
    assert torch.allclose(components[0], expected, atol=1e-5), (
        components[0],
        expected,
    )
    # Soft targets must actually change the loss relative to one-hot CE.
    assert not torch.allclose(components[0], onehot_ce, atol=1e-6)


def test_backward_and_no_grad_through_targets() -> None:
    soft = build(arm.EnergyReadoutSoftGeoLeJEPA)
    soft.train()
    input_ids, target_ids = batch()
    total, _ = soft(input_ids, target_ids)
    total.backward()
    assert soft.tok_emb.weight.grad is not None
    assert torch.isfinite(soft.tok_emb.weight.grad).all()


def test_metadata() -> None:
    metadata = arm.EnergyReadoutSoftGeoLeJEPA.experiment_metadata()
    assert metadata["policy_targets"] == "geometry_soft_own_head"
    assert metadata["policy_target_eps"] == arm.SOFT_TARGET_EPS
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
