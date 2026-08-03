"""CPU unit tests for the mixture-of-energies head arm."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from pretraining.fresh_lejepa import fresh_lejepa_train as v1
import energy_readout.fresh_lejepa_train_energy_readout_mixture as arm
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
    mix = build(arm.EnergyReadoutMixtureLeJEPA)
    assert torch.equal(base.tok_emb.weight, mix.tok_emb.weight)


def test_step0_distribution_equals_base_despite_gate_noise() -> None:
    """Identical components make the mixture exactly the base softmax."""
    base = build(nosig.EnergyReadoutPerDimNoSigregLeJEPA)
    mix = build(arm.EnergyReadoutMixtureLeJEPA)
    assert mix.blocks[-1].energy_gate_weight.abs().sum() > 0  # noise present
    base.eval()
    mix.eval()
    input_ids, target_ids = batch()
    with torch.no_grad():
        assert torch.allclose(
            base(input_ids, target_ids), mix(input_ids, target_ids), atol=1e-5
        )


def test_log_probs_normalized() -> None:
    mix = build(arm.EnergyReadoutMixtureLeJEPA)
    mix.eval()
    input_ids, _ = batch()
    with torch.no_grad():
        token_latent = mix.embed_tokens(input_ids)
        belief = mix.temporal_belief_from_token_latent(token_latent)
        predicted = mix.prediction_latent(belief)
        log_probs = mix.mixture_log_probs(predicted, None)
    total = torch.logsumexp(log_probs, dim=-1)
    assert torch.allclose(total, torch.zeros_like(total), atol=1e-5)


def test_symmetry_breaking_gradients() -> None:
    mix = build(arm.EnergyReadoutMixtureLeJEPA)
    mix.train()
    input_ids, target_ids = batch()
    total, components = mix(input_ids, target_ids)
    assert components.shape == (3,) and float(components[2]) == 0.0
    total.backward()
    owner = dict(mix.blocks[-1].named_parameters())
    bias_grad = owner["energy_mix_bias"].grad.view(arm.MIXTURE_COMPONENTS, VOCAB)
    assert bias_grad.abs().sum() > 0
    # The gate noise must make per-component posteriors differ, so the
    # component bias gradients cannot all be identical rows.
    assert not torch.allclose(bias_grad[0], bias_grad[1], atol=1e-9)
    assert owner["energy_mix_logtemp"].grad is not None


def test_new_params_flat() -> None:
    mix = build(arm.EnergyReadoutMixtureLeJEPA)
    owner = dict(mix.blocks[-1].named_parameters())
    for name in (
        "energy_mix_bias",
        "energy_mix_logtemp",
        "energy_gate_weight",
        "energy_gate_bias",
    ):
        assert owner[name].ndim == 1, name


def test_metadata() -> None:
    metadata = arm.EnergyReadoutMixtureLeJEPA.experiment_metadata()
    assert metadata["head_mixture"] == "shared_codebook_temp_bias_gate"
    assert metadata["head_mixture_components"] == arm.MIXTURE_COMPONENTS
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
