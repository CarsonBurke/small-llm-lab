from __future__ import annotations

import torch

from fresh_lejepa_train_v1_probe_shared_rms_pope import FreshLeJEPASharedRMSV1PoPE
from postraining.latent_rollout import (
    PAD_SLOT,
    THOUGHT_SLOT,
    rollout_continuations,
    trim_stream,
)
from postraining.latent_thought import LatentThoughtModel
from postraining.model_io import _pope_construction
from postraining.value_model import SeparateCritic

KWARGS = dict(
    vocab_size=32, num_layers=3, model_dim=32, num_heads=4, num_kv_heads=2,
    mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.01,
    logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
)


def _critic(seed: int = 11) -> SeparateCritic:
    torch.manual_seed(seed)
    with _pope_construction():
        trunk = FreshLeJEPASharedRMSV1PoPE(**KWARGS).eval()
    return SeparateCritic(trunk, num_bins=17, sigma_ratio=2.0).eval()


def _batch(seed: int = 5):
    torch.manual_seed(seed)
    with _pope_construction():
        backbone = FreshLeJEPASharedRMSV1PoPE(**KWARGS).eval()
    wrapper = LatentThoughtModel(backbone).eval()
    prompt_ids = torch.randint(0, 32, (2, 5))
    generator = torch.Generator().manual_seed(9)
    with torch.no_grad():
        return trim_stream(
            rollout_continuations(wrapper, prompt_ids, 4, 16, 1.0, 1.0, generator=generator)
        )


def test_fresh_critic_decodes_the_projected_prior_everywhere():
    critic = _critic()
    batch = _batch()
    with torch.no_grad():
        values = critic.values(batch)
    assert values.shape == batch.kind.shape
    # v215 head init: zero weights, prior bias -> every position decodes the
    # prior value regardless of input.  The zero prior sits on the support
    # edge and decodes the HL-Gauss truncation bias (~1.6 bins) inward.
    assert float((values - values.flatten()[0]).abs().max()) == 0.0
    assert 0.0 < float(values.flatten()[0]) < 2.0 * critic.support.bin_width


def test_thoughts_reach_the_critic_through_its_own_adapter():
    critic = _critic()
    with torch.no_grad():
        critic.head.weight.normal_(std=0.05)
    batch = _batch()
    thought_slots = int((batch.kind == 1).sum())
    if thought_slots == 0:
        raise AssertionError("rollout produced no thoughts; change the seed")
    with torch.no_grad():
        baseline_values = critic.values(batch)
        batch.thoughts.add_(torch.randn_like(batch.thoughts))
        perturbed_values = critic.values(batch)
    assert not torch.equal(baseline_values, perturbed_values)


def test_critic_dense_masked_inputs_match_compact_routing():
    critic = _critic()
    batch = _batch()
    token_latent = critic.trunk.embed_tokens(batch.token_ids)
    thought_mask = batch.kind == THOUGHT_SLOT
    expected = token_latent.clone()
    expected[thought_mask] = critic.adapter(
        batch.thoughts[thought_mask].float()
    ).to(token_latent.dtype)
    expected *= (batch.kind != PAD_SLOT)[..., None].to(expected.dtype)

    torch.testing.assert_close(critic.assemble_inputs(batch), expected)


def test_all_critic_parameters_receive_value_gradients():
    critic = _critic()
    with torch.no_grad():
        critic.head.weight.normal_(std=0.05)
    batch = _batch()
    logits = critic.value_logits(batch)
    targets = torch.full(batch.kind.shape, 0.7)
    loss = critic.support.cross_entropy(logits, targets).mean()
    loss.backward()
    named = dict(critic.named_parameters())
    with_grad = {
        name for name, parameter in named.items()
        if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
    }
    # Trunk, adapter, and head must all train; the trunk's unused output
    # heads (probes, lm head paths) legitimately get no gradient.
    assert any(name.startswith("trunk.blocks.0") for name in with_grad)
    assert "adapter.correction.weight" in with_grad
    assert "head.weight" in with_grad and "head.bias" in with_grad
    assert "trunk.tok_emb.weight" in with_grad
