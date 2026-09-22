from __future__ import annotations

import pytest
import torch

from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_rms_pope import FreshLeJEPASharedRMSV1PoPE
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
    return SeparateCritic(trunk).eval()


def _batch(seed: int = 5, model_kwargs: dict = KWARGS, rows: int = 2):
    torch.manual_seed(seed)
    with _pope_construction():
        backbone = FreshLeJEPASharedRMSV1PoPE(**model_kwargs).eval()
    wrapper = LatentThoughtModel(backbone).eval()
    prompt_ids = torch.randint(0, 32, (rows, 5))
    generator = torch.Generator().manual_seed(9)
    with torch.no_grad():
        return trim_stream(
            rollout_continuations(wrapper, prompt_ids, 4, 16, 1.0, 1.0, generator=generator)
        )


def test_fresh_critic_predicts_exactly_zero_everywhere():
    critic = _critic()
    batch = _batch()
    with torch.no_grad():
        values = critic.values(batch)
    # Zero weight and zero bias: one scalar per stream slot, input-blind.
    assert values.shape == batch.kind.shape
    assert values.dtype == torch.float32
    assert torch.equal(values, torch.zeros_like(values))


def test_scalar_head_fits_the_success_marginal_at_the_production_rate():
    """The failure the categorical head had: pinned at its prior.

    Production width (512) and the default critic rate (2e-5 AdamW, the
    actor rate) on a whole critic: a ~2% Bernoulli marginal is reached
    within the 50-step value warmup, through the head weight on normalized
    beliefs. The HL-Gauss bias moved its off-prior bins by lr per step and
    never got there in 320 steps.
    """
    wide = dict(KWARGS, model_dim=512, num_heads=8, num_kv_heads=4)
    torch.manual_seed(11)
    with _pope_construction():
        trunk = FreshLeJEPASharedRMSV1PoPE(**wide).eval()
    critic = SeparateCritic(trunk).eval()
    batch = _batch(model_kwargs=wide, rows=64)
    generator = torch.Generator().manual_seed(3)
    targets = (torch.rand(batch.kind.shape, generator=generator) < 0.02).float()
    mask = (batch.kind != PAD_SLOT).float()
    marginal = float((targets * mask).sum() / mask.sum())
    optimizer = torch.optim.AdamW(critic.parameters(), lr=2e-5, weight_decay=0.0)
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        loss = ((critic.values(batch) - targets).square() * mask).sum() / mask.sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        mean = float((critic.values(batch) * mask).sum() / mask.sum())
    assert abs(mean - marginal) < 0.25 * marginal, (mean, marginal)


def test_hiddens_reach_the_critic_through_its_own_combiner():
    critic = _critic()
    with torch.no_grad():
        critic.head.weight.normal_(std=0.05)
        # A fresh combiner is an exact identity (zero carry matrix, zero
        # type bias), so liven the content channel before perturbing hiddens.
        critic.combiner.carry.weight.normal_(std=0.05)
    batch = _batch()
    carried_slots = int((batch.kind == THOUGHT_SLOT).sum())
    if carried_slots == 0:
        raise AssertionError("rollout stored no carried hiddens; change the seed")
    with torch.no_grad():
        baseline_values = critic.values(batch)
        batch.thoughts.add_(torch.randn_like(batch.thoughts))
        perturbed_values = critic.values(batch)
    assert not torch.equal(baseline_values, perturbed_values)


def test_critic_routes_exact_stored_raw_actions_at_thought_slots():
    critic = _critic()
    batch = _batch()
    token_latent = critic.trunk.embed_tokens(batch.token_ids)
    pad_scale = (batch.kind != PAD_SLOT)[..., None].to(token_latent.dtype)
    thought_slots = batch.kind == THOUGHT_SLOT
    expected = torch.where(
        thought_slots[..., None],
        batch.thoughts.to(token_latent.dtype),
        token_latent,
    ) * pad_scale
    assert torch.equal(critic.assemble_inputs(batch), expected)


def test_critic_dense_masked_inputs_match_combiner_routing():
    critic = _critic()
    with torch.no_grad():
        critic.combiner.carry.weight.normal_(std=0.02)
        critic.combiner.type_bias.normal_(std=0.02)
        for mlp in critic.combiner.mlps:
            mlp.proj.weight.normal_(std=0.02)
    batch = _batch()
    token_latent = critic.trunk.embed_tokens(batch.token_ids)
    pad_scale = (batch.kind != PAD_SLOT)[..., None].to(token_latent.dtype)
    thought_base = batch.thoughts.to(token_latent.dtype)
    thought_input = critic.combiner(thought_base, batch.thoughts)
    expected = torch.where(
        (batch.kind == THOUGHT_SLOT)[..., None], thought_input, token_latent
    ) * pad_scale
    torch.testing.assert_close(critic.assemble_inputs(batch), expected)


def test_critic_refuses_missing_raw_thought_actions():
    critic = _critic()
    batch = _batch()
    batch.thoughts = batch.thoughts[..., :0]
    with pytest.raises(ValueError, match="stored raw thought actions"):
        critic.assemble_inputs(batch)


def test_critic_token_path_handles_zero_width_pinned_storage():
    critic = _critic()
    batch = _batch()
    batch.thoughts = batch.thoughts[..., :0]
    batch.kind[batch.kind == THOUGHT_SLOT] = PAD_SLOT
    token_latent = critic.trunk.embed_tokens(batch.token_ids)
    pad_scale = (batch.kind != PAD_SLOT)[..., None].to(token_latent.dtype)
    assert torch.equal(critic.assemble_inputs(batch), token_latent * pad_scale)


def test_all_critic_parameters_receive_value_gradients():
    critic = _critic()
    with torch.no_grad():
        critic.head.weight.normal_(std=0.05)
        # Liven the carry so the trunk sees a nonzero carry contribution
        # and its gradient reach is exercised too, not just the combiner's.
        critic.combiner.carry.weight.normal_(std=0.05)
    batch = _batch()
    targets = torch.full(batch.kind.shape, 0.7)
    loss = (critic.values(batch) - targets).square().mean()
    loss.backward()
    named = dict(critic.named_parameters())
    with_grad = {
        name for name, parameter in named.items()
        if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
    }
    # Trunk, combiner, and head must all train; the trunk's unused output
    # heads (probes, lm head paths) legitimately get no gradient.
    assert any(name.startswith("trunk.blocks.0") for name in with_grad)
    assert "combiner.carry.weight" in with_grad
    assert "combiner.type_bias" in with_grad
    # The MLP proj is zero-initialized, so at step 0 the forward weight's
    # gradient (which flows through proj) is exactly zero; proj itself moves
    # first and unlocks fc, mirroring the pretraining identity-block recipe.
    assert "combiner.mlps.0.proj.weight" in with_grad
    assert "combiner.mlps.0.proj.bias" in with_grad
    assert "head.weight" in with_grad and "head.bias" in with_grad
    assert "trunk.tok_emb.weight" in with_grad
