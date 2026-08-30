from __future__ import annotations

import pytest
import torch

from pretraining.fresh_lejepa.fresh_lejepa_train_v1_probe_shared_rms_pope import FreshLeJEPASharedRMSV1PoPE
from postraining.hl_gauss import HLGaussSupport, anchored_unit_geometry
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


def test_anchored_unit_geometry_puts_bin_centers_exactly_on_zero_and_one():
    num_bins, v_min, v_max = anchored_unit_geometry(101, 4)
    assert num_bins == 101 + 1 + 2 * 4
    support = HLGaussSupport(num_bins, v_min, v_max, sigma_ratio=1.0)
    assert abs(support.bin_width - 1.0 / 101) < 1e-12
    # fp32 linspace rounding leaves ~1e-9 on the anchors; anything far below
    # the 1e-2 bin width is exact for projection purposes.
    assert float(support.centers.abs().min()) < 1e-6
    assert float((support.centers - 1.0).abs().min()) < 1e-6
    # The margin extends beyond both anchors by more than 3 sigma, so a
    # boundary target's label Gaussian is effectively untruncated.
    assert float(support.centers.min()) < -3.0 * support.sigma
    assert float(support.centers.max()) > 1.0 + 3.0 * support.sigma


def test_anchored_unit_geometry_rejects_degenerate_grids():
    with pytest.raises(ValueError, match="interior"):
        anchored_unit_geometry(0, 4)
    with pytest.raises(ValueError, match="margin"):
        anchored_unit_geometry(101, -1)


def test_anchored_support_removes_the_boundary_truncation_bias():
    """Exact-0/exact-1 verifier targets decode exactly; the legacy grid can't.

    On [0, 1]-edge supports a boundary target's Gaussian is cut at the
    support edge and the renormalized label decodes ~0.8 sigma inward. With
    the anchors at bin centers and margin bins behind them, the projection is
    symmetric around the target and the expected-scalar decode is unbiased.
    """
    targets = torch.tensor([0.0, 1.0])
    num_bins, v_min, v_max = anchored_unit_geometry(101, 4)
    anchored = HLGaussSupport(num_bins, v_min, v_max, sigma_ratio=1.0)
    legacy = HLGaussSupport(101, 0.0, 1.0, sigma_ratio=1.0)
    anchored_decode = anchored.to_expected_scalar(
        anchored.project_to_logprobs(targets)
    )
    legacy_decode = legacy.to_expected_scalar(
        legacy.project_to_logprobs(targets)
    )
    assert float((anchored_decode - targets).abs().max()) < 1e-4
    assert float((legacy_decode - targets).abs().min()) > 0.5 * legacy.sigma


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


def test_anchored_critic_decodes_an_interior_prior_without_bias():
    """End-to-end constructor path on the anchored grid (trainer default)."""
    num_bins, v_min, v_max = anchored_unit_geometry(20, 3)
    torch.manual_seed(11)
    with _pope_construction():
        trunk = FreshLeJEPASharedRMSV1PoPE(**KWARGS).eval()
    critic = SeparateCritic(
        trunk,
        num_bins=num_bins,
        sigma_ratio=1.0,
        v_min=v_min,
        v_max=v_max,
        prior_value=0.05,
    ).eval()
    with torch.no_grad():
        values = critic.values(_batch())
    # 0.05 sits on a bin center of the width-1/20 anchored grid, so the
    # projected prior is symmetric and the decode has no truncation bias.
    assert float((values - 0.05).abs().max()) < 1e-3


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
    logits = critic.value_logits(batch)
    targets = torch.full(batch.kind.shape, 0.7)
    loss = critic.support.cross_entropy(logits, targets).mean()
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
