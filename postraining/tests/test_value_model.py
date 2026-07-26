from __future__ import annotations

import pytest
import torch

from fresh_lejepa_train_v1_probe_shared_rms_pope import FreshLeJEPASharedRMSV1PoPE
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


def test_critic_adapter_starts_orthogonal_and_norm_preserving():
    critic = _critic()
    weight = critic.adapter.projection.weight.detach()
    torch.testing.assert_close(
        weight @ weight.T,
        torch.eye(weight.shape[0]),
        rtol=1e-5,
        atol=5e-7,
    )
    assert torch.count_nonzero(critic.adapter.projection.bias) == 0
    thoughts = torch.randn(16, weight.shape[0])
    torch.testing.assert_close(
        critic.adapter(thoughts).norm(dim=-1),
        thoughts.norm(dim=-1),
        rtol=1e-5,
        atol=1e-6,
    )


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


def test_tanh_critic_transforms_raw_thoughts_before_its_adapter():
    critic = _critic()
    critic.thought_action_transform = "tanh"
    batch = _batch()
    thought_mask = batch.kind == THOUGHT_SLOT
    batch.thoughts[thought_mask] = 2.5
    token_latent = critic.trunk.embed_tokens(batch.token_ids)
    expected = token_latent.clone()
    expected[thought_mask] = critic.adapter(
        batch.thoughts[thought_mask].float().tanh()
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
    assert "adapter.projection.weight" in with_grad
    assert "adapter.projection.bias" in with_grad
    assert "head.weight" in with_grad and "head.bias" in with_grad
    assert "trunk.tok_emb.weight" in with_grad
