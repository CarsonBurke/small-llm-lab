"""Muon optimizer math and the trunk-optimizer routing in build_optimizers."""

from __future__ import annotations

import torch

from postraining.latent_thought import LatentThoughtModel
from postraining.muon import Muon, zeropower_via_newtonschulz5
from postraining.nano_backbone import NanoGPTBackbone
from postraining.train_latent_vapo import (
    build_optimizers,
    renderer_parameters,
    step_optimizers,
    zero_optimizers,
)
from postraining.value_model import SeparateCritic

KWARGS = dict(vocab_size=64, num_layers=2, model_dim=256)


def _wrapper(seed: int = 3) -> LatentThoughtModel:
    torch.manual_seed(seed)
    wrapper = LatentThoughtModel(NanoGPTBackbone(**KWARGS).float().eval())
    for parameter in wrapper.parameters():
        parameter.requires_grad_(True)
    return wrapper


def _critic(wrapper: LatentThoughtModel, seed: int = 11) -> SeparateCritic:
    torch.manual_seed(seed)
    trunk = NanoGPTBackbone(**KWARGS).float()
    return SeparateCritic(trunk, num_bins=17, sigma_ratio=2.0).eval()


def test_newtonschulz_orthogonalizes_within_bf16_tolerance():
    matrix = torch.randn(64, 48, generator=torch.Generator().manual_seed(0))
    orthogonalized = zeropower_via_newtonschulz5(matrix).float()
    singular_values = torch.linalg.svdvals(orthogonalized)
    # NS5 in bf16 lands near — not exactly on — the orthogonal manifold.
    assert singular_values.max() < 1.35
    assert singular_values.min() > 0.65
    # Transposed input follows the wide-matrix branch and matches exactly.
    transposed = zeropower_via_newtonschulz5(matrix.T).float()
    assert torch.equal(transposed, orthogonalized.T.contiguous())


def test_muon_step_updates_matrices_and_skips_missing_grads():
    torch.manual_seed(1)
    stepped = torch.nn.Parameter(torch.randn(8, 8))
    gradless = torch.nn.Parameter(torch.randn(8, 8))
    stepped.grad = torch.randn(8, 8)
    before_stepped = stepped.detach().clone()
    before_gradless = gradless.detach().clone()
    optimizer = Muon([stepped, gradless], lr=1e-2)
    optimizer.step()
    assert not torch.equal(stepped.detach(), before_stepped)
    assert torch.equal(gradless.detach(), before_gradless)
    assert "momentum" in optimizer.state[stepped]
    assert gradless not in optimizer.state

    # The rectangular scale amplifies tall-matrix updates by sqrt(rows/cols).
    tall = torch.nn.Parameter(torch.zeros(16, 4))
    wide = torch.nn.Parameter(torch.zeros(4, 16))
    tall.grad = torch.ones(16, 4)
    wide.grad = torch.ones(4, 16)
    Muon([tall, wide], lr=1.0).step()
    ratio = tall.detach().norm() / wide.detach().norm()
    assert abs(ratio - 2.0) < 0.05  # sqrt(16 / 4)


def test_muon_rejects_vectors_and_empty_parameter_lists():
    vector = torch.nn.Parameter(torch.zeros(4))
    try:
        Muon([vector], lr=1e-2)
    except ValueError:
        pass
    else:
        raise AssertionError("Muon accepted a 1-D parameter")
    try:
        Muon([], lr=1e-2)
    except ValueError:
        pass
    else:
        raise AssertionError("Muon accepted an empty parameter list")


def test_muon_state_dict_round_trip_restores_momentum():
    parameters = [
        torch.nn.Parameter(torch.randn(6, 6)),
        torch.nn.Parameter(torch.randn(12, 6)),
    ]
    for parameter in parameters:
        parameter.grad = torch.randn_like(parameter)
    optimizer = Muon(parameters, lr=1e-2)
    optimizer.step()
    saved = optimizer.state_dict()

    clones = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    restored = Muon(clones, lr=1e-2)
    restored.load_state_dict(saved)
    for original, clone in zip(parameters, clones, strict=True):
        assert torch.equal(
            restored.state[clone]["momentum"],
            optimizer.state[original]["momentum"],
        )


def test_build_optimizers_muon_layout_partitions_exactly():
    wrapper = _wrapper()
    critic = _critic(wrapper)
    optimizers = build_optimizers(
        wrapper,
        critic,
        learning_rate=1e-3,
        trunk_optimizer="muon",
        muon_learning_rate=2e-3,
        critic_muon_learning_rate=3e-3,
        fused=False,
    )
    assert set(optimizers) == {"actor", "actor_muon", "critic", "critic_muon"}
    assert isinstance(optimizers["actor_muon"], Muon)
    assert isinstance(optimizers["critic_muon"], Muon)
    assert optimizers["actor_muon"].param_groups[0]["lr"] == 2e-3
    assert optimizers["critic_muon"].param_groups[0]["lr"] == 3e-3

    backbone = wrapper.backbone
    expected_actor_muon = {
        id(p) for p in backbone.blocks.parameters() if p.ndim >= 2
    }
    actual_actor_muon = {
        id(p)
        for group in optimizers["actor_muon"].param_groups
        for p in group["params"]
    }
    assert actual_actor_muon == expected_actor_muon
    # 2 layers x (q, k, v, attn-proj, mlp-fc, mlp-proj)
    assert len(actual_actor_muon) == 12

    adamw_actor = {
        id(p)
        for group in optimizers["actor"].param_groups
        for p in group["params"]
    }
    assert not (adamw_actor & actual_actor_muon)
    assert {id(p) for p in renderer_parameters(backbone)} <= adamw_actor
    assert id(backbone.embed.weight) in adamw_actor

    critic_muon = {
        id(p)
        for group in optimizers["critic_muon"].param_groups
        for p in group["params"]
    }
    critic_adamw = {
        id(p)
        for group in optimizers["critic"].param_groups
        for p in group["params"]
    }
    assert critic_muon == {
        id(p) for p in critic.trunk.blocks.parameters() if p.ndim >= 2
    }
    assert not (critic_muon & critic_adamw)
    assert critic_muon | critic_adamw == {id(p) for p in critic.parameters()}
    assert id(critic.head.weight) in critic_adamw


def test_build_optimizers_adamw_layout_is_unchanged():
    wrapper = _wrapper()
    critic = _critic(wrapper)
    optimizers = build_optimizers(
        wrapper, critic, learning_rate=1e-3, fused=False,
    )
    assert set(optimizers) == {"actor", "critic"}
    registered = {
        id(p)
        for optimizer in optimizers.values()
        for group in optimizer.param_groups
        for p in group["params"]
    }
    assert {id(p) for p in critic.parameters()} <= registered
    assert {
        id(p) for p in wrapper.backbone.blocks.parameters() if p.ndim >= 2
    } <= registered


def test_role_helpers_drive_muon_optimizers():
    wrapper = _wrapper()
    critic = _critic(wrapper)
    optimizers = build_optimizers(
        wrapper,
        critic,
        learning_rate=1e-3,
        trunk_optimizer="muon",
        muon_learning_rate=1e-2,
        critic_muon_learning_rate=1e-2,
        fused=False,
    )
    block_weight = next(
        p for p in wrapper.backbone.blocks.parameters() if p.ndim >= 2
    )
    critic_block_weight = next(
        p for p in critic.trunk.blocks.parameters() if p.ndim >= 2
    )
    for parameter in (block_weight, critic_block_weight):
        parameter.grad = torch.randn_like(parameter)
    actor_before = block_weight.detach().clone()
    critic_before = critic_block_weight.detach().clone()

    step_optimizers(optimizers, "actor")
    assert not torch.equal(block_weight.detach(), actor_before)
    assert torch.equal(critic_block_weight.detach(), critic_before)

    step_optimizers(optimizers, "critic")
    assert not torch.equal(critic_block_weight.detach(), critic_before)

    zero_optimizers(optimizers, "actor")
    zero_optimizers(optimizers, "critic")
    assert block_weight.grad is None
    assert critic_block_weight.grad is None
