from __future__ import annotations

import pytest
import torch
from torch import nn

from pretraining.latent_moe import LatentMoEConfig, StableLatentMoE
from pretraining.latent_moe_training import (
    apply_accumulated_quantile_balance,
    enable_quantile_balance_collection,
    reset_quantile_balance_accumulators,
)


class _Block(nn.Module):
    def __init__(self, dim: int, config: LatentMoEConfig | None):
        super().__init__()
        self.attn = nn.Linear(dim, dim, bias=False)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.use_mlp = True
        self.use_moe = config is not None
        self.mlp = (
            StableLatentMoE(config, implementation="reference")
            if config is not None
            else nn.Linear(dim, dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        config = LatentMoEConfig(
            model_dim=8,
            latent_dim=4,
            routed_hidden_dim=8,
            num_routed_experts=4,
            experts_per_token=2,
            shared_hidden_dim=4,
            num_shared_experts=2,
        )
        self.embed = nn.Embedding(32, 8)
        self.norm1 = nn.RMSNorm(8)
        self.blocks = nn.Sequential(
            _Block(8, config), _Block(8, None), _Block(8, config)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.norm1(self.embed(inputs)))


def test_full_step_quantile_balance_uses_every_microbatch() -> None:
    torch.manual_seed(47)
    model = _Model().train()
    assert enable_quantile_balance_collection(model, num_bins=128) == 2
    inputs = torch.randint(0, 32, (8, 12))
    old_biases = [
        block.mlp.correction_bias.clone()
        for block in model.blocks
        if block.use_moe
    ]

    reset_quantile_balance_accumulators(model)
    for microbatch in inputs.chunk(4):
        model(microbatch)
    for block in model.blocks:
        if block.use_moe:
            assert block.mlp.get_accumulated_qb_histogram().token_count == 96
    load_cv, max_load = apply_accumulated_quantile_balance(model)

    new_biases = [
        block.mlp.correction_bias
        for block in model.blocks
        if block.use_moe
    ]
    assert all(torch.isfinite(bias).all() for bias in new_biases)
    assert all(float(bias.mean().abs()) < 1e-6 for bias in new_biases)
    assert any(not torch.equal(old, new) for old, new in zip(old_biases, new_biases))
    assert torch.isfinite(load_cv)
    assert 0.25 <= float(max_load) <= 1.0


def test_quantile_balance_helpers_reject_non_moe_model() -> None:
    model = nn.Sequential(nn.Linear(8, 8))
    with pytest.raises(ValueError, match="no LatentMoE"):
        enable_quantile_balance_collection(model)
    with pytest.raises(ValueError, match="no LatentMoE"):
        reset_quantile_balance_accumulators(model)
    with pytest.raises(ValueError, match="no LatentMoE"):
        apply_accumulated_quantile_balance(model)
