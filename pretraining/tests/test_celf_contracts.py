"""CPU contracts for CELF: masks, anchoring arithmetic, losses, density integrator.

Compiled reductions are exercised through their eager originals; neural
execution is CUDA-only and covered by test_celf_model_gpu.py through mlq.
"""

import math
from dataclasses import replace

import pytest
import torch

from pretraining.celf.config import LossConfig, ModelConfig
from pretraining.celf.evaluation import integrate_log_density, integrate_sample
from pretraining.celf.model import block_causal_mask, two_stream_mask, unit_power
from pretraining.celf.objective import (
    STAT_NAMES,
    effective_rank,
    loss_tail,
    make_flow_inputs,
    sample_times,
    summarize_statistics,
)
from pretraining.celf.training import (
    TrainConfig,
    active_components,
    schedule_multiplier,
    stage_of,
)


def eager(compiled):
    return compiled._torchdynamo_orig_callable


def test_model_config_rejects_incomplete_blocks_and_degenerate_anchor():
    config = ModelConfig()
    assert config.seq_patches == 640 and config.blocks == 80 and config.block_bytes == 32
    expected = 0.5 * 32 * math.log2(1 + (0.75 / 0.25) ** 2)
    assert math.isclose(config.anchor_capacity_bits_per_patch, expected)
    with pytest.raises(ValueError):
        replace(config, seq_bytes=2560 + 4)
    with pytest.raises(ValueError):
        replace(config, decode_time=0.0)
    with pytest.raises(ValueError):
        replace(config, decode_time=1.0)
    with pytest.raises(ValueError):
        replace(config, prior_rope_dim=65)
    with pytest.raises(ValueError):
        LossConfig(mask_probability=1.0)
    with pytest.raises(ValueError):
        LossConfig(flow_weight=-1.0)
    with pytest.raises(ValueError):
        LossConfig(attachment="target")
    assert LossConfig(attachment="history").attachment == "history"


def test_train_config_budgets_and_cadences():
    model = ModelConfig()
    TrainConfig(name="control").validate(model)
    with pytest.raises(ValueError):
        TrainConfig(name="control", batch_sequences=12).validate(model)
    with pytest.raises(ValueError):
        TrainConfig(name="control", sample_every=30).validate(model)
    with pytest.raises(ValueError):
        TrainConfig(name="control", nelbo_sequences=48).validate(model)
    with pytest.raises(ValueError):
        TrainConfig(name="bad name").validate(model)
    TrainConfig(name="staged", steps=2000, codec_steps=300).validate(model)
    with pytest.raises(ValueError):
        TrainConfig(name="staged", steps=300, codec_steps=300).validate(model)
    with pytest.raises(ValueError):
        TrainConfig(name="staged", steps=2000, codec_steps=310).validate(model)
    with pytest.raises(ValueError):
        TrainConfig(name="staged", steps=2000, codec_steps=20).validate(model)


def test_staging_schedule_is_stage_relative():
    config = TrainConfig(name="staged", steps=2300, codec_steps=300, warmup_steps=40)
    assert stage_of(0, config) == (1, 0, 300)
    assert stage_of(299, config) == (1, 299, 300)
    assert stage_of(300, config) == (2, 0, 2000)
    assert stage_of(2300, config) == (2, 2000, 2000)
    assert active_components(299, config) == ("codec",)
    assert active_components(300, config) == ("codec", "prior")
    # Codec stage: warmup then hold at full rate; no cooldown before the prior joins.
    assert schedule_multiplier(0, config) == pytest.approx(1 / 40)
    assert schedule_multiplier(299, config) == 1.0
    # Joint stage restarts warmup, holds to 30%, and cools to 5% at the last update.
    assert schedule_multiplier(300, config) == pytest.approx(1 / 40)
    assert schedule_multiplier(300 + 600, config) == 1.0
    assert schedule_multiplier(2299, config) == pytest.approx(1 - 0.95 * (1999 / 2000 - 0.3) / 0.7)
    unstaged = TrainConfig(name="control", steps=2000, warmup_steps=40)
    assert stage_of(0, unstaged)[0] == 2
    assert active_components(0, unstaged) == ("codec", "prior")
    assert schedule_multiplier(600, unstaged) == 1.0


def test_block_causal_mask_is_bidirectional_within_and_causal_across():
    mask = block_causal_mask(8, 4, torch.device("cpu"))[0, 0]
    for query in range(8):
        for key in range(8):
            assert mask[query, key].item() == (key // 4 <= query // 4)


def test_two_stream_mask_visibility():
    length, block = 8, 4
    mask = two_stream_mask(length, block, torch.device("cpu"))[0, 0]
    assert mask.shape == (16, 16)
    for query in range(16):
        for key in range(16):
            q_history, k_history = query < length, key < length
            q_block, k_block = (query % length) // block, (key % length) // block
            if q_history:
                expected = k_history and k_block <= q_block
            else:
                expected = (k_history and k_block < q_block) or (
                    not k_history and k_block == q_block
                )
            assert mask[query, key].item() == expected, (query, key)
    # A flow query never sees its own block's history nor any other flow block.
    flow_first_block = mask[length : length + block]
    assert not flow_first_block[:, :block].any()
    assert not flow_first_block[:, length + block :].any()


def test_unit_power_and_effective_rank():
    z = torch.randn(3, 5, 16) * 7.0
    normalized = unit_power(z, 1e-6)
    torch.testing.assert_close(
        normalized.square().mean(-1), torch.ones(3, 5), atol=1e-4, rtol=0
    )
    isotropic = torch.randn(64, 200, 8)
    assert effective_rank(isotropic).item() > 7.5
    direction = torch.randn(8)
    collapsed = torch.randn(64, 200, 1) * direction
    assert effective_rank(collapsed).item() < 1.05


def test_sample_times_and_flow_inputs():
    times = eager(sample_times)(torch.randn(10000), 0.25, 0.0, 1.0)
    assert times.min().item() > 0.25 and times.max().item() < 1.0
    assert abs(times.mean().item() - 0.625) < 0.02
    latents = torch.randn(2, 3, 4)
    noise = torch.randn(2, 3, 4)
    t = torch.tensor([0.3, 0.9])
    noisy, target = eager(make_flow_inputs)(latents, noise, t)
    torch.testing.assert_close(noisy[1], 0.1 * latents[1] + 0.9 * noise[1])
    torch.testing.assert_close(target, noise - latents)


def test_loss_tail_terms_and_statistics():
    batch, length, patch, vocab = 2, 3, 2, 5
    patches = torch.randint(0, vocab, (batch, length, patch))
    masked = torch.tensor([[True, False, True], [False, False, True]])
    logits = torch.randn(2 * batch, length, patch, vocab)
    velocity = torch.randn(batch, length, 4)
    target = torch.randn(batch, length, 4)
    objective, statistics = eager(loss_tail)(
        logits, patches, masked, velocity, target, 0.5, 2.0
    )
    nll = torch.nn.functional.cross_entropy(
        logits.flatten(0, 2), torch.cat((patches, patches)).flatten(), reduction="none"
    ).view(2 * batch, length, patch)
    reconstruction = nll[:batch].mean()
    masked_bytes = masked[..., None].expand(batch, length, patch)
    masked_ce = nll[batch:][masked_bytes].mean()
    flow = (velocity - target).square().mean(-1).mean()
    torch.testing.assert_close(objective, reconstruction + 0.5 * masked_ce + 2.0 * flow)
    values = dict(zip(STAT_NAMES[:9], statistics.tolist(), strict=True))
    assert values["masked_bytes"] == masked_bytes.sum().item()
    assert values["bytes"] == batch * length * patch
    assert values["patches"] == batch * length
    summary = summarize_statistics(list(statistics.tolist()) + [3.0 * batch, batch])
    assert math.isclose(summary["latent_effective_rank"], 3.0)
    assert math.isclose(summary["reconstruction_nats_per_byte"], reconstruction.item(), rel_tol=1e-5)
    assert math.isclose(summary["masked_nats_per_byte"], masked_ce.item(), rel_tol=1e-5)


def _gaussian_linear_flow(endpoint_std: float):
    """Marginal velocity of the linear path between N(0, s^2 I) and N(0, I)."""

    def variance(t: float) -> float:
        return (1 - t) ** 2 * endpoint_std**2 + t**2

    def velocity(history, state, times):
        t = times[0, 0].item()
        coefficient = (t - (1 - t) * endpoint_std**2) / variance(t)
        return coefficient * state

    return velocity, variance


def test_density_integrator_matches_analytic_gaussian_flow():
    torch.manual_seed(0)
    velocity, variance = _gaussian_linear_flow(endpoint_std=0.5)
    start = 0.25
    batch, length, dim = 3, 2, 3
    state = torch.randn(batch, length, dim) * math.sqrt(variance(start))
    # Hutchinson averages probe quadratic forms; a basis scaled by sqrt(P) makes
    # that average the exact trace.
    basis = math.sqrt(length * dim) * torch.eye(length * dim).view(
        length * dim, 1, length, dim
    ).expand(-1, batch, -1, -1)
    result = integrate_log_density(
        velocity, torch.zeros_like(state), state, start_time=start, steps=400, probe_vectors=basis
    )
    expected = torch.distributions.Normal(0.0, math.sqrt(variance(start))).log_prob(state).flatten(1).sum(1)
    torch.testing.assert_close(result["log_prob"], expected, atol=2e-3, rtol=0)
    assert result["nfe"] == 800


def test_sample_integrator_transports_gaussian_marginals():
    torch.manual_seed(0)
    velocity, variance = _gaussian_linear_flow(endpoint_std=0.5)
    initial = torch.randn(4096, 1, 2)
    sampled = integrate_sample(velocity, torch.zeros_like(initial), initial, stop_time=0.25, steps=200)
    assert abs(sampled.square().mean().item() - variance(0.25)) < 0.03
    with pytest.raises(ValueError):
        integrate_sample(velocity, torch.zeros_like(initial), initial, stop_time=1.0, steps=2)
