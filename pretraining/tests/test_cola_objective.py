"""Contracts for staged latent losses, not reconstruction-as-language-rate claims."""

import math
from dataclasses import replace

import pytest
import torch

from pretraining.cola.config import ModelConfig
from pretraining.cola.objective import (
    gaussian_terms,
    loss_tail,
    summarize_statistics,
)
from pretraining.cola.training import TrainConfig


def test_gaussian_reference_direction_and_entropy_gradients():
    mean = torch.tensor([[[0.5, -1.0]]], requires_grad=True)
    logvar = torch.tensor([[[-0.7, 0.4]]], requires_grad=True)
    reference_mean = torch.tensor([[[1.0, 0.25]]])
    reference_logvar = torch.tensor([[[0.2, -0.6]]])
    base, entropy, reference = gaussian_terms(
        mean, logvar, reference_mean, reference_logvar
    )
    current = torch.distributions.Normal(mean, (0.5 * logvar).exp())
    frozen = torch.distributions.Normal(reference_mean, (0.5 * reference_logvar).exp())
    torch.testing.assert_close(
        reference, torch.distributions.kl_divergence(current, frozen).sum()
    )
    torch.testing.assert_close(
        base,
        torch.distributions.kl_divergence(
            current, torch.distributions.Normal(0.0, 1.0)
        ).sum(),
    )
    torch.testing.assert_close(entropy, -current.entropy().sum())
    mean_grad, variance_grad = torch.autograd.grad(
        entropy, (mean, logvar), allow_unused=True
    )
    assert mean_grad is None
    torch.testing.assert_close(variance_grad, torch.full_like(logvar, -0.5))


def test_training_budget_requires_complete_fixed_shape_microbatches():
    model = ModelConfig()
    valid = TrainConfig(name="control", stage="vae")
    valid.validate(model)
    with pytest.raises(ValueError):
        replace(valid, batch_sequences=valid.batch_sequences + 1).validate(model)
    with pytest.raises(ValueError):
        replace(valid, val_sequences=0).validate(model)
    with pytest.raises(ValueError):
        replace(valid, stage="joint").validate(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_joint_loss_uses_entropy_not_gaussian_base_kl():
    # Isolate the uncertain staged objective: moving the mean relative to N(0,I)
    # must not add a hidden stage-two base penalty when the reference follows it.
    device = "cuda"
    ids = torch.zeros((1, 4), device=device, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    logits = torch.zeros((2, 4, 256), device=device, requires_grad=True)
    logvar = torch.zeros((2, 4, 2), device=device)
    zeros = torch.zeros((1, 4, 2), device=device)
    losses = []
    for mean_value in (0.0, 3.0):
        mean = torch.full_like(logvar, mean_value)
        loss, stats = loss_tail(
            logits,
            ids,
            torch.ones(256, device=device, dtype=torch.long),
            mask,
            mean,
            logvar,
            mean[:1],
            logvar[:1],
            zeros,
            zeros,
            True,
            0.1,
            1.0,
            1.0,
            1.0,
            1.0,
        )
        losses.append(loss)
        metrics = summarize_statistics(stats.cpu().tolist())
        assert metrics["reconstruction_bits_per_byte"] == pytest.approx(8.0)
        assert metrics["masked_positions"] == 0
        assert metrics["masked_nats_per_masked_position"] == 0
    torch.testing.assert_close(losses[0], losses[1])
    expected = math.log(256) - 0.1 * (1 + math.log(2 * math.pi))
    torch.testing.assert_close(losses[0], losses[0].new_tensor(expected))
    losses[1].backward()
    assert logits.grad is not None and bool(torch.count_nonzero(logits.grad))
