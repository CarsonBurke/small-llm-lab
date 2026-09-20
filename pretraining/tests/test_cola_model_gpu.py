"""Real compiled CUDA contracts. Execute through mlq; never CPU/eager substitutes.

Full-default-shape backward is opt-in with COLA_RUN_FULL_SHAPE=1 and
COLA_FULL_SHAPE_BATCHES=4,8 (comma-separated prior batch sizes). Its VAE batch is
twice the prior batch, matching joint clean/masked reconstruction. This checks
the model graph, not the trainer's complete objective/optimizer memory budget.
"""

import os
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from pretraining.cola import ColaModel, ModelConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _config():
    return ModelConfig(
        latent_dim=4,
        vae_dim=32,
        vae_layers=2,
        vae_heads=4,
        vae_ffn_dim=128,
        dit_dim=32,
        dit_layers=2,
        dit_heads=2,
        dit_rope_dim=8,
        block_size=4,
        seq_len=16,
    )


def _activate(model):
    # Default zero heads/gates would make independence/equivalence tests vacuous.
    # Emulate learned nonzero paths, without altering the architecture or masks.
    with torch.no_grad():
        model.vae.decoder.head.weight.normal_(std=0.02)
        model.prior.head.weight.normal_(std=0.02)
        model.prior.final_modulation.weight.normal_(std=0.02)
        for block in model.prior.blocks:
            block.modulation.weight.normal_(std=0.02)
            bias = block.modulation.bias.view(6, -1)
            bias[2].fill_(0.5)
            bias[5].fill_(0.5)


@pytest.fixture(scope="module")
def active_model():
    torch.manual_seed(301)
    model = ColaModel(_config()).cuda().eval()
    _activate(model)
    model.compile_components()
    return model


def _latents(model, batch=2):
    config = model.config
    shape = (batch, config.seq_len, config.latent_dim)
    clean = torch.randn(shape, device="cuda", dtype=torch.float32)
    noisy = torch.randn_like(clean)
    # Different times within each block also test per-position conditioning.
    times = torch.rand(shape[:2], device="cuda", dtype=torch.float32)
    return clean, noisy, times


def _assert_live(gradient):
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() > 0


def test_prior_cannot_read_future_clean_or_other_noisy_blocks(active_model):
    torch.manual_seed(302)
    clean, noisy, times = _latents(active_model)
    block = active_model.config.block_size
    with torch.no_grad():
        expected = active_model.prior(clean, noisy, times)
        future_clean = clean.clone()
        future_clean[:, block:] = torch.randn_like(future_clean[:, block:]) * 3
        changed_future = active_model.prior(future_clean, noisy, times)
        torch.testing.assert_close(
            changed_future[:, : 2 * block], expected[:, : 2 * block], rtol=0, atol=0
        )

        other_noisy, other_times = noisy.clone(), times.clone()
        other_noisy[:, :block] *= -4
        other_noisy[:, 2 * block :] *= 3
        other_times[:, :block] = 1 - other_times[:, :block]
        other_times[:, 2 * block :] = 1 - other_times[:, 2 * block :]
        changed_noisy = active_model.prior(clean, other_noisy, other_times)
        torch.testing.assert_close(
            changed_noisy[:, block : 2 * block],
            expected[:, block : 2 * block],
            rtol=0,
            atol=0,
        )

        predecessor = clean.clone()
        predecessor[:, :block] *= -3
        visible_change = active_model.prior(predecessor, noisy, times)
        assert not torch.allclose(
            visible_change[:, block : 2 * block],
            expected[:, block : 2 * block],
            atol=1e-4,
            rtol=1e-4,
        )


def test_full_2l_matches_cached_conditionals_without_cache_mutation(active_model):
    torch.manual_seed(303)
    clean, noisy, times = _latents(active_model)
    block = active_model.config.block_size
    cache = None
    with torch.no_grad():
        full = active_model.prior(clean, noisy, times)
        for start in range(0, active_model.config.seq_len, block):
            saved = (
                None
                if cache is None
                else tuple((k.clone(), v.clone()) for k, v in cache)
            )
            velocity = active_model.prior.block_velocity(
                noisy[:, start : start + block], times[:, start : start + block], cache
            )
            torch.testing.assert_close(
                velocity, full[:, start : start + block], rtol=2e-2, atol=3e-3
            )
            assert velocity.dtype == torch.float32
            next_cache = active_model.prior.append_clean(
                clean[:, start : start + block], cache
            )
            for layer, (key, value) in enumerate(next_cache):
                assert (
                    key.shape
                    == value.shape
                    == (
                        clean.shape[0],
                        active_model.config.dit_heads,
                        start + block,
                        active_model.config.dit_dim // active_model.config.dit_heads,
                    )
                )
                assert key.dtype == value.dtype == torch.bfloat16
                assert not key.requires_grad and not value.requires_grad
                if saved is not None:
                    torch.testing.assert_close(
                        cache[layer][0], saved[layer][0], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        cache[layer][1], saved[layer][1], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        key[:, :, :start], saved[layer][0], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        value[:, :, :start], saved[layer][1], rtol=0, atol=0
                    )
            cache = next_cache


def test_vae_prefix_independence_and_current_latent_visibility(active_model):
    torch.manual_seed(304)
    length, split = active_model.config.seq_len, 7
    ids = torch.randint(0, 256, (2, length), device="cuda")
    changed_ids = ids.clone()
    changed_ids[:, split:] = (changed_ids[:, split:] + 127) % 256
    with torch.no_grad():
        mean, logvar = active_model.vae.encode(ids)
        changed_mean, changed_logvar = active_model.vae.encode(changed_ids)
        prefix_mean, prefix_logvar = active_model.vae.encode(ids[:, :split])
        torch.testing.assert_close(
            changed_mean[:, :split], mean[:, :split], rtol=0, atol=0
        )
        torch.testing.assert_close(
            changed_logvar[:, :split], logvar[:, :split], rtol=0, atol=0
        )
        torch.testing.assert_close(prefix_mean, mean[:, :split], rtol=2e-2, atol=3e-3)
        torch.testing.assert_close(
            prefix_logvar, logvar[:, :split], rtol=2e-2, atol=3e-3
        )
        assert not torch.allclose(changed_mean[:, split], mean[:, split])
        assert mean.dtype == logvar.dtype == torch.float32

        latents = torch.randn_like(mean)
        logits = active_model.vae.decode(latents)
        changed_latents = latents.clone()
        changed_latents[:, split:] *= -4
        changed_logits = active_model.vae.decode(changed_latents)
        prefix_logits = active_model.vae.decode(latents[:, :split])
        torch.testing.assert_close(
            changed_logits[:, :split], logits[:, :split], rtol=0, atol=0
        )
        torch.testing.assert_close(
            prefix_logits, logits[:, :split], rtol=2e-2, atol=3e-3
        )
        assert not torch.allclose(
            changed_logits[:, split], logits[:, split], atol=1e-4, rtol=1e-4
        )
        assert logits.dtype == torch.float32


def test_posterior_and_reconstruction_become_live_after_zero_head_update():
    torch.manual_seed(305)
    model = ColaModel(_config()).cuda().train()
    # Reference snapshot is a separate callable/weight owner, not a wrapper
    # accidentally bound to the subsequently changing training encoder.
    reference = deepcopy(model.vae.encoder).requires_grad_(False).eval()
    reference.compile_component()
    model.compile_components()
    ids = torch.randint(0, 256, (2, model.config.seq_len), device="cuda")
    with torch.no_grad():
        reference_mean, _ = reference(ids)
    mean, logvar = model.vae.encode(ids)
    torch.testing.assert_close(mean.detach(), reference_mean, rtol=0, atol=0)
    z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
    logits = model.vae.decode(z)
    torch.testing.assert_close(logits, torch.zeros_like(logits), rtol=0, atol=0)
    F.cross_entropy(logits.flatten(0, 1), ids.flatten()).backward()
    _assert_live(model.vae.decoder.head.weight.grad)
    with torch.no_grad():
        model.vae.decoder.head.weight.add_(
            model.vae.decoder.head.weight.grad, alpha=-0.1
        )
    model.zero_grad(set_to_none=True)

    mean, logvar = model.vae.encode(ids)
    mean.retain_grad()
    logvar.retain_grad()
    logits = model.vae.decode(mean + torch.exp(0.5 * logvar) * torch.randn_like(mean))
    F.cross_entropy(logits.flatten(0, 1), ids.flatten()).backward()
    _assert_live(mean.grad)
    _assert_live(logvar.grad)
    posterior_gradient = model.vae.encoder.posterior.weight.grad
    _assert_live(posterior_gradient[: model.config.latent_dim])
    _assert_live(posterior_gradient[model.config.latent_dim :])
    _assert_live(model.vae.encoder.embedding.weight.grad)
    _assert_live(model.vae.decoder.input_projection.weight.grad)
    assert all(
        p.dtype == torch.float32 and (p.grad is None or torch.isfinite(p.grad).all())
        for p in model.parameters()
    )
    with torch.no_grad():
        model.vae.encoder.posterior.weight.add_(posterior_gradient, alpha=-1)
        updated_mean, _ = model.vae.encode(ids)
        reference_after, _ = reference(ids)
    torch.testing.assert_close(reference_after, reference_mean, rtol=0, atol=0)
    assert not torch.equal(updated_mean, reference_after)


def test_prior_noisy_input_backward_with_frozen_parameters(active_model):
    torch.manual_seed(306)
    clean, noisy, times = _latents(active_model)
    clean.requires_grad_(True)
    noisy.requires_grad_(True)
    # The likelihood evaluator freezes parameters, but must retain input VJPs.
    active_model.requires_grad_(False)
    try:
        velocity = active_model.prior(clean, noisy, times)
        noisy_gradient, clean_gradient = torch.autograd.grad(
            (velocity * torch.randn_like(velocity)).sum(),
            (noisy, clean),
            allow_unused=True,
        )
        _assert_live(noisy_gradient)
        assert clean_gradient is None
        block = active_model.config.block_size
        cache = active_model.prior.append_clean(clean[:, :block])
        current = noisy[:, block : 2 * block].detach().requires_grad_(True)
        conditional = active_model.prior.block_velocity(
            current, times[:, block : 2 * block], cache
        )
        (conditional_gradient,) = torch.autograd.grad(
            conditional.square().sum(), current
        )
        _assert_live(conditional_gradient)
    finally:
        active_model.requires_grad_(True)


@pytest.mark.skipif(
    os.environ.get("COLA_RUN_FULL_SHAPE") != "1",
    reason="opt-in full default shape; queue via mlq",
)
@pytest.mark.parametrize(
    "batch_size",
    [int(value) for value in os.environ.get("COLA_FULL_SHAPE_BATCHES", "8").split(",")],
)
def test_default_shape_compiled_joint_backward(batch_size):
    torch.manual_seed(307)
    config = ModelConfig()
    model = ColaModel(config).cuda().train()
    model.compile_components()
    ids = torch.randint(0, 256, (batch_size, config.seq_len), device="cuda")
    masked = ids.masked_fill(
        torch.rand(ids.shape, device="cuda") < 0.3, config.mask_token_id
    )
    mean, logvar = model.vae.encode(torch.cat((ids, masked), dim=0))
    sampled = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
    logits = model.vae.decode(sampled)
    clean = sampled[:batch_size]
    noise = torch.randn_like(clean)
    times = torch.rand(clean.shape[:2], device="cuda")
    noisy = (1 - times[..., None]) * clean + times[..., None] * noise
    velocity = model.prior(clean, noisy, times)
    reconstruction = F.cross_entropy(logits.flatten(0, 1), ids.repeat(2, 1).flatten())
    posterior_kl = 0.5 * (mean.square() + logvar.exp() - 1 - logvar).mean()
    loss = (
        reconstruction
        + 0.01 * posterior_kl
        + F.mse_loss(velocity, noise - clean.detach())
    )
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss)
    _assert_live(model.vae.encoder.posterior.weight.grad)
    _assert_live(model.vae.decoder.head.weight.grad)
    _assert_live(model.prior.head.weight.grad)
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
