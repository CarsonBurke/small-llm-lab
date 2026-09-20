"""Lossless and causal native-bit contracts. GPU execution must use mlq."""

import pytest
import torch

from pretraining.nanogpt_mini.nanogpt_mini_native_bits_model import (
    NativeBitsConfig,
    NativeBitsGPT,
)
from pretraining.nanogpt_mini.nanogpt_mini_native_bits_train import gradients_are_finite

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
VOCAB_SIZE = 172808


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(812)
    result = NativeBitsGPT(NativeBitsConfig(vocab_size=VOCAB_SIZE)).cuda()
    # Nonzero prior output makes a causality test sensitive to actual inputs.
    with torch.no_grad():
        result.prior.proj.weight.normal_(std=0.03)
    result.compile_components()
    result.compile(dynamic=False)
    return result


def batch():
    return torch.randint(0, VOCAB_SIZE, (2, 64), device="cuda")


def test_binary_channel_has_gradients_without_continuous_forward_leakage(model):
    ids = batch()
    codes = model.character_codes(ids)
    assert bool(((codes == 0) | (codes == 1)).all())
    latents = model.encode_latents(ids)
    assert bool(((latents == 0) | (latents == 1)).all())
    loss, _ = model(ids)
    gradients = torch.autograd.grad(
        loss,
        (
            model.table.weight,
            model.compressor.output.weight,
            model.prior.proj.weight,
            model.decoder.output.weight,
            model.identity_decoder.output.weight,
        ),
    )
    for gradient in gradients:
        assert bool(torch.isfinite(gradient).all()) and bool((gradient != 0).any())
    saved = model.table.weight.detach().clone()
    try:
        with torch.no_grad():
            # Compare the same inference graph: compiled BF16 training and
            # inference kernels need not round values identically at zero.
            before = model.encode_latents(ids)
            model.table.weight.mul_(3)
            after = model.encode_latents(ids)
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    finally:
        with torch.no_grad():
            model.table.weight.copy_(saved)


def test_future_characters_cannot_change_encoded_prefix_or_prior(model):
    ids = batch()
    changed = ids.clone()
    changed[:, 32:] = (changed[:, 32:] + 113) % VOCAB_SIZE
    with torch.no_grad():
        before, after = model.encode_latents(ids), model.encode_latents(changed)
        predicted_before, predicted_after = model.prior(before), model.prior(after)
    torch.testing.assert_close(before[:, :32], after[:, :32], rtol=0, atol=0)
    # Prior at position32 may see z31 but never z32.
    torch.testing.assert_close(
        predicted_before[:, :33], predicted_after[:, :33], rtol=0, atol=0
    )
    assert bool((before[:, 32:] != after[:, 32:]).any())
    assert bool((predicted_before[:, 33:] != predicted_after[:, 33:]).any())


def test_residual_decode_is_causal_in_transmitted_latents(model):
    latents = torch.randint(0, 2, (2, 64, 8), device="cuda").float()
    changed = latents.clone()
    changed[:, 32:] = 1 - changed[:, 32:]
    with torch.no_grad():
        before, after = model.residual_logits(latents), model.residual_logits(changed)
    torch.testing.assert_close(before[:, :32], after[:, :32], rtol=0, atol=0)
    assert bool((before[:, 32:] != after[:, 32:]).any())


def test_reported_rate_pays_for_residual_but_not_auxiliary_reconstruction(model):
    ids = batch()
    with torch.no_grad():
        total, stats = model(ids, 0.7)
        without_aux, same_stats = model(ids, 0.0)
    torch.testing.assert_close(
        stats["rate_nats"], stats["latent_nats"] + stats["residual_nats"]
    )
    torch.testing.assert_close(total, stats["rate_nats"] + 0.7 * stats["codec_nats"])
    torch.testing.assert_close(without_aux, same_stats["rate_nats"])
    torch.testing.assert_close(stats["rate_nats"], same_stats["rate_nats"])
    assert float(stats["residual_nats"]) > 0


def test_exact_recovery_survives_total_character_code_collapse(model):
    originals = [0, 1, 255, 256, 65535, 65536, VOCAB_SIZE - 1]
    ids = torch.tensor([originals], device="cuda")
    saved = model.table.weight.detach().clone()
    try:
        with torch.no_grad():
            model.table.weight.zero_()
        exported = model.export(ids)
        recovered = model.recover(exported["latents"], exported["residual"])
        assert recovered == originals
    finally:
        with torch.no_grad():
            model.table.weight.copy_(saved)


def test_packet_roundtrip_uses_only_transmitted_bits_and_same_checkpoint(model):
    from pretraining.nanogpt_mini.native_bits_wire import pack_packet, unpack_packet

    ids = batch()[:1, :37]
    exported = model.export(ids)
    packet = pack_packet(
        **exported,
        checkpoint_sha256="a" * 64,
        source_sha256="b" * 64,
        latent_bits=model.config.latent_bits,
        identity_bits=model.id_bits,
    )
    decoded = unpack_packet(packet)
    recovered = model.recover(decoded["latents"], decoded["residual"])
    assert recovered == ids[0].tolist()


def test_balanced_constant_bus_has_zero_information_metric(model):
    weight = model.compressor.output.weight.detach().clone()
    bias = model.compressor.output.bias.detach().clone()
    try:
        with torch.no_grad():
            model.compressor.output.weight.zero_()
            model.compressor.output.bias.copy_(
                torch.tensor([-2.0, 2.0] * 4, device="cuda")
            )
            _, stats = model(batch())
        assert float(stats["latent_bit_mean"]) == 0.5
        assert float(stats["latent_bit_entropy"]) == 0.0
        assert float(stats["residual_nats"]) > 0
    finally:
        with torch.no_grad():
            model.compressor.output.weight.copy_(weight)
            model.compressor.output.bias.copy_(bias)


def test_prior_cannot_change_its_current_target_through_the_encoder(model):
    ids = torch.arange(128, device="cuda").reshape(2, 64)
    weight = model.identity_decoder.output.weight.detach().clone()
    try:
        with torch.no_grad():
            # Isolate prior credit: a constant residual predictor has no
            # encoder gradient; codec_weight=0 removes the auxiliaries.
            model.identity_decoder.output.weight.zero_()
        loss, _ = model(ids, 0.0)
        (gradient,) = torch.autograd.grad(loss, (model.table.weight,))
        # Each final identity occurs once and has no later prediction that
        # could consume it as context. Earlier identities still train via context.
        assert torch.count_nonzero(gradient[ids[:, -1]]) == 0
        assert torch.count_nonzero(gradient[ids[:, :-1]]) > 0
    finally:
        with torch.no_grad():
            model.identity_decoder.output.weight.copy_(weight)


def test_finite_gradient_guard_accepts_large_values_without_modifying_them():
    gradients = [
        torch.tensor([1e20, -1e20, 0.0], device="cuda"),
        torch.tensor([1.0, -2.0], device="cuda", dtype=torch.bfloat16),
    ]
    originals = [gradient.clone() for gradient in gradients]
    assert gradients_are_finite(gradients).item()
    for gradient, original in zip(gradients, originals):
        torch.testing.assert_close(gradient, original, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_finite_gradient_guard_rejects_actual_invalid_elements(invalid):
    gradients = [
        torch.tensor([1e20, -1e20, 0.0], device="cuda"),
        torch.tensor([1.0, invalid], device="cuda", dtype=torch.bfloat16),
    ]
    assert not gradients_are_finite(gradients).item()
