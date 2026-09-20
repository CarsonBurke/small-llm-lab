"""Exact BitFlow transport and full likelihood credit. Run GPUs only through mlq."""

import pytest
import torch

from pretraining.nanogpt_mini.bit_density import binary_nll
from pretraining.nanogpt_mini.nanogpt_mini_bitflow_model import (
    BitFlowCodec,
    BitFlowConfig,
    BitFlowGPT,
    XORCoupling,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
VOCAB_SIZE = 29


def test_identity_codec_adam_update_survives_finite_gradients_above_fp32_square_range():
    from pretraining.nanogpt_mini.nanogpt_mini_native_bits_train import (
        TrainConfig,
        make_optimizers,
    )

    model = BitFlowGPT(
        BitFlowConfig(vocab_size=VOCAB_SIZE, mask_surrogate="identity")
    ).cuda()
    optimizer = make_optimizers(model, TrainConfig(data_path="unused"))[0]
    parameter = model.codec.layers[0].hidden.weight
    before = parameter.detach().clone()
    parameter.grad = torch.full_like(parameter, 1e20)
    optimizer.step()
    assert torch.isfinite(parameter).all().item()
    assert all(
        torch.isfinite(value).all().item()
        for value in optimizer.state[parameter].values()
        if isinstance(value, torch.Tensor)
    )
    # First positive-gradient Adam update: bias corrections cancel, irrespective
    # of finite gradient magnitude. Preserve the actual LR and decoupled decay.
    expected = before * (1 - 0.004 * 0.001) - 0.004
    torch.testing.assert_close(parameter, expected, rtol=1e-10, atol=1e-12)


@pytest.fixture(autouse=True)
def isolated_compile_cache():
    # Cases deliberately vary bit width, gradient mode and singleton dimensions.
    # Keep their graph specializations independent, as separate training jobs are.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.fixture(scope="module", params=["sigmoid", "identity"])
def model(request):
    torch.manual_seed(813)
    result = BitFlowGPT(
        BitFlowConfig(vocab_size=VOCAB_SIZE, mask_surrogate=request.param)
    ).cuda()
    with torch.no_grad():
        # Exercise a genuine learned permutation, not merely identity init.
        for parameter in result.codec.parameters():
            parameter.normal_(std=0.4)
        result.prior.proj.weight.normal_(std=0.03)
    result.compile_components()
    result.compile(dynamic=False)
    return result


@pytest.mark.parametrize("bits", [2, 5, 6])
def test_entire_binary_domain_is_bijective_after_arbitrary_weight_changes(bits):
    torch.manual_seed(814)
    codec = BitFlowCodec(BitFlowConfig(vocab_size=1 << bits, code_bits=bits)).cuda()
    codec.compile_components()
    ids = torch.arange(1 << bits, device="cuda")
    shifts = torch.arange(bits - 1, -1, -1, device="cuda")
    domain = ((ids[:, None] >> shifts) & 1).float().unsqueeze(0)
    with torch.no_grad():
        for layer in codec.layers:
            layer.output.weight.zero_()
            layer.output.bias.zero_()
        codec.layers[0].output.bias[0] = 1
        nontrivial = codec(domain)
        assert bool((nontrivial != domain).any())
        torch.testing.assert_close(codec.inverse(nontrivial), domain, rtol=0, atol=0)
        for scale in (0.3, 1.7):
            for parameter in codec.parameters():
                parameter.normal_(std=scale)
            encoded = codec(domain)
            assert bool(((encoded == 0) | (encoded == 1)).all())
            encoded_ids = (encoded[0].long() << shifts).sum(-1)
            torch.testing.assert_close(encoded_ids.sort().values, ids, rtol=0, atol=0)
            torch.testing.assert_close(codec.inverse(encoded), domain, rtol=0, atol=0)
            torch.testing.assert_close(
                codec(codec.inverse(domain)), domain, rtol=0, atol=0
            )


@pytest.mark.parametrize("surrogate", ["sigmoid", "identity"])
def test_surrogates_and_xor_preserve_signed_encoder_credit(surrogate):
    layer = XORCoupling(
        bits=5, width=64, update_left=False, mask_surrogate=surrogate
    ).cuda()
    with torch.no_grad():
        layer.output.weight.zero_()
        layer.output.bias.copy_(torch.tensor([2.0, -3.0, 4.0], device="cuda"))
    layer.compile(dynamic=True, fullgraph=True)
    bits = torch.tensor(
        [[[1.0, 0.0, 0.0, 1.0, 1.0]]], device="cuda", requires_grad=True
    )
    encoded = layer(bits)
    torch.testing.assert_close(
        encoded,
        torch.tensor([[[1.0, 0.0, 1.0, 1.0, 0.0]]], device="cuda"),
        rtol=0,
        atol=0,
    )
    input_gradient, mask_gradient = torch.autograd.grad(
        encoded.sum(), (bits, layer.output.bias)
    )
    torch.testing.assert_close(
        input_gradient,
        torch.tensor([[[1.0, 1.0, -1.0, 1.0, -1.0]]], device="cuda"),
        rtol=0,
        atol=0,
    )
    expected = torch.tensor([1.0, -1.0, -1.0], device="cuda")
    if surrogate == "sigmoid":
        probability = layer.output.bias.detach().sigmoid()
        expected = expected * probability * (1 - probability)
    # BF16 mask-network backward rounds before accumulating into FP32 parameters.
    torch.testing.assert_close(mask_gradient, expected, rtol=0.005, atol=1e-5)


def test_fixed_control_has_identical_initial_prior_and_an_immutable_identity_codec():
    torch.manual_seed(815)
    learned = BitFlowGPT(BitFlowConfig(vocab_size=VOCAB_SIZE)).cuda()
    torch.manual_seed(815)
    fixed = BitFlowGPT(BitFlowConfig(vocab_size=VOCAB_SIZE, learn_codec=False)).cuda()
    for name, tensor in learned.state_dict().items():
        torch.testing.assert_close(tensor, fixed.state_dict()[name], rtol=0, atol=0)
    learned.compile_components()
    learned.compile(dynamic=False)
    fixed.compile_components()
    fixed.compile(dynamic=False)
    ids = torch.arange(VOCAB_SIZE, device="cuda").unsqueeze(0)
    with torch.no_grad():
        expected = fixed.identity_bits(ids)
        torch.testing.assert_close(
            learned.encode_latents(ids), expected, rtol=0, atol=0
        )
        torch.testing.assert_close(fixed.encode_latents(ids), expected, rtol=0, atol=0)
        learned_loss, _ = learned(ids)
    optimizer = torch.optim.SGD(fixed.parameters(), lr=0.1)
    initial_head = fixed.prior.proj.bias.detach().clone()
    fixed_loss, _ = fixed(ids)
    torch.testing.assert_close(fixed_loss, learned_loss, rtol=0, atol=0)
    fixed_loss.backward()
    assert all(parameter.grad is None for parameter in fixed.codec.parameters())
    optimizer.step()
    with torch.no_grad():
        torch.testing.assert_close(fixed.encode_latents(ids), expected, rtol=0, atol=0)
    assert bool((fixed.prior.proj.bias != initial_head).any())


def test_single_character_likelihood_supplies_full_encoder_target_credit(model):
    # At T=1 there is no encoder-dependent context whatsoever. Nonzero codec
    # gradients must therefore come from the current likelihood target itself.
    ids = torch.tensor([[17]], device="cuda")
    parameters = tuple(model.codec.parameters())
    codes = model.encode_latents(ids)
    target_only = binary_nll(
        model.prior(codes).detach(), codes, model.config.mixture_components
    ).sum()
    expected = torch.autograd.grad(target_only, parameters)
    loss, _ = model(ids)
    actual = torch.autograd.grad(loss, parameters)
    for gradient, reference in zip(actual, expected):
        assert bool(torch.isfinite(gradient).all())
        assert bool((gradient != 0).any())
        torch.testing.assert_close(gradient, reference, rtol=0.02, atol=2e-5)


def test_multicharacter_encoder_credit_includes_both_targets_and_prior_context(model):
    ids = torch.tensor([[2, 7, 17, 28]], device="cuda")
    parameter = model.codec.layers[-1].output.bias
    codes = model.encode_latents(ids)
    prediction = model.prior(codes)
    reference = binary_nll(prediction, codes, model.config.mixture_components).sum()
    (full_gradient,) = torch.autograd.grad(reference, parameter)
    # Compiled backward donates buffers; use a fresh graph for each credit path.
    codes = model.encode_latents(ids)
    prediction = model.prior(codes)
    target_only = binary_nll(
        prediction.detach(), codes, model.config.mixture_components
    ).sum()
    (target_gradient,) = torch.autograd.grad(target_only, parameter)
    loss, _ = model(ids)
    (actual_gradient,) = torch.autograd.grad(loss, parameter)
    torch.testing.assert_close(actual_gradient, full_gradient, rtol=0.02, atol=2e-4)
    assert float((full_gradient - target_gradient).abs().max()) > 1e-4


def test_future_characters_cannot_change_codes_or_their_prior_predictions(model):
    ids = torch.arange(64, device="cuda").reshape(2, 32) % VOCAB_SIZE
    changed = ids.clone()
    changed[:, 16:] = (changed[:, 16:] + 7) % VOCAB_SIZE
    with torch.no_grad():
        before, after = model.encode_latents(ids), model.encode_latents(changed)
        predicted_before, predicted_after = model.prior(before), model.prior(after)
    torch.testing.assert_close(before[:, :16], after[:, :16], rtol=0, atol=0)
    # The prior at t=16 sees code15 but cannot see its own target code16.
    torch.testing.assert_close(
        predicted_before[:, :17], predicted_after[:, :17], rtol=0, atol=0
    )
    assert bool((before[:, 16:] != after[:, 16:]).any())
    assert bool((predicted_before[:, 17:] != predicted_after[:, 17:]).any())


def test_code_space_likelihood_charges_unused_identity_mass_without_auxiliaries(model):
    # Independent singleton sequences all share BOS, so enumerating every code
    # proves the density is normalized over 2**K, not only the valid alphabet.
    ids = torch.arange(1 << model.config.code_bits, device="cuda").unsqueeze(1)
    with torch.no_grad():
        codes = model.encode_latents(ids)
        nats = binary_nll(model.prior(codes), codes, model.config.mixture_components)
        probabilities = (-nats).exp()
        total, stats = model(ids[:VOCAB_SIZE], 0.0)
        weighted_total, _ = model(ids[:VOCAB_SIZE], 1.0)
    torch.testing.assert_close(probabilities.sum(), torch.ones((), device="cuda"))
    assert float(probabilities[:VOCAB_SIZE].sum()) < 1
    torch.testing.assert_close(total, nats[:VOCAB_SIZE].sum())
    torch.testing.assert_close(weighted_total, total, rtol=0, atol=0)
    torch.testing.assert_close(stats["rate_nats"], total, rtol=0, atol=0)
    assert all(not value.requires_grad and value.ndim == 0 for value in stats.values())


def test_full_vocabulary_roundtrips_in_a_residual_free_packet(model):
    from pretraining.nanogpt_mini.native_bits_wire import pack_packet, unpack_packet

    ids = torch.arange(VOCAB_SIZE, device="cuda").unsqueeze(0)
    exported = model.export(ids)
    assert exported["residual"] == [[] for _ in range(VOCAB_SIZE)]
    assert len({tuple(row) for row in exported["latents"]}) == VOCAB_SIZE
    packet = pack_packet(
        **exported,
        checkpoint_sha256="a" * 64,
        source_sha256="b" * 64,
        latent_bits=model.transport_widths[0],
        identity_bits=model.transport_widths[1],
    )
    decoded = unpack_packet(packet)
    assert model.recover(decoded["latents"], decoded["residual"]) == list(
        range(VOCAB_SIZE)
    )


def test_reserved_id_codes_are_rejected_even_after_a_nontrivial_permutation(model):
    reserved = torch.arange(VOCAB_SIZE, 1 << model.config.code_bits, device="cuda")
    with torch.no_grad():
        encoded = (
            model.encode_latents(reserved.unsqueeze(0))[0].to(torch.int64).tolist()
        )
    with pytest.raises(ValueError, match="reserved ID"):
        model.recover(encoded, [[] for _ in encoded])
    with pytest.raises(ValueError, match="outside the checkpoint alphabet"):
        model.export(torch.tensor([[-1, VOCAB_SIZE]], device="cuda"))


def test_empty_streams_and_malformed_transport_are_handled_without_dummy_bits(model):
    empty = model.export(torch.empty((1, 0), dtype=torch.long, device="cuda"))
    assert empty == {"latents": [], "residual": []}
    assert model.recover([], []) == []
    valid_row = [0] * model.config.code_bits
    with pytest.raises(ValueError, match="lengths differ"):
        model.recover([valid_row], [])
    with pytest.raises(ValueError, match="no residual bits"):
        model.recover([valid_row], [[0]])
    with pytest.raises(ValueError, match="binary stream or width"):
        model.recover([valid_row[:-1]], [[]])
    with pytest.raises(ValueError, match="binary stream or width"):
        model.recover([[2] + valid_row[1:]], [[]])
