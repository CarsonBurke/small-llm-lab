"""Prefix-factorized binary density contracts; run CUDA only through mlq."""

import pytest
import torch
import torch.nn.functional as F

from pretraining.nanogpt_mini.bit_density import PrefixBinaryHead, binary_nll
from pretraining.nanogpt_mini.nanogpt_mini_bitflow_model import (
    BitFlowConfig,
    BitFlowGPT,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
VOCAB_SIZE = 29
BITS = 5


@pytest.fixture(autouse=True)
def isolated_compile_cache():
    # Do not share graph specializations with the existing flow GPU contracts.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def all_codes(bits):
    shifts = torch.arange(bits - 1, -1, -1, device="cuda")
    return ((torch.arange(1 << bits, device="cuda")[:, None] >> shifts) & 1).float()


@pytest.fixture
def head():
    torch.manual_seed(921)
    result = PrefixBinaryHead(32, BITS, 32).cuda()
    with torch.no_grad():
        for parameter in result.parameters():
            parameter.normal_(std=0.12)
        # Keep live ReLU paths, and explicitly defeat zero-output initialization.
        result.position.weight.uniform_(0.5, 0.75)
        result.output.weight.normal_(std=0.15)
    result.compile(dynamic=False, fullgraph=True)
    return result


@pytest.fixture
def density():
    return torch.compile(binary_nll, dynamic=False, fullgraph=True)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(922)
    result = BitFlowGPT(
        BitFlowConfig(
            vocab_size=VOCAB_SIZE,
            code_bits=BITS,
            num_layers=6,
            model_dim=512,
            density_head="prefix",
            prefix_width=128,
            mixture_components=1,
            learn_codec=False,
        )
    ).cuda()
    with torch.no_grad():
        result.prior.proj.output.weight.normal_(std=0.04)
        # Zero residual projections would hide an attention-mask regression.
        for block in result.prior.blocks:
            block.attn.proj.weight.normal_(std=0.01)
            block.mlp.proj.weight.normal_(std=0.01)
    result.compile_components()
    result.compile(dynamic=False)
    return result


def test_full_domain_normalization_charges_reserved_addresses(model, density):
    # Every singleton has identical BOS context, but its own within-code prefix.
    ids = torch.arange(1 << BITS, device="cuda").unsqueeze(1)
    with torch.no_grad():
        codes = model.encode_latents(ids)
        logits = model.prior(codes)
        nats = density(logits, codes, 1)
        probability = (-nats).exp()
        loss, _ = model(ids[:VOCAB_SIZE])
    torch.testing.assert_close(
        probability.sum(), torch.ones((), device="cuda"), rtol=2e-5, atol=2e-6
    )
    valid_mass = probability[:VOCAB_SIZE].sum()
    reserved_mass = probability[VOCAB_SIZE:].sum()
    assert 0 < float(valid_mass) < 1
    assert float(reserved_mass) > 0
    torch.testing.assert_close(valid_mass, 1 - reserved_mass, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(loss, nats[:VOCAB_SIZE].sum())
    # A constant zero head would normalize while exercising no prefix behavior.
    assert float(logits.std()) > 1e-3


def test_each_bit_sees_only_its_strict_prefix(head):
    codes = all_codes(BITS)
    context = torch.randn(1, 32, device="cuda", dtype=torch.bfloat16).expand(
        len(codes), -1
    )
    with torch.no_grad():
        logits = head(context, codes)
    assert logits.dtype == torch.bfloat16
    for bit in range(BITS):
        # MSB-first enumeration groups together every suffix of a given prefix.
        grouped = logits[:, bit].reshape(1 << bit, 1 << (BITS - bit))
        torch.testing.assert_close(
            grouped, grouped[:, :1].expand_as(grouped), rtol=0, atol=0
        )
        if bit:
            flipped_previous = torch.arange(len(codes), device="cuda") ^ (
                1 << (BITS - bit)
            )
            assert bool((logits[:, bit] != logits[flipped_previous, bit]).any())


def test_teacher_forcing_matches_sequential_prefix_only_traversal(head, density):
    codes = all_codes(BITS)
    context = torch.randn(len(codes), 32, device="cuda", dtype=torch.bfloat16)
    # Unknown bits intentionally disagree with the eventual target. They must
    # not affect the next probability before that bit is revealed to the head.
    revealed = 1 - codes
    sequential_nats = torch.zeros(len(codes), device="cuda")
    with torch.no_grad():
        teacher_nats = density(head(context, codes).float(), codes, 1)
        for bit in range(BITS):
            logit = head(context, revealed)[:, bit].float()
            sequential_nats -= torch.where(
                codes[:, bit].bool(), F.logsigmoid(logit), F.logsigmoid(-logit)
            )
            revealed[:, bit] = codes[:, bit]
    torch.testing.assert_close(teacher_nats, sequential_nats, rtol=1e-6, atol=1e-6)


def test_prefix_gradients_exclude_current_and_future_targets(head, density):
    context = torch.randn(
        2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    codes = all_codes(BITS)[[7, 24]].detach().requires_grad_()
    bit = 3
    logits = head(context, codes)
    context_grad, target_grad, prefix_grad = torch.autograd.grad(
        logits[:, bit].float().sum(), (context, codes, head.prefix.weight)
    )
    for gradient in (context_grad, target_grad, prefix_grad):
        assert bool(torch.isfinite(gradient).all())
    assert bool((context_grad != 0).any())
    assert bool((target_grad[:, :bit] != 0).all())
    torch.testing.assert_close(
        target_grad[:, bit:], torch.zeros_like(target_grad[:, bit:]), rtol=0, atol=0
    )
    assert bool((prefix_grad[:, :bit] != 0).any())
    torch.testing.assert_close(
        prefix_grad[:, bit:], torch.zeros_like(prefix_grad[:, bit:]), rtol=0, atol=0
    )

    # A fresh graph avoids compiled backward buffer donation / retain_graph.
    logits = head(context, codes).float()
    expected_last_target_grad = -logits[:, -1].detach().clone()
    (target_grad,) = torch.autograd.grad(density(logits, codes, 1).sum(), codes)
    assert bool(torch.isfinite(target_grad).all())
    assert bool((expected_last_target_grad != 0).all())
    # The final bit has direct likelihood credit but no later conditional path.
    torch.testing.assert_close(
        target_grad[:, -1], expected_last_target_grad, rtol=1e-5, atol=1e-6
    )


def test_full_model_cannot_read_future_characters(model):
    ids = torch.arange(64, device="cuda").reshape(2, 32) % VOCAB_SIZE
    changed = ids.clone()
    changed[:, 16:] = (changed[:, 16:] + 7) % VOCAB_SIZE
    with torch.no_grad():
        before = model.encode_latents(ids)
        after = model.encode_latents(changed)
        before_logits = model.prior(before)
        after_logits = model.prior(after)
    torch.testing.assert_close(before[:, :16], after[:, :16], rtol=0, atol=0)
    torch.testing.assert_close(
        before_logits[:, :16], after_logits[:, :16], rtol=0, atol=0
    )
    # At the first changed character only bit zero has an empty own-code prefix.
    torch.testing.assert_close(
        before_logits[:, 16, 0], after_logits[:, 16, 0], rtol=0, atol=0
    )
    assert bool((before[:, 16:] != after[:, 16:]).any())
    assert bool((before_logits[:, 17:] != after_logits[:, 17:]).any())


def test_full_model_backward_reaches_trunk_and_prefix_but_not_fixed_codec(model):
    model.zero_grad(set_to_none=True)
    ids = torch.arange(64, device="cuda").reshape(2, 32) % VOCAB_SIZE
    loss, _ = model(ids)
    assert bool(torch.isfinite(loss))
    loss.backward()
    for parameter in model.prior.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
    credit_paths = [
        model.prior.input.weight,
        model.prior.proj.prefix.weight,
        model.prior.proj.position.weight,
        model.prior.proj.output.weight,
    ]
    credit_paths.extend(block.attn.q.weight for block in model.prior.blocks)
    credit_paths.extend(block.mlp.fc.weight for block in model.prior.blocks)
    for parameter in credit_paths:
        assert bool((parameter.grad != 0).any())
    assert all(parameter.grad is None for parameter in model.codec.parameters())
    model.zero_grad(set_to_none=True)


def test_same_seed_mixture_control_keeps_trunk_and_fixed_addresses():
    def make_model(density_head, components):
        torch.manual_seed(923)
        result = BitFlowGPT(
            BitFlowConfig(
                vocab_size=VOCAB_SIZE,
                code_bits=BITS,
                num_layers=6,
                model_dim=512,
                density_head=density_head,
                prefix_width=128,
                mixture_components=components,
                learn_codec=False,
            )
        ).cuda()
        result.compile_components()
        result.compile(dynamic=False)
        return result

    mixture = make_model("mixture", 8)
    prefix = make_model("prefix", 1)
    control = mixture.state_dict()
    for name, tensor in prefix.state_dict().items():
        if not name.startswith("prior.proj."):
            torch.testing.assert_close(tensor, control[name], rtol=0, atol=0)
    ids = torch.arange(VOCAB_SIZE, device="cuda").unsqueeze(0)
    with torch.no_grad():
        expected = prefix.identity_bits(ids)
        torch.testing.assert_close(
            mixture.encode_latents(ids), expected, rtol=0, atol=0
        )
        torch.testing.assert_close(prefix.encode_latents(ids), expected, rtol=0, atol=0)
    optimizer = torch.optim.SGD(prefix.parameters(), lr=0.01)
    before_codec = {
        name: tensor.detach().clone()
        for name, tensor in prefix.codec.state_dict().items()
    }
    initial_output = prefix.prior.proj.output.weight.detach().clone()
    loss, _ = prefix(ids)
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert all(parameter.grad is None for parameter in prefix.codec.parameters())
    optimizer.step()
    assert bool((prefix.prior.proj.output.weight != initial_output).any())
    for name, tensor in prefix.codec.state_dict().items():
        torch.testing.assert_close(tensor, before_codec[name], rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(prefix.encode_latents(ids), expected, rtol=0, atol=0)
