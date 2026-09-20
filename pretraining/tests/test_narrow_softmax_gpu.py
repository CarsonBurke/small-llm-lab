"""Narrow softmax CUDA contracts. Run exclusively through mlq."""
import os

import pytest
import torch

from pretraining.nanogpt_mini.narrow_softmax_model import NarrowSoftmaxGPT
from pretraining.nanogpt_mini.latent_carry_runtime import CausalControlLoss
from pretraining.nanogpt_mini.recurrent_slots_runtime import CUDAGraphMicrobatch, CUDAGraphValidation
from pretraining.tests.test_gated_delta_gpu import assert_relative


@pytest.fixture(autouse=True)
def cuda_only():
    if not torch.cuda.is_available():
        if os.environ.get("RECURRENT_SLOTS_REQUIRE_CUDA") == "1":
            pytest.fail("queued narrow softmax contracts require CUDA")
        pytest.skip("CUDA contracts require mlq")
    assert torch.cuda.is_bf16_supported(), "BF16 required"


def model(production=False):
    torch.manual_seed(981)
    kwargs = {} if production else dict(vocab_size=32, num_layers=2, model_dim=128)
    net = NarrowSoftmaxGPT(**kwargs).cuda()
    # Activate all output projections so attention/MLP and their gradients are
    # exercised; mini's intentional zero initialization would hide defects.
    with torch.no_grad():
        for name, p in net.named_parameters():
            if name.endswith("proj.weight"):
                p.normal_(std=.003)
    return net


def test_narrow_softmax_causality_fresh_rows_and_loss_interface():
    net = model().eval()
    x = torch.randint(32, (2, 192), device="cuda", dtype=torch.int32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        hidden, state = net.forward_hidden(x)
        assert state is None
        logits = net.logits(hidden)
        assert torch.isfinite(logits).all() and float(logits.std()) > .01
        changed = x.clone()
        changed[:, 73:] = (changed[:, 73:] + 1) % 32
        changed_hidden, _ = net.forward_hidden(changed)
        torch.testing.assert_close(changed_hidden[:, :73], hidden[:, :73], rtol=0, atol=0)
        reset, _ = net.forward_hidden(x)
        torch.testing.assert_close(reset, hidden, rtol=0, atol=0)
        targets = x.roll(-1, 1).long()
        inherited_loss = net(x, targets)
        interface_loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum")
        torch.testing.assert_close(inherited_loss, interface_loss, rtol=0, atol=0)
        # A change to a different packed row must not cross the batch boundary.
        changed[1] = (x[1] + 1) % 32
        changed[0] = x[0]
        isolated, _ = net.forward_hidden(changed)
        torch.testing.assert_close(isolated[0], hidden[0], rtol=0, atol=0)


def test_narrow_softmax_production_compiled_graph_gradients_updates_and_validation():
    torch._dynamo.reset()
    net = model(production=True).train()
    assert net.config == dict(vocab_size=1024, num_layers=6, model_dim=512, mixer_dim=128, head_dim=128)
    assert all(block.attn.num_heads == 1 and block.attn.head_dim == 128 for block in net.blocks)
    loss_fn = CausalControlLoss(net, segment_size=64)
    x = torch.randint(1024, (2, 1024), device="cuda", dtype=torch.int32)
    targets = x.roll(-1, 1).long()
    graph = CUDAGraphMicrobatch(loss_fn, batch_size=2, seq_len=1024)
    pointers = {name: p.grad.data_ptr() for name, p in net.named_parameters()}
    original_gradients = None
    for update in range(2):
        if update:
            with torch.no_grad():
                net.proj.weight.mul_(.97)
                for block in net.blocks:
                    for projection in (block.attn.q, block.attn.k, block.attn.v):
                        projection.weight.add_(torch.randn_like(projection.weight) * .002)
        graph.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            expected = loss_fn(x, targets)
            expected.backward()
        gradients = {}
        for name, p in net.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            gradients[name] = p.grad.detach().float().clone()
        name = "blocks.0.attn.q.weight"
        assert float(gradients[name].norm()) > 0
        if original_gradients is None:
            original_gradients = gradients
        else:
            sensitivity = (gradients[name] - original_gradients[name]).norm() / original_gradients[name].norm()
            assert float(sensitivity) > .02, "weight mutation did not exercise gradient freshness"
        graph.zero_grad()
        actual = graph.replay(x, targets).clone()
        torch.testing.assert_close(actual, expected.detach(), rtol=1e-5, atol=1e-3)
        for name, p in net.named_parameters():
            assert p.grad.data_ptr() == pointers[name], name
            assert_relative(p.grad, gradients[name], .01, f"graph gradient {name}")
    net.eval()
    saved = {name: p.grad.clone() for name, p in net.named_parameters()}
    validation = CUDAGraphValidation(loss_fn, batch_size=2, seq_len=1024)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        expected_validation = loss_fn(x, targets)
    outputs = validation.replay(x, targets)
    assert len(outputs) == 1, "memory-free control must not fabricate memory diagnostics"
    torch.testing.assert_close(outputs[0], expected_validation, rtol=1e-5, atol=1e-3)
    for name, p in net.named_parameters():
        assert p.grad.data_ptr() == pointers[name], name
        torch.testing.assert_close(p.grad, saved[name], rtol=0, atol=0)
    net.train()
    graph.zero_grad()
    actual = graph.replay(x, targets).clone()
    torch.testing.assert_close(actual, expected.detach(), rtol=1e-5, atol=1e-3)
