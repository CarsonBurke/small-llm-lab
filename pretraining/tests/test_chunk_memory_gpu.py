"""Chunk-memory CUDA bf16 contracts; submit through mlq, no CPU fallback."""
import os

import pytest
import torch
import torch.nn.functional as F

from pretraining.nanogpt_mini.chunk_memory import ChunkMemory
from pretraining.nanogpt_mini.chunk_memory_runtime import CompiledFullLoss
from pretraining.nanogpt_mini.recurrent_slots_runtime import CUDAGraphMicrobatch


@pytest.fixture(autouse=True)
def cuda_only():
    if not torch.cuda.is_available():
        if os.environ.get("RECURRENT_SLOTS_REQUIRE_CUDA") == "1":
            pytest.fail("queued chunk contracts require CUDA")
        pytest.skip("CUDA contracts must run through mlq")
    if not torch.cuda.is_bf16_supported():
        pytest.fail("bf16 required")


def model(production=False):
    torch.manual_seed(981)
    kwargs = {} if production else dict(vocab_size=32, num_layers=3, model_dim=32,
                                        head_dim=16, slots=5, chunk_size=4)
    net = ChunkMemory(**kwargs).cuda()
    with torch.no_grad():
        for name, p in net.named_parameters():
            if name.endswith("proj.weight"):
                p.normal_(std=0.003 if production else 0.03)
    return net


def tokens(net, chunks=3):
    return torch.randint(net.config["vocab_size"], (2, net.config["chunk_size"] * chunks),
                         device="cuda", dtype=torch.int32)


def test_causal_prefix_within_and_across_chunks():
    net = model().eval()
    x = tokens(net)
    with torch.no_grad():
        reference, _ = net.forward_hidden(x)
        for position in (2, 4, 6, 8):
            changed = x.clone()
            changed[:, position:] = (changed[:, position:] + 1) % net.config["vocab_size"]
            actual, _ = net.forward_hidden(changed)
            torch.testing.assert_close(reference[:, :position], actual[:, :position], rtol=0, atol=0)


def test_whole_chunk_incremental_execution_and_fresh_state():
    net = model().eval()
    x = tokens(net)
    initial = net.initial_memory(2, x.device)
    before = initial.clone()
    with torch.no_grad():
        expected, final = net.forward_hidden(x, initial)
        state, pieces = initial, []
        for chunk in x.split(net.config["chunk_size"], 1):
            hidden, state = net.process_chunk(chunk, state)
            pieces.append(hidden)
        torch.testing.assert_close(torch.cat(pieces, 1), expected, rtol=0, atol=0)
        torch.testing.assert_close(state, final, rtol=0, atol=0)
        torch.testing.assert_close(initial, before, rtol=0, atol=0)
        assert state.data_ptr() != initial.data_ptr()
        net.forward_hidden((x + 3) % net.config["vocab_size"])
        fresh, fresh_state = net.forward_hidden(x)
        torch.testing.assert_close(fresh, expected, rtol=0, atol=0)
        torch.testing.assert_close(fresh_state, final, rtol=0, atol=0)


def test_writer_receives_future_chunk_loss():
    net = model()
    x = tokens(net, 2)
    chunk_size = net.config["chunk_size"]
    first, state = net.process_chunk(x[:, :chunk_size], net.initial_memory(2, x.device))
    state.retain_grad()
    writer = [(name, p) for name, p in net.named_parameters() if name.startswith("writer_")]
    assert writer, "writer parameters must be identifiable"
    current = torch.autograd.grad(net.logits(first).square().mean(), [p for _, p in writer],
                                  allow_unused=True, retain_graph=True)
    assert all(gradient is None for gradient in current)
    second, _ = net.process_chunk(x[:, chunk_size:], state)
    targets = ((x[:, chunk_size:] + 1) % net.config["vocab_size"]).long()
    F.cross_entropy(net.logits(second).flatten(0, 1), targets.flatten()).backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert state.grad.float().norm() > 0
    nonzero = []
    for name, p in writer:
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        nonzero.append(float(p.grad.float().norm()))
    assert max(nonzero) > 0


def test_checkpoint_serialization_preserves_state_and_predictions(tmp_path):
    net = model().eval()
    x = tokens(net, 2)
    with torch.no_grad():
        _, state = net.process_chunk(x[:, :4], net.initial_memory(2, x.device))
        expected, final = net.process_chunk(x[:, 4:], state)
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_config": net.config, "model": net.state_dict(), "memory": state}, path)
    saved = torch.load(path, map_location="cuda", weights_only=True)
    restored = ChunkMemory(**saved["model_config"]).cuda().eval()
    restored.load_state_dict(saved["model"])
    with torch.no_grad():
        actual, restored_state = restored.process_chunk(x[:, 4:], saved["memory"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(restored_state, final, rtol=0, atol=0)


def test_production_full_context_cuda_graph_loss_and_gradient_parity():
    net = model(production=True).train()
    x = tokens(net, 4)
    assert x.shape == (2, 1024)
    targets = x.roll(-1, 1).long()
    loss_fn = CompiledFullLoss(net, segment_size=256)
    graph = CUDAGraphMicrobatch(loss_fn, batch_size=2, seq_len=1024)
    graph.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        reference_loss = loss_fn(x, targets)
        reference_loss.backward()
    expected = {name: p.grad.detach().float().clone() for name, p in net.named_parameters()}
    graph.zero_grad()
    actual_loss = graph.replay(x, targets).clone()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual_loss, reference_loss.detach(), rtol=1e-5, atol=1e-3)
    for name, p in net.named_parameters():
        actual = p.grad.detach().float()
        assert torch.isfinite(actual).all() and torch.isfinite(expected[name]).all(), name
        error = (actual.double() - expected[name].double()).norm() / expected[name].double().norm().clamp_min(1e-12)
        assert error < 0.01, f"{name}: graph gradient relative error {float(error)}"
