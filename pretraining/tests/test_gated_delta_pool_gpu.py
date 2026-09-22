"""Shared-pool GDN2 contracts on CUDA; run exclusively through mlq.

The literal FP32 per-bank recurrence below is only a numerical oracle for the
pool's event-order semantics; the model executes the chunk kernels.
"""
import os

import pytest
import torch
import torch.nn.functional as F

from pretraining.nanogpt_mini.gated_delta_bank_linear import bank_linear
from pretraining.nanogpt_mini.gated_delta_model import GatedDeltaGPT
from pretraining.nanogpt_mini.gated_delta_pool import POOL_READS, SharedPoolGatedDeltaGPT, pool_layout
from pretraining.nanogpt_mini.gated_delta_runtime import (
    CompiledGatedDeltaLoss, CompiledSharedPoolLoss, build_gated_delta_optimizers, compiled_gated_delta_loss,
)
from pretraining.nanogpt_mini.recurrent_slots_runtime import CUDAGraphMicrobatch, CUDAGraphValidation

PRODUCTION = dict(gdn_backend="fla", disable_recompute=True, custom_ops=True, fused_projections=True)
FIELDS = ("query", "key", "value", "decay", "erase", "write")


@pytest.fixture(autouse=True)
def cuda_only():
    if not torch.cuda.is_available():
        if os.environ.get("RECURRENT_SLOTS_REQUIRE_CUDA") == "1":
            pytest.fail("Queued shared-pool contracts require CUDA")
        pytest.skip("CUDA qualification must be queued through mlq")
    assert torch.cuda.is_bf16_supported()
    torch._dynamo.reset()


def relative(actual, expected, tolerance, name):
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    error = ((actual.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-12))
    assert error < tolerance, (name, float(error))


def small(*, banks=3, writer="routed", seed=733, readout=.05):
    """Two-layer pool model with a nonzero readout so the pool path carries gradient."""
    torch.manual_seed(seed)
    net = SharedPoolGatedDeltaGPT(vocab_size=32, num_layers=2, model_dim=128, head_dim=64, mixer_dim=128,
                                  pool_banks=None if writer == "layer" else banks, pool_writer=writer,
                                  **PRODUCTION).cuda()
    with torch.no_grad():
        for name, parameter in net.named_parameters():
            if name == "proj.weight" or name.endswith(("mlp.proj.weight", "attn.o_proj.weight")):
                parameter.normal_(std=.003)
            elif name.endswith("pool_o_proj.weight"):
                parameter.normal_(std=readout)
    return net


def literal_bank(q, k, v, g, b, w):
    """FP32 GDN-2 recurrence over one bank's events with the kernel's L2 normalization."""
    heads, key_dim, value_dim = q.shape[1], q.shape[2], v.shape[2]
    state = torch.zeros(heads, key_dim, value_dim, device=q.device)
    outputs = []
    for t in range(q.shape[0]):
        query, key, value, decay, erase, write = (x[t].float() for x in (q, k, v, g, b, w))
        query = query / (query.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        key = key / (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        state = decay.exp().unsqueeze(-1) * state
        prediction = ((erase * key).unsqueeze(-1) * state).sum(-2)
        state = state + key.unsqueeze(-1) * (write * value - prediction).unsqueeze(-2)
        outputs.append(((query * key_dim ** -.5).unsqueeze(-1) * state).sum(-2))
    return torch.stack(outputs)


def first_pass(net, tokens):
    embedded = net.norm1(net.embed(tokens))
    x, ports = embedded, []
    for block in net.blocks:
        x, port, _ = block.first_pass(x)
        ports.append(port)
    return ports


def pool_events(net, ports):
    """Every event's bank and kernel fields in (token, layer, event) order, plus the read events' positions."""
    layers, events, read_events = len(ports), net.events_per_layer, net.read_events
    batch, time = ports[0].write.shape
    projected = dict(zip(FIELDS, net.pool(torch.stack([port.hidden for port in ports]))))
    fields = {}
    for name, field in projected.items():
        per_event = field.new_zeros(batch, time, layers, events, *field.shape[-2:])
        for layer in range(layers):
            for event in (read_events if name == "query" else [0]):
                per_event[:, :, layer, event] = field[layer]
        fields[name] = per_event.reshape(batch, time * layers * events, *field.shape[-2:])
    banks = []
    for port in ports:
        columns = ([port.write] if net.config["pool_writer"] == "layer" else [])
        columns += [port.reads[..., index] for index in range(POOL_READS)]
        banks.append(torch.stack(columns, -1))
    route = torch.stack(banks, 2).reshape(batch, -1)
    is_read = torch.zeros(batch, time, layers, events, dtype=torch.bool, device=route.device)
    is_read[..., read_events] = True
    return route, fields, is_read.reshape(batch, -1)


def read_events_tensor(net, reads):
    """The model's per-layer reads placed at their events; zero at write-only events."""
    layers, events, read_events = len(reads), net.events_per_layer, net.read_events
    batch, time = reads[0][0].shape[:2]
    placed = reads[0][0].new_zeros(batch, time, layers, events, *reads[0][0].shape[-2:])
    for layer in range(layers):
        for index, event in enumerate(read_events):
            placed[:, :, layer, event] = reads[layer][index]
    return placed.reshape(batch, time * layers * events, *reads[0][0].shape[-2:])


def adapt(x, matrix):
    """``[n, H, Dh]`` rows through one bank's ``[latent, latent]`` adapter."""
    return (x.flatten(1).float() @ matrix.float()).view_as(x).to(x.dtype)


@pytest.mark.parametrize("width", [128, 256])
def test_bank_linear_matches_the_per_bank_reference_forward_and_backward(width):
    torch.manual_seed(101)
    batch, banks, events = 2, 3, 640
    route = torch.randint(banks, (batch, events), device="cuda")
    layout = pool_layout(route, banks, 64)
    filled = torch.zeros(layout.rows, dtype=torch.bool, device="cuda")
    filled[layout.slots.flatten()] = True
    x = torch.randn(layout.rows, width, device="cuda", dtype=torch.bfloat16) * filled[:, None]
    weight = torch.randn(banks, width, width, device="cuda", dtype=torch.bfloat16) / width ** .5
    x.requires_grad_(), weight.requires_grad_()
    y = bank_linear(x, weight, layout.tile_bank, layout.bank_tiles)
    upstream = torch.randn_like(y) * filled[:, None]   # the gather never reads filler rows
    (y.float() * upstream.float()).sum().backward()
    reference_x = x.detach().float().requires_grad_()
    reference_w = weight.detach().float().requires_grad_()
    reference = torch.zeros_like(reference_x)
    tiles = layout.tile_bank.tolist()
    for tile, bank in enumerate(tiles):
        rows = slice(tile * 64, (tile + 1) * 64)
        reference[rows] = reference_x[rows] @ reference_w[bank]
    (reference * upstream.float()).sum().backward()
    relative(y.float(), reference, 1e-2, "forward")
    relative(x.grad.float(), reference_x.grad, 1e-2, "input gradient")
    relative(weight.grad.float(), reference_w.grad, 1e-2, "weight gradient")
    assert weight.grad.dtype == weight.dtype
    assert (y[~filled] == 0).all() and (x.grad[~filled] == 0).all(), "filler rows stay inert"
    # Static shapes: the operator captures into a CUDA graph and compiles as a whole graph.
    static_x, static_w = x.detach().clone().requires_grad_(), weight.detach().clone().requires_grad_()

    def step():
        bank_linear(static_x, static_w, layout.tile_bank, layout.bank_tiles).backward(upstream)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    static_x.grad, static_w.grad = None, None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    static_x.grad.zero_(), static_w.grad.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(static_x.grad, x.grad, rtol=0, atol=0)
    torch.testing.assert_close(static_w.grad, weight.grad, rtol=0, atol=0)
    compiled = torch.compile(lambda a, w: bank_linear(a, w, layout.tile_bank, layout.bank_tiles), fullgraph=True)
    torch.testing.assert_close(compiled(x.detach(), weight.detach()), y.detach(), rtol=0, atol=0)


def test_first_pass_is_the_private_model_and_zero_readout_makes_both_passes_identical():
    net = small(readout=0.0).train()
    torch.manual_seed(733)
    plain = GatedDeltaGPT(vocab_size=32, num_layers=2, model_dim=128, head_dim=64, mixer_dim=128, **PRODUCTION).cuda()
    missing, unexpected = plain.load_state_dict(net.state_dict(), strict=False)
    assert not missing and all(".pool_" in n or ".router." in n or n.startswith(("pool.", "adapters."))
                               for n in unexpected)
    tokens = torch.randint(32, (2, 192), device="cuda", dtype=torch.int32)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        first, second, stats = net.forward_passes(tokens)
        expected, _ = plain.forward_hidden(tokens)
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, first, rtol=0, atol=0)
    assert stats["balance"].item() > 0 and stats["z"].item() > 0 and 0 < stats["load_max"].item() <= 1
    targets = torch.randint(32, tokens.shape, device="cuda")
    loss = CompiledSharedPoolLoss(net, 64)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        second_loss, first_loss, balance, z, load_max = loss(tokens, targets, diagnostics=True)
        total = loss(tokens, targets)
    torch.testing.assert_close(second_loss, first_loss, rtol=0, atol=0)
    regularizer = tokens.numel() * (0.01 * balance + 0.001 * z)
    torch.testing.assert_close(total, first_loss + regularizer, rtol=1e-5, atol=1e-2)
    loss.audit_graph_breaks()


@pytest.mark.parametrize("writer", ["routed", "layer"])
def test_pool_reads_follow_the_literal_adapted_bank_recurrence_in_event_order(writer):
    net = small(writer=writer).train()
    banks = net.config["pool_banks"]
    tokens = torch.randint(32, (2, 192), device="cuda", dtype=torch.int32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        ports = first_pass(net, tokens)
        reads, stats = net._pool(ports)
        route, fields, is_read = pool_events(net, ports)
    batch = route.shape[0]
    layout = pool_layout(route, banks, 64)
    counts = F.one_hot(route, banks).sum(1)
    assert stats["load_max"].item() == pytest.approx(counts.max().item() / route.shape[1], abs=1e-6)
    assert counts.min() > 0 and (counts % 64).any(), "banks are unequal and not chunk-aligned: real packing"
    for index, port in enumerate(ports):
        assert (port.reads[..., 0] != port.reads[..., 1]).all() and (port.weights.sum(-1) - 1).abs().max() < 1e-5
        if writer == "layer":
            assert (port.write == index).all()
        else:
            assert torch.equal(port.write, port.reads[..., 0])
    observed = read_events_tensor(net, reads)
    adapters = net.adapters
    compared = 0
    for b in range(batch):
        for bank in range(banks):
            members = (route[b] == bank).nonzero().flatten()
            slots = layout.slots[b, members]
            assert (slots.diff() == 1).all(), "a bank's events are packed contiguously in event order"
            inputs = [fields[name][b, members] for name in FIELDS]
            inputs[0] = adapt(inputs[0], adapters.query[bank])
            inputs[1] = adapt(inputs[1], adapters.key[bank])
            inputs[2] = adapt(inputs[2], adapters.value[bank])
            expected = adapt(literal_bank(*inputs), adapters.output[bank])
            read_members = is_read[b, members]
            relative(observed[b, members][read_members], expected[read_members], .02, f"sequence {b} bank {bank}")
            compared += int(read_members.sum())
    assert compared == int(is_read.sum()) == batch * 192 * 2 * POOL_READS, "every read event is read"


def test_packed_reads_and_gradients_match_one_kernel_call_per_bank():
    """The packed call with per-tile adapters equals dense per-bank calls with plain matmuls, forward and backward."""
    from pretraining.nanogpt_mini import gated_delta_ops as ops
    banks = 2
    net = small(banks=banks).train()
    tokens = torch.randint(32, (2, 128), device="cuda", dtype=torch.int32)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        ports = first_pass(net, tokens)
        for port in ports:
            port.hidden = port.hidden.detach().requires_grad_()
        reads, _ = net._pool(ports)
    upstream = [[torch.randn_like(read) for read in layer_reads] for layer_reads in reads]
    sum((read.float() * up).sum() for layer_reads, ups in zip(reads, upstream) for read, up in zip(layer_reads, ups)).backward()
    packed_grads = {name: parameter.grad.clone() for name, parameter in net.named_parameters()
                    if name.startswith(("pool.", "adapters."))}
    packed_hidden_grads = [port.hidden.grad.clone() for port in ports]
    net.zero_grad(set_to_none=True)
    for port in ports:
        port.hidden.grad = None
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        route, fields, is_read = pool_events(net, ports)
    observed = read_events_tensor(net, reads).detach()
    up_events = read_events_tensor(net, upstream)
    batch = route.shape[0]
    dense_loss = 0
    for b in range(batch):
        for bank in range(banks):
            members = (route[b] == bank).nonzero().flatten()
            padded = 64 * ((members.numel() + 63) // 64)

            def take(name, matrix=None):
                x = fields[name][b, members]
                if matrix is not None:
                    x = (x.flatten(1) @ matrix.to(x.dtype)).view_as(x)
                return F.pad(x, (0, 0, 0, 0, 0, padded - members.numel())).unsqueeze(0).contiguous()
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                out = ops.chunk_gdn2_training(take("query", net.adapters.query[bank]), take("key", net.adapters.key[bank]),
                                              take("value", net.adapters.value[bank]), take("decay"), take("erase"),
                                              take("write"), state_v_first=net.config["state_v_first"])
                out = (out[0, :members.numel()].flatten(1) @ net.adapters.output[bank].to(out.dtype)).view(
                    -1, *out.shape[-2:])
            read_members = is_read[b, members]
            relative(observed[b, members][read_members], out[read_members], 3e-3, f"sequence {b} bank {bank}")
            dense_loss = dense_loss + (out.float() * up_events[b, members]).sum()
    dense_loss.backward()
    for name, parameter in net.named_parameters():
        if name.startswith(("pool.", "adapters.")):
            relative(packed_grads[name], parameter.grad, 5e-3, f"{name} gradient")
    for index, port in enumerate(ports):
        relative(packed_hidden_grads[index], port.hidden.grad, 5e-3, f"layer {index} input gradient")


@pytest.mark.parametrize("writer", ["routed", "layer"])
def test_second_pass_is_causal_and_the_pool_path_trains_every_pool_parameter(writer):
    net = small(writer=writer).train()
    tokens = torch.randint(32, (2, 192), device="cuda", dtype=torch.int32)
    altered = tokens.clone()
    altered[0, 101:] = (altered[0, 101:] + 1) % 32
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        _, whole, _ = net.forward_passes(tokens)
        _, other, _ = net.forward_passes(altered)
    torch.testing.assert_close(other[0, :101], whole[0, :101], rtol=0, atol=0)
    torch.testing.assert_close(other[1], whole[1], rtol=0, atol=0)
    assert (other[0, 101:] - whole[0, 101:]).norm() > .01
    targets = torch.randint(32, tokens.shape, device="cuda")
    loss = compiled_gated_delta_loss(net, 64)
    assert type(loss) is CompiledSharedPoolLoss and loss.fullgraph
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        total = loss(tokens, targets)
    total.backward()
    assert loss.audit_graph_breaks() == {}
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    # The second pass reads what the first pass wrote: the private model's gradient differs from a standalone run.
    torch.manual_seed(733)
    plain = GatedDeltaGPT(vocab_size=32, num_layers=2, model_dim=128, head_dim=64, mixer_dim=128, **PRODUCTION).cuda()
    plain.load_state_dict(net.state_dict(), strict=False)
    plain_loss = CompiledGatedDeltaLoss(plain, 64)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        plain_loss(tokens, targets).backward()
    assert (plain.blocks[0].attn.o_proj.weight.grad - net.blocks[0].attn.o_proj.weight.grad).norm() > 0


def randomize_readouts(net):
    """The head, the MLP readouts and the pool readouts start at zero, which leaves the blocks without a
    cross-entropy gradient at initialization; every gradient below is asserted nonzero."""
    with torch.no_grad():
        for name, parameter in net.named_parameters():
            if name == "proj.weight" or name.endswith(("mlp.proj.weight", "pool_o_proj.weight")):
                parameter.normal_(std=.003)


def test_production_b16_shared_pool_compiles_whole_and_replays_training_and_validation_graphs():
    """Microbatch 16: at 32 the previous pool's training graph warmup ran out of memory beside the
    validation graph (mlq job 8854)."""
    import gc
    import time

    torch.manual_seed(1337)
    net = SharedPoolGatedDeltaGPT(**PRODUCTION).cuda().train()
    randomize_readouts(net)
    assert sum(p.numel() for p in net.parameters()) == 30_027_034
    loss = compiled_gated_delta_loss(net, 64)
    assert loss.fullgraph
    optimizers = build_gated_delta_optimizers(net)
    tokens = torch.randint(1024, (16, 1024), device="cuda", dtype=torch.int32)
    targets = torch.randint(1024, tokens.shape, device="cuda")
    net.eval()
    validation = CUDAGraphValidation(loss, batch_size=16, seq_len=1024)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        outputs = validation.replay(tokens, targets)
        second, first, balance, z, load_max = (o.clone() for o in outputs)
    assert torch.isfinite(second) and torch.isfinite(first) and balance > 0 and z > 0 and 0 < load_max <= 1
    net.train()

    def gradient_report():
        return {name: (bool(torch.isfinite(p.grad).all()), float(p.grad.abs().sum()), str(p.grad.dtype))
                for name, p in net.named_parameters()}

    def defective(report):
        return {name: value for name, value in report.items() if not value[0] or value[1] == 0}

    # The compiled backward at production scale without graph capture first, so a capture defect
    # is told apart from a model defect.
    net.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        loss(tokens, targets).backward()
    eager_report = gradient_report()
    print(f"eager compiled gradients: {eager_report}", flush=True)
    assert not defective(eager_report), ("eager compiled backward", defective(eager_report))
    net.zero_grad(set_to_none=True)
    # Graph capture allocates from a private pool: release the uncaptured pass's cached blocks
    # and measure the footprint of the co-resident graphs alone, which is what training holds.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    graph = CUDAGraphMicrobatch(loss, batch_size=16, seq_len=1024)
    assert loss.audit_graph_breaks() == {}
    graph.zero_grad()
    replayed = graph.replay(tokens, targets).clone()
    assert torch.isfinite(replayed)
    graph_report = gradient_report()
    assert not defective(graph_report), ("graph replay", defective(graph_report),
                                         {name: eager_report[name] for name in defective(graph_report)})
    for name, (_, eager_sum, _) in eager_report.items():
        assert graph_report[name][1] == pytest.approx(eager_sum, rel=.05), (name, eager_sum, graph_report[name])
    for optimizer in optimizers:
        optimizer.step()
    graph.zero_grad()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(4):
        graph.replay(tokens, targets)
    torch.cuda.synchronize()
    per_microbatch = (time.perf_counter() - started) / 4
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    reserved = torch.cuda.max_memory_reserved() / 2 ** 20
    print(f"shared_pool b16 microbatch {per_microbatch * 1000:.1f} ms, peak allocated {peak:.0f} MiB, reserved {reserved:.0f} MiB", flush=True)
    assert reserved < 26_000, "B16 pool graphs must leave headroom on the 32 GiB device"
    del graph, validation, loss, optimizers, net
    gc.collect()
    torch.cuda.empty_cache()


def test_production_layer_writer_compiles_whole_and_trains_every_parameter():
    import gc

    torch.manual_seed(1337)
    net = SharedPoolGatedDeltaGPT(pool_writer="layer", **PRODUCTION).cuda().train()
    randomize_readouts(net)
    assert net.config["pool_banks"] == 6 and sum(p.numel() for p in net.parameters()) == 28_435_738
    loss = compiled_gated_delta_loss(net, 64)
    tokens = torch.randint(1024, (16, 1024), device="cuda", dtype=torch.int32)
    targets = torch.randint(1024, tokens.shape, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        total = loss(tokens, targets)
    total.backward()
    assert loss.audit_graph_breaks() == {}
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0, name
    del loss, net
    gc.collect()
    torch.cuda.empty_cache()
