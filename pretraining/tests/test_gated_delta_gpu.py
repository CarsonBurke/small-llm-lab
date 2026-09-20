"""Pinned GDN-2 CUDA contracts; run exclusively through mlq.

Literal fp32 recurrence is an independent GPU numerical oracle, not a model
execution fallback. Production execution remains bf16 optimized kernels.
"""
import os

import pytest
import torch
import torch.nn.functional as F

from pretraining.gated_delta.vendor.gdn2_ops.chunk_gdn2 import chunk_gdn2
from pretraining.gated_delta.vendor.gdn2_ops.fused_recurrent_gdn2 import fused_recurrent_gdn2
from pretraining.nanogpt_mini.gated_delta_model import GatedDeltaGPT
from pretraining.nanogpt_mini.gated_delta_runtime import CompiledGatedDeltaLoss
from pretraining.nanogpt_mini.recurrent_slots_runtime import CUDAGraphMicrobatch


@pytest.fixture(autouse=True)
def cuda_only():
    if not torch.cuda.is_available():
        if os.environ.get("RECURRENT_SLOTS_REQUIRE_CUDA") == "1":
            pytest.fail("queued GDN-2 contracts require CUDA")
        pytest.skip("CUDA contracts require mlq")
    if not torch.cuda.is_bf16_supported():
        pytest.fail("bf16 required")


def literal(q, k, v, g, b, w, initial_state):
    state = initial_state.float()
    outputs = []
    for index in range(q.shape[1]):
        query, key, value, decay, erase_gate, write_gate = (
            tensor[:, index].float() for tensor in (q, k, v, g, b, w))
        state = state * decay.exp().unsqueeze(-1)
        prediction = ((erase_gate * key).unsqueeze(-1) * state).sum(-2)
        state = state + key.unsqueeze(-1) * (write_gate * value - prediction).unsqueeze(-2)
        outputs.append(((query * q.shape[-1] ** -0.5).unsqueeze(-1) * state).sum(-2))
    return torch.stack(outputs, 1), state


def assert_relative(actual, expected, tolerance, label):
    left, right = actual.double(), expected.double()
    assert torch.isfinite(left).all() and torch.isfinite(right).all(), label
    relative = (left - right).norm() / right.norm().clamp_min(1e-12)
    assert relative < tolerance, f"{label}: relative error {float(relative)}"


def kernel_inputs(length=128, head_dim=128):
    torch.manual_seed(113)
    shape = (1, length, 2, head_dim)
    q = F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16()
    k = F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16()
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.2
    g = (-0.01 - torch.rand(shape, device="cuda") * 0.1).bfloat16()
    b = torch.rand(shape, device="cuda", dtype=torch.bfloat16)
    w = torch.rand(shape, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(1, 2, head_dim, head_dim, device="cuda") * 0.01
    return [value.detach().requires_grad_() for value in (q, k, v, g, b, w, state)]


@pytest.mark.parametrize("head_dim", [64, 128])
def test_pinned_chunk_kernel_forward_backward_matches_literal_recurrence(head_dim):
    leaves = kernel_inputs(head_dim=head_dim)
    reference = [value.detach().clone().requires_grad_() for value in leaves]
    output, final = chunk_gdn2(*leaves[:6], initial_state=leaves[6], output_final_state=True)
    expected, expected_final = literal(*reference)
    assert_relative(output, expected, 0.015, "kernel output")
    assert_relative(final, expected_final, 0.015, "kernel state")
    upstream = torch.randn_like(output).float()
    final_upstream = torch.randn_like(final) * 0.01
    ((output.float() * upstream).sum() + (final * final_upstream).sum()).backward()
    ((expected * upstream).sum() + (expected_final * final_upstream).sum()).backward()
    for name, actual, expected_leaf in zip(("q", "k", "v", "g", "b", "w", "initial_state"), leaves, reference):
        assert actual.grad is not None and expected_leaf.grad is not None, name
        assert_relative(actual.grad, expected_leaf.grad, 0.05, f"gradient {name}")


def test_chunk_and_fused_recurrence_share_immediate_writes_and_continuation():
    leaves = [x.detach() for x in kernel_inputs()]
    with torch.no_grad():
        chunk, final = chunk_gdn2(*leaves[:6], initial_state=leaves[6], output_final_state=True)
        recurrent, recurrent_final = fused_recurrent_gdn2(*leaves[:6], initial_state=leaves[6], output_final_state=True)
        prefix, state = chunk_gdn2(*(x[:, :64].contiguous() for x in leaves[:6]),
                                  initial_state=leaves[6], output_final_state=True)
        suffix, resumed = chunk_gdn2(*(x[:, 64:].contiguous() for x in leaves[:6]),
                                    initial_state=state, output_final_state=True)
    assert_relative(chunk, recurrent, 0.015, "chunk versus token recurrence")
    assert_relative(final, recurrent_final, 0.015, "final recurrent state")
    assert_relative(torch.cat((prefix, suffix), 1), chunk, 0.015, "continued outputs")
    assert_relative(resumed, final, 0.015, "continued state")
    q = torch.zeros(1, 64, 1, 128, device="cuda", dtype=torch.bfloat16)
    q[..., 0] = 1
    value = torch.zeros_like(q)
    value[:, 0, :, 0] = 1
    with torch.no_grad():
        result, _ = chunk_gdn2(q, q, value, torch.full_like(q, -0.1),
                              torch.full_like(q, 0.5), torch.ones_like(q))
    assert float(result[0, 0, 0, 0]) > 0.08, "current-token write is not visible immediately"
    assert float(result[0, 1, 0, 0]) > 0.03, "within-chunk write is not visible to the next token"


def model(production=False, head_dim=128, mixer_dim=None, fused_projections=False):
    torch.manual_seed(981)
    kwargs = {} if production else dict(vocab_size=32, num_layers=2, model_dim=128, head_dim=128)
    kwargs.update(head_dim=head_dim, mixer_dim=mixer_dim, fused_projections=fused_projections)
    net = GatedDeltaGPT(**kwargs).cuda()
    with torch.no_grad():
        net.proj.weight.normal_(std=0.003)
        for name, p in net.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("mlp.proj.weight"):
                p.normal_(std=0.003)
    return net


@pytest.mark.parametrize("fused_projections", [False, True], ids=["separate", "fused"])
def test_model_suffix_causality_and_fresh_state(fused_projections):
    net = model(fused_projections=fused_projections).eval()
    x = torch.randint(32, (2, 192), device="cuda", dtype=torch.int32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected, _ = net.forward_hidden(x)
        changed = x.clone()
        changed[:, 73:] = (changed[:, 73:] + 1) % 32
        actual, _ = net.forward_hidden(changed)
        torch.testing.assert_close(actual[:, :73], expected[:, :73], rtol=0, atol=0)
        net.forward_hidden(changed)
        fresh, _ = net.forward_hidden(x)
        torch.testing.assert_close(fresh, expected, rtol=0, atol=0)


@pytest.mark.parametrize("fused_projections", [False, True], ids=["separate", "fused"])
def test_stream_cache_continuation_matches_full_prefix(fused_projections):
    net = model(fused_projections=fused_projections).eval()
    x = torch.randint(32, (2, 128), device="cuda", dtype=torch.int32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        whole, _ = net.forward_hidden(x)
        first, state = net.forward_hidden(x[:, :64], use_cache=True)
        second, state = net.forward_hidden(x[:, 64:], state=state, use_cache=True)
        assert state is not None
        assert_relative(torch.cat((first, second), 1), whole, 0.02, "stream hidden")
        fresh, _ = net.forward_hidden(x[:, :64], use_cache=True)
        assert_relative(fresh, first, 0.001, "new stream reset")


def test_fused_projection_parameter_contract_and_compiled_all_gradient_parity():
    torch._dynamo.reset()
    separate = model(head_dim=64, mixer_dim=128).train()
    fused = model(head_dim=64, mixer_dim=128, fused_projections=True).train()
    fused.load_state_dict(separate.state_dict(), strict=True)
    original_parameters = dict(fused.named_parameters())
    assert original_parameters.keys() == dict(separate.named_parameters()).keys()
    assert fused.state_dict().keys() == separate.state_dict().keys()
    assert sum(p.numel() for p in fused.parameters()) == sum(p.numel() for p in separate.parameters())
    for name, value in separate.state_dict().items():
        torch.testing.assert_close(fused.state_dict()[name], value, rtol=0, atol=0)

    x = torch.randint(32, (2, 128), device="cuda", dtype=torch.int32)
    targets = x.roll(-1, 1).long()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        hidden, _ = fused.forward_hidden(x)
        logits = fused.logits(hidden)
        assert torch.isfinite(logits).all()
        assert float(logits.std()) > 0.01, "zero readout would make fusion parity vacuous"
    separate_loss = CompiledGatedDeltaLoss(separate, segment_size=64)
    fused_loss = CompiledGatedDeltaLoss(fused, segment_size=64)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        expected = separate_loss(x, targets)
        actual = fused_loss(x, targets)
    expected.backward()
    actual.backward()
    separate_loss.audit_graph_breaks()
    fused_loss.audit_graph_breaks()
    # Fusion changes GEMM and dX accumulation order. The scalar tolerance is
    # 0.01%; every parameter gradient must agree within 2% norm error. This is
    # tighter than the existing 5% chunk-kernel-versus-fp32-oracle threshold.
    torch.testing.assert_close(actual.detach(), expected.detach(), rtol=1e-4, atol=0.002)
    references = dict(separate.named_parameters())
    for name, parameter in fused.named_parameters():
        assert parameter is original_parameters[name], f"replaced original parameter {name}"
        assert parameter.grad is not None and references[name].grad is not None, name
        assert_relative(parameter.grad, references[name].grad, 0.02, f"fused gradient {name}")
    first_projections = ("q_proj", "k_proj", "v_proj", "b_proj", "w_proj", "f_proj.0", "g_proj.0")
    for index in range(len(fused.blocks)):
        for projection in first_projections:
            name = f"blocks.{index}.attn.{projection}.weight"
            assert float(original_parameters[name].grad.norm()) > 0, f"unexercised packed weight {name}"
    assert fused.state_dict().keys() == separate.state_dict().keys(), "packing must not register persistent weights"


def test_fused_single_token_cache_steps_match_full_sequence():
    net = model(head_dim=64, mixer_dim=128, fused_projections=True).eval()
    x = torch.randint(32, (2, 72), device="cuda", dtype=torch.int32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        whole, _ = net.forward_hidden(x)
        expected = net.logits(whole[:, 64:])
        _, state = net.forward_hidden(x[:, :64], use_cache=True)
        steps = []
        for token in x[:, 64:].unbind(1):
            logits, state = net.step(token, state)
            steps.append(logits)
        assert state is not None
        # This includes the already-tested chunk/recurrent kernel difference;
        # use the same 2% continuation tolerance, not a new relaxed bound.
        assert_relative(torch.stack(steps, dim=1), expected, 0.02, "fused token-cache logits")


@pytest.mark.parametrize("head_dim,mixer_dim,fused_projections", [
    (128, 512, False), (64, 512, False), (64, 256, False), (64, 256, True),
], ids=["head128", "head64", "head64_narrow", "head64_narrow_fused"])
def test_production_optimized_full_context_gradients_and_cuda_graph_parity(head_dim, mixer_dim, fused_projections):
    torch._dynamo.reset()
    net = model(production=True, head_dim=head_dim, mixer_dim=mixer_dim,
                fused_projections=fused_projections).train()
    x = torch.randint(1024, (2, 1024), device="cuda", dtype=torch.int32)
    targets = x.roll(-1, 1).long()
    loss_fn = CompiledGatedDeltaLoss(net, segment_size=64)
    graph = CUDAGraphMicrobatch(loss_fn, batch_size=2, seq_len=1024)
    loss_fn.audit_graph_breaks()
    graph.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        expected_loss = loss_fn(x, targets)
        expected_loss.backward()
    expected = {}
    for name, p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        expected[name] = p.grad.detach().float().clone()
    graph.zero_grad()
    actual_loss = graph.replay(x, targets).clone()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual_loss, expected_loss.detach(), rtol=1e-5, atol=1e-3)
    for name, p in net.named_parameters():
        assert p.grad is not None, name
        assert_relative(p.grad, expected[name], 0.01, f"graph gradient {name}")
    # Compiled casts and captured gradients must observe optimizer-style in-place
    # weight changes; validation must preserve the training graph's buffers.
    with torch.no_grad():
        net.proj.weight.mul_(0.97)
        if fused_projections:
            # Mutate every packed source, not only the final vocabulary head.
            # A stale packed-weight cache must fail the following graph check.
            for block in net.blocks:
                for projection in (block.attn.q_proj, block.attn.k_proj, block.attn.v_proj,
                                   block.attn.b_proj, block.attn.w_proj,
                                   block.attn.f_proj[0], block.attn.g_proj[0]):
                    projection.weight.add_(torch.randn_like(projection.weight) * 0.002)
    graph.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        changed_loss = loss_fn(x, targets)
        changed_loss.backward()
    changed_gradients = {name: p.grad.detach().float().clone() for name, p in net.named_parameters()}
    if fused_projections:
        name = "blocks.0.attn.q_proj.weight"
        sensitivity = ((changed_gradients[name] - expected[name]).double().norm()
                       / expected[name].double().norm().clamp_min(1e-12))
        assert float(sensitivity) > 0.02, "packed-weight mutation did not meaningfully exercise gradient freshness"
    graph.zero_grad()
    replayed = graph.replay(x, targets).clone()
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed, changed_loss.detach(), rtol=1e-5, atol=1e-3)
    for name, p in net.named_parameters():
        assert_relative(p.grad, changed_gradients[name], 0.01, f"updated graph gradient {name}")
    from pretraining.nanogpt_mini.recurrent_slots_runtime import CUDAGraphValidation
    net.eval()
    validation = CUDAGraphValidation(loss_fn, batch_size=2, seq_len=1024)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        validation_expected = loss_fn(x, targets)
    outputs = validation.replay(x, targets)
    assert len(outputs) == 1
    torch.testing.assert_close(outputs[0], validation_expected, rtol=1e-5, atol=1e-3)
    loss_fn.audit_graph_breaks()
    net.train()
    graph.zero_grad()
    after_validation = graph.replay(x, targets).clone()
    torch.cuda.synchronize()
    torch.testing.assert_close(after_validation, changed_loss.detach(), rtol=1e-5, atol=1e-3)
