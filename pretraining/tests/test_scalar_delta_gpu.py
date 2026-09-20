"""Scalar DeltaNet CUDA contracts; execute only through mlq.

The literal FP32 recurrence is an independent numerical oracle on the GPU,
not a model execution fallback. Production paths use compiled BF16 kernels.
"""
import os

import pytest
import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

from pretraining.nanogpt_mini.scalar_delta_model import ScalarDeltaGPT
from pretraining.nanogpt_mini.gated_delta_runtime import CompiledGatedDeltaLoss
from pretraining.nanogpt_mini.recurrent_slots_runtime import CUDAGraphMicrobatch, CUDAGraphValidation
from pretraining.tests.test_gated_delta_gpu import assert_relative


@pytest.fixture(autouse=True)
def cuda_only():
    if not torch.cuda.is_available():
        if os.environ.get("RECURRENT_SLOTS_REQUIRE_CUDA") == "1":
            pytest.fail("queued scalar DeltaNet contracts require CUDA")
        pytest.skip("CUDA contracts require mlq")
    assert torch.cuda.is_bf16_supported(), "BF16 required"


def literal(q, k, v, g, beta, initial_state):
    """State layout [batch,head,value,key]; read includes this token's write."""
    state = initial_state.float()
    outputs = []
    for index in range(q.shape[1]):
        query, key, value, decay, strength = (x[:, index].float() for x in (q, k, v, g, beta))
        state = state * decay.exp()[..., None, None]
        prediction = (state * key.unsqueeze(-2)).sum(-1)
        correction = (value - prediction) * strength.unsqueeze(-1)
        state = state + correction.unsqueeze(-1) * key.unsqueeze(-2)
        outputs.append((state * (query * q.shape[-1] ** -.5).unsqueeze(-2)).sum(-1))
    return torch.stack(outputs, 1), state


def kernel_inputs(head_dim=64, expand_v=1):
    torch.manual_seed(713)
    shape = (1, 128, 2, head_dim)
    q, k = [F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16() for _ in range(2)]
    value_dim = head_dim * expand_v
    v = torch.randn((*shape[:-1], value_dim), device="cuda", dtype=torch.bfloat16) * .2
    g = (-.01 - torch.rand(shape[:-1], device="cuda") * .1).float()
    beta = torch.rand(shape[:-1], device="cuda", dtype=torch.bfloat16)
    initial = torch.randn(1, 2, value_dim, head_dim, device="cuda") * .01
    return [x.detach().requires_grad_() for x in (q, k, v, g, beta, initial)]


@pytest.mark.parametrize("head_dim,expand_v", [(64, 1), (128, 1), (64, 2)],
                         ids=["head64", "head128", "head64_expand2"])
def test_scalar_chunk_forward_backward_matches_literal(head_dim, expand_v):
    inputs = kernel_inputs(head_dim=head_dim, expand_v=expand_v)
    reference = [x.detach().clone().requires_grad_() for x in inputs]
    actual, final = chunk_gated_delta_rule(*inputs[:5], initial_state=inputs[5],
                                           output_final_state=True, state_v_first=True)
    expected, expected_final = literal(*reference)
    assert_relative(actual, expected, .015, "scalar output")
    assert_relative(final, expected_final, .015, "scalar state")
    upstream = torch.randn_like(actual).float()
    final_upstream = torch.randn_like(final) * .01
    ((actual.float() * upstream).sum() + (final * final_upstream).sum()).backward()
    ((expected * upstream).sum() + (expected_final * final_upstream).sum()).backward()
    for name, value, ref in zip(("q", "k", "v", "g", "beta", "initial_state"), inputs, reference):
        assert value.grad is not None and ref.grad is not None, name
        assert float(ref.grad.norm()) > 0, name
        assert_relative(value.grad, ref.grad, .05, name)


def test_scalar_chunk_continuation_and_immediate_write():
    inputs = [x.detach() for x in kernel_inputs()]
    with torch.no_grad():
        whole, final = chunk_gated_delta_rule(*inputs[:5], initial_state=inputs[5],
                                              output_final_state=True, state_v_first=True)
        recurrent, recurrent_final = fused_recurrent_gated_delta_rule(
            q=inputs[0], k=inputs[1], v=inputs[2], g=inputs[3], beta=inputs[4],
            initial_state=inputs[5], output_final_state=True, state_v_first=True)
        prefix, state = chunk_gated_delta_rule(*(x[:, :64].contiguous() for x in inputs[:5]),
                                               initial_state=inputs[5], output_final_state=True, state_v_first=True)
        suffix, resumed = chunk_gated_delta_rule(*(x[:, 64:].contiguous() for x in inputs[:5]),
                                                initial_state=state, output_final_state=True, state_v_first=True)
    assert_relative(whole, recurrent, .015, "chunk/recurrent outputs")
    assert_relative(final, recurrent_final, .015, "chunk/recurrent state")
    assert_relative(torch.cat((prefix, suffix), 1), whole, .015, "continued outputs")
    assert_relative(resumed, final, .015, "continued state")
    q = torch.zeros(1, 64, 1, 64, device="cuda", dtype=torch.bfloat16)
    q[..., 0] = 1
    v = torch.zeros_like(q)
    v[:, 0, :, 0] = 1
    with torch.no_grad():
        output, _ = chunk_gated_delta_rule(q, q, v, torch.full(q.shape[:-1], -.1, device="cuda"),
                                           torch.full(q.shape[:-1], .5, device="cuda", dtype=torch.bfloat16),
                                           state_v_first=True)
    assert float(output[0, 0, 0, 0]) > .06, "current write invisible"
    assert float(output[0, 1, 0, 0]) > .025, "within-chunk write invisible"


def model(production=False, head_dim=64, mixer_dim=None, fused_projections=False, expand_v=1):
    torch.manual_seed(981)
    kwargs = {} if production else dict(vocab_size=32, num_layers=2, model_dim=128, mixer_dim=128)
    kwargs["head_dim"] = head_dim
    kwargs["fused_projections"] = fused_projections
    kwargs["expand_v"] = expand_v
    if mixer_dim is not None:
        kwargs["mixer_dim"] = mixer_dim
    net = ScalarDeltaGPT(**kwargs).cuda()
    with torch.no_grad():
        net.proj.weight.normal_(std=.003)
        for name, parameter in net.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("mlp.proj.weight"):
                parameter.normal_(std=.003)
    return net


@pytest.mark.parametrize("fused_projections", [False, True], ids=["separate", "fused"])
@pytest.mark.parametrize("head_dim,expand_v", [(128, 1), (64, 2)], ids=["head128", "head64_expand2"])
def test_scalar_causality_cache_steps_and_fresh_state(fused_projections, head_dim, expand_v):
    net = model(head_dim=head_dim, mixer_dim=128, fused_projections=fused_projections,
                expand_v=expand_v).eval()
    x = torch.randint(32, (2, 136), device="cuda", dtype=torch.int32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        whole, _ = net.forward_hidden(x)
        changed = x.clone()
        changed[:, 73:] = (changed[:, 73:] + 1) % 32
        other, _ = net.forward_hidden(changed)
        torch.testing.assert_close(other[:, :73], whole[:, :73], rtol=0, atol=0)
        fresh, _ = net.forward_hidden(x)
        torch.testing.assert_close(fresh, whole, rtol=0, atol=0)
        prefix, state = net.forward_hidden(x[:, :64], use_cache=True)
        middle, state = net.forward_hidden(x[:, 64:128], state, use_cache=True)
        assert_relative(torch.cat((prefix, middle), 1), whole[:, :128], .02, "cached prefix")
        outputs = []
        for token in x[:, 128:].unbind(1):
            logits, state = net.step(token, state)
            outputs.append(logits)
        expected = net.logits(whole[:, 128:])
        assert float(expected.std()) > .01, "vacuous zero readout"
        assert_relative(torch.stack(outputs, 1), expected, .02, "cached steps")
        reset, _ = net.forward_hidden(x[:, :64], use_cache=True)
        assert_relative(reset, prefix, .001, "fresh cache")


@pytest.mark.parametrize("head_dim,mixer_dim,fused_projections,expand_v", [
    (64, 256, False, 1), (128, 128, False, 1), (128, 128, True, 1), (64, 128, True, 2),
], ids=["head64_mixer256", "head128_mixer128", "head128_mixer128_fused", "head64_mixer128_expand2_fused"])
def test_scalar_production_compiled_graph_gradients_update_and_validation(head_dim, mixer_dim, fused_projections, expand_v):
    torch._dynamo.reset()
    # All shapes retain 16,384 recurrent scalars per layer and row, while
    # changing projection width, number of independent gates, and retrieval.
    # Equal storage does not imply equivalent architectures.
    net = model(production=True, head_dim=head_dim, mixer_dim=mixer_dim,
                fused_projections=fused_projections, expand_v=expand_v).train()
    assert (mixer_dim // head_dim) * head_dim**2 * expand_v == 16_384
    assert net.config["memory_rule"] == "scalar_delta"
    assert net.config["initialization"] == "gdn2_matched_distributions"
    loss_fn = CompiledGatedDeltaLoss(net, segment_size=64)
    x = torch.randint(1024, (2, 1024), device="cuda", dtype=torch.int32)
    targets = x.roll(-1, 1).long()
    graph = CUDAGraphMicrobatch(loss_fn, batch_size=2, seq_len=1024)
    pointers = {name: p.grad.data_ptr() for name, p in net.named_parameters()}
    first_gradients = None
    for update in range(2):
        if update:
            with torch.no_grad():
                net.proj.weight.mul_(.97)
                for block in net.blocks:
                    for projection in (block.attn.q_proj, block.attn.k_proj, block.attn.v_proj,
                                       block.attn.a_proj, block.attn.b_proj, block.attn.g_proj):
                        projection.weight.add_(torch.randn_like(projection.weight) * .002)
        graph.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            expected = loss_fn(x, targets)
            expected.backward()
        gradients = {}
        for name, p in net.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            gradients[name] = p.grad.detach().float().clone()
        if first_gradients is None:
            first_gradients = gradients
        else:
            name = "blocks.0.attn.q_proj.weight"
            sensitivity = (gradients[name] - first_gradients[name]).norm() / first_gradients[name].norm()
            assert float(sensitivity) > .02, "weight update does not exercise gradient freshness"
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
    assert len(outputs) == 1
    torch.testing.assert_close(outputs[0], expected_validation, rtol=1e-5, atol=1e-3)
    for name, p in net.named_parameters():
        assert p.grad.data_ptr() == pointers[name], name
        torch.testing.assert_close(p.grad, saved[name], rtol=0, atol=0)
    loss_fn.audit_graph_breaks()
    net.train()
    graph.zero_grad()
    after_validation = graph.replay(x, targets).clone()
    torch.testing.assert_close(after_validation, expected.detach(), rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("head_dim,expand_v", [(128, 1), (64, 2)], ids=["head128", "head64_expand2"])
def test_scalar_fused_compiled_projection_loss_and_all_gradient_parity(head_dim, expand_v):
    torch._dynamo.reset()
    separate = model(head_dim=head_dim, mixer_dim=128, expand_v=expand_v).train()
    fused = model(head_dim=head_dim, mixer_dim=128, expand_v=expand_v, fused_projections=True).train()
    fused.load_state_dict(separate.state_dict(), strict=True)
    originals = dict(fused.named_parameters())
    assert originals.keys() == dict(separate.named_parameters()).keys()
    assert fused.state_dict().keys() == separate.state_dict().keys()
    for name, value in separate.state_dict().items():
        torch.testing.assert_close(fused.state_dict()[name], value, rtol=0, atol=0)
    x = torch.randint(32, (2, 128), device="cuda", dtype=torch.int32)
    targets = x.roll(-1, 1).long()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        hidden, _ = fused.forward_hidden(x)
        logits = fused.logits(hidden)
        assert torch.isfinite(logits).all() and float(logits.std()) > .01
    expected_fn = CompiledGatedDeltaLoss(separate, segment_size=64)
    actual_fn = CompiledGatedDeltaLoss(fused, segment_size=64)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        expected = expected_fn(x, targets)
        actual = actual_fn(x, targets)
    expected.backward()
    actual.backward()
    expected_fn.audit_graph_breaks()
    actual_fn.audit_graph_breaks()
    torch.testing.assert_close(actual.detach(), expected.detach(), rtol=1e-4, atol=.002)
    references = dict(separate.named_parameters())
    for name, p in fused.named_parameters():
        assert p is originals[name], f"replaced parameter {name}"
        assert p.grad is not None and references[name].grad is not None, name
        assert_relative(p.grad, references[name].grad, .02, f"fused gradient {name}")
    for index in range(len(fused.blocks)):
        for projection in ("q_proj", "k_proj", "v_proj", "a_proj", "b_proj", "g_proj"):
            name = f"blocks.{index}.attn.{projection}.weight"
            assert float(originals[name].grad.norm()) > 0, f"unexercised packed weight {name}"
    assert fused.state_dict().keys() == separate.state_dict().keys()
