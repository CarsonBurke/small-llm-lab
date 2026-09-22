"""FLA GDN2 execution qualification; run GPU tests exclusively through mlq.

These options change kernels, state layout and retained intermediates, not the
recurrence. The literal FP32 CUDA recurrence below is only a numerical oracle.
"""
import os

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gdn2.chunk import chunk_gdn2
from pretraining.nanogpt_mini.gated_delta_model import GatedDeltaGPT
from pretraining.nanogpt_mini.gated_delta_runtime import CompiledGatedDeltaLoss
from pretraining.nanogpt_mini.recurrent_slots_runtime import CUDAGraphMicrobatch


@pytest.fixture(autouse=True)
def cuda_only():
    if not torch.cuda.is_available():
        if os.environ.get("RECURRENT_SLOTS_REQUIRE_CUDA") == "1":
            pytest.fail("Queued GDN2 execution contracts require CUDA")
        pytest.skip("CUDA qualification must be queued through mlq")
    assert torch.cuda.is_bf16_supported()
    torch._dynamo.reset()


def relative(actual, expected, tolerance, name):
    assert actual is not None and expected is not None, name
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    error = ((actual.double() - expected.double()).norm()
             / expected.double().norm().clamp_min(1e-12))
    assert error < tolerance, (name, float(error))


def literal(q, k, v, g, b, w, state):
    outputs = []
    for t in range(q.shape[1]):
        query, key, value, decay, erase, write = [x[:, t].float() for x in (q, k, v, g, b, w)]
        state = decay.exp().unsqueeze(-1) * state
        prediction = ((erase * key).unsqueeze(-1) * state).sum(-2)
        state = state + key.unsqueeze(-1) * (write * value - prediction).unsqueeze(-2)
        outputs.append(((query * q.shape[-1] ** -.5).unsqueeze(-1) * state).sum(-2))
    return torch.stack(outputs, 1), state


@pytest.mark.parametrize("state_v_first,disable_recompute,key_dim", [
    (False, False, 64), (False, True, 64), (True, False, 64),
    (True, True, 64), (True, True, 128),
    (False, False, 128), (False, True, 128), (True, False, 128),
])
def test_fla_layout_and_retention_match_literal_outputs_and_all_input_gradients(
        state_v_first, disable_recompute, key_dim):
    torch.manual_seed(311)
    shape = (2, 128, 2, key_dim)
    value_shape = (*shape[:-1], 128)
    inputs = [
        F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16(),
        F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16(),
        torch.randn(value_shape, device="cuda", dtype=torch.bfloat16) * .2,
        (-.01 - torch.rand(shape, device="cuda") * .1).bfloat16(),
        torch.rand(shape, device="cuda", dtype=torch.bfloat16),
        torch.rand(value_shape, device="cuda", dtype=torch.bfloat16),
    ]
    canonical_state = torch.randn(2, 2, key_dim, 128, device="cuda") * .03
    state = canonical_state.transpose(-1, -2).contiguous() if state_v_first else canonical_state
    leaves = [x.detach().requires_grad_() for x in (*inputs, state)]
    references = [x.detach().clone().requires_grad_() for x in (*inputs, canonical_state)]
    actual, final = chunk_gdn2(*leaves[:6], initial_state=leaves[-1],
                              output_final_state=True, state_v_first=state_v_first,
                              disable_recompute=disable_recompute)
    canonical_final = final.transpose(-1, -2) if state_v_first else final
    expected, expected_final = literal(*references)
    relative(actual, expected, .015, "output")
    relative(canonical_final, expected_final, .015, "final state")
    upstream = torch.randn_like(expected)
    state_upstream = torch.randn_like(expected_final) * .03
    ((actual.float() * upstream).sum() + (canonical_final * state_upstream).sum()).backward()
    ((expected * upstream).sum() + (expected_final * state_upstream).sum()).backward()
    for name, actual_leaf, reference_leaf in zip(("query", "key", "value", "decay", "erase", "write", "initial_state"), leaves, references):
        gradient = actual_leaf.grad
        if name == "initial_state" and state_v_first:
            gradient = gradient.transpose(-1, -2)
        relative(gradient, reference_leaf.grad, .05, name)
        assert reference_leaf.grad.norm() > 0, name


def model(*, production=False, fused=True, backend="fla", state_v_first=True, disable_recompute=True,
          custom_ops=False, gate_in_kernel=False):
    torch.manual_seed(733)
    geometry = dict(vocab_size=1024, num_layers=6, model_dim=512, head_dim=128, mixer_dim=512) if production else dict(
        vocab_size=32, num_layers=2, model_dim=128, head_dim=128, mixer_dim=128)
    net = GatedDeltaGPT(**geometry, expand_v=1., fused_projections=fused,
                       gdn_backend=backend, state_v_first=state_v_first,
                       disable_recompute=disable_recompute, custom_ops=custom_ops,
                       gate_in_kernel=gate_in_kernel).cuda()
    with torch.no_grad():
        for name, parameter in net.named_parameters():
            if name == "proj.weight" or name.endswith(("mlp.proj.weight", "o_proj.weight")):
                parameter.normal_(std=.003)
    return net


@pytest.mark.parametrize("state_v_first,custom_ops,gate_in_kernel",
                         [(False, False, False), (True, False, False), (False, True, False),
                          (True, True, False), (False, True, True), (True, True, True)])
def test_optimized_model_causality_row_isolation_and_cached_continuation(state_v_first, custom_ops, gate_in_kernel):
    """Evaluation paths agree with the training-shaped forward.

    Prefixes and continuations longer than one chunk exercise the released
    chunk operator with an initial or final state, the path the custom
    operators hand back to the library; with gate_in_kernel it must take the
    raw decay projection there too.
    """
    net = model(state_v_first=state_v_first, custom_ops=custom_ops, gate_in_kernel=gate_in_kernel).eval()
    tokens = torch.randint(32, (2, 128), device="cuda", dtype=torch.int32)
    altered = tokens.clone()
    altered[0, 73:] = (altered[0, 73:] + 1) % 32
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        whole, _ = net.forward_hidden(tokens)
        other, _ = net.forward_hidden(altered)
        torch.testing.assert_close(other[0, :73], whole[0, :73], rtol=0, atol=0)
        torch.testing.assert_close(other[1], whole[1], rtol=0, atol=0)
        assert (other[0, 73:] - whole[0, 73:]).norm() > .01
        prefix, cache = net.forward_hidden(tokens[:, :64], use_cache=True)
        resumed, _ = net.forward_hidden(tokens[:, 64:], state=cache, use_cache=True)
        relative(torch.cat((prefix, resumed), 1), whole, .02, "recurrent continuation")
        for split in (96, 32):  # chunked prefix with a final state; chunked continuation from a state
            long_prefix, long_cache = net.forward_hidden(tokens[:, :split], use_cache=True)
            long_resumed, _ = net.forward_hidden(tokens[:, split:], state=long_cache, use_cache=True)
            relative(torch.cat((long_prefix, long_resumed), 1), whole, .02, f"chunk continuation at {split}")
        reset, cache = net.forward_hidden(tokens[:, :64], use_cache=True)
        relative(reset, prefix, .001, "fresh cache")
        logits = []
        for token in tokens[:, 64:72].unbind(1):
            prediction, cache = net.step(token, cache)
            logits.append(prediction)
        relative(torch.stack(logits, 1), net.logits(whole[:, 64:72]), .02, "token cache")
        fresh, _ = net.forward_hidden(tokens)
        torch.testing.assert_close(fresh, whole, rtol=0, atol=0)


@pytest.mark.parametrize("reference_kind", ["separate_projections", "vendor_backend"])
def test_optimized_model_outputs_and_parameter_gradients_match_reference(reference_kind):
    actual = model().train()
    reference = (model(fused=False) if reference_kind == "separate_projections" else
                 model(backend="vendor", state_v_first=False, disable_recompute=False)).train()
    reference.load_state_dict(actual.state_dict(), strict=True)
    assert reference.state_dict().keys() == actual.state_dict().keys()
    tokens = torch.randint(32, (2, 128), device="cuda", dtype=torch.int32)
    targets = torch.randint(32, tokens.shape, device="cuda")
    actual_loss, reference_loss = CompiledGatedDeltaLoss(actual, 64), CompiledGatedDeltaLoss(reference, 64)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        actual_hidden, _ = actual.forward_hidden(tokens)
        reference_hidden, _ = reference.forward_hidden(tokens)
        relative(actual.logits(actual_hidden), reference.logits(reference_hidden), .02, reference_kind + " logits")
        observed, expected = actual_loss(tokens, targets), reference_loss(tokens, targets)
    observed.backward()
    expected.backward()
    actual_loss.audit_graph_breaks()
    reference_loss.audit_graph_breaks()
    torch.testing.assert_close(observed, expected, rtol=1e-4, atol=.003)
    references = dict(reference.named_parameters())
    for name, parameter in actual.named_parameters():
        relative(parameter.grad, references[name].grad, .03, reference_kind + " " + name)


@pytest.mark.parametrize("custom_ops", [False, True])
def test_production_b64_full_context_graph_observes_inputs_and_packed_parameter_updates(custom_ops):
    import gc

    net = model(production=True, state_v_first=False, custom_ops=custom_ops).train()
    loss = CompiledGatedDeltaLoss(net, 64)
    assert loss.fullgraph is custom_ops
    tokens = torch.randint(1024, (64, 1024), device="cuda", dtype=torch.int32)
    targets = torch.randint(1024, tokens.shape, device="cuda")
    changed_tokens = (tokens + 7) % 1024
    changed_targets = (targets + 11) % 1024

    def reference(inputs, labels):
        net.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            expected = loss(inputs, labels)
            expected.backward()
        # CPU storage is only for comparison artifacts, never model execution.
        gradients = {name: p.grad.detach().float().cpu().clone()
                     for name, p in net.named_parameters()}
        scalar = expected.detach().cpu().clone()
        assert torch.isfinite(scalar)
        assert all(torch.isfinite(gradient).all() for gradient in gradients.values())
        net.zero_grad(set_to_none=True)
        del expected
        return scalar, gradients

    # Reference full-B64 activations cannot coexist with the graph's private
    # pool. Finish every reference before capture, retaining only CPU results.
    original = reference(tokens, targets)
    changed_input = reference(changed_tokens, changed_targets)
    packed_parameters = [projection.weight for block in net.blocks
                         for projection in (block.attn.q_proj, block.attn.k_proj,
                                            block.attn.v_proj, block.attn.b_proj,
                                            block.attn.w_proj, block.attn.f_proj[0],
                                            block.attn.g_proj[0])]
    original_weights = [parameter.detach().cpu().clone() for parameter in packed_parameters]
    with torch.no_grad():
        for parameter in packed_parameters:
            parameter.add_(torch.randn_like(parameter) * .02)
    mutated_weights = [parameter.detach().cpu().clone() for parameter in packed_parameters]
    changed_weights = reference(changed_tokens, changed_targets)
    assert abs(float(changed_input[0] - original[0])) > .01
    # A sum near 454k has FP32 spacing .03125; small useful changes can round
    # to the same scalar loss. Require changed derivatives instead, then check
    # every captured derivative against its corresponding reference below.
    # Five-percent separation makes the two 1.5%-relative-error acceptance
    # regions disjoint, so a stale gradient cannot pass both comparisons.
    assert any(torch.linalg.vector_norm(changed_weights[1][name].double() - gradient.double())
               > .05 * torch.maximum(torch.linalg.vector_norm(changed_weights[1][name].double()),
                                     torch.linalg.vector_norm(gradient.double())).clamp_min(1e-12)
               for name, gradient in changed_input[1].items())
    loss.audit_graph_breaks()
    with torch.no_grad():
        for parameter, saved in zip(packed_parameters, original_weights):
            parameter.copy_(saved)
    gc.collect()
    torch.cuda.empty_cache()

    graph = CUDAGraphMicrobatch(loss, batch_size=64, seq_len=1024)
    loss.audit_graph_breaks()

    def compare(inputs, labels, expected):
        scalar, gradients = expected
        graph.zero_grad()
        observed = graph.replay(inputs, labels).clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(observed, scalar.to(observed.device), rtol=1e-5, atol=.01)
        for name, parameter in net.named_parameters():
            relative(parameter.grad, gradients[name].to(parameter.device), .015, name)
        return observed

    compare(tokens, targets, original)
    compare(changed_tokens, changed_targets, changed_input)
    with torch.no_grad():
        for parameter, saved in zip(packed_parameters, mutated_weights):
            parameter.copy_(saved)
    compare(changed_tokens, changed_targets, changed_weights)
    loss.audit_graph_breaks()


def custom_operator_inputs(gate_in_kernel=False):
    """Kernel inputs; with ``gate_in_kernel`` ``g`` is the raw bf16 decay projection plus decay parameters."""
    torch.manual_seed(977)
    batch, time, heads, key_dim, value_dim = 2, 128, 2, 128, 128
    tensors = dict(
        q=torch.randn(batch, time, heads, key_dim, device="cuda", dtype=torch.bfloat16),
        k=torch.randn(batch, time, heads, key_dim, device="cuda", dtype=torch.bfloat16),
        v=torch.randn(batch, time, heads, value_dim, device="cuda", dtype=torch.bfloat16) * .2,
        g=-torch.rand(batch, time, heads, key_dim, device="cuda") * .1,
        b=torch.rand(batch, time, heads, key_dim, device="cuda", dtype=torch.bfloat16),
        w=torch.rand(batch, time, heads, value_dim, device="cuda", dtype=torch.bfloat16))
    decay = dict(A_log=None, dt_bias=None)
    if gate_in_kernel:
        tensors["g"] = torch.randn(batch, time, heads, key_dim, device="cuda", dtype=torch.bfloat16)
        decay = dict(A_log=torch.empty(heads, device="cuda").uniform_(1, 16).log(),
                     dt_bias=torch.empty(heads * key_dim, device="cuda").uniform_(-6, -2))
    return tensors, key_dim ** -.5, decay


def frontend_log_decay(raw, A_log, dt_bias):
    """The vendored frontend's fp32 decay from the raw projection (what the kernels compute in-kernel)."""
    heads, key_dim = raw.shape[2], raw.shape[3]
    rate = A_log.float().exp().repeat_interleave(key_dim).view(heads, key_dim)
    return -rate * F.softplus(raw.float() + dt_bias.view(heads, key_dim))


KERNEL_OPTIONS = [(False, False), (True, False), (False, True), (True, True)]


@pytest.mark.parametrize("state_v_first,gate_in_kernel", KERNEL_OPTIONS)
def test_custom_operators_satisfy_library_contracts(state_v_first, gate_in_kernel):
    """torch.library.opcheck: schemas, fake kernels, autograd registration and AOT dispatch."""
    from pretraining.nanogpt_mini import gated_delta_ops as ops
    tensors, scale, decay = custom_operator_inputs(gate_in_kernel)
    leaves = {name: tensor.clone().requires_grad_() for name, tensor in tensors.items()}
    parameters = {name: None if value is None else value.clone().requires_grad_() for name, value in decay.items()}
    forward_args = (leaves["q"], leaves["k"], leaves["v"], leaves["g"], leaves["b"], leaves["w"],
                    parameters["A_log"], parameters["dt_bias"], None, None, scale, state_v_first)
    # FLA leaves the masked triangle of the intra-chunk Aqk/Akk intermediates
    # uninitialized by design, so the dispatcher's elementwise output
    # comparison is not meaningful for the forward; whole-graph parity is
    # covered by test_whole_graph_custom_ops_model_matches_graph_break_model.
    torch.library.opcheck(ops.chunk_fwd, forward_args,
                          test_utils=("test_schema", "test_autograd_registration", "test_faketensor"))
    with torch.no_grad():
        outputs = ops.chunk_fwd(*(tensors[n] for n in "qkvgbw"), decay["A_log"], decay["dt_bias"], None, None,
                                scale, state_v_first)
    do = torch.randn_like(outputs[0])
    bwd_args = (do, *outputs[1:5], tensors["v"], outputs[5], tensors["b"], tensors["w"], *outputs[6:],
                tensors["g"] if gate_in_kernel else None, decay["A_log"], decay["dt_bias"], None, None, scale,
                state_v_first)
    torch.library.opcheck(ops.chunk_bwd, bwd_args,
                          test_utils=("test_schema", "test_autograd_registration", "test_faketensor"))
    # The backward kernels reduce with atomics, so compare its compiled
    # execution against eager within kernel-order tolerance instead.
    compiled_bwd = torch.compile(ops.chunk_bwd, fullgraph=True, dynamic=False)
    for eager, traced in zip(ops.chunk_bwd(*bwd_args), compiled_bwd(*bwd_args)):
        if eager.numel():
            relative(traced, eager, 1e-3, "compiled chunk_bwd")
    packed = torch.randn(2, 128, 3 * 128, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(128, 4, device="cuda") * .3).requires_grad_()
    x = packed[..., 128:256]
    assert not x.is_contiguous() and x.stride(-1) == 1
    torch.library.opcheck(ops.conv_fwd, (x.detach().requires_grad_(), weight))
    torch.library.opcheck(ops.conv_bwd, (x.detach(), torch.randn_like(x), weight.detach()))
    gate = torch.randn_like(tensors["v"])
    norm_weight = torch.rand(128, device="cuda").requires_grad_()
    torch.library.opcheck(ops.norm_fwd, (tensors["v"].clone().requires_grad_(), gate.clone().requires_grad_(),
                                         norm_weight, 1e-5))
    with torch.no_grad():
        y, rstd = ops.norm_fwd(tensors["v"], gate, norm_weight, 1e-5)
    torch.library.opcheck(ops.norm_bwd, (torch.randn_like(y), tensors["v"], gate, norm_weight.detach(), rstd, 1e-5))


@pytest.mark.parametrize("state_v_first,gate_in_kernel", KERNEL_OPTIONS)
def test_custom_operator_gradients_match_released_operator(state_v_first, gate_in_kernel):
    """The reference is the released operator on the frontend's fp32 decay.

    With the gate computed in-kernel the operator must reproduce the frontend
    decay arithmetic and return its A_log/dt_bias derivatives.
    """
    from pretraining.nanogpt_mini import gated_delta_ops as ops
    tensors, scale, decay = custom_operator_inputs(gate_in_kernel)
    expected_leaves = {name: tensor.clone().requires_grad_() for name, tensor in tensors.items()}
    expected_decay = {name: None if value is None else value.clone().requires_grad_() for name, value in decay.items()}
    expected_g = expected_leaves["g"]
    if gate_in_kernel:
        expected_g = frontend_log_decay(expected_g, expected_decay["A_log"], expected_decay["dt_bias"])
    expected = chunk_gdn2(expected_leaves["q"], expected_leaves["k"], expected_leaves["v"], expected_g,
                          expected_leaves["b"], expected_leaves["w"], scale=scale, use_qk_l2norm_in_kernel=True,
                          disable_recompute=True, state_v_first=state_v_first)[0]
    actual_leaves = {name: tensor.clone().requires_grad_() for name, tensor in tensors.items()}
    actual_decay = {name: None if value is None else value.clone().requires_grad_() for name, value in decay.items()}
    actual = ops.chunk_gdn2_training(*(actual_leaves[n] for n in "qkvgbw"), scale=scale, state_v_first=state_v_first,
                                     **actual_decay)
    # Gradients leave both operators in bf16. The in-kernel gate cumsum accumulates the same fp32 sums
    # in a different order, and its dt_bias gradient reduces the bf16 dg instead of the frontend's fp32
    # dg; single-ulp bf16 differences, about 4e-3 per element, reach the norm-relative comparison as
    # a few 1e-3.
    tolerance = 4e-3 if gate_in_kernel else 1e-3
    if gate_in_kernel:
        relative(actual, expected, tolerance, "output")
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    do = torch.randn_like(expected)
    expected.backward(do)
    actual.backward(do)
    for name in "qkvgbw":
        relative(actual_leaves[name].grad, expected_leaves[name].grad, tolerance, name)
        assert actual_leaves[name].grad.dtype == tensors[name].dtype
    for name in ("A_log", "dt_bias"):
        if gate_in_kernel:
            relative(actual_decay[name].grad, expected_decay[name].grad, tolerance, name)
            assert actual_decay[name].grad.dtype == torch.float32 and expected_decay[name].grad.norm() > 0


@pytest.mark.parametrize("state_v_first,gate_in_kernel", KERNEL_OPTIONS)
def test_whole_graph_custom_ops_model_matches_graph_break_model(state_v_first, gate_in_kernel):
    actual = model(state_v_first=state_v_first, custom_ops=True, gate_in_kernel=gate_in_kernel).train()
    reference = model(state_v_first=state_v_first).train()
    assert actual.state_dict().keys() == reference.state_dict().keys()
    for (name, left), (_, right) in zip(actual.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0, msg=name)
    tokens = torch.randint(32, (2, 128), device="cuda", dtype=torch.int32)
    targets = torch.randint(32, tokens.shape, device="cuda")
    # Graph-break counters are process-global: audit each model before the
    # other one compiles.
    actual_loss = CompiledGatedDeltaLoss(actual, 64)
    assert actual_loss.fullgraph
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        observed = actual_loss(tokens, targets)
    observed.backward()
    assert actual_loss.audit_graph_breaks() == {}
    reference_loss = CompiledGatedDeltaLoss(reference, 64)
    assert not reference_loss.fullgraph
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        expected = reference_loss(tokens, targets)
    expected.backward()
    assert len(reference_loss.audit_graph_breaks()) == 3
    torch.testing.assert_close(observed, expected, rtol=1e-4, atol=.003)
    references = dict(reference.named_parameters())
    for name, parameter in actual.named_parameters():
        relative(parameter.grad, references[name].grad, .01, name)
