"""GPU-only final-state carry contracts; run through mlq."""
import os

import pytest
import torch
import torch.nn.functional as F

from pretraining.nanogpt_mini.latent_carry import FinalStateDeltaCarry, delta_write
from pretraining.nanogpt_mini.latent_carry_runtime import LatentCarryLoss
from pretraining.nanogpt_mini.recurrent_slots_runtime import CUDAGraphMicrobatch, CUDAGraphValidation


@pytest.fixture(autouse=True)
def cuda_only():
    if not torch.cuda.is_available():
        if os.environ.get("RECURRENT_SLOTS_REQUIRE_CUDA") == "1":
            pytest.fail("Queued contracts require CUDA")
        pytest.skip("Requires queued CUDA execution")
    assert torch.cuda.is_bf16_supported()


def active_model(production=False):
    torch.manual_seed(733)
    model = FinalStateDeltaCarry(**({} if production else dict(
        vocab_size=32, num_layers=2, model_dim=128, heads=2, key_dim=8, value_dim=64))).cuda()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("proj.weight"):
                parameter.normal_(std=.003)
    return model


def relative(actual, expected, tolerance, name):
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    error = ((actual.double() - expected.double()).norm()
             / expected.double().norm().clamp_min(1e-12))
    assert error < tolerance, (name, float(error))


def test_delta_update_matches_independent_matrix_form_and_gradients():
    torch.manual_seed(13)
    memory = torch.randn(2, 2, 8, 16, device="cuda") * .05
    key = F.normalize(torch.randn(2, 2, 8, device="cuda"), dim=-1)
    value = torch.randn(2, 2, 16, device="cuda")
    retention = torch.rand(2, 2, device="cuda") * .4 + .5
    strength = torch.rand(2, 2, device="cuda") * .6 + .2
    leaves = [x.requires_grad_() for x in (memory, key, value, retention, strength)]
    reference = [x.detach().double().requires_grad_() for x in leaves]
    actual = delta_write(*leaves)
    m, k, v, a, b = reference
    outer = k.unsqueeze(-1) @ k.unsqueeze(-2)
    transition = a[..., None, None] * (torch.eye(8, device="cuda", dtype=torch.float64)
                                      - b[..., None, None] * outer)
    expected = transition @ m + b[..., None, None] * (k.unsqueeze(-1) @ v.unsqueeze(-2))
    torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=1e-7)
    upstream = torch.randn_like(actual)
    (actual * upstream).sum().backward()
    (expected * upstream.double()).sum().backward()
    for name, leaf, ref in zip(("memory", "key", "value", "retention", "strength"), leaves, reference):
        relative(leaf.grad, ref.grad, 1e-5, name)
    # A unit key written at full strength precisely replaces that association.
    replaced = delta_write(memory.detach(), key.detach(), value.detach(), retention, torch.ones_like(strength))
    recovered = (replaced * key.detach()[..., None]).sum(-2)
    torch.testing.assert_close(recovered, value.detach(), rtol=1e-5, atol=1e-6)


def test_same_snapshot_final_write_and_deep_feedback_to_next_first_layer():
    model = active_model().eval()
    tokens = torch.tensor([3, 7], device="cuda", dtype=torch.int32)
    memory = model.initial_memory(2, "cuda")
    original = memory.clone()
    seen = []
    hooks = [block.register_forward_pre_hook(lambda module, args: seen.append(args[1].data_ptr()))
             for block in model.blocks]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        hidden, written = model.step(tokens, memory)
        expected = model.write(hidden, memory)
        torch.testing.assert_close(written, expected, rtol=0, atol=0)
        assert seen == [memory.data_ptr()] * len(model.blocks)
        torch.testing.assert_close(memory, original, rtol=0, atol=0)
        assert written.norm() > 0
        # Changing the writer must not change the same token's prediction.
        model.write_value.weight.mul_(1.5)
        same_hidden, different_write = model.step(tokens, memory)
        torch.testing.assert_close(same_hidden, hidden, rtol=0, atol=0)
        assert (different_write - written).norm() > .1
        model.write_value.weight.div_(1.5)
        # A change in final-block computation must affect the stored deep state
        # and therefore the next token's first block, without requiring a pass.
        x_next = model.norm1(model.embed(tokens + 1))
        first_before = model.blocks[0](x_next, written)
        model.blocks[-1].mlp.proj.weight.add_(torch.randn_like(model.blocks[-1].mlp.proj.weight) * .01)
        changed_hidden, changed_memory = model.step(tokens, memory)
        first_after = model.blocks[0](x_next, changed_memory)
        assert (changed_hidden - hidden).float().norm() > .01
        assert (first_after - first_before).float().norm() > .001
    for hook in hooks:
        hook.remove()


def test_compiled_segments_preserve_causality_continuation_and_temporal_gradients():
    torch._dynamo.reset()
    model = active_model()
    model.compile_segments()
    x = torch.randint(32, (2, 16), device="cuda", dtype=torch.int32)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        whole, state = model.forward_hidden(x, segment_size=4)
        changed = x.clone()
        changed[:, 9:] = (changed[:, 9:] + 1) % 32
        other, _ = model.forward_hidden(changed, segment_size=4)
        torch.testing.assert_close(other[:, :9], whole[:, :9], rtol=0, atol=0)
        first, carried = model.forward_hidden(x[:, :8], segment_size=4)
        carried.retain_grad()
        second, continued = model.forward_hidden(x[:, 8:], memory=carried, segment_size=4)
        torch.testing.assert_close(torch.cat((first, second), 1), whole, rtol=0, atol=0)
        torch.testing.assert_close(continued, state, rtol=0, atol=0)
        # Late loss must differentiate through the prefix's final-state write.
        late_loss = model.logits(second[:, -1]).square().sum()
        late_loss.backward()
    assert carried.grad is not None and carried.grad.norm() > 0
    assert model.write_value.weight.grad.norm() > 0
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        sequential = model.initial_memory(2, "cuda")
        states = []
        for token in x.unbind(1):
            h, sequential = model.step(token, sequential)
            states.append(h)
        reference = torch.stack(states, 1)
        actual, final = model.forward_hidden(x, segment_size=4)
        relative(actual, reference, .02, "compiled versus token sequence")
        relative(final, sequential, .02, "continued final state")


def test_production_compiled_graph_gradients_mutation_and_validation():
    torch._dynamo.reset()
    model = active_model(production=True).train()
    loss = LatentCarryLoss(model, segment_size=16)
    graph = CUDAGraphMicrobatch(loss, batch_size=2, seq_len=64)
    x = torch.randint(1024, (2, 64), device="cuda", dtype=torch.int32)
    y = x.roll(-1, 1).long()

    def compare():
        graph.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            expected = loss(x, y)
            expected.backward()
        gradients = {name: p.grad.detach().float().clone() for name, p in model.named_parameters()}
        assert all(torch.isfinite(g).all() for g in gradients.values())
        graph.zero_grad()
        actual = graph.replay(x, y).clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected.detach(), rtol=1e-5, atol=.002)
        for name, p in model.named_parameters():
            relative(p.grad, gradients[name], .02, name)
        return actual

    original = compare()
    with torch.no_grad():
        model.write_value.weight.mul_(1.15)
        model.blocks[-1].mlp.proj.weight.add_(torch.randn_like(model.blocks[-1].mlp.proj.weight) * .002)
    changed = compare()
    assert abs(float(changed - original)) > .001
    model.eval()
    validation = CUDAGraphValidation(loss, batch_size=2, seq_len=64)
    outputs = validation.replay(x, y)
    assert len(outputs) == 1
    torch.testing.assert_close(outputs[0], changed, rtol=1e-5, atol=.002)
    model.train()
    graph.zero_grad()
    torch.testing.assert_close(graph.replay(x, y), changed, rtol=1e-5, atol=.002)
