"""Exact encoder/refiner contracts. GPU execution must be submitted through mlq."""
import os

import pytest
import torch
import torch.nn.functional as F

from pretraining.nanogpt_mini.latent_refiner import LatentRefiner
from pretraining.nanogpt_mini.latent_refiner_runtime import LatentRefinerLoss
from pretraining.nanogpt_mini.recurrent_slots_runtime import (
    CUDAGraphMicrobatch, CUDAGraphValidation,
)


@pytest.fixture(autouse=True)
def cuda_only():
    if not torch.cuda.is_available():
        if os.environ.get("RECURRENT_SLOTS_REQUIRE_CUDA") == "1":
            pytest.fail("Queued contracts require CUDA")
        pytest.skip("Requires queued CUDA execution")
    assert torch.cuda.is_bf16_supported()
    torch._dynamo.reset()


def active_model(source="refined", production=False, checkpoint_stride=2):
    torch.manual_seed(733)
    geometry = {} if production else dict(vocab_size=32, encoder_layers=2,
                                          model_dim=128, refiner_hidden=256)
    model = LatentRefiner(source=source, checkpoint_stride=checkpoint_stride, **geometry).cuda()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("proj.weight"):
                parameter.normal_(std=.003)
    return model


def relative(actual, expected, tolerance, name):
    assert actual is not None and expected is not None, name
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    error = ((actual.double() - expected.double()).norm()
             / expected.double().norm().clamp_min(1e-12))
    assert error < tolerance, (name, float(error))


def manual_refine(model, encoded, gates):
    """Literal residual formula and source scheduling, without model dispatch."""
    previous = torch.zeros_like(encoded[:, 0])
    outputs = []
    for index in range(encoded.shape[1]):
        if index == 0:
            source = torch.zeros_like(previous)
        elif model.config["source"] == "current":
            source = encoded[:, index]
        elif model.config["source"] == "encoder":
            source = encoded[:, index - 1]
        else:
            source = previous
        z = encoded[:, index] + gates[:, index] * source
        previous = model.final_norm(z + model.refiner_mlp(model.refiner_norm(z))).float()
        outputs.append(previous)
    return torch.stack(outputs, 1), previous


@pytest.mark.parametrize("source", ["current", "encoder", "refined"])
def test_source_contract_streamed_refinement_and_fresh_sequence(source):
    model = active_model(source).eval()
    tokens = torch.randint(32, (2, 32), device="cuda", dtype=torch.int32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        encoded, gates = model.encode(tokens)
        expected, expected_final = manual_refine(model, encoded, gates)
        actual, final = model.forward_hidden(tokens, segment_size=4)
        relative(actual, expected, .015, source)
        relative(final, expected_final, .015, source + " final")
        # Segment scheduling cannot reset carry or make writes visible late.
        other_segment, _ = model.refine_from_encoded(encoded, gates, segment_size=8)
        relative(other_segment, expected, .015, source + " segment size")
        model.forward_hidden(tokens.flip(1), segment_size=4)
        fresh, _ = model.forward_hidden(tokens, segment_size=4)
        torch.testing.assert_close(fresh, actual, rtol=0, atol=0)
        # All controls use zero source at t=0, even the current-state control.
        first = model.refine_step(encoded[:, 0], gates[:, 0], torch.zeros_like(encoded[:, 0]))
        relative(actual[:, 0], first, .015, source + " first source")


@pytest.mark.parametrize("source", ["current", "encoder", "refined"])
def test_prefix_causality_and_batch_row_isolation(source):
    model = active_model(source).eval()
    tokens = torch.randint(32, (2, 32), device="cuda", dtype=torch.int32)
    changed = tokens.clone()
    changed[0, 17:] = (changed[0, 17:] + 1) % 32
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        before, _ = model.forward_hidden(tokens, segment_size=4)
        after, _ = model.forward_hidden(changed, segment_size=4)
    torch.testing.assert_close(after[0, :17], before[0, :17], rtol=0, atol=0)
    torch.testing.assert_close(after[1], before[1], rtol=0, atol=0)
    assert (after[0, 17:] - before[0, 17:]).norm() > .01



@pytest.mark.parametrize("source", ["current", "encoder", "refined"])
def test_cached_decode_matches_full_prefix_and_resets_without_sliding(source):
    model = active_model(source).eval()
    tokens = torch.randint(32, (2, 16), device="cuda", dtype=torch.int32)
    cache = model.new_cache(batch_size=2, max_seq_len=16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        expected, _ = model.forward_hidden(tokens, segment_size=4)
    actual = torch.stack([model.decode_step(token, cache) for token in tokens.unbind(1)], 1)
    relative(actual, expected, .025, source + " cached whole prefix")
    # Compare each position, so a late-only mismatch cannot hide in a sequence norm.
    for index in range(tokens.shape[1]):
        relative(actual[:, index], expected[:, index], .025, f"{source} cached position {index}")
    assert cache.position == 16
    with pytest.raises(ValueError, match="capacity"):
        model.decode_step(tokens[:, 0], cache)
    assert cache.position == 16
    fresh = model.new_cache(batch_size=2, max_seq_len=16)
    first = model.decode_step(tokens[:, 0], fresh)
    torch.testing.assert_close(first, actual[:, 0], rtol=0, atol=0)
    assert fresh.position == 1
    # Rejected exhausted-cache access must not contaminate other cache objects.
    second = model.decode_step(tokens[:, 1], fresh)
    torch.testing.assert_close(second, actual[:, 1], rtol=0, atol=0)


def test_source_arms_have_identical_initial_parameters():
    states = []
    for source in ("current", "encoder", "refined"):
        model = active_model(source)
        states.append({name: value.clone() for name, value in model.state_dict().items()})
    assert states[0].keys() == states[1].keys() == states[2].keys()
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)
        torch.testing.assert_close(states[0][name], states[2][name], rtol=0, atol=0)


def test_late_loss_crosses_checkpoint_boundary_into_state_and_encoder():
    model = active_model(checkpoint_stride=1).train()
    tokens = torch.randint(32, (2, 16), device="cuda", dtype=torch.int32)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        encoded, _ = model.encode(tokens)
        encoded.retain_grad()
        # Raise retention only for this stress test: default .05 suppresses
        # long refiner-only derivatives and would make a detach test vacuous.
        gates = torch.full_like(encoded, .8, requires_grad=True)
        hidden, _ = model.refine_from_encoded(encoded, gates, segment_size=4)
        upstream = torch.randn_like(hidden[:, 8])
        loss = (hidden[:, 8] * upstream).sum()
        loss.backward()
    assert encoded.grad[:, 3].norm() > 1e-6
    assert gates.grad[:, 3].norm() > 1e-6
    torch.testing.assert_close(encoded.grad[:, 9:], torch.zeros_like(encoded.grad[:, 9:]), rtol=0, atol=0)
    assert model.embed.weight.grad is not None and model.embed.weight.grad.norm() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("source,checkpoint_stride", [
    ("current", 2), ("encoder", 2), ("refined", 0), ("refined", 1), ("refined", 2),
])
def test_compiled_gradients_match_uncheckpointed_token_oracle(source, checkpoint_stride):
    reference = active_model(source, checkpoint_stride=0)
    model = active_model(source, checkpoint_stride=checkpoint_stride)
    tokens = torch.randint(32, (2, 16), device="cuda", dtype=torch.int32)
    targets = torch.randint(32, tokens.shape, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        encoded, gates = reference.encode(tokens)
        hidden, _ = manual_refine(reference, encoded, gates)
        expected = F.cross_entropy(reference.logits(hidden).flatten(0, 1), targets.flatten(), reduction="sum")
        expected.backward()
        actual_hidden, _ = model.forward_hidden(tokens, segment_size=4)
        actual = F.cross_entropy(model.logits(actual_hidden).flatten(0, 1), targets.flatten(), reduction="sum")
        actual.backward()
    torch.testing.assert_close(actual, expected, rtol=.001, atol=.01)
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        relative(parameter.grad, reference_parameters[name].grad, .04, name)


@pytest.mark.parametrize("source", ["current", "encoder", "refined"])
def test_production_full_horizon_graph_loss_mutation_and_validation(source):
    model = active_model(source, production=True).train()
    model.compile_segments()
    loss = LatentRefinerLoss(model, segment_size=16)
    # All sources use the same bounded head at production T. B5 crosses the
    # 4096-row split; compare loss/gradients against the unsplit full head.
    graph = CUDAGraphMicrobatch(loss, batch_size=5, seq_len=1024)
    tokens = torch.randint(1024, (5, 1024), device="cuda", dtype=torch.int32)
    targets = torch.randint(1024, tokens.shape, device="cuda")

    def compare():
        graph.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            hidden, _ = model.forward_hidden(tokens, segment_size=16)
            expected = F.cross_entropy(model.logits(hidden).flatten(0, 1), targets.flatten(), reduction="sum")
            expected.backward()
        gradients = {name: p.grad.detach().float().clone() for name, p in model.named_parameters()}
        graph.zero_grad()
        actual = graph.replay(tokens, targets).clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected.detach(), rtol=2e-5, atol=.02)
        for name, parameter in model.named_parameters():
            relative(parameter.grad, gradients[name], .04, name)
        return actual

    before = compare()
    with torch.no_grad():
        model.proj.weight.add_(torch.randn_like(model.proj.weight) * .002)
    after = compare()
    assert abs(float(after - before)) > .01
    model.eval()
    validation = CUDAGraphValidation(loss, batch_size=5, seq_len=1024)
    outputs = validation.replay(tokens, targets)
    assert len(outputs) == 1
    torch.testing.assert_close(outputs[0], after, rtol=2e-5, atol=.02)
    model.train()
    graph.zero_grad()
    torch.testing.assert_close(graph.replay(tokens, targets), after, rtol=2e-5, atol=.02)
