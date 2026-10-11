"""Query-shift numerical qualification. GPU workloads must use mlq."""
import json
import os
from pathlib import Path

import pytest
import torch

from pretraining.nanogpt_mini.gated_delta_dense_reads import QueryShiftSharedPoolGPT
from pretraining.nanogpt_mini.gated_delta_pool import SharedPoolGatedDeltaGPT
from pretraining.tests.test_gated_delta_pool_gpu import first_pass, PRODUCTION


@pytest.fixture(autouse=True)
def cuda_only():
    if not torch.cuda.is_available():
        if os.environ.get("RECURRENT_SLOTS_REQUIRE_CUDA") == "1":
            pytest.fail("Queued query-shift qualification requires CUDA")
        pytest.skip("Query-shift CUDA qualification must be queued through mlq")
    torch._dynamo.reset()


def error(actual, expected):
    a, b = actual.detach().double().flatten(), expected.detach().double().flatten()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def pool_comparison(original, shifted, tokens):
    shifted.load_state_dict(original.state_dict(), strict=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ports = first_pass(original, tokens)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected, _ = original._pool(ports)
        actual, _ = shifted._pool(ports)
    expected = torch.stack([torch.stack(layer) for layer in expected])
    actual = torch.stack([torch.stack(layer) for layer in actual])
    upstream = torch.randn_like(expected)
    (expected.float() * upstream.float()).sum().backward()
    (actual.float() * upstream.float()).sum().backward()
    errors = {"output": error(actual, expected)}
    for (name, parameter), (other_name, other) in zip(original.named_parameters(), shifted.named_parameters()):
        assert name == other_name
        if parameter.grad is None:
            assert other.grad is None
            continue
        errors[name] = error(other.grad, parameter.grad)
    print(json.dumps({"relative_errors": errors}), flush=True)
    assert errors["output"] < .01
    assert max(value for name, value in errors.items() if name != "output") < .02


@pytest.mark.parametrize("heads,dim", [(1, 128), (2, 64)])
def test_full_context_forward_backward_and_prefix_causality(heads, dim):
    torch.manual_seed(413)
    options = dict(vocab_size=32, num_layers=3, model_dim=128, mixer_dim=128,
                   head_dim=dim, pool_heads=heads, pool_writer="layer", pool_reads=3,
                   pool_readout="concat", **PRODUCTION)
    original = SharedPoolGatedDeltaGPT(**options).cuda()
    shifted = QueryShiftSharedPoolGPT(**options).cuda()
    # Source adapters mix heads, and readouts must carry a nonzero signal.
    with torch.no_grad():
        for block in original.blocks:
            block.pool_o_proj.weight.normal_(std=.02)
        for adapter in original.adapters.parameters():
            adapter.add_(torch.randn_like(adapter) * .01)
    tokens = torch.randint(32, (2, 1024), device="cuda", dtype=torch.int32)
    pool_comparison(original, shifted, tokens)
    altered = tokens.clone()
    altered[:, 513:] = torch.randint(32, altered[:, 513:].shape, device="cuda", dtype=torch.int32)
    compiled = torch.compile(shifted.forward_hidden, fullgraph=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prefix = compiled(tokens)[0][:, :513]
        changed = compiled(altered)[0][:, :513]
    torch.testing.assert_close(changed, prefix, rtol=0, atol=0)
    altered = tokens.clone()
    altered[1] = torch.randint(32, (1024,), device="cuda", dtype=torch.int32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        first_sequence = compiled(tokens)[0][0]
        other_sequence_changed = compiled(altered)[0][0]
    torch.testing.assert_close(other_sequence_changed, first_sequence, rtol=0, atol=0)


def test_learned_six_bank_full_context_forward_backward():
    if "STATE_FLOW_CHECKPOINT" not in os.environ:
        pytest.skip("Learned numerical qualification requires STATE_FLOW_CHECKPOINT")
    checkpoint = Path(os.environ["STATE_FLOW_CHECKPOINT"])
    payload = torch.load(checkpoint, map_location="cuda", weights_only=True)
    assert payload["completed_steps"] == 1000
    original = SharedPoolGatedDeltaGPT(**payload["model_config"]).cuda()
    original.load_state_dict(payload["model"], strict=True)
    shifted = QueryShiftSharedPoolGPT(**payload["model_config"]).cuda()
    from scripts.train_recurrent_slots import PackedBatches
    config = json.loads((checkpoint.parent / "config.json").read_text())
    loader = PackedBatches(str(Path(config["data_path"]) / "fineweb_val_*.bin"), 1024, 1024)
    inputs, _ = loader.next()
    pool_comparison(original, shifted, inputs)
