"""CUDA graph and online TD update-boundary regressions; execute through mlq."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from pretraining.future_credit_stream.config import Config
from pretraining.future_credit_stream.model import StreamingFFNModel
from pretraining.future_credit_stream.training import Engine


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _model(config):
    model = StreamingFFNModel(**config.model_config).cuda()
    with torch.no_grad():
        model.proj.weight.normal_(std=0.04)
        for block in model.blocks:
            block[1].proj.weight.normal_(std=0.03)
        if model.critic is not None:
            model.critic.proj.weight.normal_(std=0.03)
            model.critic.proj.bias.fill_(0.2)
    return model


def _assert_vector_close(actual, expected, *, label, rtol=0.02, atol=1e-6):
    assert torch.isfinite(actual).all(), label
    assert torch.isfinite(expected).all(), label
    # BF16 cancellation may change individual near-zero signs. Preserve the
    # 2% vector-norm bound without asserting precision at those coordinates.
    error = (actual.float() - expected.float()).norm()
    limit = rtol * expected.float().norm() + atol
    assert error <= limit, f"{label}: vector error {error.item():.6g} > {limit.item():.6g}"


def _reference_page(model, config, page, carry, has_previous):
    """Independent eager equations: original carry is the previous fit feature.

    Current CE and frozen-weight value train the producer. The same current
    quantities, detached, supervise the previous state. No Objective/Engine
    helpers, future observation, temporal gradient, or page-end special case.
    """
    steps = page["inputs"].shape[0]
    current = carry.detach().clone()
    total = torch.zeros((), device="cuda")
    statistics = torch.zeros(4, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for observed, target, reset in zip(page["inputs"], page["targets"], page["resets"]):
            previous = current
            logits, hidden = model(observed, previous.masked_fill(reset[:, None], 0))
            ce = F.cross_entropy(logits.float(), target, reduction="none")
            total = total + ce.mean()
            statistics[0] += ce.detach().mean()
            if model.critic is not None:
                actor = model.value(hidden, detach_weights=True).masked_fill(target == 1, 0)
                total = total + config.discount * actor.mean()
                if has_previous:
                    prediction = model.value(previous.detach())
                    teacher = (ce.detach() + config.discount * actor.detach()).masked_fill(reset, 0)
                    regression = 0.5 * (prediction - teacher).square().mean()
                    total = total + regression
                    statistics[1:] += torch.stack((regression.detach(), prediction.detach().mean(),
                                                    teacher.mean()))
            current = hidden.detach().clone()
            has_previous = True
        (total / steps).backward()
    carry.copy_(current)
    return statistics, has_previous


def _page(tokens):
    return {"inputs": tokens[:-1], "targets": tokens[1:], "resets": tokens[:-1] == 1}


@pytest.mark.parametrize("objective_name", ["td", "ce"])
def test_compiled_pages_match_independent_reference_across_optimizer_updates(objective_name):
    torch.set_num_threads(4)
    torch.manual_seed(71)
    config = replace(Config(), vocab_size=32, model_dim=128, num_layers=2,
                     mlp_hidden=256, document_batch=4, stream_steps=4,
                     discount=0.7, objective=objective_name, compile_mode="reduce-overhead")
    model = _model(config)
    reference = StreamingFFNModel(**config.model_config).cuda()
    reference.load_state_dict(model.state_dict())
    engine = Engine(model, config, bos_id=1)
    optimizers = [torch.optim.AdamW(item.parameters(), lr=1e-4, fused=True)
                  for item in (model, reference)]
    # External gradient buffers survive zeroing and multiple optimizer pages.
    for item in (model, reference):
        for parameter in item.parameters():
            parameter.grad = torch.zeros_like(parameter)
    carry = model.initial_state(4, "cuda")
    reference_carry = carry.clone()
    reference_has_previous = False
    stream = torch.randint(2, 32, (3 * config.stream_steps + 1, 4), device="cuda")
    for offset in range(0, 3 * config.stream_steps, config.stream_steps):
        stream[offset, 3] = 1
        stream[offset + 2, 0] = 1
        stream[offset + config.stream_steps, 1] = 1
    for update in range(3):
        offset = update * config.stream_steps
        page = _page(stream[offset:offset + config.stream_steps + 1])
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=False)
        actual_stats = engine.backward_page(page, carry)
        reference_stats, reference_has_previous = _reference_page(
            reference, config, page, reference_carry, reference_has_previous,
        )
        assert actual_stats.shape == (4,) and not actual_stats.requires_grad
        _assert_vector_close(actual_stats, reference_stats, label=f"update {update} statistics")
        _assert_vector_close(carry, reference_carry, label=f"update {update} carry")
        assert not carry.requires_grad and carry.grad_fn is None
        for (name, actual), expected in zip(model.named_parameters(), reference.parameters()):
            assert actual.grad is not None and expected.grad is not None, name
            _assert_vector_close(actual.grad, expected.grad, label=f"update {update} {name}")
        if objective_name == "ce":
            assert model.critic is None
            assert torch.count_nonzero(actual_stats[1:]) == 0
        for optimizer in optimizers:
            optimizer.step()
        # Adam magnifies near-zero BF16 sign noise. Compare gradients and future
        # observable behavior, never independently optimized parameter equality.


def test_splitting_pages_without_update_preserves_predecessor_td_and_gradients():
    torch.set_num_threads(4)
    torch.manual_seed(89)
    config = replace(Config(), vocab_size=32, model_dim=128, num_layers=2,
                     mlp_hidden=256, document_batch=4, stream_steps=4,
                     discount=0.8, objective="td", compile_mode="reduce-overhead")
    whole_model = _model(config)
    split_model = StreamingFFNModel(**config.model_config).cuda()
    split_model.load_state_dict(whole_model.state_dict())
    whole_engine = Engine(whole_model, config, bos_id=1)
    split_engine = Engine(split_model, replace(config, stream_steps=2), bos_id=1)
    # Represent a restored stream with an existing predecessor, including a
    # terminal predecessor whose nonzero features must be fitted before reset.
    whole_engine.has_previous = split_engine.has_previous = True
    carry = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    split_carry = carry.clone()
    tokens = torch.tensor([[3, 1, 5, 6], [7, 8, 9, 10], [11, 1, 13, 14],
                           [15, 16, 1, 18], [19, 20, 21, 1]], device="cuda")
    for model in (whole_model, split_model):
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
    whole_stats = whole_engine.backward_page(_page(tokens), carry)
    first_stats = split_engine.backward_page(_page(tokens[:3]), split_carry).clone()
    second_stats = split_engine.backward_page(_page(tokens[2:]), split_carry)
    _assert_vector_close(first_stats + second_stats, whole_stats, label="page split statistics")
    _assert_vector_close(split_carry, carry, label="page split carry")
    for (name, actual), expected in zip(split_model.named_parameters(), whole_model.parameters()):
        # Two two-tick page means sum to twice the four-tick page mean.
        _assert_vector_close(actual.grad / 2, expected.grad, label=f"page split {name}")
    assert not split_carry.requires_grad and split_carry.grad_fn is None
