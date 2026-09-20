"""CUDA graph and update-boundary regressions for every objective; execute through mlq."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from pretraining.future_credit_stream.config import Config
from pretraining.future_credit_stream.model import StreamingFFNModel
from pretraining.future_credit_stream.objective import CE_STATISTICS, FUTURE_BAG_STATISTICS
from pretraining.future_credit_stream.training import Engine, build_optimizer
from pretraining.nextlat import rational_softcap


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

BOS = 1
LANES = 4


def _config(objective, **overrides):
    return replace(Config(), vocab_size=32, model_dim=128, num_layers=2,
                   mlp_hidden=256, document_batch=LANES, stream_steps=4, horizon=3,
                   discount=0.7, objective=objective, compile_mode="reduce-overhead", **overrides)


def _model(config):
    model = StreamingFFNModel(**config.model_config).cuda()
    with torch.no_grad():
        model.proj.weight.normal_(std=0.04)
        for block in model.blocks:
            block[1].proj.weight.normal_(std=0.03)
        if model.writer is not None:
            # A nontrivial gate exposes retention and gate-parameter gradients.
            model.writer.gate_hidden.weight.normal_(std=0.05)
            model.writer.gate_hidden.bias.fill_(0.5)
            model.writer.gate_carry.normal_(std=0.05)
    return model


def _assert_vector_close(actual, expected, *, label, rtol=0.02, atol=1e-6):
    assert torch.isfinite(actual).all(), label
    assert torch.isfinite(expected).all(), label
    # BF16 cancellation may change individual near-zero signs. Preserve the
    # 2% vector-norm bound without asserting precision at those coordinates.
    error = (actual.float() - expected.float()).norm()
    limit = rtol * expected.float().norm() + atol
    assert error <= limit, f"{label}: vector error {error.item():.6g} > {limit.item():.6g}"


def _partial(parameter, weight):
    return parameter.detach() + weight * (parameter - parameter.detach())


def _writer(model, hidden, previous):
    writer = model.writer
    hidden, previous = hidden.bfloat16(), previous.bfloat16()
    logit = (F.linear(hidden, writer.gate_hidden.weight.bfloat16(), writer.gate_hidden.bias.bfloat16())
             + F.linear(previous, writer.gate_carry.bfloat16()))
    gate = logit.sigmoid()
    write = F.linear(hidden, writer.write.weight.bfloat16(), writer.write.bias.bfloat16())
    size = (hidden.shape[-1],)
    return gate * F.rms_norm(write, size) + (1 - gate) * F.rms_norm(previous, size), gate


def _bag(config, target, future):
    """Discounted bag over the target and the future by explicit per-lane loops."""
    bag = torch.zeros(LANES, config.vocab_size, device="cuda")
    for lane in range(LANES):
        bag[lane, int(target[lane])] += 1.0
        if target[lane] == BOS:
            continue
        for offset in range(config.horizon):
            token = int(future[offset, lane])
            bag[lane, token] += config.discount ** (offset + 1)
            if token == BOS:
                break
    return bag / bag.sum(-1)[:, None]


def _lens(model, carry, weight):
    """Eager head-on-carry equations, independent of the module's own method."""
    carry = carry.bfloat16()
    gains = _partial(model.final_norm.gains, weight).bfloat16()
    head_weight = _partial(model.proj.weight, weight).bfloat16()
    head_bias = _partial(model.proj.bias, weight).bfloat16()
    direction = F.rms_norm(carry, (carry.shape[-1],), weight=gains)
    return rational_softcap(F.linear(direction, head_weight, head_bias).float(), softcap=15.0)


def _reference_page(model, config, page, carry):
    """Independent eager equations for the local objectives.

    Current CE trains the backbone; the written carry's cross-entropy to the
    hand-built bag, read through the head, trains the writer. No
    Objective/Engine helpers, temporal gradient, or page-end special case.
    """
    steps = page["inputs"].shape[0]
    current = carry.detach().clone()
    total = torch.zeros((), device="cuda")
    names = FUTURE_BAG_STATISTICS if model.writer is not None else CE_STATISTICS
    statistics = torch.zeros(len(names), device="cuda")
    weight = config.backbone_future_weight
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for tick, (observed, target, reset) in enumerate(zip(page["inputs"], page["targets"], page["resets"])):
            state = current.masked_fill(reset[:, None], 0)
            logits, hidden = model(observed, state)
            ce = F.cross_entropy(logits.float(), target)
            total = total + ce
            statistics[0] += ce.detach()
            if model.writer is None:
                current = hidden.detach().clone()
                continue
            write_input = hidden.detach() + weight * (hidden - hidden.detach())
            next_carry, gate = _writer(model, write_input, state)
            bag = _bag(config, target, page["future"][tick])
            log_belief = _lens(model, next_carry, weight).log_softmax(-1)
            bag_loss = -(bag * log_belief).sum(-1).mean()
            total = total + bag_loss
            statistics[1] += bag_loss.detach()
            with torch.no_grad():
                hidden_log_belief = _lens(model, hidden.detach(), 0.0).log_softmax(-1)
                statistics[2] += -(bag * hidden_log_belief).sum(-1).mean()
            statistics[3] += (-(torch.xlogy(bag, bag)).sum(-1)).mean()
            statistics[4] += gate.float().mean()
            statistics[5] += F.cosine_similarity(next_carry.detach().float(), hidden.detach().float(), dim=-1).mean()
            current = next_carry.detach().clone()
        (total / steps).backward()
    carry.copy_(current)
    return statistics


def _reference_temporal_page(model, page, carry):
    """Eager unrolled page with live carry graphs; truncated at the page edge."""
    state = carry.detach().clone()
    total = torch.zeros((), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for observed, target, reset in zip(page["inputs"], page["targets"], page["resets"]):
            state = state.masked_fill(reset[:, None], 0)
            logits, hidden = model(observed, state, temporal_gradient=True)
            total = total + F.cross_entropy(logits.float(), target)
            state = hidden
        (total / page["inputs"].shape[0]).backward()
    statistics = torch.zeros(len(CE_STATISTICS), device="cuda")
    statistics[0] = total.detach()
    carry.copy_(state.detach())
    return statistics


def _page(tokens, steps, horizon):
    """A page whose future targets are read past the page edge, like the loader."""
    inputs = tokens[:steps]
    page = {"inputs": inputs, "targets": tokens[1:steps + 1], "resets": inputs == BOS}
    if horizon:
        page["future"] = torch.stack([tokens[tick + 2:tick + 2 + horizon] for tick in range(steps)])
    return page


def _stream(config, updates):
    length = updates * config.stream_steps + 1 + config.horizon
    stream = torch.randint(2, config.vocab_size, (length, LANES), device="cuda")
    for offset in range(0, updates * config.stream_steps, config.stream_steps):
        stream[offset, 3] = BOS
        stream[offset + 2, 0] = BOS
        stream[offset + config.stream_steps, 1] = BOS
    stream[3, 2] = BOS
    return stream


def _prepare(config, seed):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    model = _model(config)
    reference = StreamingFFNModel(**config.model_config).cuda()
    reference.load_state_dict(model.state_dict())
    optimizers = [build_optimizer(item, config) for item in (model, reference)]
    # External gradient buffers survive zeroing and multiple optimizer pages.
    for item in (model, reference):
        for parameter in item.parameters():
            parameter.grad = torch.zeros_like(parameter)
    return model, reference, optimizers


@pytest.mark.parametrize("objective_name, weight", [
    ("future_bag", 0.0), ("future_bag", 0.5), ("ce", 0.0),
])
def test_compiled_pages_match_independent_reference_across_optimizer_updates(objective_name, weight):
    config = _config(objective_name, backbone_future_weight=weight)
    model, reference, optimizers = _prepare(config, 71)
    engine = Engine(model, config, bos_id=BOS)
    carry = model.initial_state(LANES, "cuda")
    reference_carry = carry.clone()
    stream = _stream(config, 3)
    for update in range(3):
        offset = update * config.stream_steps
        page = _page(stream[offset:], config.stream_steps, config.page_horizon)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=False)
        actual_stats = engine.backward_page(page, carry)
        reference_stats = _reference_page(reference, config, page, reference_carry)
        assert actual_stats.shape == (len(engine.statistics),) and not actual_stats.requires_grad
        _assert_vector_close(actual_stats, reference_stats, label=f"update {update} statistics")
        _assert_vector_close(carry, reference_carry, label=f"update {update} carry")
        assert not carry.requires_grad and carry.grad_fn is None
        for (name, actual), expected in zip(model.named_parameters(), reference.parameters()):
            assert actual.grad is not None and expected.grad is not None, name
            _assert_vector_close(actual.grad, expected.grad, label=f"update {update} {name}")
        if objective_name == "ce":
            assert model.writer is None and engine.statistics == CE_STATISTICS
        else:
            assert engine.statistics == FUTURE_BAG_STATISTICS
            assert torch.count_nonzero(actual_stats) == len(FUTURE_BAG_STATISTICS)
            assert torch.count_nonzero(model.writer.write.weight.grad) > 0
            assert torch.count_nonzero(model.proj.weight.grad) > 0
        for optimizer in optimizers:
            optimizer.step()
        # Adam magnifies near-zero BF16 sign noise. Compare gradients and future
        # observable behavior, never independently optimized parameter equality.


def test_temporal_reference_page_matches_eager_unrolled_graph():
    config = _config("tbptt")
    model, reference, optimizers = _prepare(config, 53)
    engine = Engine(model, config, bos_id=BOS)
    carry = torch.randn(LANES, 128, device="cuda", dtype=torch.bfloat16)
    reference_carry = carry.clone()
    stream = _stream(config, 2)
    for update in range(2):
        offset = update * config.stream_steps
        page = _page(stream[offset:], config.stream_steps, 0)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=False)
        actual_stats = engine.backward_page(page, carry)
        reference_stats = _reference_temporal_page(reference, page, reference_carry)
        _assert_vector_close(actual_stats, reference_stats, label=f"update {update} statistics")
        _assert_vector_close(carry, reference_carry, label=f"update {update} carry")
        assert not carry.requires_grad and carry.grad_fn is None
        for (name, actual), expected in zip(model.named_parameters(), reference.parameters()):
            _assert_vector_close(actual.grad, expected.grad, label=f"update {update} {name}")
        for optimizer in optimizers:
            optimizer.step()


def test_splitting_pages_without_update_preserves_carry_and_gradients():
    config = replace(_config("future_bag"), discount=0.8)
    torch.set_num_threads(4)
    torch.manual_seed(89)
    whole_model = _model(config)
    split_model = StreamingFFNModel(**config.model_config).cuda()
    split_model.load_state_dict(whole_model.state_dict())
    whole_engine = Engine(whole_model, config, bos_id=BOS)
    split_engine = Engine(split_model, replace(config, stream_steps=2), bos_id=BOS)
    carry = torch.randn(LANES, 128, device="cuda", dtype=torch.bfloat16)
    split_carry = carry.clone()
    tokens = torch.tensor([[3, BOS, 5, 6], [7, 8, 9, 10], [11, BOS, 13, 14],
                           [15, 16, BOS, 18], [19, 20, 21, BOS], [22, 23, 24, 25],
                           [26, 27, 28, 29], [30, 31, 2, 3]], device="cuda")
    for model in (whole_model, split_model):
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
    whole_stats = whole_engine.backward_page(_page(tokens, 4, config.horizon), carry)
    first_stats = split_engine.backward_page(_page(tokens, 2, config.horizon), split_carry).clone()
    second_stats = split_engine.backward_page(_page(tokens[2:], 2, config.horizon), split_carry)
    _assert_vector_close(first_stats + second_stats, whole_stats, label="page split statistics")
    _assert_vector_close(split_carry, carry, label="page split carry")
    for (name, actual), expected in zip(split_model.named_parameters(), whole_model.parameters()):
        # Two two-tick page means sum to twice the four-tick page mean.
        _assert_vector_close(actual.grad / 2, expected.grad, label=f"page split {name}")
    assert not split_carry.requires_grad and split_carry.grad_fn is None
