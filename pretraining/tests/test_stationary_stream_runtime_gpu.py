"""CUDA graph, ring-advance, and update-boundary regressions; execute through mlq."""

import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from pretraining.stationary_stream.config import Config
from pretraining.stationary_stream.model import StationaryFFNModel
from pretraining.stationary_stream.objective import STATISTICS
from pretraining.stationary_stream.training import Engine, build_optimizer, evaluate


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

BOS = 1
LANES = 4


def _config(**overrides):
    fields = dict(vocab_size=32, model_dim=128, num_layers=2, mlp_hidden=256, buffer_slots=3, read_heads=2,
                  read_key_dim=8, document_batch=LANES, stream_steps=4, compile_mode="reduce-overhead",
                  read_entry="value_map")
    return replace(Config(), **{**fields, **overrides})


def _model(config):
    model = StationaryFFNModel(**config.model_config).cuda()
    with torch.no_grad():
        model.proj.weight.normal_(std=0.04)
        for block in model.blocks:
            block[1].proj.weight.normal_(std=0.03)
        if config.read_entry == "value_map":
            # A nonzero value map exposes query, key, and bias gradients; a nonzero
            # value bias makes read_rms's bias exclusion observable.
            model.read.value.weight.normal_(std=0.05)
            model.read.value.bias.normal_(std=0.1)
            model.read.null_bias.normal_(std=0.5)
        else:
            # A nonzero query exposes key gradients from the first page.
            model.read.query.weight.normal_(std=0.05)
    return model


def _constant_read(model):
    constant = model.read.constant_read()
    return constant.detach() if torch.is_tensor(constant) else constant


def _assert_vector_close(actual, expected, *, label, rtol=0.02, atol=2e-5):
    assert torch.isfinite(actual).all(), label
    assert torch.isfinite(expected).all(), label
    # BF16 cancellation may change individual near-zero signs. Preserve the
    # 2% vector-norm bound without asserting precision at those coordinates.
    # The absolute term is the BF16 accumulation floor for gradients that are
    # cancellation-dominated by construction: softmax logit gradients sum to
    # zero across slots, so a shared key bias keeps only the part that differs
    # between rotations, a few 1e-5 in norm, where compiled and eager kernels
    # legitimately disagree at the last bit.
    error = (actual.float() - expected.float()).norm()
    limit = rtol * expected.float().norm() + atol
    assert error <= limit, f"{label}: vector error {error.item():.6g} > {limit.item():.6g}"


def _diagnostics(attention, valid, read, value_bias):
    """Explicit per-lane loops over FP32 attention in shift layout (slot j is j+1 old)."""
    batch, heads, width = attention.shape
    nulls, ages, count = torch.zeros((), device="cuda"), torch.zeros((), device="cuda"), 0
    for lane in range(batch):
        if not bool(valid[:, lane].any()):
            continue
        for head in range(heads):
            context = attention[lane, head, 1:]
            nulls = nulls + attention[lane, head, 0]
            ages = ages + (context * torch.arange(1, width, device="cuda")).sum() / context.sum()
            count += 1
    null_mass = nulls / max(count, 1)
    read_age = ages / max(count, 1)
    read_rms = (read.float() - value_bias).square().mean(-1).sqrt().mean()
    return torch.stack((null_mass, read_age, read_rms))


class ShiftedReference:
    """Eager shift-layout state: slot j always holds the hidden from j+1 ticks ago."""

    def __init__(self, slots, batch, dim):
        self.buffer = torch.zeros(slots, batch, dim, device="cuda", dtype=torch.bfloat16)
        self.valid = torch.zeros(slots, batch, device="cuda", dtype=torch.bool)
        self.ages = torch.arange(1, slots + 1, device="cuda")

    def advance(self, hidden, readable):
        self.buffer = torch.cat((hidden.detach()[None], self.buffer[:-1]))
        self.valid = torch.cat((torch.ones_like(readable[:1]), readable[:-1]))


def _reference_page(model, page, reference):
    """Independent eager loop: one graph per tick, explicit shift, no Engine or ring helpers."""
    steps = page["inputs"].shape[0]
    total = torch.zeros((), device="cuda")
    statistics = torch.zeros(len(STATISTICS), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for observed, target, reset in zip(page["inputs"], page["targets"], page["resets"]):
            readable = reference.valid & ~reset[None]
            logits, hidden, attention, read = model(observed, reference.buffer, readable, reference.ages)
            ce = F.cross_entropy(logits.float(), target)
            total = total + ce
            statistics[0] += ce.detach()
            statistics[1:] += _diagnostics(attention.detach(), readable, read.detach(), _constant_read(model))
            reference.advance(hidden, readable)
        (total / steps).backward()
    return statistics


def _as_shifted(state):
    """Reorder a ring into shift layout by age, for comparison with the reference."""
    order = torch.argsort(state.ages)
    return state.buffer[order], state.valid[order]


def _page(tokens, steps):
    inputs = tokens[:steps]
    return {"inputs": inputs, "targets": tokens[1:steps + 1], "resets": inputs == BOS}


def _stream(config, updates):
    length = updates * config.stream_steps + 1
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
    reference = StationaryFFNModel(**config.model_config).cuda()
    reference.load_state_dict(model.state_dict())
    optimizers = [build_optimizer(item, config) for item in (model, reference)]
    # External gradient buffers survive zeroing and multiple optimizer pages.
    for item in (model, reference):
        for parameter in item.parameters():
            parameter.grad = torch.zeros_like(parameter)
    return model, reference, optimizers


def test_optimizer_decays_matrices_but_not_biases_or_gains():
    config = _config()
    model = _model(config)
    optimizer = build_optimizer(model, config)
    decayed = {id(p) for p in optimizer.param_groups[0]["params"]}
    assert optimizer.param_groups[0]["weight_decay"] == config.weight_decay
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    assert id(model.read.value.weight) in decayed and id(model.embed.weight) in decayed
    for parameter in (model.read.null_bias, model.read.value.bias, model.final_norm.gains):
        assert id(parameter) not in decayed
    normed = _model(_config(read_entry="latent_norm"))
    normed_optimizer = build_optimizer(normed, config)
    normed_decayed = {id(p) for p in normed_optimizer.param_groups[0]["params"]}
    plain = {id(p) for p in normed_optimizer.param_groups[1]["params"]}
    assert id(normed.read.age_bias) in plain and id(normed.latent_norm.gains) in plain
    assert id(normed.read.query.weight) in normed_decayed and id(normed.read.key.weight) in normed_decayed
    assert len(normed_decayed) + len(plain) == len(list(normed.parameters()))
    assert len(decayed) + len(optimizer.param_groups[1]["params"]) == len(list(model.parameters()))


@pytest.mark.parametrize("entry", ["value_map", "latent_norm"])
def test_compiled_ring_pages_match_shifted_eager_reference_across_optimizer_updates(entry):
    config = _config(read_entry=entry)
    model, reference_model, optimizers = _prepare(config, 71)
    engine = Engine(model, config)
    state = model.initial_state(LANES, "cuda")
    reference = ShiftedReference(config.buffer_slots, LANES, config.model_dim)
    stream = _stream(config, 3)
    for update in range(3):
        offset = update * config.stream_steps
        page = _page(stream[offset:], config.stream_steps)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=False)
        actual_stats = engine.backward_page(page, state)
        reference_stats = _reference_page(reference_model, page, reference)
        assert actual_stats.shape == (len(STATISTICS),) and not actual_stats.requires_grad
        assert engine.statistics == STATISTICS
        # latent_norm has no null slot, so its null mass is exactly zero on readable lanes.
        expected_nonzero = len(STATISTICS) - (entry == "latent_norm")
        assert torch.count_nonzero(actual_stats) == expected_nonzero
        if entry == "latent_norm":
            assert actual_stats[STATISTICS.index("null_mass")] == 0
        _assert_vector_close(actual_stats, reference_stats, label=f"update {update} statistics")
        buffer, valid = _as_shifted(state)
        _assert_vector_close(buffer, reference.buffer, label=f"update {update} buffer")
        assert torch.equal(valid, reference.valid)
        assert not state.buffer.requires_grad and state.buffer.grad_fn is None
        for (name, actual), expected in zip(model.named_parameters(), reference_model.parameters()):
            assert actual.grad is not None and expected.grad is not None, name
            _assert_vector_close(actual.grad, expected.grad, label=f"update {update} {name}")
        assert torch.count_nonzero(model.read.query.weight.grad) > 0
        assert torch.count_nonzero(model.read.key.weight.grad) > 0
        assert torch.count_nonzero((model.read.null_bias if entry == "value_map" else model.read.age_bias).grad) > 0
        # Validity follows the resets of ``_stream`` (K=3, four ticks per page):
        # lane 0 resets at tick 2 (ages 1-2 valid), lane 3 at tick 0 (all valid),
        # lane 2 only at page 0 tick 3 (one valid, then all), lane 1 at tick 0
        # of pages 1+ (all valid either way).
        assert torch.equal(valid[:, 0], torch.tensor([True, True, False], device="cuda"))
        assert torch.equal(valid[:, 2], torch.tensor([True, update > 0, update > 0], device="cuda"))
        assert torch.all(valid[:, 1]) and torch.all(valid[:, 3])
        for optimizer in optimizers:
            optimizer.step()
        # Adam magnifies near-zero BF16 sign noise, so two independently
        # optimized models drift apart within a page or two. Resync the
        # reference to the compiled model's parameters after every step, so
        # each page compares gradients and state at identical weights.
        reference_model.load_state_dict(model.state_dict())


def test_splitting_pages_without_update_preserves_state_and_gradients():
    config = _config()
    torch.set_num_threads(4)
    torch.manual_seed(89)
    whole_model = _model(config)
    split_model = StationaryFFNModel(**config.model_config).cuda()
    split_model.load_state_dict(whole_model.state_dict())
    whole_engine = Engine(whole_model, config)
    split_engine = Engine(split_model, replace(config, stream_steps=2))
    state = whole_model.initial_state(LANES, "cuda")
    state.buffer.normal_()
    state.valid[:2] = True
    split_state = split_model.initial_state(LANES, "cuda")
    split_state.load_state_dict(state.state_dict())
    tokens = torch.tensor([[3, BOS, 5, 6], [7, 8, 9, 10], [11, BOS, 13, 14],
                           [15, 16, BOS, 18], [19, 20, 21, BOS], [22, 23, 24, 25],
                           [26, 27, 28, 29], [30, 31, 2, 3]], device="cuda")
    for model in (whole_model, split_model):
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
    whole_stats = whole_engine.backward_page(_page(tokens, 4), state)
    first_stats = split_engine.backward_page(_page(tokens, 2), split_state).clone()
    second_stats = split_engine.backward_page(_page(tokens[2:], 2), split_state)
    _assert_vector_close(first_stats + second_stats, whole_stats, label="page split statistics")
    _assert_vector_close(split_state.buffer, state.buffer, label="page split buffer")
    assert torch.equal(split_state.valid, state.valid) and split_state.newest == state.newest
    for (name, actual), expected in zip(split_model.named_parameters(), whole_model.parameters()):
        # Two two-tick page means sum to twice the four-tick page mean.
        _assert_vector_close(actual.grad / 2, expected.grad, label=f"page split {name}")
    assert not split_state.buffer.requires_grad and split_state.buffer.grad_fn is None


def test_evaluate_reuses_its_ring_and_matches_eager_live_and_empty_codelengths():
    config = _config()
    torch.set_num_threads(4)
    torch.manual_seed(97)
    model = _model(config)
    engine = Engine(model, config)
    stream = _stream(config, 2)
    page = _page(stream, 2 * config.stream_steps)
    # Byte tables: one byte per target token, a leading space on odd tokens,
    # and boundary tokens that cancel it, so the denominator is non-trivial.
    ids = torch.arange(config.vocab_size, device="cuda")
    byte_luts = (torch.ones(config.vocab_size, device="cuda", dtype=torch.int16),
                 (ids % 2 == 1), (ids % 5 == 0))
    first = evaluate(engine, page, byte_luts)
    ring = engine.validation_state(LANES, "cuda")
    second = evaluate(engine, page, byte_luts)
    assert engine.validation_state(LANES, "cuda") is ring
    assert first == second, "the validation ring is not reset between calls"
    # Eager references: the live stream through the shift reference, and the
    # context-free model (nothing readable) for the always-empty arm.
    model.eval()
    reference = ShiftedReference(config.buffer_slots, LANES, config.model_dim)
    empty = torch.zeros_like(reference.buffer)
    nothing = torch.zeros_like(reference.valid)
    live, context_free, byte_count = 0.0, 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for observed, target, reset in zip(page["inputs"], page["targets"], page["resets"]):
            readable = reference.valid & ~reset[None]
            logits, hidden, _, _ = model(observed, reference.buffer, readable, reference.ages)
            live += F.cross_entropy(logits.float(), target, reduction="sum").item()
            logits, _, _, _ = model(observed, empty, nothing, reference.ages)
            context_free += F.cross_entropy(logits.float(), target, reduction="sum").item()
            reference.advance(hidden, readable)
            byte_count += int((byte_luts[0][target] + (byte_luts[1][target] & ~byte_luts[2][observed]).to(torch.int16)).sum())
    denominator = math.log(2) * byte_count
    assert first["val_bytes"] == byte_count and first["val_tokens"] == page["targets"].numel()
    assert abs(first["val_bpb"] - live / denominator) <= 2e-2 * live / denominator
    assert abs(first["val_bpb_reset_latent"] - context_free / denominator) <= 2e-2 * context_free / denominator
    assert abs(first["val_memory_gain_bpb"] - (context_free - live) / denominator) <= 2e-2 * context_free / denominator
    assert first["val_memory_gain_bpb"] != 0.0
    model.train()
