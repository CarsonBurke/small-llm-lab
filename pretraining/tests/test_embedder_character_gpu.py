"""Behavioral contracts for compressed-event character models; run only via mlq."""

import itertools

import pytest
import torch

from pretraining.nanogpt_mini import character_generation
from pretraining.nanogpt_mini.embedder_generation import (
    accept_id,
    generate,
    initial_state,
    next_logits,
)
from pretraining.nanogpt_mini.nanogpt_mini_character_model import (
    CharacterConfig,
    CharacterGPT,
)
from pretraining.nanogpt_mini.nanogpt_mini_embedder_model import (
    EmbedderConfig,
    EmbedderGPT,
    gate_surrogate,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
VOCAB_SIZE = 19


@pytest.fixture(autouse=True)
def isolated_compile_cache():
    # Different scenarios intentionally exercise different shapes/grad modes;
    # they must not share a test-induced Dynamo recompilation limit.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def make_model(mode="learned", *, stride=4, emission_cost=0.02, trained=True):
    torch.manual_seed(913)
    model = EmbedderGPT(
        EmbedderConfig(
            vocab_size=VOCAB_SIZE,
            num_layers=2,
            model_dim=128,
            embedder_dim=32,
            gate_mode=mode,
            fixed_stride=stride,
            emission_cost=emission_cost,
        )
    ).cuda()
    if trained:
        # Nonzero residual/output weights exercise actual attention, recurrence
        # and readout rather than the uniform head / identity blocks at init.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("proj.weight") or name == "head.weight":
                    parameter.normal_(std=0.025)
    model.compile_components()
    return model.eval()


@pytest.fixture
def model():
    return make_model()


def ids(length, batch=1):
    values = torch.arange(batch * length, device="cuda").reshape(batch, length)
    return (values.square() + 3 * values + 7) % VOCAB_SIZE


def test_normal_microbatch_compiled_backward():
    torch.manual_seed(1337)
    model = EmbedderGPT(EmbedderConfig(vocab_size=3099)).cuda().train()
    model.compile_components()
    source = torch.randint(3099, (64, 1024), device="cuda")
    loss, stats = model(source)
    loss.backward()
    assert bool(torch.isfinite(stats["rate_nats"]))
    assert model.head.weight.grad is not None
    assert bool(torch.count_nonzero(model.head.weight.grad))
    for parameter in model.parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())


def set_constant_gate(model, emit):
    with torch.no_grad():
        model.gate.weight.zero_()
        model.gate.bias.fill_(40 if emit else -40)


def objective(model, source, result):
    return model.finish_objective(
        source,
        result["logits"],
        result["gate_logits"],
        result["routes"],
        result["baseline"],
        result["packed_capacity"],
    )


def assert_bf16_close(actual, expected):
    # Scan and serial affine composition, and dense/cached SDPA, have different
    # BF16 reduction orders. This is numerical parity, not bitwise equivalence.
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.045)


@torch.compile(dynamic=True, fullgraph=True)
def compact_reference(model, local, event_indices, latest):
    # Independently build a single unpadded event stream and hold its states.
    events = local.index_select(1, event_indices)
    global_states = model.prior(events)
    return model.readout(local, global_states.index_select(1, latest))


def test_fixed_and_learned_are_matched_initializations():
    fixed = make_model("fixed", trained=False)
    learned = make_model("learned", trained=False)
    for fixed_parameter, learned_parameter in zip(
        fixed.parameters(), learned.parameters()
    ):
        torch.testing.assert_close(fixed_parameter, learned_parameter, rtol=0, atol=0)
    # Force the learned decisions to the same all-emission policy as stride one;
    # the entire likelihood must then match, not merely the parameter layout.
    all_fixed = make_model("fixed", stride=1)
    set_constant_gate(learned, True)
    learned.load_state_dict(all_fixed.state_dict())
    set_constant_gate(learned, True)
    with torch.no_grad():
        left = all_fixed.predict(ids(17))
        right = learned.predict(ids(17))
    torch.testing.assert_close(left["logits"], right["logits"], rtol=0, atol=0)
    assert bool(left["routes"].all()) and bool(right["routes"].all())


def test_predictions_and_decisions_never_see_current_or_future_targets(model):
    source = ids(65, batch=2)
    changed = source.clone()
    changed[:, 18:] = (changed[:, 18:] + 5) % VOCAB_SIZE
    with torch.no_grad():
        before, after = model.predict(source), model.predict(changed)
    # Decision 17 consumes character 17. Prediction 18 uses that decision but
    # cannot depend on target 18; source18 first affects prediction19.
    torch.testing.assert_close(
        before["local_states"][:, :19], after["local_states"][:, :19], rtol=0, atol=0
    )
    torch.testing.assert_close(
        before["gate_logits"][:, :18], after["gate_logits"][:, :18], rtol=0, atol=0
    )
    assert torch.equal(before["routes"][:, :18], after["routes"][:, :18])
    assert_bf16_close(before["logits"][:, :19], after["logits"][:, :19])
    assert bool((before["logits"][:, 19:] != after["logits"][:, 19:]).any())


def test_boundary_consumes_current_character_before_predicting_next(model):
    source = ids(21)
    none = torch.zeros((1, 20), dtype=torch.bool, device="cuda")
    boundary = none.clone()
    boundary[:, 2] = True
    with torch.no_grad():
        before = model.predict(source, none)
        after = model.predict(source, boundary)
        local = after["local_states"]
        reference = compact_reference(
            model,
            local,
            torch.tensor([0, 3], device="cuda"),
            torch.tensor([0] * 3 + [1] * 18, device="cuda"),
        )
    assert_bf16_close(after["logits"], reference)
    assert_bf16_close(before["logits"][:, :3], after["logits"][:, :3])
    assert bool((before["logits"][:, 3] != after["logits"][:, 3]).any())
    # An event after c2 carries c2, not c1 and not a future segment's end.
    changed = source.clone()
    changed[:, 2] = (changed[:, 2] + 1) % VOCAB_SIZE
    with torch.no_grad():
        changed_result = model.predict(changed, boundary)
    torch.testing.assert_close(
        local[:, :3], changed_result["local_states"][:, :3], rtol=0, atol=0
    )
    assert bool((local[:, 3] != changed_result["local_states"][:, 3]).any())


@pytest.mark.parametrize("stride", [1, 4, 97])
def test_fixed_boundaries_and_padding_count_only_emitted_summaries(stride):
    fixed = make_model("fixed", stride=stride)
    source = ids(65, batch=2)
    with torch.no_grad():
        result = fixed.predict(source)
        _, stats = objective(fixed, source, result)
    expected = torch.arange(1, 65, device="cuda") % stride == 0
    assert torch.equal(result["routes"], expected.expand(2, -1))
    events = 2 * (64 // stride)
    assert stats["emitted_events"].item() == events
    assert stats["useful_global_positions"].item() == events + 2
    assert stats["padded_global_positions"].item() == 2 * result["packed_capacity"]
    assert stats["encoder_positions"].item() == 128
    assert stats["gate_positions"].item() == 128
    assert stats["scan_compositions"].item() == 2 * sum(64 - 2**i for i in range(6))
    assert stats["compute_nats"].item() == pytest.approx(0.02 * (events + 2))
    if stride > 64:
        assert stats["padded_global_positions"].item() == 2


def test_ragged_rows_padding_and_other_sequences_do_not_change_real_context(model):
    source = ids(65, batch=2)
    routes = torch.zeros((2, 64), dtype=torch.bool, device="cuda")
    routes[1, [1, 6, 9, 17, 18, 31, 43, 50, 58, 63]] = True
    with torch.no_grad():
        packed = model.predict(source, routes)
        _, stats = objective(model, source, packed)
        for row in range(2):
            alone = model.predict(source[row : row + 1], routes[row : row + 1])
            assert_bf16_close(packed["logits"][row : row + 1], alone["logits"])
            event_indices = torch.cat(
                (
                    torch.zeros(1, device="cuda", dtype=torch.long),
                    routes[row].nonzero().flatten() + 1,
                )
            )
            latest = torch.cat(
                (
                    torch.zeros(1, device="cuda", dtype=torch.long),
                    routes[row].long().cumsum(0),
                )
            )
            reference = compact_reference(
                model,
                packed["local_states"][row : row + 1],
                event_indices,
                latest,
            )
            assert_bf16_close(packed["logits"][row : row + 1], reference)
    assert stats["useful_global_positions"].item() == 12
    assert stats["padded_global_positions"].item() == 32
    assert packed["packed_capacity"] == 16


@pytest.mark.parametrize("emit", [False, True])
def test_learned_gate_all_or_none_without_compulsory_span_refresh(model, emit):
    set_constant_gate(model, emit)
    source = ids(257)
    with torch.no_grad():
        result = model.predict(source)
        _, stats = objective(model, source, result)
    assert bool((result["routes"] == emit).all())
    assert stats["emitted_events"].item() == (256 if emit else 0)
    assert stats["useful_global_positions"].item() == (257 if emit else 1)
    assert stats["padded_global_positions"].item() == (257 if emit else 1)
    # Continuing without events must still use current character information.
    assert bool((result["logits"][:, 1:] != result["logits"][:, :-1]).any())


@pytest.mark.parametrize("length", [1, 37])
def test_all_characters_are_scored_and_auxiliaries_are_not_bpb(model, length):
    source = ids(length, batch=2)
    with torch.no_grad():
        result = model.predict(source)
        loss, stats = model(source)
        no_codec, same_stats = model(source, codec_weight=0.0)
    logits = result["logits"]
    expected = (
        logits.logsumexp(-1) - logits.gather(-1, source[..., None]).squeeze(-1)
    ).sum()
    torch.testing.assert_close(stats["rate_nats"], expected)
    torch.testing.assert_close(
        loss,
        expected
        + stats["compute_nats"]
        + model.config.baseline_loss_weight * stats["baseline_loss"],
    )
    torch.testing.assert_close(no_codec, loss, rtol=0, atol=0)
    torch.testing.assert_close(
        same_stats["rate_nats"], stats["rate_nats"], rtol=0, atol=0
    )
    assert logits.shape == (2, length, VOCAB_SIZE)
    assert set(stats) == set(model.sum_stat_names + model.mean_stat_names)
    assert all(value.ndim == 0 and not value.requires_grad for value in stats.values())
    torch.testing.assert_close(
        logits.softmax(-1).sum(-1), torch.ones_like(source, dtype=torch.float32)
    )
    if length == 1:
        assert stats["emitted_events"].item() == 0
        assert stats["encoder_positions"].item() == 0
        assert stats["gate_entropy"].item() == 0


def test_critic_fits_mean_future_loss_without_representation_gradients(model):
    source = ids(37, batch=2)
    result = model.predict(source)
    nll = result["logits"].logsumexp(-1) - result["logits"].gather(
        -1, source[..., None]
    ).squeeze(-1)
    # Independent explicit suffix means: sum-vs-mean or off-by-one critic
    # targets would scale this loss incorrectly and fail the comparison.
    targets = torch.stack([nll[:, index:].mean(-1) for index in range(1, 37)], dim=1)
    expected = (result["baseline"] - targets.detach()).square().sum()
    _, stats = objective(model, source, result)
    torch.testing.assert_close(stats["baseline_loss"], expected.detach())
    embedding_grad, transition_grad, critic_grad = torch.autograd.grad(
        expected,
        (model.table.weight, model.encoder.transition.weight, model.baseline.weight),
        allow_unused=True,
    )
    # AOTAutograd can materialize zeros for disconnected compiled outputs.
    # The contract is no representation update, not Python's gradient sentinel.
    for gradient in (embedding_grad, transition_grad):
        assert gradient is None or not bool(torch.count_nonzero(gradient))
    assert critic_grad is not None and bool((critic_grad != 0).any())


def test_final_target_is_not_an_encoder_input(model):
    source = torch.arange(VOCAB_SIZE, device="cuda").unsqueeze(0)
    loss, _ = model(source)
    gradient = torch.autograd.grad(loss, model.table.weight)[0]
    assert bool((gradient[:-1] != 0).any())
    assert bool((gradient[-1] == 0).all())


@pytest.mark.parametrize("emit", [False, True])
def test_no_missing_gradients_and_likelihood_credit_reaches_gate_without_emission_charge(
    emit,
):
    model = make_model(emission_cost=0.0).train()
    set_constant_gate(model, emit)
    loss, stats = model(ids(33, batch=2))
    loss.backward()
    assert stats["emitted_events"].item() == (64 if emit else 0)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
    for parameter in (
        model.table.weight,
        model.encoder.transition.weight,
        model.prior.input.weight,
        model.prior.blocks[0].mlp.proj.weight,
        model.head.weight,
    ):
        assert bool((parameter.grad != 0).any())
    # -40 retains tiny but representable Bernoulli credit; +40 rounds p to 1,
    # where the exact score derivative is legitimately zero.
    if not emit:
        assert bool((model.gate.bias.grad != 0).any())


def test_single_character_backward_keeps_idle_parameter_gradients(model):
    model.train()
    loss, _ = model(ids(1, batch=2))
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name


def test_single_route_estimator_matches_exact_expected_likelihood_gradient():
    surrogate = torch.compile(gate_surrogate, dynamic=True, fullgraph=True)
    logits = torch.tensor([[0.2, -0.4]], device="cuda", requires_grad=True)
    baseline = torch.tensor([[3.2, 4.4]], device="cuda", requires_grad=True)
    probability = logits.sigmoid()
    exact = logits.new_zeros(())
    estimated = torch.zeros_like(logits)
    for first, second in itertools.product((False, True), repeat=2):
        route = torch.tensor([[first, second]], device="cuda")
        # Loss0 is action-independent; loss1 depends on action0, loss2 on both.
        nll = torch.tensor(
            [
                [
                    90.0,
                    3.0 - 0.7 * first,
                    4.0 + 0.4 * first - 1.2 * second + 0.3 * first * second,
                ]
            ],
            device="cuda",
            requires_grad=True,
        )
        mass = torch.where(route, probability, 1 - probability).prod()
        exact = exact + mass * nll.detach().sum()
        policy, _ = surrogate(logits, route, nll, baseline)
        gradient, reward_gradient, baseline_gradient = torch.autograd.grad(
            policy,
            (logits, nll, baseline),
            allow_unused=True,
        )
        estimated += mass.detach() * gradient
        assert reward_gradient is None or bool((reward_gradient == 0).all())
        assert baseline_gradient is None or bool((baseline_gradient == 0).all())
    expected = torch.autograd.grad(exact, logits)[0]
    torch.testing.assert_close(estimated, expected, rtol=2e-5, atol=2e-6)


def test_gate_credit_excludes_current_and_earlier_targets():
    surrogate = torch.compile(gate_surrogate, dynamic=True, fullgraph=True)
    logits = torch.tensor([[0.2, -0.4, 0.7]], device="cuda", requires_grad=True)
    route = torch.tensor([[True, False, True]], device="cuda")
    baseline = torch.zeros((1, 3), device="cuda")
    nll = torch.tensor([[9.0, 1.0, 2.0, 4.0]], device="cuda")
    original = torch.autograd.grad(surrogate(logits, route, nll, baseline)[0], logits)[
        0
    ]
    changed = nll.clone()
    changed[:, :2] += 100
    updated = torch.autograd.grad(
        surrogate(logits, route, changed, baseline)[0], logits
    )[0]
    torch.testing.assert_close(original[:, 1:], updated[:, 1:], rtol=0, atol=0)
    assert original[0, 0].item() != updated[0, 0].item()


@pytest.mark.parametrize("policy", ["fixed", "none", "all", "mixed"])
def test_event_cached_stream_matches_dense_likelihood_and_position_accounting(policy):
    model = make_model("fixed" if policy == "fixed" else "learned")
    if policy in ("none", "all"):
        set_constant_gate(model, policy == "all")
    source = ids(49)
    if policy == "mixed":
        # A data-dependent gate with a margin avoids meaningless sign flips from
        # BF16 scan-vs-sequential rounding. It still depends on real local state.
        with torch.no_grad():
            local, _, _ = model.encode(source)
            feature = local[0, 1:, 0].float()
            ordered = feature.sort().values
            gaps = ordered[1:] - ordered[:-1]
            middle = gaps[8:-8].argmax().item() + 8
            threshold = (ordered[middle] + ordered[middle + 1]).item() / 2
            model.gate.weight.zero_()
            model.gate.weight[0, 0] = 8
            model.gate.bias.fill_(-8 * threshold)
    with torch.no_grad():
        dense = model.predict(source)
        state = initial_state(model, source.shape[1] - 1)
        cached = [next_logits(model, state)]
        decisions = []
        for identity in source[0, :-1].tolist():
            decisions.append(accept_id(model, state, identity))
            cached.append(next_logits(model, state))
        actual = torch.stack(cached, dim=1)
    assert_bf16_close(actual, dense["logits"])
    assert decisions == dense["routes"][0].tolist()
    if policy == "mixed":
        assert any(decisions) and not all(decisions)
    events = sum(decisions)
    assert state.emitted_events == events
    assert state.backbone_calls == state.backbone_positions == events + 1
    assert state.encoder_positions == state.gate_positions == state.position == 48
    assert state.readout_positions == 49
    assert state.emission_positions == [
        index for index, emit in enumerate(decisions) if emit
    ]
    assert (
        state.accumulator.dtype
        == state.local.dtype
        == state.held_global.dtype
        == torch.bfloat16
    )
    # Initialized event-cache prefixes are finite and there is never a catch-up
    # append of non-emitting raw characters.
    assert all(
        bool(torch.isfinite(cache[:, :, : events + 1]).all())
        for cache in (*state.keys, *state.values)
    )
    if policy == "none":
        assert state.backbone_positions == 1


def test_generation_uses_observed_alphabet_and_reports_real_work(model):
    set_constant_gate(model, False)
    prompt = [1, 5, 8, 2]
    result = generate(model, prompt, 19, temperature=0.8, seed=417)
    repeated = generate(model, prompt, 19, temperature=0.8, seed=417)
    assert result["generated_ids"] == repeated["generated_ids"]
    assert len(result["generated_ids"]) == 19
    assert all(0 <= value < VOCAB_SIZE for value in result["generated_ids"])
    assert result["stop_reason"] == "output_limit"
    assert result["emitted_events"] == 0
    assert result["useful_global_positions"] == result["padded_global_positions"] == 1
    assert result["backbone_calls"] == 1
    assert (
        result["accepted_characters"]
        == result["encoder_positions"]
        == result["gate_positions"]
        == 22
    )
    assert result["readout_positions"] == 19
    assert result["event_kv_capacity"] == 23
    assert result["event_kv_bytes"] == 2 * 2 * 23 * 128 * 2
    assert result["elapsed_seconds"] > 0
    assert result["timing_includes_prompt_and_compilation"] is True


def test_greedy_generation_matches_teacher_forced_predictions_at_every_output():
    model = make_model("fixed", stride=4)
    result = generate(model, [1, 5, 8], 9, temperature=0)
    source = torch.tensor(
        [result["prompt_ids"] + result["generated_ids"]], device="cuda"
    )
    with torch.no_grad():
        dense = model.predict(source)
    assert dense["logits"][0, 3:].argmax(-1).tolist() == result["generated_ids"]
    assert result["emission_positions"] == [3, 7]
    assert result["backbone_positions"] == result["emitted_events"] + 1 == 3


def make_character_model():
    torch.manual_seed(814)
    model = (
        CharacterGPT(
            CharacterConfig(
                vocab_size=VOCAB_SIZE,
                num_layers=2,
                model_dim=128,
            )
        )
        .cuda()
        .eval()
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("proj.weight"):
                parameter.normal_(std=0.025)
        model.prior.proj.bias.copy_(
            torch.linspace(-0.5, 0.5, VOCAB_SIZE, device="cuda")
        )
    model.compile_components()
    return model


@pytest.mark.parametrize("length", [1, 65])
def test_character_control_cached_likelihood_matches_its_actual_dense_model(length):
    model = make_character_model()
    source = ids(length)
    with torch.inference_mode():
        dense = model.logits(source)
        state = character_generation.initial_state(model, length - 1)
        cached = [character_generation.next_logits(model, state)]
        for identity in source[0, :-1].tolist():
            character_generation.accept_id(model, state, identity)
            cached.append(character_generation.next_logits(model, state))
    actual = torch.stack(cached, dim=1)
    assert_bf16_close(actual, dense)
    assert state.backbone_positions == state.backbone_calls == length
    assert state.position == length - 1
    assert state.readout_positions == length
    assert state.context.dtype == torch.bfloat16
    assert all(cache.dtype == torch.bfloat16 for cache in (*state.keys, *state.values))


@pytest.mark.parametrize("kind", ["baseline", "embedder"])
def test_zero_output_performs_no_prefill_or_neural_work(kind):
    model = make_character_model() if kind == "baseline" else make_model()
    runtime_generate = character_generation.generate if kind == "baseline" else generate
    torch.cuda.synchronize()  # Exclude model initialization from this work trace.
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as trace:
        result = runtime_generate(model, [1, 2, 3], 0)
    assert result["generated_ids"] == []
    assert result["accepted_characters"] == 0
    assert (
        result["backbone_positions"]
        == result["encoder_positions"]
        == result["readout_positions"]
        == 0
    )
    assert result["prefill_seconds"] == result["decode_seconds"] == 0
    assert not any(
        event.device_type == torch.autograd.DeviceType.CUDA for event in trace.events()
    )


@pytest.mark.parametrize("kind", ["baseline", "embedder"])
def test_one_output_scores_bos_without_consuming_the_result(kind):
    model = make_character_model() if kind == "baseline" else make_model()
    runtime_generate = character_generation.generate if kind == "baseline" else generate
    result = runtime_generate(model, [], 1, temperature=0.8, seed=1337)
    assert len(result["generated_ids"]) == 1
    assert result["accepted_characters"] == result["encoder_positions"] == 0
    assert (
        result["backbone_positions"]
        == result["backbone_calls"]
        == result["readout_positions"]
        == 1
    )


def test_transport_is_exact_and_independent_of_neural_parameters(model):
    source = torch.arange(VOCAB_SIZE, device="cuda").unsqueeze(0)
    encoded = model.export(source)
    assert model.recover(encoded["latents"], encoded["residual"]) == list(
        range(VOCAB_SIZE)
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float("nan"))
    assert model.export(source) == encoded
    assert model.recover(encoded["latents"], encoded["residual"]) == list(
        range(VOCAB_SIZE)
    )
    assert model.recover([], []) == []
    assert model.export(source[:, :0]) == {"latents": [], "residual": []}
    width, residual_width = model.transport_widths
    assert residual_width == 0
    with pytest.raises(ValueError):
        model.recover([[int(bit) for bit in f"{VOCAB_SIZE:0{width}b}"]], [[]])
    with pytest.raises(ValueError):
        model.recover([[0] * width], [[1]])
    with pytest.raises(ValueError):
        model.export(torch.tensor([[VOCAB_SIZE]], device="cuda"))
