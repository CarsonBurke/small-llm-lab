"""Learned causal character routing contracts; CUDA execution only through mlq."""

import itertools

import pytest
import torch
import torch.nn.functional as F

from pretraining.nanogpt_mini import adaptive_generation as generation
from pretraining.nanogpt_mini.nanogpt_mini_adaptive_model import (
    AdaptiveConfig,
    AdaptiveGPT,
    routing_surrogate,
)
from pretraining.nanogpt_mini.native_bits_wire import (
    pack_packet,
    payload_bit_count,
    unpack_packet,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
VOCAB_SIZE = 29
BITS = 5


@pytest.fixture(autouse=True)
def isolated_compile_cache():
    # Cases vary rollout count, padding bucket, singleton dimensions and grad
    # mode. They are independent scenarios, not one production graph cache.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(923)
    result = AdaptiveGPT(
        AdaptiveConfig(
            vocab_size=VOCAB_SIZE,
            code_bits=BITS,
            num_layers=6,
            model_dim=512,
            local_layers=2,
            local_dim=128,
            prefix_width=128,
            refresh_cost=0.07,
        )
    ).cuda()
    with torch.no_grad():
        # Otherwise zero residual/output initialization conceals attention leaks,
        # wrong event positions, and disconnected local/global gradient paths.
        for trunk in (result.local, result.prior):
            for block in trunk.blocks:
                block.attn.proj.weight.normal_(std=0.02)
                block.mlp.proj.weight.normal_(std=0.02)
        result.head.output.weight.normal_(std=0.09)
        result.head.position.weight.uniform_(0.25, 0.5)
    result.compile_components()
    result.eval()
    yield result
    result.zero_grad(set_to_none=True)
    torch.compiler.reset()


@pytest.fixture(scope="module")
def observed_model(model):
    # Keep production's complete packed graph intact. This separate diagnostic
    # leaves only tensor packing eager so a spy can see real event dimensions;
    # every neural call still executes compiled CUDA kernels.
    result = AdaptiveGPT(model.config).cuda()
    result.load_state_dict(model.state_dict())
    result.eval()
    result.encode_context = torch.compile(
        result.encode_context, dynamic=True, fullgraph=True
    )
    for component in (
        result.prior,
        result.head,
        result.local_output,
        result.readout_norm,
    ):
        component.forward = torch.compile(
            component.forward, dynamic=True, fullgraph=True
        )
    return result


@pytest.fixture(scope="module")
def reference_prior(model):
    return torch.compile(model.prior.forward, dynamic=True, fullgraph=True)


@pytest.fixture(scope="module")
def readout(model):
    def reference_readout(held, local, codes):
        context = model.readout_norm(held + model.local_output(local))
        logits = model.head(context, codes).float()
        logits = 15 * logits * (logits.square() + 225).rsqrt()
        nats = F.binary_cross_entropy_with_logits(logits, codes, reduction="none").sum(
            -1
        )
        return context, logits, nats

    return torch.compile(reference_readout, dynamic=True, fullgraph=True)


@pytest.fixture(scope="module")
def head_logits(model):
    def score(context, codes):
        logits = model.head(context, codes).float()
        return 15 * logits * (logits.square() + 225).rsqrt()

    return torch.compile(score, dynamic=True, fullgraph=True)


@pytest.fixture(scope="module")
def surrogate():
    return torch.compile(routing_surrogate, dynamic=True, fullgraph=True)


def _ids(length, batch=1):
    positions = torch.arange(batch * length, device="cuda").reshape(batch, length)
    return (positions.square() + 7 * positions + 3) % VOCAB_SIZE


def _routes(length, positions):
    routes = torch.zeros((1, length), device="cuda", dtype=torch.bool)
    routes[:, positions] = True
    return routes


def _reference_row(prior, readout, local, codes, routes):
    """No bucketing: run only actual events, then explicitly hold the latest one."""
    selected = torch.nonzero(routes[0], as_tuple=False).flatten()
    events = prior(local[:, selected])
    held = events[:, routes.long().cumsum(-1)[0] - 1]
    return readout(held, local, codes)


def _assert_bf16_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.04)


def test_current_and_future_characters_cannot_change_routing_or_context(model):
    ids = _ids(65, batch=2)
    changed = ids.clone()
    changed[:, 17:] = (changed[:, 17:] + 9) % VOCAB_SIZE
    routes = _routes(65, [0, 2, 7, 17, 23, 41, 64]).expand(2, -1)
    with torch.no_grad():
        before = model.predict(ids, routes=routes)
        after = model.predict(changed, routes=routes)
    for key in ("local_states", "router_logits"):
        torch.testing.assert_close(
            before[key][:, :18], after[key][:, :18], rtol=0, atol=0
        )
        assert bool((before[key][:, 18:] != after[key][:, 18:]).any())
    torch.testing.assert_close(
        before["logits"][:, :17], after["logits"][:, :17], rtol=0, atol=0
    )
    # At t=17, bit zero has no own-address prefix. Other bits may see earlier
    # bits of the changed address, but no bit may see a later character.
    torch.testing.assert_close(
        before["logits"][:, 17, 0], after["logits"][:, 17, 0], rtol=0, atol=0
    )
    assert bool((before["logits"][:, 18:] != after["logits"][:, 18:]).any())


@pytest.mark.parametrize(
    "positions",
    [[0], [0, 1, 5, 17, 33, 64], list(range(65))],
    ids=["continue-after-bos", "irregular-events", "refresh-every-character"],
)
def test_supplied_routes_match_real_compacted_event_history(
    model, observed_model, reference_prior, readout, positions, monkeypatch
):
    ids = _ids(65)
    routes = _routes(65, positions)
    executed_lengths = []
    compiled_prior = observed_model.prior.forward

    def observed_prior(features):
        executed_lengths.append(features.shape[1])
        return compiled_prior(features)

    with torch.no_grad():
        with monkeypatch.context() as patch:
            patch.setattr(observed_model.prior, "forward", observed_prior)
            actual = observed_model.predict(ids, routes=routes)
        codes = model.identity_bits(ids)
        _, expected_logits, expected_nats = _reference_row(
            reference_prior, readout, actual["local_states"], codes, routes
        )
        nats, logits, capacity = model.pack_and_score(
            actual["local_states"], codes, routes
        )
    expected_capacity = min(65, ((len(positions) + 31) // 32) * 32)
    # Observe actual global input length, not merely a reported compute counter.
    assert executed_lengths == [expected_capacity]
    assert capacity == actual["packed_capacity"] == expected_capacity
    torch.testing.assert_close(actual["routes"], routes, rtol=0, atol=0)
    _assert_bf16_close(actual["logits"], expected_logits)
    _assert_bf16_close(logits, expected_logits)
    _assert_bf16_close(nats, expected_nats)
    assert float(logits[:, 1:].std()) > 1e-3


def test_other_rows_and_larger_padding_bucket_cannot_change_a_sequence(model):
    ids = _ids(65)
    sparse = _routes(65, [0, 2, 9, 23, 41, 64])
    companion = _routes(65, list(range(0, 65, 2)))
    with torch.no_grad():
        alone = model.predict(ids, routes=sparse)
        batched = model.predict(
            torch.cat((ids, (ids + 13) % VOCAB_SIZE)),
            routes=torch.cat((sparse, companion)),
        )
        continue_only = model.predict(ids, routes=_routes(65, [0]))
    assert alone["packed_capacity"] == 32
    assert batched["packed_capacity"] == 64
    _assert_bf16_close(alone["logits"], batched["logits"][:1])
    assert bool((alone["logits"][:, 3:] != continue_only["logits"][:, 3:]).any())


def test_next_address_normalizes_over_full_domain_without_free_reserved_mass(model):
    # All candidates have the same nonempty known prefix and the same route.
    prefix = _ids(19)
    template = torch.cat(
        (prefix, torch.zeros((1, 1), device="cuda", dtype=torch.long)), -1
    )
    routes = _routes(20, [0, 3, 11, 19])
    candidates = torch.arange(1 << BITS, device="cuda")
    ids = template.expand(1 << BITS, -1).clone()
    ids[:, -1] = candidates
    with torch.no_grad():
        prediction = model.predict(template, routes=routes)
        codes = model.identity_bits(ids)
        nats, logits, _ = model.pack_and_score(
            prediction["local_states"].expand(1 << BITS, -1, -1),
            codes,
            routes.expand(1 << BITS, -1),
        )
    probabilities = (-nats[:, -1]).exp()
    torch.testing.assert_close(
        probabilities.sum(), torch.ones((), device="cuda"), rtol=2e-5, atol=2e-6
    )
    valid_mass = probabilities[:VOCAB_SIZE].sum()
    reserved_mass = probabilities[VOCAB_SIZE:].sum()
    assert 0 < float(valid_mass) < 1
    assert float(reserved_mass) > 0
    torch.testing.assert_close(valid_mass, 1 - reserved_mass, rtol=2e-5, atol=2e-6)
    # The head must use a strict within-address prefix, not current/future bits.
    for bit in range(BITS):
        grouped = logits[:, -1, bit].reshape(1 << bit, 1 << (BITS - bit))
        torch.testing.assert_close(
            grouped, grouped[:, :1].expand_as(grouped), rtol=0, atol=0
        )
        if bit:
            flipped_previous = candidates ^ (1 << (BITS - bit))
            assert bool((logits[:, -1, bit] != logits[flipped_previous, -1, bit]).any())
    assert float(logits[:, -1].std()) > 1e-3


@pytest.mark.parametrize("length", [1, 65])
def test_eval_charges_first_and_tail_addresses_and_actual_deterministic_compute(
    model, length
):
    ids = _ids(length, batch=2)
    with torch.no_grad():
        predicted = model.predict(ids)
        total, stats = model(ids)
        codes = model.identity_bits(ids)
        expected_rate = F.binary_cross_entropy_with_logits(
            predicted["logits"], codes, reduction="sum"
        )
        expected_compute = model.config.refresh_cost * predicted["routes"].sum()
    torch.testing.assert_close(stats["rate_nats"], expected_rate)
    torch.testing.assert_close(stats["compute_nats"], expected_compute)
    torch.testing.assert_close(total, expected_rate + expected_compute)
    torch.testing.assert_close(
        stats["refresh_rate"], predicted["routes"].float().mean()
    )


def test_routing_surrogate_credits_only_current_and_future_character_losses(surrogate):
    logits = torch.tensor([[0.3, 0.2, -0.7, 1.1]], device="cuda", requires_grad=True)
    routes = torch.tensor(
        [[[True, False, True, False]], [[True, True, False, True]]], device="cuda"
    )
    nll = torch.tensor(
        [[[31.0, 2.0, 7.0, 1.0]], [[11.0, 5.0, 3.0, 9.0]]],
        device="cuda",
        requires_grad=True,
    )
    policy, rms = surrogate(logits, routes, nll)
    torch.testing.assert_close(policy, torch.zeros((), device="cuda"), rtol=0, atol=0)
    torch.testing.assert_close(rms, torch.tensor(43.0, device="cuda").sqrt())
    gate_grad, reward_grad = torch.autograd.grad(
        policy, (logits, nll), allow_unused=True
    )
    # Paired suffix differences are -7, -4, -8. BOS is forced, not a sampled action.
    torch.testing.assert_close(
        gate_grad, torch.tensor([[0.0, 3.5, -2.0, 4.0]], device="cuda")
    )
    assert reward_grad is None or bool((reward_grad == 0).all())


def test_two_sample_rloo_matches_enumerated_expected_loss_and_analytic_cost(surrogate):
    actions = list(itertools.product((0, 1), repeat=2))
    action_tensor = torch.tensor(actions, device="cuda", dtype=torch.float32)
    losses = torch.tensor(
        [
            [7.0, 0.8 + 1.7 * a, 2.5 - 1.1 * a + 2.3 * b + 0.9 * a * b]
            for a, b in actions
        ],
        device="cuda",
    )
    pairs = list(itertools.product(range(len(actions)), repeat=2))
    left = torch.tensor([a for a, _ in pairs], device="cuda")
    right = torch.tensor([b for _, b in pairs], device="cuda")
    logits = torch.tensor([[0.4, -0.7, 0.9]], device="cuda", requires_grad=True)
    probability = logits[:, 1:].sigmoid()
    path_probability = torch.where(
        action_tensor.bool(), probability, 1 - probability
    ).prod(-1)
    pair_probability = (path_probability[left] * path_probability[right]).detach()
    masks = torch.cat(
        (torch.ones((4, 1), device="cuda", dtype=torch.bool), action_tensor.bool()), -1
    )
    routes = torch.stack((masks[left], masks[right]))
    weighted_losses = (
        torch.stack((losses[left], losses[right])) * pair_probability[None, :, None]
    ).requires_grad_()
    policy, _ = surrogate(logits.expand(len(pairs), -1), routes, weighted_losses)
    cost = 0.13 * (1 + logits[:, 1:].sigmoid().sum())
    actual_grad, reward_grad = torch.autograd.grad(
        policy + cost, (logits, weighted_losses), allow_unused=True
    )
    expected_logits = logits.detach().clone().requires_grad_()
    expected_probability = expected_logits[:, 1:].sigmoid()
    expected_path_probability = torch.where(
        action_tensor.bool(), expected_probability, 1 - expected_probability
    ).prod(-1)
    exact_objective = (expected_path_probability * losses.sum(-1)).sum()
    exact_objective = exact_objective + 0.13 * (1 + expected_probability.sum())
    (expected_grad,) = torch.autograd.grad(exact_objective, expected_logits)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(policy, torch.zeros_like(policy), rtol=0, atol=0)
    assert reward_grad is None or bool((reward_grad == 0).all())
    assert bool((actual_grad[:, 1:] != 0).all())
    assert actual_grad[0, 0].item() == 0


def test_training_backward_reaches_local_global_head_and_router(model, monkeypatch):
    model.zero_grad(set_to_none=True)
    model.train()
    try:
        torch.manual_seed(924)
        ids = _ids(65, batch=2)
        with torch.no_grad():
            deterministic = model.predict(ids)
            expected_routes = deterministic["router_logits"] >= 0
            expected_routes[:, 0] = True
            torch.testing.assert_close(
                deterministic["routes"], expected_routes, rtol=0, atol=0
            )
        # Capture the policy from the actual grad-enabled pass. Recomputing it
        # under no_grad changes BF16 fusion and is not an exact cost reference.
        captured = {}
        compiled_encode = model.encode_context

        def observe_policy(codes):
            local, logits = compiled_encode(codes)
            captured["logits"] = logits.detach().clone()
            return local, logits

        with monkeypatch.context() as patch:
            patch.setattr(model, "encode_context", observe_policy)
            total, stats = model(ids)
        expected_compute = model.config.refresh_cost * (
            ids.shape[0] + captured["logits"][:, 1:].sigmoid().sum()
        )
        torch.testing.assert_close(stats["compute_nats"], expected_compute)
        assert bool(torch.isfinite(total))
        torch.testing.assert_close(
            total.detach(), stats["rate_nats"] + stats["compute_nats"]
        )
        torch.testing.assert_close(
            stats["global_positions_per_character"], 2 * stats["refresh_rate"]
        )
        assert bool(
            stats["padded_global_positions_per_character"]
            >= stats["global_positions_per_character"]
        )
        total.backward()
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert bool(torch.isfinite(parameter.grad).all())
        paths = [
            model.local.input.weight,
            model.prior.input.weight,
            model.local_output.weight,
            model.router.weight,
            model.head.context.weight,
            model.head.prefix.weight,
            model.head.position.weight,
            model.head.output.weight,
        ]
        for trunk in (model.local, model.prior):
            paths.extend(block.attn.q.weight for block in trunk.blocks)
            paths.extend(block.mlp.fc.weight for block in trunk.blocks)
        for parameter in paths:
            assert bool((parameter.grad != 0).any())
    finally:
        model.eval()
        model.zero_grad(set_to_none=True)


def test_fixed_identity_transport_roundtrips_alphabet_without_neural_execution(
    model, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "fixed identity transport must not execute neural components"
        )

    for component in (model.local, model.prior, model.head):
        monkeypatch.setattr(component, "forward", forbidden)
    monkeypatch.setattr(model, "predict", forbidden)
    ids = torch.arange(VOCAB_SIZE, device="cuda").unsqueeze(0)
    exported = model.export(ids)
    expected = [
        [(symbol >> shift) & 1 for shift in range(BITS - 1, -1, -1)]
        for symbol in range(VOCAB_SIZE)
    ]
    assert model.transport_widths == (BITS, 0)
    assert exported == {"latents": expected, "residual": [[] for _ in expected]}
    assert model.recover(**exported) == list(range(VOCAB_SIZE))
    packet = pack_packet(
        **exported,
        checkpoint_sha256="a" * 64,
        source_sha256="b" * 64,
        latent_bits=BITS,
        identity_bits=0,
    )
    decoded = unpack_packet(packet)
    assert payload_bit_count(packet) == VOCAB_SIZE * BITS
    assert model.recover(decoded["latents"], decoded["residual"]) == list(
        range(VOCAB_SIZE)
    )
    assert model.export(ids[:, :0]) == {"latents": [], "residual": []}
    assert model.recover([], []) == []


def test_transport_rejects_reserved_addresses_and_malformed_streams(model):
    for address in range(VOCAB_SIZE, 1 << BITS):
        row = [(address >> shift) & 1 for shift in range(BITS - 1, -1, -1)]
        with pytest.raises(ValueError):
            model.recover([row], [[]])
    for row in (
        [0] * (BITS - 1),
        [0] * (BITS + 1),
        [2] * BITS,
        [-1] * BITS,
        [0.0] * BITS,
    ):
        with pytest.raises(ValueError):
            model.recover([row], [[]])
    for latents, residual in (([[0] * BITS], [[0]]), ([[0] * BITS], []), ([], [[]])):
        with pytest.raises(ValueError):
            model.recover(latents, residual)
    for invalid in (-1, VOCAB_SIZE):
        with pytest.raises(ValueError):
            model.export(torch.tensor([[invalid]], device="cuda"))
    with pytest.raises(ValueError):
        model.export(torch.tensor([[0.5]], device="cuda"))
    with pytest.raises(ValueError):
        model.export(torch.zeros((2, 3), device="cuda", dtype=torch.long))


@pytest.mark.parametrize("policy", ["mixed", "continue"])
def test_cached_stream_matches_parallel_context_logits_and_real_refresh_work(
    model, reference_prior, readout, head_logits, policy, monkeypatch
):
    ids = _ids(65)
    saved_weight = model.router.weight.detach().clone()
    saved_bias = model.router.bias.detach().clone()
    global_steps = []
    compiled_global_step = generation._global_step

    def observed_global_step(*args, **kwargs):
        global_steps.append(args[-1])
        return compiled_global_step(*args, **kwargs)

    try:
        with torch.no_grad():
            if policy == "continue":
                model.router.weight.zero_()
                model.router.bias.fill_(-4)
            else:
                # Place a learned linear threshold in a wide interior gap, away
                # from BF16 rounding ambiguity, while retaining both actions.
                initial = model.predict(ids)
                ordered = initial["router_logits"][0, 1:].sort().values
                first, last = len(ordered) // 4, 3 * len(ordered) // 4
                gaps = ordered[first + 1 : last + 1] - ordered[first:last]
                index = first + int(gaps.argmax().item())
                model.router.bias.sub_((ordered[index] + ordered[index + 1]) / 2)
            parallel = model.predict(ids)
            codes = model.identity_bits(ids)
            expected_context, _, _ = _reference_row(
                reference_prior,
                readout,
                parallel["local_states"],
                codes,
                parallel["routes"],
            )
            state = generation.initial_state(model, max_characters=ids.shape[1])
            contexts, logits, refreshes = [], [], []
            with monkeypatch.context() as patch:
                patch.setattr(generation, "_global_step", observed_global_step)
                for position in range(ids.shape[1]):
                    context, refresh = generation.next_context(model, state)
                    contexts.append(context.clone())
                    logits.append(head_logits(context, codes[:, position]).clone())
                    refreshes.append(refresh)
                    generation.accept_code(model, state, codes[:, position])
                    assert state.position == position + 1
                    assert state.global_updates == sum(refreshes)
            actual_routes = torch.tensor([refreshes], device="cuda", dtype=torch.bool)
            torch.testing.assert_close(
                actual_routes, parallel["routes"], rtol=0, atol=0
            )
            _assert_bf16_close(torch.stack(contexts, 1), expected_context)
            _assert_bf16_close(torch.stack(logits, 1), parallel["logits"])
            assert (
                len(global_steps)
                == state.global_updates
                == int(actual_routes.sum().item())
            )
            assert global_steps == list(range(len(global_steps)))
            assert state.local_positions == ids.shape[1]
            if policy == "continue":
                assert refreshes == [True] + [False] * (ids.shape[1] - 1)
                assert len(global_steps) == 1
            else:
                assert 1 < len(global_steps) < ids.shape[1]
                positions = torch.nonzero(actual_routes[0], as_tuple=False).flatten()
                assert torch.unique(positions.diff()).numel() > 1
    finally:
        with torch.no_grad():
            model.router.weight.copy_(saved_weight)
            model.router.bias.copy_(saved_bias)


def test_cached_generation_continues_without_a_chunk_cap_and_reports_reserved_codes(
    model,
):
    saved = [
        parameter.detach().clone()
        for parameter in (
            model.router.weight,
            model.router.bias,
            model.head.output.weight,
            model.head.output.bias,
        )
    ]
    try:
        with torch.no_grad():
            model.router.weight.zero_()
            model.router.bias.fill_(-4)
            model.head.output.weight.zero_()
            model.head.output.bias.fill_(-2)
        generated = generation.generate(
            model, [3, 9], max_new_characters=35, temperature=0
        )
        assert generated["prompt_ids"] == [3, 9]
        assert generated["generated_ids"] == [0] * 35
        assert generated["generated_codes"] == [[0] * BITS for _ in range(35)]
        assert generated["reserved_code"] is None
        assert generated["refresh_positions"] == [0]
        assert generated["global_updates"] == 1
        assert generated["local_positions"] == 37
        with torch.no_grad():
            model.head.output.bias.fill_(2)
        reserved = generation.generate(model, [], max_new_characters=5, temperature=0)
        assert reserved["generated_ids"] == []
        assert reserved["generated_codes"] == [[1] * BITS]
        assert reserved["reserved_code"] == (1 << BITS) - 1
        assert reserved["global_updates"] == reserved["local_positions"] == 1
    finally:
        with torch.no_grad():
            for parameter, original in zip(
                (
                    model.router.weight,
                    model.router.bias,
                    model.head.output.weight,
                    model.head.output.bias,
                ),
                saved,
            ):
                parameter.copy_(original)
