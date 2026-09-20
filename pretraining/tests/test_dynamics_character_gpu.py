"""Deployed latent dynamics and exact catch-up contracts; run CUDA only through mlq."""

import copy
import math
from contextlib import contextmanager
from itertools import pairwise

import pytest
import torch
import torch.nn.functional as F

from pretraining.nanogpt_mini import dynamics_generation as generation
from pretraining.nanogpt_mini.bit_density import sample_prefix_code
from pretraining.nanogpt_mini.nanogpt_mini_dynamics_bounded_model import (
    BoundedDynamicsGPT,
    BoundedResidualTransition,
)
from pretraining.nanogpt_mini.nanogpt_mini_dynamics_model import (
    DynamicsConfig,
    DynamicsGPT,
    ResidualTransition,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
VOCAB_SIZE = 29
BITS = 5


@pytest.fixture(autouse=True)
def isolated_compile_cache():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(925)
    result = DynamicsGPT(
        DynamicsConfig(
            vocab_size=VOCAB_SIZE,
            code_bits=BITS,
            num_layers=6,
            model_dim=512,
            prefix_width=128,
            dynamics_width=128,
            rollout_horizon=2,
            refresh_cost=0.07,
        )
    ).cuda()
    with torch.no_grad():
        # Zero residual/head initialization hides cache leaks, ignored history,
        # and broken auxiliary credit. Exercise nontrivial trained-like paths.
        for block in result.prior.blocks:
            block.attn.proj.weight.normal_(std=0.02)
            block.mlp.proj.weight.normal_(std=0.02)
        result.head.output.weight.normal_(std=0.09)
        result.head.position.weight.uniform_(0.25, 0.5)
        result.transition.output.weight.normal_(std=0.002)
        result.error_predictor.output.weight.normal_(std=0.04)
        result.error_predictor.output.bias.fill_(-2)
    result.compile_components()
    result.eval()
    yield result
    result.zero_grad(set_to_none=True)
    torch.compiler.reset()


def _ids(length, batch=1):
    positions = torch.arange(batch * length, device="cuda").reshape(batch, length)
    return (positions.square() + 7 * positions + 3) % VOCAB_SIZE


def _assert_bf16_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.04)


@contextmanager
def _gate_policy(model, refresh_age=None):
    """Set actual learned gate weights, never inject a route or mock a neural call."""
    parameters = list(model.error_predictor.parameters())
    saved = [parameter.detach().clone() for parameter in parameters]
    try:
        with torch.no_grad():
            for parameter in parameters:
                parameter.zero_()
            if refresh_age is None:
                model.error_predictor.output.bias.fill_(-12)
            else:
                # One real gate neuron reads log1p(age). For short intervals the
                # decision has a wide margin. Long intervals allow BF16 rounding
                # to pick the exact boundary; the test measures that boundary.
                model.error_predictor.input.weight[0, -1] = 1
                model.error_predictor.output.weight[0, 0] = 2
                midpoint = (
                    math.log(refresh_age) ** 2 + math.log1p(refresh_age) ** 2
                ) / 2
                model.error_predictor.output.bias.fill_(
                    math.log(math.expm1(model.config.refresh_cost)) - 2 * midpoint
                )
        yield
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, saved):
                parameter.copy_(original)


def test_current_and_future_characters_cannot_change_context_or_refresh_decision(model):
    ids = _ids(65, batch=2)
    changed = ids.clone()
    changed[:, 17:] = (changed[:, 17:] + 9) % VOCAB_SIZE
    with torch.no_grad():
        before = model.predict(ids)
        after = model.predict(changed)
    for key in ("contexts", "predicted_kl", "routes"):
        torch.testing.assert_close(
            before[key][:, :18], after[key][:, :18], rtol=0, atol=0
        )
    torch.testing.assert_close(
        before["logits"][:, :17], after["logits"][:, :17], rtol=0, atol=0
    )
    # Only earlier bits of c_17 may affect later bit decisions within c_17.
    torch.testing.assert_close(
        before["logits"][:, 17, 0], after["logits"][:, 17, 0], rtol=0, atol=0
    )
    assert bool((before["contexts"][:, 18:] != after["contexts"][:, 18:]).any())
    assert bool((before["predicted_kl"][:, 18:] != after["predicted_kl"][:, 18:]).any())


def test_learned_long_catchup_matches_dense_history_without_repeated_positions(
    model, monkeypatch
):
    ids = _ids(300)
    codes = model.identity_bits(ids)
    suffixes = []
    compiled_catch_up = generation._catch_up

    def observed_catch_up(model, features, keys, values, start):
        suffixes.append((start, start + features.shape[1]))
        return compiled_catch_up(model, features, keys, values, start)

    with _gate_policy(model, refresh_age=129), torch.no_grad():
        teacher = model.teacher_context(codes)
        teacher_logits = model.score(teacher, codes)
        state = generation.initial_state(model, ids.shape[1])
        refreshed_contexts, refreshed_logits = [], []
        with monkeypatch.context() as patch:
            patch.setattr(generation, "_catch_up", observed_catch_up)
            for position in range(ids.shape[1]):
                context, refreshed = generation.next_context(model, state)
                if refreshed:
                    refreshed_contexts.append(context)
                    refreshed_logits.append(model.score(context, codes[:, position]))
                generation.accept_code(model, state, codes[:, position])
        assert 2 < state.backbone_calls < 10
        assert max(end - start for start, end in suffixes) >= 64
        assert suffixes[0] == (0, 1)
        assert all(left[1] == right[0] for left, right in pairwise(suffixes))
        assert sum(end - start for start, end in suffixes) == state.backbone_positions
        assert (
            state.backbone_positions == state.refresh_positions[-1] + 1 < ids.shape[1]
        )
        assert len(suffixes) == state.backbone_calls == len(state.refresh_positions)
        assert state.dynamics_steps == ids.shape[1] - 1
        expected_contexts = teacher[:, state.refresh_positions]
        expected_logits = teacher_logits[:, state.refresh_positions]
        _assert_bf16_close(torch.stack(refreshed_contexts, 1), expected_contexts)
        _assert_bf16_close(torch.stack(refreshed_logits, 1), expected_logits)
        selected_codes = codes[:, state.refresh_positions]
        actual_nats = F.binary_cross_entropy_with_logits(
            torch.stack(refreshed_logits, 1), selected_codes, reduction="none"
        ).sum(-1)
        expected_nats = F.binary_cross_entropy_with_logits(
            expected_logits, selected_codes, reduction="none"
        ).sum(-1)
        _assert_bf16_close(actual_nats, expected_nats)


def test_deployed_offline_likelihood_routes_and_work_match_cached_execution(model):
    ids = _ids(23, batch=2)
    codes = model.identity_bits(ids)
    with _gate_policy(model, refresh_age=3), torch.no_grad():
        offline = model.predict(ids)
        # Gate uses actual learned age weights with decisions safely separated
        # from the threshold, so BF16 trunk rounding cannot flip the route.
        assert (
            float(
                (offline["predicted_kl"][:, 1:] - model.config.refresh_cost).abs().min()
            )
            > 0.01
        )
        rows, routes, processed = [], [], []
        for row in range(ids.shape[0]):
            state = generation.initial_state(model, ids.shape[1])
            contexts, refreshes = [], []
            for position in range(ids.shape[1]):
                context, refresh = generation.next_context(model, state)
                contexts.append(context)
                refreshes.append(refresh)
                generation.accept_code(model, state, codes[row : row + 1, position])
            rows.append(torch.stack(contexts, 1))
            routes.append(refreshes)
            processed.append(state.backbone_positions)
        contexts = torch.cat(rows, 0)
        actual_routes = torch.tensor(routes, device="cuda", dtype=torch.bool)
        logits = model.score(contexts, codes)
        torch.testing.assert_close(actual_routes, offline["routes"], rtol=0, atol=0)
        assert bool(actual_routes[:, 1:].any()) and bool((~actual_routes[:, 1:]).any())
        _assert_bf16_close(contexts, offline["contexts"])
        _assert_bf16_close(logits, offline["logits"])
        actual_rate = F.binary_cross_entropy_with_logits(logits, codes, reduction="sum")
        expected_rate = F.binary_cross_entropy_with_logits(
            offline["logits"], codes, reduction="sum"
        )
        _assert_bf16_close(actual_rate, expected_rate)
        positions = torch.arange(ids.shape[1], device="cuda")
        expected_processed = torch.where(actual_routes, positions + 1, 0).amax(-1)
        assert sum(processed) == int(expected_processed.sum())
        assert sum(processed) > int(actual_routes.sum())


def test_canonical_evaluation_scores_teacher_not_unverified_rollout(model):
    ids = _ids(23, batch=2)
    codes = model.identity_bits(ids)
    with torch.no_grad():
        expected = F.binary_cross_entropy_with_logits(
            model.score(model.teacher_context(codes), codes), codes, reduction="sum"
        )
        with _gate_policy(model):
            approximate = model.predict(ids)
            assert bool((~approximate["routes"][:, 1:]).all())
            assert not torch.allclose(
                approximate["logits"], approximate["teacher_logits"]
            )
            never_refresh, stats = model(ids)
        with _gate_policy(model, refresh_age=2):
            frequent_refresh, _ = model(ids)
        torch.testing.assert_close(never_refresh, expected)
        torch.testing.assert_close(frequent_refresh, expected)
        torch.testing.assert_close(stats["rate_nats"], expected)


def test_cheap_rollout_uses_previous_character_and_can_outlive_training_horizon(model):
    ids = _ids(65)
    codes = model.identity_bits(ids)
    with _gate_policy(model), torch.no_grad():
        teacher = model.teacher_context(codes)
        explicit = teacher[:, 0]
        reference = [explicit]
        for position in range(1, ids.shape[1]):
            explicit, _ = model.advance(
                explicit,
                codes[:, position - 1],
                torch.full((1,), position, device="cuda", dtype=torch.float32),
            )
            reference.append(explicit)
        offline = model.predict(ids)
        state = generation.initial_state(model, ids.shape[1])
        cached = []
        for position in range(ids.shape[1]):
            context, refresh = generation.next_context(model, state)
            assert refresh == (position == 0)
            cached.append(context)
            generation.accept_code(model, state, codes[:, position])
        expected = torch.stack(reference, 1)
        _assert_bf16_close(offline["contexts"], expected)
        _assert_bf16_close(torch.stack(cached, 1), expected)
        assert state.backbone_calls == state.backbone_positions == 1
        assert state.dynamics_steps == 64 > model.config.rollout_horizon
        assert state.refresh_positions == [0]
        assert bool((expected[:, 1:] != teacher[:, 1:]).any())


def test_long_positive_feedback_rollout_keeps_character_likelihood_finite(model):
    parameters = list(model.transition.parameters())
    saved = [parameter.detach().clone() for parameter in parameters]
    try:
        with _gate_policy(model), torch.no_grad():
            model.transition.input.weight.zero_()
            model.transition.input.bias.zero_()
            model.transition.input.weight[0, 0] = 1
            model.transition.input.bias[0] = 1
            model.transition.output.weight.zero_()
            model.transition.output.bias.zero_()
            model.transition.output.weight[0, 0] = 1
            # Without normalized transition inputs, this positive feedback
            # repeatedly squares a growing state and overflows after a few steps.
            state = generation.initial_state(model, 65)
            code = model.identity_bits(
                torch.zeros((1,), device="cuda", dtype=torch.long)
            )
            for _ in range(65):
                context, _ = generation.next_context(model, state)
                generation.accept_code(model, state, code)
            logits = model.score(context, code)
            loss = F.binary_cross_entropy_with_logits(logits, code, reduction="sum")
            assert bool(torch.isfinite(context).all())
            assert bool(torch.isfinite(loss))
            assert state.backbone_calls == 1
            assert state.dynamics_steps == 64 > model.config.rollout_horizon
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, saved):
                parameter.copy_(original)


def test_forced_backbone_diagnostic_matches_dense_every_character(model):
    ids = _ids(17)
    codes = model.identity_bits(ids)
    with torch.no_grad():
        teacher = model.teacher_context(codes)
        state = generation.initial_state(model, ids.shape[1])
        contexts = []
        for position in range(ids.shape[1]):
            context, refresh = generation.next_context(model, state, force_refresh=True)
            assert refresh
            contexts.append(context)
            generation.accept_code(model, state, codes[:, position])
        _assert_bf16_close(torch.stack(contexts, 1), teacher)
        assert state.backbone_calls == state.backbone_positions == ids.shape[1]
        assert state.dynamics_steps == 0


def test_full_code_space_normalizes_including_reserved_mass_and_sampler_outcomes(model):
    ids = _ids(19)
    candidates = torch.arange(1 << BITS, device="cuda")
    codes = model.identity_bits(candidates)
    with torch.no_grad():
        known = model.predict(ids)["contexts"][:, -1]
        contexts = known.expand(1 << BITS, -1)
        logits = model.score(contexts, codes)
        nats = F.binary_cross_entropy_with_logits(logits, codes, reduction="none").sum(
            -1
        )
        probabilities = (-nats).exp()
        # Endpoint uniforms force each address, even those the alphabet reserves.
        sampled = sample_prefix_code(model.head, contexts, 1 - codes, 1.0)
    torch.testing.assert_close(sampled, codes, rtol=0, atol=0)
    torch.testing.assert_close(
        probabilities.sum(), torch.ones((), device="cuda"), rtol=2e-5, atol=2e-6
    )
    valid_mass = probabilities[:VOCAB_SIZE].sum()
    reserved_mass = probabilities[VOCAB_SIZE:].sum()
    assert 0 < float(valid_mass) < 1
    assert float(reserved_mass) > 0
    torch.testing.assert_close(valid_mass, 1 - reserved_mass, rtol=2e-5, atol=2e-6)
    for bit in range(BITS):
        grouped = logits[:, bit].reshape(1 << bit, 1 << (BITS - bit))
        torch.testing.assert_close(
            grouped, grouped[:, :1].expand_as(grouped), rtol=0, atol=0
        )
        if bit:
            flipped_previous = candidates ^ (1 << (BITS - bit))
            assert bool((logits[:, bit] != logits[flipped_previous, bit]).any())


@pytest.mark.parametrize("model_type", [DynamicsGPT, BoundedDynamicsGPT])
def test_normal_microbatch_backward_trains_target_and_rollout_without_nan(model_type):
    # The real 64x1024 microbatch exposed a symbolic mixed-reduction compiler
    # failure invisible to the short-sequence gradient-isolation fixtures.
    torch.manual_seed(1337)
    normal = model_type(
        DynamicsConfig(vocab_size=3099, code_bits=12, rollout_horizon=2)
    ).cuda()
    normal.compile_components()
    normal.train()
    ids = torch.arange(64 * 1024, device="cuda").reshape(64, 1024) % 3099
    loss, stats = normal(ids)
    assert bool(torch.isfinite(loss))
    assert bool(torch.isfinite(torch.stack(tuple(stats.values()))).all())
    loss.backward()
    for parameter in normal.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
    for module in (normal.prior, normal.transition, normal.error_predictor):
        assert any(
            bool((parameter.grad != 0).any()) for parameter in module.parameters()
        )


def test_bounded_ablation_preserves_every_shared_initial_parameter():
    config = DynamicsConfig(vocab_size=VOCAB_SIZE, model_dim=128, num_layers=1)
    torch.manual_seed(1337)
    reference = DynamicsGPT(config).cuda()
    torch.manual_seed(1337)
    bounded = BoundedDynamicsGPT(config).cuda()
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(
            parameter, bounded.get_parameter(name), rtol=0, atol=0
        )


def test_normalized_recurrence_prevents_radial_drift_beyond_supervised_horizon():
    config = DynamicsConfig(vocab_size=VOCAB_SIZE, model_dim=128, num_layers=1)
    reference = ResidualTransition(config).cuda()
    bounded = BoundedResidualTransition(config).cuda()
    with torch.no_grad():
        for transition in (reference, bounded):
            transition.output.weight.zero_()
            # A finite recurrent increment must not accumulate into an ever
            # larger state. This is the mechanism observed in the spike replay.
            transition.output.bias.fill_(0.25)
        old_state = torch.ones((2, 128), device="cuda", dtype=torch.bfloat16)
        new_state = old_state.clone()
        code = torch.zeros((2, config.code_bits), device="cuda")
        raw_step = torch.compile(reference, dynamic=True, fullgraph=True)
        normalized_step = torch.compile(bounded, dynamic=True, fullgraph=True)
        for _ in range(256):
            old_state = raw_step(old_state, code)
            new_state = normalized_step(new_state, code)
    assert old_state.float().square().mean().sqrt().item() > 32
    assert new_state.float().square().mean().sqrt().item() <= 1.01
    assert bool(torch.isfinite(new_state).all())


def test_pending_ownership_and_input_buffers_survive_rejected_or_mutated_codes(model):
    empty = generation.initial_state(model, 0)
    with pytest.raises(ValueError):
        generation.next_context(model, empty)
    first = generation.initial_state(model, 3)
    second = generation.initial_state(model, 3)
    code = model.identity_bits(torch.tensor([7], device="cuda"))
    with pytest.raises(RuntimeError):
        generation.accept_code(model, first, code)
    with pytest.raises(ValueError):
        generation.next_context(copy.copy(model), first)
    with _gate_policy(model), torch.inference_mode():
        exposed, _ = generation.next_context(model, first)
        generation.next_context(model, second)
        with pytest.raises(RuntimeError):
            generation.next_context(model, first)
        for invalid in (
            torch.full_like(code, 0.5),
            torch.ones_like(code),
            code[:, :-1],
        ):
            with pytest.raises(ValueError):
                generation.accept_code(model, first, invalid)
        generation.accept_code(model, first, code)
        generation.accept_code(model, second, code.clone())
        exposed.fill_(123)
        code.fill_(1)
        left, _ = generation.next_context(model, first)
        right, _ = generation.next_context(model, second)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        zero = torch.zeros_like(code)
        generation.accept_code(model, first, zero)
        generation.accept_code(model, second, zero)
        left, _ = generation.next_context(model, first, force_refresh=True)
        right, _ = generation.next_context(model, second, force_refresh=True)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        generation.accept_code(model, first, zero)
        with pytest.raises(ValueError):
            generation.next_context(model, first)


def test_partially_executed_neural_failure_permanently_invalidates_state(
    model, monkeypatch
):
    state = generation.initial_state(model, 2)
    compiled_catch_up = generation._catch_up

    def fail_after_cache_write(*args):
        compiled_catch_up(*args)
        raise RuntimeError("injected failure after real cache update")

    monkeypatch.setattr(generation, "_catch_up", fail_after_cache_write)
    with pytest.raises(RuntimeError):
        generation.next_context(model, state)
    with pytest.raises(RuntimeError):
        generation.next_context(model, state)
    with pytest.raises(RuntimeError):
        generation.accept_code(model, state, torch.zeros((1, BITS), device="cuda"))


def test_horizon_targets_follow_each_emitted_character_and_normalize_valid_starts(
    model,
):
    ids = _ids(7, batch=2)
    codes = model.identity_bits(ids)
    auxiliary = torch.compile(model.auxiliary_losses, dynamic=True, fullgraph=True)

    def independent_rollouts(teacher, teacher_logits, codes):
        losses = [teacher_logits.new_zeros(()) for _ in range(3)]
        for horizon in range(1, model.config.rollout_horizon + 1):
            starts = torch.arange(codes.shape[1] - horizon, device=codes.device)
            predicted = teacher[:, starts]
            for step in range(horizon):
                predicted, errors = model.advance(
                    predicted,
                    codes[:, starts + step],
                    torch.full_like(codes[:, starts, 0], step + 1),
                )
            targets = starts + horizon
            actual_logits = model.score(predicted, codes[:, targets])
            target_logits = teacher_logits[:, targets]
            target_probability = target_logits.sigmoid()
            kl = (
                (
                    F.binary_cross_entropy_with_logits(
                        actual_logits, target_probability, reduction="none"
                    )
                    - F.binary_cross_entropy_with_logits(
                        target_logits, target_probability, reduction="none"
                    )
                )
                .sum(-1)
                .clamp_min(0)
            )
            losses[0] = losses[0] + F.smooth_l1_loss(
                predicted.float(), teacher[:, targets].float()
            )
            losses[1] = losses[1] + kl.mean()
            losses[2] = losses[2] + F.mse_loss(errors, kl)
        return tuple(loss / model.config.rollout_horizon for loss in losses)

    reference = torch.compile(independent_rollouts, dynamic=True, fullgraph=True)
    with torch.no_grad():
        teacher = model.teacher_context(codes)
        logits = model.score(teacher, codes)
        actual = auxiliary(teacher, logits, codes)
        expected = reference(teacher, logits, codes)
    for measured, target in zip(actual, expected):
        torch.testing.assert_close(measured, target, rtol=0.03, atol=0.003)


@pytest.mark.parametrize("length", [1, 17])
def test_training_backward_is_finite_for_teacher_transition_and_gate(model, length):
    model.zero_grad(set_to_none=True)
    model.train()
    try:
        ids = _ids(length, batch=2)
        total, stats = model(ids)
        assert bool(torch.isfinite(total))
        expected = stats["rate_nats"] + ids.numel() * (
            model.config.latent_weight * stats["latent_loss"]
            + model.config.rollout_weight * stats["rollout_kl"]
            + model.config.gate_weight * stats["gate_loss"]
        )
        torch.testing.assert_close(total.detach(), expected)
        total.backward()
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert bool(torch.isfinite(parameter.grad).all())
        if length > 1:
            for module in (model.prior, model.transition, model.error_predictor):
                assert any(
                    bool((parameter.grad != 0).any())
                    for parameter in module.parameters()
                )
    finally:
        model.eval()
        model.zero_grad(set_to_none=True)


def test_auxiliary_backward_cannot_train_teacher_and_gate_cannot_train_transition(
    model,
):
    model.train()
    auxiliary = torch.compile(model.auxiliary_losses, dynamic=True, fullgraph=True)
    try:
        codes = model.identity_bits(_ids(17, batch=2))
        for gate_only in (False, True):
            model.zero_grad(set_to_none=True)
            teacher = model.teacher_context(codes)
            logits = model.score(teacher, codes)
            latent, rollout, gate = auxiliary(teacher, logits, codes)
            loss = gate if gate_only else latent + rollout + gate
            loss.backward()
            for parameter in model.prior.parameters():
                assert parameter.grad is None or bool((parameter.grad == 0).all())
            for parameter in model.error_predictor.parameters():
                assert parameter.grad is not None and bool(
                    torch.isfinite(parameter.grad).all()
                )
            assert any(
                bool((parameter.grad != 0).any())
                for parameter in model.error_predictor.parameters()
            )
            if gate_only:
                for parameter in model.transition.parameters():
                    assert parameter.grad is None or bool((parameter.grad == 0).all())
            else:
                for parameter in model.transition.parameters():
                    assert parameter.grad is not None and bool(
                        torch.isfinite(parameter.grad).all()
                    )
                assert any(
                    bool((parameter.grad != 0).any())
                    for parameter in model.transition.parameters()
                )
    finally:
        model.eval()
        model.zero_grad(set_to_none=True)
