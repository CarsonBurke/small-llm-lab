"""Exact speculation contracts; CUDA runs must be submitted through mlq."""

from contextlib import contextmanager

import pytest
import torch

from pretraining.nanogpt_mini import speculative_generation as generation
from pretraining.nanogpt_mini.nanogpt_mini_dynamics_model import (
    DynamicsConfig,
    DynamicsGPT,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
BITS = 5
VOCAB_SIZE = 29


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
        )
    ).cuda()
    with torch.no_grad():
        # Nonzero attention/MLP residuals make stale, repeated, or missing KV
        # positions observable even when head decisions have generous margins.
        for block in result.prior.blocks:
            block.attn.proj.weight.normal_(std=0.02)
            block.mlp.proj.weight.normal_(std=0.02)
        result.head.output.weight.normal_(std=0.09)
        result.head.position.weight.uniform_(0.25, 0.5)
        result.transition.output.weight.normal_(std=0.002)
    result.compile_components()
    result.requires_grad_(False)
    result.eval()
    yield result
    torch.compiler.reset()


@contextmanager
def _saved_proposal_and_head(model):
    parameters = tuple(model.head.parameters()) + tuple(model.transition.parameters())
    saved = [parameter.detach().clone() for parameter in parameters]
    try:
        with torch.inference_mode():
            yield
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, saved, strict=True):
                parameter.copy_(original)


def _separated_policy(model, prefix_ones=0, transition_increment=0.0):
    """Real neural weights: bounded teacher coordinates vs a drifting proposal.

    RMS-normalized teacher coordinates have magnitude at most sqrt(512).
    The head's threshold is 2 / .01 = 200, well away from every teacher
    state. A +150 transition crosses it on proposal two; +300 on one.
    """
    head = model.head
    head.context.weight.zero_()
    head.context.bias.zero_()
    head.context.weight[0, 0] = 0.01
    head.prefix.weight.zero_()
    head.position.weight.zero_()
    head.position.weight[:prefix_ones, 0] = 4
    head.output.weight.zero_()
    head.output.weight[0, 0] = 1
    head.output.bias.fill_(-4)
    model.transition.output.weight.zero_()
    model.transition.output.bias.zero_()
    model.transition.output.bias[0] = transition_increment


def _observe_cache(monkeypatch):
    calls = []
    compiled_catch_up = generation._catch_up

    def observed(model, features, keys, values, start, return_all=False):
        result = compiled_catch_up(model, features, keys, values, start, return_all)
        calls.append(
            {
                "start": start,
                "features": features.detach().clone(),
                "contexts": result.detach().clone(),
                "capacity": keys[0].shape[2],
            }
        )
        return result

    monkeypatch.setattr(generation, "_catch_up", observed)
    return calls


def _assert_cached_teacher(model, result, calls):
    """Compare actual suffix states to dense history after every rollback.

    The dense reference uses emitted corrected/bonus characters, never stale
    speculative inputs. The wrapper above only observes the real compiled KV
    kernel; neither logits nor model/cache responses are mocked.
    """
    prompt = model.identity_bits(torch.tensor([result["prompt_ids"]], device="cuda"))
    emitted = torch.tensor(
        [result["generated_codes"]], device="cuda", dtype=torch.float32
    )
    bits = model.config.code_bits
    with torch.inference_mode():
        initial = model.teacher_context(
            torch.cat((prompt, torch.zeros((1, 1, bits), device="cuda")), dim=1)
        )
        torch.testing.assert_close(
            calls[0]["contexts"], initial[:, -1], rtol=0.02, atol=0.04
        )
        before = 1
        assert len(calls) == len(result["rounds"]) + 1
        for call, round_ in zip(calls[1:], result["rounds"], strict=True):
            length = round_["proposal_length"]
            start = len(result["prompt_ids"]) + before
            assert call["start"] == round_["cache_start"] == start
            assert call["features"].shape[1] == length + 1
            torch.testing.assert_close(
                call["features"][:, 0].float(),
                2 * emitted[:, before - 1] - 1,
                rtol=0,
                atol=0,
            )
            draft_inputs = (call["features"][:, 1:].float() + 1) / 2
            full_history = torch.cat(
                (
                    prompt,
                    emitted[:, :before],
                    draft_inputs,
                    torch.zeros((1, 1, bits), device="cuda"),
                ),
                dim=1,
            )
            expected = model.teacher_context(full_history)[
                :, start : start + length + 1
            ]
            torch.testing.assert_close(call["contexts"], expected, rtol=0.02, atol=0.04)
            accepted = round_["accepted_full_drafts"]
            torch.testing.assert_close(
                draft_inputs[:, :accepted],
                emitted[:, before : before + accepted],
                rtol=0,
                atol=0,
            )
            assert round_["cache_end"] == start + length + 1
            assert round_["cache_end"] <= call["capacity"]
            assert (
                call["capacity"]
                == len(result["prompt_ids"]) + result["requested_characters"]
            )
            assert (
                round_["cache_committed_length"]
                == len(result["prompt_ids"]) + round_["output_count"]
            )
            before = round_["output_count"]
        assert before == len(result["generated_codes"])
    assert result["target_calls"] == len(calls) - 1
    assert result["target_positions"] == sum(
        call["features"].shape[1] for call in calls[1:]
    )
    assert result["prefill_calls"] == 1
    assert result["prefill_positions"] == len(result["prompt_ids"]) + 1
    assert result["final_cache_length"] == len(result["prompt_ids"]) + len(
        result["generated_ids"]
    )


def test_bernoulli_acceptance_and_binary_residual_recover_target_mass():
    # Integrate the actual compiled acceptor over a deterministic uniform grid.
    # In a binary domain every rejected zero becomes one, and vice versa.
    p = torch.tensor([0.125, 0.875, 0.25, 0.625], device="cuda")
    q = torch.tensor([0.875, 0.125, 0.75, 0.375], device="cuda")
    count = 8192
    uniforms = ((torch.arange(count, device="cuda") + 0.5) / count)[:, None].expand(
        -1, p.numel()
    )
    temperature = 0.7
    target_logits = (torch.logit(p) * temperature).expand_as(uniforms)
    proposal_logits = (torch.logit(q) * temperature).expand_as(uniforms)
    zeros = torch.zeros_like(uniforms)
    ones = torch.ones_like(uniforms)
    accepted_zero = (
        generation.acceptance_mask(
            zeros, proposal_logits, target_logits, uniforms, temperature
        )
        .float()
        .mean(0)
    )
    accepted_one = (
        generation.acceptance_mask(
            ones, proposal_logits, target_logits, uniforms, temperature
        )
        .float()
        .mean(0)
    )
    torch.testing.assert_close(
        q * accepted_one, torch.minimum(p, q), rtol=0, atol=1 / count
    )
    recovered_one = q * accepted_one + (1 - q) * (1 - accepted_zero)
    torch.testing.assert_close(recovered_one, p, rtol=0, atol=1 / count)
    # Independent analytic identity, including the residual normalization.
    accepted_mass = torch.minimum(p, q) + torch.minimum(1 - p, 1 - q)
    residual_one = (p - q).clamp_min(0) / (1 - accepted_mass)
    torch.testing.assert_close(
        torch.minimum(p, q) + (1 - accepted_mass) * residual_one, p, rtol=0, atol=1e-7
    )


def test_acceptance_threshold_is_strict_and_preserves_tiny_probability_mass():
    half = torch.tensor(0.5, device="cuda")
    below = torch.nextafter(half, torch.zeros_like(half))
    above = torch.nextafter(half, torch.ones_like(half))
    uniforms = torch.stack((below, half, above))[None, :]
    proposal = torch.full_like(uniforms, float("inf"))
    target = torch.zeros_like(uniforms)
    accepted = generation.acceptance_mask(
        torch.ones_like(uniforms), proposal, target, uniforms, 1.0
    )
    assert accepted.tolist() == [[True, False, False]]
    tiny_p = torch.full((1, 2), 1e-8, device="cuda")
    tiny = generation.acceptance_mask(
        torch.ones_like(tiny_p),
        torch.full_like(tiny_p, float("inf")),
        torch.logit(tiny_p),
        torch.tensor([[1e-9, 1e-7]], device="cuda"),
        1.0,
    )
    assert tiny.tolist() == [[True, False]]
    greedy = generation.acceptance_mask(
        torch.tensor([[0.0, 1.0, 0.0]], device="cuda"),
        proposal,
        torch.tensor([[0.0, 2.0, -2.0]], device="cuda"),
        uniforms,
        0.0,
    )
    assert greedy.tolist() == [[True, True, True]]


@pytest.mark.parametrize("rejected_bit", [0, 2, BITS - 1])
def test_correction_preserves_prefix_and_conditions_suffix_on_flipped_bit(
    model, rejected_bit
):
    with _saved_proposal_and_head(model):
        _separated_policy(model)
        model.head.context.weight.zero_()
        model.head.output.bias.fill_(-2)
        if rejected_bit < BITS - 1:
            # Later bits switch from negative to positive only if the rejected
            # bit was flipped before the autoregressive suffix was sampled.
            model.head.prefix.weight[0, rejected_bit] = 3
        draft = torch.ones((1, BITS), device="cuda")
        draft[:, rejected_bit:] = 0
        context = torch.zeros(
            (1, model.config.model_dim), device="cuda", dtype=torch.bfloat16
        )
        uniforms = torch.full((1, BITS), 0.5, device="cuda")
        corrected = generation.correct_code(
            model.head, context, draft, rejected_bit, uniforms, 0.8
        )
        assert corrected.tolist() == [[1.0] * BITS]
        torch.testing.assert_close(
            corrected[:, :rejected_bit], draft[:, :rejected_bit], rtol=0, atol=0
        )
        target_logits = model.score(context, corrected)
        expected_suffix = (
            uniforms[:, rejected_bit + 1 :]
            < (target_logits[:, rejected_bit + 1 :] / 0.8).sigmoid()
        ).float()
        torch.testing.assert_close(
            corrected[:, rejected_bit + 1 :], expected_suffix, rtol=0, atol=0
        )
        # Opposite endpoint uniforms force the other suffix, not the prefix.
        alternate = generation.correct_code(
            model.head, context, draft, rejected_bit, torch.ones_like(uniforms), 0.8
        )
        assert alternate.tolist() == [
            [1.0] * (rejected_bit + 1) + [0.0] * (BITS - rejected_bit - 1)
        ]


@pytest.mark.parametrize("rejected_bit,accepted_before", [(0, 0), (2, 1)])
def test_first_rejection_rolls_back_and_next_round_matches_dense_teacher(
    model, monkeypatch, rejected_bit, accepted_before
):
    with _saved_proposal_and_head(model):
        _separated_policy(
            model,
            prefix_ones=rejected_bit,
            transition_increment=300 if accepted_before == 0 else 150,
        )
        calls = _observe_cache(monkeypatch)
        result = generation.generate(
            model, [3, 9, 7], 12, temperature=0, draft_tokens=4, trace=True
        )
        expected_address = ((1 << rejected_bit) - 1) << (BITS - rejected_bit)
        assert result["generated_ids"] == [expected_address] * 12
        round_ = result["rounds"][0]
        assert round_["first_rejected_bit"] == accepted_before * BITS + rejected_bit
        assert round_["accepted_full_drafts"] == accepted_before
        assert (
            round_["cache_committed_length"]
            == round_["cache_start"] + accepted_before + 1
        )
        assert round_["cache_committed_length"] < round_["cache_end"]
        assert result["corrected_characters"] > 1
        assert result["target_positions"] > len(result["generated_ids"]) - 1
        _assert_cached_teacher(model, result, calls)
        # Later proposed bits can agree again; first rejection must still cut
        # off the whole suffix, not filter individual accepted decisions.
        assert round_["output_count"] == 2 + accepted_before


def test_all_accepted_bonus_is_pending_once_and_greedy_matches_target_only(
    model, monkeypatch
):
    with _saved_proposal_and_head(model):
        _separated_policy(model, prefix_ones=2)
        baseline = generation.generate(model, [3, 9], 13, temperature=0, draft_tokens=0)
        calls = _observe_cache(monkeypatch)
        result = generation.generate(
            model, [3, 9], 13, temperature=0, draft_tokens=4, trace=True
        )
        assert result["generated_ids"] == baseline["generated_ids"] == [24] * 13
        assert result["generated_codes"] == baseline["generated_codes"]
        assert result["corrected_characters"] == 0
        assert result["accepted_draft_characters"] == 9
        assert result["bonus_characters"] == 3
        assert [round_["proposal_length"] for round_ in result["rounds"]] == [4, 4, 1]
        assert all(round_["first_rejected_bit"] is None for round_ in result["rounds"])
        assert result["target_calls"] == 3 < baseline["target_calls"] == 12
        assert result["target_positions"] == baseline["target_positions"] == 12
        _assert_cached_teacher(model, result, calls)


def test_reserved_proposals_are_corrected_not_treated_as_committed_output(model):
    with _saved_proposal_and_head(model):
        _separated_policy(model, transition_increment=300)
        result = generation.generate(
            model, [], 7, temperature=0, draft_tokens=4, trace=True
        )
        assert result["generated_ids"] == [0] * 7
        assert result["reserved_code"] is None
        assert result["stop_reason"] == "output_limit"
        assert result["corrected_characters"] == 5
        assert result["rounds"][0]["first_rejected_bit"] == 0
        # The proposal really is address 31 under these neural weights; use
        # the compiled proposer, not a fake generation response, to show it.
        anchor = model.teacher_context(torch.zeros((1, 1, BITS), device="cuda"))[:, 0]
        codes, _ = generation._propose(
            model,
            anchor,
            torch.zeros((1, BITS), device="cuda"),
            torch.zeros((4, BITS), device="cuda"),
            0.0,
        )
        assert codes.tolist() == [[1.0] * BITS] * 4


@pytest.mark.parametrize("accept_reserved_draft", [False, True])
def test_first_committed_reserved_code_stops_without_emitting_computed_suffix(
    model, monkeypatch, accept_reserved_draft
):
    with _saved_proposal_and_head(model):
        _separated_policy(model)
        # Separate two real teacher states with a linear projection: BOS->0,
        # after character zero->31. Margins remain large after BF16 rounding.
        teacher = model.teacher_context(torch.zeros((1, 2, BITS), device="cuda"))
        anchor, following = teacher[:, 0], teacher[:, 1]
        delta = following.float() - anchor.float()
        direction = 4 * delta[0] / delta.square().sum()
        model.head.context.weight.zero_()
        model.head.context.weight[0].copy_(direction)
        model.head.context.bias[0] = -(direction * anchor[0].float()).sum()
        if accept_reserved_draft:
            model.transition.output.bias.copy_(delta[0])
        calls = _observe_cache(monkeypatch)
        result = generation.generate(
            model, [], 8, temperature=0, draft_tokens=4, trace=True
        )
        assert result["generated_ids"] == [0]
        assert result["generated_addresses"] == [0, 31]
        assert result["generated_codes"] == [[0] * BITS, [1] * BITS]
        assert result["reserved_code"] == 31
        assert result["stop_reason"] == "reserved_code"
        assert result["valid_generated_characters"] == 1
        assert result["bonus_characters"] == 0
        assert result["accepted_draft_characters"] == int(accept_reserved_draft)
        assert result["corrected_characters"] == int(not accept_reserved_draft)
        assert result["target_calls"] == len(calls) - 1 == 1
        assert result["target_positions"] == calls[1]["features"].shape[1] == 5
        assert result["rounds"][0]["output_count"] == 2


def test_output_boundaries_skip_empty_work_and_never_overrun_cache(model, monkeypatch):
    with _saved_proposal_and_head(model):
        _separated_policy(model)
        calls = _observe_cache(monkeypatch)
        empty = generation.generate(
            model, [3, 9], 0, temperature=0, draft_tokens=4, trace=True
        )
        assert empty["generated_codes"] == []
        assert empty["target_calls"] == empty["prefill_calls"] == 0
        assert calls == []
        one = generation.generate(
            model, [3, 9], 1, temperature=0, draft_tokens=4, trace=True
        )
        assert one["generated_ids"] == [0]
        assert one["target_calls"] == 0
        assert len(calls) == 1
        calls.clear()
        two = generation.generate(
            model, [3, 9], 2, temperature=0, draft_tokens=4, trace=True
        )
        assert two["generated_ids"] == [0, 0]
        assert two["proposed_draft_characters"] == 0
        assert two["target_calls"] == two["target_positions"] == 1
        _assert_cached_teacher(model, two, calls)
        model.head.context.weight.zero_()
        model.head.output.bias.fill_(4)
        calls.clear()
        reserved = generation.generate(
            model, [3, 9], 1, temperature=0, draft_tokens=4, trace=True
        )
        assert reserved["generated_ids"] == []
        assert reserved["generated_addresses"] == [31]
        assert reserved["reserved_code"] == 31
        assert reserved["target_calls"] == 0
        assert len(calls) == 1
