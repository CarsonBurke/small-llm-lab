"""Host-only rollout state contracts; no model, CUDA graph, or RNG emulation."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from postraining import fast_inference
from postraining.fast_inference import (
    CapturedTrainingRolloutEngine,
    PromptPrefixBank,
    synchronize_fused_lora_policy_,
)
from postraining.token_carry import TokenCarryCombiner


class _CarryStatePolicy(nn.Module):
    """Scalar state machine: consuming token t advances state h to h + t."""

    token_carry = True

    def __init__(self):
        super().__init__()
        self.causal_lm = nn.Module()
        self.causal_lm.register_parameter(
            "anchor", nn.Parameter(torch.zeros((), dtype=torch.bfloat16), requires_grad=False)
        )
        self.causal_lm.config = SimpleNamespace(
            num_hidden_layers=1, num_key_value_heads=1, head_dim=1,
            hidden_size=1, pad_token_id=0, vocab_size=10,
        )
        self.causal_lm.model = SimpleNamespace(layers=[])

    def carry_embeddings(self, token_ids, previous_hidden):
        return token_ids[..., None].to(torch.bfloat16) + previous_hidden

    def cached_hidden(self, input_ids=None, *, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            return inputs_embeds
        return input_ids[..., None].to(torch.bfloat16)

    def logits(self, hidden):
        return hidden.expand(-1, 10)

    def rollout_values(self, hidden):
        return hidden[:, 0].float()


@pytest.fixture
def carry_engine(monkeypatch):
    monkeypatch.setattr(
        fast_inference, "build_fused_rollout_replica", lambda source: (source, ())
    )
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: None)
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    monkeypatch.setattr(fast_inference, "compile_invariant", lambda function: function)
    engine = CapturedTrainingRolloutEngine(
        _CarryStatePolicy(), stop_ids=(9,), prompts_per_rollout=3,
        samples_per_prompt=1, cache_length=12, temperature=1.0,
        top_k=1, top_p=1.0, compile_decode=True,
    )
    engine.carry_hidden[:, 0].copy_(torch.tensor([1, 5, 8]))
    engine.active.copy_(torch.tensor([True, True, False]))
    engine.response_limit.fill_(3)
    return engine


@pytest.mark.parametrize("statistics", [False, True])
@torch.inference_mode()
def test_decode_carries_every_token_but_freezes_finished_lanes(carry_engine, statistics):
    engine = carry_engine

    def advance(tokens):
        tokens = torch.tensor(tokens)
        if statistics:
            return engine.decode(
                tokens, torch.zeros(3), engine.carry_hidden[:, 0].float(), engine.cache
            )[0]
        return engine.decode_without_statistics(tokens, engine.cache)

    # Token 2 can be a delimiter: it follows exactly the same recurrence as 4.
    logits = advance([2, 9, 7])
    assert logits[:2, 0].tolist() == [3, 14]
    assert engine.carry_hidden[:, 0].tolist() == [3, 14, 8]
    assert engine.active.tolist() == [True, False, False]
    assert advance([4, 1, 1])[0, 0].item() == 7
    assert advance([6, 1, 1])[0, 0].item() == 13
    assert engine.output_position.tolist() == [3, 1, 0]
    assert engine.active.tolist() == [False, False, False]
    advance([8, 8, 8])
    assert engine.carry_hidden[:, 0].tolist() == [13, 14, 8]
    assert engine.generated[:, :3].tolist() == [[2, 4, 6], [9, 0, 0], [0, 0, 0]]
    assert engine._carry_history[:, :3, 0].tolist() == [
        [1, 3, 7], [5, 0, 0], [0, 0, 0],
    ]


@pytest.mark.parametrize("statistics", [False, True])
@torch.inference_mode()
def test_forced_delimiter_consumes_actual_token_and_preserves_carry_producers(
    carry_engine, monkeypatch, statistics,
):
    engine = carry_engine
    engine.answer_reserve_tokens = 1
    engine.thinking_end_token_id = 2
    engine.response_limit.fill_(4)
    # Put the inactive lane at the forcing boundary as well.
    engine.output_position[2] = 2
    monkeypatch.setattr(
        fast_inference, "top_k_top_p_sample",
        lambda logits, **kwargs: logits.argmax(-1),
    )
    for proposals in ([4, 2, 2], [4, 4, 2], [4, 6, 2], [6, 6, 2]):
        logits = torch.zeros(3, 10)
        logits.scatter_(1, torch.tensor(proposals)[:, None], 10.0)
        if statistics:
            token, logprob = engine.sample(logits)
            engine.decode(
                token, logprob, engine.carry_hidden[:, 0].float(), engine.cache,
            )
        else:
            engine.decode_without_statistics(engine.sample_tokens(logits), engine.cache)

    assert engine.generated[:, :4].tolist() == [
        [4, 4, 2, 6], [2, 4, 6, 6], [0, 0, 0, 0],
    ]
    assert engine._carry_history[:, :4, 0].tolist() == [
        [1, 5, 9, 11], [5, 7, 11, 17], [0, 0, 0, 0],
    ]
    assert engine.carry_hidden[:, 0].tolist() == [17, 23, 8]
    assert engine.thinking_closed.tolist() == [True, True, False]
    assert engine.output_position.tolist() == [4, 4, 2]
    if statistics:
        assert engine.values[:, :4].tolist() == [
            [1, 5, 9, 11], [5, 7, 11, 17], [0, 0, 0, 0],
        ]
        forced_logits = torch.zeros(10)
        forced_logits[4] = 10
        torch.testing.assert_close(
            engine.logprobs[0, 2], forced_logits.log_softmax(-1)[2],
        )

@torch.inference_mode()
def test_inactive_lane_at_history_capacity_preserves_final_producer(carry_engine):
    engine = carry_engine
    engine.active.copy_(torch.tensor([True, False, False]))
    engine.output_position[0] = engine.cache_length - 1
    engine.response_limit.fill_(engine.cache_length)
    engine.decode_without_statistics(torch.tensor([2, 9, 9]), engine.cache)
    engine.decode_without_statistics(torch.tensor([9, 9, 9]), engine.cache)
    assert engine._carry_history[0, -1, 0].item() == 1
    assert engine.generated[0, -1].item() == 2


def _prefix_bank(hidden):
    return PromptPrefixBank(
        lengths=torch.tensor([1, 2]), logits=torch.zeros(2, 10, dtype=torch.bfloat16),
        values=torch.empty(0), layer_keys=torch.empty(2, 2, 0, 0, 0),
        layer_values=torch.empty(2, 2, 0, 0, 0), hidden=hidden,
    )


@torch.inference_mode()
def test_refill_uses_new_prompt_hidden_without_contaminating_other_lanes(carry_engine):
    engine = carry_engine
    engine.cache = SimpleNamespace(layers=[])
    bank = _prefix_bank(torch.tensor([[20], [1]], dtype=torch.bfloat16))
    engine.active.zero_()
    engine._admit_prompt_rows(bank, 0, [1], max_new_tokens=1)
    logits = engine.decode_without_statistics(torch.tensor([8, 2, 8]), engine.cache)
    assert logits[1, 0].item() == 22
    assert engine.carry_hidden[:, 0].tolist() == [1, 22, 8]
    engine._admit_prompt_rows(bank, 1, [1], max_new_tokens=1)
    logits = engine.decode_without_statistics(torch.tensor([8, 2, 8]), engine.cache)
    assert logits[1, 0].item() == 3
    assert engine.carry_hidden[:, 0].tolist() == [1, 3, 8]
    assert engine.generated[1, :2].tolist() == [2, 0]
    assert bank.hidden[:, 0].tolist() == [20, 1]
    assert engine._carry_history[1, 0, 0].item() == 1


@pytest.fixture
def host_generation(carry_engine, monkeypatch):
    engine = carry_engine
    engine._compile_decode = False
    monkeypatch.setattr(engine, "_synchronize_generation", lambda: None)
    monkeypatch.setattr(engine, "_ensure_continuous_cache", lambda bank: None)
    monkeypatch.setattr(engine, "_bind_flash_cache", lambda: None)
    engine.cache = SimpleNamespace(layers=[], reset=lambda: None)

    def sample_tokens(logits, **kwargs):
        hidden = logits[:, 0]
        return torch.where(
            (hidden == 1) | (hidden == 30), 2,
            torch.where(hidden == 20, 4, torch.where(hidden == 24, 6, 9)),
        ).long()

    monkeypatch.setattr(fast_inference, "top_k_top_p_sample", sample_tokens)
    return engine


@pytest.fixture
def budget_host_generation(host_generation, monkeypatch):
    engine = host_generation
    engine.answer_reserve_tokens = 1
    engine.thinking_end_token_id = 2

    def sample_tokens(logits, **kwargs):
        hidden = logits[:, 0]
        return torch.where(
            (hidden == 1) | (hidden == 60), 2,
            torch.where(
                (hidden == 11) | (hidden == 62), 9,
                torch.where((hidden == 30) | (hidden == 50), 6, 4),
            ),
        ).long()

    monkeypatch.setattr(fast_inference, "top_k_top_p_sample", sample_tokens)
    return engine


def test_continuous_budget_exports_delimiter_and_answer_producers_across_refill(
    budget_host_generation, monkeypatch,
):
    engine = budget_host_generation
    hidden = torch.tensor([[1], [60], [20], [40]], dtype=torch.bfloat16)
    bank = PromptPrefixBank(
        lengths=torch.ones(4, dtype=torch.long),
        logits=hidden.expand(-1, 10), hidden=hidden,
        values=torch.empty(0), layer_keys=torch.empty(4, 1, 0, 0, 0),
        layer_values=torch.empty(4, 1, 0, 0, 0),
    )
    monkeypatch.setattr(engine, "_build_prompt_prefix_bank", lambda *args, **kwargs: bank)
    result = engine.generate_prompt_pool(
        [torch.tensor([value]) for value in (1, 60, 20, 40)],
        max_new_tokens=4, completion_poll_steps=1,
    )
    # Prompt 60 closes naturally and finishes while prompt 1 is still answering.
    # Its replacement must reopen only that lane, then force its own delimiter.
    assert [row.tolist() for row in result.responses] == [
        [2, 4, 4, 9], [2, 9], [4, 4, 2, 6], [4, 4, 2, 6],
    ]
    assert [row[:, 0].tolist() for row in result.carry_hiddens] == [
        [1, 3, 7, 11], [60, 62], [20, 24, 28, 30], [40, 44, 48, 50],
    ]


def test_fixed_budget_exports_answer_producers_and_resets_closure(
    budget_host_generation, monkeypatch,
):
    engine = budget_host_generation
    event = SimpleNamespace(
        record=lambda: None, synchronize=lambda: None, elapsed_time=lambda other: 0.0,
    )
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: event)
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())
    monkeypatch.setattr(
        engine, "_prepare_prompts",
        lambda rows: (torch.stack(rows), torch.zeros(3, 1, dtype=torch.long), 1),
    )
    response, *_ = engine.generate_prompts(
        [torch.tensor([value]) for value in (1, 60, 20)],
        max_new_tokens=4, collect_statistics=False,
    )
    assert response.tolist() == [[2, 4, 4, 9], [2, 9, 0, 0], [4, 4, 2, 6]]
    saved = engine.last_carry_hiddens
    assert saved[:, :, 0].tolist() == [
        [1, 3, 7, 11], [60, 62, 0, 0], [20, 24, 28, 30],
    ]
    response, *_ = engine.generate_prompts(
        [torch.tensor([20])] * 3, max_new_tokens=4, collect_statistics=False,
    )
    assert response.tolist() == [[4, 4, 2, 6]] * 3
    assert engine.last_carry_hiddens[:, :, 0].tolist() == [[20, 24, 28, 30]] * 3
    assert saved[:, :, 0].tolist() == [
        [1, 3, 7, 11], [60, 62, 0, 0], [20, 24, 28, 30],
    ]


def test_continuous_exports_exact_logical_histories_across_refills(
    host_generation, monkeypatch,
):
    engine = host_generation
    engine.samples_per_prompt = 2
    bank = PromptPrefixBank(
        lengths=torch.ones(4, dtype=torch.long),
        logits=torch.tensor([1, 5, 20, 30], dtype=torch.bfloat16)[:, None].expand(-1, 10),
        values=torch.empty(0), layer_keys=torch.empty(4, 1, 0, 0, 0),
        layer_values=torch.empty(4, 1, 0, 0, 0),
        hidden=torch.tensor([[1], [5], [20], [30]], dtype=torch.bfloat16),
    )
    monkeypatch.setattr(engine, "_build_prompt_prefix_bank", lambda *args, **kwargs: bank)
    prompts = [torch.tensor([value]) for value in (1, 5, 20, 30)]
    result = engine.generate_prompt_pool(
        prompts, max_new_tokens=3, completion_poll_steps=2,
    )
    assert [row.tolist() for row in result.responses] == [
        [2, 9], [2, 9], [9], [9], [4, 6, 2], [4, 6, 2], [2, 9], [2, 9],
    ]
    expected = [[1, 3], [1, 3], [5], [5], [20, 24, 30], [20, 24, 30], [30, 32], [30, 32]]
    assert [row[:, 0].tolist() for row in result.carry_hiddens] == expected
    for response, carry in zip(result.responses, result.carry_hiddens, strict=True):
        assert carry.shape == (response.numel(), 1)
        assert carry.dtype == torch.bfloat16 and carry.device.type == "cpu"
        assert not carry.requires_grad and not carry.is_inference()
        assert carry.untyped_storage().nbytes() == carry.numel() * carry.element_size()
    replay_weight = torch.ones(1, dtype=torch.bfloat16, requires_grad=True)
    (result.carry_hiddens[0] * replay_weight).sum().backward()
    assert replay_weight.grad.item() == 4

    # Later rollout/refill and cache release must not mutate completed records.
    bank.hidden.add_(100)
    engine.generate_prompt_pool(prompts, max_new_tokens=3, completion_poll_steps=2)
    engine.release_cache()
    assert engine._carry_history is None and engine.carry_hidden is None
    assert [row[:, 0].tolist() for row in result.carry_hiddens] == expected
    engine._restore_rollout_cache()
    assert engine._carry_history.shape == (3, 12, 1)
    assert engine._continuous_decode_graph is None


def test_fixed_exports_owning_producers_with_zero_padding(host_generation, monkeypatch):
    engine = host_generation
    event = SimpleNamespace(
        record=lambda: None, synchronize=lambda: None, elapsed_time=lambda other: 0.0,
    )
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: event)
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())
    prompts = [torch.tensor([value]) for value in (1, 5, 20)]
    monkeypatch.setattr(
        engine, "_prepare_prompts",
        lambda rows: (torch.stack(rows), torch.zeros(3, 1, dtype=torch.long), 1),
    )
    response, *_ = engine.generate_prompts(
        prompts, max_new_tokens=3, collect_statistics=False,
    )
    saved = engine.last_carry_hiddens
    assert response.tolist() == [[2, 9, 0], [9, 0, 0], [4, 6, 2]]
    assert saved[:, :, 0].tolist() == [[1, 3, 0], [5, 0, 0], [20, 24, 30]]
    assert saved.shape == (3, 3, 1) and saved.dtype == torch.bfloat16
    assert saved.device.type == "cpu" and not saved.is_inference()
    assert saved.untyped_storage().nbytes() == saved.numel() * saved.element_size()
    engine.generate_prompts(
        [torch.tensor([5])] * 3, max_new_tokens=1, collect_statistics=False,
    )
    engine.release_cache()
    assert saved[:, :, 0].tolist() == [[1, 3, 0], [5, 0, 0], [20, 24, 30]]
    assert engine.last_carry_hiddens[:, :, 0].tolist() == [[5], [5], [5]]


def test_carry_admission_rejects_a_token_only_prefix_bank(carry_engine):
    with pytest.raises(ValueError, match="prompt final hidden"):
        carry_engine._admit_prompt_rows(_prefix_bank(None), 0, [1], max_new_tokens=2)


@pytest.mark.parametrize(
    "options,message",
    [
        ({"compile_decode": True, "invariant_decode": True}, "invariant/Uno"),
        ({"compile_decode": False}, "compiled CUDA"),
    ],
)
def test_unsupported_carry_modes_fail_before_replica_creation(options, message):
    with pytest.raises(ValueError, match=message):
        CapturedTrainingRolloutEngine(
            SimpleNamespace(token_carry=True), stop_ids=(9,), prompts_per_rollout=1,
            samples_per_prompt=1, cache_length=12, temperature=1.0,
            top_k=1, top_p=1.0, **options,
        )


def _synchronization_side():
    backbone = nn.Module()
    backbone.model = nn.Module()
    backbone.model.layers = nn.ModuleList()
    combiner = TokenCarryCombiner(3)
    return SimpleNamespace(causal_lm=backbone, token_carry=True, token_combiner=combiner)


@torch.no_grad()
def test_sync_refreshes_scaled_residual_without_replacing_graph_storage():
    source, destination = _synchronization_side(), _synchronization_side()
    token_address = destination.token_combiner.token_delta.weight.data_ptr()
    carry_address = destination.token_combiner.carry.weight.data_ptr()
    scale_address = destination.token_combiner.scale.data_ptr()
    token = torch.tensor([[5.0, 2.0, -1.0]])
    carry = torch.tensor([[7.0, -3.0, 4.0]])
    for token_weight, carry_weight, scale in (
        (2.0, 3.0, [-0.5, 0.0, 0.25]),
        (4.0, -1.0, [0.125, -0.75, 1.5]),
    ):
        source.token_combiner.token_delta.weight.copy_(torch.eye(3) * token_weight)
        source.token_combiner.carry.weight.copy_(torch.eye(3) * carry_weight)
        source.token_combiner.scale.copy_(torch.tensor(scale))
        synchronize_fused_lora_policy_(destination, source)
        result = destination.token_combiner(token, carry)
        expected = token + torch.tensor(scale) * (token_weight * token + carry_weight * carry)
        torch.testing.assert_close(result, expected)
        assert destination.token_combiner.token_delta.weight.data_ptr() == token_address
        assert destination.token_combiner.carry.weight.data_ptr() == carry_address
        assert destination.token_combiner.scale.data_ptr() == scale_address


def test_sync_rejects_a_token_only_replica():
    source, destination = _synchronization_side(), _synchronization_side()
    destination.token_carry = False
    with pytest.raises(ValueError, match="token-carry mode differs"):
        synchronize_fused_lora_policy_(destination, source)


def test_token_only_decoders_reject_recurrent_policy_before_allocating_cache():
    from postraining.fast_inference import FixedLengthInferenceEngine
    from postraining.nextlat_speculative import NextLatSpeculativeEngine
    from postraining.train_minicpm_vapo import RolloutEngine

    policy = SimpleNamespace(token_carry=True)
    with pytest.raises(ValueError, match="token carry"):
        FixedLengthInferenceEngine(
            policy, batch_size=1, cache_length=8, temperature=0.9,
            top_k=20, top_p=0.95, compile_decode=True,
        )
    with pytest.raises(ValueError, match="token carry"):
        RolloutEngine(
            policy, None, prompts_per_rollout=1, samples_per_prompt=1,
            cache_length=8, temperature=0.9, top_k=20, top_p=0.95,
            compile_decode=True,
        )
    with pytest.raises(ValueError, match="token carry"):
        NextLatSpeculativeEngine(
            policy, stop_ids=(1,), prompts_per_rollout=1, samples_per_prompt=1,
            cache_length=8, draft_length=2, temperature=0.9, top_p=0.95,
            compile_decode=True,
        )
