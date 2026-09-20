from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from postraining.fast_inference import (
    CapturedTrainingRolloutEngine,
    PromptPrefixBank,
    _retire_inactive_flash_rows_,
    response_token_limits,
)


def test_fused_replica_removes_training_input_hook_without_changing_source(monkeypatch):
    from torch import nn
    from postraining.hf_runtime import prepare_text_only_transformers_runtime

    prepare_text_only_transformers_runtime()
    from transformers import PretrainedConfig, PreTrainedModel
    import transformers.masking_utils as masking
    import transformers.modeling_utils as modeling
    import postraining.fast_inference as inference

    class ToyLM(PreTrainedModel):
        def __init__(self):
            super().__init__(PretrainedConfig())
            self.embedding = nn.Embedding(7, 4)

        def get_input_embeddings(self):
            return self.embedding

    class ToyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.causal_lm = ToyLM()
            self.nextlat_head = nn.Identity()

        def token_embeddings(self, ids):
            return self.causal_lm.get_input_embeddings()(ids)

    source = ToyPolicy()
    source.requires_grad_(False)
    source.causal_lm.enable_input_require_grads()
    monkeypatch.setattr(
        inference, "import_module",
        lambda name: SimpleNamespace(flash_attn_varlen_func=None),
    )
    monkeypatch.setattr(inference, "merge_lora_for_inference", lambda model: None)
    monkeypatch.setattr(inference, "fuse_llama_projections_", lambda model: ())
    monkeypatch.setattr(inference, "synchronize_fused_lora_policy_", lambda *args: 0)
    monkeypatch.setattr(modeling.ALL_ATTENTION_FUNCTIONS, "register", lambda *args: None)
    monkeypatch.setattr(masking.ALL_MASK_ATTENTION_FUNCTIONS, "register", lambda *args: None)
    replica, _ = inference.build_fused_rollout_replica(source)
    ids = torch.tensor([1, 2])
    compiled_embedding = torch.compile(
        replica.token_embeddings, backend="eager", fullgraph=True, dynamic=True
    )
    with torch.inference_mode():
        actual = compiled_embedding(ids)
    torch.testing.assert_close(actual, source.token_embeddings(ids).detach())
    assert not actual.requires_grad
    assert source.token_embeddings(ids).requires_grad


def _scheduler_harness(
    responses: tuple[tuple[int, ...], ...],
    *,
    samples_per_prompt: int,
) -> tuple[CapturedTrainingRolloutEngine, list[tuple[int, ...]]]:
    batch_size = 2
    cache_length = 8
    prompt_count = len(responses) // samples_per_prompt
    engine = object.__new__(CapturedTrainingRolloutEngine)
    engine.answer_reserve_tokens = 0
    engine.thinking_end_token_id = None
    engine.batch_size = batch_size
    engine.samples_per_prompt = samples_per_prompt
    engine.cache_length = cache_length
    engine.stop_ids = (99,)
    engine.cache = SimpleNamespace(layers=[])
    engine.generated = torch.zeros((batch_size, cache_length), dtype=torch.long)
    engine.logprobs = torch.full(
        (batch_size, cache_length), -5.0, dtype=torch.float32
    )
    engine.values = torch.full(
        (batch_size, cache_length), -7.0, dtype=torch.float32
    )
    engine._graph_logits = torch.zeros((batch_size, 1))
    engine._graph_values = torch.full((batch_size,), -11.0)
    engine.cache_position = torch.zeros(1, dtype=torch.long)
    engine.output_position = torch.zeros(batch_size, dtype=torch.long)
    engine.active = torch.zeros(batch_size, dtype=torch.bool)
    engine.thinking_closed = torch.zeros_like(engine.active)
    engine.response_limit = torch.ones(batch_size, dtype=torch.long)
    engine.position_ids = torch.zeros((batch_size, 1), dtype=torch.long)
    engine.sequence_lengths = torch.zeros(batch_size, dtype=torch.long)
    engine.flash_sequence_lengths = torch.zeros(batch_size, dtype=torch.int32)
    engine.attention_mask = torch.zeros(
        (batch_size, cache_length), dtype=torch.bool
    )
    engine._compile_decode = False
    engine._continuous_decode_graph = None
    engine.capture_logprobs = False
    engine._synchronize_generation = lambda: None
    engine._ensure_continuous_cache = lambda bank: None

    prefix_calls: list[tuple[bool, str]] = []
    bank = PromptPrefixBank(
        lengths=torch.ones(prompt_count, dtype=torch.long),
        logits=torch.zeros((prompt_count, 1)),
        values=torch.empty(0),
        layer_keys=torch.empty((prompt_count, 1, 0, 0, 0)),
        layer_values=torch.empty((prompt_count, 1, 0, 0, 0)),
    )

    def build_prefix_bank(
        prompt_ids_cpu,
        *,
        prefill_batch_prompts: int,
        collect_values: bool,
        storage_device,
    ) -> PromptPrefixBank:
        del prompt_ids_cpu, prefill_batch_prompts
        prefix_calls.append((collect_values, str(storage_device)))
        return bank

    next_sample = [0] * prompt_count

    def admit(
        bank: PromptPrefixBank,
        prompt_index: int,
        slots,
        *,
        max_new_tokens: int,
    ) -> None:
        del bank
        for slot in slots:
            sample = next_sample[prompt_index]
            next_sample[prompt_index] += 1
            logical_row = prompt_index * samples_per_prompt + sample
            engine.generated[slot].zero_()
            engine.output_position[slot] = 0
            engine.response_limit[slot] = max_new_tokens
            engine._graph_logits[slot, 0] = logical_row
            engine.active[slot] = True

    active_rows_by_step: list[tuple[int, ...]] = []

    def decode_once() -> None:
        logical_rows = tuple(
            int(engine._graph_logits[slot, 0])
            for slot in range(batch_size)
            if bool(engine.active[slot])
        )
        active_rows_by_step.append(logical_rows)
        for slot in range(batch_size):
            if not bool(engine.active[slot]):
                continue
            logical_row = int(engine._graph_logits[slot, 0])
            position = int(engine.output_position[slot])
            token = responses[logical_row][position]
            engine.generated[slot, position] = token
            engine.output_position[slot] += 1
            stopped = token in engine.stop_ids
            exhausted = (
                int(engine.output_position[slot])
                >= int(engine.response_limit[slot])
            )
            engine.active[slot] = not (stopped or exhausted)

    engine._build_prompt_prefix_bank = build_prefix_bank
    engine._admit_prompt_rows = admit
    engine._continuous_decode_once = decode_once
    engine._prefix_calls = prefix_calls
    return engine, active_rows_by_step


def test_continuous_pool_refills_next_step_and_preserves_logical_order() -> None:
    expected = (
        (99,),
        (20, 21, 22, 23),
        (30, 99),
        (99,),
        (50, 51, 99),
        (60, 61, 62, 63),
    )
    engine, active_rows_by_step = _scheduler_harness(
        expected,
        samples_per_prompt=2,
    )

    result = CapturedTrainingRolloutEngine.generate_prompt_pool(
        engine,
        [torch.tensor([prompt]) for prompt in (1, 2, 3)],
        max_new_tokens=4,
        completion_poll_steps=1,
    )

    assert tuple(tuple(row.tolist()) for row in result.responses) == expected
    assert tuple(len(row) for row in result.responses) == (1, 4, 2, 1, 3, 4)
    assert [row.tolist() for row in result.logprobs] == [
        [0.0] * len(row) for row in expected
    ]
    # The pool clears the log-probability buffer up front so lanes that do not
    # capture behavior log-probabilities never export stale values.
    torch.testing.assert_close(
        engine.logprobs,
        torch.zeros((engine.batch_size, engine.cache_length)),
    )
    assert active_rows_by_step[:2] == [(0, 1), (2, 1)]
    assert result.minimum_active_rows_with_backlog == engine.batch_size
    assert engine._prefix_calls == [(False, "cpu")]


def test_mixed_prompt_lengths_use_each_remaining_budget_after_lane_reuse() -> None:
    expected = ((10,) * 6, (20,) * 4, (30,) * 5, (40,) * 3)
    engine, active_rows = _scheduler_harness(expected, samples_per_prompt=1)
    prompts = [torch.ones(length, dtype=torch.long) for length in (2, 4, 3, 5)]
    result = engine.generate_prompt_pool(
        prompts, max_new_tokens=8, context_tokens=8, completion_poll_steps=16,
    )
    assert tuple(tuple(row.tolist()) for row in result.responses) == expected
    assert result.response_limits == (6, 4, 5, 3)
    assert all(
        prompt.numel() + response.numel() == 8
        for prompt, response in zip(prompts, result.responses, strict=True)
    )
    # The short-budget lane is polled/refilled at four, not the batch max eight.
    assert active_rows[4] == (0, 2)


@pytest.mark.parametrize("prompt_length,reserve", [(8, 0), (9, 0), (5, 2)])
def test_context_boundary_rejects_before_prefix_or_decode(prompt_length, reserve) -> None:
    engine, active_rows = _scheduler_harness(((10,) * 8,), samples_per_prompt=1)
    engine.answer_reserve_tokens = reserve
    engine.thinking_end_token_id = 9 if reserve else None
    with pytest.raises(ValueError):
        engine.generate_prompt_pool(
            [torch.ones(prompt_length, dtype=torch.long)],
            max_new_tokens=8, context_tokens=8,
        )
    assert engine._prefix_calls == []
    assert active_rows == []


def test_remaining_budget_preserves_response_cap_and_legacy_omission() -> None:
    assert response_token_limits([2, 5], max_new_tokens=4, context_tokens=8) == (4, 3)
    assert response_token_limits([2, 5], max_new_tokens=4) == (4, 4)
    assert response_token_limits(
        [4], max_new_tokens=8, context_tokens=8,
        answer_reserve_tokens=2, thinking_end_token_id=9,
    ) == (4,)

def test_collection_does_not_mistake_short_budget_padding_for_generated_eos(monkeypatch):
    import postraining.train_minicpm_vapo as trainer

    engine, _ = _scheduler_harness(((20,) * 6, (30,) * 4), samples_per_prompt=1)
    engine.policy = SimpleNamespace(
        token_carry=False, causal_lm=SimpleNamespace(config=SimpleNamespace(vocab_size=130560)),
    )
    engine.top_k = -1
    engine.top_p = 1.0
    rows = [
        {"ids": torch.ones(length, dtype=torch.long),
         "reward_model": {"ground_truth": "42", "style": "rule"}}
        for length in (2, 4)
    ]
    monkeypatch.setattr(trainer, "encode_math_prompt", lambda tokenizer, row, **kwargs: row["ids"])
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "Answer: 42")
    result = trainer.collect_rollouts(
        engine, tokenizer, rows, prompt_tokens=4, max_new_tokens=8,
        context_tokens=8, enable_thinking=False,
    )
    assert [record.response_length for record in result.records] == [6, 4]
    assert [record.token_ids.numel() for record in result.records] == [8, 8]
    assert [int(record.token_ids[-1]) for record in result.records] == [20, 30]
    metrics = trainer.rollout_diagnostics(result, samples_per_prompt=1, stop_ids=(99,))
    assert metrics["truncation_fraction"] == 1.0
    assert metrics["domain/legacy_math/capped_trajectories"] == 2



def test_carry_history_disabled_keeps_only_live_state_after_cache_restore() -> None:
    engine = object.__new__(CapturedTrainingRolloutEngine)
    engine.token_carry = True
    engine.record_carry_history = False
    engine.batch_size = 2
    engine.cache_length = 8
    engine._runtime_device = torch.device("cpu")
    engine.policy = SimpleNamespace(
        causal_lm=SimpleNamespace(config=SimpleNamespace(hidden_size=4))
    )
    engine.carry_hidden = None
    engine._carry_history = None
    engine._new_cache = lambda: SimpleNamespace(layers=[])

    engine._allocate_carry_storage()

    torch.testing.assert_close(
        engine.carry_hidden, torch.zeros((2, 4), dtype=torch.bfloat16)
    )
    assert engine._carry_history is None
    engine.carry_hidden = None
    engine._rollout_resident = False

    engine._restore_rollout_cache()

    torch.testing.assert_close(
        engine.carry_hidden, torch.zeros((2, 4), dtype=torch.bfloat16)
    )
    assert engine._carry_history is None


def test_continuous_pool_does_not_export_disabled_carry_history() -> None:
    engine, _ = _scheduler_harness(
        ((99,), (20, 99), (30, 99), (99,)),
        samples_per_prompt=2,
    )
    engine.token_carry = True
    engine.record_carry_history = False
    engine.carry_hidden = torch.zeros((2, 4), dtype=torch.bfloat16)
    engine._carry_history = None

    result = engine.generate_prompt_pool(
        [torch.tensor([1]), torch.tensor([2])],
        max_new_tokens=4,
        completion_poll_steps=1,
    )

    assert result.carry_hiddens is None
    assert engine.last_carry_hiddens is None


def test_continuous_decode_omits_discarded_rollout_statistics() -> None:
    engine = object.__new__(CapturedTrainingRolloutEngine)
    engine._graph_logits = torch.tensor([[1.0, 2.0]])
    engine._graph_values = torch.tensor([13.0])
    engine.cache = object()
    engine.capture_logprobs = False
    engine.sample_tokens = lambda logits: torch.tensor([1])
    engine.sample = lambda *args: (_ for _ in ()).throw(
        AssertionError("log-probability-producing sampler must not run")
    )
    engine.decode = lambda *args: (_ for _ in ()).throw(
        AssertionError("value-producing decode must not run")
    )
    engine.decode_without_statistics = lambda token, cache: torch.tensor(
        [[3.0, 4.0]]
    )

    engine._continuous_split_decode_step()

    torch.testing.assert_close(engine._graph_logits, torch.tensor([[3.0, 4.0]]))
    torch.testing.assert_close(engine._graph_values, torch.tensor([13.0]))


def test_continuous_decode_replays_its_dedicated_graph() -> None:
    class Graph:
        def __init__(self) -> None:
            self.replays = 0

        def replay(self) -> None:
            self.replays += 1

    engine = object.__new__(CapturedTrainingRolloutEngine)
    legacy_graph = Graph()
    continuous_graph = Graph()
    engine._compile_decode = True
    engine._decode_graph = legacy_graph
    engine._continuous_decode_graph = continuous_graph

    engine._continuous_decode_once()

    assert legacy_graph.replays == 0
    assert continuous_graph.replays == 1




def test_completed_lanes_stop_scanning_stale_kv_history() -> None:
    lengths = torch.tensor([18, 9, 27, 4], dtype=torch.int32)
    active = torch.tensor([True, False, True, False])

    _retire_inactive_flash_rows_(lengths, active)

    assert lengths.tolist() == [18, 1, 27, 1]


def test_continuous_admission_preserves_inactive_state_without_replay_buffers() -> None:
    engine = object.__new__(CapturedTrainingRolloutEngine)
    engine.token_carry = True
    engine.record_carry_history = False
    engine.carry_hidden = torch.full((2, 4), -3.0, dtype=torch.bfloat16)
    engine._carry_history = None
    engine.generated = torch.ones((2, 5), dtype=torch.long)
    engine.logprobs = torch.ones((2, 5))
    engine.values = torch.full((2, 5), 7.0)
    engine._graph_logits = torch.zeros((2, 3))
    engine._graph_values = torch.full((2,), 11.0)
    engine.attention_mask = torch.zeros((2, 5), dtype=torch.bool)
    engine.sequence_lengths = torch.zeros(2, dtype=torch.long)
    engine.flash_sequence_lengths = torch.zeros(2, dtype=torch.int32)
    engine.position_ids = torch.zeros((2, 1), dtype=torch.long)
    engine.output_position = torch.ones(2, dtype=torch.long)
    engine.response_limit = torch.ones(2, dtype=torch.long)
    engine.active = torch.zeros(2, dtype=torch.bool)
    engine.thinking_closed = torch.ones_like(engine.active)
    engine.cache = SimpleNamespace(layers=[])
    bank = PromptPrefixBank(
        lengths=torch.tensor([2]),
        logits=torch.tensor([[1.0, 2.0, 3.0]]),
        values=torch.empty(0),
        layer_keys=torch.empty((1, 2, 0, 0, 0)),
        layer_values=torch.empty((1, 2, 0, 0, 0)),
        hidden=torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16),
    )

    engine._admit_prompt_rows(bank, 0, [1], max_new_tokens=3)

    torch.testing.assert_close(engine.values, torch.full((2, 5), 7.0))
    torch.testing.assert_close(engine._graph_values, torch.full((2,), 11.0))
    assert engine.active.tolist() == [False, True]
    assert engine.thinking_closed.tolist() == [True, False]
    torch.testing.assert_close(
        engine.carry_hidden,
        torch.tensor(
            [[-3.0, -3.0, -3.0, -3.0], [1.0, 2.0, 3.0, 4.0]],
            dtype=torch.bfloat16,
        ),
    )
    assert engine._carry_history is None


def test_public_prefix_bank_keeps_value_collection_contract() -> None:
    engine = object.__new__(CapturedTrainingRolloutEngine)
    sentinel = cast(PromptPrefixBank, object())
    calls: list[tuple[int, bool, str]] = []

    def build(
        prompt_ids_cpu,
        *,
        prefill_batch_prompts,
        collect_values,
        storage_device,
    ):
        del prompt_ids_cpu
        calls.append(
            (prefill_batch_prompts, collect_values, str(storage_device))
        )
        return sentinel

    engine._build_prompt_prefix_bank = build

    result = CapturedTrainingRolloutEngine.build_prompt_prefix_bank(
        engine,
        [torch.tensor([1])],
        prefill_batch_prompts=3,
    )

    assert result is sentinel
    assert calls == [(3, True, "cpu")]
