from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from postraining.fast_inference import (
    CapturedTrainingRolloutEngine,
    PromptPrefixBank,
    _retire_inactive_flash_rows_,
)


def _scheduler_harness(
    responses: tuple[tuple[int, ...], ...],
    *,
    samples_per_prompt: int,
) -> tuple[CapturedTrainingRolloutEngine, list[tuple[int, ...]]]:
    batch_size = 2
    cache_length = 8
    prompt_count = len(responses) // samples_per_prompt
    engine = object.__new__(CapturedTrainingRolloutEngine)
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
    engine.response_limit = torch.ones(batch_size, dtype=torch.long)
    engine.position_ids = torch.zeros((batch_size, 1), dtype=torch.long)
    engine.sequence_lengths = torch.zeros(batch_size, dtype=torch.long)
    engine.flash_sequence_lengths = torch.zeros(batch_size, dtype=torch.int32)
    engine.attention_mask = torch.zeros(
        (batch_size, cache_length), dtype=torch.bool
    )
    engine._compile_decode = False
    engine._continuous_decode_graph = None
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
    torch.testing.assert_close(
        engine.logprobs,
        torch.full((engine.batch_size, engine.cache_length), -5.0),
    )
    assert active_rows_by_step[:2] == [(0, 1), (2, 1)]
    assert result.minimum_active_rows_with_backlog == engine.batch_size
    assert engine._prefix_calls == [(False, "cpu")]


def test_continuous_decode_omits_discarded_rollout_statistics() -> None:
    engine = object.__new__(CapturedTrainingRolloutEngine)
    engine._graph_logits = torch.tensor([[1.0, 2.0]])
    engine._graph_values = torch.tensor([13.0])
    engine.cache = object()
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


def test_continuous_admission_leaves_legacy_statistic_buffers_untouched() -> None:
    engine = object.__new__(CapturedTrainingRolloutEngine)
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
    engine.cache = SimpleNamespace(layers=[])
    bank = PromptPrefixBank(
        lengths=torch.tensor([2]),
        logits=torch.tensor([[1.0, 2.0, 3.0]]),
        values=torch.empty(0),
        layer_keys=torch.empty((1, 2, 0, 0, 0)),
        layer_values=torch.empty((1, 2, 0, 0, 0)),
    )

    engine._admit_prompt_rows(bank, 0, [1], max_new_tokens=3)

    torch.testing.assert_close(engine.values, torch.full((2, 5), 7.0))
    torch.testing.assert_close(engine._graph_values, torch.full((2,), 11.0))
    assert engine.active.tolist() == [False, True]


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
