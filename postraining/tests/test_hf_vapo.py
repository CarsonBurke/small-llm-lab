from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from postraining.core import generalized_advantage_estimate, length_adaptive_lambda
from postraining.hf_vapo import (
    LoRAConfig,
    MiniCPMVAPOPolicy,
    StaticCachePool,
    TrajectoryRecord,
    _nucleus_membership,
    chunked_frozen_head_logprobs,
    collate_replay_microbatch,
    exact_top_p_sample,
    inject_lora,
    plan_replay_microbatches,
    replay_storage_bytes,
)
from postraining.runtime.profiling import DeviceSampler
from postraining.train_hf_vapo import (
    RolloutEngine,
    _validate_args,
    _approximate_kl_terms,
    _build_group_records,
    build_parser,
    device_phase_metrics,
    static_kv_cache_bytes,
    tensorboard_scalars,
)
from postraining.validate_hf_vapo import prompt_ids


class _TinyLlamaBlock(nn.Module):
    def __init__(self, width: int = 6) -> None:
        super().__init__()
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            setattr(self, name, nn.Linear(width, width, bias=False))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = inputs
        for module in self.children():
            result = module(result)
        return result


def test_lora_injection_is_initially_exact_and_freezes_base() -> None:
    torch.manual_seed(1)
    model = _TinyLlamaBlock()
    inputs = torch.randn(3, 6)
    expected = model(inputs)
    names = inject_lora(model, LoRAConfig(rank=2, alpha=4))
    actual = model(inputs)
    assert len(names) == 7
    assert torch.equal(actual, expected)
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert len(trainable) == 14
    assert all(name.endswith(("lora_a", "lora_b")) for name in trainable)


def test_chunked_frozen_head_logprobs_match_dense_values_and_gradients() -> None:
    torch.manual_seed(2)
    hidden = torch.randn(11, 5, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    weight = torch.randn(17, 5)
    targets = torch.randint(0, 17, (11,))
    coefficients = torch.randn(11)

    actual = chunked_frozen_head_logprobs(
        hidden, targets, weight, chunk_tokens=3
    )
    expected = torch.log_softmax(reference_hidden @ weight.T, dim=-1).gather(
        1, targets[:, None]
    ).squeeze(1)
    (actual * coefficients).sum().backward()
    (expected * coefficients).sum().backward()

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert hidden.grad is not None
    assert reference_hidden.grad is not None
    assert torch.allclose(hidden.grad, reference_hidden.grad, atol=2e-6, rtol=2e-6)


def test_exact_top_p_sampler_prices_selected_token_under_full_policy() -> None:
    logits = torch.tensor(
        [[8.0, 7.0, 6.0, -9.0, -10.0], [4.0, 3.0, 2.0, 1.0, -8.0]]
    )
    sampled, logprobs, stats = exact_top_p_sample(
        logits,
        temperature=0.9,
        top_p=0.8,
        generator=torch.Generator().manual_seed(3),
    )
    expected = torch.log_softmax(logits, dim=-1).gather(
        1, sampled[:, None]
    ).squeeze(1)
    assert stats.scanned_vocabulary == 5
    assert stats.nucleus_mass_lower_bound == pytest.approx(0.8)
    assert torch.allclose(logprobs, expected)


def test_nucleus_membership_uses_deterministic_token_id_tie_breaking() -> None:
    logits = torch.tensor(
        [[3.0, 2.0, 1.0], [1.0, 1.0, 0.0]]
    )
    probabilities = logits.softmax(dim=-1)
    assert _nucleus_membership(
        logits[:1].expand(3, -1),
        probabilities[:1].expand(3, -1),
        torch.tensor([0, 1, 2]),
        0.8,
    ).tolist() == [True, True, False]
    assert _nucleus_membership(
        logits[1:].expand(2, -1),
        probabilities[1:].expand(2, -1),
        torch.tensor([0, 1]),
        0.4,
    ).tolist() == [True, False]


def test_exact_top_p_rejection_sampler_never_leaves_the_nucleus() -> None:
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]]).expand(4_096, -1)
    sampled, _, _ = exact_top_p_sample(
        logits,
        temperature=1.0,
        top_p=0.8,
        generator=torch.Generator().manual_seed(7),
    )
    assert set(sampled.tolist()) == {0, 1}
    expected_zero_fraction = torch.softmax(logits[0, :2], dim=0)[0]
    observed_zero_fraction = (sampled == 0).float().mean()
    assert observed_zero_fraction == pytest.approx(
        float(expected_zero_fraction), abs=0.03
    )


def test_multi_prompt_preparation_forms_one_left_padded_batch() -> None:
    class Config:
        pad_token_id = 9

    class CausalLM:
        config = Config()

    class Policy:
        causal_lm = CausalLM()

    engine = object.__new__(RolloutEngine)
    engine.policy = cast(MiniCPMVAPOPolicy, Policy())
    engine.prompts_per_rollout = 2
    engine.samples_per_prompt = 2
    engine.batch_size = 4
    engine.cache_length = 8
    engine.generated_buffer = torch.empty(4, 8)
    engine.attention_mask_buffer = torch.zeros(4, 8, dtype=torch.bool)
    engine.prompt_lengths_buffer = torch.empty(4, dtype=torch.long)
    prompt_batch, position_ids, width = engine._prepare_prompts(
        [torch.tensor([1, 2, 3]), torch.tensor([4])]
    )
    assert width == 3
    assert prompt_batch.tolist() == [
        [1, 2, 3],
        [1, 2, 3],
        [9, 9, 4],
        [9, 9, 4],
    ]
    assert position_ids.tolist() == [
        [0, 1, 2],
        [0, 1, 2],
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert engine.attention_mask_buffer[:, :3].tolist() == [
        [True, True, True],
        [True, True, True],
        [False, False, True],
        [False, False, True],
    ]
    assert engine.prompt_lengths_buffer.tolist() == [3, 3, 1, 1]

def test_ppo_approximate_kl_is_pointwise_non_negative() -> None:
    log_ratio = torch.tensor([-2.0, -0.1, 0.0, 0.2, 3.0])
    terms = _approximate_kl_terms(log_ratio)
    assert torch.all(terms >= 0)
    assert terms[2] == 0


def _record(length: int, prompt_length: int, correct: bool = False) -> TrajectoryRecord:
    response = length - prompt_length
    return TrajectoryRecord(
        token_ids=torch.arange(length, dtype=torch.int32),
        prompt_length=prompt_length,
        old_logprobs=torch.linspace(-2, -1, response, dtype=torch.float32),
        advantages=torch.zeros(response, dtype=torch.float32),
        correct=correct,
        text="answer",
    )


def test_replay_plan_and_collation_preserve_variable_length_actions() -> None:
    records = [_record(9, 4, True), _record(6, 3), _record(8, 6)]
    plan = plan_replay_microbatches(
        records, [2, 0, 1], token_budget=16, max_trajectories=2
    )
    assert sorted(index for batch in plan for index in batch) == [0, 1, 2]
    assert all(
        max(records[index].input_length for index in batch) * len(batch) <= 16
        for batch in plan
    )

    batch = collate_replay_microbatch(
        records,
        [0, 1],
        pad_token_id=99,
        correct_denominator=1,
        device=torch.device("cpu"),
    )
    assert batch.input_ids.shape == (2, 8)
    assert batch.attention_mask is not None
    assert batch.action_count == records[0].response_length + records[1].response_length
    assert batch.targets.tolist() == [4, 5, 6, 7, 8, 3, 4, 5]
    assert batch.action_positions.tolist() == [3, 4, 5, 6, 7, 2, 3, 4]
    assert batch.response_state_mask.tolist() == [
        [False, False, False, True, True, True, True, True],
        [False, False, True, True, True, False, False, False],
    ]
    assert batch.positive_weights[:5].sum() == pytest.approx(1.0)
    assert batch.positive_weights[5:].sum() == 0
    assert replay_storage_bytes(records) == sum(record.storage_bytes for record in records)


def test_replay_plan_rejects_trajectory_above_memory_budget() -> None:
    records = [_record(20, 4)]
    with pytest.raises(ValueError, match="exceeds token budget"):
        plan_replay_microbatches(
            records, [0], token_budget=12, max_trajectories=1
        )


def test_rollout_precomputes_gae_without_replay_time_scan() -> None:
    values = torch.tensor([0.1, -0.2, 0.3, 0.4])
    record = TrajectoryRecord.from_device(
        token_ids=torch.arange(7),
        prompt_length=3,
        old_logprobs=torch.full((4,), -2.0),
        old_values=values,
        correct=True,
        text="answer",
    )
    rewards = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    expected, _ = generalized_advantage_estimate(
        rewards,
        values[None],
        torch.ones_like(rewards),
        length_adaptive_lambda(torch.tensor([4.0])),
    )
    assert torch.allclose(record.advantages, expected[0], atol=1e-6)


def test_replay_record_rejects_statistic_length_mismatch() -> None:
    with pytest.raises(ValueError, match="log-probability"):
        TrajectoryRecord(
            token_ids=torch.arange(5, dtype=torch.int32),
            prompt_length=3,
            old_logprobs=torch.zeros(1, dtype=torch.float32),
            advantages=torch.zeros(2, dtype=torch.float32),
            correct=False,
            text="",
        )


def test_static_cache_pool_resets_between_prompt_groups() -> None:
    class Cache:
        def __init__(self) -> None:
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1

    created: list[Cache] = []

    def factory() -> Cache:
        cache = Cache()
        created.append(cache)
        return cache

    pool = StaticCachePool(factory, batch_size=4)
    first = pool.acquire(4)
    second = pool.acquire(4)
    assert first is second
    assert len(created) == 1
    assert second.reset_calls == 1
    assert pool.reset_count == 1
    pool.clear()
    third = pool.acquire(4)
    assert third is not first
    assert len(created) == 2
    assert pool.reset_count == 1
    with pytest.raises(ValueError, match="batch size"):
        pool.acquire(2)


def test_training_config_bounds_parallel_rollout_context() -> None:
    args = build_parser().parse_args([])
    _validate_args(args)
    assert args.thinking is True
    assert args.temperature == 0.9
    assert args.prompts_per_rollout == 4
    assert args.samples_per_prompt == 16
    assert args.max_new_tokens == 4_096
    assert args.replay_token_budget == 17_408
    assert args.value_warmup_steps == 10
    args.max_new_tokens = 18_000
    with pytest.raises(ValueError, match="replay token budget"):
        _validate_args(args)

def test_static_kv_cache_estimate_matches_minicpm_layout() -> None:
    class Config:
        num_hidden_layers = 24
        num_key_value_heads = 2
        head_dim = 128

    expected = 64 * 5_120 * 24 * 2 * 2 * 128 * 2
    assert static_kv_cache_bytes(
        Config(),
        batch_size=64,
        cache_length=5_120,
    ) == expected


def test_device_phase_metrics_keep_only_numeric_samples() -> None:
    class Sampler:
        def window(self, windows, power_floor):
            assert windows == [(1.0, 3.0)]
            assert power_floor == 400.0
            return {
                "power_draw_watts_mean": 512.5,
                "utilization_gpu_percent_readings": 7,
                "withheld": ["clocks_sm_mhz"],
            }

    assert device_phase_metrics(
        cast(DeviceSampler, Sampler()),
        started=1.0,
        ended=3.0,
        power_floor=400.0,
        prefix="rollout_",
    ) == {
        "rollout_device_power_draw_watts_mean": 512.5,
        "rollout_device_utilization_gpu_percent_readings": 7,
    }


def test_group_scoring_trims_stop_tokens_before_compacting_replay() -> None:
    class Tokenizer:
        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens
            return r"Answer: \boxed{42}"

    records, token_count = _build_group_records(
        Tokenizer(),
        {"reward_model": {"ground_truth": "42", "style": "minerva"}},
        torch.tensor([10, 11]),
        torch.tensor([[20, 99, 21], [22, 23, 24]]),
        torch.full((2, 3), -1.5),
        torch.zeros((2, 3)),
        samples_per_prompt=2,
        stop_ids=(99,),
    )
    assert token_count == 5
    assert [record.response_length for record in records] == [2, 3]
    assert [record.token_ids.tolist() for record in records] == [
        [10, 11, 20, 99],
        [10, 11, 22, 23, 24],
    ]
    assert all(record.correct for record in records)


def test_parity_prompt_ids_accept_transformers_batch_encoding() -> None:
    class Batch:
        def __getitem__(self, name):
            assert name == "input_ids"
            return torch.tensor([[1, 2, 3]])

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return Batch()

    assert prompt_ids(
        Tokenizer(), "question", torch.device("cpu")
    ).tolist() == [[1, 2, 3]]


def test_tensorboard_scalars_flush_live_cards(tmp_path) -> None:
    writer = SummaryWriter(tmp_path)
    tensorboard_scalars(
        writer,
        "rollout_live",
        {"decode_steps": 256, "non_numeric": "ignored"},
        256,
    )
    writer.close()

    events = EventAccumulator(str(tmp_path))
    events.Reload()
    assert "rollout_live/decode_steps" in events.Tags()["scalars"]
    point = events.Scalars("rollout_live/decode_steps")[0]
    assert point.step == 256
    assert point.value == 256
