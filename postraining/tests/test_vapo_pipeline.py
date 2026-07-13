from __future__ import annotations

import torch

from postraining.core import TrajectoryBatch
from postraining.train_vapo import (
    concatenate_batches,
    crossed_interval,
    generate_group,
    rollout_diagnostics,
)


def trajectory_batch(
    targets: torch.Tensor,
    mask: torch.Tensor,
    correct: torch.Tensor,
    texts: list[str],
) -> TrajectoryBatch:
    rows, length = targets.shape
    rewards = torch.zeros(rows, length)
    last = length - 1 - mask.flip(1).argmax(1)
    rewards[torch.arange(rows), last] = torch.where(correct, 1.0, -1.0)
    return TrajectoryBatch(
        input_ids=torch.arange(rows * length).reshape(rows, length),
        target_ids=targets,
        response_mask=mask,
        old_logprobs=torch.zeros(rows, length),
        old_values=torch.zeros(rows, length),
        rewards=rewards,
        correct=correct,
        texts=texts,
    )


def test_cpu_rollout_shards_are_padded_and_concatenated_in_group_order():
    first = trajectory_batch(
        torch.tensor([[9, 2, 0], [9, 8, 2]]),
        torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32),
        torch.tensor([True, False]),
        ["a", "b"],
    )
    second = trajectory_batch(
        torch.tensor([[7, 8], [7, 8]]),
        torch.ones(2, 2),
        torch.tensor([False, False]),
        ["c", "d"],
    )

    combined = concatenate_batches([first, second], pad_id=99)

    assert combined.input_ids.shape == (4, 3)
    assert combined.target_ids[2:, -1].tolist() == [99, 99]
    assert combined.response_mask[2:, -1].tolist() == [0, 0]
    assert combined.correct.tolist() == [True, False, False, False]
    assert combined.texts == ["a", "b", "c", "d"]


def test_rollout_gate_reports_positive_groups_eos_and_length_distribution():
    first = trajectory_batch(
        torch.tensor([[9, 2, 0], [9, 8, 2]]),
        torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32),
        torch.tensor([True, False]),
        ["a", "b"],
    )
    second = trajectory_batch(
        torch.tensor([[7, 8], [7, 8]]),
        torch.ones(2, 2),
        torch.tensor([False, False]),
        ["c", "d"],
    )
    batch = concatenate_batches([first, second], pad_id=99)

    metrics = rollout_diagnostics(batch, samples_per_prompt=2, eos_id=2)

    assert metrics["positive_trajectories"] == 1
    assert metrics["positive_groups"] == 1
    assert metrics["positive_group_fraction"] == 0.5
    assert metrics["eos_fraction"] == 0.5
    assert metrics["truncation_fraction"] == 0.5
    assert metrics["response_length_max"] == 3
    assert metrics["advantage_min"] <= metrics["advantage_mean"] <= metrics["advantage_max"]


def test_periodic_actions_are_deferred_to_completed_rollout_boundaries():
    assert not crossed_interval(64, 96, 100)
    assert crossed_interval(96, 128, 100)
    assert crossed_interval(0, 128, 100)
    assert not crossed_interval(96, 128, 0)


class FakeGenerationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def make_generation_cache(self, batch_size, max_length, device):
        return []

    def generation_step(self, token_ids, caches, position):
        logits = torch.tensor([0.0, 1.0, 2.0, 3.0], device=token_ids.device).repeat(
            token_ids.size(0), 1
        )
        values = torch.full(
            (token_ids.size(0),), float(position), device=token_ids.device
        )
        return logits, values, caches


class FakeTokenizer:
    def bos_id(self):
        return 0

    def eos_id(self):
        return -1

    def encode(self, prompt):
        return [1]


def test_generation_retains_aligned_action_logprobs_and_pre_action_values():
    torch.manual_seed(42)
    model, tokenizer = FakeGenerationModel(), FakeTokenizer()
    _, responses, old_logprobs, old_values = generate_group(
        model, tokenizer, "prompt", samples=2, max_new_tokens=3,
        temperature=1.0, top_p=1.0,
    )
    expected = torch.tensor([0.0, 1.0, 2.0, 3.0]).log_softmax(0)
    torch.testing.assert_close(old_logprobs, expected[responses])
    torch.testing.assert_close(
        old_values, torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    )


def test_generation_can_skip_training_statistics_for_evaluation():
    model, tokenizer = FakeGenerationModel(), FakeTokenizer()
    _, _, old_logprobs, old_values = generate_group(
        model, tokenizer, "prompt", samples=2, max_new_tokens=2,
        temperature=1.0, top_p=1.0, capture_stats=False,
    )
    assert old_logprobs is None
    assert old_values is None
