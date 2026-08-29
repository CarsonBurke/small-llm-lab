"""CPU contracts for Byte-Duo's exact uneven DDP update geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pretraining.byte_diffusion.duo import sample_branch_antithetic_times
from pretraining.byte_diffusion.training import TrainingBatch
from scripts.train_byte_duo import (
    _validation_batches,
    capture_rank_state,
    distributed_loss_scale,
    distributed_microstep_count,
    distributed_rank_positions,
    rank_checkpoint_path,
    prune_stale_rank_checkpoints,
    restore_rank_state,
)


class _CursorState:
    def __init__(self, value: int) -> None:
        self.value = value

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.value = state["value"]


class _IndexedValidationDataset:
    def __len__(self) -> int:
        return 6

    def training_batch(self, indices: np.ndarray) -> TrainingBatch:
        rows = len(indices)
        width = 8
        row_values = torch.from_numpy(indices.copy())[:, None]
        ids = row_values.expand(rows, width).clone()
        valid = torch.ones_like(ids, dtype=torch.bool)
        positions = torch.arange(width)[None].expand(rows, -1)
        documents = row_values.expand(rows, width).clone()
        return TrainingBatch(
            ids=ids,
            valid=valid,
            ar_targets=ids.clone(),
            bos_targets=ids[:, 0].clone(),
            positions=positions,
            full_valid=True,
            document_ids=documents,
            byte_indices=torch.arange(rows * width),
            byte_cu_seqlens=torch.arange(
                0, (rows + 1) * width, width, dtype=torch.int32
            ),
            patch_indices=torch.arange(rows * (width // 4)),
        )


def test_exact_249_rows_are_disjoint_exhaustive_and_remainder_rotates() -> None:
    first = [
        distributed_rank_positions(249, rank=rank, world_size=8, rotation=0)
        for rank in range(8)
    ]
    second = [
        distributed_rank_positions(249, rank=rank, world_size=8, rotation=1)
        for rank in range(8)
    ]
    np.testing.assert_array_equal(
        np.sort(np.concatenate(first)), np.arange(249, dtype=np.int64)
    )
    assert sum(map(len, first)) == 249
    assert len(np.unique(np.concatenate(first))) == 249
    assert tuple(map(len, first)) == (32, 31, 31, 31, 31, 31, 31, 31)
    assert tuple(map(len, second)) == (31, 32, 31, 31, 31, 31, 31, 31)


def test_microbatch_geometry_requires_equal_backward_collective_count() -> None:
    assert distributed_microstep_count(
        249, world_size=8, microbatch_size=32
    ) == 1
    assert distributed_microstep_count(
        249, world_size=8, microbatch_size=24
    ) == 2
    assert distributed_microstep_count(
        249, world_size=1, microbatch_size=24
    ) == 11
    with pytest.raises(ValueError, match="different backward-call counts"):
        distributed_microstep_count(249, world_size=8, microbatch_size=31)
    with pytest.raises(ValueError, match="cannot exceed"):
        distributed_microstep_count(3, world_size=4, microbatch_size=1)


def test_global_antithetic_time_ledger_is_sharded_not_resampled_per_rank() -> None:
    global_times = sample_branch_antithetic_times(
        249,
        8,
        device="cpu",
        generator=torch.Generator().manual_seed(151),
    )
    reconstructed = torch.empty_like(global_times)
    observed_positions: list[torch.Tensor] = []
    for rank in range(8):
        positions = torch.from_numpy(
            distributed_rank_positions(249, rank=rank, world_size=8, rotation=5)
        )
        observed_positions.append(positions)
        reconstructed.index_copy_(0, positions, global_times.index_select(0, positions))
    torch.testing.assert_close(reconstructed, global_times)
    assert torch.cat(observed_positions).unique().numel() == 249
    assert global_times.unique().numel() == 249 * 8


def test_validation_loader_preserves_sparse_global_row_ids_and_order() -> None:
    indices = np.array([5, 1, 3], dtype=np.int64)
    batches = tuple(
        _validation_batches(
            _IndexedValidationDataset(),
            rows=6,
            batch_size=2,
            patch_stride=4,
            device=torch.device("cpu"),
            row_indices=indices,
        )
    )
    assert [batch.row_ids.tolist() for batch in batches] == [[5, 1], [3]]
    assert torch.cat([batch.clean_ids[:, 0] for batch in batches]).tolist() == [5, 1, 3]


def test_world_scale_undoes_ddp_average_for_global_denominator() -> None:
    parameter = torch.tensor(0.7, requires_grad=True)
    local_coefficients = (torch.tensor(3.0), torch.tensor(11.0))
    global_denominator = torch.tensor(7.0)
    local_gradients = []
    for coefficient in local_coefficients:
        loss = (
            distributed_loss_scale(2)
            * coefficient
            * parameter
            / global_denominator
        )
        (gradient,) = torch.autograd.grad(loss, parameter, retain_graph=True)
        local_gradients.append(gradient)
    ddp_average = torch.stack(local_gradients).mean()
    exact_global = sum(local_coefficients) / global_denominator
    torch.testing.assert_close(ddp_average, exact_global)


def test_rank_sidecar_path_and_rng_resume_are_rank_local(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    assert (
        rank_checkpoint_path(checkpoint, 7, step=200).name
        == "checkpoint.step00000200.rank00007.pt"
    )

    for step in (100, 200, 300):
        rank_checkpoint_path(checkpoint, 7, step=step).touch()
    other_rank = rank_checkpoint_path(checkpoint, 8, step=100)
    other_rank.touch()
    prune_stale_rank_checkpoints(checkpoint, 7, committed_step=200)
    assert rank_checkpoint_path(checkpoint, 7, step=200).exists()
    assert not rank_checkpoint_path(checkpoint, 7, step=100).exists()
    assert not rank_checkpoint_path(checkpoint, 7, step=300).exists()
    assert other_rank.exists()

    torch.manual_seed(157)
    time_generator = torch.Generator().manual_seed(163)
    corruption_generator = torch.Generator().manual_seed(167)
    cursor = _CursorState(13)
    state = capture_rank_state(
        time_generator,
        corruption_generator,
        cursor=cursor,  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )
    expected = (
        torch.rand(3, generator=time_generator),
        torch.rand(3, generator=corruption_generator),
        torch.rand(3),
    )
    cursor.value = 99
    restore_rank_state(
        state,
        time_generator,
        corruption_generator,
        cursor=cursor,  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )
    observed = (
        torch.rand(3, generator=time_generator),
        torch.rand(3, generator=corruption_generator),
        torch.rand(3),
    )
    assert cursor.value == 13
    for left, right in zip(observed, expected, strict=True):
        torch.testing.assert_close(left, right)


def test_rank_checkpoint_uses_committed_host_snapshot_not_prefetched_future() -> None:
    torch.manual_seed(173)
    time_generator = torch.Generator().manual_seed(179)
    corruption_generator = torch.Generator().manual_seed(181)
    cursor = _CursorState(19)
    committed = {
        "time_generator_state": time_generator.get_state(),
        "corruption_generator_state": corruption_generator.get_state(),
        "cursor": cursor.state_dict(),
    }
    expected_time = torch.rand(4, generator=time_generator)
    expected_corruption = torch.rand(4, generator=corruption_generator)

    # Simulate the producer advancing mutable host streams through N+1 before
    # the main thread checkpoints completed update N.
    torch.rand(17, generator=time_generator)
    torch.rand(23, generator=corruption_generator)
    cursor.value = 20
    state = capture_rank_state(
        time_generator,
        corruption_generator,
        cursor=cursor,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        prepared_host_state=committed,
    )
    restore_rank_state(
        state,
        time_generator,
        corruption_generator,
        cursor=cursor,  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        torch.rand(4, generator=time_generator), expected_time
    )
    torch.testing.assert_close(
        torch.rand(4, generator=corruption_generator), expected_corruption
    )
    assert cursor.value == 19
