from __future__ import annotations

from pathlib import Path

import pytest
import torch

from checkpointing import (
    RecoveryCheckpointPolicy,
    atomic_link_or_copy,
    atomic_torch_save,
    validate_checkpoint_interval_seconds,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_recovery_deadline_uses_elapsed_monotonic_time_and_terminal_state() -> None:
    clock = FakeClock()
    policy = RecoveryCheckpointPolicy(480, clock=clock)

    clock.now = 479.9
    assert not policy.due()
    clock.now = 480.0
    assert policy.due()

    policy.committed((12, 34))
    assert not policy.terminal_due((12, 34))
    assert policy.terminal_due((13, 35))
    clock.now = 959.9
    assert not policy.due()
    clock.now = 960.0
    assert policy.due()


@pytest.mark.parametrize("seconds", [299.9, 600.1, float("nan"), float("inf")])
def test_recovery_deadline_rejects_out_of_policy_intervals(seconds: float) -> None:
    with pytest.raises(ValueError, match="between 300 and 600"):
        validate_checkpoint_interval_seconds(seconds)


def test_atomic_publication_replaces_existing_artifacts(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    atomic_torch_save({"step": 1}, checkpoint)
    atomic_torch_save({"step": 2}, checkpoint)
    assert torch.load(checkpoint, weights_only=True) == {"step": 2}

    source = tmp_path / "source.pt"
    source.write_bytes(b"final weights")
    destination = tmp_path / "published.pt"
    destination.write_bytes(b"old")
    atomic_link_or_copy(source, destination)
    assert destination.read_bytes() == source.read_bytes()
