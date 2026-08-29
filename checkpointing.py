"""Shared wall-clock policy and publication helpers for recovery checkpoints."""

from __future__ import annotations

from collections.abc import Callable
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

import torch


DEFAULT_CHECKPOINT_INTERVAL_SECONDS = 480.0
MIN_CHECKPOINT_INTERVAL_SECONDS = 300.0
MAX_CHECKPOINT_INTERVAL_SECONDS = 600.0


def validate_checkpoint_interval_seconds(value: float) -> float:
    """Return a production-safe recovery interval or raise ``ValueError``."""

    interval = float(value)
    if (
        not math.isfinite(interval)
        or not MIN_CHECKPOINT_INTERVAL_SECONDS
        <= interval
        <= MAX_CHECKPOINT_INTERVAL_SECONDS
    ):
        raise ValueError(
            "checkpoint interval must be between "
            f"{MIN_CHECKPOINT_INTERVAL_SECONDS:g} and "
            f"{MAX_CHECKPOINT_INTERVAL_SECONDS:g} seconds"
        )
    return interval


class RecoveryCheckpointPolicy:
    """Gate exact-recovery commits by a monotonic deadline.

    Callers check :meth:`due` only at exact-resume boundaries and call
    :meth:`committed` after atomic publication. A terminal commit is needed
    exactly when :meth:`terminal_due` reports that the completed state differs
    from the last committed one.
    """

    def __init__(
        self,
        interval_seconds: float = DEFAULT_CHECKPOINT_INTERVAL_SECONDS,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.interval_seconds = validate_checkpoint_interval_seconds(
            interval_seconds
        )
        self._clock = clock
        self._deadline = self._clock() + self.interval_seconds
        self.last_committed_state: object | None = None

    def due(self) -> bool:
        return self._clock() >= self._deadline

    def committed(self, state: object) -> None:
        self.last_committed_state = state
        self._deadline = self._clock() + self.interval_seconds

    def terminal_due(self, state: object) -> bool:
        return state != self.last_committed_state


def atomic_torch_save(payload: Any, path: Path) -> None:
    """Publish a torch payload with same-directory staging and replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.working")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_link_or_copy(source: Path, destination: Path) -> None:
    """Atomically publish an immutable artifact without a normal-path copy."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.working"
    )
    temporary.unlink(missing_ok=True)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
