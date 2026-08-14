"""Transactional semantic and tensor-cache state for byte decoding."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class WorkCounters:
    forwards: int = 0
    denoise_forwards: int = 0
    prefix_prefills: int = 0
    causal_replays: int = 0
    cache_appends: int = 0
    proposed_bytes: int = 0
    committed_bytes: int = 0
    rejected_bytes: int = 0


@dataclass
class CacheSnapshot:
    ids: Tensor
    tensors: dict[str, Tensor]
    generator_state: Tensor
    patch_phase: int
    terminal_eot: bool


@dataclass
class TransactionalDecodeState:
    """Committed state plus isolated scratch buffers.

    Scratch K/V is intentionally never promoted. A successful commit appends
    semantic ids and requires the caller to replay through the causal base into
    committed caches, which preserves the exact anchor distribution.
    """

    eot_id: int
    patch_stride: int = 4
    device: torch.device | str = "cpu"
    ids: Tensor = field(init=False)
    caches: dict[str, Tensor] = field(default_factory=dict)
    counters: WorkCounters = field(default_factory=WorkCounters)
    _generator: torch.Generator = field(init=False, repr=False)
    _scratch: CacheSnapshot | None = field(default=None, init=False, repr=False)
    _patch_phase: int = field(default=0, init=False, repr=False)
    _terminal_eot: bool = field(default=False, init=False, repr=False)
    _phase_storage: tuple[int, int, int] = field(
        default=(0, 0, -1), init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.ids = torch.empty(0, dtype=torch.long, device=self.device)
        self._generator = torch.Generator(device=self.device)

    @property
    def patch_phase(self) -> int:
        # Normal decode commits maintain this scalar incrementally. Refreshing
        # is needed only for direct external tensor replacement (including old
        # checkpoints/tests), never by rescanning the prefix on every action.
        # Tensor identity and length do not change under an in-place edit.
        # Include PyTorch's mutation counter so direct callers cannot leave a
        # stale EOT/patch phase behind after editing the exposed prefix tensor.
        storage = (self.ids.data_ptr(), self.ids.numel(), self.ids._version)
        if storage != self._phase_storage:
            eot = self.ids.eq(self.eot_id).nonzero().flatten()
            terminal = bool(eot.numel() and int(eot[-1]) == self.ids.numel() - 1)
            last_eot = int(eot[-1]) if eot.numel() else -1
            self._terminal_eot = terminal
            self._patch_phase = (
                0
                if terminal
                else (self.ids.numel() - last_eot - 1) % self.patch_stride
            )
            self._phase_storage = storage
        return self._patch_phase

    @property
    def incomplete_patch_buffer(self) -> Tensor:
        """Committed post-EOT suffix not yet closing a four-byte patch."""

        phase = self.patch_phase
        return self.ids[-phase:].clone() if phase else self.ids.new_empty((0,))

    def seed(self, value: int) -> None:
        if self._scratch is not None:
            raise RuntimeError("cannot reseed an open transaction")
        self._generator.manual_seed(value)

    def replace_prefix(self, ids: Tensor) -> None:
        """Install a new committed prompt and invalidate all replayed caches."""

        if self._scratch is not None:
            raise RuntimeError("cannot replace the prefix during a transaction")
        if ids.ndim != 1 or ids.dtype != torch.long or ids.numel() == 0:
            raise ValueError("decode prefix must be nonempty rank-1 int64")
        self.ids = ids.clone().to(self.device)
        self._phase_storage = (0, 0, -1)
        _ = self.patch_phase
        self.caches.clear()

    @property
    def generator(self) -> torch.Generator:
        """The sole RNG stream used by speculative and diffusion decoding."""

        return self._generator

    def snapshot(self) -> CacheSnapshot:
        _ = self.patch_phase
        return CacheSnapshot(
            # A public transaction must also isolate in-place edits made by a
            # caller. The production generator keeps its large LayerKV stores
            # outside this legacy tensor dictionary, so preserving the actual
            # rollback contract here is cheap in the optimized decode path.
            ids=self.ids.clone(),
            tensors={name: value.clone() for name, value in self.caches.items()},
            generator_state=self._generator.get_state().clone(),
            patch_phase=self._patch_phase,
            terminal_eot=self._terminal_eot,
        )

    def begin(self) -> CacheSnapshot:
        if self._scratch is not None:
            raise RuntimeError("nested decode transactions are forbidden")
        self._scratch = self.snapshot()
        return self._scratch

    def note_work(
        self, proposed_bytes: int, *, forwards: int = 1, denoise: bool = False
    ) -> None:
        if proposed_bytes < 0 or forwards < 0:
            raise ValueError("work counts cannot be negative")
        self.counters.forwards += forwards
        if denoise:
            self.counters.denoise_forwards += forwards
        self.counters.proposed_bytes += proposed_bytes

    def note_prefill(self, *, replay: bool) -> None:
        self.counters.forwards += 1
        if replay:
            self.counters.causal_replays += 1
        else:
            self.counters.prefix_prefills += 1

    def note_cache_append(self) -> None:
        """Count one incremental clean encoder/global/decoder replay."""

        self.counters.forwards += 1
        self.counters.cache_appends += 1

    def abort(self) -> None:
        if self._scratch is None:
            raise RuntimeError("no transaction is open")
        snapshot = self._scratch
        self.ids = snapshot.ids
        self.caches = snapshot.tensors
        self._generator.set_state(snapshot.generator_state)
        self._patch_phase = snapshot.patch_phase
        self._terminal_eot = snapshot.terminal_eot
        self._phase_storage = (
            self.ids.data_ptr(),
            self.ids.numel(),
            self.ids._version,
        )
        self._scratch = None

    def commit_ids(self, proposed: Tensor) -> Tensor:
        """Commit through first EOT; caller must causally replay returned ids."""

        if self._scratch is None:
            raise RuntimeError("commit requires an open transaction")
        if proposed.ndim != 1 or proposed.dtype != torch.long:
            raise ValueError("proposed ids must be a rank-1 int64 tensor")
        accepted = proposed
        eot = proposed.eq(self.eot_id).nonzero()
        if eot.numel():
            accepted = proposed[: int(eot[0]) + 1]
        self.ids = torch.cat((self.ids, accepted))
        self._terminal_eot = bool(eot.numel())
        self._patch_phase = (
            0
            if self._terminal_eot
            else (self._scratch.patch_phase + accepted.numel()) % self.patch_stride
        )
        self._phase_storage = (
            self.ids.data_ptr(),
            self.ids.numel(),
            self.ids._version,
        )
        self.counters.committed_bytes += accepted.numel()
        # Scratch tensors are never promoted; causal replay populates cache.
        # The RNG is deliberately *not* restored on commit: accepted sampling
        # consumed that stream.  Abort restores it to make a rejected proposal
        # observationally invisible.
        self.caches = self._scratch.tensors
        self._scratch = None
        return accepted

    def install_replayed_cache(self, name: str, value: Tensor) -> None:
        if self._scratch is not None:
            raise RuntimeError("cannot mutate committed cache during speculation")
        self.caches[name] = value

    def state_dict(self) -> dict[str, object]:
        if self._scratch is not None:
            raise RuntimeError("cannot checkpoint an open transaction")
        return {
            "ids": self.ids.clone(),
            "caches": {name: value.clone() for name, value in self.caches.items()},
            "generator_state": self._generator.get_state(),
            "counters": vars(self.counters).copy(),
            "eot_id": self.eot_id,
            "patch_stride": self.patch_stride,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state["eot_id"]) != self.eot_id or int(state["patch_stride"]) != self.patch_stride:
            raise ValueError("decode-state schema mismatch")
        self.ids = state["ids"].to(self.device)  # type: ignore[union-attr]
        self._phase_storage = (0, 0, -1)
        _ = self.patch_phase
        self.caches = {
            name: value.to(self.device)
            for name, value in state["caches"].items()  # type: ignore[union-attr]
        }
        self._generator.set_state(state["generator_state"])  # type: ignore[arg-type]
        self.counters = WorkCounters(**state["counters"])  # type: ignore[arg-type]
