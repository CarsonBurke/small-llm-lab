"""Additive training and generation metrics for byte diffusion.

Counters store sums and integer denominators, never batch-local means.  This
makes their states safe to merge across microbatches or distributed ranks
without reweighting examples by padding or corruption count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from .data import AtomicIdManifest, BYTE_COUNT


LOG_TWO = math.log(2.0)


def _safe_mean(total: float, count: int) -> float:
    return total / count if count else 0.0


@dataclass
class LossAccuracySums:
    """Raw negative-log-likelihood, correct predictions, and target count."""

    nll: float = 0.0
    correct: int = 0
    count: int = 0

    def add(self, *, nll: float, correct: int, count: int) -> None:
        if count < 0 or not 0 <= correct <= count:
            raise ValueError("loss/accuracy counts are invalid")
        if not math.isfinite(nll) or nll < 0.0:
            raise ValueError(f"NLL must be finite and non-negative, got {nll}")
        self.nll += float(nll)
        self.correct += int(correct)
        self.count += int(count)

    def merge(self, other: "LossAccuracySums") -> None:
        self.add(nll=other.nll, correct=other.correct, count=other.count)

    def state_dict(self) -> dict[str, float | int]:
        return {"nll": self.nll, "correct": self.correct, "count": self.count}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "LossAccuracySums":
        result = cls()
        result.add(
            nll=float(state["nll"]),
            correct=int(state["correct"]),
            count=int(state["count"]),
        )
        return result


@dataclass
class ARMetricCounter:
    """Additive shifted-AR NLL and atomic-byte accounting."""

    all_targets: LossAccuracySums = field(default_factory=LossAccuracySums)
    literal_targets: LossAccuracySums = field(default_factory=LossAccuracySums)
    special_targets: LossAccuracySums = field(default_factory=LossAccuracySums)

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        score_mask: torch.Tensor,
        manifest: AtomicIdManifest,
    ) -> None:
        """Add logits with shape ``targets.shape + (261,)``.

        Every scored clean atomic id contributes one unit to the repository's
        byte-accounting denominator.  PAD and MASK may be present only where
        ``score_mask`` is false.
        """

        if logits.shape[:-1] != targets.shape:
            raise ValueError("AR logits and target shapes disagree")
        if logits.shape[-1] != manifest.output_size:
            raise ValueError(
                f"AR logits require {manifest.output_size} clean classes, "
                f"got {logits.shape[-1]}"
            )
        if score_mask.shape != targets.shape or score_mask.dtype != torch.bool:
            raise ValueError("score_mask must be bool with the target shape")
        selected_targets = targets[score_mask].to(torch.long)
        if selected_targets.numel() == 0:
            return
        if bool(
            ((selected_targets < 0) | (selected_targets >= manifest.output_size))
            .any()
            .item()
        ):
            raise ValueError("MASK, PAD, or invalid ids cannot be AR targets")

        selected_logits = logits[score_mask].to(torch.float32)
        losses = F.cross_entropy(
            selected_logits, selected_targets, reduction="none"
        )
        predictions = selected_logits.argmax(dim=-1)
        correct = predictions.eq(selected_targets)
        literal = selected_targets < BYTE_COUNT
        self._add_partition(self.all_targets, losses, correct, None)
        self._add_partition(self.literal_targets, losses, correct, literal)
        self._add_partition(self.special_targets, losses, correct, ~literal)

    @staticmethod
    def _add_partition(
        counter: LossAccuracySums,
        losses: torch.Tensor,
        correct: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> None:
        if mask is not None:
            losses = losses[mask]
            correct = correct[mask]
        counter.add(
            nll=float(losses.sum(dtype=torch.float64).item()),
            correct=int(correct.sum().item()),
            count=losses.numel(),
        )

    def merge(self, other: "ARMetricCounter") -> None:
        self.all_targets.merge(other.all_targets)
        self.literal_targets.merge(other.literal_targets)
        self.special_targets.merge(other.special_targets)

    def state_dict(self) -> dict[str, Any]:
        return {
            "all_targets": self.all_targets.state_dict(),
            "literal_targets": self.literal_targets.state_dict(),
            "special_targets": self.special_targets.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ARMetricCounter":
        return cls(
            all_targets=LossAccuracySums.from_state_dict(state["all_targets"]),
            literal_targets=LossAccuracySums.from_state_dict(
                state["literal_targets"]
            ),
            special_targets=LossAccuracySums.from_state_dict(
                state["special_targets"]
            ),
        )

    def compute(self) -> dict[str, float | int]:
        return {
            "ar_nll": _safe_mean(self.all_targets.nll, self.all_targets.count),
            # Challenge byte accounting charges each registered special,
            # including EOT, as one byte.
            "ar_bpb": _safe_mean(
                self.all_targets.nll, self.all_targets.count
            )
            / LOG_TWO,
            "ar_atomic_bpb": _safe_mean(
                self.all_targets.nll, self.all_targets.count
            )
            / LOG_TWO,
            "ar_accuracy": _safe_mean(
                self.all_targets.correct, self.all_targets.count
            ),
            "ar_atomic_bytes": self.all_targets.count,
            "ar_literal_only_bpb": _safe_mean(
                self.literal_targets.nll, self.literal_targets.count
            )
            / LOG_TWO,
            "ar_literal_accuracy": _safe_mean(
                self.literal_targets.correct, self.literal_targets.count
            ),
            "ar_literal_bytes": self.literal_targets.count,
            "ar_special_bits": _safe_mean(
                self.special_targets.nll, self.special_targets.count
            )
            / LOG_TWO,
            "ar_special_accuracy": _safe_mean(
                self.special_targets.correct, self.special_targets.count
            ),
            "ar_special_count": self.special_targets.count,
        }


class Utf8Role(str, Enum):
    ASCII = "ascii"
    LEADING = "leading"
    CONTINUATION = "continuation"
    INVALID_LITERAL = "invalid_literal"
    EOT = "eot"
    SPECIAL = "special"


def classify_utf8_literal(byte: int) -> Utf8Role:
    """Classify a literal independently of surrounding DFA state."""

    if not 0 <= byte < BYTE_COUNT:
        raise ValueError(f"literal byte must be in [0, 256), got {byte}")
    if byte <= 0x7F:
        return Utf8Role.ASCII
    if 0xC2 <= byte <= 0xF4:
        return Utf8Role.LEADING
    if 0x80 <= byte <= 0xBF:
        return Utf8Role.CONTINUATION
    return Utf8Role.INVALID_LITERAL


class Utf8DFA:
    """Streaming strict UTF-8 validator with resumable scalar state.

    The transition bounds reject overlong encodings, UTF-16 surrogates, and
    code points above U+10FFFF.  Invalid transitions mark the whole sequence
    invalid while the state resynchronizes so subsequent errors remain
    countable.
    """

    def __init__(self) -> None:
        self.remaining = 0
        self.next_min = 0x80
        self.next_max = 0xBF
        self.valid = True
        self.invalid_transitions = 0

    @property
    def accepting(self) -> bool:
        return self.remaining == 0

    def _reset_codepoint(self) -> None:
        self.remaining = 0
        self.next_min = 0x80
        self.next_max = 0xBF

    def _start_literal(self, byte: int) -> bool:
        if byte <= 0x7F:
            return True
        if 0xC2 <= byte <= 0xDF:
            self.remaining = 1
        elif byte == 0xE0:
            self.remaining = 2
            self.next_min = 0xA0
        elif 0xE1 <= byte <= 0xEC or 0xEE <= byte <= 0xEF:
            self.remaining = 2
        elif byte == 0xED:
            self.remaining = 2
            self.next_max = 0x9F
        elif byte == 0xF0:
            self.remaining = 3
            self.next_min = 0x90
        elif 0xF1 <= byte <= 0xF3:
            self.remaining = 3
        elif byte == 0xF4:
            self.remaining = 3
            self.next_max = 0x8F
        else:
            return False
        return True

    def feed_literal(self, byte: int) -> bool:
        """Consume a literal and return whether this transition was valid."""

        if not 0 <= byte < BYTE_COUNT:
            raise ValueError(f"literal byte must be in [0, 256), got {byte}")
        if self.remaining:
            if self.next_min <= byte <= self.next_max:
                self.remaining -= 1
                self.next_min = 0x80
                self.next_max = 0xBF
                return True
            self.valid = False
            self.invalid_transitions += 1
            self._reset_codepoint()
            # Resynchronize at this byte when it is independently a valid
            # starter.  The transition itself remains invalid.
            self._start_literal(byte)
            return False
        if self._start_literal(byte):
            return True
        self.valid = False
        self.invalid_transitions += 1
        return False

    def feed_boundary(self) -> bool:
        """Consume an atomic special boundary, including EOT."""

        accepted = self.accepting
        if not accepted:
            self.valid = False
            self.invalid_transitions += 1
        self._reset_codepoint()
        return accepted

    def state_dict(self) -> dict[str, int | bool]:
        return {
            "remaining": self.remaining,
            "next_min": self.next_min,
            "next_max": self.next_max,
            "valid": self.valid,
            "invalid_transitions": self.invalid_transitions,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        remaining = int(state["remaining"])
        next_min = int(state["next_min"])
        next_max = int(state["next_max"])
        invalid_transitions = int(state["invalid_transitions"])
        if remaining not in (0, 1, 2, 3):
            raise ValueError("invalid UTF-8 continuation count")
        if not 0x80 <= next_min <= next_max <= 0xBF:
            raise ValueError("invalid UTF-8 continuation bounds")
        if invalid_transitions < 0:
            raise ValueError("invalid transition count must be non-negative")
        self.remaining = remaining
        self.next_min = next_min
        self.next_max = next_max
        self.valid = bool(state["valid"])
        self.invalid_transitions = invalid_transitions


@dataclass
class Utf8ValidityCounter:
    """Additive raw UTF-8 validity and atomic-role accounting."""

    sequences: int = 0
    valid_sequences: int = 0
    invalid_transitions: int = 0
    incomplete_sequences: int = 0
    eot_mid_codepoint: int = 0
    discarded_post_eot_atoms: int = 0
    role_counts: dict[str, int] = field(
        default_factory=lambda: {role.value: 0 for role in Utf8Role}
    )

    def update_sequence(
        self,
        atomic_ids: Sequence[int] | torch.Tensor,
        manifest: AtomicIdManifest,
        *,
        stop_at_eot: bool = True,
    ) -> None:
        if isinstance(atomic_ids, torch.Tensor):
            if atomic_ids.ndim != 1:
                raise ValueError("UTF-8 accounting expects one one-dimensional sequence")
            ids = tuple(int(value) for value in atomic_ids.detach().cpu().tolist())
        else:
            ids = tuple(int(value) for value in atomic_ids)
        dfa = Utf8DFA()
        role_counts = {role.value: 0 for role in Utf8Role}
        stopped_at_eot = False
        incomplete = False
        eot_mid_codepoint = 0
        discarded_post_eot_atoms = 0
        for index, atomic_id in enumerate(ids):
            if 0 <= atomic_id < BYTE_COUNT:
                role = classify_utf8_literal(atomic_id)
                role_counts[role.value] += 1
                dfa.feed_literal(atomic_id)
                continue
            if atomic_id == manifest.mask_id or atomic_id == manifest.pad_id:
                raise ValueError("MASK and PAD are not generated UTF-8 atoms")
            special = manifest.special_by_id.get(atomic_id)
            if special is None:
                raise ValueError(f"invalid generated atomic id {atomic_id}")
            role = Utf8Role.EOT if atomic_id == manifest.eot_id else Utf8Role.SPECIAL
            role_counts[role.value] += 1
            was_accepting = dfa.feed_boundary()
            if role is Utf8Role.EOT:
                if not was_accepting:
                    eot_mid_codepoint += 1
                if stop_at_eot:
                    stopped_at_eot = True
                    discarded_post_eot_atoms += len(ids) - index - 1
                    break
        if not stopped_at_eot and not dfa.accepting:
            incomplete = True
            dfa.valid = False
        self.sequences += 1
        self.valid_sequences += int(dfa.valid and not incomplete)
        self.invalid_transitions += dfa.invalid_transitions
        self.incomplete_sequences += int(incomplete)
        self.eot_mid_codepoint += eot_mid_codepoint
        self.discarded_post_eot_atoms += discarded_post_eot_atoms
        for role in Utf8Role:
            self.role_counts[role.value] += role_counts[role.value]

    def merge(self, other: "Utf8ValidityCounter") -> None:
        self.sequences += other.sequences
        self.valid_sequences += other.valid_sequences
        self.invalid_transitions += other.invalid_transitions
        self.incomplete_sequences += other.incomplete_sequences
        self.eot_mid_codepoint += other.eot_mid_codepoint
        self.discarded_post_eot_atoms += other.discarded_post_eot_atoms
        for role in Utf8Role:
            self.role_counts[role.value] += other.role_counts[role.value]

    def state_dict(self) -> dict[str, Any]:
        return {
            "sequences": self.sequences,
            "valid_sequences": self.valid_sequences,
            "invalid_transitions": self.invalid_transitions,
            "incomplete_sequences": self.incomplete_sequences,
            "eot_mid_codepoint": self.eot_mid_codepoint,
            "discarded_post_eot_atoms": self.discarded_post_eot_atoms,
            "role_counts": dict(self.role_counts),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "Utf8ValidityCounter":
        result = cls(
            sequences=int(state["sequences"]),
            valid_sequences=int(state["valid_sequences"]),
            invalid_transitions=int(state["invalid_transitions"]),
            incomplete_sequences=int(state["incomplete_sequences"]),
            eot_mid_codepoint=int(state["eot_mid_codepoint"]),
            discarded_post_eot_atoms=int(state["discarded_post_eot_atoms"]),
        )
        role_counts = state["role_counts"]
        result.role_counts = {role.value: int(role_counts[role.value]) for role in Utf8Role}
        return result

    def compute(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {
            "utf8_sequences": self.sequences,
            "utf8_valid_sequences": self.valid_sequences,
            "utf8_valid_fraction": _safe_mean(
                self.valid_sequences, self.sequences
            ),
            "utf8_invalid_transitions": self.invalid_transitions,
            "utf8_incomplete_sequences": self.incomplete_sequences,
            "utf8_eot_mid_codepoint": self.eot_mid_codepoint,
            "utf8_discarded_post_eot_atoms": self.discarded_post_eot_atoms,
        }
        metrics.update(
            {f"utf8_{role}_count": count for role, count in self.role_counts.items()}
        )
        return metrics


def noise_bucket(masked: int, eligible: int) -> str:
    """Return a stable K/U bucket without floating-point boundary ambiguity."""

    if eligible < 0 or masked < 0 or masked > eligible:
        raise ValueError("noise counts must satisfy 0 <= K <= U")
    if eligible == 0:
        return "empty"
    if masked == 0:
        return "zero"
    if masked == eligible:
        return "all_mask"
    if 4 * masked <= eligible:
        return "q1"
    if 2 * masked <= eligible:
        return "q2"
    if 4 * masked <= 3 * eligible:
        return "q3"
    return "q4"


NOISE_BUCKETS = ("empty", "zero", "q1", "q2", "q3", "q4", "all_mask")


@dataclass
class DiffusionMetricCounter:
    """Token- and canvas-weighted denoising metrics with K/U buckets."""

    tokens: LossAccuracySums = field(default_factory=LossAccuracySums)
    canvas_nll_sum: float = 0.0
    canvas_accuracy_sum: float = 0.0
    scored_canvases: int = 0
    canvases: int = 0
    eligible_positions: int = 0
    active_positions: int = 0
    buckets: dict[str, LossAccuracySums] = field(
        default_factory=lambda: {name: LossAccuracySums() for name in NOISE_BUCKETS}
    )
    bucket_canvases: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in NOISE_BUCKETS}
    )
    role_sums: dict[str, LossAccuracySums] = field(
        default_factory=lambda: {role.value: LossAccuracySums() for role in Utf8Role}
    )

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        active_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        manifest: AtomicIdManifest,
    ) -> None:
        """Add a batch of canvases, preserving equal-canvas diagnostics."""

        if targets.ndim < 2:
            raise ValueError("diffusion targets need batch and canvas dimensions")
        if logits.shape[:-1] != targets.shape:
            raise ValueError("diffusion logits and target shapes disagree")
        if logits.shape[-1] != manifest.output_size:
            raise ValueError("diffusion logits have the wrong output vocabulary")
        if active_mask.shape != targets.shape or valid_mask.shape != targets.shape:
            raise ValueError("diffusion masks must have the target shape")
        if active_mask.dtype != torch.bool or valid_mask.dtype != torch.bool:
            raise ValueError("diffusion masks must be boolean")
        if bool((active_mask & ~valid_mask).any().item()):
            raise ValueError("active diffusion targets must be valid positions")

        batch = targets.shape[0]
        flat_width = targets[0].numel()
        flat_targets = targets.reshape(batch, flat_width).to(torch.long)
        flat_active = active_mask.reshape(batch, flat_width)
        flat_valid = valid_mask.reshape(batch, flat_width)
        flat_logits = logits.reshape(batch, flat_width, logits.shape[-1]).to(
            torch.float32
        )
        active_targets = flat_targets[flat_active]
        if active_targets.numel() and bool(
            ((active_targets < 0) | (active_targets >= manifest.output_size))
            .any()
            .item()
        ):
            raise ValueError("MASK, PAD, or invalid ids cannot be diffusion targets")

        for row in range(batch):
            eligible = int(flat_valid[row].sum().item())
            active = flat_active[row]
            masked = int(active.sum().item())
            bucket = noise_bucket(masked, eligible)
            self.canvases += 1
            self.eligible_positions += eligible
            self.active_positions += masked
            self.bucket_canvases[bucket] += 1
            if masked == 0:
                continue

            row_targets = flat_targets[row, active]
            row_logits = flat_logits[row, active]
            row_losses = F.cross_entropy(row_logits, row_targets, reduction="none")
            row_correct = row_logits.argmax(dim=-1).eq(row_targets)
            nll = float(row_losses.sum(dtype=torch.float64).item())
            correct = int(row_correct.sum().item())
            self.tokens.add(nll=nll, correct=correct, count=masked)
            self.buckets[bucket].add(nll=nll, correct=correct, count=masked)
            self.canvas_nll_sum += nll / masked
            self.canvas_accuracy_sum += correct / masked
            self.scored_canvases += 1

            for role in Utf8Role:
                role_mask = torch.tensor(
                    [
                        _target_role(int(target), manifest) is role
                        for target in row_targets.detach().cpu().tolist()
                    ],
                    dtype=torch.bool,
                    device=row_losses.device,
                )
                role_count = int(role_mask.sum().item())
                if role_count:
                    self.role_sums[role.value].add(
                        nll=float(
                            row_losses[role_mask].sum(dtype=torch.float64).item()
                        ),
                        correct=int(row_correct[role_mask].sum().item()),
                        count=role_count,
                    )

    def merge(self, other: "DiffusionMetricCounter") -> None:
        self.tokens.merge(other.tokens)
        self.canvas_nll_sum += other.canvas_nll_sum
        self.canvas_accuracy_sum += other.canvas_accuracy_sum
        self.scored_canvases += other.scored_canvases
        self.canvases += other.canvases
        self.eligible_positions += other.eligible_positions
        self.active_positions += other.active_positions
        for bucket in NOISE_BUCKETS:
            self.buckets[bucket].merge(other.buckets[bucket])
            self.bucket_canvases[bucket] += other.bucket_canvases[bucket]
        for role in Utf8Role:
            self.role_sums[role.value].merge(other.role_sums[role.value])

    def state_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens.state_dict(),
            "canvas_nll_sum": self.canvas_nll_sum,
            "canvas_accuracy_sum": self.canvas_accuracy_sum,
            "scored_canvases": self.scored_canvases,
            "canvases": self.canvases,
            "eligible_positions": self.eligible_positions,
            "active_positions": self.active_positions,
            "buckets": {
                name: self.buckets[name].state_dict() for name in NOISE_BUCKETS
            },
            "bucket_canvases": dict(self.bucket_canvases),
            "role_sums": {
                role.value: self.role_sums[role.value].state_dict()
                for role in Utf8Role
            },
        }

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any]
    ) -> "DiffusionMetricCounter":
        result = cls(
            tokens=LossAccuracySums.from_state_dict(state["tokens"]),
            canvas_nll_sum=float(state["canvas_nll_sum"]),
            canvas_accuracy_sum=float(state["canvas_accuracy_sum"]),
            scored_canvases=int(state["scored_canvases"]),
            canvases=int(state["canvases"]),
            eligible_positions=int(state["eligible_positions"]),
            active_positions=int(state["active_positions"]),
        )
        result.buckets = {
            name: LossAccuracySums.from_state_dict(state["buckets"][name])
            for name in NOISE_BUCKETS
        }
        result.bucket_canvases = {
            name: int(state["bucket_canvases"][name]) for name in NOISE_BUCKETS
        }
        result.role_sums = {
            role.value: LossAccuracySums.from_state_dict(
                state["role_sums"][role.value]
            )
            for role in Utf8Role
        }
        return result

    def compute(self) -> dict[str, float | int]:
        metrics: dict[str, float | int] = {
            "diffusion_nll": _safe_mean(self.tokens.nll, self.tokens.count),
            "diffusion_accuracy": _safe_mean(self.tokens.correct, self.tokens.count),
            "diffusion_active_targets": self.tokens.count,
            "diffusion_canvas_nll": _safe_mean(
                self.canvas_nll_sum, self.scored_canvases
            ),
            "diffusion_canvas_accuracy": _safe_mean(
                self.canvas_accuracy_sum, self.scored_canvases
            ),
            "diffusion_scored_canvases": self.scored_canvases,
            "diffusion_canvases": self.canvases,
            "diffusion_eligible_positions": self.eligible_positions,
            "diffusion_active_positions": self.active_positions,
        }
        for bucket in NOISE_BUCKETS:
            sums = self.buckets[bucket]
            prefix = f"diffusion_{bucket}"
            metrics[f"{prefix}_nll"] = _safe_mean(sums.nll, sums.count)
            metrics[f"{prefix}_accuracy"] = _safe_mean(sums.correct, sums.count)
            metrics[f"{prefix}_targets"] = sums.count
            metrics[f"{prefix}_canvases"] = self.bucket_canvases[bucket]
        for role in Utf8Role:
            sums = self.role_sums[role.value]
            prefix = f"diffusion_{role.value}"
            metrics[f"{prefix}_nll"] = _safe_mean(sums.nll, sums.count)
            metrics[f"{prefix}_accuracy"] = _safe_mean(sums.correct, sums.count)
            metrics[f"{prefix}_targets"] = sums.count
        return metrics


def _target_role(atomic_id: int, manifest: AtomicIdManifest) -> Utf8Role:
    if 0 <= atomic_id < BYTE_COUNT:
        return classify_utf8_literal(atomic_id)
    if atomic_id == manifest.eot_id:
        return Utf8Role.EOT
    if atomic_id in manifest.special_by_id:
        return Utf8Role.SPECIAL
    raise ValueError(f"input-only or invalid target id {atomic_id}")


def merge_metric_states(counters: Iterable[Any]) -> Any:
    """Merge same-typed additive counters without mutating the inputs."""

    counters = tuple(counters)
    if not counters:
        raise ValueError("at least one metric counter is required")
    counter_type = type(counters[0])
    if any(type(counter) is not counter_type for counter in counters):
        raise TypeError("all metric counters must have the same concrete type")
    result = counter_type.from_state_dict(counters[0].state_dict())
    for counter in counters[1:]:
        result.merge(counter)
    return result
