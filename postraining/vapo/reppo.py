"""REPPO's actor stabilizers, adapted to one on-policy step per pool.

REPPO (``reppo/src/torchrl/reppo.py``) holds its actor with two learned
multipliers: an entropy temperature tuned toward a target entropy, and a KL
Lagrangian to the policy that collected the batch. The temperature carries
over directly. The KL multiplier does not: REPPO takes many actor steps per
batch, so the KL to the collecting policy grows inside the update and the
penalty has a gradient to give. This trainer takes exactly one step per
fresh batch, where KL(old || new) and its gradient are both zero at the
point the gradient is taken; a penalty there is inert. The bound is
therefore enforced on the step itself: the realized KL is measured after
the step, and a step that overshoots is backtracked along its own direction.

Two further departures from REPPO. Its critic learns a soft Q: the entropy
bonus, -gamma * temperature * log pi(a'|s'), is folded into every reward,
so Q already prices future entropy. Here Q is the plain task value and the
entropy term lives in the actor loss only, as the exact per-token entropy
of the current policy; future entropy is not credited to earlier tokens.
And REPPO steps its temperature with the actor's own optimizer; here it has
a dedicated Adam, stepped once per actor step on the entropy that step saw.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor


class EntropyTemperature:
    """REPPO's auto-tuned entropy temperature.

    The temperature is exp(log_temperature), stepped by Adam on REPPO's
    loss ``temperature * (entropy - target)``: it rises while the policy's
    entropy is under the target and decays toward zero above it, so it
    constrains entropy from below only. Adam normalizes the gradient, so
    the learning rate is about the largest change in log temperature per
    step, independent of the entropy scale.
    """

    def __init__(self, target: float, initial: float, learning_rate: float):
        if not (math.isfinite(target) and target > 0.0):
            raise ValueError("entropy target must be finite and positive")
        if not (math.isfinite(initial) and initial > 0.0):
            raise ValueError("initial temperature must be finite and positive")
        self.target = target
        self.log_temperature = torch.tensor(
            math.log(initial), dtype=torch.float64, requires_grad=True
        )
        self.optimizer = torch.optim.Adam(
            [self.log_temperature], lr=learning_rate
        )

    @property
    def temperature(self) -> float:
        return math.exp(float(self.log_temperature.detach()))

    def update(self, entropy: float) -> None:
        """One dual step on the mean entropy the current temperature saw."""
        if not math.isfinite(entropy):
            raise ValueError("entropy must be finite")
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.log_temperature.exp() * (entropy - self.target)
        loss.backward()
        self.optimizer.step()

    def state_dict(self) -> dict:
        # A copy: Adam's state dict holds its live moment tensors, which
        # load_state_dict adopts as-is when dtype and device already match,
        # so an in-memory restore would otherwise share them.
        return copy.deepcopy(
            {
                "target": self.target,
                "log_temperature": float(self.log_temperature.detach()),
                "optimizer": self.optimizer.state_dict(),
            }
        )

    def load_state_dict(self, state: dict) -> None:
        if state["target"] != self.target:
            raise ValueError(
                f"entropy temperature state targets {state['target']}, "
                f"this run {self.target}"
            )
        with torch.no_grad():
            self.log_temperature.fill_(state["log_temperature"])
        self.optimizer.load_state_dict(state["optimizer"])


def snapshot_parameters(parameters: Sequence[Tensor]) -> list[Tensor]:
    """Detached copies of ``parameters``, to backtrack a step toward."""
    return [parameter.detach().clone() for parameter in parameters]


@torch.no_grad()
def backtrack_parameters(
    parameters: Sequence[Tensor], before: Sequence[Tensor], fraction: float
) -> None:
    """Keep ``fraction`` of the step from ``before``: before + f (now - before)."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("a backtrack keeps a fraction strictly inside (0, 1)")
    for parameter, previous in zip(parameters, before, strict=True):
        parameter.lerp_(previous, 1.0 - fraction)


def enforce_kl_bound(
    parameters: Sequence[Tensor],
    before: Sequence[Tensor],
    drift: dict[str, float],
    measure: Callable[[], dict[str, float]],
    *,
    bound: float,
    max_backtracks: int,
    kl_key: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Backtrack a taken step until its measured KL is within ``bound``.

    ``drift`` is ``measure()`` after the whole step from ``before``. Each
    pass keeps the fraction of the remaining step that a quadratic KL puts
    on the bound and re-measures, at most ``max_backtracks`` times; a step
    already inside the bound is untouched. Optimizer state is not rewound:
    it keeps the whole step's gradient. Returns the final drift and the
    ``trust/*`` telemetry.
    """
    if max_backtracks < 1:
        raise ValueError("max_backtracks must be positive")
    unbounded_kl = drift[kl_key]
    step_fraction = 1.0
    backtracks = 0
    while drift[kl_key] > bound and backtracks < max_backtracks:
        fraction = kl_backtrack_fraction(drift[kl_key], bound)
        backtrack_parameters(parameters, before, fraction)
        step_fraction *= fraction
        backtracks += 1
        drift = measure()
    return drift, {
        "trust/unbounded_kl": unbounded_kl,
        "trust/step_fraction": step_fraction,
        "trust/backtracks": float(backtracks),
        "trust/kl_bound": bound,
    }


def kl_backtrack_fraction(kl: float, bound: float) -> float:
    """The fraction of a step that brings a quadratic KL down to ``bound``.

    Near the old policy KL(old || new) is a quadratic form in the parameter
    step, so scaling the step by sqrt(bound / kl) lands on the bound. The
    caller re-measures, since a large step can leave the quadratic regime.
    """
    if not (math.isfinite(kl) and kl > bound > 0.0):
        raise ValueError("backtracking needs a finite KL above a positive bound")
    return math.sqrt(bound / kl)
