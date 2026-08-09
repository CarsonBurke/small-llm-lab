"""The source trainer's Muon, without the distributed sharding.

This is a faithful port of ``Muon`` and ``PerHeadMuon`` from
``pretraining/nanogpt_mini/nanogpt_mini_gpt2vocab_kda_3to1_pm_train.py``, the
trainer that produced the KDA/NoPE checkpoint Bolmo byteifies.  Reusing that
exact optimizer is the point: a Bolmo arm whose optimizer, learning rates and
schedule match the source run measures architecture against architecture, which
a Bolmo trained under the paper's AdamW recipe cannot do against a Muon
baseline.

The arithmetic is unchanged: five quintic Newton-Schulz iterations with the
modded-nanoGPT coefficients, Nesterov momentum folded into the same compiled
function, the ``max(1, rows / cols) ** 0.5`` rectangular scaling, and decoupled
weight decay applied before the update.  Two differences are mechanical rather
than numerical.  ``step`` runs over every parameter locally instead of sharding
them across ranks and all-gathering afterwards, which is identical arithmetic at
``world_size == 1``.  And it skips parameters whose gradient is ``None`` so that
a parameter outside the current stage's loss graph neither decays nor consumes a
momentum update; the source trainer always populates every gradient and so never
needed the branch.

This is deliberately *not* ``postraining/muon.py``.  That one runs Polar Express
coefficients without the rectangular scaling and documents itself as departing
from the geometry this trunk was pretrained under.  Alignment work has to use
the geometry the source run actually used.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor


# Binds the arithmetic below into a run's training contract, so a later edit to
# the iteration, the scaling or the momentum form invalidates a resume instead
# of silently changing the optimizer under a checkpoint. Bump it on any change.
SOURCE_MUON_ALGORITHM = (
    "newtonschulz5_quintic3.4445_-4.7750_2.0315_eps1e-7_bf16_"
    "rect_sqrt_rows_over_cols_nesterov_decoupled_decay/v1"
)


def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Quintic Newton-Schulz iteration from current Muon/modded-nanoGPT. The
    # coefficients intentionally do not converge all the way to the polar
    # factor: the resulting S-shaped singular-value map is both faster and
    # empirically better behaved than the older 12-step cubic iteration.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(5):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def _muon_update(grad: Tensor, momentum: Tensor, mu: Tensor, nesterov: bool = True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


# The source trainer compiles this; CPU (tests) stays eager, where the math is
# identical and only kernel fusion differs.
muon_update = torch.compile(_muon_update) if torch.cuda.is_available() else _muon_update


class Muon(torch.optim.Optimizer):
    """Momentum-orthogonalized updates for the matrix parameters of a block."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        weight_decay: float = 0.0,
        mu: float = 0.95,
    ) -> None:
        params = list(params)
        if not params:
            raise ValueError("Muon requires at least one parameter")
        if any(parameter.ndim < 2 for parameter in params):
            raise ValueError("Muon only accepts parameters of rank two or higher")
        # Rank-shards by size in the source trainer; kept so a future
        # multi-rank port sees the same ordering it would have seen there.
        params = sorted(params, key=lambda parameter: parameter.size(), reverse=True)
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, mu=mu))

    def _state_for(self, parameter: Tensor) -> dict:
        state = self.state[parameter]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(parameter)
        if "mu" not in state:
            # Tensor-valued momentum keeps the compiled update generic across
            # the scalar momentum warmup.
            state["mu"] = torch.empty(
                (), device=parameter.device, dtype=torch.float32
            )
        return state

    def _update(self, parameter: Tensor, group: dict) -> Tensor:
        state = self._state_for(parameter)
        state["mu"].fill_(group["mu"])
        return muon_update(parameter.grad, state["momentum"], mu=state["mu"])

    @torch.no_grad()
    def step(self) -> None:  # type: ignore[override]
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                update = self._update(parameter, group)
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])


class PerHeadMuon(Muon):
    """Muon with Q/K/V orthogonalization applied independently per head."""

    def __init__(
        self,
        params_with_heads: Sequence[tuple[torch.nn.Parameter, int]],
        lr: float = 0.02,
        weight_decay: float = 0.0,
        mu: float = 0.95,
    ) -> None:
        params_with_heads = list(params_with_heads)
        for parameter, heads in params_with_heads:
            if parameter.ndim != 2:
                raise ValueError(
                    "per-head Muon needs a two-dimensional projection, got "
                    f"shape {tuple(parameter.shape)}"
                )
            if heads < 1 or parameter.size(0) % heads:
                raise ValueError(
                    f"parameter shape {tuple(parameter.shape)} is not divisible "
                    f"across {heads} heads"
                )
        self.head_counts = {id(parameter): heads for parameter, heads in params_with_heads}
        if len(self.head_counts) != len(params_with_heads):
            raise ValueError("per-head Muon received the same parameter twice")
        super().__init__(
            [parameter for parameter, _ in params_with_heads],
            lr=lr,
            weight_decay=weight_decay,
            mu=mu,
        )

    def _update(self, parameter: Tensor, group: dict) -> Tensor:
        heads = self.head_counts[id(parameter)]
        state = self._state_for(parameter)
        head_shape = (heads, parameter.size(0) // heads, parameter.size(1))
        state["mu"].fill_(group["mu"])
        return muon_update(
            parameter.grad.view(head_shape),
            state["momentum"].view(head_shape),
            mu=state["mu"],
        ).reshape_as(parameter)
