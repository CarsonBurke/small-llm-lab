"""Single-process Muon for RL post-training of nano backbones.

Replicates the pretraining optimizer's math exactly
(``nanogpt_mini_gpt2vocab_train.py``): Nesterov momentum, 12-iteration
NewtonSchulz5 orthogonalization in bf16, and the ``max(1, rows/cols)**0.5``
rectangular scaling — so an RL fine-tuning step moves trunk matrices with the
same update geometry they were pretrained under.  The pretraining class
round-robins parameters across ranks and all-gathers; the RL trainer is
single-process, so this version steps every parameter locally and skips
parameters whose grad is None (a parameter outside the current loss graph
must not decay or consume a momentum update).
"""

from __future__ import annotations

import torch
from torch import Tensor


def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def _muon_update(grad: Tensor, momentum: Tensor, mu: float = 0.95, nesterov: bool = True) -> Tensor:
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


# Pretraining compiles the update; CPU (tests) stays eager — the math is
# identical, only kernel fusion differs.
muon_update = (
    torch.compile(_muon_update) if torch.cuda.is_available() else _muon_update
)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float = 0.0, mu: float = 0.95):
        params = list(params)
        if not params:
            raise ValueError("Muon requires at least one parameter")
        for p in params:
            if p.ndim < 2:
                raise ValueError("Muon only orthogonalizes matrices (ndim >= 2)")
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
