"""Future-bag carry objective: the actor's own head judges its carry.

Definitions for one observed transition at stream time ``t``: the model reads
``x_t`` and the carry ``c_{t-1}``, produces ``h_t`` and ``ce_t``, and the writer
emits ``c_t`` for the next observation. CE recursion carries ``h_t``, the
vector whose readout through the head is the belief about ``x_{t+1}``. The
carry is asked to be the vector whose readout through the *same frozen head*
is the discounted distribution of the tokens from ``x_{t+1}`` on:

    q_t  ~  sum_{j=0}^{H} discount^j [document continues] onehot(x_{t+1+j})
    L_t  =  cross-entropy( softcap(head(norm(c_t))), q_t )

``q_t`` is the discounted return of future one-hots, computed exactly from the
corpus: no bootstrap, no learned value, and no vector residual has to be
regressed. Future tokens are targets only, exactly as ``x_{t+1}`` is for CE;
they are never inputs, and no future step's graph is touched. With
``discount`` zero the target is ``x_{t+1}`` alone, the hidden's own job, so
the identity-initialized writer starts near its optimum; larger discounts
lengthen the horizon, which is what credits retaining context that pays off
many tokens later. The extra cost is two head matmuls per token (one
gradient-free, for the hidden's reference loss) plus the writer's three
square matmuls.

``bag_entropy`` is not the floor of ``L_t``: the bag is a handful of samples
from a future the carry cannot see. The matched reference is the same
cross-entropy of the hidden itself (``hidden_bag_loss``), what CE recursion
would carry; the writer earns its keep only below that.

Gradient ownership per token, one combined backward, no temporal graph:

* current CE trains the backbone;
* ``L_t`` trains the writer through the frozen head's input gradient, and
  reaches the norm gains, head weights, and the hidden only with an explicit
  ``backbone_future_weight``.
"""

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pretraining.future_credit_stream.model import StreamingFFNModel


CE_STATISTICS = ("ce",)

FUTURE_BAG_STATISTICS = (
    "ce",
    "bag_loss",
    "hidden_bag_loss",
    "bag_entropy",
    "gate_mean",
    "carry_cosine",
)


def _validate_unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


class StreamingObjective(nn.Module):
    """Plain CE recursion: the detached hidden is the next carry."""

    statistics = CE_STATISTICS

    def __init__(self, model: StreamingFFNModel) -> None:
        super().__init__()
        if model.writer is not None:
            raise ValueError("CE recursion carries the hidden verbatim; it has no writer")
        self.model = model

    def forward(
        self, observed: Tensor, carry: Tensor, targets: Tensor, resets: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return the local CE, the detached next carry, and FP32 statistics."""
        state = carry.detach().masked_fill(resets[:, None], 0.0)
        logits, hidden = self.model(observed, state)
        loss = F.cross_entropy(logits, targets)
        return loss, hidden.detach(), loss.detach()[None]


class FutureBagObjective(nn.Module):
    """Current CE plus the carry's cross-entropy to the discounted future bag."""

    statistics = FUTURE_BAG_STATISTICS

    def __init__(
        self,
        model: StreamingFFNModel,
        bos_id: int = 1,
        discount: float = 0.9,
        backbone_future_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if model.writer is None:
            raise ValueError("the future-bag objective trains a carry writer")
        self.model = model
        self.bos_id = bos_id
        self.discount = _validate_unit_interval(discount, "discount")
        self.backbone_future_weight = _validate_unit_interval(
            backbone_future_weight, "backbone_future_weight"
        )

    @torch.no_grad()
    def future_bag(self, targets: Tensor, future: Tensor) -> Tensor:
        """Return the normalized bag [B, V] over ``x_{t+1}`` and ``future``.

        ``future[j]`` is ``x_{t+2+j}``. The current target always belongs to
        the bag with weight 1; future token ``j`` belongs with weight
        ``discount^(j+1)`` when the current target and every earlier future
        token continued the document. A document's closing BOS is its last
        member, so no row is ever empty.
        """
        horizon, batch = future.shape
        tokens = torch.cat((targets[None, :], future))
        ends = tokens == self.bos_id
        closed_before = ends.long().cumsum(0) - ends.long()
        # pow(0, 0) = 1: discount 0 leaves exactly the current target.
        decay = self.discount ** torch.arange(horizon + 1, device=future.device, dtype=torch.float32)
        weights = decay[:, None] * (closed_before == 0).float()
        bag = torch.zeros(batch, self.model.config["vocab_size"], device=future.device, dtype=torch.float32)
        bag.scatter_add_(1, tokens.t(), weights.t())
        return bag / weights.sum(0)[:, None]

    def forward(
        self,
        observed: Tensor,
        carry: Tensor,
        targets: Tensor,
        future: Tensor,
        resets: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return the local loss, the detached next carry, and FP32 statistics.

        ``targets`` [B] is ``x_{t+1}``; ``future`` [H, B] holds
        ``x_{t+2} .. x_{t+1+H}``. Reset lanes read a zero carry.
        """
        state = carry.detach().masked_fill(resets[:, None], 0.0)
        logits, hidden = self.model(observed, state)
        ce_mean = F.cross_entropy(logits, targets)

        weight = self.backbone_future_weight
        write_input = hidden.detach() if weight == 0.0 else hidden.detach() + weight * (hidden - hidden.detach())
        next_carry, gate = self.model.writer(write_input, state)

        bag = self.future_bag(targets, future)
        log_belief = self.model.carry_readout(next_carry, weight).log_softmax(-1)
        bag_loss = -(bag * log_belief).sum(-1).mean()
        loss = ce_mean + bag_loss

        with torch.no_grad():
            # What CE recursion would have carried, judged by the same bag.
            hidden_log_belief = self.model.carry_readout(hidden).log_softmax(-1)
            hidden_bag_loss = -(bag * hidden_log_belief).sum(-1).mean()
            bag_entropy = -torch.xlogy(bag, bag).sum(-1).mean()
            detached_carry = next_carry.detach()
            carry_cosine = F.cosine_similarity(detached_carry.float(), hidden.float(), dim=-1).mean()
            stats = torch.stack((
                ce_mean, bag_loss, hidden_bag_loss, bag_entropy, gate.float().mean(), carry_cosine,
            )).detach()
        return loss, detached_carry, stats


class TemporalReferenceObjective(nn.Module):
    """TBPTT reference: CE recursion with true gradients through the page.

    This is an upper reference for what any local carry objective could gain
    in this architecture, not a candidate recipe: the carry keeps its graph
    across every tick of an optimizer page and is truncated only at page
    boundaries. It carries the hidden verbatim, like plain CE recursion.
    """

    statistics = CE_STATISTICS

    def __init__(self, model: StreamingFFNModel) -> None:
        super().__init__()
        if model.writer is not None:
            raise ValueError("the temporal reference carries the hidden verbatim")
        self.model = model

    def forward(
        self, inputs: Tensor, carry: Tensor, targets: Tensor, resets: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return the page-mean CE, the detached final carry, and statistics."""
        state = carry.detach()
        total = None
        for tick in range(inputs.shape[0]):
            state = state.masked_fill(resets[tick][:, None], 0.0)
            logits, hidden = self.model(inputs[tick], state, temporal_gradient=True)
            ce_mean = F.cross_entropy(logits, targets[tick])
            total = ce_mean if total is None else total + ce_mean
            state = hidden
        loss = total / inputs.shape[0]
        return loss, state.detach(), loss.detach()[None]


__all__ = [
    "CE_STATISTICS",
    "FUTURE_BAG_STATISTICS",
    "FutureBagObjective",
    "StreamingObjective",
    "TemporalReferenceObjective",
]
