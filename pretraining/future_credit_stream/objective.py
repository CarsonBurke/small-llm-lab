"""Scalar TD future-loss learning with token-local producer gradients."""

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pretraining.future_credit_stream.model import StreamingFFNModel


class StreamingObjective(nn.Module):
    """Fit future CE values and differentiate them only through current hidden.

    Let ``ce_t`` be the teacher-forced next-token loss from ``h_t``. The scalar
    critic represents ``V(h_t) = ce_(t+1) + gamma * ce_(t+2) + ...`` within a
    finite BOS-delimited document. The producer minimizes
    ``ce_t + gamma * V(h_t; stop_gradient(critic_weights))``. Online critic
    fitting uses ``V(stop_gradient(h_(t-1)))`` against the detached target
    ``ce_t + gamma * V(h_t)`` from the actual current training observation.

    Both value branches are built in one forward graph for one combined local
    backward, with no lookahead or retained predecessor graph. Critic regression
    cannot update the FFN, the producer term cannot update the critic, and
    neither branch reaches an earlier token's graph.
    """

    def __init__(
        self,
        model: StreamingFFNModel,
        discount: float = 1.0,
        bos_id: int = 1,
    ) -> None:
        super().__init__()
        self.model = model
        self.discount = float(discount)
        if not math.isfinite(self.discount) or not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be finite and in [0, 1]")
        self.bos_id = bos_id

    def forward(
        self,
        observed: Tensor,
        carry: Tensor,
        targets: Tensor,
        resets: Tensor,
        has_previous: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return current local loss, actual hidden, and detached FP32 stats.

        Reset the producer's state at BOS, but fit the critic on the original
        pre-reset carry: ``resets`` marks terminal predecessor states whose
        future-loss target is zero. Such rows remain in regression. Current
        BOS targets mask only the future value, not the current CE.

        Only the stream's first observation has no predecessor to fit. Page and
        optimizer boundaries do not drop TD targets. Stats are current CE,
        regression, predecessor prediction mean, and predecessor target mean.
        """
        previous = carry.detach()
        state = previous.masked_fill(resets[:, None], 0.0)
        logits, hidden = self.model(observed, state)
        ce = F.cross_entropy(logits, targets, reduction="none")
        ce_mean = ce.mean()
        zero = ce_mean.new_zeros(())
        if self.model.critic is None:
            return ce_mean, hidden, torch.stack((ce_mean, zero, zero, zero)).detach()

        actor_value = self.model.value(hidden, detach_weights=True)
        future = actor_value.masked_fill(targets == self.bos_id, 0.0)
        actor = self.discount * future.mean()
        if has_previous:
            prediction = self.model.value(previous)
            target = (ce + self.discount * future).detach().masked_fill(resets, 0.0)
            regression = 0.5 * (prediction - target).square().mean()
            prediction_mean = prediction.mean()
            target_mean = target.mean()
        else:
            regression = zero
            prediction_mean = zero
            target_mean = zero

        loss = ce_mean + actor + regression
        stats = torch.stack(
            (ce_mean, regression, prediction_mean, target_mean)
        ).detach()
        return loss, hidden, stats


__all__ = ["StreamingObjective"]
