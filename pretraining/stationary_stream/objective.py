"""Plain next-token CE over the stationary buffer, with read diagnostics.

There is one objective: the current tick's CE trains the whole model,
including the read block. Nothing judges the buffered hiddens, and no
gradient crosses a tick. Diagnostics describe how the read is used:

Both attention diagnostics average over lanes with at least one readable
slot, so a lane that has just reset (null mass 1 by construction) does not
inflate them:

* ``null_mass``: mean attention on the null slot (1 means "read nothing");
* ``read_age``: mean age in steps of the non-null attention;
* ``read_rms``: RMS of the read vector without the value bias, i.e. of
  ``value.weight @ mixed`` for the ``value_map`` entry (exactly zero until
  the zero-initialized value map has grown) and of the un-normalized mixed
  vector for ``latent_norm`` (the normalized read that actually enters has
  unit RMS times the gains, so this reports how much mass the mixture
  retains). The observation it is added to has unit RMS times the
  observation-norm gains.
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from pretraining.stationary_stream.model import StationaryFFNModel


STATISTICS = ("ce", "null_mass", "read_age", "read_rms")


def read_diagnostics(
    attention: Tensor, valid: Tensor, ages: Tensor, read: Tensor, value_bias: Tensor | float
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (null_mass, read_age, read_rms) from FP32 attention [B, H, K+1] over slots aged ``ages``."""
    context = attention[..., 1:]
    mass = context.sum(-1)
    expected_age = (context * ages.float()).sum(-1) / mass.clamp_min(1e-12)
    readable = valid.any(0)[:, None].expand_as(mass).float()
    lanes = readable.sum().clamp_min(1.0)
    null_mass = (attention[..., 0] * readable).sum() / lanes
    read_age = (expected_age * readable).sum() / lanes
    read_rms = (read.float() - value_bias).square().mean(-1).sqrt().mean()
    return null_mass, read_age, read_rms


class StationaryObjective(nn.Module):
    """Current CE; the detached hidden is committed into the oldest ring slot."""

    statistics = STATISTICS

    def __init__(self, model: StationaryFFNModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, observed: Tensor, buffer: Tensor, valid: Tensor, ages: Tensor, targets: Tensor, resets: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return the local CE, the detached hidden, the next validity, and FP32 statistics."""
        readable = valid & ~resets[None]
        logits, hidden, attention, read = self.model(observed, buffer, readable, ages)
        loss = F.cross_entropy(logits, targets)
        next_valid = self.model.next_validity(readable, ages)
        with torch.no_grad():
            constant = self.model.read.constant_read()
            diagnostics = read_diagnostics(attention.detach(), readable, ages, read.detach(),
                                           constant.detach() if torch.is_tensor(constant) else constant)
            stats = torch.stack((loss.detach(), *diagnostics))
        return loss, hidden.detach(), next_valid, stats


__all__ = ["STATISTICS", "StationaryObjective", "read_diagnostics"]
