"""HL-Gauss categorical value support (cleanrl iterthink v215 critic subset).

Ported from cleanrl's ``shared/hl_gauss.HLGaussSupport`` with exactly the
configuration the v215 critic uses — ``support_is_edges`` bins, truncated
Gaussian projection, softmax-CE training, expected-scalar decode — minus the
symlog machinery, which a [0, 1] reward support does not need.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def anchored_unit_geometry(
    interior_bins: int, margin_bins: int
) -> tuple[int, float, float]:
    """Support geometry whose bin centers include exactly 0.0 and 1.0.

    ``interior_bins`` uniform divisions set the bin width 1/interior_bins;
    the grid is then laid out so 0 and 1 are bin CENTERS (Dreamer3's
    exact-zero bucket, applied to both ends of the unit target range), with
    ``margin_bins`` extra bins beyond each anchor.  Without the margin a
    boundary target's Gaussian is truncated at the support edge and its
    renormalized label decodes ~0.8*sigma inward; margin_bins + 0.5 half
    bin-widths of slack keep that bias below the truncated tail mass.

    Returns ``(num_bins, v_min, v_max)`` for ``HLGaussSupport``.
    """
    if interior_bins < 1:
        raise ValueError("anchored support needs at least one interior bin")
    if margin_bins < 0:
        raise ValueError("anchored support margin must be non-negative")
    width = 1.0 / interior_bins
    num_bins = interior_bins + 1 + 2 * margin_bins
    v_min = -(margin_bins + 0.5) * width
    v_max = 1.0 + (margin_bins + 0.5) * width
    return num_bins, v_min, v_max


class HLGaussSupport(nn.Module):
    """Uniform-edge discretized support with HL-Gauss target projection.

    ``num_bins`` intervals partition [v_min, v_max]; a scalar target becomes
    the probability a Gaussian N(target, sigma_ratio * bin_width) lands in
    each interval, renormalized over the support (truncation).  An nn.Module
    only so the bin geometry moves with ``.to(device)``.
    """

    def __init__(
        self,
        num_bins: int,
        v_min: float,
        v_max: float,
        sigma_ratio: float,
        eps: float = 1e-10,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
        self.bin_width = (v_max - v_min) / num_bins
        self.sigma = sigma_ratio * self.bin_width
        self.eps = eps
        edges = torch.linspace(v_min, v_max, num_bins + 1)
        self.register_buffer("edges", edges)
        self.register_buffer("centers", (edges[:-1] + edges[1:]) / 2.0)

    def project(self, targets: Tensor) -> Tensor:
        """(...,) scalar targets -> (..., num_bins) categorical distributions."""
        targets = targets.float().clamp(self.v_min, self.v_max).unsqueeze(-1)
        cdf = torch.erf((self.edges - targets) / (self.sigma * math.sqrt(2.0)))
        total = cdf[..., -1:] - cdf[..., :1]
        return (cdf[..., 1:] - cdf[..., :-1]) / total.clamp_min(self.eps)

    def project_to_logprobs(self, targets: Tensor, eps: float = 1e-20) -> Tensor:
        return self.project(targets).clamp_min(eps).log()

    def to_expected_scalar(self, logits: Tensor) -> Tensor:
        """Decode logits as E[bin center] under softmax(logits)."""
        return (logits.float().softmax(-1) * self.centers).sum(-1)

    def cross_entropy(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Per-position CE between projected targets and the head's softmax."""
        return -(self.project(targets) * logits.float().log_softmax(-1)).sum(-1)
