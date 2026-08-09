"""Training-time coordination for K3 Quantile Balancing."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.distributed as dist
from torch import Tensor, nn

from pretraining.latent_moe import StableLatentMoE


def iter_latent_moes(model: nn.Module) -> Iterator[StableLatentMoE]:
    for module in model.modules():
        if isinstance(module, StableLatentMoE):
            yield module


def enable_quantile_balance_collection(
    model: nn.Module,
    *,
    num_bins: int = 1000,
    collect_in_eval: bool = False,
) -> int:
    """Configure fixed-shape accumulators before compiling the model."""

    modules = list(iter_latent_moes(model))
    if not modules:
        raise ValueError("model contains no LatentMoE layers")
    for moe in modules:
        moe.enable_qb_collection(
            num_bins=num_bins, collect_in_eval=collect_in_eval
        )
    return len(modules)


@torch.no_grad()
def reset_quantile_balance_accumulators(model: nn.Module) -> None:
    """Start a step with empty statistics and one fixed correction bias."""

    modules = list(iter_latent_moes(model))
    if not modules:
        raise ValueError("model contains no LatentMoE layers")
    for moe in modules:
        moe.reset_qb_accumulators_()


@torch.no_grad()
def apply_accumulated_quantile_balance(model: nn.Module) -> tuple[Tensor, Tensor]:
    """Globally reduce a complete step and install its next routing biases.

    Compiled training forwards populate each module's fixed device buffers.
    This function runs once after all gradient-accumulation microbatches and
    before the optimizer step. The current bias was immutable throughout those
    forwards; the newly installed bias first affects the following step.

    Returns mean load CV-squared and the maximum expert load fraction.
    """

    modules = list(iter_latent_moes(model))
    if not modules:
        raise ValueError("model contains no LatentMoE layers")
    distributed = dist.is_available() and dist.is_initialized()
    layer_cvs = []
    layer_max_loads = []
    for moe in modules:
        histogram = moe.get_accumulated_qb_histogram()
        expert_loads = moe.get_accumulated_route_loads()
        if distributed:
            dist.all_reduce(histogram.counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(histogram.token_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(expert_loads, op=dist.ReduceOp.SUM)
        next_bias = moe.correction_bias_from_qb_histogram(histogram)
        moe.set_correction_bias_(next_bias)
        load_fraction = expert_loads.float() / expert_loads.sum().clamp_min(1)
        uniform = 1.0 / moe.config.num_routed_experts
        layer_cvs.append(
            (load_fraction - uniform).square().mean() / uniform**2
        )
        layer_max_loads.append(load_fraction.max())

    return torch.stack(layer_cvs).mean(), torch.stack(layer_max_loads).max()


__all__ = [
    "apply_accumulated_quantile_balance",
    "enable_quantile_balance_collection",
    "iter_latent_moes",
    "reset_quantile_balance_accumulators",
]
