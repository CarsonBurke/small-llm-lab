"""Dynamics ablation on a learned RMS-normalized recurrent-state manifold.

The teacher's exposed state is already RMS-normalized. Normalizing only the
transition INPUT does not preserve that geometry: a residual rollout can grow,
drown its incoming character bits and saturate both the decoder and error
predictor. This variant normalizes each resulting state, rather than clipping a
loss, gradient, rollout duration or error estimate. Exact target verification is
still required; a learned error estimate is not a likelihood certificate.
"""

import torch
from torch import Tensor

from pretraining.nanogpt_mini.nanogpt_mini_dynamics_model import (
    DynamicsConfig,
    DynamicsGPT,
    ResidualTransition,
)
from pretraining.nanogpt_mini.nanogpt_mini_model import RMSNorm


class BoundedResidualTransition(ResidualTransition):
    def __init__(self, config: DynamicsConfig):
        super().__init__(config)
        # RMSNorm initializes gains to one without consuming RNG. Every shared
        # teacher/transition/critic parameter therefore has the reference init.
        self.state_norm = RMSNorm(config.model_dim)

    def forward(self, state: Tensor, previous_code: Tensor) -> Tensor:
        return self.state_norm(super().forward(state, previous_code))


class BoundedDynamicsGPT(DynamicsGPT):
    transition: BoundedResidualTransition
    architecture = "nanogpt_mini_dynamics_bounded_characters_v1"
    transition_type = BoundedResidualTransition

    @torch.no_grad()
    def reset_auxiliary_parameters(self):
        super().reset_auxiliary_parameters()
        self.transition.state_norm.gains.fill_(1)
