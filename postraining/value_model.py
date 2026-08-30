"""Separate from-scratch critic for latent VAPO.

A fresh trunk (same architecture class as the policy backbone, random init,
fully trainable) reads the identical token/thought stream. Raw fp32 Gaussian
actions stored by the actor are adapted by the critic's own combined-embedding
stack; they are never redrawn. No parameters are shared with the policy.

The trainer passes ``hl_gauss.anchored_unit_geometry`` for v_min/v_max/
num_bins by default, putting bin centers at exactly 0 and 1 with margin bins
beyond each so boundary targets project without truncation bias; the
constructor defaults below are the legacy [0, 1]-edge grid.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from postraining.hl_gauss import HLGaussSupport
from postraining.latent_rollout import (
    PAD_SLOT,
    THOUGHT_SLOT,
    LatentRolloutBatch,
)
from postraining.latent_thought import CombinedEmbedding


class SeparateCritic(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        num_bins: int = 101,
        sigma_ratio: float = 2.0,
        v_min: float = 0.0,
        v_max: float = 1.0,
        prior_value: float = 0.0,
        mlp_hidden: int | None = None,
        num_blocks: int = 1,
    ):
        super().__init__()
        model_dim = trunk.tok_emb.embedding_dim
        self.trunk = trunk
        self.combiner = CombinedEmbedding(
            model_dim,
            mlp_hidden=mlp_hidden,
            num_blocks=num_blocks,
        )
        self.support = HLGaussSupport(num_bins, v_min, v_max, sigma_ratio)
        self.head = nn.Linear(model_dim, num_bins)
        with torch.no_grad():
            self.head.weight.zero_()
            # The floor bounds the bias range (v215's critic_prior_floor):
            # 1e-6 keeps far bins ~14 nats down instead of saturating the
            # softmax at -46, which would stall the early decode.
            self.head.bias.copy_(
                self.support.project_to_logprobs(torch.tensor(prior_value), eps=1e-6)
            )

    def assemble_inputs(self, batch: LatentRolloutBatch) -> Tensor:
        """Rebuild the exact token/thought stream in the critic's latent space."""
        token_latent = self.trunk.embed_tokens(batch.token_ids)
        pad_scale = (batch.kind != PAD_SLOT)[..., None].to(token_latent.dtype)
        if batch.thoughts.size(-1) == 0:
            if bool((batch.kind == THOUGHT_SLOT).any()):
                raise ValueError(
                    "latent critic replay requires stored raw thought actions"
                )
            return token_latent * pad_scale
        raw = batch.thoughts
        thought_base = raw.to(token_latent.dtype)
        thought_latent = self.combiner(thought_base, raw)
        inputs = torch.where(
            (batch.kind == THOUGHT_SLOT)[..., None],
            thought_latent,
            token_latent,
        )
        return inputs * pad_scale

    def value_logits(self, batch: LatentRolloutBatch) -> Tensor:
        """(batch, stream, num_bins) value distribution logits, fp32."""
        beliefs = self.trunk.temporal_belief_from_token_latent(
            self.assemble_inputs(batch)
        )
        # Distribution parameters are fp32 statistics even under autocast.
        with torch.autocast(device_type=beliefs.device.type, enabled=False):
            return self.head(beliefs.float())

    def values(self, batch: LatentRolloutBatch) -> Tensor:
        """Scalar value estimates via the expected-bin-center decode."""
        return self.support.to_expected_scalar(self.value_logits(batch))
