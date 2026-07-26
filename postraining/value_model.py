"""Separate from-scratch critic for latent VAPO.

A fresh trunk (same architecture class as the policy backbone, random init,
fully trainable) that reads the identical rollout stream — prompt and emitted
tokens through its own embeddings, thought latents through its own
orthogonal affine embedder — and predicts value alone.  No SIGReg,
no next-latent
prediction, no shared parameters with the policy: its only loss is HL-Gauss
cross-entropy on [0, 1] value targets (cleanrl iterthink v215 critic recipe:
softmax-CE to a Gaussian-smoothed two-hot, expected-scalar decode, zero-weight
head with the prior projected into the bias, no value clipping).

The trainer passes ``hl_gauss.anchored_unit_geometry`` for v_min/v_max/
num_bins by default, putting bin centers at exactly 0 and 1 with margin bins
beyond each so boundary targets project without truncation bias; the
constructor defaults below are the legacy [0, 1]-edge grid.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from postraining.hl_gauss import HLGaussSupport
from postraining.latent_rollout import PAD_SLOT, THOUGHT_SLOT, LatentRolloutBatch
from postraining.latent_thought import (
    CRITIC_ADAPTER_INIT_KINDS,
    CRITIC_ADAPTER_INIT_SCHEMAS,
    AffineThoughtAdapter,
)


class SeparateCritic(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        num_bins: int = 101,
        sigma_ratio: float = 2.0,
        v_min: float = 0.0,
        v_max: float = 1.0,
        prior_value: float = 0.0,
        adapter_init: str = "orthogonal",
    ):
        super().__init__()
        if adapter_init not in CRITIC_ADAPTER_INIT_KINDS:
            raise ValueError(f"unknown critic adapter init {adapter_init!r}")
        model_dim = trunk.tok_emb.embedding_dim
        self.trunk = trunk
        self.adapter_init = adapter_init
        self.adapter_init_schema = CRITIC_ADAPTER_INIT_SCHEMAS[adapter_init]
        self.adapter = AffineThoughtAdapter(
            model_dim, initialization=adapter_init
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
        """The rollout stream in the critic's own latent space.

        Mirrors ``assemble_stream_latents`` but through this model's
        embeddings and adapter — the critic shares no weights with the policy,
        so it must map the stored stream into its own representation.
        """
        token_latent = self.trunk.embed_tokens(batch.token_ids)
        pad_scale = (batch.kind != PAD_SLOT)[..., None].to(token_latent.dtype)
        if batch.thoughts.size(-1) == 0:
            # Pinned-EMIT rollouts store zero-width thoughts and contain no
            # THOUGHT slots; the adapter cannot consume a zero-width input.
            return token_latent * pad_scale
        think_mask = batch.kind == THOUGHT_SLOT
        # The kind-select makes dense adapter evaluation exactly equivalent
        # to compact boolean assignment, while avoiding the latter's
        # dynamic-shape nonzero graph break inside torch.compile.
        thought_latent = self.adapter(batch.thoughts.float()).to(
            token_latent.dtype
        )
        inputs = torch.where(
            think_mask[..., None], thought_latent, token_latent
        )
        return inputs * pad_scale

    def value_logits(self, batch: LatentRolloutBatch) -> Tensor:
        """(batch, stream, num_bins) value distribution logits, fp32."""
        beliefs = self.trunk.temporal_belief_from_token_latent(
            self.assemble_inputs(batch)
        )
        # Distribution parameters are fp32 statistics even under autocast,
        # matching the gate and transition heads' convention.
        with torch.autocast(device_type=beliefs.device.type, enabled=False):
            return self.head(beliefs.float())

    def values(self, batch: LatentRolloutBatch) -> Tensor:
        """Scalar value estimates via the expected-bin-center decode."""
        return self.support.to_expected_scalar(self.value_logits(batch))
