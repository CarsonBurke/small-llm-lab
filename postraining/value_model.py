"""Separate from-scratch critic for latent VAPO.

A fresh trunk (same architecture class as the policy backbone, random init,
fully trainable) reads the identical token/thought stream. Raw fp32 Gaussian
actions stored by the actor are adapted by the critic's own combined-embedding
stack; they are never redrawn. Under the deterministic hidden carry the critic
reads the actor's stored beliefs through that same separate combiner, at the
same generated-token slots. No parameters are shared with the policy.

The value head is one zero-initialized linear readout trained by unclipped
squared error (``CRITIC_SCHEMA``). Under Adam a zero head's output moves by
about lr * ||belief||_1 per step through its weight, so it can fit the ~2%
verifier success marginal within a few updates; the HL-Gauss categorical head
it replaced started every off-prior bin ~14 nats down and moved them only by
lr per step through the bias, which pinned its decoded value at 6e-5 through
a whole run (NOTES 2026-09-22).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from postraining.latent_rollout import (
    PAD_SLOT,
    THOUGHT_SLOT,
    LatentRolloutBatch,
    carried_hiddens,
    generated_slot_mask,
)
from postraining.latent_thought import CombinedEmbedding


class SeparateCritic(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
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
        self.head = nn.Linear(model_dim, 1)
        with torch.no_grad():
            # Zero output at init: the trunk's random features start with no
            # say, and the head learns the marginal before anything finer.
            self.head.weight.zero_()
            self.head.bias.zero_()

    def assemble_inputs(self, batch: LatentRolloutBatch) -> Tensor:
        """Rebuild the exact token/thought stream in the critic's latent space."""
        token_latent = self.trunk.embed_tokens(batch.token_ids)
        pad_scale = (batch.kind != PAD_SLOT)[..., None].to(token_latent.dtype)
        if batch.carry_injected:
            # The actor's recorded carry, projected by the critic's own
            # combiner: the value sees what the policy consumed, while the
            # two input maps stay separately trainable.
            inputs = self.combiner(
                token_latent, carried_hiddens(batch), generated_slot_mask(batch)
            )
            return inputs * pad_scale
        if batch.hiddens.size(-1):
            raise ValueError("batch stores hiddens it never injected")
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

    def values(self, batch: LatentRolloutBatch) -> Tensor:
        """(batch, stream) scalar value estimates, fp32."""
        beliefs = self.trunk.temporal_belief_from_token_latent(
            self.assemble_inputs(batch)
        )
        # The value is an fp32 regression output even under autocast: bf16's
        # 8-bit mantissa cannot resolve a ~0.02 marginal's residuals.
        with torch.autocast(device_type=beliefs.device.type, enabled=False):
            return self.head(beliefs.float()).squeeze(-1)
