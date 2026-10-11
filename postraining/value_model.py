"""Separate from-scratch critic for latent VAPO.

A fresh trunk (same architecture class as the policy backbone, random init,
fully trainable) reads the identical token/thought stream. Raw fp32 Gaussian
actions stored by the actor are adapted by the critic's own combined-embedding
stack; they are never redrawn. Under the deterministic hidden carry the critic
reads the actor's stored beliefs through that same separate combiner, at the
same generated-token slots. Under the gated-delta memory carry the critic
reads the actor's memory -- the same scan over the same stored beliefs,
through the actor's writer held as a constant -- with its own zero-init
reader, at every slot with a recorded belief behind it, so the value at a
slot conditions on what the actor will read next and the TD error carries
the value of each write. No parameters are shared with the policy, and the
value loss never trains the actor's writer.

The value head is one zero-initialized linear readout trained by unclipped
squared error (``CRITIC_SCHEMA``). Under Adam a zero head's output moves by
about lr * ||belief||_1 per step through its weight, so it can fit the ~2%
verifier success marginal within a few updates; the HL-Gauss categorical head
it replaced started every off-prior bin ~14 nats down and moved them only by
lr per step through the bias, which pinned its decoded value at 6e-5 through
a whole run (NOTES 2026-09-22).

The optional action head (``CRITIC_ACTION_HEAD_SCHEMA``) adds a dueling
Q(s_t, a_t) = V(s_t) + Adv(b_t, x_{t+1}): the critic's belief at the action
slot and its own input at the next slot -- the sampled token's embedding plus,
under the memory carry, the read that includes the write of the belief that
sampled it. V's trunk already consumes x_{t+1} at t+1, so V(s_{t+1}) is a
Q estimate too; the head makes Q - V one direct residual instead of the
difference of two trunk evaluations. Its inputs and V are detached, so the
value fit is exactly the plain critic's.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from postraining.latent_rollout import (
    PAD_SLOT,
    THOUGHT_SLOT,
    LatentRolloutBatch,
    carried_hiddens,
    generated_slot_mask,
)
from postraining.latent_thought import CombinedEmbedding
from postraining.memory_carry import (
    GatedDeltaMemory,
    MemoryReader,
    borrowed_write_terms,
    memory_write_mask,
    read_stream,
)


class ActionAdvantageHead(nn.Module):
    """Adv(b_t, x_{t+1}): a relu^2 MLP over the belief and the next input.

    The squared activation of a sum carries the belief-action products a
    sum of separate features cannot. The zero-initialized output makes a
    fresh head's Q exactly V.
    """

    def __init__(self, model_dim: int, hidden: int):
        super().__init__()
        self.belief = nn.Linear(model_dim, hidden, bias=False)
        self.action = nn.Linear(model_dim, hidden, bias=False)
        self.out = nn.Linear(hidden, 1)
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()

    def forward(self, beliefs: Tensor, next_inputs: Tensor) -> Tensor:
        hidden = self.belief(beliefs) + self.action(F.rms_norm(
            next_inputs, (next_inputs.size(-1),)
        ))
        return self.out(F.relu(hidden).square()).squeeze(-1)


class SeparateCritic(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        mlp_hidden: int | None = None,
        num_blocks: int = 1,
        actor_memory: GatedDeltaMemory | None = None,
        action_head_hidden: int | None = None,
    ):
        super().__init__()
        model_dim = trunk.tok_emb.embedding_dim
        self.trunk = trunk
        self.combiner: CombinedEmbedding | MemoryReader
        if actor_memory is None:
            self.combiner = CombinedEmbedding(
                model_dim,
                mlp_hidden=mlp_hidden,
                num_blocks=num_blocks,
            )
        else:
            self.combiner = MemoryReader(
                model_dim, actor_memory.heads, actor_memory.head_dim
            )
        # Borrowed, not owned: assigned around nn.Module registration so the
        # actor's writer is in neither the critic's parameters (its optimizer)
        # nor its state dict. It is a live reference, so the critic always
        # reads the memory the current actor writes -- the rollout's own, as
        # actor and critic step once per pool after the same forwards. A
        # deepcopy or ``.to()`` of the critic alone does not carry it along.
        object.__setattr__(
            self,
            "actor_memory_writer",
            None if actor_memory is None else actor_memory.writer,
        )
        self.head = nn.Linear(model_dim, 1)
        with torch.no_grad():
            # Zero output at init: the trunk's random features start with no
            # say, and the head learns the marginal before anything finer.
            self.head.weight.zero_()
            self.head.bias.zero_()
        self.action_head = (
            None
            if action_head_hidden is None
            else ActionAdvantageHead(model_dim, action_head_hidden)
        )

    def assemble_inputs(self, batch: LatentRolloutBatch) -> Tensor:
        """Rebuild the exact token/thought stream in the critic's latent space."""
        token_latent = self.trunk.embed_tokens(batch.token_ids)
        pad_scale = (batch.kind != PAD_SLOT)[..., None].to(token_latent.dtype)
        if (batch.carry_injected and batch.memory_carry) != (
            self.actor_memory_writer is not None
        ):
            raise ValueError(
                "the critic reads the actor's memory exactly when the batch "
                "was rolled out with the memory carry"
            )
        if self.actor_memory_writer is not None:
            hiddens = carried_hiddens(batch)
            write_mask = memory_write_mask(batch.kind, PAD_SLOT)
            inputs = read_stream(
                self.combiner,
                borrowed_write_terms(
                    self.actor_memory_writer, hiddens, write_mask
                ),
                token_latent,
                hiddens,
                write_mask,
            )
            return inputs * pad_scale
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
        return self.estimates(batch)[0]

    def estimates(self, batch: LatentRolloutBatch) -> tuple[Tensor, Tensor | None]:
        """(V, Q) over the stream, fp32; Q is None without an action head.

        Q at slot t values the action taken there, whose consequence is the
        critic's input at t + 1; the stream always holds the slot after an
        action, so Q is defined at every action slot.
        """
        inputs = self.assemble_inputs(batch)
        beliefs = self.trunk.temporal_belief_from_token_latent(inputs)
        # The value is an fp32 regression output even under autocast: bf16's
        # 8-bit mantissa cannot resolve a ~0.02 marginal's residuals.
        with torch.autocast(device_type=beliefs.device.type, enabled=False):
            values = self.head(beliefs.float()).squeeze(-1)
            if self.action_head is None:
                return values, None
            next_inputs = torch.cat(
                (inputs[:, 1:], torch.zeros_like(inputs[:, :1])), dim=1
            )
            action_values = values.detach() + self.action_head(
                beliefs.detach().float(), next_inputs.detach().float()
            )
        return values, action_values
