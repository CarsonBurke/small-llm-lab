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

The standalone Q critic (``CRITIC_Q_SCHEMA``) has no V at all. Its readout
is Q over the whole vocabulary at once, as a discrete-action Q network's is,
in the dueling form centred under the policy that samples:
Q(s_t, a) = c(f_t) + u(f_t) . (x(a) - sum_b pi(b | s_t) x(b))
(``policy_centered_q``), with x(a) the critic trunk's own readout row of
token a over the table's RMS, fixed, the table its final belief is shaped
to be read by, and f_t the trunk's belief at t -- plus, under a carry, the
actor's belief that sampled a_t, which the trunk only reads at t + 1. So
c is the policy's value and u only how actions differ within a state. The
value at an action slot is Q of the token taken there, trained end to end
through the trunk and its embedding. Because every token's Q exists, the
actor's advantage A_t = Q(s_t, a_t) - sum_a pi(a | s_t) Q(s_t, a) is exact
over the vocabulary: u_t . (x(a_t) - sum_a pi(a) x(a))
(``postraining.nano_backbone.policy_centered_token_values``). The action
enters only at that bilinear readout; the alternative, the token through
the trunk, would cost a trunk pass per candidate token.

With ``q_twin`` the standalone Q critic has two such heads on its one
trunk (``CRITIC_TWIN_Q_SCHEMA``), TD3/TD7's twin critics: independently
initialized hidden layers over the same belief and the same action table,
so they agree where the returns pin them down and disagree where the
readout extrapolates. Each regresses the same target, bootstrapped from the
smaller of their two Q values of the next taken token, and the actor reads
their mean.

The opt-in post-action binary Q critic instead reads the independent causal
trunk's belief after the sampled token, predicting success logits with a
nonlinear head. It has no pre-action value head, policy-average subtraction,
or dependency on the actor's hidden carry or memory writer. Its probability
is an action-conditioned expected binary return; exact verifier overrides
and binary cross-entropy targets belong to the trainer.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from postraining.latent_rollout import (
    PAD_SLOT,
    THOUGHT_SLOT,
    LatentRolloutBatch,
    carried_hiddens,
    generated_slot_mask,
    recorded_write_noise,
)
from postraining.latent_thought import CombinedEmbedding
from postraining.memory_carry import (
    GatedDeltaMemory,
    MemoryReader,
    borrowed_write_terms,
    memory_write_mask,
    read_stream,
)


class BeliefActionMLP(nn.Module):
    """f(b_t, x_{t+1}): a relu^2 MLP over the belief and the next input.

    The squared activation of a sum carries the belief-action products a
    sum of separate features cannot. The output is zero-initialized, so as
    the dueling head's Adv a fresh head's Q is exactly V.
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


class TokenQHead(nn.Module):
    """Q(s_t, a) = c(f_t) + u_t . (x(a) - xbar_t) for every token a, with
    u_t = D f_t + L b_t + g b_t and xbar_t the policy's mean action row
    (``policy_centered_q``).

    ``f_t`` is a relu^2 hidden layer over the critic's belief b_t at t and,
    when the policy carries one, the actor's belief that sampled a_t (unit
    RMS, so its scale is the trunk belief's). ``forward`` returns the state
    term c and the direction u; the caller supplies x
    (``SeparateCritic.action_rows``), the trunk's readout rows over their
    RMS. The linear path L lets u be the belief itself: u = s b makes
    u . x(a) the trunk's logit b . w(a), the contrast "the token this belief
    points at", which relu^2 features alone only approximate. In a toy with
    returns 1 for the readout's argmax token, the path raised the share of
    states whose highest-Q top-8 token is that argmax from 0.44-0.49 to
    0.60-0.65, and did nothing for an unrelated table (0.22 either way).

    The scalar belief gain g is L's identity component on its own rate.
    u = g b gives u . (x(a) - xbar) = (g / r)(z(a) - zbar), the trunk's
    bias-free logit z centred under the policy, with r the table's RMS:
    one number decides how much Q believes the trunk's own ranking. L could
    hold it as g I, but L trains at the direction rate (1e-4 / 512 in the
    trainer), where Adam moves an entry at most 1.25e-4 over a 640-update
    study fit, and a toy at that regime (512 dims, 4096 tokens, 640
    updates) left L's diagonal at 5e-6. g trains at the head rate. In that
    toy it raised the share of determined states (pi > 0.9) whose highest-Q
    top-8 token is the return's token from 0.37 to 0.83 when returns follow
    the actor's argmax, and from 0.27 to 0.60 when 30% of those argmaxes are
    wrong; with returns blind to the token it stayed at 3e-4, against 9e-3
    when they follow the argmax. A positive g sharpens the actor wherever it
    applies, think tokens included, by an amount the returns set.

    Every readout starts at zero, so a fresh head's Q is zero for every
    token, and the direction readouts' first gradients are f_t and b_t
    times x(a_t) - xbar_t, nonzero whenever the table is: a trunk copied
    from the actor has a trained readout, while a fresh trunk's is zero,
    and the table is fixed, so its Q could never tell tokens apart (hence
    ``--critic-init actor``). Twin heads therefore start with the same L,
    zero, as is g, and see the same residual through them, so they disagree
    through D; that is the price of a head that reads no contrast until data puts
    one there. The actor reads their mean, and an L drawn at random would
    put its noise in front of the actor at every token.

    The direction readout is unscaled: its weights and Adam steps operate
    directly in the coefficients used to predict Q. The action table has
    unit RMS overall. This differs from the inverse-dimension readout in
    critic schema v3, so those checkpoints require their original code.
    """

    def __init__(self, model_dim: int, hidden: int, reads_actor_belief: bool):
        super().__init__()
        self.belief = nn.Linear(model_dim, hidden, bias=False)
        self.actor_belief = (
            nn.Linear(model_dim, hidden, bias=False)
            if reads_actor_belief
            else None
        )
        self.state = nn.Linear(hidden, 1)
        self.direction = nn.Linear(hidden, model_dim, bias=False)
        self.belief_direction = nn.Linear(model_dim, model_dim, bias=False)
        self.belief_gain = nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            self.state.weight.zero_()
            self.state.bias.zero_()
            self.direction.weight.zero_()
            self.belief_direction.weight.zero_()

    @property
    def direction_weights(self) -> tuple[nn.Parameter, nn.Parameter]:
        """D and L, which share the direction rate; the gain g, the third
        term of u, trains at the head rate."""
        return self.direction.weight, self.belief_direction.weight

    def forward(
        self, beliefs: Tensor, actor_beliefs: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        if (actor_beliefs is None) != (self.actor_belief is None):
            raise ValueError(
                "the token Q head reads the actor's belief exactly when it "
                "was built for a carry policy"
            )
        hidden = self.belief(beliefs)
        if self.actor_belief is not None:
            hidden = hidden + self.actor_belief(
                F.rms_norm(actor_beliefs, (actor_beliefs.size(-1),))
            )
        hidden = F.relu(hidden).square()
        return (
            self.state(hidden).squeeze(-1),
            self.direction(hidden)
            + self.belief_direction(beliefs)
            + self.belief_gain * beliefs,
        )


class PostActionQHead(nn.Module):
    """Nonlinear action-conditioned success logits for a causal critic.

    The output starts at an explicit Bernoulli prior, defaulting to a 2%
    success probability. This is an initialization choice, not an optimal prior.
    Zero output weights mean the first step trains the readout; subsequent
    steps train the feature projection and independent trunk as well.
    """

    def __init__(self, model_dim: int, hidden: int, prior: float = 0.02):
        super().__init__()
        if not math.isfinite(prior) or not 0.0 < prior < 1.0:
            raise ValueError("post-action Q prior must be finite and between 0 and 1")
        self.belief = nn.Linear(model_dim, hidden, bias=False)
        self.out = nn.Linear(hidden, 1)
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.fill_(math.log(prior) - math.log1p(-prior))

    def forward(self, beliefs: Tensor) -> Tensor:
        return self.out(F.relu(self.belief(beliefs)).square()).squeeze(-1)


class SeparateCritic(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        mlp_hidden: int | None = None,
        num_blocks: int = 1,
        actor_memory: GatedDeltaMemory | None = None,
        action_head_hidden: int | None = None,
        q_head_hidden: int | None = None,
        q_head_reads_actor_belief: bool = False,
        post_action_q_hidden: int | None = None,
        post_action_q_prior: float = 0.02,
        q_twin: bool = False,
    ):
        super().__init__()
        if q_twin and q_head_hidden is None:
            raise ValueError("a twin Q head needs the token Q head it pairs with")
        if post_action_q_hidden is not None:
            if post_action_q_hidden < 1:
                raise ValueError("post-action Q hidden size must be positive")
            if (
                action_head_hidden is not None
                or q_head_hidden is not None
                or q_head_reads_actor_belief
                or actor_memory is not None
            ):
                raise ValueError(
                    "post-action Q uses an independent token critic, without "
                    "other value heads or the actor's memory writer"
                )
        if action_head_hidden is not None and q_head_hidden is not None:
            raise ValueError(
                "a standalone Q critic has no V for a dueling head to add to"
            )
        model_dim = trunk.tok_emb.embedding_dim
        self.trunk = trunk
        self.combiner: CombinedEmbedding | MemoryReader | nn.Identity
        if post_action_q_hidden is not None:
            self.combiner = nn.Identity()
        elif actor_memory is None:
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
        # Zero output at init either way: the trunk's random features start
        # with no say, and the readout learns the marginal before anything
        # finer.
        self.head: nn.Linear | None = None
        if q_head_reads_actor_belief and q_head_hidden is None:
            raise ValueError("only the token Q head reads the actor's belief")
        self.q_head = (
            None
            if q_head_hidden is None
            else TokenQHead(model_dim, q_head_hidden, q_head_reads_actor_belief)
        )
        self.post_action_q_head = (
            None
            if post_action_q_hidden is None
            else PostActionQHead(model_dim, post_action_q_hidden, post_action_q_prior)
        )
        if self.q_head is None and self.post_action_q_head is None:
            self.head = nn.Linear(model_dim, 1)
            with torch.no_grad():
                self.head.weight.zero_()
                self.head.bias.zero_()
        self.action_head = (
            None
            if action_head_hidden is None
            else BeliefActionMLP(model_dim, action_head_hidden)
        )
        # Built last, so the modules above draw the same initialization from
        # a seed with or without it.
        self.q_twin = (
            TokenQHead(model_dim, q_head_hidden, q_head_reads_actor_belief)
            if q_twin
            else None
        )

    def assemble_inputs(self, batch: LatentRolloutBatch) -> Tensor:
        """Rebuild the exact token/thought stream in the critic's latent space."""
        token_latent = self.trunk.embed_tokens(batch.token_ids)
        pad_scale = (batch.kind != PAD_SLOT)[..., None].to(token_latent.dtype)
        if self.post_action_q_head is not None:
            if batch.thoughts.size(-1):
                raise ValueError("post-action Q supports token actions only")
            torch._assert_async(
                (batch.kind != THOUGHT_SLOT).all(),
                "post-action Q supports token actions only",
            )
            # Stored actor carries are deliberately not read: this critic's
            # causal representation depends only on its own token stream.
            return token_latent * pad_scale
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
                    self.actor_memory_writer,
                    hiddens,
                    write_mask,
                    recorded_write_noise(batch, self.actor_memory_writer),
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

    @property
    def values_actions(self) -> bool:
        """Whether the critic's value at a slot is Q of the action taken
        there (the standalone Q critic) rather than V of the state."""
        return self.q_head is not None or self.post_action_q_head is not None

    def values(self, batch: LatentRolloutBatch) -> Tensor:
        """(batch, stream) scalar value estimates, fp32: V(s_t), or
        Q(s_t, a_t) for a standalone Q critic."""
        return self.estimates(batch)[0]

    def estimates(self, batch: LatentRolloutBatch) -> tuple[Tensor, Tensor | None]:
        """(values, dueling Q) over the stream, fp32.

        Q at slot t values the action taken there, whose consequence is the
        critic's input at t + 1; the stream always holds the slot after an
        action, so Q is defined at every action slot. The second entry is
        the dueling head's V + Adv, None without one; a standalone Q critic
        returns its Q as the values.
        """
        if self.q_head is not None:
            raise ValueError(
                "a token Q critic's Q is centered under the policy; read it with "
                "token_values and the policy's action offsets"
            )
        if self.post_action_q_head is not None:
            logits = self.action_success_logits(batch)
            return logits.sigmoid() * batch.action_mask.bool(), None
        inputs = self.assemble_inputs(batch)
        beliefs = self.trunk.temporal_belief_from_token_latent(inputs)
        # The value is an fp32 regression output even under autocast: bf16's
        # 8-bit mantissa cannot resolve a ~0.02 marginal's residuals.
        with torch.autocast(device_type=beliefs.device.type, enabled=False):
            values = self.head(beliefs.float()).squeeze(-1)
            if self.action_head is None:
                return values, None
            action_values = values.detach() + self.action_head(
                beliefs.detach().float(),
                next_slot_inputs(inputs).detach().float(),
            )
        return values, action_values

    def action_success_logits(self, batch: LatentRolloutBatch) -> Tensor:
        """Binary-return Q logits after the sampled token, aligned at t.

        The independent causal trunk reads a_t at slot t + 1. Its belief
        there includes the prefix and selected action, without future tokens,
        actor carries, a pre-action V head, or a policy-average subtraction.
        Non-action slots return zero logits; probability consumers must mask
        them as ``values`` does. The trainer supplies binary targets and BCE,
        and owns any exact terminal/verifier overrides.
        """
        if self.post_action_q_head is None:
            raise ValueError("success logits require a post-action Q critic")
        successor_kind = torch.cat(
            (batch.kind[:, 1:], batch.kind.new_full((batch.kind.size(0), 1), PAD_SLOT)),
            dim=1,
        )
        mask = batch.action_mask.bool()
        torch._assert_async(
            (~mask | (successor_kind != PAD_SLOT)).all(),
            "every post-action Q action requires its sampled successor token",
        )
        inputs = self.assemble_inputs(batch)
        beliefs = self.trunk.temporal_belief_from_token_latent(inputs)
        with torch.autocast(device_type=beliefs.device.type, enabled=False):
            logits = self.post_action_q_head(next_slot_inputs(beliefs).float())
        return torch.where(mask, logits, torch.zeros_like(logits))

    def fresh_parameters(self):
        """The inputs a carry policy adds to the critic: its combiner (the
        memory reader under the memory carry) and the token Q heads' actor-
        belief projections. Like the actor's ``new_parameters()``, every run
        that builds them starts them from zero or random init."""
        yield from self.combiner.parameters()
        if self.q_head is not None:
            for head in self.q_heads:
                if head.actor_belief is not None:
                    yield from head.actor_belief.parameters()

    @property
    def q_heads(self) -> tuple[TokenQHead, ...]:
        """The token Q heads: the one head, or the twin pair."""
        if self.q_head is None:
            raise ValueError("token Q heads belong to a standalone Q critic")
        return (self.q_head,) if self.q_twin is None else (self.q_head, self.q_twin)

    def token_q_parts(self, batch: LatentRolloutBatch) -> tuple[Tensor, Tensor]:
        """Every Q head's state term and direction from one trunk pass.

        Returns (heads, batch, stream) c_t and (heads, batch, stream, dim)
        u_t, both fp32. Q of the taken token a_t, the token at t + 1, needs
        its action row's offset from the policy's mean row too
        (``policy_centered_q``), which only the actor's readout has.
        """
        if self.q_head is None:
            raise ValueError("token values belong to a standalone Q critic")
        if batch.thoughts.size(-1):
            raise ValueError("a token Q critic values token actions only")
        inputs = self.assemble_inputs(batch)
        beliefs = self.trunk.temporal_belief_from_token_latent(inputs)
        actor_beliefs = (
            # hiddens[:, t + 1] is the actor's belief at t: the one that
            # sampled a_t, which the trunk first reads at t + 1.
            next_slot_inputs(carried_hiddens(batch))
            if self.q_head.actor_belief is not None
            else None
        )
        with torch.autocast(device_type=beliefs.device.type, enabled=False):
            beliefs = beliefs.float()
            actor_beliefs = None if actor_beliefs is None else actor_beliefs.float()
            states, directions = zip(*(head(beliefs, actor_beliefs) for head in self.q_heads))
            return torch.stack(states), torch.stack(directions)

    def token_value_members(self, batch: LatentRolloutBatch,
                            action_offsets: Tensor) -> tuple[Tensor, Tensor]:
        """Every Q head's Q of each taken token, and its direction.

        ``action_offsets`` (batch, stream, dim) are x(a_t) - xbar_t at each
        action slot, from the policy's readout (``policy_centered_q``).
        Returns (heads, batch, stream) Q(s_t, a_t) and the (heads, batch,
        stream, dim) direction u_t, both fp32. The heads share one trunk
        pass.
        """
        states, directions = self.token_q_parts(batch)
        return policy_centered_q(states, directions, action_offsets), directions

    def token_values(self, batch: LatentRolloutBatch,
                     action_offsets: Tensor) -> tuple[Tensor, Tensor]:
        """The critic's Q of each taken token, and its direction, as the
        actor reads them: the one head's, or the twins' mean. Q is linear in
        (c, u), so the mean direction gives the mean Q of every token.

        Returns (batch, stream) Q(s_t, a_t) and the (batch, stream, dim)
        direction u_t, both fp32 (``token_value_members``).
        """
        values, directions = self.token_value_members(batch, action_offsets)
        return values.mean(0), directions.mean(0)

    def action_rows(self, tokens: Tensor) -> Tensor:
        """x(a) of ``tokens``, fp32: the critic trunk's own readout rows w(a)
        over the whole table's RMS.

        The trunk's final belief b is shaped to be read by its untied
        readout, so with this table u(s) = s b gives u . x(a) = b . w(a),
        the state-dependent part of the trunk's logit, exactly; ILQL starts
        its token Q heads from the language model head for the same reason.
        The input table (critic schema v4) could not express it: the best
        single linear map from input latents to readout rows, fit on all
        other tokens, explains none of the numeric tokens' rows (held-out
        R^2 -0.03 on the SFT checkpoint). At determined answer starts its
        critic ranked the correct token first at 14-26% of states where the
        actor did at 94% (NOTES 2026-10-20).

        One scale for the table, not one per row: row norms carry how
        strongly a belief selects each token (1.9 to 6.2 across the SFT
        vocabulary), and a per-row norm would change which token b points
        at. The readout bias, a state-free token-frequency offset, is not
        part of x(a); 93% of it is linear in the rows anyway.

        The table is fixed features: no loss reaches the readout. Trained,
        only the rows of sampled tokens would take gradients, and a dense
        Adam steps a row first sampled late by about 3 lr at once and 30 lr
        over the moments' decay, on exactly the rare tokens whose Q the
        actor should trust least; in a toy at the trainer's rates training
        it changed nothing the critic learned (NOTES 2026-10-20).
        """
        readout = getattr(self.trunk, "proj", None)
        if not isinstance(readout, nn.Linear):
            raise ValueError("the token Q action table is the critic trunk's untied linear readout")
        with torch.no_grad(), torch.autocast(device_type=readout.weight.device.type, enabled=False):
            table = readout.weight.float()
            # rms_norm's epsilon: a zero table reads as zero, not NaN. The
            # SFT readout's mean square is about 0.02, so it moves the scale
            # by about 3e-6.
            scale = (table.square().mean() + torch.finfo(table.dtype).eps).rsqrt()
            return table[tokens] * scale

    def action_latents(self) -> Tensor:
        """(vocab, dim) fp32 x(a) of every token (``action_rows``), the
        action table of the token Q readout."""
        weight = self.trunk.proj.weight
        return self.action_rows(torch.arange(weight.size(0), device=weight.device))


def policy_centered_q(states: Tensor, directions: Tensor, action_offsets: Tensor) -> Tensor:
    """Q(s_t, a_t) = c_t + u_t . (x(a_t) - xbar_t), the dueling form.

    ``action_offsets`` x(a_t) - xbar_t carry the mean action row
    xbar_t = sum_a pi(a | s_t) x(a) under the policy that sampled a_t, so
    sum_a pi(a) Q(s_t, a) = c_t:
    the state term is the policy's value and u carries only how actions
    differ within a state. Fit uncentred, Q = c + u . x(a_t) lets the
    state's own value reach u through u . xbar, and any belief feature
    that tracks V, the policy's confidence among them, becomes a token
    contrast. The actor reads that contrast at every token, and it points
    up the policy's own logits, so MAC minimises entropy on returns that
    carry no action information (NOTES 2026-10-20). Centred, u's expected
    gradient at a state is Cov_pi(x, G) - Cov_pi(x) u, whatever c's error.
    ``states`` may carry a leading heads axis.

    The offset is formed whole, as (onehot(a_t) - pi) @ x
    (``nano_backbone.policy_action_offsets``), not as x(a_t) minus a mean
    row: under TF32 the mean row's GEMM rounds pi(a_t) x(a_t) like any
    other term, so as pi concentrates the difference keeps the rounding of
    x(a_t) while the offset itself shrinks. With pi(a_t) = 0.999 that was
    22% of u . offset in an emulation (d 512, V 50,304); whole, 0.03%.
    """
    return states + (directions * action_offsets).sum(-1)


def next_slot_inputs(inputs: Tensor) -> Tensor:
    """Each slot's successor input: the sampled action's consequence."""
    return torch.cat((inputs[:, 1:], torch.zeros_like(inputs[:, :1])), dim=1)


def next_slot_tokens(token_ids: Tensor) -> Tensor:
    """Each slot's successor token, the action taken there; the final slot,
    which takes none, gets token 0."""
    return torch.cat(
        (token_ids[:, 1:], torch.zeros_like(token_ids[:, :1])), dim=1
    )
