"""Hierarchical THINK/EMIT rollouts and their parallel replay for latent VAPO.

A rollout interleaves two action types at every generated stream position:
the gate decides THINK or EMIT from the belief; EMIT samples a token from the
renderer and feeds it back through the embedding (the pretrained closed loop),
THINK samples a latent from the transition head and feeds it back through the
adapter (it occupies a stream position but renders nothing).  Thinking is
unlimited: the only bound is ``max_stream_steps`` generated slots (thinks and
emits combined), so a row that thinks too much simply runs out of budget and
finishes with fewer emitted tokens — a cost the policy pays through reward,
not through forced actions.

Everything PPO needs later is stored as replayable *data* (token ids, sampled
thoughts, actions), not activations: ``replay_beliefs`` reassembles the exact
stream inputs and recomputes every belief in one parallel teacher-forced
forward, which is where new log-probs, values, and the transition's grounded
beta-NLL targets come from.  Rewards are terminal and task-defined by the
caller (the DAPO trainer writes binary verifier scores through
``assign_terminal_rewards``); ``continuation_reward`` survives only for the
``sample_latent --fineweb`` inspection tool.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields

import torch
from torch import Tensor

from postraining.core import top_p_sample
from postraining.latent_thought import EMIT, THINK, LatentThoughtModel

TOKEN_SLOT, THOUGHT_SLOT, PAD_SLOT = 0, 1, -1


@dataclass
class LatentRolloutBatch:
    """Row-aligned stream storage; every tensor is (batch, stream) unless noted.

    The stream starts with ``prompt_length`` teacher-forced prompt tokens.
    Positions from ``prompt_length - 1`` onward carry gate decisions: the
    action taken after consuming that position's input.  ``token_ids`` holds
    prompt and emitted tokens at TOKEN slots; ``thoughts`` (batch, stream,
    dim, fp32) holds the raw transition samples at THOUGHT slots.
    """

    kind: Tensor
    token_ids: Tensor
    thoughts: Tensor
    gate_actions: Tensor
    action_mask: Tensor
    emit_mask: Tensor
    old_gate_logprobs: Tensor
    old_token_logprobs: Tensor
    old_values: Tensor
    rewards: Tensor
    reward_scalar: Tensor  # (batch,)
    prompt_length: int

    def to(self, device: torch.device) -> "LatentRolloutBatch":
        moved = {}
        for field in fields(self):
            value = getattr(self, field.name)
            moved[field.name] = value.to(device) if isinstance(value, Tensor) else value
        return LatentRolloutBatch(**moved)

    @property
    def stream_length(self) -> int:
        return self.kind.size(1)


def continuation_reward(generated: str, reference: str) -> float:
    """Tier-0 reward: mean of prefix-match fraction and character F1.

    The F1 half is bag-of-characters and therefore gameable (generic common
    characters harvest recall without content match); Tier 0 is a
    learnability probe, not the final objective — read its learnability gate
    accordingly and rely on the prefix half plus later tiers for content.
    """
    if not reference:
        return 0.0
    prefix = 0
    for generated_char, reference_char in zip(generated, reference):
        if generated_char != reference_char:
            break
        prefix += 1
    prefix_score = prefix / len(reference)
    generated_counts = Counter(generated)
    overlap = sum((generated_counts & Counter(reference)).values())
    if generated:
        precision = overlap / len(generated)
        recall = overlap / len(reference)
        f1 = 2 * precision * recall / (precision + recall) if overlap else 0.0
    else:
        f1 = 0.0
    return 0.5 * prefix_score + 0.5 * f1


@torch.no_grad()
def rollout_continuations(
    wrapper: LatentThoughtModel,
    prompt_ids: Tensor,
    max_new_tokens: int,
    max_stream_steps: int,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None = None,
    eos_id: int | None = None,
) -> LatentRolloutBatch:
    """Roll the gate-conditioned stream forward from a (batch, P) prompt.

    ``max_stream_steps`` is the total generated-slot budget per row (thinks
    plus emits); ``max_new_tokens`` caps emitted tokens within it.  Thinking
    is never forcibly interrupted — a row that spends its whole budget
    thinking just emits fewer tokens.  With ``eos_id`` set, a row finishes
    the moment it emits that token (the EOS itself is recorded), matching
    the EOS-truncated decode the verifier scores.
    """
    if prompt_ids.dim() != 2 or prompt_ids.size(1) < 1:
        raise ValueError("prompt_ids must be (batch, length>=1)")
    if max_stream_steps < max_new_tokens:
        raise ValueError("max_stream_steps must be at least max_new_tokens")
    device = prompt_ids.device
    batch, prompt_length = prompt_ids.shape
    model_dim = wrapper.backbone.tok_emb.embedding_dim
    max_stream = prompt_length + max_stream_steps
    caches = wrapper.make_generation_cache(batch, max_stream, device)

    kind = torch.full((batch, max_stream), PAD_SLOT, dtype=torch.long, device=device)
    token_ids = torch.zeros((batch, max_stream), dtype=torch.long, device=device)
    thoughts = torch.zeros((batch, max_stream, model_dim), dtype=torch.float32, device=device)
    gate_actions = torch.zeros((batch, max_stream), dtype=torch.long, device=device)
    action_mask = torch.zeros((batch, max_stream), dtype=torch.float32, device=device)
    emit_mask = torch.zeros_like(action_mask)
    old_gate_logprobs = torch.zeros_like(action_mask)
    old_token_logprobs = torch.zeros_like(action_mask)
    # Stays zero through the rollout; refresh_old_statistics fills it from
    # the separate critic before anything consumes it.
    old_values = torch.zeros_like(action_mask)

    kind[:, :prompt_length] = TOKEN_SLOT
    token_ids[:, :prompt_length] = prompt_ids

    output = None
    for position in range(prompt_length):
        output = wrapper.token_step(prompt_ids[:, position], caches, position)
        caches = output.caches
    assert output is not None

    emitted = torch.zeros(batch, dtype=torch.long, device=device)
    ended = torch.zeros(batch, dtype=torch.bool, device=device)
    position = prompt_length - 1
    while position < max_stream - 1 and bool((~ended & (emitted < max_new_tokens)).any()):
        active = ~ended & (emitted < max_new_tokens)
        belief = output.belief
        action, gate_logprob = wrapper.gate.sample(belief, generator=generator)

        token = top_p_sample(output.logits, temperature, top_p)
        token_logprob = (
            output.logits.float().log_softmax(-1).gather(-1, token[:, None]).squeeze(-1)
        )
        thought, _ = wrapper.transition.sample(output.predicted, belief, generator=generator)

        record = active
        action_mask[record, position] = 1.0
        gate_actions[record, position] = action[record]
        old_gate_logprobs[record, position] = gate_logprob[record].float()
        emits = record & (action == EMIT)
        thinks = record & (action == THINK)
        old_token_logprobs[emits, position] = token_logprob[emits]

        next_position = position + 1
        kind[emits, next_position] = TOKEN_SLOT
        token_ids[emits, next_position] = token[emits]
        kind[thinks, next_position] = THOUGHT_SLOT
        thoughts[thinks, next_position] = thought[thinks]
        emitted += emits.long()
        if eos_id is not None:
            ended |= emits & (token == eos_id)

        # Finished rows keep stepping on token 0 (their next slot stays PAD,
        # so the zero-initialized token_ids row feeds the embedding; every
        # loss masks it); batched caches make per-row early exit impractical.
        next_input = torch.where(
            (kind[:, next_position] == THOUGHT_SLOT)[:, None, None],
            wrapper.thought_input(thought),
            wrapper.embed_tokens(token_ids[:, next_position][:, None]),
        )
        output = wrapper.step(next_input, caches, next_position)
        caches = output.caches
        position = next_position

    return LatentRolloutBatch(
        kind=kind,
        token_ids=token_ids,
        thoughts=thoughts,
        gate_actions=gate_actions,
        action_mask=action_mask,
        emit_mask=(gate_actions == EMIT).float() * action_mask,
        old_gate_logprobs=old_gate_logprobs,
        old_token_logprobs=old_token_logprobs,
        old_values=old_values,
        rewards=torch.zeros_like(action_mask),
        reward_scalar=torch.zeros(batch, dtype=torch.float32, device=device),
        prompt_length=prompt_length,
    )


def trim_stream(batch: LatentRolloutBatch) -> LatentRolloutBatch:
    """Drop all-PAD tail columns so replay never pays for the worst case."""
    used = int((batch.kind != PAD_SLOT).any(0).nonzero().max()) + 1
    trimmed = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, Tensor) and value.dim() >= 2 and value.size(1) == batch.stream_length:
            value = value[:, :used]
        trimmed[field.name] = value
    return LatentRolloutBatch(**trimmed)


def emitted_token_rows(batch: LatentRolloutBatch) -> list[list[int]]:
    """Emitted token ids per row, in stream order (excludes the prompt)."""
    rows = []
    generated = batch.kind[:, batch.prompt_length:] == TOKEN_SLOT
    tokens = batch.token_ids[:, batch.prompt_length:]
    for row in range(batch.kind.size(0)):
        rows.append(tokens[row][generated[row]].tolist())
    return rows


def assign_terminal_rewards(batch: LatentRolloutBatch, scores: Tensor) -> None:
    """Write one terminal reward per row at its final gate-decision position."""
    if scores.shape != batch.reward_scalar.shape:
        raise ValueError("one score per trajectory is required")
    positions = (
        batch.action_mask.size(1) - 1 - batch.action_mask.flip(1).argmax(1)
    ).long()
    batch.rewards.zero_()
    batch.rewards[torch.arange(batch.rewards.size(0), device=scores.device), positions] = scores
    batch.reward_scalar.copy_(scores)


def assemble_stream_latents(
    wrapper: LatentThoughtModel, batch: LatentRolloutBatch
) -> Tensor:
    """Rebuild the exact (batch, stream, dim) inputs the rollout consumed."""
    token_latent = wrapper.embed_tokens(batch.token_ids)
    thought_latent = wrapper.adapter(batch.thoughts).to(token_latent.dtype)
    inputs = torch.where(
        (batch.kind == THOUGHT_SLOT)[..., None], thought_latent, token_latent
    )
    return inputs * (batch.kind != PAD_SLOT)[..., None].to(inputs.dtype)


def replay_beliefs(
    wrapper: LatentThoughtModel, batch: LatentRolloutBatch
) -> tuple[Tensor, Tensor]:
    """One parallel teacher-forced pass over the stored stream.

    Returns (stream_inputs, beliefs); gradients flow only into the adapter
    (through frozen trunk activations) unless the caller detaches.
    """
    stream_inputs = assemble_stream_latents(wrapper, batch)
    beliefs = wrapper.backbone.temporal_belief_from_token_latent(stream_inputs)
    return stream_inputs, beliefs


def replay_head_inputs(
    wrapper: LatentThoughtModel, batch: LatentRolloutBatch
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Replay the stream and derive everything the PPO heads consume.

    Returns (beliefs, predicted, features, token_targets).  Both
    ``refresh_old_statistics`` and the trainer's update step go through this
    single code path; that is what makes the recomputed "old" statistics
    exact — epoch-0 PPO ratios are one by construction.
    """
    stream_inputs, beliefs = replay_beliefs(wrapper, batch)
    predicted = wrapper.backbone.prediction_latent(beliefs)
    features = torch.cat((stream_inputs, predicted), dim=-1)
    token_targets = torch.zeros_like(batch.token_ids)
    token_targets[:, :-1] = batch.token_ids[:, 1:]
    return beliefs, predicted, features, token_targets


@torch.no_grad()
def refresh_old_statistics(
    wrapper: LatentThoughtModel, critic, batch: LatentRolloutBatch
) -> None:
    """Overwrite the stored PPO statistics with parallel-replay recomputations.

    The stepwise rollout and the parallel replay reduce through the trunk in
    different orders; at bf16 scale that drifts log-probs enough to put a
    noise floor under PPO ratios, clip fractions, and GAE inputs.  Rewriting
    ``old_values``/``old_gate_logprobs``/``old_token_logprobs`` through the
    exact update-step code path removes the drift; positions outside the
    consuming masks are overwritten too, but nothing ever reads them.

    ``old_values`` come from the separate critic model — the rollout itself
    never computes values, so this is where GAE's baseline is filled in.
    """
    backbone = wrapper.backbone
    beliefs, _, features, token_targets = replay_head_inputs(wrapper, batch)
    batch.old_values.copy_(critic.values(batch).float())
    batch.old_gate_logprobs.copy_(
        wrapper.gate.log_prob(batch.gate_actions.float(), beliefs).float()
    )
    logits = backbone.logits_from_features(features)
    batch.old_token_logprobs.copy_(
        logits.float().log_softmax(-1).gather(-1, token_targets[..., None]).squeeze(-1)
    )


def grounded_transition_mask(batch: LatentRolloutBatch) -> Tensor:
    """Positions whose NEXT stream input is a real token (prompt or emitted).

    These are the transitions where the world model has a grounded target:
    the projected embedding of the token that actually followed.
    """
    next_is_token = torch.zeros_like(batch.action_mask)
    next_is_token[:, :-1] = (batch.kind[:, 1:] == TOKEN_SLOT).float()
    current_valid = (batch.kind != PAD_SLOT).float()
    return next_is_token * current_valid
