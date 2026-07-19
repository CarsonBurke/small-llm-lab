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
forward, which is where new log-probs and values come from — and, with the
trunk trainable at RL time, where every policy gradient enters the model.  Rewards are terminal and task-defined by the
caller (the DAPO trainer writes binary verifier scores through
``assign_terminal_rewards``); ``continuation_reward`` survives only for the
``sample_latent --fineweb`` inspection tool.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
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
    # (batch, stream, dim): per-dimension old log-probs of the thought
    # decided at each gate position, for the factored per-dim PPO ratio.
    old_thought_logprobs: Tensor
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
    stop_ids: int | Sequence[int] | None = None,
    caches: list[tuple[Tensor, ...]] | None = None,
    prompt_lengths: Tensor | None = None,
) -> LatentRolloutBatch:
    """Roll the gate-conditioned stream forward from a (batch, P) prompt.

    ``max_stream_steps`` is the total generated-slot budget per row (thinks
    plus emits); ``max_new_tokens`` caps emitted tokens within it.  Thinking
    is never forcibly interrupted — a row that spends its whole budget
    thinking just emits fewer tokens.  With ``stop_ids`` set, a row finishes
    the moment it emits any of those tokens (the stop token itself is
    recorded), matching the stop-truncated decode the verifier scores.  BOS
    belongs in ``stop_ids`` alongside EOS: pretraining shards never append
    EOS, so an emitted BOS ("next document starts here") is the model's only
    learned end-of-document signal.

    ``caches`` switches to the fixed-shape step path: the caller passes
    preallocated caches (``make_static_generation_cache``, at least
    ``prompt + max_stream_steps`` long), positions become 0-dim tensors and
    every step attends over the full cache under a ``key_mask`` — constant
    shapes, so a compiled ``wrapper.step`` replays one CUDA graph.  Reusing
    one cache set across calls of identical shape is what keeps the graph
    from re-recording.

    ``prompt_lengths`` (per-row true lengths) batches rollouts over prompts
    of unequal length: ``prompt_ids`` arrives LEFT-padded to a shared width
    and each row's padded prefix slots are masked out of attention.  The
    positional shift this introduces is exact — PoPE and RoPE scores depend
    only on position differences — and ``split_rollout_groups`` undoes the
    padding afterwards.  Eager-only (mutually exclusive with ``caches``).
    """
    if prompt_ids.dim() != 2 or prompt_ids.size(1) < 1:
        raise ValueError("prompt_ids must be (batch, length>=1)")
    if max_stream_steps < max_new_tokens:
        raise ValueError("max_stream_steps must be at least max_new_tokens")
    device = prompt_ids.device
    batch, prompt_length = prompt_ids.shape
    model_dim = wrapper.backbone.tok_emb.embedding_dim
    max_stream = prompt_length + max_stream_steps
    stop_tensor = None
    if stop_ids is not None:
        ids = (stop_ids,) if isinstance(stop_ids, int) else tuple(stop_ids)
        if ids:
            stop_tensor = torch.tensor(ids, dtype=torch.long, device=device)
    pad_lengths = None
    valid_slots = None
    if prompt_lengths is not None:
        if caches is not None:
            raise ValueError(
                "prompt_lengths (left-padded batching) is eager-only and "
                "cannot be combined with preallocated caches"
            )
        if prompt_lengths.shape != (batch,):
            raise ValueError("prompt_lengths must be one true length per row")
        prompt_lengths = prompt_lengths.to(device=device, dtype=torch.long)
        if bool((prompt_lengths < 1).any()) or bool(
            (prompt_lengths > prompt_length).any()
        ):
            raise ValueError("prompt_lengths must lie in [1, prompt_ids width]")
        pad_lengths = prompt_length - prompt_lengths
        # Slot k is a real (attendable) slot for row b iff k >= pad_lengths[b].
        valid_slots = (
            torch.arange(max_stream, device=device)[None, :] >= pad_lengths[:, None]
        )
    if caches is None:
        caches = wrapper.make_generation_cache(batch, max_stream, device)
        position_index = None
        key_masks = None
    else:
        cache_length = caches[0][0].size(2)
        if caches[0][0].size(0) != batch or cache_length < max_stream:
            raise ValueError(
                f"preallocated caches ({tuple(caches[0][0].shape)}) do not fit "
                f"batch {batch} x stream {max_stream}"
            )
        position_index = torch.zeros((), dtype=torch.long, device=device)
        # Row p is the step-p key mask; indexing it is a view, so the hot
        # loop adds no mask-construction kernels.
        key_masks = torch.ones(
            (cache_length, cache_length), dtype=torch.bool, device=device
        ).tril_()

    def step_position(position: int) -> tuple[int | Tensor, Tensor | None]:
        if position_index is not None:
            position_index.fill_(position)
            return position_index, key_masks[position]
        if valid_slots is None:
            return position, None
        mask = valid_slots[:, : position + 1]
        if position < prompt_length:
            # Rows whose query at ``position`` is still inside their own pad
            # region attend everything instead: a fully-masked SDPA row is
            # NaN, and that NaN would enter deeper layers' K/V at this slot
            # and later poison REAL queries (a masked score is -inf, and
            # -inf + NaN is NaN inside the softmax).  The finite garbage
            # output is discarded, and the slot itself stays masked for all
            # real queries via ``valid_slots``.
            mask = mask | (pad_lengths[:, None] > position)
        return position, mask

    kind = torch.full((batch, max_stream), PAD_SLOT, dtype=torch.long, device=device)
    token_ids = torch.zeros((batch, max_stream), dtype=torch.long, device=device)
    thoughts = torch.zeros((batch, max_stream, model_dim), dtype=torch.float32, device=device)
    gate_actions = torch.zeros((batch, max_stream), dtype=torch.long, device=device)
    action_mask = torch.zeros((batch, max_stream), dtype=torch.float32, device=device)
    emit_mask = torch.zeros_like(action_mask)
    old_gate_logprobs = torch.zeros_like(action_mask)
    old_token_logprobs = torch.zeros_like(action_mask)
    # Both stay zero through the rollout; refresh_old_statistics fills them
    # (values from the separate critic, per-dim thought log-probs through
    # the exact replay path) before anything consumes them.
    old_thought_logprobs = torch.zeros_like(thoughts)
    old_values = torch.zeros_like(action_mask)

    if valid_slots is None:
        kind[:, :prompt_length] = TOKEN_SLOT
        token_ids[:, :prompt_length] = prompt_ids
    else:
        prompt_valid = valid_slots[:, :prompt_length]
        kind[:, :prompt_length] = torch.where(prompt_valid, TOKEN_SLOT, PAD_SLOT)
        token_ids[:, :prompt_length] = prompt_ids * prompt_valid

    output = None
    for position in range(prompt_length):
        step_pos, key_mask = step_position(position)
        output = wrapper.token_step(prompt_ids[:, position], caches, step_pos, key_mask)
        caches = output.caches
    assert output is not None

    emitted = torch.zeros(batch, dtype=torch.long, device=device)
    ended = torch.zeros(batch, dtype=torch.bool, device=device)
    position = prompt_length - 1
    # ``bool(any())`` is a device-to-host sync that serializes this
    # launch-bound loop (the CPU cannot run ahead of the GPU), so the finish
    # check runs only every SYNC_EVERY steps.  The extra <=SYNC_EVERY-1
    # steps after all rows finish are no-ops: ``record`` is all-False, and
    # finished rows already keep stepping on token 0 by design.
    SYNC_EVERY = 16
    first_position = position
    while position < max_stream - 1:
        active = ~ended & (emitted < max_new_tokens)
        if (position - first_position) % SYNC_EVERY == 0 and not bool(active.any()):
            break
        belief = output.belief
        action, gate_logprob = wrapper.gate.sample(belief, generator=generator)

        token = top_p_sample(output.logits, temperature, top_p)
        token_logprob = (
            output.logits.float().log_softmax(-1).gather(-1, token[:, None]).squeeze(-1)
        )
        thought, _ = wrapper.transition.sample(output.predicted, generator=generator)

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
        if stop_tensor is not None:
            ended |= emits & torch.isin(token, stop_tensor)

        # Finished rows keep stepping on token 0 (their next slot stays PAD,
        # so the zero-initialized token_ids row feeds the embedding; every
        # loss masks it); batched caches make per-row early exit impractical.
        next_input = torch.where(
            (kind[:, next_position] == THOUGHT_SLOT)[:, None, None],
            wrapper.thought_input(thought),
            wrapper.embed_tokens(token_ids[:, next_position][:, None]),
        )
        step_pos, key_mask = step_position(next_position)
        # Under reduce-overhead the step outputs live in the CUDA graph's
        # static pool and are only valid until the NEXT replay: everything
        # read from ``output`` above happens before this call, and every
        # consumer copies out (float()/gather/where).  Keep it that way.
        output = wrapper.step(next_input, caches, step_pos, key_mask)
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
        old_thought_logprobs=old_thought_logprobs,
        old_values=old_values,
        rewards=torch.zeros_like(action_mask),
        reward_scalar=torch.zeros(batch, dtype=torch.float32, device=device),
        prompt_length=prompt_length,
    )


def split_rollout_groups(
    batch: LatentRolloutBatch, group_size: int, prompt_lengths: Tensor
) -> list[LatentRolloutBatch]:
    """Undo a left-padded multi-group rollout into per-group batches.

    Each consecutive ``group_size`` block of rows shares one prompt (hence
    one pad length); dropping that group's pad columns makes its batch
    column-identical to a sequential single-prompt rollout, so scoring,
    refresh, and updates run on it unchanged.
    """
    total = batch.kind.size(0)
    if total % group_size:
        raise ValueError("batch rows must divide evenly into groups")
    groups = []
    for start in range(0, total, group_size):
        rows = slice(start, start + group_size)
        group_prompt = int(prompt_lengths[start])
        if bool((prompt_lengths[rows] != group_prompt).any()):
            raise ValueError("rows within a group must share one prompt length")
        pad = batch.prompt_length - group_prompt
        sliced = {}
        for field in fields(batch):
            value = getattr(batch, field.name)
            if field.name == "prompt_length":
                value = group_prompt
            elif (
                isinstance(value, Tensor)
                and value.dim() >= 2
                and value.size(1) == batch.stream_length
            ):
                # clone(): a view here would keep the whole multi-group
                # rollout storage (including both fp32 (rows, stream, dim)
                # recorders) alive through the entire PPO update phase.
                value = value[rows, pad:].clone()
            elif isinstance(value, Tensor) and value.dim() == 1:
                value = value[rows].clone()
            sliced[field.name] = value
        groups.append(LatentRolloutBatch(**sliced))
    return groups


def trim_stream(batch: LatentRolloutBatch, multiple: int = 1) -> LatentRolloutBatch:
    """Drop all-PAD tail columns so replay never pays for the worst case.

    ``multiple`` rounds the kept length up to a bucket boundary, PADDING
    BEYOND the original stream when the content reaches into the last
    partial bucket — capping at the original length instead would leak one
    arbitrary stream shape per capped group and silently defeat the
    bounded-shape guarantee the compiled replay relies on (measured: the
    per-shape compiles never stopped).  Replay is padding-invariant — PAD
    inputs are zeroed, attention is causal, every loss is masked — so
    bucketing only bounds the set of stream shapes the compiled replay
    functions ever see.
    """
    used = int((batch.kind != PAD_SLOT).any(0).nonzero().max()) + 1
    if multiple > 1:
        used = -(-used // multiple) * multiple
    trimmed = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, Tensor) and value.dim() >= 2 and value.size(1) == batch.stream_length:
            if used <= batch.stream_length:
                value = value[:, :used]
            else:
                padding = value.new_full(
                    (value.size(0), used - batch.stream_length, *value.shape[2:]),
                    PAD_SLOT if field.name == "kind" else 0,
                )
                value = torch.cat((value, padding), dim=1)
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

    Returns (stream_inputs, beliefs).  Stored thoughts and tokens are
    constants (sampled data), so there is no BPTT through sampling; with
    grad enabled, gradients flow into the trunk, embeddings, and adapter.
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
    predicted = wrapper.thought_mean(beliefs)
    features = wrapper.renderer_features(stream_inputs, beliefs)
    token_targets = torch.zeros_like(batch.token_ids)
    token_targets[:, :-1] = batch.token_ids[:, 1:]
    return beliefs, predicted, features, token_targets


def select_thought_actions(
    batch: LatentRolloutBatch, predicted: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Select projected means and sampled actions at actual THINK positions.

    The projector has already run densely. Compacting only its consumers
    ensures EMIT/prompt/pad outputs have no gradient edge and prevents unused
    PPO ratios from overflowing before a zero mask is applied.
    """
    think_mask = (batch.gate_actions == THINK) & batch.action_mask.bool()
    thought_targets = torch.zeros_like(batch.thoughts)
    thought_targets[:, :-1] = batch.thoughts[:, 1:]
    return predicted[think_mask], thought_targets[think_mask], think_mask


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

    The forward here runs GRAD-ENABLED on purpose, even though the graph is
    discarded: under torch.compile the grad mode is a guard, and a no-grad
    trace would give this refresh a different compiled artifact (different
    kernel fusions, different bf16 reduction order) than the update step —
    reintroducing exactly the epoch-0 ratio drift it exists to remove.
    Eagerly the numerics are identical either way; the cost is one
    forward's transient activation memory.

    ``old_values`` come from the separate critic model — the rollout itself
    never computes values, so this is where GAE's baseline is filled in.
    """
    backbone = wrapper.backbone
    beliefs, predicted, features, token_targets = replay_head_inputs(wrapper, batch)
    values = critic.values(batch).float()
    gate_logprobs = wrapper.gate.log_prob(batch.gate_actions.float(), beliefs).float()
    logits = backbone.logits_from_features(features)
    token_logprobs = (
        logits.float().log_softmax(-1).gather(-1, token_targets[..., None]).squeeze(-1)
    )
    # The thought decided at gate position p is stored at p+1 — the same
    # shift as token targets — so per-dim log-probs align with think_mask.
    thought_means, thought_targets, think_mask = select_thought_actions(
        batch, predicted
    )
    compact_thought_logprobs = wrapper.transition.per_dim_log_prob(
        thought_targets, thought_means
    ).float()
    thought_logprobs = torch.zeros_like(batch.old_thought_logprobs)
    thought_logprobs[think_mask] = compact_thought_logprobs
    with torch.no_grad():
        batch.old_values.copy_(values)
        batch.old_gate_logprobs.copy_(gate_logprobs)
        batch.old_token_logprobs.copy_(token_logprobs)
        batch.old_thought_logprobs.copy_(thought_logprobs)
