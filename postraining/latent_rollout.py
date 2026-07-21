"""Hierarchical THINK/EMIT rollouts and their parallel replay for latent VAPO.

A rollout may force the first action after the prompt to be a latent THINK,
then interleaves two gate-selected action types at every subsequent generated
stream position. EMIT samples a token from the renderer and feeds it back
through the embedding (the pretrained closed loop); THINK samples a latent
from the transition head and feeds it back through the adapter (it occupies a
stream position but renders nothing). A forced thought is a continuous policy
action, but not a Bernoulli action: it receives critic/GAE and thought PPO
credit while ``gate_mask`` excludes it from gate PPO. Further thinking is
unlimited: the only bound is ``max_stream_steps`` generated slots (thinks and
emits combined), so a row that thinks too much simply runs out of budget and
finishes with fewer emitted tokens.

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
    Positions from ``prompt_length - 1`` onward carry temporal actions: the
    action taken after consuming that position's input. The first may be a
    forced THINK; every non-forced position is a Bernoulli gate decision
    selected by ``gate_mask``. ``token_ids`` holds prompt and emitted tokens
    at TOKEN slots; ``thoughts`` (batch, stream, dim, fp32) holds the raw
    transition samples at THOUGHT slots.
    """

    kind: Tensor
    token_ids: Tensor
    thoughts: Tensor
    gate_actions: Tensor
    action_mask: Tensor
    # Subset of action_mask where THINK/EMIT was actually sampled from the
    # Bernoulli. A forced initial THINK is deliberately absent.
    gate_mask: Tensor
    emit_mask: Tensor
    old_gate_logprobs: Tensor
    old_token_logprobs: Tensor
    # (batch, stream, dim): factors of the old Gaussian-vector log-probability
    # decided at each gate position. Replay sums them into one action ratio.
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


def half_forced_group_members(
    groups: int, members_per_group: int, device: torch.device
) -> Tensor:
    """Exactly half of every prompt group's contiguous members, no RNG.

    Alternating logical member indices avoids confounding the intervention
    with contiguous RNG draws or replay microbatch boundaries while
    guaranteeing a same-prompt unforced control population in every group.
    """
    if groups < 1:
        raise ValueError("groups must be positive")
    if members_per_group < 2 or members_per_group % 2:
        raise ValueError("members_per_group must be a positive even number")
    pattern = torch.arange(members_per_group, device=device).remainder(2) == 0
    return pattern.repeat(groups)


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
    force_initial_think: bool | Tensor = False,
    tensor_positions: bool = False,
    replay_storage: bool = True,
    record_likelihoods: bool = True,
    compact_finished: bool = True,
    finished_batch_size: int | None = None,
    cache_dtype: torch.dtype | None = None,
) -> LatentRolloutBatch:
    """Roll the gate-conditioned stream forward from a (batch, P) prompt.

    ``force_initial_think`` selects rows whose first action after the prompt
    is a mandatory latent THINK. Other rows sample the ordinary Bernoulli at
    that same boundary. ``max_stream_steps`` is the total generated-slot
    budget per row (forced/optional thoughts and emits);
    ``max_new_tokens`` caps emitted tokens within it.  Optional thinking is
    never forcibly interrupted — a row that spends its whole budget thinking
    just emits fewer tokens.  With ``stop_ids`` set, a row finishes
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
    padding afterwards. It is mutually exclusive with preallocated static
    ``caches``; the ordinary narrow-cache step may be eager or dynamically
    compiled.

    Evaluation sets ``replay_storage=False`` because it never replays PPO
    actions. Its full-stream thought tensor then has a zero-width final
    dimension; old thought likelihood storage is always allocated lazily by
    refresh after trimming. This avoids multiple GiB of dead storage at large
    batches while the current sampled thought is still fed into the next step.

    ``record_likelihoods=False`` skips rollout-time gate/token likelihoods
    when the caller will immediately recompute them through parallel replay.
    ``compact_finished`` removes completed rows and their KV cache entries at
    the existing 16-position synchronization points once at least 25% of the
    current rows have finished. With ``finished_batch_size``, compaction waits
    until the live count fits that one fixed tail size and retains ended filler
    rows to fill it. Compiled evaluation uses this to expose exactly one
    bounded tail specialization instead of arbitrary survivor shapes. Original
    row indices remain attached to all trajectory records, so compaction changes
    compute and RNG consumption, not the sampled policy distribution or output
    attribution.
    """
    if prompt_ids.dim() != 2 or prompt_ids.size(1) < 1:
        raise ValueError("prompt_ids must be (batch, length>=1)")
    if max_stream_steps < max_new_tokens:
        raise ValueError("max_stream_steps must be at least max_new_tokens")
    device = prompt_ids.device
    batch, prompt_length = prompt_ids.shape
    if finished_batch_size is not None and finished_batch_size < 1:
        raise ValueError("finished_batch_size must be positive")
    if isinstance(force_initial_think, bool):
        force_initial_think = torch.full(
            (batch,), force_initial_think, dtype=torch.bool, device=device
        )
    else:
        if force_initial_think.shape != (batch,):
            raise ValueError("force_initial_think must have one value per row")
        force_initial_think = force_initial_think.to(device=device, dtype=torch.bool)
    if bool(force_initial_think.any()) and max_stream_steps < max_new_tokens + 1:
        raise ValueError(
            "max_stream_steps must fit a forced initial thought plus "
            "max_new_tokens"
        )
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
    preallocated_caches = caches is not None
    if caches is None:
        caches = wrapper.make_generation_cache(
            batch, max_stream, device, dtype=cache_dtype
        )
        position_index = (
            torch.zeros((), dtype=torch.long, device=device)
            if tensor_positions
            else None
        )
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
            step_position_value: int | Tensor = position_index
            if key_masks is not None:
                return step_position_value, key_masks[position]
        else:
            step_position_value = position
        if valid_slots is None:
            return step_position_value, None
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
        return step_position_value, mask

    kind = torch.full((batch, max_stream), PAD_SLOT, dtype=torch.long, device=device)
    token_ids = torch.zeros((batch, max_stream), dtype=torch.long, device=device)
    stored_thought_dim = model_dim if replay_storage else 0
    thoughts = torch.zeros(
        (batch, max_stream, stored_thought_dim),
        dtype=torch.float32,
        device=device,
    )
    gate_actions = torch.zeros((batch, max_stream), dtype=torch.long, device=device)
    action_mask = torch.zeros((batch, max_stream), dtype=torch.float32, device=device)
    gate_mask = torch.zeros_like(action_mask)
    emit_mask = torch.zeros_like(action_mask)
    old_gate_logprobs = torch.zeros_like(action_mask)
    old_token_logprobs = torch.zeros_like(action_mask)
    # Both stay zero through the rollout; refresh_old_statistics fills them
    # (values from the separate critic, per-dim thought log-probs through
    # the exact replay path) before anything consumes them.
    old_thought_logprobs = thoughts.new_zeros(
        (batch, max_stream, 0)
    )
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
    live_rows = torch.arange(batch, device=device)
    position = prompt_length - 1
    # ``bool(any())`` is a device-to-host sync that serializes this
    # launch-bound loop (the CPU cannot run ahead of the GPU), so the finish
    # check runs only every SYNC_EVERY steps.  The extra <=SYNC_EVERY-1
    # steps after all rows finish are no-ops: ``record`` is all-False, and
    # finished rows already keep stepping on token 0 by design.
    SYNC_EVERY = 16
    first_position = position
    while position < max_stream - 1:
        initial = position == first_position
        # A forced prefix remains a real action even in the max_new_tokens=0
        # diagnostic case; unforced rows have nothing to do in that case.
        active = ~ended & (
            (emitted < max_new_tokens)
            | (force_initial_think if initial else False)
        )
        if (position - first_position) % SYNC_EVERY == 0:
            active_count = int(active.sum())
            if active_count == 0:
                break
            current_count = active.numel()
            compacted_count = active_count
            if finished_batch_size is not None:
                compacted_count = (
                    finished_batch_size
                    if active_count <= finished_batch_size
                    else current_count
                )
            should_compact = (
                compact_finished
                and not preallocated_caches
                and compacted_count < current_count
                and (
                    finished_batch_size is not None
                    or current_count - active_count
                    >= max(1, current_count // 4)
                )
            )
            if should_compact:
                keep = active.nonzero().squeeze(-1)
                if compacted_count > active_count:
                    fillers = (~active).nonzero().squeeze(-1)[
                        : compacted_count - active_count
                    ]
                    keep = torch.cat((keep, fillers))
                live_rows = live_rows.index_select(0, keep)
                emitted = emitted.index_select(0, keep)
                ended = ended.index_select(0, keep)
                force_initial_think = force_initial_think.index_select(0, keep)
                if valid_slots is not None:
                    valid_slots = valid_slots.index_select(0, keep)
                    pad_lengths = pad_lengths.index_select(0, keep)
                live_prefix = position + 1
                for layer, cache in enumerate(caches):
                    compacted = []
                    for tensor in cache:
                        target = torch.empty(
                            (compacted_count, *tensor.shape[1:]),
                            dtype=tensor.dtype,
                            device=tensor.device,
                        )
                        target[:, :, :live_prefix].copy_(
                            tensor[:, :, :live_prefix].index_select(0, keep)
                        )
                        compacted.append(target)
                    # Replace one layer at a time so old+new full-capacity
                    # caches do not coexist across all layers at peak memory.
                    caches[layer] = tuple(compacted)
                output = output.__class__(
                    belief=output.belief.index_select(0, keep),
                    predicted=output.predicted.index_select(0, keep),
                    thought_log_sigma=output.thought_log_sigma.index_select(0, keep),
                    input_latent=output.input_latent.index_select(0, keep),
                    logits=output.logits.index_select(0, keep),
                    caches=caches,
                )
                active = active.index_select(0, keep)
        belief = output.belief
        if record_likelihoods:
            action, gate_logprob = wrapper.gate.sample(
                belief, generator=generator
            )
        else:
            action = wrapper.gate.sample_action(belief, generator=generator)
            gate_logprob = None
        if initial:
            action = action.masked_fill(force_initial_think, THINK)

        token = top_p_sample(output.logits, temperature, top_p)
        token_logprob = None
        if record_likelihoods:
            token_logprob = (
                output.logits.float()
                .log_softmax(-1)
                .gather(-1, token[:, None])
                .squeeze(-1)
            )
        thought = wrapper.transition.sample_latent(
            output.predicted,
            output.thought_log_sigma,
            generator=generator,
        )

        record = active
        record_rows = live_rows[record]
        action_mask[record_rows, position] = 1.0
        sampled_gate = record & (~force_initial_think if initial else True)
        sampled_gate_rows = live_rows[sampled_gate]
        gate_mask[sampled_gate_rows, position] = 1.0
        gate_actions[record_rows, position] = action[record]
        if gate_logprob is not None:
            old_gate_logprobs[sampled_gate_rows, position] = gate_logprob[
                sampled_gate
            ].float()
        emits = record & (action == EMIT)
        thinks = record & (action == THINK)
        emit_rows = live_rows[emits]
        think_rows = live_rows[thinks]
        if token_logprob is not None:
            old_token_logprobs[emit_rows, position] = token_logprob[emits]

        next_position = position + 1
        kind[emit_rows, next_position] = TOKEN_SLOT
        token_ids[emit_rows, next_position] = token[emits]
        kind[think_rows, next_position] = THOUGHT_SLOT
        if replay_storage:
            thoughts[think_rows, next_position] = thought[thinks]
        emitted += emits.long()
        if stop_tensor is not None:
            ended |= emits & torch.isin(token, stop_tensor)

        # Finished rows keep stepping on token 0 (their next slot stays PAD,
        # so the zero-initialized token_ids row feeds the embedding; every
        # loss masks it); batched caches make per-row early exit impractical.
        next_kinds = kind[live_rows, next_position]
        next_token_ids = token_ids[live_rows, next_position]
        next_input = torch.where(
            (next_kinds == THOUGHT_SLOT)[:, None, None],
            wrapper.thought_input(thought),
            wrapper.embed_tokens(next_token_ids[:, None]),
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
        gate_mask=gate_mask,
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
        group_kind = batch.kind[rows, pad:]
        used = int((group_kind != PAD_SLOT).any(0).nonzero().max()) + 1
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
                # Slice to this group's ACTUAL used tail before cloning. A
                # full-capacity clone retained ~8 GiB across sixteen groups
                # in the 5K-context run even though typical streams used
                # only ~140 positions.
                value = value[rows, pad : pad + used].clone()
            elif isinstance(value, Tensor) and value.dim() == 1:
                value = value[rows].clone()
            sliced[field.name] = value
        groups.append(LatentRolloutBatch(**sliced))
    return groups


def trim_stream(batch: LatentRolloutBatch, multiple: int = 1) -> LatentRolloutBatch:
    """Drop all-PAD tail columns into owning storage for compact replay.

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
                if used < batch.stream_length:
                    # A narrow view would retain the full worst-case backing
                    # allocation and defeats the memory purpose of trimming.
                    value = value[:, :used].clone()
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
    # Bulk device transfers avoid a GPU synchronization for every trajectory.
    # Decoding/verifying is CPU work anyway, and this helper is called once
    # per rollout by both training and evaluation.
    generated = (
        batch.kind[:, batch.prompt_length:] == TOKEN_SLOT
    ).to(device="cpu")
    tokens = batch.token_ids[:, batch.prompt_length:].to(device="cpu")
    return [
        tokens[row][generated[row]].tolist()
        for row in range(batch.kind.size(0))
    ]


def emitted_token_and_kind_rows(
    batch: LatentRolloutBatch,
) -> tuple[list[list[int]], list[list[int]]]:
    """Bulk-copy continuation tokens and kinds for human-readable capture.

    Evaluation needs both arrays to reconstruct latent-action traces. Packing
    them before the transfer produces one synchronization and avoids first
    copying a token mask and then synchronizing again for selected kind rows.
    The ordinary scorer keeps using :func:`emitted_token_rows`, whose compact
    boolean mask transfers less data when no traces are requested.
    """
    packed = torch.stack(
        (
            batch.kind[:, batch.prompt_length :],
            batch.token_ids[:, batch.prompt_length :],
        )
    ).to(device="cpu")
    kinds, tokens = packed[0], packed[1]
    emitted = [
        tokens[row][kinds[row] == TOKEN_SLOT].tolist()
        for row in range(batch.kind.size(0))
    ]
    return emitted, kinds.tolist()


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
    think_mask = batch.kind == THOUGHT_SLOT
    # The kind-select makes dense adapter evaluation value/gradient-equivalent
    # to boolean-index assignment. Unlike the latter, it has static output
    # shapes and keeps the full replay trunk inside one Inductor graph.
    thought_latent = wrapper.adapter(batch.thoughts).to(token_latent.dtype)
    inputs = torch.where(think_mask[..., None], thought_latent, token_latent)
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

    Returns (beliefs, predicted, stream_inputs, token_targets).  Renderer
    features are deliberately formed only at EMIT positions by consumers;
    its four-layer wide probe and vocabulary projection are positionwise, so
    dense prompt/thought/pad evaluation was pure waste. Both
    ``refresh_old_statistics`` and the trainer's update step go through this
    single code path; that is what makes the recomputed "old" statistics
    exact — epoch-0 PPO ratios are one by construction.
    """
    stream_inputs, beliefs = replay_beliefs(wrapper, batch)
    predicted = wrapper.thought_mean(beliefs)
    token_targets = torch.zeros_like(batch.token_ids)
    token_targets[:, :-1] = batch.token_ids[:, 1:]
    return beliefs, predicted, stream_inputs, token_targets


def select_thought_actions(
    batch: LatentRolloutBatch, predicted: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Select fresh-head means and sampled actions at actual THINK positions.

    The mean head has already run densely. Compacting only its consumers
    ensures EMIT/prompt/pad outputs have no gradient edge and prevents unused
    PPO ratios from overflowing before a zero mask is applied.
    """
    think_mask = (batch.gate_actions == THINK) & batch.action_mask.bool()
    thought_targets = torch.zeros_like(batch.thoughts)
    thought_targets[:, :-1] = batch.thoughts[:, 1:]
    return predicted[think_mask], thought_targets[think_mask], think_mask


def trajectory_used_lengths(batch: LatentRolloutBatch) -> Tensor:
    """Last non-padding column plus one for every independent trajectory."""
    positions = torch.arange(
        1, batch.stream_length + 1, device=batch.kind.device
    )
    return ((batch.kind != PAD_SLOT) * positions).amax(1)


def select_trajectory_rows(
    batch: LatentRolloutBatch, rows: Tensor, stream_length: int
) -> LatentRolloutBatch:
    """Materialize selected rows at a compact, shared stream length."""
    if stream_length < batch.prompt_length or stream_length > batch.stream_length:
        raise ValueError("invalid replay stream length")
    selected = {}
    for field in fields(batch):
        value = getattr(batch, field.name)
        if (
            isinstance(value, Tensor)
            and value.dim() >= 2
            and value.size(1) == batch.stream_length
        ):
            value = value[rows, :stream_length]
        elif isinstance(value, Tensor) and value.dim() >= 1:
            value = value[rows]
        selected[field.name] = value
    return LatentRolloutBatch(**selected)


def iter_length_aware_microbatches(
    batch: LatentRolloutBatch,
    max_trajectories: int,
    attention_budget: int,
    bucket_multiple: int = 1,
):
    """Yield stable length-sorted replay shards under a B*L^2 budget.

    Causal full-stream replay is governed by attention area, not row count.
    Stable sorting groups similarly sized independent trajectories; normal
    ~150-position groups fit all 32 rows, while rare 1K-4K outliers
    automatically receive smaller shards. ``rows`` maps each compact shard
    back into the parent batch for refresh-stat writes.
    """
    if max_trajectories < 1:
        raise ValueError("replay max trajectories must be positive")
    if attention_budget < 1:
        raise ValueError("replay attention budget must be positive")
    if bucket_multiple < 1:
        raise ValueError("replay bucket multiple must be positive")
    lengths = trajectory_used_lengths(batch).to(device="cpu").tolist()
    order = sorted(range(len(lengths)), key=lambda row: (-lengths[row], row))

    shard: list[int] = []
    shard_length = 0

    def bucketed(length: int) -> int:
        return min(
            batch.stream_length,
            -(-length // bucket_multiple) * bucket_multiple,
        )

    for row in order:
        candidate_length = max(shard_length, bucketed(lengths[row]))
        candidate_rows = len(shard) + 1
        exceeds = (
            candidate_rows > max_trajectories
            or candidate_rows * candidate_length * candidate_length
            > attention_budget
        )
        if shard and exceeds:
            row_tensor = torch.tensor(
                shard, dtype=torch.long, device=batch.kind.device
            )
            yield (
                select_trajectory_rows(batch, row_tensor, shard_length),
                row_tensor,
                shard_length,
            )
            shard = []
            shard_length = 0
        shard.append(row)
        shard_length = max(shard_length, bucketed(lengths[row]))
    if shard:
        row_tensor = torch.tensor(
            shard, dtype=torch.long, device=batch.kind.device
        )
        yield (
            select_trajectory_rows(batch, row_tensor, shard_length),
            row_tensor,
            shard_length,
        )


def refresh_old_statistics(
    wrapper: LatentThoughtModel,
    critic,
    batch: LatentRolloutBatch,
    max_trajectories: int = 32,
    attention_budget: int = 4 * 1024 * 1024,
    bucket_multiple: int = 1,
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
    Replay is stably length-sorted and split only across independent
    trajectories under a B*L^2 budget, bounding quadratic attention memory
    without changing any consumed statistic. The update path uses the same
    deterministic planner so compiled refresh/update forwards remain
    numerically identical at epoch zero.
    """
    backbone = wrapper.backbone
    if batch.old_thought_logprobs.shape[-1] == 0:
        batch.old_thought_logprobs = torch.zeros_like(batch.thoughts)
    for microbatch, rows, stream_length in iter_length_aware_microbatches(
        batch, max_trajectories, attention_budget, bucket_multiple
    ):
        beliefs, predicted, stream_inputs, token_targets = replay_head_inputs(
            wrapper, microbatch
        )
        values = critic.values(microbatch).float()
        gate_logprobs = (
            wrapper.gate.log_prob(
                microbatch.gate_actions.float(), beliefs
            ).float()
            * microbatch.gate_mask
        )
        emit_mask = microbatch.emit_mask.bool()
        emit_features = wrapper.renderer_features(
            stream_inputs[emit_mask], beliefs[emit_mask]
        )
        emit_logits = backbone.logits_from_features(emit_features)
        compact_token_logprobs = (
            emit_logits.float()
            .log_softmax(-1)
            .gather(-1, token_targets[emit_mask][..., None])
            .squeeze(-1)
        )
        token_logprobs = torch.zeros_like(microbatch.old_token_logprobs)
        token_logprobs[emit_mask] = compact_token_logprobs
        # The thought decided at gate position p is stored at p+1 — the same
        # shift as token targets — so per-dim log-probs align with think_mask.
        thought_means, thought_targets, think_mask = select_thought_actions(
            microbatch, predicted
        )
        compact_thought_logprobs = wrapper.transition.per_dim_log_prob(
            thought_targets,
            thought_means,
            wrapper.transition.predict_log_sigma(beliefs[think_mask]),
        ).float()
        thought_logprobs = torch.zeros_like(microbatch.old_thought_logprobs)
        thought_logprobs[think_mask] = compact_thought_logprobs
        with torch.no_grad():
            # Advanced row indexing materializes a copy, so assignment must
            # target the parent explicitly (``view.copy_`` would update only
            # the temporary).
            batch.old_values[rows, :stream_length] = values
            batch.old_gate_logprobs[rows, :stream_length] = gate_logprobs
            batch.old_token_logprobs[rows, :stream_length] = token_logprobs
            batch.old_thought_logprobs[rows, :stream_length] = thought_logprobs
